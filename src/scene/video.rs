//! A video used as a layer's texture.
//!
//! A `.tex` with flag 32 (`TextureFlags_Video` in linux-wallpaperengine) holds
//! an mp4 in place of pixels: its single "mipmap" is the whole file. Scenes use
//! it for a moving background or an overlay, and it plays on a loop against the
//! scene clock. libav reads it straight from memory, and a `scale,format=rgba`
//! filter graph turns each frame into the layer's pixels at the layer's size,
//! so nothing larger than the layer is ever copied off the decoder.

use crate::export::ffmpeg as ff;
use anyhow::{Context, Result, bail};
use ffmpeg::{Packet, filter, format, frame, media::Type};
use ffmpeg_next as ffmpeg;
use image::RgbaImage;
use std::io::Cursor;

/// Closer than this, two timestamps are the same frame.
const TOLERANCE: f64 = 1e-4;

/// Further ahead than this, seek rather than decode every frame in between: a layer left undrawn while
/// hidden comes back this far behind.
const SEEK_AHEAD: f64 = 1.0;

pub struct VideoTexture {
    input: format::context::Input,
    stream: usize,
    decoder: ffmpeg::decoder::Video,
    graph: filter::Graph,
    time_base: ffmpeg::Rational,
    size: (u32, u32),
    /// Seconds per loop: the container's own figure, or learnt at the first end of stream.
    duration: Option<f64>,
    /// Timestamp of the frame last handed out.
    shown: Option<f64>,
    /// A frame decoded past the last request, kept for the next one.
    ahead: Option<(f64, frame::Video)>,
    /// The demuxer has run out and the decoder has been told so.
    drained: bool,
}

/// Open an embedded video whose frames come out at `size`.
pub fn open(bytes: Vec<u8>, size: (u32, u32)) -> Result<VideoTexture> {
    ff::init()?;
    let io = format::context::StreamIo::from_read_seek(Cursor::new(bytes)).context("wrapping the embedded video")?;
    let input = format::input_from_stream(io, Some("mp4"), None).context("opening the embedded video")?;
    let stream = input.streams().best(Type::Video).context("the embedded video has no video stream")?;
    let (index, time_base) = (stream.index(), stream.time_base());
    let decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .and_then(|mut context| {
            // libavcodec decodes on one thread unless told otherwise: ~26 ms a 4K frame (3326873240).
            context.set_threading(ffmpeg::threading::Config::kind(ffmpeg::threading::Type::Frame));
            context.decoder().video()
        })
        .context("opening the embedded video's decoder")?;
    // The container's duration is in AV_TIME_BASE units; zero or negative means it did not say.
    #[expect(clippy::cast_precision_loss, reason = "a wallpaper loop's length in microseconds, far below 2^52")]
    let duration = (input.duration() > 0).then(|| input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE));
    let graph = rgba_graph(&decoder, time_base, size)?;
    Ok(VideoTexture {
        input,
        stream: index,
        decoder,
        graph,
        time_base,
        size,
        duration,
        shown: None,
        ahead: None,
        drained: false,
    })
}

/// The frame showing at `time` seconds into the loop, or `None` when it is the
/// one already handed out (or the video has no frame that early).
pub fn frame_at(video: &mut VideoTexture, time: f32) -> Result<Option<RgbaImage>> {
    let time = match video.duration {
        Some(length) if length > 0.0 => f64::from(time).rem_euclid(length),
        _ => f64::from(time),
    };
    if video.shown.is_some_and(|shown| time + TOLERANCE < shown) {
        rewind(video)?;
    }
    let decoded_to = video.ahead.as_ref().map(|(pts, _)| *pts).or(video.shown);
    if decoded_to.map_or(time > SEEK_AHEAD, |at| time - at > SEEK_AHEAD) {
        seek(video, time)?;
    }

    let mut latest = None;
    loop {
        let next = match video.ahead.take() {
            Some(frame) => Some(frame),
            None => decode_next(video)?,
        };
        match next {
            Some((pts, frame)) if pts <= time + TOLERANCE => latest = Some((pts, frame)),
            Some(later) => {
                video.ahead = Some(later);
                break;
            }
            None => {
                // Past the last frame: that frame holds until the loop comes round.
                if video.duration.is_none() {
                    video.duration = latest.as_ref().map(|(pts, _)| pts + frame_period(video)).or(video.shown);
                }
                break;
            }
        }
    }

    let Some((pts, frame)) = latest else { return Ok(None) };
    if video.shown.is_some_and(|shown| (shown - pts).abs() < TOLERANCE) {
        return Ok(None);
    }
    video.shown = Some(pts);
    to_rgba(video, &frame).map(Some)
}

/// Back to the first frame, for the next pass round the loop.
fn rewind(video: &mut VideoTexture) -> Result<()> {
    video.input.seek(0, ..0).context("rewinding the embedded video")?;
    video.decoder.flush();
    video.shown = None;
    video.ahead = None;
    video.drained = false;
    Ok(())
}

/// To the keyframe at or before `time`, from which decoding forward reaches it.
fn seek(video: &mut VideoTexture, time: f64) -> Result<()> {
    #[expect(clippy::cast_possible_truncation, reason = "a loop position in microseconds, far below i64::MAX")]
    let target = (time * f64::from(ffmpeg::ffi::AV_TIME_BASE)) as i64;
    video.input.seek(target, ..target).context("seeking the embedded video")?;
    video.decoder.flush();
    video.shown = None;
    video.ahead = None;
    video.drained = false;
    Ok(())
}

/// The next decoded frame and its timestamp, or `None` at the end of the stream.
fn decode_next(video: &mut VideoTexture) -> Result<Option<(f64, frame::Video)>> {
    let mut decoded = frame::Video::empty();
    loop {
        if video.decoder.receive_frame(&mut decoded).is_ok() {
            #[expect(clippy::cast_precision_loss, reason = "frame counts at wallpaper lengths, far below 2^52")]
            let pts = decoded.timestamp().or_else(|| decoded.pts()).map_or(0.0, |pts| pts as f64 * f64::from(video.time_base));
            return Ok(Some((pts, decoded)));
        }
        if video.drained {
            return Ok(None);
        }
        let mut packet = Packet::empty();
        match packet.read(&mut video.input) {
            Ok(()) if packet.stream() == video.stream => {
                video.decoder.send_packet(&packet).context("decoding the embedded video")?;
            }
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) => {
                video.decoder.send_eof().context("flushing the embedded video's decoder")?;
                video.drained = true;
            }
            Err(error) => return Err(error).context("reading the embedded video"),
        }
    }
}

/// One frame's length, from the stream's declared rate or a 30 fps guess.
fn frame_period(video: &VideoTexture) -> f64 {
    video
        .input
        .stream(video.stream)
        .and_then(|stream| ff::rate_to_fps(stream.avg_frame_rate()))
        .map_or(1.0 / 30.0, |fps| 1.0 / fps)
}

fn rgba_graph(decoder: &ffmpeg::decoder::Video, time_base: ffmpeg::Rational, (width, height): (u32, u32)) -> Result<filter::Graph> {
    let args = format!(
        "video_size={}x{}:pix_fmt={}:time_base={time_base}:pixel_aspect=1/1",
        decoder.width(),
        decoder.height(),
        decoder.format().descriptor().map_or("yuv420p", |descriptor| descriptor.name()),
    );
    let mut graph = filter::Graph::new();
    graph
        .add(&filter::find("buffer").context("libavfilter has no `buffer`")?, "in", &args)
        .context("adding the video source")?;
    graph
        .add(&filter::find("buffersink").context("libavfilter has no `buffersink`")?, "out", "")
        .context("adding the video sink")?;
    // Bilinear, for the same reason `compose::scaled_tinted` is not Lanczos.
    graph
        .output("in", 0)
        .and_then(|parser| parser.input("out", 0))
        .and_then(|parser| parser.parse(&format!("scale={width}:{height}:flags=bilinear,format=rgba")))
        .context("describing the video filter")?;
    graph.validate().context("configuring the video filter")?;
    Ok(graph)
}

fn to_rgba(video: &mut VideoTexture, decoded: &frame::Video) -> Result<RgbaImage> {
    video.graph.get("in").context("the video filter has no source")?.source().add(decoded).context("filtering a video frame")?;
    let mut rgba = frame::Video::empty();
    video.graph.get("out").context("the video filter has no sink")?.sink().frame(&mut rgba).context("the video filter produced no frame")?;

    let (width, height) = video.size;
    if (rgba.width(), rgba.height()) != (width, height) {
        bail!("the video filter produced {}x{}, not {width}x{height}", rgba.width(), rgba.height());
    }
    // libav pads each row to its own alignment, so copy the rows out without it.
    let row = width as usize * 4;
    let stride = rgba.stride(0);
    let plane = rgba.data(0);
    let mut pixels = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        pixels.extend_from_slice(plane.get(y * stride..y * stride + row).context("a short video frame row")?);
    }
    RgbaImage::from_raw(width, height, pixels).context("the video frame did not fill its buffer")
}

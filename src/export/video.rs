//! Looping video output.
//!
//! Video wallpapers already contain a finished, seamlessly looping file, so
//! the job here is packaging rather than rendering: hand the user an mp4 that
//! an ordinary wallpaper app will accept, without needlessly degrading it.

use super::{Options, ffmpeg as ff, needs_reencode};
use anyhow::{Context, Result};
use ffmpeg::format::Pixel;
use ffmpeg::{Dictionary, Packet, Rational, codec, encoder, filter, format, media, picture};
use ffmpeg_next as ffmpeg;
use std::{fmt::Write as _, path::Path};

/// Container-level compatibility. Anything else gets re-encoded.
///
/// H.264 in 4:2:0 is the format every wallpaper app, phone and TV decodes.
/// A source already in it can be copied through untouched.
fn is_broadly_compatible(info: &ff::MediaInfo) -> bool {
    let codec_ok = info.codec == "h264";
    // yuv420p and its `yuvj`/`10le` relatives are distinct: only plain 8-bit
    // 4:2:0 is universally safe.
    let pixels_ok = matches!(info.pixel_format.as_str(), "yuv420p" | "yuvj420p");
    codec_ok && pixels_ok
}

/// Everything the re-encode path carries from frame to frame.
struct Encode {
    decoder: ffmpeg::decoder::Video,
    encoder: encoder::Video,
    graph: filter::Graph,
    /// The encoder's own time base, which packets are rescaled out of.
    time_base: Rational,
}

/// Write `source` to `out` as a wallpaper-app-friendly mp4.
///
/// Returns whether the video stream was copied rather than re-encoded.
/// The stream layout and timing this export settled on, once the output's
/// header is written — nothing about it changes for the rest of the export.
struct Streams {
    video_index: usize,
    audio_index: Option<usize>,
    video_time_base: Rational,
    audio_time_base: Option<Rational>,
    video_out_time_base: Rational,
    audio_out_time_base: Option<Rational>,
}

/// Open the input, declare the output's streams, and write its header.
///
/// Audio is copied, never transcoded: a wallpaper's audio is already in a
/// codec mp4 accepts, and re-encoding it would only lose quality.
fn open_streams(
    source: &Path,
    out: &Path,
    info: &ff::MediaInfo,
    options: &Options,
    copy: bool,
) -> Result<(format::context::Input, format::context::Output, Streams, Option<Encode>)> {
    ff::init()?;
    let ictx = format::input(source)
        .with_context(|| format!("opening {}", source.display()))?;
    let mut octx = format::output(out)
        .with_context(|| format!("creating {}", out.display()))?;

    let video_index = ictx
        .streams()
        .best(media::Type::Video)
        .with_context(|| format!("{} has no video stream", source.display()))?
        .index();
    let audio_index = if options.audio && info.has_audio {
        ictx.streams().best(media::Type::Audio).map(|stream| stream.index())
    } else {
        None
    };

    let video_stream = ictx
        .stream(video_index)
        .expect("the index came from this context");
    let video_time_base = video_stream.time_base();

    let encode = if copy {
        add_copied_stream(&mut octx, &video_stream)?;
        None
    } else {
        Some(setup_encode(&video_stream, &mut octx, options)?)
    };

    let audio_time_base = match audio_index {
        Some(index) => {
            let stream = ictx
                .stream(index)
                .expect("the index came from this context");
            let time_base = stream.time_base();
            add_copied_stream(&mut octx, &stream)?;
            Some(time_base)
        }
        None => None,
    };

    octx.set_metadata(ictx.metadata().to_owned());

    // Puts the index at the front so a player can start without reading the
    // whole file — what makes a wallpaper app show the video immediately.
    let mut muxer_options = Dictionary::new();
    muxer_options.set("movflags", "+faststart");
    octx.write_header_with(muxer_options)
        .with_context(|| format!("writing the mp4 header of {}", out.display()))?;

    let video_out_time_base = octx
        .stream(0)
        .expect("the video stream was added first")
        .time_base();
    let audio_out_time_base = octx.stream(1).map(|stream| stream.time_base());

    Ok((
        ictx,
        octx,
        Streams {
            video_index,
            audio_index,
            video_time_base,
            audio_time_base,
            video_out_time_base,
            audio_out_time_base,
        },
        encode,
    ))
}

/// Walk the input's packets, copying or decoding+encoding each into the
/// output, stopping once both streams have passed the requested duration.
///
/// Trimming is a container operation, so it does not force a re-encode; on
/// the copy path the cut lands on whole packets, which is fine for a loop.
fn copy_packets(
    ictx: &mut format::context::Input,
    octx: &mut format::context::Output,
    streams: &Streams,
    encode: &mut Option<Encode>,
    limit: Option<f64>,
) -> Result<()> {
    let mut video_done = false;
    let mut audio_done = streams.audio_index.is_none();

    for (stream, mut packet) in ictx.packets() {
        let index = stream.index();
        let is_video = index == streams.video_index;
        if !is_video && Some(index) != streams.audio_index {
            continue;
        }

        let time_base = if is_video {
            streams.video_time_base
        } else {
            streams
                .audio_time_base
                .expect("an audio stream implies its time base")
        };
        if past_limit(&packet, time_base, limit) {
            if is_video {
                video_done = true;
            } else {
                audio_done = true;
            }
            if video_done && audio_done {
                break;
            }
            continue;
        }

        match (is_video, encode.as_mut()) {
            (true, Some(encode)) => {
                encode
                    .decoder
                    .send_packet(&packet)
                    .context("decoding the source video")?;
                drain_decoder(encode, octx, streams.video_out_time_base)?;
            }
            (true, None) => {
                remux(&mut packet, 0, time_base, streams.video_out_time_base, octx)?;
            }
            (false, _) => {
                let out_time_base = streams
                    .audio_out_time_base
                    .expect("an audio stream implies an output stream");
                remux(&mut packet, 1, time_base, out_time_base, octx)?;
            }
        }
    }
    Ok(())
}

/// Push the decoder, filter graph and encoder to end-of-stream in turn.
fn flush_encode(
    encode: &mut Encode,
    octx: &mut format::context::Output,
    video_out_time_base: Rational,
) -> Result<()> {
    encode
        .decoder
        .send_eof()
        .context("flushing the video decoder")?;
    drain_decoder(encode, octx, video_out_time_base)?;

    encode
        .graph
        .get("in")
        .expect("the graph has a buffer source")
        .source()
        .flush()
        .context("flushing the filter graph")?;
    drain_graph(encode, octx, video_out_time_base)?;

    encode
        .encoder
        .send_eof()
        .context("flushing the video encoder")?;
    drain_encoder(encode, octx, video_out_time_base)
}

pub fn export(
    source: &Path,
    out: &Path,
    info: &ff::MediaInfo,
    options: &Options,
) -> Result<bool> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let copy = !needs_reencode(options) && is_broadly_compatible(info);
    let (mut ictx, mut octx, streams, mut encode) =
        open_streams(source, out, info, options, copy)?;

    copy_packets(&mut ictx, &mut octx, &streams, &mut encode, options.duration)?;

    if let Some(encode) = encode.as_mut() {
        flush_encode(encode, &mut octx, streams.video_out_time_base)?;
    }

    octx.write_trailer()
        .with_context(|| format!("finishing {}", out.display()))?;
    Ok(copy)
}

/// True once a packet starts beyond the requested output length.
fn past_limit(packet: &Packet, time_base: Rational, limit: Option<f64>) -> bool {
    let Some(limit) = limit else {
        return false;
    };
    let Some(pts) = packet.pts().or_else(|| packet.dts()) else {
        return false;
    };
    // Packet timestamps are frame counts at typical wallpaper lengths and
    // rates, nowhere near f64's ~2^52 exact-integer range.
    #[expect(clippy::cast_precision_loss, reason = "frame counts, nowhere near 2^52")]
    let seconds = pts as f64 * f64::from(time_base);
    seconds > limit
}

/// Declare an output stream that reuses the input stream's encoded form.
fn add_copied_stream(
    octx: &mut format::context::Output,
    ist: &format::stream::Stream,
) -> Result<()> {
    let mut ost = octx
        .add_stream(encoder::find(codec::Id::None))
        .context("adding an output stream")?;
    ost.set_parameters(ist.parameters());
    // mp4 rejects the codec tags some source containers use. There is no safe
    // API for clearing it yet.
    unsafe {
        (*ost.parameters().as_mut_ptr()).codec_tag = 0;
    }
    Ok(())
}

/// Copy one packet through to the output, rebased onto its time base.
fn remux(
    packet: &mut Packet,
    ost_index: usize,
    from: Rational,
    to: Rational,
    octx: &mut format::context::Output,
) -> Result<()> {
    packet.rescale_ts(from, to);
    packet.set_position(-1);
    packet.set_stream(ost_index);
    packet
        .write_interleaved(octx)
        .context("writing a copied packet")?;
    Ok(())
}

/// Build the decoder, filter graph and H.264 encoder for the re-encode path.
fn setup_encode(
    ist: &format::stream::Stream,
    octx: &mut format::context::Output,
    options: &Options,
) -> Result<Encode> {
    let decoder = codec::context::Context::from_parameters(ist.parameters())
        .and_then(|context| context.decoder().video())
        .context("opening the video decoder")?;

    let mut graph = build_graph(&decoder, ist, options)?;
    // The graph decides the output cadence, so its time base — not the
    // source's — is the one the encoder has to speak.
    let time_base = graph
        .get("out")
        .expect("the graph has a buffer sink")
        .sink()
        .time_base();

    let (width, height) = match options.resolution {
        Some(resolution) => (resolution.width, resolution.height),
        None => (decoder.width(), decoder.height()),
    };
    let frame_rate = options
        .fps
        .map(Rational::from)
        .or_else(|| ff::rate_to_fps(ist.avg_frame_rate()).map(Rational::from));

    let global_header = octx
        .format()
        .flags()
        .contains(format::Flags::GLOBAL_HEADER);
    let codec = encoder::find_by_name("libx264")
        .or_else(|| encoder::find(codec::Id::H264))
        .context("this ffmpeg build has no H.264 encoder")?;

    let mut encoder = codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
        .context("opening the H.264 encoder")?;
    encoder.set_width(width);
    encoder.set_height(height);
    encoder.set_format(Pixel::YUV420P);
    encoder.set_aspect_ratio(decoder.aspect_ratio());
    encoder.set_time_base(time_base);
    encoder.set_frame_rate(frame_rate);
    if global_header {
        encoder.set_flags(codec::Flags::GLOBAL_HEADER);
    }

    let mut x264 = Dictionary::new();
    x264.set("preset", "slow");
    x264.set("crf", "18");
    let encoder = encoder
        .open_with(x264)
        .context("configuring the H.264 encoder")?;

    let mut ost = octx.add_stream(codec).context("adding the video stream")?;
    ost.set_parameters(&encoder);

    Ok(Encode {
        decoder,
        encoder,
        graph,
        time_base,
    })
}

/// A `buffer -> [scale][fps]format -> buffersink` graph.
///
/// `format=yuv420p` is unconditional: a source that reached this path may be
/// 10-bit or 4:4:4, which is exactly what the output must not be.
fn build_graph(
    decoder: &ffmpeg::decoder::Video,
    ist: &format::stream::Stream,
    options: &Options,
) -> Result<filter::Graph> {
    let mut args = format!(
        "video_size={}x{}:pix_fmt={}:time_base={}:pixel_aspect={}",
        decoder.width(),
        decoder.height(),
        decoder
            .format()
            .descriptor()
            .map_or("yuv420p", |descriptor| descriptor.name()),
        ist.time_base(),
        decoder.aspect_ratio(),
    );
    // Only pass a rate the source actually declares; `fps` needs one, and
    // 0/0 is rejected outright.
    if ff::rate_to_fps(ist.avg_frame_rate()).is_some() {
        let _ = write!(args, ":frame_rate={}", ist.avg_frame_rate());
    }

    let mut spec = Vec::new();
    if let Some(resolution) = options.resolution {
        spec.push(format!(
            "scale={}:{}:flags=lanczos",
            resolution.width, resolution.height
        ));
    }
    if let Some(fps) = options.fps {
        spec.push(format!("fps={fps}"));
    }
    spec.push("format=yuv420p".to_string());

    let mut graph = filter::Graph::new();
    graph
        .add(
            &filter::find("buffer").context("libavfilter has no `buffer`")?,
            "in",
            &args,
        )
        .context("adding the filter source")?;
    graph
        .add(
            &filter::find("buffersink").context("libavfilter has no `buffersink`")?,
            "out",
            "",
        )
        .context("adding the filter sink")?;
    graph
        .get("out")
        .expect("just added")
        .set_pixel_format(Pixel::YUV420P);
    graph
        .output("in", 0)
        .and_then(|parser| parser.input("out", 0))
        .and_then(|parser| parser.parse(&spec.join(",")))
        .context("building the filter graph")?;
    graph.validate().context("validating the filter graph")?;
    Ok(graph)
}

/// Decoded frames -> filter graph.
fn drain_decoder(
    encode: &mut Encode,
    octx: &mut format::context::Output,
    ost_time_base: Rational,
) -> Result<()> {
    let mut frame = ffmpeg::frame::Video::empty();
    while encode.decoder.receive_frame(&mut frame).is_ok() {
        let timestamp = frame.timestamp();
        frame.set_pts(timestamp);
        frame.set_kind(picture::Type::None);
        encode
            .graph
            .get("in")
            .expect("the graph has a buffer source")
            .source()
            .add(&frame)
            .context("feeding the filter graph")?;
        drain_graph(encode, octx, ost_time_base)?;
    }
    Ok(())
}

/// Filtered frames -> encoder.
fn drain_graph(
    encode: &mut Encode,
    octx: &mut format::context::Output,
    ost_time_base: Rational,
) -> Result<()> {
    let mut frame = ffmpeg::frame::Video::empty();
    while encode
        .graph
        .get("out")
        .expect("the graph has a buffer sink")
        .sink()
        .frame(&mut frame)
        .is_ok()
    {
        encode
            .encoder
            .send_frame(&frame)
            .context("encoding a frame")?;
        drain_encoder(encode, octx, ost_time_base)?;
    }
    Ok(())
}

/// Encoded packets -> muxer.
fn drain_encoder(
    encode: &mut Encode,
    octx: &mut format::context::Output,
    ost_time_base: Rational,
) -> Result<()> {
    let mut packet = Packet::empty();
    while encode.encoder.receive_packet(&mut packet).is_ok() {
        packet.set_stream(0);
        packet.rescale_ts(encode.time_base, ost_time_base);
        packet
            .write_interleaved(octx)
            .context("writing an encoded packet")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(codec: &str, pixel_format: &str) -> ff::MediaInfo {
        ff::MediaInfo {
            width: 1920,
            height: 1080,
            fps: 60.0,
            duration: 10.0,
            codec: codec.to_string(),
            pixel_format: pixel_format.to_string(),
            has_audio: false,
        }
    }

    #[test]
    fn h264_in_8_bit_420_is_compatible() {
        assert!(is_broadly_compatible(&info("h264", "yuv420p")));
        assert!(is_broadly_compatible(&info("h264", "yuvj420p")));
    }

    #[test]
    fn other_codecs_and_deeper_pixel_formats_are_not() {
        // A 10-bit or 4:4:4 H.264 stream plays nowhere reliably, and neither
        // does VP9 in an mp4 — both must be re-encoded despite being "h264"
        // or already in a video container.
        assert!(!is_broadly_compatible(&info("h264", "yuv420p10le")));
        assert!(!is_broadly_compatible(&info("h264", "yuv444p")));
        assert!(!is_broadly_compatible(&info("vp9", "yuv420p")));
        assert!(!is_broadly_compatible(&info("hevc", "yuv420p")));
    }
}

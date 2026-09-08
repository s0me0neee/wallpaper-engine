//! Single-frame PNG output.
//!
//! For now the only source is a video; the scene renderer will add a second
//! entry point here that writes an already-rendered frame buffer.

use super::ffmpeg as ff;
use anyhow::{Context, Result, bail};
use ffmpeg::software::scaling;
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use std::path::Path;

/// Timestamps land on frame boundaries only approximately, so a request for
/// t=5.0 should accept the frame at 4.9999.
const TOLERANCE: f64 = 1e-3;

/// Grab one frame from `video` at `at_seconds` and write it as a PNG.
pub fn from_video(
    video: &Path,
    out: &Path,
    at_seconds: f64,
    width_height: Option<(u32, u32)>,
) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let (mut ictx, index, mut decoder) = ff::open_video(video)?;
    let time_base = ictx
        .stream(index)
        .expect("the index came from this context")
        .time_base();

    // Seek to the keyframe at or before the target so a late timestamp does
    // not mean decoding the whole file: on a 27-second 1440p60 source that is
    // the difference between instant and several seconds.
    if at_seconds > 0.0 {
        // A wallpaper's still timestamp is always a few seconds at most, so
        // this lands nowhere near i64's range; the truncation is deliberate —
        // ffmpeg's seek target is in whole microseconds.
        #[expect(clippy::cast_possible_truncation, reason = "seek target is whole microseconds")]
        let target = (at_seconds * f64::from(ffmpeg::ffi::AV_TIME_BASE)) as i64;
        ictx.seek(target, ..target)
            .with_context(|| format!("seeking to {at_seconds}s in {}", video.display()))?;
    }

    let Some(frame) = decode_frame_at(&mut ictx, index, &mut decoder, at_seconds, time_base)?
    else {
        bail!(
            "no frame at {at_seconds}s in {} — is the timestamp past the end?",
            video.display()
        );
    };

    let (width, height) = width_height.unwrap_or((frame.width(), frame.height()));
    let mut scaler = scaling::Context::get(
        frame.format(),
        frame.width(),
        frame.height(),
        ffmpeg::format::Pixel::RGB24,
        width,
        height,
        scaling::Flags::LANCZOS,
    )
    .context("setting up the frame scaler")?;

    let mut rgb = Video::empty();
    scaler.run(&frame, &mut rgb).context("scaling the frame")?;

    // libav pads each row out to its own alignment, so the plane is only a
    // valid image buffer once the padding is dropped.
    let row = width as usize * 3;
    let stride = rgb.stride(0);
    let mut pixels = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        let start = y * stride;
        pixels.extend_from_slice(&rgb.data(0)[start..start + row]);
    }

    let image = image::RgbImage::from_raw(width, height, pixels)
        .context("the decoded frame did not fill its buffer")?;
    image
        .save(out)
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

/// Decode forward from the current position to the first frame at or after
/// `at_seconds`, or `None` if the file ends first.
fn decode_frame_at(
    ictx: &mut ffmpeg::format::context::Input,
    index: usize,
    decoder: &mut ffmpeg::decoder::Video,
    at_seconds: f64,
    time_base: ffmpeg::Rational,
) -> Result<Option<Video>> {
    let mut frame = Video::empty();

    for (stream, packet) in ictx.packets() {
        if stream.index() != index {
            continue;
        }
        decoder.send_packet(&packet).context("decoding")?;
        while decoder.receive_frame(&mut frame).is_ok() {
            if frame_seconds(&frame, time_base) + TOLERANCE >= at_seconds {
                return Ok(Some(frame));
            }
        }
    }

    decoder.send_eof().context("flushing the decoder")?;
    while decoder.receive_frame(&mut frame).is_ok() {
        if frame_seconds(&frame, time_base) + TOLERANCE >= at_seconds {
            return Ok(Some(frame));
        }
    }
    Ok(None)
}

fn frame_seconds(frame: &Video, time_base: ffmpeg::Rational) -> f64 {
    // Timestamps are frame counts at typical wallpaper lengths and rates,
    // nowhere near f64's ~2^52 exact-integer range.
    #[expect(clippy::cast_precision_loss, reason = "frame counts, nowhere near 2^52")]
    frame
        .timestamp()
        .or_else(|| frame.pts())
        .map_or(0.0, |pts| pts as f64 * f64::from(time_base))
}

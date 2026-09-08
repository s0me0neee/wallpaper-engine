//! In-process libav access through the `ffmpeg-next` bindings.
//!
//! No ffmpeg binary is involved at runtime — the export pipelines link
//! libavformat/libavcodec/libavfilter directly, so an export behaves the same
//! whatever happens to be on the user's PATH. The trade is at build time: the
//! ffmpeg development libraries must be installed (`brew install ffmpeg`, or
//! `libavcodec-dev` and friends).

use anyhow::{Context, Result, anyhow};
use ffmpeg::media::Type;
use ffmpeg_next as ffmpeg;
use std::{path::Path, sync::OnceLock};

/// libav's global setup, run at most once per process.
pub fn init() -> Result<()> {
    static READY: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    READY
        .get_or_init(|| {
            ffmpeg::init().map_err(|error| error.to_string())?;
            // libav logs to stderr uninvited, and at anything above Error it
            // narrates every packet it mishandles.
            ffmpeg::log::set_level(ffmpeg::log::Level::Error);
            Ok(())
        })
        .clone()
        .map_err(|error| anyhow!("initialising libav: {error}"))
}

/// What we need to know about a source video before re-encoding it.
#[derive(Debug, Clone)]
pub struct MediaInfo {
    pub width: u32,
    pub height: u32,
    /// Frames per second, from the stream's rational frame rate.
    pub fps: f64,
    pub duration: f64,
    pub codec: String,
    pub pixel_format: String,
    pub has_audio: bool,
}

/// Convert a stream's rational rate to fps.
///
/// libav writes 0/0 for streams with no meaningful rate (audio, single
/// images), which is "unknown" rather than an error.
pub fn rate_to_fps(rate: ffmpeg::Rational) -> Option<f64> {
    if rate.numerator() <= 0 || rate.denominator() <= 0 {
        return None;
    }
    Some(f64::from(rate))
}

/// Open `path` and build a decoder for its best video stream.
///
/// Returns the input context alongside the stream index, because the caller
/// needs the index to tell that stream's packets apart while demuxing.
pub fn open_video(
    path: &Path,
) -> Result<(ffmpeg::format::context::Input, usize, ffmpeg::decoder::Video)> {
    init()?;
    let ictx = ffmpeg::format::input(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let stream = ictx
        .streams()
        .best(Type::Video)
        .with_context(|| format!("{} has no video stream", path.display()))?;
    let index = stream.index();
    let decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .and_then(|context| context.decoder().video())
        .with_context(|| format!("opening the video decoder for {}", path.display()))?;
    Ok((ictx, index, decoder))
}

/// Read the dimensions, frame rate and duration of a media file.
pub fn probe(path: &Path) -> Result<MediaInfo> {
    let (ictx, index, decoder) = open_video(path)?;
    let stream = ictx
        .stream(index)
        .expect("the index came from this context");

    // Prefer the container duration: a stream may omit it, and for our
    // purposes the whole-file length is what a loop is measured against.
    let duration = if ictx.duration() > 0 {
        ictx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
    } else if stream.duration() > 0 {
        stream.duration() as f64 * f64::from(stream.time_base())
    } else {
        0.0
    };

    let fps = rate_to_fps(stream.avg_frame_rate())
        .or_else(|| rate_to_fps(stream.rate()))
        .unwrap_or(0.0);

    let codec = ffmpeg::decoder::find(stream.parameters().id())
        .map(|codec| codec.name().to_string())
        .unwrap_or_default();

    Ok(MediaInfo {
        width: decoder.width(),
        height: decoder.height(),
        fps,
        duration,
        codec,
        pixel_format: decoder
            .format()
            .descriptor()
            .map(|descriptor| descriptor.name().to_string())
            .unwrap_or_default(),
        has_audio: ictx.streams().best(Type::Audio).is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffmpeg::Rational;

    #[test]
    fn converts_rational_frame_rates() {
        assert_eq!(rate_to_fps(Rational(60, 1)), Some(60.0));
        assert_eq!(
            rate_to_fps(Rational(30000, 1001)).map(|fps| (fps * 100.0).round()),
            Some(2997.0)
        );
    }

    #[test]
    fn treats_an_unknown_frame_rate_as_absent() {
        assert_eq!(rate_to_fps(Rational(0, 0)), None);
        assert_eq!(rate_to_fps(Rational(25, 0)), None);
        assert_eq!(rate_to_fps(Rational(-1, 1)), None);
    }
}

//! Looping video output.
//!
//! Video wallpapers already contain a finished, seamlessly looping file, so
//! the job here is packaging rather than rendering: hand the user an mp4 that
//! an ordinary wallpaper app will accept, without needlessly degrading it.

use super::{Options, ffmpeg, needs_reencode};
use anyhow::{Context, Result};
use std::{ffi::OsString, path::Path};

/// Container-level compatibility. Anything else gets re-encoded.
///
/// H.264 in 4:2:0 is the format every wallpaper app, phone and TV decodes.
/// A source already in it can be copied through untouched.
fn is_broadly_compatible(info: &ffmpeg::MediaInfo) -> bool {
    let codec_ok = info.codec == "h264";
    // yuv420p and its `yuvj`/`10le` relatives are distinct: only plain 8-bit
    // 4:2:0 is universally safe.
    let pixels_ok = matches!(info.pixel_format.as_str(), "yuv420p" | "yuvj420p");
    codec_ok && pixels_ok
}

/// Write `source` to `out` as a wallpaper-app-friendly mp4.
///
/// Returns whether the video stream was copied rather than re-encoded.
pub fn export(source: &Path, out: &Path, info: &ffmpeg::MediaInfo, options: &Options) -> Result<bool> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let copy = !needs_reencode(options) && is_broadly_compatible(info);

    let mut args: Vec<OsString> = vec!["-i".into(), source.into()];

    // Trimming is a container operation, so it does not force a re-encode; the
    // cut lands on a keyframe when copying, which is fine for a loop.
    if let Some(duration) = options.duration {
        args.push("-t".into());
        args.push(format!("{duration}").into());
    }

    args.push("-map".into());
    args.push("0:v:0".into());

    if options.audio && info.has_audio {
        args.push("-map".into());
        args.push("0:a:0".into());
        args.push("-c:a".into());
        args.push("aac".into());
    } else {
        args.push("-an".into());
    }

    if copy {
        args.push("-c:v".into());
        args.push("copy".into());
    } else {
        if let Some(resolution) = options.resolution {
            args.push("-vf".into());
            args.push(
                format!(
                    "scale={}:{}:flags=lanczos",
                    resolution.width, resolution.height
                )
                .into(),
            );
        }
        if let Some(fps) = options.fps {
            args.push("-r".into());
            args.push(format!("{fps}").into());
        }
        args.extend([
            "-c:v".into(),
            "libx264".into(),
            "-preset".into(),
            "slow".into(),
            "-crf".into(),
            "18".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
        ]);
    }

    // Puts the index at the front so a player can start without reading the
    // whole file — what makes a wallpaper app show the video immediately.
    args.push("-movflags".into());
    args.push("+faststart".into());
    args.push(out.into());

    ffmpeg::run(args)?;
    Ok(copy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(codec: &str, pixel_format: &str) -> ffmpeg::MediaInfo {
        ffmpeg::MediaInfo {
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

//! Single-frame PNG output.
//!
//! For now the only source is a video; the scene renderer will add a second
//! entry point here that writes an already-rendered frame buffer.

use super::ffmpeg;
use anyhow::{Context, Result, bail};
use std::{ffi::OsString, path::Path};

/// Grab one frame from `video` at `at_seconds` and write it as a PNG.
///
/// The seek is placed before `-i` so ffmpeg jumps to the nearest keyframe
/// instead of decoding from the start — on a 27-second 1440p60 source that is
/// the difference between instant and several seconds.
pub fn from_video(video: &Path, out: &Path, at_seconds: f64, width_height: Option<(u32, u32)>) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let mut args: Vec<OsString> = vec![
        "-ss".into(),
        format!("{at_seconds}").into(),
        "-i".into(),
        video.into(),
        "-frames:v".into(),
        "1".into(),
        // Tell the PNG muxer it is writing a single image rather than a
        // numbered sequence; without it ffmpeg warns and may refuse.
        "-update".into(),
        "1".into(),
    ];

    if let Some((width, height)) = width_height {
        args.push("-vf".into());
        args.push(format!("scale={width}:{height}:flags=lanczos").into());
    }

    args.push(out.into());
    ffmpeg::run(args)?;

    // Seeking past the end of the file is not an ffmpeg error: it simply
    // produces no frames and exits successfully, leaving no output.
    if !out.is_file() {
        bail!(
            "no frame at {at_seconds}s in {} — is the timestamp past the end?",
            video.display()
        );
    }
    Ok(())
}

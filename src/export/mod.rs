//! The shared output stage: still PNG and looping video.
//!
//! Every pipeline — video passthrough, the scene renderer, web capture —
//! converges here, so the options and the file naming live in one place and
//! each pipeline only has to produce frames.

pub mod ffmpeg;
pub mod still;
pub mod video;

use anyhow::{Result, bail};
use std::path::PathBuf;

/// A target frame size, as given by `--resolution`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl std::str::FromStr for Resolution {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (width, height) = value
            .split_once(['x', 'X', '*'])
            .ok_or_else(|| anyhow::anyhow!("expected WIDTHxHEIGHT, got {value:?}"))?;
        let width: u32 = width.trim().parse()?;
        let height: u32 = height.trim().parse()?;

        // H.264 with 4:2:0 chroma cannot represent odd dimensions, and every
        // ordinary wallpaper app expects 4:2:0. Rejecting here beats an
        // inscrutable encoder error later.
        if width == 0 || height == 0 {
            bail!("resolution must be non-zero, got {width}x{height}");
        }
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            bail!("resolution must be even for H.264, got {width}x{height}");
        }
        Ok(Resolution { width, height })
    }
}

impl std::fmt::Display for Resolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// What to produce, and how.
#[derive(Debug, Clone)]
pub struct Options {
    /// Directory this wallpaper's own files land in — already named for it,
    /// so nothing written under it needs to repeat the name.
    pub out_dir: PathBuf,
    pub video: bool,
    /// Timestamps (in seconds) to export as still PNGs. Empty means no
    /// stills: a video wallpaper already has a finished loop, so unless a
    /// frame is asked for there is nothing else to produce.
    pub frames: Vec<f64>,
    /// Target size. `None` keeps the source resolution.
    pub resolution: Option<Resolution>,
    /// Target frame rate. `None` keeps the source rate.
    pub fps: Option<f64>,
    /// Trim the output to this many seconds.
    pub duration: Option<f64>,
    /// Keep the audio track. Off by default: wallpaper apps ignore it and it
    /// is usually the bulk of the file size.
    pub audio: bool,
}

/// True when the output cannot simply reuse the source's encoded frames.
pub fn needs_reencode(options: &Options) -> bool {
    options.resolution.is_some() || options.fps.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_resolution() {
        assert_eq!(
            "1920x1080".parse::<Resolution>().unwrap(),
            Resolution { width: 1920, height: 1080 }
        );
        assert_eq!(
            "3840X2160".parse::<Resolution>().unwrap(),
            Resolution { width: 3840, height: 2160 }
        );
    }

    #[test]
    fn rejects_dimensions_h264_cannot_encode() {
        assert!("1921x1080".parse::<Resolution>().is_err());
        assert!("1920x1081".parse::<Resolution>().is_err());
        assert!("0x1080".parse::<Resolution>().is_err());
    }

    #[test]
    fn rejects_malformed_resolutions() {
        assert!("1920".parse::<Resolution>().is_err());
        assert!("wide x tall".parse::<Resolution>().is_err());
        assert!("".parse::<Resolution>().is_err());
    }
}

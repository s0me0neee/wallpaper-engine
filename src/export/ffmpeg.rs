//! Thin wrapper around the `ffmpeg` and `ffprobe` binaries.
//!
//! Deliberately a subprocess rather than `libav` bindings: it keeps the build
//! free of native library pain, and ffmpeg's CLI is a far more stable contract
//! than its C API. Arguments are passed as a vector, never a shell string —
//! wallpaper filenames contain spaces and CJK text, and no shell ever sees them.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    ffi::OsStr,
    path::Path,
    process::{Command, Stdio},
};

/// Override for either binary, so a user with a non-PATH build can point at it.
fn binary(tool: &str) -> String {
    let variable = tool.to_ascii_uppercase();
    std::env::var(&variable).unwrap_or_else(|_| tool.to_string())
}

/// Check that both binaries are runnable, with an actionable message if not.
pub fn require() -> Result<()> {
    for tool in ["ffmpeg", "ffprobe"] {
        let status = Command::new(binary(tool))
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        match status {
            Ok(status) if status.success() => {}
            Ok(status) => bail!("`{tool} -version` failed with {status}"),
            Err(error) => bail!(
                "cannot run `{tool}`: {error}. Install ffmpeg, or set the \
                 {} environment variable to its path.",
                tool.to_ascii_uppercase()
            ),
        }
    }
    Ok(())
}

/// Run ffmpeg, surfacing its own diagnostics when it fails.
///
/// ffmpeg writes everything to stderr and is extremely verbose, so on success
/// it is discarded and on failure only the tail is kept — the last few lines
/// carry the actual reason, the rest is codec banners.
pub fn run<I, S>(args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let output = Command::new(binary("ffmpeg"))
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"])
        .args(args)
        .stdout(Stdio::null())
        .output()
        .context("running ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(6).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        bail!("ffmpeg failed ({}): {}", output.status, tail.join("; "));
    }
    Ok(())
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

#[derive(Deserialize)]
struct ProbeOutput {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    #[serde(default)]
    format: ProbeFormat,
}

#[derive(Deserialize)]
struct ProbeStream {
    #[serde(default)]
    codec_type: String,
    #[serde(default)]
    codec_name: String,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    pix_fmt: Option<String>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Default, Deserialize)]
struct ProbeFormat {
    #[serde(default)]
    duration: Option<String>,
}

/// Parse ffprobe's rational frame rate, e.g. `"60/1"` or `"30000/1001"`.
///
/// A zero denominator or numerator means "unknown", which ffprobe reports for
/// streams with no meaningful rate (audio, single images) — not an error.
fn parse_rational(value: &str) -> Option<f64> {
    let (numerator, denominator) = value.split_once('/')?;
    let numerator: f64 = numerator.trim().parse().ok()?;
    let denominator: f64 = denominator.trim().parse().ok()?;
    if numerator <= 0.0 || denominator <= 0.0 {
        return None;
    }
    Some(numerator / denominator)
}

/// Read the dimensions, frame rate and duration of a media file.
pub fn probe(path: &Path) -> Result<MediaInfo> {
    let output = Command::new(binary("ffprobe"))
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(path)
        .output()
        .with_context(|| format!("probing {}", path.display()))?;

    if !output.status.success() {
        bail!(
            "ffprobe could not read {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let probe: ProbeOutput = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parsing ffprobe output for {}", path.display()))?;

    let video = probe
        .streams
        .iter()
        .find(|stream| stream.codec_type == "video")
        .with_context(|| format!("{} has no video stream", path.display()))?;

    let has_audio = probe
        .streams
        .iter()
        .any(|stream| stream.codec_type == "audio");

    // Prefer the container duration: a stream may omit it, and for our purposes
    // the whole-file length is what a loop is measured against.
    let duration = probe
        .format
        .duration
        .as_deref()
        .or(video.duration.as_deref())
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);

    let fps = video
        .avg_frame_rate
        .as_deref()
        .and_then(parse_rational)
        .or_else(|| video.r_frame_rate.as_deref().and_then(parse_rational))
        .unwrap_or(0.0);

    Ok(MediaInfo {
        width: video.width.unwrap_or(0),
        height: video.height.unwrap_or(0),
        fps,
        duration,
        codec: video.codec_name.clone(),
        pixel_format: video.pix_fmt.clone().unwrap_or_default(),
        has_audio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rational_frame_rates() {
        assert_eq!(parse_rational("60/1"), Some(60.0));
        assert_eq!(parse_rational("30000/1001").map(|v| (v * 100.0).round()), Some(2997.0));
    }

    #[test]
    fn treats_an_unknown_frame_rate_as_absent() {
        // ffprobe writes 0/0 for streams with no meaningful rate.
        assert_eq!(parse_rational("0/0"), None);
        assert_eq!(parse_rational("25"), None);
        assert_eq!(parse_rational(""), None);
    }

    #[test]
    fn reads_a_probe_document() {
        let json = r#"{
            "streams":[
                {"codec_type":"video","codec_name":"h264","width":2560,"height":1440,
                 "pix_fmt":"yuv420p","avg_frame_rate":"60/1","r_frame_rate":"60/1"},
                {"codec_type":"audio","codec_name":"aac","avg_frame_rate":"0/0"}
            ],
            "format":{"duration":"27.066667"}
        }"#;
        let probe: ProbeOutput = serde_json::from_str(json).unwrap();
        assert_eq!(probe.streams.len(), 2);
        assert_eq!(probe.streams[0].width, Some(2560));
        assert_eq!(probe.format.duration.as_deref(), Some("27.066667"));
    }
}

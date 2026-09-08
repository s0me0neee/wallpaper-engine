//! `project.json` — the manifest at the root of every wallpaper.
//!
//! This is the first file the exporter reads and the only one common to all
//! wallpaper types. Its `type` field selects the entire downstream pipeline,
//! and its `file` field names the entry point that pipeline consumes: a
//! `scene.pkg`, a video, or an `index.html`.

use crate::paths::resolve_under;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::{
    fmt,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

/// The wallpaper type, and with it the pipeline that can export it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Scene,
    Video,
    Web,
    /// A Windows executable. Nothing we can render.
    Application,
    /// Anything else, kept verbatim so the error can name it.
    Unknown(String),
}

impl Kind {
    /// Wallpaper Engine does not normalize the case it writes: real samples
    /// contain both `"Scene"` and `"scene"`. Matching exactly would silently
    /// route half the corpus to the unknown branch.
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "scene" => Kind::Scene,
            "video" => Kind::Video,
            "web" => Kind::Web,
            "application" => Kind::Application,
            _ => Kind::Unknown(raw.to_string()),
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Scene => f.write_str("scene"),
            Kind::Video => f.write_str("video"),
            Kind::Web => f.write_str("web"),
            Kind::Application => f.write_str("application"),
            Kind::Unknown(raw) => write!(f, "unknown ({raw})"),
        }
    }
}

/// The subset of `project.json` we act on.
///
/// Everything else in the file is Workshop bookkeeping — ratings, tags, the
/// Steam id — that has no effect on the export, so it is not modelled.
#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    title: String,
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    preview: Option<String>,
    /// Set by Wallpaper Engine on wallpapers too large to publish normally.
    /// In practice it flags the bundled-media monsters — hundreds of megabytes
    /// of video and audio with no single canonical frame to capture.
    #[serde(default)]
    oversized: bool,
    #[serde(default)]
    general: General,
}

#[derive(Debug, Default, Deserialize)]
struct General {
    #[serde(default)]
    properties: Map<String, Value>,
}

/// A parsed wallpaper, with its paths resolved and checked.
pub struct Project {
    pub kind: Kind,
    pub title: String,
    /// Directory holding `project.json`; every other path is relative to it.
    pub root: PathBuf,
    /// The entry point named by `file`, resolved under `root`.
    ///
    /// Absent when the manifest omits `file`, which is malformed but not worth
    /// failing on until a pipeline actually needs the entry point.
    pub entry: Option<PathBuf>,
    pub preview: Option<PathBuf>,
    /// `scene.pkg`, when present. Scene wallpapers keep their entry point and
    /// every asset inside this archive rather than loose on disk, so `entry`
    /// naming a file that does not exist is normal for them, not an error.
    pub package: Option<PathBuf>,
    pub oversized: bool,
    /// User-facing settings from `general.properties`, still as JSON. Scene
    /// rendering will bind these to shader uniforms; nothing reads them yet.
    pub properties: Map<String, Value>,
}

/// Find `project.json` given either it, or the directory containing it.
fn locate(input: &Path) -> Result<PathBuf> {
    if input.is_dir() {
        let manifest = input.join("project.json");
        if !manifest.is_file() {
            bail!(
                "{} is not a wallpaper: no project.json inside it",
                input.display()
            );
        }
        return Ok(manifest);
    }

    if input.file_name().is_some_and(|name| name == "project.json") {
        return Ok(input.to_path_buf());
    }

    bail!(
        "expected a wallpaper directory or a project.json, got {}",
        input.display()
    )
}

impl Project {
    /// Load the wallpaper at `input`, which may be its directory or its
    /// `project.json`.
    pub fn load(input: &Path) -> Result<Self> {
        let manifest_path = locate(input)?;
        let root = manifest_path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();

        let file = File::open(&manifest_path)
            .with_context(|| format!("opening {}", manifest_path.display()))?;
        let manifest: Manifest = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("parsing {}", manifest_path.display()))?;

        // `file` and `preview` come from Workshop content, so they get the same
        // traversal check as archive entries rather than a bare join.
        let entry = manifest
            .file
            .as_deref()
            .map(|name| resolve_under(&root, name))
            .transpose()
            .with_context(|| format!("resolving the \"file\" entry of {}", manifest_path.display()))?;

        let preview = manifest
            .preview
            .as_deref()
            .and_then(|name| resolve_under(&root, name).ok())
            .filter(|path| path.is_file());

        let package = Some(root.join("scene.pkg")).filter(|path| path.is_file());

        Ok(Project {
            kind: Kind::parse(&manifest.kind),
            title: manifest.title,
            root,
            entry,
            preview,
            package,
            oversized: manifest.oversized,
            properties: manifest.general.properties,
        })
    }

    /// True when the entry point is inside `scene.pkg` rather than on disk.
    pub fn entry_is_packaged(&self) -> bool {
        self.package.is_some() && self.entry.as_deref().is_some_and(|entry| !entry.exists())
    }

    /// The entry point, failing with a pipeline-specific message if the
    /// manifest omitted it or it is not on disk.
    pub fn require_entry(&self) -> Result<&Path> {
        let entry = self
            .entry
            .as_deref()
            .context("project.json has no \"file\" entry, so there is nothing to export")?;
        if !entry.exists() {
            bail!("{} is named by project.json but missing", entry.display());
        }
        Ok(entry)
    }

    /// A display name, falling back to the directory when the title is blank.
    pub fn display_name(&self) -> &str {
        if self.title.trim().is_empty() {
            self.root
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<untitled>")
        } else {
            &self.title
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_matching_ignores_case() {
        assert_eq!(Kind::parse("Scene"), Kind::Scene);
        assert_eq!(Kind::parse("scene"), Kind::Scene);
        assert_eq!(Kind::parse("video"), Kind::Video);
        assert_eq!(Kind::parse("Web"), Kind::Web);
        assert_eq!(Kind::parse(" application "), Kind::Application);
    }

    #[test]
    fn an_unrecognized_type_keeps_its_spelling_for_the_error() {
        assert_eq!(
            Kind::parse("Hologram"),
            Kind::Unknown("Hologram".to_string())
        );
    }

    #[test]
    fn a_manifest_needs_only_a_type() {
        let manifest: Manifest = serde_json::from_str(r#"{"type":"video"}"#).unwrap();
        assert_eq!(Kind::parse(&manifest.kind), Kind::Video);
        assert!(manifest.file.is_none());
        assert!(!manifest.oversized);
    }

    #[test]
    fn workshop_bookkeeping_does_not_break_parsing() {
        // Real manifests carry fields we do not model, and `workshopid` is a
        // string in some and a number in others. Neither may be fatal.
        let manifest: Manifest = serde_json::from_str(
            r#"{"type":"scene","file":"scene.pkg","workshopid":123,
                "tags":["Anime"],"approved":true,"version":5}"#,
        )
        .unwrap();
        assert_eq!(manifest.file.as_deref(), Some("scene.pkg"));
    }

    #[test]
    fn a_hostile_file_entry_is_rejected() {
        let root = Path::new("/wallpaper");
        assert!(resolve_under(root, "../../.ssh/id_rsa").is_err());
    }
}

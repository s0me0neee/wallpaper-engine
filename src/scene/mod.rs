//! Scene wallpapers: the layered, shader-driven kind.
//!
//! Everything a scene needs lives inside `scene.pkg`, so nothing here touches
//! the filesystem beyond opening that archive.

pub mod compose;
pub mod model;
pub mod particle;
pub mod puppet;
pub mod render;
pub mod script;
pub mod scripting;
pub mod sprite;
pub mod text;
pub mod video;

use anyhow::{Context, Result};
use model::Scene;

/// Read and parse `scene.json` out of an open package, with the project's settings applied.
pub fn load(archive: &mut crate::pkg::Archive, properties: &serde_json::Map<String, serde_json::Value>) -> Result<Scene> {
    let bytes = archive
        .read("scene.json")
        .context("the package has no scene.json")?;
    model::parse_scene(&bytes, properties).context("parsing scene.json")
}

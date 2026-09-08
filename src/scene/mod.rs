//! Scene wallpapers: the layered, shader-driven kind.
//!
//! Everything a scene needs lives inside `scene.pkg`, so nothing here touches
//! the filesystem beyond opening that archive.

pub mod compose;
pub mod model;
pub mod particle;
pub mod puppet;
pub mod render;

use anyhow::{Context, Result};
use model::Scene;

/// Read and parse `scene.json` out of an open package.
pub fn load(archive: &mut crate::pkg::Archive) -> Result<Scene> {
    let bytes = archive
        .read("scene.json")
        .context("the package has no scene.json")?;
    serde_json::from_slice(&bytes).context("parsing scene.json")
}

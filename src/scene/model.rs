//! Serde types for the JSON files inside a scene package.
//!
//! `scene.json` is the root. Each object names a model under `models/`, which
//! names a material under `materials/`, which names the textures it samples.
//! Resolving that chain is what turns a scene into pixels.
//!
//! Numbers arrive in two shapes: real JSON numbers, and space-separated
//! strings like `"3840.00000 2160.00000 0.00000"`. Both appear in the same
//! file, sometimes for the same concept, so vectors get a custom deserializer.
//!
//! These types describe the file format, not only the parts the current
//! compositor reads. Fields like `camera`, `zoom` and the effect passes are
//! parsed and left unused on purpose: they are what the shader renderer will
//! bind to, and discovering their shape is most of the work. Hence the
//! module-wide allow — every other module keeps the warning.
#![allow(dead_code)]

use anyhow::Result;
use serde::{Deserialize, Deserializer, de};
use serde_json::{Map, Value};

/// A 2- or 3-component vector, however the file spelled it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn splat(value: f32) -> Self {
        Vec3 { x: value, y: value, z: value }
    }
}

/// Parse `"1.0 2.0 3.0"`, tolerating two components or a bare scalar.
///
/// A missing component is 0 rather than an error: Wallpaper Engine writes
/// two-component vectors for anything with no depth, and a scalar where a
/// uniform happens to be uniform.
fn parse_vec3(raw: &str) -> Option<Vec3> {
    let mut parts = raw
        .split_whitespace()
        .map(|part| part.parse::<f32>().ok());

    let x = parts.next()??;
    let y = parts.next().unwrap_or(Some(0.0))?;
    let z = parts.next().unwrap_or(Some(0.0))?;
    if parts.next().is_some() {
        return None;
    }
    Some(Vec3 { x, y, z })
}

impl<'de> Deserialize<'de> for Vec3 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::String(raw) => parse_vec3(&raw)
                .ok_or_else(|| de::Error::custom(format!("malformed vector {raw:?}"))),
            Value::Number(number) => {
                // Scene values are small (scale factors, unit vectors); the
                // precision f32 drops here is well below what the format
                // itself carries (5 decimal digits, per the doc comment above).
                #[expect(clippy::cast_possible_truncation, reason = "scene values are small")]
                let value = number.as_f64().unwrap_or_default() as f32;
                Ok(Vec3::splat(value))
            }
            other => Err(de::Error::custom(format!(
                "expected a vector, got {other}"
            ))),
        }
    }
}

fn one() -> f32 {
    1.0
}

fn vec3_one() -> Vec3 {
    Vec3::splat(1.0)
}

fn r#true() -> bool {
    true
}

/// The root of `scene.json`.
#[derive(Debug, Deserialize)]
pub struct Scene {
    #[serde(default)]
    pub camera: Camera,
    #[serde(default)]
    pub general: General,
    #[serde(default)]
    pub objects: Vec<Object>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Camera {
    /// Camera position. Under an orthographic projection this is the centre of
    /// the visible rectangle, so it offsets the whole scene.
    #[serde(default)]
    pub eye: Vec3,
    #[serde(default)]
    pub center: Vec3,
    #[serde(default)]
    pub up: Vec3,
}

#[derive(Debug, Default, Deserialize)]
pub struct General {
    /// The canvas. Present on every wallpaper-shaped scene; absent means the
    /// scene uses a perspective camera, which we cannot flatten.
    #[serde(rename = "orthogonalprojection")]
    pub orthographic: Option<Orthographic>,
    #[serde(default)]
    pub clearcolor: Vec3,
    #[serde(default = "r#true")]
    pub clearenabled: bool,
    #[serde(default = "one")]
    pub zoom: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Orthographic {
    pub width: u32,
    pub height: u32,
}

/// One entry in the scene's object list. Layer order is array order.
///
/// The same struct covers images, particle systems and sound emitters, because
/// that is how the file models them — they are distinguished by which of
/// `image`, `particle` and `sound` is present, not by a type tag.
#[derive(Debug, Deserialize)]
pub struct Object {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub id: i64,

    /// Path to a model JSON, for image layers.
    #[serde(default)]
    pub image: Option<String>,
    /// Path to a particle preset.
    #[serde(default)]
    pub particle: Option<String>,
    /// Audio tracks. Never rendered.
    #[serde(default)]
    pub sound: Option<Vec<String>>,

    /// Centre of the object in scene units.
    #[serde(default)]
    pub origin: Vec3,
    /// Extent in scene units, before `scale`. Absent for non-image objects.
    #[serde(default)]
    pub size: Option<Vec3>,
    #[serde(default = "vec3_one")]
    pub scale: Vec3,
    #[serde(default)]
    pub angles: Vec3,

    #[serde(default = "one")]
    pub alpha: f32,
    #[serde(default = "vec3_one")]
    pub color: Vec3,
    #[serde(default = "one")]
    pub brightness: f32,
    #[serde(default = "r#true")]
    pub visible: bool,

    /// Post-process effects. Not applied yet; their presence is what tells the
    /// user the still is missing something.
    #[serde(default)]
    pub effects: Vec<Effect>,
    /// Puppet-warp clips bound to this object. Each names a baked animation id
    /// inside the model's `*_puppet.mdl`; `rate` scales playback speed.
    #[serde(default)]
    pub animationlayers: Vec<AnimationLayer>,
}

#[derive(Debug, Deserialize)]
pub struct AnimationLayer {
    /// The baked clip's id inside the puppet `.mdl`.
    #[serde(default)]
    pub animation: Option<u32>,
    #[serde(default = "one")]
    pub rate: f32,
    #[serde(default = "r#true")]
    pub visible: bool,
}

/// The first playable puppet clip bound to an object, if any.
pub fn visible_animation_layer(object: &Object) -> Option<&AnimationLayer> {
    object.animationlayers.iter().find(|layer| layer.visible)
}

/// What kind of object this is, by which field is populated.
pub fn is_image(object: &Object) -> bool {
    object.image.is_some()
}

pub fn is_particle(object: &Object) -> bool {
    object.particle.is_some()
}

pub fn is_sound(object: &Object) -> bool {
    object.sound.is_some()
}

/// A human name for an object: its `name` when it has one, else `object <id>`.
pub fn label(object: &Object) -> String {
    if object.name.is_empty() {
        format!("object {}", object.id)
    } else {
        object.name.clone()
    }
}

/// Effects that would actually be drawn.
pub fn visible_effects(object: &Object) -> impl Iterator<Item = &Effect> {
    object.effects.iter().filter(|effect| effect.visible)
}

#[derive(Debug, Deserialize)]
pub struct Effect {
    #[serde(default)]
    pub name: String,
    /// Path to the effect definition inside the package.
    #[serde(default)]
    pub file: String,
    #[serde(default = "r#true")]
    pub visible: bool,
    #[serde(default)]
    pub passes: Vec<EffectPass>,
}

#[derive(Debug, Deserialize)]
pub struct EffectPass {
    /// Material keys bound to shader uniforms for this pass.
    #[serde(default)]
    pub constantshadervalues: Map<String, Value>,
    /// Texture slots, positionally bound to `g_Texture1`, `g_Texture2`, ...
    /// (`g_Texture0` is always the previous pass and never appears here).
    /// `null` means "use the shader's own annotation default", e.g.
    /// `util/noflow`.
    #[serde(default)]
    pub textures: Vec<Option<String>>,
}

/// `effects/<name>/effect.json` — names the material(s) an effect instance's
/// passes run, one per entry in `EffectPass` above, in the same order.
#[derive(Debug, Deserialize)]
pub struct EffectDefinition {
    #[serde(default)]
    pub passes: Vec<EffectDefinitionPass>,
}

#[derive(Debug, Deserialize)]
pub struct EffectDefinitionPass {
    /// Path to the `materials/*.json` this pass renders with.
    pub material: String,
}

/// `models/*.json` — the indirection between an object and its material.
#[derive(Debug, Deserialize)]
pub struct Model {
    pub material: String,
    /// Take the layer's extent from the texture rather than from `size`.
    #[serde(default)]
    pub autosize: bool,
    /// Path to a binary `*_puppet.mdl` when the layer is a warp puppet.
    #[serde(default)]
    pub puppet: Option<String>,
}

/// `materials/*.json`.
#[derive(Debug, Deserialize)]
pub struct Material {
    #[serde(default)]
    pub passes: Vec<MaterialPass>,
}

#[derive(Debug, Deserialize)]
pub struct MaterialPass {
    #[serde(default)]
    pub shader: String,
    #[serde(default)]
    pub blending: String,
    /// Texture names, relative to `materials/` and without the `.tex`
    /// extension. A slot may be null when the shader supplies it another way.
    #[serde(default)]
    pub textures: Vec<Option<String>>,
}

/// The texture the layer is actually made of: slot 0 of the first pass.
pub fn base_texture(material: &Material) -> Option<&str> {
    material
        .passes
        .first()?
        .textures
        .first()?
        .as_deref()
        .filter(|name| !name.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_three_component_vector() {
        assert_eq!(
            parse_vec3("3840.00000 2160.00000 0.00000"),
            Some(Vec3 { x: 3840.0, y: 2160.0, z: 0.0 })
        );
    }

    #[test]
    fn pads_shorter_vectors_with_zero() {
        // Two-component vectors are written for anything flat, e.g. `size`.
        assert_eq!(parse_vec3("206 74"), Some(Vec3 { x: 206.0, y: 74.0, z: 0.0 }));
        assert_eq!(parse_vec3("-1.5"), Some(Vec3 { x: -1.5, y: 0.0, z: 0.0 }));
    }

    #[test]
    fn rejects_junk_and_overlong_vectors() {
        assert_eq!(parse_vec3("1 2 3 4"), None);
        assert_eq!(parse_vec3("left right"), None);
        assert_eq!(parse_vec3(""), None);
    }

    #[test]
    fn a_scalar_deserializes_as_a_uniform_vector() {
        // `scale` is sometimes a bare number rather than a vector string.
        let value: Vec3 = serde_json::from_str("2.5").unwrap();
        assert_eq!(value, Vec3::splat(2.5));
    }

    #[test]
    fn object_defaults_match_wallpaper_engine() {
        // Most objects omit most fields; the defaults have to be the identity
        // transform or every such layer renders wrong.
        let object: Object = serde_json::from_str(r#"{"name":"x"}"#).unwrap();
        assert_eq!(object.scale, Vec3::splat(1.0));
        assert_eq!(object.color, Vec3::splat(1.0));
        assert_eq!(object.alpha, 1.0);
        assert_eq!(object.brightness, 1.0);
        assert!(object.visible);
        assert!(object.size.is_none());
    }

    #[test]
    fn object_kind_comes_from_which_field_is_present() {
        let image: Object = serde_json::from_str(r#"{"image":"models/a.json"}"#).unwrap();
        let particle: Object =
            serde_json::from_str(r#"{"particle":"particles/presets/ember.json"}"#).unwrap();
        let sound: Object = serde_json::from_str(r#"{"sound":["sounds/a.mp3"]}"#).unwrap();

        assert!(is_image(&image) && !is_particle(&image) && !is_sound(&image));
        assert!(is_particle(&particle) && !is_image(&particle));
        assert!(is_sound(&sound) && !is_image(&sound));
    }

    #[test]
    fn reads_the_base_texture_out_of_a_material() {
        let material: Material = serde_json::from_str(
            r#"{"passes":[{"shader":"genericimage2","textures":["yilaina",null]}]}"#,
        )
        .unwrap();
        assert_eq!(base_texture(&material), Some("yilaina"));
    }

    #[test]
    fn a_material_with_no_texture_has_no_base() {
        // Solid-colour layers and effect-only passes both look like this.
        let empty: Material = serde_json::from_str(r#"{"passes":[{"textures":[null]}]}"#).unwrap();
        assert_eq!(base_texture(&empty), None);

        let blank: Material = serde_json::from_str(r#"{"passes":[{"textures":[""]}]}"#).unwrap();
        assert_eq!(base_texture(&blank), None);
    }
}

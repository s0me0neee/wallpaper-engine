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

/// Parse `scene.json`, reducing driven values to their static defaults first.
pub fn parse_scene(bytes: &[u8]) -> Result<Scene, serde_json::Error> {
    let mut document: Value = serde_json::from_slice(bytes)?;
    hoist_alpha_tracks(&mut document);
    strip_driven_values(&mut document);
    Scene::deserialize(document)
}

/// What can sit beside `value` and turn a plain number into a driven one.
///
/// - `user` — bound to one of the wallpaper's own settings
///   (`project.json`'s `general.properties`), written as
///   `{"user": "moon", "value": true}`, or as
///   `{"user": {"name": "clock", "condition": "1"}, "value": false}` when a
///   layer belongs to one setting of a combo.
/// - `script` / `scriptproperties` — driven by the wallpaper's own JavaScript.
///   `scene_example6` positions a layer with a script that multiplies a slider
///   by the canvas size, and writes the JS source inline.
/// - `animation` — driven by a keyframe track.
/// - `frame` — a keyframe control point (`c0`/`c1`/`c2`, each with `back`,
///   `front`, `lockangle` and `locklength` beside it).
const DRIVER_KEYS: [&str; 5] = ["user", "script", "scriptproperties", "animation", "frame"];

/// Replace every driven value in the document with the static value under it.
///
/// Any value in `scene.json` can be driven rather than fixed, and is then
/// written as an object carrying the driver plus a `value` — the one the
/// wallpaper was published with, and the one we take. It reaches keys no struct
/// here names (`pointsize`, `text.scriptproperties.use24hFormat`, individual
/// `constantshadervalues` entries), which is why this runs over the whole
/// document rather than field by field.
///
/// A driver key has to be present: `constantshadervalues` is a free-form map of
/// material keys, so an object that merely *has* a `value` key may well be a
/// pass's uniform table with a uniform called "value" in it.
///
/// Honouring a *changed* user setting would mean reading `project.json`, which
/// the scene loader deliberately never opens; across the corpus the two agree
/// except for `scene_example4`'s cloud opacity (0.3 here, 0.2 there). Script-
/// and animation-driven values are simply frozen at their published value.
/// Copy each object's `alpha` keyframe track somewhere `strip_driven_values`
/// will not eat it.
///
/// The strip below replaces every driven value with the static one under it,
/// which is right for a user setting or a script but throws away a track that
/// genuinely varies. Alpha is the one that changes what is on screen rather
/// than how it looks: `scene_example8`'s intro watermark fades 1 → 0 over its
/// first seven seconds, and frozen at its published 1.0 it would sit on the
/// wallpaper forever.
/// Where `hoist_alpha_tracks` parks the track, and the key `strip_driven_values`
/// leaves alone.
const HOISTED_TRACK: &str = "alphatrack";

fn hoist_alpha_tracks(node: &mut Value) {
    let Some(objects) = node.get_mut("objects").and_then(Value::as_array_mut) else {
        return;
    };
    for object in objects {
        let Some(map) = object.as_object_mut() else { continue };
        let track = map
            .get("alpha")
            .and_then(|alpha| alpha.get("animation"))
            .cloned();
        if let Some(track) = track {
            map.insert(HOISTED_TRACK.to_string(), track);
        }
    }
}

fn strip_driven_values(node: &mut Value) {
    match node {
        Value::Object(map) => {
            if DRIVER_KEYS.iter().any(|key| map.contains_key(*key))
                && let Some(mut value) = map.remove("value")
            {
                strip_driven_values(&mut value);
                *node = value;
                return;
            }
            for (key, value) in map.iter_mut() {
                // `hoist_alpha_tracks` put a keyframe track here precisely so
                // it would survive; every control point in it carries `frame`,
                // which is itself a driver key, so stripping would collapse the
                // whole track to a list of bare numbers.
                if key == HOISTED_TRACK {
                    continue;
                }
                strip_driven_values(value);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_driven_values(item);
            }
        }
        _ => {}
    }
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
    /// Scene-wide bloom over the finished frame. On for `scene_example4` and
    /// `scene_example8`, and on the latter it is the whole sunset: without it
    /// the sun is a few lit cloud edges rather than a glow.
    #[serde(default)]
    pub bloom: bool,
    /// Render the frame in floating point. Without it a shader result above 1.0
    /// clamps at the end of every pass, and a bloom threshold has nothing left
    /// to extract — see `pass::Format`.
    #[serde(default)]
    pub hdr: bool,
    #[serde(default = "bloom_strength")]
    pub bloomstrength: f32,
    #[serde(default = "bloom_threshold")]
    pub bloomthreshold: f32,
    #[serde(default = "vec3_one")]
    pub bloomtint: Vec3,
    /// The HDR bloom's own strength/threshold, plus how far it spreads. A scene
    /// with `hdr` set tunes these and leaves the pair above at their defaults —
    /// `scene_example8` writes 0.12/0.55/0.67 here and stock 2.0/0.65 there —
    /// so reading the wrong pair drives the bloom at sixteen times its
    /// intended strength.
    #[serde(default = "bloom_hdr_strength")]
    pub bloomhdrstrength: f32,
    #[serde(default = "bloom_hdr_threshold")]
    pub bloomhdrthreshold: f32,
    #[serde(default = "bloom_hdr_scatter")]
    pub bloomhdrscatter: f32,
    #[serde(default = "bloom_hdr_iterations")]
    pub bloomhdriterations: u32,
}

/// `downsample_quarter_bloom.frag`'s own annotation defaults.
fn bloom_strength() -> f32 {
    2.0
}

fn bloom_threshold() -> f32 {
    0.65
}

fn bloom_hdr_strength() -> f32 {
    2.0
}

fn bloom_hdr_threshold() -> f32 {
    1.0
}

fn bloom_hdr_scatter() -> f32 {
    1.619
}

fn bloom_hdr_iterations() -> u32 {
    8
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

    /// Id of the object this one is parented to. A scene is a tree, and
    /// `origin`/`scale`/`angles` below are relative to this parent.
    #[serde(default)]
    pub parent: Option<i64>,

    /// Path to a model JSON, for image layers.
    #[serde(default)]
    pub image: Option<String>,
    /// Path to a particle preset.
    #[serde(default)]
    pub particle: Option<String>,
    /// Per-instance tweaks a scene applies on top of a shared particle preset
    /// (the sliders in WE's own particle panel): emission count, rate, size,
    /// speed and an optional flat colour.
    #[serde(default)]
    pub instanceoverride: Option<InstanceOverride>,
    /// Audio tracks. Never rendered.
    #[serde(default)]
    pub sound: Option<Vec<String>>,

    /// A text layer's string. Usually the design-time preview of a script — see
    /// `scene::text`.
    #[serde(default)]
    pub text: Option<String>,
    /// The font: a path inside the package or the engine's assets, or
    /// `systemfont_<family>`.
    #[serde(default)]
    pub font: Option<String>,
    #[serde(default = "text_point_size")]
    pub pointsize: f32,
    #[serde(default)]
    pub horizontalalign: Option<String>,

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
    /// The `alpha` keyframe track, preserved by `hoist_alpha_tracks` from the
    /// driver object `strip_driven_values` would otherwise flatten.
    #[serde(default)]
    pub alphatrack: Option<Track>,
    #[serde(default = "vec3_one")]
    pub color: Vec3,
    #[serde(default = "one")]
    pub brightness: f32,
    #[serde(default = "r#true")]
    pub visible: bool,
    /// Photoshop-style blend of this layer against the frame beneath it, in
    /// `weBlendColor`'s numbering. Distinct from the material's
    /// `translucent`/`additive`, which is GL blend state; this one needs the
    /// destination in the shader. Zero — plain alpha-over — on every corpus
    /// object but four, all in `scene_example8`.
    #[serde(default, rename = "colorBlendMode")]
    pub color_blend_mode: i32,

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

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct InstanceOverride {
    #[serde(default = "one")]
    pub count: f32,
    #[serde(default = "one")]
    pub rate: f32,
    #[serde(default = "one")]
    pub size: f32,
    #[serde(default = "one")]
    pub speed: f32,
    #[serde(default = "one")]
    pub alpha: f32,
    /// Multiplies the particle's colour, and goes well past 1: `scene_example8`
    /// drives its orange snow at 10, which is what turns a dim sprite into an
    /// ember. Applied after `colorn`, so it scales the override too.
    #[serde(default = "one")]
    pub brightness: f32,
    /// A normalised RGB that replaces whatever colour the preset would pick.
    #[serde(default)]
    pub colorn: Option<Vec3>,
}

impl Default for InstanceOverride {
    fn default() -> Self {
        InstanceOverride {
            count: 1.0,
            rate: 1.0,
            size: 1.0,
            speed: 1.0,
            alpha: 1.0,
            brightness: 1.0,
            colorn: None,
        }
    }
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

pub fn is_text(object: &Object) -> bool {
    object.text.is_some()
}

/// A keyframe track on one scalar property.
///
/// Only channel `c0` is read: the tracks that matter here drive a single
/// number. The control points carry bezier tangents (`back`/`front`) which are
/// ignored — the segments in the corpus are straight ramps, and a linear read
/// of a straight ramp is exact.
#[derive(Debug, Clone, Deserialize)]
pub struct Track {
    #[serde(default)]
    pub c0: Vec<Keyframe>,
    #[serde(default)]
    pub options: TrackOptions,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Keyframe {
    #[serde(default)]
    pub frame: f32,
    #[serde(default)]
    pub value: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrackOptions {
    #[serde(default = "track_fps")]
    pub fps: f32,
    #[serde(default)]
    pub length: f32,
    /// `"single"` plays once and holds; anything else repeats.
    #[serde(default)]
    pub mode: String,
}

impl Default for TrackOptions {
    fn default() -> Self {
        TrackOptions { fps: track_fps(), length: 0.0, mode: String::new() }
    }
}

fn track_fps() -> f32 {
    60.0
}

/// Sample `track` at `seconds`, or `None` when it has no keyframes.
pub fn sample_track(track: &Track, seconds: f32) -> Option<f32> {
    let first = track.c0.first()?;
    let last = track.c0.last()?;
    let mut frame = seconds * track.options.fps.max(1e-3);
    if track.options.mode != "single" && track.options.length > 0.0 {
        frame = frame.rem_euclid(track.options.length);
    }

    if frame <= first.frame {
        return Some(first.value);
    }
    if frame >= last.frame {
        return Some(last.value);
    }
    for pair in track.c0.windows(2) {
        let [a, b] = pair else { continue };
        if frame >= a.frame && frame <= b.frame {
            let span = b.frame - a.frame;
            if span <= 0.0 {
                return Some(b.value);
            }
            let t = (frame - a.frame) / span;
            return Some(a.value + (b.value - a.value) * t);
        }
    }
    Some(last.value)
}

/// Wallpaper Engine's own default point size for a text layer.
fn text_point_size() -> f32 {
    32.0
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
    /// Combo values the wallpaper picked for this pass, overriding the
    /// shader's own `[COMBO]` defaults. This is how a separable blur's second
    /// pass is told to run vertically: `godrays`, `shine` and `blurprecise`
    /// all write `{"VERTICAL": 1}` here, and without it both halves of the
    /// gaussian blur the same axis.
    #[serde(default)]
    pub combos: Map<String, Value>,
    /// Texture slots, indexed by slot number: `textures[1]` is `g_Texture1`.
    /// Entry 0 stands for `g_Texture0`, which is always the previous pass, and
    /// is written as `null`. `null` elsewhere means "use the shader's own
    /// annotation default", e.g. `util/noflow`.
    ///
    /// The corpus pins the indexing down: `shake`'s pass writes `util/white`
    /// at index 2 and `godrays_downsample2`'s writes `util/clouds_256` at
    /// index 2, each the declared default of that shader's `g_Texture2`.
    #[serde(default)]
    pub textures: Vec<Option<String>>,
}

/// `effects/<name>/effect.json` — names the material(s) an effect instance's
/// passes run, one per entry in `EffectPass` above, in the same order.
#[derive(Debug, Deserialize)]
pub struct EffectDefinition {
    #[serde(default)]
    pub passes: Vec<EffectDefinitionPass>,
    /// Intermediate render targets the passes write to and read back.
    #[serde(default)]
    pub fbos: Vec<EffectFbo>,
}

/// One intermediate render target. `scale` is a divisor on the layer's size,
/// not a multiplier: `blur` does its gaussian at 4 (quarter resolution) and
/// `godrays`/`shine` at 2. Ignoring it does not merely cost sharpness — the
/// blur offsets are in texels of *this* target, so running the same pass at
/// full size blurs by a quarter as much.
#[derive(Debug, Deserialize)]
pub struct EffectFbo {
    pub name: String,
    #[serde(default = "one")]
    pub scale: f32,
}

#[derive(Debug, Deserialize)]
pub struct EffectDefinitionPass {
    /// Path to the `materials/*.json` this pass renders with. Absent on a pass
    /// that is a render-target command instead of a draw: `motionblur` writes
    /// `{"command":"copy","target":"_rt_FullCompoBuffer1", ...}` to carry its
    /// accumulation buffer over to the next frame. Such a pass has no entry in
    /// the scene's own `passes` list either — `scene_example3` gives motionblur
    /// two, for the three here.
    #[serde(default)]
    pub material: Option<String>,
    /// The `fbos` entry this pass renders into. Absent means the chain's own
    /// output, which is what the last pass of an effect writes.
    #[serde(default)]
    pub target: Option<String>,
    /// Where this pass's sampler slots come from. This — not the scene's
    /// `textures[]` — is what decides a pass's inputs: `godrays`'s combine
    /// pass binds its half-res rays at slot 0 and `previous` at slot 1, and
    /// `blur`'s binds `previous` at slot 2.
    #[serde(default)]
    pub bind: Vec<EffectBind>,
}

/// One entry of a pass's `bind` list. `name` is either `"previous"` — the
/// image the *effect* was handed, not the previous pass's output — or one of
/// the effect's own `fbos`.
#[derive(Debug, Deserialize)]
pub struct EffectBind {
    pub name: String,
    #[serde(default)]
    pub index: usize,
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
    /// Combo values the material pins on its shader.
    #[serde(default)]
    pub combos: Map<String, Value>,
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

/// How a layer's pixels combine with what is already on the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Blend {
    /// Standard alpha "over". WE's `translucent`, and also `normal` — no
    /// corpus scene leans on `normal`'s opaque-replace nuance, so folding it
    /// in here keeps flat scenes byte-identical.
    #[default]
    Over,
    /// `dst + src·srcAlpha`, saturating. How WE draws glow and particle layers.
    Add,
}

/// The blend mode string WE records for a material pass.
pub fn parse_blend(name: &str) -> Blend {
    match name {
        "additive" => Blend::Add,
        _ => Blend::Over,
    }
}

/// The blend mode of the layer's base pass (slot 0 of the first pass).
pub fn base_blend(material: &Material) -> Blend {
    material.passes.first().map_or(Blend::Over, |pass| parse_blend(&pass.blending))
}

/// Whether the material's base pass refracts the frame behind it.
///
/// `genericparticle.frag` under `#if REFRACT` binds `_rt_FullFrameBuffer` and
/// finishes with `color.rgb *= texSample2D(g_Texture3, refractTexCoord).rgb` —
/// the particle *multiplies* what is behind it rather than covering it. A white
/// sprite over a dark sky therefore disappears, which is why `scene_example8`'s
/// wet snow is invisible in Wallpaper Engine and was a field of white discs
/// here. Nine of `scene_example6`'s particle materials declare it too.
pub fn base_refracts(material: &Material) -> bool {
    material.passes.first().is_some_and(|pass| {
        pass.combos.get("REFRACT").and_then(Value::as_i64).unwrap_or(0) != 0
    })
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
    fn a_user_bound_value_parses_as_the_value_it_wraps() {
        // scene_example3 hides its girl layer behind a "girl" checkbox, and
        // scene_example4 binds a cloud's opacity to a slider.
        let scene = parse_scene(
            br#"{"objects":[
                {"image":"models/a.json","visible":{"user":"girl","value":false}},
                {"image":"models/b.json","alpha":{"user":"opacity","value":0.25},
                 "color":{"user":"color","value":"0.5 0.25 1.0"}}
            ]}"#,
        )
        .unwrap();

        assert!(!scene.objects[0].visible);
        assert_eq!(scene.objects[1].alpha, 0.25);
        assert_eq!(scene.objects[1].color, Vec3 { x: 0.5, y: 0.25, z: 1.0 });
    }

    #[test]
    fn a_scripted_or_animated_value_falls_back_to_its_static_default() {
        // `scene_example6` drives a layer's origin from inline JavaScript and
        // `scene_example8` drives a strength from a keyframe track; both write
        // the published value beside the driver.
        let scene = parse_scene(
            br#"{"objects":[
                {"image":"models/a.json","origin":{"script":"export function update(v){return v}",
                 "scriptproperties":{"x":0.5},"value":"1362.5 736.7 0.0"},
                 "alpha":{"animation":{"c0":[{"frame":0,"value":1}],
                                       "options":{"fps":60,"length":420,"mode":"single"}},"value":0.25}}
            ]}"#,
        )
        .unwrap();

        assert_eq!(scene.objects[0].origin, Vec3 { x: 1362.5, y: 736.7, z: 0.0 });
        assert_eq!(scene.objects[0].alpha, 0.25);
    }

    #[test]
    fn an_alpha_track_survives_the_strip_that_flattens_everything_else() {
        // The strip collapses driven values to their published one, which is
        // right for a slider and wrong for a fade: every control point carries
        // `frame`, itself a driver key, so an unprotected track would come back
        // as a list of bare numbers.
        let scene = parse_scene(
            br#"{"objects":[
                {"image":"models/a.json",
                 "alpha":{"animation":{"c0":[{"frame":0,"value":1},{"frame":420,"value":0}],
                                       "options":{"fps":60,"length":420,"mode":"single"}},"value":1.0}}
            ]}"#,
        )
        .unwrap();

        let track = scene.objects[0].alphatrack.as_ref().expect("the track must survive");
        assert_eq!(track.c0.len(), 2);
        // `scene_example8`'s intro watermark fades over seven seconds; halfway
        // in it is half gone, and past the end it stays gone.
        assert!((sample_track(track, 0.0).unwrap() - 1.0).abs() < 1e-6);
        assert!((sample_track(track, 3.5).unwrap() - 0.5).abs() < 1e-3);
        assert!((sample_track(track, 30.0).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn a_looping_track_wraps_instead_of_holding() {
        let scene = parse_scene(
            br#"{"objects":[
                {"image":"models/a.json",
                 "alpha":{"animation":{"c0":[{"frame":0,"value":0},{"frame":60,"value":1}],
                                       "options":{"fps":60,"length":60,"mode":"loop"}},"value":1.0}}
            ]}"#,
        )
        .unwrap();

        let track = scene.objects[0].alphatrack.as_ref().expect("the track must survive");
        // At 2.5s a one-second loop is halfway through its second repeat.
        assert!((sample_track(track, 2.5).unwrap() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn a_uniform_table_is_not_mistaken_for_a_driven_value() {
        // `constantshadervalues` is a free-form map, so a shader uniform may
        // simply be called "value"; without a driver key beside it there is
        // nothing to unwrap.
        let scene = parse_scene(
            br#"{"objects":[{"image":"models/a.json","effects":[{"file":"e.json",
                "passes":[{"constantshadervalues":{"value":0.75}}]}]}]}"#,
        )
        .unwrap();

        let pass = &scene.objects[0].effects[0].passes[0];
        assert_eq!(pass.constantshadervalues["value"], Value::from(0.75));
    }

    #[test]
    fn user_bindings_are_stripped_wherever_they_appear() {
        // The wrapper reaches keys no struct here names, and `user` is itself a
        // map when a layer is tied to one setting of a combo.
        let scene = parse_scene(
            br#"{"objects":[{
                "image":"models/a.json",
                "visible":{"user":{"name":"clock","condition":"1"},"value":true},
                "pointsize":{"user":"size","value":30.0},
                "effects":[{"file":"effects/blur/effect.json","passes":[
                    {"constantshadervalues":{"scale":{"user":"blur","value":"0.4 0.4"}}}
                ]}]
            }]}"#,
        )
        .unwrap();

        let object = &scene.objects[0];
        assert!(object.visible);
        assert_eq!(
            object.effects[0].passes[0].constantshadervalues["scale"],
            Value::String("0.4 0.4".to_string())
        );
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
    fn an_effect_pass_may_be_a_command_rather_than_a_material() {
        // motionblur's middle pass copies one render target to another; a
        // definition that fails to parse takes the layer's whole chain with it.
        let definition: EffectDefinition = serde_json::from_str(
            r#"{"passes":[
                {"material":"materials/effects/motionblur_accumulation.json","target":"_rt_FullCompoBuffer2"},
                {"command":"copy","target":"_rt_FullCompoBuffer1","source":"_rt_FullCompoBuffer2"},
                {"material":"materials/effects/motionblur_combine.json"}
            ]}"#,
        )
        .unwrap();

        assert_eq!(definition.passes.len(), 3);
        assert!(definition.passes[1].material.is_none());
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
    fn blend_mode_comes_from_the_first_pass() {
        let additive: Material =
            serde_json::from_str(r#"{"passes":[{"blending":"additive","textures":["glow"]}]}"#).unwrap();
        assert_eq!(base_blend(&additive), Blend::Add);

        // `translucent`, `normal` and an absent string all mean plain "over".
        for value in ["translucent", "normal", ""] {
            let material: Material =
                serde_json::from_str(&format!(r#"{{"passes":[{{"blending":{value:?}}}]}}"#)).unwrap();
            assert_eq!(base_blend(&material), Blend::Over, "for {value:?}");
        }

        let no_pass: Material = serde_json::from_str(r#"{"passes":[]}"#).unwrap();
        assert_eq!(base_blend(&no_pass), Blend::Over);
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

//! Running a scene's post-process effect chain on the GPU.
//!
//! Scene wallpapers of the common shape (plan.md §4.1) are one fullscreen
//! base image with a chain of effect shader passes over it, each sampling the
//! previous result as `g_Texture0`. This flattens that chain for a scene
//! that fits the shape exactly — one visible image layer — and falls back to
//! the plain composite (with the omission still noted) for anything richer,
//! same as `compose::render` already did on its own.

use super::compose::{self, Composite};
use super::model::{self, Effect, EffectDefinition, Material, Object, Scene};
use crate::export::Resolution;
use crate::pkg::Archive;
use crate::render::{capture, gpu::Gpu, pass};
use crate::shader::annotations::Declarations;
use crate::shader::bind::UniformValue;
use crate::shader::{annotations, bind, preprocess, shim};
use anyhow::{Context, Result, bail};
use image::RgbaImage;
use serde_json::Value;
use std::collections::HashMap;

/// No transform: the fixed fullscreen quad already spans clip space, so the
/// vertex shader's `g_ModelViewProjectionMatrix * a_Position` is a no-op.
const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// One WE-builtin utility texture the shim's shader annotations reference by
/// name but which ships with the engine, not with any wallpaper.
///
/// Only the flat ones are covered; `util/noise` has no synthesized equivalent
/// yet, so a shader defaulting to it degrades to flat black rather than a
/// noise-driven wobble — a documented gap, not a silent one.
fn builtin_solid(name: &str) -> Option<[u8; 4]> {
    match name {
        "util/noflow" => Some([127, 127, 0, 255]),
        "util/black" => Some([0, 0, 0, 255]),
        "util/white" => Some([255, 255, 255, 255]),
        _ => None,
    }
}

/// The scene's effect chain, if it has the shape plan.md §4.1 describes:
/// exactly one visible image layer carrying at least one visible effect.
/// `None` for anything richer (multiple image layers, particles, or an image
/// with no effects) — the caller falls back to the plain composite.
pub fn effect_chain_shape(scene: &Scene) -> Option<(&Object, Vec<&Effect>)> {
    let image_objects: Vec<&Object> = scene
        .objects
        .iter()
        .filter(|object| object.visible && model::is_image(object))
        .collect();
    let [object] = image_objects[..] else {
        return None;
    };
    let effects: Vec<&Effect> = model::visible_effects(object).collect();
    if effects.is_empty() {
        return None;
    }
    Some((object, effects))
}

/// Render one frame of a scene at `time` seconds.
///
/// Runs the GPU effect chain when the scene is exactly one visible image
/// layer carrying effects — the shape the chain in plan.md §4.1 describes —
/// and otherwise returns the plain composite unchanged, same as before this
/// existed.
pub fn render_frame(archive: &mut Archive, scene: &Scene, resolution: Option<Resolution>, time: f64) -> Result<Composite> {
    let mut composite = compose::render(archive, scene, resolution)?;

    let Some((object, effects)) = effect_chain_shape(scene) else {
        return Ok(composite);
    };

    let gpu = Gpu::new().context("opening a headless GL context")?;
    let headers = shim::headers();
    #[expect(clippy::cast_possible_truncation, reason = "a wallpaper's timestamp is always a few seconds at most")]
    let time = time as f32;
    let chain = prepare_effect_chain(&gpu.gl, archive, &effects, &composite.image, &headers)
        .with_context(|| format!("preparing the effect chain on {:?}", object.name))?;
    let target = chain
        .render(&gpu.gl, time)
        .with_context(|| format!("running the effect chain on {:?}", object.name))?;
    composite.image = capture::read_rgba(&gpu.gl, target.framebuffer, target.width, target.height)?;
    composite.omissions.retain(|note| !note.ends_with("effect(s) not applied"));
    Ok(composite)
}

/// One compiled effect pass: everything about it is fixed once prepared —
/// program, textures (including its `g_Texture0` input, the previous pass's
/// stable output handle) and every uniform except `g_Time` — so redrawing it
/// for a new frame only has to swap that one value in.
struct CompiledPass {
    program: pass::Program,
    target: pass::Target,
    textures: Vec<(String, glow::Texture)>,
    /// Every non-sampler uniform except `g_Time`, resolved once.
    floats: Vec<(String, Vec<f32>)>,
    ints: Vec<(String, i32)>,
    label: String,
}

/// A scene's effect chain, compiled and ready to redraw at any `g_Time`
/// without touching the archive, the shader preprocessor, or the GL compiler
/// again — the point of splitting this out of the old one-shot renderer is
/// that a live simulator redraws every frame but only `g_Time` ever changes.
pub struct EffectChain {
    quad: pass::Quad,
    passes: Vec<CompiledPass>,
}

impl EffectChain {
    /// Redraw every pass in order with `g_Time` set to `time`, and return the
    /// final pass's target (its texture is what a caller displays or reads
    /// back).
    pub fn render(&self, gl: &glow::Context, time: f32) -> Result<&pass::Target> {
        for pass in &self.passes {
            let mut floats = pass.floats.clone();
            floats.push(("g_Time".to_string(), vec![time]));

            let texture_refs: Vec<(&str, glow::Texture)> =
                pass.textures.iter().map(|(name, texture)| (name.as_str(), *texture)).collect();
            let float_refs: Vec<(&str, &[f32])> = floats.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
            let int_refs: Vec<(&str, i32)> = pass.ints.iter().map(|(n, v)| (n.as_str(), *v)).collect();

            pass::draw(
                gl,
                &pass::DrawCall {
                    program: &pass.program,
                    quad: &self.quad,
                    target: &pass.target,
                    textures: &texture_refs,
                    floats: &float_refs,
                    ints: &int_refs,
                    mvp: &IDENTITY,
                },
            )
            .with_context(|| format!("drawing {}", pass.label))?;
        }
        self.passes.last().map(|pass| &pass.target).context("at least one pass must have run")
    }
}

/// Compile every visible effect's every pass over `base`, in order, without
/// drawing a frame yet.
pub fn prepare_effect_chain(
    gl: &glow::Context,
    archive: &mut Archive,
    effects: &[&Effect],
    base: &RgbaImage,
    headers: &HashMap<String, String>,
) -> Result<EffectChain> {
    let (width, height) = (base.width(), base.height());
    let quad = pass::build_quad(gl)?;

    let mut current = pass::upload_texture(gl, base)?;
    let mut current_size = (width, height);
    let mut passes = Vec::new();

    for effect in effects {
        let definition: EffectDefinition = serde_json::from_slice(&archive.read(&effect.file)?)
            .with_context(|| format!("parsing {}", effect.file))?;

        for (definition_pass, effect_pass) in definition.passes.iter().zip(&effect.passes) {
            let material: Material = serde_json::from_slice(&archive.read(&definition_pass.material)?)
                .with_context(|| format!("parsing {}", definition_pass.material))?;

            for material_pass in &material.passes {
                let stem = format!("shaders/{}", material_pass.shader);
                let vertex_source = String::from_utf8(archive.read(&format!("{stem}.vert"))?)
                    .with_context(|| format!("{stem}.vert is not valid UTF-8"))?;
                let fragment_source = String::from_utf8(archive.read(&format!("{stem}.frag"))?)
                    .with_context(|| format!("{stem}.frag is not valid UTF-8"))?;

                let vertex_declarations = annotations::parse(&vertex_source);
                let fragment_declarations = annotations::parse(&fragment_source);

                let mut combos = bind::texture_combos(&fragment_declarations, &effect_pass.textures);
                combos.extend(bind::texture_combos(&vertex_declarations, &effect_pass.textures));

                let vertex_glsl = preprocess::build(&vertex_source, preprocess::Stage::Vertex, headers, &combos)
                    .with_context(|| format!("preprocessing {stem}.vert"))?;
                let fragment_glsl = preprocess::build(&fragment_source, preprocess::Stage::Fragment, headers, &combos)
                    .with_context(|| format!("preprocessing {stem}.frag"))?;
                let program = pass::compile_program(gl, &vertex_glsl, &fragment_glsl)
                    .with_context(|| format!("compiling {stem}"))?;

                let (textures, resolutions) =
                    resolve_textures(gl, archive, &fragment_declarations, &effect_pass.textures, current, current_size)?;

                let mut floats = uniform_floats(&vertex_declarations, &effect_pass.constantshadervalues);
                floats.extend(uniform_floats(&fragment_declarations, &effect_pass.constantshadervalues));
                floats.extend(resolutions);

                let ints = uniform_ints(&vertex_declarations, &effect_pass.constantshadervalues)
                    .into_iter()
                    .chain(uniform_ints(&fragment_declarations, &effect_pass.constantshadervalues))
                    .collect::<Vec<_>>();

                let target = pass::Target::new(gl, width, height)
                    .with_context(|| format!("allocating a render target for {stem}"))?;

                current = target.texture;
                current_size = (width, height);
                passes.push(CompiledPass { program, target, textures, floats, ints, label: stem });
            }
        }
    }

    if passes.is_empty() {
        bail!("at least one pass must have run");
    }
    Ok(EffectChain { quad, passes })
}

/// Bind every sampler the fragment shader declares: slot 0 is always the
/// chain's running result, higher slots come from the pass's positional
/// `textures[]` or the sampler's own annotation default.
///
/// Returns the bound textures alongside the `g_TextureNResolution` uniform
/// each one implies — `(width, height, width, height)`, since nothing here
/// packs textures into an atlas, the one case Wallpaper Engine uses the two
/// halves of that vector to tell apart.
/// Bound textures, and the `g_TextureNResolution` floats they imply.
type BoundTextures = (Vec<(String, glow::Texture)>, Vec<(String, Vec<f32>)>);

fn resolve_textures(
    gl: &glow::Context,
    archive: &mut Archive,
    declarations: &Declarations,
    textures: &[Option<String>],
    current: glow::Texture,
    current_size: (u32, u32),
) -> Result<BoundTextures> {
    let mut bound = Vec::new();
    let mut resolutions = Vec::new();

    for uniform in declarations.uniforms.iter().filter(|uniform| uniform.kind == "sampler2D") {
        let Some(slot) = uniform.name.strip_prefix("g_Texture").and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };

        let (texture, size) = if slot == 0 {
            (current, current_size)
        } else {
            let name = slot.checked_sub(1).and_then(|index| textures.get(index)).and_then(Option::as_deref);
            resolve_slot_texture(gl, archive, name, uniform.default.as_ref())?
        };

        bound.push((uniform.name.clone(), texture));
        #[expect(clippy::cast_precision_loss, reason = "texture dimensions, nowhere near f32's 2^24 exact range")]
        let (w, h) = (size.0 as f32, size.1 as f32);
        resolutions.push((format!("{}Resolution", uniform.name), vec![w, h, w, h]));
    }

    Ok((bound, resolutions))
}

/// Load and upload one texture slot: the scene's own mask/map texture when
/// the pass names one, or a synthesized stand-in for the shader's default.
fn resolve_slot_texture(
    gl: &glow::Context,
    archive: &mut Archive,
    name: Option<&str>,
    default: Option<&Value>,
) -> Result<(glow::Texture, (u32, u32))> {
    if let Some(name) = name {
        let texture_path = format!("materials/{name}.tex");
        let bytes = archive.read(&texture_path).with_context(|| format!("reading {texture_path}"))?;
        let tex = crate::tex::parse_bytes(&bytes).with_context(|| format!("parsing {texture_path}"))?;
        let mipmap = crate::tex::largest_mipmap(&tex)?;
        let decoded = crate::tex::decode_rgba(&tex, mipmap).with_context(|| format!("decoding {texture_path}"))?;
        let size = (decoded.width(), decoded.height());
        return Ok((pass::upload_texture(gl, &decoded)?, size));
    }

    let builtin = default.and_then(Value::as_str).and_then(builtin_solid).unwrap_or([0, 0, 0, 255]);
    Ok((pass::solid_texture(gl, builtin)?, (1, 1)))
}

/// Every non-sampler uniform's value, ready for `pass::DrawCall.floats`.
fn uniform_floats(declarations: &Declarations, material: &serde_json::Map<String, Value>) -> Vec<(String, Vec<f32>)> {
    bind::resolve(declarations, material)
        .into_iter()
        .filter_map(|(name, value)| value.as_floats().map(|floats| (name, floats)))
        .collect()
}

fn uniform_ints(declarations: &Declarations, material: &serde_json::Map<String, Value>) -> Vec<(String, i32)> {
    bind::resolve(declarations, material)
        .into_iter()
        .filter_map(|(name, value)| match value {
            UniformValue::Int(int) => Some((name, int)),
            _ => None,
        })
        .collect()
}

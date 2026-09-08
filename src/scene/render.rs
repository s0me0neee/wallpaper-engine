//! Running a scene's post-process effect chains on the GPU.
//!
//! Scene wallpapers of the common shape (plan.md §4.1) are an image layer with
//! a chain of effect shader passes over it, each sampling the previous result
//! as `g_Texture0`. A scene may have several such layers, each with its own
//! chain: `render_frame` prepares every visible image layer (`compose::prepare`),
//! runs that layer's own chain over its texture, then alpha-composites the
//! processed layers together in order. Layers without effects pass straight
//! through. Particles, keyframe animation and rotation are still left to the
//! plain composite and reported as omissions (plan.md §4.7).

use super::compose::{self, Composite};
use super::model::{self, Effect, EffectDefinition, Material, Object, Scene};
use crate::export::Resolution;
use crate::pkg::Archive;
use crate::render::{capture, gpu::Gpu, pass};
use crate::shader::annotations::Declarations;
use crate::shader::bind::UniformValue;
use crate::shader::{annotations, bind, preprocess, shim};
use anyhow::{Context, Result, bail};
use image::{RgbaImage, imageops};
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
/// Every visible image layer that carries effects gets its own chain run over
/// its texture before the layers are flattened together; layers without
/// effects are composited as-is. A scene with no effects anywhere comes back
/// as the plain composite, untouched.
///
/// One layer's chain failing (an effect helper we don't implement, a missing
/// engine-builtin texture) doesn't sink the whole frame: that layer is
/// composited unprocessed and the reason is recorded as an omission, the same
/// way the plain composite reports what it left out.
pub fn render_frame(archive: &mut Archive, scene: &Scene, resolution: Option<Resolution>, time: f64) -> Result<Composite> {
    #[expect(clippy::cast_possible_truncation, reason = "a wallpaper's timestamp is always a few seconds at most")]
    let time = time as f32;
    let mut layered = compose::prepare(archive, scene, resolution, time)?;

    let effected: Vec<usize> = layered
        .layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| model::visible_effects(layer.object).next().is_some())
        .map(|(index, _)| index)
        .collect();

    if !effected.is_empty() {
        let gpu = Gpu::new().context("opening a headless GL context")?;
        let headers = shim::headers();

        // The per-object "N effect(s) not applied" notes are about to become
        // wrong for every layer whose chain runs; drop them and let each
        // failed layer below re-add a note of its own.
        layered.omissions.retain(|note| !note.ends_with("effect(s) not applied"));

        for index in effected {
            let name = model::label(layered.layers[index].object);
            match run_layer_chain(&gpu, archive, &layered.layers[index], &headers, time) {
                Ok(processed) => layered.layers[index].image = processed,
                Err(error) => layered.omissions.push(format!("{name}: effect chain skipped ({error:#})")),
            }
        }
    }

    let mut output = RgbaImage::from_pixel(layered.width, layered.height, layered.background);
    for layer in &layered.layers {
        imageops::overlay(&mut output, &layer.image, layer.left, layer.top);
    }

    Ok(Composite { image: output, omissions: layered.omissions })
}

/// Compile and run one layer's effect chain over its own texture, returning
/// the processed pixels.
fn run_layer_chain(
    gpu: &Gpu,
    archive: &mut Archive,
    layer: &compose::PreparedLayer,
    headers: &HashMap<String, String>,
    time: f32,
) -> Result<RgbaImage> {
    let effects: Vec<&Effect> = model::visible_effects(layer.object).collect();
    let chain = prepare_effect_chain(&gpu.gl, archive, &effects, &layer.image, headers)
        .context("preparing the effect chain")?;
    let target = chain.render(&gpu.gl, time, &[]).context("running the effect chain")?;
    capture::read_rgba(&gpu.gl, target.framebuffer, target.width, target.height)
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

/// One shader parameter Wallpaper Engine's own Properties panel would show as
/// a slider — any scalar `float` uniform whose annotation carries a `range`,
/// e.g. `foliagesway`'s `g_Strength`. `default` is the wallpaper's own preset
/// value (`scene.json`'s `constantshadervalues`, or the shader's own default
/// if the scene doesn't override it) — the value a live override replaces.
pub struct Tweakable {
    pub label: String,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pass_index: usize,
    uniform_name: String,
}

/// Every range-annotated scalar float `declarations` names, read back out of
/// `floats` (already resolved against the wallpaper's own preset) rather than
/// the shader's own default.
fn collect_tweakables(declarations: &Declarations, floats: &[(String, Vec<f32>)], pass_index: usize) -> Vec<Tweakable> {
    declarations
        .uniforms
        .iter()
        .filter(|uniform| uniform.kind == "float")
        .filter_map(|uniform| {
            let (min, max) = uniform.range?;
            let &value = floats.iter().find(|(name, _)| *name == uniform.name)?.1.first()?;
            Some(Tweakable {
                label: uniform.material.clone().unwrap_or_else(|| uniform.name.clone()),
                min,
                max,
                default: value,
                pass_index,
                uniform_name: uniform.name.clone(),
            })
        })
        .collect()
}

/// A scene's effect chain, compiled and ready to redraw at any `g_Time`
/// without touching the archive, the shader preprocessor, or the GL compiler
/// again — the point of splitting this out of the old one-shot renderer is
/// that a live simulator redraws every frame but only `g_Time` ever changes.
pub struct EffectChain {
    quad: pass::Quad,
    passes: Vec<CompiledPass>,
    pub tweakables: Vec<Tweakable>,
}

impl EffectChain {
    /// Redraw every pass in order with `g_Time` set to `time`, and return the
    /// final pass's target (its texture is what a caller displays or reads
    /// back).
    ///
    /// `overrides` gives a live value for each of `self.tweakables`, in the
    /// same order — pass `&[]` to just use the wallpaper's own presets, as
    /// export does.
    pub fn render(&self, gl: &glow::Context, time: f32, overrides: &[f32]) -> Result<&pass::Target> {
        for (pass_index, pass) in self.passes.iter().enumerate() {
            let mut floats = pass.floats.clone();
            floats.push(("g_Time".to_string(), vec![time]));
            for (tweakable, &value) in self.tweakables.iter().zip(overrides) {
                if tweakable.pass_index == pass_index
                    && let Some(entry) = floats.iter_mut().find(|(name, _)| *name == tweakable.uniform_name)
                {
                    entry.1 = vec![value];
                }
            }

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
    let mut tweakables = Vec::new();

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

                tweakables.extend(collect_tweakables(&vertex_declarations, &floats, passes.len()));
                tweakables.extend(collect_tweakables(&fragment_declarations, &floats, passes.len()));

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
    Ok(EffectChain { quad, passes, tweakables })
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
        // A pass may positionally name an engine builtin (`util/white`,
        // `util/noflow`, ...); those ship with Wallpaper Engine, not the
        // package, so synthesize the flat ones rather than reading them.
        if let Some(solid) = builtin_solid(name) {
            return Ok((pass::solid_texture(gl, solid)?, (1, 1)));
        }

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

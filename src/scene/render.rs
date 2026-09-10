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
use super::model::{self, Effect, EffectBind, EffectDefinition, Material, Scene};
use crate::export::Resolution;
use crate::pkg::Archive;
use crate::render::{capture, gpu::Gpu, pass};
use crate::shader::annotations::Declarations;
use crate::shader::bind::UniformValue;
use crate::shader::{annotations, bind, preprocess, shim};
use anyhow::{Context, Result, bail};
use image::RgbaImage;
use noise::{NoiseFn, Perlin};
use serde_json::Value;
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::OnceLock;

/// No transform: the fixed fullscreen quad already spans clip space, so the
/// vertex shader's `g_ModelViewProjectionMatrix * a_Position` is a no-op.
const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// One WE-builtin utility texture the shim's shader annotations reference by
/// name but which ships with the engine, not with any wallpaper. Only the flat
/// ones; the noise builtins are `builtin_noise`.
fn builtin_solid(name: &str) -> Option<[u8; 4]> {
    match name {
        "util/noflow" => Some([127, 127, 0, 255]),
        "util/black" => Some([0, 0, 0, 255]),
        "util/white" => Some([255, 255, 255, 255]),
        _ => None,
    }
}

/// The synthesized stand-in for WE's builtin noise textures (`util/noise`,
/// `util/clouds_256`), which also ship with the engine, not the wallpaper.
/// Without it a shader that samples one for a wobble or dither — `pulse`,
/// `godrays_downsample2`, `foliagesway` — degrades to flat black.
///
/// 256x256 RGBA, four octaves of value noise per channel, built once. Tiling
/// is exact: each octave samples 4D Perlin (`noise` has no seamless 2D
/// generator) on a torus embedding of the UV square at integer angular
/// frequencies, so the wrap is continuous by construction — which matters
/// because these are sampled `REPEAT` well outside `[0, 1]`.
fn builtin_noise(name: &str) -> Option<RgbaImage> {
    static TEXTURE: OnceLock<RgbaImage> = OnceLock::new();
    matches!(name, "util/noise" | "util/clouds_256" | "util/clouds")
        .then(|| TEXTURE.get_or_init(build_noise_texture).clone())
}

fn build_noise_texture() -> RgbaImage {
    const SIZE: u32 = 256;
    let channels = [Perlin::new(1), Perlin::new(2), Perlin::new(3), Perlin::new(4)];

    RgbaImage::from_fn(SIZE, SIZE, |x, y| {
        let u = f64::from(x) / f64::from(SIZE) * TAU;
        let v = f64::from(y) / f64::from(SIZE) * TAU;
        let mut pixel = [0u8; 4];
        for (slot, perlin) in channels.iter().enumerate() {
            let level = noise_sample(perlin, u, v);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to 0.0..=1.0, then scaled by 255"
            )]
            let byte = (level * 255.0).round() as u8;
            pixel[slot] = byte;
        }
        image::Rgba(pixel)
    })
}

/// One channel's value at UV angles `(u, v)` on the torus: four octaves of
/// Perlin summed and remapped to `[0, 1]`. Periodic in `u` and `v` with
/// period `TAU`, since each octave uses an integer angular frequency.
fn noise_sample(perlin: &Perlin, u: f64, v: f64) -> f64 {
    const OCTAVES: u32 = 4;
    let (mut sum, mut norm, mut amplitude, mut frequency) = (0.0, 0.0, 1.0, 1.0_f64);
    for _ in 0..OCTAVES {
        sum += amplitude
            * perlin.get([
                (frequency * u).cos(),
                (frequency * u).sin(),
                (frequency * v).cos(),
                (frequency * v).sin(),
            ]);
        norm += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    ((sum / norm) * 0.5 + 0.5).clamp(0.0, 1.0)
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
        .filter(|(_, layer)| !layer.composition && model::visible_effects(layer.object).next().is_some())
        .map(|(index, _)| index)
        .collect();

    // A composition layer's input is the frame beneath it, which this path
    // only has as CPU pixels after the flatten — too late to run a chain over.
    // The live simulator, which keeps the frame on the GPU, does draw them.
    for layer in layered.layers.iter().filter(|layer| layer.composition) {
        layered
            .omissions
            .push(format!("{}: composition layer drawn only by the live simulator", model::label(layer.object)));
    }

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
                Ok((processed, skipped)) => {
                    layered.layers[index].image = processed;
                    layered.omissions.extend(skipped.into_iter().map(|note| format!("{name}: {note}")));
                }
                Err(error) => layered.omissions.push(format!("{name}: effect chain skipped ({error:#})")),
            }
        }
    }

    Ok(Composite { image: compose::flatten(&layered), omissions: layered.omissions })
}

/// Compile and run one layer's effect chain over its own texture, returning
/// the processed pixels and a note for each effect the chain left out.
fn run_layer_chain(
    gpu: &Gpu,
    archive: &mut Archive,
    layer: &compose::PreparedLayer,
    headers: &HashMap<String, String>,
    time: f32,
) -> Result<(RgbaImage, Vec<String>)> {
    let effects: Vec<&Effect> = model::visible_effects(layer.object).collect();
    let chain = prepare_effect_chain(&gpu.gl, archive, &effects, &layer.image, headers, pass::Format::Ldr)
        .context("preparing the effect chain")?;
    let target = chain.render(&gpu.gl, time, &[]).context("running the effect chain")?;
    let image = capture::read_rgba(&gpu.gl, target.framebuffer, target.width, target.height)?;
    Ok((image, chain.skipped))
}

/// One compiled effect pass: everything about it is fixed once prepared —
/// program, textures (including its `g_Texture0` input, the previous pass's
/// stable output handle) and every uniform except `g_Time` — so redrawing it
/// for a new frame only has to swap that one value in.
struct CompiledPass {
    program: pass::Program,
    target: pass::Target,
    textures: Vec<(String, PassTexture)>,
    /// Every non-sampler uniform except `g_Time`, resolved once.
    floats: Vec<(String, Vec<f32>)>,
    ints: Vec<(String, i32)>,
    label: String,
}

/// What a pass binds to one sampler slot.
#[derive(Clone, Copy)]
enum PassTexture {
    /// Resolved once at prepare time and never changes.
    Fixed(glow::Texture),
    /// The layer image that entered the chain. Kept symbolic rather than
    /// resolved, because a puppet or particle layer re-uploads that image
    /// every frame and the binding has to follow it.
    LayerBase,
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
    /// The layer image uploaded at prepare time, for `PassTexture::LayerBase`
    /// when the caller does not re-feed one.
    base: glow::Texture,
    pub tweakables: Vec<Tweakable>,
    /// Effects left out of the chain because this renderer cannot run them, one
    /// note each — the caller folds these into its omissions.
    pub skipped: Vec<String>,
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
        self.render_over(gl, None, time, overrides)
    }

    /// As `render`, but the first pass reads `base` as `g_Texture0` instead of
    /// the texture baked in at prepare time. A puppet-warp layer's image
    /// changes every frame, so its chain's input has to be re-fed each frame
    /// rather than uploaded once.
    pub fn render_over(
        &self,
        gl: &glow::Context,
        base: Option<glow::Texture>,
        time: f32,
        overrides: &[f32],
    ) -> Result<&pass::Target> {
        for (pass_index, pass) in self.passes.iter().enumerate() {
            let mut floats = pass.floats.clone();
            floats.push(("g_Time".to_string(), vec![time]));
            // The cursor, normalised over the canvas. Shaders that use it
            // declare it themselves with no annotation default, so nothing
            // binds it and GL leaves it at (0, 0) — the corner — which throws
            // anything positioned by it a full screen out.
            // `scene_example8`'s `lens_flare_sun` puts its sun there.
            // We do not track a real pointer yet (plan.md §14); the centre is
            // the neutral stand-in, and is where a flare belongs by default.
            floats.push(("g_PointerPosition".to_string(), vec![0.5, 0.5]));
            for (tweakable, &value) in self.tweakables.iter().zip(overrides) {
                if tweakable.pass_index == pass_index
                    && let Some(entry) = floats.iter_mut().find(|(name, _)| *name == tweakable.uniform_name)
                {
                    entry.1 = vec![value];
                }
            }

            let texture_refs: Vec<(&str, glow::Texture)> = pass
                .textures
                .iter()
                .map(|(name, texture)| {
                    let texture = match *texture {
                        PassTexture::LayerBase => base.unwrap_or(self.base),
                        PassTexture::Fixed(handle) => match (pass_index, base) {
                            (0, Some(base)) if name == "g_Texture0" => base,
                            _ => handle,
                        },
                    };
                    (name.as_str(), texture)
                })
                .collect();
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
    format: pass::Format,
) -> Result<EffectChain> {
    let (width, height) = (base.width(), base.height());
    let quad = pass::build_quad(gl)?;

    let base_texture = pass::upload_texture(gl, base)?;
    let mut current = base_texture;
    let mut current_size = (width, height);
    let mut passes = Vec::new();
    let mut tweakables = Vec::new();
    let mut skipped = Vec::new();

    for effect in effects {
        let definition: EffectDefinition = serde_json::from_slice(&archive.read(&effect.file)?)
            .with_context(|| format!("parsing {}", effect.file))?;

        // A definition pass with no material is a render-target command, and
        // the targets it moves between are frame history this renderer does not
        // keep (motionblur's `copy` between its two accumulation buffers). Half
        // the effect is not the effect, so drop the whole one and say so.
        if definition.passes.iter().any(|pass| pass.material.is_none()) {
            skipped.push(format!("{} needs render-target commands", effect.file));
            continue;
        }

        // `previous` in a bind list is the image this effect was handed, which
        // is where the running chain stands before its first pass — not the
        // pass that ran just before, which is what slot 0 falls back to.
        let effect_input = (current, current_size);
        let mut named: HashMap<String, (glow::Texture, (u32, u32))> = HashMap::new();

        for (definition_pass, effect_pass) in definition.passes.iter().zip(&effect.passes) {
            let Some(material_path) = definition_pass.material.as_deref() else {
                continue;
            };
            let (target_width, target_height) =
                target_size(&definition, definition_pass.target.as_deref(), width, height);
            let material: Material = serde_json::from_slice(&archive.read(material_path)?)
                .with_context(|| format!("parsing {material_path}"))?;

            for material_pass in &material.passes {
                let stem = format!("shaders/{}", material_pass.shader);
                let vertex_source = String::from_utf8(archive.read(&format!("{stem}.vert"))?)
                    .with_context(|| format!("{stem}.vert is not valid UTF-8"))?;
                let fragment_source = String::from_utf8(archive.read(&format!("{stem}.frag"))?)
                    .with_context(|| format!("{stem}.frag is not valid UTF-8"))?;

                let vertex_declarations = annotations::parse(&vertex_source);
                let fragment_declarations = annotations::parse(&fragment_source);

                // A combo's `[COMBO]` default is usually declared in one stage
                // (typically the fragment) but governs both. `godrays_downsample2`
                // keeps `v_NoiseTexCoord` under `#if NOISE == 1` in the vertex
                // shader too, yet only its fragment carries the `NOISE default:1`
                // line — so without merging the pair's defaults the vertex stage
                // defaults NOISE to 0, never writes the varying, and the program
                // fails to link. Feed the union of both stages' defaults to each.
                let mut combos = annotations::combo_defaults(&vertex_declarations);
                combos.extend(annotations::combo_defaults(&fragment_declarations));
                combos.extend(bind::texture_combos(&fragment_declarations, &effect_pass.textures));
                combos.extend(bind::texture_combos(&vertex_declarations, &effect_pass.textures));
                // The wallpaper's own choices win over both.
                combos.extend(
                    effect_pass
                        .combos
                        .iter()
                        .filter_map(|(name, value)| Some((name.clone(), value.as_i64()?))),
                );

                let vertex_glsl = preprocess::build(&vertex_source, preprocess::Stage::Vertex, headers, &combos)
                    .with_context(|| format!("preprocessing {stem}.vert"))?;
                let fragment_glsl = preprocess::build_against(
                    &fragment_source,
                    preprocess::Stage::Fragment,
                    headers,
                    &combos,
                    Some(&vertex_source),
                )
                    .with_context(|| format!("preprocessing {stem}.frag"))?;
                let program = pass::compile_program(gl, &vertex_glsl, &fragment_glsl)
                    .with_context(|| format!("compiling {stem}"))?;

                let (textures, resolutions) = resolve_textures(
                    gl,
                    archive,
                    &fragment_declarations,
                    &effect_pass.textures,
                    &bound_slots(&definition_pass.bind, effect_input, &named),
                    (current, current_size),
                    (width, height),
                )?;

                let mut floats = uniform_floats(&vertex_declarations, &effect_pass.constantshadervalues);
                floats.extend(uniform_floats(&fragment_declarations, &effect_pass.constantshadervalues));

                tweakables.extend(collect_tweakables(&vertex_declarations, &floats, passes.len()));
                tweakables.extend(collect_tweakables(&fragment_declarations, &floats, passes.len()));

                floats.extend(resolutions);

                let ints = uniform_ints(&vertex_declarations, &effect_pass.constantshadervalues)
                    .into_iter()
                    .chain(uniform_ints(&fragment_declarations, &effect_pass.constantshadervalues))
                    .collect::<Vec<_>>();

                let target = pass::Target::with_format(gl, target_width, target_height, format)
                    .with_context(|| format!("allocating a render target for {stem}"))?;

                current = target.texture;
                current_size = (target_width, target_height);
                if let Some(name) = &definition_pass.target {
                    named.insert(name.clone(), (target.texture, current_size));
                }
                passes.push(CompiledPass { program, target, textures, floats, ints, label: stem });
            }
        }
    }

    if passes.is_empty() {
        if skipped.is_empty() {
            bail!("at least one pass must have run");
        }
        bail!("{}", skipped.join("; "));
    }
    Ok(EffectChain { quad, passes, base: base_texture, tweakables, skipped })
}

/// The pixel size a pass renders at: its `fbos` entry's scale divides the
/// layer, and a pass with no named target draws at full size.
fn target_size(definition: &EffectDefinition, target: Option<&str>, width: u32, height: u32) -> (u32, u32) {
    let scale = target
        .and_then(|name| definition.fbos.iter().find(|fbo| fbo.name == name))
        .map_or(1.0, |fbo| fbo.scale);
    if !scale.is_finite() || scale <= 1.0 {
        return (width, height);
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "canvas dimensions divided by a scale of 2 or 4, floored at 1"
    )]
    let divide = |size: u32| ((size as f32 / scale).round() as u32).max(1);
    (divide(width), divide(height))
}

/// Resolve a pass's `bind` list into sampler slots. `previous` is the image the
/// effect was handed; every other name is one of the effect's own render
/// targets, which an earlier pass in the same effect has already written.
fn bound_slots(
    binds: &[EffectBind],
    effect_input: (glow::Texture, (u32, u32)),
    named: &HashMap<String, (glow::Texture, (u32, u32))>,
) -> HashMap<usize, (glow::Texture, (u32, u32))> {
    binds
        .iter()
        .filter_map(|bind| {
            let source = if bind.name == "previous" {
                effect_input
            } else {
                *named.get(&bind.name)?
            };
            Some((bind.index, source))
        })
        .collect()
}

/// Bind every sampler the fragment shader declares: the pass's own `bind` list
/// first, then slot 0 as the chain's running result, then the pass's
/// `textures[]` — indexed by slot number — or the sampler's annotation default.
///
/// Returns the bound textures alongside the `g_TextureNResolution` uniform
/// each one implies — `(width, height, width, height)`, since nothing here
/// packs textures into an atlas, the one case Wallpaper Engine uses the two
/// halves of that vector to tell apart.
/// Bound textures, and the `g_TextureNResolution` floats they imply.
type BoundTextures = (Vec<(String, PassTexture)>, Vec<(String, Vec<f32>)>);

fn resolve_textures(
    gl: &glow::Context,
    archive: &mut Archive,
    declarations: &Declarations,
    textures: &[Option<String>],
    bound_slots: &HashMap<usize, (glow::Texture, (u32, u32))>,
    current: (glow::Texture, (u32, u32)),
    layer_size: (u32, u32),
) -> Result<BoundTextures> {
    let mut bound = Vec::new();
    let mut resolutions = Vec::new();

    for uniform in declarations.uniforms.iter().filter(|uniform| uniform.kind == "sampler2D") {
        let Some(slot) = uniform.name.strip_prefix("g_Texture").and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };

        let name = textures.get(slot).and_then(Option::as_deref);
        let (texture, size) = if let Some(&(texture, size)) = bound_slots.get(&slot) {
            (PassTexture::Fixed(texture), size)
        } else if slot == 0 {
            (PassTexture::Fixed(current.0), current.1)
        } else if name.is_some_and(is_layer_composite_target) {
            (PassTexture::LayerBase, layer_size)
        } else {
            let (texture, size) =
                resolve_slot_texture(gl, archive, name.filter(|name| !is_render_target(name)), uniform.default.as_ref())?;
            (PassTexture::Fixed(texture), size)
        };

        bound.push((uniform.name.clone(), texture));
        #[expect(clippy::cast_precision_loss, reason = "texture dimensions, nowhere near f32's 2^24 exact range")]
        let (w, h) = (size.0 as f32, size.1 as f32);
        resolutions.push((format!("{}Resolution", uniform.name), vec![w, h, w, h]));
    }

    Ok((bound, resolutions))
}

/// A `_rt_*` name is one of Wallpaper Engine's runtime render targets, not a
/// file in the package — reading it as one is what used to drop a whole
/// layer's chain.
fn is_render_target(name: &str) -> bool {
    name.starts_with("_rt_")
}

/// `_rt_imageLayerComposite_<id>_<slot>` is the layer's own image as it
/// entered the effect chain. A `godrays`/`shine` chain's final `*_combine`
/// pass samples it as the albedo it blends its rays back over, so without it
/// the layer loses every effect it has.
fn is_layer_composite_target(name: &str) -> bool {
    name.starts_with("_rt_imageLayerComposite")
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
        // `util/noflow`, `util/noise`, ...); those ship with Wallpaper Engine,
        // not the package, so synthesize them rather than reading them.
        if let Some(solid) = builtin_solid(name) {
            return Ok((pass::solid_texture(gl, solid)?, (1, 1)));
        }
        if let Some(noise) = builtin_noise(name) {
            let size = (noise.width(), noise.height());
            return Ok((pass::upload_repeating_texture(gl, &noise)?, size));
        }

        let texture_path = format!("materials/{name}.tex");
        let bytes = archive.read(&texture_path).with_context(|| format!("reading {texture_path}"))?;
        let tex = crate::tex::parse_bytes(&bytes).with_context(|| format!("parsing {texture_path}"))?;
        let mipmap = crate::tex::largest_mipmap(&tex)?;
        let decoded = crate::tex::decode_rgba(&tex, mipmap).with_context(|| format!("decoding {texture_path}"))?;
        let size = (decoded.width(), decoded.height());
        return Ok((pass::upload_texture(gl, &decoded)?, size));
    }

    let default_name = default.and_then(Value::as_str);
    if let Some(noise) = default_name.and_then(builtin_noise) {
        let size = (noise.width(), noise.height());
        return Ok((pass::upload_repeating_texture(gl, &noise)?, size));
    }
    let builtin = default_name.and_then(builtin_solid).unwrap_or([0, 0, 0, 255]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_noise_matches_only_the_known_names() {
        assert!(builtin_noise("util/noise").is_some());
        assert!(builtin_noise("util/clouds_256").is_some());
        assert!(builtin_noise("util/noflow").is_none());
        assert!(builtin_noise("masks/pulse_mask").is_none());
    }

    #[test]
    fn the_noise_field_is_periodic_so_repeat_sampling_has_no_seam() {
        let perlin = Perlin::new(1);
        for step in 0..16 {
            let angle = f64::from(step) / 16.0 * TAU;
            // One full turn brings the torus sample back exactly.
            assert!((noise_sample(&perlin, angle, 0.3) - noise_sample(&perlin, angle + TAU, 0.3)).abs() < 1e-9);
            assert!((noise_sample(&perlin, 0.7, angle) - noise_sample(&perlin, 0.7, angle + TAU)).abs() < 1e-9);
        }
    }

    #[test]
    fn the_synthesized_noise_texture_is_not_flat() {
        let texture = build_noise_texture();
        let first = texture.get_pixel(0, 0).0[0];
        assert!(texture.pixels().any(|pixel| pixel.0[0] != first), "noise texture has no variation");
    }
}

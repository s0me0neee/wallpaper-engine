//! CPU particle systems.
//!
//! A WE particle object names a JSON *preset* (`particles/presets/*.json`):
//! an emitter, a list of initializers that roll each particle's birth state,
//! a list of operators that evolve it, a renderer, and optional child systems.
//! This module parses that, simulates it as a pure function of `time`, and
//! rasterizes the result into one RGBA layer the scene compositor drops in at
//! the object's z-order. Presets use a `genericparticle` material, but not all
//! of them additively — twenty of the corpus's fifty-one are `translucent`, and
//! each preset draws its own particles with its own blend.
//!
//! Determinism without frame-to-frame state: each particle's whole trajectory
//! is re-integrated from its birth on every call, and its random rolls come
//! from a PCG stream seeded off its global emission index. The sprite texture
//! (`particle/halo*`) ships with the engine, not the package, so it is
//! synthesized as a soft radial gradient.
//!
//! Exact fidelity to WE's own simulation is not recoverable — the drag law,
//! turbulence basis and several magic constants are undocumented — so the
//! tuning here aims for a plausible match in motion and density, with every
//! operator/initializer type in these presets contributing.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded particle counts, ages and pixel coordinates; see module docs"
)]

use super::model::{self, InstanceOverride, Material};
use crate::pkg::Archive;
use anyhow::{Context, Result};
use glam::{Vec2, Vec3};
use image::RgbaImage;
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::Deserialize;
use std::collections::HashMap;
use std::f32::consts::TAU;
use std::path::Path;
use super::sprite;
use tiny_skia::{
    BlendMode, FilterQuality, LineCap, Paint, PathBuilder, Pattern, Pixmap, Point,
    PremultipliedColorU8, Rect, SpreadMode, Stroke, Transform,
};

/// Resolve an initializer's optional bounds: an absent `max` is fixed at `min`,
/// and an absent `min` is the initializer's own neutral value.
/// Alpha's bounds, which unlike a size or a lifetime have a natural range.
///
/// An absent `min` is 0 and an absent `max` is 1 — the whole range — because
/// the corpus only makes sense read that way. Three presets write `{"max": 1}`
/// and one writes `{"max": 0.3}`; treating the absent `min` as 1.0 would make
/// the first a no-op and the second an inverted range, and `alpharandom` is
/// not something a scene serializes to do nothing. `{"min": 0.8}` still pins
/// both ends, which is the same "absent max is fixed at min" rule as
/// everywhere else.
fn alpha_bounds(min: Option<f32>, max: Option<f32>) -> (f32, f32) {
    (min.unwrap_or(0.0), max.or(min).unwrap_or(1.0))
}

fn bounds<T: Copy>(min: Option<T>, max: Option<T>, neutral: T) -> (T, T) {
    let low = min.unwrap_or(neutral);
    (low, max.or(min).unwrap_or(neutral))
}

/// Simulation step rate for trajectory integration.
const SIM_HZ: f32 = 60.0;
/// Default cap on integration steps for one particle: enough that a particle
/// living the full `MAX_STEPS / SIM_HZ` seconds still integrates at the true
/// 60 Hz step. The still exporter keeps this; the live simulator lowers it via
/// `Placement::max_sim_steps` so an old firefly does not cost 1000+ steps a
/// frame (once the cap bites, the step count is fixed and the path stays
/// frame-to-frame stable, just coarser).
const MAX_STEPS: u32 = 4096;
/// Recursion cap for child systems (presets in the wild nest one level).
const MAX_DEPTH: u32 = 2;
/// Turbulence-noise sample rate and the pixel speed one unit of noise implies.
const TURB_INIT_SPEED: f32 = 40.0;

// ---------------------------------------------------------------------------
// Preset schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct Preset {
    #[serde(default)]
    emitter: Vec<Emitter>,
    #[serde(default)]
    initializer: Vec<Initializer>,
    #[serde(default)]
    operator: Vec<Operator>,
    #[serde(default)]
    renderer: Vec<Renderer>,
    #[serde(default)]
    children: Option<Vec<Child>>,
    #[serde(default)]
    material: String,
    #[serde(default = "one")]
    maxcount: f32,
    #[serde(default)]
    starttime: f32,
    /// Blend the preset's own material asks for, resolved at collect time.
    ///
    /// Not every system is additive, and drawing a `translucent` one as if it
    /// were is what turns `scene_example6`'s rain into saturated white blobs:
    /// summing overlapping drops clips to white instead of letting the nearest
    /// one cover the rest. Twenty of the corpus's fifty-one particle materials
    /// are `translucent`.
    #[serde(skip, default = "additive")]
    blend: model::Blend,
    /// The sprite each of this preset's particles is drawn with, resolved at
    /// collect time from the material's first texture slot.
    #[serde(skip)]
    sprite: Option<Pixmap>,
}

fn additive() -> model::Blend {
    model::Blend::Add
}

#[derive(Debug, Deserialize)]
struct Child {
    name: String,
    #[serde(default)]
    r#type: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "name", rename_all = "lowercase")]
enum Emitter {
    /// `directions` scales the random unit direction per axis, so the emission
    /// sphere becomes an ellipse: twenty-five of the corpus's thirty-five
    /// sphere emitters set it, and `"1 0.2 0"` is a flat horizontal band that
    /// comes out as a circle if it is ignored.
    SphereRandom {
        #[serde(default)]
        origin: model::Vec3,
        #[serde(default)]
        distancemin: model::Vec3,
        #[serde(default = "vec3_one")]
        distancemax: model::Vec3,
        #[serde(default = "vec3_one")]
        directions: model::Vec3,
        #[serde(default = "one")]
        rate: f32,
    },
    BoxRandom {
        #[serde(default)]
        origin: model::Vec3,
        #[serde(default = "vec3_one")]
        distancemax: model::Vec3,
        #[serde(default = "vec3_one")]
        directions: model::Vec3,
        #[serde(default = "one")]
        rate: f32,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "name", rename_all = "lowercase")]
/// Every `*random` initializer's bounds are optional, because Wallpaper Engine
/// omits a field that still holds its default. Eight presets across
/// `scene_example3` and `scene_example4` write `colorrandom` as bare
/// `min: "255 255 255"` (`shootingstarglow` writes both halves of that same
/// white), and `scene_example8`'s `wet_snow1` writes `alpharandom` with neither
/// bound at all. An absent `max` means "fixed at `min`"; an absent `min` means
/// the neutral value for what it initialises — 1.0 for a lifetime, size or
/// alpha, white for a colour, still for a velocity. Reading either as zero
/// turns the particles black, or invisible, or dead on arrival.
enum Initializer {
    LifeTimeRandom {
        #[serde(default)]
        min: Option<f32>,
        #[serde(default)]
        max: Option<f32>,
        #[serde(default = "one")]
        exponent: f32,
    },
    SizeRandom {
        #[serde(default)]
        min: Option<f32>,
        #[serde(default)]
        max: Option<f32>,
        #[serde(default = "one")]
        exponent: f32,
    },
    AlphaRandom {
        #[serde(default)]
        min: Option<f32>,
        #[serde(default)]
        max: Option<f32>,
        #[serde(default = "one")]
        exponent: f32,
    },
    ColorRandom {
        #[serde(default)]
        min: Option<model::Vec3>,
        #[serde(default)]
        max: Option<model::Vec3>,
    },
    VelocityRandom {
        #[serde(default)]
        min: Option<model::Vec3>,
        #[serde(default)]
        max: Option<model::Vec3>,
    },
    TurbulentVelocityRandom {
        #[serde(default = "one")]
        scale: f32,
        #[serde(default)]
        offset: f32,
    },
    /// Initial roll about z, in radians.
    ///
    /// A bare `rotationrandom` — no `min`, no `max` — is eleven of the corpus's
    /// sixteen, and it is also exactly what Wallpaper Engine's own element
    /// preview scene for `rotationrandom` contains. A preview whose whole
    /// purpose is to show the element working cannot be a no-op, so the absent
    /// bounds are a full random turn, not the neutral zero every other
    /// initializer defaults to. One corpus preset writes that turn out in full
    /// (`min: "1 0 0"`, `max: "1 0 6.283"`).
    RotationRandom {
        #[serde(default)]
        min: Option<model::Vec3>,
        #[serde(default)]
        max: Option<model::Vec3>,
    },
    /// Initial spin about z, in radians per second. Bare in WE's own preview
    /// too, where it drives a bare `angularmovement`; the corpus writes
    /// `-1 .. 1` twice, which is the gentle spin such a default has to be.
    AngularVelocityRandom {
        #[serde(default)]
        min: Option<model::Vec3>,
        #[serde(default)]
        max: Option<model::Vec3>,
    },
    #[serde(other)]
    Unknown,
}

/// A bare `rotationrandom` spans a whole turn.
const FULL_TURN: (f32, f32) = (0.0, TAU);

/// A bare `angularvelocityrandom` spans ±1 rad/s.
const DEFAULT_SPIN: (f32, f32) = (-1.0, 1.0);

/// An angular initializer's bounds, read off the z component.
///
/// Both bounds absent means `default` — the one place the neutral-value rule
/// does not hold, because Wallpaper Engine's own element previews write these
/// two initializers bare and a preview cannot be a no-op. A named `min` alone
/// still pins both ends, and a named `max` alone still starts from zero.
fn spin_bounds(min: Option<model::Vec3>, max: Option<model::Vec3>, default: (f32, f32)) -> (f32, f32) {
    match (min, max) {
        (Some(min), Some(max)) => (min.z, max.z),
        (Some(min), None) => (min.z, min.z),
        (None, Some(max)) => (0.0, max.z),
        (None, None) => default,
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "name", rename_all = "lowercase")]
enum Operator {
    Movement {
        #[serde(default)]
        gravity: model::Vec3,
        #[serde(default)]
        drag: f32,
    },
    /// An omitted fade time is one second, not zero.
    ///
    /// Wallpaper Engine leaves out any field still holding its default, the
    /// same convention the initializers follow. Of the corpus's forty-four
    /// `alphafade` operators, twenty-eight name no `fadeouttime` and fourteen
    /// name neither field — reading those as zero makes the operator a no-op,
    /// and a no-op is not something a scene serializes fourteen times. An
    /// *explicit* zero still means no fade, which is the distinction serde's
    /// `default` draws for free.
    AlphaFade {
        #[serde(default = "one")]
        fadeintime: f32,
        #[serde(default = "one")]
        fadeouttime: f32,
    },
    OscillateAlpha {
        #[serde(default)]
        frequencymin: f32,
        #[serde(default = "one")]
        frequencymax: f32,
        #[serde(default)]
        scalemin: f32,
        #[serde(default = "one")]
        scalemax: f32,
    },
    OscillatePosition {
        #[serde(default)]
        frequencymin: f32,
        #[serde(default = "one")]
        frequencymax: f32,
        #[serde(default)]
        scalemin: f32,
        #[serde(default = "one")]
        scalemax: f32,
        /// Which axes the wobble is allowed on, as per-axis weights. Five
        /// corpus presets flatten it — `"1 0.5 0"` is a mostly-horizontal
        /// drift, and applying it isotropically instead makes fog bob.
        #[serde(default = "vec3_one")]
        mask: model::Vec3,
        #[serde(default)]
        phasemin: f32,
        #[serde(default = "one")]
        phasemax: f32,
    },
    OscillateSize {
        #[serde(default)]
        frequencymin: f32,
        #[serde(default = "one")]
        frequencymax: f32,
        #[serde(default = "one")]
        scalemin: f32,
        #[serde(default = "one")]
        scalemax: f32,
    },
    Turbulence {
        #[serde(default = "vec3_one")]
        mask: model::Vec3,
        #[serde(default = "one")]
        scale: f32,
        #[serde(default)]
        speedmin: f32,
        #[serde(default = "one")]
        speedmax: f32,
        #[serde(default)]
        phasemax: f32,
    },
    ControlPointAttract {
        #[serde(default)]
        scale: f32,
        #[serde(default = "one")]
        threshold: f32,
    },
    SizeChange {
        #[serde(default)]
        starttime: f32,
        #[serde(default = "one")]
        endtime: f32,
        #[serde(default = "one")]
        startvalue: f32,
        #[serde(default)]
        endvalue: f32,
    },
    /// `sizechange`'s shape applied to alpha. WE's own preview writes
    /// `startvalue: 0, endvalue: 1` — a fade *in* — so neither end is fixed and
    /// both have to be read.
    AlphaChange {
        #[serde(default)]
        starttime: f32,
        #[serde(default = "one")]
        endtime: f32,
        #[serde(default = "one")]
        startvalue: f32,
        #[serde(default)]
        endvalue: f32,
    },
    /// The same ramp on colour. Unlike `colorrandom`, whose bounds are 0..255,
    /// these are 0..1 floats — the corpus writes `"1 0.835 0.573"` and WE's
    /// preview `"1 0 0"`. An absent end is white, so naming only one end ramps
    /// from or to the untinted particle.
    ColorChange {
        #[serde(default)]
        starttime: f32,
        #[serde(default = "one")]
        endtime: f32,
        #[serde(default = "vec3_one")]
        startvalue: model::Vec3,
        #[serde(default = "vec3_one")]
        endvalue: model::Vec3,
    },
    /// Spin about the system origin: tangential speed interpolated from
    /// `speedinner` at `distanceinner` to `speedouter` at `distanceouter`.
    Vortex {
        #[serde(default)]
        distanceinner: f32,
        #[serde(default = "one")]
        distanceouter: f32,
        #[serde(default)]
        speedinner: f32,
        #[serde(default)]
        speedouter: f32,
    },
    /// Angular acceleration about z, in radians per second squared.
    AngularMovement {
        #[serde(default)]
        force: model::Vec3,
        #[serde(default)]
        drag: f32,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "name", rename_all = "lowercase")]
enum Renderer {
    Sprite,
    /// `length` scales the trail by the particle's speed; `maxlength` caps it
    /// in sprite-widths. Nine corpus renderers write `maxlength` and only five
    /// write `length` — four name `maxlength` alone, so reading `length` on its
    /// own leaves those four with no trail at all.
    SpriteTrail {
        #[serde(default)]
        length: f32,
        #[serde(default)]
        maxlength: f32,
    },
    /// A trail through the particle's own recent path rather than straight back
    /// along its velocity, so a curving particle leaves a curve. `length` is in
    /// seconds of history.
    RopeTrail {
        #[serde(default = "one")]
        length: f32,
        #[serde(default = "default_segments")]
        segments: u32,
    },
    #[serde(other)]
    Unknown,
}

fn default_segments() -> u32 {
    16
}

fn one() -> f32 {
    1.0
}

fn vec3_one() -> model::Vec3 {
    model::Vec3::splat(1.0)
}

fn xy(v: model::Vec3) -> Vec2 {
    Vec2::new(v.x, v.y)
}

// ---------------------------------------------------------------------------
// Placement: how a system's local sim units land on the output canvas
// ---------------------------------------------------------------------------

/// Everything from the scene object needed to put a particle on screen.
#[derive(Clone)]
pub struct Placement {
    /// The object's `origin`, already in output pixels (scene Y flipped).
    pub origin_px: Vec2,
    /// The object's `scale.xy`.
    pub scale: Vec2,
    /// Output pixels per scene unit.
    pub px_per_unit: f32,
    pub canvas_px: (u32, u32),
    /// `color * brightness` from the object.
    pub tint: Vec3,
    /// The object's `alpha`.
    pub alpha: f32,
    pub overrides: InstanceOverride,
    /// Per-particle integration-step ceiling. `MAX_STEPS` for an exact still;
    /// the live simulator sets it lower to bound the per-frame cost.
    pub max_sim_steps: u32,
    /// The object's roll, in radians, applied to the system's local space —
    /// so the emitter box turns and gravity leans with it.
    ///
    /// A particle layer rasterizes to the whole canvas, so rolling its quad in
    /// the compositor would swing the entire field about the canvas centre
    /// instead of about the emitter. It has to happen here. Thirty-one of
    /// `scene_example8`'s systems carry one, including two smoke banks at
    /// roughly ±90 degrees.
    pub roll: (f32, f32),
}

/// The exact-integration step ceiling (`MAX_STEPS`), for the still exporter.
pub const EXACT_SIM_STEPS: u32 = MAX_STEPS;

/// A live particle's local position (sim units) to an output-pixel point.
fn to_screen(place: &Placement, local: Vec2) -> Point {
    // Roll in the system's own (Y-up) space, before the flip to image rows.
    let (cos, sin) = place.roll;
    let rolled = Vec2::new(local.x * cos - local.y * sin, local.x * sin + local.y * cos);
    let scaled = Vec2::new(rolled.x * place.scale.x, -rolled.y * place.scale.y) * place.px_per_unit;
    let p = place.origin_px + scaled;
    Point::from_xy(p.x, p.y)
}

// ---------------------------------------------------------------------------
// Per-particle roll: birth state drawn from the seeded PCG stream
// ---------------------------------------------------------------------------

struct Rolled {
    start: Vec2,
    velocity: Vec2,
    lifetime: f32,
    size: f32,
    alpha: f32,
    color: Vec3,
    turb_speed: f32,
    turb_phase: f32,
    osc_alpha_freq: f32,
    osc_alpha_phase: f32,
    osc_pos_freq: f32,
    osc_pos_phase: f32,
    osc_pos_dir: Vec2,
    osc_pos_amp: f32,
    osc_size_freq: f32,
    osc_size_phase: f32,
    osc_pos_mask: Vec2,
    rotation: f32,
    spin: f32,
}

fn frand(rng: &mut Pcg64Mcg) -> f32 {
    rng.random_range(0.0..1.0)
}

fn rrange(rng: &mut Pcg64Mcg, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * frand(rng)
}

/// `lo..hi` with the sample biased by `exponent` (WE's non-uniform randoms).
fn rpow(rng: &mut Pcg64Mcg, lo: f32, hi: f32, exponent: f32) -> f32 {
    lo + (hi - lo) * frand(rng).powf(exponent.max(0.01))
}

fn rand_unit(rng: &mut Pcg64Mcg) -> Vec2 {
    let angle = frand(rng) * TAU;
    Vec2::new(angle.cos(), angle.sin())
}

fn emitter_sample(emitter: &Emitter, rng: &mut Pcg64Mcg) -> Vec2 {
    match emitter {
        Emitter::SphereRandom { origin, distancemin, distancemax, directions, .. } => {
            let radius = rrange(rng, distancemin.x, distancemax.x);
            xy(*origin) + rand_unit(rng) * radius * xy(*directions)
        }
        Emitter::BoxRandom { origin, distancemax, directions, .. } => {
            let extent = xy(*distancemax) * xy(*directions);
            xy(*origin) + Vec2::new(rrange(rng, -extent.x, extent.x), rrange(rng, -extent.y, extent.y))
        }
        Emitter::Unknown => Vec2::ZERO,
    }
}

/// Roll a particle's birth state. The draw order is fixed given the preset, so
/// a given global index `n` always yields the same particle.
fn roll(preset: &Preset, emitter: &Emitter, flow: &Perlin, mut rng: Pcg64Mcg) -> Rolled {
    let start = emitter_sample(emitter, &mut rng);

    let mut r = Rolled {
        start,
        velocity: Vec2::ZERO,
        lifetime: 3.0,
        size: 32.0,
        alpha: 1.0,
        color: Vec3::ONE,
        turb_speed: 0.0,
        turb_phase: 0.0,
        osc_alpha_freq: 0.0,
        osc_alpha_phase: 0.0,
        osc_pos_freq: 0.0,
        osc_pos_phase: 0.0,
        osc_pos_dir: Vec2::X,
        osc_pos_amp: 0.0,
        osc_size_freq: 0.0,
        osc_size_phase: 0.0,
        osc_pos_mask: Vec2::ONE,
        rotation: 0.0,
        spin: 0.0,
    };

    for init in &preset.initializer {
        match init {
            Initializer::LifeTimeRandom { min, max, exponent } => {
                let (low, high) = bounds(*min, *max, 1.0);
                r.lifetime = rpow(&mut rng, low, high, *exponent).max(0.01);
            }
            Initializer::SizeRandom { min, max, exponent } => {
                let (low, high) = bounds(*min, *max, 1.0);
                r.size = rpow(&mut rng, low, high, *exponent).max(0.0);
            }
            Initializer::AlphaRandom { min, max, exponent } => {
                let (low, high) = alpha_bounds(*min, *max);
                r.alpha = rpow(&mut rng, low, high, *exponent).clamp(0.0, 1.0);
            }
            Initializer::ColorRandom { min, max } => {
                let (min, max) = bounds(*min, *max, model::Vec3::splat(255.0));
                r.color = Vec3::new(
                    rrange(&mut rng, min.x, max.x),
                    rrange(&mut rng, min.y, max.y),
                    rrange(&mut rng, min.z, max.z),
                ) / 255.0;
            }
            Initializer::VelocityRandom { min, max } => {
                let (min, max) = bounds(*min, *max, model::Vec3::default());
                r.velocity += Vec2::new(
                    rrange(&mut rng, min.x, max.x),
                    rrange(&mut rng, min.y, max.y),
                );
            }
            Initializer::TurbulentVelocityRandom { scale, offset } => {
                let p = r.start * *scale + Vec2::splat(*offset * 100.0);
                r.velocity += flow_at(flow, p) * TURB_INIT_SPEED;
            }
            Initializer::RotationRandom { min, max } => {
                let (low, high) = spin_bounds(*min, *max, FULL_TURN);
                r.rotation = rrange(&mut rng, low, high);
            }
            Initializer::AngularVelocityRandom { min, max } => {
                let (low, high) = spin_bounds(*min, *max, DEFAULT_SPIN);
                r.spin = rrange(&mut rng, low, high);
            }
            Initializer::Unknown => {}
        }
    }

    for op in &preset.operator {
        match op {
            Operator::Turbulence { speedmin, speedmax, phasemax, .. } => {
                r.turb_speed = rrange(&mut rng, *speedmin, *speedmax);
                r.turb_phase = rrange(&mut rng, 0.0, phasemax.max(0.0));
            }
            Operator::OscillateAlpha { frequencymin, frequencymax, .. } => {
                r.osc_alpha_freq = rrange(&mut rng, *frequencymin, frequencymax.max(*frequencymin));
                r.osc_alpha_phase = frand(&mut rng) * TAU;
            }
            Operator::OscillatePosition { frequencymin, frequencymax, scalemin, scalemax, mask, phasemin, phasemax } => {
                r.osc_pos_freq = rrange(&mut rng, *frequencymin, frequencymax.max(*frequencymin));
                r.osc_pos_amp = rrange(&mut rng, *scalemin, *scalemax);
                r.osc_pos_phase = rrange(&mut rng, *phasemin, phasemax.max(*phasemin)) * TAU;
                r.osc_pos_dir = rand_unit(&mut rng);
                r.osc_pos_mask = xy(*mask);
            }
            Operator::OscillateSize { frequencymin, frequencymax, .. } => {
                r.osc_size_freq = rrange(&mut rng, *frequencymin, frequencymax.max(*frequencymin));
                r.osc_size_phase = frand(&mut rng) * TAU;
            }
            _ => {}
        }
    }

    r
}

fn flow_at(flow: &Perlin, p: Vec2) -> Vec2 {
    let x = flow.get([f64::from(p.x) * 0.01, f64::from(p.y) * 0.01]) as f32;
    let y = flow.get([f64::from(p.x) * 0.01 + 41.7, f64::from(p.y) * 0.01 - 17.3]) as f32;
    Vec2::new(x, y)
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

struct Live {
    pos: Vec2,
    size: f32,
    alpha: f32,
    color: Vec3,
    velocity: Vec2,
    /// Roll about the sprite's own centre, in radians.
    rotation: f32,
    /// Where the particle was over the last `TRAIL_HISTORY` seconds, newest
    /// last — what `ropetrail` draws through. Empty unless a rope renderer
    /// asked for it, since keeping it costs an allocation per particle.
    trail: Vec<Vec2>,
}

/// Where the integration left a particle, before the display-only operators.
struct Integrated {
    pos: Vec2,
    vel: Vec2,
    rotation: f32,
    trail: Vec<Vec2>,
}

/// Integrate a particle to `age` seconds, then apply the display-only
/// operators (fades, oscillators, size ramp) as closed-form functions of age.
fn simulate(preset: &Preset, r: &Rolled, flow: &Perlin, speed: f32, age: f32, max_steps: u32) -> Live {
    let path = integrate(preset, r, flow, speed, age, max_steps);
    display(preset, r, age, path)
}

/// The stepped half: forces on velocity, velocity on position, torque on roll.
fn integrate(preset: &Preset, r: &Rolled, flow: &Perlin, speed: f32, age: f32, max_steps: u32) -> Integrated {
    let steps = ((age * SIM_HZ).ceil() as u32).clamp(1, max_steps.max(1));
    let dt = age / steps as f32;

    let mut pos = r.start;
    let mut vel = r.velocity * speed;
    let mut rotation = r.rotation;
    let mut spin = r.spin;
    // A rope keeps the last `length` seconds of path, sampled at `segments`
    // points; everything else keeps nothing, since the history costs an
    // allocation per particle.
    let rope = match preset.renderer.first() {
        Some(Renderer::RopeTrail { length, segments }) => Some((*length, (*segments).clamp(2, 64))),
        _ => None,
    };
    let mut trail = Vec::new();
    let rope_from = rope.map_or(u32::MAX, |(length, segments)| {
        let history = ((length / dt.max(1e-4)).ceil() as u32).max(segments);
        steps.saturating_sub(history)
    });
    let rope_every = rope.map_or(1, |(_, segments)| (steps.saturating_sub(rope_from) / segments).max(1));

    for step in 0..steps {
        for op in &preset.operator {
            match op {
                Operator::Movement { gravity, drag } => {
                    vel += xy(*gravity) * dt;
                    vel *= (-drag * dt).exp();
                }
                Operator::Turbulence { mask, scale, .. } => {
                    let force = flow_at(flow, pos * *scale + Vec2::splat(r.turb_phase));
                    vel += force * xy(*mask) * (r.turb_speed * speed) * dt;
                }
                Operator::ControlPointAttract { scale, threshold } => {
                    // The only control point any preset attracts to sits at the
                    // system origin; `scale` is negative to push particles out.
                    let to_cp = -pos;
                    let dist = to_cp.length();
                    if dist < *threshold && dist > 1.0 {
                        vel += to_cp / dist * (*scale * speed) * dt / dist.max(4.0);
                    }
                }
                Operator::Vortex { distanceinner, distanceouter, speedinner, speedouter } => {
                    // Tangential, about the same system origin: the radius picks
                    // the speed, and the sign of the speed picks the direction.
                    let radius = pos.length();
                    if radius > 1.0 {
                        let reach = (distanceouter - distanceinner).max(1e-3);
                        let k = ((radius - distanceinner) / reach).clamp(0.0, 1.0);
                        let tangential = speedinner + (speedouter - speedinner) * k;
                        let around = Vec2::new(-pos.y, pos.x) / radius;
                        vel += around * (tangential * speed) * dt / radius.max(4.0);
                    }
                }
                Operator::AngularMovement { force, drag } => {
                    spin += force.z * dt;
                    spin *= (-drag * dt).exp();
                }
                _ => {}
            }
        }
        pos += vel * dt;
        rotation += spin * dt;
        if step >= rope_from && (step - rope_from) % rope_every == 0 {
            trail.push(pos);
        }
    }

    Integrated { pos, vel, rotation, trail }
}

/// The closed-form half: fades, oscillators and `*change` ramps, each a
/// function of age alone.
fn display(preset: &Preset, r: &Rolled, age: f32, path: Integrated) -> Live {
    let mut size = r.size;
    let mut alpha = r.alpha;
    let mut color = r.color;
    let mut draw_pos = path.pos;
    let life_frac = (age / r.lifetime).clamp(0.0, 1.0);

    for op in &preset.operator {
        match op {
            Operator::AlphaFade { fadeintime, fadeouttime } => {
                if *fadeintime > 0.0 {
                    alpha *= (age / fadeintime).clamp(0.0, 1.0);
                }
                if *fadeouttime > 0.0 {
                    alpha *= ((r.lifetime - age) / fadeouttime).clamp(0.0, 1.0);
                }
            }
            Operator::OscillateAlpha { scalemin, scalemax, .. } => {
                let s = 0.5 + 0.5 * (TAU * r.osc_alpha_freq * age + r.osc_alpha_phase).sin();
                let (low, high) = (scalemin.max(0.0), *scalemax);
                alpha *= low + (high - low) * s;
            }
            Operator::OscillateSize { scalemin, scalemax, .. } => {
                let s = 0.5 + 0.5 * (TAU * r.osc_size_freq * age + r.osc_size_phase).sin();
                size *= scalemin + (scalemax - scalemin) * s;
            }
            Operator::OscillatePosition { .. } => {
                let w = (TAU * r.osc_pos_freq * age + r.osc_pos_phase).sin() * r.osc_pos_amp;
                draw_pos += r.osc_pos_dir * r.osc_pos_mask * w;
            }
            Operator::SizeChange { starttime, endtime, startvalue, endvalue } => {
                size *= ramp(life_frac, *starttime, *endtime, *startvalue, *endvalue);
            }
            Operator::AlphaChange { starttime, endtime, startvalue, endvalue } => {
                alpha *= ramp(life_frac, *starttime, *endtime, *startvalue, *endvalue);
            }
            Operator::ColorChange { starttime, endtime, startvalue, endvalue } => {
                let (from, to) = (*startvalue, *endvalue);
                color *= Vec3::new(
                    ramp(life_frac, *starttime, *endtime, from.x, to.x),
                    ramp(life_frac, *starttime, *endtime, from.y, to.y),
                    ramp(life_frac, *starttime, *endtime, from.z, to.z),
                );
            }
            _ => {}
        }
    }

    Live {
        pos: draw_pos,
        size: size.max(0.0),
        alpha: alpha.clamp(0.0, 1.0),
        color,
        velocity: path.vel,
        rotation: path.rotation,
        trail: path.trail,
    }
}

/// A `*change` operator's value at `life_frac`: held at `from` before
/// `starttime`, at `to` after `endtime`, linear between.
fn ramp(life_frac: f32, starttime: f32, endtime: f32, from: f32, to: f32) -> f32 {
    let span = (endtime - starttime).max(1e-3);
    let k = ((life_frac - starttime) / span).clamp(0.0, 1.0);
    from + (to - from) * k
}

// ---------------------------------------------------------------------------
// Emission bookkeeping
// ---------------------------------------------------------------------------

struct Emission {
    rate: f32,
    maxcount: u32,
    starttime: f32,
}

fn emission(preset: &Preset, emitter: &Emitter, over: &InstanceOverride, starttime: f32) -> Emission {
    let base_rate = match emitter {
        Emitter::SphereRandom { rate, .. } | Emitter::BoxRandom { rate, .. } => *rate,
        Emitter::Unknown => 0.0,
    };
    let count = over.count.max(0.0);
    Emission {
        rate: (base_rate * over.rate.max(0.0) * count).max(1e-4),
        maxcount: ((preset.maxcount * count).round() as u32).max(1),
        starttime,
    }
}

/// The particle occupying `slot` at `time`, as `(global index, birth time)` —
/// the newest emission congruent to `slot` mod `maxcount` whose birth is not
/// in the future. Whether it is still alive is the caller's check.
fn slot_particle(em: &Emission, slot: u32, time: f32) -> Option<(u64, f32)> {
    if time < em.starttime {
        return None;
    }
    let newest = ((time - em.starttime) * em.rate).floor();
    if newest < 0.0 {
        return None;
    }
    let newest = newest as u64;
    let modulus = u64::from(em.maxcount);
    let slot = u64::from(slot);
    if newest < slot {
        return None;
    }
    let n = newest - (newest - slot) % modulus;
    let birth = em.starttime + n as f32 / em.rate;
    Some((n, birth))
}

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

fn seed(salt: u64, n: u64) -> Pcg64Mcg {
    Pcg64Mcg::seed_from_u64(salt ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xD1B5_4A32))
}

/// A soft round sprite: a radial gradient from the particle colour at full
/// weight to fully transparent at the rim.
/// Draw the sprite for one particle, tinted and scaled to its rect.
///
/// The sprite is a `Pattern` rather than a `draw_pixmap` blit because the
/// pattern shader carries the blend mode and the scale transform together, and
/// because tinting has to happen anyway: tiny-skia has no colour filter, so a
/// tinted copy is built once per (sprite, quantised colour) and reused. Most
/// systems fix their colour, so the cache almost always holds one entry.
fn tinted_sprite<'a>(
    cache: &'a mut HashMap<(u64, u32), Pixmap>,
    key: u64,
    base: &Pixmap,
    color: Vec3,
) -> &'a Pixmap {
    let rgb = color.clamp(Vec3::ZERO, Vec3::ONE);
    let quantized = (quantize(rgb.x) << 16) | (quantize(rgb.y) << 8) | quantize(rgb.z);
    cache.entry((key, quantized)).or_insert_with(|| {
        let mut tinted = base.clone();
        for pixel in tinted.pixels_mut() {
            // Premultiplied throughout, so scaling RGB alone keeps it valid.
            let (r, g, b, a) = (pixel.red(), pixel.green(), pixel.blue(), pixel.alpha());
            *pixel = PremultipliedColorU8::from_rgba(
                channel(r, rgb.x),
                channel(g, rgb.y),
                channel(b, rgb.z),
                a,
            )
            .unwrap_or(*pixel);
        }
        tinted
    })
}

#[expect(clippy::cast_sign_loss, reason = "clamped to 0.0..=1.0 before the cast")]
fn quantize(value: f32) -> u32 {
    (value.clamp(0.0, 1.0) * 31.0).round() as u32
}

#[expect(clippy::cast_sign_loss, reason = "a product of clamped 0..=1 factors")]
fn channel(value: u8, factor: f32) -> u8 {
    (f32::from(value) * factor.clamp(0.0, 1.0)).round() as u8
}

#[expect(clippy::too_many_arguments, reason = "one particle's full draw state")]
fn draw_particle(
    pixmap: &mut Pixmap,
    place: &Placement,
    renderer: &Renderer,
    live: &Live,
    blend: model::Blend,
    base: &Pixmap,
    cache: &mut HashMap<(u64, u32), Pixmap>,
    cache_key: u64,
) {
    let over = &place.overrides;
    let center = to_screen(place, live.pos);
    let radius = 0.5 * live.size * place.scale.x.abs() * over.size.max(0.0) * place.px_per_unit;
    if radius < 0.25 || live.alpha <= 0.001 {
        return;
    }

    let color = over.colorn.map_or(live.color, |c| Vec3::new(c.x, c.y, c.z))
        * place.tint
        * over.brightness.max(0.0);
    let weight = live.alpha * place.alpha * over.alpha.max(0.0);
    let sprite = tinted_sprite(cache, cache_key, base, color);

    let to_px = place.scale.x.abs() * place.px_per_unit;
    match renderer {
        Renderer::SpriteTrail { length, maxlength } => {
            // `length` is seconds of travel; `maxlength` caps the result in
            // sprite radii, and stands alone in four corpus renderers.
            let vel_px = Vec2::new(live.velocity.x, -live.velocity.y) * to_px;
            let by_speed = if *length > 0.0 { length * vel_px.length() } else { f32::INFINITY };
            let by_cap = if *maxlength > 0.0 { maxlength * radius } else { f32::INFINITY };
            let back = by_speed.min(by_cap);
            if back.is_finite() && back > 1.0 {
                let dir = vel_px.normalize_or_zero();
                let tail = Point::from_xy(center.x - dir.x * back, center.y - dir.y * back);
                stroke_trail(pixmap, sprite, &[tail, center], radius, weight * 0.6, blend);
            }
        }
        Renderer::RopeTrail { .. } if live.trail.len() > 1 => {
            let points: Vec<Point> = live.trail.iter().map(|p| to_screen(place, *p)).collect();
            stroke_trail(pixmap, sprite, &points, radius, weight * 0.6, blend);
        }
        _ => {}
    }

    let Some(rect) = sprite_rect(sprite, center, radius) else {
        return;
    };
    let Some(paint) = sprite_paint(sprite, weight, rect, blend) else {
        return;
    };
    // Rolled about the sprite's own centre. tiny-skia applies the transform to
    // the pattern as well as to the rectangle, so the sprite turns with it.
    let transform = if live.rotation.abs() > 1e-4 {
        Transform::from_rotate_at(live.rotation.to_degrees(), center.x, center.y)
    } else {
        Transform::identity()
    };
    pixmap.fill_rect(rect, &paint, transform, None);
}

/// Stroke a sprite-coloured line through `points` — how both trail renderers
/// draw, differing only in the path they hand over.
fn stroke_trail(
    pixmap: &mut Pixmap,
    sprite: &Pixmap,
    points: &[Point],
    radius: f32,
    weight: f32,
    blend: model::Blend,
) {
    let Some(head) = points.last() else { return };
    let Some(paint) = sprite_rect(sprite, *head, radius.max(1.0))
        .and_then(|rect| sprite_paint(sprite, weight, rect, blend))
    else {
        return;
    };
    let mut pb = PathBuilder::new();
    let Some((first, rest)) = points.split_first() else { return };
    pb.move_to(first.x, first.y);
    for point in rest {
        pb.line_to(point.x, point.y);
    }
    if let Some(path) = pb.finish() {
        let stroke = Stroke { width: radius.max(1.0), line_cap: LineCap::Round, ..Stroke::default() };
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }
}

/// The particle's on-screen rectangle: `radius` sets the longer side, and the
/// sprite's own aspect sets the other.
///
/// Squashing every sprite into a square turns a streak into a blob.
/// `scene_example8`'s rain uses a 256x1280 sheet — five times taller than it is
/// wide — and Wallpaper Engine draws it as the thin falling line it is.
fn sprite_rect(sprite: &Pixmap, center: Point, radius: f32) -> Option<Rect> {
    #[expect(clippy::cast_precision_loss, reason = "sprite sides are at most 1024")]
    let (sw, sh) = (sprite.width() as f32, sprite.height() as f32);
    if sw <= 0.0 || sh <= 0.0 {
        return None;
    }
    let longest = sw.max(sh);
    let (half_w, half_h) = (radius * sw / longest, radius * sh / longest);
    Rect::from_xywh(center.x - half_w, center.y - half_h, half_w * 2.0, half_h * 2.0)
}

/// Paint that maps `sprite` onto `rect`, at `weight` opacity.
fn sprite_paint(
    sprite: &Pixmap,
    weight: f32,
    rect: Rect,
    blend: model::Blend,
) -> Option<Paint<'_>> {
    #[expect(clippy::cast_precision_loss, reason = "sprite sides are at most 1024")]
    let (sw, sh) = (sprite.width() as f32, sprite.height() as f32);
    if sw <= 0.0 || sh <= 0.0 {
        return None;
    }
    let transform = Transform::from_row(
        rect.width() / sw,
        0.0,
        0.0,
        rect.height() / sh,
        rect.x(),
        rect.y(),
    );
    let shader = Pattern::new(
        sprite.as_ref(),
        SpreadMode::Pad,
        FilterQuality::Bilinear,
        weight.clamp(0.0, 1.0),
        transform,
    );
    Some(Paint { shader, blend_mode: blend_mode(blend), anti_alias: true, ..Paint::default() })
}

/// `Plus` sums overlapping particles, which is what an additive material does;
/// a translucent one must let the nearest particle cover the rest, or a dense
/// system clips to white.
fn blend_mode(blend: model::Blend) -> BlendMode {
    match blend {
        model::Blend::Add => BlendMode::Plus,
        model::Blend::Over => BlendMode::SourceOver,
    }
}

/// Render one preset and its child systems into `pixmap`.
///
/// A plain (sibling) child system is independent of the parent's particles and
/// is rendered **once**; an `eventfollow` child rides each parent particle, so
/// it is rendered once per live parent with its origin moved to that particle.
#[expect(clippy::too_many_arguments, reason = "recursion carries the full render context")]
fn render_preset(
    presets: &HashMap<String, Preset>,
    key: &str,
    pixmap: &mut Pixmap,
    place: &Placement,
    time: f32,
    starttime: f32,
    salt: u64,
    depth: u32,
    unsupported: &mut Vec<String>,
    cache: &mut HashMap<(u64, u32), Pixmap>,
) {
    let Some(preset) = presets.get(key) else {
        return;
    };
    let Some(emitter) = preset.emitter.first() else {
        return;
    };
    note_unsupported(preset, emitter, unsupported);

    let renderer = preset.renderer.first().unwrap_or(&Renderer::Sprite);
    let flow = Perlin::new((salt & 0xFFFF_FFFF) as u32);
    let em = emission(preset, emitter, &place.overrides, starttime);
    let speed = place.overrides.speed.max(0.0);
    let children = preset.children.iter().flatten();

    // Sibling child systems: rendered once, alongside the parent system.
    if depth + 1 < MAX_DEPTH {
        for child in children.clone().filter(|child| child.r#type != "eventfollow") {
            render_preset(
                presets, &child.name, pixmap, place, time, starttime,
                salt ^ fnv1a(&child.name), depth + 1, unsupported, cache,
            );
        }
    }

    // Each slot's whole trajectory is re-integrated from birth (the stateless
    // design that keeps frames independently addressable), which gets costly as
    // particles age — so integrate every alive slot in parallel, then rasterize
    // and hang `eventfollow` children serially since `Pixmap` is not `Sync`.
    let alive: Vec<(u64, f32, Live)> = (0..em.maxcount)
        .into_par_iter()
        .filter_map(|slot| {
            let (n, birth) = slot_particle(&em, slot, time)?;
            let age = time - birth;
            let rolled = roll(preset, emitter, &flow, seed(salt, n));
            (age < rolled.lifetime)
                .then(|| (n, birth, simulate(preset, &rolled, &flow, speed, age, place.max_sim_steps)))
        })
        .collect();

    let base = preset.sprite.clone().unwrap_or_else(|| sprite::stand_in("particle/halo"));
    let cache_key = fnv1a(key);
    for (n, birth, live) in &alive {
        draw_particle(pixmap, place, renderer, live, preset.blend, &base, cache, cache_key);
        if depth + 1 < MAX_DEPTH {
            for child in children.clone().filter(|child| child.r#type == "eventfollow") {
                let at = to_screen(place, live.pos);
                let child_place = Placement { origin_px: Vec2::new(at.x, at.y), ..place.clone() };
                render_preset(
                    presets, &child.name, pixmap, &child_place, time, *birth,
                    salt ^ n.wrapping_mul(0x9E37_79B9), depth + 1, unsupported, cache,
                );
            }
        }
    }
}

fn note_unsupported(preset: &Preset, emitter: &Emitter, out: &mut Vec<String>) {
    let mut push = |name: &str| {
        let msg = format!("{name} not simulated");
        if !out.contains(&msg) {
            out.push(msg);
        }
    };
    if matches!(emitter, Emitter::Unknown) {
        push("an emitter shape");
    }
    for init in &preset.initializer {
        if matches!(init, Initializer::Unknown) {
            push("an initializer");
        }
    }
    for op in &preset.operator {
        if matches!(op, Operator::Unknown) {
            push("an operator");
        }
    }
    if matches!(preset.renderer.first(), Some(Renderer::Unknown)) {
        push("a renderer");
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub struct ParticleLayer {
    pub image: RgbaImage,
    /// Names of preset features encountered but not simulated.
    pub unsupported: Vec<String>,
}

/// Load, simulate and rasterize a particle system at `time`, returning a
/// full-canvas RGBA layer meant to be composited additively.
pub fn render_system(
    archive: &mut Archive,
    assets: Option<&Path>,
    preset_path: &str,
    place: &Placement,
    time: f32,
) -> Result<ParticleLayer> {
    let presets = collect_system(archive, assets, preset_path)?;
    render_system_from(&presets, preset_path, place, time)
}

/// Read and parse a preset and every child it names — the file work, done once
/// so a live redraw can call `render_system_from` each frame without touching
/// the archive.
pub fn collect_system(
    archive: &mut Archive,
    assets: Option<&Path>,
    preset_path: &str,
) -> Result<HashMap<String, Preset>> {
    let mut presets = HashMap::new();
    collect_presets(archive, assets, preset_path, 0, &mut presets)
        .with_context(|| format!("loading particle preset {preset_path}"))?;
    Ok(presets)
}

/// Simulate and rasterize an already-parsed system at `time`.
pub fn render_system_from(
    presets: &HashMap<String, Preset>,
    preset_path: &str,
    place: &Placement,
    time: f32,
) -> Result<ParticleLayer> {
    let start = presets.get(preset_path).context("particle preset had no body")?.starttime;

    let (w, h) = place.canvas_px;
    let mut pixmap = Pixmap::new(w.max(1), h.max(1)).context("allocating a particle canvas")?;
    let mut unsupported = Vec::new();

    let salt = 0x5EED_u64.wrapping_add(fnv1a(preset_path));
    let mut tints = HashMap::new();
    render_preset(
        presets, preset_path, &mut pixmap, place, time, start, salt, 0, &mut unsupported, &mut tints,
    );

    Ok(ParticleLayer { image: pixmap_to_rgba(&pixmap), unsupported })
}

fn collect_presets(
    archive: &mut Archive,
    assets: Option<&Path>,
    key: &str,
    depth: u32,
    out: &mut HashMap<String, Preset>,
) -> Result<()> {
    if depth >= MAX_DEPTH || out.contains_key(key) {
        return Ok(());
    }
    let bytes = archive.read(key).with_context(|| format!("reading {key}"))?;
    let mut preset: Preset = serde_json::from_slice(&bytes).with_context(|| format!("parsing {key}"))?;
    let material = read_material(archive, &preset.material);
    preset.blend = material.as_ref().map_or(model::Blend::Add, model::base_blend);
    preset.sprite = Some(sprite::resolve(
        archive,
        assets,
        material.as_ref().and_then(model::base_texture).unwrap_or("particle/halo"),
    ));
    let child_names: Vec<String> =
        preset.children.iter().flatten().map(|c| c.name.clone()).collect();
    out.insert(key.to_string(), preset);
    for name in child_names {
        collect_presets(archive, assets, &name, depth + 1, out)?;
    }
    Ok(())
}

/// The blend the root preset's material asks for — how the whole rasterized
/// layer meets the scene. `additive` is the fallback if the lookup fails.
pub fn layer_blend(archive: &mut Archive, preset_path: &str) -> model::Blend {
    read_blend(archive, preset_path).unwrap_or(model::Blend::Add)
}

fn read_blend(archive: &mut Archive, preset_path: &str) -> Option<model::Blend> {
    let preset: Preset = serde_json::from_slice(&archive.read(preset_path).ok()?).ok()?;
    material_blend(archive, &preset.material)
}

/// Whether the root preset's material refracts the frame behind it, which the
/// compositor reproduces by multiplying the layer into the frame.
pub fn layer_refracts(archive: &mut Archive, preset_path: &str) -> bool {
    let Ok(bytes) = archive.read(preset_path) else { return false };
    let Ok(preset) = serde_json::from_slice::<Preset>(&bytes) else { return false };
    read_material(archive, &preset.material).is_some_and(|material| model::base_refracts(&material))
}

/// The blend one material file asks for.
fn material_blend(archive: &mut Archive, material_path: &str) -> Option<model::Blend> {
    Some(model::base_blend(&read_material(archive, material_path)?))
}

fn read_material(archive: &mut Archive, material_path: &str) -> Option<Material> {
    if material_path.is_empty() {
        return None;
    }
    serde_json::from_slice(&archive.read(material_path).ok()?).ok()
}

fn fnv1a(text: &str) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Undo tiny-skia's premultiplication into a plain RGBA image.
///
/// `demultiply` costs a division per channel, and a particle canvas is mostly
/// empty — the two saturated cases need no division at all, and skipping them
/// is most of this function's cost on a scene with dozens of systems.
fn pixmap_to_rgba(pixmap: &Pixmap) -> RgbaImage {
    let mut out = RgbaImage::new(pixmap.width(), pixmap.height());
    for (dst, src) in out.pixels_mut().zip(pixmap.pixels()) {
        dst.0 = match src.alpha() {
            0 => [0, 0, 0, 0],
            255 => [src.red(), src.green(), src.blue(), 255],
            _ => {
                let c = src.demultiply();
                [c.red(), c.green(), c.blue(), c.alpha()]
            }
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset(json: &str) -> Preset {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parses_a_representative_preset() {
        let p = preset(
            r#"{
                "emitter":[{"name":"boxrandom","distancemax":"512 256 0","rate":10}],
                "initializer":[
                    {"name":"lifetimerandom","min":3,"max":5},
                    {"name":"colorrandom","min":"255 143 102","max":"255 218 108"}],
                "operator":[{"name":"movement","drag":0},{"name":"alphafade","fadeintime":0.1,"fadeouttime":1}],
                "renderer":[{"name":"spritetrail","length":0.007}],
                "material":"materials/presets/ember.json",
                "maxcount":40,"starttime":3
            }"#,
        );
        assert_eq!(p.maxcount, 40.0);
        assert!(matches!(p.emitter[0], Emitter::BoxRandom { .. }));
        assert!(matches!(p.renderer[0], Renderer::SpriteTrail { .. }));
        assert_eq!(p.initializer.len(), 2);
    }

    #[test]
    fn a_colorrandom_without_a_max_is_a_fixed_colour() {
        // Half the corpus's colorrandom entries write only `min`; reading the
        // absent `max` as zero would turn every such particle black.
        let p = preset(r#"{"initializer":[{"name":"colorrandom","min":"255 255 255"}]}"#);
        let Initializer::ColorRandom { min, max } = &p.initializer[0] else {
            panic!("expected a colorrandom initializer");
        };
        assert_eq!(bounds(*min, *max, model::Vec3::splat(255.0)).1, model::Vec3::splat(255.0));
    }

    #[test]
    fn an_initializer_with_no_bounds_at_all_is_neutral() {
        // `scene_example8`'s wet_snow1 writes a bare `{"name":"alpharandom"}`,
        // which means the whole range: reading it as a fixed 1.0 makes the
        // initializer a no-op and every snowflake a solid white disc.
        let p = preset(r#"{"initializer":[{"name":"alpharandom"}]}"#);
        let Initializer::AlphaRandom { min, max, .. } = &p.initializer[0] else {
            panic!("expected an alpharandom initializer");
        };
        assert_eq!(alpha_bounds(*min, *max), (0.0, 1.0));
    }

    #[test]
    fn an_alpha_bound_given_only_as_a_maximum_starts_at_zero() {
        // `rainperspective` writes `{"max": 0.3}`. Defaulting the absent `min`
        // to 1.0 would invert the range outright.
        assert_eq!(alpha_bounds(None, Some(0.3)), (0.0, 0.3));
        // `wind-blur` writes `{"min": 0.8}`, which still pins both ends.
        assert_eq!(alpha_bounds(Some(0.8), None), (0.8, 0.8));
    }

    #[test]
    fn a_bare_rotation_initializer_spans_a_whole_turn() {
        // Eleven of the corpus's sixteen are bare, and so is Wallpaper
        // Engine's own preview scene for the element — which would show
        // nothing at all if the absent bounds meant zero.
        let p = preset(r#"{"initializer":[{"name":"rotationrandom"}]}"#);
        let Initializer::RotationRandom { min, max } = &p.initializer[0] else {
            panic!("expected a rotationrandom initializer");
        };
        assert_eq!(spin_bounds(*min, *max, FULL_TURN), (0.0, TAU));
    }

    #[test]
    fn a_named_rotation_bound_still_follows_the_usual_rule() {
        // Three corpus presets pin both ends at half a turn; the value is the
        // shader-author's own rounding of pi, so it is written out rather than
        // reached for as a constant.
        #[expect(clippy::approx_constant, reason = "the corpus's own literal, matched exactly")]
        const HALF_TURN: f32 = 3.141;
        let p = preset(r#"{"initializer":[{"name":"rotationrandom","min":"0 0 3.141","max":"0 0 3.141"}]}"#);
        let Initializer::RotationRandom { min, max } = &p.initializer[0] else {
            panic!("expected a rotationrandom initializer");
        };
        assert_eq!(spin_bounds(*min, *max, FULL_TURN), (HALF_TURN, HALF_TURN));
    }

    #[test]
    fn a_trail_capped_only_by_maxlength_still_draws() {
        // Four corpus renderers name `maxlength` and no `length`; reading
        // `length` alone leaves them with no trail at all.
        let p = preset(r#"{"renderer":[{"name":"spritetrail","maxlength":6}]}"#);
        let Renderer::SpriteTrail { length, maxlength } = &p.renderer[0] else {
            panic!("expected a spritetrail renderer");
        };
        assert!((*length - 0.0).abs() < f32::EPSILON);
        assert!((*maxlength - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn an_emitter_direction_mask_flattens_the_sphere() {
        // `"1 0.2 0"` is a flat horizontal band; ignoring it emits a circle.
        let p = preset(r#"{"emitter":[{"name":"sphererandom","distancemax":100,"directions":"1 0.2 0","rate":1}]}"#);
        let mut rng = seed(1, 7);
        let mut widest: f32 = 0.0;
        let mut tallest: f32 = 0.0;
        for _ in 0..256 {
            let at = emitter_sample(&p.emitter[0], &mut rng);
            widest = widest.max(at.x.abs());
            tallest = tallest.max(at.y.abs());
        }
        assert!(tallest < widest * 0.5, "band was {widest} x {tallest}");
    }

    #[test]
    fn a_colour_ramp_reads_zero_to_one_not_zero_to_255() {
        let p = preset(r#"{"operator":[{"name":"colorchange","startvalue":"1 0 0","endvalue":"1 1 0"}]}"#);
        let Operator::ColorChange { startvalue, endvalue, .. } = &p.operator[0] else {
            panic!("expected a colorchange operator");
        };
        assert!((startvalue.x - 1.0).abs() < f32::EPSILON);
        assert!((endvalue.y - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn unknown_variants_fall_back_rather_than_failing() {
        // All three are real Wallpaper Engine elements we do not simulate —
        // they are in its own `particleelementpreviews` tree — so this is the
        // shape a preset we have not caught up with actually has.
        let p = preset(
            r#"{"emitter":[{"name":"layerimage","rate":1}],
                "operator":[{"name":"boids","cohesionfactor":4}],
                "initializer":[{"name":"hsvcolorrandom","huesteps":6}]}"#,
        );
        assert!(matches!(p.emitter[0], Emitter::Unknown));
        assert!(matches!(p.operator[0], Operator::Unknown));
        assert!(matches!(p.initializer[0], Initializer::Unknown));
    }

    #[test]
    fn emission_slots_track_rate_and_lifetime() {
        let em = Emission { rate: 10.0, maxcount: 40, starttime: 0.0 };
        // Nothing before start.
        assert!(slot_particle(&em, 0, -1.0).is_none());
        // At t=1s, 10 particles have been emitted (indices 0..10); slot 0
        // holds index 0, slot 5 holds index 5, slot 12 is still empty.
        assert_eq!(slot_particle(&em, 0, 1.0).map(|(n, _)| n), Some(0));
        assert_eq!(slot_particle(&em, 5, 1.0).map(|(n, _)| n), Some(5));
        assert!(slot_particle(&em, 12, 1.0).is_none());
        // After a full wrap (t large enough for >40 emissions) slot 0 holds a
        // later index congruent to 0 mod 40.
        let (n, _) = slot_particle(&em, 0, 10.0).unwrap();
        assert_eq!(n % 40, 0);
        assert!(n >= 40);
    }

    #[test]
    fn a_particle_past_its_lifetime_leaves_its_slot_empty() {
        let json = r#"{"emitter":[{"name":"boxrandom","rate":1}],
            "initializer":[{"name":"lifetimerandom","min":2,"max":2}],
            "maxcount":8}"#;
        let p = preset(json);
        let emitter = &p.emitter[0];
        let flow = Perlin::new(1);
        let em = emission(&p, emitter, &InstanceOverride::default(), 0.0);
        // Slot 0's particle (index 0) is born at t=0, lifetime 2s.
        let (_, birth) = slot_particle(&em, 0, 5.0).unwrap();
        // Its replacement (index 8) is not due until t=8, so at t=5 the slot
        // is genuinely empty: age 5 > lifetime 2.
        let r = roll(&p, emitter, &flow, seed(0, 0));
        assert!((5.0 - birth) >= r.lifetime);
    }

    #[test]
    fn simulation_is_deterministic() {
        let p = preset(
            r#"{"emitter":[{"name":"sphererandom","distancemax":100,"rate":5}],
                "initializer":[{"name":"lifetimerandom","min":10,"max":10},
                               {"name":"turbulentvelocityrandom","scale":0.1}],
                "operator":[{"name":"movement","drag":0.5},
                            {"name":"turbulence","scale":0.01,"speedmin":30,"speedmax":50}],
                "maxcount":16}"#,
        );
        let flow = Perlin::new(7);
        let a = simulate(&p, &roll(&p, &p.emitter[0], &flow, seed(3, 4)), &flow, 1.0, 2.5, MAX_STEPS);
        let b = simulate(&p, &roll(&p, &p.emitter[0], &flow, seed(3, 4)), &flow, 1.0, 2.5, MAX_STEPS);
        assert_eq!(a.pos, b.pos);
        assert_eq!(a.alpha, b.alpha);
    }

    #[test]
    fn alphafade_ramps_in_and_out() {
        let p = preset(
            r#"{"emitter":[{"name":"boxrandom","rate":1}],
                "initializer":[{"name":"lifetimerandom","min":10,"max":10}],
                "operator":[{"name":"alphafade","fadeintime":1,"fadeouttime":2}],
                "maxcount":4}"#,
        );
        let flow = Perlin::new(0);
        let r = roll(&p, &p.emitter[0], &flow, seed(0, 0));
        let early = simulate(&p, &r, &flow, 1.0, 0.5, MAX_STEPS).alpha; // half way through fade-in
        let mid = simulate(&p, &r, &flow, 1.0, 5.0, MAX_STEPS).alpha; // fully faded in, not yet out
        let late = simulate(&p, &r, &flow, 1.0, 9.0, MAX_STEPS).alpha; // half way through fade-out
        assert!(early < mid, "{early} !< {mid}");
        assert!(late < mid, "{late} !< {mid}");
        assert!((mid - r.alpha).abs() < 1e-6);
    }
}

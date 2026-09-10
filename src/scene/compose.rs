//! Flattening a scene's image layers into a single still.
//!
//! This is the base composite: every visible image layer decoded, scaled and
//! alpha-blended in file order. It deliberately does *not* run the effect
//! shaders or particle systems — those need the renderer. Puppet-warp layers
//! *are* deformed here (at `time`, or the bind pose when the caller passes 0),
//! since the warp produces the layer image the effect chain then runs over.
//! What it produces is the wallpaper with nothing else moving, and it reports
//! precisely what it left out.

use super::model::{
    self, Blend, Material, Model, Object, Orthographic, Scene, Vec3, base_blend, base_texture,
    is_image, is_particle, is_sound, is_text,
};
use super::{particle, puppet, text};
use crate::{export::Resolution, pkg::Archive, render::bloom, tex};
use anyhow::{Context, Result, bail};
use glam::{Vec2, Vec3 as GVec3};
use image::{Rgba, RgbaImage, imageops};
use std::collections::{HashMap, HashSet};

/// A finished still, plus what could not be represented in it.
pub struct Composite {
    pub image: RgbaImage,
    /// Things the renderer would have drawn and we did not, in the order the
    /// scene lists them. Empty means the still is complete.
    pub omissions: Vec<String>,
}

/// One visible image layer, decoded, scaled to its on-canvas pixel extent and
/// tinted — plus where its top-left corner lands in the output.
///
/// This is the point `scene::render` takes over: it runs the layer's own
/// effect chain over `image` before the layers are flattened together, where
/// `compose::render` just overlays them as they are.
pub struct PreparedLayer<'a> {
    pub object: &'a Object,
    pub image: RgbaImage,
    pub left: i64,
    pub top: i64,
    /// How this layer combines with the layers already drawn beneath it.
    pub blend: Blend,
    /// True when a puppet-warp deformation produced `image`.
    pub warped: bool,
    /// Set when a warp was attempted but failed; the layer falls back to its
    /// flat texture and this is surfaced as an omission.
    pub warp_error: Option<String>,
    /// True for a composition layer, whose `image` is an empty placeholder:
    /// its real input is whatever the frame already holds under `rect`, which
    /// only a compositor that keeps the frame on the GPU can supply.
    pub composition: bool,
}

/// A scene resolved as far as a plain flatten can take it: the output canvas
/// size, its background fill, every visible image layer prepared in draw
/// order, and the notes for whatever still cannot be represented.
pub struct Layered<'a> {
    pub width: u32,
    pub height: u32,
    pub background: Rgba<u8>,
    pub layers: Vec<PreparedLayer<'a>>,
    pub omissions: Vec<String>,
}

/// A scene's time-independent groundwork: every layer's pixels decoded, every
/// puppet mesh parsed, every particle placement resolved. The live simulator
/// builds this once when the window opens and calls `animate` every frame;
/// `prepare` (and the still exporter under it) runs both back to back.
pub struct StaticScene<'a> {
    pub width: u32,
    pub height: u32,
    pub background: Rgba<u8>,
    /// Scene-wide bloom over the finished frame, when `general.bloom` is on —
    /// scene state like the background, not a property of any one layer.
    pub bloom: Option<bloom::Settings>,
    /// `general.hdr`: render every target in floating point.
    pub hdr: bool,
    pub items: Vec<StaticItem<'a>>,
    pub omissions: Vec<String>,
}

/// One scene object, resolved as far as time-independent work allows.
pub enum StaticItem<'a> {
    Image(StaticImage<'a>),
    Puppet(StaticPuppet<'a>),
    Particle(StaticParticle<'a>),
}

/// A plain image layer: texture already scaled to its on-canvas pixel extent
/// and tinted, so every frame draws it unchanged.
pub struct StaticImage<'a> {
    pub object: &'a Object,
    pub image: RgbaImage,
    pub left: i64,
    pub top: i64,
    pub blend: Blend,
    /// Set when this object *is* a puppet but its mesh failed to load, so it
    /// fell back to the flat texture — surfaced as an omission by `animate`.
    pub warp_error: Option<String>,
    /// See `PreparedLayer::composition`.
    pub composition: bool,
    /// Roll about the layer's own centre, in radians — `Anchor::roll`.
    pub roll: f32,
}

/// A puppet-warp layer: the parsed mesh and everything `warp_frame` needs to
/// re-skin it at a new time without touching the archive again.
pub struct StaticPuppet<'a> {
    pub object: &'a Object,
    pub puppet: puppet::Puppet,
    /// Source texture, untinted — the tint is re-applied after each raster.
    pub texture: RgbaImage,
    pub animation: Option<u32>,
    pub rate: f32,
    /// Rest-rectangle extent and top-left, in canvas pixels.
    pub rect_px: (f32, f32),
    pub rect_left: f32,
    pub rect_top: f32,
    pub blend: Blend,
    /// Roll about the layer's own centre, in radians — `Anchor::roll`.
    pub roll: f32,
}

/// A particle system: its resolved placement and preset path, re-simulated
/// each frame.
pub struct StaticParticle<'a> {
    pub object: &'a Object,
    pub preset_path: String,
    pub place: particle::Placement,
    pub blend: Blend,
    /// The system's material refracts, so the layer multiplies the frame
    /// behind it instead of covering it — see `model::base_refracts`.
    pub refract: bool,
}

/// The visible rectangle, in scene units.
///
/// The camera in `scene.json` is the editor's saved viewport, not the render
/// view: both sample wallpapers place their background layer at exactly
/// `(width/2, height/2)` with exactly the projection's extent, which only
/// lines up if the visible rectangle is `(0,0)..(width,height)`. Honouring
/// `camera.eye` instead would slide one sample half a screen off-centre.
struct Canvas {
    ortho: Orthographic,
    /// Output pixels per scene unit.
    scale: f32,
    width: u32,
    height: u32,
}

fn canvas_for(ortho: Orthographic, resolution: Option<Resolution>) -> Result<Canvas> {
    if ortho.width == 0 || ortho.height == 0 {
        bail!("scene has an empty {}x{} canvas", ortho.width, ortho.height);
    }

    // Both sides are image dimensions, always far below f32's 2^24 exact-
    // integer range, so the conversion loses no precision in practice.
    #[expect(clippy::cast_precision_loss, reason = "image dimensions, nowhere near 2^24")]
    let (width, height, scale) = match resolution {
        Some(target) => {
            // Uniform scale from the width; a request with a different aspect
            // ratio letterboxes rather than stretching the art.
            let scale = target.width as f32 / ortho.width as f32;
            (target.width, target.height, scale)
        }
        None => (ortho.width, ortho.height, 1.0),
    };

    Ok(Canvas { ortho, scale, width, height })
}

/// Scene units to output pixels.
///
/// Scene Y points up from the bottom edge; image rows go down. Getting this
/// backwards puts eyelashes at the character's feet, which is how it was
/// confirmed against the sample.
fn to_pixels(canvas: &Canvas, world_x: f32, world_y: f32) -> (f32, f32) {
    #[expect(clippy::cast_precision_loss, reason = "image dimensions, nowhere near 2^24")]
    let height = canvas.ortho.height as f32;
    (world_x * canvas.scale, (height - world_y) * canvas.scale)
}

/// An object's base texture plus the two things the material tells us about
/// how to draw it: the model's puppet path (when it is a warp puppet) and the
/// pass's blend mode.
struct LayerTexture {
    image: RgbaImage,
    puppet: Option<String>,
    blend: Blend,
}

/// Resolve an object's texture through the model and material indirection.
fn layer_texture(archive: &mut Archive, object: &Object) -> Result<Option<LayerTexture>> {
    let Some(model_path) = object.image.as_deref() else {
        return Ok(None);
    };

    let model: Model = serde_json::from_slice(&archive.read(model_path)?)
        .with_context(|| format!("parsing {model_path}"))?;

    let material: Material = serde_json::from_slice(&archive.read(&model.material)?)
        .with_context(|| format!("parsing {}", model.material))?;

    let Some(name) = base_texture(&material) else {
        return Ok(None);
    };

    // Material texture names are relative to `materials/` and carry no
    // extension, so `masks/foo` means `materials/masks/foo.tex`.
    let texture_path = format!("materials/{name}.tex");
    let bytes = archive
        .read(&texture_path)
        .with_context(|| format!("reading the texture of {:?}", object.name))?;

    let texture = tex::parse_bytes(&bytes).with_context(|| format!("parsing {texture_path}"))?;
    let mipmap = tex::largest_mipmap(&texture)?;
    let decoded = tex::decode_rgba(&texture, mipmap)
        .with_context(|| format!("decoding {texture_path}"))?;

    Ok(Some(LayerTexture { image: decoded, puppet: model.puppet, blend: base_blend(&material) }))
}

/// Scale a channel by a `0.0..=1.0` factor and round back to a byte.
///
/// The factor is always clamped before this is called, so the product stays
/// within `0.0..=255.0` and the cast neither truncates nor loses a sign —
/// clippy cannot see that bound, hence the scoped allow.
fn scale_channel(value: u8, factor: f32) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "factor is clamped to 0.0..=1.0, so the product fits in 0..=255"
    )]
    let scaled = (f32::from(value) * factor).round() as u8;
    scaled
}

/// Multiply a layer by its tint, brightness and alpha, in place.
///
/// Skipped entirely when the layer is untinted and opaque, which is the common
/// case and would otherwise mean touching every pixel of an 8K texture. The
/// comparison against `1.0` is exact on purpose: these are serde defaults, not
/// the result of arithmetic, so "equals the default" is the right question,
/// not "is close to".
fn apply_tint(layer: &mut RgbaImage, color: Vec3, brightness: f32, alpha: f32) {
    #[expect(clippy::float_cmp, reason = "checking against an untouched serde default, not a computed value")]
    let neutral = color == Vec3::splat(1.0) && brightness == 1.0 && alpha == 1.0;
    if neutral {
        return;
    }

    let red = (color.x * brightness).clamp(0.0, 1.0);
    let green = (color.y * brightness).clamp(0.0, 1.0);
    let blue = (color.z * brightness).clamp(0.0, 1.0);
    let alpha = alpha.clamp(0.0, 1.0);

    for pixel in layer.pixels_mut() {
        let [r, g, b, a] = pixel.0;
        pixel.0 = [
            scale_channel(r, red),
            scale_channel(g, green),
            scale_channel(b, blue),
            scale_channel(a, alpha),
        ];
    }
}

/// The layer's extent in scene units, from `size` when given.
fn extent(object: &Object, anchor: &Anchor, texture: &RgbaImage) -> (f32, f32) {
    // Texture dimensions, nowhere near f32's 2^24 exact-integer range.
    #[expect(clippy::cast_precision_loss, reason = "image dimensions, nowhere near 2^24")]
    let (width, height) = match object.size {
        Some(size) if size.x > 0.0 && size.y > 0.0 => (size.x, size.y),
        // `autosize` models omit the size and take it from the texture.
        _ => (texture.width() as f32, texture.height() as f32),
    };
    (width * anchor.scale.x, height * anchor.scale.y)
}

/// Where an object actually sits, once its parent chain has been applied.
///
/// A WE scene is a tree, not a flat list: `origin` and `scale` on a child are
/// relative to its parent, and the editor leans on that hard — `scene_example6`
/// parents 31 of its 59 objects, mostly to empty transform nodes that exist
/// only to move a group. Read absolutely, its `rain.mp3` at (-960, -540) under
/// a parent at (960, 540) lands a full screen off-canvas instead of at the
/// scene origin, and its two black `纯色` bars land across the middle of the
/// picture instead of just outside the frame where the author put them.
#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    pub origin: Vec3,
    pub scale: Vec3,
    /// Roll about the view axis, in radians, accumulated down the chain — a
    /// parent's rotation swings its children too. This is `angles.z`; the other
    /// two axes tilt a layer out of the plane and are not applied.
    pub roll: f32,
}

impl Default for Anchor {
    fn default() -> Self {
        Anchor { origin: Vec3::default(), scale: Vec3::splat(1.0), roll: 0.0 }
    }
}

/// Resolve every object's parent chain, index-aligned with `scene.objects`.
///
/// Indices rather than ids because `id` is `#[serde(default)]` and a scene that
/// omits it would collide every such object onto id 0.
fn resolve_anchors(scene: &Scene) -> Vec<Anchor> {
    let mut index_of: HashMap<i64, usize> = HashMap::with_capacity(scene.objects.len());
    for (index, object) in scene.objects.iter().enumerate() {
        index_of.entry(object.id).or_insert(index);
    }

    let mut anchors = Vec::with_capacity(scene.objects.len());
    for (index, object) in scene.objects.iter().enumerate() {
        // Walk to the root, then fold back down. A malformed scene could name
        // itself an ancestor, so stop the walk at the first repeat.
        let mut chain = vec![object];
        let mut seen = HashSet::from([index]);
        let mut cursor = object.parent;
        while let Some(parent_index) = cursor.and_then(|id| index_of.get(&id).copied()) {
            if !seen.insert(parent_index) {
                break;
            }
            let parent = &scene.objects[parent_index];
            chain.push(parent);
            cursor = parent.parent;
        }

        let mut anchor = Anchor::default();
        for node in chain.iter().rev() {
            anchor.origin = Vec3 {
                x: anchor.origin.x + anchor.scale.x * node.origin.x,
                y: anchor.origin.y + anchor.scale.y * node.origin.y,
                z: anchor.origin.z + anchor.scale.z * node.origin.z,
            };
            anchor.scale = Vec3 {
                x: anchor.scale.x * node.scale.x,
                y: anchor.scale.y * node.scale.y,
                z: anchor.scale.z * node.scale.z,
            };
            anchor.roll += node.angles.z;
        }
        anchors.push(anchor);
    }
    anchors
}

/// Intersect a placed rectangle with the canvas, as `(left, top, width,
/// height)`. Both sides are floored at one pixel, so a rectangle entirely off
/// the canvas degenerates rather than inverting.
fn clip_to_canvas(left: i64, top: i64, width: u32, height: u32, canvas_w: u32, canvas_h: u32) -> (i64, i64, u32, u32) {
    let clipped_left = left.max(0);
    let clipped_top = top.max(0);
    let right = (left + i64::from(width)).min(i64::from(canvas_w));
    let bottom = (top + i64::from(height)).min(i64::from(canvas_h));
    let side = |extent: i64| u32::try_from(extent.max(1)).unwrap_or(u32::MAX);
    let (clipped_w, clipped_h) = (side(right - clipped_left), side(bottom - clipped_top));
    (clipped_left, clipped_top, clipped_w, clipped_h)
}

/// A composition layer, if this object is one.
///
/// WE's Composition and Post-Processing layers name a model that ships with the
/// *program* (`models/util/composelayer.json`, `fullscreenlayer.json`), the
/// same way `common.h` and `util/white` do, so the read would fail. They carry
/// no art: their material samples `_rt_FullFrameBuffer` — the frame as
/// composited so far — cropped to the layer's own rectangle, run through the
/// layer's effect chain, and drawn back over that rectangle. A post-processing
/// layer is the same thing at full canvas size.
///
/// The image here is a transparent placeholder that only carries the size; the
/// real input arrives per frame from whatever has the frame on the GPU.
fn static_composition<'a>(
    archive: &Archive,
    canvas: &Canvas,
    object: &'a Object,
    anchor: &Anchor,
) -> Option<StaticImage<'a>> {
    let path = object.image.as_deref()?;
    if archive.contains(path) {
        return None;
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    let fullscreen = match name {
        "fullscreenlayer.json" => true,
        "composelayer.json" | "composelayer_depthtest.json" | "projectlayer.json" => false,
        // Some other model we do not ship and cannot stand in for.
        _ => return None,
    };

    #[expect(clippy::cast_precision_loss, reason = "canvas dimensions, nowhere near 2^24")]
    let full = (canvas.ortho.width as f32, canvas.ortho.height as f32);
    let (extent_x, extent_y) = match object.size {
        Some(size) if !fullscreen && size.x > 0.0 && size.y > 0.0 => {
            (size.x * anchor.scale.x, size.y * anchor.scale.y)
        }
        _ => full,
    };
    let (origin_x, origin_y) = if fullscreen {
        (full.0 / 2.0, full.1 / 2.0)
    } else {
        (anchor.origin.x, anchor.origin.y)
    };

    let (rect_left, rect_top) = to_pixels(canvas, origin_x - extent_x / 2.0, origin_y + extent_y / 2.0);
    let (width, height) = to_pixel_size(extent_x * canvas.scale, extent_y * canvas.scale);

    // Clipped to the canvas: a composition layer can be far larger than the
    // screen (`scene_example4`'s cloud layer is 3.07x the canvas in Y), and
    // only the part over the canvas has any frame under it to read. Allocating
    // the full rectangle would mean a 4k x 6.6k render target per pass.
    let (left, top, width, height) = clip_to_canvas(
        round_to_i64(rect_left),
        round_to_i64(rect_top),
        width,
        height,
        canvas.width,
        canvas.height,
    );

    Some(StaticImage {
        object,
        image: RgbaImage::new(width, height),
        left,
        top,
        blend: Blend::Over,
        warp_error: None,
        composition: true,
        roll: anchor.roll,
    })
}

/// A text layer, rendered to a raster the ordinary image path can carry.
///
/// The box is the object's own `size · scale`, and the glyphs are drawn at
/// `pointsize` scaled the same way — a text layer scales like every other
/// layer, so its type size has to travel with it.
fn static_text<'a>(
    archive: &mut Archive,
    canvas: &Canvas,
    object: &'a Object,
    anchor: &Anchor,
) -> Result<StaticImage<'a>, String> {
    let size = object.size.ok_or_else(|| "text layer has no size".to_string())?;
    let (extent_x, extent_y) = (size.x * anchor.scale.x, size.y * anchor.scale.y);
    if extent_x <= 0.0 || extent_y <= 0.0 {
        return Err("text layer has an empty box".into());
    }

    let (rect_left, rect_top) = to_pixels(
        canvas,
        anchor.origin.x - extent_x / 2.0,
        anchor.origin.y + extent_y / 2.0,
    );
    let (width, height) = to_pixel_size(extent_x * canvas.scale, extent_y * canvas.scale);
    let rendered = text::render(archive, None, object, (width, height))?;
    Ok(StaticImage {
        object,
        image: rendered.image,
        left: round_to_i64(rect_left),
        top: round_to_i64(rect_top),
        blend: Blend::Over,
        warp_error: None,
        composition: false,
        roll: anchor.roll,
    })
}

/// A solid-colour layer, if this object is one.
///
/// `models/util/solidlayer.json` is another engine-provided model, and the
/// thinnest one: its material is a single `flat` pass, whose fragment shader is
/// `gl_FragColor = vec4(g_Color, g_Alpha)` over a translucent-blended
/// rectangle. So the layer is its own rect filled with the object's tint, and
/// the ordinary image path — effect chain included — takes it from there.
/// `scene_example6` uses four: two black bars, and two that are the backdrop an
/// audio-bars effect draws over.
fn static_solid<'a>(
    archive: &Archive,
    canvas: &Canvas,
    object: &'a Object,
    anchor: &Anchor,
) -> Option<StaticImage<'a>> {
    let path = object.image.as_deref()?;
    if archive.contains(path) {
        return None;
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    if !matches!(name, "solidlayer.json" | "solidlayer_depthtest.json") {
        return None;
    }

    let size = object.size?;
    if size.x <= 0.0 || size.y <= 0.0 {
        return None;
    }
    let (extent_x, extent_y) = (size.x * anchor.scale.x, size.y * anchor.scale.y);
    let (rect_left, rect_top) = to_pixels(
        canvas,
        anchor.origin.x - extent_x / 2.0,
        anchor.origin.y + extent_y / 2.0,
    );
    let (width, height) = to_pixel_size(extent_x * canvas.scale, extent_y * canvas.scale);

    let mut image = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));
    apply_tint(&mut image, object.color, object.brightness, object.alpha);

    Some(StaticImage {
        object,
        image,
        left: round_to_i64(rect_left),
        top: round_to_i64(rect_top),
        blend: Blend::Over,
        warp_error: None,
        composition: false,
        roll: anchor.roll,
    })
}

/// Note anything about an object that this composite cannot represent.
fn omissions_for(object: &Object) -> Vec<String> {
    let mut notes = Vec::new();
    let name = model::label(object);

    if is_particle(object) {
        notes.push(format!("{name}: particle system not rendered"));
    }
    if !object.animationlayers.is_empty() {
        notes.push(format!("{name}: keyframe animation not applied"));
    }
    if is_image(object) {
        let effects = model::visible_effects(object).count();
        if effects > 0 {
            notes.push(format!("{name}: {effects} effect(s) not applied"));
        }
        // Only an out-of-plane tilt is unrepresentable now; `roll` is applied
        // by the compositor.
        if object.angles.x != 0.0 || object.angles.y != 0.0 {
            notes.push(format!("{name}: out-of-plane rotation not applied"));
        }
    }
    notes
}

/// The scene's bloom settings, when it asks for bloom at all.
///
/// An HDR scene tunes the `bloomhdr*` set and leaves the plain pair at their
/// defaults, so which pair to read follows `hdr` — `scene_example8` writes
/// 0.12/0.55 for HDR and stock 2.0/0.65 for the other, and reading the wrong
/// one drives the bloom sixteen times too hard.
fn scene_bloom(general: &model::General) -> Option<bloom::Settings> {
    general.bloom.then(|| {
        let (strength, threshold) = if general.hdr {
            (general.bloomhdrstrength, general.bloomhdrthreshold)
        } else {
            (general.bloomstrength, general.bloomthreshold)
        };
        bloom::Settings {
            strength,
            threshold,
            tint: [general.bloomtint.x, general.bloomtint.y, general.bloomtint.z],
            // The plain path has no spread of its own; WE fixes it at a
            // quarter-then-eighth pair, which is two levels at full bleed.
            scatter: if general.hdr { general.bloomhdrscatter } else { 1.0 },
            iterations: if general.hdr { general.bloomhdriterations } else { 2 },
        }
    })
}

/// The scene's background fill: the clear colour when clearing is on, fully
/// transparent when it is off.
fn background_pixel(scene: &Scene) -> Rgba<u8> {
    if scene.general.clearenabled {
        let clear = scene.general.clearcolor;
        Rgba([
            scale_channel(255, clear.x.clamp(0.0, 1.0)),
            scale_channel(255, clear.y.clamp(0.0, 1.0)),
            scale_channel(255, clear.z.clamp(0.0, 1.0)),
            255,
        ])
    } else {
        Rgba([0, 0, 0, 0])
    }
}

/// Deformation padding, as a fraction of the layer's extent on each side — the
/// warped mesh can bend outside its rest rectangle and the raster must cover
/// that. Puppet motion in practice is a few percent; 30% is comfortably safe.
const WARP_PAD: f32 = 0.3;

/// Round a scene-unit extent to a whole pixel count, floored at 1.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded and floored at 1.0 before the cast"
)]
fn to_pixel_size(width: f32, height: f32) -> (u32, u32) {
    (width.round().max(1.0) as u32, height.round().max(1.0) as u32)
}

/// The layer's source texture scaled to its on-canvas pixel extent and tinted.
fn scaled_tinted(object: &Object, texture: RgbaImage, pixel_width: u32, pixel_height: u32) -> RgbaImage {
    let mut image = if texture.width() == pixel_width && texture.height() == pixel_height {
        texture
    } else {
        imageops::resize(&texture, pixel_width, pixel_height, imageops::FilterType::Lanczos3)
    };
    apply_tint(&mut image, object.color, object.brightness, object.alpha);
    image
}

/// A wallpaper canvas is at most a few tens of thousands of pixels wide.
#[expect(clippy::cast_possible_truncation, reason = "canvas coordinates, nowhere near i64's range")]
fn round_to_i64(x: f32) -> i64 {
    x.round() as i64
}

/// Decode one image object's texture and resolve it as far as time-independent
/// work allows: a `Puppet` item carrying the parsed mesh when it is a warp
/// puppet whose mesh loads, an `Image` item otherwise (including a puppet whose
/// mesh failed to load, which falls back to the flat texture). `None` when the
/// object resolves to no texture (a solid-colour or effect-only layer).
fn static_image_or_puppet<'a>(
    archive: &mut Archive,
    canvas: &Canvas,
    object: &'a Object,
    anchor: &Anchor,
) -> Result<Option<StaticItem<'a>>> {
    let Some(LayerTexture { image: texture, puppet: puppet_path, blend }) =
        layer_texture(archive, object)?
    else {
        return Ok(None);
    };

    let (extent_x, extent_y) = extent(object, anchor, &texture);
    // `origin` is the centre of the layer, and the top edge is the one with
    // the larger scene Y.
    let (rect_left, rect_top) = to_pixels(
        canvas,
        anchor.origin.x - extent_x / 2.0,
        anchor.origin.y + extent_y / 2.0,
    );

    // A puppet layer replaces the flat scale with a skinned-mesh deformation
    // re-run every frame; keep the parsed mesh rather than the flat image.
    let mut warp_error = None;
    if let (Some(path), Some(clip)) = (puppet_path.as_deref(), model::visible_animation_layer(object)) {
        match load_puppet_mesh(archive, path) {
            Ok(puppet) => {
                return Ok(Some(StaticItem::Puppet(StaticPuppet {
                    object,
                    puppet,
                    texture,
                    animation: clip.animation,
                    rate: clip.rate,
                    rect_px: (extent_x * canvas.scale, extent_y * canvas.scale),
                    rect_left,
                    rect_top,
                    blend,
                    roll: anchor.roll,
                })));
            }
            Err(error) => warp_error = Some(format!("{error:#}")),
        }
    }

    let (pixel_width, pixel_height) = to_pixel_size(extent_x * canvas.scale, extent_y * canvas.scale);
    let image = scaled_tinted(object, texture, pixel_width, pixel_height);
    Ok(Some(StaticItem::Image(StaticImage {
        object,
        image,
        left: round_to_i64(rect_left),
        top: round_to_i64(rect_top),
        blend,
        warp_error,
        composition: false,
        roll: anchor.roll,
    })))
}

/// Read and parse a puppet `.mdl`, rejecting an empty mesh.
fn load_puppet_mesh(archive: &mut Archive, puppet_path: &str) -> Result<puppet::Puppet> {
    let data = archive.read(puppet_path).with_context(|| format!("reading {puppet_path}"))?;
    let puppet = puppet::parse(&data).with_context(|| format!("parsing {puppet_path}"))?;
    if puppet.vertices.is_empty() || puppet.triangles.len() < 3 {
        bail!("puppet mesh is empty");
    }
    Ok(puppet)
}

/// Skin `item`'s mesh at `time` and rasterize it through the source texture —
/// the whole of a puppet layer's per-frame work, touching no files. Returns the
/// deformed image (covering the rest rectangle plus a margin) and its canvas
/// top-left.
pub fn warp_frame(item: &StaticPuppet, time: f32) -> (RgbaImage, i64, i64) {
    let skins = puppet::skin_transforms(&item.puppet, item.animation, time, item.rate);
    let positions = puppet::deform(&item.puppet, &skins);
    let (u_slope, u_intercept, v_slope, v_intercept) = puppet::uv_fit(&item.puppet);

    let rect_px = item.rect_px;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "layer extents are at most a few thousand px")]
    let (out_w, out_h) = (
        ((rect_px.0 * (1.0 + 2.0 * WARP_PAD)).ceil() as u32).clamp(1, 16384),
        ((rect_px.1 * (1.0 + 2.0 * WARP_PAD)).ceil() as u32).clamp(1, 16384),
    );

    // A deformed vertex lands in the same texture rectangle a flat layer would
    // place it in (via the mesh's own vertex->UV fit), offset by the pad.
    let place = |x: f32, y: f32| -> (f32, f32) {
        let u = u_slope * x + u_intercept;
        let v = v_slope * y + v_intercept;
        ((WARP_PAD + u) * rect_px.0, (WARP_PAD + v) * rect_px.1)
    };

    let mut image = puppet::rasterize(&item.puppet, &positions, &item.texture, out_w, out_h, place);
    apply_tint(&mut image, item.object.color, item.object.brightness, item.object.alpha);

    let left = round_to_i64(item.rect_left - WARP_PAD * rect_px.0);
    let top = round_to_i64(item.rect_top - WARP_PAD * rect_px.1);
    (image, left, top)
}

/// Resolve a scene into its canvas plus every visible image layer, prepared
/// for placement but not yet flattened. `compose::render` overlays these
/// straight; `scene::render` runs each layer's effect chain first. Puppet
/// layers are deformed at `time` (pass 0 for the bind pose / a plain still).
///
/// This is `prepare_static` followed by `animate`; the live simulator splits
/// the two so the decode/parse half runs once and only `animate` runs per
/// frame.
pub fn prepare<'a>(
    archive: &mut Archive,
    scene: &'a Scene,
    resolution: Option<Resolution>,
    time: f32,
) -> Result<Layered<'a>> {
    let static_scene = prepare_static(archive, scene, resolution)?;
    Ok(animate(archive, &static_scene, time))
}

/// The time-independent half of `prepare`: decode every layer's texture, parse
/// every puppet mesh, resolve every particle placement. `animate` turns the
/// result into a frame.
pub fn prepare_static<'a>(
    archive: &mut Archive,
    scene: &'a Scene,
    resolution: Option<Resolution>,
) -> Result<StaticScene<'a>> {
    let ortho = scene
        .general
        .orthographic
        .context("scene has no orthographic projection, so it is not a flat wallpaper")?;
    let canvas = canvas_for(ortho, resolution)?;

    let mut items = Vec::new();
    let mut omissions = Vec::new();
    let anchors = resolve_anchors(scene);

    for (object, anchor) in scene.objects.iter().zip(&anchors) {
        if !object.visible || is_sound(object) {
            continue;
        }
        if is_text(object) {
            match static_text(archive, &canvas, object, anchor) {
                Ok(item) => {
                    omissions.extend(omissions_for(object));
                    items.push(StaticItem::Image(item));
                }
                Err(reason) => {
                    omissions.push(format!("{}: text layer skipped ({reason})", model::label(object)));
                }
            }
            continue;
        }
        if is_image(object) {
            // A composition layer has no texture to decode: its input is the
            // frame beneath it, which only the GPU compositors can hand it.
            if let Some(item) = static_composition(archive, &canvas, object, anchor) {
                omissions.extend(omissions_for(object));
                items.push(StaticItem::Image(item));
                continue;
            }
            // A solid-colour layer likewise names a model we do not have, but
            // its content is fully determined by the object's own tint.
            if let Some(item) = static_solid(archive, &canvas, object, anchor) {
                omissions.extend(omissions_for(object));
                items.push(StaticItem::Image(item));
                continue;
            }
            if !archive.contains(object.image.as_deref().unwrap_or_default()) {
                omissions.push(format!(
                    "{}: layer skipped, {} is not in the package",
                    model::label(object),
                    object.image.as_deref().unwrap_or_default()
                ));
                continue;
            }
        }
        omissions.extend(omissions_for(object));

        if is_particle(object) {
            match static_particle(archive, &canvas, object, anchor) {
                Ok((item, unsupported)) => {
                    reconcile_particle_note(&mut omissions, object, Ok(&unsupported));
                    items.push(StaticItem::Particle(item));
                }
                Err(error) => reconcile_particle_note(&mut omissions, object, Err(&error)),
            }
            continue;
        }

        if !is_image(object) {
            continue;
        }
        if let Some(item) = static_image_or_puppet(archive, &canvas, object, anchor)? {
            items.push(item);
        }
    }

    if items.is_empty() {
        bail!("scene has no visible image layers to draw");
    }

    Ok(StaticScene {
        width: canvas.width,
        height: canvas.height,
        background: background_pixel(scene),
        bloom: scene_bloom(&scene.general),
        hdr: scene.general.hdr,
        items,
        omissions,
    })
}

/// Produce the frame at `time`: puppet layers re-skinned, particle systems
/// re-simulated, plain image layers passed straight through. Omission notes
/// that `prepare_static` could not settle (a warp that will fail, a particle
/// preset feature) are reconciled here against what actually happened.
pub fn animate<'a>(archive: &mut Archive, static_scene: &StaticScene<'a>, time: f32) -> Layered<'a> {
    let mut omissions = static_scene.omissions.clone();
    let mut layers = Vec::with_capacity(static_scene.items.len());

    for item in &static_scene.items {
        let layer = match item {
            StaticItem::Image(image) => PreparedLayer {
                object: image.object,
                image: image.image.clone(),
                left: image.left,
                top: image.top,
                blend: image.blend,
                warped: false,
                warp_error: image.warp_error.clone(),
                composition: image.composition,
            },
            StaticItem::Puppet(puppet) => {
                let (image, left, top) = warp_frame(puppet, time);
                PreparedLayer {
                    object: puppet.object,
                    image,
                    left,
                    top,
                    blend: puppet.blend,
                    warped: true,
                    warp_error: None,
                    composition: false,
                }
            }
            StaticItem::Particle(particle) => {
                match particle::render_system(archive, &particle.preset_path, &particle.place, time) {
                    Ok(rendered) => {
                        reconcile_particle_note(&mut omissions, particle.object, Ok(&rendered.unsupported));
                        PreparedLayer {
                            object: particle.object,
                            image: rendered.image,
                            left: 0,
                            top: 0,
                            blend: particle.blend,
                            warped: false,
                            warp_error: None,
                            composition: false,
                        }
                    }
                    Err(error) => {
                        reconcile_particle_note(&mut omissions, particle.object, Err(&error));
                        continue;
                    }
                }
            }
        };
        reconcile_warp_note(&mut omissions, &layer);
        layers.push(layer);
    }

    Layered {
        width: static_scene.width,
        height: static_scene.height,
        background: static_scene.background,
        layers,
        omissions,
    }
}

/// Resolve a particle object's placement and blend, and run one simulation at
/// t=0 to learn which preset features it uses that we do not simulate.
fn static_particle<'a>(
    archive: &mut Archive,
    canvas: &Canvas,
    object: &'a Object,
    anchor: &Anchor,
) -> Result<(StaticParticle<'a>, Vec<String>)> {
    let preset_path = object
        .particle
        .as_deref()
        .context("particle object names no preset")?
        .to_string();

    let (origin_x, origin_y) = to_pixels(canvas, anchor.origin.x, anchor.origin.y);
    let place = particle::Placement {
        origin_px: Vec2::new(origin_x, origin_y),
        scale: Vec2::new(anchor.scale.x, anchor.scale.y),
        px_per_unit: canvas.scale,
        canvas_px: (canvas.width, canvas.height),
        tint: GVec3::new(
            object.color.x * object.brightness,
            object.color.y * object.brightness,
            object.color.z * object.brightness,
        ),
        alpha: object.alpha,
        overrides: object.instanceoverride.unwrap_or_default(),
        max_sim_steps: particle::EXACT_SIM_STEPS,
        roll: (anchor.roll.cos(), anchor.roll.sin()),
    };

    let blend = particle::layer_blend(archive, &preset_path);
    let rendered = particle::render_system(archive, &preset_path, &place, 0.0)
        .with_context(|| format!("simulating {preset_path}"))?;

    let refract = particle::layer_refracts(archive, &preset_path);
    Ok((StaticParticle { object, preset_path, place, blend, refract }, rendered.unsupported))
}

/// `omissions_for` flags every object with animation layers as "keyframe
/// animation not applied". Once the layer is prepared we know better: drop the
/// note when the warp ran, or swap in the reason when it was attempted and
/// failed.
fn reconcile_warp_note(omissions: &mut Vec<String>, layer: &PreparedLayer) {
    let stale = format!("{}: keyframe animation not applied", model::label(layer.object));
    if layer.warped {
        omissions.retain(|note| *note != stale);
    } else if let Some(reason) = &layer.warp_error
        && let Some(note) = omissions.iter_mut().find(|note| **note == stale)
    {
        *note = format!("{}: puppet warp skipped ({reason})", model::label(layer.object));
    }
}

/// `omissions_for` flags every particle object as "particle system not
/// rendered". Replace that once we've tried: drop it when the system rendered
/// clean, or swap in what was skipped or why it failed.
fn reconcile_particle_note(
    omissions: &mut Vec<String>,
    object: &Object,
    outcome: Result<&[String], &anyhow::Error>,
) {
    let name = model::label(object);
    let stale = format!("{name}: particle system not rendered");
    let replacement = match outcome {
        Ok([]) => None,
        Ok(unsupported) => {
            Some(format!("{name}: particle system rendered without {}", unsupported.join(", ")))
        }
        Err(error) => Some(format!("{name}: particle system skipped ({error:#})")),
    };
    match replacement {
        Some(note) => {
            if let Some(slot) = omissions.iter_mut().find(|entry| **entry == stale) {
                *slot = note;
            } else {
                omissions.push(note);
            }
        }
        None => omissions.retain(|entry| *entry != stale),
    }
}

/// Draw every prepared layer onto the background in order, each under its own
/// blend mode. This is the one place layer compositing happens — `scene::render`
/// swaps processed images into the layers first, then calls here.
pub fn flatten(layered: &Layered) -> RgbaImage {
    let mut output = RgbaImage::from_pixel(layered.width, layered.height, layered.background);
    for layer in &layered.layers {
        // A composition layer's pixels are the frame beneath it, which a flat
        // overlay has no way to feed back through an effect chain.
        if layer.composition {
            continue;
        }
        blit(&mut output, &layer.image, layer.left, layer.top, layer.blend);
    }
    output
}

/// `dst + src·factor`, saturating at 255.
fn add_channel(dst: u8, src: u8, factor: f32) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0.0..=255.0 immediately before the cast"
    )]
    let sum = (f32::from(dst) + f32::from(src) * factor).round().clamp(0.0, 255.0) as u8;
    sum
}

/// Composite `layer` onto `canvas` at `(left, top)` under `blend`. `Over` is
/// exactly `image::imageops::overlay`; `Add` accumulates `dst + src·srcAlpha`
/// per colour channel and lifts the canvas alpha to the brighter of the two,
/// so an additive glow still shows over a transparent background.
fn blit(canvas: &mut RgbaImage, layer: &RgbaImage, left: i64, top: i64, blend: Blend) {
    if blend == Blend::Over {
        imageops::overlay(canvas, layer, left, top);
        return;
    }

    let (canvas_w, canvas_h) = (i64::from(canvas.width()), i64::from(canvas.height()));
    for (lx, ly, pixel) in layer.enumerate_pixels() {
        let (x, y) = (left + i64::from(lx), top + i64::from(ly));
        if x < 0 || y < 0 || x >= canvas_w || y >= canvas_h {
            continue;
        }
        #[expect(clippy::cast_sign_loss, clippy::cast_possible_truncation, reason = "bounds-checked to 0..width/height just above")]
        let target = canvas.get_pixel_mut(x as u32, y as u32);
        let [sr, sg, sb, sa] = pixel.0;
        let factor = f32::from(sa) / 255.0;
        target.0 = [
            add_channel(target.0[0], sr, factor),
            add_channel(target.0[1], sg, factor),
            add_channel(target.0[2], sb, factor),
            target.0[3].max(sa),
        ];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::model::Orthographic;

    fn canvas(width: u32, height: u32, resolution: Option<Resolution>) -> Canvas {
        canvas_for(Orthographic { width, height }, resolution).unwrap()
    }

    /// The anchor of a parentless object: its own transform, unmodified.
    fn own_anchor(object: &Object) -> Anchor {
        Anchor {
            origin: object.origin,
            scale: object.scale,
            roll: object.angles.z,
        }
    }

    fn scene_of(objects: &str) -> Scene {
        serde_json::from_str(&format!(
            r#"{{"general":{{"orthogonalprojection":{{"width":1920,"height":1080}}}},"objects":{objects}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn a_child_origin_is_relative_to_its_parent() {
        // `scene_example6` parks its sound objects at the scene origin by
        // hanging a (-960, -540) child off a (960, 540) parent; read
        // absolutely that lands a full screen off-canvas instead.
        let scene = scene_of(
            r#"[{"id":1,"origin":"960.0 540.0 0.0"},
                {"id":2,"parent":1,"origin":"-960.0 -540.0 0.0"}]"#,
        );
        let anchors = resolve_anchors(&scene);
        assert_eq!(anchors[1].origin, Vec3 { x: 0.0, y: 0.0, z: 0.0 });
    }

    #[test]
    fn a_parent_scale_multiplies_both_the_child_offset_and_its_own() {
        let scene = scene_of(
            r#"[{"id":1,"origin":"100.0 0.0 0.0","scale":"2.0 2.0 1.0"},
                {"id":2,"parent":1,"origin":"50.0 0.0 0.0","scale":"3.0 3.0 1.0"}]"#,
        );
        let anchors = resolve_anchors(&scene);
        // 100 + 2 * 50, and the scales compound.
        assert_eq!(anchors[1].origin.x, 200.0);
        assert_eq!(anchors[1].scale.x, 6.0);
    }

    #[test]
    fn roll_accumulates_down_the_chain() {
        // A parent's rotation swings its children too, so a child under a
        // rotated parent rolls by the sum of both.
        let scene = scene_of(
            r#"[{"id":1,"angles":"0.0 0.0 0.5"},
                {"id":2,"parent":1,"name":"child","angles":"0.0 0.0 0.25","image":"models/a.json"}]"#,
        );
        let anchors = resolve_anchors(&scene);
        assert!((anchors[0].roll - 0.5).abs() < 1e-6);
        assert!((anchors[1].roll - 0.75).abs() < 1e-6, "got {}", anchors[1].roll);
    }

    #[test]
    fn only_an_out_of_plane_tilt_is_still_reported() {
        // Roll is applied by the compositor now; a tilt about x or y is not.
        let flat: Object =
            serde_json::from_str(r#"{"name":"a","image":"models/a.json","angles":"0.0 0.0 0.5"}"#).unwrap();
        assert_eq!(omissions_for(&flat), Vec::<String>::new());

        let tilted: Object =
            serde_json::from_str(r#"{"name":"b","image":"models/a.json","angles":"0.3 0.0 0.0"}"#).unwrap();
        assert_eq!(omissions_for(&tilted), vec!["b: out-of-plane rotation not applied"]);
    }

    #[test]
    fn an_object_cycle_terminates_rather_than_hanging() {
        let scene = scene_of(
            r#"[{"id":1,"parent":2,"origin":"1.0 0.0 0.0"},
                {"id":2,"parent":1,"origin":"10.0 0.0 0.0"}]"#,
        );
        let anchors = resolve_anchors(&scene);
        assert_eq!(anchors.len(), 2);
    }

    #[test]
    fn scene_y_is_measured_up_from_the_bottom() {
        let canvas = canvas(3840, 2160, None);
        // The bottom edge of the scene is the last row of the image.
        assert_eq!(to_pixels(&canvas, 0.0, 0.0), (0.0, 2160.0));
        assert_eq!(to_pixels(&canvas, 0.0, 2160.0), (0.0, 0.0));
        assert_eq!(to_pixels(&canvas, 1920.0, 1080.0), (1920.0, 1080.0));
    }

    #[test]
    fn a_target_resolution_scales_scene_units() {
        let canvas = canvas(3840, 2160, Some(Resolution { width: 1920, height: 1080 }));
        assert_eq!(canvas.scale, 0.5);
        assert_eq!(to_pixels(&canvas, 3840.0, 0.0), (1920.0, 1080.0));
    }

    #[test]
    fn an_empty_canvas_is_rejected() {
        assert!(canvas_for(Orthographic { width: 0, height: 1080 }, None).is_err());
        assert!(canvas_for(Orthographic { width: 1920, height: 0 }, None).is_err());
    }

    #[test]
    fn scale_multiplies_the_declared_size() {
        let object: Object = serde_json::from_str(
            r#"{"size":"1100.0 730.0 0.0","scale":"1.07707 1.07707 1.0"}"#,
        )
        .unwrap();
        let texture = RgbaImage::new(4, 4);
        let (width, height) = extent(&object, &own_anchor(&object), &texture);
        assert!((width - 1184.777).abs() < 0.01, "got {width}");
        assert!((height - 786.261).abs() < 0.01, "got {height}");
    }

    #[test]
    fn a_sizeless_object_takes_its_extent_from_the_texture() {
        let object: Object = serde_json::from_str(r#"{"image":"models/a.json"}"#).unwrap();
        let texture = RgbaImage::new(512, 256);
        assert_eq!(extent(&object, &own_anchor(&object), &texture), (512.0, 256.0));
    }

    #[test]
    fn tinting_is_skipped_when_it_would_change_nothing() {
        let mut layer = RgbaImage::from_pixel(1, 1, Rgba([10, 20, 30, 40]));
        apply_tint(&mut layer, Vec3::splat(1.0), 1.0, 1.0);
        assert_eq!(layer.get_pixel(0, 0).0, [10, 20, 30, 40]);
    }

    #[test]
    fn alpha_and_brightness_multiply_through() {
        let mut layer = RgbaImage::from_pixel(1, 1, Rgba([200, 100, 50, 200]));
        apply_tint(&mut layer, Vec3::splat(1.0), 0.5, 0.5);
        assert_eq!(layer.get_pixel(0, 0).0, [100, 50, 25, 100]);
    }

    #[test]
    fn particles_and_effects_are_reported_rather_than_silently_dropped() {
        let particle: Object =
            serde_json::from_str(r#"{"name":"Ember","particle":"particles/presets/ember.json"}"#)
                .unwrap();
        assert_eq!(
            omissions_for(&particle),
            vec!["Ember: particle system not rendered"]
        );

        let effected: Object = serde_json::from_str(
            r#"{"name":"bg","image":"models/a.json",
                "effects":[{"file":"e.json","visible":true},
                           {"file":"f.json","visible":false}]}"#,
        )
        .unwrap();
        assert_eq!(
            omissions_for(&effected),
            vec!["bg: 1 effect(s) not applied"]
        );
    }

    #[test]
    fn over_blit_matches_imageops_overlay() {
        let mut a = RgbaImage::from_pixel(2, 2, Rgba([10, 20, 30, 255]));
        let mut b = a.clone();
        let layer = RgbaImage::from_pixel(2, 2, Rgba([200, 100, 50, 128]));
        blit(&mut a, &layer, 0, 0, Blend::Over);
        imageops::overlay(&mut b, &layer, 0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn additive_blit_accumulates_and_saturates() {
        let mut canvas = RgbaImage::from_pixel(1, 1, Rgba([100, 0, 200, 0]));
        // src 128 at alpha 128: contributes 128 * (128/255) ≈ 64.25 per channel.
        blit(&mut canvas, &RgbaImage::from_pixel(1, 1, Rgba([128, 128, 128, 128])), 0, 0, Blend::Add);
        assert_eq!(canvas.get_pixel(0, 0).0, [164, 64, 255, 128]);
    }

    #[test]
    fn additive_blit_clips_to_the_canvas() {
        let mut canvas = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 255]));
        // Placed one pixel off the top-left: only the (1,1) texel lands.
        blit(&mut canvas, &RgbaImage::from_pixel(2, 2, Rgba([50, 50, 50, 255])), -1, -1, Blend::Add);
        assert_eq!(canvas.get_pixel(0, 0).0, [50, 50, 50, 255]);
        assert_eq!(canvas.get_pixel(1, 0).0, [0, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(1, 1).0, [0, 0, 0, 255]);
    }

    #[test]
    fn a_plain_layer_has_nothing_to_report() {
        let object: Object = serde_json::from_str(r#"{"name":"bg","image":"models/a.json"}"#).unwrap();
        assert_eq!(omissions_for(&object), Vec::<String>::new());
    }
}

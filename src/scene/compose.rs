//! Flattening a scene's image layers into a single still.
//!
//! This is the base composite: every visible image layer decoded, scaled and
//! alpha-blended in file order. It deliberately does *not* run the effect
//! shaders, particle systems or keyframe animation — those need the renderer.
//! What it produces is the wallpaper with nothing moving, which for a still is
//! most of the way there, and it reports precisely what it left out.

use super::model::{
    self, Material, Model, Object, Orthographic, Scene, Vec3, base_texture, is_image, is_particle,
    is_sound,
};
use crate::{export::Resolution, pkg::Archive, tex};
use anyhow::{Context, Result, bail};
use image::{Rgba, RgbaImage, imageops};

/// A finished still, plus what could not be represented in it.
pub struct Composite {
    pub image: RgbaImage,
    /// Things the renderer would have drawn and we did not, in the order the
    /// scene lists them. Empty means the still is complete.
    pub omissions: Vec<String>,
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

/// Resolve an object's texture through the model and material indirection.
fn layer_texture(archive: &mut Archive, object: &Object) -> Result<Option<RgbaImage>> {
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

    Ok(Some(decoded))
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
fn extent(object: &Object, texture: &RgbaImage) -> (f32, f32) {
    // Texture dimensions, nowhere near f32's 2^24 exact-integer range.
    #[expect(clippy::cast_precision_loss, reason = "image dimensions, nowhere near 2^24")]
    let (width, height) = match object.size {
        Some(size) if size.x > 0.0 && size.y > 0.0 => (size.x, size.y),
        // `autosize` models omit the size and take it from the texture.
        _ => (texture.width() as f32, texture.height() as f32),
    };
    (width * object.scale.x, height * object.scale.y)
}

/// Note anything about an object that this composite cannot represent.
fn omissions_for(object: &Object) -> Vec<String> {
    let mut notes = Vec::new();
    let name = if object.name.is_empty() {
        format!("object {}", object.id)
    } else {
        object.name.clone()
    };

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
        if object.angles != Vec3::default() {
            notes.push(format!("{name}: rotation not applied"));
        }
    }
    notes
}

/// Flatten every visible image layer into one image.
pub fn render(
    archive: &mut Archive,
    scene: &Scene,
    resolution: Option<Resolution>,
) -> Result<Composite> {
    let ortho = scene
        .general
        .orthographic
        .context("scene has no orthographic projection, so it is not a flat wallpaper")?;
    let canvas = canvas_for(ortho, resolution)?;

    let background = if scene.general.clearenabled {
        let clear = scene.general.clearcolor;
        Rgba([
            scale_channel(255, clear.x.clamp(0.0, 1.0)),
            scale_channel(255, clear.y.clamp(0.0, 1.0)),
            scale_channel(255, clear.z.clamp(0.0, 1.0)),
            255,
        ])
    } else {
        Rgba([0, 0, 0, 0])
    };
    let mut output = RgbaImage::from_pixel(canvas.width, canvas.height, background);

    let mut omissions = Vec::new();
    let mut drawn = 0usize;

    for object in &scene.objects {
        if !object.visible || is_sound(object) {
            continue;
        }
        omissions.extend(omissions_for(object));

        if !is_image(object) {
            continue;
        }

        let Some(texture) = layer_texture(archive, object)? else {
            continue;
        };

        let (extent_x, extent_y) = extent(object, &texture);
        // Rounded and floored at 1.0 above, so this always lands in u32's
        // range for any wallpaper-sized canvas.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rounded and floored at 1.0 above"
        )]
        let (pixel_width, pixel_height) = (
            (extent_x * canvas.scale).round().max(1.0) as u32,
            (extent_y * canvas.scale).round().max(1.0) as u32,
        );

        let mut layer = if texture.width() == pixel_width && texture.height() == pixel_height {
            texture
        } else {
            imageops::resize(
                &texture,
                pixel_width,
                pixel_height,
                imageops::FilterType::Lanczos3,
            )
        };
        apply_tint(&mut layer, object.color, object.brightness, object.alpha);

        // `origin` is the centre of the layer, and the top edge is the one with
        // the larger scene Y.
        let (left, top) = to_pixels(
            &canvas,
            object.origin.x - extent_x / 2.0,
            object.origin.y + extent_y / 2.0,
        );
        // A wallpaper canvas is at most a few tens of thousands of pixels
        // wide, nowhere near i64's range.
        #[expect(clippy::cast_possible_truncation, reason = "canvas coordinates, nowhere near i64's range")]
        let (left, top) = (left.round() as i64, top.round() as i64);
        imageops::overlay(&mut output, &layer, left, top);
        drawn += 1;
    }

    if drawn == 0 {
        bail!("scene has no visible image layers to draw");
    }

    Ok(Composite { image: output, omissions })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::model::Orthographic;

    fn canvas(width: u32, height: u32, resolution: Option<Resolution>) -> Canvas {
        canvas_for(Orthographic { width, height }, resolution).unwrap()
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
        let (width, height) = extent(&object, &texture);
        assert!((width - 1184.777).abs() < 0.01, "got {width}");
        assert!((height - 786.261).abs() < 0.01, "got {height}");
    }

    #[test]
    fn a_sizeless_object_takes_its_extent_from_the_texture() {
        let object: Object = serde_json::from_str(r#"{"image":"models/a.json"}"#).unwrap();
        let texture = RgbaImage::new(512, 256);
        assert_eq!(extent(&object, &texture), (512.0, 256.0));
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
        assert_eq!(omissions_for(&effected), vec!["bg: 1 effect(s) not applied"]);
    }

    #[test]
    fn a_plain_layer_has_nothing_to_report() {
        let object: Object = serde_json::from_str(r#"{"name":"bg","image":"models/a.json"}"#).unwrap();
        assert_eq!(omissions_for(&object), Vec::<String>::new());
    }
}

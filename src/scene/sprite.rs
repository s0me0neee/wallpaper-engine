//! Particle sprite textures.
//!
//! A preset's material names a texture like `particle/fog/fog1`. Workshop
//! textures ship inside the package, but the stock ones ship with the Wallpaper
//! Engine *program* and are not ours to redistribute (plan.md §5.2) — so they
//! are read from a real install when one is pointed at, and otherwise stood in
//! for by our own drawings, compiled in from `sprites/`.
//!
//! Those stand-ins are calibrated to *energy* rather than to artwork, because
//! corpus sprites fall into two classes and conflating them is what made every
//! particle a bright disc:
//!
//! | sprite                  | mean alpha | mean RGB |
//! |-------------------------|-----------:|---------:|
//! | `particle/halo`         |      0.152 |    1.000 |
//! | `particle/halo_4`       |      0.040 |    1.000 |
//! | `particle/drop`         |      0.189 |    0.992 |
//! | `particle/fog/fog1`     |      1.000 |    0.052 |
//! | `particle/nature/rain1` |      1.000 |    0.006 |
//! | `particle/misc/wave`    |      1.000 |    0.017 |
//!
//! The first class carries its picture in alpha over white. The second is
//! alpha-opaque and carries it in *RGB*, drawn additively, where black
//! contributes nothing: `nature/rain1` is 99.4% black with a few bright
//! streaks. Standing a tinted radial gradient in for it adds roughly thirty
//! times the light it should, and `scene_example6` draws sixty-four of them at
//! 800–1600 scene units each — which is why the whole canvas hazed over.
//!
//! `tools/make_sprites.py` draws the PNGs and prints what each one measures;
//! the tests below re-measure the checked-in files, so the calibration cannot
//! drift without failing.

use crate::pkg::Archive;
use crate::tex;
use std::path::Path;
use tiny_skia::{Pixmap, PremultipliedColorU8};

/// Our stand-in sprites, by file stem. Small enough (about 48 KB all told) to
/// live in the binary rather than be looked up on disk at run time.
const EMBEDDED: [(&str, &[u8]); 11] = [
    ("halo", include_bytes!("sprites/halo.png")),
    ("halo_2", include_bytes!("sprites/halo_2.png")),
    ("halo_4", include_bytes!("sprites/halo_4.png")),
    ("halo_6", include_bytes!("sprites/halo_6.png")),
    ("drop", include_bytes!("sprites/drop.png")),
    ("smoke", include_bytes!("sprites/smoke.png")),
    ("shafts", include_bytes!("sprites/shafts.png")),
    ("chromaticdot", include_bytes!("sprites/chromaticdot.png")),
    ("fog", include_bytes!("sprites/fog.png")),
    ("wave", include_bytes!("sprites/wave.png")),
    ("rain", include_bytes!("sprites/rain.png")),
];

/// Resolve one texture name to a sprite.
///
/// Package first (workshop sprites live there), then the install if one was
/// given, then our own stand-in. A texture that fails to decode falls through
/// rather than failing the layer.
pub fn resolve(archive: &mut Archive, assets: Option<&Path>, name: &str) -> Pixmap {
    if let Some(pixmap) = from_package(archive, name) {
        return pixmap;
    }
    if let Some(root) = assets
        && let Some(pixmap) = from_assets(root, name)
    {
        return pixmap;
    }
    stand_in(name)
}

fn from_package(archive: &mut Archive, name: &str) -> Option<Pixmap> {
    let bytes = archive.read(&format!("materials/{name}.tex")).ok()?;
    to_pixmap(&decode_tex(&bytes)?)
}

/// Read `<root>/materials/<name>.tex`, or a plain image beside it — so a
/// directory of PNGs stands in for an install just as well.
fn from_assets(root: &Path, name: &str) -> Option<Pixmap> {
    let base = root.join("materials").join(name);
    if let Ok(bytes) = std::fs::read(base.with_extension("tex"))
        && let Some(decoded) = decode_tex(&bytes)
    {
        return to_pixmap(&decoded);
    }
    for extension in ["png", "tga", "jpg"] {
        if let Ok(image) = image::open(base.with_extension(extension)) {
            return to_pixmap(&image.to_rgba8());
        }
    }
    None
}

fn decode_tex(bytes: &[u8]) -> Option<image::RgbaImage> {
    let texture = tex::parse_bytes(bytes).ok()?;
    let mipmap = tex::largest_mipmap(&texture).ok()?;
    tex::decode_rgba(&texture, mipmap).ok()
}

fn to_pixmap(image: &image::RgbaImage) -> Option<Pixmap> {
    let mask = is_coverage_mask(image);
    let mut pixmap = Pixmap::new(image.width(), image.height())?;
    for (destination, source) in pixmap.pixels_mut().iter_mut().zip(image.pixels()) {
        let [r, g, b, a] = source.0;
        let (r, g, b, a) = if mask {
            // The shape is the luminance; the colour comes from the particle.
            let coverage = luminance(r, g, b);
            (255, 255, 255, coverage)
        } else {
            (r, g, b, a)
        };
        // Both the archive and our own PNGs store straight alpha; tiny-skia
        // wants it premultiplied.
        *destination = PremultipliedColorU8::from_rgba(mul(r, a), mul(g, a), mul(b, a), a)?;
    }
    Some(pixmap)
}

/// Whether this sprite carries its shape in luminance rather than in alpha.
///
/// Wallpaper Engine stores several particle sprites single-channel — `R8` for
/// `particle/fog/fog1` and `particle/nature/rain1`, `RG88` with an unused
/// second channel for `particle/misc/wave` — and `tex.rs` expands those to
/// opaque grey, which is right for dumping a viewable PNG and wrong for a
/// sprite. Drawn as-is by a `translucent` system, an opaque near-black fog
/// puff *paints* near-black: `scene_example8`'s twenty fog systems were pulling
/// the whole frame down by a fifth, and the same sprites drawn additively added
/// almost no light because their RGB is nearly zero.
///
/// The test is the alpha channel itself. A sprite that is opaque everywhere
/// cannot be using alpha for shape, so its luminance must be the coverage;
/// `halo`, `drop` and `smoke2` all carry real alpha and are left alone.
fn is_coverage_mask(image: &image::RgbaImage) -> bool {
    image.pixels().all(|pixel| pixel.0[3] >= 250)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a weighted mean of three u8s, clamped before the cast"
)]
fn luminance(r: u8, g: u8, b: u8) -> u8 {
    // The same weights `downsample_quarter_bloom.frag` uses.
    let value = 0.2989 * f32::from(r) + 0.5870 * f32::from(g) + 0.1140 * f32::from(b);
    value.clamp(0.0, 255.0).round() as u8
}

#[expect(clippy::cast_possible_truncation, reason = "product of two u8s over 255 fits a u8")]
fn mul(channel: u8, alpha: u8) -> u8 {
    ((u16::from(channel) * u16::from(alpha) + 127) / 255) as u8
}

/// Pick the stand-in for a texture we do not have, from its name.
///
/// The name is all we get, but it is enough: Wallpaper Engine groups its stock
/// sprites in directories (`particle/fog/`, `particle/nature/`) and the leaf
/// names are descriptive, so the family reads straight off the path.
pub fn stand_in(name: &str) -> Pixmap {
    load(family(name))
}

/// Map a Wallpaper Engine texture name onto one of our sprite stems.
fn family(name: &str) -> &'static str {
    let full = name.to_ascii_lowercase();
    let leaf = full.rsplit('/').next().unwrap_or(&full).to_string();

    if full.contains("/fog/") || leaf.starts_with("fog") {
        return "fog";
    }
    if full.contains("/smoke/") || leaf.starts_with("smoke") {
        return "smoke";
    }
    if leaf.contains("shaft") || leaf.contains("beam") || full.contains("/light/") {
        return "shafts";
    }
    if leaf.contains("rain") || leaf.contains("дожд") {
        return "rain";
    }
    if leaf.contains("wave") || leaf.contains("ripple") {
        return "wave";
    }
    if leaf.contains("dot") {
        return "chromaticdot";
    }
    // `капля` is "drop"; one workshop pack names its sprites in Russian.
    if leaf.starts_with("drop") || leaf.contains("капля") {
        return "drop";
    }
    // Everything else is a halo, which is also what the corpus reaches for
    // most often. The suffix picks how tight it is.
    match leaf.as_str() {
        "halo_2" => "halo_2",
        "halo_4" => "halo_4",
        "halo_6" => "halo_6",
        _ => "halo",
    }
}

fn load(stem: &str) -> Pixmap {
    let bytes = EMBEDDED
        .iter()
        .find(|(name, _)| *name == stem)
        .map_or(EMBEDDED[0].1, |(_, bytes)| *bytes);
    decode_embedded(bytes).unwrap_or_else(|| {
        // Only reachable if a checked-in PNG is corrupt, which the tests catch.
        Pixmap::new(1, 1).unwrap_or_else(|| unreachable!("a 1x1 pixmap always allocates"))
    })
}

fn decode_embedded(bytes: &[u8]) -> Option<Pixmap> {
    let image = image::load_from_memory(bytes).ok()?;
    to_pixmap(&image.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_alpha(pixmap: &Pixmap) -> f32 {
        let total: f64 = pixmap.pixels().iter().map(|p| f64::from(p.alpha())).sum();
        #[expect(clippy::cast_possible_truncation, reason = "a mean of u8 samples")]
        let mean = (total / f64::from(pixmap.width() * pixmap.height()) / 255.0) as f32;
        mean
    }

    #[test]
    fn every_embedded_sprite_decodes() {
        for (name, bytes) in EMBEDDED {
            let pixmap = decode_embedded(bytes)
                .unwrap_or_else(|| panic!("{name}.png failed to decode"));
            assert!(pixmap.width() >= 32, "{name} is suspiciously small");
        }
    }

    #[test]
    fn alpha_shaped_sprites_match_the_measured_energy() {
        // The whole point of the calibration: a halo carries a specific amount
        // of light, and `halo_4` carries a quarter of what `halo` does.
        for (name, want) in [("particle/halo", 0.152), ("particle/halo_4", 0.040)] {
            let got = mean_alpha(&stand_in(name));
            assert!((got - want).abs() < 0.02, "{name}: got {got}, want {want}");
        }
    }

    #[test]
    fn a_luminance_mask_loads_as_coverage_over_white() {
        // `nature/rain1` is stored the way WE stores it: a single-channel mask,
        // 99.4% black, expanded to opaque grey by `tex.rs`. Loaded as a sprite
        // it has to become *coverage* — white where the streak is, transparent
        // elsewhere — or a translucent system paints the sheet's black over
        // the frame instead of drawing a few streaks on it.
        let rain = stand_in("particle/nature/rain1");
        assert!((mean_alpha(&rain) - 0.006).abs() < 0.004, "coverage: {}", mean_alpha(&rain));
        assert!(is_white_under_coverage(&rain), "the colour must come from the particle");
    }

    #[test]
    fn fog_is_a_dim_cloud_rather_than_a_bright_disc() {
        let fog = stand_in("particle/fog/fog1");
        assert!((mean_alpha(&fog) - 0.052).abs() < 0.02, "coverage: {}", mean_alpha(&fog));
        assert!(is_white_under_coverage(&fog));
    }

    /// Pixmaps are premultiplied, so a white sprite's stored RGB *is* its
    /// alpha. Anything else means the colour survived the remap.
    fn is_white_under_coverage(pixmap: &Pixmap) -> bool {
        pixmap.pixels().iter().all(|p| {
            let a = p.alpha();
            p.red().abs_diff(a) <= 1 && p.green().abs_diff(a) <= 1 && p.blue().abs_diff(a) <= 1
        })
    }

    #[test]
    fn an_alpha_shaped_sprite_is_left_alone() {
        // `halo` carries real alpha, so nothing should be remapped: its mean
        // coverage stays at the measured 0.152 rather than becoming its
        // luminance, which for a white sprite would be 1.0.
        let halo = stand_in("particle/halo");
        assert!((mean_alpha(&halo) - 0.152).abs() < 0.02, "got {}", mean_alpha(&halo));
    }

    #[test]
    fn names_route_to_the_family_they_belong_to() {
        assert_eq!(family("particle/fog/fog1"), "fog");
        assert_eq!(family("particle/nature/rain1"), "rain");
        assert_eq!(family("particle/light/light_shafts_0"), "shafts");
        assert_eq!(family("particle/misc/wave"), "wave");
        assert_eq!(family("particle/halo_4"), "halo_4");
        assert_eq!(family("particle/drop_normal"), "drop");
        // `scene_example6` carries a workshop pack that names sprites in
        // Russian: "размытая капля дождя 1" is "blurred raindrop 1".
        assert_eq!(family("workshop/3462439536/particle/размытая капля дождя 1"), "rain");
        // Anything unrecognised is a halo, the corpus's most common sprite.
        assert_eq!(family("workshop/1234/particle/something_odd"), "halo");
    }
}

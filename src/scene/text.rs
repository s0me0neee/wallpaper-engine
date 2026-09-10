//! Text layers.
//!
//! A scene object with a `text` field is a text layer: no model, no particle
//! preset, just a string, a font and a point size. `scene_example8` uses two —
//! a clock and the author's watermark — and `scene_example6` five.
//!
//! Two things about the format matter more than the drawing does.
//!
//! **The text is usually a script.** `text` arrives as
//! `{"script": …, "scriptproperties": …, "value": …}`, where the script is the
//! JavaScript that produces the live string and `value` is the design-time
//! preview the editor last saw. We do not run scripts, so `value` is what there
//! is — and it is sometimes a real-looking string (`"PM O8:24\nApr. 14 2025"`)
//! and sometimes a bare placeholder (`"<Date>"`). A placeholder is not drawn:
//! putting the literal text `<Date>` on the wallpaper is worse than leaving the
//! layer out, and the omission says so.
//!
//! **The font usually is not ours.** Only one font in the whole corpus ships
//! inside a package (`scene_example6`'s `Quicksand-Bold.otf`); the rest name
//! either a Wallpaper Engine asset (`fonts/Alcubierre.otf`) or a system font
//! (`systemfont_consolas`). So the search order is package, then an install if
//! one is pointed at, then the system's own fonts by family — and a text layer
//! whose font resolves nowhere is reported rather than silently substituted.

use super::model::Object;
use crate::pkg::Archive;
use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use image::{Rgba, RgbaImage};
use std::path::Path;
use std::sync::OnceLock;

/// Horizontal placement of each line within the layer's box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

pub fn parse_align(name: &str) -> Align {
    match name {
        "left" => Align::Left,
        "right" => Align::Right,
        _ => Align::Center,
    }
}

/// A rendered text layer: the raster and the note for anything skipped.
pub struct Rendered {
    pub image: RgbaImage,
}

/// Whether `value` is the editor's placeholder rather than real content.
///
/// Wallpaper Engine writes `<Date>`, `<Time and Date>` and the like when a
/// script has never run. Drawing those literally is worse than drawing nothing.
fn is_placeholder(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.is_empty() || (trimmed.starts_with('<') && trimmed.ends_with('>'))
}

/// Glyph size that makes the text fill its box.
///
/// `pointsize` alone does not determine it. Wallpaper Engine fits the box to
/// the text when the layer is made and the author then resizes the box, so the
/// two drift apart: `scene_example8`'s clock box is exactly two 32pt lines
/// tall, while its watermark box is eight times that and WE draws the
/// watermark filling it. Fitting reproduces both without a magic constant —
/// `pointsize` still sets the *relative* size of lines within the block, since
/// every line shares it.
///
/// Height drives the fit; width only ever shrinks it further, so a long line
/// stays inside the box instead of running out of it.
fn fitted_size<F: Font>(font: &F, text: &str, (width, height): (f32, f32)) -> f32 {
    const PROBE: f32 = 100.0;
    let scaled = font.as_scaled(PxScale::from(PROBE));
    let lines = text.split('\n').count().max(1);
    #[expect(clippy::cast_precision_loss, reason = "a handful of lines")]
    let line_count = lines as f32;

    let block = (scaled.height() + scaled.line_gap()) * line_count;
    let by_height = if block > 0.0 { height / block * PROBE } else { PROBE };

    let widest = text
        .split('\n')
        .map(|line| line_advance(&scaled, line))
        .fold(0.0_f32, f32::max);
    let by_width = if widest > 0.0 { width / widest * PROBE } else { by_height };

    by_height.min(by_width).max(1.0)
}

/// Render one text object into an image of `(width, height)` pixels.
///
/// Returns `Err` with a reason the caller can surface as an omission.
pub fn render(
    archive: &mut Archive,
    assets: Option<&Path>,
    object: &Object,
    (width, height): (u32, u32),
) -> Result<Rendered, String> {
    let text = object.text.as_deref().unwrap_or_default();
    if is_placeholder(text) {
        return Err("text is a script we do not run, and its stored value is a placeholder".into());
    }

    let font_name = object.font.as_deref().unwrap_or_default();
    let font = load_font(archive, assets, font_name)
        .ok_or_else(|| format!("font {font_name:?} is not in the package and no system font matched"))?;

    #[expect(clippy::cast_precision_loss, reason = "layer dimensions, nowhere near 2^24")]
    let (box_width, box_height) = (width as f32, height as f32);
    let scale = PxScale::from(fitted_size(&font, text, (box_width, box_height)));
    let scaled = font.as_scaled(scale);
    let line_height = scaled.height() + scaled.line_gap();

    let color = tint(object);
    let mut image = RgbaImage::new(width.max(1), height.max(1));
    let align = parse_align(object.horizontalalign.as_deref().unwrap_or("center"));
    let mut baseline = scaled.ascent();

    for line in text.split('\n') {
        let advance = line_advance(&scaled, line);
        let start = match align {
            Align::Left => 0.0,
            Align::Center => (box_width - advance) / 2.0,
            Align::Right => box_width - advance,
        };
        draw_line(&mut image, &scaled, line, start, baseline, color);
        baseline += line_height;
    }

    Ok(Rendered { image })
}

/// The layer's colour, as `color * brightness` with the object's alpha.
fn tint(object: &Object) -> Rgba<u8> {
    let channel = |value: f32| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 0.0..=1.0 before the cast"
        )]
        let scaled = ((value * object.brightness).clamp(0.0, 1.0) * 255.0).round() as u8;
        scaled
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0.0..=1.0 before the cast"
    )]
    let alpha = (object.alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
    Rgba([channel(object.color.x), channel(object.color.y), channel(object.color.z), alpha])
}

fn line_advance<F: Font>(scaled: &impl ScaleFont<F>, line: &str) -> f32 {
    let mut total = 0.0;
    let mut previous = None;
    for character in line.chars() {
        let id = scaled.glyph_id(character);
        if let Some(last) = previous {
            total += scaled.kern(last, id);
        }
        total += scaled.h_advance(id);
        previous = Some(id);
    }
    total
}

fn draw_line<F: Font>(
    image: &mut RgbaImage,
    scaled: &impl ScaleFont<F>,
    line: &str,
    start: f32,
    baseline: f32,
    color: Rgba<u8>,
) {
    let mut pen = start;
    let mut previous = None;
    for character in line.chars() {
        let id = scaled.glyph_id(character);
        if let Some(last) = previous {
            pen += scaled.kern(last, id);
        }
        let glyph = id.with_scale_and_position(scaled.scale(), ab_glyph::point(pen, baseline));
        if let Some(outline) = scaled.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            outline.draw(|x, y, coverage| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "glyph bounds are small pixel offsets"
                )]
                let (px, py) = (
                    (bounds.min.x as i32).saturating_add_unsigned(x),
                    (bounds.min.y as i32).saturating_add_unsigned(y),
                );
                blend_pixel(image, px, py, color, coverage);
            });
        }
        pen += scaled.h_advance(id);
        previous = Some(id);
    }
}

fn blend_pixel(image: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>, coverage: f32) {
    let (Ok(x), Ok(y)) = (u32::try_from(x), u32::try_from(y)) else {
        return;
    };
    if x >= image.width() || y >= image.height() {
        return;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "coverage is 0.0..=1.0 and alpha is a u8"
    )]
    let alpha = (f32::from(color.0[3]) * coverage.clamp(0.0, 1.0)).round() as u8;
    if alpha == 0 {
        return;
    }
    let existing = image.get_pixel(x, y).0;
    // Straight-alpha "over"; glyphs in one line rarely overlap, but accented
    // forms and tight kerning do.
    let mix = |src: u8, dst: u8| -> u8 {
        let a = f32::from(alpha) / 255.0;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a convex combination of two u8s"
        )]
        let out = (f32::from(src) * a + f32::from(dst) * (1.0 - a)).round() as u8;
        out
    };
    image.put_pixel(
        x,
        y,
        Rgba([
            mix(color.0[0], existing[0]),
            mix(color.0[1], existing[1]),
            mix(color.0[2], existing[2]),
            alpha.max(existing[3]),
        ]),
    );
}

// ---------------------------------------------------------------------------
// Font resolution
// ---------------------------------------------------------------------------

fn load_font(archive: &mut Archive, assets: Option<&Path>, name: &str) -> Option<FontVec> {
    if name.is_empty() {
        return system_font(None);
    }
    // `systemfont_arial` names a family, not a file.
    if let Some(family) = name.strip_prefix("systemfont_") {
        return system_font(Some(family));
    }
    if let Ok(bytes) = archive.read(name)
        && let Ok(font) = FontVec::try_from_vec(bytes)
    {
        return Some(font);
    }
    if let Some(root) = assets
        && let Ok(bytes) = std::fs::read(root.join(name))
        && let Ok(font) = FontVec::try_from_vec(bytes)
    {
        return Some(font);
    }
    // A Wallpaper Engine font we do not have: fall back on a system face of a
    // similar shape rather than dropping the layer entirely.
    system_font(family_hint(name))
}

/// The family a bundled font's filename suggests, for the system fallback.
fn family_hint(path: &str) -> Option<&str> {
    let stem = path.rsplit('/').next()?.split('.').next()?;
    (!stem.is_empty()).then_some(stem)
}

fn system_font(family: Option<&str>) -> Option<FontVec> {
    static DB: OnceLock<fontdb::Database> = OnceLock::new();
    let db = DB.get_or_init(|| {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        db
    });

    let mut families: Vec<fontdb::Family> = Vec::new();
    if let Some(name) = family {
        families.push(fontdb::Family::Name(name));
    }
    // Monospaced clocks are the common case; then anything at all.
    families.extend([fontdb::Family::SansSerif, fontdb::Family::Monospace, fontdb::Family::Serif]);

    let query = fontdb::Query { families: &families, ..fontdb::Query::default() };
    let id = db.query(&query)?;
    db.with_face_data(id, |data, _| FontVec::try_from_vec(data.to_vec()).ok())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_are_not_drawn() {
        // The editor writes these when a script has never run; drawing them
        // puts the literal text "<Date>" on the wallpaper.
        assert!(is_placeholder("<Date>"));
        assert!(is_placeholder("<Time and Date>"));
        assert!(is_placeholder("   "));
        // A real stored value is drawn, angle brackets and all if it has them.
        assert!(!is_placeholder("PM O8:24\nApr. 14 2025"));
        assert!(!is_placeholder("MADE BY HISTA"));
        assert!(!is_placeholder("12:34"));
    }

    #[test]
    fn alignment_parses_with_centre_as_the_default() {
        assert_eq!(parse_align("left"), Align::Left);
        assert_eq!(parse_align("right"), Align::Right);
        assert_eq!(parse_align("center"), Align::Center);
        assert_eq!(parse_align("anything else"), Align::Center);
    }

    #[test]
    fn a_font_filename_suggests_a_family_for_the_fallback() {
        assert_eq!(family_hint("fonts/Alcubierre.otf"), Some("Alcubierre"));
        assert_eq!(family_hint("fonts/workshop/2981960200/Quicksand-Bold.otf"), Some("Quicksand-Bold"));
    }

    #[test]
    fn a_system_face_is_always_available_to_fall_back_on() {
        // Every desktop has *something*; without this the fallback is a cliff.
        assert!(system_font(None).is_some());
    }
}

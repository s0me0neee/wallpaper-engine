//! Decoder for Wallpaper Engine `.tex` textures.
//!
//! A `.tex` is a container of containers, all little-endian:
//!
//! ```text
//! "TEXV0005\0"                    format version
//! "TEXI0001\0"                    image header:
//!     int32 format                TexFormat (see Format below)
//!     int32 flags                 1=NoInterpolation 2=ClampUVs 4=IsGif
//!     int32 texture_width         padded/power-of-two size
//!     int32 texture_height
//!     int32 image_width           actual visible size
//!     int32 image_height
//!     uint32 dominant_color       ARGB
//! "TEXB000n\0"                    image container:
//!     int32 image_count
//!     int32 free_image_format     FreeImage FIF_* enum; -1 = raw pixels
//!                                 |- containers v3+ only
//!     repeat image_count times:
//!         int32 unknown           always 0 in samples seen; v4+ only
//!         int32 mipmap_count
//!         repeat mipmap_count times:
//!             int32 width
//!             int32 height
//!             int32 lz4_compressed     \
//!             int32 decompressed_size  |- containers v2+ only
//!             int32 byte_count
//!             bytes data
//! "TEXS000n\0"                    sprite sheet; present iff flags has IsGif:
//!     uint32 frame_count
//!     uint32 sheet_width          \_ v3 only
//!     uint32 sheet_height         /
//!     repeat frame_count times:
//!         uint32 frame_number
//!         float  seconds
//!         float  x, y             |- int32 in v1, and in a different order
//!         float  width            |
//!         float  unknown, unknown |
//!         float  height           /
//! ```
//!
//! `free_image_format` is the crux: when it is not -1 each mipmap payload is a
//! complete encoded image file (PNG/JPEG/...) that we write straight to disk.
//! When it is -1 the payload is raw pixel data in `format`, optionally wrapped
//! in an LZ4 block, which we decode and re-encode as PNG.

use crate::reader::Reader;
use anyhow::{Context, Result, bail};
use std::{
    borrow::Cow,
    fmt::Write as _,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek},
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Rgba8888,
    Dxt5,
    Dxt3,
    Dxt1,
    Rg88,
    R8,
    Unknown(i32),
}

impl Format {
    fn from_raw(value: i32) -> Self {
        match value {
            0 => Format::Rgba8888,
            4 => Format::Dxt5,
            6 => Format::Dxt3,
            7 => Format::Dxt1,
            8 => Format::Rg88,
            9 => Format::R8,
            other => Format::Unknown(other),
        }
    }

    /// The `texpresso` equivalent, for the block-compressed formats only.
    ///
    /// The names are offset by one: DXT1/3/5 are BC1/2/3, so this mapping is
    /// worth reading twice — swapping BC2 and BC3 would silently decode every
    /// DXT texture with the wrong alpha scheme.
    fn block_format(self) -> Option<texpresso::Format> {
        match self {
            Format::Dxt1 => Some(texpresso::Format::Bc1),
            Format::Dxt3 => Some(texpresso::Format::Bc2),
            Format::Dxt5 => Some(texpresso::Format::Bc3),
            _ => None,
        }
    }

    fn label(self) -> String {
        match self {
            Format::Unknown(value) => format!("unknown({value})"),
            other => format!("{other:?}").to_uppercase(),
        }
    }
}

/// `FreeImage` `FIF_*` values we may meet as embedded files, and their extension.
fn free_image_ext(value: i32) -> Option<&'static str> {
    Some(match value {
        _ if value < 0 => return None,
        0 => "bmp",
        1 => "ico",
        2 => "jpg",
        3 => "jng",
        10 => "pcx",
        13 => "png",
        17 => "tga",
        18 => "tiff",
        19 => "wbmp",
        20 => "psd",
        25 => "gif",
        26 => "hdr",
        32 => "jp2",
        _ => "bin",
    })
}

pub struct Mipmap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
    pub lz4_compressed: bool,
    pub decompressed_size: usize,
}

/// One cell of an animated texture, in pixels of the whole sheet.
pub struct Frame {
    pub seconds: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub struct Tex {
    pub version: String,
    pub container_version: String,
    pub format: Format,
    pub flags: i32,
    pub texture_width: i32,
    pub texture_height: i32,
    pub image_width: i32,
    pub image_height: i32,
    pub dominant_color: u32,
    pub free_image_format: i32,
    pub images: Vec<Vec<Mipmap>>,
    pub frames: Vec<Frame>,
    pub bytes_consumed: u64,
}

/// Columns, rows and cycle length, for a texture that really is a grid.
///
/// A GIF stores one full-size frame per entry, so its cell is the whole
/// texture and the grid comes out 1x1 — which cannot hold its own frames.
/// That mismatch is the test, and it is why this returns `None` rather than
/// treating every animated texture as a sheet.
pub fn sheet(tex: &Tex) -> Option<(u32, u32, f32)> {
    let first = tex.frames.first()?;
    if first.width <= 0.0 || first.height <= 0.0 || tex.image_width <= 0 || tex.image_height <= 0 {
        return None;
    }
    #[expect(clippy::cast_precision_loss, reason = "texture sides, far below 2^24")]
    let (sheet_w, sheet_h) = (tex.image_width as f32, tex.image_height as f32);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a texture side over a cell side, rounded: small and positive"
    )]
    let (cols, rows) = (
        (sheet_w / first.width).round() as u32,
        (sheet_h / first.height).round() as u32,
    );
    let count = u32::try_from(tex.frames.len()).ok()?;
    if cols == 0 || rows == 0 || cols * rows < count {
        return None;
    }
    Some((cols, rows, tex.frames.iter().map(|frame| frame.seconds).sum()))
}

impl Tex {
    /// Extension of the embedded file format, or `None` for raw pixels.
    pub fn embedded_ext(&self) -> Option<&'static str> {
        free_image_ext(self.free_image_format)
    }

    fn flag_names(&self) -> String {
        let mut names = Vec::new();
        if self.flags & 1 != 0 {
            names.push("NoInterpolation");
        }
        if self.flags & 2 != 0 {
            names.push("ClampUVs");
        }
        if self.flags & 4 != 0 {
            names.push("IsGif");
        }
        if names.is_empty() {
            "none".to_string()
        } else {
            names.join(", ")
        }
    }
}

pub fn parse(path: &Path) -> Result<Tex> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    parse_from(BufReader::new(file))
}

/// Parse a texture already in memory, as served straight out of a `.pkg`.
pub fn parse_bytes(bytes: &[u8]) -> Result<Tex> {
    parse_from(std::io::Cursor::new(bytes))
}

fn parse_from<R: Read + Seek>(source: R) -> Result<Tex> {
    let mut reader = Reader::new(source);

    let version = reader.magic()?;
    if !version.starts_with("TEXV") {
        bail!("not a .tex file (magic {version:?})");
    }

    let image_magic = reader.magic()?;
    if !image_magic.starts_with("TEXI") {
        bail!("expected TEXI header, got {image_magic:?}");
    }

    let format = Format::from_raw(reader.i32()?);
    let flags = reader.i32()?;
    let texture_width = reader.i32()?;
    let texture_height = reader.i32()?;
    let image_width = reader.i32()?;
    let image_height = reader.i32()?;
    let dominant_color = reader.u32()?;

    let container_version = reader.magic()?;
    if !container_version.starts_with("TEXB") {
        bail!("expected TEXB container, got {container_version:?}");
    }
    let container_number: i32 = container_version[4..]
        .parse()
        .with_context(|| format!("parsing container version {container_version:?}"))?;

    let image_count = reader.i32()?;
    if !(0..=1024).contains(&image_count) {
        bail!("implausible image count {image_count}");
    }

    // v2 and older have no such field and are always raw pixels. v3 gained it;
    // v4 additionally gained the per-image word below, so the two must not be
    // read together or a v3 container mistakes its format for a mipmap count.
    let free_image_format = if container_number >= 3 {
        reader.i32()?
    } else {
        -1
    };

    #[expect(clippy::cast_sign_loss, reason = "just bounded to 0..=1024 above")]
    let mut images = Vec::with_capacity(image_count as usize);
    for _ in 0..image_count {
        // Purpose unknown; 0 in every v4 sample inspected. Every sample has a
        // single image, so it could equally be a one-off field before the loop.
        if container_number >= 4 {
            let _unknown = reader.i32()?;
        }

        let mipmap_count = reader.i32()?;
        if !(0..=64).contains(&mipmap_count) {
            bail!("implausible mipmap count {mipmap_count}");
        }

        #[expect(clippy::cast_sign_loss, reason = "just bounded to 0..=64 above")]
        let mut mipmaps = Vec::with_capacity(mipmap_count as usize);
        for _ in 0..mipmap_count {
            let width = reader.i32()?;
            let height = reader.i32()?;

            let (lz4_compressed, decompressed_size) = if container_number >= 2 {
                (reader.i32()? != 0, reader.i32()?)
            } else {
                (false, 0)
            };

            let byte_count = reader.i32()?;
            if width < 0 || height < 0 || byte_count < 0 || decompressed_size < 0 {
                bail!("negative mipmap dimension or size");
            }

            #[expect(clippy::cast_sign_loss, reason = "just checked non-negative above")]
            mipmaps.push(Mipmap {
                width: width as usize,
                height: height as usize,
                data: reader.bytes(byte_count as usize)?,
                lz4_compressed,
                decompressed_size: decompressed_size as usize,
            });
        }
        images.push(mipmaps);
    }

    let frames = if flags & 4 == 0 {
        Vec::new()
    } else {
        parse_frames(&mut reader).context("reading the sprite sheet frame table")?
    };

    Ok(Tex {
        version,
        container_version,
        format,
        flags,
        texture_width,
        texture_height,
        image_width,
        image_height,
        dominant_color,
        free_image_format,
        images,
        frames,
        bytes_consumed: reader.pos(),
    })
}

fn parse_frames<R: Read + Seek>(reader: &mut Reader<R>) -> Result<Vec<Frame>> {
    let magic = reader.magic()?;
    let version: i32 = match magic.strip_prefix("TEXS") {
        Some(number) => number.parse().with_context(|| format!("parsing sheet version {magic:?}"))?,
        None => bail!("expected TEXS sheet header, got {magic:?}"),
    };

    let frame_count = reader.u32()?;
    if frame_count > 4096 {
        bail!("implausible frame count {frame_count}");
    }
    // v3 states the cell size up front. It is redundant with the first frame's
    // own width and height, which is what every reader actually uses.
    if version >= 3 {
        let (_cell_width, _cell_height) = (reader.u32()?, reader.u32()?);
    }

    let mut frames = Vec::with_capacity(frame_count as usize);
    for _ in 0..frame_count {
        let _frame_number = reader.u32()?;
        let seconds = reader.f32()?;
        // v1 stores the rect as integers, and puts height last after two words
        // whose purpose is unknown; v2 and v3 keep that order in floats.
        let (x, y, width, height) = if version <= 1 {
            #[expect(clippy::cast_precision_loss, reason = "sheet pixel coordinates, far below 2^24")]
            let value = |value: u32| value as f32;
            let (x, y, width) = (value(reader.u32()?), value(reader.u32()?), value(reader.u32()?));
            let (_unknown_a, _unknown_b) = (reader.u32()?, reader.u32()?);
            (x, y, width, value(reader.u32()?))
        } else {
            let (x, y, width) = (reader.f32()?, reader.f32()?, reader.f32()?);
            let (_unknown_a, _unknown_b) = (reader.f32()?, reader.f32()?);
            (x, y, width, reader.f32()?)
        };
        frames.push(Frame { seconds, x, y, width, height });
    }
    Ok(frames)
}

/// Decompress a mipmap payload, returning it borrowed when already plain.
///
/// These are raw LZ4 blocks with no length prefix, so the size has to come
/// from the header rather than from the stream itself.
fn mipmap_pixels(mipmap: &Mipmap) -> Result<Cow<'_, [u8]>> {
    if !mipmap.lz4_compressed {
        return Ok(Cow::Borrowed(&mipmap.data));
    }

    let mut out = vec![0u8; mipmap.decompressed_size];
    let written = lz4_flex::block::decompress_into(&mipmap.data, &mut out)
        .context("decompressing LZ4 mipmap")?;
    if written != mipmap.decompressed_size {
        bail!(
            "LZ4 output was {written} bytes, expected {}",
            mipmap.decompressed_size
        );
    }
    Ok(Cow::Owned(out))
}

/// Write raw 8-bit samples as a non-interlaced PNG.
fn write_png(
    path: &Path,
    pixels: &[u8],
    width: usize,
    height: usize,
    color: png::ColorType,
) -> Result<()> {
    let needed = width * height * color.samples();
    if pixels.len() < needed {
        bail!(
            "pixel buffer too small: {} < {needed} for {width}x{height}",
            pixels.len()
        );
    }

    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    // PNG dimensions are u32 by spec; anything wider would not be a valid PNG.
    let png_width = u32::try_from(width).context("image too wide for a PNG")?;
    let png_height = u32::try_from(height).context("image too tall for a PNG")?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), png_width, png_height);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels[..needed])?;
    Ok(())
}

/// Decode a mipmap to 8-bit RGBA, whatever it was stored as.
///
/// `save_mipmap` deliberately writes each format in its narrowest PNG colour
/// type and hands embedded files through untouched, which is right for
/// inspection. Compositing needs the opposite: one uniform buffer, so a mask
/// and a photo can be blended by the same code.
pub fn decode_rgba(tex: &Tex, mipmap: &Mipmap) -> Result<image::RgbaImage> {
    let payload = mipmap_pixels(mipmap)?;
    let width = u32::try_from(mipmap.width).context("texture too wide")?;
    let height = u32::try_from(mipmap.height).context("texture too tall")?;

    // An embedded file carries its own dimensions, and they are authoritative:
    // the mipmap header describes the texture slot, not the encoded image.
    if tex.embedded_ext().is_some() {
        let decoded = image::load_from_memory(&payload)
            .context("decoding the image embedded in the texture")?;
        return Ok(decoded.into_rgba8());
    }

    if let Some(block_format) = tex.format.block_format() {
        let needed = block_format.compressed_size(mipmap.width, mipmap.height);
        if payload.len() < needed {
            bail!(
                "{} needs {needed} bytes, got {}",
                tex.format.label(),
                payload.len()
            );
        }
        let mut rgba = vec![0u8; mipmap.width * mipmap.height * 4];
        block_format.decompress(&payload, mipmap.width, mipmap.height, &mut rgba);
        return image::RgbaImage::from_raw(width, height, rgba)
            .context("block-compressed texture did not fill its buffer");
    }

    let pixels = mipmap.width * mipmap.height;
    let rgba = match tex.format {
        Format::Rgba8888 => payload.as_ref().get(..pixels * 4).map(<[u8]>::to_vec),
        // A mask: one channel replicated to grey, fully opaque. The alpha it
        // carries is applied by whatever samples it, not by the mask itself.
        Format::R8 => payload
            .as_ref()
            .get(..pixels)
            .map(|data| data.iter().flat_map(|&v| [v, v, v, 255]).collect()),
        Format::Rg88 => payload.as_ref().get(..pixels * 2).map(|data| {
            data.as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&[grey, alpha]| [grey, grey, grey, alpha])
                .collect()
        }),
        other => bail!("unsupported pixel format {}", other.label()),
    };

    let rgba = rgba.with_context(|| {
        format!(
            "pixel buffer too small for {}x{} {}",
            mipmap.width,
            mipmap.height,
            tex.format.label()
        )
    })?;
    image::RgbaImage::from_raw(width, height, rgba).context("texture did not fill its buffer")
}

/// The largest mipmap of the first image, which is the texture itself.
pub fn largest_mipmap(tex: &Tex) -> Result<&Mipmap> {
    tex.images
        .first()
        .and_then(|mipmaps| mipmaps.first())
        .context("texture contains no images")
}

/// Write one mipmap to disk, returning the path actually written.
pub fn save_mipmap(tex: &Tex, mipmap: &Mipmap, stem: &Path) -> Result<PathBuf> {
    let payload = mipmap_pixels(mipmap)?;

    // Already an encoded image file: pass it straight through untouched.
    if let Some(ext) = tex.embedded_ext() {
        let target = stem.with_extension(ext);
        std::fs::write(&target, payload.as_ref())?;
        return Ok(target);
    }

    let target = stem.with_extension("png");
    let (width, height) = (mipmap.width, mipmap.height);

    let color = match tex.format {
        Format::Rgba8888 | Format::Dxt1 | Format::Dxt3 | Format::Dxt5 => png::ColorType::Rgba,
        // Two channels; greyscale+alpha is a byte-exact PNG match.
        Format::Rg88 => png::ColorType::GrayscaleAlpha,
        Format::R8 => png::ColorType::Grayscale,
        Format::Unknown(_) => bail!("unsupported pixel format {}", tex.format.label()),
    };

    match tex.format.block_format() {
        Some(block_format) => {
            let needed = block_format.compressed_size(width, height);
            if payload.len() < needed {
                bail!(
                    "{} needs {needed} bytes, got {}",
                    tex.format.label(),
                    payload.len()
                );
            }
            let mut rgba = vec![0u8; width * height * 4];
            block_format.decompress(&payload, width, height, &mut rgba);
            write_png(&target, &rgba, width, height, color)?;
        }
        None => write_png(&target, &payload, width, height, color)?,
    }

    Ok(target)
}

pub fn describe(path: &Path, tex: &Tex, total: u64) {
    let embedded = tex.embedded_ext().unwrap_or("raw pixels");

    println!("{}", path.display());
    println!(
        "  {} / {}   format={}   payload={embedded}",
        tex.version,
        tex.container_version,
        tex.format.label()
    );
    println!(
        "  image {}x{}  texture {}x{}",
        tex.image_width, tex.image_height, tex.texture_width, tex.texture_height
    );
    println!(
        "  flags={} [{}]  dominant=#{:08x}",
        tex.flags,
        tex.flag_names(),
        tex.dominant_color
    );

    for (image_index, mipmaps) in tex.images.iter().enumerate() {
        for (level, mipmap) in mipmaps.iter().enumerate() {
            let kind = if mipmap.lz4_compressed { "lz4" } else { "store" };
            let size = if mipmap.lz4_compressed {
                mipmap.decompressed_size
            } else {
                mipmap.data.len()
            };
            println!(
                "    image{image_index} mip{level}  {:>5}x{:<5}  {kind:>5}  {:>12} -> {size:>12} bytes",
                mipmap.width,
                mipmap.height,
                mipmap.data.len()
            );
        }
    }

    if !tex.frames.is_empty() {
        let first = &tex.frames[0];
        let grid = match sheet(tex) {
            Some((cols, rows, seconds)) => format!("{cols}x{rows} grid, {seconds:.3}s cycle"),
            None => "not a grid (one full-size frame each, as a gif stores them)".to_string(),
        };
        println!(
            "  {} frame(s)  cell {}x{}  {grid}",
            tex.frames.len(),
            first.width,
            first.height
        );
    }

    let trailing = total - tex.bytes_consumed;
    let note = if trailing == 0 {
        "exact".to_string()
    } else {
        format!("{trailing} trailing bytes")
    };
    println!(
        "  consumed {} of {total} bytes: {note}",
        tex.bytes_consumed
    );
}

pub fn convert(path: &Path, out_dir: &Path, all_mipmaps: bool, info_only: bool) -> Result<()> {
    let total = std::fs::metadata(path)?.len();
    let tex = parse(path)?;
    describe(path, &tex, total);

    if info_only {
        println!();
        return Ok(());
    }

    std::fs::create_dir_all(out_dir)?;
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("texture");

    for (image_index, mipmaps) in tex.images.iter().enumerate() {
        let selected: Vec<(usize, &Mipmap)> = if all_mipmaps {
            mipmaps.iter().enumerate().collect()
        } else {
            mipmaps.iter().enumerate().take(1).collect()
        };

        for (level, mipmap) in selected {
            let mut name = stem.to_string();
            if tex.images.len() > 1 {
                let _ = write!(name, "_img{image_index}");
            }
            if all_mipmaps && mipmaps.len() > 1 {
                let _ = write!(name, "_mip{level}");
            }
            let written = save_mipmap(&tex, mipmap, &out_dir.join(name))?;
            println!("    -> {}", written.display());
        }
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED565: u16 = 31 << 11;

    /// Build a BC1 colour block from two endpoints and a packed index word.
    fn color_block(c0: u16, c1: u16, bits: u32) -> Vec<u8> {
        let mut block = Vec::new();
        block.extend_from_slice(&c0.to_le_bytes());
        block.extend_from_slice(&c1.to_le_bytes());
        block.extend_from_slice(&bits.to_le_bytes());
        block
    }

    fn decode_one_block(format: Format, block: &[u8]) -> Vec<u8> {
        let block_format = format.block_format().expect("block-compressed format");
        let mut rgba = vec![0u8; 4 * 4 * 4];
        block_format.decompress(block, 4, 4, &mut rgba);
        rgba
    }

    /// DXT1/3/5 map to BC1/2/3 — an off-by-one here would decode every
    /// compressed texture with the wrong alpha scheme.
    #[test]
    fn dxt_maps_onto_the_right_bc_format() {
        assert_eq!(Format::Dxt1.block_format(), Some(texpresso::Format::Bc1));
        assert_eq!(Format::Dxt3.block_format(), Some(texpresso::Format::Bc2));
        assert_eq!(Format::Dxt5.block_format(), Some(texpresso::Format::Bc3));
        assert_eq!(Format::Rgba8888.block_format(), None);
        assert_eq!(Format::R8.block_format(), None);
        assert_eq!(Format::Rg88.block_format(), None);
    }

    #[test]
    fn dxt1_decodes_solid_color_to_full_range() {
        // c0 == c1, every index 0. The 5-bit 0x1F must expand to 0xFF, not 0xF8.
        let out = decode_one_block(Format::Dxt1, &color_block(RED565, RED565, 0));
        assert_eq!(&out[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn dxt3_reads_explicit_alpha_before_the_color_block() {
        // BC2 puts 16 four-bit alphas first; nibble 0xF -> 255, rest -> 0.
        let mut block = 0xFu64.to_le_bytes().to_vec();
        block.extend(color_block(RED565, RED565, 0));
        let out = decode_one_block(Format::Dxt3, &block);
        assert_eq!(out[3], 255);
        assert_eq!(out[7], 0);
    }

    #[test]
    fn dxt5_reads_interpolated_alpha_before_the_color_block() {
        // BC3 puts an 8-byte alpha ramp first: a0=255, index 0 selects it.
        let mut block = vec![255u8, 0];
        block.extend_from_slice(&0u64.to_le_bytes()[0..6]);
        block.extend(color_block(RED565, RED565, 0));
        let out = decode_one_block(Format::Dxt5, &block);
        assert_eq!(&out[0..4], &[255, 0, 0, 255]);
    }

    /// Wallpaper Engine stores bare LZ4 blocks, not the size-prepended variant,
    /// which is why the size has to come from the mipmap header. Long runs
    /// exercise the overlapping-match path that mask textures are full of.
    #[test]
    fn lz4_payloads_are_raw_blocks() {
        let mut original = vec![0u8; 4096];
        original[1024..3072].fill(0xAB);

        let raw = lz4_flex::block::compress(&original);
        let mut out = vec![0u8; original.len()];
        let written = lz4_flex::block::decompress_into(&raw, &mut out).unwrap();
        assert_eq!(written, original.len());
        assert_eq!(out, original);

        // The size-prepended variant is the same block behind a 4-byte length,
        // so decoding one as the other would misread pixels as a length.
        let prepended = lz4_flex::block::compress_prepend_size(&original);
        assert_eq!(prepended.len(), raw.len() + 4);
        assert_eq!(&prepended[4..], &raw[..]);
    }

    /// The decoder rejects a block that ends with a match rather than a literal
    /// run, which the LZ4 spec forbids. Hand-rolled decoders tend to accept it.
    #[test]
    fn lz4_rejects_a_malformed_block() {
        let block = [0x10u8, b'A', 0x01, 0x00];
        let mut out = vec![0u8; 5];
        assert!(lz4_flex::block::decompress_into(&block, &mut out).is_err());
    }

    #[test]
    fn mipmap_pixels_passes_through_uncompressed_payloads() {
        let mipmap = Mipmap {
            width: 2,
            height: 1,
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            lz4_compressed: false,
            decompressed_size: 0,
        };
        assert!(matches!(mipmap_pixels(&mipmap).unwrap(), Cow::Borrowed(_)));
    }

    #[test]
    fn mipmap_pixels_rejects_a_wrong_decompressed_size() {
        let mipmap = Mipmap {
            width: 1,
            height: 1,
            data: vec![0x10, b'A', 0x01, 0x00],
            lz4_compressed: true,
            decompressed_size: 99,
        };
        assert!(mipmap_pixels(&mipmap).is_err());
    }

    /// A whole `.tex` with one 1x1 mipmap and an optional TEXS frame table,
    /// which is the only way to exercise the frame table without shipping a
    /// texture: every animated one in the corpus lives in `papers/`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a hand-built fixture whose every number is a small positive constant"
    )]
    fn animated_tex(width: i32, height: i32, cells: &[(f32, f32, f32, f32)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"TEXV0005\0");
        bytes.extend_from_slice(b"TEXI0001\0");
        for value in [0, 4, width, height, width, height, 0] {
            bytes.extend_from_slice(&i32::to_le_bytes(value));
        }
        bytes.extend_from_slice(b"TEXB0003\0");
        for value in [1, -1, 1, 1, 1, 0, 0, 4] {
            bytes.extend_from_slice(&i32::to_le_bytes(value));
        }
        bytes.extend_from_slice(&[0, 0, 0, 0]);

        bytes.extend_from_slice(b"TEXS0003\0");
        bytes.extend_from_slice(&u32::to_le_bytes(cells.len() as u32));
        bytes.extend_from_slice(&u32::to_le_bytes(cells.first().map_or(0.0, |c| c.2) as u32));
        bytes.extend_from_slice(&u32::to_le_bytes(cells.first().map_or(0.0, |c| c.3) as u32));
        for (index, (x, y, w, h)) in cells.iter().enumerate() {
            bytes.extend_from_slice(&u32::to_le_bytes(index as u32));
            for value in [0.0625, *x, *y, *w, 0.0, 0.0, *h] {
                bytes.extend_from_slice(&f32::to_le_bytes(value));
            }
        }
        bytes
    }

    /// `birds_128x120x16`'s own shape: sixteen 128x120 cells in a 1024x241
    /// texture, where the slack row is padding rather than a third row.
    #[test]
    fn a_frame_table_resolves_to_the_grid_it_describes() {
        let cells: Vec<(f32, f32, f32, f32)> = (0..16u8)
            .map(|i| (f32::from(i % 8) * 128.0, f32::from(i / 8) * 120.0, 128.0, 120.0))
            .collect();
        let tex = parse_bytes(&animated_tex(1024, 241, &cells)).expect("parses");
        assert_eq!(tex.frames.len(), 16);
        assert_eq!(tex.bytes_consumed, animated_tex(1024, 241, &cells).len() as u64);
        let (cols, rows, seconds) = sheet(&tex).expect("a grid");
        assert_eq!((cols, rows), (8, 2));
        assert!((seconds - 1.0).abs() < 1e-5, "cycle {seconds}");
    }

    /// A gif stores one full-size frame per entry, so its "grid" is 1x1 and
    /// cannot hold its own frames. Treating it as a sheet would slice every
    /// frame down to the first one.
    #[test]
    fn a_gif_is_not_mistaken_for_a_grid() {
        let cells = vec![(0.0, 0.0, 256.0, 256.0); 4];
        let tex = parse_bytes(&animated_tex(256, 256, &cells)).expect("parses");
        assert_eq!(tex.frames.len(), 4);
        assert!(sheet(&tex).is_none());
    }
}

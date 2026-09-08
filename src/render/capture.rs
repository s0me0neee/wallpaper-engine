//! Reading a rendered frame back from the GPU.

use anyhow::{Context, Result};
use glow::HasContext;
use image::RgbaImage;

/// Read the currently-bound framebuffer's colour attachment into an image.
///
/// `glReadPixels` returns rows bottom-to-top; `RgbaImage` is top-to-bottom
/// like every other image in this codebase, so the rows are reversed here
/// rather than pushing that convention onto every caller.
pub fn read_rgba(gl: &glow::Context, framebuffer: glow::Framebuffer, width: u32, height: u32) -> Result<RgbaImage> {
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (signed_width, signed_height) = (width as i32, height as i32);

    let row_bytes = width as usize * 4;
    let mut flipped = vec![0u8; row_bytes * height as usize];
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
        gl.read_pixels(
            0,
            0,
            signed_width,
            signed_height,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(Some(&mut flipped)),
        );
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }

    let mut pixels = vec![0u8; flipped.len()];
    for row in 0..height as usize {
        let source = row * row_bytes;
        let dest = (height as usize - 1 - row) * row_bytes;
        pixels[dest..dest + row_bytes].copy_from_slice(&flipped[source..source + row_bytes]);
    }

    RgbaImage::from_raw(width, height, pixels).context("the captured frame did not fill its buffer")
}

//! Reading a rendered frame back from the GPU.

use anyhow::{Context, Result};
use glow::HasContext;
use image::RgbaImage;

/// Read the currently-bound framebuffer's colour attachment into an image.
///
/// `glReadPixels` returns row 0 at the viewport's bottom, which by
/// `pass::build_quad`'s row-preserving V mapping is also texel row 0 of an
/// effect-chain target — the row the chain keeps aligned with the base
/// image's own row 0 (its visual top) through any number of passes. So the
/// buffer already comes back in the same top-to-bottom order `RgbaImage`
/// wants; no flip is needed here.
pub fn read_rgba(gl: &glow::Context, framebuffer: glow::Framebuffer, width: u32, height: u32) -> Result<RgbaImage> {
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (signed_width, signed_height) = (width as i32, height as i32);

    let mut pixels = vec![0u8; width as usize * 4 * height as usize];
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
        gl.read_pixels(
            0,
            0,
            signed_width,
            signed_height,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(Some(&mut pixels)),
        );
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }

    RgbaImage::from_raw(width, height, pixels).context("the captured frame did not fill its buffer")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{gpu::Gpu, pass};

    const IDENTITY: [f32; 16] =
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    const TOP: [u8; 4] = [255, 0, 0, 255];
    const BOTTOM: [u8; 4] = [0, 0, 255, 255];

    fn two_row_image() -> RgbaImage {
        let mut raw = TOP.to_vec();
        raw.extend_from_slice(&BOTTOM);
        RgbaImage::from_raw(1, 2, raw).expect("building the 1x2 test image")
    }

    fn passthrough_program(gl: &glow::Context) -> pass::Program {
        let vertex = "#version 330 core\n\
            layout(location = 0) in vec3 a_Position;\n\
            layout(location = 1) in vec2 a_TexCoord;\n\
            out vec2 v_TexCoord;\n\
            void main() { v_TexCoord = a_TexCoord; gl_Position = vec4(a_Position, 1.0); }\n";
        let fragment = "#version 330 core\n\
            in vec2 v_TexCoord;\n\
            out vec4 o_Color;\n\
            uniform sampler2D g_Texture0;\n\
            void main() { o_Color = texture(g_Texture0, v_TexCoord); }\n";
        pass::compile_program(gl, vertex, fragment).expect("compiling the passthrough test shader")
    }

    /// Regression test for the effect chain rendering upside down: a pass
    /// that displaces nothing must leave row order alone, so `read_rgba`
    /// hands back exactly what went in.
    ///
    /// `#[ignore]`d because `Gpu::new` requires `AppKit`'s actual process main
    /// thread, which `cargo test`'s harness never provides (it runs every
    /// test, including single-threaded ones, on a spawned worker thread) —
    /// this crate has no lib target to host a `harness = false` integration
    /// test that could. Verified manually instead: temporarily inlined into
    /// `main()` behind an env var and run via `cargo run --release`.
    #[test]
    #[ignore = "needs the AppKit main thread; the test harness can't provide one, see doc comment"]
    fn identity_pass_round_trip_preserves_row_order() {
        let gpu = Gpu::new().expect("opening a headless GL context");
        let base = two_row_image();

        let quad = pass::build_quad(&gpu.gl).expect("building the quad");
        let program = passthrough_program(&gpu.gl);
        let texture = pass::upload_texture(&gpu.gl, &base).expect("uploading the base texture");
        let target = pass::Target::new(&gpu.gl, 1, 2).expect("allocating the target");

        pass::draw(
            &gpu.gl,
            &pass::DrawCall {
                program: &program,
                quad: &quad,
                target: &target,
                textures: &[("g_Texture0", texture)],
                floats: &[],
                ints: &[],
                mvp: &IDENTITY,
            },
        )
        .expect("drawing the identity pass");

        let result = read_rgba(&gpu.gl, target.framebuffer, 1, 2).expect("reading the target back");
        assert_eq!(result.get_pixel(0, 0).0, TOP, "row 0 should stay the image's top row");
        assert_eq!(result.get_pixel(0, 1).0, BOTTOM, "row 1 should stay the image's bottom row");
    }

    /// Regression test for the simulator window rendering upside down: the
    /// blit to the screen needs its own flip, since it's the one place a
    /// row-preserving effect-chain texture meets GL's native bottom-up
    /// window space without a `read_rgba` in between to correct for it.
    #[test]
    #[ignore = "needs the AppKit main thread; see identity_pass_round_trip_preserves_row_order's doc comment"]
    fn blit_to_screen_puts_the_top_row_at_the_top() {
        let gpu = Gpu::new().expect("opening a headless GL context");
        let base = two_row_image();

        let quad = pass::build_display_quad(&gpu.gl).expect("building the display quad");
        let blit = pass::compile_blit_program(&gpu.gl).expect("compiling the blit program");
        let texture = pass::upload_texture(&gpu.gl, &base).expect("uploading the base texture");

        pass::blit_to_screen(&gpu.gl, &blit, &quad, texture, (1, 2), (1, 2));

        let mut buffer = [0u8; 8];
        unsafe {
            gpu.gl.read_pixels(0, 0, 1, 2, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut buffer)));
        }
        // glReadPixels' row 0 is the screen's bottom row, so the screen's
        // top row is the buffer's last one.
        assert_eq!(&buffer[4..8], &TOP, "the image's top row should land at the top of the screen");
        assert_eq!(&buffer[0..4], &BOTTOM, "the image's bottom row should land at the bottom of the screen");
    }
}

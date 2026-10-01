//! A video frame's YUV planes converted and scaled on the GPU, in place of swscale on the CPU.
//!
//! The output is what `video::frame_at` produces — the frame at the layer's size, BT.601 RGB, tinted as
//! `compose::apply_tint` tints — drawn into the layer's texture. 3689822347's 4K video sits on a
//! 4920x2767 layer at 1080p: swscale upscaled it to 54 MB of RGBA every frame (14 ms on six threads,
//! plus the upload), where the planes are 12 MB as decoded and the conversion is one quad.

use crate::render::pass::{self, Target};
use crate::scene::video::Planar;
use anyhow::{Result, anyhow};
use glow::HasContext;

const VERTEX: &str = "#version 330 core\n\
    layout(location = 0) in vec3 a_Position;\n\
    layout(location = 1) in vec2 a_TexCoord;\n\
    out vec2 v_TexCoord;\n\
    void main() {\n\
        v_TexCoord = a_TexCoord;\n\
        gl_Position = vec4(a_Position, 1.0);\n\
    }\n";

// swscale's conversion: BT.709 for a stream tagged so, BT.601 otherwise, chroma centred on 128.
const FRAGMENT: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Y;\n\
    uniform sampler2D u_U;\n\
    uniform sampler2D u_V;\n\
    uniform vec4 u_Tint;\n\
    uniform int u_Full;\n\
    uniform int u_Bt709;\n\
    void main() {\n\
        float y = texture(u_Y, v_TexCoord).r;\n\
        float u = texture(u_U, v_TexCoord).r - 128.0 / 255.0;\n\
        float v = texture(u_V, v_TexCoord).r - 128.0 / 255.0;\n\
        if (u_Full == 0) {\n\
            y = (y - 16.0 / 255.0) * (255.0 / 219.0);\n\
            u *= 255.0 / 224.0;\n\
            v *= 255.0 / 224.0;\n\
        }\n\
        vec3 rgb = u_Bt709 == 1\n\
            ? vec3(y + 1.5748 * v, y - 0.187324 * u - 0.468124 * v, y + 1.8556 * u)\n\
            : vec3(y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u);\n\
        o_Color = vec4(clamp(rgb, 0.0, 1.0) * u_Tint.rgb, u_Tint.a);\n\
    }\n";

pub struct Yuv {
    program: pass::Program,
    quad: pass::Quad,
    /// Y, U, V, each one channel at its own size.
    planes: [glow::Texture; 3],
    sizes: [(u32, u32); 3],
    /// The layer is under half the source on some axis: bilinear alone would skip texels, so the planes
    /// are mipmapped the way swscale's filter widens with the ratio.
    mipmapped: bool,
    planar: Planar,
    pub target: Target,
}

/// Allocate planes for 4:2:0 frames of `source` size and the `size` target they are drawn into.
pub fn build(gl: &glow::Context, source: (u32, u32), size: (u32, u32), planar: Planar) -> Result<Yuv> {
    let program = pass::compile_program(gl, VERTEX, FRAGMENT)?;
    let quad = pass::build_quad(gl)?;
    let target = Target::new(gl, size.0, size.1)?;
    let chroma = (source.0.div_ceil(2), source.1.div_ceil(2));
    let sizes = [source, chroma, chroma];
    let mipmapped = size.0 * 2 < source.0 || size.1 * 2 < source.1;
    let mut planes = Vec::with_capacity(3);
    for &(width, height) in &sizes {
        #[expect(clippy::cast_possible_wrap, reason = "a video frame is nowhere near i32::MAX a side")]
        let (width, height) = (width as i32, height as i32);
        // Safety: a plain texture allocation on the current context.
        unsafe {
            let texture = gl.create_texture().map_err(|error| anyhow!(error))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R8.cast_signed(),
                width,
                height,
                0,
                glow::RED,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            let min_filter = if mipmapped { glow::LINEAR_MIPMAP_LINEAR } else { glow::LINEAR };
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, min_filter.cast_signed());
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR.cast_signed());
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE.cast_signed());
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE.cast_signed());
            planes.push(texture);
        }
    }
    let planes = [planes[0], planes[1], planes[2]];
    Ok(Yuv { program, quad, planes, sizes, mipmapped, planar, target })
}

/// Upload one frame's planes (`(bytes, stride)` each, rows top first) and redraw the target from them.
pub fn draw(gl: &glow::Context, yuv: &Yuv, planes: [(&[u8], usize); 3], tint: [f32; 3], alpha: f32) {
    let handle = yuv.program.handle;
    #[expect(clippy::cast_possible_wrap, reason = "a layer is nowhere near i32::MAX a side")]
    let (width, height) = (yuv.target.width as i32, yuv.target.height as i32);
    // Safety: every handle was created on this context by `build`; each slice holds `stride * height`
    // bytes as the decoder laid them out, and the unpack state changed here is reset below.
    unsafe {
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        for ((&texture, &(plane_width, plane_height)), (bytes, stride)) in yuv.planes.iter().zip(&yuv.sizes).zip(planes) {
            let rows = plane_height as usize;
            let Ok(row_length) = i32::try_from(stride) else { continue };
            if stride < plane_width as usize || bytes.len() < stride * (rows - 1) + plane_width as usize {
                continue;
            }
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, row_length);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            #[expect(clippy::cast_possible_wrap, reason = "a video frame is nowhere near i32::MAX a side")]
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                plane_width as i32,
                plane_height as i32,
                glow::RED,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(bytes)),
            );
            if yuv.mipmapped {
                gl.generate_mipmap(glow::TEXTURE_2D);
            }
        }
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);

        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(yuv.target.framebuffer));
        gl.viewport(0, 0, width, height);
        gl.disable(glow::BLEND);
        gl.use_program(Some(handle));
        for ((unit, name), &texture) in [(0, "u_Y"), (1, "u_U"), (2, "u_V")].into_iter().zip(&yuv.planes) {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.uniform_1_i32(gl.get_uniform_location(handle, name).as_ref(), unit.cast_signed());
        }
        gl.uniform_4_f32(gl.get_uniform_location(handle, "u_Tint").as_ref(), tint[0], tint[1], tint[2], alpha);
        gl.uniform_1_i32(gl.get_uniform_location(handle, "u_Full").as_ref(), i32::from(yuv.planar.full_range));
        gl.uniform_1_i32(gl.get_uniform_location(handle, "u_Bt709").as_ref(), i32::from(yuv.planar.bt709));
        gl.bind_vertex_array(Some(yuv.quad.vertex_array));
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.bind_vertex_array(None);
        gl.use_program(None);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }
}

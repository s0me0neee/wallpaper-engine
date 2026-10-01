//! A puppet's deformed mesh drawn on the GPU, in place of `puppet::rasterize` on the CPU.
//!
//! The output is the same texture `compose::warp_frame` produces — same padded size, same placement, the
//! same straight "over" of each triangle onto transparent black, then the layer's tint — so the layer's
//! chain and the compositor cannot tell the two apart. Rainy Day's three puppets cost 27 ms a frame on
//! the CPU plus a 24 MB upload (§14.7); here they are a few thousand triangles.

use crate::render::pass::{self, Target};
use crate::scene::puppet::Puppet;
use anyhow::{Result, anyhow};
use glow::HasContext;
use image::RgbaImage;

const VERTEX: &str = "#version 330 core\n\
    layout(location = 0) in vec2 a_Position;\n\
    layout(location = 1) in vec2 a_TexCoord;\n\
    uniform vec2 u_Size;\n\
    out vec2 v_TexCoord;\n\
    void main() {\n\
        v_TexCoord = a_TexCoord;\n\
        gl_Position = vec4(a_Position / u_Size * 2.0 - 1.0, 0.0, 1.0);\n\
    }\n";

// Dual-source: the blend weighs by the texel's own alpha (`o_Weight`) while the colour written is already
// tinted, which is `apply_tint` after the blend, since "over" is linear in both colour and alpha.
const FRAGMENT: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    layout(location = 0, index = 0) out vec4 o_Color;\n\
    layout(location = 0, index = 1) out vec4 o_Weight;\n\
    uniform sampler2D u_Texture;\n\
    uniform vec4 u_Tint;\n\
    void main() {\n\
        vec4 texel = texture(u_Texture, v_TexCoord);\n\
        o_Color = vec4(texel.rgb * u_Tint.rgb, texel.a * u_Tint.a);\n\
        o_Weight = vec4(texel.a);\n\
    }\n";

pub struct Skin {
    program: pass::Program,
    vertex_array: glow::VertexArray,
    positions: glow::Buffer,
    vertex_count: usize,
    index_count: i32,
    texture: glow::Texture,
    pub target: Target,
}

/// Upload `puppet`'s texture, UVs and triangles once, and allocate the `size` target it is drawn into.
pub fn build(gl: &glow::Context, puppet: &Puppet, texture: &RgbaImage, size: (u32, u32)) -> Result<Skin> {
    let program = pass::compile_program(gl, VERTEX, FRAGMENT)?;
    let target = Target::new(gl, size.0, size.1)?;
    let texture = pass::upload_texture(gl, texture)?;
    let uvs: Vec<f32> = puppet.vertices.iter().flat_map(|vertex| [vertex.u, vertex.v]).collect();
    let index_count = i32::try_from(puppet.triangles.len())?;
    // Safety: plain buffer and vertex-array setup on the current context; every slice outlives its upload.
    unsafe {
        let vertex_array = gl.create_vertex_array().map_err(|error| anyhow!(error))?;
        gl.bind_vertex_array(Some(vertex_array));
        let positions = gl.create_buffer().map_err(|error| anyhow!(error))?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(positions));
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
        let uv_buffer = gl.create_buffer().map_err(|error| anyhow!(error))?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(uv_buffer));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, pass::bytes_of(&uvs), glow::STATIC_DRAW);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 8, 0);
        let indices = gl.create_buffer().map_err(|error| anyhow!(error))?;
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(indices));
        let index_bytes =
            std::slice::from_raw_parts(puppet.triangles.as_ptr().cast::<u8>(), std::mem::size_of_val(puppet.triangles.as_slice()));
        gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, index_bytes, glow::STATIC_DRAW);
        gl.bind_vertex_array(None);
        Ok(Skin { program, vertex_array, positions, vertex_count: uvs.len() / 2, index_count, texture, target })
    }
}

/// Redraw the target from `positions` (`(x, y)` per vertex, output pixels from the top-left).
pub fn draw(gl: &glow::Context, skin: &Skin, positions: &[f32], tint: [f32; 3], alpha: f32) {
    if positions.len() != skin.vertex_count * 2 {
        return;
    }
    #[expect(clippy::cast_possible_wrap, reason = "a puppet layer is at most 16384 px a side")]
    let (width, height) = (skin.target.width as i32, skin.target.height as i32);
    #[expect(clippy::cast_precision_loss, reason = "a puppet layer is at most 16384 px a side")]
    let size = [skin.target.width as f32, skin.target.height as f32];
    let handle = skin.program.handle;
    // Safety: every handle was created on this context by `build`; state touched here is reset below.
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(skin.target.framebuffer));
        gl.viewport(0, 0, width, height);
        gl.clear_color(0.0, 0.0, 0.0, 0.0);
        gl.clear(glow::COLOR_BUFFER_BIT);
        gl.use_program(Some(handle));
        gl.uniform_2_f32_slice(gl.get_uniform_location(handle, "u_Size").as_ref(), &size);
        gl.uniform_4_f32(gl.get_uniform_location(handle, "u_Tint").as_ref(), tint[0], tint[1], tint[2], alpha);
        gl.uniform_1_i32(gl.get_uniform_location(handle, "u_Texture").as_ref(), 0);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(skin.texture));
        gl.bind_vertex_array(Some(skin.vertex_array));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(skin.positions));
        // Orphaned each frame so the upload never waits on last frame's draw.
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, pass::bytes_of(positions), glow::STREAM_DRAW);
        gl.enable(glow::BLEND);
        gl.blend_equation(glow::FUNC_ADD);
        gl.blend_func_separate(glow::SRC1_ALPHA, glow::ONE_MINUS_SRC1_ALPHA, glow::ONE, glow::ONE_MINUS_SRC1_ALPHA);
        gl.draw_elements(glow::TRIANGLES, skin.index_count, glow::UNSIGNED_INT, 0);
        gl.disable(glow::BLEND);
        gl.bind_vertex_array(None);
        gl.use_program(None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }
}

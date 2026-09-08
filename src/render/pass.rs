//! Compiling one preprocessed shader and running it as a fullscreen pass.
//!
//! Post-process effects are, geometrically, always the same quad — so rather
//! than run Wallpaper Engine's own vertex shader (option (a) in plan.md §5.4),
//! every pass here uses one fixed quad and feeds it the `a_Position`/
//! `a_TexCoord` attributes and `g_ModelViewProjectionMatrix`/
//! `g_TextureNResolution` uniforms its vertex shader expects, computed for an
//! identity transform. Effects whose vertex stage displaces geometry (some
//! sway effects) are out of scope until one is found in the corpus.

use anyhow::{Result, bail};
use glow::HasContext;

/// GL enum constants are `u32`, but the API's own setter parameters are `i32`
/// — a mismatch from the spec's `GLenum`/`GLint` split, not a real risk: every
/// constant used here is a small fixed value, nowhere near `i32::MAX`.
fn gl_enum(value: u32) -> i32 {
    value.cast_signed()
}

/// A compiled vertex+fragment program.
pub struct Program {
    pub handle: glow::Program,
}

pub fn compile_program(gl: &glow::Context, vertex_src: &str, fragment_src: &str) -> Result<Program> {
    let program = unsafe { gl.create_program() }.map_err(|error| anyhow::anyhow!(error))?;

    let vertex = compile_stage(gl, glow::VERTEX_SHADER, vertex_src, "vertex")?;
    let fragment = compile_stage(gl, glow::FRAGMENT_SHADER, fragment_src, "fragment")?;

    unsafe {
        gl.attach_shader(program, vertex);
        gl.attach_shader(program, fragment);
        gl.link_program(program);
    }

    let ok = unsafe { gl.get_program_link_status(program) };
    let log = unsafe { gl.get_program_info_log(program) };
    unsafe {
        gl.detach_shader(program, vertex);
        gl.detach_shader(program, fragment);
        gl.delete_shader(vertex);
        gl.delete_shader(fragment);
    }
    if !ok {
        bail!("linking the shader program: {log}");
    }

    Ok(Program { handle: program })
}

fn compile_stage(gl: &glow::Context, kind: u32, source: &str, label: &str) -> Result<glow::Shader> {
    let shader = unsafe { gl.create_shader(kind) }.map_err(|error| anyhow::anyhow!(error))?;
    unsafe {
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
    }
    if unsafe { gl.get_shader_compile_status(shader) } {
        return Ok(shader);
    }
    let log = unsafe { gl.get_shader_info_log(shader) };
    unsafe { gl.delete_shader(shader) };
    bail!("compiling the {label} shader: {log}");
}

/// An off-screen render target: one RGBA8 texture behind one framebuffer.
pub struct Target {
    pub texture: glow::Texture,
    pub framebuffer: glow::Framebuffer,
    pub width: u32,
    pub height: u32,
}

impl Target {
    pub fn new(gl: &glow::Context, width: u32, height: u32) -> Result<Target> {
        // Wallpaper-sized canvases stay far under i32::MAX; the driver takes
        // signed dimensions because it also accepts negative border sizes we
        // never use.
        #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
        let (signed_width, signed_height) = (width as i32, height as i32);

        let texture = unsafe { gl.create_texture() }.map_err(|error| anyhow::anyhow!(error))?;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                gl_enum(glow::RGBA8),
                signed_width,
                signed_height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, gl_enum(glow::LINEAR));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, gl_enum(glow::LINEAR));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, gl_enum(glow::CLAMP_TO_EDGE));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, gl_enum(glow::CLAMP_TO_EDGE));
        }

        let framebuffer = unsafe { gl.create_framebuffer() }.map_err(|error| anyhow::anyhow!(error))?;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                bail!("framebuffer incomplete: {status:#x}");
            }
        }

        Ok(Target { texture, framebuffer, width, height })
    }
}

/// Upload an RGBA image as a `CLAMP_TO_EDGE`, linearly filtered texture.
pub fn upload_texture(gl: &glow::Context, image: &image::RgbaImage) -> Result<glow::Texture> {
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper textures are nowhere near i32::MAX")]
    let (width, height) = (image.width() as i32, image.height() as i32);

    let texture = unsafe { gl.create_texture() }.map_err(|error| anyhow::anyhow!(error))?;
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            gl_enum(glow::RGBA8),
            width,
            height,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(image.as_raw())),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, gl_enum(glow::LINEAR));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, gl_enum(glow::LINEAR));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, gl_enum(glow::CLAMP_TO_EDGE));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, gl_enum(glow::CLAMP_TO_EDGE));
    }
    Ok(texture)
}

/// A single-pixel texture, for the shim's `util/noflow` and `util/black`
/// stand-ins: a slot whose annotation says "use this when nothing is bound".
pub fn solid_texture(gl: &glow::Context, rgba: [u8; 4]) -> Result<glow::Texture> {
    let texture = unsafe { gl.create_texture() }.map_err(|error| anyhow::anyhow!(error))?;
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            gl_enum(glow::RGBA8),
            1,
            1,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(&rgba)),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, gl_enum(glow::NEAREST));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, gl_enum(glow::NEAREST));
    }
    Ok(texture)
}

/// The fixed fullscreen quad every pass draws: two triangles, `a_Position`
/// (clip-space xyz) interleaved with `a_TexCoord` (0..1, V=0 at the top so it
/// lines up with how images are decoded row-major top-down).
pub struct Quad {
    pub vertex_array: glow::VertexArray,
    _buffer: glow::Buffer,
}

pub fn build_quad(gl: &glow::Context) -> Result<Quad> {
    // The vertex layout below is fixed (xyz, then uv), so its byte offsets are
    // compile-time constants rather than a `size_of::<f32>()` cast.
    const STRIDE: i32 = 5 * 4;
    const UV_OFFSET: i32 = 3 * 4;

    #[rustfmt::skip]
    let vertices: [f32; 20] = [
        // x,    y,    z,   u,   v
        -1.0, -1.0, 0.0,  0.0, 1.0,
         1.0, -1.0, 0.0,  1.0, 1.0,
        -1.0,  1.0, 0.0,  0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 0.0,
    ];
    let bytes = bytes_of(&vertices);

    let vertex_array = unsafe { gl.create_vertex_array() }.map_err(|error| anyhow::anyhow!(error))?;
    let buffer = unsafe { gl.create_buffer() }.map_err(|error| anyhow::anyhow!(error))?;
    unsafe {
        gl.bind_vertex_array(Some(vertex_array));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);

        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, STRIDE, 0);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, STRIDE, UV_OFFSET);

        gl.bind_vertex_array(None);
    }

    Ok(Quad { vertex_array, _buffer: buffer })
}

fn bytes_of(values: &[f32]) -> &[u8] {
    // Safety: any bit pattern is a valid `f32`, and the slice's lifetime and
    // length are carried through unchanged — this is a reinterpretation, not
    // a resize.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// Everything one draw call binds beyond the program itself.
pub struct DrawCall<'a> {
    pub program: &'a Program,
    pub quad: &'a Quad,
    pub target: &'a Target,
    /// `(uniform name, texture, texture unit)`.
    pub textures: &'a [(&'a str, glow::Texture)],
    pub floats: &'a [(&'a str, &'a [f32])],
    pub ints: &'a [(&'a str, i32)],
    /// `a_Position`/`a_TexCoord` attribute locations are fixed by `build_quad`;
    /// `mvp` is the 4x4 matrix bound to `g_ModelViewProjectionMatrix`.
    pub mvp: &'a [f32; 16],
}

/// Bind everything a `DrawCall` names, draw the quad, and leave the GL state
/// as this function found it (program 0, framebuffer 0).
pub fn draw(gl: &glow::Context, call: &DrawCall) -> Result<()> {
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (width, height) = (call.target.width as i32, call.target.height as i32);

    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(call.target.framebuffer));
        gl.viewport(0, 0, width, height);
        gl.use_program(Some(call.program.handle));
        gl.bind_vertex_array(Some(call.quad.vertex_array));
    }

    set_uniform_matrix4(gl, call.program.handle, "g_ModelViewProjectionMatrix", call.mvp);
    for (name, values) in call.floats {
        set_uniform_floats(gl, call.program.handle, name, values);
    }
    for (name, value) in call.ints {
        set_uniform_int(gl, call.program.handle, name, *value);
    }
    for (unit, (name, texture)) in call.textures.iter().enumerate() {
        bind_sampler(gl, call.program.handle, name, *texture, unit);
    }

    let error = unsafe {
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.bind_vertex_array(None);
        gl.use_program(None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.get_error()
    };
    if error != glow::NO_ERROR {
        bail!("GL error {error:#x} drawing the pass");
    }
    Ok(())
}

fn bind_sampler(gl: &glow::Context, program: glow::Program, name: &str, texture: glow::Texture, unit: usize) {
    let Some(location) = (unsafe { gl.get_uniform_location(program, name) }) else {
        return;
    };
    // Wallpaper Engine shaders bind at most a handful of texture slots per
    // pass, nowhere near GL_MAX_TEXTURE_IMAGE_UNITS.
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        reason = "at most a handful of slots per pass"
    )]
    let unit = unit as i32;
    unsafe {
        #[expect(clippy::cast_sign_loss, reason = "unit is a small non-negative index")]
        gl.active_texture(glow::TEXTURE0 + unit as u32);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.uniform_1_i32(Some(&location), unit);
    }
}

fn set_uniform_matrix4(gl: &glow::Context, program: glow::Program, name: &str, values: &[f32; 16]) {
    let Some(location) = (unsafe { gl.get_uniform_location(program, name) }) else {
        return;
    };
    unsafe { gl.uniform_matrix_4_f32_slice(Some(&location), false, values) };
}

/// Set a uniform whose GLSL type is inferred from how many floats it got:
/// 1 -> float, 2 -> vec2, 3 -> vec3, 4 -> vec4.
fn set_uniform_floats(gl: &glow::Context, program: glow::Program, name: &str, values: &[f32]) {
    let Some(location) = (unsafe { gl.get_uniform_location(program, name) }) else {
        return;
    };
    let location = Some(&location);
    unsafe {
        match values {
            [x] => gl.uniform_1_f32(location, *x),
            [x, y] => gl.uniform_2_f32(location, *x, *y),
            [x, y, z] => gl.uniform_3_f32(location, *x, *y, *z),
            [x, y, z, w] => gl.uniform_4_f32(location, *x, *y, *z, *w),
            _ => {}
        }
    }
}

fn set_uniform_int(gl: &glow::Context, program: glow::Program, name: &str, value: i32) {
    let Some(location) = (unsafe { gl.get_uniform_location(program, name) }) else {
        return;
    };
    unsafe { gl.uniform_1_i32(Some(&location), value) };
}

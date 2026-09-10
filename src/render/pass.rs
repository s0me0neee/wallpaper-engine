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
use std::collections::HashMap;

/// GL enum constants are `u32`, but the API's own setter parameters are `i32`
/// — a mismatch from the spec's `GLenum`/`GLint` split, not a real risk: every
/// constant used here is a small fixed value, nowhere near `i32::MAX`.
fn gl_enum(value: u32) -> i32 {
    value.cast_signed()
}

/// A compiled vertex+fragment program.
pub struct Program {
    pub handle: glow::Program,
    /// How many floats each active float/vector uniform actually takes, read
    /// back from the linked program.
    ///
    /// The two stages do not always agree on a uniform's type — WE's own
    /// `clouds` declares `g_CloudSpeeds` as `vec2` in the vertex shader and
    /// `vec4` in the fragment — and only one of the two survives linking.
    /// Uploading the other one's width is an `INVALID_OPERATION` that kills the
    /// draw, so the link is the authority on what to send.
    float_widths: HashMap<String, usize>,
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

    Ok(Program { handle: program, float_widths: float_widths(gl, program) })
}

/// Component count per active float-typed uniform, by name.
fn float_widths(gl: &glow::Context, program: glow::Program) -> HashMap<String, usize> {
    let count = unsafe { gl.get_active_uniforms(program) };
    (0..count)
        .filter_map(|index| {
            let uniform = unsafe { gl.get_active_uniform(program, index) }?;
            let width = match uniform.utype {
                glow::FLOAT => 1,
                glow::FLOAT_VEC2 => 2,
                glow::FLOAT_VEC3 => 3,
                glow::FLOAT_VEC4 => 4,
                _ => return None,
            };
            Some((uniform.name, width))
        })
        .collect()
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

/// Colour format for a render target.
///
/// `Hdr` is what `general.hdr` means: values above 1.0 survive instead of
/// clamping at the end of every pass. It matters most for bloom, whose
/// threshold is only meaningful if something can exceed it — `scene_example8`'s
/// sun is a shader result well past 1.0, and in an 8-bit target it lands at
/// exactly 1.0 with the rest of the sky, so a 0.65 threshold extracts almost
/// nothing and the sunset loses its glow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Ldr,
    Hdr,
}

impl Format {
    fn internal(self) -> u32 {
        match self {
            Format::Ldr => glow::RGBA8,
            Format::Hdr => glow::RGBA16F,
        }
    }

    fn component(self) -> u32 {
        match self {
            Format::Ldr => glow::UNSIGNED_BYTE,
            Format::Hdr => glow::HALF_FLOAT,
        }
    }
}

impl Target {
    pub fn new(gl: &glow::Context, width: u32, height: u32) -> Result<Target> {
        Target::with_format(gl, width, height, Format::Ldr)
    }

    pub fn with_format(gl: &glow::Context, width: u32, height: u32, format: Format) -> Result<Target> {
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
                gl_enum(format.internal()),
                signed_width,
                signed_height,
                0,
                glow::RGBA,
                format.component(),
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
    upload_with_wrap(gl, image, glow::CLAMP_TO_EDGE)
}

/// As `upload_texture`, but `REPEAT` — for the synthesized tiling noise
/// builtins, which shaders sample well outside `[0, 1]` (scaled by
/// `g_NoiseScale` and offset by `g_Time`).
pub fn upload_repeating_texture(gl: &glow::Context, image: &image::RgbaImage) -> Result<glow::Texture> {
    upload_with_wrap(gl, image, glow::REPEAT)
}

/// Replace the pixels of an existing texture, which must already be `image`'s
/// dimensions. The live simulator re-uploads a puppet or particle layer's
/// image every frame; reusing the texture object avoids churning GL handles.
///
/// The full re-spec via `tex_image_2d` (rather than `tex_sub_image_2d`) is
/// deliberate: it orphans the old storage so the driver does not stall this
/// upload behind the previous frame's sampling of the same texture.
pub fn update_texture(gl: &glow::Context, texture: glow::Texture, image: &image::RgbaImage) {
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper textures are nowhere near i32::MAX")]
    let (width, height) = (image.width() as i32, image.height() as i32);
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
    }
}

/// Free a texture handle (used when a per-frame layer image changes size and
/// its texture has to be reallocated).
pub fn delete_texture(gl: &glow::Context, texture: glow::Texture) {
    unsafe { gl.delete_texture(texture) };
}

fn upload_with_wrap(gl: &glow::Context, image: &image::RgbaImage, wrap: u32) -> Result<glow::Texture> {
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
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, gl_enum(wrap));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, gl_enum(wrap));
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

/// The fixed fullscreen quad a pass draws: two triangles, `a_Position`
/// (clip-space xyz) interleaved with `a_TexCoord` (0..1).
pub struct Quad {
    pub vertex_array: glow::VertexArray,
    _buffer: glow::Buffer,
}

/// The quad every effect pass uses to resample the previous result: V is
/// aligned so texel row 0 of the input lands at texel row 0 of the output
/// (`glFramebufferTexture`'s own row 0, at the bottom of NDC) — a pass with
/// no displacement leaves row order alone. This is what keeps texel row 0
/// meaning "the base image's own row 0" (its visual top, since that is what
/// `upload_texture` puts there) through any number of chained passes.
pub fn build_quad(gl: &glow::Context) -> Result<Quad> {
    #[rustfmt::skip]
    let vertices: [f32; 20] = [
        // x,    y,    z,   u,   v
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0, -1.0, 0.0,  1.0, 0.0,
        -1.0,  1.0, 0.0,  0.0, 1.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
    ];
    build_quad_from(gl, &vertices)
}

/// The quad `blit_to_screen` uses instead: going from texel-row-0-is-top
/// texture space to an actual displayed window needs exactly one flip,
/// since GL always rasterizes NDC's top edge to the window's top row. `V=0`
/// at NDC's top supplies that flip directly, the same way `read_rgba`
/// supplies it by reversing rows after `glReadPixels` for the PNG/video path.
pub fn build_display_quad(gl: &glow::Context) -> Result<Quad> {
    #[rustfmt::skip]
    let vertices: [f32; 20] = [
        // x,    y,    z,   u,   v
        -1.0, -1.0, 0.0,  0.0, 1.0,
         1.0, -1.0, 0.0,  1.0, 1.0,
        -1.0,  1.0, 0.0,  0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 0.0,
    ];
    build_quad_from(gl, &vertices)
}

fn build_quad_from(gl: &glow::Context, vertices: &[f32; 20]) -> Result<Quad> {
    // The vertex layout is fixed (xyz, then uv), so its byte offsets are
    // compile-time constants rather than a `size_of::<f32>()` cast.
    const STRIDE: i32 = 5 * 4;
    const UV_OFFSET: i32 = 3 * 4;
    let bytes = bytes_of(vertices);

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
        // Both stages' declarations reach here, and where they disagree only
        // the linked width can be uploaded.
        if call.program.float_widths.get(*name).is_some_and(|width| *width != values.len()) {
            continue;
        }
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

/// A trivial passthrough program: samples one texture and draws it fullscreen
/// onto whichever framebuffer is bound. Its own attribute locations are
/// pinned explicitly, unlike the effect passes' shim-translated shaders,
/// since this one is ours to write from scratch.
pub struct BlitProgram {
    program: Program,
}

const BLIT_VERTEX: &str = "#version 330 core\n\
    layout(location = 0) in vec3 a_Position;\n\
    layout(location = 1) in vec2 a_TexCoord;\n\
    out vec2 v_TexCoord;\n\
    void main() {\n\
        v_TexCoord = a_TexCoord;\n\
        gl_Position = vec4(a_Position, 1.0);\n\
    }\n";
const BLIT_FRAGMENT: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Texture;\n\
    void main() {\n\
        o_Color = texture(u_Texture, v_TexCoord);\n\
    }\n";

pub fn compile_blit_program(gl: &glow::Context) -> Result<BlitProgram> {
    Ok(BlitProgram { program: compile_program(gl, BLIT_VERTEX, BLIT_FRAGMENT)? })
}

/// The sub-rectangle of a `window`-sized viewport that shows `content` at its
/// own aspect ratio, centered, letterboxed instead of stretched — `(x, y,
/// width, height)` in the framebuffer's own bottom-left-origin coordinates.
fn letterbox(content: (u32, u32), window: (i32, i32)) -> (i32, i32, i32, i32) {
    #[expect(clippy::cast_precision_loss, reason = "wallpaper/window dimensions are nowhere near f32's 2^24 exact range")]
    let (content_w, content_h, window_w, window_h) =
        (content.0 as f32, content.1 as f32, window.0 as f32, window.1 as f32);
    let scale = (window_w / content_w).min(window_h / content_h);
    #[expect(clippy::cast_possible_truncation, reason = "scaling a window-sized rect down; always fits back in i32")]
    let (width, height) = ((content_w * scale) as i32, (content_h * scale) as i32);
    (((window.0 - width) / 2).max(0), ((window.1 - height) / 2).max(0), width, height)
}

/// Draw `texture` to the window (framebuffer 0), letterboxed within
/// `window`'s dimensions to preserve `content`'s own aspect ratio — the
/// bars are cleared to black rather than stretching the image to fill an
/// arbitrarily-resized window. `quad` should be `build_display_quad`'s, not
/// `build_quad`'s — see the difference between the two.
pub fn blit_to_screen(
    gl: &glow::Context,
    blit: &BlitProgram,
    quad: &Quad,
    texture: glow::Texture,
    content: (u32, u32),
    window: (i32, i32),
) {
    let (x, y, width, height) = letterbox(content, window);
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.viewport(0, 0, window.0, window.1);
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.clear(glow::COLOR_BUFFER_BIT);

        gl.viewport(x, y, width, height);
        gl.use_program(Some(blit.program.handle));
        gl.bind_vertex_array(Some(quad.vertex_array));
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        if let Some(location) = gl.get_uniform_location(blit.program.handle, "u_Texture") {
            gl.uniform_1_i32(Some(&location), 0);
        }
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.bind_vertex_array(None);
        gl.use_program(None);
    }
}

/// Copies a rectangle out of the composited frame into a target of that size —
/// how a composition layer gets its input, since what it renders is whatever
/// the frame already holds underneath it.
pub struct RegionCopy {
    program: Program,
    quad: Quad,
}

const REGION_VERTEX: &str = "#version 330 core\n\
    layout(location = 0) in vec3 a_Position;\n\
    layout(location = 1) in vec2 a_TexCoord;\n\
    uniform vec2 u_UvOffset;\n\
    uniform vec2 u_UvScale;\n\
    out vec2 v_TexCoord;\n\
    void main() {\n\
        v_TexCoord = u_UvOffset + a_TexCoord * u_UvScale;\n\
        gl_Position = vec4(a_Position, 1.0);\n\
    }\n";
const REGION_FRAGMENT: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Texture;\n\
    void main() {\n\
        o_Color = texture(u_Texture, v_TexCoord);\n\
    }\n";

pub fn compile_region_copy(gl: &glow::Context) -> Result<RegionCopy> {
    Ok(RegionCopy { program: compile_program(gl, REGION_VERTEX, REGION_FRAGMENT)?, quad: build_quad(gl)? })
}

/// Fill `region` with the `(left, top, width, height)` rectangle of `source`,
/// a `source_size` texture in the same row-preserving orientation as the
/// composite target. Sampling is clamped at the edges, so a rectangle that
/// hangs off the canvas smears its border rather than wrapping.
pub fn copy_region(
    gl: &glow::Context,
    copier: &RegionCopy,
    region: &Target,
    source: glow::Texture,
    source_size: (u32, u32),
    (left, top, width, height): (i32, i32, i32, i32),
) {
    #[expect(clippy::cast_precision_loss, reason = "canvas dimensions, nowhere near f32's 2^24 exact range")]
    let (source_w, source_h) = (source_size.0 as f32, source_size.1 as f32);
    #[expect(clippy::cast_precision_loss, reason = "canvas coordinates, nowhere near f32's 2^24 exact range")]
    let (offset, scale) = (
        [left as f32 / source_w, top as f32 / source_h],
        [width as f32 / source_w, height as f32 / source_h],
    );

    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (target_w, target_h) = (region.width as i32, region.height as i32);
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(region.framebuffer));
        gl.viewport(0, 0, target_w, target_h);
        gl.disable(glow::BLEND);
        gl.use_program(Some(copier.program.handle));
        gl.bind_vertex_array(Some(copier.quad.vertex_array));
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(source));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, gl_enum(glow::CLAMP_TO_EDGE));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, gl_enum(glow::CLAMP_TO_EDGE));
        if let Some(location) = gl.get_uniform_location(copier.program.handle, "u_Texture") {
            gl.uniform_1_i32(Some(&location), 0);
        }
        if let Some(location) = gl.get_uniform_location(copier.program.handle, "u_UvOffset") {
            gl.uniform_2_f32_slice(Some(&location), &offset);
        }
        if let Some(location) = gl.get_uniform_location(copier.program.handle, "u_UvScale") {
            gl.uniform_2_f32_slice(Some(&location), &scale);
        }
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.bind_vertex_array(None);
        gl.use_program(None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }
}

/// Draws one placed layer texture into a canvas-sized target, under a blend
/// mode. The live simulator runs each layer's effect chain separately and then
/// stacks the results here, the GPU equivalent of `compose::flatten`.
pub struct LayerCompositor {
    program: Program,
    quad: Quad,
}

/// Places the layer's quad itself rather than leaning on the viewport, so it
/// can be rolled about its own centre.
///
/// `u_Rect` is the same `(left, top, width, height)` the viewport took before,
/// and at zero roll this reproduces it exactly: the corner at texcoord (s, t)
/// lands at `(left + s·w, top + t·h)`, which is what `glViewport` did.
/// `v_Backdrop` stays in the *unrotated* rect so the copied backdrop still
/// lines up; a rolled corner reaching outside it clamps, which is a pixel or
/// two at the angles the corpus uses.
const COMPOSITE_VERTEX: &str = "#version 330 core\n\
    layout(location = 0) in vec3 a_Position;\n\
    layout(location = 1) in vec2 a_TexCoord;\n\
    uniform vec4 u_Rect;\n\
    uniform vec2 u_Target;\n\
    uniform vec2 u_Roll;\n\
    out vec2 v_TexCoord;\n\
    out vec2 v_Backdrop;\n\
    void main() {\n\
        vec2 local = (a_TexCoord - 0.5) * u_Rect.zw;\n\
        vec2 rolled = vec2(local.x * u_Roll.x - local.y * u_Roll.y,\n\
                           local.x * u_Roll.y + local.y * u_Roll.x);\n\
        vec2 pixel = u_Rect.xy + u_Rect.zw * 0.5 + rolled;\n\
        v_TexCoord = a_TexCoord;\n\
        v_Backdrop = (pixel - u_Rect.xy) / u_Rect.zw;\n\
        gl_Position = vec4(pixel / u_Target * 2.0 - 1.0, 0.0, 1.0);\n\
    }\n";
/// The shim's blend-mode dispatch, reused verbatim.
///
/// It is self-contained GLSL — no includes, no HLSL intrinsics — so it drops
/// straight into the compositor, and reusing it means the layer blend and the
/// in-shader `ApplyBlending` an effect pass calls can never disagree. The
/// numbering was checked against Wallpaper Engine's own `common_blending.h`
/// and matches arm for arm.
const BLENDING: &str = include_str!("../shader/glsl/common_blending.glsl");

/// The blend-mode compositor's `main`.
///
/// `scene.json`'s `colorBlendMode` picks a Photoshop-style blend of the layer
/// against the frame beneath it. Unlike `translucent`/`additive` — which are GL
/// blend state — this needs the destination readable, so the backdrop is copied
/// in as a texture and the blend runs here.
///
/// The wiring is `genericimage2.frag`'s, which is where Wallpaper Engine does
/// this: it binds `_rt_FullFrameBuffer` under `#if BLENDMODE` and finishes with
///
/// ```glsl
/// gl_FragColor.rgb = ApplyBlending(BLENDMODE, screen.rgb, gl_FragColor.rgb, gl_FragColor.a);
/// gl_FragColor.a = screen.a;
/// ```
///
/// so the *frame* is the base, the layer is the blend, the layer's own alpha is
/// the opacity, and the result carries the frame's alpha — not the layer's. That
/// output then meets the frame again through the material's ordinary translucent
/// blend, which the `mix` by `dst.a` below folds in; with an opaque frame it is
/// the identity, and `scene_example8` clears to opaque.
///
/// `scene_example8` is the only corpus scene that uses this at all: four
/// objects, at modes 1 (darken), 4 (linear burn), 11 (overlay) and 12 (soft
/// light). All four darken, which is why compositing them plain-over pasted its
/// clouds onto the sky as bright slabs.
const COMPOSITE_MAIN: &str = "\n\
    in vec2 v_TexCoord;\n\
    in vec2 v_Backdrop;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Texture;\n\
    uniform sampler2D u_Backdrop;\n\
    uniform int u_Mode;\n\
    uniform float u_Alpha;\n\
    void main() {\n\
        vec4 src = texture(u_Texture, v_TexCoord);\n\
        src *= u_Alpha;\n\
        if (u_Mode == 0) { o_Color = src; return; }\n\
        vec4 dst = texture(u_Backdrop, clamp(v_Backdrop, 0.0, 1.0));\n\
        vec3 blended = ApplyBlending(u_Mode, dst.rgb, src.rgb, src.a);\n\
        o_Color = vec4(mix(dst.rgb, blended, dst.a), dst.a);\n\
    }\n";

pub fn compile_layer_compositor(gl: &glow::Context) -> Result<LayerCompositor> {
    let fragment = format!("#version 330 core\n{BLENDING}\n{COMPOSITE_MAIN}");
    Ok(LayerCompositor {
        program: compile_program(gl, COMPOSITE_VERTEX, &fragment)?,
        // Row-preserving, same as an effect pass: the target ends up with
        // texel row 0 == the canvas's visual top, which `blit_to_screen` then
        // flips once for the window.
        quad: build_quad(gl)?,
    })
}

/// Fill `target` with a solid colour (the scene background) before any layer
/// is composited onto it.
pub fn clear_target(gl: &glow::Context, target: &Target, color: [f32; 4]) {
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (width, height) = (target.width as i32, target.height as i32);
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(target.framebuffer));
        gl.viewport(0, 0, width, height);
        gl.disable(glow::BLEND);
        gl.clear_color(color[0], color[1], color[2], color[3]);
        gl.clear(glow::COLOR_BUFFER_BIT);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }
}

/// Composite `texture` into `target` at pixel rectangle `(left, top, width,
/// height)` measured from the canvas's top-left. `additive` picks between
/// `dst·(1-srcA) + src·srcA` (normal) and `dst + src·srcA` with the alpha
/// lifted to the brighter of the two (additive) — the same two rules
/// `compose::blit` uses on the CPU.
/// Composite a layer with `scene.json`'s `colorBlendMode` applied
/// against `backdrop` — the frame beneath this layer, already copied out of
/// `target` at the same rectangle. Mode 0 ignores the backdrop entirely and
/// takes the fixed-function path, which is every layer in the corpus but four.
#[expect(clippy::too_many_arguments, reason = "one layer's full composite state")]
pub fn composite_layer_blended(
    gl: &glow::Context,
    compositor: &LayerCompositor,
    target: &Target,
    texture: glow::Texture,
    (left, top, width, height): (i32, i32, i32, i32),
    additive: bool,
    mode: i32,
    backdrop: Option<glow::Texture>,
    roll: f32,
    alpha: f32,
) {
    let blended = mode != 0 && backdrop.is_some();
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (target_width, target_height) = (target.width as i32, target.height as i32);
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(target.framebuffer));
        // The quad places itself now (see `COMPOSITE_VERTEX`), so the viewport
        // is the whole target — a rolled layer reaches outside its own rect.
        gl.viewport(0, 0, target_width, target_height);
        if blended {
            // The shader has already folded the backdrop in, so writing must
            // replace rather than blend a second time.
            gl.disable(glow::BLEND);
        } else if additive {
            gl.enable(glow::BLEND);
            gl.blend_equation_separate(glow::FUNC_ADD, glow::MAX);
            gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE, glow::ONE, glow::ONE);
        } else {
            gl.enable(glow::BLEND);
            gl.blend_equation_separate(glow::FUNC_ADD, glow::FUNC_ADD);
            gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );
        }

        gl.use_program(Some(compositor.program.handle));
        gl.bind_vertex_array(Some(compositor.quad.vertex_array));
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        if let Some(location) = gl.get_uniform_location(compositor.program.handle, "u_Texture") {
            gl.uniform_1_i32(Some(&location), 0);
        }
        if let Some(location) = gl.get_uniform_location(compositor.program.handle, "u_Mode") {
            gl.uniform_1_i32(Some(&location), if blended { mode } else { 0 });
        }
        #[expect(clippy::cast_precision_loss, reason = "canvas coordinates, nowhere near 2^24")]
        let rect = [left as f32, top as f32, width as f32, height as f32];
        #[expect(clippy::cast_precision_loss, reason = "canvas dimensions, nowhere near 2^24")]
        let size = [target.width as f32, target.height as f32];
        if let Some(location) = gl.get_uniform_location(compositor.program.handle, "u_Rect") {
            gl.uniform_4_f32_slice(Some(&location), &rect);
        }
        if let Some(location) = gl.get_uniform_location(compositor.program.handle, "u_Target") {
            gl.uniform_2_f32_slice(Some(&location), &size);
        }
        if let Some(location) = gl.get_uniform_location(compositor.program.handle, "u_Roll") {
            gl.uniform_2_f32_slice(Some(&location), &[roll.cos(), roll.sin()]);
        }
        if let Some(location) = gl.get_uniform_location(compositor.program.handle, "u_Alpha") {
            gl.uniform_1_f32(Some(&location), alpha.clamp(0.0, 1.0));
        }
        gl.active_texture(glow::TEXTURE1);
        // Bind *something* even when there is no backdrop: the sampler is
        // declared either way, and an incomplete texture unit makes the whole
        // draw undefined on a strict driver (macOS logs it outright).
        gl.bind_texture(glow::TEXTURE_2D, Some(backdrop.unwrap_or(texture)));
        if let Some(location) = gl.get_uniform_location(compositor.program.handle, "u_Backdrop") {
            gl.uniform_1_i32(Some(&location), 1);
        }
        gl.active_texture(glow::TEXTURE0);
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

        gl.bind_vertex_array(None);
        gl.use_program(None);
        gl.disable(glow::BLEND);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_pillarboxes_when_the_window_is_relatively_taller() {
        let (x, y, width, height) = letterbox((1600, 900), (900, 900));
        assert_eq!((width, height), (900, 506), "scaled to fit the narrower dimension");
        assert_eq!(x, 0, "full width used, no horizontal bars");
        assert_eq!(y, 197, "vertical bars split evenly above and below");
    }

    #[test]
    fn letterbox_letterboxes_when_the_window_is_relatively_wider() {
        let (x, y, width, height) = letterbox((1600, 900), (1600, 1600));
        assert_eq!((width, height), (1600, 900), "scaled to fit the shorter dimension");
        assert_eq!(x, 0, "full width used, no horizontal bars");
        assert_eq!(y, 350, "vertical bars split evenly above and below");
    }

    #[test]
    fn letterbox_fills_the_window_exactly_when_aspect_ratios_match() {
        assert_eq!(letterbox((1920, 1080), (1920, 1080)), (0, 0, 1920, 1080));
    }
}

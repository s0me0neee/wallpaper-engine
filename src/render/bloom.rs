//! Scene bloom.
//!
//! `scene.json`'s `general` block carries `bloom`, and when it is on Wallpaper
//! Engine runs a post-process over the finished frame that we were skipping
//! entirely. It is not a small effect: `scene_example8` is a sunset, and its sun
//! is a large saturated glow that exists *only* because of this — without it the
//! horizon renders about 30% too dim and the sun reads as a few thin lit cloud
//! edges rather than a bloom.
//!
//! The extract and the combine are Wallpaper Engine's own, read from
//! `downsample_quarter_bloom.frag` and `combine.frag`: average four taps, scale
//! by how far the brightest channel clears the threshold, push saturation,
//! multiply by tint — then add the blurred result straight back onto the frame.
//!
//! The blur between them is a mip chain rather than WE's fixed
//! quarter-then-eighth pair, because that is what its HDR parameters describe:
//! `bloomhdriterations` levels, combined on the way back up by
//! `bloomhdrscatter`. A wide bloom is most of what makes a sunset read as one,
//! and two levels cannot spread light far enough across a 4K frame. WE's exact
//! HDR path (`hdr_downsample` with bicubic upsampling) also carries a feather
//! term we do not reproduce.
//!
//! Bloom only means anything if something can exceed the threshold, so an HDR
//! scene must render into floating-point targets — see `pass::Format`.

use super::pass::{Format, Program, Quad, Target, build_quad, compile_program};
use anyhow::Result;
use glow::HasContext;

const VERTEX: &str = "#version 330 core\n\
    layout(location = 0) in vec3 a_Position;\n\
    layout(location = 1) in vec2 a_TexCoord;\n\
    out vec2 v_TexCoord;\n\
    void main() {\n\
        v_TexCoord = a_TexCoord;\n\
        gl_Position = vec4(a_Position, 1.0);\n\
    }\n";

/// `downsample_quarter_bloom.frag`, minus the strength — which this applies
/// once at the end instead, so the mip chain carries unscaled light.
const EXTRACT: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Texture;\n\
    uniform vec2 u_Texel;\n\
    uniform float u_Threshold;\n\
    uniform vec3 u_Tint;\n\
    void main() {\n\
        vec3 albedo = texture(u_Texture, v_TexCoord + vec2(-u_Texel.x, -u_Texel.y)).rgb\n\
                    + texture(u_Texture, v_TexCoord + vec2( u_Texel.x, -u_Texel.y)).rgb\n\
                    + texture(u_Texture, v_TexCoord + vec2(-u_Texel.x,  u_Texel.y)).rgb\n\
                    + texture(u_Texture, v_TexCoord + vec2( u_Texel.x,  u_Texel.y)).rgb;\n\
        albedo *= 0.25;\n\
        float scale = max(max(albedo.x, albedo.y), albedo.z);\n\
        albedo *= clamp(scale - u_Threshold, 0.0, 1.0);\n\
        float grayscale = dot(vec3(0.2989, 0.5870, 0.1140), albedo);\n\
        albedo = -grayscale + albedo * 2.0;\n\
        o_Color = vec4(max(vec3(0.0), albedo * u_Tint), 1.0);\n\
    }\n";

/// Four bilinear taps on the corners of the source texel quad — one level down.
const DOWN: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Texture;\n\
    uniform vec2 u_Texel;\n\
    void main() {\n\
        vec3 albedo = texture(u_Texture, v_TexCoord + vec2(-u_Texel.x, -u_Texel.y)).rgb\n\
                    + texture(u_Texture, v_TexCoord + vec2( u_Texel.x, -u_Texel.y)).rgb\n\
                    + texture(u_Texture, v_TexCoord + vec2(-u_Texel.x,  u_Texel.y)).rgb\n\
                    + texture(u_Texture, v_TexCoord + vec2( u_Texel.x,  u_Texel.y)).rgb;\n\
        o_Color = vec4(albedo * 0.25, 1.0);\n\
    }\n";

/// A 3x3 tent back up a level, mixed into what is already there by `u_Scatter`.
const UP: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Texture;\n\
    uniform vec2 u_Texel;\n\
    uniform float u_Scatter;\n\
    void main() {\n\
        vec2 t = u_Texel;\n\
        vec3 albedo =\n\
              texture(u_Texture, v_TexCoord + vec2(-t.x,  t.y)).rgb * 1.0\n\
            + texture(u_Texture, v_TexCoord + vec2( 0.0,  t.y)).rgb * 2.0\n\
            + texture(u_Texture, v_TexCoord + vec2( t.x,  t.y)).rgb * 1.0\n\
            + texture(u_Texture, v_TexCoord + vec2(-t.x,  0.0)).rgb * 2.0\n\
            + texture(u_Texture, v_TexCoord).rgb * 4.0\n\
            + texture(u_Texture, v_TexCoord + vec2( t.x,  0.0)).rgb * 2.0\n\
            + texture(u_Texture, v_TexCoord + vec2(-t.x, -t.y)).rgb * 1.0\n\
            + texture(u_Texture, v_TexCoord + vec2( 0.0, -t.y)).rgb * 2.0\n\
            + texture(u_Texture, v_TexCoord + vec2( t.x, -t.y)).rgb * 1.0;\n\
        o_Color = vec4(albedo / 16.0, u_Scatter);\n\
    }\n";

/// `combine.frag` is `albedo += bloom`, so the draw is additive; the strength
/// rides in as the source factor.
const COMBINE: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Texture;\n\
    uniform float u_Strength;\n\
    void main() {\n\
        o_Color = vec4(texture(u_Texture, v_TexCoord).rgb * u_Strength, 1.0);\n\
    }\n";

pub struct Bloom {
    extract: Program,
    down: Program,
    up: Program,
    combine: Program,
    quad: Quad,
    /// Successively halved targets. Level 0 is half the frame.
    levels: Vec<Target>,
}

/// The `general` block's bloom settings, as the shaders take them.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub strength: f32,
    pub threshold: f32,
    pub tint: [f32; 3],
    /// How much of the wider level bleeds back into the narrower one.
    pub scatter: f32,
    /// How many halvings to chain. Clamped to what the canvas can carry.
    pub iterations: u32,
}

pub fn compile(gl: &glow::Context, width: u32, height: u32, format: Format) -> Result<Bloom> {
    Ok(Bloom {
        extract: compile_program(gl, VERTEX, EXTRACT)?,
        down: compile_program(gl, VERTEX, DOWN)?,
        up: compile_program(gl, VERTEX, UP)?,
        combine: compile_program(gl, VERTEX, COMBINE)?,
        quad: build_quad(gl)?,
        levels: mip_chain(gl, width, height, format)?,
    })
}

/// Half-size targets, stopping before either side would reach 8 px — below
/// that a level contributes a flat wash and costs a pass to get it.
fn mip_chain(gl: &glow::Context, width: u32, height: u32, format: Format) -> Result<Vec<Target>> {
    let mut levels = Vec::new();
    let (mut w, mut h) = (width / 2, height / 2);
    while w >= 8 && h >= 8 && levels.len() < 10 {
        levels.push(Target::with_format(gl, w, h, format)?);
        w /= 2;
        h /= 2;
    }
    if levels.is_empty() {
        levels.push(Target::with_format(gl, width.max(1), height.max(1), format)?);
    }
    Ok(levels)
}

/// Extract, blur down and back up, and add the result over `frame` in place.
pub fn apply(gl: &glow::Context, bloom: &Bloom, frame: &Target, settings: Settings) {
    let depth = (settings.iterations.max(1) as usize).min(bloom.levels.len());

    draw_extract(gl, bloom, frame, settings);
    for level in 1..depth {
        draw_down(gl, bloom, level);
    }
    // Back up the chain, each level bleeding into the one above it.
    for level in (1..depth).rev() {
        draw_up(gl, bloom, level, settings.scatter);
    }
    draw_combine(gl, bloom, frame, settings.strength);
}

fn draw_extract(gl: &glow::Context, bloom: &Bloom, frame: &Target, settings: Settings) {
    let Some(into) = bloom.levels.first() else { return };
    unsafe {
        bind_pass(gl, into, bloom.extract.handle, bloom.quad.vertex_array, frame.texture);
        set_vec2(gl, bloom.extract.handle, "u_Texel", half_texel(frame));
        set_float(gl, bloom.extract.handle, "u_Threshold", settings.threshold);
        if let Some(location) = gl.get_uniform_location(bloom.extract.handle, "u_Tint") {
            gl.uniform_3_f32_slice(Some(&location), &settings.tint);
        }
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        finish(gl);
    }
}

fn draw_down(gl: &glow::Context, bloom: &Bloom, level: usize) {
    let (Some(source), Some(into)) = (bloom.levels.get(level - 1), bloom.levels.get(level)) else {
        return;
    };
    unsafe {
        bind_pass(gl, into, bloom.down.handle, bloom.quad.vertex_array, source.texture);
        set_vec2(gl, bloom.down.handle, "u_Texel", half_texel(source));
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        finish(gl);
    }
}

fn draw_up(gl: &glow::Context, bloom: &Bloom, level: usize, scatter: f32) {
    let (Some(source), Some(into)) = (bloom.levels.get(level), bloom.levels.get(level - 1)) else {
        return;
    };
    unsafe {
        bind_pass(gl, into, bloom.up.handle, bloom.quad.vertex_array, source.texture);
        set_vec2(gl, bloom.up.handle, "u_Texel", half_texel(source));
        set_float(gl, bloom.up.handle, "u_Scatter", scatter.clamp(0.0, 1.0));
        // The tent writes its coverage as alpha, so the mix *is* the blend.
        gl.enable(glow::BLEND);
        gl.blend_equation(glow::FUNC_ADD);
        gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ZERO, glow::ONE);
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.disable(glow::BLEND);
        finish(gl);
    }
}

fn draw_combine(gl: &glow::Context, bloom: &Bloom, frame: &Target, strength: f32) {
    let Some(source) = bloom.levels.first() else { return };
    unsafe {
        bind_pass(gl, frame, bloom.combine.handle, bloom.quad.vertex_array, source.texture);
        set_float(gl, bloom.combine.handle, "u_Strength", strength.max(0.0));
        // `albedo += bloom`, and the frame's own alpha is left alone.
        gl.enable(glow::BLEND);
        gl.blend_equation(glow::FUNC_ADD);
        gl.blend_func_separate(glow::ONE, glow::ONE, glow::ZERO, glow::ONE);
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.disable(glow::BLEND);
        finish(gl);
    }
}

/// Half a texel of `target`, the offset the four-tap filters want.
fn half_texel(target: &Target) -> [f32; 2] {
    #[expect(clippy::cast_precision_loss, reason = "canvas dimensions, nowhere near 2^24")]
    let texel = [0.5 / target.width as f32, 0.5 / target.height as f32];
    texel
}

unsafe fn bind_pass(
    gl: &glow::Context,
    into: &Target,
    program: glow::Program,
    quad: glow::VertexArray,
    texture: glow::Texture,
) {
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (width, height) = (into.width as i32, into.height as i32);
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into.framebuffer));
        gl.viewport(0, 0, width, height);
        gl.disable(glow::BLEND);
        gl.use_program(Some(program));
        gl.bind_vertex_array(Some(quad));
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        if let Some(location) = gl.get_uniform_location(program, "u_Texture") {
            gl.uniform_1_i32(Some(&location), 0);
        }
    }
}

unsafe fn finish(gl: &glow::Context) {
    unsafe {
        gl.bind_vertex_array(None);
        gl.use_program(None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }
}

unsafe fn set_float(gl: &glow::Context, program: glow::Program, name: &str, value: f32) {
    unsafe {
        if let Some(location) = gl.get_uniform_location(program, name) {
            gl.uniform_1_f32(Some(&location), value);
        }
    }
}

unsafe fn set_vec2(gl: &glow::Context, program: glow::Program, name: &str, value: [f32; 2]) {
    unsafe {
        if let Some(location) = gl.get_uniform_location(program, name) {
            gl.uniform_2_f32_slice(Some(&location), &value);
        }
    }
}

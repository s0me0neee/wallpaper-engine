//! Particle systems drawn as instanced quads.
//!
//! The live window's frame budget was, measurably, tiny-skia filling sprite
//! rects: `scene_example3` spent 66 ms of a 66 ms frame there and 0 ms on the
//! GPU (plan.md §4.16). The simulation itself is free — a few hundred thousand
//! float operations that `rayon` already spreads over every core — so this
//! keeps the simulation exactly as it was and replaces only the fill, drawing
//! one instanced quad per `particle::DrawItem` straight into a render target.
//! That removes the per-frame texture upload with it: the pixels are already
//! where the compositor wants them.
//!
//! **The target holds premultiplied alpha**, unlike every other layer texture,
//! because that is what makes GL's fixed-function blending reproduce
//! tiny-skia's `Plus`/`SourceOver` exactly. `pass::composite_layer_blended`
//! takes a flag for it rather than this pass paying a full-canvas
//! unpremultiply it would only have to undo.
//!
//! Nothing here is platform-specific: plain GL 3.3 core, the same context the
//! effect chains already run in.

use crate::render::pass::{self, Program, Target};
use crate::scene::particle::{DrawList, Shape};
use anyhow::Result;
use glam::Vec2;
use glow::HasContext;

/// One sprite sheet, with the size the quad's aspect comes from.
pub struct Sprite {
    pub texture: glow::Texture,
    pub size: (u32, u32),
}

/// The compiled program and the two buffers a frame's instances stream through.
pub struct Particles {
    program: Program,
    vertex_array: glow::VertexArray,
    _corners: glow::Buffer,
    instances: glow::Buffer,
}

/// Per-instance attributes, in the order the vertex shader declares them.
///
/// `roll.z` is the pattern mode, not a third axis: a sprite quad carries its
/// pattern with it as it rolls, while a trail's pattern stays pinned to the
/// head's rect in canvas space. That is tiny-skia's own split — `fill_rect`
/// gets the rotation, `stroke_path` gets the identity — and the tail picking
/// up the sprite's clamped edge is what makes a trail fade.
#[repr(C)]
struct Instance {
    rect: [f32; 4],
    roll: [f32; 4],
    uv_rect: [f32; 4],
    color: [f32; 4],
}

/// Four `vec4` attributes of four bytes each.
const INSTANCE_STRIDE: i32 = 4 * 4 * 4;

const VERTEX: &str = "#version 330 core\n\
    layout(location = 0) in vec2 a_Corner;\n\
    layout(location = 1) in vec4 i_Rect;\n\
    layout(location = 2) in vec4 i_Roll;\n\
    layout(location = 3) in vec4 i_UvRect;\n\
    layout(location = 4) in vec4 i_Color;\n\
    uniform vec2 u_Target;\n\
    out vec2 v_TexCoord;\n\
    out vec4 v_Color;\n\
    void main() {\n\
        vec2 local = a_Corner * i_Rect.zw * 2.0;\n\
        vec2 rolled = vec2(local.x * i_Roll.x - local.y * i_Roll.y,\n\
                           local.x * i_Roll.y + local.y * i_Roll.x);\n\
        vec2 pixel = i_Rect.xy + rolled;\n\
        vec2 pattern = mix(local, pixel - i_UvRect.xy, i_Roll.z);\n\
        v_TexCoord = pattern / (i_UvRect.zw * 2.0) + 0.5;\n\
        v_Color = i_Color;\n\
        gl_Position = vec4(pixel / u_Target * 2.0 - 1.0, 0.0, 1.0);\n\
    }\n";

/// `clamp` is `SpreadMode::Pad`: a trail reaches well outside the head's rect
/// and must find the sprite's edge texel there, not a repeat of it.
const FRAGMENT: &str = "#version 330 core\n\
    in vec2 v_TexCoord;\n\
    in vec4 v_Color;\n\
    out vec4 o_Color;\n\
    uniform sampler2D u_Sprite;\n\
    void main() {\n\
        vec4 texel = texture(u_Sprite, clamp(v_TexCoord, 0.0, 1.0));\n\
        float alpha = texel.a * v_Color.a;\n\
        o_Color = vec4(texel.rgb * v_Color.rgb * alpha, alpha);\n\
    }\n";

pub fn compile(gl: &glow::Context) -> Result<Particles> {
    let program = pass::compile_program(gl, VERTEX, FRAGMENT)?;

    // Corners of a unit quad, as a triangle strip centred on the origin, so
    // the vertex shader scales by the instance's half-extent.
    #[rustfmt::skip]
    let corners: [f32; 8] = [
        -0.5, -0.5,
         0.5, -0.5,
        -0.5,  0.5,
         0.5,  0.5,
    ];

    let vertex_array = unsafe { gl.create_vertex_array() }.map_err(|error| anyhow::anyhow!(error))?;
    let corner_buffer = unsafe { gl.create_buffer() }.map_err(|error| anyhow::anyhow!(error))?;
    let instances = unsafe { gl.create_buffer() }.map_err(|error| anyhow::anyhow!(error))?;
    unsafe {
        gl.bind_vertex_array(Some(vertex_array));

        gl.bind_buffer(glow::ARRAY_BUFFER, Some(corner_buffer));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, pass::bytes_of(&corners), glow::STATIC_DRAW);
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 2 * 4, 0);

        gl.bind_buffer(glow::ARRAY_BUFFER, Some(instances));
        for slot in 0..4 {
            let location = 1 + slot;
            gl.enable_vertex_attrib_array(location);
            gl.vertex_attrib_pointer_f32(location, 4, glow::FLOAT, false, INSTANCE_STRIDE, slot.cast_signed() * 4 * 4);
            gl.vertex_attrib_divisor(location, 1);
        }

        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    Ok(Particles { program, vertex_array, _corners: corner_buffer, instances })
}

/// A run of instances sharing one sprite and one blend, so it draws in one call.
struct Batch {
    sprite: usize,
    additive: bool,
    start: usize,
    count: usize,
}

/// `(left, top, width, height)` in canvas pixels.
pub type Rect = (i32, i32, i32, i32);

/// What the particles are being drawn onto.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Ground {
    /// A scratch target this system owns, cleared to transparent first, to be
    /// composited afterwards. Needed by a system that mixes blends, or whose
    /// layer carries a `colorBlendMode` or an alpha track.
    Fresh,
    /// The frame itself. Nothing is cleared and nothing is composited: the
    /// particles blend straight onto what is already there.
    ///
    /// This is the path that matters on a big scene. A canvas-sized composite
    /// per system is what actually costs — `scene_example8`'s 35 fog and smoke
    /// systems each cover most of a 4K canvas, so scissoring saves nothing and
    /// the composites alone were 82 ms of an 88 ms frame, against 6 ms for the
    /// particles themselves.
    Over,
}

/// Draw `list` into `target` at `alpha`, returning the rectangle it touched.
///
/// `None` means nothing was drawn — an empty system, or one whose particles
/// have not been emitted yet — and the caller can skip the composite entirely.
/// On `Ground::Fresh` the target is cleared inside that rectangle only, never
/// all of it.
pub fn draw(
    gl: &glow::Context,
    particles: &Particles,
    target: &Target,
    list: &DrawList,
    sprites: &[Sprite],
    alpha: f32,
    ground: Ground,
) -> Option<Rect> {
    let sizes: Vec<(u32, u32)> = sprites.iter().map(|sprite| sprite.size).collect();
    let (instances, batches, bounds) = build(list, &sizes, alpha.clamp(0.0, 1.0));
    let rect = clip(bounds?, target);
    if instances.is_empty() || rect.2 <= 0 || rect.3 <= 0 {
        return None;
    }

    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (width, height) = (target.width as i32, target.height as i32);
    unsafe {
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(particles.instances));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, instance_bytes(&instances), glow::STREAM_DRAW);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(target.framebuffer));
        gl.viewport(0, 0, width, height);
        if ground == Ground::Fresh {
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(rect.0, flip_row(rect, height), rect.2, rect.3);
            gl.disable(glow::BLEND);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }

        gl.use_program(Some(particles.program.handle));
        gl.bind_vertex_array(Some(particles.vertex_array));
        #[expect(clippy::cast_precision_loss, reason = "canvas dimensions, nowhere near 2^24")]
        let size = [target.width as f32, target.height as f32];
        if let Some(location) = gl.get_uniform_location(particles.program.handle, "u_Target") {
            gl.uniform_2_f32_slice(Some(&location), &size);
        }
        if let Some(location) = gl.get_uniform_location(particles.program.handle, "u_Sprite") {
            gl.uniform_1_i32(Some(&location), 0);
        }
        gl.active_texture(glow::TEXTURE0);
        gl.enable(glow::BLEND);

        // The instance buffer stays bound: each batch re-points the attributes
        // at its own slice of it. `glDrawArraysInstancedBaseInstance` would
        // say this in one call, but it is GL 4.2 and macOS stops at 4.1.
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(particles.instances));

        let mut bound = None;
        let mut blend = None;
        for batch in &batches {
            let Some(sprite) = sprites.get(batch.sprite) else { continue };
            if bound != Some(batch.sprite) {
                gl.bind_texture(glow::TEXTURE_2D, Some(sprite.texture));
                bound = Some(batch.sprite);
            }
            if blend != Some(batch.additive) {
                // Premultiplied source throughout, so `Plus` is a plain sum and
                // `SourceOver` needs no source factor of its own.
                if batch.additive {
                    // Alpha takes the brighter of the two rather than summing,
                    // drawn straight onto the frame: that is the rule the layer
                    // composite applied to a whole additive layer, and this is
                    // the same step with the layer taken out of the middle.
                    let alpha = if ground == Ground::Over { glow::MAX } else { glow::FUNC_ADD };
                    gl.blend_equation_separate(glow::FUNC_ADD, alpha);
                    gl.blend_func_separate(glow::ONE, glow::ONE, glow::ONE, glow::ONE);
                } else {
                    gl.blend_equation(glow::FUNC_ADD);
                    gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
                }
                blend = Some(batch.additive);
            }
            #[expect(
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "instance counts are bounded by the presets' own maxcount"
            )]
            let (base, count) = ((batch.start as i32) * INSTANCE_STRIDE, batch.count as i32);
            for slot in 0..4 {
                #[expect(clippy::cast_possible_wrap, reason = "four fixed attribute slots")]
                let offset = base + slot as i32 * 4 * 4;
                gl.vertex_attrib_pointer_f32(1 + slot, 4, glow::FLOAT, false, INSTANCE_STRIDE, offset);
            }
            gl.draw_arrays_instanced(glow::TRIANGLE_STRIP, 0, 4, count);
        }
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        gl.bind_vertex_array(None);
        gl.use_program(None);
        gl.blend_equation(glow::FUNC_ADD);
        gl.disable(glow::BLEND);
        gl.disable(glow::SCISSOR_TEST);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }
    Some(rect)
}

/// GL measures the scissor box from the bottom row; our rectangles, like every
/// other canvas coordinate here, come from the top.
fn flip_row((_, top, _, height): Rect, target_height: i32) -> i32 {
    target_height - top - height
}

/// Round `bounds` out to whole pixels and clip it to the target.
fn clip(bounds: (Vec2, Vec2), target: &Target) -> Rect {
    #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvases are nowhere near i32::MAX")]
    let (width, height) = (target.width as i32, target.height as i32);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "particle coordinates are canvas pixels; the clamp below bounds them regardless"
    )]
    let (min, max) = (
        (bounds.0.floor().x as i32, bounds.0.floor().y as i32),
        (bounds.1.ceil().x as i32, bounds.1.ceil().y as i32),
    );
    let left = min.0.clamp(0, width);
    let top = min.1.clamp(0, height);
    (left, top, max.0.clamp(0, width) - left, max.1.clamp(0, height) - top)
}

/// Flatten a draw list into instances, runs that can share a draw call, and
/// the bounding box of everything in it.
///
/// Batching is by *consecutive* run rather than by sprite, because a
/// translucent system draws back-to-front and reordering it would let a far
/// particle cover a near one. Runs are long in practice: the order only breaks
/// where an `eventfollow` child rides its parent.
fn build(
    list: &DrawList,
    sizes: &[(u32, u32)],
    alpha: f32,
) -> (Vec<Instance>, Vec<Batch>, Option<(Vec2, Vec2)>) {
    let mut instances = Vec::with_capacity(list.items.len());
    let mut batches: Vec<Batch> = Vec::new();
    let mut bounds: Option<(Vec2, Vec2)> = None;

    for item in &list.items {
        let Some(&sprite) = sizes.get(item.sprite) else { continue };
        let additive = matches!(item.blend, crate::scene::model::Blend::Add);
        let start = instances.len();
        match &item.shape {
            Shape::Sprite { center, radius, rotation } => {
                let half = half_extent(sprite, *radius);
                instances.push(Instance {
                    rect: [center.x, center.y, half.x, half.y],
                    roll: [rotation.cos(), rotation.sin(), 0.0, 0.0],
                    uv_rect: [center.x, center.y, half.x, half.y],
                    color: [item.color.x, item.color.y, item.color.z, item.weight.clamp(0.0, 1.0) * alpha],
                });
                grow(&mut bounds, *center, half.length());
            }
            Shape::Trail { points, radius } => {
                let Some(head) = points.last() else { continue };
                let width = radius.max(1.0);
                let uv = half_extent(sprite, width);
                for pair in points.windows(2) {
                    let (from, to) = (pair[0], pair[1]);
                    let span = to - from;
                    let length = span.length();
                    if length <= f32::EPSILON {
                        continue;
                    }
                    let along = span / length;
                    // Extended by a half-width at each end: tiny-skia strokes
                    // these with round caps, and a bare segment would leave a
                    // notch at every joint.
                    let half = Vec2::new(length.mul_add(0.5, width * 0.5), width * 0.5);
                    let center = (from + to) * 0.5;
                    instances.push(Instance {
                        rect: [center.x, center.y, half.x, half.y],
                        roll: [along.x, along.y, 1.0, 0.0],
                        uv_rect: [head.x, head.y, uv.x, uv.y],
                        color: [item.color.x, item.color.y, item.color.z, item.weight.clamp(0.0, 1.0) * alpha],
                    });
                    grow(&mut bounds, center, half.length());
                }
            }
        }
        let count = instances.len() - start;
        if count == 0 {
            continue;
        }
        match batches.last_mut() {
            Some(open) if open.sprite == item.sprite && open.additive == additive => open.count += count,
            _ => batches.push(Batch { sprite: item.sprite, additive, start, count }),
        }
    }
    (instances, batches, bounds)
}

/// The sprite's rect at `radius`: the longer side is `radius`, the sprite's own
/// aspect sets the other. Squashing a 256x1280 rain streak into a square turns
/// a falling line into a blob (plan.md §4.10).
fn half_extent(sprite: (u32, u32), radius: f32) -> Vec2 {
    #[expect(clippy::cast_precision_loss, reason = "sprite sides are at most a few thousand")]
    let size = Vec2::new(sprite.0 as f32, sprite.1 as f32);
    let longest = size.x.max(size.y);
    if longest <= 0.0 {
        return Vec2::ZERO;
    }
    size * (radius / longest)
}

/// Extend `bounds` by a quad at `center` whose corners all sit within
/// `circumradius` — true at any roll, which is why the box is not computed
/// from the rotated corners.
fn grow(bounds: &mut Option<(Vec2, Vec2)>, center: Vec2, circumradius: f32) {
    let (low, high) = (center - circumradius, center + circumradius);
    match bounds {
        Some((min, max)) => {
            *min = min.min(low);
            *max = max.max(high);
        }
        None => *bounds = Some((low, high)),
    }
}

fn instance_bytes(instances: &[Instance]) -> &[u8] {
    // Safety: `Instance` is `repr(C)` and every field is an `f32` array, so it
    // has no padding and no invalid bit patterns; this reinterprets the same
    // allocation without resizing it.
    unsafe {
        std::slice::from_raw_parts(
            instances.as_ptr().cast::<u8>(),
            std::mem::size_of_val(instances),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::model::Blend;
    use crate::scene::particle::{DrawItem, DrawList};

    /// Only the sizes matter to `build`; the textures belong to the draw call.
    const SPRITES: [(u32, u32); 2] = [(256, 1280), (64, 64)];

    fn item(sprite: usize, blend: Blend, shape: Shape) -> DrawItem {
        DrawItem { sprite, blend, color: glam::Vec3::ONE, weight: 1.0, shape }
    }

    fn list(items: Vec<DrawItem>) -> DrawList {
        DrawList { canvas_px: (1920, 1080), items, unsupported: Vec::new() }
    }

    #[test]
    fn a_sprite_keeps_its_own_aspect() {
        // 256x1280 at radius 100 is a 20x100 half-extent, not 100x100 — the
        // streak/blob distinction §4.10 settled.
        let half = half_extent(SPRITES[0], 100.0);
        assert!((half.x - 20.0).abs() < 1e-4, "{half:?}");
        assert!((half.y - 100.0).abs() < 1e-4, "{half:?}");
    }

    #[test]
    fn consecutive_items_sharing_a_sprite_and_blend_share_a_draw_call() {
        let shape = || Shape::Sprite { center: Vec2::new(10.0, 10.0), radius: 4.0, rotation: 0.0 };
        let (instances, batches, _) = build(
            &list(vec![
                item(0, Blend::Add, shape()),
                item(0, Blend::Add, shape()),
                item(1, Blend::Add, shape()),
                item(0, Blend::Add, shape()),
            ]),
            &SPRITES,
            1.0,
        );
        assert_eq!(instances.len(), 4);
        // Three runs, not two: reordering the last one back onto the first
        // batch would move it in front of the sprite drawn between them.
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].count, 2);
    }

    #[test]
    fn a_blend_change_breaks_the_batch() {
        let shape = || Shape::Sprite { center: Vec2::ZERO, radius: 4.0, rotation: 0.0 };
        let (_, batches, _) = build(
            &list(vec![item(0, Blend::Add, shape()), item(0, Blend::Over, shape())]),
            &SPRITES,
            1.0,
        );
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn a_trail_becomes_one_quad_per_segment() {
        let points = vec![Vec2::new(0.0, 0.0), Vec2::new(30.0, 0.0), Vec2::new(60.0, 0.0)];
        let (instances, batches, _) = build(
            &list(vec![item(1, Blend::Add, Shape::Trail { points, radius: 8.0 })]),
            &SPRITES,
            1.0,
        );
        assert_eq!(instances.len(), 2);
        assert_eq!(batches.len(), 1);
        // Both segments anchor their pattern on the head, which is what makes
        // the tail fade into the sprite's clamped edge rather than repeat it.
        assert!((instances[0].uv_rect[0] - 60.0).abs() < 1e-4);
        assert!((instances[1].uv_rect[0] - 60.0).abs() < 1e-4);
        // Mode 1: the pattern stays in canvas space as the quad turns.
        assert!((instances[0].roll[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn the_dirty_rectangle_covers_every_particle_and_stops_at_the_canvas() {
        let items = vec![
            item(1, Blend::Add, Shape::Sprite { center: Vec2::new(-20.0, 40.0), radius: 10.0, rotation: 0.0 }),
            item(1, Blend::Add, Shape::Sprite { center: Vec2::new(500.0, 400.0), radius: 10.0, rotation: 0.0 }),
        ];
        let (_, _, bounds) = build(&list(items), &SPRITES, 1.0);
        let (min, max) = bounds.expect("two particles have a bounding box");
        assert!(min.x < 0.0 && max.x > 500.0);
    }

    #[test]
    fn an_empty_system_has_no_bounding_box() {
        let (instances, batches, bounds) = build(&list(Vec::new()), &SPRITES, 1.0);
        assert!(instances.is_empty() && batches.is_empty() && bounds.is_none());
    }
}

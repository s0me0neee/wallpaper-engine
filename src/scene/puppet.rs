//! Wallpaper Engine puppet-warp models (`models/*_puppet.mdl`).
//!
//! A puppet layer is a textured 2D triangle mesh with a small bone skeleton
//! and one or more baked animation clips (per-bone translation, Z rotation and
//! scale frames). Playing it means: sample the referenced clip at `g_Time`, build a
//! skin transform per bone (`animated_world * bind_world⁻¹`), blend each
//! vertex by its bone weights, then rasterize the deformed mesh through the
//! layer's texture. The result replaces the flat layer image in the per-layer
//! pipeline (`compose::prepare`), so a layer's effect chain still runs over
//! the warped image.
//!
//! The `MDLV0013` container layout and the skinning maths were reverse-
//! engineered by 3x-haust/workshop-wallpaper-bridge (MIT); ported here with a
//! fix for the 4-byte trailer that separates animation records (a two-clip
//! model in the corpus needs it).
//!
//! This module is coordinate and pixel arithmetic throughout — bone indices,
//! vertex counts, UV/position floats, raster bounds. The lossy-cast pedantic
//! lints fire on nearly every line and add nothing here (a puppet has at most
//! a few hundred vertices and its raster a few thousand pixels a side), so
//! they are scoped off module-wide rather than annotated per site.
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded mesh/pixel arithmetic; see module docs"
)]

use anyhow::{Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use image::{Rgba, RgbaImage};
use smallvec::SmallVec;
use std::io::{Cursor, Read, Seek, SeekFrom};

/// A bone-indexed transient collection. Corpus puppets have 2–10 bones; the
/// cap is 256, but the common case stays on the stack.
type PerBone<T> = SmallVec<[T; 16]>;

/// `pos(12) + 4 bone indices(16) + 4 weights(16) + uv(8)`, the layout the
/// vertex-format word's low half calls 9.
const VERTEX_SIZE: usize = 52;

/// Format 15 inserts a normal, a tangent and a handedness sign between the
/// position and the bone indices: `pos(12) + normal(12) + tangent(12) +
/// handedness(4) + bones(16) + weights(16) + uv(8)`. None of the three matters
/// to a flat warp — every normal in the corpus is exactly `(0, 0, 1)` — so
/// they are skipped, but the stride they add is what tells the two apart.
const VERTEX_SIZE_LIT: usize = 80;
const LIT_VERTEX_PREFIX: i64 = 28;
const FRAME_SIZE: usize = 36; // 9 floats: translation(3), euler(3), scale(3)
const BONE_MATRIX_SIZE: usize = 64; // 4x4 f32 bind matrix, unused (bind pose == clip frame 0)

const MAX_VERTICES: usize = 1 << 17;
const MAX_BONES: usize = 256;
const MAX_FRAMES: usize = 8192;
const MAX_ANIMATIONS: usize = 64;

#[derive(Clone, Copy)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub u: f32,
    pub v: f32,
    pub bones: [i32; 4],
    pub weights: [f32; 4],
}

pub struct Bone {
    pub parent: i32,
}

/// One sampled bone pose: local translation, rotation about Z (radians) and
/// axis scale. Scale carries the blinks — an eye clip squashes `scale_y` to
/// ~0 and leaves translation and rotation almost untouched.
#[derive(Clone, Copy)]
pub struct Pose {
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
    pub scale_x: f32,
    pub scale_y: f32,
}

pub const POSE_REST: Pose = Pose { x: 0.0, y: 0.0, rotation: 0.0, scale_x: 1.0, scale_y: 1.0 };

pub struct Animation {
    pub id: u32,
    pub mirrors: bool,
    pub fps: f32,
    /// `frames[frame][bone]`.
    pub frames: Vec<Vec<Pose>>,
}

pub struct Puppet {
    pub vertices: Vec<Vertex>,
    /// Flat triangle list into `vertices`.
    pub triangles: Vec<u32>,
    pub bones: Vec<Bone>,
    pub animations: Vec<Animation>,
}

/// A NUL-terminated string, as the `.mdl` format uses for the material path
/// and clip names.
fn read_cstring(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let mut bytes = Vec::new();
    loop {
        let byte = cursor.read_u8()?;
        if byte == 0 {
            break;
        }
        bytes.push(byte);
        if bytes.len() > 4096 {
            bail!("unterminated string in puppet model");
        }
    }
    Ok(String::from_utf8(bytes)?)
}

fn skip(cursor: &mut Cursor<&[u8]>, count: i64) -> Result<()> {
    cursor.seek(SeekFrom::Current(count))?;
    Ok(())
}

/// Parse a puppet: an `MDLV` mesh block, an `MDLS` skeleton, and an optional
/// `MDLA` animation block.
///
/// Two container versions are in the corpus and they differ in exactly one
/// thing that matters — `MDLV0023` carries lit vertices (see `VERTEX_SIZE_LIT`)
/// and declares the format again just before the buffer. Its `MDLS0004`
/// skeleton and `MDLA0006` animation blocks are laid out identically to
/// `MDLS0001`/`MDLA0001`; what looked like differences were two fields this
/// parser had been reading as fixed one-byte skips and which are really
/// strings — a bone's name (`眼球` in one corpus model) and a bone's editor
/// metadata (a JSON blob in `MDLS0004`, empty in `MDLS0001`).
pub fn parse(bytes: &[u8]) -> Result<Puppet> {
    let mut cursor = Cursor::new(bytes);

    let mut magic = [0u8; 8];
    cursor.read_exact(&mut magic)?;
    let lit = match &magic {
        b"MDLV0013" => false,
        b"MDLV0023" => true,
        other => bail!("not a puppet model ({})", String::from_utf8_lossy(other)),
    };
    read_cstring(&mut cursor)?; // model name, empty throughout the corpus
    cursor.read_u32::<LittleEndian>()?;
    cursor.read_u32::<LittleEndian>()?;
    cursor.read_u32::<LittleEndian>()?;
    read_cstring(&mut cursor)?; // material path, resolved via the model JSON instead

    // The newer container repeats the vertex format immediately before the
    // buffer, behind 28 bytes that are zero in every corpus model. Only the
    // low half of the word carries the attribute mask: one model writes it as
    // `0x0000000f` where the rest write `0x0180000f`.
    let vertex_size = if lit {
        skip(&mut cursor, LIT_VERTEX_PREFIX)?;
        match cursor.read_u32::<LittleEndian>()? & 0xffff {
            15 => VERTEX_SIZE_LIT,
            9 => VERTEX_SIZE,
            other => bail!("unknown puppet vertex format ({other})"),
        }
    } else {
        cursor.read_u32::<LittleEndian>()?;
        VERTEX_SIZE
    };

    let vertex_bytes = cursor.read_u32::<LittleEndian>()? as usize;
    if !vertex_bytes.is_multiple_of(vertex_size) || vertex_bytes / vertex_size > MAX_VERTICES {
        bail!("implausible puppet vertex block ({vertex_bytes} bytes)");
    }
    let mut vertices = Vec::with_capacity(vertex_bytes / vertex_size);
    for _ in 0..vertex_bytes / vertex_size {
        let x = cursor.read_f32::<LittleEndian>()?;
        let y = cursor.read_f32::<LittleEndian>()?;
        cursor.read_f32::<LittleEndian>()?; // z, always 0 for these 2D puppets
        if vertex_size == VERTEX_SIZE_LIT {
            skip(&mut cursor, LIT_VERTEX_PREFIX)?; // normal, tangent, handedness
        }
        let mut bones = [0i32; 4];
        let mut weights = [0f32; 4];
        for bone in &mut bones {
            *bone = cursor.read_i32::<LittleEndian>()?;
        }
        for weight in &mut weights {
            *weight = cursor.read_f32::<LittleEndian>()?;
        }
        let u = cursor.read_f32::<LittleEndian>()?;
        let v = cursor.read_f32::<LittleEndian>()?;
        vertices.push(Vertex { x, y, u, v, bones, weights });
    }

    let index_bytes = cursor.read_u32::<LittleEndian>()? as usize;
    if !index_bytes.is_multiple_of(2) {
        bail!("odd puppet index block ({index_bytes} bytes)");
    }
    let mut triangles = Vec::with_capacity(index_bytes / 2);
    for _ in 0..index_bytes / 2 {
        let index = u32::from(cursor.read_u16::<LittleEndian>()?);
        if index as usize >= vertices.len() {
            bail!("puppet triangle index {index} out of range");
        }
        triangles.push(index);
    }

    // `MDLV0023` follows its index block with a per-bone trailer (sixteen bytes
    // each behind a ten-byte header) that `MDLV0013` does not have, so the
    // skeleton is found rather than assumed to come next — in the older
    // container the search matches at offset zero.
    let after_indices = usize::try_from(cursor.position()).unwrap_or(usize::MAX).min(bytes.len());
    let Some(at) = find(&bytes[after_indices..], b"MDLS") else {
        bail!("puppet model has no MDLS skeleton block");
    };
    cursor.seek(SeekFrom::Start((after_indices + at + 8) as u64))?;
    read_cstring(&mut cursor)?; // skeleton name, empty throughout the corpus
    cursor.read_u32::<LittleEndian>()?;
    let bone_count = cursor.read_u32::<LittleEndian>()? as usize;
    if bone_count == 0 || bone_count > MAX_BONES {
        bail!("implausible puppet bone count ({bone_count})");
    }
    let mut bones = Vec::with_capacity(bone_count);
    for _ in 0..bone_count {
        read_cstring(&mut cursor)?; // bone name
        cursor.read_u32::<LittleEndian>()?;
        let parent = cursor.read_i32::<LittleEndian>()?;
        let matrix_bytes = cursor.read_u32::<LittleEndian>()? as usize;
        if matrix_bytes != BONE_MATRIX_SIZE {
            bail!("unexpected puppet bone matrix size ({matrix_bytes} bytes)");
        }
        skip(&mut cursor, 64)?; // BONE_MATRIX_SIZE
        read_cstring(&mut cursor)?; // editor metadata: JSON in MDLS0004, empty in MDLS0001
        bones.push(Bone { parent });
    }

    // `MDLS0004` follows its bones with an editor rig section — handle
    // positions, colours, a per-bone record we have no use for — so the
    // animation block is found by looking for it rather than by assuming it
    // comes next. In `MDLS0001` there is nothing in between and the search
    // matches at offset zero.
    let consumed = usize::try_from(cursor.position()).unwrap_or(usize::MAX).min(bytes.len());
    let animations = match find(&bytes[consumed..], b"MDLA") {
        Some(at) => {
            cursor.seek(SeekFrom::Start((consumed + at) as u64))?;
            parse_animations(&mut cursor, bone_count)?
        }
        None => Vec::new(),
    };

    Ok(Puppet { vertices, triangles, bones, animations })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn parse_animations(cursor: &mut Cursor<&[u8]>, bone_count: usize) -> Result<Vec<Animation>> {
    // The two block versions lay a record out identically and differ only in
    // the run of bytes separating one record from the next: four in
    // `MDLA0001`, thirty-five in `MDLA0006`. Both are measured off the
    // multi-clip models — `jiemao` for the older, `人物`'s three clips for the
    // newer — and a single-clip model cannot distinguish them, since what
    // follows the last record is the end of the file either way.
    let mut magic = [0u8; 8];
    cursor.read_exact(&mut magic)?;
    let trailer: i64 = if &magic == b"MDLA0001" { 4 } else { 35 };
    read_cstring(cursor)?; // block name, empty throughout the corpus
    cursor.read_u32::<LittleEndian>()?;
    let animation_count = cursor.read_u32::<LittleEndian>()? as usize;
    if animation_count > MAX_ANIMATIONS {
        bail!("implausible puppet animation count ({animation_count})");
    }

    let mut animations = Vec::with_capacity(animation_count);
    for _ in 0..animation_count {
        let id = cursor.read_u32::<LittleEndian>()?;
        cursor.read_u32::<LittleEndian>()?;
        read_cstring(cursor)?; // clip name, unused
        let mode = read_cstring(cursor)?;
        let fps = cursor.read_f32::<LittleEndian>()?;
        cursor.read_u32::<LittleEndian>()?; // declared frame count; the tracks carry frame count + 1
        cursor.read_u32::<LittleEndian>()?;
        let track_count = cursor.read_u32::<LittleEndian>()? as usize;
        if track_count != bone_count {
            bail!("puppet animation has {track_count} tracks for {bone_count} bones");
        }

        // tracks[bone][frame]
        let mut tracks: Vec<Vec<Pose>> = Vec::with_capacity(track_count);
        for _ in 0..track_count {
            cursor.read_u32::<LittleEndian>()?;
            let track_bytes = cursor.read_u32::<LittleEndian>()? as usize;
            if !track_bytes.is_multiple_of(FRAME_SIZE) || track_bytes / FRAME_SIZE > MAX_FRAMES {
                bail!("implausible puppet track ({track_bytes} bytes)");
            }
            let mut poses = Vec::with_capacity(track_bytes / FRAME_SIZE);
            for _ in 0..track_bytes / FRAME_SIZE {
                let x = cursor.read_f32::<LittleEndian>()?;
                let y = cursor.read_f32::<LittleEndian>()?;
                skip(cursor, 4)?; // translation z
                skip(cursor, 8)?; // euler x, y
                let rotation = cursor.read_f32::<LittleEndian>()?; // euler z
                let scale_x = cursor.read_f32::<LittleEndian>()?;
                let scale_y = cursor.read_f32::<LittleEndian>()?;
                skip(cursor, 4)?; // scale z
                poses.push(Pose { x, y, rotation, scale_x, scale_y });
            }
            tracks.push(poses);
        }
        skip(cursor, trailer)?;

        let frame_count = tracks.iter().map(Vec::len).min().unwrap_or(0);
        if frame_count == 0 {
            continue;
        }
        let frames = (0..frame_count)
            .map(|frame| tracks.iter().map(|track| track[frame]).collect())
            .collect();

        animations.push(Animation {
            id,
            mirrors: mode.eq_ignore_ascii_case("mirror"),
            fps: if fps > 0.0 { fps } else { 30.0 },
            frames,
        });
    }
    Ok(animations)
}

/// A 2D affine map: `(x, y) -> (a·x + c·y + tx, b·x + d·y + ty)`.
#[derive(Clone, Copy)]
pub struct Affine {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tx: f32,
    pub ty: f32,
}

pub const AFFINE_IDENTITY: Affine = Affine { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: 0.0, ty: 0.0 };

/// `translate · rotate · scale`, the order Wallpaper Engine composes a bone's
/// local transform in.
fn affine_trs(pose: Pose) -> Affine {
    let (sin, cos) = pose.rotation.sin_cos();
    Affine {
        a: cos * pose.scale_x,
        b: sin * pose.scale_x,
        c: -sin * pose.scale_y,
        d: cos * pose.scale_y,
        tx: pose.x,
        ty: pose.y,
    }
}

fn affine_mul(m: Affine, n: Affine) -> Affine {
    Affine {
        a: m.a * n.a + m.c * n.b,
        b: m.b * n.a + m.d * n.b,
        c: m.a * n.c + m.c * n.d,
        d: m.b * n.c + m.d * n.d,
        tx: m.a * n.tx + m.c * n.ty + m.tx,
        ty: m.b * n.tx + m.d * n.ty + m.ty,
    }
}

fn affine_inv(m: Affine) -> Affine {
    let determinant = m.a * m.d - m.b * m.c;
    if determinant.abs() < 1e-12 {
        return AFFINE_IDENTITY;
    }
    let inv = 1.0 / determinant;
    Affine {
        a: m.d * inv,
        b: -m.b * inv,
        c: -m.c * inv,
        d: m.a * inv,
        tx: (m.c * m.ty - m.d * m.tx) * inv,
        ty: (m.b * m.tx - m.a * m.ty) * inv,
    }
}

fn affine_apply(m: Affine, x: f32, y: f32) -> (f32, f32) {
    (m.a * x + m.c * y + m.tx, m.b * x + m.d * y + m.ty)
}

/// World transform per bone, following parents (`parent` is always earlier in
/// the list, or `-1` for a root).
fn world_transforms(bones: &[Bone], poses: &[Pose]) -> PerBone<Affine> {
    let mut world: PerBone<Affine> = smallvec::smallvec![AFFINE_IDENTITY; bones.len()];
    for (index, bone) in bones.iter().enumerate() {
        let pose = poses.get(index).copied().unwrap_or(POSE_REST);
        let local = affine_trs(pose);
        world[index] = if bone.parent >= 0 && (bone.parent as usize) < index {
            affine_mul(world[bone.parent as usize], local)
        } else {
            local
        };
    }
    world
}

/// Sample every bone's pose at `time` seconds, looping (or ping-ponging, for
/// `mirror` clips) and linearly interpolating between stored frames.
fn sample_poses(animation: &Animation, time: f32, rate: f32) -> PerBone<Pose> {
    let Some(first) = animation.frames.first() else {
        return PerBone::new();
    };
    if animation.frames.len() == 1 {
        return first.iter().copied().collect();
    }
    let span = (animation.frames.len() - 1) as f32;
    let mut position = time * animation.fps.max(0.01) * rate;
    if animation.mirrors {
        let cycle = positive_rem(position, span * 2.0);
        position = if cycle <= span { cycle } else { span * 2.0 - cycle };
    } else {
        position = positive_rem(position, span);
    }
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "position is non-negative and < frame count")]
    let lower = (position as usize).min(animation.frames.len() - 2);
    let fraction = position - lower as f32;
    animation.frames[lower]
        .iter()
        .zip(&animation.frames[lower + 1])
        .map(|(start, end)| Pose {
            x: start.x + (end.x - start.x) * fraction,
            y: start.y + (end.y - start.y) * fraction,
            rotation: start.rotation + (end.rotation - start.rotation) * fraction,
            scale_x: start.scale_x + (end.scale_x - start.scale_x) * fraction,
            scale_y: start.scale_y + (end.scale_y - start.scale_y) * fraction,
        })
        .collect()
}

fn positive_rem(value: f32, divisor: f32) -> f32 {
    let remainder = value % divisor;
    if remainder >= 0.0 { remainder } else { remainder + divisor }
}

/// The bone this animation belongs to, by id — falling back to the first clip.
fn pick_animation(puppet: &Puppet, animation_id: Option<u32>) -> Option<&Animation> {
    match animation_id {
        Some(id) => puppet.animations.iter().find(|a| a.id == id).or_else(|| puppet.animations.first()),
        None => puppet.animations.first(),
    }
}

/// One skin transform per bone: maps a bind-pose vertex into its animated
/// position. Built against clip frame 0, which is the bind pose, so `time = 0`
/// is an exact identity.
pub fn skin_transforms(puppet: &Puppet, animation_id: Option<u32>, time: f32, rate: f32) -> Vec<Affine> {
    let Some(animation) = pick_animation(puppet, animation_id) else {
        return vec![AFFINE_IDENTITY; puppet.bones.len()];
    };
    let Some(bind) = animation.frames.first() else {
        return vec![AFFINE_IDENTITY; puppet.bones.len()];
    };
    let bind_world = world_transforms(&puppet.bones, bind);
    let animated_world = world_transforms(&puppet.bones, &sample_poses(animation, time, rate));
    animated_world
        .into_iter()
        .zip(bind_world)
        .map(|(animated, bind)| affine_mul(animated, affine_inv(bind)))
        .collect()
}

/// Deform every vertex by its weighted bone skins.
pub fn deform(puppet: &Puppet, skins: &[Affine]) -> Vec<(f32, f32)> {
    puppet
        .vertices
        .iter()
        .map(|vertex| {
            let (mut x, mut y, mut total) = (0.0, 0.0, 0.0);
            for slot in 0..4 {
                let weight = vertex.weights[slot];
                if weight <= 1e-4 {
                    continue;
                }
                let bone = vertex.bones[slot];
                if bone < 0 || bone as usize >= skins.len() {
                    continue;
                }
                let (px, py) = affine_apply(skins[bone as usize], vertex.x, vertex.y);
                x += px * weight;
                y += py * weight;
                total += weight;
            }
            if total > 1e-4 { (x / total, y / total) } else { (vertex.x, vertex.y) }
        })
        .collect()
}

/// Least-squares `value ≈ slope·coord + intercept` over paired samples.
fn linear_fit(coords: impl Iterator<Item = (f32, f32)> + Clone) -> (f32, f32) {
    let n = coords.clone().count() as f32;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (x, y) in coords {
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denominator = n * sxx - sx * sx;
    if denominator.abs() < 1e-9 {
        return (0.0, sy / n);
    }
    let slope = (n * sxy - sx * sy) / denominator;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

/// The mesh's own linear map from vertex space to UV space, `(u, v)` as
/// functions of `(x, y)`. Recovering it lets a deformed vertex be placed in
/// the layer's texture rectangle exactly the way a flat layer is, so a puppet
/// lands where the un-warped image would and the animation only perturbs it.
pub fn uv_fit(puppet: &Puppet) -> (f32, f32, f32, f32) {
    let (u_slope, u_intercept) = linear_fit(puppet.vertices.iter().map(|vertex| (vertex.x, vertex.u)));
    let (v_slope, v_intercept) = linear_fit(puppet.vertices.iter().map(|vertex| (vertex.y, vertex.v)));
    (u_slope, u_intercept, v_slope, v_intercept)
}

/// Rasterize the deformed mesh into an `out_w × out_h` RGBA image, sampling
/// `texture` bilinearly through each triangle's UVs and compositing triangles
/// front-to-back with straight "over" alpha. `place` maps a deformed vertex
/// `(x, y)` to a pixel in the output.
pub fn rasterize(
    puppet: &Puppet,
    positions: &[(f32, f32)],
    texture: &RgbaImage,
    out_w: u32,
    out_h: u32,
    place: impl Fn(f32, f32) -> (f32, f32),
) -> RgbaImage {
    let mut output = RgbaImage::new(out_w, out_h);
    let (tw, th) = (texture.width(), texture.height());

    for triangle in 0..puppet.triangles.len() / 3 {
        let base = triangle * 3;
        let [i0, i1, i2] = [
            puppet.triangles[base] as usize,
            puppet.triangles[base + 1] as usize,
            puppet.triangles[base + 2] as usize,
        ];
        let (v0, v1, v2) = (puppet.vertices[i0], puppet.vertices[i1], puppet.vertices[i2]);
        let p0 = place(positions[i0].0, positions[i0].1);
        let p1 = place(positions[i1].0, positions[i1].1);
        let p2 = place(positions[i2].0, positions[i2].1);

        let area = (p1.0 - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (p1.1 - p0.1);
        if area.abs() < 1e-6 {
            continue;
        }
        let inv_area = 1.0 / area;

        let min_x = p0.0.min(p1.0).min(p2.0).floor().max(0.0) as u32;
        let max_x = ((p0.0.max(p1.0).max(p2.0).ceil()) as i64).clamp(0, i64::from(out_w) - 1) as u32;
        let min_y = p0.1.min(p1.1).min(p2.1).floor().max(0.0) as u32;
        let max_y = ((p0.1.max(p1.1).max(p2.1).ceil()) as i64).clamp(0, i64::from(out_h) - 1) as u32;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let (sx, sy) = (px as f32 + 0.5, py as f32 + 0.5);
                let w0 = ((p1.0 - sx) * (p2.1 - sy) - (p2.0 - sx) * (p1.1 - sy)) * inv_area;
                let w1 = ((p2.0 - sx) * (p0.1 - sy) - (p0.0 - sx) * (p2.1 - sy)) * inv_area;
                let w2 = 1.0 - w0 - w1;
                if w0 < -1e-3 || w1 < -1e-3 || w2 < -1e-3 {
                    continue;
                }
                let u = w0 * v0.u + w1 * v1.u + w2 * v2.u;
                let v = w0 * v0.v + w1 * v1.v + w2 * v2.v;
                let sample = sample_bilinear(texture, tw, th, u, v);
                blend_over(output.get_pixel_mut(px, py), sample);
            }
        }
    }
    output
}

fn sample_bilinear(texture: &RgbaImage, tw: u32, th: u32, u: f32, v: f32) -> [f32; 4] {
    let fx = u * tw as f32 - 0.5;
    let fy = v * th as f32 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let (tx, ty) = (fx - x0, fy - y0);

    let at = |ix: f32, iy: f32| -> [f32; 4] {
        let cx = (ix as i64).clamp(0, i64::from(tw) - 1) as u32;
        let cy = (iy as i64).clamp(0, i64::from(th) - 1) as u32;
        texture.get_pixel(cx, cy).0.map(f32::from)
    };
    let p00 = at(x0, y0);
    let p10 = at(x0 + 1.0, y0);
    let p01 = at(x0, y0 + 1.0);
    let p11 = at(x0 + 1.0, y0 + 1.0);

    let mut out = [0.0; 4];
    for (channel, slot) in out.iter_mut().enumerate() {
        let top = p00[channel] * (1.0 - tx) + p10[channel] * tx;
        let bottom = p01[channel] * (1.0 - tx) + p11[channel] * tx;
        *slot = top * (1.0 - ty) + bottom * ty;
    }
    out
}

fn blend_over(dst: &mut Rgba<u8>, src: [f32; 4]) {
    let sa = src[3] / 255.0;
    if sa <= 0.0 {
        return;
    }
    let da = f32::from(dst.0[3]) / 255.0;
    let out_a = sa + da * (1.0 - sa);
    let byte = |value: f32| value.round().clamp(0.0, 255.0) as u8;
    for (channel, slot) in dst.0[..3].iter_mut().enumerate() {
        *slot = byte(src[channel] * sa + f32::from(*slot) * (1.0 - sa));
    }
    dst.0[3] = byte(out_a * 255.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_inverse_round_trips() {
        let m = affine_mul(
            affine_trs(Pose { x: 12.0, y: -4.0, rotation: 0.7, ..POSE_REST }),
            Affine { a: 1.3, b: 0.0, c: 0.0, d: 0.8, tx: 0.0, ty: 0.0 },
        );
        let round = affine_mul(m, affine_inv(m));
        assert!((round.a - 1.0).abs() < 1e-4 && round.b.abs() < 1e-4);
        assert!((round.d - 1.0).abs() < 1e-4 && round.c.abs() < 1e-4);
        assert!(round.tx.abs() < 1e-3 && round.ty.abs() < 1e-3);
    }

    #[test]
    fn bind_pose_is_identity_skinning() {
        // Two bones, one clip whose first frame is the bind pose: at t=0 every
        // skin transform must be identity.
        let puppet = Puppet {
            vertices: vec![],
            triangles: vec![],
            bones: vec![Bone { parent: -1 }, Bone { parent: 0 }],
            animations: vec![Animation {
                id: 1,
                mirrors: false,
                fps: 30.0,
                frames: vec![
                    vec![
                        Pose { x: 5.0, y: 2.0, rotation: 0.1, ..POSE_REST },
                        Pose { x: 3.0, y: 0.0, rotation: -0.2, ..POSE_REST },
                    ],
                    vec![
                        Pose { x: 9.0, y: 7.0, rotation: 0.4, scale_x: 0.5, scale_y: 0.2 },
                        Pose { x: 1.0, y: 5.0, rotation: 0.3, scale_x: 1.4, scale_y: 0.0 },
                    ],
                ],
            }],
        };
        for skin in skin_transforms(&puppet, Some(1), 0.0, 1.0) {
            assert!((skin.a - 1.0).abs() < 1e-4 && (skin.d - 1.0).abs() < 1e-4);
            assert!(skin.b.abs() < 1e-4 && skin.c.abs() < 1e-4);
            assert!(skin.tx.abs() < 1e-3 && skin.ty.abs() < 1e-3);
        }
    }

    #[test]
    fn a_scale_only_clip_still_squashes_the_mesh() {
        // How an eye blinks: translation and rotation hold still and `scale_y`
        // collapses. Dropping the scale channel makes such a clip a no-op.
        let puppet = Puppet {
            vertices: vec![Vertex { x: 0.0, y: 10.0, u: 0.0, v: 0.0, bones: [0, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0] }],
            triangles: vec![],
            bones: vec![Bone { parent: -1 }],
            animations: vec![Animation {
                id: 1,
                mirrors: false,
                fps: 1.0,
                frames: vec![vec![POSE_REST], vec![Pose { scale_y: 0.0, ..POSE_REST }]],
            }],
        };
        // Half way to the flattened frame, so the sample does not wrap round.
        let squashed = deform(&puppet, &skin_transforms(&puppet, Some(1), 0.5, 1.0));
        assert!((squashed[0].1 - 5.0).abs() < 1e-3, "scale_y should halve y, got {}", squashed[0].1);
    }

    #[test]
    fn linear_fit_recovers_a_line() {
        let (slope, intercept) = linear_fit((0..10).map(|i| (i as f32, 3.0 * i as f32 - 7.0)));
        assert!((slope - 3.0).abs() < 1e-3);
        assert!((intercept + 7.0).abs() < 1e-3);
    }
}

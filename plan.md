# Wallpaper Engine → live desktop wallpaper app

**Goal.** Play a Wallpaper Engine wallpaper directly as the desktop
background — a live GL/video loop pinned behind the desktop icons, with a
Tauri front end to pick and tweak wallpapers, the way Wallpaper Engine itself
works (§14). Still-PNG export and the container/shader work under it remain
useful building blocks (thumbnails, debugging, Video-type passthrough) but are
no longer the deliverable; exporting a finished mp4 for some *other* wallpaper
app to play is superseded by that goal and dropped from the roadmap (§7, §8).

**Status.** Container layer, type routing and the Video pipeline are built and
verified against a six-wallpaper corpus. The Scene renderer now runs the real
GPU effect chain for the common one-image-plus-effects shape (§4); richer
scenes (multiple image layers, particles) still fall back to the effect-free
composite (§4.7). A `simulate` subcommand plays that same chain live in a real
window at wall-clock speed — the base the live desktop backend (§14) builds
on. Web capture and the desktop backend itself are still to build.

---

## 1. What we already have

| Component | State |
|---|---|
| `.pkg` archive parsing + extraction | Done, verified byte-exact |
| `.tex` texture decode (RGBA8888 / RG88 / R8 / DXT1/3/5, LZ4, embedded PNG passthrough) | Done, verified pixel-identical against an independent implementation |
| CLI skeleton (`clap`), recursive discovery (`walkdir`) | Done |
| `project.json` parsing and type routing | Done, all six sample wallpapers route correctly |
| Video pipeline (still PNG + looping mp4) | Done, both sample video wallpapers export |
| Scene model (`scene.json` / model / material) | Done |
| Scene base composite (layers flattened to a still) | Done, both sample scenes match their previews |
| Shader preprocessor + shim (`#include`, combos, dialect rewrite) | Done, 28/28 corpus shaders compile |
| Headless GL renderer + effect chain | Done for one image layer + effects (§4); multi-layer/particle scenes fall back to the composite |

Crates in use: `anyhow`, `clap`, `ffmpeg-next`, `glow`, `glsl-include`,
`glutin`, `image`, `lz4_flex`, `objc2`/`objc2-app-kit`/`objc2-foundation`
(headless GL context on macOS), `png`, `raw-window-handle`, `regex`, `serde`,
`serde_json`, `texpresso`, `walkdir`.

---

## 2. Wallpaper taxonomy and routing

`project.json` carries a `type` field. It is the first thing the tool reads,
and it selects the entire downstream pipeline.

**Matching is case-insensitive.** Wallpaper Engine does not normalize what it
writes: the corpus contains both `"Scene"` and `"scene"`. Exact matching would
silently route half of it to the unknown branch.

| Type | Pipeline | Effort | Notes |
|---|---|---|---|
| `Video` | Passthrough — the mp4/webm is already inside the `.pkg` | Trivial | Extract, optionally transcode/resize. Biggest coverage-per-effort win. |
| `Scene` | Full shader renderer | Large | The core project. What our sample wallpaper is. |
| `Web` | Headless Chrome capture | Medium | Separate pipeline, shares the encode/loop stage. |
| `Application` | Unsupported | — | A Windows `.exe`. Detect and fail with a clear message. |

**Design consequence:** the type router is the top-level seam. Each pipeline
produces the same intermediate — a sequence of RGBA frames plus a frame rate —
and what consumes that (a still PNG, or the live desktop window, §14) is
shared by all three. Build the shared frame-source stage once.

---

## 3. Target module layout

```
src/
  main.rs            CLI, type routing
  reader.rs          (done) counting little-endian reader
  pkg.rs             (done) archive parse + extract
  tex.rs             (done) texture decode
  project.rs         project.json: type detection, title, metadata

  scene/
    model.rs         serde types for scene.json / effect.json / material.json
    resolve.rs       asset path resolution, dependency graph

  shader/
    preprocess.rs    #include resolution, COMBO defines, dialect rewrite
    shim.rs          our replacement for common*.h
    translate.rs     WE GLSL -> target dialect

  render/
    gpu.rs           wgpu device/queue setup, offscreen targets
    pass.rs          one effect pass (shader + textures + uniforms)
    chain.rs         the post-process chain over a base image
    capture.rs       GPU -> CPU frame readback

  web/
    server.rs        local static server for the unpacked wallpaper
    capture.rs       headless Chrome driving + frame capture

  export/
    still.rs         single-frame PNG
    loops.rs         loop period detection
    video.rs         ffmpeg encode
```

---

## 4. The Scene renderer

### 4.0 Scene geometry (settled while building the composite)

Two conventions had to be pinned down before anything could be placed, and
both are easy to get backwards:

- **Scene Y points up from the bottom edge.** Image rows go down, so the
  transform is `row = ortho.height - world_y`. Confirmed against the sample by
  where it puts the eyelash and hair layers of a portrait — the wrong sign puts
  them at the character's feet.
- **`camera.eye` is the editor's saved viewport, not the render view.** The
  visible rectangle is simply `(0,0)..(ortho.width, ortho.height)`. Both samples
  place their background layer at exactly the canvas centre with exactly the
  canvas extent, which only lines up under this reading; honouring `camera.eye`
  slides one of them half a screen off-centre.

An object's `origin` is its centre, `size` is its extent in scene units before
`scale`, and layer order is array order. Objects are distinguished by which of
`image`, `particle` and `sound` is present — there is no type tag.

### 4.1 Render model (established from the sample)

Scene wallpapers of the common kind are **one fullscreen image with a chain of
post-process passes**. No geometry, no 3D.

```
base image texture
   ↓
 effect[0] pass[0]  ── samples previous result as g_Texture0
   ↓                    plus its own mask textures as g_Texture1..3
 effect[1] pass[0]
   ↓
 effect[2] pass[0]
   ↓
final frame
```

Each pass ping-pongs between two offscreen render targets.

### 4.2 Parameter binding is data-driven

Every uniform in a WE shader carries a JSON comment naming its material key:

```
uniform float g_Amp;      // {"material":"strength","default":0.1,"range":[0.01,0.5]}
uniform vec2  g_Friction; // {"material":"friction","default":"1 1"}
```

`scene.json` supplies the values under `passes[].constantshadervalues`:

```
"constantshadervalues": { "strength": 0.1, "friction": "1 1", "speed": 1.0 }
```

So the binding table is **parsed out of the shader source itself**. No
per-effect special-casing, and defaults come from the same annotations when
`scene.json` omits a key. This is the single most important property of the
format for us — it means new effects mostly work without new code.

### 4.3 Textures per pass

`passes[].textures` is a positional array mapping to `g_Texture1`, `g_Texture2`,
… with `null` meaning "use the shader annotation's `default`" (e.g.
`util/noflow`, `util/black`). Those defaults are WE built-in utility textures we
will need to synthesise ourselves — they are small, constant, and describable
from their names (a flat 0.5/0.5 flow map, an all-black opacity mask, etc.).
`g_Texture0` is always the previous pass result.

### 4.4 Animation

All motion is a pure function of the `g_Time` uniform. There is no simulation
state. Frames are therefore **independently addressable** — render frame `n` at
`t = n / fps` — which makes the renderer stateless, trivially parallel, and
exactly reproducible. This is a large simplification and we should preserve it.
Confirmed in practice: re-rendering the same timestamp twice is byte-identical.

### 4.5 What building the chain actually settled

- **Vertex stage: option (b), always.** Every pass draws one fixed clip-space
  quad (`render/pass.rs::build_quad`) with `g_ModelViewProjectionMatrix` set to
  the identity, rather than running Wallpaper Engine's own vertex shader. It
  still expects that vertex shader's declared attributes/uniforms
  (`a_Position`, `a_TexCoord`, `g_TextureNResolution`) to exist and be fed, so
  its varyings come out identical to running the real thing — cheaper than
  option (a) without losing correctness, for the fullscreen case.
- **`g_TextureNResolution` is `(width, height, width, height)`.** Two shaders
  read it two different ways — `shake.vert` takes the ratio `.z/.x` (an
  atlas-packing remap, which is 1 when nothing is atlas-packed, as here) and
  `foliagesway.vert` takes `.z/.w` directly (the texture's real aspect ratio).
  Both are satisfied at once by setting all four components to the texture's
  actual pixel size.
- **A texture slot's combo switches on exactly when the slot is bound.** A
  sampler's own annotation can name a combo (`{"combo":"MASK"}`); it must read
  `1` exactly when `scene.json`'s positional `textures[]` actually names
  something for that slot, `0` otherwise — not from any value in `scene.json`
  itself, which never mentions combo names. (`shader::bind::texture_combos`.)
- **`textures[i]` binds `g_Texture{i+1}`; `g_Texture0` is always the running
  chain result**, never a positional entry — confirmed against
  `scene_example1`'s three effects, none of which puts a texture at index 0.
- **`util/noise` has no synthesized default yet**, unlike `util/noflow` and
  `util/black`; `foliagesway`'s noise-driven wobble falls back to flat black
  until one exists. Documented gap, not a silent one.
- **Compiling a chain is separate from redrawing it.** `scene::render::EffectChain`
  splits shader compilation, texture uploads and `scene.json` parsing (done
  once, in `prepare_effect_chain`) from the actual draw calls (`EffectChain::render`,
  called with a new `g_Time` every frame). One-shot export and the live
  simulator both build on this; only the simulator calls `render` more than
  once per chain.
- **Two different fullscreen quads, not one.** `build_quad`'s V mapping
  preserves texel row order (a pass with no displacement leaves the image
  alone), which is what lets any number of chained passes still agree with
  `read_rgba`'s row order with no flip needed there. `build_display_quad` is
  the one exception: going from that same row-preserving texture to an
  actual on-screen window needs exactly one flip, since GL always puts NDC's
  top edge at the window's top row — `blit_to_screen` uses this one, never
  `build_quad`. Mixing them up is exactly how the simulator first shipped
  upside down: the export path's `read_rgba` was quietly absorbing a flip
  that every pass introduced, and the direct-to-screen blit had nothing to
  absorb it. See `render/capture.rs`'s two `#[ignore]`d tests for the
  regression case (they need the AppKit main thread, which `cargo test`
  can't provide without a lib target — verified manually instead, by
  temporarily inlining them into `main()` and running via `cargo run`).

### 4.6 Live simulator

`wallpaper-engine simulate <wallpaper>` opens a real window (`winit` +
`glutin-winit`, GL 3.3 core same as the headless exporter) and redraws the
compiled chain every frame with `g_Time = start.elapsed()`, blitting the
result straight to the window instead of reading it back to the CPU. Only
`g_Time` is live; every other input Wallpaper Engine can feed a shader —
pointer position, system audio spectrum, time-of-day sync, now-playing
media — is not wired up (deliberately, for now: none of it is exercised by
this scene, and system audio loopback in particular is nontrivial on macOS,
needing a virtual device or a `ScreenCaptureKit` audio tap). A scene without
the one-image-plus-effects shape just shows the static composite in the
window rather than refusing to open one.

Resizing the window doesn't stretch the image to the new shape: `State`
records the scene's own pixel dimensions once at open time (every render
target matches it, chain or no chain), and `pass::blit_to_screen` letterboxes
— clears the whole window to black, then draws into a centered sub-viewport
scaled to fit while preserving that aspect ratio (`pass::letterbox`, unit
tested directly since it's pure arithmetic, no GL context needed).

An `egui` (`egui_glow`, winit-integrated) overlay draws an "Effect
Parameters" panel on top with one slider per range-annotated scalar `float`
uniform any pass declares (`scene::render::Tweakable`, `collect_tweakables`)
— e.g. `foliagesway`'s `g_Strength`, the wave-distortion knob that prompted
this. Each slider starts at the wallpaper's own preset (`scene.json`'s
`constantshadervalues`, not the shader's generic default — `scene_example1`
ships `strength: 0.65` for the sway and `0.1` for the shake, both above their
shaders' own `0.4`/no-op defaults) and its declared `range`; dragging one
feeds `EffectChain::render`'s `overrides` slice, which only overwrites that
one named uniform in the one pass it belongs to, leaving everything else —
including `export`, which always calls `render` with `overrides: &[]` — on
the wallpaper's own values. Discovery is fully generic (no per-effect
hardcoding): any current or future effect whose shader carries a `range`
annotation gets a slider for free.
`glow` is pinned to 0.17 rather than 0.18 specifically because that's what
`egui_glow` depends on and 0.x crates don't unify across minor versions —
confirmed a clean drop-in with no code changes elsewhere.

### 4.7 Multi-layer scenes: what's still missing

`papers/scene_example2` is the concrete case (§13's "scope gap" scene): 12
objects — 9 image layers each with their own effect list (one carrying 13),
3 with keyframe animation, and 2 particle systems. Today's renderer shows the
static composite with every one of those listed as "not simulated." Four
separate gaps, not one:

- **Per-layer effect chains.** `effect_chain_shape` only fires for *exactly
  one* visible image layer, and even then `render_frame` flattens the whole
  scene to one canvas first and runs a single chain over that flattened
  image. Correct rendering is the other way round, per layer: render each
  image layer to its own texture, run *that layer's own* effect chain on it,
  then alpha-composite the processed layers together in order. `EffectChain`
  itself barely changes; the orchestration around it (currently
  `compose::render` once, then one chain) does.
- **Keyframe animation.** `Object::animationlayers` deserializes as opaque
  `Vec<Value>` today and is only checked for non-emptiness (to print the
  omission). Needs real types for keyframe tracks — which property (origin,
  scale, angle, alpha), timestamped values, an interpolation mode — evaluated
  at the render timestamp and fed into that layer's transform.
- **Particle systems.** `Object::particle` names a preset path that is never
  read. A new subsystem: parse the preset (emitter shape/rate, particle
  lifetime, per-particle velocity/size/color, the material it draws with),
  simulate particle state as a function of time, render instanced sprites at
  the right z-order. Not a generalization of existing code — nothing to
  extend.
- **Blend modes.** `MaterialPass::blending` is parsed but never read;
  `compose::render` always does plain "over" blending via `imageops::overlay`.
  Particles in particular are almost always additive, so this has to land
  before particles look right, not after.

(Layer rotation — `angles` — is also an already-documented composite gap in
the same code path, but doesn't come up in `scene_example2` specifically.)

This is comparable in size to §4's original effect-chain build, and is a
different fork of work than the desktop backend (§14) — the backend generalizes
*where* the chain's output goes (a real desktop window instead of one still);
this generalizes what the chain covers in the first place, for any wallpaper
the backend then plays. Tracked as its own phase, §11.

### 4.8 Composition layers — the gap `scene_example3`/`4` exposed

Two more corpus scenes (`papers/scene_example3` "Hope", `scene_example4` "Into
the night") turned up three format shapes the loader had never seen, each of
which failed the *whole* wallpaper rather than one layer. All three are now
parsed; only the last is a missing feature rather than a missing field.

- **User-property bindings.** Any value in `scene.json` may be bound to a
  wallpaper setting and is then written as `{"user": "moon", "value": true}` in
  place of the bare value — or `{"user": {"name": "clock", "condition": "1"},
  "value": false}` when a layer belongs to one setting of a combo. It reaches
  keys no struct models (`pointsize`, `text.scriptproperties.use24hFormat`,
  individual `constantshadervalues` entries), so `model::parse_scene` strips the
  wrapper document-wide before serde sees it. We keep `value`, the published
  default; the user's *own* setting lives in `project.json`'s
  `general.properties`, which `scene/` deliberately never opens. Across the
  corpus the two agree except `scene_example4`'s cloud opacity (0.3 vs 0.2), so
  wiring the two together is a real but small future decision — it would also
  give the simulator a Properties panel matching WE's own.
- **`colorrandom` with no `max`.** Half the corpus's colour initializers write
  only `min`; the absent half means "same as `min`", not zero. Reading it as
  zero would have turned those particles black — but in practice it failed the
  parse, and with it every particle system in both scenes.
- **Effect passes that are commands.** `motionblur`'s middle pass is
  `{"command":"copy"}` between two persistent `_rt_FullCompoBuffer` targets, and
  carries no `material`. The buffers it ping-pongs are *frame history*, which
  no chain here keeps, so that effect is skipped with a note instead of being
  run half-wired — the rest of the layer's chain still runs.

The feature gap underneath: **composition and post-processing layers**. WE's
"Composition Layer" and "Post-Processing Layer" are image layers whose model is
`models/util/composelayer.json` / `fullscreenlayer.json` — files that ship with
the *program*, not the wallpaper, exactly like `common.h` and `util/white`.
They carry no art. Their effect chain runs over **whatever is composited beneath
them**, and the result replaces that region. `render_frame` and the simulator
both run each chain over its own layer's texture only, so there is nothing to
feed them; today they are skipped with an omission naming what was lost. That is
six of `scene_example4`'s ten image layers (its blur, shine and cloud passes)
and two of `scene_example3`'s, so both scenes render their art correctly but
flat.

Implementing it is a change to the compositor, not to `EffectChain`: composite
in scene order into an accumulation target, and when a composition layer comes
up, bind that accumulation as the chain's base (`PassTexture::LayerBase`
already exists for exactly this shape, since `*_combine` passes bind
`_rt_imageLayerComposite_*`), run the chain, and continue compositing from its
output. The layer's own `origin`/`size`/`scale` bound the region;
`fullscreenlayer` is the whole canvas.

### 4.9 Cross-reference against Wallpaper Engine's own asset tree

A copy of WE's `assets/` (its `shaders/`, `materials/util/`, `models/util/`,
`effects/`, `particles/`, `presets/`) was read once as a reference. Nothing from
it is copied into this repository — §5.2 still stands, the shim stays our own
work, and `--we-assets` stays the way to run against the real thing. What the
read settles is *which of our guesses are wrong*, in priority order.

**Wrong output in wallpapers we already render:**

- **`ApplyBlending`'s mode numbering is not ours.** WE dispatches on the
  `BLENDMODE` *macro* (its `blendMode` argument is ignored) over 32 modes:
  1 darken, 2 multiply, 3 colorburn, 4 subtract, 5 `min`, 6 lighten, 7 screen,
  8 colordodge, 9 add, 10 `max`, 11 overlay, 12 softlight, 13 hardlight,
  14 vividlight, 15 linearlight, 16 pinlight, 17 hardmix, 18 difference,
  19 exclusion, 20 subtract, 21 reflect, 22 glow, 23 phoenix, 24 average,
  25 negation, 26 hue, 27 saturation, 28 color, 29 luminosity, 30 tint,
  31 `A + B·opacity`, 32 `mix(A, A + A·B, opacity)`, anything else normal. Our
  `common_blending.glsl` invented an 8-mode table (1 add, 2 multiply, 3 screen,
  …), so *every* mode above 7 falls through to normal — a plain replace. Eight
  call sites in the corpus: `shine_combine`, `godrays_combine` and `pulse`
  default to 9 (add), `lightshafts` (in `scene_example1`, which we treat as
  correct today) to 31, `caustics` to 32. Those combine passes are currently
  overwriting the layer where they should be adding to it.
- **`TEXnFORMAT` is never defined.** `common_fragment.h` declares
  `FORMAT_RGBA8888 0 … FORMAT_BC7 12` and WE injects `TEX0FORMAT`/`TEX1FORMAT`/
  `TEX2FORMAT` per bound texture. Our `common_fragment.glsl` is empty, so
  `#if TEX2FORMAT == FORMAT_R8` in `lightshafts.frag` compares 0 to 0 and takes
  the single-channel branch for an RGBA texture. `tex.rs` already parses the
  format out of the `.tex` header, so the values are in hand at bind time.
- **`blur7a` is a different kernel.** WE's is an asymmetric four-tap
  (offsets +2.352, +0.469, −1.409, −3.0; weights .2028/.4045/.3214/.0713), not
  the symmetric five-tap we wrote. `blur13a`'s offsets and weights are close but
  not equal; `blur3a` matches exactly. Four call sites.
- **Effect FBO scales are ignored.** An effect's `fbos` block declares
  intermediate targets with a `scale` divisor — `blur` works at scale 4,
  `godrays` and `shine` at 2 — and `prepare_effect_chain` allocates every target
  at full layer size. The blur radius is in texels of its target, so ours comes
  out several times weaker than WE's, at several times the cost.
- **`squareToQuad` has no degenerate branch.** WE falls back to an affine map
  when `det == 0`; ours divides by it.

**Missing pieces of the shim** (nothing in the corpus calls them yet, so they
fail only on the next wallpaper): `common.h`'s `hsv2rgb`, `rgb2hsv`,
`greyscale`, `M_PI_HALF`, `SQRT_2`, `SQRT_3`; all of `common_fragment.h`'s
`DecompressNormal`, `ComputeLight*`, `ConvertTexture0Format`, `ConvertSampleR8`
(10, 5 and 1 uses across WE's own shaders); and the seven headers we have no
shim for at all (`common_composite`, `common_fog`, `common_foliage`,
`common_particles`, `common_pbr`, `common_pbr_2`, `common_vertex`).
Confirmed correct as written: `rotateVec2` (identical), `M_PI`/`M_PI_2` (note
`M_PI_2` is 2π, not π/2), `blur3a`, `squareToQuad`'s main branch, and the whole
HLSL-alias approach — WE ships no GLSL alias header, so those macros do come
from its compiler and writing our own was the only option.

**Structural gaps the assets make concrete:**

- **`bind` is the authority for pass inputs, and `previous` means the effect's
  input image — not the previous pass.** Every definition pass carries
  `bind: [{"name": "previous" | "_rt_<fbo>", "index": n}]` (43 of 83 passes in
  the 46-effect library). `godrays`/`shine` bind their half-res rays at index 0
  and `previous` at index 1; `blur` binds `previous` at index 2. Our
  "slot 0 is always the previous pass, everything else from scene.json" model
  gets the same answer for linear chains and for the three combine passes we
  special-cased, and would get the wrong one for anything else.
- **Composition layers** resolve exactly as §4.8 guessed:
  `models/util/composelayer.json` → `materials/util/composelayer.json` →
  shader `composelayer` with `textures: ["_rt_FullFrameBuffer"]` and
  `passthrough: true`; `composelayer.vert` draws a quad in texture space while
  carrying the layer's MVP-transformed position through as `v_ScreenCoord`, and
  the fragment samples the frame buffer at `screenCoord/w · 0.5 + 0.5` — i.e.
  the layer's image *is* the frame so far, cropped to the layer's on-screen
  rect. `fullscreenlayer` is the same with the `passthrough` shader and the
  whole canvas. There is also `solidlayer` (shader `flat`), `projectlayer` and
  `composelayer_depthtest`, and a `CLEARALPHA` variant.
- **Layer-level blend modes go through the same 32-mode table.**
  `genericimage2.frag` binds `g_Texture4 = _rt_FullFrameBuffer` under
  `#if BLENDMODE` and blends the layer against the frame in-shader. That is
  `scene.json`'s `colorBlendMode`, which we do not parse — it is 0 on all 22
  corpus objects, so nothing is wrong yet, but it is not the material's
  `translucent`/`additive` (that one is GL blend state) and needs the frame
  buffer bound to work at all.
- **Particle vocabulary is about half implemented.** Across WE's ~250 presets:
  emitters `sphererandom`/`boxrandom` — both done. Initializers missing
  `rotationrandom` (93 uses), `angularvelocityrandom` (45),
  `mapsequencebetweencontrolpoints`, `mapsequencearoundcontrolpoint`,
  `positionoffsetrandom`, `remapinitialvalue`, `hsvcolorrandom`. Operators
  missing `angularmovement` (45), `colorchange` (30), `remapvalue`,
  `alphachange`, `reducemovementnearcontrolpoint`, `vortex`(+`_v2`),
  `maintaindistance*`, `boids`, `capvelocity`. Renderers missing `rope` (24)
  and `ropetrail` (17). These are exactly the "rendered without an
  initializer / without a renderer" notes `scene_example3`/`4` print.
- **`.tex` formats.** WE's format enum runs to `FORMAT_BC7 12` and includes
  RGB888, RGB565, ETC1/ETC2, RG1616F and R16F; `tex.rs` handles RGBA8888,
  DXT1/3/5, RG88 and R8. BC7 is the likely one to hit.
- **WE's JSON parser is lenient.** Its own
  `effects/fluidsimulation/effect.json` has a trailing comma, which
  `serde_json` rejects — and a wallpaper using that effect ships a copy of it.
- **Builtin textures exist as real files**: `materials/util/` has `white`,
  `black`, `noflow`, `noise`, `clouds_256`, `perlin_256`, `uniform_256`,
  `flatnormal`, `fur`. We synthesize the first four and stand in for the noise
  ones; the synthesized noise is not WE's, so any effect sampling it differs.
- **Not modelled at all**: text/clock layers (`text`, `font`, `pointsize`,
  `horizontalalign`, … — the clock in both new scenes), `parallaxDepth`
  (nonzero on 16 corpus objects; WE moves layers with the cursor), and the
  `assets/scripts/` JS binding that drives user-property scripts.

**Landed from this pass** (all of it our own code; nothing of theirs shipped):

- the 32-mode blend table, the exact blur kernels (`blur3a`/`7a`/`13a` plus the
  radial trio), `hsv2rgb`/`rgb2hsv`/`greyscale`/`M_PI_HALF`/`SQRT_2`/`SQRT_3`,
  `squareToQuad`'s affine branch, and `common_fragment`'s `FORMAT_*` table with
  `TEXnFORMAT` defaulting to RGBA;
- `preprocess` no longer hands a shader a zero-valued stand-in for a name the
  shim defines — that was what made `#if TEX2FORMAT == FORMAT_R8` true for every
  texture;
- effect `fbos`/`target`/`bind`: passes render at their declared scale (blur at
  4, godrays and shine at 2) and take their inputs from the `bind` list;
- composition and post-processing layers, in the live simulator: the frame is
  lifted out of the composite by `pass::copy_region` into a rect-sized target,
  fed to the layer's chain as its base, and drawn back over the same rectangle.
  The rectangle is clipped to the canvas — `scene_example4`'s cloud layer is
  3.07x the canvas in Y, and only the part over the canvas has a frame under it.
  `export` still lists them as omissions: that path only has the frame as CPU
  pixels after the flatten.

Two more gaps surfaced while doing it, both fixed:

- **`scene.json` sets combos per pass.** An effect pass carries a `combos` map
  next to `constantshadervalues`, and it is how a separable blur's second pass
  is told to run vertically — `godrays`, `shine` and `blurprecise` all write
  `{"VERTICAL": 1}`. Ignoring it meant both halves of every gaussian blurred the
  same axis.
- **The two stages disagree about a uniform's type.** WE's own `clouds`
  declares `g_CloudSpeeds` as `vec2` in the vertex shader and `vec4` in the
  fragment. Only one survives linking, and uploading the other width is an
  `INVALID_OPERATION` that kills the draw — which is why that layer failed the
  moment composition layers started running it. `pass::Program` now reads the
  active uniform widths back from the linked program and skips any value that
  does not match.

**The shim is now verified against the real headers, symbol by symbol.** Every
function and macro Wallpaper Engine's five wallpaper-facing headers export was
diffed against ours by name and arity: **zero mismatches**, and the only names
we do not carry are `inverse(mat3)` (WE declares it only under `#if HLSL`;
GLSL 330 has it built in) and the `ComputeLight*` / `ComputeMaterialSpecular*`
family, which only WE's own model shaders call and no wallpaper packages. Ours
additionally carries the HLSL aliases (`mul`, `saturate`, `CAST*`,
`texSample2D`, …) that WE injects from its compiler rather than a header, and
the `TEXnFORMAT` fallbacks. Those five are the complete surface: across all
eight scene wallpapers, packaged shaders include *only* `common.h`,
`common_blending.h`, `common_perspective.h`, `common_blur.h` and
`common_fragment.h` — the other seven headers belong to engine shaders that
never ship inside a wallpaper.

That matters because it settles attribution. With the headers proven
equivalent, a shader that still fails is failing on our own code or on its own
source, and the split turned out to be:

- **Ours, now fixed.** A `varying` the two stages declare with *different types*
  — `scene_example6`'s `color_grading` says `vec4` in the vertex stage and
  `vec2` in the fragment. GLSL 1.20's `varying` was matched loosely enough to
  survive that; rewritten to 330's `in`/`out` the linker rejects it. The
  fragment's declaration now takes the vertex's type and the body reads it
  through a `#define` restoring the width it expected (`preprocess::
  reconcile_varyings`). Five of that scene's six chains were failing on it.
  Also **unbalanced conditionals**: `scene_example5`'s `iris_movement__.vert`
  carries one `#endif` too many, which WE consumes because it resolves `#if`
  itself and we do not (§6). `balance_conditionals` drops directives with
  nothing open and closes what is left open.
- **Theirs, not fixable without an HLSL front end.** Every remaining failure —
  nine chains across `scene_example5`/`7`/`8`, all third-party Workshop shaders —
  is a construct only HLSL accepts: brace array initialisers
  (`const vec2 kernel[16] = { … }`), implicit `float`→`int` in an initialiser
  (`for (int i = u_MinFreqRange; …)`), implicit `bool`→`float`
  (`depth *= (depth < limit) * 6.0`), implicit vector truncation
  (`vec2 da = someVec4 * someVec2`), argument-order scalar promotion
  (`max(0, someVec3)`), and `%` on floats. Wallpaper Engine compiles these as
  HLSL on Windows; its own GL backend would reject them the same way we do.
  The layer still renders, minus that one effect.

**Decision this forces.** §5.2 chose a shim written from measured usage because
the headers are not ours to ship. That still holds for their *source*, but the
mode numbering above is an interop fact and the blend formulas are standard
compositing math we can write ourselves; matching them is a rewrite of our own
header, not a copy. The alternative — keep the shim approximate and treat
`--we-assets` as the accurate path — is only defensible if the accurate path is
reachable at render time, which today it is not: `--we-assets` exists on the
`shaders` subcommand alone. Wiring it into `simulate` and `export` would also
make the shim's error measurable (dump the same frame both ways, PSNR the
difference), which is what §12 asks for.

### 4.10 Particle sprites — settled against real Wallpaper Engine output

The Windows peer session captured a real WE desktop at 1920x1080 for
`scene_example6` and `scene_example8`, which is the first behavioural ground
truth this project has had (`preview.gif` is a 160x160 centre crop and is
useless for comparison). Two things fell out of it.

**Object parenting was missing entirely.** A scene is a tree: `origin`, `scale`
and `angles` on a child are relative to its parent. `scene_example6` parents 31
of its 59 objects, and read absolutely its umbrella sits at knee height rather
than over the character's head. Fixed in `compose::resolve_anchors`; the other
seven corpus scenes have no parenting and re-render bit-identically.

**Every particle was drawn as a bright radial gradient**, which cost a 31%
whole-frame brightness error. Measuring the real sprites shows they fall into
two classes, and conflating them is the bug:

| sprite | mean alpha | mean RGB |
|---|---:|---:|
| `particle/halo` | 0.152 | 1.000 |
| `particle/halo_4` | 0.040 | 1.000 |
| `particle/fog/fog1` | 1.000 | 0.052 |
| `particle/nature/rain1` | 1.000 | 0.006 |

The second class is alpha-*opaque* with the picture in RGB, drawn additively,
where black contributes nothing — `nature/rain1` is 99.4% black with a few
bright streaks, and `scene_example6` draws 64 of them at 800–1600 scene units
each. A tinted gradient in its place adds roughly thirty times the light it
should, which is exactly the fog that blanketed our canvas.

**Decision.** These are engine assets, so §5.2 applies: not ours to ship, and
downloading a copy from elsewhere does not change that. Instead
`tools/make_sprites.py` *draws* eleven stand-ins and calibrates each one's mean
to the measured table above; they are checked in under `src/scene/sprites/` and
`include_bytes!`d (about 48 KB). Energy is matched by construction, shape only
approximately — which is the right trade, because energy was the systematic
error and shape is what a Perlin cloud gets close enough on. `sprite::resolve`
still prefers a real texture: package first (workshop sprites live there), then
an install if one is pointed at, then the stand-in. The tests re-measure the
checked-in PNGs, so the calibration cannot drift silently.

### 4.11 `colorBlendMode` — the layer's blend against the frame

`scene.json`'s `colorBlendMode` picks a Photoshop-style blend of a layer against
everything under it, in `weBlendColor`'s numbering. It is **not** the material's
`translucent`/`additive`, which is GL blend state; this one needs the
destination readable in the shader, so `composite_layer_blended` copies the
frame's rectangle out with `copy_region` and blends against it, taking the
fixed-function path at mode 0.

§4.9 recorded this as harmless because it was 0 on all 22 objects then in view.
Across the whole corpus it is 0 on 179 of 183 — but the four exceptions are all
in `scene_example8`, at modes 1 (darken), 4 (linear burn), 11 (overlay) and 12
(soft light). Two are its cloud layers, and all four *darken*: compositing them
plain-over pasted the clouds onto the sky as bright white slabs. Honouring the
mode is worth **+3.4 dB** on that scene by itself. The other two are text layers,
which nothing renders yet.

### 4.12 Measured against real Wallpaper Engine

Seven captures, one per corpus scene bar `Magic-Hat` (not a Workshop item).
Whole-frame PSNR against the capture, and mean luminance beside the reference's:

| scene | PSNR before | PSNR now | luminance | reference |
|---|---:|---:|---:|---:|
| ex1 ATRI | 28.62 | 28.62 | 178.0 | 179.0 |
| ex2 Dusk Town | 28.28 | 28.28 | 149.6 | 149.7 |
| ex3 Hope | 17.84 | 17.87 | 131.6 | 131.0 |
| ex4 Into the night | 25.21 | 25.68 | 68.0 | 69.0 |
| ex5 听星·伊蕾娜 | 21.58 | 21.35 | 89.7 | 89.5 |
| ex6 Rainy Day | 12.99 | 18.52 | 56.4 | 55.6 |
| ex8 Matte Clouds | 14.17 | 22.94 | 86.7 | 81.3 |

Caveats that decide which numbers are trustworthy, all from the install's
`config.json`: ex2/3/4/5 had **no** saved property overrides, so they render at
stock defaults and their numbers are clean. ex6's reference has "Screen water
flow" and "Screen raindrops" switched on when both default to off, so part of
its residual gap is a different picture rather than a worse one. ex1's has
`rate: 200`, which scales animation speed, so its timing cannot match. ex8 is
the single most trustworthy target — no overrides, no audio-reactive layers.

ex5's 0.23 dB dip is particle *phase*: our RNG places its stars differently from
WE's, and two previously-dropped chains now contribute. Its luminance moved
closer to the reference over the same change.

### 4.13 Scene bloom and HDR — what `scene_example8`'s sunset was missing

`general.bloom` is a post-process over the finished frame that nothing here ran.
It is not a garnish: ex8 is a sunset whose sun is a large saturated glow that
exists *only* because of it, and without it the horizon renders ~30% dim and the
sun reads as a few lit cloud edges. `render/bloom.rs` implements it — the
extract and the combine verbatim from `downsample_quarter_bloom.frag` and
`combine.frag`, the blur as a mip chain because that is what `bloomhdriterations`
and `bloomhdrscatter` describe.

Two things had to land with it:

- **`general.hdr` means floating-point targets** (`pass::Format`). A bloom
  threshold is meaningless if nothing can exceed it, and in an 8-bit target the
  sun clamps to 1.0 alongside the rest of the sky, so a 0.65 threshold extracts
  almost nothing. Worth 0.46 dB on ex8 by itself.
- **An HDR scene tunes `bloomhdr*` and leaves the plain pair at defaults.** ex8
  writes 0.12/0.55/0.67 for HDR and stock 2.0/0.65 for the other, so reading the
  wrong pair drives the bloom sixteen times too hard.

Also found while chasing the sun: `g_PointerPosition` is declared by the shaders
that use it, never by a header, so nothing bound it and GL left it at (0, 0) —
the corner. Anything positioned by the cursor was a full screen out. It is bound
to the centre until a real pointer is tracked (§14).

### 4.14 Two particle bugs the per-layer profile found

After the bloom landed, ex8 was still 19% *dim* overall, and the deficit grew
with brightness — which reads like a tone curve and is not one. A gamma of ~0.8
recovered 1.4 dB, but 0.8 is no colour-space exponent and the sRGB encode
(0.4545) made it markedly worse, so the "wrong colour space" explanation was
dead. Rather than fit a curve, the thing to do was ask *which layer* darkens the
frame: compositing the stack one layer at a time and measuring gives that
directly.

The answer was stark. **MAIN alone** — the base image and its own chain — comes
out at luminance 79.8 against the capture's 81.3, with the sun at 165 against
157. It was already right. Adding the twenty fog and smoke systems above it
dropped the frame to 58.6. The particles were *subtracting* light.

Two distinct causes, both now fixed:

- **Single-channel sprites are coverage masks, not grey images.** WE stores
  `particle/fog/fog1` and `particle/nature/rain1` as `R8`, and
  `particle/misc/wave` as `RG88` with an unused second channel; `tex.rs` expands
  those to opaque grey, which is right for dumping a viewable PNG and wrong for a
  sprite. Drawn by a `translucent` system, an opaque near-black fog puff *paints*
  near-black. `sprite.rs` now remaps any sprite that is opaque everywhere to
  white-over-luminance-coverage. Note this is provably a no-op for *additive*
  systems — premultiplied, `rgb 0.05 × a 1.0` and `rgb 1.0 × a 0.05` are the same
  colour — which is why it moved ex8 and left ex3–ex6 alone.
- **`REFRACT` particles multiply the frame; they do not cover it.**
  `genericparticle.frag` under `#if REFRACT` binds `_rt_FullFrameBuffer` and ends
  `color.rgb *= texSample2D(g_Texture3, refractTexCoord).rgb`. A white sprite over
  a dark sky therefore *disappears* — which is why `wet_snow1` is invisible in
  Wallpaper Engine and was a field of white discs here. The compositor already
  had the machinery from §4.11: multiplying the layer in is `mix(dst, dst·src,
  src.a)`, blend mode 2, and reproduces it exactly bar the screen-space offset
  the normal map adds. Nine of `scene_example6`'s particle materials declare
  `REFRACT`, one of `scene_example4`'s, three of `scene_example8`'s.

Together these are worth **+4.1 dB** on ex8, +0.8 on ex6 and +0.5 on ex4, and
they also confirmed the bloom parameters: with the frame's brightness correct,
the scene's own `bloomhdrstrength` of 0.12 is the clear optimum and anything
stronger is monotonically worse. The earlier sweep that preferred 4.0 was
compensating for the mask bug.

### 4.15 Rotation, text layers, and the noise floor

**Roll.** `angles.z` is applied by the compositor: the layer's quad places itself
from `u_Rect` rather than from the viewport, so it can turn about its own centre
and reach outside its rect. At zero roll it reproduces the old viewport
placement exactly, which `scene_example1` and `scene_example2` — neither of
which rotates anything — confirm by re-rendering bit-identically.

Particle layers are the exception and take their roll in `particle::Placement`
instead. Their raster is the whole canvas, so rolling the quad would swing the
entire field about the canvas centre rather than about the emitter. Most of the
corpus's rotation is on particles: 31 systems in `scene_example8`, 16 in
`scene_example6`, against five image layers between them.

**Text layers.** An object with a `text` field is a text layer, rendered to a
raster the ordinary image path then carries. Two things about the format
dominate the implementation:

- *The text is usually a script.* It arrives as `{"script", "scriptproperties",
  "value"}`, and `value` is the design-time preview. Sometimes that is a real
  string (`"PM O8:24\nApr. 14 2025"`), sometimes a bare placeholder
  (`"<Date>"`). A placeholder is not drawn — putting the literal text `<Date>` on
  a wallpaper is worse than leaving the layer out — and the omission says which.
- *The font usually is not ours.* Exactly one font in the corpus ships inside a
  package; the rest name a Wallpaper Engine asset or a system family. The search
  is package, then an install, then the system's own fonts by family, so a
  missing engine font degrades to a similar face instead of dropping the layer.

Glyph size is **fitted to the box**, not taken from `pointsize`. The two do not
agree: `scene_example8`'s clock box is exactly two 32pt lines tall while its
watermark box is eight times that, and WE draws the watermark filling it. Fitting
reproduces both without a magic constant.

**Alpha keyframe tracks**, because text needs them: `scene_example8`'s intro
watermark fades 1 → 0 over its first seven seconds, and frozen at its published
1.0 it would sit on the wallpaper forever. `hoist_alpha_tracks` copies the track
somewhere `strip_driven_values` will not flatten it — every control point carries
`frame`, which is itself a driver key — and the compositor multiplies the layer's
baked alpha by the sampled ratio.

**A note on the metric.** Rendering `scene_example8` at t=2.6, t=10 and t=60 and
scoring each against the capture taken at that same moment gives 21.7, 20.5 and
22.4 dB — for one unchanged renderer. The scene is animated and our particles do
not track WE's frame for frame, so whole-frame PSNR here carries a noise floor
of roughly ±1.5 dB. Differences smaller than that are not evidence. The findings
in §4.11–4.14 were worth acting on because they were *multiple* dB and because
each had a mechanism behind it; the last few tenths are not worth chasing with
this metric, and a region crop or a structural check is the better tool at that
scale.

**What is left on ex8.** Luminance is 87 against 83 — a few percent over, from
the wrong side now. Camera zoom (1.03) and camera shake are both on in the
wallpaper's own settings and neither is implemented, which offsets every pixel in
the frame. The sprites remain plausible rather than identical: a Perlin cloud is
not WE's cloud.

### 4.16 Where the live frame time actually goes

The live simulator runs the corpus's heavy scenes at 9–17 fps where Wallpaper
Engine runs them at 60, and the gap is entirely CPU: `scene_example3` settles at
66 ms of CPU a frame against 4 ms of upload and 0 ms of GPU, `scene_example6` at
~105 ms, `scene_example8` at ~40 ms.

The obvious suspect is wrong, and it is worth writing down so it is not
re-suspected. Particles are re-integrated from birth every frame — the stateless
design that makes any `g_Time` independently addressable — and the per-frame cost
visibly climbs (10 ms → 66 ms over `scene_example3`'s first few seconds), which
looks exactly like an integration cost growing with particle age. It is not.
Measured by disabling the particle raster and leaving everything else running:

| `scene_example3` | CPU/frame |
|---|---|
| full | 66 ms |
| particle raster disabled, integrating from birth | 9 ms |
| particle raster disabled, one incremental step per particle | 9 ms |

Integration is free — 1600 particles × up to 90 steps is a few hundred thousand
float operations spread across every core by `rayon`. The climb is the particle
*population* filling to `maxcount`, and the cost is tiny-skia filling sprite
rects. It is pixel-bound, not particle-bound: dropping `particle_sim_scale` from
0.28 to 0.2 (a 0.51× area) takes 65 ms to 33 ms, and turning off anti-aliasing
and dropping to nearest-neighbour sampling change nothing at all.

So an incremental live particle state — keeping each slot's `Motion` and
advancing it by one frame — buys nothing, and was reverted after being measured.
The fix is to stop rasterizing particles on the CPU: draw them as instanced
quads through the GL context that is already open, which removes the fill *and*
the 4–19 ms per frame spent uploading the finished raster back to the GPU.

### 4.17 Particles on the GPU — what it cost and what it bought

Done (`render/particles.rs`). Simulation now stops at a `particle::DrawList`;
what turns that into pixels is the caller's, so the same simulation feeds the
instanced GL pass and tiny-skia both. The CPU raster survives as the still
exporter's path and as the live fallback for a particle layer carrying effects
— no corpus particle object has any, all 41 across eight scenes declare zero.

| live, at `g_Time` 6 s | before | after |
|---|---|---|
| `scene_example3` | 15 fps (cpu 66, upload 4, gpu 0 ms) | **120 fps** (cpu 0, upload 0, gpu 1 ms) |
| `scene_example6` | 10 fps (cpu 95, upload 15, gpu 2 ms) | **50 fps** (cpu 15, upload 4, gpu 3 ms) |
| `scene_example8` | 15 fps (cpu 43, upload 19, gpu 5 ms) | 15 fps (cpu 1, upload 0, gpu 50 ms) |

Verified as a number, not an impression: forcing the CPU path at full
resolution and diffing the composited frame puts the instanced pass at 51.2 dB
PSNR on `scene_example3` and 47.5 dB on `scene_example6`. Against the
*unmodified* pre-change baseline `scene_example3` is 65.8 dB — the remaining
gap on the other two is resolution, not rasterization, because the CPU path had
to shrink its canvas and this one does not.

Three things were only learnt by measuring:

- **The per-layer composite was the cost, not the fill.** The obvious shape —
  each system into its own scratch target, then composite — made
  `scene_example8` *worse* than the CPU path (11 fps, 88 ms of GPU). Dropping
  the scratch to 0.35× the canvas moved it by 5 ms, which is what proved the
  fill was never the problem: 35 canvas-sized composites were. Drawing
  straight onto the frame instead removed both the clear and the composite and
  took it to 50 ms. A system only needs a layer of its own when it mixes
  `additive` and `translucent` presets, or its layer carries a
  `colorBlendMode` or an alpha track; otherwise summing into a transparent
  layer and adding that layer is the same arithmetic as adding each particle,
  and alpha-over is associative.
- **Premultiplied alpha is what makes the blends match.** GL's fixed-function
  blending reproduces tiny-skia's `Plus`/`SourceOver` exactly on premultiplied
  source and cannot on straight. So the particle pass writes premultiplied and
  `composite_layer_blended` takes a flag, rather than the pass paying a
  full-canvas unpremultiply the compositor would only undo. The flag matters
  for exactly one thing: the blend-mode arms are written against straight
  alpha, which is the path `scene_example8`'s refracting wet snow takes.
- **`glDrawArraysInstancedBaseInstance` is GL 4.2 and macOS stops at 4.1.**
  Each batch re-points its four instance attributes at its own slice instead.

`scene_example8` is unchanged in frame rate but draws its particles at the full
4K canvas instead of 0.24× of it, and is visibly closer to its own
`preview.gif` for it — the upscale was smearing fog across the whole frame.
Its remaining 50 ms is real overdraw: 35 fog and smoke systems whose sprites
each cover most of a 4K canvas. The next move for it is one of

- merge a *run* of consecutive particle layers into one reduced-resolution
  scratch composited once — `scene_example8`'s 41 particle objects fall into
  runs of 29, 1, 6 and 4, so the big run would collapse to a single composite
  and could then afford to be drawn at half resolution;
- or render the whole scene at display resolution rather than at the authored
  canvas resolution, which is what Wallpaper Engine itself does and would help
  every layer, not just particles.

### 4.18 Render resolution is the next lever, and it is the *chains* that matter

Measured, not built — the implementation is written up at the end of this
section along with the bug that stopped it.

`scene_example8` is not particle-bound and never was after §4.17. Ablations,
all at `g_Time` 6 s:

| `scene_example8` | GPU/frame |
|---|---|
| full | 46 ms |
| particle pass disabled entirely | 41 ms |
| particle coverage shrunk 4x | 45 ms |

So the whole particle pass is 5 ms of 46, and its *fill* is nearly free. Its
batching is already tight as well: 1343 instances in 37 draw calls across 40
systems, so draw-call count is not it either.

What is left is resolution — but only if the **layer images** scale with it.
This distinction cost a wrong conclusion and is the point of writing it down.
Scaling the composite target and every layer rect while leaving each layer's
own image at canvas size leaves every effect chain running at 4K, because a
chain's passes are sized from the image it was compiled against:

| `scene_example8` render scale | composite + rects only | + layer images |
|---|---|---|
| 1.0 | 44 ms | 45 ms |
| 0.667 | 39 ms | **21 ms** |
| 0.5 | 37 ms | **11 ms** |

Quartering the pixel count is worth 16% if the chains stay at 4K and **4x** if
they do not — 20 fps to 66. The chains are the cost, and they are only reachable
through the image handed to `prepare_effect_chain`.

This is also what Wallpaper Engine does. Captures from a real install on a
1080p machine come back 1920x1080 for `scene_example8`, whose authored canvas
is 3840x2160 — it renders at the display, not at the canvas, and 0.5 is exactly
the scale that gives us 66 fps.

**The attempt, and why it is not in.** A `render_scale` taken once from
`window.current_monitor()` (capped at 1.0 and at the canvas), layer images
resized to it before upload and chain compile, every rect, particle placement,
composite and bloom target in render pixels — all inside `simulate.rs`, no
`compose.rs` change needed. It compiles clean, keeps all 171 tests, and leaves
`scene_example3` untouched at 120 fps because its canvas is already 1080p so
the scale is exactly 1.0.

It renders wrong. `scene_example8` at 0.89 comes out with large right-angled
transparent blocks — 10.5 dB PSNR against the same frame rendered full-size and
downscaled, with the *alpha* channel at 6.5 dB, which is the tell: something is
writing alpha 0 over the frame. The only path that writes without blending is
the `colorBlendMode` arm of `composite_layer_blended`, which replaces the
destination with `dst.a` taken from the backdrop copy. So the suspect is a
layer whose backdrop rectangle and backdrop *target* now disagree about which
space they are in — `blend_backdrop` is still handed the authored canvas size
for particle layers while `full_rect` is in render pixels, and
`scene_example8`'s layer 30 is a soft-light layer whose rect (`-152,884
5877x3306`) extends outside the canvas on three sides, which is exactly where a
space mismatch would show up first. Not yet confirmed.

Whoever picks this up: every canvas-space quantity has to move to render space
in one go, and the ones that are easy to miss are the ones not derived from an
image — `full_rect`, `blend_backdrop`'s size, a composition layer's `region`
target, and both particle scales, which must compose with the render scale
rather than replace it.

### 4.19 Measured against real Wallpaper Engine, second round

§4.12's captures are gone and its numbers are not comparable to these: those were
taken before the bloom, the sprite-coverage and REFRACT fixes (§4.13, §4.14) and
before particles moved to the GPU (§4.17). A fresh set was taken on a Windows box
running the real thing — seven wallpapers, lossless 1920x1080 `CopyFromScreen`
grabs, each reloaded through `wallpaper32.exe -control openWallpaper` with a
stopwatch started when the control call returned and the frame taken at a recorded
~10.02 s. They live in `papers/_we_captures/`, ignored like the rest of `papers/`.

Two things about the captures decide how to read them. The desktop is **1920x1080**,
so Wallpaper Engine renders `scene_example8` at half its authored 4K canvas and ours
has to be downscaled to match — the same finding §4.18 reaches from the performance
side, from the other direction. And a taskbar occupies the bottom ~44 px of every
frame, excluded from every number below.

Only two of the seven are not clean baselines: `scene_example6` has "Screen water
flow" and "Screen raindrops" switched on where both default to off, and
`scene_example1` runs at `rate: 200`, so its capture sits at a `g_Time` of ~20.06 s
rather than ~10.03 s. `scene_example6` and `scene_example8` also carry a text layer
driven by the real system clock, which no `g_Time` will ever match.

| scene | PSNR | our luminance | reference |
|---|---:|---:|---:|
| ex1 ATRI | 25.22 | 188.6 | 190.8 |
| ex2 Dusk Town | 29.88 | 153.3 | 152.6 |
| ex3 Hope | 14.73 | 162.7 | 133.1 |
| ex4 Into the night | 23.82 | 62.5 | 62.1 |
| ex5 听星·伊蕾娜 | 19.73 | 83.0 | 82.5 |
| ex6 Rainy Day | 14.77 | 42.0 | 43.6 |
| ex8 Matte Clouds | 19.44 | 91.5 | 77.3 |

**There is no global tone or colour-space error, and that is now settled rather
than suspected.** §4.14 rejected a gamma by argument; this rejects it by
measurement. Binning every pixel by the reference's own value and taking our mean
in each bin gives, for `scene_example2` and `scene_example4`, a deviation of at
most 2 levels anywhere from black to white — an identity transfer. Whatever is
wrong with ex3 and ex8 is *something in the frame*, not a curve over it, which is
a far smaller search space than a curve fit.

For ex3 the same transfer is a flat **+35 to +40 levels at every input level** up
to ~200 — the signature of a uniform veil laid over the picture. `SIMULATE_LAYERS`
finds it in one layer: compositing 0–13 gives luminance 140.1, and adding layer 14
alone — a `particles/presets/fog1.json` system with `instanceoverride`
`{alpha: 0.3, rate: 0.6}` — takes it to 162.9 and costs 1.8 dB. Solving
`162.9 = 140.1(1-a) + 255a` puts that layer at a 20 % white veil over the whole
canvas. The override *is* read and applied (`particle.rs` folds `alpha` into the
sprite weight and `rate` into the emission rate), so the surplus is population
rather than a dropped field: `maxcount` is a hard slot count, so a system that
fills to it ignores `rate` entirely at steady state and runs ~1.7x too many
particles.

For ex8 the surplus is +15 levels through the dark and middle range and nothing at
the top, and two causes are stacked. One is layer 30, "Basic Clouds Movement" —
soft light, scale 3.06, roll 0.103, rect `-152,884 5877x3306`. It composites a
bright band with a hard tilted edge across the upper sky that the capture does not
have; dropping that one layer is worth **+2.3 dB** (19.65 → 21.94) and moves
luminance 90.7 → 84.9. It is the one corpus layer whose blend-mode backdrop is
copied over a rectangle mostly *off* the canvas, and §4.18's render scaling landed
on the same layer from a different direction, which makes that backdrop path the
prime suspect for both. The other cause is underneath it: sampling the frame before
layer 30 lands gives [42,42,37] and [66,63,53] where the capture has [34,39,43] and
[45,50,54] — our dark upper sky is both brighter and *warmer*, while the sun itself
matches ([255,222,168] against [255,217,139]). A warm lift confined to the dark end
with the bright end already correct is what an over-wide bloom scatter looks like —
but a soft-light layer clamping to the wrong backdrop edge would produce it too, so
check the backdrop copy at scale 1.0 before assuming the base frame is
independently wrong. One cause may explain both.

Two dead ends worth not re-running. `pow` with a negative base is undefined in
GLSL and `workshop/2098390419/scroll` relies on D3D folding `pow(v, 2.0)` into
`v*v` — but a shim that mirrors the sign for integral exponents changes the frame
by **exactly zero bits** on this driver, which evidently already folds it. And
`blurprecise`, the obvious suspect for a smear, cannot be one: its offsets are
`g_Scale / g_TextureNResolution`, and `SIMULATE_TRACE` confirms the resolutions it
is handed are the real target sizes, putting its widest tap at about two pixels.

---

## 5. The `common.h` problem

### 5.1 What is missing

Shaders `#include` four headers that ship with **Wallpaper Engine itself**, not
with wallpapers:

- `common.h`
- `common_fragment.h`
- `common_blending.h`
- `common_perspective.h`

None are in the package.

### 5.2 Chosen approach: write our own shim

Measured usage across the sample's six shaders is small:

| Identifier | Uses | Meaning |
|---|---|---|
| `texSample2D` | 18 | texture sample wrapper |
| `CAST2` / `CAST4` | 13 | HLSL-style splat constructors |
| `mul` | 5 | HLSL matrix multiply (operand order differs from GLSL) |
| `M_PI`, `M_PI_2` | 6 | constants |
| `saturate` | 3 | `clamp(x, 0, 1)` |
| `frac` | 2 | `fract` |

That is a shim of roughly 30 lines, not a library. WE's shader dialect is
HLSL-flavoured GLSL because the same source is compiled for both D3D and GL —
so most of these are mechanical HLSL→GLSL aliases.

**Watch out for `mul`.** HLSL's `mul(v, M)` is row-vector convention; GLSL's
`M * v` is column-vector. Getting this backwards produces plausible-looking but
wrong geometry. It needs a deliberate test, not a guess.

`common_blending.h` (blend modes, driven by the `BLENDMODE` combo) and
`common_perspective.h` (vertex transforms) are larger surfaces than the core
shim and should be filled in on demand, per effect.

### 5.3 Fallback and verification

- **Fallback:** a `--we-assets <path>` flag that reads the real headers from a
  Wallpaper Engine install. Useful for cross-checking our shim and for effects
  whose helpers we have not yet reimplemented. These files are **never
  redistributed** — read from the user's own install at runtime only.
- **Verification:** golden-image comparison. Render a frame with the shim, then
  the same frame with the real headers, and diff. That converts "does our shim
  match?" from guesswork into a measurement. This is the highest-risk area of
  the project and deserves the strongest verification.

### 5.4 Vertex shaders

Effect vertex shaders compute varyings (`v_TexCoord`, `v_Bounds`,
`v_TexCoordMask`) and include `common_perspective.h`. Two options:

- **(a) Run them as written**, supplying the standard `g_ModelViewProjection`
  style uniforms. Faithful, but pulls in the whole perspective header.
- **(b) Substitute our own fullscreen-quad vertex stage** that produces the
  varyings the fragment shaders actually read.

Start with (b) for the fullscreen post-process case — it is far less surface —
and fall back to (a) if any effect's vertex stage turns out to do real work
(some sway effects may displace per-vertex). Decide per effect, not globally.

---

## 6. Shader translation target

Source dialect: GLSL ~1.20 with `varying`, `gl_FragColor`, plus HLSL intrinsics
and `[COMBO]` preprocessor options (`NOISE`, `DIRECTION`, `MASK`, `BLENDMODE`,
`RAYMODE`, …) that select variants via `#if`.

The preprocessor must therefore: resolve `#include`s against our shim, inject
`#define`s for the combo values chosen in `scene.json`, and rewrite the dialect.

| Option | Pros | Cons |
|---|---|---|
| **wgpu + naga**, translating to GLSL 330 | Cross-platform (Metal/Vulkan/DX12), actively maintained, future-proof on macOS | naga's GLSL *frontend* is less mature than its WGSL path; may hit unsupported constructs |
| **glow + glutin** (OpenGL 3.3/4.1) | Accepts near-source GLSL, least translation work | OpenGL is deprecated on macOS (capped at 4.1); a dead end long-term |

**Decision: glow + glutin, reversing the recommendation below.** Measured
against the real corpus rather than assumed: naga's GLSL frontend rejects
combined `uniform sampler2D` declarations and bare (non-block) global
uniforms, both used throughout, so 0 of 12 sampled shaders parsed. glow+glutin
compiles all 28 corpus shaders through a real GL 3.3 core context with only
the mechanical dialect rewrite `preprocess.rs` already does. OpenGL's macOS
deprecation is a real long-term cost, accepted for now; `naga`'s SPIR-V path
remains the documented escape hatch if a future macOS drops OpenGL outright.

*(Superseded reasoning, kept for the record: "We need a preprocessor
regardless… so `varying`→`in`/`out` and `gl_FragColor`→a declared output is a
small increment on work already required" — true, but it assumed naga's GLSL
frontend would accept the corpus's shaders at all, which it did not.)*

### 6.1 HLSL-isms the GL driver rejects (`shader/hlsl.rs`)

The dialect rewrite is not enough. Wallpaper Engine compiles the same source as
HLSL on Direct3D, and workshop authors write against whichever backend they run,
so shaders that have only ever been compiled on Windows carry constructs GLSL
rejects. A rejected shader drops that layer's **entire** chain, and six chains
across four corpus scenes were being lost this way — including `lens_flare_sun`,
which is the whole reason `scene_example8` rendered with no sun on the horizon.

Eight constructs account for every failure. HLSL's rule behind the first four is
*implicit truncation*: a wider vector assigned to or combined with a narrower one
silently keeps the leading components. GLSL does exactly that through a
constructor — `float(v3)` and `vec2(v4)` are legal — so the fix is to make each
conversion explicit rather than to reimplement it.

| construct | GLSL says | seen in |
|---|---|---|
| `float g = <vec3>;` | incompatible types | `edge_glow`, `lens_flare_sun` |
| `int i = <float>;` | incompatible types | `test_shader` |
| `albedo.rgb = <vec4>;` | incompatible in assignment | `gaussian` |
| `<vec4> * <vec2>` | `*` does not operate on those | `iris_movement`, `gaussian` |
| `texSample2D(t, <vec4>)` | no matching overload | `bokeh_blur` |
| `(a < b) * 6.0` | `*` does not operate on bool | `gaussian` |
| `max(0, <vec3>)` | no matching overload | `test_shader` |
| `vec2 k[N] = { … };` | syntax error at `{` | `bokeh_blur` |

**Decision: lexical relaxation, not a front end.** The passes rewrite only when
they can identify the operand types from declarations in the same file, and
leave the source untouched otherwise — so an expression the module does not
understand compiles exactly as it did before, and a wrong guess is impossible
rather than merely unlikely. That conservatism is what makes a lexical approach
defensible here, and it is also forced: the source still carries `#if`
directives, since §6 leaves conditionals to the driver, so there is no single
well-formed parse tree to work from. `glsl-lang` and `naga` were both considered
and both want a directive-free unit.

Type information comes from one sweep for `<builtin type> <name>` pairs — which
covers globals, locals, parameters and struct members alike — plus `#define`
constants, function signatures, and a small table of builtin return types. Only
`Simple_Audio_Bars` still fails, and only on `uint`/`int` mixing; it is an audio
visualiser and we feed no audio, so it is left alone.

**macOS has no true surfaceless or pbuffer GL surface.** `glutin`'s CGL backend
(checked against 0.31 and 0.32) rejects both outright —
`create_pbuffer_surface` returns `NotSupported`. The only surface CGL offers
is a window's. The renderer therefore builds one real, one-pixel `NSWindow`
via `objc2-app-kit`, never orders it onto the screen, and attaches the GL
context to its content view — see `render/gpu.rs`.

---

## 7. Seamless looping — superseded

**No longer a goal.** This section was about cutting a finite exported video
at a seamless loop point, for some other wallpaper app to play on repeat. Now
that this app plays the Scene live off wall-clock `g_Time` (§14), there is no
finite file to loop at all — periodic effects just keep going. Kept for the
record in case a bounded export is ever wanted again (e.g. sharing a clip).

Effects are periodic in `g_Time`, but with unrelated periods — in the sample,
`shake` at `speed 1.0`, `foliagesway` at `speeduv 1.5`, `lightshafts` at
`rayspeed 0.15`. A naive common multiple could be minutes long, and any
noise-driven effect may not be periodic at all.

Three strategies, in order of preference:

1. **Empirical loop search.** Render a candidate window, then find the frame
   whose image best matches frame 0 (perceptual hash or MSE), and cut there.
   Search within a target range (say 5–30 s) and pick the best residual.
2. **Analytic period.** Derive each effect's period from its speed uniforms and
   rationalise to a bounded common multiple. Exact when it works; fails on
   noise-based effects.
3. **Crossfade fallback.** If the best residual is above a threshold, blend the
   tail into the head over a short window. Always works, slightly soft at the
   seam.

**Plan: (1) with (3) as automatic fallback**, and (2) later as an optimisation
where it applies. Report the achieved seam error to the user so a bad loop is
visible rather than silent.

---

## 8. Export

### Still PNG
Render one frame through the full pipeline at a chosen `--time` (default 0) and
write it. Also keep a "base image only, no effects" mode — that already works
today and is the fastest path to a usable wallpaper.

### Video export to a file — superseded
**No longer a goal**, for the same reason as §7: nothing needs to hand a
finished mp4 to some other player once this app plays the wallpaper itself.
`export::video`'s encoder machinery isn't going away — the live desktop
backend still wants it to decode and loop a Video-type wallpaper's source
file (§14.2) — but *producing a new video file as the deliverable* is not a
phase anymore. Kept for the record below.

Defaults aimed at maximum compatibility with ordinary wallpaper apps: H.264,
`yuv420p`, `+faststart` mp4. Offer HEVC/VP9 as opt-in.

**Render at output resolution, not source.** The sample's base texture is
7680×4320; rendering a 1080p video at 8K and downscaling wastes enormous GPU
memory and time. Scale the base image once at load.

Audio: the sample carries three FLAC tracks. Most wallpaper apps ignore audio —
make muxing opt-in, off by default.

---

## 9. Web wallpapers

1. Detect `type: Web`; find the entry HTML in the unpacked package.
2. Serve the unpacked directory over a local static HTTP server (`tiny_http` or
   `axum`) — `file://` will trip CORS for many wallpapers.
3. Drive headless Chrome via CDP (`chromiumoxide`) at the target resolution.
4. Capture frames. Prefer **`Emulation.setVirtualTimePolicy`** over wall-clock
   screenshots: it advances page time deterministically, giving exact frame
   pacing and reproducible output — the same property that makes the Scene
   renderer tractable.
5. Hand frames to the live desktop backend (§14) the same way the Scene
   renderer does, rather than to an export/encode stage (§7, §8 — superseded).

Expect to stub WE's JS API (`window.wallpaperPropertyListener`,
`wallpaperRegisterAudioListener`, and friends) or wallpapers will error on load.

---

## 10. CLI shape

Built:

```
wallpaper-engine info     <wallpaper-dir>
wallpaper-engine export   <wallpaper-dir> [OPTIONS]
wallpaper-engine shaders  <wallpaper-dir> [OPTIONS]
wallpaper-engine simulate <wallpaper-dir>   # Scene only; opens a live window

  --out DIR              parent directory for the wallpaper's own output
                         folder (default: the current directory), named
                         after its title
  --png-only             skip the video; write only the requested --frame(s)
  --frame SECS           export a still at this timestamp; repeatable
  --resolution WxH       default: keep the source resolution
  --fps N                default: keep the source rate
  --duration SECS        trim to this length
  --audio                keep the audio track (default: drop it)
```

`unpack` and `tex` remain as debugging subcommands, alongside `shaders`.

**Deviation from the original sketch:** the default resolution keeps the source
rather than forcing 1920 wide. For a video wallpaper that default would
re-encode a finished 4K file down to 1080p unasked, which is both lossy and
slow; the scene renderer will want the opposite default and can set it itself.

**Deviation, second round:** a video export writes only the video by default —
no automatic still. The wallpaper already contains a finished loop, so there
is no single frame more canonical than any other to default to; `--frame` (or
several) opts in explicitly. Each wallpaper's output also lands in its own
folder named after its title rather than stem-named files in a flat directory,
since a `--out` used across many exports would otherwise collide or blur
together. Titles carry `/`, `|` and other filesystem-hostile punctuation, so
the folder name is sanitized, not the raw string.

**Deviation, third round:** for a Scene wallpaper that runs the effect chain
(§4.5), `--frame` now means something different than for Video — it is the
`g_Time` the chain renders at (first value if several are given), not a seek
into an existing file. `--png-only` stays moot there: Scene has never had a
video path.

Still to add:

```
  --no-effects           export the base image only, skip the renderer
  --we-assets PATH       read real common*.h from a WE install (fallback)
```

---

## 11. Phasing

Each phase ships something independently useful.

| Phase | Deliverable | Risk |
|---|---|---|
| ~~**1. Type routing + Video passthrough**~~ | **Done.** `project.json` parsing; Video wallpapers export with no rendering at all | Low |
| ~~**2. Scene model + still export**~~ | **Done.** serde types for scene/effect/material; correct-resolution base PNG | Low |
| ~~**3. Shader pipeline**~~ | **Done** for the one-image-layer shape: preprocessor, shim, headless glow/glutin context, the full effect chain per `--frame` timestamp. Verified deterministic (byte-identical re-renders) and animated (distinct frames at t=0 vs t=30 on `scene_example1`). Multi-layer/particle scenes (`scene_example2`) still fall back to the effect-free composite | **High**, now landed |
| ~~**3.5. Live simulator**~~ | **Done.** `simulate` plays the same compiled chain in a real window at wall-clock speed instead of one still per run (§4.6). Only `g_Time` is live — no mouse/audio/daytime input yet | Low |
| **3.6. Multi-layer scenes** | Per-layer effect chains, keyframe animation, particle systems, blend modes (§4.7) — the shape `scene_example2` needs, still falls back to the static composite today | High |
| **4. Web capture** | Local server, headless Chrome, virtual-time frames | Medium |
| **5. Desktop wallpaper backend + Tauri front end** | Borderless desktop-level window, multi-monitor, live control channel, library UI (§14) | High |

Scene video export (rendering a `g_Time` sweep to an mp4 via `export::video`,
for some *other* wallpaper app to play) is dropped: the goal is now this app
playing the wallpaper live, so a Scene never needs to be baked to a video file
at all. `export::video` itself is not going away — phase 5 still wants it for
Video-type wallpapers' live desktop loop (§14.2) — only the "export a finished
video" phase built around it is gone.

Phase 3 is the project. Phases 1–2 are worth doing first anyway because they
are cheap, they cover a lot of real wallpapers, and they build the scene model
that phase 3 consumes.

---

## 12. Verification strategy

The container work was validated by diffing against an independent
implementation and confirming exact byte accounting. The renderer needs an
equivalent standard, because "looks about right" is not a test.

- **Shim correctness:** golden-image diff, our shim vs. real WE headers.
- **Parameter binding:** assert the parsed uniform↔material table against the
  shader annotations, per effect.
- **`mul` convention:** a dedicated test — this is a silent-corruption bug.
- **Determinism:** rendering frame `n` twice must be byte-identical.
- **Loop quality:** report seam error as a number, not a vibe.
- **Regression corpus:** a set of wallpapers with known-good output frames.

---

## 13. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| Shim semantics diverge from real `common.h` | Wrong output, subtly | Golden-image diff; `--we-assets` escape hatch |
| naga GLSL frontend too limited | Blocks phase 3 | Fall back to glslang→SPIR-V |
| Effect long tail (WE ships ~40 built-ins) | Coverage gaps | Data-driven binding means most work unchanged; fill helpers on demand |
| Non-periodic effects | Visible loop seam | Crossfade fallback, report seam error |
| Scene features beyond image+effects (particles, puppet warp, audio-reactive, 3D models) | Out of reach | Detect and report clearly rather than rendering something wrong |
| 8K textures | GPU memory | Downscale at load to output resolution |
| WE JS API surface for Web | Web wallpapers fail to load | Stub incrementally against real wallpapers |

**Open:** the corpus is six wallpapers — two scene, two video, two web. Enough
to have built and checked the router against, not enough for phase 3: only two
are scenes, and one of those needs particle systems (see below). More scene
wallpapers covering varied effects are needed before phase 3 can be called done.

**Scope gap found in the corpus.** The second scene sample is not the shape
this plan was written against. It has 12 objects — nine layered images with
per-object effects, one carrying 13 — plus two particle systems (`Fireflies`,
`Ember`) and an audio object. Particle systems are visible motion, so skipping
them changes the output rather than merely simplifying it. Decide before phase 3
whether they are in scope or a documented gap.

**Oversized web wallpapers are declined, not attempted.** One sample is 3.1 GB
with `"oversized": true`: 167 background stills, 15 music videos, 689 MB of
audio, driven by a `data.json` playlist. There is no canonical frame to capture
— a recording would catch whatever track happened to be playing. The router
rejects these with a clear message instead of producing something wrong.

**Legal:** WE's shader headers and engine assets are not redistributable. The
shim is our own work; `--we-assets` reads the user's own install. Output is the
user's own purchased content, converted for personal use.

---

## 14. Desktop wallpaper backend + Tauri front end

**New goal, layered on top of the exporter.** Rather than exporting to a file
and importing that into some other wallpaper app, run this engine itself as
the wallpaper: keep `simulate`'s GL renderer, but instead of a titled window,
attach it to a borderless window pinned at `kCGDesktopWindowLevel` (macOS'
own desktop-picture layer, one below desktop icons), so a Scene or Video
wallpaper plays live, continuously, as the actual desktop background — the
way Wallpaper Engine itself works on Windows. A Tauri app then becomes the
control surface: pick a wallpaper from a library, tweak its live parameters,
assign wallpapers per monitor.

### 14.1 Two halves

- **Backend (Rust, extends this crate).** A long-running process that owns
  one GL-rendered desktop window per monitor and keeps it in sync with
  whatever the user has selected. Reuses `scene::render`, `render::pass`,
  `render::gpu`, and `simulate.rs`'s redraw loop almost unchanged — the work
  is new window plumbing, not new rendering.
- **Front end (Tauri).** A webview app for the parts WE's own UI covers:
  library browsing with thumbnails, a settings panel per wallpaper (the same
  tweakables `simulate`'s `egui` panel already exposes, just in a web UI
  instead), per-monitor assignment, autostart.

### 14.2 What's missing to get there

Everything below is new; `simulate.rs` today only proves the GL rendering
loop works, not any of the "act like a real wallpaper" behaviour:

| Gap | Why `simulate.rs` doesn't already cover it |
|---|---|
| **Desktop-level window** | `open_window` makes an ordinary titled, focusable `NSWindow` at the default level. Needs a borderless style mask, no title bar, window level set to `kCGDesktopWindowLevel` (or `kCGDesktopIconWindowLevel - 1` to sit under icons), `ignoresMouseEvents` so desktop clicks pass through, and `.canJoinAllSpaces`/`.stationary` collection behaviour so it doesn't scroll away with Spaces or get picked up by Mission Control/Cmd+Tab. |
| **Runs with no Dock icon** | Needs `LSUIElement` (agent app) so it doesn't show a Dock icon or app-switcher entry — it's a background service, not an app window. |
| **Multi-monitor** | `simulate` opens exactly one window on the default display. A real backend needs one window per `NSScreen`, independently sized, reacting to displays connecting/disconnecting/resolution changes. |
| **Live Video playback, not just Scene** | The GL desktop window only has a renderer for Scene's effect chain. Video-type wallpapers have no live-playback path at all yet — only `export::video`'s file-to-file transcode. Playing one as a live desktop background needs a decode-and-blit loop (e.g. `ffmpeg-next` decoding into a texture) alongside `EffectChain::render`. |
| **A control channel** | The CLI today is one-shot: run, do one thing, exit. A background renderer the Tauri UI can drive live (switch wallpaper, drag a slider, pause) needs a persistent process with some IPC — a local socket, or Tauri's own command bridge — that doesn't exist yet. |
| **Persisted state** | Which wallpaper is active per monitor, per-wallpaper tweakable overrides (the `egui` panel's values currently reset every run), and any autostart-at-login flag all need saving and reloading — no config file exists yet. |
| **A library view** | `info` reports on one wallpaper at a time. Nothing today scans a folder of many wallpapers and produces a browsable list with a thumbnail, title, and type — the data the front end's picker needs. |
| **Idle/foreground courtesy** | WE pauses rendering when a window is fullscreen over it or the machine is on battery, to avoid burning GPU for nothing. Nothing like that exists — today's renderer always runs full tilt. Not blocking for a first version, but expected of anything calling itself a wallpaper engine. |
| **The Tauri app itself** | No front-end project exists yet — a from-scratch scaffold, wired to the backend however the control channel above ends up shaped. |

### 14.3 Open architecture question (decide before starting)

Tauri's own backend *is* Rust, which raises a real fork: does the GL desktop
renderer run **inside the Tauri process** (one binary, no IPC, but the
renderer's lifetime is tied to the UI app running), or as a **separate
background daemon** the Tauri UI merely talks to over a socket (renderer
survives the UI quitting — matching how WE itself behaves: quitting its
control UI leaves the wallpaper running)? This changes the shape of §14.2's
control-channel item and is worth deciding deliberately before writing code,
not defaulting into one.

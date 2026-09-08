# Wallpaper Engine → PNG / Video Exporter

**Goal.** Turn a Wallpaper Engine wallpaper into a still PNG and a seamlessly
looping video that any ordinary wallpaper app can consume, with the animated
effects baked in.

**Status.** Container layer, type routing and the Video pipeline are built and
verified against a six-wallpaper corpus. The Scene renderer now runs the real
GPU effect chain for the common one-image-plus-effects shape (§4); richer
scenes (multiple image layers, particles) still fall back to the effect-free
composite. A `simulate` subcommand plays that same chain live in a real window
at wall-clock speed, ahead of video export existing (§10, §11). Web capture is
still to build.

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
and the export stage (still PNG, loop detection, video encode) is shared by all
three. Build the shared export stage once.

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

**macOS has no true surfaceless or pbuffer GL surface.** `glutin`'s CGL backend
(checked against 0.31 and 0.32) rejects both outright —
`create_pbuffer_surface` returns `NotSupported`. The only surface CGL offers
is a window's. The renderer therefore builds one real, one-pixel `NSWindow`
via `objc2-app-kit`, never orders it onto the screen, and attaches the GL
context to its content view — see `render/gpu.rs`.

---

## 7. Seamless looping

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

### Video
Shell out to **ffmpeg** via `std::process`. Deliberately not `libav` bindings:
ffmpeg-as-a-subprocess is the battle-tested, portable, low-maintenance choice,
and it keeps the build free of native library pain. `ffmpeg-sidecar` is worth
considering to locate/fetch the binary.

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
5. Hand frames to the shared loop-detection and encode stage.

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
  --auto-loop            detect the best loop point (default, once built)
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
| **4. Animation + video** | `g_Time` sweep across a whole video's worth of frames, loop detection, feeding the chain's output into `export::video` instead of one still | Medium |
| **5. Web capture** | Local server, headless Chrome, virtual-time frames | Medium |

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

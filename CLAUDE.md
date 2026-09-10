# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build --release          # always: one scene_example2 frame is 1.5s release vs 19.6s debug
cargo test                     # 114 tests, all unit tests inside src/
cargo test <substring>         # single test, e.g. cargo test blend_mode_comes_from_the_first_pass
cargo clippy --all-targets     # must be clean: clippy::pedantic is deny, as are unwrap_used/panic/todo/exit
```

`cargo clippy` is not advisory here — `[lints.clippy]` in `Cargo.toml` denies `pedantic`, so a
pedantic lint is a hard error. Where a lint is genuinely wrong for the code (bounded pixel/mesh
arithmetic, for instance), the codebase scopes it off with `#[expect(..., reason = "...")]` rather
than loosening the crate lint level. Follow that.

**Do not run `cargo fmt`.** There is no `rustfmt.toml` and the codebase is not rustfmt-clean with
default settings — `cargo fmt` would reflow ~3200 lines. Match the surrounding style (wide lines,
up to ~120 chars) by hand.

Build needs the ffmpeg development libraries (`brew install ffmpeg`); `ffmpeg-next`'s major version
tracks ffmpeg's own, so 9.x means ffmpeg 9.

Rendering is plain GL 3.3 core throughout — no Metal, no compute, nothing above 3.3 (see
`render/particles.rs` on why `glDrawArraysInstancedBaseInstance` is out) — so the renderer itself is
portable. Only the *context* is per-platform, and only for the headless `export` path
(`render/gpu.rs`): CGL rejects surfaceless and pbuffer surfaces, so macOS anchors to a never-shown
one-pixel `NSWindow`, while everything else takes an EGL pbuffer, with the AppKit crates scoped to
macOS in `Cargo.toml`. `simulate` takes its context from a real window via `glutin-winit` and is
portable already. **The non-macOS backend has not been compiled on Linux** — it is written against
glutin 0.32's API but only macOS has been built and run here.

### Debugging subcommands

The CLI carries the tools each layer was built with; reach for them before adding new scaffolding.

```bash
cargo run --release -- info    papers/scene_example2      # what a wallpaper is, and what we can't render
cargo run --release -- unpack  papers/scene_example2/scene.pkg -o /tmp/s2
cargo run --release -- tex     /tmp/s2/materials -o /tmp/tex   # .tex -> PNG, incl. the effect masks
cargo run --release -- shaders papers/scene_example2 -o /tmp/glsl  # preprocess only, no GPU needed
cargo run --release -- simulate papers/scene_example2      # live window, egui parameter sliders
```

`simulate` has a headless dump hook, which is the fastest way to verify a rendering change:

```bash
SIMULATE_DUMP=/tmp/frame.png SIMULATE_TIME=2.6 cargo run --release -- simulate papers/scene_example2
```

It renders the composited frame at a pinned `g_Time` through the *live* pipeline, writes it, and
exits. Its startup line also reports the tweakable-parameter count and any layer whose effect chain
failed to compile — a chain silently dropping is the failure mode this catches.

`export` on a scene writes only `still.png` and honours only the *first* `--frame`; the repeated
`--frame` and video options apply to video wallpapers.

## Architecture

### Routing

`project.json`'s `type` field selects the whole downstream pipeline (`project.rs`, `Kind`). Case is
not normalized by Wallpaper Engine, so parsing lowercases first. Video wallpapers already contain a
finished loop and are packaged directly; Scene wallpapers are rendered; Web and Application are not
built.

### The scene resolution chain

Everything a scene needs is inside `scene.pkg`, so `scene/` never touches the filesystem beyond
opening that archive (`pkg.rs`, a flat path→(offset,length) table over one data blob).

```
scene.json  →  models/*.json  →  materials/*.json  →  materials/*.tex
 (objects,        (material,        (shader, blending,     (tex.rs: DXT/LZ4/
  layer order)     puppet?)          texture slots)         raw payloads)
```

`scene/model.rs` is the serde layer for all of it. It deliberately parses fields the renderer does
not use yet — discovering their shape is most of the work — hence the module-wide `allow(dead_code)`.
Numbers arrive both as JSON numbers and as `"3840.00000 2160.00000 0.00000"` strings, sometimes for
the same concept, so `Vec3` has a custom deserializer.

### Two render paths over one preparation phase

`scene/compose.rs` owns the time-independent groundwork and splits into two halves on purpose:

- `prepare_static` — decode every layer texture, parse every puppet mesh, resolve every particle
  placement. Runs **once**.
- `animate` — turn that into one frame: `warp_frame` re-skins each puppet, particles re-simulate.
  Runs **per frame**.

`prepare` is the two back to back, for one-shot callers. On top of that:

- `scene/render.rs::render_frame` — the GPU path. Per visible image layer: run *that layer's own*
  effect chain over its texture, then alpha-composite the processed layers in scene order honouring
  each layer's blend mode. Used by `export`.
- `compose::render` — flattens layers with no effects at all. The fallback, and what reports
  `omissions` (the list of things a still cannot represent).
- `simulate.rs` — the live window. Calls `prepare_static` + `prepare_effect_chain` once when the
  window opens, then per frame re-skins puppets, re-simulates particles, uploads the fresh images,
  re-runs each layer's compiled chain, and composites on the GPU.

An `EffectChain` is compiled once and redrawn at any `g_Time` without touching the archive, the
preprocessor or the GL compiler again — that split is why the live simulator is viable.

### The shader pipeline

Packaged shaders `#include "common.h"`, which ships with the Wallpaper Engine *program*, not with
any wallpaper — and is not ours to redistribute. So:

- `shader/shim.rs` — our own replacement headers, written from measured usage (exactly fourteen
  identifiers across the corpus are called but neither declared locally nor part of GLSL). They live
  beside the module as real `.glsl` files. `--we-assets` on the `shaders` subcommand reads a real
  install instead, for verification.
- `shader/preprocess.rs` — expands includes (`glsl-include`), injects the pass's chosen combo values
  as `#define`s, and rewrites GLSL 1.20-era spellings. Targets `#version 330 core`, not newer,
  because `sample` became reserved in GLSL 4.00 and five corpus shaders use it as a local.
  Conditionals are left to the driver's own compiler.
- `shader/annotations.rs` — parses the JSON metadata in shader comments
  (`uniform float g_Speed; // {"material":"speed","range":[0.01,50]}`). This drives everything:
  uniform↔material binding, which combos exist and their defaults, and which parameters get an egui
  slider for free.
- `shader/bind.rs` — resolves material values onto uniforms, and derives combo values from which
  texture slots are bound.

`mul` puts the matrix on the left (HLSL convention). Getting this wrong is silent corruption; there
is a dedicated test.

### Format conventions that are easy to get wrong

These were each a real bug. They are not guessable from the file format alone — the corpus is the
evidence.

- **Effect texture slots are indexed by slot number.** `EffectPass::textures[1]` is `g_Texture1`;
  entry 0 is a null placeholder for `g_Texture0`, which is always the previous pass. It is *not*
  offset by one. Two passes pin this down: `shake` writes `util/white` at index 2 and
  `godrays_downsample2` writes `util/clouds_256` at index 2, each the declared default of that
  shader's own `g_Texture2`.
- **A sampler's annotation can name a combo** (`{"mode":"opacitymask","combo":"MASK"}`) that must be
  1 exactly when that slot is bound. Leaving a mask unbound does not merely skip the mask — the
  shader falls back to `mask = 1.0` and applies the effect to the *whole layer*. One layer in
  `scene_example2` stacks twelve masked `waterwaves` passes, so unmasked they warp the whole
  character.
- **`_rt_*` names are runtime render targets, not files in the package.** Reading one as
  `materials/<name>.tex` fails and drops that layer's entire chain.
  `_rt_imageLayerComposite_<id>_<slot>` is the layer's pre-effect image, which a `godrays`/`shine`
  `*_combine` pass blends its rays back over; it binds to the chain's base image and is kept
  symbolic (`PassTexture::LayerBase`) because puppet and particle layers re-upload that image every
  frame. Unknown `_rt_*` names fall back to the shader's annotation default rather than failing.
- **Combo defaults must be merged across both stages.** A combo's `[COMBO]` default is typically
  declared only in the fragment shader but governs the vertex shader too; defaulting it
  independently per stage makes the program fail to link.
- **Puppet local transforms are `translate · rotate · scale`** (`scene/puppet.rs`). Pose translations
  are local to the parent and accumulate through the hierarchy; verified by checking that each bone's
  accumulated world position lands on the centroid of the vertices it weights. Clip frame 0 is the
  bind pose, so `t=0` must skin to an exact identity. The scale channel is not decorative — eye-blink
  clips squash `scale_y` to ~0 and barely touch translation or rotation.
- **`g_TextureNResolution` is `(w, h, w, h)`.** The two halves differ only for atlas-packed
  textures, which nothing here produces.

### Verification

`papers/` holds a six-wallpaper corpus (two each of scene/video/web) and each wallpaper ships its own
`preview.gif`/`preview.jpg` — the ground truth for what a render *should* look like. The established
standard is that "looks about right" is not a test (plan.md §12): prefer a number. Useful moves,
all used before:

- dump frames with `SIMULATE_DUMP` and compare regions against the preview with
  `ffmpeg ... -filter_complex "[0]crop=...[a];[1]crop=...[b];[a][b]psnr"` — motion in a region that
  should be still is measurable as a PSNR delta;
- decode the effect masks with the `tex` subcommand to see what area an effect was *meant* to touch;
- watch the `simulate` startup report for chains that stopped compiling.

The two `#[ignore]`d tests in `render/capture.rs` need AppKit's real process main thread, which
`cargo test`'s harness never provides (it runs every test on a spawned worker), so
`cargo test -- --ignored` fails with "the renderer must be set up on the main thread" — that is
expected, not a regression. They cannot be un-ignored without a lib target to host a
`harness = false` integration test; they are verified manually.

## Design notes

`plan.md` is the living design document and the place decisions and open questions are recorded —
read the relevant section before reshaping a subsystem, and update it when a decision changes. Note
its goal has shifted: the deliverable is now playing a wallpaper live as the desktop background
(§14), with still-PNG export and the container/shader work as building blocks under it; exporting a
finished mp4 for some other app to play is superseded and dropped (§7, §8).

Commits are single-purpose with an imperative subject line and a body explaining *why*, including the
corpus evidence for a format decision. Work goes straight to `master`; this repo does not use feature
branches.

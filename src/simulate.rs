//! A live window that plays a whole scene in real time.
//!
//! Unlike `export`, which renders one frame with a throwaway headless context,
//! this attaches a GL context to a visible window and, every frame at
//! wall-clock `g_Time`:
//!
//! - re-skins each puppet-warp layer and re-simulates each particle system on
//!   the CPU, uploading the fresh image;
//! - runs every image layer's own compiled effect chain over its texture;
//! - composites the processed layers in z-order, honouring each layer's blend
//!   mode, into one canvas-sized target that is then blitted to the window.
//!
//! The decode/compile/parse groundwork (`compose::prepare_static`,
//! `render::prepare_effect_chain`) happens once when the window opens; only the
//! per-frame work above repeats. Nothing else Wallpaper Engine feeds a shader —
//! mouse position, system audio, time of day, now-playing media — is wired up
//! yet; see plan.md. An `egui` overlay lets you drag any range-annotated shader
//! parameter away from the wallpaper's own preset, live.

use crate::pkg::Archive;
use crate::render::{capture, pass};
use crate::scene::compose::{self, StaticItem};
use crate::scene::model::{self, Blend, Scene};
use crate::scene::particle;
use crate::scene::render::{self, EffectChain};
use crate::shader::shim;
use anyhow::{Context, Result, anyhow};
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext, Version};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use image::RgbaImage;
use raw_window_handle::HasWindowHandle;
use std::collections::HashMap;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Open a window and play `scene` in it until it is closed.
pub fn run(archive: &mut Archive, scene: &Scene, title: &str) -> Result<()> {
    let static_scene = compose::prepare_static(archive, scene, None)?;

    // What is left unsimulated depends on which layer chains compile, which
    // needs the window's GL context — so the report is printed from
    // `open_window`, not here.
    let event_loop = EventLoop::new().context("opening a window event loop")?;
    let mut app = App {
        title: title.to_string(),
        archive,
        static_scene,
        headers: shim::headers(),
        start: Instant::now(),
        state: None,
    };
    event_loop.run_app(&mut app).context("running the simulator window")
}

struct App<'a> {
    title: String,
    archive: &'a mut Archive,
    static_scene: compose::StaticScene<'a>,
    headers: HashMap<String, String>,
    start: Instant,
    state: Option<State>,
}

/// Which per-frame work a live layer needs before it is composited.
enum LiveKind {
    /// A plain image layer: texture uploaded once, never changes.
    Image,
    /// A puppet-warp layer: `static_scene.items[usize]` is re-skinned each frame.
    Puppet(usize),
    /// A particle system, re-simulated each frame into `placement`'s canvas —
    /// which may be a fraction of the real one (soft glows survive a bilinear
    /// upscale, and a full-res tiny-skia raster every frame is what pins a big
    /// scene to single digits). The compositor stretches it back to full size.
    Particle { placement: particle::Placement, preset_path: String },
}

/// Simulate particles into a canvas this fraction of the real one when the real
/// one is large. The soft additive sprites these presets use survive a bilinear
/// upscale, and the raster + per-frame texture upload both scale with the pixel
/// count, so a big scene drops to ~0.4 (≈1/6 the pixels).
fn particle_sim_scale(canvas: (u32, u32)) -> f32 {
    match canvas.0.max(canvas.1) {
        0..=1999 => 1.0,
        2000..=3199 => 0.5,
        _ => 0.4,
    }
}

/// `place` rescaled so a `scale`-of-canvas simulation lands in the same spots.
fn scaled_placement(place: &particle::Placement, scale: f32) -> particle::Placement {
    let down = |side: u32| -> u32 {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a wallpaper canvas side is a few thousand px; scaled down and rounded it stays a small positive integer"
        )]
        let scaled = (f64::from(side) * f64::from(scale)).round().max(1.0) as u32;
        scaled
    };
    particle::Placement {
        origin_px: place.origin_px * scale,
        px_per_unit: place.px_per_unit * scale,
        canvas_px: (down(place.canvas_px.0), down(place.canvas_px.1)),
        ..place.clone()
    }
}

/// The texture(s) behind a layer: one uploaded once for a static image layer,
/// or a two-deep ring for a puppet / particle layer whose image is re-uploaded
/// every frame — alternating targets so a frame's upload never stalls behind
/// the previous frame still sampling the same texture.
struct LayerTextures {
    ring: Vec<glow::Texture>,
    next: usize,
    size: (u32, u32),
}

impl LayerTextures {
    fn once(gl: &glow::Context, image: &RgbaImage) -> Result<Self> {
        Ok(LayerTextures { ring: vec![pass::upload_texture(gl, image)?], next: 0, size: image.dimensions() })
    }

    fn ring(gl: &glow::Context, image: &RgbaImage) -> Result<Self> {
        Ok(LayerTextures {
            ring: vec![pass::upload_texture(gl, image)?, pass::upload_texture(gl, image)?],
            next: 0,
            size: image.dimensions(),
        })
    }

    fn current(&self) -> glow::Texture {
        self.ring[self.next]
    }

    /// Advance to the next texture in the ring and upload `image` into it,
    /// reallocating the whole ring only if the dimensions changed.
    fn refresh(&mut self, gl: &glow::Context, image: &RgbaImage) -> Result<()> {
        self.next = (self.next + 1) % self.ring.len();
        if image.dimensions() == self.size {
            pass::update_texture(gl, self.ring[self.next], image);
        } else {
            for texture in self.ring.drain(..) {
                pass::delete_texture(gl, texture);
            }
            self.ring = vec![pass::upload_texture(gl, image)?, pass::upload_texture(gl, image)?];
            self.next = 0;
            self.size = image.dimensions();
        }
        Ok(())
    }
}

/// One scene layer, in z-order, with everything the redraw loop needs.
struct LiveLayer {
    kind: LiveKind,
    /// The object's name, for the tweakable panel.
    name: String,
    additive: bool,
    /// The layer's own effect chain, compiled once. `None` if it has no effects.
    chain: Option<EffectChain>,
    /// This chain's slice of `State::tweak_values`, one entry per tweakable.
    tweaks: Range<usize>,
    /// The current image: a single texture for `Image`, a re-uploaded ring for
    /// `Puppet` / `Particle`.
    textures: LayerTextures,
    /// Placement on the canvas — `(left, top, width, height)` in pixels,
    /// measured from the top-left. Re-derived each frame for `Puppet`.
    rect: (i32, i32, i32, i32),
}

/// Everything that only exists once the window itself does.
struct State {
    window: Window,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    gl: Arc<glow::Context>,
    display_quad: pass::Quad,
    blit: pass::BlitProgram,
    compositor: pass::LayerCompositor,
    /// Canvas-sized accumulator the layers are stacked into each frame.
    composite: pass::Target,
    /// The scene's own pixel dimensions — every layer and the composite match
    /// this, so the window letterboxes to it rather than stretching.
    content_size: (u32, u32),
    background: [f32; 4],
    layers: Vec<LiveLayer>,
    egui: egui_glow::winit::EguiGlow,
    /// One live value per tweakable across every chain, concatenated in layer
    /// order; each `LiveLayer::tweaks` indexes its own span.
    tweak_values: Vec<f32>,
    /// Rolling frame counter for the once-a-second FPS line.
    frames_since: (Instant, u32),
}

/// `(left, top, width, height)` in canvas pixels for an image placed with its
/// top-left at `(left, top)`.
#[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "wallpaper canvas coords and sizes are nowhere near i32::MAX")]
fn rect_of(left: i64, top: i64, image: &RgbaImage) -> (i32, i32, i32, i32) {
    (left as i32, top as i32, image.width() as i32, image.height() as i32)
}

impl App<'_> {
    fn open_window(&mut self, event_loop: &ActiveEventLoop) -> Result<State> {
        let (width, height) = (self.static_scene.width, self.static_scene.height);
        let attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::PhysicalSize::new(width, height));

        let template = ConfigTemplateBuilder::new().with_alpha_size(8);
        let (window, config) = DisplayBuilder::new()
            .with_window_attributes(Some(attributes))
            .build(event_loop, template, |mut configs| {
                configs.next().expect("the platform reports at least one GL config")
            })
            .map_err(|error| anyhow!("opening the window's GL display: {error}"))?;
        let window = window.context("glutin-winit did not create a window")?;

        let display = config.display();
        let raw_window_handle = window.window_handle().ok().map(|handle| handle.as_raw());
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .build(raw_window_handle);
        // Safety: `config` and `raw_window_handle` both come from the window
        // and display created just above, which outlive this call.
        let not_current = unsafe { display.create_context(&config, &context_attributes) }
            .context("creating a GL 3.3 core context")?;

        let surface_attributes = window
            .build_surface_attributes(SurfaceAttributesBuilder::<WindowSurface>::new())
            .map_err(|error| anyhow!("building the window's surface attributes: {error}"))?;
        // Safety: the surface attributes were just built from this same window.
        let surface = unsafe { display.create_window_surface(&config, &surface_attributes) }
            .context("attaching the GL context to the window")?;

        let context = not_current.make_current(&surface).context("making the GL context current")?;
        let one = NonZeroU32::new(1).context("1 is non-zero")?;
        surface.set_swap_interval(&context, SwapInterval::Wait(one)).context("enabling vsync")?;

        let gl = unsafe {
            glow::Context::from_loader_function(|name| {
                let name = CString::new(name).unwrap_or_default();
                display.get_proc_address(&name).cast()
            })
        };
        let gl = Arc::new(gl);

        let display_quad = pass::build_display_quad(&gl)?;
        let blit = pass::compile_blit_program(&gl)?;
        let compositor = pass::compile_layer_compositor(&gl)?;
        let composite = pass::Target::new(&gl, width, height).context("allocating the composite target")?;

        let Built { layers, tweak_values, omissions } = self.build_layers(&gl)?;
        for note in &omissions {
            println!("  not simulated: {note}");
        }
        report_tweakables(&layers);

        let egui = egui_glow::winit::EguiGlow::new(event_loop, Arc::clone(&gl), None, None, true);

        window.request_redraw();
        Ok(State {
            window,
            surface,
            context,
            gl,
            display_quad,
            blit,
            compositor,
            composite,
            content_size: (width, height),
            background: normalized_rgba(self.static_scene.background),
            layers,
            egui,
            tweak_values,
            frames_since: (Instant::now(), 0),
        })
    }

    /// Build one `LiveLayer` per static item: take its t=0 image, upload it,
    /// and — if the layer carries effects — compile its chain over that image.
    /// A layer whose chain will not compile (an unimplemented effect helper, a
    /// missing engine texture) still renders, just unprocessed, exactly as the
    /// still exporter degrades it.
    fn build_layers(&mut self, gl: &glow::Context) -> Result<Built> {
        let App { archive, static_scene, headers, .. } = self;
        let mut layers = Vec::with_capacity(static_scene.items.len());
        let mut tweak_values = Vec::new();

        // Start from what the static pass could not settle, then drop every
        // "N effect(s) not applied" note — a chain either runs below or re-adds
        // its own "skipped" note, the same handoff `render_frame` does.
        let mut omissions: Vec<String> = static_scene
            .omissions
            .iter()
            .filter(|note| !note.ends_with("effect(s) not applied"))
            .cloned()
            .collect();

        let sim_scale = particle_sim_scale((static_scene.width, static_scene.height));
        #[expect(clippy::cast_possible_wrap, reason = "wallpaper canvas dims are nowhere near i32::MAX")]
        let full_rect = (0, 0, static_scene.width as i32, static_scene.height as i32);

        for (index, item) in static_scene.items.iter().enumerate() {
            let (image, rect, blend, kind, object) = match item {
                StaticItem::Image(layer) => {
                    if let Some(error) = &layer.warp_error {
                        swap_note(
                            &mut omissions,
                            &format!("{}: keyframe animation not applied", model::label(layer.object)),
                            format!("{}: puppet warp skipped ({error})", model::label(layer.object)),
                        );
                    }
                    (
                        layer.image.clone(),
                        rect_of(layer.left, layer.top, &layer.image),
                        layer.blend,
                        LiveKind::Image,
                        layer.object,
                    )
                }
                StaticItem::Puppet(puppet) => {
                    omissions.retain(|note| {
                        *note != format!("{}: keyframe animation not applied", model::label(puppet.object))
                    });
                    let (image, left, top) = compose::warp_frame(puppet, 0.0);
                    let rect = rect_of(left, top, &image);
                    (image, rect, puppet.blend, LiveKind::Puppet(index), puppet.object)
                }
                StaticItem::Particle(system) => {
                    let placement = scaled_placement(&system.place, sim_scale);
                    let image = particle::render_system(archive, &system.preset_path, &placement, 0.0)
                        .with_context(|| format!("simulating {}", system.preset_path))?
                        .image;
                    let kind = LiveKind::Particle { placement, preset_path: system.preset_path.clone() };
                    (image, full_rect, system.blend, kind, system.object)
                }
            };

            let effects: Vec<_> = model::visible_effects(object).collect();
            let chain = if effects.is_empty() {
                None
            } else {
                match render::prepare_effect_chain(gl, archive, &effects, &image, headers) {
                    Ok(chain) => Some(chain),
                    Err(error) => {
                        omissions.push(format!("{}: effect chain skipped ({error:#})", model::label(object)));
                        None
                    }
                }
            };

            let start = tweak_values.len();
            if let Some(chain) = &chain {
                tweak_values.extend(chain.tweakables.iter().map(|tweakable| tweakable.default));
            }

            let textures = if matches!(kind, LiveKind::Image) {
                LayerTextures::once(gl, &image)?
            } else {
                LayerTextures::ring(gl, &image)?
            };
            layers.push(LiveLayer {
                kind,
                name: model::label(object),
                additive: blend == Blend::Add,
                chain,
                tweaks: start..tweak_values.len(),
                textures,
                rect,
            });
        }

        Ok(Built { layers, tweak_values, omissions })
    }
}

/// `build_layers`' output: the z-ordered layers, the flattened tweakable
/// values, and the reconciled list of what is still not simulated.
struct Built {
    layers: Vec<LiveLayer>,
    tweak_values: Vec<f32>,
    omissions: Vec<String>,
}

/// Replace `stale` with `replacement` in `notes`, if present.
fn swap_note(notes: &mut [String], stale: &str, replacement: String) {
    if let Some(slot) = notes.iter_mut().find(|note| *note == stale) {
        *slot = replacement;
    }
}

/// Returns `true` when the caller should close the window (the debug dump hook
/// asks for this after writing its frame).
fn redraw(app: &mut App, time: f32) -> Result<bool> {
    let App { archive, static_scene, state, .. } = app;
    let Some(state) = state.as_mut() else { return Ok(false) };

    run_panel(state);

    // Per-frame CPU work: re-skin puppets, re-simulate particles, then upload
    // each fresh image into the next texture in its ring.
    for layer in &mut state.layers {
        let refreshed = match &layer.kind {
            LiveKind::Image => None,
            LiveKind::Puppet(index) => {
                let StaticItem::Puppet(puppet) = &static_scene.items[*index] else {
                    unreachable!("a Puppet LiveKind always points at a Puppet item")
                };
                let (image, left, top) = compose::warp_frame(puppet, time);
                let rect = rect_of(left, top, &image);
                Some((image, rect))
            }
            LiveKind::Particle { placement, preset_path } => {
                let image = particle::render_system(archive, preset_path, placement, time)
                    .with_context(|| format!("simulating {preset_path}"))?
                    .image;
                Some((image, layer.rect))
            }
        };
        if let Some((image, rect)) = refreshed {
            layer.rect = rect;
            layer.textures.refresh(&state.gl, &image)?;
        }
    }

    // Stack the layers into the composite target, each under its blend mode.
    pass::clear_target(&state.gl, &state.composite, state.background);
    for layer in &state.layers {
        let source = match &layer.chain {
            Some(chain) => {
                // A static image's chain keeps its baked-in base; a puppet or
                // particle layer's image changed this frame, so re-feed it.
                let base = match layer.kind {
                    LiveKind::Image => None,
                    LiveKind::Puppet(_) | LiveKind::Particle { .. } => Some(layer.textures.current()),
                };
                chain
                    .render_over(&state.gl, base, time, &state.tweak_values[layer.tweaks.clone()])?
                    .texture
            }
            None => layer.textures.current(),
        };
        pass::composite_layer(&state.gl, &state.compositor, &state.composite, source, layer.rect, layer.additive);
    }

    // Debug hook: `SIMULATE_DUMP=<path>` writes the composited frame (the live
    // pipeline's own output, before the window blit) and exits, so the
    // per-layer chains + GPU compositing can be eyeballed headlessly.
    // Pair with `SIMULATE_TIME=<secs>` to pin `g_Time`.
    if let Ok(path) = std::env::var("SIMULATE_DUMP") {
        state.frames_since.1 += 1;
        if state.frames_since.1 >= 2 {
            let frame = capture::read_rgba(
                &state.gl,
                state.composite.framebuffer,
                state.composite.width,
                state.composite.height,
            )?;
            frame.save(&path).with_context(|| format!("writing {path}"))?;
            println!("  wrote {path}");
            return Ok(true);
        }
    }

    let size = state.window.inner_size();
    #[expect(clippy::cast_possible_wrap, reason = "window dimensions are nowhere near i32::MAX")]
    let window = (size.width as i32, size.height as i32);
    pass::blit_to_screen(&state.gl, &state.blit, &state.display_quad, state.composite.texture, state.content_size, window);
    state.egui.paint(&state.window);
    state.surface.swap_buffers(&state.context).context("swapping buffers")?;

    report_fps(&mut state.frames_since);
    Ok(false)
}

/// Print a frame-rate line about once a second so a slow scene is visible
/// without a profiler.
fn report_fps(counter: &mut (Instant, u32)) {
    counter.1 += 1;
    let elapsed = counter.0.elapsed();
    if elapsed.as_secs() >= 1 {
        let fps = f64::from(counter.1) / elapsed.as_secs_f64();
        println!("  {fps:.0} fps");
        *counter = (Instant::now(), 0);
    }
}

/// Run the egui pass and copy the slider values back into `state.tweak_values`.
fn run_panel(state: &mut State) {
    let labels = panel_labels(state);
    let mut values = state.tweak_values.clone();
    state.egui.run(&state.window, |ctx| {
        egui::Window::new("Effect Parameters").show(ctx, |ui| {
            if labels.is_empty() {
                ui.label("No tweakable parameters in this scene.");
            }
            for ((label, min, max), value) in labels.iter().zip(values.iter_mut()) {
                ui.add(egui::Slider::new(value, *min..=*max).text(label));
            }
        });
    });
    state.tweak_values = values;
}

/// One `(label, min, max)` per tweakable across every layer's chain, in the
/// same order as `state.tweak_values`. Labels are prefixed with the layer name
/// so sliders from different layers stay apart.
fn panel_labels(state: &State) -> Vec<(String, f32, f32)> {
    let mut labels = Vec::new();
    for layer in &state.layers {
        let Some(chain) = &layer.chain else { continue };
        for tweakable in &chain.tweakables {
            labels.push((format!("{} · {}", layer.name, tweakable.label), tweakable.min, tweakable.max));
        }
    }
    labels
}

fn report_tweakables(layers: &[LiveLayer]) {
    let total: usize = layers.iter().filter_map(|layer| layer.chain.as_ref()).map(|chain| chain.tweakables.len()).sum();
    if total == 0 {
        println!("  no tweakable parameters in this scene");
    } else {
        println!("  {total} tweakable parameter(s) — drag them in the Effect Parameters panel");
    }
}

/// A scene background pixel as a normalized RGBA clear colour.
fn normalized_rgba(pixel: image::Rgba<u8>) -> [f32; 4] {
    pixel.0.map(|channel| f32::from(channel) / 255.0)
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        event_loop.set_control_flow(ControlFlow::Poll);
        match self.open_window(event_loop) {
            Ok(state) => self.state = Some(state),
            Err(error) => {
                eprintln!("Error: opening the simulator window\nCaused by: {error:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if self.state.is_none() {
            return;
        }
        if let Some(state) = &mut self.state {
            let _ = state.egui.on_window_event(&state.window, &event);
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state.is_pressed() && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let (Some(width), Some(height), Some(state)) =
                    (NonZeroU32::new(size.width), NonZeroU32::new(size.height), &self.state)
                {
                    state.surface.resize(&state.context, width, height);
                }
            }
            WindowEvent::RedrawRequested => {
                let time = std::env::var("SIMULATE_TIME")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| self.start.elapsed().as_secs_f32());
                match redraw(self, time) {
                    Ok(true) => event_loop.exit(),
                    Ok(false) => {}
                    Err(error) => {
                        eprintln!("Error: drawing a frame\nCaused by: {error:#}");
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.egui.destroy();
        }
    }
}

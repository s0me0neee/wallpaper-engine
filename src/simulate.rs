//! A live window that plays a scene's effect chain in real time.
//!
//! Unlike `export`, which renders one frame and reads it back to the CPU,
//! this attaches a GL context directly to a visible window and redraws every
//! frame with wall-clock `g_Time`. Nothing else Wallpaper Engine feeds a
//! shader — mouse position, system audio, time-of-day, now-playing media —
//! is wired up yet; see plan.md. An `egui` overlay does let you drag any
//! range-annotated shader parameter (e.g. `foliagesway`'s wave strength)
//! away from the wallpaper's own preset, live.

use crate::pkg::Archive;
use crate::render::pass;
use crate::scene::compose;
use crate::scene::model::{Effect, Scene};
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
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Open a window and play `scene` in it until it is closed.
///
/// Runs the effect chain when the scene has the shape `scene::render`
/// handles; otherwise the window just shows the static composite, same
/// fallback `export` uses, since a still preview is still a useful window.
pub fn run(archive: &mut Archive, scene: &Scene, title: &str) -> Result<()> {
    let mut composite = compose::render(archive, scene, None)?;

    let effects = render::effect_chain_shape(scene).map(|(_, effects)| effects);
    if effects.is_some() {
        // The chain covers what this note describes; `render_frame` drops it
        // the same way once the chain actually runs.
        composite.omissions.retain(|note| !note.ends_with("effect(s) not applied"));
    } else {
        println!("  no effect chain on this scene; showing the static composite");
    }
    for note in &composite.omissions {
        println!("  not simulated: {note}");
    }

    let event_loop = EventLoop::new().context("opening a window event loop")?;
    let mut app = App {
        title: title.to_string(),
        archive,
        effects,
        base: composite.image,
        headers: shim::headers(),
        start: Instant::now(),
        state: None,
    };
    event_loop.run_app(&mut app).context("running the simulator window")
}

struct App<'a> {
    title: String,
    archive: &'a mut Archive,
    effects: Option<Vec<&'a Effect>>,
    base: RgbaImage,
    headers: HashMap<String, String>,
    start: Instant,
    state: Option<State>,
}

/// Everything that only exists once the window itself does — winit only
/// hands out a window from inside `resumed`, never before.
struct State {
    window: Window,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    /// Shared with `egui`'s painter, which needs to hold its own handle to it.
    gl: Arc<glow::Context>,
    quad: pass::Quad,
    blit: pass::BlitProgram,
    chain: Option<EffectChain>,
    base_texture: glow::Texture,
    egui: egui_glow::winit::EguiGlow,
    /// One live value per `chain`'s `tweakables`, in the same order —
    /// starts at the wallpaper's own preset and moves as the panel's
    /// sliders are dragged.
    values: Vec<f32>,
}

impl App<'_> {
    fn open_window(&mut self, event_loop: &ActiveEventLoop) -> Result<State> {
        let (width, height) = self.base.dimensions();
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

        let quad = pass::build_display_quad(&gl)?;
        let blit = pass::compile_blit_program(&gl)?;
        let base_texture = pass::upload_texture(&gl, &self.base)?;
        let chain = self
            .effects
            .as_ref()
            .map(|effects| render::prepare_effect_chain(&gl, self.archive, effects, &self.base, &self.headers))
            .transpose()
            .context("preparing the effect chain")?;
        if let Some(chain) = &chain {
            if chain.tweakables.is_empty() {
                println!("  no tweakable parameters in this chain");
            } else {
                println!("  tweakable parameters (drag them in the Effect Parameters panel):");
                for tweakable in &chain.tweakables {
                    println!("    {} = {} (range {}..{})", tweakable.label, tweakable.default, tweakable.min, tweakable.max);
                }
            }
        }
        let values = chain.as_ref().map(|chain| chain.tweakables.iter().map(|t| t.default).collect()).unwrap_or_default();
        let egui = egui_glow::winit::EguiGlow::new(event_loop, Arc::clone(&gl), None, None, true);

        window.request_redraw();
        Ok(State { window, surface, context, gl, quad, blit, chain, base_texture, egui, values })
    }
}

/// The panel's static description of each tweakable — cloned out of `state`
/// up front so the closure `egui`'s `run` takes doesn't need to borrow
/// `state` at all (it can't: `run` already holds `state.egui` mutably).
fn panel_labels(state: &State) -> Vec<(String, f32, f32)> {
    state
        .chain
        .as_ref()
        .map(|chain| chain.tweakables.iter().map(|t| (t.label.clone(), t.min, t.max)).collect())
        .unwrap_or_default()
}

fn redraw(state: &mut State, time: f32) -> Result<()> {
    let labels = panel_labels(state);
    let mut values = state.values.clone();
    state.egui.run(&state.window, |ctx| {
        egui::Window::new("Effect Parameters").show(ctx, |ui| {
            if labels.is_empty() {
                ui.label("No tweakable parameters in this chain.");
            }
            for ((label, min, max), value) in labels.iter().zip(values.iter_mut()) {
                ui.add(egui::Slider::new(value, *min..=*max).text(label));
            }
        });
    });
    state.values = values;

    let texture = match &state.chain {
        Some(chain) => chain.render(&state.gl, time, &state.values)?.texture,
        None => state.base_texture,
    };

    let size = state.window.inner_size();
    #[expect(clippy::cast_possible_wrap, reason = "window dimensions are nowhere near i32::MAX")]
    let (width, height) = (size.width as i32, size.height as i32);
    pass::blit_to_screen(&state.gl, &state.blit, &state.quad, texture, width, height);
    state.egui.paint(&state.window);
    state.surface.swap_buffers(&state.context).context("swapping buffers")
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
        let Some(state) = &mut self.state else { return };
        // Feed the panel every event so its sliders can be dragged; still
        // handle Resized/CloseRequested/Escape below regardless of whether
        // egui says it "consumed" one, since those are the window's business.
        let _ = state.egui.on_window_event(&state.window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state.is_pressed() && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let (Some(width), Some(height)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) {
                    state.surface.resize(&state.context, width, height);
                }
            }
            WindowEvent::RedrawRequested => {
                let time = self.start.elapsed().as_secs_f32();
                if let Err(error) = redraw(state, time) {
                    eprintln!("Error: drawing a frame\nCaused by: {error:#}");
                    event_loop.exit();
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

//! A headless OpenGL 3.3 core context.
//!
//! Never shown to the user: Wallpaper Engine's shaders are rendered
//! off-screen, frame by frame, then read back to CPU memory for encoding.
//! Every actual target is an FBO the caller attaches its own textures to.
//!
//! Only `export` opens one of these. The live simulator (`simulate.rs`) has a
//! real window and takes its context from `glutin-winit`, so nothing in the
//! live path depends on anything below.
//!
//! Two backends, because the platforms disagree about what a context with no
//! window attaches to. macOS's CGL rejects both surfaceless and pbuffer
//! surfaces outright (measured against glutin 0.31 and 0.32 alike), so it gets
//! a one-pixel `NSWindow` that is built, used as a backing view, and never
//! ordered onto the screen. Everywhere else, an EGL pbuffer is exactly this
//! and needs no window-system object at all.

use anyhow::Result;

/// The GL context, and whatever the platform needs kept alive alongside it.
pub struct Gpu {
    pub gl: glow::Context,
    _anchor: backend::Anchor,
}

impl Gpu {
    /// Open a headless OpenGL 3.3 core context.
    pub fn new() -> Result<Gpu> {
        let (gl, anchor) = backend::open()?;
        Ok(Gpu { gl, _anchor: anchor })
    }
}

#[cfg(target_os = "macos")]
mod backend {
    use anyhow::{Context, Result, anyhow};
    use glutin::config::ConfigTemplateBuilder;
    use glutin::context::{
        ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext, Version,
    };
    use glutin::display::{Display, DisplayApiPreference, GlDisplay};
    use glutin::surface::{Surface, SurfaceAttributesBuilder, WindowSurface};
    use objc2::rc::Retained;
    use objc2::{MainThreadMarker, MainThreadOnly};
    use objc2_app_kit::{NSBackingStoreType, NSView, NSWindow, NSWindowStyleMask};
    use objc2_foundation::{NSPoint, NSRect, NSSize};
    use raw_window_handle::{
        AppKitDisplayHandle, AppKitWindowHandle, RawDisplayHandle, RawWindowHandle,
    };
    use std::ffi::CString;
    use std::num::NonZeroU32;
    use std::ptr::NonNull;

    /// Everything that has to outlive the context: the surface, and the window
    /// whose view backs it.
    pub struct Anchor {
        _context: PossiblyCurrentContext,
        _surface: Surface<WindowSurface>,
        _window: Retained<NSWindow>,
    }

    /// Build a 1x1 `NSWindow` and return it with its content view's raw handle.
    ///
    /// The window is never ordered front, so it is never drawn or visible; it
    /// exists purely because CGL requires a real view to attach a context to.
    fn hidden_window(mtm: MainThreadMarker) -> Result<(Retained<NSWindow>, NonNull<std::ffi::c_void>)> {
        let rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1.0, 1.0));
        // Safety: a plain, undecorated window with no delegate or event handling
        // attached; `defer: false` means it is fully created here, before this
        // function returns anything referencing its view.
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                rect,
                NSWindowStyleMask::Borderless,
                NSBackingStoreType::Buffered,
                false,
            )
        };

        let view: Retained<NSView> = window.contentView().context("the window has no content view")?;
        let pointer = Retained::as_ptr(&view).cast_mut().cast::<std::ffi::c_void>();
        let handle = NonNull::new(pointer).context("content view pointer was null")?;
        Ok((window, handle))
    }

    pub fn open() -> Result<(glow::Context, Anchor)> {
        let mtm = MainThreadMarker::new()
            .context("the renderer must be set up on the main thread (AppKit requires it)")?;
        let (window, view_handle) = hidden_window(mtm)?;

        let raw_display = RawDisplayHandle::AppKit(AppKitDisplayHandle::new());
        // Safety: the handle carries no live window and is valid for the
        // display's whole lifetime, which is what CGL requires of it.
        let display = unsafe { Display::new(raw_display, DisplayApiPreference::Cgl) }
            .context("opening a CGL display")?;

        let template = ConfigTemplateBuilder::new().with_alpha_size(8).build();
        // Safety: `template` describes no window either; CGL picks a pixel
        // format purely from the requested buffer layout.
        let config = unsafe { display.find_configs(template) }
            .map_err(|error| anyhow!("finding a GL config: {error}"))?
            .next()
            .context("no GL config satisfies the requested pixel format")?;

        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .build(None);
        // Safety: no window is referenced by these attributes either.
        let not_current = unsafe { display.create_context(&config, &context_attributes) }
            .context("creating a GL 3.3 core context")?;

        let raw_window_handle = RawWindowHandle::AppKit(AppKitWindowHandle::new(view_handle));
        let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            raw_window_handle,
            NonZeroU32::new(1).context("1 is non-zero")?,
            NonZeroU32::new(1).context("1 is non-zero")?,
        );
        // Safety: the window above outlives this call and is never shown; the
        // view is real, on the main thread, and stays alive in `Anchor`.
        let surface = unsafe { display.create_window_surface(&config, &surface_attributes) }
            .context("attaching the GL context to the hidden window's view")?;

        let context = not_current
            .make_current(&surface)
            .context("making the GL context current")?;

        let gl = unsafe {
            glow::Context::from_loader_function(|name| {
                let name = CString::new(name).unwrap_or_default();
                display.get_proc_address(&name).cast()
            })
        };

        Ok((gl, Anchor { _context: context, _surface: surface, _window: window }))
    }
}

#[cfg(not(target_os = "macos"))]
mod backend {
    use anyhow::{Context, Result, anyhow};
    use glutin::config::ConfigTemplateBuilder;
    use glutin::context::{
        ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext, Version,
    };
    use glutin::display::{Display, DisplayApiPreference, GlDisplay};
    use glutin::surface::{PbufferSurface, Surface, SurfaceAttributesBuilder};
    use raw_window_handle::{RawDisplayHandle, XlibDisplayHandle};
    use std::ffi::CString;
    use std::num::NonZeroU32;

    pub struct Anchor {
        _context: PossiblyCurrentContext,
        _surface: Surface<PbufferSurface>,
    }

    pub fn open() -> Result<(glow::Context, Anchor)> {
        // An `Xlib` handle with no display pointer is EGL_DEFAULT_DISPLAY, so
        // this asks the EGL implementation for whatever the session already
        // has rather than opening a connection of its own.
        let raw_display = RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0));
        // Safety: the handle owns no connection and names no window, so there
        // is nothing for it to outlive.
        let display = unsafe { Display::new(raw_display, DisplayApiPreference::Egl) }
            .context("opening an EGL display")?;

        // A pbuffer config has to say so: a driver may well offer configs that
        // can only ever back a window.
        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_surface_type(glutin::config::ConfigSurfaceTypes::PBUFFER)
            .build();
        // Safety: `template` names no window.
        let config = unsafe { display.find_configs(template) }
            .map_err(|error| anyhow!("finding a GL config: {error}"))?
            .next()
            .context("no GL config satisfies the requested pixel format")?;

        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .build(None);
        // Safety: no window is referenced by these attributes.
        let not_current = unsafe { display.create_context(&config, &context_attributes) }
            .context("creating a GL 3.3 core context")?;

        // 1x1, like the macOS window: every real target is an FBO, so the
        // surface is only ever the thing the context is made current against.
        let one = NonZeroU32::new(1).context("1 is non-zero")?;
        let surface_attributes = SurfaceAttributesBuilder::<PbufferSurface>::new().build(one, one);
        // Safety: the attributes describe a pbuffer, which owns its own
        // storage and borrows nothing.
        let surface = unsafe { display.create_pbuffer_surface(&config, &surface_attributes) }
            .context("creating a 1x1 pbuffer surface")?;

        let context = not_current
            .make_current(&surface)
            .context("making the GL context current")?;

        let gl = unsafe {
            glow::Context::from_loader_function(|name| {
                let name = CString::new(name).unwrap_or_default();
                display.get_proc_address(&name).cast()
            })
        };

        Ok((gl, Anchor { _context: context, _surface: surface }))
    }
}

//! Pinning the live renderer's window to the desktop background.
//!
//! `simulate` renders into an ordinary titled window. A wallpaper is the same
//! renderer in a window the window server treats differently: borderless,
//! sitting between the desktop picture and the desktop icons, transparent to
//! the mouse, present on every Space, and owned by a process with no Dock icon.
//! None of that touches the GL path — `simulate.rs` draws exactly the same
//! frames either way — so it lives here rather than spreading through it.
//!
//! macOS only so far. The other platforms want an entirely different mechanism
//! (X11's `_NET_WM_WINDOW_TYPE_DESKTOP`, Wayland's `wlr-layer-shell`), neither
//! of which winit exposes, so `attach` says so instead of quietly handing back
//! an ordinary window that floats over everything.

use anyhow::{Context, Result};
use winit::event_loop::EventLoop;
use winit::window::Window;

/// Where a live scene is being drawn.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Presentation {
    /// A titled, focusable window with the `egui` parameter panel — `simulate`.
    Window,
    /// The desktop background, under the icons — `desktop`.
    Background,
}

/// Build the event loop a `presentation` needs.
///
/// Activation policy is a property of the *application*, not of any one window,
/// and winit only lets it be set before the loop is built — so a background
/// renderer has to be decided here rather than after the window exists.
pub fn event_loop(presentation: Presentation) -> Result<EventLoop<()>> {
    let mut builder = EventLoop::builder();
    #[cfg(target_os = "macos")]
    if presentation == Presentation::Background {
        use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
        // `Accessory` is LSUIElement at runtime: no Dock icon, no Cmd-Tab
        // entry, no menu bar. `with_activate_ignoring_other_apps(false)` stops
        // it stealing focus from whatever the user was doing at launch.
        builder.with_activation_policy(ActivationPolicy::Accessory);
        builder.with_activate_ignoring_other_apps(false);
        builder.with_default_menu(false);
    }
    // Nothing off macOS reads it yet; `attach` is what refuses there.
    let _ = presentation;
    builder.build().context("opening a window event loop")
}

/// Sink `window` into the desktop background.
///
/// Called once, after the window exists but before it has drawn anything.
#[cfg(target_os = "macos")]
pub fn attach(window: &Window) -> Result<()> {
    use objc2_app_kit::{NSView, NSWindowCollectionBehavior};
    use objc2_core_graphics::kCGDesktopIconWindowLevel;
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let handle = window.window_handle().context("the wallpaper window has no raw handle")?;
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        anyhow::bail!("the wallpaper window is not an AppKit window");
    };

    // winit links its own (older) `objc2-app-kit`, so this pointer cannot come
    // back as our `NSView` type through the type system — but both are markers
    // for the same Objective-C class, and the selectors below are the same
    // selectors. Safety: winit owns the view and keeps it alive for the
    // window's whole lifetime, which outlives this borrow.
    let view: &NSView = unsafe { handle.ns_view.cast::<NSView>().as_ref() };
    let ns_window = view.window().context("the wallpaper view is not in a window")?;

    // One below the icons, which is one *above* the desktop picture. Sitting at
    // `kCGDesktopWindowLevel` instead would put us level with the Dock's own
    // picture window and leave the ordering between them to chance.
    ns_window.setLevel(kCGDesktopIconWindowLevel as isize - 1);
    // Clicks, drags and rubber-band selections belong to the desktop beneath.
    ns_window.setIgnoresMouseEvents(true);
    ns_window.setHasShadow(false);
    // A wallpaper is on every Space at once and does not slide with them, and
    // Mission Control and Cmd-Tab should not offer it as a window to switch to.
    ns_window.setCollectionBehavior(
        NSWindowCollectionBehavior::CanJoinAllSpaces
            | NSWindowCollectionBehavior::Stationary
            | NSWindowCollectionBehavior::IgnoresCycle,
    );

    // Read back rather than echo: AppKit clamps a level it will not honour, and
    // the frame is the one number that says whether the window actually covers
    // the screen it was sized for.
    let frame = ns_window.frame();
    println!(
        "  window level {} (icons sit at {kCGDesktopIconWindowLevel}), frame {}x{} at {}, {}",
        ns_window.level(),
        frame.size.width,
        frame.size.height,
        frame.origin.x,
        frame.origin.y,
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn attach(_window: &Window) -> Result<()> {
    anyhow::bail!(
        "running as the desktop background is implemented for macOS only: X11 wants \
         _NET_WM_WINDOW_TYPE_DESKTOP and Wayland wants wlr-layer-shell, and winit exposes neither"
    )
}

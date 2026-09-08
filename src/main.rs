//! Turn a Wallpaper Engine wallpaper into a still PNG and a looping video that
//! an ordinary wallpaper app can use.
//!
//! `project.json` selects the pipeline. Video wallpapers already contain a
//! finished loop and are packaged directly; scene and web wallpapers need
//! rendering and are not built yet. `unpack` and `tex` remain as the debugging
//! tools the container work was built with.

mod export;
mod paths;
mod pkg;
mod project;
mod reader;
mod render;
mod scene;
mod shader;
mod simulate;
mod tex;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use export::{Options, Resolution};
use project::{Kind, Project};
use scene::model;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Export Wallpaper Engine wallpapers to ordinary images and video.
#[derive(Parser)]
#[command(name = "wallpaper-engine", version, about, long_about = None, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert a wallpaper to a still PNG and a looping video.
    Export(ExportArgs),
    /// Report what a wallpaper is and whether we can export it.
    Info(InfoArgs),
    /// Extract the contents of a .pkg archive.
    Unpack(UnpackArgs),
    /// Decode .tex textures to PNG.
    Tex(TexArgs),
    /// Preprocess a scene's shaders into compilable GLSL.
    Shaders(ShadersArgs),
    /// Open a live window playing a scene's effect chain in real time.
    Simulate(SimulateArgs),
}

#[derive(Args)]
struct SimulateArgs {
    /// Wallpaper directory, or its project.json. Must be a Scene wallpaper.
    wallpaper: PathBuf,
}

#[derive(Args)]
struct ShadersArgs {
    /// Wallpaper directory, or its project.json.
    wallpaper: PathBuf,

    /// Directory to write the preprocessed GLSL into.
    #[arg(short, long, default_value = "shaders")]
    out: PathBuf,

    /// Read the real common*.h from a Wallpaper Engine install instead of
    /// using our shim. Never redistributed — read from your own install.
    #[arg(long)]
    we_assets: Option<PathBuf>,
}

#[derive(Args)]
struct InfoArgs {
    /// Wallpaper directory, or its project.json.
    wallpaper: PathBuf,
}

#[derive(Args)]
struct ExportArgs {
    /// Wallpaper directory, or its project.json.
    wallpaper: PathBuf,

    /// Directory the wallpaper's own output folder is created under.
    #[arg(short, long, default_value = ".")]
    out: PathBuf,

    /// Skip the video; write only the still(s) named by `--frame`.
    #[arg(long)]
    png_only: bool,

    /// Export a still at this timestamp, in seconds. Repeatable, so several
    /// frames can be exported in one run. A video wallpaper exports the video
    /// only unless at least one `--frame` is given.
    #[arg(long = "frame", value_name = "SECS")]
    frames: Vec<f64>,

    /// Output size as `WIDTHxHEIGHT`. Defaults to the source resolution.
    #[arg(long)]
    resolution: Option<Resolution>,

    /// Output frame rate. Defaults to the source rate.
    #[arg(long)]
    fps: Option<f64>,

    /// Trim the video to this many seconds.
    #[arg(long)]
    duration: Option<f64>,

    /// Keep the audio track. Off by default; wallpaper apps ignore it.
    #[arg(long)]
    audio: bool,
}

#[derive(Args)]
struct UnpackArgs {
    /// Archive to read.
    #[arg(default_value = "papers/scene_example1/scene.pkg")]
    pkg: PathBuf,

    /// Directory to extract into.
    #[arg(short, long, default_value = "unpacked")]
    out: PathBuf,

    /// List the contents without extracting anything.
    #[arg(short, long)]
    list: bool,
}

#[derive(Args)]
struct TexArgs {
    /// Texture files, or directories to search recursively.
    #[arg(default_value = "unpacked")]
    paths: Vec<PathBuf>,

    /// Directory to write images into.
    #[arg(short, long, default_value = "textures")]
    out: PathBuf,

    /// Describe the textures without converting them.
    #[arg(short, long)]
    info: bool,

    /// Export every mip level, not just the largest.
    #[arg(short, long)]
    all_mipmaps: bool,
}

/// Collect `.tex` files, expanding any directory argument recursively.
fn collect_textures(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();

    for path in paths {
        if !path.is_dir() {
            found.push(path.clone());
            continue;
        }

        for entry in WalkDir::new(path).sort_by_file_name() {
            let entry = entry?;
            let is_tex = entry
                .path()
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("tex"));
            if entry.file_type().is_file() && is_tex {
                found.push(entry.into_path());
            }
        }
    }

    Ok(found)
}

fn run_unpack(args: &UnpackArgs) -> Result<()> {
    pkg::unpack(&args.pkg, &args.out, args.list)
}

fn run_tex(args: &TexArgs) -> Result<()> {
    let targets = collect_textures(&args.paths)?;
    if targets.is_empty() {
        bail!("no .tex files found");
    }

    let mut failures = 0;
    for target in &targets {
        if let Err(error) = tex::convert(target, &args.out, args.all_mipmaps, args.info) {
            eprintln!("error: {}: {error:#}", target.display());
            failures += 1;
        }
    }

    if failures > 0 {
        bail!("{failures} of {} textures failed", targets.len());
    }
    Ok(())
}

/// Preprocess every shader in a scene package into compilable GLSL.
///
/// A debugging tool for the renderer: the combo values are all defaulted to 0
/// rather than resolved per effect pass, so what comes out is the base variant
/// of each shader. It exists to answer "does the shim cover this wallpaper?"
/// without needing a GPU.
fn run_shaders(args: &ShadersArgs) -> Result<()> {
    let project = project::load(&args.wallpaper)?;
    let package = project::require_package(&project)?;
    let mut archive = pkg::Archive::open(package)?;

    let headers = match &args.we_assets {
        Some(root) => shader::shim::headers_from_install(root)?,
        None => shader::shim::headers(),
    };

    let is_shader = |path: &&str| {
        Path::new(path)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("frag") || extension.eq_ignore_ascii_case("vert"))
    };
    let mut targets: Vec<String> = archive.paths().filter(is_shader).map(str::to_string).collect();
    targets.sort();

    if targets.is_empty() {
        bail!("{} contains no shaders", package.display());
    }
    std::fs::create_dir_all(&args.out)?;

    let mut failures = 0;
    for name in &targets {
        let is_vertex = Path::new(name)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("vert"));
        let stage = if is_vertex {
            shader::preprocess::Stage::Vertex
        } else {
            shader::preprocess::Stage::Fragment
        };

        let source = String::from_utf8(archive.read(name)?)
            .with_context(|| format!("{name} is not valid UTF-8"))?;

        // Combos default to 0 here; the renderer will supply real values.
        let combos = std::collections::BTreeMap::new();
        match shader::preprocess::build(&source, stage, &headers, &combos) {
            Ok(glsl) => {
                let stem = name.replace('/', "_");
                let target = args.out.join(format!("{stem}.glsl"));
                std::fs::write(&target, glsl)?;
                println!("  {} -> {}", name, target.display());
            }
            Err(error) => {
                eprintln!("error: {name}: {error:#}");
                failures += 1;
            }
        }
    }

    println!("\n{} shader(s) written to {}/", targets.len() - failures, args.out.display());
    if failures > 0 {
        bail!("{failures} of {} shaders failed", targets.len());
    }
    Ok(())
}

/// Turn a wallpaper's title into a safe directory name.
///
/// Titles are free text: real ones carry `/`, `|`, quotes and any Unicode
/// script. Path separators and the handful of characters Windows forbids in a
/// filename become `-`; CJK and other non-ASCII text is left alone, since it
/// is exactly what makes the folder recognisable.
fn sanitize_component(name: &str) -> String {
    let replaced: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '-',
            c if c.is_control() => '-',
            c => c,
        })
        .collect();

    // Windows also rejects trailing dots and spaces on a directory name.
    let trimmed = replaced.trim().trim_end_matches('.').trim();
    if trimmed.is_empty() {
        "wallpaper".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The name of the still a video timestamp produces.
///
/// Plain `still.png` unless more than one `--frame` was requested, since then
/// each needs to be told apart.
fn still_name(time: f64, multiple: bool) -> String {
    if multiple {
        format!("still_t{time}s.png")
    } else {
        "still.png".to_string()
    }
}

/// Explain why a wallpaper cannot be exported, or `None` if it can.
///
/// Kept separate from the export itself so `info` and `export` agree, and so
/// the reasons stay in one readable list as pipelines land.
fn unsupported_reason(project: &Project) -> Option<String> {
    match &project.kind {
        // Scene is always exportable as a still: the layers composite even
        // where the effect chain doesn't apply (see `scene_runs_effects` in
        // `run_info` for exactly which scenes that is).
        Kind::Scene if project.package.is_none() => {
            Some("no scene.pkg beside project.json, so there are no assets to render".to_string())
        }
        Kind::Video | Kind::Scene => None,
        Kind::Web => {
            if project.oversized {
                // These are media-player apps, not wallpapers: hundreds of
                // megabytes of bundled video and audio with no canonical
                // frame. Capturing one would record whatever happened to be
                // playing, which is worse than declining.
                Some("oversized web wallpapers bundle their own media player".to_string())
            } else {
                Some("web wallpapers need headless capture, which is not built yet".to_string())
            }
        }
        Kind::Application => {
            Some("application wallpapers are Windows executables and cannot be rendered".to_string())
        }
        Kind::Unknown(raw) => Some(format!("unrecognized wallpaper type {raw:?}")),
    }
}

fn run_info(args: &InfoArgs) -> Result<()> {
    let project = project::load(&args.wallpaper)?;

    println!("{}", project::display_name(&project));
    println!("  type       {}", project.kind);
    println!("  directory  {}", project.root.display());
    match &project.entry {
        Some(entry) if project::entry_is_packaged(&project) => {
            println!("  entry      {} (inside scene.pkg)", entry.display());
        }
        Some(entry) => {
            let missing = if entry.exists() { "" } else { "  (MISSING)" };
            println!("  entry      {}{missing}", entry.display());
        }
        None => println!("  entry      <none>"),
    }
    if let Some(package) = &project.package {
        println!("  package    {}", package.display());
    }
    if let Some(preview) = &project.preview {
        println!("  preview    {}", preview.display());
    }
    if project.oversized {
        println!("  oversized  yes");
    }
    if !project.properties.is_empty() {
        println!("  settings   {}", project.properties.len());
    }

    // A probe is cheap and is the thing you actually want to know about a
    // video wallpaper before exporting it.
    if project.kind == Kind::Video
        && let Some(entry) = project.entry.as_deref().filter(|entry| entry.exists())
        && let Ok(info) = export::ffmpeg::probe(entry)
    {
        println!(
            "  video      {}x{} {:.3} fps, {:.2}s, {} / {}{}",
            info.width,
            info.height,
            info.fps,
            info.duration,
            info.codec,
            info.pixel_format,
            if info.has_audio { ", audio" } else { "" }
        );
    }

    // A scene's canvas and layer count come out of the package, which is worth
    // knowing before asking for a 33-megapixel still. It also decides which
    // "export" message below is honest: the effect chain only runs for the
    // one-image-layer shape plan.md §4.1 describes.
    let mut scene_runs_effects = false;
    if project.kind == Kind::Scene
        && let Some(package) = project.package.as_deref()
        && let Ok(mut archive) = pkg::Archive::open(package)
        && let Ok(scene) = scene::load(&mut archive)
    {
        let visible_images: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| o.visible && model::is_image(o))
            .collect();
        let images = visible_images.len();
        let particles = scene.objects.iter().filter(|o| model::is_particle(o)).count();
        let effects: usize = scene
            .objects
            .iter()
            .map(|o| model::visible_effects(o).count())
            .sum();
        scene_runs_effects =
            images == 1 && model::visible_effects(visible_images[0]).next().is_some();

        if let Some(ortho) = scene.general.orthographic {
            println!("  canvas     {}x{}", ortho.width, ortho.height);
        }
        println!("  layers     {images} image, {particles} particle, {effects} effect(s)");
    }

    match unsupported_reason(&project) {
        Some(reason) => println!("  export     no: {reason}"),
        None if project.kind == Kind::Scene && scene_runs_effects => {
            println!("  export     yes (effect chain rendered; particles are not)");
        }
        None if project.kind == Kind::Scene => {
            println!("  export     still only (effects and particles are not rendered)");
        }
        None => println!("  export     yes"),
    }
    Ok(())
}

/// Video wallpapers ship a finished looping file; exporting is packaging.
///
/// The video is always written unless `--png-only` says otherwise; a still is
/// written only for each `--frame` explicitly asked for, since the wallpaper
/// already contains a finished loop and there is no single canonical frame to
/// default to.
fn export_video(project: &Project, options: &Options) -> Result<()> {
    let source = project::require_entry(project)?;
    export::ffmpeg::init()?;

    let info = export::ffmpeg::probe(source)?;
    println!(
        "  source     {}x{} {:.3} fps, {:.2}s, {}",
        info.width, info.height, info.fps, info.duration, info.codec
    );

    if !options.video && options.frames.is_empty() {
        bail!(
            "nothing to export: pass --frame <SECS> for a still, or drop --png-only for the video"
        );
    }

    let multiple = options.frames.len() > 1;
    let size = options
        .resolution
        .map(|resolution| (resolution.width, resolution.height));
    for &time in &options.frames {
        let out = options.out_dir.join(still_name(time, multiple));
        export::still::from_video(source, &out, time, size)
            .with_context(|| format!("writing the frame at {time}s of {}", source.display()))?;
        println!("  still      {} (t={time}s)", out.display());
    }

    if options.video {
        let out = options.out_dir.join("video.mp4");
        let copied = export::video::export(source, &out, &info, options)
            .with_context(|| format!("writing the video of {}", source.display()))?;
        let how = if copied { "stream copy" } else { "re-encoded" };
        println!("  video      {} ({how})", out.display());
    }

    Ok(())
}

/// Scene wallpapers composite their layers, running the effect chain over
/// them where the scene fits the shape `scene::render` handles.
///
/// Still video-less: the effect chain gives one frame at a time, not a loop,
/// so `--frame` picks which moment `g_Time` sees (the first one given; a
/// second still would need a second run) and `--png-only` is moot — there was
/// never a video path here. Anything the still cannot represent is listed
/// rather than dropped.
fn export_scene(project: &Project, options: &Options) -> Result<()> {
    let package = project::require_package(project)?;
    let mut archive = pkg::Archive::open(package)?;
    let scene = scene::load(&mut archive)?;

    let time = options.frames.first().copied().unwrap_or(0.0);
    let composite = scene::render::render_frame(&mut archive, &scene, options.resolution, time)?;

    let out = options.out_dir.join("still.png");
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    composite
        .image
        .save(&out)
        .with_context(|| format!("writing {}", out.display()))?;

    println!(
        "  still      {} ({}x{})",
        out.display(),
        composite.image.width(),
        composite.image.height()
    );

    if !composite.omissions.is_empty() {
        println!("  not rendered:");
        for note in &composite.omissions {
            println!("    - {note}");
        }
    }
    Ok(())
}

fn run_export(args: &ExportArgs) -> Result<()> {
    let project = project::load(&args.wallpaper)?;

    println!("{}", project::display_name(&project));
    println!("  type       {}", project.kind);

    if let Some(reason) = unsupported_reason(&project) {
        bail!(
            "cannot export {}: {reason}",
            project::display_name(&project)
        );
    }

    // Each wallpaper gets its own named folder rather than dumping stem-named
    // files into a shared one: the title is what the user recognises the
    // wallpaper by, and it is what `info` already shows first.
    let name = sanitize_component(project::display_name(&project));
    // `--out .` is the default, and `Path::join` would otherwise print every
    // path with a `./` nobody asked for.
    let out_dir = if args.out == Path::new(".") {
        PathBuf::from(&name)
    } else {
        args.out.join(&name)
    };
    println!("  output     {}", out_dir.display());

    let options = Options {
        out_dir,
        video: !args.png_only,
        frames: args.frames.clone(),
        resolution: args.resolution,
        fps: args.fps,
        duration: args.duration,
        audio: args.audio,
    };

    match project.kind {
        Kind::Video => export_video(&project, &options),
        Kind::Scene => export_scene(&project, &options),
        // Every other type was rejected above; this stays exhaustive so a new
        // pipeline cannot be added to the router without being wired in here.
        _ => unreachable!("unsupported types are rejected before dispatch"),
    }
}

/// Open a live window playing a scene wallpaper's effect chain in real time.
fn run_simulate(args: &SimulateArgs) -> Result<()> {
    let project = project::load(&args.wallpaper)?;
    if project.kind != Kind::Scene {
        bail!("only Scene wallpapers can be simulated (this one is {})", project.kind);
    }
    if let Some(reason) = unsupported_reason(&project) {
        bail!("cannot simulate {}: {reason}", project::display_name(&project));
    }

    let package = project::require_package(&project)?;
    let mut archive = pkg::Archive::open(package)?;
    let scene = scene::load(&mut archive)?;

    let title = project::display_name(&project);
    println!("{title}");
    simulate::run(&mut archive, &scene, title)
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Export(args) => run_export(&args),
        Command::Info(args) => run_info(&args),
        Command::Unpack(args) => run_unpack(&args),
        Command::Tex(args) => run_tex(&args),
        Command::Shaders(args) => run_shaders(&args),
        Command::Simulate(args) => run_simulate(&args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn frame_can_be_repeated() {
        use clap::Parser;
        let cli = Cli::parse_from([
            "wallpaper-engine",
            "export",
            "wp",
            "--frame",
            "0",
            "--frame",
            "5.5",
        ]);
        let Command::Export(args) = cli.command else {
            panic!("expected the export subcommand");
        };
        assert_eq!(args.frames, vec![0.0, 5.5]);
    }

    #[test]
    fn hostile_title_characters_become_dashes() {
        // Real Workshop titles carry exactly this kind of punctuation.
        assert_eq!(
            sanitize_component("[2k] Code 81800 | Biboo Outro"),
            "[2k] Code 81800 - Biboo Outro"
        );
        assert_eq!(sanitize_component("a/b\\c:d*e?f\"g<h>i"), "a-b-c-d-e-f-g-h-i");
    }

    #[test]
    fn non_ascii_titles_pass_through_unchanged() {
        // CJK and other scripts are exactly what makes a folder recognisable;
        // only the filesystem-hostile ASCII punctuation is replaced.
        assert_eq!(sanitize_component("亚托莉 [アトリ] 8K"), "亚托莉 [アトリ] 8K");
    }

    #[test]
    fn an_empty_or_dot_only_title_falls_back() {
        // Both are what's left after Windows-illegal trailing dots/spaces are
        // trimmed; a title of pure path separators is not this case — `-` is
        // a perfectly legal directory name character, so `///` becomes `---`.
        assert_eq!(sanitize_component(""), "wallpaper");
        assert_eq!(sanitize_component("..."), "wallpaper");
        assert_eq!(sanitize_component("  "), "wallpaper");
    }

    #[test]
    fn a_single_frame_gets_the_plain_still_name() {
        assert_eq!(still_name(3.0, false), "still.png");
    }

    #[test]
    fn multiple_frames_are_disambiguated_by_timestamp() {
        assert_eq!(still_name(0.0, true), "still_t0s.png");
        assert_eq!(still_name(5.5, true), "still_t5.5s.png");
    }
}

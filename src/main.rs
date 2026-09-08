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
mod scene;
mod shader;
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

    /// Directory to write the exported files into.
    #[arg(short, long, default_value = "export")]
    out: PathBuf,

    /// Write only the still PNG.
    #[arg(long, conflicts_with = "video_only")]
    png_only: bool,

    /// Write only the looping video.
    #[arg(long, conflicts_with = "png_only")]
    video_only: bool,

    /// Output size as `WIDTHxHEIGHT`. Defaults to the source resolution.
    #[arg(long)]
    resolution: Option<Resolution>,

    /// Output frame rate. Defaults to the source rate.
    #[arg(long)]
    fps: Option<f64>,

    /// Trim the video to this many seconds.
    #[arg(long)]
    duration: Option<f64>,

    /// Timestamp in seconds for the still frame.
    #[arg(long, default_value_t = 0.0)]
    time: f64,

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

/// A filesystem-safe stem for the exported files.
///
/// Titles are unusable here: real ones contain `/`, `|`, brackets and CJK.
/// The wallpaper's own directory name is already a valid filename and is what
/// the user recognises it by.
fn output_stem(project: &Project) -> String {
    project
        .root
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != ".")
        .unwrap_or("wallpaper")
        .to_string()
}

/// Explain why a wallpaper cannot be exported, or `None` if it can.
///
/// Kept separate from the export itself so `info` and `export` agree, and so
/// the reasons stay in one readable list as pipelines land.
fn unsupported_reason(project: &Project) -> Option<String> {
    match &project.kind {
        // Scene is exportable as a still only: the layers composite, but the
        // effect shaders that animate them do not run yet.
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
    // knowing before asking for a 33-megapixel still.
    if project.kind == Kind::Scene
        && let Some(package) = project.package.as_deref()
        && let Ok(mut archive) = pkg::Archive::open(package)
        && let Ok(scene) = scene::load(&mut archive)
    {
        let images = scene.objects.iter().filter(|o| model::is_image(o)).count();
        let particles = scene.objects.iter().filter(|o| model::is_particle(o)).count();
        let effects: usize = scene
            .objects
            .iter()
            .map(|o| model::visible_effects(o).count())
            .sum();

        if let Some(ortho) = scene.general.orthographic {
            println!("  canvas     {}x{}", ortho.width, ortho.height);
        }
        println!("  layers     {images} image, {particles} particle, {effects} effect(s)");
    }

    match unsupported_reason(&project) {
        Some(reason) => println!("  export     no: {reason}"),
        None if project.kind == Kind::Scene => {
            println!("  export     still only (effects and particles are not rendered)");
        }
        None => println!("  export     yes"),
    }
    Ok(())
}

/// Video wallpapers ship a finished looping file; exporting is packaging.
fn export_video(project: &Project, stem: &str, options: &Options) -> Result<()> {
    let source = project::require_entry(project)?;
    export::ffmpeg::init()?;

    let info = export::ffmpeg::probe(source)?;
    println!(
        "  source     {}x{} {:.3} fps, {:.2}s, {}",
        info.width, info.height, info.fps, info.duration, info.codec
    );

    if options.still {
        let out = options.out_dir.join(format!("{stem}.png"));
        let size = options
            .resolution
            .map(|resolution| (resolution.width, resolution.height));
        export::still::from_video(source, &out, options.still_time, size)
            .with_context(|| format!("writing the still frame of {}", source.display()))?;
        println!("  still      {}", out.display());
    }

    if options.video {
        let out = options.out_dir.join(format!("{stem}.mp4"));
        let copied = export::video::export(source, &out, &info, options)
            .with_context(|| format!("writing the video of {}", source.display()))?;
        let how = if copied { "stream copy" } else { "re-encoded" };
        println!("  video      {} ({how})", out.display());
    }

    Ok(())
}

/// Scene wallpapers composite their layers into a still.
///
/// No video yet: the motion lives in the effect shaders, so a video of the
/// base composite would be a still repeated, which is worse than not offering
/// one. Anything the composite cannot represent is listed rather than dropped.
fn export_scene(project: &Project, stem: &str, options: &Options) -> Result<()> {
    if !options.still {
        bail!("scene wallpapers export a still only; --video-only has nothing to produce");
    }
    let package = project::require_package(project)?;
    let mut archive = pkg::Archive::open(package)?;
    let scene = scene::load(&mut archive)?;

    let composite = scene::compose::render(&mut archive, &scene, options.resolution)?;

    let out = options.out_dir.join(format!("{stem}.png"));
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

    let options = Options {
        out_dir: args.out.clone(),
        still: !args.video_only,
        video: !args.png_only,
        resolution: args.resolution,
        fps: args.fps,
        duration: args.duration,
        still_time: args.time,
        audio: args.audio,
    };
    let stem = output_stem(&project);

    match project.kind {
        Kind::Video => export_video(&project, &stem, &options),
        Kind::Scene => export_scene(&project, &stem, &options),
        // Every other type was rejected above; this stays exhaustive so a new
        // pipeline cannot be added to the router without being wired in here.
        _ => unreachable!("unsupported types are rejected before dispatch"),
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Export(args) => run_export(&args),
        Command::Info(args) => run_info(&args),
        Command::Unpack(args) => run_unpack(&args),
        Command::Tex(args) => run_tex(&args),
        Command::Shaders(args) => run_shaders(&args),
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
}

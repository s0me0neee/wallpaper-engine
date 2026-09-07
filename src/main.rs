//! Wallpaper Engine scene analyzer: unpacks `.pkg` archives and decodes the
//! `.tex` textures inside them.

mod pkg;
mod reader;
mod tex;

use anyhow::{Result, bail};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use walkdir::WalkDir;

/// Unpack and inspect Wallpaper Engine scene packages.
///
/// With no subcommand, runs the whole pipeline with default paths:
/// content/scene.pkg -> unpacked/ -> textures/
#[derive(Parser)]
#[command(name = "wallpaper-engine", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Extract the contents of a .pkg archive.
    Unpack(UnpackArgs),
    /// Decode .tex textures to PNG.
    Tex(TexArgs),
}

#[derive(Args)]
struct UnpackArgs {
    /// Archive to read.
    #[arg(default_value = "content/scene.pkg")]
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

impl Default for UnpackArgs {
    fn default() -> Self {
        Self {
            pkg: PathBuf::from("content/scene.pkg"),
            out: PathBuf::from("unpacked"),
            list: false,
        }
    }
}

impl Default for TexArgs {
    fn default() -> Self {
        Self {
            paths: vec![PathBuf::from("unpacked")],
            out: PathBuf::from("textures"),
            info: false,
            all_mipmaps: false,
        }
    }
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

fn main() -> Result<()> {
    match Cli::parse().command {
        Some(Command::Unpack(args)) => run_unpack(&args),
        Some(Command::Tex(args)) => run_tex(&args),
        // No subcommand: run the whole pipeline with default paths.
        None => {
            let unpack_args = UnpackArgs::default();
            run_unpack(&unpack_args)?;
            println!();
            run_tex(&TexArgs {
                paths: vec![unpack_args.out.clone()],
                ..TexArgs::default()
            })
        }
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

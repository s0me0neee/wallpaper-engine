//! Unpacker for Wallpaper Engine `.pkg` scene archives.
//!
//! Layout (little-endian throughout):
//!
//! ```text
//! int32   len(version)      // e.g. 8
//! bytes   version           // e.g. "PKGV0023"
//! int32   entry_count
//! repeat entry_count times:
//!     int32 len(path)
//!     bytes path            // utf-8, '/' separated
//!     int32 offset          // relative to the start of the data blob
//!     int32 length
//! bytes   data_blob         // entries are slices of this
//! ```
//!
//! Some very old packages have no version string; there the first int32 is the
//! entry count itself. We detect that by checking for a "PKGV" prefix.

use crate::reader::Reader;
use anyhow::{Context, Result, bail};
use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::{Component, Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct Entry {
    pub path: String,
    pub offset: u32,
    pub length: u32,
}

pub struct Package {
    pub version: String,
    pub entries: Vec<Entry>,
    /// Absolute offset that entry offsets are measured from.
    pub blob_start: u64,
}

/// Parse the archive header, leaving `reader` positioned at the data blob.
pub fn read_header<R: Read + io::Seek>(reader: &mut Reader<R>) -> Result<Package> {
    let version = reader.string()?;

    let (version, entry_count) = if version.starts_with("PKGV") {
        let count = reader.i32()?;
        (version, count)
    } else {
        // No version string: what we just read was the first entry's path, so
        // rewind and treat the leading int32 as the entry count.
        reader.seek_to(0)?;
        (String::new(), reader.i32()?)
    };

    if !(0..=1_000_000).contains(&entry_count) {
        bail!("implausible entry count {entry_count}");
    }

    let mut entries = Vec::with_capacity(entry_count as usize);
    for index in 0..entry_count {
        let path = reader.string()?;
        let offset = reader.i32()?;
        let length = reader.i32()?;
        if offset < 0 || length < 0 {
            bail!("entry {index} ({path}) has negative offset/length");
        }
        entries.push(Entry {
            path,
            offset: offset as u32,
            length: length as u32,
        });
    }

    Ok(Package {
        version,
        entries,
        blob_start: reader.pos(),
    })
}

/// Resolve an archive path under `root`, refusing anything that escapes it.
fn sanitize(path: &str, root: &Path) -> Result<PathBuf> {
    let normalized = path.replace('\\', "/");
    let mut out = root.to_path_buf();

    for component in Path::new(&normalized).components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => bail!("entry escapes output directory: {path:?}"),
        }
    }

    if out == root {
        bail!("entry has an empty path");
    }
    Ok(out)
}

pub fn unpack(pkg_path: &Path, out_dir: &Path, list_only: bool) -> Result<()> {
    let file = File::open(pkg_path)
        .with_context(|| format!("opening {}", pkg_path.display()))?;
    let total = file.metadata()?.len();

    let mut reader = Reader::new(BufReader::new(file));
    let package = read_header(&mut reader)?;
    let blob_start = package.blob_start;

    let label = if package.version.is_empty() {
        "<no version>"
    } else {
        &package.version
    };
    println!(
        "{}: {label}, {} entries, data blob at {blob_start:#x}",
        pkg_path.display(),
        package.entries.len()
    );

    // Reading in offset order keeps the extraction close to sequential.
    let mut ordered: Vec<&Entry> = package.entries.iter().collect();
    ordered.sort_by_key(|entry| entry.offset);

    let mut inner = reader.into_inner();

    for entry in ordered {
        let start = blob_start + entry.offset as u64;
        let end = start + entry.length as u64;
        if end > total {
            bail!(
                "entry {:?} runs past end of file ({end} > {total})",
                entry.path
            );
        }

        println!("  {:>12}  {}", entry.length, entry.path);
        if list_only {
            continue;
        }

        let target = sanitize(&entry.path, out_dir)?;
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut reader = Reader::new(&mut inner);
        reader.seek_to(start)?;
        let mut chunk = reader.into_inner().take(entry.length as u64);
        let mut out = File::create(&target)
            .with_context(|| format!("creating {}", target.display()))?;
        io::copy(&mut chunk, &mut out)?;
    }

    if !list_only {
        println!("\nextracted to {}/", out_dir.display());
    }
    Ok(())
}

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

use crate::paths::resolve_under;
use crate::reader::Reader;
use anyhow::{Context, Result, bail};
use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
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

    #[expect(clippy::cast_sign_loss, reason = "just bounded to 0..=1_000_000 above")]
    let mut entries = Vec::with_capacity(entry_count as usize);
    for index in 0..entry_count {
        let path = reader.string()?;
        let offset = reader.i32()?;
        let length = reader.i32()?;
        if offset < 0 || length < 0 {
            bail!("entry {index} ({path}) has negative offset/length");
        }
        #[expect(clippy::cast_sign_loss, reason = "just checked non-negative above")]
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

/// An open archive that can serve individual entries without extracting.
///
/// The scene renderer needs a handful of files out of a package that is often
/// most of a gigabyte, so it reads them on demand rather than unpacking the
/// whole thing to a temporary directory first.
pub struct Archive {
    file: BufReader<File>,
    blob_start: u64,
    total: u64,
    index: HashMap<String, Entry>,
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let total = file.metadata()?.len();

        let mut reader = Reader::new(BufReader::new(file));
        let package = read_header(&mut reader)?;
        let blob_start = package.blob_start;

        // Archive paths use '/' but Wallpaper Engine's own references are not
        // consistent about separators, so the index is keyed on a normalized
        // form and looked up the same way.
        let index = package
            .entries
            .into_iter()
            .map(|entry| (normalize(&entry.path), entry))
            .collect();

        Ok(Archive {
            file: reader.into_inner(),
            blob_start,
            total,
            index,
        })
    }

    /// Every entry path in the archive, in no particular order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.index.values().map(|entry| entry.path.as_str())
    }

    /// Whether the archive holds an entry, without reading it.
    pub fn contains(&self, path: &str) -> bool {
        self.index.contains_key(&normalize(path))
    }

    /// Read one entry into memory.
    pub fn read(&mut self, path: &str) -> Result<Vec<u8>> {
        let entry = self
            .index
            .get(&normalize(path))
            .with_context(|| format!("{path:?} is not in the package"))?;

        let start = self.blob_start + u64::from(entry.offset);
        let end = start + u64::from(entry.length);
        if end > self.total {
            bail!(
                "entry {:?} runs past end of file ({end} > {})",
                entry.path,
                self.total
            );
        }

        let length = entry.length as usize;
        let mut reader = Reader::new(&mut self.file);
        reader.seek_to(start)?;
        let mut buffer = vec![0u8; length];
        reader.into_inner().read_exact(&mut buffer)?;
        Ok(buffer)
    }
}

/// Key archive paths on separator and case, both of which vary in the wild.
fn normalize(path: &str) -> String {
    path.replace('\\', "/").to_lowercase()
}

/// Extract every entry into `out_dir`, or just list them.
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
        let start = blob_start + u64::from(entry.offset);
        let end = start + u64::from(entry.length);
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

        let target = resolve_under(out_dir, &entry.path)?;
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut reader = Reader::new(&mut inner);
        reader.seek_to(start)?;
        let mut chunk = reader.into_inner().take(u64::from(entry.length));
        let mut out = File::create(&target)
            .with_context(|| format!("creating {}", target.display()))?;
        io::copy(&mut chunk, &mut out)?;
    }

    if !list_only {
        println!("\nextracted to {}/", out_dir.display());
    }
    Ok(())
}

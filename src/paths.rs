//! Resolving untrusted relative paths against a trusted root.
//!
//! Both `.pkg` entry names and the `file`/`preview` fields of `project.json`
//! come from Workshop content, so neither may be joined onto a local path
//! without checking. A single `..` component is the whole attack.

use anyhow::{Result, bail};
use std::path::{Component, Path, PathBuf};

/// Join `path` onto `root`, refusing anything that could escape it.
///
/// Wallpaper Engine is a Windows program and writes `\` separators, so those
/// are normalized first. Only plain name components survive: `..`, a root
/// prefix and a Windows drive letter are all rejected rather than stripped,
/// because silently rewriting a path is how a traversal check gets bypassed.
pub fn resolve_under(root: &Path, path: &str) -> Result<PathBuf> {
    let normalized = path.replace('\\', "/");
    let mut out = root.to_path_buf();

    for component in Path::new(&normalized).components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => bail!("path escapes the wallpaper directory: {path:?}"),
        }
    }

    if out == root {
        bail!("path is empty");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_nested_relative_path() {
        let resolved = resolve_under(Path::new("/root"), "materials/masks/a.tex").unwrap();
        assert_eq!(resolved, Path::new("/root/materials/masks/a.tex"));
    }

    #[test]
    fn normalizes_windows_separators() {
        let resolved = resolve_under(Path::new("/root"), "shaders\\effects\\shake.frag").unwrap();
        assert_eq!(resolved, Path::new("/root/shaders/effects/shake.frag"));
    }

    #[test]
    fn rejects_traversal_and_absolute_paths() {
        for hostile in ["../etc/passwd", "a/../../b", "/etc/passwd", "\\windows\\x"] {
            assert!(
                resolve_under(Path::new("/root"), hostile).is_err(),
                "should have rejected {hostile:?}"
            );
        }
    }

    #[test]
    fn rejects_an_empty_path() {
        assert!(resolve_under(Path::new("/root"), "").is_err());
        assert!(resolve_under(Path::new("/root"), "./").is_err());
    }
}

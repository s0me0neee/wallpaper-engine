//! Our replacement for the headers Wallpaper Engine ships with the program.
//!
//! Shaders `#include "common.h"` and friends, but those files live in the
//! Wallpaper Engine install, not in any wallpaper package — so a package alone
//! cannot be compiled. They are also not ours to redistribute, hence a shim
//! written from the observed usage rather than a copy.
//!
//! Scope is set by measurement, not guesswork. Across the sample corpus's 1522
//! lines of shader, exactly fourteen identifiers are called but neither
//! declared locally nor part of GLSL:
//!
//! ```text
//! texSample2D 48   CAST4 18   rotateVec2 18   mul 13   CAST2 10
//! saturate 10      frac 4     ApplyBlending 4  atan2 2  CAST3 1
//! blur3a/7a/13a 2 each        squareToQuad 1
//! ```
//!
//! Everything here is a mechanical HLSL→GLSL alias, because Wallpaper Engine
//! compiles the same source for both D3D and OpenGL.
//!
//! The headers live beside this file as real `.glsl` sources rather than
//! string literals, so they are editable as shader code. They keep a `.glsl`
//! extension on disk and their `.h` include name in the table below — nothing
//! requires the two to match, and `.h` makes editors parse them as C.

use anyhow::{Context, Result, bail};
use std::{collections::HashMap, path::Path};

/// The headers we supply, keyed by the name a shader includes them under.
pub fn headers() -> HashMap<String, String> {
    [
        ("common.h", include_str!("glsl/common.glsl")),
        ("common_fragment.h", include_str!("glsl/common_fragment.glsl")),
        ("common_blending.h", include_str!("glsl/common_blending.glsl")),
        ("common_blur.h", include_str!("glsl/common_blur.glsl")),
        (
            "common_perspective.h",
            include_str!("glsl/common_perspective.glsl"),
        ),
    ]
    .into_iter()
    .map(|(name, text)| (name.to_string(), text.to_string()))
    .collect()
}

/// Replace shim headers with the real ones from a Wallpaper Engine install.
///
/// The escape hatch for effects whose helpers the shim gets wrong, and the
/// reference side of a golden-image comparison. These files are read from the
/// user's own install at run time and never copied into this repository or
/// into any output.
pub fn headers_from_install(root: &Path) -> Result<HashMap<String, String>> {
    // Wallpaper Engine keeps them under `shaders/` inside its install.
    let directory = if root.join("shaders").is_dir() {
        root.join("shaders")
    } else {
        root.to_path_buf()
    };

    let mut headers = headers();
    let names: Vec<String> = headers.keys().cloned().collect();
    let mut found = 0_usize;
    for name in names {
        let path = directory.join(&name);
        if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            headers.insert(name, text);
            found = found.saturating_add(1);
        }
    }

    if found == 0 {
        bail!(
            "no common*.h found under {} — point --we-assets at a Wallpaper Engine install",
            directory.display()
        );
    }
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::preprocess::{Stage, build};
    use std::collections::BTreeMap;

    fn expand(source: &str, stage: Stage) -> String {
        build(source, stage, &headers(), &BTreeMap::new()).unwrap()
    }

    #[test]
    fn every_header_the_corpus_includes_is_present() {
        let headers = headers();
        for name in [
            "common.h",
            "common_fragment.h",
            "common_blending.h",
            "common_blur.h",
            "common_perspective.h",
        ] {
            assert!(headers.contains_key(name), "missing {name}");
        }
    }

    #[test]
    fn mul_puts_the_matrix_on_the_left() {
        // The trap flagged in plan.md. HLSL's `mul(v, M)` is row-vector, so
        // the GLSL spelling is `M * v` — the operands swap. Getting this
        // backwards transposes every transform and still compiles.
        let expanded = expand("void main() {}\n", Stage::Vertex);
        assert!(
            expanded.contains("#define mul(a, b) ((b) * (a))"),
            "operands not swapped: the matrix must end up on the left"
        );
    }

    #[test]
    fn the_shim_supplies_every_identifier_the_corpus_needs() {
        let all: String = headers().values().cloned().collect();
        for name in [
            "texSample2D",
            "CAST2",
            "CAST3",
            "CAST4",
            "mul",
            "saturate",
            "frac",
            "atan2",
            "rotateVec2",
            "ApplyBlending",
            "blur3a",
            "blur7a",
            "blur13a",
            "squareToQuad",
            "M_PI",
            "M_PI_2",
        ] {
            assert!(all.contains(name), "shim does not define {name}");
        }
    }

    #[test]
    fn blending_takes_its_mode_as_an_argument() {
        // Every call site is `ApplyBlending(BLENDMODE, base, blend, alpha)`,
        // so the mode has to be a parameter, not a preprocessor branch.
        let expanded = expand("#include \"common_blending.h\"\nvoid main() {}\n", Stage::Fragment);
        assert!(
            expanded.contains("vec3 ApplyBlending(int mode, vec3 base, vec3 blend, float alpha)"),
            "wrong ApplyBlending signature"
        );
    }

    #[test]
    fn the_blur_helpers_bind_their_sampler_at_the_call_site() {
        // Call sites are `blur13a(v_TexCoord.xy, v_TexCoord.zw)` with no
        // sampler. Binding g_Texture0 via a macro rather than inside the
        // function body is what lets one shader include this header above its
        // own g_Texture0 declaration.
        let expanded = expand("#include \"common_blur.h\"\nvoid main() {}\n", Stage::Fragment);
        assert!(expanded.contains("#define blur13a(uv, direction) weBlur13a(g_Texture0,"));
        assert!(expanded.contains("vec4 weBlur13a(sampler2D image, vec2 uv, vec2 direction)"));
    }

    #[test]
    fn the_shim_contains_no_pre_core_spellings() {
        // The dialect rewrite runs over the shim too, so a `varying` or
        // `gl_FragColor` in here would be silently mangled per stage.
        let all: String = headers().values().cloned().collect();
        assert!(!all.contains("varying"));
        assert!(!all.contains("attribute"));
        assert!(!all.contains("gl_FragColor"));
    }

    #[test]
    fn every_header_is_guarded_against_double_inclusion() {
        // common.h is prepended as well as included, and headers include each
        // other; without guards the GL compiler sees duplicate definitions.
        for (name, text) in headers() {
            assert!(
                text.contains("#ifndef WE_") && text.contains("#endif"),
                "{name} has no include guard"
            );
        }
    }
}

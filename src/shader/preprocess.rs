//! Turning a packaged Wallpaper Engine shader into something a GL driver
//! will compile.
//!
//! Deliberately small. The driver's own GLSL compiler evaluates `#if`,
//! `#ifdef` and `#define` — the 122 conditionals in the sample corpus are its
//! job, not ours — so all that is left is:
//!
//! 1. expand `#include`, which desktop GL has no support for;
//! 2. inject the combo values chosen for this pass as `#define`s;
//! 3. rewrite the GLSL 1.20-era spellings the shaders are written in.
//!
//! Include expansion uses `glsl-include`, which exists for exactly this.

use super::{annotations, hlsl};
use anyhow::{Context, Result};
use glsl_include::Context as Includer;
use regex::Regex;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write as _,
    sync::{Mutex, OnceLock, PoisonError},
};

/// Which stage a shader is for. Decides how `varying` is rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
}

/// GLSL version the output targets.
///
/// 330 rather than something newer on purpose: `sample` became a reserved
/// word in GLSL 4.00, and five shaders in the corpus use it as an ordinary
/// local variable. 330 core is available everywhere desktop GL is.
const VERSION: &str = "#version 330 core";

/// Build the compilable source for one shader.
///
/// `headers` is the shim; `selected` are the combo values this pass chose.
///
/// Every combo the shader declares gets a `#define`, whether the pass selected
/// it or not. C would treat an undefined name in `#if` as zero, but the GL
/// driver rejects `#if MASK == 1` outright when `MASK` has no definition — so
/// the shader's own `[COMBO]` defaults have to be emitted as a baseline.
pub fn build(
    source: &str,
    stage: Stage,
    headers: &HashMap<String, String>,
    selected: &BTreeMap<String, i64>,
) -> Result<String> {
    build_against(source, stage, headers, selected, None)
}

/// As `build`, but with the other stage's source in hand, so a `varying` the
/// two stages declare differently can be reconciled before the linker sees it.
pub fn build_against(
    source: &str,
    stage: Stage,
    headers: &HashMap<String, String>,
    selected: &BTreeMap<String, i64>,
    counterpart: Option<&str>,
) -> Result<String> {
    let declared = annotations::parse(source);
    let mut combos = annotations::combo_defaults(&declared);

    // Anything the shader branches on but never declares still needs a value —
    // unless a shim header defines it. `lightshafts` asks
    // `#if TEX2FORMAT == FORMAT_R8`, and giving both names a zero stand-in both
    // redefines the header's macro and makes the test true for every texture.
    let from_shim = shim_macros(headers);
    for name in annotations::undeclared_conditionals(source) {
        if !from_shim.contains(&name) {
            combos.entry(name).or_insert(0);
        }
    }
    combos.extend(selected.iter().map(|(name, value)| (name.clone(), *value)));
    let combos = &combos;

    let mut includer = Includer::new();
    for (name, text) in headers {
        includer.include(name.clone(), text.clone());
    }

    // `common.h` is prepended as well as included: several shaders call `mul`
    // and `saturate` without including anything, so Wallpaper Engine must
    // prepend it too. The header's include guard makes the double safe.
    let common = headers
        .get("common.h")
        .context("the shim is missing common.h")?;
    let combined = format!("{common}\n{source}");

    let expanded = includer
        .expand(combined)
        .map_err(|error| anyhow::anyhow!("expanding #include: {error}"))?;

    let body = balance_conditionals(&strip_line_directives(&expanded));
    let body = match (stage, counterpart) {
        (Stage::Fragment, Some(vertex)) => reconcile_varyings(&body, &varying_types(vertex)),
        _ => body,
    };
    Ok(assemble(&body, stage, combos))
}

/// Drop `#endif`s with nothing open, and close anything left open at the end.
///
/// Wallpaper Engine resolves `#if` itself before handing a shader to the
/// driver, so a stray directive is simply consumed; we leave conditionals to
/// the driver (plan.md §6), which rejects the file outright. `scene_example5`'s
/// `iris_movement__.vert` carries one `#endif` too many and is otherwise fine,
/// and dropping it is what a tolerant preprocessor does — a source this touches
/// was already invalid, so there is no correct build to break.
fn balance_conditionals(source: &str) -> String {
    let mut depth = 0_usize;
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let directive = line.trim_start().strip_prefix('#').map(str::trim_start);
        match directive {
            Some(rest) if rest.starts_with("if") => depth = depth.saturating_add(1),
            Some(rest) if rest.starts_with("endif") => {
                if depth == 0 {
                    continue; // nothing to close; the driver would stop here.
                }
                depth -= 1;
            }
            _ => {}
        }
        out.push_str(line);
        out.push('\n');
    }
    for _ in 0..depth {
        out.push_str("#endif\n");
    }
    out
}

/// `varying <type> <name>;` declarations, by name.
fn varying_types(source: &str) -> BTreeMap<String, String> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        #[expect(clippy::unwrap_used, reason = "a fixed literal pattern, checked by the tests below")]
        Regex::new(r"(?m)^\s*varying\s+(float|vec2|vec3|vec4)\s+([A-Za-z_]\w*)\s*;").unwrap()
    });
    pattern
        .captures_iter(source)
        .map(|caps| (caps[2].to_string(), caps[1].to_string()))
        .collect()
}

fn components(kind: &str) -> usize {
    match kind {
        "vec2" => 2,
        "vec3" => 3,
        "vec4" => 4,
        _ => 1,
    }
}

/// Make the fragment stage's `varying` declarations agree with the vertex
/// stage's, where the shader itself disagrees.
///
/// `varying` was matched loosely enough that a shader could declare
/// `vec4 v_TexCoord` in one stage and `vec2` in the other and still run —
/// `scene_example6`'s `color_grading` does exactly that. Rewritten to
/// `#version 330 core`'s `in`/`out` the linker rejects it outright ("differs in
/// type/qualifiers"), so the declaration takes the vertex's type — it is the
/// one that writes — and the body reads it through a `#define` that restores
/// the width the fragment expected.
fn reconcile_varyings(body: &str, produced: &BTreeMap<String, String>) -> String {
    let mut body = body.to_string();
    for (name, declared) in varying_types(&body) {
        let Some(written) = produced.get(&name) else { continue };
        if *written == declared {
            continue;
        }

        let alias = format!("we_{name}");
        body = replace_word(&body, &name, &alias);

        let (from, to) = (components(written), components(&declared));
        let read = if to < from {
            format!("{name}.{}", &"xyzw"[..to])
        } else {
            let padding = ", 0.0".repeat(to - from);
            format!("{declared}({name}{padding})")
        };

        #[expect(clippy::unwrap_used, reason = "a fixed literal pattern built from an identifier")]
        let declaration =
            Regex::new(&format!(r"(?m)^[ \t]*varying\s+{declared}\s+{alias}\s*;")).unwrap();
        body = declaration
            .replace(&body, format!("in {written} {name};\n#define {alias} {read}").as_str())
            .into_owned();
    }
    body
}

/// Every macro name the shim headers define, function-like ones included.
fn shim_macros(headers: &HashMap<String, String>) -> HashSet<String> {
    headers
        .values()
        .flat_map(|text| text.lines())
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("#define")?;
            let name: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// Drop the `#line` directives include expansion leaves behind.
///
/// They are meant to make compiler errors point back at the original file, but
/// the macOS GL compiler rejects the two-argument `#line N M` form outright —
/// and because it renumbers first, every later error is reported against a
/// line that has nothing wrong with it. Removing them costs accurate line
/// attribution in errors and buys shaders that compile at all.
fn strip_line_directives(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("#line"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Prepend the version and defines, and rewrite the dialect.
fn assemble(body: &str, stage: Stage, combos: &BTreeMap<String, i64>) -> String {
    let varying = match stage {
        Stage::Vertex => "out",
        Stage::Fragment => "in",
    };

    let mut body = replace_word(body, "attribute", "in");
    body = replace_word(&body, "varying", varying);

    // The combo defines lead the body rather than the output, because a combo
    // is often the only thing that types an operand: `Simple_Audio_Bars` writes
    // `frequency % RESOLUTION`, and with `RESOLUTION` unknown the modulo
    // relaxation disables itself and the whole audio-bars chain stops compiling.
    let mut prefix = String::with_capacity(512);
    for (name, value) in combos {
        // Infallible: writing into a String never fails.
        let _ = writeln!(prefix, "#define {name} {value}");
    }

    if stage == Stage::Fragment && body.contains("gl_FragColor") {
        prefix.push_str("out vec4 we_FragColor;\n");
        body = replace_word(&body, "gl_FragColor", "we_FragColor");
    }

    let mut out = String::with_capacity(body.len().saturating_add(512));
    out.push_str(VERSION);
    out.push('\n');
    // Last, so the relaxations see the final spelling of every declaration.
    out.push_str(&hlsl::relax(&(prefix + &body)));
    out
}

/// Replace whole-word occurrences only.
///
/// `varyingScale` and `attributeCount` are ordinary identifiers in this
/// corpus; a plain substring replace would corrupt them, so the match is
/// anchored on word boundaries.
fn replace_word(source: &str, from: &str, to: &str) -> String {
    static CACHE: OnceLock<Mutex<HashMap<String, Regex>>> = OnceLock::new();

    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let pattern = {
        let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
        cache
            .entry(from.to_string())
            .or_insert_with(|| {
                #[expect(
                    clippy::expect_used,
                    reason = "the pattern is a literal escaped by regex itself"
                )]
                Regex::new(&format!(r"\b{}\b", regex::escape(from)))
                    .expect("escaped literal is a valid pattern")
            })
            .clone()
    };

    pattern.replace_all(source, regex::NoExpand(to)).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::shim;

    fn combos(pairs: &[(&str, i64)]) -> BTreeMap<String, i64> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), *value))
            .collect()
    }

    #[test]
    fn rewrites_only_whole_words() {
        let source = "varying vec4 v_X;\nfloat varyingScale;\nattribute vec3 a_P;\nint attributeCount;\n";
        let out = assemble(source, Stage::Vertex, &BTreeMap::new());
        assert!(out.contains("out vec4 v_X;"));
        assert!(out.contains("float varyingScale;"));
        assert!(out.contains("in vec3 a_P;"));
        assert!(out.contains("int attributeCount;"));
    }

    #[test]
    fn varying_becomes_in_for_a_fragment_shader() {
        let out = assemble("varying vec4 v_TexCoord;\n", Stage::Fragment, &BTreeMap::new());
        assert!(out.contains("in vec4 v_TexCoord;"));
    }

    #[test]
    fn gl_fragcolor_gets_a_declared_output() {
        let out = assemble(
            "void main() { gl_FragColor = vec4(1.0); }\n",
            Stage::Fragment,
            &BTreeMap::new(),
        );
        assert!(out.contains("out vec4 we_FragColor;"));
        assert!(out.contains("we_FragColor = vec4(1.0);"));
        assert!(!out.contains("gl_FragColor"));
    }

    #[test]
    fn a_vertex_shader_gets_no_fragment_output() {
        let out = assemble("void main() {}\n", Stage::Vertex, &BTreeMap::new());
        assert!(!out.contains("we_FragColor"));
    }

    #[test]
    fn targets_330_because_sample_is_reserved_from_400() {
        // Five corpus shaders declare `vec4 sample = ...`, which stops
        // compiling at #version 400 and above.
        let out = assemble("vec4 sample;\n", Stage::Fragment, &BTreeMap::new());
        assert!(out.starts_with("#version 330 core\n"), "got {out:?}");
    }

    #[test]
    fn combos_are_emitted_as_defines_before_the_body() {
        let out = assemble("body\n", Stage::Fragment, &combos(&[("MASK", 1), ("NOISE", 0)]));
        let defines_at = out.find("#define MASK 1").unwrap();
        let body_at = out.find("body").unwrap();
        assert!(defines_at < body_at, "defines must precede the body");
        assert!(out.contains("#define NOISE 0"));
    }

    #[test]
    fn includes_are_expanded_from_the_shim() {
        let out = build(
            "#include \"common_blur.h\"\nvoid main() {}\n",
            Stage::Fragment,
            &shim::headers(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(out.contains("blur13a"), "blur header not expanded");
    }

    #[test]
    fn common_is_prepended_even_without_an_include() {
        // shake.vert uses `mul` and never includes anything.
        let out = build(
            "void main() { gl_Position = mul(p, m); }\n",
            Stage::Vertex,
            &shim::headers(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(out.contains("#define mul"), "common.h was not prepended");
    }

    #[test]
    fn a_doubly_included_header_is_guarded_against_redefinition() {
        // common.h is both prepended and #included, so its text appears twice.
        // Include expansion is purely textual — what stops the second copy
        // from redefining rotateVec2, which the GL compiler rejects outright,
        // is the header's own guard, evaluated later by the driver.
        let out = build(
            "#include \"common.h\"\nvoid main() {}\n",
            Stage::Fragment,
            &shim::headers(),
            &BTreeMap::new(),
        )
        .unwrap();

        let definitions = out.matches("vec2 rotateVec2(vec2 v, float angle)").count();
        assert!(definitions >= 1, "rotateVec2 missing entirely");
        if definitions > 1 {
            assert_eq!(
                out.matches("#ifndef WE_COMMON_H").count(),
                definitions,
                "every copy of common.h must carry its guard"
            );
        }
    }

    #[test]
    fn line_directives_are_stripped_from_expanded_output() {
        // Include expansion inserts `#line N M`, which the macOS GL compiler
        // rejects — and since it renumbers before failing, every later error
        // is then reported against an innocent line. This one cost an hour.
        let out = build(
            "#include \"common_blending.h\"\nvoid main() {}\n",
            Stage::Fragment,
            &shim::headers(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(!out.contains("#line"), "a #line directive survived");
    }

    #[test]
    fn a_conditional_the_shader_never_declares_still_gets_defined() {
        // Otherwise the driver rejects `#if AUDIOPROCESSING` outright.
        let out = build(
            "#if AUDIOPROCESSING\nfloat x;\n#endif\nvoid main() {}\n",
            Stage::Fragment,
            &shim::headers(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(out.contains("#define AUDIOPROCESSING 0"), "got {out:.400}");
    }

    #[test]
    fn a_selected_combo_overrides_the_shader_default() {
        let out = build(
            "// [COMBO] {\"combo\":\"NOISE\",\"default\":0}\nvoid main() {}\n",
            Stage::Fragment,
            &shim::headers(),
            &combos(&[("NOISE", 1)]),
        )
        .unwrap();
        assert!(out.contains("#define NOISE 1"));
        assert!(!out.contains("#define NOISE 0"));
    }

    #[test]
    fn a_missing_header_is_an_error_not_a_silent_skip() {
        let error = build(
            "#include \"nope.h\"\n",
            Stage::Fragment,
            &shim::headers(),
            &BTreeMap::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("nope.h"), "got {error}");
    }
}

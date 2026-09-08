//! Reading the JSON annotations Wallpaper Engine embeds in shader comments.
//!
//! The format is self-describing, which is the single most useful property it
//! has: a shader declares its own variants and its own parameter names, so
//! binding is data-driven and a new effect mostly works without new code.
//!
//! Two kinds of annotation appear:
//!
//! ```text
//! // [COMBO] {"combo":"NOISE","type":"options","default":0}
//! uniform float g_Speed;    // {"material":"speed","default":1,"range":[0,10]}
//! uniform sampler2D g_Tex3; // {"mode":"opacitymask","combo":"MASK"}
//! ```
//!
//! A `[COMBO]` line declares a preprocessor variant and its default. A uniform
//! annotation names the `constantshadervalues` key that feeds it, and on a
//! sampler may declare a combo that is defined when that slot is bound.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// A `[COMBO]` declaration: a preprocessor variant the shader offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Combo {
    pub name: String,
    pub default: i64,
}

/// A `uniform` declaration and whatever its trailing comment said about it.
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform {
    /// GLSL type, e.g. `float`, `vec2`, `sampler2D`.
    pub kind: String,
    /// GLSL name, e.g. `g_Speed`.
    pub name: String,
    /// The `constantshadervalues` key that supplies this uniform's value.
    pub material: Option<String>,
    /// Default value from the annotation, still as JSON.
    pub default: Option<Value>,
    /// For samplers, the combo that gets defined when this slot is bound.
    pub combo: Option<String>,
    /// The slider range Wallpaper Engine's own Properties panel would show
    /// for this uniform, e.g. `"range":[0.01, 1]` on `foliagesway`'s
    /// `g_Strength` — `(min, max)`.
    pub range: Option<(f32, f32)>,
}

/// What a shader declares about itself.
#[derive(Debug, Default)]
pub struct Declarations {
    pub combos: Vec<Combo>,
    pub uniforms: Vec<Uniform>,
}

#[derive(Deserialize)]
struct ComboAnnotation {
    combo: String,
    #[serde(default)]
    default: Option<Value>,
}

#[derive(Deserialize)]
struct UniformAnnotation {
    #[serde(default)]
    material: Option<String>,
    #[serde(default)]
    default: Option<Value>,
    #[serde(default)]
    combo: Option<String>,
    #[serde(default)]
    range: Option<(f32, f32)>,
}

/// Coerce an annotation's `default` to the integer a `#define` needs.
///
/// Combo defaults are written as numbers, but `true`/`false` and quoted
/// numbers both appear in the wild.
fn as_int(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Number(number)) => number.as_i64().unwrap_or(0),
        Some(Value::Bool(true)) => 1,
        Some(Value::String(text)) => text.trim().parse().unwrap_or(0),
        _ => 0,
    }
}

/// The JSON object in a trailing `//` comment, if there is one.
fn trailing_json(line: &str) -> Option<&str> {
    let (_, comment) = line.split_once("//")?;
    let rest = comment.trim_start();
    let rest = rest.strip_prefix("[COMBO]").unwrap_or(rest).trim_start();
    rest.starts_with('{').then_some(rest)
}

/// Parse `uniform <type> <name>` from the start of a declaration.
///
/// Returns `None` for anything that is not a plain uniform declaration —
/// uniform blocks and array declarations included, since neither is bound by
/// name in the way the parameter table needs.
fn uniform_declaration(line: &str) -> Option<(String, String)> {
    let rest = line.trim_start().strip_prefix("uniform")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }

    let declaration = rest.split("//").next()?.trim().trim_end_matches(';').trim();
    let mut words = declaration.split_whitespace();
    let kind = words.next()?.to_string();
    let name = words.next()?.to_string();
    if words.next().is_some() {
        return None;
    }

    // `g_AudioSpectrum16Left[16]` is an array; it is fed wholesale rather than
    // through the material table, so it is not a bindable parameter.
    if name.contains('[') || kind.contains('[') {
        return None;
    }
    Some((kind, name))
}

/// Read every `[COMBO]` and `uniform` annotation out of a shader.
pub fn parse(source: &str) -> Declarations {
    let mut declarations = Declarations::default();

    for line in source.lines() {
        let trimmed = line.trim_start();

        if trimmed.starts_with("//") && trimmed.contains("[COMBO]") {
            if let Some(json) = trailing_json(line)
                && let Ok(annotation) = serde_json::from_str::<ComboAnnotation>(json)
            {
                declarations.combos.push(Combo {
                    default: as_int(annotation.default.as_ref()),
                    name: annotation.combo,
                });
            }
            continue;
        }

        let Some((kind, name)) = uniform_declaration(line) else {
            continue;
        };

        let annotation = trailing_json(line)
            .and_then(|json| serde_json::from_str::<UniformAnnotation>(json).ok());

        // A sampler slot's `combo` is declared here rather than in a [COMBO]
        // line, and is what MASK/TIMEOFFSET and friends key off.
        if let Some(combo) = annotation.as_ref().and_then(|a| a.combo.clone())
            && !declarations.combos.iter().any(|c| c.name == combo)
        {
            declarations.combos.push(Combo { name: combo, default: 0 });
        }

        declarations.uniforms.push(Uniform {
            kind,
            name,
            material: annotation.as_ref().and_then(|a| a.material.clone()),
            default: annotation.as_ref().and_then(|a| a.default.clone()),
            combo: annotation.as_ref().and_then(|a| a.combo.clone()),
            range: annotation.and_then(|a| a.range),
        });
    }

    declarations
}

/// The `#define` set for a shader, before any pass-specific overrides.
///
/// Every combo a shader mentions must be defined, not merely the ones a pass
/// selects: this GL driver rejects `#if MASK == 1` outright when `MASK` has no
/// definition, rather than treating it as zero the way C would.
pub fn combo_defaults(declarations: &Declarations) -> BTreeMap<String, i64> {
    declarations
        .combos
        .iter()
        .map(|combo| (combo.name.clone(), combo.default))
        .collect()
}

/// Names used in a `#if`/`#elif` that the shader never defines for itself.
///
/// Annotations are not a complete list of what a shader tests. `shake.frag`
/// branches on `AUDIOPROCESSING`, which is only declared in `shake.vert` — the
/// pair shares combos, but each file is compiled alone. Anything left dangling
/// has to be defined to something or the shader will not compile at all.
pub fn undeclared_conditionals(source: &str) -> Vec<String> {
    let mut defined = Vec::new();
    let mut referenced = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim_start();
        let Some(body) = trimmed.strip_prefix('#') else {
            continue;
        };
        let body = body.trim_start();

        if let Some(rest) = body.strip_prefix("define") {
            if let Some(name) = first_identifier(rest) {
                defined.push(name);
            }
            continue;
        }

        // `#ifdef`/`#ifndef` test definedness, which is well defined for a
        // missing name; only the value-testing forms are a problem.
        let condition = body
            .strip_prefix("elif")
            .or_else(|| strip_if(body))
            .unwrap_or("");

        for name in identifiers(condition) {
            if name != "defined" && !referenced.contains(&name) {
                referenced.push(name);
            }
        }
    }

    referenced
        .into_iter()
        .filter(|name| !defined.contains(name))
        .collect()
}

/// `#if` but not `#ifdef` / `#ifndef`.
fn strip_if(body: &str) -> Option<&str> {
    let rest = body.strip_prefix("if")?;
    match rest.chars().next() {
        Some(c) if c.is_alphanumeric() || c == '_' => None,
        _ => Some(rest),
    }
}

fn first_identifier(text: &str) -> Option<String> {
    identifiers(text).into_iter().next()
}

/// Every identifier in a fragment of preprocessor expression.
fn identifiers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if character.is_alphanumeric() || character == '_' {
            current.push(character);
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }

    // A leading digit means a number, not a name.
    out.retain(|name| !name.starts_with(|c: char| c.is_ascii_digit()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_combo_declaration_with_its_default() {
        let source = r#"
// [COMBO] {"material":"ui_editor_properties_noise","combo":"NOISE","type":"options","default":0}
// [COMBO] {"combo":"DIRECTION","type":"options","default":2}
"#;
        let declarations = parse(source);
        assert_eq!(
            declarations.combos,
            vec![
                Combo { name: "NOISE".into(), default: 0 },
                Combo { name: "DIRECTION".into(), default: 2 },
            ]
        );
    }

    #[test]
    fn reads_a_uniform_and_the_material_key_that_feeds_it() {
        let source = r#"uniform float g_Amp; // {"material":"strength","default":0.1,"range":[0.01, 0.5]}"#;
        let uniforms = parse(source).uniforms;
        assert_eq!(uniforms.len(), 1);
        assert_eq!(uniforms[0].kind, "float");
        assert_eq!(uniforms[0].name, "g_Amp");
        assert_eq!(uniforms[0].material.as_deref(), Some("strength"));
        assert_eq!(uniforms[0].default, Some(serde_json::json!(0.1)));
    }

    #[test]
    fn an_unannotated_uniform_is_still_recorded() {
        // g_Time has no comment but is very much a uniform we must bind.
        let uniforms = parse("uniform float g_Time;\n").uniforms;
        assert_eq!(uniforms.len(), 1);
        assert_eq!(uniforms[0].name, "g_Time");
        assert!(uniforms[0].material.is_none());
    }

    #[test]
    fn a_sampler_slot_can_declare_its_own_combo() {
        // This is how MASK gets defined: not by a [COMBO] line, but by the
        // texture slot that switches it on.
        let source = r#"uniform sampler2D g_Texture3; // {"mode":"opacitymask","combo":"MASK"}"#;
        let declarations = parse(source);
        assert_eq!(declarations.uniforms[0].combo.as_deref(), Some("MASK"));
        assert_eq!(
            declarations.combos,
            vec![Combo { name: "MASK".into(), default: 0 }]
        );
    }

    #[test]
    fn a_combo_declared_twice_is_not_duplicated() {
        let source = concat!(
            "// [COMBO] {\"combo\":\"MASK\",\"default\":1}\n",
            "uniform sampler2D g_Texture3; // {\"combo\":\"MASK\"}\n",
        );
        let declarations = parse(source);
        assert_eq!(declarations.combos.len(), 1);
        // The explicit [COMBO] default wins over the sampler slot's implicit 0.
        assert_eq!(declarations.combos[0].default, 1);
    }

    #[test]
    fn array_uniforms_are_skipped() {
        // Audio spectra are fed wholesale, not through the material table.
        let uniforms = parse("uniform float g_AudioSpectrum16Left[16];\n").uniforms;
        assert_eq!(uniforms, Vec::new(), "array uniforms must be skipped");
    }

    #[test]
    fn a_word_starting_with_uniform_is_not_a_declaration() {
        assert_eq!(parse("float uniformity = 1.0;\n").uniforms, Vec::new());
        assert_eq!(parse("uniformly(x);\n").uniforms, Vec::new());
    }

    #[test]
    fn defaults_cover_every_combo_the_shader_mentions() {
        // The GL driver rejects `#if MASK == 1` when MASK is undefined, so a
        // combo that is only mentioned must still get a define.
        let source = concat!(
            "// [COMBO] {\"combo\":\"NOISE\",\"default\":1}\n",
            "uniform sampler2D g_Texture2; // {\"combo\":\"TIMEOFFSET\"}\n",
        );
        let defaults = combo_defaults(&parse(source));
        assert_eq!(defaults.get("NOISE"), Some(&1));
        assert_eq!(defaults.get("TIMEOFFSET"), Some(&0));
    }

    #[test]
    fn reads_the_real_sampler_lines_from_the_corpus() {
        // Verbatim from shake.frag, CRLF and all — the file is Windows-authored.
        let source = concat!(
            "uniform sampler2D g_Texture2; // {\"label\":\"ui_editor_properties_time_offset\",\"mode\":\"opacitymask\",\"default\":\"util/black\",\"combo\":\"TIMEOFFSET\"}\r\n",
            "uniform sampler2D g_Texture3; // {\"label\":\"ui_editor_properties_opacity\",\"mode\":\"opacitymask\",\"combo\":\"MASK\"}\r\n",
        );
        let defaults = combo_defaults(&parse(source));
        assert_eq!(defaults.get("TIMEOFFSET"), Some(&0), "got {defaults:?}");
        assert_eq!(defaults.get("MASK"), Some(&0), "got {defaults:?}");
    }

    #[test]
    fn finds_a_conditional_the_shader_never_declares() {
        // shake.frag branches on AUDIOPROCESSING but only shake.vert declares
        // it; each file is compiled alone, so it must still get a define.
        let source = "#if AUDIOPROCESSING\nx\n#endif\n";
        assert_eq!(undeclared_conditionals(source), vec!["AUDIOPROCESSING"]);
    }

    #[test]
    fn a_name_the_shader_defines_itself_is_left_alone() {
        // Redefining it would clobber the shader's own value.
        let source = "#define KERNEL 2\n#if KERNEL == 2\nx\n#endif\n";
        assert_eq!(undeclared_conditionals(source), Vec::<String>::new());
    }

    #[test]
    fn ifdef_is_not_treated_as_undeclared() {
        // `#ifdef` on a missing name is well defined; only value tests break.
        assert_eq!(undeclared_conditionals("#ifdef HLSL_SM30\nx\n#endif\n"), Vec::<String>::new());
        assert_eq!(undeclared_conditionals("#ifndef GUARD\nx\n#endif\n"), Vec::<String>::new());
    }

    #[test]
    fn collects_every_name_in_a_compound_condition() {
        let names = undeclared_conditionals("#if TEX2FORMAT == FORMAT_R8 || A\nx\n#endif\n");
        assert_eq!(names, vec!["TEX2FORMAT", "FORMAT_R8", "A"]);
    }

    #[test]
    fn numbers_and_defined_are_not_names() {
        let names = undeclared_conditionals("#if defined(A) && B == 12\nx\n#endif\n");
        assert_eq!(names, vec!["A", "B"]);
    }

    #[test]
    fn elif_conditions_are_scanned_too() {
        let names = undeclared_conditionals("#if A\nx\n#elif RAYMODE == 2\ny\n#endif\n");
        assert_eq!(names, vec!["A", "RAYMODE"]);
    }

    #[test]
    fn malformed_annotation_json_is_ignored_not_fatal() {
        // Workshop content is untrusted; a broken comment must not stop the
        // shader from being read.
        let declarations = parse("uniform float g_Speed; // {not json at all\n");
        assert_eq!(declarations.uniforms.len(), 1);
        assert!(declarations.uniforms[0].material.is_none());
    }
}

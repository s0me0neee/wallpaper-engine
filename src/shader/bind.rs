//! Turning a shader's uniform annotations plus a pass's material values into
//! concrete numbers to upload.
//!
//! This is plan.md §4.2's data-driven binding: the shader says what each
//! uniform means (`{"material":"strength", ...}`) and `scene.json` supplies
//! the value under that key, so no per-effect code is needed here at all.

use super::annotations::Declarations;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// A resolved, GLSL-typed uniform value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UniformValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
}

impl UniformValue {
    pub fn as_floats(self) -> Option<Vec<f32>> {
        match self {
            UniformValue::Float(x) => Some(vec![x]),
            UniformValue::Vec2(v) => Some(v.to_vec()),
            UniformValue::Vec3(v) => Some(v.to_vec()),
            UniformValue::Vec4(v) => Some(v.to_vec()),
            UniformValue::Int(_) => None,
        }
    }
}

/// Every uniform the renderer itself supplies — transform and per-slot
/// texture metadata — rather than the material table.
fn is_engine_supplied(name: &str) -> bool {
    name == "g_Time"
        || name == "g_ModelViewProjectionMatrix"
        || (name.starts_with("g_Texture") && name.ends_with("Resolution"))
}

/// Resolve every bindable uniform a shader declares against a pass's
/// `constantshadervalues`, falling back to the annotation's own default.
///
/// Samplers and engine-supplied uniforms are left out: the caller binds
/// textures and the transform separately.
pub fn resolve(declarations: &Declarations, material: &Map<String, Value>) -> Vec<(String, UniformValue)> {
    declarations
        .uniforms
        .iter()
        .filter(|uniform| uniform.kind != "sampler2D" && !is_engine_supplied(&uniform.name))
        .filter_map(|uniform| {
            let raw = uniform
                .material
                .as_deref()
                .and_then(|key| material.get(key))
                .or(uniform.default.as_ref())?;
            let value = coerce(&uniform.kind, raw)?;
            Some((uniform.name.clone(), value))
        })
        .collect()
}

/// Which combos a pass's texture slots switch on.
///
/// A sampler's own annotation can name a combo (`{"mode":"opacitymask",
/// "combo":"MASK"}`) that the shader expects to be `1` exactly when that slot
/// is actually bound — this is how `shake.frag`'s `MASK` and `TIMEOFFSET`
/// combos get set without scene.json ever mentioning them by name.
/// `textures[i]` binds `g_Texture{i+1}`; `g_Texture0` is always the previous
/// pass and is never switched by this.
pub fn texture_combos(declarations: &Declarations, textures: &[Option<String>]) -> BTreeMap<String, i64> {
    let mut combos = BTreeMap::new();
    for uniform in &declarations.uniforms {
        let (Some(combo), Some(slot)) = (&uniform.combo, sampler_slot(&uniform.name)) else {
            continue;
        };
        let Some(index) = slot.checked_sub(1) else {
            continue; // slot 0 is the previous pass, not a positional entry.
        };
        let bound = textures.get(index).is_some_and(Option::is_some);
        combos.insert(combo.clone(), i64::from(bound));
    }
    combos
}

fn sampler_slot(name: &str) -> Option<usize> {
    name.strip_prefix("g_Texture")?.parse().ok()
}

fn coerce(kind: &str, value: &Value) -> Option<UniformValue> {
    match kind {
        "float" => Some(UniformValue::Float(as_f32(value)?)),
        "int" | "bool" => {
            // Combo-adjacent flags and small counters; always well within
            // i32's range, unlike an arbitrary cast from an untrusted float.
            #[expect(clippy::cast_possible_truncation, reason = "material ints are small flags/counters")]
            let int = as_f32(value)? as i32;
            Some(UniformValue::Int(int))
        }
        "vec2" => as_floats(value, 2).map(|v| UniformValue::Vec2([v[0], v[1]])),
        "vec3" => as_floats(value, 3).map(|v| UniformValue::Vec3([v[0], v[1], v[2]])),
        "vec4" => as_floats(value, 4).map(|v| UniformValue::Vec4([v[0], v[1], v[2], v[3]])),
        _ => None,
    }
}

fn as_f32(value: &Value) -> Option<f32> {
    match value {
        // Material values are small (speeds, ranges, colour components), far
        // below f32's precision limit for any value this format carries.
        #[expect(clippy::cast_possible_truncation, reason = "scene material values are small")]
        Value::Number(number) => Some(number.as_f64()? as f32),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Parse `value` as `n` floats, however the file spelled it: a JSON array, a
/// bare number, or Wallpaper Engine's `"1.0 2.0"` space-separated strings.
///
/// A lone scalar splats to every component (`"friction": 2` meaning uniform
/// friction on both axes); a short vector pads with zero, matching
/// `scene::model::Vec3`'s tolerance for the same shorthand.
fn as_floats(value: &Value, n: usize) -> Option<Vec<f32>> {
    let parts: Vec<f32> = match value {
        Value::String(raw) => raw.split_whitespace().map(str::parse).collect::<Result<_, _>>().ok()?,
        Value::Array(items) => items.iter().map(as_f32).collect::<Option<_>>()?,
        other => vec![as_f32(other)?],
    };

    match parts.len() {
        1 => Some(vec![parts[0]; n]),
        len if len >= n => Some(parts[..n].to_vec()),
        _ => {
            let mut padded = parts;
            padded.resize(n, 0.0);
            Some(padded)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::annotations::parse;
    use serde_json::json;

    fn material(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn a_material_key_overrides_the_annotation_default() {
        let declarations = parse(r#"uniform float g_Amp; // {"material":"strength","default":0.1}"#);
        let bound = resolve(&declarations, &material(&[("strength", json!(0.5))]));
        assert_eq!(bound, vec![("g_Amp".to_string(), UniformValue::Float(0.5))]);
    }

    #[test]
    fn the_annotation_default_is_used_when_the_material_omits_the_key() {
        let declarations = parse(r#"uniform float g_Amp; // {"material":"strength","default":0.1}"#);
        let bound = resolve(&declarations, &Map::new());
        assert_eq!(bound, vec![("g_Amp".to_string(), UniformValue::Float(0.1))]);
    }

    #[test]
    fn a_space_separated_vec2_string_parses() {
        let declarations = parse(r#"uniform vec2 g_Friction; // {"material":"friction","default":"1 1"}"#);
        let bound = resolve(&declarations, &Map::new());
        assert_eq!(bound, vec![("g_Friction".to_string(), UniformValue::Vec2([1.0, 1.0]))]);
    }

    #[test]
    fn a_scalar_splats_to_every_component() {
        let declarations = parse(r#"uniform vec3 g_Color; // {"material":"color","default":1}"#);
        let bound = resolve(&declarations, &material(&[("color", json!(0.5))]));
        assert_eq!(bound, vec![("g_Color".to_string(), UniformValue::Vec3([0.5, 0.5, 0.5]))]);
    }

    #[test]
    fn samplers_and_engine_uniforms_are_left_for_the_caller() {
        let declarations = parse(concat!(
            "uniform sampler2D g_Texture0;\n",
            "uniform mat4 g_ModelViewProjectionMatrix;\n",
            "uniform vec4 g_Texture1Resolution;\n",
            "uniform float g_Time;\n",
        ));
        assert_eq!(resolve(&declarations, &Map::new()), Vec::new());
    }

    #[test]
    fn a_bool_default_becomes_an_int() {
        let declarations = parse(r#"uniform bool g_Enabled; // {"material":"enabled","default":true}"#);
        let bound = resolve(&declarations, &Map::new());
        assert_eq!(bound, vec![("g_Enabled".to_string(), UniformValue::Int(1))]);
    }

    #[test]
    fn an_uniform_with_no_default_anywhere_is_skipped() {
        let declarations = parse("uniform float g_Mystery;\n");
        assert_eq!(resolve(&declarations, &Map::new()), Vec::new());
    }

    #[test]
    fn a_bound_texture_slot_switches_on_its_combo() {
        // Verbatim shape from shake.frag: g_Texture2 is TIMEOFFSET, g_Texture3
        // is MASK; textures[] positionally binds g_Texture1.. onward.
        let declarations = parse(concat!(
            "uniform sampler2D g_Texture2; // {\"combo\":\"TIMEOFFSET\"}\n",
            "uniform sampler2D g_Texture3; // {\"combo\":\"MASK\"}\n",
        ));
        let textures = vec![None, Some("masks/shake_mask".to_string())];
        let combos = texture_combos(&declarations, &textures);
        assert_eq!(combos.get("TIMEOFFSET"), Some(&1), "textures[1] is bound");
        assert_eq!(combos.get("MASK"), Some(&0), "textures[2] is out of range");
    }

    #[test]
    fn g_texture0_is_never_a_positional_slot() {
        // It is always the previous pass, whatever scene.json's textures[] says.
        let declarations = parse("uniform sampler2D g_Texture0; // {\"combo\":\"SOMETHING\"}\n");
        assert_eq!(texture_combos(&declarations, &[Some("x".to_string())]), BTreeMap::new());
    }
}

//! Relaxing HLSL-isms that GLSL rejects.
//!
//! Wallpaper Engine compiles the same shader source as HLSL on Direct3D and as
//! GLSL elsewhere, and workshop authors write against whichever one they happen
//! to run. HLSL's implicit conversions are far looser than GLSL's, so a shader
//! that has only ever been compiled on Windows routinely contains constructs
//! the GL driver rejects outright — and a rejected shader drops that layer's
//! *entire* effect chain, which is why `scene_example8` renders with no sun
//! glow (`lens_flare_sun`) and `scene_example6` with no colour filter
//! (`gaussian`).
//!
//! Six constructs account for every failure in the corpus:
//!
//! | HLSL | GLSL says | seen in |
//! |---|---|---|
//! | `float g = <vec3>;` | incompatible types | `edge_glow`, `lens_flare_sun` |
//! | `int i = <float>;` | incompatible types | `test_shader` |
//! | `<vec4> * <vec2>` | `*` does not operate on those | `iris_movement` |
//! | `(a < b) * 6.0` | `*` does not operate on bool | `gaussian` |
//! | `max(0, <vec3>)` | no matching overload | `test_shader` |
//! | `<float> % <int>` | `%` is integer-only | `Simple_Audio_Bars` |
//!
//! HLSL's rule in the first three is *implicit truncation*: assigning or
//! combining a wider vector with a narrower one silently keeps the leading
//! components. GLSL will do exactly that through a constructor — `float(v)`
//! and `vec2(v4)` are legal and take the leading components — so the fix is to
//! make the conversion explicit rather than to reimplement it.
//!
//! This is deliberately lexical rather than a real front end. The passes only
//! rewrite when they can identify the operand types from declarations, and
//! leave the source untouched otherwise, so an expression this module does not
//! understand compiles exactly as it did before. That matters because the
//! source still carries `#if` directives — plan.md §6 leaves conditionals to
//! the driver — so there is no single well-formed parse tree to work from.

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Apply every relaxation, innermost constructs first so that the declaration
/// cast wraps an expression the earlier passes have already made legal.
pub fn relax(source: &str) -> String {
    // One chunk per top-level function body, so a local name types only inside
    // the function that declares it. Without that, one file's worth of text —
    // the shader plus every shim header expanded into it — puts `s` at both
    // `float` and `vec4`, and `delta` at both `float` and `vec2`, and whichever
    // way that is resolved corrupts one of the two.
    let (chunks, scope) = top_level_chunks(source);
    let global = declared_types(&scope);
    let signatures = function_parameters(source);

    let mut out = String::with_capacity(source.len());
    for &(start, end) in &chunks {
        let chunk = &source[start..end];
        let mut types = global.clone();
        types.extend(declared_types(chunk));
        out.push_str(&relax_chunk(chunk, &types, &signatures));
    }
    out
}

fn relax_chunk(
    source: &str,
    types: &HashMap<String, Type>,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) -> String {
    let mut out = array_initializers(source);
    out = bool_in_arithmetic(&out);
    out = float_modulo(&out, types);
    out = match_integer_signedness(&out, types);
    out = promote_scalar_arguments(&out, types);
    out = truncate_call_arguments(&out, types, signatures);
    out = truncate_mixed_operands(&out, types);
    out = cast_assignments(&out, types);
    cast_initializers(&out)
}

/// Chunks that each end just after a brace-depth-0 `}` (so a chunk holds at
/// most one function body), and separately the text that lies outside every
/// body — the uniforms, varyings and file-scope constants every chunk can see.
/// Comments are skipped so a brace inside one does not open a body.
fn top_level_chunks(source: &str) -> (Vec<(usize, usize)>, String) {
    let bytes = source.as_bytes();
    let mut chunks = Vec::new();
    let mut scope = String::with_capacity(source.len() / 4);
    let (mut start, mut outside, mut depth, mut at) = (0_usize, 0_usize, 0_i32, 0_usize);
    while at < bytes.len() {
        match bytes[at] {
            b'/' if bytes.get(at + 1) == Some(&b'/') => {
                while at < bytes.len() && bytes[at] != b'\n' {
                    at += 1;
                }
            }
            b'/' if bytes.get(at + 1) == Some(&b'*') => {
                at += 2;
                while at + 1 < bytes.len() && !(bytes[at] == b'*' && bytes[at + 1] == b'/') {
                    at += 1;
                }
                at = (at + 2).min(bytes.len());
            }
            b'{' => {
                if depth == 0 {
                    scope.push_str(&source[outside..at]);
                    scope.push('\n');
                }
                depth += 1;
                at += 1;
            }
            b'}' => {
                depth -= 1;
                at += 1;
                if depth <= 0 {
                    depth = 0;
                    outside = at;
                    chunks.push((start, at));
                    start = at;
                }
            }
            _ => at += 1,
        }
    }
    if outside < bytes.len() {
        scope.push_str(&source[outside..]);
    }
    if start < bytes.len() {
        chunks.push((start, bytes.len()));
    }
    if chunks.is_empty() {
        chunks.push((0, bytes.len()));
    }
    (chunks, scope)
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Type {
    pub kind: Kind,
    /// 1 for a scalar, 2..=4 for a vector. Matrices are not tracked.
    pub components: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Float,
    Int,
    Uint,
    Bool,
}

fn builtin_type(name: &str) -> Option<Type> {
    let (kind, components) = match name {
        "float" => (Kind::Float, 1),
        "int" => (Kind::Int, 1),
        "uint" => (Kind::Uint, 1),
        "bool" => (Kind::Bool, 1),
        "vec2" => (Kind::Float, 2),
        "vec3" => (Kind::Float, 3),
        "vec4" => (Kind::Float, 4),
        "ivec2" => (Kind::Int, 2),
        "ivec3" => (Kind::Int, 3),
        "ivec4" => (Kind::Int, 4),
        "uvec2" => (Kind::Uint, 2),
        "uvec3" => (Kind::Uint, 3),
        "uvec4" => (Kind::Uint, 4),
        "bvec2" => (Kind::Bool, 2),
        "bvec3" => (Kind::Bool, 3),
        "bvec4" => (Kind::Bool, 4),
        _ => return None,
    };
    Some(Type { kind, components })
}

/// The GLSL spelling of a type, for constructing one.
fn spell(ty: Type) -> &'static str {
    // Only 1..=4 components are ever constructed; anything else would be a
    // matrix, which this module does not track.
    let width = usize::from(ty.components.clamp(1, 4)) - 1;
    match ty.kind {
        Kind::Float => ["float", "vec2", "vec3", "vec4"][width],
        Kind::Int => ["int", "ivec2", "ivec3", "ivec4"][width],
        Kind::Uint => ["uint", "uvec2", "uvec3", "uvec4"][width],
        Kind::Bool => ["bool", "bvec2", "bvec3", "bvec4"][width],
    }
}

/// Every `<builtin type> <name>` pair in the source.
///
/// Scope is ignored on purpose: a name declared twice with different types in
/// different functions would be mistyped, but the passes below only act when a
/// rewrite is unambiguous, and the corpus does not do it. A type name followed
/// by an identifier is a declaration essentially everywhere it appears —
/// globals, locals, parameters and struct members alike — so one sweep covers
/// all four.
fn declared_types(source: &str) -> HashMap<String, Type> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(r"\b(float|int|uint|bool|vec2|vec3|vec4|ivec2|ivec3|ivec4|uvec2|uvec3|uvec4|bvec2|bvec3|bvec4)\s+([A-Za-z_]\w*)")
            .unwrap_or_else(|_| unreachable!("the declaration pattern is a literal"))
    });

    let mut types: HashMap<String, Type> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();
    for capture in pattern.captures_iter(source) {
        let (Some(kind), Some(name)) = (capture.get(1), capture.get(2)) else {
            continue;
        };
        if let Some(ty) = builtin_type(kind.as_str()) {
            let name = name.as_str();
            match types.get(name) {
                Some(seen) if *seen != ty => {
                    ambiguous.insert(name.to_string());
                }
                _ => {
                    types.insert(name.to_string(), ty);
                }
            }
        }
    }
    // A name the source declares at two different types is unresolvable
    // without scope, so it types nothing and every rewrite depending on it
    // stays disabled. This is not hypothetical: the shim headers are expanded
    // into the same text, and `scene_example6`'s `Simple_Audio_Bars` declares
    // `delta` as a `vec2` where `common_blending`'s `RGBToHSL` declares it as a
    // `float` — resolving that in the shader's favour rewrote *our own header*
    // into `delta / vec2(high + low)` and dropped the layer's whole chain.
    for name in ambiguous {
        types.remove(&name);
    }
    types.extend(defined_constants(source));
    types
}

/// `#define RESOLUTION 64` — a macro standing in for a literal.
///
/// The combo values this preprocessor injects arrive the same way, and
/// `Simple_Audio_Bars` takes a float modulo against one, so the macros have to
/// carry a type for that rewrite to be decidable at all.
fn defined_constants(source: &str) -> HashMap<String, Type> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(r"(?m)^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+(-?[0-9][0-9A-Za-z_.+-]*)[ \t]*$")
            .unwrap_or_else(|_| unreachable!("the define pattern is a literal"))
    });

    let mut types = HashMap::new();
    for capture in pattern.captures_iter(source) {
        let (Some(name), Some(value)) = (capture.get(1), capture.get(2)) else {
            continue;
        };
        let text = value.as_str();
        let kind = if text.contains('.') || text.contains('e') || text.ends_with('f') {
            Kind::Float
        } else if text.chars().skip(usize::from(text.starts_with('-'))).all(|c| c.is_ascii_digit()) {
            Kind::Int
        } else {
            continue;
        };
        types.insert(name.as_str().to_string(), Type { kind, components: 1 });
    }
    types
}

/// The type of a simple operand: an identifier, a swizzle of one, a numeric
/// literal, or a constructor call. Anything else is `None`, which disables
/// every rewrite that would have depended on it.
fn operand_type(text: &str, types: &HashMap<String, Type>) -> Option<Type> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    // A trailing swizzle narrows whatever it is applied to.
    if let Some((base, swizzle)) = split_trailing_swizzle(text) {
        let ty = operand_type(base, types)?;
        let width = swizzle.len();
        if width == 0 || width > 4 || width > usize::from(ty.components).max(4) {
            return None;
        }
        #[expect(clippy::cast_possible_truncation, reason = "checked to be 1..=4 just above")]
        return Some(Type { kind: ty.kind, components: width as u8 });
    }

    // A call: a constructor names its own type, and a handful of builtins have
    // a return type that does not depend on their arguments.
    if text.ends_with(')')
        && let Some(open) = matching_open(text)
    {
        let name = text[..open].trim();
        // An empty callee is a parenthesised group, not a call — leave it to
        // the group branch below rather than giving up on it here.
        if !name.is_empty() {
            if let Some(ty) = builtin_type(name).or_else(|| builtin_return(name)) {
                return Some(ty);
            }
            // Component-wise builtins take the shape of their first argument.
            if follows_first_argument(name) {
                let args = split_top_level(&text[open + 1..text.len() - 1], ',');
                return operand_type(args.first()?.trim(), types);
            }
            return None;
        }
    }

    // A numeric literal.
    if text.chars().next().is_some_and(|c| c.is_ascii_digit())
        || (text.starts_with('.') && text.len() > 1)
    {
        let kind = if text.contains('.') || text.contains('e') || text.ends_with('f') {
            Kind::Float
        } else if text.ends_with('u') || text.ends_with('U') {
            Kind::Uint
        } else {
            Kind::Int
        };
        return Some(Type { kind, components: 1 });
    }

    if text.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return types.get(text).copied();
    }

    // A parenthesised group is whatever it contains.
    if text.starts_with('(') && text.ends_with(')') && matching_open(text) == Some(0) {
        return operand_type(&text[1..text.len() - 1], types);
    }

    // An arithmetic expression takes the width of its widest operand: a vector
    // combined with a scalar stays a vector, and by the time this is asked two
    // vectors of different widths have already been reconciled by
    // `truncate_mixed_operands`.
    arithmetic_type(text, types)
}

fn arithmetic_type(text: &str, types: &HashMap<String, Type>) -> Option<Type> {
    let mut parts: Vec<&str> = vec![text];
    for op in ['+', '-', '*', '/'] {
        parts = parts.iter().flat_map(|part| split_top_level(part, op)).collect();
    }
    let parts: Vec<&str> = parts.into_iter().map(str::trim).filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }

    let mut widest: Option<Type> = None;
    for part in parts {
        let ty = operand_type(part, types)?;
        widest = Some(match widest {
            Some(current) if current.components >= ty.components => current,
            _ => ty,
        });
    }
    widest
}

fn is_swizzle_component(c: char) -> bool {
    matches!(c, 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a' | 's' | 't' | 'p' | 'q')
}

/// Split `expr.xyz` into `("expr", "xyz")`, ignoring dots inside parentheses
/// and inside numeric literals.
fn split_trailing_swizzle(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    let mut depth = 0_i32;
    let mut dot = None;
    for (index, c) in text.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '.' if depth == 0 => dot = Some(index),
            _ => {}
        }
    }
    let dot = dot?;
    let (base, swizzle) = (&text[..dot], &text[dot + 1..]);
    if base.is_empty() || swizzle.is_empty() || !swizzle.chars().all(is_swizzle_component) {
        return None;
    }
    // `1.0` is a literal, not a swizzle of `1`.
    if bytes[dot.saturating_sub(1)].is_ascii_digit() && base.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((base, swizzle))
}

/// Index of the `(` matching the `)` that ends `text`.
fn matching_open(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0_i32;
    let mut at = bytes.len();
    while at > 0 {
        at -= 1;
        match bytes[at] {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    return Some(at);
                }
            }
            _ => {}
        }
    }
    None
}

/// Builtins that return whatever shape their first argument has.
fn follows_first_argument(name: &str) -> bool {
    matches!(
        name,
        // `step` and `smoothstep` are deliberately absent: they take their
        // shape from their *last* argument, so reading the first would type a
        // `step(float, vec3)` as a scalar.
        "mix" | "clamp" | "min" | "max" | "abs" | "floor" | "ceil" | "fract" | "pow"
            | "normalize" | "saturate" | "sqrt" | "exp" | "log" | "sin" | "cos" | "mod"
            | "sign" | "reflect"
    )
}

/// Builtins whose return type does not depend on their arguments.
fn builtin_return(name: &str) -> Option<Type> {
    let (kind, components) = match name {
        "texSample2D" | "texSample2DLod" | "texture" | "texture2D" | "texture2DLod"
        | "textureLod" | "texSample3D" | "texSampleCube" => (Kind::Float, 4),
        "length" | "dot" | "distance" | "determinant" | "noise" => (Kind::Float, 1),
        _ => return None,
    };
    Some(Type { kind, components })
}

// ---------------------------------------------------------------------------
// Scanning helpers
// ---------------------------------------------------------------------------

/// Whether `at` sits inside a comment or string in `source`.
fn in_comment(source: &str, at: usize) -> bool {
    let head = &source[..at];
    match head.rfind('\n') {
        Some(line_start) => head[line_start..].contains("//"),
        None => head.contains("//"),
    }
}

/// Byte index just past the operand ending at `end` (exclusive), scanning left.
///
/// An operand here is a primary expression: a parenthesised group, or an
/// identifier with any swizzle, indexing and call parentheses attached.
fn operand_start(source: &str, end: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut at = end;
    while at > 0 && bytes[at - 1].is_ascii_whitespace() {
        at -= 1;
    }
    if at == 0 {
        return None;
    }

    if bytes[at - 1] == b')' || bytes[at - 1] == b']' {
        let close = bytes[at - 1];
        let open = if close == b')' { b'(' } else { b'[' };
        let mut depth = 0_i32;
        let mut scan = at;
        while scan > 0 {
            scan -= 1;
            if bytes[scan] == close {
                depth += 1;
            } else if bytes[scan] == open {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
        if depth != 0 {
            return None;
        }
        // A call keeps its function name.
        let mut name = scan;
        while name > 0 && (bytes[name - 1].is_ascii_alphanumeric() || bytes[name - 1] == b'_') {
            name -= 1;
        }
        return Some(name);
    }

    let mut start = at;
    while start > 0 {
        let c = bytes[start - 1];
        if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' {
            start -= 1;
        } else {
            break;
        }
    }
    (start < at).then_some(start)
}

/// Byte index of the end of the operand starting at `start`, scanning right.
fn operand_end(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut at = start;
    while at < bytes.len() && bytes[at].is_ascii_whitespace() {
        at += 1;
    }
    if at >= bytes.len() {
        return None;
    }

    // A leading sign or negation belongs to the operand.
    if bytes[at] == b'-' || bytes[at] == b'+' || bytes[at] == b'!' {
        at += 1;
    }

    let mut end = at;
    while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_' || bytes[end] == b'.') {
        end += 1;
    }
    // Trailing call or index parentheses.
    while end < bytes.len() && (bytes[end] == b'(' || bytes[end] == b'[') {
        let open = bytes[end];
        let close = if open == b'(' { b')' } else { b']' };
        let mut depth = 0_i32;
        while end < bytes.len() {
            if bytes[end] == open {
                depth += 1;
            } else if bytes[end] == close {
                depth -= 1;
                if depth == 0 {
                    end += 1;
                    break;
                }
            }
            end += 1;
        }
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_' || bytes[end] == b'.') {
            end += 1;
        }
    }
    (end > at).then_some(end)
}

/// Split on `sep` at paren/bracket depth zero.
fn split_top_level(text: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0_i32;
    let mut start = 0;
    for (index, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ if c == sep && depth == 0 => {
                parts.push(&text[start..index]);
                start = index + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

// ---------------------------------------------------------------------------
// Pass 0: brace-initialised arrays
// ---------------------------------------------------------------------------

/// `const vec2 kernel[N] = { a, b, c };` — HLSL's aggregate initializer.
///
/// GLSL spells the same thing `vec2[](a, b, c)`, and rejects the trailing
/// comma HLSL tolerates before the closing brace. `bokeh_blur` declares three
/// of these, one per `QUALITY` level, which is why `scene_example8`'s
/// post-processing layer lost its depth of field entirely.
fn array_initializers(source: &str) -> String {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(
            r"(?m)^[ \t]*(?:(?:const|highp|mediump|lowp)[ \t]+)*(float|int|uint|bool|vec2|vec3|vec4|ivec2|ivec3|ivec4|uvec2|uvec3|uvec4|mat2|mat3|mat4)[ \t]+[A-Za-z_]\w*[ \t]*\[[^\]]*\][ \t]*=[ \t]*\{",
        )
        .unwrap_or_else(|_| unreachable!("the array pattern is a literal"))
    });

    let bytes = source.as_bytes();
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for capture in pattern.captures_iter(source) {
        let (Some(whole), Some(kind)) = (capture.get(0), capture.get(1)) else {
            continue;
        };
        if in_comment(source, whole.start()) {
            continue;
        }
        let open = whole.end() - 1;
        let mut depth = 0_i32;
        let mut close = open;
        while close < bytes.len() {
            if bytes[close] == b'{' {
                depth += 1;
            } else if bytes[close] == b'}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            close += 1;
        }
        if depth != 0 {
            continue;
        }

        let inner = &source[open + 1..close];
        let elements: Vec<&str> = split_top_level(inner, ',')
            .into_iter()
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect();
        if elements.is_empty() {
            continue;
        }
        replacements.push((
            open,
            close + 1,
            format!("{}[]({})", kind.as_str(), elements.join(", ")),
        ));
    }

    splice(source, replacements)
}

// ---------------------------------------------------------------------------
// Pass 1: bool used as a number
// ---------------------------------------------------------------------------

/// `(a < b) * 6.0` — HLSL promotes the bool to 0.0/1.0.
///
/// Only a parenthesised comparison directly adjacent to `*` or `/` is
/// rewritten. A group containing `&&`, `||` or `?` is left alone: it may well
/// be a plain condition, and guessing wrong there would change control flow
/// rather than a number.
fn bool_in_arithmetic(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut edits: Vec<(usize, usize)> = Vec::new();

    for (index, _) in source.match_indices(')') {
        if in_comment(source, index) {
            continue;
        }
        let mut after = index + 1;
        while after < bytes.len() && bytes[after].is_ascii_whitespace() {
            after += 1;
        }
        let multiplied = after < bytes.len() && (bytes[after] == b'*' || bytes[after] == b'/');
        let before_is_product = {
            let mut scan = index;
            // Find the matching open paren first.
            let mut depth = 0_i32;
            let mut open = None;
            while scan > 0 {
                scan -= 1;
                if bytes[scan] == b')' {
                    depth += 1;
                } else if bytes[scan] == b'(' {
                    if depth == 0 {
                        open = Some(scan);
                        break;
                    }
                    depth -= 1;
                }
            }
            open
        };
        let Some(open) = before_is_product else { continue };

        let mut prior = open;
        while prior > 0 && bytes[prior - 1].is_ascii_whitespace() {
            prior -= 1;
        }
        let preceded = prior > 0 && (bytes[prior - 1] == b'*' || bytes[prior - 1] == b'/');
        if !multiplied && !preceded {
            continue;
        }

        let inner = &source[open + 1..index];
        let comparison = ["<=", ">=", "==", "!=", "<", ">"]
            .iter()
            .any(|op| split_top_level(inner, '\n').iter().any(|line| line.contains(op)));
        if !comparison || inner.contains("&&") || inner.contains("||") || inner.contains('?') {
            continue;
        }
        // A call's argument list is not a bool, however it reads.
        if prior > 0 && (bytes[prior - 1].is_ascii_alphanumeric() || bytes[prior - 1] == b'_') {
            continue;
        }
        edits.push((open, index + 1));
    }

    apply(source, &edits, |text| format!("float{text}"))
}

// ---------------------------------------------------------------------------
// Pass 2: `%` on floats
// ---------------------------------------------------------------------------

/// HLSL's `%` is `fmod` for floats; GLSL's is integer-only.
fn float_modulo(source: &str, types: &HashMap<String, Type>) -> String {
    let bytes = source.as_bytes();
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for (index, _) in source.match_indices('%') {
        if in_comment(source, index) || bytes.get(index + 1) == Some(&b'=') {
            continue;
        }
        let (Some(left), Some(right)) = (operand_start(source, index), operand_end(source, index + 1))
        else {
            continue;
        };
        let left_text = source[left..index].trim();
        let right_text = source[index + 1..right].trim();
        let (Some(lt), Some(rt)) = (operand_type(left_text, types), operand_type(right_text, types))
        else {
            continue;
        };
        if lt.kind != Kind::Float && rt.kind != Kind::Float {
            continue;
        }
        let cast = |text: &str, ty: Type| {
            if ty.kind == Kind::Float { text.to_string() } else { format!("float({text})") }
        };
        replacements.push((
            left,
            right,
            format!("mod({}, {})", cast(left_text, lt), cast(right_text, rt)),
        ));
    }

    splice(source, replacements)
}

// ---------------------------------------------------------------------------
// Pass 3: scalar promoted to a vector in a call
// ---------------------------------------------------------------------------

/// `max(0, albedo.rgb)` — HLSL broadcasts the scalar. GLSL offers
/// `max(genType, float)` but not `max(float, genType)`, so promoting the
/// literal to the vector's own type satisfies the first overload either way.
fn promote_scalar_arguments(source: &str, types: &HashMap<String, Type>) -> String {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(r"\b(max|min|mod|pow|step|atan)\s*\(")
            .unwrap_or_else(|_| unreachable!("the call pattern is a literal"))
    });

    let bytes = source.as_bytes();
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for capture in pattern.captures_iter(source) {
        let Some(whole) = capture.get(0) else { continue };
        if in_comment(source, whole.start()) {
            continue;
        }
        let open = whole.end() - 1;
        let mut depth = 0_i32;
        let mut close = open;
        while close < bytes.len() {
            if bytes[close] == b'(' {
                depth += 1;
            } else if bytes[close] == b')' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            close += 1;
        }
        if depth != 0 {
            continue;
        }

        let args = split_top_level(&source[open + 1..close], ',');
        if args.len() != 2 {
            continue;
        }
        let (Some(a), Some(b)) = (operand_type(args[0].trim(), types), operand_type(args[1].trim(), types))
        else {
            continue;
        };
        if a.components == b.components {
            continue;
        }
        // Widen the scalar to the vector's type; leave anything else alone.
        let (scalar, vector, index) = if a.components == 1 && b.components > 1 {
            (a, b, 0)
        } else if b.components == 1 && a.components > 1 {
            (b, a, 1)
        } else {
            continue;
        };
        if scalar.kind != Kind::Float && scalar.kind != Kind::Int {
            continue;
        }
        let mut widened: Vec<String> = args.iter().map(|arg| arg.trim().to_string()).collect();
        widened[index] = format!("{}({})", spell(vector), widened[index]);
        replacements.push((
            open + 1,
            close,
            widened.join(", "),
        ));
    }

    splice(source, replacements)
}

/// Parameter types of every function the source defines, plus the samplers.
///
/// `bokeh_blur` passes a `vec4 v_TexCoord` varying straight into a 2D sampler
/// and into its own `maskBokeh(vec2, float)`; both of its stages really do
/// declare the varying four wide, so there is no mismatch for
/// `reconcile_varyings` to catch and the truncation has to happen at the call.
fn function_parameters(source: &str) -> HashMap<String, Vec<Option<Type>>> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(r"(?m)^[ \t]*(?:\w+)[ \t]+([A-Za-z_]\w*)[ \t]*\(([^)]*)\)[ \t]*\{")
            .unwrap_or_else(|_| unreachable!("the signature pattern is a literal"))
    });

    let mut signatures: HashMap<String, Vec<Option<Type>>> = HashMap::new();
    // A 2D sampler's coordinate is always two wide; the sampler itself is not
    // a type this module tracks, so it stays `None`.
    for name in ["texSample2D", "texture2D", "texSample2DLod", "texture2DLod"] {
        signatures.insert(
            name.to_string(),
            vec![None, Some(Type { kind: Kind::Float, components: 2 })],
        );
    }

    for capture in pattern.captures_iter(source) {
        let (Some(name), Some(params)) = (capture.get(1), capture.get(2)) else {
            continue;
        };
        if matches!(name.as_str(), "if" | "for" | "while" | "switch") {
            continue;
        }
        let list = split_top_level(params.as_str(), ',')
            .into_iter()
            .map(|param| {
                param
                    .split_whitespace()
                    .find_map(builtin_type)
                    .filter(|_| !param.contains('['))
            })
            .collect();
        signatures.insert(name.as_str().to_string(), list);
    }
    signatures
}

/// Narrow any argument passed wider than the parameter it binds to.
/// `signatures` comes from the whole file, not from `source`: a function is
/// routinely defined in one chunk and called from another.
fn truncate_call_arguments(
    source: &str,
    types: &HashMap<String, Type>,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) -> String {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(r"\b([A-Za-z_]\w*)\s*\(")
            .unwrap_or_else(|_| unreachable!("the call pattern is a literal"))
    });

    let bytes = source.as_bytes();
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for capture in pattern.captures_iter(source) {
        let (Some(whole), Some(name)) = (capture.get(0), capture.get(1)) else {
            continue;
        };
        if in_comment(source, whole.start()) {
            continue;
        }
        let Some(params) = signatures.get(name.as_str()) else { continue };

        let open = whole.end() - 1;
        let mut depth = 0_i32;
        let mut close = open;
        while close < bytes.len() {
            if bytes[close] == b'(' {
                depth += 1;
            } else if bytes[close] == b')' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            close += 1;
        }
        if depth != 0 {
            continue;
        }

        let args = split_top_level(&source[open + 1..close], ',');
        if args.len() != params.len() {
            continue;
        }
        let mut cursor = open + 1;
        for (arg, param) in args.iter().zip(params) {
            let start = cursor;
            cursor += arg.len() + 1;
            let Some(want) = *param else { continue };
            let text = arg.trim();
            let Some(have) = operand_type(text, types) else { continue };
            if have.components <= want.components {
                continue;
            }
            let offset = arg.len() - arg.trim_start().len();
            replacements.push((
                start + offset,
                start + offset + text.len(),
                format!("{}({})", spell(want), text),
            ));
        }
    }

    splice(source, replacements)
}

// ---------------------------------------------------------------------------
// Pass 4: mismatched vector widths in a binary operation
// ---------------------------------------------------------------------------

/// `vec4 * vec2` — HLSL truncates the wider operand to the narrower one.
/// Whether the character at `index` is a binary arithmetic operator, as opposed
/// to a sign, an increment, a compound assignment, a comment delimiter or the
/// `-` inside a float exponent.
fn is_binary_operator(source: &str, index: usize) -> bool {
    if in_comment(source, index) {
        return false;
    }
    let bytes = source.as_bytes();
    let here = bytes[index];
    if matches!(bytes.get(index + 1), Some(b'=')) || bytes.get(index + 1) == Some(&here) {
        return false;
    }
    if here == b'*' || here == b'/' {
        return bytes.get(index + 1) != Some(&b'*') && bytes.get(index.wrapping_sub(1)) != Some(&b'/');
    }
    if here == b'%' {
        return true;
    }
    let previous = index
        .checked_sub(1)
        .map(|at| bytes[at])
        .map(|c| if c.is_ascii_whitespace() { b' ' } else { c });
    !matches!(previous, None | Some(b'(' | b',' | b'[' | b'=' | b'+' | b'-' | b'*' | b'/' | b'<' | b'>' | b'?' | b':' | b'&' | b'|' | b'e' | b'E' | b'{' | b';' | b'!'))
}

/// `barFreq1 + 1` — HLSL promotes freely between `int` and `uint`; GLSL 3.30
/// says it converts `int` to `uint` implicitly, but Apple's frontend does not,
/// and rejects the expression outright. `Simple_Audio_Bars` indexes the audio
/// spectrum through `uint barFreq2 = uint((barFreq1 + 1) % RESOLUTION)`, where
/// both the `1` and the `RESOLUTION` combo are `int` — one rejected line drops
/// the whole audio-bars chain.
fn match_integer_signedness(source: &str, types: &HashMap<String, Type>) -> String {
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for (index, _) in source.match_indices(['*', '/', '+', '-', '%']) {
        if !is_binary_operator(source, index) {
            continue;
        }
        let (Some(left), Some(right)) = (operand_start(source, index), operand_end(source, index + 1))
        else {
            continue;
        };
        let left_text = source[left..index].trim();
        let right_text = source[index + 1..right].trim();
        let (Some(lt), Some(rt)) = (operand_type(left_text, types), operand_type(right_text, types))
        else {
            continue;
        };
        // Only an int/uint pair of the same width: a differing width is
        // truncation's business, and a float on either side is not this bug.
        if lt.components != rt.components || {
            let pair = (lt.kind, rt.kind);
            pair != (Kind::Int, Kind::Uint) && pair != (Kind::Uint, Kind::Int)
        } {
            continue;
        }
        let target = Type { kind: Kind::Uint, components: lt.components };
        let (text, start, end) = if lt.kind == Kind::Int {
            (left_text, left, left + left_text.len())
        } else {
            let offset = source[index + 1..right].len() - source[index + 1..right].trim_start().len();
            (right_text, index + 1 + offset, index + 1 + offset + right_text.len())
        };
        replacements.push((start, end, format!("{}({})", spell(target), text)));
    }

    splice(source, replacements)
}

fn truncate_mixed_operands(source: &str, types: &HashMap<String, Type>) -> String {
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for (index, _) in source.match_indices(['*', '/', '+', '-']) {
        if !is_binary_operator(source, index) {
            continue;
        }
        let (Some(left), Some(right)) = (operand_start(source, index), operand_end(source, index + 1))
        else {
            continue;
        };
        let left_text = source[left..index].trim();
        let right_text = source[index + 1..right].trim();
        let (Some(lt), Some(rt)) = (operand_type(left_text, types), operand_type(right_text, types))
        else {
            continue;
        };
        if lt.components < 2 || rt.components < 2 || lt.components == rt.components {
            continue;
        }
        let narrow = lt.components.min(rt.components);
        let target = Type { kind: Kind::Float, components: narrow };
        // Span the operand's own text only, so the whitespace around the
        // operator survives and the rewritten line still reads like GLSL.
        let (text, start, end) = if lt.components > rt.components {
            (left_text, left, left + left_text.len())
        } else {
            let offset = source[index + 1..right].len() - source[index + 1..right].trim_start().len();
            (right_text, index + 1 + offset, index + 1 + offset + right_text.len())
        };
        replacements.push((start, end, format!("{}({})", spell(target), text)));
    }

    splice(source, replacements)
}

/// `albedo.rgb = mix(<vec4>, <vec4>, f);` — HLSL truncates on assignment too,
/// not only on declaration. `gaussian` blends two `vec4`s into a `vec3`
/// swizzle this way.
fn cast_assignments(source: &str, types: &HashMap<String, Type>) -> String {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(r"(?m)^[ \t]*([A-Za-z_]\w*(?:\.[xyzwrgbastpq]+)?)[ \t]*=[ \t]*")
            .unwrap_or_else(|_| unreachable!("the assignment pattern is a literal"))
    });

    let bytes = source.as_bytes();
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for capture in pattern.captures_iter(source) {
        let (Some(whole), Some(target)) = (capture.get(0), capture.get(1)) else {
            continue;
        };
        if in_comment(source, whole.start()) {
            continue;
        }
        // A declaration, not an assignment — `cast_initializers` owns those.
        if builtin_type(target.as_str()).is_some() {
            continue;
        }
        let Some(want) = operand_type(target.as_str(), types) else { continue };

        let value_start = whole.end();
        let mut depth = 0_i32;
        let mut at = value_start;
        let mut end = None;
        while at < bytes.len() {
            match bytes[at] {
                b'(' | b'[' => depth += 1,
                b')' | b']' => depth -= 1,
                b';' if depth == 0 => {
                    end = Some(at);
                    break;
                }
                _ => {}
            }
            at += 1;
        }
        let Some(end) = end else { continue };

        let value = source[value_start..end].trim();
        if value.is_empty() || value.contains('#') {
            continue;
        }
        let Some(have) = operand_type(value, types) else { continue };
        if have.components <= want.components {
            continue;
        }
        replacements.push((
            value_start,
            end,
            format!("{}({})", spell(want), value),
        ));
    }

    splice(source, replacements)
}

// ---------------------------------------------------------------------------
// Pass 5: the declaration's own type
// ---------------------------------------------------------------------------

/// `float g = <vec3>;` and `int i = <float>;`.
///
/// Wrapping the initializer in the declared type's own constructor is a no-op
/// when the types already agree, and is exactly HLSL's conversion when they do
/// not: GLSL constructors truncate a wider vector and convert between scalar
/// kinds. Declarations with several declarators, or with an array on either
/// side, are skipped — a constructor cannot express those.
fn cast_initializers(source: &str) -> String {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(
            r"(?m)(^[ \t]*|\bfor[ \t]*\([ \t]*)(?:(?:const|highp|mediump|lowp)[ \t]+)*(float|int|uint|bool|vec2|vec3|vec4|ivec2|ivec3|ivec4|uvec2|uvec3|uvec4|bvec2|bvec3|bvec4)[ \t]+([A-Za-z_]\w*)[ \t]*=[ \t]*",
        )
        .unwrap_or_else(|_| unreachable!("the declaration pattern is a literal"))
    });

    let bytes = source.as_bytes();
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for capture in pattern.captures_iter(source) {
        let (Some(whole), Some(kind)) = (capture.get(0), capture.get(2)) else {
            continue;
        };
        if in_comment(source, whole.start()) {
            continue;
        }
        let Some(declared) = builtin_type(kind.as_str()) else { continue };

        // The initializer runs to the next `;` at depth zero.
        let value_start = whole.end();
        let mut depth = 0_i32;
        let mut at = value_start;
        let mut end = None;
        while at < bytes.len() {
            match bytes[at] {
                b'(' | b'[' => depth += 1,
                b')' | b']' => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                b';' if depth == 0 => {
                    end = Some(at);
                    break;
                }
                b'\n' if depth == 0 && bytes[at.saturating_sub(1)] != b',' => {}
                _ => {}
            }
            at += 1;
        }
        let Some(end) = end else { continue };

        let value = &source[value_start..end];
        // Several declarators in one statement, or a preprocessor line in the
        // middle of the initializer: not something a constructor can wrap.
        if split_top_level(value, ',').len() > 1 || value.contains('#') || value.trim().is_empty() {
            continue;
        }
        // Already exactly this constructor and nothing else — `vec2(a) * b`
        // merely starts with one, and still needs the wrap.
        if is_sole_constructor(value.trim(), kind.as_str()) {
            continue;
        }
        // A conditional yields either branch; wrapping it is still correct but
        // the parentheses have to survive, which they do.
        replacements.push((
            value_start,
            end,
            format!("{}({})", spell(declared), value.trim()),
        ));
    }

    splice(source, replacements)
}

/// Whether `value` is exactly `kind(...)` with the call spanning all of it.
fn is_sole_constructor(value: &str, kind: &str) -> bool {
    let Some(rest) = value.strip_prefix(kind) else { return false };
    let rest = rest.trim_start();
    if !rest.starts_with('(') || !rest.ends_with(')') {
        return false;
    }
    // The opening paren must close only at the very end.
    let mut depth = 0_i32;
    for (index, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return index + 1 == rest.len();
                }
            }
            _ => {}
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Edit application
// ---------------------------------------------------------------------------

/// Apply `(start, end)` spans, rewriting each with `make`, right to left.
fn apply(source: &str, edits: &[(usize, usize)], make: impl Fn(&str) -> String) -> String {
    let mut spans: Vec<(usize, usize, String)> =
        edits.iter().map(|&(s, e)| (s, e, make(&source[s..e]))).collect();
    spans.sort_by_key(|&(s, _, _)| s);
    splice(source, spans)
}

/// Splice replacements into `source`, dropping any that overlap an earlier one.
fn splice(source: &str, mut replacements: Vec<(usize, usize, String)>) -> String {
    if replacements.is_empty() {
        return source.to_string();
    }
    replacements.sort_by_key(|&(start, _, _)| start);

    let mut out = String::with_capacity(source.len() + 64);
    let mut cursor = 0;
    for (start, end, text) in replacements {
        if start < cursor || end > source.len() || start > end {
            continue;
        }
        out.push_str(&source[cursor..start]);
        out.push_str(&text);
        cursor = end;
    }
    out.push_str(&source[cursor..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_narrower_declaration_truncates_its_initializer() {
        // `edge_glow` computes a Sobel magnitude into a float from two vec3s.
        let source = "vec3 gx; vec3 gy;\n\tfloat g = gx*gx + gy*gy;\n";
        assert!(relax(source).contains("float g = float(gx*gx + gy*gy);"));
    }

    #[test]
    fn a_float_initializer_becomes_an_int_where_the_declaration_says_so() {
        // `test_shader` runs `for (int i = u_MinFreqRange; ...)` over a float
        // uniform, which HLSL converts and GLSL will not.
        let source = "uniform float u_MinFreqRange;\n\tfor (int i = u_MinFreqRange; i < 4; i++) {}\n";
        assert!(relax(source).contains("for (int i = int(u_MinFreqRange);"));
    }

    #[test]
    fn matching_types_are_wrapped_harmlessly() {
        // The wrap is unconditional, so it has to be a no-op when types agree.
        let source = "\tfloat x = 1.0;\n";
        assert_eq!(relax(source), "\tfloat x = float(1.0);\n");
    }

    #[test]
    fn a_wider_operand_truncates_to_the_narrower_one() {
        // `iris_movement` multiplies a vec4 cursor position by a vec2 scale.
        let source = "vec4 cursor; uniform vec2 g_CursorScale;\n\tvec2 da = cursor * g_CursorScale;\n";
        let out = relax(source);
        assert!(out.contains("vec2(cursor) * g_CursorScale"), "got {out}");
    }

    #[test]
    fn a_comparison_used_as_a_number_becomes_one() {
        // `gaussian` writes `depth *= (depth < limit) * 6.0;`.
        let source = "float depth; float limit;\n\tdepth *= (depth < limit) * 6.0;\n";
        let out = relax(source);
        assert!(out.contains("float(depth < limit) * 6.0"), "got {out}");
    }

    #[test]
    fn a_condition_is_not_mistaken_for_arithmetic() {
        // No adjacent `*` or `/`, so nothing to promote.
        let source = "float a; float b;\n\tif (a < b) { }\n";
        assert_eq!(relax(source), source);
    }

    #[test]
    fn a_logical_group_is_left_alone() {
        // Rewriting this would change control flow, not a number.
        let source = "float a; float b; float c;\n\tc = (a < b && b < 2.0) ? 1.0 : 2.0;\n";
        assert!(!relax(source).contains("float(a < b"));
    }

    #[test]
    fn a_scalar_argument_is_promoted_to_the_vector_it_meets() {
        // `test_shader` writes `max(0, albedo.rgb)`.
        let source = "vec4 albedo;\n\tvec3 c = max(0, albedo.rgb);\n";
        let out = relax(source);
        assert!(out.contains("max(vec3(0), albedo.rgb)"), "got {out}");
    }

    #[test]
    fn modulo_on_a_float_becomes_mod() {
        // `Simple_Audio_Bars` writes `frequency % RESOLUTION`.
        let source = "float frequency; int RESOLUTION;\n\tfloat f = frequency % RESOLUTION;\n";
        let out = relax(source);
        assert!(out.contains("mod(frequency, float(RESOLUTION))"), "got {out}");
    }

    #[test]
    fn an_int_meeting_a_uint_becomes_a_uint() {
        // `Simple_Audio_Bars` writes `uint((barFreq1 + 1) % RESOLUTION)`.
        let source = "uint barFreq1;\n#define RESOLUTION 32\n\tuint b = (barFreq1 + 1) % RESOLUTION;\n";
        let out = relax(source);
        assert!(out.contains("barFreq1 + uint(1)"), "{out}");
        assert!(out.contains("% uint(RESOLUTION)"), "{out}");
    }

    #[test]
    fn a_local_name_types_only_inside_its_own_function() {
        // The shim headers land in the same text as the shader, so `s` is
        // genuinely a `float` in one function and a `vec4` in another.
        // Resolving that file-wide corrupts whichever one loses.
        let source = concat!(
            "uniform float k;\n",
            "vec2 shim(float angle) { float s = sin(angle); return vec2(s * k, s); }\n",
            "vec3 body(vec2 uv) { vec4 s = texture(g_Texture0, uv); return pow(s.rgb, k); }\n",
        );
        let out = relax(source);
        assert!(out.contains("vec2(s * k, s)"), "{out}");
        assert!(out.contains("pow(s.rgb, vec3(k))"), "{out}");
    }

    #[test]
    fn a_signature_carries_across_function_bodies() {
        // Per-function scoping must not hide a callee defined in another
        // chunk: `bokeh_blur` calls `maskBokeh` forty lines after defining it.
        let source = concat!(
            "in vec4 v_TexCoord;\n",
            "vec3 maskBokeh(vec2 coord, float depth) { return vec3(coord, depth); }\n",
            "void main() { vec3 c = maskBokeh(v_TexCoord, 1.0); }\n",
        );
        let out = relax(source);
        assert!(out.contains("maskBokeh(vec2(v_TexCoord), 1.0)"), "{out}");
    }

    #[test]
    fn two_ints_keep_their_signedness() {
        let source = "int a; int b;\n\tint c = a + b;\n";
        assert!(relax(source).contains("a + b"));
    }

    #[test]
    fn integer_modulo_is_left_alone() {
        let source = "int a; int b;\n\tint c = a % b;\n";
        assert!(relax(source).contains("a % b"));
    }

    #[test]
    fn an_unknown_operand_disables_the_rewrite() {
        // Nothing declares `mystery`, so its width is unknown and the source
        // has to survive untouched rather than be guessed at.
        let source = "uniform vec2 known;\n\tvec2 d = mystery * known;\n";
        let out = relax(source);
        assert!(!out.contains("vec2(mystery)"), "got {out}");
    }

    #[test]
    fn comments_are_not_rewritten() {
        let source = "vec3 gx;\n\t// float g = gx*gx;\n";
        assert_eq!(relax(source), source);
    }

    #[test]
    fn a_multi_declarator_statement_is_skipped() {
        // `float a = 1.0, b = 2.0;` cannot be wrapped by one constructor.
        let source = "\tfloat a = 1.0, b = 2.0;\n";
        assert_eq!(relax(source), source);
    }

    #[test]
    fn a_brace_initialised_array_becomes_a_glsl_constructor() {
        // `bokeh_blur` declares its sample kernel this way, trailing comma and
        // all, which GLSL rejects twice over.
        let source = "const vec2 kernel[3] = {\n\tvec2(0, 0),\n\tvec2(0.5, 0),\n};\n";
        let out = relax(source);
        assert!(out.contains("vec2[](vec2(0, 0), vec2(0.5, 0))"), "got {out}");
        assert!(!out.contains('{'), "the braces must be gone: {out}");
    }

    #[test]
    fn a_function_body_is_not_mistaken_for_an_array_initializer() {
        let source = "void main() {\n\tfloat x = 1.0;\n}\n";
        assert!(relax(source).contains("void main() {"));
    }

    #[test]
    fn a_defined_constant_carries_a_type() {
        // Without this the modulo rewrite cannot tell what `RESOLUTION` is.
        let source = "#define RESOLUTION 64\nfloat frequency;\n\tfloat f = frequency % RESOLUTION;\n";
        let out = relax(source);
        assert!(out.contains("mod(frequency, float(RESOLUTION))"), "got {out}");
    }

    #[test]
    fn declared_types_sees_uniforms_locals_and_parameters() {
        let types = declared_types("uniform vec2 g_Scale;\nfloat helper(vec3 n) { int k = 1; }");
        assert_eq!(types.get("g_Scale").map(|t| t.components), Some(2));
        assert_eq!(types.get("n").map(|t| t.components), Some(3));
        assert_eq!(types.get("k").map(|t| t.kind), Some(Kind::Int));
    }
}

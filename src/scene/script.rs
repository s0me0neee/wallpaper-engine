//! Running a text layer's own JavaScript.
//!
//! A text layer's string is usually not a string at all but a script, and what
//! `scene.json` stores beside it is the design-time preview the editor last saw
//! (plan.md §4.15). That preview is frozen: `scene_example8` ships a clock
//! reading `"12:34"` and `scene_example3` one reading the same, and a wallpaper
//! that has been running for years still shows it. Sometimes the preview was
//! never drawable — `"<Date>"`, `"<Time and Date>"` — and those layers were
//! dropped entirely.
//!
//! The scripts are small ES modules over a host Wallpaper Engine provides:
//!
//! ```js
//! export var scriptProperties = createScriptProperties()
//!     .addCheckbox({ name: 'use24hFormat', value: true })
//!     .addText({ name: 'delimiter', value: ':' })
//!     .finish();
//!
//! export function update(value) { … return value; }
//! ```
//!
//! Only that contract is implemented: build `scriptProperties`, overlay the
//! layer's own settings, call `update` with the stored value and take what comes
//! back. Wallpaper Engine's scripting reaches much further — it drives layer
//! positions, colours and visibility, and exposes `engine`, `thisLayer` and the
//! `wevector`/`wecolor` modules — and none of that is here. Anything a script
//! does beyond this contract fails, and a failure falls back to the stored
//! value, which is exactly where we were before.
//!
//! `export` is stripped rather than a module loader wired up: these are
//! single-file scripts with no imports between them, so a module graph would be
//! plumbing with nothing to resolve.

use rquickjs::{Context, Runtime};
use serde_json::{Map, Value};

/// Wallpaper Engine's own host surface, as far as this contract needs it.
///
/// `createScriptProperties` is a builder whose `addX` calls each declare one
/// setting with a default and return the builder; `finish` hands back the
/// settings as a plain object. The engine's editor uses the rest of each
/// spec — labels, ranges, ordering — to draw its properties panel, which is
/// why the whole spec is passed and only `name` and `value` are read.
const HOST: &str = r"
globalThis.createScriptProperties = function () {
    var values = {};
    var builder = {};
    var add = function (spec) {
        if (spec && spec.name !== undefined) { values[spec.name] = spec.value; }
        return builder;
    };
    ['addCheckbox', 'addText', 'addTextArea', 'addSlider', 'addColor',
     'addCombo', 'addSpinner', 'addFilePicker', 'addLabel']
        .forEach(function (name) { builder[name] = add; });
    builder.finish = function () { return values; };
    return builder;
};
";

/// Rewrite an ES module's exports into plain declarations.
fn strip_exports(source: &str) -> String {
    source
        .replace("export function", "function")
        .replace("export var", "var")
        .replace("export let", "let")
        .replace("export const", "const")
        .replace("export default", "var weDefault =")
}

/// Run `script`'s `update` and return what it produced.
///
/// `properties` are the layer's own settings, which override the defaults the
/// script declares; `value` is the stored string, handed to `update` the way
/// Wallpaper Engine hands it the property's current value.
pub fn run_text(script: &str, properties: Option<&Map<String, Value>>, value: &str) -> Result<String, String> {
    let runtime = Runtime::new().map_err(|error| format!("starting the script runtime: {error}"))?;
    let context = Context::full(&runtime).map_err(|error| format!("creating a script context: {error}"))?;

    context.with(|ctx| {
        let describe = |error: rquickjs::Error| match ctx.catch().as_exception() {
            Some(exception) => exception.to_string(),
            None => error.to_string(),
        };

        ctx.eval::<(), _>(HOST).map_err(|error| format!("the script host: {}", describe(error)))?;
        ctx.eval::<(), _>(strip_exports(script)).map_err(|error| format!("running the script: {}", describe(error)))?;

        // The declared defaults are in place now, so the layer's own settings
        // go on top. Assigning rather than replacing keeps any property the
        // script declared but the layer never overrode.
        let overlay = Value::Object(properties.cloned().unwrap_or_default()).to_string();
        // `typeof` rather than a truth test: a script need not declare
        // `scriptProperties` at all, and a bare reference to an undeclared name
        // is a ReferenceError rather than `undefined`.
        let assign = format!(
            "(function (o) {{ if (typeof scriptProperties === 'object' && scriptProperties) \
             {{ for (var k in o) {{ scriptProperties[k] = o[k]; }} }} }})({overlay});"
        );
        ctx.eval::<(), _>(assign).map_err(|error| format!("applying script properties: {}", describe(error)))?;

        let call = format!("String(update({}))", Value::String(value.to_string()));
        let has_update = ctx.eval::<bool, _>("typeof update === 'function'").unwrap_or(false);
        if !has_update {
            return Err("the script exports no update()".to_string());
        }
        ctx.eval::<String, _>(call).map_err(|error| format!("calling update(): {}", describe(error)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scripts_declared_defaults_reach_update() {
        let script = "export var scriptProperties = createScriptProperties()\n\
             .addText({ name: 'sep', value: '-' }).finish();\n\
             export function update(value) { return value + scriptProperties.sep; }";
        assert_eq!(run_text(script, None, "a"), Ok("a-".to_string()));
    }

    #[test]
    fn the_layers_own_settings_override_the_declared_default() {
        // `scene.json` carries the user's choice per layer; the script only
        // carries what it shipped with.
        let script = "export var scriptProperties = createScriptProperties()\n\
             .addCheckbox({ name: 'flag', value: true }).finish();\n\
             export function update(value) { return String(scriptProperties.flag); }";
        let mut properties = Map::new();
        properties.insert("flag".to_string(), Value::Bool(false));
        assert_eq!(run_text(script, Some(&properties), ""), Ok("false".to_string()));
    }

    #[test]
    fn a_script_that_throws_reports_why_instead_of_panicking() {
        let script = "export function update(value) { throw new Error('nope'); }";
        let error = run_text(script, None, "").expect_err("the script throws");
        assert!(error.contains("nope"), "{error}");
    }

    #[test]
    fn a_script_with_no_update_is_an_error_not_an_empty_string() {
        // Some layers carry only event hooks (`mediaPropertiesChange`), which
        // this contract cannot drive; falling back to the stored value is right.
        let script = "export var scriptProperties = createScriptProperties().finish();";
        assert!(run_text(script, None, "kept").is_err());
    }

    #[test]
    fn the_clock_contract_from_the_corpus_runs() {
        // `scene_example8`'s clock, trimmed to the shape that matters: it reads
        // the real clock, so only the format is asserted.
        let script = "export var scriptProperties = createScriptProperties()\n\
             .addCheckbox({ name: 'use24hFormat', value: true })\n\
             .addText({ name: 'delimiter', value: ':' }).finish();\n\
             export function update(value) {\n\
               let time = new Date();\n\
               var hours = time.getHours();\n\
               if (!scriptProperties.use24hFormat) { hours %= 12; if (hours == 0) { hours = 12; } }\n\
               hours = ('00' + hours).slice(-2);\n\
               let minutes = ('00' + time.getMinutes()).slice(-2);\n\
               return hours + scriptProperties.delimiter + minutes;\n\
             }";
        let text = run_text(script, None, "12:34").expect("the clock runs");
        assert_eq!(text.len(), 5, "{text}");
        assert_eq!(text.as_bytes()[2], b':', "{text}");
        assert!(text.bytes().enumerate().all(|(i, b)| i == 2 || b.is_ascii_digit()), "{text}");
    }
}

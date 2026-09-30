//! `SceneScript`: the JavaScript a wallpaper runs every frame to drive its own values.
//!
//! Any value in `scene.json` can be a script instead of a number (`model::PropertyScript`), and 53 of an
//! 87-scene library use one: `visible` 277 times, `scale` 267, `origin` 257, `alpha` 189. Each is an ES
//! module that WE calls `init(value)` on once and `update(value)` on every frame, taking what comes back as
//! the property's new value. They reach one another through `shared`, other layers through
//! `thisScene.getLayer`, and the host through `engine` (clock, time of day, canvas, user settings, audio,
//! timers) — `scripting.js` is that host.
//!
//! One `QuickJS` context runs the whole scene, since `shared` and the layers are scene-wide; each script is
//! wrapped in a function of its own, which is the isolation a module would have given it. A script that
//! fails to load or throws is dropped from then on, and its value stays wherever it last was — at worst
//! the published value, which is where a scene without scripting was.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use regex::Regex;
use rquickjs::{Context, Ctx, Runtime};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::scene::model::{Object, Scene};

const HOST: &str = include_str!("scripting.js");

/// How long one call into the scripts may run before it is interrupted: a script stuck in a loop costs
/// a frame, not the wallpaper.
const BUDGET: Duration = Duration::from_millis(250);

/// A layer as the scripts left it after a step, `angles` back in radians. Index-aligned with `scene.objects`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LayerState {
    pub visible: bool,
    pub alpha: f32,
    pub origin: [f32; 3],
    pub scale: [f32; 3],
    pub angles: [f32; 3],
    pub color: [f32; 3],
    pub text: Option<String>,
}

/// One step's result.
#[derive(Debug, Deserialize)]
pub struct Frame {
    pub layers: Vec<LayerState>,
    /// The value of each script not bound to a layer field (an effect uniform, say), in registration order.
    pub values: Vec<Value>,
    /// Why each script stopped, if it did.
    pub failures: Vec<Option<String>>,
    /// What the scripts logged through `console`, capped.
    pub messages: Vec<String>,
}

/// Where each registered script sits: `(object index, path)`, in the order `Frame::values` lists them.
pub type Bindings = Vec<(usize, String)>;

pub struct ScriptHost {
    // Declared before `runtime` so the context is dropped first.
    context: Context,
    runtime: Runtime,
    deadline: Rc<Cell<Option<Instant>>>,
    pub bindings: Bindings,
    /// Scripts that did not even compile, with why.
    pub load_failures: Vec<String>,
}

/// The scene's scripts, loaded and `init`ed, or `None` when it has none.
///
/// `properties` is `project.json`'s `general.properties` (what `engine.userProperties` exposes);
/// `epoch_ms` pins the scripts' clock (`Date`, `engine.timeOfDay`) to a moment, for comparing against a
/// capture taken then — `None` is the real clock.
pub fn start(scene: &Scene, properties: &Map<String, Value>, canvas: (u32, u32), epoch_ms: Option<f64>) -> Result<Option<ScriptHost>> {
    if scene.objects.iter().all(|object| object.scripts.is_empty()) {
        return Ok(None);
    }
    let runtime = Runtime::new().context("starting the script runtime")?;
    runtime.set_memory_limit(512 << 20);
    let deadline: Rc<Cell<Option<Instant>>> = Rc::new(Cell::new(None));
    let watch = Rc::clone(&deadline);
    runtime.set_interrupt_handler(Some(Box::new(move || watch.get().is_some_and(|at| Instant::now() > at))));
    let context = Context::full(&runtime).context("creating the script context")?;
    let mut host = ScriptHost { context, runtime, deadline, bindings: Vec::new(), load_failures: Vec::new() };

    let user: Map<String, Value> =
        properties.iter().filter_map(|(name, spec)| Some((name.clone(), spec.get("value")?.clone()))).collect();
    let config = json!({
        "canvas": [canvas.0, canvas.1],
        "userProperties": user,
        "epoch": epoch_ms,
        "layers": scene.objects.iter().map(layer_config).collect::<Vec<_>>(),
    });
    host.eval(HOST).map_err(|error| anyhow!("the script host: {error}"))?;
    host.eval(&format!("__we.configure({config});")).map_err(|error| anyhow!("configuring the scripts: {error}"))?;

    for (index, object) in scene.objects.iter().enumerate() {
        for script in &object.scripts {
            let properties = script.properties.as_ref().map_or(Value::Null, |map| Value::Object(map.clone()));
            let registration = format!(
                "__we.register({index}, {path}, function (thisLayer, thisObject) {{\n{body}\n;return {{ \
                 init: typeof init === 'function' ? init : undefined, \
                 update: typeof update === 'function' ? update : undefined, \
                 applyUserProperties: typeof applyUserProperties === 'function' ? applyUserProperties : undefined, \
                 properties: () => (typeof scriptProperties === 'object' ? scriptProperties : undefined) }}; }}, \
                 {properties}, {initial});",
                path = Value::String(script.path.clone()),
                body = module_body(&script.script),
                initial = script.initial,
            );
            match host.eval(&registration) {
                Ok(()) => host.bindings.push((index, script.path.clone())),
                Err(error) => host.load_failures.push(format!("{}.{}: {error}", crate::scene::model::label(object), script.path)),
            }
        }
    }
    host.run_scripts()?;
    Ok(Some(host))
}

/// Advance every script to `time` seconds, `dt` after the last step, and read the layers back.
pub fn step(host: &ScriptHost, time: f32, dt: f32) -> Result<Frame> {
    if host.eval(&format!("__we.begin({time}, {dt});")).is_err() {
        // A timer that would not finish; timers are few, so all of them go rather than a guess at which.
        host.eval("__we.dropTimers();").map_err(|error| anyhow!("dropping the timers: {error}"))?;
    }
    host.run_scripts()?;
    let state = host.eval_string("__we.state()").map_err(|error| anyhow!("reading the scripts' state: {error}"))?;
    let mut frame: Frame = serde_json::from_str(&state).context("parsing the scripts' state")?;
    for layer in &mut frame.layers {
        layer.angles = layer.angles.map(f32::to_radians);
    }
    Ok(frame)
}

/// A script's value as uniform components: a number, a bool, a `"r g b"` string, an array or a vector.
pub fn floats(value: &Value) -> Option<Vec<f32>> {
    #[expect(clippy::cast_possible_truncation, reason = "shader uniforms are f32")]
    let number = |value: &Value| value.as_f64().map(|n| n as f32).or_else(|| value.as_bool().map(|b| f32::from(u8::from(b))));
    match value {
        Value::String(text) => text.split_whitespace().map(|part| part.parse().ok()).collect(),
        Value::Array(items) => items.iter().map(number).collect(),
        Value::Object(map) => {
            let components: Vec<f32> = ["x", "y", "z", "w"].iter().map_while(|key| map.get(*key).and_then(number)).collect();
            (!components.is_empty()).then_some(components)
        }
        other => number(other).map(|n| vec![n]),
    }
}

/// Draw a layer whose alpha a script drives at full alpha, so the compositor can apply what the script
/// says: baked in, a layer published at 0 and faded in by its script would have nothing to show. Call it
/// after `start`, which hands the scripts the published alpha.
pub fn unbake_scripted_alpha(scene: &mut Scene) {
    for object in &mut scene.objects {
        if object.scripts.iter().any(|script| script.path == "alpha") {
            object.alpha = 1.0;
        }
    }
}

impl ScriptHost {
    /// Run the current pass over every script, resuming past each one the budget interrupts.
    fn run_scripts(&self) -> Result<()> {
        for _ in 0..=self.bindings.len() {
            if self.eval("__we.run();").is_ok() {
                return Ok(());
            }
            self.eval("__we.interrupted();").map_err(|error| anyhow!("recovering from a script: {error}"))?;
        }
        Err(anyhow!("the scripts kept overrunning"))
    }

    fn eval(&self, source: &str) -> Result<(), String> {
        self.guarded(|ctx| ctx.eval::<(), _>(source))
    }

    fn eval_string(&self, source: &str) -> Result<String, String> {
        self.guarded(|ctx| ctx.eval::<String, _>(source))
    }

    fn guarded<T>(&self, run: impl for<'js> FnOnce(&Ctx<'js>) -> rquickjs::Result<T>) -> Result<T, String> {
        self.deadline.set(Some(Instant::now() + BUDGET));
        let result = self.context.with(|ctx| {
            run(&ctx).map_err(|error| match ctx.catch().as_exception() {
                Some(exception) => exception.to_string(),
                None => error.to_string(),
            })
        });
        self.deadline.set(None);
        self.runtime.run_gc();
        result
    }
}

/// An object's fields as the scripts see them on its layer.
fn layer_config(object: &Object) -> Value {
    let vector = |v: crate::scene::model::Vec3| json!([v.x, v.y, v.z]);
    json!({
        "id": object.id,
        "name": object.name,
        "parentId": object.parent,
        "origin": vector(object.origin),
        "scale": vector(object.scale),
        // Scripts see degrees (3219908811 swings a 5-degree `angles.z` by `smoothValue * 30`); scene.json radians.
        "angles": [object.angles.x.to_degrees(), object.angles.y.to_degrees(), object.angles.z.to_degrees()],
        "color": vector(object.color),
        "size": object.size.map_or(json!([0, 0, 0]), vector),
        "visible": object.visible,
        "alpha": object.alpha,
        "brightness": object.brightness,
        "text": object.text,
        "parallaxDepth": [1, 1],
    })
}

/// A script module's source as the body of a function: exports become plain declarations and imports
/// read the host's modules.
fn module_body(source: &str) -> String {
    static NAMESPACE: OnceLock<Regex> = OnceLock::new();
    static NAMED: OnceLock<Regex> = OnceLock::new();
    #[expect(clippy::unwrap_used, reason = "fixed literal patterns")]
    let (namespace, named) = (
        NAMESPACE.get_or_init(|| Regex::new(r#"import\s+\*\s+as\s+(\w+)\s+from\s+['"]([\w/.]+)['"]\s*;?"#).unwrap()),
        NAMED.get_or_init(|| Regex::new(r#"import\s*\{([^}]*)\}\s*from\s*['"]([\w/.]+)['"]\s*;?"#).unwrap()),
    );
    let source = namespace.replace_all(source, "const $1 = __we.modules['$2'] || {};");
    let source = named.replace_all(&source, |caps: &regex::Captures| {
        format!("const {{{}}} = __we.modules['{}'] || {{}};", caps[1].replace(" as ", ": "), &caps[2])
    });
    crate::scene::script::strip_exports(&source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::model::parse_scene;

    fn scene(json: &str) -> Scene {
        parse_scene(json.as_bytes(), &Map::new()).unwrap()
    }

    fn run(json: &str, steps: &[f32]) -> Frame {
        let scene = scene(json);
        let host = start(&scene, &Map::new(), (1920, 1080), Some(0.0)).unwrap().unwrap();
        let mut last = 0.0;
        let mut frame = None;
        for &time in steps {
            frame = Some(step(&host, time, time - last).unwrap());
            last = time;
        }
        frame.unwrap()
    }

    #[test]
    fn update_drives_its_property_and_shared_links_scripts() {
        // 3438699689: one layer's alpha waits on a flag another layer's script sets.
        let frame = run(
            r#"{"objects":[
                {"id":1,"image":"models/a.json","visible":{"script":"export function init(value) { shared.kj = true; return value; }","value":true}},
                {"id":2,"image":"models/b.json","alpha":{"script":"'use strict';\nexport function update(value) { return shared.kj ? 0.2 : 0; }","value":0.6}}
            ]}"#,
            &[0.1],
        );
        assert!((frame.layers[1].alpha - 0.2).abs() < 1e-6);
        assert!(frame.failures.iter().all(Option::is_none), "{:?}", frame.failures);
    }

    #[test]
    fn a_number_returned_for_a_vector_spreads_over_it() {
        let frame = run(
            r#"{"objects":[{"id":1,"image":"models/a.json","scale":{"script":"let s; export function init(v) { s = v.x; } export function update() { return s * 2; }","value":"1.5 1.5 1"}}]}"#,
            &[0.1],
        );
        assert_eq!(frame.layers[0].scale, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn modules_timers_and_other_layers_are_reachable() {
        let frame = run(
            r#"{"objects":[
                {"id":1,"name":"target","image":"models/a.json"},
                {"id":2,"image":"models/b.json","origin":{"script":"import * as WEMath from 'WEMath';\nexport function init(v) { engine.setTimeout(() => { thisScene.getLayer('target').visible = false; }, 500); }\nexport function update(v) { return new Vec3(WEMath.mix(0, 100, 0.5), v.y, v.z); }","value":"0 0 0"}}
            ]}"#,
            &[0.25, 0.75],
        );
        assert!(!frame.layers[0].visible, "the timer fired at 0.5 s");
        assert_eq!(frame.layers[1].origin, [50.0, 0.0, 0.0]);
    }

    #[test]
    fn the_pinned_clock_sets_the_time_of_day() {
        // Noon UTC on 2026-09-29, and a script that shows a layer only by day.
        let scene = scene(
            r#"{"objects":[{"id":1,"image":"models/a.json","visible":{"script":"export function update() { const h = new Date().getUTCHours(); return h >= 7 && h < 18; }","value":false}}]}"#,
        );
        let host = start(&scene, &Map::new(), (1920, 1080), Some(1_790_683_200_000.0)).unwrap().unwrap();
        assert!(step(&host, 0.0, 0.0).unwrap().layers[0].visible);
    }

    #[test]
    fn a_broken_or_runaway_script_is_dropped_not_fatal() {
        let frame = run(
            r#"{"objects":[
                {"id":1,"image":"models/a.json","alpha":{"script":"export function update() { while (true) {} }","value":0.5}},
                {"id":2,"image":"models/b.json","alpha":{"script":"export function update( {","value":0.5}},
                {"id":3,"image":"models/c.json","alpha":{"script":"export function update() { return 0.25; }","value":0.5}}
            ]}"#,
            &[0.1, 0.2],
        );
        assert!(frame.failures[0].is_some(), "the loop was interrupted");
        assert!((frame.layers[0].alpha - 0.5).abs() < 1e-6, "and its value stays where it was");
        assert!((frame.layers[2].alpha - 0.25).abs() < 1e-6, "the others still run");
    }
}

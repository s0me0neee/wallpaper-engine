// Wallpaper Engine's SceneScript host, as far as the library's scripts reach.
//
// Each scripted value is its own module in WE; here it is a factory function, so its `let`s and `var`s stay
// its own. They share `shared`, the layers, `engine`, `input` and the clock. `__we` is ours: the Rust side
// registers scripts through it, steps it once a frame and reads the layers back.
'use strict';

(function () {
    const vectorArgs = (length, args) => {
        if (args.length === 1 && Array.isArray(args[0])) {
            return Array.from({ length }, (_, i) => Number(args[0][i] ?? args[0][0] ?? 0));
        }
        if (args.length === 1 && typeof args[0] === 'object' && args[0] !== null) {
            return [args[0].x, args[0].y, args[0].z, args[0].w].slice(0, length).map((v) => v ?? 0);
        }
        if (args.length === 1 && typeof args[0] === 'string') {
            const parts = args[0].trim().split(/\s+/).map(Number);
            return Array.from({ length }, (_, i) => parts[i] ?? parts[0] ?? 0);
        }
        if (args.length === 1 && typeof args[0] === 'number') {
            return Array(length).fill(args[0]);
        }
        return Array.from({ length }, (_, i) => (typeof args[i] === 'number' ? args[i] : 0));
    };

    const vectorClass = (names) => {
        const length = names.length;
        class Vector {
            constructor(...args) {
                vectorArgs(length, args).forEach((v, i) => { this[names[i]] = v; });
            }
            static fromArray(values) { return new Vector(...values); }
            toArray() { return names.map((n) => this[n]); }
            map(f, other) {
                const o = typeof other === 'number' || other === undefined ? null : other;
                return new Vector(...names.map((n) => f(this[n], o ? o[n] ?? 0 : other)));
            }
            add(o) { return this.map((a, b) => a + b, o); }
            subtract(o) { return this.map((a, b) => a - b, o); }
            multiply(o) { return this.map((a, b) => a * b, o); }
            divide(o) { return this.map((a, b) => a / b, o); }
            min(o) { return this.map((a, b) => Math.min(a, b), o); }
            max(o) { return this.map((a, b) => Math.max(a, b), o); }
            mix(o, t) { return this.map((a, b) => a + (b - a) * t, o); }
            abs() { return this.map(Math.abs); }
            floor() { return this.map(Math.floor); }
            ceil() { return this.map(Math.ceil); }
            round() { return this.map(Math.round); }
            negate() { return this.map((a) => -a); }
            copy() { return new Vector(this); }
            dot(o) { return names.reduce((sum, n) => sum + this[n] * (o[n] ?? 0), 0); }
            lengthSqr() { return this.dot(this); }
            length() { return Math.sqrt(this.lengthSqr()); }
            distance(o) { return this.subtract(o).length(); }
            normalize() { const l = this.length(); return l > 0 ? this.divide(l) : this.copy(); }
            reflect(n) { return this.subtract(n.multiply(2 * this.dot(n))); }
            equals(o) { return names.every((n) => this[n] === o[n]); }
            cross(o) {
                return new Vector(this.y * o.z - this.z * o.y, this.z * o.x - this.x * o.z, this.x * o.y - this.y * o.x);
            }
            perpendicular() { return new Vector(-this.y, this.x); }
            toString() { return names.map((n) => this[n]).join(' '); }
        }
        return Vector;
    };
    const Vec2 = vectorClass(['x', 'y']);
    const Vec3 = vectorClass(['x', 'y', 'z']);
    const Vec4 = vectorClass(['x', 'y', 'z', 'w']);
    Object.assign(globalThis, { Vec2, Vec3, Vec4 });

    const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
    const WEMath = {
        PI: Math.PI, pi: Math.PI, deg2rad: Math.PI / 180, rad2deg: 180 / Math.PI,
        mix: (a, b, t) => a + (b - a) * t,
        clamp,
        smoothStep: (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); },
        smoothstep: (a, b, x) => WEMath.smoothStep(a, b, x),
        fract: (x) => x - Math.floor(x),
        sign: Math.sign,
    };
    const hsv2rgb = (c) => {
        const h = ((c.x % 1) + 1) % 1, s = c.y, v = c.z;
        const k = (n) => { const m = (n + h * 6) % 6; return v - v * s * Math.max(0, Math.min(m, 4 - m, 1)); };
        return new Vec3(k(5), k(3), k(1));
    };
    const rgb2hsv = (c) => {
        const max = Math.max(c.x, c.y, c.z), min = Math.min(c.x, c.y, c.z), d = max - min;
        let h = 0;
        if (d > 0) {
            if (max === c.x) h = ((c.y - c.z) / d) % 6;
            else if (max === c.y) h = (c.z - c.x) / d + 2;
            else h = (c.x - c.y) / d + 4;
            h = ((h / 6) % 1 + 1) % 1;
        }
        return new Vec3(h, max > 0 ? d / max : 0, max);
    };
    const WEColor = { hsv2rgb, rgb2hsv, normalizeColor: (c) => new Vec3(c).divide(255), expandColor: (c) => new Vec3(c).multiply(255) };
    const WEVector = {
        Vec2, Vec3, Vec4,
        angleVector2: (angle) => new Vec2(Math.cos(angle * WEMath.deg2rad), Math.sin(angle * WEMath.deg2rad)),
        vectorAngle2: (v) => Math.atan2(v.y, v.x) * WEMath.rad2deg,
    };
    const modules = { WEMath, WEColor, WEVector };

    // The clock. `now` is pinned when the Rust side asks (to match a capture), else the real one.
    const RealDate = Date;
    let epoch = null;
    let runtime = 0;
    const now = () => (epoch === null ? RealDate.now() : epoch + runtime * 1000);
    class PinnedDate extends RealDate {
        constructor(...args) { if (args.length === 0) super(now()); else super(...args); }
        static now() { return now(); }
    }
    globalThis.Date = PinnedDate;

    let timers = [];
    let timerId = 0;
    const schedule = (callback, ms, repeat) => {
        timerId += 1;
        timers.push({ id: timerId, callback, due: runtime + (ms || 0) / 1000, every: repeat ? Math.max(ms || 0, 1) / 1000 : 0 });
        return timerId;
    };
    const cancel = (id) => { timers = timers.filter((t) => t.id !== id); };

    const audio = (resolution) => {
        const n = resolution || 16;
        return { left: new Float32Array(n), right: new Float32Array(n), average: new Float32Array(n), resolution: n };
    };

    const engine = {
        frametime: 0, runtime: 0, timeOfDay: 0,
        canvasSize: new Vec2(1920, 1080), screenResolution: new Vec2(1920, 1080),
        userProperties: {},
        AUDIO_RESOLUTION_16: 16, AUDIO_RESOLUTION_32: 32, AUDIO_RESOLUTION_64: 64,
        registerAudioBuffers: audio,
        setTimeout: (f, ms) => schedule(f, ms, false), setInterval: (f, ms) => schedule(f, ms, true),
        clearTimeout: cancel, clearInterval: cancel,
        isDesktopDevice: () => true, isMobileDevice: () => false, isWallpaper: () => true, isScreensaver: () => false,
        isRunningInEditor: () => false, isPortrait: () => false, isLandscape: () => true,
        openUserShortcut: () => {},
    };
    const input = { cursorWorldPosition: new Vec3(960, 540, 0), cursorScreenPosition: new Vec2(0.5, 0.5), cursorLeftDown: false };
    const messages = [];
    const note = (...args) => { if (messages.length < 50) messages.push(args.map(String).join(' ')); };
    const storage = new Map();

    const animation = () => ({
        frameCount: 1, rate: 1, duration: 0, visible: true,
        play() {}, pause() {}, stop() {}, setFrame() {}, getFrame: () => 0, isPlaying: () => false,
        playSingleAnimation() {}, name: '',
        addEndedCallback() {}, addFrameCallback() {}, removeEndedCallback() {}, removeFrameCallback() {},
    });
    const material = () => ({ setValue() {}, getValue: () => undefined });
    const vectorFields = ['origin', 'scale', 'angles', 'color', 'size'];

    const layers = [];
    const byId = new Map();
    class Layer {
        constructor(state) {
            Object.assign(this, state);
            for (const field of vectorFields) this[field] = new Vec3(state[field]);
            this.parallaxDepth = new Vec2(state.parallaxDepth);
        }
        getParent() { return this.parentId === null ? null : byId.get(this.parentId) || null; }
        setParent(layer) { this.parentId = layer ? layer.id : null; }
        getChildren() { return layers.filter((l) => l.parentId === this.id); }
        getTextureAnimation() { return animation(); }
        getAnimation() { return animation(); }
        getAnimationLayer() { return animation(); }
        getAnimationLayerCount() { return 0; }
        getEffect() { return { visible: true, name: '', getMaterial: material }; }
        getMaterial() { return material(); }
        // A sound layer's; nothing plays here.
        play() {}
        stop() {}
        pause() {}
        isPlaying() { return false; }
        getEffectCount() { return 0; }
        getParticleSystem() { return null; }
        getVideoTexture() { return null; }
        emitParticles() {}
        getTransformMatrix() { return { m: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }; }
    }
    const thisScene = {
        getLayer: (key) => (typeof key === 'number' ? byId.get(key) : layers.find((l) => l.name === key)) || null,
        getLayerIndex: (layer) => layers.indexOf(layer),
        enumerateLayers: () => layers.slice(),
        getInitialLayerConfig: () => null,
        createLayer: () => null, destroyLayer() {}, sortLayer() {},
        bloom: false, bloomstrength: 1, bloomthreshold: 1,
    };

    Object.assign(globalThis, {
        engine, input, shared: {}, thisScene, WEMath, WEColor, WEVector,
        console: { log: note, warn: note, error: note, info: note, debug: note },
        localStorage: {
            get: (k) => storage.get(k), set: (k, v) => { storage.set(k, v); }, delete: (k) => { storage.delete(k); },
            clear: () => storage.clear(), LOCATION_GLOBAL: 0, LOCATION_SCREEN: 1,
        },
        setTimeout: engine.setTimeout, setInterval: engine.setInterval,
        clearTimeout: cancel, clearInterval: cancel,
        MediaPlaybackEvent: { PLAYBACK_STOPPED: 0, PLAYBACK_PLAYING: 1, PLAYBACK_PAUSED: 2 },
        createScriptProperties: () => {
            const values = {};
            const builder = {};
            const add = (spec) => { if (spec && spec.name !== undefined) values[spec.name] = spec.value; return builder; };
            ['addCheckbox', 'addText', 'addTextArea', 'addSlider', 'addColor', 'addCombo', 'addSpinner',
                'addFilePicker', 'addDirectoryPicker', 'addLabel', 'addTextInput']
                .forEach((name) => { builder[name] = add; });
            builder.finish = () => values;
            return builder;
        },
    });

    // A property's value in the shape the layer keeps it: a number spreads over a vector.
    const coerce = (field, value) => {
        if (vectorFields.includes(field)) return new Vec3(value);
        if (field === 'parallaxDepth') return new Vec2(value);
        if (field === 'visible') return !!value;
        if (field === 'alpha') return Number(value);
        return value;
    };
    const layerFields = new Set([...vectorFields, 'visible', 'alpha', 'text', 'parallaxDepth', 'brightness']);

    const scripts = [];
    const fail = (script, stage, error) => {
        script.failed = `${stage}: ${error && error.message ? error.message : error}`;
    };
    // A copy, as WE hands it: 3219908811 keeps `init`'s value as its baseline and mutates `update`'s.
    const get = (script) => {
        const value = script.field ? script.layer[script.field] : script.value;
        return value instanceof Vec3 ? new Vec3(value) : value instanceof Vec2 ? new Vec2(value) : value;
    };
    const set = (script, value) => {
        if (value === undefined) return;
        if (script.field) script.layer[script.field] = coerce(script.field, value);
        else script.value = value;
    };

    globalThis.__we = {
        modules,
        configure(config) {
            engine.canvasSize = new Vec2(config.canvas[0], config.canvas[1]);
            engine.screenResolution = new Vec2(config.canvas[0], config.canvas[1]);
            input.cursorWorldPosition = new Vec3(config.canvas[0] / 2, config.canvas[1] / 2, 0);
            engine.userProperties = config.userProperties;
            epoch = config.epoch;
            for (const state of config.layers) {
                const layer = new Layer(state);
                layers.push(layer);
                byId.set(layer.id, layer);
            }
        },
        register(index, path, factory, properties, initial) {
            const layer = layers[index];
            const field = layerFields.has(path) ? path : null;
            const script = { layer, path, field, value: initial, module: null, failed: null };
            scripts.push(script);
            try {
                script.module = factory(layer, layer);
                const own = script.module.properties();
                if (own && properties) Object.assign(own, properties);
            } catch (error) {
                fail(script, 'loading', error);
            }
        },
        // A script QuickJS interrupts cannot catch that itself, so the pass keeps a cursor: `run` goes on from
        // the script after the one being called, and `interrupted` marks that one failed.
        cursor: 0,
        stage: 'init',
        run() {
            while (this.cursor < scripts.length) {
                const script = scripts[this.cursor];
                this.cursor += 1;
                if (script.failed || !script.module) continue;
                try {
                    if (this.stage === 'init') {
                        if (typeof script.module.init === 'function') set(script, script.module.init(get(script)));
                    } else if (this.stage === 'apply') {
                        // After every init, not after each: 3497488774's layers apply their settings through
                        // helpers a later layer's init has not finished preparing yet.
                        if (typeof script.module.applyUserProperties === 'function') {
                            script.module.applyUserProperties(engine.userProperties);
                        }
                    } else if (typeof script.module.update === 'function') {
                        set(script, script.module.update(get(script)));
                    }
                } catch (error) {
                    fail(script, this.stage, error);
                }
            }
        },
        interrupted() {
            if (this.cursor > 0) fail(scripts[this.cursor - 1], this.stage, 'interrupted: ran past its budget');
        },
        dropTimers() { timers = []; },
        begin_stage(stage) { this.stage = stage; this.cursor = 0; },
        begin(time, dt) {
            runtime = time;
            engine.runtime = time;
            engine.frametime = dt;
            const d = new PinnedDate();
            engine.timeOfDay = (d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds()) / 86400;
            const due = timers.filter((t) => t.due <= time);
            timers = timers.filter((t) => t.due > time || t.every > 0);
            for (const timer of due) {
                while (timer.every > 0 && timer.due <= time) timer.due += timer.every;
                try { timer.callback(); } catch (error) { note('timer:', error); }
            }
            this.cursor = 0;
            this.stage = 'update';
        },
        state() {
            // Whatever a script left in a vector field, a vector comes out.
            const vector = (v) => new Vec3(v ?? 0).toArray().map((c) => Number(c) || 0);
            return JSON.stringify({
                layers: layers.map((l) => ({
                    visible: !!l.visible, alpha: Number(l.alpha), origin: vector(l.origin), scale: vector(l.scale),
                    angles: vector(l.angles), color: vector(l.color), text: typeof l.text === 'string' ? l.text : null,
                })),
                values: scripts.map((s) => (s.field ? null : s.value === undefined ? null : s.value)),
                failures: scripts.map((s) => s.failed),
                messages,
            });
        },
    };
})();

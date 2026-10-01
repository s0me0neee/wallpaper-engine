//! GPU time per labelled span of a frame (`SIMULATE_PROFILE`), from GL timestamp queries.
//!
//! The fps line's `gpu` only counts issuing commands; this is what the GPU actually spent. Each frame's
//! queries are read back `LAG` frames later, by which time they have landed, so profiling never stalls
//! the pipeline it is measuring.

use glow::HasContext;
use std::collections::HashMap;

const LAG: usize = 3;

#[derive(Default)]
struct Frame {
    /// One timestamp query per mark, reused frame to frame.
    queries: Vec<glow::Query>,
    /// The label of the span each mark opens; the last mark only closes.
    labels: Vec<String>,
}

pub struct Timers {
    frames: Vec<Frame>,
    current: usize,
    /// Nanoseconds per label, summed since the last `take`, and the frames that sum covers.
    totals: HashMap<String, u64>,
    frames_counted: u32,
}

pub fn new() -> Timers {
    Timers { frames: (0..LAG).map(|_| Frame::default()).collect(), current: 0, totals: HashMap::new(), frames_counted: 0 }
}

/// Close the running span and open one called `label`.
pub fn mark(gl: &glow::Context, timers: &mut Timers, label: String) {
    let frame = &mut timers.frames[timers.current];
    let index = frame.labels.len();
    if index == frame.queries.len() {
        // Safety: a plain query object on the current context.
        match unsafe { gl.create_query() } {
            Ok(query) => frame.queries.push(query),
            Err(_) => return,
        }
    }
    // Safety: the query was created on this context and is not in use by an active begin/end pair.
    unsafe { gl.query_counter(frame.queries[index], glow::TIMESTAMP) };
    frame.labels.push(label);
}

/// Close the frame's last span, then fold in the frame whose queries have had `LAG` frames to land.
pub fn end_frame(gl: &glow::Context, timers: &mut Timers) {
    mark(gl, timers, String::new());
    timers.current = (timers.current + 1) % LAG;
    let frame = &mut timers.frames[timers.current];
    if frame.labels.len() > 1 {
        // Safety: every query below was written by `query_counter` at least `LAG - 1` frames ago.
        let stamps: Vec<u64> = frame
            .queries
            .iter()
            .take(frame.labels.len())
            .map(|&query| unsafe { gl.get_query_parameter_u64(query, glow::QUERY_RESULT) })
            .collect();
        for (label, pair) in frame.labels.iter().zip(stamps.windows(2)) {
            *timers.totals.entry(label.clone()).or_default() += pair[1].saturating_sub(pair[0]);
        }
        timers.frames_counted += 1;
    }
    frame.labels.clear();
}

/// The `count` most expensive spans as `(label, ms per frame)`, and the whole frame's GPU ms, since the
/// last call.
#[expect(clippy::cast_precision_loss, reason = "nanoseconds per second of frames, nowhere near 2^52")]
pub fn take(timers: &mut Timers, count: usize) -> (Vec<(String, f64)>, f64) {
    let frames = f64::from(timers.frames_counted.max(1));
    let mut spans: Vec<(String, f64)> =
        timers.totals.drain().map(|(label, ns)| (label, ns as f64 / 1e6 / frames)).collect();
    timers.frames_counted = 0;
    let total = spans.iter().map(|(_, ms)| ms).sum();
    spans.sort_by(|a, b| b.1.total_cmp(&a.1));
    spans.truncate(count);
    (spans, total)
}

//! Interleaved timing: every side runs batches of at least 1 ms, the sides alternate sample by
//! sample (ours, ref₁, ref₂, …, ours, …), and each reports the median per operation.
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Set when a reference returned an error (PARI's or NTL's, caught in the shims): the side that
/// ran is then left out of the timing, and its row says why.
static FAILED: AtomicBool = AtomicBool::new(false);
static MESSAGE: Mutex<String> = Mutex::new(String::new());

pub fn ref_failed(msg: &str) {
    FAILED.store(true, Ordering::SeqCst);
    *MESSAGE.lock().unwrap() = msg.to_string();
}

/// A side of a comparison: its name and the work, `f(rep)` one operation on the instance
/// `rep % instances` of the row.
pub struct Side<'a> {
    pub name: String,
    pub f: Box<dyn FnMut(usize) + 'a>,
}

impl<'a> Side<'a> {
    pub fn new(name: impl Into<String>, f: impl FnMut(usize) + 'a) -> Self {
        Side {
            name: name.into(),
            f: Box::new(f),
        }
    }
}

#[allow(dead_code)]
pub struct Timing {
    /// The error of each side that failed on its first run (and was not timed).
    pub failed: Vec<Option<String>>,
    /// Median nanoseconds per operation of each side, in the order given (NaN if it failed).
    pub median: Vec<f64>,
    /// (q3 − q1) / median of each side.
    pub iqr: Vec<f64>,
    pub samples: usize,
    pub batch: Vec<usize>,
}

/// The settings of a run: the samples wanted, the least time of a batch, and the time a row
/// may take before the samples are cut (never below `min_samples`).
#[derive(Clone, Copy)]
pub struct Plan {
    /// One run of each side, no warm-up: to plan the sizes.
    pub probe: bool,
    pub samples: usize,
    pub min_samples: usize,
    pub batch_ns: f64,
    pub row_budget_ns: f64,
}

impl Plan {
    pub fn from_env() -> Plan {
        let quick = std::env::var("LATBENCH_QUICK").is_ok();
        let budget: f64 = std::env::var("LATBENCH_ROW_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if quick { 2.0 } else { 40.0 });
        Plan {
            probe: std::env::var("LATBENCH_PROBE").is_ok(),
            samples: if quick { 3 } else { 15 },
            min_samples: if quick { 1 } else { 5 },
            batch_ns: 1e6,
            row_budget_ns: budget * 1e9,
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn quantile(v: &[f64], q: f64) -> f64 {
    // v sorted
    let x = q * (v.len() - 1) as f64;
    let i = x.floor() as usize;
    let j = (i + 1).min(v.len() - 1);
    v[i] + (x - i as f64) * (v[j] - v[i])
}

pub fn measure(plan: &Plan, sides: &mut [Side<'_>]) -> Timing {
    let n = sides.len();
    let mut rep = vec![0usize; n];
    if plan.probe {
        let mut med = Vec::new();
        for s in sides.iter_mut() {
            let t = Instant::now();
            (s.f)(0);
            med.push(t.elapsed().as_nanos() as f64);
        }
        return Timing {
            failed: vec![None; n],
            iqr: vec![0.0; n],
            median: med,
            samples: 1,
            batch: vec![1; n],
        };
    }
    // Warm up: one run each, then a second for the estimate (the first is the estimate of an
    // operation of a third of a second or more).
    let mut est = vec![0f64; n];
    let mut failed: Vec<Option<String>> = vec![None; n];
    for (i, s) in sides.iter_mut().enumerate() {
        FAILED.store(false, Ordering::SeqCst);
        let t = Instant::now();
        (s.f)(rep[i]);
        rep[i] += 1;
        est[i] = t.elapsed().as_nanos() as f64;
        if FAILED.load(Ordering::SeqCst) {
            failed[i] = Some(MESSAGE.lock().unwrap().clone());
            est[i] = 0.0;
            continue;
        }
        if est[i] < 3e8 {
            let t = Instant::now();
            (s.f)(rep[i]);
            rep[i] += 1;
            est[i] = t.elapsed().as_nanos() as f64;
        }
    }
    // Refine the estimate of fast operations.
    for (i, s) in sides.iter_mut().enumerate() {
        if failed[i].is_none() && est[i] < plan.batch_ns / 10.0 {
            let k = ((plan.batch_ns / 4.0) / est[i].max(20.0)).ceil() as usize;
            let t = Instant::now();
            for _ in 0..k {
                (s.f)(rep[i]);
                rep[i] += 1;
            }
            est[i] = t.elapsed().as_nanos() as f64 / k as f64;
        }
    }
    let batch: Vec<usize> = est
        .iter()
        .map(|&t| ((plan.batch_ns / t.max(1.0)).ceil() as usize).max(1))
        .collect();
    let per_sample: f64 = est.iter().zip(&batch).map(|(t, k)| t * *k as f64).sum();
    let mut samples = plan.samples;
    if per_sample * samples as f64 > plan.row_budget_ns {
        // At least min_samples, three for samples of more than a minute.
        let floor = if per_sample > 6e10 { plan.min_samples.min(3) } else { plan.min_samples };
        samples = ((plan.row_budget_ns / per_sample) as usize).max(floor);
    }
    let mut times = vec![Vec::with_capacity(samples); n];
    for _ in 0..samples {
        for (i, s) in sides.iter_mut().enumerate() {
            if failed[i].is_some() {
                continue;
            }
            let t = Instant::now();
            for _ in 0..batch[i] {
                (s.f)(rep[i]);
                rep[i] += 1;
            }
            times[i].push(t.elapsed().as_nanos() as f64 / batch[i] as f64);
        }
    }
    let mut med = Vec::new();
    let mut iqr = Vec::new();
    for t in &mut times {
        if t.is_empty() {
            med.push(f64::NAN);
            iqr.push(0.0);
            continue;
        }
        let m = median(t);
        let q = (quantile(t, 0.75) - quantile(t, 0.25)) / m;
        med.push(m);
        iqr.push(q);
    }
    Timing {
        failed,
        median: med,
        iqr,
        samples,
        batch,
    }
}

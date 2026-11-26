//! The timing harness: contenders alternated sample by sample (ABAB…, the first one rotating
//! each round), each sample batched to at least `min_ns`, the median over `samples` rounds.

use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Stat {
    /// Median nanoseconds per call.
    pub median: f64,
    /// Interquartile range relative to the median.
    pub iqr: f64,
}

pub struct Cfg {
    pub samples: usize,
    pub min_ns: f64,
    pub warm_ns: f64,
}

impl Cfg {
    pub fn from_env() -> Cfg {
        let get = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(d)
        };
        Cfg {
            samples: get("FFTBENCH_SAMPLES", 15.0) as usize,
            min_ns: get("FFTBENCH_MIN_MS", 1.0) * 1e6,
            warm_ns: get("FFTBENCH_WARM_MS", 3.0) * 1e6,
        }
    }
}

pub type Job<'a> = Box<dyn FnMut() + 'a>;

fn time_iters(f: &mut Job, iters: u64) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_nanos() as f64
}

/// Times every contender; returns the statistics per call in the order given.
pub fn run(jobs: &mut [Job], cfg: &Cfg) -> Vec<Stat> {
    let k = jobs.len();
    // Warm-up: at least two calls and `warm_ns` each.
    for f in jobs.iter_mut() {
        let t = Instant::now();
        let mut calls = 0;
        while calls < 2 || (t.elapsed().as_nanos() as f64) < cfg.warm_ns {
            f();
            calls += 1;
            if calls > 1_000_000_000 {
                break;
            }
        }
    }
    // Calibration: iterations per sample of at least `min_ns`.
    let mut iters = vec![1u64; k];
    for (i, f) in jobs.iter_mut().enumerate() {
        let mut it = 1u64;
        loop {
            let ns = time_iters(f, it);
            if ns >= cfg.min_ns {
                break;
            }
            let grow = if ns <= 0.0 {
                10.0
            } else {
                (cfg.min_ns * 1.25 / ns).clamp(1.5, 10.0)
            };
            it = ((it as f64) * grow).ceil() as u64;
        }
        iters[i] = it;
    }
    let mut per: Vec<Vec<f64>> = vec![Vec::with_capacity(cfg.samples); k];
    for r in 0..cfg.samples {
        for j in 0..k {
            let i = (j + r) % k;
            let ns = time_iters(&mut jobs[i], iters[i]);
            per[i].push(ns / iters[i] as f64);
        }
    }
    per.into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
            let m = q(0.5);
            Stat {
                median: m,
                iqr: (q(0.75) - q(0.25)) / m,
            }
        })
        .collect()
}

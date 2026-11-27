//! Alternating timing: every contender's samples are taken in turn (A B C … A B C …), each
//! batched to at least a millisecond, after a warm-up; the median per operation is reported.

use std::hint::black_box;
use std::time::{Duration, Instant};

pub struct Contender<'a> {
    pub name: String,
    pub f: Box<dyn FnMut() + 'a>,
}

impl<'a> Contender<'a> {
    pub fn new(name: impl Into<String>, f: impl FnMut() + 'a) -> Self {
        Contender { name: name.into(), f: Box::new(f) }
    }
}

#[derive(Clone, Debug)]
pub struct Stat {
    pub name: String,
    pub median: f64,
    pub lo: f64,
    pub hi: f64,
    pub samples: usize,
}

impl Stat {
    /// "spread p10–p90 ±x%" when the spread is wide (above 10 %), else empty.
    pub fn spread_note(&self) -> String {
        let w = (self.hi - self.lo) / self.median;
        let mut s = String::new();
        if w > 0.10 {
            s += &format!("spread p10-p90 {:.0}%", 100.0 * w);
        }
        if self.samples < 15 {
            if !s.is_empty() {
                s += "; ";
            }
            s += &format!("{} samples", self.samples);
        }
        s
    }
}

pub fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Times the contenders against each other. `budget` bounds the time a row may take: when one
/// round of all contenders (one sample each) takes longer than budget/15, fewer samples are
/// taken, never fewer than 5 (3 for rounds above 20 s).
pub fn race(cs: &mut [Contender<'_>]) -> Vec<Stat> {
    let min_batch = Duration::from_micros(env_f64("BENCH_MIN_BATCH_US", 1000.0) as u64);
    let budget = env_f64("BENCH_ROW_BUDGET_S", 60.0);
    let target_samples = env_f64("BENCH_SAMPLES", 15.0) as usize;
    // Warm up and calibrate the repetitions of each contender.
    let mut reps = Vec::with_capacity(cs.len());
    for c in cs.iter_mut() {
        let t0 = Instant::now();
        (c.f)();
        let mut t = t0.elapsed();
        let mut r = 1usize;
        // A second calibration for fast operations: the first call is cold.
        if t < min_batch {
            let est = ((min_batch.as_secs_f64() / t.as_secs_f64().max(1e-8)).ceil() as usize).clamp(1, 1 << 24);
            let t0 = Instant::now();
            for _ in 0..est {
                (c.f)();
            }
            t = t0.elapsed() / est as u32;
            r = ((min_batch.as_secs_f64() / t.as_secs_f64().max(1e-9)).ceil() as usize).clamp(1, 1 << 26);
        }
        reps.push((r, t));
    }
    let round: f64 = reps.iter().map(|(r, t)| *r as f64 * t.as_secs_f64()).sum();
    let mut samples = target_samples;
    if round * target_samples as f64 > budget {
        let floor = if round > 20.0 { 3 } else { 5 };
        samples = ((budget / round).floor() as usize).clamp(floor, target_samples);
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); cs.len()];
    for s in 0..samples {
        // Rotate the order of the contenders from sample to sample.
        let k = cs.len();
        for off in 0..k {
            let ci = (s + off) % k;
            let r = reps[ci].0;
            let f = &mut cs[ci].f;
            let t0 = Instant::now();
            for _ in 0..r {
                f();
            }
            let dt = t0.elapsed();
            times[ci].push(dt.as_nanos() as f64 / r as f64);
        }
    }
    cs.iter()
        .zip(times)
        .map(|(c, mut v)| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
            Stat { name: c.name.clone(), median: q(0.5), lo: q(0.1), hi: q(0.9), samples: v.len() }
        })
        .collect()
}

/// Keeps a value alive so that the optimiser cannot drop the work making it.
#[inline]
pub fn keep<T>(x: T) {
    black_box(x);
}

/// Prints the scoreboard rows of one race: the first contender is ours, every other one a
/// reference. `notes` gives a same-work note per reference (by index, the first entry for ours
/// is prefixed to every row).
pub fn rows(group: &str, operation: &str, params: &str, stats: &[Stat], notes: &[String]) {
    let ours = &stats[0];
    for (i, st) in stats.iter().enumerate().skip(1) {
        let mut n = Vec::new();
        if let Some(s) = notes.first() {
            if !s.is_empty() {
                n.push(s.clone());
            }
        }
        if let Some(s) = notes.get(i) {
            if !s.is_empty() {
                n.push(s.clone());
            }
        }
        let so = ours.spread_note();
        if !so.is_empty() {
            n.push(format!("ours {so}"));
        }
        let sr = st.spread_note();
        if !sr.is_empty() {
            n.push(format!("ref {sr}"));
        }
        let ratio = ours.median / st.median;
        println!(
            "sparse\t{group}\t{operation}\t{params}\t{:.0}\t{}\t{:.0}\t{}\t{}",
            ours.median,
            st.name,
            st.median,
            sig3(ratio),
            n.join("; ")
        );
    }
}

/// A row with no timing of ours (a quality row) or computed outside a race.
pub fn row_raw(group: &str, operation: &str, params: &str, ours: f64, reference: &str, refv: f64, notes: &str) {
    println!(
        "sparse\t{group}\t{operation}\t{params}\t{:.0}\t{reference}\t{:.0}\t{}\t{notes}",
        ours,
        refv,
        sig3(ours / refv)
    );
}

pub fn sig3(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    if x == 0.0 {
        return "0".into();
    }
    let d = 2 - x.abs().log10().floor() as i32;
    if d > 0 { format!("{:.*}", d as usize, x) } else { format!("{:.0}", x) }
}

/// The memory a process may use here, in bytes: MemAvailable of /proc/meminfo (Linux), else
/// sysctl hw.memsize (macOS); BENCH_MEM_GB overrides both. None when it cannot be told.
pub fn mem_available() -> Option<f64> {
    if let Some(g) = std::env::var("BENCH_MEM_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
        return Some(g * 1e9);
    }
    if let Ok(t) = std::fs::read_to_string("/proc/meminfo") {
        let kb = |key: &str| {
            t.lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
                .map(|k| k * 1024.0)
        };
        return kb("MemAvailable:").or_else(|| kb("MemTotal:"));
    }
    let o = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
    String::from_utf8_lossy(&o.stdout).trim().parse::<f64>().ok()
}

/// Whether a problem whose peak memory is about `need_gb` GB fits in this machine's memory (with
/// a quarter more for safety). When it does not, prints a row per operation it would have timed,
/// without figures, saying why, and returns false: the problem skips itself instead of driving the
/// machine out of memory.
pub fn fits(group: &str, ops: &[&str], params: &str, need_gb: f64) -> bool {
    if need_gb <= 0.0 {
        return true;
    }
    let Some(have) = mem_available() else {
        return true;
    };
    if need_gb * 1.25e9 <= have {
        return true;
    }
    let why = format!(
        "skipped: needs about {need_gb} GB of memory at its peak (and a quarter more to spare), this machine has {:.1} GB available",
        have / 1e9
    );
    note(&format!("{params}: {why}"));
    for op in ops {
        println!("sparse\t{group}\t{op}\t{params}\t-\t-\t-\t-\t{why}");
    }
    false
}

/// Progress on stderr.
pub fn note(s: &str) {
    eprintln!("[sparse-bench] {s}");
}

/// Whether the matrix `name` is selected by BENCH_ONLY (a comma-separated list of names; unset:
/// all), so that run.sh can split a group into short timed commands.
pub fn selected(name: &str) -> bool {
    // BENCH_LIST: print the names the group would visit, run nothing (for run.sh).
    if std::env::var("BENCH_LIST").is_ok() {
        println!("{name}");
        return false;
    }
    match std::env::var("BENCH_ONLY") {
        Ok(v) if !v.is_empty() => v.split(',').any(|x| x == name),
        _ => true,
    }
}

/// Whether the part `p` of a group is selected by BENCH_PART (comma-separated; unset: all).
pub fn part(p: &str) -> bool {
    match std::env::var("BENCH_PART") {
        Ok(v) if !v.is_empty() => v.split(',').any(|x| x == p),
        _ => true,
    }
}

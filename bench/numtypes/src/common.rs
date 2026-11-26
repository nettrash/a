//! The timing harness shared by the benchmark binaries.
//!
//! Each side of a comparison is a closure that runs one batch (a pass over a pool of inputs,
//! `items` operations) and writes its results somewhere the optimiser cannot drop them. The
//! closures of one row run round-robin, sample by sample (ours, ref 1, ref 2, …, ours, …), so that
//! load and clock changes hit all of them; each sample is at least `MIN_SAMPLE` long; the
//! reported figure is the median time per item over `SAMPLES` samples, and the spread is the
//! inter-quartile range over the median.

#![allow(dead_code)]

use std::hint::black_box;
use std::io::Write;
use std::time::{Duration, Instant};

pub const SAMPLES: usize = 21;
pub const MIN_SAMPLE: Duration = Duration::from_micros(1500);

/// NTBENCH_QUICK=1: 3 short samples, for a smoke test of the same-work checks (not for figures).
fn quick() -> bool {
    std::env::var_os("NTBENCH_QUICK").is_some()
}

/// A side of a comparison: its name (the reference, or "ours") and the batch it runs.
pub struct Side<'a> {
    pub name: String,
    pub run: Box<dyn FnMut() + 'a>,
    pub note: String,
}

pub fn side<'a>(name: &str, run: impl FnMut() + 'a) -> Side<'a> {
    Side { name: name.to_string(), run: Box::new(run), note: String::new() }
}

pub fn side_n<'a>(name: &str, note: &str, run: impl FnMut() + 'a) -> Side<'a> {
    Side { name: name.to_string(), run: Box::new(run), note: note.to_string() }
}

#[derive(Clone, Copy, Debug)]
pub struct Stat {
    pub median: f64,
    pub spread: f64,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos - pos.floor());
    if i + 1 < sorted.len() { sorted[i] * (1.0 - f) + sorted[i + 1] * f } else { sorted[i] }
}

/// Times the sides round-robin and returns the median ns per item of each.
pub fn measure(sides: &mut [Side<'_>], items: usize) -> Vec<Stat> {
    // Warm up and calibrate: repetitions per sample so that a sample takes MIN_SAMPLE.
    let (samples, min_sample) =
        if quick() { (3, Duration::from_micros(100)) } else { (SAMPLES, MIN_SAMPLE) };
    let mut reps = Vec::with_capacity(sides.len());
    for s in sides.iter_mut() {
        (s.run)();
        let mut r = 1usize;
        loop {
            let t = Instant::now();
            for _ in 0..r {
                (s.run)();
            }
            let e = t.elapsed();
            if e >= min_sample / 4 {
                let per = e.as_secs_f64() / r as f64;
                r = ((min_sample.as_secs_f64() * 1.2) / per).ceil().max(1.0) as usize;
                break;
            }
            r *= 4;
        }
        reps.push(r);
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); sides.len()];
    for _ in 0..samples {
        for (k, s) in sides.iter_mut().enumerate() {
            let t = Instant::now();
            for _ in 0..reps[k] {
                (s.run)();
            }
            let e = t.elapsed();
            times[k].push(e.as_secs_f64() * 1e9 / (reps[k] * items) as f64);
        }
    }
    times
        .into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = quantile(&v, 0.5);
            let iqr = quantile(&v, 0.75) - quantile(&v, 0.25);
            Stat { median: m, spread: iqr / m }
        })
        .collect()
}

/// Formats a number of nanoseconds with 4 significant digits.
pub fn fmt_ns(x: f64) -> String {
    if x >= 1000.0 {
        format!("{:.0}", x)
    } else if x >= 100.0 {
        format!("{:.1}", x)
    } else if x >= 10.0 {
        format!("{:.2}", x)
    } else {
        format!("{:.3}", x)
    }
}

pub fn fmt_ratio(x: f64) -> String {
    // three significant digits
    if x >= 100.0 {
        format!("{:.0}", x)
    } else if x >= 10.0 {
        format!("{:.1}", x)
    } else if x >= 1.0 {
        format!("{:.2}", x)
    } else if x >= 0.1 {
        format!("{:.3}", x)
    } else {
        format!("{:.4}", x)
    }
}

/// A row group: prints one TSV row per reference of an operation (or one row with the
/// reference `none` when nothing else does it).
pub struct Out {
    pub area: &'static str,
    pub filter: Option<String>,
}

impl Out {
    pub fn new(area: &'static str) -> Out {
        Out { area, filter: std::env::var("NTBENCH_FILTER").ok() }
    }

    pub fn wants(&self, group: &str, operation: &str) -> bool {
        match &self.filter {
            None => true,
            Some(f) => format!("{group} {operation}").contains(f.as_str()),
        }
    }

    /// Times `sides` (the first is ours) and prints the rows. `check` is the same-work note
    /// (outputs compared before timing), `notes` the common caveats.
    pub fn row(
        &self,
        group: &str,
        operation: &str,
        params: &str,
        items: usize,
        notes: &str,
        mut sides: Vec<Side<'_>>,
    ) {
        if !self.wants(group, operation) {
            return;
        }
        let st = measure(&mut sides, items);
        let ours = st[0];
        let mut lines = String::new();
        let spread_note = |s: Stat| {
            if s.spread > 0.10 { format!("spread {:.0}%", s.spread * 100.0) } else { String::new() }
        };
        let join = |parts: &[&str]| {
            parts.iter().filter(|p| !p.is_empty()).cloned().collect::<Vec<_>>().join("; ")
        };
        let ours_note =
            if sides[0].note.is_empty() { String::new() } else { format!("ours: {}", sides[0].note) };
        if sides.len() == 1 {
            let n = join(&[notes, &ours_note, &spread_note(ours)]);
            lines.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\tnone\t-\t-\t{}\n",
                self.area,
                group,
                operation,
                params,
                fmt_ns(ours.median),
                n
            ));
        }
        for (k, s) in sides.iter().enumerate().skip(1) {
            let r = st[k];
            let ours_spread = spread_note(ours);
            let ref_spread = spread_note(r).replace("spread", "ref spread");
            let ref_note = if s.note.is_empty() { String::new() } else { format!("{}: {}", s.name, s.note) };
            let n = join(&[notes, &ours_note, &ref_note, &ours_spread, &ref_spread]);
            lines.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                self.area,
                group,
                operation,
                params,
                fmt_ns(ours.median),
                s.name,
                fmt_ns(r.median),
                fmt_ratio(ours.median / r.median),
                n
            ));
        }
        let mut so = std::io::stdout().lock();
        so.write_all(lines.as_bytes()).unwrap();
        so.flush().unwrap();
    }
}

/// xorshift64*, seeded.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }
    pub fn limbs<const N: usize>(&mut self) -> [u64; N] {
        let mut l = [0u64; N];
        for x in l.iter_mut() {
            *x = self.next();
        }
        l
    }
}

/// The pool length for items of `bytes` bytes: about 96 KiB of inputs, within [64, 8192].
pub fn pool_len(bytes: usize) -> usize {
    (96 * 1024 / bytes.max(1)).clamp(64, 8192)
}

/// A side that maps each input of the pool through `f`, collecting the results (dropped at the
/// start of the next batch).
pub fn mk<'a, I: 'a, O: 'a>(
    name: &str,
    note: &str,
    inputs: &'a [I],
    f: impl Fn(&I) -> O + 'a,
) -> Side<'a> {
    let mut o: Vec<O> = Vec::with_capacity(inputs.len());
    side_n(name, note, move || {
        let inp = black_box(inputs);
        o.clear();
        for x in inp {
            o.push(f(x));
        }
        black_box(&mut o);
    })
}

/// Same work: `f` over the inputs gives `want`, exactly.
#[track_caller]
pub fn same<I, B: PartialEq + std::fmt::Debug>(
    what: &str,
    name: &str,
    inputs: &[I],
    want: &[B],
    f: impl Fn(&I) -> B,
) {
    for (i, (x, w)) in inputs.iter().zip(want).enumerate() {
        let y = f(x);
        if &y != w {
            panic!("{what}: {name} differs from ours at item {i}: {y:?} vs {w:?}");
        }
    }
}

/// Keeps a value alive for the optimiser.
#[inline(always)]
pub fn keep<T>(x: T) -> T {
    black_box(x)
}

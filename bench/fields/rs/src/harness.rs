//! Timing and the rows of the scoreboard.
//!
//! Every side of a comparison (ours and each reference) is a closure that runs `iters` operations
//! of the same loop. [`measure`] calibrates each side to at least [`TARGET`] per sample, warms it
//! up, then takes the samples **alternately** — ours, reference 1, reference 2, …, ours, … — so that
//! load and clock changes hit all sides alike, and returns the median time per operation of each.
//!
//! The loops, which every side follows exactly (`x[j]` the inputs taken in turn, `acc` an element
//! started at `x[0]`):
//!
//! | op | loop body |
//! | --- | --- |
//! | mul | `acc = acc · x[j]` (a dependent chain, as in powers and pairings) |
//! | square | `acc = acc²` |
//! | add, sub | `acc = acc ± x[j]` |
//! | inverse, sqrt, pow, frobenius, half_trace, quadratic_root | `acc = acc + f(x[j])` (`sqrt` of `x[j]²`) |
//! | legendre, is_square, trace | `s = s + f(x[j])` (an integer) |
//! | div (binary fields) | `acc = acc / x[j]` |
//!
//! The additions these loops add cost a few percent of an inversion, a root or a power, and the
//! same on every side.

use std::hint::black_box;
use std::time::Instant;

/// The operations of the C shims (csrc/shim.h), with the same numbers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
#[allow(dead_code)]
pub enum Op {
    Mul = 0,
    Sqr = 1,
    Add = 2,
    Sub = 3,
    Inv = 4,
    Sqrt = 5,
    Leg = 6,
    Pow = 7,
    Frob = 8,
    Norm = 9,
    Trace = 10,
    IsSq = 11,
    Div = 12,
    HTrace = 13,
    QRoot = 14,
    PowU = 15,
    // Rust-only operations (the C shims answer UNSUPPORTED)
    Neg = 100,
    Dbl = 101,
    Batch = 102,
    FromBytes = 103,
    ToBytes = 104,
    MulByXi = 110,
    MulBy01 = 111,
    MulBy1 = 112,
    CycSqr = 113,
    CycPow = 114,
    Sparse = 115,
    Conj = 116,
    Frob2 = 117,
    MulByV = 118,
    MulByBase = 119,
    Ghash = 120,
    Sbox = 121,
    Clmul = 122,
    ToNormal = 123,
    ToPoly = 124,
    Irred = 125,
}

impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Op::Mul => "mul",
            Op::Sqr => "square",
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Inv => "inverse",
            Op::Sqrt => "sqrt",
            Op::Leg => "legendre",
            Op::Pow => "pow",
            Op::Frob => "frobenius",
            Op::Norm => "norm",
            Op::Trace => "trace",
            Op::IsSq => "is_square",
            Op::Div => "div",
            Op::HTrace => "half_trace",
            Op::QRoot => "quadratic_root",
            Op::PowU => "pow_u64",
            Op::Neg => "neg",
            Op::Dbl => "double",
            Op::Batch => "batch_invert",
            Op::FromBytes => "from_bytes",
            Op::ToBytes => "to_bytes",
            Op::MulByXi => "mul_by_xi",
            Op::MulBy01 => "mul_by_01",
            Op::MulBy1 => "mul_by_1",
            Op::CycSqr => "cyclotomic_square",
            Op::CycPow => "cyclotomic_pow",
            Op::Sparse => "sparse_mul",
            Op::Conj => "conjugate",
            Op::Frob2 => "frobenius(2)",
            Op::MulByV => "mul_by_v",
            Op::MulByBase => "mul_by_base",
            Op::Ghash => "ghash",
            Op::Sbox => "aes_sbox",
            Op::Clmul => "clmul",
            Op::ToNormal => "to_normal",
            Op::ToPoly => "to_polynomial",
            Op::Irred => "is_irreducible",
        }
    }
}

/// The C shims' answer for an operation they do not have.
pub const UNSUPPORTED: i32 = -1000;

/// The least time of one sample, in nanoseconds.
pub const TARGET: f64 = 2.0e6;

/// The settings of a run.
pub struct Config {
    /// Samples per side (at least 15).
    pub samples: usize,
    /// Samples for operations slower than 20 ms a call.
    pub slow_samples: usize,
    /// Only the checks of the outputs, and each loop run once: no timing (needs no slot).
    pub check_only: bool,
}

pub static mut CONFIG: Config = Config { samples: 21, slow_samples: 15, check_only: false };

fn config() -> &'static Config {
    // SAFETY: written once in main before any measurement, read only afterwards.
    unsafe { &*std::ptr::addr_of!(CONFIG) }
}

/// One side of a comparison.
pub struct Side<'a> {
    #[allow(dead_code)]
    pub name: String,
    pub run: Box<dyn FnMut(u64) + 'a>,
}

impl<'a> Side<'a> {
    pub fn new(name: impl Into<String>, run: impl FnMut(u64) + 'a) -> Side<'a> {
        Side { name: name.into(), run: Box::new(run) }
    }
}

/// The median time per operation of a side, and the spread of its samples.
#[derive(Clone, Copy, Debug)]
pub struct Stat {
    pub median: f64,
    /// (q3 − q1) / median.
    pub iqr: f64,
}

fn time(f: &mut dyn FnMut(u64), iters: u64) -> f64 {
    let t = Instant::now();
    f(iters);
    t.elapsed().as_nanos() as f64
}

/// Calibrates, warms up and measures the sides alternately; the medians in the order given.
pub fn measure(sides: &mut [Side]) -> Vec<Stat> {
    let cfg = config();
    if cfg.check_only {
        for s in sides.iter_mut() {
            (s.run)(1);
        }
        return vec![Stat { median: f64::NAN, iqr: 0.0 }; sides.len()];
    }
    let mut iters = vec![1u64; sides.len()];
    let mut slow = false;
    for (k, s) in sides.iter_mut().enumerate() {
        let mut n = 1u64;
        loop {
            let t = time(&mut *s.run, n);
            if t >= TARGET {
                break;
            }
            let grow = if t < TARGET / 64.0 { 16 } else { 2 };
            n *= grow;
        }
        // one more sample at the final count, as a warm-up
        let t = time(&mut *s.run, n);
        if t / n as f64 > 2.0e7 {
            slow = true;
        }
        iters[k] = n;
    }
    let count = if slow { cfg.slow_samples } else { cfg.samples };
    let mut samples = vec![Vec::with_capacity(count); sides.len()];
    for _ in 0..count {
        for (k, s) in sides.iter_mut().enumerate() {
            let t = time(&mut *s.run, iters[k]);
            samples[k].push(t / iters[k] as f64);
        }
    }
    samples
        .into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
            let median = q(0.5);
            Stat { median, iqr: (q(0.75) - q(0.25)) / median }
        })
        .collect()
}

/// A number to about four significant digits.
pub fn ns(x: f64) -> String {
    if x >= 1000.0 {
        format!("{x:.0}")
    } else if x >= 100.0 {
        format!("{x:.1}")
    } else if x >= 10.0 {
        format!("{x:.2}")
    } else {
        format!("{x:.3}")
    }
}

/// A ratio to three significant digits.
pub fn sig3(r: f64) -> String {
    if !r.is_finite() || r <= 0.0 {
        return "-".into();
    }
    let digits = 2 - r.log10().floor() as i32;
    format!("{r:.*}", digits.max(0) as usize)
}

/// Prints the rows of one measured operation: one row per reference, with ours beside it.
/// `sides[0]` is ours. `notes[k]` are the notes of side `k`'s row.
pub fn report(group: &str, op: &str, params: &str, names: &[String], stats: &[Stat], notes: &[String]) {
    let ours = stats[0];
    let spread = |s: Stat| if s.iqr > 0.05 { format!("iqr {:.0}%", s.iqr * 100.0) } else { String::new() };
    if names.len() == 1 {
        let mut n = vec![notes[0].clone(), "no reference in this harness".into()];
        n.push(spread(ours));
        row(group, op, params, ours.median, "none", f64::NAN, &join(&n));
        return;
    }
    for k in 1..names.len() {
        let mut n = vec![notes[k].clone()];
        let (so, sr) = (spread(ours), spread(stats[k]));
        if !so.is_empty() {
            n.push(format!("ours {so}"));
        }
        if !sr.is_empty() {
            n.push(format!("ref {sr}"));
        }
        if !notes[0].is_empty() {
            n.push(notes[0].clone());
        }
        row(group, op, params, ours.median, &names[k], stats[k].median, &join(&n));
    }
}

fn join(v: &[String]) -> String {
    v.iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join("; ")
}

/// Prints one row of the scoreboard.
pub fn row(group: &str, op: &str, params: &str, ours: f64, reference: &str, refns: f64, notes: &str) {
    let (r, rn) = if refns.is_nan() { ("-".to_string(), "-".to_string()) } else { (sig3(ours / refns), ns(refns)) };
    println!("fields\t{group}\t{op}\t{params}\t{}\t{reference}\t{rn}\t{r}\t{notes}", ns(ours));
}

/// A check failed: say so on stderr and in the row.
pub fn mismatch(what: &str) -> String {
    eprintln!("MISMATCH: {what}");
    format!("MISMATCH ({what})")
}

/// splitmix64, for the inputs.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Keeps a value alive past the loop.
#[inline(always)]
pub fn keep<T>(x: T) -> T {
    black_box(x)
}

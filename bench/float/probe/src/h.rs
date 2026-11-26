//! The timing harness: contenders (ours and the references) alternate sample by sample in one
//! process, each sample a batch of at least 1 ms over a pool of fresh inputs; medians per call.
use std::time::Instant;

pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// An exact value `±M·2^exp`, `M` of exactly `bits` bits (top bit set), limbs little-endian.
#[derive(Clone, Debug)]
pub struct Raw {
    pub neg: bool,
    pub limbs: Vec<u64>,
    pub bits: u64,
    pub exp: i64,
}

impl Raw {
    /// A random value of `bits` significant bits whose magnitude lies in `[2^(k-1), 2^k)`.
    pub fn random(rng: &mut Rng, bits: u64, k: i64, neg: bool) -> Raw {
        let n = bits.div_ceil(64) as usize;
        let mut limbs: Vec<u64> = (0..n).map(|_| rng.next()).collect();
        let top = bits - 64 * (n as u64 - 1); // bits in the top limb, 1..=64
        if top < 64 {
            limbs[n - 1] &= (1u64 << top) - 1;
        }
        limbs[n - 1] |= 1u64 << (top - 1);
        // value M·2^exp with 2^(bits-1) ≤ M < 2^bits; want 2^(k-1) ≤ |x| < 2^k.
        Raw { neg, limbs, bits, exp: k - bits as i64 }
    }
    /// Random with the exponent `k` drawn from `kmin..=kmax` and a random sign when `signed`.
    pub fn rand_range(rng: &mut Rng, bits: u64, kmin: i64, kmax: i64, signed: bool) -> Raw {
        let k = kmin + rng.below((kmax - kmin + 1) as u64) as i64;
        let neg = signed && rng.next() & 1 == 1;
        Raw::random(rng, bits, k, neg)
    }
    pub fn from_f64(d: f64) -> Raw {
        assert!(d.is_finite() && d != 0.0);
        let b = d.to_bits();
        let e = ((b >> 52) & 0x7ff) as i64;
        let m = if e == 0 { (b & ((1 << 52) - 1)) << 1 } else { (b & ((1 << 52) - 1)) | (1 << 52) };
        let ee = if e == 0 { -1075 } else { e - 1075 };
        let bits = 64 - m.leading_zeros() as u64;
        Raw { neg: d < 0.0, limbs: vec![m], bits, exp: ee }
    }
    /// The exponent `e` of `0.1…·2^e`.
    pub fn e(&self) -> i64 {
        self.exp + self.bits as i64
    }
}

/// A canonical value for comparing outputs across libraries: an odd mantissa and its exponent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Canon {
    Nan,
    Inf(bool),
    Zero(bool),
    Fin(bool, Vec<u64>, i64),
    Str(String),
    Many(Vec<Canon>),
    Skip,
}

impl Canon {
    /// `±M·2^e` with `M` given by little-endian limbs (any length, may have high zeros).
    pub fn fin(neg: bool, mut limbs: Vec<u64>, mut e: i64) -> Canon {
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        if limbs.is_empty() {
            return Canon::Zero(neg);
        }
        let lz = limbs.iter().take_while(|&&w| w == 0).count();
        limbs.drain(..lz);
        e += 64 * lz as i64;
        let tz = limbs[0].trailing_zeros();
        if tz > 0 {
            for i in 0..limbs.len() {
                let hi = if i + 1 < limbs.len() { limbs[i + 1] << (64 - tz) } else { 0 };
                limbs[i] = (limbs[i] >> tz) | hi;
            }
            e += tz as i64;
        }
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        Canon::Fin(neg, limbs, e)
    }
    pub fn short(&self) -> String {
        match self {
            Canon::Fin(n, l, e) => {
                let bits = 64 * l.len() as i64 - l.last().unwrap().leading_zeros() as i64;
                format!("{}0x{:x}..({} bits)p{}", if *n { "-" } else { "" }, l[l.len() - 1], bits, e)
            }
            Canon::Str(s) => s.chars().take(40).collect(),
            Canon::Many(v) => v.iter().map(|c| c.short()).collect::<Vec<_>>().join(" , "),
            o => format!("{o:?}"),
        }
    }
}

pub enum Mode {
    Time(usize),
    Check(usize),
}
pub enum Out {
    T(f64),
    C(Canon),
}
pub type Kern<'a> = Box<dyn FnMut(Mode) -> Out + 'a>;

/// A kernel over a pool of `l` (a power of two) inputs: `|k| op` with `k` the pool index;
/// `|v| canon` turns the op's value (bound to `v`) into a `Canon` for the same-work check.
#[macro_export]
macro_rules! kern {
    ($l:expr, |$k:ident| $op:expr, |$v:ident| $res:expr) => {{
        let mask: usize = ($l) - 1;
        Box::new(move |m: $crate::h::Mode| -> $crate::h::Out {
            match m {
                $crate::h::Mode::Time(n) => {
                    let t = std::time::Instant::now();
                    for i in 0..n {
                        #[allow(unused_variables)]
                        let $k = i & mask;
                        std::hint::black_box($op);
                    }
                    $crate::h::Out::T(t.elapsed().as_secs_f64())
                }
                $crate::h::Mode::Check(i) => {
                    #[allow(unused_variables)]
                    let $k = i & mask;
                    #[allow(unused_variables)]
                    let $v = $op;
                    $crate::h::Out::C($res)
                }
            }
        }) as $crate::h::Kern
    }};
}

pub struct Cont<'a> {
    pub name: String,
    /// `Some(tag)` for one of ours (tag "" is the default), `None` for a reference.
    pub ours: Option<String>,
    /// For a reference: the tag of ours it is compared with.
    pub against: String,
    pub note: String,
    /// Whether the outputs must agree with ours (false for Arb's balls and other "does less").
    pub exact: bool,
    pub k: Kern<'a>,
}

impl<'a> Cont<'a> {
    pub fn ours(k: Kern<'a>) -> Cont<'a> {
        Cont { name: "a".into(), ours: Some(String::new()), against: String::new(), note: String::new(), exact: true, k }
    }
    pub fn ours_tag(tag: &str, k: Kern<'a>) -> Cont<'a> {
        Cont { name: format!("a@{tag}"), ours: Some(tag.into()), against: String::new(), note: String::new(), exact: true, k }
    }
    pub fn rf(name: &str, k: Kern<'a>) -> Cont<'a> {
        Cont { name: name.into(), ours: None, against: String::new(), note: String::new(), exact: true, k }
    }
    pub fn vs(mut self, tag: &str) -> Self {
        self.against = tag.into();
        self
    }
    pub fn note(mut self, n: &str) -> Self {
        if !self.note.is_empty() {
            self.note.push_str("; ");
        }
        self.note.push_str(n);
        self
    }
    pub fn loose(mut self) -> Self {
        self.exact = false;
        self
    }
}

pub struct Cfg {
    pub samples: usize,
    pub target: f64,
    pub max_call: f64,
    pub row_budget: f64,
    pub filter: Vec<String>,
    pub skip: Vec<String>,
    pub quick: bool,
    pub check_only: bool,
}

pub static mut CFG: Cfg =
    Cfg { samples: 15, target: 1.0e-3, max_call: 3.0, row_budget: 25.0, filter: Vec::new(), skip: Vec::new(), quick: false, check_only: false };

#[allow(static_mut_refs)]
pub fn cfg() -> &'static Cfg {
    unsafe { &CFG }
}

pub fn fmt_ns(x: f64) -> String {
    if x >= 100.0 {
        format!("{:.0}", x)
    } else if x >= 10.0 {
        format!("{:.1}", x)
    } else {
        format!("{:.2}", x)
    }
}

pub fn sig3(x: f64) -> String {
    if !x.is_finite() {
        return "nan".into();
    }
    let d = if x == 0.0 { 0 } else { x.abs().log10().floor() as i32 };
    let dec = (2 - d).max(0) as usize;
    format!("{:.*}", dec, x)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn quart(v: &[f64], q: f64) -> f64 {
    // v sorted
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

/// Per-call times of the references at the last size measured, to leave out a reference whose
/// call at a larger size would take more than `max_call` (predicted as `t·(size/size0)^1.7`).
static PERF: std::sync::Mutex<Vec<(String, f64, f64)>> = std::sync::Mutex::new(Vec::new());

/// The family of a row (its params with the last number taken out) and its size (that number),
/// or `None` when the params hold no number: "prec=53 arg=m*2^10000" is of the family
/// "prec=53 arg=m*2^#" and the size 10000.
fn family(group: &str, op: &str, params: &str, name: &str) -> Option<(String, f64)> {
    let b = params.as_bytes();
    let end = b.iter().rposition(|c| c.is_ascii_digit())? + 1;
    let start = b[..end].iter().rposition(|c| !c.is_ascii_digit()).map_or(0, |i| i + 1);
    let size: f64 = params[start..end].parse().ok()?;
    Some((format!("{group}/{op}/{}#{}/{name}", &params[..start], &params[end..]), size))
}

pub fn predicted(group: &str, op: &str, params: &str, name: &str) -> Option<(f64, f64, f64)> {
    let (k, size) = family(group, op, params, name)?;
    let v = PERF.lock().unwrap();
    // Only sizes of one family that grow by a factor (not labels such as "d=0", "d=1").
    let (_, s0, t0) = v.iter().rev().find(|(kk, s0, _)| *kk == k && *s0 >= 8.0 && *s0 < size)?;
    Some((t0 * (size / s0).powf(1.7), *s0, *t0))
}

pub fn record(group: &str, op: &str, params: &str, name: &str, t: f64) {
    if let Some((k, size)) = family(group, op, params, name) {
        PERF.lock().unwrap().push((k, size, t));
    }
}

/// Whether the row passes the filter (a substring of "group/operation/params").
pub fn wanted(group: &str, op: &str, params: &str) -> bool {
    let s = format!("{group}/{op}/{params}");
    cfg().filter.iter().all(|f| s.contains(f.as_str())) && !cfg().skip.iter().any(|f| s.contains(f.as_str()))
}

/// Checks, measures and prints one row per reference.
pub fn row(group: &str, op: &str, params: &str, ncheck: usize, mut conts: Vec<Cont>) {
    if !wanted(group, op, params) {
        return;
    }
    let cfg = cfg();
    let nc = conts.len();
    // 1. Same work: outputs on the first `ncheck` inputs.
    let mut canon: Vec<Vec<Canon>> = Vec::with_capacity(nc);
    let mut skip = vec![false; nc];
    let mut first_call = vec![0.0f64; nc];
    let mut pred_note: Vec<String> = vec![String::new(); nc];
    for (ci, c) in conts.iter_mut().enumerate() {
        let mut v = Vec::new();
        if c.ours.is_none() {
            if let Some((t, s0, t0)) = predicted(group, op, params, &c.name) {
                if t > cfg.max_call {
                    skip[ci] = true;
                    first_call[ci] = t;
                    pred_note[ci] = format!("not run: one call predicted {t:.0} s from {t0:.2} s at size {s0}");
                    canon.push(vec![Canon::Skip; ncheck]);
                    continue;
                }
            }
        }
        let mut limit = ncheck.max(1);
        let mut i = 0;
        while i < limit {
            let t0 = Instant::now();
            let out = (c.k)(Mode::Check(i));
            let dt = t0.elapsed().as_secs_f64();
            if i == 0 {
                first_call[ci] = dt;
            }
            if i < ncheck {
                match out {
                    Out::C(x) => v.push(x),
                    Out::T(_) => unreachable!(),
                }
            }
            if i == 0 && dt > cfg.max_call && c.ours.is_none() {
                skip[ci] = true;
                break;
            }
            if i == 0 && dt > 0.25 {
                limit = limit.min(2);
            }
            i += 1;
        }
        while v.len() < ncheck {
            v.push(Canon::Skip);
        }
        canon.push(v);
    }
    let ours_idx = |tag: &str, conts: &[Cont]| conts.iter().position(|c| c.ours.as_deref() == Some(tag));
    let mut check_note: Vec<String> = vec![String::new(); nc];
    for i in 0..nc {
        if conts[i].ours.is_some() {
            continue;
        }
        let Some(o) = ours_idx(&conts[i].against, &conts) else { continue };
        let mut bad = 0;
        let mut first = String::new();
        let mut skipped = 0;
        for j in 0..ncheck {
            if canon[i][j] == Canon::Skip || canon[o][j] == Canon::Skip {
                skipped += 1;
                continue;
            }
            if canon[i][j] != canon[o][j] {
                if bad == 0 {
                    first = match (&canon[o][j], &canon[i][j]) {
                        (Canon::Many(x), Canon::Many(y)) if x.len() == y.len() => {
                            let d = (0..x.len()).find(|&t| x[t] != y[t]).unwrap_or(0);
                            format!("part {d} of {}: ours {} vs {}", x.len(), x[d].short(), y[d].short())
                        }
                        _ => format!("ours {} vs {}", canon[o][j].short(), canon[i][j].short()),
                    };
                }
                bad += 1;
            }
        }
        let checked = ncheck - skipped;
        if checked == 0 {
            check_note[i] = "outputs not compared".into();
        } else if bad == 0 {
            check_note[i] = format!("outputs equal on {checked}");
        } else if conts[i].exact {
            check_note[i] = format!("MISMATCH {bad}/{checked} ({first})");
            eprintln!("MISMATCH {group}/{op}/{params} {}: {bad}/{checked}: {first}", conts[i].name);
        } else {
            check_note[i] = format!("differs on {bad}/{checked} (does not round the same)");
        }
    }
    if cfg.check_only {
        for i in 0..nc {
            if conts[i].ours.is_none() {
                println!("check\t{group}\t{op}\t{params}\t{}\t{}\t{:.3e}s", conts[i].name, check_note[i], first_call[i]);
            }
        }
        return;
    }
    // 2. Calibrate (also warms up): a batch of at least `target` seconds.
    let mut n = vec![1usize; nc];
    let mut per_call = first_call.clone();
    for i in 0..nc {
        if skip[i] {
            continue;
        }
        loop {
            let t = match (conts[i].k)(Mode::Time(n[i])) {
                Out::T(t) => t,
                _ => unreachable!(),
            };
            per_call[i] = t / n[i] as f64;
            if t > cfg.max_call && n[i] == 1 && conts[i].ours.is_none() {
                skip[i] = true;
                break;
            }
            if t >= cfg.target {
                break;
            }
            let f = if t <= 0.0 { 16.0 } else { (cfg.target * 1.2 / t).clamp(1.5, 64.0) };
            n[i] = ((n[i] as f64) * f).ceil() as usize;
        }
        if !skip[i] {
            let _ = (conts[i].k)(Mode::Time(n[i]));
        }
    }
    // 3. Samples, round-robin from a rotating start, so that load hits every contender.
    let round: f64 = (0..nc).filter(|&i| !skip[i]).map(|i| per_call[i] * n[i] as f64).sum();
    let mut samples = cfg.samples;
    if cfg.quick {
        samples = 3;
    } else if round * samples as f64 > cfg.row_budget {
        samples = ((cfg.row_budget / round).floor() as usize).max(5).min(samples);
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); nc];
    for s in 0..samples {
        for j in 0..nc {
            let i = (s + j) % nc;
            if skip[i] {
                continue;
            }
            if let Out::T(t) = (conts[i].k)(Mode::Time(n[i])) {
                times[i].push(t / n[i] as f64 * 1e9);
            }
        }
    }
    let mut med = vec![f64::NAN; nc];
    let mut spread = vec![0.0; nc];
    for i in 0..nc {
        if skip[i] {
            continue;
        }
        med[i] = median(&mut times[i]);
        let q1 = quart(&times[i], 0.25);
        let q3 = quart(&times[i], 0.75);
        spread[i] = (q3 - q1) / med[i];
    }
    for i in 0..nc {
        if conts[i].ours.is_none() && pred_note[i].is_empty() {
            let t = if skip[i] { first_call[i] } else { med[i] * 1e-9 };
            record(group, op, params, &conts[i].name, t);
        }
    }
    // 4. Rows.
    for i in 0..nc {
        if conts[i].ours.is_some() {
            continue;
        }
        let Some(o) = ours_idx(&conts[i].against, &conts) else { continue };
        let mut notes: Vec<String> = Vec::new();
        if !conts[i].note.is_empty() {
            notes.push(conts[i].note.clone());
        }
        if !conts[o].note.is_empty() {
            notes.push(format!("ours: {}", conts[o].note));
        }
        if !conts[i].against.is_empty() {
            notes.push(format!("ours at {}", conts[i].against));
        }
        if !check_note[i].is_empty() {
            notes.push(check_note[i].clone());
        }
        if samples != cfg.samples {
            notes.push(format!("{samples} samples"));
        }
        if spread[o] > 0.10 {
            notes.push(format!("ours IQR {:.0}%", spread[o] * 100.0));
        }
        if skip[i] {
            if pred_note[i].is_empty() {
                notes.push(format!("not timed: one call > {:.0} s ({:.1} s)", cfg.max_call, per_call[i]));
            } else {
                notes.push(pred_note[i].clone());
            }
            println!(
                "float\t{group}\t{op}\t{params}\t{}\t{}\t\t\t{}",
                fmt_ns(med[o]),
                conts[i].name,
                notes.join("; ")
            );
            continue;
        }
        if spread[i] > 0.10 {
            notes.push(format!("ref IQR {:.0}%", spread[i] * 100.0));
        }
        println!(
            "float\t{group}\t{op}\t{params}\t{}\t{}\t{}\t{}\t{}",
            fmt_ns(med[o]),
            conts[i].name,
            fmt_ns(med[i]),
            sig3(med[o] / med[i]),
            notes.join("; ")
        );
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

/// Times one call (for the constants' first call in a fresh process).
pub fn once<F: FnOnce()>(f: F) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e9
}

/// Pool length for values of `p` bits and `per` values per item: a power of two.
pub fn pool(p: u64, per: usize) -> usize {
    let bytes = (p / 8 + 32) as usize * per;
    let mut l = 64usize;
    while l > 2 && l * bytes > 4 << 20 {
        l /= 2;
    }
    l
}

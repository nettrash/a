//! Scoreboard harness for `a::math::special` (the v0.2 speed pass).
//!
//! A *case* is one operation over one argument region: a generator of random argument records
//! (`[T; 4]`, the unused slots spare), our batch function and the references' batch functions,
//! all computing the same values. The harness draws a pool of 16,384 records (deterministic
//! seed from the case's name), checks every reference against ours on the pool (ulps relative
//! to our value: max, 99th percentile, median; NaN/inf mismatches), then times them in a
//! throughput loop (one call per record, results stored), alternating the implementations
//! sample by sample (rotated order), each sample at least 1 ms, 21 samples after 2 warm-up
//! rounds, and reports the medians in ns per call.

use std::hint::black_box;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

pub mod cases_bessel;
pub mod cases_elliptic;
pub mod cases_erf;
pub mod cases_f32;
pub mod cases_gamma;
pub mod cases_ortho;
pub mod cases_zeta;
pub mod ffi;

pub trait Elem: Copy + Default + Send + Sync + 'static {
    fn to64(self) -> f64;
    fn from64(x: f64) -> Self;
    const EPS: f64;
}
impl Elem for f64 {
    fn to64(self) -> f64 {
        self
    }
    fn from64(x: f64) -> f64 {
        x
    }
    const EPS: f64 = f64::EPSILON;
}
impl Elem for f32 {
    fn to64(self) -> f64 {
        self as f64
    }
    fn from64(x: f64) -> f32 {
        x as f32
    }
    const EPS: f64 = f32::EPSILON as f64;
}

pub type Rec<T> = [T; 4];
pub type BatchFn<T> = Box<dyn Fn(&[Rec<T>], &mut [Rec<T>]) + Send + Sync>;

/// One output value per record.
pub fn b1<T: Elem, F: Fn(&Rec<T>) -> T + Send + Sync + 'static>(f: F) -> BatchFn<T> {
    Box::new(move |i: &[Rec<T>], o: &mut [Rec<T>]| {
        for (x, y) in i.iter().zip(o.iter_mut()) {
            y[0] = f(x);
        }
    })
}

/// Several output values per record.
pub fn bn<T: Elem, F: Fn(&Rec<T>, &mut Rec<T>) + Send + Sync + 'static>(f: F) -> BatchFn<T> {
    Box::new(move |i: &[Rec<T>], o: &mut [Rec<T>]| {
        for (x, y) in i.iter().zip(o.iter_mut()) {
            f(x, y);
        }
    })
}

/// A C batch function `void f(const double (*)[4], double (*)[4], size_t)`.
#[macro_export]
macro_rules! cf64 {
    ($f:ident) => {{
        unsafe extern "C" {
            fn $f(i: *const [f64; 4], o: *mut [f64; 4], n: usize);
        }
        let b: $crate::BatchFn<f64> = Box::new(|i: &[[f64; 4]], o: &mut [[f64; 4]]| unsafe {
            $f(i.as_ptr(), o.as_mut_ptr(), i.len())
        });
        b
    }};
}

/// A C batch function `void f(const float (*)[4], float (*)[4], size_t)`.
#[macro_export]
macro_rules! cf32 {
    ($f:ident) => {{
        unsafe extern "C" {
            fn $f(i: *const [f32; 4], o: *mut [f32; 4], n: usize);
        }
        let b: $crate::BatchFn<f32> = Box::new(|i: &[[f32; 4]], o: &mut [[f32; 4]]| unsafe {
            $f(i.as_ptr(), o.as_mut_ptr(), i.len())
        });
        b
    }};
}

/// The system libm's name in the rows.
#[cfg(target_os = "macos")]
pub const SYS: &str = "apple-libm";
#[cfg(not(target_os = "macos"))]
pub const SYS: &str = "glibc-libm";

/// `sys!(c, f)`: the system libm's C batch `f`; `bo!` Boost's, `gs!` GSL's (when built).
#[macro_export]
macro_rules! sys {
    ($c:ident, $f:ident) => {
        $c.r($crate::SYS, "", $crate::cf64!($f));
    };
    ($c:ident, $f:ident, $n:expr) => {
        $c.r($crate::SYS, $n, $crate::cf64!($f));
    };
}
#[macro_export]
macro_rules! bo {
    ($c:ident, $f:ident) => {
        #[cfg(has_boost)]
        {
            $c.r("boost", "", $crate::cf64!($f));
        }
    };
    ($c:ident, $f:ident, $n:expr) => {
        #[cfg(has_boost)]
        {
            $c.r("boost", $n, $crate::cf64!($f));
        }
    };
}
#[macro_export]
macro_rules! gs {
    ($c:ident, $f:ident) => {
        #[cfg(has_gsl)]
        {
            $c.r("gsl", "", $crate::cf64!($f));
        }
    };
    ($c:ident, $f:ident, $n:expr) => {
        #[cfg(has_gsl)]
        {
            $c.r("gsl", $n, $crate::cf64!($f));
        }
    };
}

/// How a reference's outputs are compared with ours.
#[derive(Clone, Copy, PartialEq)]
pub enum Cmp {
    /// Component by component, in ulps of our value.
    Rel,
    /// As `Rel`, the second component (an imaginary part, an argument) first moved by the
    /// multiple of 2π nearest to ours.
    ImMod2Pi,
}

/// How the differences from ours are measured.
#[derive(Clone, Copy, PartialEq)]
pub enum Metric {
    /// ulps of our value, component by component.
    Rel,
    /// ulps of max(|our value|, M), M the median |value| over the checked records (the page's
    /// convention for oscillatory functions, whose zeros make relative errors meaningless).
    Scaled,
    /// |w - w_ref| / (eps |w|) of the complex result (components 0, 1).
    Complex,
}

impl Metric {
    pub fn code(self) -> &'static str {
        match self {
            Metric::Rel => "rel",
            Metric::Scaled => "scaled",
            Metric::Complex => "complex",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Metric::Rel => "diff vs ours in ulps",
            Metric::Scaled => "diff vs ours in ulps of max(|v|, median |v|)",
            Metric::Complex => "diff vs ours in eps of |w|",
        }
    }
}

pub struct Imp<T> {
    pub name: String,
    pub note: String,
    pub cmp: Cmp,
    pub f: BatchFn<T>,
}

pub struct Case<T> {
    pub group: &'static str,
    pub op: String,
    pub params: String,
    /// The number of output components compared.
    pub nout: usize,
    pub generator: Box<dyn Fn(&mut Rng) -> Rec<T> + Send + Sync>,
    pub ours: BatchFn<T>,
    pub refs: Vec<Imp<T>>,
    /// The SciPy call for the Python driver: `ufunc:args[:outs]` (see scipy_bench.py).
    pub scipy: Option<String>,
    pub note: String,
    pub metric: Metric,
}

impl<T: Elem> Case<T> {
    pub fn new(
        group: &'static str,
        op: &str,
        params: &str,
        nout: usize,
        generator: impl Fn(&mut Rng) -> Rec<T> + Send + Sync + 'static,
        ours: BatchFn<T>,
    ) -> Self {
        Case {
            group,
            op: op.into(),
            params: params.into(),
            nout,
            generator: Box::new(generator),
            ours,
            refs: Vec::new(),
            scipy: None,
            note: String::new(),
            metric: Metric::Rel,
        }
    }
    pub fn r(&mut self, name: &str, note: &str, f: BatchFn<T>) {
        self.refs.push(Imp { name: name.into(), note: note.into(), cmp: Cmp::Rel, f });
    }
    pub fn rc(&mut self, name: &str, note: &str, cmp: Cmp, f: BatchFn<T>) {
        self.refs.push(Imp { name: name.into(), note: note.into(), cmp, f });
    }
    pub fn sp(&mut self, spec: &str) {
        self.scipy = Some(spec.into());
    }
    /// Oscillatory: measure differences in ulps of max(|v|, median |v|).
    pub fn osc(&mut self) {
        self.metric = Metric::Scaled;
    }
    /// A complex result in components 0 and 1.
    pub fn cplx(&mut self) {
        self.metric = Metric::Complex;
    }
    pub fn note(&mut self, n: &str) {
        self.note = n.into();
    }
    pub fn key(&self) -> String {
        format!("{}|{}|{}", self.group, self.op, self.params)
    }
}

/// SplitMix64.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn uni(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
    /// Log-uniform in [a, b], 0 < a < b.
    pub fn logu(&mut self, a: f64, b: f64) -> f64 {
        (a.ln() + (b.ln() - a.ln()) * self.f()).exp()
    }
    /// Uniform integer in [a, b].
    pub fn int(&mut self, a: i64, b: i64) -> i64 {
        a + (self.next() % ((b - a + 1) as u64)) as i64
    }
}

pub fn seed_of(key: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

pub const POOL: usize = 16384;

pub fn pool<T: Elem>(c: &Case<T>, n: usize) -> Vec<Rec<T>> {
    let mut rng = Rng(seed_of(&c.key()));
    (0..n).map(|_| (c.generator)(&mut rng)).collect()
}

pub struct Opts {
    pub samples: usize,
    pub warm: usize,
    /// Least duration of one sample, ns.
    pub target_ns: f64,
    /// Most duration of one sample of the slowest implementation, ns (where the fastest's
    /// target would make it longer, the fastest repeats a shorter stretch of the pool).
    pub max_sample_ns: f64,
    pub ours_only: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { samples: 21, warm: 2, target_ns: 1.0e6, max_sample_ns: 30.0e6, ours_only: false }
    }
}

#[derive(Clone, Default)]
pub struct CmpStats {
    pub n: usize,
    pub max: f64,
    pub p99: f64,
    pub med: f64,
    pub mism: usize,
}

pub struct ImpResult {
    pub name: String,
    pub note: String,
    pub failed: Option<String>,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
    pub cmp: Option<CmpStats>,
}

pub struct CaseResult {
    pub ours: ImpResult,
    pub refs: Vec<ImpResult>,
}

fn call<T: Elem>(f: &BatchFn<T>, inp: &[Rec<T>], out: &mut [Rec<T>]) -> bool {
    catch_unwind(AssertUnwindSafe(|| f(inp, out))).is_ok()
}

/// Time `k` passes over the `m` records from `off` (cyclic), ns in all; None on a panic.
fn timed<T: Elem>(
    f: &BatchFn<T>,
    pool: &[Rec<T>],
    out: &mut [Rec<T>],
    off: usize,
    m: usize,
    k: usize,
) -> Option<f64> {
    let p = pool.len();
    let r = catch_unwind(AssertUnwindSafe(|| {
        let t0 = Instant::now();
        for _ in 0..k {
            let mut left = m;
            let mut o = off;
            while left > 0 {
                let len = left.min(p - o);
                f(&pool[o..o + len], &mut out[o..o + len]);
                left -= len;
                o += len;
                if o == p {
                    o = 0;
                }
            }
        }
        t0.elapsed().as_nanos() as f64
    }));
    black_box(&mut *out);
    r.ok()
}

fn ulps(a: f64, b: f64, eps: f64) -> Option<f64> {
    // None: a NaN/inf mismatch
    if a.is_finite() && b.is_finite() {
        if a == b {
            return Some(0.0);
        }
        let s = a.abs().max(f64::MIN_POSITIVE);
        Some((a - b).abs() / (s * eps))
    } else if (a.is_nan() && b.is_nan()) || a == b {
        Some(0.0)
    } else {
        None
    }
}

fn compare<T: Elem>(ours: &[Rec<T>], r: &[Rec<T>], nout: usize, cmp: Cmp, metric: Metric) -> CmpStats {
    let mut d = Vec::with_capacity(ours.len());
    let mut mism = 0;
    // the floors of the scaled metric: median |ours| per component
    let mut floor = [f64::MIN_POSITIVE; 4];
    if metric == Metric::Scaled {
        for (j, f) in floor.iter_mut().enumerate().take(nout) {
            let mut v: Vec<f64> =
                ours.iter().map(|a| a[j].to64().abs()).filter(|x| x.is_finite()).collect();
            if !v.is_empty() {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                *f = v[v.len() / 2].max(f64::MIN_POSITIVE);
            }
        }
    }
    for (a, b) in ours.iter().zip(r) {
        let mut worst: f64 = 0.0;
        let mut bad = false;
        let mut y1 = if nout > 1 { b[1].to64() } else { 0.0 };
        if cmp == Cmp::ImMod2Pi && nout > 1 {
            let x = a[1].to64();
            if x.is_finite() && y1.is_finite() {
                let tau = std::f64::consts::TAU;
                y1 += ((x - y1) / tau).round() * tau;
            }
        }
        if metric == Metric::Complex {
            let (xr, xi, yr, yi) = (a[0].to64(), a[1].to64(), b[0].to64(), y1);
            let fin = xr.is_finite() && xi.is_finite() && yr.is_finite() && yi.is_finite();
            if fin {
                let m = xr.hypot(xi).max(f64::MIN_POSITIVE);
                let e = (xr - yr).hypot(xi - yi);
                if e > 0.0 {
                    worst = e / (m * T::EPS);
                }
            } else if !(ulps(xr, yr, T::EPS).is_some() && ulps(xi, yi, T::EPS).is_some()) {
                bad = true;
            }
        } else {
            for j in 0..nout {
                let x = a[j].to64();
                let y = if j == 1 { y1 } else { b[j].to64() };
                let u = if metric == Metric::Scaled && x.is_finite() && y.is_finite() {
                    Some((x - y).abs() / (x.abs().max(floor[j]) * T::EPS))
                } else {
                    ulps(x, y, T::EPS)
                };
                match u {
                    Some(u) => worst = worst.max(u),
                    None => bad = true,
                }
            }
        }
        if bad {
            mism += 1;
        } else {
            d.push(worst);
        }
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| if d.is_empty() { f64::NAN } else { d[((d.len() - 1) as f64 * p) as usize] };
    CmpStats { n: ours.len(), max: q(1.0), p99: q(0.99), med: q(0.5), mism }
}

fn quant(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let x = (v.len() - 1) as f64 * p;
    let i = x.floor() as usize;
    let j = (i + 1).min(v.len() - 1);
    v[i] + (v[j] - v[i]) * (x - i as f64)
}

pub fn run_case<T: Elem>(c: &Case<T>, o: &Opts) -> CaseResult {
    let pool = pool(c, POOL);
    let mut out = vec![[T::default(); 4]; POOL];

    struct St<'a, T> {
        name: String,
        note: String,
        cmp: Cmp,
        f: &'a BatchFn<T>,
        failed: Option<String>,
        t: f64,
        samples: Vec<f64>,
        k: usize,
        stats: Option<CmpStats>,
    }
    let mut st: Vec<St<T>> = vec![St {
        name: "ours".into(),
        note: String::new(),
        cmp: Cmp::Rel,
        f: &c.ours,
        failed: None,
        t: 0.0,
        samples: vec![],
        k: 1,
        stats: None,
    }];
    if !o.ours_only {
        for r in &c.refs {
            st.push(St {
                name: r.name.clone(),
                note: r.note.clone(),
                cmp: r.cmp,
                f: &r.f,
                failed: None,
                t: 0.0,
                samples: vec![],
                k: 1,
                stats: None,
            });
        }
    }

    // Calibration: ns per call on a few records, then on more if quick.
    for s in st.iter_mut() {
        let mut n = 1;
        loop {
            let t0 = Instant::now();
            if !call(s.f, &pool[..n], &mut out[..n]) {
                s.failed = Some("panics".into());
                break;
            }
            let dt = t0.elapsed().as_nanos() as f64 / n as f64;
            s.t = dt;
            if dt > 20.0e6 {
                s.failed = Some(format!("too slow ({:.0} ms a call)", dt / 1e6));
                break;
            }
            if n >= 1024 || dt * n as f64 * 4.0 > 20.0e6 {
                break;
            }
            n *= 4;
        }
    }
    if st[0].failed.is_some() {
        panic!("ours panics on {}", c.key());
    }

    // Check against ours.
    let tmax = st.iter().filter(|s| s.failed.is_none()).map(|s| s.t).fold(0.0, f64::max);
    let ncheck = ((60.0e6 / tmax) as usize).clamp(64, POOL);
    let mut ours_out = vec![[T::default(); 4]; ncheck];
    (c.ours)(&pool[..ncheck], &mut ours_out);
    for s in st.iter_mut().skip(1) {
        if s.failed.is_some() {
            continue;
        }
        let mut ro = vec![[T::default(); 4]; ncheck];
        if !call(s.f, &pool[..ncheck], &mut ro) {
            s.failed = Some("panics".into());
            continue;
        }
        let cs = compare(&ours_out, &ro, c.nout, s.cmp, c.metric);
        if cs.mism * 5 > cs.n {
            s.failed = Some(format!("NaN/inf where ours is finite (or the reverse) on {}/{}", cs.mism, cs.n));
        }
        s.stats = Some(cs);
    }

    // Timing.
    let alive: Vec<usize> = (0..st.len()).filter(|&i| st[i].failed.is_none()).collect();
    let tfast = alive.iter().map(|&i| st[i].t).fold(f64::INFINITY, f64::min).max(0.05);
    let tslow = alive.iter().map(|&i| st[i].t).fold(0.0, f64::max).max(0.05);
    let mut m = (o.target_ns / tfast).ceil() as usize;
    if m as f64 * tslow > o.max_sample_ns {
        m = ((o.max_sample_ns / tslow).ceil() as usize).max(16);
    }
    for &i in &alive {
        st[i].k = ((o.target_ns / (m as f64 * st[i].t)).ceil() as usize).max(1);
    }
    let rounds = o.warm + o.samples;
    for r in 0..rounds {
        let off = (r * m) % POOL;
        for j in 0..alive.len() {
            let i = alive[(j + r) % alive.len()];
            if st[i].failed.is_some() {
                continue;
            }
            let k = st[i].k;
            match timed(st[i].f, &pool, &mut out, off, m, k) {
                Some(ns) => {
                    if r >= o.warm {
                        st[i].samples.push(ns / (m * k) as f64)
                    }
                }
                None => st[i].failed = Some("panics while timed".into()),
            }
        }
    }

    let mut res: Vec<ImpResult> = st
        .into_iter()
        .map(|mut s| {
            let (med, q25, q75) = if s.failed.is_none() && !s.samples.is_empty() {
                (quant(&mut s.samples, 0.5), quant(&mut s.samples, 0.25), quant(&mut s.samples, 0.75))
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            };
            ImpResult { name: s.name, note: s.note, failed: s.failed, median: med, q25, q75, cmp: s.stats }
        })
        .collect();
    let ours = res.remove(0);
    CaseResult { ours, refs: res }
}

pub fn sig3(x: f64) -> String {
    if !x.is_finite() {
        return "-".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    let e = x.abs().log10().floor() as i32;
    let d = (2 - e).max(0) as usize;
    let p = 10f64.powi(2 - e);
    let r = (x * p).round() / p;
    format!("{r:.d$}")
}

fn fmt_u(x: f64) -> String {
    if !x.is_finite() {
        "-".into()
    } else if x < 0.05 {
        "0".into()
    } else if x < 100.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.1e}")
    }
}

/// TSV rows (without the header) of a case's result.
pub fn rows<T: Elem>(c: &Case<T>, r: &CaseResult, lto: Option<f64>) -> Vec<String> {
    let mut v = Vec::new();
    let spread = |x: &ImpResult| (x.q75 - x.q25) / x.median;
    let ours_ns = r.ours.median;
    let mut base_notes = Vec::new();
    if !c.note.is_empty() {
        base_notes.push(c.note.clone());
    }
    if let Some(l) = lto {
        if (l / ours_ns - 1.0).abs() > 0.10 {
            base_notes.push(format!("ours with fat LTO {} ns", sig3(l)));
        }
    }
    if c.refs.is_empty() && c.scipy.is_none() {
        let mut n = base_notes.clone();
        n.push("no reference computes this".into());
        v.push(format!(
            "special\t{}\t{}\t{}\t{}\tnone\t-\t-\t{}",
            c.group,
            c.op,
            c.params,
            sig3(ours_ns),
            n.join("; ")
        ));
    }
    for x in &r.refs {
        let mut n = base_notes.clone();
        if !x.note.is_empty() {
            n.push(x.note.clone());
        }
        if let Some(s) = &x.cmp {
            let mut t = format!(
                "{}: max {} p99 {} median {} (n={})",
                c.metric.label(),
                fmt_u(s.max),
                fmt_u(s.p99),
                fmt_u(s.med),
                s.n
            );
            if s.mism > 0 {
                t.push_str(&format!("; NaN/inf mismatches {}/{}", s.mism, s.n));
            }
            n.push(t);
        }
        let (rn, ratio) = match &x.failed {
            Some(f) => {
                n.push(format!("not timed: {f}"));
                ("-".to_string(), "-".to_string())
            }
            None => {
                let sp = spread(x).max(spread(&r.ours));
                if sp > 0.10 {
                    n.push(format!(
                        "spread (IQR/median) ours {:.0}% ref {:.0}%",
                        100.0 * spread(&r.ours),
                        100.0 * spread(x)
                    ));
                }
                (sig3(x.median), sig3(ours_ns / x.median))
            }
        };
        v.push(format!(
            "special\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            c.group,
            c.op,
            c.params,
            sig3(ours_ns),
            x.name,
            rn,
            ratio,
            n.join("; ")
        ));
    }
    v
}

/// The difference metric of a case: complex results by |w|; the oscillatory functions (with
/// zeros inside the region) in ulps of max(|v|, median |v|); the rest relative.
fn classify<T: Elem>(c: &mut Case<T>) {
    let op = c.op.as_str();
    let osc = c.group == "orthopoly"
        || op.starts_with("bessel_j")
        || op.starts_with("bessel_y")
        || op.starts_with("sph_bessel_j")
        || op.starts_with("sph_bessel_y")
        || op.starts_with("jacobi_elliptic")
        || (op.starts_with("airy") && c.params.starts_with("x in [-"))
        || (op == "polylog" && c.params.starts_with("s integer in [-"));
    if c.group == "complex" {
        c.metric = Metric::Complex;
    } else if osc {
        c.metric = Metric::Scaled;
    }
}

pub fn cases64() -> Vec<Case<f64>> {
    let mut v = Vec::new();
    cases_gamma::add(&mut v);
    cases_erf::add(&mut v);
    cases_bessel::add(&mut v);
    cases_zeta::add(&mut v);
    cases_elliptic::add(&mut v);
    cases_ortho::add(&mut v);
    v.iter_mut().for_each(classify);
    v
}

pub fn cases32() -> Vec<Case<f32>> {
    let mut v = Vec::new();
    cases_f32::add(&mut v);
    v.iter_mut().for_each(classify);
    v
}

/// Generator helpers: a record from one uniform argument and so on.
pub fn g_x(a: f64, b: f64) -> impl Fn(&mut Rng) -> Rec<f64> + Send + Sync + 'static {
    move |r| [r.uni(a, b), 0.0, 0.0, 0.0]
}
pub fn g_logx(a: f64, b: f64) -> impl Fn(&mut Rng) -> Rec<f64> + Send + Sync + 'static {
    move |r| [r.logu(a, b), 0.0, 0.0, 0.0]
}

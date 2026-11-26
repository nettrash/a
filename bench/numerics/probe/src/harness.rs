//! Timing: every closure is one operation; the closures of one comparison are sampled in turn
//! (ABAB…), each sample batched to at least ~1 ms, and the median ns per operation is reported.

use std::hint::black_box;
use std::time::Instant;

/// NUMPROBE_QUICK=1: one short sample per closure, a functional check of the harness, not a timing.
pub fn quick() -> bool {
    std::env::var("NUMPROBE_QUICK").is_ok()
}

pub fn samples() -> usize {
    if quick() {
        return 1;
    }
    std::env::var("NUMPROBE_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15)
        .max(15)
}

/// The batch target of one sample, in ns.
const TARGET_NS: f64 = 1.2e6;

pub struct Timing {
    pub median: f64,
    /// (q3 − q1) / median over the samples.
    pub spread: f64,
}

/// Times each closure: warm-up, calibration, then `samples()` alternating samples.
pub fn measure(fs: &mut [&mut dyn FnMut()]) -> Vec<Timing> {
    let k = fs.len();
    let mut batch = vec![1usize; k];
    // Warm-up and calibration: run each for ~5 ms or 3 calls, whichever is longer.
    for (i, f) in fs.iter_mut().enumerate() {
        let start = Instant::now();
        let mut calls = 0usize;
        loop {
            f();
            calls += 1;
            let e = start.elapsed().as_nanos() as f64;
            if quick() || (e > 5e6 && calls >= 3) || e > 2e9 {
                let per = e / calls as f64;
                batch[i] = if quick() { 1 } else { ((TARGET_NS / per).ceil() as usize).max(1) };
                break;
            }
        }
    }
    let s = samples();
    let mut t: Vec<Vec<f64>> = vec![Vec::with_capacity(s); k];
    for round in 0..s {
        for j in 0..k {
            // ABAB… with the order turned every other round, so that no side always goes first.
            let i = if round % 2 == 0 { j } else { k - 1 - j };
            let f = &mut fs[i];
            let start = Instant::now();
            for _ in 0..batch[i] {
                f();
            }
            let e = start.elapsed().as_nanos() as f64;
            t[i].push(e / batch[i] as f64);
        }
    }
    t.into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            let med = if n % 2 == 1 {
                v[n / 2]
            } else {
                0.5 * (v[n / 2 - 1] + v[n / 2])
            };
            let q1 = v[n / 4];
            let q3 = v[(3 * n) / 4];
            Timing {
                median: med,
                spread: (q3 - q1) / med,
            }
        })
        .collect()
}

/// One row of the scoreboard.
pub struct Row {
    pub group: String,
    pub op: String,
    pub params: String,
    pub ours: f64,
    pub reference: String,
    pub refns: Option<f64>,
    pub notes: String,
}

pub fn fmt_ns(x: f64) -> String {
    if !x.is_finite() {
        "NA".into()
    } else if x >= 100.0 {
        format!("{:.0}", x)
    } else if x >= 10.0 {
        format!("{:.1}", x)
    } else {
        format!("{:.2}", x)
    }
}

pub fn sig3(x: f64) -> String {
    if !x.is_finite() {
        return "NA".into();
    }
    let d = if x == 0.0 {
        0
    } else {
        (2 - x.abs().log10().floor() as i32).max(0)
    };
    format!("{:.*}", d as usize, x)
}

unsafe extern "C" {
    fn probe_gsl_version() -> *const std::ffi::c_char;
    fn probe_boost_version() -> *const std::ffi::c_char;
}

/// The reference names are written with the versions of this machine (GSL 2.8, Boost 1.92); the
/// versions actually linked replace them when they differ (another machine, CI).
fn versioned(s: &str) -> String {
    use std::sync::OnceLock;
    static V: OnceLock<(String, String)> = OnceLock::new();
    let (g, b) = V.get_or_init(|| unsafe {
        (
            std::ffi::CStr::from_ptr(probe_gsl_version()).to_string_lossy().into_owned(),
            std::ffi::CStr::from_ptr(probe_boost_version()).to_string_lossy().into_owned(),
        )
    });
    s.replace("GSL 2.8", &format!("GSL {g}")).replace("Boost 1.92", &format!("Boost {b}"))
}

impl Row {
    pub fn print(&self) {
        let ratio = match self.refns {
            Some(r) if r > 0.0 => sig3(self.ours / r),
            _ => "NA".into(),
        };
        let refns = self.refns.map(fmt_ns).unwrap_or_else(|| "NA".into());
        println!(
            "numerics\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.group,
            self.op,
            self.params,
            fmt_ns(self.ours),
            versioned(&self.reference),
            refns,
            ratio,
            versioned(&self.notes).replace('\t', " ")
        );
    }
}

/// Spread note, when wide (> 10 %).
pub fn spread_note(t: &[&Timing]) -> String {
    let w: f64 = t.iter().map(|t| t.spread).fold(0.0, f64::max);
    if w > 0.10 {
        format!("; spread {:.0}%", 100.0 * w)
    } else {
        String::new()
    }
}

/// Prints rows for one comparison: `ours` against each reference.
pub struct Cmp<'a> {
    pub group: &'a str,
    pub op: &'a str,
    pub params: &'a str,
}

impl Cmp<'_> {
    /// `fs[0]` is ours; `names[i]`/`notes[i]` for each reference `fs[i+1]`; `ours_note` goes first
    /// in every row's notes. `per` divides the times (operations per call).
    pub fn run(
        &self,
        fs: &mut [&mut dyn FnMut()],
        names: &[&str],
        ours_note: &str,
        notes: &[String],
        per: f64,
    ) -> Vec<Timing> {
        let t = measure(fs);
        for (i, name) in names.iter().enumerate() {
            let sp = spread_note(&[&t[0], &t[i + 1]]);
            let mut note = String::new();
            if !ours_note.is_empty() {
                note.push_str("ours: ");
                note.push_str(ours_note);
            }
            if !notes[i].is_empty() {
                if !note.is_empty() {
                    note.push_str("; ");
                }
                note.push_str("ref: ");
                note.push_str(&notes[i]);
            }
            note.push_str(&sp);
            Row {
                group: self.group.into(),
                op: self.op.into(),
                params: self.params.into(),
                ours: t[0].median / per,
                reference: (*name).into(),
                refns: Some(t[i + 1].median / per),
                notes: note,
            }
            .print();
        }
        if names.is_empty() {
            Row {
                group: self.group.into(),
                op: self.op.into(),
                params: self.params.into(),
                ours: t[0].median / per,
                reference: "none".into(),
                refns: None,
                notes: format!("ours: {ours_note}{}", spread_note(&[&t[0]])),
            }
            .print();
        }
        t
    }
}

/// The Python references (SciPy, NumPy, HiGHS), read from the file $NUMPROBE_PYREF that
/// py/refs.py wrote: key → (reference, kind, ns, evaluations, note).
pub struct PyRef {
    pub reference: String,
    pub kind: String,
    pub ns: f64,
    pub nfev: f64,
    pub note: String,
}

pub fn pyrefs() -> &'static std::collections::HashMap<String, Vec<PyRef>> {
    use std::sync::OnceLock;
    static R: OnceLock<std::collections::HashMap<String, Vec<PyRef>>> = OnceLock::new();
    R.get_or_init(|| {
        let mut m: std::collections::HashMap<String, Vec<PyRef>> = Default::default();
        if let Ok(path) = std::env::var("NUMPROBE_PYREF") {
            if let Ok(s) = std::fs::read_to_string(path) {
                for line in s.lines() {
                    let f: Vec<&str> = line.split('\t').collect();
                    if f.len() < 6 {
                        continue;
                    }
                    m.entry(f[0].to_string()).or_default().push(PyRef {
                        reference: f[1].to_string(),
                        kind: f[2].to_string(),
                        ns: f[3].parse().unwrap_or(f64::NAN),
                        nfev: f[4].parse().unwrap_or(0.0),
                        note: f[5].to_string(),
                    });
                }
            }
        }
        m
    })
}

/// Prints a row for every Python reference recorded under `key`, against ours (already timed).
/// `per_eval` is ours cost of one evaluation of the user function (for the "solver" kind).
pub fn py_rows(group: &str, op: &str, params: &str, key: &str, ours: f64, ours_note: &str, per_eval: Option<f64>) {
    let Some(refs) = pyrefs().get(key) else {
        if std::env::var("NUMPROBE_PYREF").is_ok() {
            eprintln!("no Python reference for {key}");
        }
        return;
    };
    for r in refs {
        let (refns, how) = match r.kind.as_str() {
            "ns" => (Some(r.ns), "separate process (not interleaved)".to_string()),
            "solver" => {
                let pe = per_eval.unwrap_or(0.0);
                (Some(r.ns + r.nfev * pe), format!("ref_ns = SciPy's solver time {:.0} ns + {} evaluations × ours {:.0} ns per f+g; separate process", r.ns, r.nfev, pe))
            }
            _ => (None, "counts only".to_string()),
        };
        let mut note = String::new();
        if !ours_note.is_empty() {
            note.push_str("ours: ");
            note.push_str(ours_note);
            note.push_str("; ");
        }
        note.push_str("ref: ");
        note.push_str(&r.note);
        if !r.note.contains(&how) {
            note.push_str("; ");
            note.push_str(&how);
        }
        Row {
            group: group.into(),
            op: op.into(),
            params: params.into(),
            ours,
            reference: r.reference.clone(),
            refns,
            notes: note,
        }
        .print();
    }
}

/// Times ours alone and prints a row against an externally measured reference (Python), whose
/// time `refns` was measured in another process (not interleaved).
pub fn row_external(
    group: &str,
    op: &str,
    params: &str,
    f: &mut dyn FnMut(),
    per: f64,
    reference: &str,
    refns: Option<f64>,
    note: String,
) {
    let t = measure(&mut [f]);
    let sp = spread_note(&[&t[0]]);
    Row {
        group: group.into(),
        op: op.into(),
        params: params.into(),
        ours: t[0].median / per,
        reference: reference.into(),
        refns,
        notes: format!("{note}{sp}"),
    }
    .print();
}

/// A deterministic stream in [0, 1).
pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub fn bb<T>(x: T) -> T {
    black_box(x)
}

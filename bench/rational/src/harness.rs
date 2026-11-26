// Timing: each contender is a closure that performs `count` operations over its own pool of
// inputs. Batches are calibrated to at least 1 ms; the samples alternate the contenders in an
// order that turns with each sample (ABAB...), and the median of each is reported.
#![allow(dead_code)]
use std::io::Write;
use std::time::Instant;

pub struct Ent<'a> {
    pub name: &'static str,
    pub run: Box<dyn FnMut(usize) + 'a>,
    pub note: String,
}

impl<'a> Ent<'a> {
    pub fn new(name: &'static str, run: impl FnMut(usize) + 'a) -> Ent<'a> {
        Ent { name, run: Box::new(run), note: String::new() }
    }
    pub fn note(mut self, s: &str) -> Ent<'a> {
        if !s.is_empty() {
            if !self.note.is_empty() {
                self.note.push_str("; ");
            }
            self.note.push_str(s);
        }
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stat {
    pub med: f64,
    pub q1: f64,
    pub q3: f64,
    pub samples: usize,
}

fn env_f(name: &str, d: f64) -> f64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn time(run: &mut Box<dyn FnMut(usize) + '_>, count: usize) -> f64 {
    let t = Instant::now();
    run(count);
    t.elapsed().as_nanos() as f64
}

/// Runs the race; `per` divides each operation's time (items per operation).
pub fn race(ents: &mut [Ent], per: f64) -> Vec<Stat> {
    let target = env_f("RB_TARGET_MS", 1.2) * 1e6;
    let mut samples = env_f("RB_SAMPLES", 21.0) as usize;
    let k = ents.len();
    // Calibrate: warm up, then grow the batch until it takes at least the target.
    let mut counts = vec![1usize; k];
    let mut single = vec![0f64; k];
    for (i, e) in ents.iter_mut().enumerate() {
        // The first call warms up; a long one is not repeated.
        let mut t1 = time(&mut e.run, 1);
        if t1 < 10e6 {
            t1 = t1.min(time(&mut e.run, 1));
        }
        single[i] = t1;
        let mut c = ((target / t1.max(1.0)).ceil() as usize).clamp(1, 1 << 26);
        if t1 < target {
            loop {
                let t = time(&mut e.run, c);
                if t >= target || c >= 1 << 26 {
                    break;
                }
                c = (c as f64 * (target / t.max(1.0)) * 1.1).ceil() as usize + 1;
            }
        }
        counts[i] = c;
    }
    // Long operations: 15 samples.
    let longest = single.iter().cloned().fold(0.0, f64::max);
    if longest > 30e6 {
        samples = samples.min(env_f("RB_SAMPLES_LONG", 15.0) as usize);
    }
    let mut ts: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); k];
    for s in 0..samples {
        for o in 0..k {
            let i = (s + o) % k;
            let t = time(&mut ents[i].run, counts[i]);
            ts[i].push(t / counts[i] as f64 / per);
        }
    }
    ts.into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
            Stat { med: q(0.5), q1: q(0.25), q3: q(0.75), samples: v.len() }
        })
        .collect()
}

fn sig3(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let d = x.abs().log10().floor() as i32;
    let p = (2 - d).max(0) as usize;
    format!("{x:.p$}")
}

fn ns(x: f64) -> String {
    if x >= 1000.0 {
        format!("{x:.0}")
    } else if x >= 10.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.2}")
    }
}

fn spread(s: &Stat) -> f64 {
    (s.q3 - s.q1) / s.med
}

/// Runs and writes one row per reference: ours is the first contender.
pub fn row(group: &str, op: &str, params: &str, mut ents: Vec<Ent>, per: f64, note: &str) {
    // RB_FILTER: comma-separated substrings of "group|operation|params"; a row matching none is
    // left out (for quick runs while tuning).
    if let Ok(f) = std::env::var("RB_FILTER") {
        let key = format!("{group}|{op}|{params}");
        if !f.split(',').any(|p| key.contains(p)) {
            return;
        }
    }
    if ents.len() < 2 {
        eprintln!("skip {group} {op} {params}: no reference");
        return;
    }
    let st = race(&mut ents, per);
    let out = std::io::stdout();
    let mut out = out.lock();
    let o = st[0];
    for (i, e) in ents.iter().enumerate().skip(1) {
        let r = st[i];
        let mut notes: Vec<String> = Vec::new();
        if !note.is_empty() {
            notes.push(note.to_string());
        }
        if !ents[0].note.is_empty() {
            notes.push(format!("ours: {}", ents[0].note));
        }
        if !e.note.is_empty() {
            notes.push(e.note.clone());
        }
        if spread(&o) > 0.10 || spread(&r) > 0.10 {
            notes.push(format!("IQR/median ours {:.0}% ref {:.0}%", 100.0 * spread(&o), 100.0 * spread(&r)));
        }
        if o.samples < 21 {
            notes.push(format!("{} samples", o.samples));
        }
        writeln!(
            out,
            "rational\t{group}\t{op}\t{params}\t{}\t{}\t{}\t{}\t{}",
            ns(o.med),
            e.name,
            ns(r.med),
            sig3(o.med / r.med),
            notes.join("; ")
        )
        .unwrap();
    }
    out.flush().unwrap();
}

/// Compares check strings; returns a note when some differ ("" when all agree).
pub fn agree(ours: &[String], theirs: &[String], what: &str) -> String {
    let bad = ours.iter().zip(theirs).filter(|(a, b)| a != b).count();
    if bad > 0 {
        let (i, (a, b)) = ours.iter().zip(theirs).enumerate().find(|(_, (a, b))| a != b).unwrap();
        let cut = |s: &str| s.chars().take(80).collect::<String>();
        eprintln!("MISMATCH {what}: {bad}/{} differ; first at {i}: ours {} theirs {}", ours.len(), cut(a), cut(b));
        format!("MISMATCH {bad}/{}", ours.len())
    } else {
        String::new()
    }
}

#[macro_export]
macro_rules! cyc {
    ($j:ident, $n:expr, $count:expr, $body:expr) => {{
        let n = $n;
        for _ in 0..$count {
            $body;
            $j += 1;
            if $j == n {
                $j = 0;
            }
        }
    }};
}

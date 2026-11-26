// The timing harness shared by the benchmarks of the area "curves" (the v0.2 speed pass).
//
// Every row times our operation and each reference in the same process, sample by sample in
// rotating order (ABAB…), so that load and clock changes hit all sides alike. A sample is a batch
// of at least `BENCH_MIN_NS` (default 1 ms) of work; the reported figure is the median over
// `BENCH_SAMPLES` (default 21) samples of the time per operation. Inputs cycle through pools of
// distinct values, indexed by a running counter, so that repeated identical inputs do not flatter
// caches or branch predictors.
//
// Environment knobs (for quick checks while developing; run.sh leaves them unset):
//   BENCH_SAMPLES  samples per row (>= 15 for the scoreboard)
//   BENCH_MIN_NS   minimum batch length in ns
//   BENCH_FILTER   run only the rows whose operation or params contain this substring

#![allow(dead_code)]

pub use std::hint::black_box;
use std::time::Instant;

/// SplitMix64, the seeded generator of every input.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n + 8);
        while v.len() < n {
            v.extend_from_slice(&self.next().to_le_bytes());
        }
        v.truncate(n);
        v
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One side of a row: a name, notes for its row, and the operation on input number `i`.
pub struct Side<'a> {
    pub name: String,
    pub notes: String,
    pub f: Box<dyn FnMut(usize) + 'a>,
}

pub fn side<'a>(name: &str, notes: &str, f: impl FnMut(usize) + 'a) -> Side<'a> {
    Side {
        name: name.to_string(),
        notes: notes.to_string(),
        f: Box::new(f),
    }
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

pub struct Bench {
    pub group: String,
    samples: usize,
    min_ns: f64,
    filter: Option<String>,
}

/// Formats a number to 3 significant digits.
pub fn sig3(x: f64) -> String {
    if !x.is_finite() || x == 0.0 {
        return format!("{x}");
    }
    let d = (x.abs().log10().floor() as i32) + 1;
    let decimals = (3 - d).max(0) as usize;
    format!("{x:.decimals$}")
}

fn fmt_ns(x: f64) -> String {
    if x >= 1e5 {
        format!("{x:.0}")
    } else if x >= 100.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.2}")
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The interquartile range over the median, in percent.
fn spread(v: &[f64], med: f64) -> f64 {
    let n = v.len();
    let q1 = v[n / 4];
    let q3 = v[(3 * n) / 4];
    (q3 - q1) / med * 100.0
}

impl Bench {
    pub fn new(group: &str) -> Bench {
        Bench {
            group: group.to_string(),
            samples: env_usize("BENCH_SAMPLES", 21),
            min_ns: env_usize("BENCH_MIN_NS", 1_000_000) as f64,
            filter: std::env::var("BENCH_FILTER").ok().filter(|s| !s.is_empty()),
        }
    }

    pub fn set_group(&mut self, group: &str) {
        self.group = group.to_string();
    }

    pub fn wants(&self, op: &str, params: &str) -> bool {
        match &self.filter {
            None => true,
            Some(f) => op.contains(f.as_str()) || params.contains(f.as_str()),
        }
    }

    /// Times `ours` against each reference and prints one TSV row per reference (one row with
    /// reference `none` when there is none).
    pub fn run<'a>(
        &self,
        op: &str,
        params: &str,
        ours_notes: &str,
        ours: impl FnMut(usize) + 'a,
        refs: Vec<Side<'a>>,
    ) {
        if !self.wants(op, params) {
            return;
        }
        let mut sides: Vec<Side<'a>> = Vec::with_capacity(refs.len() + 1);
        sides.push(side("ours", ours_notes, ours));
        sides.extend(refs);
        let k = sides.len();
        let mut counters = vec![0usize; k];
        // Calibrate: the batch length that reaches min_ns.
        let mut batch = vec![1usize; k];
        for s in 0..k {
            let mut n = 1usize;
            loop {
                let t0 = Instant::now();
                for j in 0..n {
                    (sides[s].f)(counters[s] + j);
                }
                let t = t0.elapsed().as_nanos() as f64;
                counters[s] += n;
                if t >= self.min_ns || n >= 1 << 26 {
                    break;
                }
                let want = ((self.min_ns * 1.25) / t.max(1.0) * n as f64).ceil() as usize;
                n = want.clamp(n + 1, n.saturating_mul(16));
            }
            batch[s] = n;
        }
        // Warm up once more at the final batch length.
        for s in 0..k {
            for j in 0..batch[s] {
                (sides[s].f)(counters[s] + j);
            }
            counters[s] += batch[s];
        }
        let mut per_op: Vec<Vec<f64>> = vec![Vec::with_capacity(self.samples); k];
        for i in 0..self.samples {
            for r in 0..k {
                let s = (i + r) % k;
                let n = batch[s];
                let c = counters[s];
                let t0 = Instant::now();
                for j in 0..n {
                    (sides[s].f)(c + j);
                }
                let t = t0.elapsed().as_nanos() as f64;
                counters[s] += n;
                per_op[s].push(t / n as f64);
            }
        }
        let meds: Vec<f64> = per_op.iter_mut().map(|v| median(v)).collect();
        let spreads: Vec<f64> = per_op
            .iter()
            .zip(&meds)
            .map(|(v, &m)| spread(v, m))
            .collect();
        let ours_ns = meds[0];
        let wide = |s: f64| s > 10.0;
        let base_notes = |extra: &str, ref_spread: Option<f64>| {
            let mut parts: Vec<String> = Vec::new();
            match (sides[0].notes.is_empty(), extra.is_empty()) {
                (true, true) => {}
                (false, true) => parts.push(format!("ours: {}", sides[0].notes)),
                (true, false) => parts.push(extra.to_string()),
                (false, false) => parts.push(format!("ours: {}; ref: {}", sides[0].notes, extra)),
            }
            let mut sp = Vec::new();
            if wide(spreads[0]) {
                sp.push(format!("ours {:.0}%", spreads[0]));
            }
            if let Some(r) = ref_spread {
                if wide(r) {
                    sp.push(format!("ref {r:.0}%"));
                }
            }
            if !sp.is_empty() {
                parts.push(format!("spread (IQR/median) {}", sp.join(", ")));
            }
            parts.join("; ")
        };
        if k == 1 {
            println!(
                "curves\t{}\t{}\t{}\t{}\tnone\t\t\t{}",
                self.group,
                op,
                params,
                fmt_ns(ours_ns),
                {
                    let mut n = base_notes("", None);
                    n = if n.is_empty() { "no reference does this work".to_string() } else { format!("{n}; no reference does this work") };
                    n
                }
            );
        }
        for s in 1..k {
            println!(
                "curves\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                self.group,
                op,
                params,
                fmt_ns(ours_ns),
                sides[s].name,
                fmt_ns(meds[s]),
                sig3(ours_ns / meds[s]),
                base_notes(&sides[s].notes, Some(spreads[s]))
            );
        }
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
}

/// Lower-case hexadecimal.
pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Big-endian bytes of a `Number`, `len` long.
pub fn number_be(n: &a::math::numbers::Number, len: usize) -> Vec<u8> {
    use a::math::numbers::ByteOrder;
    let mut v = n.value.clone();
    if n.byte_order == ByteOrder::LittleEndian {
        v.reverse();
    }
    while v.len() > 1 && v[0] == 0 {
        v.remove(0);
    }
    if v == [0] {
        v.clear();
    }
    assert!(v.len() <= len, "number longer than {len} bytes");
    let mut out = vec![0u8; len - v.len()];
    out.extend(v);
    out
}

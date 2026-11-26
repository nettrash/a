//! Timing, pools of inputs, rows of the scoreboard.
#![allow(dead_code)]
use crate::ffi::H;
use a::math::numbers::{ByteOrder, Number};
use std::cell::OnceCell;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

// ---------------------------------------------------------------------------------------------
// Random numbers (splitmix64), deterministic per seed.

pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A value below `n` (n > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
    /// A random number of exactly `bits` bits (top bit set), optionally odd.
    pub fn bits(&mut self, bits: u32, odd: bool) -> Number {
        let limbs = bits.div_ceil(64) as usize;
        let mut v: Vec<u64> = (0..limbs).map(|_| self.next()).collect();
        let top = bits - 64 * (limbs as u32 - 1);
        if top < 64 {
            v[limbs - 1] &= (1u64 << top) - 1;
        }
        v[limbs - 1] |= 1u64 << (top - 1);
        if odd {
            v[0] |= 1;
        }
        from_limbs(&v)
    }
    /// A random number below `m`.
    pub fn below_num(&mut self, m: &Number) -> Number {
        let b = m.bits() as u32;
        let limbs = b.div_ceil(64) as usize + 1;
        let v: Vec<u64> = (0..limbs).map(|_| self.next()).collect();
        &from_limbs(&v) % m
    }
    /// A random prime of exactly `bits` bits (next prime after a random number of `bits` bits
    /// whose second-highest bit is clear, so that it does not overflow).
    pub fn prime(&mut self, bits: u32) -> Number {
        loop {
            let x = self.bits(bits, true);
            let p = x.next_prime();
            if p.bits() == bits as u64 {
                return p;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Conversions.

pub fn from_limbs(l: &[u64]) -> Number {
    let mut b = Vec::with_capacity(l.len() * 8);
    for x in l {
        b.extend_from_slice(&x.to_le_bytes());
    }
    while b.len() > 1 && *b.last().unwrap() == 0 {
        b.pop();
    }
    Number::new(b, ByteOrder::LittleEndian)
}

pub fn limbs(n: &Number) -> Vec<u64> {
    let mut b = n.value.clone();
    if n.byte_order == ByteOrder::BigEndian {
        b.reverse();
    }
    while b.last() == Some(&0) {
        b.pop();
    }
    b.chunks(8)
        .map(|c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(w)
        })
        .collect()
}

/// The low 64 bits of the magnitude.
pub fn lo(n: &Number) -> u64 {
    match n.byte_order {
        ByteOrder::LittleEndian => {
            let mut w = [0u8; 8];
            let k = n.value.len().min(8);
            w[..k].copy_from_slice(&n.value[..k]);
            u64::from_le_bytes(w)
        }
        ByteOrder::BigEndian => {
            let v = &n.value;
            let k = v.len().min(8);
            let mut w = [0u8; 8];
            for i in 0..k {
                w[i] = v[v.len() - 1 - i];
            }
            u64::from_le_bytes(w)
        }
    }
}

pub fn num(x: u64) -> Number {
    Number::new_from_u64(x, ByteOrder::LittleEndian)
}
pub fn num128(x: u128) -> Number {
    Number::new_from_u128(x, ByteOrder::LittleEndian)
}
pub fn to_u128(n: &Number) -> u128 {
    let l = limbs(n);
    l.first().copied().unwrap_or(0) as u128 | (l.get(1).copied().unwrap_or(0) as u128) << 64
}
/// A decimal number.
pub fn parse(s: &str) -> Number {
    Number::from_str_radix(s, 10, ByteOrder::LittleEndian).expect("a decimal number")
}

/// The smaller of `f` and `n / f`, its low limb: the same for either prime of a semiprime.
pub fn lo_minpair(f: &Number, n: &Number) -> u64 {
    let g = n / f;
    if g < *f { lo(&g) } else { lo(f) }
}

// ---------------------------------------------------------------------------------------------
// Pools: the numbers, and each library's copy, made on first use.

pub struct Pool {
    pub nums: Vec<Number>,
    limbs: Vec<u64>,
    off: Vec<u64>,
    g: OnceCell<usize>,
    f: OnceCell<usize>,
    p: OnceCell<usize>,
}

impl Pool {
    pub fn new(nums: Vec<Number>) -> Pool {
        let mut l = vec![];
        let mut off = vec![0u64];
        for n in &nums {
            l.extend(limbs(n));
            off.push(l.len() as u64);
        }
        Pool { nums, limbs: l, off, g: OnceCell::new(), f: OnceCell::new(), p: OnceCell::new() }
    }
    pub fn len(&self) -> usize {
        self.nums.len()
    }
    #[cfg(has_gmp)]
    pub fn gmp(&self) -> H {
        *self.g.get_or_init(|| unsafe { crate::ffi::gmp_pool(self.limbs.as_ptr(), self.off.as_ptr(), self.nums.len()) } as usize) as H
    }
    #[cfg(has_flint)]
    pub fn flint(&self) -> H {
        *self.f.get_or_init(|| unsafe { crate::ffi::flint_pool(self.limbs.as_ptr(), self.off.as_ptr(), self.nums.len()) } as usize) as H
    }
    #[cfg(has_pari)]
    pub fn pari(&self) -> H {
        *self.p.get_or_init(|| unsafe { crate::ffi::pari_pool(self.limbs.as_ptr(), self.off.as_ptr(), self.nums.len()) } as usize) as H
    }
    pub fn get(&self, i: usize) -> &Number {
        &self.nums[i % self.nums.len()]
    }
}

pub fn null() -> H {
    std::ptr::null_mut()
}

// ---------------------------------------------------------------------------------------------
// The timer: `Instant` (wall time), or with NT_CPUTIME=1 the calling thread's CPU time, for runs
// on a machine so loaded (load average above the cores) that wall time measures the scheduler.

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}
unsafe extern "C" {
    fn clock_gettime(clk: i32, tp: *mut Timespec) -> i32;
}
#[cfg(target_os = "macos")]
const CLOCK_THREAD_CPUTIME_ID: i32 = 16;
#[cfg(not(target_os = "macos"))]
const CLOCK_THREAD_CPUTIME_ID: i32 = 3;

fn cpu_ns() -> f64 {
    let mut t = Timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime writes the struct it is given.
    unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut t) };
    t.tv_sec as f64 * 1e9 + t.tv_nsec as f64
}

pub fn cputime() -> bool {
    std::env::var("NT_CPUTIME").is_ok_and(|s| s == "1")
}

/// Nanoseconds that `f` took.
fn timed<T>(f: impl FnOnce() -> T) -> f64 {
    if cputime() {
        let t = cpu_ns();
        black_box(f());
        cpu_ns() - t
    } else {
        let t = Instant::now();
        black_box(f());
        t.elapsed().as_nanos() as f64
    }
}

// ---------------------------------------------------------------------------------------------
// Contenders and races.

pub struct C<'a> {
    pub name: String,
    pub f: Box<dyn FnMut(usize, usize) -> u64 + 'a>,
    pub note: String,
    pub check: bool,
    /// Items per call of this contender (its time per call is divided by it, and by the row's).
    pub per: f64,
}

/// A contender: `f(i0, k)` makes `k` calls on the items `i0, i0 + 1, ...` of its pool (modulo its
/// length) and returns a checksum of the results.
pub fn c<'a>(name: &str, f: impl FnMut(usize, usize) -> u64 + 'a) -> C<'a> {
    C { name: name.to_string(), f: Box::new(f), note: String::new(), check: true, per: 1.0 }
}

impl<'a> C<'a> {
    /// A note for this reference's row (same-work caveats).
    pub fn note(mut self, s: &str) -> Self {
        self.note = s.to_string();
        self
    }
    /// Items per call.
    pub fn per(mut self, p: f64) -> Self {
        self.per = p;
        self
    }
    /// Its checksum is not compared with ours (it computes a valid but different answer).
    pub fn nocheck(mut self) -> Self {
        self.check = false;
        self
    }
}

pub struct Row {
    pub op: String,
    pub params: String,
    /// Items per call: the time per call is divided by it (sets of numbers timed as one call).
    pub per: f64,
    pub notes: String,
    /// The items the check runs (0 = no check).
    pub n_check: usize,
    /// The largest batch of calls (to keep the inputs fresh within a pool).
    pub max_k: usize,
    /// Inputs whose costs differ from item to item: every contender runs the same items in a
    /// sample (the largest of their batches, cut so that the slowest takes at most 2 s), so that
    /// all measure the mean over the same items, not a median of single items.
    pub same_k: bool,
}

pub fn row(op: &str, params: &str) -> Row {
    Row { op: op.to_string(), params: params.to_string(), per: 1.0, notes: String::new(), n_check: 16, max_k: usize::MAX, same_k: false }
}
impl Row {
    pub fn per(mut self, p: f64) -> Self {
        self.per = p;
        self
    }
    pub fn notes(mut self, s: &str) -> Self {
        self.notes = s.to_string();
        self
    }
    pub fn check(mut self, n: usize) -> Self {
        self.n_check = n;
        self
    }
    pub fn max_k(mut self, k: usize) -> Self {
        self.max_k = k;
        self
    }
    pub fn same_k(mut self) -> Self {
        self.same_k = true;
        self
    }
}

pub struct Ctx {
    pub group: String,
    pub samples: usize,
    pub budget_s: f64,
    pub min_batch_ns: f64,
    pub filter: Option<String>,
    pub long: bool,
    /// Only the long rows (NT_LONG=only).
    pub only_long: bool,
    pub rows: usize,
    pub mismatches: usize,
}

impl Ctx {
    pub fn new(group: &str) -> Ctx {
        let env = |k: &str| std::env::var(k).ok();
        Ctx {
            group: group.to_string(),
            samples: env("NT_SAMPLES").and_then(|s| s.parse().ok()).unwrap_or(15),
            budget_s: env("NT_BUDGET").and_then(|s| s.parse().ok()).unwrap_or(15.0),
            min_batch_ns: env("NT_MIN_BATCH_MS").and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0) * 1e6,
            filter: env("NT_FILTER"),
            long: env("NT_LONG").is_some_and(|s| s == "1" || s == "only"),
            only_long: env("NT_LONG").is_some_and(|s| s == "only"),
            rows: 0,
            mismatches: 0,
        }
    }

    /// Whether a row of this length runs: long rows with NT_LONG=1 or only, the others unless only.
    pub fn runs(&self, long: bool) -> bool {
        if long { self.long } else { !self.only_long }
    }

    pub fn wants(&self, op: &str, params: &str) -> bool {
        match &self.filter {
            None => true,
            Some(f) => format!("{op} {params}").contains(f.as_str()),
        }
    }

    /// Times `cs[0]` (ours) against the references `cs[1..]`, sample by sample in turn, and
    /// prints one row per reference (one row with reference `none` when there is none).
    pub fn race(&mut self, r: Row, mut cs: Vec<C<'_>>) {
        if !self.wants(&r.op, &r.params) {
            return;
        }
        // NT_OURS_ONLY=1: ours alone (for a second build profile of ours).
        if std::env::var("NT_OURS_ONLY").is_ok_and(|s| s == "1") {
            cs.truncate(1);
        }
        let t_row = Instant::now();
        // Check: the same items, the same checksum.
        let mut check_note = vec![String::new(); cs.len()];
        let mut check_t: Vec<Option<f64>> = vec![None; cs.len()];
        if r.n_check > 0 {
            let mut want = 0;
            check_t[0] = Some(timed(|| want = (cs[0].f)(0, r.n_check)));
            for (i, c) in cs.iter_mut().enumerate().skip(1) {
                if !c.check {
                    continue;
                }
                let mut got = 0;
                check_t[i] = Some(timed(|| got = (c.f)(0, r.n_check)));
                if got != want {
                    check_note[i] = format!("CHECK MISMATCH ({got} vs ours {want} over {} items)", r.n_check);
                    self.mismatches += 1;
                    eprintln!("!! {} {} {}: {}", r.op, r.params, c.name, check_note[i]);
                }
            }
        }
        // Calibration (and warm-up): batches of at least min_batch_ns.
        let mut ks = vec![];
        let mut first = vec![];
        for (i, c) in cs.iter_mut().enumerate() {
            // Whole-set calls: the check's run (one call) is the warm-up.
            if r.max_k == 1 && r.n_check == 1 {
                if let Some(dt) = check_t[i] {
                    ks.push(1);
                    first.push(dt);
                    continue;
                }
            }
            let mut k = 1usize;
            loop {
                let dt = timed(|| (c.f)(0, k));
                if dt >= self.min_batch_ns || k >= r.max_k {
                    ks.push(k);
                    first.push(dt / k as f64);
                    break;
                }
                let grow = (self.min_batch_ns * 1.3 / dt.max(1.0)).ceil() as usize;
                k = (k * grow.clamp(2, 1000)).min(r.max_k);
            }
        }
        if r.same_k && cs.len() > 1 {
            let kmax = *ks.iter().max().unwrap();
            let cap = first.iter().map(|&t| ((2e9 / t.max(1.0)) as usize).max(1)).min().unwrap();
            let k = kmax.min(cap).min(r.max_k).max(1);
            ks.iter_mut().for_each(|x| *x = k);
        }
        let round: f64 = cs.iter().enumerate().map(|(i, _)| first[i] * ks[i] as f64).sum();
        let samples = ((self.budget_s * 1e9 / round.max(1.0)) as usize).clamp(3, self.samples.max(3)).min(self.samples);
        let samples = if round > 60e9 { 1 } else if round > 20e9 { samples.min(2) } else { samples };
        let mut t: Vec<Vec<f64>> = vec![vec![]; cs.len()];
        for s in 0..samples {
            let n = cs.len();
            for j in 0..n {
                let i = (j + s) % n;
                let k = ks[i];
                let i0 = s * k + 7 * s;
                let dt = timed(|| (cs[i].f)(i0, k));
                t[i].push(dt / k as f64 / r.per / cs[i].per);
            }
        }
        let stats: Vec<(f64, f64)> = t.iter_mut().map(|v| median_spread(v)).collect();
        let (ours, ours_sp) = stats[0];
        let mut base_notes = vec![];
        if !r.notes.is_empty() {
            base_notes.push(r.notes.clone());
        }
        if samples < 15 {
            base_notes.push(format!("{samples} samples"));
        }
        if cputime() {
            base_notes.push("thread CPU time (machine load 60-90)".into());
        }
        if r.same_k && cs.len() > 1 {
            base_notes.push(format!("the same {} items a sample for all", ks[0]));
        }
        let out = std::io::stdout();
        let mut out = out.lock();
        if cs.len() == 1 {
            let mut notes = base_notes.clone();
            if ours_sp > 0.1 {
                notes.push(format!("spread ours {:.0}%", ours_sp * 100.0));
            }
            writeln!(out, "nt\t{}\t{}\t{}\t{}\tnone\t-\t-\t{}", self.group, r.op, r.params, fmt(ours), notes.join("; ")).unwrap();
            self.rows += 1;
        }
        for i in 1..cs.len() {
            let (rn, rsp) = stats[i];
            let mut notes = base_notes.clone();
            if !cs[i].note.is_empty() {
                notes.insert(0, cs[i].note.clone());
            }
            if !check_note[i].is_empty() {
                notes.push(check_note[i].clone());
            } else if !cs[i].check {
                notes.push("output not compared (differs validly)".into());
            }
            if ours_sp > 0.1 || rsp > 0.1 {
                notes.push(format!("spread ours {:.0}% ref {:.0}%", ours_sp * 100.0, rsp * 100.0));
            }
            writeln!(
                out,
                "nt\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                self.group,
                r.op,
                r.params,
                fmt(ours),
                cs[i].name,
                fmt(rn),
                ratio(ours / rn),
                notes.join("; ")
            )
            .unwrap();
            self.rows += 1;
        }
        out.flush().unwrap();
        eprintln!(
            "  {:<40} {:<28} ours {:>12}  {}   [{} samples, {:.1} s]",
            r.op,
            r.params,
            fmt(ours),
            (1..cs.len()).map(|i| format!("{} {} ({})", cs[i].name, fmt(stats[i].0), ratio(ours / stats[i].0))).collect::<Vec<_>>().join("  "),
            samples,
            t_row.elapsed().as_secs_f64()
        );
    }
}

/// The median and the interquartile spread relative to it.
fn median_spread(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let med = if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) };
    if n < 4 {
        let sp = (v[n - 1] - v[0]) / med / 2.0;
        return (med, sp);
    }
    let q1 = v[n / 4];
    let q3 = v[(3 * n) / 4];
    (med, (q3 - q1) / med / 2.0)
}

/// Four significant digits (integers above 1000 in full).
pub fn fmt(x: f64) -> String {
    if !x.is_finite() {
        return "-".into();
    }
    if x >= 1000.0 {
        format!("{:.0}", x)
    } else if x >= 100.0 {
        format!("{:.1}", x)
    } else if x >= 10.0 {
        format!("{:.2}", x)
    } else if x >= 1.0 {
        format!("{:.3}", x)
    } else {
        format!("{:.4}", x)
    }
}

/// Three significant digits.
pub fn ratio(x: f64) -> String {
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

/// The processor's clock, from a chain of dependent additions (one cycle each).
pub fn ghz() -> f64 {
    let n: u64 = 50_000_000;
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let x = chain(n);
        black_box(x);
        best = best.min(t.elapsed().as_nanos() as f64);
    }
    (n * 8) as f64 / best
}

#[inline(never)]
fn chain(n: u64) -> u64 {
    let mut x: u64 = black_box(1);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // SAFETY: register arithmetic only.
        core::arch::asm!(
            "2:",
            "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
            "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            x = inout(reg) x,
            n = inout(reg) n => _,
        );
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // SAFETY: register arithmetic only.
        core::arch::asm!(
            "2:",
            "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
            "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
            "sub {n}, 1",
            "jnz 2b",
            x = inout(reg) x,
            n = inout(reg) n => _,
        );
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for _ in 0..n * 8 {
        x = black_box(x + 1);
    }
    x
}

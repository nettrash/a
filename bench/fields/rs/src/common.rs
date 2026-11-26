//! What the families share: the sides as objects, the checks of outputs, limbs and numbers.

use crate::harness::{self, Op, Side, Stat};
use a::math::numbers::{ByteOrder, Number};

/// A side of a comparison of one family: its outputs, for the check, and its timed loop.
pub trait Bench {
    fn name(&self) -> String;
    /// The output of `op` on input `i` (and `i + 1`): limbs and a flag or symbol; `None` if this
    /// side does not have the operation.
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)>;
    /// `iters` operations of the loop of `op`.
    fn run(&mut self, op: Op, iters: u64);
    /// A note for this side's rows.
    fn note(&self, _op: Op) -> String {
        String::new()
    }
}

impl Bench for crate::ffi::CRef {
    fn name(&self) -> String {
        self.name.to_string()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        crate::ffi::CRef::out(self, op, i)
    }
    fn run(&mut self, op: Op, iters: u64) {
        crate::ffi::CRef::run(self, op, iters)
    }
}

/// How two outputs are compared.
#[derive(Clone, Copy)]
pub enum Cmp {
    /// Equal limbs and equal flags.
    Exact,
    /// Square roots: equal up to the sign, modulo the prime of these limbs (for each block of
    /// `n` limbs separately when `per` is set... the whole element otherwise).
    UpToSign,
    /// Only the returned flag or symbol.
    Flag,
    /// Only the limbs.
    Limbs,
    /// Roots of `z² + z = c`: equal, or differing by 1.
    UpToOne,
}

/// `p - r` for `r` below `p`, both of `p.len()` limbs (`r = 0` stays 0).
pub fn neg_mod(r: &[u64], p: &[u64]) -> Vec<u64> {
    if r.iter().all(|&x| x == 0) {
        return r.to_vec();
    }
    let mut out = vec![0u64; p.len()];
    let mut borrow = 0u64;
    for i in 0..p.len() {
        let (d, b1) = p[i].overflowing_sub(r[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        out[i] = d;
        borrow = (b1 | b2) as u64;
    }
    out
}

/// The negation of an element of `k` coefficients of `n` limbs each.
pub fn neg_elem(r: &[u64], p: &[u64]) -> Vec<u64> {
    r.chunks(p.len()).flat_map(|c| neg_mod(c, p)).collect()
}

/// Compares the outputs of every side with ours (`sides[0]`) on `checks` inputs; the note of
/// each reference's row (empty, "checked …", or a mismatch).
pub fn check(sides: &mut [Box<dyn Bench + '_>], op: Op, checks: usize, cmp: Cmp, p: &[u64]) -> Vec<Option<String>> {
    let ours: Vec<_> = (0..checks).map(|i| sides[0].out(op, i)).collect();
    let mut notes = vec![Some(String::new())];
    for k in 1..sides.len() {
        let mut note = Some(String::new());
        for (i, o) in ours.iter().enumerate() {
            let Some(o) = o else { break };
            let Some(r) = sides[k].out(op, i) else {
                note = None;
                break;
            };
            let same = match cmp {
                Cmp::Exact => *o == r,
                Cmp::Limbs => o.0 == r.0,
                Cmp::Flag => o.1 == r.1,
                Cmp::UpToSign => o.0 == r.0 || o.0 == neg_elem(&r.0, p),
                Cmp::UpToOne => {
                    let mut z = r.0.clone();
                    z[0] ^= 1;
                    o.0 == r.0 || o.0 == z
                }
            };
            if !same {
                let what = format!("{} {} input {i}: ours {:x?} ref {:x?}", sides[k].name(), op.name(), o, r);
                note = Some(harness::mismatch(&what));
                break;
            }
        }
        notes.push(note);
    }
    notes
}

/// Checks, then measures every side that has `op` and agrees with ours, and prints the rows.
#[allow(clippy::too_many_arguments)]
pub fn compare(
    group: &str,
    opname: &str,
    params: &str,
    sides: &mut [Box<dyn Bench + '_>],
    op: Op,
    checks: usize,
    cmp: Cmp,
    p: &[u64],
) {
    if let Ok(only) = std::env::var("FIELDBENCH_OPS") {
        // a comma-separated list of operation names: the others are skipped
        if !only.split(',').any(|o| o == opname || o == op.name()) {
            return;
        }
    }
    let notes = check(sides, op, checks, cmp, p);
    let mut idx = vec![];
    let mut names = vec![];
    let mut row_notes = vec![];
    for (k, n) in notes.iter().enumerate() {
        if let Some(n) = n {
            idx.push(k);
            names.push(sides[k].name());
            let mut v = vec![sides[k].note(op)];
            if !n.is_empty() {
                v.push(n.clone());
            }
            row_notes.push(v.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("; "));
        }
    }
    let mismatched: Vec<bool> = row_notes.iter().map(|n| n.contains("MISMATCH")).collect();
    // Time only the sides that agree; a mismatch gets its row without a time.
    let timed: Vec<usize> = (0..idx.len()).filter(|&j| !mismatched[j]).collect();
    let mut taken = vec![false; sides.len()];
    for &j in &timed {
        taken[idx[j]] = true;
    }
    let mut chosen: Vec<Side> = vec![];
    for (k, s) in sides.iter_mut().enumerate() {
        if taken[k] {
            chosen.push(Side::new(s.name(), move |n| s.run(op, n)));
        }
    }
    let mut stats: Vec<Stat> = harness::measure(&mut chosen);
    drop(chosen);
    if op == Op::Batch {
        // A batch inversion's loop runs whole batches of 256 elements: the time per element.
        for s in &mut stats {
            s.median /= 256.0;
        }
    }
    let mut full = vec![Stat { median: f64::NAN, iqr: 0.0 }; idx.len()];
    for (t, &j) in timed.iter().enumerate() {
        full[j] = stats[t];
    }
    harness::report(group, opname, params, &names, &full, &row_notes);
}

/// The little-endian limbs of a nonnegative number, `n` of them.
pub fn limbs(x: &Number, n: usize) -> Vec<u64> {
    let mut b = x.value.clone();
    if matches!(x.byte_order(), ByteOrder::BigEndian) {
        b.reverse();
    }
    b.resize(8 * n.max(b.len().div_ceil(8)), 0);
    b.chunks(8).take(n).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// The number of little-endian limbs.
pub fn number(l: &[u64]) -> Number {
    let b: Vec<u8> = l.iter().flat_map(|w| w.to_le_bytes()).collect();
    Number::new(b, ByteOrder::LittleEndian)
}

/// The number of a hexadecimal string.
pub fn hex(s: &str) -> Number {
    Number::from_str_radix(s, 16, ByteOrder::LittleEndian).unwrap()
}

/// Bytes to limbs, little-endian.
pub fn le_limbs(b: &[u8], n: usize) -> Vec<u64> {
    let mut v = b.to_vec();
    v.resize(8 * n, 0);
    v.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Limbs to `len` bytes, little-endian.
pub fn limbs_le(l: &[u64], len: usize) -> Vec<u8> {
    let mut v: Vec<u8> = l.iter().flat_map(|w| w.to_le_bytes()).collect();
    v.resize(len, 0);
    v
}

/// Whether `a < b` for limbs of the same length.
pub fn lt(a: &[u64], b: &[u64]) -> bool {
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// A random number below `p` (limbs), of `p.len()` limbs.
pub fn below(rng: &mut harness::Rng, p: &[u64]) -> Vec<u64> {
    let n = p.len();
    let top = 64 - p[n - 1].leading_zeros();
    loop {
        let mut v: Vec<u64> = (0..n).map(|_| rng.next()).collect();
        v[n - 1] &= if top == 64 { u64::MAX } else { (1u64 << top) - 1 };
        if lt(&v, p) && v.iter().any(|&x| x != 0) {
            return v;
        }
    }
}

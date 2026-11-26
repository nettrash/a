//! Products of points by scalars on binary curves.
//!
//! - One scalar: the width-`w` non-adjacent form, odd digits below `2^(w − 1)` in absolute value
//!   with at least `w − 1` zeros after each, over the point's odd multiples in affine form, so that
//!   each digit costs a mixed addition; on Koblitz curves the τ-adic form of
//!   [`koblitz`](super::koblitz) takes its place.
//! - A few scalars: Straus's method, the expansions read together with one run of doublings (or
//!   of Frobenius maps) for all of them.
//! - Many scalars: Pippenger's method, the terms sorted by digit into buckets for each window.
//! - A point known beforehand: its multiples by each digit at each position of a signed fixed
//!   window, computed once, so that a product takes additions only.
//! - The `x`-coordinate alone: the Montgomery ladder of López and Dahab ("Fast multiplication on
//!   elliptic curves over GF(2ᵐ) without precomputation", CHES 1999), which keeps `x` of `j·P` and
//!   `(j + 1)·P` in projective form, `X / Z`, 6M + 5S a bit, and needs neither `y` nor `a`.

use super::arith::{Affine, Ld, Ops};
use super::narrow::Narrow;
use crate::math::field::F2m;
use crate::math::numbers::{ByteOrder, Number};

/// The limbs of the magnitude of `k`, the least significant first, without zeros at the top.
pub(super) fn magnitude(k: &Number) -> Vec<u64> {
    let mut bytes = k.value.clone();
    if k.byte_order == ByteOrder::BigEndian {
        bytes.reverse();
    }
    let mut limbs: Vec<u64> = bytes
        .chunks(8)
        .map(|c| {
            let mut b = [0; 8];
            b[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(b)
        })
        .collect();
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
    limbs
}

/// The bits of the magnitude `k`, counted in 64 bits: on 32-bit targets those of 512 MiB pass
/// `usize::MAX`.
pub(super) fn bit_length(k: &[u64]) -> u64 {
    match k.iter().rposition(|&l| l != 0) {
        Some(i) => 64 * i as u64 + 64 - u64::from(k[i].leading_zeros()),
        None => 0,
    }
}

/// Bit `i` of the magnitude `k`.
fn bit(k: &[u64], i: u64) -> bool {
    usize::try_from(i / 64)
        .ok()
        .and_then(|l| k.get(l))
        .is_some_and(|l| l >> (i % 64) & 1 == 1)
}

/// The `n` bits of `k` from bit `at`, with zeros above it.
fn bits_at(k: &[u64], at: usize, n: u32) -> u64 {
    let (l, s) = (at / 64, at % 64);
    let low = k.get(l).copied().unwrap_or(0) >> s;
    let high = if s == 0 {
        0
    } else {
        k.get(l + 1).copied().unwrap_or(0) << (64 - s)
    };
    (low | high) & ((1 << n) - 1)
}

/// The width of a non-adjacent form for a scalar of `bits` bits: the table costs `2^(w − 2)`
/// additions and the product about `bits / (w + 1)`. At most 8, which keeps the digits in an `i8`
/// and the τ-adic form within the widths whose termination the tests' script checks.
pub(super) fn window(bits: u64) -> u32 {
    match bits {
        0..=23 => 2,
        24..=75 => 3,
        76..=199 => 4,
        200..=639 => 5,
        640..=1279 => 6,
        1280..=3071 => 7,
        _ => 8,
    }
}

/// The width-`w` non-adjacent form of the magnitude `k`, the lowest digit first.
pub(super) fn wnaf(k: &[u64], w: u32) -> Vec<i8> {
    // A digit a bit, and one for the last carry: past `usize::MAX` too many to allocate.
    let len = usize::try_from(bit_length(k) + 1).unwrap_or(usize::MAX);
    let mut digits = vec![0i8; len];
    let full = 1u64 << w;
    // Whether the digits so far leave a 1 to add at `at`.
    let mut carry = 0;
    let mut at = 0;
    while at < len {
        let d = carry + bits_at(k, at, w);
        if d & 1 == 0 {
            // An even value: the carry stays where it is, a bit higher.
            at += 1;
            continue;
        }
        if d < full / 2 {
            carry = 0;
            digits[at] = d as i8;
        } else {
            carry = 1;
            digits[at] = (d as i64 - full as i64) as i8;
        }
        at += w as usize;
    }
    digits
}

/// The odd multiples `p, 3 p, …, (2 n − 1) p` in affine form, `None` for those at infinity, with
/// one inversion for them all.
pub(super) fn odd_multiples<'f>(ops: &Ops<'f>, p: &Ld<'f>, n: usize) -> Vec<Option<Affine<'f>>> {
    let mut all = Vec::with_capacity(n);
    all.push(p.clone());
    if n > 1 {
        let two = ops.double(p);
        for i in 1..n {
            let next = ops.add(&all[i - 1], &two);
            all.push(next);
        }
    }
    ops.affine_all(&all)
}

/// How a run of digits moves to the next one: a doubling, or the Frobenius map of a Koblitz
/// curve, whose powers of τ take the place of powers of 2.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
    /// `r → 2 r`.
    Double,
    /// `r → τ(r)`.
    Frobenius,
}

/// A scalar's expansion: its odd digits, the lowest first, and the points that digit `u` stands
/// for, `u` from 1 by twos, at `(u − 1) / 2`.
pub(super) struct Expansion<'f> {
    /// The digits.
    pub(super) digits: Vec<i8>,
    /// The points of the digits.
    pub(super) table: Vec<Option<Affine<'f>>>,
}

/// The expansion of `k p` in the non-adjacent form.
pub(super) fn wnaf_expansion<'f>(ops: &Ops<'f>, p: &Ld<'f>, k: &[u64]) -> Expansion<'f> {
    let w = window(bit_length(k));
    Expansion {
        digits: wnaf(k, w),
        table: odd_multiples(ops, p, 1 << (w - 2)),
    }
}

/// The sum of the expansions, read together from the top digit by Horner's rule.
pub(super) fn horner<'f>(ops: &Ops<'f>, terms: &[Expansion<'f>], step: Step) -> Ld<'f> {
    if let Some(narrow) = Narrow::new(ops) {
        return narrow.horner(ops, terms, step);
    }
    let len = terms.iter().map(|t| t.digits.len()).max().unwrap_or(0);
    let mut r = ops.identity();
    for i in (0..len).rev() {
        if !r.z.is_zero() {
            r = match step {
                Step::Double => ops.double(&r),
                Step::Frobenius => ops.frobenius(&r),
            };
        }
        for t in terms {
            let d = t.digits.get(i).copied().unwrap_or(0);
            if d != 0 {
                let entry = t.table[(d.unsigned_abs() as usize - 1) / 2].as_ref();
                r = ops.add_option(&r, entry, d < 0);
            }
        }
    }
    r
}

/// `k p` for the magnitude `k`, by the non-adjacent form.
pub(super) fn mul<'f>(ops: &Ops<'f>, p: &Ld<'f>, k: &[u64]) -> Ld<'f> {
    if bit_length(k) == 0 || p.z.is_zero() {
        return ops.identity();
    }
    if let Some(narrow) = Narrow::new(ops) {
        return narrow.mul_naf(ops, p, k);
    }
    horner(ops, &[wnaf_expansion(ops, p, k)], Step::Double)
}

/// The width of Pippenger's windows for `n` terms, `log₂ n − 1`, as each window costs an addition
/// for each term and two for each bucket.
fn bucket_bits(n: usize) -> u32 {
    (usize::BITS - n.leading_zeros()).saturating_sub(2).max(2)
}

/// The digit of the signed window of `c` bits at bit `at` of `k`, with the `carry` of the window
/// below, which it updates: digits from `−2^(c − 1)` to `2^(c − 1)`.
fn signed_digit(k: &[u64], at: usize, c: u32, carry: &mut u64) -> i64 {
    let d = bits_at(k, at, c) + *carry;
    if d > 1 << (c - 1) {
        *carry = 1;
        d as i64 - (1i64 << c)
    } else {
        *carry = 0;
        d as i64
    }
}

/// Pippenger's method for the sum of `k p` over the magnitudes `k` and points `p` (the signs
/// already applied to the points): for each window, from the top, the terms go into the bucket of
/// their digit, and the buckets are summed from the top down with a running sum, so that bucket
/// `d` counts `d` times at two additions a bucket.
pub(super) fn pippenger<'f>(ops: &Ops<'f>, terms: &[(Vec<u64>, Ld<'f>)]) -> Ld<'f> {
    let c = bucket_bits(terms.len());
    let bits = terms.iter().map(|t| bit_length(&t.0)).max().unwrap_or(0);
    let windows = usize::try_from((bits + 1).div_ceil(u64::from(c))).unwrap_or(usize::MAX);
    let points: Vec<Ld<'f>> = terms.iter().map(|t| t.1.clone()).collect();
    let table = ops.affine_all(&points);
    let mut digits = vec![0i64; windows.saturating_mul(terms.len())];
    for (t, (k, _)) in terms.iter().enumerate() {
        let mut carry = 0;
        for w in 0..windows {
            digits[w * terms.len() + t] = signed_digit(k, w * c as usize, c, &mut carry);
        }
    }
    let mut r = ops.identity();
    for w in (0..windows).rev() {
        for _ in 0..c {
            r = ops.double(&r);
        }
        let mut buckets = vec![ops.identity(); 1 << (c - 1)];
        for (t, entry) in table.iter().enumerate() {
            let d = digits[w * terms.len() + t];
            if d != 0 {
                let b = (d.unsigned_abs() - 1) as usize;
                buckets[b] = ops.add_option(&buckets[b], entry.as_ref(), d < 0);
            }
        }
        let mut running = ops.identity();
        let mut sum = ops.identity();
        for b in buckets.iter().rev() {
            running = ops.add(&running, b);
            sum = ops.add(&sum, &running);
        }
        r = ops.add(&r, &sum);
    }
    r
}

/// The multiples of a point fixed in advance, for scalars of up to `bits` bits: for each window of
/// `width` bits, the digits from 1 to `2^(width − 1)` times the window's power of two, stored as
/// words, a flag word (0 at infinity) then `x` and `y`.
#[derive(Clone, Debug)]
pub(super) struct Table {
    /// The words of an element.
    words: usize,
    /// The entries.
    entries: Vec<u64>,
    /// The bits of a window.
    width: u32,
    /// The bits of the longest scalar.
    pub(super) bits: usize,
}

/// The window of a table for scalars of `bits` bits: a product costs an addition a window, and
/// each bit of width doubles the table, 1,144 points for 571 bits at width 4. Narrower than the
/// tables of the prime-field curves, as long as the binary fields' products are slow.
fn table_width(bits: usize) -> u32 {
    match bits {
        0..=63 => 3,
        _ => 4,
    }
}

/// The table of `p` for scalars of up to `bits` bits.
pub(super) fn prepare<'f>(ops: &Ops<'f>, p: &Ld<'f>, bits: usize) -> Table {
    let width = table_width(bits);
    // `(bits + 1).div_ceil(width)`, which would wrap at `usize::MAX`; the entries saturate, as so
    // many cannot be allocated.
    let windows = bits / width as usize + 1;
    let digits = 1usize << (width - 1);
    let mut all = Vec::with_capacity(windows.saturating_mul(digits));
    let mut base = p.clone();
    for _ in 0..windows {
        let mut m = base.clone();
        all.push(m.clone());
        for _ in 1..digits {
            m = ops.add(&m, &base);
            all.push(m.clone());
        }
        // `2^(width − 1) base`, doubled.
        base = ops.double(&m);
    }
    let words = ops.f.zero().words().len();
    let size = 1 + 2 * words;
    let mut entries = vec![0; all.len() * size];
    for (a, e) in ops.affine_all(&all).iter().zip(entries.chunks_mut(size)) {
        if let Some(a) = a {
            e[0] = 1;
            e[1..=words].copy_from_slice(a.x.words());
            e[words + 1..].copy_from_slice(a.y.words());
        }
    }
    Table {
        words,
        entries,
        width,
        bits,
    }
}

/// `k p` from the table of `p`, for the magnitude `k` of at most `table.bits` bits.
pub(super) fn prepared_mul<'f>(ops: &Ops<'f>, table: &Table, k: &[u64]) -> Ld<'f> {
    let width = table.width;
    let windows = (table.bits + 1).div_ceil(width as usize);
    let digits = 1usize << (width - 1);
    let words = table.words;
    let size = 1 + 2 * words;
    let mut r = ops.identity();
    let mut carry = 0;
    for w in 0..windows {
        let d = signed_digit(k, w * width as usize, width, &mut carry);
        if d != 0 {
            let at = (w * digits + d.unsigned_abs() as usize - 1) * size;
            let e = &table.entries[at..at + size];
            if e[0] == 1 {
                let q = Affine {
                    x: ops.f.element(&e[1..=words]),
                    y: ops.f.element(&e[words + 1..]),
                };
                r = ops.add_option(&r, Some(&q), d < 0);
            }
        }
    }
    r
}

/// `x` of `k P` for the magnitude `k` and a point `P` with the coordinate `x`, on the curve or on
/// its quadratic twist, by the ladder of López and Dahab, or `None` when `k P` is the point at
/// infinity.
///
/// With `(X₁ : Z₁)` and `(X₂ : Z₂)` the points `j P` and `(j + 1) P`, whose difference is `P`, the
/// sum is `Z = (X₁·Z₂ + X₂·Z₁)²`, `X = x·Z + X₁·Z₂·X₂·Z₁`, and the double of `(X : Z)` is
/// `(X⁴ + b·Z⁴ : X²·Z²)`; both give `Z = 0` exactly for the point at infinity, so no step needs a
/// case of its own, not even `x = 0`, the point of order 2.
pub(super) fn x_ladder<'f>(ops: &Ops<'f>, x: &F2m<'f>, k: &[u64]) -> Option<F2m<'f>> {
    let bits = bit_length(k);
    if bits == 0 {
        return None;
    }
    if let Some(narrow) = Narrow::new(ops) {
        return narrow.x_ladder(ops, x, k, bits);
    }
    let double = |p: &(F2m<'f>, F2m<'f>)| {
        let x2 = p.0.square();
        let z2 = p.1.square();
        (&x2.square() + &ops.times_b(&z2.square()), &x2 * &z2)
    };
    let add = |p: &(F2m<'f>, F2m<'f>), q: &(F2m<'f>, F2m<'f>)| {
        let t = &p.0 * &q.1;
        let u = &q.0 * &p.1;
        let z = (&t + &u).square();
        (&(x * &z) + &(&t * &u), z)
    };
    let mut p1 = (x.clone(), ops.f.one());
    let mut p2 = double(&p1);
    for i in (0..bits - 1).rev() {
        if bit(k, i) {
            p1 = add(&p1, &p2);
            p2 = double(&p2);
        } else {
            p2 = add(&p1, &p2);
            p1 = double(&p1);
        }
    }
    let zi = p1.1.inverse()?;
    Some(&p1.0 * &zi)
}

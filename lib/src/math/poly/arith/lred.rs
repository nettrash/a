//! Residues modulo an `n` of `k ≥ 2` limbs: the remainder of any number of limbs by the
//! normalized `n` with the reciprocal of its top two limbs (Möller and Granlund's division of
//! three limbs by two), one quotient limb a step, without allocating for the lengths of the
//! products of residues.
//!
//! For `k = 2` the remainder stays in two registers and each step is one division of three limbs
//! by two; beyond, each step is that division of the top limbs and a product of the quotient limb
//! by the other `k - 2` limbs subtracted, with the rare correction by one more `n`.

use crate::math::numbers::nat::{self, div_3by2, invert_3by2};

/// The limbs of a normalized `n` kept inline.
const INLINE: usize = 8;

/// Up to this many limbs, the steps of the division here; beyond, the crate's division of limbs
/// (measured on an Apple M4: 20% faster at 16 limbs, 5% at 9, 7% slower at 4).
const OWN_MAX: usize = 4;

/// The limbs of the dividends shifted on the stack.
const STACK: usize = 48;

/// The normalized limbs of `n`, inline up to [`INLINE`].
#[derive(Clone, Debug)]
enum Limbs {
    Inline([u64; INLINE]),
    Heap(Vec<u64>),
}

/// The reduction modulo `n` of `k ≥ 2` limbs.
#[derive(Clone, Debug)]
pub(crate) struct LimbRed {
    /// The limbs of `n`.
    pub(crate) k: usize,
    /// The shift that normalizes `n`.
    s: u32,
    /// `n << s`, its top bit set.
    d: Limbs,
    /// The reciprocal of the top two limbs of `d`.
    v: u64,
}

/// `r -= x·y` for `x` as long as `r`: the limb that borrows out.
#[inline(always)]
fn submul_1(r: &mut [u64], x: &[u64], y: u64) -> u64 {
    let mut borrow = 0u64;
    for (o, &a) in r.iter_mut().zip(x) {
        let p = u128::from(a) * u128::from(y) + u128::from(borrow);
        let (lo, hi) = (p as u64, (p >> 64) as u64);
        let (t, b) = o.overflowing_sub(lo);
        *o = t;
        borrow = hi + u64::from(b);
    }
    borrow
}

/// `r += x` for `x` as long as `r`: the carry out.
#[inline(always)]
fn add_n(r: &mut [u64], x: &[u64]) -> bool {
    let mut carry = false;
    for (o, &a) in r.iter_mut().zip(x) {
        let (t, c1) = o.overflowing_add(a);
        let (t, c2) = t.overflowing_add(u64::from(carry));
        *o = t;
        carry = c1 || c2;
    }
    carry
}

impl LimbRed {
    /// The reduction modulo the `k ≥ 2` limbs `n`, the top one not zero.
    pub(crate) fn new(n: &[u64]) -> LimbRed {
        let k = n.len();
        debug_assert!(k >= 2 && n[k - 1] != 0);
        let s = n[k - 1].leading_zeros();
        let shifted = |i: usize| {
            if s == 0 {
                n[i]
            } else {
                (n[i] << s) | if i > 0 { n[i - 1] >> (64 - s) } else { 0 }
            }
        };
        let d = if k <= INLINE {
            let mut d = [0u64; INLINE];
            for (i, o) in d[..k].iter_mut().enumerate() {
                *o = shifted(i);
            }
            Limbs::Inline(d)
        } else {
            Limbs::Heap((0..k).map(shifted).collect())
        };
        let dl = match &d {
            Limbs::Inline(a) => &a[..k],
            Limbs::Heap(v) => &v[..],
        };
        let v = invert_3by2(dl[k - 1], dl[k - 2]);
        LimbRed { k, s, d, v }
    }

    /// The normalized limbs of `n`.
    #[inline(always)]
    fn d(&self) -> &[u64] {
        match &self.d {
            Limbs::Inline(a) => &a[..self.k],
            Limbs::Heap(v) => v,
        }
    }

    /// Sets the `k` limbs `out` to `x mod n` for the limbs `x`, any number of them.
    #[inline]
    pub(crate) fn reduce(&self, x: &[u64], out: &mut [u64]) {
        let k = self.k;
        debug_assert!(out.len() == k);
        let len = x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
        if len < k {
            out[..len].copy_from_slice(&x[..len]);
            out[len..].fill(0);
            return;
        }
        if k == 2 {
            self.reduce2(&x[..len], out);
            return;
        }
        if len < STACK {
            let mut u = [0u64; STACK];
            self.reduce_in(&x[..len], &mut u[..len + 1], out);
        } else {
            let mut u = vec![0u64; len + 1];
            self.reduce_in(&x[..len], &mut u, out);
        }
    }

    /// [`reduce`](LimbRed::reduce) for `k = 2`, `x` of at least two limbs: the shifted limbs from
    /// the top, the remainder in two registers.
    #[inline(always)]
    fn reduce2(&self, x: &[u64], out: &mut [u64]) {
        let (s, d) = (self.s, self.d());
        let (d1, d0) = (d[1], d[0]);
        let len = x.len();
        // The limb `i` of `x << s`, for `i` in `0..=len`.
        let limb = |i: usize| -> u64 {
            if s == 0 {
                if i < len { x[i] } else { 0 }
            } else {
                let hi = if i < len { x[i] << s } else { 0 };
                let lo = if i > 0 { x[i - 1] >> (64 - s) } else { 0 };
                hi | lo
            }
        };
        // The top limb is below 2^s <= d1 (0 for s = 0), so the top two are below the divisor.
        let (mut r1, mut r0) = (limb(len), limb(len - 1));
        for i in (0..len - 1).rev() {
            let (_, a, b) = div_3by2(r1, r0, limb(i), d1, d0, self.v);
            (r1, r0) = (a, b);
        }
        if s == 0 {
            out[0] = r0;
            out[1] = r1;
        } else {
            out[0] = (r0 >> s) | (r1 << (64 - s));
            out[1] = r1 >> s;
        }
    }

    /// [`reduce`](LimbRed::reduce) for `k ≥ 3` and `x` of `len ≥ k` limbs, with `u` of `len + 1`
    /// limbs to divide in.
    fn reduce_in(&self, x: &[u64], u: &mut [u64], out: &mut [u64]) {
        let (k, s, d) = (self.k, self.s, self.d());
        let len = x.len();
        debug_assert!(u.len() == len + 1 && len >= k && k >= 3);
        if s == 0 {
            u[..len].copy_from_slice(x);
            u[len] = 0;
        } else {
            u[0] = x[0] << s;
            for i in 1..len {
                u[i] = (x[i] << s) | (x[i - 1] >> (64 - s));
            }
            u[len] = x[len - 1] >> (64 - s);
        }
        if k > OWN_MAX {
            // The crate's division, whose products of limbs by a limb run in assembly.
            let mut qb = [0u64; STACK];
            let mut qv = Vec::new();
            let q: &mut [u64] = if len + 1 - k <= STACK {
                &mut qb[..len + 1 - k]
            } else {
                qv.resize(len + 1 - k, 0);
                &mut qv
            };
            nat::div_qr(q, u, d, &mut Vec::new());
        } else {
            self.steps(u, len);
        }
        // The remainder u[..k], shifted back.
        if s == 0 {
            out.copy_from_slice(&u[..k]);
        } else {
            for i in 0..k - 1 {
                out[i] = (u[i] >> s) | (u[i + 1] << (64 - s));
            }
            out[k - 1] = u[k - 1] >> s;
        }
    }

    /// The steps of the division of the `len + 1` shifted limbs `u` by `d`, the remainder left
    /// in `u[..k]`.
    #[inline(always)]
    fn steps(&self, u: &mut [u64], len: usize) {
        let (k, d) = (self.k, self.d());
        let (d1, d0) = (d[k - 1], d[k - 2]);
        // The top limb is below 2^s <= d1 (0 for s = 0), so the top k limbs are below d.
        // Each step: the quotient limb of u[i..i + k + 1] by d, from the top.
        let mut n1 = u[len];
        for i in (0..len + 1 - k).rev() {
            // u[i + k] (= n1) and u[i + k - 1], u[i + k - 2] against d1, d0.
            if n1 == d1 && u[i + k - 1] == d0 {
                // The estimate would overflow: the quotient limb is B - 1.
                let b = submul_1(&mut u[i..i + k], d, u64::MAX);
                debug_assert_eq!(b, n1);
                n1 = u[i + k - 1];
                continue;
            }
            let (q, mut r1, r0) = div_3by2(n1, u[i + k - 1], u[i + k - 2], d1, d0, self.v);
            // u[i..i + k - 2] -= q·d[..k - 2], the borrow taken from (r1, r0).
            let cy = submul_1(&mut u[i..i + k - 2], &d[..k - 2], q);
            let (t0, b0) = r0.overflowing_sub(cy);
            let (t1, b1) = r1.overflowing_sub(u64::from(b0));
            u[i + k - 2] = t0;
            r1 = t1;
            if b1 {
                // One too many: add d back.
                let c = add_n(&mut u[i..i + k - 1], &d[..k - 1]);
                r1 = r1.wrapping_add(d1).wrapping_add(u64::from(c));
            }
            n1 = r1;
        }
        u[k - 1] = n1;
    }
}

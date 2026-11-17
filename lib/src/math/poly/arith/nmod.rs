//! Residues modulo a word `n ≥ 2`: the reciprocal of the normalized `n` (Möller and Granlund),
//! which reduces a sum of up to three words with two multiplications a word, and Shoup's
//! products by a constant with its precomputed quotient.

use crate::math::numbers::nat::{div_2by1, invert_limb};

/// The arithmetic modulo a word `n ≥ 2`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WordMod {
    /// The modulus.
    pub(crate) n: u64,
    /// The leading zeros of `n`.
    pub(crate) s: u32,
    /// `n << s`, normalized: its top bit set.
    pub(crate) d: u64,
    /// The reciprocal of `d`, `floor((2^128 - 1) / d) - 2^64`.
    pub(crate) v: u64,
    /// `floor((2^64 - 1) / n)`, Barrett's constant for one word.
    pub(crate) b: u64,
}

/// The high word of `a·b`.
#[inline(always)]
pub(crate) fn mulhi(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) >> 64) as u64
}

impl WordMod {
    /// The arithmetic modulo `n ≥ 2`.
    #[inline]
    pub(crate) fn new(n: u64) -> WordMod {
        debug_assert!(n >= 2);
        let s = n.leading_zeros();
        let d = n << s;
        WordMod {
            n,
            s,
            d,
            v: invert_limb(d),
            b: u64::MAX / n,
        }
    }

    /// The bits of `n`.
    #[inline(always)]
    pub(crate) fn bits(&self) -> u32 {
        64 - self.s
    }

    /// `a + b` for residues.
    #[inline(always)]
    pub(crate) fn add(&self, a: u64, b: u64) -> u64 {
        let (s, c) = a.overflowing_add(b);
        if c || s >= self.n {
            s.wrapping_sub(self.n)
        } else {
            s
        }
    }

    /// `a - b` for residues.
    #[inline(always)]
    pub(crate) fn sub(&self, a: u64, b: u64) -> u64 {
        let (s, c) = a.overflowing_sub(b);
        if c { s.wrapping_add(self.n) } else { s }
    }

    /// `-a` for a residue.
    #[inline(always)]
    pub(crate) fn neg(&self, a: u64) -> u64 {
        if a == 0 { 0 } else { self.n - a }
    }

    /// `(u1 2^64 + u0) mod n` for `u1 < n`.
    #[inline(always)]
    pub(crate) fn reduce2(&self, u1: u64, u0: u64) -> u64 {
        debug_assert!(u1 < self.n);
        let s = self.s;
        // `u0 >> (64 - s)`, 0 for `s = 0`, without a shift by 64.
        let x1 = (u1 << s) | ((u0 >> 1) >> (63 - s));
        div_2by1(x1, u0 << s, self.d, self.v).1 >> s
    }

    /// `x mod n` for any word, by Barrett's method: the quotient `floor(x b / 2^64)` is at most 2
    /// below the true one, as `b n > 2^64 - 2n`, and two corrections without branches follow, for `n < 2^61`
    /// (so that `3n` fits a word); one correction for `n ≥ 2^63`, and the reciprocal between.
    #[inline(always)]
    pub(crate) fn reduce1(&self, x: u64) -> u64 {
        if self.n >> 63 != 0 {
            return if x >= self.n { x - self.n } else { x };
        }
        if self.n >> 61 != 0 {
            return self.reduce2(0, x);
        }
        let q = mulhi(x, self.b);
        let mut r = x.wrapping_sub(q.wrapping_mul(self.n));
        r = if r >= self.n { r - self.n } else { r };
        if r >= self.n { r - self.n } else { r }
    }

    /// `(u2 2^128 + u1 2^64 + u0) mod n` for any words.
    #[inline(always)]
    pub(crate) fn reduce3(&self, u2: u64, u1: u64, u0: u64) -> u64 {
        let u2 = self.reduce1(u2);
        let s = self.s;
        let x2 = (u2 << s) | ((u1 >> 1) >> (63 - s));
        let x1 = (u1 << s) | ((u0 >> 1) >> (63 - s));
        let r = div_2by1(x2, x1, self.d, self.v).1;
        div_2by1(r, u0 << s, self.d, self.v).1 >> s
    }

    /// `t mod n` for `t < n 2^64`.
    #[inline(always)]
    pub(crate) fn reduce_wide(&self, t: u128) -> u64 {
        self.reduce2((t >> 64) as u64, t as u64)
    }

    /// `t mod n` for any `t`.
    #[inline(always)]
    pub(crate) fn reduce_u128(&self, t: u128) -> u64 {
        let (hi, lo) = ((t >> 64) as u64, t as u64);
        if hi == 0 {
            return self.reduce1(lo);
        }
        self.reduce3(0, hi, lo)
    }

    /// `a·b` for residues.
    #[inline(always)]
    pub(crate) fn mul(&self, a: u64, b: u64) -> u64 {
        self.reduce_wide(u128::from(a) * u128::from(b))
    }

    /// Shoup's constant of the residue `w`: `floor(w 2^64 / n)`.
    #[inline(always)]
    pub(crate) fn shoup(&self, w: u64) -> u64 {
        debug_assert!(w < self.n);
        // floor(w 2^64 / n) = floor((w 2^s) 2^64 / d), a quotient below 2^64 as w < n.
        let s = self.s;
        let x1 = w << s;
        div_2by1(x1, 0, self.d, self.v).0
    }

    /// `a·w` for the residue `w` of Shoup's constant `wp`, any word `a`, for `n < 2^63`.
    #[inline(always)]
    pub(crate) fn mul_shoup(&self, a: u64, w: u64, wp: u64) -> u64 {
        debug_assert!(self.n >> 63 == 0);
        let q = mulhi(a, wp);
        let r = a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(self.n));
        if r >= self.n { r - self.n } else { r }
    }

    /// The inverse of `a`, `None` when `gcd(a, n) > 1`: the extended Euclidean algorithm on
    /// `(n, a)`, the cofactor of `a` kept modulo `n`.
    pub(crate) fn inv(&self, a: u64) -> Option<u64> {
        let (mut r0, mut r1) = (self.n, a % self.n);
        let (mut t0, mut t1) = (0u64, 1u64);
        while r1 != 0 {
            // Most quotients are 1 to 3: subtractions instead of a division then.
            let d1 = r0 - r1;
            let (q, r) = if d1 < r1 {
                (1, d1)
            } else if d1 - r1 < r1 {
                (2, d1 - r1)
            } else {
                (r0 / r1, r0 % r1)
            };
            (r0, r1) = (r1, r);
            let qt = if q == 1 {
                t1
            } else {
                self.mul(self.reduce1(q), t1)
            };
            (t0, t1) = (t1, self.sub(t0, qt));
        }
        (r0 == 1).then_some(t0)
    }
}

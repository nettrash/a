//! Word arithmetic modulo a prime `p < 2^64`: sums and differences with one comparison, products
//! reduced by the reciprocal of the normalised `p` (Möller and Granlund's division of two words by
//! one), products by a fixed multiplier with Shoup's precomputed quotient, and dot products whose
//! sums of products are kept in one, two or three words, as their length and `p` allow, and
//! reduced once (FLINT's `nmod_mat` keeps them so).

use crate::math::numbers::nat::{div_2by1, invert_limb};

/// The arithmetic modulo a prime `p ≥ 2`.
#[derive(Clone, Copy, Debug)]
pub(in crate::math::matrix) struct Field {
    /// The prime.
    pub(in crate::math::matrix) p: u64,
    /// The leading zeros of `p`.
    s: u32,
    /// `p << s`, its top bit set.
    d: u64,
    /// The reciprocal of `d`, `floor((2^128 - 1) / d) - 2^64`.
    v: u64,
}

/// How many words hold the sum of the products of a dot product before its one reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::math::matrix) enum Width {
    /// The sum stays below `2^64`.
    One,
    /// The sum stays below `2^128`.
    Two,
    /// Up to `2^64` products of any size, with a word counting the carries out of 128 bits.
    Three,
}

impl Field {
    /// The arithmetic modulo `p ≥ 2`.
    #[inline]
    pub(in crate::math::matrix) fn new(p: u64) -> Field {
        debug_assert!(p >= 2);
        let s = p.leading_zeros();
        let d = p << s;
        Field {
            p,
            s,
            d,
            v: invert_limb(d),
        }
    }

    /// `a + b` for residues.
    #[inline(always)]
    pub(in crate::math::matrix) fn add(self, a: u64, b: u64) -> u64 {
        let (s, c) = a.overflowing_add(b);
        if c || s >= self.p {
            s.wrapping_sub(self.p)
        } else {
            s
        }
    }

    /// `a - b` for residues.
    #[inline(always)]
    pub(in crate::math::matrix) fn sub(self, a: u64, b: u64) -> u64 {
        let (s, c) = a.overflowing_sub(b);
        if c { s.wrapping_add(self.p) } else { s }
    }

    /// `-a` for a residue.
    #[inline(always)]
    pub(in crate::math::matrix) fn neg(self, a: u64) -> u64 {
        if a == 0 { 0 } else { self.p - a }
    }

    /// `(hi·2^64 + lo) mod p` for `hi < p`: the remainder of the shifted value by the normalised
    /// `p`, shifted back.
    #[inline(always)]
    pub(in crate::math::matrix) fn reduce2(self, hi: u64, lo: u64) -> u64 {
        debug_assert!(hi < self.p);
        let s = self.s;
        // `lo >> (64 - s)`, 0 for `s = 0`, without a shift by 64.
        let x1 = (hi << s) | ((lo >> 1) >> (63 - s));
        div_2by1(x1, lo << s, self.d, self.v).1 >> s
    }

    /// `x mod p` for any word.
    #[inline(always)]
    pub(in crate::math::matrix) fn reduce(self, x: u64) -> u64 {
        if x < self.p { x } else { self.reduce2(0, x) }
    }

    /// `t mod p` for any `t < 2^128`.
    #[inline(always)]
    pub(in crate::math::matrix) fn reduce_wide(self, t: u128) -> u64 {
        self.reduce2(self.reduce((t >> 64) as u64), t as u64)
    }

    /// `(u2·2^128 + u1·2^64 + u0) mod p` for any words.
    #[inline(always)]
    pub(in crate::math::matrix) fn reduce3(self, u2: u64, u1: u64, u0: u64) -> u64 {
        let r = self.reduce2(self.reduce(u2), u1);
        self.reduce2(r, u0)
    }

    /// `a·b mod p` for residues.
    #[inline(always)]
    pub(in crate::math::matrix) fn mul(self, a: u64, b: u64) -> u64 {
        let t = a as u128 * b as u128;
        // `a·b < p^2`, so its high word is below `p`.
        self.reduce2((t >> 64) as u64, t as u64)
    }

    /// The inverse of the residue `a`, which is not zero, by Euclid's algorithm on words.
    #[inline]
    pub(in crate::math::matrix) fn inv(self, a: u64) -> u64 {
        debug_assert!(a != 0 && a < self.p);
        crate::math::numbers::nat::inverse_word(a, self.p).expect("a residue prime to p")
    }

    /// Shoup's quotient of the residue `w`, `floor(w·2^64 / p)`, for products by `w`.
    #[inline(always)]
    pub(in crate::math::matrix) fn shoup(self, w: u64) -> u64 {
        // `w·2^s < p·2^s = d`, so the quotient of `(w·2^s)·2^64` by `d` is one word.
        div_2by1(w << self.s, 0, self.d, self.v).0
    }

    /// `x·w mod p` for any word `x`, the residue `w` and its quotient `ws`: the quotient
    /// `floor(x·ws / 2^64)` is the true one or one below it, so the remainder is below `2p`.
    #[inline(always)]
    pub(in crate::math::matrix) fn mul_shoup(self, x: u64, w: u64, ws: u64) -> u64 {
        let q = ((x as u128 * ws as u128) >> 64) as u64;
        if self.p >> 63 == 0 {
            let r = x.wrapping_mul(w).wrapping_sub(q.wrapping_mul(self.p));
            if r >= self.p { r - self.p } else { r }
        } else {
            // `2p` passes a word: the remainder in 128 bits.
            let r = x as u128 * w as u128 - q as u128 * self.p as u128;
            let p = self.p as u128;
            (if r >= p { r - p } else { r }) as u64
        }
    }

    /// `y ← y - w·x` element by element, for residues.
    #[inline]
    pub(in crate::math::matrix) fn sub_mul(self, y: &mut [u64], x: &[u64], w: u64) {
        if w == 0 {
            return;
        }
        let ws = self.shoup(w);
        for (y, &x) in y.iter_mut().zip(x) {
            *y = self.sub(*y, self.mul_shoup(x, w, ws));
        }
    }

    /// `y ← w·y` element by element, for residues.
    #[inline]
    pub(in crate::math::matrix) fn scale(self, y: &mut [u64], w: u64) {
        let ws = self.shoup(w);
        for y in y.iter_mut() {
            *y = self.mul_shoup(*y, w, ws);
        }
    }

    /// The width that holds the sums of `len` products of residues: one word while
    /// `len·(p - 1)^2 < 2^64`, two while it is below `2^128`, and three beyond.
    #[inline]
    pub(in crate::math::matrix) fn width(self, len: usize) -> Width {
        let m = (self.p - 1) as u128 * (self.p - 1) as u128;
        match m.checked_mul(len as u128) {
            Some(t) if t <= u64::MAX as u128 => Width::One,
            Some(_) => Width::Two,
            None => Width::Three,
        }
    }

    /// `Σ a_i·b_i mod p` for residues, the sum of up to `2^64 / (p - 1)^2` products in one word.
    #[inline(always)]
    pub(in crate::math::matrix) fn dot1(self, a: &[u64], b: &[u64]) -> u64 {
        let mut s = 0u64;
        for (&x, &y) in a.iter().zip(b) {
            s += x * y;
        }
        self.reduce(s)
    }

    /// `Σ a_i·b_i mod p` for residues, the sum below `2^128`.
    #[inline(always)]
    pub(in crate::math::matrix) fn dot2(self, a: &[u64], b: &[u64]) -> u64 {
        let mut s = 0u128;
        for (&x, &y) in a.iter().zip(b) {
            s += x as u128 * y as u128;
        }
        self.reduce_wide(s)
    }

    /// `Σ a_i·b_i mod p` for residues, in three words.
    #[inline(always)]
    pub(in crate::math::matrix) fn dot3(self, a: &[u64], b: &[u64]) -> u64 {
        let (mut lo, mut hi) = (0u128, 0u64);
        for (&x, &y) in a.iter().zip(b) {
            let (t, c) = lo.overflowing_add(x as u128 * y as u128);
            lo = t;
            hi += c as u64;
        }
        self.reduce3(hi, (lo >> 64) as u64, lo as u64)
    }

    /// The residue of the integer of the sign `negative` and the magnitude `limbs`, the least
    /// significant first: Horner's rule from the top, a division of two words by one a limb.
    #[inline]
    pub(in crate::math::matrix) fn of_limbs(self, negative: bool, limbs: &[u64]) -> u64 {
        let mut r = 0;
        for &l in limbs.iter().rev() {
            r = self.reduce2(r, l);
        }
        if negative { self.neg(r) } else { r }
    }

    /// The residues of the powers `2^(64·i)` for `i < len`.
    pub(in crate::math::matrix) fn powers(self, len: usize) -> Vec<u64> {
        let base = self.reduce2(self.reduce(1), 0);
        let mut v = Vec::with_capacity(len);
        let mut x = self.reduce(1);
        for _ in 0..len {
            v.push(x);
            x = self.mul(x, base);
        }
        v
    }

    /// The residue of the integer of the sign `negative` and the magnitude `limbs`, the least
    /// significant first, with the residues `powers` of `2^(64·i)` from [`powers`](Self::powers)
    /// for at least as many limbs: `Σ limb_i·(2^(64·i) mod p)` in three words and one reduction,
    /// whose products are independent of each other, where Horner's rule is a chain of
    /// divisions.
    #[inline]
    pub(in crate::math::matrix) fn of_limbs_by(
        self,
        negative: bool,
        limbs: &[u64],
        powers: &[u64],
    ) -> u64 {
        debug_assert!(powers.len() >= limbs.len());
        let (mut lo, mut hi) = (0u128, 0u64);
        for (&l, &w) in limbs.iter().zip(powers) {
            let (t, c) = lo.overflowing_add(l as u128 * w as u128);
            lo = t;
            hi += c as u64;
        }
        let r = self.reduce3(hi, (lo >> 64) as u64, lo as u64);
        if negative { self.neg(r) } else { r }
    }

    /// The residue of the integer `x`.
    #[inline(always)]
    pub(in crate::math::matrix) fn of_i64(self, x: i64) -> u64 {
        let r = self.reduce(x.unsigned_abs());
        if x < 0 { self.neg(r) } else { r }
    }
}

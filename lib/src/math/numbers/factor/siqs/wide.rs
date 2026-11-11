//! Numbers of a fixed number `w` of limbs in two's complement, the arithmetic modulo `2^(64·w)`:
//! it gives the coefficients and the values of the sieve's polynomials exactly, as they are known
//! to lie below `2^(64·w − 1)` in absolute value, whatever the intermediate results, and an exact
//! quotient is the dividend times the inverse of the divisor.

/// The most limbs: polynomial values of up to 511 bits, for numbers of up to about 950 bits.
pub(super) const MAX_LIMBS: usize = 8;

/// A number modulo `2^(64·w)`, least significant limb first; the limbs from `w` up are zero.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(super) struct Wide(pub(super) [u64; MAX_LIMBS]);

impl Wide {
    /// The number of the limbs `x`, least significant first, modulo `2^(64·w)`.
    pub(super) fn of(x: &[u64], w: usize) -> Wide {
        let mut r = [0u64; MAX_LIMBS];
        let n = x.len().min(w);
        r[..n].copy_from_slice(&x[..n]);
        Wide(r)
    }

    /// The word `x`.
    pub(super) fn word(x: u64) -> Wide {
        let mut r = [0u64; MAX_LIMBS];
        r[0] = x;
        Wide(r)
    }

    /// `self + b`.
    #[inline]
    pub(super) fn add(&self, b: &Wide, w: usize) -> Wide {
        let mut r = [0u64; MAX_LIMBS];
        let mut c = false;
        for (i, ri) in r.iter_mut().enumerate().take(w) {
            let (s, c1) = self.0[i].overflowing_add(b.0[i]);
            let (s, c2) = s.overflowing_add(u64::from(c));
            *ri = s;
            c = c1 | c2;
        }
        Wide(r)
    }

    /// `self − b`.
    #[inline]
    pub(super) fn sub(&self, b: &Wide, w: usize) -> Wide {
        let mut r = [0u64; MAX_LIMBS];
        let mut c = false;
        for (i, ri) in r.iter_mut().enumerate().take(w) {
            let (s, c1) = self.0[i].overflowing_sub(b.0[i]);
            let (s, c2) = s.overflowing_sub(u64::from(c));
            *ri = s;
            c = c1 | c2;
        }
        Wide(r)
    }

    /// `−self`.
    #[inline]
    pub(super) fn neg(&self, w: usize) -> Wide {
        Wide::default().sub(self, w)
    }

    /// `self · b`, the low `w` limbs of the product.
    #[inline]
    pub(super) fn mul(&self, b: &Wide, w: usize) -> Wide {
        let mut r = [0u64; MAX_LIMBS];
        for i in 0..w {
            let mut c = 0u64;
            let a = u128::from(self.0[i]);
            for j in 0..w - i {
                let t = a * u128::from(b.0[j]) + u128::from(r[i + j]) + u128::from(c);
                r[i + j] = t as u64;
                c = (t >> 64) as u64;
            }
        }
        Wide(r)
    }

    /// `self · x` for a signed word `x`.
    #[inline]
    pub(super) fn mul_i64(&self, x: i64, w: usize) -> Wide {
        let m = u128::from(x.unsigned_abs());
        let mut r = [0u64; MAX_LIMBS];
        let mut c = 0u64;
        for (i, ri) in r.iter_mut().enumerate().take(w) {
            let t = u128::from(self.0[i]) * m + u128::from(c);
            *ri = t as u64;
            c = (t >> 64) as u64;
        }
        let r = Wide(r);
        if x < 0 { r.neg(w) } else { r }
    }

    /// `self / 4` for `self` a multiple of 4, as a signed number: the shift right by two bits that
    /// keeps the sign.
    #[inline]
    pub(super) fn shr2(&self, w: usize) -> Wide {
        let mut r = [0u64; MAX_LIMBS];
        let top = if self.is_negative(w) { u64::MAX } else { 0 };
        for (i, ri) in r.iter_mut().enumerate().take(w) {
            let hi = if i + 1 < w { self.0[i + 1] } else { top };
            *ri = (self.0[i] >> 2) | (hi << 62);
        }
        Wide(r)
    }

    /// Whether the top bit of the `w` limbs is set: `self` is negative as a signed number.
    #[inline]
    pub(super) fn is_negative(&self, w: usize) -> bool {
        self.0[w - 1] >> 63 == 1
    }

    /// The inverse of the odd `self` modulo `2^(64·w)`, by Newton's steps `x ← x·(2 − self·x)`, each
    /// doubling the bits that are right.
    pub(super) fn inverse(&self, w: usize) -> Wide {
        let mut x = Wide::word(super::arith::inverse_64(self.0[0]));
        let two = Wide::word(2);
        let mut good = 64;
        while good < 64 * w {
            x = x.mul(&two.sub(&self.mul(&x, w), w), w);
            good *= 2;
        }
        x
    }
}

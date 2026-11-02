//! The field of P-521's prime `2^521 - 1`, in nine limbs of 58 bits.

use super::{Field, array_of, number_of};
use crate::math::numbers::Number;

/// The bits of a limb.
const BITS: u32 = 58;

/// The low `BITS` bits.
const LOW: u64 = (1 << BITS) - 1;

/// `8 p`, with every limb at least `2^60 - 8`: `4 (2^58 - 1)` at each of the nine positions is
/// `4 (2^522 - 1)`, and 4 less at the lowest makes it `8 (2^521 - 1)`. A difference adds it, so
/// that no limb goes below zero.
const EIGHT_P: [u64; 9] = {
    let mut m = [4 * LOW; 9];
    m[0] -= 4;
    m
};

/// The field modulo `p = 2^521 - 1`, in nine limbs of 58 bits, the lowest first, which leave a
/// product's columns room to add up in 128 bits: a product is the 81 products of limbs, with no
/// carry chain, and the columns spread over the limbs at once, each in three parts, 58 bits where
/// it is, 58 one limb up and the rest two up, as `2^522 = 2` modulo `p`. Measured on an Apple M4
/// Max, a product took 71 cycles after the one before, a square 45 and a doubling of P-521 540,
/// against 98, 94 and 670 in nine limbs of 64 bits, whose carry chains went through the flags.
///
/// An element is any nine limbs below `2^60 - 8`, the value `l_0 + l_1 2^58 + ... + l_8 2^464`, not
/// always below `p`: a product gives limbs below `3 2^58 + 2^10`, and a sum or a difference carries
/// each limb's bits above 58 into the next, all at once, for limbs below `2^58 + 16`. Only
/// [`canonical`](P521::canonical) makes an element unique.
pub(in super::super) struct P521 {
    /// The prime as a number.
    p: Number,
}

impl P521 {
    /// The field modulo `p`, which is `2^521 - 1`.
    pub(in super::super) fn new(p: &Number) -> Self {
        P521 { p: p.clone() }
    }

    /// The unique element congruent to `a`, below `p`, in limbs of 58 bits but the top one's 57.
    fn canonical(&self, a: &[u64; 9]) -> [u64; 9] {
        let mut r = *a;
        // The bits from 521 up come back at the bottom, as `2^521 = 1`: fewer than `2^4` of them
        // the first time, and then at most a carry that went through every limb.
        for _ in 0..2 {
            for i in 0..8 {
                r[i + 1] += r[i] >> BITS;
                r[i] &= LOW;
            }
            r[0] += r[8] >> (BITS - 1);
            r[8] &= LOW >> 1;
        }
        // Below `2^521` now, so only `p` itself is to be taken away.
        if r[..8].iter().all(|&l| l == LOW) && r[8] == LOW >> 1 {
            return [0; 9];
        }
        r
    }
}

/// `x` in a general register, which stops the compiler from moving the limbs of a sum to vector
/// registers and back: measured on an Apple M4 Max, a sum so took 9 cycles after the one before,
/// against 11 in vector registers.
#[inline(always)]
fn pin(x: u64) -> u64 {
    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
    {
        let mut x = x;
        // SAFETY: the block is empty.
        unsafe {
            core::arch::asm!("/* {0} */", inout(reg) x, options(pure, nomem, nostack, preserves_flags));
        }
        x
    }
    #[cfg(not(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri))))]
    x
}

/// The product of two limbs.
#[inline(always)]
fn wide(a: u64, b: u64) -> u128 {
    u128::from(a) * u128::from(b)
}

/// Column `k` of `a b`, for `b2 = 2 b`: the products of the limbs whose positions add up to `k`,
/// and twice those whose positions add up to `k + 9`, as `2^522 = 2`.
#[inline(always)]
fn column(a: &[u64; 9], b: &[u64; 9], b2: &[u64; 9], k: usize) -> u128 {
    let mut s = 0;
    for i in 0..=k {
        s += wide(a[i], b[k - i]);
    }
    for i in k + 1..9 {
        s += wide(a[i], b2[k + 9 - i]);
    }
    s
}

/// Column `k` of `a^2`, for `a2 = 2 a`: each pair of different limbs once, doubled, and twice that
/// at `k + 9`.
#[inline(always)]
fn square_column(a: &[u64; 9], a2: &[u64; 9], k: usize) -> u128 {
    let mut s = 0;
    for i in 0..k.div_ceil(2) {
        s += wide(a2[i], a[k - i]);
    }
    for i in k + 1..(k + 10) / 2 {
        s += wide(a2[i], a2[k + 9 - i]);
    }
    // The square of a limb, at `k` or at `k + 9`, whichever is even.
    if k.is_multiple_of(2) {
        s += wide(a[k / 2], a[k / 2]);
    } else {
        s += wide(a[(k + 9) / 2], a2[(k + 9) / 2]);
    }
    s
}

/// The nine columns, each from a constant `k`, so that the loops of each unroll.
#[inline(always)]
fn columns(column: impl Fn(usize) -> u128) -> [u128; 9] {
    [
        column(0),
        column(1),
        column(2),
        column(3),
        column(4),
        column(5),
        column(6),
        column(7),
        column(8),
    ]
}

/// The element of the columns `c`, below `2^128` each: the low 58 bits of a column stay at its
/// limb, the next 58 go one limb up and the rest two up, and those above the ninth limb come back
/// twice at the bottom.
#[inline(always)]
fn spread(c: &[u128; 9]) -> [u64; 9] {
    let mut r = [0; 9];
    for i in 0..9 {
        let mid = (c[i] >> BITS) as u64 & LOW;
        let top = (c[i] >> (2 * BITS)) as u64;
        r[i] += c[i] as u64 & LOW;
        if i < 8 {
            r[i + 1] += mid;
        } else {
            r[0] += 2 * mid;
        }
        if i < 7 {
            r[i + 2] += top;
        } else {
            r[i - 7] += 2 * top;
        }
    }
    r
}

/// `x` with each limb's bits above 58 carried into the next, all at once.
#[inline(always)]
fn carry(x: &[u64; 9]) -> [u64; 9] {
    let mut r = [0; 9];
    r[0] = pin((x[0] & LOW) + 2 * (x[8] >> BITS));
    for i in 1..9 {
        r[i] = pin((x[i] & LOW) + (x[i - 1] >> BITS));
    }
    r
}

/// Nine limbs of 64 bits from nine of 58, the top one's 57.
fn pack(x: &[u64; 9]) -> [u64; 9] {
    let mut r = [0; 9];
    for (i, &l) in x.iter().enumerate() {
        let (at, s) = (BITS as usize * i / 64, BITS as usize * i % 64);
        r[at] |= l << s;
        if s > 64 - BITS as usize {
            r[at + 1] |= l >> (64 - s);
        }
    }
    r
}

/// Nine limbs of 58 bits from nine of 64, of a number below `2^522`.
fn unpack(x: &[u64; 9]) -> [u64; 9] {
    let mut r = [0; 9];
    for (i, l) in r.iter_mut().enumerate() {
        let (at, s) = (BITS as usize * i / 64, BITS as usize * i % 64);
        let mut v = x[at] >> s;
        if s > 64 - BITS as usize {
            v |= x[at + 1] << (64 - s);
        }
        *l = v & LOW;
    }
    r
}

impl Field for P521 {
    type E = [u64; 9];

    const CHEAP_SQUARES: bool = true;

    #[inline(always)]
    fn limbs(&self) -> usize {
        9
    }

    #[inline(always)]
    fn load(&self, x: &[u64]) -> [u64; 9] {
        let mut r = [0; 9];
        r.copy_from_slice(&x[..9]);
        r
    }

    #[inline(always)]
    fn store(&self, x: &[u64; 9], r: &mut [u64]) {
        r[..9].copy_from_slice(x);
    }

    #[inline(always)]
    fn zero(&self) -> [u64; 9] {
        [0; 9]
    }

    #[inline(always)]
    fn one(&self) -> [u64; 9] {
        let mut r = [0; 9];
        r[0] = 1;
        r
    }

    #[inline(always)]
    fn add(&self, a: &[u64; 9], b: &[u64; 9]) -> [u64; 9] {
        let mut r = [0; 9];
        for i in 0..9 {
            r[i] = pin(a[i] + b[i]);
        }
        carry(&r)
    }

    #[inline(always)]
    fn sub(&self, a: &[u64; 9], b: &[u64; 9]) -> [u64; 9] {
        let mut r = [0; 9];
        for i in 0..9 {
            r[i] = pin(a[i] + EIGHT_P[i] - b[i]);
        }
        carry(&r)
    }

    #[inline(always)]
    fn neg(&self, a: &[u64; 9]) -> [u64; 9] {
        self.sub(&[0; 9], a)
    }

    // `a`, or `a + p` when `a` is odd, its limbs below `2^60 + 2^58`, halved a limb at a time: the
    // low bit of each comes in at the top of the one below, for limbs below `3 2^58`.
    #[inline(always)]
    fn half(&self, a: &[u64; 9]) -> [u64; 9] {
        let mask = (a[0] & 1).wrapping_neg();
        let mut s = [0; 9];
        for i in 0..9 {
            let p = if i < 8 { LOW } else { LOW >> 1 };
            s[i] = pin(a[i] + (p & mask));
        }
        let mut r = [0; 9];
        for i in 0..8 {
            r[i] = pin((s[i] >> 1) + ((s[i + 1] & 1) << (BITS - 1)));
        }
        r[8] = pin(s[8] >> 1);
        r
    }

    // The columns of limbs below `2^60` are below `17 2^120`: a column has a product for each of
    // the `k + 1` pairs of limbs whose positions add up to its own `k`, and two for each of the
    // `8 - k` whose positions add up to `k + 9`.
    #[inline(always)]
    fn mul(&self, a: &[u64; 9], b: &[u64; 9]) -> [u64; 9] {
        let mut b2 = [0; 9];
        for j in 0..9 {
            b2[j] = 2 * b[j];
        }
        spread(&columns(|k| column(a, b, &b2, k)))
    }

    #[inline(always)]
    fn sqr(&self, a: &[u64; 9]) -> [u64; 9] {
        let mut a2 = [0; 9];
        for j in 0..9 {
            a2[j] = 2 * a[j];
        }
        spread(&columns(|k| square_column(a, &a2, k)))
    }

    #[inline(always)]
    fn mul_word(&self, a: &[u64; 9], w: u64, _e: &[u64; 9]) -> [u64; 9] {
        spread(&columns(|k| wide(a[k], w)))
    }

    #[inline(always)]
    fn is_zero(&self, a: &[u64; 9]) -> bool {
        self.canonical(a) == [0; 9]
    }

    fn enter(&self, x: &Number) -> [u64; 9] {
        unpack(&array_of(&x.rem_euclid(&self.p)))
    }

    fn leave(&self, x: &[u64; 9]) -> Number {
        number_of(&pack(&self.canonical(x)), self.p.byte_order)
    }

    #[inline(always)]
    fn modulus(&self) -> &Number {
        &self.p
    }
}

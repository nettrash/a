//! The remainder alone of a long dividend by a divisor of one limb: `a % b` and `rem_euclid` for a
//! `b` of at most eight bytes, the remainders of a one-limb [`Divisor`](super::super::Divisor), and
//! those of the primitive integers, read straight from the dividend's bytes.
//!
//! Dividing limb by limb waits at each step for the remainder of the one before (see
//! [`div_qr_1`](super::limbs::div_qr_1)). The remainder alone needs no quotient, and with the
//! powers `c_k = B^k mod d` it takes eight limbs a step: with `R1 B + R0` the two limbs of the
//! partial remainder so far and `u_7 … u_0` the next eight limbs,
//!
//! `(R1 B + R0) B^8 + Σ u_k B^k ≡ R1 c_9 + R0 c_8 + Σ_{k≥1} u_k c_k + u_0 (mod d)`,
//!
//! nine products and a limb, whose sum is at most `9 (d - 1)(B - 1) + B - 1`, below `B²` for
//! `d ≤ B / 9`: two limbs again, the next partial remainder. Seven of the products do not depend on
//! the partial remainder, so a step waits for one multiplication and three additions while the core
//! computes the others, as GMP's `mpn_mod_1s_4p` does with four limbs a step. A larger divisor
//! keeps a third limb, which counts the carries out of the two: with the partial remainder
//! `R2 B² + R1 B + R0`, where `R2 <= 9`, a step takes `R2 c_10` more, and its sum stays below
//! `10 B²`. The dividend is read in its own byte order, without the shifted copy in limbs that the
//! division makes.

use super::super::add::read;
use super::super::mul::Operand;
use super::limbs::{div_2by1, invert_limb};

/// The largest divisor whose partial remainders [`rem_1`] keeps in two limbs: the sum of a step,
/// `9 (d - 1)(B - 1) + B - 1`, is below `B²` for `d - 1 <= B / 9`. Larger ones take a third limb.
pub(in super::super) const REM_1_MAX: u64 = u64::MAX / 9;

/// From this many bytes in the dividend, its remainder by a limb of at most [`REM_1_MAX`] is
/// computed by [`rem_1`], the powers included: measured on an Apple M4 Max as `a %= &b`, 80 bytes
/// by 10 took 131 cycles so against 155 by the division, and by `2^60 - 59` 155 against 150; 128
/// bytes 140 and 159 against 198 and 193, and 1 KiB 234 against 1,200. Below, the division by
/// limbs costs less than the powers.
pub(in super::super) const REM_1_BYTES: usize = 80;

/// From this many bytes in the dividend, its remainder by a limb above [`REM_1_MAX`] is computed by
/// [`rem_1`], with three limbs of partial remainder: measured on an Apple M4 Max as `a %= &b` by
/// random limbs, 80 bytes took 169 cycles so against 152 by the division, 96 bytes 177 against 177,
/// 128 bytes 180 against 206, and 1 KiB 351 against 1,165.
pub(in super::super) const REM_1_BYTES_WIDE: usize = 112;

/// From this many bytes in the dividend, its remainder by a [`Divisor`](super::super::Divisor) of
/// one limb, which keeps the powers, is computed by [`rem_1`]: every dividend the registers do not
/// take. Measured on an Apple M4 Max as `a %= &d`, 65 bytes by 10 took 34 cycles so against 104 by
/// the division, and 1 KiB 166 against 1,126.
pub(in super::super) const REM_1_PREPARED: usize = 65;

/// The powers of `B` modulo a divisor `d` that [`rem_1`] takes, with `d` normalized and its
/// reciprocal for the last step.
#[derive(Clone, Copy)]
pub(in super::super) struct Powers {
    /// The divisor.
    pub(in super::super) d: u64,
    /// `c[k] = B^k mod d` for `k` from 1 to 10; `c[0]` is not used.
    c: [u64; 11],
    /// The shift that sets the top bit of `d`.
    shift: u32,
    /// `d` shifted so that its top bit is set.
    dn: u64,
    /// The reciprocal of `dn`.
    v: u64,
}

impl Powers {
    /// The powers for the divisor `d`, at least 2.
    ///
    /// `c_1` comes from the reciprocal of `d` by a multiplication, and each of the others is the
    /// remainder of a product of two before it, so that they take four steps of a division one
    /// after another, not eight: `c_2` from `c_1²`, `c_3` and `c_4` from `c_2 c_1` and `c_2²`,
    /// `c_5` to `c_8` from `c_4 c_1` to `c_4²`, and `c_9` and `c_10` from `c_8 c_1` and `c_8 c_2`.
    #[inline]
    pub(in super::super) fn of(d: u64) -> Powers {
        debug_assert!(d >= 2);
        let shift = d.leading_zeros();
        let dn = d << shift;
        let v = invert_limb(dn);
        // `t` is `c` shifted, `c << shift`, the remainder by `dn` of the shifted product: for
        // `a, b < d`, `(a << shift) b` is below `d dn`, so its high limb is below `dn`.
        let product = |t: u64, c: u64| {
            let p = u128::from(t) * u128::from(c);
            div_2by1((p >> 64) as u64, p as u64, dn, v).1
        };
        let mut t = [0u64; 11];
        // `(B << shift) mod dn` without a division: `v` is `⌊(B² - 1) / dn⌋ - B`, so the quotient
        // of `B << shift` by `dn` is `(1 << shift) + (v >> (64 - shift))`, or one less, and the
        // remainder `B << shift` less its product with `dn`, modulo `B`, is in `[0, 2 dn)`: one
        // comparison reduces it. With no shift, `d > B / 2`, and `B mod d` is `B - d`.
        t[1] = if shift == 0 {
            dn.wrapping_neg()
        } else {
            let q = (1 << shift) | (v >> (64 - shift));
            let r = dn.wrapping_mul(q).wrapping_neg();
            if r >= dn { r - dn } else { r }
        };
        let c1 = t[1] >> shift;
        t[2] = product(t[1], c1);
        t[3] = product(t[2], c1);
        t[4] = product(t[2], t[2] >> shift);
        for k in 1..=4 {
            t[4 + k] = product(t[4], t[k] >> shift);
        }
        t[9] = product(t[8], c1);
        t[10] = product(t[8], t[2] >> shift);
        let mut c = [0; 11];
        for k in 1..11 {
            c[k] = t[k] >> shift;
        }
        Powers { d, c, shift, dn, v }
    }
}

/// The two limbs of `x` as the high and the low one.
#[inline(always)]
fn split(x: u128) -> (u64, u64) {
    ((x >> 64) as u64, x as u64)
}

/// The product of two limbs as a `u128`.
#[inline(always)]
fn mul(a: u64, b: u64) -> u128 {
    u128::from(a) * u128::from(b)
}

/// The remainder of the magnitude of the operand `x` by the divisor `d` of the powers `p`.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
#[inline(never)]
pub(in super::super) unsafe fn rem_1(x: Operand, p: &Powers) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        if p.d <= REM_1_MAX {
            rem_2(x, p)
        } else {
            rem_3(x, p)
        }
    }
}

/// Limb `k` of the magnitude of `x`, the bytes `[8k, 8k + 8)` from the least significant one.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and `8k + 8` at most its significant length.
#[inline(always)]
unsafe fn limb(x: Operand, k: usize) -> u64 {
    // SAFETY: the bytes lie inside the significant bytes: little-endian the first of the vector,
    // big-endian its last.
    unsafe {
        if x.big_endian {
            u64::from_be_bytes(read(x.ptr.add(x.len - 8 * k - 8)))
        } else {
            u64::from_le_bytes(read(x.ptr.add(8 * k)))
        }
    }
}

/// The bytes of the magnitude of `x` above its whole limbs, as a number.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
#[inline(always)]
unsafe fn top(x: Operand) -> u64 {
    let s = x.significant;
    let mut top = 0u64;
    for i in (s / 8 * 8..s).rev() {
        // SAFETY: byte `i < s` from the least significant one lies inside the significant bytes.
        let b = unsafe {
            if x.big_endian {
                *x.ptr.add(x.len - 1 - i)
            } else {
                *x.ptr.add(i)
            }
        };
        top = (top << 8) | u64::from(b);
    }
    top
}

/// [`rem_1`] for `d <= REM_1_MAX`: the partial remainder in two limbs.
///
/// # Safety
///
/// As for [`rem_1`].
#[inline(always)]
unsafe fn rem_2(x: Operand, p: &Powers) -> u64 {
    let c = &p.c;
    // SAFETY: as the caller guarantees; every limb read is below `x.significant / 8`.
    let limb = |k: usize| unsafe { limb(x, k) };
    // SAFETY: as the caller guarantees.
    let (mut r1, mut r0) = (0u64, unsafe { top(x) });
    let mut k = x.significant / 8;
    if k % 8 >= 4 {
        // Four limbs, first, when the partial remainder is the top bytes alone: `R0 B^4 + Σ u_k
        // B^k ≡ R0 c_4 + Σ_{k≥1} u_k c_k + u_0`, below `B²` as the step of eight is.
        k -= 4;
        let low = mul(limb(k + 3), c[3]) + mul(limb(k + 2), c[2]) + mul(limb(k + 1), c[1]);
        (r1, r0) = split(mul(r0, c[4]) + low + u128::from(limb(k)));
    }
    while !k.is_multiple_of(8) {
        // One limb: `R B + u ≡ R1 c_2 + R0 c_1 + u`, at most `2 (d - 1)(B - 1) + B - 1`.
        k -= 1;
        (r1, r0) = split(mul(r1, c[2]) + mul(r0, c[1]) + u128::from(limb(k)));
    }
    while k > 0 {
        k -= 8;
        let mut low = u128::from(limb(k));
        for (i, &ci) in c.iter().enumerate().take(8).skip(1) {
            low += mul(limb(k + i), ci);
        }
        (r1, r0) = split(mul(r1, c[9]) + mul(r0, c[8]) + low);
    }
    // `R1 B + R0` shifted by `shift`, at least 3 as `d <= B / 9`: three limbs, the top one below
    // `1 << shift <= dn`, divided by `dn` in two steps; the remainder shifted back.
    let sh = p.shift;
    let (n2, n1, n0) = (r1 >> (64 - sh), (r1 << sh) | (r0 >> (64 - sh)), r0 << sh);
    let r = div_2by1(n2, n1, p.dn, p.v).1;
    div_2by1(r, n0, p.dn, p.v).1 >> sh
}

/// Adds `p` to the two limbs `low` and counts their carry out in `high`.
#[inline(always)]
fn add3(high: &mut u64, low: &mut u128, p: u128) {
    let (s, carry) = low.overflowing_add(p);
    *low = s;
    *high += u64::from(carry);
}

/// [`rem_1`] for `d > REM_1_MAX`: the partial remainder in three limbs, `R2 B² + R1 B + R0` with
/// `R2 <= 9`, so that each step's sum, the products of the limbs and `R2 c` more, stays below
/// `10 B²`.
///
/// # Safety
///
/// As for [`rem_1`].
#[inline(always)]
unsafe fn rem_3(x: Operand, p: &Powers) -> u64 {
    let c = &p.c;
    // SAFETY: as the caller guarantees; every limb read is below `x.significant / 8`.
    let limb = |k: usize| unsafe { limb(x, k) };
    // SAFETY: as the caller guarantees.
    let (mut r2, mut r1, mut r0) = (0u64, 0u64, unsafe { top(x) });
    let mut k = x.significant / 8;
    if k % 8 >= 4 {
        // Four limbs, first, when the partial remainder is the top bytes alone: `R0 c_4 +
        // Σ_{k≥1} u_k c_k + u_0`, below `5 B²`.
        k -= 4;
        let (mut high, mut low) = (0, u128::from(limb(k)));
        for (i, &ci) in c.iter().enumerate().take(4).skip(1) {
            add3(&mut high, &mut low, mul(limb(k + i), ci));
        }
        add3(&mut high, &mut low, mul(r0, c[4]));
        (r2, (r1, r0)) = (high, split(low));
    }
    while !k.is_multiple_of(8) {
        // One limb: `R2 c_3 + R1 c_2 + R0 c_1 + u`, below `4 B²`.
        k -= 1;
        let (mut high, mut low) = (0, u128::from(limb(k)));
        add3(&mut high, &mut low, mul(r0, c[1]));
        add3(&mut high, &mut low, mul(r1, c[2]));
        add3(&mut high, &mut low, mul(r2, c[3]));
        (r2, (r1, r0)) = (high, split(low));
    }
    while k > 0 {
        k -= 8;
        let (mut high, mut low) = (0, u128::from(limb(k)));
        for (i, &ci) in c.iter().enumerate().take(8).skip(1) {
            add3(&mut high, &mut low, mul(limb(k + i), ci));
        }
        add3(&mut high, &mut low, mul(r0, c[8]));
        add3(&mut high, &mut low, mul(r1, c[9]));
        add3(&mut high, &mut low, mul(r2, c[10]));
        (r2, (r1, r0)) = (high, split(low));
    }
    // `R2 B² + R1 B + R0` shifted by `shift`, at most 3 as `d > B / 9`: three limbs, the top one
    // below `80 <= dn`, divided by `dn` in two steps; the remainder shifted back.
    let sh = p.shift;
    let (n2, n1, n0) = if sh == 0 {
        (r2, r1, r0)
    } else {
        (
            (r2 << sh) | (r1 >> (64 - sh)),
            (r1 << sh) | (r0 >> (64 - sh)),
            r0 << sh,
        )
    };
    let r = div_2by1(n2, n1, p.dn, p.v).1;
    div_2by1(r, n0, p.dn, p.v).1 >> sh
}

/// The magnitude of the operand `y`, of at most eight significant bytes, as a limb.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and its significant bytes at most eight.
#[inline(always)]
pub(in super::super) unsafe fn limb_of(y: Operand) -> u64 {
    debug_assert!(y.significant <= 8);
    let mut d = 0u64;
    for i in (0..y.significant).rev() {
        // SAFETY: byte `i` from the least significant one lies inside the significant bytes.
        let b = unsafe {
            if y.big_endian {
                *y.ptr.add(y.len - 1 - i)
            } else {
                *y.ptr.add(i)
            }
        };
        d = (d << 8) | u64::from(b);
    }
    d
}

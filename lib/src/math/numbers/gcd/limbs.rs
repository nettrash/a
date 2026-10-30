//! The greatest common divisor of magnitudes held as limbs: native `u64`s, the least significant
//! first.
//!
//! # Reductions
//!
//! Euclid's algorithm replaces the larger of two numbers by its remainder modulo the smaller one,
//! `a - q b`. A run of such steps is a matrix: with `(A; B) = M (a; b)`, `M` is a product of the
//! matrices `(1, q; 0, 1)` (for `a - q b`) and `(1, 0; q, 1)` (for `b - q a`), of non-negative
//! entries and determinant 1, and its inverse `(m11, -m01; -m10, m00)` takes the original pair to
//! the reduced one. Lehmer's method finds the quotients of many steps from the top limbs of the
//! numbers alone, and applies their matrix to the whole numbers at once.
//!
//! [`hgcd2`] is that search, Möller's "hgcd2" ("On Schönhage's algorithm and subquadratic integer
//! gcd computation", Math. Comp. 77, 2008): from the top two limbs it reduces as long as neither
//! number falls below `2^65`, which keeps every entry of the matrix below `2^63` and below the
//! reduced numbers, so the quotients are those of the whole numbers too. The steps that hgcd2
//! cannot take, when one number is much smaller than the other or the two are close, are taken
//! one at a time by a division ([`subdiv`] and [`subdiv_ext`]).
//!
//! # Cofactors
//!
//! The extended algorithm keeps one row of `M`, `(u0, u1) = (m10, m11)`, which gives the cofactors
//! of `A`: `a = u1 A - m01 B` and `b = -u0 A + m00 B`. A step only adds a multiple of one entry
//! of the row to the other, so the row stays non-negative, and its signs are implicit: `+u1` for
//! `a`, `-u0` for `b`.
//!
//! Every step is a step of Euclid's own sequence of remainders, the last of them perhaps not
//! taken in full: hgcd2 stops before a step that would take a number below its bound, and a step
//! that subtracts the smaller number from the larger one only once leaves the larger one larger.
//! So the cofactor found at the end is Euclid's, which is the one of least magnitude: `|s| < |B| /
//! 2g` but for the few cases that GMP's `mpz_gcdext` documents, and they come out as it states
//! them. When the two numbers meet, `a = b = g`, both `+u1` and `-u0` are cofactors of `g`, and
//! Euclid's is the smaller one (`+1` when both are 1).

use super::super::div::limbs::{
    DC_DIV, bdiv_q, binvert_limb, div_qr, div_qr_1, div_qr_2, invert_3by2, invert_limb, qr_itch,
};
use super::super::mul::limbs::{add_1, add_n, cmp_n, mul, sub_1, sub_n};
use super::super::mul::{on_heap, on_stack};
use super::kernels::{addmul_1, extend_row, gcd_22, reduce_pair, small_q1, small_q2};
use core::cmp::Ordering;
use core::mem::MaybeUninit;
use core::ptr;

/// A reduction matrix of single limbs, `(u00, u01; u10, u11)`, of determinant 1, whose entries
/// are below `2^63`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Matrix1 {
    pub(super) u00: u64,
    pub(super) u01: u64,
    pub(super) u10: u64,
    pub(super) u11: u64,
}

/// The two limbs `h B + l` as a `u128`.
#[inline(always)]
fn join(h: u64, l: u64) -> u128 {
    (u128::from(h) << 64) | u128::from(l)
}

/// The high limb of `x`.
#[inline(always)]
fn high(x: u128) -> u64 {
    (x >> 64) as u64
}

/// The quotient and the remainder of `n` by `d`, for `n >= d` and a quotient below `2^63`: three
/// conditional subtractions in the kernels when the quotient is below 8, as 83 % of Euclid's
/// quotients are, and otherwise one bit of the quotient at a time from the top, as many as the
/// lengths of `n` and `d` differ by, plus one.
///
/// With `ONES`, a quotient of 1 is taken apart first, by a branch: see [`hgcd2_steps`].
#[inline(always)]
fn divmod2<const ONES: bool>(n: u128, d: u128) -> (u64, u128) {
    if ONES {
        if n < d {
            return (0, n);
        }
        let t = n - d;
        if t < d {
            return (1, t);
        }
    }
    if (n >> 3) < d && d >> 125 == 0 {
        return small_q2(n, d);
    }
    long_q2(n, d)
}

/// [`divmod2`] for a quotient of 8 or more, or a divisor of `2^125` or more: one bit of the
/// quotient at a time. Out of line, so that the steps of [`hgcd2`] keep their registers.
#[inline(never)]
fn long_q2(mut n: u128, d: u128) -> (u64, u128) {
    let k = d.leading_zeros().saturating_sub(n.leading_zeros());
    let mut d = d << k;
    let mut q = 0u64;
    for _ in 0..=k {
        let take = n >= d;
        q = (q << 1) | u64::from(take);
        n -= if take { d } else { 0 };
        d >>= 1;
    }
    (q, n)
}

/// The quotient and the remainder of `n` by `d`, for `n >= d > 0`: three conditional subtractions
/// in the kernels when the quotient is below 8 and `d` below `2^61`, the hardware's division
/// otherwise.
///
/// With `ONES`, a quotient of 1 is taken apart first, by a branch: see [`hgcd2_steps`].
#[inline(always)]
fn divmod1<const ONES: bool>(n: u64, d: u64) -> (u64, u64) {
    debug_assert!(n >= d);
    if ONES {
        let t = n - d;
        if t < d {
            return (1, t);
        }
    }
    if (n >> 3) < d && d >> 61 == 0 {
        return small_q1(n, d);
    }
    let q = n / d;
    (q, n - q * d)
}

/// The bound below which [`hgcd2`] reduces no number on two limbs: `2^65`, a high limb of 2.
const LIMIT_2: u64 = 2;

/// The bound below which [`hgcd2`] reduces no number on one limb, the top 64 bits of numbers below
/// `2^96`: `2^33`, which is `2^65` again.
const LIMIT_1: u64 = 1 << 33;

/// A column of a [`Matrix1`]: the entries that multiply one of the two numbers.
type Column = (u64, u64);

/// `c + q d` for columns, whose entries stay below `2^63`.
#[inline(always)]
fn plus_times(c: Column, q: u64, d: Column) -> Column {
    (c.0 + q * d.0, c.1 + q * d.1)
}

/// Möller's hgcd2: the matrix of Euclid's steps that the two-limb numbers `a = ah B + al` and `b =
/// bh B + bl` take while neither of them falls below `2^65`, or `None` when not even one step can
/// be taken so. The top bit of `ah` or of `bh` is set, the numbers being the top limbs of longer
/// ones shifted together, or the whole numbers.
///
/// Each step divides the larger number by the smaller, `r0 = q r1 + r`, and the pair becomes `(r1,
/// r)`, so the larger and the smaller swap places at every step, and the columns of the matrix
/// that belong to them swap with them: `r0 - q r1` adds `q` times `r0`'s column to `r1`'s. The
/// steps are taken on the two limbs while the larger number is at least `2^96`, and then on its top
/// 64 bits alone, where a step costs a quarter as much; the bound is `2^33` there. A step whose
/// remainder would fall below the bound is taken with the quotient one less, which leaves the
/// number above the smaller one, and ends the search; one that cannot be taken at all, a first
/// step that would subtract the smaller number once and fall below, is `None`.
///
/// Since the numbers stay at least `2^65` and start below `2^128`, the entries of the matrix stay
/// below `2^63`; on the whole numbers they are then below the reduced numbers' top two limbs, and
/// the low limbs, which hgcd2 does not see, change those by less than one unit of their lowest
/// limb: every step is a step of the whole numbers as well. On the top 64 bits alone, the numbers
/// are below `2^96`, so the entries that the steps there add are below `2^31`, and the 32 bits not
/// seen change the reduced numbers by less than `2^63`, again below the bound.
///
/// The quotients are found by three conditional subtractions in the kernels when they are below
/// 8, two instructions deep a bit on one limb, three on two, and otherwise by a division. Measured
/// on an Apple M4 Max over 4096 random pairs, a call took 560 cycles so, against 730 with the
/// steps alternating by a branch on whether the quotient is 1 and quotients from a division, and
/// 820 for GMP 6.3's `mpn_hgcd2`, whose matrices are the same.
pub(super) fn hgcd2(ah: u64, al: u64, bh: u64, bl: u64) -> Option<Matrix1> {
    hgcd2_steps::<false>(ah, al, bh, bl).map(|(m, _)| m)
}

/// From this many steps of one call of [`hgcd2_steps`] on, the next call takes quotients of 1
/// apart: random numbers take about 37 steps to reduce 63 bits, as Euclid's steps reduce numbers
/// by 1.71 bits each on average, and numbers whose quotients are all 1, such as consecutive
/// Fibonacci numbers, about 91.
pub(super) const ONES_STEPS: u32 = 64;

/// [`hgcd2`], and the number of steps its matrix holds.
///
/// With `ONES`, each step first takes a quotient of 1 apart by a branch, as GMP's `mpn_hgcd2`
/// does: that branch is as good as random on random numbers, where 41 % of Euclid's quotients
/// are 1, but it is always taken on numbers whose quotients are all 1, the worst case of Euclid's
/// algorithm, and then a step is a subtraction and a comparison. Lehmer's method takes that
/// variant after a call of at least [`ONES_STEPS`] steps. Measured on an Apple M4 Max, consecutive
/// Fibonacci numbers of 1 KiB took 132,000 cycles for their cofactors so, against 204,000 with
/// the conditional subtractions alone and 123,000 for GMP 6.3's `mpz_gcdext`, and their greatest
/// common divisor 88,000 against 114,000 and 104,000; random numbers took as long as before.
#[inline(always)]
pub(super) fn hgcd2_steps<const ONES: bool>(
    ah: u64,
    al: u64,
    bh: u64,
    bl: u64,
) -> Option<(Matrix1, u32)> {
    if ah < LIMIT_2 || bh < LIMIT_2 {
        return None;
    }
    let (a, b) = (join(ah, al), join(bh, bl));
    // The larger `r0` and the smaller `r1`, with their columns `c0` and `c1`, and whether `r0` is
    // `a`, whose column is the first.
    let (mut r0, mut r1, mut r0_is_a) = if a > b { (a, b, true) } else { (b, a, false) };
    let (mut c0, mut c1): (Column, Column) = if r0_is_a {
        ((1, 0), (0, 1))
    } else {
        ((0, 1), (1, 0))
    };
    let first = r0 - r1;
    if high(first) < LIMIT_2 {
        return None;
    }
    let (q, r) = divmod2::<ONES>(first, r1);
    if high(r) < LIMIT_2 {
        return Some((matrix1(c0, plus_times(c1, q, c0), r0_is_a), 1));
    }
    (c0, c1) = (plus_times(c1, q + 1, c0), c0);
    (r0, r1) = (r1, r);
    r0_is_a = !r0_is_a;
    let mut steps = 1;
    while high(r0) >= 1 << 32 {
        let (q, r) = divmod2::<ONES>(r0, r1);
        if high(r) < LIMIT_2 {
            return Some((matrix1(c0, plus_times(c1, q - 1, c0), r0_is_a), steps + 1));
        }
        (c0, c1) = (plus_times(c1, q, c0), c0);
        (r0, r1) = (r1, r);
        r0_is_a = !r0_is_a;
        steps += 1;
    }
    let (mut s0, mut s1) = ((r0 >> 32) as u64, (r1 >> 32) as u64);
    loop {
        let (q, r) = divmod1::<ONES>(s0, s1);
        if r < LIMIT_1 {
            return Some((matrix1(c0, plus_times(c1, q - 1, c0), r0_is_a), steps + 1));
        }
        (c0, c1) = (plus_times(c1, q, c0), c0);
        (s0, s1) = (s1, r);
        r0_is_a = !r0_is_a;
        steps += 1;
    }
}

/// [`hgcd2`] for Lehmer's method: the variant that takes quotients of 1 apart when `ones` says
/// that the last call took at least [`ONES_STEPS`] steps, and `ones` set for the next call.
#[inline(always)]
pub(super) fn hgcd2_hinted(ah: u64, al: u64, bh: u64, bl: u64, ones: &mut bool) -> Option<Matrix1> {
    let found = if *ones {
        hgcd2_steps::<true>(ah, al, bh, bl)
    } else {
        hgcd2_steps::<false>(ah, al, bh, bl)
    };
    found.map(|(m, steps)| {
        *ones = steps >= ONES_STEPS;
        m
    })
}

/// The matrix of the columns `c0` of the larger number and `c1` of the smaller, the larger being
/// `a` when `r0_is_a`.
#[inline(always)]
fn matrix1(c0: Column, c1: Column, r0_is_a: bool) -> Matrix1 {
    let (ca, cb) = if r0_is_a { (c0, c1) } else { (c1, c0) };
    Matrix1 {
        u00: ca.0,
        u01: cb.0,
        u10: ca.1,
        u11: cb.1,
    }
}

// Binary steps on one and two limbs

/// The greatest common divisor of the odd limbs `u` and `v`, by the binary method: the larger is
/// replaced by the difference, whose zero bits at the low end are shifted out, until the two meet.
/// The difference, the smaller one and the shift are found together, without branches.
#[inline]
pub(super) fn gcd_11(mut u: u64, mut v: u64) -> u64 {
    debug_assert!(u & v & 1 == 1);
    while u != v {
        let (d, below) = u.overflowing_sub(v);
        let zeros = d.trailing_zeros();
        if below {
            v = u;
        }
        u = if below { d.wrapping_neg() } else { d } >> zeros;
    }
    u
}

/// The greatest common divisor of `x` and `y`, of up to two limbs each, either of which may be
/// zero: the zero bits that both have at the low end are set apart, the rest shifted out of each,
/// and the odd numbers left go to [`gcd_22`] or [`gcd_11`].
#[inline(always)]
pub(super) fn gcd_u128(x: u128, y: u128) -> u128 {
    if (x | y) >> 64 == 0 {
        return u128::from(gcd_u64(x as u64, y as u64));
    }
    gcd_wide(x, y)
}

/// [`gcd_u128`] of one limb each. When one is longer than the other by 16 bits or more, as GMP's
/// `mpn_gcd_1` does, it is first replaced by its remainder modulo the other: the binary method
/// takes off about a bit a step, the division all of the difference at once.
#[inline(always)]
pub(super) fn gcd_u64(x: u64, y: u64) -> u64 {
    if x == 0 || y == 0 {
        return x | y;
    }
    let k = (x | y).trailing_zeros();
    let (mut x, mut y) = (x >> x.trailing_zeros(), y >> y.trailing_zeros());
    if x < y {
        (x, y) = (y, x);
    }
    if x >> 16 > y {
        x %= y;
        if x == 0 {
            return y << k;
        }
        x >>= x.trailing_zeros();
    }
    gcd_11(x, y) << k
}

/// [`gcd_u128`] of numbers of which one has two limbs. Out of line, so that the one-limb path
/// keeps its registers.
#[inline(never)]
fn gcd_wide(x: u128, y: u128) -> u128 {
    if x == 0 || y == 0 {
        return x | y;
    }
    let k = (x | y).trailing_zeros();
    let (x, y) = (x >> x.trailing_zeros(), y >> y.trailing_zeros());
    let g = if (x | y) >> 64 == 0 {
        u128::from(gcd_11(x as u64, y as u64))
    } else {
        gcd_22(x, y)
    };
    g << k
}

/// The greatest common divisor `g` of the limbs `a` and `b`, not zero, and the cofactors `u`, `v`
/// of Euclid's algorithm, `g = u a + v b`, with `|u| <= b / 2g` and `|v| <= a / 2g`. As in
/// [`hgcd2`], each step divides the larger number by the smaller, the pair moving on to the smaller
/// and the remainder with their cofactors, and the quotient takes three conditional subtractions
/// when it is below 8. Every cofactor but the last that the algorithm computes is below `2^63`, as
/// `|s_i| r_(i-1) + |s_(i-1)| r_i` is the other number for consecutive remainders `r`, and the
/// last one, of a zero remainder, is not computed.
#[inline]
pub(super) fn gcdext_1(a: u64, b: u64) -> (u64, i64, i64) {
    debug_assert!(a != 0 && b != 0);
    if a < b {
        let (g, v, u) = gcdext_1(b, a);
        return (g, u, v);
    }
    // r0 = u0 a + v0 b and r1 = u1 a + v1 b.
    let (mut r0, mut r1) = (a, b);
    let (mut u0, mut v0, mut u1, mut v1) = (1i64, 0i64, 0i64, 1i64);
    loop {
        let (q, r) = divmod1::<false>(r0, r1);
        if r == 0 {
            return (r1, u1, v1);
        }
        // `r1 >= 2` here, so `q < 2^63`.
        let q = q as i64;
        (u0, u1) = (u1, u0 - q * u1);
        (v0, v1) = (v1, v0 - q * v1);
        (r0, r1) = (r1, r);
    }
}

// Limbs

/// The number of limbs at `p`, of `n`, without the zero limbs at the top.
///
/// # Safety
///
/// The `n` limbs must be valid for reads.
#[inline(always)]
pub(super) unsafe fn normalized(p: *const u64, mut n: usize) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        while n > 0 && *p.add(n - 1) == 0 {
            n -= 1;
        }
    }
    n
}

/// Sets the `n` limbs at `r` to the `n` limbs at `x` shifted left by `s` bits, `s < 64`, and
/// returns the bits shifted out of the top. `r` may be `x`.
///
/// # Safety
///
/// The limbs must be valid; `r` is `x` or apart from it.
#[inline]
pub(super) unsafe fn lshift(r: *mut u64, x: *const u64, n: usize, s: u32) -> u64 {
    // SAFETY: as the caller guarantees; limb `i - 1` of `x` is read before limb `i - 1` of `r` is
    // written.
    unsafe {
        if s == 0 {
            if r.cast_const() != x {
                ptr::copy_nonoverlapping(x, r, n);
            }
            return 0;
        }
        let out = *x.add(n - 1) >> (64 - s);
        for i in (1..n).rev() {
            *r.add(i) = (*x.add(i) << s) | (*x.add(i - 1) >> (64 - s));
        }
        *r = *x << s;
        out
    }
}

/// Sets the `n` limbs at `r` to the `n` limbs at `x` shifted right by `s` bits, `s < 64`. `r` may
/// be `x`.
///
/// # Safety
///
/// The limbs must be valid; `r` is `x` or apart from it.
#[inline]
pub(super) unsafe fn rshift(r: *mut u64, x: *const u64, n: usize, s: u32) {
    // SAFETY: as the caller guarantees; limb `i + 1` of `x` is read before limb `i + 1` of `r` is
    // written.
    unsafe {
        if s == 0 {
            if r.cast_const() != x {
                ptr::copy_nonoverlapping(x, r, n);
            }
            return;
        }
        for i in 0..n - 1 {
            *r.add(i) = (*x.add(i) >> s) | (*x.add(i + 1) << (64 - s));
        }
        *r.add(n - 1) = *x.add(n - 1) >> s;
    }
}

// Division

/// The limbs of scratch space that [`tdiv_qr`] needs for a dividend of `nn` limbs and a divisor of
/// `dn`.
pub(super) fn tdiv_itch(nn: usize, dn: usize) -> usize {
    if dn <= 2 {
        nn + 1
    } else {
        1 + dn + (nn + 1) + qr_itch(nn + 1, dn).max(qr_itch(nn, dn))
    }
}

/// Sets the `nn - dn + 1` limbs at `q` to the quotient of the `nn` limbs at `n` by the `dn` limbs
/// at `d`, whose top limb is not zero, and the low `dn` limbs of `n` to the remainder: the
/// division's own steps (see [`div_qr`]) on copies of both shifted so that the divisor's top bit
/// is set, and the remainder shifted back.
///
/// # Safety
///
/// `nn >= dn >= 1`; the limbs at `n` must be valid for reads and writes, those at `d` for reads,
/// those at `q` for writes and the [`tdiv_itch`] limbs at `t` for reads and writes, all apart.
pub(super) unsafe fn tdiv_qr(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    t: *mut u64,
) {
    debug_assert!(nn >= dn && dn >= 1);
    // SAFETY: as the caller guarantees: the shifted dividend takes `nn + 1` limbs of the scratch,
    // after the shifted divisor of `dn` limbs and one to start it on 16 bytes for the longer ones.
    unsafe {
        let s = (*d.add(dn - 1)).leading_zeros();
        match dn {
            1 => {
                let d0 = *d << s;
                let v = invert_limb(d0);
                let top = lshift(t, n, nn, s);
                *n = div_qr_1::<true>(q, t, nn, top, d0, v) >> s;
            }
            2 => {
                let (d1, d0) = (*d.add(1), *d);
                let (d1, d0) = if s == 0 {
                    (d1, d0)
                } else {
                    ((d1 << s) | (d0 >> (64 - s)), d0 << s)
                };
                let v = invert_3by2(d1, d0);
                *t.add(nn) = lshift(t, n, nn, s);
                let (r1, r0) = div_qr_2(q, t, nn + 1, d1, d0, v);
                *n = if s == 0 {
                    r0
                } else {
                    (r0 >> s) | (r1 << (64 - s))
                };
                *n.add(1) = r1 >> s;
            }
            _ => {
                let dd = t.add((t.addr() / 8) & 1);
                lshift(dd, d, dn, s);
                let u = dd.add(dn);
                *u.add(nn) = lshift(u, n, nn, s);
                let v = invert_3by2(*dd.add(dn - 1), *dd.add(dn - 2));
                let w = u.add(nn + 1);
                // The limb above the dividend takes part only when the shift put bits in it, as in
                // the division's own general path.
                let wide = *u.add(nn) != 0;
                let un = if wide { nn + 1 } else { nn };
                let qh = div_qr(q, u, un, dd, dn, v, w);
                if wide {
                    debug_assert!(!qh);
                } else {
                    *q.add(nn - dn) = u64::from(qh);
                }
                rshift(n, u, dn, s);
            }
        }
    }
}

/// Runs `f` on `need` limbs of scratch: the `avail` limbs at `t` when they are enough, and
/// otherwise a buffer on the stack or the heap, for the rare divisions that need more than a
/// caller planned for.
///
/// # Safety
///
/// The `avail` limbs at `t` must be valid for reads and writes.
#[inline(always)]
pub(super) unsafe fn with_scratch<R>(
    t: *mut u64,
    avail: usize,
    need: usize,
    f: impl FnOnce(*mut u64) -> R,
) -> R {
    if need <= avail {
        f(t)
    } else {
        more_scratch(need, f)
    }
}

/// The scratch of [`with_scratch`] when the caller's is not enough. Out of line and cold.
#[cold]
#[inline(never)]
fn more_scratch<R>(need: usize, f: impl FnOnce(*mut u64) -> R) -> R {
    let run = |s: *mut MaybeUninit<u64>| f(s.cast());
    if need <= super::super::mul::STACK_LIMBS {
        on_stack(run)
    } else {
        on_heap(need, run)
    }
}

/// A bound on the limbs of scratch space that [`mul`] needs for operands of at most `x` and `y`
/// limbs, in either order: [`mul_itch`](super::super::mul::limbs::mul_itch) is not monotonic, a
/// shorter operand can take another method that needs more (176 limbs by 43 need 162, by 45 only
/// 136), and the lengths that the scratch is planned for are bounds of the lengths multiplied. Computed over every pair of
/// lengths up to 40,000 limbs, `mul_itch` is at most 2.07 times their sum.
pub(super) fn mul_bound(x: usize, y: usize) -> usize {
    3 * (x + y) + 64
}

/// The limbs of scratch space that [`divexact`] needs for a dividend of `nn` limbs and a divisor
/// of `dn`.
pub(super) fn divexact_itch(nn: usize, dn: usize) -> usize {
    (nn + 1) + (dn + 1) + tdiv_itch(nn, dn)
}

/// From this many limb products of Hensel's method, an exact division by a divisor long enough for
/// divide and conquer is left to the ordinary division, as [`Number::div_exact`] leaves it.
///
/// [`Number::div_exact`]: super::super::Number::div_exact
const HENSEL_LIMIT: usize = 1 << 18;

/// Sets the `nn - dn + 1` limbs at `q` to the quotient of the `nn` limbs at `n` by the `dn` limbs
/// at `d`, whose top limb is not zero, when the division is exact: by Hensel's method on copies of
/// both shifted right by the divisor's zero bits at the low end, or by the ordinary division where
/// divide and conquer is faster, as `div_exact` chooses.
///
/// # Safety
///
/// `nn >= dn >= 1`, the division exact; the limbs at `n` and `d` must be valid for reads, those at
/// `q` for writes, and the `avail` limbs at `t`, at least [`divexact_itch`] of them, for reads and
/// writes, all apart.
pub(super) unsafe fn divexact(
    q: *mut u64,
    n: *const u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    t: *mut u64,
    avail: usize,
) {
    let qn = nn - dn + 1;
    let m = qn.min(dn);
    // SAFETY: as the caller guarantees: the copies take `nn + 1` and `dn + 1` limbs of the scratch,
    // and the ordinary division's scratch comes after them.
    unsafe {
        let (nc, dc) = (t, t.add(nn + 1));
        if dn >= DC_DIV && qn * m - m * m / 2 >= HENSEL_LIMIT {
            ptr::copy_nonoverlapping(n, nc, nn);
            let room = avail.saturating_sub(nn + 1 + dn + 1);
            with_scratch(dc, room, tdiv_itch(nn, dn), |w| {
                tdiv_qr(q, nc, nn, d, dn, w)
            });
            return;
        }
        let mut skip = 0;
        while *d.add(skip) == 0 {
            skip += 1;
        }
        let bits = (*d.add(skip)).trailing_zeros();
        // The quotient's limbs from the low end: `qn` of the dividend and as many of the divisor,
        // from the limbs both have past the divisor's zero bits.
        let take = |p: *const u64, len: usize, out: *mut u64, want: usize| {
            let have = len - skip;
            let k = have.min(want + 1);
            ptr::copy_nonoverlapping(p.add(skip), out, k);
            if k < want + 1 {
                ptr::write_bytes(out.add(k), 0, want + 1 - k);
            }
            rshift(out, out, want + 1, bits);
        };
        take(n, nn, nc, qn);
        let dl = (dn - skip).min(qn);
        take(d, dn, dc, dl);
        bdiv_q(q, nc, qn, dc, dl, binvert_limb(*dc));
    }
}

// Euclid's steps, one at a time

/// What a step of [`subdiv`] or [`subdiv_ext`] leaves.
pub(super) enum Step {
    /// The numbers, reduced, of the given number of limbs.
    Size(usize),
    /// The greatest common divisor is found: the limbs at the pointer, of the given number.
    Gcd(*const u64, usize),
}

/// One step of Euclid's algorithm on the numbers of `n` limbs at `a` and `b`, not both zero, for
/// the greatest common divisor alone, when [`hgcd2`] makes no progress: the larger is replaced by
/// its remainder modulo the smaller. The two pointers may be swapped, and the numbers left are of
/// the smaller one's limbs, above which the limbs are left as they are.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes and apart, and the `avail` limbs
/// at `t` for reads and writes, apart from them.
pub(super) unsafe fn subdiv(
    a: &mut *mut u64,
    b: &mut *mut u64,
    n: usize,
    t: *mut u64,
    avail: usize,
) -> Step {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut an = normalized(*a, n);
        let mut bn = normalized(*b, n);
        // The smaller one in `a`.
        let swap = match an.cmp(&bn) {
            Ordering::Equal => match cmp_n(*a, *b, an) {
                Ordering::Equal => return Step::Gcd(*a, an),
                order => order == Ordering::Greater,
            },
            order => order == Ordering::Greater,
        };
        if swap {
            core::mem::swap(a, b);
            core::mem::swap(&mut an, &mut bn);
        }
        if an == 0 {
            return Step::Gcd(*b, bn);
        }
        let (a, b) = (*a, *b);
        let need = (bn - an + 1) + tdiv_itch(bn, an);
        with_scratch(t, avail, need, |t| {
            tdiv_qr(t, b, bn, a, an, t.add(bn - an + 1))
        });
        if normalized(b, an) == 0 {
            return Step::Gcd(a, an);
        }
        Step::Size(an)
    }
}

/// The top two limbs of the numbers of `n >= 2` limbs at `a` and `b`, shifted together so that
/// the top bit of one of them is set: with the bits from limb `n - 3`, or zeros for two limbs.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads, and not both zero at the top.
#[inline(always)]
pub(super) unsafe fn top_limbs(a: *const u64, b: *const u64, n: usize) -> (u64, u64, u64, u64) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let (a1, a0, b1, b0) = (*a.add(n - 1), *a.add(n - 2), *b.add(n - 1), *b.add(n - 2));
        let s = (a1 | b1).leading_zeros();
        if s == 0 {
            return (a1, a0, b1, b0);
        }
        let (a_, b_) = if n > 2 {
            (*a.add(n - 3), *b.add(n - 3))
        } else {
            (0, 0)
        };
        (
            (a1 << s) | (a0 >> (64 - s)),
            (a0 << s) | (a_ >> (64 - s)),
            (b1 << s) | (b0 >> (64 - s)),
            (b0 << s) | (b_ >> (64 - s)),
        )
    }
}

/// The greatest common divisor of the numbers of `n` limbs at `a` and `b`, not both zero at the
/// top, by Lehmer's method: [`hgcd2`]'s matrices applied to the whole numbers in place, the steps
/// it cannot take by [`subdiv`], down to two limbs, which the binary method finishes. Writes the
/// greatest common divisor at `g` and returns its number of limbs; the numbers are clobbered.
///
/// # Safety
///
/// The `n` limbs at `a` and at `b` must be valid for reads and writes, those at `g` for writes,
/// and the `avail` limbs at `t` for reads and writes, all apart.
pub(super) unsafe fn gcd_lehmer(
    g: *mut u64,
    mut a: *mut u64,
    mut b: *mut u64,
    mut n: usize,
    t: *mut u64,
    avail: usize,
) -> usize {
    // SAFETY: as the caller guarantees; the numbers keep at least `n` limbs, of which the top ones
    // are not both zero.
    unsafe {
        // Whether the last call of hgcd2 took so many steps that the quotients are mostly 1.
        let mut ones = false;
        while n > 2 {
            let (ah, al, bh, bl) = top_limbs(a, b, n);
            if let Some(m) = hgcd2_hinted(ah, al, bh, bl, &mut ones) {
                reduce_pair(a, b, n, m);
                n -= usize::from(*a.add(n - 1) | *b.add(n - 1) == 0);
            } else {
                match subdiv(&mut a, &mut b, n, t, avail) {
                    Step::Size(k) => n = k,
                    Step::Gcd(p, k) => {
                        ptr::copy(p, g, k);
                        return k;
                    }
                }
            }
        }
        let x = if n == 2 {
            join(*a.add(1), *a)
        } else {
            u128::from(*a)
        };
        let y = if n == 2 {
            join(*b.add(1), *b)
        } else {
            u128::from(*b)
        };
        let r = gcd_u128(x, y);
        *g = r as u64;
        if high(r) != 0 {
            *g.add(1) = high(r);
            2
        } else {
            1
        }
    }
}

/// The remainder of the `n` limbs at `a` by the limb `d`, not zero.
///
/// # Safety
///
/// `n >= 1`; the limbs must be valid for reads, and the `n` limbs at `t` for writes.
pub(super) unsafe fn rem_1(a: *const u64, n: usize, d: u64, t: *mut u64) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        if n == 1 {
            return *a % d;
        }
        let s = d.leading_zeros();
        let d0 = d << s;
        let v = invert_limb(d0);
        let top = lshift(t, a, n, s);
        div_qr_1::<false>(t, t, n, top, d0, v) >> s
    }
}

// The extended algorithm

/// The row of the reduction matrix that the extended algorithm keeps: `(u0, u1)`, of `un` limbs
/// each and room for more, the cofactors `+u1` of `a` and `-u0` of `b`.
pub(super) struct Row {
    /// The entry `m10`, the cofactor of `b` with its sign turned over.
    pub(super) u0: *mut u64,
    /// The entry `m11`, the cofactor of `a`.
    pub(super) u1: *mut u64,
    /// The limbs both entries are taken to have; zero above them.
    pub(super) un: usize,
}

/// A cofactor found: its limbs, at the pointer, of the given number, and whether it is negative.
#[derive(Clone, Copy)]
pub(super) struct Cofactor {
    pub(super) p: *const u64,
    pub(super) n: usize,
    pub(super) negative: bool,
}

impl Row {
    /// The cofactor of `a`, `+u1`.
    ///
    /// # Safety
    ///
    /// The row's limbs must be valid for reads.
    #[inline(always)]
    unsafe fn of_a(&self) -> Cofactor {
        // SAFETY: as the caller guarantees.
        let n = unsafe { normalized(self.u1, self.un) };
        Cofactor {
            p: self.u1,
            n,
            negative: false,
        }
    }

    /// The cofactor of `b`, `-u0`.
    ///
    /// # Safety
    ///
    /// The row's limbs must be valid for reads.
    #[inline(always)]
    unsafe fn of_b(&self) -> Cofactor {
        // SAFETY: as the caller guarantees.
        let n = unsafe { normalized(self.u0, self.un) };
        Cofactor {
            p: self.u0,
            n,
            negative: n != 0,
        }
    }

    /// The cofactor of `g` when `a = b = g`: `+u1` or `-u0`, whichever is the smaller, `+u1` when
    /// they are equal, which happens only when both are 1.
    ///
    /// # Safety
    ///
    /// The row's limbs must be valid for reads.
    #[inline(always)]
    pub(super) unsafe fn smaller(&self) -> Cofactor {
        // SAFETY: as the caller guarantees.
        unsafe {
            if cmp_n(self.u0, self.u1, self.un) == Ordering::Less {
                self.of_b()
            } else {
                self.of_a()
            }
        }
    }

    /// Adds `q` times one entry to the other, `u0 += q u1` when `to_u0` and `u1 += q u0`
    /// otherwise, for a quotient `q` of `qn` limbs, the top one not zero, with the scratch at `t`
    /// for the product of more than one limb, and grows the row's length as the sum needs.
    ///
    /// # Safety
    ///
    /// The row's limbs must be valid for reads and writes up to the length the sum takes and one
    /// more, and zero above the row's length; the [`add_times_itch`] limbs at `t` for writes.
    unsafe fn add_times(&mut self, to_u0: bool, q: *const u64, qn: usize, t: *mut u64) {
        let (to, from) = if to_u0 {
            (self.u0, self.u1.cast_const())
        } else {
            (self.u1, self.u0.cast_const())
        };
        // SAFETY: as the caller guarantees; the other entry is zero above the row's length, so a
        // sum that grows the length leaves the row zero above it.
        unsafe {
            let un = self.un;
            if qn == 1 {
                let c = if *q == 1 {
                    u64::from(add_n(to, to, from, un))
                } else {
                    addmul_1(to, from, un, *q)
                };
                *to.add(un) = c;
                self.un = un + usize::from(c != 0);
                return;
            }
            let fnorm = normalized(from, un);
            if fnorm == 0 {
                return;
            }
            let (long, short) = if qn >= fnorm {
                ((q, qn), (from, fnorm))
            } else {
                ((from, fnorm), (q, qn))
            };
            let mut pn = fnorm + qn;
            mul(t, long.0, long.1, short.0, short.1, t.add(pn));
            pn -= usize::from(*t.add(pn - 1) == 0);
            if pn >= un {
                let c = add_to(t, t, pn, to, un);
                ptr::copy_nonoverlapping(t, to, pn);
                *to.add(pn) = c;
                self.un = pn + usize::from(c != 0);
            } else {
                let c = add_to(to, to, un, t, pn);
                *to.add(un) = c;
                self.un = un + usize::from(c != 0);
            }
        }
    }
}

/// The limbs of scratch space that [`Row::add_times`] needs for a quotient of `qn` limbs and a row
/// of `un`.
pub(super) fn add_times_itch(un: usize, qn: usize) -> usize {
    if qn == 1 {
        0
    } else {
        un + qn + mul_bound(un, qn)
    }
}

/// Sets the `na` limbs at `r` to `a + b`, for the `na` limbs at `a` and the `nb <= na` at `b`,
/// and returns the carry out. `r` may be `a`.
///
/// # Safety
///
/// The limbs must be valid; `r` is `a` or apart from it, and apart from `b`.
pub(super) unsafe fn add_to(
    r: *mut u64,
    a: *const u64,
    na: usize,
    b: *const u64,
    nb: usize,
) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let c = add_n(r, a, b, nb);
        if r.cast_const() != a {
            ptr::copy_nonoverlapping(a.add(nb), r.add(nb), na - nb);
        }
        add_1(r.add(nb), na - nb, u64::from(c))
    }
}

/// Sets the `na` limbs at `r` to `a - b`, for the `na` limbs at `a` and the `nb <= na` at `b`,
/// and returns the borrow out. `r` may be `a`.
///
/// # Safety
///
/// As for [`add_to`].
pub(super) unsafe fn sub_from(
    r: *mut u64,
    a: *const u64,
    na: usize,
    b: *const u64,
    nb: usize,
) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let c = sub_n(r, a, b, nb);
        if r.cast_const() != a {
            ptr::copy_nonoverlapping(a.add(nb), r.add(nb), na - nb);
        }
        sub_1(r.add(nb), na - nb, u64::from(c))
    }
}

/// What a step of [`subdiv_ext`] leaves when it finds the greatest common divisor.
pub(super) struct Found {
    pub(super) g: *const u64,
    pub(super) gn: usize,
    pub(super) cofactor: Cofactor,
}

/// What a step of [`subdiv_ext`] leaves.
pub(super) enum StepExt {
    /// The numbers, reduced, of the given number of limbs.
    Size(usize),
    /// The greatest common divisor and its cofactor.
    Gcd(Found),
}

/// One step of Euclid's algorithm on the numbers of `n` limbs at `a` and `b`, both not zero, for
/// the extended algorithm, when [`hgcd2`] makes no progress: the smaller number is subtracted from
/// the larger once, and if that leaves it the larger, it is replaced by its remainder modulo the
/// smaller, the quotient added into the row. Each number stays in its own limbs, with its own
/// entry of the row: the smaller one is the one subtracted, whichever it is.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes and apart; the row's for reads
/// and writes, with room for `n + 1` limbs each; and the `avail` limbs at `t` for reads and writes,
/// apart from all of them.
pub(super) unsafe fn subdiv_ext(
    a: *mut u64,
    b: *mut u64,
    n: usize,
    row: &mut Row,
    t: *mut u64,
    avail: usize,
) -> StepExt {
    // SAFETY: as the caller guarantees.
    unsafe {
        let (mut pa, mut pb) = (a, b);
        let mut an = normalized(pa, n);
        let mut bn = normalized(pb, n);
        debug_assert!(an > 0 && bn > 0);
        let swap = match an.cmp(&bn) {
            Ordering::Equal => match cmp_n(pa, pb, an) {
                Ordering::Equal => {
                    return StepExt::Gcd(Found {
                        g: pa,
                        gn: an,
                        cofactor: row.smaller(),
                    });
                }
                order => order == Ordering::Greater,
            },
            order => order == Ordering::Greater,
        };
        // The smaller one in `a`, with `u1` its entry.
        if swap {
            core::mem::swap(&mut pa, &mut pb);
            core::mem::swap(&mut an, &mut bn);
            core::mem::swap(&mut row.u0, &mut row.u1);
        }
        // b - a, once, and how it compares with a.
        let borrow = sub_from(pb, pb, bn, pa, an);
        debug_assert!(borrow == 0);
        bn = normalized(pb, bn);
        let order = if bn == an {
            cmp_n(pb, pa, an)
        } else {
            bn.cmp(&an)
        };
        let step = if order == Ordering::Equal {
            // b - a = a: the greatest common divisor is `a`, with its cofactor.
            StepExt::Gcd(Found {
                g: pa,
                gn: an,
                cofactor: of_a(row, swap),
            })
        } else {
            // The step `b - a`: `u0 += u1`.
            row.add_times(true, &1, 1, t);
            if order == Ordering::Greater {
                // `b - a` is still the larger: its remainder modulo `a`, the quotient added in.
                let qn = bn - an + 1;
                let (q, w) = (t, t.add(qn));
                let room = avail.saturating_sub(qn);
                with_scratch(w, room, tdiv_itch(bn, an), |w| {
                    tdiv_qr(q, pb, bn, pa, an, w)
                });
                if normalized(pb, an) == 0 {
                    StepExt::Gcd(Found {
                        g: pa,
                        gn: an,
                        cofactor: of_a(row, swap),
                    })
                } else {
                    let qn = normalized(q, qn);
                    if qn > 0 {
                        with_scratch(w, room, add_times_itch(row.un, qn), |w| {
                            row.add_times(true, q, qn, w)
                        });
                    }
                    StepExt::Size(an)
                }
            } else {
                StepExt::Size(an)
            }
        };
        if swap {
            core::mem::swap(&mut row.u0, &mut row.u1);
        }
        step
    }
}

/// The cofactor of the number at `a` in [`subdiv_ext`], whose row has its entries swapped when
/// `swapped`: `+u1` for the original `a`, `-u0` for the original `b`, which the swap put in `u1`.
///
/// # Safety
///
/// The row's limbs must be valid for reads.
#[inline(always)]
unsafe fn of_a(row: &Row, swapped: bool) -> Cofactor {
    // SAFETY: as the caller guarantees.
    let n = unsafe { normalized(row.u1, row.un) };
    Cofactor {
        p: row.u1,
        n,
        negative: swapped && n != 0,
    }
}

/// The limbs of scratch space that [`gcdext_lehmer`] needs for numbers of `n` limbs, besides the
/// row and the results.
pub(super) fn lehmer_ext_itch(n: usize) -> usize {
    // The quotient of a step and its division, or the product of a quotient and an entry.
    (n + 1) + tdiv_itch(n, 1).max(2 * n + 4 + mul_bound(n + 1, n + 1))
}

/// The greatest common divisor `g` of the numbers of `n` limbs at `a` and `b`, both not zero,
/// with the cofactor that the row gives it, by Lehmer's method: [`hgcd2`]'s matrices applied to
/// both numbers and to the row, the steps it cannot take by [`subdiv_ext`], down to one limb each,
/// which [`gcdext_1`] finishes. The row starts as the caller sets it, `(0, 1)` for the cofactor of
/// `a` itself.
///
/// Writes `g` at `g` and its cofactor, `+u1 x - u0 y` for the cofactors `x` of the last `a` and `y`
/// of the last `b`, at `c`, and returns their lengths and the cofactor's sign; the numbers and the
/// row are clobbered.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes; the row's for reads and writes,
/// with room for `n + 1` limbs each, zero above its length; the `n` limbs at `g` and `n + 3` at `c`
/// for writes; and the `avail` limbs at `t` for reads and writes, all apart.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gcdext_lehmer(
    g: *mut u64,
    c: *mut u64,
    a: *mut u64,
    b: *mut u64,
    mut n: usize,
    row: &mut Row,
    t: *mut u64,
    avail: usize,
) -> (usize, usize, bool) {
    // SAFETY: as the caller guarantees; the numbers keep at least `n` limbs, of which the top ones
    // are not both zero.
    unsafe {
        // Whether the last call of hgcd2 took so many steps that the quotients are mostly 1.
        let mut ones = false;
        while n >= 2 {
            let (ah, al, bh, bl) = top_limbs(a, b, n);
            if let Some(m) = hgcd2_hinted(ah, al, bh, bl, &mut ones) {
                reduce_pair(a, b, n, m);
                n -= usize::from(*a.add(n - 1) | *b.add(n - 1) == 0);
                let un = row.un;
                let (c0, c1) = extend_row(row.u0, row.u1, un, m);
                *row.u0.add(un) = c0;
                *row.u1.add(un) = c1;
                row.un = un + usize::from(c0 | c1 != 0);
            } else {
                match subdiv_ext(a, b, n, row, t, avail) {
                    StepExt::Size(k) => n = k,
                    StepExt::Gcd(found) => {
                        ptr::copy(found.g, g, found.gn);
                        let f = found.cofactor;
                        ptr::copy(f.p, c, f.n);
                        return (found.gn, f.n, f.negative);
                    }
                }
            }
        }
        let (x, y) = (*a, *b);
        if x == y {
            *g = x;
            let f = row.smaller();
            ptr::copy(f.p, c, f.n);
            return (1, f.n, f.negative);
        }
        let (gl, u, v) = gcdext_1(x, y);
        *g = gl;
        let un = row.un;
        // g = u x + v y, so its cofactor is `u u1 - v u0`; `u` and `v` have opposite signs, or one
        // of them is zero and the other one 1.
        if u == 0 {
            let f = row.of_b();
            ptr::copy(f.p, c, f.n);
            return (1, f.n, f.negative);
        }
        if v == 0 {
            let f = row.of_a();
            ptr::copy(f.p, c, f.n);
            return (1, f.n, f.negative);
        }
        let negative = u < 0;
        let (u, v) = (u.unsigned_abs(), v.unsigned_abs());
        ptr::write_bytes(c, 0, un + 2);
        *c.add(un) = addmul_1(c, row.u1, un, u);
        let carry = addmul_1(c, row.u0, un, v);
        let (top, over) = (*c.add(un)).overflowing_add(carry);
        *c.add(un) = top;
        *c.add(un + 1) = u64::from(over);
        (1, normalized(c, un + 2), negative)
    }
}

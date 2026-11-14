//! Sums and differences of operands of as many limbs as their result, `n ≥ 3`, after MPFR's
//! `mpfr_add1sp` and `mpfr_sub1sp`: the smaller term shifted right into place, one pass of the
//! crate's kernels, and the rounding in the result's own vector, which is the larger term's copy.
//!
//! The bits of the smaller term shifted out are kept as two bits and a sticky bit, `g = 0.r1 r2
//! s…` units of the lowest limb's lowest bit. A sum's true value is `A + g`; a difference's
//! `A - g = (A - 1) + (1 - g)` when `g > 0`, and the two bits and sticky bit of `1 - g` are those
//! of `3 - G` with a sticky bit when `s` is set (`G = 2r1 + r2`), of `4 - G` exactly otherwise. A
//! difference of exponents apart by two or more cancels at most one bit, which the second bit
//! refills; one of exponents at most one apart shifts out at most one bit, and is exact.

use super::limbs;
use super::round::{away, overflow, signed, underflow};
use super::{BigFloat, Kind, Round};
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// The bits of the `n`-limb mantissa `y` (aligned at its top bit) from bit `d` up, into `ys`, and
/// the two bits below them and whether anything lies below those: `(r1, r2, s)`.
#[inline]
fn shift_in(y: &[u64], d: u64, ys: &mut [u64]) -> (bool, bool, bool) {
    let n = y.len();
    let total = 64 * n as u64;
    let (k, b) = ((d / 64) as usize, (d % 64) as u32);
    if d >= total {
        ys.fill(0);
    } else if b == 0 {
        ys[..n - k].copy_from_slice(&y[k..]);
        ys[n - k..].fill(0);
    } else {
        for i in 0..n - k - 1 {
            ys[i] = y[k + i] >> b | y[k + i + 1] << (64 - b);
        }
        ys[n - k - 1] = y[n - 1] >> b;
        ys[n - k..].fill(0);
    }
    let bit = |p: u64| p < total && limbs::bit(y, p);
    let r1 = d >= 1 && bit(d - 1);
    let r2 = d >= 2 && bit(d - 2);
    let s = d >= 3 && !limbs::zero_below(y, (d - 2).min(total));
    (r1, r2, s)
}

/// `x ± y` for mantissas of `n` limbs each (`n ≥ 3`, the result's), into `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add(
    nx: bool,
    x: &[u64],
    ex: i128,
    ny: bool,
    y: &[u64],
    ey: i128,
    prec: u64,
    round: Round,
    mut out: Vec<u64>,
) -> (Kind, Ordering) {
    let n = x.len();
    debug_assert!(n == y.len() && n >= 3 && n as u64 == prec.div_ceil(64));
    let sub = nx != ny;
    let (x, ex, y, ey, negative) = if ex > ey {
        (x, ex, y, ey, nx)
    } else if ex < ey {
        (y, ey, x, ex, ny)
    } else if sub {
        match x.iter().rev().cmp(y.iter().rev()) {
            Ordering::Greater => (x, ex, y, ey, nx),
            Ordering::Less => (y, ey, x, ex, ny),
            Ordering::Equal => {
                return (
                    Kind::Zero {
                        negative: round == Round::Down,
                    },
                    Ordering::Equal,
                );
            }
        }
    } else {
        (x, ex, y, ey, nx)
    };
    let d = ex - ey;
    let d = if d > 64 * n as i128 + 2 {
        64 * n as u64 + 3
    } else {
        d as u64
    };
    out.clear();
    out.extend_from_slice(x);
    let mut stack = [0u64; 64];
    let mut heap = Vec::new();
    let ys: &mut [u64] = if n <= 64 {
        &mut stack[..n]
    } else {
        heap.resize(n, 0);
        &mut heap
    };
    let (r1, r2, s) = shift_in(y, d, ys);
    let mut exp = ex;
    // `A` in `out`, and the bit below it and whether anything lies below that.
    let (rb, sb);
    if !sub {
        if nat::add_assign(&mut out, ys) {
            // One bit out of the top: shifted right, the bit out below.
            let low = out[0] & 1;
            shr1(&mut out);
            out[n - 1] |= 1 << 63;
            exp += 1;
            rb = low == 1;
            sb = r1 || r2 || s;
        } else {
            rb = r1;
            sb = r2 || s;
        }
    } else {
        let g = u8::from(r1) << 1 | u8::from(r2);
        let borrow = nat::sub_assign(&mut out, ys);
        debug_assert!(!borrow);
        // `1 - g` as two bits and a sticky bit, `A` one less, when `g > 0`.
        let (f, fs) = if g == 0 && !s {
            (0u8, false)
        } else {
            let borrow = nat::sub_assign(&mut out, &[1]);
            debug_assert!(!borrow);
            if s { (3 - g, true) } else { (4 - g, false) }
        };
        let top = bits(&out);
        if top == 0 {
            // `x - y < 1` unit: only for one exponent apart, `x = 2^(64n-1)`, `y` all ones;
            // the difference is the fraction, exactly.
            debug_assert!(d == 1 && f == 2 && !fs);
            out[n - 1] = 1 << 63;
            exp -= 64 * n as i128;
            rb = false;
            sb = false;
        } else if top == 64 * n as u64 {
            rb = f >> 1 == 1;
            sb = f & 1 == 1 || fs;
        } else if d >= 2 {
            // One bit cancelled: the first bit of the fraction comes in.
            debug_assert!(top == 64 * n as u64 - 1);
            shl(&mut out, 1, f >> 1 == 1);
            exp -= 1;
            rb = f & 1 == 1;
            sb = fs;
        } else {
            // Exact: at most one bit of fraction, which comes in with the shift.
            debug_assert!(!fs && f & 1 == 0);
            let z = 64 * n as u64 - top;
            shl(&mut out, z, f >> 1 == 1);
            exp -= i128::from(z);
            rb = false;
            sb = false;
        }
    }
    finish(negative, out, exp, rb, sb, prec, round)
}

/// The number of bits of `v`, whose top limbs may be zero.
#[inline]
fn bits(v: &[u64]) -> u64 {
    match v.iter().rposition(|&l| l != 0) {
        Some(i) => 64 * i as u64 + u64::from(64 - v[i].leading_zeros()),
        None => 0,
    }
}

/// `v >> 1` in place.
#[inline]
fn shr1(v: &mut [u64]) {
    let n = v.len();
    for i in 0..n - 1 {
        v[i] = v[i] >> 1 | v[i + 1] << 63;
    }
    v[n - 1] >>= 1;
}

/// `v·2^z + bit·2^(z - 1)` in place, for `0 < z < 64·len` and `v` below `2^(64·len - z)`: the
/// fraction's bit comes in below.
#[inline]
fn shl(v: &mut [u64], z: u64, bit: bool) {
    let n = v.len();
    let (k, b) = ((z / 64) as usize, (z % 64) as u32);
    if b == 0 {
        v.copy_within(..n - k, k);
    } else {
        for i in (k + 1..n).rev() {
            v[i] = v[i - k] << b | v[i - k - 1] >> (64 - b);
        }
        v[k] = v[0] << b;
    }
    v[..k].fill(0);
    if bit {
        let p = z - 1;
        v[(p / 64) as usize] |= 1 << (p % 64);
    }
}

/// The rounding of the `n`-limb `A` in `out`, aligned at its top bit, of the exponent `exp`, with
/// the bit `rb` below it and `sb` for anything below that, to `prec` bits.
#[inline]
fn finish(
    negative: bool,
    mut out: Vec<u64>,
    exp: i128,
    rb: bool,
    sb: bool,
    prec: u64,
    round: Round,
) -> (Kind, Ordering) {
    let n = out.len();
    let sh = 64 * n as u64 - prec;
    let (half, rest) = if sh == 0 {
        (rb, sb)
    } else {
        let low = out[0] & ((1u64 << sh) - 1);
        let h = 1u64 << (sh - 1);
        (low & h != 0, low & (h - 1) != 0 || rb || sb)
    };
    let unit = 1u64 << sh;
    out[0] &= unit.wrapping_neg();
    let up = match round {
        Round::Nearest => half && (rest || out[0] & unit != 0),
        _ => away(negative, round) && (half || rest),
    };
    let mut e = exp;
    if up {
        let (v, mut carry) = out[0].overflowing_add(unit);
        out[0] = v;
        let mut i = 1;
        while carry && i < n {
            (out[i], carry) = out[i].overflowing_add(1);
            i += 1;
        }
        if carry {
            out[n - 1] = 1 << 63;
            e += 1;
        }
    }
    let magnitude = if !(half || rest) {
        Ordering::Equal
    } else if up {
        Ordering::Greater
    } else {
        Ordering::Less
    };
    if e > i128::from(BigFloat::EXP_MAX) {
        let (x, o) = overflow(negative, prec, round);
        return (x.kind, o);
    }
    if e < i128::from(BigFloat::EXP_MIN) {
        let power = magnitude == Ordering::Equal
            && out[n - 1] == 1 << 63
            && out[..n - 1].iter().all(|&v| v == 0);
        let (x, o) = underflow(negative, exp, power, prec, round);
        return (x.kind, o);
    }
    (
        Kind::Finite {
            negative,
            exp: e as i64,
            mant: out,
        },
        signed(negative, magnitude),
    )
}

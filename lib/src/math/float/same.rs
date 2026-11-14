//! Sums and differences of operands of as many limbs as their result, `n ≥ 3`, after MPFR's
//! `mpfr_add1sp` and `mpfr_sub1sp`, written into the result's limbs in place: the smaller term
//! shifted right into them, one pass of the crate's kernels with the larger, and the rounding on
//! the lowest limb, with no vector of their own.
//!
//! The bits of the smaller term shifted out of the window are kept as the word below it, `L` (the
//! 64 bits just below the window's lowest), and a sticky bit `s` for any below that. A sum is the
//! window and `L` with `s`, exactly as it stands, shifted right by one when it carries. A
//! difference is `x·2^64 - (Y·2^64 + L + f)` for the window `Y` and the fraction `f` below `L`:
//! its truncation `x·2^64 - Y·2^64 - L - [f > 0]` in `n + 1` limbs, the word below `-L - [f > 0]`
//! and a borrow from the window when it is not zero, and the rest `1 - f` (strictly between 0 and
//! 1 unit of the word below) a sticky bit when `f > 0`. Terms of exponents apart by two or more
//! cancel at most one bit, which a shift left by one refills from the word below; terms apart by
//! at most one leave no fraction (`L` holds the one bit shifted out) and are exact, whatever they
//! cancel. The round bit lies in the window or at the top of the word below, at least 63 bits
//! above the lowest bit of the `n + 1` limbs, so the sticky bit of a refilled difference only
//! adds to what is already below the round bit.

use super::round::{away, overflow, signed, underflow};
use super::{BigFloat, Kind, Round};
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// The result of a kernel written into the limbs of a number: finite, of the sign, the exponent
/// of `0.m·2^exp` and the direction of the rounding, its mantissa in the limbs; or a value of its
/// own (a zero, an infinity, or a number of the range's ends), the limbs then not the result.
pub(crate) enum Put {
    Finite(bool, i64, Ordering),
    Other(Kind, Ordering),
}

/// Sets `out` (as long as `y`) to the bits of the mantissa `y` from bit `d` up, zero above `y`,
/// and returns the word of the 64 bits below them and whether any bit below that is set. Each
/// limb is a funnel shift of two, which takes `d` a multiple of 64 too, with no copy of the
/// library's for these short lengths.
#[inline]
fn shift_window(y: &[u64], d: u64, out: &mut [u64]) -> (u64, bool) {
    let n = y.len();
    debug_assert!(out.len() == n);
    let (k, b) = ((d / 64) as usize, (d % 64) as u32);
    if k > n {
        zero(out);
        return (0, true);
    }
    let pair = |hi: u64, lo: u64| lo >> b | hi << 1 << (63 - b);
    if k < n {
        for i in 0..n - k - 1 {
            out[i] = pair(y[k + i + 1], y[k + i]);
        }
        out[n - k - 1] = y[n - 1] >> b;
    }
    if k > 0 {
        zero(&mut out[n - k..]);
    }
    let low = match k {
        0 => pair(y[0], 0),
        _ if k == n => pair(0, y[n - 1]),
        _ => pair(y[k], y[k - 1]),
    };
    let sticky = k >= 1 && (y[k - 1] << 1 << (63 - b) != 0 || y[..k - 1].iter().any(|&v| v != 0));
    (low, sticky)
}

/// Sets the limbs `v` to zero.
#[inline]
fn zero(v: &mut [u64]) {
    for x in v.iter_mut() {
        *x = 0;
    }
}

/// Shifts the `n + 1` limbs of `out` above the word `f` left by `z` bits, `z < 64n`, the bits
/// of `f` coming in at the bottom of `out`; returns the new word below.
#[inline]
fn shift_in(out: &mut [u64], f: u64, z: u64) -> u64 {
    let n = out.len();
    let (zk, zb) = ((z / 64) as usize, (z % 64) as u32);
    if zk == 0 {
        debug_assert!(zb > 0);
        for i in (1..n).rev() {
            out[i] = out[i] << zb | out[i - 1] >> (64 - zb);
        }
        out[0] = out[0] << zb | f >> (64 - zb);
        return f << zb;
    }
    // Limb `j` of the `n + 1` limbs `f, out[0], ..., out[n - 1]`, from the top down, so that
    // each is read before it is written.
    let get = |out: &[u64], j: isize| -> u64 {
        if j < 0 {
            0
        } else if j == 0 {
            f
        } else {
            out[j as usize - 1]
        }
    };
    for i in (0..n).rev() {
        let j = i as isize + 1 - zk as isize;
        let v = if zb == 0 {
            get(out, j)
        } else {
            get(out, j) << zb | get(out, j - 1) >> (64 - zb)
        };
        out[i] = v;
    }
    0
}

/// The number of zero bits at the top of the limbs `v`, `64·len` when they are all zero.
#[inline]
fn leading_zeros(v: &[u64]) -> u64 {
    match v.iter().rposition(|&l| l != 0) {
        Some(i) => 64 * (v.len() - 1 - i) as u64 + u64::from(v[i].leading_zeros()),
        None => 64 * v.len() as u64,
    }
}

/// `x ± y` for mantissas of `n` limbs each (`n ≥ 3`, the result's), into the `n` limbs of
/// `out`, which must be apart from both.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn add(
    nx: bool,
    x: &[u64],
    ex: i128,
    ny: bool,
    y: &[u64],
    ey: i128,
    prec: u64,
    round: Round,
    out: &mut [u64],
) -> Put {
    let n = x.len();
    debug_assert!(n == y.len() && n == out.len() && n >= 3 && n as u64 == prec.div_ceil(64));
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
                return Put::Other(
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
    let d = if d > 64 * (n as i128 + 1) {
        64 * (n as u64 + 2)
    } else {
        d as u64
    };
    let mut exp = ex;
    if d == 0 {
        // Exact, straight into `out`: a sum carries, a difference may cancel any number of bits.
        if !sub {
            let carry = nat::add_n_into(out, x, y);
            debug_assert!(carry);
            let bit = out[0] & 1;
            for i in 0..n - 1 {
                out[i] = out[i] >> 1 | out[i + 1] << 63;
            }
            out[n - 1] = out[n - 1] >> 1 | 1 << 63;
            return finish(negative, out, bit << 63, false, exp + 1, prec, round);
        }
        let borrow = nat::sub_n_into(out, x, y);
        debug_assert!(!borrow);
        let z = leading_zeros(out);
        if z > 0 {
            exp -= i128::from(z);
            shift_in(out, 0, z);
        }
        return finish(negative, out, 0, false, exp, prec, round);
    }
    let (low, sticky) = shift_window(y, d, out);
    if !sub {
        if nat::add_assign(out, x) {
            // One bit out of the top: shifted right, the bit out into the word below.
            let bit = out[0] & 1;
            for i in 0..n - 1 {
                out[i] = out[i] >> 1 | out[i + 1] << 63;
            }
            out[n - 1] = out[n - 1] >> 1 | 1 << 63;
            exp += 1;
            return finish(
                negative,
                out,
                bit << 63 | low >> 1,
                sticky || low & 1 == 1,
                exp,
                prec,
                round,
            );
        }
        return finish(negative, out, low, sticky, exp, prec, round);
    }
    // `x - Y - [L + f > 0]` in the window, `-L - [f > 0]` below it.
    let f = low.wrapping_neg().wrapping_sub(u64::from(sticky));
    let borrow = nat::sub_from(out, x);
    debug_assert!(!borrow);
    if low != 0 || sticky {
        let borrow = nat::sub_assign(out, &[1]);
        debug_assert!(!borrow);
    }
    let z = leading_zeros(out);
    let f = if z == 0 {
        f
    } else if z == 64 * n as u64 {
        // Only for exponents one apart, `x = 2^(64n-1)` and `y` all ones: the difference is the
        // word below, `-(y << 63) = 2^63`, exactly.
        debug_assert!(d == 1 && !sticky && f == 1 << 63);
        out.fill(0);
        out[n - 1] = 1 << 63;
        exp -= 64 * n as i128;
        0
    } else {
        debug_assert!(d <= 1 || z == 1);
        exp -= i128::from(z);
        shift_in(out, f, z)
    };
    finish(negative, out, f, sticky, exp, prec, round)
}

/// The rounding of the `n` limbs `out`, aligned at their top bit, of the exponent `exp`, followed
/// by the word `low` and, below it, the sticky bit `sticky`, to `prec` bits.
#[inline]
pub(crate) fn finish(
    negative: bool,
    out: &mut [u64],
    low: u64,
    sticky: bool,
    exp: i128,
    prec: u64,
    round: Round,
) -> Put {
    let n = out.len();
    let sh = 64 * n as u64 - prec;
    let (half, rest) = if sh == 0 {
        (low >> 63 == 1, low << 1 != 0 || sticky)
    } else {
        let h = 1u64 << (sh - 1);
        let bits = out[0] & ((h << 1) - 1);
        out[0] ^= bits;
        (bits & h != 0, bits & (h - 1) != 0 || low != 0 || sticky)
    };
    let unit = 1u64 << sh;
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
        return Put::Other(x.kind, o);
    }
    if e < i128::from(BigFloat::EXP_MIN) {
        let power = magnitude == Ordering::Equal
            && out[n - 1] == 1 << 63
            && out[..n - 1].iter().all(|&v| v == 0);
        let (x, o) = underflow(negative, exp, power, prec, round);
        return Put::Other(x.kind, o);
    }
    Put::Finite(negative, e as i64, signed(negative, magnitude))
}

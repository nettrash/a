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
#[inline(always)]
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
        let r = 64 - zb;
        // From the top down in blocks of eight limbs, each block read whole before it is written
        // and the limb below it read before that block is shifted: fixed-size arrays, so that
        // the compiler vectorises the block without bounds checks, whatever it inlines around.
        let mut i = n;
        while i > 8 {
            let (lo, hi) = out.split_at_mut(i - 8);
            let below = lo[lo.len() - 1];
            let block: &mut [u64; 8] = (&mut hi[..8]).try_into().expect("eight limbs");
            let old = *block;
            block[0] = old[0] << zb | below >> r;
            for j in 1..8 {
                block[j] = old[j] << zb | old[j - 1] >> r;
            }
            i -= 8;
        }
        // The bottom limbs (all of them for eight or fewer), in a slice of their own length.
        let tail = &mut out[..i];
        for k in (1..i).rev() {
            tail[k] = tail[k] << zb | tail[k - 1] >> r;
        }
        tail[0] = tail[0] << zb | f >> r;
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

/// [`shift_window`] of the mantissa `ys` padded below with zero limbs to `n` (the same value, of
/// `n` limbs): the limbs that only zeros reach set to zero at once, the others as there.
#[inline]
fn shift_window_padded(ys: &[u64], n: usize, d: u64, out: &mut [u64]) -> (u64, bool) {
    debug_assert!(out.len() == n && ys.len() < n);
    let z = n - ys.len();
    let get = |j: usize| if j < z { 0 } else { ys[j - z] };
    let (k, b) = ((d / 64) as usize, (d % 64) as u32);
    if k > n {
        zero(out);
        return (0, true);
    }
    let pair = |hi: u64, lo: u64| lo >> b | hi << 1 << (63 - b);
    if k < n {
        // `out[i]` reads limbs `k + i` and `k + i + 1`, zeros below `z`.
        let start = (z.saturating_sub(k + 1)).min(n - k - 1);
        out[..start].fill(0);
        for (i, o) in out.iter_mut().enumerate().take(n - k - 1).skip(start) {
            *o = pair(get(k + i + 1), get(k + i));
        }
        out[n - k - 1] = get(n - 1) >> b;
    }
    if k > 0 {
        zero(&mut out[n - k..]);
    }
    let low = match k {
        0 => pair(get(0), 0),
        _ if k == n => pair(0, get(n - 1)),
        _ => pair(get(k), get(k - 1)),
    };
    let sticky = k >= 1
        && (get(k - 1) << 1 << (63 - b) != 0
            || ys[..(k - 1).saturating_sub(z).min(ys.len())]
                .iter()
                .any(|&v| v != 0));
    (low, sticky)
}

/// `x ± y` for `x` of `n` limbs (`n ≥ 3`, the result's) and a shorter `y` of a smaller exponent,
/// into the `n` limbs of `out`, apart from both, as [`add`] adds `y` padded below with zeros to
/// `n` limbs (the same value), but with the zeros never shifted nor read: `None` when the
/// exponent of `y` is not the smaller.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn add_long_short(
    nx: bool,
    x: &[u64],
    ex: i128,
    ny: bool,
    y: &[u64],
    ey: i128,
    prec: u64,
    round: Round,
    out: &mut [u64],
) -> Option<Put> {
    let n = x.len();
    debug_assert!(n == out.len() && n >= 3 && n as u64 == prec.div_ceil(64) && y.len() < n);
    if ex <= ey {
        return None;
    }
    let sub = nx != ny;
    let negative = nx;
    let d = ex - ey;
    let d = if d > 64 * (n as i128 + 1) {
        64 * (n as u64 + 2)
    } else {
        d as u64
    };
    let mut exp = ex;
    let (low, sticky) = if y.len() < SHORT_Y {
        out.copy_from_slice(x);
        return Some(add_short_in_place(nx, out, ex, ny, y, ey, prec, round));
    } else {
        shift_window_padded(y, n, d, out)
    };
    if !sub {
        if nat::add_assign(out, x) {
            // One bit out of the top: shifted right, the bit out into the word below.
            let bit = out[0] & 1;
            for i in 0..n - 1 {
                out[i] = out[i] >> 1 | out[i + 1] << 63;
            }
            out[n - 1] = out[n - 1] >> 1 | 1 << 63;
            exp += 1;
            return Some(finish(
                negative,
                out,
                bit << 63 | low >> 1,
                sticky || low & 1 == 1,
                exp,
                prec,
                round,
            ));
        }
        return Some(finish(negative, out, low, sticky, exp, prec, round));
    }
    // As in [`add`]: `x - Y - [L + f > 0]` in the window, `-L - [f > 0]` below it, the exponents
    // apart by one at least, so that at most one bit cancels unless they are one apart.
    let borrow = nat::sub_from(out, x);
    debug_assert!(!borrow);
    Some(sub_finish(negative, out, low, sticky, d, exp, prec, round))
}

/// `x ± y` into `out`, which holds `x` of `n` limbs (`n ≥ 3`, the result's), for `y` of fewer
/// than [`SHORT_Y`] limbs and of a smaller exponent (the caller sees to both): the few limbs of the
/// window that `y` reaches added to `out` or taken from it in place, the carry or borrow running
/// up as far as it goes, and the rounding, as [`add`] has them.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn add_short_in_place(
    nx: bool,
    out: &mut [u64],
    ex: i128,
    ny: bool,
    y: &[u64],
    ey: i128,
    prec: u64,
    round: Round,
) -> Put {
    let n = out.len();
    debug_assert!(n >= 3 && n as u64 == prec.div_ceil(64) && y.len() < SHORT_Y && ex > ey);
    let (sub, negative) = (nx != ny, nx);
    let d = ex - ey;
    let d = if d > 64 * (n as i128 + 1) {
        64 * (n as u64 + 2)
    } else {
        d as u64
    };
    let mut exp = ex;
    let mut w = [0u64; SHORT_Y + 1];
    let (start, cnt, low, sticky) = short_window(y, n, d, &mut w);
    let w = &w[..cnt];
    if !sub {
        if nat::add_assign(&mut out[start..], w) {
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
    let borrow = nat::sub_assign(&mut out[start..], w);
    debug_assert!(!borrow);
    sub_finish(negative, out, low, sticky, d, exp, prec, round)
}

/// `y ± x` for a short `y` of fewer than [`SHORT_Y`] limbs and of the larger exponent, and `x` of
/// `n` limbs (`n ≥ 3`, the result's), into the `n` limbs of `out`, apart from both: as [`add`]
/// has it with `y` padded below with zeros to `n` limbs (the same value), the long `x` shifted
/// into `out` and `y` added at the top, or `out` negated and `y` added for a difference, the
/// zeros never read. `None` unless `y`'s exponent is the larger.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn add_short_above(
    nx: bool,
    x: &[u64],
    ex: i128,
    ny: bool,
    y: &[u64],
    ey: i128,
    prec: u64,
    round: Round,
    out: &mut [u64],
) -> Option<Put> {
    let n = x.len();
    let ly = y.len();
    if ey <= ex || ly >= SHORT_Y || ly >= n || n != out.len() {
        return None;
    }
    debug_assert!(n >= 3 && n as u64 == prec.div_ceil(64));
    let (sub, negative) = (nx != ny, ny);
    let d = ey - ex;
    let d = if d > 64 * (n as i128 + 1) {
        64 * (n as u64 + 2)
    } else {
        d as u64
    };
    let mut exp = ey;
    let (low, sticky) = shift_window(x, d, out);
    if !sub {
        if nat::add_assign(&mut out[n - ly..], y) {
            // One bit out of the top: shifted right, the bit out into the word below.
            let bit = out[0] & 1;
            for i in 0..n - 1 {
                out[i] = out[i] >> 1 | out[i + 1] << 63;
            }
            out[n - 1] = out[n - 1] >> 1 | 1 << 63;
            exp += 1;
            return Some(finish(
                negative,
                out,
                bit << 63 | low >> 1,
                sticky || low & 1 == 1,
                exp,
                prec,
                round,
            ));
        }
        return Some(finish(negative, out, low, sticky, exp, prec, round));
    }
    // `Y·B^(n - ly) - X'` modulo `B^n` (the difference is below it, and positive): `-X'` in two's
    // complement, then `y` added at the top, the carry out of the top dropped.
    let mut carry = true;
    for v in out.iter_mut() {
        let (s, c) = (!*v).overflowing_add(u64::from(carry));
        *v = s;
        carry = c;
    }
    let _ = nat::add_assign(&mut out[n - ly..], y);
    Some(sub_finish(negative, out, low, sticky, d, exp, prec, round))
}

/// The most limbs of the short operand that [`add_long_short`] adds in place into a copy of the
/// long one (its window on the stack); a longer one is shifted into the result's limbs whole.
/// Measured on an Apple M4 Max (`add_round`, `set_add` and `+=` of a long operand of 16 to 1,563
/// limbs and one of 2 to 31 limbs below or above it; bounds 3 to 32, and 10 in a build of its own):
/// in place the short operands below take 0.04 to 0.86 of the shifted path's time at every length
/// tried, but a larger bound costs the sums of a short operand above a long one of 48 to 64 limbs
/// (whose padded sum on the stack is faster) 1 to 17 %, and with the bound at 10 the in-place sums
/// of 2 to 7 limbs 1 to 20 % (the window's code).
pub(crate) const SHORT_Y: usize = 8;

/// The limbs of the window of `ys` padded below with zeros to `n` limbs and shifted down by `d`
/// bits, as [`shift_window`] sets them, that are not certainly zero: from limb `start`, `cnt` of
/// them into `w`; and the word below the window and the sticky bit below that.
#[inline]
fn short_window(ys: &[u64], n: usize, d: u64, w: &mut [u64]) -> (usize, usize, u64, bool) {
    let z = n - ys.len();
    let get = |j: usize| if j < z { 0 } else { ys[j - z] };
    let (k, b) = ((d / 64) as usize, (d % 64) as u32);
    if k > n {
        return (0, 0, 0, true);
    }
    let pair = |hi: u64, lo: u64| lo >> b | hi << 1 << (63 - b);
    let (start, cnt) = if k < n {
        let start = (z.saturating_sub(k + 1)).min(n - k - 1);
        for i in start..n - k - 1 {
            w[i - start] = pair(get(k + i + 1), get(k + i));
        }
        w[n - k - 1 - start] = get(n - 1) >> b;
        (start, n - k - start)
    } else {
        (0, 0)
    };
    let low = match k {
        0 => pair(get(0), 0),
        _ if k == n => pair(0, get(n - 1)),
        _ => pair(get(k), get(k - 1)),
    };
    let sticky = k >= 1
        && (get(k - 1) << 1 << (63 - b) != 0
            || ys[..(k - 1).saturating_sub(z).min(ys.len())]
                .iter()
                .any(|&v| v != 0));
    (start, cnt, low, sticky)
}

/// The rest of a difference of [`add_long_short`] from `x - Y` in `out` (the window `Y` taken,
/// the word below it `low` and the sticky bit below that not yet): the borrow of `L + f`, the
/// bit that may cancel refilled, and the rounding.
#[allow(clippy::too_many_arguments)]
#[inline]
fn sub_finish(
    negative: bool,
    out: &mut [u64],
    low: u64,
    sticky: bool,
    d: u64,
    mut exp: i128,
    prec: u64,
    round: Round,
) -> Put {
    let n = out.len();
    let f = low.wrapping_neg().wrapping_sub(u64::from(sticky));
    if low != 0 || sticky {
        let borrow = nat::sub_assign(out, &[1]);
        debug_assert!(!borrow);
    }
    let z = leading_zeros(out);
    let f = if z == 0 {
        f
    } else if z == 64 * n as u64 {
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

/// The most limbs of the operands of [`div`], whose dividend and quotient are on the stack.
/// Measured on an Apple M4 Max (`div_round`, `set_div` and `/=`, 3 to 24 limbs; bounds 8 to 12): on
/// the stack these quotients take 0.61 to 0.87 of the general division's time to 7 limbs, 0.83 to
/// 0.97 at 8 to 10, 0.93 to 1.01 at 11 and 12 and 1.01 to 1.18 from 14; the larger buffers of a
/// bound of 11 or 12 cost `set_div` of 3 to 6 limbs 3 to 4 % (of 10, at most 2 %).
pub(crate) const DIV_STACK: usize = 10;

/// `x / y` for mantissas of `n` limbs each (`3 ≤ n ≤ DIV_STACK`, the result's), into the `n`
/// limbs of `out`, apart from both: the quotient `Q = floor(X·B^(n+1) / Y)` of `n + 1` limbs (and
/// a top bit when `X ≥ Y`) and its remainder by the schoolbook division of the crate's integers,
/// on the stack; its top `n` limbs (shifted down by one for the top bit) are the window, the next
/// word and the remainder the rounding's. `None` for other lengths.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn div(
    negative: bool,
    x: &[u64],
    ex: i128,
    y: &[u64],
    ey: i128,
    prec: u64,
    round: Round,
    out: &mut [u64],
) -> Option<Put> {
    let n = x.len();
    if n != y.len() || n != out.len() || !(3..=DIV_STACK).contains(&n) {
        return None;
    }
    let qn = n + 1;
    let mut num = [0u64; 2 * DIV_STACK + 1];
    let num = &mut num[..qn + n];
    num[qn..].copy_from_slice(x);
    let mut q = [0u64; DIV_STACK + 1];
    let q = &mut q[..qn];
    // No working memory below divide and conquer.
    let qh = nat::div_qr(q, num, y, &mut Vec::new());
    let rest = num[..n].iter().any(|&v| v != 0);
    if qh {
        // `B^qn + q`, shifted down by one: its top `n` limbs, the word below, the bit out.
        let bit = q[0] & 1;
        for i in 0..qn - 1 {
            q[i] = q[i] >> 1 | q[i + 1] << 63;
        }
        q[qn - 1] = q[qn - 1] >> 1 | 1 << 63;
        out.copy_from_slice(&q[1..]);
        Some(finish(
            negative,
            out,
            q[0],
            rest || bit == 1,
            ex - ey + 1,
            prec,
            round,
        ))
    } else {
        out.copy_from_slice(&q[1..]);
        Some(finish(negative, out, q[0], rest, ex - ey, prec, round))
    }
}

/// The rounding of the `n` limbs `out`, aligned at their top bit, of the exponent `exp`, followed
/// by the word `low` and, below it, the sticky bit `sticky`, to `prec` bits.
#[inline(always)]
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

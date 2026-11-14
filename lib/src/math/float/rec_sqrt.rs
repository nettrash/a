//! The reciprocal square root at the precisions where it is not exact: in registers for a mantissa
//! of one or two limbs and at most 127 bits, as the integer square root of a quotient of four
//! limbs, and beyond by Newton's iteration on the reciprocal square root itself, as MPFR's
//! `mpfr_rec_sqrt` computes it, on numbers of whole limbs with a bound of the error that decides
//! the rounding but within a few units of a boundary (when the exact method of the caller takes
//! over). `1/sqrt(x)` is exact only for `x = 4^k`, which the caller takes first, and never halfway
//! between two numbers of a precision (its square would be `2^(2e)/m^2` for an odd `m > 1`), so
//! an approximation decides the rounding once the bound keeps it away from the boundaries.

use super::general::Op;
use super::limbs::{self, Work};
use super::power_fixed::certain;
use super::round::{Rounded, round_into};
use super::sqrt_small::{root1, root2};
use super::{Kind, Round};
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// `1/sqrt(x)` for a positive `x` whose mantissa `M` of `l ≤ 2` limbs is not a power of two,
/// rounded to `prec ≤ 127` bits, in registers: with `x = M·2^(ex - 64l)`, `1/sqrt(x) =
/// sqrt(2^E / M)·2^(-(E - 64l + ex)/2)` for `E` of the parity of `ex`, and `floor(sqrt(floor(2^E /
/// M))) = floor(sqrt(2^E / M))`, the integer root of the quotient `Q`. For `prec ≤ 63` and one limb,
/// `E = 190` or 191 makes `Q` of two limbs and the root of 64 bits ([`root1`]); otherwise `E = 64l
/// + 254` or `64l + 255` makes `Q` of four limbs, in `(2^254, 2^256)`, and the root of 128 bits
/// ([`root2`]). The value is never exact, so the remainders are a sticky bit.
#[inline]
pub(super) fn small(m: &[u64], ex: i128, prec: u64, round: Round) -> Rounded {
    let odd = ex & 1;
    match *m {
        [d] if prec <= 63 => {
            // `2^E`, `E = 190 + odd`, is the limb `2^(62 + odd)` above two zero limbs, below `d`.
            let v = nat::invert_limb(d);
            let (q1, r) = nat::div_2by1(1 << (62 + odd), 0, d, v);
            let (q0, _) = nat::div_2by1(r, 0, d, v);
            let e = 190 + odd;
            let q = u128::from(q1) << 64 | u128::from(q0);
            root1(q, true, 64 - (e - 64 + ex) / 2, prec, round)
        }
        [d] => {
            let v = nat::invert_limb(d);
            let mut q = [0u64; 4];
            let mut r = 1u64 << (62 + odd);
            for limb in q.iter_mut().rev() {
                let (t, rem) = nat::div_2by1(r, 0, d, v);
                *limb = t;
                r = rem;
            }
            let e = 318 + odd;
            quotient(q, 128 - (e - 64 + ex) / 2, prec, round)
        }
        [d0, d1] => {
            let v = nat::invert_3by2(d1, d0);
            let mut u = [0u64; 6];
            u[5] = 1 << (62 + odd);
            let mut q = [0u64; 4];
            nat::div_qr_2(&mut q, &u, d1, d0, v);
            let e = 382 + odd;
            quotient(q, 128 - (e - 128 + ex) / 2, prec, round)
        }
        _ => unreachable!("a mantissa of one or two limbs"),
    }
}

/// The root of the quotient `q` of four limbs, as `0.S·2^exp`, with a sticky bit.
#[inline(always)]
fn quotient(q: [u64; 4], exp: i128, prec: u64, round: Round) -> Rounded {
    let h = u128::from(q[3]) << 64 | u128::from(q[2]);
    root2(h, q[1], q[0], true, exp, prec, round)
}

/// From this precision on, Newton's iteration ([`newton`]); below, the square root and a division
/// ([`via_sqrt`]). Measured on an Apple M4 Max (loaded): at 1,000 bits 540 ns against 600, at
/// 3,000 bits 2.6 µs against 1.8, at 10^4 bits 34 µs against 9.6.
pub(super) const NEWTON: u64 = 2000;

/// `2^64` as an `f64`.
const TWO_64: f64 = 18_446_744_073_709_551_616.0;

/// The working lengths of Newton's iteration, in limbs after the point, the first one first: from
/// `mf` down, each `ceil((m' + 1)/2)` of the next `m'`, so that `m' ≤ 2m - 1` (and the step from
/// one limb to two, `m' = 2m`), down to one limb.
fn schedule(mf: usize) -> Vec<usize> {
    let mut v = vec![mf];
    while let Some(&m) = v.last()
        && m > 2
    {
        v.push((m + 2) / 2);
    }
    if v.last() != Some(&1) {
        v.push(1);
    }
    v.reverse();
    v
}

/// `1/sqrt(x)` for a positive `x` that is not a power of 4, rounded to `prec` bits, by Newton's
/// iteration `y ← y + y·(1 - a·y^2)/2` on `y ≈ 1/sqrt(a)` for `x = a·2^ex'`, `ex'` even and `a =
/// 0.M` or `0.M/2` in `[1/4, 1)`, so `y ∈ (1, 2)`; `None` when the bound leaves the rounding
/// undecided.
///
/// `y` is `Y·B^-m`, `B = 2^64`, `Y` of `m + 1` limbs, with a relative error `e`. A step to `m'`
/// limbs (`m' ≤ 2m`) computes `T = Y^2` exactly, `U = A·T` for `A`, the top `m' + 1` limbs of `a`
/// after the point, cut to `m' + 1` limbs after the point, `R = 1 - U` (small, of either sign),
/// and `Y' = Y·B^(m' - m) + floor(Y·R / 2)` at `m'` limbs. Exact, the step has the relative error
/// `-(3/2)δ^2 - δ^3/2` for `y_k = y(1 + δ)`, at most `1.5e^2 + 0.5e^3`; the cut of `a` (below
/// `B^-(m' + 1)`, times `y^3/2 < 4`), that of `U` (below `B^-(m' + 1)`, times `y/2 < 1`) and the
/// floor (one unit of `B^-m'`) add less than `(1 + 2^-60)·B^-m'`, relative too as `y > 1`. The
/// start is `1/sqrt` of the top limb of `a` in double precision, within `2^-51` relatively (the
/// conversion, the limbs below it, the square root and the division), cut to one limb: `e ≤
/// 2^-50`. The bound is carried as `c = e·B^m`, in an `f64`.
///
/// At the end, `|Y - y·B^mf| ≤ y·c < 2c` units, so `y·B^mf` lies strictly inside `[Y - K, Y + K]`
/// for `K = ceil(2c) + 1`, and rounds as `Y - K` with a sticky bit when every integer up to `Y + K`
/// does ([`certain`]).
pub(super) fn newton(x: Op<'_>, prec: u64, round: Round, w: &mut Work) -> Option<(Kind, Ordering)> {
    let odd = x.exp & 1 != 0;
    // `a = A·B^-la`: the mantissa, or for an odd exponent the mantissa halved, one limb longer.
    let shifted;
    let a: &[u64] = if odd {
        let mut v = Vec::with_capacity(x.m.len() + 1);
        v.push(x.m[0] << 63);
        v.extend(x.m.windows(2).map(|p| p[0] >> 1 | p[1] << 63));
        v.push(x.m[x.m.len() - 1] >> 1);
        shifted = v;
        &shifted
    } else {
        x.m
    };
    let ex = x.exp + i128::from(odd);
    let la = a.len();
    let mf = (prec + 14).div_ceil(64) as usize;
    let Work {
        a: y,
        b: t,
        c: u,
        scratch,
    } = w;
    // The start: `1/sqrt(a)` from the top limb, `a ≥ 1/4`, so at most 2.
    let af = a[la - 1] as f64 / TWO_64;
    let y0 = ((1.0 / af.sqrt()) * TWO_64) as u128;
    y.clear();
    y.extend_from_slice(&[y0 as u64, (y0 >> 64) as u64]);
    let mut m = 1usize;
    let mut c = 2f64.powi(14);
    for mn in schedule(mf) {
        // `T = Y^2`, of `2m + 2` limbs, `y^2 = T·B^-2m`.
        nat::sqr_into(t, y, scratch);
        // `U = A'·T`, `A'` the top `m' + 1` limbs of `a` (or all of it), `a·y^2 ≈ U·B^-(ls + 2m)`.
        let ls = la.min(mn + 1);
        nat::mul_into(u, &a[la - ls..], t, scratch);
        // `U` cut to `m' + 1` limbs after the point: `U[d..]`, `d = ls + 2m - (m' + 1) ≥ 0`.
        let d = ls + 2 * m - (mn + 1);
        let top = &u[d..];
        // `R = B^(m' + 1) - U'`: its sign and magnitude, in `t`.
        let over = top[mn + 1..].iter().any(|&v| v != 0);
        t.clear();
        if over {
            // `U' - B^(m' + 1)`: the limbs from `m' + 1` up hold 1 (and zeros), so take the rest.
            debug_assert!(top[mn + 1] == 1 && top[mn + 2..].iter().all(|&v| v == 0));
            t.extend_from_slice(&top[..mn + 1]);
        } else {
            // The two's complement of the low `m' + 1` limbs.
            t.extend(top[..mn + 1].iter().map(|&v| !v));
            nat::add_assign(t, &[1]);
        }
        limbs::trim(t);
        // `Y' = Y·B^(m' - m) ± floor(Y·|R| / (2·B^(m + 1)))`.
        if t.is_empty() {
            u.clear();
        } else {
            nat::mul_into(u, y, t, scratch);
        }
        t.clear();
        t.resize(mn - m, 0);
        t.extend_from_slice(y);
        if u.len() > m + 1 {
            let dlt = &mut u[m + 1..];
            // Halved: shifted right by one bit.
            let mut carry = 0u64;
            for v in dlt.iter_mut().rev() {
                let low = *v & 1;
                *v = *v >> 1 | carry << 63;
                carry = low;
            }
            let k = dlt.len().min(t.len());
            debug_assert!(dlt[k..].iter().all(|&v| v == 0));
            if over {
                nat::sub_assign(t, &dlt[..k]);
            } else {
                nat::add_assign(t, &dlt[..k]);
            }
        }
        core::mem::swap(y, t);
        // The bound: `c' = 1.5c^2·B^(m' - 2m) + 0.5c^3·B^(m' - 3m) + 1 + 2^-60`.
        let s2 = 2f64.powi(64 * (mn as i32 - 2 * m as i32));
        let s3 = 2f64.powi(64 * (mn as i32 - 3 * m as i32));
        c = (1.5 * c * c * s2 + 0.5 * c * c * c * s3 + 1.0 + 1e-18) * (1.0 + 1e-15);
        m = mn;
    }
    // A bound that is not finite, or large, is left to the exact method.
    if c.is_nan() || c >= 1e6 {
        return None;
    }
    let k = (2.0 * c).ceil() as u64 + 1;
    // `Y - K`, whose integers up to `Y + K` must round alike.
    if nat::sub_assign(y, &[k]) {
        return None;
    }
    limbs::trim(y);
    if !certain(y, 2 * k, prec) {
        return None;
    }
    Some(round_into(
        false,
        y,
        -64 * m as i128 - ex / 2,
        true,
        prec,
        round,
        Vec::new(),
    ))
}

/// `1/sqrt(x)` for a positive `x` that is not a power of 4, rounded to `prec` bits, from the square
/// root and one division: with `c` the parity of the exponent, the top `2q` limbs `X` of `0.M·2^-c`
/// (`q` the limbs of `prec + 16` bits, 2 at least) have the root `s = floor(sqrt(X))` of `64q`
/// bits, that of the whole radicand too, which lies in `[s, s + 1)`; so `Y = 2^K /
/// sqrt(radicand)`, `K = 128q - 1`, lies in `(2^K/(s + 1), 2^K/s]`, an interval shorter than 2
/// units, that is in `(y - 2, y + 1)` for `y = floor(2^K/s)`. `None` when that does not decide the
/// rounding.
pub(super) fn via_sqrt(
    x: Op<'_>,
    prec: u64,
    round: Round,
    w: &mut Work,
) -> Option<(Kind, Ordering)> {
    let q = (prec + 16).div_ceil(64).max(2) as usize;
    let c = (x.exp & 1) as u32;
    let Work {
        a: r,
        b: s,
        c: num,
        scratch,
    } = w;
    r.clear();
    r.resize(2 * q, 0);
    limbs::top_limbs(x.m, c, r);
    nat::sqrt_rem(s, None, r, scratch);
    debug_assert!(s.len() == q && s[q - 1] >> 63 == 1);
    num.clear();
    num.resize(2 * q, 0);
    num[2 * q - 1] = 1 << 63;
    r.clear();
    r.resize(q, 0);
    if nat::div_qr(r, num, s, scratch) {
        // `s = 2^(64q - 1)`: the quotient `2^(64q)`, left to the exact method.
        return None;
    }
    if nat::sub_assign(r, &[2]) {
        return None;
    }
    limbs::trim(r);
    if !certain(r, 2, prec) {
        return None;
    }
    let lsb = -(64 * q as i128 - 1) - (x.exp + i128::from(c)) / 2;
    Some(round_into(false, r, lsb, true, prec, round, Vec::new()))
}

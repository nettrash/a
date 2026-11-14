//! Roots of degree 3 to [`NMAX`] at short precisions, in registers and arrays: the root of 64 bits
//! from a double-precision estimate made sure by one step of Newton's method on the exact
//! remainder, and the root of 128 bits from that one by a second step, as MPFR has none (its
//! `mpfr_rootn_ui` and `mpfr_cbrt` take the integer root of GMP's `mpz_root` at every size).
//!
//! For `x = 0.M·2^e`, `e = nq + r` with `-n < r ≤ 0`: the root is `2^q` times that of `y = 0.M·2^r`,
//! in `[2^-n, 1)`, which is `ρ = (R·2^(-64n))^(1/n)` for the radicand `R = floor(y·2^(64n))` of
//! `n` limbs (or `2n` for 128 bits), the bits of `M` below it a tail, as the floor of the root of a
//! truncated radicand is that of the exact one. `ρ·2^64` lies in `[2^63, 2^64)`.

use super::Round;
use super::general::Op;
use super::limbs;
use super::round::Rounded;
use super::small::{round_top, round_top2};
use crate::math::numbers::nat;

/// The largest degree of the short path.
pub(super) const NMAX: usize = 16;

/// The limbs of a radicand of 64 bits of root, and one more.
const LEN: usize = NMAX + 1;

/// Those of a radicand of 128 bits of root, and two more.
const LEN2: usize = 2 * NMAX + 2;

/// `2^64` as an `f64`.
const TWO_64: f64 = 18_446_744_073_709_551_616.0;

/// `s^(n - 1)` into `pm` and `s^n` into `p`, of `n - 1` and `n` limbs, the least significant
/// first (the limbs above them untouched).
#[inline]
fn powers(s: u64, n: usize, pm: &mut [u64; LEN], p: &mut [u64; LEN]) {
    p[0] = s;
    for len in 1..n {
        if len == n - 1 {
            pm[..len].copy_from_slice(&p[..len]);
        }
        let mut carry = 0u64;
        for v in p[..len].iter_mut() {
            let t = u128::from(*v) * u128::from(s) + u128::from(carry);
            *v = t as u64;
            carry = (t >> 64) as u64;
        }
        p[len] = carry;
    }
}

/// The order of the magnitudes of `n` limbs `a` and `b`.
#[inline]
fn cmp(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    a.iter().rev().cmp(b.iter().rev())
}

/// The value of the limbs `m` as `f·2^(64k)`: `f` from the top two limbs that are not zero, within
/// `2^-52` relatively; `(0, 0)` for zero.
#[inline]
fn to_f64(m: &[u64]) -> (f64, i32) {
    let Some(i) = m.iter().rposition(|&v| v != 0) else {
        return (0.0, 0);
    };
    let lo = if i > 0 { m[i - 1] as f64 } else { 0.0 };
    (m[i] as f64 * TWO_64 + lo, i as i32 - 1)
}

/// An estimate `s` of `F = floor(ρ)` for `ρ = R^(1/n)` and the radicand `R` of `n` limbs (`ρ ∈
/// [2^63, 2^64)`), with `F ∈ {s - 1, s, s + 1}`; `None` when the double-precision start is too
/// far off to be sure (which a working `cbrt` or `powf` never is).
///
/// The start `s0 ∈ [2^63, 2^64)` is `ρ` in double precision, of any accuracy. With the exact
/// remainder `E = R - s0^n` and `c = E / (n·s0^(n-1))`, `|ρ - s0| ≤ Δ = n|c|` (from `(1 + u)^(1/n)
/// ≤ 1 + u/n` and `(1 - u)^(1/n) ≥ 1 - u`); Newton's step `g = s0 + c` overshoots `ρ` by at most
/// `(n - 1)/2·Δ^2/s0·(1 + nΔ/s0)`, below `2^-12` for `Δ ≤ 2^24`, and `c` is computed within
/// `2^-49` of its size, below `2^-25`. So `s0 + c` is within `2^-11` of `ρ`, and its floor within
/// one of `F`.
#[inline]
fn estimate(r: &[u64], n: usize) -> Option<u64> {
    let (top, _) = to_f64(&r[n - 2..n]);
    // `y = R·2^(-64n)` from its top two limbs, at least `2^-n`.
    let y = top / (TWO_64 * TWO_64);
    let rho = if n == 3 {
        y.cbrt()
    } else {
        y.powf(1.0 / n as f64)
    };
    let s0 = ((rho * TWO_64) as u64).max(1 << 63);
    let (mut pm, mut p) = ([0u64; LEN], [0u64; LEN]);
    powers(s0, n, &mut pm, &mut p);
    let (below, e) = match cmp(&r[..n], &p[..n]) {
        core::cmp::Ordering::Less => (true, sub(&p[..n], &r[..n])),
        _ => (false, sub(&r[..n], &p[..n])),
    };
    let (ef, ke) = to_f64(&e[..n]);
    let (df, kd) = to_f64(&pm[..n - 1]);
    let c = ef / (n as f64 * df) * 2f64.powi(64 * (ke - kd).clamp(-20, 20));
    if c.is_nan() || c * n as f64 > 16_777_216.0 {
        return None;
    }
    let step = if below { -c } else { c }.floor() as i64;
    let s = (i128::from(s0) + i128::from(step)).clamp(1 << 63, i128::from(u64::MAX));
    Some(s as u64)
}

/// The number of bits of the limbs `m`, zero limbs at the top allowed.
#[inline]
fn bits(m: &[u64]) -> u64 {
    m.iter()
        .rposition(|&v| v != 0)
        .map_or(0, |i| 64 * i as u64 + u64::from(64 - m[i].leading_zeros()))
}

/// `a - b` for `a ≥ b`, of `n` limbs.
#[inline]
fn sub(a: &[u64], b: &[u64]) -> [u64; LEN] {
    let mut d = [0u64; LEN];
    let mut borrow = false;
    for i in 0..a.len() {
        let (v, b1) = a[i].overflowing_sub(b[i]);
        let (v, b2) = v.overflowing_sub(u64::from(borrow));
        d[i] = v;
        borrow = b1 | b2;
    }
    d
}

/// `F = floor(R^(1/n))` from `s` within one of it, with `F^(n-1)` in `pm` and `F^n` in `p`.
#[inline]
fn floor_root(r: &[u64], n: usize, s: u64, pm: &mut [u64; LEN], p: &mut [u64; LEN]) -> u64 {
    powers(s, n, pm, p);
    if cmp(&p[..n], &r[..n]).is_gt() {
        powers(s - 1, n, pm, p);
        return s - 1;
    }
    if s < u64::MAX {
        let (mut pm1, mut p1) = ([0u64; LEN], [0u64; LEN]);
        powers(s + 1, n, &mut pm1, &mut p1);
        if cmp(&p1[..n], &r[..n]).is_le() {
            *pm = pm1;
            *p = p1;
            return s + 1;
        }
    }
    s
}

/// Whether the bits `low` below the round bit (`k` of them) of an estimate known within `lo` below
/// and `hi` above decide the rounding with a sticky bit: every candidate has some bit set there
/// and the same bits above.
#[inline(always)]
fn decided(low: u128, k: u64, lo: u128, hi: u128) -> bool {
    (2..127).contains(&k) && low > lo && low < (1u128 << k) - hi
}

/// The radicand `floor(0.M·2^(r + 64·len))` of `len` limbs for `-n < r ≤ 0`, and whether bits of
/// `M` lie below it.
#[inline]
fn radicand<const N: usize>(m: &[u64], r: i128, len: usize) -> ([u64; N], bool) {
    let shift = 64 * len as i128 + r - 64 * m.len() as i128;
    let mut out = [0u64; N];
    for (i, v) in out[..len].iter_mut().enumerate() {
        *v = limbs::extract(m, 64 * i as i128 - shift);
    }
    (out, shift < 0 && limbs::any_below(m, -shift))
}

/// The `n`-th root, `3 ≤ n ≤ NMAX`, of the finite `x` (negative only for an odd `n`) rounded to
/// `prec ≤ 127` bits, when the short path decides it; `None` otherwise.
pub(super) fn root(x: Op<'_>, n: u64, prec: u64, round: Round) -> Option<Rounded> {
    let negative = x.negative;
    let ni = i128::from(n);
    let n = n as usize;
    // `e = nq + r`, `-n < r ≤ 0`.
    let q = -(-x.exp).div_euclid(ni);
    let r = x.exp - ni * q;
    if prec <= 64 {
        let (rad, tail) = radicand::<LEN>(x.m, r, n);
        let s = estimate(&rad, n)?;
        if prec <= 61 {
            let k = 63 - prec;
            if decided(u128::from(s) & ((1 << k) - 1), k, 1, 1) {
                return Some(round_top(
                    negative,
                    u128::from(s) << 64,
                    q,
                    true,
                    prec,
                    round,
                ));
            }
        }
        let (mut pm, mut p) = ([0u64; LEN], [0u64; LEN]);
        let f = floor_root(&rad, n, s, &mut pm, &mut p);
        let exact = p[..n] == rad[..n] && !tail;
        if prec == 64 {
            // The round bit below `F`: `ρ ≥ F + 1/2` exactly when `2^n·R ≥ (2F + 1)^n`, never equal
            // (odd and even), and a tail leaves it open.
            if tail {
                return None;
            }
            let half = above_half(&rad, n, f);
            return Some(round_top(
                negative,
                u128::from(f) << 64 | u128::from(half) << 63,
                q,
                !exact,
                prec,
                round,
            ));
        }
        return Some(round_top(
            negative,
            u128::from(f) << 64,
            q,
            !exact,
            prec,
            round,
        ));
    }
    // 128 bits: the root `S_hi` of the top `n` limbs, exact, then Newton's step from `S_hi·2^64`,
    // which overshoots the root by less than `n - 1` units (`Δ < 2^64`, `s ≥ 2^127`): with `N =
    // (R_top - S_hi^n)·2^64 + R[n - 1]` and `D = n·S_hi^(n-1)`, the root lies in `[S_hi·2^64 + N/D
    // - n + 1, S_hi·2^64 + (N + 1)/D]`. `N/D` is estimated from the top 64 bits of `D` and the
    // bits of `N` from there, `q` with `Q = floor(N/D) ∈ [q - 2, q + 1]` (the top bits of `D` are
    // within one unit, of `2^63` or more, and `q < 2^64`), so the root `F` lies in `[S - n - 2, S +
    // 2]` for `S = S_hi·2^64 + q`; the tail of `M` does not change `F`, nor a decided rounding.
    let (rad, _) = radicand::<LEN2>(x.m, r, 2 * n);
    let top = &rad[n..2 * n];
    let s = estimate(top, n)?;
    let (mut pm, mut p) = ([0u64; LEN], [0u64; LEN]);
    let sh = floor_root(top, n, s, &mut pm, &mut p);
    let mut num = [0u64; LEN + 1];
    num[1..=n].copy_from_slice(&sub(top, &p[..n])[..n]);
    num[0] = rad[n - 1];
    // `D = n·S_hi^(n-1)`, `n` limbs at most.
    let mut d = [0u64; LEN];
    let mut carry = 0u64;
    for i in 0..n - 1 {
        let t = u128::from(pm[i]) * n as u128 + u128::from(carry);
        d[i] = t as u64;
        carry = (t >> 64) as u64;
    }
    d[n - 1] = carry;
    let bd = bits(&d[..n]);
    let sft = i128::from(bd) - 64;
    let d1 = limbs::extract(&d[..n], sft);
    let n1 = u128::from(limbs::extract(&num[..=n], sft + 64)) << 64
        | u128::from(limbs::extract(&num[..=n], sft));
    if bits(&num[..=n]) > bd + 64 || (n1 >> 64) as u64 >= d1 {
        return None;
    }
    let (qe, _) = nat::div_2by1((n1 >> 64) as u64, n1 as u64, d1, nat::invert_limb(d1));
    let root = u128::from(sh) << 64 | u128::from(qe);
    let k = 127 - prec;
    if decided(root & ((1 << k) - 1), k, n as u128 + 3, 2) {
        return Some(round_top2(negative, root, 0, q, true, prec, round));
    }
    None
}

/// Whether `2^n·R > (2F + 1)^n` for the radicand `R` of `n` limbs.
#[inline]
fn above_half(r: &[u64], n: usize, f: u64) -> bool {
    // `(2F + 1)^n`, of `n + 1` limbs, by products in place by the two limbs `b0 + 2^64` of `b = 2F
    // + 1` (`F ≥ 2^63`): `p·b0` plus `p` a limb up.
    let b0 = f << 1 | 1;
    let mut p = [0u64; LEN + 1];
    p[0] = b0;
    p[1] = 1;
    for len in 2..=n {
        // `p` has `len` limbs; the product `len + 1`.
        let mut carry = 0u64;
        let mut prev = 0u64;
        for v in p[..=len].iter_mut() {
            let cur = *v;
            let t = u128::from(cur) * u128::from(b0) + u128::from(prev) + u128::from(carry);
            *v = t as u64;
            carry = (t >> 64) as u64;
            prev = cur;
        }
    }
    // `2^n·R`, `n + 1` limbs.
    let mut s = [0u64; LEN + 1];
    for (i, v) in s[..=n].iter_mut().enumerate() {
        let lo = if i < n { r[i] << n } else { 0 };
        let hi = if i > 0 { r[i - 1] >> (64 - n) } else { 0 };
        *v = lo | hi;
    }
    cmp(&s[..=n], &p[..=n]).is_gt()
}

//! Integer powers at short precisions on numbers of a fixed number of limbs, in arrays: the binary
//! powering of [`powi_round`](super::BigFloat::powi_round) beyond the 128 bits of its register
//! path, with every product cut to its top `W` limbs, aligned at the top bit, and the rounding
//! decided from the bound of the error, without a vector until the result.
//!
//! A value `v > 0` is known from below, `c ≤ v < c·(1 + ε)^k` for `c = 0.C·2^e`, `C` of `W` limbs
//! with its top bit set, and `ε = 2^(1 - 64W)`: a product of two such numbers has its top bit at
//! one of the two top places of its `2W` limbs, and keeping the top `64W` bits from there drops
//! less than one unit of a value of at least `2^(64W - 1)` units. A square takes `k` to `2k + 1`,
//! a product by a base of the bound `kb` to `k + kb + 1`, so that the power by `n` has `k ≤ (kb +
//! 2)·n`, as for the products cut to whole limbs of the general path; and `c(1 + ε)^k - c ≤
//! 2kε·c < 4k` units of the lowest bit of `C` for `kε ≤ 1`. The value is neither a number of the
//! precision nor halfway between two (the caller sees to that), so it rounds as `C` with a sticky
//! bit when every integer of `[C, C + 4k]` does ([`certain`]).
//!
//! Up to [`MAX_W`] limbs the numbers are arrays and the products unrolled ([`pow`]); beyond, or for
//! a reciprocal of a mantissa of more than two limbs, vectors and the crate's products
//! ([`pow_vec`]), which Ziv's strategy calls again on more limbs when the bound does not decide.

use super::general::Op;
use super::limbs::{self, Work};
use super::round::round_into;
use super::{Kind, Round};
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// Whether every integer from `c` to `c + hi` rounds to `prec` bits as `c` with a sticky bit does,
/// for `c` of at least `prec + 2` bits: `c` has a bit set below its round bit, and adding `hi`
/// carries nothing into the round bit, so that all of them have the same bits from the round bit
/// up and some bit set below it. A value known to lie in `[c, c + hi]` and to be neither a number
/// of the precision nor halfway then rounds as `c` with a sticky bit.
#[inline]
pub(super) fn certain(c: &[u64], hi: u64, prec: u64) -> bool {
    let bits = limbs::bit_len(c);
    if bits < prec + 2 {
        return false;
    }
    // The bits below the round bit: whole limbs, and `rb` bits of the limb `k`.
    let r = bits - prec - 1;
    let (k, rb) = ((r / 64) as usize, (r % 64) as u32);
    let mask = (1u64 << rb) - 1;
    let below = c[..k].iter().any(|&v| v != 0) || c[k] & mask != 0;
    if !below {
        return false;
    }
    // `c` plus `hi`: the carry out of the low limb runs through limbs of ones only.
    if k == 0 {
        return (c[0] & mask).checked_add(hi).is_some_and(|v| v <= mask);
    }
    let (_, carry) = c[0].overflowing_add(hi);
    !carry || c[1..k].iter().any(|&v| v != u64::MAX) || c[k] & mask != mask
}

/// The most limbs of the fixed-width path: results of up to about 480 bits.
pub(super) const MAX_W: usize = 8;

/// `a·b` of `W` limbs each, `D = 2W` limbs, by the schoolbook method.
#[inline(always)]
fn mul<const W: usize, const D: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; D] {
    let mut r = [0u64; D];
    for i in 0..W {
        let mut carry = 0u64;
        for j in 0..W {
            let t = u128::from(a[i]) * u128::from(b[j]) + u128::from(r[i + j]) + u128::from(carry);
            r[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        r[i + W] = carry;
    }
    r
}

/// `a^2` of `W` limbs, `D = 2W` limbs: the products of two different limbs once, doubled, and the
/// squares of the limbs.
#[inline(always)]
fn sqr<const W: usize, const D: usize>(a: &[u64; W]) -> [u64; D] {
    let mut r = [0u64; D];
    for i in 0..W {
        let mut carry = 0u64;
        for j in i + 1..W {
            let t = u128::from(a[i]) * u128::from(a[j]) + u128::from(r[i + j]) + u128::from(carry);
            r[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        r[i + W] = carry;
    }
    // Doubled, and the squares added.
    let mut top = 0u64;
    for v in r.iter_mut() {
        let next = *v >> 63;
        *v = *v << 1 | top;
        top = next;
    }
    let mut carry = 0u128;
    for i in 0..W {
        let s = u128::from(a[i]) * u128::from(a[i]);
        let lo = u128::from(r[2 * i]) + (s as u64 as u128) + carry;
        r[2 * i] = lo as u64;
        let hi = u128::from(r[2 * i + 1]) + (s >> 64) + (lo >> 64);
        r[2 * i + 1] = hi as u64;
        carry = hi >> 64;
    }
    r
}

/// The top `W` limbs of the product `p` of two numbers with their top bits set, aligned at its top
/// bit, and the exponent `e` of `0.p·2^e` moved by the shift.
#[inline(always)]
fn top<const W: usize, const D: usize>(p: &[u64; D], e: i128) -> ([u64; W], i128) {
    let mut c = [0u64; W];
    if p[D - 1] >> 63 == 1 {
        c.copy_from_slice(&p[W..]);
        (c, e)
    } else {
        for i in 0..W {
            c[i] = p[W + i] << 1 | p[W + i - 1] >> 63;
        }
        (c, e - 1)
    }
}

/// `|x|^(±n)` of the sign `negative` to `prec` bits on `W` limbs, when the bound decides it; `None`
/// otherwise, or when the base cannot be had here (a reciprocal of a mantissa of more than `W`
/// limbs).
///
/// The base: the top `W` limbs of the mantissa (exact for a mantissa of at most `W` limbs, `kb =
/// 0`, a cut otherwise, `kb = 1`), or for `n < 0` the reciprocal `floor(2^(64W + 64l - 1) / M)` of
/// the mantissa `M` of `l ≤ 2` limbs, which is not a power of two: of `W` limbs with its top bit
/// set, below `1/M` by less than one unit, `kb = 1`.
#[inline(always)]
fn pow_w<const W: usize, const D: usize>(
    x: Op<'_>,
    negative: bool,
    n_negative: bool,
    n: u64,
    prec: u64,
    round: Round,
) -> Option<(Kind, Ordering)> {
    let mut base = [0u64; W];
    let (eb, kb) = if n_negative {
        // `1/x = 2^(64l - exp) / M`, so `0.q·2^(1 - exp)` for the quotient `q` of `2^(64W + 64l - 1)`.
        match *x.m {
            [m] => {
                let v = nat::invert_limb(m);
                let mut r = 1u64 << 63;
                for b in base.iter_mut().rev() {
                    let (q, rem) = nat::div_2by1(r, 0, m, v);
                    *b = q;
                    r = rem;
                }
            }
            [d0, d1] => {
                let v = nat::invert_3by2(d1, d0);
                let mut u = [0u64; 10];
                u[W + 1] = 1 << 63;
                nat::div_qr_2(&mut base, &u[..W + 2], d1, d0, v);
            }
            _ => return None,
        }
        (1 - x.exp, 1u64)
    } else {
        let l = x.m.len();
        if l >= W {
            base.copy_from_slice(&x.m[l - W..]);
        } else {
            base[W - l..].copy_from_slice(x.m);
        }
        (
            x.exp,
            u64::from(x.m[..l.saturating_sub(W)].iter().any(|&v| v != 0)),
        )
    };
    debug_assert!(base[W - 1] >> 63 == 1);
    let mut acc = base;
    let mut e = eb;
    for i in (0..63 - n.leading_zeros()).rev() {
        (acc, e) = top::<W, D>(&sqr::<W, D>(&acc), 2 * e);
        if n >> i & 1 == 1 {
            (acc, e) = top::<W, D>(&mul::<W, D>(&acc, &base), e + eb);
        }
    }
    let k = (kb + 2).checked_mul(n)?.checked_mul(4)?;
    if !certain(&acc, k, prec) {
        return None;
    }
    Some(round_into(
        negative,
        &acc,
        e - 64 * W as i128,
        true,
        prec,
        round,
        Vec::new(),
    ))
}

/// The number of limbs of the fixed-width path for a result of `prec` bits and a power by `n`,
/// with a bound `kb ≤ 1` of the base: enough for the round bit, the `log2(4k)` bits of the error
/// and 12 more, so that the bound rarely leaves the rounding undecided; `None` beyond [`MAX_W`].
#[inline]
pub(super) fn width(prec: u64, n: u64) -> Option<usize> {
    // `4k ≤ 12n < 2^(bits(n) + 4)`.
    let err = u64::from(64 - n.leading_zeros()) + 4;
    let w = (prec + 1 + err + 12).div_ceil(64) as usize;
    (w <= MAX_W).then_some(w.max(2))
}

/// `|x|^(±n)` of the sign `negative` to `prec` bits on [`width`] limbs, when the bound decides it;
/// `None` otherwise. `x` is finite and not zero, `|x|^n` neither a number of the precision nor
/// halfway between two, and within the exponent range's reach (see [`beyond`](super::power)).
pub(super) fn pow(
    x: Op<'_>,
    negative: bool,
    n_negative: bool,
    n: u64,
    prec: u64,
    round: Round,
) -> Option<(Kind, Ordering)> {
    let a = (x, negative, n_negative, n, prec, round);
    match width(prec, n)? {
        2 => pow_w::<2, 4>(a.0, a.1, a.2, a.3, a.4, a.5),
        3 => pow_w::<3, 6>(a.0, a.1, a.2, a.3, a.4, a.5),
        4 => pow_w::<4, 8>(a.0, a.1, a.2, a.3, a.4, a.5),
        5 => pow_w::<5, 10>(a.0, a.1, a.2, a.3, a.4, a.5),
        6 => pow_w::<6, 12>(a.0, a.1, a.2, a.3, a.4, a.5),
        7 => pow_w::<7, 14>(a.0, a.1, a.2, a.3, a.4, a.5),
        _ => pow_w::<8, 16>(a.0, a.1, a.2, a.3, a.4, a.5),
    }
}

/// The top `wl` limbs of the product `t` (of `2wl` limbs) of two numbers of `wl` limbs with their
/// top bits set, aligned at its top bit, into `acc`: whether it was shifted by one bit.
#[inline]
fn align(t: &[u64], acc: &mut Vec<u64>, wl: usize) -> bool {
    debug_assert_eq!(t.len(), 2 * wl);
    acc.clear();
    if t[2 * wl - 1] >> 63 == 1 {
        acc.extend_from_slice(&t[wl..]);
        false
    } else {
        acc.extend((wl..2 * wl).map(|i| t[i] << 1 | t[i - 1] >> 63));
        true
    }
}

/// `|x|^(±n)` of the sign `negative` to `prec` bits on `wl ≥ 2` limbs (`n` in limbs, not zero),
/// with the vectors of `w`, when the bound decides it; `None` otherwise. As [`pow_w`], with the
/// products of the crate's integers; the reciprocal for `n < 0` is `floor(2^(64wl + 64l - 1) /
/// M)` for the mantissa `M` of `l ≤ wl + 1` limbs (`kb = 1`), or for a longer one of its top `wl +
/// 1` limbs and one unit more when the others are not all zero (`D ≥ M`, at most `1 + 2^(-64wl)`
/// times it, so `kb = 2`). The error of `4k ≤ 4(kb + 2)·n < 2^(bits(n) + 4)` units is read on the
/// limbs left when as many are dropped as keep it below `2^60` of their units.
#[allow(clippy::too_many_arguments)]
pub(super) fn pow_vec(
    x: Op<'_>,
    negative: bool,
    n_negative: bool,
    n: &[u64],
    wl: usize,
    prec: u64,
    round: Round,
    w: &mut Work,
) -> Option<(Kind, Ordering)> {
    let Work {
        a: base,
        b: acc,
        c: t,
        scratch,
    } = w;
    let l = x.m.len();
    base.clear();
    let (eb, kb) = if n_negative {
        let (dl, up) = if l <= wl + 1 {
            (l, false)
        } else {
            (wl + 1, x.m[..l - wl - 1].iter().any(|&v| v != 0))
        };
        acc.clear();
        acc.extend_from_slice(&x.m[l - dl..]);
        if up && nat::add_assign(acc, &[1]) {
            // `2^(64(wl + 1))`: rare enough for the general path.
            return None;
        }
        t.clear();
        t.resize(wl + dl, 0);
        t[wl + dl - 1] = 1 << 63;
        base.resize(wl, 0);
        let high = nat::div_qr(base, t, acc, scratch);
        if high || base[wl - 1] >> 63 == 0 {
            // A divisor of `2^(64dl - 1)`: a power of two, which the caller has taken.
            return None;
        }
        (1 - x.exp, if up { 2u64 } else { 1 })
    } else {
        if l >= wl {
            base.extend_from_slice(&x.m[l - wl..]);
        } else {
            base.resize(wl - l, 0);
            base.extend_from_slice(x.m);
        }
        (
            x.exp,
            u64::from(x.m[..l.saturating_sub(wl)].iter().any(|&v| v != 0)),
        )
    };
    acc.clear();
    acc.extend_from_slice(base);
    let mut e = eb;
    let bits = limbs::bit_len(n);
    for i in (0..bits - 1).rev() {
        nat::sqr_into(t, acc, scratch);
        e = 2 * e - i128::from(align(t, acc, wl));
        if limbs::bit(n, i) {
            nat::mul_into(t, acc, base, scratch);
            e = e + eb - i128::from(align(t, acc, wl));
        }
    }
    // The error, `4(kb + 2)·n` units exactly for a short `n`, below `2^(bits + 4)` otherwise.
    let (d, hi) = match *n {
        [v] if bits <= 58 => (0, 4 * (kb + 2) * v),
        _ => {
            let d = (bits + 4 - 60).div_ceil(64) as usize;
            (d, (1u64 << (bits + 4 - 64 * d as u64)) + 1)
        }
    };
    if d + 2 > wl || !certain(&acc[d..], hi, prec) {
        return None;
    }
    Some(round_into(
        negative,
        &acc[d..],
        e - 64 * (wl - d) as i128,
        true,
        prec,
        round,
        Vec::new(),
    ))
}

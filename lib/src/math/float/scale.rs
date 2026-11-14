//! Products by powers of a radix `b`, exact or by Ziv's strategy, for the conversions between a
//! binary float and its digits in radix `b`: the integer that `M·2^s·b^k` rounds to in a
//! direction ([`round_int`], for writing), and the float of a precision that `D·b^k` rounds to
//! ([`round_value`], for reading).
//!
//! Both compute exactly when the result may be a boundary of the rounding (an integer or half an
//! integer for [`round_int`], a number of the precision or the midpoint of two for
//! [`round_value`]) and when the exact work is no larger than the operands and the result; they
//! use Ziv's strategy otherwise, with the power `b^k` approximated, which is the only way for the
//! exponents of `±2^62` that a float may have. With `b = 2^a·c`, `c` odd and above one:
//!
//! - `y = M·2^s·c^k·2^(a·k)` for `k ≥ 0` is a half-integer only when `2y ≥ M_odd·c^k`, `M_odd`
//!   the odd part of `M`, of `q` bits: when `q - 1 + k·log2(c) < log2(2y)`. For `k = -j < 0` it
//!   is one only when `c^j` divides `M_odd`, so `j·log2(c) < q`, and then
//!   `2y ≥ M_odd/c^j`, so `q - 1 - j·log2(c) < log2(2y)`.
//! - `v = D·c^k·2^(a·k)` is a number of `p` bits or the midpoint of two only when its odd part
//!   has at most `p + 1` bits: for `k ≥ 0` when `q - 1 + k·log2(c) < p + 1`; for `k = -j < 0`
//!   when `c^j` divides `D_odd` (`j·log2(c) < q`) and `q - 1 - j·log2(c) < p + 1`.
//!
//! Outside those bounds the exact value is never a boundary, so Ziv's loop ends: the interval
//! of its approximation shrinks with the working precision until it holds no boundary.
//!
//! The approximations are lower bounds truncated to `w` bits, each truncation of a number of at
//! least `w` bits losing less than `u = 2^(1-w)` of it relatively; a value that `K` truncations
//! made, `K·u ≤ 1`, lies within `[A, A·(1 + u)^K] ⊂ [A, A·(1 + 2Ku)]` of the result `A`.

use super::arith::div_mags;
use super::limbs;
use super::round::{overflow, round_mag, underflow};
use super::ziv::{Approx, round_approx};
use super::{BigFloat, Round};
use crate::math::numbers::ByteOrder;
use core::cmp::Ordering;

/// `floor(log_b(2)·2^128)` for the radix `b` (index `b`), from 3 to 36; 2 is never read.
const LOG_B_2: [u128; 37] = [
    0,
    0,
    u128::MAX,
    0xa1849cc1a9a9e94e043eaf7791f52142,
    0x80000000000000000000000000000000,
    0x6e40d1a4143dcb9433d522368f0d1d89,
    0x6308c91b702a7cf4ff85a5c1b80aaa91,
    0x5b3064eb3aa6d3889bd82cc11a7209d2,
    0x55555555555555555555555555555555,
    0x50c24e60d4d4f4a7021f57bbc8fa90a1,
    0x4d104d427de7fbcc47c4acd605be48bc,
    0x4a00270775914e8870b466920e51e1f7,
    0x4768ce0d05818e127f122e2f4c79f9ca,
    0x452e53e365907bda2bf75000cfb72251,
    0x433cfffb4b5aae55c2d2e89586d2b763,
    0x41867711b4f8535537bbdca4fca609de,
    0x40000000000000000000000000000000,
    0x3ea16afd58b10966e1c51ddbeac65f02,
    0x3d64598d154dc4de0da34544e21084a1,
    0x3c43c23018bb55630369e97d641961e5,
    0x3b3b9a42873069c702cceaea8207233f,
    0x3a4898f06cf41ac990409adae68a5d43,
    0x39680b13582e7c1876f62d7317e2d8bd,
    0x3897b2b751ae561ab0f3e4b3bda6639c,
    0x37d5aed131f19c98cd9850af9a126d7e,
    0x372068d20a1ee5ca19ea911b47868ec4,
    0x3676867e5d60de291912e33748b4029f,
    0x35d6deeb388df86f56bf8fd285fc606b,
    0x354071d61c77fa2e37ac410062da9305,
    0x34b260c5671b18acf3315689e7fc958f,
    0x342be986572b45cc8d5dad3f1f35ccc3,
    0x33ac61b998fbbdf2b55bac355a82ee98,
    0x33333333333333333333333333333333,
    0x32bfd90114c12861c220c028e9dbc15a,
    0x3251dcf6169e45f2bed2f23982c11654,
    0x31e8d59f180dc6309a55d658e0cac095,
    0x3184648db8153e7a7fc2d2e0dc055548,
];

/// `floor(log2(b)·2^58)` for the radix `b` (index `b`), from 2 to 36; exact for the powers of 2.
const LOG2_B: [u64; 37] = [
    0,
    0,
    0x0400000000000000,
    0x06570068e7ef5a1e,
    0x0800000000000000,
    0x0949a784bcd1b8af,
    0x0a570068e7ef5a1e,
    0x0b3abb3faa02166c,
    0x0c00000000000000,
    0x0cae00d1cfdeb43c,
    0x0d49a784bcd1b8af,
    0x0dd6753e032ea0ef,
    0x0e570068e7ef5a1e,
    0x0ecd4011c8f11979,
    0x0f3abb3faa02166c,
    0x0fa0a7eda4c112ce,
    0x1000000000000000,
    0x10598fdbeb244c59,
    0x10ae00d1cfdeb43c,
    0x10fde0b5c8134051,
    0x1149a784bcd1b8af,
    0x1191bba891f1708b,
    0x11d6753e032ea0ef,
    0x121820a01ac754cb,
    0x12570068e7ef5a1e,
    0x12934f0979a3715f,
    0x12cd4011c8f11979,
    0x1305013ab7ce0e5b,
    0x133abb3faa02166c,
    0x136e9291eaa65b49,
    0x13a0a7eda4c112ce,
    0x13d118d66c4d4e55,
    0x1400000000000000,
    0x142d75a6eb1dfb0e,
    0x14598fdbeb244c59,
    0x148462c466d3cf1c,
    0x14ae00d1cfdeb43c,
];

/// The radix `b` as `2^a·c`, `c` odd.
#[inline]
pub(crate) fn split(b: u32) -> (u32, u64) {
    let a = b.trailing_zeros();
    (a, u64::from(b >> a))
}

/// `floor(e·log_b(2))` for `|e| < 2^64` and the radix `b` from 2 to 36, and whether it is
/// certain: when it is not, the floor is the value or one more. Exact for the powers of two.
pub(crate) fn floor_log(e: i128, b: u32) -> (i128, bool) {
    let (a, c) = split(b);
    if c == 1 {
        return (e.div_euclid(i128::from(a)), true);
    }
    if e == 0 {
        return (0, true);
    }
    let u = e.unsigned_abs();
    debug_assert!(u >> 64 == 0);
    let l = LOG_B_2[b as usize];
    let (h, lo) = (l >> 64, l & u128::from(u64::MAX));
    // `u·l = hi·2^64 + low`, the integer part above bit 128 and the fraction below; the true
    // `u·log_b(2)·2^128` lies in `[u·l, u·l + u)`, as the table rounds down by less than one.
    let low = u * lo;
    let hi = u * h + (low >> 64);
    let int = (hi >> 64) as i128;
    let frac = (hi << 64) | (low & u128::from(u64::MAX));
    let certain = frac <= u128::MAX - u;
    if e > 0 {
        (int, certain)
    } else if certain {
        // `-(int + θ)` for `0 < θ < 1`: `log_b(2)` is irrational, so `θ` is not zero.
        (-int - 1, true)
    } else {
        (-int - 2, false)
    }
}

/// Bounds `(lo, hi)` of `k·log2(b)`, `lo ≤ k·log2(b) ≤ hi`, for `0 ≤ k < 2^66`: from the table
/// of `log2(b)` to 58 bits, which rounds down by less than `2^-58`.
#[inline]
pub(crate) fn log2_pow(b: u32, k: u128) -> (i128, i128) {
    debug_assert!(k >> 66 == 0);
    let prod = k * u128::from(LOG2_B[b as usize]);
    let lo = (prod >> 58) as i128;
    (lo, lo + (k >> 58) as i128 + 1)
}

/// The number of bits of `n`.
#[inline]
pub(crate) fn bitlen(n: u128) -> u64 {
    u64::from(128 - n.leading_zeros())
}

/// `v·y` in place.
pub(crate) fn mul_1(v: &mut Vec<u64>, y: u64) {
    let mut carry = 0u64;
    for x in v.iter_mut() {
        let t = u128::from(*x) * u128::from(y) + u128::from(carry);
        *x = t as u64;
        carry = (t >> 64) as u64;
    }
    if carry != 0 {
        v.push(carry);
    }
}

/// The largest power `c^t` of the odd `c > 1` below `2^64`, and `t`.
#[inline]
fn word_power(c: u64) -> (u64, u64) {
    let (mut p, mut t) = (c, 1);
    while let Some(q) = p.checked_mul(c) {
        p = q;
        t += 1;
    }
    (p, t)
}

/// `c^k` exactly, for the odd `c > 1`: by squares of `c^t`, the largest power in a word, from the
/// top bit of `k/t`, then `c^(k mod t)`.
pub(crate) fn pow_exact(c: u64, k: u64) -> Vec<u64> {
    let (base, t) = word_power(c);
    let (q, r) = (k / t, k % t);
    let mut v = vec![1u64];
    if q > 0 {
        v = vec![base];
        let mut scratch = Vec::new();
        let mut sq = Vec::new();
        for i in (0..63 - q.leading_zeros()).rev() {
            crate::math::numbers::nat::sqr_into(&mut sq, &v, &mut scratch);
            limbs::trim(&mut sq);
            core::mem::swap(&mut v, &mut sq);
            if q >> i & 1 == 1 {
                mul_1(&mut v, base);
            }
        }
    }
    if r > 0 {
        mul_1(&mut v, c.pow(r as u32));
    }
    v
}

/// Truncates `v` to its top `w` bits, when longer, and returns the bits dropped.
#[inline]
fn truncate(v: &mut Vec<u64>, w: u64) -> u64 {
    let bits = limbs::bit_len(v);
    if bits <= w {
        return 0;
    }
    let d = bits - w;
    // In place: the limbs from `k` on, shifted down by `b` bits.
    let (k, b) = ((d / 64) as usize, (d % 64) as u32);
    let n = v.len();
    if b == 0 {
        v.copy_within(k.., 0);
    } else {
        for i in k..n {
            let hi = if i + 1 < n { v[i + 1] << (64 - b) } else { 0 };
            v[i - k] = v[i] >> b | hi;
        }
    }
    v.truncate(n - k);
    limbs::trim(v);
    d
}

/// A lower bound `P·2^s` of `c^k`, for the odd `c > 1`, with `P` of at most `w` bits and a count
/// `K` of truncations, `c^k ≤ P·2^s·(1 + 2^(1-w))^K`: by squares of `c^t` from the top bit of
/// `k/t`, each square and product truncated to `w` bits. A square of a bound of `K` truncations is
/// one of `2K`, and `2K + 1` truncated; a product by the exact `c^t`, `K + 1`.
pub(crate) fn pow_approx(c: u64, k: u128, w: u64) -> (Vec<u64>, i128, u128) {
    let (base, t) = word_power(c);
    let (q, r) = (k / u128::from(t), (k % u128::from(t)) as u32);
    let (mut v, mut s, mut count) = (vec![1u64], 0i128, 0u128);
    let mut scratch = Vec::new();
    let mut sq = Vec::new();
    if q > 0 {
        v = vec![base];
        for i in (0..127 - q.leading_zeros()).rev() {
            crate::math::numbers::nat::sqr_into(&mut sq, &v, &mut scratch);
            limbs::trim(&mut sq);
            core::mem::swap(&mut v, &mut sq);
            s *= 2;
            count = count.saturating_mul(2);
            let d = truncate(&mut v, w);
            if d > 0 {
                s += i128::from(d);
                count = count.saturating_add(1);
            }
            if q >> i & 1 == 1 {
                mul_1(&mut v, base);
                let d = truncate(&mut v, w);
                if d > 0 {
                    s += i128::from(d);
                    count = count.saturating_add(1);
                }
            }
        }
    }
    if r > 0 {
        mul_1(&mut v, c.pow(r));
        let d = truncate(&mut v, w);
        if d > 0 {
            s += i128::from(d);
            count = count.saturating_add(1);
        }
    }
    (v, s, count)
}

/// A direction of rounding of a magnitude to an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dir {
    /// Down, toward zero.
    Floor,
    /// Up, away from zero.
    Ceil,
    /// To the nearest, ties to the even integer.
    Nearest,
    /// To the nearest, ties away from zero.
    Away,
}

impl Dir {
    /// The direction of the magnitude of a number of the sign `negative` rounded in `round`.
    #[inline]
    pub(crate) fn of(negative: bool, round: Round) -> Dir {
        match round {
            Round::Nearest => Dir::Nearest,
            Round::TowardZero => Dir::Floor,
            Round::Up if negative => Dir::Floor,
            Round::Up => Dir::Ceil,
            Round::Down if negative => Dir::Ceil,
            Round::Down => Dir::Floor,
        }
    }
}

/// The integer part of `m·2^s` and the order of its fractional part to one half, and whether the
/// fractional part is zero.
pub(crate) fn int_parts(m: &[u64], s: i128) -> (Vec<u64>, Ordering, bool) {
    if m.is_empty() {
        return (Vec::new(), Ordering::Less, true);
    }
    if s >= 0 {
        return (limbs::shl(m, s as u64), Ordering::Less, true);
    }
    let bits = limbs::bit_len(m);
    let d = (-s).min(i128::from(bits) + 2) as u64;
    let t = limbs::shr(m, d);
    let half = limbs::bit(m, d - 1);
    let rest = !limbs::zero_below(m, d - 1);
    let order = match (half, rest) {
        (false, _) => Ordering::Less,
        (true, false) => Ordering::Equal,
        (true, true) => Ordering::Greater,
    };
    (t, order, !half && !rest)
}

/// The integer `t` or `t + 1` that a magnitude of the integer part `t`, its fractional part of the
/// order `half` to one half and zero when `zero`, rounds to in the direction `dir`, and the
/// direction of that rounding.
pub(crate) fn decide(
    mut t: Vec<u64>,
    half: Ordering,
    zero: bool,
    dir: Dir,
) -> (Vec<u64>, Ordering) {
    let up = match dir {
        Dir::Floor => false,
        Dir::Ceil => !zero,
        Dir::Nearest => {
            half == Ordering::Greater
                || half == Ordering::Equal && t.first().is_some_and(|&x| x & 1 == 1)
        }
        Dir::Away => half != Ordering::Less,
    };
    if zero {
        return (t, Ordering::Equal);
    }
    if up {
        limbs::increment(&mut t);
        (t, Ordering::Greater)
    } else {
        (t, Ordering::Less)
    }
}

/// `v /= d` for a limb `d` not zero, in place, without zero limbs at the top: the remainder.
pub(crate) fn div_1(v: &mut Vec<u64>, d: u64) -> u64 {
    let mut r = 0u64;
    for x in v.iter_mut().rev() {
        let t = u128::from(r) << 64 | u128::from(*x);
        *x = (t / u128::from(d)) as u64;
        r = (t % u128::from(d)) as u64;
    }
    limbs::trim(v);
    r
}

/// The longest dividend that [`div_rem`] divides by [`schoolbook`] rather than through `Number`.
const SCHOOLBOOK: usize = 40;

/// `(a / b, a mod b)` by Knuth's Algorithm D (TAOCP 4.3.1), for `a ≥ b` and `b` of at least two
/// limbs: the divisor normalized to its top bit, each quotient limb estimated from the top two
/// limbs of the remainder by the top limb of the divisor, corrected by the second (at most twice),
/// then the product subtracted and, when it borrows, the divisor added back once.
fn schoolbook(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let n = b.len();
    let sh = b[n - 1].leading_zeros();
    let v = limbs::shl(b, u64::from(sh));
    let mut u = limbs::shl(a, u64::from(sh));
    u.resize(a.len() + 1, 0);
    let m = u.len() - n;
    let mut q = vec![0u64; m];
    let (v1, v0) = (u128::from(v[n - 1]), u128::from(v[n - 2]));
    for j in (0..m).rev() {
        let top = u128::from(u[j + n]) << 64 | u128::from(u[j + n - 1]);
        let mut qh = top / v1;
        let mut rh = top % v1;
        while qh >> 64 != 0 || qh * v0 > (rh << 64 | u128::from(u[j + n - 2])) {
            qh -= 1;
            rh += v1;
            if rh >> 64 != 0 {
                break;
            }
        }
        // `u[j..=j+n] -= qh·v`.
        let (mut borrow, mut carry) = (0u64, 0u64);
        for i in 0..n {
            let p = qh * u128::from(v[i]) + u128::from(carry);
            carry = (p >> 64) as u64;
            let (d, b1) = u[j + i].overflowing_sub(p as u64);
            let (d, b2) = d.overflowing_sub(borrow);
            u[j + i] = d;
            borrow = u64::from(b1) + u64::from(b2);
        }
        let (d, b1) = u[j + n].overflowing_sub(carry);
        let (d, b2) = d.overflowing_sub(borrow);
        u[j + n] = d;
        if b1 || b2 {
            // One too many: add the divisor back.
            qh -= 1;
            let mut c = false;
            for i in 0..n {
                let (s1, c1) = u[j + i].overflowing_add(v[i]);
                let (s2, c2) = s1.overflowing_add(u64::from(c));
                u[j + i] = s2;
                c = c1 || c2;
            }
            u[j + n] = u[j + n].wrapping_add(u64::from(c));
        }
        q[j] = qh as u64;
    }
    limbs::trim(&mut q);
    u.truncate(n);
    let mut r = limbs::shr(&u, u64::from(sh));
    limbs::trim(&mut r);
    (q, r)
}

/// The top `w` bits of `m` (all of it when shorter), `floor(M/2^s)`, and the bits `s` dropped.
#[inline]
fn top_bits(m: &[u64], w: u64) -> (Vec<u64>, u64) {
    let bits = limbs::bit_len(m);
    if bits <= w {
        return (m.to_vec(), 0);
    }
    (limbs::shr(m, bits - w), bits - w)
}

/// The order of the remainder `r` to half the divisor `d`.
fn half_order(r: &[u64], d: &[u64]) -> Ordering {
    if r.is_empty() {
        return Ordering::Less;
    }
    limbs::cmp(&limbs::shl(r, 1), d)
}

/// `(a / b, a mod b)` of magnitudes, `b` not zero.
pub(crate) fn div_rem(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    if limbs::cmp(a, b) == Ordering::Less {
        return (Vec::new(), a.to_vec());
    }
    if b.len() == 1 {
        let mut q = a.to_vec();
        let r = div_1(&mut q, b[0]);
        return (q, if r == 0 { Vec::new() } else { vec![r] });
    }
    if a.len() <= SCHOOLBOOK {
        return schoolbook(a, b);
    }
    let n = limbs::to_number(a, false, ByteOrder::LittleEndian);
    let d = limbs::to_number(b, false, ByteOrder::LittleEndian);
    let (q, r) = n.div_rem(&d);
    (limbs::of_number(&q), limbs::of_number(&r))
}

/// The integer that `y = M·2^lsb·b^k` rounds to in the direction `dir`, for the magnitude `m`
/// (not zero, without zero limbs at the top) and the radix `b` from 2 to 36, and the direction
/// of the rounding. Exact when `y` may be a half-integer or the exact work is small; by Ziv's
/// strategy otherwise (see the [module](self)).
pub(crate) fn round_int(m: &[u64], lsb: i128, b: u32, k: i128, dir: Dir) -> (Vec<u64>, Ordering) {
    debug_assert!(!m.is_empty() && m.last() != Some(&0));
    let (a, c) = split(b);
    if c == 1 {
        let (t, h, z) = int_parts(m, lsb + i128::from(a) * k);
        return decide(t, h, z, dir);
    }
    let bits = i128::from(limbs::bit_len(m));
    let j = k.unsigned_abs();
    let (plo, phi) = log2_pow(b, j);
    // `y < 2^yhi`.
    let yhi = if k >= 0 {
        bits + lsb + phi
    } else {
        bits + lsb - plo
    };
    if yhi <= -2 {
        // Below a quarter: zero, or one rounding up.
        return decide(Vec::new(), Ordering::Less, false, dir);
    }
    let q = bits - i128::from(limbs::trailing_zeros(m));
    let (clo, chi) = log2_pow(c as u32, j);
    let boundary = if k >= 0 {
        q + clo < yhi + 2
    } else {
        clo < q && q < yhi + 2 + chi
    };
    let cheap = bits + phi <= 2 * (yhi + 128);
    if boundary || cheap {
        if k >= 0 {
            let n = limbs::mul(m, &pow_exact(c, j as u64));
            let (t, h, z) = int_parts(&n, lsb + i128::from(a) * k);
            return decide(t, h, z, dir);
        }
        let p = pow_exact(c, j as u64);
        let s = lsb - i128::from(a) * j as i128;
        let (num, den) = if s >= 0 {
            (limbs::shl(m, s as u64), p)
        } else {
            (m.to_vec(), limbs::shl(&p, (-s) as u64))
        };
        let (t, r) = div_rem(&num, &den);
        let z = r.is_empty();
        return decide(t, half_order(&r, &den), z, dir);
    }
    // Ziv: `y` is no half-integer. The working precision covers the integer part and the count of
    // truncations, `K ≤ 2·|k| + 3 < 2^67`, so that `K·u ≤ 1` and the bounds below hold.
    let mut w = yhi as u64 + 64 + 2 * bitlen(j);
    loop {
        let (x, sx) = top_bits(m, w);
        let (p, sp, count) = pow_approx(c, j, w);
        let (lo, hi, s) = if k >= 0 {
            // `y ∈ [A, A·(1 + u)^K]·2^s` with `K = count + 2` (one more for `x`, one for the
            // product), within `[A, A + 4K]` as `A < 2^w`.
            let mut aa = limbs::mul(&x, &p);
            let sa = truncate(&mut aa, w);
            let r = 4 * (count + 2);
            let hi = limbs::add(&aa, &limbs_of(r));
            (
                aa,
                hi,
                lsb + i128::from(sx) + sp + i128::from(sa) + i128::from(a) * k,
            )
        } else {
            // `Q = floor(X·2^t/P)`, `2^w ≤ Q < 2^(w+2)`: `y ≤ (Q + 1)(1 + u)·2^s ≤ (Q + 9)·2^s`
            // and `y ≥ Q·(1 + u)^-K·2^s ≥ Q·(1 - K·u)·2^s ≥ (Q - 8K)·2^s`, `K = count`.
            let t = w + 1 + limbs::bit_len(&p) - limbs::bit_len(&x);
            let (qq, _) = div_rem(&limbs::shl(&x, t), &p);
            let r = limbs_of(8 * count + 10);
            let lo = limbs::sub(&qq, &r);
            let hi = limbs::add(&qq, &r);
            let s = lsb + i128::from(sx) - sp - i128::from(t) - i128::from(a) * j as i128;
            (lo, hi, s)
        };
        let (tlo, hlo, _) = int_parts(&lo, s);
        let (thi, hhi, _) = int_parts(&hi, s);
        if tlo == thi {
            // `T ≤ lo ≤ y ≤ hi < T + 1`, and `y` is not an integer.
            let up = match dir {
                Dir::Floor => Some(false),
                Dir::Ceil => Some(true),
                // `hi ≤ T + 1/2`: `y < T + 1/2`, which `y` is not; `lo ≥ T + 1/2`: `y > T + 1/2`.
                Dir::Nearest | Dir::Away => {
                    if hhi != Ordering::Greater {
                        Some(false)
                    } else if hlo != Ordering::Less {
                        Some(true)
                    } else {
                        None
                    }
                }
            };
            if let Some(up) = up {
                let mut t = tlo;
                return if up {
                    limbs::increment(&mut t);
                    (t, Ordering::Greater)
                } else {
                    (t, Ordering::Less)
                };
            }
        }
        w += w / 2;
    }
}

/// The limbs of `n`, without zero limbs at the top.
#[inline]
pub(crate) fn limbs_of(n: u128) -> Vec<u64> {
    let mut v = vec![n as u64, (n >> 64) as u64];
    limbs::trim(&mut v);
    v
}

/// `±D·b^e` for the magnitude `d` (not zero, without zero limbs at the top), the radix `b` from 2
/// to 36 and `|e| ≤ 2^64`, rounded to `prec` bits in the mode `round`, and the direction of the
/// rounding, with the exponent range applied. Exact when `D·b^e` may be a boundary of the
/// rounding or the exact work is small; by Ziv's strategy otherwise (see the [module](self)).
pub(crate) fn round_value(
    negative: bool,
    d: &[u64],
    b: u32,
    e: i128,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    debug_assert!(!d.is_empty() && d.last() != Some(&0));
    debug_assert!(e.unsigned_abs() <= 1 << 64);
    let (a, c) = split(b);
    if c == 1 {
        return round_mag(negative, d, i128::from(a) * e, false, prec, round);
    }
    let bits = i128::from(limbs::bit_len(d));
    let j = e.unsigned_abs();
    let (plo, phi) = log2_pow(b, j);
    // `2^vlo ≤ v < 2^vhi`.
    let (vlo, vhi) = if e >= 0 {
        (bits - 1 + plo, bits + phi)
    } else {
        (bits - 1 - phi, bits - plo)
    };
    if vlo >= i128::from(BigFloat::EXP_MAX) {
        return overflow(negative, prec, round);
    }
    if vhi <= i128::from(BigFloat::EXP_MIN) - 2 {
        // Below `2^(EXP_MIN - 2)`: an exponent of `EXP_MIN - 2` or less.
        return underflow(
            negative,
            i128::from(BigFloat::EXP_MIN) - 2,
            false,
            prec,
            round,
        );
    }
    let q = bits - i128::from(limbs::trailing_zeros(d));
    let (clo, chi) = log2_pow(c as u32, j);
    let p = i128::from(prec);
    let boundary = if e >= 0 {
        q + clo < p + 2
    } else {
        clo < q && q < p + 2 + chi
    };
    let cheap = bits + phi <= 2 * (p + 128);
    if boundary || cheap {
        let pw = pow_exact(c, j as u64);
        return if e >= 0 {
            round_mag(
                negative,
                &limbs::mul(d, &pw),
                i128::from(a) * e,
                false,
                prec,
                round,
            )
        } else if pw.len() == 1 {
            // `D·2^t / c^j` to more than `prec + 1` bits, the remainder the sticky bit.
            let t = (prec + 66).saturating_sub(limbs::bit_len(d));
            let mut n = limbs::shl(d, t);
            let r = div_1(&mut n, pw[0]);
            round_mag(
                negative,
                &n,
                -i128::from(t) - i128::from(a) * j as i128,
                r != 0,
                prec,
                round,
            )
        } else {
            let t = (prec + 2 + limbs::bit_len(&pw)).saturating_sub(limbs::bit_len(d));
            let num = limbs::shl(d, t);
            if num.len() <= SCHOOLBOOK {
                let (q, r) = schoolbook(&num, &pw);
                round_mag(
                    negative,
                    &q,
                    -i128::from(t) - i128::from(a) * j as i128,
                    !r.is_empty(),
                    prec,
                    round,
                )
            } else {
                div_mags(negative, d, 0, &pw, i128::from(a) * j as i128, prec, round)
            }
        };
    }
    // Ziv: `v` is no boundary. As in `round_int`, with `K ≤ 2·|e| + 3 < 2^67`.
    let mut w = prec + 64 + 2 * bitlen(j);
    loop {
        let (x, sx) = top_bits(d, w);
        let (pw, sp, count) = pow_approx(c, j, w);
        let approx = if e >= 0 {
            // `v ∈ [A, A + 4K]·2^s`, `K = count + 2`.
            let mut aa = limbs::mul(&x, &pw);
            let sa = truncate(&mut aa, w);
            let s = i128::from(sx) + sp + i128::from(sa) + i128::from(a) * e;
            Approx {
                negative,
                m: aa,
                lsb: s,
                err: s + i128::from(bitlen(4 * (count + 2))),
            }
        } else {
            // `v ∈ [Q - 8K, Q + 9]·2^s`, `K = count`.
            let t = w + 1 + limbs::bit_len(&pw) - limbs::bit_len(&x);
            let (qq, _) = div_rem(&limbs::shl(&x, t), &pw);
            let s = i128::from(sx) - sp - i128::from(t) - i128::from(a) * j as i128;
            Approx {
                negative,
                m: qq,
                lsb: s,
                err: s + i128::from(bitlen(8 * count + 10)),
            }
        };
        if let Some(r) = round_approx(&approx, prec, round) {
            return r;
        }
        w += w / 2;
    }
}

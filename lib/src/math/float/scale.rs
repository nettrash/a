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
//! made, `K·u ≤ 1`, lies within `[A, A·(1 + u)^K] ⊂ [A, A·(1 + 2Ku)]` of the result `A`. The
//! power is cut to whole limbs, `64·ceil(w/64)` bits, which loses less still; up to 384 bits it
//! is squared in registers and on the stack, and up to 768 bits the rest of an attempt of
//! [`round_int`] works in fixed buffers on the stack too.

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

/// The largest power `b^t` below `2^64` of each `b` from 2 to 36 (index `b`), and `t`.
const WORD_POWERS: [(u64, u64); 37] = {
    let mut table = [(0u64, 0u64); 37];
    let mut b = 2u64;
    while b <= 36 {
        let (mut p, mut t) = (b, 1u64);
        while let Some(q) = p.checked_mul(b) {
            p = q;
            t += 1;
        }
        table[b as usize] = (p, t);
        b += 1;
    }
    table
};

/// The largest power `b^t` of `b` from 2 to 36 below `2^64`, and `t`.
#[inline]
pub(crate) fn word_power(b: u64) -> (u64, u64) {
    WORD_POWERS[b as usize]
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

/// `x²` into `r`, `N = 2L`, column by column, the products off the diagonal once and doubled, with
/// a three-word accumulator: for the short operands of [`pow_fixed`], whose loops unroll.
#[inline(always)]
fn sqr_fixed<const L: usize, const N: usize>(x: &[u64; L], r: &mut [u64; N]) {
    // The running sum, three words: `acc` and `top` above it.
    let (mut acc, mut top) = (0u128, 0u64);
    for k in 0..N - 1 {
        // Column `k`: twice the products `x_i·x_(k-i)` for `i < k - i`, and the square of
        // `x_(k/2)` for an even `k`.
        let (mut c, mut ct) = (0u128, 0u64);
        let mut i = (k + 1).saturating_sub(L);
        while 2 * i < k {
            let (sum, o) = c.overflowing_add(u128::from(x[i]) * u128::from(x[k - i]));
            c = sum;
            ct += u64::from(o);
            i += 1;
        }
        ct = ct << 1 | (c >> 127) as u64;
        c <<= 1;
        if k % 2 == 0 {
            let (sum, o) = c.overflowing_add(u128::from(x[k / 2]) * u128::from(x[k / 2]));
            c = sum;
            ct += u64::from(o);
        }
        let (sum, o) = acc.overflowing_add(c);
        acc = sum;
        top += ct + u64::from(o);
        r[k] = acc as u64;
        acc = acc >> 64 | u128::from(top) << 64;
        top = 0;
    }
    r[N - 1] = acc as u64;
}

/// [`pow_approx`] in `L` limbs held in registers and on the stack (`N = 2L`): every square and
/// product cut to its top `L` limbs, shifted to its top bit, the low limbs zero at first. Each
/// step counts as a truncation, which the bound `K ≤ 2q - 1` covers.
fn pow_fixed<const L: usize, const N: usize>(
    c: u64,
    base: u64,
    q: u128,
    r: u32,
) -> (Vec<u64>, i128, u128) {
    let mut v = [0u64; L];
    let sh = base.leading_zeros();
    v[L - 1] = base << sh;
    let mut s = -i128::from(sh) - 64 * (L as i128 - 1);
    let mut count = 0u128;
    let mut sq = [0u64; N];
    let times = |v: &mut [u64; L], m: u64| -> i128 {
        let mut p = [0u64; L];
        let mut carry = 0u64;
        for i in 0..L {
            let t = u128::from(v[i]) * u128::from(m) + u128::from(carry);
            p[i] = t as u64;
            carry = (t >> 64) as u64;
        }
        // `v ≥ 2^(64L - 1)` and `m ≥ 3`: the carry is not zero.
        let sh = carry.leading_zeros();
        if sh == 0 {
            v[..L - 1].copy_from_slice(&p[1..]);
            v[L - 1] = carry;
        } else {
            for i in 0..L - 1 {
                v[i] = p[i + 1] << sh | p[i] >> (64 - sh);
            }
            v[L - 1] = carry << sh | p[L - 1] >> (64 - sh);
        }
        64 - i128::from(sh)
    };
    for i in (0..127 - q.leading_zeros()).rev() {
        sqr_fixed(&v, &mut sq);
        // At least `2^(128L - 2)`: shifted by one bit at most.
        if sq[N - 1] >> 63 == 0 {
            for j in 0..L {
                v[j] = sq[L + j] << 1 | sq[L + j - 1] >> 63;
            }
            s = 2 * s + 64 * L as i128 - 1;
        } else {
            v.copy_from_slice(&sq[L..]);
            s = 2 * s + 64 * L as i128;
        }
        count = 2 * count + 1;
        if q >> i & 1 == 1 {
            s += times(&mut v, base);
            count += 1;
        }
    }
    if r > 0 {
        s += times(&mut v, c.pow(r));
        count += 1;
    }
    let mut out = v.to_vec();
    limbs::trim(&mut out);
    (out, s, count)
}

/// The limbs that [`pow_approx`] keeps on the stack; longer ones go to the heap.
const POW_STACK: usize = 12;

/// The top `ln = min(rn, l)` limbs of `r[..rn]` (top limb not zero) shifted to its top bit, into
/// `v`: `r = (v + θ)·2^ds`, `0 ≤ θ < 1`; their number, `ds`, and whether limbs were dropped
/// (`θ` may then be anything below one, a relative error below `2^(1 - 64l)`, `v` being at least
/// `2^(64l - 1)`).
#[inline]
fn take_top(v: &mut [u64], r: &[u64], rn: usize, l: usize) -> (usize, i128, bool) {
    let sh = r[rn - 1].leading_zeros();
    let ln = rn.min(l);
    let d = rn - ln;
    if sh == 0 {
        v[..ln].copy_from_slice(&r[d..rn]);
    } else {
        for (i, o) in v[..ln].iter_mut().enumerate() {
            let k = d + i;
            let below = if k > 0 { r[k - 1] >> (64 - sh) } else { 0 };
            *o = r[k] << sh | below;
        }
    }
    (ln, 64 * d as i128 - i128::from(sh), d > 0)
}

/// `v[..len]·m` into `r`, then its top `l` limbs back into `v`, as [`take_top`].
#[inline]
fn times(v: &mut [u64], r: &mut [u64], len: usize, m: u64, l: usize) -> (usize, i128, bool) {
    let mut carry = 0u64;
    for (o, &x) in r[..len].iter_mut().zip(&v[..len]) {
        let p = u128::from(x) * u128::from(m) + u128::from(carry);
        *o = p as u64;
        carry = (p >> 64) as u64;
    }
    r[len] = carry;
    let rn = if carry == 0 { len } else { len + 1 };
    take_top(v, r, rn, l)
}

/// A lower bound `P·2^s` of `c^k`, for the odd `c > 1`, with `P` of at most `64l` bits, `l =
/// ceil(w/64)`, and a count `K` of truncations, `c^k ≤ P·2^s·(1 + 2^(1-64l))^K`: by squares of
/// `c^t` (the largest power of `c` in a word) from the top bit of `q = floor(k/t)`, each square
/// and product shifted to its top bit and cut to its top `l` limbs, which loses less than
/// `2^(1-64l)` of it, then `c^(k mod t)`, in two buffers of `2l + 1` and `l` limbs (on the stack
/// to [`POW_STACK`] limbs). A square of a bound of `K` truncations is one of `2K`, and `2K + 1`
/// cut; a product by the exact `c^t`, `K + 1`: after the bits of `q` down to a prefix `p`, `K ≤
/// 2p - 2` (`2(2p - 2) + 2 ≤ 2(2p + 1) - 2`), so that in the end `K ≤ 2q - 1 < 2k/t`.
pub(crate) fn pow_approx(c: u64, k: u128, w: u64) -> (Vec<u64>, i128, u128) {
    let (base, t) = word_power(c);
    let (q, r) = (k / u128::from(t), (k % u128::from(t)) as u32);
    if q == 0 {
        return (vec![c.pow(r)], 0, 0);
    }
    let l = w.div_ceil(64).max(1) as usize;
    match l {
        1 | 2 => return pow_fixed::<2, 4>(c, base, q, r),
        3 => return pow_fixed::<3, 6>(c, base, q, r),
        4 => return pow_fixed::<4, 8>(c, base, q, r),
        5 => return pow_fixed::<5, 10>(c, base, q, r),
        6 => return pow_fixed::<6, 12>(c, base, q, r),
        _ => {}
    }
    pow_general(c, base, q, r, l)
}

/// [`pow_approx`] in `l` limbs, by the crate's squares.
fn pow_general(c: u64, base: u64, q: u128, r: u32, l: usize) -> (Vec<u64>, i128, u128) {
    let mut stack = [0u64; 3 * POW_STACK + 1];
    let mut heap = Vec::new();
    let buf: &mut [u64] = if l <= POW_STACK {
        &mut stack
    } else {
        heap.resize(3 * l + 1, 0);
        &mut heap
    };
    let (v, r2) = buf.split_at_mut(l);
    let mut scratch = Vec::new();
    let sh = base.leading_zeros();
    v[0] = base << sh;
    let (mut len, mut s, mut count) = (1usize, -i128::from(sh), 0u128);
    for i in (0..127 - q.leading_zeros()).rev() {
        {
            let (sq, _) = r2.split_at_mut(2 * len);
            crate::math::numbers::nat::sqr_to(sq, &v[..len], &mut scratch);
            // `v ≥ 2^(64·len - 1)`: the top limb of the square is at least `2^62`.
            let (ln, ds, cut) = take_top(v, sq, 2 * len, l);
            len = ln;
            s = 2 * s + ds;
            count = 2 * count + u128::from(cut);
        }
        if q >> i & 1 == 1 {
            let (ln, ds, cut) = times(v, r2, len, base, l);
            len = ln;
            s += ds;
            count += u128::from(cut);
        }
    }
    if r > 0 {
        let (ln, ds, cut) = times(v, r2, len, c.pow(r), l);
        len = ln;
        s += ds;
        count += u128::from(cut);
    }
    (v[..len].to_vec(), s, count)
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
    let (d, o, _) = round_int_wide(m, lsb, b, k, dir, |_| false);
    (d, o)
}

/// The parts of `y/b` from those of `y`: its integer part `t`, the order of its fractional part
/// `f` to one half and whether `f` is zero. With `t = b·t' + r`, `y/b = t' + (r + f)/b`, whose
/// fraction is zero when `r` and `f` are, and compares to one half as `2r + 2f` to `b`: above
/// for `2r ≥ b + 1`, and for `2r = b` unless `f` is zero, as `f` to one half for `2r = b - 1`,
/// below otherwise.
fn narrow(mut t: Vec<u64>, half: Ordering, zero: bool, b: u32) -> (Vec<u64>, Ordering, bool) {
    let r = div_1(&mut t, u64::from(b));
    let d = 2 * r as i64 - i64::from(b);
    let h = match d {
        1.. => Ordering::Greater,
        0 if zero => Ordering::Equal,
        0 => Ordering::Greater,
        -1 => half,
        _ => Ordering::Less,
    };
    (t, h, zero && r == 0)
}

/// [`round_int`], or, when the integer part `T` of `y` is one that `wide(T)` holds (the caller's
/// test of a digit too many), the integer that `y/b` rounds to, from the same work: the integer,
/// the direction, and whether it is that of `y/b`.
pub(crate) fn round_int_wide(
    m: &[u64],
    lsb: i128,
    b: u32,
    k: i128,
    dir: Dir,
    wide: impl Fn(&[u64]) -> bool,
) -> (Vec<u64>, Ordering, bool) {
    debug_assert!(!m.is_empty() && m.last() != Some(&0));
    // The rounding of the exact parts, of `y` or of `y/b`.
    let exact = |t: Vec<u64>, h: Ordering, z: bool| {
        if wide(&t) {
            let (t, h, z) = narrow(t, h, z, b);
            let (d, o) = decide(t, h, z, dir);
            (d, o, true)
        } else {
            let (d, o) = decide(t, h, z, dir);
            (d, o, false)
        }
    };
    let (a, c) = split(b);
    if c == 1 {
        let (t, h, z) = int_parts(m, lsb + i128::from(a) * k);
        return exact(t, h, z);
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
        return exact(Vec::new(), Ordering::Less, false);
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
            return exact(t, h, z);
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
        return exact(t, half_order(&r, &den), z);
    }
    // Ziv: `y` is no half-integer, and so neither is `y/b` (`y/b = t + 1/2` makes `y` an integer
    // or a half-integer). The working precision covers the integer part, the count of
    // truncations, `K ≤ 2·|k| + 3 < 2^67` (see [`pow_approx`]), so that `K·u ≤ 1` and the bounds
    // below hold, and 12 bits more, whole limbs: the error, `8|k|/t + 10` units (`t ≥ 3` the
    // digits of `c` in a word), leaves the decision open for about one input in `2^9` or fewer,
    // which then costs a second attempt at a working precision half as large again.
    let mut w = (yhi as u64 + bitlen(j) + 12).next_multiple_of(64);
    while w <= 64 * POW_STACK as u64 {
        if let Some(r) = ziv_small(m, lsb, b, k, w, dir, &wide) {
            return r;
        }
        w += w / 2;
        w = w.next_multiple_of(64);
    }
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
            let mut t = tlo;
            let divided = wide(&t);
            let nearest = if divided {
                // `y/b = T' + (r + f)/b`, `f` the fraction of `y`, neither zero nor a half: above
                // one half for `2r ≥ b`, below for `2r ≤ b - 2`, as `f` for `2r = b - 1`.
                let r = div_1(&mut t, u64::from(b));
                match 2 * r as i64 - i64::from(b) {
                    0.. => Some(true),
                    -1 => None,
                    _ => Some(false),
                }
            } else {
                None
            };
            let up = match dir {
                Dir::Floor => Some(false),
                Dir::Ceil => Some(true),
                // `hi ≤ T + 1/2`: `y < T + 1/2`, which `y` is not; `lo ≥ T + 1/2`: `y > T + 1/2`.
                Dir::Nearest | Dir::Away => nearest.or(if hhi != Ordering::Greater {
                    Some(false)
                } else if hlo != Ordering::Less {
                    Some(true)
                } else {
                    None
                }),
            };
            if let Some(up) = up {
                return if up {
                    limbs::increment(&mut t);
                    (t, Ordering::Greater, divided)
                } else {
                    (t, Ordering::Less, divided)
                };
            }
        }
        w += w / 2;
    }
}

/// The limbs of the fixed buffers of [`ziv_small`].
const ZIV_LIMBS: usize = 2 * POW_STACK + 4;

/// `v·2^sh` into `out` (`sh < 64`): its length, without a zero limb at the top.
#[inline]
fn shl_small(out: &mut [u64], v: &[u64], sh: u64) -> usize {
    let (w, b) = ((sh / 64) as usize, (sh % 64) as u32);
    out[..w].fill(0);
    if b == 0 {
        out[w..w + v.len()].copy_from_slice(v);
        return w + v.len();
    }
    let mut carry = 0;
    for (i, &x) in v.iter().enumerate() {
        out[w + i] = x << b | carry;
        carry = x >> (64 - b);
    }
    out[w + v.len()] = carry;
    if carry == 0 {
        w + v.len()
    } else {
        w + v.len() + 1
    }
}

/// `floor(v/2^d)` into `out`: its length, without zero limbs at the top.
#[inline]
fn shr_small(out: &mut [u64], v: &[u64], d: u64) -> usize {
    let (w, b) = ((d / 64) as usize, (d % 64) as u32);
    if w >= v.len() {
        return 0;
    }
    let n = v.len() - w;
    for i in 0..n {
        let hi = if b > 0 && w + i + 1 < v.len() {
            v[w + i + 1] << (64 - b)
        } else {
            0
        };
        out[i] = if b == 0 { v[w + i] } else { v[w + i] >> b | hi };
    }
    let mut n = n;
    while n > 0 && out[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// `v ± e` into `out` (`v ≥ e` for a difference): its length, without zero limbs at the top.
#[inline]
fn add_small(out: &mut [u64], v: &[u64], e: u128, sub: bool) -> usize {
    let mut carry = e;
    let mut n = v.len();
    for (i, &x) in v.iter().enumerate() {
        let c = carry as u64;
        let (r, o) = if sub {
            x.overflowing_sub(c)
        } else {
            x.overflowing_add(c)
        };
        out[i] = r;
        carry = (carry >> 64) + u128::from(o);
    }
    if carry != 0 {
        debug_assert!(!sub && carry >> 64 == 0);
        out[n] = carry as u64;
        n += 1;
    }
    while n > 0 && out[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// The order to one half of the fraction of `v·2^-d`, `d ≥ 1`.
#[inline]
fn fraction_half(v: &[u64], d: u64) -> Ordering {
    let hb = d - 1;
    if !limbs::bit(v, hb) {
        Ordering::Less
    } else if limbs::zero_below(v, hb) {
        Ordering::Equal
    } else {
        Ordering::Greater
    }
}

/// One attempt of Ziv's strategy of [`round_int_wide`] at the working precision `w` (a multiple
/// of 64, at most `64·POW_STACK`), as the loop there, in fixed buffers on the stack with the
/// crate's product and division: `Some` of the result when the attempt decides it.
fn ziv_small(
    m: &[u64],
    lsb: i128,
    b: u32,
    k: i128,
    w: u64,
    dir: Dir,
    wide: &impl Fn(&[u64]) -> bool,
) -> Option<(Vec<u64>, Ordering, bool)> {
    let (a, c) = split(b);
    let j = k.unsigned_abs();
    let bits = limbs::bit_len(m);
    let sx = bits.saturating_sub(w);
    let mut x = [0u64; ZIV_LIMBS];
    let xl = shr_small(&mut x, m, sx);
    let x = &x[..xl];
    let (p, sp, count) = pow_approx(c, j, w);
    let mut lo = [0u64; ZIV_LIMBS];
    let mut hi = [0u64; ZIV_LIMBS];
    let mut scratch = Vec::new();
    let (ll, hl, s) = if k >= 0 {
        // As in `round_int_wide`: `y ∈ [A, A + 4K]·2^s`, `K = count + 2`.
        let mut prod = [0u64; ZIV_LIMBS];
        let pn = xl + p.len();
        crate::math::numbers::nat::mul_to(&mut prod[..pn], x, &p, &mut scratch);
        let pn = if prod[pn - 1] == 0 { pn - 1 } else { pn };
        let sa = limbs::bit_len(&prod[..pn]).saturating_sub(w);
        let ll = shr_small(&mut lo, &prod[..pn], sa);
        let hl = add_small(&mut hi, &lo[..ll], 4 * (count + 2), false);
        let s = lsb + i128::from(sx) + sp + i128::from(sa) + i128::from(a) * k;
        (ll, hl, s)
    } else {
        // As in `round_int_wide`: `y ∈ [Q - 8K, Q + 9]·2^s`, `K = count`, with `Q =
        // floor(X·2^t/P)` from the divisor shifted to its top bit.
        let sh = u64::from(p[p.len() - 1].leading_zeros());
        let mut dv = [0u64; ZIV_LIMBS];
        let dl = shl_small(&mut dv, &p, sh);
        let t = w + 1 + limbs::bit_len(&p) - limbs::bit_len(x);
        let mut num = [0u64; ZIV_LIMBS];
        let nl = {
            // `X·2^(t + sh)`: whole limbs, then the bits.
            let s = t + sh;
            let (wl, _) = ((s / 64) as usize, s % 64);
            let n = shl_small(&mut num[wl..], x, s % 64);
            num[..wl].fill(0);
            wl + n
        };
        let qn = nl - dl;
        let mut q = [0u64; ZIV_LIMBS];
        let top = crate::math::numbers::nat::div_qr(
            &mut q[..qn],
            &mut num[..nl],
            &dv[..dl],
            &mut scratch,
        );
        q[qn] = u64::from(top);
        let ql = if top { qn + 1 } else { qn };
        let e = 8 * count + 10;
        let ll = add_small(&mut lo, &q[..ql], e, true);
        let hl = add_small(&mut hi, &q[..ql], e, false);
        let s = lsb + i128::from(sx) - sp - i128::from(t) - i128::from(a) * j as i128;
        (ll, hl, s)
    };
    let (lo, hi) = (&lo[..ll], &hi[..hl]);
    if s >= 0 {
        // Integers at this scale: the loop on vectors takes them.
        return None;
    }
    let d = (-s) as u64;
    let mut tl = [0u64; ZIV_LIMBS];
    let mut th = [0u64; ZIV_LIMBS];
    let tn = shr_small(&mut tl, lo, d);
    let hn = shr_small(&mut th, hi, d);
    if tl[..tn] != th[..hn] {
        return None;
    }
    // `T ≤ lo ≤ y ≤ hi < T + 1`, and `y` is not an integer.
    let mut t = tl[..tn].to_vec();
    let divided = wide(&t);
    let nearest = if divided {
        let r = div_1(&mut t, u64::from(b));
        match 2 * r as i64 - i64::from(b) {
            0.. => Some(true),
            -1 => None,
            _ => Some(false),
        }
    } else {
        None
    };
    let up = match dir {
        Dir::Floor => Some(false),
        Dir::Ceil => Some(true),
        Dir::Nearest | Dir::Away => nearest.or(if fraction_half(hi, d) != Ordering::Greater {
            Some(false)
        } else if fraction_half(lo, d) != Ordering::Less {
            Some(true)
        } else {
            None
        }),
    }?;
    Some(if up {
        limbs::increment(&mut t);
        (t, Ordering::Greater, divided)
    } else {
        (t, Ordering::Less, divided)
    })
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
    let mut w = (prec + bitlen(j) + 12).next_multiple_of(64);
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

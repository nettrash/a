//! Lucas sequences of integers: `U_n(P, Q)` and `V_n(P, Q)` exactly.
//!
//! `U_k` and `U_(k-1)` are doubled from the top bit of `n` down with two squares a bit, as the
//! Fibonacci numbers are ([`super::fibonacci`]), the identities taking `Q^k` beside them:
//!
//! - `U_(2k-1) = U_k^2 - Q U_(k-1)^2`,
//! - `U_(2k+1) = (P^2 - 3Q) U_k^2 - Q^2 U_(k-1)^2 + 2 Q^k`,
//! - `U_(2k) = (U_(2k+1) + Q U_(2k-1)) / P`, a division that is exact,
//!
//! the first from `U_(m+n) = U_m U_(n+1) - Q U_(m-1) U_n`, the second from it with
//! `U_(k+1) = P U_k - Q U_(k-1)` and `U_k^2 - P U_k U_(k-1) + Q U_(k-1)^2 = Q^(k-1)`, the third the
//! recurrence. Then `V_n = P U_n - 2 Q U_(n-1)`. `P = 1, Q = -1` are the Fibonacci and Lucas
//! numbers, taken from there, and `P = ±3, Q = 1` the even ones (`U_n(3, 1) = F_(2n)`,
//! `V_n(3, 1) = L_(2n)`); `P = 0`, `Q = 0` and `P^2 = 4Q` have closed forms.
//!
//! For `|P|` and `|Q|` below `2^30` the doubling runs on limbs ([`limb_sequence`]): its first steps
//! in `i128`s while the bit lengths keep every product below `2^125`, then the squares by the
//! kernels of [`super::fibonacci`], and each of the two values a step keeps in one pass over the
//! squares' limbs ([`combine`]): the linear combination in an `i128` a limb, and for `U_(2k)` the
//! exact division by `P` in the same pass, by Hensel's method from the low limb up. Longer `P` and
//! `Q` take the same steps on numbers ([`doubled`]).

use super::super::div::limbs::binvert_limb;
use super::super::root::{limbs_number, mul_1, small_number};
use super::super::{ByteOrder, Number};
use super::fibonacci::{even, fibonacci_and_lucas, square, with_memory};

/// The number in the byte order `order`, its bytes turned around when it is in the other one.
fn in_order(mut x: Number, order: ByteOrder) -> Number {
    if x.byte_order != order {
        x.value.reverse();
        x.byte_order = order;
    }
    x
}

/// `x^e` for an exponent of any size: `x^e = x^(e mod 2^32) (x^(2^32))^(e / 2^32)`, the powers of
/// 32-bit exponents by [`Number::pow`].
fn pow(x: &Number, e: u64) -> Number {
    if let Ok(e) = u32::try_from(e) {
        return x.pow(e);
    }
    let low = x.pow(e as u32);
    let h = x.pow(1 << 31);
    &low * &pow(&(&h * &h), e >> 32)
}

/// The integer of `x` when it fits an `i64`.
fn small(x: &Number) -> Option<i64> {
    i64::try_from(x).ok()
}

/// `(U_n(P, Q), V_n(P, Q))`, in the byte order of `p`.
pub(in super::super) fn lucas_sequence(p: &Number, q: &Number, n: u64) -> (Number, Number) {
    let order = p.byte_order;
    let int = |v: i64| Number::new_from_i64(v, order);
    if n == 0 {
        return (int(0), int(2));
    }
    let (sp, sq) = (small(p), small(q));
    if n == 1 {
        return (int(1), in_order(p.clone() + 0, order));
    }
    if sq == Some(1)
        && matches!(sp, Some(3) | Some(-3))
        && let Some(m) = n.checked_mul(2)
    {
        // `U_n(3, 1) = F_(2n)` and `V_n(3, 1) = L_(2n)`, `3 = L_2` and `1 = (-1)^2`; with the signs
        // of `U_n(-P, Q) = (-1)^(n-1) U_n(P, Q)` and `V_n(-P, Q) = (-1)^n V_n(P, Q)`.
        let (mut u, mut v) = fibonacci_and_lucas(m, order);
        if sp == Some(-3) {
            if n & 1 == 0 {
                u = -u;
            } else {
                v = -v;
            }
        }
        return (u, v);
    }
    if let (Some(ps), Some(qs)) = (sp, sq)
        && ps.unsigned_abs() < LIMB_PARAMETER
        && qs.unsigned_abs() < LIMB_PARAMETER
        && ps != 0
        && qs != 0
        && !(qs == -1 && ps.unsigned_abs() == 1)
        && ps * ps != 4 * qs
    {
        return limb_sequence(ps, qs, n, order);
    }
    if sq == Some(-1) && matches!(sp, Some(1) | Some(-1)) {
        // The Fibonacci and Lucas numbers: `U_n(-P, Q) = (-1)^(n-1) U_n(P, Q)` and
        // `V_n(-P, Q) = (-1)^n V_n(P, Q)`.
        let (mut u, mut v) = fibonacci_and_lucas(n, order);
        if sp == Some(-1) {
            if n & 1 == 0 {
                u = -u;
            } else {
                v = -v;
            }
        }
        return (u, v);
    }
    if sq == Some(0) {
        // `U_n = P^(n-1)`, `V_n = P^n`.
        let u = pow(p, n - 1);
        let v = &u * p;
        return (in_order(u, order), in_order(v, order));
    }
    if sp == Some(0) {
        // `U_(2j) = 0`, `U_(2j+1) = (-Q)^j`, `V_(2j) = 2 (-Q)^j`, `V_(2j+1) = 0`.
        let x = pow(&-q, n / 2);
        return if n & 1 == 1 {
            (in_order(x, order), int(0))
        } else {
            (int(0), in_order(x << 1, order))
        };
    }
    let d = &(p * p) - &(q * 4i32);
    if d.is_zero() {
        // `P = 2c`, `Q = c^2`: `U_n = n c^(n-1)`, `V_n = 2 c^n`.
        let c = p >> 1;
        let x = pow(&c, n - 1);
        let v = &(&x * &c) << 1;
        let u = x * Number::new_from_u64(n, order);
        return (in_order(u, order), in_order(v, order));
    }
    let (u, w) = doubled(p, q, n);
    // `V_n = P U_n - 2 Q U_(n-1)`.
    let v = &(&u * p) - &(&(&w * q) << 1);
    (in_order(u, order), in_order(v, order))
}

/// The doubling of [`doubled`] in `i128`s, for the top bits of `n` from the first: the number of
/// bits of `n` still to take, and `(U_k, U_(k-1), Q^k)` for `k` the bits taken, as far as every
/// value fits.
fn doubled_i128(p: i64, q: i64, n: u64) -> (u32, (i128, i128, i128)) {
    let bits = 64 - n.leading_zeros();
    let (p, q) = (i128::from(p), i128::from(q));
    let step = |u: i128, w: i128, qk: i128, bit: bool| -> Option<(i128, i128, i128)> {
        let a = u.checked_mul(u)?;
        let b = w.checked_mul(w)?;
        let c1 = p.checked_mul(p)?.checked_sub(q.checked_mul(3)?)?;
        let lower = a.checked_sub(q.checked_mul(b)?)?;
        let upper = c1
            .checked_mul(a)?
            .checked_sub(q.checked_mul(q)?.checked_mul(b)?)?
            .checked_add(qk.checked_mul(2)?)?;
        let even = upper.checked_add(q.checked_mul(lower)?)? / p;
        let q2k = qk.checked_mul(qk)?;
        Some(if bit {
            (upper, even, q2k.checked_mul(q)?)
        } else {
            (even, lower, q2k)
        })
    };
    let (mut u, mut w, mut qk) = (1i128, 0i128, q);
    let mut i = bits - 1;
    while i > 0 {
        match step(u, w, qk, (n >> (i - 1)) & 1 == 1) {
            Some(next) => (u, w, qk) = next,
            None => break,
        }
        i -= 1;
    }
    (i, (u, w, qk))
}

/// `(U_n, U_(n-1))` for `n >= 2`, `P` not 0, `Q` not 0 and `P^2 != 4Q`, by the doubling of the
/// module's documentation.
fn doubled(p: &Number, q: &Number, n: u64) -> (Number, Number) {
    let order = p.byte_order;
    // `Q^k` for `Q = ±1` is a sign: whether it is -1.
    let unit = match small(q) {
        Some(1) => Some(false),
        Some(-1) => Some(true),
        _ => None,
    };
    let c1 = &(p * p) - &(q * 3i32);
    let q2 = q * q;
    let p_unit = small(p).filter(|x| x.unsigned_abs() == 1);
    // `k = 1`: `U_1 = 1`, `U_0 = 0`, `Q^1 = Q`; in `i128`s while they hold the values.
    let bits = 64 - n.leading_zeros();
    let (start, (u, w, qk)) = match (small(p), small(q)) {
        (Some(ps), Some(qs)) => doubled_i128(ps, qs, n),
        _ => (bits - 1, (1, 0, 0)),
    };
    let mut u = Number::new_from_i128(u, order);
    let mut w = Number::new_from_i128(w, order);
    let mut qk = if start == bits - 1 {
        q.clone()
    } else {
        Number::new_from_i128(qk, order)
    };
    // Whether `Q^k = -1` for `Q = ±1`: `k` odd.
    let mut negative = unit == Some(true) && (n >> start) & 1 == 1;
    for i in (0..start).rev() {
        let a = &u * &u;
        let b = &w * &w;
        let (qb, q2b) = match unit {
            Some(neg) => (if neg { -&b } else { b.clone() }, b),
            None => (q * &b, &q2 * &b),
        };
        // `U_(2k-1)` and `U_(2k+1)`.
        let lower = &a - &qb;
        let mut upper = &(&c1 * &a) - &q2b;
        match unit {
            Some(_) => upper += if negative { -2i32 } else { 2i32 },
            None => upper += &qk << 1,
        }
        // `U_(2k) = (U_(2k+1) + Q U_(2k-1)) / P`.
        let sum = match unit {
            Some(neg) => {
                if neg {
                    &upper - &lower
                } else {
                    &upper + &lower
                }
            }
            None => &upper + &(q * &lower),
        };
        let even = match p_unit {
            Some(1) => sum,
            Some(_) => -sum,
            None => sum.div_exact(p),
        };
        let bit = (n >> i) & 1 == 1;
        if bit {
            (u, w) = (upper, even);
        } else {
            (u, w) = (even, lower);
        }
        if i > 0 {
            match unit {
                Some(neg) => negative = neg && bit,
                None => {
                    qk = &qk * &qk;
                    if bit {
                        qk = &qk * q;
                    }
                }
            }
        }
    }
    (u, w)
}

/// The bound below which `|P|` and `|Q|` take the doubling on limbs ([`limb_sequence`]): the
/// coefficients of its linear combinations, `P^2 - 3Q`, `P^2 - 2Q`, `Q^2` and `2Q^2`, are then below
/// `2^61`, so that three terms of them times limbs, and a carry, fit an `i128`.
const LIMB_PARAMETER: u64 = 1 << 30;

/// An upper bound of `log2 M` for `M = (|P| + √(P^2 + 4|Q|)) / 2`, the largest root's magnitude
/// bound: `|U_k| <= M^k` by induction (`M^2 = |P| M + |Q|`, and `|U_(k+1)| <= |P| |U_k| + |Q|
/// |U_(k-1)|`), and `|Q|^k <= M^(2k)`. In doubles, made larger by far more than their roundings.
fn log2_root_bound(p: i64, q: i64) -> f64 {
    let (pa, qa) = (p.unsigned_abs() as f64, q.unsigned_abs() as f64);
    let m = (pa + (pa * pa + 4.0 * qa).sqrt()) / 2.0;
    m.log2().max(0.0) * (1.0 + 1e-9) + 1e-9
}

/// The length and sign of a signed number whose magnitude is in a buffer of limbs: the number of
/// its limbs without zero limbs at the top (0 for zero), and its sign.
#[derive(Clone, Copy)]
struct Signed {
    /// The number of limbs.
    n: usize,
    /// Whether the number is negative; never for zero.
    negative: bool,
}

/// Writes the magnitude of `x` at `r`, two limbs, and returns its length and sign.
///
/// # Safety
///
/// `r` must be valid for writing two limbs.
unsafe fn put_i128(r: *mut u64, x: i128) -> Signed {
    let m = x.unsigned_abs();
    // SAFETY: as the caller guarantees.
    unsafe {
        *r = m as u64;
        *r.add(1) = (m >> 64) as u64;
    }
    Signed {
        n: if m >> 64 != 0 { 2 } else { usize::from(m != 0) },
        negative: x < 0,
    }
}

/// The `n` limbs at `a` without the zero limbs at the top: possibly none.
///
/// # Safety
///
/// The `n` limbs must be valid for reads.
#[inline(always)]
unsafe fn trim(a: *const u64, mut n: usize) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        while n > 0 && *a.add(n - 1) == 0 {
            n -= 1;
        }
    }
    n
}

/// Writes zeros at the limbs of `a` from `from` up to `to`.
///
/// # Safety
///
/// The limbs up to `to` must be valid for writes.
#[inline(always)]
unsafe fn zeros(a: *mut u64, from: usize, to: usize) {
    if to > from {
        // SAFETY: as the caller guarantees.
        unsafe { core::ptr::write_bytes(a.add(from), 0, to - from) };
    }
}

/// An exact division by `d = 2^t d'`, `d'` odd, of the linear combinations of [`combine`]: `d'`,
/// its inverse modulo `2^64`, and `t`.
#[derive(Clone, Copy)]
struct Divisor {
    /// The odd part.
    odd: u64,
    /// Its inverse modulo `2^64`.
    inverse: u64,
    /// The power of two.
    twos: u32,
}

/// Sets the limbs at `r` to `(cx x + cy y + cz z + c0) / d`, for the magnitudes `x`, `y` and `z` of
/// `len` limbs each (zero-filled up to `len`, `z` read only when `Z`), small coefficients
/// (`|c| < 2^61`), and the divisor `d` when `div` is given, which divides the sum exactly; returns
/// the result's length and sign.
///
/// The sum is formed limb by limb in an `i128`, in two's complement over `len + 1` limbs, where it
/// fits; the division by `d'` is Hensel's, from the low limb up in the same pass, which needs no
/// sign (`d'` is odd and divides the sum: the quotient modulo `2^(64 (len + 1))` is its two's
/// complement); then the sign is taken from the top limb, the magnitude formed, and the power of
/// two shifted out.
///
/// # Safety
///
/// `x`, `y` (and `z`) must be valid for reading `len` limbs, and `r` for writing `len + 1`, `r`
/// apart from them.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn combine<const Z: bool>(
    r: *mut u64,
    (x, cx): (*const u64, i64),
    (y, cy): (*const u64, i64),
    (z, cz): (*const u64, i64),
    c0: i64,
    len: usize,
    div: Option<Divisor>,
) -> Signed {
    // SAFETY: as the caller guarantees.
    unsafe {
        // Limb `i` of the sum and the carry above it.
        let term = |i: usize, carry: i128| -> i128 {
            if i < len {
                let mut t = i128::from(cx) * i128::from(*x.add(i))
                    + i128::from(cy) * i128::from(*y.add(i))
                    + carry;
                if Z {
                    t += i128::from(cz) * i128::from(*z.add(i));
                }
                t
            } else {
                carry
            }
        };
        let mut carry = i128::from(c0);
        match div {
            Some(d) if d.odd > 1 => {
                // Hensel's division by the odd part, in the same pass.
                let mut borrow = 0u64;
                for i in 0..=len {
                    let t = term(i, carry);
                    carry = t >> 64;
                    let (t, b) = (t as u64).overflowing_sub(borrow);
                    let q = t.wrapping_mul(d.inverse);
                    borrow = ((u128::from(q) * u128::from(d.odd)) >> 64) as u64 + u64::from(b);
                    *r.add(i) = q;
                }
            }
            _ => {
                for i in 0..=len {
                    let t = term(i, carry);
                    carry = t >> 64;
                    *r.add(i) = t as u64;
                }
            }
        }
        // The sign, and the magnitude of a negative result.
        let negative = (*r.add(len) as i64) < 0;
        if negative {
            let mut c = 1u64;
            for i in 0..=len {
                let (v, o) = (!*r.add(i)).overflowing_add(c);
                *r.add(i) = v;
                c = u64::from(o);
            }
        }
        let mut n = trim(r, len + 1);
        if let Some(d) = div
            && d.twos > 0
            && n > 0
        {
            let t = d.twos;
            for i in 0..n - 1 {
                *r.add(i) = (*r.add(i) >> t) | (*r.add(i + 1) << (64 - t));
            }
            *r.add(n - 1) >>= t;
            n = trim(r, n);
        }
        Signed {
            n,
            negative: negative && n != 0,
        }
    }
}

/// `x / d` for a `d` that divides `x`: the power of two shifted out, then a product by the inverse
/// of the odd part modulo `2^128`, `inverse`.
#[inline(always)]
fn exact_i128(x: i128, d: i64, twos: u32, inverse: u128) -> i128 {
    let q = ((x >> twos) as u128).wrapping_mul(inverse) as i128;
    if d < 0 { q.wrapping_neg() } else { q }
}

/// The inverse of the odd `d` modulo `2^128`: that modulo `2^64` and a step of Newton's method.
fn inverse_u128(d: u64) -> u128 {
    let x = u128::from(binvert_limb(d));
    x.wrapping_mul(2u128.wrapping_sub(u128::from(d).wrapping_mul(x)))
}

/// The first steps of [`limb_sequence`] in `i128`s: the number of bits of `n` still to take, and
/// `(U_k, U_(k-1), Q^k)` for `k` the bits taken. A step is taken while the bit lengths of the
/// values and of the coefficients keep every product below `2^125`, so that nothing is checked
/// for overflow; `P U_(2k)` is divided by `P` exactly.
fn start_i128(p: i64, q: i64, n: u64) -> (u32, i128, i128, i128) {
    let bits = 64 - n.leading_zeros();
    let (pp, qq) = (i128::from(p), i128::from(q));
    let (c1, c2, q2) = (pp * pp - 3 * qq, pp * pp - 2 * qq, qq * qq);
    let largest = c1
        .unsigned_abs()
        .max(c2.unsigned_abs())
        .max(2 * q2.unsigned_abs());
    let cb = 128 - largest.leading_zeros();
    let qb = 64 - q.unsigned_abs().leading_zeros();
    let pa = p.unsigned_abs();
    let (twos, inverse) = (pa.trailing_zeros(), inverse_u128(pa >> pa.trailing_zeros()));
    let (mut u, mut w, mut qk) = (1i128, 0i128, qq);
    let mut i = bits - 1;
    while i > 0 {
        let m = 128 - (u.unsigned_abs() | w.unsigned_abs()).leading_zeros();
        let mq = 128 - qk.unsigned_abs().leading_zeros();
        if 2 * m + cb + 2 > 125 || 2 * mq + qb > 125 {
            break;
        }
        let (a, b) = (u * u, w * w);
        let lower = a - qq * b;
        let upper = c1 * a - q2 * b + 2 * qk;
        let even = exact_i128(c2 * a - 2 * q2 * b + 2 * qk, p, twos, inverse);
        i -= 1;
        let bit = (n >> i) & 1 == 1;
        (u, w) = if bit { (upper, even) } else { (even, lower) };
        qk *= qk;
        if bit {
            qk *= qq;
        }
    }
    (i, u, w, qk)
}

/// `(U_n, V_n)` for `|P|` and `|Q|` below [`LIMB_PARAMETER`], `P` and `Q` not zero and `P^2 != 4Q`,
/// `n >= 2`, in the byte order `order`: the doubling of the module's documentation on limbs, after
/// its first steps in `i128`s ([`start_i128`]).
///
/// A step squares `U_k` and `U_(k-1)` (and `Q^k`), then forms the two of `U_(2k+1) =
/// (P^2 - 3Q) U_k^2 - Q^2 U_(k-1)^2 + 2 Q^k`, `P U_(2k) = (P^2 - 2Q) U_k^2 - 2 Q^2 U_(k-1)^2 + 2 Q^k`
/// and `U_(2k-1) = U_k^2 - Q U_(k-1)^2` that the bit keeps, each in one pass of [`combine`], `U_(2k)`
/// with its exact division by `P`. For `Q = ±1`, `Q^k` is a sign.
fn limb_sequence(p: i64, q: i64, n: u64, order: ByteOrder) -> (Number, Number) {
    let (left, u0, w0, qk0) = start_i128(p, q, n);
    if left == 0 {
        // `V_n = P U_n - 2 Q U_(n-1)` in an `i128` when it fits.
        let m = 128 - (u0.unsigned_abs() | w0.unsigned_abs()).leading_zeros();
        if m + 33 <= 126 {
            let v = i128::from(p) * u0 - 2 * i128::from(q) * w0;
            return (
                small_number(order, u0.unsigned_abs(), u0 < 0),
                small_number(order, v.unsigned_abs(), v < 0),
            );
        }
    }
    // Every value has an index of at most `n + 1`, and at most the limbs of `M^(n+1)`, with three
    // to spare for the passes' top limbs; `Q^k` is squared only up to `Q^(n/2)`.
    let lm = log2_root_bound(p, q);
    let limbs = ((n as f64 + 1.0) * lm / 64.0).ceil() + 4.0;
    // Beyond `2^62` limbs no buffer can be had: refused as the allocation would be.
    let lb =
        even(usize::try_from(limbs.min((1u64 << 62) as f64) as u64).expect("capacity overflow"));
    let need = lb.checked_mul(6).expect("capacity overflow");
    let unit = q.unsigned_abs() == 1;
    let pa = p.unsigned_abs();
    let div = (pa != 1).then(|| {
        let odd = pa >> pa.trailing_zeros();
        Divisor {
            odd,
            inverse: binvert_limb(odd),
            twos: pa.trailing_zeros(),
        }
    });
    // `P^2 - 3Q`, `P^2 - 2Q` and `Q^2`.
    let (c1, c2, q2) = (p * p - 3 * q, p * p - 2 * q, q * q);
    with_memory(need, |mem, mut spare| {
        // SAFETY: the working memory holds six buffers of `lb` limbs: `U_k`, `U_(k-1)`, their
        // squares, `Q^k` and its square; every value fits one (see above).
        unsafe {
            let (u, w) = (mem, mem.add(lb));
            let (a, b) = (mem.add(2 * lb), mem.add(3 * lb));
            let (mut qk, mut qq) = (mem.add(4 * lb), mem.add(5 * lb));
            let mut su = put_i128(u, u0);
            let mut sw = put_i128(w, w0);
            // `Q^k`: on limbs, or for `Q = ±1` whether it is -1.
            let mut sq = if unit {
                Signed {
                    n: 0,
                    negative: false,
                }
            } else {
                put_i128(qk, qk0)
            };
            let mut q_minus = unit && q < 0 && (n >> left) & 1 == 1;
            for i in (0..left).rev() {
                let bit = (n >> i) & 1 == 1;
                let la = 2 * su.n;
                let lw2 = 2 * sw.n;
                if su.n > 0 {
                    square(a, u, su.n, &mut spare);
                }
                if sw.n > 0 {
                    square(b, w, sw.n, &mut spare);
                }
                let len = la.max(lw2).max(sq.n);
                zeros(a, la, len);
                zeros(b, lw2, len);
                // `2 Q^k`: a term on limbs, or a constant for `Q = ±1`.
                let (cz, k0) = if unit {
                    (0, if q_minus { -2 } else { 2 })
                } else {
                    zeros(qk, sq.n, len);
                    (if sq.negative { -2 } else { 2 }, 0)
                };
                let terms =
                    |r: *mut u64, ca: i64, cb: i64, cq: i64, c0: i64, d: Option<Divisor>| {
                        if unit {
                            combine::<false>(r, (a, ca), (b, cb), (a, 0), c0, len, d)
                        } else {
                            combine::<true>(r, (a, ca), (b, cb), (qk, cq), c0, len, d)
                        }
                    };
                // `P U_(2k)`, divided by `P`, of the sign of `P`.
                let even = |r: *mut u64| {
                    let mut e = terms(r, c2, -2 * q2, cz, k0, div);
                    e.negative ^= p < 0 && e.n != 0;
                    e
                };
                if bit {
                    // `(U_(2k+1), U_(2k))`.
                    su = terms(u, c1, -q2, cz, k0, None);
                    sw = even(w);
                } else {
                    // `(U_(2k), U_(2k-1))`.
                    su = even(u);
                    sw = terms(w, 1, -q, 0, 0, None);
                }
                // `Q^(2k)`, or `Q^(2k+1)`, but after the last step.
                if unit {
                    q_minus = q < 0 && bit;
                } else if i > 0 {
                    square(qq, qk, sq.n, &mut spare);
                    let mut lq = trim(qq, 2 * sq.n);
                    let mut negative = false;
                    if bit {
                        let c = mul_1(qq, qq, lq, q.unsigned_abs());
                        *qq.add(lq) = c;
                        lq += usize::from(c != 0);
                        negative = q < 0;
                    }
                    sq = Signed { n: lq, negative };
                    core::mem::swap(&mut qk, &mut qq);
                }
            }
            // `V_n = P U_n - 2 Q U_(n-1)`, into the square's buffer.
            let len = su.n.max(sw.n);
            zeros(u, su.n, len);
            zeros(w, sw.n, len);
            let cu = if su.negative { -p } else { p };
            let cw = if sw.negative { 2 * q } else { -2 * q };
            let sv = combine::<false>(a, (u, cu), (w, cw), (u, 0), 0, len, None);
            (
                limbs_number(u, su.n, su.negative, order),
                limbs_number(a, sv.n, sv.negative, order),
            )
        }
    })
}

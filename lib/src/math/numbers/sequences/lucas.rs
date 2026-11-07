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
//! numbers, taken from there; `P = 0`, `Q = 0` and `P^2 = 4Q` have closed forms.

use super::super::{ByteOrder, Number};
use super::fibonacci::fibonacci_and_lucas;

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

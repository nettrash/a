//! Square roots in a prime field, by the method that fits the prime, chosen once per field.
//!
//! With `p - 1 = q 2^s`, `q` odd:
//!
//! - `s = 1`, `p = 3 (mod 4)`: `a^((p + 1) / 4)`, one power (Lagrange's).
//! - `s = 2`, `p = 5 (mod 8)`: Atkin's, one power: with `b = (2a)^((p - 5) / 8)` and
//!   `i = 2a b^2`, a square root of -1 when `a` is a square, the root is `a b (i - 1)` (Atkin,
//!   1992; Cohen, *A Course in Computational Algebraic Number Theory*, 1.5.1).
//! - `s >= 3`: Tonelli and Shanks's method (Cohen, algorithm 1.5.1): the power `a^((q + 1) / 2)`,
//!   then up to `s - 1` corrections by powers of `z^q`, for a non-square `z` found once per field.
//!   A round whose bound on the order of `a^q` is `2^m` takes `m` squarings and two products, and
//!   `m` falls at least by one from `s` to 2: up to `s (s + 1) / 2 - 1` squarings for a square, and
//!   `s` to find a non-square out.
//! - `s` large, `s^2 > 12 b` for the `b` bits of `p`: Cipolla and Lehmer's method (Cohen, 1.5.1,
//!   remark), the power
//!   `(t + w)^((p + 1) / 2)` in `F_p[w] / (w^2 - (t^2 - a))` for a `t` with `t^2 - a` not a square,
//!   about five products for each bit of `p` whatever `s` is: past that bound Tonelli and Shanks's
//!   squarings cost more than Cipolla's whole power, on the counts of products.
//!
//! A result is checked by squaring it, apart from Tonelli and Shanks's, which finds a non-square
//! out by itself, and Cipolla's, which asks for the Legendre symbol first.

use super::prime_form::{Arith, Value, legendre};
use crate::math::ec::field::limbs_of;
use crate::math::numbers::Number;

/// What the square roots of a field need, computed once.
#[derive(Clone)]
pub(super) enum Plan {
    /// `GF(2)`, where every element is its own root.
    Two,
    /// `p = 3 (mod 4)`: the exponent `(p + 1) / 4`.
    ThreeMod4 {
        /// `(p + 1) / 4`.
        e: Vec<u64>,
    },
    /// `p = 5 (mod 8)`: the exponent `(p - 5) / 8` of Atkin's method.
    FiveMod8 {
        /// `(p - 5) / 8`.
        e: Vec<u64>,
    },
    /// Tonelli and Shanks's method.
    Tonelli {
        /// `s`, at least 3.
        s: u32,
        /// `(q - 1) / 2`.
        e: Vec<u64>,
        /// `z^q` for the least non-square `z`, of the order `2^s`, in the field's form.
        c: Value,
    },
    /// Cipolla and Lehmer's method.
    Cipolla {
        /// `(p + 1) / 2`.
        e: Vec<u64>,
    },
}

impl Plan {
    /// The plan for the field `f` of the prime `p`.
    pub(super) fn new<F: Arith>(f: &F, p: &Number) -> Plan {
        if *p == 2u8 {
            return Plan::Two;
        }
        let one = Number::from(1u8);
        let p1 = p - &one;
        let s = p1.trailing_zeros().unwrap_or(0);
        match s {
            1 => Plan::ThreeMod4 {
                e: limbs_of(&((p + &one) >> 2usize)),
            },
            2 => Plan::FiveMod8 {
                e: limbs_of(&((p - &Number::from(5u8)) >> 3usize)),
            },
            _ if cipolla(s, p.bits()) => Plan::Cipolla {
                e: limbs_of(&((p + &one) >> 1usize)),
            },
            _ => {
                let q = &p1 >> s as usize;
                // 2 is a square modulo the primes 1 modulo 8; the least non-square is small.
                let mut z = Number::from(3u8);
                while z.jacobi(p) != -1 {
                    z = &z + &one;
                }
                let c = f.power(&f.enter(&z), &limbs_of(&q));
                Plan::Tonelli {
                    s: s as u32,
                    e: limbs_of(&((&q - &one) >> 1usize)),
                    c: f.put(c),
                }
            }
        }
    }

    /// The name of the method, for the documentation's tests and `Debug`.
    pub(super) fn name(&self) -> &'static str {
        match self {
            Plan::Two => "GF(2)",
            Plan::ThreeMod4 { .. } => "p = 3 mod 4",
            Plan::FiveMod8 { .. } => "Atkin",
            Plan::Tonelli { .. } => "Tonelli-Shanks",
            Plan::Cipolla { .. } => "Cipolla",
        }
    }
}

/// Whether Cipolla's method is cheaper than Tonelli and Shanks's for `p - 1 = q 2^s` of `bits`
/// bits: Tonelli and Shanks's power costs about `1.2 bits` products and its corrections up to
/// `s (s + 1) / 2 - 1` squarings, about `s^2 / 4` for a random square; Cipolla's power about
/// `5.5 bits` products. The bound, `s^2 > 12 bits`, lies between the average and the worst case.
fn cipolla(s: u64, bits: u64) -> bool {
    s * s > 12 * bits
}

/// A square root of `a`, or `None` when it is not a square; either root.
pub(super) fn sqrt<F: Arith>(f: &F, plan: &Plan, a: &F::E) -> Option<F::E> {
    if f.is_zero(a) {
        return Some(f.zero());
    }
    match plan {
        Plan::Two => Some(a.clone()),
        Plan::ThreeMod4 { e } => {
            let r = f.power(a, e);
            f.same(&f.sqr(&r), a).then_some(r)
        }
        Plan::FiveMod8 { e } => {
            let a2 = f.twice(a);
            let b = f.power(&a2, e);
            let i = f.mul(&a2, &f.sqr(&b));
            let r = f.mul(&f.mul(a, &b), &f.sub(&i, &f.one()));
            f.same(&f.sqr(&r), a).then_some(r)
        }
        Plan::Tonelli { s, e, c } => tonelli(f, *s, e, f.get(c), a),
        Plan::Cipolla { e } => cipolla_lehmer(f, e, a),
    }
}

/// Tonelli and Shanks's method for a nonzero `a`, with `c = z^q` of the order `2^s`.
///
/// `r = a^((q + 1) / 2)` has `r^2 = a b` for `b = a^q`, whose order is a power of two below
/// `2^s` exactly when `a` is a square. Each round finds the order `2^t` of `b` by squaring and
/// multiplies `r` by the power of `c` of the order `2^(t + 1)`, which lowers the order of `b`.
fn tonelli<F: Arith>(f: &F, s: u32, e: &[u64], c: &F::E, a: &F::E) -> Option<F::E> {
    let x = f.power(a, e);
    let mut r = f.mul(a, &x);
    let mut b = f.mul(&r, &x);
    let one = f.one();
    let mut c = c.clone();
    let mut m = s;
    while !f.same(&b, &one) {
        let mut t = 0;
        let mut b2 = b.clone();
        while !f.same(&b2, &one) {
            b2 = f.sqr(&b2);
            t += 1;
            if t == m {
                return None;
            }
        }
        let mut g = c;
        for _ in 0..m - t - 1 {
            g = f.sqr(&g);
        }
        r = f.mul(&r, &g);
        c = f.sqr(&g);
        b = f.mul(&b, &c);
        m = t;
    }
    Some(r)
}

/// Cipolla and Lehmer's method for a nonzero `a`, with `e = (p + 1) / 2`.
///
/// For `t` with `w = t^2 - a` not a square, `(t + w')^(p + 1)` is the norm `t^2 - w = a` of
/// `t + w'` in `F_p[w'] / (w'^2 - w)`, so for a square `a` its power `(p + 1) / 2` lies in `F_p`
/// and is a root.
fn cipolla_lehmer<F: Arith>(f: &F, e: &[u64], a: &F::E) -> Option<F::E> {
    if legendre(f, a) != 1 {
        return None;
    }
    let one = f.one();
    // `t = 1, 2, ...`: each is the one with probability about 1/2.
    let mut t = one.clone();
    let w = loop {
        let w = f.sub(&f.sqr(&t), a);
        if f.is_zero(&w) {
            return Some(t);
        }
        if legendre(f, &w) == -1 {
            break w;
        }
        t = f.add(&t, &one);
    };
    // `(x0 + x1 w')`, from the top bit of `e` down.
    let top = 64 * e.len() - e.last().map_or(64, |l| l.leading_zeros() as usize);
    let (mut x0, mut x1) = (t.clone(), one);
    for i in (0..top - 1).rev() {
        let square0 = f.add(&f.sqr(&x0), &f.mul(&w, &f.sqr(&x1)));
        let square1 = f.twice(&f.mul(&x0, &x1));
        (x0, x1) = (square0, square1);
        if (e[i / 64] >> (i % 64)) & 1 != 0 {
            let times0 = f.add(&f.mul(&x0, &t), &f.mul(&w, &x1));
            let times1 = f.add(&x0, &f.mul(&t, &x1));
            (x0, x1) = (times0, times1);
        }
    }
    Some(x0)
}

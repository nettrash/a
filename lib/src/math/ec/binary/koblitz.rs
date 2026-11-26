//! Koblitz curves, `y² + x·y = x³ + a·x² + 1` with `a ∈ {0, 1}`, and their products by the
//! τ-adic non-adjacent form, after Solinas ("Efficient arithmetic on Koblitz curves", Designs,
//! Codes and Cryptography 19, 2000), as Hankerson, Menezes and Vanstone give it (*Guide to Elliptic
//! Curve Cryptography*, §3.4, algorithms 3.61 to 3.70).
//!
//! The Frobenius map `τ(x, y) = (x², y²)` is an endomorphism of such a curve, and satisfies
//! `τ² − μ·τ + 2 = 0` with `μ = (−1)^(1 − a)`, so that `Z[τ]` acts on the points, and `τᵐ` is the
//! identity on the points over `GF(2ᵐ)`. A scalar `k` is first reduced modulo `τᵐ − 1` in `Z[τ]`:
//! `k = q·(τᵐ − 1) + ρ` with `q` the rounding of `k / (τᵐ − 1)` and `ρ` of norm at most `4/7` that
//! of `τᵐ − 1`, the number of points; `ρ·P = k·P` for every point. Then `ρ` is written in the
//! width-`w` τ-adic non-adjacent form, digits `±α_u` for odd `u < 2^(w − 1)`, `α_u` the element of
//! least norm congruent to `u` modulo `τʷ`, of about `m` digits of which one in `w + 1` is not zero.
//! A product so takes about `m / (w + 1)` additions and `m` Frobenius maps, three squares each, and
//! no doublings.
//!
//! The numbers of points come with it: `#E(GF(2ᵐ)) = 2ᵐ + 1 − V_m`, for the Lucas sequence
//! `V_0 = 2`, `V_1 = μ`, `V_(k+1) = μ·V_k − 2·V_(k−1)` of `τᵐ + τ̄ᵐ`.

use super::arith::{Ld, Ops};
use super::mul::{Expansion, Step, horner, window};
use super::narrow::Narrow;
use crate::math::numbers::{ByteOrder, Number};

/// The number `x`.
fn int(x: i64) -> Number {
    Number::new_from_i64(x, ByteOrder::LittleEndian)
}

/// What the products of a Koblitz curve over `GF(2ᵐ)` need.
#[derive(Clone, Debug)]
pub(super) struct Frobenius {
    /// `μ = (−1)^(1 − a)`.
    mu: i64,
    /// `τᵐ − 1 = d₀ + d₁·τ`.
    d0: Number,
    /// See `d0`.
    d1: Number,
    /// The norm of `τᵐ − 1`, the number of points.
    points: Number,
}

/// The norm `a² + μ·a·b + 2·b²` of `a + b·τ`.
fn norm(a: &Number, b: &Number, mu: i64) -> Number {
    let ab = a * b;
    let cross = if mu == 1 { ab } else { -ab };
    &(&(a * a) + &cross) + &(&int(2) * &(b * b))
}

/// The integers `(q₀, q₁)` that `λ = (n₀ + n₁·τ) / d` rounds to in `Z[τ]`, for `d > 0`, by
/// Solinas's rounding (the *Guide*'s algorithm 3.63), exactly: each `ηᵢ = λᵢ − round(λᵢ)` is held
/// as `eᵢ = d·ηᵢ`, and each comparison of the algorithm multiplied by `d`.
fn round(n0: &Number, n1: &Number, d: &Number, mu: i64) -> (Number, Number) {
    // round(n / d) = ⌊(2 n + d) / (2 d)⌋.
    let two_d = &int(2) * d;
    let nearest = |n: &Number| (&(&int(2) * n) + d).div_floor(&two_d);
    let (f0, f1) = (nearest(n0), nearest(n1));
    let e0 = n0 - &(&f0 * d);
    let e1 = n1 - &(&f1 * d);
    let mu_e1 = if mu == 1 { e1.clone() } else { -&e1 };
    let e = &(&int(2) * &e0) + &mu_e1;
    let minus = &e0 - &(&int(3) * &mu_e1);
    let plus = &e0 + &(&int(4) * &mu_e1);
    let (neg_d, neg_two_d) = (-d, -&two_d);
    let (mut h0, mut h1) = (0, 0);
    if &e >= d {
        if minus < neg_d {
            h1 = mu;
        } else {
            h0 = 1;
        }
    } else if plus >= two_d {
        h1 = mu;
    }
    if e < neg_d {
        if &minus >= d {
            h1 = -mu;
        } else {
            h0 = -1;
        }
    } else if plus < neg_two_d {
        h1 = -mu;
    }
    (&f0 + &int(h0), &f1 + &int(h1))
}

/// `k` modulo `δ = d₀ + d₁·τ` of norm `n`: `ρ = k − q·δ` for the rounding `q` of `k / δ`, which is
/// `k·δ̄ / n` with `δ̄ = (d₀ + μ·d₁) − d₁·τ`.
fn reduce(k: &Number, d0: &Number, d1: &Number, n: &Number, mu: i64) -> (Number, Number) {
    let mu_d1 = if mu == 1 { d1.clone() } else { -d1 };
    let s0 = d0 + &mu_d1;
    let (q0, q1) = round(&(k * &s0), &-&(k * d1), n, mu);
    // (q₀ + q₁·τ)·(d₀ + d₁·τ) = q₀·d₀ − 2·q₁·d₁ + (q₀·d₁ + q₁·d₀ + μ·q₁·d₁)·τ, as τ² = μ·τ − 2.
    let q1d1 = &q1 * d1;
    let r0 = &(k - &(&q0 * d0)) + &(&int(2) * &q1d1);
    let mu_q1d1 = if mu == 1 { q1d1 } else { -q1d1 };
    let r1 = -(&(&(&q0 * d1) + &(&q1 * d0)) + &mu_q1d1);
    (r0, r1)
}

/// The Lucas sequence `U_k(μ, 2)`: `τᵏ = U_k·τ − 2·U_(k−1)`.
fn lucas_u(mu: i64, k: u64) -> Number {
    Number::lucas_sequence(&int(mu), &int(2), k).0
}

/// The low `w` bits of `x` in two's complement, `x` modulo `2ʷ`.
fn low_bits(x: &Number, w: u32) -> i64 {
    (0..w).fold(0, |acc, i| acc | (i64::from(x.bit(u64::from(i))) << i))
}

/// What the width-`w` τ-adic form needs: `t_w`, with `τ ≡ t_w` modulo `τʷ`, and `α_u = β + γ·τ`
/// for each odd `u < 2^(w − 1)`, `u` modulo `τʷ` of least norm.
fn digit_set(mu: i64, w: u32) -> (i64, Vec<(i64, i64)>) {
    let modulus = 1i64 << w;
    let small = |x: &Number| {
        i64::try_from(x).unwrap_or_else(|_| unreachable!("the terms of a short Lucas sequence"))
    };
    let (uw, uw1) = (
        small(&lucas_u(mu, u64::from(w))),
        small(&lucas_u(mu, u64::from(w - 1))),
    );
    // `U_w` is odd, so it has an inverse modulo `2ʷ`; `t_w = 2·U_(w−1) / U_w` modulo `2ʷ`.
    let inverse = (1..modulus)
        .step_by(2)
        .find(|i| (i * uw).rem_euclid(modulus) == 1)
        .unwrap_or_else(|| unreachable!("U_w is odd"));
    let tw = (2 * uw1 * inverse).rem_euclid(modulus);
    // `τʷ = −2·U_(w−1) + U_w·τ`, of norm `2ʷ`.
    let (e0, e1, n) = (int(-2 * uw1), int(uw), int(modulus));
    let alphas = (1..modulus / 2)
        .step_by(2)
        .map(|u| {
            let (b, g) = reduce(&int(u), &e0, &e1, &n, mu);
            (small(&b), small(&g))
        })
        .collect();
    (tw, alphas)
}

/// The width-`w` τ-adic non-adjacent form of `r₀ + r₁·τ`, the lowest digit first (the *Guide*'s
/// algorithm 3.69): at an odd `r₀` the digit is `±u` for `u ≡ r₀ + r₁·t_w` modulo `2ʷ`, and `±α_u`
/// is taken away, which leaves a multiple of `τʷ`; then the element is divided by τ,
/// `(r₀ + r₁·τ) / τ = (r₁ + μ·r₀ / 2) − (r₀ / 2)·τ`.
///
/// It ends: outside a disc of radius `|α| / (√2 − 1)` about 0, `|α|` the largest of the digits,
/// each digit and division makes the norm smaller, and the script
/// `tests/math/ec/data/binary_tnaf_termination.py` runs the algorithm from every element of a box
/// that holds the disc, for both values of `μ` and the widths 2 to 8, and finds that each ends.
fn tnaf(
    mut r0: Number,
    mut r1: Number,
    mu: i64,
    w: u32,
    tw: i64,
    alphas: &[(i64, i64)],
) -> Vec<i8> {
    let modulus = 1i64 << w;
    let mut digits = Vec::new();
    while !(r0.is_zero() && r1.is_zero()) {
        if r0.bit(0) {
            let v = (low_bits(&r0, w) + low_bits(&r1, w) * tw).rem_euclid(modulus);
            let u = if v >= modulus / 2 { v - modulus } else { v };
            let (b, g) = alphas[(u.unsigned_abs() as usize - 1) / 2];
            if u > 0 {
                r0 -= int(b);
                r1 -= int(g);
            } else {
                r0 += int(b);
                r1 += int(g);
            }
            digits.push(u as i8);
        } else {
            digits.push(0);
        }
        // `r₀` is even now, so the halving is exact.
        let half = &r0 >> 1usize;
        r0 = if mu == 1 { &r1 + &half } else { &r1 - &half };
        r1 = -half;
    }
    digits
}

impl Frobenius {
    /// The Koblitz curve over `GF(2ᵐ)` with `a = 1` when `a_is_one`, and `a = 0` otherwise.
    pub(super) fn new(m: u32, a_is_one: bool) -> Frobenius {
        let mu = if a_is_one { 1 } else { -1 };
        let (um, vm) = Number::lucas_sequence(&int(mu), &int(2), u64::from(m));
        let um1 = lucas_u(mu, u64::from(m) - 1);
        // τᵐ − 1 = (−2·U_(m−1) − 1) + U_m·τ.
        let d0 = &(&int(-2) * &um1) - &int(1);
        let points = &(&(int(1) << m as usize) + &int(1)) - &vm;
        debug_assert!(norm(&d0, &um, mu) == points);
        Frobenius {
            mu,
            d0,
            d1: um,
            points,
        }
    }

    /// The number of points of the curve over `GF(2ᵐ)`.
    pub(super) fn points(&self) -> &Number {
        &self.points
    }

    /// The expansion of `k p` in the τ-adic non-adjacent form, `k` of any sign and length.
    pub(super) fn expansion<'f>(&self, ops: &Ops<'f>, p: &Ld<'f>, k: &Number) -> Expansion<'f> {
        let (digits, alphas) = self.digits(k);
        // α_u·P = β·P + γ·τ(P), with `|β|` and `|γ|` at most 11 for the widths up to 8: from the
        // multiples `j·P` and `j·τ(P)` up to the largest, by additions, with no inversion.
        let largest = alphas
            .iter()
            .map(|&(b, g)| b.unsigned_abs().max(g.unsigned_abs()))
            .max()
            .unwrap_or(1) as usize;
        let multiples = |q: Ld<'f>| {
            let mut all = vec![ops.identity(), q.clone()];
            for j in 2..=largest {
                let next = ops.add(&all[j - 1], &q);
                all.push(next);
            }
            all
        };
        let (of_p, of_tp) = (multiples(p.clone()), multiples(ops.frobenius(p)));
        let signed = |all: &[Ld<'f>], c: i64| {
            let q = &all[c.unsigned_abs() as usize];
            if c < 0 { ops.neg(q) } else { q.clone() }
        };
        let points: Vec<Ld<'f>> = alphas
            .iter()
            .map(|&(b, g)| ops.add(&signed(&of_p, b), &signed(&of_tp, g)))
            .collect();
        Expansion {
            digits,
            table: ops.affine_all(&points),
        }
    }

    /// The τ-adic digits of `k`, of any sign and length, and the digit set: `α_u = β + γ·τ` as
    /// `(β, γ)`, `u` from 1 by twos.
    pub(super) fn digits(&self, k: &Number) -> (Vec<i8>, Vec<(i64, i64)>) {
        let (r0, r1) = reduce(k, &self.d0, &self.d1, &self.points, self.mu);
        let w = window(self.points.bits());
        let (tw, alphas) = digit_set(self.mu, w);
        (tnaf(r0, r1, self.mu, w, tw, &alphas), alphas)
    }

    /// `k p` for `k` of any sign and length.
    pub(super) fn mul<'f>(&self, ops: &Ops<'f>, p: &Ld<'f>, k: &Number) -> Ld<'f> {
        if p.z.is_zero() {
            return ops.identity();
        }
        if let Some(narrow) = Narrow::new(ops) {
            let (digits, alphas) = self.digits(k);
            return narrow.mul_tnaf(ops, p, &digits, &alphas);
        }
        horner(ops, &[self.expansion(ops, p, k)], Step::Frobenius)
    }
}

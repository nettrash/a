//! The maps of [`Fpk`] to and over `GF(p)`: the Frobenius map by its matrix, the norm by a
//! resultant, the trace by the power sums of the roots of `g` from Newton's identities (Lidl and
//! Niederreiter, *Finite Fields*, §2.3, for the norm, the trace and the conjugates `x^(pⁱ)` as
//! the roots of `g`).

use super::{ExtensionField, Fp, Fpk, ext_arith};
use crate::math::numbers::Number;

/// The matrix of `a ↦ aᵖ`, column `j` the `k` coefficients of `x^(p·j) mod g`, at
/// `j·k..(j + 1)·k`: `xᵖ mod g` by squares modulo `g`, then its powers.
fn frobenius_matrix(f: &ExtensionField) -> Vec<Number> {
    let k = f.k;
    let p = f.base.modulus();
    let one = f.g.with_coeffs(&[Number::from(1u8)]);
    let xp =
        f.g.with_coeffs(&[Number::default(), Number::from(1u8)])
            .pow_mod(p, &f.g);
    let mut m = Vec::with_capacity(k * k);
    let mut col = one;
    for j in 0..k {
        if j > 0 {
            col = if j == 1 {
                xp.clone()
            } else {
                col.mul_mod(&xp, &f.g)
            };
        }
        for i in 0..k {
            m.push(col.coeff(i));
        }
    }
    m
}

/// `a^(pⁱ)` for `i` below `k`: `i` products by the matrix of the Frobenius map, each sum of
/// products reduced once modulo `p`.
pub(super) fn frobenius<'f>(a: &Fpk<'f>, i: usize) -> Fpk<'f> {
    let f = a.field();
    let k = f.k;
    if i == 0 || k == 1 {
        return a.clone();
    }
    let m = f.frob.get_or_init(|| frobenius_matrix(f));
    let mut v: Vec<Number> = a.c.iter().map(Fp::value).collect();
    for _ in 0..i {
        let mut acc = vec![Number::default(); k];
        for (j, aj) in v.iter().enumerate() {
            if aj.is_zero() {
                continue;
            }
            for (s, mij) in acc.iter_mut().zip(&m[j * k..(j + 1) * k]) {
                if !mij.is_zero() {
                    *s += &(aj * mij);
                }
            }
        }
        let p = f.base.modulus();
        v = acc.iter().map(|s| s.rem_euclid(p)).collect();
    }
    Fpk::of(f, v.iter().map(|x| f.base.element(x)).collect())
}

/// `N(a) = res(g, a)`, the product of `a` at the roots of the monic `g`.
pub(super) fn norm<'f>(a: &Fpk<'f>) -> Fp<'f> {
    let f = a.field();
    if f.k == 1 {
        return a.c[0].clone();
    }
    if let Some(c) = a.to_base() {
        // N(c) = cᵏ.
        return c.pow(&Number::from(f.k as u64));
    }
    let r = f.g.resultant(&ext_arith::to_polynomial(f, &a.c));
    f.base.element(&r)
}

/// The traces of `1, x, …, x^(k−1)`, the power sums `sᵢ` of the roots of
/// `g = xᵏ + e₁xᵏ⁻¹ + … + eₖ` by Newton's identities: `s₀ = k` and
/// `sᵢ = −(i·eᵢ + Σⱼ₌₁^(i−1) eⱼ·sᵢ₋ⱼ)`.
fn traces(f: &ExtensionField) -> Vec<Number> {
    let k = f.k;
    let base = &f.base;
    // e[j] = the coefficient of x^(k−j), for j in 1..k.
    let e: Vec<Fp<'_>> = (0..k)
        .map(|j| {
            if j == 0 {
                base.one()
            } else {
                base.element(&f.g.coeff(k - j))
            }
        })
        .collect();
    let mut s: Vec<Fp<'_>> = Vec::with_capacity(k);
    s.push(base.element(&Number::from(k as u64)));
    for i in 1..k {
        let mut t = &base.from_u64(i as u64) * &e[i];
        for j in 1..i {
            t += &e[j] * &s[i - j];
        }
        s.push(-t);
    }
    s.iter().map(Fp::value).collect()
}

/// `Tr(a) = Σ aᵢ·Tr(xⁱ)`.
pub(super) fn trace<'f>(a: &Fpk<'f>) -> Fp<'f> {
    let f = a.field();
    if f.k == 1 {
        return a.c[0].clone();
    }
    let t = f.traces.get_or_init(|| traces(f));
    let mut acc = Number::default();
    for (ai, ti) in a.c.iter().zip(t) {
        if !ai.is_zero() && !ti.is_zero() {
            acc += &(&ai.value() * ti);
        }
    }
    f.base.element(&acc)
}

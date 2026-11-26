//! The maps of [`Fpk`] to and over `GF(p)`: the Frobenius map by its matrix, the norm by a
//! resultant, the trace by the power sums of the roots of `g` from Newton's identities (Lidl and
//! Niederreiter, *Finite Fields*, §2.3, for the norm, the trace and the conjugates `x^(pⁱ)` as
//! the roots of `g`).

use super::ext::Repr;
use super::prime_form::{Arith, inline, with_form};
use super::{ExtensionField, Fp, Fpk, ext_arith};
use crate::math::ec::field::Field;
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
    if let Some(z) = &f.zech {
        return z.frobenius(a, i);
    }
    if let Repr::Words(w, _) = &a.r {
        // The matrix as words; each output coefficient a sum of k products reduced once.
        let m = f.frob_words.get_or_init(|| {
            f.frob
                .get_or_init(|| frobenius_matrix(f))
                .iter()
                .map(|x| inline::<1>(f.base.element(x).raw())[0])
                .collect()
        });
        let mut v = w.clone();
        for _ in 0..i {
            v = ext_arith::word_matrix_product(f, m, &v);
        }
        return Fpk::of_words(f, v);
    }
    let m = f.frob.get_or_init(|| frobenius_matrix(f));
    let mut v: Vec<Number> = a.c().iter().map(Fp::value).collect();
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
        return a.c()[0].clone();
    }
    if let Some(z) = &f.zech {
        return z.norm(a);
    }
    if f.k == 2 {
        // g = x² + g₁x + g₀ with roots r, s: N = (a₀ + a₁r)(a₀ + a₁s) = a₀² − g₁a₀a₁ + g₀a₁².
        let (a0, a1) = (&a.c()[0], &a.c()[1]);
        let mut n = a0.square();
        for (j, h) in &f.low {
            // xᵏ ≡ Σ hⱼxʲ: h₁ = −g₁, h₀ = −g₀.
            let h = f.base.element(h);
            if *j == 1 {
                n += &(&(a0 * a1) * &h);
            } else {
                n -= &(&a1.square() * &h);
            }
        }
        return n;
    }
    if let Some(c) = a.to_base() {
        // N(c) = cᵏ.
        return c.pow(&Number::from(f.k as u64));
    }
    let r = f.g.resultant(&ext_arith::to_polynomial(f, a.c()));
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
        return a.c()[0].clone();
    }
    let t = f.traces_form.get_or_init(|| {
        f.traces
            .get_or_init(|| traces(f))
            .iter()
            .map(|x| f.base.element(x).raw().clone())
            .collect()
    });
    // Σ aᵢ·Tr(xⁱ) in the base field's form, one match of the form for the whole sum.
    let v = with_form!(f.base.form_ref(), g => {
        let mut acc = g.zero();
        for (ai, ti) in a.c().iter().zip(t) {
            let (x, y) = (g.get(ai.raw()), g.get(ti));
            if !g.is_zero(x) && !g.is_zero(y) {
                acc = g.add(&acc, &g.mul(x, y));
            }
        }
        g.put(acc)
    });
    Fp::from_raw(&f.base, v)
}

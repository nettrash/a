//! The arithmetic of [`Fpk`]: products reduced modulo `g`, inverses by the extended Euclidean
//! algorithm, powers.
//!
//! A product is the schoolbook product of the coefficients in `GF(p)`, `k²` products, followed by
//! the reduction by the terms of `xᵏ mod g`, `(k − 1)·w` more for the `w` terms of `g` below `xᵏ`
//! (one or two for the binomials and trinomials [`ExtensionField::with_degree`] chooses, so far
//! as they exist). From the degree [`poly_min`] gives it is the product and the remainder of
//! [`ModPolynomial`] instead: sums of products reduced once, number-theoretic transforms or
//! Kronecker substitution for long polynomials, and Newton's division, which pay for the
//! conversion of the coefficients there.

use super::{ExtensionField, Fp, Fpk};
use crate::math::numbers::Number;
use crate::math::poly::ModPolynomial;

/// The degree from which the products of a field of the prime `p`, whose `g` has `w` nonzero
/// terms below `xᵏ`, go through [`ModPolynomial`]: where the two ways took the same time, measured
/// on the prime fields' forms for `k` up to 64. For a `g` of at most four such terms: 40 for
/// `p = 2` and 3 (whose many zero coefficients the schoolbook product skips), 20 for the other
/// primes of one word, 48 for two to eight limbs (Montgomery's form and its special shapes), 28
/// for nine, and 4 beyond (the prime fields' `Big` form, which is slower than the products of
/// `ModPolynomial` on pieces of the coefficients). Half of that, and at least 3, for a denser `g`,
/// whose reduction costs as much as the product.
pub(super) fn poly_min(p: &Number, w: usize) -> usize {
    let sparse = if *p <= 3u8 {
        40
    } else {
        match p.bits().div_ceil(64) {
            1 => 20,
            2..=8 => 48,
            9 => 28,
            _ => 4,
        }
    };
    if w > 4 { (sparse / 2).max(3) } else { sparse }
}

/// The polynomial of the coefficients `c` modulo `p`, sharing the modulus of `g`.
pub(super) fn to_polynomial(f: &ExtensionField, c: &[Fp<'_>]) -> ModPolynomial {
    let v: Vec<Number> = c.iter().map(Fp::value).collect();
    f.g.with_coeffs(&v)
}

/// The element of the polynomial `a`, of a degree below `k`.
pub(super) fn from_polynomial<'f>(f: &'f ExtensionField, a: &ModPolynomial) -> Fpk<'f> {
    debug_assert!(a.length() <= f.k);
    let mut c: Vec<Fp<'f>> = a.coeffs().iter().map(|x| f.base.element(x)).collect();
    c.resize(f.k, f.base.zero());
    Fpk::of(f, c)
}

/// Reduces the polynomial of the coefficients `c`, of any length, modulo `g`, leaving `k`
/// coefficients: from the top down, `xⁱ = x^(i−k)·xᵏ` replaced by the terms of `xᵏ mod g`.
pub(super) fn reduce<'f>(f: &'f ExtensionField, c: &mut Vec<Fp<'f>>) {
    let k = f.k;
    if c.len() > k {
        let low: Vec<(usize, Fp<'f>)> =
            f.low.iter().map(|(j, h)| (*j, f.base.element(h))).collect();
        for i in (k..c.len()).rev() {
            if c[i].is_zero() {
                continue;
            }
            let t = c[i].clone();
            for (j, h) in &low {
                c[i - k + j] += &t * h;
            }
        }
        c.truncate(k);
    }
    c.resize(k, f.base.zero());
}

/// `a·b`.
pub(super) fn mul<'f>(a: &Fpk<'f>, b: &Fpk<'f>) -> Fpk<'f> {
    let f = a.field();
    let k = f.k;
    if k == 1 {
        return Fpk::of(f, vec![&a.c[0] * &b.c[0]]);
    }
    if k >= f.poly_min {
        let r = to_polynomial(f, &a.c).mul_mod(&to_polynomial(f, &b.c), &f.g);
        return from_polynomial(f, &r);
    }
    let mut t = vec![f.base.zero(); 2 * k - 1];
    for (i, x) in a.c.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.c.iter().enumerate() {
            if !y.is_zero() {
                t[i + j] += x * y;
            }
        }
    }
    reduce(f, &mut t);
    Fpk::of(f, t)
}

/// The inverse, `None` for 0.
pub(super) fn inverse<'f>(a: &Fpk<'f>) -> Option<Fpk<'f>> {
    let f = a.field();
    if f.k == 1 {
        return a.c[0].inverse().map(|v| Fpk::of(f, vec![v]));
    }
    if a.is_zero() {
        return None;
    }
    if let Some(c) = a.to_base() {
        let mut v = vec![f.base.zero(); f.k];
        v[0] = c.inverse().expect("not zero");
        return Some(Fpk::of(f, v));
    }
    // g is irreducible and a is not a multiple of it, so the gcd is 1 and s·a ≡ 1 (mod g), s of
    // a degree below k.
    let (d, s, _) = to_polynomial(f, &a.c).xgcd(&f.g);
    debug_assert!(d.is_one());
    Some(from_polynomial(f, &s))
}

/// `aᵉ` for `e ≥ 0`.
pub(super) fn pow<'f>(a: &Fpk<'f>, e: &Number) -> Fpk<'f> {
    let f = a.field();
    if e.is_zero() {
        return f.one();
    }
    if a.is_zero() {
        return f.zero();
    }
    // The group of units has the order pᵏ − 1.
    let units = &f.q - &Number::from(1u8);
    let e = if *e >= units {
        e.rem_euclid(&units)
    } else {
        e.clone()
    };
    if e.is_zero() {
        return f.one();
    }
    if f.k == 1 {
        return Fpk::of(f, vec![a.c[0].pow(&e)]);
    }
    let mut r = a.clone();
    for i in (0..e.bits() - 1).rev() {
        r = r.square();
        if e.bit(i) {
            r = &r * a;
        }
    }
    r
}

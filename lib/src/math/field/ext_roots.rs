//! Square roots, the quadratic character and multiplicative orders in [`Fpk`].
//!
//! Square roots for odd `p` are Tonelli and Shanks's algorithm in `GF(pᵏ)` (Cohen, *A Course in
//! Computational Algebraic Number Theory*, Algorithm 1.5.1, with `GF(pᵏ)` for `Z/pZ`), after the
//! quadratic character through the norm: `a^((pᵏ − 1)/2) = N(a)^((p − 1)/2)`. For `p = 2`,
//! squaring is a bijection and the root is the inverse of the Frobenius map. Orders come from the
//! factorisation of `pᵏ − 1`, which is that of the values of the cyclotomic polynomials `Φ_d(p)`
//! for the divisors `d` of `k`, each prime power taken out of the order while the power stays 1.

use super::{ExtensionField, Fp, Fpk, ext_arith, ext_maps};
use crate::math::numbers::{ByteOrder, Factorization, Number};

/// The data of Tonelli and Shanks's algorithm for a field of odd characteristic:
/// `pᵏ − 1 = 2ˢ·t` with `t` odd, and `z^t` for a non-square `z`.
#[derive(Clone, Debug)]
pub(super) struct Shanks {
    /// `s`.
    s: u64,
    /// `(t − 1)/2`.
    half: Number,
    /// The coefficients of `z^t`, of order `2ˢ`.
    c: Vec<Number>,
}

/// The data of Tonelli and Shanks's algorithm: the first non-square `z` among the constants
/// `2, 3, …` for odd `k` (a non-square modulo `p` stays one, `N(c) = cᵏ`), and among `x, x + 1, …`
/// (the digits of `p, p + 1, …` in base `p`) for even `k`, where every constant is a square.
fn shanks(f: &ExtensionField) -> Shanks {
    let units = &f.q - &Number::from(1u8);
    let s = units.trailing_zeros().expect("pᵏ − 1 is not zero");
    let t = &units >> s as usize;
    let half = &t >> 1usize;
    let z = if f.k % 2 == 1 {
        let c = (2u64..)
            .map(|n| f.base.from_u64(n))
            .find(|c| c.legendre() == -1)
            .expect("an odd prime has non-squares");
        f.from_base(&c)
    } else {
        let mut n = f.base.modulus().clone();
        loop {
            let z = f.digits_element(&n);
            if legendre(&z) == -1 {
                break z;
            }
            n += &Number::from(1u8);
        }
    };
    let c = ext_arith::pow(&z, &t).c().iter().map(Fp::value).collect();
    Shanks { s, half, c }
}

/// The quadratic character.
pub(super) fn legendre(a: &Fpk<'_>) -> i32 {
    if a.is_zero() {
        return 0;
    }
    let f = a.field();
    if *f.base.modulus() == 2u8 {
        return 1;
    }
    if let Some(z) = &f.zech {
        return z.legendre(a);
    }
    ext_maps::norm(a).legendre()
}

/// Of `r` and `−r`, the one whose highest nonzero coefficient is below `p/2`, the smaller in the
/// order of the coefficients from the top down.
fn smaller<'f>(r: Fpk<'f>) -> Fpk<'f> {
    let p = r.field().base.modulus();
    match r.c().iter().rev().find(|c| !c.is_zero()) {
        Some(top) if &(top.value() << 1usize) > p => -r,
        _ => r,
    }
}

/// A square root for `p = 2`: `√a = Σ a₂ᵢ·xⁱ + √x·Σ a₂ᵢ₊₁·xⁱ`, `√x = x^(2ᵏ⁻¹) mod g`.
fn sqrt_two<'f>(a: &Fpk<'f>) -> Fpk<'f> {
    let f = a.field();
    let even: Vec<Fp<'f>> = a.c().iter().step_by(2).cloned().collect();
    let odd: Vec<Fp<'f>> = a.c().iter().skip(1).step_by(2).cloned().collect();
    let even = f.reduced(even);
    if odd.iter().all(Fp::is_zero) {
        return even;
    }
    let root = f.root_x.get_or_init(|| {
        let e = Number::from(1u8) << (f.k - 1);
        let x = f.g.with_coeffs(&[Number::default(), Number::from(1u8)]);
        x.pow_mod(&e, &f.g).coeffs()
    });
    let root = f.element(root);
    &even + &(&root * &f.reduced(odd))
}

/// A square root, the smaller of the two, or `None` for a non-square.
pub(super) fn sqrt<'f>(a: &Fpk<'f>) -> Option<Fpk<'f>> {
    let f = a.field();
    if a.is_zero() {
        return Some(a.clone());
    }
    if *f.base.modulus() == 2u8 {
        return Some(sqrt_two(a));
    }
    if let Some(z) = &f.zech {
        return z.sqrt_smaller(a);
    }
    if let Some(c) = a.to_base() {
        if c.legendre() == 1 {
            return Some(f.from_base(&c.sqrt().expect("a square")));
        }
        if f.k % 2 == 1 {
            // N(c) = cᵏ is not a square.
            return None;
        }
    }
    if legendre(a) != 1 {
        return None;
    }
    let sh = f.shanks.get_or_init(|| shanks(f));
    let u = ext_arith::pow(a, &sh.half);
    let mut r = a * &u; // a^((t + 1)/2)
    let mut w = &r * &u; // a^t, of an order dividing 2^(s−1) since a is a square
    let mut c = f.element(&sh.c);
    let mut m = sh.s;
    // r² = a·w throughout; each step halves the order of w.
    while !w.is_one() {
        let mut i = 0;
        let mut w2 = w.clone();
        while !w2.is_one() {
            w2 = w2.square();
            i += 1;
            assert!(i < m, "a square has a root");
        }
        let mut b = c;
        for _ in 0..m - i - 1 {
            b = b.square();
        }
        m = i;
        c = b.square();
        w = &w * &c;
        r = &r * &b;
    }
    Some(smaller(r))
}

/// The divisors of `k`, increasing.
fn divisors(k: usize) -> Vec<usize> {
    (1..=k).filter(|&d| k.is_multiple_of(d)).collect()
}

/// The Möbius function of `n ≥ 1`.
fn mobius(mut n: usize) -> i32 {
    let mut m = 1;
    let mut r = 2;
    while r * r <= n {
        if n.is_multiple_of(r) {
            n /= r;
            if n.is_multiple_of(r) {
                return 0;
            }
            m = -m;
        }
        r += 1;
    }
    if n > 1 { -m } else { m }
}

/// The factorisation of `pᵏ − 1 = Π_{d | k} Φ_d(p)`, with `Φ_d(p) = Π_{e | d} (pᵉ − 1)^μ(d/e)`.
pub(super) fn factor_unit_order(p: &Number, k: usize) -> Factorization {
    let one = Number::from(1u8);
    let mut all = Vec::new();
    for d in divisors(k) {
        let (mut num, mut den) = (one.clone(), one.clone());
        for e in divisors(d) {
            let v = &p.pow(u32::try_from(e).expect("a degree below 2^32")) - &one;
            match mobius(d / e) {
                1 => num = &num * &v,
                -1 => den = &den * &v,
                _ => {}
            }
        }
        let phi = &num / &den;
        if phi > one {
            all.extend(phi.factor().into_factors());
        }
    }
    Factorization::new(all, ByteOrder::LittleEndian)
}

/// The multiplicative order, `None` for 0.
pub(super) fn multiplicative_order(a: &Fpk<'_>) -> Option<Number> {
    if a.is_zero() {
        return None;
    }
    let f = a.field();
    if let Some(z) = &f.zech {
        return z.order(a);
    }
    let mut n = &f.q - &Number::from(1u8);
    for (r, e) in f.factored_unit_order().factors() {
        n = &n / &r.pow(*e);
        let mut y = ext_arith::pow(a, &n);
        while !y.is_one() {
            y = ext_arith::pow(&y, r);
            n = &n * r;
        }
    }
    Some(n)
}

/// Whether `a` generates the group of units.
pub(super) fn is_primitive(a: &Fpk<'_>) -> bool {
    if a.is_zero() {
        return false;
    }
    let f = a.field();
    let units = &f.q - &Number::from(1u8);
    if let Some(z) = &f.zech {
        return z.order(a).is_some_and(|n| n == units);
    }
    f.factored_unit_order()
        .factors()
        .iter()
        .all(|(r, _)| !ext_arith::pow(a, &(&units / r)).is_one())
}

/// The first primitive element in the order of the coefficients from the top down, past the
/// constants for a degree of 2 or more (their orders divide `p − 1`).
pub(super) fn primitive_element(f: &ExtensionField) -> Fpk<'_> {
    if f.q == 2u8 {
        return f.one();
    }
    let mut n = if f.k == 1 {
        Number::from(2u8)
    } else {
        f.base.modulus().clone()
    };
    loop {
        let a = f.digits_element(&n);
        if is_primitive(&a) {
            return a;
        }
        n += &Number::from(1u8);
    }
}

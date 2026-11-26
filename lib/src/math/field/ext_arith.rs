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

use super::ext::Repr;
use super::prime_form::{Arith, Value, inline, with_form};
use super::{ExtensionField, Fp, Fpk};
use crate::math::numbers::Number;
use crate::math::poly::ModPolynomial;

/// The degree from which the products of a field of the prime `p`, whose `g` has `w` nonzero
/// terms below `xᵏ`, go through [`ModPolynomial`] (its products and remainder, which pay for
/// the conversion of the coefficients): where the two ways take the same time. For a `g` of at
/// most four such terms: 320 for `p = 2`, 160 for the other primes of one word (those below `2⁶³`
/// take [`word_product`] up to [`WORD_MAX`]), 130 for two and three limbs (where `ModPolynomial`'s
/// products change method), 100 for four, 36 for five, 48 for six, 32 for seven and eight, 28
/// for nine and 8 beyond (the prime fields' Wide form). For a denser `g`, whose reduction costs
/// about as much as the product: 224, 64, 80 for two limbs and 52 for three, 40, 24, 24, 15, 14
/// and 6.
///
/// Measured on an Apple M4 Max: chains of products on random elements, each way forced in turn and
/// alternated (21 and 31 samples), `k` from 2 to 512, generic primes of full and of short top
/// limbs (300 to 2048 bits), BN254's, P-256's, P-384's and BLS12-381's, and trinomials, binomials
/// and random dense moduli. Where the prime's form changes the crossover, the bound is the lower
/// one, so that no field is slower than with the former bounds (40, 20, 48, 28 and 4, halved for a
/// dense `g`): for a sparse `g` the schoolbook would stay the faster up to 130 for BN254's and
/// P-256's primes and up to 52–55 for P-384's and BLS12-381's, where a generic prime of six full
/// limbs crosses at 35, so that the bounds of six limbs stay as they were.
pub(super) fn poly_min(p: &Number, w: usize) -> usize {
    let (sparse, dense) = if *p == 2u8 {
        (320, 224)
    } else {
        match p.bits().div_ceil(64) {
            1 => (160, 64),
            2 => (130, 80),
            3 => (130, 52),
            4 => (100, 40),
            5 => (36, 24),
            6 => (48, 24),
            7 | 8 => (32, 15),
            9 => (28, 14),
            _ => (8, 6),
        }
    };
    if w > 4 { dense } else { sparse }
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

/// `a + b mod p` for words below `p < 2⁶³`.
#[inline(always)]
fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

/// `a − b mod p` for words below `p`.
#[inline(always)]
fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { a + p - b }
}

/// `a + b`: coefficient by coefficient, on the words for a word prime, by Zech's table of
/// `log(1 + hⁱ)` for the fields of logarithms.
pub(super) fn add<'f>(a: &Fpk<'f>, b: &Fpk<'f>) -> Fpk<'f> {
    let f = a.field();
    match (&a.r, &b.r) {
        (Repr::Log(x, _), Repr::Log(y, _)) => Fpk::of_log(f, f.zech().add_log(*x, *y)),
        (Repr::Words(x, _), Repr::Words(y, _)) => {
            let p = f.word.0;
            Fpk::of_words(
                f,
                x.iter().zip(y).map(|(&u, &v)| add_mod(u, v, p)).collect(),
            )
        }
        _ => Fpk::of(f, a.c().iter().zip(b.c()).map(|(x, y)| x + y).collect()),
    }
}

/// `a − b`.
pub(super) fn sub<'f>(a: &Fpk<'f>, b: &Fpk<'f>) -> Fpk<'f> {
    let f = a.field();
    match (&a.r, &b.r) {
        (Repr::Log(x, _), Repr::Log(y, _)) => {
            let z = f.zech();
            Fpk::of_log(f, z.add_log(*x, z.neg_log(*y)))
        }
        (Repr::Words(x, _), Repr::Words(y, _)) => {
            let p = f.word.0;
            Fpk::of_words(
                f,
                x.iter().zip(y).map(|(&u, &v)| sub_mod(u, v, p)).collect(),
            )
        }
        _ => Fpk::of(f, a.c().iter().zip(b.c()).map(|(x, y)| x - y).collect()),
    }
}

/// `−a`.
pub(super) fn neg<'f>(a: &Fpk<'f>) -> Fpk<'f> {
    let f = a.field();
    match &a.r {
        Repr::Log(x, _) => Fpk::of_log(f, f.zech().neg_log(*x)),
        Repr::Words(x, _) => {
            let p = f.word.0;
            Fpk::of_words(f, x.iter().map(|&u| sub_mod(0, u, p)).collect())
        }
        Repr::General(c) => Fpk::of(f, c.iter().map(|x| -x).collect()),
    }
}

/// `a·b`.
pub(super) fn mul<'f>(a: &Fpk<'f>, b: &Fpk<'f>) -> Fpk<'f> {
    let f = a.field();
    let k = f.k;
    match (&a.r, &b.r) {
        (Repr::Log(x, _), Repr::Log(y, _)) => return Fpk::of_log(f, f.zech().mul_log(*x, *y)),
        (Repr::Words(x, _), Repr::Words(y, _)) if word_products(f) => {
            return Fpk::of_words(f, word_product(f, x, Some(y)));
        }
        _ => {}
    }
    if k == 1 {
        return Fpk::of(f, vec![&a.c()[0] * &b.c()[0]]);
    }
    if k >= f.poly_min {
        let r = to_polynomial(f, a.c()).mul_mod(&to_polynomial(f, b.c()), &f.g);
        return from_polynomial(f, &r);
    }
    Fpk::of(
        f,
        with_form!(f.base.form_ref(), g => schoolbook(g, f, a.c(), Some(b.c()))),
    )
}

/// `a²`: the schoolbook square, `k(k + 1)/2` products, and the reduction, or the product of
/// [`ModPolynomial`] from its degree.
pub(super) fn square<'f>(a: &Fpk<'f>) -> Fpk<'f> {
    let f = a.field();
    match &a.r {
        Repr::Log(x, _) => return Fpk::of_log(f, f.zech().mul_log(*x, *x)),
        Repr::Words(x, _) if word_products(f) => return Fpk::of_words(f, word_product(f, x, None)),
        _ => {}
    }
    if f.k == 1 || f.k >= f.poly_min {
        return mul(a, a);
    }
    Fpk::of(
        f,
        with_form!(f.base.form_ref(), g => schoolbook(g, f, a.c(), None)),
    )
}

/// REDC of `t < p·2⁶⁴` modulo the word prime `p`: `t / 2⁶⁴ mod p`, below `p`.
#[inline(always)]
fn redc_word(t: u128, p: u64, minv: u64) -> u64 {
    let (lo, hi) = (t as u64, (t >> 64) as u64);
    // m p agrees with -t in the low word: their sum carries exactly when that word is not 0.
    let m = lo.wrapping_mul(minv);
    let mp = u128::from(m) * u128::from(p);
    let u = u128::from(hi) + (mp >> 64) + u128::from(lo != 0);
    let p = u128::from(p);
    (if u >= p { u - p } else { u }) as u64
}

/// `a + b mod p` for words below `p`.
#[inline(always)]
fn add_word(a: u64, b: u64, p: u64) -> u64 {
    let (s, c) = a.overflowing_add(b);
    if c || s >= p { s.wrapping_sub(p) } else { s }
}

/// The most coefficients of the products of [`word_product`], on the stack, and the degree up to
/// which the fields of word primes take them, in place of [`ModPolynomial`]'s products, for a `g`
/// of at most four terms below `xᵏ`. Measured on an Apple M4 Max (chains of products and of
/// squares on random elements, the two ways alternated, k from 16 to 256, primes of 2 to 63 bits):
/// the words take 0.4–0.9 times the time of `ModPolynomial::mul_mod` and its conversions at 256
/// (0.89 for a prime of 63 bits), less below.
pub(super) const WORD_MAX: usize = 256;

/// [`WORD_MAX`] for a denser `g`, whose reduction on the words costs a product and a REDC for each
/// of its terms and coefficients: at 128 coefficients the words took 1.4 times the time of
/// `ModPolynomial` for a prime of 63 bits and 0.9 times for 3, at 64 0.8 and 0.55 times.
const WORD_MAX_DENSE: usize = 64;

/// Whether the products of `f`, of a word prime, take [`word_product`].
#[inline]
fn word_products(f: &ExtensionField) -> bool {
    f.k <= if f.low.len() > 4 {
        WORD_MAX_DENSE
    } else {
        WORD_MAX
    }
}

/// The degree up to which the inverses of a field of the word prime `p` take [`word_inverse`],
/// whose division steps each invert a leading coefficient by the binary extended Euclid on its
/// word (a cost that grows with the bits of `p`), in place of `ModPolynomial::xgcd`. Measured on
/// an Apple M4 Max (chains of inverses, alternated, k from 2 to 128, primes of 2 to 63 bits): the
/// two take the same time at about `30000 / b²` coefficients for a prime of `b ≥ 9` bits (64 at
/// 17 and 20 bits, 32 at 31, 16 at 40, 12 at 50, 8 at 61), at 80 for the primes of 3 to 8 bits,
/// and beyond 128 for 3.
fn word_inverse_max(p: u64) -> usize {
    let b = (64 - p.leading_zeros()) as usize;
    match b {
        0..=2 => 128,
        3..=8 => 80,
        _ => (30000 / (b * b)).min(64),
    }
}

/// The schoolbook product (or square) for a prime of one word in Montgomery's form, with lazy
/// reduction: the products of each output coefficient summed in 128 bits, at most `2⁶⁴ / p` of
/// them, so that the sum stays below `p·2⁶⁴`, and reduced once by REDC (a sum of products of
/// residues `xR·yR` is `(Σxy)R²`, whose REDC is the residue of `Σxy`); a square sums the products
/// `xᵢxⱼ` for `i < j` once and doubles them. Then the reduction modulo `g` by the terms of
/// `xᵏ mod g`, each a product and a REDC. The words of the coefficients are read and written in
/// the form as they are, on the stack, for at most [`WORD_MAX`] coefficients.
pub(super) fn word_product(f: &ExtensionField, a: &[u64], b: Option<&[u64]>) -> Vec<u64> {
    // Buffers of the class of the degree, so that the zeros they start with cost little.
    match f.k {
        0..=4 => word_product_in::<4>(f, a, b),
        5..=8 => word_product_in::<8>(f, a, b),
        9..=16 => word_product_in::<16>(f, a, b),
        17..=32 => word_product_in::<32>(f, a, b),
        33..=64 => word_product_in::<64>(f, a, b),
        65..=128 => word_product_in::<128>(f, a, b),
        _ => word_product_in::<WORD_MAX>(f, a, b),
    }
}

/// [`word_product`] with buffers of `W ≥ k` coefficients.
fn word_product_in<const W: usize>(f: &ExtensionField, a: &[u64], b: Option<&[u64]>) -> Vec<u64> {
    let k = f.k;
    let (p, minv) = f.word;
    let mut x = [0u64; W];
    let mut y = [0u64; W];
    x[..k].copy_from_slice(a);
    if let Some(b) = b {
        y[..k].copy_from_slice(b);
    }
    // Up to ⌊(2⁶⁴ − 1)/p⌋ products of at most (p − 1)² stay below p·2⁶⁴, REDC's bound.
    let per = (u64::MAX / p).max(1);
    let mut t = [[0u64; W], [0u64; W]];
    let t = t.as_flattened_mut();
    // The sum of the products xᵢyⱼ for i + j = o with lo ≤ i ≤ hi: one REDC when there are at most
    // `per` of them, with no branch in the sum (zeros are summed too); else one per `per`.
    let sum = |o: usize, lo: usize, hi: usize, y: &[u64]| {
        if (hi + 1 - lo) as u64 <= per {
            let mut acc = 0u128;
            for i in lo..=hi {
                acc += u128::from(x[i]) * u128::from(y[o - i]);
            }
            return redc_word(acc, p, minv);
        }
        let (mut acc, mut r) = (0u128, 0u64);
        for (n, i) in (lo..=hi).enumerate() {
            acc += u128::from(x[i]) * u128::from(y[o - i]);
            if (n as u64 + 1).is_multiple_of(per) {
                r = add_word(r, redc_word(acc, p, minv), p);
                acc = 0;
            }
        }
        add_word(r, redc_word(acc, p, minv), p)
    };
    for (o, out) in t.iter_mut().enumerate().take(2 * k - 1) {
        let lo = o.saturating_sub(k - 1);
        let hi = o.min(k - 1);
        *out = match b {
            Some(_) => sum(o, lo, hi, &y),
            None => {
                // i < o − i: the pairs below the middle, doubled, and the middle's square.
                let cross = if lo < o.div_ceil(2) {
                    sum(o, lo, (o - 1) / 2, &x)
                } else {
                    0
                };
                let mid = if o % 2 == 0 && x[o / 2] != 0 {
                    redc_word(u128::from(x[o / 2]) * u128::from(x[o / 2]), p, minv)
                } else {
                    0
                };
                add_word(add_word(cross, cross, p), mid, p)
            }
        };
    }
    let low = low_terms(f);
    for i in (k..2 * k - 1).rev() {
        let c = t[i];
        if c == 0 {
            continue;
        }
        for (j, h) in low {
            let m = redc_word(u128::from(c) * u128::from(inline::<1>(h)[0]), p, minv);
            t[i - k + j] = add_word(t[i - k + j], m, p);
        }
    }
    t[..k].to_vec()
}

/// `M·v` for the matrix `M` of residues of one word, column `j` at `j·k..(j + 1)·k`: each output
/// coefficient the sum of its `k` products in 128 bits, reduced once (at most `2⁶⁴ / p` products a
/// reduction, as in [`word_product`]).
pub(super) fn word_matrix_product(f: &ExtensionField, m: &[u64], v: &[u64]) -> Vec<u64> {
    let k = f.k;
    let (p, minv) = f.word;
    let per = (u64::MAX / p).max(1);
    (0..k)
        .map(|i| {
            let (mut acc, mut n, mut r) = (0u128, 0u64, 0u64);
            for (j, &x) in v.iter().enumerate() {
                acc += u128::from(x) * u128::from(m[j * k + i]);
                n += 1;
                if n == per {
                    r = add_word(r, redc_word(acc, p, minv), p);
                    acc = 0;
                    n = 0;
                }
            }
            add_word(r, redc_word(acc, p, minv), p)
        })
        .collect()
}

/// `x·y` of Montgomery residues of one word.
#[inline(always)]
fn mont_word(x: u64, y: u64, p: u64, minv: u64) -> u64 {
    redc_word(u128::from(x) * u128::from(y), p, minv)
}

/// The inverse of a nonzero Montgomery residue of one word: the binary extended Euclid on its
/// value, and back into the form.
fn inverse_residue(x: u64, f: &ExtensionField) -> u64 {
    let (p, minv) = f.word;
    let v = redc_word(u128::from(x), p, minv);
    let (mut u, mut w, mut x1, mut x2) = (v, p, 1u64, 0u64);
    let half = |x: u64| {
        if x & 1 == 0 {
            x >> 1
        } else {
            (x >> 1) + (p >> 1) + 1
        }
    };
    while u != 1 && w != 1 {
        while u & 1 == 0 {
            u >>= 1;
            x1 = half(x1);
        }
        while w & 1 == 0 {
            w >>= 1;
            x2 = half(x2);
        }
        if u >= w {
            u -= w;
            x1 = sub_mod(x1, x2, p);
        } else {
            w -= u;
            x2 = sub_mod(x2, x1, p);
        }
    }
    let inv = if u == 1 { x1 } else { x2 };
    // The residue inv·R: Montgomery's product by R² mod p.
    mont_word(inv, words_r2(f), p, minv)
}

/// `R² mod p` of the word prime, `R = 2⁶⁴`: Montgomery's product of a value by it is the value's
/// residue `x R`.
fn words_r2(f: &ExtensionField) -> u64 {
    let (p, _) = f.word;
    let r = (u128::from(u64::MAX) + 1) % u128::from(p); // R mod p
    ((r * r) % u128::from(p)) as u64 // R² mod p
}

/// The inverse of the element of the words `w` modulo `g` (for a word prime, `k` up to
/// [`word_inverse_max`]): the extended Euclidean algorithm on polynomials of residues, `s·a ≡ r (mod g)`
/// carried down to a constant `r`, then `s / r`.
fn word_inverse(f: &ExtensionField, w: &[u64]) -> Vec<u64> {
    let (p, minv) = f.word;
    let k = f.k;
    let mul = |x: u64, y: u64| mont_word(x, y, p, minv);
    // g, monic, from the terms of xᵏ mod g: gⱼ = −(those terms).
    let mut r0 = vec![0u64; k + 1];
    for (j, h) in low_terms(f) {
        r0[*j] = sub_mod(0, inline::<1>(h)[0], p);
    }
    r0[k] = inline::<1>(f.base.one().raw())[0];
    let trim = |v: &mut Vec<u64>| {
        while v.len() > 1 && *v.last().expect("not empty") == 0 {
            v.pop();
        }
    };
    let mut r1 = w.to_vec();
    trim(&mut r1);
    let (mut s0, mut s1) = (vec![0u64], vec![inline::<1>(f.base.one().raw())[0]]);
    while r1.len() > 1 {
        // r0 = q r1 + rem, by the inverse of r1's leading coefficient.
        let lead = inverse_residue(*r1.last().expect("not empty"), f);
        let mut rem = r0.clone();
        let mut q = vec![0u64; rem.len().saturating_sub(r1.len()) + 1];
        while rem.len() >= r1.len() && !(rem.len() == 1 && rem[0] == 0) {
            let shift = rem.len() - r1.len();
            let c = mul(*rem.last().expect("not empty"), lead);
            q[shift] = c;
            for (i, &b) in r1.iter().enumerate() {
                rem[i + shift] = sub_mod(rem[i + shift], mul(c, b), p);
            }
            rem.pop();
            trim(&mut rem);
            if rem.len() < r1.len() {
                break;
            }
        }
        // s0 − q s1.
        let mut s2 = vec![0u64; (q.len() + s1.len()).max(s0.len())];
        s2[..s0.len()].copy_from_slice(&s0);
        for (i, &qi) in q.iter().enumerate() {
            if qi == 0 {
                continue;
            }
            for (j, &sj) in s1.iter().enumerate() {
                s2[i + j] = sub_mod(s2[i + j], mul(qi, sj), p);
            }
        }
        trim(&mut s2);
        r0 = core::mem::replace(&mut r1, rem);
        s0 = core::mem::replace(&mut s1, s2);
    }
    // r1 is a nonzero constant c: s1·a ≡ c, so a⁻¹ = s1 / c.
    let ci = inverse_residue(r1[0], f);
    let mut out: Vec<u64> = s1.iter().map(|&x| mul(x, ci)).collect();
    out.resize(k, 0);
    out
}

/// The terms of `xᵏ mod g` in the base field's form, made on the first use.
fn low_terms(f: &ExtensionField) -> &[(usize, Value)] {
    f.low_form.get_or_init(|| {
        f.low
            .iter()
            .map(|(j, h)| (*j, f.base.element(h).raw().clone()))
            .collect()
    })
}

/// The schoolbook product of `a` and `b` (the square of `a` for `None`), reduced by the terms of
/// `xᵏ mod g`, in the base field's form `g`: one match of the form for the whole product, each
/// coefficient's sum built in place.
fn schoolbook<'f, G: Arith>(
    g: &G,
    f: &'f ExtensionField,
    a: &[Fp<'f>],
    b: Option<&[Fp<'f>]>,
) -> Vec<Fp<'f>> {
    let k = f.k;
    let zero = || Fp::from_raw(&f.base, g.put(g.zero()));
    let mut t: Vec<Fp<'f>> = (0..2 * k - 1).map(|_| zero()).collect();
    let x: Vec<&G::E> = a.iter().map(|c| g.get(c.raw())).collect();
    match b {
        Some(b) => {
            let y: Vec<&G::E> = b.iter().map(|c| g.get(c.raw())).collect();
            for (i, xi) in x.iter().enumerate() {
                if g.is_zero(xi) {
                    continue;
                }
                for (j, yj) in y.iter().enumerate() {
                    if !g.is_zero(yj) {
                        let p = g.mul(xi, yj);
                        g.update(t[i + j].raw_mut(), |s| g.add(s, &p));
                    }
                }
            }
        }
        None => {
            for (i, xi) in x.iter().enumerate() {
                if g.is_zero(xi) {
                    continue;
                }
                let s = g.sqr(xi);
                g.update(t[2 * i].raw_mut(), |u| g.add(u, &s));
                for (j, xj) in x.iter().enumerate().skip(i + 1) {
                    if !g.is_zero(xj) {
                        let p = g.mul(xi, xj);
                        let p = g.add(&p, &p);
                        g.update(t[i + j].raw_mut(), |u| g.add(u, &p));
                    }
                }
            }
        }
    }
    let low = low_terms(f);
    for i in (k..2 * k - 1).rev() {
        let c = g.get(t[i].raw()).clone();
        if g.is_zero(&c) {
            continue;
        }
        for (j, h) in low {
            let p = g.mul(&c, g.get(h));
            g.update(t[i - k + j].raw_mut(), |u| g.add(u, &p));
        }
    }
    t.truncate(k);
    t
}

/// The inverse, `None` for 0.
pub(super) fn inverse<'f>(a: &Fpk<'f>) -> Option<Fpk<'f>> {
    let f = a.field();
    if f.k == 1 {
        return a.c()[0].inverse().map(|v| Fpk::of(f, vec![v]));
    }
    if let Some(z) = &f.zech {
        return z.inverse(a);
    }
    if a.is_zero() {
        return None;
    }
    if let Repr::Words(w, _) = &a.r
        && f.k <= word_inverse_max(f.word.0)
    {
        return Some(Fpk::of_words(f, word_inverse(f, w)));
    }
    if let Some(c) = a.to_base() {
        let mut v = vec![f.base.zero(); f.k];
        v[0] = c.inverse().expect("not zero");
        return Some(Fpk::of(f, v));
    }
    // g is irreducible and a is not a multiple of it, so the gcd is 1 and s·a ≡ 1 (mod g), s of
    // a degree below k.
    let (d, s, _) = to_polynomial(f, a.c()).xgcd(&f.g);
    debug_assert!(d.is_one());
    Some(from_polynomial(f, &s))
}

/// `aᵉ` for `e ≥ 0`.
pub(super) fn pow<'f>(a: &Fpk<'f>, e: &Number) -> Fpk<'f> {
    let f = a.field();
    if let Some(z) = &f.zech {
        return z.pow(a, e);
    }
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
        return Fpk::of(f, vec![a.c()[0].pow(&e)]);
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

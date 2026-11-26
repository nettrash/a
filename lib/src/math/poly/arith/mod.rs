//! The fast arithmetic of polynomials: products by the schoolbook method, Karatsuba's method,
//! Kronecker substitution and number-theoretic transforms, chosen by the lengths and the sizes
//! of the coefficients; power series inverses and division by Newton's method.

pub(crate) mod fdiv;
pub(crate) mod field;
pub(crate) mod gf2;
pub(crate) mod lred;
pub(crate) mod mdiv;
pub(crate) mod mmul;
pub(crate) mod nmod;
pub(crate) mod pack;
pub(crate) mod wmul;
pub(crate) mod wrem;
pub(crate) mod zdiv;
pub(crate) mod zmul;

use super::Polynomial;
use super::coeff::Field;
use super::modular::ModPolynomial;

/// Whether the pseudo-division of `a` by `d` over the integers goes by Kronecker substitution
/// (see `Polynomial::pseudo_div_rem`): measured on an Apple M4 against the schoolbook method on
/// limbs ([`zdiv`]), whose intermediate coefficients grow by a factor `l` a step where the
/// field must hold the largest, from 384 coefficients in the quotient, or 32 for divisors of
/// 1,000 bits and more.
pub(crate) fn pseudo_by_ks(
    a: &Polynomial<crate::math::numbers::Number>,
    d: &Polynomial<crate::math::numbers::Number>,
) -> bool {
    let (m, n) = (a.length(), d.length());
    if m < n || n < 2 {
        return false;
    }
    let lq = m - n + 1;
    if cfg!(miri) {
        return lq >= 3;
    }
    lq >= 384 || (lq >= 32 && zmul::max_bits(d.coeffs()) >= 1000)
}

/// Before Kronecker substitution, the pseudo-division by the schoolbook method with the fewest
/// factors of `l` is tried, and given up once this many of its steps need one (the general case,
/// for which the substitution is faster): exact and nearly exact divisions need none or one.
pub(crate) const LAZY_FACTORS: usize = 4;

/// Whether an exact division over the integers of `a` by `d` goes by Kronecker substitution:
/// measured against the schoolbook method on limbs, from 12 coefficients in the quotient, or 48
/// for divisors of 512 bits and more, whose products of coefficients the schoolbook method
/// takes at the speed of the integers' multiplication.
pub(crate) fn exact_by_ks(
    a: &Polynomial<crate::math::numbers::Number>,
    d: &Polynomial<crate::math::numbers::Number>,
) -> bool {
    let (m, n) = (a.length(), d.length());
    if m < n || n < 2 {
        return false;
    }
    let lq = m - n + 1;
    if cfg!(miri) {
        return lq >= 3;
    }
    lq >= if zmul::max_bits(d.coeffs()) >= 512 {
        48
    } else {
        12
    }
}

/// Up to this many coefficients, `Polynomial<Number>::eval` is Horner's rule on numbers; beyond,
/// the evaluation on limbs of the eval area (one point of `eval_many`): measured on an Apple M4,
/// 1.5x slower at 8 coefficients, equal at 32, 0.75x at 64 and 0.01x at 2,500 by a 12,000-bit
/// point.
pub(crate) const EVAL_HORNER: usize = if cfg!(miri) { 4 } else { 31 };

/// Whether the schoolbook pseudo-division or exact division of `m` by `n` coefficients runs on
/// limbs ([`zdiv`]): beyond the smallest, whose set-up costs more than it saves on the numbers.
pub(crate) fn limb_school(m: usize, n: usize) -> bool {
    m >= n && (m - n + 1) * n.saturating_sub(1) > if cfg!(miri) { 2 } else { 6 }
}

/// The division over a field by Newton's method for long quotients and divisors.
pub(crate) fn field_div_rem<T: Field>(
    a: &Polynomial<T>,
    d: &Polynomial<T>,
    inv: &T,
) -> Option<(Polynomial<T>, Polynomial<T>)> {
    let (la, ld) = (a.length(), d.length());
    if la < ld {
        return None;
    }
    let lq = la - ld + 1;
    (lq >= fdiv::NEWTON_MIN && ld >= fdiv::NEWTON_MIN).then(|| a.div_rem_newton(d, inv))
}

/// The product of two polynomials of the same modulus, by the method their lengths and the
/// modulus call for; a square when they are the same polynomial.
pub(crate) fn mod_mul(a: &ModPolynomial, b: &ModPolynomial) -> ModPolynomial {
    if let (Some(x), Some(y), Some(w)) = (a.c.plain(), b.c.plain(), a.m.word_mod())
        && w.n != 2
    {
        // A word modulus but 2: on the vectors.
        let (la, lb) = (x.len(), y.len());
        if la == 0 || lb == 0 {
            return a.zero_like();
        }
        let c = if la == 1 || lb == 1 {
            // By a constant: no method to choose.
            let (p, c) = if lb == 1 { (x, y[0]) } else { (y, x[0]) };
            p.iter().map(|&v| w.mul(v, c)).collect()
        } else if core::ptr::eq(a, b) || x == y {
            wmul::sqr(x, w)
        } else {
            wmul::mul(x, y, w)
        };
        return ModPolynomial::of(a.m.clone(), c);
    }
    let (la, lb) = (a.length(), b.length());
    if la == 0 || lb == 0 {
        return a.zero_like();
    }
    if a.m.is_two() {
        let square = core::ptr::eq(a, b) || (la == lb && a.c == b.c);
        let (s, l) = (la.min(lb).div_ceil(64), la.max(lb).div_ceil(64));
        if !square && !a.c.is_given() && !b.c.is_given() && gf2::by_words(s, l) {
            // Long operands of words whose product goes by the transforms on words: on the words
            // themselves, the product left as words, without the packing and unpacking of the
            // packed product's own switch to the transforms (which operands made packed take).
            let w = a.m.word_mod().expect("the word modulus 2");
            let mut r = vec![0u64; la + lb - 1];
            wmul::mul_into_words(&mut r, &a.c, &b.c, w);
            return ModPolynomial::of(a.m.clone(), r);
        }
        // Packed in and out: the operands' packed forms made once, the product left packed.
        let (x, y) = (a.c.bits(), b.c.bits());
        let p = if core::ptr::eq(a, b) || x == y {
            gf2::sqr(&x)
        } else {
            gf2::mul(&x, &y)
        };
        return ModPolynomial::of_bits(a.m.clone(), p);
    }
    if let (Some(w), true) = (a.m.word_mod(), la == 1 || lb == 1) {
        // By a constant: no method to choose.
        let (p, c) = if lb == 1 {
            (&a.c, b.c[0])
        } else {
            (&b.c, a.c[0])
        };
        let r: Vec<u64> = p.iter().map(|&x| w.mul(x, c)).collect();
        return ModPolynomial::of(a.m.clone(), r);
    }
    let square = core::ptr::eq(a, b) || (la == lb && a.c == b.c);
    if let Some(w) = a.m.word_mod() {
        let c = if square {
            wmul::sqr(&a.c, w)
        } else {
            wmul::mul(&a.c, &b.c, w)
        };
        return ModPolynomial::of(a.m.clone(), c);
    }
    let red = a.m.l.as_ref().expect("several limbs");
    let c = mmul::mul(&a.c, if square { &[] } else { &b.c }, &a.m.d, red);
    ModPolynomial::of(a.m.clone(), c)
}

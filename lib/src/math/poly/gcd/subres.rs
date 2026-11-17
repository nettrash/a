//! The subresultant polynomial remainder sequence over the integers, with Lazard's and Ducos's
//! optimisation of the defective steps.

use crate::math::numbers::Number;
use crate::math::poly::Polynomial;

/// `p / c`, exactly.
fn div_scalar(p: &Polynomial<Number>, c: &Number) -> Polynomial<Number> {
    if crate::math::rational::is_one(c) {
        return p.clone();
    }
    p.map(|x| x.div_exact(c))
}

/// `xⁿ / yⁿ⁻¹` for `n ≥ 1`, exact, by Lazard's binary method: every intermediate power divided
/// as soon as it is formed, so that none grows past the result.
fn lazard(x: &Number, y: &Number, n: usize) -> Number {
    let mut a = 1usize << (usize::BITS - 1 - n.leading_zeros());
    let mut c = x.clone();
    let mut n = n - a;
    while a > 1 {
        a /= 2;
        c = (&c * &c).div_exact(y);
        if n >= a {
            c = (&c * x).div_exact(y);
            n -= a;
        }
    }
    c
}

/// The non-zero subresultants of `p` and `q`, `deg p ≥ deg q ≥ 1`, the lowest index first (as
/// Sage's `subresultants`): each step's pseudo-remainder divided by the known factor
/// `s^δ·lc(a)`, and the subresultant after a defective step, `lc(b)^(δ-1)·b / s^(δ-1)`, by
/// Lazard's powering.
pub(crate) fn subresultants(
    p: &Polynomial<Number>,
    q: &Polynomial<Number>,
) -> Vec<Polynomial<Number>> {
    let mut out = Vec::new();
    chain(p, q, Some(&mut out));
    out.reverse();
    out
}

/// The resultant of `p` and `q`, `deg p ≥ deg q ≥ 1`, as the last subresultant of the chain:
/// for short polynomials, where the multimodular algorithm's primes cost more.
pub(crate) fn resultant(p: &Polynomial<Number>, q: &Polynomial<Number>) -> Number {
    let last = chain(p, q, None);
    match last.degree() {
        Some(0) => last.coeffs()[0].clone(),
        _ => Number::default(),
    }
}

/// The chain of [`subresultants`], highest index first into `out`; the last subresultant formed
/// (the zero polynomial when the chain ends at a zero remainder).
fn chain(
    p: &Polynomial<Number>,
    q: &Polynomial<Number>,
    mut out: Option<&mut Vec<Polynomial<Number>>>,
) -> Polynomial<Number> {
    let dp = p.degree().expect("not zero");
    let dq = q.degree().expect("not zero");
    let mut s = q.leading().expect("not zero").pow((dp - dq) as u32);
    let mut a = q.clone();
    let mut b = p.pseudo_div_rem(&-q).1;
    loop {
        if b.is_zero() {
            return b;
        }
        let d = a.degree().expect("not zero");
        let e = b.degree().expect("not zero");
        if let Some(out) = out.as_deref_mut() {
            out.push(b.clone());
        }
        let delta = d - e;
        let c = if delta > 1 {
            let x = b.leading().expect("not zero");
            let c = lazard(x, &s, delta - 1);
            let c = div_scalar(&b.scale(&c), &s);
            if let Some(out) = out.as_deref_mut() {
                out.push(c.clone());
            }
            c
        } else {
            b.clone()
        };
        if e == 0 {
            return c;
        }
        let r = a.pseudo_div_rem(&-&b).1;
        let f = &s.pow(delta as u32) * a.leading().expect("not zero");
        b = div_scalar(&r, &f);
        s = c.leading().expect("not zero").clone();
        a = c;
    }
}

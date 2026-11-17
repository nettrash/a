//! Products of polynomials over the rationals, the floats and the complex floats.
//!
//! Over the rationals, each polynomial is brought to a common denominator, `a = A / da` with `A`
//! over the integers, the integer polynomials are multiplied by Kronecker substitution, and each
//! coefficient of `A B / (da db)` is brought to lowest terms: one product of integers and a
//! greatest common divisor a coefficient, instead of a product, a sum and their greatest common
//! divisors for each pair of coefficients.
//!
//! Over the floats, the schoolbook method computes each coefficient as a sum of rounded products,
//! with the componentwise bound `|ĉ_k - c_k| <= γ_m Σ |a_i| |b_(k-i)|` (`γ_m = m ε / (1 - m ε)`,
//! `ε = 2^-53`, `m` the terms); from about 512 coefficients in the shorter operand (80 for
//! complex floats; see [`by_fft`]) the product
//! is taken by a fast Fourier transform of length `N` (a power of 2 at least the product's
//! length), whose error is bounded in norm rather than componentwise: `|ĉ_k - c_k| <= 11 log2(N)
//! ε ‖a‖₂ ‖b‖₂` (Percival, 2003, with twiddles accurate to `ε`), so that coefficients much
//! smaller than the norms may lose their relative accuracy.

use super::zmul;
use crate::math::complex::Complex;
use crate::math::numbers::Number;
use crate::math::poly::coeff::sealed::schoolbook;
use crate::math::rational::Rational;

// The rationals

/// The common denominator of `a` (the least common multiple of the denominators) and the
/// numerators over it.
fn common(a: &[Rational]) -> (Number, Vec<Number>) {
    let mut den = Number::from(1u8);
    for c in a {
        let d = c.denom();
        if !crate::math::rational::is_one(d) {
            let g = den.gcd(d);
            den = &den / &g * d;
        }
    }
    let unit = crate::math::rational::is_one(&den);
    let ints = a
        .iter()
        .map(|c| {
            if unit {
                c.numer().clone()
            } else if crate::math::rational::is_one(c.denom()) {
                c.numer() * &den
            } else {
                c.numer() * &(&den / c.denom())
            }
        })
        .collect();
    (den, ints)
}

/// The rationals `c_i / d`, each in lowest terms.
fn over(c: Vec<Number>, d: &Number) -> Vec<Rational> {
    if crate::math::rational::is_one(d) {
        return c.into_iter().map(Rational::from).collect();
    }
    c.into_iter().map(|x| Rational::new(x, d.clone())).collect()
}

/// Up to this many coefficients in the shorter operand, products over the rationals by the
/// schoolbook method: measured on an Apple M4, its products of rationals in words beat the
/// common denominator, whose lcm grows with the length, up to about 16 coefficients.
const RAT_SCHOOL: usize = if cfg!(miri) { 2 } else { 16 };

/// The product of polynomials over the rationals, both not empty.
pub(crate) fn rat_mul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    if a.len().min(b.len()) <= RAT_SCHOOL {
        return schoolbook(a, b);
    }
    let (da, x) = common(a);
    let (db, y) = common(b);
    over(zmul::mul(&x, &y), &(&da * &db))
}

/// The square of a polynomial over the rationals, not empty.
pub(crate) fn rat_sqr(a: &[Rational]) -> Vec<Rational> {
    if a.len() <= RAT_SCHOOL {
        return schoolbook(a, a);
    }
    let (da, x) = common(a);
    over(zmul::sqr(&x), &(&da * &da))
}

/// The quotient and the remainder over the rationals through the integers, for quotients of at
/// least this many coefficients and divisors of at least two: `a = A / da` and `d = D / dd` over
/// common denominators, the pseudo-division `l^k A = Q D + R` (`l` the leading coefficient of
/// `D`, `k` the quotient's length) by Kronecker substitution, and then `q = Q dd / (da l^k)` and
/// `r = R / (da l^k)`, each coefficient brought to lowest terms once, instead of a product, a
/// difference and their greatest common divisors for every pair of coefficients.
pub(crate) const RAT_DIV_MIN: usize = if cfg!(miri) { 3 } else { 16 };

/// The division of [`RAT_DIV_MIN`].
pub(crate) fn rat_div_rem(
    a: &crate::math::poly::Polynomial<Rational>,
    d: &crate::math::poly::Polynomial<Rational>,
) -> Option<(
    crate::math::poly::Polynomial<Rational>,
    crate::math::poly::Polynomial<Rational>,
)> {
    use crate::math::poly::Polynomial;
    let (la, ld) = (a.length(), d.length());
    if la < ld || ld < 2 || la - ld + 1 < RAT_DIV_MIN {
        return None;
    }
    let (da, x) = common(a.coeffs());
    let (dd, y) = common(d.coeffs());
    let (xp, yp) = (Polynomial::new(x), Polynomial::new(y));
    let k = (la - ld + 1) as u32;
    let lk = yp.leading().expect("not zero").pow(k);
    let (q, r) = xp.pseudo_div_rem(&yp);
    let den = &da * &lk;
    let qden = den.clone();
    let qc: Vec<Rational> = q
        .into_coeffs()
        .into_iter()
        .map(|c| {
            let c = if crate::math::rational::is_one(&dd) {
                c
            } else {
                &c * &dd
            };
            normalized(c, &qden)
        })
        .collect();
    let rc: Vec<Rational> = r
        .into_coeffs()
        .into_iter()
        .map(|c| normalized(c, &den))
        .collect();
    Some((Polynomial::new(qc), Polynomial::new(rc)))
}

/// `c / d` in lowest terms for a non-zero `d` of either sign.
fn normalized(c: Number, d: &Number) -> Rational {
    Rational::new(c, d.clone())
}

/// From this many coefficients, the inverse of a power series over the rationals may go through
/// the integers ([`rat_inv_series`]); below, the recurrence on rationals.
pub(crate) const RAT_INV_MIN: usize = if cfg!(miri) { 3 } else { 8 };

/// The most bits of the common denominator of a series whose inverse goes through the integers
/// by the substitution of [`rat_inv_series`]: integer series, and those over one short
/// denominator; beyond, [`rat_inv_common`] from [`RAT_INV_COMMON_MIN`] coefficients.
const RAT_INV_DEN_BITS: u64 = 64;

/// From this many coefficients, the inverse of a series over a long common denominator goes by
/// Newton's iteration over one common denominator ([`rat_inv_common`]): measured on an Apple M4,
/// 0.45x the time of the iteration on rationals at 32 coefficients (31-bit parts), 0.08x at 128,
/// 0.13x on a 17-term series at 513; 1.5x to 3x slower below 32.
const RAT_INV_COMMON_MIN: usize = if cfg!(miri) { 6 } else { 32 };

/// `b^e mod f` over the rationals through the integers: `f = F/df`, `b = B/db` over common
/// denominators (a remainder modulo `f` is one modulo `F`), and every value kept as `N / (|l|^E
/// db^P)` with the integer `N`: a product is one of the integers, its remainder modulo `F` the
/// pseudo-remainder with the fewest factors of `l = lc(F)` (`|l|^D x ≡ r`, `E` growing by `D`),
/// and only the `deg f` coefficients of the result are brought to lowest terms, once. `None` for
/// a constant `f` or a zero `b`, left to the generic powers.
pub(crate) fn rat_pow_mod(b: &[Rational], e: &Number, f: &[Rational]) -> Option<Vec<Rational>> {
    if f.len() < 2 || b.is_empty() {
        return None;
    }
    let (_, fz) = common(f);
    let (db, bz) = common(b);
    let top = |x: &[Number]| x.iter().rposition(|c| !c.is_zero()).map_or(0, |i| i + 1);
    let reduce = |x: Vec<Number>| -> (Vec<Number>, usize) {
        let t = top(&x);
        if t < fz.len() {
            let mut x = x;
            x.truncate(t);
            return (x, 0);
        }
        let (mut r, dd) = super::zdiv::rem_fewest(&x[..t], &fz);
        r.truncate(top(&r));
        (r, dd)
    };
    // (N, E, P): the value N / (|l|^E db^P).
    let mul = |x: &(Vec<Number>, usize, usize), y: &(Vec<Number>, usize, usize)| {
        if x.0.is_empty() || y.0.is_empty() {
            return (Vec::new(), 0, 0);
        }
        let p = if core::ptr::eq(x, y) {
            zmul::sqr(&x.0)
        } else {
            zmul::mul(&x.0, &y.0)
        };
        let (r, dd) = reduce(p);
        (r, x.1 + y.1 + dd, x.2 + y.2)
    };
    let (b0, d0) = reduce(bz);
    let base = (b0, d0, 1usize);
    let mut r = (vec![Number::from(1u8)], 0usize, 0usize);
    for i in (0..e.bits()).rev() {
        r = mul(&r, &r);
        if e.bit(i) {
            r = mul(&r, &base);
        }
    }
    let (n, ee, pp) = r;
    if n.is_empty() {
        return Some(Vec::new());
    }
    let l = fz.last().expect("not empty").abs();
    let mut den = l.pow(ee as u32);
    if pp > 0 && !crate::math::rational::is_one(&db) {
        den = &den * &db.pow(pp as u32);
    }
    Some(
        n.into_iter()
            .map(|x| Rational::new(x, den.clone()))
            .collect(),
    )
}

/// The inverse modulo `x^n` of the power series of the integers `u`, `u[0] = 1`: `n` coefficients,
/// the recurrence `g_j = -Σ u_i g_(j-i)` for short series, Newton's iteration
/// `g ← g - x^h·((u g - 1) / x^h · g mod x^(n-h))` beyond, exact over the integers as the constant
/// is 1.
fn int_inv_series(u: &[Number], n: usize) -> Vec<Number> {
    if n <= 16 {
        let mut g: Vec<Number> = Vec::with_capacity(n);
        g.push(Number::from(1u8));
        for j in 1..n {
            let mut acc = Number::default();
            for i in 1..=j.min(u.len() - 1) {
                if !u[i].is_zero() && !g[j - i].is_zero() {
                    acc = &acc + &(&u[i] * &g[j - i]);
                }
            }
            g.push(-acc);
        }
        return g;
    }
    let h = n.div_ceil(2);
    let mut g = int_inv_series(u, h);
    // The products take operands without zeros at the top (their signs are read there).
    let top = |x: &[Number]| x.iter().rposition(|c| !c.is_zero()).map_or(0, |i| i + 1);
    let ul = &u[..top(&u[..u.len().min(n)])];
    let gl = &g[..top(&g)];
    let mut e = zmul::mul(ul, gl);
    e.resize(n.max(e.len()), Number::default());
    // e = 1 + x^h t mod x^n.
    let t: Vec<Number> = e[h..n].to_vec();
    let tn = top(&t);
    if tn > 0 {
        let w = zmul::mul(&t[..tn], &g[..top(&g)]);
        g.extend(w.into_iter().take(n - h).map(|c| -c));
    }
    g.resize(n, Number::default());
    g
}

/// `num / den` in lowest terms for `den` a divisor of a power of `c` (both not zero), without a
/// greatest common divisor of the long numbers: every prime both share divides `c`, so the common
/// part is a product of powers of `t = gcd(num, gcd(c, den))`, a number the size of `c` (each
/// operand reduced modulo `c` first); the largest power of `t` dividing both is found by
/// doubling its exponent and taken out, and `t` computed again until it is 1.
fn lowest_over_power(mut num: Number, mut den: Number, c: &Number) -> Rational {
    if den.is_negative() {
        num = -num;
        den = -den;
    }
    let c = c.abs();
    let divides = |x: &Number, p: &Number| x.rem_euclid(p).is_zero();
    loop {
        if num.is_zero() {
            return Rational::default();
        }
        let cd = c.gcd(&den.rem_euclid(&c));
        if crate::math::rational::is_one(&cd) {
            break;
        }
        let t = cd.gcd(&num.rem_euclid(&cd));
        if crate::math::rational::is_one(&t) {
            break;
        }
        // t, t^2, t^4, ... while they divide both, then the largest power by descent.
        let mut pws = vec![t];
        loop {
            let last = pws.last().expect("not empty");
            let next = last * last;
            if next.bits() > num.bits().min(den.bits())
                || !divides(&num, &next)
                || !divides(&den, &next)
            {
                break;
            }
            pws.push(next);
        }
        for p in pws.iter().rev() {
            if divides(&num, p) && divides(&den, p) {
                num = &num / p;
                den = &den / p;
            }
        }
    }
    let order = crate::math::numbers::ByteOrder::LittleEndian;
    let mut num = crate::math::rational::in_order(num, order);
    let mut den = crate::math::rational::in_order(den, order);
    crate::math::rational::trim(&mut num);
    crate::math::rational::trim(&mut den);
    Rational::from_reduced(num, den)
}

/// The content of `c` with `d`: the greatest common divisor of `d` and every coefficient, not
/// negative, stopping at 1.
fn content_with(c: &[Number], d: &Number) -> Number {
    let mut g = d.abs();
    for x in c {
        if crate::math::rational::is_one(&g) {
            break;
        }
        if !x.is_zero() {
            g = g.gcd(x);
        }
    }
    g
}

/// [`rat_inv_series`] over one common denominator, as FLINT's `fmpq_poly_inv_series`: `g = N/d`
/// with the integer `N`, Newton's iteration `g ← g (2 - a g)` on integer products,
/// `N' = N (2 den d - A N) mod x^m` over `den d²`, the common content taken out at each step,
/// and each coefficient brought to lowest terms at the end.
fn rat_inv_common(a: &[Rational], den: &Number, ints: &[Number], n: usize) -> Vec<Rational> {
    let top = |x: &[Number]| x.iter().rposition(|c| !c.is_zero()).map_or(0, |i| i + 1);
    // The start: the recurrence on rationals for a few coefficients, over their common denominator.
    let m0 = n.min(16);
    let p = super::super::Polynomial::new(a[..a.len().min(m0)].to_vec());
    let g0 = super::super::Polynomial::<Rational>::inv_series_school(&p, m0);
    let (mut d, mut nn) = common(g0.coeffs());
    nn.resize(m0, Number::default());
    let mut m = m0;
    while m < n {
        let m2 = (2 * m).min(n);
        let al = &ints[..top(&ints[..ints.len().min(m2)])];
        let mut pq = zmul::mul(al, &nn[..top(&nn).max(1)]);
        pq.truncate(m2);
        pq.resize(m2, Number::default());
        // q = 2 den d - A N.
        for x in pq.iter_mut() {
            *x = -core::mem::take(x);
        }
        pq[0] = &pq[0] + &(&(den * &d) * &Number::from(2u8));
        let qt = top(&pq);
        let mut nq = if qt == 0 {
            vec![Number::default(); m2]
        } else {
            zmul::mul(&nn[..top(&nn).max(1)], &pq[..qt])
        };
        nq.truncate(m2);
        nq.resize(m2, Number::default());
        let mut d2 = &(den * &d) * &d;
        let g = content_with(&nq, &d2);
        if !crate::math::rational::is_one(&g) {
            for x in nq.iter_mut() {
                if !x.is_zero() {
                    *x = &*x / &g;
                }
            }
            d2 = &d2 / &g;
        }
        if d2.is_negative() {
            d2 = -d2;
            for x in nq.iter_mut() {
                *x = -core::mem::take(x);
            }
        }
        nn = nq;
        d = d2;
        m = m2;
    }
    nn.into_iter()
        .map(|x| Rational::new(x, d.clone()))
        .collect()
}

/// The inverse of the power series `a` over the rationals modulo `x^n` (the constant not zero),
/// through the integers, as FLINT's `fmpq_poly_inv_series`: `a = A / den` over a common
/// denominator, `A(c y) = c·U(y)` for `c = A_0` with the integer `U = 1 + Σ A_j c^(j-1) y^j`,
/// whose inverse `V` is exact over the integers by Newton's iteration on integer products, and
/// then `1/a = den / A` has the coefficients `den·v_j / c^(j+1)`, each brought to lowest terms by
/// common factors of `c` alone.
///
/// For a common denominator of more than [`RAT_INV_DEN_BITS`] bits the denominators `c^(j+1)`
/// would overshoot the true ones by far (these grow by about `bits(c)/len` a coefficient), and
/// the inverse goes by [`rat_inv_common`] instead. `None` (the caller's Newton iteration on
/// rationals) below [`RAT_INV_MIN`] coefficients, or below [`RAT_INV_COMMON_MIN`] for a long
/// common denominator.
pub(crate) fn rat_inv_series(a: &[Rational], n: usize) -> Option<Vec<Rational>> {
    if n < RAT_INV_MIN {
        return None;
    }
    let len = a.len().min(n);
    let (den, ints) = common(&a[..len]);
    if den.bits() > RAT_INV_DEN_BITS {
        return (n >= RAT_INV_COMMON_MIN).then(|| rat_inv_common(a, &den, &ints, n));
    }
    let c = ints[0].clone();
    let mut u: Vec<Number> = Vec::with_capacity(len);
    u.push(Number::from(1u8));
    let mut cp = Number::from(1u8);
    for (j, x) in ints.iter().enumerate().skip(1) {
        if j > 1 {
            cp = &cp * &c;
        }
        u.push(x * &cp);
    }
    let v = int_inv_series(&u, n);
    let mut out = Vec::with_capacity(n);
    let mut cpow = c.clone();
    for vj in v {
        let num = if crate::math::rational::is_one(&den) {
            vj
        } else {
            &den * &vj
        };
        out.push(lowest_over_power(num, cpow.clone(), &c));
        cpow = &cpow * &c;
    }
    Some(out)
}

// The fast Fourier transform

/// Whether a product of `la` by `lb` coefficients goes by the transform: measured on an Apple M4,
/// the schoolbook loop (which the compiler vectorizes) costs about 0.1 ns a product of floats and
/// 0.45 of complex floats, and the transform about 1.9 ns a point and level of a length `N`, the
/// power of 2 at least the product's length; so the transform from `la lb > c N log2(N)`, `c =
/// 19` for floats and 5 for complex floats (from about 512 and 80 coefficients each).
fn by_fft(la: usize, lb: usize, complex: bool) -> bool {
    if cfg!(miri) {
        return la.min(lb) >= 16;
    }
    let n = fft_len(la + lb - 1);
    let c = if complex { 5 } else { 19 };
    la.min(lb) >= 32 && la * lb > c * n * n.trailing_zeros() as usize
}

/// The twiddles `e^(-2πi j / n)` for `j < n / 2`, each from its own sine and cosine.
fn twiddles(n: usize) -> (Vec<f64>, Vec<f64>) {
    let h = n / 2;
    let mut re = Vec::with_capacity(h);
    let mut im = Vec::with_capacity(h);
    for j in 0..h {
        // The angle reduced to the first octant for the accuracy of sin and cos.
        let t = -2.0 * core::f64::consts::PI * (j as f64) / (n as f64);
        re.push(t.cos());
        im.push(t.sin());
    }
    (re, im)
}

/// The transform of `(re, im)` in place, of a power-of-2 length, forward (`e^(-2πi/n)`) or the
/// inverse without the factor `1/n`: iterative radix 2, decimation in frequency forward (the
/// result in bit-reversed order) and in time inverse (from bit-reversed order), so that no
/// permutation is needed between them.
fn transform(re: &mut [f64], im: &mut [f64], tw: &(Vec<f64>, Vec<f64>), inverse: bool) {
    let n = re.len();
    let (wr, wi) = (&tw.0, &tw.1);
    let sign = if inverse { -1.0 } else { 1.0 };
    if !inverse {
        // Gentleman–Sande: butterflies (a, b) -> (a + b, (a - b) w).
        let mut len = n;
        while len >= 2 {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for j in 0..half {
                    let (c, s) = (wr[j * step], sign * wi[j * step]);
                    let (p, q) = (start + j, start + j + half);
                    let (ar, ai, br, bi) = (re[p], im[p], re[q], im[q]);
                    re[p] = ar + br;
                    im[p] = ai + bi;
                    let (dr, di) = (ar - br, ai - bi);
                    re[q] = dr * c - di * s;
                    im[q] = dr * s + di * c;
                }
            }
            len = half;
        }
    } else {
        // Cooley–Tukey: (a, b) -> (a + b w, a - b w), w conjugated.
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for j in 0..half {
                    let (c, s) = (wr[j * step], sign * wi[j * step]);
                    let (p, q) = (start + j, start + j + half);
                    let (br, bi) = (re[q] * c - im[q] * s, re[q] * s + im[q] * c);
                    let (ar, ai) = (re[p], im[p]);
                    re[p] = ar + br;
                    im[p] = ai + bi;
                    re[q] = ar - br;
                    im[q] = ai - bi;
                }
            }
            len *= 2;
        }
    }
}

/// The length of the transform of a product of `len` coefficients.
fn fft_len(len: usize) -> usize {
    len.next_power_of_two().max(2)
}

/// The product of the complex polynomials `(ar, ai)` and `(br, bi)` (a square when `b` is
/// `None`) by the transform: `len` coefficients.
fn fft_product(
    a: (&[f64], &[f64]),
    b: Option<(&[f64], &[f64])>,
    len: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = fft_len(len);
    let tw = twiddles(n);
    let load = |x: (&[f64], &[f64])| {
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        re[..x.0.len()].copy_from_slice(x.0);
        im[..x.1.len()].copy_from_slice(x.1);
        (re, im)
    };
    let (mut xr, mut xi) = load(a);
    transform(&mut xr, &mut xi, &tw, false);
    match b {
        None => {
            for (r, i) in xr.iter_mut().zip(xi.iter_mut()) {
                let (u, v) = (*r, *i);
                *r = u * u - v * v;
                *i = 2.0 * u * v;
            }
        }
        Some(b) => {
            let (mut yr, mut yi) = load(b);
            transform(&mut yr, &mut yi, &tw, false);
            for ((r, i), (s, t)) in xr.iter_mut().zip(xi.iter_mut()).zip(yr.iter().zip(&yi)) {
                let (u, v) = (*r, *i);
                *r = u * s - v * t;
                *i = u * t + v * s;
            }
        }
    }
    transform(&mut xr, &mut xi, &tw, true);
    let scale = 1.0 / n as f64;
    xr.truncate(len);
    xi.truncate(len);
    for (r, i) in xr.iter_mut().zip(xi.iter_mut()) {
        *r *= scale;
        *i *= scale;
    }
    (xr, xi)
}

/// The product of polynomials over the floats, both not empty.
pub(crate) fn f64_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if !by_fft(a.len(), b.len(), false) {
        return schoolbook(a, b);
    }
    // Both real: a + i b transformed at once would mix their errors; one complex product of a
    // and b as they are keeps the bound of the norms.
    let zero_a = vec![0.0; a.len()];
    let zero_b = vec![0.0; b.len()];
    fft_product((a, &zero_a), Some((b, &zero_b)), a.len() + b.len() - 1).0
}

/// The square of a polynomial over the floats, not empty.
pub(crate) fn f64_sqr(a: &[f64]) -> Vec<f64> {
    if !by_fft(a.len(), a.len(), false) {
        return schoolbook(a, a);
    }
    let zero = vec![0.0; a.len()];
    fft_product((a, &zero), None, 2 * a.len() - 1).0
}

/// The parts of complex coefficients.
fn parts(a: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>) {
    (
        a.iter().map(|c| c.re).collect(),
        a.iter().map(|c| c.im).collect(),
    )
}

/// The product of polynomials over the complex floats, both not empty.
pub(crate) fn complex_mul(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    if !by_fft(a.len(), b.len(), true) {
        return schoolbook(a, b);
    }
    let (ar, ai) = parts(a);
    let (br, bi) = parts(b);
    let (r, i) = fft_product((&ar, &ai), Some((&br, &bi)), a.len() + b.len() - 1);
    r.into_iter()
        .zip(i)
        .map(|(re, im)| Complex::new(re, im))
        .collect()
}

/// The square of a polynomial over the complex floats, not empty.
pub(crate) fn complex_sqr(a: &[Complex<f64>]) -> Vec<Complex<f64>> {
    if !by_fft(a.len(), a.len(), true) {
        return schoolbook(a, a);
    }
    let (ar, ai) = parts(a);
    let (r, i) = fft_product((&ar, &ai), None, 2 * a.len() - 1);
    r.into_iter()
        .zip(i)
        .map(|(re, im)| Complex::new(re, im))
        .collect()
}

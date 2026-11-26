//! Products of polynomials over the rationals, the floats and the complex floats.
//!
//! Over the rationals, each polynomial is brought to a common denominator, `a = A / da` with `A`
//! over the integers, the integer polynomials are multiplied by Kronecker substitution, and each
//! coefficient of `A B / (da db)` is brought to lowest terms: one product of integers and a
//! greatest common divisor a coefficient, instead of a product, a sum and their greatest common
//! divisors for each pair of coefficients.
//!
//! Over the floats and the complex floats, the schoolbook method computes each coefficient as a
//! sum of rounded products, with the componentwise bound `|ĉ_k - c_k| <= γ_m Σ |a_i| |b_(k-i)|`
//! (`γ_m = m ε / (1 - m ε)`, `ε = 2^-53`, `m` the terms), for every pair of lengths [`by_fft`]
//! leaves to it: below 32 coefficients in the shorter operand, and while `la·lb ≤ c·N·log2(N)`
//! (`c = 19` for floats, 5 for complex floats, `N` the least power of 2 at least the product's
//! length), which for square products is up to 441 coefficients (101 complex). Beyond, the
//! crate's linear convolution by fast Fourier transforms
//! ([`fft::convolve`](crate::math::fft::convolve)), whose error is bounded in norm rather than
//! componentwise, a few units of rounding times `‖a‖₂ ‖b‖₂ log2(N)` for a transform of length
//! `N`, so that coefficients much smaller than the norms may lose their relative accuracy, and an
//! infinity or a NaN among the coefficients may make every coefficient NaN.

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

/// The composition `p(g)` over the rationals through the integers, as FLINT's
/// `fmpq_poly_compose`: `p = P / dp` and `g = G / dg` over common denominators, so that
/// `p(g) = Q(G) / (dp·dg^(n-1))` with the integer `Q_i = P_i·dg^(n-1-i)`, `Q(G)` composed over the
/// integers and each coefficient brought to lowest terms once. `None` for a constant `p` or `g`.
pub(crate) fn rat_compose(p: &[Rational], g: &[Rational]) -> Option<Vec<Rational>> {
    let n = p.len();
    if n < 2 || g.len() < 2 {
        return None;
    }
    let (dp, pn) = common(p);
    let (dg, gn) = common(g);
    let mut q = vec![Number::default(); n];
    let mut pw = Number::from(1u8);
    for i in (0..n).rev() {
        q[i] = &pn[i] * &pw;
        if i > 0 {
            pw = &pw * &dg;
        }
    }
    // pw = dg^(n-1)
    let h = crate::math::poly::Polynomial::new(q).compose(&crate::math::poly::Polynomial::new(gn));
    Some(over(h.into_coeffs(), &(&dp * &pw)))
}

/// Whether a product of `la` by `lb` coefficients goes by the transforms, rather than by the
/// schoolbook method: from 32 coefficients in the shorter operand when `la·lb > c·N·log2(N)`,
/// `c = 19` for floats and 5 for complex floats, `N` the least power of 2 at least `la + lb - 1`
/// (square products of 442 to 512 coefficients and of 655 or more, complex ones of 102 to 128
/// and of 152 or more; against 5,000 coefficients from 405 in the shorter, 107 complex). This is
/// the line the page and the rustdoc of `*` draw for the componentwise error bound of the
/// schoolbook method, not a tuned
/// crossover: it was the crossover of the radix-2 transform this module had before, and a faster
/// transform is no reason to give the bound up for shorter products.
fn by_fft(la: usize, lb: usize, complex: bool) -> bool {
    let n = (la + lb - 1).next_power_of_two().max(2);
    let c = if complex { 5 } else { 19 };
    la.min(lb) >= 32 && la * lb > c * n * n.trailing_zeros() as usize
}

/// The coefficients of a product the schoolbook kernels below compute together, in registers.
/// Measured on an Apple M4 Max (8, 16 and 32 lanes, `la` 4 to 384, `lb/la` 1 to 64): 32 lanes
/// (16 NEON registers of sums) 4 to 9% faster than 16 wherever the kernel runs, 8 slower than
/// both.
const LANES: usize = 32;

/// The same for complex floats (twice as many sums, the real and the imaginary parts): measured,
/// 8 faster than 4 and 16 wherever the kernel runs (sweep 4, 8, 16; `la` 2 to 64, `lb` 8 to
/// 512).
const CLANES: usize = 8;

/// The terms `i` of the block of `w` coefficients `k0..k0 + w` of a product of `la` by `lb ≥ w`
/// coefficients, ascending, in three ranges `(lo..p, p..q, q..hi)`: in the middle one every
/// coefficient of the block has the term `i` (`0 ≤ k − i < lb`), in the others some do not.
#[inline(always)]
fn block_ranges(k0: usize, w: usize, la: usize, lb: usize) -> [(usize, usize); 3] {
    // The block's terms: k0 + 1 - lb ≤ i ≤ k0 + w - 1, i < la; every coefficient has the term i
    // when k0 + w - lb ≤ i ≤ k0.
    let lo = (k0 + 1).saturating_sub(lb);
    let hi = (k0 + w).min(la);
    let p = (k0 + w).saturating_sub(lb).clamp(lo, hi);
    let q = (k0 + 1).clamp(p, hi);
    [(lo, p), (p, q), (q, hi)]
}

/// The sums `s` of the block `k0..k0 + LANES` of [`school_f64`] with the terms `i` of `r`, every
/// coefficient of the block having each.
#[inline(always)]
fn f64_full(
    mut s: [f64; LANES],
    a: &[f64],
    b: &[f64],
    k0: usize,
    r: (usize, usize),
) -> [f64; LANES] {
    for i in r.0..r.1 {
        let x = a[i];
        if x == 0.0 {
            continue;
        }
        let y: &[f64; LANES] = b[k0 - i..][..LANES].try_into().expect("LANES values");
        for j in 0..LANES {
            s[j] += x * y[j];
        }
    }
    s
}

/// As [`f64_full`] for terms some coefficients of the block do not have: `bp` is `b` between
/// `LANES` zeros, and a coefficient without the term keeps its sum (a select, so that `x·0`
/// for an infinite `x` reaches no sum).
#[inline(always)]
fn f64_part(
    mut s: [f64; LANES],
    a: &[f64],
    bp: &[f64],
    lb: usize,
    k0: usize,
    r: (usize, usize),
) -> [f64; LANES] {
    for i in r.0..r.1 {
        let x = a[i];
        if x == 0.0 {
            continue;
        }
        let y: &[f64; LANES] = bp[k0 + LANES - i..][..LANES]
            .try_into()
            .expect("LANES values");
        // The coefficients k0 + j with the term: i ≤ k0 + j < i + lb.
        let (jlo, jhi) = (i.saturating_sub(k0), (i + lb).saturating_sub(k0));
        for j in 0..LANES {
            let v = s[j] + x * y[j];
            s[j] = if j >= jlo && j < jhi { v } else { s[j] };
        }
    }
    s
}

/// The product of the floats `a` and `b` by the schoolbook method, both not empty: bit for bit
/// the generic schoolbook's (each `c_k` the sum from `+0` of the rounded products `a_i·b_(k-i)`
/// over the non-zero `a_i`, by increasing `i`, so that a term with an infinity or a NaN reaches
/// only the coefficients it belongs to). For an `a` of at least 16 coefficients and a `b` of at
/// least 448 and 4 times as many (where it was measured faster), [`LANES`] coefficients at a
/// time: their sums stay in registers while the terms run past them, each step one product of
/// `a_i` by `LANES` consecutive values of `b` and one sum, which the compiler takes on vectors (a
/// product and a sum, not a fused multiply-add, which would round otherwise); else the generic
/// loop, which the compiler vectorizes over `b`.
fn school_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (la, lb) = (a.len(), b.len());
    // Measured on an Apple M4 Max (`la` 4 to 384, `lb` 8 to 12,288): the generic loop is faster
    // below 16 coefficients in `a` and below about 450 in `b` (2.6 times at 16 by 64, 1.2 at 64
    // by 256), the kernel from there for `lb ≥ 4·la` (1.2 times faster at 64 by 1,024, 1.4 at
    // 192 by 2,048, more where the generic loop's loads and stores alias). The kernel takes any
    // `la`, and `lb ≥ LANES`.
    if la < 16 || lb < 4 * la || lb < 448 {
        return schoolbook(a, b);
    }
    let mut bp = vec![0.0f64; lb + 2 * LANES];
    bp[LANES..LANES + lb].copy_from_slice(b);
    let mut c = vec![0.0f64; la + lb - 1];
    for (blk, out) in c.chunks_mut(LANES).enumerate() {
        let k0 = blk * LANES;
        let [r0, r1, r2] = block_ranges(k0, LANES, la, lb);
        let mut s = f64_part([0.0; LANES], a, &bp, lb, k0, r0);
        s = f64_full(s, a, b, k0, r1);
        s = f64_part(s, a, &bp, lb, k0, r2);
        out.copy_from_slice(&s[..out.len()]);
    }
    c
}

/// [`school_f64`] for complex floats: each step the products `a_i·b_(k-i)` of [`CLANES`]
/// coefficients as the generic schoolbook takes them (`(ar·br − ai·bi, ar·bi + ai·br)`, then
/// added to the real and the imaginary sums), on copies of the real and the imaginary parts of
/// `b`, bit for bit the generic schoolbook's; the generic loop for short operands (where it was
/// measured faster).
fn school_complex(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let (la, lb) = (a.len(), b.len());
    // Measured on an Apple M4 Max (`la` 2 to 64, `lb` 8 to 512): the kernel 1.05 to 1.6 times
    // faster than the generic loop from 12 coefficients in `a` and 64 in `b` (from 96 in `b` at 8
    // in `a`, up to 12% slower below), the generic loop faster below them.
    if la < CLANES || lb < 8 * CLANES {
        return schoolbook(a, b);
    }
    let br: Vec<f64> = b.iter().map(|z| z.re).collect();
    let bi: Vec<f64> = b.iter().map(|z| z.im).collect();
    let mut c = vec![Complex::new(0.0, 0.0); la + lb - 1];
    for (blk, out) in c.chunks_mut(CLANES).enumerate() {
        let k0 = blk * CLANES;
        let mut sr = [0.0f64; CLANES];
        let mut si = [0.0f64; CLANES];
        let [(l0, p), (p0, q), (q0, h)] = block_ranges(k0, CLANES, la, lb);
        let partial = |sr: &mut [f64; CLANES], si: &mut [f64; CLANES], i: usize| {
            let (xr, xi) = (a[i].re, a[i].im);
            if xr == 0.0 && xi == 0.0 {
                return;
            }
            for j in 0..CLANES {
                // The term of the coefficient k0 + j, when it has one.
                let t = (k0 + j).wrapping_sub(i);
                if t < lb {
                    sr[j] += xr * br[t] - xi * bi[t];
                    si[j] += xr * bi[t] + xi * br[t];
                }
            }
        };
        for i in l0..p {
            partial(&mut sr, &mut si, i);
        }
        for i in p0..q {
            let (xr, xi) = (a[i].re, a[i].im);
            if xr == 0.0 && xi == 0.0 {
                continue;
            }
            let yr: &[f64; CLANES] = br[k0 - i..][..CLANES].try_into().expect("CLANES values");
            let yi: &[f64; CLANES] = bi[k0 - i..][..CLANES].try_into().expect("CLANES values");
            for j in 0..CLANES {
                sr[j] += xr * yr[j] - xi * yi[j];
                si[j] += xr * yi[j] + xi * yr[j];
            }
        }
        for i in q0..h {
            partial(&mut sr, &mut si, i);
        }
        for (j, z) in out.iter_mut().enumerate() {
            *z = Complex::new(sr[j], si[j]);
        }
    }
    c
}

/// The product of polynomials over the floats, both not empty: the schoolbook method
/// ([`school_f64`]) wherever [`by_fft`] leaves the product to it, else the crate's linear
/// convolution ([`fft::convolve`](crate::math::fft::convolve)), by real transforms of a length
/// `2^a·3^b·5^c·7^d` (with overlap-add for unbalanced operands, the plans cached) or its direct
/// kernels, as its model of their costs chooses.
pub(crate) fn f64_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if by_fft(a.len(), b.len(), false) {
        return crate::math::fft::convolve(a, b);
    }
    school_f64(a, b)
}

/// The square of a polynomial over the floats, not empty.
pub(crate) fn f64_sqr(a: &[f64]) -> Vec<f64> {
    f64_mul(a, a)
}

/// The product of polynomials over the complex floats, both not empty: the schoolbook method
/// ([`school_complex`]) wherever [`by_fft`] leaves the product to it, else the crate's linear
/// convolution of complex values.
pub(crate) fn complex_mul(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    if by_fft(a.len(), b.len(), true) {
        return crate::math::fft::convolve(a, b);
    }
    school_complex(a, b)
}

/// The square of a polynomial over the complex floats, not empty.
pub(crate) fn complex_sqr(a: &[Complex<f64>]) -> Vec<Complex<f64>> {
    complex_mul(a, a)
}

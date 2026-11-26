//! Newton's method on the orthonormal polynomial of a Gaussian rule, and the weights at its
//! zeros.
//!
//! A family is given by the coefficients of its three-term recurrence,
//! `b_{k+1} p_{k+1}(x) = (x − a_k) p_k(x) − b_k p_{k−1}(x)`, the diagonal `a` and the
//! off-diagonal `b` of its Jacobi matrix, held in double-double. The polynomials are normalised
//! to `p₀ = 1`, so that they are the orthonormal ones times `√μ₀`, and they and their derivatives
//! are rescaled by powers of two as they grow: the Hermite and Laguerre polynomials of high degree
//! at their outermost zeros are far beyond the range of an `f64`, and the scaling keeps them in it
//! without a single rounding.
//!
//! The weight of a zero `x` is Golub and Welsch's `μ₀ v₀²`, `v` the unit eigenvector of the
//! Jacobi matrix for the eigenvalue `x`. That eigenvector is `(p₀(x), …, p_{n−1}(x))` divided by
//! its length, so the weight is the Christoffel number `μ₀ / Σ_{k<n} p_k(x)²` — a sum of
//! positive terms, accurate to the last bit when the node is, and a weight too small for an
//! `f64` comes out as a subnormal number or zero.

use super::dd::{Dd, frexp, ldexp, pow2};

/// The coefficients of a three-term recurrence of degree `n`.
pub(super) struct Recurrence {
    /// The degree of the polynomial whose zeros are the nodes.
    n: usize,
    /// The diagonal `a₀ … a_{n−1}`, or empty for the symmetric families, whose diagonal is 0.
    a: Vec<Dd>,
    /// `b_k` at `k`, `b₀ = 0`, for `k < n`.
    b: Vec<Dd>,
    /// `1/b_k` at `k`, for `1 ≤ k ≤ n` (the first entry unused).
    r: Vec<Dd>,
}

impl Recurrence {
    /// The recurrence of the off-diagonal `b(k)`, `1 ≤ k ≤ n`, and the diagonal `a(k)`, `k < n`,
    /// if there is one.
    fn new(n: usize, b: impl Fn(usize) -> Dd, a: Option<&dyn Fn(usize) -> Dd>) -> Self {
        let mut bs = Vec::with_capacity(n);
        let mut rs = Vec::with_capacity(n + 1);
        bs.push(Dd::ZERO);
        rs.push(Dd::ZERO);
        for k in 1..=n {
            let v = b(k);
            if k < n {
                bs.push(v);
            }
            rs.push(Dd::ONE.div(v));
        }
        let a = a.map(|a| (0..n).map(a).collect()).unwrap_or_default();
        Self { n, a, b: bs, r: rs }
    }

    /// The Legendre polynomials: `a_k = 0`, `b_k = k/√(4k² − 1)`.
    pub(super) fn legendre(n: usize) -> Self {
        Self::new(
            n,
            |k| {
                let k = k as f64;
                Dd::new(k).div(Dd::new(4.0 * k * k - 1.0).sqrt())
            },
            None,
        )
    }

    /// The physicists' Hermite polynomials: `a_k = 0`, `b_k = √(k/2)`.
    pub(super) fn hermite(n: usize) -> Self {
        Self::new(n, |k| Dd::new(0.5 * k as f64).sqrt(), None)
    }

    /// The Laguerre polynomials: `a_k = 2k + 1`, `b_k = k`.
    pub(super) fn laguerre(n: usize) -> Self {
        Self::new(
            n,
            |k| Dd::new(k as f64),
            Some(&|k| Dd::new(2.0 * k as f64 + 1.0)),
        )
    }

    /// The Jacobi polynomials `P^(α,β)`, with the coefficients of Golub and Welsch (1969):
    /// `a_k = (β² − α²)/((2k + α + β)(2k + α + β + 2))` and
    /// `b_k² = 4k(k + α)(k + β)(k + α + β)/((2k + α + β)²(2k + α + β + 1)(2k + α + β − 1))`, the
    /// factor `k + α + β` cancelled at `k = 1`, where it can vanish, and `a₀ = (β − α)/(α + β + 2)`.
    pub(super) fn jacobi(n: usize, alpha: f64, beta: f64) -> Self {
        let (al, be) = (Dd::new(alpha), Dd::new(beta));
        let ab = al.add(be);
        let b = |k: usize| {
            let kd = Dd::new(k as f64);
            let s = kd.add(kd).add(ab);
            let one = Dd::ONE;
            let b2 = if k == 1 {
                Dd::new(4.0)
                    .mul(one.add(al))
                    .mul(one.add(be))
                    .div(s.mul(s).mul(s.add(one)))
            } else {
                Dd::new(4.0)
                    .mul(kd)
                    .mul(kd.add(al))
                    .mul(kd.add(be))
                    .mul(kd.add(ab))
                    .div(s.mul(s).mul(s.add(one)).mul(s.sub(one)))
            };
            b2.sqrt()
        };
        let a = |k: usize| {
            if k == 0 {
                be.sub(al).div(ab.add(Dd::new(2.0)))
            } else {
                let kd = Dd::new(k as f64);
                let s = kd.add(kd).add(ab);
                be.sub(al).mul(be.add(al)).div(s.mul(s.add(Dd::new(2.0))))
            }
        };
        Self::new(n, b, Some(&a))
    }

    /// The degree.
    pub(super) fn degree(&self) -> usize {
        self.n
    }

    /// Golub and Welsch's first guesses: the eigenvalues of the symmetric tridiagonal Jacobi
    /// matrix, ascending, by the tridiagonal QL and QR algorithm of `a::math::matrix` (LAPACK's
    /// `dsterf`) in `O(n²)`, the matrix being tridiagonal already; by its Hermitian eigensolver,
    /// which reduces a dense copy to tridiagonal form first, should that not converge.
    pub(super) fn eigenvalues(&self) -> Vec<f64> {
        use crate::math::matrix::Matrix;
        let n = self.n;
        let mut d: Vec<f64> = if self.a.is_empty() {
            vec![0.0; n]
        } else {
            self.a.iter().map(|a| a.hi).collect()
        };
        let mut e: Vec<f64> = (1..n).map(|k| self.b[k].hi).collect();
        if crate::math::matrix::tridiagonal_eigenvalues(&mut d, &mut e) {
            return d;
        }
        let mut j = Matrix::<f64>::zeros(n, n);
        for k in 0..n {
            if !self.a.is_empty() {
                j[(k, k)] = self.a[k].hi;
            }
            if k + 1 < n {
                let v = self.b[k + 1].hi;
                j[(k + 1, k)] = v;
                j[(k, k + 1)] = v;
            }
        }
        j.hermitian_eigenvalues()
    }
}

/// The arithmetic the recurrence runs in: `f64` for the first iterations, double-double for the
/// last ones and the weights.
trait Arith: Copy {
    const ZERO: Self;
    const ONE: Self;
    fn coef(c: Dd) -> Self;
    fn lead(self) -> f64;
    fn plus(self, o: Self) -> Self;
    fn minus(self, o: Self) -> Self;
    fn times(self, o: Self) -> Self;
    fn scaled(self, k: i32) -> Self;
}

impl Arith for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn coef(c: Dd) -> Self {
        c.hi
    }
    #[inline(always)]
    fn lead(self) -> f64 {
        self
    }
    #[inline(always)]
    fn plus(self, o: Self) -> Self {
        self + o
    }
    #[inline(always)]
    fn minus(self, o: Self) -> Self {
        self - o
    }
    #[inline(always)]
    fn times(self, o: Self) -> Self {
        self * o
    }
    #[inline(always)]
    fn scaled(self, k: i32) -> Self {
        self * pow2(k)
    }
}

impl Arith for Dd {
    const ZERO: Self = Dd::ZERO;
    const ONE: Self = Dd::ONE;
    #[inline(always)]
    fn coef(c: Dd) -> Self {
        c
    }
    #[inline(always)]
    fn lead(self) -> f64 {
        self.hi
    }
    #[inline(always)]
    fn plus(self, o: Self) -> Self {
        self.add(o)
    }
    #[inline(always)]
    fn minus(self, o: Self) -> Self {
        self.sub(o)
    }
    #[inline(always)]
    fn times(self, o: Self) -> Self {
        self.mul(o)
    }
    #[inline(always)]
    fn scaled(self, k: i32) -> Self {
        self.scale(k)
    }
}

/// The value of the polynomial of degree `n` at a point, its derivative, and the sum of the
/// squares of the polynomials of degree below `n`, all scaled: the values are `p · 2^e` and
/// `p′ · 2^e`, the sum `Σ · 2^(2e)`.
struct Eval<A> {
    p: A,
    dp: A,
    sum: A,
    e: i64,
}

/// Past this size the values are scaled down by `2^SHIFT`.
const BIG: f64 = 1.157_920_892_373_162e77; // 2^256
const SHIFT: i32 = 512;

/// The recurrence and its derivative at `x`, with the Christoffel sum.
fn evaluate<A: Arith>(rec: &Recurrence, x: A) -> Eval<A> {
    let (mut p0, mut p1) = (A::ZERO, A::ONE);
    let (mut d0, mut d1) = (A::ZERO, A::ZERO);
    let mut sum = A::ZERO;
    let mut e = 0i64;
    let diagonal = !rec.a.is_empty();
    for k in 0..rec.n {
        sum = sum.plus(p1.times(p1));
        let xa = if diagonal {
            x.minus(A::coef(rec.a[k]))
        } else {
            x
        };
        let bk = A::coef(rec.b[k]);
        let rk = A::coef(rec.r[k + 1]);
        let p2 = xa.times(p1).minus(bk.times(p0)).times(rk);
        let d2 = p1.plus(xa.times(d1)).minus(bk.times(d0)).times(rk);
        p0 = p1;
        p1 = p2;
        d0 = d1;
        d1 = d2;
        if p1.lead().abs() > BIG || d1.lead().abs() > BIG {
            p0 = p0.scaled(-SHIFT);
            p1 = p1.scaled(-SHIFT);
            d0 = d0.scaled(-SHIFT);
            d1 = d1.scaled(-SHIFT);
            // The sum scales as the squares do; terms that fall below the normal range in the
            // process are below 2^−1022 of the next one and do not count.
            sum = sum.scaled(-SHIFT).scaled(-SHIFT);
            e += i64::from(SHIFT);
        }
    }
    Eval {
        p: p1,
        dp: d1,
        sum,
        e,
    }
}

/// A zero of the polynomial, to about 100 bits, with the Christoffel sum there,
/// `sum · 2^(2·exp)`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Zero {
    pub(super) x: Dd,
    sum: Dd,
    exp: i64,
}

impl Zero {
    /// The zero `−x` of a symmetric family, whose Christoffel sum is the same.
    pub(super) fn mirrored(&self) -> Zero {
        Zero {
            x: self.x.neg(),
            ..*self
        }
    }

    /// The weight `μ₀ / Σ p_k(x)²` for the total weight `μ₀ = mu0 · 2^mu0_exp`, rounded once: a
    /// subnormal number or zero where it is too small for an `f64`, infinite only where it is too
    /// large for one.
    pub(super) fn weight(&self, mu0: Dd, mu0_exp: i64) -> f64 {
        let q = mu0.div(self.sum);
        ldexp(q.value(), mu0_exp - 2 * self.exp)
    }

    /// The weight times `e^(g(x))`, from its logarithm, where `g` is the exponent of the weight
    /// function (`x²` for Hermite's, `x` for Laguerre's): such a weight stays near 1 where the
    /// plain one leaves the range of an `f64`. `ln_mu0` is the logarithm of the total weight.
    pub(super) fn scaled_weight(&self, ln_mu0: Dd, g: impl Fn(Dd) -> Dd) -> f64 {
        // ln w = ln μ₀ + g(x) − ln(m) − (f + 2e) ln 2, with sum = m · 2^f, m in [1, 2).
        let (_, f) = frexp(self.sum.hi);
        let m = self.sum.scale(-(f as i32));
        let t = ln_mu0
            .add(g(self.x))
            .sub(Dd::new(m.ln_near_one()))
            .sub(Dd::LN_2.mul_f64((f + 2 * self.exp) as f64));
        t.hi.exp() * (1.0 + t.lo)
    }
}

/// The Christoffel sum at the exact point `x`, for a node known exactly (the middle node 0 of a
/// symmetric rule of odd order).
pub(super) fn at(rec: &Recurrence, x: Dd) -> Zero {
    let v = evaluate::<Dd>(rec, x);
    Zero {
        x,
        sum: v.sum,
        exp: v.e,
    }
}

/// The step of Newton's method in double-double below which a zero is taken as found, relative
/// to the zero: `2^−80`, 27 bits below an ulp of an `f64`. The recurrence cannot do much better
/// than that at the largest zeros of high degree, where its last step cancels: near the largest
/// zero of `Lₙ` the step is only good to about `n · 2^−104` of the zero.
const SETTLED: f64 = 8.271_806_125_530_277e-25; // 2^-80

/// The size below which a zero counts as 0 when its steps are measured against it: the middle
/// zero of a symmetric rule of odd order is 0, and is reached absolutely.
const TINY: f64 = 1e-30;

/// A step that has stopped shrinking but is below this (`2^−64` of the zero) still leaves the
/// zero far better than an ulp of an `f64` away.
const STALLED: f64 = 5.421_010_862_427_522e-20; // 2^-64

/// The zero Newton's method finds from `guess`: in `f64` until the step is below 10⁻¹³ of the
/// point or stops shrinking near the rounding errors of the recurrence, then in double-double
/// until it is below [`SETTLED`] of it. `None` if the iteration fails (a zero derivative or a
/// step that is not a number) or does not settle.
pub(super) fn newton(rec: &Recurrence, guess: f64) -> Option<Zero> {
    let mut x = guess;
    let mut previous = f64::INFINITY;
    for _ in 0..30 {
        let v = evaluate::<f64>(rec, x);
        let step = v.p / v.dp;
        if !step.is_finite() {
            return None;
        }
        x -= step;
        let size = step.abs() / x.abs().max(TINY);
        // Converging quadratically a step is far below the one before it; once it is not, and
        // already small, the `f64` recurrence's rounding is all that is left to remove.
        if size <= 1e-13 || (size <= 1e-8 && step.abs() > 0.25 * previous) {
            break;
        }
        previous = step.abs();
    }
    let mut xd = Dd::new(x);
    let mut last = None;
    for _ in 0..6 {
        let v = evaluate::<Dd>(rec, xd);
        let step = v.p.div(v.dp);
        if !step.hi.is_finite() {
            return None;
        }
        xd = xd.sub(step);
        // The sum was taken at the point before this step, which moved it by less than 2^−64 of
        // itself: the weight there is the weight at the zero to far better than an ulp (the
        // weight changes like 1/(1 − x) at an end of the interval, by n² at most).
        let size = step.hi.abs() / xd.hi.abs().max(TINY);
        let zero = Zero {
            x: xd,
            sum: v.sum,
            exp: v.e,
        };
        if size <= SETTLED {
            return Some(zero);
        }
        last = (size <= STALLED).then_some(zero);
    }
    last
}

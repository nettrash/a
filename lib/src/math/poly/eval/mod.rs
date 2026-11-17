//! Evaluation at many points, products of linear factors and interpolation: Horner's rule below
//! the crossovers; above them subproduct trees, the scaled remainder tree (the transposed
//! multipoint evaluation of Bostan, Lecerf and Schost) and the linear combination of
//! interpolation going up the tree.

mod exact;
mod kron;
mod modtree;
mod modular;
mod mont;
pub(crate) mod tree;
mod word;

pub(crate) use exact::{
    number_eval_many, number_from_roots, rational_eval_many, rational_from_roots,
    rational_interpolate,
};

use super::Polynomial;
use super::basic::trim;
use super::coeff::{Coefficient, Field, sealed};

/// Newton's divided differences of the points `xs` and values `ys`, then the expansion of
/// Newton's form by Horner's rule on polynomials: `O(m²)` operations of the field.
#[track_caller]
pub(crate) fn newton<T: sealed::Field>(xs: &[T], ys: &[T]) -> Vec<T> {
    let m = xs.len();
    let mut c = ys.to_vec();
    for j in 1..m {
        for i in (j..m).rev() {
            let d = T::sub(&xs[i], &xs[i - j]);
            if d.is_zero() {
                repeated(i);
            }
            let t = T::sub(&c[i], &c[i - 1]);
            c[i] = T::div(&t, &d);
        }
    }
    let Some(top) = c.last() else {
        return Vec::new();
    };
    let mut p = vec![top.clone()];
    for i in (0..m - 1).rev() {
        // p ← p·(x - xᵢ) + cᵢ.
        let t = p.last().expect("not empty").clone();
        p.push(t);
        for j in (1..p.len() - 1).rev() {
            let t = T::mul(&xs[i], &p[j]);
            p[j] = T::sub(&p[j - 1], &t);
        }
        p[0] = T::sub(&c[i], &T::mul(&xs[i], &p[0]));
    }
    p
}

/// Lagrange's form `Σ yᵢ/wᵢ · M/(x - xᵢ)`, `wᵢ = Π_{j≠i}(xᵢ - xⱼ)`, each quotient by synthetic
/// division of `M`: `O(m²)` operations of the field.
#[track_caller]
fn lagrange<T: sealed::Field>(xs: &[T], ys: &[T]) -> Vec<T> {
    let m = xs.len();
    if m == 0 {
        return Vec::new();
    }
    let big = T::poly_from_roots(xs);
    let mut out = vec![T::zero(); m];
    for i in 0..m {
        let mut w = T::one();
        for (j, x) in xs.iter().enumerate() {
            if j != i {
                let d = T::sub(&xs[i], x);
                if d.is_zero() {
                    repeated(i.max(j));
                }
                T::mul_assign(&mut w, &d);
            }
        }
        let c = T::div(&ys[i], &w);
        // M / (x - xᵢ): q_{m-1} = 1, q_{j-1} = M_j + xᵢ·q_j.
        let mut q = T::one();
        for j in (0..m).rev() {
            T::mul_add(&mut out[j], &c, &q);
            if j > 0 {
                T::mul_assign(&mut q, &xs[i]);
                T::add_assign(&mut q, &big[j]);
            }
        }
    }
    out
}

/// The panic of a repeated point.
#[track_caller]
fn repeated(i: usize) -> ! {
    panic!("the point {i} is repeated: the points must be distinct")
}

/// The panic of points and values of different lengths.
#[track_caller]
fn lengths(xs: usize, ys: usize) {
    assert_eq!(xs, ys, "as many values as points are needed to interpolate");
}

impl<T: Coefficient> Polynomial<T> {
    /// The values at each of the points `xs`, as [`eval`](Polynomial::eval) gives them one at a
    /// time. Floats take Horner's rule at eight points at once, each value the same as `eval`'s;
    /// integers and rationals Horner's rule for short polynomials or few points, otherwise the
    /// subproduct tree of the points and the scaled remainder tree (the transposed algorithm),
    /// the points in groups of as many as the coefficients; rational points through their common
    /// denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// // x² - 2 at -1, 0, 1, 2.
    /// let p = Polynomial::new(vec![Number::from(-2), Number::from(0), Number::from(1)]);
    /// let xs: Vec<Number> = (-1..=2).map(Number::from).collect();
    /// let v: Vec<Number> = [-1, -2, -1, 2].into_iter().map(Number::from).collect();
    /// assert_eq!(p.eval_many(&xs), v);
    ///
    /// let q = Polynomial::new(vec![0.5, 0.0, 2.0]);
    /// assert_eq!(q.eval_many(&[1.0, -2.0]), [2.5, 8.5]);
    /// ```
    pub fn eval_many(&self, xs: &[T]) -> Vec<T> {
        T::poly_eval_many(&self.c, xs)
    }

    /// The monic polynomial `Π (x - rᵢ)` of the roots `roots`, by the product tree of the linear
    /// factors (one factor after the other for a few); rational roots `aᵢ/bᵢ` as
    /// `Π (bᵢ x - aᵢ) / Π bᵢ` over the integers. The constant 1 for no roots.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let r = [Number::from(1), Number::from(-2), Number::from(3)];
    /// let p = Polynomial::from_roots(&r);
    /// assert_eq!(p.to_string(), "x^3 - 2*x^2 - 5*x + 6");
    /// ```
    pub fn from_roots(roots: &[T]) -> Self {
        Polynomial::new(T::poly_from_roots(roots))
    }
}

impl<T: Field> Polynomial<T> {
    /// The polynomial of degree below `xs.len()` that takes the value `ys[i]` at `xs[i]`; the zero
    /// polynomial for no points. Over the rationals by the subproduct tree of the points brought
    /// to integers by their common denominator: the weights `1/M'(xᵢ)` by the scaled remainder
    /// tree and the linear combination up the tree over a denominator a node; for floats by
    /// Newton's form ([`interpolate_newton`](Polynomial::interpolate_newton)), as the trees lose
    /// the accuracy of floats.
    ///
    /// # Panics
    ///
    /// Panics if `xs` and `ys` differ in length or a point is repeated.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// // The parabola through (0, 1), (1, 0) and (2, 3): 2x² - 3x + 1.
    /// let xs: Vec<Rational> = (0..3).map(Rational::from).collect();
    /// let ys: Vec<Rational> = [1, 0, 3].into_iter().map(Rational::from).collect();
    /// let p = Polynomial::interpolate(&xs, &ys);
    /// assert_eq!(p.to_string(), "2*x^2 - 3*x + 1");
    /// ```
    #[track_caller]
    pub fn interpolate(xs: &[T], ys: &[T]) -> Self {
        lengths(xs.len(), ys.len());
        Polynomial::new(T::poly_interpolate(xs, ys))
    }

    /// [`interpolate`](Polynomial::interpolate) by Lagrange's form, `Σ yᵢ · Π_{j≠i} (x - xⱼ) /
    /// (xᵢ - xⱼ)`: `O(m²)` operations of the field for `m` points.
    ///
    /// # Panics
    ///
    /// As [`interpolate`](Polynomial::interpolate).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// let p = Polynomial::interpolate_lagrange(&[0.0, 1.0, 2.0], &[1.0, 0.0, 3.0]);
    /// assert_eq!(p.coeffs(), [1.0, -3.0, 2.0]);
    /// ```
    #[track_caller]
    pub fn interpolate_lagrange(xs: &[T], ys: &[T]) -> Self {
        lengths(xs.len(), ys.len());
        let mut c = lagrange(xs, ys);
        trim(&mut c);
        Polynomial { c }
    }

    /// [`interpolate`](Polynomial::interpolate) by Newton's divided differences and the expansion
    /// of Newton's form: `O(m²)` operations of the field for `m` points.
    ///
    /// # Panics
    ///
    /// As [`interpolate`](Polynomial::interpolate).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// let p = Polynomial::interpolate_newton(&[0.0, 1.0, 2.0], &[1.0, 0.0, 3.0]);
    /// assert_eq!(p.coeffs(), [1.0, -3.0, 2.0]);
    /// ```
    #[track_caller]
    pub fn interpolate_newton(xs: &[T], ys: &[T]) -> Self {
        lengths(xs.len(), ys.len());
        Polynomial::new(newton(xs, ys))
    }
}

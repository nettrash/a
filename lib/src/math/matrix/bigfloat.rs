//! Matrices of arbitrary-precision floats and complex numbers: [`BigFloat`] and [`BigComplex`]
//! as elements, with products whose every element is correctly rounded (the exact dot product,
//! rounded once), and determinants, solutions, inverses, ranks and norms at the working
//! precision of the elements.

use super::element::sealed;
use super::{Element, Matrix, Norm};
use crate::math::float::complex::fast::{self, Reg};
use crate::math::float::sum::{Term, Work, round_summed, sqrt_summed, sum_terms_in};
use crate::math::float::{BigComplex, BigFloat, Round};
use core::cmp::Ordering;

impl Element for BigFloat {}
impl Element for BigComplex {}

/// `1` of one bit, which rounds nothing it multiplies.
fn one_bit() -> BigFloat {
    BigFloat::from(1u8).with_prec(1)
}

impl sealed::Element for BigFloat {
    /// `+0` of one bit, which an operator's larger precision leaves out.
    #[inline]
    fn zero() -> Self {
        BigFloat::zero(1)
    }
    #[inline]
    fn one() -> Self {
        one_bit()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        BigFloat::is_zero(self)
    }
    #[inline]
    fn conj(&self) -> Self {
        self.clone()
    }
    #[inline]
    fn add(a: &Self, b: &Self) -> Self {
        a + b
    }
    #[inline]
    fn sub(a: &Self, b: &Self) -> Self {
        a - b
    }
    #[inline]
    fn mul(a: &Self, b: &Self) -> Self {
        a * b
    }
    #[inline]
    fn neg(a: &Self) -> Self {
        -a
    }
    #[inline]
    fn add_assign(a: &mut Self, b: &Self) {
        *a += b;
    }
    #[inline]
    fn sub_assign(a: &mut Self, b: &Self) {
        *a -= b;
    }
    #[inline]
    fn mul_assign(a: &mut Self, b: &Self) {
        *a *= b;
    }
    /// `c + a·b` rounded once, at the largest precision of the three.
    #[inline]
    fn mul_add(c: &mut Self, a: &Self, b: &Self) {
        *c = a.mul_add(b, c);
    }
    /// Each element the exact dot product rounded once to nearest at the largest precision of
    /// the elements of both matrices.
    fn product(c: &mut [Self], a: &[Self], b: &[Self], m: usize, k: usize, n: usize) {
        let prec = max_prec(a.iter().chain(b)).max(1);
        let mut work = Work::default();
        let mut terms = Vec::with_capacity(k);
        let finite = a.iter().chain(b).all(BigFloat::is_finite);
        for j in 0..n {
            let col = &b[j * k..(j + 1) * k];
            for i in 0..m {
                let term = |p: usize| fast::Term::product(&a[i + p * m], &col[p], false);
                if finite {
                    let mut x = BigFloat::zero(prec);
                    if fast::short_dot(&mut x, k, term, Round::Nearest).is_some() {
                        c[i + j * m] = x;
                        continue;
                    }
                }
                terms.clear();
                terms.extend((0..k).map(|p| Term::product(&a[i + p * m], &col[p], false)));
                c[i + j * m] =
                    round_summed(sum_terms_in(&terms, prec, &mut work), prec, Round::Nearest).0;
            }
        }
    }
}

impl sealed::Element for BigComplex {
    /// `+0 + 0i` of one bit.
    #[inline]
    fn zero() -> Self {
        BigComplex::zero(1)
    }
    #[inline]
    fn one() -> Self {
        BigComplex::from(one_bit())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        BigComplex::is_zero(self)
    }
    #[inline]
    fn conj(&self) -> Self {
        BigComplex::conj(self)
    }
    #[inline]
    fn add(a: &Self, b: &Self) -> Self {
        a + b
    }
    #[inline]
    fn sub(a: &Self, b: &Self) -> Self {
        a - b
    }
    #[inline]
    fn mul(a: &Self, b: &Self) -> Self {
        a * b
    }
    #[inline]
    fn neg(a: &Self) -> Self {
        -a
    }
    #[inline]
    fn add_assign(a: &mut Self, b: &Self) {
        *a += b;
    }
    #[inline]
    fn sub_assign(a: &mut Self, b: &Self) {
        *a -= b;
    }
    #[inline]
    fn mul_assign(a: &mut Self, b: &Self) {
        *a *= b;
    }
    /// `c + a·b`, each part rounded once at the largest precision of that part of the three,
    /// when all are finite; with an infinity or NaN, the product and then the sum.
    fn mul_add(c: &mut Self, a: &Self, b: &Self) {
        if !(a.is_finite() && b.is_finite() && c.is_finite()) {
            *c = &*c + &(a * b);
            return;
        }
        let pr = c.re().prec().max(a.re().prec()).max(b.re().prec());
        let pi = c.im().prec().max(a.im().prec()).max(b.im().prec());
        let mut work = Work::default();
        let re = [
            Term::one(c.re(), false),
            Term::product(a.re(), b.re(), false),
            Term::product(a.im(), b.im(), true),
        ];
        let im = [
            Term::one(c.im(), false),
            Term::product(a.re(), b.im(), false),
            Term::product(a.im(), b.re(), false),
        ];
        let re = round_summed(sum_terms_in(&re, pr, &mut work), pr, Round::Nearest).0;
        let im = round_summed(sum_terms_in(&im, pi, &mut work), pi, Round::Nearest).0;
        *c = BigComplex::new(re, im);
    }
    /// Each part of each element the exact sum of the exact products, rounded once to nearest
    /// at the largest precision of the parts of both matrices, when they are all finite; with
    /// an infinity or NaN, the products and sums one by one.
    fn product(c: &mut [Self], a: &[Self], b: &[Self], m: usize, k: usize, n: usize) {
        if !a.iter().chain(b).all(BigComplex::is_finite) {
            for j in 0..n {
                for i in 0..m {
                    let mut s = BigComplex::zero(1);
                    for p in 0..k {
                        s = &s + &(&a[i + p * m] * &b[p + j * k]);
                    }
                    c[i + j * m] = s;
                }
            }
            return;
        }
        let prec = a
            .iter()
            .chain(b)
            .map(|z| z.re().prec().max(z.im().prec()))
            .max()
            .unwrap_or(1);
        let mut work = Work::default();
        let mut re = Vec::with_capacity(2 * k);
        let mut im = Vec::with_capacity(2 * k);
        for j in 0..n {
            let col = &b[j * k..(j + 1) * k];
            for i in 0..m {
                // Parts of one or two limbs: each part's exact sum on the stack.
                let tr = |t: usize| {
                    let (x, y) = (&a[i + (t / 2) * m], &col[t / 2]);
                    if t.is_multiple_of(2) {
                        fast::Term::product(x.re(), y.re(), false)
                    } else {
                        fast::Term::product(x.im(), y.im(), true)
                    }
                };
                let ti = |t: usize| {
                    let (x, y) = (&a[i + (t / 2) * m], &col[t / 2]);
                    if t.is_multiple_of(2) {
                        fast::Term::product(x.re(), y.im(), false)
                    } else {
                        fast::Term::product(x.im(), y.re(), false)
                    }
                };
                let n = Round::Nearest;
                if let (Some(r), Some(s)) = (
                    fast::short_dot_value(2 * k, tr, prec, n),
                    fast::short_dot_value(2 * k, ti, prec, n),
                ) {
                    let mut z = BigComplex::zero(prec);
                    fast::put_short(&mut z.re, r, n);
                    fast::put_short(&mut z.im, s, n);
                    c[i + j * m] = z;
                    continue;
                }
                re.clear();
                im.clear();
                for (p, y) in col.iter().enumerate() {
                    let x = &a[i + p * m];
                    re.push(Term::product(x.re(), y.re(), false));
                    re.push(Term::product(x.im(), y.im(), true));
                    im.push(Term::product(x.re(), y.im(), false));
                    im.push(Term::product(x.im(), y.re(), false));
                }
                let r = round_summed(sum_terms_in(&re, prec, &mut work), prec, Round::Nearest).0;
                let s = round_summed(sum_terms_in(&im, prec, &mut work), prec, Round::Nearest).0;
                c[i + j * m] = BigComplex::new(r, s);
            }
        }
    }
}

/// The largest precision of the floats.
fn max_prec<'a>(xs: impl Iterator<Item = &'a BigFloat>) -> u64 {
    xs.map(BigFloat::prec).max().unwrap_or(1)
}

/// What Gaussian elimination needs of an element beyond the ring operations.
trait Field: Element + 'static {
    /// The largest precision of the parts.
    fn precision(&self) -> u64;
    /// A zero of the precision `prec` in each part, to write into.
    fn blank(prec: u64) -> Self;
    /// A size to choose the pivots by: `|x|`, or `|z|` to 64 bits.
    fn size(&self) -> BigFloat;
    /// Whether `self` is larger than `other` by [`size`](Field::size), NaN never.
    fn larger(&self, other: &Self) -> bool {
        self.size().partial_cmp(&other.size()) == Some(Ordering::Greater)
    }
    /// `a / b` rounded to nearest at `prec`.
    fn quotient(a: &Self, b: &Self, prec: u64) -> Self;
    /// `a · b` rounded to nearest at `prec`.
    fn times(a: &Self, b: &Self, prec: u64) -> Self;
    /// `c - Σ x_p·y_p`, each part rounded once to nearest at `prec`.
    fn dot_sub<'a>(
        c: &Self,
        xy: impl Iterator<Item = (&'a Self, &'a Self)>,
        prec: u64,
        work: &mut Work,
    ) -> Self;
    /// Sets `out`, each part of the precision `prec`, to `c - Σ_{p < len} x_p·y_p` (`c` is `out`
    /// itself when `None`), as [`dot_sub`](Field::dot_sub) has it, into the vectors of `out`.
    fn dot_sub_in<'a>(
        out: &mut Self,
        c: Option<&Self>,
        len: usize,
        xy: impl Fn(usize) -> (&'a Self, &'a Self),
        prec: u64,
        work: &mut Work,
    ) {
        let v = {
            let c = c.unwrap_or(out);
            Self::dot_sub(c, (0..len).map(xy), prec, work)
        };
        *out = v;
    }
    /// Sets `x`, of the precision `prec`, to `x / d` rounded to nearest, into its vectors.
    fn quotient_in(x: &mut Self, d: &Self, prec: u64) {
        *x = Self::quotient(x, d, prec);
    }
    /// The number at the precision `prec`, exactly when it is of that precision or less.
    fn at_prec(&self, prec: u64) -> Self;
    /// Gives the vectors of the number to the thread's pool, where [`at_prec`](Field::at_prec)
    /// takes them.
    fn recycle(self);
    /// The factorisation in registers when the elements allow it (see [`RegFactors`]): `None`
    /// when they do not, `Some(None)` for a pivot that is exactly zero.
    fn reg_factor(_a: &Matrix<Self>, _prec: u64) -> Option<Option<RegFactors>> {
        None
    }
    /// The solution by the factors in registers, `None` when a result leaves the exponent
    /// range.
    fn reg_solve(_f: &RegFactors, _b: &Matrix<Self>, _prec: u64) -> Option<Matrix<Self>> {
        None
    }
    /// The determinant by the factors in registers, `None` when a result leaves the exponent
    /// range.
    fn reg_det(_f: &RegFactors, _prec: u64) -> Option<Self> {
        None
    }
}

impl Field for BigFloat {
    #[inline]
    fn precision(&self) -> u64 {
        self.prec()
    }
    #[inline]
    fn blank(prec: u64) -> Self {
        BigFloat::zero(prec)
    }
    #[inline]
    fn size(&self) -> BigFloat {
        self.abs()
    }
    #[inline]
    fn larger(&self, other: &Self) -> bool {
        self.cmp_abs(other) == Some(Ordering::Greater)
    }
    #[inline]
    fn quotient(a: &Self, b: &Self, prec: u64) -> Self {
        a.div_round(b, prec, Round::Nearest).0
    }
    #[inline]
    fn times(a: &Self, b: &Self, prec: u64) -> Self {
        a.mul_round(b, prec, Round::Nearest).0
    }
    fn dot_sub<'a>(
        c: &Self,
        xy: impl Iterator<Item = (&'a Self, &'a Self)>,
        prec: u64,
        work: &mut Work,
    ) -> Self {
        let mut xy = xy.peekable();
        if xy.peek().is_none() {
            return c.prec_round(prec, Round::Nearest).0;
        }
        let mut terms = vec![Term::one(c, false)];
        terms.extend(xy.map(|(x, y)| Term::product(x, y, true)));
        round_summed(sum_terms_in(&terms, prec, work), prec, Round::Nearest).0
    }
    /// The exact sum on the stack ([`fast::short_dot`]) when the numbers are finite and of one or
    /// two limbs, rounded into `out`; otherwise [`dot_sub`](Field::dot_sub).
    fn dot_sub_in<'a>(
        out: &mut Self,
        c: Option<&Self>,
        len: usize,
        xy: impl Fn(usize) -> (&'a Self, &'a Self),
        prec: u64,
        work: &mut Work,
    ) {
        debug_assert_eq!(out.prec(), prec);
        let done = {
            let c = c.unwrap_or(out);
            let finite = c.is_finite()
                && (0..len).all(|p| {
                    let (x, y) = xy(p);
                    x.is_finite() && y.is_finite()
                });
            let term = |i: usize| {
                if i == 0 {
                    fast::Term::one(c, false)
                } else {
                    let (x, y) = xy(i - 1);
                    fast::Term::product(x, y, true)
                }
            };
            if finite {
                fast::short_dot_value(len + 1, term, prec, Round::Nearest)
            } else {
                None
            }
        };
        if let Some(r) = done {
            fast::put_short(out, r, Round::Nearest);
            return;
        }
        let v = {
            let c = c.unwrap_or(out);
            Self::dot_sub(c, (0..len).map(xy), prec, work)
        };
        *out = v;
    }
    #[inline]
    fn quotient_in(x: &mut Self, d: &Self, prec: u64) {
        debug_assert_eq!(x.prec(), prec);
        x.div_assign_round(d, Round::Nearest);
    }
    #[inline]
    fn at_prec(&self, prec: u64) -> Self {
        if self.prec() == prec {
            fast::pooled_copy(self)
        } else {
            self.with_prec(prec)
        }
    }
    #[inline]
    fn recycle(self) {
        fast::recycle(self);
    }
    fn reg_factor(a: &Matrix<Self>, prec: u64) -> Option<Option<RegFactors>> {
        RegFactors::of(a, prec)
    }
    fn reg_solve(f: &RegFactors, b: &Matrix<Self>, prec: u64) -> Option<Matrix<Self>> {
        f.solve(b, prec)
    }
    fn reg_det(f: &RegFactors, prec: u64) -> Option<Self> {
        f.det(prec)
    }
}

impl Field for BigComplex {
    #[inline]
    fn precision(&self) -> u64 {
        self.re().prec().max(self.im().prec())
    }
    #[inline]
    fn blank(prec: u64) -> Self {
        BigComplex::zero(prec)
    }
    #[inline]
    fn size(&self) -> BigFloat {
        self.abs_round(64, Round::Nearest).0
    }
    #[inline]
    fn quotient(a: &Self, b: &Self, prec: u64) -> Self {
        a.div_round(b, prec, Round::Nearest).0
    }
    #[inline]
    fn times(a: &Self, b: &Self, prec: u64) -> Self {
        a.mul_round(b, prec, Round::Nearest).0
    }
    fn dot_sub<'a>(
        c: &Self,
        xy: impl Iterator<Item = (&'a Self, &'a Self)>,
        prec: u64,
        work: &mut Work,
    ) -> Self {
        let xy: Vec<(&BigComplex, &BigComplex)> = xy.collect();
        if !(c.is_finite() && xy.iter().all(|(x, y)| x.is_finite() && y.is_finite())) {
            let mut s = c.with_prec(prec);
            for (x, y) in &xy {
                s = s
                    .sub_round(
                        &x.mul_round(y, prec, Round::Nearest).0,
                        prec,
                        Round::Nearest,
                    )
                    .0;
            }
            return s;
        }
        let mut re = Vec::with_capacity(2 * xy.len() + 1);
        let mut im = Vec::with_capacity(2 * xy.len() + 1);
        re.push(Term::one(c.re(), false));
        im.push(Term::one(c.im(), false));
        for (x, y) in &xy {
            re.push(Term::product(x.re(), y.re(), true));
            re.push(Term::product(x.im(), y.im(), false));
            im.push(Term::product(x.re(), y.im(), true));
            im.push(Term::product(x.im(), y.re(), true));
        }
        let r = round_summed(sum_terms_in(&re, prec, work), prec, Round::Nearest).0;
        let i = round_summed(sum_terms_in(&im, prec, work), prec, Round::Nearest).0;
        BigComplex::new(r, i)
    }
    #[inline]
    fn at_prec(&self, prec: u64) -> Self {
        BigComplex::new(self.re().at_prec(prec), self.im().at_prec(prec))
    }
    /// Each part's exact sum on the stack when the parts are finite and of one or two limbs,
    /// rounded into `out`; otherwise [`dot_sub`](Field::dot_sub).
    fn dot_sub_in<'a>(
        out: &mut Self,
        c: Option<&Self>,
        len: usize,
        xy: impl Fn(usize) -> (&'a Self, &'a Self),
        prec: u64,
        work: &mut Work,
    ) {
        let n = Round::Nearest;
        let parts = {
            let c = c.unwrap_or(out);
            let tr = |t: usize| {
                if t == 0 {
                    return fast::Term::one(c.re(), false);
                }
                let (x, y) = xy((t - 1) / 2);
                if t % 2 == 1 {
                    fast::Term::product(x.re(), y.re(), true)
                } else {
                    fast::Term::product(x.im(), y.im(), false)
                }
            };
            let ti = |t: usize| {
                if t == 0 {
                    return fast::Term::one(c.im(), false);
                }
                let (x, y) = xy((t - 1) / 2);
                if t % 2 == 1 {
                    fast::Term::product(x.re(), y.im(), true)
                } else {
                    fast::Term::product(x.im(), y.re(), true)
                }
            };
            match fast::short_dot_value(2 * len + 1, tr, prec, n) {
                Some(r) => fast::short_dot_value(2 * len + 1, ti, prec, n).map(|s| (r, s)),
                None => None,
            }
        };
        if let Some((r, s)) = parts {
            fast::put_short(out.re_mut(), r, n);
            fast::put_short(out.im_mut(), s, n);
            return;
        }
        let v = {
            let c = c.unwrap_or(out);
            Self::dot_sub(c, (0..len).map(xy), prec, work)
        };
        *out = v;
    }
    #[inline]
    fn recycle(self) {
        let (re, im) = self.into_parts();
        fast::recycle(re);
        fast::recycle(im);
    }
}

/// An LU factorisation with partial pivoting, `P·A = L·U`, of a square matrix: `L` below the
/// diagonal (its diagonal ones), `U` on and above, column after column, the row of `A` at each
/// row of the factors, and the sign of the permutation; `None` for a pivot that is exactly zero.
struct Factors<T> {
    lu: Vec<T>,
    n: usize,
    perm: Vec<usize>,
    odd: bool,
}

/// Doolittle's method with partial pivoting at the precision `prec`: each element of `U`, and of
/// `L` before its division by the pivot, is one dot product of the elements before it, exactly,
/// rounded once.
fn factor<T: Field>(a: &Matrix<T>, prec: u64) -> Option<Factors<T>> {
    let n = a.rows();
    // At the working precision, exactly, to be changed in place.
    let mut lu: Vec<T> = a.as_slice().iter().map(|x| x.at_prec(prec)).collect();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut odd = false;
    let mut work = Work::default();
    // `lu[at] -= Σ_{p<len} lu[x(p)]·lu[y(p)]`, the element taken out while the others are read.
    let mut update = |lu: &mut Vec<T>,
                      at: usize,
                      len: usize,
                      x: &dyn Fn(usize) -> usize,
                      y: &dyn Fn(usize) -> usize| {
        let mut v = core::mem::replace(&mut lu[at], T::zero());
        {
            let lu = &*lu;
            T::dot_sub_in(
                &mut v,
                None,
                len,
                |p| (&lu[x(p)], &lu[y(p)]),
                prec,
                &mut work,
            );
        }
        lu[at] = v;
    };
    for k in 0..n {
        // Column k from the diagonal down: `a_ik - Σ_{p<k} l_ip·u_pk`.
        for i in k..n {
            update(&mut lu, i + k * n, k, &|p| i + p * n, &|p| p + k * n);
        }
        // The pivot: the largest from the diagonal down.
        let mut r = k;
        for i in k + 1..n {
            if lu[i + k * n].larger(&lu[r + k * n]) {
                r = i;
            }
        }
        // A pivot that is zero (or NaN): no factorisation.
        if !lu[r + k * n].larger(&T::zero()) {
            return None;
        }
        if r != k {
            for j in 0..n {
                lu.swap(r + j * n, k + j * n);
            }
            perm.swap(r, k);
            odd = !odd;
        }
        for i in k + 1..n {
            let mut v = core::mem::replace(&mut lu[i + k * n], T::zero());
            T::quotient_in(&mut v, &lu[k + k * n], prec);
            lu[i + k * n] = v;
        }
        // Row k right of the diagonal: `a_kj - Σ_{p<k} l_kp·u_pj`.
        for j in k + 1..n {
            update(&mut lu, k + j * n, k, &|p| k + p * n, &|p| p + j * n);
        }
    }
    Some(Factors { lu, n, perm, odd })
}

impl<T: Field> Factors<T> {
    /// Gives the vectors of the factors to the thread's pool.
    fn recycle(self) {
        for x in self.lu {
            x.recycle();
        }
    }

    /// The solution of `A·X = B`: `L·Y = P·B` from the top, then `U·X = Y` from the bottom, each
    /// element one dot product rounded once, divided by the pivot.
    fn solve(&self, b: &Matrix<T>, prec: u64) -> Matrix<T> {
        let (n, cols) = (self.n, b.cols());
        let lu = &self.lu;
        let mut work = Work::default();
        let mut out = Vec::with_capacity(n * cols);
        for j in 0..cols {
            let col = b.column(j);
            // `y_i = b_perm(i) - Σ_{p<i} l_ip·y_p`, then `x_i = (y_i - Σ_{p>i} u_ip·x_p)/u_ii`
            // in the place of `y_i`.
            let mut y: Vec<T> = Vec::with_capacity(n);
            for i in 0..n {
                let mut v = T::blank(prec);
                T::dot_sub_in(
                    &mut v,
                    Some(&col[self.perm[i]]),
                    i,
                    |p| (&lu[i + p * n], &y[p]),
                    prec,
                    &mut work,
                );
                y.push(v);
            }
            for i in (0..n).rev() {
                let mut v = core::mem::replace(&mut y[i], T::zero());
                {
                    let x = &y;
                    T::dot_sub_in(
                        &mut v,
                        None,
                        n - 1 - i,
                        |q| (&lu[i + (i + 1 + q) * n], &x[i + 1 + q]),
                        prec,
                        &mut work,
                    );
                }
                T::quotient_in(&mut v, &lu[i + i * n], prec);
                y[i] = v;
            }
            out.extend(y);
        }
        Matrix::from_column_major(n, cols, out)
    }

    /// The determinant: the product of the pivots, rounded at each step, and the permutation's
    /// sign.
    fn det(&self, prec: u64) -> T {
        let n = self.n;
        let mut d = T::one();
        for i in 0..n {
            d = T::times(&d, &self.lu[i + i * n], prec);
        }
        if self.odd { T::neg(&d) } else { d }
    }
}

/// The factorisation of [`factor`] of a matrix of finite [`BigFloat`]s of at most two limbs each
/// (to 128 bits) in registers ([`Reg`]): each element of the factors one exact dot product
/// rounded once to nearest, as [`factor`] has it, on the stack, so that its results are those of
/// [`factor`] exactly, without a vector for each element; `None` when an exact sum spans more
/// than the accumulator or a result leaves the exponent range.
pub(crate) struct RegFactors {
    lu: Vec<Reg>,
    n: usize,
    perm: Vec<usize>,
    odd: bool,
}

impl RegFactors {
    /// The factors of `a` at the precision `prec` (at most 128 bits), as [`Field::reg_factor`].
    fn of(a: &Matrix<BigFloat>, prec: u64) -> Option<Option<RegFactors>> {
        if prec > 128 {
            return None;
        }
        let n = a.rows();
        let mut lu: Vec<Reg> = a.as_slice().iter().map(Reg::of).collect::<Option<_>>()?;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut odd = false;
        // `lu[at] - Σ_{p<len} lu[x(p)]·lu[y(p)]`.
        let dot = |lu: &[Reg],
                   at: usize,
                   len: usize,
                   x: &dyn Fn(usize) -> usize,
                   y: &dyn Fn(usize) -> usize| {
            fast::reg_dot(
                len + 1,
                |t| {
                    if t == 0 {
                        (lu[at], None, false)
                    } else {
                        (lu[x(t - 1)], Some(lu[y(t - 1)]), true)
                    }
                },
                prec,
            )
        };
        for k in 0..n {
            for i in k..n {
                lu[i + k * n] = dot(&lu, i + k * n, k, &|p| i + p * n, &|p| p + k * n)?;
            }
            let mut r = k;
            for i in k + 1..n {
                if lu[i + k * n].larger(lu[r + k * n]) {
                    r = i;
                }
            }
            if lu[r + k * n].is_zero() {
                return Some(None);
            }
            if r != k {
                for j in 0..n {
                    lu.swap(r + j * n, k + j * n);
                }
                perm.swap(r, k);
                odd = !odd;
            }
            let pivot = lu[k + k * n];
            for i in k + 1..n {
                lu[i + k * n] = lu[i + k * n].div(pivot, prec)?;
            }
            for j in k + 1..n {
                lu[k + j * n] = dot(&lu, k + j * n, k, &|p| k + p * n, &|p| p + j * n)?;
            }
        }
        Some(Some(RegFactors { lu, n, perm, odd }))
    }

    /// The solution of `A·X = B`, as [`Factors::solve`] has it.
    fn solve(&self, b: &Matrix<BigFloat>, prec: u64) -> Option<Matrix<BigFloat>> {
        let (n, cols) = (self.n, b.cols());
        let lu = &self.lu;
        let mut out = Vec::with_capacity(n * cols);
        let mut y: Vec<Reg> = Vec::with_capacity(n);
        for j in 0..cols {
            let col: Vec<Reg> = b.column(j).iter().map(Reg::of).collect::<Option<_>>()?;
            y.clear();
            for i in 0..n {
                let c = col[self.perm[i]];
                let v = fast::reg_dot(
                    i + 1,
                    |t| {
                        if t == 0 {
                            (c, None, false)
                        } else {
                            (lu[i + (t - 1) * n], Some(y[t - 1]), true)
                        }
                    },
                    prec,
                )?;
                y.push(v);
            }
            for i in (0..n).rev() {
                let v = fast::reg_dot(
                    n - i,
                    |t| {
                        if t == 0 {
                            (y[i], None, false)
                        } else {
                            (lu[i + (i + t) * n], Some(y[i + t]), true)
                        }
                    },
                    prec,
                )?;
                y[i] = v.div(lu[i + i * n], prec)?;
            }
            out.extend(y.iter().map(|x| x.to_float(prec)));
        }
        Some(Matrix::from_column_major(n, cols, out))
    }

    /// The determinant, as [`Factors::det`] has it.
    fn det(&self, prec: u64) -> Option<BigFloat> {
        let n = self.n;
        let mut d = Reg::of(&one_bit()).expect("one");
        for i in 0..n {
            d = d.mul(self.lu[i + i * n], prec)?;
        }
        if self.odd {
            d = d.neg();
        }
        Some(d.to_float(prec))
    }
}

/// The working precision of a matrix: the largest of its elements', 1 for none.
fn work_prec<T: Field>(a: &Matrix<T>) -> u64 {
    a.as_slice().iter().map(Field::precision).max().unwrap_or(1)
}

/// The determinant, solution and inverse of a square matrix, generic over the two element types.
fn det_of<T: Field>(a: &Matrix<T>) -> T {
    assert!(a.is_square(), "the matrix is not square");
    let prec = work_prec(a);
    match T::reg_factor(a, prec) {
        Some(Some(f)) => {
            if let Some(d) = T::reg_det(&f, prec) {
                return d;
            }
        }
        Some(None) => return T::zero(),
        None => {}
    }
    match factor(a, prec) {
        Some(f) => {
            let d = f.det(prec);
            f.recycle();
            d
        }
        // A column of zeros in the remaining block, exactly: the determinant is zero.
        None => T::zero(),
    }
}

fn solve_of<T: Field>(a: &Matrix<T>, b: &Matrix<T>) -> Option<Matrix<T>> {
    assert!(a.is_square(), "the matrix is not square");
    assert_eq!(
        a.rows(),
        b.rows(),
        "the right-hand side has not as many rows as the matrix"
    );
    let prec = work_prec(a).max(work_prec(b));
    match T::reg_factor(a, prec) {
        Some(Some(f)) => {
            if let Some(x) = T::reg_solve(&f, b, prec) {
                return Some(x);
            }
        }
        Some(None) => return None,
        None => {}
    }
    factor(a, prec).map(|f| {
        let x = f.solve(b, prec);
        f.recycle();
        x
    })
}

fn inverse_of<T: Field>(a: &Matrix<T>) -> Option<Matrix<T>> {
    assert!(a.is_square(), "the matrix is not square");
    let prec = work_prec(a);
    let n = a.rows();
    match T::reg_factor(a, prec) {
        Some(Some(f)) => {
            if let Some(x) = T::reg_solve(&f, &Matrix::identity(n), prec) {
                return Some(x);
            }
        }
        Some(None) => return None,
        None => {}
    }
    factor(a, prec).map(|f| {
        let x = f.solve(&Matrix::identity(n), prec);
        f.recycle();
        x
    })
}

/// The rank by Gaussian elimination with complete pivoting at the precision `prec`: the number
/// of pivots above `tol`, the elimination stopping at the first that is not.
fn rank_of<T: Field>(a: &Matrix<T>, tol: &BigFloat, prec: u64) -> usize {
    let (m, n) = a.shape();
    let mut w = a.as_slice().to_vec();
    let mut work = Work::default();
    let mut rank = 0;
    for k in 0..m.min(n) {
        // The largest element left.
        let (mut r, mut c, mut best) = (k, k, BigFloat::zero(1));
        for j in k..n {
            for i in k..m {
                let s = w[i + j * m].size();
                if s.partial_cmp(&best) == Some(Ordering::Greater) {
                    (r, c, best) = (i, j, s);
                }
            }
        }
        if best.partial_cmp(tol) != Some(Ordering::Greater) {
            break;
        }
        rank += 1;
        for j in 0..n {
            w.swap(r + j * m, k + j * m);
        }
        for i in 0..m {
            w.swap(i + c * m, i + k * m);
        }
        let pivot = w[k + k * m].clone();
        for i in k + 1..m {
            let l = T::quotient(&w[i + k * m], &pivot, prec);
            for j in k + 1..n {
                let v = T::dot_sub(
                    &w[i + j * m],
                    [(&l, &w[k + j * m])].into_iter(),
                    prec,
                    &mut work,
                );
                w[i + j * m] = v;
            }
        }
    }
    rank
}

/// The default tolerance of a rank: `max(m, n)·2^(1 - w)·max|a_ij|`, NumPy's `matrix_rank` with
/// the unit `2^(1 - w)` of the working precision `w` and pivots for singular values; for a
/// complex matrix the sizes are `|z|` to 64 bits.
fn default_tol<T: Field>(a: &Matrix<T>, prec: u64) -> BigFloat {
    let (m, n) = a.shape();
    let max = a
        .as_slice()
        .iter()
        .map(Field::size)
        .fold(BigFloat::zero(64), |acc, s| if s > acc { s } else { acc });
    let unit = BigFloat::from_parts(
        &crate::math::numbers::Number::from(m.max(n) as u64),
        1 - prec as i64,
        64,
        Round::Up,
    )
    .0;
    max.mul_round(&unit, 64, Round::Up).0
}

impl Matrix<BigFloat> {
    /// The determinant, by an LU factorisation with partial pivoting at the working precision:
    /// the largest precision of the elements. Each element of the factors is one dot product of
    /// the elements before it, exactly, rounded once to nearest (and divided by the pivot in
    /// `L`), and the determinant the product of the pivots, rounded at each step; each operation
    /// is correctly rounded, the determinant as a whole is not. Zero when a pivot is exactly
    /// zero; the determinant of the `0 × 0` matrix is 1.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::matrix::Matrix;
    ///
    /// let a = Matrix::from_rows(&[[2, 1], [4, 4]]).map(|&x| BigFloat::from(x).with_prec(100));
    /// assert_eq!(a.det(), BigFloat::from(4));
    /// ```
    #[track_caller]
    pub fn det(&self) -> BigFloat {
        det_of(self)
    }

    /// The solution `X` of `A·X = B`, by the factorisation of [`det`](Matrix::<BigFloat>::det)
    /// and substitutions whose every element is one dot product rounded once and divided by the
    /// pivot, at the larger working precision of `A` and `B`; `None` when a pivot is exactly
    /// zero. Each operation is correctly rounded; the solution as a whole is not, and it is as
    /// accurate as the condition of `A` allows at that precision.
    ///
    /// # Panics
    ///
    /// Panics if `A` is not square or `B` has not as many rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::matrix::Matrix;
    ///
    /// let a = Matrix::from_rows(&[[1, 2], [4, 4]]).map(|&x| BigFloat::from(x));
    /// let b = Matrix::from_rows(&[[5], [12]]).map(|&x| BigFloat::from(x));
    /// let x = a.solve(&b).unwrap();
    /// assert_eq!(x, Matrix::from_rows(&[[1], [2]]).map(|&x| BigFloat::from(x)));
    /// ```
    #[track_caller]
    pub fn solve(&self, b: &Matrix<BigFloat>) -> Option<Matrix<BigFloat>> {
        solve_of(self, b)
    }

    /// The inverse, as the solution of `A·X = I` (see [`solve`](Matrix::<BigFloat>::solve));
    /// `None` when a pivot is exactly zero.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::matrix::Matrix;
    ///
    /// let a = Matrix::from_rows(&[[2, 0], [0, 4]]).map(|&x| BigFloat::from(x));
    /// let inv = a.inverse().unwrap();
    /// assert_eq!(inv[(1, 1)], BigFloat::from(0.25));
    /// ```
    #[track_caller]
    pub fn inverse(&self) -> Option<Matrix<BigFloat>> {
        inverse_of(self)
    }

    /// The numerical rank: the number of pivots of Gaussian elimination with complete pivoting,
    /// at the working precision `w`, above `max(m, n)·2^(1 - w)·max|a_ij|`, as NumPy's
    /// `matrix_rank` with pivots for singular values. An estimate: a matrix of exactly
    /// representable elements that is singular can have a rank one above its exact one when its
    /// elimination rounds; [`Matrix<Rational>`](crate::math::rational::Rational) has exact ranks.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::matrix::Matrix;
    ///
    /// let a = Matrix::from_rows(&[[1, 2, 3], [2, 4, 6], [1, 0, 1]]).map(|&x| BigFloat::from(x));
    /// assert_eq!(a.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        let prec = work_prec(self);
        rank_of(self, &default_tol(self, prec), prec)
    }

    /// The rank with the pivots above `tol` counted (see [`rank`](Matrix::<BigFloat>::rank));
    /// a tolerance of zero counts every pivot that is not exactly zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::matrix::Matrix;
    ///
    /// let a = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1e-30]]).map(|&x| BigFloat::from(x));
    /// assert_eq!(a.rank_with_tolerance(&BigFloat::from(1e-20)), 1);
    /// assert_eq!(a.rank_with_tolerance(&BigFloat::zero(53)), 2);
    /// ```
    pub fn rank_with_tolerance(&self, tol: &BigFloat) -> usize {
        rank_of(self, &tol.abs(), work_prec(self))
    }

    /// The norm of the kind given, correctly rounded to nearest at the working precision: the
    /// one-norm the largest of the columns' sums of absolute values, each sum exact and rounded
    /// once; the infinity-norm the same of the rows; the Frobenius norm the square root of the
    /// exact sum of the squares, rounded once; and the largest absolute value. NaN when an
    /// element is NaN.
    ///
    /// # Panics
    ///
    /// Panics for [`Norm::Two`] and [`Norm::Nuclear`], which take singular values.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::matrix::{Matrix, Norm};
    ///
    /// let a = Matrix::from_rows(&[[1, -2], [-3, 4]]).map(|&x| BigFloat::from(x));
    /// assert_eq!(a.norm(Norm::One), BigFloat::from(6));
    /// assert_eq!(a.norm(Norm::Infinity), BigFloat::from(7));
    /// assert_eq!(a.norm(Norm::Max), BigFloat::from(4));
    /// assert_eq!(a.norm(Norm::Frobenius), BigFloat::from(30).sqrt());
    /// ```
    #[track_caller]
    pub fn norm(&self, norm: Norm) -> BigFloat {
        let prec = work_prec(self);
        let (m, n) = self.shape();
        let mut work = Work::default();
        let sum_abs = |xs: &mut dyn Iterator<Item = &BigFloat>, work: &mut Work| {
            let abs: Vec<BigFloat> = xs.map(BigFloat::abs).collect();
            let terms: Vec<Term<'_>> = abs.iter().map(|x| Term::one(x, false)).collect();
            round_summed(sum_terms_in(&terms, prec, work), prec, Round::Nearest).0
        };
        let largest = |xs: Vec<BigFloat>| -> BigFloat {
            xs.into_iter().fold(BigFloat::zero(prec), |acc, x| {
                if acc.is_nan() || x.is_nan() {
                    BigFloat::nan(prec)
                } else if x > acc {
                    x.with_prec(prec)
                } else {
                    acc
                }
            })
        };
        match norm {
            Norm::One => largest(
                (0..n)
                    .map(|j| sum_abs(&mut self.column(j).iter(), &mut work))
                    .collect(),
            ),
            Norm::Infinity => largest(
                (0..m)
                    .map(|i| sum_abs(&mut self.row(i), &mut work))
                    .collect(),
            ),
            Norm::Max => largest(self.as_slice().iter().map(BigFloat::abs).collect()),
            Norm::Frobenius => {
                let terms: Vec<Term<'_>> = self
                    .as_slice()
                    .iter()
                    .map(|x| Term::product(x, x, false))
                    .collect();
                sqrt_summed(
                    sum_terms_in(&terms, 2 * prec + 4, &mut work),
                    prec,
                    Round::Nearest,
                )
                .0
            }
            Norm::Two | Norm::Nuclear => {
                panic!(
                    "the 2-norm and the nuclear norm of a matrix of BigFloats take singular values"
                )
            }
        }
    }
}

impl Matrix<BigComplex> {
    /// The determinant, as [`Matrix::<BigFloat>::det`] has it: an LU factorisation with partial
    /// pivoting by `|z|` at the largest precision of the parts, each part of each element of
    /// the factors one dot product rounded once, the quotients and the product of the pivots
    /// correctly rounded at each step.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    /// use a::math::matrix::Matrix;
    ///
    /// let i = BigComplex::i(53);
    /// let one = BigComplex::from(1);
    /// let a = Matrix::from_rows(&[[one.clone(), i.clone()], [i.clone(), one.clone()]]);
    /// assert_eq!(a.det(), BigComplex::from(2)); // 1 - i²
    /// ```
    #[track_caller]
    pub fn det(&self) -> BigComplex {
        det_of(self)
    }

    /// The solution `X` of `A·X = B`, as [`Matrix::<BigFloat>::solve`] has it; `None` when a
    /// pivot is exactly zero.
    ///
    /// # Panics
    ///
    /// Panics if `A` is not square or `B` has not as many rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    /// use a::math::matrix::Matrix;
    ///
    /// let a = Matrix::from_rows(&[[BigComplex::i(53)]]);
    /// let b = Matrix::from_rows(&[[BigComplex::from(2)]]);
    /// let x = a.solve(&b).unwrap();
    /// assert_eq!(x[(0, 0)], BigComplex::new(BigFloat::from(0), BigFloat::from(-2)));
    /// ```
    #[track_caller]
    pub fn solve(&self, b: &Matrix<BigComplex>) -> Option<Matrix<BigComplex>> {
        solve_of(self, b)
    }

    /// The inverse, as the solution of `A·X = I`; `None` when a pivot is exactly zero.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    /// use a::math::matrix::Matrix;
    ///
    /// let a = Matrix::from_rows(&[[BigComplex::new(BigFloat::from(1), BigFloat::from(1))]]);
    /// let inv = a.inverse().unwrap();
    /// assert_eq!(inv[(0, 0)], BigComplex::new(BigFloat::from(0.5), BigFloat::from(-0.5)));
    /// ```
    #[track_caller]
    pub fn inverse(&self) -> Option<Matrix<BigComplex>> {
        inverse_of(self)
    }

    /// The numerical rank, as [`Matrix::<BigFloat>::rank`] has it, the pivots chosen and
    /// compared by `|z|` against `max(m, n)·2^(1 - w)·max|z_ij|`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    /// use a::math::matrix::Matrix;
    ///
    /// let i = BigComplex::i(53);
    /// let a = Matrix::from_rows(&[[BigComplex::from(1), i.clone()], [i.clone(), BigComplex::from(-1)]]);
    /// assert_eq!(a.rank(), 1);
    /// ```
    pub fn rank(&self) -> usize {
        let prec = work_prec(self);
        rank_of(self, &default_tol(self, prec), prec)
    }
}

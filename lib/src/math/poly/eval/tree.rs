//! The subproduct tree of points of a ring `T` and the product of linear factors by it, Horner's
//! rule at many points at once, and the products of coefficients the trees take. (The scaled
//! remainder tree is on packed residues, in [`modtree`](super::modtree): over the integers and
//! the rationals the values grow with the degree, and evaluation point by point costs as much.)
//!
//! The products are a [`Mul`]: the coefficient type's hook, or a kernel of this module.

use crate::math::poly::coeff::sealed::Coefficient;

/// The product of the coefficients `a` and `b`, both not empty: `a.len() + b.len() - 1` of them.
pub(crate) type Mul<T> = fn(&[T], &[T]) -> Vec<T>;

/// How many points Horner's rule carries at once.
const BLOCK: usize = 8;

/// The values of the polynomial of the coefficients `c` at the points `xs`, by Horner's rule as
/// [`eval`](crate::math::poly::Polynomial::eval), `BLOCK` points at once for independent chains:
/// each value the same as `eval`'s, bit for bit for floats.
pub(crate) fn horner_many<T: Coefficient>(c: &[T], xs: &[T]) -> Vec<T> {
    let Some((top, rest)) = c.split_last() else {
        return vec![T::zero(); xs.len()];
    };
    let mut out = Vec::with_capacity(xs.len());
    for x in xs.chunks(BLOCK) {
        let mut v: Vec<T> = vec![top.clone(); x.len()];
        for a in rest.iter().rev() {
            for (v, x) in v.iter_mut().zip(x) {
                T::mul_assign(v, x);
                T::add_assign(v, a);
            }
        }
        out.extend(v);
    }
    out
}

/// The subproduct tree of `m ≥ 1` points: the levels `1..=h` (the first empty), each node the
/// monic product of the linear factors of its leaves.
pub(crate) struct Tree<T> {
    pts: Vec<T>,
    lv: Vec<Vec<Vec<T>>>,
    h: usize,
}

/// The tree of the points `pts`, at least one, by the products `mul`.
pub(crate) fn build<T: Coefficient>(pts: &[T], mul: Mul<T>) -> Tree<T> {
    let m = pts.len();
    debug_assert!(m >= 1);
    let mut h = 0;
    while (1usize << h) < m {
        h += 1;
    }
    let mut lv: Vec<Vec<Vec<T>>> = vec![Vec::new()];
    if h >= 1 {
        let v = pts
            .chunks(2)
            .map(|p| match p {
                [a, b] => vec![T::mul(a, b), T::neg(&T::add(a, b)), T::one()],
                [a] => vec![T::neg(a), T::one()],
                _ => unreachable!(),
            })
            .collect();
        lv.push(v);
    }
    for l in 1..h {
        let v = lv[l]
            .chunks(2)
            .map(|p| match p {
                [a, b] => mul(a, b),
                [a] => a.clone(),
                _ => unreachable!(),
            })
            .collect();
        lv.push(v);
    }
    Tree {
        pts: pts.to_vec(),
        lv,
        h,
    }
}

/// The product of the polynomials `p`, not empty, each not empty, by a product tree.
pub(crate) fn product<T: Coefficient>(mut p: Vec<Vec<T>>, mul: Mul<T>) -> Vec<T> {
    while p.len() > 1 {
        p = p
            .chunks(2)
            .map(|q| match q {
                [a, b] => mul(a, b),
                [a] => a.clone(),
                _ => unreachable!(),
            })
            .collect();
    }
    p.pop().expect("one polynomial")
}

/// Up to this many roots, [`from_roots`] multiplies by one linear factor after the other.
const LINEAR_ROOTS: usize = 16;

/// `Π (x - rᵢ)`, monic: one linear factor after the other for a few roots, otherwise by the product
/// tree.
pub(crate) fn from_roots<T: Coefficient>(r: &[T], mul: Mul<T>) -> Vec<T> {
    if r.len() <= LINEAR_ROOTS {
        let mut p = Vec::with_capacity(r.len() + 1);
        p.push(T::one());
        for a in r {
            // p ← p·(x - a): p[i] = p[i-1] - a·p[i], from the top.
            let top = p.last().expect("not empty").clone();
            p.push(top);
            for i in (1..p.len() - 1).rev() {
                let t = T::mul(a, &p[i]);
                p[i] = T::sub(&p[i - 1], &t);
            }
            p[0] = T::neg(&T::mul(a, &p[0]));
        }
        return p;
    }
    build(r, mul).root()
}

impl<T: Coefficient> Tree<T> {
    /// The node `j` of the level `l`, the linear factor of the point `j` for `l = 0`.
    pub(crate) fn node(&self, l: usize, j: usize) -> std::borrow::Cow<'_, [T]> {
        if l == 0 {
            std::borrow::Cow::Owned(vec![T::neg(&self.pts[j]), T::one()])
        } else {
            std::borrow::Cow::Borrowed(&self.lv[l][j])
        }
    }

    /// The monic product of all the linear factors.
    pub(crate) fn root(&self) -> Vec<T> {
        if self.h == 0 {
            return vec![T::neg(&self.pts[0]), T::one()];
        }
        self.lv[self.h][0].clone()
    }
}

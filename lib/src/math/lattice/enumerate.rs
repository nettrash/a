//! The lattice points in an ellipsoid by enumeration: Fincke and Pohst's method (*Improved
//! methods for calculating vectors of short length in a lattice, including a complexity
//! analysis*, Math. Comp. 44, 1985) in Schnorr and Euchner's zig-zag order with the radius cut
//! after each better solution (*Lattice basis reduction: improved practical algorithms and
//! solving subset sum problems*, Math. Programming 66, 1994), on the Gram–Schmidt
//! orthogonalisation of an LLL-reduced basis.
//!
//! Every answer is exact. The Gram–Schmidt data are computed exactly, as integers (from their
//! residues modulo primes, [`Gso::of_residues`]), and rounded once to `f64`; the search compares rigorous lower bounds of its partial norms with a radius
//! enlarged past their rounding, so it visits every node it must, and each leaf is checked in
//! integers. Where floats cannot follow the data (Gram–Schmidt norms below `2⁻⁴⁰` of the
//! radius, or coefficients beyond `2⁵⁰`) the same search runs in integers. The top levels no
//! solution can use are left out first: without a target, the last rows whose Gram–Schmidt
//! norms all exceed the radius; with one, those where a coefficient other than zero would take
//! the distance beyond it.

use super::bkz::reduce;
use super::enum_gso::{Gso, dot};
use super::enum_search::{Levels, Target, all, least};
use crate::math::matrix::Matrix;
use crate::math::numbers::Number;

/// A vector of a lattice: its coordinates `x` in the basis given, the vector `v = Σ xᵢ·bᵢ`, and
/// `|v|²`, or for [`closest_vector`] the squared distance `|v − t|²` to the target.
///
/// # Examples
///
/// ```
/// use a::math::lattice::shortest_vector;
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// let b = Matrix::from_rows(&[[3, 1], [7, 3]]).map(|&x| Number::from(x));
/// let s = shortest_vector(&b).unwrap();
/// // The vector is the combination of the rows its coordinates say.
/// for j in 0..2 {
///     let sum = &(&s.coordinates[0] * &b[(0, j)]) + &(&s.coordinates[1] * &b[(1, j)]);
///     assert_eq!(sum, s.vector[j]);
/// }
/// assert_eq!(s.norm2, Number::from(2u8));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeVector {
    /// The coordinates in the basis given, one for each row.
    pub coordinates: Vec<Number>,
    /// The vector.
    pub vector: Vec<Number>,
    /// Its squared norm, or its squared distance to the target.
    pub norm2: Number,
}

/// A basis made ready for an enumeration: the nonzero rows of its LLL reduction, the rows of the
/// transformation that give them from the rows given, and their Gram–Schmidt data.
struct Prepared {
    rows: Vec<Vec<Number>>,
    trans: Vec<Vec<Number>>,
    gso: Gso,
}

impl Prepared {
    fn new(basis: &Matrix<Number>) -> Self {
        let r = reduce(basis, 0.99, 0.51);
        let zeros = basis.rows() - r.rank();
        let take = |m: &Matrix<Number>| -> Vec<Vec<Number>> {
            (zeros..m.rows())
                .map(|i| (0..m.cols()).map(|j| m[(i, j)].clone()).collect())
                .collect()
        };
        let rows = take(r.basis());
        let trans = take(r.transform());
        let gso = Gso::of_residues(&rows, rows.len())
            .expect("the rows of a reduced basis are independent");
        Prepared { rows, trans, gso }
    }

    fn rank(&self) -> usize {
        self.rows.len()
    }

    /// The vector of the coefficients `x` on the reduced rows, and its coordinates in the basis
    /// given.
    fn vector(&self, x: &[Number], cols: usize, n: usize) -> (Vec<Number>, Vec<Number>) {
        (combine(&self.rows, x, cols), combine(&self.trans, x, n))
    }
}

/// `Σ xᵢ·rowsᵢ`, of `width` elements.
fn combine(rows: &[Vec<Number>], x: &[Number], width: usize) -> Vec<Number> {
    let mut v = vec![Number::default(); width];
    for (row, xi) in rows.iter().zip(x) {
        if xi.is_zero() {
            continue;
        }
        for (s, e) in v.iter_mut().zip(row) {
            if !e.is_zero() {
                *s += &(xi * e);
            }
        }
    }
    v
}

fn numbers(x: &[i64]) -> Vec<Number> {
    x.iter().map(|&v| Number::from(v)).collect()
}

/// Of `±v`, the one whose first nonzero element is positive (its coordinates with it).
fn normalise(v: &mut LatticeVector) {
    if v.vector
        .iter()
        .find(|e| !e.is_zero())
        .is_some_and(Number::is_negative)
    {
        for e in v.vector.iter_mut().chain(v.coordinates.iter_mut()) {
            *e = -&*e;
        }
    }
}

/// A shortest nonzero vector of the lattice of the rows of `basis`, exactly; `None` if the
/// lattice is `{0}`. Of `±v`, the one whose first nonzero element is positive.
///
/// The rows may be dependent and fewer or more than the columns. The basis is LLL-reduced
/// first, and the search starts from the radius of its first vector, shrinking at each shorter
/// one; it visits `2^O(n²)` nodes for a lattice of rank `n`, which is practical up to a rank of
/// about 40 to 50.
///
/// # Panics
///
/// Panics if a coefficient of the search, on the reduced basis, does not fit in 64 bits, which
/// takes a rank far beyond what an enumeration can finish.
///
/// # Examples
///
/// ```
/// use a::math::lattice::shortest_vector;
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// // A basis of the lattice of the integer vectors with an even sum, D₃: its shortest vectors
/// // have the norm 2, and no row given is one of them.
/// let b = Matrix::from_rows(&[[1, 1, 4], [3, 1, 0], [5, 2, 1]]).map(|&x| Number::from(x));
/// let s = shortest_vector(&b).unwrap();
/// assert_eq!(s.norm2, Number::from(2u8));
/// assert_eq!(s.vector.iter().map(|e| e.abs()).sum::<Number>(), Number::from(2u8));
/// assert!(shortest_vector(&Matrix::<Number>::zeros(2, 3)).is_none());
/// ```
#[must_use]
pub fn shortest_vector(basis: &Matrix<Number>) -> Option<LatticeVector> {
    let p = Prepared::new(basis);
    let n = p.rank();
    if n == 0 {
        return None;
    }
    let levels = Levels {
        gso: &p.gso,
        lo: 0,
        hi: n,
        target: None,
    };
    // Shorter than the first vector of the reduced basis, |b₀|² = d₁.
    let first = &p.gso.d[1];
    let x = match least(&levels, &(first - 1u8)) {
        Some((x, _)) => numbers(&x),
        None => (0..n).map(|i| Number::from(u8::from(i == 0))).collect(),
    };
    let (vector, coordinates) = p.vector(&x, basis.cols(), basis.rows());
    let norm2 = dot(&vector, &vector);
    let mut v = LatticeVector {
        coordinates,
        vector,
        norm2,
    };
    normalise(&mut v);
    Some(v)
}

/// A vector of the lattice of the rows of `basis` closest to `target`, exactly; `None` if the
/// basis has no rows. The [`norm2`](LatticeVector::norm2) is the squared distance
/// `|v − target|²`; of several closest vectors, one.
///
/// The basis is LLL-reduced and the target brought near the origin by Babai's nearest plane
/// method, whose lattice vector is the first bound; the enumeration then searches around the
/// target for closer ones, at the cost of [`shortest_vector`]. A target outside the span of the
/// rows is measured with its component orthogonal to them.
///
/// # Panics
///
/// Panics if `target` has not as many elements as `basis` has columns, or, as
/// [`shortest_vector`], if a coefficient of the search does not fit in 64 bits.
///
/// # Examples
///
/// ```
/// use a::math::lattice::closest_vector;
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// let b = Matrix::from_rows(&[[2, 0, 0], [0, 3, 0], [1, 1, 5]]).map(|&x| Number::from(x));
/// let t = [7, -4, 11].map(Number::from);
/// let c = closest_vector(&b, &t).unwrap();
/// // 3·(2, 0, 0) − 2·(0, 3, 0) + 2·(1, 1, 5) = (8, −4, 10) is at the squared distance 1 + 0 + 1,
/// // and (6, −4, 10) too.
/// assert_eq!(c.norm2, Number::from(2u8));
/// let d: Number = c.vector.iter().zip(&t).map(|(v, t)| (v - t) * (v - t)).sum();
/// assert_eq!(d, c.norm2);
/// ```
#[track_caller]
#[must_use]
pub fn closest_vector(basis: &Matrix<Number>, target: &[Number]) -> Option<LatticeVector> {
    let (rows, cols) = basis.shape();
    assert_eq!(
        target.len(),
        cols,
        "the target has {} elements, the basis {cols} columns",
        target.len()
    );
    if rows == 0 {
        return None;
    }
    let p = Prepared::new(basis);
    let n = p.rank();
    let g = &p.gso;
    // Babai's nearest plane method, exactly: t − Σ qₗ·bₗ has |μ_{t,l}| ≤ 1/2 at every level.
    let mut lam = g.coefficients(&p.rows, target);
    let mut t = target.to_vec();
    let mut q = vec![Number::default(); n];
    for l in (0..n).rev() {
        let den = &g.d[l + 1];
        let ql = (&(&lam[l] * 2u8) + den).div_floor(&(den * 2u8));
        if ql.is_zero() {
            continue;
        }
        for (e, b) in t.iter_mut().zip(&p.rows[l]) {
            *e -= &(&ql * b);
        }
        for (lj, glj) in lam.iter_mut().zip(&g.lam[l]) {
            *lj -= &(&ql * glj);
        }
        lam[l] -= &(&ql * den);
        q[l] = ql;
    }
    let babai = dot(&t, &t);
    let norms = g.projected_norms(&babai, &lam);
    let levels = Levels {
        gso: g,
        lo: 0,
        hi: n,
        target: Some(Target {
            lam: &lam,
            norms: &norms,
        }),
    };
    let mut x = q;
    if let Some((y, _)) = least(&levels, &(&babai - 1u8)) {
        for (xi, yi) in x.iter_mut().zip(&y) {
            *xi += *yi;
        }
    }
    let (vector, coordinates) = p.vector(&x, cols, rows);
    let diff: Vec<Number> = vector.iter().zip(target).map(|(v, t)| v - t).collect();
    let norm2 = dot(&diff, &diff);
    Some(LatticeVector {
        coordinates,
        vector,
        norm2,
    })
}

/// The nonzero vectors `v` of the lattice of the rows of `basis` with `|v|² ≤ bound`, one of
/// each pair `±v` (the one whose first nonzero element is positive), in order of their norms
/// and, among equal norms, lexicographically by their elements: the vectors PARI's `qfminim`
/// finds, in an order that depends on the lattice alone, not on its basis.
///
/// All of them are found, exactly, by the enumeration of [`shortest_vector`] with its radius
/// held at `bound`. Their number is about the volume of the ball of that radius over the
/// lattice's, `(π·bound)^(n/2) / (Γ(n/2 + 1)·vol(L))` for a rank `n`, and the work grows with it.
///
/// # Panics
///
/// Panics if a coefficient of the search does not fit in 64 bits: a bound that large holds more
/// vectors than memory can.
///
/// # Examples
///
/// ```
/// use a::math::lattice::short_vectors;
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// // The hexagonal lattice A₂ has six vectors of the norm 2, three pairs.
/// let b = Matrix::from_rows(&[[1, -1, 0], [0, 1, -1]]).map(|&x| Number::from(x));
/// let s = short_vectors(&b, &Number::from(2u8));
/// assert_eq!(s.len(), 3);
/// assert!(s.iter().all(|v| v.norm2 == Number::from(2u8)));
/// assert_eq!(s[0].vector, [0, 1, -1].map(Number::from));
/// // The next norm is 6.
/// assert_eq!(short_vectors(&b, &Number::from(6u8)).len(), 6);
/// ```
#[must_use]
pub fn short_vectors(basis: &Matrix<Number>, bound: &Number) -> Vec<LatticeVector> {
    if *bound < 1u8 {
        return Vec::new();
    }
    let p = Prepared::new(basis);
    let n = p.rank();
    if n == 0 {
        return Vec::new();
    }
    let levels = Levels {
        gso: &p.gso,
        lo: 0,
        hi: n,
        target: None,
    };
    let mut out: Vec<LatticeVector> = all(&levels, bound)
        .into_iter()
        .map(|(x, norm2)| {
            let (vector, coordinates) = p.vector(&numbers(&x), basis.cols(), basis.rows());
            let mut v = LatticeVector {
                coordinates,
                vector,
                norm2,
            };
            normalise(&mut v);
            v
        })
        .collect();
    out.sort_by(|a, b| a.norm2.cmp(&b.norm2).then_with(|| a.vector.cmp(&b.vector)));
    out
}

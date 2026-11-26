//! The rank profiles of a matrix of integers, found modulo a prime and then proved over the
//! rationals.
//!
//! The column rank profile `J` of an `m × n` matrix `A` of rank `r` is the first `r` columns, from
//! the left, that are independent: column `c` is in `J` when it is not in the span of the columns
//! before it. The pivots of the Hermite normal form of the rows are in exactly those columns. The
//! row rank profile from the bottom `R` is the same for the rows taken from the last one up; the
//! rows outside it, `S`, are then each a combination of rows of `R` below them, which makes the
//! transformation of [`super::hermite`] the one of the Hermite normal form of `[A | I]`.
//!
//! Both are found modulo a prime `p` below `2^62` by the PLE decomposition of
//! [`ModMatrix`](super::super::ModMatrix): its pivots are the column rank profile, and those of
//! the decomposition of `Aᵀ` with its columns reversed are the rows from the bottom (when only
//! some independent rows are wanted, the first `r` rows of the first decomposition's permutation
//! are). That costs `O(m·n·r^(ω−2))` operations on words, and gives the true profiles unless `p`
//! divides one of a few minors. They are then proved exactly:
//! with `M = A[R, J]` nonsingular (its determinant is not zero modulo `p`), the solution
//! `X = M⁻¹·A[R, K]` for the columns `K` outside `J` must satisfy `A[S, J]·X = A[S, K]`, which
//! makes every column a combination of the columns `J` and the rank `r`, and must be zero where a
//! column of `J` comes after the column of `K` it expresses, which makes `J` the column profile;
//! when `R` is wanted too, `Y = A[S, J]·M⁻¹` must be zero where a row of `R` comes before the row
//! of `S` it expresses (`Y` is computed as the integers `N = |det M|·Y`, from their residues modulo
//! primes, which are zero where `Y` is). When a check fails, the next prime is tried. Each prime that fails divides
//! one of the `m + n` minors the profiles depend on, so a matrix fails at most a number of primes
//! that is the length of those minors in bits over 61, and as a rule none.

use super::super::modp::{Field, TopPrimes, decompose};
use super::super::{Matrix, ModMatrix};
use super::hermite::matrix_of;
use super::inverse::times_inverse;
use super::pernet;
use crate::math::numbers::Number;
use crate::math::rational::Rational;
use core::cell::OnceCell;

/// The rank profiles of a matrix, proved, and what the proof computed.
pub(crate) struct Basis {
    /// The rank `r`.
    pub(crate) rank: usize,
    /// The column rank profile `J`, ascending: the columns of the pivots.
    pub(crate) cols: Vec<usize>,
    /// The columns outside `J`, ascending.
    pub(crate) free: Vec<usize>,
    /// The row rank profile from the bottom `R`, ascending; a set of `r` independent rows when
    /// it was not asked for.
    pub(crate) rows: Vec<usize>,
    /// The rows outside `R`, ascending.
    pub(crate) others: Vec<usize>,
    /// `M = A[R, J]`, nonsingular.
    pub(crate) minor: Matrix<Number>,
    /// `|det M|`, positive: 1 for rank 0; found when first asked for ([`Basis::det`]), or set
    /// by whoever finds it on the way.
    pub(crate) det: OnceCell<Number>,
    /// `X = M⁻¹·A[R, K]`, `r × (n − r)`: `A[:, K] = A[:, J]·X`.
    pub(crate) x: Matrix<Rational>,
    /// `N = |det M|·Y` for `Y = A[S, J]·M⁻¹`, `(m − r) × r`, of integers: `A[S, :] = Y·A[R, :]`;
    /// when asked for.
    pub(crate) n: Option<Matrix<Number>>,
    /// The Hermite form of the rows of `A[:, J]`, when it is known already.
    pub(crate) hj: OnceCell<Matrix<Number>>,
}

impl Basis {
    /// `|det M|`: the determinant of the minor, computed the first time.
    pub(crate) fn det(&self) -> &Number {
        self.det.get_or_init(|| self.minor.det().abs())
    }
}

/// The rank profiles of `a`, proved; with the row rank profile from the bottom when `rows_too` is
/// set, and some `r` independent rows otherwise.
pub(crate) fn basis(a: &Matrix<Number>, rows_too: bool) -> Basis {
    // The primes from the largest below `2^62` down.
    for p in TopPrimes::default() {
        if let Some(b) = prove(a, &candidate(a, p, rows_too), rows_too) {
            return b;
        }
    }
    unreachable!("a matrix whose rank profiles are wrong modulo every prime below 2^62")
}

/// Rank profiles found modulo a prime: the columns and the rows from the bottom, ascending.
struct Candidate {
    cols: Vec<usize>,
    rows: Vec<usize>,
}

/// The rank profiles of `a` modulo the prime `p`: the pivot columns of the PLE decomposition of
/// `A`, the column rank profile; and, when `rows_too` is set, the pivot columns of the
/// decomposition of `Aᵀ` with its columns reversed, the rows from the bottom that are
/// independent, otherwise the rows the first decomposition took its pivots from, independent.
fn candidate(a: &Matrix<Number>, p: u64, rows_too: bool) -> Candidate {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Candidate {
            cols: Vec::new(),
            rows: Vec::new(),
        };
    }
    let f = Field::new(p);
    let residues = ModMatrix::from_matrix_prime(a, p);
    let src = residues.as_slice();
    let mut work = src.to_vec();
    let e = decompose(f, &mut work, m, n);
    let mut rows: Vec<usize> = if rows_too {
        // `B[j][c] = A[m − 1 − c][j]`: column `c` of `B` is row `m − 1 − c` of `A`.
        let mut b = vec![0u64; n * m];
        for (i, row) in src.chunks_exact(n).enumerate() {
            for (j, &x) in row.iter().enumerate() {
                b[j * m + (m - 1 - i)] = x;
            }
        }
        decompose(f, &mut b, n, m)
            .pivots
            .iter()
            .map(|&c| m - 1 - c)
            .collect()
    } else {
        e.perm[..e.rank].to_vec()
    };
    rows.sort_unstable();
    Candidate {
        cols: e.pivots,
        rows,
    }
}

/// The indices below `len` that are not in the ascending list `of`.
fn complement(of: &[usize], len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len - of.len());
    let mut it = of.iter().peekable();
    for i in 0..len {
        if it.peek() == Some(&&i) {
            it.next();
        } else {
            out.push(i);
        }
    }
    out
}

/// The submatrix of `a` on the rows `rows` and the columns `cols`.
pub(crate) fn select(a: &Matrix<Number>, rows: &[usize], cols: &[usize]) -> Matrix<Number> {
    Matrix::from_fn(rows.len(), cols.len(), |i, j| a[(rows[i], cols[j])].clone())
}

/// The matrix of integers `a` as rationals.
pub(crate) fn rationals(a: &Matrix<Number>) -> Matrix<Rational> {
    a.map(|x| Rational::from(x.clone()))
}

/// The candidate `c` proved to be the column rank profile of `a` and, when `rows_too` is set, its
/// row rank profile from the bottom, with what the proof computed; or `None` when it is not.
fn prove(a: &Matrix<Number>, c: &Candidate, rows_too: bool) -> Option<Basis> {
    let (m, n) = a.shape();
    let r = c.cols.len();
    let free = complement(&c.cols, n);
    let others = complement(&c.rows, m);
    if r == 0 {
        // Rank 0 modulo the prime: the profiles are empty when the matrix is zero.
        if !a.as_slice().iter().all(Number::is_zero) {
            return None;
        }
        return Some(Basis {
            rank: 0,
            cols: Vec::new(),
            free,
            rows: Vec::new(),
            others,
            minor: Matrix::zeros(0, 0),
            det: OnceCell::from(Number::from(1u8)),
            x: Matrix::zeros(0, n),
            n: rows_too.then(|| Matrix::zeros(m, 0)),
            hj: OnceCell::new(),
        });
    }
    let minor = select(a, &c.rows, &c.cols);
    // With every column in `J` there is nothing to solve for (and no elimination to spend on it).
    let x = if free.is_empty() {
        Matrix::zeros(r, 0)
    } else {
        minor
            .solve(&select(a, &c.rows, &free))
            .expect("a minor that is nonsingular modulo a prime")
    };
    // A column of `K` may depend only on the columns of `J` before it.
    for (f, &col) in free.iter().enumerate() {
        for (k, &p) in c.cols.iter().enumerate() {
            if p > col && !x[(k, f)].is_zero() {
                return None;
            }
        }
    }
    // Every row outside `R` must agree with `X`: then `A[:, K] = A[:, J]·X` and the rank is `r`.
    if !others.is_empty() && !free.is_empty() {
        let left = rationals(&select(a, &others, &c.cols));
        if &left * &x != rationals(&select(a, &others, &free)) {
            return None;
        }
    }
    let det = OnceCell::new();
    let n = if rows_too {
        // `N = δ·A[S, J]·M⁻¹`, of integers (`δ·M⁻¹` is the adjugate up to sign), from its
        // residues; it is zero where `Y` is.
        let n = if others.is_empty() {
            Matrix::zeros(0, r)
        } else {
            let delta = det.get_or_init(|| minor.det().abs());
            times_inverse(&select(a, &others, &c.cols), &minor, delta, delta)
        };
        // A row of `S` may depend only on the rows of `R` below it.
        for (s, &row) in others.iter().enumerate() {
            for (k, &p) in c.rows.iter().enumerate() {
                if p < row && !n[(s, k)].is_zero() {
                    return None;
                }
            }
        }
        Some(n)
    } else {
        None
    };
    Some(Basis {
        rank: r,
        cols: c.cols.clone(),
        free,
        rows: c.rows.clone(),
        others,
        minor,
        det,
        x,
        n,
        hj: OnceCell::new(),
    })
}

/// The profiles of the square `a` of order [`pernet::FROM`] or more when Pernet and Stein's method
/// finds its Hermite form of full rank, which proves it nonsingular: every column and every row,
/// the minor `a` itself, its determinant the product of the pivots, and the form; `None`
/// otherwise (a singular matrix, or one whose leading block is singular), for [`basis`].
pub(crate) fn square_basis(a: &Matrix<Number>) -> Option<Basis> {
    let n = a.rows();
    if !a.is_square() || n < pernet::FROM.max(3) {
        return None;
    }
    let h = pernet::square(a)?;
    let det: Number = (0..n).map(|i| h[i][i].clone()).product();
    Some(Basis {
        rank: n,
        cols: (0..n).collect(),
        free: Vec::new(),
        rows: (0..n).collect(),
        others: Vec::new(),
        minor: a.clone(),
        det: OnceCell::from(det),
        x: Matrix::zeros(n, 0),
        n: Some(Matrix::zeros(0, n)),
        hj: OnceCell::from(matrix_of(&h, n, n)),
    })
}

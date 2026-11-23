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
//! Both are found together by one elimination modulo a prime `p` below `2^62`: the rows from the
//! bottom, each reduced by the echelon basis of the rows kept so far, kept when something is left,
//! whose first element is then a new pivot column. That costs `O(m·n·r)` operations on words, and
//! gives the true profiles unless `p` divides one of a few minors. They are then proved exactly:
//! with `M = A[R, J]` nonsingular (its determinant is not zero modulo `p`), the solution
//! `X = M⁻¹·A[R, K]` for the columns `K` outside `J` must satisfy `A[S, J]·X = A[S, K]`, which
//! makes every column a combination of the columns `J` and the rank `r`, and must be zero where a
//! column of `J` comes after the column of `K` it expresses, which makes `J` the column profile;
//! when `R` is wanted too, `Y = A[S, J]·M⁻¹` must be zero where a row of `R` comes before the row
//! of `S` it expresses. When a check fails, the next prime is tried. Each prime that fails divides
//! one of the `m + n` minors the profiles depend on, so a matrix fails at most a number of primes
//! that is the length of those minors in bits over 61, and as a rule none.

use super::super::Matrix;
use crate::math::numbers::nat::inverse_word;
use crate::math::numbers::{Divisor, Number, is_prime_u64};
use crate::math::rational::Rational;

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
    /// `|det M|`, positive: 1 for rank 0.
    pub(crate) det: Number,
    /// `X = M⁻¹·A[R, K]`, `r × (n − r)`: `A[:, K] = A[:, J]·X`.
    pub(crate) x: Matrix<Rational>,
    /// `Y = A[S, J]·M⁻¹`, `(m − r) × r`: `A[S, :] = Y·A[R, :]`; when asked for.
    pub(crate) y: Option<Matrix<Rational>>,
}

/// The primes the profiles are computed modulo: the largest below `2^62`, then each the largest
/// below the one before.
struct Primes(u64);

impl Iterator for Primes {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        while self.0 > 3 {
            self.0 -= 2;
            if is_prime_u64(self.0) {
                return Some(self.0);
            }
        }
        None
    }
}

/// The rank profiles of `a`, proved; with the row rank profile from the bottom when `rows_too` is
/// set, and some `r` independent rows otherwise.
pub(crate) fn basis(a: &Matrix<Number>, rows_too: bool) -> Basis {
    // From `2^62 + 1`, so that the first prime tried is the largest below `2^62`.
    for p in Primes((1 << 62) + 1) {
        if let Some(b) = prove(a, &candidate(a, p), rows_too) {
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

/// `a·b mod p`.
#[inline]
fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(p)) as u64
}

/// The rank profiles of `a` modulo the prime `p`: the rows from the bottom, each reduced by the
/// echelon basis of the rows kept so far, its basis vectors in the order of their first columns,
/// each scaled to a 1 there; a row with something left is kept, and its first nonzero column is a
/// pivot.
fn candidate(a: &Matrix<Number>, p: u64) -> Candidate {
    let (m, n) = a.shape();
    let divisor = Divisor::new(&Number::from(p));
    let mut basis: Vec<(usize, Vec<u64>)> = Vec::new();
    let mut rows = Vec::new();
    let mut v = vec![0u64; n];
    for i in (0..m).rev() {
        if basis.len() == n {
            // The rows kept span everything: the others depend on them.
            break;
        }
        for (j, x) in v.iter_mut().enumerate() {
            let r = divisor.rem_euclid(&a[(i, j)]);
            *x = u64::try_from(&r).expect("a residue below the prime");
        }
        for (lead, b) in &basis {
            let c = v[*lead];
            if c == 0 {
                continue;
            }
            for (x, &y) in v[*lead..].iter_mut().zip(&b[*lead..]) {
                let t = mul_mod(c, y, p);
                *x = if *x >= t { *x - t } else { *x + (p - t) };
            }
        }
        if let Some(lead) = v.iter().position(|&x| x != 0) {
            let inv = inverse_word(v[lead], p).expect("a unit modulo a prime");
            for x in &mut v[lead..] {
                *x = mul_mod(*x, inv, p);
            }
            let at = basis.partition_point(|(l, _)| *l < lead);
            basis.insert(at, (lead, v.clone()));
            rows.push(i);
        }
    }
    rows.reverse();
    Candidate {
        cols: basis.into_iter().map(|(l, _)| l).collect(),
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
            det: Number::from(1u8),
            x: Matrix::zeros(0, n),
            y: rows_too.then(|| Matrix::zeros(m, 0)),
        });
    }
    let minor = select(a, &c.rows, &c.cols);
    let det = minor.det().abs();
    let x = minor
        .solve(&select(a, &c.rows, &free))
        .expect("a minor that is nonsingular modulo a prime");
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
    let y = if rows_too {
        let y = minor
            .transpose()
            .solve(&select(a, &others, &c.cols).transpose())
            .expect("a nonsingular minor")
            .transpose();
        // A row of `S` may depend only on the rows of `R` below it.
        for (s, &row) in others.iter().enumerate() {
            for (k, &p) in c.rows.iter().enumerate() {
                if p < row && !y[(s, k)].is_zero() {
                    return None;
                }
            }
        }
        Some(y)
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
        y,
    })
}

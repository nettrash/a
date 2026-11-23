//! Normal forms of matrices of integers: the Hermite normal form and the Smith normal form, with
//! their unimodular transformations.
//!
//! - The Hermite normal form of the rows is computed modulo the determinant `D` of a nonsingular
//!   minor of full rank, as Domich, Kannan and Trotter showed it can be (*Hermite normal form
//!   computation using modulo determinant arithmetic*, Math. Oper. Res. 12, 1987; Cohen, *A
//!   Course in Computational Algebraic Number Theory*, Algorithm 2.4.8), so that no element grows
//!   past `D`; the minor is that of the rank profiles, found modulo a prime and proved exactly
//!   ([`profile`]). The columns outside the profile and the transformation come from exact solutions
//!   with that minor ([`hermite`]).
//! - The diagonal of the Smith normal form is computed modulo the same `D` (Cohen, Algorithm
//!   2.4.14, for any shape and rank), and the form with its transformations by alternating
//!   Hermite forms of the rows and of the columns, as Kannan and Bachem do (*Polynomial algorithms
//!   for computing the Smith and Hermite normal forms of an integer matrix*, SIAM J. Comput. 8,
//!   1979), the divisibility of the diagonal then fixed by steps `diag(a, b) → diag(gcd, lcm)`
//!   ([`smith`]).

mod hermite;
mod profile;
mod smith;
mod transform;

use super::Matrix;
use crate::math::numbers::{ByteOrder, Number};
use crate::math::rational::in_order;

/// The matrix `m` in the byte order `order`.
fn ordered(m: Matrix<Number>, order: ByteOrder) -> Matrix<Number> {
    let (rows, cols) = m.shape();
    let data = m
        .into_vec()
        .into_iter()
        .map(|x| in_order(x, order))
        .collect();
    Matrix::from_column_major(rows, cols, data)
}

/// The byte order of the top left element of `a`, little-endian for an empty matrix.
fn order_of(a: &Matrix<Number>) -> ByteOrder {
    a.as_slice()
        .first()
        .map_or(ByteOrder::LittleEndian, Number::byte_order)
}

impl Matrix<Number> {
    /// The Hermite normal form `H` of the rows, with a unimodular `U` such that `U·self = H`: `H`
    /// is in row echelon form, its nonzero rows first, each pivot positive and the elements above
    /// a pivot in `[0, pivot)`; the zero rows, one for each dependency among the rows, last. The
    /// rows of `H` are the one basis of this form of the lattice the rows span, so `H` is unique;
    /// its pivots are in the columns of the column rank profile, the first columns from the left
    /// that are independent. This is FLINT's `fmpz_mat_hnf`. PARI's `mathnf` triangularises the
    /// columns from the other end: for `B`, `A` with its columns in the opposite order, `H`
    /// without its zero rows is `mathnf(Bᵀ)ᵀ` with the order of its rows and of its columns
    /// reversed.
    ///
    /// `U` is the one of the Hermite normal form of `[A | I]`, which is `[H | U]`: its rows from
    /// the rank `r` on are the Hermite normal form of the lattice of the left kernel
    /// `{u : u·A = 0}`, their pivots in the rows of `A` that are combinations of the rows below
    /// them, and above each such pivot the elements of `U` are in `[0, pivot)`. For a matrix of
    /// full row rank `U` is the only one.
    ///
    /// Method: the rank profiles (the pivot columns `J`, and the rows `R` from the bottom that
    /// are independent) modulo a prime below `2^62`, proved exactly by the solutions with the
    /// minor `M = A[R, J]` (another prime when the proof fails); the Hermite form of the rows of
    /// `A[:, J]` modulo `D = |det M|`, so that no element grows past `D` (Domich, Kannan and
    /// Trotter), and the other columns of `H` by an exact solution with `M`. `U` is built from the
    /// structure of `[A[:, J] | E_S]`, `[A | I]` on the columns of its pivots, `E_S` the columns of
    /// the identity for the rows `S` outside `R`: with `N = D·A[S, J]·M⁻¹`, its last `m − r` rows
    /// are the Hermite form of the lattice `{z : z·N ≡ 0 (mod D)}` on the columns `S`, whose
    /// pivots above 1 are at most `log₂ D`, found from the last row up with the subgroup of
    /// `(Z/D)^r` that the rows after each span; its first `r` rows are the solutions of
    /// `w·N ≡ D·H[:, J]·M⁻¹ (mod D)` those rows leave reduced. The work is about `m·r²`
    /// operations on numbers below `D` for `H` and `m·(r² + r·t + t²)` more for `U`, `t` the
    /// pivots above 1, besides the determinant of `M`, the solutions with it and the `m × m`
    /// result; `D` has at most `r·(b + ½·log₂ r)` bits for elements of `b` bits (Hadamard's
    /// bound). [`hermite_form`] skips `U`. The results are in the byte order of the top left
    /// element.
    ///
    /// [`hermite_form`]: Matrix::<Number>::hermite_form
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[2, 3, 5], [4, 6, 11], [6, 9, 16]]).map(|&x| Number::from(x));
    /// let (h, u) = a.hermite_normal_form();
    /// // The third row is the sum of the first two: one zero row.
    /// assert_eq!(h, Matrix::from_rows(&[[2, 3, 0], [0, 0, 1], [0, 0, 0]]).map(|&x| Number::from(x)));
    /// assert_eq!(&u * &a, h);
    /// assert_eq!(u.det().abs(), 1);
    /// ```
    #[must_use]
    pub fn hermite_normal_form(&self) -> (Matrix<Number>, Matrix<Number>) {
        let order = order_of(self);
        let basis = profile::basis(self, true);
        let (h, u) = hermite::form_and_transform(self, &basis);
        (ordered(h, order), ordered(u.into_matrix(false), order))
    }

    /// The Hermite normal form `H` of the rows, as
    /// [`hermite_normal_form`](Matrix::<Number>::hermite_normal_form) gives it, without the
    /// transformation: the Hermite form of the rows of `A[:, J]` modulo `D = |det A[R, J]|`, of `r`
    /// columns rather than `m`, about `m·r²` operations on numbers below `D`, and the other
    /// columns by one exact solution with `A[R, J]`. A square nonsingular matrix takes the
    /// elimination modulo a prime, its determinant and the form modulo it, and nothing else. The
    /// result is in the byte order of the top left element.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[3, 1], [1, 3]]).map(|&x| Number::from(x));
    /// // The lattice of index 8 = det in Z², with the vector (0, 8) and (1, 3).
    /// assert_eq!(a.hermite_form(), Matrix::from_rows(&[[1, 3], [0, 8]]).map(|&x| Number::from(x)));
    /// ```
    #[must_use]
    pub fn hermite_form(&self) -> Matrix<Number> {
        let order = order_of(self);
        let basis = profile::basis(self, false);
        ordered(hermite::form(self, &basis), order)
    }

    /// The Smith normal form `S`, with unimodular `U` and `V` such that `U·self·V = S`: `S` is
    /// diagonal, its diagonal elements `d₁, d₂, …` nonnegative, each dividing the next, the zeros
    /// last. `S` is unique (`dᵢ` is the quotient of the greatest common divisors of the minors of
    /// order `i` and `i − 1`); `U` and `V` are not. The ones given start as the transformation
    /// `U₁` of [`hermite_normal_form`](Matrix::<Number>::hermite_normal_form) and the transpose
    /// `V₁` of a transformation of the Hermite form of the columns of its nonzero rows `H[..r]`
    /// (the one of the form of `[H[..r]ᵀ | E_K]`, `K` the columns of `H` outside its pivots and
    /// `E_K` the columns of the identity for them), which leave a lower triangular matrix of order
    /// `r`; only their first `r` rows (of `U`) and columns (of `V`) change after that. That matrix
    /// is put in its Hermite form by rows and by columns in turn until it is diagonal (Kannan and
    /// Bachem), each form computed modulo its determinant with the unique transformation of a
    /// nonsingular matrix; then steps `diag(a, b) → diag(gcd(a, b), lcm(a, b))` make each element
    /// divide the next, in the order of Batcher's odd-even merge sort, which takes each position
    /// through `O(log² r)` steps, with the larger of each step's two factors on the side, `U` or
    /// `V`, whose two lines are the shorter; last, the first rows of `U` are reduced modulo its
    /// last `m − r` rows, a basis of the left kernel `{u : u·A = 0}` in Hermite form, and the
    /// first columns of `V` modulo its last `n − r` columns, a basis of the kernel
    /// `{v : A·v = 0}`, each element in the line of a pivot of theirs into `[0, pivot)`. No bound
    /// on the lengths of `U` and `V` is proved. Measured against Hadamard's bound
    /// `k·(b + ½·log₂ k)` bits, `k = min(m, n)` and `b` the length of the elements: at most 2.12
    /// times on 1800 random matrices of every kind up to `32 × 32`; on diagonal matrices at most
    /// 1.08 times for the first 40 and the first 100 primes in either order, 1.9 for 40 random
    /// numbers of 100 bits and 1.7 for 100, 1.75 for the 100 largest primes below `2^62`; for 100
    /// random numbers of 100 bits on the diagonal with small ones beside it 1.7 to 1.9 on five
    /// matrices and 2.36 on another. The longest known are those of matrices `U₀·diag(d)·V₀`
    /// built to have a given Smith form, `d` random odd numbers and `U₀`, `V₀` the identity
    /// through `2k` random additions of −2 to 2 times a row (a column) to another and a
    /// permutation: 2.25 to 2.74 times for `50 × 50` to `150 × 150` with 100- and 300-bit `d`,
    /// and 3.43 times on a `100 × 100` with elements of 306 bits. PARI's `matsnf(A, 1)` gives
    /// transformations about 40 times as long on such a `50 × 50` matrix. The transformation of
    /// the Hermite form stayed within the bound on all these matrices. This is FLINT's `fmpz_mat_snf`; PARI's
    /// `matsnf` gives the diagonal in the opposite order. The results are in the byte order of
    /// the top left element.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[2, 4, 4], [-6, 6, 12], [10, -4, -16]]).map(|&x| Number::from(x));
    /// let (s, u, v) = a.smith_normal_form();
    /// assert_eq!(s, Matrix::from_diagonal(&[2, 6, 12].map(Number::from)));
    /// assert_eq!(&(&u * &a) * &v, s);
    /// assert_eq!((u.det().abs(), v.det().abs()), (Number::from(1), Number::from(1)));
    /// ```
    #[must_use]
    pub fn smith_normal_form(&self) -> (Matrix<Number>, Matrix<Number>, Matrix<Number>) {
        let order = order_of(self);
        let basis = profile::basis(self, true);
        let (s, u, v) = smith::form(self, &basis);
        (ordered(s, order), ordered(u, order), ordered(v, order))
    }

    /// The diagonal of the Smith normal form, the elementary divisors `d₁ | d₂ | …`, as many as
    /// the smaller dimension, the zeros last; without the transformations. The rank `r` and the
    /// nonsingular minor `M` of the rank profiles come as for the Hermite form; the product
    /// `d₁⋯d_r` divides `D = |det M|`, so the Smith form over `Z/DZ` gives the divisors, each
    /// position's pivot made to divide the rest of its block and the modulus then divided by it
    /// (Cohen, Algorithm 2.4.14), every element below `D`: about `m·n·r` operations on numbers as
    /// long as `D`. The results are in the byte order of the top left element.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// // Z³ / (rows) is Z/2 × Z/6 × Z/12.
    /// let a = Matrix::from_rows(&[[2, 4, 4], [-6, 6, 12], [10, -4, -16]]).map(|&x| Number::from(x));
    /// assert_eq!(a.elementary_divisors(), [2, 6, 12].map(Number::from));
    /// // Rank 1: one divisor and a zero.
    /// let b = Matrix::from_rows(&[[2, 4], [3, 6], [5, 10]]).map(|&x| Number::from(x));
    /// assert_eq!(b.elementary_divisors(), [1, 0].map(Number::from));
    /// ```
    #[must_use]
    pub fn elementary_divisors(&self) -> Vec<Number> {
        let order = order_of(self);
        let basis = profile::basis(self, false);
        smith::divisors(self, &basis)
            .into_iter()
            .map(|x| in_order(x, order))
            .collect()
    }
}

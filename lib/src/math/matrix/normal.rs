//! Normal forms of matrices of integers: the Hermite normal form and the Smith normal form, with
//! their unimodular transformations.
//!
//! - The Hermite normal form of the rows starts from a nonsingular minor `M` of full rank, that of
//!   the rank profiles, found modulo a prime and proved exactly ([`profile`]). The form of `M` is
//!   Pernet and Stein's (*Fast computation of Hermite normal forms of random integer matrices*, J.
//!   Number Theory 130, 2010): two determinants of order `r − 1` and a solution by Dixon's
//!   lifting, a form modulo their greatest common divisor, and two rows added ([`pernet`]); the
//!   other rows are added to it. Small minors, and those the method does not apply to, take the
//!   form modulo `D = |det M|`, as Domich, Kannan and Trotter showed it can be computed (*Hermite
//!   normal form computation using modulo determinant arithmetic*, Math. Oper. Res. 12, 1987;
//!   Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 2.4.8), so that no
//!   element grows past `D`, in words when `D` is below `2^63` ([`word`]). The columns outside the
//!   profile and the transformation come from exact solutions with that minor ([`hermite`]), the
//!   products by `M⁻¹` from their residues modulo primes ([`inverse`]).
//! - The diagonal of the Smith normal form comes from the Hermite form: a pivot 1 gives a divisor
//!   1, and the block of the other pivots' rows has the other divisors, computed modulo the
//!   product of its pivots (Cohen, Algorithm 2.4.14, for any shape and rank). The form with its
//!   transformations is computed by alternating Hermite forms of the rows and of the columns, as
//!   Kannan and Bachem do (*Polynomial algorithms for computing the Smith and Hermite normal forms
//!   of an integer matrix*, SIAM J. Comput. 8, 1979), the divisibility of the diagonal then fixed
//!   by steps `diag(a, b) → diag(gcd, lcm)` ([`smith`]).

mod hermite;
mod inverse;
mod pernet;
mod profile;
mod smith;
mod transform;
mod word;

use super::Matrix;
use crate::math::numbers::{ByteOrder, Number};
use crate::math::rational::in_order;

/// `x` in the byte order `order`, a zero as the empty number of that order, whichever way the
/// arithmetic made it.
fn canonical(x: Number, order: ByteOrder) -> Number {
    if x.is_zero() {
        in_order(Number::default(), order)
    } else {
        in_order(x, order)
    }
}

/// The matrix `m` in the byte order `order`, its zeros the empty number.
fn ordered(m: Matrix<Number>, order: ByteOrder) -> Matrix<Number> {
    let (rows, cols) = m.shape();
    let data = m
        .into_vec()
        .into_iter()
        .map(|x| canonical(x, order))
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
    /// are independent) modulo a prime below `2^62` by PLE decompositions, proved exactly by the
    /// solutions with the minor `M = A[R, J]` (another prime when the proof fails). The form of
    /// `M`, from order 8, by Pernet and Stein's method: for `W₁` its leading block of order
    /// `r − 1` and `W₂` that block with the last row of `M` for the one before, `d₁ = det W₁`,
    /// `d₂ = det W₂` and the solution `W₁⁻¹·[w | e]` (`w` the last column, `e` a unit vector)
    /// from one lifting of Dixon's and the residues of `d₁` modulo a prime or two; the block `W` with `s·c + t·d` for its last row, `c` and `d` the last two rows and
    /// `g = s·d₁ + t·d₂ = gcd(d₁, d₂)`, has the determinant `g`, small as a rule, so its form is
    /// computed modulo `g`, the last column by Sherman and Morrison's formula from the solution;
    /// then the last two rows of `M` are added, by extended Euclidean steps against the pivots.
    /// The rows of `A[:, J]` outside `R` are added to that form in the same way, every element
    /// kept below `D = |det M|`, the product of its pivots. A smaller minor, or one whose leading
    /// block is singular, takes the form of the rows of `A[:, J]` modulo `D` (Domich, Kannan and
    /// Trotter), in words when `D` is below `2^63`. The other columns of `H` come from an exact
    /// solution with `M`. `U` is built from the structure of `[A[:, J] | E_S]`, `[A | I]` on the
    /// columns of its pivots, `E_S` the columns of the identity for the rows `S` outside `R`: with
    /// `N = D·A[S, J]·M⁻¹`, its last `m − r` rows are the Hermite form of the lattice
    /// `{z : z·N ≡ 0 (mod D)}` on the columns `S`, whose pivots above 1 are at most `log₂ D`,
    /// found from the last row up with the subgroup of `(Z/D)^r` that the rows after each span;
    /// its first `r` rows are the solutions of `w·N ≡ D·H[:, J]·M⁻¹ (mod D)` those rows leave
    /// reduced, `N` and `D·H[:, J]·M⁻¹` (`H·M⁻¹` itself when every row is in `R`) computed from
    /// their residues modulo word primes, as many as Hadamard's bound on the adjugate asks for. For a
    /// random square matrix the form is the identity but for its last column and the work is
    /// about that of one solution of order `n − 1`: `O(n³)` operations on words and `O(n²·L)`
    /// for a lifting of `L` steps, `L` about the length of `D` over 61 bits; `U` then costs
    /// `O(n^ω)` operations on words for each of about as many primes. The form modulo `D` is about
    /// `m·r²` operations on numbers below `D`, and the general `U` `m·(r² + r·t + t²)` more, `t`
    /// the pivots above 1, besides the solutions with `M` and the `m × m` result; `D` has at
    /// most `r·(b + ½·log₂ r)` bits for elements of `b` bits (Hadamard's bound).
    /// [`hermite_form`] skips `U`. The results are in the byte order of the top left element,
    /// each zero the empty number.
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
        let basis = profile::square_basis(self).unwrap_or_else(|| profile::basis(self, true));
        let (h, u) = hermite::form_and_transform(self, &basis);
        (ordered(h, order), ordered(u.into_matrix(false), order))
    }

    /// The Hermite normal form `H` of the rows, as
    /// [`hermite_normal_form`](Matrix::<Number>::hermite_normal_form) gives it, without the
    /// transformation: the form of the minor `A[R, J]` by Pernet and Stein's method and the other
    /// rows of `A[:, J]` added to it (or the form of the rows of `A[:, J]` modulo
    /// `D = |det A[R, J]|`), and the other columns by one exact solution with `A[R, J]`. A square
    /// matrix of order 8 or more takes the method first, with no profile: a form of full rank
    /// proves it nonsingular, its profiles every row and column; for a random one, about the
    /// work of one solution of a system of order `n − 1`. The result
    /// is in the byte order of the top left element, each zero the empty number.
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
        let basis = profile::square_basis(self).unwrap_or_else(|| profile::basis(self, false));
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
    /// Bachem), each form computed as [`hermite_form`](Matrix::<Number>::hermite_form) computes
    /// that of a square nonsingular matrix, with the unique transformation `H·B⁻¹` from its
    /// residues modulo primes; then steps `diag(a, b) → diag(gcd(a, b), lcm(a, b))` make each element
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
    /// the top left element, each zero the empty number.
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
        let basis = profile::square_basis(self).unwrap_or_else(|| profile::basis(self, true));
        let (s, u, v) = smith::form(self, &basis);
        (ordered(s, order), ordered(u, order), ordered(v, order))
    }

    /// The diagonal of the Smith normal form, the elementary divisors `d₁ | d₂ | …`, as many as
    /// the smaller dimension, the zeros last; without the transformations. They are those of the
    /// Hermite form `H` of [`hermite_form`](Matrix::<Number>::hermite_form): a row of `H` whose
    /// pivot is 1 gives a divisor 1, as the pivot is alone in its column (the elements above it
    /// are in `[0, 1)`) and the column steps that clear the rest of its row change nothing else;
    /// the rows of the other pivots are zero in those columns, and their block, on their pivots'
    /// columns and the columns outside the profile, has the other divisors, whose product divides
    /// the product `P` of its pivots. So the Smith form of that block over `Z/PZ` gives them, each
    /// position's pivot made to divide the rest of its block and the modulus then divided by it
    /// (Cohen, Algorithm 2.4.14). For a random square matrix `H` is the identity but for its last
    /// column, the divisors are `1, …, 1, |det A|`, and the work is that of the Hermite form;
    /// otherwise add about `t²·(t + k)` operations on numbers below `P`, for `t` pivots above 1
    /// and `k` columns outside the profile; a wide matrix is taken by its transpose, which has the
    /// same divisors and fewer columns (none outside the profile at full rank). The results are in
    /// the byte order of the top left element, each zero the empty number.
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
        // A matrix and its transpose have the same divisors; a wide one is taken by its columns,
        // which spares the solution for the columns outside the profile.
        let t;
        let a = if self.cols() > self.rows() {
            t = self.transpose();
            &t
        } else {
            self
        };
        let basis = profile::square_basis(a).unwrap_or_else(|| profile::basis(a, false));
        smith::divisors(a, &basis)
            .into_iter()
            .map(|x| canonical(x, order))
            .collect()
    }
}

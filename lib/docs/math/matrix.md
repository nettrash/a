[A](../../../README.md) › [Math](../../../README.md#math)

# Matrices

`Matrix<T>` (in `a::math::matrix`) is a rectangular array of elements of the type `T`, held column after column in one vector, of any size chosen when it is built. `T` is one of six types, the sealed trait `Element`:

- `f32` and `f64`, and [`Complex<f32>` and `Complex<f64>`](complex.md): the trait `Scalar`, with every operation below. `T::Real` is the float type of their parts, of norms, singular values and the eigenvalues of Hermitian matrices.
- [`Number`](numbers.md), an integer of any size: the arithmetic, the transposes, the row and column operations, the trace and powers, and, exactly, the determinant, the rank, the characteristic polynomial, the adjugate, the inverse as the adjugate and the determinant or as rationals, the solutions over the rationals, and the null space.
- [`Rational`](rational.md), a fraction of numbers in lowest terms: the same arithmetic, and, exactly, the determinant, the rank, the solutions, the inverse and the null space.

## Building and reading

- `Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]])` and `from_columns` take rows or columns of anything that is a slice; `from_row_major(rows, cols, vec)` and `from_column_major(rows, cols, vec)` a vector, the second without copying; `from_fn(rows, cols, |i, j| …)` a function of the indices. `zeros(rows, cols)`, `identity(n)` and `from_diagonal(&d)` need the element's 0 and 1. A shape that doesn't fit panics.
- `rows()`, `cols()`, `shape()`, `is_square()` and `is_empty()`; `a[(i, j)]` reads and writes an element and panics outside, `get(i, j)` and `get_mut(i, j)` give `None` there; `column(j)` is a slice, `row(i)` an iterator; `as_slice()`, `as_mut_slice()` and `into_vec()` the elements column after column; `diagonal()`; `map(|x| …)` a matrix of another type, as `a.map(|&x| Complex::from(x))` makes a real matrix complex.
- Matrices compare with `==`, clone, hash when their elements do, and default to `0 × 0`. `Display` writes the rows on lines of their own, each column right-aligned, two spaces apart, with the precision given; `Debug` writes them as lists, `[[1.0, 2.0], [3.0, 4.0]]`.

## Arithmetic and transformations

- `+`, `-` and their assignments add and subtract matrices of the same shape, elementwise; `-a` negates; `*` multiplies matrices whose inner dimensions agree, and a matrix by an element on either side, `&a * 2.0` and `2.0 * &a`; `/` divides by an element, for floats, complex numbers and rationals. Operands may be owned or borrowed; an owned left operand, or the target of an assignment, keeps its vector. Shapes that don't fit panic.
- `a.hadamard(&b)` is the product elementwise `A ∘ B`; `transpose()` is `Aᵀ` and `conjugate_transpose()` is `A*`, the transpose for real elements.
- `swap_rows(i, j)`, `scale_row(i, &k)` and `add_scaled_row(target, source, &k)`, which adds `k` times row `source` to row `target`, are the elementary row operations, in place; `swap_columns`, `scale_column` and `add_scaled_column` the column ones.
- `trace()` is the sum of the diagonal; `pow(n)` is `Aⁿ` for `n: u32` by repeated squaring, `A⁰ = I`; `powi(n)` takes an `i32`, a negative power of the inverse, and `None` when that doesn't exist.

## Decompositions

Each is a type of its own that gives its factors as matrices:

- `lu()`: `P·A = L·U` by Gaussian elimination with partial pivoting, for any shape: `l()`, `u()`, `p()`, `permutation()`, `det()`, `solve(&b)`, `inverse()` and `is_singular()`.
- `qr()`: `A = Q·R` by Householder reflections, for any shape: `q()` with orthonormal columns, `q_full()` square, `r()` with a real diagonal, and `solve(&b)`, the least squares for at least as many rows as columns.
- `cholesky()`: `A = L·L*` for a Hermitian positive definite matrix, from its lower triangle, or `None`: `l()`, `det()`, `solve(&b)`, `inverse()`.
- `svd()`: `A = U·Σ·V*`, thin: `u()`, `singular_values()`, descending, and `v()`.
- `schur()`: `A = Q·T·Q*` with `Q` unitary: `T` upper triangular for complex elements, and for real ones the real Schur form, with a `2 × 2` block `[[a, b], [c, a]]`, `b·c < 0`, for each pair of complex conjugate eigenvalues; `q()`, `t()` and `eigenvalues()`.
- `eigen()`: the eigenvalues and the eigenvectors, complex, each of length 1, with `A·V = V·Λ`; for a Hermitian matrix `hermitian_eigen()` gives real eigenvalues in ascending order and orthonormal eigenvectors of the matrix's own type, with `A = V·Λ·V*`.

## Properties

- `det()`, `inverse()` (`None` when singular) and `solve(&b)` from the LU decomposition.
- `rank()`, the number of singular values above `max(m, n)·ε·σ₁` as NumPy's `matrix_rank`, and `null_space()`, an orthonormal basis of `{x : A·x = 0}` as columns.
- `eigenvalues()` and `hermitian_eigenvalues()` without the eigenvectors, and `singular_values()` without `U` and `V`, for less work.
- `characteristic_polynomial()`, the coefficients of `det(x·I − A)` from `xⁿ` down.
- `norm(Norm::…)`: `One`, the largest column sum of absolute values; `Infinity`, the largest row sum; `Frobenius`; `Max`, the largest absolute value; `Two`, the largest singular value; `Nuclear`, their sum. `cond(Norm::…)` is `‖A‖·‖A⁻¹‖`, infinite for a singular matrix; in the 2-norm `σ₁/σₙ`, for any shape.

## Functions of matrices

- `exp()` is `e^A`, `sqrt()` the principal square root, whose eigenvalues have positive real parts, and `log()` the principal logarithm, whose eigenvalues have imaginary parts in `(−π, π]`. `sqrt` and `log` give `None` for a real matrix whose square root or logarithm is not real, with a negative real eigenvalue, and for a singular matrix that has none: a real matrix can be made complex first.
- `pow` and `powi` are the integer powers.

A matrix with a NaN or an infinity gives NaNs, or `None` from `sqrt` and `log`.

## Exactly, on numbers and rationals

For `Matrix<Number>`, `det()` is Bareiss's fraction-free elimination, `rank()` its number of pivots, `characteristic_polynomial()` Berkowitz's method without a division, `adjugate()` `adj(A)` with `A·adj(A) = det(A)·I` also for singular matrices, `inverse()` the pair `(adj(A), det(A))`, `None` for a zero determinant, and `null_space()` a basis of integer vectors without common factors. `solve(&b)` is the solution `X` of `A·X = B` as a `Matrix<Rational>`, and `inverse_rational()` the inverse as one, each `None` for a singular matrix.

For `Matrix<Rational>`, `det()`, `rank()`, `solve(&b)`, `inverse()` and `null_space()` are exact: the determinant as a rational, `None` from `solve` and `inverse` for a singular matrix, and the null space as the basis of the reduced row echelon form, a 1 in each column without a pivot and 0s in the others. Each matrix of rationals is first made one of integers, row by row, so that no fraction is formed inside the elimination. The results are in the byte order of the top left element, little-endian for an empty matrix. A matrix that is not square panics in `det`, `solve` and `inverse`, and so does a right-hand side with a different number of rows.

## Algorithms

The floating-point algorithms are LAPACK's, arranged so that most of their work is products of blocks, and computed in the elements' own precision; real matrices stay in real arithmetic, also when their eigenvalues are complex.

- Products, `C = β·C + α·op(A)·op(B)` with `op` the matrix, its transpose or its conjugate transpose: Goto and van de Geijn's method as BLIS arranges it. Blocks of `op(B)` of 512 rows and 1020 columns (2040 for `f32`) and of `op(A)` of 128 rows are packed into panels that stay in the caches, transposed or conjugated as they are packed, and multiplied by kernels of 8 × 6 elements in registers for `f64` and 8 × 12 for `f32`, with narrower ones for a block's last columns. Complex products run on the real kernels by Van Zee's 1m method, at their speed. Products of up to 64³ whose operands are close together in memory skip the packing, on tiles of registers; products of up to 32 rows copy `op(A)` and read `op(B)` where it is. The kernels are NEON on little-endian aarch64, and portable ones of the same shapes elsewhere.
- Triangular solves: the triangle halved down to 16 rows, so that most of the work is the products of the blocks off the diagonal. A leaf copies its triangle to the stack scaled by the reciprocals of its diagonal, so that the substitution is fused multiply-adds alone, with NEON kernels for `f32` and `f64`; a triangle whose diagonal has a zero, a subnormal or a non-finite element, and columns whose results are not all finite, are solved by plain substitution with divisions, so that infinities and NaNs reach only the elements they would.
- `lu()`: partial pivoting, the pivot the first element of the largest `|re| + |im|` in its column, as LAPACK chooses it, and a division rather than a product by its reciprocal when that would overflow, as LAPACK's `getf2` does. Up to 16 × 16 on the stack in unrolled loops, the pivots chosen and the rows swapped by selects rather than branches; up to 24 columns a column at a time; larger matrices by Toledo's recursion, the left half, then the right half updated by it by a triangular solve and a product, down to panels of 16 columns, the row swaps of each half applied to the other a column at a time.
- `inverse()`: up to 12 × 12 on the stack in unrolled loops; larger matrices as LAPACK's `getri` does, `U⁻¹`, then `X·L = U⁻¹` by one triangular solve, then the columns swapped back, `2n³` operations in all.
- `cholesky()`: up to 7 × 7 (4 × 4 for complex elements) unrolled; up to 32 × 32 (16 × 16) column by column, each column in registers for `f32` and `f64`; larger matrices halved, `L11`, then `L21 = A21·L11⁻*` by a triangular solve, then `A22 − L21·L21*` on its lower triangle alone, in blocks of 64 columns.
- `qr()`: Householder reflections. A matrix of up to 4096 elements one reflection at a time (`geqr2`), each applied to the columns after it four at a time; a larger one in blocks of 8 to 32 columns chosen by its shape, each block's panel halved down to 8 columns as `geqrt3` does, so that the panel's own work is products too, its reflections gathered as `I − V·T·V*` (the compact WY form) and applied to the columns after it by three products. The factors `T` are kept: `q()`, `q_full()` and `solve(&b)` apply them by blocks, and the whole `Q` of a matrix of at least 4 times as many rows as columns is formed as one block, in half the work.
- `hermitian_eigen()`: a reduction to a real symmetric tridiagonal matrix by reflections in panels of 32 columns, whose trailing updates are products of rank 64 (`hetrd`, `latrd`); the tridiagonal matrix's eigenvectors by Cuppen's divide and conquer (`stedc`), with LAPACK's deflations, the secular equations solved as `laed4` does and Gu and Eisenstat's recomputed vectors, so that the eigenvectors are orthogonal, and the implicit QL algorithm with Wilkinson's shifts below order 25; the reflections applied back in blocks of 32 by products. Up to 80 to 160 rows by the type, where that machinery costs more than it saves, the reduction goes one column at a time, its reflections are multiplied out into `Q`, two at a time, and the implicit QL algorithm's rotations are applied to `Q`. `hermitian_eigenvalues()`: the same reduction, then Pal, Walker and Kahan's QL and QR without square roots (`sterf`).
- `svd()`: a reduction to a real bidiagonal matrix by reflections from both sides in panels of 16 columns, whose trailing updates are two products (`gebrd`, `labrd`), after a QR decomposition for a matrix of at least 2.5 times as many rows as columns; the bidiagonal matrix's vectors by divide and conquer (`bdsdc`), with the implicit QR algorithm below order 25; the reflections applied back in blocks of 32. `singular_values()`: the same reduction, from 1.75 times as many rows as columns after a QR decomposition, then dqds (`lasq1`–`lasq6`), which computes the singular values of the bidiagonal matrix to high relative accuracy. Up to 88 to 112 columns by the type (to 96 to 192 for the values alone), the reduction goes one column at a time, each right reflection applied together with the next left one, `U` and `V` are multiplied out, and the bidiagonal matrix takes Golub and Kahan's implicit QR algorithm with its rotations applied to them. A wide matrix is decomposed as its conjugate transpose.
- `schur()`, `eigenvalues()` and `eigen()`: a reduction to Hessenberg form in panels of 32 columns (`gehrd`, `lahr2`), then the QR algorithm. From order 120 (90 for complex elements with the vectors), Braman, Byers and Mathias's sweeps of many small bulges at once, with aggressive early deflation (`laqr0`–`laqr5`), the bulges' reflections gathered and applied to the rest of the matrix by products; below that, the double-shift QR algorithm for real matrices and the single-shift one for complex matrices (`lahqr`), and the real 2 × 2 blocks standardized as `lanv2` does. The eigenvectors by back substitution on the triangular or quasi-triangular form, 64 at a time, with the work off the diagonal as products, and a vector that would overflow solved again alone with LAPACK's scaling (`trevc3`); then `Q·X`.
- `hermitian_eigen()`, `svd()` and their value-only forms first scale a matrix whose largest element is near overflow or underflow, as LAPACK's drivers do.
- `rank()`, `null_space()`, `norm(Norm::Two)`, `norm(Norm::Nuclear)` and `cond(Norm::Two)` from the singular values (and `V` for the null space); `cond` in the other norms from the inverse; `characteristic_polynomial()` by La Budde's recurrence on the Hessenberg form, about `n³` operations.
- `exp()`: scaling and squaring, Higham's `expm` (2005): `A / 2^s` with a Padé approximant of degree 3, 5, 7, 9 or 13 chosen by its 1-norm (3, 5 or 7 in single precision), squared `s` times. `sqrt()`: the Schur method (Björck and Hammarling), `R² = T` solved element by element for the triangular `R`, then `Q·R·Q*`. `log()`: inverse scaling and squaring, Higham's `logm` (2001): square roots of the Schur form until it is within 0.264 of `I`, a Padé approximant of degree 7 in partial fractions by Gauss and Legendre's rule of 7 points, times `2^k`, with Al-Mohy and Higham's accurate diagonal. A real matrix's square root and logarithm are computed on its complex Schur form, and are real when they exist.
- The product of matrices of numbers is the sums of the products of their elements. That of matrices of rationals clears the rows of `A` and the columns of `B` of their denominators, multiplying them by the least common multiples `l_i` and `r_j`, multiplies the integers, in 64-bit registers when the lengths of the elements and of the sums allow it, in 128 bits when they allow that, and then divides each element once, by `l_i·r_j`, with one greatest common divisor: about `n³` products of integers and `n²` greatest common divisors, where the products of rationals would take a gcd or two each. A product whose inner dimension is 2, or 4 with denominators of more than 32 bits, is the sums of the products of rationals, which are done in registers when their parts fit.
- On `Number`: Bareiss's elimination, `a_ij ← (a_ij·a_kk − a_ik·a_kj) / p` with the previous pivot `p` dividing exactly, takes about `n³/3` steps, each two products and an exact division of numbers no longer than the determinant, the minors of the matrix; Berkowitz's method about `n⁴/2` products; the adjugate is `(−1)ⁿ⁻¹·(Aⁿ⁻¹ + c₁·Aⁿ⁻² + … + cₙ₋₁·I)` by Cayley and Hamilton, from the characteristic polynomial's coefficients `c`, for singular matrices too; the null space comes from the fraction-free echelon form, one vector for each column without a pivot.
- Solutions, on `Number` and `Rational`: Bareiss's elimination of `[A | B]` to an echelon form whose last pivot is `D = ±det(A)`, then a fraction-free back substitution, `y_i = (D·b_i − Σ u_ij·y_j) / u_ii` from the bottom, whose integers `y = D·x` are Cramer's numerators, every division exact; then one reduction of each element `y_i / D` by a greatest common divisor. The inverse is the solution of `A·X = I`, whose integers `D·A⁻¹` are also the adjugate up to the sign of the row swaps, which `inverse()` of numbers returns. A matrix of rationals has each row multiplied by the least common multiple `l_i` of its denominators (of `A` and `B` together for `solve`), which leaves the solutions and the null space as they were and multiplies the determinant by `∏ l_i`; `A⁻¹` is then `(L·A)⁻¹·L`.
- The integers of the elimination are held in registers while their magnitudes are below `2^63`: a step is then two products in 128 bits, a subtraction, and an exact division by a product with the inverse of the divisor's odd part modulo `2^64`, a comparison telling whether the quotient fits. Beyond, each element is a vector of limbs that the steps write over, the products go to two buffers of working memory, and the exact quotient is Hensel's, from the low end; from quotients of 150 limbs it is a product by the inverse of the divisor modulo a power of `2^64`, computed once for the elements that the same pivot divides. A step then allocates nothing once the vectors have grown.
- Large systems, on `Number` and `Rational`: from 24 rows (16 when an integer of the cleared matrix passes 63 bits), a system with at most an eighth as many right-hand sides as rows is solved by Dixon's p-adic lifting. `A` is factored once modulo a prime `p` just below `2^62`, in Montgomery's form, the products of the substitutions summed four at a time before a reduction; each step solves `A·x ≡ r (mod p)` and lifts the right-hand side, `r ← (r − A·x) / p`, exactly, the rows of `A` multiplied by the digits limb by limb; after `L` steps, `p^L` past twice the product of Hadamard's bounds on Cramer's numerators and on the determinant, the digits are joined by halves into `X mod p^L`, and each element is the one fraction within the bounds that is congruent to it, found by the extended Euclidean algorithm (Wang's rational reconstruction) with Lehmer's steps in registers, as a rule for the first element alone: the others are their product by the common denominator so far, when that is within the bounds. A matrix singular modulo the prime goes to the next one, and after four to Bareiss's elimination. The work is `O(n³ + L·n²)` operations on words instead of `n³/3` products of numbers as long as the minors.
- Large determinants: from 28 rows (22 when an integer passes 63 bits), the least common multiple `l` of the denominators of the solution of `A·x = b` for a fixed pseudorandom `b` by the lifting, which divides `det(A)` and is most of it as a rule, then `det(A) / l` from the determinants modulo primes, the product of the factors' diagonal, by the Chinese remainder theorem, until the primes' product passes twice Hadamard's bound over `l`: one prime, as a rule. This is FLINT's `fmpz_mat_det_modular_accelerated`.

## Measurements

Measured on an Apple M4 Max, one core, for `f64`, against faer 0.24.4 (`Par::Seq`), nalgebra 0.35.0 and Apple's Accelerate (`VECLIB_MAXIMUM_THREADS=1`: `dgemm`, and LAPACK's `dgetrf`, `dgetri`, `dpotrf`, `dgeqrf`, `dgesdd`, `dsyevd` and `dgeev`), on the same square matrices of random elements uniform in `[−1, 1)`, `A·Aᵀ + n·I` for `cholesky()` and the Hermitian eigenvalues (the median of up to 200 calls in 0.4 s, in µs). The others' counterparts: faer's `partial_piv_lu`, `partial_piv_lu().inverse()`, `llt`, `qr`, `thin_svd`, `singular_values`, `self_adjoint_eigen`, `self_adjoint_eigenvalues`, `eigen` and `eigenvalues`, and nalgebra's `lu`, `try_inverse`, `cholesky`, `qr`, `svd(true, true)`, `singular_values`, `symmetric_eigen` and `exp`; "–" where a library has no such function, or where it was not measured here.

| Operation | n | `Matrix` | faer | nalgebra | Accelerate |
| --- | --- | --- | --- | --- | --- |
| `a * b` | 8 | 0.04 | 0.08 | 0.08 | 0.08 |
| `a * b` | 32 | 1.08 | 1.33 | 1.21 | 0.67 |
| `a * b` | 128 | 74.0 | 80.7 | 76.2 | 12.2 |
| `a * b` | 512 | 4,345 | 4,792 | 4,627 | 612 |
| `a * b` | 1024 | 34,894 | 38,375 | 37,131 | 4,985 |
| `lu()` | 8 | 0.12 | 0.38 | 0.17 | 0.25 |
| `lu()` | 32 | 1.83 | 2.79 | 2.50 | 2.96 |
| `lu()` | 128 | 48.0 | 67.8 | 96.1 | 38.7 |
| `lu()` | 512 | 1,930 | 2,509 | 7,956 | 828 |
| `lu()` | 1024 | 13,776 | 17,180 | 59,943 | 4,062 |
| `inverse()` | 8 | 0.21 | 0.83 | 0.50 | 0.58 |
| `inverse()` | 32 | 4.25 | 8.83 | 13.3 | 7.54 |
| `inverse()` | 128 | 124 | 174 | 423 | 75.5 |
| `inverse()` | 512 | 5,414 | 7,005 | 29,490 | 1,715 |
| `inverse()` | 1024 | 40,098 | 50,225 | 220,728 | 9,229 |
| `cholesky()` | 8 | 0.08 | 0.21 | 0.08 | 0.04 |
| `cholesky()` | 32 | 0.96 | 1.29 | 1.75 | 0.50 |
| `cholesky()` | 128 | 27.3 | 37.5 | 53.9 | 12.1 |
| `cholesky()` | 512 | 1,056 | 1,217 | 3,560 | 200 |
| `cholesky()` | 1024 | 7,388 | 8,469 | 27,834 | 1,153 |
| `qr()` | 8 | 0.21 | 0.46 | 0.29 | 0.42 |
| `qr()` | 32 | 2.46 | 6.71 | 4.12 | 3.88 |
| `qr()` | 128 | 78.8 | 140 | 169 | 105 |
| `qr()` | 512 | 3,929 | 4,601 | 12,106 | 1,996 |
| `qr()` | 1024 | 28,977 | 32,458 | 90,739 | 10,655 |
| `svd()` | 8 | 2.08 | 4.08 | 2.83 | 5.50 |
| `svd()` | 32 | 33.8 | 75.0 | 54.1 | 76.3 |
| `svd()` | 128 | 1,055 | 1,336 | 2,471 | 1,070 |
| `svd()` | 512 | 35,029 | 48,891 | 663,666 | 20,749 |
| `svd()` | 1024 | 243,916 | 330,782 | 5,885,939 | 103,470 |
| `singular_values()` | 8 | 1.67 | 3.50 | 2.29 | 2.12 |
| `singular_values()` | 32 | 18.6 | 55.0 | 32.2 | 21.9 |
| `singular_values()` | 128 | 460 | 922 | 645 | 555 |
| `singular_values()` | 512 | 15,848 | 28,704 | 30,747 | 10,238 |
| `singular_values()` | 1024 | 109,713 | 183,136 | 219,850 | 50,326 |
| `hermitian_eigen()` | 8 | 1.29 | 2.25 | 1.67 | 2.33 |
| `hermitian_eigen()` | 32 | 17.4 | 35.2 | 25.5 | 40.0 |
| `hermitian_eigen()` | 128 | 562 | 789 | 859 | 619 |
| `hermitian_eigen()` | 512 | 18,201 | 21,946 | 51,940 | 11,930 |
| `hermitian_eigen()` | 1024 | 113,630 | 145,524 | 396,273 | 64,084 |
| `hermitian_eigenvalues()` | 8 | 1.17 | 2.04 | – | – |
| `hermitian_eigenvalues()` | 32 | 12.8 | 25.7 | – | – |
| `hermitian_eigenvalues()` | 128 | 232 | 418 | – | – |
| `hermitian_eigenvalues()` | 512 | 8,041 | 11,708 | – | – |
| `hermitian_eigenvalues()` | 1024 | 46,139 | 72,953 | – | – |
| `eigen()` | 8 | 3.88 | 7.96 | – | 5.79 |
| `eigen()` | 32 | 85.0 | 149 | – | 109 |
| `eigen()` | 128 | 2,813 | 3,663 | – | 4,702 |
| `eigen()` | 512 | 83,189 | 106,737 | – | 108,339 |
| `eigen()` | 1024 | 497,656 | 759,539 | – | 409,343 |
| `eigenvalues()` | 8 | 3.25 | 5.75 | – | – |
| `eigenvalues()` | 32 | 49.3 | 111 | – | – |
| `eigenvalues()` | 128 | 1,599 | 2,290 | – | – |
| `eigenvalues()` | 512 | 47,326 | 71,632 | – | – |
| `eigenvalues()` | 1024 | 276,825 | 531,718 | – | – |
| `exp()` | 8 | 1.08 | – | 3.33 | – |
| `exp()` | 32 | 19.2 | – | 36.8 | – |
| `exp()` | 128 | 969 | – | 1,440 | – |
| `exp()` | 512 | 60,683 | – | 94,904 | – |
| `exp()` | 1024 | 511,593 | – | 777,554 | – |

- Every operation is faster than faer's and nalgebra's at every size measured, but `cholesky()` of `8 × 8`, as fast as nalgebra's. At `1024 × 1024` faer takes 1.10 times as long for the product, 1.25 for `lu()` and `inverse()`, 1.15 for `cholesky()`, 1.12 for `qr()`, 1.36 for `svd()`, 1.67 for `singular_values()`, 1.28 for `hermitian_eigen()`, 1.58 for `hermitian_eigenvalues()`, 1.53 for `eigen()` and 1.92 for `eigenvalues()`; nalgebra 1.52 for `exp()`.
- The product runs at about 60 GFLOP/s in `f64` and 120 in `f32` from `256 × 256`, what a loop of fused multiply-adds alone sustains on one core of this machine. Accelerate runs its products on the M4's matrix unit (SME), several times faster from `32 × 32`, and is faster than these wherever products are most of the work: `cholesky()` at every size, `lu()` and `inverse()` from `128 × 128`, `qr()`, `svd()`, `singular_values()` and `hermitian_eigen()` from `512 × 512`, and `eigen()` at `1024 × 1024`; at the smaller sizes it is slower.
- For `f32` and complex elements, measured while the algorithms were tuned, the standing is the same with a few exceptions, where faer or nalgebra is faster: for `Complex<f64>`, `lu()` of `3 × 3`, `4 × 4`, `32 × 32` and `64 × 64`, `inverse()` of `2 × 2`, `3 × 3`, `32 × 32` and `64 × 64`, and `cholesky()` of `32 × 32` and `128 × 128`; `eigenvalues()` of `128 × 128` complex matrices, by about 10 %; `hermitian_eigen()` of `Complex<f32>` from `96 × 96`, by 5 to 8 %; and in `f32`, `inverse()` of `2 × 2` to `4 × 4` and `cholesky()` of `8 × 8`. nalgebra inverts `3 × 3` and `4 × 4` matrices by their cofactors, which also takes less time than the elimination in `f64` (18 ns against 21 for `3 × 3`), but gives no `None` for a pivot that is zero.

### Exactly

Measured on an Apple M4 Max on 2026-11-11, one core, with nothing else running (load average 1.3 to 3.6), in µs (or ms and s), the median of 5 rounds of fresh random matrices, interleaved in one process, against FLINT 3.6's defaults (`fmpz_mat_det`, `fmpq_mat_solve_fmpz_mat`, `fmpz_mat_inv` then `fmpq_mat_set_fmpz_mat_div_fmpz`), FLINT's Bareiss alone (`fmpz_mat_det_bareiss`, `fmpz_mat_solve_fflu`), PARI 2.19 (`det`, `RgM_solve`, `RgM_inv`) and this crate's 0.2.33 (its Bareiss elimination on numbers, and `inverse()` by Berkowitz's polynomial), with each one's time over `a`'s in brackets. The tables for matrices of rationals are on the [rational numbers](rational.md) page.

`Matrix<Number>::det()`, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.33 | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.05 µs | 0.13 µs (2.46) | 0.01 µs (0.10) | 0.09 µs (1.78) | 0.02 µs (0.37) |
| 4 | 0.13 µs | 0.93 µs (7.32) | 0.04 µs (0.28) | 0.22 µs (1.72) | 0.67 µs (5.31) |
| 8 | 0.50 µs | 6.99 µs (14.01) | 0.84 µs (1.68) | 0.99 µs (1.98) | 3.05 µs (6.10) |
| 16 | 6.79 µs | 64.1 µs (9.44) | 7.63 µs (1.12) | 8.19 µs (1.21) | 36.4 µs (5.36) |
| 32 | 109.8 µs | 664.6 µs (6.05) | 68.4 µs (0.62) | 175.7 µs (1.60) | 216.4 µs (1.97) |
| 64 | 492.9 µs | 7.15 ms (14.50) | 559.6 µs (1.14) | 2.86 ms (5.80) | 1.87 ms (3.79) |
| 128 | 3.17 ms | 87.2 ms (27.51) | 3.56 ms (1.12) | 39.3 ms (12.40) | 18.3 ms (5.77) |

`Matrix<Number>::det()`, 64-bit integers (median, and the time over `a`'s):

| n | `a` | `a` 0.2.33 | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.12 µs | 0.12 µs (1.08) | 0.02 µs (0.19) | 0.15 µs (1.28) | 0.02 µs (0.17) |
| 4 | 0.74 µs | 1.01 µs (1.36) | 0.22 µs (0.29) | 0.62 µs (0.83) | 5.17 µs (6.98) |
| 8 | 7.13 µs | 11.5 µs (1.61) | 5.61 µs (0.79) | 6.14 µs (0.86) | 27.6 µs (3.87) |
| 16 | 87.9 µs | 153.1 µs (1.74) | 80.7 µs (0.92) | 81.4 µs (0.93) | 162.4 µs (1.85) |
| 32 | 554.8 µs | 2.27 ms (4.10) | 511.4 µs (0.92) | 1.32 ms (2.38) | 1.5 ms (2.70) |
| 64 | 2.92 ms | 38 ms (13.02) | 3.19 ms (1.09) | 27.6 ms (9.45) | 13.4 ms (4.59) |

`Matrix<Number>::solve(&b)`, one column, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a`, `map(Rational::from)` first | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.16 µs | 0.29 µs (1.84) | 0.06 µs (0.36) | 0.20 µs (1.28) | 1.36 µs (8.63) |
| 4 | 0.37 µs | 0.75 µs (2.05) | 0.37 µs (1.01) | 0.45 µs (1.22) | 2.50 µs (6.82) |
| 8 | 1.23 µs | 2.52 µs (2.05) | 1.46 µs (1.19) | 1.65 µs (1.34) | 12.5 µs (10.15) |
| 16 | 12.2 µs | 17.0 µs (1.39) | 29.1 µs (2.38) | 15.5 µs (1.27) | 53.8 µs (4.41) |
| 32 | 96.9 µs | 121.7 µs (1.26) | 120.3 µs (1.24) | 218.5 µs (2.26) | 321.9 µs (3.32) |
| 64 | 447.5 µs | 531.0 µs (1.19) | 590.2 µs (1.32) | 3.17 ms (7.08) | 5.14 ms (11.49) |
| 128 | 2.47 ms | 2.84 ms (1.15) | 3.62 ms (1.46) | 41.7 ms (16.88) | 49.9 ms (20.21) |

`Matrix<Number>::solve(&b)`, one column, 64-bit integers (median, and the time over `a`'s):

| n | `a` | `a`, `map(Rational::from)` first | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.55 µs | 0.70 µs (1.27) | 0.35 µs (0.63) | 0.63 µs (1.15) | 17.7 µs (32.05) |
| 4 | 3.40 µs | 3.73 µs (1.10) | 3.14 µs (0.92) | 3.33 µs (0.98) | 45.6 µs (13.39) |
| 8 | 18.3 µs | 19.2 µs (1.05) | 18.3 µs (1.00) | 18.9 µs (1.03) | 228.5 µs (12.47) |
| 16 | 148.8 µs | 154.8 µs (1.04) | 244.7 µs (1.64) | 146.5 µs (0.98) | 1.12 ms (7.56) |
| 32 | 682.9 µs | 703.8 µs (1.03) | 904.3 µs (1.32) | 1.76 ms (2.58) | 6.59 ms (9.66) |
| 64 | 3.65 ms | 3.73 ms (1.02) | 5.34 ms (1.46) | 31.3 ms (8.57) | 47.6 ms (13.03) |

`Matrix<Number>::inverse_rational()` (`inverse()`: the adjugate and the determinant), integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a`, `map(Rational::from)` first | `a`, `inverse()` | `a` 0.2.33 | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.21 µs | 0.31 µs (1.43) | 0.16 µs (0.74) | 0.55 µs (2.57) | 0.08 µs (0.37) | 0.25 µs (1.16) | 0.43 µs (2.03) |
| 4 | 0.85 µs | 1.16 µs (1.37) | 0.51 µs (0.60) | 5.33 µs (6.29) | 0.79 µs (0.93) | 0.85 µs (1.00) | 11.2 µs (13.26) |
| 8 | 5.75 µs | 6.77 µs (1.18) | 2.29 µs (0.40) | 77.8 µs (13.53) | 5.22 µs (0.91) | 5.39 µs (0.94) | 18.5 µs (3.21) |
| 16 | 86.3 µs | 89.3 µs (1.03) | 55.3 µs (0.64) | 1.25 ms (14.51) | 207.6 µs (2.41) | 109.6 µs (1.27) | 107.3 µs (1.24) |
| 32 | 1.08 ms | 1.08 ms (1.01) | 637.0 µs (0.59) | 25.2 ms (23.33) | 2.01 ms (1.86) | 1.42 ms (1.31) | 1.13 ms (1.05) |
| 64 | 13.3 ms | 13.3 ms (1.00) | 7.92 ms (0.60) | 507 ms (38.28) | 23.7 ms (1.79) | 18.6 ms (1.41) | 9.25 ms (0.70) |
| 128 | 169 ms | 170 ms (1.00) | 121 ms (0.72) | 10 s (59.31) | 236 ms (1.40) | 232 ms (1.38) | 93.7 ms (0.56) |

`Matrix<Number>::inverse_rational()` (`inverse()`: the adjugate and the determinant), 64-bit integers (median, and the time over `a`'s):

| n | `a` | `a`, `map(Rational::from)` first | `a`, `inverse()` | `a` 0.2.33 | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.81 µs | 0.90 µs (1.11) | 0.37 µs (0.46) | 0.56 µs (0.70) | 0.37 µs (0.46) | 0.73 µs (0.91) | 0.86 µs (1.07) |
| 4 | 8.00 µs | 8.34 µs (1.04) | 2.66 µs (0.33) | 7.43 µs (0.93) | 8.28 µs (1.04) | 8.30 µs (1.04) | 35.0 µs (4.38) |
| 8 | 81.2 µs | 81.9 µs (1.01) | 23.0 µs (0.28) | 163.0 µs (2.01) | 90.4 µs (1.11) | 91.3 µs (1.12) | 172.6 µs (2.13) |
| 16 | 884.4 µs | 882.7 µs (1.00) | 280.6 µs (0.32) | 2.65 ms (3.00) | 2.64 ms (2.99) | 1.1 ms (1.25) | 1.03 ms (1.16) |
| 32 | 10.9 ms | 10.9 ms (1.00) | 5.12 ms (0.47) | 47.3 ms (4.35) | 27 ms (2.48) | 14.1 ms (1.29) | 9.27 ms (0.85) |

- The determinant, the inverse as the adjugate and the solutions run on registers and limbs rather than numbers, 1.08 to 27.5 times as fast as 0.2.33's determinant, and `inverse()` 7 to 83 times as fast as by Berkowitz's polynomial from 8 × 8 (and faster at every size, 1.5 times at 2 × 2).
- FLINT is faster at 2 × 2 and for determinants at 4 × 4, by cofactors in words, for determinants of small integers at 32 × 32 (0.62 of the time) and of 64-bit ones from 8 × 8 to 32 × 32 (0.79 to 0.92); from 64 × 64 the determinant and the solutions, by Dixon's lifting here and in FLINT, are 1.09 to 1.46 times as fast as FLINT's. PARI's multimodular inverse is faster from 64 × 64 for small integers (0.56 to 0.70 of the time) and at 32 × 32 for 64-bit ones (0.85).

## Examples

```rust
use a::math::complex::Complex;
use a::math::matrix::{Matrix, Norm};
use a::math::numbers::Number;
use a::math::rational::Rational;

let a: Matrix<f64> = Matrix::from_rows(&[[4.0, 2.0], [2.0, 3.0]]);
let b = Matrix::from_rows(&[[1.0], [2.0]]);

// A·x = b, by LU; and the same with the inverse.
let x = a.solve(&b).unwrap();
assert!((&(&a * &x) - &b).norm(Norm::Max) < 1e-15);
assert!((&(&a.inverse().unwrap() * &b) - &x).norm(Norm::Max) < 1e-15);
assert!((a.det() - 8.0).abs() < 1e-15);

// Symmetric positive definite: Cholesky, and real eigenvalues.
let l = a.cholesky().unwrap().l();
assert!((&(&l * &l.transpose()) - &a).norm(Norm::Max) < 1e-15);
let e = a.hermitian_eigen();
assert!((e.values()[0] + e.values()[1] - a.trace()).abs() < 1e-14);

// A rotation has eigenvalues ±i, so complex ones.
let r: Matrix<f64> = Matrix::from_rows(&[[0.0, -1.0], [1.0, 0.0]]);
let mut w = r.eigenvalues();
w.sort_by(|p, q| p.im.total_cmp(&q.im));
assert_eq!(w, [Complex::new(0.0, -1.0), Complex::new(0.0, 1.0)]);

// Its exponential rotates by a radian, and its logarithm is back where it started.
let t: Matrix<f64> = Matrix::from_rows(&[[0.0, -1.0], [1.0, 0.0]]);
let rot = t.exp();
assert!((rot[(0, 0)] - 1f64.cos()).abs() < 1e-15);
assert!((&rot.log().unwrap() - &t).norm(Norm::Max) < 1e-14);

// Exactly, on numbers.
let n = Matrix::from_rows(&[[2, 1], [5, 3]]).map(|&x| Number::from(x));
assert_eq!(n.det(), 1);
let (adj, det) = n.inverse().unwrap();
assert_eq!(&n * &adj, &Matrix::identity(2) * &det);

// And on rationals: the Hilbert matrix, whose inverse is of integers.
let h = Matrix::from_fn(3, 3, |i, j| Rational::from(1) / (i + j + 1) as u32);
assert_eq!(h.det().to_string(), "1/2160");
let inv = h.inverse().unwrap();
assert_eq!(inv[(1, 1)], 192);
assert_eq!(&h * &inv, Matrix::identity(3));
let x = h.solve(&Matrix::from_rows(&[[1], [0], [0]]).map(|&x| Rational::from(x))).unwrap();
assert_eq!(x, Matrix::from_columns(&[inv.column(0)]));
```

## Efficiency

- Time complexity: `2·m·n·k` operations for an `m × k` by `k × n` product; for `n × n` matrices, about `2n³/3` for `lu()`, `2n³` for `inverse()`, `n³/3` for `cholesky()`, `4n³/3` for `qr()`, `4n³/3` for `hermitian_eigenvalues()` and `4.7n³` with the eigenvectors, `8n³/3` for `singular_values()` and `9.3n³` for `svd()`, and about `10n³` for `eigenvalues()` and `25n³` for `eigen()`, the QR algorithm's sweeps, whose number depends on the matrix, taking most of it; `exp()` up to six products and one solve, and a product for each squaring; `sqrt()` and `log()` a Schur decomposition, then about `n³/3` for each square root of its triangular factor. On numbers and rationals, `det()` and `rank()` about `n³/3` steps of two products and an exact division of numbers as long as the minors, `solve(&b)` for `k` right-hand sides about `n³/3 + k·n²` and `inverse()` about `4n³/3`, then `n·k` and `n²` greatest common divisors; for the large systems and determinants, Dixon's lifting `n³/3` products modulo a prime and `2·n²` products of words for each of its `L` steps, `L` about the length of the determinant over 61 bits, twice for the solutions; `characteristic_polynomial()` about `n⁴/2` products and `adjugate()` `3n⁴/2`.
- Space complexity: a matrix is its elements in one vector, column after column, and its shape; a decomposition keeps its factors (`lu()` the matrix and the permutation, `qr()` the reflections and the triangular factors of their blocks, at most 64 elements per column); working memory of at most a few times the matrix, and a product packs blocks of up to `512 × 1020` elements of `op(B)` (`512 × 2040` for `f32`) and `128 × 512` of `op(A)`.
- Soundness: `unsafe` code is the product's kernels and the leaves of the triangular solves and of the Cholesky decomposition, each with the extent of what it reads and writes checked before it is called; Miri runs the portable kernels, which have the same shapes as the NEON ones. The exact algorithms use the numbers' limb kernels through checked slices. A shape that doesn't fit panics; the iterations of the QR algorithms stop after a bounded number of sweeps.

[A](../../../README.md) › [Math](../../../README.md#math)

# Matrices

`Matrix<T>` (in `a::math::matrix`) is a rectangular array of elements of the type `T`, held column after column in one vector, of any size chosen when it is built. `T` is one of five types, the sealed trait `Element`:

- `f32` and `f64`, and [`Complex<f32>` and `Complex<f64>`](complex.md): the trait `Scalar`, with every operation below. `T::Real` is the float type of their parts, of norms, singular values and the eigenvalues of Hermitian matrices.
- [`Number`](numbers.md), an integer of any size: the arithmetic, the transposes, the row and column operations, the trace and powers, and, exactly, the determinant, the rank, the characteristic polynomial, the adjugate, the inverse as the adjugate and the determinant, and the null space.

## Building and reading

- `Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]])` and `from_columns` take rows or columns of anything that is a slice; `from_row_major(rows, cols, vec)` and `from_column_major(rows, cols, vec)` a vector, the second without copying; `from_fn(rows, cols, |i, j| …)` a function of the indices. `zeros(rows, cols)`, `identity(n)` and `from_diagonal(&d)` need the element's 0 and 1. A shape that doesn't fit panics.
- `rows()`, `cols()`, `shape()`, `is_square()` and `is_empty()`; `a[(i, j)]` reads and writes an element and panics outside, `get(i, j)` and `get_mut(i, j)` give `None` there; `column(j)` is a slice, `row(i)` an iterator; `as_slice()`, `as_mut_slice()` and `into_vec()` the elements column after column; `diagonal()`; `map(|x| …)` a matrix of another type, as `a.map(|&x| Complex::from(x))` makes a real matrix complex.
- Matrices compare with `==`, clone, hash when their elements do, and default to `0 × 0`. `Display` writes the rows on lines of their own, each column right-aligned, two spaces apart, with the precision given; `Debug` writes them as lists, `[[1.0, 2.0], [3.0, 4.0]]`.

## Arithmetic and transformations

- `+`, `-` and their assignments add and subtract matrices of the same shape, elementwise; `-a` negates; `*` multiplies matrices whose inner dimensions agree, and a matrix by an element on either side, `&a * 2.0` and `2.0 * &a`; `/` divides by an element. Operands may be owned or borrowed; an owned left operand, or the target of an assignment, keeps its vector. Shapes that don't fit panic.
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

## Exactly, on numbers

For `Matrix<Number>`, `det()` is Bareiss's fraction-free elimination, `rank()` its number of pivots, `characteristic_polynomial()` Berkowitz's method without a division, `adjugate()` `adj(A)` with `A·adj(A) = det(A)·I` also for singular matrices, `inverse()` the pair `(adj(A), det(A))`, `None` for a zero determinant, and `null_space()` a basis of integer vectors without common factors.

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
- On `Number`: Bareiss's elimination takes about `n³` products of numbers no longer than the determinant; Berkowitz's method about `n⁴/2`; the adjugate is `(−1)ⁿ⁻¹·(Aⁿ⁻¹ + c₁·Aⁿ⁻² + … + cₙ₋₁·I)` by Cayley and Hamilton, from the characteristic polynomial's coefficients `c`, for singular matrices too; the null space comes from the fraction-free echelon form, one vector for each column without a pivot.

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

## Examples

```rust
use a::math::complex::Complex;
use a::math::matrix::{Matrix, Norm};
use a::math::numbers::Number;

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
```

## Efficiency

- Time complexity: `2·m·n·k` operations for an `m × k` by `k × n` product; for `n × n` matrices, about `2n³/3` for `lu()`, `2n³` for `inverse()`, `n³/3` for `cholesky()`, `4n³/3` for `qr()`, `4n³/3` for `hermitian_eigenvalues()` and `4.7n³` with the eigenvectors, `8n³/3` for `singular_values()` and `9.3n³` for `svd()`, and about `10n³` for `eigenvalues()` and `25n³` for `eigen()`, the QR algorithm's sweeps, whose number depends on the matrix, taking most of it; `exp()` up to six products and one solve, and a product for each squaring; `sqrt()` and `log()` a Schur decomposition, then about `n³/3` for each square root of its triangular factor. On numbers, `det()` and `rank()` about `n³` products of numbers as long as the minors, `characteristic_polynomial()` about `n⁴/2` and `adjugate()` `3n⁴/2`.
- Space complexity: a matrix is its elements in one vector, column after column, and its shape; a decomposition keeps its factors (`lu()` the matrix and the permutation, `qr()` the reflections and the triangular factors of their blocks, at most 64 elements per column); working memory of at most a few times the matrix, and a product packs blocks of up to `512 × 1020` elements of `op(B)` (`512 × 2040` for `f32`) and `128 × 512` of `op(A)`.
- Soundness: `unsafe` code is the product's kernels and the leaves of the triangular solves and of the Cholesky decomposition, each with the extent of what it reads and writes checked before it is called; Miri runs the portable kernels, which have the same shapes as the NEON ones. A shape that doesn't fit panics; the iterations of the QR algorithms stop after a bounded number of sweeps.

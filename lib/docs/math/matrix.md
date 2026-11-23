[A](../../../README.md) › [Math](../../../README.md#math)

# Matrices

`Matrix<T>` (in `a::math::matrix`) is a rectangular array of elements of the type `T`, held column after column in one vector, of any size chosen when it is built. `T` is one of eight types, the sealed trait `Element`:

- `f32` and `f64`, and [`Complex<f32>` and `Complex<f64>`](complex.md): the trait `Scalar`, with every operation below. `T::Real` is the float type of their parts, of norms, singular values and the eigenvalues of Hermitian matrices.
- [`Number`](numbers.md), an integer of any size: the arithmetic, the transposes, the row and column operations, the trace and powers, and, exactly, the determinant, the rank, the characteristic polynomial, the adjugate, the inverse as the adjugate and the determinant or as rationals, the solutions over the rationals, the null space, and the [Hermite and Smith normal forms](#normal-forms) with their transformations.
- [`Rational`](rational.md), a fraction of numbers in lowest terms: the same arithmetic, and, exactly, the determinant, the rank, the solutions, the inverse and the null space.
- [`BigFloat` and `BigComplex`](float.md#matrices-of-floats), floats of any precision: the arithmetic, with products whose every element is correctly rounded, and the determinant, the solutions, the inverse, the rank and some norms at the working precision, on the page of floats.

Two more types are matrices over finite fields, held row after row: [`ModMatrix`](#matrices-over-gfp), over `GF(p)` for a prime `p < 2^64`, each element a word, and [`BitMatrix`](#matrices-over-gf2), over `GF(2)`, 64 elements a word. The rows of a `Matrix<Number>` are the basis of a lattice for [`a::math::lattice`](lattice.md), which reduces it and finds its short vectors.

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

For `Matrix<Number>`, `det()` is the determinant by Bareiss's fraction-free elimination, by its residues modulo primes or by Dixon's divisor, chosen by the order and the length of the integers, `rank()` the number of pivots of Bareiss's elimination, `characteristic_polynomial()` Berkowitz's method without a division, `adjugate()` `adj(A)` with `A·adj(A) = det(A)·I` also for singular matrices, `inverse()` the pair `(adj(A), det(A))`, `None` for a zero determinant, and `null_space()` a basis of integer vectors without common factors. `solve(&b)` is the solution `X` of `A·X = B` as a `Matrix<Rational>`, and `inverse_rational()` the inverse as one, each `None` for a singular matrix, which is proved singular. `*` is Strassen–Winograd's recursion from a size that falls as the elements grow.

For `Matrix<Rational>`, `det()`, `rank()`, `solve(&b)`, `inverse()` and `null_space()` are exact: the determinant as a rational, `None` from `solve` and `inverse` for a singular matrix, and the null space as the basis of the reduced row echelon form, a 1 in each column without a pivot and 0s in the others. Each matrix of rationals is first made one of integers, row by row, so that no fraction is formed inside the elimination. The results are in the byte order of the top left element, little-endian for an empty matrix. A matrix that is not square panics in `det`, `solve` and `inverse`, and so does a right-hand side with a different number of rows.

## Normal forms

`Matrix<Number>` gives the Hermite normal form of its rows and the Smith normal form, exactly, with unimodular transformations, for any shape and rank:

| Method | Returns | |
| --- | --- | --- |
| `hermite_normal_form()` | `(H, U)` with `U·A = H` | `H` the Hermite normal form of the rows; `U` unimodular, the transformation of the Hermite form of `[A \| I]` |
| `hermite_form()` | `H` | the same `H` without `U`, for less work |
| `smith_normal_form()` | `(S, U, V)` with `U·A·V = S` | `S` the Smith normal form; `U` and `V` unimodular |
| `elementary_divisors()` | `Vec<Number>` | the diagonal of `S`, `min(m, n)` elements, without the transformations |

**Conventions.** The forms act on the rows, as FLINT's `fmpz_mat_hnf` and `fmpz_mat_snf` do (and as lattice bases are rows in [`a::math::lattice`](lattice.md)). `H` is in row echelon form: its `r` nonzero rows first, `r` the rank, the first nonzero element of each, its pivot, positive and to the right of the one above, every element above a pivot in `[0, pivot)`, and the `m − r` zero rows last. Its pivots are in the column rank profile, the first columns from the left that are independent. `H` is unique, the one basis of this form of the lattice the rows span; so is `S`, diagonal, its elements `d₁ | d₂ | …` nonnegative, the zeros last, `dᵢ` the quotient of the greatest common divisors of the minors of orders `i` and `i − 1`. `U` of the Hermite form is unique when the rows are independent; for dependent rows it is the one for which `[H | U]` is the Hermite form of `[A | I]`, so its last `m − r` rows are the Hermite form of the lattice of the left kernel `{u : u·A = 0}`, their pivots in the rows of `A` that are combinations of the rows below them, with the elements of `U` above those pivots in `[0, pivot)`. The transformations of the Smith form are not unique; the last `m − r` rows of its `U` are a basis of the left kernel, those of the Hermite form's `U`, and the last `n − r` columns of its `V` one of the kernel `{v : A·v = 0}`. PARI's `mathnf` triangularises the columns from the other end: for `B`, `A` with its columns in the opposite order, `H` without its zero rows is `mathnf(Bᵀ)ᵀ` with the order of its rows and of its columns reversed. PARI's `matsnf` gives the divisors from the largest, padded in front with zeros to the number of rows. The results are in the byte order of the top left element, little-endian for an empty matrix.

### Methods

- **Rank profiles.** With `J` the column rank profile and `R` the row rank profile from the bottom (row `i` in it when it is not a combination of the rows below it), `M = A[R, J]` is a nonsingular `r × r` minor. Both profiles come from one elimination modulo a prime `p` below `2^62`, the rows from the bottom each reduced by the echelon basis of the ones kept, `O(m·n·r)` operations on words; they are then proved over the integers: `X = M⁻¹·A[R, K]` for the columns `K` outside `J` must satisfy `A[S, J]·X = A[S, K]` on the other rows `S` (so the rank is `r`) and be zero where a column of `J` comes after the column it expresses (so `J` is the profile); for the transformation, `Y = A[S, J]·M⁻¹` must be zero where a row of `R` comes before the row it expresses. A prime that fails a check (it divides one of the `m + n` minors the profiles depend on) is followed by the next one below it. `X` and `Y` are exact solutions with `M` (`solve`, by Bareiss's elimination or Dixon's lifting), and `D = |det M|`.
- **Hermite normal form modulo `D`** (Domich, Kannan and Trotter, *Hermite normal form computation using modulo determinant arithmetic*, Math. Oper. Res. 12, 1987; Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 2.4.8, here for rows). A lattice of full rank `n` whose determinant divides `D` contains `D·Zⁿ`, so its form is that of its generators with `D·Zⁿ` added and every element can be kept in `[0, D)`. Column `k` in turn: the rows with an element there are combined by extended Euclidean steps of two rows into one row `g` with the greatest common divisor `g_k` of the column; the pivot is `d_k = gcd(g_k, R_k)`, the row `u·g` with `u·g_k ≡ d_k (mod R_k)`; the vectors with zeros up to column `k` form a lattice of determinant `det / (d₀⋯d_k)`, which the other rows span together with `R_{k+1}·Z^{n−k−1}`, `R_{k+1} = R_k / d_k`, so the rest goes on modulo `R_{k+1}`. Last, the elements above each pivot are brought into `[0, d_l)` by its row. No element grows past `D`, which is at most `r·(b + ½·log₂ r)` bits for elements of `b` bits (Hadamard's bound).
- **The Hermite form of any matrix.** The columns `J` of `H` are the Hermite form of the rows of `A[:, J]`, of full rank `r` with a determinant that divides `D`, computed modulo `D`; the other columns are `H[:, K] = H[:, J]·X`, since a vector of the row space is fixed by its elements in `J`. `hermite_form()` stops there: a square nonsingular matrix takes the elimination modulo a prime, its determinant and the form modulo it, and nothing else.
- **The transformation.** `U` is that of the square `B = [A[:, J] | E_S]`, `E_S` the columns of the identity for the rows `S` outside `R`, of the same determinant `±D`: `U = H_B·B⁻¹`, unimodular as the transformation between two bases of one lattice. `B` is `[A | I]` on the columns of its pivots, so `H_B` is the form of `[A | I]` on them and `U` its transformation. `H_B` is not computed as the form of an `m × m` matrix, which would cost `O(m³)`, but from its structure. With `N = D·Y`, a matrix of integers, its last `m − r` rows are `(−z·Y, z)` on the columns `R` and `S` for the rows `z` of the Hermite form of the lattice `Λ = {z : z·N ≡ 0 (mod D)}`. `Z^{m−r} / Λ` is the subgroup of `(Z/D)^r` that the rows of `N` span, so at most `t ≤ log₂ D` pivots of that form are above 1, and each row is its pivot and elements below the pivots above 1 after it. The pivot at row `k` is the order of `N_k` modulo the subgroup the later rows span, which is held, from the last row up, as a triangular basis of that subgroup with `D·Z^r` added, each basis row with its coefficients on the rows of `N` it came from: one pass down that basis gives the order and the coefficients, and a new pivot above 1 joins the basis by extended Euclidean steps. The first `r` rows are `((Z_i − w·N) / D, w)` for `Z = D·H[:, J]·M⁻¹` and `w` the solution of `w·N ≡ Z_i (mod D)` from the same basis, reduced by the rows of the form of `Λ`. Each row costs `O(r² + r·t + t²)` operations on numbers below `D`.
- **Elementary divisors** (Cohen, Algorithm 2.4.14, here for any shape and rank). `d₁⋯d_r` is the greatest common divisor of the `r × r` minors, so it divides `D`, and the Smith form of `A` over `Z/DZ` is `diag(d₁, …, d_r, 0, …)`. Position `t` takes a nonzero element of what remains as its pivot and clears its column and its row by steps of two rows or two columns: a subtraction when the pivot divides, an extended Euclidean step otherwise, which makes the pivot a proper divisor of itself. When `g = gcd(pivot, R)` does not divide some element of the block, that element's row is added to the pivot's and the clearing goes on with a smaller pivot. Then `d_t = g`, and the block is reduced modulo `R / d_t`, which the product of the divisors left still divides.
- **The Smith form with its transformations** (alternating Hermite forms, as Kannan and Bachem do in *Polynomial algorithms for computing the Smith and Hermite normal forms of an integer matrix*, SIAM J. Comput. 8, 1979). `U₁·A = H` as above; the Hermite form of the columns of its top `r` rows, `H[..r]·V₁ = [K | 0]`, `K` lower triangular of order `r`. `V₁` is the transpose of the transformation of the form of `[H[..r]ᵀ | E_K]`, `K` the columns of `H` outside its pivots, computed as above from profiles known from `A`'s (the columns of `H[..r]ᵀ` independent, its rows `J` and `Y = Xᵀ`); it is a transformation of the Hermite form of the columns, not that of `[H[..r]ᵀ | I]`. Both `U₁` and `V₁` keep their kernel parts from here on: only the first `r` rows of `U` and columns of `V` change, so the work stays linear in `m` and `n`. `K` is put in its Hermite form by rows and by columns in turn, each time modulo its determinant with the unique transformation of a nonsingular matrix (an exact solution), until it is diagonal; a leading row and column that are clear stay so, and the turns go on with the block after them. Each turn either makes the leading element a proper divisor of itself or clears its row and its column, so there are at most `r` turns plus the number of prime factors of the determinants of the blocks. Then pairs `diag(a, b)` become `diag(gcd, lcm)` by `P = [[x, y], [−b/g, a/g]]` and `Q = [[1, −y·b/g], [1, x·a/g]]`, `P·diag(a, b)·Q` or `Qᵀ·diag(a, b)·Pᵀ`, for `g = x·a + y·b`. gcd and lcm sort the exponents of every prime at once, so the pairs can be the comparators of any sorting network; they are those of Batcher's odd-even merge sort, `O(r·log² r)` steps and `O(log² r)` on each position, with the larger factor `Q` on the side, `U` or `V`, whose two lines are the shorter. Last, the first rows of `U` are reduced modulo its last rows and the first columns of `V` modulo its last columns, each element in the line of a pivot of the kernel part into `[0, pivot)`.

The work, with `D` as above: `hermite_form()` about `m·r²` operations on numbers below `D`, and `hermite_normal_form()` about `m·(r² + r·t + t²)` more for `U`, besides the determinant of `M`, the solutions with it and the `m × m` result; `elementary_divisors()` about `m·n·r` such operations after the proof of the profiles; `smith_normal_form()` the same for `U` and for `V`, then an `r × r` Hermite form and an exact solution for each turn (none for 84 % of the random matrices of the sweeps below, at most four), and `O(r·log² r)` gcd and lcm steps on lines of length `m` and `n`.

### Examples

```rust
use a::math::matrix::Matrix;
use a::math::numbers::Number;

let a = Matrix::from_rows(&[[2, 4, 4], [-6, 6, 12], [10, -4, -16]]).map(|&x| Number::from(x));

// The Hermite form of the rows, and its transformation.
let (h, u) = a.hermite_normal_form();
assert_eq!(h, Matrix::from_rows(&[[2, 4, 4], [0, 6, 0], [0, 0, 12]]).map(|&x| Number::from(x)));
assert_eq!(&u * &a, h);
assert_eq!(u.det().abs(), 1);
assert_eq!(a.hermite_form(), h);

// The Smith form: Z³ modulo the rows is Z/2 × Z/6 × Z/12.
let (s, u, v) = a.smith_normal_form();
assert_eq!(s, Matrix::from_diagonal(&[2, 6, 12].map(Number::from)));
assert_eq!(&(&u * &a) * &v, s);
assert_eq!(a.elementary_divisors(), s.diagonal());

// Dependent rows: zero rows at the bottom of H, and the left kernel in the last rows of U.
let b = Matrix::from_rows(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]).map(|&x| Number::from(x));
let (h, u) = b.hermite_normal_form();
assert_eq!(h, Matrix::from_rows(&[[1, 2, 3], [0, 3, 6], [0, 0, 0]]).map(|&x| Number::from(x)));
let kernel = Matrix::from_fn(1, 3, |_, j| u[(2, j)].clone());
assert!((&kernel * &b).as_slice().iter().all(|x| x.is_zero()));
assert_eq!(b.elementary_divisors(), [1, 3, 0].map(Number::from));
```

### Verified

By `lib/tests/math/matrix/normal.rs`:

- **Against FLINT 3.6 and PARI/GP 2.19**, on 46 matrices from a seeded generator that the test and `tests/math/matrix/data/gen_normal.py` share: random elements of 2 to 3000 bits in shapes from `1 × 1` to `30 × 30` (`1 × 5`, `5 × 1`, `3 × 5`, `5 × 3`, `12 × 7`, `7 × 12`, `3 × 8` and `8 × 3` of 3000 bits, `8 × 8` of 2000 bits, `16 × 16` of 300 bits, `30 × 30` of 100 bits), products of rank 2 to 20 (`30 × 30` of rank 20 with 60-bit factors, `8 × 8` of rank 5 with 1000-bit ones), elements in `−1..=1` and `−2..=2` up to `20 × 25`, sparse ones, diagonal matrices of divisors `2^a·3^b·5^c`, a fifth of them zero, taken through random row and column operations up to `25 × 30`, and tall and wide ones for the transformations (`160 × 3`, `120 × 6` of rank 2, `100 × 4`, `90 × 5` sparse, `60 × 6` and `6 × 60` of the diagonal kind, `3 × 160`, `6 × 120` of rank 2). `H` equals `fmpz_mat_hnf(A)` and PARI's `mathnf` in FLINT's convention (which the script checks agree), `U` equals the last `m` columns of `fmpz_mat_hnf([A | I])`, and the diagonal of `S` equals `fmpz_mat_snf(A)` and PARI's `matsnf`: no difference. The script stores the elementary divisors in full and `H` and `U` as 64-bit FNV-1a digests of their text (in full when short), 16 KB in all.
- **Sweeps** (ignored tests, run for this page): 1500 random matrices of every kind up to `12 × 12` (`gen_normal.py sweep 1500 23`) and 300 up to `32 × 32` with elements of up to 1000 bits (`gen_normal.py sweep 300 24 big`): `H`, `U` and `S` equal FLINT's and PARI's on every one. The test also gives the longest elements of the transformations against Hadamard's bound `k·(b + ½·log₂ k)` bits, `k = min(m, n)` and `b` the length of the elements: the Smith transformations up to 1.87 and 2.12 times it, the Hermite one up to 1.00 times.
- **The cost of the transformations of tall and wide matrices**: `U` of `2000 × 3` (and of a `1500 × 4` of rank 2) with `U·A = H`, `[H | U]` in Hermite form and `|det U| = 1` from the triangular `U·[A[:, J] | E_S]` without a determinant of order 2000; the Smith forms of `1 × 2000`, `3 × 1500` and `1500 × 3` with `U·A·V = S` and the kernels, and of `1 × 300`, `2 × 250`, `250 × 2` and `200 × 4` with every condition. Each transformation is timed against the schoolbook product `U·A` (`U·A·V`) that checks it, by a loop in the test on the same matrices, the fastest of a few runs of each, and the tests fail above 25 times it: the ratios measured are 0.6 to 3.1, in debug and in release, where the form of the `m × m` matrix `[A[:, J] | E_S]`, which the transformations were before, cubic in the long side, gives 280 to 570 and fails them. The memory now is that of the `m × m` result.
- **The length of the Smith transformations**: the diagonal of the first 40 primes within 1.2 times Hadamard's bound (451 bits against 426; 3859 before, and in FLINT's `fmpz_mat_snf_transform`, which takes the pairs one after the other too), and 40 random odd numbers of 100 bits on a diagonal within twice it (7707 bits against 4106; about 76 000 before). An ignored test measures more (`cargo test --release --test math -- --ignored --nocapture matrix::normal::lengths`): the first 100 primes 0.98 and 1.08 times (from the smallest and from the largest), 100 random odd numbers of 100 bits 1.68 (465 631 bits before, 45 times), the 100 largest primes below `2^62` 1.75, those 100 numbers with small numbers beside the diagonal 1.72 to 1.89 on five matrices and 2.36 on one more, and a matrix `U₀·diag(d)·V₀` with a given Smith form, `50 × 50` with `d` of 300 bits, 2.25. The longest known are of that kind: `d` random odd numbers, `U₀` and `V₀` the identity through `2k` random additions of −2 to 2 times a row (a column) to another and a permutation. The confirmation of the audit measured 2.45 to 2.74 times on five of them from `50 × 50` to `150 × 150` with `d` of 100 and 300 bits, and 3.43 times on a `100 × 100` with elements of 306 bits (`U` 85 524 bits, `V` 106 186, against 30 932), which this crate reproduces; PARI's `matsnf(A, 1)` gives transformations about 40 times as long on such a `50 × 50` matrix (`U` of 1 561 617 bits against 39 664 here). The Hermite form's `U` stays within the bound on all of them.
- **Every output** of those tests, and of 60 more random matrices of every rank up to `6 × 6` and of every rank of seven wide, tall and square shapes, is checked for its conditions: `U·A = H`, `|det U| = 1`, `H` in the form with as many nonzero rows as the rank, `hermite_form()` the same `H`; `U·A·V = S`, `|det U| = |det V| = 1`, `S` diagonal, nonnegative, each element dividing the next, the zeros last, as many nonzero elements as the rank, and `elementary_divisors()` its diagonal; for square matrices, the product of the nonzero divisors and that of the diagonal of `H` are `|det A|`.
- **Edge cases**: `0 × 0`, `0 × 3`, `3 × 0`, zero matrices (forms equal to themselves, identity transformations), `1 × 1` of each sign, a single row and a single column, a diagonal matrix with a zero, dependent rows (the left kernel in `U`, the kernel in `V`), the transformation of a few small matrices against the form of `[A | I]` computed by the method itself, big-endian input.
- **The proof of the rank profiles**, on matrices made to deceive the first primes: a column that is zero modulo the four largest primes below `2^62`, a determinant divisible by the two largest (rank 2 that looks like 1, square, tall and wide), and a row that vanishes modulo the largest (a wrong row profile from the bottom): each gives the true form.
- **Classical examples**: the Smith form `diag(2, 6, 12)` of `[[2, 4, 4], [−6, 6, 12], [10, −4, −16]]`; the Hermite form of `[[2, 3, 6, 2], [5, 6, 1, 6], [8, 3, 1, 1]]`, `[[1, 0, 50, −11], [0, 3, 28, −2], [0, 0, 61, −13]]` (PARI).
- On x86_64 (Ubuntu 26.04 with its x86-64-v3 glibc, Rust 1.100, emulated): the 22 tests pass, the sweep and the measurement ignored.

To reproduce: `python3 lib/tests/math/matrix/data/gen_normal.py` regenerates `normal_expected.txt` (it needs `cc` with FLINT, GMP and MPFR, and `gp`); `cargo test --test math -- matrix::normal` runs the tests; `python3 lib/tests/math/matrix/data/gen_normal.py sweep 1500 23 <file>` and `A_NORMAL_SWEEP=<file> cargo test --release --test math -- --ignored matrix::normal::sweep` run a sweep.

### Limits

- The Hermite form modulo `D` does `O(n³)` operations on numbers as long as the determinant for an `n × n` matrix, `O(n⁴·b)` bit operations and more for elements of `b` bits; FLINT's default for large nonsingular matrices, the algorithm of Pernet and Stein (a determinant, two solutions and the addition of a column and a row, for the matrices whose form has one nontrivial column), is not implemented, nor are Storjohann's. So the gap to FLINT grows with the order: on random square matrices of 10-bit elements an audit found `hermite_form` taking 5 times as long as FLINT's `fmpz_mat_hnf` at 50 rows, 50 times as long at 200 and 75 at 300, and `elementary_divisors` about 30 times as long as `fmpz_mat_snf` at 200 and 300 rows.
- The transformations are dense matrices of `Number`s, 32 bytes for each element, zeros too: `U` of `3200` rows is 330 MB.
- No bound is proved on the lengths of the Smith transformations. They are the products of the transformations of the Hermite forms, of the turns and of the gcd and lcm steps, reduced only modulo the kernels; the measurements above are the worst cases known, 3.43 times Hadamard's bound at most, on a matrix built as `U₀·diag(d)·V₀`. The turns on the 1800 matrices of the sweeps: none for 1509 matrices, one for 214, two for 72, three for 4 and four for 1. The references' transformations are as long or longer: on the diagonal of the first 40 primes FLINT's `V` has 3859 bits and PARI's `U` 5073.
- The rank profiles are proved, not assumed: a matrix made so that its minors are divisible by many of the primes below `2^62` taken in turn (from the largest) costs a determinant and the solutions for each prime it deceives. The primes are the same for every input, so such a matrix can be built on purpose (an audit built one of 20 rows that deceives the 64 largest); the results stay right.

## Matrices over GF(p)

`ModMatrix` (in `a::math::matrix`) is a matrix over the prime field `GF(p)` for a prime `p < 2^64`, `p = 2` included: the modulus is stored once, and each element is a word in `[0, p)`, held row after row. Every constructor checks that `p` is a prime with the crate's test for words (trial division, then Baillie–PSW, which no composite below `2^64` passes) and panics if it is not; elements given as words are reduced modulo `p`. Operations on two matrices over different primes panic, and so do shapes that do not fit, as for `Matrix`.

| Function | What it gives |
| --- | --- |
| `ModMatrix::zeros(rows, cols, p)`, `identity(n, p)` | the zero and the identity matrices |
| `from_fn(rows, cols, p, \|i, j\| …)`, `from_rows(&rows, p)` | a matrix of words, each reduced modulo `p` |
| `from_matrix(&a, p)` | the residues in `[0, p)` of a `Matrix<Number>` of elements of any size and sign |
| `to_matrix()` | the `Matrix<Number>` of the elements, little-endian |
| `rows()`, `cols()`, `shape()`, `is_square()`, `modulus()` | the shape and the prime |
| `get(i, j)`, `a[(i, j)]`, `set(i, j, x)`, `row(i)`, `as_slice()` | an element, a row and all of them row after row; `set` reduces `x`; outside the matrix, a panic |
| `+`, `-`, `*`, unary `-` | owned or borrowed, as for `Matrix`; `*` by Strassen–Winograd's recursion from 128 rows and columns |
| `scale(c)`, `transpose()` | `c·A` and `Aᵀ` |
| `ple()` | the PLE decomposition, a `ModPle`: `permutation()`, `l()`, `e()`, `pivots()` and `rank()`, with row `i` of `L·E` row `permutation()[i]` of `A`, `L` of `m × r` lower triangular with ones on its diagonal, `E` of `r × n` in row echelon form, its pivot columns increasing |
| `rank()` | the rank |
| `det()` | the determinant; 1 for `0 × 0`; a panic for a matrix that is not square |
| `inverse()` | `A⁻¹`, `None` when `A` is singular |
| `solve(&b)` | `X` with `A·X = B` for a square `A` and the columns of `B`, `None` when `A` is singular |
| `rref()` | the reduced row echelon form and its pivot columns |
| `null_space()` | a basis of `{x : A·x = 0}` as the columns of an `n × (n − rank)` matrix: for each column `c` without a pivot, `x_c = 1`, the other such coordinates 0, and `x_(q_i) = −R[i][c]` at the pivot columns `q_i` of the reduced form `R`; the basis FLINT's `nmod_mat_nullspace` gives |

### Methods

- Arithmetic on words. Sums and differences are one comparison. A product is reduced by Möller and Granlund's division of two words by one with the precomputed reciprocal of the normalised `p` ("Improved division by invariant integers", 2011), a Barrett-style reduction with no division instruction; a product by a multiplier fixed for a row operation by Shoup's method, with the precomputed `⌊w·2^64/p⌋`, two multiplications and a correction (in 128 bits for `p ≥ 2^63`, where twice `p` passes a word). Inverses by Euclid's algorithm on words.
- Products. The classical product packs `Bᵀ` once and takes each element as a dot product of a row of `A` with a row of `Bᵀ`, its sum of products held unreduced in one word while `k·(p − 1)² < 2^64`, in two while it is below `2^128`, and in three beyond (a word counting the carries), and reduced once, as FLINT's `nmod_mat` does: `m·k·n` products of words and `m·n` reductions. From `min(m, k, n) ≥ 128` the product is Strassen–Winograd's recursion (Winograd's form of Strassen's method, seven products of halves and fifteen additions, in the schedule of Boyer, Dumas, Pernet and Zhou, ISSAC 2009) down to the classical product, `O(n^2.81)` operations; a dimension that is odd is peeled, the even part going through the recursion and the last row, the last column and the rank-one term of the last inner index taken classically. The residues are those of the classical product. The crossover was chosen by measurement, among 64 to 256, for primes of 20 to 64 bits.
- Elimination. Everything else rests on the PLE decomposition (Jeannerod, Pernet and Storjohann, "Rank-profile revealing Gaussian elimination and the CUP matrix decomposition", J. Symb. Comput. 2013; Ibarra, Moran and Hui's LSP, 1982), computed recursively on the columns: the left half is decomposed, its row swaps applied to whole rows; the top right block becomes `L₁₁⁻¹·A₁₂` by a triangular solve and the bottom right block `A₂₂ − L₂₁·A₁₂` by a product; then the bottom right block is decomposed. Blocks of up to 16 columns are eliminated classically, a pivot row at a time, the first nonzero element of the column, with Shoup's products; triangular solves are halved down to 16 rows, the work off their diagonals products. So most of the work is products, which Strassen–Winograd's recursion takes from 128 rows: `O(n^ω)` operations for `ω = log₂ 7`, and about `n³/3` products of words and as many sums (at most `m·n·r` for an `m × n` matrix of rank `r`) below the crossover. The factors are stored over a copy of the matrix as LAPACK stores its LU factors, the multiplier `L[i][k]` in column `q_k` of row `i`, where `E` has a zero.
- `rank()` is `r`; `det()` the product of the pivots with the sign of the permutation, 0 below full rank; `inverse()` is `U⁻¹·L⁻¹·P` by two triangular solves on the permuted identity, and `solve(&b)` the same on `P·B`; `rref()` scales the rows of `E` to leading ones and applies `U⁻¹`, `U` the unit upper triangular matrix of `E`'s pivot columns, to its other columns, one triangular solve; `null_space()` reads its vectors off that reduced form.

### Example

```rust
use a::math::matrix::{Matrix, ModMatrix};
use a::math::numbers::Number;

let p = 18_446_744_073_709_551_557; // the largest prime below 2^64
let a = ModMatrix::from_rows(&[[2, 1, 4], [1, 3, 5], [3, 4, 9]], p);
// The last row is the sum of the first two: rank 2, determinant 0.
assert_eq!(a.rank(), 2);
assert_eq!(a.det(), 0);
assert_eq!(a.inverse(), None);
let null = a.null_space();
assert_eq!((null.rows(), null.cols()), (3, 1));
assert_eq!(&a * &null, ModMatrix::zeros(3, 1, p));
let (r, pivots) = a.rref();
assert_eq!(pivots, [0, 1]);
assert_eq!(r.row(2), &[0, 0, 0]);

// A·X = B, and the PLE decomposition: row i of L·E is row permutation()[i] of A.
let b = ModMatrix::from_rows(&[[0, 1], [1, 0]], 7);
let c = ModMatrix::from_rows(&[[3, 5], [6, 1]], 7);
let x = b.solve(&c).unwrap();
assert_eq!(&b * &x, c);
let d = b.ple();
assert_eq!(d.permutation(), &[1, 0]);
assert_eq!((d.l() * d.e()).row(0), b.row(1));

// From a matrix of numbers: the residues of its determinant.
let n = Matrix::from_rows(&[[10, -3], [4, 7]]).map(|&x| Number::from(x));
assert_eq!(ModMatrix::from_matrix(&n, 5).det(), 2); // 82 = 2 (mod 5)
```

### Verified

The tests are in `lib/tests/math/matrix/modp.rs` (26 tests, one of them ignored, a heavier sweep for the release profile); the reference results in `lib/tests/math/matrix/data/modp_flint.txt` (47 KB), written by `data/gen_modp.cpp` with FLINT 3.6 and NTL 11.6.

- Against FLINT's `nmod_mat_rank`, `nmod_mat_det`, `nmod_mat_rref`, `nmod_mat_inv`, `nmod_mat_nullspace` and `nmod_mat_mul` (of `A·Aᵀ`), on 456 seeded matrices: 22 shapes from `1 × 1` to `150 × 150`, square, wide and tall, of full rank and below it (products of random factors of the target rank, and the rank they come out with modulo small primes), each element drawn or zero with two densities, modulo the thirteen primes 2, 3, 5, 251, 65 521, 1 000 003, `2^31 − 1`, `2^32 − 5`, `2^50 − 27`, `2^61 − 1`, `2^62 − 57`, `2^63 − 25` and `2^64 − 59` (the large shapes each modulo three or four of them). The rank and the determinant are compared, and hashes of the whole reduced form, inverse, null space basis and product: all 456 agree, the null space basis included. The generator also runs NTL's `mat_zz_p` (below `2^60`) or `mat_ZZ_p` (beyond) on every matrix and stops unless NTL's rank, determinant and inverse are FLINT's and NTL's kernel spans the same space as FLINT's null space; it ran to the end.
- Against a plain Gauss–Jordan elimination in 128-bit arithmetic written in the test, on 22 shapes from `0 × 0` to `130 × 130` (past the classical block, so that the recursion, its triangular solves and its products run), two densities and the thirteen primes: the rank, the pivots and the whole reduced form, the determinant, `A·A⁻¹ = A⁻¹·A = I`, and for the null space `A·N = 0`, its rank, its dimension and its free coordinates; the ignored sweep repeats this on `300 × 300` (of full rank and one below), `257 × 400` and `400 × 257` modulo every prime, with their products, in the release profile.
- The decomposition of `256 × 256` matrices (modulo 65 521 and `2^64 − 59`, of full rank and of rank 255), whose update of the bottom right block is a product of three dimensions of 128 and so goes through Strassen–Winograd's recursion (into a temporary, then subtracted), as do the inverse's triangular solves: the determinant, the reduced form and the null space against the plain elimination, and `A·A⁻¹ = I` by `*` and by the classical product in 128 bits. This runs in the debug profile, as CI runs it.
- The PLE decomposition gives the matrix back, row for row through its permutation, with `L` unit lower triangular and `E` in echelon form, on the same shapes and primes; solutions satisfy their systems (orders 0 to 70, 0 to 3 right-hand sides), and singular systems return `None`.
- Products against the classical product in 128 bits: from `0 × 3 × 2` to `64 × 64 × 64` for every prime, and at the crossover (`127 × 128 × 129`, `128³`, `129³`, `200 × 131 × 197`, `131 × 263 × 133`) and with two levels and odd dimensions at each (`257 × 259 × 261`) for primes of each kind of sum.
- `from_matrix` against `Number`'s remainder for elements of 300 bytes of both signs and byte orders, and the residues of the determinant of a `20 × 20` matrix of 16-byte numbers against `Matrix<Number>::det`.
- The panics: a modulus of 0, 1, `u64::MAX` or the square of a prime; different primes for sums, products and right-hand sides; shapes that do not fit; `det` and `inverse` of a rectangle; elements outside the matrix; rows of different lengths.

To reproduce: `cargo test --test math -- matrix::modp` (and `--release -- --ignored matrix::modp` for the sweep); `data/gen_modp.cpp` has its build line, and rewrites the data file exactly.

### Limits

- Prime moduli only; FLINT's `nmod_mat` also takes composite ones.
- `solve` takes a square matrix and gives `None` when it is singular; a consistent system with a singular or rectangular matrix is solved from `rref()` by hand.
- `inverse` runs both triangular solves on the whole permuted identity, about `2n³` products of words after the decomposition, without using the triangular shape of `L⁻¹·P`.
- The classical product packs `Bᵀ` but does not block for the caches, and nothing is vectorised by hand; the recursion's temporaries take about `4.5·n²` words at the top level.
- An element is a word whatever `p` is: `BitMatrix` holds matrices over `GF(2)` 64 elements a word.

## Matrices over GF(2)

`BitMatrix` (in `a::math::matrix`) is a matrix over the field of two elements, `GF(2)`, packed into bits as M4RI packs them: the rows one after another, each in `⌈cols / 64⌉` words of 64 bits, the element of column `j` bit `j mod 64` of word `j / 64` of its row. The bits of a row's last word beyond its last column are zero after every operation, so `==` and `Hash`, derived over the shape and the words, compare the elements and nothing else. A sum is the exclusive or of the words; products are by M4RM, the method of the Four Russians, with Strassen and Winograd's recursion above it, and every elimination is by M4RI, Bard's method of the Four Russians for echelon forms.

| Function | What it gives |
| --- | --- |
| `BitMatrix::zeros(m, n)`, `identity(n)`, `from_fn(m, n, \|i, j\| …)`, `from_rows(&[[true, false], …])` | a matrix; rows of different lengths panic |
| `BitMatrix::from_words(m, n, words)`, `row_words(i)`, `set_row_words(i, &words)`, `as_words()`, `words_per_row()` | the packed words, row after row; a set bit beyond the last column, or the wrong number of words, panics |
| `BitMatrix::from(&Matrix<Number>)`, `to_matrix()` | the parities of integers of any size and either sign (`−3 ≡ 1`), and back as a matrix of 0s and 1s |
| `get(i, j)`, `set(i, j, b)`, `flip(i, j)`, `swap_rows(a, b)`, `add_row(target, source)` | the elements and the row operations; an index outside the matrix panics; a row added to itself becomes zero |
| `rows()`, `cols()`, `shape()`, `is_square()` | the shape |
| `&a + &b`, `a += &b` | the sum, element by element: the exclusive or; owned or borrowed operands |
| `&a * &b`, `a *= &b` | the product: M4RM, and Strassen–Winograd while all three dimensions are at least 3072 |
| `a.mul_strassen(&b, cutoff)` | the product with Strassen–Winograd while all three dimensions are at least `cutoff` (from 128), as M4RI's `mzd_mul(C, A, B, cutoff)` |
| `transpose()` | `Aᵀ`, by blocks of 64 × 64 bits |
| `rank()` | the rank |
| `det()` | the determinant as a `bool`: `true` exactly when the matrix is invertible, and for `0 × 0`; a matrix that is not square panics |
| `inverse()` | `A⁻¹`, or `None` for a singular matrix |
| `solve(&b)` | a solution `X` of `A·X = B`, for `A` of any shape, its rows of the columns without a pivot zero; `None` when there is none; `B` with another number of rows panics |
| `null_space()` | an `n × (n − rank)` matrix `N` with `A·N = 0` whose columns are the canonical basis of the null space |
| `reduced_row_echelon_form()`, `echelonize()` | the reduced row echelon form and its pivot columns, ascending, both unique; `echelonize` in place |

`Display` writes each row on a line of its own as 0s and 1s; `Debug` writes the shape and the rows as strings, `BitMatrix { rows: 2, cols: 2, data: ["10", "01"] }`. `BitMatrix::default()` is `0 × 0`.

### Methods

- **Products, M4RM** (Arlazarov, Dinic, Kronrod and Faradzev, 1970, as Albrecht, Bard and Hart arrange it in "Algorithm 898: Efficient multiplication of dense matrices over GF(2)", ACM TOMS 37, 2010). For `C = A·B` the rows of `B` are taken in groups of `k`. The `2^k` sums of a group's rows are built into a table in Gray-code order, entry `g ^ (g >> 1)` the one before it plus the row of bit `trailing_zeros(g)`, so each entry is one row addition; each row of `C` then adds the entry that its row of `A` selects with its `k` bits in the group's columns. Eight tables are built at a time, so that a row of `C` is visited once for eight groups, and the columns of `B` and `C` are taken in blocks narrow enough that the eight tables fill about half a megabyte. `k = ⌊log₂ m⌋ − 1`, from 1 to 8; below eight rows `k` is 1 and the product is the schoolbook one, a row of `B` added for each 1 of `A`. A product of `m × l` by `l × n` costs about `m·l·n / (64·k)` word additions and `(l / k)·2^k·n / 64` for the tables, where the schoolbook product takes `m·l·n / 128`.
- **Strassen–Winograd.** While all three dimensions are at least the cutoff (3072 for `*`), the leading part whose dimensions are multiples of 128 is split into halves at multiples of 64 columns, so that no word is shifted, and multiplied by Winograd's form of Strassen's method, seven products of halves and fifteen sums (over `GF(2)` a difference is a sum); the fewer than 128 rows, inner columns and columns beyond it are peeled off and multiplied by M4RM, as M4RI's `mzd_mul` does. The cutoff 3072 keeps every half at 1536 or more: on the machine it was chosen on, halves of 1024 cost more than they save and halves of 1536 to 2048 about break even. With the recursion the product is `O(n^log₂7)`.
- **Elimination, M4RI** (Bard, "Accelerating cryptanalysis with the Method of Four Russians", 2006; Albrecht and Bard's M4RI library). The columns are taken in strips of `k = ⌊log₂ m⌋ − 1` columns, from 4, at the 32 rows below which the elimination is classical, to 16, which it reaches at `2^17` rows; the table of `2^k` entries then costs at most half the additions that clear the rows and holds at most half as many words as the matrix. In a strip, up to `k` pivots are found among the rows not yet used by classical elimination on the strip's bits alone: a row is the pivot of a column when its strip bits, reduced by the strip's pivots so far, have that column's bit set, and only the row chosen is reduced in full. Each row's strip bits are read at most once a strip, when a column first needs a row past those read, and the rows whose bits are not zero are kept in a list, reduced lazily by the pivots found since: a dense strip reads a few rows, and a column without a pivot costs a pass over that list and, the first such column of a strip, a read of the rows not yet read. A row whose strip bits are zero is searched once for its first 1 among the columns that may still hold a pivot (those of `A` for `solve` and `inverse`); the column is kept, and the row costs one comparison a strip until the strips reach it, as nothing changes the row before then. A row with no 1 left in them (a zero row, or in `solve` and `inverse` a row of `[A | B]` zero in `A` but not in `B`) can give no pivot: it is moved below the rows in play and not read again. A strip that finds no pivot has read every row in play, so the strips go on from the first 1 of any of them. The pivot rows are moved to their places when the strip's search is done. So a row costs O(1) in a strip it takes no part in, besides one search of its words for its first 1 after each run of zeros, and an elimination costs about its strips that find pivots, each a pass over the rows in play. The pivot rows are reduced among themselves to the identity on their pivot columns, the `2^k̄` sums of the `k̄` pivot rows are built into a table in Gray-code order, indexed by a row's strip bits with the columns without a pivot masked off (so a strip's pivots need not be adjacent), and every other row is cleared of the pivot columns by adding the one entry its own bits select. A strip costs about `2^k + m` row additions where classical elimination takes `k·m / 2`: `O(n³ / log n)` word operations in all. Below 32 rows the elimination is classical, a column at a time. `rank()` and `det()` clear the rows below the pivots only; the reduced form clears those above too.
- **Solutions, inverses, null spaces.** `solve(&b)` brings `[A | B]` to its reduced echelon form with the pivots chosen in the columns of `A` alone, the columns of `B` starting at the next word so that nothing is shifted: the system is solvable exactly when the rows past the rank are zero in `B`, and row `pivots[i]` of `X` is then row `i` of the right half. `inverse()` is the same with `B = I` and a pivot in every column, as M4RI's `mzd_inv_m4ri`. `null_space()` takes, for each column `f` without a pivot, the vector with a 1 in row `f`, the element in column `f` of row `i` of the reduced form in row `pivots[i]`, and zeros elsewhere: the basis that sympy's `nullspace` gives.
- **Transposes** by blocks of 64 × 64 bits, each transposed in its 64 words by six rounds of swaps of halves (Warren, *Hacker's Delight*, 2nd ed., 7–3): round `s` (32, 16, …, 1) exchanges bit `s` of the row index with bit `s` of the column index; `n²/64` words read and written, `6·32` steps a block.

### Example

```rust
use a::math::matrix::{BitMatrix, Matrix};
use a::math::numbers::Number;

// An integer matrix of determinant 4: invertible over the rationals, singular modulo 2.
let m = Matrix::from_rows(&[[1, 3, 0], [2, 1, 1], [3, 0, 1]]).map(|&x| Number::from(x));
assert_eq!(m.det(), 4);
let a = BitMatrix::from(&m);
assert_eq!(a, BitMatrix::from_rows(&[[true, true, false], [false, true, true], [true, false, true]]));
assert_eq!(a.rank(), 2);
assert!(!a.det() && a.inverse().is_none());

// The third row is the sum of the first two.
let (r, pivots) = a.reduced_row_echelon_form();
assert_eq!(pivots, [0, 1]);
assert_eq!(r, BitMatrix::from_rows(&[[true, false, true], [false, true, true], [false; 3]]));
let n = a.null_space();
assert_eq!(n, BitMatrix::from_rows(&[[true], [true], [true]]));
assert_eq!(&a * &n, BitMatrix::zeros(3, 1));

// A·x = b is solvable exactly when b₃ = b₁ + b₂; the solution has 0 in the free row.
let b = BitMatrix::from_rows(&[[true], [true], [false]]);
assert_eq!(a.solve(&b), Some(BitMatrix::from_rows(&[[false], [true], [false]])));
assert_eq!(a.solve(&BitMatrix::from_rows(&[[true], [false], [false]])), None);

// The upper triangle of ones is I + N + N² + … for the shift N, so its inverse is I + N.
let u = BitMatrix::from_fn(100, 100, |i, j| i <= j);
let v = u.inverse().unwrap();
assert_eq!(v, BitMatrix::from_fn(100, 100, |i, j| j == i || j == i + 1));
assert_eq!(&u * &v, BitMatrix::identity(100));

// A product by M4RM against the definition, and by Strassen–Winograd from 128.
let x = BitMatrix::from_fn(300, 200, |i, j| (i * 31 + j * 17) % 7 < 3);
let y = BitMatrix::from_fn(200, 260, |i, j| (i ^ j) % 3 == 0);
let p = &x * &y;
for (i, j) in [(0, 0), (299, 259), (57, 31)] {
    let s = (0..200).fold(false, |s, k| s ^ (x.get(i, k) & y.get(k, j)));
    assert_eq!(p.get(i, j), s);
}
assert_eq!(x.mul_strassen(&y, 128), p);
assert_eq!(p.transpose(), &y.transpose() * &x.transpose());
```

### Verified

- **Against M4RI 20260122.** `lib/tests/math/matrix/data/gen_gf2_m4ri.c` builds seeded matrices (SplitMix64, mirrored word for word by the test; `gen_gf2.h`) of the sizes 0, 1, 63, 64, 65, 127, 128, 129, 1000 and 2048, square and `n × (n + 65)` and `(n + 65) × n`, of five kinds: uniform, sparse (each bit 1 with probability 1/8), deficient (every fourth row the sum of two before it, every third column the sum of the two before it), of low rank (a product `X·Y` of inner dimension `min(m, n) / 3`) and invertible sparse (a sparse unit upper triangle with its rows shuffled). It writes `gf2_reference.txt` (235 lines, 13 KB): 35 products by `mzd_mul_m4rm`, each equal to `mzd_mul`'s, of the squares, of `sparse · deficient`, and of 16 other shapes from `0 × 5 · 5 × 7` and `1 × 2048 · 2048 × 1` to `2113 × 2177 · 2177 × 2100` and `4096 × 4096 · 4096 × 4096`; 100 reduced echelon forms by `mzd_echelonize_m4ri`, each equal to `mzd_echelonize_pluq`'s, with their ranks and the canonical bases of their null spaces, checked there to give `A·N = 0` and to have as many columns as `mzd_kernel_left_pluq`'s kernel; 40 inverses by `mzd_inv_m4ri`, checked there by `A·A⁻¹ = I` (25 invertible, 15 singular); and 60 systems `A·X = B` with the ranks of `A` and `[A | B]` and the solution from the reduced form of `[A | B]` (45 solvable, 15 not, every right-hand side `A·X₀` among the solvable). The matrices with no elements are left to their definitions there, not to M4RI. Each output is stored as a 64-bit hash of the shape and the words, which the test recomputes from its own output: the outputs are identical, all 235 lines, the 4096 product in a test that runs with `--ignored`.
- **Against NTL 11.6.** `gen_gf2_ntl.cpp` rebuilds the same matrices and recomputes with `mat_GF2` every product, rank (`gauss`), kernel dimension (`kernel` of the transpose), determinant and inverse (`inv`), and the ranks of `A` and `[A | B]`: 416 checks, no disagreement.
- **Against sympy 1.14.** `gen_gf2_sympy.py` rebuilds the matrices of up to 194 rows and columns and recomputes with `DomainMatrix` over `GF(2)` the reduced echelon forms and their ranks, the null spaces (the basis of `nullspace()` is the same canonical one), the determinants and inverses, and the solutions from the reduced form of `[A | B]`: 379 checks, no disagreement.
- **In the tests, independently.** Every reference product against a schoolbook product written in the test, and 60 random products of up to 129 rows and columns element by element against `Σ a_ik·b_kj`; the 8 reference products that run by default and have all three dimensions at least 128 (128³, 129³ and 1000³ twice each, 2048³ and `2113 × 2177 · 2177 × 2100`; the other 26 have a dimension below 128, which the recursion never halves) again by `mul_strassen` from 128, down to halves of 64, and from 300 (the hashes again identical); 40 products of random shapes of 128 to 600 in each dimension, with and without remainders to peel, by `mul_strassen` from 128, 129 and 200, down to halves of 64, against the schoolbook product; and a `2112 × 2112 · 2112 × 2177` product by two levels of it, every dimension peeled at the first, against the product of thin matrices taken the other way round; each reference echelon form against the definition (zeros before each pivot, a single 1 in each pivot column, zero rows past the rank, ascending pivots), `echelonize` in place against it, the null space by `A·N = 0` and `rank(N) = n − rank`, the rank of `Aᵀ`, and up to 200 rows and columns against a Gauss–Jordan elimination on `bool`s written in the test; each inverse by `A·A⁻¹ = A⁻¹·A = I` and `det` against its existence; each solution by `A·X = B`. 400 random matrices of up to 69 × 139, on both sides of the switch to M4RI at 32 rows and across the words' boundaries, of the five kinds, against the Gauss–Jordan elimination: the reduced form and pivots, the rank, the null space element by element against the canonical basis, a solution of a right-hand side from the image with zeros in the free rows, solvability of a random one against the rank of `[A | b]`, and, square, the inverse against the right half of the elimination of `[A | I]`. Transposes element by element and `(A·B)ᵀ = Bᵀ·Aᵀ`; the padding bits zero after every operation, on shapes of 1 to 200 columns; equal matrices built in different orders equal and hashed alike, and shapes with no elements told apart; parities of numbers of 8000 bits of either sign; edge cases (`0 × 0`, `0 × n`, `n × 0`, `1 × 1`, zero, identity and all-ones matrices); every documented panic; and the cost of three eliminations against one of the same work in the same test, the fastest of three runs each, each held within 20 times its reference: 32 rows whose bits are all in the last 64 of 65536 columns against the same rows with those bits first (measured 0.08 times; 206 times for a search that read such rows again in every strip), an unsolvable wide system whose rows zero in `A` are not zero in `[A | b]` against a solvable one (1.0 times; 161 times), and an unsolvable 1024 × 131072 system whose odd rows have one 1 each, spread over all the columns, and whose even rows are zero, against its odd rows alone, so that strips with pivots keep coming after the rows without a 1 in `A` are known (1.8 times; 89 times with those rows kept in play).
- **Every path.** The whole suite also passes with the elimination forced to be classical at every size, and forced to be M4RI at every size with strips of 1, 2, 3, 8 and 16 columns; with the M4RM groups fixed at 1, 2, 3, 5 and 8 rows; with Strassen–Winograd from 128; with one table and blocks of a single word, and with three tables of 3000 words. Deliberate errors in the clearing of the rows above the pivots, the Gray-code index, the reduction of earlier pivot rows, the strip bits read across two words, the lazy reduction of the search's list, the bound on a row's first 1 (set a word too far, or compared a column short), the jump past strips without a pivot, the placing of the pivot rows after the search and the following of the rows without a 1 through it, the consistency check of `solve`, one of Winograd's sums, the peeling and the transposes' masks are each caught.
- 43 tests (one ignored) run in `cargo test --test math -- matrix::gf2`, `--include-ignored` for the 4096 product; all 43 pass on x86_64 too (Ubuntu 26.04, Rust 1.100, emulated). In `lib/tests/math/matrix/data`, `cc -O2 gen_gf2_m4ri.c -I/opt/homebrew/include -L/opt/homebrew/lib -lm4ri -o gen_gf2_m4ri && ./gen_gf2_m4ri > gf2_reference.txt` regenerates the references; `c++ -O2 -std=c++17 gen_gf2_ntl.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lntl -lgmp -o gen_gf2_ntl && ./gen_gf2_ntl < gf2_reference.txt` and `python3 gen_gf2_sympy.py < gf2_reference.txt` check them against NTL and sympy.
### Limits

The matrices are dense; nothing here exploits sparsity. Elimination is M4RI alone, `O(n³ / log n)`: there is no PLE or PLUQ decomposition by blocks with fast products (M4RI's `mzd_echelonize_pluq`, `O(n^ω)`), which is asymptotically faster for large matrices, and `inverse()` and `solve` eliminate the augmented matrix, `n + 64·⌈n/64⌉` columns wide for the inverse. The strip width stops growing at 16 columns, from `2^17` rows, and the groups of M4RM at 8 rows, from 512 rows: a product too thin for Strassen–Winograd costs about a quarter of the schoolbook product's row additions however large it is. The parameters (the strip width, the group size, the eight tables, the half megabyte of tables and the cutoff 3072) were chosen by the method's cost and one machine's caches, not tuned, and the kernels are portable code without SIMD. Strassen–Winograd allocates at each level the quarters of the operands, their eight sums and the seven products, a few times the size of the operands. `solve` gives one solution, the one with zeros in the rows of the free columns; the others are it plus the null space. There are no views of blocks and no column operations: a column operation is a row operation of the transpose.

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
- The product of matrices of numbers is Strassen–Winograd's recursion, as for `ModMatrix` (above), from a crossover that falls as the elements grow, by the length of the longest element of the two operands: 128 rows and columns up to 64 bytes, 32 up to 256 bytes, 16 beyond. A sum of numbers costs about as much as a product while they are short, so the seven products save less there; the crossovers were chosen by measurement. Below the crossover, and at the leaves of the recursion, it is the sums of the products of the elements. The numbers are those of the classical product, little-endian as it leaves them.
- The product of matrices of rationals clears the rows of `A` and the columns of `B` of their denominators, multiplying them by the least common multiples `l_i` and `r_j`, multiplies the integers, in 64-bit registers when the lengths of the elements and of the sums allow it, in 128 bits when they allow that, and then divides each element once, by `l_i·r_j`, with one greatest common divisor: about `n³` products of integers and `n²` greatest common divisors, where the products of rationals would take a gcd or two each. When every element is in words and the cleared integers fit in them, all of it runs on the stack, each rational's words read in place. Otherwise a product whose inner dimension is 2, or 4 with denominators of more than 32 bits, is the sums of the products of rationals, which are done in registers when their parts fit.
- On `Number`: Bareiss's elimination, `a_ij ← (a_ij·a_kk − a_ik·a_kj) / p` with the previous pivot `p` dividing exactly, takes about `n³/3` steps, each two products and an exact division of numbers no longer than the determinant, the minors of the matrix; Berkowitz's method about `n⁴/2` products; the adjugate is `(−1)ⁿ⁻¹·(Aⁿ⁻¹ + c₁·Aⁿ⁻² + … + cₙ₋₁·I)` by Cayley and Hamilton, from the characteristic polynomial's coefficients `c`, for singular matrices too; the null space comes from the fraction-free echelon form, one vector for each column without a pivot.
- Solutions, on `Number` and `Rational`: Bareiss's elimination of `[A | B]` to an echelon form whose last pivot is `D = ±det(A)`, then a fraction-free back substitution, `y_i = (D·b_i − Σ u_ij·y_j) / u_ii` from the bottom, whose integers `y = D·x` are Cramer's numerators, every division exact; then one reduction of each element `y_i / D` by a greatest common divisor. The inverse is the solution of `A·X = I`, whose integers `D·A⁻¹` are also the adjugate up to the sign of the row swaps, which `inverse()` of numbers returns. A matrix of rationals has each row multiplied by the least common multiple `l_i` of its denominators (of `A` and `B` together for `solve`), which leaves the solutions and the null space as they were and multiplies the determinant by `∏ l_i`; `A⁻¹` is then `(L·A)⁻¹·L`.
- Matrices of up to 8 rows, of integers of one limb or of rationals whose parts are in words, run on the stack, with nothing allocated but the result. The rows are cleared into an array of `i64`s, and one product of the squares of the rows' norms, Hadamard's bound, tells whether every minor stays below `2^61`. If it does, the elimination and the back substitution run with no check at all. A step is two products in 128 bits, a subtraction, and an exact division by the previous pivot: the processor's division when the pivot divides one or two elements, and otherwise a product by the inverse of its odd part modulo `2^64`. Order 2 is Cramer's rule in 128 bits.
- Beyond that, the integers of the elimination are held in registers while their magnitudes are below `2^63`: a step is then two products in 128 bits, a subtraction, and an exact division by a product with the inverse of the divisor's odd part modulo `2^64`, with a comparison telling whether the quotient fits. Beyond the registers, each element is a vector of limbs that the steps write over, the products go to two buffers of working memory, and the exact quotient is Hensel's, from the low end. From quotients of 150 limbs it is a product by the inverse of the divisor modulo a power of `2^64`, computed once for the elements that the same pivot divides. A step then allocates nothing once the vectors have grown.
- Large systems, on `Number` and `Rational`: from 24 rows (16 when an integer of the cleared matrix passes 63 bits), a system with at most an eighth as many right-hand sides as rows is solved by Dixon's p-adic lifting. `A` is factored once modulo a prime `p` just below `2^62`, in Montgomery's form, the products of the substitutions summed four at a time before a reduction; each step solves `A·x ≡ r (mod p)` and lifts the right-hand side, `r ← (r − A·x) / p`, exactly, the rows of `A` multiplied by the digits limb by limb; after `L` steps, `p^L` past twice the product of Hadamard's bounds on Cramer's numerators and on the determinant, the digits are joined by halves into `X mod p^L`, and each element is the one fraction within the bounds that is congruent to it, found by the extended Euclidean algorithm (Wang's rational reconstruction) with Lehmer's steps in registers, as a rule for the first element alone: the others are their product by the common denominator so far, when that is within the bounds. The prime is the first, from the top down below `2^62`, modulo which `A` is nonsingular, and a singular matrix is proved singular (below). The work is `O(n³ + L·n²)` operations on words instead of `n³/3` products of numbers as long as the minors.
- The determinant of `Matrix<Number>` (and of the integers a `Matrix<Rational>` is cleared to) is computed by one of three methods, chosen by the order `n` and the length of the longest integer where each was measured to be the fastest: Bareiss's elimination below the band of the table, in registers while the minors fit; the residues modulo primes in the band; Dixon's divisor times residues above it. Up to 8 rows in registers on the stack, when the minors stay below `2^61` (above).

| Longest integer | Bareiss's elimination below | the residues from there to (not included) | Dixon's divisor from |
| --- | --- | --- | --- |
| up to 24 bits | 28 rows | (none) | 28 rows |
| 25 to 63 bits | 20 | 28 | 28 |
| 64 to 192 bits | 20 | 32 | 32 |
| 193 to 384 bits | 16 | 40 | 40 |
| 385 to 768 bits | 16 | 56 | 56 |
| 769 to 1536 bits | 16 | 80 | 80 |
| 1537 to 3072 bits | 12 | 112 | 112 |
| 3073 to 6144 bits | 12 | `3·⌈bits/64⌉`, at least 160 | the same |
| 6145 to 12 288 bits | 14 | the same | the same |
| up to 16 384 bits | 16 | the same | the same |
| up to 24 576 bits | 20 | the same | the same |
| up to 32 768 bits | 24 | the same | the same |
| up to 65 536 bits | 28 | the same | the same |
| up to 131 072 bits | 36 | the same | the same |
| up to 262 144 bits | 40 | the same | the same |
| up to 600 000 bits | 80 | the same | the same |
| beyond | `bits / 7500` | the same | the same |

The edges up to 262 144 bits were measured (Apple M4 Max, release profile, random matrices, the methods at the orders around each edge); the edge at 600 000 bits is from the growth of the two methods' times between 12 and 24 rows there (the residues `n³`, the elimination `n^3.5`), and beyond it the growth is continued. Dixon's divisor for long integers is taken from three rows a limb, where its lifting (`2·n·L` steps of `n²·(L + 1)` products of words) overtakes the residues (`n·L` primes of `n²·L + n³/3`), as the measured upper edges at 1024 to 4096 bits bear out.
- Multimodular determinant: `det(A) mod p` for word primes `p` taken down from `2^62` (each above `2^61`), by the elimination over `GF(p)` above (the PLE decomposition), until the primes' product passes `2^(h + 1)`, `h = ⌈bits(D²)/2⌉` for the smaller of the products `D²` of the squared lengths of the rows and of the columns (Hadamard's bound, `|det A| ≤ D`); then `Number::crt` and the residue of least magnitude. The primes stop at the bound, never when the residues seem to settle, so the result is certain for every matrix, singular ones included; a zero row or column gives 0 at once. For `n × n` integers of up to `L` limbs that is about `c ≈ 64·n·L/61` primes, and for each prime `n³/3` products of words for the elimination and the residues of the `n²` integers. Up to 511 limbs the residues are taken one prime at a time, `Σ limb_i·(2^(64·i) mod p)`, `L` independent products of words an integer, so `n²·L·c ≈ n³·L²` in all, which passes the eliminations' `n³·c/3 ≈ n⁴·L/3` once `L > n/3`. From 512 limbs they come from a remainder tree over the primes (as FLINT's `fmpz_comb` does): the primes in runs of 16, the products of the runs paired up a tree, and each integer reduced down it modulo each node, about `n·M(L)·log L` an integer for all the primes (`M(L)` a product of `L` limbs), so the whole is about `n³·M(L)·log L + n⁴·L/3` products of words. Bareiss's elimination is `n³/3` steps on numbers of up to `n·L` limbs, quasi-linear in `L`; that is why the band's lower edge rises with the length of the integers.
- Dixon's divisor: the least common multiple `l` of the denominators of the solution of `A·x = b` for a fixed pseudorandom `b` by the lifting, which divides `det(A)` and is most of it as a rule, then `det(A) / l` from the determinants modulo primes, the product of the factors' diagonal, by the Chinese remainder theorem, until the primes' product passes twice Hadamard's bound over `l`: one prime, as a rule. The residues of `det(A) / l` are taken modulo the primes after the lifting's, passing over those that divide `l`. This is FLINT's `fmpz_mat_det_modular_accelerated`.
- Singular matrices are proved singular, cheaply and for certain, wherever Dixon's lifting is used (`solve` of `Matrix<Number>` and `Matrix<Rational>` from 24 rows, 16 when an integer passes 63 bits, and `det` in the band of Dixon's divisor) and, from the same orders as `solve`, before the elimination of a system with more than an eighth as many right-hand sides as rows and of the inverses (`Matrix<Number>::inverse` and `inverse_rational`, `Matrix<Rational>::inverse`): the lifting's prime is the first, from the top down below `2^62`, modulo which `A` is nonsingular. When `A` is singular modulo the four largest, its rank profile modulo the last of them gives a block `A[R, Q]` nonsingular modulo that prime and a column `c` outside `Q`; the lifting solves `A[R, Q]·y = −A[R, c]` over the rationals, and `w`, `d·y` on `Q` and the common denominator `d` at `c`, is checked against every row of `A` exactly. `A·w = 0` proves `A` singular (it holds whenever the rank modulo the prime is the rank, as `A`'s other rows are combinations of the rows `R`). Otherwise the search goes on with the next primes, lifting modulo the first one modulo which `A` is nonsingular (a nonsingular `A` whose determinant the first primes divide), trying a null vector again whenever the rank modulo a prime is the highest so far, and stops with `A` singular once the primes modulo which it was singular have a product past twice Hadamard's bound.

### Verified: products, determinants and systems of integers

For the exact algorithms that came with 0.2.43, above (the products of numbers, the choice of the determinant's method, the residues and the proofs of singularity), the tests are in `lib/tests/math/matrix/strassen.rs` (5 tests), `multimodular.rs` (5) and `dixon.rs` (8), with `data/multimod_flint.txt` and `data/dixon_flint.txt` (3.9 KB together), written by `data/gen_multimod.c` and `data/gen_dixon.c` with FLINT 3.6; the tests of `exact.rs` and `rational.rs` from before run through the new dispatch. Which method each input takes was checked on a copy of the crate with a mark on each path, running these tests in the debug profile.

- Strassen–Winograd on numbers against the classical product written in the test, with elements of both signs and byte orders and some zeros: 300-byte elements at `15 × 16 × 17`, `16³`, `17³`, `16 × 17 × 16`, `33 × 32 × 35` and `32³` (two levels); 100-byte elements at `31 × 32 × 33`, `32³`, `33³`, `40 × 35 × 33` and `64 × 64 × 65`; 1- and 8-byte elements at `129 × 128 × 130`; a zero dimension or a single row or column; every result equal and little-endian; and `det(A·B) = det(A)·det(B)` at `20 × 20` with 300-byte elements.
- The determinant against FLINT's `fmpz_mat_det` on 58 seeded matrices chosen in each method's band (`data/gen_multimod.c` says which): random, singular (the last row the sum of the first two) and `L·U` with unit triangular factors (determinant 1, elements about twice as long); the residues modulo `2^61 − 1` and `2^64 − 59`, the sign and the length in bits all agree. By the dispatch, 10 go through Bareiss's elimination (`9 × 9` to `19 × 19`), 27 through the residues (from `24 × 24` of 30 bits through `30 × 30` of 100, `36 × 36` of 250, `48 × 48` of 500, `20 × 20` of 1400 and `32 × 32` of 2000 bits, a determinant of 64 026 bits, to `16 × 16` of 4000, and 4 with a first row of 34 000 or 40 000 bits at 32 and 36 rows, whose residues come from the remainder tree), and 21 through Dixon's divisor (`30 × 30` of 8 bits to `100 × 100` of 8, among them `60 × 60` of 100 bits; the 7 singular ones proved singular by a null vector). All 58 run in the debug profile.
- Against Bareiss's elimination (the determinant `inverse()` returns beside the adjugate, from its own elimination of `[A | I]`) and, to 16 rows, Berkowitz's characteristic polynomial, on matrices at the edges of the bands and inside them (`19 × 19` and `20 × 20` of 64 bits, `31 × 31` of 100, `16 × 16` of 300, `20 × 20` of 512, `11 × 11` and `12 × 12` of 1600, `15 × 15` of 1000), with their residues by `ModMatrix` modulo 2, 3, 1 000 003 and `2^64 − 59`, `det(−A)` and a swap of two rows; singular matrices (a combination of two rows, a zero row, a zero column, rank `n − 2`) at 24 rows of 64 and 300 bits; elements at the edges of the registers (`2^63 − 1`, `2^63`, `2^64 − 1`, `−2^63`, `−(2^64 − 1)`, near them, 0 and 1); and a `Matrix<Rational>` whose cleared integers fall in a band, `det(A/d) = det(A)/dⁿ`.
- Dixon against FLINT's `fmpq_mat_solve_fmpz_mat_dixon` and `fmpq_mat_solve_dixon` on 34 seeded systems: 14 of integers from `16 × 16` to `100 × 100`, with 1 to 12 right-hand sides (below and past an eighth of the rows, so that both the lifting and the elimination run) and elements of 8 to 1000 bits, their 14 singular twins, and 6 of rationals with denominators up to 1000 from `16 × 16` to `100 × 100`. The least common multiple of the denominators (modulo `2^61 − 1` and its length in bits) and a digest of every numerator and denominator agree with FLINT's for all of them, `None` for every singular one, and every solution satisfies its system exactly. Also: rank-deficient systems (rank `n − 1`, `n − 3` and 1 at 30 rows, with a consistent right-hand side too) give `None`; a `24 × 24` system whose determinant is the product of the four primes the lifting tries first is lifted modulo the fifth; singular systems of 40 rows of 8 and 200 bits, with 1 and 6 right-hand sides (the lifting, and the check before the elimination), of integers and of rationals, and their inverses (`inverse`, `inverse_rational`, `Matrix<Rational>::inverse`), proved singular by a null vector, and a matrix with a zero row by Hadamard's bound; a singular `30 × 30` matrix whose rank drops further modulo the fourth prime, whose first null vector fails the check and which the fifth prime proves singular; the determinant's residue prime that divides the divisor `l` (32 rows, the first column `p₂·b` for the lifting's own `b`, so that `l = p₂`), passed over, and the one that divides the determinant but not `l` (28 rows, the first column `b`, `l = 1`), whose residue is 0, both against Bareiss's determinant; systems with no right-hand side; elements of 4000 bits at 24 rows, whose common denominator divides the determinant. On the marked copy every one of these paths was reached in the debug profile, and so was Strassen–Winograd's product into a temporary inside the elimination (15 times).
- Dixon's lifting (bounds, steps, reconstruction) was audited: its bounds hold (the numerator bound is Hadamard's `D·|b|/min|a_j|`, rounded up; the steps make `p^L > 2·N·D`; the shortcut that takes an element as `x·den` when that is within `N·den` is sound because the least common multiple of `den` and the element's denominator divides `det(A)`; the fractions of Wang's reconstruction are unique under the same bound).

To reproduce: `cargo test --test math -- matrix::strassen matrix::multimodular matrix::dixon`; the generators have their build lines and rewrite their files exactly.

### Limits: products, determinants and systems of integers

- The edges of the determinant's bands were measured on one shared machine (an Apple M4 Max, loaded by other work: single timings varied by up to two times), on random matrices; the edge at 600 000 bits and those beyond are extrapolated, and so is the upper edge for long integers (three rows a limb), from a cost model that matched the measured edges at 1024 to 4096 bits.
- The residues are computed up to Hadamard's bound, even when the determinant is far below it (a matrix of determinant 1 costs as much as a random one); FLINT's `fmpz_mat_det` can stop earlier with Dixon's divisor, which this crate uses above the bands.
- For integers much longer than the order (for 600 000 bits below about 80 rows), Bareiss's elimination stays the method; the residues, even from the remainder tree, cost more there.
- Strassen–Winograd on numbers saves little while the elements are short, where a sum costs about as much as a product; the product of matrices of rationals does not use it.
- Systems with more than an eighth as many right-hand sides as rows, and the inverses, are solved by Bareiss's elimination once the matrix is known to be nonsingular, not by the lifting.

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

Measured on an Apple M4 Max on 2026-11-12, one thread, with nothing else running (load average 2.2 to 5.9), `env -u MallocNanoZone`, fat LTO, in µs (or ms and s), the median of 5 rounds of fresh random matrices, interleaved in one process. The references are FLINT 3.6's defaults (`fmpz_mat_det`, `fmpq_mat_solve_fmpz_mat`, `fmpz_mat_inv` then `fmpq_mat_set_fmpz_mat_div_fmpz`), FLINT's Bareiss alone (`fmpz_mat_det_bareiss`, `fmpz_mat_solve_fflu`), PARI 2.19 (`det`, `RgM_solve`, `RgM_inv`), and this crate's 0.2.34, before matrices of up to 8 rows ran on the stack. Each one's time over `a`'s is in brackets. The tables for matrices of rationals are on the [rational numbers](rational.md) page. These times are from before 0.2.43, which changed some of the methods measured here (the determinants by residues modulo primes, Strassen–Winograd's products of numbers, the check for singularity before the larger inverses); they will be measured again for the release.

`Matrix<Number>::det()`, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.02 µs | 0.05 µs (2.13) | 0.01 µs (0.22) | 0.09 µs (3.70) | 0.02 µs (0.79) |
| 3 | 0.05 µs | 0.08 µs (1.86) | 0.01 µs (0.33) | 0.15 µs (3.26) | 0.08 µs (1.69) |
| 4 | 0.07 µs | 0.13 µs (1.98) | 0.04 µs (0.54) | 0.23 µs (3.47) | 0.70 µs (10.44) |
| 5 | 0.11 µs | 0.19 µs (1.77) | 0.26 µs (2.47) | 0.33 µs (3.05) | 0.96 µs (9.00) |
| 6 | 0.16 µs | 0.28 µs (1.77) | 0.41 µs (2.63) | 0.52 µs (3.36) | 1.47 µs (9.50) |
| 8 | 0.25 µs | 0.51 µs (1.99) | 0.88 µs (3.47) | 1.05 µs (4.12) | 3.10 µs (12.17) |
| 16 | 6.86 µs | 6.92 µs (1.01) | 7.80 µs (1.14) | 8.38 µs (1.22) | 37.1 µs (5.42) |
| 32 | 109.4 µs | 109.6 µs (1.00) | 69.3 µs (0.63) | 178.3 µs (1.63) | 222.2 µs (2.03) |
| 64 | 472.6 µs | 480.8 µs (1.02) | 575.3 µs (1.22) | 2.91 ms (6.16) | 1.89 ms (4.01) |
| 128 | 3.11 ms | 3.15 ms (1.02) | 3.67 ms (1.18) | 40.2 ms (12.93) | 18.8 ms (6.05) |

`Matrix<Number>::det()`, 64-bit integers (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.13 µs | 0.12 µs (0.98) | 0.02 µs (0.17) | 0.16 µs (1.25) | 0.02 µs (0.15) |
| 3 | 0.37 µs | 0.32 µs (0.86) | 0.07 µs (0.19) | 0.32 µs (0.87) | 0.09 µs (0.25) |
| 4 | 0.78 µs | 0.75 µs (0.95) | 0.22 µs (0.28) | 0.65 µs (0.83) | 5.45 µs (6.96) |
| 5 | 1.62 µs | 1.59 µs (0.98) | 1.05 µs (0.65) | 1.27 µs (0.79) | 8.49 µs (5.25) |
| 6 | 2.87 µs | 2.82 µs (0.98) | 2.01 µs (0.70) | 2.37 µs (0.83) | 12.5 µs (4.34) |
| 8 | 7.40 µs | 7.33 µs (0.99) | 5.82 µs (0.79) | 6.37 µs (0.86) | 28.4 µs (3.83) |
| 16 | 89.2 µs | 89.2 µs (1.00) | 81.0 µs (0.91) | 83.8 µs (0.94) | 166.8 µs (1.87) |
| 32 | 574.5 µs | 569.7 µs (0.99) | 525.4 µs (0.91) | 1.34 ms (2.33) | 1.54 ms (2.67) |
| 64 | 3.02 ms | 3.01 ms (1.00) | 3.28 ms (1.08) | 27.9 ms (9.23) | 13.6 ms (4.50) |

`Matrix<Number>::solve(&b)`, one column, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, `map(Rational::from)` first | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.04 µs | 0.16 µs (4.09) | 0.08 µs (2.01) | 0.06 µs (1.49) | 0.21 µs (5.37) | 1.45 µs (36.45) |
| 3 | 0.09 µs | 0.26 µs (2.79) | 0.15 µs (1.65) | 0.11 µs (1.21) | 0.32 µs (3.49) | 1.74 µs (18.78) |
| 4 | 0.15 µs | 0.37 µs (2.53) | 0.24 µs (1.63) | 0.39 µs (2.64) | 0.45 µs (3.08) | 2.42 µs (16.52) |
| 5 | 0.23 µs | 0.53 µs (2.27) | 0.35 µs (1.52) | 0.55 µs (2.39) | 0.64 µs (2.75) | 5.51 µs (23.76) |
| 6 | 0.36 µs | 0.73 µs (2.04) | 0.51 µs (1.42) | 0.80 µs (2.24) | 0.96 µs (2.67) | 7.66 µs (21.32) |
| 8 | 0.67 µs | 1.27 µs (1.90) | 0.94 µs (1.41) | 1.54 µs (2.30) | 1.74 µs (2.60) | 13.1 µs (19.49) |
| 16 | 12.8 µs | 13.0 µs (1.01) | 13.5 µs (1.06) | 31.5 µs (2.47) | 16.2 µs (1.27) | 57.3 µs (4.48) |
| 32 | 99.9 µs | 100.3 µs (1.00) | 103.1 µs (1.03) | 127.0 µs (1.27) | 231.3 µs (2.32) | 341.2 µs (3.42) |
| 64 | 454.6 µs | 458.0 µs (1.01) | 468.4 µs (1.03) | 640.4 µs (1.41) | 3.35 ms (7.37) | 5.43 ms (11.95) |
| 128 | 2.52 ms | 2.63 ms (1.05) | 2.58 ms (1.03) | 3.91 ms (1.56) | 43.5 ms (17.27) | 53 ms (21.05) |

`Matrix<Number>::solve(&b)`, one column, 64-bit integers (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, `map(Rational::from)` first | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.59 µs | 0.57 µs (0.97) | 0.63 µs (1.07) | 0.38 µs (0.65) | 0.69 µs (1.17) | 18.9 µs (31.98) |
| 3 | 1.76 µs | 1.83 µs (1.04) | 1.93 µs (1.10) | 1.32 µs (0.75) | 1.86 µs (1.05) | 23.3 µs (13.23) |
| 4 | 3.64 µs | 3.43 µs (0.94) | 3.72 µs (1.02) | 3.36 µs (0.92) | 3.53 µs (0.97) | 48.0 µs (13.18) |
| 5 | 6.11 µs | 6.15 µs (1.01) | 6.28 µs (1.03) | 5.92 µs (0.97) | 6.25 µs (1.02) | 67.7 µs (11.08) |
| 6 | 9.40 µs | 9.32 µs (0.99) | 9.40 µs (1.00) | 9.16 µs (0.97) | 9.65 µs (1.03) | 63.3 µs (6.74) |
| 8 | 19.5 µs | 19.2 µs (0.99) | 19.6 µs (1.00) | 19.3 µs (0.99) | 20.0 µs (1.03) | 244.2 µs (12.53) |
| 16 | 158.0 µs | 157.3 µs (1.00) | 159.8 µs (1.01) | 252.4 µs (1.60) | 155.6 µs (0.98) | 1.24 ms (7.84) |
| 32 | 725.0 µs | 724.6 µs (1.00) | 718.6 µs (0.99) | 973.0 µs (1.34) | 1.84 ms (2.54) | 7.09 ms (9.78) |
| 64 | 3.78 ms | 3.78 ms (1.00) | 3.71 ms (0.98) | 5.72 ms (1.51) | 32.1 ms (8.48) | 49.8 ms (13.16) |

`Matrix<Number>::inverse_rational()` (`inverse()`: the adjugate and the determinant), integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, `map(Rational::from)` first | `a`, `inverse()` | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.04 µs | 0.21 µs (4.74) | 0.06 µs (1.46) | 0.08 µs (1.76) | 0.08 µs (1.76) | 0.26 µs (5.75) | 0.44 µs (9.94) |
| 3 | 0.15 µs | 0.47 µs (3.14) | 0.19 µs (1.30) | 0.18 µs (1.22) | 0.27 µs (1.80) | 0.51 µs (3.42) | 5.78 µs (38.75) |
| 4 | 0.31 µs | 0.85 µs (2.79) | 0.38 µs (1.23) | 0.29 µs (0.96) | 0.79 µs (2.58) | 0.84 µs (2.75) | 11.5 µs (37.63) |
| 5 | 0.61 µs | 1.49 µs (2.45) | 0.69 µs (1.13) | 0.48 µs (0.79) | 1.45 µs (2.37) | 1.46 µs (2.40) | 13.2 µs (21.59) |
| 6 | 1.08 µs | 2.60 µs (2.40) | 1.19 µs (1.10) | 0.70 µs (0.65) | 2.58 µs (2.38) | 2.72 µs (2.51) | 8.88 µs (8.19) |
| 8 | 2.62 µs | 5.94 µs (2.26) | 2.83 µs (1.08) | 1.28 µs (0.49) | 5.66 µs (2.16) | 5.86 µs (2.23) | 18.7 µs (7.14) |
| 16 | 87.8 µs | 87.2 µs (0.99) | 88.2 µs (1.00) | 55.4 µs (0.63) | 214.7 µs (2.45) | 115.2 µs (1.31) | 109.5 µs (1.25) |
| 32 | 1.14 ms | 1.11 ms (0.98) | 1.11 ms (0.97) | 644.6 µs (0.57) | 2.14 ms (1.88) | 1.49 ms (1.31) | 1.19 ms (1.04) |
| 64 | 13.9 ms | 13.8 ms (0.99) | 13.9 ms (1.00) | 8.16 ms (0.59) | 24.9 ms (1.79) | 19.7 ms (1.41) | 9.61 ms (0.69) |
| 128 | 177 ms | 177 ms (1.00) | 178 ms (1.00) | 127 ms (0.72) | 259 ms (1.46) | 251 ms (1.42) | 98.5 ms (0.56) |

`Matrix<Number>::inverse_rational()` (`inverse()`: the adjugate and the determinant), 64-bit integers (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, `map(Rational::from)` first | `a`, `inverse()` | FLINT | FLINT, Bareiss | PARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.73 µs | 0.82 µs (1.12) | 0.52 µs (0.71) | 0.39 µs (0.53) | 0.38 µs (0.52) | 0.75 µs (1.03) | 0.87 µs (1.19) |
| 3 | 2.84 µs | 2.73 µs (0.96) | 2.86 µs (1.01) | 1.23 µs (0.43) | 1.94 µs (0.68) | 2.73 µs (0.96) | 18.9 µs (6.63) |
| 4 | 8.72 µs | 8.52 µs (0.98) | 8.78 µs (1.01) | 2.78 µs (0.32) | 9.02 µs (1.03) | 9.21 µs (1.06) | 36.8 µs (4.22) |
| 5 | 18.4 µs | 18.3 µs (0.99) | 18.5 µs (1.01) | 5.41 µs (0.29) | 19.6 µs (1.07) | 20.0 µs (1.09) | 52.1 µs (2.83) |
| 6 | 33.4 µs | 33.6 µs (1.01) | 33.7 µs (1.01) | 9.19 µs (0.28) | 37.5 µs (1.12) | 37.6 µs (1.13) | 85.4 µs (2.56) |
| 8 | 87.6 µs | 85.5 µs (0.98) | 86.1 µs (0.98) | 24.4 µs (0.28) | 99.4 µs (1.13) | 98.3 µs (1.12) | 182.6 µs (2.09) |
| 16 | 942.6 µs | 940.4 µs (1.00) | 915.2 µs (0.97) | 294.0 µs (0.31) | 2.86 ms (3.04) | 1.19 ms (1.27) | 1.08 ms (1.15) |
| 32 | 11.2 ms | 11.4 ms (1.01) | 11.3 ms (1.00) | 5.24 ms (0.47) | 28.5 ms (2.53) | 14.9 ms (1.32) | 9.63 ms (0.86) |

- Up to 8 × 8 the matrices of small integers run on the stack: determinants 1.8 to 2.1 times as fast as in 0.2.34, solutions 1.9 to 4.1 times and inverses 2.3 to 4.7 times. From 16 × 16 the times are 0.2.34's, within the noise. So are those of 64-bit integers, which pass Hadamard's bound at once: from 2 × 2 to 4 × 4 the check that turns them away costs up to 14 ns, 2 to 3 % of the time when measured alone (0.2.34 takes 0.86 to 1.12 times as long in the table, the noise of a single pass). Against 0.2.33, measured in #16: the determinant, the inverse as the adjugate and the solutions run on registers and limbs rather than numbers, 1.08 to 27.5 times as fast as 0.2.33's determinant, and `inverse()` is 7 to 83 times as fast as by Berkowitz's polynomial from 8 × 8 (and faster at every size, 1.5 times at 2 × 2).
- FLINT is faster for determinants of small integers up to 4 × 4 (0.22 to 0.54 of the time), by cofactors in words, where the work takes a few nanoseconds and writing the result as a `Number` costs more than that. It is also faster for small integers at 32 × 32 (0.63) and for 64-bit integers from 2 × 2 to 32 × 32 (0.17 to 0.91 for determinants, 0.65 to 0.92 for solutions up to 4 × 4, 0.52 and 0.68 for the inverse at 2 × 2 and 3 × 3), where the integers pass a word and its arithmetic on them is GMP's. From 64 × 64 the determinant and the solutions, by Dixon's lifting here and in FLINT, are 1.08 to 1.56 times as fast as FLINT's. Solutions of small integers are 1.2 to 2.6 times as fast as FLINT's at every size from 3 × 3. PARI's multimodular inverse is faster from 64 × 64 for small integers (0.56 to 0.69 of the time) and at 32 × 32 for 64-bit ones (0.86).

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

- Time complexity: `2·m·n·k` operations for an `m × k` by `k × n` product; for `n × n` matrices, about `2n³/3` for `lu()`, `2n³` for `inverse()`, `n³/3` for `cholesky()`, `4n³/3` for `qr()`, `4n³/3` for `hermitian_eigenvalues()` and `4.7n³` with the eigenvectors, `8n³/3` for `singular_values()` and `9.3n³` for `svd()`, and about `10n³` for `eigenvalues()` and `25n³` for `eigen()`, the QR algorithm's sweeps, whose number depends on the matrix, taking most of it; `exp()` up to six products and one solve, and a product for each squaring; `sqrt()` and `log()` a Schur decomposition, then about `n³/3` for each square root of its triangular factor. On numbers and rationals, `rank()`, and `det()` below its bands, about `n³/3` steps of two products and an exact division of numbers as long as the minors; `det()` by residues about `64·n·L/61` primes for integers of `L` limbs, each `n³/3` products of words and the residues of the `n²` integers, `L` products of words each, or from 512 limbs about `n·M(L)·log L` an integer for all the primes, `M(L)` a product of `L` limbs; the product of numbers `O(n^2.81)` products of elements from its crossover; `solve(&b)` for `k` right-hand sides about `n³/3 + k·n²` and `inverse()` about `4n³/3`, then `n·k` and `n²` greatest common divisors; for the large systems and determinants, Dixon's lifting `n³/3` products modulo a prime and `2·n²` products of words for each of its `L` steps, `L` about the length of the determinant over 61 bits, twice for the solutions; `characteristic_polynomial()` about `n⁴/2` products and `adjugate()` `3n⁴/2`; the Hermite form of an `m × n` matrix of rank `r` about `m·r²` operations on numbers below the determinant `D` of a nonsingular `r × r` minor, the elementary divisors about `m·n·r`, and the transformations as in [Normal forms](#normal-forms). `ModMatrix`: products `m·k·n` products of words, and Strassen–Winograd's `O(n^2.81)` from 128 rows and columns; the PLE decomposition, and what rests on it, about `n³/3` products of words below that and `O(n^2.81)` above. `BitMatrix`: products about `m·l·n / (64·k)` word additions by M4RM, `k = ⌊log₂ m⌋ − 1` from 1 to 8, and `O(n^2.81)` from 3072 rows and columns; the elimination `O(n³ / log n)` word operations.
- Space complexity: a matrix is its elements in one vector, column after column, and its shape; a decomposition keeps its factors (`lu()` the matrix and the permutation, `qr()` the reflections and the triangular factors of their blocks, at most 64 elements per column); working memory of at most a few times the matrix, and a product packs blocks of up to `512 × 1020` elements of `op(B)` (`512 × 2040` for `f32`) and `128 × 512` of `op(A)`. A `ModMatrix` is a word an element, row after row, and a `BitMatrix` a bit an element, `⌈n/64⌉` words a row; the transformations of the normal forms are dense matrices of numbers.
- Soundness: `unsafe` code is the product's kernels and the leaves of the triangular solves and of the Cholesky decomposition, each with the extent of what it reads and writes checked before it is called; Miri runs the portable kernels, which have the same shapes as the NEON ones. The exact algorithms use the numbers' limb kernels through checked slices; the normal forms, `ModMatrix` and `BitMatrix` have no `unsafe` code. A shape that doesn't fit panics; the iterations of the QR algorithms stop after a bounded number of sweeps.

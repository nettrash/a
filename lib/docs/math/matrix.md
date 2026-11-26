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

- `lu()`: `P·A = L·U` by Gaussian elimination with partial pivoting, for any shape: `l()`, `u()`, `p()`, `permutation()`, `det()`, `solve(&b)`, `inverse()` and `is_singular()`; `solve_into(&b, &mut x)`, the same solution to the last bit, without allocating when `b` has at most 16 rows, or at most 128 rows and 1,024 elements (`false` for a singular matrix, `x` left alone), and `refactor(&a)`, which makes the decomposition `a.lu()` in its own memory, allocating nothing for a square matrix of at most 100 rows once a `refactor` of one at least as large has given it its working memory — for solving with one matrix after another, as the Newton iterations of the implicit integrators do.
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

- **Rank profiles.** With `J` the column rank profile and `R` the row rank profile from the bottom (row `i` in it when it is not a combination of the rows below it), `M = A[R, J]` is a nonsingular `r × r` minor. Both profiles come from PLE decompositions modulo a prime `p` below `2^62`, as `ModMatrix::ple` computes them: `J` is the pivot columns of that of `A`, `R` those of `Aᵀ` with its columns reversed (when only some independent rows are wanted, the rows of the first decomposition's pivots), `O(m·n·r^(ω−2))` operations on words; they are then proved over the integers: `X = M⁻¹·A[R, K]` for the columns `K` outside `J` must satisfy `A[S, J]·X = A[S, K]` on the other rows `S` (so the rank is `r`) and be zero where a column of `J` comes after the column it expresses (so `J` is the profile); for the transformation, `Y = A[S, J]·M⁻¹` must be zero where a row of `R` comes before the row it expresses. A prime that fails a check (it divides one of the `m + n` minors the profiles depend on) is followed by the next one below it. `X` is an exact solution with `M` (`solve`, by Bareiss's elimination or Dixon's lifting), `Y` is computed as the integers `N = D·Y`, `D = |det M|`, from their residues modulo primes (as `Z` below), and `D` is `M`'s determinant, or, when only `X` is needed, the product of the pivots of the form of `M` below.
- **The Hermite form of a square nonsingular matrix** (Pernet and Stein, *Fast computation of Hermite normal forms of random integer matrices*, J. Number Theory 130, 2010, after Micciancio and Warinschi, ISSAC 2001), from order 8. For `M` of order `n`, `W₁` its leading block of order `n − 1`, `W₂` that block with the last row of `M` (on the first `n − 1` columns) in place of the one before, `c` and `d` those two rows: one solution `W₁⁻¹·[w | e]` by Dixon's lifting (`w` the last column, `e` the last unit vector) gives a divisor of `d₁ = det W₁` (the least common multiple of its denominators, most of it as a rule, the rest from residues modulo a prime or two) and `d₂ = d₁·(1 + (d − c)·z₂)`. With `g = gcd(d₁, d₂) = s·d₁ + t·d₂`, the block `W` with `s·c + t·d` for its last row has the determinant `g`, as the determinant is linear in a row, and its rows are in the lattice; its Hermite form `H_W` is computed modulo `g` (the identity for `g = 1`), and that of `[W | w']` is `[H_W | H_W·W⁻¹·w']`, `W⁻¹·w'` by Sherman and Morrison's formula from the solution, `W` being `W₁` with one row changed. The last two rows of `M` are then added: each reduced by the rows whose pivots are in its nonzero columns (a subtraction when the pivot divides, an extended Euclidean step otherwise), what is left becoming a row. Last, the elements above each pivot are brought into `[0, pivot)`. For a random matrix `g` is 1 or small and the form is the identity but for its last column, so the work is about that of one solution of order `n − 1`. A leading block that is singular sends the matrix to the form modulo `D` below.
- **Hermite normal form modulo `D`** (Domich, Kannan and Trotter, *Hermite normal form computation using modulo determinant arithmetic*, Math. Oper. Res. 12, 1987; Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 2.4.8, here for rows). A lattice of full rank `n` whose determinant divides `D` contains `D·Zⁿ`, so its form is that of its generators with `D·Zⁿ` added and every element can be kept in `[0, D)`. Column `k` in turn: the rows with an element there are combined by extended Euclidean steps of two rows into one row `g` with the greatest common divisor `g_k` of the column; the pivot is `d_k = gcd(g_k, R_k)`, the row `u·g` with `u·g_k ≡ d_k (mod R_k)`; the vectors with zeros up to column `k` form a lattice of determinant `det / (d₀⋯d_k)`, which the other rows span together with `R_{k+1}·Z^{n−k−1}`, `R_{k+1} = R_k / d_k`, so the rest goes on modulo `R_{k+1}`. Last, the elements above each pivot are brought into `[0, d_l)` by its row. No element grows past `D`, which is at most `r·(b + ½·log₂ r)` bits for elements of `b` bits (Hadamard's bound). For `D` below `2^63` the same steps run in words, every product reduced in registers. This is the form of the minors of order below 8, of those whose leading block is singular, and the form modulo `g` above.
- **The Hermite form of any matrix.** The columns `J` of `H` are the Hermite form of the rows of `A[:, J]`, of full rank `r` with a determinant that divides `D`: the form of `M` as above, with the other rows of `A[:, J]` added to it in the same way, every element kept below the product of the pivots, a multiple of the determinant (or, for `r` below 8, the form of all the rows modulo `D`); the other columns are `H[:, K] = H[:, J]·X`, since a vector of the row space is fixed by its elements in `J`. `hermite_form()` stops there. A square matrix of order 8 or more takes the form of the matrix by the method above first, with no profile: a form of full rank proves it nonsingular, and its profiles are then every row and every column (this for all four functions); a singular one, or one whose leading block is singular, takes the profiles.
- **The transformation.** `U` is that of the square `B = [A[:, J] | E_S]`, `E_S` the columns of the identity for the rows `S` outside `R`, of the same determinant `±D`: `U = H_B·B⁻¹`, unimodular as the transformation between two bases of one lattice. `B` is `[A | I]` on the columns of its pivots, so `H_B` is the form of `[A | I]` on them and `U` its transformation. `H_B` is not computed as the form of an `m × m` matrix, which would cost `O(m³)`, but from its structure. With `N = D·Y`, a matrix of integers, its last `m − r` rows are `(−z·Y, z)` on the columns `R` and `S` for the rows `z` of the Hermite form of the lattice `Λ = {z : z·N ≡ 0 (mod D)}`. `Z^{m−r} / Λ` is the subgroup of `(Z/D)^r` that the rows of `N` span, so at most `t ≤ log₂ D` pivots of that form are above 1, and each row is its pivot and elements below the pivots above 1 after it. The pivot at row `k` is the order of `N_k` modulo the subgroup the later rows span, which is held, from the last row up, as a triangular basis of that subgroup with `D·Z^r` added, each basis row with its coefficients on the rows of `N` it came from: one pass down that basis gives the order and the coefficients, and a new pivot above 1 joins the basis by extended Euclidean steps. The first `r` rows are `((Z_i − w·N) / D, w)` for `Z = D·H[:, J]·M⁻¹` and `w` the solution of `w·N ≡ Z_i (mod D)` from the same basis, reduced by the rows of the form of `Λ`; with no rows outside `R`, a square nonsingular matrix among them, `U = H·M⁻¹` itself. Each row costs `O(r² + r·t + t²)` operations on numbers below `D`. `Z` and `H·M⁻¹` are matrices of integers, computed from their residues modulo word primes that do not divide `D`, `M⁻¹` by `ModMatrix::inverse` and the product by `ModMatrix`'s (row after row for a sparse `H`), as many primes as twice Hadamard's bound on the adjugate times the longest row of `H` over `D` asks for, then the Chinese remainder theorem.
- **Elementary divisors**, from the Hermite form `H`. A pivot 1 is alone in its column (the elements above it are in `[0, 1)`), so the column steps that clear the rest of its row change nothing else, and it gives a divisor 1; the rows of the pivots above 1 are zero in those columns, so the other divisors are those of their block on their pivots' columns and the columns outside `J`, of full row rank `t`. For a random square matrix `t = 1` and the divisors are `1, …, 1, |det A|`. A wide matrix is taken by its transpose, which has the same divisors and fewer columns. The block's are computed modulo the product `P` of its pivots (Cohen, Algorithm 2.4.14, here for any shape and rank): their product is the greatest common divisor of the `t × t` minors, so it divides `P`, and the Smith form over `Z/PZ` is `diag(d₁, …, d_t)`. Position `t` takes a nonzero element of what remains as its pivot and clears its column and its row by steps of two rows or two columns: a subtraction when the pivot divides, an extended Euclidean step otherwise, which makes the pivot a proper divisor of itself. When `g = gcd(pivot, R)` does not divide some element of the block, that element's row is added to the pivot's and the clearing goes on with a smaller pivot. Then `d_t = g`, and the block is reduced modulo `R / d_t`, which the product of the divisors left still divides.
- **The Smith form with its transformations** (alternating Hermite forms, as Kannan and Bachem do in *Polynomial algorithms for computing the Smith and Hermite normal forms of an integer matrix*, SIAM J. Comput. 8, 1979). `U₁·A = H` as above; the Hermite form of the columns of its top `r` rows, `H[..r]·V₁ = [K | 0]`, `K` lower triangular of order `r`. `V₁` is the transpose of the transformation of the form of `[H[..r]ᵀ | E_K]`, `K` the columns of `H` outside its pivots, computed as above from profiles known from `A`'s (the columns of `H[..r]ᵀ` independent, its rows `J` and `Y = Xᵀ`); it is a transformation of the Hermite form of the columns, not that of `[H[..r]ᵀ | I]`. Both `U₁` and `V₁` keep their kernel parts from here on: only the first `r` rows of `U` and columns of `V` change, so the work stays linear in `m` and `n`. `K` is put in its Hermite form by rows and by columns in turn, each time as the form of a square nonsingular matrix above, with its unique transformation `H·K⁻¹` from residues, until it is diagonal; a leading row and column that are clear stay so, and the turns go on with the block after them. Each turn either makes the leading element a proper divisor of itself or clears its row and its column, so there are at most `r` turns plus the number of prime factors of the determinants of the blocks. Then pairs `diag(a, b)` become `diag(gcd, lcm)` by `P = [[x, y], [−b/g, a/g]]` and `Q = [[1, −y·b/g], [1, x·a/g]]`, `P·diag(a, b)·Q` or `Qᵀ·diag(a, b)·Pᵀ`, for `g = x·a + y·b`. gcd and lcm sort the exponents of every prime at once, so the pairs can be the comparators of any sorting network; they are those of Batcher's odd-even merge sort, `O(r·log² r)` steps and `O(log² r)` on each position, with the larger factor `Q` on the side, `U` or `V`, whose two lines are the shorter. Last, the first rows of `U` are reduced modulo its last rows and the first columns of `V` modulo its last columns, each element in the line of a pivot of the kernel part into `[0, pivot)`.

The work, with `D` as above: for a random square matrix, `hermite_form()` about one solution of order `n − 1` by Dixon's lifting, `O(n³)` operations on words and `O(n²·L)` for its `L` steps, `L` about the length of `D` over 61 bits, and `hermite_normal_form()` then `O(n^ω)` operations on words for each of about `L` primes; in general, the form of `M` plus `O((m − r)·r·(1 + t'))` operations on numbers below `D` for the other rows, `t'` the columns of pivots above 1, or the form modulo `D`, about `m·r²` operations on numbers below `D`, and `hermite_normal_form()` about `m·(r² + r·t + t²)` more for `U`, besides the solutions with `M` and the `m × m` result; `elementary_divisors()` the Hermite form and about `t²·(t + n − r)` operations on numbers below the product of the `t` pivots above 1; `smith_normal_form()` the same for `U` and for `V`, then an `r × r` Hermite form and an exact solution for each turn (none for 84 % of the random matrices of the sweeps below, at most four), and `O(r·log² r)` gcd and lcm steps on lines of length `m` and `n`.

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

- Pernet and Stein's method is fast when the greatest common divisor `g` of its two determinants is small and the form has few pivots above 1, as for random matrices. Structured ones (q-ary bases, diagonal and triangular matrices with long elements, products built to have many invariant factors) can have `g` as long as the determinant and many such pivots: the form modulo `g` then does `O(n³)` operations on numbers as long as `g`, `O(n⁴·b)` bit operations and more for elements of `b` bits, as the form modulo `D` does for all the minors below order 8 and those whose leading block is singular. Storjohann's algorithms are not implemented.
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
- Products. The classical product packs `Bᵀ` once and takes each element as a dot product of a row of `A` with a row of `Bᵀ`, its sum of products held unreduced in one word while `k·(p − 1)² < 2^64`, in two while it is below `2^128`, and in three beyond (a word counting the carries), and reduced once, as FLINT's `nmod_mat` does: `m·k·n` products of words and `m·n` reductions. From `min(m, k, n) ≥ 128` the product is Strassen–Winograd's recursion (Winograd's form of Strassen's method, seven products of halves and fifteen additions, in the schedule of Boyer, Dumas, Pernet and Zhou, ISSAC 2009) down to the classical product, `O(n^2.81)` operations; a dimension that is odd is peeled, the even part going through the recursion and the last row, the last column and the rank-one term of the last inner index taken classically. The residues are those of the classical product. The crossover was chosen by measurement, among 64 to 256, for primes of 20 to 64 bits. Modulo a prime below `2^32` the product is computed exactly in doubles by the floating-point kernels of `Matrix<f64>`: residues for which `k·(p − 1)² ≤ 2^53` are multiplied whole, every sum of products an integer the doubles hold, from 13 rows and columns; larger ones are split into halves of 16 bits and multiplied as Karatsuba's three products, the inner dimension in runs that keep the sums below `2^53`, from 64 rows and columns while the classical product holds a sum of `k` products in one word (`k·(p − 1)² < 2^64`) and from 24 where it needs two; each element is reduced once. The edges were measured against the integer kernels and their recursion, for primes of 16 to 32 bits and up to 4096 rows (where the doubles still took 0.30 of the time for whole residues and 0.73 for halves).
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
- Products against the classical product in 128 bits: from `0 × 3 × 2` to `64 × 64 × 64` for every prime, and at the crossover (`127 × 128 × 129`, `128³`, `129³`, `200 × 131 × 197`, `131 × 263 × 133`) and with two levels and odd dimensions at each (`257 × 259 × 261`) for primes of each kind of sum. On both sides of each edge of the doubles: whole residues at 12 and 13 rows (primes of 16 and 20 bits), halves at 63 and 64 for a prime of 26 bits and at 23 and 24 for primes of 31 and 32 bits, square and thin, with an unreduced sum of `(p − 1)²` terms.
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
| `&a * &b`, `a *= &b` | the product: M4RM, and Strassen–Winograd while all three dimensions are at least 8192 |
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

- **Products, M4RM** (Arlazarov, Dinic, Kronrod and Faradzev, 1970, as Albrecht, Bard and Hart arrange it in "Algorithm 898: Efficient multiplication of dense matrices over GF(2)", ACM TOMS 37, 2010). For `C = A·B` the rows of `B` are taken in groups of `k`. The `2^k` sums of a group's rows are built into a table in Gray-code order, entry `g ^ (g >> 1)` the one before it plus the row of bit `trailing_zeros(g)`, so each entry is one row addition; each row of `C` then adds the entry that its row of `A` selects with its `k` bits in the group's columns. Eight tables are built at a time, so that a row of `C` is visited once for eight groups, the eight entries its bits select added to it in one pass, and the columns of `B` and `C` are taken in blocks narrow enough that the eight tables fill at most 4 MiB (measured: on the M4 a block fewer pays more than the tables' staying in a smaller cache). `k = ⌊log₂ m⌋ − 1`, from 1 to 8; below eight rows `k` is 1 and the product is the schoolbook one, a row of `B` added for each 1 of `A`. A product of `m × l` by `l × n` costs about `m·l·n / (64·k)` word additions and `(l / k)·2^k·n / 64` for the tables, where the schoolbook product takes `m·l·n / 128`.
- **Strassen–Winograd.** While all three dimensions are at least the cutoff (8192 for `*`), the leading part whose dimensions are multiples of 128 is split into halves at multiples of 64 columns, so that no word is shifted, and multiplied by Winograd's form of Strassen's method, seven products of halves and fifteen sums (over `GF(2)` a difference is a sum); the fewer than 128 rows, inner columns and columns beyond it are peeled off and multiplied by M4RM, as M4RI's `mzd_mul` does. The cutoff 8192 keeps every half at 4096 or more: in the final measurements one level of the recursion on 4096 rows took 20.5 ms where M4RM alone took 18.4 ms, and on 8192 rows the default product, with halves of 4096, 132 ms where a second level (`mul_strassen` from 4096) took 149 ms. With the recursion the product is `O(n^log₂7)`.
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
- **In the tests, independently.** Every reference product against a schoolbook product written in the test, and 60 random products of up to 129 rows and columns element by element against `Σ a_ik·b_kj`; the 8 reference products that run by default and have all three dimensions at least 128 (128³, 129³ and 1000³ twice each, 2048³ and `2113 × 2177 · 2177 × 2100`; the other 26 have a dimension below 128, which the recursion never halves) again by `mul_strassen` from 128, down to halves of 64, and from 300 (the hashes again identical); 40 products of random shapes of 128 to 600 in each dimension, with and without remainders to peel, by `mul_strassen` from 128, 129 and 200, down to halves of 64, against the schoolbook product; and a `2112 × 2112 · 2112 × 2177` product by two levels of it, every dimension peeled at the first, against the product of thin matrices taken the other way round; a `512 × 70 · 70 × 16700` product, whose columns M4RM takes in two blocks, against the product of the transposes and the definition; each reference echelon form against the definition (zeros before each pivot, a single 1 in each pivot column, zero rows past the rank, ascending pivots), `echelonize` in place against it, the null space by `A·N = 0` and `rank(N) = n − rank`, the rank of `Aᵀ`, and up to 200 rows and columns against a Gauss–Jordan elimination on `bool`s written in the test; each inverse by `A·A⁻¹ = A⁻¹·A = I` and `det` against its existence; each solution by `A·X = B`. 400 random matrices of up to 69 × 139, on both sides of the switch to M4RI at 32 rows and across the words' boundaries, of the five kinds, against the Gauss–Jordan elimination: the reduced form and pivots, the rank, the null space element by element against the canonical basis, a solution of a right-hand side from the image with zeros in the free rows, solvability of a random one against the rank of `[A | b]`, and, square, the inverse against the right half of the elimination of `[A | I]`. Transposes element by element and `(A·B)ᵀ = Bᵀ·Aᵀ`; the padding bits zero after every operation, on shapes of 1 to 200 columns; equal matrices built in different orders equal and hashed alike, and shapes with no elements told apart; parities of numbers of 8000 bits of either sign; edge cases (`0 × 0`, `0 × n`, `n × 0`, `1 × 1`, zero, identity and all-ones matrices); every documented panic; and the cost of three eliminations against one of the same work in the same test, the fastest of three runs each, each held within 20 times its reference: 32 rows whose bits are all in the last 64 of 65536 columns against the same rows with those bits first (measured 0.08 times; 206 times for a search that read such rows again in every strip), an unsolvable wide system whose rows zero in `A` are not zero in `[A | b]` against a solvable one (1.0 times; 161 times), and an unsolvable 1024 × 131072 system whose odd rows have one 1 each, spread over all the columns, and whose even rows are zero, against its odd rows alone, so that strips with pivots keep coming after the rows without a 1 in `A` are known (1.8 times; 89 times with those rows kept in play).
- **Every path.** The whole suite also passes with the elimination forced to be classical at every size, and forced to be M4RI at every size with strips of 1, 2, 3, 8 and 16 columns; with the M4RM groups fixed at 1, 2, 3, 5 and 8 rows; with Strassen–Winograd from 128; with one table and blocks of a single word, and with three tables of 3000 words. Deliberate errors in the clearing of the rows above the pivots, the Gray-code index, the reduction of earlier pivot rows, the strip bits read across two words, the lazy reduction of the search's list, the bound on a row's first 1 (set a word too far, or compared a column short), the jump past strips without a pivot, the placing of the pivot rows after the search and the following of the rows without a 1 through it, the consistency check of `solve`, one of Winograd's sums, the peeling and the transposes' masks are each caught.
- 43 tests (one ignored) run in `cargo test --test math -- matrix::gf2`, `--include-ignored` for the 4096 product; all 43 pass on x86_64 too (Ubuntu 26.04, Rust 1.100, emulated). In `lib/tests/math/matrix/data`, `cc -O2 gen_gf2_m4ri.c -I/opt/homebrew/include -L/opt/homebrew/lib -lm4ri -o gen_gf2_m4ri && ./gen_gf2_m4ri > gf2_reference.txt` regenerates the references; `c++ -O2 -std=c++17 gen_gf2_ntl.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lntl -lgmp -o gen_gf2_ntl && ./gen_gf2_ntl < gf2_reference.txt` and `python3 gen_gf2_sympy.py < gf2_reference.txt` check them against NTL and sympy.
### Limits

The matrices are dense; nothing here exploits sparsity. Elimination is M4RI alone, `O(n³ / log n)`: there is no PLE or PLUQ decomposition by blocks with fast products (M4RI's `mzd_echelonize_pluq`, `O(n^ω)`), which is asymptotically faster for large matrices, and `inverse()` and `solve` eliminate the augmented matrix, `n + 64·⌈n/64⌉` columns wide for the inverse. The strip width stops growing at 16 columns, from `2^17` rows, and the groups of M4RM at 8 rows, from 512 rows: a product too thin for Strassen–Winograd costs about a quarter of the schoolbook product's row additions however large it is. The cutoff 8192 and the 4 MiB of tables were chosen by measurement on an Apple M4 Max; the strip width, the group size and the eight tables by the method's cost and one machine's caches, not tuned; and the kernels are portable code without SIMD. Strassen–Winograd allocates at each level the quarters of the operands, their eight sums and the seven products, a few times the size of the operands. `solve` gives one solution, the one with zeros in the rows of the free columns; the others are it plus the null space. There are no views of blocks and no column operations: a column operation is a row operation of the transpose.

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
- The product of matrices of numbers: when every element fits in a word, the sums of products are computed in machine arithmetic, in doubles by the floating-point kernels of `Matrix<f64>` while `k·max|a|·max|b| ≤ 2^53` (every partial sum then an integer they hold exactly), in `i128`s while the magnitudes are below `2^63` and the sums below `2^127`, and in three words beyond, four columns at a time. Longer elements go by their residues modulo word primes above `2^61`, as many as make their product pass twice the bound on every element, the products of `ModMatrix`'s kernels, then each element rebuilt by the Chinese remainder theorem one prime at a time, when the order is large beside the length of the elements (the residues cost `(m·k + k·n)·L·c`, the products `c·m·k·n` and the rebuilding `m·n·c²` against `m·k·n·L²` for the sums of products of limbs); otherwise Strassen–Winograd's recursion from a crossover that falls as the elements grow, by the length of the longest element of the two operands: 96 rows and columns up to 16 bytes, 72 up to 48, 64 up to 64, 48 up to 128, 32 up to 256, 16 up to 1024 and 12 beyond (measured against the classical product for elements of 2 to 256 limbs), its leaves the classical product with each sum accumulated in limbs, the products by the limb kernels into working memory, one number made from each sum. The numbers are those of the classical product, little-endian as it leaves them.
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

- Strassen–Winograd on numbers against the classical product written in the test, with elements of both signs and byte orders and some zeros: 300-byte elements at `15 × 16 × 17`, `16³`, `17³` and `16 × 17 × 16` (one level), `33 × 32 × 35` and `32³` (by the residues); 72-byte elements at `47 × 48 × 49`, `48³`, `49³`, `50 × 52 × 49` and `56³`, and 140-byte ones at `31 × 32 × 33`, `32³` and `33³`, below and from the crossovers of 48 and 32; 100-byte elements at `64 × 64 × 65` (by the residues); 16-, 24- and 56-byte elements below and at the crossovers of 96, 72 and 64; 1100-byte elements at `11 × 12 × 13`, `12³` and `13 × 13 × 12`, below and from 12; 1- and 8-byte elements at `129 × 128 × 130` (in machine arithmetic); a zero dimension or a single row or column; every result equal and little-endian; and `det(A·B) = det(A)·det(B)` at `20 × 20` with 300-byte elements.
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

Measured on an Apple M4 Max (macOS 27.2) on 2026-11-26, in the final run of the release, one area after another with nothing else running, by the harness `bench/lattice` of the repository (`bench/README.md` has the method): one thread on each side; the crate as a user builds it, with Rust 1.100 (stable) and the default release profile, without LTO or `target-cpu`; FLINT 3.6.0, NTL 11.6.0, M4RI 20260122 and PARI/GP 2.19.1 (Homebrew's) in the same process through C and C++ shims compiled at `-O3 -mcpu=native`, alternated with ours sample by sample; the median of 15 samples of at least 1 ms each, short operations cycling through three fresh matrices; and every result checked against ours before the clock starts (PARI's Hermite forms are of the columns of `Aᵀ`, its convention). Each row gives the fastest reference and the ratio of our time to its time: below 1, ours is faster. The [rational numbers](rational.md) page has the matrices of rationals in more detail, from its own harness.

`Matrix<Number>` and `Matrix<Rational>`:

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `Matrix<Number>::det` | n = 16, integers in `−99..=99` | 8.00 µs | FLINT fmpz_mat_det | 8.28 µs | 0.967 |
| `Matrix<Number>::det` | n = 32, integers in `−99..=99` | 127 µs | FLINT fmpz_mat_det | 72.0 µs | 1.76 |
| `Matrix<Number>::det` | n = 64, integers in `−99..=99` | 585 µs | FLINT fmpz_mat_det | 608 µs | 0.962 |
| `Matrix<Number>::det` | n = 128, integers in `−99..=99` | 3.95 ms | FLINT fmpz_mat_det | 3.80 ms | 1.04 |
| `Matrix<Number>::det` | n = 256, integers in `−99..=99` | 35.1 ms | FLINT fmpz_mat_det | 29.6 ms | 1.18 |
| `Matrix<Number>::det` | n = 16, 64-bit integers | 102 µs | FLINT fmpz_mat_det_bareiss | 81.5 µs | 1.25 |
| `Matrix<Number>::det` | n = 32, 64-bit integers | 607 µs | FLINT fmpz_mat_det | 552 µs | 1.10 |
| `Matrix<Number>::det` | n = 64, 64-bit integers | 3.17 ms | FLINT fmpz_mat_det | 3.45 ms | 0.917 |
| `Matrix<Number>::det` | n = 128, 64-bit integers | 20.4 ms | PARI det | 16.9 ms | 1.21 |
| `Matrix<Number>::det` | n = 10, 1000-bit integers | 512 µs | FLINT fmpz_mat_det_bareiss | 464 µs | 1.10 |
| `Matrix<Number>::det` | n = 20, 1000-bit integers | 3.44 ms | PARI det | 2.62 ms | 1.31 |
| `Matrix<Number>::det` | n = 40, 1000-bit integers | 24.5 ms | PARI det | 9.24 ms | 2.66 |
| `Matrix<Number>::det` | n = 40, 64-bit integers, singular (row n = row 1 + row 2) | 1.32 ms | FLINT fmpz_mat_det | 1.85 ms | 0.717 |
| `Matrix<Number>::det` | n = 64, integers in `−99..=99`, singular (row n = row 1 + row 2) | 1.34 ms | FLINT fmpz_mat_det | 689 µs | 1.94 |
| `Matrix<Number>::solve` | n = 16, one column, integers in `−99..=99` | 12.3 µs | FLINT fmpz_mat_solve_fflu + division | 15.5 µs | 0.791 |
| `Matrix<Number>::solve` | n = 32, one column, integers in `−99..=99` | 104 µs | FLINT fmpq_mat_solve_fmpz_mat | 129 µs | 0.806 |
| `Matrix<Number>::solve` | n = 64, one column, integers in `−99..=99` | 537 µs | FLINT fmpq_mat_solve_fmpz_mat | 638 µs | 0.842 |
| `Matrix<Number>::solve` | n = 128, one column, integers in `−99..=99` | 3.32 ms | FLINT fmpq_mat_solve_fmpz_mat | 4.04 ms | 0.821 |
| `Matrix<Number>::solve` | n = 256, one column, integers in `−99..=99` | 22.4 ms | FLINT fmpq_mat_solve_fmpz_mat | 31.9 ms | 0.703 |
| `Matrix<Number>::solve` | n = 16, one column, 64-bit integers | 179 µs | FLINT fmpz_mat_solve_fflu + division | 161 µs | 1.11 |
| `Matrix<Number>::solve` | n = 32, one column, 64-bit integers | 808 µs | FLINT fmpq_mat_solve_fmpz_mat | 1.20 ms | 0.673 |
| `Matrix<Number>::solve` | n = 64, one column, 64-bit integers | 3.97 ms | FLINT fmpq_mat_solve_fmpz_mat | 6.45 ms | 0.616 |
| `Matrix<Number>::solve` | n = 128, one column, 64-bit integers | 22.7 ms | FLINT fmpq_mat_solve_fmpz_mat | 38.4 ms | 0.590 |
| `Matrix<Number>::solve` | n = 10, one column, 1000-bit integers | 1.26 ms | FLINT fmpz_mat_solve_fflu + division | 1.31 ms | 0.965 |
| `Matrix<Number>::solve` | n = 20, one column, 1000-bit integers | 9.71 ms | FLINT fmpq_mat_solve_fmpz_mat | 14.1 ms | 0.688 |
| `Matrix<Number>::solve` | n = 40, one column, 1000-bit integers | 48.2 ms | FLINT fmpq_mat_solve_fmpz_mat | 86.2 ms | 0.560 |
| `Matrix<Number>::solve` | n = 64, 16 columns, integers in `−99..=99` | 5.33 ms | FLINT fmpq_mat_solve_fmpz_mat | 4.60 ms | 1.16 |
| `Matrix<Number>::solve` | n = 40, one column, integers in `−99..=99`, singular | 459 µs | FLINT fmpq_mat_solve_fmpz_mat | 225 µs | 2.04 |
| `Matrix<Number>::solve` | n = 40, one column, 64-bit integers, singular | 1.32 ms | FLINT fmpq_mat_solve_fmpz_mat | 1.70 ms | 0.774 |
| `Matrix<Number>::inverse_rational` | n = 16, integers in `−99..=99` | 89.9 µs | FLINT fmpz_mat_inv + fmpq_mat_set_fmpz_mat_div_fmpz | 221 µs | 0.407 |
| `Matrix<Number>::inverse` | n = 16, integers in `−99..=99` | 59.6 µs | FLINT fmpz_mat_inv (B, den) | 177 µs | 0.337 |
| `Matrix<Number>::inverse_rational` | n = 32, integers in `−99..=99` | 1.23 ms | FLINT fmpz_mat_inv + fmpq_mat_set_fmpz_mat_div_fmpz | 2.30 ms | 0.535 |
| `Matrix<Number>::inverse` | n = 32, integers in `−99..=99` | 706 µs | FLINT fmpz_mat_inv (B, den) | 1.68 ms | 0.419 |
| `Matrix<Number>::inverse_rational` | n = 64, integers in `−99..=99` | 14.3 ms | PARI RgM_inv | 12.1 ms | 1.18 |
| `Matrix<Number>::inverse` | n = 64, integers in `−99..=99` | 8.59 ms | FLINT fmpz_mat_inv (B, den) | 17.8 ms | 0.481 |
| `Matrix<Number>::inverse_rational` | n = 128, integers in `−99..=99` | 186 ms | PARI RgM_inv | 61.8 ms | 3.01 |
| `Matrix<Number>::inverse_rational` | n = 16, 64-bit integers | 1.06 ms | FLINT fmpz_mat_inv + fmpq_mat_set_fmpz_mat_div_fmpz | 3.13 ms | 0.338 |
| `Matrix<Number>::inverse` | n = 16, 64-bit integers | 353 µs | FLINT fmpz_mat_inv (B, den) | 2.10 ms | 0.168 |
| `Matrix<Number>::inverse_rational` | n = 32, 64-bit integers | 11.9 ms | PARI RgM_inv | 11.3 ms | 1.06 |
| `Matrix<Number>::inverse` | n = 32, 64-bit integers | 5.71 ms | FLINT fmpz_mat_inv (B, den) | 20.8 ms | 0.274 |
| `Matrix<Number> * Matrix<Number>` | n = 64, integers in `−99..=99` | 87.7 µs | FLINT fmpz_mat_mul | 85.5 µs | 1.03 |
| `Matrix<Number> * Matrix<Number>` | n = 128, integers in `−99..=99` | 383 µs | FLINT fmpz_mat_mul | 776 µs | 0.494 |
| `Matrix<Number> * Matrix<Number>` | n = 256, integers in `−99..=99` | 1.79 ms | FLINT fmpz_mat_mul | 5.66 ms | 0.317 |
| `Matrix<Number> * Matrix<Number>` | n = 32, 64-bit integers | 57.6 µs | FLINT fmpz_mat_mul | 63.7 µs | 0.905 |
| `Matrix<Number> * Matrix<Number>` | n = 64, 64-bit integers | 337 µs | FLINT fmpz_mat_mul | 427 µs | 0.789 |
| `Matrix<Number> * Matrix<Number>` | n = 128, 64-bit integers | 2.27 ms | FLINT fmpz_mat_mul | 3.15 ms | 0.721 |
| `Matrix<Number> * Matrix<Number>` | n = 256, 64-bit integers | 16.1 ms | FLINT fmpz_mat_mul | 20.4 ms | 0.788 |
| `Matrix<Number> * Matrix<Number>` | n = 16, 1000-bit integers | 475 µs | FLINT fmpz_mat_mul | 231 µs | 2.06 |
| `Matrix<Number> * Matrix<Number>` | n = 32, 1000-bit integers | 3.25 ms | FLINT fmpz_mat_mul | 1.73 ms | 1.88 |
| `Matrix<Number> * Matrix<Number>` | n = 64, 1000-bit integers | 10.2 ms | FLINT fmpz_mat_mul | 10.0 ms | 1.02 |
| `Matrix<Rational>::solve` | n = 16, fractions `p/q`, `−99 ≤ p ≤ 99`, `1 ≤ q ≤ 99`, one column | 113 µs | FLINT fmpq_mat_solve | 121 µs | 0.933 |
| `Matrix<Rational>::det` | n = 16, fractions `p/q`, `−99 ≤ p ≤ 99`, `1 ≤ q ≤ 99` | 71.8 µs | FLINT fmpq_mat_det | 65.6 µs | 1.09 |
| `Matrix<Rational>::solve` | n = 32, fractions `p/q`, `−99 ≤ p ≤ 99`, `1 ≤ q ≤ 99`, one column | 914 µs | FLINT fmpq_mat_solve | 1.36 ms | 0.674 |
| `Matrix<Rational>::det` | n = 32, fractions `p/q`, `−99 ≤ p ≤ 99`, `1 ≤ q ≤ 99` | 972 µs | FLINT fmpq_mat_det | 700 µs | 1.39 |
| `Matrix<Rational>::solve` | n = 64, fractions `p/q`, `−99 ≤ p ≤ 99`, `1 ≤ q ≤ 99`, one column | 6.80 ms | FLINT fmpq_mat_solve | 8.72 ms | 0.780 |
| `Matrix<Rational>::det` | n = 64, fractions `p/q`, `−99 ≤ p ≤ 99`, `1 ≤ q ≤ 99` | 8.26 ms | PARI det | 4.40 ms | 1.88 |

Where the references are faster:

- The determinant at the edges of its bands — 32 × 32 of small integers (1.76 times FLINT's), 64-bit integers at 16, 32 and 128 rows (1.10 to 1.25 times FLINT's and PARI's), 1000-bit integers (1.10 to 2.66 times FLINT's Bareiss and PARI's) and 256 × 256 of small integers (1.18) — and of a singular matrix of 64 rows (1.94 times FLINT's): the method is chosen by the order and the length of the integers, and between the measured edges one of the three is slower than the reference's choice; the residues of a singular matrix are computed up to Hadamard's bound, where FLINT's `fmpz_mat_det` can stop earlier with Dixon's divisor. `Matrix<Rational>::det` (1.09 to 1.88 times FLINT's and PARI's) is the determinant of the integers the rows are cleared to, longer than those of the tables above.
- `solve`: 64-bit integers at 16 rows (1.11 times FLINT's Bareiss), 16 right-hand sides at 64 rows (1.16 times FLINT's Dixon) and the proof that a singular system of 40 rows of small integers is singular (2.04 times FLINT's); from 32 rows of one column it takes 0.560 to 0.842 of FLINT's time.
- `inverse_rational` of 64 and 128 rows of small integers and of 32 rows of 64-bit integers (1.06 to 3.01 times PARI's multimodular inverse); `inverse`, the adjugate and the determinant, takes 0.168 to 0.481 of FLINT's time.
- `*` of 1000-bit integers at 16 and 32 rows (2.06 and 1.88 times FLINT's): the product by residues modulo word primes is taken there although it does not pay at these orders for integers this long (the choice by the sum of the lengths is left for later); from 64 rows it ties FLINT's (1.02), and the products of small and 64-bit integers take 0.317 to 1.03 of its time.

### Normal forms

The Hermite form `H` alone (`hermite_form`), with its transformation (`hermite_normal_form`), the elementary divisors and the Smith form with its transformations, of square matrices of random elements of the given bits, of products of low rank, and of wide and tall ones; conditions as under [Exactly](#exactly):

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `hermite_form` | square n = 10, 10 bits | 25.3 µs | FLINT fmpz_mat_hnf | 12.4 µs | 2.04 |
| `hermite_normal_form` | square n = 10, 10 bits | 43.6 µs | FLINT fmpz_mat_hnf_transform | 22.7 µs | 1.92 |
| `elementary_divisors` | square n = 10, 10 bits | 25.2 µs | FLINT fmpz_mat_snf | 16.4 µs | 1.53 |
| `smith_normal_form` | square n = 10, 10 bits | 62.7 µs | FLINT fmpz_mat_snf_transform | 31.8 µs | 1.97 |
| `hermite_form` | square n = 20, 10 bits | 134 µs | FLINT fmpz_mat_hnf | 86.9 µs | 1.54 |
| `hermite_normal_form` | square n = 20, 10 bits | 252 µs | FLINT fmpz_mat_hnf_transform | 196 µs | 1.28 |
| `elementary_divisors` | square n = 20, 10 bits | 133 µs | FLINT fmpz_mat_snf | 107 µs | 1.24 |
| `smith_normal_form` | square n = 20, 10 bits | 393 µs | FLINT fmpz_mat_snf_transform | 272 µs | 1.44 |
| `hermite_form` | square n = 50, 10 bits | 903 µs | PARI mathnf (columns of Aᵀ) | 2.22 ms | 0.407 |
| `hermite_normal_form` | square n = 50, 10 bits | 2.84 ms | FLINT fmpz_mat_hnf_transform | 4.13 ms | 0.689 |
| `elementary_divisors` | square n = 50, 10 bits | 878 µs | FLINT fmpz_mat_snf | 2.48 ms | 0.354 |
| `smith_normal_form` | square n = 50, 10 bits | 4.16 ms | FLINT fmpz_mat_snf_transform | 6.09 ms | 0.684 |
| `hermite_form` | square n = 100, 10 bits | 5.21 ms | FLINT fmpz_mat_hnf | 13.1 ms | 0.398 |
| `hermite_normal_form` | square n = 100, 10 bits | 25.7 ms | FLINT fmpz_mat_hnf_transform | 70.8 ms | 0.362 |
| `elementary_divisors` | square n = 100, 10 bits | 5.16 ms | FLINT fmpz_mat_snf | 18.2 ms | 0.284 |
| `hermite_form` | square n = 200, 10 bits | 39.0 ms | FLINT fmpz_mat_hnf | 151 ms | 0.257 |
| `elementary_divisors` | square n = 200, 10 bits | 39.1 ms | FLINT fmpz_mat_snf | 222 ms | 0.176 |
| `hermite_form` | square n = 10, 100 bits | 112 µs | FLINT fmpz_mat_hnf | 50.1 µs | 2.24 |
| `hermite_normal_form` | square n = 10, 100 bits | 241 µs | FLINT fmpz_mat_hnf_transform | 93.9 µs | 2.57 |
| `elementary_divisors` | square n = 10, 100 bits | 114 µs | FLINT fmpz_mat_snf | 55.1 µs | 2.06 |
| `smith_normal_form` | square n = 10, 100 bits | 368 µs | FLINT fmpz_mat_snf_transform | 109 µs | 3.38 |
| `hermite_form` | square n = 20, 100 bits | 806 µs | FLINT fmpz_mat_hnf | 501 µs | 1.61 |
| `hermite_normal_form` | square n = 20, 100 bits | 1.98 ms | FLINT fmpz_mat_hnf_transform | 1.40 ms | 1.41 |
| `elementary_divisors` | square n = 20, 100 bits | 803 µs | FLINT fmpz_mat_snf | 511 µs | 1.57 |
| `smith_normal_form` | square n = 20, 100 bits | 2.65 ms | FLINT fmpz_mat_snf_transform | 1.44 ms | 1.84 |
| `hermite_form` | square n = 50, 100 bits | 7.67 ms | FLINT fmpz_mat_hnf | 24.8 ms | 0.310 |
| `hermite_normal_form` | square n = 50, 100 bits | 27.6 ms | FLINT fmpz_mat_hnf_transform | 78.7 ms | 0.351 |
| `elementary_divisors` | square n = 50, 100 bits | 7.56 ms | FLINT fmpz_mat_snf | 24.9 ms | 0.304 |
| `smith_normal_form` | square n = 50, 100 bits | 44.6 ms | FLINT fmpz_mat_snf_transform | 114 ms | 0.392 |
| `hermite_form` | 30×30 rank 20 (product of 10-bit factors) | 1.03 ms | PARI mathnf (columns of Aᵀ) | 389 µs | 2.65 |
| `hermite_normal_form` | 30×30 rank 20 (product of 10-bit factors) | 2.01 ms | PARI mathnf(, 1) (columns of Aᵀ) | 761 µs | 2.64 |
| `elementary_divisors` | 30×30 rank 20 (product of 10-bit factors) | 1.04 ms | PARI matsnf | 877 µs | 1.19 |
| `smith_normal_form` | 30×30 rank 20 (product of 10-bit factors) | 2.35 ms | PARI matsnf(, 1) | 2.23 ms | 1.06 |
| `hermite_form` | 60×60 rank 40 (product of 8-bit factors) | 6.95 ms | PARI mathnf (columns of Aᵀ) | 3.52 ms | 1.98 |
| `hermite_normal_form` | 60×60 rank 40 (product of 8-bit factors) | 12.0 ms | PARI mathnf(, 1) (columns of Aᵀ) | 6.94 ms | 1.74 |
| `elementary_divisors` | 60×60 rank 40 (product of 8-bit factors) | 6.92 ms | PARI matsnf | 8.62 ms | 0.803 |
| `hermite_form` | wide 20×40, 10 bits | 828 µs | PARI mathnf (columns of Aᵀ) | 358 µs | 2.31 |
| `hermite_normal_form` | wide 20×40, 10 bits | 997 µs | PARI mathnf(, 1) (columns of Aᵀ) | 556 µs | 1.79 |
| `elementary_divisors` | wide 20×40, 10 bits | 195 µs | PARI matsnf | 130 µs | 1.49 |
| `smith_normal_form` | wide 20×40, 10 bits | 1.48 ms | PARI matsnf(, 1) | 1.23 ms | 1.20 |
| `hermite_form` | tall 40×20, 10 bits | 155 µs | FLINT fmpz_mat_hnf | 114 µs | 1.36 |
| `hermite_normal_form` | tall 40×20, 10 bits | 920 µs | FLINT fmpz_mat_hnf_transform | 356 µs | 2.58 |
| `elementary_divisors` | tall 40×20, 10 bits | 158 µs | FLINT fmpz_mat_snf | 120 µs | 1.32 |
| `smith_normal_form` | tall 40×20, 10 bits | 959 µs | FLINT fmpz_mat_snf_transform | 805 µs | 1.19 |
| `hermite_form` | tall 500×5, 10 bits | 66.3 µs | PARI mathnf (columns of Aᵀ) | 89.4 µs | 0.741 |
| `elementary_divisors` | tall 500×5, 10 bits | 58.9 µs | FLINT fmpz_mat_snf | 119 µs | 0.495 |

Where the references are faster: below 50 rows (1.24 to 3.38 times FLINT's) the fixed work of Pernet and Stein's method — a solution by Dixon's lifting, two determinants, the last rows added in — is more than FLINT's elimination costs at these sizes; and so, on the products of low rank and the wide and tall matrices (1.06 to 2.65 times FLINT's and PARI's), is the work around a minor: the rank profiles, proved, the form of the minor, and the other rows added modulo its determinant. From 50 rows the square matrices take 0.176 to 0.689 of the fastest reference's time; the elementary divisors of the product of rank 40 take 0.803 of PARI's, and the tall `500 × 5` matrix 0.495 and 0.741.

### Over GF(p)

`ModMatrix`, uniform random elements, conditions as under [Exactly](#exactly):

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `ModMatrix * ModMatrix` | n = 64, p = 2^31−1 | 54.1 µs | NTL mat_zz_p mul | 56.2 µs | 0.963 |
| `ModMatrix * ModMatrix` | n = 128, p = 2^31−1 | 342 µs | NTL mat_zz_p mul | 436 µs | 0.785 |
| `ModMatrix * ModMatrix` | n = 192, p = 2^31−1 | 965 µs | NTL mat_zz_p mul | 1.43 ms | 0.674 |
| `ModMatrix * ModMatrix` | n = 256, p = 2^31−1 | 2.22 ms | NTL mat_zz_p mul | 3.49 ms | 0.635 |
| `ModMatrix * ModMatrix` | n = 1024, p = 2^31−1 | 114 ms | NTL mat_zz_p mul | 169 ms | 0.676 |
| `ModMatrix * ModMatrix` | n = 4096, p = 2^31−1 | 7.07 s | NTL mat_zz_p mul | 8.55 s | 0.827 |
| `ModMatrix * ModMatrix` | n = 64, p = 2^62−57 | 104 µs | FLINT nmod_mat_mul | 115 µs | 0.899 |
| `ModMatrix * ModMatrix` | n = 128, p = 2^62−57 | 736 µs | FLINT nmod_mat_mul | 728 µs | 1.01 |
| `ModMatrix * ModMatrix` | n = 192, p = 2^62−57 | 2.26 ms | FLINT nmod_mat_mul | 2.10 ms | 1.08 |
| `ModMatrix * ModMatrix` | n = 256, p = 2^62−57 | 5.26 ms | FLINT nmod_mat_mul | 5.21 ms | 1.01 |
| `ModMatrix * ModMatrix` | n = 1024, p = 2^62−57 | 263 ms | FLINT nmod_mat_mul | 265 ms | 0.992 |
| `ModMatrix * ModMatrix` | n = 4096, p = 2^62−57 | 12.7 s | FLINT nmod_mat_mul | 12.8 s | 0.986 |
| `ModMatrix * ModMatrix` | n = 256, p = 65521 (16 bits) | 793 µs | NTL mat_zz_p mul | 694 µs | 1.14 |
| `ModMatrix * ModMatrix` | n = 1024, p = 65521 (16 bits) | 39.2 ms | NTL mat_zz_p mul | 39.7 ms | 0.989 |
| `ModMatrix * ModMatrix` | n = 256, p = 2^50−27 | 3.88 ms | NTL mat_zz_p mul | 3.77 ms | 1.03 |
| `ModMatrix * ModMatrix` | n = 1024, p = 2^50−27 | 196 ms | NTL mat_zz_p mul | 194 ms | 1.01 |
| `ModMatrix * ModMatrix` | n = 256, p = 2^64−59 | 5.27 ms | FLINT nmod_mat_mul | 6.99 ms | 0.754 |
| `ModMatrix * ModMatrix` | n = 1024, p = 2^64−59 | 254 ms | FLINT nmod_mat_mul | 337 ms | 0.752 |
| `ModMatrix::ple` | n = 64, p = 2^31−1 | 48.2 µs | FLINT nmod_mat_lu (in place on a copy) | 53.1 µs | 0.906 |
| `ModMatrix::ple` | n = 256, p = 2^31−1 | 1.53 ms | FLINT nmod_mat_lu (in place on a copy) | 2.63 ms | 0.583 |
| `ModMatrix::ple` | n = 1024, p = 2^31−1 | 58.0 ms | FLINT nmod_mat_lu (in place on a copy) | 132 ms | 0.438 |
| `ModMatrix::ple` | n = 2048, p = 2^31−1 | 395 ms | FLINT nmod_mat_lu (in place on a copy) | 932 ms | 0.424 |
| `ModMatrix::ple` | n = 64, p = 2^62−57 | 55.3 µs | FLINT nmod_mat_lu (in place on a copy) | 75.0 µs | 0.737 |
| `ModMatrix::ple` | n = 256, p = 2^62−57 | 2.44 ms | FLINT nmod_mat_lu (in place on a copy) | 2.87 ms | 0.848 |
| `ModMatrix::ple` | n = 1024, p = 2^62−57 | 119 ms | FLINT nmod_mat_lu (in place on a copy) | 129 ms | 0.928 |
| `ModMatrix::ple` | n = 2048, p = 2^62−57 | 841 ms | FLINT nmod_mat_lu (in place on a copy) | 884 ms | 0.951 |
| `ModMatrix::ple` | n = 4096, p = 2^31−1 | 2.80 s | FLINT nmod_mat_lu (in place on a copy) | 6.56 s | 0.427 |
| `ModMatrix::ple` | n = 4096, p = 2^62−57 | 5.97 s | FLINT nmod_mat_lu (in place on a copy) | 6.16 s | 0.968 |
| `ModMatrix::ple` | n = 256, p = 65521 (16 bits) | 943 µs | FLINT nmod_mat_lu (in place on a copy) | 2.06 ms | 0.458 |
| `ModMatrix::ple` | n = 1024, p = 65521 (16 bits) | 27.9 ms | FLINT nmod_mat_lu (in place on a copy) | 95.1 ms | 0.294 |
| `ModMatrix::ple` | n = 256, p = 2^50−27 | 1.98 ms | FLINT nmod_mat_lu (in place on a copy) | 2.25 ms | 0.880 |
| `ModMatrix::ple` | n = 1024, p = 2^50−27 | 92.5 ms | FLINT nmod_mat_lu (in place on a copy) | 114 ms | 0.814 |
| `ModMatrix::ple` | n = 256, p = 2^64−59 | 2.43 ms | FLINT nmod_mat_lu (in place on a copy) | 3.20 ms | 0.760 |
| `ModMatrix::ple` | n = 1024, p = 2^64−59 | 120 ms | FLINT nmod_mat_lu (in place on a copy) | 162 ms | 0.744 |
| `ModMatrix::rank` | n = 64, p = 2^31−1 | 45.5 µs | FLINT nmod_mat_rank | 52.1 µs | 0.873 |
| `ModMatrix::rank` | n = 256, p = 2^31−1 | 1.51 ms | NTL mat_zz_p gauss (rank, on a copy) | 2.45 ms | 0.618 |
| `ModMatrix::rank` | n = 1024, p = 2^31−1 | 57.2 ms | NTL mat_zz_p gauss (rank, on a copy) | 93.4 ms | 0.612 |
| `ModMatrix::rank` | n = 2048, p = 2^31−1 | 392 ms | NTL mat_zz_p gauss (rank, on a copy) | 655 ms | 0.599 |
| `ModMatrix::rank` | n = 64, p = 2^62−57 | 53.9 µs | FLINT nmod_mat_rank | 72.7 µs | 0.742 |
| `ModMatrix::rank` | n = 256, p = 2^62−57 | 2.45 ms | FLINT nmod_mat_rank | 2.94 ms | 0.832 |
| `ModMatrix::rank` | n = 1024, p = 2^62−57 | 121 ms | FLINT nmod_mat_rank | 131 ms | 0.924 |
| `ModMatrix::rank` | n = 2048, p = 2^62−57 | 847 ms | FLINT nmod_mat_rank | 888 ms | 0.954 |
| `ModMatrix::det` | n = 64, p = 2^31−1 | 46.8 µs | FLINT nmod_mat_det | 53.9 µs | 0.870 |
| `ModMatrix::det` | n = 256, p = 2^31−1 | 1.51 ms | NTL mat_zz_p determinant | 1.88 ms | 0.803 |
| `ModMatrix::det` | n = 1024, p = 2^31−1 | 58.5 ms | NTL mat_zz_p determinant | 86.0 ms | 0.680 |
| `ModMatrix::det` | n = 2048, p = 2^31−1 | 401 ms | NTL mat_zz_p determinant | 633 ms | 0.635 |
| `ModMatrix::det` | n = 64, p = 2^62−57 | 56.1 µs | FLINT nmod_mat_det | 76.3 µs | 0.735 |
| `ModMatrix::det` | n = 256, p = 2^62−57 | 2.52 ms | FLINT nmod_mat_det | 3.01 ms | 0.837 |
| `ModMatrix::det` | n = 1024, p = 2^62−57 | 122 ms | FLINT nmod_mat_det | 133 ms | 0.919 |
| `ModMatrix::det` | n = 2048, p = 2^62−57 | 871 ms | FLINT nmod_mat_det | 924 ms | 0.942 |
| `ModMatrix::solve` | n = 64, p = 2^31−1 | 54.9 µs | FLINT nmod_mat_solve | 61.2 µs | 0.897 |
| `ModMatrix::solve` | n = 256, p = 2^31−1 | 1.69 ms | NTL mat_zz_p solve(d, A, x, b) | 2.38 ms | 0.709 |
| `ModMatrix::solve` | n = 1024, p = 2^31−1 | 60.5 ms | NTL mat_zz_p solve(d, A, x, b) | 93.3 ms | 0.648 |
| `ModMatrix::solve` | n = 2048, p = 2^31−1 | 409 ms | NTL mat_zz_p solve(d, A, x, b) | 665 ms | 0.615 |
| `ModMatrix::solve` | n = 64, p = 2^62−57 | 62.8 µs | FLINT nmod_mat_solve | 84.8 µs | 0.740 |
| `ModMatrix::solve` | n = 256, p = 2^62−57 | 2.63 ms | FLINT nmod_mat_solve | 3.06 ms | 0.859 |
| `ModMatrix::solve` | n = 1024, p = 2^62−57 | 124 ms | FLINT nmod_mat_solve | 134 ms | 0.924 |
| `ModMatrix::solve` | n = 2048, p = 2^62−57 | 871 ms | FLINT nmod_mat_solve | 921 ms | 0.945 |
| `ModMatrix::inverse` | n = 64, p = 2^31−1 | 158 µs | NTL mat_zz_p inv(d, X, A) | 163 µs | 0.970 |
| `ModMatrix::inverse` | n = 256, p = 2^31−1 | 5.33 ms | NTL mat_zz_p inv(d, X, A) | 4.46 ms | 1.19 |
| `ModMatrix::inverse` | n = 1024, p = 2^31−1 | 215 ms | NTL mat_zz_p inv(d, X, A) | 242 ms | 0.889 |
| `ModMatrix::inverse` | n = 2048, p = 2^31−1 | 1.52 s | NTL mat_zz_p inv(d, X, A) | 1.87 s | 0.811 |
| `ModMatrix::inverse` | n = 64, p = 2^62−57 | 189 µs | FLINT nmod_mat_inv | 320 µs | 0.591 |
| `ModMatrix::inverse` | n = 256, p = 2^62−57 | 9.37 ms | FLINT nmod_mat_inv | 11.8 ms | 0.797 |
| `ModMatrix::inverse` | n = 1024, p = 2^62−57 | 480 ms | FLINT nmod_mat_inv | 523 ms | 0.918 |
| `ModMatrix::inverse` | n = 2048, p = 2^62−57 | 3.38 s | FLINT nmod_mat_inv | 3.59 s | 0.942 |
| `ModMatrix::rref` | n = 64, p = 2^31−1 | 51.6 µs | FLINT nmod_mat_rref (on a copy) | 55.3 µs | 0.933 |
| `ModMatrix::rref` | n = 256, p = 2^31−1 | 1.62 ms | FLINT nmod_mat_rref (on a copy) | 2.69 ms | 0.602 |
| `ModMatrix::rref` | n = 1024, p = 2^31−1 | 61.3 ms | FLINT nmod_mat_rref (on a copy) | 140 ms | 0.437 |
| `ModMatrix::rref` | n = 2048, p = 2^31−1 | 421 ms | FLINT nmod_mat_rref (on a copy) | 992 ms | 0.424 |
| `ModMatrix::rref` | n = 64, p = 2^62−57 | 62.2 µs | FLINT nmod_mat_rref (on a copy) | 78.1 µs | 0.796 |
| `ModMatrix::rref` | n = 256, p = 2^62−57 | 2.67 ms | FLINT nmod_mat_rref (on a copy) | 3.05 ms | 0.875 |
| `ModMatrix::rref` | n = 1024, p = 2^62−57 | 127 ms | FLINT nmod_mat_rref (on a copy) | 138 ms | 0.915 |
| `ModMatrix::rref` | n = 2048, p = 2^62−57 | 879 ms | FLINT nmod_mat_rref (on a copy) | 933 ms | 0.943 |
| `ModMatrix::null_space` | n = 256, p = 2^31−1 | 1.85 ms | NTL mat_zz_p kernel of Aᵀ (its dimension) | 2.63 ms | 0.703 |
| `ModMatrix::null_space` | n = 1024, p = 2^31−1 | 68.7 ms | NTL mat_zz_p kernel of Aᵀ (its dimension) | 105 ms | 0.656 |
| `ModMatrix::null_space` | n = 256, p = 2^62−57 | 2.91 ms | FLINT nmod_mat_nullspace | 3.46 ms | 0.842 |
| `ModMatrix::null_space` | n = 1024, p = 2^62−57 | 140 ms | FLINT nmod_mat_nullspace | 152 ms | 0.921 |
| `ModMatrix + ModMatrix` | n = 1024, p = 2^62−57 | 216 µs | FLINT nmod_mat_add | 417 µs | 0.518 |
| `ModMatrix - ModMatrix` | n = 1024, p = 2^62−57 | 222 µs | FLINT nmod_mat_sub | 414 µs | 0.536 |
| `ModMatrix::scale` | n = 1024, p = 2^62−57 | 431 µs | FLINT nmod_mat_scalar_mul | 696 µs | 0.619 |
| `ModMatrix::transpose` | n = 1024, p = 2^62−57 | 588 µs | FLINT nmod_mat_transpose | 1.03 ms | 0.572 |
| `ModMatrix::from_matrix` | n = 256, entries of 64 bits, p = 2^62−57 | 273 µs | FLINT fmpz_mat_get_nmod_mat | 461 µs | 0.591 |
| `ModMatrix::from_matrix` | n = 256, entries of 1000 bits, p = 2^62−57 | 2.68 ms | FLINT fmpz_mat_get_nmod_mat | 1.47 ms | 1.82 |
| `ModMatrix::from_matrix` | n = 1024, entries of 64 bits, p = 2^62−57 | 4.92 ms | FLINT fmpz_mat_get_nmod_mat | 14.7 ms | 0.335 |

Where the references are faster: the product at `2^62 − 57` ties FLINT's (0.899 to 1.08); at 65 521 and 256 rows (1.14 times NTL's, 0.989 at 1024) and at `2^50 − 27` (1.01 and 1.03) NTL's floating-point kernels are as fast or a little faster. `inverse` modulo `2^31 − 1` at 256 rows (1.19 times NTL's): both triangular solves run on the whole permuted identity, about `2n³` products after the decomposition. `from_matrix` of elements of 1000 bits (1.82 times FLINT's): each element is reduced on its own; with elements of 64 bits it takes 0.335 and 0.591 of FLINT's time.

### Over GF(2)

`BitMatrix`, uniform random matrices, of rank `n/2` and invertible ones, conditions as under [Exactly](#exactly); `mul_strassen` against M4RI at the same cutoff:

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `BitMatrix * BitMatrix` | n = 256, uniform | 23.8 µs | M4RI mzd_mul (default cutoff 4096) | 16.6 µs | 1.43 |
| `BitMatrix * BitMatrix` | n = 1024, uniform | 531 µs | M4RI mzd_mul (default cutoff 4096) | 653 µs | 0.813 |
| `BitMatrix::rank` | n = 256, uniform | 31.0 µs | M4RI mzd_echelonize_m4ri, not full | 32.5 µs | 0.953 |
| `BitMatrix::reduced_row_echelon_form` | n = 256, uniform | 37.4 µs | M4RI mzd_echelonize_m4ri, full | 28.6 µs | 1.31 |
| `BitMatrix::echelonize` | n = 256, uniform | 38.2 µs | M4RI mzd_echelonize (default), full | 53.4 µs | 0.714 |
| `BitMatrix::rank` | n = 256, rank n/2 | 20.0 µs | M4RI mzd_echelonize_m4ri, not full | 20.4 µs | 0.979 |
| `BitMatrix::reduced_row_echelon_form` | n = 256, rank n/2 | 22.5 µs | M4RI mzd_echelonize_m4ri, full | 18.8 µs | 1.20 |
| `BitMatrix::null_space` | n = 256, rank n/2 | 30.4 µs | M4RI mzd_kernel_left_pluq | 43.0 µs | 0.708 |
| `BitMatrix::inverse` | n = 256, invertible | 51.9 µs | M4RI mzd_inv_m4ri | 45.5 µs | 1.14 |
| `BitMatrix::solve` | n = 256, invertible, one column | 44.4 µs | M4RI mzd_solve_left (consistency checked) | 66.0 µs | 0.672 |
| `BitMatrix::det` | n = 256, invertible | 29.9 µs | M4RI mzd_echelonize (default) rank = n | 43.2 µs | 0.692 |
| `BitMatrix::rank` | n = 1024, uniform | 499 µs | M4RI mzd_echelonize_m4ri, not full | 526 µs | 0.948 |
| `BitMatrix::reduced_row_echelon_form` | n = 1024, uniform | 654 µs | M4RI mzd_echelonize_m4ri, full | 574 µs | 1.14 |
| `BitMatrix::echelonize` | n = 1024, uniform | 640 µs | M4RI mzd_echelonize (default), full | 927 µs | 0.691 |
| `BitMatrix::rank` | n = 1024, rank n/2 | 329 µs | M4RI mzd_echelonize_m4ri, not full | 372 µs | 0.885 |
| `BitMatrix::reduced_row_echelon_form` | n = 1024, rank n/2 | 378 µs | M4RI mzd_echelonize_m4ri, full | 354 µs | 1.07 |
| `BitMatrix::null_space` | n = 1024, rank n/2 | 511 µs | M4RI mzd_kernel_left_pluq | 794 µs | 0.643 |
| `BitMatrix::inverse` | n = 1024, invertible | 1.03 ms | M4RI mzd_inv_m4ri | 1.16 ms | 0.882 |
| `BitMatrix::solve` | n = 1024, invertible, one column | 664 µs | M4RI mzd_solve_left (consistency checked) | 1.13 ms | 0.588 |
| `BitMatrix::det` | n = 1024, invertible | 503 µs | M4RI mzd_echelonize (default) rank = n | 701 µs | 0.717 |
| `BitMatrix::transpose` | n = 1024 | 28.7 µs | M4RI mzd_transpose | 36.6 µs | 0.785 |
| `BitMatrix + BitMatrix` | n = 1024 | 4.78 µs | M4RI mzd_add | 6.79 µs | 0.704 |
| `BitMatrix * BitMatrix` | n = 4096, uniform | 18.4 ms | M4RI mzd_mul_m4rm | 27.4 ms | 0.674 |
| `BitMatrix::mul_strassen` | n = 4096, cutoff 1024 | 47.6 ms | M4RI mzd_mul (cutoff 1024) | 35.0 ms | 1.36 |
| `BitMatrix::mul_strassen` | n = 4096, cutoff 2048 | 27.6 ms | M4RI mzd_mul (cutoff 2048) | 29.0 ms | 0.952 |
| `BitMatrix::mul_strassen` | n = 4096, cutoff 4096 | 20.5 ms | M4RI mzd_mul (cutoff 4096) | 27.2 ms | 0.754 |
| `BitMatrix::rank` | n = 4096, uniform | 13.7 ms | M4RI mzd_echelonize_m4ri, not full | 12.6 ms | 1.09 |
| `BitMatrix::reduced_row_echelon_form` | n = 4096, uniform | 17.8 ms | M4RI mzd_echelonize_pluq, full | 15.7 ms | 1.14 |
| `BitMatrix::echelonize` | n = 4096, uniform | 17.9 ms | M4RI mzd_echelonize (default), full | 15.8 ms | 1.14 |
| `BitMatrix::rank` | n = 4096, rank n/2 | 10.5 ms | M4RI mzd_echelonize_m4ri, not full | 11.0 ms | 0.950 |
| `BitMatrix::reduced_row_echelon_form` | n = 4096, rank n/2 | 12.4 ms | M4RI mzd_echelonize_m4ri, full | 13.4 ms | 0.928 |
| `BitMatrix::null_space` | n = 4096, rank n/2 | 14.6 ms | M4RI mzd_kernel_left_pluq | 15.1 ms | 0.968 |
| `BitMatrix::inverse` | n = 4096, invertible | 46.8 ms | M4RI mzd_inv_m4ri | 44.2 ms | 1.06 |
| `BitMatrix::solve` | n = 4096, invertible, one column | 19.5 ms | M4RI mzd_solve_left (consistency checked) | 17.5 ms | 1.12 |
| `BitMatrix::det` | n = 4096, invertible | 13.7 ms | M4RI mzd_echelonize (default) rank = n | 13.5 ms | 1.01 |
| `BitMatrix::transpose` | n = 4096 | 500 µs | M4RI mzd_transpose | 1.20 ms | 0.416 |
| `BitMatrix + BitMatrix` | n = 4096 | 72.1 µs | M4RI mzd_add | 99.1 µs | 0.728 |
| `BitMatrix * BitMatrix` | n = 8192, uniform | 132 ms | M4RI mzd_mul (default cutoff 4096) | 196 ms | 0.674 |
| `BitMatrix::mul_strassen` | n = 8192, cutoff 1024 | 341 ms | M4RI mzd_mul (cutoff 1024) | 250 ms | 1.36 |
| `BitMatrix::mul_strassen` | n = 8192, cutoff 2048 | 198 ms | M4RI mzd_mul (cutoff 2048) | 217 ms | 0.914 |
| `BitMatrix::mul_strassen` | n = 8192, cutoff 4096 | 149 ms | M4RI mzd_mul (cutoff 4096) | 196 ms | 0.760 |
| `BitMatrix::rank` | n = 8192, uniform | 90.7 ms | M4RI mzd_echelonize_m4ri, not full | 80.3 ms | 1.13 |
| `BitMatrix::reduced_row_echelon_form` | n = 8192, uniform | 121 ms | M4RI mzd_echelonize_pluq, full | 89.1 ms | 1.36 |
| `BitMatrix::rank` | n = 8192, rank n/2 | 73.1 ms | M4RI mzd_echelonize_m4ri, not full | 68.7 ms | 1.06 |
| `BitMatrix::reduced_row_echelon_form` | n = 8192, rank n/2 | 88.2 ms | M4RI mzd_echelonize_m4ri, full | 86.2 ms | 1.02 |
| `BitMatrix::null_space` | n = 8192, rank n/2 | 96.2 ms | M4RI mzd_kernel_left_pluq | 92.3 ms | 1.04 |
| `BitMatrix::inverse` | n = 8192, invertible | 373 ms | M4RI mzd_inv_m4ri | 305 ms | 1.22 |
| `BitMatrix::solve` | n = 8192, invertible, one column | 125 ms | M4RI mzd_solve_left (consistency checked) | 95.0 ms | 1.32 |
| `BitMatrix::det` | n = 8192, invertible | 90.0 ms | M4RI mzd_echelonize (default) rank = n | 87.2 ms | 1.03 |
| `BitMatrix * BitMatrix` | n = 16384, uniform | 922 ms | M4RI mzd_mul (default cutoff 4096) | 1.37 s | 0.673 |
| `BitMatrix::rank` | n = 16384, uniform | 802 ms | M4RI mzd_echelonize_m4ri, not full | 546 ms | 1.47 |
| `BitMatrix::reduced_row_echelon_form` | n = 16384, uniform | 1.16 s | M4RI mzd_echelonize_pluq, full | 664 ms | 1.75 |
| `BitMatrix::rank` | n = 16384, rank n/2 | 714 ms | M4RI mzd_echelonize_m4ri, not full | 467 ms | 1.53 |
| `BitMatrix::reduced_row_echelon_form` | n = 16384, rank n/2 | 906 ms | M4RI mzd_echelonize_m4ri, full | 594 ms | 1.53 |
| `BitMatrix::transpose` | n = 16384 | 17.8 ms | M4RI mzd_transpose | 56.0 ms | 0.317 |
| `BitMatrix + BitMatrix` | n = 16384 | 1.17 ms | M4RI mzd_add | 1.47 ms | 0.797 |
| `BitMatrix::add_row` | 64 × 1024, rows 0 and 32 | 4.00 ns | M4RI mzd_row_add | 6.70 ns | 0.598 |
| `BitMatrix::swap_rows` | 64 × 1024, rows 0 and 32 | 4.50 ns | M4RI mzd_row_swap | 5.40 ns | 0.834 |
| `BitMatrix::add_row` | 256 × 16384, rows 0 and 128 | 25.9 ns | M4RI mzd_row_add | 78.1 ns | 0.332 |
| `BitMatrix::swap_rows` | 256 × 16384, rows 0 and 128 | 34.7 ns | M4RI mzd_row_swap | 35.6 ns | 0.974 |

`BitMatrix::from(&Matrix<Number>)` has no reference.

Where the references are faster:

- The eliminations of the largest matrices — `rank` and `reduced_row_echelon_form` at 16 384 rows (1.47 to 1.75 times M4RI's), `rank`, `reduced_row_echelon_form` and `echelonize` of the uniform matrices of 4096 and 8192 rows (1.09 to 1.36), and `inverse` and `solve` at 4096 and 8192 (1.06 to 1.32): the elimination is M4RI's method alone, `O(n³ / log n)`, where M4RI also has the PLUQ decomposition by blocks with fast products, `O(n^ω)`, which its `mzd_echelonize` takes for large matrices.
- `reduced_row_echelon_form` at 256 and 1024 rows (1.07 to 1.31) and `inverse` at 256 (1.14), against M4RI's implementation of the same method: constant factors, not examined one by one.
- `*` at 256 rows (1.43 times M4RI's): M4RM's constants at a size where the tables cost as much as the additions; from 1024 rows it takes 0.673 to 0.813 of M4RI's time.
- `mul_strassen` with a cutoff of 1024 (1.36 times M4RI's at the same cutoff, at 4096 and 8192 rows): the recursion allocates at each level the quarters of the operands, their sums and the seven products.

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

- Time complexity: `2·m·n·k` operations for an `m × k` by `k × n` product; for `n × n` matrices, about `2n³/3` for `lu()`, `2n³` for `inverse()`, `n³/3` for `cholesky()`, `4n³/3` for `qr()`, `4n³/3` for `hermitian_eigenvalues()` and `4.7n³` with the eigenvectors, `8n³/3` for `singular_values()` and `9.3n³` for `svd()`, and about `10n³` for `eigenvalues()` and `25n³` for `eigen()`, the QR algorithm's sweeps, whose number depends on the matrix, taking most of it; `exp()` up to six products and one solve, and a product for each squaring; `sqrt()` and `log()` a Schur decomposition, then about `n³/3` for each square root of its triangular factor. On numbers and rationals, `rank()`, and `det()` below its bands, about `n³/3` steps of two products and an exact division of numbers as long as the minors; `det()` by residues about `64·n·L/61` primes for integers of `L` limbs, each `n³/3` products of words and the residues of the `n²` integers, `L` products of words each, or from 512 limbs about `n·M(L)·log L` an integer for all the primes, `M(L)` a product of `L` limbs; the product of numbers `O(n^2.81)` products of elements from its crossover; `solve(&b)` for `k` right-hand sides about `n³/3 + k·n²` and `inverse()` about `4n³/3`, then `n·k` and `n²` greatest common divisors; for the large systems and determinants, Dixon's lifting `n³/3` products modulo a prime and `2·n²` products of words for each of its `L` steps, `L` about the length of the determinant over 61 bits, twice for the solutions; `characteristic_polynomial()` about `n⁴/2` products and `adjugate()` `3n⁴/2`; the Hermite form of a random square matrix about one solution of a system by Dixon's lifting (Pernet and Stein's method), and in general up to `m·r²` operations on numbers below the determinant `D` of a nonsingular `r × r` minor, the elementary divisors that and less, and the transformations as in [Normal forms](#normal-forms). `ModMatrix`: products `m·k·n` products of words, and Strassen–Winograd's `O(n^2.81)` from 128 rows and columns; the PLE decomposition, and what rests on it, about `n³/3` products of words below that and `O(n^2.81)` above. `BitMatrix`: products about `m·l·n / (64·k)` word additions by M4RM, `k = ⌊log₂ m⌋ − 1` from 1 to 8, and `O(n^2.81)` from 8192 rows and columns; the elimination `O(n³ / log n)` word operations.
- Space complexity: a matrix is its elements in one vector, column after column, and its shape; a decomposition keeps its factors (`lu()` the matrix and the permutation, `qr()` the reflections and the triangular factors of their blocks, at most 64 elements per column); working memory of at most a few times the matrix, and a product packs blocks of up to `512 × 1020` elements of `op(B)` (`512 × 2040` for `f32`) and `128 × 512` of `op(A)`. A `ModMatrix` is a word an element, row after row, and a `BitMatrix` a bit an element, `⌈n/64⌉` words a row; the transformations of the normal forms are dense matrices of numbers.
- Soundness: `unsafe` code is the product's kernels and the leaves of the triangular solves and of the Cholesky decomposition, each with the extent of what it reads and writes checked before it is called; Miri runs the portable kernels, which have the same shapes as the NEON ones. The exact algorithms use the numbers' limb kernels through checked slices; the normal forms, `ModMatrix` and `BitMatrix` have no `unsafe` code. A shape that doesn't fit panics; the iterations of the QR algorithms stop after a bounded number of sweeps.

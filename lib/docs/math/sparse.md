[A](../../../README.md) › [Math](../../../README.md#math)

# Sparse matrices

`a::math::sparse` holds matrices most of whose elements are zero by their nonzeros alone, and solves with them in time and memory that follow the nonzeros rather than the square of the order. The elements are `f32`, `f64`, `Complex<f32>` or `Complex<f64>`, the trait `Scalar` of [`a::math::matrix`](matrix.md), whose dense kernels the supernodal Cholesky factorisations, the supernodal LU factorisation and the multifrontal QR decomposition use for their dense blocks.

- Storage: compressed sparse rows (`SparseMatrix`, also named `CsrMatrix`), compressed sparse columns (`CscMatrix`) and coordinates (`CooMatrix`), each converted to the others and to and from a dense `Matrix`; Matrix Market files, read and written.
- Operations: products by vectors, by the transpose and the conjugate transpose without forming them, by dense and by sparse matrices; sums and products by an element, transposes, sub-matrices, triangles, permutations, norms; and `LinearOperator`, for matrices that are never formed.
- Orderings that keep the fill of a factorisation low: reverse Cuthill–McKee, AMD, COLAMD and nested dissection, with the elimination tree and its postorder.
- Direct methods: the supernodal Cholesky factorisation `P·A·Pᵀ = L·Lᴴ` (`SparseCholesky`) and its `L·D·Lᴴ` form for Hermitian matrices that are not definite (`SparseLdl`), the LU factorisation with partial pivoting `P·A·Q = L·U` (`SparseLu`), and the multifrontal QR decomposition `A·Q_c = Q·R` for least squares (`SparseQr`).
- Iterative methods: the conjugate gradient method, MINRES, restarted GMRES and BiCGSTAB, with the preconditioners Jacobi, ILU(0) and IC(0).
- Eigenvalues: a few eigenvalues and their vectors by the implicitly restarted Lanczos and Arnoldi methods, as ARPACK has them, and by shift-and-invert near a target.

A permutation is `p[new] = old` throughout: `P·A·Pᵀ` holds the element `(p[i], p[j])` of `A` at `(i, j)`. No method passes off an answer it has not verified: a Cholesky or `L·D·Lᴴ` factorisation that does not exist is `None`; LU and QR report a matrix singular to working precision (`is_singular()`, `rank()`) and refuse to solve it; the iterative methods return their best point with `converged` false and the reason; the eigenvalue methods return only the pairs that passed a check of their residuals. A shape that doesn't fit panics.

## Examples

```rust
use a::math::sparse::{
    CooMatrix, LanczosWhich, SolveOptions, SparseCholesky, SparseLu, SparseMatrix, SparseQr,
    cg_with, lanczos,
};

// The 1-D Laplacian of 100 points, assembled from triplets.
let n = 100;
let mut coo = CooMatrix::new(n, n);
for i in 0..n {
    coo.push(i, i, 2.0f64);
    if i + 1 < n {
        coo.push(i, i + 1, -1.0);
        coo.push(i + 1, i, -1.0);
    }
}
let a: SparseMatrix<f64> = coo.to_csr();
assert_eq!(a.nnz(), 3 * n - 2);

// A·x = 1 has the solution xᵢ = (i + 1)(n − i)/2.
let b = vec![1.0; n];
let exact: Vec<f64> = (0..n).map(|i| ((i + 1) * (n - i)) as f64 / 2.0).collect();
let off = |x: &[f64]| x.iter().zip(&exact).map(|(x, e)| ((x - e) / e).abs()).fold(0.0, f64::max);

// By Cholesky, in the order AMD chooses; by LU with partial pivoting; by QR.
assert!(off(&SparseCholesky::new(&a).unwrap().solve_vec(&b)) < 1e-12);
assert!(off(&SparseLu::new(&a).solve_vec(&b).unwrap()) < 1e-12);
assert!(off(&SparseQr::new(&a).solve_vec(&b).unwrap()) < 1e-12);

// By the conjugate gradient method, from products with the matrix alone.
let r = cg_with(&a, &b, &SolveOptions::new().with_rtol(1e-12), None).unwrap();
assert!(r.converged && off(&r.x) < 1e-8);

// Its smallest eigenvalue, 2 − 2·cos(π/(n + 1)), by the Lanczos method.
let e = lanczos(&a, 1, LanczosWhich::SmallestAlgebraic).unwrap();
let smallest = 2.0 - 2.0 * (core::f64::consts::PI / (n + 1) as f64).cos();
assert!(e.converged == 1 && (e.values[0] - smallest).abs() < 1e-12);
```

## Storage and operations

A sparse matrix is held in one of three forms, which hold the same matrix and convert to each other and to a dense [`Matrix`](matrix.md) in time proportional to the stored elements and the dimensions:

| Type | Form | What it is for |
| --- | --- | --- |
| `CooMatrix<T>` | coordinates: the triplets `(row, column, value)` in the order given, a cell possibly given several times | assembling a matrix, element after element; what a Matrix Market file is read into |
| `SparseMatrix<T>`, alias `CsrMatrix<T>` | compressed sparse rows: the values row after row, the column of each, and where each row begins | every method of the module: products, solvers, orderings |
| `CscMatrix<T>` | compressed sparse columns: the same by columns | reading a column in the time its elements take |

The two compressed forms are canonical: within a row (or a column) the indices strictly ascend and no zero is stored, so two equal matrices have equal arrays. Every constructor restores that form: the entries of one cell are summed, in the order they are given in, and zeros are left out; a NaN is not a zero and is stored. Like `Matrix<f64>`, the sparse forms are `PartialEq` and neither `Eq` nor `Hash`, since a matrix holding a NaN is not equal to itself.

| Building and taking apart | |
| --- | --- |
| `SparseMatrix::zeros`, `identity`, `from_diagonal`, `tridiagonal`, `from_fn` | the usual matrices; `from_fn` calls its function row after row |
| `SparseMatrix::from_coo(rows, cols, &[(i, j, x)])` | from a slice of triplets |
| `SparseMatrix::from_csr(rows, cols, row_ptrs, col_indices, values)`, `CscMatrix::from_csc(…)`, `CooMatrix::from_triplets(…)` | from the raw arrays, validated (a wrong shape panics): the compressed ones sorted and summed where they are not canonical and taken without a copy where they are, the triplets taken as they are |
| `into_parts()` on each form | the arrays back, without a copy |
| `CooMatrix::new`, `with_capacity`, `push(i, j, x)` | assembly; `nnz()` counts the triplets, zeros and repeats included |
| `row_pointers()`, `col_indices()`, `values()`; `col_pointers()`, `row_indices()`; `row_indices()` of the triplets | the arrays, read only, since a stored zero would break the canonical form; `map` and `into_parts` change the values |

| Converting | |
| --- | --- |
| `to_csr()`, `to_csc()`, `to_coo()`, `to_matrix()` and `from_matrix()` on each form | every direction, also as `From` for owned and borrowed operands |
| `CooMatrix::transpose()`, `into_transpose()` | the triplets with rows and columns exchanged; `into_transpose` swaps the two vectors at no cost |

| Operations | |
| --- | --- |
| `mul_vec`, `mul_vec_into` | `y = A·x`; `_into` writes into a slice and allocates nothing, as a Krylov method wants |
| `mul_vec_transpose[_into]`, `mul_vec_adjoint[_into]` | `Aᵀ·x` and `Aᴴ·x` without a transposed copy (on both compressed forms) |
| `mul_matrix`, `&a * &x`, `&x * &a` | sparse times dense and dense times sparse, owned or borrowed operands |
| `&a * &b` | sparse times sparse |
| `+`, `-`, unary `-`, `*` and `/` by an element, and their assignments | elementwise; an element that is not stored stays an exact zero |
| `transpose()`, `conjugate_transpose()` | `Aᵀ` and `Aᴴ`, on both compressed forms |
| `submatrix(rows, cols)` | the block of two ranges |
| `select(&rows, &cols)` | any lists of indices, in any order, with repeats: the element `(i, j)` is `a[(rows[i], cols[j])]` |
| `upper_triangle()`, `lower_triangle()` | with the diagonal, of any shape |
| `permute(&row_perm, &col_perm)` | the element `(i, j)` is `a[(row_perm[i], col_perm[j])]`, `p[new] = old` as everywhere in the module: `P·A·Qᵀ`, and `P·A·Pᵀ` for `permute(&p, &p)` |
| `norm(Norm::…)`, `trace()`, `diagonal()`, `density()` | as the dense matrix has them; `trace` panics on a matrix that is not square, as `Matrix::trace` does |
| `is_symmetric()`, `is_hermitian()`, `is_symmetric_pattern()` | one pass over the stored elements, without a transpose |
| `LinearOperator` | implemented by `SparseMatrix`, `CscMatrix`, `Matrix` and `FnOperator`, a closure with its dimensions, so that the iterative methods also run on a matrix that is never formed |

### Methods

- **Assembly** (`from_coo`, `CooMatrix::to_csr`, `to_csc`): the entries are sorted by column by a stable least-significant-digit radix sort, each digit as wide as `max(nnz, 256)` buckets allow (one counting pass when there are no more columns than that), and then dealt out to their rows by a counting sort by row, which leaves every row in ascending order of column with the entries of one cell in the order given; one more pass sums them and drops zeros. The cost is `O(nnz + rows)` and at most eight passes of `O(nnz)` for any order of the entries and any number of columns: a matrix of 2⁶³ columns and a few thousand elements is assembled in the time its elements take. `from_csr` and `from_csc` check canonical arrays in one pass and take them as they are.
- **Conversions** between the compressed forms are the transpose: a counting sort by column, `O(nnz + rows + cols)`. A `CscMatrix` is held as the compressed rows of its transpose: a column of the matrix is a row of that transpose, its arrays are the transpose's, and its product by a vector is the transpose's product, scattered.
- **Products** by a vector gather each row into four running sums (the element at the offset `k` of the row to the sum `k mod 4`, added as `(s₀ + s₁) + (s₂ + s₃)`), four chains of additions instead of one; a product by a dense matrix reads each row once for four of its columns and sums each column as the product by that column alone does. The products by the transpose and the adjoint scatter each row into the result, so that `Aᵀ` is never formed, adding the terms of each entry one after another: they agree with the product by the formed transpose to rounding — the tests hold the two within `16·ε·max(rows, cols)` of the sum of their norms — and are the same bits where no column holds four elements. The order of the sums can change an overflow as it changes the rounding: a row whose terms overflow in one grouping and not in another, beside an infinity of the other sign, gives NaN in one form and ±∞ in the other (the row `[1.5·10³⁰⁸, −∞, 1, 1, 10³⁰⁸]` times a vector of ones is NaN gathered, its first and last terms overflowing to +∞ in the same running sum, and −∞ scattered or summed in order). Every product sums in a fixed order, so the same input gives the same bits. Dense times sparse adds, for every stored `(k, j)`, column `k` of the dense matrix times the element into column `j` of the product. Sparse times sparse is Gustavson's row-by-row algorithm (*Two fast algorithms for sparse matrices: multiplication and permuted transposition*, ACM TOMS 4, 1978): each row of the product is gathered into an accumulator as long as a row, over the rows of `B` that the row of `A` points at, its columns sorted once and read out, so the cost is the multiplications plus `O(r log r)` for a row of `r` elements of the product.
- **Sums** merge the two rows of every row at once, `O(nnz(A) + nnz(B) + rows)`; a cell that cancels exactly leaves the pattern.
- **Sub-matrices**: `submatrix` bounds each row taken by two binary searches; `select` sorts the list of columns once, stably, and then searches the shorter of a row and that list in the longer, each search beginning where the one before ended, so nothing is paid for the rows and columns not taken; when the matrix has no more than four times as many columns as those searches and the two lists are long, each element of a row no longer than four times the list is looked up in a map of the columns instead; the triangles cut each row at its diagonal by one binary search; `permute` and `symmetric_permutation` map the columns through the inverse permutation: when no row holds more than 32 elements they gather the rows in their new order and sort each by insertion, otherwise they deal the elements out twice by counting, to their new columns in the order of the new rows and then from the columns in order back to the rows, which leaves every row ascending.
- **Norms**: `One`, `Infinity`, `Max` and `Frobenius` in what the stored elements cost, every sum kept in `f64` whatever the element type, with Neumaier's compensation, and Frobenius's squares scaled by a power of two so that they neither overflow nor underflow. A NaN makes every norm NaN and an infinity makes it infinite (Frobenius's infinite also beside a NaN), as the dense norm has it. `Two` and `Nuclear` densify.

### Measured

The times on this page are not asserted by the tests. They were measured on an Apple M4 Max (macOS 27.2) on 2026-11-26, in the final run of the release, one area after another with nothing else running, by the harness `bench/sparse` of the repository (`bench/README.md` has the method): one thread on both sides; the crate as a user builds it, with Rust 1.100 (stable) and the default release profile, without LTO or `target-cpu`, and `usize` indices; the median of 15 samples of at least 1 ms each (fewer where a sample takes seconds), ours and the references alternated sample by sample in one process — SciPy's through a C interface of the crate that its Python process loads — and every answer checked against ours before the clock starts. The references: SuiteSparse 7.14.1 (CHOLMOD 5.3, UMFPACK 6.3, KLU 2.3, SPQR 4.3, CSparse, AMD, CAMD, COLAMD and SYMAMD 3.3, LDL 3.3, and the METIS 5 that CHOLMOD bundles) with OpenBLAS 0.3.34 on one thread, called from a C shim compiled at `-O3 -mcpu=native` with 32-bit indices; Apple's Accelerate, its sparse solvers with the BLAS set to one thread, whose dense blocks may still run on the M4's matrix unit (AMX), which no Rust code can reach but by assembly; faer 0.24, sprs 0.11 and nalgebra-sparse 0.11, built as this crate is; and SciPy 1.18.0 with NumPy 2.3.3 (its sparse kernels, `csgraph`, SuperLU, ARPACK, and `fast_matrix_market` on one thread).

The matrices: `lap2d-k` the five-point Laplacian of the `k × k` grid and `lap3d-k` the seven-point one of the `k³` grid, `ani2d-k` and `ani3d-k` anisotropic ones; `elast3d-k` a structural matrix, three unknowns a node of the 27-point graph of the `k³` grid; `convdiff2d-k`, `upwind2d-k` and `cd2d-k` the five-point convection–diffusion operator with three strengths of convection, `convdiff3d-k` the seven-point one; `randunsym-n` random diagonally dominant unsymmetric matrices, `randspd-n` random graph Laplacians plus the identity, `circuit-n` netlists with a power law of degrees and unsymmetric values on a nearly symmetric pattern; `tridiag-n`, `arrow-n` (a full last row and column) and `saddle-k`, a quasi-definite saddle point on the `k × k` Laplacian; for least squares `curve-n` (`[I; 2·D]`), `tallband-n` (a band of twenty times as many rows as columns), `gridls-k` (`[I; 2·Dx; 2·Dy]` on the `k × k` grid) and `randtall-n` (random, three times as many rows as columns); "scrambled", under a random symmetric permutation. Each row gives the fastest reference that does the same work and the ratio of our time to its time: below 1, ours is faster. Left out as doing less: results whose columns are left unsorted (CSparse's products, sums, permutations and compression, SciPy's product), norms summed plainly where ours are compensated, and the eigenvalues ARPACK and faer return where they returned a wrong set; as doing more, UMFPACK's solves, which refine their solutions. Our factorisations include their estimates of the condition, and our operations allocate their results where most references write into arrays given to them: that is counted in our time.

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `mul_vec` (A·x) | lap2d-100, n = 10000, nnz = 49600 | 34.5 µs | faer (u32) | 17.3 µs | 1.99 |
| `mul_vec` (A·x) | lap2d-1000, n = 1000000, nnz = 4996000 | 1.74 ms | faer (u32) | 1.85 ms | 0.941 |
| `mul_vec` (A·x) | lap3d-100, n = 1000000, nnz = 6940000 | 2.15 ms | SciPy A @ x | 2.45 ms | 0.880 |
| `mul_vec` (A·x) | elast3d-30, n = 81000, nnz = 6133248 | 1.45 ms | Accelerate SparseMultiply (CSC of Aᵀ, transposed) | 1.47 ms | 0.988 |
| `mul_vec` (A·x) | randunsym-1000000, n = 1000000, nnz = 8999966 | 6.13 ms | SciPy A @ x | 6.42 ms | 0.955 |
| `mul_vec` (A·x) | circuit-1000000, n = 1000000, nnz = 4181182 | 5.09 ms | faer (u32) | 4.57 ms | 1.11 |
| `mul_vec_transpose` (Aᵀ·x) | randunsym-1000000, n = 1000000, nnz = 8999966 | 9.07 ms | Accelerate SparseMultiply (our arrays as CSC of Aᵀ) | 8.49 ms | 1.07 |
| `mul_vec_transpose` (Aᵀ·x) | elast3d-30, n = 81000, nnz = 6133248 | 1.72 ms | Accelerate SparseMultiply (our arrays as CSC of Aᵀ) | 1.77 ms | 0.973 |
| `mul_vec_transpose` (Aᵀ·x) | lap2d-1000, n = 1000000, nnz = 4996000 | 2.13 ms | Accelerate SparseMultiply (our arrays as CSC of Aᵀ) | 1.92 ms | 1.11 |
| `CscMatrix::mul_vec` (A·x by columns) | randunsym-1000000, n = 1000000, nnz = 8999966 | 12.0 ms | Accelerate SparseMultiply (CSC) | 9.90 ms | 1.21 |
| `mul_vec` (A·x) f32 | lap2d-1000, n = 1000000, nnz = 4996000 | 1.49 ms | faer (usize) | 1.79 ms | 0.834 |
| `mul_vec` (A·x) Complex<f64> | lap2d-1000, n = 1000000, nnz = 4996000 | 2.67 ms | faer (usize) | 3.03 ms | 0.880 |
| `mul_matrix` (A·X dense) | lap2d-316, n = 99856, nnz = 498016, k = 4 | 427 µs | CHOLMOD sdmult (CSC of Aᵀ, transposed) | 433 µs | 0.987 |
| `mul_matrix` (A·X dense) | lap2d-316, n = 99856, nnz = 498016, k = 16 | 1.73 ms | sprs (row-major dense) | 1.55 ms | 1.12 |
| `mul_matrix` (A·X dense) | lap2d-316, n = 99856, nnz = 498016, k = 64 | 7.03 ms | sprs (row-major dense) | 4.64 ms | 1.52 |
| `mul_matrix` (A·X dense) | randunsym-100000, n = 100000, nnz = 899953, k = 4 | 2.02 ms | CHOLMOD sdmult (CSC of Aᵀ, transposed) | 1.97 ms | 1.02 |
| `mul_matrix` (A·X dense) | randunsym-100000, n = 100000, nnz = 899953, k = 16 | 8.08 ms | CHOLMOD sdmult (CSC of Aᵀ, transposed) | 7.99 ms | 1.01 |
| `mul_matrix` (A·X dense) | randunsym-100000, n = 100000, nnz = 899953, k = 64 | 32.1 ms | CHOLMOD sdmult (CSC of Aᵀ, transposed) | 32.5 ms | 0.987 |
| `dense × sparse` (X·A) | lap2d-316, n = 99856, nnz = 498016, k = 16 | 1.31 ms | faer (CSC right factor) | 6.32 ms | 0.208 |
| `sparse × sparse` (A·A) | lap2d-316, n = 99856, nnz = 498016 | 6.18 ms | CHOLMOD ssmult (sorted) | 4.97 ms | 1.24 |
| `sparse × sparse` (A·A) | lap3d-46, n = 97336, nnz = 668656 | 12.9 ms | CHOLMOD ssmult (sorted) | 12.5 ms | 1.03 |
| `sparse × sparse` (A·A) | randunsym-100000, n = 100000, nnz = 899953 | 101 ms | nalgebra-sparse | 95.6 ms | 1.06 |
| `sparse × sparse` (A·A) | elast3d-16, n = 12288, nnz = 876024 | 73.8 ms | nalgebra-sparse | 73.8 ms | 0.999 |
| `add` (A + B) | randunsym-100000, n = 100000, nnz = 899953 | 9.47 ms | CHOLMOD add (sorted) | 7.01 ms | 1.35 |
| `add` (A + B) | randunsym-1000000, n = 1000000, nnz = 8999966 | 98.8 ms | CHOLMOD add (sorted) | 73.6 ms | 1.34 |
| `scale` (A·α) | randunsym-1000000, n = 1000000, nnz = 8999966 | 3.80 ms | SciPy A * 1.5 | 1.88 ms | 2.02 |
| `transpose` (Aᵀ) | randunsym-1000000, n = 1000000, nnz = 8999966 | 67.8 ms | SciPy A.tocsc() | 56.4 ms | 1.20 |
| `transpose` (Aᵀ) | lap3d-100, n = 1000000, nnz = 6940000 | 13.6 ms | CHOLMOD transpose | 8.84 ms | 1.53 |
| `transpose` (Aᵀ) | elast3d-30, n = 81000, nnz = 6133248 | 10.8 ms | CHOLMOD transpose | 7.42 ms | 1.46 |
| `assembly` (triplets → CSR) | FE assembly 300×300 Q1, 1440000 triplets | 9.03 ms | CSparse cs_compress + cs_dupl + sort | 5.09 ms | 1.77 |
| `from_coo` (tuples → CSR) | FE assembly 300×300 Q1, 1440000 triplets | 8.69 ms | CHOLMOD triplet_to_sparse | 6.12 ms | 1.42 |
| `assembly` (triplets → CSR) | FE assembly 1000×1000 Q1, 16000000 triplets | 91.1 ms | CSparse cs_compress + cs_dupl + sort | 63.7 ms | 1.43 |
| `from_coo` (tuples → CSR) | FE assembly 1000×1000 Q1, 16000000 triplets | 93.6 ms | CHOLMOD triplet_to_sparse | 66.3 ms | 1.41 |
| `assembly` (triplets → CSR) | random triplets n=1000000, 10000000 triplets | 252 ms | Accelerate SparseConvertFromCoordinate | 115 ms | 2.19 |
| `from_coo` (tuples → CSR) | random triplets n=1000000, 10000000 triplets | 204 ms | CHOLMOD triplet_to_sparse | 172 ms | 1.19 |
| `permute` (P·A·Pᵀ) | lap2d-1000, n = 1000000, nnz = 4996000 | 74.5 ms | CHOLMOD submatrix (A(p,p), sorted) | 88.5 ms | 0.842 |
| `permute` (P·A·Pᵀ) | elast3d-30, n = 81000, nnz = 6133248 | 56.6 ms | SciPy A[p][:, p] | 24.4 ms | 2.32 |
| `submatrix` (ranges, middle half) | lap2d-1000, n = 1000000, nnz = 4996000 | 6.90 ms | SciPy A[r0:r1, c0:c1] | 2.86 ms | 2.42 |
| `select` (random half of rows and columns) | lap2d-1000, n = 1000000, nnz = 4996000 | 65.8 ms | CHOLMOD submatrix (index lists) | 52.1 ms | 1.26 |
| `upper_triangle` | randunsym-1000000, n = 1000000, nnz = 8999966 | 20.9 ms | CHOLMOD band | 14.2 ms | 1.47 |
| `lower_triangle` | randunsym-1000000, n = 1000000, nnz = 8999966 | 20.9 ms | CHOLMOD band | 14.1 ms | 1.48 |
| `norm(Max)` | randunsym-1000000, n = 1000000, nnz = 8999966 | 2.53 ms | SciPy abs(A).max() | 6.02 ms | 0.420 |
| `to_matrix` (sparse → dense) | randunsym, n = 2000, nnz = 41823 | 209 µs | nalgebra-sparse (DMatrix::from) | 196 µs | 1.07 |
| `from_matrix` (dense → sparse) | randunsym, n = 2000, nnz = 41823 | 3.27 ms | nalgebra-sparse (CsrMatrix::from(&DMatrix)) | 2.83 ms | 1.16 |
| `is_symmetric` | lap3d-100, n = 1000000, nnz = 6940000 | 10.3 ms | CHOLMOD symmetry (option 1) | 17.9 ms | 0.573 |
| `is_symmetric_pattern` | lap3d-100, n = 1000000, nnz = 6940000 | 6.63 ms | CHOLMOD symmetry (option 1) | 18.0 ms | 0.369 |
| `diagonal` | lap3d-100, n = 1000000, nnz = 6940000 | 2.83 ms | SciPy A.diagonal() | 2.38 ms | 1.19 |
| `from_csr` (validated arrays) | lap2d-1000, n = 1000000, nnz = 4996000 | 3.58 ms | faer (new_checked) | 3.17 ms | 1.13 |
| `scale` (A·α) | randunsym-100000, n = 100000, nnz = 899953 | 365 µs | SciPy A * 1.5 | 193 µs | 1.89 |

`norm(One)`, `norm(Infinity)` and `norm(Frobenius)` have no reference that does the same work: every reference sums plainly, where these sum with compensation and the Frobenius norm also scales; against them ours takes 1.30 to 1.75 (`One`), 0.563 to 0.662 (`Infinity`) and 2.01 to 18.3 (`Frobenius`) of their time.

Where the references are faster:

- `mul_vec` on the Laplacian of 10⁴ unknowns (1.99 times faer with `u32` indices) and on the circuit of 10⁶ (1.11): the indices are `usize`, 16 bytes a nonzero with its value against faer's 12. On the small Laplacian, whose vectors stay in the caches, our samples spread widely (by 34 %), and the same product timed in the SciPy process took 17.4 µs, 0.744 of SciPy's: that gap is not settled. `mul_vec_transpose` (1.07 and 1.11) and `CscMatrix::mul_vec` (1.21) against Accelerate's kernels; against the fastest of the others 0.957, 0.982 and 1.15.
- `mul_matrix` with 16 and 64 columns on the Laplacian (1.12 and 1.52 times sprs's product into a dense matrix held by rows): `X` is held by columns, so a row of `A` meets the `k` columns of `X` one after another, where sprs reads a row of its `X` in one stretch; with 4 columns, and on the random matrix, ours ties CHOLMOD (0.987 to 1.02).
- `sparse × sparse` on the 2-D Laplacian (1.24 times CHOLMOD's sorted product; 0.999 to 1.06 elsewhere) and `add` (1.34 and 1.35 times CHOLMOD's sorted sum): the columns of each row of the result are sorted as the row is made, where CHOLMOD sorts by transposing.
- `scale` (1.89 and 2.02 times SciPy's `A * 1.5`): the canonical form is kept, a product that underflows to zero dropped, where SciPy multiplies its array of values.
- `transpose` (1.20 to 1.53 times CHOLMOD's and SciPy's, the same counting sort) and `upper_triangle` and `lower_triangle` (1.47 and 1.48 times CHOLMOD's `band`): every element moved is 16 bytes, against the references' 12.
- `assembly` and `from_coo` (1.41 to 1.77 times CSparse's compression with its sort and CHOLMOD's `triplet_to_sparse` on the meshes, 1.19 and 2.19 times CHOLMOD and Accelerate on random triplets): radix sorts of the triplets by row and by column, on 64-bit indices, with the repeats summed in the order given.
- `permute` of the elasticity matrix (2.32 times SciPy's indexing; 0.842 of CHOLMOD's on the Laplacian), `submatrix` (2.42 times SciPy's slice), `select` (1.26 times CHOLMOD's), `diagonal` (1.19 times SciPy's), `from_csr` (1.13 times faer's checked constructor, which does not look for stored zeros), `to_matrix` and `from_matrix` (1.07 and 1.16 times nalgebra-sparse's): the constant factors of the passes that keep the canonical form, not examined one by one for the release.

## Matrix Market files

`read_matrix_market(reader)` reads a Matrix Market file (Boisvert, Pozo and Remington, *The Matrix Market exchange formats: initial design*, NISTIR 5935, 1996) from any `BufRead` into a `CooMatrix<T>`, `parse_matrix_market(&str)` from a string, and `SparseMatrix::read_matrix_market` and `SparseMatrix::from_matrix_market` straight into compressed rows. `write_matrix_market(&m, writer)` and `to_matrix_market(&m)` write a `SparseMatrix`, `CscMatrix` or `CooMatrix` in coordinate format and a dense `Matrix` in array format; the `_with` forms take `MatrixMarketOptions`: a symmetry, of which the lower triangle alone is written, a field, and a comment.

- Both formats, `coordinate` and `array`; the fields `real`, `integer`, `complex` and `pattern`; the symmetries `general`, `symmetric`, `skew-symmetric` and `hermitian`, expanded to both triangles on reading (a skew-symmetric or Hermitian mirror negated or conjugated; the values of a file that is not `complex` stay real, so a complex type gets the imaginary part `+0` for them). A `pattern` file gives ones. A `complex` file read into a real type is an error; the other fields read into a complex type.
- Keywords in any case, `%` comment lines and blank lines anywhere after the banner, a byte order mark, any whitespace, Windows line ends, exponents marked by `e`, `E`, `d` or `D`, `inf` and `nan`. Each number is rounded once from its digits to the element type, so an `f32` is never rounded twice through an `f64`. An `integer` file holds integers alone.
- Numbers are written by `{:e}`, the shortest digits that read back as the same number, so that a file read again gives the matrix bit for bit (a NaN comes back as a NaN, and of a dense file, whose zeros are not stored on reading, a `−0` as a zero). An `integer` field is written in full digits and refuses a value with a fraction, an infinity or a NaN; a `real` one refuses an imaginary part; a symmetry is refused for a matrix that hasn't it, a NaN mirroring nothing.
- `MatrixMarketError` names what is wrong and, for a file, its line: a bad banner, an unknown keyword, a pattern in array format, a complex file for a real type, a bad size line or a count no `usize` holds, a symmetry declared for a matrix that is not square, an entry with the wrong fields, a number that isn't one, an index outside the matrix, more or fewer entries than declared, an entry above the diagonal of a file with a symmetry, a nonzero on the diagonal of a skew-symmetric file, an imaginary part on the diagonal of a Hermitian one, and a failure of the reader or the writer. Nothing in a file makes the reader panic. Whatever count a file declares, the reader makes room for at most 2²⁰ triplets before it has read them; compressed rows need an offset for every row a file declares, which `SparseMatrix::read_matrix_market` checks it can reserve before it converts, and is `TooLarge` otherwise.
- Three banners that the definition leaves out are read for what they can only mean, as scipy reads them: `real hermitian` (and `integer hermitian`) is symmetric, `pattern hermitian` is a symmetric pattern, and `pattern skew-symmetric` has −1 above the diagonal. The writer writes `symmetric` for a Hermitian matrix of real values and never writes a skew-symmetric pattern.

### Example

```rust
use a::math::sparse::{
    CooMatrix, CscMatrix, MatrixMarketOptions, MatrixMarketSymmetry, SparseMatrix,
    parse_matrix_market, to_matrix_market_with,
};

// The 1-D Laplacian of five points, assembled from the element matrices of its four intervals:
// the triplets of one cell are summed.
let mut coo = CooMatrix::new(5, 5);
for k in 0..4 {
    coo.push(k, k, 1.0);
    coo.push(k, k + 1, -1.0);
    coo.push(k + 1, k, -1.0);
    coo.push(k + 1, k + 1, 1.0);
}
coo.push(0, 0, 1.0);
coo.push(4, 4, 1.0);
let a: SparseMatrix<f64> = coo.to_csr();
assert_eq!(a.nnz(), 13);
assert_eq!(a, SparseMatrix::tridiagonal(&[-1.0; 4], &[2.0; 5], &[-1.0; 4]));

// By columns; the products by the matrix and by its transpose, which is never formed.
let c: CscMatrix<f64> = a.to_csc();
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
assert_eq!(a.mul_vec(&x), [0.0, 0.0, 0.0, 0.0, 6.0]);
assert_eq!(c.mul_vec(&x), a.mul_vec(&x));
assert_eq!(a.mul_vec_transpose(&x), a.mul_vec(&x));

// A block, and the symmetric permutation that reverses the order: p[new] = old.
let block = a.submatrix(1..3, 0..3);
assert_eq!(block.to_matrix().as_slice(), &[-1.0, 0.0, 2.0, -1.0, -1.0, 2.0]);
let p = [4, 3, 2, 1, 0];
assert_eq!(a.permute(&p, &p), a);

// A symmetric Matrix Market file holds the lower triangle alone, and reads back whole.
let options = MatrixMarketOptions::new().with_symmetry(MatrixMarketSymmetry::Symmetric);
let text = to_matrix_market_with(&a, &options).unwrap();
assert!(text.starts_with("%%MatrixMarket matrix coordinate real symmetric\n5 5 9\n"));
assert_eq!(parse_matrix_market::<f64>(&text).unwrap().to_csr(), a);
```

### Measured

Against SciPy 1.18.0 (`scipy.io.mmwrite` and `scipy.io.mmread`, the fast_matrix_market reader and writer) with NumPy 2.3.3, by `tests/math/sparse/data/gen_matrix_market.py`:

- **Reading**: the 22 files `scipy.io.mmwrite` writes for every combination of format, field and symmetry the format defines (coordinate: real, integer and complex in general, symmetric and skew-symmetric, complex Hermitian, pattern general and symmetric; array: the same but for the patterns), of 5 × 4 to 7 × 7 matrices holding repeated cells, explicit zeros and numbers at the edges of both precisions, are each read into the four element types (a complex file into a real type is the error it should be): every cell is bit for bit the value `scipy.io.mmread` reads, signs of zeros included, and, in the `f32` types, the correctly rounded `f32` of the file's digits, which the script computes exactly from fractions (checked against mpmath at 24 bits on 20 011 decimals: no difference). The samples hold 70 cells where rounding through `f64` would give another `f32`, from `1.0000000596046448` (an `f32` tie as an `f64`, above it as a decimal) and `3.4028235677973366E38` (the `f32` overflow threshold as an `f64`, below it as a decimal).
- **Writing**: the 44 files the writer makes of those matrices, from `f64` and `Complex<f64>` and from `f32` and `Complex<f32>`, are read by `scipy.io.mmread`: 1 374 cells compared, no difference (the `f64` files bit for bit; the `f32` files through the exact rounding of their digits). The files are kept in `data/ours`, and the tests check that the writer still writes them byte for byte.
- **Round trips**: random matrices of every shape up to 7 × 7, the empty ones included, written from each of the four forms and, when square, as symmetric, skew-symmetric and Hermitian, read back bit for bit in all four element types.
- Seen in SciPy, not in this reader: its coordinate reader mirrors the complex skew-symmetric entry `−2.5e−7 + ∞·i` as `NaN − ∞·i` (a multiplication by −1 where a negation is meant); its array reader reads `−0` as `+0`; it reads `1.5D+02` as 1.5 and `0x1p3` as 0, and truncates `1.5` in an integer file to 1; and it refuses a banner written in lower case, `%%matrixmarket`.

Against the crate's own dense `Matrix`:

- Products of matrices of small integers, which no element type rounds, on 300 shapes up to 6 × 6 (empty ones included) for each element type: sparse × sparse, sparse × dense, dense × sparse, `A·x`, `Aᵀ·x` and `Aᴴ·x` are the dense products bit for bit; on random matrices they agree to `16·ε·n` of the norms.
- Every conversion, transpose, sub-matrix, selection, triangle and permutation is the dense one bit for bit, and the canonical form holds after every operation.
- Norms: the `f32` Frobenius norm of the 2048 × 2048 matrix of ones is 2048 and that of the 5-point Laplacian of the 256 × 256 grid 1144.4196, the correctly rounded `√1 309 696` (the earlier `hypot` chain gave 2020.7085 and 1145.6541); a row of `2²⁴` and a thousand ones has the `f32` row sum `16 778 216`, and one of `2⁵³` and a thousand ones the `f64` sum `2⁵³ + 1000`, exactly. `|3e200 − 4e200·i|` is `4.9999999999999995e200`, the correctly rounded value (mpmath).

To reproduce: `python3 lib/tests/math/sparse/data/gen_matrix_market.py write` regenerates `data/scipy` and `data/expected`; `A_MATRIX_MARKET_OUT=<dir> cargo test --test math -- sparse::io::the_files_written` writes the files of `data/ours` into `<dir>`, and `python3 lib/tests/math/sparse/data/gen_matrix_market.py check <dir>` reads them with scipy; `cargo test --test math -- sparse::` runs the rest.

**Time** (conditions as under [Storage and operations](#storage-and-operations)): reading files from the page cache and writing them, with SciPy's `fast_matrix_market` on one thread:

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `read_matrix_market` (to COO) | lap2d-316, n = 99856, nnz = 498016, general file of 8.3 MB | 27.1 ms | SciPy io.mmread (fast_matrix_market, 1 thread) | 19.3 ms | 1.41 |
| `SparseMatrix::read_matrix_market` (to CSR) | lap2d-316, n = 99856, nnz = 498016, general file of 8.3 MB | 30.0 ms | SciPy io.mmread(...).tocsr() | 21.2 ms | 1.42 |
| `write_matrix_market` | lap2d-316, n = 99856, nnz = 498016, general file of 8.3 MB | 23.9 ms | sprs write_matrix_market_to_bufwrite | 21.6 ms | 1.11 |
| `read_matrix_market` (to COO) | lap2d-316, n = 99856, nnz = 498016, symmetric file of 4.9 MB | 17.3 ms | SciPy io.mmread (fast_matrix_market, 1 thread) | 13.5 ms | 1.28 |
| `SparseMatrix::read_matrix_market` (to CSR) | lap2d-316, n = 99856, nnz = 498016, symmetric file of 4.9 MB | 20.2 ms | SciPy io.mmread(...).tocsr() | 14.6 ms | 1.39 |
| `read_matrix_market` (to COO) | lap2d-1000, n = 1000000, nnz = 4996000, general file of 92.8 MB | 288 ms | SciPy io.mmread (fast_matrix_market, 1 thread) | 202 ms | 1.42 |
| `SparseMatrix::read_matrix_market` (to CSR) | lap2d-1000, n = 1000000, nnz = 4996000, general file of 92.8 MB | 326 ms | SciPy io.mmread(...).tocsr() | 222 ms | 1.47 |
| `write_matrix_market` | lap2d-1000, n = 1000000, nnz = 4996000, general file of 92.8 MB | 249 ms | sprs write_matrix_market_to_bufwrite | 225 ms | 1.11 |
| `read_matrix_market` (to COO) | lap2d-1000, n = 1000000, nnz = 4996000, symmetric file of 55.3 MB | 174 ms | SciPy io.mmread (fast_matrix_market, 1 thread) | 139 ms | 1.25 |
| `SparseMatrix::read_matrix_market` (to CSR) | lap2d-1000, n = 1000000, nnz = 4996000, symmetric file of 55.3 MB | 208 ms | SciPy io.mmread(...).tocsr() | 151 ms | 1.38 |
| `read_matrix_market` (to COO) | randunsym-1000000, n = 1000000, nnz = 8999966, general file of 317.3 MB | 722 ms | SciPy io.mmread (fast_matrix_market, 1 thread) | 461 ms | 1.57 |
| `SparseMatrix::read_matrix_market` (to CSR) | randunsym-1000000, n = 1000000, nnz = 8999966, general file of 317.3 MB | 937 ms | SciPy io.mmread(...).tocsr() | 492 ms | 1.91 |
| `write_matrix_market` | randunsym-1000000, n = 1000000, nnz = 8999966, general file of 317.3 MB | 750 ms | sprs write_matrix_market_to_bufwrite | 665 ms | 1.13 |
| `read_matrix_market` (to COO) | elast3d-30, n = 81000, nnz = 6133248, general file of 102.3 MB | 354 ms | SciPy io.mmread (fast_matrix_market, 1 thread) | 241 ms | 1.47 |
| `SparseMatrix::read_matrix_market` (to CSR) | elast3d-30, n = 81000, nnz = 6133248, general file of 102.3 MB | 410 ms | SciPy io.mmread(...).tocsr() | 264 ms | 1.55 |
| `write_matrix_market` | elast3d-30, n = 81000, nnz = 6133248, general file of 102.3 MB | 300 ms | sprs write_matrix_market_to_bufwrite | 273 ms | 1.10 |
| `read_matrix_market` (to COO) | elast3d-30, n = 81000, nnz = 6133248, symmetric file of 52.0 MB | 191 ms | SciPy io.mmread (fast_matrix_market, 1 thread) | 135 ms | 1.41 |
| `SparseMatrix::read_matrix_market` (to CSR) | elast3d-30, n = 81000, nnz = 6133248, symmetric file of 52.0 MB | 235 ms | SciPy io.mmread(...).tocsr() | 150 ms | 1.57 |
| `write_matrix_market` | lap2d-316, n = 99856, nnz = 498016, symmetric file of 4.9 MB | 15.7 ms | SciPy io.mmwrite (fast_matrix_market, 1 thread) | 20.0 ms | 0.787 |
| `write_matrix_market` | lap2d-1000, n = 1000000, nnz = 4996000, symmetric file of 55.3 MB | 156 ms | SciPy io.mmwrite (fast_matrix_market, 1 thread) | 194 ms | 0.802 |
| `write_matrix_market` | elast3d-30, n = 81000, nnz = 6133248, symmetric file of 52.0 MB | 165 ms | SciPy io.mmwrite (fast_matrix_market, 1 thread) | 206 ms | 0.802 |

Reading takes 1.25 to 1.57 times SciPy's time into triplets and 1.38 to 1.91 times into compressed rows, and writing a general file 1.10 to 1.13 times sprs's (a symmetric one 0.787 to 0.802 of SciPy's): the number parser, which rounds each decimal once into the element type (into an `f32` without passing through an `f64`), is slower than `fast_matrix_market`'s, and the read into compressed rows adds the sort of the triplets.

### Limits

- `norm(Norm::Two)` and `norm(Norm::Nuclear)` form the dense matrix and its singular values: `O(rows·cols)` memory.
- `SparseMatrix::column(j)` searches every row for the column, `O(rows·log)`; for more than a few columns, `to_csc()` gives them all in one pass. `a[(i, j)]` and `find(i, j)` are a binary search of a row.
- The product of two sparse matrices sorts the columns of each row of the result, a logarithm over the linear cost, and keeps two accumulators as long as a row of the result, as SciPy does, so that a second factor of some 2⁴⁰ columns or more cannot be multiplied.
- Duplicate entries are summed in the order given, so two orders of the same entries can differ by rounding, and even in the pattern (`1 + 1e-17 − 1` stores nothing, `1 − 1 + 1e-17` stores `1e-17`).
- A matrix with more rows than a vector of offsets can hold (`usize::MAX`) cannot be held by rows; it panics with that message, the same in every build. The triplets and the compressed columns of such a matrix are fine when its columns are few.
- The values of the compressed forms are read only: `map`, or `into_parts` and `from_csr`, change them.
- A file that declares a vast number of rows makes `SparseMatrix::read_matrix_market` allocate their offsets, twice, during the conversion; it checks that one such vector can be reserved, but an allocation within the allocator's reach can still exhaust memory. The triplets of `read_matrix_market` allocate nothing for each row; `write_matrix_market` of a `CooMatrix`, though, converts it to compressed rows first, and so takes memory for every row it declares.
- The reader is stricter than scipy's: an extra field on a line, an entry above the diagonal of a symmetric file, a nonzero on a skew-symmetric diagonal, an imaginary part on a Hermitian diagonal and a point or an exponent in an `integer` file are errors there, where scipy ignores, mirrors, keeps or truncates them. An `integer` field is written in full digits even beyond 2⁶³, which a reader into 64-bit integers, such as scipy's, refuses.

## Orderings

What a direct method costs is settled before it does any arithmetic, by the order in which it eliminates the unknowns: the fill of the Cholesky factor of `P·A·Pᵀ`, and of the LU or QR factors of `A·Q`, depends on the permutation and on the pattern of `A` alone. The orderings read the pattern and nothing else — which cells store an element, read for the symmetric ones as the pattern of `A + Aᵀ` without its diagonal — and return a permutation `p` with `p[new] = old`. They take any of the four element types and give the same permutation for any values in the same pattern; a matrix, its transpose and its symmetric part have the same symmetric orderings.

| Function | Returns | For |
| --- | --- | --- |
| `reverse_cuthill_mckee(&a)` | `p` that narrows the band and shortens the profile of `P·A·Pᵀ` | band and profile solvers; a matrix numbered badly by a mesh generator |
| `amd(&a)` | `p` that keeps the fill of the Cholesky factor of `P·A·Pᵀ` low, in a postorder of its elimination tree | the symmetric factorisations: the default ordering |
| `colamd(&a)` | `q` of the columns of a matrix of any shape that keeps the fill of the Cholesky factor of `(A·Q)ᴴ·(A·Q)` low, and with it that of LU and QR, in a postorder of the column elimination tree | LU with partial pivoting and QR, which choose their rows themselves |
| `nested_dissection(&a)` | `p` in which every separator is numbered after the two parts it separates | large meshes, in three dimensions above all; a short and wide elimination tree |
| `Ordering` | `Natural`, `ReverseCuthillMckee`, `Amd` (the default), `Colamd`, `NestedDissection` or `Given(p)`; `symmetric(&a)` and `columns(&a)` give its permutation | naming an ordering for a factorisation |
| `elimination_tree(&a)` | `parent[j]` of every column of the Cholesky factor of `A + Aᵀ`, `None` for a root | the dependencies between the columns; a parent has a larger index than its children |
| `postorder(&parent)` | the vertices of a forest with every vertex after its descendants and every subtree in one run, as a permutation | renumbering by the elimination tree, which leaves the fill as it is |
| `inverse_permutation(&p)` | `q` with `q[p[k]] = k` | scattering back what `p` gathered; relabelling indices |
| `symmetric_permutation(&a, &p)` | `P·A·Pᵀ` | applying a symmetric ordering, in the time the stored elements take |
| `connected_components(&a)` | the block of every row, numbered from 0 in the order the blocks are met | the independent blocks, the pattern read both ways |
| `band_width(&a)`, `profile(&a)` | the largest distance of a stored element from the diagonal; the envelope of `A + Aᵀ`, the diagonal counted | judging an ordering |

What each `Ordering` means for the two kinds of factorisation:

| Variant | `symmetric(&a)`: Cholesky, `L·D·Lᴴ` | `columns(&a)`: LU, QR |
| --- | --- | --- |
| `Natural` | the order the matrix is in | the order the columns are in |
| `ReverseCuthillMckee` | `reverse_cuthill_mckee(&a)` | the same, of the pattern of `Aᴴ·A` |
| `Amd` | `amd(&a)` | the same, of the pattern of `Aᴴ·A` |
| `Colamd` | `colamd` of the matrix with a row for every edge of the graph, holding its two ends, whose `Mᵀ·M` has the pattern of `A + Aᵀ` (SYMAMD) | `colamd(&a)` |
| `NestedDissection` | `nested_dissection(&a)` | the same, of the pattern of `Aᴴ·A` |
| `Given(p)` | `p`, checked to be a permutation | `p`, checked |

The pattern of `Aᴴ·A` is formed from the pattern of `A`, so nothing cancels out of it, at the cost of the sum of the squares of the row lengths; a full row makes it full. `Colamd` never forms it and sets such rows aside, which is why it is the column ordering to give LU and QR.

### Methods

- **Reverse Cuthill-McKee** (Cuthill and McKee, "Reducing the bandwidth of sparse symmetric matrices", ACM National Conference 1969; George and Liu, *Computer Solution of Large Sparse Positive Definite Systems*, 1981). Every connected block is numbered by itself from a pseudo-peripheral vertex, found by the iteration of George and Liu: the vertex of least degree in the last level of a breadth-first search becomes the start while its own level structure is deeper — two to four searches as a rule, twelve at most. The Cuthill-McKee numbering then takes the vertices in the order they were numbered and appends the unnumbered neighbours of each by increasing degree, then index, and the whole order is reversed, which leaves the band as it is and never lengthens the profile (Liu and Sherman, SIAM J. Numer. Anal. 13, 1976). One array of marks serves every search, so a block costs what it holds: `O(nnz·log(largest degree) + n)` in all.
- **AMD** (Amestoy, Davis and Duff, "An approximate minimum degree ordering algorithm", SIAM J. Matrix Anal. Appl. 17, 1996). Minimum degree on the quotient graph, where an eliminated vertex stays as an element standing for the clique its elimination made, so that nothing is stored for the fill: the approximate external degree of the paper, `min(n − k, d̄ᵢ + |Lp \ i|, |Aᵢ \ i| + |Lp \ i| + Σ |Le \ Lp|)` with the weights of supervariables; mass elimination; indistinguishable variables found by hashing their lists and merged into supervariables; element absorption and aggressive absorption; rows of more than `max(16, 10·√n)` elements off the diagonal set aside and put after the others; and the pivots put out in a postorder of the assembly tree, the tree of the elements and of what absorbed them. All the lists live in one array of the size of the pattern, a fifth more and one entry for every vertex, compacted when its free end runs out; nothing is allocated per vertex. The assembly tree is not the elimination tree — an element absorbed aggressively, or a row set aside, hangs elsewhere in the one than in the other — so last the whole order is renumbered by a postorder of the elimination tree of `P·A·Pᵀ`, built by Liu's algorithm. Every order that keeps each vertex of that tree after its descendants has the same fill (Liu 1990, below), and a postorder is the one that puts the columns of every subtree next to each other, as supernodal and multifrontal factorisations want; a row set aside still follows every row it is joined to.
- **COLAMD** (Davis, Gilbert, Larimore and Ng, "A column approximate minimum degree ordering algorithm", ACM Trans. Math. Softw. 30, 2004). `A` itself, its rows read as elements and its columns as variables, is a quotient graph of `Aᵀ·A`, and the elimination of AMD runs on it: supercolumns, absorption of rows, degrees bounded by the set differences of the rows with the pivot row, the initial degree of a column the sum of the lengths of its rows less one each. A column of more than `max(16, 10·√min(m, n))` elements is put after the others, a row of more than `max(16, 10·√n)` elements in the remaining columns is ignored, a column whose rows are all ignored is put just before the dense ones, and an empty column last of all. The pivots come out in a postorder of COLAMD's own assembly tree, of the rows and the pivot rows that absorbed them, and last the order is renumbered by a postorder of the column elimination tree of `A·Q`, the elimination tree of `(A·Q)ᵀ·(A·Q)`. That tree is built from the rows of `A`, the ignored ones too since they are part of `Aᵀ·A`: in Liu's algorithm each column is joined, for each of its rows, to the column of that row met just before it, which gives the tree of `Aᵀ·A` without forming it. As with AMD the fill is unchanged, every subtree of the tree is one run of columns, a column set aside still follows every column it shares a row with, and the empty ones stay last. `Aᵀ·A` is never formed.
- **Nested dissection** (George, "Nested dissection of a regular finite element mesh", SIAM J. Numer. Anal. 10, 1973; George and Liu, "An automatic nested dissection algorithm for irregular finite element problems", SIAM J. Numer. Anal. 15, 1978). The blocks of the graph are taken one by one, and a connected piece of more than 128 vertices gets two separators: the part of a level of the breadth-first search from a pseudo-peripheral vertex that touches the next level, the level chosen for few such vertices and a quarter of the piece on either side; and a multilevel bisection after Karypis and Kumar ("A fast and high quality multilevel scheme for partitioning irregular graphs", SIAM J. Sci. Comput. 20, 1998): matchings of the heaviest edges contracted down to about a hundred vertices, cuts grown breadth-first there from six starting vertices, and at every level on the way back the vertex-separator form of the refinement of Fiduccia and Mattheyses, each part kept within three fifths of the whole. Both are improved by the matching method of Liu ("A graph partitioning algorithm by node separators", ACM Trans. Math. Softw. 15, 1989) — when a largest matching between a separator and its neighbours in one part leaves some of it unmatched, the smallest vertex cover of that bipartite graph is a smaller separator — and the smaller of the two is kept. The parts are ordered the same way, the separator after both. A piece of at most 128 vertices, or one whose best separator holds more than an eighth of it or leaves less than an eighth of the rest on one side, is ordered by AMD on the graph it induces; rows of more than `max(16, 10·√n)` elements are set aside first and ordered last. Since every cut leaves an eighth of a piece on either side, the recursion is at most `log n / log(8/7)` deep.
- **Elimination tree** (Liu, "The role of elimination trees in sparse factorization", SIAM J. Matrix Anal. Appl. 11, 1990). The rows are taken in order and, from every element `(i, k)` with `k < i`, the path of ancestors of `k` is followed to its root, which becomes a child of `i`; the paths are compressed as they are walked. The factor is never formed: the cost is within a logarithmic factor of the stored elements. The result is a `Vec<Option<usize>>`, `None` marking a root, so that a forest of several blocks needs no convention.
- **Postorder**: a depth-first search kept on a stack of its own, the children of a vertex and the roots taken in ascending order; a parent array that closes a cycle, or points outside the forest, panics.

### Example

```rust
use a::math::sparse::{
    Ordering, SparseMatrix, amd, band_width, colamd, elimination_tree, nested_dissection,
    postorder, reverse_cuthill_mckee, symmetric_permutation,
};

// The five-point Laplacian of a 20 × 20 grid, its vertices numbered badly.
let (k, n) = (20, 400);
let label = |x: usize, y: usize| (7 * (y * k + x) + 3) % n;
let mut entries = Vec::new();
for y in 0..k {
    for x in 0..k {
        let v = label(x, y);
        entries.push((v, v, 4.0));
        for w in [(x + 1 < k).then(|| label(x + 1, y)), (y + 1 < k).then(|| label(x, y + 1))]
            .into_iter()
            .flatten()
        {
            entries.extend([(v, w, -1.0), (w, v, -1.0)]);
        }
    }
}
let a = SparseMatrix::from_coo(n, n, &entries);

// Reverse Cuthill-McKee brings the band back to the width of the grid.
let b = symmetric_permutation(&a, &reverse_cuthill_mckee(&a));
assert!(band_width(&a) > 300);
assert_eq!(band_width(&b), k);

// The nonzeros of the Cholesky factor of P·A·Pᵀ, counted along its elimination tree: row i
// reaches the columns on the paths up the tree from its elements below the diagonal.
let fill = |p: &[usize]| {
    let b = symmetric_permutation(&a, p);
    let parent = elimination_tree(&b);
    let mut mark = vec![usize::MAX; n];
    let mut count = n;
    for i in 0..n {
        mark[i] = i;
        for (mut j, _) in b.row(i) {
            while j < i && mark[j] != i {
                mark[j] = i;
                count += 1;
                j = parent[j].unwrap();
            }
        }
    }
    count
};
let natural: Vec<usize> = (0..n).collect();
assert!(3 * fill(&amd(&a)) < fill(&natural));
assert!(3 * fill(&nested_dissection(&a)) < fill(&natural));

// amd puts out a postorder of the elimination tree of P·A·Pᵀ: every subtree is one run of
// columns already, so a postorder of the tree leaves the order as it is.
let p = amd(&a);
let identity: Vec<usize> = (0..n).collect();
assert_eq!(postorder(&elimination_tree(&symmetric_permutation(&a, &p))), identity);

// Any other order can be renumbered so, and keeps its fill: the order the grid came in, say.
let post = postorder(&elimination_tree(&a));
assert_ne!(post, natural);
assert_eq!(fill(&post), fill(&natural));

// An Ordering names one of them for a factorisation; LU and QR order columns, of any shape.
assert_eq!(Ordering::Amd.symmetric(&a), p);
let tall = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0), (1, 0, 1.0), (1, 1, 1.0), (2, 1, 1.0)]);
let mut q = colamd(&tall);
q.sort_unstable();
assert_eq!(q, [0, 1]);
```

### Measured

The numbers below were measured, not bounded. The references come from `lib/tests/math/sparse/data/gen_ordering.py`, which builds the same graphs as `lib/tests/math/sparse/ordering.rs`, from the same generator: SciPy 1.18's `scipy.sparse.csgraph.reverse_cuthill_mckee`, SuperLU's `MMD_AT_PLUS_A` and `COLAMD` through `scipy.sparse.linalg.splu` (its `perm_c` read as `perm_c[old] = new`, which reproduces SuperLU's own `L.nnz`), and a true minimum degree ordering on the elimination graph written for the purpose, its ties to the lowest index. Fill is nnz(L), the diagonal counted, from the pattern alone. `cargo test --test math -- sparse::ordering:: --nocapture` prints the rows of the first three tables, and `cargo test --release --test math -- --ignored sparse::ordering --nocapture` those of the grids and the times; the tests insist on the ratios with a margin (1.1 of SciPy's band and profile, 1.1 of the better reference for AMD, 1.15 for COLAMD).

Reverse Cuthill-McKee against SciPy's, band and profile. SciPy starts each block from a vertex of least degree, this from a pseudo-peripheral one, and on the regular graphs the two give the same band and profile:

| Graph | band | SciPy | profile | SciPy |
| --- | --- | --- | --- | --- |
| the 30 × 30 grid, numbered by rows | 30 | 30 | 19,315 | 19,315 |
| the full band of half-width 9, n = 120 | 9 | 9 | 1,155 | 1,155 |
| the 40 × 25 grid, scrambled | 25 | 25 | 21,075 | 21,075 |
| the 20 × 20 nine-point grid, scrambled | 39 | 39 | 10,147 | 10,147 |
| the 8 × 8 × 8 grid, scrambled | 52 | 52 | 19,566 | 19,566 |
| the 12 × 12 × 12 grid | 114 | 114 | 143,595 | 143,595 |
| a random graph, n = 500, m = 1000 | 203 | 194 | 44,909 | 46,460 |
| a random graph, n = 2000, m = 8000 | 1,257 | 1,219 | 1,240,835 | 1,228,526 |
| a grid, a path, a random graph and 20 isolated vertices, scrambled | 49 | 55 | 3,882 | 4,073 |
| the arrow of one hub, n = 300 | 298 | 298 | 599 | 599 |
| the binary tree of 511 vertices | 128 | 128 | 17,818 | 17,818 |
| the 64 × 64 grid | 64 | 64 | 180,832 | 180,832 |
| the 48 × 48 grid, scrambled | 48 | 48 | 77,128 | 77,128 |
| the arrow of three hubs, n = 400, scrambled | 398 | 398 | 1,594 | 1,594 |
| the band of half-width 4, n = 500, scrambled | 4 | 4 | 2,490 | 2,490 |

Band and profile are SciPy's on twelve of these graphs, and the profile is within 0.953 to 1.010 of SciPy's on all fifteen. On 71 further graphs, made by an independent review of this code — random graphs of 50 to 2,000 vertices and average degree 1 to 8, random trees, grids and cubes numbered well and badly, power-law and geometric graphs of up to 5,000 vertices, a grid with three dense rows, a union of graphs, unsymmetric patterns — the profile is 0.826 to 1.117 of SciPy's and the band at most 10 against SciPy's 7, on a random graph of 50 vertices: the same class, the two choosing their starting vertices differently. On all 71 a re-implementation of the procedure described above, written for the review, gives the same permutation. A Cuthill-McKee order is a definition, and the tests check it on the fifteen graphs and on sixty random ones: the first vertex of every block touches nothing numbered before it, and the unnumbered neighbours of each vertex come next, together, by degree and then index.

AMD against SuperLU's multiple minimum degree and the exact minimum degree, nnz(L):

| Graph | natural | SuperLU MMD | exact | `amd` | / MMD | / exact |
| --- | --- | --- | --- | --- | --- | --- |
| the 30 × 30 grid | 27,029 | 10,098 | 10,351 | 10,209 | 1.011 | 0.986 |
| the full band of half-width 9 | 1,155 | 1,158 | 1,155 | 1,165 | 1.006 | 1.009 |
| the 40 × 25 grid, scrambled | 79,345 | 11,688 | 11,831 | 11,423 | 0.977 | 0.966 |
| the 20 × 20 nine-point grid, scrambled | 17,527 | 6,061 | 6,213 | 6,257 | 1.032 | 1.007 |
| the 8 × 8 × 8 grid, scrambled | 38,484 | 12,117 | 12,274 | 11,451 | 0.945 | 0.933 |
| the 12 × 12 × 12 grid | 231,419 | 74,273 | 79,517 | 70,825 | 0.954 | 0.891 |
| a random graph, n = 500, m = 1000 | 37,729 | 9,446 | 9,467 | 9,663 | 1.023 | 1.021 |
| a random graph, n = 2000, m = 8000 | 1,157,080 | 404,484 | 412,097 | 409,884 | 1.013 | 0.995 |
| the disconnected union, scrambled | 5,159 | 2,108 | 2,090 | 2,069 | 0.981 | 0.990 |
| the arrow of one hub, n = 300 | 45,150 | 599 | 599 | 599 | 1.000 | 1.000 |
| the binary tree of 511 vertices | 66,046 | 1,021 | 1,021 | 1,021 | 1.000 | 1.000 |
| the 64 × 64 grid | 262,207 | 63,266 | 70,587 | 66,394 | 1.049 | 0.941 |
| the 48 × 48 grid, scrambled | 309,592 | 34,606 | 35,829 | 34,728 | 1.004 | 0.969 |
| the arrow of three hubs, n = 400, scrambled (dense rows) | 79,409 | 1,594 | 1,594 | 1,594 | 1.000 | 1.000 |
| the band of half-width 4, n = 500, scrambled | 3,811 | 2,490 | 2,490 | 2,491 | 1.000 | 1.000 |

So 0.945 to 1.049 of the better of the two references (0.891 to 1.021 of the exact minimum degree); a graph without cycles (an arrow of one hub, a tree, a path) is ordered without any fill. `Ordering::Colamd.symmetric` (SYMAMD) gives the same fill as `amd` on twelve of these graphs and differs by less than one per cent on the other three; it is within 1.049 of SuperLU's on all; on larger graphs in three dimensions, of up to 60,000 vertices, a review measured 0.948 to 1.164 of `amd`'s fill, the variation between two minimum degree heuristics. On the cubes of the 27-point stencil, 6³ to 18³ in their natural order, `amd` fills in 1 to 10 per cent more than SuperLU's MMD, and 18 per cent more at 14³ (373,929 against 316,920), still less than the exact minimum degree there (406,392); on the cubes of the 7-point stencil, 10³ to 22³, it fills in 2 to 14 per cent less than MMD.

The final postorder of the elimination tree, in `amd`, SYMAMD and `colamd`, changes no number in these tables: every fill is the one of the order before it, as the theory of equivalent orderings says. That was checked on the fifteen graphs and six matrices here and on the 71 graphs and 35 matrices of the review, every fill the same to the element. On the 76 graphs of another probe of the review, a grid with three dense rows besides, and on its 3,000 small random patterns, the elimination tree of every new order of `amd` and SYMAMD, and the column elimination tree of every order of `colamd`, is postordered; before, the tree of `amd`'s order was not on 64 of the 76 graphs and 547 of the patterns, and that of `colamd`'s on 51 and 158.

COLAMD, nnz(L) of the Cholesky factor of `(A·Q)ᵀ·(A·Q)`. SuperLU orders square matrices only; the rectangular ones are compared with `amd` applied to the explicit pattern of `AᵀA` and with the exact minimum degree of it:

| Matrix | natural | SuperLU COLAMD | `amd` of `AᵀA` | exact | `colamd` |
| --- | --- | --- | --- | --- | --- |
| random 300 × 300, 1,488 elements with the diagonal | 37,088 | 20,426 | 20,425 | 20,329 | 20,426 |
| random 1000 × 1000, 4,989 elements with the diagonal | 404,885 | 209,665 | 208,486 | 207,607 | 209,665 |
| the upwind stencil of the 30 × 30 grid | 52,258 | 29,314 | 28,444 | 30,581 | 29,026 |
| random 600 × 300, 1,491 elements | 28,125 | — | 12,618 | 12,562 | 12,483 |
| random 200 × 500, 1,197 elements | 73,777 | — | 21,134 | 21,524 | 21,270 |
| random 800 × 400, 2,786 elements with the diagonal | 64,906 | — | 38,046 | 37,909 | 38,241 |

On the two random square matrices `colamd` gives SuperLU's COLAMD fill to the element; on all six it is within 0.989 to 1.020 of `amd` of the formed `AᵀA`, and 0.949 to 1.010 of the exact minimum degree.

Nested dissection on the `k × k` five-point grid, numbered by rows:

| k | nnz(L) | / (n·log₂ n) | `amd` | / (n·log₂ n) | dissection / `amd` |
| --- | --- | --- | --- | --- | --- |
| 16 | 2,291 | 1.119 | 2,131 | 1.041 | 1.075 |
| 32 | 13,262 | 1.295 | 11,873 | 1.159 | 1.117 |
| 64 | 68,379 | 1.391 | 66,394 | 1.351 | 1.030 |
| 128 | 341,720 | 1.490 | 362,226 | 1.579 | 0.943 |
| 256 | 1,637,535 | 1.562 | 1,936,338 | 1.847 | 0.846 |
| 512 | 7,626,016 | 1.616 | 10,005,770 | 2.120 | 0.762 |

The ratio to `n·log₂ n` grows by less and less (0.18, 0.10, 0.10, 0.07, 0.05 a doubling of `k`), as a fill of order `n·log n` does, where that of `amd` grows by more and more. The tests check, from the order alone, that the dissection holds real separators — the last positions of a piece that leave the rest in two runs no edge joins, each at least an eighth — on grids, the nine-point grid, a cube, a scrambled grid, two grids side by side and two random geometric graphs.

The elimination tree of the dissection is short. On the `k × k` grid it is 51, 113, 221, 441 and 926 columns high for `k` = 16, 32, 64, 128 and 256, about `3.5·k`: a path from a leaf to the root passes through one separator of every level, of about `k`, `k/2`, `k/2`, `k/4`, … vertices. That of `amd` is 57, 141, 275, 787 and 1,880 high, a growing multiple of `k`; a test checks the dissection's height up to `k` = 128 against `3·k + 128`, which is no bound (at `k` = 256 the height is 926), and against 0.7 of `amd`'s height at `k` = 128. On other graphs, the fill against `amd`:

| Graph | dissection / `amd` |
| --- | --- |
| the nine-point 256 × 256 grid | 0.643 |
| the cubes of 8³, 16³, 24³, 32³ | 0.969, 0.782, 0.690, 0.622 |
| a random geometric graph of 4,000 points in the cube, about 16 neighbours | 0.836 |
| a random geometric graph of 5,000 points in the square, about 12 neighbours | 1.169 |
| Delaunay triangulations of 2,000, 20,000 and 200,000 random points in the square | 1.217, 1.162, 1.149 |
| Delaunay triangulations of 5,000 and 30,000 random points in the cube | 0.814, 0.610 |
| the 16 × 256 strip | 1.352 |
| random trees of 1,000, 3,000 and 10,000 vertices | 1.130, 1.143, 1.137 |
| random graphs of 700 to 5,000 vertices and average degree 1.5 | 1.249 to 1.387 |
| the same, of average degree 2 | 1.527 to 1.727 |
| the same, of average degree 2.5 | 1.412 to 1.770 |
| random graphs of 700 to 100,000 vertices and average degree 3 to 8 | 1.000 (handed to `amd`) |

The Delaunay meshes (made by `scipy.spatial.Delaunay`) are not part of the tests. On them the two separators earn their keep: with the level sets alone the dissection of the 200,000-point mesh fills in 1.909 times what `amd` does, and with the multilevel one beside them 1.149; a dissection by straight cuts at the coordinate medians, AMD on its leaves, gives 1.50 there, so on irregular meshes of the plane of this size minimum degree is the better ordering.

**Time** (conditions as under [Storage and operations](#storage-and-operations)), the ordering alone; the elimination tree and the postorder of the matrix as given:

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `amd` | lap2d-316 scrambled, n = 99856, nnz = 498016 | 38.1 ms | SuiteSparse CAMD (no constraints) | 25.5 ms | 1.50 |
| `Ordering::Colamd.symmetric` (SYMAMD) | lap2d-316 scrambled, n = 99856, nnz = 498016 | 51.0 ms | SuiteSparse SYMAMD | 26.1 ms | 1.95 |
| `nested_dissection` | lap2d-316 scrambled, n = 99856, nnz = 498016 | 468 ms | METIS NodeND (via CHOLMOD) | 262 ms | 1.78 |
| `reverse_cuthill_mckee` | lap2d-316 scrambled, n = 99856, nnz = 498016 | 8.22 ms | SciPy csgraph.reverse_cuthill_mckee | 5.38 ms | 1.53 |
| `amd` | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 873 ms | SuiteSparse CAMD (no constraints) | 535 ms | 1.63 |
| `Ordering::Colamd.symmetric` (SYMAMD) | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 1.06 s | SuiteSparse SYMAMD | 550 ms | 1.93 |
| `nested_dissection` | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 7.39 s | METIS NodeND (via CHOLMOD) | 3.42 s | 2.16 |
| `reverse_cuthill_mckee` | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 142 ms | SciPy csgraph.reverse_cuthill_mckee | 88.5 ms | 1.60 |
| `elimination_tree` | lap2d-1000, n = 1000000, nnz = 4996000 | 11.8 ms | CHOLMOD etree | 4.48 ms | 2.64 |
| `postorder` | lap2d-1000, n = 1000000, nnz = 4996000 | 3.46 ms | CHOLMOD postorder | 2.39 ms | 1.45 |
| `amd` | lap3d-46, n = 97336, nnz = 668656 | 39.0 ms | SuiteSparse CAMD (no constraints) | 26.6 ms | 1.46 |
| `Ordering::Colamd.symmetric` (SYMAMD) | lap3d-46, n = 97336, nnz = 668656 | 48.5 ms | SuiteSparse SYMAMD | 29.5 ms | 1.65 |
| `nested_dissection` | lap3d-46, n = 97336, nnz = 668656 | 499 ms | METIS NodeND (via CHOLMOD) | 375 ms | 1.33 |
| `reverse_cuthill_mckee` | lap3d-46 scrambled, n = 97336, nnz = 668656 | 10.3 ms | SciPy csgraph.reverse_cuthill_mckee | 6.67 ms | 1.54 |
| `amd` | elast3d-16, n = 12288, nnz = 876024 | 6.46 ms | SuiteSparse CAMD (no constraints) | 3.51 ms | 1.84 |
| `Ordering::Colamd.symmetric` (SYMAMD) | elast3d-16, n = 12288, nnz = 876024 | 17.4 ms | SuiteSparse SYMAMD | 8.58 ms | 2.03 |
| `nested_dissection` | elast3d-16, n = 12288, nnz = 876024 | 87.0 ms | CHOLMOD NESDIS | 22.0 ms | 3.96 |
| `amd` | randspd-100000, n = 100000, nnz = 499978 | 112 ms | SuiteSparse AMD | 89.5 ms | 1.26 |
| `Ordering::Colamd.symmetric` (SYMAMD) | randspd-100000, n = 100000, nnz = 499978 | 140 ms | SuiteSparse SYMAMD | 86.3 ms | 1.62 |
| `amd` | circuit-100000, n = 100000, nnz = 417650 | 220 ms | SuiteSparse CAMD (no constraints) | 148 ms | 1.49 |
| `Ordering::Colamd.symmetric` (SYMAMD) | circuit-100000, n = 100000, nnz = 417650 | 257 ms | SuiteSparse SYMAMD | 165 ms | 1.56 |
| `reverse_cuthill_mckee` | circuit-100000 scrambled, n = 100000, nnz = 417650 | 9.77 ms | sprs RCM | 7.71 ms | 1.27 |
| `colamd` | convdiff2d-316, 99856 × 99856, nnz = 498016 | 21.5 ms | SuiteSparse COLAMD | 10.6 ms | 2.02 |
| `colamd` | circuit-10000, 10000 × 10000, nnz = 41824 | 12.9 ms | SuiteSparse COLAMD | 9.61 ms | 1.34 |
| `colamd` | randunsym-10000, 10000 × 10000, nnz = 89959 | 35.3 ms | SuiteSparse COLAMD | 31.7 ms | 1.11 |
| `colamd` | randtall-30000, 90000 × 30000, nnz = 269988 | 127 ms | SuiteSparse COLAMD | 95.5 ms | 1.33 |
| `colamd` | gridls-183, 100101 × 33489, nnz = 166713 | 5.91 ms | SuiteSparse COLAMD | 2.96 ms | 2.00 |
| `colamd` | convdiff3d-30, 27000 × 27000, nnz = 183600 | 17.7 ms | SuiteSparse COLAMD | 13.1 ms | 1.35 |
| `elimination_tree` | lap3d-100, n = 1000000, nnz = 6940000 | 25.2 ms | CHOLMOD etree | 17.0 ms | 1.49 |
| `postorder` | lap3d-100, n = 1000000, nnz = 6940000 | 3.55 ms | CHOLMOD postorder | 2.33 ms | 1.52 |
| `elimination_tree` | elast3d-30, n = 81000, nnz = 6133248 | 11.9 ms | CHOLMOD etree | 5.33 ms | 2.24 |
| `postorder` | elast3d-30, n = 81000, nnz = 6133248 | 237 µs | CHOLMOD postorder | 184 µs | 1.29 |
| `connected_components` | circuit-1000000, n = 1000000, nnz = 4181182 | 46.5 ms | SciPy csgraph.connected_components | 44.2 ms | 1.05 |
| `connected_components` | lap3d-100, n = 1000000, nnz = 6940000 | 38.2 ms | SciPy csgraph.connected_components | 36.9 ms | 1.03 |

The quality of the orders on the same matrices, the number of nonzeros of the factor (of `L` for the symmetric orderings, of the Cholesky factor of `AᵀA` for `colamd`), the band and the profile; the ratio is ours over the best reference's, below 1 when ours is better:

| Measure | Matrix | Ours | Best reference | Its value | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `amd fill nnz(L)` | lap2d-316 scrambled, n = 99856, nnz = 498016 | 3,395,921 | SuiteSparse AMD | 3,315,439 | 1.02 |
| `symamd fill nnz(L)` | lap2d-316 scrambled, n = 99856, nnz = 498016 | 3,399,728 | SuiteSparse SYMAMD | 3,373,620 | 1.01 |
| `nested_dissection fill nnz(L)` | lap2d-316 scrambled, n = 99856, nnz = 498016 | 2,630,850 | METIS NodeND (via CHOLMOD) | 2,707,600 | 0.972 |
| `reverse_cuthill_mckee band` | lap2d-316 scrambled, n = 99856, nnz = 498016 | 316 | sprs RCM | 316 | 1.00 |
| `reverse_cuthill_mckee profile` | lap2d-316 scrambled, n = 99856, nnz = 498016 | 21,185,746 | sprs RCM | 21,185,746 | 1.00 |
| `amd fill nnz(L)` | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 50,903,193 | SuiteSparse AMD | 50,669,720 | 1.00 |
| `symamd fill nnz(L)` | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 49,931,467 | SuiteSparse SYMAMD | 50,477,754 | 0.989 |
| `nested_dissection fill nnz(L)` | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 33,167,904 | METIS NodeND (via CHOLMOD) | 34,592,781 | 0.959 |
| `reverse_cuthill_mckee band` | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 1,000 | sprs RCM | 1,000 | 1.00 |
| `reverse_cuthill_mckee profile` | lap2d-1000 scrambled, n = 1000000, nnz = 4996000 | 668,165,500 | sprs RCM | 668,165,500 | 1.00 |
| `amd fill nnz(L)` | lap3d-46, n = 97336, nnz = 668656 | 42,058,035 | SuiteSparse AMD | 42,533,837 | 0.989 |
| `symamd fill nnz(L)` | lap3d-46, n = 97336, nnz = 668656 | 42,058,035 | SuiteSparse SYMAMD | 40,843,503 | 1.03 |
| `nested_dissection fill nnz(L)` | lap3d-46, n = 97336, nnz = 668656 | 21,790,767 | METIS NodeND (via CHOLMOD) | 26,352,685 | 0.827 |
| `reverse_cuthill_mckee band` | lap3d-46 scrambled, n = 97336, nnz = 668656 | 1,610 | sprs RCM | 1,610 | 1.00 |
| `reverse_cuthill_mckee profile` | lap3d-46 scrambled, n = 97336, nnz = 668656 | 114,355,885 | sprs RCM | 114,355,885 | 1.00 |
| `amd fill nnz(L)` | elast3d-16, n = 12288, nnz = 876024 | 6,163,377 | SuiteSparse AMD | 6,254,745 | 0.985 |
| `symamd fill nnz(L)` | elast3d-16, n = 12288, nnz = 876024 | 6,359,991 | SuiteSparse SYMAMD | 6,648,846 | 0.957 |
| `nested_dissection fill nnz(L)` | elast3d-16, n = 12288, nnz = 876024 | 4,733,352 | METIS NodeND (via CHOLMOD) | 4,522,011 | 1.05 |
| `amd fill nnz(L)` | randspd-100000, n = 100000, nnz = 499978 | 297,302,690 | SuiteSparse AMD | 297,103,622 | 1.00 |
| `symamd fill nnz(L)` | randspd-100000, n = 100000, nnz = 499978 | 297,021,810 | SuiteSparse SYMAMD | 297,712,976 | 0.998 |
| `amd fill nnz(L)` | circuit-100000, n = 100000, nnz = 417650 | 60,914,197 | SuiteSparse AMD | 60,984,831 | 0.999 |
| `symamd fill nnz(L)` | circuit-100000, n = 100000, nnz = 417650 | 60,980,305 | SuiteSparse SYMAMD | 92,812,729 | 0.657 |
| `reverse_cuthill_mckee band` | circuit-100000 scrambled, n = 100000, nnz = 417650 | 36,799 | SciPy csgraph.reverse_cuthill_mckee | 99,279 | 0.371 |
| `reverse_cuthill_mckee profile` | circuit-100000 scrambled, n = 100000, nnz = 417650 | 1,784,796,146 | SciPy csgraph.reverse_cuthill_mckee | 1,857,414,797 | 0.961 |
| `colamd fill nnz(chol(AᵀA))` | convdiff2d-316, 99856 × 99856, nnz = 498016 | 9,904,056 | SuiteSparse COLAMD | 9,821,400 | 1.01 |
| `colamd fill nnz(chol(AᵀA))` | circuit-10000, 10000 × 10000, nnz = 41824 | 8,456,189 | SuiteSparse COLAMD | 8,428,115 | 1.00 |
| `colamd fill nnz(chol(AᵀA))` | randunsym-10000, 10000 × 10000, nnz = 89959 | 35,177,580 | SuiteSparse COLAMD | 35,149,811 | 1.00 |
| `colamd fill nnz(chol(AᵀA))` | randtall-30000, 90000 × 30000, nnz = 269988 | 201,844,145 | SuiteSparse COLAMD | 201,704,628 | 1.00 |
| `colamd fill nnz(chol(AᵀA))` | gridls-183, 100101 × 33489, nnz = 166713 | 866,156 | SuiteSparse COLAMD | 846,570 | 1.02 |
| `colamd fill nnz(chol(AᵀA))` | convdiff3d-30, 27000 × 27000, nnz = 183600 | 24,191,941 | SuiteSparse COLAMD | 23,239,276 | 1.04 |

`symmetric_permutation` has no reference that does the same work: CSparse's `cs_permute` leaves the columns unsorted.

Where the references are faster:

- `amd` (1.26 to 1.84 times SuiteSparse's AMD or CAMD), SYMAMD (1.56 to 2.03) and `colamd` (1.11 to 2.02), with fills within 0.957 to 1.04 of theirs (SYMAMD's 0.657 of SuiteSparse's on the circuit): the quotient graph in arrays of `usize` and of structures, about twice the memory traffic of SuiteSparse's 32-bit arrays, and the final postorder of the elimination tree, which AMD does not make.
- `nested_dissection` (1.33 to 2.16 times METIS, 3.96 times CHOLMOD's NESDIS on the elasticity matrix), with 0.827 to 1.05 of METIS's fill: the refinement of its separators, move by move.
- `reverse_cuthill_mckee` (1.27 to 1.60 times SciPy's and sprs's), whose band and profile are theirs or smaller: the search for a pseudo-peripheral vertex to start each block from, where SciPy starts from a vertex of least degree.
- `elimination_tree` (1.49 to 2.64 times CHOLMOD's): the tree is computed from the pattern of `A + Aᵀ`, formed first, where CHOLMOD walks the stored triangle. `postorder` (1.29 to 1.52 times CHOLMOD's) was not examined.

The guards in the tests are of the complexity class: `reverse_cuthill_mckee` of an arrow of 100,000 vertices with its hub first, of a binary tree of 131,071, of a diagonal of 100,000 and of 50,000 blocks of two, and the elimination trees of the arrow and of the tree, must each take less than 20 seconds in a debug build, which code of the order of `n²` would not; `amd` and `nested_dissection` of a scrambled 300 × 300 grid less than 60.

### Limits

- The orderings are heuristics: none finds the least fill, which is NP-hard to find. Ties between vertices of equal degree are settled by their index and their place in the lists, so a renumbering of the same graph can give a slightly different fill.
- `amd` has no useful bound on its time in the worst case, as the paper says; on the graphs above it is linear in the stored elements. On a graph of many rows with just under `10·√n` neighbours, the threshold at which they would be set aside, it takes time of the order of `n^1.5` (on stars of that many leaves strung on a path), which is what minimum degree with that threshold costs. Aggressive absorption is always on, and the dense-row threshold `max(16, 10·√n)` is fixed: there are no options.
- In the orders of `amd` and `colamd` a row or column set aside as dense comes after every row or column it is joined to; the final postorder may put after it others that it does not touch, so it is not always last. `nested_dissection` does put its dense rows last.
- `nested_dissection` gives less fill than `amd` on regular grids from about 10⁴ unknowns and on meshes in three dimensions, more on irregular meshes of the plane of up to a few hundred thousand vertices (by 15 to 22 per cent on those above, and by 15 to 27 per cent on the random geometric graphs of the plane of a review) and on long thin strips (1.38 to 1.63 times `amd`'s fill on strips of 16, 8 and 4 rows, the review measured). On a path, a tridiagonal matrix, it fills in 1.37 to 1.50 times what `amd` does, from 300 to 10⁵ vertices. It also gives more on trees and on random graphs of average degree below three, by 13 to 20 per cent on random trees and by up to 77 per cent at average degree 2.5: such graphs have small separators but little fill to save, and cutting them puts back fill that minimum degree avoids. Its leaves are ordered by `amd` without regard to the separators around them.
- `nested_dissection` takes about 8 to 14 times the time of `amd` on the matrices of **Measured** (468 ms against 38.1 ms on the scrambled 316 × 316 grid). Its refinement keeps the gains up to date move by move: a vertex of high degree costs its degree when it enters or leaves the separator, not at every move next to it.
- The order of `nested_dissection` keeps every separator after the two parts it separates, and is not always a postorder of its elimination tree (on 28 of the 77 graphs above it was not): a postorder could put vertices of a part that hang below a later vertex of the separator after the earlier ones. `postorder` renumbers it, with the same fill, where a factorisation needs that.
- `Ordering::columns` with `Amd`, `ReverseCuthillMckee` or `NestedDissection` forms the pattern of `Aᴴ·A`, in time and memory of the order of the sum of the squares of the row lengths; `Colamd` is the column ordering that does not.
- `elimination_tree` compresses paths without balancing them, which bounds its time by the stored elements times a logarithm rather than by the inverse of Ackermann's function.

## Sparse Cholesky and L·D·Lᴴ

`SparseCholesky` factors a sparse Hermitian (for real elements, symmetric) positive definite matrix as `P·A·Pᵀ = L·Lᴴ`, and `SparseLdl` factors a Hermitian matrix that need not be definite as `P·A·Pᵀ = L·D·Lᴴ`, with `L` unit lower triangular and `D` real. `P` is a fill-reducing permutation that an `Ordering` chooses, `Ordering::Amd` by default. Both are supernodal, and both run their arithmetic in the dense kernels of [`a::math::matrix`](matrix.md): the packed `gemm` for the updates, `trsm` for the triangular solves and the blocked dense Cholesky for the blocks on the diagonal.

The matrix must be square, and **only its lower triangle is read**: the upper one is taken to be its conjugate transpose and may hold anything, NaNs included, as for the dense `Matrix::cholesky`. The ordering, too, is shown the lower triangle only. Of a diagonal element only the real part is read, so a stray imaginary part of rounding size changes nothing.

| Call | Gives |
| --- | --- |
| `SparseCholesky::new(&a)`, `SparseCholesky::with_ordering(&a, &ordering)` | the factorisation, or `None` when a pivot is not positive or not finite: the matrix is not positive definite |
| `SparseLdl::new(&a)`, `SparseLdl::with_ordering(&a, &ordering)` | the factorisation, or `None` when a pivot is zero or not finite |
| `solve(&b)` | `X` with `A·X = B`, for a `Matrix` `B` of any number of columns |
| `solve_vec(&b)` | `x` with `A·x = b` |
| `l()` | `L` of the permuted matrix, as a `SparseMatrix` (an element the arithmetic cancelled exactly is not stored) |
| `permutation()` | `P` as the ordering gave it, `p[new] = old`: row `i` of `P·A·Pᵀ` is row `permutation()[i]` of `A` |
| `det()` | the determinant; `log_det()` its logarithm, for `SparseCholesky`, which does not overflow |
| `d()`, `inertia()` | for `SparseLdl`: the pivots, and the numbers of positive, negative and zero ones |
| `nnz()`, `supernodes()` | the nonzeros of `L` as the pattern and the ordering decide them, and the number of dense blocks |

### Method

The *symbolic analysis* reads the pattern alone and costs about the nonzeros of `A` and the row indices it writes, never `n²`. The ordering gives `P`, and the lower triangle of `P·A·Pᵀ` is laid out by columns and by rows. The elimination tree comes from Liu's algorithm with path compression (J. W. H. Liu, "A compact row storage scheme for Cholesky factors using elimination trees", ACM TOMS 12, 1986). The number of nonzeros of every column of `L` comes from the algorithm of Gilbert, Ng and Peyton ("An efficient algorithm to compute row and column counts for sparse Cholesky factorization", SIAM J. Matrix Anal. Appl. 15, 1994), which never forms a column to count it. The columns are then renumbered in a postorder of the tree, in which the child with the longest column comes last, next to its parent. That reordering has the same fill and the same operations, and it makes each subtree a run of consecutive columns. Last come the supernodes, and the rows below each, found by climbing the tree of supernodes from the nonzeros of each row of `A`, every row index written once.

The *numeric factorisation* is left-looking and supernodal, as in Ng and Peyton ("Block sparse Cholesky algorithms on advanced uniprocessor computers", SIAM J. Sci. Comput. 14, 1993) and in CHOLMOD. A supernode is held as two dense column-major blocks: the square on the diagonal and the rows below it. For each supernode in turn:

1. its blocks receive the elements of the matrix;
2. every earlier supernode with rows among its columns (a descendant in the elimination tree) takes its part out, as the product of two of its blocks by `gemm`, subtracted where the rows of the two meet;
3. the block on the diagonal is factored by the dense Cholesky method, or for `L·D·Lᴴ` by a dense `L·D·Lᴴ` without square roots (column by column up to 32 columns, halved recursively above that, nearly all of its work in `gemm`);
4. the block below is solved against it by `trsm`, and for `L·D·Lᴴ` divided by the pivots.

The descendants are not searched for. A finished supernode waits in the list of the supernode that holds its next row and moves on when it has been used, as the columns of the classical sparse column Cholesky do. The triangular solves go supernode by supernode through the same blocks, with `trsm` on the diagonal block and `gemm` for the rows below it.

### Supernodes

A *fundamental* supernode (Liu, Ng and Peyton, "On finding supernodes for sparse matrix computations", SIAM J. Matrix Anal. Appl. 14, 1993) is a longest run of columns in which each is the only child of the next in the elimination tree and has one nonzero more than it. Its columns make one dense block exactly. The fundamental supernodes are then merged, from the leaves up, into their parents where that costs few explicit zeros. This is the relaxed amalgamation of Ashcraft and Grimes ("The influence of relaxed supernode partitions on the multifrontal method", ACM TOMS 15, 1989). A child is merged into its parent:

- always, when the merge adds no zero or the merged block has at most 4 columns;
- when the block has up to 16 columns and its zeros, those of earlier merges counted, are under 80 % of what it holds on and below its diagonal;
- up to 48 columns, under 10 %;
- for any number of columns, under 5 %.

Only the last child of a supernode lies next to it and can be merged, and the postorder has put the child with the longest column there. The thresholds are CHOLMOD's defaults (Chen, Davis, Hager and Rajamanickam, "Algorithm 887: CHOLMOD, supernodal sparse Cholesky factorization and update/downdate", ACM TOMS 35, 2008) and have not been tuned here. Under them the zeros stay below four fifths of what the blocks hold on and below their diagonals, so those parts hold fewer than five times the nonzeros of `L`, and the blocks in all, whose squares on the diagonal are stored whole, fewer than ten times. Over 9 000 factorisations of varied patterns the worst seen were 4.75 and 8.5 times, both for a tridiagonal matrix in its own order, which is held in supernodes of 16 columns that are mostly zeros. A full matrix is one supernode, and a diagonal one has a supernode for every column.

### When `SparseLdl` can be relied on

There is no numerical pivoting: the pivots are taken in the order the ordering fixed before any arithmetic, as CHOLMOD's and Eigen's `LDLᵀ` take them. The factorisation exists exactly when every leading block of `P·A·Pᵀ` is nonsingular. That is assured, under every permutation, for positive definite and negative definite matrices, where it is as stable as the Cholesky factorisation. It is assured too for quasi-definite ones, `[[H, Bᴴ], [B, −C]]` with `H` and `C` positive definite, the matrices of regularised saddle-point and interior-point problems (Vanderbei, "Symmetric quasidefinite matrices", SIAM J. Optim. 5, 1995). There the factors are accurate while the coupling `B` is not large against the two definite blocks (Gill, Saunders and Shinnerl, "On the stability of Cholesky factorization for symmetric quasidefinite systems", SIAM J. Matrix Anal. Appl. 17, 1996).

A general indefinite matrix is another matter. It can have a zero pivot although it is nonsingular, as `[[0, 1], [1, 0]]` does, and then gives `None`. Worse, a pivot can be tiny without being zero, and then the factors are huge and the solutions wrong in most of their digits, with nothing to tell. Such a matrix needs `2 × 2` pivots chosen during the factorisation (Bunch and Kaufman), which `SparseLdl` does not use. `SparseLu`, which pivots, is the method for it, or the residual of the solution can be checked. `inertia()` counts the signs of the pivots, which by Sylvester's law are those of the eigenvalues; a pivot that rounding carries across zero is counted with the wrong sign, which can happen only to one that is tiny beside the elements of `|L|·|D|·|Lᴴ|`, which exceed those of the matrix where the factors grow.

### Example

```rust
use a::math::sparse::{Ordering, SparseCholesky, SparseLdl, SparseMatrix};

// The 5-point Laplacian of a 30 × 30 grid: 900 unknowns, at most five nonzeros a row.
let k = 30;
let n = k * k;
let mut entries = Vec::new();
for i in 0..n {
    entries.push((i, i, 4.0f64));
    for j in [i + 1, i + k] {
        if j < n && (j != i + 1 || j % k != 0) {
            entries.push((i, j, -1.0));
            entries.push((j, i, -1.0));
        }
    }
}
let a = SparseMatrix::from_coo(n, n, &entries);
let c = SparseCholesky::new(&a).unwrap();

// A·x = b to rounding, and the logarithm of a determinant of about e¹⁰⁶⁵, which as a
// determinant is out of the range of `f64`.
let b = vec![1.0; n];
let x = c.solve_vec(&b);
let residual = a.mul_vec(&x).iter().zip(&b).map(|(y, b)| (y - b).abs()).fold(0.0, f64::max);
assert!(residual < 1e-11);
assert!((c.log_det() - 1065.000688354234).abs() < 1e-9);
assert!(c.det().is_infinite());

// In the order of the grid the factor fills the band: up to 31 nonzeros a column.
let natural = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
assert_eq!(natural.nnz(), 27_029);

// The grid tied by 100 constraints of three points each, [[A, Bᵀ], [B, −I]]: quasi-definite,
// so it has L·D·Lᴴ in any order, with 900 positive pivots and 100 negative ones.
let m = 100;
let mut saddle = entries.clone();
for row in 0..m {
    for point in [9 * row, 9 * row + 1, 9 * row + 4] {
        saddle.push((n + row, point, 0.5));
        saddle.push((point, n + row, 0.5));
    }
    saddle.push((n + row, n + row, -1.0));
}
let saddle = SparseMatrix::from_coo(n + m, n + m, &saddle);
let f = SparseLdl::new(&saddle).unwrap();
assert_eq!(f.inertia(), (n, m, 0));
let b = vec![1.0; n + m];
let x = f.solve_vec(&b);
let residual = saddle.mul_vec(&x).iter().zip(&b).map(|(y, b)| (y - b).abs()).fold(0.0, f64::max);
assert!(residual < 1e-11);

// Without pivoting, a zero pivot stops it, even where the matrix is nonsingular.
let swap = SparseMatrix::from_coo(2, 2, &[(1, 0, 1.0), (0, 1, 1.0)]);
assert!(SparseLdl::new(&swap).is_none());
```

### Measured

Accuracy is measured against references computed outside the crate, and against what defines each factorisation. The embedded cases of `lib/tests/math/sparse/data/gen_cholesky.py` were computed with mpmath in 50 digits from the matrices as the doubles the tests read, and checked against NumPy (`linalg.solve`, `cholesky`, `slogdet`, `eigvalsh`) and SciPy (`sparse.linalg.spsolve`). `python3 gen_cholesky.py` regenerates the two data files byte for byte, and `cargo test --test math -- sparse::cholesky:: sparse::ldl::` runs the comparisons. The largest relative errors observed, in `f64`, under the natural order, a random permutation and the default ordering:

| Case | Solution | `L` | Pivots | `log det` / `det` |
| --- | --- | --- | --- | --- |
| Cholesky, real, order 30 (a 6 × 5 grid and 12 random couplings, condition 7.3) | 3.3·10⁻¹⁶ | 2.0·10⁻¹⁶ | | 0 / 1.1·10⁻¹⁵ |
| Cholesky, complex Hermitian, order 18 (a ring and 14 random couplings, condition 9.3) | 2.7·10⁻¹⁶ | 1.7·10⁻¹⁶ | | 0 / 8.3·10⁻¹⁶ |
| `L·D·Lᴴ`, real saddle point, order 24 (16 + 8, condition 11.2) | 2.2·10⁻¹⁶ | 1.4·10⁻¹⁶ | 2.3·10⁻¹⁶ | 1.2·10⁻¹⁶ (`det`) |
| `L·D·Lᴴ`, complex saddle point, order 16 (10 + 6, condition 9.2) | 2.9·10⁻¹⁶ | 1.6·10⁻¹⁶ | 4.3·10⁻¹⁶ | 9.2·10⁻¹⁶ (`det`) |

The inertias of the two saddle points, (16, 8, 0) and (10, 6, 0), are those of the eigenvalues, and the numbers of nonzeros of `L` are those of the elimination game on the pattern. In single precision the solutions of the two Cholesky cases are 2.1·10⁻⁷ and 1.1·10⁻⁷ off.

The logarithm of the determinant of a grid Laplacian is a sum of logarithms of its eigenvalues, `2 − 2·cos(kπ/(m + 1))` summed over the dimensions, known in closed form. In `f64` `log_det` equals it to the last digit on grids of 30 × 30, 100 × 100 (10⁴ unknowns, `log det` = 11 717.108862069537) and 20 × 20 × 20, and is 7.8·10⁻¹⁶ off relatively on a line of 30 points. In `f32` the relative error is at most 5·10⁻⁷, and 4.4·10⁻⁸ at 10⁴ unknowns; a plain sum of the logarithms would be 2.1·10⁻⁵ off there.

The tests check what defines the factorisations on generated matrices of all four element types, complex Hermitian ones with nonreal elements off the diagonal among them, with right-hand sides of 0 to 19 columns. They run under every ordering: the natural order, reverse Cuthill–McKee, AMD, COLAMD, nested dissection and random permutations (AMD, COLAMD and nested dissection were stubs returning the natural order when the figures here were measured; in the crate the same tests pass under the real orderings). The checks are `‖P·A·Pᵀ − L·Lᴴ‖`, `‖P·A·Pᵀ − L·D·Lᴴ‖` against `|L|·|D|·|Lᴴ|`, the backward error of the solutions, agreement with the dense `Matrix::cholesky`, the count of nonzeros against the elimination game, and `None` for matrices that are indefinite, singular, semidefinite, have a zero pivot or hold a NaN or an infinity in the lower triangle. A wider sweep outside the test suite made 38 400 factorisations: 9 600 random Hermitian positive definite matrices of order 0 to 89, in eight families of pattern (random, banded, arrows, blocks, trees, grids, stars, full), each under four orderings (the natural order, reverse Cuthill–McKee and two random permutations), 2 400 matrices of each element type. The worst `‖P·A·Pᵀ − L·Lᴴ‖/‖A‖` was 0.53·ε·(n + 1), the worst backward error of a solution 0.32·ε·(n + 1), and the worst distance from the dense Cholesky factor 0.23·ε·(n + 1). The `L·D·Lᴴ` of each matrix and of its negative came within 0.22·ε·(n + 1), and the counts of nonzeros never differed from the elimination game. The same matrices were then shifted by an amount near their smallest eigenvalue, to be definite, indefinite or nearly singular, and factored again under four orderings, some 37 000 factorisations in all. `SparseCholesky` refused exactly the matrices the dense Cholesky refused. `SparseLdl` existed exactly where a dense `L·D·Lᴴ` without pivoting existed, and its inertia was that of the eigenvalues wherever none of them was within rounding of zero. Repeated with other seeds and with matrices of order up to about 360, some 56 000 more factorisations of definite matrices and 54 000 of shifted ones came out the same, the worst `‖P·A·Pᵀ − L·Lᴴ‖/‖A‖` again 0.50·ε·(n + 1).

Fill for large matrices, in the order of the grid and in a geometric nested dissection order given as a permutation (`Ordering::Given`):

| Matrix | Unknowns | Order | `nnz(L)` | Supernodes |
| --- | --- | --- | --- | --- |
| 2-D Laplacian, 316 × 316 | 99 856 | the grid's | 31 554 811 | 2 082 |
| the same | 99 856 | nested dissection | 3 770 326 | 7 145 |
| 3-D Laplacian, 46 × 46 × 46 | 97 336 | nested dissection | 38 920 774 | 5 464 |
| 3-D Laplacian, 22 × 22 × 22 | 10 648 | the grid's | 4 940 209 | 204 |
| saddle point on the 2-D Laplacian of 283 × 283, 20 022 constraints (`SparseLdl`) | 100 111 | nested dissection, each constraint after its points | 5 293 001 | 7 884 |
| tridiagonal | 1 000 000 | its own | 1 999 999 | 62 500 |

The 3-D grid of 46 points a side was not factored in the grid's order, whose factor would have about 2·10⁸ nonzeros. Every one of these solved with a backward error `‖A·x − b‖/(‖A‖·‖x‖ + ‖b‖)` (Frobenius and Euclidean norms) of at most 3.2·10⁻¹⁸. The test `the_cost_grows_with_the_order_as_the_nonzeros_do` checks that the work grows with the nonzeros of the factor and not with `n²`. One tridiagonal matrix of 100 000 rows, and one arrow, must take no more than four times as long as eight of 12 500 rows; a linear cost takes about as long, and a quadratic one eight times as long.

**Time** (conditions as under [Storage and operations](#storage-and-operations)): the factorisations with their analysis, by each library's own ordering and again with our AMD order given to all, which compares the numeric phases; the solves with one right-hand side and with 16, each library on its own factor:

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `new` (own ordering) | lap2d-31, n = 961, nnz = 4681 | 303 µs | CHOLMOD default (auto, AMD/METIS) | 193 µs | 1.57 |
| `with_ordering` (same given order) | lap2d-31, n = 961, nnz = 4681 | 117 µs | CHOLMOD simplicial LLᵀ (given) | 137 µs | 0.853 |
| `solve_vec` | lap2d-31, n = 961, nnz = 4681 | 20.9 µs | faer solve_in_place | 11.0 µs | 1.90 |
| `solve` (16 columns) | lap2d-31, n = 961, nnz = 4681 | 111 µs | faer solve_in_place | 82.6 µs | 1.34 |
| `new` (own ordering) | lap2d-100, n = 10000, nnz = 49600 | 3.43 ms | Accelerate SparseFactor Cholesky (default order) | 3.11 ms | 1.10 |
| `with_ordering` (same given order) | lap2d-100, n = 10000, nnz = 49600 | 1.94 ms | Accelerate SparseFactor Cholesky (given) | 2.57 ms | 0.755 |
| `solve_vec` | lap2d-100, n = 10000, nnz = 49600 | 268 µs | faer solve_in_place | 189 µs | 1.42 |
| `solve` (16 columns) | lap2d-100, n = 10000, nnz = 49600 | 1.60 ms | faer solve_in_place | 1.45 ms | 1.10 |
| `new` (own ordering) | lap2d-316, n = 99856, nnz = 498016 | 45.0 ms | Accelerate SparseFactor Cholesky (default order) | 35.3 ms | 1.27 |
| `with_ordering` (same given order) | lap2d-316, n = 99856, nnz = 498016 | 29.1 ms | Accelerate SparseFactor Cholesky (given) | 30.0 ms | 0.971 |
| `solve_vec` | lap2d-316, n = 99856, nnz = 498016 | 3.45 ms | Accelerate SparseSolve | 4.44 ms | 0.777 |
| `solve` (16 columns) | lap2d-316, n = 99856, nnz = 498016 | 19.4 ms | CHOLMOD supernodal solve2 (reused workspace) | 21.5 ms | 0.903 |
| `SparseLdl::new` | lap2d-316, n = 99856, nnz = 498016 | 46.6 ms | Accelerate SparseFactor LDLTUnpivoted | 38.4 ms | 1.21 |
| `SparseLdl::solve_vec` | lap2d-316, n = 99856, nnz = 498016 | 3.27 ms | CHOLMOD simplicial LDLᵀ solve2 | 2.27 ms | 1.44 |
| `new` (own ordering) | lap2d-1000, n = 1000000, nnz = 4996000 | 920 ms | Accelerate SparseFactor Cholesky (default order) | 443 ms | 2.07 |
| `with_ordering` (same given order) | lap2d-1000, n = 1000000, nnz = 4996000 | 721 ms | Accelerate SparseFactor Cholesky (given) | 419 ms | 1.72 |
| `solve_vec` | lap2d-1000, n = 1000000, nnz = 4996000 | 42.6 ms | Accelerate SparseSolve | 46.4 ms | 0.918 |
| `solve` (16 columns) | lap2d-1000, n = 1000000, nnz = 4996000 | 224 ms | CHOLMOD supernodal solve | 234 ms | 0.955 |
| `new` (own ordering) | lap3d-10, n = 1000, nnz = 6400 | 698 µs | CHOLMOD default (auto, AMD/METIS) | 630 µs | 1.11 |
| `with_ordering` (same given order) | lap3d-10, n = 1000, nnz = 6400 | 290 µs | Accelerate SparseFactor Cholesky (given) | 374 µs | 0.775 |
| `solve_vec` | lap3d-10, n = 1000, nnz = 6400 | 31.6 µs | faer solve_in_place | 26.7 µs | 1.18 |
| `solve` (16 columns) | lap3d-10, n = 1000, nnz = 6400 | 167 µs | faer solve_in_place | 197 µs | 0.846 |
| `new` (own ordering) | lap3d-22, n = 10648, nnz = 71632 | 23.2 ms | Accelerate SparseFactor Cholesky (default order) | 12.0 ms | 1.93 |
| `with_ordering` (same given order) | lap3d-22, n = 10648, nnz = 71632 | 18.9 ms | Accelerate SparseFactor Cholesky (given) | 9.46 ms | 2.00 |
| `solve_vec` | lap3d-22, n = 10648, nnz = 71632 | 883 µs | CHOLMOD supernodal solve2 (reused workspace) | 755 µs | 1.17 |
| `solve` (16 columns) | lap3d-22, n = 10648, nnz = 71632 | 3.99 ms | Accelerate SparseSolve | 3.72 ms | 1.07 |
| `SparseLdl::new` | lap3d-22, n = 10648, nnz = 71632 | 23.2 ms | Accelerate SparseFactor LDLTUnpivoted | 16.1 ms | 1.44 |
| `SparseLdl::solve_vec` | lap3d-22, n = 10648, nnz = 71632 | 788 µs | faer Ldlt solve_in_place | 705 µs | 1.12 |
| `new` (own ordering) | lap3d-46, n = 97336, nnz = 668656 | 2.46 s | Accelerate SparseFactor Cholesky (default order) | 573 ms | 4.29 |
| `with_ordering` (same given order) | lap3d-46, n = 97336, nnz = 668656 | 2.43 s | Accelerate SparseFactor Cholesky (given) | 562 ms | 4.32 |
| `solve_vec` | lap3d-46, n = 97336, nnz = 668656 | 24.8 ms | Accelerate SparseSolve | 19.8 ms | 1.25 |
| `solve` (16 columns) | lap3d-46, n = 97336, nnz = 668656 | 89.2 ms | Accelerate SparseSolve | 54.0 ms | 1.65 |
| `new` (own ordering) | elast3d-10, n = 3000, nnz = 197568 | 9.61 ms | Accelerate SparseFactor Cholesky (default order) | 5.29 ms | 1.82 |
| `with_ordering` (same given order) | elast3d-10, n = 3000, nnz = 197568 | 8.17 ms | Accelerate SparseFactor Cholesky (given) | 4.60 ms | 1.78 |
| `solve_vec` | elast3d-10, n = 3000, nnz = 197568 | 347 µs | Accelerate SparseSolve | 210 µs | 1.65 |
| `solve` (16 columns) | elast3d-10, n = 3000, nnz = 197568 | 1.47 ms | Accelerate SparseSolve | 1.15 ms | 1.28 |
| `SparseLdl::new` | elast3d-10, n = 3000, nnz = 197568 | 9.90 ms | Accelerate SparseFactor LDLTUnpivoted | 7.36 ms | 1.34 |
| `SparseLdl::solve_vec` | elast3d-10, n = 3000, nnz = 197568 | 338 µs | faer Ldlt solve_in_place | 243 µs | 1.39 |
| `new` (own ordering) | elast3d-16, n = 12288, nnz = 876024 | 175 ms | Accelerate SparseFactor Cholesky (default order) | 48.8 ms | 3.58 |
| `with_ordering` (same given order) | elast3d-16, n = 12288, nnz = 876024 | 170 ms | Accelerate SparseFactor Cholesky (given) | 45.8 ms | 3.71 |
| `solve_vec` | elast3d-16, n = 12288, nnz = 876024 | 3.49 ms | Accelerate SparseSolve | 2.28 ms | 1.53 |
| `solve` (16 columns) | elast3d-16, n = 12288, nnz = 876024 | 13.2 ms | Accelerate SparseSolve | 7.58 ms | 1.74 |
| `new` (own ordering) | randspd-10000, n = 10000, nnz = 49990 | 120 ms | Accelerate SparseFactor Cholesky (default order) | 36.6 ms | 3.28 |
| `with_ordering` (same given order) | randspd-10000, n = 10000, nnz = 49990 | 114 ms | Accelerate SparseFactor Cholesky (given) | 33.0 ms | 3.45 |
| `solve_vec` | randspd-10000, n = 10000, nnz = 49990 | 2.31 ms | CHOLMOD supernodal solve2 (reused workspace) | 1.65 ms | 1.40 |
| `solve` (16 columns) | randspd-10000, n = 10000, nnz = 49990 | 6.92 ms | Accelerate SparseSolve | 4.40 ms | 1.57 |
| `new` (own ordering) | tridiag-1000000, n = 1000000, nnz = 2999998 | 134 ms | faer Llt (AMD, auto) | 62.3 ms | 2.15 |
| `with_ordering` (same given order) | tridiag-1000000, n = 1000000, nnz = 2999998 | 70.1 ms | faer simplicial (given) | 35.8 ms | 1.96 |
| `solve_vec` | tridiag-1000000, n = 1000000, nnz = 2999998 | 15.1 ms | faer solve_in_place | 7.61 ms | 1.99 |
| `solve` (16 columns) | tridiag-1000000, n = 1000000, nnz = 2999998 | 46.0 ms | faer solve_in_place | 51.8 ms | 0.889 |
| `new` (own ordering) | arrow-100000, n = 100000, nnz = 299998 | 12.8 ms | faer Llt (AMD, auto) | 5.25 ms | 2.43 |
| `with_ordering` (same given order) | arrow-100000, n = 100000, nnz = 299998 | 9.20 ms | faer simplicial (given) | 2.86 ms | 3.21 |
| `solve_vec` | arrow-100000, n = 100000, nnz = 299998 | 1.55 ms | faer solve_in_place | 389 µs | 3.98 |
| `solve` (16 columns) | arrow-100000, n = 100000, nnz = 299998 | 24.6 ms | faer solve_in_place | 3.25 ms | 7.58 |
| `SparseLdl::new` | saddle-100, n = 12000, nnz = 59600 | 6.11 ms | SuiteSparse LDL (our AMD order given) | 4.71 ms | 1.30 |
| `SparseLdl::solve_vec` | saddle-100, n = 12000, nnz = 59600 | 308 µs | CHOLMOD simplicial LDLᵀ solve2 | 176 µs | 1.74 |
| `SparseLdl::new` | saddle-283, n = 96106, nnz = 479398 | 60.7 ms | Accelerate SparseFactor LDLTUnpivoted | 54.7 ms | 1.11 |
| `SparseLdl::solve_vec` | saddle-283, n = 96106, nnz = 479398 | 3.10 ms | CHOLMOD simplicial LDLᵀ solve2 | 2.28 ms | 1.36 |

Where the references are faster:

- Accelerate, wherever it is the fastest reference for a factorisation (`new` 1.10 to 4.29 times, `with_ordering` 1.72 to 4.32, `SparseLdl::new` 1.11 to 1.44) and for the solves with 16 right-hand sides on the 3-D grids, the elasticity meshes and the random matrix (1.07 to 1.74): its dense blocks run on the M4's matrix unit, which this crate cannot reach without assembly for SME. On the same matrices, against CHOLMOD and faer, which run on the cores, `with_ordering` takes 0.591 to 1.20 of the time of the faster of them — 1.03 to 1.20 on the 3-D grids from 22³ and the elasticity meshes, where the largest supernodes run on the crate's own products and triangular solves against OpenBLAS's —, `SparseLdl::new` 0.794 to 0.933 of faer's and the solves with 16 right-hand sides 0.826 to 0.988.
- `new` with its own ordering against CHOLMOD on the smallest matrices (1.57 and 1.11) and, beside Accelerate, on the 46³ grid and the elasticity mesh of 16³ nodes (1.93 and 1.75): the ordering, `amd`'s time above, and on the larger ones the less fill of CHOLMOD's METIS. `SparseLdl::new` on the smaller saddle point: 1.30 times SuiteSparse's LDL given our order.
- The thin patterns, the tridiagonal matrix and the arrow (`new` 2.15 and 2.43 times faer's, the given order 1.96 and 3.21, `solve_vec` 1.99 and 3.98, the arrow's 16 right-hand sides 7.58): supernodes of one or two columns and the bookkeeping of each, where faer and CHOLMOD switch to a simplicial factor for such patterns.
- `solve_vec` below 10⁵ unknowns (1.17 to 1.90 times the fastest of faer, CHOLMOD and Accelerate; 0.777 and 0.918 at 10⁵ and 10⁶ on the 2-D grids), `solve` with 16 right-hand sides on the two smaller 2-D grids (1.34 and 1.10 times faer's) and `SparseLdl::solve_vec` (1.12 to 1.74 times CHOLMOD's simplicial solve and faer's): the right-hand side is permuted into a fresh vector and back, and every supernode, however small, goes through the dense triangular solve and product.

### Limits

- One thread: the independent subtrees of the elimination tree are not factored in parallel.
- No update or downdate of a factor for a change of the matrix (CHOLMOD's `cholmod_updown`): a changed matrix is factored again. The symbolic analysis is not kept apart either, so a second matrix of the same pattern is analysed again.
- No `2 × 2` pivots and no pivoting after the ordering in `SparseLdl` (see above).
- For complex elements the factorisations are of Hermitian matrices, `L·Lᴴ` and `L·D·Lᴴ`. A complex *symmetric* matrix (`Aᵀ = A`, not `Aᴴ = A`) has no factorisation here.
- A singular positive semidefinite matrix gives `None` when rounding leaves a pivot at zero or below it, and a factor with a tiny pivot when rounding leaves it above zero, as the dense method does.
- The determinant is the product of the pivots and overflows or underflows for a large matrix; `SparseCholesky::log_det` is the one to use there (it is summed with compensation). `SparseLdl` has no logarithm of its determinant.

## Sparse LU with partial pivoting

`SparseLu<T>` factors a square `SparseMatrix<T>` as `P·A·Q = L·U`: `Q` a permutation of the columns chosen beforehand from the pattern, so that the factors fill in little; `P` the permutation of the rows that partial pivoting chooses while the elimination runs; `L` lower triangular with ones on its diagonal and `U` upper triangular, both held by columns. The API follows the dense [`Lu`](matrix.md), except that a matrix singular to working precision is reported as singular, not only one with a pivot of exactly zero.

| Function | What it gives |
| --- | --- |
| `SparseLu::new(&a)` | `P·A·Q = L·U` by the strategy that suits the pattern, by UMFPACK's rule with its earlier symmetry threshold (below): for a nearly symmetric pattern (at least half of the elements off the diagonal mirrored, nine tenths of the diagonal stored) the columns by `amd` of `A + Aᵀ`, the diagonal pivot down to 10⁻⁴ of the largest element of its column (multipliers up to 10 000) and otherwise an element of at least 0.1 of it, the elimination refactored by the unsymmetric strategy as soon as it fills twice what that order predicts or grows 16-fold, or when it ends singular unless the pattern is (see below); otherwise the columns by `Ordering::Colamd` and partial pivoting |
| `SparseLu::with_ordering(&a, &ordering)` | partial pivoting (the threshold 1) with any `Ordering` of the columns: `Natural`, `Colamd`, `Given(q)`, or a symmetric one (`Amd`, `ReverseCuthillMckee`, `NestedDissection`) applied to the pattern of `Aᴴ·A` |
| `SparseLu::with_options(&a, &options)` | the strategy of a `SparseLuOptions` (`new()`, `with_strategy`, `with_ordering`, `with_pivot_threshold`): `LuStrategy::Auto`, the default, which is what `new` does (`with_options(&a, &SparseLuOptions::default())` is `new(&a)`); `Symmetric` or `Unsymmetric`, the two it chooses between, without the watch; or `Explicit`, the options' own ordering and threshold, which setting either of them chooses. `Auto` holds only while `ordering` and `pivot_threshold` have their defaults (`Colamd` and 1): options whose fields are set to anything else, by a builder, directly or by struct update syntax, factor as `Explicit`. `None` for a threshold outside `(0, 1]` whatever the strategy, a panic for a `Given` ordering of the wrong shape whatever the threshold |
| `solve(&b)`, `solve_vec(&b)` | `A·X = B` for the columns of a `Matrix<T>`, or `A·x = b` for a slice; `None` when `is_singular()` or when the solution is not finite |
| `solve_transpose(&b)`, `solve_transpose_vec(&b)` | `Aᵀ·X = B` from the same factors |
| `solve_conjugate_transpose(&b)`, `solve_conjugate_transpose_vec(&b)` | `Aᴴ·X = B` from the same factors; `Aᵀ` again for a real matrix |
| `det()` | the product of the pivots with the signs of both permutations; 0 whenever `is_singular()`, even for a matrix refused for its growth whose pivots are exact (Limits); 1 for `0 × 0`; NaN when the matrix or its elimination was not finite |
| `l()`, `u()` | the factors as `SparseMatrix<T>`, the ones of `L` stored |
| `p()`, `q()` | the permutation matrices, built in `O(n)` |
| `permutation()`, `column_permutation()` | the rows and the columns of `P·A·Q`, `p[new] = old` |
| `is_singular()` | singular to working precision: a column without a pivot, a pattern that is singular whatever its elements, an elimination that was not finite, or an estimated condition number beyond the precision of the type (below; a decision near a bound, with the rare errors the Limits give) |
| `nnz()` | the elements of `L` below its diagonal and of `U`, as the elimination reaches them |
| `growth()` | `max|U| / max|A|`, the growth of the elements |
| `rcond()` | an estimate of `1/(‖A‖₁·‖A⁻¹‖₁)` of the matrix as it is, 0 when `is_singular()` |

### The method

The elimination is the left-looking one of Gilbert and Peierls, *Sparse partial pivoting in time proportional to arithmetic operations* (SIAM J. Sci. Stat. Comput. 9, 1988). Step `j` solves `L·x = A·q_j` with the columns of `L` found so far. The rows in which `x` can be nonzero are those reachable from the nonzeros of the column of `A` in the graph that joins the pivotal row of each column of `L` to the rows of that column; a depth-first search with its own stack (no recursion, so a chain of dependencies as long as the matrix costs memory and nothing else) finds them in an order in which every row follows those it depends on, and the numerical solve runs over that list alone. The part of `x` in rows already pivotal is column `j` of `U`, the rest divided by the pivot is column `j` of `L`. Time is proportional to the floating-point operations plus the order and the nonzeros of `A`, memory to the nonzeros of the factors. The searches are shortened by the symmetric pruning of Eisenstat and Liu, *Exploiting structural symmetry in a sparse partial pivoting code* (SIAM J. Sci. Comput. 14, 1993): once `U` has an element `(k, j)` and column `k` of `L` an element in the pivotal row of step `j`, the search need follow column `k` only as far as its rows that are pivotal by then, which are moved to its front; the rows reached, and the factors, are the same.

The elimination begins a column at a time, which is the faster while the factors are sparse. In a matrix of order 64 or more, once a step has made 96 elements of `L` and `U` in its column (the pivot not counted), the rest of the elimination is supernodal, as in Demmel, Eisenstat, Gilbert, Li and Liu, *A supernodal approach to sparse partial pivoting* (SIAM J. Matrix Anal. Appl. 20, 1999), the columns made before becoming supernodes of one column; a matrix whose factors never have such a column (a tridiagonal one, of any order) is eliminated a column at a time to the end. A *supernode* is a run of consecutive columns of `L` with the same rows below the diagonal but for the pivotal row of each next column; a column joins the supernode of the one before when it depends on it and has as many rows that are not pivotal. Its elements are one dense block, column after column, the pivotal rows of its own columns first. A column that depends on a column of a supernode depends on all its later ones, so its part in the pivotal rows of a supernode is one dense segment, and the search runs over supernodes, pruned as above (a pruned supernode keeps, of its rows pivotal by then, one per supernode they lie in). The columns are taken in panels of 16: each supernode before a panel updates every column of the panel that depends on it at once, a dense triangular solve of their segments (padded with zeros to one length) with `trsm` and the product of the block below with them by `gemm`, subtracted at its rows; then the columns of the panel are finished one after the other, with the supernodes of its own columns, their pivots chosen from the whole column once every update is in, exactly as above. The factors have the same elements in the same places and the same pivots but for rounding; the blocks group the sums otherwise, which can, rarely, turn the choice between two elements within rounding of each other. When the elimination ends the blocks are unpacked into the columns of `L` the solves read.

The pivot is chosen by threshold partial pivoting among the rows not yet pivotal. The size of an element is `|re| + |im|`, as in the dense `Lu`, LAPACK and SuperLU, rather than its modulus: a deliberate reading of "largest magnitude", which costs no square root and changes the multiplier bound by at most √2. The diagonal element (row `q[j]` of `A`) is kept when it is nonzero and at least `pivot_threshold` times the largest; otherwise the largest is taken, the lowest row of `A` among equals. The default threshold of 1 is partial pivoting with ties going to the diagonal, and its multipliers are at most 1 in absolute value (√2 for complex elements); a threshold `t` below 1 allows multipliers up to `1/t` (`√2/t`) and in exchange keeps the pivots where a symmetric ordering put them. The threshold is rounded to the real type of the elements, so one below its range (`10⁻⁶⁰` for `f32`) becomes 0 and keeps any diagonal element that is not zero.

`new` chooses its strategy by the rule UMFPACK uses (T. A. Davis, *Algorithm 832: UMFPACK V4.3, an unsymmetric-pattern multifrontal method*, ACM Trans. Math. Softw. 30, 2004): a pattern with at least half of its elements off the diagonal mirrored and nine tenths of its diagonal stored takes the symmetric strategy, the others the unsymmetric one, COLAMD and partial pivoting. UMFPACK 6.3 (SuiteSparse 7.14) asks three tenths for the symmetry (`STRATEGY_THRESH_SYM`; one half, the value here, in its earlier versions) and nine tenths for the diagonal (`STRATEGY_THRESH_NNZDIAG`). The symmetric strategy orders the columns by `amd` of `A + Aᵀ` and keeps the diagonal element as the pivot down to 10⁻⁴ of the largest element of its column, a tenth of UMFPACK's `SYM_PIVOT_TOLERANCE` (0.001); otherwise the pivot is an element of at least 0.1 of the largest (UMFPACK's `PIVOT_TOLERANCE`): the largest such in a row whose own column is eliminated already, if there is one, and the largest of all if not. A row whose column is eliminated is one whose diagonal an earlier step passed over, which no later column can take on its diagonal; the row of the column that took this column's diagonal is one, and a pivot there is the one UMFPACK makes when it takes the mirror of an off-diagonal pivot as the next diagonal, its likeness of a pivot of two rows and two columns. UMFPACK chooses among its acceptable rows by their sparsity, which a left-looking elimination does not know, and refines its solutions (two steps by default), which `SparseLu` does not. The order bounds the fill only while the pivots stay on the diagonal, and diagonal pivots of a ten-thousandth of their column make multipliers of ten thousand, so `new` watches the symmetric elimination. After every step it compares the elements made so far with what as many steps make with every pivot on the diagonal — the first columns and rows of the Cholesky factor of `A + Aᵀ` in that order, counted by the elimination tree and the row and column counts of Gilbert, Ng and Peyton (*An efficient algorithm to compute row and column counts for sparse Cholesky factorization*, SIAM J. Matrix Anal. Appl. 15, 1994) once a pivot has left the diagonal — and the largest element of `U` with the largest of `A`. Beyond twice the prediction and the elements of `A` more, or beyond 16 times the largest element of `A`, the elimination stops and the matrix is factored again from the start by COLAMD and partial pivoting, whose decomposition, element for element, is then the result; so it is when the symmetric factors come out singular (a pivot of zero, an overflow, or a condition beyond the bound of the test below, which the growth raises), unless the pattern made them so. Everything the tests read is a function of the matrix, so a matrix always takes the same path. `LuStrategy::Symmetric` is the symmetric elimination without the watch.

### Singular matrices

A column with no candidate row, or only zeros, has no pivot: the elimination goes on with a pivot of zero, so that `P·A·Q = L·U` still holds. A matrix can be singular by its pattern alone, with no `n` nonzeros in distinct rows and columns, and rounding can leave such a matrix a pivot of the size of the rounding error instead of zero; so the pattern is checked as well. The pivots that are elements of `A` form a matching of rows to columns, which is as a rule complete already; the columns left over are matched by the augmenting paths of Hopcroft and Karp, *An n^5/2 algorithm for maximum matchings in bipartite graphs* (SIAM J. Comput. 2, 1973), at `O(√n·(n + nnz))` at worst, and a matrix with no complete matching is reported singular.

Most singular matrices show no pivot of zero, though: rounding leaves the pivot that should vanish of the size of the rounding error, or makes it exactly zero, by the accident of the order of the columns. The integers 1 to 9 in rows of three end with a pivot of about `10⁻¹⁶` in four of the six orders of their columns and with zero in the other two, and the dense `Lu` calls them regular and solves them into numbers of the order of `10¹⁵`. So `is_singular()` also reports a matrix that is *singular to working precision* by an estimate of its condition, which is how LAPACK's `xGESVX` and SuperLU's `xGSSVX` define it: the reciprocal condition number below the machine epsilon `ε`, where the bound on the error of a solution is as large as the solution. Three refinements make the test hold for the factors as they are computed.

- The condition is that of the matrix with its rows and then its columns scaled to a largest absolute value of 1, as LAPACK's `xGEEQU` scales them, so that rows of very different sizes, and diagonal matrices such as `diag(10⁻²⁰, 1)`, whose `rcond()` is `10⁻²⁰`, are not taken for a matrix near a singular one. One pass of that scaling undoes any scaling of the rows; a scaling of the columns it undoes only in part, and columns whose sizes differ by more than about `1/ε` can make a well-conditioned matrix reported (Limits).
- The bound `ε` presumes factors within about `ε` of `A`; by Wilkinson's analysis of Gaussian elimination they are within about `ρ·ε`, `ρ` the growth, so the bound is `τ = ε·max(1, ρ)`. Partial pivoting keeps `ρ` within a few units on all but contrived matrices; a pivot threshold below 1 can let it reach `1/ε` (a threshold of `10⁻⁸` in single precision often does), and then the factors can tell nothing from a singular matrix and a well-conditioned matrix is reported too, as its solutions by those factors would be noise.
- The elimination pivots on the matrix as it is and is backward stable in its norm, not always in the norm of the scaled matrix: a small row eliminated against large ones can leave an exactly singular matrix several `τ` from a singular one in the scaled norm while the unscaled norm still shows it within `τ`. So the matrix is reported when the smaller of its two reciprocal condition numbers, scaled and unscaled, is below `τ`, unless the scaled one is 16 times `τ` or more, which says that the matrix is well conditioned and merely badly scaled.

The condition is estimated once, when the matrix is factored, by Hager's method as Higham refined it for LAPACK (*FORTRAN codes for estimating the one-norm of a real or complex matrix*, ACM Trans. Math. Softw. 14, 1988), with two more steps of its search, neither of them LAPACK's. The search starts from a vector of equal elements, which has no part along the left null vector of a matrix with two equal rows, and two rows that the scaling makes equal are enough to make it stop hundreds of times short of the norm; one more step from Higham's vector of alternating signs recovers it. The left null vector `(…, 1, 1, −1, −1, …)` of a matrix whose rows satisfy `r_i + r_{i+1} = r_{i+2} + r_{i+3}` is orthogonal to both of those vectors, wherever it begins, and with elements of `±1` the search is then left to the ties that rounding breaks: review round 2 found such exactly singular matrices estimated up to `10¹⁴` times short and called regular at the default threshold. So the search also starts from a vector of pseudo-random signs and sizes in `[1, 2)` (SplitMix64 with a fixed seed, so that a matrix gets the same verdict each time it is factored), which has a part along every such vector (checked to order `10⁶`), and takes one step from it. An estimate costs at most seventeen solves with the factors, and the unscaled one is made only when the scaled one leaves the answer open. The estimate is made from the computed factors, so it depends on the order of the columns and on the rounding, as the pivots do, though far less: the matrix of the review was called regular in 8 of 2 000 random orders of its columns before the third start vector, in none after it. `rcond()` is the same estimate for the matrix as it is, on vectors scaled by `‖A‖₁` so that the solves overflow only where the condition number does (the `1 × 1` matrix of the smallest positive `f64` has 1); the estimate of the norm is a lower bound, so the reciprocal is never below the true one but for rounding.

The solves are triangular solves by columns, each right-hand side in time proportional to the nonzeros of the factors; the transposed and conjugate-transposed systems use the same factors.

### Example

```rust
use a::math::matrix::{Matrix, Norm};
use a::math::sparse::{Ordering, SparseLu, SparseLuOptions, SparseMatrix};

// A zero on the diagonal: partial pivoting must take its rows from elsewhere.
let a = SparseMatrix::from_coo(3, 3, &[
    (0, 1, 2.0), (0, 2, 1.0),
    (1, 0, 4.0), (1, 1, 1.0),
    (2, 0, 2.0), (2, 2, 3.0),
]);
let lu = SparseLu::new(&a);
assert!(!lu.is_singular());
let x = lu.solve_vec(&[3.0, 5.0, 5.0]).unwrap();
assert!(x.iter().all(|&v| (v - 1.0f64).abs() < 1e-14));
// Aᵀ·y = b from the same factors, and the determinant with the signs of P and Q.
let y = lu.solve_transpose_vec(&[6.0, 3.0, 4.0]).unwrap();
assert!(y.iter().all(|&v| (v - 1.0f64).abs() < 1e-14));
assert!((lu.det() + 26.0f64).abs() < 1e-12);
// P·A·Q = L·U, to rounding.
let paq = (&(&lu.p() * &a) * &lu.q()).to_matrix();
let product = &lu.l().to_matrix() * &lu.u().to_matrix();
assert!((&paq - &product).norm(Norm::Max) < 1e-15);

// Several right-hand sides at once, as the columns of a matrix.
let b = Matrix::from_rows(&[[3.0, 2.0], [5.0, 4.0], [5.0, 2.0]]);
let xs = lu.solve(&b).unwrap();
assert!((&a.mul_matrix(&xs) - &b).norm(Norm::Max) < 1e-14);

// A singular matrix is reported, and never solved: by a pivot of zero...
let s = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 4.0)]);
let singular = SparseLu::new(&s);
assert!(singular.is_singular() && singular.det() == 0.0);
assert!(singular.solve_vec(&[1.0, 1.0]).is_none());
// ...or, as most are, by a condition beyond the precision of the type: in their own order
// the integers 1 to 9 end with a pivot of about 1e-16 rather than zero.
let nine = SparseMatrix::from_fn(3, 3, |i, j| (3 * i + j + 1) as f64);
let lu9 = SparseLu::with_ordering(&nine, &Ordering::Natural);
assert!(lu9.u().diagonal().iter().all(|&p| p != 0.0));
assert!(lu9.is_singular() && lu9.det() == 0.0);
assert!(lu9.solve_vec(&[1.0, 2.0, 3.0]).is_none());
// Rows of very different sizes do not make a matrix singular.
let d = SparseMatrix::from_diagonal(&[1e-20, 1.0]);
assert_eq!(SparseLu::new(&d).solve_vec(&[1e-20, 1.0]), Some(vec![1.0, 1.0]));

// A pivot threshold below 1 keeps a diagonal element that is large enough.
let c = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 3.0), (1, 1, 4.0)]);
let partial = SparseLu::with_ordering(&c, &Ordering::Natural);
assert_eq!(partial.permutation(), [1, 0]);
let options = SparseLuOptions::new()
    .with_ordering(Ordering::Natural)
    .with_pivot_threshold(0.5);
let kept = SparseLu::with_options(&c, &options).unwrap();
assert_eq!(kept.permutation(), [0, 1]);
assert!(kept.rcond() > 0.1 && kept.growth() <= 2.0);
```

### Measured

The tests are in `lib/tests/math/sparse/lu.rs` (64 tests, the generic ones for all four element types), the reference data in `lib/tests/math/sparse/data/lu_reference.rs`, written by `data/gen_lu.py` with NumPy 2.3 and SciPy 1.18, and the 35 singular matrices of review round 2 in `data/lu_singular_cases.rs`, written by `data/gen_lu_singular_cases.py` with SymPy 1.14, which checks their exact rank `n − 1` and that their left null vectors are orthogonal to both fixed start vectors.

- `P·A·Q − L·U`: below 4 units of `n·ε·max|L|·max|U|` in every test, on random matrices singular ones included, for every ordering and pivot threshold. The factors and the elements the elimination reaches are those of a dense left-looking elimination that follows the same pivots, and every pivot is one the rule allows in that arithmetic.
- Against SciPy and NumPy, seven systems (a random unsymmetric matrix with ten zeros on its diagonal, a convection–diffusion and a Laplacian grid, a diagonal of `10⁻¹⁰` under elements of order one, a shuffled block triangular matrix, a random complex matrix and a complex Helmholtz grid, of orders 24 to 36, condition numbers 22 to 2 500), with the natural, the COLAMD and the reverse Cuthill–McKee orderings: the solutions of `A·x = b`, `Aᵀ·x = b` and `Aᴴ·x = b` are within 0.14 `cond·ε` of `scipy.sparse.linalg.spsolve` and `splu` in double precision and 0.083 in single, the determinant within 0.038 `n·cond·ε` of `numpy.linalg.det`, and `rcond()` equals `1/numpy.linalg.cond(A, 1)` to nine digits in double precision (within `10⁻⁶` in single). With the columns in their natural order the rows are pivoted in exactly SuperLU's order (`splu` with `permc_spec="NATURAL"`, `diag_pivot_thresh=1`, no equilibration) on all seven, and `nnz()` equals the fill of a symbolic elimination with those pivots.
- Twelve thin random patterns whose structural rank `scipy.sparse.csgraph.structural_rank` gives, six of them structurally singular: `is_singular()` agrees on every one, for every ordering and threshold. A 3 × 3 matrix whose pattern makes it singular, and whose elimination ends with a pivot of about `10⁻¹⁸` rather than zero, is reported singular.
- Singular to working precision, against an oracle that inverts each matrix exactly in 320-bit arithmetic (`Matrix<BigFloat>` and `Matrix<BigComplex>`) and takes the true reciprocal condition of the matrix scaled as the test scales it: 60 000 random matrices of orders 3 to 30 over the four element types, ten families (two equal rows; a row an exact multiple of another by 2, −1, ½, −4, ¼ or by 3, 7, 5, −3; a row the sum of two others; graph Laplacians; rows nearly dependent; rows multiplied by factors that round; diagonally dominant matrices with rows and columns scaled over moderate and over wide ranges; random matrices scaled over wide ranges), orderings natural, reverse Cuthill–McKee, COLAMD and random, thresholds 1 to `10⁻⁸`. On the final code all 30 000 exactly singular ones were reported, and none was accepted whose true scaled reciprocal condition is below `ε`. Of the 687 reported although their true scaled reciprocal condition is above `ε`, the solution that their own factors would have given (solved in double precision from `l()` and `u()`, against the 320-bit solution) had a relative error above 0.1 in 461 cases, between `10⁻³` and 0.1 in 196, and below `10⁻⁶` in 7, 5 of these with nearly dependent rows just above the bound. The diagonally dominant matrices with rows and columns scaled by factors up to `10^±6` (`10^±3` in single precision) were never reported in double precision, and 66 of 3 000 times in single precision, every one of whose solutions would have been wrong by more than `10⁻³`.
- Review round 1's campaign of matrices with a row or a column an exact multiple (1, −1, 3, 7, 5, −3, ½) of another, orders up to 40, every ordering and thresholds 1 to `10⁻⁸`, rerun on the final code: 1 599 999 of 1 600 000 reported (four seeds), the one missed at a threshold of `10⁻⁸`, and none in this family at the default threshold (the families of review round 2 above showed that this was not true of every family before the third start vector); before the test of the condition 69 % of them were called regular and solved. The integers 1 to 9 are reported in all six orders of their columns, and 8 000 matrices with two equal rows (2 000 a type) all are.
- In single precision with thresholds of `10⁻⁴` and `10⁻⁸`, 1 229 of the matrices of review round 1's full campaign (1 160 000 factorisations, rerun on the final code) with condition numbers below `10⁴` were reported singular (1 208 at `10⁻⁸`), at a median growth of `1.25·10⁷`: the solutions their factors would have given had errors above 0.1 in 1 111 and below `10⁻³` in 16; for the same reason 5 integer matrices with a determinant other than zero were reported at `10⁻⁸`. None was reported at the default threshold or in double precision.
- Review round 2's campaign of exactly singular matrices of `−1`, 0 and 1, of orders 9 to 29, with four consecutive rows related by `r_i + r_{i+1} = r_{i+2} + r_{i+3}` and in half of them two equal columns as well, 20 000 for each element type and threshold (1, 0.1, `10⁻⁴`) and two seeds, natural, reverse Cuthill–McKee and random column orders: 0 of 2 880 000 called regular, where the estimate without the pseudo-random start vector had missed 77 of the same matrices, 31 of them at the default threshold, with solutions of the order of `1/ε`. The 35 cases it had missed (`data/lu_singular_cases.rs`) in 204 column orders, three thresholds and the four element types: 0 of 85 680, against 82 to 98 for each type before. The review's twelve families, 3 456 000 factorisations: 7 called regular, all at a threshold of `10⁻⁴` (the boundary effect under Limits).
- The eigenvalue methods of the module, which factor `A − σI` for shift-and-invert and refuse a shift that makes it singular, pass all their tests with this decomposition; a shift within rounding of an eigenvalue is now refused, as their documentation says, rather than only one that meets a pivot of exactly zero.
- An earlier randomised campaign in release builds, 920 000 matrices over the four element types, of orders up to 200, all the orderings and thresholds from 1 to `10⁻⁶`, against a dense elimination, the dense `Lu`, the exact condition number and a structural-rank oracle: worst factor error 0.70 units of `n·ε·max|L|·max|U|`, worst backward error of the three solves 0.83 units of `n·ε` times the growth and the largest multiplier, no pivot outside the rule, the condition estimate between 0.99995 and 21.6 times the exact one, and every one of the 264 678 structurally singular matrices reported, 3 611 of them by the pattern check alone. It ran before the test of the condition and the two extra steps of its estimate, which only report more matrices and bring the estimate closer.
- On the 2-D grids below, with the column order SuperLU computes with COLAMD given as `Ordering::Given`, the row permutation is SuperLU's element by element at all three sizes and on all three problems, up to 67 251 rows taken off the diagonal at 99 856 unknowns, and the fill equals SuperLU's on the Laplacian and is within 69 elements of it (exact zeros that SciPy's count leaves out) on the others.
- The strategy of `new`, in `f64`, against master's `new` (COLAMD and partial pivoting) and, for the fill and the errors, UMFPACK and KLU with their defaults (SuiteSparse 7.14, one thread). On the five-point grid with `δ` on the diagonal and −1 to the neighbours, the version before (a threshold of 0.1, the largest element otherwise) filled 3.9, 5.7 and 8.7 times what master does at 50 × 50, 100 × 100 and 200 × 200 (δ = 0.09). Now, for δ from 0.01 to 0.5, the symmetric elimination keeps all but at most two of its pivots on the diagonal and AMD's fill (2.10 million elements at 200 × 200; UMFPACK 2.16 million, KLU 2.16 to 2.17 million), but its elements grow 400 to 7 700 times, and the watch refactors every one of these by partial pivoting: 3.79 to 4.17 million elements at 200 × 200, within 1 % of master's at every size, growth 2.4 to 11; at 200 × 200 forward errors at most 8.8·10⁻¹³ (master 5.1·10⁻¹³, UMFPACK, which refines, 6.9·10⁻¹⁴, KLU 1.0·10⁻¹¹ to 7.0·10⁻¹¹) and backward errors at most 230 `ε·‖A‖∞·‖x‖∞` (master 206), at 100 × 100 forward errors at most 2.5·10⁻¹² (master 2.1·10⁻¹²). With convection (`−1 ± c` and `−1 ± c/2`, c = 3 or 1.5) the pivots leave the diagonal. At δ = 4 the symmetric strategy runs to its end, with 0.61 of master's fill. `new` takes 0.33 to 0.78 of master's time on these grids without convection, 0.55 to 0.98 with it and 0.17 to 0.45 at δ = 4 (21 alternated samples, twice); all 27 decompositions are bit for bit those of the diagonal kept down to 0.001 with the watch at 64, in 0.87 to 1.01 of their time. Shift-and-invert, `lanczos_shift_invert` and `arnoldi_shift_invert` of the 100 × 100 and 150 × 150 Laplacians for four eigenvalues, factors `A − σI` by `new`: at σ = 1, 3.95 and 4.03 `A − σI` is indefinite, its diagonal weak at the last two, and it is refactored; the calls take 0.53 to 0.65 of master's time (`arnoldi_shift_invert` at σ = 4.03 on the smaller grid 0.80 to 0.82), with the decompositions of the values before, bit for bit. On the matrices of the scoreboard (the Laplacian, convection–diffusion and upwind grids of 316 × 316, circuits of 3 000 and 10 000 nodes, the 22³ convection–diffusion cube), and on 3-D Laplacian and elasticity meshes, every pivot stays on the diagonal and the growth is at most 2.3: the watch never stops them, the factors are those of the version before, and bit for bit those of the diagonal kept down to 0.001 with the watch at 64, the fill 0.14 to 0.57 of master's on the scoreboard matrices, 0.38 to 0.50 on 3-D Laplacians of 10³ to 30³ points and 0.68 to 0.85 on elasticity meshes of 648 to 12 288 unknowns. On the 450 random systems of nine families of the review, refused: 64, as master (UMFPACK 51, KLU 51, the version before 65); 135 of the 252 nearly symmetric ones were refactored; the worst backward error per family of the nearly symmetric ones is 0.8 to 23 `ε·‖A‖∞·‖x‖∞` (master 0.8 to 14.5, KLU, which keeps the diagonal down to 0.001 and does not refine, 6 to 65 000, UMFPACK 0.4 to 1.0), the worst forward error per family at most master's, and the fill together 0.98 of master's (UMFPACK 0.64). With the diagonal kept down to 0.001 and the watch at 64 times the largest element, the values before, 104 were refactored, with worst backward errors of 34 to 197.

Growth with the order, in a release build, `f64`, on the five-point operator of a `k × k` grid with 4 on the diagonal and `−1 ∓ cx`, `−1 ∓ cy` to the neighbours: the Laplacian (`cx = cy = 0`), and two problems whose rows must be interchanged, "pivot" (`cx = 1.5`, `cy = 3.5`) and "wind" (`cx = 3`, `cy = 6`). "Dissection" is the geometric nested dissection of the grid the tests use, "SuperLU COLAMD" SciPy's column order. Each entry is `nnz(L+U)`. The times of these factorisations were taken on a machine shared with other work and are not given; the times of the final measurements, on the matrices of the scoreboard, are under **Time** below. The test of singularity that every factorisation makes, an estimate of the condition, is at most seventeen solves (thirty-four near the bound).

| Problem | Column order | Threshold | `n = 6 241` | `n = 24 964` | `n = 99 856` |
| --- | --- | --- | --- | --- | --- |
| Laplacian | natural (band) | 1 | 979 993 | 7 863 974 | 63 009 766 |
| Laplacian | dissection | 1 | 299 727 | 1 539 292 | 7 579 424 |
| Laplacian | SuperLU COLAMD | 1 | 360 583 | 1 984 434 | 10 269 784 |
| pivot | natural (band) | 1 | 968 292 | 7 815 617 | 62 813 205 |
| pivot | dissection | 1 | 708 710 | 5 186 712 | 31 097 572 |
| pivot | dissection | 0.1 | 299 727 | 1 539 292 | 7 579 424 |
| pivot | SuperLU COLAMD | 1 | 389 760 | 2 145 667 | 11 339 713 |
| wind | dissection | 1 | 1 206 011 | 9 751 847 | 70 644 089 |
| wind | dissection | 0.1 | 299 727 | 1 539 292 | 7 579 424 |
| wind | SuperLU COLAMD | 1 | 382 934 | 2 113 423 | 11 184 206 |

For each fourfold step in the order, the band grows eightfold in elements and sixteenfold in arithmetic (`n^1.5` and `n²`, as the elimination of a band must); the dissection, when no rows are exchanged, about fivefold in elements and ninefold in arithmetic (close to `n log n` and `n^1.5`). With a threshold of 1 the convection-dominated problems take 63 to 88 % of their pivots off the diagonal of the dissection order and lose most of its advantage; with 0.1 every pivot stays on the diagonal, the fill is the Laplacian's, the growth at most 2.5 and the backward error below `1.2·10⁻¹⁵`. The worst backward error in the table is `6.6·10⁻¹⁵` ("wind", dissection, threshold 1, growth 5.9), none of these matrices is reported singular, and the condition estimates at `n = 6 241` agree with NumPy's exact `1/cond₁` (2.651·10⁻⁴, 3.389·10⁻³ and 2.648·10⁻³) to four digits.

**Time** (conditions as under [Storage and operations](#storage-and-operations)): `new` by the default options (the symmetric or the unsymmetric strategy, chosen from the pattern, and the estimate of the condition); `with_options` with AMD of `A + Aᵀ` and a pivot threshold of 0.1, which is UMFPACK's symmetric strategy, and with our COLAMD order given to all and partial pivoting, which compares the numeric phases; the solves on each library's own factors:

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `new` | convdiff2d-31, n = 961, nnz = 4681 | 706 µs | KLU (defaults: BTF, AMD) | 350 µs | 2.02 |
| `with_options` (same given column order) | convdiff2d-31, n = 961, nnz = 4681 | 729 µs | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 616 µs | 1.18 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | convdiff2d-31, n = 961, nnz = 4681 | 503 µs | KLU (defaults: BTF, AMD) | 334 µs | 1.51 |
| `solve_vec` | convdiff2d-31, n = 961, nnz = 4681 | 12.7 µs | KLU | 11.2 µs | 1.13 |
| `solve_transpose_vec` | convdiff2d-31, n = 961, nnz = 4681 | 17.5 µs | KLU | 14.7 µs | 1.20 |
| `new` | convdiff2d-100, n = 10000, nnz = 49600 | 9.73 ms | UMFPACK (defaults) | 8.33 ms | 1.17 |
| `with_options` (same given column order) | convdiff2d-100, n = 10000, nnz = 49600 | 13.5 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 9.09 ms | 1.49 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | convdiff2d-100, n = 10000, nnz = 49600 | 8.15 ms | UMFPACK (defaults) | 8.58 ms | 0.950 |
| `solve_vec` | convdiff2d-100, n = 10000, nnz = 49600 | 224 µs | KLU | 216 µs | 1.04 |
| `solve_transpose_vec` | convdiff2d-100, n = 10000, nnz = 49600 | 316 µs | faer | 279 µs | 1.13 |
| `new` | convdiff2d-316, n = 99856, nnz = 498016 | 151 ms | UMFPACK (defaults) | 139 ms | 1.09 |
| `with_options` (same given column order) | convdiff2d-316, n = 99856, nnz = 498016 | 282 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 158 ms | 1.79 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | convdiff2d-316, n = 99856, nnz = 498016 | 137 ms | UMFPACK (defaults) | 145 ms | 0.941 |
| `solve_vec` | convdiff2d-316, n = 99856, nnz = 498016 | 3.60 ms | faer | 5.36 ms | 0.672 |
| `solve_transpose_vec` | convdiff2d-316, n = 99856, nnz = 498016 | 5.20 ms | faer | 4.88 ms | 1.06 |
| `new` | upwind2d-316, n = 99856, nnz = 498016 | 154 ms | UMFPACK (unsymmetric, COLAMD, pivot 1, no scaling) | 136 ms | 1.14 |
| `with_options` (same given column order) | upwind2d-316, n = 99856, nnz = 498016 | 254 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 135 ms | 1.88 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | upwind2d-316, n = 99856, nnz = 498016 | 136 ms | UMFPACK (defaults) | 146 ms | 0.934 |
| `solve_vec` | upwind2d-316, n = 99856, nnz = 498016 | 3.00 ms | SciPy SuperLU.solve | 5.04 ms | 0.595 |
| `solve_transpose_vec` | upwind2d-316, n = 99856, nnz = 498016 | 5.36 ms | faer | 4.91 ms | 1.09 |
| `new` | lap2d-316, n = 99856, nnz = 498016 | 156 ms | UMFPACK (unsymmetric, COLAMD, pivot 1, no scaling) | 135 ms | 1.16 |
| `with_options` (same given column order) | lap2d-316, n = 99856, nnz = 498016 | 256 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 134 ms | 1.90 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | lap2d-316, n = 99856, nnz = 498016 | 137 ms | UMFPACK (defaults) | 145 ms | 0.939 |
| `solve_vec` | lap2d-316, n = 99856, nnz = 498016 | 3.63 ms | CSparse cs_lu solve | 5.03 ms | 0.723 |
| `solve_transpose_vec` | lap2d-316, n = 99856, nnz = 498016 | 5.26 ms | faer | 4.73 ms | 1.11 |
| `new` | convdiff3d-10, n = 1000, nnz = 6400 | 1.99 ms | UMFPACK (defaults) | 1.60 ms | 1.24 |
| `with_options` (same given column order) | convdiff3d-10, n = 1000, nnz = 6400 | 2.71 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 1.65 ms | 1.65 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | convdiff3d-10, n = 1000, nnz = 6400 | 1.53 ms | UMFPACK (defaults) | 1.62 ms | 0.946 |
| `solve_vec` | convdiff3d-10, n = 1000, nnz = 6400 | 35.5 µs | KLU | 33.0 µs | 1.08 |
| `solve_transpose_vec` | convdiff3d-10, n = 1000, nnz = 6400 | 53.5 µs | faer | 39.0 µs | 1.37 |
| `new` | convdiff3d-22, n = 10648, nnz = 71632 | 71.6 ms | UMFPACK (defaults) | 56.1 ms | 1.28 |
| `with_options` (same given column order) | convdiff3d-22, n = 10648, nnz = 71632 | 290 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 143 ms | 2.02 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | convdiff3d-22, n = 10648, nnz = 71632 | 67.5 ms | UMFPACK (defaults) | 57.6 ms | 1.17 |
| `solve_vec` | convdiff3d-22, n = 10648, nnz = 71632 | 1.33 ms | Accelerate SparseSolve (LU) | 1.37 ms | 0.976 |
| `solve_transpose_vec` | convdiff3d-22, n = 10648, nnz = 71632 | 2.30 ms | faer | 2.42 ms | 0.952 |
| `new` | circuit-3000, n = 3000, nnz = 12539 | 6.01 ms | UMFPACK (defaults) | 6.33 ms | 0.949 |
| `with_options` (same given column order) | circuit-3000, n = 3000, nnz = 12539 | 70.1 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 28.3 ms | 2.48 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | circuit-3000, n = 3000, nnz = 12539 | 4.63 ms | UMFPACK (defaults) | 6.58 ms | 0.703 |
| `solve_vec` | circuit-3000, n = 3000, nnz = 12539 | 98.9 µs | KLU | 93.4 µs | 1.06 |
| `solve_transpose_vec` | circuit-3000, n = 3000, nnz = 12539 | 156 µs | KLU | 146 µs | 1.07 |
| `new` | circuit-10000, n = 10000, nnz = 41824 | 65.2 ms | Accelerate SparseFactor LU (default) | 62.5 ms | 1.04 |
| `with_options` (same given column order) | circuit-10000, n = 10000, nnz = 41824 | 2.32 s | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 935 ms | 2.48 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | circuit-10000, n = 10000, nnz = 41824 | 60.4 ms | Accelerate SparseFactor LU (default) | 62.4 ms | 0.968 |
| `solve_vec` | circuit-10000, n = 10000, nnz = 41824 | 914 µs | KLU | 1.17 ms | 0.783 |
| `solve_transpose_vec` | circuit-10000, n = 10000, nnz = 41824 | 1.63 ms | KLU | 1.68 ms | 0.972 |
| `new` | tridiag-100000, n = 100000, nnz = 299998 | 21.1 ms | KLU (defaults: BTF, AMD) | 6.99 ms | 3.02 |
| `with_options` (same given column order) | tridiag-100000, n = 100000, nnz = 299998 | 14.7 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 23.9 ms | 0.617 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | tridiag-100000, n = 100000, nnz = 299998 | 14.5 ms | KLU (defaults: BTF, AMD) | 7.03 ms | 2.07 |
| `solve_vec` | tridiag-100000, n = 100000, nnz = 299998 | 928 µs | KLU | 666 µs | 1.39 |
| `solve_transpose_vec` | tridiag-100000, n = 100000, nnz = 299998 | 908 µs | KLU | 722 µs | 1.26 |
| `new` | tridiag-1000000, n = 1000000, nnz = 2999998 | 216 ms | KLU (defaults: BTF, AMD) | 72.0 ms | 3.00 |
| `with_options` (same given column order) | tridiag-1000000, n = 1000000, nnz = 2999998 | 144 ms | UMFPACK (given Q, unsymmetric, pivot 1, no scaling) | 238 ms | 0.605 |
| `with_options` (AMD of A+Aᵀ, threshold 0.1) | tridiag-1000000, n = 1000000, nnz = 2999998 | 150 ms | KLU (defaults: BTF, AMD) | 71.2 ms | 2.11 |
| `solve_vec` | tridiag-1000000, n = 1000000, nnz = 2999998 | 9.36 ms | KLU | 7.18 ms | 1.30 |
| `solve_transpose_vec` | tridiag-1000000, n = 1000000, nnz = 2999998 | 9.00 ms | KLU | 7.13 ms | 1.26 |

The rows of `new` were measured with the diagonal of the symmetric strategy kept down to 0.001 and its watch at 64 times the largest element. On every matrix of the table the present values, 10⁻⁴ and 16, give the same factors bit for bit (every pivot on the diagonal, the growth at most 2.3), and `new` took 0.97 to 1.03 of the time of the values before (21 alternated samples, twice), so the rows stand.

Where the references are faster:

- `new` on the tridiagonal matrices (3.00 and 3.02 times KLU's defaults, 2.07 and 2.11 for the symmetric strategy): on a matrix that fills nothing the ordering and the estimate of the condition, at most seventeen solves, which KLU's defaults do not compute, are a large part of the time. On the smallest matrix (2.02 times KLU's) it is the fixed cost of the factorisation.
- `new` on the grids (1.09 to 1.28 times UMFPACK's defaults, 1.16 and 1.14 times its unsymmetric strategy on the Laplacian and the upwind grid): the estimate of the condition in every factorisation, and the supernodal elimination on the crate's own dense kernels against UMFPACK's frontal matrices on OpenBLAS. By the symmetric strategy it takes 0.934 to 0.950 of UMFPACK's time on the 2-D grids and the cube of 10³, 1.17 times it on the cube of 22³, and 0.703 to 0.968 of the fastest reference's on the circuits.
- The same given column order (1.18 to 2.48 times UMFPACK; 0.605 and 0.617 on the tridiagonal): UMFPACK still refines the order within its fronts, and its multifrontal elimination keeps the work in dense frontal matrices, where the left-looking supernodal elimination here starts from columns.
- The solves: `solve_vec` 1.04 to 1.39 times KLU's on the small matrices, the circuit of 3,000 nodes and the tridiagonal ones, and `solve_transpose_vec` 1.06 to 1.37 times faer's or KLU's on all but two: the permutations into fresh vectors and the fixed cost of a solve. `solve_vec` on the larger grids and the circuit of 10⁴ nodes takes 0.595 to 0.976 of the fastest reference's time.

### Limits

The scaled matrices below are diagonally dominant by rows on a scrambled diagonal, of orders up to 31, factored with the default threshold under random and COLAMD orders of their columns, 2 000 of them for each element type, kind of scaling and range; `10^±d` means rows or columns multiplied by factors between `10⁻ᵈ` and `10ᵈ`, and the error is that of a solution whose elements have the sizes of the columns (`x_j = (1 + |u|)/c_j`), relative to its largest element (a probe outside the crate).

- No permutation to block triangular form, no iterative refinement, no parallelism. Matrices of order below 64 are eliminated a column at a time, without dense blocks, and so are larger ones until a column of the factors holds 96 elements of `L` and `U`: eliminated in its natural order, an arrow whose first column holds 95 elements below the diagonal never switches, one whose first column holds 96 switches after its first step, and a tridiagonal matrix never does.
- Singular to working precision is decided from an estimate, against a bound, and both kinds of error remain, rarely. In the campaigns of review round 2 on the final code, 7 of 6.3 million factorisations of exactly singular matrices were called regular, all with a pivot threshold of `10⁻⁴`, where the estimates were exact and the factors lay 1 to 5 `τ` from a singular matrix: `τ = ε·max(1, ρ)` has no factor of the order `n`, which the backward error of an elimination carries. None was missed at the default threshold or at 0.1. A matrix with a scaled condition just beyond the bound, or one whose elimination grew, is reported though it is regular. A matrix whose condition number is close to `1/ε` gives solutions with few correct digits whichever way it is decided.
- Rows of very different sizes never make a matrix reported (tested to `10^±10` in single and `10^±20` in double precision). The rows are not scaled for the elimination, though, and partial pivoting on unscaled rows is unstable when their sizes differ by far more than `1/ε`: no solution was wrong by more than 0.1 up to `10^±4` in single and `10^±12` in double precision, but 1 and 6 of 2 000 were in `f32` at `10^±5` and `10^±6` (2 in `Complex<f32>` at `10^±6`), 1 in `f64` and 6 in `Complex<f64>` at `10^±15`, and more beyond, without `is_singular()` or `growth()`, taken over the whole matrix, showing it. Pivoting on the scaled rows, or iterative refinement after Skeel, would cure it; neither is done.
- Columns of very different sizes can make a well-conditioned matrix reported singular, because one pass of the scaling, rows first, does not undo them: none up to `10^±3` in single and `10^±6` in double precision; 0.1 % at `10^±8` in double; 5 to 6 % at `10^±4` in single and 8 to 9 % at `10^±10` in double precision; about 70 % at the widest ranges tried. Their factors would have solved them, to `10⁻⁶` as a rule: the answer `None` is a refusal, not a wrong result, and a solution that is returned was never wrong by more than `3·10⁻⁶`. Bringing the columns to like sizes before the factorisation, and solving for the correspondingly scaled unknowns, avoids it.
- With rows and columns both scaled, nothing is reported and no solution is wrong by more than 0.1 up to `10^±3` in single and `10^±6` in double precision; beyond, refusals as for the columns, and in some of the ranges one of the 2 000 matrices accepted with a solution wrong by more than 0.1. A solution of all ones measures the problem rather than the elimination here: its components in the small columns are not determined by `b` to the precision of the type, and its errors are large for any solver.
- A matrix refused for the growth of its elimination need not be near a singular one. Wilkinson's matrix (ones on the diagonal and in the last column, −1 below the diagonal), of condition number `n` and growth `2^(n−1)`, is reported from order 20 in single and 48 in double precision, and `det()` is then 0, although its pivots are exact and its determinant is `2^(n−1)`.
- Once a matrix is reported, every solve returns `None`: there is no solve that ignores the verdict, as LAPACK's `xGESVX` returns its solution with a warning. Shift-and-invert with a shift within rounding of an eigenvalue, where the large component of the solution is the one wanted, is refused for that reason.
- The fill is decided by the column ordering and the pivots. `new` orders a nearly symmetric pattern by `amd` of `A + Aᵀ` and any other by COLAMD; the numbers of the table above are for named orderings. On the matrices of grids and meshes with a strong diagonal the symmetric order fills 0.4 to 0.85 of what COLAMD with partial pivoting does: 0.5 to 0.7 on 2-D grids, 0.4 to 0.5 on 3-D seven-point grids of 10³ to 30³ points, 0.7 to 0.85 on 3-D elasticity meshes of 648 to 12 288 unknowns (Measured). A nearly symmetric pattern whose diagonal is weak — shifted Laplacians and Helmholtz operators, `A − σI` for a shift inside the spectrum — keeps its pivots on the diagonal and its fill, but its elements grow by hundreds to tens of thousands, and it is factored a second time by partial pivoting: its fill is then that of COLAMD with partial pivoting, 1.6 to 1.9 times what UMFPACK's symmetric strategy, which keeps such a diagonal and refines the solutions, makes of it, and its time that of partial pivoting and of the part of the symmetric elimination made before the watch stopped it.
- Partial pivoting can spoil the fill of a symmetric ordering on an unsymmetric problem, as the convection-dominated grids above show (70.6 million elements against 7.6 million); a pivot threshold below 1, or an unsymmetric ordering such as COLAMD, keeps it.
- The symmetric orderings are computed on the pattern of `Aᴴ·A`, which a dense row of `A` makes dense, and with it the time and memory of the ordering.
- `det()` is a product of `n` pivots and overflows or underflows easily; there is no logarithm of it. It is NaN, not 0, for a matrix with an element that is not finite, and for complex elements an overflow can come out as NaN rather than infinity (`diag(10³)` of order 20 in `Complex<f32>`).
- `rcond()` is a lower bound of the norm of the inverse turned into an upper bound of the reciprocal, and not always a close one: on 920 000 random matrices it was up to 22 times the true value, before the two steps after the search were added, which can only bring it closer. It is 0 for a matrix `is_singular()` reports and when the true value underflows.
- The estimate of the condition is made in every factorisation, at most seventeen solves (thirty-four near the bound), which a later version could make lazily.
- `nnz()` counts the elements the elimination reaches, the few that come out exactly zero among them; `l()` and `u()` leave those out, as every `SparseMatrix` does.
- Whether the symmetric pruning shortens the searches has not been measured (the reach and the factors are the same with or without it).

## Sparse QR

`SparseQr<T>` is the QR decomposition `A·Q_c = Q·R` of a sparse `m × n` matrix with `m ≥ n`: `Q_c` a permutation of the columns chosen to keep the factors sparse, `Q` of `m × n` with orthonormal columns, kept as the Householder reflections that make it, and `R` of `n × n`, upper triangular and sparse. It solves the least-squares problem `min ‖A·x − b‖₂` without forming `Aᴴ·A`, whose condition number is the square of that of `A`.

| Call | Gives |
| --- | --- |
| `SparseQr::new(&a)` | the decomposition, with the columns in the order of `Ordering::Colamd` |
| `SparseQr::with_ordering(&a, &ordering)` | the same in the order of any `Ordering`: `Natural`, `Given(q)`, `Colamd` on the pattern of `A`, or `Amd`, `ReverseCuthillMckee` and `NestedDissection` on that of `Aᴴ·A` |
| `r()` | `R` as a `SparseMatrix<T>`, the elements of its pattern that came out zero left out |
| `apply_q(&x)`, `apply_qh(&b)` | `Q·X` for `X` of `n` rows and `Qᴴ·B` for `B` of `m` rows, without forming `Q` |
| `q()` | the thin `Q`, `m × n` and dense: for small matrices and tests |
| `column_permutation()` | `Q_c` as `q` with `q[new] = old`: column `k` of `A·Q_c` is column `q[k]` of `A` |
| `solve(&b)`, `solve_vec(&b)` | the least-squares solution of each column of `B`, or `None` when `R` is singular to working precision: a small diagonal element, or an estimate of the condition number beyond `1/(max(m, 32)·ε)` |
| `rank()` | an estimate of the rank, from the diagonal of `R` and the condition estimate: below `n` exactly when `solve` refuses on account of the matrix |
| `nnz()` | the elements stored for `R` and for the Householder vectors |

A matrix of fewer rows than columns panics, and so do a right-hand side or a product of the wrong height and a given ordering that is not a permutation of the columns. A matrix that holds a NaN or an infinity, or overflows, is decomposed all the same, and `solve` refuses it; it also refuses a right-hand side that holds one, and a solution beyond the range of the type.

### The method

The symbolic analysis reads the pattern alone, and comes first (George and Heath, "Solution of sparse linear least squares problems using Givens rotations", *Linear Algebra Appl.* 34, 1980; Liu, "On general row merging schemes for sparse Givens transformations", *SIAM J. Sci. Stat. Comput.* 7, 1986; Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006, chapter 5):

- The column elimination tree, the elimination tree of `(A·Q_c)ᴴ·(A·Q_c)`, by Liu's algorithm with path compression over the rows of `A`. A row with elements in the columns `c₁ < c₂ < …` makes a clique of them in `Aᴴ·A`, whose tree is the path `c₁ → c₂ → …`, so each column is linked to the previous column of each of its rows alone and the product is never formed; the cost is nearly linear in the stored elements.
- The fronts: chains of the tree whose rows of `R` have the same pattern beyond their leading columns, the fundamental supernodes, each with the columns its rows reach. The row of `R` of a column reaches the columns of the rows of `A` that begin in it and those its children in the tree pass on, a child that has no row left over passing none. So the patterns are those the rows really make: never larger than the Cholesky factor's of `Aᴴ·A`, and equal to it for a matrix with the strong Hall property (Coleman, Edenbrandt and Gilbert, "Predicting fill for sparse orthogonal factorization", *J. ACM* 33, 1986). Then, in a matrix of 128 columns or more, small fronts are merged into their parents, from the leaves up, by the rule of the Cholesky factorisation's supernodes on the zeros the merges add to `R`, those of earlier merges counted: always when the merged front has at most 4 pivot columns or the merge adds no zero, up to 16 pivot columns with less than 30 % zeros, up to 48 with less than 10 %, any number with less than 5 % (CHOLMOD's thresholds, which SuiteSparseQR's analysis uses too, but for its 80 % for up to 16 pivot columns, which on a path stored nearly twice the elements for a slower decomposition); along paths of the tree only, and only a child that has no row to drop below its triangle: a merged child puts its pivot columns before its parent's, its rows of `R` take all the parent's columns, and its rows are reflected in the parent's front instead of being reduced to its own triangle first. The zeros that adds to the rows of `R` stay exact, so `r()` still has the pattern the rows make; the vectors grow by the rows that are reflected with the parent's columns.
- A postorder of the fronts and, for each, its rows sorted by their first columns, the staircase. From it come the length of every Householder vector, the pattern of every row of `R`, and the place of every contribution block in a stack, with the stack's peak, so that the numeric phase allocates its results once and works in two buffers, the largest front and the stack.

The numeric phase is the multifrontal Householder QR (Matstoms, "Sparse QR factorization in MATLAB", *ACM Trans. Math. Softw.* 20, 1994; Amestoy, Duff and Puglisi, "Multifrontal QR factorization in a multiprocessor environment", *Numer. Linear Algebra Appl.* 3, 1996; Davis, "Algorithm 915, SuiteSparseQR", *ACM Trans. Math. Softw.* 38, 2011). Each front, in postorder, is assembled from the rows of `A` that begin in its pivot columns and from the contribution blocks of its children, and reduced to an upper triangle by Householder reflections (LAPACK's `larfg`), each confined to the rows of its column's step of the staircase: in a front of at most 64 columns each reflection is applied to the columns after it as it is made, and in a wider front they come in panels of 16 columns: within a panel each reflection is applied to the panel's later columns at once, and the columns right of the panel take the panel's reflections four columns at a time, one reflection after another, the panel's vectors staying in the cache. A front of 128 reflections or more whose staircase fills at least half of its rectangle is held dense, and the columns right of each panel take the panel's reflections as one block `I − V·T·Vᴴ` (the compact WY form of Schreiber and Van Loan, *SIAM J. Sci. Stat. Comput.* 10, 1989), by three products of dense blocks; below that size the columns at a time were faster. The first rows of the triangle are rows of `R`, the rest of it is the contribution block that the parent takes in, and the rows below it are zero and dropped. The work is that of the reflections inside the staircases and the memory that of `R`, of the vectors and of the two buffers: nothing is of the order of `m·n` unless the factors are.

`Q` is never formed. It is the product of the fronts' reflections, each a dense vector on the rows of its front (Lu and Barlow, "Multifrontal computation with the orthogonal factors of sparse matrices", *SIAM J. Matrix Anal. Appl.* 17, 1996), applied front by front by `apply_q` and `apply_qh` at the cost of their stored elements for each column. `solve` applies `Qᴴ`, solves `R·z = Qᴴ·b` by back substitution on the rows of `R`, and puts the columns back in their order.

The diagonal of `R` is real and keeps the signs the reflections leave, as LAPACK's: `R_jj = −sign(Re α)·‖(α, x)‖`, the sign of zero taken as positive, for `α` the first element of the column in the rows that reach it and `x` the others; where `x` is zero and `α` real there is nothing to reflect and `R_jj = α`. So a real diagonal matrix is its own `R` (a complex one is turned real: `diag(i, 3 − 4i)` has the `R` of `diag(−1, −5)`), and up to the signs of its rows `R` is the conjugate transpose of the Cholesky factor of `(A·Q_c)ᴴ·(A·Q_c)`.

There is no pivoting for rank, and `solve` doesn't trust the diagonal of `R` alone to say whether `A` is singular to working precision. The smallest singular value of a triangle is no larger than its smallest diagonal element, but it can be much smaller: the bidiagonal matrix of 1 on the diagonal and −2 above it has every diagonal element 1 and a condition number of `2.3·10¹⁸` at `n = 60`, and a column that is an exact combination of others with large coefficients leaves on the diagonal a rounding error that they amplify (with a column `2ᵖ` times the difference of two others, every diagonal element passed the tolerance below in up to a third of random such matrices, by up to about `2ᵖ/10` times). So `solve` refuses `R` when a diagonal element is below `max(m, 32)·ε·c`, `c` the largest 2-norm of a column of `A`, or when a lower bound of the condition number of `R`, which is that of `A`, passes `1/(max(m, 32)·ε)`. The bound is `c·‖R⁻¹·x‖₂/‖x‖₂` (`c` is no more than `‖R‖₂`) for the vectors of six solves with `R` and `Rᴴ`: LINPACK's estimate (Cline, Moler, Stewart and Wilkinson, "An estimate for the condition number of a matrix", *SIAM J. Numer. Anal.* 16, 1979), which solves `Rᴴ·y = e` choosing each `e_k` of modulus 1 to make `|y_k|` the largest, then five steps of the power method on `(Rᴴ·R)⁻¹`, each normalised; the solves cost the stored elements of `R` each, with `R` read divided by `c` and each stopped once its norm passes the limit, so that nothing overflows. The bound never passes the condition number, so a matrix below the limit is not refused, up to the rounding errors of the decomposition; it can fall short of it, by up to `√n` from `c` alone: in the families measured below no matrix beyond twice the limit was solved, but a review solved Kahan's matrices with condition numbers up to 6.9 times the limit (`n = 138`) and random dense ones up to 4 times it. With an accurate estimate of the inverse, a matrix the estimate lets through has a condition number below about `√n` times the limit.

`rank()` counts the diagonal elements of `R` above that tolerance, one fewer when that is all of them but the condition estimate passes the limit, so that the count is below `n` exactly when `solve` refuses a matrix of finite elements on account of the matrix (a right-hand side that is not finite, or a solution beyond the range of the type, is refused at full rank). The tolerance is NumPy's `matrix_rank`, `max(m, n)·ε·σ₁` on the singular values, with the diagonal of `R` for the singular values and `c`, between `σ₁/√n` and `σ₁`, for `σ₁`; the floor of 32 rows covers the rounding error that a column which copies, scales by a small number or sums others leaves on the diagonal of a small matrix, up to `7·ε·c` over millions of random ones from `2 × 2` to `32 × 32` (and about `0.07·m·ε·c` at thousands of rows, under `m·ε·c`). The count is an estimate and can miss either way: two columns hidden as above are counted as one, and a dependent column eliminated before an independent one can leave the independent one too little (see Limits).

### Example

```rust
use a::math::matrix::{Matrix, Norm};
use a::math::sparse::{Ordering, SparseMatrix, SparseQr};

// The straight line nearest to four points, three of them on y = 1 + 2x.
let a = SparseMatrix::from_coo(4, 2, &[
    (0, 0, 1.0f64),
    (1, 0, 1.0), (1, 1, 1.0),
    (2, 0, 1.0), (2, 1, 2.0),
    (3, 0, 1.0), (3, 1, 3.0),
]);
let b = Matrix::from_rows(&[[1.0], [3.0], [5.0], [6.0]]);
let qr = SparseQr::new(&a);
let x = qr.solve(&b).unwrap();
assert!((x[(0, 0)] - 1.2).abs() < 1e-14 && (x[(1, 0)] - 1.7).abs() < 1e-14);

// The residual is orthogonal to the columns, and Q·R is A with its columns reordered.
assert!((&a.transpose() * &(&a * &x - &b)).norm(Norm::Max) < 1e-14);
let permuted = Matrix::from_fn(4, 2, |i, j| a[(i, qr.column_permutation()[j])]);
assert!((&qr.q() * &qr.r().to_matrix() - &permuted).norm(Norm::Max) < 1e-14);

// Two equal columns: rank 1, and no single solution.
let twice = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 2.0)]);
let qr = SparseQr::new(&twice);
assert_eq!(qr.rank(), 1);
assert!(qr.solve_vec(&[1.0, 2.0]).is_none());

// A curve of 1000 noisy samples smoothed by least squares, [I; 2·D] with D the first
// differences: three elements to a column, and the factors stay as sparse.
let n = 1000;
let mut entries = Vec::new();
for i in 0..n {
    entries.push((i, i, 1.0f64));
}
for i in 0..n - 1 {
    entries.push((n + i, i, -2.0));
    entries.push((n + i, i + 1, 2.0));
}
let a = SparseMatrix::from_coo(2 * n - 1, n, &entries);
let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
// R is upper bidiagonal, and R with the vectors stores 7 elements a column.
assert_eq!(qr.r().nnz(), 2 * n - 1);
assert_eq!(qr.nnz(), 7 * n - 7);
let data: Vec<f64> = (0..2 * n - 1)
    .map(|i| if i < n { ((37 * i) % 9) as f64 / 4.0 - 1.0 } else { 0.0 })
    .collect();
let x = qr.solve_vec(&data).unwrap();
// The normal equations hold: Aᵀ·(A·x − b) = 0.
let residual: Vec<f64> = a.mul_vec(&x).iter().zip(&data).map(|(ax, b)| ax - b).collect();
let normal = a.transpose().mul_vec(&residual);
assert!(normal.iter().all(|v| v.abs() < 1e-12));

// A triangle singular to working precision with no small diagonal element: 1 on the diagonal
// and −2 above it, its own R, of condition number 2.3·10¹⁸ at n = 60.
let n = 60;
let a = SparseMatrix::from_fn(n, n, |i, j| {
    if i == j { 1.0f64 } else if j == i + 1 { -2.0 } else { 0.0 }
});
let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
assert!(qr.r().diagonal().iter().all(|d| d.abs() == 1.0));
// The condition estimate finds it: one short of full rank, and no solution.
assert_eq!(qr.rank(), n - 1);
assert!(qr.solve_vec(&vec![1.0; n]).is_none());
```

### Measured

The tests are in `lib/tests/math/sparse/qr.rs` (`cargo test --test math -- sparse::qr::`); the numbers below are printed by its ignored test, `cargo test --release --test math -- --ignored --nocapture sparse::qr::`.

Against NumPy: seven least-squares problems built by integer formulas whose values are dyadic, exact in `f32` and `f64` alike, in `lib/tests/math/sparse/data/gen_qr.py`: square `12 × 12`, tall `36 × 12`, very tall `300 × 15` (`m = 20·n`), the smoothing of a curve `79 × 40`, and complex `9 × 9`, `40 × 10` and `200 × 10`. The references are `numpy.linalg.lstsq`'s solutions, which the script checks against the normal equations solved by mpmath at 60 digits (they agree to `1.2·10⁻¹⁵`) and against `scipy.sparse.linalg.lsqr` (to `7.3·10⁻¹⁶`). The worst distance of `solve` from them over four orderings, relative to the solution, in units of `κ·ε` with `κ = cond + cond²·‖r‖/(‖A‖·‖x‖)` the condition number of the least squares (from 2.4 to 14.4 here); the real problems run on the complex types too, as complex matrices without imaginary parts:

| type | square | tall | very tall | curve | complex square | complex tall | complex very tall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `f32` | 1.39 | 0.35 | 0.19 | 0.16 | | | |
| `f64` | 2.68 | 0.55 | 0.42 | 0.40 | | | |
| `Complex<f32>` | 1.39 | 0.35 | 0.19 | 0.16 | 0.79 | 0.27 | 0.29 |
| `Complex<f64>` | 2.68 | 0.55 | 0.42 | 0.40 | 1.27 | 0.70 | 0.64 |

The tests bound that distance at 8, and the norm of the residual at `8·cond·ε·(‖A‖·‖x‖ + ‖b‖)` from NumPy's. On random matrices of every type, from `1 × 1` to `400 × 20` under every `Ordering` and two random given permutations, they bound `‖A·Q_c − Q·R‖_F` and `‖QᴴQ − I‖_F` at `16·max(m, n)·ε` times the norms (plus one), check `Qᴴ·(A·Q_c) = R` and `Q·Qᴴ·y = y` for `y` in the range of `A` at `16·max(m, n)·ε`, and `R` row by row against the dense Cholesky factor of `(A·Q_c)ᴴ·(A·Q_c)`, up to the sign of the row, at `4·cond²·n·ε`; on strong Hall matrices the two have the same pattern, element for element. On random matrices up to `1000 × 50`, the solutions of three right-hand sides are within `32·max(n, 3)·ε` (times the norms, plus one) of the dense `Matrix::qr().solve`, and `‖Aᴴ·(A·x − b)‖` is below `16·m·ε·‖A‖·(‖A‖·‖x‖ + ‖b‖)`.

Rank: a column scaled by any power of two from `2⁻²⁰` to `2²⁰` (`2⁻⁶` to `2⁶` in single precision), copied, negated, multiplied by a number from 2 to 10, added to another, combined with small integers or left empty, in 2000 random matrices of each type, from square to twenty times as many rows as columns:

| type | `m = n` | `m = 2·n` | `m = 5·n` | `m = 20·n` |
| --- | --- | --- | --- | --- |
| `f32` | 416 | 499 | 500 | 500 |
| `f64` | 424 | 499 | 500 | 500 |
| `Complex<f32>` | 436 | 495 | 500 | 500 |
| `Complex<f64>` | 439 | 495 | 500 | 500 |

of 500 counted right under a random column order, 500 of 500 under the order that puts the dependent columns last, and none of these, whose dependencies have small coefficients, counted above the rank; the counts that fell short fell short by up to 3, and `solve` refused every one of these matrices. The ranks of `gen_qr.py`'s two rank-deficient cases are `numpy.linalg.matrix_rank`'s in the natural order, where each dependent column comes after those it depends on.

Singular matrices whose diagonal doesn't show it: 2000 of each type made near the limit, upper bidiagonal matrices of 1 and about `−β`, and sparse matrices with a column that is a combination of others with coefficients up to `2¹¹` and a small random part, each in a random column order. By their condition number in units of the limit `1/(max(m, 32)·ε)`, from the dense singular values in double precision, those refused of those tried:

| type | below 0.5 | 0.5 to 0.9 | 0.9 to 1 | 1 to 1.1 | 1.1 to 2 | 2 to 4 | 4 to 100 | beyond 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `f32` | 0 of 474 | 0 of 47 | 0 of 12 | 3 of 9 | 37 of 39 | 56 of 56 | 321 of 321 | 1042 of 1042 |
| `f64` | 0 of 416 | 0 of 45 | 0 of 10 | 9 of 12 | 44 of 45 | 58 of 58 | 276 of 276 | 1138 of 1138 |
| `Complex<f32>` | 0 of 486 | 0 of 58 | 0 of 7 | 3 of 7 | 32 of 34 | 53 of 53 | 263 of 263 | 1092 of 1092 |
| `Complex<f64>` | 0 of 382 | 0 of 46 | 0 of 13 | 3 of 9 | 36 of 38 | 55 of 55 | 308 of 308 | 1149 of 1149 |

None below the limit was refused, and none of these beyond twice the limit solved; the tests check that margin of 2 on both sides for these families, and that the count of the rank is below `n` exactly when `solve` refuses. Of 3000 matrices of each type with a column `2ᵖ` times the difference of two others (`p` from 4 to 11, the elements exact), each in four column orders, all 12 000 were refused, and the count of the rank was at most one above the rank of the dense matrix. The tests also refuse, in every column order and every type, a `3 × 3` matrix with a column `2⁸` times the difference of the two others, two structurally rank-deficient matrices (`13 × 5` and `18 × 16`, SciPy's `structural_rank` 4 and 15), and the bidiagonal matrix above at `n = 60`.

Growth, in the release build: three problems, the last in two orders, at about `10⁴`, `3·10⁴` and `10⁵` rows; the table shows the largest. What the factors store is deterministic, and printed by the ignored test. `nnz(R)` is `r().nnz()`, the elements of `R` that came out nonzero; `nnz()` counts the patterns, the zeros the merges of fronts add included. Of the natural order's 6.1 million, 1.2 million are rounding residues, at most `2·10⁻¹⁶` of their rows, in places the rows' pattern holds that are zero in exact arithmetic: its fronts of 128 reflections or more are held dense and reduced by blocks, whose rounding does not leave them exactly zero as the reflections one at a time do (4.9 million when only fronts of 256 reflections or more were). The work is the number of elements the reflections of the numeric phase touch, counted in a build of the code with a counter added for the purpose, which is free of the noise of the clock; the times of the final measurements are under **Time** below.

| problem | `m` | `n` | `nnz(A)` | `nnz(R)` | `nnz()` | growth of `nnz()` with `m` | growth of the work |
| --- | --- | --- | --- | --- | --- | --- | --- |
| curve `[I; 2·D]`, natural order | 99 999 | 50 000 | 149 998 | 99 999 | 349 993 | `m¹·⁰⁰` | `m¹·⁰⁰` |
| very tall band, `m = 20·n`, natural order | 100 000 | 5 000 | 270 951 | 48 112 | 1 043 704 | `m¹·⁰⁰` | `m¹·⁰¹` |
| grid `[I; 2·Dx; 2·Dy]` of `183 × 183`, nested dissection | 100 101 | 33 489 | 166 713 | 1 100 043 | 4 633 467 | `m¹·¹⁵` | `m¹·⁵⁴` |
| the same grid, natural order (band 183) | 100 101 | 33 489 | 166 713 | 6 112 888 | 30 526 040 | `m¹·⁵⁰` | `m¹·⁹⁹` |

The exponents are `log(y₃/y₂)/log(m₃/m₂)`, between the two larger sizes. They are those of a sparse method: linear for the curve and the band, about `n·log n` storage and `n¹·⁵` work for the grid in nested dissection order, and `n¹·⁵` storage and `n²` work for the grid as a band, which is what its fill is. The nested dissection order of the grid is given as `Ordering::Given`, the bisection of the grid by its middle lines. The condition estimate reads `6·nnz(R)` elements: one and a half times the work of the reflections for the curve, whose fronts have two columns, and 5 %, 4 % and 2 % of it for the band and the grid in its two orders.

**Time** (conditions as under [Storage and operations](#storage-and-operations)): `new`, the analysis, the factorisation and the estimate of the condition, against each library's own; `solve_vec` the least-squares solution of one right-hand side on each library's factors:

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `new` | curve-5000, 9999 × 5000, nnz = 14998 | 1.88 ms | Accelerate SparseFactor QR | 862 µs | 2.19 |
| `solve_vec` | curve-5000, 9999 × 5000, nnz = 14998 | 123 µs | Accelerate SparseSolve (QR) | 521 µs | 0.237 |
| `new` | curve-50000, 99999 × 50000, nnz = 149998 | 18.2 ms | Accelerate SparseFactor QR | 8.33 ms | 2.19 |
| `solve_vec` | curve-50000, 99999 × 50000, nnz = 149998 | 1.14 ms | Accelerate SparseSolve (QR) | 5.17 ms | 0.221 |
| `new` | tallband-500, 10000 × 500, nnz = 26748 | 1.73 ms | Accelerate SparseFactor QR | 1.65 ms | 1.05 |
| `solve_vec` | tallband-500, 10000 × 500, nnz = 26748 | 88.9 µs | Accelerate SparseSolve (QR) | 346 µs | 0.257 |
| `new` | tallband-5000, 100000 × 5000, nnz = 267924 | 23.5 ms | Accelerate SparseFactor QR | 21.0 ms | 1.12 |
| `solve_vec` | tallband-5000, 100000 × 5000, nnz = 267924 | 1.40 ms | Accelerate SparseSolve (QR) | 4.16 ms | 0.336 |
| `new` | gridls-31, 2821 × 961, nnz = 4681 | 877 µs | Accelerate SparseFactor QR | 668 µs | 1.31 |
| `solve_vec` | gridls-31, 2821 × 961, nnz = 4681 | 59.6 µs | faer solve_in_place | 83.9 µs | 0.711 |
| `new` | gridls-100, 29800 × 10000, nnz = 49600 | 12.4 ms | Accelerate SparseFactor QR | 9.44 ms | 1.31 |
| `solve_vec` | gridls-100, 29800 × 10000, nnz = 49600 | 837 µs | faer solve_in_place | 1.27 ms | 0.659 |
| `new` | gridls-183, 100101 × 33489, nnz = 166713 | 51.3 ms | Accelerate SparseFactor QR | 38.1 ms | 1.35 |
| `solve_vec` | gridls-183, 100101 × 33489, nnz = 166713 | 3.28 ms | faer solve_in_place | 5.04 ms | 0.651 |
| `new` | randtall-1000, 3000 × 1000, nnz = 8991 | 46.4 ms | Accelerate SparseFactor QR | 20.2 ms | 2.29 |
| `solve_vec` | randtall-1000, 3000 × 1000, nnz = 8991 | 971 µs | faer solve_in_place | 750 µs | 1.29 |
| `new` | randtall-3000, 9000 × 3000, nnz = 26992 | 974 ms | Accelerate SparseFactor QR | 539 ms | 1.81 |
| `solve_vec` | randtall-3000, 9000 × 3000, nnz = 26992 | 8.62 ms | faer solve_in_place | 6.32 ms | 1.36 |
| `new` | lap2d-100, 10000 × 10000, nnz = 49600 | 17.3 ms | Accelerate SparseFactor QR | 12.9 ms | 1.34 |
| `solve_vec` | lap2d-100, 10000 × 10000, nnz = 49600 | 884 µs | faer solve_in_place | 1.17 ms | 0.754 |

Where the references are faster:

- Accelerate (`new` 1.05 to 2.29 times): its fronts run on the M4's matrix unit. Against SPQR and faer, on the cores, `new` takes 0.854 to 1.46 of their time: 1.46 times SPQR on the curve, whose fronts have two columns each and cost what their bookkeeping costs, and 1.06 to 1.20 on the grids and the random tall matrices, where fronts of fewer than 128 reflections are reduced in panels by the crate's own kernels, against SPQR's and faer's blocked products.
- `solve_vec` on the random tall matrices (1.29 and 1.36 times faer's); everywhere else it takes 0.221 to 0.754 of the fastest reference's time.

### Limits

- A matrix of fewer rows than columns panics: there is no minimum-norm solution of an underdetermined system.
- There is no pivoting for rank, so `rank()` is an estimate. In a square matrix a dependent column eliminated before an independent one still takes a row of `R`, as in the dense decomposition, and the count can come out below the rank (in 12 to 17 % of the square matrices above). It can come out above the rank too: two or more columns that depend on others through large coefficients, each leaving a diagonal element above the tolerance, are found by the condition estimate as one. `solve` refuses all of these. A column whose norm is below the tolerance counts as dependent whatever its direction. Heath's rule of dropping such a column from its front, as SuiteSparseQR does, is not applied.
- The condition estimate is a lower bound, which can fall short of the condition number; a matrix whose condition number is just beyond the limit can be solved (in the measurements, some between 1 and 2 times the limit were, and in a review Kahan's matrices up to 6.9 times it, since the bound takes the largest column norm `c`, up to `√n` below `‖R‖₂`), and like any estimate built from a few vectors it has no guarantee against a matrix made to deceive it. It costs six solves with `R`, about `6·nnz(R)` operations, once, in the decomposition.
- The limit `1/(max(m, 32)·ε)` falls with the number of rows, every row counted, empty ones too, as in NumPy's `matrix_rank` and SuiteSparseQR's tolerance, because the rounding errors of the reflections grow with their lengths. In single precision that is low for a very tall matrix: at `m = 10⁵` the limit is 84, and a problem of that height whose condition number is 100 is refused although its solution would have a few correct digits; such a problem is better solved in double precision, whose limit there is `4.5·10¹⁰`. From about `8.4·10⁶` rows, where `m·ε` reaches 1 in single precision, the tolerance reaches `c`: `rank()` is 0 and every matrix is refused, one with orthonormal columns too.
- A matrix whose largest column has a 2-norm beyond the largest finite number of the type gets `rank()` 0 and is refused, however well conditioned (`[[s, s], [0, s]]` with `s = 0.8·MAX`, of condition number 2.6), where the dense `Matrix::qr` solves it.
- A dense row or a dense column makes the fronts it reaches dense, and with them `Q`: a column with an element in every row puts the whole matrix in one front. The fill depends on the column order, which the ordering decides: `Amd`, `ReverseCuthillMckee` and `NestedDissection` order the pattern of `Aᴴ·A`, which is formed for them and which a dense row makes dense, where `Colamd`, the default, works on `A` itself.
- `nnz()` counts the elements of the patterns, including those that came out zero and those the merging of fronts adds: the zeros of a merged child's rows of `R` in its parent's columns, and the longer vectors of the parent's columns, which reflect the child's rows rather than the triangle the child would have reduced them to. `r()` drops the zeros.

## Iterative solvers and preconditioners

The Krylov methods solve `A·x = b` by improving a guess instead of factoring `A`. They touch the matrix only through products with vectors, so they are written against `LinearOperator<T>` and run on a `SparseMatrix<T>`, a dense `Matrix<T>`, or a function wrapped in an `FnOperator` that never forms the matrix at all. Complex systems use the inner product `Σ ūᵢ·vᵢ`.

| Function | For | Preconditioner `M`, and how | What its recurrence tracks | Products a step | Memory |
| --- | --- | --- | --- | --- | --- |
| `cg`, `cg_with` | Hermitian positive definite `A` | Hermitian positive definite, split (`L⁻¹·A·L⁻ᴴ` for `M = L·Lᴴ`, only `M⁻¹` applied) | `b − A·x` | 1 | 6 vectors, 7 once a best point is kept |
| `minres`, `minres_with` | Hermitian `A`, indefinite or singular | Hermitian positive definite, split | `‖r‖` in the norm of `M⁻¹`; with `M`, `b − A·x` carried beside it | 1 | 8 vectors, 9 with `M`, one more once a best point is kept |
| `gmres`, `gmres_with` | any square `A` | any nonsingular, on the right (`A·M⁻¹·u = b`, `x = M⁻¹·u`) | `‖b − A·x‖₂` itself | 1 | `m + 4` vectors (`m + 5` with `M`) and `m²/2` numbers, taken as the steps are |
| `bicgstab`, `bicgstab_with` | any square `A` | any nonsingular, on the right in both half steps | `b − A·x` | 2 (1 for a last half step) | 9 vectors, 10 once a best point is kept |

The memory counts vectors of the length of `b`, the solution among them and, for `cg` and `bicgstab`, a copy of the iterate of the smallest residual so far. A starting guess is not copied: the options hold it.

Each comes as `name(a, b)` with the default options and no preconditioner, and as `name_with(a, b, &opts, pre)` with `pre: Option<&dyn Preconditioner<T>>`. `SolveOptions` is built by `new()` and `with_rtol`, `with_atol`, `with_max_iter`, `with_restart` and `with_initial_guess`; the defaults are `rtol = √ε` of the element type, `atol = 0`, `max_iter` ten times the number of unknowns, a restart length of 30 (clamped to the number of unknowns) and the zero vector to start from. A negative or non-finite tolerance, and a restart length of 0 for `gmres`, are the only inputs that give `None`; wrong shapes panic.

### When a method stops, and what a result says

All four stop on one test, on the residual of the system as it was given whatever the preconditioner:

```text
‖b − A·x‖₂ ≤ max(rtol·‖b‖₂, atol)
```

The residual a recurrence carries drifts away from `b − A·x` as rounding errors accumulate, so it only says when to look. Before a result is marked converged its residual is formed afresh from the `x` returned, and the test is made on that; for a `SparseMatrix` and a `Matrix` that residual is formed in twice the working precision (Ogita, Rump and Oishi's `Dot2`), so it is the exact residual rounded once. If the look fails the method starts again from that residual, and a second look that finds no improvement ends it with `StopReason::Stagnated`. A tolerance below `ε·‖b‖₂` is looked at once the recurrence reaches that level, which is as far as a residual formed afresh can follow it. A `SolveResult` carries `x`, `residual` (that fresh norm, for the `x` returned), `iterations`, `matvecs` (every product with `A`, the fresh residuals included), `converged` and a `reason`: `Converged`, `IterationLimit`, `Breakdown`, `NotPositiveDefinite`, `NonFinite` or `Stagnated`.

A zero right-hand side gives `x = 0` at once, and a starting guess that already passes the test is returned at once. NaN or infinity in `b`, `A`, the guess or what the preconditioner returns ends the run with `NonFinite`. A run that fails returns the best point it reached, as MATLAB's solvers do, and the starting guess counts among the points reached, so no result is worse than where the run began: for `cg` and `bicgstab`, whose residuals rise and fall, the iterate of the smallest residual the recurrence carried, set against the last one by their fresh residuals; for `gmres` the better end of its last cycle; for `minres` the best of the points it verified on the way and the last. On its default budget no method runs for ever; with `max_iter` set to `usize::MAX`, `cg` on a matrix that is not positive definite, such as a singular one whose range `b` is not in, can go on until the arithmetic stops it, since its residual neither comes down nor shows that it cannot.

### The methods

`cg` is Hestenes and Stiefel's conjugate gradient method (J. Res. NBS 49, 1952) with the preconditioner applied as `z = M⁻¹·r`. Only a positive definite matrix makes `pᴴ·A·p` positive for every direction, so a direction with `pᴴ·A·p ≤ 0` ends the method with `NotPositiveDefinite`, and so does a residual with `rᴴ·M⁻¹·r ≤ 0`. That catches a preconditioner that is not positive definite.

`minres` is Paige and Saunders' MINRES (SIAM J. Numer. Anal. 12, 1975). The Lanczos process, in Paige's stable arrangement (`α` taken after `β·vₖ₋₁` is subtracted), reduces `A` to a tridiagonal matrix `T` whose QR factorisation by one Givens rotation a step gives the iterate of least residual over the Krylov space, and that residual's norm `|φ|` from the rotations alone. The first Lanczos vector is divided by its norm, so the scale of `b` enters `φ` alone. On a singular consistent system it converges, from `x = 0`, to the solution of least norm.

On a singular system whose right-hand side is not in the range, the residual comes down to its least squares value, after which, in floating point, the iterates drift along the null space while `|φ|` goes on falling. MINRES keeps watch with Choi, Paige and Saunders' estimate of `‖A·r‖` for the residual of its iterate ("MINRES-QLP", SIAM J. Sci. Comput. 33, 2011), which the rotations give for nothing: `‖A·rₖ₋₁‖ = |φ|·√(γ̄ₖ² + (cₖ₋₁·βₖ₊₁)²)`, `γ̄ₖ` the rotated pivot before the step and `cₖ₋₁` the cosine of the rotation before (the derivation is in a comment of the code). Measured against `‖T‖·‖r‖`, with `‖T‖` the largest column sum of the tridiagonal matrices so far, it decides:

- at most `4·ε`: `r` is a null vector to working precision and the iterate a least squares solution; the method stops with `Breakdown`. This is the clean case, the Krylov space exhausted in one step, as on a path's Laplacian;
- below `64·√ε`, and again at each level an eighth of the last: the iterate is verified, its residual formed afresh, and kept if it is the best. If `|φ|` then claims a residual a sixty-fourth below the smallest one formed afresh so far, the residual is formed afresh again. Once a residual verified anywhere in the run was a null vector to half the working precision (`√ε`), one that does not come out below that smallest residual means the iterates are drifting; one that comes out below it by less than half the decrease claimed means `|φ|` has lost track, and its next claim is checked against the new residual in turn. The evidence of a null vector is the run's, not the process's, since a process begun from a drifting iterate starts far from the least squares point and its own residuals may never come that near a null vector again; and the smallest residual, not the last one verified, is the measure, since a decrease from a point worse than the best says nothing about whether the best can be improved on. (An earlier version measured a claim against the point just verified and on that process's own evidence, and stopped watching once a claim was borne out at all. On `x86_64` that let the complex path of the tests, whose phases come from the platform's sine and cosine, run to its iteration limit while the iterates drifted to a residual of `10¹⁶`: a decrease of 2 % had come with the first step of a drift where `|φ|` claimed 18 %. The same happened on exactly singular systems, on right-hand sides a unit or two in the last place from the tests' and on four of twelve exactly singular complex paths in `Complex<f32>`.)

A drift so found, or an estimate of convergence that the residual formed afresh does not bear out, begins a new Lanczos process from the iterate, as long as each process improves on the best point (by a sixty-fourth after a drift); when one does not, the method stops with the best point, with `Breakdown` if a verified residual was a null vector to `√ε` and `Stagnated` otherwise. The new process starts from the iterate, not from the best point: on an ill-conditioned matrix in single precision the recurrence loses track of the residual long before any least squares question arises, and the iterate holds what the process found of the hard part of the solution even where rounding errors have spoilt its residual (on a graded diffusion problem of 100 unknowns in `f32` from a random test campaign, the iterate after 352 steps had a residual of 6.9, more than the start's, and a process begun from it brought that to 1.3·10⁻² in 108 steps; begun from the best point instead, the residual never went below 1.2; MINRES now ends there at 2.7·10⁻³, at 4.4 times the tolerance, where scipy does not converge either). There is no stall rule: a stretch of steps in which `|φ|` hardly falls does not stop the method.

`gmres` is Saad and Schultz's GMRES(m) (SIAM J. Sci. Stat. Comput. 7, 1986), the Arnoldi basis built by classical Gram–Schmidt (every coefficient from one pass over the basis in blocks of rows, the combination taken out in a second) with a second such pass whenever the first cancels more than a factor `√2` of the vector (Daniel, Gragg, Kaufman and Stewart's test, with "twice is enough"), and Givens rotations on the Hessenberg matrix. The rotated right-hand side is the true residual norm, since the preconditioner is on the right. When the Krylov space becomes invariant (the lucky breakdown) the iterate is formed from every column found, the last included, which is exact but for rounding and solves the identity in one step. If that last column is itself a combination of the others, the matrix is singular on the space; the column is dropped, the least squares solution taken, and the run ends with `Breakdown` once a cycle cannot improve on it. A cycle that leaves the residual where it was ends the run with `Stagnated` instead of repeating for ever. `max_iter` counts every step, inside a cycle too.

`bicgstab` is van der Vorst's Bi-CGSTAB (SIAM J. Sci. Stat. Comput. 13, 1992), with the direction update `p = r + β·(p − ω·v)`, `M⁻¹` applied to `p` and to `s` before their products, and the early exit when the first half step already passes the test. The three inner products it divides by (`ρ = r̂ᴴ·r`, `r̂ᴴ·A·p̂` and `tᴴ·s`) end the method with `Breakdown` when one is below `ε²` times the norms of its two vectors, Eigen's measure. A small one that is not that small is information: near convergence `ρ` falls faster than the residual. On the damped Helmholtz problem of the tests (`Complex<f64>`) the cosine of `r̂` and `r` falls from about 10⁻¹⁰ to 4·10⁻¹⁴ over the last dozen steps, the smallest at step 108, and the method converges at step 109 (scipy at 116).

The inner products are summed in `f64` for every element type, and the scalar recurrences of `cg` and `bicgstab` (`ρ`, `α`, `β`, `ω`) are carried in `f64` too, with `β` of BiCGSTAB taken from the step lengths as they were applied to `x` and `r`. The inner products that square the scale of the right-hand side are taken of vectors brought near a norm of one by a power of two, which is exact, and that factor is renewed when the residual has come more than `2²⁰⁰` from it. So a right-hand side of any size whose 2-norm the type holds is solved, subnormal ones included (one whose norm overflows, as `3·10³⁸` in each of four elements in `f32`, ends with `NonFinite`), and so is a system in `f32` started from a guess `10³⁰` times its solution, where the scaled products used to underflow `f32` and end `cg` with `NotPositiveDefinite`.

### Preconditioners

| Type | `M` | Exists when | Cost to build | Cost to apply |
| --- | --- | --- | --- | --- |
| `Jacobi::new(&a)` | `diag(A)` | every diagonal entry stored, nonzero and finite | `n` divisions | `n` products |
| `Ilu0::new(&a)` | `L·U` on the pattern of `A` | every diagonal entry stored and every pivot nonzero and finite | the sum over the rows of the lengths of the rows each one uses | the nonzeros of `A` |
| `Ic0::new(&a)`, `Ic0::with_shift(&a, α)` | `L·Lᴴ` on the pattern of the lower triangle of `A` (of `A + α·diag(A)`) | every pivot positive and finite | the same, on the lower triangle | twice the nonzeros of the lower triangle |

Each implements `Preconditioner<T>`, whose `apply(&self, r, z)` writes `z = M⁻¹·r` and allocates nothing; a constructor returns `None` when its factor does not exist, and anything else that maps a residual to a correction can implement the trait. `Ilu0` is Saad's IKJ elimination restricted to the pattern (*Iterative Methods for Sparse Linear Systems*, 2nd ed., algorithm 10.4). Each row is reduced by the rows above it that it has an element under, the place of a column found in a table, not searched for; it leaves `(L·U)ᵢⱼ = aᵢⱼ` on every stored `(i, j)`, and `l()` and `u()` return the factors. `Ic0` computes each element of its factor as the complete Cholesky factorisation would, with the sums taken over the pattern, so that `(L·Lᴴ)ᵢⱼ = aᵢⱼ` on the stored lower triangle; it reads only the lower triangle and the real part of the diagonal. It exists for an M-matrix (Meijerink and van der Vorst, 1977) and a diagonally dominant one, but not for every positive definite matrix: Kershaw's 4 × 4 matrix reaches a last pivot of −5. `with_shift` then factors `A + α·diag(A)` for an `α` large enough (Manteuffel, 1980).

### Example

```rust
use a::math::sparse::{
    Ic0, Ilu0, SolveOptions, SparseMatrix, StopReason, bicgstab_with, cg, cg_with, gmres_with,
    minres,
};

// The 2-D Poisson problem on a 20 x 20 grid: Hermitian positive definite.
let k = 20;
let mut e = Vec::new();
for r in 0..k {
    for c in 0..k {
        let i = r * k + c;
        e.push((i, i, 4.0));
        if c + 1 < k {
            e.extend([(i, i + 1, -1.0), (i + 1, i, -1.0)]);
        }
        if r + 1 < k {
            e.extend([(i, i + k, -1.0), (i + k, i, -1.0)]);
        }
    }
}
let a = SparseMatrix::<f64>::from_coo(k * k, k * k, &e);
let b: Vec<f64> = (0..k * k).map(|i| ((13 * i + 5) % 17) as f64 - 8.0).collect();

// Conjugate gradients, plain and with IC(0): the same answer in fewer than half the iterations
// (70 and 26).
let opts = SolveOptions::new().with_rtol(1e-10);
let plain = cg_with(&a, &b, &opts, None).unwrap();
let ic = Ic0::new(&a).unwrap();
let fast = cg_with(&a, &b, &opts, Some(&ic)).unwrap();
assert!(plain.converged && fast.converged);
assert!(2 * fast.iterations < plain.iterations);
// `residual` is ‖b − A·x‖₂ formed afresh from the x returned, within the tolerance.
let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt();
assert!(fast.residual <= 1e-10 * norm_b);

// Convection and diffusion in one dimension is not symmetric: GMRES(10) or BiCGSTAB, here
// with ILU(0), which on a tridiagonal matrix is the complete factorisation.
let n = 50;
let c = SparseMatrix::tridiagonal(&vec![-1.5; n - 1], &vec![2.0; n], &vec![-0.5; n - 1]);
let f = vec![1.0; n];
let m = Ilu0::new(&c).unwrap();
let g = gmres_with(&c, &f, &SolveOptions::new().with_restart(10), Some(&m)).unwrap();
let s = bicgstab_with(&c, &f, &SolveOptions::new(), Some(&m)).unwrap();
assert!(g.converged && g.iterations == 1 && s.converged && s.iterations == 1);

// A symmetric indefinite matrix is MINRES's, and the conjugate gradient method says so.
let d = SparseMatrix::from_diagonal(&[2.0, -1.0, 3.0]);
assert!(minres(&d, &[2.0, 1.0, 3.0]).unwrap().converged);
assert_eq!(cg(&d, &[2.0, 1.0, 3.0]).unwrap().reason, StopReason::NotPositiveDefinite);
```

### Measured

Everything below comes from `lib/tests/math/sparse/solve.rs` and `precond.rs`. The scipy numbers are in `lib/tests/math/sparse/data/iter_scipy.txt`, made by `data/gen_iter.py` (numpy 2.3.3, scipy 1.18.0; `python3 gen_iter.py` in that directory regenerates it in about ten seconds). The test problems are defined by formulas whose values are exact in `f32` and `f64`, so Rust and Python build the same systems bit for bit. They are 2-D Poisson (12 × 12 and 30 × 30), the same with rows scaled by 2^((i mod 4) − 1), the 1-D Laplacian of 64 from a starting guess, the shifted Laplacian `lap2d − 2.5·I` (symmetric indefinite, 24 negative eigenvalues of 100), a saddle point matrix with a zero block, the singular Neumann Laplacian with a consistent right-hand side, convection–diffusion in 1-D and 2-D, a 1 × 1 system, and, in complex arithmetic, a magnetic Laplacian (Hermitian, a flux of a quarter turn per cell; positive definite with 3 on the diagonal, indefinite with 1), a damped Helmholtz operator (complex symmetric) and a complex convection–diffusion operator, and, in both, a graded diffusion matrix `−(c·u')' + 2⁻¹⁰·u` of 100 points whose coefficients run from 1 to 2⁸ and back every 24 points (condition number about 10⁶; Hermitian with the phases `iⁱ` in complex arithmetic). That makes 216 runs: every method that applies, with no preconditioner, Jacobi, IC(0) and ILU(0), GMRES with restart lengths 1, 5, 10, 30, 50, 100, 144 and 200 (at least `n` among them), from zero and from a starting guess, at `rtol = 10⁻¹⁰` in double and `10⁻⁴` in single precision. scipy's preconditioned GMRES is run on `A·M⁻¹`, since scipy preconditions on the left and this module on the right; its MINRES, where it stops by a test of its own short of the tolerance (on the graded matrix in single precision and in `Complex<f64>`), begins again from its iterate while that helps, as this module's does; and its IC(0) and ILU(0) are a textbook right-looking elimination in the working precision, a different loop order from the Rust code.

- **Iteration counts.** Rounding alone moves a count, so the generator also records the range of scipy's counts when `b` is multiplied by `1 + k·δ`, `k = 1..6`, `δ = 2⁻⁴⁴` (double) or `2⁻²⁰` (single). For CG, MINRES and GMRES that range is at most 2 wide in double precision and 5 in single (MINRES on the indefinite magnetic Laplacian, 79 to 84); for BiCGSTAB on that matrix it is 130 to 186 in `Complex<f64>`. All but three of our counts lie within that range or at most two iterations outside it, and those three within the margins the test allows (in `f64`, BiCGSTAB on the row-scaled Laplacian, 163 against 145 to 159, and GMRES(30) on the shifted Laplacian, 1 142 against 1 146; in `Complex<f32>`, BiCGSTAB on the damped Helmholtz problem, 81 against 72 to 78): the inner products in four running sums and the Gram–Schmidt of the speed pass moved them. Against scipy's count on the unperturbed data the largest differences are: in `f64`, CG 2 (1.1 %), MINRES 1 (1.7 %), GMRES 4 (0.3 %), BiCGSTAB 15 (10 %, the row-scaled Laplacian above); in `f32`, CG 0, MINRES 4 (1.9 %), GMRES 1 (1.7 %), BiCGSTAB 6 (10.7 %, inside scipy's range); in `Complex<f32>`, CG 2 (2.0 %), MINRES 2 (2.4 %), GMRES 0, BiCGSTAB 16 (11 %, on the indefinite magnetic Laplacian, inside scipy's 104 to 146). In `Complex<f64>` it is CG 0, MINRES 1 and GMRES 0, and BiCGSTAB 31 on the indefinite magnetic Laplacian: 155 against scipy's 186, inside scipy's own 130 to 186. In `Complex<f32>` MINRES with Jacobi converges on the graded matrix (115 iterations, 114 from the guess) where scipy, which sums its `float32` inner products in `float32`, does not; the test allows that in single precision only, the residual being checked by a computation of its own. BiCGSTAB is that sensitive to rounding: on the 20 × 20 Laplacian a Python model of the arithmetic of the version before, its inner products summed in order, gave its 55 iterations exactly, and the same model without the fused multiply-adds the textbook's 51; with the inner products in four running sums it takes 54.
- **Solutions.** For every run, `|w·(x − x_scipy)| ≤ ‖w‖·‖A⁻¹‖₂·(‖r‖ + ‖r_scipy‖)` holds for the weights `wᵢ = ((3i + 1) mod 5) − 2`, `‖A⁻¹‖₂` from a dense SVD (of the pseudo-inverse for the singular matrix). Our solution and scipy's are as close as their two residuals allow. Every converged result's residual, formed again in the test with compensated sums, is within the tolerance, and the reported `residual` agrees with it to 10⁻⁵.
- **Preconditioners.** On 2-D Poisson (900 unknowns, `f64`, `rtol = 10⁻¹⁰`) the counts are CG 103, with Jacobi 103 (a constant diagonal changes nothing), with IC(0) 36; MINRES 102 and 36 with IC(0); GMRES(30) 154 and 37 with ILU(0); BiCGSTAB 80 and 28. In `f32` at `10⁻⁴` they are 36/36/11, 33/11, 35/10 and 24/7. On 2-D convection–diffusion (144 unknowns) GMRES(30) takes 52 and 15 with ILU(0), and BiCGSTAB 28 and 9 (`f32`: 25/7, 19/4). On the magnetic Laplacian (`Complex<f64>`) CG takes 56 and 25 with IC(0), GMRES(30) 60 and 25 with ILU(0), and BiCGSTAB 45 and 17. On complex convection–diffusion GMRES(30) takes 35 and 11, and BiCGSTAB 21 and 6. On the row-scaled Laplacian, Jacobi cuts CG from 177 to 45, MINRES from 174 to 45, GMRES(30) from 454 to 60 and BiCGSTAB from 163 to 31. These are scipy's counts with the same preconditioners, to the margins above; `the_preconditioners_cut_the_iterations_on_poisson_and_convection_diffusion` prints them (`cargo test --test math -- --nocapture the_preconditioners_cut`). Restarted GMRES on convection–diffusion takes 190, 77, 52 and 44 steps for `m` = 1, 5, 30 and 144. The defining properties hold on random matrices of the four types, with complex entries off the diagonal: `(L·U)ᵢⱼ = aᵢⱼ` for ILU(0) and `(L·Lᴴ)ᵢⱼ = aᵢⱼ` for IC(0) on every stored element, to `4(n + 2)·ε·(|L|·|U|)ᵢⱼ`, nothing stored outside the pattern, `M⁻¹` Hermitian for IC(0), and IC(0) and ILU(0) of a tridiagonal matrix equal to its Cholesky and LU factors.
- **Preconditioned MINRES on graded diffusion** (an earlier version ended it with `Stagnated` after 17 iterations at 8.6 % of `‖b‖`, by a stall rule that watched the 2-norm of the residual, which preconditioned MINRES does not minimise). On the example `c = e^(7·sin(0.3·i))` on 100 points with Jacobi at `rtol = 10⁻⁶`, MINRES converges in 59 iterations in `f64` and in `Complex<f64>` (the system taken to `U·A·Uᴴ` by a diagonal unitary `U`), scipy in 59; on a single precision one (80 points, `a = 2.5`, `rtol = 10⁻³`) in 65 in `f32` and 64 in `Complex<f32>`, scipy in 65. On the graded matrix of the table, with Jacobi: 100, 105, 101 and 115 iterations in `f64`, `f32`, `Complex<f64>` and `Complex<f32>`, scipy 100, 106 and 101 (and none in `Complex<f32>`).
- **MINRES on singular systems that are not consistent** (an earlier version returned points with residuals of 10⁴ to 10¹⁹). The Laplacians of a path (10, 30 and 100 nodes) and of a grid (6 × 6, 10 × 10, 16 × 16) with free ends, three right-hand sides outside the range each (a ramp on a constant, one summing to 1, `e₀`): MINRES ends with `Breakdown` at the least squares residual `|Σ bᵢ|/√n`, and so it does for each right-hand side moved by a unit or two in the last place (four such per right-hand side), and for twelve exactly singular Hermitian paths with complex elements (the path's Laplacian taken to `U·L·Uᴴ` by a diagonal unitary `U`, its elements Gaussian integers such as `−(3 + 4i)`) with a complex right-hand side: to nine digits in `f64` and `Complex<f64>` and within 0.91 % in `f32` and `Complex<f32>`, in at most 349 and 516 iterations on `aarch64` (370 and 516 on `x86_64`). A complex path whose phases `e^(iθj)` are computed by the platform's sine and cosine is singular only up to the rounding of its elements, a slightly different matrix on every platform; there the method may end with `Breakdown` or at the iteration limit, and its point must be no worse than a least squares solution of the singular matrix it rounds (both `aarch64` and `x86_64` end with `Breakdown`, after 63 and 96 iterations in `Complex<f64>`, at the least squares residual to nine digits). On random perturbations of these systems by up to two units in the last place (6 000 exactly singular ones, of `b` or of the complex paths' right-hand sides, and 5 040 singular only up to rounding, of the matrix's elements or phases), every exactly singular one now ends with `Breakdown` at its least squares residual, where the earlier version ran 0.4 to 25 % of them to the iteration limit depending on type and family; of those singular only up to rounding at most 0.7 % do, against 4 to 23 % before. With Jacobi the least squares solution is one in the norm of `M⁻¹`, and its residual is within 5 % of the least squares one.
- **A random test campaign** (kept outside the crate: 6440 problems of seven classes, all four types, every method and preconditioner, each result checked against an exact residual and a dense solution, and against scipy in the same precision): no dishonest result, no reported residual off the exact one by more than 2·10⁻⁷ relative, no hang, panic or non-finite iterate; one run where scipy converged and this did not (GMRES in `Complex<f32>` at 1.001 times the tolerance), against 20 for the version before. Against that version, 24 MINRES runs converge that did not, and 25 no longer do, all of these at 1.0 to 5.3 times the tolerance, 22 of them in single precision, and none of them solved by scipy either. The later change to MINRES's watch (the claims measured against the smallest residual formed afresh, on the run's evidence of a null vector) alters 52 of the campaign's 2 659 MINRES runs, all in single precision (51 on graded diffusion matrices, one singular): none stops converging, one now converges, ten that ran to the iteration limit end with `Breakdown`, and the residuals of the runs that do not converge move both ways (23 smaller, 24 larger, by at most 2.25 times), with 2 % fewer products in all.
- **Edge cases.** Covered on all four element types: systems of size 0 and 1, `b = 0`, a starting guess that already solves the system (0 iterations, 1 product), the iteration limit (exactly `max_iter` iterations, GMRES's inside a cycle too), NaN and ±∞ in `b`, `A`, the guess and the preconditioner, and a 1 × 1, identity, diagonal or zero matrix and an exact preconditioner for GMRES (each case under a watchdog). Also: singular systems with `b` in the null space or outside the range, the cyclic shift with restart lengths 2 and 3, and a skew-symmetric matrix for BiCGSTAB. Right-hand sides of `10¹⁹`, `10²⁰`, `10⁻¹⁸` to `10⁻³⁰` and the subnormal `10⁻⁴⁰` in `f32` and of `10¹⁶⁰`, `10³⁰⁰`, `10⁻¹⁷⁰`, `10⁻³⁰⁰` and the subnormal `10⁻³¹⁰` in `f64` are all solved, and so is the 10 × 10 Laplacian in `f32` from a guess of `10³⁰`. BiCGSTAB reports the exact breakdowns `ρ = 0` and `ω = 0` of two 3 × 3 matrices with `b = e₀`, as scipy does. On a system no method can solve, none returns a point worse than its start. Each method stops at the first iterate that passes the test, at thirty tolerances in double precision, and GMRES over the whole space needs 46 iterations at `rtol = 10⁻¹²` on a diagonal matrix of condition 10⁶ and 44 on the bidiagonal one, in `f64` and in `Complex<f64>` (43 and 44 with the modified Gram–Schmidt of the version before; 69 and 62 with one pass of it). `rtol = atol = 0` ends with `Stagnated` or `Converged` and the point kept, not with an iteration limit or a lost iterate, on the default budget (with `max_iter` set to `usize::MAX`, BiCGSTAB on the graded matrix in `f32` ended with `NonFinite` after 8,528 steps, its best point kept, in a review). GMRES with a restart length of the whole space on 10⁵ unknowns takes memory for the dozen steps it runs.

**Time** (conditions as under [Storage and operations](#storage-and-operations)), to `rtol = 10⁻⁸`, every method taking the iterations of the reference within a few (the counts are in the scoreboard):

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `cg_with` (none) | lap2d-316, n = 99856, nnz = 498016 | 282 ms | SciPy sparse.linalg.cg | 275 ms | 1.02 |
| `cg_with` (Jacobi) | lap2d-316, n = 99856, nnz = 498016 | 305 ms | SciPy sparse.linalg.cg | 301 ms | 1.01 |
| `cg_with` (none) | lap2d-1000, n = 1000000, nnz = 4996000 | 8.78 s | SciPy sparse.linalg.cg | 8.55 s | 1.03 |
| `cg_with` (Jacobi) | lap2d-1000, n = 1000000, nnz = 4996000 | 9.55 s | SciPy sparse.linalg.cg | 9.41 s | 1.02 |
| `cg_with` (none) | lap3d-46, n = 97336, nnz = 668656 | 57.5 ms | SciPy sparse.linalg.cg | 57.9 ms | 0.992 |
| `cg_with` (Jacobi) | lap3d-46, n = 97336, nnz = 668656 | 60.7 ms | SciPy sparse.linalg.cg | 62.2 ms | 0.975 |
| `cg_with` (none) | lap3d-100, n = 1000000, nnz = 6940000 | 1.28 s | SciPy sparse.linalg.cg | 1.28 s | 1.01 |
| `cg_with` (Jacobi) | lap3d-100, n = 1000000, nnz = 6940000 | 1.35 s | SciPy sparse.linalg.cg | 1.36 s | 0.992 |
| `bicgstab_with` (none) | upwind2d-316, n = 99856, nnz = 498016 | 332 ms | SciPy sparse.linalg.bicgstab | 333 ms | 0.997 |
| `gmres_with` (none) | upwind2d-316, n = 99856, nnz = 498016 | 1.65 s | Accelerate SparseSolve GMRES(30) (none) | 372 ms | 4.44 |
| `bicgstab_with` (Jacobi) | upwind2d-316, n = 99856, nnz = 498016 | 348 ms | SciPy sparse.linalg.bicgstab | 358 ms | 0.971 |
| `gmres_with` (Jacobi) | upwind2d-316, n = 99856, nnz = 498016 | 1.67 s | Accelerate SparseSolve GMRES(30) (Jacobi) | 401 ms | 4.18 |
| `bicgstab_with` (none) | upwind2d-1000, n = 1000000, nnz = 4996000 | 10.2 s | SciPy sparse.linalg.bicgstab | 10.1 s | 1.01 |
| `gmres_with` (none) | upwind2d-1000, n = 1000000, nnz = 4996000 | 41.0 s | Accelerate SparseSolve GMRES(30) (none) | 10.8 s | 3.82 |
| `bicgstab_with` (Jacobi) | upwind2d-1000, n = 1000000, nnz = 4996000 | 10.5 s | SciPy sparse.linalg.bicgstab | 10.8 s | 0.975 |
| `gmres_with` (Jacobi) | upwind2d-1000, n = 1000000, nnz = 4996000 | 40.7 s | Accelerate SparseSolve GMRES(30) (Jacobi) | 11.1 s | 3.65 |
| `bicgstab_with` (none) | convdiff3d-46, n = 97336, nnz = 668656 | 74.8 ms | SciPy sparse.linalg.bicgstab | 74.2 ms | 1.01 |
| `gmres_with` (none) | convdiff3d-46, n = 97336, nnz = 668656 | 345 ms | Accelerate SparseSolve GMRES(30) (none) | 96.7 ms | 3.57 |
| `bicgstab_with` (Jacobi) | convdiff3d-46, n = 97336, nnz = 668656 | 76.8 ms | SciPy sparse.linalg.bicgstab | 78.5 ms | 0.978 |
| `gmres_with` (Jacobi) | convdiff3d-46, n = 97336, nnz = 668656 | 350 ms | Accelerate SparseSolve GMRES(30) (Jacobi) | 102 ms | 3.43 |
| `minres_with` (none) | lap2d-316 − 0.5·I, n = 99856, nnz = 498016 | 8.02 s | SciPy sparse.linalg.minres | 9.41 s | 0.852 |
| `minres_with` (Jacobi) | lap2d-316 − 0.5·I, n = 99856, nnz = 498016 | 9.23 s | SciPy sparse.linalg.minres | 10.5 s | 0.879 |
| `minres_with` (none) | lap3d-46 − 0.5·I, n = 97336, nnz = 668656 | 899 ms | SciPy sparse.linalg.minres | 1.12 s | 0.803 |
| `minres_with` (Jacobi) | lap3d-46 − 0.5·I, n = 97336, nnz = 668656 | 1.01 s | SciPy sparse.linalg.minres | 1.16 s | 0.873 |

GMRES(30) takes 3.43 to 4.44 times Accelerate's time (1.67 to 2.09 times SciPy's), with the same products: its Gram–Schmidt, one pass over the basis for the coefficients and one for the combination (both twice when the first cancels), runs on the crate's own kernels against Accelerate's and NumPy's BLAS; what else makes up the gap was not measured. CG and BiCGSTAB are within 0.971 to 1.03 of SciPy's, MINRES takes 0.803 to 0.879 of its time.

### Limits

- The methods are not robust against every matrix they accept. Restarted GMRES can stagnate on an indefinite or strongly non-normal matrix (GMRES(30) needs 1142 steps on the shifted Laplacian in `f64`, scipy 1146, and GMRES(100) 57), and BiCGSTAB's count on an indefinite matrix varies by a third with the rounding. `gmres` reports `Stagnated` rather than `IterationLimit` when `max_iter` cuts its first cycle short before the residual has fallen (the cyclic shift of order 10 with `max_iter = 9`). The conjugate gradient method recognises an indefinite matrix only when a direction of non-positive curvature turns up; on a non-Hermitian one it runs to its iteration limit, reporting it honestly.
- MINRES on a singular system that is not consistent returns a least squares solution, not the one of least norm (that is MINRES-QLP's), and may need a second Lanczos process to be sure of it. It can take many steps more than it needs to stop: on the Laplacian of a random graph of two components (400 unknowns, `rtol = 10⁻¹²`, in a review) it had reached the least squares residual by step 50 and ended with `Breakdown` after 3,823, just inside the default budget of 4,000, where scipy stops at 52. Its reason, `Breakdown` or `Stagnated`, rests on the size of `‖A·r‖` at the points it verified: an ill-conditioned matrix that is not singular can show a residual that is a null vector to `√ε` and end with `Breakdown` once its restarts stop helping, where the tolerance is out of reach (a review saw it at condition numbers of 10⁸ to 10¹² in double precision and 10⁴ to 10⁵ in single, where scipy did not reach the tolerance either); and a matrix singular to within a few `ε`, as `diag(1, 2·10⁻⁷)` in `f32`, can end with `Breakdown` at the first step, its residual 0.71 of `‖b‖` where the next step would have brought it to 0.05. A matrix that is singular only up to the rounding of its elements may also run to the iteration limit, with the best point it verified.
- MINRES's attainable accuracy is about `ε·κ·‖b‖` (Sleijpen, van der Vorst and Modersitzki, SIAM J. Matrix Anal. Appl. 22, 2001), worse than that of CG, and in single precision its recurrence can lose track of the residual on an ill-conditioned matrix long before convergence. The restarts from the iterate recover much of that, but on a positive definite matrix `cg` is the method to use.
- A residual formed by an operator known only through its products (`FnOperator`) is taken in the working precision and carries its rounding error, about `ε·‖|A|·|x| + |b|‖₂`. A tolerance below that cannot be verified.
- The inner products are robust to the scale of the right-hand side, not to that of the matrix: a matrix whose squared entries leave the range of the type (`10¹⁵⁴` in `f64`, `10¹⁹` in `f32`) can make BiCGSTAB's `tᴴ·t` overflow, which ends in `NonFinite`.
- `Ilu0` does not pivot and can be unstable where elimination without pivoting would be. `Ic0` breaks down on some positive definite matrices, and the shift has to be found by retrying. The cost of ILU(0) is `Σ (row length × lengths of the rows it uses)`, which a dense row of `U` makes quadratic.

## Eigenvalues

`lanczos` and `arnoldi` find a few eigenvalues of a large matrix, and their eigenvectors, from products with the matrix alone. They take any `LinearOperator` — a `SparseMatrix`, a dense `Matrix`, or a function in an `FnOperator`, so they also run matrix-free. Both are the implicitly restarted methods of ARPACK: Sorensen, *Implicit application of polynomial filters in a k-step Arnoldi method* (SIAM J. Matrix Anal. Appl. 13, 1992), as Lehoucq, Sorensen and Yang's *ARPACK Users' Guide* (SIAM, 1998) lays them out. The code is the crate's own.

| Function | Operator | Selections | Result |
| --- | --- | --- | --- |
| `lanczos`, `lanczos_with` | Hermitian: real symmetric or complex Hermitian | `LanczosWhich`: `LargestAlgebraic`, `SmallestAlgebraic`, `LargestMagnitude`, `BothEnds` | `LanczosResult`: real eigenvalues, ascending; orthonormal eigenvectors |
| `arnoldi`, `arnoldi_with` | any square one | `ArnoldiWhich`: `LargestMagnitude`, `LargestReal`, `SmallestReal`, `LargestImaginary`, `SmallestImaginary` | `ArnoldiResult`: complex eigenvalues, the most wanted first; eigenvectors of length 1 |
| `lanczos_shift_invert`, `lanczos_shift_invert_with` | a Hermitian `SparseMatrix` and a real target `σ` | the nearest `σ`, on either side | `LanczosResult` |
| `arnoldi_shift_invert`, `arnoldi_shift_invert_with` | a square `SparseMatrix` and a target of its element type | the nearest `σ` | `ArnoldiResult` |
| `lanczos_inverse_with`, `arnoldi_inverse_with` | an operator `A` and an operator that applies `(A − σI)⁻¹` | the nearest `σ` | as above |

`EigenOptions` holds the size `m` of the basis (`max(2k + 1, 20)` by default, raised to `k + 1` for Lanczos and `k + 2` for Arnoldi if it is smaller, lowered to `n`), the tolerance (`ε` of the type), the budget of restarts of the whole call (300) and the starting vector, by default one from a fixed seed of a xorshift generator, so that a call is repeatable and asks nothing of the system. A result carries the eigenvalues, the eigenvectors as the columns of a `Matrix`, the residual `‖A·v − λ·v‖` of every pair from a product with the operator after the iteration, how many pairs `converged`, and the `restarts` and operator `applications` it took.

**Lanczos.** The method keeps an `m`-step factorisation `A·V = V·T + f·eₘᵀ`, `T` real, symmetric and tridiagonal, for a complex Hermitian operator too. Every new vector is orthogonalised against the whole basis by classical Gram–Schmidt applied twice, with a third pass when the second still removes more than `1 − 1/√2` of it (Daniel, Gragg, Kaufman and Stewart): the basis stays orthonormal to working precision, where the three-term recurrence alone loses orthogonality as soon as a Ritz value converges. The Ritz values are the eigenvalues of `T`, from the crate's dense Hermitian solver. A pair has converged when its estimate `‖f‖·|sₘ| ≤ tol·max(ε^(2/3)·‖A‖, |θ|)`: ARPACK's test, with its floor made relative to the size of the operator (estimated by the largest `‖A·v‖` and `|θ|` met), so that the test is the same for `A` and `c·A` — or when the estimate is at most `ε·‖A‖`, the rounding of the operator's own products, below which no residual can be told from zero; that matters for a tolerance below `ε^(1/3)`. A restart applies the unwanted Ritz values as exact shifts, by implicit QR steps on `T` and the same rotations on `V`, and keeps the leading columns: a shorter factorisation, of the starting vector filtered by the polynomial with the shifts as roots, which is then extended back to `m` steps. It keeps more than `k` Ritz values — `k + min(converged, shifts/2)`, ARPACK's safeguard against stagnation, fewer when a conjugate pair at the edge of the count would leave no shift. Converged pairs are retained by the exact-shift strategy: they are never shifts. When an element of the subdiagonal of `T` is zero — the factorisation broke down and went on from a fresh random vector, or the QR test `|β| ≤ ε·√(|αᵢ|·|αᵢ₊₁|)` finds it negligible — the block above it spans an invariant subspace, and the method *locks* it: its vectors leave the factorisation, which goes on with all `m` columns in their complement, and its eigenvalues, exact as far as the arithmetic can tell, stand beside the Ritz values in every decision. An operator with few distinct eigenvalues breaks down again and again — a Krylov space of the graph Laplacian of the hypercube `Q₈`, of 9 distinct eigenvalues, holds 9 vectors — and kept in the basis such blocks would fill it with values that no shift can move.

**Arnoldi.** The factorisation is `A·V = V·H + f·eₘᵀ` with `H` upper Hessenberg, the Ritz values and vectors come from the crate's dense eigensolver (the Schur form of `H`, then back substitution), the convergence test is the same, and so are the restart, by implicitly shifted QR steps on `H` — the Schur form is used for the Ritz values only, never for the restart — and the locking of a block that splits off. For a real operator the factorisation stays real. Its complex Ritz values come in conjugate pairs; a pair is wanted, kept, shifted and returned as a whole (when the `k`-th value is one of a pair, `k + 1` come back), and a pair of shifts is one Francis double-shift step in real arithmetic, as ARPACK's `dnapps` does, its first column formed divided by `|h₀₀ − Re μ| + |Im μ| + |h₁₀|` as LAPACK's `dlahqr` forms it, so that it neither overflows nor underflows for any operator whose elements the type can hold. Carrying a real operator in complex arithmetic would cost four times the work on the basis and twice the memory, and rounding would no longer keep the conjugate of a Ritz value a Ritz value. For a complex operator every shift is one single-shift step with complex rotations. Every small matrix is divided by a power of two before the dense eigensolver sees it, which is exact: the solver's absolute thresholds near the underflow level lost pairs of operators of norm below about `10⁻²⁹⁰`. `LargestImaginary` and `SmallestImaginary` rank the eigenvalues of a real operator by `|Im λ|`, ARPACK's convention for a spectrum symmetric about the real axis, and those of a complex one by `Im λ`.

**Multiple eigenvalues.** The Krylov space of one starting vector holds only one vector of each eigenspace, so a run sees one copy of an eigenvalue of multiplicity two or more at first. The other copies come from two sources within the run. Rounding, which the full reorthogonalisation keeps at the level of `ε`, seeds the directions of the other copies, and the restarts grow them: a run usually converges to every copy its basis has room for — on the 2-D Laplacian of 12 × 12 below, the first run alone finds both copies of each double eigenvalue (the test passes with the search for copies turned off). And when the operator has few distinct eigenvalues, breakdowns: the Krylov space is then an invariant subspace, the run locks it and goes on from a fresh random vector of its complement, which holds the next copy. To catch what a run missed, the methods then lock what it found and start another run from a fresh random vector, the operator restricted to the complement, until a run finds nothing more wanted than the least wanted eigenvalue found: such a run stops as soon as, at every end of the spectrum the selection takes from, its Ritz values have converged inwards up to one that is not wanted more — for a Hermitian operator, as soon as a Ritz value lies four of its residual estimates short of that line. These runs find the copies a run had no room for — on the diagonal of 1, 2, 3 and 5, each 30 times, the first run returns five copies of 5 and a 3 as the six largest, and the next replaces the 3 — and an extreme eigenvalue a run passed over; they rest on the premise every Krylov method rests on, that the most wanted Ritz value of a run from a random vector approximates the most wanted eigenvalue there is. For `LargestMagnitude` of a Hermitian operator they watch both ends of the spectrum.

**The answer.** Every restart replaces the basis by `V·Q`, and its rounding moves the lengths and angles of the columns kept by an `ε` or so each time, a drift that adds up: a block locked, or Ritz vectors `V·s` taken as they were, were up to `150ε` from orthonormal after a few hundred restarts and `1000ε` after a few thousand, and so were the eigenvectors made from them. So what a run locks and the Ritz vectors it ends with are made orthonormal again by Gram–Schmidt, which moves them by no more than they had drifted, before they join the vectors found. The vectors of all the runs go through a Rayleigh–Ritz step with the operator — the eigenpairs of `Zᴴ·A·Z` taken back by `Z`, which also turns the locked blocks and the vectors found in a complement into eigenvectors of `A`, of length 1 when `Z` is orthonormal — and each pair is checked by its true residual, from those products, against four times the tolerance of the convergence test plus a room for rounding of `64·√w·ε·‖A‖`, `w` the larger of `n` and `m·(restarts + 1)`. The answer is, at each end of the selection, the pairs from the most wanted inward up to the first that had not converged or does not pass: an eigenvalue less wanted than one not found is never reported. So `converged < k` says that the budget of restarts ran out before every pair asked for had converged — or before the search for copies had confirmed that none was missed —, that a shift-and-invert run stopped with restarts left because no restart could move the pairs still open, that the operator gave values that are not finite, or that a pair failed its check, and the pairs reported are then the most wanted ones. An operator handed to `lanczos` that is not Hermitian, which the method cannot see, gives pairs that fail. `k ≥ n` is taken as `n`: the basis is then the whole space and one factorisation finds every eigenvalue. `k = 0` and an empty operator give an empty result; a tolerance that is not a positive finite number, or a starting vector that is zero or not finite, gives `None`.

**Shift-and-invert.** The eigenvalues of `A` nearest `σ` belong to the largest eigenvalues `θ = 1/(λ − σ)` of `(A − σI)⁻¹`, which are well apart where those of `A` near `σ` are not. `lanczos_shift_invert` and `arnoldi_shift_invert` run the methods with `LargestMagnitude` on that operator, applied by a `SparseLu` of `A − σI`, and give `None` when it is singular — `σ` an eigenvalue to working precision. `lanczos_inverse_with` and `arnoldi_inverse_with` take the inverse as an operator of the caller's own: a Cholesky or LU factorisation of their choice, or an iterative solver run tightly enough. A pair of the inverse is judged by its residual with `A` itself: its eigenvalue is the Rayleigh quotient `λ = xᴴ·A·x`, and it has converged when `‖A·x − λ·x‖ ≤ tol·max(|λ|, ε^(2/3)·‖A‖)` with room for the rounding of that product, `8·√n·ε·‖A‖` — `‖A‖` bounded by `√(‖A‖₁·‖A‖∞)` for a `SparseMatrix`, estimated from below by a few steps of the power method otherwise. A test on the inverse would be scaled by its size `1/δ`, `δ` the distance of `σ` from an eigenvalue, and pass the other pairs with residuals with `A` that grow as `ε/δ`. In a run that works in the complement of vectors found before, the residual is that of the operator restricted to the complement: for an `A` that is not normal, its eigenvectors there are Schur vectors, which the final step turns into eigenvectors. A converged Ritz value eight times or more larger than the open ones — `1/δ` next to values of the order of one — is locked at once and the run starts again from the open Ritz vectors in its complement: kept in the factorisation, the rotations of every restart would mix it into them with errors of the order of `ε` times itself. In shift-and-invert only exact zeros split the factorisation, and a block that splits off at one is locked only if every pair of it passes the test with `A`: near an eigenvalue the inverse as computed has subspaces that are invariant to its rounding and are no invariant subspaces of `A`, and a block refused stays in the factorisation, judged at every decision like the rest. The Rayleigh–Ritz step at the end is made with `A`, whose small matrix is well scaled where that of the inverse is not; the pairs are ranked by their Rayleigh quotients with the inverse, which needs no `σ`. A run stops, with restarts left, once the pairs asked for that are still open have converged on the inverse to the rounding of their own size and nothing in the factorisation is large enough to spoil them: no restart can move them further. Every pair reported is an eigenpair of `A` to the tolerance, whatever the inverse does; that they are the nearest `σ` rests on the inverse applying `(A − σI)⁻¹`. `‖A‖` of a `SparseMatrix` is formed from its elements divided by the largest, so that it neither overflows nor underflows for any matrix the type can hold.

For complex elements the shift-and-invert Lanczos forms carry an Arnoldi factorisation, with `H` full, and keep the search and the answer of a Hermitian operator, so that the eigenvalues are real and the vectors orthonormal. The solves of a complex Hermitian `A − σI` are not Hermitian: the rounding of the small pivot of a nearly singular `A − σI` is a complex number, and the largest eigenvalue of the computed inverse has an imaginary part of relative size about `ε·‖A‖/δ` (measured: 0.005 to 0.13 times it, for `δ` from `10⁻⁴` to `10⁻¹⁰·‖A‖`), which the real tridiagonal `T` of a Lanczos factorisation has no room for — with it, not even the nearest pair passed the test with `A` once `δ` was below about `10⁻⁵·‖A‖`. In real arithmetic the rounding of a pivot is real, the solves stay symmetric to a few `ε`, and the real forms keep the Lanczos factorisation.

Near an eigenvalue of a matrix that is not normal, the rounding of the solves limits the pairs other than the nearest, in every precision. A vector orthogonal to the eigenvector nearest `σ` still has a component along its left eigenvector, which differs from it, so a solve gives it a component `1/δ` times larger along the nearest eigenvector, and the rounding of that component leaves the pair `j` with a residual with `A` of about `ε·‖A‖·|λⱼ − σ|/δ`: no method that applies the inverse in floating point removes it, and ARPACK's pairs carry it too. Once that is above the room of the test, from about `δ = 10⁻³·‖A‖` on, only the nearest pair comes back at the default tolerance, and `converged < k`. A tolerance of `10·ε·‖A‖/δ`, with `δ = |λ₁ − σ|` from the nearest pair of a first call, brings the others back. A normal matrix — a Hermitian one among them — has no such limit.

### Example

```rust
use a::math::sparse::{LanczosWhich, SparseMatrix, lanczos, lanczos_shift_invert};

// The 2-D Laplacian of a 10 × 10 grid, eigenvalues 4 − 2·cos(iπ/11) − 2·cos(jπ/11): the
// second is double, λ(1, 2) = λ(2, 1).
let n = 10;
let mut entries: Vec<(usize, usize, f64)> = Vec::new();
for y in 0..n {
    for x in 0..n {
        let p = x + y * n;
        entries.push((p, p, 4.0));
        if x > 0 { entries.push((p, p - 1, -1.0)); }
        if x + 1 < n { entries.push((p, p + 1, -1.0)); }
        if y > 0 { entries.push((p, p - n, -1.0)); }
        if y + 1 < n { entries.push((p, p + n, -1.0)); }
    }
}
let a = SparseMatrix::from_coo(n * n, n * n, &entries);
let h = core::f64::consts::PI / (n + 1) as f64;
let mode = |i: f64, j: f64| 4.0 - 2.0 * (i * h).cos() - 2.0 * (j * h).cos();

// The three smallest: both copies of the double eigenvalue.
let r = lanczos(&a, 3, LanczosWhich::SmallestAlgebraic).unwrap();
assert_eq!(r.converged, 3);
for (&value, exact) in r.values.iter().zip([mode(1.0, 1.0), mode(1.0, 2.0), mode(2.0, 1.0)]) {
    assert!((value - exact).abs() < 1e-12);
}
assert!(r.residuals.iter().all(|&x| x < 1e-12));

// The same three by shift-and-invert around 0, in far fewer products.
let s = lanczos_shift_invert(&a, 0.0, 3).unwrap();
assert_eq!(s.converged, 3);
assert!((s.values[1] - mode(1.0, 2.0)).abs() < 1e-12);
assert!(s.applications < r.applications);
```

### Measured

Every number below is from `lib/tests/math/sparse/eigen.rs` or from a probe run on its matrices. The reference values are ARPACK's through SciPy 1.18 (`eigsh`, `eigs`, `tol = 0`), each checked against LAPACK through numpy, and are regenerated by `lib/tests/math/sparse/data/gen_eigen.py`, which builds the same matrices from the same generator as the tests. The tests check, for every pair reported and all four element types, the true residual recomputed in the test against `tol·‖A‖` with a room of `64·√n·ε·‖A‖` (`‖A‖` bounded by `√(‖A‖₁·‖A‖∞)`), the agreement with the reference, the orthonormality of Lanczos vectors and that a real matrix's conjugate pairs come together.

Accuracy, the largest distance of an eigenvalue from the reference (the eigenvalues are of size 1 to 40), with the tolerance `10⁻¹²` in `f64` and `Complex<f64>` and the default `ε` in `f32` and `Complex<f32>`, the default basis unless stated:

| Case | `f64` | `Complex<f64>` | `f32` | `Complex<f32>` |
| --- | --- | --- | --- | --- |
| 1-D Laplacian of 100, 4 largest / 4 smallest, against the closed form | 1.3·10⁻¹⁵ / 9.9·10⁻¹⁷ | 3.6·10⁻¹⁵ / 8.7·10⁻¹⁷ | 7.3·10⁻⁷ / 2.0·10⁻⁹ | 9.6·10⁻⁷ / 2.1·10⁻⁹ |
| 2-D Laplacian of 12 × 12, 8 largest / 8 smallest, three of them double | 8.0·10⁻¹⁵ / 4.4·10⁻¹⁶ | 8.9·10⁻¹⁵ / 4.4·10⁻¹⁶ | 4.3·10⁻⁶ / 2.6·10⁻⁷ | 5.6·10⁻⁶ / 4.5·10⁻⁷ |
| random sparse symmetric (150) or Hermitian (120), 5 of each of the four selections, against ARPACK | 2.3·10⁻¹⁴ | 4.4·10⁻¹⁴ | 2.9·10⁻⁶ | 4.3·10⁻⁶ |
| diagonal of 200 with clusters 10⁻³, 10⁻⁶ and 10⁻⁸ apart and a triple 40, 6 at each end, basis 30 | 4.3·10⁻¹⁴ | 2.8·10⁻¹⁴ | 1.1·10⁻⁵ | 1.5·10⁻⁵ |
| random sparse unsymmetric, real (150) or complex (120), 6 of every selection, against LAPACK | 1.4·10⁻¹³ | 3.8·10⁻¹³ | 2.5·10⁻⁶ | 4.3·10⁻⁶ |
| real, 50 rotation blocks, all eigenvalues complex pairs, LM / LR / SR / LI | 1.4·10⁻¹⁴ | — | 2.2·10⁻⁶ | — |
| shift-and-invert, dense LU: 2-D Laplacian near 2.5, two double eigenvalues among the six / unsymmetric near 0.2 (real) and 0.1 + 0.2i (complex), 4 | 3.6·10⁻¹⁵ / 6.7·10⁻¹⁵ | 3.6·10⁻¹⁵ / 2.0·10⁻¹³ | 1.1·10⁻⁶ / 1.1·10⁻⁷ | 2.3·10⁻⁶ / 3.1·10⁻⁷ |

With the default options the clusters come out as well: 7.1·10⁻¹⁵ in `f64`, 5.0·10⁻¹⁴ in `Complex<f64>`, 7.6·10⁻⁶ in `f32` and in `Complex<f32>`. The eigenvectors of the answer are of length 1, and those of Lanczos orthogonal, to within `5ε`, measured with exact inner products, in every case tried, up to 4,400 restarts: the 1-D Laplacians of 200 to 1,000 points with bases of 8 to 12, and rotation matrices of 100 to 1,000 rows; on x86_64 the cases of the test, within `3ε`. Taken as they were, what the runs locked and ended with had left them up to `980ε` from length 1 and `470ε` from orthogonal. The largest true residual of a reported pair of these cases was 3.6·10⁻¹² in `f64` at the tolerance `10⁻¹²` and 3.8·10⁻⁵ in single precision. Across the tolerances `10⁻¹²` to `10⁻²` in double precision and `10⁻⁵` to `10⁻²` in single, on random matrices of every type (600 calls of each method a type, about half of them at such tolerances), the true residual of a reported pair stayed below `tol·‖A‖₂`, at most 0.98 of it; at `10⁻⁶` in single precision, eight times `ε`, it reached `3.3·tol·‖A‖₂`: there the rounding of the factorisation, which the check after the iteration allows for, is the larger term.

Multiple eigenvalues, all four types, the default options: the hypercube `Q₈` gives 16 and three copies of 14 as its four largest, and four copies for five; the diagonal of 1, 2, 3 and 5 gives `k` copies of 5 as its `k` largest for `k` from 6 to 10; five copies of a random Hermitian block give five copies of its largest and of its smallest eigenvalue; the 3-D Laplacians of 6³ (`k` = 4, 7, 10) and of 3³ (`k` = 9, 10) every copy at both ends, through both methods. A starting vector in the 18-dimensional invariant subspace of the first unit vectors of the diagonal 1..100 gives 98, 99 and 100, where ARPACK stops with "No shifts could be applied". A probe of 1,217 calls a type — complete graphs, cycles, stars, triangular graphs, hypercubes up to `Q₁₀`, 3-D Laplacians, a Kronecker sum, copies of blocks, diagonals of few values, every selection, `k` from 1 to 12 — found no wrong result and no call that stopped with restarts left; three calls came back incomplete after their whole budget of 300 restarts, on the cycle `C₂₅₆`, whose eigenvalues at the ends lie `1.5·10⁻⁴` apart, relative: `BothEnds` with `k = 2` in `Complex<f64>` (`f64` converges there), and `LargestMagnitude` with `k = 10` in `f32` and `Complex<f32>`, with 9 of the 10 (318 and 409 restarts complete them).

Scaling, all four types: Arnoldi on `c·A` gives what it gives on `A` for `c` from `10⁻²⁸⁰` to `10²⁸⁰` in `f64` and `10⁻³⁰` to `10³⁰` in `f32`, on the rotation matrix and a random unsymmetric one, by the largest absolute value, real and imaginary part; at `10⁻³⁰⁰` and `10⁻³⁵`, where the rounding of a residual is a subnormal number, the most wanted eigenvalue still comes out. Lanczos on `c·A` likewise. The `SparseMatrix` shift-and-invert drivers, through both methods, give `c` times what they give for `c = 1` on a diagonal of 200 (target 100.37, `k = 6`, a basis of 7) and the 1-D Laplacian of 300 (target 0, `k = 4`), at `c = 10¹⁶⁰` and `10⁻¹⁷⁰` in `f64` and `10²⁰` and `10⁻²⁵` in `f32`, every pair within the test it converged by.

Shift-and-invert near an eigenvalue, the diagonal 1..60 and the targets 7 + 1 ulp, 7 + 10⁻⁹ and 7 + 10⁻⁶, through both `SparseMatrix` drivers and both inverse forms with the dense LU, all four types: 6, 7 and 8 every time, the largest residual `1.1·ε·‖A‖`; a target that rounds to 7 in the type gives `None`. ARPACK finds the same.

Complex Hermitian matrices near an eigenvalue: on 20 random sparse Hermitian (or, for the real types, symmetric) matrices of order 100 a cell, targets `10⁻²` to `10⁻¹²·‖A‖` from an eigenvalue, `k = 1` and `3`, through `lanczos_inverse_with` and the dense LU, every call came back complete in all four types, every pair among the nearest, with residuals within 0.93 of the room of the test; with the Lanczos factorisation the complex calls had come back empty from `10⁻⁵·‖A‖` on (4 and 8 of 20 at `10⁻⁵`, none nearer, in `Complex<f64>` and `Complex<f32>`). Through `lanczos_shift_invert` with `SparseLu`, the same in `f64` and `Complex<f64>`; in single precision that factorisation calls `A − σI` singular, and the driver gives `None`, for targets within some tens of ulps of the eigenvalue (2 and 9 calls of 20 at `10⁻⁶·‖A‖`, every one nearer). On the matrix of the tests, `hermitian(11, 120, 3)`, the three nearest come back at `10⁻⁶` and `10⁻⁹·‖A‖` in `Complex<f64>`, as ARPACK finds them in double precision, and at `10⁻⁵·‖A‖` in `Complex<f32>`.

Matrices that are not normal near an eigenvalue: on 20 random sparse unsymmetric matrices of order 80 a cell (a real target next to a real eigenvalue, or a complex one next to any eigenvalue of a complex matrix), `k = 3` and the default tolerance, the calls came back complete 18, 9, 0, 0 and 0 times at `δ = 10⁻³, 10⁻⁴, 10⁻⁵, 10⁻⁶` and `10⁻⁸·‖A‖` in `f64`, 15, 3, 0, 0, 0 in `Complex<f64>`, 20, 10, 0, 0, 0 in `f32` and 19, 3, 0, 0, 0 in `Complex<f32>`, the nearest eigenvalue first, and right, in every call and no pair wrong; the pairs other than the nearest that came back had residuals up to twice `ε·‖A‖·|λⱼ − σ|/δ`. With the tolerance `10·ε·‖A‖/δ` every call came back complete, for `δ` from `10⁻³` to `10⁻⁸·‖A‖` in double precision and to `10⁻⁵·‖A‖` in single, where that tolerance is about `0.12`, every residual within it. Such a call now ends once the pairs still open have converged on the inverse: with `k = 1`, 0.3 restarts and 46 solves on average at `δ = 10⁻⁶·‖A‖` in `f64`, where without that stop it spent its 300 restarts and 3,400 solves on confirming a pair it had found at the first.

On random matrices of each type and kind with interior targets and the dense LU (100 calls a type), every pair reported was among the nearest and passed with `A`.

The work, in restarts over all the runs of a call — the check for missed copies included — against ARPACK's iterations, one restart each, on the same matrix with the same basis of 20 and the default tolerance `ε` (`dsaupd`/`dnaupd` through SciPy, which make no such check, from another starting vector): the 1-D Laplacian's 4 largest 29 (ARPACK 26), 4 smallest 27 (32); the 2-D Laplacian's 8 largest 16 (18), 8 smallest 17 (17); the random symmetric matrix's 5 of largest magnitude 11 (9); Arnoldi on the random unsymmetric matrix, 6 of largest magnitude 38 (28) and of largest real part 27 (17), on the complex one 42 (35), on the rotations 35 (14) and 67 (67). The smallest four of the 1-D Laplacian of 400 take plain Lanczos 213 restarts and 3,429 products; shift-and-invert around 0 takes none, 44 solves.

The 6 of largest real part of the rotation matrix lie inside the convex hull of its spectrum, which a Krylov method converges to slowly: from 20 starting vectors of the tests' xorshift generator (seeds `7919·s + 1`) this method found the right six every time (in 54 restarts on average, 48 to 69), and ARPACK, from the same vectors, 18 times — twice it converged to wrong eigenvalues.

A release build on the 2-D Laplacian of 316 × 316 = 99,856 unknowns (498,016 nonzeros), the 6 largest, two of them double, by a small program outside the crate (its times, taken on a machine shared with other jobs before the speed pass of the release, are not given):

| Method, basis, tolerance | restarts | products | eigenvalues from the closed form | residuals |
| --- | --- | --- | --- | --- |
| `lanczos_with`, 40, 10⁻¹⁰ | 179 | 5,985 | 1.4·10⁻¹³ | 6.5·10⁻¹⁰ |
| `lanczos_with`, 60, `ε` | 94 | 5,070 | 1.3·10⁻¹³ | 1.3·10⁻¹³ |
| `arnoldi_with`, `LargestReal`, 40, 10⁻¹⁰ | 208 | 6,991 | 5.9·10⁻¹⁴ | 6.9·10⁻¹⁰ |
| `lanczos`, the defaults: 20, `ε`, 300 restarts | 300 | 4,220 | none converged | — |

Over a hundred restarts and more about six eigenvalues `8·10⁻⁴` apart, the path of the iteration follows the rounding: a change in the last digits of the arithmetic — the residual kept as its direction and its length instead of as one vector — moved these counts by up to a third (from 140, 98 and 178 restarts), where on the matrices of the tests it moved none of the counts given above.

With the default options — a basis of 20, the tolerance `ε` and 300 restarts — the call converges none of the six: they lie within `8·10⁻⁴` of each other at the end of a spectrum of width 8, and a basis of 20 closes in on them too slowly for the budget; ARPACK with a basis of 20, `tol = 0` and 300 iterations fails on it as well. Both copies of each double eigenvalue come out of the larger bases. Give a problem like this a basis of several times the number of eigenvalues wanted and a larger budget, or use shift-and-invert.

**Time** (conditions as under [Storage and operations](#storage-and-operations)), six eigenvalues, a basis of 20, `tol = 10⁻¹⁰`, the same start vector as ARPACK's; ARPACK's time includes its eigenvectors:

| Operation | Matrix | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `lanczos_with` | ani2d-316, n = 99856, nnz = 498016, 6 eigenvalues, LA | 9.06 s | SciPy eigsh (ARPACK) | 3.67 s | 2.47 |
| `lanczos_with` | ani3d-22, n = 10648, nnz = 71632, 6 eigenvalues, LA | 57.6 ms | SciPy eigsh (ARPACK) | 23.8 ms | 2.42 |
| `lanczos_with` | randspd-100000, n = 100000, nnz = 499978, 6 eigenvalues, LA | 328 ms | SciPy eigsh (ARPACK) | 152 ms | 2.15 |
| `arnoldi_with` | circuit-100000, n = 100000, nnz = 417650, 6 eigenvalues, LM | 110 ms | SciPy eigs (ARPACK) | 24.6 ms | 4.45 |
| `arnoldi_with` | randunsym-100000, n = 100000, nnz = 899953, 6 eigenvalues, LM | 1.78 s | SciPy eigs (ARPACK) | 544 ms | 3.27 |
| `lanczos_with` | ani2d-100, n = 10000, nnz = 49600, 6 eigenvalues, SA | 133 ms | SciPy eigsh (ARPACK) | 65.1 ms | 2.05 |
| `lanczos_with` | ani3d-22, n = 10648, nnz = 71632, 6 eigenvalues, SA | 57.5 ms | SciPy eigsh (ARPACK) | 25.5 ms | 2.26 |
| `lanczos_shift_invert_with` | ani2d-316, n = 99856, nnz = 498016, 6 eigenvalues, σ = 0 | 414 ms | SciPy eigsh(sigma=0) (ARPACK + SuperLU) | 352 ms | 1.18 |
| `lanczos_shift_invert_with` | ani3d-22, n = 10648, nnz = 71632, 6 eigenvalues, σ = 0 | 160 ms | SciPy eigsh(sigma=0) (ARPACK + SuperLU) | 244 ms | 0.658 |
| `arnoldi_with` | cd2d-100, n = 10000, nnz = 49600, 6 eigenvalues, LR | 196 ms | SciPy eigs (ARPACK) | 59.1 ms | 3.32 |
| `arnoldi_shift_invert_with` | cd2d-316, n = 99856, nnz = 498016, 6 eigenvalues, σ = −0.5 | 6.69 s | SciPy eigs(sigma=-0.5) (ARPACK + SuperLU) | 3.97 s | 1.68 |

On the convection–diffusion operator `cd2d-316` (`arnoldi_with`, largest magnitude) ARPACK and faer returned a wrong set of eigenvalues, so the case has no reference that does the same work.

Lanczos takes 2.05 to 2.47 times ARPACK's time, Arnoldi 3.27 to 4.45 and the shift-and-invert drivers 1.18 and 1.68 (0.658 on the 3-D grid): more products than ARPACK (8,687 against 7,315 on `ani2d-316`, 81 against 32 on the circuit), and more work in each restart — the basis orthogonalised fully, in two passes, a dense eigenvalue problem of the basis solved at every restart, and the search for copies of an eigenvalue that the restarts missed, a further run.

### Limits

- The check for missed copies rests on the premise above; it finds the copies of the multiple eigenvalues of the tests, and an extreme eigenvalue a run missed, but cannot help where every run misconverges. Selections that are interior in the spectrum are where that happens: `SmallestImaginary` of a real matrix is always interior (between the pairs above and below the axis), and with the default basis of 20 it converges, on the rotation matrix, to genuine eigenvalues that are not the nearest the axis — as ARPACK does; a basis of 60 finds the right ones. The selections by the imaginary part of any matrix, and by the real part of a complex matrix whose eigenvalues crowd the edge of its spectrum, can do the same with a small basis or a loose tolerance: random stress of 250 calls a type and seed found such results for a few of them (a real Ritz value of a real matrix ties with every other at `|Im λ| = 0`), never for Lanczos. For such eigenvalues, shift-and-invert is the method.
- A basis much smaller than `2k` restarts with one or two shifts and stagnates on a clustered spectrum, as ARPACK's does: the 1-D Laplacian of 150 takes ARPACK 2,498 restarts with a basis of 5 for its 3 largest and fails with 4. Such a call spends its budget and reports the most wanted pairs that converged; it never stops before. Shift-and-invert does stop before when no restart can change the pairs still open: the basis spans all of the space that is left, or they have converged on the inverse as far as its rounding allows.
- For a non-normal matrix with nearly parallel eigenvectors, the orthonormal basis a run locks is that of the span of its Ritz vectors, as good as their conditioning allows; no Schur reordering is used. On the highly non-normal Grcar matrix and Jordan blocks, `f64` returns nothing within the budget, and `f32` returns pairs with small residuals that lie far from the eigenvalues — pseudospectral values, as rounding makes of such matrices.
- `‖A‖` in the convergence test is estimated from below by the products the method made, which makes the relative floor of the test no larger than it should be.
- The tolerance should not be set below the relative accuracy of the operator's products (of the solves, for an inverse given as an operator): no residual goes below it, and the check after the iteration allows for rounding only.
- Shift-and-invert of a matrix that is not normal, with a target `δ` from an eigenvalue, in every precision: the pairs other than the nearest stop at residuals of about `ε·‖A‖·|λⱼ − σ|/δ`, the rounding of the inverse along the nearest eigenvector, above the room of the test at the default tolerance from about `δ = 10⁻³·‖A‖` on (measured above). Only the nearest then comes back, and `converged < k`; a tolerance of `10·ε·‖A‖/δ`, `δ = |λ₁ − σ|` from a first call, brings the others back, with residuals of that size, as ARPACK reports them. The pairs reported always pass with `A`.
- The shift-and-invert forms on a `SparseMatrix` rely on `SparseLu`. They were written against a draft of it that was wrong whenever partial pivoting interchanged two rows, so their accuracy was measured on matrices whose `A − σI` needs no interchange, and interior targets through `lanczos_inverse_with` and `arnoldi_inverse_with` with the dense LU; the tests that need an interchange check the nearest values only where the factorisation solves. With `SparseLu` itself every test of the area passes, those parts included. With a factorisation that does not solve `A − σI` the pairs reported are still eigenpairs of `A`, by their residuals, but not necessarily the nearest `σ`.
- Every restart costs a dense eigensolve of order `m`, Gram–Schmidt on `n·m²` (two passes, in blocks of rows, four columns of the basis at a time) and the products with the operator, and shift-and-invert a product with `A` for each pair it judges; the search for missed copies is a further run.

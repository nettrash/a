[A](../../../README.md) › [Math](../../../README.md#math)

# Lattices

`a::math::lattice` reduces bases of integer lattices, finds their shortest and closest vectors, and finds integer relations among real numbers. A basis is a `Matrix<Number>` of [`a::math::matrix`](matrix.md) whose **rows** are the vectors `b₀, …, b_{n−1}`, which generate the lattice `{Σ xᵢ·bᵢ : xᵢ ∈ ℤ}`, as fplll, NTL and FLINT have them. The rows may be linearly dependent, and fewer or more than the columns. The Hermite and Smith normal forms of a basis, and matrices over `GF(p)` and `GF(2)`, are on the page of [matrices](matrix.md#normal-forms).

- LLL reduction exactly, on integers (`lll`); by the L² algorithm of Nguyen and Stehlé in floating point, the result proved afterwards (`lll_fp`); with deep insertions (`deep_lll`); and the exact test `is_lll_reduced`.
- Shortest and closest vectors, and every vector of squared norm at most a bound, by enumeration, exactly (`shortest_vector`, `closest_vector`, `short_vectors`).
- BKZ reduction (`bkz`), Schnorr and Euchner's: unless a limit on its tours stops it, every block's first Gram–Schmidt vector ends a shortest vector of the block's projected lattice, exactly.
- Integer relations among real numbers by PSLQ (`pslq`, `pslq_f64`), and a proved bound below which there is none.

A reduction returns a `Reduced`: the reduced basis `B'`, a zero row first for each linear dependency among the rows given, then the reduced basis of the lattice; the unimodular `U` with `U·B = B'`; and the rank. The reductions and the enumerations are exact: floating point steers some of them, and their results are then proved, or computed again in integers. PSLQ's answers are about its numbers as given, to their precision. Invalid parameters and shapes that do not fit panic with a message.

**Which basis.** A lattice has many reduced bases, and which one a reduction returns is not promised. What is promised is what each function states — the reduction to the given `δ` and `η` (and BKZ's blocks), `U` unimodular with `U·B = B'`, the zero rows first, the rank — and that the same input and options give the same result with the same version of the crate on the same platform (no randomness, threads or timing take part; a few choices go through the platform's `log2`). The basis is not promised to equal another implementation's (fplll's, FLINT's, NTL's, PARI's) or another version's of this crate. Since the speed pass of 0.2, for instance, `lll` starts from the basis of `lll_fp` and returns another reduced basis than 0.2.45 did for about half of the inputs (1,034 of 1,800 the same in the audit of the pass), and so, where they build on it, do `deep_lll` and `bkz`; `lll_fp` returned 0.2.45's basis on 1,899 of 1,900 (the other the same but for the sign of a row; see its run in machine words below). The same holds of PSLQ's relations to the precision (below).

## Examples

```rust
use a::math::lattice::{BkzOptions, LllOptions, bkz, is_lll_reduced, lll, lll_fp, shortest_vector};
use a::math::matrix::Matrix;
use a::math::numbers::Number;

let b = Matrix::from_rows(&[[1, 1, 1], [-1, 0, 2], [3, 5, 6]]).map(|&x| Number::from(x));
let r = lll(&b, &LllOptions::new());
assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
assert_eq!(&r.transform().clone() * &b, *r.basis());
assert_eq!(r.rank(), 3);

// The same lattice by L², and by BKZ with a block as large as the lattice, whose first vector is
// then a shortest one.
let f = lll_fp(&b, &LllOptions::new());
assert!(is_lll_reduced(f.basis(), 0.99, 0.51));
let h = bkz(&b, &BkzOptions::new(3));
let first: Number = (0..3).map(|j| &h.basis()[(0, j)] * &h.basis()[(0, j)]).sum();
assert_eq!(first, shortest_vector(&b).unwrap().norm2);
```

The sections below give each part's functions, its method and sources, an example, what it was verified against, and its limits. There are no timings here yet: they come with the release.

## LLL reduction

A basis `b₀, …, b_{n−1}` (the rows of a `Matrix<Number>`) is `(δ, η)`-LLL-reduced when its Gram–Schmidt coefficients `μᵢⱼ = ⟨bᵢ, b*ⱼ⟩/|b*ⱼ|²` are size-reduced, `|μᵢⱼ| ≤ η` for `j < i`, and Lovász's condition `|b*ₖ|² ≥ (δ − μ²ₖ,ₖ₋₁)·|b*ₖ₋₁|²` holds for each `k ≥ 1`. Then `|b₀|² ≤ (δ − η²)^−(n−1)·λ₁²`, `λ₁` the length of a shortest nonzero vector. Rows that are linearly dependent come out as zero rows, first, one for each dependency, and the reduced basis of the lattice after them.

| Function | Method | Arithmetic | The result |
| --- | --- | --- | --- |
| `lll(&b, &LllOptions)` | integral LLL (de Weger; Cohen, Algorithm 2.6.7), dependent rows by Pohst's MLLL in the integral form of Cohen's Algorithm 2.7.2 | integers only | `(δ, 1/2)`-LLL-reduced exactly, for `δ` in `(1/4, 1]` |
| `lll_fp(&b, &LllOptions)` | the L² algorithm of Nguyen and Stehlé | the basis and its Gram matrix in integers (machine words while they fit), the Gram–Schmidt coefficients in plain doubles or 53-bit floats with an exponent of their own, then in `BigFloat`s | `(δ, η)`-LLL-reduced, proved on the result, for `δ` in `(1/4, 1)` and `η` in `(1/2, √δ)` |
| `deep_lll(&b, &DeepLllOptions)` | LLL with deep insertions (Schnorr and Euchner) on the machinery of L²; for `η = 1/2` the integral one | as `lll_fp`; integers only for `η = 1/2` | `(δ, η)`-LLL-reduced and the deep exchange condition at every place the depth allows, proved, for `δ` in `(1/4, 1)` and `η` in `[1/2, √δ)` |
| `is_lll_reduced(&b, δ, η)` | integral Gram–Schmidt orthogonalisation | integers only | whether `b` is `(δ, η)`-LLL-reduced, exactly, the zero rows first |

The parameters are floats taken exactly, as the dyadic rationals they are; a `δ` or `η` out of its range panics with a message.

| Type | |
| --- | --- |
| `LllOptions` | `delta` (0.99 by default) and `eta` (0.51), as fplll's; `new()`, `with_delta`, `with_eta`, `Default` |
| `DeepLllOptions` | `delta`, `eta`, and `depth`: `None` (the default) for insertions anywhere, `Some(t)` for insertions only at the first `t` places of the lattice's rows or the last `t` before the vector, and always at the place just before it, LLL's exchange, so that `Some(0)` is LLL |
| `Reduced` | `basis()`, the reduced rows with the zero rows first; `transform()`, the unimodular `U` with `U·B = B'`; `rank()`, the number of nonzero rows; `into_parts()` |

### Methods

- **`lll`**, from 4 rows (and for `δ > 0.51² + 1/64`), first reduces the rows by `lll_fp` to `(min(δ, 0.99), 0.51)`, proved, and goes on from that basis, its transformation carried on; the integral reduction then has little more to do than the size reduction to `1/2` and the few exchanges that reduction asks for (or, for `δ` above `0.99`, the rest of the way). The integral reduction keeps the Gram–Schmidt data as integers: `dᵢ`, the Gram determinant of the first `i` rows, and `λᵢⱼ = dⱼ₊₁·μᵢⱼ`, computed a row at a time as the reduction first reaches it. The size reduction subtracts the integer nearest to `λₖₗ/dₗ₊₁`, the exchange updates `d` and `λ` by exact divisions, and Lovász's condition is the integer test `den·(dₖ₊₁·dₖ₋₁ + λ²ₖ,ₖ₋₁) ≥ a·dₖ²` for `δ = a/den`. For dependent rows each row carries a flag, `b*ᵢ ≠ 0`, and `dᵢ` is the Gram determinant of the independent rows among the first `i`; a row with `b*ₖ = 0` is exchanged downwards: while `μ = λₖ,ₖ₋₁/dₖ` is not zero the exchange replaces `b*ₖ₋₁` by `μ·b*ₖ₋₁` (`|μ| ≤ 1/2`, a step of Euclid's algorithm on the coefficients), and when it is zero the zero Gram–Schmidt vector moves down a place, until the row is the zero vector at the front. The potential, the product of the `dᵢ₊₁` of the independent rows, is a positive integer that falls below `δ` times itself at an exchange of independent rows and to a quarter of itself or less at an exchange that multiplies a Gram–Schmidt vector by the coefficient `μ` of a dependent row; the other exchanges, which move a zero Gram–Schmidt vector down a place, happen at most `n²` times. For a fixed `δ < 1` that makes `O(n²·log B)` exchanges for `n` rows of norm at most `B`, each followed by `O(n·(n + m))` operations on integers of `O(n·log B)` bits for `m` columns. `δ = 1` ends too, for the same reasons (the potential is an integer that falls at every exchange of the first two kinds), but with no polynomial bound. `η` is checked, but the reduction is to `1/2`.
- **`lll_fp`** is L² (*Floating-point LLL revisited*, Eurocrypt 2005; *An LLL algorithm with quadratic complexity*, SIAM J. Comput. 39, 2009). The basis, `U` and the Gram matrix are kept in integers: in machine words while they fit (fplll's choice of `long`), and beyond as a sign and limbs that the size reductions update in place, `y ← y − x·z` by the limb kernels into working memory, so that nothing is allocated once the rows have their lengths. The Gram–Schmidt coefficients of a row are computed from the exact Gram matrix by the Cholesky factorisation algorithm, in floating point; the size reduction is lazy (Figure 5): while some `|μ̄ₖⱼ| > η̄`, the coefficients are rounded from the top down, each correcting those below it, the row and the Gram matrix are updated exactly, and the coefficients computed again. The vector is then moved down to the place where Lovász's condition stops failing (Figure 4). The tests use `δ̄ = δ + min((1 − δ)/4, 1/64)` and `η̄ = η − min((η − 1/2)/2, 1/64)`, stricter than `δ` and `η` so that rounding does not leave the result short of them, computed exactly and rounded towards `δ` and `η` at the precision of each run: `δ ≤ δ̄ < 1` and `1/2 < η̄ ≤ η` always. A `BigFloat` run holds them exactly, with their margin; a 53-bit float holds no float strictly between `δ` and `δ̄` when `1 − δ < 2⁻⁵¹`, nor between `η̄` and `η` when `η = 1/2 + 2⁻⁵³`, and the 53-bit run then tests with `δ` or `η` itself. A row whose exact squared norm becomes zero is a dependency: it is moved out of the way and comes first in the result. When the rows, `U` and the Gram matrix all fit in machine words (every Gram entry below `2^60`), the first run keeps them as flat arrays of `i64`, the whole Gram matrix in both triangles, each row with its largest magnitude: a row operation `y ← y − x·z` is checked once, `max|y| + |x|·max|z| < 2⁶³`, and then runs on the words without a test, with the same steps and the same plain doubles as below; a step that would overflow ends that run, and the next goes on in integers of any size from the basis it reached (its steps starting again there, so that its reduced basis may differ from that of a run in integers from the start: on some bases of dependent rows, whose transformations pass `2^63`, another reduced basis or the same but for the sign of a row; either is proved). Otherwise, or after it, the run is in plain doubles when the Gram matrix's entries are below `2^1000`, and for entries from `2^1000` to `2^3000` in plain doubles of the entries times `2⁻ˢ`, the largest brought to about `2^1000` (the norms and products of the orthogonalisation are all scaled by the same power of two and the coefficients not, so the steps are the same) (the orthogonalisation's products are at most its largest entry, by Cauchy and Schwarz): the arithmetic of the 53-bit floats with an exponent of their own below, the same rounding at every step, without the exponent's work, so that the run takes the same steps until a value overflows or underflows, which turns it into a NaN and ends the run before it decides anything on it. Then, or from the start for longer entries, in 53-bit floats with an exponent of their own (fplll's `dpe`), so that Gram matrices of entries of any size fit; then, if needed, runs in `BigFloat` of `p = d·log₂ρ + 2·log₂(d + 1) + 32` bits (at least 106) for `d` rows and `ρ = (1 + η)²/(δ − η²)`, the precision of Nguyen and Stehlé's theorem, about `1.6·d` for the default parameters, then `2p` and `4p`, each from the basis the run before left, as far as `2^40` bits, the greatest precision of a `BigFloat` (`δ − η²` is taken exactly: it is at least `2⁻¹⁰⁶` for valid parameters, though its value in an `f64` may be 0; beyond `2^40` bits the exact reduction is left to finish). A run fails, rather than loops, at a coefficient that is not finite, a squared norm that is not positive, a size reduction that stops shortening the row, an exchange that undoes the one before it with a row that its size reduction left as it was (which exact arithmetic never asks for), or more iterations than the bound from the potential. Each result is proved: its Gram matrix is computed again, exactly, and factorised in interval arithmetic, first in doubles stepped a unit outwards after each operation (while its entries are below `2^500`; the intervals widen with the depth of the factorisation, so they decide bases of up to some tens of rows), then in balls of double-doubles (a centre of two doubles by the double-word algorithms of Joldes, Muller and Popescu, whose relative errors are below `5u²` for sums and products, and a radius in doubles that takes them at `32u²`, `128u²` for the quotients, with the propagated radii, enlarged against its own rounding; entries below `2^400`), which decided the reduced knapsack bases of 50 to 200 rows and uniform ones of 8 to 10 bits and up to 200 rows that were measured, at about the cost of the factorisation in double-doubles, but not q-ary bases `[[I, A], [0, q·I]]` of 150 and 200 rows with `q` of 30 bits (Gram entries of 40 bits), which go on to the next stage — how many rows the double-doubles decide depends on the conditioning, not on the size alone — then in `BigFloat` bounds rounded outwards at `64 + 2d` bits, then twice and four times that; a condition that the intervals leave open goes to the exact test of `is_lll_reduced`. If no run is proved reduced, the integral reduction of `lll` finishes from the basis the runs left. Nguyen and Stehlé bound L² at their precision by `O(d⁴·m·(d + log B)·log B)` bit operations, quadratic in `log B` where the integral reduction is cubic; the proof of a result adds `O(d²·m)` products for the Gram matrix and `O(d³)` operations on intervals of `O(d)` bits.
- **`deep_lll`** follows Schnorr and Euchner (*Lattice basis reduction: improved practical algorithms and solving subset sum problems*, Math. Programming 66, 1994): after the size reduction of `bₖ`, `|πᵢ(bₖ)|²`, the squared norm of its projection orthogonal to `b₀, …, bᵢ₋₁`, is computed for each `i < k`, and `bₖ` is inserted at the first place allowed where `|πᵢ(bₖ)|² < δ̄·|b*ᵢ|²`. With insertions anywhere the work goes on after the inserted vector; with a depth, at it, since the places allowed change with the position. The runs, the proof (with the deep conditions) and the last resort are those of `lll_fp`; the integral engine inserts deeply too, with `Tᵢ = dᵢ·|πᵢ(bₖ)|²`, the Gram determinant of `b₀, …, bᵢ₋₁, bₖ`, an integer, from `Tₖ = dₖ₊₁` down by `Tᵢ = (dᵢ·Tᵢ₊₁ + λ²ₖᵢ)/dᵢ₊₁`, and the condition `den·Tᵢ ≥ a·dᵢ₊₁`. For `η = 1/2` the whole reduction is the integral one: a coefficient of exactly `1/2` may be computed in floating point as a little more, and a size reduction to `1/2` then turns it back and forth. An insertion can raise LLL's potential; the reduction ends because the sequence `(|b*₀|², |b*₁|², …)` falls lexicographically, and no polynomial bound is known.
- **`is_lll_reduced`** factorises the exact Gram matrix of the rows after the leading zero rows in the intervals of the proof of `lll_fp`: doubles stepped outwards, balls of double-doubles, then (for entries of more than 5 bits, whose integral data are long) `BigFloat` bounds at `64 + n` bits and twice that, each condition settled for certain when its interval lies on one side of its bound. A condition they leave open goes to the integral Gram–Schmidt data, which refuse a dependent row, with the tests `eb·|λₖⱼ| ≤ ea·dⱼ₊₁` for `η = ea/eb` and Lovász's condition in integers: `O(n²·(n + m))` operations on integers of `O(n·log B)` bits.

### Example

```rust
use a::math::lattice::{DeepLllOptions, LllOptions, deep_lll, is_lll_reduced, lll, lll_fp};
use a::math::matrix::Matrix;
use a::math::numbers::Number;

// A knapsack basis [aᵢ | eᵢ] of 10 numbers of 80 bits, and an eleventh row, the sum of the first
// two, which makes the rows dependent.
let a: Vec<Number> = (1..=10u32)
    .map(|i| Number::from(7u8).pow(28) / Number::from(i) + Number::from(i))
    .collect();
let mut rows: Vec<Vec<Number>> = (0..10)
    .map(|i| {
        (0..11)
            .map(|j| if j == 0 { a[i].clone() } else { Number::from(u8::from(i + 1 == j)) })
            .collect()
    })
    .collect();
let sum: Vec<Number> = rows[0].iter().zip(&rows[1]).map(|(x, y)| x + y).collect();
rows.push(sum);
let b = Matrix::from_fn(11, 11, |i, j| rows[i][j].clone());

let exact = lll(&b, &LllOptions::new());
let fast = lll_fp(&b, &LllOptions::new());
let deep = deep_lll(&b, &DeepLllOptions::new().with_depth(Some(4)));
for r in [&exact, &fast, &deep] {
    // One zero row first, for the dependency, then a reduced basis of the lattice of rank 10.
    assert_eq!(r.rank(), 10);
    assert!((0..11).all(|j| r.basis()[(0, j)].is_zero()));
    assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
    assert_eq!(&r.transform().clone() * &b, *r.basis());
}
// The exact reduction is size-reduced to 1/2.
assert!(is_lll_reduced(exact.basis(), 0.99, 0.5));
```

### Verified

- **fplll and FLINT.** 66 bases (`lib/tests/math/lattice/data/lll_reference.txt`, written by `gen_lll_reference.py` with `gen_lll_flint.c`), each generated by the tests again from its seed: 30 random square bases of 2 to 20 rows of 8, 30 and 100 bits with `δ` = 0.99 or 0.75; 24 knapsack bases `[aᵢ | eᵢ]` of 4 to 24 numbers of up to 240 bits; 12 rank-deficient bases of 3 to 16 rows, each row a combination of `n − 2` random rows. fplll 5.5 (`fplll -a lll -d δ -e 0.51`) and FLINT 3.6 (`fmpz_lll`) reduced each. The ranks agree in all 66. Of the first rows, `lll` gives fplll's squared norm in all 49 cases of 12 rows or fewer; `lll_fp` gives fplll's in 62 of 66 and fplll's or FLINT's in 64 (fplll and FLINT agree in 57); the base-2 logarithm of the ratio to fplll's is −0.004 on average, between −0.28 and +0.21. Every result is reduced, by the exact test, and has `U·B = B'` with `U` unimodular (its determinant ±1, computed exactly).
- **Unique reduced bases.** A lower-triangular `T` with the diagonal `10·100ⁱ` and `|μᵢⱼ| ≤ 0.4` has a reduced basis unique up to the signs of the rows; hidden by a random unimodular matrix, with up to 3 zero rows mixed in, up to 14 rows, `lll` (for `δ` = 0.75, 0.99 and 1), `lll_fp` and `deep_lll` with `η = 1/2` give `±T` row for row, the zero rows first. fplll and FLINT give `±T` on the six `lll_fp` cases, 12 of 12.
- **The test itself.** `is_lll_reduced` agrees with the definition evaluated in `Rational`s (`b*ᵢ = bᵢ − Σ μᵢⱼ·b*ⱼ` in the tests, sharing no code with the library) on 120 matrices, random, reduced, and reduced then spoilt by a row operation or an exchange, each answer given more than 20 times. The same rational code checks the deep exchange condition on the results of `deep_lll`, for no depth and depths 0, 1, 3 and 2.
- **Deep insertions.** NTL 11.6's `LLL_FP(B, 0.99, deep)` on 24 knapsack bases of 12 to 24 numbers (`deep_reference.txt`, from `gen_deep_reference.py` with `gen_deep_ntl.cpp`): `deep_lll` gives NTL's first vector in 24 of 24 both with insertions anywhere (NTL's `deep = n`) and at depth 5, and `lll_fp` gives that of NTL's plain `LLL_FP` in 20 of 24. On 16 knapsack bases of 20 numbers of 40 bits, the first vector of `deep_lll`, with no depth and with depth 5, is shorter than that of `lll_fp` in 16 of the 32 runs and as long in the other 16. Depth `Some(0)` gives `lll_fp`'s basis exactly, on every basis tried.
- **Every stage of the proof.** Bases built from sums of two squares (`gen_l2_boundary.py`) fail Lovász's condition, or the deep exchange condition at the first place, by 2 in `δ·A²`, for `δ = 1 − 2⁻⁵³` and `A = 2³⁵, 2²⁰⁰, 2⁵⁰⁰`. The 53-bit run, which has no margin at this `δ`, does not see the failure; the intervals prove it (`A = 2³⁵`) or leave it open and the exact test finds it (`2²⁰⁰`, `2⁵⁰⁰`); then the 106-bit run, whose `δ̄ = δ + 2⁻⁵⁵` holds a margin far wider than the failure, repairs it. Each result has the short row first, as it must. With an added row whose coefficient is 0.7, which a floating-point run at `η = 0.9` leaves as it is and the exact reduction takes down to −0.3, the tests also show which stage produced the result: the BigFloat run for `lll_fp` and `deep_lll`, the exact reduction with deep insertions for `deep_lll` at `η = 1/2`. A knapsack of 24 numbers of 240 bits with such a row is reduced at `δ = 1 − 2⁻⁵³` by the floating-point runs, keeping it (the iteration bound allows them the iterations they need). On a reduced basis of 16 rows of 2000 bits, `lll_fp` proves the result in less than an eighth of the time the exact test takes on it, timed in the same test. Each of these tests fails if the BigFloat stage always fails, if the proof always refuses or always leaves the question open, or if the iteration bound falls back to its old formula (each tried).
- **Edges.** 0 × 0, 0 × n and n × 0 matrices, one row, zero rows anywhere, repeated and proportional rows, rank 1 among many rows, more rows than columns; entries of 1500 to 5000 bits for `lll_fp` (Gram entries far beyond the range of an `f64`) and of up to 3000 bits for `lll`; `δ` from 0.2500001 to `1 − 2⁻⁵³`, and 1 for `lll`; `η` from `1/2 + 2⁻⁵³` to 0.994; coefficients of exactly `1/2`. Each invalid `δ` and `η` panics with its message.
- **Larger sizes**, in release mode (the ignored test `the_sizes_it_is_for`): 100 × 100 of 100 bits, knapsacks of 50 numbers of 1000 bits, of 100 of 100 bits and of 40 of 400 bits, 60 rows of rank 50, 70 rows of 60 columns: each reduced, with `U·B = B'` and the rank of the matrix.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 45 tests pass.

To run the tests: `cargo test --test math -- lattice::lll lattice::l2 lattice::deep`, and `cargo test --release --test math -- --ignored lattice::l2` for the larger sizes. The data files are written again by the scripts beside them, with fplll, FLINT and NTL installed.

### Limits

- `lll` reduces to `η = 1/2` whatever `η` is given, and its integers grow to `O(n·log B)` bits: it is the reduction for small and middling dimensions, and `lll_fp` the one for large ones.
- `lll_fp` needs `δ < 1` and `η > 1/2`, as L² does; `lll` takes `δ = 1` and is size-reduced to `1/2`.
- With the margins of the BigFloat runs, a run at the theorem's precision sees every failure the proof could find, so the runs at `2p` and `4p` and the exact reduction after them are reached by no input known at valid parameters; they are kept as a safety net, the exact reduction being the engine of `lll` (and of `deep_lll` at `η = 1/2`), tested there. The step that passes the runs' basis on to the exact reduction, and takes the rank from it, is reached by no test: the fixes of the audit moved the last of the boundary bases above to the 106-bit run, and the crate has no hook to force it, so the tests of every stage of the proof stop short of it. Nor does a test reach the failure of a run at an exchange that undoes the one before it, which never fired in some 170,000 traced calls. An audit forced the fallback, every proof made to refuse, on 9,000 seeded calls of up to 14 rows: `U·B = B'`, the rank and an independent exact check of the reduction held on all of them.
- A floating-point run stops at a bound on its iterations: twice the number of exchanges that would lower the potential, at most `Πᵢ |bᵢ|^(2(n − i))`, by the factor `(1 + δ̄)/2` each, plus `4n + 1000`, kept below about `2·10¹⁵`; a run whose tests err by less than that margin does not reach it, and for `δ` near 1 it does not stop a run in practice. For deep insertions, which the potential does not bound, the bound is that one times `n + 1`, a guess; a run that reaches it only hands the basis on to the next stage.
- `deep_lll` has no polynomial bound on its work, and with `η = 1/2` it runs in integers throughout.
- The transformation is always computed: for dependent rows its entries hold the integer relations among the rows, which can be far longer than the entries of the basis, and updating them is then a large part of the work.
- The speed pass of the release moved the exact updates of the rows, the Gram matrix and `U` to machine words and limbs updated in place, and the first run to plain doubles; for the bases whose Gram matrix fits in words, the whole run to flat arrays of words checked once a row operation; and for Gram matrices of up to `2^3000`, the floats with an exponent of their own to plain doubles scaled by one power of two. Of fplll's other speed-ups, an exponent for each row and the early reduction of the first rows are not in. The transformation is always computed. The times are under [Measured](#measured).

## Shortest and closest vectors

`shortest_vector`, `closest_vector` and `short_vectors` find lattice vectors by enumeration: every integer combination of the rows whose norm, or distance to a target, is below a radius. All three are exact: a shortest vector is a true shortest nonzero vector, a closest vector a truly closest one, and the list of short vectors complete, each with its norm as an integer. The rows may be dependent, and fewer or more than the columns; the zero lattice and rows of thousands of bits are handled.

| Item | What it is |
| --- | --- |
| `shortest_vector(&b)` | a shortest nonzero vector of the lattice of the rows of `b`, of `±v` the one whose first nonzero element is positive; `None` for the lattice `{0}` (no rows, or only zero rows) |
| `closest_vector(&b, &t)` | a lattice vector closest to the integer vector `t`, its `norm2` the squared distance `|v − t|²`; a target outside the span of the rows is measured with its component orthogonal to them; `None` for a basis of no rows, the zero vector for a zero lattice; panics if `t` has not as many elements as `b` has columns |
| `short_vectors(&b, &bound)` | every nonzero vector with `|v|² ≤ bound`, one of each pair `±v` (the one whose first nonzero element is positive), sorted by norm and, among equal norms, lexicographically by their elements, so that the list depends on the lattice alone and not on its basis; empty for a bound below 1 |
| `LatticeVector` | `coordinates` in the rows given (one for each row; for dependent rows, one choice of them), `vector`, and `norm2`: `|v|²`, or `|v − t|²` for `closest_vector` |

### Method

- The basis is LLL-reduced first (δ = 0.99, η = 0.51) by the L² algorithm, `lll_fp`, whose result is proved `(δ, η)`-reduced; dependent rows come out of it as zero rows, which are left out, and the rows of its transformation that give the others carry the coordinates back to the rows given.
- The Gram–Schmidt data of the reduced basis are computed exactly, in the integral form of de Weger and Cohen (Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 2.6.7): the Gram determinants `dᵢ` and `λᵢⱼ = dⱼ₊₁·μᵢⱼ` are integers, and so is `d_l·|π_l(w)|²` for every integral vector `w`. From 16 rows they come from their residues modulo word primes: they are minors of the Gram matrix `G`, and modulo a prime that divides none of the `dᵢ` the `L·D·Lᵀ` factorisation of `G` (the PLE decomposition of `ModMatrix`, with no row exchanged) gives `μᵢⱼ` and `rⱼⱼ = dⱼ₊₁/dⱼ`; each is rebuilt by the Chinese remainder theorem once the product of the primes passes twice its bound by Hadamard's inequality (`dⱼ ≤ Π_{l<j} G_ll`, `|λᵢⱼ| ≤ √(Gᵢᵢ·Gⱼⱼ)·Π_{l<j} G_ll`). A prime that divides a `dᵢ` is skipped; more such primes than the length of `Π dᵢ` allows would prove the rows dependent. About `n·b/30` primes for Gram–Schmidt vectors of `b` bits, each `O(n^ω)` operations on words, against `n³/6` steps on numbers as long as the minors row after row.
- The search is Fincke and Pohst's (*Improved methods for calculating vectors of short length in a lattice*, Math. Comp. 44, 1985) in Schnorr and Euchner's order (*Lattice basis reduction: improved practical algorithms and solving subset sum problems*, Math. Programming 66, 1994): depth first from the last level, each coefficient running in the zig-zag order `round(c), round(c) ± 1, …` of its distance to the centre `c_l = μ_{t,l} − Σ_{j>l} x_j·μ_{j,l}`, and the radius cut to just below each better vector found. The centres are kept as partial sums brought up to date from the highest coefficient that changed (Gama, Nguyen and Regev, *Lattice enumeration using extreme pruning*, Eurocrypt 2010, appendix), so that a node costs `O(1)` on average. Without a target only one of `±v` is visited and the zero vector is not.
- `shortest_vector` starts from the radius of the first reduced vector and searches for anything strictly shorter. `closest_vector` first moves the target near the origin by Babai's nearest plane method, exactly, so that every `|μ_{t,l}| ≤ 1/2`; the lattice vector that took away is the first answer, and the search looks for strictly closer ones. `short_vectors` keeps its radius at the bound.
- Top levels that no solution can use are left out: without a target, the last rows whose Gram–Schmidt norms all exceed the radius, since their projected lattice has no nonzero vector within it; with a target, the levels where a coefficient other than zero, at least `1/2` from its centre, would put the distance beyond the radius. This keeps the huge last Gram–Schmidt norms of a knapsack lattice out of the search.
- **Exactness.** The search runs in `f64` on the exact data rounded once, but it compares a rigorous lower bound of each partial norm with a radius enlarged past the rounding of that bound: the error of each centre is bounded by `(2n + 16)·u·(Σ_{j>l}|x_j|·max|μ_{j,l}| + |μ_{t,l}| + |c_l|)` (`u = 2⁻⁵³`; Higham's bound for a recursive sum, with room for the data's own rounding), the distance to the centre is lowered by it, each Gram–Schmidt norm is lowered by `4u`, and as every term of a partial norm is non-negative, the computed bound is at most `(1 + u)^{5n}` above the exact one; the radius is enlarged by `(8n + 32)·u`. So every node of the exact tree is visited, and every leaf is then measured in integers. Where floats cannot follow the data — a Gram–Schmidt norm below `2⁻⁴⁰` of the radius, which happens for a target half-way between rows of very different lengths, or a coefficient beyond `2⁵⁰` — the same search runs on the integers themselves: the centre `C_l = d_{l+1}·c_l` and the partial norm `L_l = d_l·|π_l(w)|²` follow from the level above by `L_l = (d_l·L_{l+1} + (x_l·d_{l+1} − C_l)²)/d_{l+1}`, an exact division.
- The cost is that of the enumeration tree, `2^O(n²)` nodes for a lattice of rank `n` on an LLL-reduced basis, besides the LLL reduction and the exact Gram–Schmidt data, which are polynomial. It is practical up to a rank of about 45; `short_vectors` costs, besides, about the number of vectors found, near `(π·bound)^(n/2) / (Γ(n/2 + 1)·vol(L))`.

### Example

```rust
use a::math::lattice::{closest_vector, short_vectors, shortest_vector};
use a::math::matrix::Matrix;
use a::math::numbers::Number;

// The root lattice E₈, doubled to be integral: 240 vectors of the norm 8.
let e8 = Matrix::from_rows(&[
    [1, -1, -1, -1, -1, -1, -1, 1],
    [2, 2, 0, 0, 0, 0, 0, 0],
    [-2, 2, 0, 0, 0, 0, 0, 0],
    [0, -2, 2, 0, 0, 0, 0, 0],
    [0, 0, -2, 2, 0, 0, 0, 0],
    [0, 0, 0, -2, 2, 0, 0, 0],
    [0, 0, 0, 0, -2, 2, 0, 0],
    [0, 0, 0, 0, 0, -2, 2, 0],
])
.map(|&x| Number::from(x));
let s = shortest_vector(&e8).unwrap();
assert_eq!(s.norm2, Number::from(8u8));
assert_eq!(short_vectors(&e8, &Number::from(8u8)).len(), 120);

// The coordinates give the vector from the rows.
for j in 0..8 {
    let e: Number = (0..8).map(|i| &s.coordinates[i] * &e8[(i, j)]).sum();
    assert_eq!(e, s.vector[j]);
}

// Five dependent rows in three dimensions, which span the vectors (2a, 2b, a): the shortest
// is (0, 2, 0), and the closest to (3, 3, 3) are (4, 2, 2) and (4, 4, 2), at 1 + 1 + 1.
let b = Matrix::from_rows(&[[6, 0, 3], [4, 2, 2], [0, 6, 0], [2, 2, 1], [10, 4, 5]])
    .map(|&x| Number::from(x));
assert_eq!(shortest_vector(&b).unwrap().vector, [0, 2, 0].map(Number::from));
let c = closest_vector(&b, &[3, 3, 3].map(Number::from)).unwrap();
assert_eq!(c.norm2, Number::from(3u8));
assert_eq!(c.vector[0], Number::from(4u8));
```

### Verified

`lib/tests/math/lattice/data/gen_enum.py` makes every basis from its seed as the test makes it (the SplitMix64 generator of the tests, draw for draw) and computes the references with fplll 5.5, PARI/GP 2.19 and an exact rational Fincke–Pohst enumeration of its own, a plain recursion over fractions that shares nothing with the crate or the tools; `python3 gen_enum.py` in that directory regenerates `enum_svp.txt`, `enum_cvp.txt` and `enum_short.txt`, and `cargo test --test math -- lattice::enumerate` runs the comparisons (the heavier sweep with `--ignored`).

- **Shortest vectors**, 112 bases in the default tests and 26 in the heavier sweep: random square bases of ranks 2 to 30 with 10, 30 and 60-bit entries; knapsack lattices `[aᵢ | I]` of ranks 2 to 30 with `10·n`-bit numbers; Hilbert matrices `⌊2^k/(i + j + 1)⌋` of orders 2 to 12; bases made of Gram matrices `A·Aᵀ`; q-ary lattices `[I | A; 0 | q·I]` of ranks 4 to 30; rows of 500 to 3000 bits; wide (rank below the columns), tall (more rows than columns) and dependent bases. The norm equals that of `fplll -a svp` (on the nonzero rows of `fplll -a lll`), which equals the minimum of PARI's `qfminim` (flag 2, on the Gram matrix of a `mathnf` basis) on every one of the 138, and the exact enumeration's up to rank 16. Every vector returned is the combination of the rows its coordinates say, of the norm it says, with its first nonzero element positive.
- **Closest vectors**, 91 bases and targets: random targets of up to 900 bits far from the lattice, targets within 7 of a lattice vector, targets far outside the span of wide bases, on random, knapsack, Hilbert, q-ary and tall bases of ranks 1 to 20; and 24 bases with the Gram–Schmidt norms `4, 2^(2e), 2^(4e), …` (`e` = 40 to 500, ranks 2 to 6) with the target half-way at every level but the first, 20 of which the search has to run in integers (the other 4, of rank 2, leave no level at all after the trimming: Babai's point is closest). The squared distance equals the exact enumeration's on all 91 (with the target first moved by Babai's nearest plane method, exactly). fplll's `-a cvp`, which is not proved, gives the same distance on 60, a farther vector on 8 (a random lattice of rank 10, knapsack lattices of rank 20 and of rank 6 with 500-bit numbers, and the 5 half-way cases with `e = 40`), fails on the 4 lattices of rank 1 and had not finished after 15 s on the other 19 half-way cases; it never gives a closer one. On lattices of a rank below their dimension fplll was given the target projected on the span, scaled to integers, since with the component off the span its search had not ended after minutes.
- **Short vectors**, 75 bases and bounds of 1 to 5 times the minimum: the number of pairs, the number of each norm and a sign-free checksum `Σ (Σⱼ (j + 1)·vⱼ)²` of the vectors equal PARI's `qfminim(G, bound, , 2)` on every one; the list is sorted, without repeats, every vector normalised and within the bound.
- **Small lattices against brute force**: on 204 random lattices of ranks 1 to 4 with 3-bit entries (of 300 drawn; the others singular or too large to search), every combination with coefficients up to the bound that `|xᵢ| ≤ |v|·|column i of B⁻¹|` gives was tried: the shortest norm, the number of vectors within three times it and the distance to a random target all agree.
- **Known lattices**: `ℤⁿ` for `n` up to 6 (`n` vectors of the norm 1, `n·(n − 1)` pairs of the norm 2), `A₂` (3 pairs), `D₄` (12 pairs of the norm 2, 12 of the norm 4), `2·E₈` (120 pairs of the norm 8 and 1080 of the norm 16, the theta series `1 + 240q + 2160q² + …`), and `A₂` in another basis, which gives the same list in the same order.
- **Edge cases**: `0 × 0`, `0 × 3` and `3 × 0` bases, a zero matrix (whose closest vector is 0), `1 × 1`, rows on one line, six rows of 200 bits in four dimensions (relations of some 800 bits to remove; PARI's `mathnf` gives the index 2), a shortest vector hidden 2¹⁰⁰⁰ deep, a target of 3000 bits, targets on lattice vectors (distance 0), bounds below 1, and a target of the wrong length (a panic). The integer search also runs on a lattice with the Gram–Schmidt norms 4, 2⁵⁰⁰ and 2¹⁰⁰⁰ and a target half-way at every level, whose closest vectors are 2⁴⁹⁸ + 2⁹⁹⁸ away, 1 nearer than Babai's point.

### Limits

- No pruning, linear or extreme: the whole tree of the radius is searched, which is what makes the answers exact, and keeps the rank within reach to about 45.
- The search needs no more of the reduction than that it is LLL-reduced; the exact Gram–Schmidt data of the reduced basis cost `O(n³)` operations on numbers of `O(n·log B)` bits, which for bases of large entries is a large part of the work below a rank of about 40.
- `short_vectors` returns its list whole; there is no limit on its length. A coefficient of a search beyond 64 bits panics, which takes a rank far beyond what an enumeration finishes, or a bound with more vectors than memory holds.
- Of several shortest or closest vectors, one is returned; the same input gives the same one, but another basis of the same lattice may give another.

## BKZ

`bkz` reduces a basis by blocks: Schnorr and Euchner's block Korkine–Zolotarev reduction (1994), as fplll's BKZ 1.0 without pruning, with the enumeration above for the shortest vector of each block. It returns a `Reduced`, as the LLL reductions do: the basis `B'`, the unimodular `U` with `U·B = B'`, and the rank, dependent rows giving zero rows first.

| Item | What it is |
| --- | --- |
| `bkz(&b, &opts)` | the BKZ reduction of the rows of `b`; panics for a block size below 2, `δ` not in `(1/4, 1)` or `η` not in `[1/2, √δ)` |
| `BkzOptions::new(β)` | blocks of `β`, `δ = 0.99`, `η = 0.51` for the LLL reductions inside, tours until one changes nothing |
| `.with_delta(δ)`, `.with_eta(η)`, `.with_max_tours(Some(t))` | the LLL parameters, and a limit on the tours (`None`, the default, for none) |

### Method

- A tour visits the blocks `[k, min(k + β, n))` for `k = 0, …, n − 2`: the enumeration finds a shortest vector of the block's lattice projected orthogonally to `b₀, …, b_{k−1}`, and a vector short enough is inserted at `k`. Short enough is first, as Schnorr and Euchner and fplll have it, `|πₖ(v)|² < δ·|b*ₖ|²`; once a tour inserts nothing, any vector shorter than `b*ₖ`, until a tour inserts nothing again. The first kind saves tours the second would take from the start (28 and 36 tours in all against 42 and 38 with the second kind from the start, on two knapsack lattices of rank 60 with `β = 20`); the second makes the end exact.
- **First, the certified L² reduction** (`lll_fp`, the exact `lll` for `η = 1/2`) of the rows, which also sets the zero rows of dependent ones apart: the float tours start from an LLL-reduced basis, whose orthogonalisation plain scaled doubles hold.
- **Most tours run in floating point, in fplll's design.** The rows, `U` and the Gram matrix are kept exactly, as integers in a machine word while they fit and as `Number`s beyond (fplll's choice of `long` where it can), and the Gram–Schmidt orthogonalisation in floating point, computed a row at a time from the exact Gram matrix by the Cholesky factorisation, as L² computes it, and kept from one insertion and one tour to the next: a change of a row invalidates only the rows from it on. The floats are plain `f64`s scaled by one power of two once the basis is reduced and its squared norms fit, and floats with an exponent of their own otherwise (rows of thousands of bits). Each block is searched by the float enumeration above; the vector found is inserted by replacing a row whose coordinate is `±1` (the common case) and moving it to `k`, or else by unimodular gcd steps on its coordinates (each pair `(y, z)` with `g = a·y + b·z` becomes `(g, 0)` by `(r, s) ← ((y/g)·r + (z/g)·s, −b·r + a·s)`), and a local L² — lazy size reduction and Lovász's test, no certificate — reduces the rows of the block alone, none moved below `k`; a row the next block brings in is reduced before it is searched. Rows proved independent (full rank modulo the prime `2⁶¹ − 1`) are reduced this way from the start; dependent rows first go through the certified LLL reduction, which sets their zero rows apart.
- **Then, once, the certified work.** The whole basis is reduced by the L² algorithm `lll_fp`, whose result is proved `(δ, η)`-reduced (by the exact `lll` for `η = 1/2` exactly, which L² does not take), and the tours go on in exact arithmetic: the exact integral Gram–Schmidt data (computed from their residues modulo word primes, `d_i` and `λᵢⱼ = d_{j+1}·μᵢⱼ` being minors of the Gram matrix: the `L·D·Lᵀ` factorisation modulo each prime that divides none of the `d_i`, as many primes as Hadamard's bound on each asks for, and the Chinese remainder theorem), the exact enumeration, any vector strictly shorter than `b*ₖ` (`d_k·|πₖ(v)|² < d_{k+1}`, compared in integers), each insertion followed by the certified reduction of the rows to the block's end, until a tour inserts nothing; after the float tours that is usually the first. Should the floats fail (a coefficient that is not finite, a norm that is not positive, a size reduction that does not end, or more than `4n + 64` tours), the certified reduction takes the basis as it is and the float tours start again from it, twice at most, before the exact tours do the rest (each start in plain scaled doubles, and if they fail, once more in floats with an exponent of their own). Last, when the exact tours ended that way, the basis is size-reduced to `|μᵢⱼ| ≤ 1/2`, exactly (Cohen's RED on the integral data), which changes no Gram–Schmidt vector and no projected block.
- **What holds.** The result is always `(δ, η)`-LLL-reduced, `U` unimodular and `U·B = B'`: every step is an exact unimodular change, and the floats only steer. When the reduction ends because an exact tour inserted nothing — always, unless `max_tours` stops it — every block's first Gram–Schmidt vector `b*ₖ` is a shortest nonzero vector of the projected lattice of `b_k, …, b_{min(k+β,n)−1}`, exactly; fplll's BKZ promises `δ·|b*ₖ|²` at most the block's minimum. The basis is then also size-reduced to `|μᵢⱼ| ≤ 1/2`, whatever `η`, and satisfies Lovász's condition with `δ = 1` (each `b*ₖ₋₁` is shortest in its block, which holds `πₖ₋₁(bₖ − q·bₖ₋₁)` for every `q`). With `β ≥ n` this is HKZ reduction and `b₀` a shortest vector of the lattice. When `max_tours` stops it (counting every tour, of floats or exact), the basis is LLL-reduced and the blocks are promised nothing; `max_tours: Some(0)` is the LLL reduction alone.
- The reduction ends: the float tours are at most `4n + 64` per start, and in the exact ones each insertion and each exchange of LLL makes the Gram determinants `(d₁, d₂, …)`, positive integers, smaller in lexicographic order, a well-order. The work of a block is an enumeration in dimension `β`, `2^O(β²)`, so blocks up to about 30 are practical; an insertion costs an LLL reduction of the block's rows and their orthogonalisation in floats.

### Example

```rust
use a::math::lattice::{BkzOptions, bkz, is_lll_reduced, shortest_vector};
use a::math::matrix::Matrix;
use a::math::numbers::Number;

// A knapsack lattice [aᵢ | I] of rank 8.
let a = [31_415_926i64, 27_182_818, 14_142_135, 17_320_508, 22_360_679, 16_180_339, 26_457_513, 30_000_001];
let b = Matrix::from_fn(8, 9, |i, j| Number::from(if j == 0 { a[i] } else { i64::from(i + 1 == j) }));
let r = bkz(&b, &BkzOptions::new(4));
assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
assert_eq!(&r.transform().clone() * &b, *r.basis());
assert_eq!(r.transform().det().abs(), Number::from(1u8));

// Blocks as large as the lattice: HKZ, size-reduced to 1/2, the first vector a shortest one.
let h = bkz(&b, &BkzOptions::new(8));
assert!(is_lll_reduced(h.basis(), 1.0, 0.5));
let first: Number = (0..9).map(|j| &h.basis()[(0, j)] * &h.basis()[(0, j)]).sum();
assert_eq!(first, shortest_vector(&b).unwrap().norm2);

// Without a tour, the LLL reduction.
let l = bkz(&b, &BkzOptions::new(4).with_max_tours(Some(0)));
assert!(is_lll_reduced(l.basis(), 0.99, 0.51));
```

### Verified

`cargo test --test math -- lattice::bkz` runs these checks; `python3 gen_bkz.py` in `lib/tests/math/lattice/data` regenerates `bkz_fplll.txt`.

- **The block conditions, exactly, by two oracles**, on every reduction the tests make: for each `k`, the rows `b_k, …` of the block projected orthogonally to the rows before it, in rationals and times `d_k`, which makes them integral, form a lattice whose `shortest_vector` has the norm of the first, `d_k²·|b*ₖ|²`; and an exact rational Fincke–Pohst search written in the test alone (a plain recursion over every integer coordinate within the radius, sharing nothing with the crate's enumeration) finds no nonzero vector of the block shorter than `b*ₖ`. The two must agree on every block, and on LLL-reduced knapsack bases whose first vector is not a shortest one both find the full block not reduced. With the invariants of every reduction (`U·B = B'`, `|det U| = 1`, the zero rows first and as many as the rank of `B` says, `is_lll_reduced` with the `δ` and `η` given, and with `δ = 1` and `η = 1/2`), on knapsack lattices of ranks 4 to 12 and random bases of ranks 4 to 12 with `β` = 2, 3, 5 and 8, q-ary lattices of ranks 8 and 12, dependent bases (a combination of the other rows; nine rows in five dimensions), a knapsack lattice of 1000-bit numbers, and `(δ, η)` = (0.75, 0.5), (0.9, 0.55), (0.999, 0.51) and (0.3, 0.5).
- **`β = 2`**: Lovász's condition holds with `δ = 1` on knapsack lattices of ranks 3 to 14. **`β ≥ n`**: the first vector has the norm of `shortest_vector` on knapsack lattices of ranks 4 to 16, with `β = n` and `β = n + 5`, every block of the HKZ condition checked as above, and the basis size-reduced to 1/2, as on the 4 × 4 basis on which an earlier version, with `η = 0.51`, left `μ₂₁ ≈ −0.50285`.
- **Against fplll's BKZ** (`fplll -a bkz -b β`, BKZ 1.0, δ = 0.99): on 36 bases (knapsack lattices of ranks 10, 14 and 18, q-ary of 12 and 16, random of 12 and 16; `β` = 4, 6 and 8) the geometric mean of `|b₀|²` over fplll's is 0.9996 and the mean root Hermite factors `(|b₀|/vol^{1/n})^{1/n}` are 1.00366 against fplll's 1.00367; on the 16 of the ignored test (knapsack lattices of ranks 30, 40 and 60, q-ary of rank 40; `β` = 10 and 20) 0.9950, and 1.01255 against 1.01264. On no basis is one `|b₀|²` more than 1.0044 times the other (1.0805 in the larger dimensions). The tests hold the average to 1.02 and the root Hermite factor to fplll's plus 0.0005: room for the spread of single bases, which another LLL reduction inside would move, against a deficit of several percent for a BKZ short of its blocks, whose blocks would fail the exact checks besides.
- **Dependent rows**, through both reductions (`η` = 0.51, the L² algorithm, and `η` = 1/2, the exact LLL): seven bases with more rows than columns (of up to 300 bits), zero rows among the others, rows repeated and negated, and combinations with 30-bit coefficients: the invariants and the block conditions above, the rank, and the same shortest vector from the rows given and from the reduced basis.
- **Edge cases**: `0 × 0`, rows of zeros, no columns, one row; `max_tours` of 0, 1 and 2; and the panics for `β = 1`, `δ = 1`, `δ = 1/4`, `η = 0.4` and `η² ≥ δ`.

### Limits

- No pruning, no early abort and none of the strategies of BKZ 2.0 (preprocessing of the blocks, the Gaussian heuristic as the radius): a block costs a full enumeration, so blocks beyond about 30 are out of reach.
- The exact end asks for more tours than fplll's BKZ, which stops once no block has a vector below `δ·|b*ₖ|²`: on knapsack lattices of rank 80 with `β = 20` the tours with any shorter vector inserted go on for as many tours again.
- The first LLL reduction of a basis of large entries (a knapsack lattice of numbers of hundreds of bits) runs in floats on rows of `Number`s, and is much of the work at ranks 80 to 100.
- Against fplll's BKZ 1.0 (`fplll -a bkz -b β`), which stops when every block is within `δ` of its minimum, `bkz` takes 0.304 to 3.25 times its time on the bases of [Measured](#measured), the difference mostly in that first reduction and in the exact tours at the end.
- Four paths are reached by no test, because no natural input takes them: the insertion by gcd steps in the floating-point tours (when the vector found has no coordinate ±1), the fresh start after a floating-point LLL fails, the floating-point tours on numbers with an exponent of their own for huge entries, and an insertion in the exact tours (in the tests the exact tours only confirm). An independent check forced each of them, on 200 to 1100 runs, and found every result right; a defect there would not fail a test.

## Integer relations

`pslq(&x, &opts)` looks for an integer relation among real numbers `x₀, …, x_{n−1}` given as `BigFloat`s: integers `m₀, …, m_{n−1}`, not all zero, with `Σ mᵢ·xᵢ = 0` to the precision of the numbers. It finds the minimal polynomial of an algebraic number from its powers, a formula for a constant in terms of others, or proves that no relation of small norm exists.

| Item | What it is |
| --- | --- |
| `pslq(&x, &opts) -> Pslq` | the search, on two or more finite `BigFloat`s of any precisions |
| `pslq_f64(&x, &opts)` | the same on `f64`s, each taken at its 53 bits |
| `PslqOptions` | `max_norm: Option<Number>` (stop once every relation must be longer; `None`), `max_iter: Option<usize>` (`None`: the bound of the analysis), `gamma: f64` (`1.16`), `error_bits: u32` (`8`), with `new()` and `with_*` |
| `Pslq::Relation(m)` | a relation, primitive, its first nonzero entry positive |
| `Pslq::NoneBelow(b)` | no relation of norm below `b`, and `b ≥ max_norm` (or, with no `max_norm`, the precision ran out) |
| `Pslq::PrecisionExhausted(b)` | no relation of norm below `b`, the precision ran out with `b < max_norm` |
| `Pslq::IterationLimit(b)` | no relation of norm below `b`, `max_iter` ran out |
| `relation()`, `bound()` | the relation or the bound, if there is one |

**What a relation means.** Each `xᵢ` of precision `pᵢ`, `2^(eᵢ−1) ≤ |xᵢ| < 2^eᵢ`, stands for some real within `δᵢ = 2^error_bits·2^(eᵢ−pᵢ)`, `2^error_bits` units in its last place; `m` is a relation when some reals in those intervals satisfy it, which is when `|Σ mᵢ·xᵢ| ≤ Σ |mᵢ|·δᵢ`, and that is what `Relation(m)` guarantees, tested exactly. It cannot say more: every `n` numbers of `p` bits have relations of norm about `2^(p/n)` to that precision (Dirichlet), so a relation found is evidence of a true one only when it is far shorter than that, and the more so the more precision is left over (a degree-35 polynomial with coefficients of 22 bits needs about 36·22 ≈ 800 bits of the powers; at 850 bits PSLQ finds a relation to the precision with coefficients of 32 bits, at 900 the minimal polynomial). With `max_norm` set well below `2^(p/n)` a search on numbers with no relation ends with `NoneBelow`; with no `max_norm` it ends with such a relation to the precision, as PARI's `lindep` always returns a vector. Each number has its own precision: a zero is exact (its unit vector is a relation, returned at once), and an integer is no more exact than its precision, so `BigFloat::from(1)`, of 32 bits, beside numbers of 1000 bits limits the search to 32 bits — give it as `BigFloat::from(1).with_prec(1000)`. Every bound is proved: no integer vector of Euclidean norm below it is a relation of any reals within the intervals.

A relation that turns up is returned whatever its norm, also above `max_norm`: the shortest of the columns of `B` that pass the test when the first one does, a relation to the precision and not always the shortest relation. Which one follows the rounding of `H`: the same input and options give the same answer with the same version of the crate, but a relation to the precision (one not unique up to sign) is not promised to equal another implementation's or another version's. This version's, with its iterations in doubles, matched 0.2.45's on all 2,320 ordinary inputs of the audit of the speed pass and on 1,689 of 1,700 whose magnitudes spread over ±1500 bits; the 11 others are valid relations that differ in the huge coefficient of a tiny number. Of `1, 2^−200, 3·2^−200` at 53 bits, the first reduction of `H` already makes a column `(1, −a, −b)` with `a ≈ 2^200/10` and `b ≈ 3·2^200/10`. It is no exact relation (`a + 3b` misses `2^200` by about `2^76`, a residual of about `2^−123`), but the tolerance `Σ |mᵢ|·δᵢ` is about `2^−43` (1 at 53 bits contributes `2^−44`, and so do entries near `2^199` times the errors `2^−243` of the small numbers), so the search returns it before its first iteration. With exact detection (a relation only when `yⱼ = 0`), PSLQ goes on and finds `(0, 3, −1)` at the first iteration; 1 at 300 bits, or `error_bits = 0`, does not change the answer here, since the small numbers' own errors set the tolerance.

### Method

PSLQ, by Ferguson and Bailey (*A polynomial time, numerically stable integer relation algorithm*, RNR-91-032, 1992), as Ferguson, Bailey and Arno state and analyse it (*Analysis of PSLQ, an integer relation finding algorithm*, Math. Comp. 68, 1999):

- **The numbers made integers.** `xᵢ = mᵢ·2^qᵢ` becomes `Xᵢ = mᵢ·2^(qᵢ − L)`, `L = min qᵢ`, exactly; the span `s` is the bits from `2^L` to the top of the largest. The vector `y = X·B` is then held exactly in integers, as are `A` and `B = A⁻¹`, and a relation is detected by the exact test above on the entry of `y` of each column of `B` (a column whose `y` has too many bits is passed over first by its length alone). Of the columns that pass, the shortest is returned, checked once more from the numbers.
- **`H` in fixed point** of `w = s + 64 + 2·b` fraction bits, `b` the bits of `n`, the only rounded quantity: `Hⱼⱼ = sⱼ₊₁/sⱼ`, `Hᵢⱼ = −XᵢXⱼ/(sⱼsⱼ₊₁)` with `sⱼ² = Σ_{k≥j} Xₖ²`, reduced in Hermite's way; each iteration exchanges the rows `r`, `r + 1` that maximise `γʳ·|Hᵣᵣ|`, rotates the corner `Hᵣ,ᵣ₊₁` away, and reduces the rows below `r` by the columns up to `r + 1`, each multiplier `t = nint(Hᵢⱼ/Hⱼⱼ)` applied exactly to `y`, `A` and `B`. `γ` must exceed `√(4/3)`, for which the analysis bounds the iterations that find a relation of norm `M` by `C(n, 2)·log(γ^(n−1)·M)/log τ`, `1/τ² = 1/γ² + 1/4`; the default `max_iter` is that bound for `M = 2^w`. A larger `γ` lets the search settle for longer relations: in a probe at 1000 bits, `γ` from 1.1548 to 1.3 found the degree-35 polynomial of the tests below, 1.5 and 2 only relations to the precision; 1.16 found it at 900 bits too.
- **Blocks in doubles** (from `n = 4`; the two levels of the multi-level PSLQ of Bailey and Broadhurst, *Parallel integer relation detection: techniques and applications*, Math. Comp. 70, 2001). The iterations run on `H/2^w` and `y` taken to doubles, with the transforms `A'` and `B'` of the block in machine words: the same exchanges, rotations and reductions, the multipliers rounded from the doubles. A block ends before an iteration at which the exact search could stop — an entry of `y` (a changed one, the dot product of the first `y` with its column of `B'`) that its error bound cannot tell from the tolerance of a relation, the floating bound within half a bit of a proof, the precision test within two bits, `A` and `B` taken at their largest — and once an entry of `A'` or `B'` has passed `2^26`, about the bits the doubles of `H` lose; at most 4096 iterations. The exact state then takes it: `y ← y·B'`, `B ← B·B'`, `A ← A'·A` and `H ← A'·H`, all exactly. The rotations are not carried over: `H` is then `A·H₀` times an orthogonal matrix, with the Gram matrix of its rows, all that PSLQ reads of it, unchanged; the next block starts from its lower factor in doubles (Householder reflections), and the exact lower factor, by reflections rounded at `w` bits (the rotations' factor up to the signs of its columns), is made only for exact iterations and where a test of the stops is within those margins. An iteration that the doubles cannot finish (a multiplier or an entry beyond `2^31`, a zero divisor) has its other reductions made exactly; one whose rotation they cannot make is left to the exact search, which makes the iterations after a block of fewer than 16 for the next 64. The doubles only steer: every multiplier is an integer and every step on `y`, `A` and `B` exact, so the answers are tested and proved as before. The degree-35 polynomial takes 31 blocks of about 500 iterations and one exact lower form (after an iteration cut short); on the 1,043 inputs of a comparison with the exact iterations of the previous version (the bench's three largest polynomials and 40 random vectors, and 1,000 random vectors of 3 to 24 numbers at 64 to 992 bits, two thirds with a planted relation, a quarter with a `max_norm`) every answer was the same; on inputs whose magnitudes spread over many bits a few relations to the precision differ from 0.2.45's (above).
- **The bound.** Ferguson, Bailey and Arno show that every relation has norm at least `1/max |Hⱼⱼ|`. The bound returned is proved from the exact `A` instead of read off the rounded `H`: for a relation `m` of reals `ξ` within the intervals and `j` the first row of `A` with `⟨m, aⱼ⟩ ≠ 0`, any real `c` with `aⱼ = c₀·X + Σ_{i<j} cᵢ·aᵢ + rⱼ` gives `1 ≤ |⟨m, aⱼ⟩| ≤ |m|·(|c₀|·|δ| + |rⱼ|)`, since `|⟨m, X⟩| = |⟨m, X − ξ⟩| ≤ |m|·|δ|`. The `c` are the coefficients of the projections of the Gram–Schmidt orthogonalisation of `X, a₀, …, a_{n−2}` (modified Gram–Schmidt, twice, at `w + 64` bits), taken as the dyadic rationals they are, and `rⱼ`, its norm and `|δ|` are computed from them in integers, rounded up; the last row needs nothing, since a relation orthogonal to all the others is a multiple of the last column of `B`, which the search has tested. So the bound is `1/max_j(|c₀|·|δ| + |rⱼ|)` rounded down, at 64 bits, and at least 1; it is the floating `1/max |Hⱼⱼ|` but for the errors of the numbers, which lower it only when the precision is nearly spent. It is proved when the floating bound passes `max_norm` (and again each time it doubles, should the proof fall short) and when the search stops.
- **Stopping.** At a relation; when the proved bound reaches `max_norm`; when `A` or `B` has an entry within 20 bits of the largest `|Hⱼⱼ|` in fixed point, where the rounding errors that the reductions multiply by `A` reach the diagonal, or a divisor of `H` is zero (the precision is exhausted); or after `max_iter` iterations. The guard bits are chosen so that relations to the precision of the numbers turn up before `H` fails; `PrecisionExhausted` is for the arithmetic of `H` failing first, which no input the tests tried did.

**Cost.** An iteration in a block takes `O(n²)` operations on doubles and words; a block's refresh `O(n³)` operations of words on integers of `w` bits, and `O(n³)` on doubles for the next lower factor (the exact one, `O(n³)` products of integers of `w` bits, only where an exact step needs it). An exact iteration takes `O(n²)` multipliers, most of them seen to be zero from the lengths of the two entries, and `O(n)` operations on the entries of `y`, `A` and `B` and `O(j)` on those of `H` for each nonzero one, all on integers of about `s + 64` bits for `y` and `H` and of the size of the multipliers so far for `A` and `B`, each `z ← z ± t·w` done in place, in a machine word while the entries fit and by one pass of a word times the limbs beyond, without allocating; the multiplier itself from the top bits of the two entries, checked by its remainder in place. The number of iterations is far below the bound above in practice (the reference took 15,618 for the degree-35 polynomial, and the search here returns its relation, in 31 blocks). Each proof of the bound is `O(n³)` operations at `w + 64` bits. There is no third level (double-doubles, or a lower multiprecision) and no multipair variant yet.

### Example

```rust
use a::math::float::BigFloat;
use a::math::lattice::{Pslq, PslqOptions, pslq};
use a::math::numbers::Number;

// The minimal polynomial of √2 + √3 from its powers, to 200 bits (1 given at 200 bits too).
let p = 200;
let a = BigFloat::from(2).with_prec(p).sqrt() + BigFloat::from(3).with_prec(p).sqrt();
let x: Vec<BigFloat> = (0..5).map(|k| a.powi(k)).collect();
let m = pslq(&x, &PslqOptions::new());
assert_eq!(m, Pslq::Relation([1, 0, -10, 0, 1].map(Number::from).to_vec())); // x⁴ − 10x² + 1

// Machin's formula π/4 = 4·atan(1/5) − atan(1/239).
let one = BigFloat::from(1).with_prec(p);
let x = [BigFloat::pi(p) / 4u32, (one.clone() / 5u32).atan(), (one / 239u32).atan()];
assert_eq!(pslq(&x, &PslqOptions::new()).relation().unwrap(), [1, -4, 1].map(Number::from));

// π, e and 1 have no relation of norm below 10⁹: a proved bound, not a failure to find one.
let x = [BigFloat::pi(p), BigFloat::e(p), BigFloat::from(1).with_prec(p)];
let opts = PslqOptions::new().with_max_norm(Some(Number::from(1_000_000_000)));
let Pslq::NoneBelow(bound) = pslq(&x, &opts) else { panic!() };
assert!(bound >= 1_000_000_000);
```

### Verified

By `lib/tests/math/lattice/pslq.rs`, against `lib/tests/math/lattice/data/pslq_expected.txt`, which `data/pslq_gen.py` writes with mpmath 1.3 and PARI/GP 2.19. The inputs are computed in the test with the crate's correctly rounded functions (each at `p + 64` bits, rounded once to `p`), and in the script with mpmath the same way; the test checks that its inputs are the script's, by the low 64 bits of every mantissa.

- **Against a reference PSLQ** written out in Python from the paper (the same steps and rules, but `H` in mpmath floats of twice the precision instead of fixed point, and the bound from a Gram–Schmidt orthogonalisation at three times the precision): on 16 of the 17 cases the same answer — the same relation, or a bound equal to 10⁻¹² relative — and on the 17th, `1, 2^−200, 3·2^−200`, a relation whose entries near `2^200/10` agree to `2^−100` relative (their low bits follow the rounding of `H`). They are the minimal polynomials of `√2 + √3` (degree 4, 200 bits, 21 iterations), `∛3 + ⁴√2` (degree 12, 600 bits, 740), `⁴√3 − ⁴√2` (degree 16, 400 bits, 1,273) and `⁵√2 + ⁷√3` (degree 35, 1000 bits, 15,618, the polynomial PARI's `polresultant((x − y)⁷ − 3, y⁵ − 2, y)` gives, coefficients up to 2,785,860); Machin's formula and the four-term one `π/4 = 12·acot 49 + 32·acot 57 − 5·acot 239 + 12·acot 110443`; the formula of Bailey, Borwein and Plouffe, `π = 4·S₁ − 2·S₄ − S₅ − S₆` for `Sⱼ = Σ 16⁻ᵏ/(8k + j)`; `31·ln 2 − 17·ln 3 + 11·ln 5 + 4·ln 7` against its value; π and `acot 2, …, acot 10`, which have several relations (the same one is found, `acot 2 = acot 3 + acot 7`); `1, 2^−200, 3·2^−200`; the bounds that rule out relations of norm below 10¹² among `1, π, ζ(3)` (proved 1.6·10¹²), below 10²⁰ between `ζ(3)` and `π³` (2.8·10²⁰), below 10⁶ among `1, γ, …, γ⁶` at 700 bits (1.15·10⁶, mpmath's documented check that Euler's constant is no root of such a polynomial), below 10⁸ among `1, π, e, ln 2, γ, G` (1.04·10⁸), and below 10⁶ between π and `e` (1.395·10⁶); and the relations to the precision that `π, e` (norm 2⁹⁵) and `π, e, 1` (2⁶⁴) have at 200 bits, where the search is the same to the last iteration (PARI's `lindep` gives other ones of the same kind).
- **Against mpmath's `pslq` and `findpoly` and PARI's `lindep`** (an LLL method) at the same precision, on the eight cases whose relation is unique up to sign (the minimal polynomials, the Machin formulas, BBP, the logarithms): the same relation from `pslq` and `lindep` on all eight and from `findpoly` on the polynomials of degree 4, 12 and 16 (it tries every degree from 1, a PSLQ each, too slow at degree 35).
- **Every relation checked independently** of the crate's own test, by signs of exact dot products (`BigFloat::dot_round`): normalised, primitive, `|Σ mᵢ·xᵢ| ≤ Σ |mᵢ|·δᵢ`. On 1,000 random vectors of 2 to 5 numbers, half with a planted relation with coefficients in `−5..=5`, half of one precision from 12 to 90 bits and half with a precision from 8 to 120 bits for each number, a third with exponents spread over ±40 bits, at random `error_bits`, `γ` (1.16, 1.155, 1.5, 4), `max_norm` and, for a third, `max_iter` below 25: every relation holds, every `NoneBelow` reaches `max_norm`, and every bound below 64, 12, 7 and 4 for 2, 3, 4 and 5 numbers is checked by enumerating every integer vector of smaller norm: none is a relation (753 relations, 234 bounds, 106 enumerated, 51 of them of mixed precisions and 48 `IterationLimit` bounds proved in mid-search). An ignored test does the same on 20,000 vectors, enumerating below 96, 20, 10 and 7: 15,090 relations, 4,684 bounds, 2,659 enumerated (1,234 of mixed precisions, 1,224 `IterationLimit`), no failure.
- **The bound never passes a relation**: for `1, α, …, α¹²` (the only relations multiples of the minimal polynomial, of norm 644.6), `max_norm` 1, 2, 5, …, 200 gives `NoneBelow` with the bounds 1.00, 2.09, 5.11, 11.3, 21.4, 51.8, 104, 217, growing with `max_norm` and below 644.6, and 400 to 1000 the polynomial.
- **Edge cases**: zeros of either sign (their unit vector); numbers whose errors reach zero (5 at 3 bits; 4 at 9 bits, exactly 2⁸ ulps; every number at `error_bits = u32::MAX`), and 3 beside 4 at 20 bits (`4, −3`); integers (`6, −4` gives `2, 3`; `6, 10, 15`; `2²⁰⁰ + 1, 2²⁰⁰ + 3, 1`); exponents of `±2⁴⁰` to `±2⁶¹` (the relation of the mantissas); `1, 2^−1000` and `π, π·2^−300`; mixed precisions (`π` at 53 bits is `π` at 200 within its errors; `1` at 32 bits beside `√2` at 200 gives a rational approximation of `√2`, at 200 bits none below 10¹⁵); `error_bits` from 0 to 10 on `π, π + 2^−90` (the relation `(1, −1)` from 7 on, exactly where it holds); `f64` inputs; `γ` from 1.155 to 2; `max_iter` 0 and 20; `max_norm` 0 and 1 (the bound of the first reduction, and `6, −4` stopped before its relation); `γ = f64::MAX`; and the panics on fewer than two numbers, NaN, infinities, `γ` at the `f64` nearest `√(4/3)` (below it), `γ` NaN or infinite (as not finite), a negative `max_norm`, and spans beyond `2⁴⁰` bits (also `1` at 53 bits beside `2^−(2⁴⁰−100)` at 200: the span runs to the smallest last place).

To reproduce: `python3 lib/tests/math/lattice/data/pslq_gen.py lib/tests/math/lattice/data/pslq_expected.txt` (some minutes, most of them the reference and mpmath on degree 35), then `cargo test --test math -- lattice::pslq`, and `cargo test --release --test math -- --ignored lattice::pslq` for the sweep of 20,000 vectors.

### Limits

- The answer is about the numbers as given: a relation is one to their precision, never a proof that the reals they approximate satisfy it, and a bound holds for the reals within `2^error_bits` ulps of them, no further. Numbers computed with errors of more ulps need a larger `error_bits`, or the true relation fails the test.
- PSLQ finds a relation, not the shortest, and returns it even above `max_norm`.
- An integer given at a low precision limits the search to that precision (above).
- The cost follows the span of the bits, not the precision of each number: numbers of very different sizes, or of very different precisions, are held as integers of the whole span, and the span, from the top of the largest to the smallest last place of any of them, is limited to `2⁴⁰` bits (a panic beyond).
- `PrecisionExhausted` was not seen, nor `NoneBelow` from the precision with no `max_norm`: in every test and probe, and in some 20,000 runs of an independent audit, the precision ran out by a relation to the precision turning up first. The path (the arithmetic of `H` failing before that) is kept and untested.
- No multipair variant. The times are under [Measured](#measured).

## Measured

Measured on an Apple M4 Max (macOS 27.2) on 2026-11-26, in the final run of the release, one area after another with nothing else running, by the harness `bench/lattice` of the repository (`bench/README.md` has the method): one thread on each side; the crate as a user builds it, with Rust 1.100 (stable) and the default release profile, without LTO or `target-cpu`; the references in the same process through C and C++ shims compiled at `-O3 -mcpu=native` on Homebrew's fplll 5.5.0, FLINT 3.6.0, NTL 11.6.0 and PARI/GP 2.19.1, and mpmath 1.3.0 (its pure-Python backend) in a process of its own, whose pipe round trip, 16 µs, is in its time. Ours and the references alternate sample by sample, and each figure is the median of 15 samples of at least 1 ms (fewer, down to 3, where a sample takes a minute); every reference's output was checked against ours before the clock started — a reduced basis by the exact test of `(δ, η)`, the norm of a shortest vector, the distance of a closest one, the count of short vectors, the relation. The bases: knapsack lattices `[aᵢ | eᵢ]` of `n` numbers of the given bits, uniform random square bases, NTRU-like and q-ary lattices of rank `n` and modulus `q`, and Goldstein and Mayer's lattices; LLL at `δ = 0.99`, `η = 0.51`, every reduction computing `U`. The ratio is our time over the reference's: below 1, ours is faster.

A reference that does less is left out of "the fastest": one that computes no `U` (ours always does), one that does not prove its result (fplll's heuristic LLL and its fast enumeration, NTL's floating-point LLL and BKZ, PARI's `lindep`, an LLL method that proves no bound), one that reduces less (FLINT's LLL to `η = 0.51` where `lll` reduces to `η = 1/2`; fplll's BKZ 1.0, which stops when every block is within `δ` of its minimum, and its BKZ 2.0, which prunes and aborts early), one that returns its list unsorted and in coordinates (PARI's `qfminim`), and one whose answer is wrong (PARI's `qfcvp` on Goldstein and Mayer's lattices, whose floating-point search returns a farther vector). Where such a reference is the comparison that matters it is given below, labelled.

### LLL reduction

`lll_fp`, against the fastest reference that computes `U` and is proved — FLINT's `fmpz_lll`, fplll's wrapper and proved method, PARI's `qflll` with and without FLATTER — and, as a second figure, the fastest that computes no `U`:

| Basis | Ours | Fastest reference with U | Its time | Ratio | Fastest without U (does less) | Ratio |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| knapsack, n = 20, 200 bits | 2.80 ms | FLINT fmpz_lll | 1.15 ms | 2.44 | FLINT fmpz_lll 896 µs | 3.12 |
| knapsack, n = 40, 400 bits | 59.0 ms | FLINT fmpz_lll | 26.1 ms | 2.26 | FLINT fmpz_lll 21.4 ms | 2.76 |
| knapsack, n = 60, 600 bits | 355 ms | FLINT fmpz_lll | 184 ms | 1.93 | FLINT fmpz_lll 137 ms | 2.60 |
| knapsack, n = 80, 800 bits | 1.32 s | FLINT fmpz_lll | 713 ms | 1.85 | FLINT fmpz_lll 491 ms | 2.69 |
| knapsack, n = 100, 100 bits | 56.7 ms | FLINT fmpz_lll | 18.5 ms | 3.06 | FLINT fmpz_lll 13.6 ms | 4.18 |
| knapsack, n = 50, 1000 bits | 413 ms | FLINT fmpz_lll | 175 ms | 2.36 | FLINT fmpz_lll 134 ms | 3.07 |
| knapsack, n = 100, 1000 bits | 3.87 s | PARI qflll (FLATTER), T and x·T | 1.71 s | 2.26 | FLINT fmpz_lll 1.22 s | 3.16 |
| NTRU-like, n = 40, q of 10 bits | 5.86 ms | FLINT fmpz_lll | 8.95 ms | 0.656 | fplll lll wrapper 4.69 ms | 1.25 |
| NTRU-like, n = 80, q of 12 bits | 98.3 ms | FLINT fmpz_lll | 271 ms | 0.363 | fplll lll wrapper 83.4 ms | 1.18 |
| NTRU-like, n = 80, q of 20 bits | 171 ms | FLINT fmpz_lll | 453 ms | 0.378 | fplll lll wrapper 138 ms | 1.24 |
| NTRU-like, n = 160, q of 14 bits | 1.85 s | FLINT fmpz_lll | 6.38 s | 0.290 | fplll lll wrapper 1.38 s | 1.34 |
| q-ary, n = 40, k = 20, q of 20 bits | 11.5 ms | FLINT fmpz_lll | 17.9 ms | 0.646 | fplll lll wrapper 8.86 ms | 1.30 |
| q-ary, n = 80, k = 40, q of 20 bits | 173 ms | FLINT fmpz_lll | 474 ms | 0.365 | fplll lll wrapper 140 ms | 1.24 |
| q-ary, n = 120, k = 60, q of 30 bits | 6.74 s | PARI qflll (FLATTER), T and x·T | 3.70 s | 1.82 | fplll lll wrapper 1.15 s | 5.84 |
| uniform, n = 20, 10 bits | 116 µs | FLINT fmpz_lll | 116 µs | 1.000 | FLINT fmpz_lll 106 µs | 1.09 |
| uniform, n = 50, 100 bits | 7.19 ms | FLINT fmpz_lll | 2.70 ms | 2.67 | FLINT fmpz_lll 2.51 ms | 2.86 |
| uniform, n = 100, 10 bits | 8.18 ms | FLINT fmpz_lll | 10.5 ms | 0.779 | FLINT fmpz_lll 9.65 ms | 0.848 |
| uniform, n = 100, 100 bits | 41.7 ms | FLINT fmpz_lll | 11.8 ms | 3.54 | FLINT fmpz_lll 11.3 ms | 3.68 |
| uniform, n = 200, 10 bits | 61.4 ms | FLINT fmpz_lll | 75.1 ms | 0.817 | FLINT fmpz_lll 71.4 ms | 0.860 |
| Goldstein-Mayer, n = 40, 400 bits | 61.4 ms | FLINT fmpz_lll | 28.3 ms | 2.17 | FLINT fmpz_lll 22.6 ms | 2.72 |
| Goldstein-Mayer, n = 60, 600 bits | 365 ms | FLINT fmpz_lll | 189 ms | 1.93 | FLINT fmpz_lll 139 ms | 2.62 |
| Goldstein-Mayer, n = 80, 800 bits | 1.35 s | PARI qflll (FLATTER), T and x·T | 708 ms | 1.91 | FLINT fmpz_lll 505 ms | 2.67 |
| Goldstein-Mayer, n = 100, 1000 bits | 3.75 s | PARI qflll (FLATTER), T and x·T | 1.71 s | 2.20 | FLINT fmpz_lll 1.28 s | 2.93 |

`lll`, exact to `η = 1/2`, against NTL's exact integral LLL, the same algorithm, and `is_lll_reduced` against FLINT's test:

| Operation | Basis | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `lll` | knapsack, n = 10, 100 bits | 250 µs | NTL LLL (exact, a/b=99/100), with U | 451 µs | 0.555 |
| `lll` | knapsack, n = 20, 200 bits | 3.32 ms | NTL LLL (exact, a/b=99/100), with U | 8.03 ms | 0.414 |
| `lll` | knapsack, n = 30, 300 bits | 18.5 ms | NTL LLL (exact, a/b=99/100), with U | 49.9 ms | 0.371 |
| `lll` | knapsack, n = 40, 400 bits | 61.8 ms | NTL LLL (exact, a/b=99/100), with U | 169 ms | 0.366 |
| `lll` | uniform, n = 10, 30 bits | 56.3 µs | NTL LLL (exact, a/b=99/100), with U | 50.9 µs | 1.11 |
| `lll` | uniform, n = 20, 10 bits | 290 µs | NTL LLL (exact, a/b=99/100), with U | 547 µs | 0.531 |
| `lll` | uniform, n = 30, 10 bits | 904 µs | NTL LLL (exact, a/b=99/100), with U | 1.64 ms | 0.551 |
| `lll` | NTRU-like, n = 20, q of 10 bits | 544 µs | NTL LLL (exact, a/b=99/100), with U | 2.70 ms | 0.201 |
| `lll` | NTRU-like, n = 40, q of 12 bits | 7.81 ms | NTL LLL (exact, a/b=99/100), with U | 68.7 ms | 0.114 |
| `lll` | Goldstein-Mayer, n = 20, 200 bits | 3.17 ms | NTL LLL (exact, a/b=99/100), with U | 7.89 ms | 0.401 |
| `is_lll_reduced` | lll_fp output of knapsack, n = 40, 400 bits | 119 µs | FLINT fmpz_lll_is_reduced (prec 144) | 365 µs | 0.326 |
| `is_lll_reduced` | lll_fp output of knapsack, n = 100, 100 bits | 2.90 ms | FLINT fmpz_lll_is_reduced (prec 264) | 4.91 ms | 0.591 |
| `is_lll_reduced` | lll_fp output of uniform, n = 50, 100 bits | 1.10 ms | FLINT fmpz_lll_is_reduced (prec 164) | 764 µs | 1.45 |
| `is_lll_reduced` | lll_fp output of Goldstein-Mayer, n = 80, 800 bits | 1.64 ms | FLINT fmpz_lll_is_reduced (prec 224) | 2.74 ms | 0.599 |
| `is_lll_reduced` | lll_fp output of q-ary, n = 80, k = 40, q of 20 bits | 1.63 ms | FLINT fmpz_lll_is_reduced (prec 224) | 2.78 ms | 0.588 |

Against FLINT's `fmpz_lll`, which reduces to `η = 0.51` only and so does less, `lll` takes 0.753 to 3.62 times its time on these bases.

`deep_lll` has no reference that does the same work: NTL's `LLL_FP` with deep insertions proves nothing of its result (and its depth is of the first places only). Against it:

| Basis, depth | Ours | Fastest reference (it does less) | Its time | Ratio |
| --- | ---: | --- | ---: | ---: |
| knapsack, n = 16, 160 bits; depth none (anywhere) | 1.37 ms | NTL LLL_FP deep=16, with U | 1.07 ms | 1.28 |
| knapsack, n = 16, 160 bits; depth 5 | 1.38 ms | NTL LLL_FP deep=5, with U | 1.08 ms | 1.28 |
| knapsack, n = 24, 240 bits; depth none (anywhere) | 11.2 ms | NTL LLL_FP deep=24, with U | 9.09 ms | 1.23 |
| knapsack, n = 24, 240 bits; depth 5 | 11.1 ms | NTL LLL_FP deep=5, with U | 7.77 ms | 1.42 |
| knapsack, n = 32, 320 bits; depth none (anywhere) | 66.8 ms | NTL LLL_FP deep=32, with U | 50.6 ms | 1.32 |
| knapsack, n = 32, 320 bits; depth 5 | 46.1 ms | NTL LLL_FP deep=5, with U | 29.6 ms | 1.56 |
| knapsack, n = 40, 400 bits; depth none (anywhere) | 309 ms | NTL LLL_FP deep=40, with U | 236 ms | 1.31 |
| knapsack, n = 40, 400 bits; depth 5 | 167 ms | NTL LLL_FP deep=5, with U | 91.5 ms | 1.82 |

### Shortest and closest vectors

Both sides include the LLL reduction of the basis:

| Operation | Basis | Ours | Fastest reference | Its time | Ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `shortest_vector` | Goldstein-Mayer, n = 20, 200 bits | 3.46 ms | fplll LLL + shortest_vector (SVPM_PROVED) | 2.08 ms | 1.67 |
| `shortest_vector` | Goldstein-Mayer, n = 30, 300 bits | 18.3 ms | fplll LLL + shortest_vector (SVPM_PROVED) | 10.6 ms | 1.73 |
| `shortest_vector` | Goldstein-Mayer, n = 36, 360 bits | 49.2 ms | fplll LLL + shortest_vector (SVPM_PROVED) | 26.6 ms | 1.85 |
| `shortest_vector` | Goldstein-Mayer, n = 40, 400 bits | 416 ms | fplll LLL + shortest_vector (SVPM_PROVED) | 393 ms | 1.06 |
| `shortest_vector` | Goldstein-Mayer, n = 44, 440 bits | 1.38 s | fplll LLL + shortest_vector (SVPM_PROVED) | 2.50 s | 0.551 |
| `shortest_vector` | knapsack, n = 30, 300 bits | 18.5 ms | fplll LLL + shortest_vector (SVPM_PROVED) | 11.0 ms | 1.68 |
| `shortest_vector` | uniform, n = 30, 20 bits | 1.25 ms | fplll LLL + shortest_vector (SVPM_PROVED) | 861 µs | 1.45 |
| `shortest_vector` | q-ary, n = 40, k = 20, q of 12 bits | 65.3 ms | fplll LLL + shortest_vector (SVPM_PROVED) | 87.3 ms | 0.748 |
| `closest_vector` | uniform, n = 20, 20 bits; random target of 22 bits | 530 µs | PARI solve + Gram + qfcvp | 2.62 ms | 0.203 |
| `closest_vector` | uniform, n = 30, 20 bits; random target of 22 bits | 1.62 ms | PARI solve + Gram + qfcvp | 14.7 ms | 0.110 |

`closest_vector` on Goldstein and Mayer's lattices has no reference that does the same work: fplll's `CVPM_FAST` does not prove its answer, and PARI's `qfcvp` returns a farther vector there. Against fplll:

| Basis, target | Ours | Fastest reference (it does less) | Its time | Ratio |
| --- | ---: | --- | ---: | ---: |
| Goldstein-Mayer, n = 20, 200 bits; random target | 3.50 ms | fplll LLL + closest_vector (CVPM_FAST) | 2.10 ms | 1.67 |
| Goldstein-Mayer, n = 30, 300 bits; random target | 18.9 ms | fplll LLL + closest_vector (CVPM_FAST) | 11.1 ms | 1.71 |
| Goldstein-Mayer, n = 36, 360 bits; random target | 77.9 ms | fplll LLL + closest_vector (CVPM_FAST) | 54.9 ms | 1.42 |
| Goldstein-Mayer, n = 40, 400 bits; random target | 205 ms | fplll LLL + closest_vector (CVPM_FAST) | 148 ms | 1.39 |

`short_vectors` against PARI's `qfminim` on the Gram matrix, which returns its vectors unsorted and in coordinates where these are sorted and in the ambient space:

| Basis, bound | Ours | Fastest reference (it does less) | Its time | Ratio |
| --- | ---: | --- | ---: | ---: |
| uniform, n = 12, 6 bits; bound 3·λ1² | 6.12 ms | PARI Gram + qfminim(G, B) | 1.19 ms | 5.14 |
| uniform, n = 16, 6 bits; bound 2·λ1² | 303 µs | PARI Gram + qfminim(G, B) | 255 µs | 1.19 |
| uniform, n = 20, 8 bits; bound 2·λ1² | 6.01 ms | PARI Gram + qfminim(G, B) | 1.90 ms | 3.16 |
| Goldstein-Mayer, n = 24, 240 bits; bound 2·λ1² | 177 ms | PARI Gram + qfminim(G, B) | 64.3 ms | 2.75 |
| Goldstein-Mayer, n = 30, 300 bits; bound 2·λ1² | 925 ms | PARI Gram + qfminim(G, B) | 456 ms | 2.03 |

### BKZ

`bkz` has no reference that does the same work: its exact end makes every block's first Gram–Schmidt vector a shortest vector of the block, where fplll's BKZ 1.0 (without pruning) stops when every block is within `δ` of it and NTL's BKZ proves nothing of its result (`BKZ_XD` where `BKZ_FP` stops on entries as long as these). Against both, the time of the first LLL reduction included on every side:

| Basis, block | Ours | fplll BKZ 1.0 | Ratio | NTL BKZ | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| knapsack, n = 40, 400 bits; β = 10 | 65.9 ms | 61.9 ms | 1.07 | 54.7 ms (BKZ_FP) | 1.21 |
| knapsack, n = 40, 400 bits; β = 20 | 74.1 ms | 63.9 ms | 1.16 | 66.5 ms (BKZ_FP) | 1.11 |
| knapsack, n = 40, 400 bits; β = 30 | 123 ms | 99.1 ms | 1.24 | 105 ms (BKZ_FP) | 1.16 |
| knapsack, n = 60, 600 bits; β = 10 | 408 ms | 363 ms | 1.12 | 812 ms (BKZ_XD) | 0.503 |
| knapsack, n = 60, 600 bits; β = 20 | 510 ms | 425 ms | 1.20 | 1.10 s (BKZ_XD) | 0.462 |
| knapsack, n = 80, 800 bits; β = 10 | 1.51 s | 1.22 s | 1.24 | 2.82 s (BKZ_XD) | 0.537 |
| knapsack, n = 80, 800 bits; β = 20 | 2.17 s | 1.31 s | 1.65 | 3.61 s (BKZ_XD) | 0.599 |
| q-ary, n = 60, k = 30, q of 20 bits; β = 10 | 104 ms | 342 ms | 0.304 | 254 ms (BKZ_FP) | 0.409 |
| q-ary, n = 60, k = 30, q of 20 bits; β = 20 | 179 ms | 343 ms | 0.522 | 281 ms (BKZ_FP) | 0.637 |
| q-ary, n = 60, k = 30, q of 20 bits; β = 30 | 4.09 s | 1.26 s | 3.25 | 1.92 s (BKZ_FP) | 2.13 |
| q-ary, n = 80, k = 40, q of 20 bits; β = 10 | 386 ms | 1.17 s | 0.331 | 959 ms (BKZ_FP) | 0.402 |
| q-ary, n = 80, k = 40, q of 20 bits; β = 20 | 1.95 s | 1.27 s | 1.53 | 1.40 s (BKZ_FP) | 1.39 |
| NTRU-like, n = 60, q of 12 bits; β = 10 | 75.1 ms | 194 ms | 0.387 | 167 ms (BKZ_FP) | 0.451 |
| NTRU-like, n = 60, q of 12 bits; β = 20 | 234 ms | 251 ms | 0.935 | 262 ms (BKZ_FP) | 0.894 |
| Goldstein-Mayer, n = 60, 600 bits; β = 10 | 421 ms | 355 ms | 1.19 | 738 ms (BKZ_XD) | 0.571 |
| Goldstein-Mayer, n = 60, 600 bits; β = 20 | 614 ms | 386 ms | 1.59 | 1.59 s (BKZ_XD) | 0.386 |
| Goldstein-Mayer, n = 60, 600 bits; β = 30 | 2.46 s | 2.03 s | 1.21 | 17.7 s (BKZ_XD) | 0.139 |
| uniform, n = 60, 20 bits; β = 10 | 20.5 ms | 15.9 ms | 1.29 | 8.54 ms (BKZ_FP) | 2.40 |
| uniform, n = 60, 20 bits; β = 20 | 31.2 ms | 15.6 ms | 2.00 | 11.4 ms (BKZ_FP) | 2.74 |

### Integer relations

`pslq` against mpmath's `pslq` at the same precision, and, as a second figure, PARI's `lindep`, which proves no bound (its rows are the cases with a relation; it found the same one):

| Operation | Numbers | Ours | mpmath | Ratio | PARI lindep (proves no bound) | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pslq` | minimal polynomial of √2+√3 (n = 5, 200 bits) | 13.1 µs | 523 µs | 0.0251 | 138 µs | 0.0955 |
| `pslq` | minimal polynomial of √2+∛3 (n = 7, 128 bits) | 30.0 µs | 4.40 ms | 0.00682 | 339 µs | 0.0884 |
| `pslq` | minimal polynomial of √2+∛3 (n = 7, 256 bits) | 34.4 µs | 5.41 ms | 0.00636 | 725 µs | 0.0475 |
| `pslq` | minimal polynomial of √2+∛3 (n = 7, 512 bits) | 40.1 µs | 8.56 ms | 0.00469 | 1.41 ms | 0.0285 |
| `pslq` | minimal polynomial of √2+∛3 (n = 7, 1024 bits) | 56.4 µs | 16.8 ms | 0.00336 | 2.42 ms | 0.0233 |
| `pslq` | minimal polynomial of √2+∛3 (n = 7, 2048 bits) | 101 µs | 40.8 ms | 0.00248 | 4.30 ms | 0.0235 |
| `pslq` | minimal polynomial of ∛3+⁴√2 (n = 13, 600 bits) | 453 µs | 265 ms | 0.00171 | 4.66 ms | 0.0973 |
| `pslq` | minimal polynomial of ⁴√3−⁴√2 (n = 17, 400 bits) | 1.31 ms | 707 ms | 0.00185 | 5.91 ms | 0.221 |
| `pslq` | minimal polynomial of ⁵√2+⁷√3 (n = 36, 1000 bits) | 64.9 ms | – | – | 75.9 ms | 0.855 |
| `pslq` | Machin π/4, atan 1/5, atan 1/239 (n = 3, 200 bits) | 3.94 µs | 44.8 µs | 0.0880 | 40.1 µs | 0.0984 |
| `pslq` | four-term Machin (n = 5, 200 bits) | 12.2 µs | 417 µs | 0.0292 | 142 µs | 0.0856 |
| `pslq` | logarithms ln 2, ln 3, ln 5, ln 7 and a combination (n = 5, 300 bits) | 13.8 µs | 918 µs | 0.0150 | 165 µs | 0.0840 |
| `pslq` | no relation below 10^12 among 1, π, ζ(3) (n = 3, 300 bits) | 114 µs | 854 µs | 0.134 | – | – |
| `pslq` | no relation below 10^8 among 1, π, e, ln 2, γ, G (n = 6, 400 bits) | 233 µs | 22.5 ms | 0.0104 | – | – |
| `pslq` | no polynomial below 10^6 for γ, degree 6 (n = 7, 700 bits) | 468 µs | 49.4 ms | 0.00947 | – | – |
| `pslq_f64` | Machin π/4, atan 1/5, atan 1/239 (f64) | 2.58 µs | 35.7 µs | 0.0723 | 26.8 µs | 0.0964 |
| `pslq_f64` | ln 2, ln 3, ln 6 (f64) | 2.70 µs | 31.7 µs | 0.0852 | 20.6 µs | 0.131 |
| `pslq_f64` | 1, φ, φ² (f64) | 2.58 µs | 31.9 µs | 0.0808 | 20.2 µs | 0.128 |
| `pslq_f64` | no relation below 100 between π, e, 1 (f64) | 24.7 µs | 216 µs | 0.114 | – | – |

The relation of degree 35 (`n = 36`) has no reference that does the same work: mpmath takes minutes there and was not timed.

### Where the references are faster

- **`lll_fp`** on the knapsack lattices, the uniform bases of 100 bits and Goldstein and Mayer's (1.85 to 3.54 times FLINT's `fmpz_lll` with `U`, or PARI's `qflll` with FLATTER), against 0.290 to 0.817 of FLINT's time on the NTRU-like and q-ary lattices up to 80 rows and the uniform bases of 10 bits from 100 rows (1.000 at 20): on bases of long entries the Gram matrix and `U` are numbers of several limbs, updated exactly at every size reduction, where FLINT runs on a floating-point orthogonalisation of the basis and keeps only the basis and `U` exact; and the result is proved at the end. Of fplll's speed-ups, an exponent for each row and the early reduction of the first rows are not in. On the q-ary lattice of 120 rows (1.82 times) PARI's FLATTER reduces by recursive blocks, which this crate does not do.
- **`lll`** on the uniform basis of 10 rows and 30 bits (1.11 times NTL's): the fixed cost of the exact reduction on a small basis; on the other nine it takes 0.114 to 0.555 of NTL's time.
- **`is_lll_reduced`** on the uniform basis of 50 rows and 100 bits (1.45 times FLINT's): the exact Gram matrix is formed first, `n²·m` products of the entries, where FLINT's first stage works on the basis in doubles.
- **`shortest_vector`** (1.06 to 1.85 times fplll's proved enumeration on six of the eight bases, up to rank 40; 0.748 on the q-ary lattice of rank 40): below rank 40 the LLL reduction that comes first is a large part of the time, and its causes are those of `lll_fp`; at rank 44, where the search itself dominates, ours takes 0.551 of fplll's time.
- Against the references that do less: `deep_lll` 1.23 to 1.82 times NTL's (the causes of `lll_fp`, and the proof); `closest_vector` on Goldstein and Mayer's lattices 1.39 to 1.71 times fplll's unproved search; `short_vectors` 1.19 to 5.14 times PARI's unsorted list in coordinates; `bkz` 1.07 to 3.25 times fplll's BKZ 1.0 on 14 of the 19 bases, mostly in the first reduction and in the exact tours at the end, which the exact block condition asks for (0.304 to 0.935 on the other five), and 1.11 to 2.74 times NTL's on 7 (0.139 to 0.894 on the other twelve).

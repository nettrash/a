# A

[![rust workflow](https://github.com/nettrash/a/actions/workflows/rust.yml/badge.svg)](https://github.com/nettrash/a/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/nettrash-a.svg)](https://crates.io/crates/nettrash-a)
[![docs.rs](https://img.shields.io/docsrs/nettrash-a)](https://docs.rs/nettrash-a)
[![license](https://img.shields.io/crates/l/nettrash-a.svg)](https://github.com/nettrash/a/blob/HEAD/LICENSE)

Algorithms in pure Rust, with no dependencies:

- **Sorting** (`a::sort`): 23 sorting algorithms by comparison and by key, and a default sort.
- **Integers of any size** (`a::math::numbers`): `Number`, with its arithmetic up to transforms for the longest products, roots, powers modulo a number, greatest common divisors, primes and text in any radix; prime sieves and counting, factorisation up to the quadratic sieve, arithmetic functions, modular arithmetic and discrete logarithms, and integer sequences.
- **Fixed-width integers** (`a::math::uint`): `Uint<N>` and `Int<N>`, unsigned and signed integers of `N` 64-bit limbs on the stack, with the methods of the primitive integers and their behaviour on overflow, text and conversions; Montgomery multiplication, and `pow_mod` modulo any number.
- **Rational numbers** (`a::math::rational`): `Rational`, in lowest terms, with its arithmetic, conversions to and from floats and text, continued fractions, best approximations, rational reconstruction, and the Bernoulli and harmonic numbers.
- **Arbitrary-precision floats** (`a::math::float`): `BigFloat` and `BigComplex`, correctly rounded in four modes as MPFR and GNU MPC, with their arithmetic, roots and powers, conversions and text, the exponential, logarithmic, circular and hyperbolic functions and their inverses, and the constants π, `e`, `ln 2`, `ln 10`, `γ` and Catalan's `G`.
- **Polynomials** (`a::math::poly`): `Polynomial<T>` over the integers, the rationals, floats and complex floats, and `ModPolynomial` modulo any integer, with fast products and division, greatest common divisors and resultants, evaluation at many points and interpolation, roots in double precision, certified or correctly rounded, and factorisation modulo primes and over the integers and the rationals.
- **Fourier transforms** (`a::math::fft`): complex transforms of any length, real transforms, the cosine and sine transforms of types I to IV, transforms of two and three dimensions, and linear and circular convolutions and correlations, on `f32` and `f64`.
- **Special functions** (`a::math::special`): the gamma family, the error functions, Bessel and Airy functions of real order, the zeta function and polylogarithms, Lambert W and the exponential integrals, elliptic integrals, Jacobi elliptic functions and orthogonal polynomials — of `f32`, `f64` and complex arguments, each with a stated ulp bound.
- **Finite fields** (`a::math::field`): prime fields of any size, their extensions and the towers of pairings, binary fields in polynomial and normal bases, and the fields of AES and GCM.
- **Elliptic curves** (`a::math::ec`): Weierstrass, Montgomery and Edwards curves over prime fields, secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519 among them; curves over binary fields, with the 18 of SEC 2; and hashing to curves by the suites of RFC 9380.
- **Pairings** (`a::math::pairing`): the optimal ate pairing on BN254 and BLS12-381, with its groups, their encodings and Ethereum's precompiles, and hashing to `G1` and `G2`.
- **Complex numbers** (`a::math::complex`) and **matrices** (`a::math::matrix`): decompositions, eigenvalues, singular values and functions of matrices of floats and complex numbers; exact determinants, solutions and inverses of matrices of integers and rationals, and the Hermite and Smith normal forms; and matrices over `GF(p)` and `GF(2)`.
- **Quaternions** (`a::math::quaternion`): their algebra, exponentials, logarithms and powers, and rotations: axes and angles, rotation vectors and matrices, slerp and nlerp, shortest arcs, and Euler angles in SciPy's 24 conventions.
- **Dual numbers** (`a::math::dual`): forward-mode automatic differentiation by dual and hyper-dual numbers, which nest: first and second derivatives, higher ones by nesting, gradients, Jacobians and Hessians, in the shapes the numerical methods take.
- **Lattices** (`a::math::lattice`): LLL reduction, exactly and by the L² algorithm, with deep insertions, and BKZ; shortest and closest vectors and short vectors by enumeration, exactly; and integer relations among real numbers by PSLQ.
- **Sparse matrices** (`a::math::sparse`): compressed row, compressed column and coordinate storage of floats and complex numbers, Matrix Market files, fill-reducing orderings, sparse Cholesky, `L·D·Lᴴ`, LU and QR factorisations, Krylov solvers with preconditioners, and a few eigenvalues by the Lanczos and Arnoldi methods.
- **Numerical methods** (`a::math::numerics`): the zeros of a function and of a system, integrals in one and several dimensions and the Gaussian rules, initial value problems for stiff and non-stiff ordinary differential equations with dense output and events, minima with and without bounds, least squares and linear programming, splines, Chebyshev and rational approximation, and numerical differentiation, on `f32` and `f64`.
- **Hash functions** (`a::hash`): SHA-224, SHA-256, SHA-384 and SHA-512, and RFC 9380's `expand_message_xmd`.

Every algorithm has a page with its method, examples and measurements against the reference implementations (GMP, MPFR, GNU MPC, PARI, FLINT, Arb, NTL, fplll, M4RI, primesieve, primecount, GMP-ECM, OpenSSL, mcl, blst, arkworks, malachite, ruint, crypto-bigint, num-dual, glam, faer, nalgebra, SuiteSparse, FFTW, rustfft, Apple's Accelerate, GSL, Boost, SciPy and others); the tables below link to them. The pages of the lattices, of the newest exact linear algebra, of the finite fields, the pairings, the curves over binary fields, the hashing to curves, SHA-2, the fixed-width integers, the quaternions and the dual numbers also say what they were verified against (fplll, FLINT, PARI, NTL, M4RI, GMP, mpmath, sympy, SciPy, OpenSSL, arkworks, the `bls12_381` crate, blst, py_ecc, RustCrypto, ruint, crypto-bigint, bnum, glam, nalgebra, num-dual, and the published test vectors of NIST, the RFCs, Ethereum and gnark-crypto). These pages are timed now as well: the timings of everything v0.2 added come from one run of all its areas on 2026-11-26, those of the other pages from their own runs, and each page gives its conditions. The API reference is on [docs.rs](https://docs.rs/nettrash-a). The crate needs Rust 1.88 (edition 2024). Its fastest paths are, on aarch64, NEON and assembly kernels, among them generated assembly for the prime fields of the curves, the finite fields and the pairings, and the ARMv8 SHA-256 and SHA-512 instructions; on x86_64, assembly for the sums and differences, products and quotients of integers, SSE2 and AVX2 kernels for the Fourier transforms and the SHA extensions for SHA-256; and the binary fields, GHASH and the polynomials modulo 2 multiply by the carry-less products `PMULL` and `PCLMULQDQ`. The instructions are taken where the CPU has them, and everything else is portable Rust.

The harnesses that made the measurements are in [`bench/`](https://github.com/nettrash/a/blob/HEAD/bench/README.md); the timings are of an Apple M4 Max, and the x86_64 paths are tested but were not timed for the pages. The tests run on the portable fallbacks of the SIMD and instruction-set paths with `RUSTFLAGS="--cfg a_baseline"`, which makes every run-time check of a CPU feature report it absent.

## Usage

Add the crate with `cargo add nettrash-a`, or to `Cargo.toml`:

```toml
[dependencies]
nettrash-a = "0.2"
```

Its library is named `a`:

```rust
use a::math::matrix::Matrix;
use a::math::numbers::Number;
use a::sort::sort;

fn main() {
    let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
    sort(&mut v);
    assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let x: Number = "123456789012345678901234567890".parse().unwrap();
    assert_eq!((&x * &x).to_string(), "15241578753238836750495351562536198787501905199875019052100");

    let a: Matrix<f64> = Matrix::from_rows(&[[4.0, 2.0], [2.0, 3.0]]);
    let b = Matrix::from_rows(&[[2.0], [1.0]]);
    let solution = a.solve(&b).unwrap();
    assert!((solution[(0, 0)] - 0.5).abs() < 1e-15 && solution[(1, 0)].abs() < 1e-15);
}
```

## Sorting

Every sort takes a mutable slice and leaves it sorted; the extra memory is what a sort uses besides the slice. [`sort`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/default.md) is the default: use it when there is no reason to prefer a particular algorithm.

### Comparison sorts

These sort any `T: Ord`.

| Sort | Functions | Stable | Best | Average | Worst | Extra memory |
| --- | --- | --- | --- | --- | --- | --- |
| [Default](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/default.md) | `sort` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Bubble](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/bubble.md) | `bubble` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Selection](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/selection.md) | `selection` | no | O(n) | O(n²) | O(n²) | O(1) |
| [Insertion](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/insertion.md) | `insertion` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Cocktail Shaker](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/cocktail_shaker.md) | `cocktail_shaker` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Gnome](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/gnome.md) | `gnome` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Comb](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/comb.md) | `comb` | no | O(n log n) | O(n log n)¹ | O(n²)¹ | O(1) |
| [Merge](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/merge.md) | `merge` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Quick](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/quick.md) | `quick` | no | O(n) | O(n log n) | O(n log n) | O(log n) |
| [Heap](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/heap.md) | `heap` | no | O(n) | O(n log n) | O(n log n) | O(1) |
| [Tim](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/tim.md) | `tim` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Intro](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/intro.md) | `intro` | no | O(n) | O(n log n) | O(n log n) | O(log n) |
| [Shell](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/shell.md) | `shell` | no | O(n) | unknown² | O(n^(4/3))² | O(1) |
| [Drift](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/drift.md) | `drift` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Tree](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/tree.md) | `tree` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Cycle](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/cycle.md) | `cycle` | no | O(n) | O(n²) | O(n²) | O(1) |
| [Bitonic](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/bitonic.md) | `bitonic`, `bitonic_parallel` | no | O(n log² n) | O(n log² n) | O(n log² n) | O(log n) |
| [Odd-Even](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/odd_even.md) | `odd_even`, `odd_even_parallel` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Stooge](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/stooge.md) | `stooge` | no | O(n^2.71) | O(n^2.71) | O(n^2.71) | O(log n) |

¹ About n·log₁.₃ n comparisons in practice; no O(n log n) bound has been proved, and no known input reaches O(n²).

² No exact bound is known; measured, 1.6–2.8·n·log₂ n comparisons up to a million elements. The worst case is O(n^(4/3)) with Sedgewick's gaps (elements larger than 8 bytes); no bound is known for Ciura's (the others).

### Key sorts

These sort by integer or float keys without comparing elements. `k` is the number of possible keys between the smallest and the largest; integer keys are the primitive integers, `char` and `bool`.

| Sort | Functions | Keys | Stable | Time | Extra memory |
| --- | --- | --- | --- | --- | --- |
| [Counting](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/counting.md) | `counting`, `counting_by_key` | integer | `counting_by_key` | O(n + k) | O(k), O(n + k) by key |
| [Radix](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/radix.md) | `radix`, `radix_by_key` | integer | `radix_by_key` | O(n·b/r) for keys that differ in `b` bits, digits of `r` ≈ log₂ n bits | O(n) |
| [Bucket](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/bucket.md) | `bucket`, `bucket_by_key` | integer, `f32`, `f64` | `bucket_by_key` | O(n) for keys spread evenly, O(n·w) for `w`-bit keys at worst | O(n) |
| [Pigeonhole](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/pigeonhole.md) | `pigeonhole`, `pigeonhole_by_key` | integer | no | O(n + k) | O(k), O(n + k) by key |
| [Sleep](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/sleep.md) | `sleep` | integer | — | (max − min) × `unit` of sleeping | a thread per element |

Sleep sort is a joke: a thread per element sleeps in proportion to its value, the result is not always sorted, and `sleep` returns whether it is.

### Against the standard library

Measured on an Apple M4 Max with new input for every sort; each page has the other inputs, sizes and element types:

| Sort | Against | Speed |
| --- | --- | --- |
| [`sort`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/default.md) | `slice::sort` | 1.3–3 times faster on 8–64 random integers, 1.03–1.06 times slower from 100 to 100,000 and as fast beyond; 1.7 times faster on a million random strings |
| [`merge`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/merge.md) | `slice::sort` | 1.6–2.4 times faster on up to 64 random integers, 1.08–1.23 times slower from 1000; 3–6 times slower with few distinct keys |
| [`tim`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/tim.md) | `slice::sort` | 1.2–1.45 times slower on random integers from 1000, 1.5–3.7 times faster on nearly sorted input |
| [`quick`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/quick.md) | `slice::sort_unstable` | 1.1–1.2 times slower on random integers from 100 |
| [`intro`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/intro.md) | `slice::sort_unstable` | 1.45–1.75 times slower on random integers |
| [`radix`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/radix.md) | `slice::sort_unstable` | 1.8–5 times faster on random `u32`s from 1000, up to 2.3 times on random `u64`s |
| [`counting`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/counting.md) | `slice::sort_unstable` | 3.2–10 times faster on random `u32`s and `u64`s with about as many distinct keys as elements, from 100 |
| [`bucket`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/bucket.md) | `slice::sort_unstable` | 1.3–3 times faster on random integers and 1.7–3.7 times on uniform floats, from 100 |
| [`pigeonhole`](https://github.com/nettrash/a/blob/HEAD/lib/docs/sorting/pigeonhole.md) | `slice::sort_unstable` | 1.15–2.2 times faster on random `u32` keys in a range of about `n`, from 1000 |

## Math

### Integers of any size

`a::math::numbers::Number` is an integer of any size, its sign and the bytes of its absolute value in either byte order. Operators take owned or borrowed operands and build the result in an operand's vector when it has room.

| Page | Contents |
| --- | --- |
| [Numbers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/numbers.md) | `Number`: how it is stored, built and read |
| [Traits and conversions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/traits.md) | Comparison and hashing by value; conversions from and to the primitive integers, to `f64` and `f32`, and from strings; `Sum` and `Product`; arithmetic with primitive integers |
| [Addition](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/add.md) | `+` and `+=` |
| [Subtraction](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/sub.md) | `-` and `-=` |
| [Multiplication](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/mul.md) | `*` and `*=`: schoolbook, Karatsuba, Toom-3, Toom-4 and number-theoretic transforms, with assembly kernels on aarch64 and x86_64 |
| [Division](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/div.md) | `/`, `%`, `div_rem`, Euclid's division, a prepared `Divisor` and exact division: Knuth's method, divide and conquer, and Newton's method for the longest |
| [Shifts](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/shift.md) | `<<` and `>>` by any number of bits, rounding as `i32` does |
| [Roots](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/root.md) | `isqrt`, `isqrt_rem` and `nth_root`: Zimmermann's Karatsuba square root and Newton's method |
| [Factorials](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/factorial.md) | `factorial` and `binomial`, from prime factorisations by balanced product trees |
| [Bitwise operations](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/bits.md) | `&`, `\|`, `^` and `!` in two's complement; `bits`, `count_ones`, `trailing_zeros`, `bit` and `set_bit` |
| [Text](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/radix.md) | `Display`, hexadecimal, octal and binary formatting; `to_str_radix` and `from_str_radix` in any radix from 2 to 36 |
| [Greatest common divisor](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/gcd.md) | `gcd`, `lcm`, `gcd_ext` and `mod_inverse`: Lehmer's method and the subquadratic half gcd |
| [Powers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/pow.md) | `pow`, and `pow_mod` by Montgomery's multiplication, with generated assembly kernels on aarch64 |
| [Primes](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/prime.md) | `is_probable_prime` by Baillie–PSW (below 2³² the strong test to base 2 and a table of its pseudoprimes), exact below 2⁶⁴, `next_prime` and `prev_prime` |
| [Prime sieves and counting](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/sieve.md) | `primes(range)`, `prime_pi` and `nth_prime` on `u64`: primesieve's segmented sieve, Deléglise–Rivat with Gourdon's sorting of the special leaves |
| [Factorisation](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/factor.md) | `factor` into primes and exponents, the method chosen by size: trial division, Hart's one line method, Pollard's rho, `p − 1`, the elliptic curve method on Montgomery curves and the quadratic sieve; `trial`, `rho`, `pm1`, `pp1` and `ecm` on their own, their second stages by baby steps and giant steps or by fast polynomial arithmetic (Montgomery and Kruppa's continuation for `p ± 1`, the FFT continuation for the curves); `perfect_power` |
| [Quadratic sieve](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/siqs.md) | `factor::siqs`: a factor by the self-initialising quadratic sieve, with one and two large primes, Gaussian elimination and block Lanczos, for numbers of 24 to 100 digits |
| [Squares modulo a number](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/quadratic.md) | `jacobi` and `kronecker` by the binary method up to six limbs (in assembly on aarch64), Lehmer's steps to 23 limbs and the greatest common divisor's reductions beyond, subquadratic from 800 limbs; `sqrt_mod` and `sqrt_mod_all` modulo any number by Tonelli–Shanks, Atkin's method, Cipolla, Hensel's lifting and the Chinese remainder theorem; Cornacchia's `x² + d·y² = m` |
| [Arithmetic functions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/arith.md) | `divisors`, `divisor_count`, `divisor_sigma`, `totient`, `carmichael`, `mobius`, `is_squarefree` and `radical`, of a number or of its factorisation |
| [Modular arithmetic](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/modular.md) | `crt` for any moduli, `multiplicative_order`, `primitive_root`, and `discrete_log` by Pohlig–Hellman, baby steps and giant steps, Pollard's rho (`dlog`), and the index calculus modulo primes up to 112 bits |
| [Sequences](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/sequences.md) | `fibonacci` and `lucas` by fast doubling, `lucas_sequence` and `lucas_sequence_mod` (Montgomery ladders, any index), `primorial` by product trees, and `partitions` by the Hardy–Ramanujan–Rademacher series |

Measured on an Apple M4 Max in cycles per call, over random operands of the length given, little-endian, against GMP 6.3 and num-bigint 0.4; each page has the conditions, more sizes and the other operations. GMP is faster on some short operands; "–" is not measured:

| Operation | Size | `a` | GMP | num-bigint |
| --- | --- | ---: | ---: | ---: |
| `a + b` | 64 B | 25 | 37 | – |
| | 4 KiB | 469 | 585 | – |
| `a * b` | 64 B | 109 | 107 | 217 |
| | 1 KiB | 9,545 | 11,261 | 29,665 |
| | 64 KiB | 1.12 M | 2.85 M | 17.1 M |
| | 1 MiB | 24.5 M | 86.3 M | 1,058 M |
| `a / b` | 64 B / 32 B | 197 | 174 | 738 |
| | 1 KiB / 512 B | 4,401 | 5,148 | 22,811 |
| | 64 KiB / 32 KiB | 1.30 M | 2.82 M | 15.0 M |
| `isqrt` | 64 B | 426 | 335 | 2,798 |
| | 1 KiB | 4,770 | 4,615 | 101,731 |
| | 64 KiB | 1.46 M | 2.31 M | 136 M |
| `gcd` | 64 B | 5,138 | 6,853 | 26,954 |
| | 1 KiB | 117,118 | 163,209 | 1.15 M |
| | 64 KiB | 55.5 M | 64.4 M | 4,013 M |
| `pow_mod`, the exponent as long as the modulus | 64 B | 71,173 | 137,731 | 248,009 |
| | 256 B | 4.70 M | 5.53 M | 10.0 M |
| | 1 KiB | 231 M | 255 M | 705 M |
| `to_str_radix(10)` | 128 B | 983 | 2,453 | 4,417 |
| | 8 KiB | 504,112 | 607,962 | 2.33 M |
| | 64 KiB | 7.29 M | 12.6 M | 50.3 M |

Number theory, measured on 2026-11-26 on an Apple M4 Max on one thread (the references too), against the fastest of primesieve 12.18, primecount 9.0, PARI 2.19.1, FLINT 3.6.0, GMP 6.3.0, GMP-ECM 7.0.7, msieve 1.53, machine-prime 1.5.7, machine-factor 0.2.5 and malachite 0.15.0 that does the same work; in time per call; each page has the conditions, the other contenders and every case where a reference is faster:

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `primes(a..=b).count()` | [10¹², 10¹² + 10¹⁰] | 1.00 s | primesieve 1.09 s |
| `prime_pi(x)` | x = 10¹⁶ | 1.01 s | primecount 1.08 s |
| `nth_prime(n)` | n = 10¹² | 33.3 ms | primecount 38.3 ms |
| `is_probable_prime` of a prime | 256 bits | 11.4 µs | FLINT 25.8 µs |
| | 2048 bits | 2.94 ms | FLINT 3.50 ms |
| `prev_prime` | 256 bits | 45.3 µs | GMP 133 µs |
| `factor`, two primes of 32 bits | 64 bits | 26.1 µs | PARI 129 µs |
| `ecm`, one curve, stage 1, `B1` = 50,000 | 256 bits | 7.25 ms | GMP-ECM 15.6 ms (`-param 1`) |
| `ecm`, one curve, both stages to `100·B1`, `B1` = 250,000 | 512 bits | 122 ms | GMP-ECM 206 ms (`-param 1`) |
| `pm1`, both stages to `100·B1`, `B1` = 10⁶ | 512 bits | 56.3 ms | GMP-ECM 108 ms |
| `factor`, two primes | 28 digits | 469 µs | PARI 818 µs |
| `factor`, two primes | 40 digits | 5.07 ms | PARI 12.2 ms |
| `factor::siqs`, two primes | 60 digits | 323 ms | PARI 901 ms |
| | 70 digits | 3.83 s | PARI 13.6 s |
| `kronecker` | 1024 bits | 2.56 µs | malachite 3.00 µs |
| `Factorization::sqrt_mod` | prime of 256 bits | 4.21 µs | FLINT 11.6 µs |
| `crt` | 1000 primes of 64 bits | 1.03 ms | PARI 1.18 ms |
| `discrete_log` | safe prime of 64 bits | 1.47 ms | PARI 40.4 ms |
| `fibonacci` | n = 10⁶ | 451 µs | malachite 736 µs |
| `primorial` | n = 10⁶ | 3.10 ms | FLINT 4.54 ms |
| `partitions` | n = 10⁶ | 137 µs | FLINT 200 µs |

The references are faster on the primality test of words (FLINT's `n_is_prime` 1.18 to 1.29 times at 32 bits, machine-prime 1.41 to 1.53 times at 64), the next prime of a word (FLINT's `n_nextprime`, 1.18 and 1.20), primes of 4096 bits (FLINT, 1.15), the powers of words (malachite, 1.76 and 1.82), numbers with many primes of 24 to 28 bits (FLINT, 1.08 and 1.48), all the roots modulo four primes of 16 bits (FLINT, 2.89), and results of a few nanoseconds, where this crate allocates a new number (`fibonacci(100)`, small Lucas sequences, the divisor functions of a factorisation).

### Fixed-width integers

[`a::math::uint`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md) has `Uint<N>` and `Int<N>`, unsigned and signed integers of `N` 64-bit limbs, on the stack and `Copy`, which behave as a `u{64·N}` and an `i{64·N}` would: the operators panic on overflow when the crate is built with debug assertions and wrap without, as the primitive integers do under the default profiles, and the arithmetic has its `checked_*`, `wrapping_*`, `overflowing_*` and `saturating_*` forms. `Montgomery<N>` is the arithmetic modulo an odd number in Montgomery form, and `pow_mod` takes any nonzero modulus. Costs are for `N` limbs.

| Operations | Functions | Method | Cost |
| --- | --- | --- | --- |
| [Arithmetic](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#methods) | `+`, `-`, `*`, `/`, `%`, `&`, `\|`, `^`, `!`, shifts, rotations; `widening_mul`, `carrying_mul`, `div_rem`, `div_euclid` | carries along the limbs; schoolbook products; Knuth's algorithm D | O(N) a sum, O(N²) a product or a quotient |
| Powers, roots, logarithms | `pow`, `isqrt`, `ilog`, `ilog2`, `ilog10` | squaring and multiplying; Newton's iteration; division by the largest power of the base in a limb | O(N²·log e) a power |
| [Text and conversions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#conversions) | `Display`, `{:x}`, `{:o}`, `{:b}` with the formatter's flags, `to_str_radix`, `from_str_radix` (`const`); the primitive integers, `Number`, `resize`, floats, bytes | the bits, or divisions by the largest power of the radix in a limb; one rounding to the nearest float | O(N²) |
| [Montgomery](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#montgomery-multiplication-and-pow_mod) | `Montgomery::new`, `mul`, `square`, `pow`, `inverse`, `to_montgomery`, `from_montgomery` | Montgomery's REDC interleaved with the product (CIOS); sliding windows of 1 to 6 bits; binary inversion | 2N² + N products of limbs a product |
| Modular arithmetic | `pow_mod`, `mul_mod`, `add_mod`, `sub_mod`, `mod_inverse` | an odd modulus in Montgomery form; an even one `2ᵏ·q` modulo `q` and `2ᵏ`, then the Chinese remainder theorem | O(N²·ℓ) for an exponent of ℓ bits |

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md) verified:

| Part | Against | Verified |
| --- | --- | --- |
| [The integers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#the-integers) | `u64`, `i64`, `u128`, `i128`, `Number`, ruint 1.21, bnum 0.12, crypto-bigint 0.6 | about 1.4 million comparisons with the primitives, method by method, in both profiles; every width from 1 to 9 limbs, 16 and 32 against `Number`; 4,190,175 comparisons with the three crates, every difference explained |
| [Montgomery and `pow_mod`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#montgomery-multiplication-and-pow_mod) | `Number`, GMP 6.3, Python's `pow`, ruint 1.21, crypto-bigint 0.6, bnum 0.13 | odd and even moduli of every bit length at 1 to 9 and 16 limbs; 2,312,500 random cases as GMP's; 185,000 as the three crates' |

Measured on an Apple M4 Max on one thread on 2026-11-26, in ns or µs a call, against the fastest of ruint 1.21, crypto-bigint 0.6, bnum 0.13, malachite-nz 0.14, GMP 6.3 and OpenSSL 3.5 that does the same work (and the primitive integers at 64 and 128 bits), `a` and the crates as a user builds them (the default release profile); the page has the conditions, more sizes and every loss. Of the 315 cases `a` is faster in 139 and within 5 % in 71 (median ratio 0.98): decimal text from 256 bits, `ilog10`, the signed quotients, `Montgomery::new`, the conversions and `inverse`, and `pow_mod` at 128 to 384 and at 1,024 bits are faster than every reference. The references are faster for the quotients (1.08 to 2.18), the truncated and checked products, the Montgomery products and powers from 512 bits against OpenSSL's assembly (1.11 to 1.82, but the powers at 1,024 bits) and `mod_inverse` to 512 bits (1.64 to 2.92).

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `wrapping_mul` | 256 bits | 2.18 ns | ruint 1.64 ns |
| `div_rem`, a divisor of half the width | 256 bits | 23.9 ns | ruint 15.9 ns |
| `div_rem_u64` | 1024 bits | 49.5 ns | GMP 32.5 ns |
| `isqrt` | 1024 bits | 149 ns | GMP 169 ns |
| `ilog10` | 1024 bits | 82.2 ns | ruint 733 ns |
| `to_string` | 256 bits | 75.3 ns | ruint 96.5 ns |
| `from_str_radix(10)` | 256 bits | 50.5 ns | bnum 68.3 ns |
| `Int::div_rem` | 256 bits | 20.8 ns | crypto-bigint 78.7 ns |
| `Montgomery::mul` | 256 bits | 11.7 ns | ruint 10.9 ns |
|  | 2048 bits | 545 ns | OpenSSL 518 ns |
| `Uint::pow_mod` | 256 bits | 4.97 µs | OpenSSL 6.64 µs |
|  | 4096 bits | 8.07 ms | OpenSSL 6.63 ms |
| `Uint::mod_inverse` | 256 bits | 859 ns | ruint 515 ns |
|  | 4096 bits | 15.0 µs | malachite-nz 16.9 µs |

### Rational numbers

| Page | Contents |
| --- | --- |
| [Rational numbers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/rational.md) | `a::math::rational::Rational`, a fraction of numbers in lowest terms with a positive denominator, held in two words when both parts fit in 64 bits: `+`, `-`, `*` and `/` reduced by Knuth's two gcds, with numbers and primitive integers on either side; comparisons bounded from the top bits; `floor`, `ceil`, `round`, `pow`; `Sum` and `Product` in a balanced tree; exact conversions from floats and correctly rounded ones to them, text in any radix and decimals with exponents; continued fractions and convergents, the best approximation with a bounded denominator, the simplest rational in an interval and rational reconstruction, on the half gcd; the Bernoulli numbers by the zeta function and Harvey's residues modulo primes, and the harmonic numbers by binary splitting; and the exact determinants, solutions and inverses of [matrices](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md) of rationals, by Bareiss's elimination and Dixon's p-adic lifting |

Measured on an Apple M4 Max on one thread on 2026-11-26, in ns, µs, ms or s a call, over random parts of the limbs (64 bits) given, against the fastest of GMP 6.3 (`mpq_t`, into a variable that exists), FLINT 3.6 (`fmpq`, the same), PARI 2.19, malachite-q 0.14, num-rational 0.4 and rug 1.30 that does the same work, `a` as a user builds it (the default release profile); the page has the conditions, more sizes and every loss. Parts that fit in 64 bits are kept in two words, and their arithmetic, comparisons and conversions run in registers without allocating ([#27](https://github.com/nettrash/a/issues/27)): below 2¹⁶ `+` takes 0.65 of FLINT's time and a comparison 0.37 of the fastest reference's. Of the 584 cases `a` is faster in 418 and within 5 % in 35 (median ratio 0.78). GMP is faster for a fraction of short parts with a long one (1.13 to 1.49 times), GMP and FLINT writing into a variable that exists for the results that are new `Number`s or vectors (`floor`, `trunc`, `clone`, `abs`, `-x` of long parts, up to 7.10 times), malachite-q at hashing, at `==` of long parts and at `Number::try_from` of a small integer; FLINT's continued fractions are faster from 16 to 256 limbs (1.20 to 1.53), its products of matrices of fractions from 32 × 32 (1.78 to 1.92) and its determinants of integer matrices of a few rows (`Matrix<Number>::det`, 1.76 to 5.03 up to 4 × 4). The rows of matrices are the page's own run (`bench/rational`), up to 16 × 16, below the thresholds of Dixon's lifting; the larger exact matrices are the run of the matrices' page (`bench/lattice`), in the table of lattices and exact linear algebra below.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `&a + &b` | parts below 2¹⁶ | 16.0 ns | FLINT 24.5 ns |
|  | 1 limb | 75.4 ns | GMP 80.6 ns |
|  | 16 limbs | 2.78 µs | FLINT 3.51 µs |
|  | 1024 limbs | 671 µs | GMP 925 µs |
| `&a * &b` | parts below 2¹⁶ | 20.2 ns | FLINT 30.7 ns |
|  | 1 limb | 74.9 ns | GMP 121 ns |
|  | 1024 limbs | 1.23 ms | GMP 1.61 ms |
| `a += &b` | parts below 2¹⁶ | 20.1 ns | FLINT 23.6 ns |
|  | 2 limbs | 182 ns | GMP 172 ns |
| `&a + &b`, a part below 2¹⁶ with one of 64 limbs | s + 64 limbs | 230 ns | GMP 159 ns |
| `a.cmp(&b)` | parts below 2¹⁶ | 1.87 ns | num-rational `Ratio<i64>` 5.04 ns |
|  | 1024 limbs | 8.87 ns | malachite-q 11.3 µs |
| `Rational::new` | 1024 limbs | 583 µs | FLINT 742 µs |
| `floor` | parts below 2¹⁶ | 8.84 ns | FLINT 1.95 ns |
| `H_n` by `Iterator::sum` | n = 10⁵, a term | 264 ns | FLINT 332 ns |
| `Rational::try_from(f64)`, parts in words | floats in [2⁻¹¹, 2¹²) | 1.92 ns | malachite-q 7.94 ns |
| `to_f64`, correctly rounded | parts of 64 bits | 4.85 ns | num-rational 35.3 ns |
|  | 4096 bits | 13.6 ns | malachite-q 212 ns |
| `"p/q".parse::<Rational>()` | parts below 2¹⁶ | 19.9 ns | malachite-q 36.0 ns |
|  | 10⁶ digits | 124 ms | malachite-q 198 ms |
| `to_string` | parts below 2¹⁶ | 29.1 ns | FLINT 14.7 ns |
|  | 10⁶ digits | 31.8 ms | FLINT 40.2 ms |
| `continued_fraction` | parts below 2¹⁶ | 122 ns | malachite-q 138 ns |
|  | 64 limbs | 38.6 µs | FLINT 25.2 µs |
| `continued_fraction_into` | 64 limbs | 15.7 µs | FLINT 22.2 µs |
| `limit_denominator`, a bound of half the length | 4096 limbs | 3.71 ms | malachite-q 356 ms |
| `simplest_between` | parts below 2¹⁶ | 12.6 ns | FLINT 43.6 ns |
| `reconstruct` | 100,000 limbs | 260 ms | FLINT 444 ms |
| `Rational::bernoulli` | n = 102 | 26.9 ns | FLINT 1.07 µs |
|  | n = 10⁶ | 7.63 s | FLINT 11.8 s |
| `Rational::bernoulli_numbers` | up to 10⁴ | 382 ms | FLINT 509 ms |
| `Rational::harmonic` | n = 10⁶ | 78.9 ms | malachite-q 258 ms |
| `Matrix<Rational>::det`, integers in −99..=99 | 8 × 8 | 230 ns | FLINT 1.13 µs |
| `Matrix<Rational>::det`, 64-bit parts | 8 × 8 | 75.9 µs | FLINT 69.6 µs |
| `Matrix<Rational>::solve`, integers in −99..=99, one column | 16 × 16 | 12.0 µs | FLINT 30.0 µs |
| `Matrix<Rational>::inverse`, 64-bit parts | 8 × 8 | 1.06 ms | FLINT 1.46 ms |
| `&a * &b`, `Matrix<Rational>`, integers in −99..=99 | 64 × 64 | 73.1 µs | FLINT 129 µs |
| `&a * &b`, `Matrix<Rational>`, fractions of parts ≤ 99 | 32 × 32 | 617 µs | FLINT 320 µs |
| `Matrix<Number>::det`, integers in −99..=99 | 2 × 2 | 29.2 ns | FLINT 5.81 ns |

### Arbitrary-precision floats

| Page | Contents |
| --- | --- |
| [Arbitrary-precision floats](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/float.md) | `a::math::float::BigFloat`, a binary float of any precision from 1 bit to 2⁴⁰, each result correctly rounded in four modes with MPFR's ternary value and the exponent range `±(2⁶² − 1)`: `+`, `-`, `*`, `/`, the fused multiply-add and the square root in registers for one and two limbs and on windows of the limbs the rounding needs beyond, with Karatsuba, Toom-Cook, transforms, Newton's division and Zimmermann's square root at length; results into existing numbers (`set_add`, `set_mul_add`, `set_sqrt`, `set_f64`, `set_floor`, `set_rint`, `add_assign_round`) without allocating; primitive numbers and `Number`s on either side of the operators; exact sums and dot products rounded once; `1/√x`, roots of any degree, integer and rational powers with their exact and halfway cases found first; conversions to and from `f64` and `f32` with the subnormals, to integers (`floor`, `ceil`, `rint_round`, saturating `to_i64_round`) and exact comparisons with integers, floats, numbers and rationals; `Display` as the shortest decimal that reads back, the digits of `mpfr_get_str` in any radix from 2 to 36, and reading any number of digits with `mpfr_strtofr`'s syntax; `exp`, `exp2`, `exp10`, `exp_m1` by Smith's rectangular splitting and the bit burst, `ln`, `log2`, `log10`, `ln_1p` from tables to 512 bits, Newton's iteration on the exponential beyond and the AGM from 3,400 bits, `pow` with C99's special values, and the hyperbolic functions and their inverses; `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `sin_cos`, `atan`, `asin`, `acos` and `atan2`, the argument reduced with π to the bits its exponent needs, a table and Horner's rule in arrays to 512 bits, the series after halvings beyond, and Brent's bit burst from about 770,000 bits for the sine and 205,000 for `atan`; π, `e`, `ln 2`, `ln 10`, Euler's `γ` and Catalan's `G` from tables up to 16,384 bits and beyond by binary splitting of the Chudnovskys', Zuniga's and the Pilehroods' series and the Brent–McMillan algorithm in dual numbers, kept per process; `BigComplex`, as GNU MPC, each part of every result rounded in a mode of its own, with its arithmetic (also into existing numbers, `set_add` and the others), square root, norm and absolute value and `exp`, `ln`, `log10`, the circular and hyperbolic functions and their inverses, `pow`, `arg` and `proj` with MPC's branch cuts and special values; and `Matrix<BigFloat>` and `Matrix<BigComplex>`, products whose every element is the exact dot product rounded once and LU solutions of exact dot products |

Measured on an Apple M4 Max on one thread on 2026-11-26, in ns, µs, ms or s a call at the precision given in bits (both parts' for complex numbers), against the fastest of MPFR 4.2.2 and GNU MPC 1.4.1 (through FFI, into a variable that exists), rug 1.30, malachite-float 0.15, astro-float 0.9.6 and dashu-float 0.6.2 that does the same work, `a` as a user builds it (the default release profile); Arb in FLINT 3.6, whose balls are not correctly rounded, does less and is not counted. The page has the conditions, more sizes and every loss. Of the 1,305 cases with such a reference `a` is faster in 1,020 and within 5 % in 105 (median ratio 0.64): the elementary functions take 0.15 to 1.10 of the fastest reference's time, the functions of complex numbers 0.0085 to 1.03 of MPC's, the text 0.073 to 1.00 and the constants' first call 0.00060 to 0.56. The references are faster in 180 cases: sums of four and sixteen limbs and the assigning forms of one and two limbs (up to 1.44), cached constants against MPFR's copy into a variable (1.10 to 2.62), comparisons and conversions through `Number` and `Rational` (up to 3.41), `BigComplex::proj` (1.11 to 4.52) and `from_f64_round` from 10⁴ bits against dashu-float, which keeps the 53 bits of the float (1.83 to 50.3).

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `set_add` | 53 bits | 3.44 ns | MPFR 4.57 ns |
|  | 1000 bits | 18.6 ns | rug 13.9 ns |
|  | 10⁵ bits | 540 ns | rug 1.19 µs |
| `set_add`, operands of 10⁵ bits | 53 bits | 16.4 ns | rug 214 ns |
| `set_mul` | 53 bits | 2.44 ns | MPFR 5.24 ns |
|  | 10⁶ bits | 567 µs | MPFR 1.41 ms |
| `set_div` | 10⁶ bits | 1.34 ms | MPFR 3.99 ms |
| `set_sqrt` | 256 bits | 118 ns | MPFR 106 ns |
|  | 10⁶ bits | 1.45 ms | MPFR 3.29 ms |
| `add_round`, a new number | 53 bits | 5.54 ns | malachite-float 8.54 ns |
| `mul_add_round` | 113 bits | 15.8 ns | rug 36.0 ns |
|  | 10⁶ bits | 571 µs | malachite-float 968 µs |
| `add_assign_round` | 256 bits | 14.1 ns | MPFR 9.86 ns |
| `sum_round`, 4 terms | 1000 bits | 82.5 ns | MPFR 56.1 ns |
| `dot_round`, 100 terms | 53 bits | 1.36 µs | MPFR 2.82 µs |
| `rec_sqrt_round` | 10⁴ bits | 7.73 µs | MPFR 9.18 µs |
| `nth_root_round(100)` | 1000 bits | 6.38 µs | MPFR 43.9 µs |
| `powi_round(1_000_000_007)` | 53 bits | 120 ns | MPFR 429 ns |
| `to_f64_round` | 53 bits | 3.20 ns | MPFR 5.36 ns |
| `from_f64_round` | 53 bits | 4.48 ns | malachite-float 14.1 ns |
|  | 10⁶ bits | 734 ns | dashu-float 14.6 ns |
| `a < b` | 53 bits | 2.42 ns | MPFR 2.06 ns |
| `to_number_round` | 53 bits | 17.7 ns | malachite-float 5.20 ns |
| `to_digits(10, None)` | 53 bits | 47.0 ns | MPFR 207 ns |
|  | 10⁶ bits | 4.76 ms | MPFR 9.86 ms |
| `from_str_radix`, 53 bits | 10⁶ digits | 36.9 µs | MPFR 504 µs |
| `exp` | 53 bits | 352 ns | MPFR 550 ns |
|  | 10⁶ bits | 95.9 ms | malachite-float 167 ms |
| `ln` | 53 bits | 207 ns | MPFR 390 ns |
|  | 10⁴ bits | 160 µs | MPFR 168 µs |
|  | 10⁶ bits | 72.5 ms | MPFR 170 ms |
| `pow` | 113 bits | 948 ns | MPFR 2.03 µs |
| `sinh` | 10⁵ bits | 5.71 ms | MPFR 9.00 ms |
| `sin` | 53 bits | 213 ns | MPFR 571 ns |
|  | 10⁶ bits | 243 ms | malachite-float 248 ms |
| `cos`, an argument `m·2^(10⁶)` | 53 bits | 1.37 ms | MPFR 3.91 ms |
| `atan` | 10⁶ bits | 244 ms | MPFR 346 ms |
| `BigFloat::pi`, first call | 10⁶ bits | 19.3 ms | MPFR 84.9 ms |
| `BigFloat::euler_gamma`, first call | 10⁶ bits | 507 ms | malachite-float 1.97 s |
| `BigFloat::pi`, cached | 1000 bits | 17.8 ns | MPFR 7.56 ns |
| `BigComplex::set_mul` | 53 bits | 21.6 ns | MPC 48.9 ns |
| `BigComplex::mul_round` | 10⁵ bits | 161 µs | MPC 316 µs |
| `BigComplex::sqrt_round` | 53 bits | 82.2 ns | MPC 182 ns |
| `BigComplex::exp_round` | 1000 bits | 8.51 µs | MPC 18.4 µs |
| `BigComplex::asin_round` | 1000 bits | 19.8 µs | MPC 141 µs |
| `BigComplex::pow_round` | 10⁵ bits | 29.6 ms | MPC 130 ms |
| `BigComplex::proj` | 1000 bits | 29.8 ns | MPC 6.59 ns |
| `&a * &b`, `Matrix<BigFloat>`, 8 × 8 | 256 bits | 15.0 µs | MPFR 23.3 µs |

### Polynomials

| Page | Contents |
| --- | --- |
| [Polynomials](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/poly.md) | `a::math::poly::Polynomial<T>`, dense, with coefficients that are `Number`s, `Rational`s, `f64`s or `Complex<f64>`s, and `ModPolynomial`, with residues modulo any `n ≥ 2` that it holds once, in one word below 2⁶⁴: `+`, `-`, `*`, `sqr` and `pow` by the schoolbook method with each sum of products reduced once (NEON's multiply-adds for small moduli, machine integers for small integer coefficients), the number-theoretic transforms on the coefficients or on their pieces, Kronecker substitution, and the FFT for long float products; power series inverses, division by Newton's method, pseudo-division and exact division on limbs and by Kronecker substitution, `mul_mod`, `pow_mod` and Brent and Kung's `compose_mod` with the inverse of the modulus kept, and `compose` by divide and conquer; greatest common divisors by Euclid's algorithm in Montgomery's form and the half gcd, over the integers the heuristic gcd and the multimodular algorithm, `xgcd` (multimodular over the rationals), `lcm`, resultants and discriminants (multimodular over the integers) and the subresultants by Lazard's and Ducos's sequence; evaluation at many points by Horner's rule at many points at once and the scaled remainder tree, products of linear factors, and interpolation on the subproduct tree or by Lagrange's and Newton's forms; complex roots in double precision by the Aberth–Ehrlich iteration in exact conjugate pairs, certified complex roots of integer and rational polynomials to any precision, real roots isolated by Descartes's rule and continued fractions, counted, and correctly rounded at any precision; and square-free decompositions, factorisations and tests of irreducibility modulo any prime (Kaltofen and Shoup's baby steps and giant steps, Cantor and Zassenhaus's splitting, Berlekamp's method, Ben-Or's test, the roots) and over the integers and the rationals (Zassenhaus's method with Hensel's lifting) |

Measured on an Apple M4 Max on one thread, in ns, µs, ms or s a call, against the fastest of FLINT 3.6, NTL 11.6, PARI/GP 2.19 and, for the roots, Arb in FLINT that does the same work (FLINT's preinverse and NTL's `Modulus` built before the clock, NTL's equal- and distinct-degree steps given `x^p mod f`, PARI's distinct roots and the other references that do less are not counted; the page lists them); the page has the conditions, every size and all the losses. Of 1,806 cases, 573 are losses and 1,113 below 0.95 of the fastest reference's time, the median about 0.78. Modulo a word the products take 0.532 to 0.857 of the fastest reference's time from 128 coefficients for 50- to 64-bit moduli, and `x.pow_mod(&e, &f)` 0.084 to 0.556 of PARI's from 256; evaluation at many points modulo a word 0.304 to 0.485 of FLINT's from 4,096 points; the factorisations modulo 50- to 64-bit primes 0.337 to 0.862 from degree 30, and over the integers 0.305 to 1.18; the certified complex roots 0.150 to 0.834 of Arb's time, and the isolation of real roots 0.025 to 0.754 of FLINT's from degree 50. FLINT is faster at `Polynomial<Rational>` (products 1.16 to 9.09 times: every coefficient in lowest terms, where `fmpq_poly` keeps one denominator), over the integers at coefficients of 32 bits (1.34 to 6.15) and of 64 bits from 128 coefficients (1.22 to 1.96: each `Number` coefficient is a vector of its own), modulo 65,521 from 256 coefficients (1.09 to 1.46: two 29-bit transform primes against one 50-bit prime and truncated transforms), and at a few coefficients (a new polynomial each call); NTL's `ZZ_pX` modulo primes of 4 to 16 limbs from 128 coefficients (1.06 to 1.75) and its `GF2X` modulo 2 (products 1.89 to 10.9, squares 4.34 to 11.4).

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `ModPolynomial` `*`, n = 2⁶³ − 25 | 10⁵ by 10⁵ | 5.71 ms | FLINT 7.10 ms |
| | 10⁶ by 10⁶ | 55.4 ms | FLINT 76.8 ms |
| `ModPolynomial` `*`, n = 65,521 | 10⁴ by 10⁴ | 165 µs | FLINT 150 µs |
| `ModPolynomial` `*`, n = 2 | 10⁵ by 10⁵ | 302 µs | NTL `GF2X` 143 µs |
| `div_rem`, n = 2⁶³ − 25 | 2·10⁴ by 10⁴ | 1.47 ms | FLINT 1.94 ms |
| `inv_series`, n = 2⁶³ − 25 | 10⁴ | 768 µs | FLINT 1.13 ms |
| `x.pow_mod(&e, &f)`, n = e = 2⁶³ − 25 | deg f = 1,000 | 3.57 ms | PARI 26.6 ms |
| `compose_mod`, n = 2⁶³ − 25 | 1,000 | 4.98 ms | PARI 32.7 ms |
| `ModPolynomial` `*`, n = 2¹²⁷ − 1 | 1,000 by 1,000 | 128 µs | NTL 158 µs |
| `Polynomial<Number>` `*`, 64-bit coefficients | 10⁴ by 10⁴ | 1.19 ms | FLINT 855 µs |
| `Polynomial<Number>` `*`, 1,000-bit coefficients | 10⁴ by 10⁴ | 17.3 ms | NTL 24.3 ms |
| `pseudo_div_rem`, 64-bit coefficients | 256 by 128 | 1.23 ms | PARI 1.30 ms |
| `div_exact`, 64-bit coefficients | 3,999 by 2,000 | 701 µs | NTL 806 µs |
| `gcd`, n = 2⁵⁰ − 27, a common factor | degree 10⁴ | 10.1 ms | NTL 13.4 ms |
| `gcd` over the integers, 64 bits, a common factor | degree 1,000 | 936 µs | NTL 1.02 ms |
| `resultant` over the integers, 64 bits | degree 100 | 1.61 ms | FLINT 4.53 ms |
| `eval_many_u64`, n = 2⁵⁰ − 27 | 4,096 at 4,096 points | 1.33 ms | FLINT 3.14 ms |
| `interpolate_u64`, n = 2⁵⁰ − 27 | 4,096 points | 2.25 ms | FLINT 4.89 ms |
| `Polynomial<f64>::roots` | degree 100 | 343 µs | Arb `acb_poly_find_roots` 2.40 ms |
| `complex_roots(53)`, 16-bit coefficients | degree 100 | 4.12 ms | Arb 15.5 ms |
| `real_root_intervals`, 16-bit coefficients | degree 1,000 | 32.2 ms | FLINT 65.5 ms |
| `factor`, n = 2⁶⁴ − 59 | degree 1,000 | 74.9 ms | FLINT 169 ms |
| `factor` over the integers, 64-bit coefficients | degree 100 | 535 µs | FLINT 1.28 ms |
| `factor`, Swinnerton-Dyer polynomial | degree 32 | 576 µs | FLINT 877 µs |

### Fourier transforms

| Page | Contents |
| --- | --- |
| [Fourier transforms](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/fft.md) | `a::math::fft`: `Fft<T>`, the complex transform of `f32` and `f64` values of any length, planned once, by Stockham passes of the radices 2, 4, 8, 3, 5, 7 and the primes to 19 (NEON, SSE2 or AVX2 with FMA chosen at run time, portable elsewhere), a larger prime factor by its own plan, and prime lengths by Rader's or Bluestein's algorithm with a smooth inner length; `RealFft<T>`, the transform of reals into the `n/2 + 1` values of the spectrum, by the half-length complex transform for even lengths and a split by the smallest factor for odd ones; `Dct<T>` and `Dst<T>`, the eight cosine and sine transforms of FFTW's `REDFT` and `RODFT` kinds by FFTW's accurate algorithms; `Fft2<T>` and `Fft3<T>` by tiles of columns; every inverse unscaled, every method on batches, and `_with_scratch` forms that never allocate; and `convolve`, `correlate`, `circular_convolve` and `circular_correlate` of real or complex sequences, by SIMD direct sums, padded transforms or overlap-add, chosen by a measured model of their costs, with the plans kept in a bounded cache shared by the threads |

Measured on an Apple M4 Max on one thread, in ns, µs or ms a call, against the fastest of FFTW 3.3.11 (Homebrew's build, which on arm64 has only its scalar codelets), rustfft 6.4.1, realfft 3.5.1, rustdct 0.7.1 and Apple's Accelerate vDSP that has the transform; the page has the conditions, every size and all the losses. Of 1,602 cases, 300 are losses and 1,157 below 0.95 of the fastest reference's time, the median 0.762. The complex transform of `f64` takes 0.649 to 1.02 of the fastest reference's time for the powers of two from 64 (0.494 of FFTW's at 2²⁰), smooth lengths 0.422 to 1.32, primes and twice primes 0.313 to 0.838; `Fft3` 0.336 to 0.767; the convolutions against an operand of 10⁴ values or more 0.183 to 1.32. vDSP is faster in `f32`, 1.06 to 3.58 times, at the complex transform from 64 values, the real one from 64, DCT-II to IV from 16 and `Fft2` (Apple's `f32` kernels run at about five times its own `f64` speed; `a` is as fast as vDSP or faster in `f64`); FFTW and rustfft at single complex transforms of 2 to 36 values and batches of short ones (1.06 to 1.88: a call's dispatch), FFTW at real transforms of 8 to 80 values and odd lengths 15 to 243 (1.06 to 1.78: straight-line real codelets), rustdct at the in-place cosine and sine transforms of 16 (1.15 to 2.12), vDSP's direct sums at short convolutions (1.07 to 3.67), and FFTW's `ESTIMATE` planning at 2²⁰ (3.36 and 4.50: correctly rounded twiddles).

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `Fft::<f64>` forward | 1,024 | 1.26 µs | vDSP 1.37 µs |
| | 2²⁰ | 4.02 ms | rustfft 5.69 ms |
| | 10⁶ | 3.30 ms | FFTW 7.69 ms |
| `Fft::<f64>` forward, prime | 100,003 | 1.58 ms | FFTW 2.19 ms |
| `Fft::<f64>` forward | 16 | 10.4 ns | FFTW 11.6 ns |
| `Fft::<f32>` forward | 1,024 | 617 ns | vDSP 278 ns |
| `RealFft::<f64>` forward | 1,024 | 700 ns | vDSP 855 ns |
| | 10⁶ | 1.76 ms | FFTW 2.69 ms |
| | 2,187 | 3.17 µs | FFTW 3.63 µs |
| `RealFft::<f32>` forward | 65,536 | 45.5 µs | vDSP 22.3 µs |
| `Dct::<f64>`, DCT-II | 1,024 | 1.20 µs | FFTW 1.48 µs |
| `Dct::<f32>`, DCT-II | 1,024 | 714 ns | vDSP 411 ns |
| `Fft2::<f64>` forward | 1000 × 1000 | 3.75 ms | FFTW 6.01 ms |
| `Fft3::<f64>` forward | 128 × 128 × 128 | 9.30 ms | FFTW 13.3 ms |
| `convolve_into`, `f64` | 10⁵ by 3 | 11.3 µs | vDSP 27.8 µs |
| | 10⁵ by 10⁵ | 1.19 ms | FFTW 1.46 ms |
| `convolve_into`, `Complex<f32>` | 10⁴ by 16 | 12.0 µs | rustfft 24.2 µs |

### Special functions

| Page | Contents |
| --- | --- |
| [Special functions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/special.md) | `a::math::special`, of `f32`, `f64` and complex arguments, every function with a stated ulp bound tested against mpmath at 256 bits: `gamma`, `lgamma` (with `lgamma_sign`), `digamma`, `polygamma`, `beta`, `lbeta`, `beta_inc` with its complement and their inverses, `gamma_p`, `gamma_q` and their inverses, `factorial`, `binomial`, `pochhammer`; `erf`, `erfc`, `erfcx`, `erf_inv`, `erfc_inv`, `faddeeva`, `fresnel`, `dawson`; `bessel_j`, `bessel_y`, `bessel_i`, `bessel_k` of integer and real order (with `bessel_i_scaled` and `bessel_k_scaled`) and the spherical `sph_bessel_j`, `sph_bessel_y`, `sph_bessel_i`, `sph_bessel_k`, and `airy_ai`, `airy_bi` with their derivatives; `zeta`, `hurwitz_zeta`, `dilog`, `polylog`, `lambert_w0`, `lambert_wm1`, `expint_e1`, `expint_ei`, `expint_en`; `ellip_k`, `ellip_e`, `ellip_pi`, `ellip_f`, `ellip_e_inc`, `ellip_pi_inc`, Carlson's `carlson_rc`, `carlson_rf`, `carlson_rd`, `carlson_rj`, `jacobi_elliptic` (the `sn`, `cn` and `dn` at once), and the orthogonal polynomials `legendre_p`, `assoc_legendre_p`, `chebyshev_t`, `chebyshev_u`, `gegenbauer_c`, `jacobi_p`, `laguerre_l`, `assoc_laguerre_l`, `hermite_h`, `hermite_he` |

Measured on an Apple M4 Max on one thread, in ns a call over 16,384 random arguments of each region, against the system's libm (Apple's), Boost.Math 1.92 (without promotion to `long double`), GSL 2.8, SciPy 1.18 and the crates libm 0.2, statrs 0.18, special 0.11, puruspe 0.4 and lambert_w 1.2; a reference counts only where it stays within twice `a`'s error bound of `a`'s values on those arguments (the page gives the faster ones beyond the bound beside them, with their largest difference). Of 318 cases, 126 have such a reference: 59 losses and 61 cases below 0.95 of its time. `gamma` takes 0.657 to 0.723 of Apple's libm's time from 0.5 to 170, `lgamma` 0.424 of libm's on [3, 8], `erf` and `erfc` 0.334 to 0.784 of the fastest's, `dawson` 0.0624 to 0.0722 of GSL's, `zeta` 0.330 to 0.711, `lambert_w0` 0.0856 of GSL's below 3, `ellip_k` and `ellip_e` 0.345 to 0.515 of SciPy's below `m` = 0.8, and the `f32` Bessel functions of orders 0 and 1 0.274 to 0.414 of Boost's. The references within the bound are faster at the integer orders of `K`, `Y` and `J` (1.86 to 3.59), the real orders at small and moderate `x` (1.60 to 2.64), `K₀` and `Y₀`, `Y₁` beyond 25 (1.74 to 2.05), `ψ′` (1.27, 2.33), the inverse of `Q` at tiny `q` (1.98, 3.10), the complex `ln Γ` at large `|z|` (3.98), `E₁` and `Ei` (1.55 to 2.03), the polylogarithm past −1 (1.25 to 1.73), `jacobi_elliptic` (2.33, and 4.24 near `m` = 1) and `ellip_k`, `ellip_e` from `m` = 0.8 (1.79, 1.89): the double-double steps that hold `a`'s bounds. No reference holds the bounds of the orthogonal polynomials or of the `f64` Airy functions on these regions.

| Function | Domain | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `gamma` | [0.5, 1.5] | 4.16 ns | Apple libm 6.33 ns |
| `lgamma` | [3, 8] | 3.64 ns | libm 8.60 ns |
| `erf` | [0.5, 3] | 4.59 ns | SciPy 6.52 ns |
| `erfc` | [3, 8] | 4.31 ns | libm 10.2 ns |
| `dawson` | [8, 10⁴] | 1.89 ns | GSL 30.3 ns |
| `zeta` | [−0.5, 0.9] | 4.11 ns | Boost 12.5 ns |
| `zeta`, complex | `σ` in [1.1, 10], `t` in [−5, 5] | 477 ns | SciPy 1.54 µs |
| `lambert_w0` | [−0.36, 3] | 7.13 ns | GSL 83.2 ns |
| `expint_ei` | [6, 50] | 7.85 ns | GSL 34.7 ns |
| `ellip_e` | `m` in [0, 0.8) | 2.21 ns | SciPy 6.42 ns |
| `bessel_j` | ν in [0, 3], x in [30, 10³] | 69.1 ns | SciPy 172 ns |
| `bessel_k`, integer `n` | `n` in [2, 20], x in [0.1, 50] | 59.2 ns | puruspe 16.5 ns |
| `gamma_q_inv` | `a` in [0.01, 1], tiny `q` | 346 ns | Boost 112 ns |
| `jacobi_elliptic` | `u` in [−4, 4], `m` in [0.99, 1) | 140 ns | GSL 33.1 ns |

### Finite fields

[`a::math::field`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/field.md) holds finite fields and their elements. A field is built once, checks what it is given and chooses the form of its arithmetic; its elements borrow it and combine with `+`, `-`, `*` and `/`. Elements of different fields do not mix: an operation on two of them panics.

| Operations | Functions | Method | Cost |
| --- | --- | --- | --- |
| [Prime fields](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/field.md#prime-fields) | `PrimeField::new`, `Fp`: `+`, `-`, `*`, `/`, `pow`, `inverse`, `batch_invert`, bytes; the fields of BN254, BLS12-381 and the named curves | a form chosen from `p`: `GF(2)`, P-256's, P-384's, P-521's, `2^k − c`, Montgomery's up to nine limbs, `Number` beyond; inverses by Lehmer's extended Euclid; batches by Montgomery's trick | O(n²) products of words a product |
| Square roots | `sqrt`, `legendre`, `is_square`, `sgn0` | by `p`: a power for 3 mod 4, Atkin's for 5 mod 8, Tonelli and Shanks's or Cipolla and Lehmer's for 1 mod 8 | a power, and up to `s(s + 1)/2 − 1` squarings for Tonelli and Shanks's |
| [Extension fields](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/field.md#extension-fields) | `ExtensionField::new`, `with_degree`, `first_irreducible`, `Fpk` | `GF(p)[x]/g`, `g` checked by Ben-Or's test, or the smallest irreducible `g`; schoolbook products, `ModPolynomial`'s from a degree set by `p`; inverses by the extended Euclidean algorithm | `k² + (k − 1)·w` products of `GF(p)` a product, `w` the terms of `g` |
| Frobenius, norm, trace | `frobenius`, `norm`, `trace`, `sqrt`, `multiplicative_order`, `primitive_element` | the matrix of `a ↦ aᵖ`; the resultant; Newton's identities; Tonelli and Shanks's method in `GF(pᵏ)`; the factorisation of `pᵏ − 1` | `O(i·k²)`, `O(k²)`, `k` products |
| [Towers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/field.md#towers) | `Tower`, `Fp2`, `Fp6`, `Fp12`, BN254's and BLS12-381's | `F_p²[v]/(v³ − ξ)`, `F_p⁶[w]/(w² − v)`: Karatsuba's products, Chung and Hasan's squares, sparse products by lines, Frobenius maps by constants, Granger and Scott's cyclotomic squares, the complex method for roots in `F_p²` | 54 products of `F_p` a product of `F_p¹²` |
| [Binary fields](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/field.md#binary-fields) | `BinaryField::new`, `F2m`: `+`, `*`, `/`, `sqrt`, `trace`, `half_trace`, `quadratic_root`, `is_irreducible` | `PMULL`, `PCLMULQDQ` or a portable window method, Karatsuba's across words; a sparse or Barrett's reduction; Euclid's inverse; Rabin's test | O(n^1.58) products of words a product |
| Normal bases | `NormalBasis`, `NormalElement`, `BasisConversion` | Gaussian normal bases as FIPS 186-4 has them: Massey and Omura's products, Itoh and Tsujii's inverses, conversions by Berlekamp's trace algorithm | a square is a rotation |
| AES and GCM | `Gf2_8` with `aes_sbox`, `Gf2_128` with `ghash` and `gcm_ghash` | tables of logarithms; a carry-less product of 128 bits and its reduction | |

Measured on an Apple M4 Max on one thread, in the default release profile, in ns or µs an operation, the median of 21 samples alternated with the references in one process, against the fastest of arkworks 0.5, blst 0.3.17, the `bls12_381` crate 0.8, mcl 4.44.0, FLINT 3.6, NTL 11.6, GMP 6.3, OpenSSL 4.1 and RustCrypto's `ghash` 0.6 that does the same work; the page has the conditions, every field and all the losses with their causes. Of 737 cases, 526 take at most 1.05 times the fastest reference's time (the median 0.817): inversion beats every reference at every size but one, `TypedFp`'s products take 0.501 to 0.940 of the fastest's time on every named prime, BN254's tower 0.467 to 0.664 of arkworks's for products and squares, and GHASH 0.458 to 0.800 (1.04 at 64 bytes). The references are faster at: the run-time `Fp`'s sums, differences, negations and doublings (1.08 to 4.59 times the fastest, mostly arkworks's: the price of choosing the form at run time; `TypedFp` ties or wins all but its sums, 1.28 to 1.58, and two negations, 1.26 and 1.30), products and squares from eight limbs (1.07 to 1.63 times OpenSSL's hand-written assembly, arkworks's and FLINT's), small extension fields (`GF(3⁵)` 1.42 to 6.71 times FLINT's `fq_zech`, word primes 1.11 to 3.83 times FLINT's `fq_nmod` and NTL), BLS12-381's `F_p¹²` (1.20 to 1.23 times mcl's assembly), and squares and powers in `GF(2ᵐ)` (1.09 to 1.87 times NTL's).

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `Fp` product | BN254's `p` | 13.5 ns | arkworks 12.7 ns |
| `TypedFp` product | BN254's `p` | 11.5 ns | arkworks 12.7 ns |
| `Fp` product | `2²⁵⁵ − 19` | 8.91 ns | OpenSSL 13.7 ns |
| `Fp` product | a prime of 512 bits | 50.6 ns | OpenSSL 38.8 ns |
| `Fp` sum | BN254's `p` | 3.48 ns | arkworks 1.12 ns |
| `Fp::inverse` | BLS12-381's `p` | 980 ns | GMP 1.08 µs |
| `Fp::sqrt` | P-384's `p` | 6.83 µs | arkworks 14.9 µs |
| `Fp12` product | BN254 | 670 ns | arkworks 1.38 µs |
| `Fp12` product | BLS12-381 | 1.16 µs | mcl 938 ns |
| `Fpk` product | BN254's `p`, `k = 12` | 1.62 µs | FLINT 2.95 µs |
| `Fpk` product | `p = 10⁹ + 7`, `k = 8` | 82.8 ns | FLINT 74.5 ns |
| `F2m` product | `GF(2⁵⁷¹)` | 84.2 ns | NTL 109 ns |
| `F2m::inverse` | `GF(2²³³)` | 479 ns | NTL 1.76 µs |
| `Gf2_128::ghash` | 1 KiB | 47.3 ns | RustCrypto `ghash` 80.5 ns |

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/field.md) verified:

| Part | Against | Verified |
| --- | --- | --- |
| Prime fields | `Number`, PARI/GP 2.19 | 79 primes in all 17 forms and 5 methods of square roots, on 2,188 pairs of inputs; 896 Legendre symbols and 489 square roots as PARI's |
| Extension fields | PARI/GP 2.19, FLINT 3.6 | 148 elements of 34 fields: products, inverses, powers, Frobenius maps, norms, traces, roots and orders; the smallest irreducible polynomial for 140 pairs `(p, k)` |
| Towers | arkworks 0.5, PARI/GP 2.19 | 28 operations on 16 elements of each curve's tower as arkworks'; 84 towers of other non-residues against schoolbook products |
| Binary fields | NTL 11.6, OpenSSL 4.1, PARI/GP 2.19, FIPS 186-4 | 16,432 outputs in 83 fields as NTL's, 12,677 of them also OpenSSL's; normal bases with FIPS 186-4's values and 121 of PARI's minimal polynomials |
| AES and GCM | FIPS 197, the GCM specification, OpenSSL 4.1 | both S-boxes at all 256 bytes; GHASH of all 18 test cases, and 64 random AES-GCM tags |

### Elliptic curves

| Page | Contents |
| --- | --- |
| [Elliptic curves](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/ec.md) | `a::math::ec`: short Weierstrass, Montgomery and twisted Edwards curves over prime fields of any length; `+`, `-`, `*`, compression, `multi_mul`, prepared points and maps between the forms; secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519 ready made |
| [Curves over binary fields](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/ec.md#curves-over-binary-fields) | `a::math::ec::binary`: `y² + x·y = x³ + a·x² + b` over `GF(2ᵐ)`, in López–Dahab coordinates; the τ-adic non-adjacent form on Koblitz curves, the ladder on `x` alone, SEC 1's three encodings; the 18 curves of SEC 2 ready made, NIST's K- and B- curves among them |
| [Hashing to curves](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/ec.md#hashing-to-curves) | `a::math::ec::hash`: RFC 9380's twelve suites for P-256, P-384, P-521, secp256k1, curve25519 and edwards25519; `hash_to_field`, and the Shallue–van de Woestijne, simplified SWU and Elligator 2 maps and isogenies for other curves |

The curves over prime fields, measured on an Apple M4 Max in cycles, against OpenSSL 3.6 and curve25519-dalek 4.1; `k·P` multiplies a point, `k·G` the generator, by a table where the library has one. OpenSSL's and dalek's products of a point are constant-time, which costs them time this crate doesn't spend:

| Curve | Against | `k·P` | theirs | `k·G` | theirs |
| --- | --- | ---: | ---: | ---: | ---: |
| secp256k1 | OpenSSL | 40,100 | 555,000 | 7,250 | 559,000 |
| P-256 | OpenSSL | 82,600 | 90,900 | 10,600 | 14,300 |
| P-384 | OpenSSL | 311,000 | 548,000 | 40,200 | 267,000 |
| P-521 | OpenSSL | 358,000 | 496,000 | 52,500 | 269,000 |
| Ed25519 | dalek | 45,300 | 72,000 | 5,350 | 22,100 |
| Curve25519, X25519 | dalek | 55,400 | 78,500 | | |

The curves over binary fields and the hashing to curves, measured on an Apple M4 Max on one thread, in the default release profile, the median of 21 samples alternated in one process, against the faster of OpenSSL 3.5.9 and 4.1.0, RustCrypto's `p256`, `p384`, `p521` and `k256` 0.13 and curve25519-dalek 5.0; the page has the conditions, every curve and all the losses with their causes. On the binary curves, products take 0.178 to 0.428 of OpenSSL's time, `multi_mul` 0.115 to 0.233, and the compressed encoding and its decoding 0.138 to 0.727; `to_bytes` up to 409 bits (1.07 to 1.89 times, its allocations) and the uncompressed decoding at 113 bits (1.41, 1.44) are slower. The RFC 9380 suites of the NIST curves and secp256k1 take 0.204 to 0.329 of RustCrypto's time; edwards25519's 1.66 and 1.75 times dalek's.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `*` | sect233r1 | 49.2 µs | OpenSSL 190 µs |
| `multi_mul`, 64 terms | sect409r1 | 5.10 ms | OpenSSL 43.9 ms |
| `to_bytes` | sect163r2 | 135 ns | OpenSSL 78.9 ns |
| `hash_to_curve` | P-256 | 7.86 µs | RustCrypto `p256` 27.3 µs |
| `hash_to_curve` | edwards25519 | 9.04 µs | curve25519-dalek 5.16 µs |

What the page verified them against:

| Part | Against | Verified |
| --- | --- | --- |
| [Curves over binary fields](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/ec.md#curves-over-binary-fields) | SEC 2, OpenSSL 4.1, PARI/GP 2.19, an exhaustive search | the 126 parameters of the 18 curves; on every curve, 108 products, 54 sums, 162 encodings and 432 verdicts on encodings as OpenSSL's; the 4,116 points of 55 curves over `GF(2)` to `GF(2⁸)` against the textbook's group law |
| [Hashing to curves](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/ec.md#hashing-to-curves) | RFC 9380, RustCrypto's `p256` and `p384` | the 60 vectors of appendix J for the twelve suites; 20,000 points of 5,000 random messages |

### Pairings

[`a::math::pairing`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/pairing.md) computes the optimal ate pairing `e: G1 × G2 → GT` on BN254 and BLS12-381, with their groups and their encodings, and hashes to `G1` and `G2`. Its values are arkworks' on both curves, and those of the `bls12_381` crate, blst and RELIC on BLS12-381. There is no protocol layer: no BLS signatures, keys or random numbers.

| Operations | Functions | Method | Cost |
| --- | --- | --- | --- |
| [The pairing](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/pairing.md#the-pairing) | `pairing`, `multi_pairing`, `miller_loop`, `final_exponentiation` | Vercauteren's optimal ate pairing: Miller's loop on the twist with the lines of Costello, Lange and Naehrig, then the hard part of the exponentiation by Hayashida, Hayasaka and Teruya (BLS12-381) and by Fuentes-Castañeda, Knapp and Rodríguez-Henríquez (BN254) | 63 doublings and 5 additions (BLS12-381), 65 and 21 (BN254); five or three powers by `x` |
| [The groups](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/pairing.md#the-groups) | `G1`, `G2`, `Gt`: `+`, `-`, `*`, `is_in_subgroup`, `clear_cofactor`, `psi` | Jacobian coordinates and the width-4 non-adjacent form; membership by endomorphisms (Scott; El Housni, Guillevic and Piellard); RFC 9380's and Budroni and Pintore's clearings on BLS12-381 | at most two products by 64 or 128 bits a membership test |
| Encodings | `to_zcash_compressed`, `from_zcash`, `to_ethereum`, `from_ethereum`, `ethereum::{ec_add, ec_mul, ec_pairing}` | ZCash's for BLS12-381, EIP-196 and EIP-197 for BN254, every decoding checked | |
| [Hashing to G1 and G2](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/pairing.md#hashing-to-g1-and-g2) | `hash::bls12_381_g1()`, `hash::bls12_381_g2()`, `hash::bn254_g1()` | RFC 9380's four suites for BLS12-381: the simplified SWU map, over `F_p²` for `G2`, and the 11- and 3-isogenies; for BN254's `G1`, gnark-crypto's suite by the Shallue–van de Woestijne map | `hash_to_curve`: two maps, an addition and a clearing |

Measured on an Apple M4 Max on one thread, in the default release profile, the median of 21 samples alternated in one process, against the fastest of mcl 4.44.0 (its 384-bit build), blst 0.3.17, arkworks 0.5, the `bls12_381` crate 0.8 and substrate-bn 0.6 that does the same work; the page has the conditions, every operation and all the losses with their causes. The pairing takes 1.09 (BLS12-381) and 1.14 (BN254) times mcl's time, 0.972 of blst's, and the pairings and multi-pairings 0.510 to 0.540 of arkworks's; mcl is faster at the Miller loop (1.19), the products of `GT` (1.30, 1.31) and by scalars on `G2` (1.28, 1.34), and blst at hashing to `G2` (1.23 to 1.33); affine coordinates, the encodings to bytes and Ethereum's precompiles take 0.449 to 0.786 of the fastest reference's time.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `pairing` | BLS12-381 | 306 µs | mcl 281 µs |
| `pairing` | BN254 | 216 µs | mcl 189 µs |
| `multi_pairing` | BLS12-381, 16 pairs | 1.69 ms | mcl 1.53 ms |
| `G1 * k` | BLS12-381 | 35.3 µs | mcl 32.8 µs |
| `G2 * k` | BLS12-381 | 71.7 µs | mcl 56.0 µs |
| `Gt::pow` | BLS12-381 | 110 µs | mcl 97.0 µs |
| `G2::from_zcash`, compressed | BLS12-381 | 51.5 µs | blst 40.9 µs |
| `ethereum::ec_pairing` | 2 pairs | 367 µs | arkworks 704 µs |
| `hash_to_curve` | BLS12-381's `G2` | 119 µs | blst 89.8 µs |

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/pairing.md) verified:

| Part | Against | Verified |
| --- | --- | --- |
| The pairing | arkworks 0.5, the `bls12_381` crate 0.8, blst 0.3, RELIC, py_ecc 8, the IETF draft | 6 seeded pairings of each curve and their product as arkworks', and the six on BLS12-381 as the other two crates'; `e(g₁, g₂)` as RELIC's; the IETF draft's and py_ecc's values by the documented power |
| Ethereum | go-ethereum's tests of the precompiles | 16 vectors of `ecAdd`, 19 of `ecMul` and 14 of `ecPairing` |
| ZCash encodings | the `bls12_381` crate | the compressed and uncompressed encodings of the first 1,000 multiples of each generator, byte for byte; 123 corrupted encodings accepted or refused as it does |
| Subgroups | arkworks 0.5, PARI/GP 2.19 | membership and clearing of points outside the subgroups; points of every prime order dividing the cofactors refused |
| Hashing to G1 and G2 | RFC 9380, arkworks 0.5, the `bls12_381` crate 0.8, gnark-crypto's vectors | the 20 vectors of appendices J.9 and J.10; 5,000 random messages, 20,000 elements and 30,000 points as theirs; BN254's 10 published vectors |

### Complex numbers

| Page | Contents |
| --- | --- |
| [Complex numbers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/complex.md) | `a::math::complex`: `Complex<f32>` and `Complex<f64>`, their arithmetic, `abs`, `arg`, `exp`, `ln`, `sqrt` and powers, with C99's branch cuts; division and square roots that don't overflow before their results do |

Measured on an Apple M4 Max for `Complex<f64>`, in nanoseconds per operation over 1,024 independent ones, against num-complex 0.4.6, whose division by `|b|²` as it is overflows from `|b|` of about 10¹⁵⁴:

| Operation | `Complex` | num-complex |
| --- | ---: | ---: |
| `a * b` | 0.43 | 0.44 |
| `a / b` | 1.99 | 0.88 |
| `sqrt` | 4.41 | 9.12 |
| `exp` | 5.15 | 5.46 |
| `ln` | 7.54 | 7.40 |
| `powc` | 15.75 | 15.48 |

### Quaternions

[`a::math::quaternion`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/quaternion.md) has `Quaternion<f32>` and `Quaternion<f64>`, `w + x·i + y·j + z·k` with Hamilton's rules, for the algebra and for rotations: a unit quaternion is a rotation of three-dimensional space, and the methods that read a quaternion as a rotation take it as `q/|q|`, whatever its norm.

| Operations | Functions | Method |
| --- | --- | --- |
| [The algebra](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/quaternion.md#the-algebra) | `+`, `-`, `*` (Hamilton's product), `/` (on the right), `conj`, `norm`, `normalize`, `recip`, `dot` | the norm and quotients scaled by powers of two at the ends of the range; a product that overflows has infinities in its parts, not NaNs |
| Functions | `exp`, `ln`, `sqrt`, `powi`, `powf`, `powq` | the principal values, those of the complex functions with the vector's direction for `i`, and `Complex`'s branch cuts |
| [Rotations](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/quaternion.md#rotations) | `from_axis_angle`, `to_axis_angle`, `from_rotation_vector`, `to_rotation_vector`, `rotate`, `to_rotation_matrix`, `from_rotation_matrix`, `slerp`, `nlerp`, `angle_to`, `from_rotation_arc` | `q·v·q⁻¹`, the right factor first; Shepperd's method for matrices; Shoemake's slerp on the shorter arc |
| [Euler angles](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/quaternion.md#euler-angles) | `from_euler`, `to_euler`: 12 sequences, intrinsic or extrinsic | SciPy's conventions, ranges and choice at gimbal lock; Bernardes and Viollet's direct method |

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/quaternion.md) verified:

| Part | Against | Verified |
| --- | --- | --- |
| The algebra | mpmath at 60 digits, 160-bit `BigFloat`, `Complex` | `exp`, `ln`, `sqrt` and powers on 24 hard inputs, the largest error seen 1.00 ε times the condition; identities on 2,000 random triples; 4,000 products with parts up to `2¹⁰⁰⁰`, 2,555 of them overflowing, against 160-bit ones |
| Rotations | SciPy 1.18, glam 0.33, nalgebra 0.35 | matrices, rotation vectors, `apply`, `Slerp` and `align_vectors` within `5.3·10⁻¹⁶` of SciPy's on random rotations; 20,000 random cases as each crate's |
| Euler angles | SciPy 1.18, glam 0.33, nalgebra 0.35 | all 24 conventions within `4.4·10⁻¹⁶` of SciPy's, and its choice at gimbal lock |

Measured on an Apple M4 Max on one thread on 2026-11-26, in ns a call, against the faster of glam 0.33 and nalgebra 0.35 (the default release profile on both sides); the page has the conditions and every loss. The algebra and the Euler angles do the same work on every side: of their 38 cases `a` is faster in 6 (`sqrt` 0.43 to 0.56, `f32` `mul` 0.85) and slower in 28, the overflow-safe paths of the products, quotients, `exp` and `ln` running on every call (1.15 to 3.05). The rotations read any quaternion as `q/|q|`, where glam and nalgebra take unit ones, so those rows are not counted: they take 0.69 to 12 times the references' time.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `a * b` | `f32` | 1.11 ns | glam 1.30 ns |
| `sqrt` | `f64` | 5.89 ns | nalgebra 13.7 ns |
| `ln` | `f64` | 13.4 ns | nalgebra 5.38 ns |
| `to_euler`, ZXZ intrinsic | `f64` | 21.6 ns | glam 20.6 ns |

### Dual numbers

[`a::math::dual`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md) differentiates functions by forward-mode automatic differentiation: `Dual<T>` is `re + eps·ε` with `ε² = 0`, and `HyperDual<T>` carries a second derivative too. Both are generic over the trait `DualNum` and nest, so that a function written once over `D: DualNum` gives its value for `f64` and its derivatives for the dual types, exact up to the rounding of the arithmetic but for [one rule at `0·∞`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#limits).

| Operations | Functions | Method | Passes |
| --- | --- | --- | --- |
| Arithmetic | `+`, `-`, `*`, `/`, with a float on the right (and on the left for the concrete types), comparisons of the values | the rules of the algebra | |
| [Functions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#the-derivatives-at-the-edges) | powers and roots, exponentials and logarithms, the circular and hyperbolic functions and their inverses, `atan2`, `hypot` | the value from the standard library, the derivatives formed so that they keep their digits | |
| [Derivatives](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#forward-mode-and-what-it-costs) | `derivative`, `second_derivative` | a dual pass; a hyper-dual pass (Fike and Alonso) | 1 |
| Gradients, Jacobians | `gradient`, `jacobian` | a dual pass a variable | `n` |
| Hessians | `hessian`, `value_gradient_and_hessian` | a hyper-dual pass an entry | `n(n + 1)/2` |

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md) verified:

| Part | Against | Verified |
| --- | --- | --- |
| The functions | SymPy 1.14 and mpmath 1.3 at 600 digits | 514 points of 54 functions in `f64` and `f32`, through every form: 8,733 components within their bounds, 4 ulps for values and 8, 16 and 32 for the first three derivatives (eight times that for high powers and compositions) |
| num-dual | num-dual 0.16 | 417 points of 45 functions: within 2 ulps in the first derivative and 7 in the second, but where num-dual's own error is the larger |
| [With the solvers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#with-the-numerical-methods) | `a::math::numerics` | Newton's and Halley's methods, BFGS, L-BFGS-B, Levenberg–Marquardt, Newton's method for systems and the stiff methods, with exact derivatives, gradients and Jacobians |

Measured on an Apple M4 Max on one thread on 2026-11-26, in ns or µs a call, against num-dual 0.16 (the default release profile on both sides); the page has the conditions and every loss. 64 of the 70 elementary functions of `Dual` and `HyperDual` are at par or faster (0.38 to 1.00), and the Hessians of Rosenbrock's function from 4 variables take 0.15 to 0.28 of num-dual's time; gradients, Jacobians and Hessians of few variables are slower (2.00 to 4.37 at 2 variables), each variable a pass where num-dual carries all of them in one.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `HyperDual` `exp` | `f64` | 2.52 ns | num-dual 4.47 ns |
| `Dual` `pow` | `f64` | 12.8 ns | num-dual 4.68 ns |
| `gradient`, Rosenbrock | 2 variables | 15.8 ns | num-dual 3.62 ns |
|  | 32 variables | 1.20 µs | num-dual 4.75 µs |
| `hessian`, Rosenbrock | 32 variables | 39.4 µs | num-dual 259 µs |

### Matrices

[`a::math::matrix::Matrix<T>`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md) is a dense matrix of any size, column after column, of `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `Number`, `Rational`, [`BigFloat` or `BigComplex`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/float.md#matrices-of-floats). The algorithms are LAPACK's, arranged so that most of their work is products of blocks, and the products run on NEON kernels; real matrices stay in real arithmetic. `ModMatrix` and `BitMatrix` are dense matrices over `GF(p)`, a word an element, and over `GF(2)`, a bit an element, row after row. Costs are for `n × n` matrices.

| Operations | Functions | Method | Cost |
| --- | --- | --- | --- |
| Arithmetic | `+`, `-`, `*`, by an element, `hadamard`, `transpose`, `conjugate_transpose`, row and column operations, `trace`, `pow`, `powi` | the product by Goto and van de Geijn's blocking | 2n³ for a product |
| LU | `lu`, `det`, `solve`, `inverse` | partial pivoting, recursive; the inverse as LAPACK's `getri` | 2n³/3, the inverse 2n³ |
| Cholesky | `cholesky` | recursive, blocked | n³/3 |
| QR | `qr`, `q`, `q_full`, least squares | Householder reflections in blocks, the panels halved recursively | 4n³/3 |
| Singular values | `svd`, `singular_values`, `rank`, `null_space`, `norm`, `cond` | a blocked reduction to bidiagonal form, then divide and conquer, or dqds for the values alone | 9.3n³, the values 8n³/3 |
| Hermitian eigenvalues | `hermitian_eigen`, `hermitian_eigenvalues` | a blocked reduction to tridiagonal form, then divide and conquer | 4.7n³, the values 4n³/3 |
| Eigenvalues | `schur`, `eigen`, `eigenvalues`, `characteristic_polynomial` | a blocked reduction to Hessenberg form, then the multishift QR algorithm with aggressive early deflation | about 25n³, the values 10n³ |
| Functions | `exp`, `sqrt`, `log` | scaling and squaring with Padé approximants; the Schur method; inverse scaling and squaring | `exp` a few products, `sqrt` and `log` a Schur decomposition |
| Exactly, on `Number` and `Rational` | `det`, `rank`, `solve`, `inverse`, `inverse_rational`, `null_space`, `characteristic_polynomial`, `adjugate`, `*` | Bareiss's elimination on the rows cleared of their denominators; for larger matrices the determinant by residues modulo primes or by Dixon's divisor, by the order and the length of the integers, and the systems by Dixon's p-adic lifting, a singular matrix proved singular; Strassen–Winograd's product of numbers; Berkowitz's method for the polynomial | n³/3 products of numbers as long as the minors, or O(n³) operations on words for each prime or step of the lifting; 3n⁴/2 for the polynomial |
| [Normal forms](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#normal-forms), on `Number` | `hermite_normal_form`, `hermite_form`, `smith_normal_form`, `elementary_divisors` | rank profiles modulo a prime, proved; the Hermite form by Pernet and Stein's method from order 8 (residues and a Dixon solve), else modulo the determinant `D` of a minor (Domich, Kannan and Trotter); the elementary divisors modulo it; the Smith transformations by alternating Hermite forms (Kannan and Bachem) | n³ operations on numbers below `D` |
| [Over GF(p)](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#matrices-over-gfp), `ModMatrix` | `+`, `-`, `*`, `ple`, `rank`, `det`, `inverse`, `solve`, `rref`, `null_space` | products reduced once, Strassen–Winograd from 128; the recursive PLE decomposition | n³/3 products of words, O(n^2.81) from 128 |
| [Over GF(2)](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#matrices-over-gf2), `BitMatrix` | `+`, `*`, `mul_strassen`, `transpose`, `rank`, `det`, `inverse`, `solve`, `null_space`, `reduced_row_echelon_form` | the method of the Four Russians (M4RM), Strassen–Winograd from 8192; Bard's elimination (M4RI) | n³/(64·log n) word additions, O(n^2.81) from 8192 |

In `f64` on one core of an Apple M4 Max, each of the twelve operations [measured](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#measurements) was at least as fast as faer's and nalgebra's at every size from 8 × 8 to 1024 × 1024; at 1024 × 1024, in milliseconds:

| `n = 1024` | `a` | faer | nalgebra |
| --- | --- | --- | --- |
| `a * b` | 34.9 | 38.4 | 37.1 |
| `lu()` | 13.8 | 17.2 | 59.9 |
| `inverse()` | 40.1 | 50.2 | 221 |
| `cholesky()` | 7.4 | 8.5 | 27.8 |
| `qr()` | 29.0 | 32.5 | 90.7 |
| `svd()` | 244 | 331 | 5,886 |
| `hermitian_eigen()` | 114 | 146 | 396 |
| `eigen()` | 498 | 760 | – |
| `exp()` | 512 | – | 778 |

Apple's Accelerate, which runs its products on the M4's matrix unit, is faster than all three at this size; it has no `exp`.

### Lattices and exact linear algebra

[`a::math::lattice`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/lattice.md) reduces the bases of integer lattices, the rows of a `Matrix<Number>`, which may be dependent; finds their shortest and closest vectors; and finds integer relations among real numbers. A reduction returns the reduced basis, a zero row first for each dependency, and the unimodular `U` with `U·B = B'`. The reductions and the enumerations are exact: floating point steers them, and its results are proved or computed again in integers. The normal forms and the matrices over `GF(p)` and `GF(2)` are in the table of matrices above. Costs are for `n` rows of entries of `log B` bits.

| Operations | Functions | Method | Cost |
| --- | --- | --- | --- |
| LLL | `lll`, `is_lll_reduced` | the integral LLL of de Weger and Cohen, Pohst's MLLL for dependent rows | O(n²·log B) exchanges on integers of O(n·log B) bits |
| L² | `lll_fp` | Nguyen and Stehlé's L², in 53-bit floats with an exponent, then in `BigFloat`s; the result proved by interval arithmetic | quadratic in log B |
| Deep insertions | `deep_lll` | Schnorr and Euchner, on the machinery of L² | no polynomial bound |
| Enumeration | `shortest_vector`, `closest_vector`, `short_vectors` | Fincke and Pohst in Schnorr and Euchner's order, in floats with rigorous error bounds or in integers | 2^O(n²) nodes, to a rank of about 45 |
| BKZ | `bkz` | Schnorr and Euchner, as fplll's BKZ 1.0 without pruning: tours in floats, then exact ones | an enumeration of rank β a block, β up to about 30 |
| Integer relations | `pslq`, `pslq_f64` | Ferguson and Bailey's PSLQ, `H` in fixed point and `A`, `B` and `y` exact; the bound proved | O(n²) multipliers an iteration |

What the pages verified:

| Part | Against | Verified |
| --- | --- | --- |
| [LLL, L², deep insertions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/lattice.md#lll-reduction) | fplll 5.5, FLINT 3.6, NTL 11.6 | the ranks of 66 bases; fplll's first vector from `lll` on all 49 bases of up to 12 rows, NTL's from `deep_lll` on 24 of 24; every result reduced by the exact test, with `U·B = B'` and `det U = ±1` |
| [Enumeration](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/lattice.md#shortest-and-closest-vectors) | fplll 5.5, PARI 2.19, an exact rational enumeration | the norms of 138 shortest vectors, the distances of 91 closest ones, and 75 lists of short vectors |
| [BKZ](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/lattice.md#bkz) | fplll 5.5, two exact checks of every block | every block's first vector a shortest one of its block; the first vector's squared norm within 1.0044 times fplll's on 36 bases |
| [PSLQ](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/lattice.md#integer-relations) | mpmath 1.3, PARI's `lindep`, a reference PSLQ | the relations and bounds of 17 cases; 1,000 random vectors (20,000 in a longer run), every relation and bound checked exactly |
| [Normal forms](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#normal-forms) | FLINT 3.6, PARI 2.19 | `H`, `U` and the Smith form of 46 matrices and of 1,800 more in sweeps, of up to 3000 bits |
| [GF(p)](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#matrices-over-gfp) | FLINT 3.6, NTL 11.6 | the rank, determinant, reduced form, inverse, null space and product of 456 matrices modulo 13 primes from 2 to `2^64 − 59` |
| [GF(2)](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#matrices-over-gf2) | M4RI, NTL 11.6, sympy 1.14 | 235 products, echelon forms, inverses and systems, as M4RI's to the last bit, up to `4096 × 4096` |
| [Determinants and Dixon's lifting](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#verified-products-determinants-and-systems-of-integers) | FLINT 3.6 | 58 determinants across the three methods; 34 systems, the singular ones proved singular |

Timed on an Apple M4 Max on one thread against the fastest of fplll 5.5, FLINT 3.6, NTL 11.6, M4RI, PARI 2.19 and mpmath 1.3 that does the same work — computes `U`, proves its result, reduces as far; the pages have the conditions, the second figures against the references that do less, and the causes. Of 321 cases, `a` is faster on 190, within 5 % on 33 and slower on 98. The rows of `Matrix<Number>` are the run of the matrices' page (`bench/lattice`), from 16 × 16; at 16 × 16, which the rationals' page (`bench/rational`) times too, the two runs give different figures and fastest references.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `lll_fp` | knapsack, 100 numbers of 1000 bits | 3.87 s | PARI `qflll` 1.71 s |
| `lll_fp` | NTRU-like, rank 160 | 1.85 s | FLINT 6.38 s |
| `lll`, exact | knapsack, 40 numbers of 400 bits | 61.8 ms | NTL 169 ms |
| `shortest_vector` | Goldstein–Mayer, rank 44 | 1.38 s | fplll 2.50 s |
| `pslq` | a polynomial of degree 16, 400 bits | 1.31 ms | mpmath 707 ms |
| `hermite_form` | 200 × 200, 10 bits | 39.0 ms | FLINT 151 ms |
| `Matrix<Number>::det` | 64 × 64, `−99..=99` | 585 µs | FLINT 608 µs |
| `Matrix<Number>::solve` | 128 × 128, 64 bits | 22.7 ms | FLINT 38.4 ms |
| `Matrix<Number> * Matrix<Number>` | 256 × 256, `−99..=99` | 1.79 ms | FLINT 5.66 ms |
| `ModMatrix * ModMatrix` | 1024 × 1024, `p = 2^31 − 1` | 114 ms | NTL 169 ms |
| `ModMatrix::ple` | 2048 × 2048, `p = 2^31 − 1` | 395 ms | FLINT 932 ms |
| `BitMatrix * BitMatrix` | 16,384 × 16,384 | 922 ms | M4RI 1.37 s |
| `BitMatrix::rank` | 16,384 × 16,384 | 802 ms | M4RI 546 ms |

### Sparse matrices

[`a::math::sparse`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/sparse.md) holds matrices of `f32`, `f64`, `Complex<f32>` or `Complex<f64>` by their nonzeros, and solves with them in time and memory that follow the nonzeros and the fill of the factors. Permutations are `p[new] = old`; a factorisation that does not exist is `None`, and a matrix singular to working precision is reported, never solved.

| Operations | Functions | Method | Cost |
| --- | --- | --- | --- |
| Storage | `SparseMatrix` (`CsrMatrix`), `CscMatrix`, `CooMatrix`, the conversions, `read_matrix_market`, `write_matrix_market` | canonical compressed rows and columns; triplets summed by radix sorts | nnz + n |
| Operations | `mul_vec`, `mul_vec_transpose`, `mul_vec_adjoint`, `*`, `+`, `-`, `transpose`, `submatrix`, `select`, `permute`, `norm`, `LinearOperator` | Gustavson's row-by-row product | the multiplications |
| Orderings | `reverse_cuthill_mckee`, `amd`, `colamd`, `nested_dissection`, `elimination_tree`, `postorder` | George and Liu; approximate minimum degree; its column form; separators by level sets and multilevel bisection | about nnz in practice, the dissection 8 to 14 times `amd`'s time |
| Cholesky | `SparseCholesky`, `SparseLdl` (`L·D·Lᴴ`), `solve`, `log_det`, `inertia` | supernodal, left-looking, the blocks by the dense `gemm` and `trsm` | the arithmetic of the factor |
| LU | `SparseLu`, `solve`, `solve_transpose`, `is_singular`, `rcond` | Gilbert and Peierls, then supernodal panels (order 64 or more, once a column of the factors holds 96 elements); an automatic strategy (AMD of `A + Aᵀ` and diagonal pivots down to 10⁻⁴ of their column for a nearly symmetric pattern, falling back to COLAMD and partial pivoting when the fill passes twice what that order predicts or the elements grow 16-fold); a condition estimate | the arithmetic of the factors |
| QR | `SparseQr`, `solve` (least squares), `apply_q`, `rank` | multifrontal Householder reflections | the reflections in the fronts |
| Iterative | `cg`, `minres`, `gmres`, `bicgstab`, `Jacobi`, `Ilu0`, `Ic0` | Krylov methods, the residual verified afresh | a product with the matrix a step |
| Eigenvalues | `lanczos`, `arnoldi`, `lanczos_shift_invert`, `arnoldi_shift_invert` | implicit restarts as ARPACK's; shift-and-invert by `SparseLu` | products, Gram–Schmidt and a dense eigensolve of order `m` a restart |

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/sparse.md) measured, against SciPy 1.18 and the libraries behind it:

| Part | Against | Measured |
| --- | --- | --- |
| Matrix Market | `scipy.io.mmread`, `mmwrite` | the 22 files SciPy writes read bit for bit; the 44 written read back by SciPy with no difference |
| Orderings | SciPy's RCM, SuperLU's MMD and COLAMD, an exact minimum degree | RCM SciPy's band and profile on 12 of 15 graphs; `amd` 0.945 to 1.049 of the better reference's fill; `colamd` SuperLU's COLAMD fill on its two random square matrices |
| Cholesky, `L·D·Lᴴ` | mpmath at 50 digits | solutions within 3.3·10⁻¹⁶ in `f64` |
| LU | `spsolve`, `splu` | solutions within 0.14 `cond·ε`; SuperLU's row pivots on all seven systems in the natural column order |
| QR | `numpy.linalg.lstsq` | solutions within 2.49 `κ·ε` in `f64` |
| Iterative | scipy's `cg`, `minres`, `gmres`, `bicgstab` | every count of 216 runs within scipy's own spread under rounding, or at most two outside it |
| Eigenvalues | ARPACK (`eigsh`, `eigs`) | within 2.6·10⁻¹⁴ of ARPACK's on random sparse symmetric matrices in `f64` |

Timed on an Apple M4 Max on one thread against the fastest of SuiteSparse 7.14 (CHOLMOD, UMFPACK, KLU, SPQR, CSparse, AMD, METIS), Apple's Accelerate, faer 0.24, sprs 0.11, nalgebra-sparse 0.11 and SciPy 1.18 that does the same work; the page has the conditions, the matrices and the causes. Of 272 cases, `a` is faster on 45, within 5 % on 35 and slower on 192; the dense blocks of Accelerate's factorisations run on the M4's matrix unit, which `a` cannot reach without assembly for SME.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `mul_vec` | 2-D Laplacian, 10⁶ unknowns | 1.74 ms | faer 1.85 ms |
| `SparseCholesky::with_ordering` | 2-D Laplacian, 99,856 unknowns | 29.1 ms | Accelerate 30.0 ms |
| `SparseCholesky::new` | 3-D Laplacian, 46³ | 2.46 s | Accelerate 573 ms (CHOLMOD 1.28 s) |
| `SparseLu::new` | 2-D Laplacian, 99,856 unknowns | 156 ms | UMFPACK 135 ms |
| `SparseLu::new` | circuit, 10⁴ nodes | 65.2 ms | Accelerate 62.5 ms |
| `SparseQr::new` | grid least squares, 100,101 × 33,489 | 51.3 ms | Accelerate 38.1 ms |
| `SparseQr::solve_vec` | curve, 99,999 × 50,000 | 1.14 ms | Accelerate 5.17 ms |
| `cg_with` | 2-D Laplacian, 10⁶ unknowns | 8.78 s | SciPy 8.55 s |
| `gmres_with` | upwind grid, 99,856 unknowns | 1.65 s | Accelerate 372 ms |
| `lanczos_with`, 6 eigenvalues | anisotropic grid, 99,856 unknowns | 9.06 s | ARPACK 3.67 s |
| `amd` | 2-D Laplacian, 10⁶ unknowns, scrambled | 873 ms | SuiteSparse CAMD 535 ms |
| `nested_dissection` | the same | 7.39 s | METIS 3.42 s |
| `read_matrix_market` | a file of 92.8 MB | 288 ms | fast_matrix_market 202 ms |

### Numerical methods

[`a::math::numerics`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/numerics.md) solves the standard problems of numerical analysis on `f32` and `f64` in generic code, over the sealed trait `Real`; its linear algebra is `a::math::matrix`. A method never reports an answer it has not verified: one that cannot reach its tolerance returns `None`, or its best point or estimate with `converged` false.

| Page | Contents |
| --- | --- |
| [Numerical methods](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/numerics.md) | `a::math::numerics`, with `_with` forms that take options: the zeros of a function by `bisect`, `brent`, `secant`, `newton` and `halley`, and of a system by `newton_system`, `broyden` and `broyden_with_jacobian`; integrals by `quad`, `quad_g10k21`, `quad_g7k15`, `quad_half_infinite`, `quad_infinite`, `clenshaw_curtis`, `tanh_sinh`, `cubature`, `monte_carlo` and `quasi_monte_carlo` on `Sobol`, and the Gaussian rules `legendre`, `hermite`, `hermite_scaled`, `laguerre`, `laguerre_scaled` and `jacobi`; `y′ = f(t, y)` by `solve`, `integrate`, `solve_at` and `solve_ivp`, with `DenseOutput` and `Event`, by `Rk4`, `DormandPrince5`, `DormandPrince853` and the stiff `RadauIia` and `Bdf`; minima by `golden_section`, `minimize`, `nelder_mead`, `conjugate_gradient`, `bfgs` and, within bounds, `lbfgsb`, sums of squares by `gauss_newton` and `levenberg_marquardt`, and linear programmes by `LinearProgram`; `Spline` (cubic, clamped, not-a-knot, Akima, PCHIP), `chebyshev`, `chebyshev_nodes`, `barycentric_weights`, `barycentric_interpolate`, `aaa` and `aaa_complex`; and `derivative` and `second_derivative` by Richardson extrapolation |

Exact derivatives for these methods, gradients, Jacobians and Hessians, come from the [dual numbers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#with-the-numerical-methods) of `a::math::dual`, with no step to choose.

The page measures the methods against SciPy 1.18 and the codes behind it, and against mpmath. `brent` takes exactly the steps of `brentq` on 57 of 63 brackets; `quad` converges on all 46 integrable integrals of a battery of 48, one more than `scipy.integrate.quad`, none of them beyond its own error estimate; `DormandPrince5` takes the steps of DOPRI5 in every run, `RadauIia` 0.92 to 1.01 times those of SciPy's `Radau` and `Bdf` 0.94 to 1.00 times those of its `BDF` on five stiff problems; `lbfgsb` ends on SciPy's active set on 42 of 43 problems; and the simplex method gives HiGHS's status on 600 random programmes and its optima to 5.0·10⁻¹³.

Timed on an Apple M4 Max on one thread against the fastest of GSL 2.8, Boost 1.92, argmin 0.11, ode_solvers 0.6, roots 0.0.8, SciPy 1.18 and HiGHS 1.12 that does the same work; the page has the conditions, every problem and the causes of the losses. Of the 263 cases with such a reference, `a` is faster on 126, within 5 % on 15 and slower on 122 — most of them in quadrature, the Gaussian rules, the initial value problems and Newton's and Broyden's methods for systems; the simplex method is faster than HiGHS on 12 of its 13 programmes.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `brent`, `x² − p` | 64 brackets | 93.9 ns | Boost 103 ns |
| `newton_system`, Rosenbrock | n = 2 | 2.58 µs | GSL 431 ns |
| `quad`, a smooth integrand | 21 evaluations | 64.2 ns | Boost 11.8 ns |
| `tanh_sinh`, `x^(−1/2)` on `[0, 1]` | 16 instances | 1.04 µs | Boost 90.3 ns |
| `cubature`, Genz product peak | d = 2 | 86.9 µs | SciPy 6.34 ms |
| `legendre` | n = 100 | 198 µs | GSL 143 µs |
| `hermite` | n = 10⁴ | 1.93 s | SciPy 14.5 ms |
| `DormandPrince5`, Arenstorf orbit | `rtol = 10⁻⁶` | 22.8 µs | ode_solvers 19.5 µs |
| `Bdf`, Van der Pol, μ = 1000 | `rtol = 10⁻⁸` | 570 µs | GSL 890 µs |
| `bfgs`, chained Rosenbrock | n = 50 | 478 µs | argmin 15.2 ms |
| `lbfgsb`, Rosenbrock in a box | n = 10⁴ | 3.66 ms | SciPy 14.8 ms |
| `LinearProgram::solve`, dense | 550 rows, 1000 variables | 226 ms | HiGHS 789 ms |
| `LinearProgram::solve`, density 0.02 | 550 rows, 1000 variables | 57.1 ms | HiGHS 44.0 ms (interior point) |
| `Spline::evaluate` | 10⁴ knots, a point | 11.4 ns | GSL 27.9 ns |
| `aaa`, the absolute value | 1000 samples | 12.2 ms | SciPy 8.94 ms |

## Hash

[`a::hash`](https://github.com/nettrash/a/blob/HEAD/lib/docs/hash/sha2.md) computes the hash functions SHA-224, SHA-256, SHA-384 and SHA-512 of FIPS 180-4 and RFC 9380's `expand_message_xmd`, which hashing to curves builds on. The compression functions take the ARMv8 SHA-256 and SHA-512 instructions, and x86_64's SHA extensions for SHA-256, where the CPU has them, and portable Rust elsewhere; they are not constant-time.

| Page | Contents |
| --- | --- |
| [SHA-2](https://github.com/nettrash/a/blob/HEAD/lib/docs/hash/sha2.md) | `Sha224`, `Sha256`, `Sha384` and `Sha512`, streaming hashers (`new`, `update`, `finalize`; `Clone` forks a hash midway); `sha224`, `sha256`, `sha384` and `sha512` in one call; the trait `Digest` over them; `expand_message_xmd::<H>` on any `Digest`, with the hashing of tags longer than 255 bytes |

Measured on an Apple M4 Max on one thread, in the default release profile, the median of 21 samples alternated in one process, against RustCrypto's `sha2` 0.11.1 and 0.10.9, ring 0.17.14, OpenSSL 3.5.9 and 4.1.0, blst 0.3.17 and RustCrypto's `elliptic-curve` 0.13.8; the page has the conditions and all the losses. SHA-512 and SHA-384 take 0.943 to 1.00 of the fastest reference's time from 64 bytes (1.26 and 1.24 times `sha2` 0.11's for the empty message), SHA-256 and SHA-224 0.751 to 1.00 up to 64 bytes and 1.16 to 1.20 from 1 KiB, against OpenSSL's and ring's hand-scheduled assembly (`sha2` 0.11's intrinsics run at this crate's speed); `expand_message_xmd` takes 1.20 to 1.35 of blst's time with SHA-256 and 0.468 to 0.523 of `elliptic-curve`'s with SHA-384 and SHA-512.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `sha256` | 64 B | 32.1 ns | RustCrypto `sha2` 0.11 34.9 ns |
| `sha256` | 1 MiB | 362 µs | ring 302 µs |
| `sha512` | 1 MiB | 526 µs | RustCrypto `sha2` 0.11 528 µs |
| `expand_message_xmd::<Sha256>` | 96 bytes | 255 ns | blst 194 ns |

What the page verified:

| Part | Against | Verified |
| --- | --- | --- |
| SHA-2 | FIPS 180-4, NIST's CAVP vectors, OpenSSL 4.1, RustCrypto's `sha2` 0.10 | FIPS 180-4's examples, a million `a`s and 1 GiB; 44 short and 8 long messages and 40 Monte Carlo checkpoints of CAVP; 1,204 digests of every length to 300 bytes as OpenSSL's; 10,005 as RustCrypto's |
| `expand_message_xmd` | RFC 9380, RustCrypto's `elliptic-curve` 0.13 | the 30 vectors of appendix K; 20 more cases up to 255 blocks; 3,000 random messages and tags as RustCrypto's |

## Not constant-time

Nothing here is written to take the same time whatever its operands: `pow_mod` (of `Number` and of the fixed-width `Uint<N>`, with `Montgomery<N>`'s arithmetic), the finite fields, the elliptic curves, the pairings and the hashing to curves in particular branch on the values of their exponents, scalars and elements; the fixed-width integers' products, quotients and inverses take a time that depends on their values; and the hash functions on the lengths of their messages. The quaternions and the dual numbers are no exception. Use them on public data, for verification and research, not on secret keys or scalars.

## Changelog

What each version added, changed and fixed is in [CHANGELOG.md](https://github.com/nettrash/a/blob/HEAD/CHANGELOG.md).

## License

[MIT](https://github.com/nettrash/a/blob/HEAD/LICENSE).

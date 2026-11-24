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

Every algorithm has a page with its method, examples and measurements against the reference implementations (GMP, MPFR, GNU MPC, PARI, FLINT, Arb, NTL, primesieve, GMP-ECM, OpenSSL, malachite, faer, nalgebra, FFTW, rustfft, Apple's Accelerate, SciPy and others); the tables below link to them. The pages of the lattices, of the newest exact linear algebra, of the finite fields, the pairings, the curves over binary fields, the hashing to curves, SHA-2, the fixed-width integers, the quaternions and the dual numbers say what they were verified against (fplll, FLINT, PARI, NTL, M4RI, GMP, mpmath, sympy, SciPy, OpenSSL, arkworks, the `bls12_381` crate, blst, py_ecc, RustCrypto, ruint, crypto-bigint, bnum, glam, nalgebra, num-dual, and the published test vectors of NIST, the RFCs, Ethereum and gnark-crypto), and have no timings yet. The API reference is on [docs.rs](https://docs.rs/nettrash-a). The crate needs Rust 1.88 (edition 2024). Its fastest paths are NEON and assembly kernels on aarch64, and assembly for the products of integers and SSE2 and AVX2 kernels for the Fourier transforms on x86_64; the binary fields multiply by the carry-less products `PMULL` and `PCLMULQDQ` where the CPU has them; everything else is portable Rust.

## Usage

Add the crate with `cargo add nettrash-a`, or to `Cargo.toml`:

```toml
[dependencies]
nettrash-a = "0.1"
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
| [Squares modulo a number](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/quadratic.md) | `jacobi` and `kronecker` by the binary method up to six limbs (in assembly on aarch64) and Lehmer's beyond; `sqrt_mod` and `sqrt_mod_all` modulo any number by Tonelli–Shanks, Atkin's method, Cipolla, Hensel's lifting and the Chinese remainder theorem; Cornacchia's `x² + d·y² = m` |
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
| `is_probable_prime` of a prime | 256 bits | 49,258 | 170,831 | – |
| | 2048 bits | 16.1 M | 21.6 M | – |

Number theory, measured on an Apple M4 Max on one thread (the references too), against the fastest of primesieve 12.16, primecount 8.8, PARI 2.19, FLINT 3.6, GMP 6.3, GMP-ECM 7.0.7 and msieve 1.53 that has the function; in cycles, or in time where marked; each page has the conditions and the other contenders:

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `primes(a..=b).count()` | [10¹², 10¹² + 10¹⁰] | 909 ms | primesieve 1.05 s |
| `prime_pi(x)` | x = 10¹⁶ | 988 ms | primecount 1.35 s |
| `nth_prime(n)` | n = 10¹² | 32.8 ms | primecount 44.9 ms |
| `prev_prime` | 256 bits | 206,438 | GMP 613,061 |
| `factor`, two primes of 32 bits | 64 bits | 25.6 µs | PARI 133 µs |
| `ecm`, one curve, stage 1, `B1` = 50,000 | 256 bits | 7.73 ms | GMP-ECM 16.9 ms (`-param 1`) |
| `ecm`, one curve, stage 2 to `100·B1`, `B1` = 250,000 | 512 bits | 22.5 ms | GMP-ECM 41.2 ms (`-param 0`) |
| `pm1`, stage 2 to `100·B1`, `B1` = 10⁶ | 512 bits | 8.91 ms | GMP-ECM 22.5 ms |
| `factor`, two primes | 28 digits | 0.542 ms | PARI 0.832 ms |
| `factor`, two primes | 40 digits | 5.14 ms | PARI 12.0 ms |
| `factor::siqs`, two primes | 60 digits | 0.33 s | PARI 0.99 s |
| | 80 digits | 26.0 s | msieve 93.2 s |
| | 100 digits | 2,340 s | msieve 7,430 s |
| `kronecker` | 1024 bits | 13,115 | FLINT 17,048 |
| `Factorization::sqrt_mod` | prime of 256 bits | 22,022 | FLINT 65,840 |
| `crt` | 1000 primes of 64 bits | 4.84 M | PARI 5.36 M |
| `discrete_log` | safe prime of 64 bits | 7.19 M | PARI 211 M |
| `fibonacci` | n = 10⁶ | 1.99 M | GMP 4.17 M |
| `primorial` | n = 10⁶ | 13.3 M | FLINT 20.1 M |
| `partitions` | n = 10⁶ | 138 µs | FLINT 196 µs (244 µs with its caches cleared) |

### Fixed-width integers

[`a::math::uint`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md) has `Uint<N>` and `Int<N>`, unsigned and signed integers of `N` 64-bit limbs, on the stack and `Copy`, which behave as a `u{64·N}` and an `i{64·N}` would: the operators panic on overflow when the crate is built with debug assertions and wrap without, as the primitive integers do under the default profiles, and the arithmetic has its `checked_*`, `wrapping_*`, `overflowing_*` and `saturating_*` forms. `Montgomery<N>` is the arithmetic modulo an odd number in Montgomery form, and `pow_mod` takes any nonzero modulus. Costs are for `N` limbs.

| Operations | Functions | Method | Cost |
| --- | --- | --- | --- |
| [Arithmetic](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#methods) | `+`, `-`, `*`, `/`, `%`, `&`, `\|`, `^`, `!`, shifts, rotations; `widening_mul`, `carrying_mul`, `div_rem`, `div_euclid` | carries along the limbs; schoolbook products; Knuth's algorithm D | O(N) a sum, O(N²) a product or a quotient |
| Powers, roots, logarithms | `pow`, `isqrt`, `ilog`, `ilog2`, `ilog10` | squaring and multiplying; Newton's iteration; division by the largest power of the base in a limb | O(N²·log e) a power |
| [Text and conversions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#conversions) | `Display`, `{:x}`, `{:o}`, `{:b}` with the formatter's flags, `to_str_radix`, `from_str_radix` (`const`); the primitive integers, `Number`, `resize`, floats, bytes | the bits, or divisions by the largest power of the radix in a limb; one rounding to the nearest float | O(N²) |
| [Montgomery](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#montgomery-multiplication-and-pow_mod) | `Montgomery::new`, `mul`, `square`, `pow`, `inverse`, `to_montgomery`, `from_montgomery` | Montgomery's REDC interleaved with the product (CIOS); sliding windows of 1 to 6 bits; binary inversion | 2N² + N products of limbs a product |
| Modular arithmetic | `pow_mod`, `mul_mod`, `add_mod`, `sub_mod`, `mod_inverse` | an odd modulus in Montgomery form; an even one `2ᵏ·q` modulo `q` and `2ᵏ`, then the Chinese remainder theorem | O(N²·ℓ) for an exponent of ℓ bits |

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md) verified, without timings yet:

| Part | Against | Verified |
| --- | --- | --- |
| [The integers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#the-integers) | `u64`, `i64`, `u128`, `i128`, `Number`, ruint 1.21, bnum 0.12, crypto-bigint 0.6 | about 1.4 million comparisons with the primitives, method by method, in both profiles; every width from 1 to 9 limbs, 16 and 32 against `Number`; 4,190,175 comparisons with the three crates, every difference explained |
| [Montgomery and `pow_mod`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/uint.md#montgomery-multiplication-and-pow_mod) | `Number`, GMP 6.3, Python's `pow`, ruint 1.21, crypto-bigint 0.6, bnum 0.13 | odd and even moduli of every bit length at 1 to 9 and 16 limbs; 2,312,500 random cases as GMP's; 185,000 as the three crates' |

### Rational numbers

| Page | Contents |
| --- | --- |
| [Rational numbers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/rational.md) | `a::math::rational::Rational`, a fraction of numbers in lowest terms with a positive denominator, held in two words when both parts fit in 64 bits: `+`, `-`, `*` and `/` reduced by Knuth's two gcds, with numbers and primitive integers on either side; comparisons bounded from the top bits; `floor`, `ceil`, `round`, `pow`; `Sum` and `Product` in a balanced tree; exact conversions from floats and correctly rounded ones to them, text in any radix and decimals with exponents; continued fractions and convergents, the best approximation with a bounded denominator, the simplest rational in an interval and rational reconstruction, on the half gcd; the Bernoulli numbers by the zeta function and Harvey's residues modulo primes, and the harmonic numbers by binary splitting; and the exact determinants, solutions and inverses of [matrices](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md) of rationals, by Bareiss's elimination and Dixon's p-adic lifting |

Measured on an Apple M4 Max on one thread, in cycles per call unless a time is given, over random parts of the limbs (64 bits) given, against the fastest of GMP 6.3 (`mpq_t`), FLINT 3.6 (`fmpq`), PARI 2.19, malachite-q 0.14 and num-rational 0.4 that has the function; the page has the conditions, more sizes and the other operations. Parts that fit in 64 bits are kept in two words, as FLINT and malachite-q keep them, and their arithmetic, comparisons and conversions run in registers without allocating ([#27](https://github.com/nettrash/a/issues/27)): below 2¹⁶ `+` takes 0.48 of FLINT's time and a comparison a third of GMP's. FLINT is faster at `floor`, `ceil` and `trunc` of short parts (7 to 8 cycles against 32 to 36, as they give a `Number`, which allocates) and at `Rational::new` of short parts (89 against 122); GMP is faster for a fraction of short parts with a long one (1.1 to 1.7 times), and malachite-q at hashing and at `Number::try_from` of a small integer; FLINT's continued fractions are faster from 16 to 256 limbs, its products of matrices of fractions from 32 × 32 and its determinants of 64-bit parts from 3 × 3 to 8 × 8 (0.86 to 0.97 of the time).

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `&a + &b` | parts below 2¹⁶ | 40 | FLINT 84 |
| | 1 limb | 318 | GMP 351 |
| | 16 limbs | 12,598 | malachite 15,634 |
| | 1024 limbs | 2.86 M | GMP 3.94 M |
| `&a * &b` | parts below 2¹⁶ | 70 | FLINT 95 |
| | 1 limb | 308 | GMP 538 |
| | 1024 limbs | 5.06 M | GMP 6.76 M |
| `a += &b` | parts below 2¹⁶ | 38 | FLINT 69 |
| | 2 limbs | 742 | FLINT 863 |
| `a.cmp(&b)` | parts below 2¹⁶ | 7.6 | GMP 23 |
| | 1024 limbs | 38 | num-rational 19,037 |
| `Rational::new` | 64 limbs | 52,390 | malachite 63,203 |
| `H_n` by `Iterator::sum` | n = 10⁵ | 1,200 a term | FLINT 1,504 (the same tree) |
| `Rational::try_from(f64)`, parts in words | floats in [2⁻¹¹, 2¹²) | 2.0 ns | malachite 6.4 ns (GMP into an existing `mpq_t` 10.7 ns) |
| `to_f64`, correctly rounded | parts of 64 bits | 4.6 ns | GMP 19.3 ns (truncated) |
| | 4096 bits | 13.9 ns | GMP 53.7 ns (truncated) |
| `"p/q".parse::<Rational>()` | parts below 2¹⁶ | 16.5 ns | malachite 32.2 ns |
| | parts of 64 bits | 75.4 ns | GMP 127.6 ns (with `mpq_canonicalize`) |
| | 10⁶ digits | 55.5 ms | GMP 86.8 ms (with `mpq_canonicalize`) |
| `to_string` | 10⁶ digits | 15.0 ms | FLINT 19.1 ms |
| `to_str_radix(16)` | parts of 64 bits | 35.0 ns | GMP 55.0 ns |
| `continued_fraction` | parts below 2¹⁶ | 535 | malachite 660 |
| | 64 limbs | 179,418 | FLINT 129,629 |
| | 1024 limbs | 4.79 M | FLINT 5.35 M |
| `convergents` | 16 limbs | 160,518 | malachite 284,964 |
| `limit_denominator`, a bound of half the length | 1024 limbs | 2.65 M | malachite 24.3 ms |
| `simplest_between` | 1 limb | 34 | malachite 810 |
| `reconstruct` | 1 limb | 145 | malachite 241 |
| | 1024 limbs | 1.28 M | PARI 1.81 M |
| | 100,000 limbs | 109 ms | PARI 198 ms |
| `Rational::bernoulli`, from the table | n = 10 | 10 | PARI 9 (a pointer into its cache) |
| `Rational::bernoulli` | n = 10⁴ | 3.43 ms | FLINT 5.23 ms |
| | n = 10⁶ | 7.92 s | FLINT 13.1 s |
| `Rational::bernoulli_numbers` | up to 10 | 183 | FLINT 292 (PARI 26 from its cache) |
| | up to 10⁴ | 414 ms | FLINT 548 ms |
| `Rational::harmonic`, from the table | n = 10 | 9 | FLINT 8 |
| `Rational::harmonic` | n = 10⁶ | 84.4 ms | malachite 277 ms |
| `Matrix<Rational>::det`, integers in −99..=99 | 4 × 4 | 0.05 µs | FLINT 0.15 µs |
| `Matrix<Rational>::det`, fractions of parts ≤ 99 | 3 × 3 | 0.09 µs | FLINT 0.17 µs |
| | 128 × 128 | 76.7 ms | FLINT 116 ms |
| `Matrix<Rational>::det`, 64-bit parts | 4 × 4 | 5.06 µs | FLINT 4.38 µs |
| `Matrix<Rational>::solve`, integers in −99..=99, one column | 8 × 8 | 0.67 µs | FLINT 1.90 µs (Bareiss) |
| `Matrix<Rational>::solve`, fractions of parts ≤ 99, one column | 64 × 64 | 6.88 ms | FLINT 8.80 ms |
| `Matrix<Rational>::inverse`, integers in −99..=99 | 4 × 4 | 0.29 µs | FLINT 0.89 µs |
| | 128 × 128 | 175 ms | PARI 99.6 ms |
| `Matrix<Rational>::inverse`, 64-bit parts | 16 × 16 | 34.8 ms | FLINT 45.2 ms (Bareiss; its default 81.7 ms) |
| `&a * &b`, `Matrix<Rational>`, integers in −99..=99 | 8 × 8 | 0.52 µs | FLINT 1.35 µs |
| `&a * &b`, `Matrix<Rational>`, fractions of parts ≤ 99 | 32 × 32 | 683 µs | FLINT 318 µs |

### Arbitrary-precision floats

| Page | Contents |
| --- | --- |
| [Arbitrary-precision floats](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/float.md) | `a::math::float::BigFloat`, a binary float of any precision from 1 bit to 2⁴⁰, each result correctly rounded in four modes with MPFR's ternary value and the exponent range `±(2⁶² − 1)`: `+`, `-`, `*`, `/`, the fused multiply-add and the square root in registers for one and two limbs and on windows of the limbs the rounding needs beyond, with Karatsuba, Toom-Cook, transforms, Newton's division and Zimmermann's square root at length; results into existing numbers (`set_add`, `set_mul_add`, `set_sqrt`, `set_f64`, `set_floor`, `set_rint`, `add_assign_round`) without allocating; primitive numbers and `Number`s on either side of the operators; exact sums and dot products rounded once; `1/√x`, roots of any degree, integer and rational powers with their exact and halfway cases found first; conversions to and from `f64` and `f32` with the subnormals, to integers (`floor`, `ceil`, `rint_round`, saturating `to_i64_round`) and exact comparisons with integers, floats, numbers and rationals; `Display` as the shortest decimal that reads back, the digits of `mpfr_get_str` in any radix from 2 to 36, and reading any number of digits with `mpfr_strtofr`'s syntax; `exp`, `exp2`, `exp10`, `exp_m1` by Smith's rectangular splitting and the bit burst, `ln`, `log2`, `log10`, `ln_1p` from tables to 512 bits, Newton's iteration on the exponential beyond and the AGM from 3,400 bits, `pow` with C99's special values, and the hyperbolic functions and their inverses; `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `sin_cos`, `atan`, `asin`, `acos` and `atan2`, the argument reduced with π to the bits its exponent needs, a table and Horner's rule in arrays to 512 bits, the series after halvings beyond, and Brent's bit burst from about 770,000 bits for the sine and 205,000 for `atan`; π, `e`, `ln 2`, `ln 10`, Euler's `γ` and Catalan's `G` from tables up to 16,384 bits and beyond by binary splitting of the Chudnovskys', Zuniga's and the Pilehroods' series and the Brent–McMillan algorithm in dual numbers, kept per process; `BigComplex`, as GNU MPC, each part of every result rounded in a mode of its own, with its arithmetic (also into existing numbers, `set_add` and the others), square root, norm and absolute value and `exp`, `ln`, `log10`, the circular and hyperbolic functions and their inverses, `pow`, `arg` and `proj` with MPC's branch cuts and special values; and `Matrix<BigFloat>` and `Matrix<BigComplex>`, products whose every element is the exact dot product rounded once and LU solutions of exact dot products |

Measured on an Apple M4 Max on one thread, in ns, µs, ms or s a call at the precision given in bits (both parts' for complex numbers), against the fastest of MPFR 4.2 (through FFI, into a variable that exists), rug 1.30, Arb in FLINT 3.6 (whose balls are not correctly rounded: for the time only) and GNU MPC 1.4.1 that has the function; the page has the conditions, more sizes and the other operations. MPFR and MPC write into a variable that exists and `a` returns a new number, unless the operation is a `set_*` form.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `set_add` | 53 bits | 5 ns | MPFR 5.1 ns |
|  | 1000 bits | 17.7 ns | MPFR 14.6 ns |
|  | 10⁵ bits | 483 ns | rug 988 ns |
| `set_add`, operands of 10⁵ bits | 1000 bits | 55.1 ns | rug 271 ns |
| `set_mul` | 53 bits | 3.5 ns | MPFR 5.4 ns |
|  | 10⁶ bits | 559 µs | rug 1.41 ms |
| `set_div` | 10⁶ bits | 1.36 ms | rug 4 ms |
| `set_sqrt` | 64 bits | 12.7 ns | MPFR 6.8 ns |
|  | 10⁶ bits | 1.44 ms | rug 3.23 ms |
| `mul_add_round` | 113 bits | 21.9 ns | MPFR 22.8 ns |
|  | 10⁵ bits | 50.4 µs | rug 109 µs |
| `rec_sqrt_round` | 10⁴ bits | 8.49 µs | MPFR 9.33 µs |
| `nth_root_round(100)` | 1000 bits | 6.33 µs | MPFR 43.4 µs |
| `powi_round(7)` | 113 bits | 50.2 ns | MPFR 75.7 ns |
| `powi_round(1_000_000_007)` | 53 bits | 125 ns | rug 385 ns |
| `to_f64_round` | 53 bits | 1.6 ns | rug 5.4 ns |
| `set_f64` | 53 bits | 2.8 ns | MPFR 5.6 ns |
| `from_f64_round` | 53 bits | 9.9 ns | MPFR 5.6 ns |
| `set_floor` | 53 bits | 7.2 ns | MPFR 8.2 ns |
| `floor` | 53 bits | 16.9 ns | MPFR 8.2 ns |
| `to_digits(10, None)` | 53 bits | 46.2 ns | MPFR 219 ns |
|  | 10⁶ bits | 4.84 ms | MPFR 9.56 ms |
| `to_string()` | 53 bits | 133 ns | rug 310 ns |
| `from_str_radix`, 53 bits | 17 digits | 38.1 ns | MPFR 107 ns |
|  | 10⁶ digits | 40.1 µs | MPFR 486 µs |
| `exp` | 53 bits | 334 ns | Arb 84.4 ns |
|  | 10⁶ bits | 94.9 ms | Arb 75.3 ms |
| `ln` | 53 bits | 186 ns | Arb 103 ns |
|  | 10⁴ bits | 168 µs | Arb 78.5 µs |
|  | 10⁶ bits | 70.5 ms | Arb 87 ms |
| `pow` | 128 bits | 887 ns | Arb 344 ns |
| `sinh` | 10⁵ bits | 5.63 ms | Arb 3.68 ms |
| `sin` | 53 bits | 213 ns | Arb 86.6 ns |
|  | 10⁵ bits | 6.96 ms | Arb 4.87 ms |
| `cos`, an argument `m·2^(10⁶)` | 53 bits | 1.38 ms | MPFR 4.05 ms |
| `atan` | 256 bits | 1.65 µs | Arb 232 ns |
|  | 10⁶ bits | 241 ms | Arb 154 ms |
| `atan2` | 1000 bits | 8.4 µs | Arb 2.18 µs |
| `BigFloat::pi`, first call | 10⁶ bits | 19.7 ms | Arb 28.1 ms |
|  | 10⁸ bits | 4.44 s | Arb 6.48 s |
| `BigFloat::ln2`, first call | 10⁶ bits | 39 ms | Arb 55.4 ms |
| `BigFloat::euler_gamma`, first call | 10⁶ bits | 511 ms | Arb 950 ms |
| `BigFloat::catalan`, first call | 10⁶ bits | 113 ms | Arb 180 ms |
| `BigFloat::pi`, cached | 64 bits | 14 ns | Arb 8 ns |
| `BigComplex::set_add` | 53 bits | 10.4 ns | rug 12.9 ns |
| `BigComplex::add_round`, a new number | 53 bits | 28.4 ns | rug 12.9 ns |
| `BigComplex::set_mul` | 53 bits | 26.6 ns | MPC 53.9 ns |
| `BigComplex::mul_round` | 10⁵ bits | 160 µs | MPC 322 µs |
| `BigComplex::sqrt_round` | 53 bits | 105 ns | rug 156 ns |
| `BigComplex::div_round` | 10⁵ bits | 535 µs | rug 820 µs |
| `BigComplex::exp_round` | 1000 bits | 8.35 µs | MPC 18.5 µs |
| `BigComplex::ln_round` | 256 bits | 3.16 µs | MPC 8 µs |
| `BigComplex::tan_round` | 53 bits | 1.99 µs | MPC 4.22 µs |
| `BigComplex::asin_round` | 1000 bits | 19.9 µs | MPC 158 µs |
| `BigComplex::pow_round` | 10⁴ bits | 929 µs | MPC 2.73 ms |
| `BigComplex::powi_round(17)` | 113 bits | 830 ns | MPC 810 ns |
| `&a * &b`, `Matrix<BigFloat>`, 8 × 8 | 256 bits | 13.2 µs | an `mpfr_fma` loop 18.1 µs |
| `a.solve(&b)`, `Matrix<BigFloat>`, 8 × 8 | 53 bits | 3.71 µs | Gaussian elimination by `mpfr_fms` 3.11 µs |

### Polynomials

| Page | Contents |
| --- | --- |
| [Polynomials](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/poly.md) | `a::math::poly::Polynomial<T>`, dense, with coefficients that are `Number`s, `Rational`s, `f64`s or `Complex<f64>`s, and `ModPolynomial`, with residues modulo any `n ≥ 2` that it holds once, in one word below 2⁶⁴: `+`, `-`, `*`, `sqr` and `pow` by the schoolbook method with each sum of products reduced once (NEON's multiply-adds for small moduli, machine integers for small integer coefficients), the number-theoretic transforms on the coefficients or on their pieces, Kronecker substitution, and the FFT for long float products; power series inverses, division by Newton's method, pseudo-division and exact division on limbs and by Kronecker substitution, `mul_mod`, `pow_mod` and Brent and Kung's `compose_mod` with the inverse of the modulus kept, and `compose` by divide and conquer; greatest common divisors by Euclid's algorithm in Montgomery's form and the half gcd, over the integers the heuristic gcd and the multimodular algorithm, `xgcd` (multimodular over the rationals), `lcm`, resultants and discriminants (multimodular over the integers) and the subresultants by Lazard's and Ducos's sequence; evaluation at many points by Horner's rule at many points at once and the scaled remainder tree, products of linear factors, and interpolation on the subproduct tree or by Lagrange's and Newton's forms; complex roots in double precision by the Aberth–Ehrlich iteration in exact conjugate pairs, certified complex roots of integer and rational polynomials to any precision, real roots isolated by Descartes's rule and continued fractions, counted, and correctly rounded at any precision; and square-free decompositions, factorisations and tests of irreducibility modulo any prime (Kaltofen and Shoup's baby steps and giant steps, Cantor and Zassenhaus's splitting, Berlekamp's method, Ben-Or's test, the roots) and over the integers and the rationals (Zassenhaus's method with Hensel's lifting) |

Measured on an Apple M4 Max on one thread, in ns, µs, ms or s a call, against the fastest of FLINT 3.6, NTL 11.6, PARI/GP 2.19 and, for the roots, Arb in FLINT that has the function; the page has the conditions, every size and all the losses. Modulo a word the products take 0.54 to 1.0 of FLINT's time from 128 coefficients for 50- to 64-bit moduli, and `x.pow_mod(&e, &f)` (the Frobenius of a factorisation) 0.2 to 0.32 from 256; evaluation at many points, products of linear factors and interpolation take 0.3 to 0.7 of FLINT's time from 1024 points; the factorisations modulo word primes 0.34 to 0.75 of FLINT's from degree 30 (but modulo 3 at degree 1000, 1.5 times), and over the integers 0.1 to 0.9 on random polynomials; the certified complex roots 0.15 to 0.92 of Arb's time, and the isolation of real roots 0.03 to 0.79 of FLINT's from degree 50. FLINT is faster at products of `Polynomial<Rational>` (2.5 to 9.4 times: every coefficient is kept in lowest terms, where `fmpq_poly` keeps one denominator), over the integers for coefficients of 32 and 64 bits (1.1 to 5.5: each `Number` coefficient is a vector of its own, where `fmpz` keeps a small one in a word), modulo primes below `2³²` from 256 coefficients (1.04 to 1.6: two 29-bit transform primes and lengths of powers of two against one 50-bit prime and truncated transforms), and below about 8 coefficients (its timing loops reuse the result's storage); NTL's `ZZ_pX` is faster modulo primes of 4 to 16 limbs from 128 to 256 coefficients (1.1 to 1.8: its 60-bit primes need no pieces of the residues).

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `ModPolynomial` `*`, n = 2⁶³ − 25 | 10⁵ by 10⁵ | 5.74 ms | FLINT 7.04 ms |
| | 10⁶ by 10⁶ | 54.5 ms | FLINT 77.1 ms |
| `ModPolynomial` `*`, n = 65,521 | 10⁴ by 10⁴ | 208 µs | FLINT 153 µs |
| `div_rem`, n = 2⁶³ − 25 | 2·10⁴ by 10⁴ | 2.09 ms | FLINT 1.94 ms |
| `inv_series`, n = 2⁶³ − 25 | 10⁴ | 978 µs | FLINT 1.14 ms |
| `x.pow_mod(&e, &f)`, n = e = 2⁶³ − 25 | deg f = 1,000 | 3.79 ms | FLINT 14.8 ms |
| `compose_mod`, n = 2⁶³ − 25 | 1,000 | 5.23 ms | FLINT 7.67 ms |
| `ModPolynomial` `*`, n = 2¹²⁷ − 1 | 1,000 by 1,000 | 128 µs | NTL 159 µs |
| `Polynomial<Number>` `*`, 64-bit coefficients | 10⁴ by 10⁴ | 1.15 ms | FLINT 885 µs |
| `Polynomial<Number>` `*`, 1,000-bit coefficients | 10⁴ by 10⁴ | 17.0 ms | NTL 24.4 ms |
| `pseudo_div_rem`, 64-bit coefficients | 256 by 128 | 1.28 ms | NTL 1.44 ms |
| `div_exact`, 64-bit coefficients | 4,000 by 2,000 | 680 µs | NTL 810 µs |
| `gcd`, n = 2⁵⁰ − 27, a common factor | degree 10⁴ | 11.0 ms | NTL 12.8 ms |
| `gcd` over the integers, 64 bits, a common factor | degree 1,000 | 887 µs | NTL 1.02 ms |
| `resultant` over the integers, 64 bits | degree 100 | 1.51 ms | PARI 1.89 ms |
| `eval_many_u64`, n = 2⁵⁰ − 27 | 4,096 at 4,096 points | 1.47 ms | FLINT 3.20 ms |
| `interpolate_u64`, n = 2⁵⁰ − 27 | 4,096 points | 2.36 ms | FLINT 4.92 ms |
| `Polynomial<f64>::roots` | degree 100 | 314 µs | FLINT `cd_poly_find_roots` 509 µs |
| `complex_roots(53)`, 16-bit coefficients | degree 100 | 4.18 ms | Arb 14.9 ms |
| `real_root_intervals`, 16-bit coefficients | degree 1,000 | 27.5 ms | FLINT 65.9 ms |
| `factor`, n = 2⁶⁴ − 59 | degree 1,000 | 82.8 ms | FLINT 191 ms |
| `factor` over the integers, 64-bit coefficients | degree 100 | 507 µs | FLINT 1.14 ms |
| `factor`, Swinnerton-Dyer polynomial | degree 32 | 539 µs | FLINT 781 µs |

### Fourier transforms

| Page | Contents |
| --- | --- |
| [Fourier transforms](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/fft.md) | `a::math::fft`: `Fft<T>`, the complex transform of `f32` and `f64` values of any length, planned once, by Stockham passes of the radices 2, 4, 8, 3, 5, 7 and the primes to 19 (NEON, SSE2 or AVX2 with FMA chosen at run time, portable elsewhere), a larger prime factor by its own plan, and prime lengths by Rader's or Bluestein's algorithm with a smooth inner length; `RealFft<T>`, the transform of reals into the `n/2 + 1` values of the spectrum, by the half-length complex transform for even lengths and a split by the smallest factor for odd ones; `Dct<T>` and `Dst<T>`, the eight cosine and sine transforms of FFTW's `REDFT` and `RODFT` kinds by FFTW's accurate algorithms; `Fft2<T>` and `Fft3<T>` by tiles of columns; every inverse unscaled, every method on batches, and `_with_scratch` forms that never allocate; and `convolve`, `correlate`, `circular_convolve` and `circular_correlate` of real or complex sequences, by SIMD direct sums, padded transforms or overlap-add, chosen by a measured model of their costs, with the plans kept in a bounded cache shared by the threads |

Measured on an Apple M4 Max on one thread, in ns, µs or ms a call, the median of three runs, against the fastest of FFTW 3.3.11 (Homebrew's build, which on arm64 has only its scalar codelets), rustfft 6.4.1 and realfft 3.5.1, and Apple's Accelerate vDSP that has the transform; the page has the conditions, every size and all the losses. The complex transform takes 0.52 to 0.70 of FFTW's time and 0.59 to 0.80 of rustfft's for powers of two from 64 in `f64` (0.55 to 1.04 of vDSP's), and 0.30 to 0.45 and 0.59 to 0.82 in `f32`; other smooth lengths from 25 take 0.24 to 0.89 of FFTW's time and 0.19 to 0.83 of rustfft's, and primes from 101 and twice primes 0.19 to 0.76 and 0.23 to 0.81. The real transforms of powers of two from 512 take 0.71 to 0.97 of FFTW's time in `f64` and 0.40 to 0.60 in `f32`, those of 10³ to 10⁶ values (`10^k`, 6,000 and 30,000) and of the primes 101 to 100,003 0.28 to 0.97 of the fastest reference's; the cosine and sine transforms of types II to IV from 1000 values (but 1023) 0.57 to 1.03 of FFTW's time in `f64` and 0.36 to 0.66 in `f32`; `Fft2` and `Fft3` from 128 values a side 0.43 to 0.99 of FFTW's (but 1024 × 1024 in `f64`, 1.19 times); and the convolutions against an operand of 10⁴ values or more 0.18 to 1.07 of the fastest reference's time. vDSP is faster in `f32`, 1.2 to 3.7 times, at the complex transform from 128 values, the real one from 64, the cosine transforms up to 16,384 and the two-dimensional one (it works on split complex data, `a` on interleaved values); FFTW and rustfft are faster at single complex transforms of up to 32 values and of 6 to 24, and rustfft at batches of 16 and 32 (1.1 to 3.7 times: about 5 ns of dispatch a call and no composite codelets); FFTW or realfft at real transforms of up to 64 values and their batches (1.2 to 7.1), at odd real lengths (1.1 to 6.2: real codelets for odd radices), at DCT-I and DST-I of `2^k ± 1` (1.1 to 1.5), at cosine and sine transforms of 255 and 1023 and short ones in `f64` (1.1 to 2.3), and at some two- and three-dimensional sizes (1.1 to 2.1; the tiles are copied value by value); vDSP at short convolutions (operands of up to 100 values, 1.1 to 1.8 times, and 1,000 by 64 in `f32`, 1.12); and FFTW and rustfft in three cells near the switch of the convolutions' paths (1.34 to 1.35). The x86_64 paths were tested under emulation only, never timed.

| Operation | Size | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `Fft::<f64>` forward | 1,024 | 1.33 µs | vDSP 1.41 µs |
| | 2²⁰ | 4.22 ms | rustfft 5.51 ms |
| | 10⁶ | 3.38 ms | FFTW 7.64 ms |
| `Fft::<f64>` forward, prime | 100,003 | 1.60 ms | FFTW 2.24 ms |
| `Fft::<f64>` forward | 16 | 19.0 ns | FFTW 12.0 ns |
| `Fft::<f32>` forward | 1,024 | 705 ns | vDSP 306 ns |
| `RealFft::<f64>` forward | 1,024 | 954 ns | vDSP 899 ns |
| | 10⁶ | 1.87 ms | FFTW 2.85 ms |
| | 2,187 | 5.00 µs | FFTW 3.53 µs |
| `RealFft::<f32>` forward | 65,536 | 50.9 µs | vDSP 30.1 µs |
| `Dct::<f64>`, DCT-II | 1,024 | 1.30 µs | FFTW 1.45 µs |
| `Dct::<f32>`, DCT-II | 1,024 | 729 ns | vDSP 412 ns |
| `Fft2::<f64>` forward | 1000 × 1000 | 3.80 ms | FFTW 5.47 ms |
| `Fft3::<f64>` forward | 128 × 128 × 128 | 10.5 ms | FFTW 13.4 ms |
| `convolve`, `f64` | 10⁵ by 3 | 18.2 µs | vDSP 42.1 µs |
| | 10⁵ by 10⁵ | 1.15 ms | FFTW 1.46 ms |
| `convolve`, `Complex<f32>` | 10⁴ by 16 | 12.3 µs | rustfft 25.0 µs |

### Special functions

| Page | Contents |
| --- | --- |
| [Special functions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/special.md) | `a::math::special`, of `f32`, `f64` and complex arguments, every function with a stated ulp bound tested against mpmath at 256 bits: `gamma`, `lgamma` (with `lgamma_sign`), `digamma`, `polygamma`, `beta`, `lbeta`, `beta_inc` with its complement and their inverses, `gamma_p`, `gamma_q` and their inverses, `factorial`, `binomial`, `pochhammer`; `erf`, `erfc`, `erfcx`, `erf_inv`, `erfc_inv`, `faddeeva`, `fresnel`, `dawson`; `bessel_j`, `bessel_y`, `bessel_i`, `bessel_k` of integer and real order (with `bessel_i_scaled` and `bessel_k_scaled`) and the spherical `sph_bessel_j`, `sph_bessel_y`, `sph_bessel_i`, `sph_bessel_k`, and `airy_ai`, `airy_bi` with their derivatives; `zeta`, `hurwitz_zeta`, `dilog`, `polylog`, `lambert_w0`, `lambert_wm1`, `expint_e1`, `expint_ei`, `expint_en`; `ellip_k`, `ellip_e`, `ellip_pi`, `ellip_f`, `ellip_e_inc`, `ellip_pi_inc`, Carlson's `carlson_rc`, `carlson_rf`, `carlson_rd`, `carlson_rj`, `jacobi_elliptic` (the `sn`, `cn` and `dn` at once), and the orthogonal polynomials `legendre_p`, `assoc_legendre_p`, `chebyshev_t`, `chebyshev_u`, `gegenbauer_c`, `jacobi_p`, `laguerre_l`, `assoc_laguerre_l`, `hermite_h`, `hermite_he` |

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of three quiet rounds, against libm 0.2, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`), Cephes and the crates statrs 0.18, special 0.11 and puruspe 0.4; the page has the conditions, every domain and all the losses, each beside the accuracy of the reference that is fastest there (several of them are 10³ ulps or more off in that cell). `lgamma` takes 0.52 to 0.71 of the fastest reference's time for arguments 0.5 to 8, `gamma_p` and `gamma_q` near the mean 0.33 from `a` of 200 to 10⁶ and `beta_inc_inv` 0.36 at `a` and `b` of 10⁵ to 10⁷; `erf` and `erfc` 0.49 to 0.66 of the fastest's, `dawson` 0.18 to 0.61 and `fresnel` 0.45 to 0.56; `zeta` 0.31 to 0.62, `polylog` of order 3 0.12 of Cephes's and `E₃` 0.15 of statrs's; `ellip_k` and `ellip_e` 0.35 to 0.53 of Boost's below `m` = 0.8; the Airy functions 0.24 to 0.56, `bessel_i` at large arguments and small orders 0.22 of puruspe's, and `chebyshev_t` of degree 1000 0.20 of Boost's. These timings predate the fixes of the recheck of 2026-11-21 and 22; where those slowed a function quoted here the page says by how much: `lgamma` 33 to 37 % on [3, 8) and 12 to 31 % on [0.5, 1.5], `zeta` on [−0.5, 0.9] from 4.5 to 5.1 ns, `polylog` of order 3 on [−1, 1] from 149 to 162 ns and `lambert_wm1` on [−0.36, −0.01] from 10.8 to 15.9 ns (the last three timed without LTO on a loaded machine), the Airy functions within 10 %. The references are faster at: puruspe's `bessel_j` and `bessel_y` of order 0 and 1 (1.7 to 1.8 times), and the general-order Bessel and Neumann functions of moderate order and argument (1.4 to 4.9: GSL, Cephes and puruspe — the last 10¹³ ulps off at the general-order `J`); Boost's `digamma` and `polygamma` at negative arguments (1.9 to 2.1); `binomial` of large arguments (2.9, Boost 386 ulps off) and `gamma_q_inv` for tiny `q` (3.2); GSL's `fermi_dirac_int` at `polylog` past `x` = −2 (84, GSL 10² ulps off there); puruspe's and special's Lambert W (2 to 3); statrs's `Eₙ` for large `n` (about 3, at 40 ulps); Boost's `carlson_rd` and `carlson_rj` (1.8 to 1.9) and `ellip_pi` at `n` > 1 (19 — Boost's principal value is 5·10¹¹ ulps off there); GSL's `jacobi_elliptic` (1.8, against its 10⁶ ulps); and Boost's closed forms for the orthogonal polynomials up to degree 20 (4 to 6 — `a` wins from degree 100 on). The x86_64 paths were tested under emulation only, never timed.

| Function | Domain | `a` | Fastest reference |
| --- | --- | ---: | --- |
| `lgamma` | [3, 8] | 3.31 ns | puruspe 6.33 ns |
| `gamma_p`, `gamma_q` | 200 < `a` < 10⁴, `x` ~ `a` | 47.8 ns | special 145.4 ns |
| `beta_inc_inv` | `a`, `b` in [10⁵, 10⁷], `x` within 3σ of the mean | 776 ns | Boost 2175 ns |
| `erfc` | [3, 8] | 4.66 ns | statrs 9.23 ns |
| `dawson` | [0.5, 8] | 1.78 ns | cephes 10.01 ns |
| `fresnel` | [8, 10⁴] | 11.11 ns | cephes 24.55 ns |
| `zeta` | [−0.5, 0.9] | 4.26 ns | cephes 13.58 ns |
| `polylog` | `s` = 3, `x` in [−1, 1] | 148 ns | cephes 1239 ns |
| `expint_en` | `n` = 3, [0.1, 2] | 21.6 ns | statrs 148 ns |
| `ellip_e` | `m` in [0, 0.8) | 2.40 ns | Boost 6.86 ns |
| `airy_bi` | [−10, 10] | 11.29 ns | puruspe 46.58 ns |
| `bessel_i` | ν in [0, 3], x in [30, 10³] | 28.3 ns | puruspe 128.5 ns |
| `chebyshev_t` | degree 1000, [−1, 1] | 143 ns | Boost 699 ns |

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

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/field.md) verified, without timings yet:

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

The curves over binary fields and the hashing to curves have no timings yet; what the page verified them against:

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

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/pairing.md) verified, without timings yet:

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

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/quaternion.md) verified, without timings yet:

| Part | Against | Verified |
| --- | --- | --- |
| The algebra | mpmath at 60 digits, 160-bit `BigFloat`, `Complex` | `exp`, `ln`, `sqrt` and powers on 24 hard inputs, the largest error seen 1.00 ε times the condition; identities on 2,000 random triples; 4,000 products with parts up to `2¹⁰⁰⁰`, 2,555 of them overflowing, against 160-bit ones |
| Rotations | SciPy 1.18, glam 0.33, nalgebra 0.35 | matrices, rotation vectors, `apply`, `Slerp` and `align_vectors` within `5.3·10⁻¹⁶` of SciPy's on random rotations; 20,000 random cases as each crate's |
| Euler angles | SciPy 1.18, glam 0.33, nalgebra 0.35 | all 24 conventions within `4.4·10⁻¹⁶` of SciPy's, and its choice at gimbal lock |

### Dual numbers

[`a::math::dual`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md) differentiates functions by forward-mode automatic differentiation: `Dual<T>` is `re + eps·ε` with `ε² = 0`, and `HyperDual<T>` carries a second derivative too. Both are generic over the trait `DualNum` and nest, so that a function written once over `D: DualNum` gives its value for `f64` and its derivatives for the dual types, exact up to the rounding of the arithmetic but for [one rule at `0·∞`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#limits).

| Operations | Functions | Method | Passes |
| --- | --- | --- | --- |
| Arithmetic | `+`, `-`, `*`, `/`, with a float on the right (and on the left for the concrete types), comparisons of the values | the rules of the algebra | |
| [Functions](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#the-derivatives-at-the-edges) | powers and roots, exponentials and logarithms, the circular and hyperbolic functions and their inverses, `atan2`, `hypot` | the value from the standard library, the derivatives formed so that they keep their digits | |
| [Derivatives](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#forward-mode-and-what-it-costs) | `derivative`, `second_derivative` | a dual pass; a hyper-dual pass (Fike and Alonso) | 1 |
| Gradients, Jacobians | `gradient`, `jacobian` | a dual pass a variable | `n` |
| Hessians | `hessian`, `value_gradient_and_hessian` | a hyper-dual pass an entry | `n(n + 1)/2` |

What the [page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md) verified, without timings yet:

| Part | Against | Verified |
| --- | --- | --- |
| The functions | SymPy 1.14 and mpmath 1.3 at 600 digits | 514 points of 54 functions in `f64` and `f32`, through every form: 8,733 components within their bounds, 4 ulps for values and 8, 16 and 32 for the first three derivatives (eight times that for high powers and compositions) |
| num-dual | num-dual 0.16 | 417 points of 45 functions: within 2 ulps in the first derivative and 7 in the second, but where num-dual's own error is the larger |
| [With the solvers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#with-the-numerical-methods) | `a::math::numerics` | Newton's and Halley's methods, BFGS, L-BFGS-B, Levenberg–Marquardt, Newton's method for systems and the stiff methods, with exact derivatives, gradients and Jacobians |

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
| [Normal forms](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#normal-forms), on `Number` | `hermite_normal_form`, `hermite_form`, `smith_normal_form`, `elementary_divisors` | rank profiles modulo a prime, proved; the Hermite form modulo the determinant `D` of a minor (Domich, Kannan and Trotter); the elementary divisors modulo it; the Smith transformations by alternating Hermite forms (Kannan and Bachem) | n³ operations on numbers below `D` |
| [Over GF(p)](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#matrices-over-gfp), `ModMatrix` | `+`, `-`, `*`, `ple`, `rank`, `det`, `inverse`, `solve`, `rref`, `null_space` | products reduced once, Strassen–Winograd from 128; the recursive PLE decomposition | n³/3 products of words, O(n^2.81) from 128 |
| [Over GF(2)](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md#matrices-over-gf2), `BitMatrix` | `+`, `*`, `mul_strassen`, `transpose`, `rank`, `det`, `inverse`, `solve`, `null_space`, `reduced_row_echelon_form` | the method of the Four Russians (M4RM), Strassen–Winograd from 3072; Bard's elimination (M4RI) | n³/(64·log n) word additions, O(n^2.81) from 3072 |

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

What the pages verified, without timings yet:

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

### Sparse matrices

[`a::math::sparse`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/sparse.md) holds matrices of `f32`, `f64`, `Complex<f32>` or `Complex<f64>` by their nonzeros, and solves with them in time and memory that follow the nonzeros and the fill of the factors. Permutations are `p[new] = old`; a factorisation that does not exist is `None`, and a matrix singular to working precision is reported, never solved.

| Operations | Functions | Method | Cost |
| --- | --- | --- | --- |
| Storage | `SparseMatrix` (`CsrMatrix`), `CscMatrix`, `CooMatrix`, the conversions, `read_matrix_market`, `write_matrix_market` | canonical compressed rows and columns; triplets summed by radix sorts | nnz + n |
| Operations | `mul_vec`, `mul_vec_transpose`, `mul_vec_adjoint`, `*`, `+`, `-`, `transpose`, `submatrix`, `select`, `permute`, `norm`, `LinearOperator` | Gustavson's row-by-row product | the multiplications |
| Orderings | `reverse_cuthill_mckee`, `amd`, `colamd`, `nested_dissection`, `elimination_tree`, `postorder` | George and Liu; approximate minimum degree; its column form; separators by level sets and multilevel bisection | about nnz in practice, the dissection ten to twenty times `amd`'s time |
| Cholesky | `SparseCholesky`, `SparseLdl` (`L·D·Lᴴ`), `solve`, `log_det`, `inertia` | supernodal, left-looking, the blocks by the dense `gemm` and `trsm` | the arithmetic of the factor |
| LU | `SparseLu`, `solve`, `solve_transpose`, `is_singular`, `rcond` | Gilbert and Peierls, partial pivoting, a condition estimate | the arithmetic of the factors |
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

### Numerical methods

[`a::math::numerics`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/numerics.md) solves the standard problems of numerical analysis on `f32` and `f64` in generic code, over the sealed trait `Real`; its linear algebra is `a::math::matrix`. A method never reports an answer it has not verified: one that cannot reach its tolerance returns `None`, or its best point or estimate with `converged` false.

| Page | Contents |
| --- | --- |
| [Numerical methods](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/numerics.md) | `a::math::numerics`, with `_with` forms that take options: the zeros of a function by `bisect`, `brent`, `secant`, `newton` and `halley`, and of a system by `newton_system`, `broyden` and `broyden_with_jacobian`; integrals by `quad`, `quad_g10k21`, `quad_g7k15`, `quad_half_infinite`, `quad_infinite`, `clenshaw_curtis`, `tanh_sinh`, `cubature`, `monte_carlo` and `quasi_monte_carlo` on `Sobol`, and the Gaussian rules `legendre`, `hermite`, `hermite_scaled`, `laguerre`, `laguerre_scaled` and `jacobi`; `y′ = f(t, y)` by `solve`, `integrate`, `solve_at` and `solve_ivp`, with `DenseOutput` and `Event`, by `Rk4`, `DormandPrince5`, `DormandPrince853` and the stiff `RadauIia` and `Bdf`; minima by `golden_section`, `minimize`, `nelder_mead`, `conjugate_gradient`, `bfgs` and, within bounds, `lbfgsb`, sums of squares by `gauss_newton` and `levenberg_marquardt`, and linear programmes by `LinearProgram`; `Spline` (cubic, clamped, not-a-knot, Akima, PCHIP), `chebyshev`, `chebyshev_nodes`, `barycentric_weights`, `barycentric_interpolate`, `aaa` and `aaa_complex`; and `derivative` and `second_derivative` by Richardson extrapolation |

Exact derivatives for these methods, gradients, Jacobians and Hessians, come from the [dual numbers](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/dual.md#with-the-numerical-methods) of `a::math::dual`, with no step to choose.

The page measures the methods against SciPy 1.18 and the codes behind it, and against mpmath. `brent` takes exactly the steps of `brentq` on 57 of 63 brackets; `quad` converges on all 46 integrable integrals of a battery of 48, one more than `scipy.integrate.quad`, none of them beyond its own error estimate; `DormandPrince5` takes the steps of DOPRI5 in every run, `RadauIia` 0.92 to 1.01 times those of SciPy's `Radau` and `Bdf` 0.94 to 1.00 times those of its `BDF` on five stiff problems; `lbfgsb` ends on SciPy's active set on 42 of 43 problems; and the simplex method gives HiGHS's status on 600 random programmes and its optima to 5.0·10⁻¹³.

## Hash

[`a::hash`](https://github.com/nettrash/a/blob/HEAD/lib/docs/hash/sha2.md) computes the hash functions SHA-224, SHA-256, SHA-384 and SHA-512 of FIPS 180-4 and RFC 9380's `expand_message_xmd`, which hashing to curves builds on. The compression functions are portable Rust, and not constant-time.

| Page | Contents |
| --- | --- |
| [SHA-2](https://github.com/nettrash/a/blob/HEAD/lib/docs/hash/sha2.md) | `Sha224`, `Sha256`, `Sha384` and `Sha512`, streaming hashers (`new`, `update`, `finalize`; `Clone` forks a hash midway); `sha224`, `sha256`, `sha384` and `sha512` in one call; the trait `Digest` over them; `expand_message_xmd::<H>` on any `Digest`, with the hashing of tags longer than 255 bytes |

What the page verified, without timings yet:

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

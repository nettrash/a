# A

[![rust workflow](https://github.com/nettrash/a/actions/workflows/rust.yml/badge.svg)](https://github.com/nettrash/a/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/nettrash-a.svg)](https://crates.io/crates/nettrash-a)
[![docs.rs](https://img.shields.io/docsrs/nettrash-a)](https://docs.rs/nettrash-a)
[![license](https://img.shields.io/crates/l/nettrash-a.svg)](https://github.com/nettrash/a/blob/HEAD/LICENSE)

Algorithms in pure Rust, with no dependencies:

- **Sorting** (`a::sort`): 23 sorting algorithms by comparison and by key, and a default sort.
- **Integers of any size** (`a::math::numbers`): `Number`, with its arithmetic up to transforms for the longest products, roots, powers modulo a number, greatest common divisors, primes and text in any radix; prime sieves and counting, factorisation up to the quadratic sieve, arithmetic functions, modular arithmetic and discrete logarithms, and integer sequences.
- **Elliptic curves** (`a::math::ec`): Weierstrass, Montgomery and Edwards curves over prime fields, secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519 among them.
- **Complex numbers** (`a::math::complex`) and **matrices** (`a::math::matrix`): decompositions, eigenvalues, singular values and functions of matrices of floats and complex numbers, and exact algorithms on matrices of integers.

Every algorithm has a page with its method, examples and measurements against the reference implementations (GMP, PARI, FLINT, primesieve, GMP-ECM, OpenSSL, faer, nalgebra and others); the tables below link to them. The API reference is on [docs.rs](https://docs.rs/nettrash-a). The crate needs Rust 1.88 (edition 2024). Its fastest paths are NEON and assembly kernels on aarch64, and assembly for the products of integers on x86_64; everything else is portable Rust.

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

### Elliptic curves

| Page | Contents |
| --- | --- |
| [Elliptic curves](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/ec.md) | `a::math::ec`: short Weierstrass, Montgomery and twisted Edwards curves over prime fields of any length; `+`, `-`, `*`, compression, `multi_mul`, prepared points and maps between the forms; secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519 ready made |

Measured on an Apple M4 Max in cycles, against OpenSSL 3.6 and curve25519-dalek 4.1; `k·P` multiplies a point, `k·G` the generator, by a table where the library has one. OpenSSL's and dalek's products of a point are constant-time, which costs them time this crate doesn't spend:

| Curve | Against | `k·P` | theirs | `k·G` | theirs |
| --- | --- | ---: | ---: | ---: | ---: |
| secp256k1 | OpenSSL | 40,100 | 555,000 | 7,250 | 559,000 |
| P-256 | OpenSSL | 82,600 | 90,900 | 10,600 | 14,300 |
| P-384 | OpenSSL | 311,000 | 548,000 | 40,200 | 267,000 |
| P-521 | OpenSSL | 358,000 | 496,000 | 52,500 | 269,000 |
| Ed25519 | dalek | 45,300 | 72,000 | 5,350 | 22,100 |
| Curve25519, X25519 | dalek | 55,400 | 78,500 | | |

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

### Matrices

[`a::math::matrix::Matrix<T>`](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/matrix.md) is a dense matrix of any size, column after column, of `f32`, `f64`, `Complex<f32>`, `Complex<f64>` or `Number`. The algorithms are LAPACK's, arranged so that most of their work is products of blocks, and the products run on NEON kernels; real matrices stay in real arithmetic. Costs are for `n × n` matrices.

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
| Exactly, on `Number` | `det`, `rank`, `characteristic_polynomial`, `adjugate`, `inverse`, `null_space` | Bareiss's elimination and Berkowitz's method, without fractions | n³ to 3n⁴/2 products of numbers |

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

## Not constant-time

Nothing here is written to take the same time whatever its operands: `pow_mod` and the elliptic curves in particular branch on the values of their exponents and scalars. Use them on public data, for verification and research, not on secret keys or scalars.

## Changelog

What each version added, changed and fixed is in [CHANGELOG.md](https://github.com/nettrash/a/blob/HEAD/CHANGELOG.md).

## License

[MIT](https://github.com/nettrash/a/blob/HEAD/LICENSE).

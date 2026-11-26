# Changelog

All notable changes to [`nettrash-a`](https://crates.io/crates/nettrash-a) are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.46](https://crates.io/crates/nettrash-a/0.2.46) - 2026-11-26

The second release: number theory, fixed-width integers, rational numbers, floats of any precision, polynomials, Fourier transforms, special functions, finite fields, pairings, quaternions, dual numbers, lattices, sparse matrices, numerical methods and SHA-2, from issues #15 to #25 and #27, and a speed pass over all of them. As in 0.1.29, each has a page of its own with its method, examples, what it was verified against and its measurements.

### Added

- `a::math::numbers`: number theory on `Number` and `u64`.
  - Primes: `primes(range)`, the primes of a range of `u64`s by a segmented sieve after Walisch's primesieve; `prime_pi`, by Deléglise and Rivat's method with Gourdon's sorting of the special leaves, and `nth_prime`; and `Number::prev_prime`.
  - Factorisation: `Number::factor` into a `Factorization` of primes and exponents, the method chosen by size: trial division, Hart's one line method, Pollard's rho, `p − 1`, the elliptic curve method on Montgomery curves and the self-initialising quadratic sieve with one and two large primes and block Lanczos. These and Williams's `p + 1` are also on their own in `a::math::numbers::factor` (`trial`, `rho`, `pm1`, `pp1`, `ecm`, `siqs`), their second stages by baby steps and giant steps or by fast polynomial arithmetic; and `perfect_power`.
  - Arithmetic functions of a number or of its factorisation: `divisors`, `divisor_count`, `divisor_sigma`, `totient`, `carmichael`, `mobius`, `is_squarefree` and `radical`.
  - Squares modulo a number: `kronecker`, `sqrt_mod_all`, and `cornacchia` for the primitive solutions of `x² + d·y² = m`.
  - Modular arithmetic: `crt` for any moduli, `multiplicative_order`, `primitive_root`, and `discrete_log` by Pohlig and Hellman's reduction, baby steps and giant steps, Pollard's rho and, modulo primes up to 112 bits, the index calculus; the methods on their own in `a::math::numbers::dlog`.
  - Sequences: `fibonacci` and `lucas` by fast doubling, `lucas_sequence` and `lucas_sequence_mod` for an index of any size, `primorial`, and `partitions` by the Hardy–Ramanujan–Rademacher series.
- `a::math::uint`: `Uint<N>` and `Int<N>`, unsigned and signed integers of `N` 64-bit limbs on the stack, `Copy`, which behave as a `u{64·N}` and an `i{64·N}` would.
  - The operators, and the arithmetic's `checked_*`, `wrapping_*`, `overflowing_*` and `saturating_*` forms, with the primitive integers' behaviour on overflow; `widening_mul`, `carrying_mul`, `div_rem`, `div_euclid`, `pow`, `isqrt`, `ilog`, `ilog2` and `ilog10`.
  - Text in any radix from 2 to 36, with the formatter's flags and a `const` `from_str_radix`; conversions to and from the primitive integers, floats, bytes and `Number`.
  - `Montgomery<N>`, the arithmetic modulo an odd number in Montgomery's form, and `pow_mod`, `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` for any nonzero modulus, odd or even.
- `a::math::rational`: `Rational`, a fraction of numbers in lowest terms with a positive denominator, held in two words, without allocating, when both parts fit in 64 bits.
  - `+`, `-`, `*` and `/` with `Rational`s, `Number`s and primitive integers on either side; comparisons, `floor`, `ceil`, `round` and their `_into` forms, `pow`, `Sum` and `Product`.
  - Exact conversions from `f64` and `f32` and correctly rounded ones to them; text in any radix from 2 to 36, and decimals with exponents.
  - Continued fractions and their convergents, `limit_denominator`, `simplest_between`, and rational reconstruction (`reconstruct`).
  - The Bernoulli numbers and the harmonic numbers (`bernoulli`, `bernoulli_numbers`, `harmonic`).
- `a::math::float`: `BigFloat`, a binary float of any precision from 1 bit to 2⁴⁰, and `BigComplex`, of two such parts, each result correctly rounded in the four modes of `Round` with MPFR's ternary value, as MPFR and GNU MPC round it.
  - `+`, `-`, `*`, `/`, the fused multiply-add and the square root, also into numbers that exist, without allocating (`set_add` and the rest); exact sums and dot products rounded once; roots of any degree, and integer and rational powers.
  - Conversions to and from `f64`, `f32`, integers, `Number`s and `Rational`s, and exact comparisons with them; text in any radix from 2 to 36, and `Display` as the shortest decimal that reads back.
  - The exponential, logarithmic, circular and hyperbolic functions and their inverses, and the constants π, `e`, `ln 2`, `ln 10`, `γ` and Catalan's `G`.
  - `BigComplex`'s arithmetic and elementary functions, as GNU MPC's.
- `a::math::poly`: `Polynomial<T>`, with coefficients that are `Number`s, `Rational`s, `f64`s or `Complex<f64>`s, and `ModPolynomial`, modulo any integer `n ≥ 2`.
  - Products by the schoolbook method, Kronecker substitution and number-theoretic transforms, and long float products by the FFT; division by Newton's method, pseudo-division, `mul_mod`, `pow_mod`, `compose` and `compose_mod`.
  - Greatest common divisors by Euclid's algorithm and the half gcd, over the integers by the heuristic gcd and the multimodular algorithm; `xgcd`, `lcm`, `resultant`, `discriminant` and `subresultants`.
  - Evaluation at many points (`eval_many`) and interpolation.
  - Complex roots in double precision by the Aberth–Ehrlich iteration, certified complex roots to any precision, and real roots isolated, counted and correctly rounded.
  - Square-free decompositions, factorisations and tests of irreducibility modulo primes, and factorisations over the integers and the rationals by Zassenhaus's method with Hensel's lifting.
- `a::math::fft`: Fourier transforms of `f32` and `f64` values, planned once and shared between threads.
  - `Fft<T>`, the complex transform of any length, by Stockham passes on NEON, SSE2 or AVX2 as the CPU has them, and by Rader's or Bluestein's algorithm for prime lengths; `RealFft<T>`, the transform of reals.
  - `Dct<T>` and `Dst<T>`, the cosine and sine transforms of types I to IV, FFTW's `REDFT` and `RODFT`; `Fft2<T>` and `Fft3<T>`, in two and three dimensions.
  - Every inverse unscaled, every method on batches, and `_with_scratch` forms that never allocate.
  - `convolve`, `correlate`, `circular_convolve` and `circular_correlate` of real or complex sequences, by direct sums, padded transforms or overlap-add, chosen by a model of their costs.
- `a::math::special`: special functions of `f32`, `f64` and, for some, complex arguments, as free functions, each with a stated bound in ulps, tested against mpmath at 256 bits.
  - The gamma family: `gamma`, `lgamma`, `digamma`, `polygamma`, `beta`, `pochhammer`, and the incomplete gamma and beta functions with their inverses.
  - The error functions `erf`, `erfc` and `erfcx`, with the inverses of the first two, `faddeeva`, `fresnel` and `dawson`.
  - The Bessel functions `J`, `Y`, `I` and `K` of integer and real order, the spherical ones, and the Airy functions with their derivatives.
  - `zeta`, `hurwitz_zeta`, `dilog`, `polylog`, `lambert_w0`, `lambert_wm1` and the exponential integrals.
  - Complete and incomplete elliptic integrals, Carlson's forms, `jacobi_elliptic`, and the classical orthogonal polynomials.
- `a::math::field`: finite fields, each built once, checked and given the form of arithmetic that suits it; elements borrow their field and combine with `+`, `-`, `*` and `/`.
  - `PrimeField` and `Fp`: prime fields of any size, with square roots, the Legendre symbol, inversion one at a time and in batches, and encodings; the fields of BN254, BLS12-381 and the named curves ready made. `TypedPrime` and `TypedFp`: `Copy` elements of primes known at compile time, in `field::named` or declared by `typed_prime!`.
  - `ExtensionField` and `Fpk`: `GF(pᵏ)` by a given or the smallest irreducible polynomial, with the Frobenius map, norms, traces, square roots, orders and primitive elements.
  - `Tower`, `Fp2`, `Fp6` and `Fp12`: the towers `F_p² → F_p⁶ → F_p¹²` of the pairings, BN254's and BLS12-381's.
  - `BinaryField` and `F2m`: `GF(2ᵐ)` by the carry-less products `PMULL` and `PCLMULQDQ` or a portable kernel, with traces, half-traces and quadratic equations; Gaussian normal bases (`NormalBasis`, `NormalElement`, `BasisConversion`); and `Gf2_8` and `Gf2_128`, the fields of AES and GCM, with the AES S-box and GHASH.
- `a::math::ec::binary`: curves `y² + x·y = x³ + a·x² + b` over `GF(2ᵐ)` (`BinaryCurve`, `BinaryPoint`, `BinaryDomain`, `BinaryPrepared`) in López–Dahab coordinates, with the τ-adic non-adjacent form on Koblitz curves, the ladder on `x` alone (`x_mul`), `multi_mul` and SEC 1's encodings; the 18 curves of SEC 2, NIST's K- and B- curves among them, in `binary::named`.
- `a::math::ec::hash`: hashing to curves as RFC 9380 specifies, its twelve suites for P-256, P-384, P-521, secp256k1, curve25519 and edwards25519 ready made, and `hash_to_field` and the maps `ShallueVanDeWoestijne`, `SimplifiedSwu`, `Elligator2` and `IsogenyMap` for other curves.
- `a::math::pairing`: the optimal ate pairing `e: G1 × G2 → GT` on BN254 and BLS12-381 (`PairingCurve`), its values those of arkworks.
  - `pairing`, `multi_pairing`, `miller_loop` and `final_exponentiation`.
  - The groups `G1`, `G2` and `Gt`: their arithmetic, membership by endomorphisms, the clearing of cofactors, ZCash's encodings of BLS12-381 and Ethereum's of BN254, and Ethereum's precompiles in `pairing::ethereum`.
  - Hashing to `G1` and `G2` in `pairing::hash`: RFC 9380's four suites for BLS12-381, and gnark-crypto's for BN254's `G1`.
  - No protocol layer: no signatures, keys or random numbers.
- `a::math::matrix`: `ModMatrix` and `BitMatrix`, dense matrices over `GF(p)` for a prime `p < 2⁶⁴`, a word an element, and over `GF(2)`, a bit an element.
  - `ModMatrix`: products by Strassen–Winograd's recursion from order 128, the PLE decomposition (`ple`), `rank`, `det`, `inverse`, `solve`, `rref` and `null_space`.
  - `BitMatrix`: products by the method of the Four Russians and `mul_strassen`, `transpose`, `rank`, `det`, `inverse`, `solve`, `null_space`, and `reduced_row_echelon_form` by Bard's M4RI.
- `a::math::quaternion`: `Quaternion<f32>` and `Quaternion<f64>`.
  - Hamilton's product and the rest of the algebra, whose products that overflow give infinities rather than NaNs; `exp`, `ln`, `sqrt` and powers.
  - Rotations, each quaternion read as `q/|q|`: axes and angles, rotation vectors and matrices, `rotate`, `slerp`, `nlerp`, `angle_to` and `from_rotation_arc`; Euler angles in SciPy's 24 conventions (`EulerSequence`, `EulerFrame`).
- `a::math::dual`: forward-mode automatic differentiation by `Dual<T>` and `HyperDual<T>`, generic over the trait `DualNum` and nesting, with the elementary functions; `derivative`, `second_derivative`, `gradient`, `jacobian` and `hessian`, in the shapes the methods of `a::math::numerics` take.
- `a::math::lattice`: integer lattices, their bases the rows of a `Matrix<Number>`, which may be dependent.
  - `lll`, exactly; `lll_fp`, by Nguyen and Stehlé's L² algorithm, its result proved; `deep_lll`, with Schnorr and Euchner's deep insertions; `bkz`; and the exact test `is_lll_reduced`. A reduction returns a `Reduced`: the basis, a zero row for each dependency, and the unimodular `U` with `U·B = B'`.
  - `shortest_vector`, `closest_vector` and `short_vectors` by enumeration, exactly.
  - Integer relations among real numbers by PSLQ (`pslq`, `pslq_f64`), with a proved bound below which there is none.
- `a::math::sparse`: sparse matrices of `f32`, `f64`, `Complex<f32>` and `Complex<f64>`.
  - `SparseMatrix` (`CsrMatrix`), `CscMatrix` and `CooMatrix`: their conversions, products by vectors and matrices, sums, permutations and norms; `LinearOperator`; and Matrix Market files (`read_matrix_market`, `write_matrix_market`).
  - Orderings that keep the fill low: `reverse_cuthill_mckee`, `amd`, `colamd` and `nested_dissection`.
  - Direct methods: the supernodal `SparseCholesky` and `SparseLdl` (`L·D·Lᴴ`), `SparseLu`, and the multifrontal `SparseQr` for least squares.
  - Krylov methods, each residual verified afresh: `cg`, `minres`, `gmres` and `bicgstab`, with the preconditioners `Jacobi`, `Ilu0` and `Ic0`.
  - A few eigenvalues by the implicitly restarted Lanczos and Arnoldi methods (`lanczos`, `arnoldi`), and by shift-and-invert near a target.
- `a::math::numerics`: the standard problems of numerical analysis on `f32` and `f64`, generic over the sealed trait `Real`, its linear algebra on `a::math::matrix`. A method returns `None`, or its best point or estimate with `converged` false, rather than an answer it has not verified; options are given through a `_with` form taking an options struct, or, for the quadrature rules other than `quad`, as an argument.
  - The zeros of a function of one variable by bisection, Brent, Alefeld, Potra and Shi's Algorithm 748 (`toms748`), the secant method, Newton and Halley, and of a system by Newton with a Jacobian and by Broyden's rank-one update, both with a line search.
  - Definite integrals by QUADPACK's adaptive Gauss–Kronrod quadrature (G7–K15 and G10–K21) with Wynn's extrapolation, by Clenshaw–Curtis and by tanh-sinh quadrature, over infinite intervals too; in several dimensions by Genz and Malik's cubature, by Monte Carlo and by quasi-Monte Carlo on a randomly shifted Sobol' sequence; and the nodes and weights of the Gauss–Legendre, Hermite, Laguerre and Jacobi rules, by Golub and Welsch's method and, for many points, by Newton's method from asymptotic first guesses.
  - The initial value problem `y′ = f(t, y)` by the classical fourth-order Runge–Kutta method, by Dormand and Prince's 5(4) pair and their 8(5,3) method, by the L-stable three-stage Radau IIA method and by the variable-order backward differentiation formulas, Shampine and Reichelt's NDFs by default, each with its own step-size control, with dense output and the location of events.
  - Minima by the golden section and by Brent's method in one variable, by Nelder and Mead's simplex, the Polak–Ribière conjugate gradient and BFGS in several, the last two on Moré and Thuente's strong-Wolfe line search, and within bounds by L-BFGS-B; sums of squares by Gauss–Newton and by Moré's Levenberg–Marquardt, both on the QR decomposition; and linear programmes by the revised simplex method on bounded variables, primal in two phases, and dual from 20 rows by default, its basis checked and finished by the primal method.
  - Interpolation by the cubic, clamped, not-a-knot, Akima and PCHIP splines, by Chebyshev interpolation evaluated with Clenshaw's recurrence, and by barycentric interpolation; rational approximation by the AAA algorithm, with its poles, residues and zeros; differentiation by Richardson extrapolation on symmetric differences.
- `a::hash`: SHA-224, SHA-256, SHA-384 and SHA-512 of FIPS 180-4, as streaming hashers (`Sha224`, `Sha256`, `Sha384`, `Sha512`) and in one call (`sha224`, `sha256`, `sha384`, `sha512`), over the trait `Digest`; and RFC 9380's `expand_message_xmd`. The compression functions take the ARMv8 SHA-256 and SHA-512 instructions and x86_64's SHA extensions where the CPU has them.
- [`bench/`](https://github.com/nettrash/a/blob/HEAD/bench/README.md): the harnesses of the speed pass, outside the published crate: each operation of v0.2 against the libraries that do the same work, the scoreboards they write, and a workflow that runs them on an x86_64 runner by hand.
- `--cfg a_baseline`: built with `RUSTFLAGS="--cfg a_baseline"`, every run-time check of a CPU feature reports it absent, so that the tests run the portable fallbacks of the SIMD and instruction-set paths; a new CI job runs the tests so.

### Changed

- `a::math::numbers::Number`:
  - `pow_mod`: on aarch64, kernels in inline assembly for moduli of two to sixteen limbs, generated by a script in the tree (`pow/montgomery/kernels.py`); moduli of one to three limbs take the exponent's bits from the low end; REDC by products from 86 limbs (100 before), its high half from Mulders' short product below the transforms' threshold; and the low halves of squares for even moduli. On an Apple M4 Max, moduli of 16 to 64 bytes take 0.34 to 0.52 of GMP 6.3's time with an exponent as long as the modulus.
  - `is_probable_prime`, and so `next_prime`: on the kernels of `pow_mod`, which made the test of a prime of 512 bits twice as fast on aarch64; and below 2³², the strong test to base 2 and a table of the composites that pass it.
  - `jacobi`: the binary method up to six limbs (in assembly on aarch64), Lehmer's steps up to 23 limbs, and the reductions of the greatest common divisor beyond, subquadratic from 800 limbs; 0.1.29 took the binary method at every length.
  - `sqrt_mod` takes any positive modulus and gives the smallest root, or `None`; 0.1.29 took an odd prime only, for which the result is the same, and panicked for an even modulus. It panics for a zero or negative modulus only.
- `a::math::matrix::Matrix`:
  - Its elements may also be `Rational`, `BigFloat` and `BigComplex`: exact determinants, ranks, solutions, inverses and null spaces of `Matrix<Rational>`, by Bareiss's elimination and Dixon's p-adic lifting, and `/` by a rational; correctly rounded products of `Matrix<BigFloat>` and `Matrix<BigComplex>`, and their determinants, solutions and inverses at the working precision.
  - `Matrix<Number>`: `det` by residues modulo primes or by Dixon's divisor for the larger matrices; `solve` and `inverse_rational`, exact, as a `Matrix<Rational>`, a singular matrix proved singular; products by Strassen–Winograd's recursion; and the Hermite and Smith normal forms with their unimodular transformations (`hermite_normal_form`, `hermite_form`, `smith_normal_form`, `elementary_divisors`).
  - `Lu::solve_into`, a solution into a matrix that exists, without allocating for small right-hand sides, and `Lu::refactor`, the decomposition of another matrix in the memory of this one.
- `a::math::ec`: the fields of the curves over prime fields serve `a::math::field` and the pairings too, and on aarch64 their sums and differences of four, six and nine limbs are in inline assembly.
- The published package leaves out the tests' reference data (`lib/tests/**/data`), which stays in the repository: with it the package would come close to crates.io's size limit.
- The speed pass: each operation that v0.2 added was measured against the fastest libraries that do the same work, optimised, audited independently for correctness, and had its constants tuned on a quiet machine; the harnesses are in `bench/`. The pages' Measured sections give the final comparisons, from one run of every area on 2026-11-26 on an Apple M4 Max, and every case where a reference is faster, with its cause.
- What the speed pass changed in the interfaces the v0.2 issues had merged:
  - `LpOptions` has a new public field, `dual_threshold`, and `SparseLuOptions` another, `strategy`, of the new enum `LuStrategy`: struct literals of these options must add them, or end in `..LpOptions::new()` and `..SparseLuOptions::new()`.
  - The sealed trait `a::math::numerics::Real` has the supertrait `a::math::fft::FftFloat`.
  - The `Debug` output of `DenseStep` and of `Montgomery` shows their new private fields, and `PrimeField::form()`, and so its `Debug`, says `"Montgomery, more than 9 limbs"` for a prime of more than 576 bits, where it said `"Number"`.
  - `GOLUB_WELSCH_LARGEST`, the largest order of a Gaussian rule by Golub and Welsch's method, is 60 (it was 100): the Legendre, Hermite and Laguerre rules of higher orders are computed by Newton's method from asymptotic first guesses.
  - `RealFft::<f64>` of length 64 needs 32 values of working memory (`scratch_len()`), where it needed none.
  - `lll`, `bkz` and `deep_lll` may return another valid reduced basis than before for the same input: `lll` starts from the basis of `lll_fp`.
  - `SparseLu::new` chooses its strategy by the pattern of the matrix (`LuStrategy::Auto`): for a nearly symmetric one, AMD of `A + Aᵀ` with diagonal pivots, watched, and factored again by COLAMD and partial pivoting when its fill or growth passes its bounds; otherwise COLAMD and partial pivoting.

### Fixed

- `isqrt`, `isqrt_rem` and `next_prime` could write past their working memory at some lengths (on aarch64, for instance, the square roots of numbers of 1,517 limbs, and the search from a number of 437 limbs): the memory was planned for one length and used at another, and the plan of a shorter operand can be the larger.
- `pow_mod` with an exponent of more than 2³² bits gave a wrong result or did not end.
- On 32-bit targets, where sizes counted in `usize` wrapped: the working memory of the cofactors of the greatest common divisor, of `pow_mod` and of the primality test could be planned too short and overrun, where it should have failed to allocate; `binomial` was wrong for `2³² < n < 2³³`, and the sieve of the factorials and binomials could hang or lose primes near `2³³`; `pow_mod` did not end for a base divisible by `2^(2³²)`; sums, bit operations, text and factorials panicked at lengths they can handle; and the elliptic curves gave wrong products for scalars of `2³²` bits or more.
- `Point::prepare(usize::MAX)` built a table whose products were the identity; a table too long to allocate now panics.
- On x32 (`x86_64-unknown-linux-gnux32`) the x86_64 assembly of the sums, products and quotients, written for 64-bit pointers, was compiled in; x32 takes the portable code now.

### Security

- Nothing in the crate is constant-time: `pow_mod` of `Number` and of `Uint<N>`, with the arithmetic of `Montgomery<N>`, the finite fields, the elliptic curves over prime and binary fields, the hashing to curves and the pairings branch on the values of their exponents, scalars and elements; the products, quotients and inverses of the fixed-width integers take a time that depends on their values, and the hash functions on the lengths of their messages. Use them on public data, not on secret keys or scalars.

## [0.1.29](https://crates.io/crates/nettrash-a/0.1.29) - 2026-11-04

The first release. Algorithms in pure Rust with no dependencies, each with a page of its own that gives its method, examples and measurements against the reference implementations.

### Added

- `a::sort`: 23 sorting algorithms and a default sort, each taking a mutable slice.
  - By comparison, for any `T: Ord`: bubble, selection, insertion, cocktail shaker, gnome, comb, merge, quick, heap, Tim, intro, Shell, Drift, tree, cycle, bitonic, odd-even and stooge sorts. Bitonic and odd-even sorts also come in parallel forms.
  - By integer or float keys, with `_by_key` forms: counting, radix, bucket and pigeonhole sorts, and sleep sort as a joke.
  - `sort`, the default: Drift sort, chosen over the other general-purpose sorts by measurement.
- `a::math::numbers::Number`: an integer of any size, stored as its sign and the bytes of its magnitude, in either byte order.
  - `+`, `-`, `*`, `/` and `%` with their assignments, with owned or borrowed operands and with primitive integers.
  - Products by the schoolbook method, Karatsuba's, Toom-3, Toom-4 and number-theoretic transforms.
  - Quotients by Knuth's method with Möller and Granlund's reciprocals, by divide and conquer and by Newton's method; `div_rem`, Euclid's division, a prepared `Divisor`, and exact division.
  - Shifts, bitwise operations in two's complement, and bit queries.
  - Text in any radix from 2 to 36.
  - Roots, factorials and binomials, greatest common divisors and modular inverses, powers and modular powers.
  - Primality by Baillie–PSW, the next prime, the Jacobi symbol, and square roots modulo a prime.
  - Comparison, hashing, conversions from and to the primitive integers, to floats and from strings, and `Sum` and `Product`.
  - Assembly kernels on aarch64 and x86_64, and NEON transforms on aarch64.
- `a::math::ec`: elliptic curves over prime fields of any length, on `Number`.
  - Short Weierstrass, Montgomery and twisted Edwards forms.
  - Points that add with `+`, subtract with `-` and multiply by a number with `*`.
  - Compression, `multi_mul` by Straus's and Pippenger's methods, and prepared points for repeated products.
  - Invariants and maps between the forms.
  - secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519 ready made, with field kernels of their own.
  - `k·P` and `k·G` are faster than OpenSSL 3.6 on its curves, and than curve25519-dalek 4.1 on Ed25519 and X25519.
- `a::math::complex`: `Complex<f32>` and `Complex<f64>`.
  - Their arithmetic, with a float of the same type on either side.
  - `abs`, `arg`, `exp`, `ln`, `sqrt` and powers, as principal values with C99's branch cuts.
  - Division and square roots that neither overflow nor underflow before their results do.
- `a::math::matrix`: `Matrix<T>` of `f32`, `f64`, `Complex<f32>`, `Complex<f64>` or `Number`.
  - Arithmetic: `+`, `-` and `*`, by an element, and Hadamard's product. Transposes and conjugate transposes, row and column operations, the trace, and powers.
  - LU, QR, Cholesky, singular value, Schur and eigendecompositions.
  - Determinants, inverses, solutions of linear systems and least squares, ranks, null spaces, characteristic polynomials, six norms and condition numbers.
  - `exp`, `sqrt` and `log`.
  - On `Number`: exact determinants, ranks, characteristic polynomials, adjugates, inverses and null spaces.
  - In `f64` on one core, at least as fast as faer and nalgebra at every size measured, from 8 × 8 to 1024 × 1024.

### Security

- Nothing in the crate is constant-time: `pow_mod` and the elliptic curves in particular branch on the values of their exponents and scalars. Use them on public data, not on secret keys.

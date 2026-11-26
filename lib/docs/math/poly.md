[A](../../../README.md) › [Math](../../../README.md#math)

# Polynomials

`Polynomial<T>` (in `a::math::poly`) is a polynomial in one variable with coefficients of the type `T`: a [`Number`](numbers.md), a [`Rational`](rational.md), an `f64` or a [`Complex<f64>`](complex.md). `ModPolynomial` is a polynomial with coefficients modulo an integer `n ≥ 2`, which it holds once for all its coefficients and shares with its copies. Both are dense: the coefficients, the constant first, without zeros at the top, so that the zero polynomial has none and every other its leading coefficient last.

- `Polynomial::new(coeffs)` takes the coefficients, the constant first, and drops the zeros at the top, `from_slice` copies them; `zero`, `one`, `x`, `constant(c)` and `monomial(c, n)` make the simplest ones, and `Default` is zero. `coeffs` gives the coefficients back and `into_coeffs` their vector, `coeff(i)` one of them (0 above the degree), `leading` the leading one; `degree` is `None` for the zero polynomial, and `length`, as FLINT's `fmpz_poly_length`, is the degree plus one. `is_zero`, `is_one`, `is_constant` and `is_monic` test them; `==` and `Hash` compare the coefficients.
- They add with `+` and `+=`, subtract with `-` and `-=` and multiply with `*` and `*=`, each operand owned or borrowed; `-p` negates, `scale(&c)` multiplies by a constant, `sqr` squares and `pow(n)` raises to a `u32` power, `p⁰ = 1`. Over a field (`Rational`, `f64`, `Complex<f64>`) `/` and `%` and their assigning forms give the quotient and the remainder, `div_rem` both, and `to_monic` divides by the leading coefficient; over the integers `pseudo_div_rem` gives `lᵏ·a = q·d + r` with `l` the leading coefficient of `d`, and `div_exact` the quotient when there is no remainder. The methods of the products and of the division are [below](#products-and-division).
- `eval(&x)` evaluates by Horner's rule (over the integers, from 32 coefficients, on limbs), `derivative` differentiates and `compose(&g)` gives `p(g(x))`; `shift_left(n)` multiplies by `xⁿ`, `shift_right(n)` divides by it, `truncate(n)` keeps the coefficients below `xⁿ` and `reverse(n)` is `xⁿ⁻¹·p(1/x)`; `map` makes a polynomial of other coefficients.
- `content` and `primitive_part` split a polynomial over the integers into the greatest common divisor of its coefficients and the rest, with a positive leading coefficient; `to_primitive` splits a polynomial over the rationals into a rational and a primitive polynomial over the integers, and `From` makes a polynomial over the rationals from one over the integers.
- `Display` writes the terms from the leading one down, as PARI/GP does: `3*x^2 - x + 1`, a rational coefficient as `1/2*x`, a complex one in parentheses.
- `ModPolynomial::new(coeffs, &n)` reduces each coefficient to its residue in `0..n`, `from_u64(coeffs, n)` takes words, `from_polynomial(&p, &n)` a polynomial over the integers, and `with_coeffs` other coefficients with the modulus of `self`; `zero(&n)`, `one(&n)` and `x(&n)` make the simplest ones. `modulus` gives `n`, `coeffs` the residues, `word_coeffs` them as words for `n < 2^64`, and `to_polynomial` and `to_symmetric` a polynomial over the integers, with the residues in `0..n` or in `-n/2..n/2`. A residue is held as the limbs of `n`, one word for `n < 2^64`. The same arithmetic as `Polynomial` (`pow` with a `u64` exponent), `eval`, `derivative`, `compose` and `scale`; and, for a prime `n` (or whenever the coefficients they divide by are units: the leading ones, and the constant one for `inv_series`), `div_rem`, `/`, `%`, `to_monic`, `mul_mod`, `pow_mod`, `compose_mod` and `inv_series`. Operations on two polynomials need the same modulus and panic otherwise. `Display` writes `x^2 + 5 (mod 7)`.
- The products and the division, the greatest common divisors and resultants, evaluation at many points and interpolation, the roots, and the factorisation are below, then the examples; the methods are after the examples, then how the module was tested and the measurements against FLINT, NTL and PARI.

## Products and division

- `*` and `sqr` multiply by the method the coefficients and the lengths call for ([below](#products-and-division-1)): the schoolbook method, Kronecker substitution into the integers' multiplication, or the number-theoretic transforms on the coefficients, and the fast Fourier transform for long float products; `pow(n)` is repeated squaring on them. Over the integers the product of two polynomials of 10,000 coefficients of 64 bits takes 1.19 ms, of 1,000 bits 17.3 ms; modulo a word, of 100,000 coefficients each, 1.98 ms to 5.72 ms, and modulo 2 302 µs ([below](#measured)).
- `inv_series(n)` is the inverse of a power series modulo `xⁿ`, for a polynomial over a field and a `ModPolynomial` whose constant coefficient is a unit (it panics otherwise): `g` with `self·g ≡ 1 (mod xⁿ)`, the zero polynomial for `n = 0`. Over the rationals it goes through the integers from 8 coefficients for a common denominator of up to 64 bits, and over one common denominator from 32 coefficients otherwise.
- `div_rem`, `/` and `%` divide with a remainder by the schoolbook method, and by Newton's method past a crossover; over the rationals from 16 coefficients in the quotient through the integers. Over the integers `pseudo_div_rem` and `div_exact` divide by the schoolbook method on limbs, and by Kronecker substitution into the division of integers for long quotients; `pseudo_div_rem` tries the schoolbook method with the fewest factors of the leading coefficient first there, so that an exact or nearly exact pseudo-division costs about what an exact division does. `div_exact` is `None` as soon as a quotient coefficient does not divide or the divisor is wider than any divisor of the dividend can be.
- `mul_mod(&b, &f)`, `pow_mod(&e, &f)` (the exponent a `Number`, not negative) and `compose_mod(&g, &f)` (Brent and Kung's method) compute products, powers and compositions modulo `f`, for a polynomial over a field and a `ModPolynomial`; over the rationals `pow_mod` works on the integers. `compose(&g)` composes by divide and conquer.

## Greatest common divisors and resultants

- `gcd(&b)` is the greatest common divisor: monic over a field and modulo `n`, over the integers the greatest common divisor of the contents times that of the primitive parts, with a positive leading coefficient. `xgcd(&b)` gives `(g, s, t)` with `g = s·a + t·b`, `s` and `t` of degrees below those of `b/g` and `a/g` (the cofactors of the remainder sequence, unique), over a field and modulo `n`. The zero polynomial is the gcd of two zeros, and `xgcd` of two zeros is `(0, 0, 0)`.
- `lcm(&b)` is the least common multiple: monic over a field and modulo `n`, over the integers `a·b / gcd(a, b)` with a positive leading coefficient; the zero polynomial when either is zero.
- `resultant(&b)` is the determinant of the Sylvester matrix, `lc(a)^deg b · Π b(α)` over the roots of `a`: a `Number` over the integers, a `Rational` over the rationals, an `f64` or a `Complex<f64>` over the floats, and a `Number` in `0..n` for a `ModPolynomial`. It is 0 when either is zero and 1 for two non-zero constants, `c^deg b` for a constant `c` against `b`, as PARI's `polresultant`.
- `discriminant()` is `(-1)^(d(d-1)/2) res(f, f') / lc(f)` for `f` of the degree `d ≥ 1`, `lc(f)^(2d-2) Π (αᵢ - αⱼ)²` over the pairs of roots; 0 for a constant and for the zero polynomial (as PARI's `poldisc`). Modulo `n`, `f'` is taken of the formal degree `d - 1`, so that the identity with the roots holds also when `n` divides `d`.
- `subresultants(&b)` (over the integers) gives the non-zero subresultants, the lowest index first, as Sage's `subresultants`: for `p` and `q` of the degrees `m ≥ n` (taken in that order), `Sⱼ` for `j < n` is the polynomial of the determinants of the rows `x^(n-j-1)p, …, p, x^(m-j-1)q, …, q` cut to their top `m + n - 2j - 1` columns and one more. `S₀` is the resultant, the last non-zero one the gcd up to a constant; those inside the gaps of the defective steps vanish and are left out. Empty when either is zero, `[qᵐ]` for a constant `q`.
- Modulo `n`, every division needs its divisor's leading coefficient to be a unit, as it is for a prime `n`: `gcd`, `xgcd`, `lcm`, `resultant` and `discriminant` panic with "the leading coefficient is not a unit modulo n" when one is not, and when the moduli differ.

## Evaluation and interpolation

- `p.eval_many(&xs)` gives the values of `p` at each point of `xs`, as `eval` gives them one at a time: for every coefficient type and for `ModPolynomial` (points and values as `Number`s, the values in `0..n`), with `eval_many_u64(&xs)` on words for a modulus below `2^64`. Any modulus `n ≥ 2` will do.
- `Polynomial::from_roots(&roots)` is the monic `Π (x - rᵢ)`, the constant 1 for no roots; `ModPolynomial::from_roots(&roots, &n)` and `from_roots_u64(&roots, n)` the same modulo `n`, any `n ≥ 2`.
- `Polynomial::interpolate(&xs, &ys)` over a field (`Rational`, `f64`, `Complex<f64>`) is the polynomial of degree below `xs.len()` that takes the value `ys[i]` at `xs[i]`, the zero polynomial for no points; `ModPolynomial::interpolate(&xs, &ys, &n)` and `interpolate_u64(&xs, &ys, n)` the same modulo a prime `n`, or any `n` whose differences of the points are units. `interpolate_lagrange` and `interpolate_newton` give the same polynomial by the textbook forms, in `O(m²)` operations for `m` points. All of them panic when `xs` and `ys` differ in length or a point is repeated (modulo `n`: when two points are the same modulo `n`, or for a composite `n` differ by a non-unit); a float point is repeated when two are equal. For integer points with rational values, the points are rationals too: `Polynomial<Rational>::interpolate`.

## Roots

- `roots()` of a `Polynomial<f64>` or a `Polynomial<Complex<f64>>` gives its complex roots in double precision, each as often as its multiplicity, ordered by their real and then their imaginary parts; the roots of a real polynomial that are not real come in pairs of exact conjugates, the one of the negative imaginary part first, and the real ones have an imaginary part of exactly zero. A constant has none; the zero polynomial and a coefficient that is not finite panic. Each root is a root of a polynomial whose coefficients differ from the given ones by a few units in their last place, and each root of the polynomial has one of them near it, as near as its conditioning allows (the largest backward error measured, `|p(z)|/Σ|aᵢ||z|ⁱ`, is below `2^−47` at degree 1000, and `7·10^−16` on products of random roots, ill-conditioned, to degree 400).
- `companion()` of a `Polynomial<f64>`, `Polynomial<Complex<f64>>` or `Polynomial<Rational>` is its companion matrix, ones under the diagonal and `−aᵢ/aₙ` in the last column, whose characteristic polynomial is `p/aₙ` and whose eigenvalues ([`Matrix::eigenvalues`](matrix.md)) are the roots; `0 × 0` for a constant, a panic for the zero polynomial. For a rational polynomial it is exact.
- `complex_roots(prec)` of a `Polynomial<Number>` or `Polynomial<Rational>` gives the distinct complex roots, certified: a `ComplexRoot` for each, its `value()` a [`BigComplex`](float.md#complex-numbers) of at least `prec` bits in each part, its `radius()` a `BigFloat` with the root within that distance of the value, and its `multiplicity()`. The discs are pairwise disjoint, so that each holds exactly its root, and each radius is at most `2^(1−prec)·|value|` (0 for the root 0); a value has more bits than `prec` where fewer would let two discs meet. A root proven real has an imaginary part of exactly zero, and a pair of roots proven conjugate has values that are exact conjugates. The real roots come first, increasing, then the others by their real parts, each with the positive imaginary part before its conjugate, as Arb's `arb_fmpz_poly_complex_roots` orders them.
- `real_root_intervals()` gives the real roots of a `Polynomial<Number>` or `Polynomial<Rational>` as isolating intervals with their multiplicities, increasing: `(a, a, m)` for a rational root `a` found exactly (every root of a factor of degree 1, and any found at a midpoint), `(a, b, m)` with `a < b` for a root that is the only root of the polynomial in the open interval `(a, b)`, whose ends are not roots; no two overlap. `real_root_intervals_descartes()` and `real_root_intervals_cf()` give them by each of the two methods below, each on its own; `real_root_intervals()` chooses by the input: continued fractions where the bound of the roots is large (Wilkinson's of the degrees 50 and 100: 104 µs and 758 µs, against 388 µs and 3.06 ms by Descartes's rule), else Descartes's rule (random polynomials of the degrees 200 and 1000: 349 µs and 31.7 ms, against 719 µs and 169 ms by continued fractions; Chebyshev's of the degrees 50 and 200: 275 µs and 17.2 ms, against 864 µs and 159 ms), its nodes deep in the bisection going on by continued fractions (Mignotte's close pairs of the degrees 20 and 100: 87.4 µs and 258 ms, against 233 µs and 8.98 s by Descartes's rule alone).
- `real_roots(prec)` gives the real roots, increasing, each rounded to nearest (ties to even) at `prec` bits, correctly, with its multiplicity. `count_real_roots()` is the number of distinct real roots, as PARI's `polsturm`. All of them panic on the zero polynomial, and `prec` outside `1..=2^40` panics as for `BigFloat`.

## Factorisation

`squarefree` splits a polynomial into square-free parts by multiplicity, and `factor` into irreducible factors with their multiplicities, over the integers, the rationals and the integers modulo a prime of any size, 2 and 3 included. The factors come sorted by degree, then by their coefficients from the leading one down; every method that needs a prime modulus checks it (by a probable-prime test above `2^64`) and panics with "the modulus is not prime" otherwise.

- `Polynomial<Number>::squarefree()` gives `(c, [(g, e), …])`: `c` the content with the sign of the leading coefficient, and primitive, pairwise coprime, square-free `g` of positive leading coefficients with distinct multiplicities `e`, `f = c·Π gᵉ`. `Polynomial<Rational>::squarefree()` gives the same for the primitive part, `c` a rational and the `g` over the integers. `ModPolynomial::squarefree()` gives the leading coefficient and monic parts. The zero polynomial gives `(0, [])`, a constant `c` gives `(c, [])`.
- `Polynomial<Number>::factor()` gives the content with its sign and the primitive irreducible factors of positive leading coefficients with their multiplicities; `Polynomial<Rational>::factor()` a rational and the same factors of its primitive part; `ModPolynomial::factor()` the leading coefficient and the monic irreducible factors. `ModPolynomial::factor_berlekamp()` gives the same by Berlekamp's method.
- `ModPolynomial::distinct_degree_factor()` takes a square-free polynomial (and panics on another) and gives, for each degree `d` of its irreducible factors, their monic product and `d`; `equal_degree_factor(d)` splits such a product into its factors (and panics on a polynomial that is not one, when a part does not split). The random splittings start from a fixed seed: the same input gives the same output every time.
- `is_irreducible()`: modulo a prime, whether the polynomial is irreducible; over the integers and the rationals, whether it is irreducible over the rationals, so that the content does not count (`2x + 2` is). Constants are not.
- `ModPolynomial::roots()` gives the roots modulo the prime with their multiplicities, in increasing order (and panics on the zero polynomial).

## Examples

```rust
use a::math::complex::Complex;
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;

let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
let q = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Rational::from(x)).collect());

// x² − 1 and x + 1 over the integers.
let a = p(&[-1, 0, 1]);
let b = p(&[1, 1]);
assert_eq!((&a * &b).to_string(), "x^3 + x^2 - x - 1");
assert_eq!(a.eval(&Number::from(3)), Number::from(8));
assert_eq!(a.derivative(), p(&[0, 2]));
assert_eq!(a.compose(&b), p(&[0, 2, 1])); // (x + 1)² − 1

// Division: exact, pseudo (4·(x² + 1) = (2x − 1)(2x + 1) + 5) and over the rationals.
assert_eq!(a.div_exact(&b), Some(p(&[-1, 1])));
assert_eq!(a.div_exact(&p(&[1, 2])), None);
assert_eq!(p(&[1, 0, 1]).pseudo_div_rem(&p(&[1, 2])), (p(&[-1, 2]), p(&[5])));
let ar = Polynomial::<Rational>::from(&a);
let (quot, rem) = ar.div_rem(&q(&[1, 2]));
assert_eq!(quot.to_string(), "1/2*x - 1/4");
assert_eq!(rem.to_string(), "-3/4");
assert_eq!(q(&[1, -1]).inv_series(4), q(&[1, 1, 1, 1])); // 1/(1 − x) modulo x⁴

// Modulo 7: x² + 1 is irreducible, so x^49 ≡ x modulo it.
let f = ModPolynomial::from_u64(&[1, 0, 1], 7);
let x = ModPolynomial::from_u64(&[0, 1], 7);
assert_eq!(x.pow_mod(&Number::from(49), &f), x);
let y = ModPolynomial::from_u64(&[1, 1], 7);
assert_eq!(y.mul_mod(&y, &f), ModPolynomial::from_u64(&[0, 2], 7)); // (x + 1)² ≡ 2x

// Greatest common divisors and resultants, with (x − 1)(x − 2).
let c = p(&[2, -3, 1]);
assert_eq!(a.gcd(&c), p(&[-1, 1]));
assert_eq!(a.lcm(&c), p(&[2, -1, -2, 1]));
assert_eq!(a.resultant(&p(&[-2, 1])), Number::from(3)); // (1 − 2)(−1 − 2)
assert_eq!(a.discriminant(), Number::from(4));
assert_eq!(a.subresultants(&c), vec![p(&[3, -3])]); // S₀ = 0 is left out; S₁ = −3(x − 1)
let br = q(&[-2, 1]);
let (g, s, t) = ar.xgcd(&br);
assert!(g.is_one());
assert_eq!(&(&s * &ar) + &(&t * &br), g);

// Evaluation and interpolation.
assert_eq!(a.eval_many(&[0, 1, 2, 3].map(Number::from)), [-1, 0, 3, 8].map(Number::from));
assert_eq!(Polynomial::from_roots(&[1, -1].map(Number::from)), a);
let xs = [0, 1, 2].map(Rational::from);
let ys = [1, 2, 5].map(Rational::from);
assert_eq!(Polynomial::interpolate(&xs, &ys), q(&[1, 0, 1]));
assert_eq!(f.eval_many_u64(&[0, 1, 2, 3]), [1, 2, 5, 3]);
assert_eq!(ModPolynomial::interpolate_u64(&[0, 1, 2], &[1, 2, 5], 7), f);
assert_eq!(ModPolynomial::from_roots_u64(&[3, 4], 7), ModPolynomial::from_u64(&[5, 0, 1], 7));

// Roots: x² + 1 in double precision, ±i, exact conjugates.
let r = Polynomial::new(vec![1.0, 0.0, 1.0]).roots();
assert_eq!(r[1], r[0].conj());
assert!((r[1] - Complex::I).abs() < 1e-15);

// (x − 2)²(x² + 2), certified at 100 bits: 2 twice, then ±i√2.
let e = p(&[8, -8, 6, -4, 1]);
let r = e.complex_roots(100);
assert_eq!((r.len(), r[0].multiplicity()), (3, 2));
assert!(r[0].value().im().is_zero());
let s2 = BigFloat::from(2).sqrt_round(100, Round::Nearest).0;
assert!((r[1].value().im() - &s2).abs() <= *r[1].radius());

// (x − 1)²(x² − 2): −√2, 1 twice, √2; √2 correctly rounded at 200 bits.
let h = p(&[-2, 4, -1, -2, 1]);
assert_eq!(h.real_root_intervals()[1], (Rational::from(1), Rational::from(1), 2));
assert_eq!(h.count_real_roots(), 3);
assert_eq!(h.real_roots(200)[2].0, BigFloat::from(2).sqrt_round(200, Round::Nearest).0);

// Factorisation: 2x⁴ − 2 = 2(x − 1)(x + 1)(x² + 1), and x³ + x² = x²(x + 1).
let (k, fs) = p(&[-2, 0, 0, 0, 2]).factor();
assert_eq!(k, Number::from(2));
assert_eq!(fs, vec![(p(&[-1, 1]), 1), (p(&[1, 1]), 1), (p(&[1, 0, 1]), 1)]);
assert_eq!(
    p(&[0, 0, 1, 1]).squarefree(),
    (Number::from(1), vec![(p(&[0, 1]), 2), (p(&[1, 1]), 1)])
);
// x⁴ + 1 is irreducible over the integers and reducible modulo every prime.
assert!(p(&[1, 0, 0, 0, 1]).is_irreducible());
assert!(!ModPolynomial::from_u64(&[1, 0, 0, 0, 1], 7).is_irreducible());
// Modulo 2: x⁵ + x⁴ + 1 = (x² + x + 1)(x³ + x + 1); modulo 7: x² + 5 = (x − 3)(x − 4).
let m = ModPolynomial::from_u64(&[1, 0, 0, 0, 1, 1], 2);
assert_eq!(m.distinct_degree_factor().len(), 2);
assert_eq!(m.factor().1.len(), 2);
assert_eq!(m.factor_berlekamp(), m.factor());
let n = ModPolynomial::from_u64(&[5, 0, 1], 7);
assert_eq!(n.roots(), vec![(Number::from(3), 1), (Number::from(4), 1)]);
```

## The method

### Products and division

- **Modulo a word** (FLINT's `nmod_poly`): the schoolbook method computes each coefficient of a product as one sum of products and reduces it once, in a word when the sums fit (residues below `2^32`, by NEON's widening multiply-adds, 16 coefficients of the product at a time, up to 128 coefficients), in two words (for moduli below about `2^63/√m`), or in three; for operands of up to 4 coefficients (8 in the longer) the sums lie on the stack, without the vector kernel's set-up. The reduction is Möller and Granlund's division by the normalized modulus with its precomputed reciprocal, or Barrett's for one word; products by a constant take Shoup's precomputed quotient. From 64 coefficients in the shorter operand (96 for a square; 128 in the longer for residues whose sums of products fit a word, on NEON) the product goes to the number-theoretic transforms of the integers' multiplication, applied to the coefficients themselves: each coefficient cut into one to three digits of at most 28 bits, the convolution computed modulo the fewest of the eight primes `c·2²¹ + 1` below `2²⁹` whose product exceeds twice the largest sum (`2·m·(n−1)²`: 2 primes for small moduli, 5 or 6 for 63- and 64-bit ones), on NEON lanes of four 32-bit residues, and each coefficient brought back modulo `n` by the Chinese remainder theorem as `Σ yᵢ·((P/qᵢ) mod n) + k·(−P mod n)`, a sum in 128 bits and one division (the constants taken times the modulus's normalizing shift); with two primes, `P < 2^58`, the coefficient itself is computed in a word and reduced by Barrett's constant of `n`. A product of more than `2²¹` coefficients cuts the longer operand into pieces when the shorter has at most `2¹⁸`, and goes by Kronecker substitution into the integers' multiplication (Toom's methods on top of its transforms) when both are longer, as the pieces would then take a number of transforms growing as the product of the lengths. Karatsuba's method was measured and does not pay for words: the sums of products cost 0.06 ns a product below `2³²` and 0.25 ns above, which its additions and subtractions modulo `n` outweigh before the transforms take over. On targets without NEON the transforms run on portable lanes, four times slower, and Kronecker substitution into the integers' multiplication takes the lengths up to 8,192.
- **Modulo 2** (NTL's `GF2X`): the coefficients packed 64 to a word, coefficient `i` at bit `i mod 64` of word `⌊i/64⌋`. A `ModPolynomial` modulo 2 holds its coefficients one a word, and the packed form beside them once an operation has made it (a product, quotient or remainder modulo 2 is made packed, and its words are written out the first time something reads them; changing the words drops the packed form). Products by carry-less multiplications of words (`PMULL` on aarch64, `PCLMULQDQ` on x86-64, both detected at run time; a window method on four bits of one operand elsewhere and under Miri), Karatsuba's method from 28 words in both operands (its buffers carved from one scratch area a product), the longer operand cut into pieces of the shorter's length and the last, shorter piece multiplied the same way (pieces of its length, and so on, as the lengths in Euclid's algorithm); for long operands the product of words above, one word a coefficient, by the transforms (on a `ModPolynomial`'s own words, without packing them), whose `O(n log n)` beats Karatsuba's `O(n^1.585)` there: from 4,096 words in the shorter operand when `la^0.585·lb ≥ 12.25·(la + lb)·log₂(64(la + lb))` (`la ≤ lb` in words), a model of the two costs measured on this machine, which takes operands of equal lengths from 48,814 words (3,124,033 coefficients), `lb = 1.5la` from 35,356 words, `lb = 2la` from 29,537, `lb = 4la` from 22,162, and never a shorter operand below 18,751 words; squares by spreading the bits, as a square over GF(2) is linear. Remainders and quotients 64 quotient coefficients at a time by Barrett's method on words (the divisor shifted to a degree that is a multiple of 64, the quotient word from the top word by the low word of `⌊x¹²⁸/F⌋` for the divisor's top 65 bits, exact over GF(2)); for a quotient of 4,096 coefficients or more and a divisor of degree 14,336 or more kept for many remainders (`pow_mod`, a product modulo a fixed `f`, the factorisation) or of degree 65,536 or more for one division, by Barrett's method on polynomials: `μ = ⌊x^(2d)/f⌋` (`d = deg f`) by Newton's iteration on the reversed `f`, which over GF(2) is `g ← f·g² mod x^(2k)` (one square and one product a step), made once for a divisor kept and, for one division, only to the precision its quotient needs; each window of the top `2d` coefficients of the dividend reduced by two packed products, `q = ⌊⌊a/x^d⌋·μ/x^d⌋` exactly. `pow_mod` runs packed from start to end: squares and products (for the base `x` squares and shifts), each followed by such a remainder. Greatest common divisors by Euclid's algorithm on the packed words (shifted subtractions of the shorter from the longer) while the shorter operand's degree is below 6,144, and from degree 6,144 (6,145 coefficients) by the half gcd on the packed products, its base Euclid's steps on the packed words with their matrix below degree 768.
- **Modulo several limbs** (FLINT's `fmpz_mod_poly`): every residue of a sum of products is the remainder of its limbs by the normalized `n` with the reciprocal of its top two limbs (Möller and Granlund's division of three limbs by two), computed once for the modulus: one quotient limb a step, the remainder in two registers for `n` of two limbs, a product of the quotient limb by the other limbs subtracted beyond (the crate's division of limbs from 5 limbs, whose products run in assembly), for any `n`, even or odd, without allocating. The schoolbook method takes up to 6 coefficients (4 from 3 limbs), its sums of products kept in `2k + 1` limbs on the stack and divided once. From 48 coefficients in the shorter operand, for 2 to 9 limbs and products of up to about 6,000 coefficients, the transforms on pieces of the coefficients (NTL's way, with the crate's primes): each coefficient cut into `P` pieces of at most 112 bits (2 of 64 bits for a 127-bit `n`, 4 of 64 bits for 255 bits, 5 of 105 bits for 521, with 5 to 8 primes; `P` and the number of primes chosen by an estimate of the cost), each piece a polynomial transformed once a prime, the `2P − 1` pieces of the product `C_t = Σ A_s·B_(t−s)` summed in the transforms (the sums reduced every four products, within the 32-bit lanes) and brought back by one inverse transform each, every `c_t` exact by the theorem (below half the primes' product), and the coefficient `Σ c_t·2^(t·w)` divided by `n` once. Otherwise Kronecker substitution, each polynomial evaluated at `2^s` with `s = 2·bits(n−1) + bits(m)`, one product of integers (Karatsuba's, Toom's or the transforms), and each field of the product divided by `n` once.
- **Over the integers** (FLINT's `fmpz_poly_mul`): up to 12 coefficients in the shorter operand (48 in the longer), for coefficients of at most 64 bits, the schoolbook method in machine integers (each sum of products in an `i128` when it fits, else the positive and the negative products apart in three words), one number built a coefficient; up to 4 coefficients of up to 2,048 bits, the schoolbook method on limbs (each product of magnitudes added to the positive or the negative sum). From 48 coefficients in the shorter operand, for coefficients of at most 112 bits whose sums of products stay below `2²²⁴`, the transforms on the coefficients directly, as modulo a word: each coefficient cut into digits of at most 28 bits, negated for a negative one, the convolution modulo the fewest primes whose product `P` exceeds four times the largest sum, and the residues shifted by those of `⌊P/2⌋`, so that the theorem gives `c + ⌊P/2⌋` in `[0, P)` and the signed `c` comes out exactly in 128 bits for up to four primes (`P < 2^116`), three limbs for up to six, four beyond. Otherwise Kronecker substitution with signs: a polynomial whose leading coefficient is negative is negated, its value at `2^s` (`s` the bits of the largest product sum plus a sign bit) is the sum of the positive coefficients' fields less that of the negative ones', and the product's coefficients come back from the bottom, a field `v` (with the borrow from below) being `v` below `2^(s−1)` and `v − 2^s`, a borrow of 1 from the field above, from it.
- **Over the rationals**: both polynomials over common denominators, the integer polynomials multiplied so, each coefficient of `AB/(da·db)` then in lowest terms (the schoolbook method up to 16 coefficients, where products of rationals in words beat a common denominator that grows with the length).
- **Over the floats and the complex floats**: the schoolbook method, whose error is bounded componentwise, `|ĉ_k − c_k| ≤ γ_m Σ|a_i||b_(k−i)|`: each coefficient the sum from `+0` of the rounded products `a_i·b_(k−i)` over the non-zero `a_i` by increasing `i` (a product and a sum each, no fused multiply-add), so that an infinity or a NaN reaches only the coefficients of its terms; for a first operand of at least 16 coefficients and at most a quarter of the second's length, the second of at least 448 (complex: 8, and a second of at least 64), 32 coefficients of the product (8 complex) at a time in registers, the same sums in the same order; below, the generic loop over the second operand, which is faster there. It is the method whenever the shorter operand has fewer than 32 coefficients or `la·lb ≤ c·N·log₂N` (`c` = 19 for floats, 5 for complex floats, whose schoolbook products cost four real ones; `N` the least power of 2 at least `la + lb − 1`), the line of the radix-2 transform this crate used before, kept for the bound: for square products up to 441 coefficients (and again from 513 to 654), complex up to 101 (and from 129 to 151); against an operand of 5,000 coefficients up to 404 in the shorter (106 complex). Beyond, the crate's linear convolution ([`fft::convolve`](fft.md)): transforms of a length `2^a·3^b·5^c·7^d` (real ones for floats), overlap-add when one operand is much longer, the plans cached, or its direct kernels, as its model of their costs chooses; its error is bounded in norm, a few units of rounding times `‖a‖₂‖b‖₂·log₂N` for a transform of the length `N`. Coefficients much smaller than the norms can then lose their relative accuracy, which the schoolbook method keeps, and an infinity or a NaN among the coefficients may make every coefficient NaN.
- **Power series inverses**: the schoolbook recurrence `g_j = −c₀⁻¹ Σ fᵢ g_(j−i)` (dot products reduced once) up to 160 coefficients modulo a word (16 for several limbs, 32 over a field), Newton's iteration `g ← g − xʰ·((f·g − 1)/xʰ · g mod x^(n−h))` beyond, about three products of the length `n`. Modulo a word the first product is a middle product, only its coefficients `h…n−1` wanted: when the transforms are the method (on NEON), by the transform of the fewest points whose wrapped coefficients land below `h`, which are known (about `n` points where the whole product takes `n + h`), the theorem applied to the coefficients kept; otherwise the whole product. Over the rationals through the integers (FLINT's `fmpq_poly_inv_series` way, with each coefficient then in lowest terms): `a = A/den`, `A(c·y) = c·U(y)` for `c = A₀` with the integer `U = 1 + Σ A_j c^(j−1) y^j`, whose inverse is exact over the integers by Newton's iteration on integer products, and `1/a` has the coefficients `den·v_j / c^(j+1)`, brought to lowest terms by the powers of `gcd(num, gcd(c, den))` alone (every common prime divides `c`), found by doubling; that for a common denominator of up to 64 bits, as the powers `c^(j+1)` overshoot the true denominators by far beyond. Over a longer common denominator, Newton's iteration on one common denominator as FLINT does, `g = N/d`, `N' = N·(2·den·d − A·N) mod x^m` over `den·d²` with the common content taken out at each step, and each coefficient brought to lowest terms once at the end.
- **Division**: the schoolbook method for a quotient or a divisor of up to 256 coefficients modulo a word (128 modulo 2 to 4 limbs and 32 beyond, its sums in `2k + 2` limbs), the remainder's coefficients kept as unreduced sums `a_j + Σ(n − t_i)·d_(j−i)` (in a word, two or three) and only the one that gives the next quotient coefficient reduced at each step; beyond, Newton's method: the reversed quotient is the reversed dividend times the inverse of the reversed divisor modulo `x^(la−ld+1)`, the remainder `a − q·d` on its low coefficients. Over the rationals from 16 coefficients in the quotient through the integers: the pseudo-division of the numerators over common denominators, then `q = Q·dd/(da·lᵏ)` and `r = R/(da·lᵏ)` in lowest terms. Over the floats Newton's method from 512 coefficients in the quotient and the divisor. Modulo 2 the packed division above: Barrett's method on words, and on polynomials for long divisors and quotients.
- **Pseudo-division and exact division over the integers**: the schoolbook method on limbs, every coefficient a magnitude and a sign in a slot of a width fixed by a bound (`|R|` grows by at most `2·max(|l|, H)` a step), the operations in place. The pseudo-division runs the recurrence `R ← l·R − t·xⁱ·d` lazily: a coefficient keeps the count of the factors `l` it has had and takes those it misses in one product by a power of `l` (computed once) when a step changes it, so a step multiplies only the `deg d` coefficients it subtracts from, and the quotient's coefficients `t·l^(k−1−s)` take their powers once at the end, where the classic method multiplies every coefficient below the top, and the quotient, by `l` at every step; a negative `l` divides by `−d` and turns the signs of `q` and `r` by the parity of `k`. The exact division takes each quotient coefficient as the top coefficient divided by `l`, `None` at the first that does not divide or when a value outgrows the slots, sized by Mignotte's bound on a divisor's cofactor (which an exact division never exceeds). Below 7 products in all (`(la − ld + 1)·(ld − 1) ≤ 6`) the schoolbook method on numbers, whose set-up is less; by a constant `l`, `q = l^(deg a)·a` and `r = 0` directly (one power, one product a coefficient).
- **Kronecker substitution into the division of integers** takes over from 12 coefficients in the quotient for `div_exact` (48 for divisors of 512 bits and more) and from 384 (32 for divisors of 1,000 bits) for `pseudo_div_rem`. For `div_exact` the values at `2^s` divide when the polynomials do, so a remainder of the integers gives `None`; the quotient read back is checked by one product, the field doubled while the check fails, up to Mignotte's bound `2^(deg q)·‖a‖₂` on a divisor's coefficients. It says `None` at once for a divisor wider than Mignotte's bound `2^(deg d)·‖a‖₂` on a divisor of `a`, and tries fields that hold both operands, so that a divisor much wider than the dividend neither breaks the packing nor hides a true quotient. For `lᵏ·a = q·d + r` the field must hold every coefficient of `q` and `r`, bounded from `|q_j| ≤ (|lᵏa| + n·H·|q|)/|l|` (`H` the largest coefficient of `d`), and so the largest power of `l`: the quotient's value is the nearest integer to the quotient of the two values, as the remainder's is below half the divisor's, and the remainder takes one product. Past that crossover the pseudo-division first tries the schoolbook method with the fewest factors of `l`, as FLINT's `fmpz_poly_pseudo_divrem` runs it: a step whose top coefficient `l` divides takes `t/l` into the quotient with no factor, every coefficient keeps the number of factors it was last brought to and takes the missing power in one product when a step touches it, and at the end each coefficient of `q` and `r` is multiplied by the power of `l` it misses, the powers in increasing order each from the one before, so that the result is `lᵏ·a = q·d + r` exactly. An exact or nearly exact division then stays at about the size of its operands (an exact one of 800 by 400 coefficients of 64 bits takes 2.1 ms instead of 31); a division whose fifth step needs a factor goes on to Kronecker substitution, after a few products.
- **Modulo a fixed polynomial** (FLINT's `_preinv` functions): the inverse of the reversed `f` modulo `x^(deg f)` is computed once; a remainder of a polynomial of degree below `2 deg f − 1` is then two products, the second modulo `x^L − 1` for `L ≥ deg f + 1` (the coefficients of `q·f` from `deg f` on are those of the dividend, so the wrapped part is known: `r_j = a_j − (qf mod x^L−1)_j + a_(j+L)`). Modulo a word the remainders, products and squares run on vectors of residues reused from step to step, with the schoolbook division up to 160 coefficients; for an `f` of up to 16 coefficients a product modulo `f` is one pass on the stack, its sums of products in three words divided by `f` in place and reduced once each. Modulo several limbs, from 49 coefficients in `f` (where the products go by the transforms on pieces), the pieces of the inverse and of `f`'s low coefficients are transformed once, and each remainder's two products transform only the other operand and apply the theorem to the coefficients they keep (the quotient's low `lq`, the remainder's low `deg f`). `pow_mod` uses sliding windows of up to 6 bits of the exponent (the odd powers computed once), and for the base `x` squares and shifts, a shift and one step of the division costing `O(deg f)`. Over the rationals `pow_mod` works on the integers: `f = F/d_f` and the base `B/d_b` over common denominators, each value kept as `N/(|l|^E·d_b^P)`, a product an integer product and its remainder modulo `F` the pseudo-remainder with the fewest factors of `l`, and only the `deg f` coefficients of the result brought to lowest terms, once. `compose_mod` is Brent and Kung's: `m = ⌈√(deg p + 1)⌉` powers `g⁰…gᵐ⁻¹` modulo `f`, each block of `m` coefficients of `p` evaluated at `g` as rows of a product of matrices (each entry a dot product reduced once), and the blocks joined by Horner's rule in `gᵐ`: about `2√deg p` products modulo `f`; modulo a word all of it on vectors of residues reused from step to step (one pass on the stack for an `f` of up to 16 coefficients).
- **Composition**: `p = lo + xʰ·hi` gives `p(g) = lo(g) + g^h·hi(g)`, `h` the largest power of 2 below the length, with the powers `g^(2^i)` computed once and Horner's rule for pieces of up to 8 coefficients: `O(M(n·deg g) log n)` instead of Horner's `n` products of growing length.
- **Evaluation at one point**: modulo a word Horner's rule with Shoup's quotient of `x`, in two chains (the even and the odd coefficients, in `x²`) that the core overlaps. Over the integers, from 32 coefficients, the point is taken as one point of `eval_many` (below): Horner's rule on word blocks on limbs, and divide and conquer when the value grows past a few thousand bits.

### Greatest common divisors and resultants

- **Modulo a word `n`.** Euclid's algorithm in Montgomery's form for an odd `n`: each step clears two coefficients of the dividend at a time, the two products a coefficient summed and reduced by one REDC (two multiplications), for `n < 2^63`. A step whose quotient has one or two coefficients and whose divisor is shorter than 600 takes the pseudo-remainder `lc(b)^(δ+1)·a - q·b` instead, three products summed before one REDC and no inverse (an inverse of a word costs about what a pass over 500 coefficients does, a quarter of the time at degree 1000 otherwise); for `n < 2^62`. Each remainder is then the true one times a known factor; the resultant divides the leading coefficients by those factors at the end, all inverted at once by Montgomery's trick, and the product of the leading coefficients is inverted once to check that they were units. An even `n` takes the same steps with remainders by the reciprocal (Möller and Granlund's division of two words by one), one reduction for two rows.
- **Modulo `n` of `k ≥ 2` limbs.** Euclid's algorithm on the delayed-reduction schoolbook remainders of the division (the remainder's coefficients kept as sums and divided by `n` once), and for divisors shorter than 24 coefficients the pseudo-remainder step without an inverse (three products of a coefficient summed in `2k + 2` limbs, one reduction by the division by `n`), the factors tracked and inverted together as on words. The half gcd's own divisions are the two-row steps.
- **The half gcd.** Past a degree, Thull and Yap's formulation of the Knuth–Schönhage algorithm, over any of the coefficient rings (modulo 2 on the packed products, from degree 6,144 in the shorter operand, its base Euclid's steps on the packed words with their matrix below degree 768): for `a` of the degree `n` and `m = ⌈n/2⌉`, the quotients of the top halves `a div xᵐ` and `b div xᵐ` are those of `a` and `b` as long as the divisor keeps half the degree of the top halves, so the first half of the steps comes from a recursive call on the top halves, one step is a division, and the second half a recursive call on the top part of the new pair; the matrices of the steps apply to the low parts, and the whole takes `O(M(n) log n)`. `gcd` takes a half gcd and a division in turn until the degree falls below the crossover, then Euclid's algorithm; `xgcd` keeps the product of the matrices, whose second column gives the cofactors (the tail by Euclid's algorithm with the matrix in Montgomery's form); the resultant records the degree and the leading coefficient of every divisor the half gcd uses, at their true degrees (those of the top parts, shifted), and takes `res(r₀, r₁) = (-1)^(d₀d₁) l₁^(d₀ - d₂) res(r₁, r₂)` over them. The products are those above (the schoolbook method, Kronecker substitution, the number-theoretic transforms on words; the schoolbook method, splitting and Kronecker substitution on limbs). The crossovers, measured with them, are on words by the size of `n`, as the half gcd's products take more transform primes as `n` grows while a step of Euclid's costs about the same for every `n` below `2^63`: degree 1000 for `gcd` and the resultant and 600 for `xgcd` up to 26 bits, 1900 and 950 for 27 to 34 bits, 3000 and 1500 for 35 to 63 bits, and 2000 and 850 from `2^63` (at 17 bits and the degree 3000 the gcd takes 1.28 ms, against 2.44 ms with the crossover of 35 to 63 bits); on limbs 250 and 80; the recursion's base below degree 100 (words) and 60 (limbs).
- **The content over the integers** in a word while it fits one: from the first non-zero coefficient, `w ← gcd(w, |c| mod w)` (one division for a coefficient `w` divides, as most are once it has settled, Stein's algorithm on words otherwise), a coefficient of several limbs first reduced modulo `w`, stopping at 1; the gcd of the numbers only from a first non-zero coefficient beyond a word.
- **Over the integers, the greatest common divisor** of the primitive parts by one of two methods. The multimodular algorithm (by default): the monic gcd modulo primes below `2^62`, scaled to the gcd `l` of the leading coefficients, joined by the Chinese remainder theorem in the symmetric residues (one prime at a time, Garner's step) until the images stop changing; the primitive part is then kept when it divides both. A prime that divides a leading coefficient is passed over, one whose gcd has a higher degree is unlucky and passed over, one of a lower degree starts the images again, and one of degree 0 proves the two coprime (the common case of random polynomials: one prime). The heuristic gcd (Char, Geddes and Gonnet) for the shortest polynomials (up to 8 coefficients) and for long ones of short coefficients (at most 48 bits from 200 coefficients), where one gcd of integers beats the steps modulo two primes: both polynomials evaluated at `ξ = 2^s`, `s` past the largest coefficient, the gcd of the two integers read back in balanced digits of base `ξ`, its primitive part kept when it divides both; for `ξ` above twice the smaller of the two largest coefficients such a divisor is the gcd. When it gives up (three sizes of `ξ`), the multimodular algorithm. Before the contents and the primitive parts, for coefficients of several limbs or polynomials of up to 8 coefficients: the gcd modulo the first prime that divides neither leading coefficient (of the first two); when it is a constant the two are coprime over the rationals (the degree of the gcd modulo such a prime is at least that over the rationals), and the gcd over the integers is the positive gcd of all their coefficients, without the primitive parts; otherwise that prime's gcd is the multimodular algorithm's first image.
- **Divisibility over the integers** by Kronecker substitution: the integer quotient of the values at `2^s` read back as a polynomial, kept when its coefficients are small enough that its product by the divisor has coefficients below `2^(s-1)`, as the dividend's are, so that the two values agree as polynomials; first at the size of the dividend's coefficients, then at Mignotte's bound on the coefficients of a factor, which decides. The leading and the constant coefficients refute most non-divisors first.
- **The resultant over the integers** modulo as many primes below `2^62` (dividing neither leading coefficient) as Hadamard's bound `|a|₂^deg b · |b|₂^deg a` asks, each by the word kernels above, joined by the crate's Chinese remainder theorem (product and remainder trees past 150 moduli) in the symmetric residue. Up to 7 coefficients in the shorter polynomial (and coefficients of 40 bits or more) the last subresultant of the chain instead, on the pseudo-division. The discriminant from the resultant with `f'`, divided exactly by `lc(f)`.
- **The subresultants** by the subresultant remainder sequence with Lazard's optimisation of the defective steps, as Ducos presents it: each pseudo-remainder `prem(a, -b)` divided exactly by the factor `s^δ·lc(a)` the sequence knows, and after a gap of `δ` degrees the subresultant `lc(b)^(δ-1)·b / s^(δ-1)` by Lazard's powering, every intermediate power divided at once so that none grows past the result.
- **Over the rationals**, everything on the primitive integer polynomials: `gcd` that of `to_primitive`'s parts, made monic; the resultant `c^deg q · d^deg p · res(p, q)` for `c·p` and `d·q`; `xgcd` the gcd, then for the coprime primitive parts `a'` and `b'` of `a/g` and `b/g` the cofactors with `s·a' + t·b' = res(a', b')`, multimodular: the extended gcds modulo primes times the resultants there, joined past Hadamard's bound, which bounds every minor of the Sylvester matrix and so (Cramer's rule) every coefficient of `s` and `t`; the primes that divide the resultant are passed over. Euclid's algorithm over the rationals grows its coefficients at every step; FLINT's `fmpq_poly_xgcd` takes the same multimodular route.
- **Over the floats**, Euclid's algorithm on the division above, and the resultant from its remainders. A remainder is taken for zero only when it is exactly zero, so the gcd of polynomials known only to rounding is that of their exact binary values; the half gcd brings no accuracy and is not used.

### Evaluation and interpolation

- Modulo a word, Horner's rule runs at eight points at once, each its own chain of products: Shoup's product by the point, `a·x − ⌊a·x′/2⁶⁴⌋·n` with `x′ = ⌊x·2⁶⁴/n⌋` computed once a point, the values kept below `3n` between the steps for `n < 2⁶²` and reduced at the end. From `2⁶²` an odd `n` takes Montgomery's products by `x·2⁶⁴ mod n`, `REDC(v·x·2⁶⁴) = v·x mod n`, so no residue is converted and each step is three products and one correction; an even `n` takes Shoup's products below `2⁶³` and Möller and Granlund's reciprocal above. Below `2³⁰` on aarch64 it runs on NEON, 32 points at once in 32-bit lanes (`umull`, `umull2` and `uzp2` for the high half of `v·x′`, `mul` and `mls` for `v·x − q·n`), about 0.14 ns a coefficient and a point on an Apple M4 Max (0.27 ns on Shoup's products below `2⁵⁰`, at 256 to 1024 coefficients and points). Fewer than four coefficients or points take the reciprocal's products without the constants of the points.
- Modulo an odd `n` of 2 to 16 limbs, Horner's rule takes Montgomery's products in registers, `v ← REDC(v·x′) + c` with `x′ = x·R mod n` (`R = 2^(64k)`, `x′` one product by `R² mod n`), four points at once: the crate's Montgomery kernels of the integers' modular powers, inline assembly on aarch64 (in registers up to eight limbs, the operands in memory from nine), portable Rust elsewhere. A product of two residues takes two such products, the second by `R² mod n`, and a product by a fixed factor one. An even modulus, or one of more than 16 limbs, takes the residues' products by division.
- Above the crossover in coefficients and in points (modulo a word 1408, 1280, 448 and 448 for `n` below `2³⁰`, `2⁶²`, `2⁶³` and `2⁶⁴`; modulo an odd `n` of `k` limbs 960 for two limbs, 448 for three, 304 for eight, 224 for nine and 152 for sixteen; `8 + 2k` for an even `n`), `eval_many` builds the subproduct tree of the points, `ceil(log₂ m)` levels of monic nodes held one vector a level, and carries scaled remainders down it: Bernstein's scaled remainder tree, which is the transposed multipoint evaluation of Bostan, Lecerf and Schost. The root's `σ`, the first `m` coefficients of `(f mod M)/M` in `1/x`, is one product of the reversed `f` with the power series inverse of the reversed root (`inv_series`, the root's constant being 1); a child `u` with the sibling `w` of the node `v` then has `σ_u = (σ_v · M_w)[deg w .. deg v)`, two middle products a node and no division, and the values at the last level are `σ₀ − b·σ₁` and `σ₀ − a·σ₁` for the node `(x − a)(x − b)`. More points than coefficients are taken in groups of as many as the coefficients, a tree each; fewer, with the inverse to as many terms as the coefficients.
- The products of the trees are `ModPolynomial`'s products (the schoolbook method, Kronecker substitution and the number-theoretic transforms modulo a word; Kronecker substitution modulo several limbs), with two savings. The nodes are monic, `(x^a + u)(x^b + w) = x^(a+b) + x^a·w + x^b·u + u·w`: only `u·w` is a product, of half the transform of the whole when the degrees are powers of two. A middle product, the coefficients `lo..hi` of `a·b`, is the cyclic product modulo `x^L − 1` for the least power of two `L` at least `hi`, both lengths and `len(a) + len(b) − 1 − lo`, whose wrapped coefficients all fall below `lo`, with the Chinese remainder theorem taken for the wanted coefficients alone: in the down pass `σ_v` of `2^ℓ` coefficients times the half `M_w` takes the transform of length `2^ℓ`, where the whole product takes `2^(ℓ+1)`. Measured on an Apple M4 at 4096 and 16384 points modulo a 50-bit prime, the cyclic middle products take 0.75–0.78 of the time of full ones, the monic products 0.66 of `from_roots`'s. Below the transforms' crossover the middle product is the schoolbook method on the wanted coefficients alone, each a sum reduced once against the reversed operand; modulo several limbs, up to 16 coefficients, the same with the sums in `2k + 1` limbs.
- `from_roots` takes blocks of roots one linear factor after the other, `r ← r·(x − a)` from the top down (16 modulo a word below `2³⁰`, 32 above, by Shoup's products by `−a`; 16 modulo several limbs, one Montgomery product a step by the factor of `−a`), and then the product tree of the blocks on the monic products. Over the integers, blocks of 32 roots; roots below `2⁶⁴` in magnitude take one vector of coefficients in two's complement, as many limbs each as the bound `Π (1 + |rᵢ|)` of the roots so far, one pass of products by a word a step; larger roots take signed limb vectors. Floats take up to 16 roots one after the other, and the product tree of the linear factors above.
- `interpolate` modulo `n` builds the tree, takes the weights `M′(xᵢ)` by the scaled remainder tree, inverts them together by Montgomery's trick (one inverse and three products each; a weight that is not a unit is the panic), and combines `Σ (yᵢ/M′(xᵢ))·M/(x − xᵢ)` up the tree, `r_v = r_u·M_w + r_w·M_u`, on the monic products. Up to 96, 176, 160 and 112 points modulo a word (by the same classes) Lagrange's form is faster; modulo an odd `n` of several limbs Newton's form up to 52 points (44 from nine limbs, 32 from thirteen), and up to 8 for an even `n`.
- `interpolate_lagrange` modulo a word takes `M` by `from_roots`, the weights `M′(xᵢ)` by Horner's rule at all the points, inverted together, and the sum `Σ cᵢ·M/(x − xᵢ)` by synthetic division, which is Horner's rule on `M` at `xᵢ` with its partial sums: eight points at once, Shoup's products by `xᵢ` and by `cᵢ` (Montgomery's from `2⁶³` for an odd `n`), two a coefficient and a point. Modulo several limbs, and over a field, it takes the weights `Π_{j≠i}(xᵢ − xⱼ)` one product at a time. `interpolate_newton` takes all `m(m−1)/2` differences of the points at once, inverted together by one inverse, the divided differences a column at a time, and expands Newton's form by Horner's rule on polynomials, one Montgomery product a step by the factor of the point.
- Over the integers each value has about `deg·bits(x)` bits, and so has every remainder of the tree at every level: the tree costs as much as evaluating at each point and loses to it by its constants (by 10x at 300 points of 16 bits). `Polynomial<Number>::eval_many` therefore evaluates point by point on limbs, the coefficients converted once: for a point of at most 30 bits Horner's rule on blocks of `k` coefficients, `k·bits(x) ≤ 64`, the long value multiplied by the word `xᵏ` once a block and the block's sum of words in 128 bits; for a polynomial whose `deg·bits(x)` passes 12,000, divide and conquer, `lo(x) + x^h·hi(x)` with the powers `x^(2^j)` by squaring, down to parts of 12,000 bits. The integer trees of `from_roots` and `interpolate` take the products over the integers (Kronecker substitution, and the multimodular transforms for long operands of small coefficients).
- Over the rationals, `eval_many` writes `p = k·P` with `P` primitive over the integers and evaluates `P`: at integer points directly; at points of a common denominator `L` of at most twice the bits of the largest one, `Q(L·x)/L^d` with `Q(z) = L^d·P(z/L)` over the integers; otherwise each point `a/b` by Horner's rule on `Σ Pⱼ·aʲ·b^(d−j)`. `from_roots` takes `Π (bᵢx − aᵢ)/Π bᵢ` for the roots `aᵢ/bᵢ` by the integer product tree. `interpolate` brings the points to integers `Xᵢ = L·xᵢ` by their common denominator, takes the weights `Wᵢ = Π_{j≠i}(Xᵢ − Xⱼ)` directly (the differences multiplied together in a word while their product fits it, each word into a limb vector in place, the parts by a product tree: over the integers cheaper than `M′` by the tree), and combines `Σ (yᵢ/Wᵢ)·M/(z − Xᵢ)` up the integer tree as integer polynomials over one denominator a node, `D_v = lcm(D_u, D_w)`; the coefficients are then `Qⱼ·Lʲ/D`, each reduced.
- Floats evaluate by Horner's rule at eight points at once, each value the same as `eval`'s bit for bit, and interpolate by Newton's form: the trees lose the accuracy a float has (the remainders cancel), and the monomial basis is ill-conditioned for many points whatever the method. `from_roots` takes the product tree.

### Roots

- `roots()`: the Aberth–Ehrlich iteration, as Bini's `polzeros` and MPSolve's floating-point stage run it. The polynomial is scaled by a power of two (exactly) to a largest coefficient of about 1, and the roots at 0 are taken out. The starting points lie on circles of the radii the Newton polygon of the moduli gives (the upper convex hull of `(i, log2|aᵢ|)`, an edge from `i` to `j` a circle of `j − i` points of the radius `(|aᵢ|/|aⱼ|)^(1/(j−i))`), turned by `2π·i/n + 0.7` so that none is on the real axis. Each sweep refines every approximation not yet done by `zᵢ ← zᵢ − N/(1 − N·Σⱼ 1/(zᵢ − zⱼ))` with `N = p(zᵢ)/p'(zᵢ)`, the new values used at once (Gauss–Seidel); the sum is two loops over the real and imaginary parts held apart, which the compiler vectorises. A point outside the unit circle is evaluated through the reversed polynomial at `1/z`, `p/p' = z·q/(n·q − w·q')`, so that nothing overflows. The sum is by Smith's division for approximations beyond `2^±480`, whose differences' squares would overflow. An approximation on another is moved off it by a few units of its size (Newton's step alone would keep the two together). An approximation is done when `|p(z)|` is within the running error bound of Horner's rule (`8u·Σ|bₖ||z|ᵏ` over the partial values), after one more step kept only if it stays within that bound (near close roots `p'` is small and a step of the noise over it can jump far), or when Newton's correction is below its rounding — unless another approximation is within its error ball `(|p| + e)/|p'|`, two at one simple root, which take up to 20 more steps for the iteration's repulsion to separate them; at most 300 sweeps. Then the approximations whose inclusion discs `n·(|p| + e)/|p'|` meet (near a multiple root, or where the polynomial is so ill conditioned that several stopped at the rounding of double precision in one region while a root next to it has none) are polished by the same iteration in double-double arithmetic (about 106 bits: the coefficients are exact there, the sums in double precision from the double-double differences), a polished value that is no longer a root to double precision's rounding falling back to the one before. A group of them whose double-double discs overlap and which all lie within `4·(4e₀/|t_m|)^(1/m)` of their centroid — the radius the rounding `e₀` of double-double arithmetic scatters an `m`-fold root's approximations to, `t_m` the `m`-th Taylor coefficient there and not vanishing to its rounding — is a root of the multiplicity `m`, all `m` set to it after Newton's method on `p^(m−1)` from the centroid. (In double precision that radius tends to 1 as `m` grows, and distinct roots far apart passed for one multiple root.) For a real polynomial the roots are then made exact conjugates and reals with every value still a root to its rounding: an approximation above the axis and the one below nearest its conjugate are a pair when their discs meet and the mirror is nearer than the axis; those whose discs reach the axis and whose real parts are still roots to their rounding are real; each of the rest, the farthest from the axis first, stays as it is with its exact conjugate, in the place of the unused approximation nearest that conjugate; one left over is real, by Newton's method on the real axis from its real part. `O(n²)` operations a sweep; the iteration evaluates by Horner's rule and multiplies no polynomials.
- The evaluations of an integer polynomial at a dyadic point, for the certified and the real roots, are Horner's rule in fixed point on limbs. The point is `x = ±X·2^(−64f)`; each value of Horner's rule is held in `W` limbs in two's complement of a unit `2^(64u)`, the unit the least that holds the value's bound with two bits to spare, so that a step is one product of limbs with the low limbs dropped and one addition, no allocation, no normalisation, no rounding. The bound is `Σ|aⱼ|·max(1, |x|)^(n−i)` for the real roots' evaluations and the exact running `Σ_{j≥i} |aⱼ||z|^(j−i)` (carried in double precision, rounded up, step by step) for the complex ones. Each dropped limb is an error below one unit, carried through `|x|` (through `|Re z| + |Im z|` for the parts of a complex value) and the change of unit in double precision, rounded up at the end, so that every value comes with a bound of its error; a value is exact when nothing nonzero was dropped, which a width of `⌈(log2 bound + 3)/64⌉ + n·f` limbs ensures. Below the products' schoolbook sizes (eight limbs) the product is a schoolbook on the stack. From eight limbs and the degree 8 a value alone is evaluated by rectangular splitting (Paterson and Stockmeyer's, for small coefficients): the powers `x^j` for `j ≤ m ≈ √(n + 1)` each to `W` limbs of its own bound, the blocks `Σ_{j<m} a_{km+j}·x^j` by products of a coefficient's few limbs with a power's, and Horner's rule in `x^m` on the blocks, `√n` full products and `n` short ones against Horner's `n` full ones, the errors counted alike. The sign of a value is certain when its magnitude exceeds its error bound; else the width doubles, and a value that is 0 modulo `2^61 − 1` (always so at a root) is evaluated at the exact width at once. Multipoint evaluation is not used: the refinement evaluates at one point a root, a point that moves with each Newton step, in fixed point to a precision of its own, where a subproduct tree would take exact products of the points' linear factors, whose sizes grow with the tree, and random polynomials have `O(log n)` real roots.
- `complex_roots(prec)`: the square-free factorisation by Yun's algorithm with the greatest common divisor over the integers (after a test modulo `2^61 − 1` from the coefficients' limbs, Euclid's algorithm without divisions, that settles the common square-free case without a gcd over the integers), the root 0 taken out exactly. For each factor `f` of degree `d`, starting points from the iteration of `roots()` on the coefficients rounded to `f64` (scaled by a power of two) when they are within `2^1000` of each other (so that none underflows in it), else from the Newton polygon in `BigComplex` at 64 bits followed by the iteration there; then the Aberth–Ehrlich iteration at doubling precisions (the sum `Σ 1/(zᵢ − zⱼ)` in double precision from the parts, the differences of nearby points taken exactly) until the discs are isolating, one evaluation of `f` and `f'` a step serving the correction and the stopping test. The disc about `z` of the radius `d·(|f(z)| + E)/(|f'(z)| − E')` holds a root, since `f'/f = Σ 1/(z − rₖ)`; `f`, `f'` and their error bounds `E`, `E'` come from the evaluation in fixed point (above) to about `w + 2·log2 d + 8` bits, or, at a point whose parts are of very different sizes (as `2^3000 + 2^−3000·i`), from Horner's rule in `BigComplex` at the precision `w` with `E = (2d + 2)·2^−w·Σ|aᵢ||z|ⁱ` and `E' = (4d + 4)·2^−w·Σ i|aᵢ||z|^(i−1)`; the bounds and the radius are rounded up (`Round::Up`, `Round::Down` for the denominator). `d` such discs that are pairwise disjoint (tested in double precision with room for its errors, and with directed roundings where that does not decide) hold one root each. An approximation on another (or whose sum would hold a zero difference) is moved off it by `2^(−w/2)` of its size, and the sum takes each term by Smith's division, which cannot overflow where the squares of differences near `2^814` would. A group of discs that still overlaps after a doubling of the precision converges linearly, as to a multiple root: it is restarted on a circle about the root `ρ` of `f^(m−1)` (Newton's method from the centroid) of the radius `|f(ρ)/(f^(m)(ρ)/m!)|^(1/m)`, MPSolve's cluster restart; this is what isolates Mignotte's pairs. When the same groups overlap through three doublings (a start in double precision that its range spoilt, as for a root near `2^900`), the factor starts again, once, from the Newton polygon at the working precision. Each isolated disc is then refined by Newton's method at doubling precisions, a new disc accepted only inside the old one (`|z' − z| + r' ≤ r`), until its radius is below `2^−(prec+2)·|z|`; more precision when the error of the evaluation holds the radius, and a new isolation of the factor at four times the precision if Newton's method would not shrink a disc. A disc of a real polynomial whose centre, moved to the real axis with its radius grown by the distance, meets no other disc holds a real root; the conjugate of a disc that meets only one other disc is that disc's root's: those are made real or conjugate exactly, and only one of each pair is refined. The values are rounded to `prec` bits (more where two rounded discs would meet: a root whose rounded disc meets another's takes, once a round, the bits `log2(|z|/distance) + 6` the nearest one needs, and at least 8 more, a conjugate always at its root's precision) and the radii grown by the rounding. `O(d²)` operations at the working precision a sweep, `O(d)` a Newton step of a root.
- `real_root_intervals_descartes()`: the Vincent–Collins–Akritas bisection on each square-free factor, as Rouillier and Zimmermann and FLINT's `fmpz_poly_isolate_real_roots` run it, for the positive roots of `f(x)` and of `f(−x)`. The positive roots are below `2^s`, `s` from the smaller of Kioustelidis's bound and the local-max-quadratic bound of Akritas, Strzeboński and Vigklas; the polynomial `f(2^s·x)` holds them in `(0, 1)`. A node is the polynomial `q` of the interval `(c/2^k, (c+1)/2^k)` mapped to `(0, 1)`; its test is the number of sign variations of `(x + 1)ⁿ·q(1/(x + 1))`: none, no root; one, an isolating interval; more, the halves `2ⁿ·q(x/2)` and its shift by 1, with the midpoint tested (`q(1/2) = 0` is an exact root). The polynomials are flat vectors of limbs, every coefficient in two's complement of one width, so a Taylor shift is `n(n+1)/2` additions of one width (an `adcs` chain on aarch64, two limbs a step), with the common power of two divided out after each step; from 256 coefficients of at most 96 bits (512 to 1,200 bits, about `512 + bits/10` beyond) the Taylor shift by 1 is divide and conquer, `p(x + 1) = lo(x + 1) + (x + 1)^m·hi(x + 1)` for the largest power of two `m` below the length, the powers `(x + 1)^(2^j)` by squarings, the products those of `Polynomial<Number>` (Kronecker substitution into the integers' products), Horner's scheme below 64 coefficients. The test runs in double precision first, values and magnitudes side by side through the shift, a sign taken only when the value exceeds `(ops + n + 2)·u·1.03` times its magnitude, and stops at the second variation; only when a sign is not certain is the test exact. A right half waits as its left half plus "shift by 1": the shift is done only if the node is split or its test needs it exactly, so that the exact work is one Taylor shift for each split right half. The double-precision test covers `n·shifts ≤ 1700`; past that the shift is exact and the test in double precision on it, up to `n = 1700`.
- `real_root_intervals_cf()`: Vincent's theorem by continued fractions, Akritas and Strzeboński's VAS: a node is a polynomial and a Möbius transformation `(a·x + b)/(c·x + d)`; one sign variation is an interval, none no root; else the polynomial is shifted by the local-max-quadratic lower bound of its positive roots, rounded down to a power of two `2^s` (Horner's shift with each addition shifted by `s`), when it is at least 1, and split into `q(x + 1)` (the roots above 1) and `(x + 1)ⁿ·q(1/(x + 1))` (those in `(0, 1)`), the second skipped when the first kept every variation.
- `real_root_intervals()` (and `count_real_roots`, `real_roots`) choose for the positive and the negative roots of each factor apart: continued fractions when the bound `2^s` of those roots has `s ≥ 7` (roots far apart in a long interval: Wilkinson's have `s ≥ 8` from the degree 10, Chebyshev's at most 5 to the degree 200, random ones at most 4), else Descartes's rule, whose nodes at the depth `2·log2 n + 16` and below (past the depths that separate the roots of random and Chebyshev's polynomials) go on by continued fractions, from `(y + 1)ⁿ·q(1/(y + 1))` and the transformation `2^(s−k)·(c·y + c + 1)/(y + 1)`: Mignotte's close pairs then cost what continued fractions cost, not one level of the bisection per bit of their distance. (Up to the degree 100 continued fractions are also faster on random polynomials (52.0 µs against 90.7 µs at 100), but slower on Chebyshev's (44.1 µs against 28.1 µs at 20, 864 µs against 275 µs at 50), so the degree alone does not choose.)
- The isolating intervals of the factors are then bisected (the signs at the dyadic midpoints by the evaluation in fixed point, exactly at other rationals) until no two overlap and no end of one is a root; the sign just right of the left end of an interval is found only when a bisection or the refinement needs it, and the one just left of the right end is its opposite, the root inside being simple.
- `real_roots(prec)`: an isolating interval of one sign over more than two binades (as from the bound of the roots) is first halved in its exponent, at the power of two between its ends', down to a few binades. Each isolating interval is narrowed by Newton's method on dyadic points safeguarded by bisection (an iterate outside the interval halves it; one whose step shrank less than 8 times, linear convergence as towards a close pair of roots, cuts the interval there), at a precision at least 32 bits finer than the interval, the last iterate proven by the signs at `y ± 2|c|` (`c` its correction), then at doubling precisions to `prec + 32` bits (at least 96), one step at each (a step much larger than the last level's error is linear convergence: 16 bisections and Newton's method again from the narrowed interval); the value to `p + κ` bits relative to its bound, the derivative to about `p/2 + κ` bits (by Horner's rule together, or each by rectangular splitting when wide), where `κ` is the conditioning found as it goes: the bits by which the uncertainty of the correction `v/d` exceeds `2^−p·|x|` are added to it, and it is passed on from one root of a factor to the next (about 120 bits near ±1 for Chebyshev's `T₁₀₀`). A value 0 to its error and modulo `2^61 − 1` is tested exactly: the point is then the root. The value rounded to `prec` bits is proven correctly rounded by the signs of the factor at the ends of the interval of the reals that round to it (clipped to the isolating interval): different signs put the root inside; a zero sign at an end means the root is that end, a midpoint, rounded ties to even; the same signs cut the interval there and raise the precision by 32 bits.
- `count_real_roots()`: for each square-free factor of the degree 12 or more, the first Sturm remainder `−rem(f, f')` (one pseudo-division, `O(n)` operations); when its degree is at most `min(16, n/4)` (`xⁿ` plus a polynomial of a low degree, as Mignotte's) Sturm's sequence of pseudo-remainders with the true remainders' signs, each divided by its content (a short divisor's pseudo-remainder with the dividend's coefficients taken into a window of `m + 1` one at a time, each times the power of the leading coefficient the steps before would have given it: `O(n·m)` operations), and the count `V(−∞) − V(+∞)`; otherwise the isolation's method, counting its intervals without building them; plus one for the root 0.

### Factorisation

- **Square-free parts.** Over the integers, Yun's method on the primitive part — `a₀ = gcd(f, f')`, then `aᵢ = gcd(bᵢ, cᵢ - bᵢ')` with exact divisions — after a quick test: a polynomial square-free modulo one of three primes from `2^31` up (not dividing the leading coefficient, unlikely to divide the discriminant) is square-free, which settles most inputs with one greatest common divisor modulo a word. Modulo `p`, Yun's method on what the derivative sees and, for the rest, whose derivative is 0, the `p`-th root `Σ aᵢ x^(ip) = (Σ aᵢ xⁱ)^p`, recursively with the multiplicities times `p`.
- **Distinct degrees.** For a small `p` and a short polynomial (the degree times the bits of `p` at most 48), `x^(p^i) mod f` one `p`-th power after another, with a greatest common divisor with `x^(p^i) - x` each time. Otherwise the baby steps and giant steps of Kaltofen and Shoup (von zur Gathen and Shoup's interval method): `l = ⌈√(n/2)⌉` baby steps `hᵢ = x^(p^i)` and the giant steps `Hⱼ = x^(p^(lj))`, the factors of the degrees in `(l(j-1), lj]` from the greatest common divisor with `Πᵢ (Hⱼ - hᵢ) mod f`, and each degree from those with `Hⱼ - hᵢ`; it stops when what is left is too short to be reducible. Each giant step is a Brent and Kung modular composition with the powers of `x^(p^l)`, each baby step one with the powers of `x^p`, or for a small `p` (its bits at most `2√(n/(l-1))`) the `p`-th power of the last, which costs fewer products modulo `f`. One table of powers serves every composition with the same `h`, so its rows are `m = √(uses·n)` instead of `√n` for `uses` compositions: the powers cost one product modulo `f` each and a composition `⌈n/m⌉ - 1` (its matrix product costs `n²` products of residues whatever `m`), which takes a third off the factorisation modulo a 64-bit prime at degree 300 to 1,000.
- **Equal degrees.** Cantor and Zassenhaus's random splitting by `gcd(g, a^((p^d - 1)/2) - 1)`, for `p = 2` by the trace `a + a² + … + a^(2^(d-1))`. For a large `p` (its bits squared above four times the degree), `a^((p^d-1)/2) = N^((p-1)/2)` with `N = a·a^p·…·a^(p^(d-1))`, the `a^(p^i)` by compositions with `x^p`, which the distinct-degree step already has: `d - 1` compositions and the bits of `p` in squarings, instead of `d` times the bits.
- **Berlekamp.** The matrix `Q` of the Frobenius map `a ↦ aᵖ` on `F_p[x]/(f)` from `x^p mod f` and its powers, and the kernel of `Q - I` (its dimension the number `r` of factors) by Gaussian elimination to echelon form, the entries below the pivots kept as unreduced sums of products in one, two or three words (each takes at most one product a pivot) and reduced only as their column or row is reached, then back-substitution for each free column. Its elements split `f`: for `p` up to 11 by greatest common divisors with `v - s` for every `s` of `F_p`; above, a random element `v` of the kernel is a constant modulo each factor, so its minimal polynomial modulo a part `u` — that of the sequence `⟨w, vⁱ mod u⟩` for a random `w` and `i < 2r`, by Berlekamp and Massey's algorithm — is `Π (x - c)` over its distinct values, and its roots `c` (of a polynomial of degree at most `r`) split `u` by `gcd(u, v - c)`, all of them at once (NTL's way, instead of a power `v^((p-1)/2)` modulo each part). `factor` takes Berlekamp's method itself for `p` up to 7 and parts of degree up to 175 (`p` = 2), 600 (3) or 300 (5, 7), where its elimination in a word costs less than the distinct-degree steps (half the time at `p = 3` and degree 100 to 300; modulo 2, where both run on packed words, 1.5 times faster at degree 100, and the distinct-degree steps 1.8 times faster at 600).
- **Irreducibility modulo p.** Ben-Or's test: `gcd(f, x^(p^i) - x) = 1` for every `i` up to `n/2`, the roots (`i = 1`) first, then the baby steps and giant steps of the distinct-degree factorisation, one greatest common divisor with the product of the `l` differences `Hⱼ - hᵢ` for each interval of degrees, stopping at the first common factor (FLINT's `nmod_poly_is_irreducible_ddf` is the same idea).
- **Roots.** For `p` below 64, from `f` folded modulo `x^p − x` (each `x^k`, `k ≥ 1`, is `x^(1 + (k−1) mod (p−1))` at every point of `F_p`), with no greatest common divisor at all: the roots are those of the folded polynomial, of a degree below `p`, evaluated at every residue; the multiplicity of a root `r` is the first `k` with `H_k f(r) ≠ 0` for the Hasse derivatives `H_k f = Σ C(i, k)·c_i·x^(i−k)` (the Taylor coefficients of `f` at `r`), each folded in one pass with `C(i, k) ≡ C(i mod p, k)` (Lucas's theorem, `k < p`); 0 has the number of zero coefficients at the bottom, and a multiplicity of `p` or more is counted by synthetic divisions by `x − r`. For a word `p` from 64 up to the degree, every residue tried by synthetic division, each root divided out as often as it divides. Beyond, for each square-free part, `gcd(g, x^p - x)` split into factors of degree 1 by Cantor and Zassenhaus's method.
- **Arithmetic modulo p.** The products, remainders and powers are those of the polynomials modulo `n` ([above](#products-and-division-1)): modulo a word, the schoolbook products with one reduction a coefficient and the number-theoretic transforms on the coefficients, and modulo a fixed `f` the remainders with the inverse of its reversal kept; modulo several limbs, Kronecker substitution with one reduction a coefficient and the delayed-reduction schoolbook division. Where the factorisation measured a choice of its own faster, it keeps it: the transforms of a product modulo a word start from 64 to 150 coefficients by the size of `p` (the schoolbook method beats them up to there, as their length doubles past 64 coefficients); for `p` of up to 13 bits Kronecker substitution into the crate's product of integers takes the lengths up to 300 to 2,048 coefficients (its fields of `2·bits(p) + log₂ n` bits are short, and it beats the transforms 2 to 3 times at `p = 3`), and the remainders modulo `f` go by the Newton inverse of the reversed `f` on those products; for `f` of up to 150 coefficients the remainders are a delayed-reduction schoolbook division by the monic `f` (no product by the leading coefficient's inverse), and for `f` of up to 32 coefficients (80 above 32-bit primes, 48 for squares) a product modulo `f` takes one pass: the sums of products of the operands unreduced, the division run on those sums, one reduction a coefficient of the product in all, 1.3 to 1.6 times faster than a product and a division; the inverse of a word residue is the extended Euclidean algorithm with the cofactors kept as magnitudes and the quotients 1 to 3 by subtractions. The matrix products of the Brent–Kung composition are dot products of the blocks of `g` with the columns of the powers, summed in one, two or three words and reduced once, modulo several limbs in `2k + 1` limbs and reduced once. The greatest common divisors are those of the section [above](#greatest-common-divisors-and-resultants-1): modulo a word Euclid's algorithm in Montgomery's form and the half gcd (Euclid's algorithm on the delayed-reduction remainders for operands of up to 3 coefficients, up to 6 modulo several limbs), over the integers the multimodular gcd (only for inputs that are not square-free), and the cofactors of the Hensel tree by `ModPolynomial::xgcd`.
- **Over the integers.** The factors of a square-free primitive polynomial by Zassenhaus's method. The power of `x` comes off first. One good prime from 3 up (not dividing the leading coefficient, the polynomial square-free modulo it), and more, up to six, while the fewest factors are more than 12 (all six when the bound of the coefficients of a factor has 12 bits a degree or more, where the lifting is long), give their distinct-degree factorisations: the number of factors and their degrees, whose subset sums intersect into the degrees a true factor can have — one factor, or no degree left but 0 and `n`, and the polynomial is irreducible. (More primes prune more but cost a distinct-degree factorisation each: one prime takes half the time of three on random polynomials and products of two at degree 100 to 300, while random polynomials of 1,000-bit coefficients take a tenth of the time with six, which prove them irreducible without a lift, and `x^210 - 1` a tenth with up to six, which bring its local factors down.) The prime with the fewest factors is split completely, and its factors lifted at once in a binary tree (halved by degree) by Hensel's lemma, quadratically (von zur Gathen and Gerhard's Algorithm 15.10 for monic factors; from two limbs of `pᵃ` up, the corrections of a step from `m` to `m₂` are made modulo `m₂/m`, from the errors divided exactly by the odd `m` from the low end, so that only `g·h` and `s·g + t·h` take the full precision), to `pᵃ > 2B` with Mignotte's bound `B = |lc|·C(K, ⌊K/2⌋)·(‖f‖ + |lc|)` for the factors of degree up to `K = ⌊n/2⌋`. The recombination tries the subsets of one lifted factor, then of two, …, up to half (a subset whose degree is above half is tried by its complement, so that the bound of half the degree holds): its degree against the allowed set; the power sums `p₁` and `p₂` of its roots, the sums of those of its factors, scaled by `lc` and `lc²` and kept within the bounds of the roots of `f` (Mignotte's and Fujiwara's; the second catches the even factors of an even polynomial such as the Swinnerton-Dyer polynomials, whose traces are 0), on two words as long as `pᵃ < 2^127`, the partial sums kept from one subset to the next; its constant coefficient dividing `lc·f(0)`; and only then the product and a trial division (the schoolbook one, stopping at the first quotient coefficient that does not divide, for a candidate of degree up to 4; the exact division by Kronecker substitution beyond). When half the factors make a subset, only those with the first are tried, the others being their complements.
- The time of Zassenhaus's recombination grows as `2^r` for `r` lifted factors when the true factors are few, as for the Swinnerton-Dyer polynomials (`r = 2^(k-1)` for every prime): 16 factors at degree 32 take about 0.6 milliseconds, 32 at degree 64 would take some 2·10⁹ subsets, about 40 seconds; FLINT and PARI switch to van Hoeij's method there. Van Hoeij's method, with lattice reduction, comes with [#23](https://github.com/nettrash/a/issues/23).

## Testing

Besides the tests of `lib/tests/math/poly`, the module was fuzzed differentially against FLINT 3.6 (with Arb) and PARI/GP 2.19: a harness in Rust with an oracle in C, every exact operation of `Polynomial<Number>`, `Polynomial<Rational>` and `ModPolynomial` and the roots of every type, on degenerate inputs (the zero polynomial, constants, zeros at the top, sparse `xⁿ ± c`), repeated and known factors, pairs equal, negated, coprime, divisible or with a common factor, the families `xⁿ − 1`, Wilkinson's, Mignotte's, Chebyshev's and Swinnerton-Dyer's, lengths about every crossover up to 4,001 coefficients, coefficients of 1 to 5,000 bits, and moduli from 2 to primes and composites of 512 bits. Its first passes, 14.9 million cases and 1.7 million targeted ones, found twelve faults, in the roots, the arithmetic over the integers and the rationals and the evaluation at one point (wrong or missing roots of ill-conditioned float polynomials of degree 100 and more, a panic of `div_exact` by a divisor much wider than the dividend, hangs of the certified and real roots, and slow cases), each fixed with a test of its own. The last pass compared 16,744,116 cases with no wrong result, panic or hang (two roots of degree 6 whose Arb balls were too wide to compare with, every root here of a backward error below `1.3·10⁻¹⁶` and inside its ball); of its seven slow cases, the pseudo-divisions and the power modulo a polynomial over the rationals were then made fast (above), and the real roots of a short polynomial with huge coefficients still take about 1.1 s, against Arb's 18 ms. Not compared: NTL (FLINT and PARI cover every operation), the arithmetic of floats and complex floats (which would need tolerances), `companion`, `Display`, the shifts, `truncate`, `reverse` and `scale`, and gcds modulo a composite (not unique: checked by their properties).

## Measured

Measured on an Apple M4 Max (macOS 27.2.1) on 2026-11-26 between 07:05 and 07:55, in the final run of the release's speed pass, which timed every area of the crate in turn with nothing else running: one thread on every side, by the harness `bench/poly` of the repository (its method is in `bench/README.md`). `a` is built as a user builds it, by rustc 1.100 stable with the default release profile (no LTO, no `target-cpu`), and so are the Rust references, realfft 3.5.1 and rustfft 6.4.1. The C and C++ references are Homebrew's builds, FLINT 3.6.0 (with its Arb), NTL 11.6.0 (its own `GF2X`, not the gf2x library) and PARI 2.19.1 (its parallel engine set to one thread), and Accelerate's LAPACK of macOS 27.2.1, called from shims built with `-O3 -mcpu=native` that build their inputs before the clock. Each case first compares every reference's output with `a`'s (exact words modulo a word, a fingerprint modulo `2⁶¹ − 1` otherwise, the numbers of factors, roots and intervals, float roots within `10⁻⁶`), then alternates the contenders sample by sample in one process: at least 15 samples of at least 1 ms, fewer (3 to 14) where a round of calls exceeds a quarter of a second, the median of each. The inputs are random (eight variants cycled up to 64 coefficients, two up to 2,000, one beyond), the gcds of pairs of degrees `d` and `d − 1` with a common factor of degree `d/2` unless the table says coprime. FLINT's and NTL's loops write into storage that exists, where `a` returns a new polynomial each call.

A case is an operation at one size; its ratio `a/x` is `a`'s time over that of the fastest reference that does the same work (below 1 `a` is faster), and a loss is a ratio above 1.05. Counted as the same work: FLINT's `fmpq_poly`, which keeps one common denominator where `Polynomial<Rational>` keeps each coefficient in lowest terms; NTL's `ZZ_pX` on word moduli and its `eval` on a vector, which is Horner's rule point by point. Not counted (they do less): FLINT's preinverse of `f` and NTL's `Modulus`, built before the clock for `pow_mod`, `x.pow_mod` and `compose_mod` (PARI, which builds what it needs in the call, is the reference there); PARI's `Flx_roots` and `FpX_roots`, which give the distinct roots without multiplicities; NTL's `DDF` and `EDF`, given `x^p mod f` built before the clock; FLINT's pseudo-division, which may use a lower power of the leading coefficient; NTL's resultant over the integers, randomised by default; the transforms of float products where `a` keeps the schoolbook method and its componentwise bound (and the schoolbook loop where `a` takes the transforms); LAPACK's eigenvalues of the companion matrix, whose backward errors reach 2.7·10⁻¹³ where `a`'s are at most 3.8·10⁻¹⁵ (the rows' notes); Arb's balls and PARI's uncertified roots against `real_roots(prec)`, which rounds every root correctly, and PARI's uncertified `polroots` against `complex_roots(prec)`. Of the 1,814 cases, 1,806 have a reference that does the same work (all but the 8 of `real_roots(prec)`): 573 losses, 120 ties within 5 % and 1,113 cases below 0.95; the median ratio is about 0.78.

Each cell gives `a`'s time, then `a/x` and the reference `x`; in parentheses, marked †, the ratio against a faster reference that does less, not counted. `NTL GF2X` and `PARI F2x` are those libraries' polynomials over GF(2); Arb is FLINT's `arb_` and `acb_` functions.

### Products and division

Products modulo a word, `&a * &b`, both operands of the length:

| length | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` | `2` |
| ---: | --- | --- | --- | --- | --- |
| 1 | 16.7 ns, 2.12 FLINT | 13.7 ns, 2.12 FLINT | 13.6 ns, 2.12 FLINT | 13.7 ns, 2.21 FLINT |  |
| 2 | 20.2 ns, 1.89 NTL | 23.8 ns, 1.91 FLINT | 24.1 ns, 1.93 FLINT | 24.5 ns, 2.06 FLINT |  |
| 4 | 25.7 ns, 1.53 NTL | 33.8 ns, 1.66 NTL | 34.4 ns, 1.15 FLINT | 40.3 ns, 1.07 FLINT |  |
| 8 | 50.1 ns, 1.23 NTL | 93.7 ns, 1.87 NTL | 118 ns, 1.27 FLINT | 100 ns, 1.07 FLINT |  |
| 16 | 69.6 ns, 0.771 FLINT | 204 ns, 1.13 NTL | 263 ns, 1.07 FLINT | 226 ns, 0.913 FLINT |  |
| 64 | 364 ns, 0.810 FLINT | 1.65 µs, 1.29 FLINT | 2.04 µs, 1.22 FLINT | 1.79 µs, 1.05 FLINT | 33.6 ns, 10.9 NTL GF2X |
| 128 | 1.11 µs, 0.931 FLINT | 2.79 µs, 0.857 FLINT | 3.54 µs, 0.744 FLINT | 3.54 µs, 0.738 FLINT |  |
| 256 | 3.03 µs, 1.30 FLINT | 5.44 µs, 0.605 FLINT | 6.91 µs, 0.758 FLINT | 6.91 µs, 0.766 FLINT |  |
| 1000 | 12.4 µs, 1.12 FLINT | 22.2 µs, 0.532 FLINT | 28.1 µs, 0.669 FLINT | 28.2 µs, 0.668 FLINT | 183 ns, 2.67 NTL GF2X |
| 5000 | 80.3 µs, 1.10 FLINT | 146 µs, 0.557 FLINT | 183 µs, 0.688 FLINT | 183 µs, 0.690 FLINT |  |
| 10,000 | 165 µs, 1.10 FLINT | 344 µs, 0.600 FLINT | 378 µs, 0.659 FLINT | 375 µs, 0.664 FLINT | 7.42 µs, 1.89 NTL GF2X |
| 100,000 | 1.98 ms, 1.09 FLINT | 4.68 ms, 0.665 FLINT | 5.71 ms, 0.804 FLINT | 5.72 ms, 0.811 FLINT | 302 µs, 2.12 NTL GF2X |
| 1,000,000 | 19.5 ms, 0.450 FLINT | 45.1 ms, 0.592 FLINT | 55.4 ms, 0.721 FLINT | 55.8 ms, 0.734 FLINT | 11.3 ms, 2.34 NTL GF2X |

Squares modulo a word, `&a * &a`:

| length | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` | `2` |
| ---: | --- | --- | --- | --- | --- |
| 1 | 13.5 ns, 2.10 FLINT | 13.8 ns, 2.13 FLINT | 13.7 ns, 2.13 FLINT | 13.7 ns, 2.13 FLINT |  |
| 2 | 19.1 ns, 2.05 NTL | 22.7 ns, 1.97 NTL | 23.1 ns, 1.85 FLINT | 23.0 ns, 1.84 FLINT |  |
| 4 | 24.1 ns, 1.69 NTL | 32.2 ns, 1.95 NTL | 32.5 ns, 1.07 PARI | 39.9 ns, 1.30 PARI |  |
| 8 | 46.8 ns, 1.62 NTL | 99.3 ns, 2.86 NTL | 110 ns, 1.15 FLINT | 90.4 ns, 0.948 FLINT |  |
| 16 | 67.9 ns, 0.925 NTL | 212 ns, 2.17 NTL | 233 ns, 0.972 FLINT | 194 ns, 0.815 FLINT |  |
| 64 | 361 ns, 0.806 FLINT | 1.27 µs, 1.21 NTL | 1.41 µs, 0.875 FLINT | 1.20 µs, 0.675 FLINT | 25.2 ns, 11.4 NTL GF2X |
| 128 | 1.09 µs, 0.930 FLINT | 2.33 µs, 0.714 NTL | 2.88 µs, 0.664 FLINT | 2.93 µs, 0.670 FLINT |  |
| 256 | 2.48 µs, 1.46 FLINT | 4.30 µs, 0.633 FLINT | 5.43 µs, 0.785 FLINT | 5.48 µs, 0.785 FLINT |  |
| 1000 | 9.60 µs, 1.23 FLINT | 16.9 µs, 0.550 FLINT | 21.2 µs, 0.683 FLINT | 21.2 µs, 0.678 FLINT | 33.3 ns, 5.31 NTL GF2X |
| 5000 | 62.0 µs, 1.20 FLINT | 110 µs, 0.552 FLINT | 136 µs, 0.683 FLINT | 136 µs, 0.687 FLINT |  |
| 10,000 | 127 µs, 1.19 FLINT | 257 µs, 0.591 FLINT | 278 µs, 0.644 FLINT | 277 µs, 0.647 FLINT | 199 ns, 4.34 NTL GF2X |
| 100,000 | 1.50 ms, 1.18 FLINT | 3.48 ms, 0.656 FLINT | 4.23 ms, 0.795 FLINT | 4.23 ms, 0.801 FLINT | 1.80 µs, 4.68 NTL GF2X |
| 1,000,000 | 14.6 ms, 0.477 FLINT | 32.9 ms, 0.582 FLINT | 40.0 ms, 0.700 FLINT | 40.4 ms, 0.714 FLINT | 17.0 µs, 4.50 NTL GF2X |

`div_rem` modulo a word, `2n` by `n` coefficients:

| lengths | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` | `2` |
| ---: | --- | --- | --- | --- | --- |
| 4 / 2 | 77.6 ns, 2.96 PARI | 150 ns, 1.52 PARI | 179 ns, 1.55 PARI | 190 ns, 1.73 PARI |  |
| 16 / 8 | 143 ns, 1.50 FLINT | 271 ns, 1.35 NTL | 338 ns, 1.16 FLINT | 311 ns, 1.16 FLINT |  |
| 64 / 32 | 476 ns, 0.804 FLINT | 931 ns, 1.09 NTL | 1.14 µs, 1.28 FLINT | 1.04 µs, 1.15 FLINT |  |
| 128 / 64 | 1.15 µs, 0.629 PARI | 2.48 µs, 1.09 FLINT | 3.03 µs, 1.33 FLINT | 2.80 µs, 1.23 FLINT | 174 ns, 1.40 PARI F2x |
| 256 / 128 | 3.41 µs, 0.520 PARI | 8.28 µs, 1.18 FLINT | 9.32 µs, 0.945 FLINT | 9.21 µs, 0.956 FLINT |  |
| 1024 / 512 | 29.4 µs, 1.08 FLINT | 49.8 µs, 0.772 FLINT | 60.7 µs, 0.878 FLINT | 61.0 µs, 0.870 FLINT |  |
| 2000 / 1000 | 50.4 µs, 1.04 FLINT | 82.5 µs, 0.613 FLINT | 103 µs, 0.714 FLINT | 102 µs, 0.713 FLINT | 397 ns, 0.720 NTL GF2X |
| 4000 / 2000 | 101 µs, 1.04 FLINT | 170 µs, 0.554 FLINT | 212 µs, 0.690 FLINT | 214 µs, 0.686 FLINT |  |
| 20,000 / 10,000 | 689 µs, 1.21 FLINT | 1.28 ms, 0.663 FLINT | 1.47 ms, 0.759 FLINT | 1.47 ms, 0.758 FLINT | 14.0 µs, 0.887 NTL GF2X |
| 200,000 / 100,000 | 7.62 ms, 1.09 FLINT | 16.4 ms, 0.700 FLINT | 18.3 ms, 0.781 FLINT | 19.8 ms, 0.842 FLINT | 1.09 ms, 2.13 NTL GF2X |

`inv_series(n)` modulo a word (NTL's `InvTrunc`, PARI's `Flxn_inv`):

| length, terms | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| ---: | --- | --- | --- | --- |
| 2 | 49.4 ns, 2.71 NTL | 112 ns, 1.49 NTL | 133 ns, 1.27 FLINT | 142 ns, 1.30 FLINT |
| 4 | 60.9 ns, 1.68 NTL | 127 ns, 1.51 NTL | 158 ns, 1.21 FLINT | 157 ns, 1.16 FLINT |
| 8 | 102 ns, 1.19 FLINT | 197 ns, 1.36 NTL | 221 ns, 1.02 FLINT | 230 ns, 0.998 FLINT |
| 16 | 197 ns, 1.01 NTL | 314 ns, 1.29 NTL | 387 ns, 0.953 FLINT | 348 ns, 0.856 FLINT |
| 64 | 818 ns, 0.671 FLINT | 1.20 µs, 0.771 FLINT | 1.48 µs, 0.763 FLINT | 1.32 µs, 0.678 FLINT |
| 256 | 4.74 µs, 0.672 FLINT | 8.57 µs, 0.714 FLINT | 11.0 µs, 0.625 FLINT | 10.4 µs, 0.605 FLINT |
| 1000 | 21.6 µs, 0.816 FLINT | 39.4 µs, 0.508 FLINT | 50.4 µs, 0.596 FLINT | 50.2 µs, 0.591 FLINT |
| 5000 | 160 µs, 1.04 FLINT | 291 µs, 0.578 FLINT | 372 µs, 0.721 FLINT | 370 µs, 0.720 FLINT |
| 10,000 | 331 µs, 1.01 FLINT | 607 µs, 0.552 FLINT | 768 µs, 0.681 FLINT | 769 µs, 0.688 FLINT |
| 100,000 | 3.65 ms, 0.880 FLINT | 8.50 ms, 0.604 FLINT | 8.80 ms, 0.623 FLINT | 10.1 ms, 0.719 FLINT |

`mul_mod(&b, &f)` modulo a word, `f` of `len(f)` coefficients (no precomputed inverse on any side):

| `len(f)` | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| ---: | --- | --- | --- | --- |
| 4 | 92.7 ns, 1.96 PARI | 164 ns, 1.54 PARI | 194 ns, 1.58 PARI | 205 ns, 1.65 PARI |
| 16 | 275 ns, 1.11 FLINT | 579 ns, 1.25 NTL | 753 ns, 1.19 FLINT | 656 ns, 1.06 FLINT |
| 64 | 1.44 µs, 0.581 FLINT | 3.93 µs, 1.15 FLINT | 4.83 µs, 1.25 FLINT | 4.32 µs, 1.13 FLINT |
| 256 | 15.0 µs, 0.851 FLINT | 33.9 µs, 1.11 FLINT | 38.0 µs, 1.08 FLINT | 38.6 µs, 1.09 FLINT |
| 1000 | 62.4 µs, 1.07 FLINT | 105 µs, 0.581 FLINT | 130 µs, 0.696 FLINT | 131 µs, 0.695 FLINT |
| 5000 | 417 µs, 1.23 FLINT | 725 µs, 0.604 FLINT | 900 µs, 0.746 FLINT | 900 µs, 0.749 FLINT |
| 10,000 | 852 µs, 1.18 FLINT | 1.62 ms, 0.639 FLINT | 1.84 ms, 0.727 FLINT | 1.85 ms, 0.724 FLINT |
| 100,000 | 9.56 ms, 1.08 FLINT | 21.2 ms, 0.702 FLINT | 24.1 ms, 0.790 FLINT | 25.5 ms, 0.834 FLINT |

`x.pow_mod(&e, &f)` modulo a word for `e = 2⁶³ − 25` (FLINT's `nmod_poly_powmod_x_fmpz_preinv` and `nmod_poly_powmod_fmpz_binexp_preinv`, NTL's `PowerXMod`, PARI's `Flxq_pow`):

| `len(f)` | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` | `2` |
| ---: | --- | --- | --- | --- | --- |
| 4 | 2.94 µs, 0.777 PARI (1.17 FLINT†) | 4.31 µs, 0.498 PARI (1.20 FLINT†) | 4.48 µs, 0.442 PARI (1.23 FLINT†) | 4.21 µs, 0.419 PARI (1.11 FLINT†) |  |
| 16 | 14.4 µs, 0.754 PARI (0.931 FLINT†) | 27.3 µs, 0.309 PARI (1.13 FLINT†) | 29.1 µs, 0.302 PARI (0.909 FLINT†) | 26.6 µs, 0.273 PARI (0.828 FLINT†) |  |
| 64 | 80.9 µs, 0.563 PARI (0.870 FLINT†) | 201 µs, 0.431 PARI (1.05 FLINT†) | 244 µs, 0.383 PARI (0.933 FLINT†) | 218 µs, 0.338 PARI (0.805 FLINT†) | 4.42 µs, 0.427 PARI F2x (2.33 NTL GF2X†) |
| 256 | 432 µs, 0.556 PARI (0.984 FLINT†) | 738 µs, 0.247 PARI (0.549 NTL†) | 942 µs, 0.241 PARI (0.474 FLINT†) | 939 µs, 0.239 PARI (0.495 FLINT†) |  |
| 1000 | 1.64 ms, 0.319 PARI (0.845 FLINT†) | 2.83 ms, 0.146 PARI (0.476 NTL†) | 3.57 ms, 0.134 PARI (0.462 FLINT†) | 3.56 ms, 0.132 PARI (0.464 FLINT†) | 17.2 µs, 0.096 PARI F2x (1.64 NTL GF2X†) |
| 5000 | 10.4 ms, 0.240 PARI (0.805 FLINT†) | 18.4 ms, 0.155 PARI (0.457 NTL†) | 23.1 ms, 0.164 PARI (0.496 FLINT†) | 23.1 ms, 0.164 PARI (0.492 FLINT†) |  |
| 10,000 | 20.7 ms, 0.227 PARI (0.778 FLINT†) | 42.8 ms, 0.188 PARI (0.503 NTL†) | 46.2 ms, 0.159 PARI (0.485 FLINT†) | 46.1 ms, 0.160 PARI (0.483 FLINT†) | 704 µs, 0.084 PARI F2x (1.70 NTL GF2X†) |
| 30,000 | 57.4 ms, 0.205 PARI (0.716 FLINT†) | 126 ms, 0.146 PARI (0.529 NTL†) | 129 ms, 0.105 PARI (0.448 FLINT†) | 154 ms, 0.116 PARI (0.535 FLINT†) |  |

`a.pow_mod(&e, &f)` modulo a word for an `e` of 1024 bits:

| `len(f)` | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| ---: | --- | --- | --- | --- |
| 4 | 38.9 µs, 0.739 PARI (1.04 FLINT†) | 67.9 µs, 0.538 PARI (1.13 FLINT†) | 71.0 µs, 0.480 PARI (1.19 FLINT†) | 61.7 µs, 0.417 PARI (0.860 FLINT†) |
| 16 | 253 µs, 0.766 PARI | 512 µs, 0.320 PARI (0.961 FLINT†) | 543 µs, 0.308 PARI (0.758 FLINT†) | 495 µs, 0.276 PARI (0.709 FLINT†) |
| 64 | 1.56 ms, 0.532 PARI (0.752 FLINT†) | 4.16 ms, 0.449 PARI (0.940 FLINT†) | 5.07 ms, 0.399 PARI (0.823 FLINT†) | 4.56 ms, 0.353 PARI (0.731 FLINT†) |
| 256 | 8.73 ms, 0.532 PARI (0.825 FLINT†) | 15.6 ms, 0.244 PARI (0.471 NTL†) | 19.7 ms, 0.237 PARI (0.437 FLINT†) | 19.7 ms, 0.237 PARI (0.439 FLINT†) |
| 1000 | 34.8 ms, 0.304 PARI (0.736 FLINT†) | 61.4 ms, 0.144 PARI (0.416 NTL†) | 77.8 ms, 0.132 PARI (0.404 FLINT†) | 77.7 ms, 0.130 PARI (0.412 FLINT†) |
| 2000 | 70.8 ms, 0.225 PARI (0.686 FLINT†) | 126 ms, 0.123 PARI (0.411 NTL†) | 159 ms, 0.141 PARI (0.396 FLINT†) | 161 ms, 0.142 PARI (0.403 FLINT†) |

`compose(&g)` modulo a word, both of the length:

| length | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| ---: | --- | --- | --- | --- |
| 2 | 22.1 ns, 2.12 FLINT | 22.3 ns, 2.16 FLINT | 22.7 ns, 2.24 FLINT | 22.6 ns, 2.39 FLINT |
| 4 | 103 ns, 1.63 FLINT | 118 ns, 1.49 FLINT | 118 ns, 1.49 FLINT | 124 ns, 1.28 FLINT |
| 8 | 698 ns, 1.41 FLINT | 1.55 µs, 1.66 FLINT | 1.88 µs, 1.60 FLINT | 1.64 µs, 1.39 FLINT |
| 16 | 3.54 µs, 0.986 FLINT | 11.0 µs, 1.21 FLINT | 13.5 µs, 1.10 FLINT | 12.2 µs, 0.981 FLINT |
| 64 | 158 µs, 1.24 FLINT | 345 µs, 0.685 FLINT | 494 µs, 0.940 FLINT | 466 µs, 0.872 FLINT |
| 256 | 4.21 ms, 1.30 FLINT | 8.04 ms, 0.620 FLINT | 9.52 ms, 0.732 FLINT | 9.73 ms, 0.746 FLINT |
| 1000 | 90.6 ms, 1.03 FLINT | 191 ms, 0.649 FLINT | 224 ms, 0.757 FLINT | 232 ms, 0.786 FLINT |
| 2000 | 525 ms, 0.990 FLINT | 1.23 s, 0.885 FLINT | 1.40 s, 0.999 FLINT | 1.48 s, 1.07 FLINT |

`compose_mod(&g, &f)` modulo a word, all of `len(f)` coefficients (FLINT's `nmod_poly_compose_mod_brent_kung_preinv`, NTL's `CompMod`, PARI's `Flx_Flxq_eval`):

| `len(f)` | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| ---: | --- | --- | --- | --- |
| 4 | 217 ns, 1.39 PARI (1.52 FLINT†) | 315 ns, 1.35 PARI (1.74 FLINT†) | 348 ns, 1.38 PARI (1.94 FLINT†) | 330 ns, 1.31 PARI (1.66 FLINT†) |
| 16 | 1.71 µs, 0.824 PARI (1.02 FLINT†) | 3.12 µs, 0.364 PARI (1.28 FLINT†) | 3.27 µs, 0.337 PARI (1.04 FLINT†) | 3.10 µs, 0.314 PARI (0.945 FLINT†) |
| 64 | 21.3 µs, 0.507 PARI (1.00 FLINT†) | 56.9 µs, 0.473 PARI (1.32 FLINT†) | 71.0 µs, 0.430 PARI (1.18 FLINT†) | 64.4 µs, 0.386 PARI (1.05 FLINT†) |
| 256 | 270 µs, 0.578 PARI (1.19 FLINT†) | 464 µs, 0.270 PARI (0.886 NTL†) | 598 µs, 0.267 PARI (0.655 FLINT†) | 591 µs, 0.262 PARI (0.642 FLINT†) |
| 1000 | 2.26 ms, 0.343 PARI (1.13 FLINT†) | 3.90 ms, 0.165 PARI (0.801 NTL†) | 4.98 ms, 0.152 PARI (0.656 FLINT†) | 4.91 ms, 0.149 PARI (0.647 FLINT†) |
| 5000 | 35.1 ms, 0.258 PARI (1.16 FLINT†) | 62.4 ms, 0.173 PARI (0.809 NTL†) | 78.7 ms, 0.184 PARI (0.776 FLINT†) | 77.8 ms, 0.182 PARI (0.770 FLINT†) |

`pow(7)` modulo a word:

| length | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| ---: | --- | --- | --- | --- |
| 2 | 129 ns, 1.67 PARI | 153 ns, 1.55 FLINT | 154 ns, 1.55 FLINT | 174 ns, 1.78 FLINT |
| 16 | 629 ns, 0.691 FLINT | 2.28 µs, 1.08 FLINT | 2.77 µs, 0.976 FLINT | 2.41 µs, 0.838 FLINT |
| 128 | 12.4 µs, 1.07 FLINT | 22.2 µs, 0.519 FLINT | 27.9 µs, 0.627 FLINT | 27.9 µs, 0.626 FLINT |
| 1000 | 107 µs, 1.07 FLINT | 192 µs, 0.461 FLINT | 239 µs, 0.575 FLINT | 238 µs, 0.577 FLINT |
| 10,000 | 1.23 ms, 0.992 FLINT | 2.65 ms, 0.524 FLINT | 2.76 ms, 0.545 FLINT | 3.08 ms, 0.606 FLINT |

The linear operations modulo a word: `eval` at one point (`a`'s takes and returns a `Number`), `scale`, `derivative` and `+`:

| operation, length | `65,521` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| --- | --- | --- | --- | --- |
| `eval`, 16 | 38.9 ns, 2.18 FLINT | 43.2 ns, 2.37 FLINT | 41.7 ns, 1.37 FLINT | 78.4 ns, 1.79 FLINT |
| `eval`, 1000 | 457 ns, 1.09 FLINT | 609 ns, 1.23 FLINT | 730 ns, 1.05 FLINT | 658 ns, 0.876 FLINT |
| `eval`, 100,000 | 28.8 µs, 0.899 FLINT | 44.1 µs, 1.46 FLINT | 55.0 µs, 1.36 FLINT | 48.2 µs, 1.17 FLINT |
| `scale`, 16 | 19.1 ns, 2.05 PARI | 21.4 ns, 2.13 NTL | 19.8 ns, 1.03 FLINT | 27.1 ns, 1.87 FLINT |
| `scale`, 1000 | 311 ns, 0.890 NTL | 367 ns, 1.05 NTL | 277 ns, 0.646 FLINT | 677 ns, 1.61 FLINT |
| `scale`, 100,000 | 25.0 µs, 0.715 NTL | 25.1 µs, 0.722 NTL | 25.4 µs, 0.638 FLINT | 67.0 µs, 1.70 FLINT |
| `derivative`, 16 | 22.8 ns, 2.51 PARI | 22.6 ns, 2.35 PARI | 27.7 ns, 1.06 PARI | 27.4 ns, 1.02 PARI |
| `derivative`, 1000 | 914 ns, 1.91 PARI | 909 ns, 1.85 PARI | 1.33 µs, 0.598 PARI | 1.33 µs, 0.709 PARI |
| `derivative`, 100,000 | 118 µs, 2.43 PARI | 140 µs, 1.03 NTL | 143 µs, 0.633 PARI | 141 µs, 0.750 PARI |
| `+`, 16 | 16.2 ns, 1.86 NTL | 16.2 ns, 1.82 NTL | 16.0 ns, 1.55 FLINT | 15.9 ns, 1.60 FLINT |
| `+`, 1000 | 203 ns, 0.656 NTL | 203 ns, 0.658 NTL | 201 ns, 0.580 FLINT | 220 ns, 0.641 FLINT |
| `+`, 100,000 | 19.7 µs, 0.636 NTL | 19.6 µs, 0.634 NTL | 19.8 µs, 0.572 FLINT | 19.8 µs, 0.580 FLINT |

Products over the integers, coefficients of the bits given with random signs:

| length | 32 bits | 64 bits | 1,000 bits | 10,000 bits |
| ---: | --- | --- | --- | --- |
| 1 | 40.7 ns, 5.06 FLINT | 39.0 ns, 4.49 FLINT | 130 ns, 2.05 FLINT | 3.38 µs, 1.25 FLINT |
| 2 | 101 ns, 4.37 FLINT | 124 ns, 3.41 FLINT | 775 ns, 2.98 FLINT | 15.3 µs, 1.73 FLINT |
| 4 | 185 ns, 4.50 FLINT | 212 ns, 1.62 FLINT | 2.03 µs, 2.04 NTL | 40.9 µs, 1.47 FLINT |
| 16 | 974 ns, 4.39 FLINT | 1.27 µs, 1.22 PARI | 16.1 µs, 1.60 NTL | 213 µs, 0.689 NTL |
| 64 | 4.49 µs, 2.49 FLINT | 6.00 µs, 0.942 FLINT | 71.3 µs, 1.29 FLINT | 906 µs, 0.610 FLINT |
| 128 | 8.19 µs, 2.23 FLINT | 10.9 µs, 1.60 FLINT | 156 µs, 0.570 FLINT | 1.80 ms, 0.595 FLINT |
| 256 | 16.2 µs, 2.20 FLINT | 21.5 µs, 1.59 FLINT | 311 µs, 0.624 NTL | 3.94 ms, 0.632 FLINT |
| 1000 | 67.0 µs, 1.99 FLINT | 85.6 µs, 1.38 FLINT | 1.51 ms, 0.662 NTL | 16.8 ms, 0.629 FLINT |
| 5000 | 386 µs, 1.83 FLINT | 501 µs, 1.28 FLINT | 8.16 ms, 0.678 NTL | 141 ms, 1.06 FLINT |
| 10,000 | 827 µs, 1.85 FLINT | 1.19 ms, 1.39 FLINT | 17.3 ms, 0.711 NTL | 281 ms, 1.04 FLINT |
| 30,000 | 2.19 ms, 1.45 FLINT | 3.01 ms, 1.05 FLINT | 59.4 ms, 0.797 NTL | 1.46 s, 1.63 FLINT |
| 100,000 | 7.90 ms, 1.34 FLINT | 11.5 ms, 1.04 FLINT | 285 ms, 1.03 FLINT |  |

Squares over the integers:

| length | 32 bits | 64 bits | 1,000 bits | 10,000 bits |
| ---: | --- | --- | --- | --- |
| 1 | 42.0 ns, 6.15 FLINT | 38.9 ns, 5.37 FLINT | 115 ns, 1.70 PARI | 2.62 µs, 0.947 PARI |
| 2 | 89.8 ns, 4.71 FLINT | 101 ns, 5.22 FLINT | 581 ns, 2.80 FLINT | 11.5 µs, 1.29 FLINT |
| 4 | 158 ns, 4.77 FLINT | 198 ns, 2.28 FLINT | 1.79 µs, 2.47 FLINT | 30.3 µs, 1.09 FLINT |
| 16 | 810 ns, 5.22 FLINT | 1.03 µs, 1.18 PARI | 12.2 µs, 1.52 NTL | 156 µs, 0.593 FLINT |
| 64 | 3.91 µs, 3.33 FLINT | 5.33 µs, 0.908 FLINT | 53.4 µs, 1.16 FLINT | 665 µs, 0.571 FLINT |
| 128 | 7.15 µs, 1.87 FLINT | 9.28 µs, 1.96 FLINT | 115 µs, 0.527 FLINT | 1.34 ms, 0.567 FLINT |
| 256 | 14.2 µs, 2.50 FLINT | 17.9 µs, 1.86 FLINT | 232 µs, 0.667 NTL | 2.90 ms, 0.592 FLINT |
| 1000 | 58.2 µs, 2.26 FLINT | 72.9 µs, 1.73 FLINT | 1.12 ms, 0.694 NTL | 12.4 ms, 0.597 FLINT |
| 5000 | 326 µs, 2.01 FLINT | 415 µs, 1.46 FLINT | 6.07 ms, 0.667 NTL | 104 ms, 1.10 FLINT |
| 10,000 | 694 µs, 2.04 FLINT | 974 µs, 1.55 FLINT | 12.7 ms, 0.682 NTL | 210 ms, 1.10 FLINT |
| 30,000 | 1.87 ms, 1.77 FLINT | 2.54 ms, 1.27 FLINT | 43.3 ms, 0.771 NTL | 1.10 s, 1.61 FLINT |
| 100,000 | 6.66 ms, 1.58 FLINT | 9.53 ms, 1.22 FLINT | 215 ms, 1.09 FLINT |  |

`pseudo_div_rem` over the integers, `2n` by `n` coefficients:

| lengths | 64 bits | 1,000 bits |
| ---: | --- | --- |
| 4 / 2 | 277 ns, 1.36 PARI | 1.96 µs, 0.888 PARI |
| 8 / 4 | 1.08 µs, 1.74 PARI | 10.0 µs, 0.739 PARI |
| 16 / 8 | 2.88 µs, 1.32 PARI | 57.9 µs, 0.662 NTL |
| 64 / 32 | 39.0 µs, 0.982 PARI (1.07 FLINT†) | 2.84 ms, 0.727 NTL |
| 256 / 128 | 1.23 ms, 0.943 PARI | 49.4 ms, 0.225 NTL (0.503 FLINT†) |
| 1024 / 512 | 52.8 ms, 0.758 PARI | 1.69 s, 0.125 NTL (0.752 FLINT†) |
| 4000 / 2000 | 1.87 s, 0.438 NTL (1.04 FLINT†) |  |

`div_exact` over the integers of a product by a factor of `n` coefficients (FLINT's `fmpz_poly_divides`, NTL's `divide`):

| lengths | 64 bits | 1,000 bits |
| ---: | --- | --- |
| 3 / 2 | 189 ns, 1.04 FLINT | 751 ns, 0.656 FLINT |
| 7 / 4 | 507 ns, 1.11 FLINT | 2.83 µs, 1.08 FLINT |
| 15 / 8 | 1.16 µs, 0.828 FLINT | 8.24 µs, 1.12 FLINT |
| 63 / 32 | 6.85 µs, 0.842 NTL | 96.1 µs, 1.33 FLINT |
| 255 / 128 | 34.8 µs, 0.688 NTL | 518 µs, 0.990 FLINT |
| 1023 / 512 | 161 µs, 0.822 NTL | 2.21 ms, 1.10 NTL |
| 3999 / 2000 | 701 µs, 0.869 NTL |  |

The other operations over the integers:

| operation, length | 64-bit coefficients |
| --- | --- |
| `compose`, 4 | 1.49 µs, 2.76 FLINT |
| `compose`, 16 | 209 µs, 0.919 FLINT |
| `compose`, 64 | 15.1 ms, 0.657 FLINT |
| `pow(7)`, 4 | 4.17 µs, 2.09 FLINT |
| `pow(7)`, 16 | 27.2 µs, 0.928 FLINT |
| `pow(7)`, 128 | 259 µs, 0.529 FLINT |
| `eval` at a 64-bit point, 16 | 474 ns, 2.13 FLINT |
| `eval` at a 64-bit point, 256 | 13.0 µs, 1.29 FLINT |
| `eval` at a 64-bit point, 4096 | 471 µs, 0.765 FLINT |
| `+`, 16 | 216 ns, 2.97 NTL |
| `+`, 1000 | 11.5 µs, 2.36 NTL |
| `+`, 100,000 | 1.42 ms, 1.57 NTL |
| `derivative`, 16 | 230 ns, 4.69 NTL |
| `derivative`, 1000 | 14.5 µs, 5.25 NTL |
| `derivative`, 100,000 | 1.71 ms, 5.37 NTL |
| `content` (a 32-bit common factor), 16 |  |
| `content` (a 32-bit common factor), 1000 |  |
| `content` (a 32-bit common factor), 100,000 |  |

Products modulo `n` of several limbs:

| length | `2¹²⁷ − 1` | `2²⁵⁵ − 19` | `2⁵²¹ − 1` | `2¹⁰²⁴ − 105` |
| ---: | --- | --- | --- | --- |
| 1 | 32.8 ns, 0.986 FLINT | 69.4 ns, 1.16 FLINT | 175 ns, 1.33 FLINT | 332 ns, 1.23 FLINT |
| 2 | 63.7 ns, 0.662 NTL | 178 ns, 1.01 FLINT | 471 ns, 1.14 FLINT | 975 ns, 1.09 FLINT |
| 4 | 153 ns, 0.551 FLINT | 460 ns, 1.00 FLINT | 1.23 µs, 1.09 FLINT | 2.67 µs, 1.03 FLINT |
| 8 | 514 ns, 0.576 PARI | 1.52 µs, 0.851 PARI | 3.89 µs, 1.06 NTL | 9.36 µs, 1.41 FLINT |
| 16 | 1.25 µs, 0.558 FLINT | 3.76 µs, 0.837 FLINT | 10.1 µs, 1.12 NTL | 20.0 µs, 1.02 NTL |
| 64 | 7.75 µs, 0.593 FLINT | 19.8 µs, 0.682 FLINT | 45.3 µs, 0.911 FLINT | 87.5 µs, 0.947 FLINT |
| 128 | 14.9 µs, 0.598 FLINT | 40.2 µs, 0.506 PARI | 91.4 µs, 0.553 FLINT | 189 µs, 0.569 FLINT |
| 256 | 31.1 µs, 0.863 NTL | 82.8 µs, 1.15 NTL | 187 µs, 1.25 NTL | 419 µs, 1.22 NTL |
| 1000 | 128 µs, 0.812 NTL | 336 µs, 1.06 NTL | 761 µs, 1.18 NTL | 1.77 ms, 1.19 NTL |
| 5000 | 965 µs, 1.04 NTL | 2.13 ms, 1.13 NTL | 4.68 ms, 1.24 NTL | 9.41 ms, 1.09 NTL |
| 10,000 | 1.93 ms, 0.994 NTL | 4.62 ms, 1.17 NTL | 9.33 ms, 1.19 NTL | 20.0 ms, 1.13 NTL |
| 30,000 | 6.70 ms, 1.11 NTL | 15.6 ms, 1.28 NTL | 31.7 ms, 1.30 NTL | 67.8 ms, 1.26 NTL |
| 100,000 | 21.0 ms, 0.859 NTL | 46.4 ms, 0.932 NTL | 156 ms, 1.64 NTL | 312 ms, 1.43 NTL |

Squares modulo `n` of several limbs:

| length | `2¹²⁷ − 1` | `2²⁵⁵ − 19` | `2⁵²¹ − 1` | `2¹⁰²⁴ − 105` |
| ---: | --- | --- | --- | --- |
| 1 | 30.0 ns, 0.945 FLINT | 69.3 ns, 1.24 FLINT | 171 ns, 1.33 FLINT | 325 ns, 1.19 PARI |
| 2 | 62.9 ns, 0.718 FLINT | 178 ns, 1.10 FLINT | 472 ns, 1.28 FLINT | 970 ns, 1.16 FLINT |
| 4 | 153 ns, 0.631 NTL | 455 ns, 1.12 FLINT | 1.22 µs, 1.29 FLINT | 2.69 µs, 1.21 FLINT |
| 8 | 391 ns, 0.545 NTL | 1.26 µs, 0.953 NTL | 3.27 µs, 1.04 NTL | 7.71 µs, 1.20 FLINT |
| 16 | 983 ns, 0.517 PARI | 3.09 µs, 0.742 PARI | 8.12 µs, 1.05 NTL | 17.2 µs, 1.05 NTL |
| 64 | 6.98 µs, 0.558 PARI | 18.3 µs, 0.717 FLINT | 42.0 µs, 1.04 FLINT | 73.7 µs, 0.920 FLINT |
| 128 | 13.6 µs, 1.04 NTL | 37.0 µs, 1.43 NTL | 84.9 µs, 1.58 NTL | 153 µs, 1.26 NTL |
| 256 | 27.5 µs, 0.985 NTL | 76.4 µs, 1.42 NTL | 172 µs, 1.56 NTL | 346 µs, 1.37 NTL |
| 1000 | 111 µs, 0.928 NTL | 306 µs, 1.29 NTL | 691 µs, 1.46 NTL | 1.42 ms, 1.31 NTL |
| 5000 | 755 µs, 1.07 NTL | 1.72 ms, 1.21 NTL | 3.73 ms, 1.34 NTL | 7.59 ms, 1.21 NTL |
| 10,000 | 1.50 ms, 1.02 NTL | 3.66 ms, 1.25 NTL | 7.49 ms, 1.30 NTL | 15.9 ms, 1.23 NTL |
| 30,000 | 5.13 ms, 1.15 NTL | 12.2 ms, 1.35 NTL | 25.1 ms, 1.42 NTL | 52.6 ms, 1.35 NTL |
| 100,000 | 16.1 ms, 0.903 NTL | 36.8 ms, 1.04 NTL | 122 ms, 1.75 NTL | 245 ms, 1.58 NTL |

`div_rem` modulo `n` of several limbs:

| lengths | `2¹²⁷ − 1` | `2²⁵⁵ − 19` | `2⁵²¹ − 1` | `2¹⁰²⁴ − 105` |
| ---: | --- | --- | --- | --- |
| 8 / 4 | 703 ns, 0.835 PARI | 1.44 µs, 0.914 NTL | 3.12 µs, 0.955 FLINT | 6.30 µs, 0.895 FLINT |
| 32 / 16 | 2.33 µs, 0.500 NTL | 5.24 µs, 0.772 FLINT | 13.7 µs, 0.981 FLINT | 32.3 µs, 0.930 FLINT |
| 128 / 64 | 21.7 µs, 0.683 FLINT | 45.5 µs, 0.847 FLINT | 187 µs, 1.73 FLINT | 420 µs, 1.54 FLINT |
| 256 / 128 | 79.4 µs, 1.02 FLINT | 162 µs, 0.992 NTL | 359 µs, 1.11 NTL | 883 µs, 1.22 NTL |
| 512 / 256 | 134 µs, 0.824 FLINT | 327 µs, 0.984 NTL | 718 µs, 1.09 NTL | 1.86 ms, 1.26 NTL |
| 1024 / 512 | 271 µs, 0.850 FLINT | 665 µs, 0.948 NTL | 1.45 ms, 1.05 NTL | 3.85 ms, 1.25 NTL |
| 2000 / 1000 | 489 µs, 1.01 NTL | 1.21 ms, 1.29 NTL | 2.64 ms, 1.38 NTL | 7.55 ms, 1.72 NTL |
| 4000 / 2000 | 986 µs, 0.970 NTL | 2.44 ms, 1.24 NTL | 5.38 ms, 1.35 NTL | 16.1 ms, 1.77 NTL |

`x.pow_mod(&n, &f)` modulo `n` of several limbs (FLINT's `fmpz_mod_poly_powmod_x_fmpz_preinv`, NTL's `PowerXMod`, PARI's `FpXQ_pow`):

| `len(f)` | `2¹²⁷ − 1` | `2²⁵⁵ − 19` | `2⁵²¹ − 1` | `2¹⁰²⁴ − 105` |
| ---: | --- | --- | --- | --- |
| 4 | 68.5 µs, 0.542 PARI (1.03 FLINT†) | 302 µs, 0.630 PARI (1.27 FLINT†) | 1.38 ms, 0.689 PARI (1.32 FLINT†) | 5.46 ms, 0.672 PARI (1.24 FLINT†) |
| 8 | 156 µs, 0.482 PARI (0.686 FLINT†) | 773 µs, 0.668 PARI (0.970 FLINT†) | 4.08 ms, 0.814 PARI (1.09 FLINT†) | 17.0 ms, 0.767 PARI (0.968 FLINT†) |
| 16 | 375 µs, 0.399 PARI (0.537 FLINT†) | 2.01 ms, 0.606 PARI (0.696 FLINT†) | 10.6 ms, 0.743 PARI (0.837 NTL†) | 47.1 ms, 0.703 PARI (0.870 FLINT†) |
| 32 | 1.09 ms, 0.405 PARI (1.07 NTL†) | 5.60 ms, 0.538 PARI (1.36 NTL†) | 29.1 ms, 0.639 PARI (1.68 NTL†) | 135 ms, 0.603 PARI (1.76 NTL†) |
| 64 | 2.34 ms, 0.422 PARI (1.14 NTL†) | 12.0 ms, 0.507 PARI (1.47 NTL†) | 55.6 ms, 0.421 PARI (1.59 NTL†) | 269 ms, 0.440 PARI (1.72 NTL†) |
| 128 | 4.45 ms, 0.326 PARI (1.08 NTL†) | 24.0 ms, 0.395 PARI (1.43 NTL†) | 110 ms, 0.370 PARI (1.53 NTL†) | 575 ms, 0.464 PARI (1.81 NTL†) |
| 256 | 8.91 ms, 0.265 PARI (1.04 NTL†) | 48.3 ms, 0.323 PARI (1.40 NTL†) | 223 ms, 0.390 PARI (1.53 NTL†) | 1.17 s, 0.438 PARI (1.79 NTL†) |
| 512 | 17.7 ms, 0.218 PARI (1.00 NTL†) | 97.2 ms, 0.314 PARI (1.37 NTL†) | 449 ms, 0.346 PARI (1.48 NTL†) | 2.59 s, 0.481 PARI (1.92 NTL†) |

`inv_series`, `mul_mod` and `compose_mod` modulo `n` of several limbs:

| operation, length | `2¹²⁷ − 1` | `2²⁵⁵ − 19` | `2⁵²¹ − 1` | `2¹⁰²⁴ − 105` |
| --- | --- | --- | --- | --- |
| `inv_series`, 16 | 2.96 µs, 2.07 FLINT | 8.03 µs, 2.71 FLINT | 18.3 µs, 2.80 FLINT | 37.2 µs, 2.25 FLINT |
| `inv_series`, 256 | 69.1 µs, 1.30 NTL | 175 µs, 1.69 NTL | 388 µs, 1.84 NTL | 944 µs, 1.91 NTL |
| `inv_series`, 2000 | 573 µs, 1.21 NTL | 1.43 ms, 1.53 NTL | 3.16 ms, 1.67 NTL | 8.28 ms, 1.93 NTL |
| `mul_mod`, 16 | 3.18 µs, 0.522 PARI | 8.19 µs, 0.807 FLINT | 20.9 µs, 0.947 FLINT | 46.9 µs, 0.944 FLINT |
| `mul_mod`, 256 | 151 µs, 0.995 NTL | 378 µs, 1.31 NTL | 838 µs, 1.41 NTL | 2.09 ms, 1.52 NTL |
| `mul_mod`, 2000 | 1.25 ms, 0.923 NTL | 3.12 ms, 1.18 NTL | 6.97 ms, 1.31 NTL | 19.8 ms, 1.63 NTL |
| `compose_mod`, 16 | 19.4 µs, 0.462 PARI (0.566 FLINT†) | 50.4 µs, 0.649 PARI (0.734 FLINT†) | 130 µs, 0.754 PARI (0.856 NTL†) | 291 µs, 0.715 PARI (0.873 FLINT†) |
| `compose_mod`, 256 | 2.68 ms, 0.281 PARI (1.29 NTL†) | 6.72 ms, 0.331 PARI (1.81 NTL†) | 15.1 ms, 0.402 PARI (1.97 NTL†) | 41.5 ms, 0.474 PARI (2.19 NTL†) |
| `compose_mod`, 2000 | 82.9 ms, 0.234 PARI (1.91 NTL†) | 195 ms, 0.299 PARI (2.32 NTL†) | 460 ms, 0.325 PARI (2.67 NTL†) | 1.37 s, 0.432 PARI (3.54 NTL†) |

Over the rationals, numerators and denominators of 31 bits (`div_rem` and `inv_series` of 31-bit integers over one 31-bit denominator):

| operation, length | `Polynomial<Rational>` |
| --- | --- |
| `*`, 1 | 44.2 ns, 1.16 FLINT |
| `*`, 2 | 198 ns, 2.21 FLINT |
| `*`, 4 | 1.28 µs, 3.14 FLINT |
| `*`, 16 | 37.2 µs, 4.47 FLINT |
| `*`, 64 | 856 µs, 6.96 FLINT |
| `*`, 256 | 17.0 ms, 4.61 FLINT |
| `*`, 1000 | 464 ms, 6.67 FLINT |
| `*`, 2000 | 2.61 s, 9.09 FLINT |
| `div_rem`, 4 / 2 | 678 ns, 1.23 FLINT |
| `div_rem`, 16 / 8 | 27.9 µs, 9.29 FLINT |
| `div_rem`, 64 / 32 | 173 µs, 5.77 FLINT |
| `div_rem`, 256 / 128 | 2.86 ms, 3.74 FLINT |
| `div_rem`, 1024 / 512 | 80.4 ms, 2.48 FLINT |
| `inv_series`, 2 | 81.1 ns, 0.615 PARI |
| `inv_series`, 8 | 4.93 µs, 2.17 FLINT |
| `inv_series`, 32 | 71.5 µs, 3.07 FLINT |
| `inv_series`, 128 | 1.14 ms, 2.44 FLINT |
| `inv_series`, 512 | 23.2 ms, 2.11 FLINT |
| `compose`, 4 | 9.06 µs, 3.80 FLINT |
| `compose`, 16 | 4.63 ms, 2.17 FLINT |

Products of floats and of complex floats, against a schoolbook loop below the line where `a` takes the transforms and against realfft's and rustfft's convolutions (planned before the clock, as `a`'s plans are kept) above it:

| length | `Polynomial<f64>` | `Polynomial<Complex<f64>>` |
| ---: | --- | --- |
| 4 | 21.1 ns, 1.24 schoolbook | 21.3 ns, 1.03 schoolbook |
| 16 | 67.6 ns, 0.942 schoolbook | 135 ns, 1.04 schoolbook |
| 64 | 459 ns, 0.902 schoolbook (1.03 realfft†) | 1.52 µs, 0.856 schoolbook (1.94 rustfft†) |
| 128 | 1.64 µs, 0.948 schoolbook (1.84 realfft†) | 1.38 µs, 0.830 rustfft |
| 255 | 6.36 µs, 0.991 schoolbook (3.42 realfft†) | 2.61 µs, 0.748 rustfft |
| 256 | 6.40 µs, 0.981 schoolbook (3.43 realfft†) | 2.61 µs, 0.742 rustfft |
| 399 | 15.7 µs, 1.01 schoolbook (3.96 realfft†) | 4.87 µs, 0.655 rustfft |
| 400 | 15.6 µs, 1.00 schoolbook (3.94 realfft†) | 4.85 µs, 0.657 rustfft |
| 512 | 2.95 µs, 0.745 realfft | 6.20 µs, 0.829 rustfft |
| 1000 | 6.91 µs, 0.808 realfft | 14.3 µs, 0.846 rustfft |
| 2000 | 15.1 µs, 0.785 realfft | 28.8 µs, 0.738 rustfft |
| 5000 | 42.6 µs, 0.466 realfft | 79.7 µs, 0.371 rustfft |
| 10,000 | 89.2 µs, 0.397 realfft | 164 µs, 0.413 rustfft |
| 100,000 | 1.11 ms, 0.604 realfft | 2.27 ms, 0.546 rustfft |

### Greatest common divisors and resultants

Modulo a word prime:

| operation, degree | `2¹⁷ − 1` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| --- | --- | --- | --- | --- |
| `gcd`, 10 | 190 ns, 0.924 PARI | 264 ns, 0.507 PARI | 510 ns, 0.850 PARI | 553 ns, 0.911 PARI |
| `gcd`, 100 | 3.77 µs, 0.816 FLINT | 3.83 µs, 0.433 FLINT | 6.37 µs, 0.499 FLINT | 8.34 µs, 0.733 FLINT |
| `gcd`, 1000 | 288 µs, 1.20 FLINT | 300 µs, 0.892 FLINT | 314 µs, 0.484 FLINT | 482 µs, 0.978 FLINT |
| `gcd`, 3000 | 1.28 ms, 1.25 FLINT | 2.61 ms, 0.889 NTL | 2.66 ms, 0.775 FLINT | 2.91 ms, 0.817 FLINT |
| `gcd`, 10,000 | 5.54 ms, 1.27 FLINT | 10.1 ms, 0.755 NTL | 12.3 ms, 0.829 FLINT | 12.3 ms, 0.796 FLINT |
| `gcd`, 30,000 |  | 36.9 ms, 0.789 NTL |  |  |
| `gcd`, 100,000 |  | 156 ms, 0.717 NTL |  |  |
| `xgcd`, 10 | 582 ns, 1.26 PARI | 705 ns, 0.744 NTL | 795 ns, 0.751 PARI | 839 ns, 0.833 PARI |
| `xgcd`, 100 | 7.94 µs, 0.648 PARI | 9.25 µs, 0.539 NTL | 9.80 µs, 0.430 FLINT | 13.1 µs, 0.618 FLINT |
| `xgcd`, 1000 | 318 µs, 1.10 FLINT | 523 µs, 0.715 NTL | 540 µs, 0.597 FLINT | 745 µs, 0.797 FLINT |
| `xgcd`, 3000 | 1.37 ms, 1.14 FLINT | 2.59 ms, 0.812 NTL | 3.10 ms, 0.768 FLINT | 3.12 ms, 0.748 FLINT |
| `xgcd`, 10,000 | 5.78 ms, 1.17 FLINT | 10.7 ms, 0.726 NTL | 13.0 ms, 0.764 FLINT | 13.0 ms, 0.739 FLINT |
| `xgcd`, 30,000 |  | 38.6 ms, 0.779 NTL |  |  |
| `resultant`, 10 | 384 ns, 1.08 PARI | 397 ns, 0.456 PARI | 768 ns, 0.767 PARI | 800 ns, 0.783 PARI |
| `resultant`, 100 | 6.22 µs, 0.845 FLINT | 6.32 µs, 0.414 FLINT | 11.2 µs, 0.540 FLINT | 13.6 µs, 0.718 FLINT |
| `resultant`, 1000 | 401 µs, 1.16 FLINT | 416 µs, 0.844 FLINT | 491 µs, 0.538 FLINT | 752 µs, 1.07 FLINT |
| `resultant`, 3000 | 2.08 ms, 1.19 FLINT | 3.60 ms, 0.791 NTL | 3.77 ms, 0.685 FLINT | 4.48 ms, 0.887 FLINT |
| `resultant`, 10,000 | 9.58 ms, 1.27 FLINT | 17.1 ms, 0.765 FLINT | 20.3 ms, 0.809 FLINT | 21.2 ms, 0.820 FLINT |
| `resultant`, 30,000 |  | 65.2 ms, 0.804 NTL |  |  |
| `resultant`, 100,000 |  | 279 ms, 0.723 NTL |  |  |
| `discriminant`, 10 | 558 ns, 1.09 FLINT | 618 ns, 0.499 FLINT | 991 ns, 0.657 FLINT | 1.04 µs, 0.708 FLINT |
| `discriminant`, 100 | 6.47 µs, 0.840 FLINT | 6.68 µs, 0.434 FLINT | 11.6 µs, 0.549 FLINT | 14.0 µs, 0.730 FLINT |
| `discriminant`, 1000 | 403 µs, 1.15 FLINT | 413 µs, 0.833 FLINT | 474 µs, 0.524 FLINT | 705 µs, 0.998 FLINT |

Modulo primes of several limbs (`prev_prime(2^(64k))`):

| operation, degree | 2 limbs | 3 limbs | 8 limbs | 16 limbs |
| --- | --- | --- | --- | --- |
| `gcd`, 10 | 2.67 µs, 0.637 PARI | 4.75 µs, 0.780 PARI | 12.7 µs, 0.878 PARI | 28.9 µs, 0.868 PARI |
| `gcd`, 30 | 12.7 µs, 0.547 NTL | 24.7 µs, 0.792 FLINT | 70.2 µs, 0.947 FLINT | 170 µs, 0.904 FLINT |
| `gcd`, 100 | 91.3 µs, 0.481 NTL | 179 µs, 0.717 FLINT | 596 µs, 1.02 FLINT | 1.45 ms, 0.949 FLINT |
| `gcd`, 300 | 679 µs, 0.891 NTL | 1.29 ms, 1.24 NTL | 3.96 ms, 1.45 NTL | 8.84 ms, 1.29 NTL |
| `gcd`, 1000 | 3.40 ms, 1.09 NTL | 6.25 ms, 1.48 NTL | 19.3 ms, 1.71 NTL | 39.8 ms, 1.44 NTL |
| `gcd`, 3000 | 14.1 ms, 1.07 NTL |  |  |  |
| `gcd`, 10,000 | 59.6 ms, 1.21 NTL |  |  |  |
| `xgcd`, 10 | 4.85 µs, 0.730 PARI | 8.12 µs, 0.878 NTL | 21.0 µs, 0.961 NTL | 46.0 µs, 0.860 NTL |
| `xgcd`, 30 | 21.5 µs, 0.519 PARI | 40.7 µs, 0.722 NTL | 114 µs, 0.880 FLINT | 270 µs, 0.847 FLINT |
| `xgcd`, 100 | 146 µs, 0.677 NTL | 285 µs, 0.986 NTL | 875 µs, 1.15 NTL | 2.00 ms, 1.03 NTL |
| `xgcd`, 300 | 774 µs, 0.911 NTL | 1.47 ms, 1.27 NTL | 4.53 ms, 1.48 NTL | 9.60 ms, 1.26 NTL |
| `xgcd`, 1000 | 3.66 ms, 1.09 NTL | 6.80 ms, 1.50 NTL | 21.0 ms, 1.73 NTL | 43.2 ms, 1.44 NTL |
| `resultant`, 10 | 5.48 µs, 0.824 PARI | 9.54 µs, 0.947 PARI | 24.2 µs, 0.990 PARI | 54.8 µs, 0.938 PARI |
| `resultant`, 30 | 23.3 µs, 0.643 NTL | 42.7 µs, 0.875 NTL | 118 µs, 0.954 FLINT | 277 µs, 0.915 FLINT |
| `resultant`, 100 | 150 µs, 0.552 NTL | 284 µs, 0.766 NTL | 897 µs, 0.966 FLINT | 2.14 ms, 0.919 FLINT |
| `resultant`, 300 | 998 µs, 0.735 NTL | 1.91 ms, 1.04 NTL | 5.90 ms, 1.22 NTL | 13.5 ms, 1.10 NTL |
| `resultant`, 1000 | 5.60 ms, 1.01 NTL | 10.5 ms, 1.39 NTL | 32.2 ms, 1.62 NTL | 68.5 ms, 1.39 NTL |

`gcd` over the integers, a common factor of half the degree (its coefficients of half the bits) or coprime:

| degree | common factor, 64 bits | 1,000 bits | 10,000 bits | coprime, 64 bits | 1,000 bits | 10,000 bits |
| ---: | --- | --- | --- | --- | --- | --- |
| 5 | 2.33 µs, 1.59 FLINT | 17.9 µs, 0.924 FLINT | 442 µs, 0.969 NTL | 540 ns, 0.679 PARI | 2.86 µs, 0.365 PARI | 34.3 µs, 0.247 NTL |
| 10 | 3.28 µs, 1.00 FLINT | 23.1 µs, 0.814 FLINT | 526 µs, 0.891 NTL | 1.10 µs, 0.720 PARI | 3.39 µs, 0.381 PARI | 39.4 µs, 0.278 NTL |
| 30 | 6.59 µs, 0.576 FLINT | 61.8 µs, 0.867 FLINT | 1.09 ms, 0.553 NTL | 2.35 µs, 0.443 PARI | 5.92 µs, 0.457 NTL | 52.1 µs, 0.344 NTL |
| 100 | 22.9 µs, 0.525 FLINT | 227 µs, 0.755 FLINT | 3.28 ms, 0.511 NTL | 9.56 µs, 0.436 NTL | 17.6 µs, 0.580 NTL | 101 µs, 0.629 NTL |
| 300 | 110 µs, 0.589 FLINT | 911 µs, 0.664 NTL |  | 50.4 µs, 0.507 NTL | 72.0 µs, 0.652 NTL |  |
| 1000 | 936 µs, 0.921 NTL | 5.40 ms, 0.975 NTL |  | 452 µs, 0.799 NTL | 524 µs, 0.887 NTL |  |
| 3000 | 6.08 ms, 1.51 NTL |  |  |  |  |  |

Resultants and discriminants over the integers and resultants over the rationals:

| degree | `resultant` ℤ, 64 bits | 1,000 bits | `discriminant` ℤ, 64 bits | 1,000 bits | `resultant` ℚ, 64 bits | 1,000 bits |
| ---: | --- | --- | --- | --- | --- | --- |
| 5 | 2.98 µs, 2.26 FLINT | 47.5 µs, 0.731 FLINT | 3.30 µs, 2.18 FLINT | 49.4 µs, 0.744 FLINT | 5.69 µs, 2.00 FLINT | 73.5 µs, 0.819 FLINT |
| 10 | 14.7 µs, 1.55 FLINT | 438 µs, 0.561 FLINT (0.660 NTL†) | 14.9 µs, 1.53 FLINT | 451 µs, 0.674 NTL | 19.0 µs, 1.51 FLINT | 498 µs, 0.594 FLINT |
| 30 | 102 µs, 0.200 FLINT (0.316 NTL†) | 2.84 ms, 0.336 FLINT (0.486 NTL†) | 111 µs, 0.333 NTL | 2.83 ms, 0.486 NTL | 118 µs, 0.229 FLINT | 3.07 ms, 0.352 FLINT |
| 100 | 1.61 ms, 0.356 FLINT | 32.0 ms, 0.365 FLINT (0.411 NTL†) | 1.76 ms, 0.380 NTL | 32.0 ms, 0.413 NTL | 1.66 ms, 0.366 FLINT | 33.1 ms, 0.372 FLINT |
| 300 | 29.7 ms, 0.500 FLINT |  | 31.5 ms, 0.502 FLINT |  | 29.7 ms, 0.500 FLINT |  |

`gcd` and `xgcd` over the rationals with a common factor, and `lcm` over the integers:

| degree | `gcd` ℚ, 64 bits | 1,000 bits | `xgcd` ℚ, 64 bits | 1,000 bits |
| --- | --- | --- | --- | --- |
| 5 | 4.21 µs, 2.35 FLINT | 31.1 µs, 1.25 FLINT | 11.5 µs, 1.71 FLINT | 189 µs, 1.86 FLINT |
| 10 | 6.47 µs, 1.98 FLINT | 43.5 µs, 1.31 FLINT | 25.2 µs, 1.65 FLINT | 525 µs, 1.71 FLINT |
| 30 | 15.4 µs, 1.32 FLINT | 103 µs, 1.36 FLINT | 208 µs, 1.78 FLINT | 5.00 ms, 1.01 FLINT |
| 100 | 50.3 µs, 1.14 FLINT | 354 µs, 1.13 FLINT | 2.46 ms, 1.28 FLINT | 95.9 ms, 0.961 FLINT |
| 300 | 189 µs, 1.01 FLINT |  |  |  |
| `lcm` ℤ, 10, 64 bits | 4.77 µs, 0.973 FLINT |  |  |  |
| `lcm` ℤ, 10, 1000 bits | 33.8 µs, 0.846 FLINT |  |  |  |
| `lcm` ℤ, 10, 10000 bits | 654 µs, 0.492 FLINT |  |  |  |
| `lcm` ℤ, 100, 64 bits | 37.2 µs, 0.454 FLINT |  |  |  |
| `lcm` ℤ, 100, 1000 bits | 372 µs, 0.497 FLINT |  |  |  |
| `lcm` ℤ, 100, 10000 bits | 4.77 ms, 0.193 FLINT |  |  |  |

### Evaluation and interpolation

Modulo a word prime (`p20`, `p50`, `p63` and `p64` of 20, 50, 63 and 64 bits): `n` coefficients at `n` points, `n` roots or `n` points (FLINT's `nmod_poly_evaluate_nmod_vec` and its `_fast` form, `nmod_poly_product_roots_nmod_vec`, `nmod_poly_interpolate_nmod_vec_fast`, `_newton` and `_barycentric`; NTL's `eval`, `BuildFromRoots` and `interpolate`; PARI's `Flx_Flv_multieval`, `Flv_roots_to_pol` and `Flv_polint`):

| operation, n | `p20` | `p50` | `p63` | `p64` |
| --- | --- | --- | --- | --- |
| `eval_many_u64`, 1 | 13.0 ns, 3.32 NTL | 13.0 ns, 3.30 NTL | 13.5 ns, 2.79 FLINT | 13.9 ns, 2.83 FLINT |
| `eval_many_u64`, 4 | 35.1 ns, 1.54 NTL | 36.0 ns, 1.52 NTL | 37.2 ns, 1.42 FLINT | 38.1 ns, 1.47 FLINT |
| `eval_many_u64`, 16 | 98.9 ns, 0.424 FLINT | 96.2 ns, 0.351 FLINT | 195 ns, 0.412 FLINT | 196 ns, 0.289 FLINT |
| `eval_many_u64`, 256 | 8.93 µs, 0.212 FLINT | 17.6 µs, 0.352 FLINT | 44.7 µs, 0.753 FLINT | 45.8 µs, 0.770 FLINT |
| `eval_many_u64`, 4096 | 776 µs, 0.441 FLINT | 1.33 ms, 0.425 FLINT | 1.78 ms, 0.485 FLINT | 1.79 ms, 0.475 FLINT |
| `eval_many_u64`, 65,536 | 18.6 ms, 0.304 FLINT | 37.4 ms, 0.343 FLINT | 43.8 ms, 0.372 FLINT | 47.0 ms, 0.378 FLINT |
| `from_roots_u64`, 1 | 11.4 ns, 2.75 FLINT | 11.6 ns, 2.76 FLINT | 11.8 ns, 2.78 FLINT | 12.0 ns, 2.75 FLINT |
| `from_roots_u64`, 4 | 41.0 ns, 2.13 NTL | 42.0 ns, 2.49 NTL | 41.2 ns, 1.33 FLINT | 47.7 ns, 1.53 FLINT |
| `from_roots_u64`, 16 | 177 ns, 1.09 NTL | 182 ns, 1.05 NTL | 184 ns, 0.513 FLINT | 253 ns, 0.696 FLINT |
| `from_roots_u64`, 256 | 6.01 µs, 0.679 PARI | 12.7 µs, 0.829 FLINT | 15.3 µs, 0.656 FLINT | 16.5 µs, 0.692 FLINT |
| `from_roots_u64`, 4096 | 194 µs, 0.546 FLINT | 382 µs, 0.546 FLINT | 474 µs, 0.553 FLINT | 495 µs, 0.565 FLINT |
| `from_roots_u64`, 65,536 | 5.00 ms, 0.483 FLINT | 10.3 ms, 0.507 FLINT | 12.1 ms, 0.528 FLINT | 13.5 ms, 0.557 FLINT |
| `interpolate_u64`, 1 | 11.8 ns, 0.409 PARI | 12.2 ns, 0.410 NTL | 12.1 ns, 0.380 PARI | 11.9 ns, 0.375 PARI |
| `interpolate_u64`, 4 | 259 ns, 1.74 NTL | 318 ns, 1.06 NTL | 349 ns, 1.05 PARI | 369 ns, 1.11 PARI |
| `interpolate_u64`, 16 | 885 ns, 0.780 PARI | 963 ns, 0.488 NTL | 1.12 µs, 0.350 PARI | 1.20 µs, 0.372 PARI |
| `interpolate_u64`, 256 | 47.3 µs, 0.700 FLINT | 86.2 µs, 0.830 FLINT | 107 µs, 0.784 FLINT | 102 µs, 0.743 FLINT |
| `interpolate_u64`, 4096 | 1.23 ms, 0.492 FLINT | 2.25 ms, 0.460 FLINT | 2.97 ms, 0.512 FLINT | 2.87 ms, 0.492 FLINT |
| `interpolate_u64`, 65,536 | 30.0 ms, 0.364 FLINT | 61.3 ms, 0.402 FLINT | 71.9 ms, 0.426 FLINT | 72.9 ms, 0.433 FLINT |
| `interpolate_newton`, 1 | 15.4 ns, 3.91 FLINT | 17.7 ns, 4.21 FLINT | 18.2 ns, 4.24 FLINT | 17.3 ns, 4.06 FLINT |
| `interpolate_newton`, 4 | 365 ns, 1.73 FLINT | 445 ns, 0.854 FLINT | 466 ns, 0.650 FLINT | 472 ns, 0.661 FLINT |
| `interpolate_newton`, 16 | 3.43 µs, 0.899 FLINT | 3.90 µs, 0.361 FLINT | 4.17 µs, 0.290 FLINT | 3.90 µs, 0.283 FLINT |
| `interpolate_newton`, 256 | 751 µs, 0.560 FLINT | 765 µs, 0.243 FLINT | 800 µs, 0.197 FLINT | 801 µs, 0.195 FLINT |
| `interpolate_newton`, 4096 | 178 ms, 0.496 FLINT | 178 ms, 0.219 FLINT | 181 ms, 0.174 FLINT | 190 ms, 0.170 FLINT |
| `interpolate_lagrange`, 1 | 15.4 ns, 3.68 FLINT | 18.1 ns, 4.32 FLINT | 17.7 ns, 4.09 FLINT | 16.8 ns, 3.96 FLINT |
| `interpolate_lagrange`, 4 | 281 ns, 0.940 FLINT | 340 ns, 0.668 FLINT | 377 ns, 0.627 FLINT | 400 ns, 0.673 FLINT |
| `interpolate_lagrange`, 16 | 930 ns, 0.291 FLINT | 1.07 µs, 0.261 FLINT | 1.24 µs, 0.266 FLINT | 1.30 µs, 0.271 FLINT |
| `interpolate_lagrange`, 256 | 102 µs, 0.144 FLINT | 121 µs, 0.166 FLINT | 154 µs, 0.202 FLINT | 160 µs, 0.200 FLINT |
| `interpolate_lagrange`, 4096 | 24.5 ms, 0.137 FLINT | 27.6 ms, 0.154 FLINT | 35.1 ms, 0.190 FLINT | 38.6 ms, 0.179 FLINT |

Modulo primes of several limbs (`p127`, `p130`, `p521`):

| operation, n | `p127` | `p130` | `p521` |
| --- | --- | --- | --- |
| `eval_many`, 1 | 25.7 ns, 2.21 FLINT | 26.0 ns, 2.00 FLINT | 26.4 ns, 2.30 FLINT |
| `eval_many`, 4 | 187 ns, 0.952 FLINT | 264 ns, 0.318 NTL | 901 ns, 0.475 FLINT |
| `eval_many`, 16 | 1.23 µs, 0.403 FLINT | 2.15 µs, 0.144 PARI | 11.4 µs, 0.402 FLINT |
| `eval_many`, 64 | 14.5 µs, 0.173 FLINT | 28.1 µs, 0.223 FLINT | 180 µs, 0.643 FLINT |
| `eval_many`, 256 | 217 µs, 0.320 FLINT | 421 µs, 0.450 FLINT | 2.47 ms, 0.986 FLINT |
| `eval_many`, 1024 | 2.35 ms, 0.548 FLINT | 2.80 ms, 0.472 FLINT | 13.3 ms, 0.641 FLINT |
| `eval_many`, 4096 | 12.5 ms, 0.526 FLINT | 14.5 ms, 0.456 FLINT | 69.8 ms, 0.530 FLINT |
| `from_roots`, 1 | 59.7 ns, 4.32 FLINT | 69.6 ns, 4.62 FLINT | 97.6 ns, 7.22 FLINT |
| `from_roots`, 4 | 273 ns, 1.62 FLINT | 338 ns, 0.941 FLINT | 895 ns, 0.986 FLINT |
| `from_roots`, 16 | 2.13 µs, 0.526 FLINT | 2.54 µs, 0.460 PARI | 8.23 µs, 0.522 PARI |
| `from_roots`, 64 | 15.1 µs, 0.624 NTL | 18.5 µs, 0.526 NTL | 80.1 µs, 0.904 NTL |
| `from_roots`, 256 | 94.9 µs, 0.725 NTL | 112 µs, 0.640 NTL | 510 µs, 1.04 NTL |
| `from_roots`, 1024 | 534 µs, 0.770 NTL | 653 µs, 0.691 NTL | 2.92 ms, 1.10 NTL |
| `from_roots`, 4096 | 2.78 ms, 0.788 NTL | 3.30 ms, 0.710 NTL | 15.3 ms, 1.12 NTL |
| `interpolate`, 1 | 33.1 ns, 0.226 NTL | 43.7 ns, 0.287 NTL | 70.6 ns, 0.362 PARI |
| `interpolate`, 4 | 1.04 µs, 0.507 NTL | 1.28 µs, 0.526 NTL | 4.31 µs, 0.572 NTL |
| `interpolate`, 16 | 9.59 µs, 0.499 PARI | 13.1 µs, 0.489 PARI | 51.3 µs, 0.707 PARI |
| `interpolate`, 64 | 96.7 µs, 0.619 PARI | 130 µs, 0.603 PARI | 616 µs, 1.02 PARI |
| `interpolate`, 256 | 613 µs, 0.428 PARI | 756 µs, 0.427 PARI | 3.67 ms, 0.636 PARI |
| `interpolate`, 1024 | 3.39 ms, 0.319 PARI | 4.11 ms, 0.329 PARI | 19.8 ms, 0.455 PARI |
| `interpolate`, 4096 | 17.8 ms, 0.261 PARI | 21.2 ms, 0.274 PARI | 101 ms, 0.374 PARI |

Over the integers, `eval_many` of `n` coefficients at `n` points and `from_roots` of `n` roots:

| n | `eval_many`, 64-bit coefficients, 16-bit points | 64-bit points | 1,000-bit coefficients, 64-bit points | 64-bit coefficients, 1,000-bit points | `from_roots`, 16-bit roots | 64 bits | 1,000 bits |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40.7 ns, 6.26 FLINT | 38.5 ns, 5.79 FLINT | 71.8 ns, 9.83 FLINT | 41.3 ns, 6.14 FLINT | 47.0 ns, 7.68 FLINT | 46.9 ns, 5.99 FLINT | 82.2 ns, 8.05 FLINT |
| 4 | 356 ns, 1.97 FLINT | 490 ns, 2.65 FLINT | 673 ns, 2.62 PARI | 1.75 µs, 1.22 FLINT | 215 ns, 6.14 FLINT | 259 ns, 3.31 FLINT | 1.52 µs, 1.61 FLINT |
| 16 | 2.57 µs, 0.825 FLINT | 4.48 µs, 1.08 PARI | 5.43 µs, 1.07 FLINT | 109 µs, 0.604 PARI | 1.21 µs, 1.35 FLINT | 2.40 µs, 1.78 FLINT | 56.1 µs, 0.795 FLINT |
| 64 | 29.7 µs, 0.477 FLINT | 80.6 µs, 0.778 PARI | 98.0 µs, 0.815 PARI | 3.64 ms, 0.880 FLINT | 27.6 µs, 0.876 FLINT | 102 µs, 0.569 FLINT | 2.38 ms, 0.591 FLINT |
| 256 | 529 µs, 0.454 FLINT | 2.71 ms, 0.889 FLINT | 3.10 ms, 0.715 FLINT | 94.0 ms, 0.677 FLINT | 572 µs, 0.639 FLINT | 2.64 ms, 0.566 FLINT | 51.3 ms, 0.605 FLINT |
| 1024 | 14.7 ms, 0.558 FLINT | 77.2 ms, 0.863 FLINT | 88.5 ms, 0.778 FLINT | 2.09 s, 0.597 FLINT | 12.1 ms, 0.611 FLINT | 54.1 ms, 0.606 FLINT | 2.07 s, 1.33 FLINT |

`Polynomial<Rational>::interpolate` at `n` integer points:

| n | values of 64 bits at 16-bit points | values of 64 bits at 64-bit points | values of 1,000 bits at 64-bit points |
| ---: | --- | --- | --- |
| 1 | 113 ns, 2.64 FLINT | 110 ns, 2.30 FLINT | 132 ns, 2.63 FLINT |
| 4 | 3.77 µs, 3.88 FLINT | 8.04 µs, 2.97 FLINT | 10.9 µs, 3.31 FLINT |
| 16 | 94.4 µs, 3.66 FLINT | 593 µs, 3.74 FLINT | 694 µs, 0.309 PARI |
| 64 | 4.87 ms, 2.26 FLINT | 101 ms, 1.47 FLINT | 102 ms, 1.46 FLINT |
| 256 | 270 ms, 1.77 FLINT | 13.7 s, 3.54 FLINT | 13.6 s, 3.46 FLINT |
| 1024 | 4.74 s, 1.91 FLINT |  |  |

### Roots

`roots()` of random polynomials, the coefficients (or their parts) uniform in [−1, 1], against Arb's `acb_poly_find_roots` at 53 bits and PARI's `polroots` at 64 bits:

| degree | `Polynomial<f64>` | `Polynomial<Complex<f64>>` |
| ---: | --- | --- |
| 10 | 4.61 µs, 0.158 Arb (0.717 LAPACK†) | 4.63 µs, 0.164 Arb (0.289 LAPACK†) |
| 30 | 33.9 µs, 0.110 Arb (0.523 LAPACK†) |  |
| 100 | 343 µs, 0.143 Arb (0.213 LAPACK†) | 334 µs, 0.148 Arb |
| 300 | 2.79 ms, 0.146 Arb |  |
| 1000 | 30.7 ms, 0.144 Arb | 29.5 ms, 0.131 Arb |

Isolating intervals of the real roots and their count, against FLINT's `fmpz_poly_isolate_real_roots`, `fmpz_poly_num_real_roots` and `_sturm`, and PARI's `polsturm`:

| polynomial, degree | `real_root_intervals` | `_descartes` | `_cf` | `count_real_roots` |
| --- | --- | --- | --- | --- |
| random 10, 16 bits | 4.94 µs, 3.23 FLINT | 4.52 µs, 3.11 FLINT | 3.33 µs, 2.25 FLINT |  |
| random 20, 16 bits | 10.4 µs, 1.92 FLINT | 9.93 µs, 1.85 FLINT | 6.65 µs, 1.22 FLINT |  |
| random 50, 16 bits | 39.2 µs, 0.713 FLINT | 37.2 µs, 0.685 FLINT | 29.8 µs, 0.548 FLINT | 44.8 µs, 0.319 PARI |
| random 100, 16 bits | 94.6 µs, 0.752 FLINT | 90.7 µs, 0.722 FLINT | 52.0 µs, 0.415 FLINT |  |
| random 200, 16 bits | 362 µs, 0.382 FLINT | 349 µs, 0.371 FLINT | 719 µs, 0.753 FLINT | 412 µs, 0.315 PARI |
| random 500, 16 bits | 1.43 ms, 0.462 FLINT | 1.31 ms, 0.428 FLINT | 1.42 ms, 0.463 FLINT | 1.59 ms, 0.317 PARI |
| random 1000, 16 bits | 32.2 ms, 0.491 FLINT | 31.7 ms, 0.478 FLINT | 169 ms, 2.55 FLINT |  |
| random 100, 1000 bits | 166 µs, 0.537 FLINT | 161 µs, 0.526 FLINT | 333 µs, 1.11 FLINT |  |
| random 300, 1000 bits | 2.89 ms, 0.308 FLINT | 2.81 ms, 0.303 FLINT | 12.7 ms, 1.34 FLINT |  |
| random 100, 10000 bits | 949 µs, 0.557 FLINT | 939 µs, 0.551 FLINT | 938 µs, 0.551 FLINT |  |
| Chebyshev `T` 20 | 28.2 µs, 0.672 FLINT | 28.1 µs, 0.673 FLINT | 44.1 µs, 1.08 FLINT |  |
| Chebyshev `T` 50 | 273 µs, 0.308 FLINT | 275 µs, 0.307 FLINT | 864 µs, 0.977 FLINT |  |
| Chebyshev `T` 100 | 2.13 ms, 0.308 FLINT | 2.10 ms, 0.297 FLINT | 11.2 ms, 1.59 FLINT | 2.11 ms, 1.26 PARI |
| Chebyshev `T` 200 | 17.1 ms, 0.246 FLINT | 17.2 ms, 0.251 FLINT | 159 ms, 2.32 FLINT |  |
| Wilkinson 20 | 16.7 µs, 1.05 FLINT | 36.1 µs, 2.28 FLINT | 16.2 µs, 0.997 FLINT |  |
| Wilkinson 50 | 103 µs, 0.326 FLINT | 388 µs, 1.24 FLINT | 104 µs, 0.328 FLINT | 99.1 µs, 0.373 PARI |
| Wilkinson 100 | 766 µs, 0.350 FLINT | 3.06 ms, 1.40 FLINT | 758 µs, 0.342 FLINT |  |
| Mignotte 20, a = 2¹⁰ + 1 | 87.4 µs, 0.221 FLINT | 233 µs, 0.595 FLINT | 56.5 µs, 0.144 FLINT |  |
| Mignotte 40, a = 2³⁰ + 1 | 3.88 ms, 0.118 FLINT | 32.4 ms, 0.984 FLINT | 3.56 ms, 0.109 FLINT | 16.5 µs, 2.77 FLINT |
| Mignotte 100, a = 2⁶⁴ + 1 | 258 ms, 0.025 FLINT | 8.98 s, 0.866 FLINT | 328 ms, 0.032 FLINT |  |
| sparse 1000, 64 bits | 7.94 ms, 0.754 FLINT | 8.17 ms, 0.765 FLINT | 4.27 ms, 0.406 FLINT |  |

`complex_roots(prec)`, certified, against Arb's `arb_fmpz_poly_complex_roots`:

| polynomial, degree, precision | `complex_roots(prec)` |
| --- | --- |
| random 10, 16 bits, prec 53 | 68.6 µs, 0.415 Arb |
| random 30, 16 bits, prec 53 | 449 µs, 0.321 Arb |
| random 100, 16 bits, prec 53 | 4.12 ms, 0.266 Arb |
| random 300, 16 bits, prec 53 | 67.6 ms, 0.383 Arb |
| random 30, 16 bits, prec 333 | 1.10 ms, 0.424 Arb |
| random 100, 16 bits, prec 333 | 10.1 ms, 0.368 Arb |
| random 30, 16 bits, prec 3333 | 13.6 ms, 0.568 Arb |
| random 100, 1000 bits, prec 53 | 4.13 ms, 0.156 Arb |
| Chebyshev `T` 30, prec 53 | 499 µs, 0.834 Arb |
| Mignotte 30, a = 2¹⁰ + 1, prec 53 | 2.44 ms, 0.150 Arb (0.317 PARI†) |
| Wilkinson 30, prec 100 | 2.62 ms, 0.487 Arb |

`real_roots(prec)`, correctly rounded, has no reference that does the same: Arb's `arb_fmpz_poly_real_roots` gives balls of the precision, PARI's `polrootsreal` uncertified roots. Against the faster of the two:

| polynomial, degree, precision | `real_roots(prec)` |
| --- | --- |
| random 20, 16 bits, prec 53 | 34.4 µs (1.16 Arb†) |
| random 100, 16 bits, prec 53 | 255 µs (1.32 Arb†) |
| random 100, 16 bits, prec 333 | 282 µs (1.23 Arb†) |
| random 300, 16 bits, prec 53 | 2.06 ms (0.585 Arb†) |
| Chebyshev `T` 30, prec 53 | 293 µs (1.88 Arb†) |
| Chebyshev `T` 100, prec 53 | 4.24 ms (1.24 Arb†) |
| Chebyshev `T` 30, prec 3333 | 1.64 ms (2.49 Arb†) |
| Wilkinson 30, prec 333 | 32.9 µs (0.390 Arb†) |

### Factorisation

Modulo a word prime, of random monic polynomials and of irreducible ones (`roots` with multiplicities against FLINT's `nmod_poly_roots`):

| operation, degree | `3` | `2⁵⁰ − 27` | `2⁶³ − 25` | `2⁶⁴ − 59` |
| --- | --- | --- | --- | --- |
| `factor`, 10 | 2.91 µs, 1.42 PARI | 23.4 µs, 0.897 NTL | 42.4 µs, 0.640 PARI | 41.9 µs, 0.611 PARI |
| `factor`, 30 | 11.1 µs, 0.642 FLINT | 88 µs, 0.756 NTL | 170 µs, 0.337 PARI | 135 µs, 0.704 FLINT |
| `factor`, 100 | 105 µs, 0.539 FLINT | 1.26 ms, 0.862 FLINT | 1.53 ms, 0.640 FLINT | 1.75 ms, 0.722 FLINT |
| `factor`, 300 | 1.33 ms, 0.604 FLINT | 9.84 ms, 0.622 NTL | 11 ms, 0.504 FLINT | 9.74 ms, 0.351 FLINT |
| `factor`, 1000 | 27.5 ms, 1.55 FLINT | 68.3 ms, 0.561 NTL | 102 ms, 0.397 FLINT | 74.9 ms, 0.442 FLINT |
| `factor_berlekamp`, 10 | 2.67 µs, 0.798 FLINT | 34.6 µs, 1.08 NTL | 48.3 µs, 0.281 FLINT | 50.6 µs, 0.283 FLINT |
| `factor_berlekamp`, 30 | 11 µs, 0.360 FLINT | 121 µs, 0.896 NTL | 181 µs, 0.186 FLINT | 155 µs, 0.211 FLINT |
| `factor_berlekamp`, 100 | 104 µs, 0.384 FLINT | 1.34 ms, 0.778 NTL | 1.78 ms, 0.186 FLINT | 1.73 ms, 0.099 FLINT |
| `factor_berlekamp`, 300 | 1.32 ms, 0.650 NTL | 12.5 ms, 0.883 NTL | 17 ms, 0.145 FLINT | 17 ms, 0.240 FLINT |
| `roots`, 10 | 143 ns, 0.226 FLINT | 13.3 µs, 0.557 FLINT | 20.7 µs, 0.516 FLINT | 18.6 µs, 0.503 FLINT |
| `roots`, 30 | 199 ns, 0.207 FLINT | 44.2 µs, 0.418 FLINT | 78.4 µs, 0.435 FLINT | 70.6 µs, 0.405 FLINT |
| `roots`, 100 | 381 ns, 0.073 FLINT | 319 µs, 0.507 FLINT | 478 µs, 0.399 FLINT | 505 µs, 0.428 FLINT |
| `roots`, 300 | 1.4 µs, 0.042 FLINT | 981 µs, 0.230 FLINT | 1.65 ms, 0.293 FLINT | 1.67 ms, 0.306 FLINT |
| `roots`, 1000 | 1.47 µs, 0.006 FLINT | 3.1 ms, 0.236 FLINT | 4.79 ms, 0.267 FLINT | 5.33 ms, 0.309 FLINT |
| `is_irreducible`, 10 | 2.19 µs, 0.875 PARI | 15.4 µs, 0.751 FLINT | 23.9 µs, 0.788 FLINT | 22.5 µs, 0.735 FLINT |
| `is_irreducible`, 30 | 15.5 µs, 0.721 FLINT | 84.1 µs, 0.628 FLINT | 129 µs, 0.657 FLINT | 122 µs, 0.615 FLINT |
| `is_irreducible`, 100 | 299 µs, 1.26 FLINT | 1.42 ms, 0.802 FLINT | 1.82 ms, 0.612 FLINT | 1.83 ms, 0.618 FLINT |
| `is_irreducible`, 300 | 2.74 ms, 1.19 FLINT | 10.2 ms, 0.454 NTL | 14 ms, 0.444 FLINT | 14.1 ms, 0.454 FLINT |

The steps of the factorisation, and modulo primes of several limbs:

| operation, degree | `a` |
| --- | --- |
| `factor` modulo 2, 100 | 11.2 µs, 1.38 NTL |
| `factor` modulo 2, 1000 | 1.04 ms, 3.33 NTL |
| `squarefree` of `f₁f₂²f₃³` modulo `2⁵⁰ − 27`, 120 | 13.8 µs, 0.635 FLINT |
| `squarefree` of `f₁f₂²f₃³` modulo `2⁵⁰ − 27`, 1200 | 812 µs, 1.03 FLINT |
| `distinct_degree_factor` modulo `2⁵⁰ − 27`, 30 | 80.7 µs, 0.708 FLINT |
| `distinct_degree_factor` modulo `2⁵⁰ − 27`, 100 | 1.42 ms, 0.813 FLINT |
| `distinct_degree_factor` modulo `2⁵⁰ − 27`, 300 | 10.2 ms, 0.352 FLINT |
| `equal_degree_factor` modulo `2⁵⁰ − 27`, 10 factors of degree 5 | 676 µs, 0.221 FLINT |
| `equal_degree_factor` modulo `2⁵⁰ − 27`, 10 factors of degree 10 | 2.33 ms, 0.170 FLINT |
| `equal_degree_factor` modulo `2⁵⁰ − 27`, 5 factors of degree 20 | 3.17 ms, 0.139 FLINT |
| `factor` modulo `2⁶⁴ + 13`, 10 | 103 µs, 0.510 NTL |
| `factor` modulo `2⁶⁴ + 13`, 30 | 643 µs, 0.947 NTL |
| `factor` modulo `2⁶⁴ + 13`, 100 | 4.23 ms, 0.840 NTL |
| `factor` modulo `2¹²⁷ − 1`, 10 | 456 µs, 0.728 NTL |
| `factor` modulo `2¹²⁷ − 1`, 30 | 1.52 ms, 1.02 NTL |
| `factor` modulo `2¹²⁷ − 1`, 100 | 9.4 ms, 1.06 NTL |
| `roots` modulo `2¹²⁷ − 1`, 10 | 255 µs, 0.520 FLINT |
| `roots` modulo `2¹²⁷ − 1`, 30 | 1.08 ms, 0.607 FLINT |
| `roots` modulo `2¹²⁷ − 1`, 100 | 4.32 ms, 0.525 FLINT |
| `is_irreducible` modulo `2¹²⁷ − 1`, 10 | 257 µs, 0.488 NTL |
| `is_irreducible` modulo `2¹²⁷ − 1`, 30 | 1.54 ms, 0.819 NTL |
| `is_irreducible` modulo `2¹²⁷ − 1`, 100 | 10.6 ms, 0.842 NTL |

Over the integers:

| polynomial | `factor` |
| --- | --- |
| random, 8-bit, degree 10 | 11.3 µs, 0.705 FLINT |
| random, 8-bit, degree 30 | 55.8 µs, 0.380 FLINT |
| random, 8-bit, degree 100 | 421 µs, 0.373 FLINT |
| random, 8-bit, degree 300 | 3.33 ms, 0.350 NTL |
| random, 64-bit, degree 10 | 11.2 µs, 0.385 NTL |
| random, 64-bit, degree 30 | 73.3 µs, 0.377 NTL |
| random, 64-bit, degree 100 | 535 µs, 0.417 FLINT |
| random, 1,000-bit, degree 10 | 17.9 µs, 0.539 NTL |
| random, 1,000-bit, degree 30 | 102 µs, 0.829 NTL |
| product of two, 8-bit, degree 10 | 26.0 µs, 0.689 FLINT |
| product of two, 8-bit, degree 30 | 92.5 µs, 0.425 FLINT |
| product of two, 8-bit, degree 100 | 438 µs, 0.305 FLINT |
| product of two, 8-bit, degree 300 | 4.59 ms, 0.416 FLINT |
| product of two, 64-bit, degree 30 | 196 µs, 0.524 FLINT |
| product of two, 64-bit, degree 100 | 1.29 ms, 0.402 FLINT |
| product of two, 1,000-bit, degree 20 | 1.41 ms, 1.13 FLINT |
| product of two, 10,000-bit, degree 10 | 10.8 ms, 0.961 NTL |
| `Π(x − i)`, degree 10 | 53.6 µs, 1.18 FLINT |
| `Π(x − i)`, degree 20 | 187 µs, 1.04 FLINT |
| `Π(x − i)`, degree 50 | 908 µs, 0.897 FLINT |
| Swinnerton-Dyer, degree 8 | 21.4 µs, 0.470 FLINT |
| Swinnerton-Dyer, degree 16 | 57.0 µs, 0.401 PARI |
| Swinnerton-Dyer, degree 32 | 576 µs, 0.656 FLINT |
| `xⁿ − 1`, degree 30 | 88.4 µs, 0.440 PARI |
| `xⁿ − 1`, degree 60 | 337 µs, 0.749 PARI |
| `xⁿ − 1`, degree 105 | 829 µs, 0.585 PARI |
| `xⁿ − 1`, degree 120 | 964 µs, 0.859 PARI |
| `xⁿ − 1`, degree 210 | 3.57 ms, 0.864 PARI |
| `squarefree` of `f₁f₂²f₃³`, 64-bit, degree 30 | 140 µs, 1.94 FLINT |
| `squarefree` of `f₁f₂²f₃³`, 64-bit, degree 300 | 2.53 ms, 1.49 NTL |
| `is_irreducible`, random, 8-bit, degree 30 | 53.5 µs, 0.154 PARI |
| `is_irreducible`, random, 8-bit, degree 100 | 418 µs, 0.134 PARI |
| `is_irreducible`, Swinnerton-Dyer, degree 16 | 55.3 µs, 0.401 PARI |
| `is_irreducible`, Swinnerton-Dyer, degree 32 | 557 µs, 0.663 PARI |

### Where the references are faster

Every loss of the cases above (a ratio above 1.05 against the fastest reference that does the same work), by cause:

- **Fixed costs at one to a few coefficients, points or roots.** `a`'s operators and functions return a new polynomial each time (an allocation and the modulus's reference count), where FLINT's and NTL's loops write into storage that exists; a `ModPolynomial` made from points or roots builds its modulus too, and every `Number` coefficient is a vector of its own, where FLINT's `fmpz` keeps a small value in the word. Modulo a word: products and squares of 1 to 8 coefficients (1.07 to 2.86; one coefficient 13.5 ns to 16.7 ns, 2.10 to 2.21), `div_rem` of 4 / 2 to 16 / 8 (1.13 to 2.96), `inv_series` of 2 to 16 (1.16 to 2.71), `mul_mod` to 16 (1.06 to 1.96), `compose` of 2 to 16 (1.10 to 2.39), `pow(7)` of 2 coefficients (1.55 to 1.78), `compose_mod` of 4 (1.31 to 1.39); modulo several limbs, products and squares of 1 to 8 (1.06 to 1.41); over the integers, products and squares of 1 to 16 (to 6.15 at one 32-bit coefficient), `compose`, `pow(7)` and `eval` at 4 to 256 (1.29 to 2.76), `pseudo_div_rem` of 64-bit coefficients at 4 / 2 to 16 / 8 (1.32 to 1.74, PARI); and at 1 to 16 points or roots, `eval_many_u64`, `from_roots_u64` and `interpolate_u64` (1.06 to 3.32), `interpolate_newton` and `interpolate_lagrange` at one point (3.68 to 4.32; Newton's at 4 modulo `p20` 1.73), modulo several limbs `eval_many` and `from_roots` at 1 and 4 (to 7.22 for one root modulo `p521`), over the integers `eval_many` and `from_roots` to 16 (to 9.83 and 8.05).
- **Moduli below `2³²` (65,521, and gcds modulo `2¹⁷ − 1`) at long lengths, against FLINT.** Products and squares from 256 to 10⁵ coefficients (1.09 to 1.46), `div_rem` from 1024 / 512 (1.08 to 1.24), `mul_mod` from 1,000 (1.06 to 1.23), `compose` at 32 to 512 (1.22 to 1.31), `pow(7)` of 128 and 1,000 (1.07); the gcds, extended gcds, resultants and discriminants modulo `2¹⁷ − 1` from degree 1000 (1.10 to 1.27; at degree 10 1.08 to 1.26). FLINT's `fft_small` takes one 50-bit prime and truncated transforms of any length; `a` takes two 29-bit primes and power-of-two lengths, and its Euclid's steps run on the 64-bit kernel where FLINT reduces lazily in one word.
- **The division family modulo 50- to 64-bit words at 8 to 512 coefficients**: `div_rem` of 32 / 16 to 512 / 256 (1.09 to 1.43), `mul_mod` of 32 to 256 (1.06 to 1.25), products and squares of 16 to 64 coefficients, where the transforms take over (1.07 to 2.17, NTL at `2⁵⁰ − 27`), `pow(7)` of 16 modulo `2⁵⁰ − 27` (1.08) and `compose` of 2,000 modulo `2⁶⁴ − 59` (1.07): FLINT's and NTL's divide-and-conquer division and middle products at those lengths, against the schoolbook division and Newton's method with the crate's middle products.
- **Modulo 2, against NTL's `GF2X`**: products of 64 to 10⁶ coefficients 1.89 to 10.9 times its time (one word: 33.6 ns against 3.09 ns, a call's fixed costs), squares 4.34 to 11.4 (the bits spread by shifts and masks, half a word at a time), `div_rem` of 128 / 64 and 200,000 / 100,000 (1.40 against PARI, 2.13), `factor` at degree 100 and 1000 (1.38, 3.33). `a` multiplies the packed words by PMULL with Karatsuba's method, and the longest by the transforms on words; why NTL's products of 1,000 coefficients and more are about twice as fast was not determined.
- **Moduli of 4 to 16 limbs against NTL's `ZZ_pX`**: products and squares from 128 coefficients (1.06 to 1.75), `div_rem` (1.09 to 1.77, FLINT at 128 / 64 modulo `2⁵²¹ − 1` and `2¹⁰²⁴ − 105`, 1.73 and 1.54), `inv_series` (1.21 to 2.80, FLINT at 16), `mul_mod` from 256 (1.18 to 1.63), `from_roots` of 1,024 and 4,096 roots modulo `p521` (1.10, 1.12); and the gcds, extended gcds and resultants modulo 2 to 16 limbs from degree 100 to 300 on (1.07 to 1.73), which run on those products. NTL holds a 127-bit residue's sums of products in about five 60-bit primes, one forward and one inverse transform a product, and reduces modulo `f` in its transforms; `a` cuts each residue into 2 to 6 pieces for 29-bit primes, and `2¹⁰²⁴ − 105` stays on Kronecker substitution, the pieces measured slower there.
- **`Number` coefficients over the integers**: products and squares of 32-bit coefficients at every length (1.34 to 6.15), of 64-bit ones from 128 coefficients (1.22 to 1.96), of 1,000 bits up to 64 (1.16 to 2.98) and squares of 10⁵ of them (1.09) and of 10,000 bits at 5,000 to 3·10⁴ (1.06 to 1.63, one product of integers of about 3·10⁸ bits by Kronecker substitution at 3·10⁴); `div_exact` of 1,000-bit coefficients at 7 / 4 to 1023 / 512 (1.08 to 1.33, FLINT's divide and conquer with an early abort); `+` and `derivative` (1.57 to 2.97 and 4.69 to 5.37, NTL) and `content` at 1,000 (1.17); the gcd at degree 5 and 3000 (1.59, 1.51), the resultants and discriminants at degree 5 and 10 (1.53 to 2.26); `from_roots` of 1,024 roots of 1,000 bits (1.33); `squarefree` (1.49, 1.94) and `factor` of `Π(x − i)` at degree 10 and of a product of two 1,000-bit factors of degree 20 (1.18, 1.13). Every coefficient is a vector of bytes that a product reads and builds anew, and the small degrees pay the setup of the heuristic and modular gcds on such vectors.
- **The linear operations modulo a word**: `eval` (1.09 to 2.37, FLINT), `scale` (1.61 to 2.13), `derivative` (1.06 to 2.51, PARI) and `+` of 16 coefficients (1.55 to 1.86): per-coefficient generic modular arithmetic (`derivative` reduces each index), with no word or SIMD fast path.
- **Coefficients in lowest terms over the rationals**: products (1.16 to 9.09), `div_rem` (1.23 to 9.29), `inv_series` (2.11 to 3.07), `compose` (2.17, 3.80), `interpolate` (1.46 to 3.88), `gcd` and `xgcd` to degree 100 (1.13 to 2.35), the resultants at degree 5 and 10 (1.51, 2.00). FLINT's `fmpq_poly` keeps one common denominator; every coefficient here is a `Rational` in lowest terms, a gcd of numbers the size of the common denominator for each coefficient of the result.
- **Floats**: the product of 4 by 4 `f64` coefficients (1.24 against the schoolbook loop), the call's fixed costs.
- **Root isolation and counting, small degrees and edges**: `real_root_intervals` and its two methods at degree 10 and 20 (1.22 to 3.23), the continued fractions alone on dense polynomials from degree 100 and on Chebyshev's from 20 (1.08 to 2.55; the automatic choice takes Descartes's rule there), Descartes's rule alone on Wilkinson's (1.24 to 2.28; the automatic choice takes the continued fractions), all against FLINT; `count_real_roots` of Mignotte's of degree 40 (2.77, FLINT: the square-free test and Sturm's sequence over `Number`s) and of `T₁₀₀` (1.26, PARI).
- **Factorisation, edges**: modulo 3 `factor` at degree 10 and 1000 (1.42 PARI, 1.55 FLINT) and `is_irreducible` at 100 and 300 (1.26, 1.19), which spend their time in the gcds modulo a small prime (above); `factor_berlekamp` at degree 10 modulo `2⁵⁰ − 27` (1.08) and `factor` at degree 100 modulo `2¹²⁷ − 1` (1.06), against NTL.

Not counted, against references that do less: FLINT's preinverse and NTL's `Modulus` built before the clock are faster than `a`, which builds its own in the call, at `x.pow_mod` modulo a word for `f` of 4 to 32 coefficients (1.09 to 1.23 against FLINT) and modulo 2 (1.64 to 2.33 against NTL's `GF2X`), at `x.pow_mod` modulo several limbs from 32 coefficients (1.07 to 1.92 against NTL) and at 4 to 32 (1.09 to 1.32 against FLINT), at `pow_mod` of 4 coefficients (1.13, 1.19), and at `compose_mod` modulo a word to 128 coefficients and modulo 65,521 to 5,000 (1.08 to 1.94, FLINT) and modulo several limbs from 256 (1.29 to 3.54, NTL); against PARI, which builds them in the call, `a` takes 0.084 to 0.814 of the time of `x.pow_mod` and `pow_mod`, and loses `compose_mod` only at 4 coefficients. PARI's distinct roots without multiplicities are faster modulo 3 at every degree (1.56 to 4.35); FLINT's pseudo-division with fewer factors of the leading coefficient at 8 / 4 to 64 / 32 coefficients of 64 bits (1.07 to 1.52); the float products' transforms below the line where `a` keeps the schoolbook method and its componentwise bound, from 128 `f64` coefficients (1.84 to 3.96 against realfft at 128 to 400) and at 64 complex ones (1.94 against rustfft), the price of the bound; and for `real_roots(prec)` Arb's balls (to 2.49, Chebyshev's `T₃₀` at 3333 bits) and PARI's uncertified roots (to 1.37). LAPACK's eigenvalues, NTL's randomised resultant and PARI's uncertified complex roots are slower than `a` everywhere.

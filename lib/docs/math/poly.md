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

- `*` and `sqr` multiply by the method the coefficients and the lengths call for ([below](#products-and-division-1)): the schoolbook method, Kronecker substitution into the integers' multiplication, or the number-theoretic transforms on the coefficients, and the fast Fourier transform for long float products; `pow(n)` is repeated squaring on them. Over the integers the product of two polynomials of 10,000 coefficients of 64 bits takes 1.2 ms, of 1,000 bits 17 ms; modulo a word, of 100,000 coefficients each, 1.9 to 5.8 ms.
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
- `real_root_intervals()` gives the real roots of a `Polynomial<Number>` or `Polynomial<Rational>` as isolating intervals with their multiplicities, increasing: `(a, a, m)` for a rational root `a` found exactly (every root of a factor of degree 1, and any found at a midpoint), `(a, b, m)` with `a < b` for a root that is the only root of the polynomial in the open interval `(a, b)`, whose ends are not roots; no two overlap. `real_root_intervals_descartes()` and `real_root_intervals_cf()` give them by each of the two methods below, each on its own; `real_root_intervals()` chooses by the input: continued fractions where the bound of the roots is large (Wilkinson's: four times faster than Descartes's rule at the degrees 50 and 100), else Descartes's rule (on dense polynomials from the degree 200 up to six times faster than continued fractions on random ones, and three to ten times from the degree 50 on Chebyshev's), its nodes deep in the bisection going on by continued fractions (Mignotte's close pairs: 2.6 to 33 times faster than Descartes's rule alone).
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
- **Modulo several limbs** (FLINT's `fmpz_mod_poly`): every residue of a sum of products is the remainder of its limbs by the normalized `n` with the reciprocal of its top two limbs (Möller and Granlund's division of three limbs by two), computed once for the modulus: one quotient limb a step, the remainder in two registers for `n` of two limbs, a product of the quotient limb by the other limbs subtracted beyond (the crate's division of limbs from 5 limbs, whose products run in assembly), for any `n`, even or odd, without allocating. The schoolbook method takes up to 6 coefficients (4 from 3 limbs), its sums of products kept in `2k + 1` limbs on the stack and divided once. From 48 coefficients in the shorter operand, for 2 to 9 limbs and products of up to about 6,000 coefficients, the transforms on pieces of the coefficients (NTL's way, with the crate's primes): each coefficient cut into `P` pieces of at most 112 bits (2 of 64 bits for a 127-bit `n`, 4 of 64 bits for 255 bits, 5 of 105 bits for 521, with 5 to 8 primes; `P` and the number of primes chosen by an estimate of the cost), each piece a polynomial transformed once a prime, the `2P − 1` pieces of the product `C_t = Σ A_s·B_(t−s)` summed in the transforms (the sums reduced every four products, within the 32-bit lanes) and brought back by one inverse transform each, every `c_t` exact by the theorem (below half the primes' product), and the coefficient `Σ c_t·2^(t·w)` divided by `n` once. Otherwise Kronecker substitution, each polynomial evaluated at `2^s` with `s = 2·bits(n−1) + bits(m)`, one product of integers (Karatsuba's, Toom's or the transforms), and each field of the product divided by `n` once.
- **Over the integers** (FLINT's `fmpz_poly_mul`): up to 12 coefficients in the shorter operand (48 in the longer), for coefficients of at most 64 bits, the schoolbook method in machine integers (each sum of products in an `i128` when it fits, else the positive and the negative products apart in three words), one number built a coefficient; up to 4 coefficients of up to 2,048 bits, the schoolbook method on limbs (each product of magnitudes added to the positive or the negative sum). From 48 coefficients in the shorter operand, for coefficients of at most 112 bits whose sums of products stay below `2²²⁴`, the transforms on the coefficients directly, as modulo a word: each coefficient cut into digits of at most 28 bits, negated for a negative one, the convolution modulo the fewest primes whose product `P` exceeds four times the largest sum, and the residues shifted by those of `⌊P/2⌋`, so that the theorem gives `c + ⌊P/2⌋` in `[0, P)` and the signed `c` comes out exactly in 128 bits for up to four primes (`P < 2^116`), three limbs for up to six, four beyond. Otherwise Kronecker substitution with signs: a polynomial whose leading coefficient is negative is negated, its value at `2^s` (`s` the bits of the largest product sum plus a sign bit) is the sum of the positive coefficients' fields less that of the negative ones', and the product's coefficients come back from the bottom, a field `v` (with the borrow from below) being `v` below `2^(s−1)` and `v − 2^s`, a borrow of 1 from the field above, from it.
- **Over the rationals**: both polynomials over common denominators, the integer polynomials multiplied so, each coefficient of `AB/(da·db)` then in lowest terms (the schoolbook method up to 16 coefficients, where products of rationals in words beat a common denominator that grows with the length).
- **Over the floats and the complex floats**: the schoolbook method, whose error is bounded componentwise, `|ĉ_k − c_k| ≤ γ_m Σ|a_i||b_(k−i)|`; from about 512 coefficients in the shorter operand (80 for complex floats, whose schoolbook products cost four real ones; exactly, when `la·lb` exceeds 19 or 5 times `N log₂ N`) a radix-2 fast Fourier transform of a length `N`, twiddles from their own sines and cosines, decimation in frequency forward and in time back so that no permutation is needed, whose error is bounded in norm: `|ĉ_k − c_k| ≤ 11·log₂(N)·ε·‖a‖₂‖b‖₂` (Percival, 2003). Coefficients much smaller than the norms can then lose their relative accuracy; the schoolbook method keeps it.
- **Power series inverses**: the schoolbook recurrence `g_j = −c₀⁻¹ Σ fᵢ g_(j−i)` (dot products reduced once) up to 160 coefficients modulo a word (16 for several limbs, 32 over a field), Newton's iteration `g ← g − xʰ·((f·g − 1)/xʰ · g mod x^(n−h))` beyond, about three products of the length `n`. Modulo a word the first product is taken modulo `x^L − 1` for `L ≥ n` a power of 2: its coefficients past `L` wrap onto those below `h`, which are known, so the transform is up to half as long. Over the rationals through the integers (FLINT's `fmpq_poly_inv_series` way, with each coefficient then in lowest terms): `a = A/den`, `A(c·y) = c·U(y)` for `c = A₀` with the integer `U = 1 + Σ A_j c^(j−1) y^j`, whose inverse is exact over the integers by Newton's iteration on integer products, and `1/a` has the coefficients `den·v_j / c^(j+1)`, brought to lowest terms by the powers of `gcd(num, gcd(c, den))` alone (every common prime divides `c`), found by doubling; that for a common denominator of up to 64 bits, as the powers `c^(j+1)` overshoot the true denominators by far beyond. Over a longer common denominator, Newton's iteration on one common denominator as FLINT does, `g = N/d`, `N' = N·(2·den·d − A·N) mod x^m` over `den·d²` with the common content taken out at each step, and each coefficient brought to lowest terms once at the end.
- **Division**: the schoolbook method for a quotient or a divisor of up to 256 coefficients modulo a word (128 modulo 2 to 4 limbs and 32 beyond, its sums in `2k + 2` limbs), the remainder's coefficients kept as unreduced sums `a_j + Σ(n − t_i)·d_(j−i)` (in a word, two or three) and only the one that gives the next quotient coefficient reduced at each step; beyond, Newton's method: the reversed quotient is the reversed dividend times the inverse of the reversed divisor modulo `x^(la−ld+1)`, the remainder `a − q·d` on its low coefficients. Over the rationals from 16 coefficients in the quotient through the integers: the pseudo-division of the numerators over common denominators, then `q = Q·dd/(da·lᵏ)` and `r = R/(da·lᵏ)` in lowest terms. Over the floats Newton's method from 512 coefficients in the quotient and the divisor.
- **Pseudo-division and exact division over the integers**: the schoolbook method on limbs, every coefficient a magnitude and a sign in a slot of a width fixed by a bound (`|R|` grows by at most `2·max(|l|, H)` a step), the operations in place. The pseudo-division runs the recurrence `R ← l·R − t·xⁱ·d` lazily: a coefficient keeps the count of the factors `l` it has had and takes those it misses in one product by a power of `l` (computed once) when a step changes it, so a step multiplies only the `deg d` coefficients it subtracts from, and the quotient's coefficients `t·l^(k−1−s)` take their powers once at the end, where the classic method multiplies every coefficient below the top, and the quotient, by `l` at every step; a negative `l` divides by `−d` and turns the signs of `q` and `r` by the parity of `k`. The exact division takes each quotient coefficient as the top coefficient divided by `l`, `None` at the first that does not divide or when a value outgrows the slots, sized by Mignotte's bound on a divisor's cofactor (which an exact division never exceeds). Below 7 products in all (`(la − ld + 1)·(ld − 1) ≤ 6`) the schoolbook method on numbers, whose set-up is less; by a constant `l`, `q = l^(deg a)·a` and `r = 0` directly (one power, one product a coefficient).
- **Kronecker substitution into the division of integers** takes over from 12 coefficients in the quotient for `div_exact` (48 for divisors of 512 bits and more) and from 384 (32 for divisors of 1,000 bits) for `pseudo_div_rem`. For `div_exact` the values at `2^s` divide when the polynomials do, so a remainder of the integers gives `None`; the quotient read back is checked by one product, the field doubled while the check fails, up to Mignotte's bound `2^(deg q)·‖a‖₂` on a divisor's coefficients. It says `None` at once for a divisor wider than Mignotte's bound `2^(deg d)·‖a‖₂` on a divisor of `a`, and tries fields that hold both operands, so that a divisor much wider than the dividend neither breaks the packing nor hides a true quotient. For `lᵏ·a = q·d + r` the field must hold every coefficient of `q` and `r`, bounded from `|q_j| ≤ (|lᵏa| + n·H·|q|)/|l|` (`H` the largest coefficient of `d`), and so the largest power of `l`: the quotient's value is the nearest integer to the quotient of the two values, as the remainder's is below half the divisor's, and the remainder takes one product. Past that crossover the pseudo-division first tries the schoolbook method with the fewest factors of `l`, as FLINT's `fmpz_poly_pseudo_divrem` runs it: a step whose top coefficient `l` divides takes `t/l` into the quotient with no factor, every coefficient keeps the number of factors it was last brought to and takes the missing power in one product when a step touches it, and at the end each coefficient of `q` and `r` is multiplied by the power of `l` it misses, the powers in increasing order each from the one before, so that the result is `lᵏ·a = q·d + r` exactly. An exact or nearly exact division then stays at about the size of its operands (an exact one of 800 by 400 coefficients of 64 bits takes 2.1 ms instead of 31); a division whose fifth step needs a factor goes on to Kronecker substitution, after a few products.
- **Modulo a fixed polynomial** (FLINT's `_preinv` functions): the inverse of the reversed `f` modulo `x^(deg f)` is computed once; a remainder of a polynomial of degree below `2 deg f − 1` is then two products, the second modulo `x^L − 1` for `L ≥ deg f + 1` (the coefficients of `q·f` from `deg f` on are those of the dividend, so the wrapped part is known: `r_j = a_j − (qf mod x^L−1)_j + a_(j+L)`). Modulo a word the remainders, products and squares run on vectors of residues reused from step to step, with the schoolbook division up to 160 coefficients; for an `f` of up to 16 coefficients a product modulo `f` is one pass on the stack, its sums of products in three words divided by `f` in place and reduced once each. Modulo several limbs, from 49 coefficients in `f` (where the products go by the transforms on pieces), the pieces of the inverse and of `f`'s low coefficients are transformed once, and each remainder's two products transform only the other operand and apply the theorem to the coefficients they keep (the quotient's low `lq`, the remainder's low `deg f`). `pow_mod` uses sliding windows of up to 6 bits of the exponent (the odd powers computed once), and for the base `x` squares and shifts, a shift and one step of the division costing `O(deg f)`. Over the rationals `pow_mod` works on the integers: `f = F/d_f` and the base `B/d_b` over common denominators, each value kept as `N/(|l|^E·d_b^P)`, a product an integer product and its remainder modulo `F` the pseudo-remainder with the fewest factors of `l`, and only the `deg f` coefficients of the result brought to lowest terms, once. `compose_mod` is Brent and Kung's: `m = ⌈√(deg p + 1)⌉` powers `g⁰…gᵐ⁻¹` modulo `f`, each block of `m` coefficients of `p` evaluated at `g` as rows of a product of matrices (each entry a dot product reduced once), and the blocks joined by Horner's rule in `gᵐ`: about `2√deg p` products modulo `f`; modulo a word all of it on vectors of residues reused from step to step (one pass on the stack for an `f` of up to 16 coefficients).
- **Composition**: `p = lo + xʰ·hi` gives `p(g) = lo(g) + g^h·hi(g)`, `h` the largest power of 2 below the length, with the powers `g^(2^i)` computed once and Horner's rule for pieces of up to 8 coefficients: `O(M(n·deg g) log n)` instead of Horner's `n` products of growing length.
- **Evaluation at one point**: modulo a word Horner's rule with Shoup's quotient of `x`, in two chains (the even and the odd coefficients, in `x²`) that the core overlaps. Over the integers, from 32 coefficients, the point is taken as one point of `eval_many` (below): Horner's rule on word blocks on limbs, and divide and conquer when the value grows past a few thousand bits.

### Greatest common divisors and resultants

- **Modulo a word `n`.** Euclid's algorithm in Montgomery's form for an odd `n`: each step clears two coefficients of the dividend at a time, the two products a coefficient summed and reduced by one REDC (two multiplications), for `n < 2^63`. A step whose quotient has one or two coefficients and whose divisor is shorter than 600 takes the pseudo-remainder `lc(b)^(δ+1)·a - q·b` instead, three products summed before one REDC and no inverse (an inverse of a word costs about what a pass over 500 coefficients does, a quarter of the time at degree 1000 otherwise); for `n < 2^62`. Each remainder is then the true one times a known factor; the resultant divides the leading coefficients by those factors at the end, all inverted at once by Montgomery's trick, and the product of the leading coefficients is inverted once to check that they were units. An even `n` takes the same steps with remainders by the reciprocal (Möller and Granlund's division of two words by one), one reduction for two rows.
- **Modulo `n` of `k ≥ 2` limbs.** Euclid's algorithm on the delayed-reduction schoolbook remainders of the division (the remainder's coefficients kept as sums and divided by `n` once), and for divisors shorter than 24 coefficients the pseudo-remainder step without an inverse (three products of a coefficient summed in `2k + 2` limbs, one reduction by the division by `n`), the factors tracked and inverted together as on words. The half gcd's own divisions are the two-row steps.
- **The half gcd.** Past a degree, Thull and Yap's formulation of the Knuth–Schönhage algorithm, over any of the coefficient rings: for `a` of the degree `n` and `m = ⌈n/2⌉`, the quotients of the top halves `a div xᵐ` and `b div xᵐ` are those of `a` and `b` as long as the divisor keeps half the degree of the top halves, so the first half of the steps comes from a recursive call on the top halves, one step is a division, and the second half a recursive call on the top part of the new pair; the matrices of the steps apply to the low parts, and the whole takes `O(M(n) log n)`. `gcd` takes a half gcd and a division in turn until the degree falls below the crossover, then Euclid's algorithm; `xgcd` keeps the product of the matrices, whose second column gives the cofactors (the tail by Euclid's algorithm with the matrix in Montgomery's form); the resultant records the degree and the leading coefficient of every divisor the half gcd uses, at their true degrees (those of the top parts, shifted), and takes `res(r₀, r₁) = (-1)^(d₀d₁) l₁^(d₀ - d₂) res(r₁, r₂)` over them. The products are those above (the schoolbook method, Kronecker substitution, the number-theoretic transforms on words; the schoolbook method, splitting and Kronecker substitution on limbs). The crossovers, measured with them, are on words by the size of `n`, as the half gcd's products take more transform primes as `n` grows while a step of Euclid's costs about the same for every `n` below `2^63`: degree 1000 for `gcd` and the resultant and 600 for `xgcd` up to 26 bits, 1900 and 950 for 27 to 34 bits, 3000 and 1500 for 35 to 63 bits, and 2000 and 850 from `2^63` (at 17 bits and the degree 3000 the gcd takes 1.30 ms, against 2.44 ms with the crossover of 35 to 63 bits); on limbs 250 and 80; the recursion's base below degree 100 (words) and 60 (limbs).
- **Over the integers, the greatest common divisor** of the primitive parts by one of two methods. The multimodular algorithm (by default): the monic gcd modulo primes below `2^62`, scaled to the gcd `l` of the leading coefficients, joined by the Chinese remainder theorem in the symmetric residues (one prime at a time, Garner's step) until the images stop changing; the primitive part is then kept when it divides both. A prime that divides a leading coefficient is passed over, one whose gcd has a higher degree is unlucky and passed over, one of a lower degree starts the images again, and one of degree 0 proves the two coprime (the common case of random polynomials: one prime). The heuristic gcd (Char, Geddes and Gonnet) for the shortest polynomials (up to 8 coefficients) and for long ones of short coefficients (at most 48 bits from 200 coefficients), where one gcd of integers beats the steps modulo two primes: both polynomials evaluated at `ξ = 2^s`, `s` past the largest coefficient, the gcd of the two integers read back in balanced digits of base `ξ`, its primitive part kept when it divides both; for `ξ` above twice the smaller of the two largest coefficients such a divisor is the gcd. When it gives up (three sizes of `ξ`), the multimodular algorithm.
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
- `real_root_intervals()` (and `count_real_roots`, `real_roots`) choose for the positive and the negative roots of each factor apart: continued fractions when the bound `2^s` of those roots has `s ≥ 7` (roots far apart in a long interval: Wilkinson's have `s ≥ 8` from the degree 10, Chebyshev's at most 5 to the degree 200, random ones at most 4), else Descartes's rule, whose nodes at the depth `2·log2 n + 16` and below (past the depths that separate the roots of random and Chebyshev's polynomials) go on by continued fractions, from `(y + 1)ⁿ·q(1/(y + 1))` and the transformation `2^(s−k)·(c·y + c + 1)/(y + 1)`: Mignotte's close pairs then cost what continued fractions cost, not one level of the bisection per bit of their distance. (Up to the degree 100 continued fractions are also faster on random polynomials, by 1.1 to 1.8 times, but slower on Chebyshev's, 1.5 times at 20 and 3 at 50, so the degree alone does not choose.)
- The isolating intervals of the factors are then bisected (the signs at the dyadic midpoints by the evaluation in fixed point, exactly at other rationals) until no two overlap and no end of one is a root; the sign just right of the left end of an interval is found only when a bisection or the refinement needs it, and the one just left of the right end is its opposite, the root inside being simple.
- `real_roots(prec)`: an isolating interval of one sign over more than two binades (as from the bound of the roots) is first halved in its exponent, at the power of two between its ends', down to a few binades. Each isolating interval is narrowed by Newton's method on dyadic points safeguarded by bisection (an iterate outside the interval halves it; one whose step shrank less than 8 times, linear convergence as towards a close pair of roots, cuts the interval there), at a precision at least 32 bits finer than the interval, the last iterate proven by the signs at `y ± 2|c|` (`c` its correction), then at doubling precisions to `prec + 32` bits (at least 96), one step at each (a step much larger than the last level's error is linear convergence: 16 bisections and Newton's method again from the narrowed interval); the value to `p + κ` bits relative to its bound, the derivative to about `p/2 + κ` bits (by Horner's rule together, or each by rectangular splitting when wide), where `κ` is the conditioning found as it goes: the bits by which the uncertainty of the correction `v/d` exceeds `2^−p·|x|` are added to it, and it is passed on from one root of a factor to the next (about 120 bits near ±1 for Chebyshev's `T₁₀₀`). A value 0 to its error and modulo `2^61 − 1` is tested exactly: the point is then the root. The value rounded to `prec` bits is proven correctly rounded by the signs of the factor at the ends of the interval of the reals that round to it (clipped to the isolating interval): different signs put the root inside; a zero sign at an end means the root is that end, a midpoint, rounded ties to even; the same signs cut the interval there and raise the precision by 32 bits.
- `count_real_roots()`: for each square-free factor of the degree 12 or more, the first Sturm remainder `−rem(f, f')` (one pseudo-division, `O(n)` operations); when its degree is at most `min(16, n/4)` (`xⁿ` plus a polynomial of a low degree, as Mignotte's) Sturm's sequence of pseudo-remainders with the true remainders' signs, each divided by its content (a short divisor's pseudo-remainder with the dividend's coefficients taken into a window of `m + 1` one at a time, each times the power of the leading coefficient the steps before would have given it: `O(n·m)` operations), and the count `V(−∞) − V(+∞)`; otherwise the isolation's method, counting its intervals without building them; plus one for the root 0.

### Factorisation

- **Square-free parts.** Over the integers, Yun's method on the primitive part — `a₀ = gcd(f, f')`, then `aᵢ = gcd(bᵢ, cᵢ - bᵢ')` with exact divisions — after a quick test: a polynomial square-free modulo one of three primes from `2^31` up (not dividing the leading coefficient, unlikely to divide the discriminant) is square-free, which settles most inputs with one greatest common divisor modulo a word. Modulo `p`, Yun's method on what the derivative sees and, for the rest, whose derivative is 0, the `p`-th root `Σ aᵢ x^(ip) = (Σ aᵢ xⁱ)^p`, recursively with the multiplicities times `p`.
- **Distinct degrees.** For a small `p` and a short polynomial (the degree times the bits of `p` at most 48), `x^(p^i) mod f` one `p`-th power after another, with a greatest common divisor with `x^(p^i) - x` each time. Otherwise the baby steps and giant steps of Kaltofen and Shoup (von zur Gathen and Shoup's interval method): `l = ⌈√(n/2)⌉` baby steps `hᵢ = x^(p^i)` and the giant steps `Hⱼ = x^(p^(lj))`, the factors of the degrees in `(l(j-1), lj]` from the greatest common divisor with `Πᵢ (Hⱼ - hᵢ) mod f`, and each degree from those with `Hⱼ - hᵢ`; it stops when what is left is too short to be reducible. Each giant step is a Brent and Kung modular composition with the powers of `x^(p^l)`, each baby step one with the powers of `x^p`, or for a small `p` (its bits at most `2√(n/(l-1))`) the `p`-th power of the last, which costs fewer products modulo `f`. One table of powers serves every composition with the same `h`, so its rows are `m = √(uses·n)` instead of `√n` for `uses` compositions: the powers cost one product modulo `f` each and a composition `⌈n/m⌉ - 1` (its matrix product costs `n²` products of residues whatever `m`), which takes a third off the factorisation modulo a 64-bit prime at degree 300 to 1,000.
- **Equal degrees.** Cantor and Zassenhaus's random splitting by `gcd(g, a^((p^d - 1)/2) - 1)`, for `p = 2` by the trace `a + a² + … + a^(2^(d-1))`. For a large `p` (its bits squared above four times the degree), `a^((p^d-1)/2) = N^((p-1)/2)` with `N = a·a^p·…·a^(p^(d-1))`, the `a^(p^i)` by compositions with `x^p`, which the distinct-degree step already has: `d - 1` compositions and the bits of `p` in squarings, instead of `d` times the bits.
- **Berlekamp.** The matrix `Q` of the Frobenius map `a ↦ aᵖ` on `F_p[x]/(f)` from `x^p mod f` and its powers, and the kernel of `Q - I` (its dimension the number `r` of factors) by Gaussian elimination to echelon form, the entries below the pivots kept as unreduced sums of products in one, two or three words (each takes at most one product a pivot) and reduced only as their column or row is reached, then back-substitution for each free column. Its elements split `f`: for `p` up to 11 by greatest common divisors with `v - s` for every `s` of `F_p`; above, a random element `v` of the kernel is a constant modulo each factor, so its minimal polynomial modulo a part `u` — that of the sequence `⟨w, vⁱ mod u⟩` for a random `w` and `i < 2r`, by Berlekamp and Massey's algorithm — is `Π (x - c)` over its distinct values, and its roots `c` (of a polynomial of degree at most `r`) split `u` by `gcd(u, v - c)`, all of them at once (NTL's way, instead of a power `v^((p-1)/2)` modulo each part). `factor` takes Berlekamp's method itself for `p` up to 7 and parts of degree up to 600 (`p` = 2, 3) or 300 (5, 7), where its elimination in a word costs less than the distinct-degree steps (half the time at `p = 3` and degree 100 to 300).
- **Irreducibility modulo p.** Ben-Or's test: `gcd(f, x^(p^i) - x) = 1` for every `i` up to `n/2`, the roots (`i = 1`) first, then the baby steps and giant steps of the distinct-degree factorisation, one greatest common divisor with the product of the `l` differences `Hⱼ - hᵢ` for each interval of degrees, stopping at the first common factor (FLINT's `nmod_poly_is_irreducible_ddf` is the same idea).
- **Roots.** For each square-free part, `gcd(g, x^p - x)` split into factors of degree 1 by Cantor and Zassenhaus's method; for `p` up to 64 or the degree, every residue tried by synthetic division, each root divided out as often as it divides, with no greatest common divisor at all (a tenth of the time of FLINT's at `p = 3` and degree 1,000).
- **Arithmetic modulo p.** The products, remainders and powers are those of the polynomials modulo `n` ([above](#products-and-division-1)): modulo a word, the schoolbook products with one reduction a coefficient and the number-theoretic transforms on the coefficients, and modulo a fixed `f` the remainders with the inverse of its reversal kept; modulo several limbs, Kronecker substitution with one reduction a coefficient and the delayed-reduction schoolbook division. Where the factorisation measured a choice of its own faster, it keeps it: the transforms of a product modulo a word start from 64 to 150 coefficients by the size of `p` (the schoolbook method beats them up to there, as their length doubles past 64 coefficients); for `p` of up to 13 bits Kronecker substitution into the crate's product of integers takes the lengths up to 300 to 2,048 coefficients (its fields of `2·bits(p) + log₂ n` bits are short, and it beats the transforms 2 to 3 times at `p = 3`), and the remainders modulo `f` go by the Newton inverse of the reversed `f` on those products; for `f` of up to 150 coefficients the remainders are a delayed-reduction schoolbook division by the monic `f` (no product by the leading coefficient's inverse), and for `f` of up to 32 coefficients (80 above 32-bit primes, 48 for squares) a product modulo `f` takes one pass: the sums of products of the operands unreduced, the division run on those sums, one reduction a coefficient of the product in all, 1.3 to 1.6 times faster than a product and a division; the inverse of a word residue is the extended Euclidean algorithm with the cofactors kept as magnitudes and the quotients 1 to 3 by subtractions. The matrix products of the Brent–Kung composition are dot products of the blocks of `g` with the columns of the powers, summed in one, two or three words and reduced once, modulo several limbs in `2k + 1` limbs and reduced once. The greatest common divisors are those of the section [above](#greatest-common-divisors-and-resultants-1): modulo a word Euclid's algorithm in Montgomery's form and the half gcd (Euclid's algorithm on the delayed-reduction remainders for operands of up to 3 coefficients, up to 6 modulo several limbs), over the integers the multimodular gcd (only for inputs that are not square-free), and the cofactors of the Hensel tree by `ModPolynomial::xgcd`.
- **Over the integers.** The factors of a square-free primitive polynomial by Zassenhaus's method. The power of `x` comes off first. One good prime from 3 up (not dividing the leading coefficient, the polynomial square-free modulo it), and more, up to six, while the fewest factors are more than 12 (all six when the bound of the coefficients of a factor has 12 bits a degree or more, where the lifting is long), give their distinct-degree factorisations: the number of factors and their degrees, whose subset sums intersect into the degrees a true factor can have — one factor, or no degree left but 0 and `n`, and the polynomial is irreducible. (More primes prune more but cost a distinct-degree factorisation each: one prime takes half the time of three on random polynomials and products of two at degree 100 to 300, while random polynomials of 1,000-bit coefficients take a tenth of the time with six, which prove them irreducible without a lift, and `x^210 - 1` a tenth with up to six, which bring its local factors down.) The prime with the fewest factors is split completely, and its factors lifted at once in a binary tree (halved by degree) by Hensel's lemma, quadratically (von zur Gathen and Gerhard's Algorithm 15.10 for monic factors; from two limbs of `pᵃ` up, the corrections of a step from `m` to `m₂` are made modulo `m₂/m`, from the errors divided exactly by the odd `m` from the low end, so that only `g·h` and `s·g + t·h` take the full precision), to `pᵃ > 2B` with Mignotte's bound `B = |lc|·C(K, ⌊K/2⌋)·(‖f‖ + |lc|)` for the factors of degree up to `K = ⌊n/2⌋`. The recombination tries the subsets of one lifted factor, then of two, …, up to half (a subset whose degree is above half is tried by its complement, so that the bound of half the degree holds): its degree against the allowed set; the power sums `p₁` and `p₂` of its roots, the sums of those of its factors, scaled by `lc` and `lc²` and kept within the bounds of the roots of `f` (Mignotte's and Fujiwara's; the second catches the even factors of an even polynomial such as the Swinnerton-Dyer polynomials, whose traces are 0), on two words as long as `pᵃ < 2^127`, the partial sums kept from one subset to the next; its constant coefficient dividing `lc·f(0)`; and only then the product and a trial division (the schoolbook one, stopping at the first quotient coefficient that does not divide, for a candidate of degree up to 4; the exact division by Kronecker substitution beyond). When half the factors make a subset, only those with the first are tried, the others being their complements.
- The time of Zassenhaus's recombination grows as `2^r` for `r` lifted factors when the true factors are few, as for the Swinnerton-Dyer polynomials (`r = 2^(k-1)` for every prime): 16 factors at degree 32 take about 0.6 milliseconds, 32 at degree 64 would take some 2·10⁹ subsets, about 40 seconds; FLINT and PARI switch to van Hoeij's method there. Van Hoeij's method, with lattice reduction, comes with [#23](https://github.com/nettrash/a/issues/23).

## Testing

Besides the tests of `lib/tests/math/poly`, the module was fuzzed differentially against FLINT 3.6 (with Arb) and PARI/GP 2.19: a harness in Rust with an oracle in C, every exact operation of `Polynomial<Number>`, `Polynomial<Rational>` and `ModPolynomial` and the roots of every type, on degenerate inputs (the zero polynomial, constants, zeros at the top, sparse `xⁿ ± c`), repeated and known factors, pairs equal, negated, coprime, divisible or with a common factor, the families `xⁿ − 1`, Wilkinson's, Mignotte's, Chebyshev's and Swinnerton-Dyer's, lengths about every crossover up to 4,001 coefficients, coefficients of 1 to 5,000 bits, and moduli from 2 to primes and composites of 512 bits. Its first passes, 14.9 million cases and 1.7 million targeted ones, found twelve faults, in the roots, the arithmetic over the integers and the rationals and the evaluation at one point (wrong or missing roots of ill-conditioned float polynomials of degree 100 and more, a panic of `div_exact` by a divisor much wider than the dividend, hangs of the certified and real roots, and slow cases), each fixed with a test of its own. The last pass compared 16,744,116 cases with no wrong result, panic or hang (two roots of degree 6 whose Arb balls were too wide to compare with, every root here of a backward error below `1.3·10⁻¹⁶` and inside its ball); of its seven slow cases, the pseudo-divisions and the power modulo a polynomial over the rationals were then made fast (above), and the real roots of a short polynomial with huge coefficients still take about 1.1 s, against Arb's 18 ms. Not compared: NTL (FLINT and PARI cover every operation), the arithmetic of floats and complex floats (which would need tolerances), `companion`, `Display`, the shifts, `truncate`, `reverse` and `scale`, and gcds modulo a composite (not unique: checked by their properties).

## Measured

Measured on an Apple M4 Max on 2026-11-16 between 18:17 and 19:16 (the factorisation, the roots, evaluation and interpolation, and the products and division in turn; load averages 1.7 to 3.7 at the start and the end of each, the machine otherwise idle), and the greatest common divisors again in three runs between 19:32 and 19:53 after a change to their crossovers, with nothing else running, one thread, `env -u MallocNanoZone`, the harnesses built with fat LTO and one codegen unit, against FLINT 3.6, NTL 11.6 and PARI/GP 2.19 (Homebrew's builds, called through C and C++ shims that build their inputs before the clock starts; PARI on one thread) and, for the roots, Arb in FLINT. Every round draws fresh random inputs and turns the contenders' order, and every harness compares each reference's result with `a`'s before it times them (exactly modulo a word, by a fingerprint modulo `2⁶¹ − 1` otherwise, and the numbers of factors and roots); each section gives its rounds and how its medians are taken. Times are in ns, µs, ms or s a call; a ratio `a/x` is `a`'s time over the reference's: below 1, `a` is faster.

### Products and division

The median of five rounds. FLINT's `nmod_poly`, `fmpz_poly`, `fmpz_mod_poly` and `fmpq_poly`; NTL's `zz_pX` (below `2⁶⁰`), `ZZ_pX` and `ZZX`; PARI's `Flx`, `ZX` and `FpX`. FLINT's and NTL's timing loops reuse their result's storage, where `a`'s operators return a new polynomial each time (an allocation, about 7 ns, and the modulus's reference count, about 4 ns), which shows below about 8 coefficients; and `Polynomial<Number>` holds each coefficient in its own vector of bytes, which every product reads and builds anew (about 10 ns a coefficient), where FLINT's `fmpz` keeps a small one in a word.

Products modulo a word, `&a * &b`, both operands of the length and their residues random:

`n = 65,521`:

| length | `a` | FLINT | PARI | NTL | a/FLINT | a/PARI | a/NTL |
|---|---|---|---|---|---|---|---|
| 1 | 19 ns | 6 ns | 11 ns | 7 ns | 3.36 | 1.69 | 2.85 |
| 2 | 21 ns | 11 ns | 13 ns | 10 ns | 1.85 | 1.65 | 2.03 |
| 4 | 28 ns | 21 ns | 20 ns | 17 ns | 1.32 | 1.41 | 1.70 |
| 8 | 48 ns | 50 ns | 42 ns | 39 ns | 0.96 | 1.16 | 1.23 |
| 16 | 70 ns | 88 ns | 114 ns | 143 ns | 0.80 | 0.61 | 0.49 |
| 32 | 140 ns | 216 ns | 262 ns | 443 ns | 0.65 | 0.53 | 0.32 |
| 64 | 360 ns | 441 ns | 644 ns | 1.34 µs | 0.82 | 0.56 | 0.27 |
| 128 | 1.10 µs | 1.19 µs | 1.72 µs | 4.18 µs | 0.93 | 0.64 | 0.26 |
| 256 | 2.94 µs | 2.38 µs | 4.58 µs | 12.79 µs | 1.24 | 0.64 | 0.23 |
| 512 | 5.61 µs | 5.04 µs | 12.61 µs | 19.52 µs | 1.11 | 0.44 | 0.29 |
| 1000 | 11.64 µs | 11.19 µs | 33.82 µs | 41.99 µs | 1.04 | 0.34 | 0.28 |
| 2000 | 23.80 µs | 23.73 µs | 85.02 µs | 90.19 µs | 1.00 | 0.28 | 0.26 |
| 5000 | 95.28 µs | 73.68 µs | 307.50 µs | 266.60 µs | 1.29 | 0.31 | 0.36 |
| 10000 | 207.92 µs | 153.00 µs | 658.83 µs | 566.33 µs | 1.36 | 0.32 | 0.37 |
| 30000 | 445.13 µs | 468.12 µs | 2.09 ms | 1.79 ms | 0.95 | 0.21 | 0.25 |
| 100000 | 1.88 ms | 1.84 ms | 10.72 ms | 7.50 ms | 1.02 | 0.18 | 0.25 |
| 300000 | 7.85 ms | 13.44 ms | 36.37 ms | 25.95 ms | 0.58 | 0.22 | 0.30 |
| 1000000 | 18.31 ms | 43.51 ms | 170.07 ms | 87.35 ms | 0.42 | 0.11 | 0.21 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | PARI | NTL | a/FLINT | a/PARI | a/NTL |
|---|---|---|---|---|---|---|---|
| 1 | 18 ns | 5 ns | 11 ns | 6 ns | 3.46 | 1.58 | 3.16 |
| 2 | 23 ns | 11 ns | 17 ns | 12 ns | 2.02 | 1.33 | 1.87 |
| 4 | 32 ns | 28 ns | 47 ns | 20 ns | 1.15 | 0.70 | 1.64 |
| 8 | 92 ns | 68 ns | 164 ns | 49 ns | 1.36 | 0.56 | 1.88 |
| 16 | 206 ns | 178 ns | 371 ns | 181 ns | 1.16 | 0.55 | 1.14 |
| 32 | 537 ns | 497 ns | 938 ns | 548 ns | 1.08 | 0.57 | 0.98 |
| 64 | 1.73 µs | 1.28 µs | 2.73 µs | 1.66 µs | 1.35 | 0.63 | 1.05 |
| 128 | 3.00 µs | 3.26 µs | 7.23 µs | 5.10 µs | 0.92 | 0.41 | 0.59 |
| 256 | 5.72 µs | 8.94 µs | 18.65 µs | 15.49 µs | 0.64 | 0.31 | 0.37 |
| 512 | 11.22 µs | 18.92 µs | 54.04 µs | 29.13 µs | 0.59 | 0.21 | 0.39 |
| 1000 | 23.27 µs | 41.00 µs | 127.37 µs | 63.24 µs | 0.57 | 0.18 | 0.37 |
| 2000 | 47.97 µs | 88.24 µs | 308.15 µs | 134.00 µs | 0.54 | 0.16 | 0.36 |
| 5000 | 188.81 µs | 271.15 µs | 841.00 µs | 396.20 µs | 0.70 | 0.22 | 0.48 |
| 10000 | 509.53 µs | 586.00 µs | 1.63 ms | 839.50 µs | 0.87 | 0.31 | 0.61 |
| 30000 | 1.10 ms | 1.82 ms | 6.38 ms | 2.68 ms | 0.60 | 0.17 | 0.41 |
| 100000 | 4.69 ms | 6.99 ms | 23.99 ms | 11.31 ms | 0.67 | 0.20 | 0.41 |
| 300000 | 19.54 ms | 23.68 ms | 149.25 ms | 38.53 ms | 0.83 | 0.13 | 0.51 |
| 1000000 | 45.13 ms | 76.42 ms | 401.16 ms | 129.23 ms | 0.59 | 0.11 | 0.35 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 1 | 18 ns | 5 ns | 11 ns | 3.48 | 1.59 |
| 2 | 23 ns | 12 ns | 17 ns | 1.97 | 1.34 |
| 4 | 33 ns | 28 ns | 47 ns | 1.15 | 0.69 |
| 8 | 118 ns | 92 ns | 200 ns | 1.28 | 0.59 |
| 16 | 264 ns | 243 ns | 495 ns | 1.09 | 0.53 |
| 32 | 670 ns | 707 ns | 1.26 µs | 0.95 | 0.53 |
| 64 | 2.05 µs | 1.64 µs | 3.58 µs | 1.25 | 0.57 |
| 128 | 3.84 µs | 4.78 µs | 10.19 µs | 0.80 | 0.38 |
| 256 | 7.30 µs | 9.11 µs | 23.81 µs | 0.80 | 0.31 |
| 512 | 14.21 µs | 19.05 µs | 67.68 µs | 0.75 | 0.21 |
| 1000 | 29.10 µs | 42.27 µs | 173.62 µs | 0.69 | 0.17 |
| 2000 | 60.76 µs | 87.92 µs | 335.82 µs | 0.69 | 0.18 |
| 5000 | 238.43 µs | 270.92 µs | 998.50 µs | 0.88 | 0.24 |
| 10000 | 513.73 µs | 583.67 µs | 2.10 ms | 0.88 | 0.24 |
| 30000 | 1.13 ms | 1.86 ms | 9.41 ms | 0.61 | 0.12 |
| 100000 | 5.74 ms | 7.04 ms | 33.88 ms | 0.82 | 0.17 |
| 300000 | 23.81 ms | 23.77 ms | 161.31 ms | 1.00 | 0.15 |
| 1000000 | 54.49 ms | 77.09 ms | 480.96 ms | 0.71 | 0.11 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 1 | 18 ns | 5 ns | 11 ns | 3.42 | 1.56 |
| 2 | 23 ns | 12 ns | 17 ns | 1.93 | 1.33 |
| 4 | 39 ns | 37 ns | 47 ns | 1.06 | 0.83 |
| 8 | 99 ns | 92 ns | 206 ns | 1.07 | 0.48 |
| 16 | 224 ns | 244 ns | 508 ns | 0.92 | 0.44 |
| 32 | 584 ns | 715 ns | 1.28 µs | 0.82 | 0.46 |
| 64 | 1.80 µs | 1.69 µs | 3.76 µs | 1.07 | 0.48 |
| 128 | 3.86 µs | 4.87 µs | 10.26 µs | 0.79 | 0.38 |
| 256 | 7.33 µs | 9.18 µs | 24.39 µs | 0.80 | 0.30 |
| 512 | 14.29 µs | 19.59 µs | 67.93 µs | 0.73 | 0.21 |
| 1000 | 29.53 µs | 42.64 µs | 173.55 µs | 0.69 | 0.17 |
| 2000 | 60.43 µs | 87.24 µs | 331.92 µs | 0.69 | 0.18 |
| 5000 | 237.18 µs | 270.38 µs | 995.00 µs | 0.88 | 0.24 |
| 10000 | 510.42 µs | 581.67 µs | 2.09 ms | 0.88 | 0.24 |
| 30000 | 1.35 ms | 1.85 ms | 10.08 ms | 0.73 | 0.13 |
| 100000 | 5.75 ms | 7.03 ms | 36.14 ms | 0.82 | 0.16 |
| 300000 | 23.73 ms | 23.79 ms | 162.66 ms | 1.00 | 0.15 |
| 1000000 | 54.43 ms | 76.20 ms | 481.06 ms | 0.71 | 0.11 |

Squares modulo a word, `&a * &a`:

`n = 65,521`:

| length | `a` | FLINT | PARI | NTL | a/FLINT | a/PARI | a/NTL |
|---|---|---|---|---|---|---|---|
| 1 | 14 ns | 6 ns | 9 ns | 8 ns | 2.59 | 1.67 | 1.81 |
| 2 | 18 ns | 11 ns | 11 ns | 10 ns | 1.68 | 1.70 | 1.91 |
| 4 | 22 ns | 17 ns | 16 ns | 14 ns | 1.27 | 1.34 | 1.55 |
| 8 | 46 ns | 35 ns | 32 ns | 28 ns | 1.31 | 1.42 | 1.65 |
| 16 | 68 ns | 74 ns | 77 ns | 70 ns | 0.93 | 0.89 | 0.98 |
| 32 | 138 ns | 212 ns | 232 ns | 257 ns | 0.65 | 0.60 | 0.54 |
| 64 | 358 ns | 446 ns | 532 ns | 800 ns | 0.80 | 0.67 | 0.45 |
| 128 | 1.09 µs | 1.19 µs | 1.36 µs | 2.51 µs | 0.92 | 0.80 | 0.44 |
| 256 | 2.41 µs | 1.71 µs | 3.73 µs | 7.76 µs | 1.40 | 0.65 | 0.31 |
| 512 | 4.53 µs | 3.65 µs | 10.49 µs | 14.18 µs | 1.24 | 0.43 | 0.32 |
| 1000 | 9.23 µs | 8.01 µs | 26.87 µs | 30.06 µs | 1.15 | 0.34 | 0.31 |
| 2000 | 18.99 µs | 16.73 µs | 78.56 µs | 65.00 µs | 1.14 | 0.24 | 0.29 |
| 5000 | 72.19 µs | 52.49 µs | 215.53 µs | 194.50 µs | 1.38 | 0.33 | 0.37 |
| 10000 | 152.46 µs | 108.92 µs | 461.00 µs | 410.60 µs | 1.40 | 0.33 | 0.37 |
| 30000 | 335.63 µs | 326.08 µs | 1.48 ms | 1.26 ms | 1.03 | 0.23 | 0.27 |
| 100000 | 1.40 ms | 1.27 ms | 7.90 ms | 5.66 ms | 1.10 | 0.18 | 0.25 |
| 300000 | 5.74 ms | 9.61 ms | 26.68 ms | 18.01 ms | 0.60 | 0.22 | 0.32 |
| 1000000 | 13.35 ms | 30.76 ms | 130.94 ms | 59.06 ms | 0.43 | 0.10 | 0.23 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | PARI | NTL | a/FLINT | a/PARI | a/NTL |
|---|---|---|---|---|---|---|---|
| 1 | 15 ns | 6 ns | 10 ns | 9 ns | 2.70 | 1.50 | 1.75 |
| 2 | 21 ns | 12 ns | 15 ns | 11 ns | 1.76 | 1.43 | 1.88 |
| 4 | 31 ns | 39 ns | 30 ns | 16 ns | 0.80 | 1.04 | 1.93 |
| 8 | 98 ns | 75 ns | 96 ns | 34 ns | 1.31 | 1.03 | 2.86 |
| 16 | 212 ns | 183 ns | 307 ns | 97 ns | 1.16 | 0.69 | 2.18 |
| 32 | 497 ns | 473 ns | 777 ns | 343 ns | 1.05 | 0.64 | 1.45 |
| 64 | 1.27 µs | 1.32 µs | 2.14 µs | 1.05 µs | 0.96 | 0.59 | 1.21 |
| 128 | 2.53 µs | 3.43 µs | 5.79 µs | 3.28 µs | 0.74 | 0.44 | 0.77 |
| 256 | 4.71 µs | 6.84 µs | 16.50 µs | 10.11 µs | 0.69 | 0.29 | 0.47 |
| 512 | 8.88 µs | 14.23 µs | 41.46 µs | 21.14 µs | 0.62 | 0.21 | 0.42 |
| 1000 | 18.21 µs | 31.04 µs | 104.92 µs | 45.21 µs | 0.59 | 0.17 | 0.40 |
| 2000 | 37.17 µs | 64.90 µs | 215.50 µs | 96.56 µs | 0.57 | 0.17 | 0.38 |
| 5000 | 141.59 µs | 202.82 µs | 606.00 µs | 287.33 µs | 0.70 | 0.23 | 0.49 |
| 10000 | 375.34 µs | 443.12 µs | 1.23 ms | 611.00 µs | 0.85 | 0.30 | 0.61 |
| 30000 | 824.41 µs | 1.40 ms | 4.69 ms | 1.87 ms | 0.59 | 0.18 | 0.44 |
| 100000 | 3.49 ms | 5.26 ms | 17.48 ms | 7.91 ms | 0.66 | 0.20 | 0.44 |
| 300000 | 14.24 ms | 17.80 ms | 115.45 ms | 27.43 ms | 0.80 | 0.12 | 0.52 |
| 1000000 | 32.74 ms | 56.72 ms | 273.55 ms | 88.42 ms | 0.58 | 0.12 | 0.37 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 1 | 16 ns | 6 ns | 10 ns | 2.75 | 1.57 |
| 2 | 21 ns | 12 ns | 14 ns | 1.78 | 1.46 |
| 4 | 31 ns | 39 ns | 30 ns | 0.79 | 1.04 |
| 8 | 108 ns | 92 ns | 95 ns | 1.18 | 1.14 |
| 16 | 233 ns | 238 ns | 406 ns | 0.98 | 0.57 |
| 32 | 550 ns | 616 ns | 1.05 µs | 0.89 | 0.52 |
| 64 | 1.43 µs | 1.60 µs | 2.92 µs | 0.89 | 0.49 |
| 128 | 3.12 µs | 4.37 µs | 8.36 µs | 0.71 | 0.37 |
| 256 | 5.81 µs | 6.90 µs | 21.55 µs | 0.84 | 0.27 |
| 512 | 11.12 µs | 14.47 µs | 54.88 µs | 0.77 | 0.20 |
| 1000 | 22.38 µs | 31.35 µs | 138.29 µs | 0.71 | 0.16 |
| 2000 | 46.08 µs | 65.77 µs | 251.31 µs | 0.70 | 0.18 |
| 5000 | 178.05 µs | 204.35 µs | 721.40 µs | 0.87 | 0.25 |
| 10000 | 371.67 µs | 444.25 µs | 1.49 ms | 0.84 | 0.25 |
| 30000 | 841.56 µs | 1.41 ms | 6.23 ms | 0.60 | 0.14 |
| 100000 | 4.24 ms | 5.31 ms | 28.39 ms | 0.80 | 0.15 |
| 300000 | 17.22 ms | 17.63 ms | 124.91 ms | 0.98 | 0.14 |
| 1000000 | 39.31 ms | 57.27 ms | 343.14 ms | 0.69 | 0.11 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 1 | 16 ns | 6 ns | 10 ns | 2.74 | 1.61 |
| 2 | 21 ns | 12 ns | 14 ns | 1.81 | 1.50 |
| 4 | 39 ns | 36 ns | 30 ns | 1.09 | 1.31 |
| 8 | 89 ns | 94 ns | 95 ns | 0.95 | 0.94 |
| 16 | 194 ns | 234 ns | 419 ns | 0.83 | 0.46 |
| 32 | 463 ns | 616 ns | 1.08 µs | 0.75 | 0.43 |
| 64 | 1.21 µs | 1.76 µs | 2.95 µs | 0.69 | 0.41 |
| 128 | 3.17 µs | 4.38 µs | 8.29 µs | 0.72 | 0.38 |
| 256 | 5.80 µs | 6.96 µs | 22.22 µs | 0.83 | 0.26 |
| 512 | 11.11 µs | 14.55 µs | 57.45 µs | 0.76 | 0.19 |
| 1000 | 22.20 µs | 31.21 µs | 141.93 µs | 0.71 | 0.16 |
| 2000 | 46.58 µs | 65.71 µs | 250.53 µs | 0.71 | 0.19 |
| 5000 | 176.64 µs | 202.29 µs | 712.20 µs | 0.87 | 0.25 |
| 10000 | 374.26 µs | 443.00 µs | 1.51 ms | 0.84 | 0.25 |
| 30000 | 997.92 µs | 1.40 ms | 6.09 ms | 0.71 | 0.16 |
| 100000 | 4.23 ms | 5.29 ms | 27.36 ms | 0.80 | 0.15 |
| 300000 | 17.17 ms | 17.73 ms | 125.55 ms | 0.97 | 0.14 |
| 1000000 | 39.69 ms | 56.34 ms | 343.71 ms | 0.70 | 0.12 |

`div_rem` modulo a word, 2n by n coefficients:

`n = 65,521`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 2 | 72 ns | 31 ns | 42 ns | 22 ns | 2.31 | 1.71 | 3.24 |
| 4 | 100 ns | 50 ns | 71 ns | 41 ns | 1.99 | 1.41 | 2.46 |
| 8 | 146 ns | 82 ns | 138 ns | 95 ns | 1.78 | 1.06 | 1.54 |
| 16 | 246 ns | 183 ns | 269 ns | 220 ns | 1.35 | 0.91 | 1.12 |
| 32 | 438 ns | 576 ns | 777 ns | 580 ns | 0.76 | 0.56 | 0.75 |
| 64 | 849 ns | 2.13 µs | 2.68 µs | 2.05 µs | 0.40 | 0.32 | 0.41 |
| 128 | 2.20 µs | 8.47 µs | 9.93 µs | 7.74 µs | 0.26 | 0.22 | 0.28 |
| 256 | 7.72 µs | 15.21 µs | 36.83 µs | 19.21 µs | 0.51 | 0.21 | 0.40 |
| 512 | 37.23 µs | 27.50 µs | 136.55 µs | 50.96 µs | 1.35 | 0.27 | 0.73 |
| 1000 | 56.83 µs | 48.67 µs | 174.18 µs | 130.67 µs | 1.17 | 0.33 | 0.43 |
| 2000 | 115.81 µs | 98.93 µs | 360.27 µs | 334.45 µs | 1.17 | 0.32 | 0.35 |
| 5000 | 430.14 µs | 273.29 µs | 1.55 ms | 1.21 ms | 1.57 | 0.28 | 0.36 |
| 10000 | 905.89 µs | 571.43 µs | 3.31 ms | 2.72 ms | 1.59 | 0.27 | 0.33 |
| 30000 | 2.05 ms | 1.84 ms | 7.25 ms | 9.18 ms | 1.12 | 0.28 | 0.22 |
| 100000 | 8.42 ms | 7.06 ms | 34.00 ms | 41.64 ms | 1.19 | 0.25 | 0.20 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 2 | 153 ns | 115 ns | 95 ns | 102 ns | 1.33 | 1.62 | 1.50 |
| 4 | 187 ns | 164 ns | 120 ns | 188 ns | 1.14 | 1.56 | 0.99 |
| 8 | 269 ns | 255 ns | 185 ns | 380 ns | 1.05 | 1.45 | 0.71 |
| 16 | 450 ns | 452 ns | 323 ns | 1.30 µs | 0.99 | 1.39 | 0.35 |
| 32 | 872 ns | 898 ns | 848 ns | 4.22 µs | 0.97 | 1.03 | 0.21 |
| 64 | 2.20 µs | 2.27 µs | 2.75 µs | 10.76 µs | 0.97 | 0.80 | 0.20 |
| 128 | 6.05 µs | 6.96 µs | 10.34 µs | 28.74 µs | 0.87 | 0.59 | 0.21 |
| 256 | 19.27 µs | 20.84 µs | 37.56 µs | 79.46 µs | 0.92 | 0.51 | 0.24 |
| 512 | 66.44 µs | 64.42 µs | 141.75 µs | 205.06 µs | 1.03 | 0.47 | 0.32 |
| 1000 | 101.13 µs | 136.08 µs | 303.69 µs | 477.38 µs | 0.74 | 0.33 | 0.21 |
| 2000 | 207.60 µs | 301.17 µs | 579.14 µs | 1.21 ms | 0.69 | 0.36 | 0.17 |
| 5000 | 804.93 µs | 909.25 µs | 2.33 ms | 3.52 ms | 0.89 | 0.35 | 0.23 |
| 10000 | 1.90 ms | 1.94 ms | 4.96 ms | 7.52 ms | 0.98 | 0.38 | 0.25 |
| 30000 | 4.42 ms | 6.16 ms | 10.89 ms | 27.43 ms | 0.72 | 0.41 | 0.16 |
| 100000 | 19.29 ms | 23.25 ms | 50.36 ms | 107.45 ms | 0.83 | 0.38 | 0.18 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 2 | 165 ns | 129 ns | 106 ns | 1.29 | 1.55 |
| 4 | 212 ns | 183 ns | 205 ns | 1.16 | 1.03 |
| 8 | 358 ns | 275 ns | 397 ns | 1.30 | 0.90 |
| 16 | 566 ns | 447 ns | 1.59 µs | 1.27 | 0.36 |
| 32 | 1.13 µs | 885 ns | 5.27 µs | 1.28 | 0.21 |
| 64 | 2.95 µs | 2.26 µs | 14.49 µs | 1.30 | 0.20 |
| 128 | 9.50 µs | 9.73 µs | 37.47 µs | 0.98 | 0.25 |
| 256 | 32.05 µs | 25.42 µs | 103.57 µs | 1.26 | 0.31 |
| 512 | 82.77 µs | 71.30 µs | 257.88 µs | 1.16 | 0.32 |
| 1000 | 125.02 µs | 145.00 µs | 668.00 µs | 0.86 | 0.19 |
| 2000 | 259.26 µs | 314.08 µs | 1.58 ms | 0.83 | 0.16 |
| 5000 | 996.07 µs | 914.00 µs | 4.26 ms | 1.09 | 0.23 |
| 10000 | 2.09 ms | 1.94 ms | 9.23 ms | 1.08 | 0.23 |
| 30000 | 4.74 ms | 6.19 ms | 38.79 ms | 0.77 | 0.12 |
| 100000 | 21.32 ms | 23.52 ms | 149.55 ms | 0.91 | 0.14 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 2 | 192 ns | 134 ns | 107 ns | 1.43 | 1.79 |
| 4 | 224 ns | 177 ns | 189 ns | 1.27 | 1.19 |
| 8 | 322 ns | 266 ns | 379 ns | 1.21 | 0.85 |
| 16 | 535 ns | 452 ns | 1.63 µs | 1.18 | 0.33 |
| 32 | 1.04 µs | 900 ns | 5.39 µs | 1.15 | 0.19 |
| 64 | 2.78 µs | 2.26 µs | 15.18 µs | 1.23 | 0.18 |
| 128 | 9.19 µs | 9.77 µs | 39.27 µs | 0.94 | 0.23 |
| 256 | 32.14 µs | 25.24 µs | 103.59 µs | 1.27 | 0.31 |
| 512 | 81.61 µs | 71.95 µs | 263.33 µs | 1.13 | 0.31 |
| 1000 | 124.81 µs | 144.33 µs | 678.60 µs | 0.86 | 0.18 |
| 2000 | 257.15 µs | 311.17 µs | 1.47 ms | 0.83 | 0.18 |
| 5000 | 991.59 µs | 914.25 µs | 4.33 ms | 1.08 | 0.23 |
| 10000 | 2.09 ms | 1.94 ms | 9.29 ms | 1.08 | 0.22 |
| 30000 | 5.14 ms | 6.17 ms | 39.91 ms | 0.83 | 0.13 |
| 100000 | 23.10 ms | 23.51 ms | 150.74 ms | 0.98 | 0.15 |

`inv_series(n)` modulo a word (NTL's `InvTrunc`):

`n = 65,521`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 2 | 42 ns | 8 ns | 8 ns | 5.53 | 5.19 |
| 4 | 60 ns | 13 ns | 15 ns | 4.62 | 4.11 |
| 8 | 104 ns | 34 ns | 47 ns | 3.03 | 2.23 |
| 16 | 192 ns | 108 ns | 138 ns | 1.78 | 1.40 |
| 32 | 376 ns | 311 ns | 487 ns | 1.21 | 0.77 |
| 64 | 798 ns | 915 ns | 1.91 µs | 0.87 | 0.42 |
| 128 | 2.14 µs | 3.01 µs | 7.90 µs | 0.71 | 0.27 |
| 256 | 4.88 µs | 6.42 µs | 31.69 µs | 0.76 | 0.15 |
| 512 | 11.00 µs | 12.42 µs | 56.97 µs | 0.89 | 0.19 |
| 1000 | 22.66 µs | 25.98 µs | 109.89 µs | 0.87 | 0.21 |
| 2000 | 48.01 µs | 55.16 µs | 225.22 µs | 0.87 | 0.21 |
| 5000 | 194.99 µs | 157.40 µs | 971.50 µs | 1.24 | 0.20 |
| 10000 | 395.61 µs | 328.83 µs | 2.04 ms | 1.20 | 0.19 |
| 30000 | 885.36 µs | 1.08 ms | 4.48 ms | 0.82 | 0.20 |
| 100000 | 3.68 ms | 4.13 ms | 21.34 ms | 0.89 | 0.17 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 2 | 37 ns | 8 ns | 8 ns | 4.71 | 4.65 |
| 4 | 64 ns | 18 ns | 14 ns | 3.61 | 4.47 |
| 8 | 126 ns | 55 ns | 46 ns | 2.27 | 2.76 |
| 16 | 242 ns | 167 ns | 136 ns | 1.45 | 1.77 |
| 32 | 486 ns | 436 ns | 484 ns | 1.11 | 1.00 |
| 64 | 1.10 µs | 1.17 µs | 2.00 µs | 0.94 | 0.55 |
| 128 | 3.02 µs | 3.38 µs | 7.88 µs | 0.89 | 0.38 |
| 256 | 9.17 µs | 10.91 µs | 31.83 µs | 0.84 | 0.29 |
| 512 | 21.19 µs | 31.02 µs | 128.23 µs | 0.68 | 0.17 |
| 1000 | 44.22 µs | 76.82 µs | 211.00 µs | 0.58 | 0.21 |
| 2000 | 92.45 µs | 175.75 µs | 380.70 µs | 0.53 | 0.24 |
| 5000 | 376.46 µs | 501.57 µs | 1.49 ms | 0.75 | 0.25 |
| 10000 | 770.86 µs | 1.12 ms | 3.10 ms | 0.69 | 0.25 |
| 30000 | 1.91 ms | 3.70 ms | 6.77 ms | 0.52 | 0.28 |
| 100000 | 8.75 ms | 14.05 ms | 31.25 ms | 0.62 | 0.28 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 2 | 39 ns | 8 ns | 5.09 |
| 4 | 64 ns | 17 ns | 3.68 |
| 8 | 135 ns | 80 ns | 1.68 |
| 16 | 280 ns | 224 ns | 1.25 |
| 32 | 588 ns | 569 ns | 1.03 |
| 64 | 1.36 µs | 1.55 µs | 0.87 |
| 128 | 3.69 µs | 4.85 µs | 0.76 |
| 256 | 11.53 µs | 16.19 µs | 0.71 |
| 512 | 26.75 µs | 36.30 µs | 0.74 |
| 1000 | 56.60 µs | 84.19 µs | 0.67 |
| 2000 | 117.21 µs | 184.95 µs | 0.63 |
| 5000 | 473.82 µs | 530.86 µs | 0.89 |
| 10000 | 977.90 µs | 1.14 ms | 0.86 |
| 30000 | 2.16 ms | 3.71 ms | 0.58 |
| 100000 | 9.30 ms | 14.14 ms | 0.66 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 2 | 36 ns | 7 ns | 4.86 |
| 4 | 62 ns | 24 ns | 2.58 |
| 8 | 123 ns | 81 ns | 1.52 |
| 16 | 248 ns | 223 ns | 1.11 |
| 32 | 508 ns | 565 ns | 0.90 |
| 64 | 1.21 µs | 1.57 µs | 0.77 |
| 128 | 3.39 µs | 4.89 µs | 0.69 |
| 256 | 11.20 µs | 16.26 µs | 0.69 |
| 512 | 26.46 µs | 36.92 µs | 0.72 |
| 1000 | 56.02 µs | 83.93 µs | 0.67 |
| 2000 | 115.89 µs | 182.55 µs | 0.63 |
| 5000 | 471.85 µs | 525.57 µs | 0.90 |
| 10000 | 971.96 µs | 1.13 ms | 0.86 |
| 30000 | 2.17 ms | 3.72 ms | 0.58 |
| 100000 | 10.66 ms | 14.14 ms | 0.75 |

`mul_mod(&b, &f)` modulo a word, `f` of n coefficients (FLINT's `nmod_poly_mulmod` and NTL's `MulMod`, neither with a precomputed inverse):

`n = 65,521`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 90 ns | 50 ns | 83 ns | 1.81 | 1.09 |
| 8 | 156 ns | 116 ns | 150 ns | 1.34 | 1.04 |
| 16 | 282 ns | 242 ns | 343 ns | 1.16 | 0.82 |
| 32 | 521 ns | 736 ns | 1.17 µs | 0.71 | 0.45 |
| 64 | 1.12 µs | 2.47 µs | 3.92 µs | 0.45 | 0.29 |
| 128 | 2.90 µs | 9.37 µs | 13.77 µs | 0.31 | 0.21 |
| 256 | 9.50 µs | 17.19 µs | 50.11 µs | 0.55 | 0.19 |
| 512 | 33.11 µs | 30.98 µs | 106.11 µs | 1.07 | 0.31 |
| 1000 | 68.42 µs | 59.97 µs | 215.56 µs | 1.14 | 0.32 |
| 2000 | 138.45 µs | 121.68 µs | 449.33 µs | 1.14 | 0.31 |
| 5000 | 527.97 µs | 347.55 µs | 1.82 ms | 1.52 | 0.29 |
| 10000 | 1.12 ms | 719.80 µs | 3.86 ms | 1.55 | 0.29 |
| 30000 | 2.49 ms | 2.29 ms | 9.00 ms | 1.09 | 0.28 |
| 100000 | 10.34 ms | 8.88 ms | 41.85 ms | 1.16 | 0.25 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 186 ns | 124 ns | 128 ns | 1.50 | 1.45 |
| 8 | 299 ns | 262 ns | 210 ns | 1.14 | 1.43 |
| 16 | 575 ns | 554 ns | 450 ns | 1.04 | 1.28 |
| 32 | 1.31 µs | 1.31 µs | 1.33 µs | 1.00 | 0.99 |
| 64 | 3.83 µs | 3.44 µs | 4.30 µs | 1.12 | 0.89 |
| 128 | 8.99 µs | 10.23 µs | 15.40 µs | 0.88 | 0.58 |
| 256 | 24.99 µs | 30.37 µs | 53.62 µs | 0.82 | 0.47 |
| 512 | 60.61 µs | 80.70 µs | 173.00 µs | 0.75 | 0.35 |
| 1000 | 124.28 µs | 180.16 µs | 368.00 µs | 0.69 | 0.34 |
| 2000 | 253.09 µs | 399.11 µs | 715.20 µs | 0.63 | 0.35 |
| 5000 | 993.72 µs | 1.21 ms | 2.76 ms | 0.82 | 0.36 |
| 10000 | 2.40 ms | 2.52 ms | 5.79 ms | 0.95 | 0.41 |
| 30000 | 5.54 ms | 7.99 ms | 13.45 ms | 0.69 | 0.41 |
| 100000 | 23.96 ms | 30.51 ms | 61.55 ms | 0.79 | 0.39 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 4 | 209 ns | 146 ns | 1.43 |
| 8 | 394 ns | 297 ns | 1.33 |
| 16 | 738 ns | 620 ns | 1.19 |
| 32 | 1.68 µs | 1.48 µs | 1.14 |
| 64 | 4.88 µs | 3.84 µs | 1.27 |
| 128 | 12.99 µs | 14.44 µs | 0.90 |
| 256 | 38.07 µs | 35.18 µs | 1.08 |
| 512 | 75.25 µs | 88.35 µs | 0.85 |
| 1000 | 153.81 µs | 194.05 µs | 0.79 |
| 2000 | 318.66 µs | 414.67 µs | 0.77 |
| 5000 | 1.23 ms | 1.22 ms | 1.01 |
| 10000 | 2.61 ms | 2.51 ms | 1.04 |
| 30000 | 5.87 ms | 8.08 ms | 0.73 |
| 100000 | 27.46 ms | 30.34 ms | 0.91 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 4 | 211 ns | 142 ns | 1.48 |
| 8 | 351 ns | 291 ns | 1.21 |
| 16 | 659 ns | 637 ns | 1.03 |
| 32 | 1.52 µs | 1.52 µs | 1.00 |
| 64 | 4.41 µs | 3.88 µs | 1.14 |
| 128 | 12.48 µs | 17.25 µs | 0.72 |
| 256 | 38.25 µs | 34.80 µs | 1.10 |
| 512 | 74.01 µs | 89.08 µs | 0.83 |
| 1000 | 152.54 µs | 192.19 µs | 0.79 |
| 2000 | 317.30 µs | 413.78 µs | 0.77 |
| 5000 | 1.22 ms | 1.21 ms | 1.01 |
| 10000 | 2.59 ms | 2.53 ms | 1.02 |
| 30000 | 6.50 ms | 8.02 ms | 0.81 |
| 100000 | 28.68 ms | 30.59 ms | 0.94 |

`x.pow_mod(&e, &f)` modulo a word for `e = 2⁶³ − 25`, `f` of n coefficients (FLINT's `nmod_poly_powmod_fmpz_binexp_preinv` with the inverse computed outside the timing, NTL's `PowerMod` with a `zz_pXModulus` built outside, PARI's `Flxq_pow`):

`n = 65,521`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 4 | 2.91 µs | 3.24 µs | 9.02 µs | 3.58 µs | 0.90 | 0.32 | 0.81 |
| 8 | 6.67 µs | 12.06 µs | 14.04 µs | 8.52 µs | 0.55 | 0.48 | 0.78 |
| 16 | 15.42 µs | 26.82 µs | 24.87 µs | 19.09 µs | 0.58 | 0.62 | 0.81 |
| 32 | 27.63 µs | 72.33 µs | 67.07 µs | 48.86 µs | 0.38 | 0.41 | 0.57 |
| 64 | 61.83 µs | 168.25 µs | 208.10 µs | 152.91 µs | 0.37 | 0.30 | 0.40 |
| 128 | 166.00 µs | 443.50 µs | 431.33 µs | 308.50 µs | 0.37 | 0.38 | 0.54 |
| 256 | 414.69 µs | 864.25 µs | 1.01 ms | 825.50 µs | 0.48 | 0.41 | 0.50 |
| 512 | 772.07 µs | 1.80 ms | 1.93 ms | 2.10 ms | 0.43 | 0.40 | 0.37 |
| 1000 | 1.59 ms | 3.93 ms | 4.08 ms | 5.18 ms | 0.41 | 0.39 | 0.31 |
| 2000 | 3.16 ms | 8.22 ms | 8.63 ms | 13.13 ms | 0.38 | 0.37 | 0.24 |
| 5000 | 11.95 ms | 25.52 ms | 27.63 ms | 43.17 ms | 0.47 | 0.43 | 0.28 |
| 10000 | 24.68 ms | 52.35 ms | 58.11 ms | 90.53 ms | 0.47 | 0.42 | 0.27 |
| 30000 | 53.30 ms | 160.53 ms | 162.44 ms | 280.42 ms | 0.33 | 0.33 | 0.19 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 4 | 4.53 µs | 4.84 µs | 14.98 µs | 9.07 µs | 0.94 | 0.30 | 0.50 |
| 8 | 10.91 µs | 15.39 µs | 20.08 µs | 26.59 µs | 0.71 | 0.54 | 0.41 |
| 16 | 27.66 µs | 42.82 µs | 31.89 µs | 87.45 µs | 0.65 | 0.87 | 0.32 |
| 32 | 69.42 µs | 119.58 µs | 79.35 µs | 173.61 µs | 0.58 | 0.87 | 0.40 |
| 64 | 186.44 µs | 353.09 µs | 225.62 µs | 464.75 µs | 0.53 | 0.83 | 0.40 |
| 128 | 475.56 µs | 1.07 ms | 776.00 µs | 1.20 ms | 0.44 | 0.61 | 0.40 |
| 256 | 792.21 µs | 3.43 ms | 1.40 ms | 3.04 ms | 0.23 | 0.57 | 0.26 |
| 512 | 1.49 ms | 7.00 ms | 2.85 ms | 9.60 ms | 0.21 | 0.52 | 0.16 |
| 1000 | 3.02 ms | 14.52 ms | 6.05 ms | 19.24 ms | 0.21 | 0.50 | 0.16 |
| 2000 | 6.11 ms | 30.49 ms | 12.69 ms | 44.80 ms | 0.20 | 0.48 | 0.14 |
| 5000 | 23.29 ms | 89.57 ms | 40.70 ms | 118.66 ms | 0.26 | 0.57 | 0.20 |
| 10000 | 58.68 ms | 184.77 ms | 85.96 ms | 241.31 ms | 0.32 | 0.68 | 0.24 |
| 30000 | 126.51 ms | 558.07 ms | 239.96 ms | 866.12 ms | 0.23 | 0.53 | 0.15 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 4 | 4.74 µs | 4.87 µs | 10.34 µs | 0.97 | 0.46 |
| 8 | 11.46 µs | 19.98 µs | 28.18 µs | 0.57 | 0.41 |
| 16 | 28.79 µs | 56.39 µs | 95.08 µs | 0.51 | 0.30 |
| 32 | 87.64 µs | 165.22 µs | 230.75 µs | 0.53 | 0.38 |
| 64 | 240.20 µs | 488.25 µs | 645.50 µs | 0.49 | 0.37 |
| 128 | 678.60 µs | 1.56 ms | 1.67 ms | 0.44 | 0.41 |
| 256 | 999.46 µs | 3.45 ms | 3.98 ms | 0.29 | 0.25 |
| 512 | 1.89 ms | 7.04 ms | 10.86 ms | 0.27 | 0.17 |
| 1000 | 3.79 ms | 14.84 ms | 26.61 ms | 0.26 | 0.14 |
| 2000 | 7.67 ms | 30.75 ms | 49.70 ms | 0.25 | 0.15 |
| 5000 | 28.68 ms | 90.30 ms | 140.77 ms | 0.32 | 0.20 |
| 10000 | 59.85 ms | 185.60 ms | 290.59 ms | 0.32 | 0.21 |
| 30000 | 130.11 ms | 566.68 ms | 1.24 s | 0.23 | 0.10 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 4 | 4.22 µs | 5.37 µs | 9.90 µs | 0.79 | 0.43 |
| 8 | 10.70 µs | 20.11 µs | 29.08 µs | 0.53 | 0.37 |
| 16 | 26.67 µs | 56.03 µs | 96.25 µs | 0.48 | 0.28 |
| 32 | 77.88 µs | 165.17 µs | 232.94 µs | 0.47 | 0.33 |
| 64 | 220.09 µs | 504.14 µs | 655.83 µs | 0.44 | 0.34 |
| 128 | 656.15 µs | 1.55 ms | 1.71 ms | 0.42 | 0.38 |
| 256 | 994.43 µs | 3.42 ms | 3.99 ms | 0.29 | 0.25 |
| 512 | 1.89 ms | 6.98 ms | 10.73 ms | 0.27 | 0.18 |
| 1000 | 3.77 ms | 14.79 ms | 27.10 ms | 0.25 | 0.14 |
| 2000 | 7.68 ms | 30.68 ms | 49.91 ms | 0.25 | 0.15 |
| 5000 | 28.96 ms | 90.11 ms | 141.43 ms | 0.32 | 0.20 |
| 10000 | 59.78 ms | 185.50 ms | 290.98 ms | 0.32 | 0.21 |
| 30000 | 153.50 ms | 565.38 ms | 1.30 s | 0.27 | 0.12 |

`a.pow_mod(&e, &f)` modulo a word for an `e` of 1024 bits and a random `a`:

`n = 65,521`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 4 | 40.84 µs | 38.90 µs | 119.78 µs | 52.38 µs | 1.05 | 0.34 | 0.78 |
| 8 | 105.09 µs | 150.89 µs | 227.62 µs | 130.11 µs | 0.70 | 0.46 | 0.81 |
| 16 | 274.05 µs | 335.18 µs | 506.43 µs | 339.60 µs | 0.82 | 0.54 | 0.81 |
| 32 | 510.19 µs | 941.00 µs | 1.65 ms | 963.33 µs | 0.54 | 0.31 | 0.53 |
| 64 | 1.18 ms | 2.11 ms | 5.50 ms | 2.97 ms | 0.56 | 0.22 | 0.40 |
| 128 | 3.32 ms | 5.68 ms | 9.93 ms | 6.41 ms | 0.58 | 0.33 | 0.52 |
| 256 | 8.43 ms | 10.80 ms | 23.05 ms | 17.40 ms | 0.78 | 0.37 | 0.48 |
| 512 | 16.38 ms | 22.44 ms | 45.09 ms | 45.12 ms | 0.73 | 0.36 | 0.36 |
| 1000 | 34.22 ms | 48.02 ms | 97.40 ms | 116.19 ms | 0.71 | 0.35 | 0.29 |
| 2000 | 68.68 ms | 102.76 ms | 208.19 ms | 309.73 ms | 0.67 | 0.33 | 0.22 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 4 | 67.95 µs | 61.00 µs | 194.40 µs | 131.63 µs | 1.11 | 0.35 | 0.52 |
| 8 | 188.22 µs | 196.68 µs | 319.08 µs | 402.78 µs | 0.96 | 0.59 | 0.47 |
| 16 | 516.68 µs | 542.86 µs | 639.33 µs | 1.62 ms | 0.95 | 0.81 | 0.32 |
| 32 | 1.35 ms | 1.51 ms | 1.86 ms | 3.27 ms | 0.90 | 0.73 | 0.41 |
| 64 | 3.85 ms | 4.45 ms | 5.96 ms | 9.72 ms | 0.87 | 0.65 | 0.40 |
| 128 | 10.00 ms | 13.46 ms | 21.57 ms | 25.37 ms | 0.74 | 0.46 | 0.39 |
| 256 | 16.89 ms | 46.75 ms | 32.95 ms | 65.28 ms | 0.36 | 0.51 | 0.26 |
| 512 | 32.09 ms | 92.21 ms | 67.89 ms | 212.39 ms | 0.35 | 0.47 | 0.15 |
| 1000 | 65.87 ms | 193.69 ms | 146.10 ms | 428.12 ms | 0.34 | 0.45 | 0.15 |
| 2000 | 135.68 ms | 403.00 ms | 310.43 ms | 1.03 s | 0.34 | 0.44 | 0.13 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 4 | 73.43 µs | 61.31 µs | 156.17 µs | 1.20 | 0.47 |
| 8 | 205.71 µs | 254.27 µs | 425.89 µs | 0.81 | 0.48 |
| 16 | 539.53 µs | 714.40 µs | 1.80 ms | 0.76 | 0.30 |
| 32 | 1.75 ms | 2.10 ms | 4.51 ms | 0.84 | 0.39 |
| 64 | 5.03 ms | 6.12 ms | 13.06 ms | 0.82 | 0.39 |
| 128 | 14.37 ms | 19.85 ms | 34.80 ms | 0.72 | 0.41 |
| 256 | 21.15 ms | 45.67 ms | 85.44 ms | 0.46 | 0.25 |
| 512 | 41.13 ms | 92.75 ms | 242.30 ms | 0.44 | 0.17 |
| 1000 | 83.18 ms | 194.61 ms | 597.89 ms | 0.43 | 0.14 |
| 2000 | 171.20 ms | 403.65 ms | 1.14 s | 0.42 | 0.15 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | PARI | a/FLINT | a/PARI |
|---|---|---|---|---|---|
| 4 | 63.27 µs | 72.20 µs | 141.50 µs | 0.88 | 0.45 |
| 8 | 186.36 µs | 258.56 µs | 427.75 µs | 0.72 | 0.44 |
| 16 | 529.28 µs | 744.20 µs | 1.99 ms | 0.71 | 0.27 |
| 32 | 1.57 ms | 2.06 ms | 4.55 ms | 0.76 | 0.34 |
| 64 | 4.57 ms | 6.35 ms | 13.21 ms | 0.72 | 0.35 |
| 128 | 13.88 ms | 19.92 ms | 36.14 ms | 0.70 | 0.38 |
| 256 | 21.09 ms | 45.93 ms | 86.74 ms | 0.46 | 0.24 |
| 512 | 40.41 ms | 92.08 ms | 238.61 ms | 0.44 | 0.17 |
| 1000 | 83.15 ms | 194.80 ms | 605.72 ms | 0.43 | 0.14 |
| 2000 | 171.10 ms | 401.71 ms | 1.14 s | 0.43 | 0.15 |

`compose(&g)` modulo a word, both of n coefficients:

`n = 65,521`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 2 | 40 ns | 8 ns | 4.75 |
| 4 | 146 ns | 61 ns | 2.37 |
| 8 | 720 ns | 495 ns | 1.45 |
| 16 | 3.69 µs | 3.58 µs | 1.03 |
| 32 | 27.46 µs | 22.81 µs | 1.20 |
| 64 | 155.17 µs | 131.81 µs | 1.18 |
| 128 | 867.69 µs | 670.00 µs | 1.30 |
| 256 | 4.13 ms | 3.26 ms | 1.26 |
| 512 | 19.29 ms | 16.30 ms | 1.18 |
| 1000 | 87.95 ms | 88.78 ms | 0.99 |
| 2000 | 512.59 ms | 535.64 ms | 0.96 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 2 | 42 ns | 8 ns | 5.06 |
| 4 | 174 ns | 78 ns | 2.24 |
| 8 | 1.60 µs | 944 ns | 1.69 |
| 16 | 11.62 µs | 9.27 µs | 1.25 |
| 32 | 68.80 µs | 73.96 µs | 0.93 |
| 64 | 379.14 µs | 500.00 µs | 0.76 |
| 128 | 1.73 ms | 2.76 ms | 0.63 |
| 256 | 8.57 ms | 13.22 ms | 0.65 |
| 512 | 41.74 ms | 64.14 ms | 0.65 |
| 1000 | 197.72 ms | 297.81 ms | 0.66 |
| 2000 | 1.25 s | 1.41 s | 0.89 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 2 | 42 ns | 9 ns | 4.82 |
| 4 | 174 ns | 78 ns | 2.23 |
| 8 | 2.01 µs | 1.22 µs | 1.64 |
| 16 | 14.40 µs | 12.38 µs | 1.16 |
| 32 | 86.92 µs | 86.97 µs | 1.00 |
| 64 | 530.60 µs | 528.43 µs | 1.00 |
| 128 | 2.16 ms | 2.77 ms | 0.78 |
| 256 | 10.11 ms | 13.33 ms | 0.76 |
| 512 | 48.33 ms | 64.71 ms | 0.75 |
| 1000 | 232.64 ms | 300.20 ms | 0.77 |
| 2000 | 1.42 s | 1.42 s | 1.00 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 2 | 42 ns | 8 ns | 5.30 |
| 4 | 194 ns | 98 ns | 1.98 |
| 8 | 1.70 µs | 1.21 µs | 1.41 |
| 16 | 13.06 µs | 12.59 µs | 1.04 |
| 32 | 80.30 µs | 88.46 µs | 0.91 |
| 64 | 493.05 µs | 529.00 µs | 0.93 |
| 128 | 2.16 ms | 2.77 ms | 0.78 |
| 256 | 10.39 ms | 13.30 ms | 0.78 |
| 512 | 50.76 ms | 64.36 ms | 0.79 |
| 1000 | 240.07 ms | 299.57 ms | 0.80 |
| 2000 | 1.50 s | 1.40 s | 1.07 |

`compose_mod(&g, &f)` modulo a word, all of n coefficients (FLINT's `nmod_poly_compose_mod_brent_kung_preinv`, the inverse computed outside):

`n = 65,521`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 4 | 194 ns | 155 ns | 1.25 |
| 8 | 493 ns | 566 ns | 0.87 |
| 16 | 1.82 µs | 1.70 µs | 1.07 |
| 32 | 5.09 µs | 6.93 µs | 0.73 |
| 64 | 16.72 µs | 21.93 µs | 0.76 |
| 128 | 66.89 µs | 84.07 µs | 0.80 |
| 256 | 256.13 µs | 232.00 µs | 1.10 |
| 512 | 755.33 µs | 684.80 µs | 1.10 |
| 1000 | 2.21 ms | 2.04 ms | 1.09 |
| 2000 | 6.51 ms | 6.14 ms | 1.06 |
| 5000 | 40.35 ms | 30.24 ms | 1.33 |

`n = 2⁵⁰ − 27`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 4 | 280 ns | 177 ns | 1.58 |
| 8 | 782 ns | 688 ns | 1.14 |
| 16 | 3.15 µs | 2.51 µs | 1.25 |
| 32 | 12.02 µs | 10.99 µs | 1.09 |
| 64 | 54.51 µs | 43.67 µs | 1.25 |
| 128 | 196.24 µs | 200.60 µs | 0.98 |
| 256 | 490.86 µs | 885.75 µs | 0.55 |
| 512 | 1.44 ms | 2.61 ms | 0.55 |
| 1000 | 4.14 ms | 7.49 ms | 0.55 |
| 2000 | 12.30 ms | 21.48 ms | 0.57 |
| 5000 | 76.92 ms | 97.63 ms | 0.79 |

`n = 2⁶³ − 25`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 4 | 320 ns | 177 ns | 1.81 |
| 8 | 850 ns | 841 ns | 1.01 |
| 16 | 3.20 µs | 3.13 µs | 1.02 |
| 32 | 15.67 µs | 15.01 µs | 1.04 |
| 64 | 69.94 µs | 67.27 µs | 1.04 |
| 128 | 272.99 µs | 321.91 µs | 0.85 |
| 256 | 622.98 µs | 916.00 µs | 0.68 |
| 512 | 1.81 ms | 2.64 ms | 0.69 |
| 1000 | 5.23 ms | 7.67 ms | 0.68 |
| 2000 | 15.76 ms | 22.40 ms | 0.70 |
| 5000 | 96.13 ms | 101.21 ms | 0.95 |

`n = 2⁶⁴ − 59`:

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 4 | 285 ns | 190 ns | 1.50 |
| 8 | 796 ns | 865 ns | 0.92 |
| 16 | 3.04 µs | 3.27 µs | 0.93 |
| 32 | 14.02 µs | 15.04 µs | 0.93 |
| 64 | 63.52 µs | 61.48 µs | 1.03 |
| 128 | 264.71 µs | 289.46 µs | 0.91 |
| 256 | 610.42 µs | 916.50 µs | 0.67 |
| 512 | 1.80 ms | 2.65 ms | 0.68 |
| 1000 | 5.09 ms | 7.66 ms | 0.66 |
| 2000 | 15.48 ms | 22.24 ms | 0.70 |
| 5000 | 94.88 ms | 101.31 ms | 0.94 |

Products over the integers, coefficients of the bits given with random signs:

Coefficients of 32 bits:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 33 ns | 6 ns | 32 ns | 8 ns | 5.06 | 1.01 | 3.92 |
| 2 | 90 ns | 19 ns | 54 ns | 34 ns | 4.75 | 1.68 | 2.61 |
| 4 | 136 ns | 32 ns | 128 ns | 140 ns | 4.21 | 1.06 | 0.97 |
| 8 | 299 ns | 72 ns | 437 ns | 332 ns | 4.15 | 0.68 | 0.90 |
| 16 | 812 ns | 178 ns | 1.74 µs | 769 ns | 4.57 | 0.47 | 1.06 |
| 32 | 1.71 µs | 544 ns | 7.26 µs | 1.74 µs | 3.15 | 0.24 | 0.99 |
| 64 | 3.78 µs | 1.76 µs | 23.35 µs | 4.87 µs | 2.14 | 0.16 | 0.78 |
| 128 | 6.98 µs | 3.61 µs | 75.86 µs | 16.76 µs | 1.93 | 0.09 | 0.42 |
| 256 | 13.89 µs | 7.13 µs | 35.25 µs | 29.77 µs | 1.95 | 0.39 | 0.47 |
| 512 | 29.62 µs | 15.02 µs | 71.86 µs | 79.80 µs | 1.97 | 0.41 | 0.37 |
| 1000 | 58.92 µs | 32.41 µs | 144.16 µs | 180.95 µs | 1.82 | 0.41 | 0.33 |
| 2000 | 119.09 µs | 68.71 µs | 300.25 µs | 380.90 µs | 1.73 | 0.40 | 0.31 |
| 5000 | 348.91 µs | 211.50 µs | 798.75 µs | 1.12 ms | 1.65 | 0.44 | 0.31 |
| 10000 | 727.53 µs | 457.86 µs | 1.62 ms | 2.35 ms | 1.59 | 0.45 | 0.31 |
| 30000 | 1.93 ms | 1.59 ms | 5.10 ms | 7.39 ms | 1.22 | 0.38 | 0.26 |
| 100000 | 6.95 ms | 6.38 ms | 18.33 ms | 36.27 ms | 1.09 | 0.38 | 0.19 |

Coefficients of 64 bits:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 33 ns | 7 ns | 34 ns | 8 ns | 4.61 | 0.98 | 3.95 |
| 2 | 106 ns | 30 ns | 53 ns | 38 ns | 3.53 | 2.00 | 2.75 |
| 4 | 199 ns | 125 ns | 131 ns | 174 ns | 1.59 | 1.52 | 1.15 |
| 8 | 375 ns | 571 ns | 476 ns | 392 ns | 0.66 | 0.79 | 0.96 |
| 16 | 1.08 µs | 1.21 µs | 1.90 µs | 996 ns | 0.89 | 0.57 | 1.08 |
| 32 | 2.42 µs | 2.80 µs | 7.82 µs | 2.66 µs | 0.86 | 0.31 | 0.91 |
| 64 | 5.47 µs | 6.23 µs | 26.78 µs | 7.72 µs | 0.88 | 0.20 | 0.71 |
| 128 | 10.14 µs | 6.83 µs | 87.79 µs | 17.97 µs | 1.49 | 0.12 | 0.56 |
| 256 | 20.30 µs | 13.67 µs | 49.28 µs | 46.46 µs | 1.48 | 0.41 | 0.44 |
| 512 | 41.53 µs | 28.67 µs | 98.70 µs | 128.93 µs | 1.45 | 0.42 | 0.32 |
| 1000 | 82.95 µs | 61.89 µs | 200.05 µs | 269.36 µs | 1.34 | 0.41 | 0.31 |
| 2000 | 170.96 µs | 131.24 µs | 403.11 µs | 602.83 µs | 1.30 | 0.42 | 0.28 |
| 5000 | 502.79 µs | 412.62 µs | 1.08 ms | 1.52 ms | 1.22 | 0.46 | 0.33 |
| 10000 | 1.15 ms | 885.00 µs | 2.28 ms | 3.22 ms | 1.30 | 0.51 | 0.36 |
| 30000 | 2.96 ms | 3.00 ms | 6.90 ms | 10.64 ms | 0.99 | 0.43 | 0.28 |
| 100000 | 11.10 ms | 11.31 ms | 25.44 ms | 58.57 ms | 0.98 | 0.44 | 0.19 |

Coefficients of 1,000 bits:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 128 ns | 59 ns | 134 ns | 83 ns | 2.18 | 0.96 | 1.54 |
| 2 | 700 ns | 256 ns | 304 ns | 286 ns | 2.73 | 2.31 | 2.45 |
| 4 | 1.93 µs | 1.07 µs | 952 ns | 2.70 µs | 1.79 | 2.02 | 0.71 |
| 8 | 7.16 µs | 3.04 µs | 3.09 µs | 10.59 µs | 2.36 | 2.31 | 0.68 |
| 16 | 16.08 µs | 11.37 µs | 9.81 µs | 21.29 µs | 1.41 | 1.64 | 0.76 |
| 32 | 34.15 µs | 24.63 µs | 30.24 µs | 59.94 µs | 1.39 | 1.13 | 0.57 |
| 64 | 71.88 µs | 54.23 µs | 94.38 µs | 156.20 µs | 1.33 | 0.76 | 0.46 |
| 128 | 152.20 µs | 271.08 µs | 298.00 µs | 330.45 µs | 0.56 | 0.51 | 0.46 |
| 256 | 310.13 µs | 576.50 µs | 484.57 µs | 749.80 µs | 0.54 | 0.64 | 0.41 |
| 512 | 655.88 µs | 1.20 ms | 1.06 ms | 1.58 ms | 0.54 | 0.62 | 0.41 |
| 1000 | 1.53 ms | 2.42 ms | 2.29 ms | 4.37 ms | 0.63 | 0.67 | 0.35 |
| 2000 | 3.25 ms | 4.98 ms | 4.92 ms | 9.05 ms | 0.65 | 0.66 | 0.36 |
| 5000 | 8.30 ms | 13.08 ms | 11.72 ms | 22.03 ms | 0.63 | 0.71 | 0.38 |
| 10000 | 16.99 ms | 27.29 ms | 24.40 ms | 57.01 ms | 0.62 | 0.70 | 0.30 |
| 30000 | 59.16 ms | 88.50 ms | 74.39 ms | 177.50 ms | 0.67 | 0.80 | 0.33 |
| 100000 | 289.30 ms | 276.69 ms | 280.84 ms | 708.71 ms | 1.05 | 1.03 | 0.41 |

Coefficients of 10,000 bits:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 3.27 µs | 2.57 µs | 3.71 µs | 3.54 µs | 1.27 | 0.88 | 0.92 |
| 2 | 15.47 µs | 8.54 µs | 10.91 µs | 11.04 µs | 1.81 | 1.42 | 1.40 |
| 4 | 39.36 µs | 26.78 µs | 34.09 µs | 64.79 µs | 1.47 | 1.15 | 0.61 |
| 8 | 96.13 µs | 161.26 µs | 102.14 µs | 185.15 µs | 0.60 | 0.94 | 0.52 |
| 16 | 203.59 µs | 326.00 µs | 309.82 µs | 387.70 µs | 0.62 | 0.66 | 0.53 |
| 32 | 427.38 µs | 711.60 µs | 984.00 µs | 954.25 µs | 0.60 | 0.43 | 0.45 |
| 64 | 878.66 µs | 1.48 ms | 2.96 ms | 1.96 ms | 0.59 | 0.30 | 0.45 |
| 128 | 1.85 ms | 3.06 ms | 4.51 ms | 4.06 ms | 0.61 | 0.41 | 0.46 |
| 256 | 3.82 ms | 6.22 ms | 8.37 ms | 10.49 ms | 0.61 | 0.46 | 0.36 |
| 512 | 8.09 ms | 13.02 ms | 15.11 ms | 21.94 ms | 0.62 | 0.54 | 0.37 |
| 1000 | 16.53 ms | 26.68 ms | 31.39 ms | 55.66 ms | 0.62 | 0.53 | 0.30 |
| 2000 | 34.20 ms | 55.56 ms | 64.02 ms | 149.16 ms | 0.62 | 0.53 | 0.23 |
| 5000 | 136.82 ms | 131.95 ms | 168.81 ms | 371.47 ms | 1.04 | 0.81 | 0.37 |
| 10000 | 284.62 ms | 267.64 ms | 466.07 ms | 591.05 ms | 1.06 | 0.61 | 0.48 |
| 30000 | 1.48 s | 906.19 ms | 1.85 s | 2.05 s | 1.63 | 0.80 | 0.72 |

Squares over the integers:

Coefficients of 32 bits:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 33 ns | 6 ns | 31 ns | 7 ns | 5.93 | 1.08 | 4.96 |
| 2 | 64 ns | 16 ns | 47 ns | 42 ns | 3.89 | 1.37 | 1.53 |
| 4 | 116 ns | 25 ns | 105 ns | 118 ns | 4.58 | 1.11 | 0.99 |
| 8 | 268 ns | 55 ns | 307 ns | 267 ns | 4.88 | 0.88 | 1.01 |
| 16 | 680 ns | 124 ns | 1.07 µs | 633 ns | 5.46 | 0.64 | 1.07 |
| 32 | 1.37 µs | 328 ns | 3.94 µs | 1.52 µs | 4.19 | 0.35 | 0.90 |
| 64 | 3.45 µs | 1.14 µs | 17.34 µs | 4.06 µs | 3.03 | 0.20 | 0.85 |
| 128 | 6.42 µs | 3.81 µs | 56.45 µs | 9.96 µs | 1.69 | 0.11 | 0.64 |
| 256 | 11.98 µs | 5.58 µs | 29.01 µs | 23.54 µs | 2.15 | 0.41 | 0.51 |
| 512 | 24.34 µs | 11.76 µs | 58.32 µs | 58.95 µs | 2.07 | 0.42 | 0.41 |
| 1000 | 50.99 µs | 24.85 µs | 116.31 µs | 148.32 µs | 2.05 | 0.44 | 0.34 |
| 2000 | 104.67 µs | 53.56 µs | 238.47 µs | 287.92 µs | 1.95 | 0.44 | 0.36 |
| 5000 | 301.41 µs | 165.83 µs | 645.60 µs | 822.00 µs | 1.82 | 0.47 | 0.37 |
| 10000 | 618.83 µs | 355.25 µs | 1.32 ms | 1.74 ms | 1.74 | 0.47 | 0.35 |
| 30000 | 1.66 ms | 1.19 ms | 4.07 ms | 5.59 ms | 1.40 | 0.41 | 0.30 |
| 100000 | 5.93 ms | 4.83 ms | 14.74 ms | 26.99 ms | 1.23 | 0.40 | 0.22 |

Coefficients of 64 bits:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 34 ns | 7 ns | 31 ns | 6 ns | 4.93 | 1.09 | 5.23 |
| 2 | 84 ns | 18 ns | 48 ns | 42 ns | 4.56 | 1.76 | 2.01 |
| 4 | 165 ns | 76 ns | 113 ns | 136 ns | 2.18 | 1.46 | 1.21 |
| 8 | 349 ns | 528 ns | 312 ns | 340 ns | 0.66 | 1.12 | 1.03 |
| 16 | 836 ns | 1.20 µs | 1.07 µs | 884 ns | 0.70 | 0.78 | 0.95 |
| 32 | 1.91 µs | 2.68 µs | 4.03 µs | 2.15 µs | 0.71 | 0.47 | 0.89 |
| 64 | 4.91 µs | 6.01 µs | 20.57 µs | 6.10 µs | 0.82 | 0.24 | 0.80 |
| 128 | 8.94 µs | 4.81 µs | 66.10 µs | 14.86 µs | 1.86 | 0.14 | 0.60 |
| 256 | 17.00 µs | 9.59 µs | 40.40 µs | 35.92 µs | 1.77 | 0.42 | 0.47 |
| 512 | 35.35 µs | 20.08 µs | 80.69 µs | 95.10 µs | 1.76 | 0.44 | 0.37 |
| 1000 | 71.17 µs | 42.90 µs | 163.45 µs | 204.94 µs | 1.66 | 0.44 | 0.35 |
| 2000 | 141.49 µs | 94.14 µs | 325.20 µs | 455.62 µs | 1.50 | 0.44 | 0.31 |
| 5000 | 412.32 µs | 303.20 µs | 876.75 µs | 1.09 ms | 1.36 | 0.47 | 0.38 |
| 10000 | 930.86 µs | 662.20 µs | 1.86 ms | 2.32 ms | 1.41 | 0.50 | 0.40 |
| 30000 | 2.44 ms | 2.16 ms | 5.53 ms | 8.02 ms | 1.13 | 0.44 | 0.30 |
| 100000 | 9.05 ms | 8.14 ms | 20.20 ms | 42.38 ms | 1.11 | 0.45 | 0.21 |

Coefficients of 1,000 bits:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 113 ns | 68 ns | 117 ns | 65 ns | 1.66 | 0.96 | 1.74 |
| 2 | 584 ns | 204 ns | 250 ns | 290 ns | 2.86 | 2.34 | 2.01 |
| 4 | 1.68 µs | 709 ns | 783 ns | 1.00 µs | 2.36 | 2.14 | 1.68 |
| 8 | 5.35 µs | 3.11 µs | 2.53 µs | 3.61 µs | 1.72 | 2.12 | 1.48 |
| 16 | 12.51 µs | 10.38 µs | 7.97 µs | 13.37 µs | 1.20 | 1.57 | 0.94 |
| 32 | 25.99 µs | 21.89 µs | 24.90 µs | 51.17 µs | 1.19 | 1.04 | 0.51 |
| 64 | 54.06 µs | 46.97 µs | 77.48 µs | 129.10 µs | 1.15 | 0.70 | 0.42 |
| 128 | 111.94 µs | 209.06 µs | 241.29 µs | 237.87 µs | 0.54 | 0.46 | 0.47 |
| 256 | 231.11 µs | 437.00 µs | 345.90 µs | 558.57 µs | 0.53 | 0.67 | 0.41 |
| 512 | 480.67 µs | 913.75 µs | 754.20 µs | 1.19 ms | 0.53 | 0.64 | 0.40 |
| 1000 | 1.14 ms | 1.84 ms | 1.63 ms | 3.18 ms | 0.62 | 0.70 | 0.36 |
| 2000 | 2.42 ms | 3.76 ms | 3.49 ms | 6.57 ms | 0.64 | 0.69 | 0.37 |
| 5000 | 6.15 ms | 9.91 ms | 9.17 ms | 15.68 ms | 0.62 | 0.67 | 0.39 |
| 10000 | 12.60 ms | 20.36 ms | 18.90 ms | 40.10 ms | 0.62 | 0.67 | 0.31 |
| 30000 | 43.01 ms | 66.03 ms | 56.89 ms | 131.29 ms | 0.65 | 0.76 | 0.33 |
| 100000 | 215.48 ms | 199.84 ms | 208.08 ms | 484.11 ms | 1.08 | 1.04 | 0.45 |

Coefficients of 10,000 bits:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 2.49 µs | 2.86 µs | 3.04 µs | 2.81 µs | 0.87 | 0.82 | 0.89 |
| 2 | 11.35 µs | 9.30 µs | 9.50 µs | 9.93 µs | 1.22 | 1.20 | 1.14 |
| 4 | 29.75 µs | 28.84 µs | 28.71 µs | 35.02 µs | 1.03 | 1.04 | 0.85 |
| 8 | 72.04 µs | 123.30 µs | 92.22 µs | 129.00 µs | 0.58 | 0.78 | 0.56 |
| 16 | 151.97 µs | 252.07 µs | 280.85 µs | 280.07 µs | 0.60 | 0.54 | 0.54 |
| 32 | 317.82 µs | 540.86 µs | 835.00 µs | 708.40 µs | 0.59 | 0.38 | 0.45 |
| 64 | 652.82 µs | 1.13 ms | 2.60 ms | 1.44 ms | 0.58 | 0.25 | 0.45 |
| 128 | 1.37 ms | 2.32 ms | 2.89 ms | 3.02 ms | 0.59 | 0.48 | 0.45 |
| 256 | 2.84 ms | 4.73 ms | 5.90 ms | 7.53 ms | 0.60 | 0.48 | 0.38 |
| 512 | 5.97 ms | 9.81 ms | 11.97 ms | 15.57 ms | 0.61 | 0.50 | 0.38 |
| 1000 | 12.22 ms | 19.66 ms | 25.04 ms | 39.70 ms | 0.62 | 0.49 | 0.31 |
| 2000 | 25.26 ms | 41.05 ms | 51.17 ms | 109.55 ms | 0.62 | 0.49 | 0.23 |
| 5000 | 102.79 ms | 94.64 ms | 133.59 ms | 261.95 ms | 1.09 | 0.77 | 0.39 |
| 10000 | 211.87 ms | 192.70 ms | 336.86 ms | 423.64 ms | 1.10 | 0.63 | 0.50 |
| 30000 | 1.10 s | 669.25 ms | 1.48 s | 1.46 s | 1.64 | 0.74 | 0.75 |

`pseudo_div_rem` over the integers, 2n by n coefficients (FLINT and NTL may multiply by a lower power of `l`):

Coefficients of 64 bits:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 2 | 231 ns | 283 ns | 228 ns | 0.82 | 1.02 |
| 4 | 1.03 µs | 691 ns | 776 ns | 1.49 | 1.33 |
| 8 | 2.67 µs | 2.08 µs | 2.74 µs | 1.29 | 0.98 |
| 16 | 8.88 µs | 7.44 µs | 10.33 µs | 1.19 | 0.86 |
| 32 | 37.90 µs | 33.67 µs | 45.47 µs | 1.13 | 0.83 |
| 64 | 205.19 µs | 403.22 µs | 224.75 µs | 0.51 | 0.91 |
| 128 | 1.28 ms | 2.63 ms | 1.44 ms | 0.49 | 0.89 |
| 256 | 8.96 ms | 14.08 ms | 10.63 ms | 0.64 | 0.84 |
| 512 | 52.78 ms | 72.11 ms | 77.45 ms | 0.73 | 0.68 |
| 1000 | 222.33 ms | 350.41 ms | 543.98 ms | 0.63 | 0.41 |
| 2000 | 1.83 s | 1.78 s | 4.21 s | 1.03 | 0.43 |

Coefficients of 1,000 bits:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 2 | 1.89 µs | 3.63 µs | 2.64 µs | 0.52 | 0.72 |
| 4 | 9.77 µs | 20.14 µs | 14.13 µs | 0.49 | 0.69 |
| 8 | 56.95 µs | 117.00 µs | 86.00 µs | 0.49 | 0.66 |
| 16 | 383.31 µs | 763.00 µs | 553.00 µs | 0.50 | 0.69 |
| 32 | 2.77 ms | 5.72 ms | 3.89 ms | 0.48 | 0.71 |
| 64 | 11.76 ms | 19.31 ms | 28.47 ms | 0.61 | 0.41 |
| 128 | 49.33 ms | 97.92 ms | 218.75 ms | 0.50 | 0.23 |
| 256 | 206.73 ms | 469.84 ms | 1.70 s | 0.44 | 0.12 |
| 512 | 1.63 s | 2.24 s | 13.35 s | 0.73 | 0.12 |

`div_exact` over the integers of a product by a factor of n coefficients (FLINT's `fmpz_poly_divides`, NTL's `divide`):

Coefficients of 64 bits:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 2 | 159 ns | 166 ns | 665 ns | 0.96 | 0.24 |
| 4 | 439 ns | 402 ns | 724 ns | 1.09 | 0.61 |
| 8 | 1.06 µs | 1.32 µs | 1.42 µs | 0.80 | 0.74 |
| 16 | 2.77 µs | 3.66 µs | 3.96 µs | 0.76 | 0.70 |
| 32 | 6.06 µs | 11.38 µs | 7.60 µs | 0.53 | 0.80 |
| 64 | 14.56 µs | 30.37 µs | 16.94 µs | 0.48 | 0.86 |
| 128 | 33.47 µs | 79.24 µs | 50.59 µs | 0.42 | 0.66 |
| 256 | 81.85 µs | 189.81 µs | 99.86 µs | 0.43 | 0.82 |
| 512 | 157.91 µs | 463.29 µs | 194.63 µs | 0.34 | 0.81 |
| 1000 | 328.10 µs | 1.09 ms | 391.00 µs | 0.30 | 0.84 |
| 2000 | 680.42 µs | 2.44 ms | 810.00 µs | 0.28 | 0.84 |

Coefficients of 1,000 bits:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 2 | 721 ns | 1.10 µs | 8.87 µs | 0.66 | 0.08 |
| 4 | 2.69 µs | 2.53 µs | 10.35 µs | 1.06 | 0.26 |
| 8 | 7.91 µs | 7.29 µs | 16.10 µs | 1.09 | 0.49 |
| 16 | 25.96 µs | 22.56 µs | 55.18 µs | 1.15 | 0.47 |
| 32 | 92.35 µs | 69.43 µs | 103.52 µs | 1.33 | 0.89 |
| 64 | 244.69 µs | 199.32 µs | 225.18 µs | 1.23 | 1.09 |
| 128 | 508.31 µs | 519.57 µs | 569.00 µs | 0.98 | 0.89 |
| 256 | 1.06 ms | 1.60 ms | 1.01 ms | 0.66 | 1.05 |
| 512 | 2.17 ms | 4.44 ms | 2.00 ms | 0.49 | 1.09 |

Products modulo `n` of several limbs:

`n = 2¹²⁷ − 1`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 32 ns | 30 ns | 34 ns | 33 ns | 1.08 | 0.94 | 0.96 |
| 2 | 66 ns | 95 ns | 95 ns | 109 ns | 0.70 | 0.70 | 0.61 |
| 4 | 159 ns | 280 ns | 296 ns | 358 ns | 0.57 | 0.54 | 0.44 |
| 8 | 513 ns | 969 ns | 969 ns | 861 ns | 0.53 | 0.53 | 0.60 |
| 16 | 1.23 µs | 2.22 µs | 3.44 µs | 2.21 µs | 0.56 | 0.36 | 0.56 |
| 32 | 3.15 µs | 5.21 µs | 12.36 µs | 5.83 µs | 0.61 | 0.25 | 0.54 |
| 64 | 7.91 µs | 12.88 µs | 34.74 µs | 17.07 µs | 0.61 | 0.23 | 0.46 |
| 128 | 15.40 µs | 24.91 µs | 101.19 µs | 37.77 µs | 0.62 | 0.15 | 0.41 |
| 256 | 30.76 µs | 48.57 µs | 36.00 µs | 91.33 µs | 0.63 | 0.85 | 0.34 |
| 512 | 63.73 µs | 101.31 µs | 76.02 µs | 230.44 µs | 0.63 | 0.84 | 0.28 |
| 1000 | 128.38 µs | 214.41 µs | 158.76 µs | 487.62 µs | 0.60 | 0.81 | 0.26 |
| 2000 | 260.10 µs | 421.78 µs | 329.18 µs | 1.08 ms | 0.62 | 0.79 | 0.24 |
| 5000 | 944.47 µs | 1.14 ms | 937.75 µs | 3.04 ms | 0.83 | 1.01 | 0.31 |
| 10000 | 1.96 ms | 2.31 ms | 1.98 ms | 6.63 ms | 0.85 | 0.99 | 0.30 |
| 30000 | 6.68 ms | 6.98 ms | 6.16 ms | 21.94 ms | 0.96 | 1.08 | 0.30 |
| 100000 | 20.43 ms | 25.02 ms | 24.62 ms | 79.39 ms | 0.82 | 0.83 | 0.26 |

`n = 2²⁵⁵ − 19`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 69 ns | 56 ns | 61 ns | 60 ns | 1.24 | 1.13 | 1.14 |
| 2 | 179 ns | 171 ns | 185 ns | 184 ns | 1.05 | 0.97 | 0.97 |
| 4 | 463 ns | 460 ns | 546 ns | 723 ns | 1.01 | 0.85 | 0.64 |
| 8 | 1.50 µs | 1.69 µs | 2.14 µs | 1.75 µs | 0.88 | 0.70 | 0.86 |
| 16 | 3.64 µs | 4.25 µs | 5.50 µs | 4.90 µs | 0.86 | 0.66 | 0.74 |
| 32 | 9.59 µs | 10.94 µs | 15.01 µs | 16.38 µs | 0.88 | 0.64 | 0.59 |
| 64 | 19.99 µs | 30.01 µs | 42.26 µs | 30.32 µs | 0.67 | 0.47 | 0.66 |
| 128 | 40.43 µs | 79.24 µs | 121.88 µs | 80.32 µs | 0.51 | 0.33 | 0.50 |
| 256 | 83.19 µs | 167.32 µs | 72.39 µs | 211.94 µs | 0.50 | 1.15 | 0.39 |
| 512 | 165.90 µs | 341.55 µs | 150.12 µs | 430.56 µs | 0.49 | 1.11 | 0.39 |
| 1000 | 333.17 µs | 704.80 µs | 314.46 µs | 912.00 µs | 0.47 | 1.06 | 0.37 |
| 2000 | 686.21 µs | 1.47 ms | 662.60 µs | 2.00 ms | 0.47 | 1.04 | 0.34 |
| 5000 | 2.13 ms | 3.92 ms | 1.88 ms | 5.04 ms | 0.54 | 1.14 | 0.42 |
| 10000 | 4.48 ms | 8.03 ms | 3.98 ms | 10.71 ms | 0.56 | 1.13 | 0.42 |
| 30000 | 15.39 ms | 24.71 ms | 12.37 ms | 49.57 ms | 0.62 | 1.24 | 0.31 |
| 100000 | 45.75 ms | 82.86 ms | 49.47 ms | 183.56 ms | 0.55 | 0.92 | 0.25 |

`n = 2⁵²¹ − 1`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 169 ns | 125 ns | 140 ns | 134 ns | 1.35 | 1.21 | 1.26 |
| 2 | 446 ns | 401 ns | 449 ns | 416 ns | 1.11 | 0.99 | 1.07 |
| 4 | 1.17 µs | 1.11 µs | 1.53 µs | 1.67 µs | 1.06 | 0.77 | 0.70 |
| 8 | 3.83 µs | 4.39 µs | 3.60 µs | 4.76 µs | 0.87 | 1.06 | 0.80 |
| 16 | 9.91 µs | 9.84 µs | 8.98 µs | 12.29 µs | 1.01 | 1.10 | 0.81 |
| 32 | 22.39 µs | 21.93 µs | 22.53 µs | 28.35 µs | 1.02 | 0.99 | 0.79 |
| 64 | 45.87 µs | 48.08 µs | 64.05 µs | 75.15 µs | 0.95 | 0.72 | 0.61 |
| 128 | 94.00 µs | 161.95 µs | 168.13 µs | 190.33 µs | 0.58 | 0.56 | 0.49 |
| 256 | 188.00 µs | 340.91 µs | 149.12 µs | 376.20 µs | 0.55 | 1.26 | 0.50 |
| 512 | 381.29 µs | 708.00 µs | 306.83 µs | 873.25 µs | 0.54 | 1.24 | 0.44 |
| 1000 | 759.39 µs | 1.43 ms | 635.83 µs | 1.78 ms | 0.53 | 1.19 | 0.43 |
| 2000 | 1.58 ms | 2.91 ms | 1.33 ms | 4.89 ms | 0.54 | 1.19 | 0.32 |
| 5000 | 4.53 ms | 7.77 ms | 3.75 ms | 13.46 ms | 0.58 | 1.21 | 0.34 |
| 10000 | 9.40 ms | 16.18 ms | 7.84 ms | 30.77 ms | 0.58 | 1.20 | 0.31 |
| 30000 | 31.48 ms | 50.79 ms | 24.35 ms | 84.63 ms | 0.62 | 1.29 | 0.37 |
| 100000 | 152.28 ms | 163.97 ms | 95.18 ms | 397.71 ms | 0.93 | 1.60 | 0.38 |

`n = 2¹⁰²⁴ − 105`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 327 ns | 262 ns | 295 ns | 287 ns | 1.25 | 1.11 | 1.14 |
| 2 | 936 ns | 862 ns | 1.18 µs | 894 ns | 1.09 | 0.79 | 1.05 |
| 4 | 2.59 µs | 2.52 µs | 2.88 µs | 4.18 µs | 1.03 | 0.90 | 0.62 |
| 8 | 9.32 µs | 6.44 µs | 7.36 µs | 12.65 µs | 1.45 | 1.27 | 0.74 |
| 16 | 20.24 µs | 19.69 µs | 18.92 µs | 27.26 µs | 1.03 | 1.07 | 0.74 |
| 32 | 41.98 µs | 42.89 µs | 50.58 µs | 74.02 µs | 0.98 | 0.83 | 0.57 |
| 64 | 86.37 µs | 91.65 µs | 139.88 µs | 181.10 µs | 0.94 | 0.62 | 0.48 |
| 128 | 182.75 µs | 328.55 µs | 383.67 µs | 372.90 µs | 0.56 | 0.48 | 0.49 |
| 256 | 418.01 µs | 686.80 µs | 349.80 µs | 889.25 µs | 0.61 | 1.20 | 0.47 |
| 512 | 878.83 µs | 1.43 ms | 728.20 µs | 1.83 ms | 0.62 | 1.21 | 0.48 |
| 1000 | 1.78 ms | 2.87 ms | 1.50 ms | 4.48 ms | 0.62 | 1.19 | 0.40 |
| 2000 | 3.72 ms | 5.85 ms | 3.10 ms | 10.25 ms | 0.64 | 1.20 | 0.36 |
| 5000 | 9.39 ms | 15.64 ms | 8.55 ms | 29.45 ms | 0.60 | 1.10 | 0.32 |
| 10000 | 19.26 ms | 32.52 ms | 17.64 ms | 62.20 ms | 0.59 | 1.09 | 0.31 |
| 30000 | 66.27 ms | 101.33 ms | 54.04 ms | 202.91 ms | 0.65 | 1.23 | 0.33 |
| 100000 | 315.02 ms | 328.22 ms | 218.48 ms | 748.83 ms | 0.96 | 1.44 | 0.42 |

Squares modulo `n` of several limbs:

`n = 2¹²⁷ − 1`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 30 ns | 30 ns | 34 ns | 31 ns | 1.00 | 0.88 | 0.97 |
| 2 | 67 ns | 86 ns | 93 ns | 112 ns | 0.78 | 0.73 | 0.60 |
| 4 | 158 ns | 236 ns | 247 ns | 297 ns | 0.67 | 0.64 | 0.53 |
| 8 | 547 ns | 1.02 µs | 694 ns | 781 ns | 0.54 | 0.79 | 0.70 |
| 16 | 977 ns | 2.34 µs | 2.25 µs | 1.92 µs | 0.42 | 0.43 | 0.51 |
| 32 | 2.54 µs | 5.56 µs | 9.50 µs | 4.86 µs | 0.46 | 0.27 | 0.52 |
| 64 | 7.26 µs | 13.46 µs | 26.33 µs | 12.54 µs | 0.54 | 0.28 | 0.58 |
| 128 | 13.96 µs | 20.16 µs | 13.23 µs | 30.16 µs | 0.69 | 1.06 | 0.46 |
| 256 | 27.70 µs | 39.73 µs | 27.58 µs | 76.80 µs | 0.70 | 1.00 | 0.36 |
| 512 | 57.27 µs | 81.96 µs | 57.05 µs | 198.05 µs | 0.70 | 1.00 | 0.29 |
| 1000 | 114.60 µs | 167.59 µs | 119.41 µs | 375.00 µs | 0.68 | 0.96 | 0.31 |
| 2000 | 230.95 µs | 344.36 µs | 249.92 µs | 788.00 µs | 0.67 | 0.92 | 0.29 |
| 5000 | 734.32 µs | 913.75 µs | 703.25 µs | 2.25 ms | 0.80 | 1.04 | 0.33 |
| 10000 | 1.52 ms | 1.84 ms | 1.49 ms | 4.98 ms | 0.83 | 1.02 | 0.31 |
| 30000 | 5.17 ms | 5.61 ms | 4.57 ms | 16.43 ms | 0.92 | 1.13 | 0.31 |
| 100000 | 15.99 ms | 19.97 ms | 17.97 ms | 58.23 ms | 0.80 | 0.89 | 0.27 |

`n = 2²⁵⁵ − 19`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 67 ns | 54 ns | 62 ns | 57 ns | 1.25 | 1.09 | 1.18 |
| 2 | 180 ns | 157 ns | 177 ns | 191 ns | 1.15 | 1.02 | 0.94 |
| 4 | 467 ns | 402 ns | 475 ns | 488 ns | 1.16 | 0.98 | 0.96 |
| 8 | 1.26 µs | 1.82 µs | 1.26 µs | 1.36 µs | 0.69 | 1.00 | 0.92 |
| 16 | 3.06 µs | 4.60 µs | 4.63 µs | 4.09 µs | 0.66 | 0.66 | 0.75 |
| 32 | 7.70 µs | 11.20 µs | 12.03 µs | 10.82 µs | 0.69 | 0.64 | 0.71 |
| 64 | 18.63 µs | 26.23 µs | 33.80 µs | 26.27 µs | 0.71 | 0.55 | 0.71 |
| 128 | 37.40 µs | 64.63 µs | 25.96 µs | 65.72 µs | 0.58 | 1.44 | 0.57 |
| 256 | 76.49 µs | 135.78 µs | 54.21 µs | 168.36 µs | 0.56 | 1.41 | 0.45 |
| 512 | 152.93 µs | 283.77 µs | 112.25 µs | 326.55 µs | 0.54 | 1.36 | 0.47 |
| 1000 | 305.14 µs | 580.17 µs | 238.62 µs | 683.40 µs | 0.53 | 1.28 | 0.45 |
| 2000 | 629.56 µs | 1.20 ms | 497.14 µs | 1.53 ms | 0.52 | 1.27 | 0.41 |
| 5000 | 1.73 ms | 3.12 ms | 1.40 ms | 3.78 ms | 0.55 | 1.23 | 0.46 |
| 10000 | 3.58 ms | 6.36 ms | 2.96 ms | 8.07 ms | 0.56 | 1.21 | 0.44 |
| 30000 | 12.01 ms | 20.03 ms | 9.09 ms | 35.28 ms | 0.60 | 1.32 | 0.34 |
| 100000 | 37.09 ms | 63.98 ms | 36.17 ms | 138.75 ms | 0.58 | 1.03 | 0.27 |

`n = 2⁵²¹ − 1`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 166 ns | 123 ns | 138 ns | 130 ns | 1.34 | 1.19 | 1.28 |
| 2 | 456 ns | 369 ns | 420 ns | 449 ns | 1.24 | 1.08 | 1.02 |
| 4 | 1.19 µs | 958 ns | 1.12 µs | 1.17 µs | 1.24 | 1.06 | 1.02 |
| 8 | 3.29 µs | 4.01 µs | 3.22 µs | 3.10 µs | 0.82 | 1.02 | 1.06 |
| 16 | 8.05 µs | 8.62 µs | 7.70 µs | 8.65 µs | 0.93 | 1.05 | 0.93 |
| 32 | 19.05 µs | 18.56 µs | 19.52 µs | 27.34 µs | 1.03 | 0.98 | 0.70 |
| 64 | 43.10 µs | 39.95 µs | 53.58 µs | 61.54 µs | 1.08 | 0.80 | 0.70 |
| 128 | 87.46 µs | 131.57 µs | 53.42 µs | 159.40 µs | 0.66 | 1.64 | 0.55 |
| 256 | 175.14 µs | 276.92 µs | 110.26 µs | 292.31 µs | 0.63 | 1.59 | 0.60 |
| 512 | 354.17 µs | 572.67 µs | 225.41 µs | 657.83 µs | 0.62 | 1.57 | 0.54 |
| 1000 | 702.25 µs | 1.17 ms | 471.88 µs | 1.37 ms | 0.60 | 1.49 | 0.51 |
| 2000 | 1.46 ms | 2.35 ms | 976.33 µs | 3.63 ms | 0.62 | 1.49 | 0.40 |
| 5000 | 3.67 ms | 6.11 ms | 2.73 ms | 8.14 ms | 0.60 | 1.34 | 0.45 |
| 10000 | 7.54 ms | 12.66 ms | 5.77 ms | 23.65 ms | 0.60 | 1.31 | 0.32 |
| 30000 | 25.20 ms | 39.68 ms | 17.53 ms | 62.41 ms | 0.64 | 1.44 | 0.40 |
| 100000 | 120.57 ms | 124.53 ms | 68.83 ms | 287.96 ms | 0.97 | 1.75 | 0.42 |

`n = 2¹⁰²⁴ − 105`:

| length | `a` | FLINT | NTL | PARI | a/FLINT | a/NTL | a/PARI |
|---|---|---|---|---|---|---|---|
| 1 | 324 ns | 276 ns | 280 ns | 270 ns | 1.17 | 1.15 | 1.20 |
| 2 | 940 ns | 836 ns | 878 ns | 907 ns | 1.13 | 1.07 | 1.04 |
| 4 | 2.62 µs | 2.20 µs | 2.55 µs | 2.44 µs | 1.19 | 1.03 | 1.08 |
| 8 | 7.70 µs | 6.42 µs | 6.42 µs | 6.84 µs | 1.20 | 1.20 | 1.12 |
| 16 | 17.29 µs | 18.18 µs | 16.22 µs | 20.64 µs | 0.95 | 1.07 | 0.84 |
| 32 | 35.57 µs | 38.48 µs | 42.37 µs | 69.09 µs | 0.92 | 0.84 | 0.51 |
| 64 | 73.62 µs | 80.77 µs | 117.68 µs | 152.96 µs | 0.91 | 0.63 | 0.48 |
| 128 | 152.66 µs | 269.00 µs | 124.19 µs | 293.85 µs | 0.57 | 1.23 | 0.52 |
| 256 | 345.61 µs | 555.86 µs | 255.00 µs | 711.00 µs | 0.62 | 1.36 | 0.49 |
| 512 | 710.35 µs | 1.15 ms | 534.43 µs | 1.43 ms | 0.62 | 1.33 | 0.50 |
| 1000 | 1.45 ms | 2.31 ms | 1.10 ms | 3.46 ms | 0.63 | 1.32 | 0.42 |
| 2000 | 3.01 ms | 4.67 ms | 2.26 ms | 6.89 ms | 0.64 | 1.33 | 0.44 |
| 5000 | 7.64 ms | 12.20 ms | 6.21 ms | 22.57 ms | 0.63 | 1.23 | 0.34 |
| 10000 | 15.64 ms | 25.16 ms | 12.89 ms | 47.10 ms | 0.62 | 1.21 | 0.33 |
| 30000 | 52.49 ms | 79.90 ms | 39.48 ms | 158.62 ms | 0.66 | 1.33 | 0.33 |
| 100000 | 247.65 ms | 248.07 ms | 157.75 ms | 525.89 ms | 1.00 | 1.57 | 0.47 |

`div_rem` modulo `n` of several limbs, 2n by n coefficients:

`n = 2¹²⁷ − 1`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 663 ns | 954 ns | 915 ns | 0.69 | 0.72 |
| 8 | 1.08 µs | 1.93 µs | 1.76 µs | 0.56 | 0.61 |
| 16 | 2.31 µs | 4.65 µs | 4.47 µs | 0.50 | 0.52 |
| 32 | 6.36 µs | 8.92 µs | 14.29 µs | 0.71 | 0.45 |
| 64 | 21.44 µs | 30.37 µs | 51.70 µs | 0.71 | 0.41 |
| 128 | 78.17 µs | 75.31 µs | 84.70 µs | 1.04 | 0.92 |
| 256 | 136.80 µs | 161.71 µs | 174.59 µs | 0.85 | 0.78 |
| 512 | 275.91 µs | 318.25 µs | 362.73 µs | 0.87 | 0.76 |
| 1000 | 491.85 µs | 632.00 µs | 490.62 µs | 0.78 | 1.00 |
| 2000 | 1.01 ms | 1.30 ms | 1.02 ms | 0.78 | 0.99 |

`n = 2²⁵⁵ − 19`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 1.44 µs | 1.62 µs | 1.60 µs | 0.89 | 0.90 |
| 8 | 2.41 µs | 3.01 µs | 3.02 µs | 0.80 | 0.80 |
| 16 | 5.23 µs | 6.86 µs | 7.24 µs | 0.76 | 0.72 |
| 32 | 14.53 µs | 18.90 µs | 21.75 µs | 0.77 | 0.67 |
| 64 | 45.61 µs | 54.61 µs | 72.53 µs | 0.84 | 0.63 |
| 128 | 160.84 µs | 200.53 µs | 163.71 µs | 0.80 | 0.98 |
| 256 | 333.64 µs | 507.86 µs | 339.55 µs | 0.66 | 0.98 |
| 512 | 675.10 µs | 1.15 ms | 712.60 µs | 0.59 | 0.95 |
| 1000 | 1.21 ms | 2.38 ms | 949.75 µs | 0.51 | 1.27 |
| 2000 | 2.45 ms | 5.02 ms | 2.00 ms | 0.49 | 1.23 |

`n = 2⁵²¹ − 1`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 3.11 µs | 3.28 µs | 3.36 µs | 0.95 | 0.92 |
| 8 | 5.80 µs | 6.08 µs | 6.52 µs | 0.95 | 0.89 |
| 16 | 13.62 µs | 14.02 µs | 15.92 µs | 0.97 | 0.86 |
| 32 | 39.50 µs | 41.40 µs | 46.99 µs | 0.95 | 0.84 |
| 64 | 187.29 µs | 107.26 µs | 158.84 µs | 1.75 | 1.18 |
| 128 | 367.47 µs | 384.30 µs | 325.80 µs | 0.96 | 1.13 |
| 256 | 728.82 µs | 1.04 ms | 670.60 µs | 0.70 | 1.09 |
| 512 | 1.48 ms | 2.30 ms | 1.39 ms | 0.64 | 1.06 |
| 1000 | 2.65 ms | 4.80 ms | 1.91 ms | 0.55 | 1.39 |
| 2000 | 5.50 ms | 10.12 ms | 3.97 ms | 0.54 | 1.38 |

`n = 2¹⁰²⁴ − 105`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 6.25 µs | 6.93 µs | 7.21 µs | 0.90 | 0.87 |
| 8 | 12.69 µs | 13.80 µs | 15.30 µs | 0.92 | 0.83 |
| 16 | 31.91 µs | 34.29 µs | 40.33 µs | 0.93 | 0.79 |
| 32 | 99.35 µs | 96.62 µs | 129.53 µs | 1.03 | 0.77 |
| 64 | 416.20 µs | 272.93 µs | 455.00 µs | 1.52 | 0.91 |
| 128 | 883.15 µs | 759.20 µs | 733.25 µs | 1.16 | 1.20 |
| 256 | 1.86 ms | 2.04 ms | 1.50 ms | 0.91 | 1.24 |
| 512 | 3.87 ms | 4.59 ms | 3.09 ms | 0.84 | 1.25 |
| 1000 | 7.65 ms | 9.76 ms | 4.38 ms | 0.78 | 1.75 |
| 2000 | 16.13 ms | 20.32 ms | 9.02 ms | 0.79 | 1.79 |

`x.pow_mod(&n, &f)` modulo `n` of several limbs (FLINT's `fmpz_mod_poly_powmod_fmpz_binexp_preinv`, NTL's `PowerMod`):

`n = 2¹²⁷ − 1`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 66.32 µs | 86.09 µs | 190.35 µs | 0.77 | 0.35 |
| 8 | 155.09 µs | 301.50 µs | 365.20 µs | 0.51 | 0.42 |
| 16 | 377.55 µs | 906.50 µs | 921.00 µs | 0.42 | 0.41 |
| 32 | 1.11 ms | 2.26 ms | 1.12 ms | 0.49 | 0.99 |
| 64 | 2.45 ms | 5.68 ms | 2.21 ms | 0.43 | 1.11 |
| 128 | 4.64 ms | 12.31 ms | 4.47 ms | 0.38 | 1.04 |
| 256 | 9.10 ms | 24.68 ms | 9.06 ms | 0.37 | 1.00 |
| 512 | 18.45 ms | 49.36 ms | 18.74 ms | 0.37 | 0.98 |

`n = 2²⁵⁵ − 19`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 306.14 µs | 272.14 µs | 644.50 µs | 1.12 | 0.48 |
| 8 | 784.41 µs | 975.75 µs | 1.24 ms | 0.80 | 0.63 |
| 16 | 2.01 ms | 3.68 ms | 3.25 ms | 0.55 | 0.62 |
| 32 | 5.59 ms | 9.83 ms | 4.22 ms | 0.57 | 1.32 |
| 64 | 12.18 ms | 26.21 ms | 8.47 ms | 0.46 | 1.44 |
| 128 | 23.83 ms | 76.69 ms | 17.37 ms | 0.31 | 1.37 |
| 256 | 48.87 ms | 160.24 ms | 35.64 ms | 0.30 | 1.37 |
| 512 | 97.97 ms | 330.10 ms | 73.22 ms | 0.30 | 1.34 |

`n = 2⁵²¹ − 1`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 1.42 ms | 1.15 ms | 2.69 ms | 1.23 | 0.53 |
| 8 | 3.95 ms | 4.49 ms | 5.55 ms | 0.88 | 0.71 |
| 16 | 10.69 ms | 21.38 ms | 13.06 ms | 0.50 | 0.82 |
| 32 | 29.12 ms | 48.52 ms | 17.62 ms | 0.60 | 1.65 |
| 64 | 56.55 ms | 108.08 ms | 35.59 ms | 0.52 | 1.59 |
| 128 | 113.79 ms | 316.23 ms | 72.86 ms | 0.36 | 1.56 |

`n = 2¹⁰²⁴ − 105`:

| length | `a` | FLINT | NTL | a/FLINT | a/NTL |
|---|---|---|---|---|---|
| 4 | 5.39 ms | 4.63 ms | 10.90 ms | 1.16 | 0.49 |
| 8 | 16.96 ms | 20.06 ms | 23.86 ms | 0.85 | 0.71 |
| 16 | 47.46 ms | 85.37 ms | 61.31 ms | 0.56 | 0.77 |
| 32 | 135.37 ms | 191.55 ms | 79.87 ms | 0.71 | 1.69 |
| 64 | 274.46 ms | 421.29 ms | 163.94 ms | 0.65 | 1.67 |
| 128 | 573.18 ms | 1.27 s | 332.10 ms | 0.45 | 1.73 |

Products over the rationals, numerators and denominators of 31 bits (FLINT's `fmpq_poly` keeps one common denominator, `Polynomial<Rational>` each coefficient in lowest terms):

| length | `a` | FLINT | a/FLINT |
|---|---|---|---|
| 1 | 37 ns | 29 ns | 1.29 |
| 2 | 175 ns | 70 ns | 2.51 |
| 4 | 1.05 µs | 352 ns | 2.98 |
| 8 | 6.50 µs | 1.87 µs | 3.48 |
| 16 | 35.27 µs | 7.70 µs | 4.58 |
| 32 | 176.91 µs | 26.20 µs | 6.75 |
| 64 | 814.67 µs | 113.00 µs | 7.21 |
| 128 | 3.52 ms | 883.50 µs | 3.99 |
| 256 | 16.05 ms | 3.94 ms | 4.08 |
| 512 | 80.81 ms | 16.39 ms | 4.93 |
| 1000 | 430.20 ms | 65.61 ms | 6.56 |
| 2000 | 2.51 s | 266.38 ms | 9.41 |

Products of floats and of complex floats, against the plain schoolbook loop (FLINT has no polynomials of floats):

`f64`:

| length | `a` | schoolbook | a/schoolbook |
|---|---|---|---|
| 4 | 21 ns | 17 ns | 1.19 |
| 16 | 67 ns | 71 ns | 0.94 |
| 64 | 442 ns | 466 ns | 0.95 |
| 128 | 1.63 µs | 1.65 µs | 0.99 |
| 255 | 6.37 µs | 6.35 µs | 1.00 |
| 256 | 6.40 µs | 6.42 µs | 1.00 |
| 399 | 15.60 µs | 15.61 µs | 1.00 |
| 400 | 15.70 µs | 15.59 µs | 1.01 |
| 512 | 18.37 µs | 25.35 µs | 0.72 |
| 1000 | 39.52 µs | 97.29 µs | 0.41 |
| 2000 | 85.27 µs | 405.96 µs | 0.21 |
| 5000 | 428.26 µs | 2.54 ms | 0.17 |
| 10000 | 1.12 ms | 17.81 ms | 0.06 |
| 100000 | 12.34 ms |  |  |

`Complex<f64>`:

| length | `a` | schoolbook | a/schoolbook |
|---|---|---|---|
| 4 | 25 ns | 18 ns | 1.33 |
| 16 | 134 ns | 129 ns | 1.04 |
| 64 | 1.79 µs | 1.79 µs | 1.00 |
| 128 | 4.16 µs | 7.04 µs | 0.59 |
| 255 | 9.04 µs | 27.78 µs | 0.33 |
| 256 | 8.81 µs | 28.42 µs | 0.31 |
| 399 | 18.80 µs | 69.32 µs | 0.27 |
| 400 | 18.84 µs | 69.14 µs | 0.27 |
| 512 | 19.10 µs | 113.74 µs | 0.17 |
| 1000 | 41.12 µs | 435.38 µs | 0.09 |
| 2000 | 86.79 µs | 1.73 ms | 0.05 |
| 5000 | 436.28 µs | 10.94 ms | 0.04 |
| 10000 | 1.15 ms | 43.26 ms | 0.03 |
| 100000 | 12.40 ms |  |  |

### Greatest common divisors and resultants

Each row is the median of three runs (at 19:32, 19:50 and 19:52, on the final code; load averages 1.8 to 2.5 at their starts), each run the median of its rounds. The gcds are of pairs of degrees `d` and `d − 1` with a common factor of degree `d/2` (over the integers of half the bits), unless the table says coprime; the resultants and discriminants of random polynomials. Over the rationals NTL has no `QQX`, and PARI's `polresultant` over the rationals is not timed.

`gcd` modulo a word prime of 17 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.2 | 0.3 | 0.84 | 0.4 | 0.63 | 0.2 | 1.11 |
| 30 | 0.5 | 0.8 | 0.68 | 1.1 | 0.49 | 0.8 | 0.64 |
| 100 | 3.5 | 4.1 | 0.85 | 6.5 | 0.54 | 6.3 | 0.55 |
| 300 | 26.0 | 27.9 | 0.93 | 44.2 | 0.59 | 44.8 | 0.58 |
| 1000 | 275.4 | 226.5 | 1.22 | 456.8 | 0.60 | 352.2 | 0.78 |
| 3000 | 1299.3 | 977.0 | 1.33 | 2125.0 | 0.61 | 1741.0 | 0.75 |
| 10000 | 5508.6 | 4122.0 | 1.34 | 9065.0 | 0.61 | 9546.0 | 0.58 |

`xgcd` modulo a word prime of 17 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.7 | 0.5 | 1.50 | 0.6 | 1.18 | 0.4 | 1.66 |
| 30 | 1.6 | 1.6 | 0.98 | 2.0 | 0.76 | 1.9 | 0.83 |
| 100 | 7.7 | 12.1 | 0.64 | 14.4 | 0.54 | 11.7 | 0.66 |
| 300 | 53.7 | 56.7 | 0.95 | 121.8 | 0.44 | 62.6 | 0.86 |
| 1000 | 315.4 | 272.1 | 1.16 | 504.5 | 0.63 | 354.4 | 0.89 |
| 3000 | 1384.2 | 1148.0 | 1.21 | 2313.0 | 0.60 | 1748.0 | 0.79 |
| 10000 | 5864.2 | 4720.0 | 1.24 | 9963.0 | 0.59 | 9448.0 | 0.62 |

`resultant` modulo a word prime of 17 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.4 | 0.4 | 0.87 | 0.5 | 0.71 | 0.3 | 1.12 |
| 30 | 1.1 | 1.4 | 0.76 | 1.9 | 0.56 | 1.4 | 0.78 |
| 100 | 5.9 | 6.8 | 0.87 | 10.3 | 0.57 | 9.2 | 0.64 |
| 300 | 39.5 | 41.9 | 0.94 | 65.5 | 0.60 | 64.8 | 0.61 |
| 1000 | 381.1 | 330.2 | 1.15 | 621.3 | 0.61 | 540.0 | 0.71 |
| 3000 | 2151.4 | 1641.0 | 1.31 | 3341.0 | 0.64 | 2868.0 | 0.75 |
| 10000 | 9597.6 | 7226.0 | 1.33 | 15744 | 0.61 | 15362 | 0.62 |

`gcd` modulo a word prime of 50 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.3 | 0.7 | 0.49 | 0.7 | 0.47 | 0.5 | 0.66 |
| 30 | 0.6 | 2.2 | 0.29 | 2.0 | 0.32 | 2.3 | 0.27 |
| 100 | 3.6 | 8.6 | 0.41 | 9.6 | 0.37 | 17.2 | 0.21 |
| 300 | 26.3 | 41.2 | 0.64 | 52.4 | 0.50 | 130.3 | 0.20 |
| 1000 | 294.6 | 324.2 | 0.91 | 440.0 | 0.67 | 1284.0 | 0.23 |
| 3000 | 2514.2 | 3056.0 | 0.82 | 2850.0 | 0.88 | 6143.0 | 0.41 |
| 10000 | 10963 | 12872 | 0.85 | 12830 | 0.85 | 31947 | 0.34 |
| 30000 | 37694 | 51105 | 0.74 | 46268 | 0.81 | 128784 | 0.29 |
| 100000 | 172379 | 224346 | 0.77 | 214211 | 0.80 | 564684 | 0.31 |

`xgcd` modulo a word prime of 50 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.9 | 1.0 | 0.88 | 1.0 | 0.94 | 0.9 | 0.96 |
| 30 | 2.1 | 3.4 | 0.61 | 3.0 | 0.70 | 6.9 | 0.30 |
| 100 | 8.9 | 17.8 | 0.50 | 16.2 | 0.55 | 39.0 | 0.23 |
| 300 | 56.1 | 114.2 | 0.49 | 104.6 | 0.54 | 231.1 | 0.24 |
| 1000 | 513.6 | 724.0 | 0.71 | 706.5 | 0.73 | 1281.0 | 0.40 |
| 3000 | 2809.2 | 3561.0 | 0.79 | 3132.0 | 0.90 | 6253.0 | 0.45 |
| 10000 | 11670 | 14894 | 0.78 | 14292 | 0.82 | 31897 | 0.37 |
| 30000 | 39222 | 57523 | 0.68 | 48495 | 0.81 | 129433 | 0.30 |

`resultant` modulo a word prime of 50 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.4 | 1.1 | 0.39 | 1.1 | 0.40 | 0.9 | 0.51 |
| 30 | 1.1 | 3.9 | 0.29 | 3.6 | 0.31 | 3.9 | 0.29 |
| 100 | 5.9 | 14.8 | 0.40 | 16.2 | 0.36 | 25.7 | 0.23 |
| 300 | 39.2 | 66.0 | 0.59 | 80.6 | 0.49 | 181.6 | 0.22 |
| 1000 | 397.6 | 469.5 | 0.85 | 626.0 | 0.64 | 1793.0 | 0.22 |
| 3000 | 3453.3 | 4840.0 | 0.71 | 4417.0 | 0.78 | 9591.0 | 0.36 |
| 10000 | 18307 | 21627 | 0.85 | 22019 | 0.83 | 51883 | 0.35 |
| 30000 | 66429 | 88678 | 0.75 | 79936 | 0.83 | 218482 | 0.30 |
| 100000 | 306095 | 401379 | 0.76 | 380007 | 0.81 | 989585 | 0.31 |

`gcd` modulo a word prime of 63 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.6 | 0.8 | 0.75 | — | — | 0.6 | 1.02 |
| 30 | 1.4 | 2.6 | 0.56 | — | — | 2.6 | 0.55 |
| 100 | 6.2 | 12.2 | 0.51 | — | — | 17.8 | 0.35 |
| 300 | 34.3 | 69.8 | 0.49 | — | — | 133.2 | 0.26 |
| 1000 | 304.2 | 620.7 | 0.49 | — | — | 1706.0 | 0.18 |
| 3000 | 2567.0 | 3317.0 | 0.77 | — | — | 8182.0 | 0.31 |
| 10000 | 13175 | 14189 | 0.93 | — | — | 41462 | 0.32 |

`xgcd` modulo a word prime of 63 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.0 | 1.2 | 0.82 | — | — | 1.0 | 0.96 |
| 30 | 2.2 | 3.8 | 0.58 | — | — | 7.5 | 0.29 |
| 100 | 9.5 | 21.4 | 0.45 | — | — | 48.8 | 0.19 |
| 300 | 58.1 | 123.4 | 0.47 | — | — | 283.2 | 0.21 |
| 1000 | 525.5 | 852.5 | 0.62 | — | — | 1719.0 | 0.31 |
| 3000 | 3346.5 | 3852.0 | 0.87 | — | — | 8256.0 | 0.41 |
| 10000 | 14111 | 16168 | 0.87 | — | — | 41647 | 0.34 |

`resultant` modulo a word prime of 63 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.8 | 1.3 | 0.63 | — | — | 1.0 | 0.83 |
| 30 | 2.6 | 4.5 | 0.57 | — | — | 4.3 | 0.60 |
| 100 | 11.1 | 20.4 | 0.54 | — | — | 27.1 | 0.41 |
| 300 | 55.8 | 103.7 | 0.54 | — | — | 187.4 | 0.30 |
| 1000 | 475.8 | 868.5 | 0.55 | — | — | 2297.0 | 0.21 |
| 3000 | 3616.1 | 5302.0 | 0.68 | — | — | 12453 | 0.29 |
| 10000 | 21615 | 23856 | 0.91 | — | — | 67227 | 0.32 |

`gcd` modulo a word prime of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.6 | 0.8 | 0.76 | — | — | — | — |
| 30 | 1.6 | 2.4 | 0.65 | — | — | — | — |
| 100 | 7.9 | 10.9 | 0.72 | — | — | — | — |
| 300 | 50.9 | 56.8 | 0.90 | — | — | — | — |
| 1000 | 469.8 | 474.8 | 0.99 | — | — | — | — |
| 3000 | 3187.5 | 3434.0 | 0.93 | — | — | — | — |
| 10000 | 13346 | 14876 | 0.90 | — | — | — | — |

`xgcd` modulo a word prime of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.0 | 1.2 | 0.87 | — | — | — | — |
| 30 | 2.4 | 3.7 | 0.67 | — | — | — | — |
| 100 | 12.9 | 20.1 | 0.64 | — | — | — | — |
| 300 | 88.8 | 138.4 | 0.64 | — | — | — | — |
| 1000 | 743.9 | 878.0 | 0.85 | — | — | — | — |
| 3000 | 3408.1 | 4021.0 | 0.85 | — | — | — | — |
| 10000 | 14459 | 16940 | 0.85 | — | — | — | — |

`resultant` modulo a word prime of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.8 | 1.3 | 0.62 | — | — | — | — |
| 30 | 2.8 | 4.2 | 0.65 | — | — | — | — |
| 100 | 13.1 | 18.4 | 0.71 | — | — | — | — |
| 300 | 76.8 | 87.4 | 0.88 | — | — | — | — |
| 1000 | 730.0 | 671.7 | 1.09 | — | — | — | — |
| 3000 | 4680.5 | 4895.0 | 0.96 | — | — | — | — |
| 10000 | 22566 | 24784 | 0.91 | — | — | — | — |

`gcd` modulo a prime of 2 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.6 | 4.2 | 0.62 | 4.6 | 0.57 | 3.9 | 0.66 |
| 30 | 12.5 | 21.1 | 0.59 | 21.3 | 0.59 | 23.2 | 0.54 |
| 100 | 90.9 | 176.5 | 0.52 | 172.4 | 0.53 | 211.9 | 0.43 |
| 300 | 668.7 | 1196.0 | 0.56 | 733.0 | 0.91 | 1488.0 | 0.45 |
| 1000 | 3402.2 | 6572.0 | 0.52 | 3038.0 | 1.12 | 8127.0 | 0.42 |
| 3000 | 14190 | 23381 | 0.61 | 13020 | 1.09 | 35754 | 0.40 |
| 10000 | 59739 | 91879 | 0.65 | 48273 | 1.24 | 160170 | 0.37 |

`xgcd` modulo a prime of 2 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 4.9 | 7.7 | 0.64 | 6.6 | 0.74 | 6.1 | 0.79 |
| 30 | 21.9 | 38.3 | 0.57 | 40.9 | 0.54 | 37.8 | 0.58 |
| 100 | 149.5 | 273.4 | 0.55 | 209.4 | 0.71 | 297.5 | 0.50 |
| 300 | 771.9 | 1459.0 | 0.53 | 833.0 | 0.93 | 1485.0 | 0.52 |
| 1000 | 3658.4 | 7446.0 | 0.49 | 3289.0 | 1.11 | 8163.0 | 0.45 |

`resultant` modulo a prime of 2 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5.5 | 7.3 | 0.76 | 7.1 | 0.78 | 6.2 | 0.89 |
| 30 | 23.3 | 37.2 | 0.63 | 33.4 | 0.70 | 35.5 | 0.66 |
| 100 | 147.0 | 302.0 | 0.49 | 251.0 | 0.59 | 303.8 | 0.48 |
| 300 | 995.7 | 1867.0 | 0.53 | 1282.0 | 0.78 | 2194.0 | 0.45 |
| 1000 | 5564.7 | 9312.0 | 0.60 | 5338.0 | 1.04 | 12827 | 0.43 |

`gcd` modulo a prime of 3 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 4.8 | 6.1 | 0.78 | 6.4 | 0.75 | 5.8 | 0.83 |
| 30 | 23.9 | 29.4 | 0.81 | 30.0 | 0.80 | 33.6 | 0.71 |
| 100 | 178.7 | 238.0 | 0.75 | 241.3 | 0.74 | 300.8 | 0.59 |
| 300 | 1257.5 | 1764.0 | 0.71 | 1017.0 | 1.24 | 2198.0 | 0.57 |
| 1000 | 6165.9 | 9583.0 | 0.64 | 4167.0 | 1.48 | 11929 | 0.52 |

`xgcd` modulo a prime of 3 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 8.4 | 10.9 | 0.77 | 9.0 | 0.93 | 9.2 | 0.91 |
| 30 | 40.8 | 53.4 | 0.76 | 56.6 | 0.72 | 55.1 | 0.74 |
| 100 | 286.5 | 398.8 | 0.72 | 291.0 | 0.98 | 418.0 | 0.69 |
| 300 | 1425.6 | 2227.0 | 0.64 | 1135.0 | 1.26 | 2219.0 | 0.64 |
| 1000 | 6665.4 | 11006 | 0.61 | 4479.0 | 1.49 | 12017 | 0.55 |

`resultant` modulo a prime of 3 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 9.6 | 10.7 | 0.90 | 10.1 | 0.95 | 9.7 | 0.99 |
| 30 | 42.6 | 51.0 | 0.84 | 46.7 | 0.91 | 52.6 | 0.81 |
| 100 | 278.4 | 393.2 | 0.71 | 348.6 | 0.80 | 432.0 | 0.64 |
| 300 | 1844.5 | 2650.0 | 0.70 | 1765.0 | 1.05 | 3143.0 | 0.59 |
| 1000 | 10173 | 13821 | 0.74 | 7264.0 | 1.40 | 18762 | 0.54 |

`gcd` modulo a prime of 8 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 12.2 | 14.7 | 0.83 | 15.3 | 0.80 | 13.7 | 0.89 |
| 30 | 67.5 | 70.6 | 0.96 | 75.1 | 0.90 | 78.4 | 0.86 |
| 100 | 576.3 | 581.3 | 0.99 | 624.7 | 0.92 | 680.0 | 0.85 |
| 300 | 4153.5 | 4090.0 | 1.02 | 2667.0 | 1.56 | 6140.0 | 0.68 |
| 1000 | 18984 | 24917 | 0.76 | 10978 | 1.73 | 33198 | 0.57 |

`xgcd` modulo a prime of 8 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20.7 | 25.5 | 0.81 | 21.6 | 0.96 | 23.6 | 0.88 |
| 30 | 110.9 | 123.2 | 0.90 | 142.2 | 0.78 | 136.8 | 0.81 |
| 100 | 842.5 | 947.5 | 0.89 | 736.5 | 1.14 | 1131.0 | 0.74 |
| 300 | 4390.2 | 5175.0 | 0.85 | 2977.0 | 1.47 | 6187.0 | 0.71 |
| 1000 | 20410 | 30127 | 0.68 | 11905 | 1.71 | 33266 | 0.61 |

`resultant` modulo a prime of 8 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 24.0 | 24.8 | 0.97 | 24.7 | 0.97 | 23.2 | 1.03 |
| 30 | 115.4 | 117.7 | 0.98 | 117.8 | 0.98 | 122.8 | 0.94 |
| 100 | 867.5 | 919.5 | 0.94 | 906.5 | 0.96 | 996.0 | 0.87 |
| 300 | 5862.8 | 6094.0 | 0.96 | 4658.0 | 1.26 | 8359.0 | 0.70 |
| 1000 | 31510 | 35729 | 0.88 | 19143 | 1.65 | 52520 | 0.60 |

`gcd` modulo a prime of 16 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 28.4 | 33.5 | 0.85 | 35.3 | 0.80 | 31.6 | 0.90 |
| 30 | 163.3 | 173.9 | 0.94 | 187.8 | 0.87 | 190.0 | 0.86 |
| 100 | 1429.7 | 1472.0 | 0.97 | 1654.0 | 0.86 | 1730.0 | 0.83 |
| 300 | 8515.8 | 9001.0 | 0.95 | 6563.0 | 1.30 | 14529 | 0.59 |
| 1000 | 38735 | 55771 | 0.69 | 26771 | 1.45 | 75326 | 0.51 |

`xgcd` modulo a prime of 16 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 44.8 | 58.5 | 0.77 | 51.6 | 0.87 | 52.8 | 0.85 |
| 30 | 254.8 | 298.0 | 0.86 | 361.2 | 0.71 | 331.4 | 0.77 |
| 100 | 1930.7 | 2359.0 | 0.82 | 1913.0 | 1.01 | 2844.0 | 0.68 |
| 300 | 9293.5 | 11239 | 0.83 | 7355.0 | 1.26 | 14611 | 0.64 |
| 1000 | 41733 | 66263 | 0.63 | 28945 | 1.44 | 75272 | 0.55 |

`resultant` modulo a prime of 16 limbs:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 53.3 | 55.5 | 0.96 | 58.0 | 0.92 | 53.9 | 0.99 |
| 30 | 272.0 | 281.5 | 0.97 | 305.0 | 0.89 | 298.0 | 0.91 |
| 100 | 2124.0 | 2275.0 | 0.93 | 2507.0 | 0.85 | 2538.0 | 0.84 |
| 300 | 13063 | 13926 | 0.94 | 12861 | 1.02 | 20247 | 0.65 |
| 1000 | 66525 | 76114 | 0.87 | 47852 | 1.39 | 120981 | 0.55 |

`gcd` over the integers, a common factor of half the degree, coefficients of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.8 | 1.3 | 1.39 | 3.8 | 0.47 | 2.9 | 0.60 |
| 10 | 2.9 | 2.8 | 1.04 | 5.2 | 0.55 | 5.0 | 0.58 |
| 30 | 5.7 | 9.6 | 0.59 | 13.0 | 0.44 | 21.3 | 0.27 |
| 100 | 21.3 | 40.2 | 0.53 | 50.5 | 0.42 | 167.8 | 0.13 |
| 300 | 103.0 | 176.6 | 0.58 | 224.7 | 0.46 | 1159.0 | 0.09 |
| 1000 | 886.7 | 1193.0 | 0.74 | 1018.0 | 0.87 | 12302 | 0.07 |
| 3000 | 6068.8 | 6077.0 | 1.00 | 3994.0 | 1.52 | 85036 | 0.07 |

`gcd` over the integers, coprime, coefficients of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.2 | 1.1 | 1.14 | 1.4 | 0.87 | 0.7 | 1.71 |
| 10 | 0.9 | 1.9 | 0.48 | 2.2 | 0.42 | 1.5 | 0.63 |
| 30 | 1.9 | 5.8 | 0.33 | 5.4 | 0.36 | 5.3 | 0.36 |
| 100 | 8.3 | 21.9 | 0.38 | 21.4 | 0.39 | 29.9 | 0.28 |
| 300 | 47.0 | 96.9 | 0.49 | 97.8 | 0.48 | 200.1 | 0.23 |
| 1000 | 440.9 | 690.5 | 0.64 | 556.3 | 0.79 | 2327.0 | 0.19 |

`gcd` over the integers, a common factor of half the degree, coefficients of 1,000 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 17.1 | 19.0 | 0.90 | 26.6 | 0.65 | 125.8 | 0.14 |
| 10 | 22.2 | 26.6 | 0.83 | 32.8 | 0.68 | 132.9 | 0.17 |
| 30 | 57.2 | 65.4 | 0.87 | 90.3 | 0.63 | 203.1 | 0.28 |
| 100 | 213.6 | 281.2 | 0.76 | 295.8 | 0.72 | 690.5 | 0.31 |
| 300 | 829.8 | 1331.0 | 0.62 | 1311.0 | 0.63 | 4001.0 | 0.21 |
| 1000 | 5195.8 | 8495.0 | 0.61 | 5472.0 | 0.95 | 41630 | 0.12 |

`gcd` over the integers, coprime, coefficients of 1,000 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 17.5 | 11.0 | 1.60 | 8.2 | 2.13 | 7.9 | 2.23 |
| 10 | 7.4 | 12.8 | 0.58 | 9.1 | 0.82 | 9.0 | 0.82 |
| 30 | 9.8 | 19.8 | 0.49 | 12.8 | 0.76 | 13.8 | 0.71 |
| 100 | 21.0 | 34.6 | 0.61 | 29.5 | 0.71 | 42.7 | 0.49 |
| 300 | 73.7 | 118.9 | 0.62 | 109.6 | 0.67 | 227.9 | 0.32 |
| 1000 | 520.4 | 728.5 | 0.71 | 574.7 | 0.91 | 2383.0 | 0.22 |

`gcd` over the integers, a common factor of half the degree, coefficients of 10,000 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 415.1 | 578.7 | 0.72 | 416.2 | 1.00 | 1949.0 | 0.21 |
| 10 | 485.1 | 883.0 | 0.55 | 560.7 | 0.87 | 2226.0 | 0.22 |
| 30 | 1016.8 | 2369.0 | 0.43 | 1863.0 | 0.55 | 3770.0 | 0.27 |
| 100 | 3078.6 | 8701.0 | 0.35 | 6211.0 | 0.50 | 17183 | 0.18 |

`gcd` over the integers, coprime, coefficients of 10,000 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 441.8 | 192.4 | 2.30 | 112.2 | 3.94 | 118.2 | 3.74 |
| 10 | 97.1 | 226.6 | 0.43 | 116.6 | 0.83 | 124.2 | 0.78 |
| 30 | 108.2 | 359.2 | 0.30 | 122.1 | 0.89 | 142.0 | 0.76 |
| 100 | 157.9 | 242.0 | 0.65 | 154.5 | 1.02 | 210.4 | 0.75 |

`resultant` over the integers, coefficients of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3.3 | 1.2 | 2.76 | 9.1 | 0.36 | 137.5 | 0.02 |
| 10 | 13.7 | 8.2 | 1.67 | 34.1 | 0.40 | 169.5 | 0.08 |
| 30 | 94.5 | 479.0 | 0.20 | 312.8 | 0.30 | 513.7 | 0.18 |
| 100 | 1506.4 | 4473.0 | 0.34 | 4629.0 | 0.33 | 1891.0 | 0.80 |
| 300 | 28307 | 58182 | 0.49 | 61935 | 0.46 | 15202 | 1.86 |

`resultant` over the integers, coefficients of 1,000 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 47.2 | 64.2 | 0.74 | 179.5 | 0.26 | 914.0 | 0.05 |
| 10 | 428.9 | 727.5 | 0.59 | 658.0 | 0.65 | 1780.0 | 0.24 |
| 30 | 2781.6 | 8585.0 | 0.32 | 5849.0 | 0.48 | 5132.0 | 0.54 |
| 100 | 33145 | 86939 | 0.38 | 76989 | 0.43 | 21258 | 1.56 |

`discriminant` over the integers, coefficients of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3.9 | 1.4 | 2.89 | 9.2 | 0.43 | 143.5 | 0.03 |
| 10 | 14.9 | 9.0 | 1.65 | 35.7 | 0.42 | 176.0 | 0.08 |
| 30 | 101.7 | 500.3 | 0.20 | 326.6 | 0.31 | 544.0 | 0.19 |
| 100 | 1616.7 | 4667.0 | 0.35 | 4792.0 | 0.34 | 1871.0 | 0.86 |
| 300 | 30078 | 61270 | 0.49 | 65611 | 0.46 | 15275 | 1.97 |

`discriminant` over the integers, coefficients of 1,000 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 48.9 | 64.0 | 0.76 | 181.5 | 0.27 | 891.5 | 0.05 |
| 10 | 429.5 | 757.0 | 0.57 | 665.0 | 0.65 | 1668.0 | 0.26 |
| 30 | 2771.9 | 8501.0 | 0.33 | 5985.0 | 0.46 | 4929.0 | 0.56 |
| 100 | 31472 | 86864 | 0.36 | 76881 | 0.41 | 20302 | 1.55 |

`gcd` over the rationals, a common factor, coefficients of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3.9 | 1.5 | 2.57 | — | — | 3.7 | 1.04 |
| 10 | 6.3 | 2.7 | 2.28 | — | — | 6.6 | 0.95 |
| 30 | 14.0 | 9.8 | 1.43 | — | — | 25.3 | 0.55 |
| 100 | 47.0 | 41.0 | 1.15 | — | — | 184.4 | 0.25 |
| 300 | 179.3 | 179.0 | 1.00 | — | — | 1222.0 | 0.15 |

`gcd` over the rationals, a common factor, coefficients of 1,000 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 30.4 | 23.5 | 1.29 | — | — | 135.5 | 0.22 |
| 10 | 42.8 | 31.1 | 1.38 | — | — | 148.2 | 0.29 |
| 30 | 101.4 | 70.8 | 1.43 | — | — | 225.1 | 0.45 |
| 100 | 331.0 | 284.0 | 1.17 | — | — | 736.5 | 0.45 |

`resultant` over the rationals, coefficients of 64 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 6.1 | 2.7 | 2.28 | — | — | — | — |
| 10 | 19.0 | 11.2 | 1.69 | — | — | — | — |
| 30 | 112.0 | 510.7 | 0.22 | — | — | — | — |
| 100 | 1644.7 | 4510.0 | 0.36 | — | — | — | — |
| 300 | 28742 | 57622 | 0.50 | — | — | — | — |

`resultant` over the rationals, coefficients of 1,000 bits:

| degree | a µs | FLINT µs | a/FLINT | NTL µs | a/NTL | PARI µs | a/PARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 74.4 | 83.5 | 0.89 | — | — | — | — |
| 10 | 525.4 | 818.0 | 0.64 | — | — | — | — |
| 30 | 3055.0 | 8671.0 | 0.35 | — | — | — | — |
| 100 | 31871 | 88336 | 0.36 | — | — | — | — |

### Evaluation and interpolation

The median of three passes of each pass's median of five rounds, `n` coefficients at `n` points, `n` roots or `n` points. FLINT is `nmod_poly_evaluate_nmod_vec` (its own choice of Horner's rule or the tree), `nmod_poly_product_roots_nmod_vec`, `nmod_poly_interpolate_nmod_vec_fast`, `_newton` and `_barycentric` (against `interpolate_lagrange`), `fmpz_mod_poly_evaluate_fmpz_vec`, `fmpz_mod_poly_product_roots_fmpz_vec`, `fmpz_poly_evaluate_fmpz_vec`, `fmpz_poly_product_roots_fmpz_vec` and `fmpq_poly_interpolate_fmpz_vec`; NTL is `eval` on `vec_zz_p` (Horner's rule), `BuildFromRoots` and `interpolate` on `zz_pX` and `ZZ_pX`; PARI is `Flx_Flv_multieval`, `Flv_roots_to_pol`, `Flv_polint`, the `FpX` and `FpV` forms, `poleval` at each point and `RgV_polint` (`polinterpolate`). NTL's Horner's rule on vectors and PARI's evaluations over the integers and the rationals are timed only up to a size. The word moduli `p20`, `p50`, `p63` and `p64` are primes just below `2²⁰`, `2⁵⁰`, `2⁶³` and `2⁶⁴`; `p127`, `p130` and `p521` are `2¹²⁷ − 1`, `2¹³⁰ − 5` and `2⁵²¹ − 1`; `cBxC` are coefficients of `B` bits at points of `C` bits. The textbook forms are measured through the `Number` interface, conversions included. The tables give `a`'s time and its ratios to the references' times; `FLINT (tree)` is `nmod_poly_evaluate_nmod_vec_fast` and `fmpz_mod_poly_evaluate_fmpz_vec_fast`, the subproduct tree at every size.

`eval_many_u64` modulo `p20`, n coefficients at n points:

| n | `a` | a/FLINT | a/FLINT (tree) | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|---:|
| 1 | 11 ns | 3.69 | 0.48 | 3.34 | 0.72 |
| 4 | 34 ns | 1.52 | 0.40 | 1.50 | 0.50 |
| 16 | 96 ns | 0.41 | 0.17 | 0.17 | 0.16 |
| 64 | 605 ns | 0.14 | 0.14 | 0.04 | 0.12 |
| 256 | 9.2 µs | 0.21 | 0.21 | 0.04 | 0.20 |
| 1024 | 145.9 µs | 0.49 | 0.49 | 0.04 | 0.33 |
| 4096 | 788.6 µs | 0.43 | 0.43 | 0.01 | 0.22 |
| 16384 | 3.82 ms | 0.34 | 0.34 |  | 0.15 |
| 65536 | 18.68 ms | 0.30 | 0.30 |  | 0.12 |

`eval_many_u64` modulo `p50`, n coefficients at n points:

| n | `a` | a/FLINT | a/FLINT (tree) | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|---:|
| 1 | 11 ns | 3.69 | 0.49 | 3.06 | 0.79 |
| 4 | 34 ns | 1.50 | 0.38 | 1.50 | 0.29 |
| 16 | 95 ns | 0.39 | 0.10 | 0.16 | 0.06 |
| 64 | 1.2 µs | 0.17 | 0.17 | 0.08 | 0.05 |
| 256 | 18.1 µs | 0.35 | 0.35 | 0.07 | 0.07 |
| 1024 | 287.7 µs | 0.66 | 0.66 | 0.07 | 0.16 |
| 4096 | 1.47 ms | 0.46 | 0.46 | 0.02 | 0.12 |
| 16384 | 7.59 ms | 0.38 | 0.38 |  | 0.10 |
| 65536 | 37.84 ms | 0.35 | 0.35 |  | 0.09 |

`eval_many_u64` modulo `p63`, n coefficients at n points:

| n | `a` | a/FLINT | a/FLINT (tree) | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 11 ns | 3.69 | 0.50 | 0.78 |
| 4 | 34 ns | 1.52 | 0.38 | 0.29 |
| 16 | 189 ns | 0.44 | 0.21 | 0.11 |
| 64 | 2.8 µs | 0.41 | 0.41 | 0.11 |
| 256 | 43.5 µs | 0.75 | 0.75 | 0.16 |
| 1024 | 360.9 µs | 0.70 | 0.70 | 0.17 |
| 4096 | 1.83 ms | 0.51 | 0.51 | 0.12 |
| 16384 | 8.99 ms | 0.43 | 0.42 | 0.10 |
| 65536 | 43.80 ms | 0.38 | 0.38 | 0.08 |

`eval_many_u64` modulo `p64`, n coefficients at n points:

| n | `a` | a/FLINT | a/FLINT (tree) | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 11 ns | 3.69 | 0.50 | 0.79 |
| 4 | 34 ns | 1.51 | 0.39 | 0.30 |
| 16 | 190 ns | 0.29 | 0.21 | 0.11 |
| 64 | 2.7 µs | 0.41 | 0.41 | 0.11 |
| 256 | 43.4 µs | 0.75 | 0.75 | 0.16 |
| 1024 | 348.2 µs | 0.68 | 0.68 | 0.16 |
| 4096 | 1.78 ms | 0.49 | 0.50 | 0.11 |
| 16384 | 8.92 ms | 0.42 | 0.42 | 0.09 |
| 65536 | 45.09 ms | 0.39 | 0.39 | 0.08 |

`from_roots_u64` modulo `p20`, n roots:

| n | `a` | a/FLINT | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 80 ns | 29.81 | 6.19 | 12.01 |
| 4 | 94 ns | 3.34 | 5.20 | 3.74 |
| 16 | 246 ns | 0.71 | 1.52 | 1.33 |
| 64 | 1.2 µs | 0.63 | 0.63 | 0.94 |
| 256 | 6.4 µs | 0.59 | 0.22 | 0.72 |
| 1024 | 37.3 µs | 0.63 | 0.24 | 0.55 |
| 4096 | 198.0 µs | 0.54 | 0.24 | 0.40 |
| 16384 | 998.8 µs | 0.49 | 0.25 | 0.31 |
| 65536 | 4.93 ms | 0.47 | 0.25 | 0.26 |

`from_roots_u64` modulo `p50`, n roots:

| n | `a` | a/FLINT | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 76 ns | 28.04 | 6.15 | 9.58 |
| 4 | 94 ns | 3.30 | 5.18 | 2.22 |
| 16 | 238 ns | 0.68 | 1.46 | 0.50 |
| 64 | 1.8 µs | 0.74 | 0.90 | 0.46 |
| 256 | 13.5 µs | 0.87 | 0.46 | 0.46 |
| 1024 | 77.0 µs | 0.67 | 0.43 | 0.37 |
| 4096 | 403.5 µs | 0.57 | 0.41 | 0.28 |
| 16384 | 2.09 ms | 0.53 | 0.41 | 0.25 |
| 65536 | 10.71 ms | 0.53 | 0.41 | 0.23 |

`from_roots_u64` modulo `p63`, n roots:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 76 ns | 28.07 | 9.84 |
| 4 | 91 ns | 3.23 | 2.17 |
| 16 | 246 ns | 0.71 | 0.48 |
| 64 | 1.9 µs | 0.69 | 0.42 |
| 256 | 15.4 µs | 0.67 | 0.40 |
| 1024 | 90.7 µs | 0.63 | 0.33 |
| 4096 | 485.1 µs | 0.58 | 0.26 |
| 16384 | 2.47 ms | 0.55 | 0.22 |
| 65536 | 12.26 ms | 0.54 | 0.19 |

`from_roots_u64` modulo `p64`, n roots:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 76 ns | 28.26 | 9.91 |
| 4 | 106 ns | 3.79 | 2.57 |
| 16 | 288 ns | 0.83 | 0.57 |
| 64 | 2.3 µs | 0.84 | 0.50 |
| 256 | 16.5 µs | 0.73 | 0.45 |
| 1024 | 95.2 µs | 0.66 | 0.35 |
| 4096 | 502.1 µs | 0.60 | 0.27 |
| 16384 | 2.54 ms | 0.57 | 0.23 |
| 65536 | 12.96 ms | 0.58 | 0.21 |

`interpolate_u64` modulo `p20`, n points:

| n | `a` | a/FLINT | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 51 ns | 1.14 | 1.78 | 1.80 |
| 4 | 307 ns | 1.00 | 2.33 | 1.84 |
| 16 | 968 ns | 0.63 | 0.87 | 0.84 |
| 64 | 8.3 µs | 0.92 | 0.47 | 1.00 |
| 256 | 48.5 µs | 0.71 | 0.16 | 0.73 |
| 1024 | 250.6 µs | 0.61 | 0.05 | 0.43 |
| 4096 | 1.24 ms | 0.49 | 0.02 | 0.27 |
| 16384 | 6.02 ms | 0.40 |  | 0.19 |
| 65536 | 29.74 ms | 0.36 |  | 0.16 |

`interpolate_u64` modulo `p50`, n points:

| n | `a` | a/FLINT | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 51 ns | 1.16 | 1.77 | 1.69 |
| 4 | 360 ns | 0.68 | 1.43 | 1.19 |
| 16 | 1.0 µs | 0.36 | 0.54 | 0.35 |
| 64 | 9.4 µs | 0.58 | 0.46 | 0.29 |
| 256 | 89.5 µs | 0.87 | 0.28 | 0.29 |
| 1024 | 470.9 µs | 0.62 | 0.10 | 0.21 |
| 4096 | 2.36 ms | 0.48 | 0.03 | 0.16 |
| 16384 | 12.13 ms | 0.42 |  | 0.13 |
| 65536 | 61.26 ms | 0.40 |  | 0.12 |

`interpolate_u64` modulo `p63`, n points:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 50 ns | 1.16 | 1.68 |
| 4 | 392 ns | 0.65 | 1.23 |
| 16 | 1.1 µs | 0.35 | 0.37 |
| 64 | 11.2 µs | 0.59 | 0.32 |
| 256 | 106.4 µs | 0.81 | 0.30 |
| 1024 | 572.9 µs | 0.64 | 0.21 |
| 4096 | 2.91 ms | 0.51 | 0.15 |
| 16384 | 14.38 ms | 0.46 | 0.12 |
| 65536 | 71.03 ms | 0.44 | 0.11 |

`interpolate_u64` modulo `p64`, n points:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 50 ns | 1.13 | 1.68 |
| 4 | 393 ns | 0.67 | 1.27 |
| 16 | 1.2 µs | 0.38 | 0.39 |
| 64 | 12.0 µs | 0.64 | 0.34 |
| 256 | 100.8 µs | 0.77 | 0.28 |
| 1024 | 554.1 µs | 0.59 | 0.20 |
| 4096 | 2.83 ms | 0.51 | 0.14 |
| 16384 | 14.20 ms | 0.46 | 0.12 |
| 65536 | 72.50 ms | 0.45 | 0.11 |

`interpolate_newton` modulo `p20`, n points:

| n | `a` | a/FLINT |
|---:|---:|---:|
| 1 | 100 ns | 37.07 |
| 4 | 417 ns | 1.88 |
| 16 | 3.4 µs | 0.85 |
| 64 | 46.0 µs | 0.70 |
| 256 | 704.5 µs | 0.68 |
| 1024 | 11.12 ms | 0.67 |
| 4096 | 178.87 ms | 0.67 |

`interpolate_newton` modulo `p50`, n points:

| n | `a` | a/FLINT |
|---:|---:|---:|
| 1 | 101 ns | 37.48 |
| 4 | 480 ns | 0.90 |
| 16 | 3.4 µs | 0.33 |
| 64 | 45.6 µs | 0.27 |
| 256 | 702.5 µs | 0.25 |
| 1024 | 11.01 ms | 0.25 |
| 4096 | 178.78 ms | 0.25 |

`interpolate_newton` modulo `p63`, n points:

| n | `a` | a/FLINT |
|---:|---:|---:|
| 1 | 97 ns | 35.89 |
| 4 | 493 ns | 0.74 |
| 16 | 3.4 µs | 0.26 |
| 64 | 45.6 µs | 0.21 |
| 256 | 704.4 µs | 0.20 |
| 1024 | 11.34 ms | 0.20 |
| 4096 | 178.60 ms | 0.20 |

`interpolate_newton` modulo `p64`, n points:

| n | `a` | a/FLINT |
|---:|---:|---:|
| 1 | 98 ns | 36.44 |
| 4 | 494 ns | 0.74 |
| 16 | 3.4 µs | 0.26 |
| 64 | 45.0 µs | 0.21 |
| 256 | 704.8 µs | 0.20 |
| 1024 | 11.08 ms | 0.20 |
| 4096 | 178.88 ms | 0.20 |

`interpolate_lagrange` modulo `p20`, n points, against FLINT's barycentric form:

| n | `a` | a/FLINT |
|---:|---:|---:|
| 1 | 94 ns | 35.00 |
| 4 | 378 ns | 1.32 |
| 16 | 1.2 µs | 0.37 |
| 64 | 8.9 µs | 0.19 |
| 256 | 109.2 µs | 0.15 |
| 1024 | 1.60 ms | 0.14 |
| 4096 | 25.01 ms | 0.14 |

`interpolate_lagrange` modulo `p50`, n points, against FLINT's barycentric form:

| n | `a` | a/FLINT |
|---:|---:|---:|
| 1 | 96 ns | 35.67 |
| 4 | 447 ns | 0.90 |
| 16 | 1.3 µs | 0.31 |
| 64 | 10.3 µs | 0.21 |
| 256 | 125.8 µs | 0.17 |
| 1024 | 1.79 ms | 0.16 |
| 4096 | 27.48 ms | 0.15 |

`interpolate_lagrange` modulo `p63`, n points, against FLINT's barycentric form:

| n | `a` | a/FLINT |
|---:|---:|---:|
| 1 | 91 ns | 33.78 |
| 4 | 465 ns | 0.81 |
| 16 | 1.4 µs | 0.31 |
| 64 | 11.9 µs | 0.23 |
| 256 | 152.4 µs | 0.21 |
| 1024 | 2.21 ms | 0.19 |
| 4096 | 34.09 ms | 0.19 |

`interpolate_lagrange` modulo `p64`, n points, against FLINT's barycentric form:

| n | `a` | a/FLINT |
|---:|---:|---:|
| 1 | 93 ns | 31.97 |
| 4 | 472 ns | 0.84 |
| 16 | 1.4 µs | 0.32 |
| 64 | 12.8 µs | 0.24 |
| 256 | 159.3 µs | 0.20 |
| 1024 | 2.30 ms | 0.19 |
| 4096 | 35.51 ms | 0.19 |

`eval_many` modulo `p127`, n coefficients at n points:

| n | `a` | a/FLINT | a/FLINT (tree) | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|---:|
| 1 | 98 ns | 9.83 | 1.46 | 3.98 | 2.36 |
| 4 | 199 ns | 1.06 | 0.26 | 0.36 | 0.23 |
| 16 | 1.1 µs | 0.39 | 0.12 | 0.12 | 0.11 |
| 64 | 13.5 µs | 0.17 | 0.17 | 0.08 | 0.14 |
| 256 | 202.4 µs | 0.31 | 0.31 | 0.08 | 0.20 |
| 1024 | 2.25 ms | 0.54 | 0.55 | 0.05 | 0.28 |
| 4096 | 12.23 ms | 0.54 | 0.54 |  | 0.23 |

`eval_many` modulo `p130`, n coefficients at n points:

| n | `a` | a/FLINT | a/FLINT (tree) | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|---:|
| 1 | 126 ns | 13.03 | 1.89 | 5.12 | 3.08 |
| 4 | 286 ns | 0.37 | 0.25 | 0.40 | 0.23 |
| 16 | 2.0 µs | 0.13 | 0.15 | 0.13 | 0.15 |
| 64 | 26.3 µs | 0.23 | 0.23 | 0.10 | 0.20 |
| 256 | 403.4 µs | 0.45 | 0.45 | 0.10 | 0.31 |
| 1024 | 2.70 ms | 0.48 | 0.48 | 0.04 | 0.28 |
| 4096 | 14.42 ms | 0.48 | 0.47 |  | 0.24 |

`eval_many` modulo `p521`, n coefficients at n points:

| n | `a` | a/FLINT | a/FLINT (tree) | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|---:|
| 1 | 293 ns | 30.25 | 4.51 | 10.05 | 6.30 |
| 4 | 954 ns | 0.54 | 0.44 | 0.51 | 0.32 |
| 16 | 10.9 µs | 0.32 | 0.41 | 0.31 | 0.31 |
| 64 | 169.3 µs | 0.63 | 0.64 | 0.29 | 0.49 |
| 256 | 2.37 ms | 0.98 | 0.98 | 0.25 | 0.60 |
| 1024 | 12.99 ms | 0.66 | 0.66 | 0.08 | 0.40 |
| 4096 | 67.40 ms | 0.54 | 0.54 |  | 0.33 |

`from_roots` modulo `p127`, n roots:

| n | `a` | a/FLINT | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 182 ns | 15.79 | 8.25 | 5.28 |
| 4 | 316 ns | 2.10 | 1.21 | 0.65 |
| 16 | 1.7 µs | 0.45 | 0.36 | 0.45 |
| 64 | 14.1 µs | 0.52 | 0.59 | 0.53 |
| 256 | 91.3 µs | 0.53 | 0.71 | 0.51 |
| 1024 | 504.0 µs | 0.54 | 0.76 | 0.44 |
| 4096 | 2.62 ms | 0.55 | 0.78 | 0.39 |

`from_roots` modulo `p130`, n roots:

| n | `a` | a/FLINT | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 226 ns | 20.15 | 10.55 | 6.25 |
| 4 | 410 ns | 1.30 | 1.19 | 0.63 |
| 16 | 2.1 µs | 0.32 | 0.32 | 0.41 |
| 64 | 16.9 µs | 0.40 | 0.53 | 0.50 |
| 256 | 110.0 µs | 0.44 | 0.64 | 0.51 |
| 1024 | 602.5 µs | 0.46 | 0.68 | 0.46 |
| 4096 | 3.16 ms | 0.48 | 0.72 | 0.42 |

`from_roots` modulo `p521`, n roots:

| n | `a` | a/FLINT | a/NTL | a/PARI |
|---:|---:|---:|---:|---:|
| 1 | 424 ns | 35.36 | 15.26 | 9.84 |
| 4 | 1.1 µs | 1.25 | 1.12 | 0.71 |
| 16 | 7.9 µs | 0.48 | 0.45 | 0.52 |
| 64 | 75.9 µs | 0.64 | 0.91 | 0.62 |
| 256 | 499.9 µs | 0.65 | 1.06 | 0.59 |
| 1024 | 2.77 ms | 0.61 | 1.10 | 0.53 |
| 4096 | 14.57 ms | 0.60 | 1.12 | 0.49 |

`interpolate` modulo `p127`, n points:

| n | `a` | a/NTL | a/PARI |
|---:|---:|---:|---:|
| 1 | 106 ns | 0.75 | 0.63 |
| 4 | 1.0 µs | 0.53 | 0.45 |
| 16 | 8.7 µs | 0.36 | 0.47 |
| 64 | 92.9 µs | 0.27 | 0.60 |
| 256 | 587.5 µs | 0.11 | 0.43 |
| 1024 | 3.27 ms | 0.04 | 0.32 |
| 4096 | 17.73 ms | 0.01 | 0.27 |

`interpolate` modulo `p130`, n points:

| n | `a` | a/NTL | a/PARI |
|---:|---:|---:|---:|
| 1 | 111 ns | 0.80 | 0.68 |
| 4 | 1.3 µs | 0.58 | 0.43 |
| 16 | 11.9 µs | 0.38 | 0.49 |
| 64 | 126.9 µs | 0.24 | 0.63 |
| 256 | 735.2 µs | 0.09 | 0.43 |
| 1024 | 4.00 ms | 0.03 | 0.33 |
| 4096 | 21.17 ms | 0.01 | 0.28 |

`interpolate` modulo `p521`, n points:

| n | `a` | a/NTL | a/PARI |
|---:|---:|---:|---:|
| 1 | 154 ns | 0.80 | 0.82 |
| 4 | 4.2 µs | 0.63 | 0.54 |
| 16 | 48.5 µs | 0.56 | 0.71 |
| 64 | 585.2 µs | 0.47 | 1.01 |
| 256 | 3.53 ms | 0.18 | 0.63 |
| 1024 | 19.20 ms | 0.06 | 0.46 |
| 4096 | 99.07 ms | 0.02 | 0.38 |

`Polynomial<Number>::eval_many`, n coefficients of 64 bits at n points of 16 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 67 ns | 14.57 | 5.15 |
| 4 | 281 ns | 1.81 | 1.38 |
| 16 | 2.2 µs | 0.82 | 0.54 |
| 64 | 26.7 µs | 0.51 | 0.32 |
| 256 | 500.0 µs | 0.51 | 0.25 |
| 1024 | 13.86 ms | 0.66 |  |

`Polynomial<Number>::eval_many`, n coefficients of 64 bits at n points of 64 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 62 ns | 14.14 | 4.78 |
| 4 | 371 ns | 2.36 | 1.77 |
| 16 | 3.8 µs | 1.08 | 0.84 |
| 64 | 74.3 µs | 0.77 | 0.69 |
| 256 | 2.62 ms | 0.95 | 0.54 |
| 1024 | 74.86 ms | 0.90 |  |

`Polynomial<Number>::eval_many`, n coefficients of 1,000 bits at n points of 64 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 94 ns | 18.17 | 6.10 |
| 4 | 564 ns | 2.58 | 2.10 |
| 16 | 4.9 µs | 1.12 | 0.91 |
| 64 | 99.5 µs | 0.55 | 0.76 |
| 256 | 3.14 ms | 0.75 | 0.56 |
| 1024 | 84.51 ms | 0.76 |  |

`Polynomial<Number>::eval_many`, n coefficients of 64 bits at n points of 1,000 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 90 ns | 19.65 | 7.06 |
| 4 | 1.7 µs | 1.21 | 1.11 |
| 16 | 103.9 µs | 0.56 | 0.59 |
| 64 | 3.60 ms | 0.84 | 0.29 |
| 256 | 90.49 ms | 0.65 | 0.11 |

`Polynomial<Number>::from_roots`, n roots of 16 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 70 ns | 15.80 | 1.84 |
| 4 | 181 ns | 5.85 | 1.13 |
| 16 | 1.2 µs | 1.40 | 0.60 |
| 64 | 25.6 µs | 0.89 | 0.71 |
| 256 | 538.2 µs | 0.67 | 0.48 |
| 1024 | 11.47 ms | 0.65 | 0.46 |

`Polynomial<Number>::from_roots`, n roots of 64 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 70 ns | 12.03 | 1.92 |
| 4 | 216 ns | 3.48 | 1.14 |
| 16 | 2.2 µs | 1.79 | 0.44 |
| 64 | 98.4 µs | 0.60 | 0.46 |
| 256 | 2.48 ms | 0.58 | 0.34 |
| 1024 | 50.17 ms | 0.60 | 0.32 |

`Polynomial<Number>::from_roots`, n roots of 1,000 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 143 ns | 22.36 | 2.59 |
| 4 | 1.4 µs | 1.55 | 1.26 |
| 16 | 54.7 µs | 0.79 | 0.34 |
| 64 | 2.29 ms | 0.59 | 0.42 |
| 256 | 49.54 ms | 0.61 | 0.37 |

`Polynomial<Rational>::interpolate`, integer values of 64 bits at n integer points of 16 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 86 ns | 2.19 | 1.15 |
| 4 | 2.1 µs | 3.16 | 1.47 |
| 16 | 13.2 µs | 0.51 | 0.26 |
| 64 | 130.6 µs | 0.69 | 0.02 |
| 256 | 2.31 ms | 1.10 | 0.00 |
| 1024 | 50.11 ms | 0.81 |  |

`Polynomial<Rational>::interpolate`, integer values of 64 bits at n integer points of 64 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 83 ns | 2.28 | 1.11 |
| 4 | 2.0 µs | 3.04 | 1.34 |
| 16 | 12.5 µs | 0.48 | 0.25 |
| 64 | 129.4 µs | 0.69 | 0.02 |
| 256 | 2.30 ms | 1.09 | 0.00 |
| 1024 | 49.72 ms | 0.81 |  |

`Polynomial<Rational>::interpolate`, integer values of 1,000 bits at n integer points of 64 bits:

| n | `a` | a/FLINT | a/PARI |
|---:|---:|---:|---:|
| 1 | 102 ns | 2.48 | 1.16 |
| 4 | 3.3 µs | 4.14 | 1.33 |
| 16 | 27.9 µs | 2.97 | 0.41 |
| 64 | 290.9 µs | 0.17 | 0.04 |
| 256 | 3.98 ms | 1.48 | 0.01 |
| 1024 | 60.63 ms | 0.90 |  |

### Roots

The median of three passes of each pass's median of seven rounds, each measurement repeating a call for 20 ms. FLINT and Arb are `fmpz_poly_isolate_real_roots`, `fmpz_poly_num_real_roots`, `fmpz_poly_num_real_roots_sturm`, `arb_fmpz_poly_real_roots`, `arb_fmpz_poly_complex_roots` and `cd_poly_find_roots`; PARI is `polroots` at 64 bits on the doubles, `polroots` and `polrootsreal` at the precision, and `polsturm`; and the crate's `Matrix::eigenvalues` of the companion matrix. Arb's certified roots need a square-free input, which every input here is. In the table of the isolation `a_auto` is `real_root_intervals()`, against the two methods alone and FLINT.

`roots()` of random `Polynomial<f64>`s, the coefficients uniform in [−1, 1], against the eigenvalues of the companion matrix, FLINT's `cd_poly_find_roots` and PARI's `polroots` at 64 bits, in µs:

| case | a_roots (µs) | a_companion_eig (µs) | a/a_companion_eig | flint_cd_find_roots (µs) | a/flint_cd_find_roots | pari_polroots64 (µs) | a/pari_polroots64 |
|---|---|---|---|---|---|---|---|
| degree 10 | 4.2 | 4.4 | 0.98 | 4.8 | 0.88 | 720 | 0.01 |
| degree 30 | 31.7 | 32.4 | 0.98 | 68.8 | 0.46 | 7056 | 0.00 |
| degree 100 | 314 | 690 | 0.45 | 509 | 0.62 | 83994 | 0.00 |
| degree 300 | 2643 | 18689 | 0.14 | 4545 | 0.58 | 3784921 | 0.00 |
| degree 1000 | 29206 | 295936 | 0.10 | 53494 | 0.55 | – | – |

Largest backward error of a root, `|p(z)|/Σ|aᵢ||z|ⁱ` at 256 bits, over the rounds and passes:

| case | a_roots | a_companion_eig | flint_cd_find_roots | pari_polroots64 |
|---|---|---|---|---|
| degree 10 | 3.5e-16 | 4.2e-15 | 3.9e-01 | 3.3e-16 |
| degree 30 | 4.8e-16 | 6.0e-15 | 4.1e-01 | 4.8e-16 |
| degree 100 | 1.5e-15 | 3.1e-14 | 3.1e-01 | 8.1e-16 |
| degree 300 | 2.9e-15 | 1.2e-13 | 2.0e-01 | 1.3e-15 |
| degree 1000 | 5.3e-15 | 3.7e-13 | 4.2e-02 | nan |

Isolating intervals of the real roots, `real_root_intervals()` (`a_auto`) against `real_root_intervals_descartes()`, `real_root_intervals_cf()` and FLINT's `fmpz_poly_isolate_real_roots`, in µs:

| case | a_auto (µs) | a_descartes (µs) | a/a_descartes | a_cf (µs) | a/a_cf | flint_isolate (µs) | a/flint_isolate |
|---|---|---|---|---|---|---|---|
| random 10, 16 bits | 4.1 | 4.1 | 1.00 | 3.2 | 1.27 | 1.4 | 2.84 |
| random 20, 16 bits | 9.3 | 8.9 | 1.04 | 5.3 | 1.75 | 4.2 | 2.19 |
| random 50, 16 bits | 36.2 | 35.2 | 1.03 | 22.1 | 1.63 | 46.0 | 0.79 |
| random 100, 16 bits | 114 | 124 | 0.92 | 104 | 1.09 | 212 | 0.54 |
| random 200, 16 bits | 451 | 430 | 1.05 | 742 | 0.61 | 796 | 0.57 |
| random 500, 16 bits | 3011 | 2917 | 1.03 | 19331 | 0.16 | 14644 | 0.21 |
| random 1000, 16 bits | 27468 | 27159 | 1.01 | 93608 | 0.29 | 65886 | 0.42 |
| random 100, 1000 bits | 137 | 133 | 1.02 | 177 | 0.77 | 268 | 0.51 |
| random 300, 1000 bits | 899 | 847 | 1.06 | 1965 | 0.46 | 3243 | 0.28 |
| random 100, 10000 bits | 768 | 754 | 1.02 | 1628 | 0.47 | 2048 | 0.37 |
| Chebyshev T 20 | 26.5 | 26.3 | 1.01 | 40.6 | 0.65 | 38.8 | 0.68 |
| Chebyshev T 50 | 250 | 249 | 1.00 | 803 | 0.31 | 836 | 0.30 |
| Chebyshev T 100 | 1909 | 1913 | 1.00 | 10546 | 0.18 | 6656 | 0.29 |
| Chebyshev T 200 | 15819 | 15880 | 1.00 | 151241 | 0.10 | 66750 | 0.24 |
| Wilkinson 20 | 14.7 | 33.4 | 0.44 | 14.8 | 1.00 | 15.0 | 0.98 |
| Wilkinson 50 | 94.8 | 365 | 0.26 | 94.8 | 1.00 | 308 | 0.31 |
| Wilkinson 100 | 701 | 2892 | 0.24 | 701 | 1.00 | 2104 | 0.33 |
| Mignotte 20, a = 2^10+1 | 83.4 | 221 | 0.38 | 51.1 | 1.63 | 379 | 0.22 |
| Mignotte 40, a = 2^30+1 | 3620 | 29091 | 0.12 | 3392 | 1.07 | 31036 | 0.12 |
| Mignotte 100, a = 2^64+1 | 247167 | 8276070 | 0.03 | 311569 | 0.79 | 9697142 | 0.03 |
| sparse 1000, 64 bits | 10859 | 10865 | 1.00 | 3445 | 3.15 | 15940 | 0.68 |

`count_real_roots()` against FLINT's `fmpz_poly_num_real_roots` and `fmpz_poly_num_real_roots_sturm` and PARI's `polsturm`, in µs:

| case | a_count (µs) | flint_numreal (µs) | a/flint_numreal | flint_sturm (µs) | a/flint_sturm | pari_polsturm (µs) | a/pari_polsturm |
|---|---|---|---|---|---|---|---|
| random 50, 16 bits | 42.1 | 428 | 0.10 | 429 | 0.10 | 120 | 0.35 |
| random 200, 16 bits | 475 | 73263 | 0.01 | 73406 | 0.01 | 2399 | 0.20 |
| random 500, 16 bits | 2989 | 2121773 | 0.00 | 2121983 | 0.00 | 24067 | 0.12 |
| Chebyshev T 100 | 1930 | 12482 | 0.15 | 12450 | 0.15 | 1608 | 1.20 |
| Wilkinson 50 | 87.8 | 2444 | 0.04 | 2441 | 0.04 | 264 | 0.33 |
| Mignotte 40, a = 2^30+1 | 13.3 | 5.7 | 2.34 | 5.7 | 2.35 | 114712 | 0.00 |

`real_roots(prec)`, correctly rounded, against Arb's `arb_fmpz_poly_real_roots` and PARI's `polrootsreal` to the precision, in µs:

| case | a_real_roots (µs) | arb_real_roots (µs) | a/arb_real_roots | pari_polrootsreal (µs) | a/pari_polrootsreal |
|---|---|---|---|---|---|
| random 20, 16 bits, 53 bits | 31.5 | 28.8 | 1.10 | 51.7 | 0.61 |
| random 100, 16 bits, 53 bits | 313 | 314 | 1.00 | 1058 | 0.30 |
| random 100, 16 bits, 333 bits | 367 | 395 | 0.93 | 1176 | 0.31 |
| random 300, 16 bits, 53 bits | 2338 | 4311 | 0.54 | 13596 | 0.17 |
| Chebyshev T 30, 53 bits | 264 | 150 | 1.75 | 228 | 1.16 |
| Chebyshev T 100, 53 bits | 3951 | 3161 | 1.25 | 4251 | 0.93 |
| Chebyshev T 30, 3333 bits | 1632 | 631 | 2.58 | 1158 | 1.41 |
| Wilkinson 30, 333 bits | 31.8 | 82.5 | 0.39 | 121 | 0.26 |

`complex_roots(prec)`, certified, against Arb's `arb_fmpz_poly_complex_roots` and PARI's `polroots` to the precision, in µs:

| case | a_complex_roots (µs) | arb_complex_roots (µs) | a/arb_complex_roots | pari_polroots (µs) | a/pari_polroots |
|---|---|---|---|---|---|
| random 10, 16 bits, 53 bits | 71.4 | 118 | 0.61 | 844 | 0.08 |
| random 30, 16 bits, 53 bits | 440 | 1223 | 0.36 | 7035 | 0.06 |
| random 100, 16 bits, 53 bits | 4178 | 14861 | 0.28 | 91873 | 0.05 |
| random 300, 16 bits, 53 bits | 66357 | 169612 | 0.39 | 3713528 | 0.02 |
| random 30, 16 bits, 333 bits | 1065 | 2417 | 0.44 | 7805 | 0.14 |
| random 100, 16 bits, 333 bits | 10039 | 27057 | 0.37 | 100170 | 0.10 |
| random 30, 16 bits, 3333 bits | 13135 | 23701 | 0.55 | 25752 | 0.51 |
| random 100, 1000 bits, 53 bits | 4267 | 16001 | 0.27 | 122524 | 0.03 |
| Chebyshev T 30, 53 bits | 492 | 537 | 0.92 | 866 | 0.57 |
| Mignotte 30, a = 2^10+1, 53 bits | 2325 | 15925 | 0.15 | 7414 | 0.31 |
| Wilkinson 30, 100 bits | 2526 | 5151 | 0.49 | 10793 | 0.23 |

### Factorisation

The median over three passes of five rounds, each measurement the median call of its repetitions for 20 ms. FLINT's `nmod_poly`, `fmpz_mod_poly` and `fmpz_poly` factorisations, NTL's `zz_pX` (below `2⁶⁰`), `ZZ_pX` and `ZZX`, and PARI's `FpX` and `ZX` ones; the numbers of factors (or roots) agree with every reference in every round.

`factor` modulo a word prime, a random monic polynomial; `a`'s time over the reference's in parentheses:

| p | degree | `a` | FLINT nmod_poly_factor | NTL CanZass | PARI FpX_factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | 2.2 µs | 1.8 µs (1.20) | 5.3 µs (0.42) | 2.3 µs (0.95) |
| 3 | 30 | 9.4 µs | 13.0 µs (0.72) | 34.2 µs (0.27) | 25.4 µs (0.37) |
| 3 | 100 | 90.6 µs | 186.3 µs (0.49) | 397.8 µs (0.23) | 229.7 µs (0.39) |
| 3 | 300 | 1.24 ms | 1.73 ms (0.72) | 3.02 ms (0.41) | 2.58 ms (0.48) |
| 3 | 1000 | 27.41 ms | 17.96 ms (1.53) | 25.08 ms (1.09) | 41.00 ms (0.67) |
| 2^50 − 27 | 10 | 16.1 µs | 82.3 µs (0.20) | 21.7 µs (0.74) | 47.7 µs (0.34) |
| 2^50 − 27 | 30 | 74.5 µs | 122.2 µs (0.61) | 108.9 µs (0.68) | 262.4 µs (0.28) |
| 2^50 − 27 | 100 | 1.04 ms | 1.39 ms (0.75) | 1.71 ms (0.61) | 3.05 ms (0.34) |
| 2^50 − 27 | 300 | 9.46 ms | 22.23 ms (0.43) | 14.17 ms (0.67) | 31.91 ms (0.30) |
| 2^50 − 27 | 1000 | 69.96 ms | 203.48 ms (0.34) | 123.52 ms (0.57) | 465.86 ms (0.15) |
| 2^63 − 25 | 10 | 22.4 µs | 29.2 µs (0.77) | 170.9 µs (0.13) | 54.7 µs (0.41) |
| 2^63 − 25 | 30 | 118.2 µs | 170.9 µs (0.69) | 682.0 µs (0.17) | 401.6 µs (0.29) |
| 2^63 − 25 | 100 | 1.53 ms | 2.16 ms (0.71) | 4.54 ms (0.34) | 4.47 ms (0.34) |
| 2^63 − 25 | 300 | 13.14 ms | 22.95 ms (0.57) | 33.13 ms (0.40) | 45.47 ms (0.29) |
| 2^63 − 25 | 1000 | 91.02 ms | 212.62 ms (0.43) | 272.37 ms (0.33) | 636.91 ms (0.14) |
| 2^64 − 59 | 10 | 19.9 µs | 29.1 µs (0.68) | 178.8 µs (0.11) | 55.2 µs (0.36) |
| 2^64 − 59 | 30 | 107.9 µs | 165.2 µs (0.65) | 671.2 µs (0.16) | 408.8 µs (0.26) |
| 2^64 − 59 | 100 | 1.46 ms | 2.07 ms (0.71) | 4.55 ms (0.32) | 4.50 ms (0.33) |
| 2^64 − 59 | 300 | 12.85 ms | 23.19 ms (0.55) | 33.83 ms (0.38) | 47.10 ms (0.27) |
| 2^64 − 59 | 1000 | 82.81 ms | 190.81 ms (0.43) | 265.40 ms (0.31) | 636.88 ms (0.13) |

`factor_berlekamp` against Berlekamp's method of FLINT and NTL (NTL's zz_p only below 2^60):

| p | degree | `a` | FLINT berlekamp | NTL berlekamp |
| --- | ---: | ---: | ---: | ---: |
| 3 | 10 | 2.2 µs | 4.5 µs (0.50) | 6.6 µs (0.34) |
| 3 | 30 | 9.3 µs | 27.3 µs (0.34) | 30.6 µs (0.30) |
| 3 | 100 | 90.5 µs | 343.8 µs (0.26) | 298.1 µs (0.30) |
| 3 | 300 | 1.23 ms | 3.64 ms (0.34) | 1.81 ms (0.68) |
| 2^50 − 27 | 10 | 29.7 µs | 105.5 µs (0.28) | 33.4 µs (0.89) |
| 2^50 − 27 | 30 | 95.4 µs | 582.3 µs (0.16) | 113.5 µs (0.84) |
| 2^50 − 27 | 100 | 1.13 ms | 5.01 ms (0.23) | 1.66 ms (0.68) |
| 2^50 − 27 | 300 | 13.28 ms | 71.76 ms (0.19) | 13.61 ms (0.98) |
| 2^63 − 25 | 10 | 37.6 µs | 120.2 µs (0.31) |  |
| 2^63 − 25 | 30 | 134.5 µs | 736.4 µs (0.18) |  |
| 2^63 − 25 | 100 | 1.73 ms | 7.86 ms (0.22) |  |
| 2^63 − 25 | 300 | 17.38 ms | 93.46 ms (0.19) |  |
| 2^64 − 59 | 10 | 33.3 µs | 125.5 µs (0.27) |  |
| 2^64 − 59 | 30 | 144.2 µs | 808.1 µs (0.18) |  |
| 2^64 − 59 | 100 | 1.66 ms | 8.48 ms (0.20) |  |
| 2^64 − 59 | 300 | 17.39 ms | 88.11 ms (0.20) |  |

`is_irreducible` of a random irreducible polynomial:

| p | degree | `a` | FLINT nmod_poly_is_irreducible | NTL DetIrredTest | PARI FpX_is_irred |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | 2.0 µs | 3.2 µs (0.61) | 8.0 µs (0.25) | 2.2 µs (0.90) |
| 3 | 30 | 13.3 µs | 20.1 µs (0.66) | 71.8 µs (0.19) | 25.3 µs (0.52) |
| 3 | 100 | 261.9 µs | 220.6 µs (1.19) | 605.6 µs (0.43) | 248.5 µs (1.05) |
| 3 | 300 | 2.63 ms | 2.18 ms (1.21) | 4.83 ms (0.54) | 2.67 ms (0.98) |
| 2^50 − 27 | 10 | 14.0 µs | 19.0 µs (0.74) | 21.7 µs (0.65) | 44.2 µs (0.32) |
| 2^50 − 27 | 30 | 76.8 µs | 125.1 µs (0.61) | 194.1 µs (0.40) | 258.5 µs (0.30) |
| 2^50 − 27 | 100 | 1.18 ms | 1.66 ms (0.71) | 3.52 ms (0.33) | 3.00 ms (0.39) |
| 2^50 − 27 | 300 | 11.49 ms | 28.05 ms (0.41) | 22.14 ms (0.52) | 32.06 ms (0.36) |
| 2^63 − 25 | 10 | 21.0 µs | 27.1 µs (0.78) | 196.3 µs (0.11) | 54.1 µs (0.39) |
| 2^63 − 25 | 30 | 122.5 µs | 177.1 µs (0.69) | 964.2 µs (0.13) | 389.1 µs (0.31) |
| 2^63 − 25 | 100 | 1.79 ms | 2.78 ms (0.64) | 7.16 ms (0.25) | 4.40 ms (0.41) |
| 2^63 − 25 | 300 | 15.32 ms | 29.75 ms (0.51) | 47.37 ms (0.32) | 45.91 ms (0.33) |
| 2^64 − 59 | 10 | 19.7 µs | 27.4 µs (0.72) | 198.0 µs (0.10) | 54.8 µs (0.36) |
| 2^64 − 59 | 30 | 115.1 µs | 180.0 µs (0.64) | 983.5 µs (0.12) | 394.2 µs (0.29) |
| 2^64 − 59 | 100 | 1.78 ms | 2.80 ms (0.64) | 7.49 ms (0.24) | 4.46 ms (0.40) |
| 2^64 − 59 | 300 | 15.48 ms | 29.62 ms (0.52) | 48.04 ms (0.32) | 46.80 ms (0.33) |

`roots` of a random monic polynomial, with multiplicities:

| p | degree | `a` | FLINT nmod_poly_roots | PARI FpX_roots |
| --- | ---: | ---: | ---: | ---: |
| 3 | 10 | 291 ns | 459 ns (0.63) | 292 ns (1.00) |
| 3 | 30 | 750 ns | 917 ns (0.82) | 1.0 µs (0.72) |
| 3 | 100 | 2.4 µs | 4.9 µs (0.49) | 6.9 µs (0.35) |
| 3 | 300 | 7.1 µs | 30.6 µs (0.23) | 47.2 µs (0.15) |
| 3 | 1000 | 23.6 µs | 239.2 µs (0.10) | 369.5 µs (0.06) |
| 2^50 − 27 | 10 | 11.2 µs | 21.7 µs (0.52) | 33.1 µs (0.34) |
| 2^50 − 27 | 30 | 39.4 µs | 100.3 µs (0.39) | 138.2 µs (0.29) |
| 2^50 − 27 | 100 | 262.5 µs | 596.2 µs (0.44) | 720.4 µs (0.36) |
| 2^50 − 27 | 300 | 1.09 ms | 4.01 ms (0.27) | 3.36 ms (0.33) |
| 2^50 − 27 | 1000 | 3.10 ms | 12.50 ms (0.25) | 18.25 ms (0.17) |
| 2^63 − 25 | 10 | 18.0 µs | 33.8 µs (0.53) | 42.2 µs (0.43) |
| 2^63 − 25 | 30 | 68.2 µs | 162.2 µs (0.42) | 228.0 µs (0.30) |
| 2^63 − 25 | 100 | 482.8 µs | 1.14 ms (0.43) | 1.27 ms (0.38) |
| 2^63 − 25 | 300 | 1.77 ms | 5.29 ms (0.34) | 5.86 ms (0.30) |
| 2^63 − 25 | 1000 | 4.75 ms | 17.07 ms (0.28) | 31.00 ms (0.15) |
| 2^64 − 59 | 10 | 16.9 µs | 33.0 µs (0.51) | 42.8 µs (0.39) |
| 2^64 − 59 | 30 | 64.2 µs | 161.0 µs (0.40) | 235.6 µs (0.27) |
| 2^64 − 59 | 100 | 471.5 µs | 1.13 ms (0.42) | 1.29 ms (0.37) |
| 2^64 − 59 | 300 | 1.84 ms | 5.20 ms (0.35) | 5.98 ms (0.31) |
| 2^64 − 59 | 1000 | 5.30 ms | 16.37 ms (0.32) | 31.88 ms (0.17) |

`factor` modulo a prime of two limbs:

| p | degree | `a` | FLINT fmpz_mod_poly_factor | NTL CanZass (ZZ_pX) | PARI FpX_factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2^64 + 13 | 10 | 99.1 µs | 232.7 µs (0.43) | 193.8 µs (0.51) | 199.7 µs (0.50) |
| 2^64 + 13 | 30 | 644.3 µs | 1.17 ms (0.55) | 660.0 µs (0.98) | 1.67 ms (0.39) |
| 2^64 + 13 | 100 | 3.83 ms | 9.77 ms (0.39) | 4.73 ms (0.81) | 13.36 ms (0.29) |
| 2^127 − 1 | 10 | 341.0 µs | 564.2 µs (0.60) | 565.9 µs (0.60) | 597.0 µs (0.57) |
| 2^127 − 1 | 30 | 1.43 ms | 2.39 ms (0.60) | 1.45 ms (0.99) | 3.47 ms (0.41) |
| 2^127 − 1 | 100 | 9.45 ms | 18.63 ms (0.51) | 8.86 ms (1.07) | 25.68 ms (0.37) |

`factor` over the integers:

| input | degree | `a` | FLINT fmpz_poly_factor | NTL factor (ZZX) | PARI factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| random, 8-bit | 10 | 15.6 µs | 17.9 µs (0.87) | 28.0 µs (0.56) | 44.7 µs (0.35) |
| random, 8-bit | 30 | 46.0 µs | 103.5 µs (0.44) | 124.0 µs (0.37) | 317.2 µs (0.14) |
| random, 8-bit | 100 | 385.8 µs | 987.8 µs (0.39) | 1.42 ms (0.27) | 2.90 ms (0.13) |
| random, 8-bit | 300 | 3.33 ms | 9.55 ms (0.35) | 15.53 ms (0.21) | 15.25 ms (0.22) |
| random, 64-bit | 10 | 11.9 µs | 34.7 µs (0.34) | 32.6 µs (0.37) | 22.1 µs (0.54) |
| random, 64-bit | 30 | 84.1 µs | 181.8 µs (0.46) | 173.2 µs (0.49) | 356.2 µs (0.24) |
| random, 64-bit | 100 | 507.1 µs | 1.14 ms (0.44) | 1.74 ms (0.29) | 2.96 ms (0.17) |
| random, 1000-bit | 10 | 20.5 µs | 168.2 µs (0.12) | 41.5 µs (0.49) | 195.0 µs (0.11) |
| random, 1000-bit | 30 | 72.9 µs | 698.3 µs (0.10) | 181.6 µs (0.40) | 937.3 µs (0.08) |
| product of two, 8-bit | 10 | 26.4 µs | 28.4 µs (0.93) | 121.8 µs (0.22) | 57.7 µs (0.46) |
| product of two, 8-bit | 30 | 65.2 µs | 162.7 µs (0.40) | 613.8 µs (0.11) | 367.7 µs (0.18) |
| product of two, 8-bit | 100 | 432.3 µs | 1.35 ms (0.32) | 4.32 ms (0.10) | 3.24 ms (0.13) |
| product of two, 8-bit | 300 | 3.87 ms | 10.16 ms (0.38) | 28.27 ms (0.14) | 33.07 ms (0.12) |
| product of two, 64-bit | 30 | 186.0 µs | 315.3 µs (0.59) | 781.7 µs (0.24) | 484.9 µs (0.38) |
| product of two, 64-bit | 100 | 1.09 ms | 1.85 ms (0.59) | 5.11 ms (0.21) | 3.86 ms (0.28) |
| product of two, 1000-bit | 20 | 1.54 ms | 1.43 ms (1.08) | 1.80 ms (0.86) | 1.74 ms (0.88) |
| product of two, 10⁴-bit | 10 | 13.88 ms | 12.51 ms (1.11) | 12.75 ms (1.09) | 14.90 ms (0.93) |
| Π (x − i) | 10 | 54.7 µs | 44.8 µs (1.22) | 169.6 µs (0.32) | 64.5 µs (0.85) |
| Π (x − i) | 20 | 174.2 µs | 171.6 µs (1.01) | 533.0 µs (0.33) | 198.3 µs (0.88) |
| Π (x − i) | 50 | 779.3 µs | 919.2 µs (0.85) | 2.24 ms (0.35) | 963.5 µs (0.81) |
| Swinnerton-Dyer | 8 | 21.7 µs | 44.7 µs (0.49) | 152.0 µs (0.14) | 47.0 µs (0.46) |
| Swinnerton-Dyer | 16 | 58.7 µs | 171.5 µs (0.34) | 411.5 µs (0.14) | 137.1 µs (0.43) |
| Swinnerton-Dyer | 32 | 539.0 µs | 781.4 µs (0.69) | 2.05 ms (0.26) | 808.1 µs (0.67) |
| xⁿ − 1 | 30 | 89.1 µs | 407.2 µs (0.22) | 495.4 µs (0.18) | 185.2 µs (0.48) |
| xⁿ − 1 | 60 | 322.8 µs | 1.18 ms (0.27) | 1.16 ms (0.28) | 418.5 µs (0.77) |
| xⁿ − 1 | 105 | 758.2 µs | 1.45 ms (0.52) | 3.90 ms (0.19) | 1.19 ms (0.63) |
| xⁿ − 1 | 120 | 894.9 µs | 5.79 ms (0.15) | 2.93 ms (0.31) | 998.2 µs (0.90) |
| xⁿ − 1 | 210 | 3.26 ms | 5.59 ms (0.58) | 9.46 ms (0.34) | 3.83 ms (0.85) |

### Where the references are faster

Every row where `a` takes more than 1.10 times a reference's time, by cause:

- **Fixed costs, at 1 to 4 coefficients, points or roots**, and for some operations to 16: `a`'s operators and functions return a new polynomial each time (an allocation, about 7 ns, and the modulus's reference count, about 4 ns), where the timing loops of FLINT and NTL write into storage that exists; a `ModPolynomial` made from points or roots builds its modulus too (a reference-counted `Number`, three allocations), and a `Number` coefficient is a vector of its own, where FLINT's `fmpz` keeps a small one in a word. Products modulo a word of one coefficient take 18 to 19 ns against FLINT's 5 to 6 (3.4 to 3.5 times); `inv_series`, `compose` and `div_rem` of 2 to 4 coefficients 1.1 to 5.5 times FLINT's time, `compose_mod` of 4 1.25 to 1.8, products over the integers of 1 to 4 coefficients 1.3 to 5.9 and modulo several limbs 1.1 to 1.35; and at one point or root, `from_roots_u64` 28 to 30 times FLINT's (76 ns), `interpolate_newton` and `interpolate_lagrange` 32 to 37 times (about 100 ns), `eval_many` modulo several limbs 10 to 30 times and over the integers 14 to 20. At 4 points `eval_many_u64` is 1.5 times FLINT's and NTL's time, `interpolate_u64` 1.2 to 2.3 times NTL's and PARI's, and `from_roots_u64` 2.2 to 5.2 times the references'; from 16 `a` is faster modulo `n` everywhere but at `from_roots_u64` of 16 roots modulo `p20` and `p50` (1.3 to 1.5 times NTL's and PARI's). At 8 to 16 coefficients modulo a word FLINT is still 1.1 to 3.0 times faster at `inv_series`, 1.2 to 1.8 at `div_rem`, 1.2 to 1.7 at `compose` and 1.1 to 1.4 at products; NTL's `zz_pX` 1.1 to 2.8 at products, `div_rem` and `inv_series` of 8 and 16 coefficients, and 1.2 to 2.9 at squares modulo `2⁵⁰ − 27` up to 64. Over the integers and the rationals the gcds, resultants and discriminants of degree 5 to 10 take 1.1 to 3.9 times the references' time (the heuristic gcd and the divisions build one vector a coefficient), `xgcd` modulo a 17-bit prime at degree 10 1.5 times; the isolation of the real roots of degree 10 and 20 2.2 to 2.8 times (4.1 µs against 1.4), `count_real_roots` of Mignotte's polynomial of degree 40 2.3 times (13 µs against 5.7: the square-free test and Sturm's sequence of pseudo-remainders over `Number`s), and the factorisations of degree 10 modulo 3 and of `Π (x − i)` 1.2 times.
- **`Number` coefficients**, each a vector of bytes that every product reads and builds anew (about 10 ns a coefficient), against FLINT's `fmpz`, a word for a small value: products and squares over the integers of 32-bit coefficients 1.6 to 5.5 times FLINT's time from 8 to 10⁴ coefficients (1.2 to 1.4 at 3·10⁴ and 10⁵), of 64-bit ones 1.1 to 1.9 from 128 coefficients, and of 1,000 bits 1.15 to 2.4 at 8 to 64 coefficients (NTL 1.1 to 2.3); `pseudo_div_rem` of 64-bit coefficients at 8 to 32 quotient coefficients 1.1 to 1.3, its slots taking multi-limb values uniformly (from 64, 0.5 to 1.03 of FLINT's time); `from_roots` of 16 roots 1.4 to 1.8, and `eval_many` of 1,000-bit coefficients at 16 points 1.1. `div_exact` of 1,000-bit coefficients at 16 to 64 quotient coefficients is 1.15 to 1.33 times FLINT's `fmpz_poly_divides`, which divides by divide and conquer with an early abort, against the schoolbook method up to 48 and Kronecker substitution beyond. The largest products, 3·10⁴ coefficients of 10⁴ bits (1.6 times FLINT's, both for products and squares), are one product of integers of about 3·10⁸ bits each by Kronecker substitution; that size has not been looked into further.
- **Coefficients in lowest terms over the rationals**: products of `Polynomial<Rational>` with parts of 31 bits take 2.5 to 9.4 times FLINT's time from 2 coefficients, `interpolate` of 1,000-bit values 1.5 to 3.0 at 16 to 256 points, and the gcds over the rationals 1.1 to 1.4 to degree 100. FLINT's `fmpq_poly` keeps one common denominator; every coefficient here is a `Rational` in lowest terms, which costs a gcd of numbers the size of the common denominator for each coefficient of the result, most of the time of a product (the product itself about 5 %).
- **Moduli below `2³²`**: products and squares modulo 65,521 from 256 to 10⁴ coefficients take 1.04 to 1.40 times FLINT's time, `div_rem` from 512 to 10⁵ 1.1 to 1.6, `mul_mod` from 1,000 1.14 to 1.55, `inv_series` at 5,000 and 10⁴ 1.2, `compose` at 32 to 512 1.2 to 1.3 and `compose_mod` at 5,000 1.33. FLINT's `fft_small` takes one 50-bit prime and truncated transforms; `a` takes two 29-bit primes and lengths of powers of two, so that a product of 10⁴ by 10⁴ coefficients takes 32,768 points, 39 % of them unused. The gcds, extended gcds and resultants modulo a 17-bit prime from degree 1000 take 1.15 to 1.34 times FLINT's time, whose arithmetic modulo a prime below `2³²` reduces lazily in one word, where `a`'s Euclid's steps are the 64-bit kernel's; `is_irreducible` modulo 3 at degree 100 and 300 (1.2 times) and `factor` modulo 3 at degree 1000 (1.5 times FLINT's, 1.09 NTL's) spend their time in those gcds.
- **The division family modulo a word at 8 to 512 coefficients**: `div_rem` modulo 63- and 64-bit primes 1.13 to 1.30 times FLINT's time, `mul_mod` modulo `2⁶³ − 25` 1.14 to 1.33 to 64 coefficients, and products modulo 50- and 63-bit primes at 64 coefficients, where the transforms take over, 1.25 to 1.35. FLINT has middle products and a divide-and-conquer division there; `a` takes the schoolbook division up to 256 coefficients and Newton's method beyond.
- **Moduli of 4 to 16 limbs against NTL's `ZZ_pX`**: products modulo `2²⁵⁵ − 19`, `2⁵²¹ − 1` and `2¹⁰²⁴ − 105` from 256 coefficients take 1.11 to 1.60 times NTL's time, squares from 128 coefficients 1.2 to 1.75, `div_rem` 1.13 to 1.79 and `x.pow_mod` from 32 coefficients 1.1 to 1.73. NTL's 60-bit primes hold a 127-bit residue's sums of products with about five primes and no pieces, one forward and one inverse transform a product, and NTL reduces modulo `f` in its transforms; with 29-bit primes `a` cuts each residue into 2 to 6 pieces, and `2¹⁰²⁴ − 105` stays on Kronecker substitution, the pieces measured slower there. FLINT is faster at `div_rem` of 64 coefficients modulo `2⁵²¹ − 1` and `2¹⁰²⁴ − 105` (1.5 to 1.75 times) and at products of 8 coefficients modulo `2¹⁰²⁴ − 105` (1.2 to 1.45). The gcds, extended gcds and resultants modulo primes of 2 limbs from degree 1000 take 1.11 to 1.24 times NTL's time and of 3 to 16 limbs from degree 300 1.24 to 1.73: they run on those products, and Euclid's steps on limbs invert their leading coefficients through `Number`'s modular inverse for divisors of 24 coefficients and more. `from_roots` of 4096 roots modulo `2⁵²¹ − 1` is 1.12 times NTL's.
- **Real roots of Chebyshev's polynomials**, whose roots crowd towards ±1: `real_roots` takes 1.25 to 2.6 times Arb's time and 1.16 to 1.41 PARI's at the degrees 30 and 100, the refinement carrying the conditioning it finds (about 120 bits near ±1 for `T₁₀₀`) into the precision of every evaluation; `count_real_roots` of `T₁₀₀` is 1.2 times PARI's `polsturm`.
- **PARI's `ZX_resultant` and NTL's `GCD` at some sizes**: the resultants over the integers at degree 300 (64-bit coefficients) and 100 (1,000 bits) take 1.6 to 1.9 times PARI's time (the discriminants, through the resultant, 1.55 to 2.0), where `a` takes 0.36 to 0.49 of FLINT's and NTL's; and the gcd with a common factor of 3000 coefficients of 64 bits takes 1.5 times NTL's time (1.00 FLINT's). The causes were not investigated.

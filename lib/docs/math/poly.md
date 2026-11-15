[A](../../../README.md) › [Math](../../../README.md#math)

# Polynomials

`Polynomial<T>` (in `a::math::poly`) is a polynomial in one variable with coefficients of the type `T`: a [`Number`](numbers.md), a [`Rational`](rational.md), an `f64` or a [`Complex<f64>`](complex.md). `ModPolynomial` is a polynomial with coefficients modulo an integer `n ≥ 2`, which it holds once for all its coefficients and shares with its copies. Both are dense: the coefficients, the constant first, without zeros at the top, so that the zero polynomial has none and every other its leading coefficient last.

- `Polynomial::new(coeffs)` takes the coefficients, the constant first, and drops the zeros at the top; `zero`, `one`, `x`, `constant(c)` and `monomial(c, n)` make the simplest ones. `coeffs` gives them back, `coeff(i)` one of them (0 above the degree), `degree` is `None` for the zero polynomial, and `length`, as FLINT's `fmpz_poly_length`, is the degree plus one.
- They add with `+` and `+=`, subtract with `-` and `-=` and multiply with `*` and `*=`, each operand owned or borrowed; `scale(&c)` multiplies by a constant, `sqr` squares and `pow(n)` raises to a power. Over a field (`Rational`, `f64`, `Complex<f64>`) `/` and `%` give the quotient and the remainder, `div_rem` both; over the integers `pseudo_div_rem` gives `lᵏ·a = q·d + r` with `l` the leading coefficient of `d`, and `div_exact` the quotient when there is no remainder.
- `eval(&x)` evaluates by Horner's rule, `derivative` differentiates and `compose(&g)` gives `p(g(x))`; `shift_left`, `shift_right`, `truncate` and `reverse` move and cut the coefficients, and `map` makes a polynomial of other coefficients.
- `content` and `primitive_part` split a polynomial over the integers into the greatest common divisor of its coefficients and the rest, with a positive leading coefficient; `to_primitive` splits a polynomial over the rationals into a rational and a primitive polynomial over the integers.
- `gcd` is the monic greatest common divisor over a field and `xgcd` its cofactors too, `g = s·a + t·b`; over the integers `gcd` is the greatest common divisor of the contents times that of the primitive parts.
- `Display` writes the terms from the leading one down, as PARI/GP does: `3*x^2 - x + 1`, a complex coefficient in parentheses.
- `ModPolynomial::new(coeffs, &n)` reduces each coefficient to its residue in `0..n`, `from_u64(coeffs, n)` takes words, and `from_polynomial` a polynomial over the integers; `to_polynomial` and `to_symmetric` give one back, with the residues in `0..n` or in `-n/2..n/2`. A residue is held as the limbs of `n`, one word for `n < 2^64`. The same arithmetic, evaluation, derivative and composition as `Polynomial`, and for a prime `n` (or whenever the leading coefficients it divides by are units) `div_rem`, `/`, `%`, `to_monic`, `gcd`, `xgcd`, `mul_mod` and `pow_mod`; operations on two polynomials need the same modulus and panic otherwise. `Display` writes `x^2 + 5 (mod 7)`.

## Examples

```rust
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;

let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
let a = p(&[-1, 0, 1]); // x² - 1
let b = p(&[1, 1]); // x + 1
assert_eq!((&a * &b).to_string(), "x^3 + x^2 - x - 1");
assert_eq!(a.div_exact(&b), Some(p(&[-1, 1])));
assert_eq!(a.gcd(&p(&[2, 2])), p(&[1, 1]));
assert_eq!(a.eval(&Number::from(3)), Number::from(8));

let q: Polynomial<Rational> = a.map(|c| Rational::from(c.clone()));
let (quot, rem) = q.div_rem(&Polynomial::new(vec![Rational::from(1), Rational::from(2)]));
assert_eq!(quot.to_string(), "1/2*x - 1/4");
assert_eq!(rem.to_string(), "-3/4");

let f = ModPolynomial::from_u64(&[1, 0, 1], 7); // x² + 1, irreducible modulo 7
let x = ModPolynomial::from_u64(&[0, 1], 7);
assert_eq!(x.pow_mod(&Number::from(49), &f), x);
```

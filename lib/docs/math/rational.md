[A](../../../README.md) › [Math](../../../README.md#math)

# Rational numbers

`Rational` (in `a::math::rational`) is the fraction `p/q` of two [numbers](numbers.md), always in lowest terms with a positive denominator, so that each rational number has one form: zero is `0/1`, an integer `n` is `n/1`, and two rationals are equal when their parts are. The parts are private, so that the form holds; `numer`, `denom` and `into_parts` give them back.

- `Rational::new(p, q)` reduces any fraction with `q ≠ 0` to the form, `Rational::from_integer(n)` and `From` make integers, from a `Number` or any primitive integer.
- They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` and divide with `/` and `/=`, each operand owned or borrowed, and take a `Number` or a primitive integer as an operand on either side: `r + 1`, `2 * &r`, `&n / r`. Dividing by zero panics, as it does for integers.
- `-r` negates, `recip` is `1/r`, `abs` and `signum` are as for integers, and `pow` takes an `i32` exponent, negative ones by the reciprocal; `0⁰` is 1.
- `==`, `<` and the rest compare by value, with numbers on either side and primitive integers on the right; `Hash` agrees with `==`, `Default` is zero, `Sum` and `Product` take iterators of owned or borrowed rationals. `Display` and `Debug` write `-3/4`, or `5` for an integer, the formatter's width, fill and `+` applying to the whole.
- `floor`, `ceil`, `trunc`, `round` (halves away from zero, as `f64::round`) and `round_ties_even` give integers as `Number`s, and `fract` the rational `r − trunc(r)`, of the sign of `r`.
- Each part is in the byte order of the numerator, and the result of an operator in the byte order of its left operand, as for numbers.

## Examples

```rust
use a::math::numbers::Number;
use a::math::rational::Rational;

let a = Rational::new(Number::from(6), Number::from(-8));
assert_eq!(a.to_string(), "-3/4");
let b = Rational::from(1) / 3u32;
assert_eq!((&a + &b).to_string(), "-5/12");
assert_eq!((&a * &b).to_string(), "-1/4");
assert_eq!((&a / &b).to_string(), "-9/4");
assert!(a < b);

assert_eq!(a.floor(), -1);
assert_eq!(a.round(), -1);
assert_eq!(a.fract().to_string(), "-3/4");
assert_eq!(a.pow(-2).to_string(), "16/9");

let h: Rational = (1..=10u32).map(|k| Rational::from(1) / k).sum();
assert_eq!(h.to_string(), "7381/2520");
```

The method:

- A sum `a/b + c/d` is reduced as Knuth's *The Art of Computer Programming*, volume 2, section 4.5.1, has it: with `g = gcd(b, d)`, the numerator `t = a·(d/g) + c·(b/g)` can only share a factor with `g`, so the result is `(t/g₂) / ((b/g)·(d/g₂))` with `g₂ = gcd(t, g)`. Two gcds of the short parts take the place of one of the long unreduced ones. Equal denominators, and integers on either side, skip the gcds they need not take.
- A product `(a/b)·(c/d)` is `((a/g₁)·(c/g₂)) / ((b/g₂)·(d/g₁))` with `g₁ = gcd(a, d)` and `g₂ = gcd(c, b)`, which is in lowest terms as it is; a quotient is the product by the reciprocal.
- When every part fits in 64 bits, the arithmetic, the comparisons and the rounding are done in registers, the products in 128 bits, and the results written in the vectors of the operand assigned to or taken by value; a sum whose numerator would pass 128 bits takes the general path.
- Two rationals compare by their signs, then by `a·d` against `c·b`, whose lengths decide most pairs without a product.

## Efficiency

- Time complexity: a sum or a product of parts of `n` limbs takes two or three greatest common divisors and four products of `n`-limb numbers, `O(M(n) log n)` with the half gcd for the longest; a comparison at most two products.
- Space complexity: `O(n)`, the parts and the intermediate products.

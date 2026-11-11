[A](../../../README.md) › [Math](../../../README.md#math)

# Rational numbers

`Rational` (in `a::math::rational`) is the fraction `p/q` of two [numbers](numbers.md), always in lowest terms with a positive denominator, so that each rational number has one form: zero is `0/1`, an integer `n` is `n/1`, and two rationals are equal when their parts are. The parts are private, so that the form holds; `numer`, `denom` and `into_parts` give them back.

- `Rational::new(p, q)` reduces any fraction with `q ≠ 0` to the form, `Rational::from_integer(n)` and `From` make integers, from a `Number` or any primitive integer.
- They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` and divide with `/` and `/=`, each operand owned or borrowed, and take a `Number` or a primitive integer as an operand on either side: `r + 1`, `2 * &r`, `&n / r`. Dividing by zero panics, as it does for integers.
- `-r` negates, `recip` is `1/r`, `abs` and `signum` are as for integers, and `pow` takes an `i32` exponent, negative ones by the reciprocal; `0⁰` is 1.
- `==`, `<` and the rest compare by value, with numbers on either side and primitive integers on the right; `Hash` agrees with `==`, `Default` is zero, `Sum` and `Product` take iterators of owned or borrowed rationals. `Display` and `Debug` write `-3/4`, or `5` for an integer, the formatter's width, fill and `+` applying to the whole.
- `Rational::try_from(x)` takes any finite `f64` or `f32` exactly: every float is `m·2^e`, so `0.1` is `3602879701896397/36028797018963968`, and `-0.0` is zero; NaN and the infinities are a `TryFromFloatError`. `to_f64` and `to_f32` go back to the nearest float, ties to even, rounded once and correctly, subnormal numbers included: a number too large is infinity of its sign, one too small zero, `-0.0` for a negative one as IEEE 754 has it, and `0.0` for zero.
- `Number::try_from(&r)` and `u8::try_from(&r)` up to `i128` and `usize` give an integer, a `TryFromRationalError` for a fraction or an integer out of the type's range.
- `"-3/4".parse::<Rational>()` reads integers, fractions `p/q` (the denominator without a sign, not zero, the fraction reduced) and decimals `1.25`, `.5`, `5.`, with an exponent `e` or `E` on decimals and integers, `-1.5e-3`, `2e10`, exactly: `0.1` is `1/10`. There is no whitespace, no `_`, nothing else, as for a `Number`. The exponent may be up to `Rational::MAX_PARSE_EXPONENT`, `1000000`, in magnitude: `1e1000000` is a numerator of 415 KB, read in about a millisecond, and the bound keeps a short text from asking for gigabytes. A `ParseRationalError` has a `kind()`: `Empty`, `InvalidDigit`, `ZeroDenominator` or `ExponentOutOfRange`.
- `to_str_radix(radix)` writes `-p/q` in a radix from 2 to 36, and `from_str_radix(s, radix)` reads integers and fractions in it, letters in either case, with no point or exponent in any radix (`e` is a digit from radix 15 on).
- `floor`, `ceil`, `trunc`, `round` (halves away from zero, as `f64::round`) and `round_ties_even` give integers as `Number`s, and `fract` the rational `r − trunc(r)`, of the sign of `r`.
- Each part is in the byte order of the numerator, and the result of an operator in the byte order of its left operand, as for numbers.
- Continued fractions, the best approximations with a bounded denominator, the simplest rational in an interval and rational reconstruction modulo `m` are [below](#continued-fractions-and-approximations); the Bernoulli and harmonic numbers [after them](#bernoulli-and-harmonic-numbers); and matrices of rationals, solved and inverted exactly, [last](#matrices-of-rationals).

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

```rust
use a::math::rational::{Rational, RationalErrorKind};

let r: Rational = "-1.5e-3".parse().unwrap();
assert_eq!(r.to_string(), "-3/2000");
assert_eq!(r.to_f64(), -0.0015);
assert_eq!(Rational::try_from(0.75).unwrap().to_string(), "3/4");
assert_eq!(Rational::from_str_radix("-ff/1e", 16).unwrap().to_str_radix(16), "-11/2");
assert_eq!(*"1/0".parse::<Rational>().unwrap_err().kind(), RationalErrorKind::ZeroDenominator);
```

The method:

- A sum `a/b + c/d` is reduced as Knuth's *The Art of Computer Programming*, volume 2, section 4.5.1, has it: with `g = gcd(b, d)`, the numerator `t = a·(d/g) + c·(b/g)` can only share a factor with `g`, so the result is `(t/g₂) / ((b/g)·(d/g₂))` with `g₂ = gcd(t, g)`. Two gcds of the short parts take the place of one of the long unreduced ones. Equal denominators, and integers on either side, skip the gcds they need not take.
- A product `(a/b)·(c/d)` is `((a/g₁)·(c/g₂)) / ((b/g₂)·(d/g₁))` with `g₁ = gcd(a, d)` and `g₂ = gcd(c, b)`, which is in lowest terms as it is; a quotient is the product by the reciprocal.
- When every part fits in 64 bits, the arithmetic, the comparisons and the rounding are done in registers, the products in 128 bits, a sum whose numerator passes 128 bits with its 129th bit apart and Hensel's exact division modulo `2^128`. Parts of up to 16 bytes are read straight from their bytes as `u128`s: the gcds of two `u128`s, the exact quotients by Hensel's division modulo `2^128`, the products in four limbs and a sum's numerator in five, all in registers, the limbs taking only a sum whose denominators share more than 64 bits. Otherwise the four parts are read into limbs once, the gcds (Lehmer's method and the half gcd of [numbers](numbers.md)), exact divisions and products are done on limbs in working memory (in the operation's own frame for parts of up to 12 limbs together), and the two parts of the result are written once, into the vectors of the operand assigned to or taken by value where they have room, with an eighth to spare when they grow. A part of one limb takes the steps for one limb: its gcd with a long part is a remainder by it, by the powers of `B` modulo it (computed once for both remainders of a sum), and the exact divisions and products by it are single passes. This is what a sum of many terms does at every term: `H_10000` by `+=` takes 0.45 of the time of the same sum on numbers.
- `Sum` and `Product` combine their terms in a balanced tree, as a binary counter merges them, with at most `log₂ n` partial results kept: each addition is of two sums of about the same number of terms, so the parts of a sum of many short terms grow in a few long additions rather than one long addition a term. `H_n` takes `O(M(n) log n)` instead of `O(n²)`: `H_100000` takes 1,259 cycles a term against 14,131 by `+=`.
- `Rational::new` reduces in the vectors it is given: parts of up to 8 bytes by the gcd of two limbs, up to 16 by that of two `u128`s and Hensel's exact division, longer ones on limbs; a gcd of 1, the common case, writes nothing.
- Two rationals compare by their signs, then by `a·d` against `c·b`, first bounded from the top 63 bits of each part: a part of `L > 63` bits lies in `[h, h + 1]·2^(L − 63)` for its top 63 bits `h`, and the products of the bounds, in 126 bits, decide unless the two numbers agree to about 120 bits. So a comparison takes a few dozen cycles at any length, and only numbers that close are multiplied out.
- `floor`, `ceil` and `trunc` take the truncated quotient, one less or more by the sign (a remainder over a denominator above 1 is never zero in lowest terms); `round` and `round_ties_even` the quotient and the remainder on limbs, the remainder doubled against the denominator; parts of up to 16 bytes in registers.
- A float is `±m·2^e` from its bits: `m` is divided by its trailing zeros, which are added to `e`, and the rational is `m·2^e` when `e ≥ 0` and `m/2^-e` otherwise, in lowest terms as it is, both parts built directly as little-endian bytes. Its cost is the two vectors a rational has.
- `to_f64` of `a/b`: when both parts are below `2^53`, both are floats exactly and one division of floats is correctly rounded. Otherwise the top 128 bits of `|a|` are divided by the top 64 bits of `b` in registers, by the reciprocal of the divisor as a division of limbs takes it (Möller and Granlund's), which gives a quotient `q` of 64 or 65 bits with `q − 4 < |a|/b·2^s < q + 1`; the result is `q` rounded with a sticky bit, unless the one halfway point between two floats near `q` lies in that interval. When `b` fits in 64 bits the quotient and its remainder are exact, and the bits of `a` below its top 128 are read only when the remainder is zero. Otherwise, about one fraction in 700, `|a|·2^s` (or `|a|/2^s`, its dropped bits kept for the sticky bit) is divided by `b` for a quotient of 65 or 66 bits, its remainder setting the sticky bit: linear in the length of `b`. The rounding is done on the integer bits, without branches, and a carry out of the significand runs into the exponent and up to infinity on its own. GMP's `mpq_get_d` and FLINT's `fmpq_get_d`, which calls it, truncate towards zero instead, so they differ from the nearest float in about half of all fractions; they also divide at the full length of the denominator, so they take time linear in it.
- `to_f32` rounds the rational to 24 bits the same way, once: rounding the `f64` again would be one off where the `f64` lands on a halfway point between two `f32`s.
- `FromStr` finds the first `/`, `.`, `e` or `E` and reads the parts as numbers are read (by divide and conquer from a few hundred digits). A fraction is reduced by the gcd of its parts, which, for text already in lowest terms, is 81 to 92% of the time from 100 digits on. A decimal `d₁…dₙ.f₁…fₘ e x` is the integer `d₁…dₙf₁…fₘ`, its trailing zeros dropped, times `10^(x − m)`, built as `5^k·2^k`. A negative power leaves `N/10^k` in lowest terms but for the factors 2 or 5 of `N`, which its last digit tells apart: `1, 3, 7, 9` share neither, an even digit only twos, which a shift removes, and `5` only fives, which one remainder by `5^27` counts (a gcd with `5^k` past 27). Up to 19 digits, with an exponent of up to 19, are read in registers, the fraction reduced by the binary gcd of two words.
- `Display` writes parts of up to 64 bits two digits at a time from a table, and fractions of up to 320 characters on the stack, so that the only allocation is the text the formatter keeps; `to_str_radix(10)` writes both parts into one vector.

## Continued fractions and approximations

- `continued_fraction` gives the regular continued fraction `[a0; a1, …, an]` of a rational as `Number`s in its byte order: `a0 = ⌊x⌋`, which may be negative, the other terms positive, and the last at least 2 unless the rational is an integer, the one term `[x]`. That is the canonical form, one for each rational. `from_continued_fraction` takes any list of terms back to the rational, the canonical form or not (`[1; 1]` is 2), in the byte order of `a0`; it panics on an empty list, or on a term after the first that is zero or negative. `convergents` gives every `pk/qk = [a0; …, ak]`, the last one the rational itself.
- `limit_denominator(D)` is the closest rational with a denominator at most `D`, `D ≥ 1` (it panics on `D ≤ 0`): of two equally close ones the one of the smaller denominator, and at a half between two integers with `D = 1` the even one, as `round_ties_even`. It is the last convergent `pk/qk` with `qk ≤ D` or the semiconvergent `(pk-1 + j·pk)/(qk-1 + j·qk)` of the largest `j` that keeps the denominator at most `D`, whichever is closer, as Python's `Fraction.limit_denominator` and malachite's `approximate` choose; the rational itself when its denominator is at most `D`.
- `simplest_between(a, b)` is the rational of the smallest denominator in the closed interval between `a` and `b`, in either order, and of those the one of the smallest magnitude, in the byte order of `a`. Only integers can tie on the denominator, so it is one rational: 0 when the interval holds 0, the integer nearest 0 when it holds an integer, and the point of the Stern–Brocot tree nearest its root otherwise. (FLINT's `fmpq_simplest_between` gives the lower end's floor for an interval about 0, −1 for [−5/4, 4/5].)
- `reconstruct_with_bounds(r, m, n, d)` is rational reconstruction: the `p/q` in lowest terms with `p ≡ r·q (mod m)`, `|p| ≤ n`, `0 < q ≤ d` and `gcd(q, m) = 1`, in the byte order of `r`, or `None`. When `2·n·d < m` there is at most one, and `None` tells that there is none. When `2·n·d ≥ m` there may be several: a rational returned still has every property, but `None` does not prove that there is none. `reconstruct(r, m)` takes `n = d = ⌊√((m − 1)/2)⌋`, which keeps `2·n·d < m`. A modulus `m ≤ 0` panics; `n < 0` or `d < 1` allow no rational and give `None`.

```rust
use a::math::numbers::Number;
use a::math::rational::Rational;

let r = Rational::from(415) / 93u32;
let terms: Vec<String> = r.continued_fraction().iter().map(|t| t.to_string()).collect();
assert_eq!(terms, ["4", "2", "6", "7"]);
assert_eq!(Rational::from_continued_fraction(&r.continued_fraction()), r);
let c: Vec<String> = r.convergents().iter().map(|c| c.to_string()).collect();
assert_eq!(c, ["4", "9/2", "58/13", "415/93"]);

let pi = Rational::new(
    "314159265358979323846".parse().unwrap(),
    "100000000000000000000".parse().unwrap(),
);
assert_eq!(pi.limit_denominator(&Number::from(1000)).to_string(), "355/113");

let lo = Rational::from(314159) / 100000u32;
let hi = Rational::from(31416) / 10000u32;
assert_eq!(Rational::simplest_between(&lo, &hi).to_string(), "355/113");

// -3/7 modulo 10007 is 8577: 7 · 8577 = -3 (mod 10007).
let m = Number::from(10007);
let x = Rational::reconstruct(&Number::from(8577), &m).unwrap();
assert_eq!(x.to_string(), "-3/7");
```

The method:

- The terms of a continued fraction are the quotients of Euclid's algorithm on the denominator and the remainder of the numerator. When the parts fit in 128 bits it runs in registers. Longer parts take the greatest common divisor's own reductions, which find many of Euclid's steps at once and apply their matrix to the whole numbers: hgcd2's matrices from the top two limbs (Lehmer's method), and from 600 limbs Möller's half gcd on the top two thirds of the numbers. These reductions now tell of the steps they take, to a sink that the greatest common divisors leave empty, which costs them nothing (measured below); a step that they do not take in full is followed by the rest of its quotient on the same number, so the quotients of consecutive steps on one number add up to Euclid's. A continued fraction so costs what a greatest common divisor costs, `O(M(n)·log n)` for parts of `n` limbs, and the writing of its terms, about 0.58 a bit of the denominator for random numbers: each term is a `Number` with a vector of its own, and those allocations are about half the time from 16 to 256 limbs.
- `from_continued_fraction` multiplies the matrices `(ai, 1; 1, 0)`, whose product is `(pn, pn-1; qn, qn-1)`: runs of terms in registers while the entries fit in 64 bits, and the runs' matrices by a balanced tree of products, `O(M(n)·log n)` for a result of `n` limbs. `convergents` takes the recurrence `pk = ak·pk-1 + pk-2` from the terms, in registers for parts of 64 bits, where the terms are not written at all.
- `limit_denominator` and the reconstructions stop Euclid's algorithm at a bound: on the cofactors, which are the convergents' denominators (`limit_denominator`, on the denominator and the numerator's remainder), or on the remainders (the reconstruction, on `m` and `r mod m`, the first remainder at most `n`). They keep one row of the reduction matrix, the cofactors of the second number, and the bound gives a length to which the numbers can be reduced without passing it, as `|t_i|·r_(i-1) ≤ A` ties the cofactors to the remainders: the half gcd takes them there, on top parts sized so that it never goes below that length, and the row is multiplied by each of its matrices. The last steps are hgcd2's matrices tried against the bound, and when one passes it, its steps one at a time from where it started. Then `limit_denominator` chooses the convergent or the semiconvergent by one comparison of the remainders, `(2j·Q_S + Q_L)·S ≤ Q_S·L`, and the reconstruction checks `|t| ≤ d` and `gcd(r_i, t) = 1`, Wang's conditions. Both are `O(M(n)·log n)`.
- `simplest_between` goes down the Stern–Brocot tree from both ends at once, without ordering them first: the floors by one division each, an integer between them being the answer; then the terms of the fractional parts in step, until they part. There the end whose term is the smaller, or whose expansion ends, is the lower one, and the rational takes the common terms and that end's term, plus one unless the expansion ends there. The common terms are found by the reductions of the longer end, without the last step when it is not taken in full: a reduction is kept when the other end's pair stays positive under its matrix with its larger number in the same place, which is when its quotients are the same. From 1400 limbs the half gcd on the top half of the longer end goes first, on shorter tops where the ends part within it (twice the last top kept, then halved); then hgcd2's matrices, and in the one where the ends part its steps one at a time. The result comes out of the matrix of the common steps. Past 600 limbs of common terms left to Lehmer's method, both expansions are computed in full and compared, `O(M(n)·log n)`.

## Bernoulli and harmonic numbers

`Rational::bernoulli(n, order)` is the Bernoulli number `B_n`, `Rational::bernoulli_numbers(n, order)` the vector `B_0, B_1, ..., B_n`, and `Rational::harmonic(n, order)` the harmonic number `H_n = 1 + 1/2 + ... + 1/n`.

- `B_0 = 1`, `B_1 = −1/2`, `B_2 = 1/6`, `B_4 = −1/30`, and `B_n = 0` for every odd `n > 1`: the numbers of `x / (e^x − 1) = Σ B_n x^n / n!`, the convention of PARI, FLINT and Mathematica. The other convention, `B_1 = +1/2` (of `x / (1 − e^−x)`), differs only there: it is `-Rational::bernoulli(1, o)`, or `(−1)^n B_n` for any `n`. `H_0 = 0`.
- Each result is in lowest terms, both parts in the byte order `order`, in the shortest form. The indices are `u64`s; nothing panics but when a result or its working memory cannot be allocated: `B_n` has about `n log2(n / (2πe))` bits (1.25 M for `n = 10^5`, 15.8 M for `10^6`), `B_0, ..., B_n` together about `n² log2(n / 17) / 4` bits (27 MB for `n = 10^4`, 3.7 GB for `10^5`), and `H_n` about `1.44 n` in each part.
- Taking the vector costs far less than taking the numbers one by one: up to `10^4`, 0.39 s against 5.7 s.

The method, for one Bernoulli number:

- Up to `B_100` from tables: to `B_58` numerators in `i128`s (`B_60` is the first above `2^127`), then the 75 limbs of the numerators from `B_60` to `B_100` one after another, 789 bytes with their ends and denominators.
- Beyond, `|B_n| = 2 n! ζ(n) / (2π)^n`, as PARI's `bernfrac` and FLINT's `bernoulli_fmpq_ui` compute it. The denominator `d` is von Staudt and Clausen's: the product of the primes `p` with `p − 1` dividing `n`, from the divisors of `n`. With `K = 2 d n!` exactly and `U` a bound of the bits of the numerator `|B_n| d`, the quotient `K ζ(n) / (2π)^n` is computed to `w = U + 8` bits relatively and rounded: `π` from the 16,384 bits of the partition numbers' table or their Chudnovsky series, `(2π)^n` by squares and products cut to `w + log2 n + 6` bits (an early error is raised to the rest of the power), and `1/ζ(n) = Π_p (1 − p^−n)` in fixed point over the primes with `p^n < 2^w`, each factor `z ← z − z / p^n` with `z` and `p^n` cut to the bits `w − n log2 p` that it moves the product by and `log2 n + 8` more, `p^n` exact while it is short. The bounds of the errors add up to less than `2^−w` relatively, below `1/2^8` of a unit. The products, squares and quotients reuse a few vectors for the whole computation, the quotients by divisors of up to 8 limbs by Knuth's algorithm D in them (each step a division of two limbs by one by the reciprocal of the divisor's top limb), longer ones on numbers: a short number allocates a handful of times rather than at every step, which took `B_300` from 22,800 cycles to 17,000 and `B_1000` from 166,000 to 152,000.
- From `n = 8000`, as FLINT's `_bernoulli_fmpq_ui_multi_mod` does, a share of the numerator's bits, from 10 % at 8000 to 22 % from `10^5` (`α = 0.1 + 0.05 log2(n / 10^4)`), is taken modulo primes instead: the residues of `|B_n| d` modulo the primes from 5 up, by Harvey's method ("A multimodular algorithm for computing Bernoulli numbers", 2010), joined by the Chinese remainder theorem into `R` modulo their product `M`; the formula is then needed only within `M / 2^11`, `w = U − log2 M + 12` bits, and the result is `R` plus the multiple of `M` nearest to the formula's value less `R`. The Euler product's primes are those below `2^(w/n)`, so a fifth fewer bits leaves the formula about a third of its time: `B_(10^5)` took 362 ms by the formula alone, 157 ms with 20 % of its bits modulo primes.
- Modulo a prime `p ≥ 5` with `2^n ≠ 1`, Voronoi's congruence with `c = 2` is `(2^n − 1) B_n = n 2^(n−1) Σ_(j=1)^(p−1) j^(n−1) floor(2j / p)` modulo `p`, and `floor(2j / p)` is the first bit of the binary expansion of `j / p`. Taken with `p − j`, whose bit is the complement and whose power the opposite, the sum is `Σ_(j ∈ H) j^(n−1) (2 bit(j) − 1)` over half the units. Doubling `x` shifts the expansion of `x / p` by a bit and multiplies the power by `c = 2^(n−1)`, so `H` is taken as cosets `x_t ⟨2⟩` of the order `m` of 2 (half of each when `m` is even, `−1` then being `2^(m/2)`), `x_t = g^t` for a generator `g` of the cosets: per coset, the expansion of `x_t / p` 64 bits a step, `floor(y 2^64 / p)` by Granlund and Montgomery's product with `ceil(2^128 / p)` (exact, as `y 2^64 < 2^96`), and the bits weighted by tables of the sums of `c^j` over the bits of each chunk of a step — sixteen tables of 16 below `2^18`, eight of 256 above — times the powers `c^(64β)` of the steps; the chains of remainders in four lanes at once. About 0.11 cycles a unit of `H` for the long primes, and 0.5 to 2 µs of preparation for each (the order of 2, the generator, the tables).

For the vector:

- Up to `B_100` from the tables; up to `n = 150`, from the tangent numbers by Brent and Harvey's recurrence ("Fast computation of Bernoulli, Tangent and Secant numbers", 2011): `T_k = (k − 1)!`, then `T_j ← (j − k) T_(j−1) + (j − k + 2) T_j` for each `k` and `j ≥ k`, `m²/2` linear steps in place for `m = n / 2`; then `|B_2k| = 2k T_k / (4^k (4^k − 1))`, an exact division for `|B_2k| d_2k`.
- Beyond, the formula from the top index down, as FLINT's `bernoulli_fmpq_vec_no_cache` does: `G_k = 2 k! / (2π)^k` from `G_(k+2)` by a product with `(2π)²` (cut to the limbs that count) and a division by the limb `(k + 2)(k + 1)`; `ζ(k) = (1 − 2^−k)^−1 Σ_(j odd) j^−k` from the powers `j^−k` in fixed point of whole limbs, each from `j^−(k+2)` by a product with the limb `j²` and a cut of the limbs no longer needed, in one pass that also adds it into the sum. The precisions are those of `|B_k| d_k` with `log2 n + 8` bits more, made to fall with `k` by taking the largest denominator up to each index; the bounds of the errors of `G` and of every power are kept in units of their last bits, and a number whose bound would not leave less than 1/4 would be computed alone, which the bounds do not let happen. Two products of the precision for each index, and linear passes over the powers.

For a harmonic number:

- Up to `H_88`, the last whose parts are below `2^128`, from a table made at compile time by Knuth's sums; up to `H_124`, `H_88` plus the remaining terms, summed in a `u128` while they fit and added as rationals, which costs less than the splitting's preparation for so few terms (`H_89` in 509 cycles against 2487, `H_120` 2484 against 2968).
- Beyond, the terms are gathered by their odd parts, as FLINT's `fmpq_harmonic_ui` does: with `T = floor(log2 n)` and `t_m = floor(log2(n / m))`, `H_n = 2^−T Σ_(m odd) c_m / m`, `c_m = 2^(T+1) − 2^(T−t_m)`, a sum over half the terms whose denominator `Q`, the product of the odd numbers up to `n`, has half the bits of `n!`. The sum `P / Q` is taken by binary splitting: leaves of as many terms as fit in a word and a `u128`, added in place into pairs of up to 16 limbs, `P ← P q + p Q` and `Q ← Q q`, then the pairs merged two of the same rank at a time.
- The fraction is reduced without the greatest common divisor of `P` and `Q`, which costs more than the sum for long ones. Its denominator divides `2^T L`, `L` the least common multiple of the odd numbers up to `n`, and `A = H_n 2^T L` is an integer of `b ≤ bits(L) + T + 7` bits, so `A = P L Q^−1` modulo `2^b` (`Q` is odd): an inverse by Newton's method and two products of `b` bits, and the top levels of the tree, whose parts pass `b` bits, also taken modulo `2^b`. For an odd prime `p` with `p^e ≤ n < p^(e+1)`, the terms of `H_n` with `p^e` in their denominators are `1/(j p^e)` for `j ≤ k = floor(n / p^e) < p`, so `p^e` divides the denominator unless `p` divides the numerator of `H_k`; those primes, rare (`p = 5` for `n` from 500 to 624, as `H_4 = 25/12`), are found from the table or by `H_k` modulo `p`, and only their powers are taken out of `A` and `L`, by a greatest common divisor with their product.

Examples:

```rust
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;

let o = ByteOrder::LittleEndian;
assert_eq!(Rational::bernoulli(12, o).to_string(), "-691/2730");
assert_eq!(Rational::bernoulli(1, o).to_string(), "-1/2");
// B_1 = +1/2 in the other convention.
assert_eq!((-Rational::bernoulli(1, o)).to_string(), "1/2");

// Von Staudt and Clausen: B_n + Σ_((p-1) | n) 1/p is an integer; for n = 100, p = 2, 3, 5, 11, 101.
let b = Rational::bernoulli(100, o);
assert_eq!(b.denom().to_string(), "33330");
let s = [2u32, 3, 5, 11, 101].iter().fold(b, |s, &p| s + Rational::from(1) / p);
assert!(s.is_integer());

// The recurrence Σ_(k=0)^m C(m+1, k) B_k = 0, for m = 20.
let v = Rational::bernoulli_numbers(20, o);
let r: Rational = (0..=20u64).map(|k| &v[k as usize] * Number::binomial(21, k, o)).sum();
assert!(r.is_zero());

assert_eq!(Rational::harmonic(10, o).to_string(), "7381/2520");
// Wolstenholme: p^2 divides the numerator of H_(p-1), for p = 101.
let h = Rational::harmonic(100, o);
assert!(h.numer().is_multiple_of(&Number::from(101u32 * 101)));
```

## Matrices of rationals

- `Rational` is an element of [matrices](matrix.md): `Matrix<Rational>` adds, subtracts and multiplies with `+`, `-` and `*`, by a rational on either side and divides by one with `/`, transposes, has the row and column operations, the trace and the powers, and, exactly, `det()`, `rank()`, `solve(&b)` (`None` for a singular matrix), `inverse()` (`None` likewise) and `null_space()`, the basis of the reduced row echelon form: a 1 in each column without a pivot and 0s in the others.
- `Matrix<Number>` gains `solve(&b)`, the solution of `A·X = B` as a `Matrix<Rational>`, and `inverse_rational()`, the inverse as one, beside `inverse()`, which still gives the adjugate and the determinant.
- Their results are in the byte order of the top left element of the matrix, little-endian for an empty matrix; a matrix that is not square panics in `det`, `solve` and `inverse`, and so does a right-hand side with a different number of rows.

```rust
use a::math::matrix::Matrix;
use a::math::numbers::Number;
use a::math::rational::Rational;

// The Hilbert matrix of order 4, whose inverse is of integers.
let h = Matrix::from_fn(4, 4, |i, j| Rational::from(1) / (i + j + 1) as u32);
assert_eq!(h.det().to_string(), "1/6048000");
let inv = h.inverse().unwrap();
assert_eq!(inv[(3, 3)], 2800);
assert_eq!(&h * &inv, Matrix::identity(4));

// A system of integers, solved over the rationals.
let a = Matrix::from_rows(&[[2, 1], [1, 3]]).map(|&x| Number::from(x));
let b = Matrix::from_rows(&[[1], [1]]).map(|&x| Number::from(x));
let x = a.solve(&b).unwrap();
assert_eq!(x[(0, 0)].to_string(), "2/5");

// The null space of a plane.
let p = Matrix::from_rows(&[[1, 2, 3]]).map(|&x| Rational::from(x) / 7u32);
assert_eq!(p.null_space().column(0), [-2, 1, 0].map(Rational::from));
```

The method:

- A matrix of rationals is made one of integers row by row: each row is multiplied by the least common multiple `l_i` of its denominators, of `A` and `B` together for a solution, which leaves the solutions and the null space as they were and multiplies the determinant by `∏ l_i`; `A⁻¹` is `(L·A)⁻¹·L`. No fraction is formed after that, and each element of a result is reduced once, by one greatest common divisor.
- The integers are brought to an echelon form by Bareiss's fraction-free elimination, `a_ij ← (a_ij·a_kk − a_ik·a_kj) / p` with `p` the previous pivot, which divides exactly: every element is then a minor of the matrix, so the numbers grow no longer than the determinant, and the last pivot is `D = ±det`. A solution is then a fraction-free back substitution, `y_i = (D·b_i − Σ u_ij·y_j) / u_ii` from the bottom, each division exact as `y = D·x` holds Cramer's numerators. The inverse is the solution of `A·X = I`, whose integers are also the adjugate up to the sign of the row swaps, which `Matrix<Number>::inverse()` now returns instead of computing Berkowitz's polynomial.
- The integers are held in registers while their magnitudes are below `2^63`: a step is then two products in 128 bits, a subtraction, and an exact division as a product by the inverse of the divisor's odd part modulo `2^64`, a comparison with a precomputed bound telling whether the quotient fits. Beyond, each element is a vector of limbs that the steps write over, the products go to two buffers of working memory, and the exact quotient is Hensel's, from the low end; from quotients of 150 limbs it is a product by the inverse of the divisor modulo a power of `2^64`, computed once by Newton's iteration for all the elements one pivot divides. A step then allocates nothing once the vectors have grown.
- Large systems with few right-hand sides go by Dixon's p-adic lifting (Dixon 1982), as FLINT's do: from 24 rows, or 16 when an integer of the cleared matrix passes 63 bits, with at most an eighth as many right-hand sides. `A` is factored once modulo a prime just below `2^62`, in Montgomery's form; each step solves `A·x ≡ r` modulo it and lifts `r ← (r − A·x) / p` exactly; once `p^L` passes twice the product of Hadamard's bounds on Cramer's numerators and on the determinant, the digits are joined by halves, and each element is the one fraction within the bounds congruent to the result (Wang's rational reconstruction, by Lehmer's steps in registers), as a rule for the first element alone, the others then by the common denominator. The work is `O(n³ + L·n²)` operations on words, against `n³/3` products of numbers as long as the minors.
- Large determinants, from 28 rows (22 with an integer past 63 bits), are the least common multiple `l` of the denominators of a solution of `A·x = b` for a fixed pseudorandom `b`, which divides the determinant and is most of it as a rule, times `det(A) / l` from the determinants modulo primes by the Chinese remainder theorem, until the primes' product passes twice Hadamard's bound over `l`: one prime, as a rule (FLINT's `fmpz_mat_det_modular_accelerated`).
- The product `A·B` of matrices of rationals clears the rows of `A` and the columns of `B`, multiplies the integers, in 64-bit registers when their lengths and the length of the sums allow it, in 128 bits when they allow that, and divides each element once, by `l_i·r_j`: `n³` products of integers and `n²` greatest common divisors, against `n³` products and sums of rationals, each with a gcd or two. With an inner dimension of 2, or of 4 and a denominator past 32 bits, it is the sums of products of rationals, whose register paths are faster there.

## Measured

### Arithmetic

Measured on an Apple M4 Max on 2026-11-11, with nothing else running (load average 1.6 to 1.9), one thread, in cycles per operation at the clock measured before each round (4.5 GHz): the median of 21 rounds (three passes of seven, interleaved, the contenders' order rotating), over pools of random rationals (fresh each round) whose numerators and denominators have the number of limbs given (`s`: parts below `2^16`), little-endian (big-endian within 5 %), against GMP 6.3's `mpq_t` (linked statically, into a variable with room for its result: no allocation), FLINT 3.6's `fmpq` (the same), num-rational 0.4.2 (`BigRational`) and malachite-q 0.14.0. "Common factors": the denominators share a factor of half their length (for `+` and `−`), or each numerator shares one with the other's denominator (for `·` and `/`).

`&a + &b`, a new rational each time:

| Limbs | ours | GMP | FLINT | malachite | num-rational |
| --- | ---: | ---: | ---: | ---: | ---: |
| s | 145 | 108 | 38 | 105 | 1,741 |
| 1 | 341 | 327 | 440 | 490 | 7,950 |
| 2 | 667 | 730 | 819 | 1,746 | 16,690 |
| 4 | 2,552 | 3,010 | 3,177 | 3,034 | 34,677 |
| 16 | 12,028 | 15,880 | 15,664 | 14,878 | 178,990 |
| 64 | 62,188 | 81,979 | 80,359 | 79,229 | 1.62 M |
| 1024 | 2.98 M | 4.15 M | 4.19 M | 4.4 M | 348 M |

`&a * &b`:

| Limbs | ours | GMP | FLINT | malachite | num-rational |
| --- | ---: | ---: | ---: | ---: | ---: |
| s | 128 | 145 | 42 | 116 | 2,023 |
| 1 | 502 | 514 | 700 | 668 | 9,891 |
| 2 | 1,125 | 1,223 | 1,396 | 3,113 | 20,031 |
| 4 | 4,939 | 6,141 | 6,300 | 6,262 | 42,733 |
| 16 | 22,877 | 30,591 | 30,804 | 28,396 | 223,269 |
| 64 | 113,370 | 151,263 | 151,188 | 138,243 | 1.92 M |
| 1024 | 5.44 M | 7.14 M | 7.18 M | 7.89 M | 409 M |

`a += &b` and `a *= &b` on a fresh copy of `a` (GMP and FLINT: `mpq_add(a, a, b)` on a fresh copy), random parts and common factors:

| Limbs | `+=` ours | GMP | FLINT | common: ours | GMP | `*=` ours | GMP | common: ours | GMP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s | 120 | 250 | 38 | 100 | 280 | 112 | 300 | 64 | 297 |
| 1 | 350 | 477 | 448 | 405 | 572 | 527 | 640 | 226 | 471 |
| 2 | 679 | 870 | 840 | 879 | 1,037 | 1,153 | 1,333 | 710 | 894 |
| 8 | 5,760 | 7,387 | 7,219 | 6,759 | 8,330 | 10,928 | 14,264 | 6,988 | 8,796 |
| 64 | 63,160 | 82,901 | 82,486 | 65,716 | 87,016 | 112,850 | 152,786 | 62,621 | 84,866 |
| 1024 | 3.01 M | 4.2 M | 4.23 M | 3.54 M | 4.67 M | 5.49 M | 7.24 M | 4.29 M | 5.01 M |

Comparisons (`cmp`), random signs, and positive with all parts of the same length:

| Limbs | random: ours | GMP | FLINT | malachite | num-rational | same length: ours | GMP | FLINT | malachite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s | 19 | 14 | 17 | 23 | 378 | 17 | 46 | 17 | 52 |
| 2 | 24 | 43 | 97 | 73 | 670 | 35 | 56 | 140 | 120 |
| 8 | 29 | 135 | 147 | 116 | 753 | 47 | 240 | 244 | 215 |
| 64 | 23 | 3,401 | 3,099 | 2,468 | 2,006 | 36 | 7,287 | 6,572 | 4,385 |
| 1024 | 39 | 200,398 | 208,949 | 98,796 | 16,099 | 53 | 524,246 | 550,554 | 202,891 |

`Rational::new` (GMP and FLINT: `mpq_canonicalize` of a fresh copy), `floor` (GMP and FLINT: `mpz_fdiv_q` into a variable) and `round` (FLINT: `fmpz_ndiv_qr`; GMP has none), the numerator of the given limbs and the denominator of half as many for the rounding:

| Limbs | `new`: ours | GMP | FLINT | malachite | `floor`: ours | GMP | FLINT | malachite | `round`: ours | FLINT | malachite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s | 58 | 44 | 44 | 58 | 61 | 56 | 9 | 22 | 60 | 18 | 23 |
| 1 | 268 | 240 | 326 | 312 | 74 | 55 | 59 | 25 | 78 | 135 | 26 |
| 2 | 569 | 583 | 661 | 1,490 | 118 | 53 | 62 | 88 | 122 | 144 | 90 |
| 8 | 5,205 | 6,831 | 6,866 | 6,073 | 258 | 204 | 223 | 311 | 364 | 274 | 371 |
| 64 | 51,889 | 71,172 | 70,883 | 61,367 | 1,771 | 2,735 | 2,741 | 4,078 | 2,579 | 2,803 | 5,235 |
| 1024 | 2.59 M | 3.39 M | 3.35 M | 3.75 M | 149,036 | 220,520 | 220,826 | 283,710 | 184,935 | 220,691 | 315,500 |

Sums of many terms, in cycles a term: the harmonic numbers `H_n` and sums of `n` random fractions of parts below `2^16` and `2^32` of random signs, term by term by `+=` (GMP and FLINT: `mpq_add(h, h, t)`), and by `Iterator::sum`, against GMP's and FLINT's additions in the same binary-counter tree (malachite's `sum` is a tree as well; num-rational's, which is term by term, takes minutes from 10,000 terms and is left out); and products of `n` random fractions of parts below `2^16` the same two ways:

| Sum | `+=`: ours | GMP | FLINT | malachite | tree: ours | GMP | FLINT | malachite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `H_1000` | 586 | 486 | 491 | 570 | 360 | 533 | 342 | 530 |
| `H_10000` | 1,929 | 1,986 | 2,370 | 3,277 | 630 | 882 | 690 | 908 |
| `H_100000` | 14,131 | 16,923 | 16,782 | 31,285 | 1,259 | 1,692 | 1,502 | 1,771 |
| 10,000 terms of 16 bits | 1,835 | 1,679 | 2,210 | 3,026 | 451 | 654 | 437 | 631 |
| 10,000 terms of 32 bits | 8,182 | 8,188 | 8,907 | 16,917 | 1,302 | 1,686 | 1,493 | 1,833 |
| product of 10,000 of 16 bits | 2,128 | 1,864 | 2,789 | 3,209 | 579 | 805 | 555 | 727 |

- From 4 limbs up, `+`, `−`, `·` and `/` take 0.72 to 0.97 of GMP's time, 0.71 to 0.91 of FLINT's and 0.65 to 0.91 of malachite's: the gcds that dominate them are Lehmer's steps and the half gcd of [gcd](gcd.md), which take 0.67 to 0.86 of GMP's. Against the same arithmetic on numbers (the core's), the limbs save 1 to 10 % from 4 to 16 limbs (most with common factors, which take more steps) and nothing measurable beyond.
- Below that the time goes to the two allocations of a new rational's parts, about 85 cycles, which GMP's and FLINT's variables do not need, and FLINT and malachite keep parts below `2^62` and `2^64` in the word itself: parts below `2^16` take 145 cycles for `&a + &b` against GMP's 108, FLINT's 38 and malachite's 105, and one limb 341 against GMP's 327. Two limbs, with the binary gcd of two pairs of limbs in registers (as GMP's `gcd_22`), take 0.90 to 0.92 of GMP's time, but 1.06 for `+` and `−` with common factors. In place, where the vectors are there, two limbs take 0.78 to 0.87 of GMP's time, one 0.48 to 0.82 and short parts 0.22 to 0.48 (GMP reallocates its fresh copy as well), FLINT's words staying two to three times faster at short parts.
- A comparison takes 17 to 53 cycles at every length, 0.2 to 0.6 of GMP's time at 2 to 8 limbs and a five- to ten-thousandth at 1024; at short parts, 19 cycles against GMP's 14, the reads of the four parts' bytes.
- `floor` of a few limbs (2 to 8) takes 1.2 to 2.2 times GMP's or FLINT's time, and `round` at 4 and 8 limbs 1.3 to 1.5 times FLINT's, which divide into a variable with room for the quotient: the quotient's allocation is half of ours there. From 64 limbs both take 0.61 to 0.92.
- Sums of many terms by `+=` take 0.83 to 1.2 times GMP's time: each term is a pass of a remainder, an exact division and two products by one limb over the sum, as GMP's; at `H_1000` (23 limbs) the passes' fixed costs (the powers of `B` modulo the limb, 100 cycles, and the inverse) leave us at 1.20. By `Sum` the tree takes 0.68 to 0.79 of GMP's time in the same tree, and 0.84 to 1.11 of FLINT's, whose words make the many short additions at the leaves cheaper.

### Conversions

On an Apple M4 Max on 2026-11-11, with nothing else running (load average 1.2 to 1.5), one thread, `env -u MallocNanoZone`, fat LTO; the median of 21 rounds (three passes of seven) with fresh inputs and rotated order: ns or µs an operation, and in brackets the reference's time over ours (above 1: we are faster). GMP 6.3 (mpq_t), FLINT 3.6 (fmpq), malachite-q 0.14.0, num-rational 0.4.2.

`to_f64`, parts of the bits given (numerator/denominator):

| Size | a | gmp | flint | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: |
| 53b | 3.7 ns | 20.7 ns (5.63×) | 42.3 ns (11.47×) | 44.9 ns (12.17×) | 6.7 ns (1.81×) |
| 64b | 11.2 ns | 18.8 ns (1.68×) | 36.8 ns (3.28×) | 43.8 ns (3.91×) | 48.4 ns (4.32×) |
| 128b | 12.4 ns | 26.5 ns (2.14×) | 33.8 ns (2.73×) | 83.5 ns (6.75×) | 143.1 ns (11.57×) |
| 512b | 13.1 ns | 33.2 ns (2.54×) | 39.1 ns (2.99×) | 115.1 ns (8.80×) | 164.8 ns (12.61×) |
| 4096b | 13.1 ns | 51.9 ns (3.95×) | 55.8 ns (4.25×) | 201.5 ns (15.36×) | 405.6 ns (30.92×) |
| 65536b | 13.4 ns | 161.9 ns (12.11×) | 171.2 ns (12.80×) | 1.47 µs (110.05×) | 3.28 µs (245.16×) |
| 65536/64b | 3.7 ns | 11.0 ns (2.95×) | 22.3 ns (5.98×) | 5.8 ns (1.55×) | 256.4 ns (68.75×) |
| 64/65536b | 3.5 ns | 117.2 ns (33.71×) | 124.0 ns (35.67×) | 6.7 ns (1.93×) | 247.3 ns (71.12×) |

`to_f32`:

| Size | a | malachite |
| --- | ---: | ---: |
| 64b | 11.2 ns | 43.8 ns (3.91×) |
| 4096b | 12.5 ns | 190.1 ns (15.19×) |

`Rational::try_from(f64)`: `mid` floats in [1e-10, 1e10], `any` of any exponent:

| Size | a | gmp | gmp-alloc | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: |
| mid | 16.8 ns | 8.6 ns (0.51×) | 39.5 ns (2.35×) | 9.4 ns (0.56×) | 282.2 ns (16.78×) |
| any | 25.6 ns | 7.2 ns (0.28×) | 42.4 ns (1.65×) | 16.9 ns (0.66×) | 205.2 ns (8.00×) |

`FromStr` of a fraction in lowest terms, digits of both parts:

| Size | a | a-parts | gmp | gmp-canon | flint-canon | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 42.5 ns | 35.5 ns (0.84×) | 43.2 ns (1.02×) | 60.9 ns (1.43×) | 87.7 ns (2.06×) | 33.5 ns (0.79×) | 244.6 ns (5.75×) |
| 100 | 322.1 ns | 62.5 ns (0.19×) | 119.9 ns (0.37×) | 379.9 ns (1.18×) | 441.5 ns (1.37×) | 479.0 ns (1.49×) | 1.70 µs (5.29×) |
| 1000 | 4.79 µs | 396.8 ns (0.08×) | 1.10 µs (0.23×) | 6.81 µs (1.42×) | 6.91 µs (1.44×) | 5.69 µs (1.19×) | 22.74 µs (4.75×) |
| 10000 | 87.29 µs | 16.10 µs (0.18×) | 21.55 µs (0.25×) | 119.78 µs (1.37×) | 120.05 µs (1.38×) | 122.69 µs (1.41×) | 1.11 ms (12.75×) |
| 100000 | 2.67 ms | 419.75 µs (0.16×) | 585.49 µs (0.22×) | 3.59 ms (1.34×) | 3.59 ms (1.34×) | 4.07 ms (1.52×) | 101.68 ms (38.02×) |
| 1000000 | 53.62 ms | 7.09 ms (0.13×) | 12.85 ms (0.24×) | 84.00 ms (1.57×) | 83.75 ms (1.56×) | 85.06 ms (1.59×) | — |

`FromStr` of a decimal `I.F`, digits in all:

| Size | a | malachite |
| --- | ---: | ---: |
| 10 | 51.1 ns | 84.1 ns (1.64×) |
| 100 | 151.4 ns | 670.2 ns (4.43×) |
| 1000 | 1.08 µs | 7.11 µs (6.56×) |
| 10000 | 27.72 µs | 165.72 µs (5.98×) |
| 100000 | 574.37 µs | 4.89 ms (8.51×) |
| 1000000 | 8.78 ms | 94.71 ms (10.79×) |

`to_string` of a fraction, digits of both parts:

| Size | a | a-radix | core | gmp | flint | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 39.4 ns | 25.7 ns (0.65×) | 113.9 ns (2.89×) | 57.0 ns (1.45×) | 31.8 ns (0.81×) | 41.3 ns (1.05×) | 110.4 ns (2.80×) |
| 100 | 129.4 ns | 98.4 ns (0.76×) | 179.1 ns (1.38×) | 141.6 ns (1.09×) | 102.5 ns (0.79×) | 271.5 ns (2.10×) | 250.9 ns (1.94×) |
| 1000 | 1.01 µs | 962.9 ns (0.96×) | 1.14 µs (1.13×) | 2.41 µs (2.40×) | 1.58 µs (1.57×) | 3.68 µs (3.66×) | 5.25 µs (5.21×) |
| 10000 | 30.88 µs | 30.88 µs (1.00×) | 31.38 µs (1.02×) | 40.90 µs (1.32×) | 40.83 µs (1.32×) | 67.76 µs (2.19×) | 155.43 µs (5.03×) |
| 100000 | 834.28 µs | 838.17 µs (1.00×) | 839.28 µs (1.01×) | 1.21 ms (1.45×) | 1.26 ms (1.51×) | 1.63 ms (1.95×) | 9.65 ms (11.57×) |
| 1000000 | 14.46 ms | 14.49 ms (1.00×) | 14.52 ms (1.00×) | 28.95 ms (2.00×) | 18.42 ms (1.27×) | 29.91 ms (2.07×) | 738.04 ms (51.04×) |

GMP's `mpq_get_d` and FLINT's `fmpq_get_d` (which calls it) truncate towards zero: of 500 random fractions of 64 bits they gave another float than the nearest for 241 to 246; malachite-q's `f64::rounding_from(_, Nearest)` and num-rational's `to_f64` agreed with ours on every one, at every size. num-rational's `to_f32` is its `to_f64` rounded again, so it is not compared. GMP's `mpq_set_d` writes into an `mpq_t` that exists already; `gmp-alloc` makes a new one each time, as a new rational is. Reading text, `gmp` is `mpq_set_str` alone, which leaves the fraction as written, `gmp-canon` the same with `mpq_canonicalize`, as every `Rational` is reduced; `a-parts` reads the two parts as `Number`s without the gcd, and shows that the gcd is 81 to 92% of a fraction's reading from 100 digits on. `a-radix` is `to_str_radix(10)`, `core` the `Display` of the core before this area.

Where a reference is faster: parsing 10 digits (malachite 34 ns, ours 43 ns) and making a rational of a float (GMP into an existing `mpq_t` 7 to 9 ns, malachite 9 to 17 ns, ours 17 to 26 ns) — a `Rational` always holds two vectors, and two allocations take about 20 ns, where malachite keeps numbers of one limb in place and allocates nothing. Writing a fraction of 10 to 100 digits with `to_string`, FLINT takes 32 and 103 ns against our 39 and 129; `to_str_radix(10)` takes 26 and 98 ns: the difference is the formatting machinery of `to_string`, which copies the text once more.

### Continued fractions and approximations

Measured on an Apple M4 Max on 2026-11-11, with nothing else running (load average 1.4 to 3.3, falling as the machine settled), single-threaded, in cycles per call at the clock measured before each pass (4.5 GHz), or in time where marked, against FLINT 3.6 (`fmpq_get_cfrac` with `fmpq_cfrac_bound` and the vector's allocation, `fmpq_set_cfrac`, `fmpq_simplest_between`, `fmpq_reconstruct_fmpz` and `fmpq_reconstruct_fmpz_2`), PARI 2.19 (`contfrac`, `bestappr(x, D)`, `bestappr(Mod(r, m))`) and malachite-q 0.14.0 (`continued_fraction` collected, `from_continued_fraction_ref`, `convergents` collected, `approximate`, `simplest_rational_in_closed_interval`, `reconstruct_ref`, `reconstruct_with_bounds_ref`); num-rational 0.4 has none of these. Random parts of `n` limbs, positive; every process with `MallocNanoZone` unset, the harness with fat LTO and one codegen unit; three passes of seven rounds, each cell the median of the 21, each round on fresh inputs in an order that turns with the round. PARI and malachite, whose continued fractions and approximations are quadratic, are not measured past 4096 limbs, nor FLINT's `fmpq_simplest_between` past 16,384.

`continued_fraction`, parts of n limbs:

| n | `a` | FLINT | PARI | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,598 | 4,007 | 3,558 | 2,143 | 0.75 |
| 2 | 3,664 | 12,127 | 9,485 | 10,238 | 0.39 |
| 4 | 9,677 | 27,642 | 21,101 | 32,390 | 0.46 |
| 16 | 36,510 | 32,984 | 95,696 | 178,623 | 1.11 |
| 64 | 174,333 | 118,048 | 543,267 | 998,179 | 1.48 |
| 256 | 807,029 | 583,222 | 5.14 M | 8.52 M | 1.38 |
| 1024 | 4.62 M | 5.1 M | 15.3 ms | 22.5 ms | 0.91 |
| 4096 | 26.8 M | 35.1 M | 229 ms | 327 ms | 0.76 |
| 16384 | 33 ms | 48.3 ms | – | – | 0.68 |
| 100000 | 287 ms | 441 ms | – | – | 0.65 |

`from_continued_fraction`, the terms of parts of n limbs:

| n | `a` | FLINT | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: | ---: |
| 1 | 809 | 1,290 | 2,017 | 0.63 |
| 2 | 2,342 | 3,017 | 3,839 | 0.78 |
| 4 | 5,192 | 6,450 | 7,763 | 0.80 |
| 16 | 21,591 | 27,736 | 39,743 | 0.78 |
| 64 | 102,121 | 129,575 | 350,868 | 0.79 |
| 256 | 552,683 | 665,692 | 5.36 M | 0.83 |
| 1024 | 3.24 M | 3.97 M | 19.1 ms | 0.82 |
| 4096 | 17.5 M | 25.1 M | 306 ms | 0.70 |
| 16384 | 20.2 ms | 31 ms | – | 0.65 |
| 100000 | 168 ms | 238 ms | – | 0.71 |

`convergents`, parts of n limbs:

| n | `a` | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: |
| 1 | 3,707 | 4,754 | 0.78 |
| 2 | 11,946 | 17,006 | 0.70 |
| 4 | 42,093 | 49,177 | 0.86 |
| 16 | 260,733 | 272,162 | 0.96 |
| 64 | 1.8 M | 1.74 M | 1.03 |
| 256 | 14.4 M | 17.2 M | 0.84 |

`limit_denominator`, parts of n limbs, a bound of n/2:

| n | `a` | PARI | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: | ---: |
| 1 | 439 | 4,129 | 2,798 | 0.16 |
| 2 | 1,453 | 10,419 | 11,845 | 0.14 |
| 4 | 4,619 | 20,110 | 32,152 | 0.23 |
| 16 | 11,808 | 102,715 | 155,282 | 0.11 |
| 64 | 49,784 | 654,940 | 886,584 | 0.08 |
| 256 | 326,559 | 7.34 M | 8.53 M | 0.04 |
| 1024 | 2.57 M | 28.7 ms | 23.2 ms | 0.02 |
| 4096 | 16.4 M | 362 ms | 359 ms | 0.01 |
| 16384 | 19.8 ms | – | – | – |
| 100000 | 177 ms | – | – | – |

`limit_denominator`, parts of n limbs, a bound of 60 bits:

| n | `a` | PARI | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: | ---: |
| 1 | 610 | 7,274 | 4,257 | 0.14 |
| 2 | 1,397 | 9,852 | 10,827 | 0.14 |
| 4 | 3,544 | 9,527 | 13,997 | 0.37 |
| 16 | 4,292 | 10,943 | 16,364 | 0.39 |
| 64 | 6,090 | 18,614 | 27,533 | 0.33 |
| 256 | 12,190 | 60,645 | 68,337 | 0.20 |
| 1024 | 35,093 | 197,372 | 305,554 | 0.18 |
| 4096 | 122,910 | 705,740 | 781,716 | 0.17 |
| 16384 | 434,193 | 2.69 M | 3.11 M | 0.16 |
| 100000 | 2.64 M | 16.5 M | 18.5 M | 0.16 |

`simplest_between`, two random ends of n limbs:

| n | `a` | FLINT | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: | ---: |
| 1 | 293 | 808 | 797 | 0.37 |
| 2 | 374 | 1,058 | 1,719 | 0.35 |
| 4 | 1,313 | 1,040 | 2,003 | 1.26 |
| 16 | 1,680 | 2,513 | 6,251 | 0.67 |
| 64 | 2,498 | 10,262 | 19,557 | 0.24 |
| 256 | 4,996 | 67,511 | 105,266 | 0.07 |
| 1024 | 17,043 | 601,045 | 300,337 | 0.06 |
| 4096 | 24,849 | 2.45 M | 52,955 | 0.47 |
| 16384 | 247,481 | 11.1 M | 2.04 M | 0.12 |
| 100000 | 1.61 M | 15.2 ms | 12.4 M | 0.13 |

`simplest_between`, x and x + 1/q, q of 1.5 n limbs:

| n | `a` | FLINT | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5,451 | 6,592 | 11,091 | 0.83 |
| 2 | 6,873 | 20,382 | 29,124 | 0.34 |
| 4 | 8,689 | 8,239 | 14,449 | 1.05 |
| 16 | 26,474 | 59,288 | 116,153 | 0.45 |
| 64 | 133,850 | 220,476 | 405,602 | 0.61 |
| 256 | 1.23 M | 1.64 M | 3.37 M | 0.75 |
| 1024 | 13.2 M | 16 M | 32.8 M | 0.82 |
| 4096 | 17.9 ms | 40.9 ms | 82.9 ms | 0.44 |
| 16384 | 101 ms | 522 ms | – | 0.19 |
| 100000 | 891 ms | – | – | – |

`reconstruct`, a modulus of n limbs, p/q of up to n/2:

| n | `a` | FLINT | PARI | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 274 | 488 | 2,358 | 218 | 1.26 |
| 2 | 1,396 | 1,103 | 4,889 | 1,179 | 1.27 |
| 4 | 2,339 | 1,833 | 4,063 | 3,326 | 1.28 |
| 16 | 7,018 | 8,023 | 10,412 | 10,727 | 0.87 |
| 64 | 28,567 | 35,547 | 44,245 | 46,997 | 0.80 |
| 256 | 174,371 | 241,371 | 284,038 | 369,859 | 0.72 |
| 1024 | 1.22 M | 2.29 M | 1.76 M | 3.17 M | 0.69 |
| 4096 | 7.23 M | 16.3 M | 11.2 M | 18.3 M | 0.64 |
| 16384 | 38.8 M | 19.5 ms | 15.4 ms | 20.9 ms | 0.56 |
| 100000 | 107 ms | 201 ms | 195 ms | 223 ms | 0.55 |

`reconstruct_with_bounds`, a modulus of n limbs, bounds of 0.15 and 0.8 of it:

| n | `a` | FLINT | malachite | `a` over the fastest |
| --- | ---: | ---: | ---: | ---: |
| 1 | 352 | 508 | 292 | 1.21 |
| 2 | 1,356 | 1,317 | 1,369 | 1.03 |
| 4 | 2,305 | 2,211 | 3,540 | 1.04 |
| 16 | 6,634 | 8,876 | 12,291 | 0.75 |
| 64 | 30,904 | 40,959 | 61,068 | 0.75 |
| 256 | 200,407 | 299,043 | 487,644 | 0.67 |
| 1024 | 1.58 M | 3.01 M | 4.31 M | 0.53 |
| 4096 | 10.8 M | 19.7 M | 25.2 M | 0.55 |
| 16384 | 14.1 ms | 28.5 ms | 31.8 ms | 0.49 |
| 100000 | 99.9 ms | 189 ms | 207 ms | 0.53 |

Against the fastest reference that has the function: continued fractions 0.39 to 0.75 of FLINT's or PARI's time up to 4 limbs and 0.65 to 0.91 from 1024 limbs, but 1.11 to 1.48 times FLINT's from 16 to 256 limbs, where half of the time is the allocation of a vector for each term (FLINT keeps small terms in its `fmpz` words); `from_continued_fraction` 0.63 to 0.83 of FLINT's; `convergents` 0.70 to 0.96 of malachite's but 1.03 at 64 limbs; `limit_denominator` 0.01 to 0.39 of PARI's or malachite's, quadratic both; `simplest_between` 0.06 to 0.67 on random ends but 1.26 times FLINT's at 4 limbs, and on close ends 0.19 to 0.83 but for 4 limbs (1.05); the reconstructions 0.49 to 0.87 from 16 limbs, and 1.03 to 1.28 times malachite's or FLINT's up to 4 limbs, malachite's words needing no allocation for the result.

The greatest common divisors with the sink threaded through the half gcd, against the frozen core, in one process (the ratio of the medians, above 1 slower), random numbers of the given limbs:

| limbs | gcd | gcd_ext |
| --- | ---: | ---: |
| 3 | 0.979 | 0.987 |
| 16 | 0.951 | 1.028 |
| 100 | 1.003 | 1.005 |
| 1000 | 0.994 | 0.996 |
| 3000 | 1.003 | 0.998 |

### Bernoulli and harmonic numbers

Measured on an Apple M4 Max on 2026-11-11, with nothing else running (load average 1.5 to 2.2), single-threaded, every process with `MallocNanoZone` unset, the harness built with fat LTO and one codegen unit, in cycles per call (milliseconds from `10^5` cycles), the cycles from the clock measured just before each round (4.3 to 4.5 GHz): two passes of interleaved rounds, the contenders' order rotating each round, each cell timed in batches of calls of at least 30 ms over four indices in turn (`n` to `n + 6` by 2 for the Bernoulli numbers, `n` to `n + 3` for the harmonic numbers, `n` alone for the vectors), the median of 9 rounds a pass (3 for `B_(10^5)`, `H_(10^6)` and the vector to `10^4`, 1 for `B_(10^6)`), every value checked against FLINT's first; against FLINT 3.6 and PARI 2.19 called from the same binary, which write into results they keep, and malachite-q 0.14 (harmonic numbers; it has no Bernoulli numbers, and num-rational 0.4 and GMP neither). In brackets, the reference's time over this crate's.

`Rational::bernoulli(n)` against FLINT's `_bernoulli_fmpq_ui` (`bernoulli_fmpq_ui` with its cache), PARI's `bernfrac` (with and without its cache) (cycles, or ms; in brackets the reference's time over this crate's):

| n | `a` | FLINT | FLINT, cached | FLINT ζ only | PARI | PARI, no cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 75 | 18 (0.24) | 20 (0.27) |  | 10 (0.13) | 0.125 ms (7408.17) |
| 60 | 91 | 3,150 (34.64) | 3,178 (34.95) |  | 10 (0.11) | 0.126 ms (6157.81) |
| 100 | 4,004 | 4,678 (1.17) | 4,643 (1.16) |  | 10 (0.00) | 0.124 ms (137.58) |
| 300 | 18,657 | 17,795 (0.95) | 17,892 (0.96) |  | 22,658 (1.21) | 0.13 ms (30.88) |
| 1,000 | 0.0371 ms | 0.035 ms (0.94) | 0.0347 ms (0.94) |  | 0.0447 ms (1.21) | 0.17 ms (4.60) |
| 3,000 | 0.342 ms | 0.365 ms (1.07) | 0.364 ms (1.07) |  | 0.466 ms (1.36) | 0.595 ms (1.74) |
| 10^4 | 3.21 ms | 4.9 ms (1.52) | 4.9 ms (1.52) |  | 6.04 ms (1.88) | 6.18 ms (1.92) |
| 10^5 | 130 ms | 256 ms (1.97) |  | 640 ms (4.92) | 731 ms (5.61) |  |
| 10^6 | 7.51 s | 12.3 s (1.64) |  |  | 78.2 s (10.41) |  |

`Rational::harmonic(n)` against FLINT's `_fmpq_harmonic_ui`, PARI's `harmonic` and malachite-q's `Rational::harmonic_number` (cycles, or ms; in brackets the reference's time over this crate's):

| n | `a` | FLINT | PARI | malachite-q |
| --- | ---: | ---: | ---: | ---: |
| 10 | 73 | 8 (0.11) | 1,491 (20.37) | 49 (0.67) |
| 89 | 697 | 2,200 (3.16) | 20,272 (29.08) | 2,872 (4.12) |
| 100 | 1,157 | 2,860 (2.47) | 22,384 (19.34) | 3,536 (3.06) |
| 300 | 6,676 | 8,520 (1.28) | 86,622 (12.98) | 10,931 (1.64) |
| 1,000 | 21,854 | 42,655 (1.95) | 0.0837 ms (17.07) | 55,652 (2.55) |
| 10^4 | 0.178 ms | 0.406 ms (2.27) | 1.91 ms (10.69) | 0.546 ms (3.06) |
| 10^5 | 4.25 ms | 13.8 ms (3.25) | 38.9 ms (9.14) | 14.8 ms (3.48) |
| 10^6 | 79.1 ms | 262 ms (3.31) | 889 ms (11.24) | 257 ms (3.26) |

`Rational::bernoulli_numbers(n)` against FLINT's `bernoulli_fmpq_vec_no_cache` and `arith_bernoulli_number_vec`, PARI's `bernvec(n/2)` (with and without its cache) (cycles, or ms; in brackets the reference's time over this crate's):

| n | `a` | FLINT | FLINT `arith` | PARI | PARI, no cache |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 842 | 300 (0.36) | 304 (0.36) | 27 (0.03) | 0.125 ms (658.61) |
| 100 | 7,822 | 51,614 (6.60) | 51,606 (6.60) | 79 (0.01) | 0.123 ms (69.87) |
| 300 | 0.0919 ms | 0.115 ms (1.26) | 0.153 ms (1.66) | 207 (0.00) | 0.164 ms (1.79) |
| 1,000 | 1.3 ms | 1.54 ms (1.18) | 6.27 ms (4.81) | 590 (0.00) | 1.75 ms (1.34) |
| 10^4 | 384 ms | 513 ms (1.34) |  |  | 551 ms (1.44) |

- `B_n` from `n = 3000` faster than FLINT's, 1.07 times at 3000, 1.5 at `10^4`, 2.0 at `10^5` and 1.6 at `10^6` (7.5 s against 12.3 s), by the residues modulo primes; FLINT's formula alone (`_bernoulli_fmpq_ui_zeta`) takes 4.9 times this crate's time at `10^5`. Up to `B_100` from the tables, 91 cycles against FLINT's 3150 at 60. At 300 and 1000, FLINT's takes 0.95 of this crate's time (17,795 cycles against 18,657 at 300), the formula's arithmetic at these lengths. PARI keeps the numbers it has computed and returns the small ones from its cache in about 10 cycles; with the cache dropped before each call, each call rebuilds it, 0.12 ms or more.
- A table lookup, `B_10` or `H_10`, takes the allocation of the two vectors of the result, 73 to 75 cycles against FLINT's 8 to 18, which writes into numbers it keeps (small ones in a word).
- `H_n` from `n = 89`, just past the table, faster than FLINT's, 3.2 times at 89 (697 cycles against 2200), 1.3 at 300, 2.0 to 2.3 from 1000 to `10^4` and 3.3 at `10^5` and `10^6`, by the reduction without a greatest common divisor and the pairs added in place; malachite-q, which sums the odd parts as FLINT does, takes 1.6 to 4.1 times this crate's time from 89 up.
- The vector from 300 faster than FLINT's, which computes it the same way, 1.2 to 1.3 times from 300 to `10^4`, by the powers' pass that updates and adds in one, in limbs, and the denominators' bound kept by index; up to 100, from the tables, 7822 cycles against FLINT's 51,614 at 100. PARI's `bernvec` copies its cache once it is filled.

### Matrices

On an Apple M4 Max on 2026-11-11, with nothing else running (load average 1.3 to 3.6), one thread, `env -u MallocNanoZone`, fat LTO, the median of 5 rounds with fresh random inputs each, interleaved in one process, in µs (or ms and s), and each reference's time over this crate's in brackets. FLINT is 3.6, its defaults (`fmpq_mat_det`, `fmpq_mat_solve`, `fmpq_mat_inv`, `fmpq_mat_mul`) and its Bareiss alone (the rows cleared as `fmpq_mat` does, then `fmpz_mat_det_bareiss` or `fmpz_mat_solve_fflu`, and one division per element); PARI is 2.19 (`det`, `RgM_solve`, `RgM_inv`, `gmul`); "`a`, Bareiss alone" is this crate with Dixon's lifting switched off; num-rational 0.4.2 and malachite-q 0.14.0 have no matrices, so they run a Gaussian elimination of fractions (the first non-zero pivot, back substitution, the inverse as the solution of `A·X = I`) and the product as sums of products. The tables for `Matrix<Number>` are on the [matrices](matrix.md#exactly) page.

`&a * &b` of `Matrix<Rational>`, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a`, sums of products of rationals | FLINT | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.29 µs | 0.33 µs (1.16) | 0.13 µs (0.44) | 0.07 µs (0.23) | 3.63 µs (12.58) | 0.24 µs (0.85) |
| 4 | 0.89 µs | 2.62 µs (2.94) | 0.39 µs (0.44) | 0.61 µs (0.69) | 28.8 µs (32.33) | 2.07 µs (2.33) |
| 8 | 3.56 µs | 21.3 µs (5.99) | 1.35 µs (0.38) | 5.62 µs (1.58) | 233.4 µs (65.50) | 18.4 µs (5.17) |
| 16 | 15.2 µs | 188.1 µs (12.38) | 5.32 µs (0.35) | 47.4 µs (3.12) | 1.85 ms (121.68) | 176.8 µs (11.63) |
| 32 | 71.5 µs | 1.74 ms (24.32) | 24.9 µs (0.35) | 380.6 µs (5.32) | 15 ms (209.72) | 1.56 ms (21.76) |
| 64 | 337.6 µs | 15.4 ms (45.50) | 131.8 µs (0.39) | 2.92 ms (8.66) | 121 ms (357.33) | 13.5 ms (39.89) |
| 128 | 1.68 ms | 122 ms (72.38) | 910.1 µs (0.54) | 6.63 ms (3.94) | 962 ms (571.50) | 103 ms (61.20) |

`&a * &b` of `Matrix<Rational>`, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99 (median, and the time over `a`'s):

| n | `a` | `a`, sums of products of rationals | FLINT | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.31 µs | 0.38 µs (1.22) | 0.21 µs (0.67) | 0.80 µs (2.59) | 5.33 µs (17.31) | 0.31 µs (1.02) |
| 4 | 1.48 µs | 3.03 µs (2.05) | 0.85 µs (0.57) | 5.01 µs (3.39) | 50.1 µs (33.85) | 3.39 µs (2.29) |
| 8 | 7.79 µs | 25.3 µs (3.24) | 5.27 µs (0.68) | 36.6 µs (4.70) | 448.4 µs (57.55) | 34.8 µs (4.47) |
| 16 | 52.4 µs | 257.1 µs (4.91) | 45.2 µs (0.86) | 210.6 µs (4.02) | 4.42 ms (84.48) | 449.7 µs (8.59) |
| 32 | 807.4 µs | 2.79 ms (3.46) | 301.8 µs (0.37) | 1.27 ms (1.57) | 44.6 ms (55.24) | 4.32 ms (5.35) |
| 64 | 6.35 ms | 25.2 ms (3.97) | 2.19 ms (0.34) | 7.75 ms (1.22) | 430 ms (67.73) | 35.9 ms (5.66) |
| 128 | 40.8 ms | 202 ms (4.96) | 15 ms (0.37) | 32.6 ms (0.80) | 4.16 s (101.87) | 296 ms (7.26) |

`&a * &b` of `Matrix<Rational>`, 64-bit parts (median, and the time over `a`'s):

| n | `a` | `a`, sums of products of rationals | FLINT | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.33 µs | 1.41 µs (1.06) | 2.25 µs (1.69) | 4.40 µs (3.31) | 38.6 µs (29.01) | 2.53 µs (1.91) |
| 4 | 16.9 µs | 19.4 µs (1.14) | 20.1 µs (1.18) | 44.0 µs (2.60) | 435.9 µs (25.73) | 28.9 µs (1.70) |
| 8 | 193.9 µs | 230.1 µs (1.19) | 186.4 µs (0.96) | 988.7 µs (5.10) | 5.64 ms (29.11) | 298.2 µs (1.54) |
| 16 | 1.92 ms | 2.77 ms (1.44) | 2.02 ms (1.05) | 36.6 ms (19.02) | 83.3 ms (43.31) | 2.92 ms (1.52) |
| 32 | 21.1 ms | 28.7 ms (1.36) | 19.4 ms (0.92) | 1.34 s (63.73) | 1.44 s (68.12) | 30.1 ms (1.43) |
| 64 | 306 ms | 289 ms (0.94) | 217 ms (0.71) | 46.5 s (152.13) | 29.9 s (97.70) | 332 ms (1.09) |

`Matrix<Rational>::det()`, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.08 µs | 0.08 µs (0.98) | 0.02 µs (0.20) | 0.10 µs (1.30) | 0.02 µs (0.24) | 1.63 µs (21.07) | 0.13 µs (1.70) |
| 4 | 0.19 µs | 0.18 µs (0.99) | 0.13 µs (0.73) | 0.26 µs (1.42) | 0.65 µs (3.50) | 13.8 µs (74.66) | 0.99 µs (5.36) |
| 8 | 0.68 µs | 0.67 µs (0.98) | 1.12 µs (1.65) | 1.11 µs (1.63) | 3.07 µs (4.50) | 170.0 µs (249.58) | 12.0 µs (17.67) |
| 16 | 7.57 µs | 7.32 µs (0.97) | 8.77 µs (1.16) | 8.78 µs (1.16) | 36.3 µs (4.80) | 2.2 ms (290.28) | 197.6 µs (26.12) |
| 32 | 110.7 µs | 160.5 µs (1.45) | 72.2 µs (0.65) | 178.7 µs (1.61) | 216.5 µs (1.96) | 34.7 ms (313.28) | 4.33 ms (39.08) |
| 64 | 483.9 µs | 2.47 ms (5.11) | 580.8 µs (1.20) | 2.87 ms (5.93) | 1.87 ms (3.86) | 583 ms (1205.32) | 70 ms (144.75) |
| 128 | 3.13 ms | 36.9 ms (11.79) | 3.65 ms (1.17) | 39.8 ms (12.71) | 19.2 ms (6.15) | 10.6 s (3390.12) | 1.06 s (338.26) |

`Matrix<Rational>::det()`, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99 (median, and the time over `a`'s):

| n | `a` | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.12 µs | 0.13 µs (1.05) | 0.03 µs (0.21) | 0.15 µs (1.18) | 0.11 µs (0.87) | 1.95 µs (15.64) | 0.15 µs (1.21) |
| 4 | 0.46 µs | 0.46 µs (1.00) | 0.36 µs (0.79) | 0.50 µs (1.08) | 4.07 µs (8.87) | 21.6 µs (47.10) | 1.42 µs (3.10) |
| 8 | 5.02 µs | 4.97 µs (0.99) | 4.29 µs (0.86) | 4.31 µs (0.86) | 44.1 µs (8.80) | 369.3 µs (73.64) | 36.6 µs (7.30) |
| 16 | 76.0 µs | 76.2 µs (1.00) | 66.6 µs (0.88) | 67.6 µs (0.89) | 433.2 µs (5.70) | 8.07 ms (106.23) | 1.03 ms (13.57) |
| 32 | 907.3 µs | 1.62 ms (1.78) | 585.4 µs (0.65) | 1.45 ms (1.60) | 3.66 ms (4.04) | 218 ms (239.86) | 25.8 ms (28.39) |
| 64 | 8.58 ms | 53.7 ms (6.25) | 7.2 ms (0.84) | 50 ms (5.82) | 31.4 ms (3.66) | 7.91 s (921.96) | 652 ms (75.97) |
| 128 | 73.1 ms | 1.86 s (25.42) | 106 ms (1.45) | 1.76 s (24.11) | 311 ms (4.26) | – | 18.3 s (250.09) |

`Matrix<Rational>::det()`, 64-bit parts (median, and the time over `a`'s):

| n | `a` | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.23 µs | 1.20 µs (0.98) | 0.37 µs (0.30) | 0.75 µs (0.61) | 0.40 µs (0.33) | 14.4 µs (11.67) | 0.60 µs (0.49) |
| 4 | 7.66 µs | 7.76 µs (1.01) | 3.85 µs (0.50) | 4.59 µs (0.60) | 245.5 µs (32.05) | 254.2 µs (33.20) | 21.4 µs (2.80) |
| 8 | 85.4 µs | 86.7 µs (1.01) | 66.5 µs (0.78) | 66.5 µs (0.78) | 4.81 ms (56.30) | 7.3 ms (85.47) | 638.0 µs (7.47) |
| 16 | 4.1 ms | 4.12 ms (1.01) | 3.87 ms (0.94) | 3.83 ms (0.93) | 172 ms (41.98) | 472 ms (115.17) | 27.3 ms (6.65) |
| 32 | 63.9 ms | 220 ms (3.44) | 39.5 ms (0.62) | 296 ms (4.63) | 9.84 s (153.97) | 48.7 s (762.72) | 1.41 s (22.09) |
| 64 | 1.23 s | 7.99 s (6.50) | 804 ms (0.65) | 19.5 s (15.89) | – | – | 86.2 s (70.08) |

`Matrix<Rational>::det()`, 256-bit parts (median, and the time over `a`'s):

| n | `a` | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4.83 µs | 4.80 µs (0.99) | 3.18 µs (0.66) | 3.76 µs (0.78) | 3.34 µs (0.69) | 67.5 µs (13.98) | 4.21 µs (0.87) |
| 4 | 31.9 µs | 32.0 µs (1.00) | 23.8 µs (0.75) | 27.4 µs (0.86) | 1.36 ms (42.73) | 1.53 ms (47.80) | 101.8 µs (3.19) |
| 8 | 753.5 µs | 751.3 µs (1.00) | 666.4 µs (0.88) | 664.9 µs (0.88) | 36.4 ms (48.30) | 80.1 ms (106.26) | 4.42 ms (5.87) |
| 16 | 32 ms | 31.8 ms (0.99) | 43.3 ms (1.35) | 43 ms (1.35) | 1.8 s (56.38) | 7.48 s (234.03) | 217 ms (6.78) |
| 32 | 871 ms | 1.11 s (1.28) | 328 ms (0.38) | 2.77 s (3.18) | 103 s (118.46) | – | 12.6 s (14.46) |

`Matrix<Rational>::solve(&b)`, one column, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.17 µs | 0.18 µs (1.02) | 0.10 µs (0.58) | 0.22 µs (1.25) | 1.35 µs (7.83) | 3.78 µs (21.86) | 0.31 µs (1.77) |
| 4 | 0.41 µs | 0.39 µs (0.97) | 0.47 µs (1.17) | 0.49 µs (1.20) | 2.34 µs (5.78) | 27.8 µs (68.62) | 2.05 µs (5.07) |
| 8 | 1.35 µs | 1.31 µs (0.97) | 1.74 µs (1.29) | 1.73 µs (1.28) | 13.6 µs (10.07) | 268.7 µs (198.43) | 20.7 µs (15.26) |
| 16 | 12.6 µs | 12.5 µs (0.99) | 30.6 µs (2.43) | 15.7 µs (1.24) | 55.0 µs (4.36) | 2.97 ms (235.20) | 310.8 µs (24.61) |
| 32 | 99.8 µs | 192.9 µs (1.93) | 137.6 µs (1.38) | 223.8 µs (2.24) | 321.8 µs (3.22) | 41.6 ms (417.01) | 5.44 ms (54.51) |
| 64 | 448.9 µs | 2.66 ms (5.94) | 615.4 µs (1.37) | 3.16 ms (7.03) | 5.14 ms (11.45) | 646 ms (1438.48) | 78.6 ms (175.18) |
| 128 | 2.49 ms | 38.2 ms (15.33) | 3.69 ms (1.48) | 42 ms (16.87) | 50.3 ms (20.21) | 11.3 s (4532.92) | 1.14 s (459.01) |

`Matrix<Rational>::solve(&b)`, one column, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99 (median, and the time over `a`'s):

| n | `a` | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.29 µs | 0.27 µs (0.94) | 0.16 µs (0.55) | 0.32 µs (1.09) | 1.85 µs (6.40) | 5.51 µs (19.04) | 0.41 µs (1.40) |
| 4 | 1.13 µs | 1.14 µs (1.00) | 1.29 µs (1.14) | 1.33 µs (1.17) | 16.3 µs (14.36) | 50.0 µs (44.11) | 3.94 µs (3.48) |
| 8 | 9.90 µs | 10.2 µs (1.03) | 9.76 µs (0.99) | 9.67 µs (0.98) | 59.9 µs (6.05) | 691.2 µs (69.81) | 72.2 µs (7.30) |
| 16 | 112.2 µs | 113.2 µs (1.01) | 142.5 µs (1.27) | 116.5 µs (1.04) | 505.9 µs (4.51) | 12.7 ms (113.36) | 1.68 ms (14.97) |
| 32 | 846.2 µs | 2.01 ms (2.37) | 1.04 ms (1.23) | 1.97 ms (2.33) | 7.4 ms (8.75) | 289 ms (341.40) | 34 ms (40.20) |
| 64 | 6.89 ms | 59.5 ms (8.64) | 8.38 ms (1.22) | 57.9 ms (8.41) | 54.2 ms (7.86) | 9.42 s (1367.18) | 770 ms (111.80) |
| 128 | 49.7 ms | 1.94 s (39.02) | 99.6 ms (2.01) | 1.88 s (37.85) | 486 ms (9.78) | – | 20.3 s (408.37) |

`Matrix<Rational>::solve(&b)`, one column, 64-bit parts (median, and the time over `a`'s):

| n | `a` | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.39 µs | 3.31 µs (0.98) | 1.88 µs (0.56) | 2.31 µs (0.68) | 22.8 µs (6.73) | 47.3 µs (13.96) | 2.43 µs (0.72) |
| 4 | 19.9 µs | 19.9 µs (1.00) | 17.0 µs (0.85) | 17.3 µs (0.87) | 157.2 µs (7.90) | 762.5 µs (38.32) | 54.2 µs (2.72) |
| 8 | 218.6 µs | 215.8 µs (0.99) | 236.3 µs (1.08) | 234.6 µs (1.07) | 1.73 ms (7.93) | 19.6 ms (89.48) | 1.56 ms (7.14) |
| 16 | 5.62 ms | 7.14 ms (1.27) | 9.55 ms (1.70) | 7.39 ms (1.32) | 21.5 ms (3.83) | 1.03 s (182.37) | 50.4 ms (8.97) |
| 32 | 81.2 ms | 277 ms (3.41) | 131 ms (1.61) | 385 ms (4.74) | 313 ms (3.85) | 79 s (973.07) | 2.17 s (26.70) |
| 64 | 1.4 s | 8.9 s (6.36) | 2.33 s (1.67) | 21.5 s (15.34) | 5.34 s (3.81) | – | 107 s (76.64) |

`Matrix<Rational>::solve(&b)`, one column, 256-bit parts (median, and the time over `a`'s):

| n | `a` | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 12.1 µs | 12.1 µs (1.00) | 10.4 µs (0.85) | 11.5 µs (0.95) | 308.8 µs (25.47) | 237.4 µs (19.58) | 13.5 µs (1.11) |
| 4 | 96.1 µs | 95.9 µs (1.00) | 107.1 µs (1.11) | 105.4 µs (1.10) | 2.36 ms (24.58) | 5.34 ms (55.61) | 286.2 µs (2.98) |
| 8 | 2 ms | 2 ms (1.00) | 2.13 ms (1.07) | 2.11 ms (1.06) | 21.1 ms (10.55) | 248 ms (124.02) | 10.5 ms (5.28) |
| 16 | 64.2 ms | 53.7 ms (0.84) | 99.3 ms (1.55) | 74.5 ms (1.16) | 144 ms (2.24) | 16.5 s (256.81) | 422 ms (6.57) |
| 32 | 1.05 s | 1.46 s (1.39) | 1.57 s (1.50) | 3.39 s (3.23) | 2.21 s (2.11) | – | 18.6 s (17.66) |

`Matrix<Rational>::inverse()`, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.23 µs | 0.06 µs (0.28) | 0.27 µs (1.16) | 0.43 µs (1.83) | 4.90 µs (21.08) | 0.34 µs (1.47) |
| 4 | 0.87 µs | 0.87 µs (1.00) | 0.88 µs (1.02) | 10.1 µs (11.64) | 54.5 µs (62.85) | 3.42 µs (3.94) |
| 8 | 5.78 µs | 5.46 µs (0.94) | 5.52 µs (0.96) | 13.8 µs (2.39) | 794.5 µs (137.44) | 66.0 µs (11.42) |
| 16 | 86.3 µs | 151.2 µs (1.75) | 113.3 µs (1.31) | 149.6 µs (1.73) | 12.7 ms (147.50) | 1.66 ms (19.24) |
| 32 | 1.08 ms | 1.44 ms (1.33) | 1.44 ms (1.33) | 1.1 ms (1.02) | 218 ms (201.59) | 32.3 ms (29.85) |
| 64 | 13.3 ms | 16.6 ms (1.25) | 18.6 ms (1.40) | 9.72 ms (0.73) | 3.9 s (293.39) | 539 ms (40.52) |
| 128 | 169 ms | 173 ms (1.02) | 232 ms (1.37) | 96 ms (0.57) | 76 s (449.74) | 9.21 s (54.50) |

`Matrix<Rational>::inverse()`, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99 (median, and the time over `a`'s):

| n | `a` | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.29 µs | 0.09 µs (0.29) | 0.32 µs (1.08) | 3.62 µs (12.41) | 5.69 µs (19.51) | 0.37 µs (1.28) |
| 4 | 2.30 µs | 3.30 µs (1.44) | 3.31 µs (1.44) | 18.3 µs (7.95) | 101.7 µs (44.26) | 6.43 µs (2.80) |
| 8 | 39.5 µs | 40.9 µs (1.03) | 41.0 µs (1.04) | 220.1 µs (5.57) | 2.5 ms (63.26) | 251.4 µs (6.36) |
| 16 | 637.2 µs | 1.12 ms (1.76) | 834.7 µs (1.31) | 2.28 ms (3.58) | 68.8 ms (108.02) | 8.49 ms (13.33) |
| 32 | 11.7 ms | 17.9 ms (1.53) | 15.1 ms (1.29) | 23.2 ms (1.98) | 2.19 s (186.57) | 233 ms (19.86) |
| 64 | 311 ms | 350 ms (1.13) | 384 ms (1.24) | 247 ms (0.79) | 91 s (292.78) | 6.75 s (21.72) |
| 128 | 8.85 s | 6 s (0.68) | 10.9 s (1.23) | 2.73 s (0.31) | – | – |

`Matrix<Rational>::inverse()`, 64-bit parts (median, and the time over `a`'s):

| n | `a` | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.97 µs | 0.88 µs (0.30) | 2.54 µs (0.85) | 27.3 µs (9.19) | 51.8 µs (17.41) | 1.79 µs (0.60) |
| 4 | 42.1 µs | 46.7 µs (1.11) | 46.5 µs (1.10) | 581.1 µs (13.81) | 1.67 ms (39.60) | 96.5 µs (2.29) |
| 8 | 949.0 µs | 1.3 ms (1.37) | 1.3 ms (1.37) | 21.7 ms (22.84) | 82.8 ms (87.27) | 5.79 ms (6.11) |
| 16 | 34.7 ms | 80.3 ms (2.32) | 44.1 ms (1.27) | 1.21 s (34.87) | 7.77 s (223.89) | 323 ms (9.30) |
| 32 | 1.35 s | 2.97 s (2.20) | 2.07 s (1.54) | 63 s (46.76) | – | 23.2 s (17.21) |

`Matrix<Rational>::inverse()`, 256-bit parts (median, and the time over `a`'s):

| n | `a` | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 11.3 µs | 6.52 µs (0.58) | 12.0 µs (1.07) | 155.5 µs (13.76) | 248.4 µs (21.98) | 9.60 µs (0.85) |
| 4 | 206.0 µs | 290.1 µs (1.41) | 289.4 µs (1.41) | 3.72 ms (18.07) | 11.5 ms (55.80) | 539.7 µs (2.62) |
| 8 | 7.21 ms | 9.5 ms (1.32) | 9.65 ms (1.34) | 165 ms (22.94) | 1.1 s (153.17) | 40.5 ms (5.62) |
| 16 | 262 ms | 896 ms (3.42) | 398 ms (1.52) | 8.57 s (32.70) | 130 s (496.90) | 2.98 s (11.39) |

- Below the thresholds the method is the issue's, Bareiss's elimination on the rows cleared of their denominators. Against FLINT's own Bareiss on the same rows (`fmpz_mat_solve_fflu`, `fmpz_mat_det_bareiss`): for small integers 1.2 to 1.7 times as fast, for small fractions 0.87 to 1.19 of its speed, for 64-bit and 256-bit parts 0.59 to 1.39 from 2 × 2 to 16 × 16, where FLINT's GMP-based exact division of a few limbs to a few dozen is faster for the shorter ones. The inverse, Bareiss's here and in FLINT up to its multimodular range, is 1.13 to 3.4 times as fast as FLINT's from 16 × 16 to 64 × 64.
- From the thresholds, Dixon's lifting: solutions of 32 × 32 to 128 × 128 take 0.50 to 0.82 of FLINT's time (FLINT's `fmpq_mat_solve` is Dixon's lifting too) and 1/1.4 to 1/39 of Bareiss's; determinants of small integers are 1.17 to 1.2 times as fast as FLINT's from 64 × 64, of fractions 0.84 times at 64 × 64 and 1.45 at 128 × 128.
- Losses: FLINT is faster for the smallest matrices, 2 × 2 (1.5 to 5 times as fast), where the cost here is allocations, a rational holding its parts in two vectors where FLINT keeps a small integer in a word. Its determinants are faster at 4 × 4 (0.50 to 0.79 of the time), for fractions and 64-bit or 256-bit parts up to 16 × 16 (0.78 to 0.94; but 1.35 times as fast here for 256-bit parts at 16 × 16), at 32 × 32 (0.62 to 0.65, and 0.38 for 256-bit parts, where FLINT's modular determinant needs fewer residues) and for 64-bit parts at 64 × 64 (0.65). PARI's multimodular inverse is faster from 64 × 64 for small integers (0.57 to 0.73 of the time) and fractions (0.31 to 0.79), and FLINT's at 128 × 128 for fractions (0.68). Dixon's lifting at its lower threshold for 256-bit parts, 16 × 16, takes 1.2 times as long as Bareiss's elimination for a solution (64.2 against 53.7 ms). The product of matrices of small integers or fractions takes FLINT 0.34 to 0.86 of the time, its integer products running on packed kernels; for 64-bit parts 0.71 to 1.69.

## Efficiency

- Time complexity: a sum or a product of parts of `n` limbs takes two or three greatest common divisors and four products of `n`-limb numbers, `O(M(n) log n)` with the half gcd for the longest; a comparison at most two products.
- Space complexity: `O(n)`, the parts and the intermediate products.
- `to_f64` and `to_f32`: `O(1)` but for about one fraction in 700 of random parts, `O(n)` for a denominator of `n` limbs; the conversion from a float `O(1)` and two vectors of at most 135 bytes.
- `FromStr` and `from_str_radix`: the reading of the parts, `O(M(n) log n)` for `n` digits, and for a fraction the gcd of the parts; a decimal with a negative exponent at most a remainder by `5^27`, or a gcd with `5^k` when its digits end in more than 27 factors of 5; `to_string` and `to_str_radix` the writing of the parts, `O(M(n) log n)`.
- Time complexity: `B_n` in about `O(n² log² n)` word operations, the Euler product's primes up to `n / 17` at falling precisions and the residues' `O(p)` for each prime up to about `0.15 n log2 n`; `B_0, ..., B_n` in `O(n³ log n)` word operations for the powers' passes, plus two products of each number's bits for each index; `H_n` in `O(M(n log n) log n)` for the tree, `M` the time of a product, plus `O(M(n))` for the reduction.
- Space complexity: the result and a few times its length for `B_n`, the residues' tables of 1 KB or 8 KB; for the vector, the results, and at most `n / 34` powers of the top index's precision.
- `det()` and `rank()` of an `n × n` matrix: about `n³/3` steps, each two products and an exact division of integers as long as the minors, `O(n³ M(n·b))` for elements of `b` bits; `solve(&b)` with `k` right-hand sides about `n³/3 + n²·k` of them and `n·k` gcds, `inverse()` about `4n³/3` and `n²` gcds; the product `n³` products of integers and `n²` gcds. The large determinants and systems: `n³/3` products modulo a prime and about `2·L·n²` products of words, `L` the length of the determinant over 61 bits, twice for the solutions, `O(n³ + n³·b/61)` word operations. Space: the matrix of integers and the right-hand sides, a vector of limbs for each element beyond the registers; Dixon's lifting the factors modulo the prime, `n²` words, and the digits, `L·n·k` words.

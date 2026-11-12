[A](../../../README.md) › [Math](../../../README.md#math)

# Rational numbers

`Rational` (in `a::math::rational`) is the fraction `p/q` of two [numbers](numbers.md), always in lowest terms with a positive denominator, so that each rational number has one form: zero is `0/1`, an integer `n` is `n/1`, and two rationals are equal when their parts are. The parts are private, so that the form holds; `numer`, `denom` and `into_parts` give them back.

- `Rational::new(p, q)` reduces any fraction with `q ≠ 0` to the form, `Rational::from_integer(n)` and `From` make integers, from a `Number` or any primitive integer.
- They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` and divide with `/` and `/=`, each operand owned or borrowed, and take a `Number` or a primitive integer as an operand on either side: `r + 1`, `2 * &r`, `&n / r`. Dividing by zero panics, as it does for integers.
- `-r` negates, `recip` is `1/r`, `abs` and `signum` are as for integers, and `pow` takes an `i32` exponent, negative ones by the reciprocal; `0⁰` is 1.
- `==`, `<` and the rest compare by value, with numbers on either side and primitive integers on the right; `Hash` agrees with `==`, `Default` is zero, `Sum` and `Product` take iterators of owned or borrowed rationals. `Display` and `Debug` write `-3/4`, or `5` for an integer, the formatter's width, fill and `+` applying to the whole.
- `Rational::try_from(x)` takes any finite `f64` or `f32` exactly: every float is `m·2^e`, so `0.1` is `3602879701896397/36028797018963968`, and `-0.0` is zero; NaN and the infinities are a `TryFromFloatError`. A float whose parts fit in 64 bits, as every float from `2^-11` to `2^12` does, is made in words with no allocation. `to_f64` and `to_f32` go back to the nearest float, ties to even, rounded once and correctly, subnormal numbers included: a number too large is infinity of its sign, one too small zero, `-0.0` for a negative one as IEEE 754 has it, and `0.0` for zero. Parts in words below `2^53` (`2^24` for `f32`) are floats exactly and one division rounds their quotient; other parts in words take the quotient of their nearest `f64`s, a float within two units in the last place, and the exact remainder of the fraction by it, in 128 bits, which says whether the float or a neighbour is the nearest.
- `Number::try_from(&r)` and `u8::try_from(&r)` up to `i128` and `usize` give an integer, a `TryFromRationalError` for a fraction or an integer out of the type's range.
- `"-3/4".parse::<Rational>()` reads integers, fractions `p/q` (the denominator without a sign, not zero, the fraction reduced) and decimals `1.25`, `.5`, `5.`, with an exponent `e` or `E` on decimals and integers, `-1.5e-3`, `2e10`, exactly: `0.1` is `1/10`. There is no whitespace, no `_`, nothing else, as for a `Number`. Parts of up to 64 bits, 19 or 20 digits, are read eight digits at a time into words and reduced there, with no allocation, and so are decimals whose digits make a word, with an exponent of up to 19 in magnitude: their twos or fives are the only factors they share with the power of ten. The exponent may be up to `Rational::MAX_PARSE_EXPONENT`, `1000000`, in magnitude: `1e1000000` is a numerator of 415 KB, read in about a millisecond, and the bound keeps a short text from asking for gigabytes. A `ParseRationalError` has a `kind()`: `Empty`, `InvalidDigit`, `ZeroDenominator` or `ExponentOutOfRange`.
- `to_str_radix(radix)` writes `-p/q` in a radix from 2 to 36, and `from_str_radix(s, radix)` reads integers and fractions in it, letters in either case, with no point or exponent in any radix (`e` is a digit from radix 15 on). Parts in words are written and read in registers in every radix: binary eight digits at a time, the other powers of two by shifts, and the other radices in blocks below `2^32`, each digit by a product with the radix's reciprocal; the only allocation is the text written.
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
- A rational whose parts both fit in 64 bits holds them in two words, with its sign and byte order, and allocates nothing; any other holds two numbers. Every value has the form its parts call for, whatever made it, so that the equality and the hash need not look past the form. `numer`, `denom` and `into_parts` give numbers: for a rational in words they are made the first time they are asked for, and kept in a `OnceLock` inside it, so that a `Rational` stays `Send` and `Sync`; the arithmetic, the comparisons, the rounding and the conversions never ask for them, and read the words in place, also when the other operand is long. `size_of::<Rational>()` is 64 bytes, as it was with two numbers.
- When every part fits in 64 bits, the arithmetic, the comparisons and the rounding are done in registers, the products in 128 bits. The time there is the greatest common divisors, by the binary method (four cycles a step on an Apple M4, about 0.7 steps a bit, as GMP's `mpn_gcd_11` takes them), and the one branch of each that no predictor can know, the end of its loop: random operands make about one misprediction a gcd, 13 cycles, and a branch on their signs or on whether a gcd is 1 another one in two or three. So the sum `(a·d ± c·b)/(b·d)` is computed whole, without branches (both the sum and the difference of the products, one chosen), before `g = gcd(b, d)`, which the processor then runs beside it: when `g = 1`, as for three pairs of random denominators in five, that is the result in lowest terms. Otherwise it is divided by `g·gcd(t/g, g)`, Knuth's reduction for the whole numerator `t`, which is `gcd(t mod g², g²)` since `t/g` is prime to `b/g` and to `d/g`: one remainder, one short gcd and two divisions, in words when the products fit in 64 bits, and Knuth's steps in 128 bits when they do not, with the numerator's 129th bit apart and Hensel's exact division modulo `2^128`. A product's two gcds, `gcd(a, d)` and `gcd(c, b)`, run in step in one loop, both chains of dependent steps side by side and one loop end to mispredict rather than two, and the whole products, computed beside them, are the result when both are 1. The binary method stops when one of the two numbers is 1, as FLINT's `n_gcd` does, and takes a remainder first when one number is longer than the other by 16 bits or more. A comparison is the two products and the signs, chosen without branches. On random parts below `2^16`, `&a + &b` takes 40 cycles where the inputs are new to the predictor, against FLINT's 84 and 193 before #27; on a pool of 512 that it has learnt, 26 against FLINT's 38.
- Parts of up to 16 bytes are read straight from their bytes as `u128`s: the gcds of two `u128`s (a limb with two longer by 16 bits or more by a remainder first, as `mpz_gcd` does, since the binary method would take a step a bit of the difference), the exact quotients by Hensel's division modulo `2^128`, the products in four limbs and a sum's numerator in five, all in registers, the limbs taking only a sum whose denominators share more than 64 bits. Otherwise the four parts are read into limbs once, the gcds (Lehmer's method and the half gcd of [numbers](numbers.md)), exact divisions and products are done on limbs in working memory (in the operation's own frame for parts of up to 12 limbs together), and the two parts of the result are written once, into the vectors of the operand assigned to or taken by value where they have room, with an eighth to spare when they grow. A part of one limb takes the steps for one limb: its gcd with a long part is a remainder by it, by the powers of `B` modulo it (computed once for both remainders of a sum), and the exact divisions and products by it are single passes. This is what a sum of many terms does at every term: `H_10000` by `+=` takes 0.45 of the time of the same sum on numbers.
- `Sum` and `Product` combine their terms in a balanced tree, as a binary counter merges them, with at most `log₂ n` partial results kept: each addition is of two sums of about the same number of terms, so the parts of a sum of many short terms grow in a few long additions rather than one long addition a term. `H_n` takes `O(M(n) log n)` instead of `O(n²)`: `H_100000` takes 1,200 cycles a term against 14,129 by `+=`. While the sum is in words the terms are added one at a time in registers, and the tree takes over from there, its first 8 partial results kept in place rather than in a vector.
- `Rational::new` reduces in the vectors it is given: parts of up to 8 bytes by the gcd of two limbs, up to 16 by that of two `u128`s and Hensel's exact division, longer ones on limbs; a gcd of 1, the common case, writes nothing.
- Two rationals compare by their signs, then by `a·d` against `c·b`, first bounded from the top 63 bits of each part: a part of `L > 63` bits lies in `[h, h + 1]·2^(L − 63)` for its top 63 bits `h`, and the products of the bounds, in 126 bits, decide unless the two numbers agree to about 120 bits. So a comparison takes a few dozen cycles at any length, and only numbers that close are multiplied out.
- `floor`, `ceil` and `trunc` take the truncated quotient, one less or more by the sign (a remainder over a denominator above 1 is never zero in lowest terms); `round` and `round_ties_even` the quotient and the remainder on limbs, the remainder doubled against the denominator; parts of up to 16 bytes in registers. Each gives a `Number`, a vector: of the 32 to 36 cycles of a `floor`, `ceil` or `trunc` of short parts, 25 are the allocation of its bytes and their release, where FLINT's `fmpz_fdiv_q` writes a word into a variable that exists (7 to 8 cycles) and malachite's `Integer` keeps a word in place (16 to 27). `fract`, `abs`, `signum`, `recip`, `-x` and `pow` of short parts give rationals in words, allocating nothing.
- A float is `±m·2^e` from its bits: `m` is divided by its trailing zeros, which are added to `e`, and the rational is `m·2^e` when `e ≥ 0` and `m/2^-e` otherwise, in lowest terms as it is, both parts built directly as little-endian bytes. Its cost is the two vectors a rational has when a part passes 64 bits, and nothing otherwise.
- `to_f64` of `a/b`: when both parts are below `2^53`, both are floats exactly and one division of floats is correctly rounded. Otherwise the top 128 bits of `|a|` are divided by the top 64 bits of `b` in registers, by the reciprocal of the divisor as a division of limbs takes it (Möller and Granlund's), which gives a quotient `q` of 64 or 65 bits with `q − 4 < |a|/b·2^s < q + 1`; the result is `q` rounded with a sticky bit, unless the one halfway point between two floats near `q` lies in that interval. When `b` fits in 64 bits the quotient and its remainder are exact, and the bits of `a` below its top 128 are read only when the remainder is zero. Otherwise, about one fraction in 700, `|a|·2^s` (or `|a|/2^s`, its dropped bits kept for the sticky bit) is divided by `b` for a quotient of 65 or 66 bits, its remainder setting the sticky bit: linear in the length of `b`. The rounding is done on the integer bits, without branches, and a carry out of the significand runs into the exponent and up to infinity on its own. GMP's `mpq_get_d` and FLINT's `fmpq_get_d`, which calls it, truncate towards zero instead, so they differ from the nearest float in about half of all fractions; they also divide at the full length of the denominator, so they take time linear in it.
- `to_f32` rounds the rational to 24 bits the same way, once: rounding the `f64` again would be one off where the `f64` lands on a halfway point between two `f32`s.
- `FromStr` finds the first `/`, `.`, `e` or `E` and reads the parts as numbers are read (by divide and conquer from a few hundred digits). A fraction is reduced by the gcd of its parts, which, for text already in lowest terms, is 81 to 92% of the time from 100 digits on. A decimal `d₁…dₙ.f₁…fₘ e x` is the integer `d₁…dₙf₁…fₘ`, its trailing zeros dropped, times `10^(x − m)`, built as `5^k·2^k`. A negative power leaves `N/10^k` in lowest terms but for the factors 2 or 5 of `N`, which its last digit tells apart: `1, 3, 7, 9` share neither, an even digit only twos, which a shift removes, and `5` only fives, which one remainder by `5^27` counts (a gcd with `5^k` past 27). Parts and decimals whose digits make a word are read in registers, eight digits a step by a check and three products of the eight bytes at once, a fraction reduced by the binary gcd of two words and a decimal by its twos or fives alone.
- `Display` writes a fraction in words backward into one buffer of 42 bytes, blocks of eight digits as four independent pairs from a table, and gives the formatter the whole in one write when there is no width and no `+`; longer fractions of up to 320 characters are written on the stack too, so that the only allocation is the text the formatter keeps; `to_str_radix(10)` writes both parts into one vector.

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

- The terms of a continued fraction are the quotients of Euclid's algorithm on the denominator and the remainder of the numerator. When the parts fit in 128 bits it runs in registers. Longer parts take the greatest common divisor's own reductions, which find many of Euclid's steps at once and apply their matrix to the whole numbers: hgcd2's matrices from the top two limbs (Lehmer's method), and from 600 limbs Möller's half gcd on the top two thirds of the numbers. These reductions now tell of the steps they take, to a sink that the greatest common divisors leave empty, which costs them nothing (measured below); a step that they do not take in full is followed by the rest of its quotient on the same number, so the quotients of consecutive steps on one number add up to Euclid's. A continued fraction so costs what a greatest common divisor costs, `O(M(n)·log n)` for parts of `n` limbs, and the writing of its terms, about 0.58 a bit of the denominator for random numbers. Each term is a `Number` with a vector of its own, and its allocation and release are the rest of the cost: 33 cycles a term on an Apple M4 Max, exactly what a bare allocation and release of 8 bytes take, so nearly all of the time for parts of a limb and about half of it from 16 to 256 limbs, where FLINT keeps small terms in the words of its `fmpz`s and allocates nothing for them. `Vec<Number>` leaves no way around one allocation a term; a term found in registers is written into a vector of room for 8 bytes by one store of the whole limb, cut to its significant bytes (the allocator gives a request of 1 to 16 bytes the same block), which took a continued fraction of 64-bit parts from 1450 cycles to 1310 against a copy of the exact length.
- A rational whose parts both fit in 64 bits is read in place, from its words: `continued_fraction`, `convergents`, `limit_denominator`, `simplest_between` and the reconstructions run Euclid's algorithm on them in registers, and give a result whose parts fit in 64 bits as words, so that nothing is allocated but the terms of a continued fraction. `from_continued_fraction` multiplies the first run of terms of 64 bits in registers while the entries fit, which for short terms is all of them, and goes on from where it stopped on numbers otherwise; `simplest_between` of two ends in words takes their floors and ceilings in registers, and `limit_denominator` with a bound of 1 rounds in registers. Parts of up to 128 bits run in registers too: a quotient of Euclid's algorithm is found by at most three subtractions when it is 1, 2 or 3, two thirds of them by Gauss and Kuzmin's law (41.5 %, 17 % and 9.3 %), and by the library's division of `u128`s, some 40 cycles, otherwise.
- `from_continued_fraction` multiplies the matrices `(ai, 1; 1, 0)`, whose product is `(pn, pn-1; qn, qn-1)`: runs of terms in registers while the entries fit in 64 bits, and the runs' matrices by a balanced tree of products, `O(M(n)·log n)` for a result of `n` limbs. `convergents` takes the recurrence `pk = ak·pk-1 + pk-2` from the terms: for parts of up to 128 bits in registers, where the terms are not written at all; for longer ones in registers while the entries fit in 127 bits, and then on limbs in place, `pk` written over the vector of `pk-2`, so that each convergent costs the two vectors of its own parts rather than four, 0.55 to 0.60 of the time from 4 to 16 limbs.
- `limit_denominator` and the reconstructions stop Euclid's algorithm at a bound: on the cofactors, which are the convergents' denominators (`limit_denominator`, on the denominator and the numerator's remainder), or on the remainders (the reconstruction, on `m` and `r mod m`, the first remainder at most `n`). They keep one row of the reduction matrix, the cofactors of the second number, and the bound gives a length to which the numbers can be reduced without passing it, as `|t_i|·r_(i-1) ≤ A` ties the cofactors to the remainders: the half gcd takes them there, on top parts sized so that it never goes below that length, and the row is multiplied by each of its matrices. The last steps are hgcd2's matrices tried against the bound, and when one passes it, its steps one at a time from where it started. Then `limit_denominator` chooses the convergent or the semiconvergent by one comparison of the remainders, `(2j·Q_S + Q_L)·S ≤ Q_S·L`, and the reconstruction checks `|t| ≤ d` and `gcd(r_i, t) = 1`, Wang's conditions. Both are `O(M(n)·log n)`.
- `simplest_between` goes down the Stern–Brocot tree from both ends at once, without ordering them first: the floors by one division each, an integer between them being the answer; then the terms of the fractional parts in step, until they part. There the end whose term is the smaller, or whose expansion ends, is the lower one, and the rational takes the common terms and that end's term, plus one unless the expansion ends there. The common terms are found by the reductions of the longer end, without the last step when it is not taken in full: a reduction is kept when the other end's pair stays positive under its matrix with its larger number in the same place, which is when its quotients are the same. From 1400 limbs the half gcd on the top half of the longer end goes first, on shorter tops where the ends part within it (twice the last top kept, then halved); then hgcd2's matrices, and in the one where the ends part its steps one at a time. The result comes out of the matrix of the common steps. Past 600 limbs of common terms left to Lehmer's method, both expansions are computed in full and compared, `O(M(n)·log n)`.

## Bernoulli and harmonic numbers

`Rational::bernoulli(n, order)` is the Bernoulli number `B_n`, `Rational::bernoulli_numbers(n, order)` the vector `B_0, B_1, ..., B_n`, and `Rational::harmonic(n, order)` the harmonic number `H_n = 1 + 1/2 + ... + 1/n`.

- `B_0 = 1`, `B_1 = −1/2`, `B_2 = 1/6`, `B_4 = −1/30`, and `B_n = 0` for every odd `n > 1`: the numbers of `x / (e^x − 1) = Σ B_n x^n / n!`, the convention of PARI, FLINT and Mathematica. The other convention, `B_1 = +1/2` (of `x / (1 − e^−x)`), differs only there: it is `-Rational::bernoulli(1, o)`, or `(−1)^n B_n` for any `n`. `H_0 = 0`.
- Each result is in lowest terms, both parts in the byte order `order`, in the shortest form. The indices are `u64`s; nothing panics but when a result or its working memory cannot be allocated: `B_n` has about `n log2(n / (2πe))` bits (1.25 M for `n = 10^5`, 15.8 M for `10^6`), `B_0, ..., B_n` together about `n² log2(n / 17) / 4` bits (27 MB for `n = 10^4`, 3.7 GB for `10^5`), and `H_n` about `1.44 n` in each part.
- Taking the vector costs far less than taking the numbers one by one: up to `10^4`, 0.39 s against 5.7 s.

The method, for one Bernoulli number:

- Up to `B_100` from tables: to `B_58` numerators in `i128`s (`B_60` is the first above `2^127`), then the 75 limbs of the numerators from `B_60` to `B_100` one after another, 789 bytes with their ends and denominators. Those whose parts fit in 64 bits, `B_0` to `B_34` and `B_38`, and the zeros, are written as words in the caller: the lookup is inlined and the rest is out of line, so that it saves no registers and allocates nothing.
- Beyond, `|B_n| = 2 n! ζ(n) / (2π)^n`, as PARI's `bernfrac` and FLINT's `bernoulli_fmpq_ui` compute it. The denominator `d` is von Staudt and Clausen's: the product of the primes `p` with `p − 1` dividing `n`, from the divisors of `n`. With `K = 2 d n!` exactly and `U` a bound of the bits of the numerator `|B_n| d`, the quotient `K ζ(n) / (2π)^n` is computed to `w = U + 8` bits relatively and rounded: `π` from the 16,384 bits of the partition numbers' table or their Chudnovsky series, `(2π)^n` by squares and products cut to `w + log2 n + 6` bits (an early error is raised to the rest of the power), and `1/ζ(n) = Π_p (1 − p^−n)` in fixed point over the primes with `p^n < 2^w`, each factor `z ← z − z / p^n` with `z` and `p^n` cut to the bits `w − n log2 p` that it moves the product by and `log2 n + 8` more, `p^n` exact while it is short. The bounds of the errors add up to less than `2^−w` relatively, below `1/2^8` of a unit. The products, squares and quotients reuse a few vectors for the whole computation, the quotients by divisors of up to 8 limbs by Knuth's algorithm D in them (each step a division of two limbs by one by the reciprocal of the divisor's top limb), longer ones on numbers: a short number allocates a handful of times rather than at every step, which took `B_300` from 22,800 cycles to 17,000 and `B_1000` from 166,000 to 152,000.
- From `n = 8000`, as FLINT's `_bernoulli_fmpq_ui_multi_mod` does, a share of the numerator's bits, from 10 % at 8000 to 22 % from `10^5` (`α = 0.1 + 0.05 log2(n / 10^4)`), is taken modulo primes instead: the residues of `|B_n| d` modulo the primes from 5 up, by Harvey's method ("A multimodular algorithm for computing Bernoulli numbers", 2010), joined by the Chinese remainder theorem into `R` modulo their product `M`; the formula is then needed only within `M / 2^11`, `w = U − log2 M + 12` bits, and the result is `R` plus the multiple of `M` nearest to the formula's value less `R`. The Euler product's primes are those below `2^(w/n)`, so a fifth fewer bits leaves the formula about a third of its time: `B_(10^5)` took 362 ms by the formula alone, 157 ms with 20 % of its bits modulo primes.
- Modulo a prime `p ≥ 5` with `2^n ≠ 1`, Voronoi's congruence with `c = 2` is `(2^n − 1) B_n = n 2^(n−1) Σ_(j=1)^(p−1) j^(n−1) floor(2j / p)` modulo `p`, and `floor(2j / p)` is the first bit of the binary expansion of `j / p`. Taken with `p − j`, whose bit is the complement and whose power the opposite, the sum is `Σ_(j ∈ H) j^(n−1) (2 bit(j) − 1)` over half the units. Doubling `x` shifts the expansion of `x / p` by a bit and multiplies the power by `c = 2^(n−1)`, so `H` is taken as cosets `x_t ⟨2⟩` of the order `m` of 2 (half of each when `m` is even, `−1` then being `2^(m/2)`), `x_t = g^t` for a generator `g` of the cosets: per coset, the expansion of `x_t / p` 64 bits a step, `floor(y 2^64 / p)` by Granlund and Montgomery's product with `ceil(2^128 / p)` (exact, as `y 2^64 < 2^96`), and the bits weighted by tables of the sums of `c^j` over the bits of each chunk of a step — sixteen tables of 16 below `2^18`, eight of 256 above — times the powers `c^(64β)` of the steps; the chains of remainders in four lanes at once. About 0.11 cycles a unit of `H` for the long primes, and 0.5 to 2 µs of preparation for each (the order of 2, the generator, the tables).

For the vector:

- Up to `B_100` from the tables, those in words written as words; up to `n = 150`, from the tangent numbers by Brent and Harvey's recurrence ("Fast computation of Bernoulli, Tangent and Secant numbers", 2011): `T_k = (k − 1)!`, then `T_j ← (j − k) T_(j−1) + (j − k + 2) T_j` for each `k` and `j ≥ k`, `m²/2` linear steps in place for `m = n / 2`; then `|B_2k| = 2k T_k / (4^k (4^k − 1))`, an exact division for `|B_2k| d_2k`.
- Beyond, the formula from the top index down, as FLINT's `bernoulli_fmpq_vec_no_cache` does: `G_k = 2 k! / (2π)^k` from `G_(k+2)` by a product with `(2π)²` (cut to the limbs that count) and a division by the limb `(k + 2)(k + 1)`; `ζ(k) = (1 − 2^−k)^−1 Σ_(j odd) j^−k` from the powers `j^−k` in fixed point of whole limbs, each from `j^−(k+2)` by a product with the limb `j²` and a cut of the limbs no longer needed, in one pass that also adds it into the sum. The precisions are those of `|B_k| d_k` with `log2 n + 8` bits more, made to fall with `k` by taking the largest denominator up to each index; the bounds of the errors of `G` and of every power are kept in units of their last bits, and a number whose bound would not leave less than 1/4 would be computed alone, which the bounds do not let happen. Two products of the precision for each index, and linear passes over the powers.

For a harmonic number:

- Up to `H_88`, the last whose parts are below `2^128`, from a table made at compile time by Knuth's sums, to `H_46` written as words in the caller as the Bernoulli numbers in words are; up to `H_124`, `H_88` plus the remaining terms, summed in a `u128` while they fit and added as rationals, which costs less than the splitting's preparation for so few terms (`H_89` in 509 cycles against 2487, `H_120` 2484 against 2968).
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
- Matrices of up to 8 rows whose elements are in words run on the stack, with nothing allocated but the result. The rows are cleared into an array of `i64`s, each rational's words read in place, never through `numer()`, which would make and keep two numbers for it. One product of the squares of the rows' norms in floating point, Hadamard's bound, then tells whether every minor of the matrix stays below `2^61`. If it does, the elimination and the back substitution run with no check at all. A step is two products in 128 bits, a subtraction, and an exact division by the previous pivot: the processor's division when the pivot divides one or two elements, and otherwise a product by the inverse of its odd part modulo `2^64`. The sums of the back substitution fit in 128 bits, and each element of the result is one or two greatest common divisors of words and a rational written in words. Order 2 is Cramer's rule in 128 bits, which needs no bound. A `2 × 2` determinant with a fraction in it is `a·d − b·c` by the rationals' own arithmetic, as FLINT's `fmpq_mat_det` takes it, and so is the inverse of a `2 × 2` matrix beyond the registers: its adjugate times the reciprocal of its determinant.
- Beyond that, the integers are held in registers while their magnitudes are below `2^63`: a step is then two products in 128 bits, a subtraction, and an exact division as a product by the inverse of the divisor's odd part modulo `2^64`, with a comparison against a precomputed bound telling whether the quotient fits. Beyond the registers, each element is a vector of limbs that the steps write over, the products go to two buffers of working memory, and the exact quotient is Hensel's, from the low end. From quotients of 150 limbs it is a product by the inverse of the divisor modulo a power of `2^64`, computed once by Newton's iteration for all the elements one pivot divides. A step then allocates nothing once the vectors have grown. A matrix in words whose bound passes `2^61` brings its cleared integers along, so it is not read again. A row of rationals in words whose common denominator passes a word is cleared on limbs: a remainder, a greatest common divisor of words and a product by a word for each denominator, then each numerator times the multiple's exact quotient by its denominator, with no number made.
- Large systems with few right-hand sides go by Dixon's p-adic lifting (Dixon 1982), as FLINT's do: from 24 rows, or 16 when an integer of the cleared matrix passes 63 bits, with at most an eighth as many right-hand sides. `A` is factored once modulo a prime just below `2^62`, in Montgomery's form; each step solves `A·x ≡ r` modulo it and lifts `r ← (r − A·x) / p` exactly; once `p^L` passes twice the product of Hadamard's bounds on Cramer's numerators and on the determinant, the digits are joined by halves, and each element is the one fraction within the bounds congruent to the result (Wang's rational reconstruction, by Lehmer's steps in registers), as a rule for the first element alone, the others then by the common denominator. The work is `O(n³ + L·n²)` operations on words, against `n³/3` products of numbers as long as the minors.
- Large determinants, from 28 rows (22 with an integer past 63 bits), are the least common multiple `l` of the denominators of a solution of `A·x = b` for a fixed pseudorandom `b`, which divides the determinant and is most of it as a rule, times `det(A) / l` from the determinants modulo primes by the Chinese remainder theorem, until the primes' product passes twice Hadamard's bound over `l`: one prime, as a rule (FLINT's `fmpz_mat_det_modular_accelerated`).
- The product `A·B` of matrices of rationals clears the rows of `A` and the columns of `B` and multiplies the integers: in 64-bit registers when their lengths and the length of the sums allow it, in 128 bits when they allow that. It then divides each element once, by `l_i·r_j`, through its greatest common divisor with `l_i` and then with `r_j`. That is `n³` products of integers and `n²` reductions, against `n³` products and sums of rationals, each with a gcd or two. When every element is in words and the cleared integers fit in them, all of it runs on the stack for up to 64 elements a side (in vectors beyond), each rational's words read in place, and an element whose multiples are both 1 needs no gcd at all. Otherwise, with an inner dimension of 2, or of 4 and a denominator past 32 bits, it is the sums of products of rationals, whose register paths are faster there.

## Measured

### Arithmetic

Measured on an Apple M4 Max on 2026-11-12, with nothing else running (load average 2.1 to 5.1), one thread, in cycles per operation at the clock measured before each round (4.5 GHz): the median of 21 rounds (three passes of seven, interleaved, the contenders' order rotating), over pools of random rationals (fresh each round) whose numerators and denominators have the number of limbs given (`s`: parts below `2^16`; `1`: parts of up to 64 bits, both forms in words), little-endian (big-endian within 10 % at short parts and 4 % from one limb), against GMP 6.3's `mpq_t` (linked statically, into a variable with room for its result: no allocation), FLINT 3.6's `fmpq` (the same), num-rational 0.4.2 (`BigRational`) and malachite-q 0.14.0. The pools hold 16,384 rationals (16,384 limbs' worth for long parts), too many for the branch predictor to learn the gcds' loops; the pools of 512 of the #16 campaign, which it does learn, have a table of their own. "Common factors": the denominators share a factor of half their length (for `+` and `−`), or each numerator shares one with the other's denominator (for `·` and `/`). The comparisons were measured again after a change to their inlining, the same day (load average 2.4 to 2.7).

`&a + &b`, a new rational each time, random parts:

| Limbs | ours | GMP | FLINT | malachite | num-rational |
| --- | ---: | ---: | ---: | ---: | ---: |
| s | 40 | 183 | 84 | 187 | 1,639 |
| 1 | 318 | 351 | 466 | 509 | 7,532 |
| 2 | 722 | 767 | 836 | 1,724 | 15,635 |
| 4 | 2,865 | 3,386 | 3,338 | 3,556 | 32,332 |
| 16 | 12,598 | 16,442 | 15,891 | 15,634 | 170,428 |
| 64 | 64,136 | 82,877 | 80,134 | 78,773 | 1.52 M |
| 1024 | 2.86 M | 3.94 M | 4.06 M | 4.27 M | 335 M |

`&a * &b`:

| Limbs | ours | GMP | FLINT | malachite | num-rational |
| --- | ---: | ---: | ---: | ---: | ---: |
| s | 70 | 207 | 95 | 220 | 2,028 |
| 1 | 308 | 538 | 728 | 671 | 9,375 |
| 2 | 1,139 | 1,272 | 1,408 | 3,128 | 19,027 |
| 4 | 5,117 | 6,294 | 6,366 | 6,660 | 40,047 |
| 16 | 22,940 | 31,156 | 30,908 | 27,615 | 211,425 |
| 64 | 112,234 | 149,308 | 150,413 | 136,297 | 1.81 M |
| 1024 | 5.06 M | 6.76 M | 6.84 M | 7.45 M | 385 M |

`a += &b` and `a *= &b` on a fresh copy of `a` (GMP and FLINT: `mpq_add(a, a, b)` on a fresh copy), random parts and common factors:

| Limbs | `+=` ours | GMP | FLINT | common: ours | GMP | FLINT | `*=` ours | GMP | FLINT | common: ours | GMP | FLINT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s | 38 | 263 | 69 | 55 | 320 | 96 | 57 | 317 | 85 | 40 | 335 | 66 |
| 1 | 304 | 484 | 494 | 379 | 587 | 591 | 305 | 656 | 835 | 138 | 465 | 456 |
| 2 | 742 | 905 | 863 | 966 | 1,115 | 1,157 | 1,209 | 1,370 | 1,511 | 766 | 977 | 1,434 |
| 8 | 6,110 | 7,766 | 7,660 | 7,075 | 8,659 | 8,589 | 11,306 | 14,397 | 14,715 | 7,100 | 8,936 | 9,208 |
| 64 | 65,649 | 84,008 | 85,230 | 68,639 | 89,022 | 89,051 | 112,392 | 151,891 | 152,928 | 62,803 | 84,029 | 85,108 |
| 1024 | 2.81 M | 3.94 M | 4.08 M | 3.32 M | 4.38 M | 4.37 M | 5.26 M | 6.96 M | 7.08 M | 4.1 M | 4.89 M | 4.82 M |

The same at short parts and one limb on a pool of 512 rationals timed over and over, whose branches the predictor learns (the #16 campaign's pools):

| Operation | Limbs | ours | GMP | FLINT | malachite |
| --- | --- | ---: | ---: | ---: | ---: |
| `&a + &b` | s | 26 | 108 | 38 | 101 |
| `&a + &b` | 1 | 319 | 323 | 438 | 494 |
| `&a * &b` | s | 37 | 144 | 40 | 107 |
| `&a * &b` | 1 | 314 | 531 | 688 | 680 |
| `a += &b` | s | 24 | 240 | 37 | 100 |
| `a += &b` | 1 | 288 | 448 | 431 | 479 |
| `a *= &b` | s | 36 | 291 | 40 | 104 |
| `a *= &b` | 1 | 300 | 640 | 715 | 637 |

Operands of different sizes (`a` and `b` of the limbs given, `s` for parts below `2^16`), the words against numbers and numbers of one limb past them:

| Limbs of a, b | `+` ours | GMP | FLINT | malachite | `·` ours | GMP | FLINT | malachite | `+=` ours | GMP | FLINT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s+2 | 308 | 228 | 287 | 577 | 378 | 294 | 306 | 544 | 300 | 423 | 301 |
| 1+2 | 471 | 395 | 524 | 834 | 699 | 627 | 808 | 991 | 472 | 595 | 565 |
| 2+1 | 469 | 393 | 525 | 832 | 708 | 638 | 824 | 983 | 489 | 589 | 546 |
| 1+4 | 601 | 419 | 553 | 902 | 830 | 660 | 852 | 1,052 | 589 | 596 | 584 |
| 1+16 | 857 | 592 | 721 | 1,148 | 1,122 | 881 | 1,196 | 1,362 | 835 | 764 | 748 |
| s+64 | 1,136 | 670 | 1,239 | 2,079 | 1,269 | 786 | 1,663 | 2,210 | 1,111 | 888 | 1,431 |
| 1+64 | 1,400 | 967 | 1,669 | 2,419 | 1,603 | 1,384 | 2,481 | 2,742 | 1,328 | 1,173 | 1,827 |
| 1+1024 | 11,349 | 9,660 | 12,210 | 24,357 | 11,283 | 11,646 | 26,520 | 26,160 | 11,525 | 9,948 | 12,504 |

Comparisons (`cmp`), random signs, and positive with all parts of the same length:

| Limbs | random: ours | GMP | FLINT | malachite | num-rational | same length: ours | GMP | FLINT | malachite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s | 7.6 | 23 | 25 | 35 | 398 | 7.5 | 48 | 26 | 55 |
| 1 | 7.7 | 46 | 99 | 59 | 503 | 7.6 | 48 | 141 | 81 |
| 2 | 38 | 47 | 93 | 77 | 673 | 56 | 56 | 137 | 120 |
| 8 | 33 | 136 | 148 | 126 | 778 | 55 | 233 | 238 | 210 |
| 64 | 37 | 3,616 | 3,369 | 2,572 | 1,966 | 56 | 7,279 | 6,687 | 4,900 |
| 1024 | 38 | 233,009 | 248,217 | 77,670 | 19,037 | 51 | 523,070 | 560,881 | 184,364 |

`Rational::new` (GMP and FLINT: `mpq_canonicalize` of a fresh copy), `floor` (GMP and FLINT: `mpz_fdiv_q` into a variable) and `round` (FLINT: `fmpz_ndiv_qr`; GMP has none), the numerator of the given limbs and the denominator of half as many for the rounding:

| Limbs | `new`: ours | GMP | FLINT | malachite | `floor`: ours | GMP | FLINT | malachite | `round`: ours | FLINT | malachite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s | 122 | 93 | 89 | 118 | 36 | 74 | 8.2 | 25 | 38 | 29 | 33 |
| 1 | 277 | 264 | 378 | 343 | 34 | 52 | 57 | 28 | 35 | 129 | 42 |
| 2 | 576 | 611 | 704 | 1,477 | 115 | 58 | 77 | 118 | 138 | 176 | 143 |
| 8 | 5,316 | 7,165 | 7,190 | 6,461 | 259 | 209 | 226 | 324 | 364 | 290 | 390 |
| 64 | 52,390 | 72,072 | 71,898 | 63,203 | 1,707 | 2,695 | 2,697 | 4,147 | 2,591 | 2,762 | 5,282 |
| 1024 | 2.46 M | 3.27 M | 3.23 M | 3.61 M | 142,624 | 212,478 | 209,417 | 276,430 | 174,053 | 215,323 | 314,837 |

The other operations of short rationals, parts below `2^16` and of 64 bits (GMP and FLINT into a variable that exists: `mpz_cdiv_q`, `mpz_tdiv_q`, `mpq_abs`, `mpq_neg`, `mpq_inv`, `mpq_set`, `mpq_set_si`, `mpq_equal` and FLINT's `fmpq_` and `fmpz_` equivalents; `from(i64)` of a random integer of the size; `eq` of random pairs, one in four equal; `hash` into a hasher that multiplies in a word at a time):

| Operation | 16 bits: ours | GMP | FLINT | malachite | num-rational | 64 bits: ours | GMP | FLINT | malachite | num-rational |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ceil | 32 | 69 | 7.9 | 27 | 420 | 31 | 64 | 92 | 33 | 503 |
| trunc | 34 | 32 | 7.3 | 16 | 199 | 32 | 37 | 76 | 14 | 201 |
| round_ties_even | 36 | – | – | – | – | 36 | – | – | – | – |
| fract | 11 | – | – | – | 181 | 11 | – | – | – | 236 |
| abs | 9.9 | 12 | 11 | 5.4 | 78 | 10 | 12 | 37 | 5.3 | 78 |
| signum | 4.9 | – | 6.1 | – | 72 | 5.0 | – | 8.4 | – | 72 |
| neg | 6.6 | 12 | 11 | 6.5 | 66 | 6.8 | 12 | 36 | 6.4 | 66 |
| recip | 13 | 22 | 39 | 23 | 88 | 14 | 23 | 60 | 24 | 89 |
| pow(3) | 11 | – | 40 | 9.2 | 238 | 177 | – | 119 | 285 | 645 |
| clone | 8.6 | 11 | 11 | 5.7 | 68 | 8.7 | 11 | 35 | 5.6 | 67 |
| from(i64) | 4.3 | 6.1 | 10 | 4.1 | 87 | 4.3 | 5.4 | 14 | 4.1 | 94 |
| eq | 4.3 | 11 | 16 | 7.1 | 304 | 4.5 | 9.3 | 26 | 6.9 | 381 |
| hash | 34 | – | – | 4.8 | 596 | 18 | – | – | 4.9 | 4,423 |

Short rationals with primitive integers and `Number`s (FLINT: `fmpq_add_si`, `fmpq_sub_si`, `fmpq_mul_si`, `fmpq_add_fmpz`, `fmpq_mul_fmpz`, `fmpq_cmp_si`):

| Operation | 16 bits: ours | FLINT | 64 bits: ours | FLINT |
| --- | ---: | ---: | ---: | ---: |
| `&a + u32` | 14 | 28 | 79 | 97 |
| `&a - u32` | 14 | 27 | 82 | 100 |
| `&a * i64` | 29 | 54 | 296 | 303 |
| `a += i64` | 12 | 31 | 61 | 113 |
| `&a + &n` | 30 | 33 | 87 | 106 |
| `&a * &n` | 61 | 56 | 308 | 329 |
| `a.cmp(i32)` | 4.6 | 22 | 4.6 | 29 |
| `a == i64` | 3.8 | – | 3.8 | – |

Sums of many terms, in cycles a term: the harmonic numbers `H_n` and sums of `n` random fractions of parts below `2^16` and `2^32` of random signs, term by term by `+=` (GMP and FLINT: `mpq_add(h, h, t)`), and by `Iterator::sum`, against GMP's and FLINT's additions in the same binary-counter tree (malachite's `sum` is a tree as well; num-rational's, which is term by term, takes minutes from 10,000 terms and is left out); and products of `n` random fractions of parts below `2^16` the same two ways:

| Sum | `+=`: ours | GMP | FLINT | malachite | tree: ours | GMP | FLINT | malachite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `H_1000` | 601 | 486 | 501 | 576 | 337 | 544 | 377 | 557 |
| `H_10000` | 1,983 | 2,035 | 2,410 | 3,371 | 573 | 862 | 684 | 899 |
| `H_100000` | 14,129 | 16,810 | 16,883 | 31,197 | 1,200 | 1,682 | 1,504 | 1,757 |
| 10,000 terms of 16 bits | 1,861 | 1,704 | 2,212 | 3,121 | 361 | 637 | 439 | 652 |
| 10,000 terms of 32 bits | 8,367 | 8,190 | 9,096 | 17,460 | 1,208 | 1,681 | 1,482 | 1,791 |
| product of 10,000 of 16 bits | 2,187 | 1,868 | 2,859 | 3,283 | 461 | 783 | 555 | 740 |

- Parts in words are added, multiplied and compared in registers with nothing allocated. Below `2^16`, `&a + &b` takes 40 cycles against FLINT's 84, GMP's 183 and malachite's 187, and `&a * &b` 70 against FLINT's 95; in place 38 and 57 against 69 and 85; on the pools of 512 that the predictor learns, 24 to 37 cycles against FLINT's 37 to 40. Before #27 the same operations took 177 to 230 cycles, so they now take 0.19 to 0.33 of that time. At one limb, parts of up to 64 bits, the binary gcds of 64-bit words are most of the time: `+` takes 318 cycles against GMP's 351, `·` 308 against GMP's 538, and `a *= &b` with common factors, where the gcds end early, 138 against GMP's 465; 0.41 to 0.95 of the time before #27.
- From 2 limbs the arithmetic is on limbs, as in #16: `+` and `·` take 0.73 to 0.94 of the time of the fastest reference, and 0.97 to 1.05 of their time before #27 in the same process, within the spread of the rounds (up to 12 %).
- A short fraction with a long one is where GMP is faster: for `+` and `·` from `s+2` to `1+1024` it takes 0.59 to 0.90 of our time (but 1.03 for `·` at `1+1024`), and in place from `1+16` 0.80 to 0.91. A new rational there makes two vectors (about 70 cycles), copies the long part between bytes and limbs, and prepares the remainder by the short part (about 100 cycles): the path of #16 for limbs, which takes the time it took, but 0.43 to 0.78 of it up to two limbs.
- A comparison of rationals in words takes 7.5 cycles, without branches, against GMP's 23 to 48 and FLINT's 25 to 141; from 2 limbs 33 to 56 cycles at every length, by the bounds from the top bits, 0.8 to 1.0 of GMP's time at 2 limbs and a six- to ten-thousandth of it at 1024. There it takes a few cycles more than before #27 (28 to 59), the check of the two forms before the bounds and the layout of the code, which varies between builds by as much (0.88 to 1.30 of the time before in this build, 0.88 to 1.10 in another).
- `Rational::new` of short parts takes 122 cycles against FLINT's 89 and GMP's 93: it takes two `Number`s, whose vectors it releases when the result is in words, where GMP and FLINT reduce in a variable that stays; at one limb 277 against GMP's 264, and from 2 limbs 0.76 to 0.94 of the fastest reference's time.
- `floor`, `ceil` and `trunc` of short parts take 32 to 36 cycles against FLINT's 7 to 8 and malachite's 16 to 27: each gives a `Number`, a vector, whose allocation and release are 25 of them, where FLINT writes a word into a variable that exists and malachite's `Integer` keeps a word in place. At one limb `floor` takes 34 cycles against FLINT's 57 and malachite's 28. `floor` of 2 to 8 limbs takes 1.2 to 2.0 times GMP's time and `round` at 8 limbs 1.26 times FLINT's, the quotient's allocation being much of it there; from 64 limbs both take 0.63 to 0.94 of the fastest.
- The other operations of short rationals give words and allocate nothing: `fract`, `abs`, `signum`, `-x`, `recip`, `clone`, `==` and `From<i64>` take 4.3 to 14 cycles, 0.07 to 0.24 of their time before #27, malachite's being within a few cycles (`abs` 5.4 against 9.9, `clone` 5.7 against 8.6). `hash` takes 34 cycles at short parts and 18 at 64 bits against malachite's 4.8: a word is hashed as a `Number` of its value is, its sign, its length and its significant bytes, so that the hash is what it was before #27, a write of a variable length where malachite writes words. `pow(3)` of short parts takes 11 cycles, but of 64-bit parts 177 against FLINT's 119: a result of 192-bit parts is two vectors; it is also 1.09 to 1.15 of its time before #27, the branch on the parts' lengths taken before the same powers.
- With primitive integers and `Number`s, short rationals take 3.8 to 30 cycles against FLINT's 22 to 54, but `&a * &n` 61 against 56; at 64 bits 3.8 to 308 against 29 to 329.
- Sums of many terms by `+=` take 0.84 to 1.24 times GMP's time: each term is a pass of a remainder, an exact division and two products by one limb over the sum, as GMP's; at `H_1000` (23 limbs) the passes' fixed costs (the powers of `B` modulo the limb, 100 cycles, and the inverse) leave us at 1.24. By `Sum` the terms are added in words while the sum fits, then in the tree: 0.57 to 0.72 of GMP's time in the same tree, 0.80 to 0.89 of FLINT's, and 0.81 to 0.97 of the time before #27.

### Conversions

On an Apple M4 Max on 2026-11-12 (load average 2.9 to 9.3, falling through the first pass), one thread, `env -u MallocNanoZone`, fat LTO; the median of 21 rounds (three passes of seven) with fresh inputs and rotated order: ns or µs an operation, and in brackets the reference's time over ours (above 1: we are faster); the clock, measured in each round, in the title. GMP 6.3 (mpq_t), FLINT 3.6 (fmpq), malachite-q 0.14.0, num-rational 0.4.2. The sizes `16b`, `32b` and `64b` are parts of that many bits (down to 3 fewer), in words; the others have parts of more than 64 bits, or the digits given.

`to_f64`, parts of the bits given (numerator/denominator) (clock 4.19 GHz):

| Size | a | gmp | flint | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16b | 1.5 ns | 21.2 ns (14.19×) | 43.8 ns (29.39×) | 41.5 ns (27.83×) | 6.9 ns (4.63×) |
| 32b | 1.5 ns | 21.0 ns (14.07×) | 43.8 ns (29.29×) | 50.2 ns (33.56×) | 6.9 ns (4.59×) |
| 53b | 1.5 ns | 21.1 ns (14.19×) | 43.6 ns (29.25×) | 47.3 ns (31.69×) | 6.9 ns (4.61×) |
| 64b | 4.6 ns | 19.3 ns (4.18×) | 38.1 ns (8.23×) | 46.7 ns (10.10×) | 49.9 ns (10.79×) |
| 128b | 12.7 ns | 27.0 ns (2.12×) | 34.8 ns (2.74×) | 84.8 ns (6.66×) | 148.4 ns (11.66×) |
| 512b | 13.3 ns | 34.2 ns (2.58×) | 40.6 ns (3.06×) | 117.7 ns (8.86×) | 171.2 ns (12.88×) |
| 4096b | 13.9 ns | 53.7 ns (3.87×) | 57.2 ns (4.12×) | 206.1 ns (14.85×) | 419.7 ns (30.25×) |
| 65536b | 14.0 ns | 167.5 ns (12.01×) | 177.5 ns (12.72×) | 1.54 µs (110.58×) | 3.39 µs (242.65×) |
| 65536/64b | 3.8 ns | 11.2 ns (2.95×) | 22.5 ns (5.91×) | 5.8 ns (1.51×) | 265.4 ns (69.72×) |
| 64/65536b | 3.8 ns | 126.4 ns (33.15×) | 134.5 ns (35.26×) | 7.0 ns (1.83×) | 269.5 ns (70.67×) |

`to_f32`, parts of the bits given (clock 4.26 GHz):

| Size | a | malachite |
| --- | ---: | ---: |
| 16b | 1.5 ns | 22.7 ns (15.25×) |
| 64b | 3.9 ns | 46.9 ns (12.17×) |
| 4096b | 13.0 ns | 203.8 ns (15.65×) |

`Rational::try_from(f64)`: `word` floats in [2^-11, 2^12) (parts in words), `mid` in [1e-10, 1e10], `any` of any exponent; `f32`: `try_from(f32)` of any `f32` (clock 4.21 GHz):

| Size | a | gmp | gmp-alloc | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: |
| word | 2.0 ns | 10.7 ns (5.44×) | 37.4 ns (19.05×) | 6.4 ns (3.26×) | 293.9 ns (149.55×) |
| mid | 9.4 ns | 8.6 ns (0.92×) | 40.4 ns (4.31×) | 12.2 ns (1.30×) | 288.8 ns (30.85×) |
| any | 27.3 ns | 7.0 ns (0.26×) | 43.7 ns (1.60×) | 20.8 ns (0.76×) | 208.5 ns (7.62×) |
| f32 | 15.6 ns | — | — | 13.9 ns (0.89×) | 133.5 ns (8.56×) |

`FromStr` of a fraction in lowest terms: parts of the bits given, or digits of both parts (clock 4.30 GHz):

| Size | a | a-parts | gmp | gmp-canon | flint-canon | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16b | 16.5 ns | 37.4 ns (2.27×) | 45.0 ns (2.72×) | 63.1 ns (3.82×) | 93.5 ns (5.66×) | 32.2 ns (1.95×) | 239.3 ns (14.50×) |
| 32b | 30.1 ns | 41.8 ns (1.39×) | 49.7 ns (1.65×) | 82.2 ns (2.73×) | 115.0 ns (3.82×) | 63.2 ns (2.10×) | 370.6 ns (12.32×) |
| 64b | 75.4 ns | 39.5 ns (0.52×) | 64.9 ns (0.86×) | 127.6 ns (1.69×) | 164.8 ns (2.19×) | 119.9 ns (1.59×) | 638.9 ns (8.47×) |
| 10 | 18.1 ns | 37.2 ns (2.05×) | 43.2 ns (2.38×) | 62.4 ns (3.45×) | 90.3 ns (4.98×) | 37.3 ns (2.06×) | 244.2 ns (13.48×) |
| 100 | 360.3 ns | 75.2 ns (0.21×) | 120.6 ns (0.33×) | 393.2 ns (1.09×) | 462.2 ns (1.28×) | 505.8 ns (1.40×) | 1.74 µs (4.84×) |
| 1000 | 5.08 µs | 421.7 ns (0.08×) | 1.12 µs (0.22×) | 7.10 µs (1.40×) | 7.24 µs (1.43×) | 5.82 µs (1.15×) | 22.98 µs (4.53×) |
| 10000 | 89.58 µs | 17.34 µs (0.19×) | 21.99 µs (0.25×) | 124.29 µs (1.39×) | 126.05 µs (1.41×) | 125.83 µs (1.40×) | 1.13 ms (12.64×) |
| 100000 | 2.75 ms | 426.57 µs (0.16×) | 590.69 µs (0.21×) | 3.70 ms (1.35×) | 3.72 ms (1.35×) | 4.18 ms (1.52×) | 105.30 ms (38.31×) |
| 1000000 | 55.51 ms | 7.33 ms (0.13×) | 13.05 ms (0.24×) | 86.75 ms (1.56×) | 86.77 ms (1.56×) | 88.09 ms (1.59×) | — |

`FromStr` of a decimal `I.F`, digits in all (`10e`: with an exponent from -9 to 9) (clock 4.31 GHz):

| Size | a | malachite |
| --- | ---: | ---: |
| 4 | 11.9 ns | 60.7 ns (5.09×) |
| 10 | 18.8 ns | 88.0 ns (4.68×) |
| 19 | 26.8 ns | 121.2 ns (4.52×) |
| 10e | 20.5 ns | 92.3 ns (4.50×) |
| 100 | 172.2 ns | 692.8 ns (4.02×) |
| 1000 | 1.18 µs | 7.20 µs (6.10×) |
| 10000 | 28.57 µs | 170.86 µs (5.98×) |
| 100000 | 595.72 µs | 5.02 ms (8.43×) |
| 1000000 | 9.10 ms | 97.44 ms (10.70×) |

`to_string` of a fraction: parts of the bits given, or digits of both parts (clock 4.27 GHz):

| Size | a | a-radix | gmp | flint | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16b | 27.4 ns | 23.1 ns (0.85×) | 61.1 ns (2.23×) | 31.9 ns (1.17×) | 46.9 ns (1.71×) | 120.4 ns (4.40×) |
| 32b | 28.0 ns | 24.0 ns (0.86×) | 76.5 ns (2.74×) | 37.9 ns (1.36×) | 71.6 ns (2.56×) | 155.2 ns (5.55×) |
| 64b | 32.5 ns | 29.2 ns (0.90×) | 82.4 ns (2.53×) | 54.5 ns (1.68×) | 84.5 ns (2.60×) | 182.7 ns (5.62×) |
| 10 | 25.5 ns | 21.4 ns (0.84×) | 59.1 ns (2.32×) | 32.3 ns (1.27×) | 42.9 ns (1.68×) | 115.5 ns (4.53×) |
| 100 | 133.5 ns | 102.8 ns (0.77×) | 144.6 ns (1.08×) | 104.5 ns (0.78×) | 283.2 ns (2.12×) | 260.4 ns (1.95×) |
| 1000 | 1.01 µs | 962.0 ns (0.95×) | 2.54 µs (2.51×) | 1.64 µs (1.62×) | 3.76 µs (3.71×) | 5.46 µs (5.40×) |
| 10000 | 31.12 µs | 31.24 µs (1.00×) | 41.43 µs (1.33×) | 41.36 µs (1.33×) | 69.31 µs (2.23×) | 162.24 µs (5.21×) |
| 100000 | 851.33 µs | 851.98 µs (1.00×) | 1.24 ms (1.45×) | 1.31 ms (1.53×) | 1.69 ms (1.99×) | 10.02 ms (11.77×) |
| 1000000 | 15.04 ms | 15.03 ms (1.00×) | 29.59 ms (1.97×) | 19.11 ms (1.27×) | 31.08 ms (2.07×) | 774.39 ms (51.49×) |

`to_str_radix(2)`, parts of the bits given (clock 4.27 GHz):

| Size | a | gmp | flint | malachite |
| --- | ---: | ---: | ---: | ---: |
| 16b | 32.0 ns | 73.0 ns (2.28×) | 85.5 ns (2.67×) | 160.8 ns (5.03×) |
| 64b | 43.1 ns | 120.4 ns (2.80×) | 234.5 ns (5.45×) | 378.2 ns (8.78×) |
| 128b | 76.4 ns | 199.7 ns (2.61×) | 195.4 ns (2.56×) | 911.9 ns (11.94×) |
| 4096b | 465.4 ns | 4.32 µs (9.28×) | 4.28 µs (9.19×) | 20.14 µs (43.28×) |

`to_str_radix(7)`, parts of the bits given (clock 4.26 GHz):

| Size | a | gmp | flint | malachite |
| --- | ---: | ---: | ---: | ---: |
| 16b | 36.5 ns | 71.5 ns (1.96×) | 63.7 ns (1.74×) | 83.7 ns (2.29×) |
| 64b | 59.6 ns | 141.4 ns (2.37×) | 147.8 ns (2.48×) | 238.5 ns (4.00×) |
| 128b | 256.2 ns | 164.1 ns (0.64×) | 187.3 ns (0.73×) | 344.6 ns (1.35×) |
| 4096b | 6.01 µs | 7.73 µs (1.29×) | 7.72 µs (1.28×) | 12.82 µs (2.13×) |

`to_str_radix(16)`, parts of the bits given (clock 4.25 GHz):

| Size | a | gmp | flint | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16b | 25.9 ns | 55.9 ns (2.16×) | 37.3 ns (1.44×) | 65.0 ns (2.51×) | 92.2 ns (3.56×) |
| 64b | 35.0 ns | 55.0 ns (1.57×) | 68.6 ns (1.96×) | 156.8 ns (4.48×) | 147.2 ns (4.20×) |
| 128b | 60.2 ns | 72.3 ns (1.20×) | 71.4 ns (1.19×) | 322.8 ns (5.36×) | 166.6 ns (2.77×) |
| 4096b | 303.3 ns | 1.17 µs (3.86×) | 1.18 µs (3.90×) | 5.42 µs (17.86×) | 862.6 ns (2.84×) |

`from_str_radix(_, 2)`, parts of the bits given (clock 4.32 GHz):

| Size | a | gmp-canon | flint-canon | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16b | 17.2 ns | 85.7 ns (4.98×) | 121.6 ns (7.07×) | 47.7 ns (2.77×) | 267.4 ns (15.55×) |
| 64b | 85.2 ns | 232.0 ns (2.72×) | 273.8 ns (3.21×) | 164.1 ns (1.93×) | 738.3 ns (8.66×) |
| 128b | 298.6 ns | 456.7 ns (1.53×) | 517.4 ns (1.73×) | 960.2 ns (3.22×) | 1.50 µs (5.02×) |
| 4096b | 16.62 µs | 25.79 µs (1.55×) | 25.95 µs (1.56×) | 35.60 µs (2.14×) | 88.48 µs (5.32×) |

`from_str_radix(_, 7)`, parts of the bits given (clock 4.31 GHz):

| Size | a | gmp-canon | flint-canon | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16b | 16.1 ns | 65.9 ns (4.09×) | 95.8 ns (5.95×) | 105.6 ns (6.55×) | 244.5 ns (15.18×) |
| 64b | 92.7 ns | 140.3 ns (1.51×) | 180.2 ns (1.94×) | 252.6 ns (2.72×) | 661.2 ns (7.13×) |
| 128b | 256.7 ns | 262.1 ns (1.02×) | 314.9 ns (1.23×) | 639.4 ns (2.49×) | 1.36 µs (5.31×) |
| 4096b | 15.35 µs | 20.63 µs (1.34×) | 20.89 µs (1.36×) | 23.05 µs (1.50×) | 86.27 µs (5.62×) |

`from_str_radix(_, 16)`, parts of the bits given (clock 4.32 GHz):

| Size | a | gmp-canon | flint-canon | malachite | num |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16b | 14.4 ns | 59.6 ns (4.14×) | 89.5 ns (6.21×) | 35.8 ns (2.49×) | 246.6 ns (17.12×) |
| 64b | 76.2 ns | 118.9 ns (1.56×) | 161.0 ns (2.11×) | 101.7 ns (1.33×) | 677.8 ns (8.89×) |
| 128b | 181.1 ns | 236.3 ns (1.30×) | 295.4 ns (1.63×) | 420.4 ns (2.32×) | 1.35 µs (7.47×) |
| 4096b | 13.28 µs | 18.95 µs (1.43×) | 19.14 µs (1.44×) | 16.32 µs (1.23×) | 86.58 µs (6.52×) |

`Number::try_from(&r)` of an integer of the bits given (clock 4.19 GHz):

| Size | a | malachite |
| --- | ---: | ---: |
| 16b | 7.4 ns | 1.1 ns (0.15×) |
| 64b | 7.5 ns | 1.1 ns (0.15×) |
| 128b | 9.4 ns | 11.9 ns (1.27×) |
| 4096b | 21.0 ns | 23.9 ns (1.14×) |

`i128::try_from(&r)` of an integer of the bits given (clock 4.16 GHz):

| Size | a | malachite |
| --- | ---: | ---: |
| 16b | 0.6 ns | 1.3 ns (2.38×) |
| 64b | 0.6 ns | 1.3 ns (2.38×) |
| 128b | 2.6 ns | 1.6 ns (0.62×) |

`Rational::from(x)` of an `i64` (16 and 63 bits) or an `i128` (127 bits) (clock 4.09 GHz):

| Size | a | malachite |
| --- | ---: | ---: |
| 16b | 1.3 ns | 0.8 ns (0.65×) |
| 64b | 1.3 ns | 0.8 ns (0.65×) |
| 128b | 14.4 ns | 7.6 ns (0.53×) |

GMP's `mpq_get_d` and FLINT's `fmpq_get_d` (which calls it) truncate towards zero: of 500 random fractions of 16 to 512 bits they gave another float than the nearest for 35 to 52 % of them; malachite-q's `f64::rounding_from(_, Nearest)` and num-rational's `to_f64` agreed with ours on every one, at every size. num-rational's `to_f32` is its `to_f64` rounded again, so it is not compared. GMP's `mpq_set_d` writes into an `mpq_t` that exists already; `gmp-alloc` makes a new one each time, as a new rational is. Reading text, `gmp` is `mpq_set_str` alone, which leaves the fraction as written, `gmp-canon` and `flint-canon` the same with the reduction, as every `Rational` is reduced; `a-parts` reads the two parts as `Number`s without the gcd, and shows that the gcd is 79 to 92 % of a fraction's reading from 100 digits on. `a-radix` is `to_str_radix(10)`.

Parts in words take no allocation but the text a conversion returns: `to_f64` of 64-bit parts takes 4.6 ns, a quarter of GMP's truncated 19.3 ns, and of parts below `2^53` 1.5 ns; a float in words becomes a rational in 2.0 ns, a third of malachite's 6.4 ns and a fifth of GMP's 10.7 ns into an existing `mpq_t`; fractions of parts below `2^16` are read in 16.5 ns (malachite 32.2 ns) and written in 27.4 ns (FLINT 31.9 ns), and in radices 2, 7 and 16 read in 14 to 17 ns and written in 26 to 37 ns; decimals of up to 19 digits are read in 12 to 27 ns, a fifth of malachite's time. Against the crate before #27 these take 0.09 to 0.82 of the time, the long paths 0.97 to 1.03 of it but for a few rows measured at 0.88 to 0.95 here and at 0.99 to 1.06 when timed alone (`to_f64` of lopsided parts, 0.3 ns, and decimals of 100 to 1000 digits). Where a reference is faster:

- A float whose parts pass 64 bits (a third of the `mid` floats, most of `any`): two vectors, the numerator and the denominator, take about 20 ns with their frees, where GMP writes into the limbs its `mpq_t` has (7.0 and 8.6 ns) and malachite keeps a part of one limb in place and allocates one (20.8 ns for `any`, ours 27.3 ns); `try_from(f32)` the same, 15.6 ns against malachite's 13.9 ns.
- Reading a fraction of 64-bit parts, `mpq_set_str` without the reduction takes 64.9 ns against our 75.4 ns, of which about 45 ns are the gcd of the two words that the reduction needs (and `mpq_canonicalize` 127.6 ns).
- `Number::try_from(&r)` of an integer in words makes the vector of a `Number`, 7.4 ns, where malachite's `Integer` holds a small value in place (1.1 ns); `Rational::from` of an `i64` takes 1.3 ns against malachite's 0.8, and of an `i128` past 64 bits makes two vectors (14.4 ns, malachite one, 7.6 ns); `i128::try_from` of a 128-bit integer reads its bytes (2.6 ns, malachite 1.6 ns).
- `to_str_radix(7)` of 128-bit parts writes them as numbers do (256 ns, GMP 164 ns).
- Writing a fraction of 100 digits, FLINT takes 104.5 ns against our 133.5 ns by `to_string` and 102.8 ns by `to_str_radix(10)`: the difference is the formatting machinery of `to_string`, which copies the text once more.

### Continued fractions and approximations

Measured on an Apple M4 Max on 2026-11-12, with nothing else running (load average 1.9 to 4.4), single-threaded, in cycles per call at the clock measured before each pass (4.5 GHz), or in time where marked, against FLINT 3.6 (`fmpq_get_cfrac` with `fmpq_cfrac_bound` and the vector's allocation, `fmpq_set_cfrac`, `fmpq_simplest_between`, `fmpq_reconstruct_fmpz` and `fmpq_reconstruct_fmpz_2`), PARI 2.19 (`contfrac`, `bestappr(x, D)`, `bestappr(Mod(r, m))`) and malachite-q 0.14.0 (`continued_fraction` collected, `from_continued_fraction_ref`, `convergents` collected, `approximate`, `simplest_rational_in_closed_interval`, `reconstruct_ref`, `reconstruct_with_bounds_ref`), and this crate before issue #27 (0.2.34, whose rationals always held two `Number`s) in the same process; num-rational 0.4 has none of these. Random parts of `n` limbs, or of 16 bits, positive; every process with `MallocNanoZone` unset, the harness with fat LTO and one codegen unit; three passes of seven rounds, each cell the median of the 21, each round on fresh inputs in an order that turns with the round. PARI and malachite, whose continued fractions and approximations are quadratic, are not measured past 4096 limbs, nor FLINT's `fmpq_simplest_between` past 16,384.

`continued_fraction`, parts of n limbs (or of 16 bits):

| n | `a` | before #27 | FLINT | PARI | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 bits | 535 | 610 | 1,382 | 1,075 | 660 | 0.81 | 0.88 |
| 1 | 1,477 | 1,650 | 4,145 | 3,668 | 2,224 | 0.66 | 0.90 |
| 2 | 3,660 | 3,828 | 12,620 | 9,579 | 10,494 | 0.38 | 0.96 |
| 4 | 9,863 | 9,855 | 28,730 | 21,425 | 33,329 | 0.46 | 1.00 |
| 16 | 37,411 | 37,409 | 35,777 | 96,738 | 186,300 | 1.05 | 1.00 |
| 64 | 179,418 | 182,252 | 129,629 | 558,817 | 1.05 M | 1.38 | 0.98 |
| 256 | 820,497 | 823,016 | 630,327 | 5.26 M | 8.71 M | 1.30 | 1.00 |
| 1024 | 4.79 M | 4.78 M | 5.35 M | 16.2 ms | 23.7 ms | 0.90 | 1.00 |
| 4096 | 27.8 M | 27.9 M | 37.1 M | 244 ms | 351 ms | 0.75 | 1.00 |
| 16384 | 34.4 ms | 34.5 ms | 50.9 ms | – | – | 0.68 | 1.00 |
| 100000 | 295 ms | 297 ms | 456 ms | – | – | 0.65 | 0.99 |

`from_continued_fraction`, the terms of parts of n limbs:

| n | `a` | before #27 | FLINT | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 bits | 143 | 510 | 286 | 490 | 0.50 | 0.28 |
| 1 | 392 | 826 | 1,311 | 1,954 | 0.30 | 0.47 |
| 2 | 2,249 | 2,423 | 3,064 | 3,797 | 0.73 | 0.93 |
| 4 | 5,186 | 5,381 | 6,571 | 7,946 | 0.79 | 0.96 |
| 16 | 21,705 | 22,524 | 28,176 | 40,260 | 0.77 | 0.96 |
| 64 | 104,160 | 107,771 | 133,733 | 357,969 | 0.78 | 0.97 |
| 256 | 555,288 | 569,057 | 677,580 | 5.46 M | 0.82 | 0.98 |
| 1024 | 3.28 M | 3.31 M | 3.98 M | 19.7 ms | 0.83 | 0.99 |
| 4096 | 18 M | 17.9 M | 25.4 M | 316 ms | 0.71 | 1.00 |
| 16384 | 21.1 ms | 21.5 ms | 32.2 ms | – | 0.65 | 0.98 |
| 100000 | 172 ms | 177 ms | 248 ms | – | 0.69 | 0.97 |

`convergents`, parts of n limbs:

| n | `a` | before #27 | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 bits | 333 | 1,085 | 1,429 | 0.23 | 0.31 |
| 1 | 829 | 3,815 | 4,809 | 0.17 | 0.22 |
| 2 | 10,389 | 12,271 | 17,273 | 0.60 | 0.85 |
| 4 | 28,471 | 43,424 | 50,777 | 0.56 | 0.66 |
| 16 | 160,518 | 270,456 | 284,964 | 0.56 | 0.59 |
| 64 | 986,836 | 1.85 M | 1.81 M | 0.54 | 0.53 |
| 256 | 8.78 M | 14.7 M | 17.9 M | 0.49 | 0.60 |

`limit_denominator`, parts of n limbs, a bound of n/2:

| n | `a` | before #27 | PARI | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 bits | 51 | 234 | 1,518 | 981 | 0.05 | 0.22 |
| 1 | 259 | 458 | 4,346 | 2,853 | 0.09 | 0.57 |
| 2 | 1,140 | 1,503 | 10,734 | 11,552 | 0.11 | 0.76 |
| 4 | 4,699 | 4,774 | 20,621 | 33,019 | 0.23 | 0.98 |
| 16 | 12,266 | 12,041 | 104,549 | 161,143 | 0.12 | 1.02 |
| 64 | 50,676 | 50,177 | 677,216 | 919,561 | 0.07 | 1.01 |
| 256 | 340,284 | 340,571 | 7.67 M | 8.88 M | 0.04 | 1.00 |
| 1024 | 2.65 M | 2.65 M | 30.3 ms | 24.3 ms | 0.02 | 1.00 |
| 4096 | 17 M | 17 M | 380 ms | 376 ms | 0.01 | 1.00 |
| 16384 | 20.6 ms | 20.5 ms | – | – | – | 1.00 |
| 100000 | 183 ms | 183 ms | – | – | – | 1.00 |

`limit_denominator`, parts of n limbs, a bound of 60 bits:

| n | `a` | before #27 | PARI | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 504 | 638 | 7,294 | 4,345 | 0.12 | 0.79 |
| 2 | 1,099 | 1,470 | 10,264 | 11,055 | 0.11 | 0.75 |
| 4 | 3,758 | 3,753 | 9,816 | 14,922 | 0.38 | 1.00 |
| 16 | 4,457 | 4,440 | 11,240 | 17,071 | 0.40 | 1.00 |
| 64 | 6,507 | 6,412 | 19,276 | 28,120 | 0.34 | 1.01 |
| 256 | 12,360 | 12,373 | 63,563 | 72,304 | 0.19 | 1.00 |
| 1024 | 36,467 | 35,211 | 205,548 | 318,887 | 0.18 | 1.04 |
| 4096 | 125,898 | 126,655 | 728,308 | 829,357 | 0.17 | 0.99 |
| 16384 | 438,557 | 436,679 | 2.78 M | 3.24 M | 0.16 | 1.00 |
| 100000 | 2.74 M | 2.62 M | 17.7 M | 19.4 M | 0.15 | 1.05 |

`simplest_between`, two random ends of n limbs:

| n | `a` | before #27 | FLINT | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 bits | 33 | 304 | 243 | 545 | 0.13 | 0.11 |
| 1 | 34 | 296 | 825 | 810 | 0.04 | 0.12 |
| 2 | 193 | 384 | 1,063 | 1,739 | 0.18 | 0.50 |
| 4 | 1,207 | 1,361 | 1,075 | 2,056 | 1.12 | 0.89 |
| 16 | 1,548 | 1,692 | 2,603 | 6,408 | 0.59 | 0.92 |
| 64 | 2,487 | 2,612 | 10,506 | 20,062 | 0.24 | 0.95 |
| 256 | 4,926 | 5,164 | 68,280 | 108,941 | 0.07 | 0.95 |
| 1024 | 16,842 | 16,796 | 618,025 | 312,053 | 0.05 | 1.00 |
| 4096 | 25,464 | 25,280 | 2.5 M | 55,074 | 0.46 | 1.01 |
| 16384 | 260,513 | 258,934 | 11.7 M | 2.23 M | 0.12 | 1.01 |
| 100000 | 1.65 M | 1.65 M | 15.9 ms | 13.5 M | 0.12 | 1.00 |

`simplest_between`, x and x + 1/q, q of 1.5 n limbs:

| n | `a` | before #27 | FLINT | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 bits | 101 | 422 | 862 | 1,412 | 0.12 | 0.24 |
| 1 | 5,716 | 5,621 | 6,806 | 11,312 | 0.84 | 1.02 |
| 2 | 7,331 | 7,361 | 21,013 | 30,611 | 0.35 | 1.00 |
| 4 | 9,383 | 9,270 | 8,756 | 14,978 | 1.07 | 1.01 |
| 16 | 27,562 | 27,914 | 61,872 | 121,709 | 0.45 | 0.99 |
| 64 | 141,942 | 141,476 | 238,599 | 426,219 | 0.59 | 1.00 |
| 256 | 1.27 M | 1.27 M | 1.72 M | 3.55 M | 0.74 | 1.00 |
| 1024 | 13.5 M | 13.7 M | 16.8 M | 34.7 M | 0.81 | 0.99 |
| 4096 | 18.6 ms | 18.6 ms | 42.9 ms | 88 ms | 0.43 | 1.00 |
| 16384 | 105 ms | 105 ms | 548 ms | – | 0.19 | 1.00 |
| 100000 | 917 ms | 918 ms | – | – | – | 1.00 |

`reconstruct`, a modulus of n limbs, p/q of up to n/2:

| n | `a` | before #27 | FLINT | PARI | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 bits | 60 | 185 | 113 | 1,395 | 83 | 0.73 | 0.32 |
| 1 | 145 | 300 | 534 | 2,443 | 241 | 0.60 | 0.49 |
| 2 | 1,084 | 1,500 | 1,186 | 5,070 | 1,254 | 0.91 | 0.72 |
| 4 | 2,461 | 2,451 | 1,935 | 4,242 | 3,446 | 1.27 | 1.00 |
| 16 | 7,289 | 7,348 | 8,595 | 10,990 | 11,253 | 0.85 | 0.99 |
| 64 | 30,506 | 31,156 | 37,883 | 46,602 | 49,850 | 0.81 | 0.98 |
| 256 | 184,450 | 179,328 | 253,343 | 293,913 | 390,318 | 0.73 | 1.03 |
| 1024 | 1.28 M | 1.24 M | 2.46 M | 1.81 M | 3.25 M | 0.70 | 1.03 |
| 4096 | 7.34 M | 7.44 M | 16.6 M | 11.5 M | 19.1 M | 0.64 | 0.99 |
| 16384 | 41.2 M | 40.9 M | 20.5 ms | 16.3 ms | 22.1 ms | 0.56 | 1.01 |
| 100000 | 109 ms | 109 ms | 213 ms | 198 ms | 235 ms | 0.55 | 1.00 |

`reconstruct_with_bounds`, a modulus of n limbs, bounds of 0.15 and 0.8 of it:

| n | `a` | before #27 | FLINT | malachite | `a` over the fastest | `a` over before #27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 bits | 92 | 217 | 105 | 75 | 1.22 | 0.42 |
| 1 | 210 | 375 | 517 | 297 | 0.71 | 0.56 |
| 2 | 1,021 | 1,465 | 1,353 | 1,430 | 0.75 | 0.70 |
| 4 | 2,408 | 2,432 | 2,282 | 3,703 | 1.06 | 0.99 |
| 16 | 6,778 | 6,839 | 9,592 | 12,645 | 0.71 | 0.99 |
| 64 | 32,012 | 33,216 | 43,195 | 64,065 | 0.74 | 0.96 |
| 256 | 212,769 | 212,660 | 322,699 | 521,658 | 0.66 | 1.00 |
| 1024 | 1.66 M | 1.66 M | 3.11 M | 4.49 M | 0.53 | 1.00 |
| 4096 | 11 M | 10.7 M | 20.1 M | 25.9 M | 0.55 | 1.02 |
| 16384 | 14.7 ms | 14.6 ms | 29.4 ms | 33.1 ms | 0.50 | 1.01 |
| 100000 | 102 ms | 103 ms | 194 ms | 217 ms | 0.53 | 0.99 |

Parts of up to 64 bits are read from the words of the rational and the results written as words: against this crate before #27, continued fractions take 0.88 to 0.90 of the time at 16 and 64 bits (the rest is the terms' vectors), `from_continued_fraction` 0.28 to 0.47, `convergents` 0.22 to 0.31, `limit_denominator` 0.22 to 0.79, `simplest_between` 0.11 to 0.24 and the reconstructions 0.32 to 0.56. Parts of two limbs, in registers with the quotients up to 3 by subtractions, take 0.50 to 0.96 of the time before; `convergents` of long parts, on limbs in place, 0.53 to 0.66 from 4 to 256 limbs. The rest, from two limbs up and for a short end with a long one (the close ends of one limb, 1.02), takes 0.95 to 1.05 of the time before, the noise of the runs. Against the fastest reference that has the function: continued fractions 0.38 to 0.81 up to 4 limbs and 0.65 to 0.90 from 1024 limbs, but 1.05 to 1.38 times FLINT's from 16 to 256 limbs, where about half of the time is the allocation and release of a vector for each term, 33 cycles each, and FLINT keeps small terms in its `fmpz` words; `from_continued_fraction` 0.30 to 0.83 of FLINT's or malachite's; `convergents` 0.17 to 0.60 of malachite's; `limit_denominator` 0.01 to 0.40 of PARI's or malachite's, quadratic both; `simplest_between` 0.04 to 0.84, but 1.12 times FLINT's for two ends of 4 limbs and 1.07 for close ends of 4 limbs; the reconstructions 0.50 to 0.91, but 1.27 and 1.06 times FLINT's at 4 limbs, and 1.22 times malachite's for `reconstruct_with_bounds` of a 16-bit modulus (92 cycles against 75), malachite's words needing no allocation for the result.

The greatest common divisors with the sink threaded through the half gcd, against the frozen core, in one process (the ratio of the medians, above 1 slower), random numbers of the given limbs:

| limbs | gcd | gcd_ext |
| --- | ---: | ---: |
| 3 | 0.979 | 0.987 |
| 16 | 0.951 | 1.028 |
| 100 | 1.003 | 1.005 |
| 1000 | 0.994 | 0.996 |
| 3000 | 1.003 | 0.998 |

### Bernoulli and harmonic numbers

Measured on an Apple M4 Max on 2026-11-12, with nothing else running (load average 3.6 to 4.9), single-threaded, every process with `MallocNanoZone` unset, the harness built with fat LTO and one codegen unit, in cycles per call (milliseconds from `10^5` cycles), the cycles from the clock measured just before each round: two passes of interleaved rounds, the contenders' order rotating each round, each cell timed in batches of calls of at least 30 ms over four indices in turn (`n` to `n + 6` by 2 for the Bernoulli numbers, `n` to `n + 3` for the harmonic numbers, `n` alone for the vectors), the median of 9 rounds a pass (3 for `B_(10^5)`, `H_(10^6)` and the vector to `10^4`, 1 for `B_(10^6)`), every value checked against FLINT's first; against FLINT 3.6 and PARI 2.19 called from the same binary, which write into results they keep, malachite-q 0.14 (harmonic numbers; it has no Bernoulli numbers, and num-rational 0.4 and GMP neither) and this crate before issue #27 (0.2.34). In brackets, the other's time over this crate's. `B_40`, `B_60` and `H_60` come from the tables in numbers, `B_102` is the first by the formula and `H_125` the first by binary splitting.

`Rational::bernoulli(n)` against FLINT's `_bernoulli_fmpq_ui` (`bernoulli_fmpq_ui` with its cache), PARI's `bernfrac` (with and without its cache), and this crate before issue #27 (cycles, or ms; in brackets the other's time over this crate's):

| n | `a` | before #27 | FLINT | FLINT, cached | FLINT ζ only | PARI | PARI, no cache |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 10 | 72 (7.34) | 17 (1.73) | 19 (1.93) |  | 9 (0.95) | 0.134 ms (55103.26) |
| 40 | 73 | 73 (1.00) | 2,170 (29.90) | 2,159 (29.75) |  | 9 (0.12) | 0.133 ms (7265.88) |
| 60 | 88 | 90 (1.03) | 3,026 (34.55) | 3,007 (34.33) |  | 9 (0.10) | 0.134 ms (6096.48) |
| 100 | 3,940 | 3,929 (1.00) | 4,393 (1.12) | 4,398 (1.12) |  | 9 (0.00) | 0.133 ms (133.80) |
| 102 | 5,360 | 5,348 (1.00) | 4,472 (0.83) | 4,428 (0.83) |  | 9 (0.00) | 0.133 ms (97.47) |
| 300 | 17,231 | 17,373 (1.01) | 17,161 (1.00) | 17,209 (1.00) |  | 21,387 (1.24) | 0.139 ms (32.07) |
| 1,000 | 0.0387 ms | 0.0387 ms (1.00) | 0.0377 ms (0.98) | 0.0377 ms (0.98) |  | 0.0474 ms (1.22) | 0.181 ms (4.68) |
| 3,000 | 0.361 ms | 0.363 ms (1.01) | 0.393 ms (1.09) | 0.391 ms (1.08) |  | 0.505 ms (1.40) | 0.639 ms (1.77) |
| 10^4 | 3.43 ms | 3.42 ms (1.00) | 5.23 ms (1.52) | 5.17 ms (1.51) |  | 6.59 ms (1.92) | 6.62 ms (1.93) |
| 10^5 | 138 ms | 139 ms (1.00) | 276 ms (2.00) |  | 688 ms (4.97) | 785 ms (5.68) |  |
| 10^6 | 7.92 s | 7.94 s (1.00) | 13.1 s (1.65) |  |  | 81.6 s (10.30) |  |

`Rational::harmonic(n)` against FLINT's `_fmpq_harmonic_ui`, PARI's `harmonic` and malachite-q's `Rational::harmonic_number`, and this crate before issue #27 (cycles, or ms; in brackets the other's time over this crate's):

| n | `a` | before #27 | FLINT | PARI | malachite-q |
| --- | --- | --- | --- | --- | --- |
| 10 | 9 | 72 (7.97) | 8 (0.86) | 1,403 (155.46) | 46 (5.14) |
| 60 | 72 | 80 (1.11) | 1,424 (19.67) | 10,993 (151.87) | 1,931 (26.67) |
| 89 | 471 | 693 (1.47) | 2,209 (4.69) | 19,152 (40.65) | 2,840 (6.03) |
| 100 | 1,114 | 1,190 (1.07) | 2,910 (2.61) | 21,660 (19.45) | 3,428 (3.08) |
| 125 | 3,118 | 3,197 (1.03) | 3,437 (1.10) | 27,937 (8.96) | 3,972 (1.27) |
| 300 | 6,281 | 6,474 (1.03) | 8,346 (1.33) | 82,970 (13.21) | 10,254 (1.63) |
| 1,000 | 20,572 | 21,209 (1.03) | 41,532 (2.02) | 0.0922 ms (17.89) | 53,610 (2.61) |
| 10^4 | 0.189 ms | 0.189 ms (1.00) | 0.434 ms (2.30) | 2.14 ms (11.33) | 0.591 ms (3.13) |
| 10^5 | 4.48 ms | 4.52 ms (1.01) | 14.7 ms (3.27) | 42.7 ms (9.53) | 15.9 ms (3.56) |
| 10^6 | 84.4 ms | 84.2 ms (1.00) | 278 ms (3.30) | 969 ms (11.48) | 277 ms (3.28) |

`Rational::bernoulli_numbers(n)` against FLINT's `bernoulli_fmpq_vec_no_cache` and `arith_bernoulli_number_vec`, PARI's `bernvec(n/2)` (with and without its cache), and this crate before issue #27 (cycles, or ms; in brackets the other's time over this crate's):

| n | `a` | before #27 | FLINT | FLINT `arith` | PARI | PARI, no cache |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 183 | 833 (4.56) | 302 (1.65) | 292 (1.60) | 26 (0.14) | 0.133 ms (2947.69) |
| 30 | 390 | 2,232 (5.72) | 806 (2.07) | 801 (2.05) | 38 (0.10) | 0.133 ms (1357.62) |
| 100 | 3,478 | 7,390 (2.12) | 48,413 (13.92) | 48,315 (13.89) | 74 (0.02) | 0.135 ms (151.44) |
| 150 | 0.0266 ms | 0.0284 ms (1.07) | 0.0376 ms (1.41) | 0.0319 ms (1.20) | 101 (0.00) | 0.134 ms (5.02) |
| 300 | 0.0944 ms | 0.0989 ms (1.05) | 0.125 ms (1.32) | 0.162 ms (1.72) | 211 (0.00) | 0.177 ms (1.88) |
| 1,000 | 1.38 ms | 1.41 ms (1.02) | 1.68 ms (1.22) | 6.89 ms (5.01) | 571 (0.00) | 1.91 ms (1.39) |
| 10^4 | 414 ms | 414 ms (1.00) | 548 ms (1.32) |  |  | 600 ms (1.45) |

- `B_n` from `n = 3000` faster than FLINT's, 1.09 times at 3000, 1.52 at `10^4`, 2.00 at `10^5` and 1.65 at `10^6` (7.92 s against 13.1 s), by the residues modulo primes; FLINT's formula alone (`_bernoulli_fmpq_ui_zeta`) takes 4.97 times this crate's time at `10^5`. At 300 and 1000 FLINT's takes 0.98 to 1.00 of this crate's time, the formula's arithmetic at these lengths, and `B_102`, the first by the formula, 0.83 (4,472 cycles against 5,360), as before #27. PARI keeps the numbers it has computed and returns the small ones from its cache in about 10 cycles; with the cache dropped before each call, each call rebuilds it, 0.13 ms or more.
- A table lookup in words, `B_10` or `H_10`, writes the two words of the result in the caller, 9 to 10 cycles against 72 before #27 (the two vectors of a `Number` each), FLINT's 8 to 17 and PARI's 9 for a pointer into its cache. In numbers (`B_40`, `B_60`, `H_60`), 72 to 88 cycles, as before or less (`H_60` 72 against 80, its parts written by one store each), against FLINT's 1,424 to 3,026. The vector of the short ones is written as words too: `B_0, ..., B_10` in 183 cycles against 833 before and FLINT's 302, to `B_100` 3,478 against 7,390 and 48,413. The long computations take 0.93 to 1.00 of their time before.
- `H_n` from `n = 89`, just past the table, faster than FLINT's, 4.7 times at 89 (471 cycles against 2,209), 1.1 at 125, 1.3 at 300, 2.0 to 2.3 from 1000 to `10^4` and 3.3 at `10^5` and `10^6`, by the reduction without a greatest common divisor and the pairs added in place; malachite-q, which sums the odd parts as FLINT does, takes 1.3 to 6.0 times this crate's time from 89 up.
- The vector from 150 faster than FLINT's, which computes it the same way, 1.2 to 1.4 times from 150 to `10^4`, by the powers' pass that updates and adds in one, in limbs, and the denominators' bound kept by index. PARI's `bernvec` copies its cache once it is filled.

### Matrices

On an Apple M4 Max on 2026-11-12, with nothing else running (load average 2.2 to 5.9), one thread, `env -u MallocNanoZone`, fat LTO, the median of 5 rounds with fresh random inputs each, interleaved in one process, in µs (or ms and s), and each one's time over this crate's in brackets; the clock read 4.2 GHz in the rounds (3.3 to 4.5). "`a` 0.2.34" is this crate before rationals held small parts in words, in the same process. FLINT is 3.6, its defaults (`fmpq_mat_det`, `fmpq_mat_solve`, `fmpq_mat_inv`, `fmpq_mat_mul`) and its Bareiss alone (the rows cleared as `fmpq_mat` does, then `fmpz_mat_det_bareiss` or `fmpz_mat_solve_fflu`, and one division per element). PARI is 2.19 (`det`, `RgM_solve`, `RgM_inv`, `gmul`). "`a`, Bareiss alone" is this crate with Dixon's lifting switched off. num-rational 0.4.2 and malachite-q 0.14.0 have no matrices, so they run a Gaussian elimination of fractions (the first non-zero pivot, back substitution, the inverse as the solution of `A·X = I`) and the product as sums of products. The tables for `Matrix<Number>` are on the [matrices](matrix.md#exactly) page.

`&a * &b` of `Matrix<Rational>`, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, sums of products of rationals | FLINT | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.07 µs | 0.30 µs (4.26) | 0.09 µs (1.33) | 0.13 µs (1.79) | 0.07 µs (0.96) | 3.87 µs (54.78) | 0.26 µs (3.68) |
| 3 | 0.11 µs | 0.54 µs (4.88) | 0.31 µs (2.79) | 0.22 µs (2.01) | 0.27 µs (2.43) | 12.5 µs (112.15) | 0.89 µs (7.95) |
| 4 | 0.16 µs | 0.92 µs (5.81) | 0.72 µs (4.55) | 0.39 µs (2.48) | 0.63 µs (4.02) | 30.5 µs (193.41) | 2.22 µs (14.05) |
| 5 | 0.22 µs | 1.38 µs (6.21) | 1.40 µs (6.29) | 0.53 µs (2.40) | 1.22 µs (5.49) | 59.4 µs (266.68) | 4.42 µs (19.84) |
| 6 | 0.30 µs | 2.04 µs (6.73) | 2.37 µs (7.84) | 0.79 µs (2.60) | 2.27 µs (7.50) | 102.5 µs (338.41) | 8.00 µs (26.40) |
| 8 | 0.52 µs | 3.63 µs (6.98) | 5.57 µs (10.72) | 1.35 µs (2.59) | 5.59 µs (10.75) | 243.4 µs (468.24) | 20.1 µs (38.60) |
| 16 | 2.42 µs | 15.8 µs (6.52) | 45.6 µs (18.81) | 5.55 µs (2.29) | 48.9 µs (20.20) | 1.97 ms (813.12) | 196.4 µs (81.05) |
| 32 | 12.9 µs | 74.1 µs (5.73) | 354.2 µs (27.38) | 26.1 µs (2.02) | 385.1 µs (29.77) | 15.4 ms (1190.47) | 1.63 ms (126.00) |
| 64 | 81.7 µs | 350.6 µs (4.29) | 2.97 ms (36.32) | 137.9 µs (1.69) | 2.97 ms (36.39) | 126 ms (1537.48) | 14 ms (171.13) |
| 128 | 559.5 µs | 1.73 ms (3.08) | 25.6 ms (45.70) | 941.2 µs (1.68) | 6.68 ms (11.95) | 1 s (1787.34) | 108 ms (192.64) |

`&a * &b` of `Matrix<Rational>`, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, sums of products of rationals | FLINT | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.11 µs | 0.31 µs (2.81) | 0.14 µs (1.22) | 0.21 µs (1.93) | 0.83 µs (7.54) | 5.30 µs (47.91) | 0.33 µs (2.96) |
| 3 | 0.24 µs | 0.85 µs (3.53) | 0.47 µs (1.97) | 0.47 µs (1.97) | 2.07 µs (8.64) | 19.8 µs (82.75) | 1.25 µs (5.23) |
| 4 | 0.54 µs | 1.55 µs (2.89) | 1.23 µs (2.30) | 0.93 µs (1.73) | 5.26 µs (9.81) | 50.8 µs (94.90) | 3.44 µs (6.42) |
| 5 | 0.87 µs | 2.42 µs (2.78) | 2.53 µs (2.91) | 1.42 µs (1.63) | 9.37 µs (10.77) | 104.4 µs (120.01) | 7.19 µs (8.26) |
| 6 | 1.50 µs | 3.98 µs (2.66) | 4.73 µs (3.16) | 2.35 µs (1.57) | 18.7 µs (12.47) | 192.1 µs (128.32) | 13.4 µs (8.98) |
| 8 | 3.35 µs | 8.22 µs (2.45) | 11.2 µs (3.35) | 5.65 µs (1.69) | 38.3 µs (11.43) | 463.6 µs (138.24) | 36.3 µs (10.81) |
| 16 | 41.7 µs | 57.6 µs (1.38) | 124.8 µs (2.99) | 46.9 µs (1.13) | 224.2 µs (5.38) | 4.51 ms (108.09) | 493.2 µs (11.83) |
| 32 | 682.9 µs | 858.2 µs (1.26) | 1.81 ms (2.65) | 318.3 µs (0.47) | 1.33 ms (1.95) | 44.9 ms (65.71) | 4.54 ms (6.65) |
| 64 | 5.45 ms | 6.63 ms (1.22) | 16.2 ms (2.97) | 2.38 ms (0.44) | 8.13 ms (1.49) | 435 ms (79.70) | 38.3 ms (7.02) |
| 128 | 36.6 ms | 42.4 ms (1.16) | 138 ms (3.78) | 15.9 ms (0.43) | 33.8 ms (0.92) | 4.09 s (111.57) | 313 ms (8.55) |

`&a * &b` of `Matrix<Rational>`, 64-bit parts (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, sums of products of rationals | FLINT | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.23 µs | 1.41 µs (1.15) | 1.36 µs (1.11) | 2.44 µs (1.99) | 4.70 µs (3.82) | 40.0 µs (32.56) | 2.75 µs (2.24) |
| 3 | 5.62 µs | 6.54 µs (1.16) | 6.06 µs (1.08) | 8.78 µs (1.56) | 16.1 µs (2.87) | 159.0 µs (28.27) | 11.5 µs (2.05) |
| 4 | 16.0 µs | 18.2 µs (1.14) | 17.2 µs (1.07) | 22.9 µs (1.43) | 46.9 µs (2.92) | 441.7 µs (27.56) | 32.3 µs (2.01) |
| 5 | 39.6 µs | 51.0 µs (1.29) | 40.0 µs (1.01) | 41.6 µs (1.05) | 111.5 µs (2.81) | 980.3 µs (24.73) | 65.6 µs (1.65) |
| 6 | 69.9 µs | 86.1 µs (1.23) | 73.4 µs (1.05) | 74.0 µs (1.06) | 260.9 µs (3.73) | 1.95 ms (27.89) | 119.9 µs (1.71) |
| 8 | 172.8 µs | 204.3 µs (1.18) | 210.8 µs (1.22) | 193.7 µs (1.12) | 1.05 ms (6.06) | 5.69 ms (32.96) | 314.1 µs (1.82) |
| 16 | 1.8 ms | 1.99 ms (1.11) | 2.62 ms (1.46) | 2.1 ms (1.17) | 37.2 ms (20.70) | 84.3 ms (46.89) | 3.06 ms (1.70) |
| 32 | 21.3 ms | 21.9 ms (1.03) | 28.4 ms (1.34) | 20.4 ms (0.96) | 1.47 s (68.87) | 1.46 s (68.84) | 31.5 ms (1.48) |
| 64 | 305 ms | 311 ms (1.02) | 293 ms (0.96) | 227 ms (0.74) | 48.2 s (158.14) | 30.7 s (100.64) | 350 ms (1.15) |

`Matrix<Rational>::det()`, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.02 µs | 0.09 µs (5.01) | 0.02 µs (0.96) | 0.02 µs (1.02) | 0.11 µs (6.37) | 0.02 µs (1.06) | 1.69 µs (97.45) | 0.14 µs (8.22) |
| 3 | 0.04 µs | 0.14 µs (3.81) | 0.04 µs (0.97) | 0.08 µs (2.28) | 0.18 µs (4.97) | 0.08 µs (2.12) | 5.13 µs (140.06) | 0.39 µs (10.78) |
| 4 | 0.05 µs | 0.20 µs (3.68) | 0.05 µs (1.00) | 0.15 µs (2.71) | 0.28 µs (5.15) | 0.71 µs (12.91) | 14.2 µs (258.69) | 1.08 µs (19.65) |
| 5 | 0.09 µs | 0.27 µs (2.89) | 0.10 µs (1.03) | 0.43 µs (4.55) | 0.43 µs (4.50) | 1.03 µs (10.89) | 31.6 µs (333.42) | 2.40 µs (25.31) |
| 6 | 0.13 µs | 0.38 µs (2.84) | 0.13 µs (0.99) | 0.64 µs (4.81) | 0.64 µs (4.77) | 1.54 µs (11.58) | 59.9 µs (449.47) | 4.33 µs (32.50) |
| 8 | 0.24 µs | 0.70 µs (2.98) | 0.24 µs (1.00) | 1.21 µs (5.14) | 1.24 µs (5.24) | 3.24 µs (13.70) | 168.5 µs (713.27) | 12.7 µs (53.85) |
| 16 | 7.00 µs | 7.80 µs (1.11) | 6.97 µs (1.00) | 9.31 µs (1.33) | 9.42 µs (1.35) | 38.4 µs (5.48) | 2.22 ms (317.39) | 221.2 µs (31.60) |
| 32 | 112.8 µs | 116.7 µs (1.03) | 161.2 µs (1.43) | 76.4 µs (0.68) | 190.7 µs (1.69) | 228.3 µs (2.02) | 35.6 ms (315.38) | 4.72 ms (41.86) |
| 64 | 482.8 µs | 501.5 µs (1.04) | 2.55 ms (5.29) | 618.1 µs (1.28) | 3 ms (6.22) | 1.99 ms (4.12) | 595 ms (1231.65) | 77.3 ms (160.11) |
| 128 | 3.26 ms | 3.35 ms (1.03) | 38 ms (11.68) | 3.93 ms (1.21) | 41.9 ms (12.87) | 20.4 ms (6.25) | 11 s (3387.21) | 1.18 s (363.80) |

`Matrix<Rational>::det()`, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.02 µs | 0.13 µs (6.01) | 0.02 µs (1.05) | 0.03 µs (1.20) | 0.15 µs (6.93) | 0.11 µs (5.10) | 2.17 µs (99.02) | 0.16 µs (7.11) |
| 3 | 0.09 µs | 0.26 µs (2.97) | 0.08 µs (0.98) | 0.17 µs (2.02) | 0.33 µs (3.80) | 0.71 µs (8.18) | 7.51 µs (86.87) | 0.53 µs (6.10) |
| 4 | 0.31 µs | 0.49 µs (1.61) | 0.31 µs (1.03) | 0.38 µs (1.24) | 0.55 µs (1.80) | 4.33 µs (14.18) | 22.3 µs (73.05) | 1.55 µs (5.09) |
| 5 | 0.60 µs | 0.87 µs (1.43) | 0.58 µs (0.96) | 0.81 µs (1.34) | 0.83 µs (1.38) | 8.64 µs (14.31) | 50.8 µs (84.10) | 3.66 µs (6.07) |
| 6 | 1.61 µs | 2.04 µs (1.27) | 1.60 µs (1.00) | 1.62 µs (1.01) | 1.61 µs (1.00) | 16.5 µs (10.27) | 114.0 µs (70.74) | 9.41 µs (5.84) |
| 8 | 4.54 µs | 5.26 µs (1.16) | 4.47 µs (0.99) | 4.58 µs (1.01) | 4.57 µs (1.01) | 48.1 µs (10.60) | 374.3 µs (82.47) | 40.2 µs (8.86) |
| 16 | 75.6 µs | 80.1 µs (1.06) | 74.6 µs (0.99) | 73.4 µs (0.97) | 73.9 µs (0.98) | 473.8 µs (6.26) | 8.45 ms (111.73) | 1.15 ms (15.15) |
| 32 | 915.5 µs | 979.2 µs (1.07) | 1.62 ms (1.77) | 626.5 µs (0.68) | 1.55 ms (1.69) | 3.92 ms (4.28) | 230 ms (250.98) | 29 ms (31.69) |
| 64 | 8.69 ms | 9.22 ms (1.06) | 56.3 ms (6.48) | 7.9 ms (0.91) | 53.6 ms (6.17) | 33.6 ms (3.87) | 8.36 s (961.40) | 732 ms (84.22) |
| 128 | 76.7 ms | 78.7 ms (1.03) | 1.95 s (25.50) | 116 ms (1.51) | 1.87 s (24.41) | 335 ms (4.37) | – | 20.3 s (264.83) |

`Matrix<Rational>::det()`, 64-bit parts (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.27 µs | 1.35 µs (5.00) | 0.32 µs (1.18) | 0.41 µs (1.51) | 0.85 µs (3.14) | 0.44 µs (1.64) | 14.5 µs (53.80) | 0.66 µs (2.45) |
| 3 | 2.37 µs | 4.15 µs (1.75) | 2.34 µs (0.99) | 2.26 µs (0.96) | 2.46 µs (1.04) | 4.25 µs (1.80) | 74.9 µs (31.60) | 6.04 µs (2.55) |
| 4 | 5.06 µs | 8.65 µs (1.71) | 4.99 µs (0.99) | 4.38 µs (0.86) | 5.12 µs (1.01) | 285.1 µs (56.32) | 266.2 µs (52.59) | 23.8 µs (4.71) |
| 5 | 10.5 µs | 16.5 µs (1.57) | 10.5 µs (1.01) | 10.1 µs (0.97) | 10.1 µs (0.97) | 661.5 µs (63.09) | 735.8 µs (70.18) | 67.3 µs (6.42) |
| 6 | 20.6 µs | 29.6 µs (1.44) | 20.4 µs (0.99) | 19.4 µs (0.94) | 19.3 µs (0.93) | 1.43 ms (69.12) | 1.78 ms (86.53) | 158.0 µs (7.66) |
| 8 | 73.6 µs | 92.0 µs (1.25) | 73.6 µs (1.00) | 71.2 µs (0.97) | 71.2 µs (0.97) | 5.39 ms (73.19) | 7.69 ms (104.48) | 737.0 µs (10.01) |
| 16 | 4.21 ms | 4.42 ms (1.05) | 4.23 ms (1.01) | 4.11 ms (0.98) | 4.05 ms (0.96) | 190 ms (45.24) | 504 ms (119.86) | 30.2 ms (7.18) |
| 32 | 68.2 ms | 69.2 ms (1.01) | 232 ms (3.41) | 43.5 ms (0.64) | 316 ms (4.63) | 10.7 s (156.66) | 52.1 s (763.54) | 1.53 s (22.45) |
| 64 | 1.31 s | 1.31 s (1.00) | 8.43 s (6.45) | 861 ms (0.66) | 20.6 s (15.78) | – | – | 91.5 s (69.98) |

`Matrix<Rational>::det()`, 256-bit parts (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.24 µs | 5.26 µs (1.62) | 3.17 µs (0.98) | 3.58 µs (1.11) | 4.23 µs (1.30) | 3.84 µs (1.18) | 73.1 µs (22.55) | 4.75 µs (1.46) |
| 4 | 35.2 µs | 34.4 µs (0.98) | 35.4 µs (1.00) | 26.1 µs (0.74) | 29.7 µs (0.84) | 1.54 ms (43.71) | 1.6 ms (45.33) | 111.2 µs (3.16) |
| 8 | 783.0 µs | 782.2 µs (1.00) | 781.2 µs (1.00) | 696.1 µs (0.89) | 690.3 µs (0.88) | 38.9 ms (49.66) | 83.5 ms (106.71) | 4.73 ms (6.05) |
| 16 | 33.5 ms | 33.3 ms (0.99) | 33.5 ms (1.00) | 45.5 ms (1.36) | 45 ms (1.34) | 1.92 s (57.44) | 7.84 s (234.05) | 232 ms (6.92) |
| 32 | 931 ms | 931 ms (1.00) | 1.16 s (1.25) | 355 ms (0.38) | 2.92 s (3.13) | 110 s (118.22) | – | 13.5 s (14.47) |

`Matrix<Rational>::solve(&b)`, one column, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.04 µs | 0.18 µs (4.69) | 0.04 µs (1.04) | 0.10 µs (2.68) | 0.23 µs (5.92) | 1.53 µs (39.29) | 3.91 µs (100.37) | 0.31 µs (7.84) |
| 3 | 0.09 µs | 0.29 µs (3.33) | 0.09 µs (1.00) | 0.19 µs (2.20) | 0.36 µs (4.12) | 2.39 µs (27.46) | 11.3 µs (129.24) | 0.83 µs (9.58) |
| 4 | 0.14 µs | 0.43 µs (3.06) | 0.14 µs (0.99) | 0.52 µs (3.74) | 0.53 µs (3.80) | 2.41 µs (17.22) | 28.4 µs (202.92) | 2.15 µs (15.33) |
| 5 | 0.23 µs | 0.60 µs (2.60) | 0.24 µs (1.04) | 0.74 µs (3.20) | 0.74 µs (3.22) | 9.83 µs (42.61) | 59.8 µs (259.15) | 4.33 µs (18.78) |
| 6 | 0.34 µs | 0.81 µs (2.41) | 0.33 µs (1.00) | 1.03 µs (3.08) | 1.06 µs (3.16) | 7.83 µs (23.35) | 102.2 µs (305.02) | 7.81 µs (23.29) |
| 8 | 0.67 µs | 1.38 µs (2.07) | 0.68 µs (1.02) | 1.92 µs (2.89) | 1.90 µs (2.85) | 14.6 µs (21.96) | 274.2 µs (411.79) | 22.0 µs (32.98) |
| 16 | 12.5 µs | 13.3 µs (1.06) | 12.5 µs (1.00) | 32.5 µs (2.60) | 16.7 µs (1.33) | 59.0 µs (4.71) | 3.09 ms (246.66) | 335.2 µs (26.79) |
| 24 | 60.7 µs | 63.4 µs (1.04) | 65.2 µs (1.07) | 70.8 µs (1.17) | 79.1 µs (1.30) | 204.9 µs (3.37) | 14.1 ms (231.82) | 1.81 ms (29.77) |
| 32 | 99.6 µs | 103.6 µs (1.04) | 194.7 µs (1.96) | 149.4 µs (1.50) | 235.5 µs (2.37) | 349.7 µs (3.51) | 42.6 ms (427.71) | 5.9 ms (59.24) |
| 64 | 450.2 µs | 484.4 µs (1.08) | 2.74 ms (6.10) | 680.6 µs (1.51) | 3.36 ms (7.46) | 5.49 ms (12.19) | 663 ms (1473.54) | 86.8 ms (192.72) |
| 128 | 2.58 ms | 2.66 ms (1.03) | 39.5 ms (15.30) | 4.15 ms (1.61) | 44.1 ms (17.06) | 53.3 ms (20.63) | 11.8 s (4570.30) | 1.26 s (487.85) |

`Matrix<Rational>::solve(&b)`, one column, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.07 µs | 0.27 µs (3.97) | 0.07 µs (0.96) | 0.17 µs (2.43) | 0.33 µs (4.78) | 2.13 µs (30.86) | 5.47 µs (79.26) | 0.43 µs (6.27) |
| 3 | 0.19 µs | 0.51 µs (2.65) | 0.19 µs (0.96) | 0.34 µs (1.75) | 0.59 µs (3.04) | 6.54 µs (33.93) | 18.1 µs (94.11) | 1.40 µs (7.28) |
| 4 | 0.96 µs | 1.18 µs (1.23) | 0.95 µs (0.99) | 1.35 µs (1.41) | 1.39 µs (1.44) | 17.6 µs (18.39) | 50.6 µs (52.76) | 4.16 µs (4.34) |
| 5 | 1.77 µs | 2.08 µs (1.18) | 1.75 µs (0.99) | 2.15 µs (1.22) | 2.17 µs (1.23) | 17.5 µs (9.93) | 108.7 µs (61.54) | 9.01 µs (5.10) |
| 6 | 4.13 µs | 4.68 µs (1.13) | 4.15 µs (1.00) | 4.35 µs (1.05) | 4.40 µs (1.07) | 43.6 µs (10.54) | 234.3 µs (56.68) | 21.8 µs (5.28) |
| 8 | 9.82 µs | 10.7 µs (1.09) | 9.64 µs (0.98) | 10.5 µs (1.07) | 10.4 µs (1.06) | 65.7 µs (6.70) | 711.6 µs (72.47) | 80.4 µs (8.18) |
| 16 | 114.7 µs | 121.4 µs (1.06) | 117.0 µs (1.02) | 157.3 µs (1.37) | 124.6 µs (1.09) | 558.1 µs (4.87) | 13.3 ms (116.00) | 1.86 ms (16.24) |
| 24 | 347.5 µs | 370.6 µs (1.07) | 568.2 µs (1.64) | 447.0 µs (1.29) | 606.8 µs (1.75) | 2.37 ms (6.83) | 78.9 ms (227.07) | 10.6 ms (30.37) |
| 32 | 831.5 µs | 907.5 µs (1.09) | 2.01 ms (2.41) | 1.13 ms (1.35) | 2.14 ms (2.58) | 7.96 ms (9.58) | 304 ms (365.26) | 37.1 ms (44.66) |
| 64 | 6.88 ms | 7.36 ms (1.07) | 61.3 ms (8.91) | 8.8 ms (1.28) | 60.8 ms (8.84) | 58.1 ms (8.45) | 9.81 s (1426.12) | 843 ms (122.56) |
| 128 | 50 ms | 51.9 ms (1.04) | 2.01 s (40.11) | 106 ms (2.13) | 1.96 s (39.18) | 510 ms (10.20) | – | 22.1 s (441.20) |

`Matrix<Rational>::solve(&b)`, one column, 64-bit parts (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.38 µs | 3.55 µs (1.49) | 2.43 µs (1.02) | 2.14 µs (0.90) | 2.56 µs (1.08) | 24.2 µs (10.16) | 47.8 µs (20.12) | 2.55 µs (1.07) |
| 3 | 7.07 µs | 9.60 µs (1.36) | 7.01 µs (0.99) | 6.99 µs (0.99) | 7.98 µs (1.13) | 155.4 µs (21.98) | 234.6 µs (33.18) | 16.5 µs (2.33) |
| 4 | 16.9 µs | 21.4 µs (1.26) | 16.6 µs (0.98) | 19.3 µs (1.14) | 19.3 µs (1.14) | 181.7 µs (10.74) | 801.3 µs (47.38) | 59.6 µs (3.52) |
| 5 | 34.5 µs | 42.3 µs (1.23) | 33.9 µs (0.98) | 39.7 µs (1.15) | 39.7 µs (1.15) | 540.9 µs (15.70) | 2.15 ms (62.26) | 156.0 µs (4.53) |
| 6 | 65.8 µs | 78.0 µs (1.18) | 65.9 µs (1.00) | 77.3 µs (1.17) | 77.8 µs (1.18) | 1.36 ms (20.61) | 4.99 ms (75.81) | 370.2 µs (5.63) |
| 8 | 211.2 µs | 232.4 µs (1.10) | 211.6 µs (1.00) | 252.5 µs (1.20) | 252.1 µs (1.19) | 1.91 ms (9.06) | 20.6 ms (97.36) | 1.69 ms (7.99) |
| 16 | 6.02 ms | 6.02 ms (1.00) | 7.42 ms (1.23) | 10.3 ms (1.71) | 7.89 ms (1.31) | 22.8 ms (3.79) | 1.07 s (177.48) | 53.9 ms (8.96) |
| 32 | 85.3 ms | 86 ms (1.01) | 289 ms (3.39) | 140 ms (1.64) | 404 ms (4.74) | 341 ms (3.99) | 83.3 s (977.00) | 2.32 s (27.17) |
| 64 | 1.47 s | 1.47 s (1.00) | 9.31 s (6.34) | 2.48 s (1.69) | 22.8 s (15.51) | 5.71 s (3.89) | – | 114 s (77.72) |

`Matrix<Rational>::solve(&b)`, one column, 256-bit parts (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | `a`, Bareiss alone | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 13.7 µs | 13.1 µs (0.96) | 13.8 µs (1.01) | 11.6 µs (0.85) | 12.7 µs (0.93) | 349.9 µs (25.57) | 252.6 µs (18.46) | 14.5 µs (1.06) |
| 4 | 104.7 µs | 102.4 µs (0.98) | 105.0 µs (1.00) | 116.2 µs (1.11) | 115.3 µs (1.10) | 2.66 ms (25.44) | 5.58 ms (53.36) | 321.5 µs (3.07) |
| 8 | 2.12 ms | 2.14 ms (1.01) | 2.09 ms (0.99) | 2.25 ms (1.06) | 2.3 ms (1.08) | 23.1 ms (10.88) | 260 ms (122.36) | 11.2 ms (5.30) |
| 16 | 68.5 ms | 68.7 ms (1.00) | 56.6 ms (0.83) | 106 ms (1.55) | 79.5 ms (1.16) | 153 ms (2.24) | 17.4 s (254.41) | 454 ms (6.63) |
| 32 | 1.09 s | 1.08 s (0.99) | 1.49 s (1.36) | 1.61 s (1.47) | 3.47 s (3.18) | 2.27 s (2.07) | – | 19.1 s (17.46) |

`Matrix<Rational>::inverse()`, integers in −99..=99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.05 µs | 0.24 µs (5.30) | 0.07 µs (1.55) | 0.28 µs (6.20) | 0.44 µs (9.53) | 5.25 µs (114.78) | 0.35 µs (7.76) |
| 3 | 0.14 µs | 0.50 µs (3.64) | 0.33 µs (2.38) | 0.54 µs (3.86) | 7.40 µs (53.38) | 19.0 µs (137.05) | 1.20 µs (8.68) |
| 4 | 0.29 µs | 0.96 µs (3.33) | 0.89 µs (3.07) | 0.92 µs (3.21) | 10.5 µs (36.27) | 58.4 µs (202.53) | 3.57 µs (12.40) |
| 5 | 0.58 µs | 1.52 µs (2.62) | 1.61 µs (2.76) | 1.61 µs (2.77) | 7.81 µs (13.44) | 136.8 µs (235.39) | 9.07 µs (15.61) |
| 6 | 0.98 µs | 2.77 µs (2.82) | 2.64 µs (2.69) | 2.72 µs (2.77) | 9.77 µs (9.96) | 264.7 µs (269.79) | 18.0 µs (18.35) |
| 8 | 2.48 µs | 6.10 µs (2.46) | 5.91 µs (2.39) | 5.96 µs (2.40) | 14.3 µs (5.78) | 818.1 µs (330.10) | 71.4 µs (28.81) |
| 16 | 89.5 µs | 90.9 µs (1.02) | 159.1 µs (1.78) | 119.5 µs (1.34) | 157.9 µs (1.76) | 12.8 ms (143.09) | 1.83 ms (20.49) |
| 32 | 1.16 ms | 1.17 ms (1.01) | 1.57 ms (1.35) | 1.54 ms (1.33) | 1.17 ms (1.01) | 224 ms (193.40) | 35.4 ms (30.51) |
| 64 | 13.8 ms | 14 ms (1.01) | 17.8 ms (1.29) | 19.9 ms (1.44) | 10 ms (0.73) | 3.98 s (287.70) | 585 ms (42.29) |
| 128 | 175 ms | 175 ms (1.00) | 187 ms (1.07) | 250 ms (1.43) | 99.6 ms (0.57) | 78.4 s (448.41) | 10.1 s (58.00) |

`Matrix<Rational>::inverse()`, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99 (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.10 µs | 0.31 µs (3.19) | 0.09 µs (0.89) | 0.34 µs (3.54) | 3.84 µs (39.82) | 6.21 µs (64.45) | 0.40 µs (4.10) |
| 3 | 0.31 µs | 0.81 µs (2.65) | 0.48 µs (1.56) | 0.75 µs (2.45) | 6.25 µs (20.42) | 30.0 µs (98.02) | 1.88 µs (6.14) |
| 4 | 2.18 µs | 2.44 µs (1.12) | 3.55 µs (1.63) | 3.56 µs (1.63) | 19.2 µs (8.80) | 100.6 µs (46.05) | 6.80 µs (3.11) |
| 5 | 5.33 µs | 5.50 µs (1.03) | 6.22 µs (1.17) | 6.18 µs (1.16) | 49.3 µs (9.24) | 266.6 µs (50.00) | 24.7 µs (4.63) |
| 6 | 15.2 µs | 15.4 µs (1.01) | 15.5 µs (1.02) | 15.7 µs (1.03) | 84.6 µs (5.57) | 679.1 µs (44.66) | 69.9 µs (4.59) |
| 8 | 42.5 µs | 43.0 µs (1.01) | 45.6 µs (1.07) | 45.2 µs (1.06) | 235.5 µs (5.54) | 2.54 ms (59.78) | 278.6 µs (6.56) |
| 16 | 704.1 µs | 698.3 µs (0.99) | 1.2 ms (1.71) | 909.9 µs (1.29) | 2.49 ms (3.54) | 70.7 ms (100.45) | 9.31 ms (13.22) |
| 32 | 12.6 ms | 12.7 ms (1.01) | 20.2 ms (1.61) | 16.3 ms (1.30) | 24 ms (1.91) | 2.26 s (179.89) | 254 ms (20.26) |
| 64 | 314 ms | 315 ms (1.00) | 363 ms (1.16) | 395 ms (1.26) | 251 ms (0.80) | 93.8 s (298.71) | 7.3 s (23.23) |
| 128 | 8.91 s | 8.95 s (1.00) | 6.13 s (0.69) | 10.6 s (1.19) | 2.72 s (0.31) | – | – |

`Matrix<Rational>::inverse()`, 64-bit parts (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.87 µs | 3.10 µs (3.55) | 0.91 µs (1.04) | 2.60 µs (2.97) | 28.3 µs (32.32) | 51.6 µs (59.03) | 1.79 µs (2.05) |
| 3 | 12.2 µs | 13.9 µs (1.14) | 12.4 µs (1.01) | 14.3 µs (1.17) | 158.8 µs (13.00) | 379.9 µs (31.10) | 21.0 µs (1.72) |
| 4 | 40.1 µs | 43.2 µs (1.08) | 45.9 µs (1.14) | 47.1 µs (1.18) | 587.2 µs (14.66) | 1.69 ms (42.15) | 100.3 µs (2.50) |
| 5 | 102.9 µs | 107.5 µs (1.04) | 131.5 µs (1.28) | 128.4 µs (1.25) | 1.76 ms (17.09) | 5.54 ms (53.86) | 338.0 µs (3.29) |
| 6 | 228.8 µs | 238.3 µs (1.04) | 320.9 µs (1.40) | 331.3 µs (1.45) | 4.59 ms (20.05) | 15.3 ms (66.95) | 1.1 ms (4.82) |
| 8 | 951.4 µs | 992.7 µs (1.04) | 1.33 ms (1.40) | 1.3 ms (1.37) | 22.8 ms (23.94) | 82 ms (86.23) | 5.9 ms (6.20) |
| 16 | 34.8 ms | 34.6 ms (1.00) | 81.7 ms (2.35) | 45.2 ms (1.30) | 1.23 s (35.34) | 7.83 s (225.15) | 329 ms (9.47) |
| 32 | 1.37 s | 1.39 s (1.02) | 3.08 s (2.25) | 2.11 s (1.54) | 63.4 s (46.28) | – | 23.9 s (17.48) |

`Matrix<Rational>::inverse()`, 256-bit parts (median, and the time over `a`'s):

| n | `a` | `a` 0.2.34 | FLINT | FLINT, Bareiss | PARI | num-rational | malachite-q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6.82 µs | 12.4 µs (1.81) | 7.05 µs (1.03) | 13.2 µs (1.94) | 168.9 µs (24.77) | 256.8 µs (37.66) | 10.5 µs (1.54) |
| 4 | 229.9 µs | 225.7 µs (0.98) | 314.4 µs (1.37) | 307.3 µs (1.34) | 4.08 ms (17.76) | 12 ms (52.37) | 637.8 µs (2.77) |
| 8 | 7.62 ms | 7.66 ms (1.01) | 10.5 ms (1.38) | 10.3 ms (1.36) | 174 ms (22.87) | 1.15 s (150.53) | 42.9 ms (5.64) |
| 16 | 275 ms | 274 ms (0.99) | 935 ms (3.39) | 415 ms (1.51) | 8.87 s (32.21) | 135 s (490.42) | 3.13 s (11.38) |

- Up to 8 × 8 the matrices of small integers and fractions run in registers. Against the crate before, they are 1.01 to 7.0 times as fast: allocations were most of the time there, and the reading of every element through numbers. Against FLINT, determinants of small integers are as fast at 2 × 2 and 2.3 to 5.1 times as fast from 3 × 3 to 8 × 8, solutions 2.2 to 3.7 times and inverses 1.6 to 3.1 times. For small fractions, determinants are 1.01 to 2.0 times as fast, solutions 1.05 to 2.4 times and inverses 1.02 to 1.6 times from 3 × 3, but 0.89 at 2 × 2. The product of small integers takes 0.39 to 0.60 of FLINT's time at every size up to 128 × 128, and of small fractions 0.52 to 0.88 of it up to 16 × 16.
- 64-bit parts, whose rows clear beyond the words, gain 1.04 to 5.0 times over the crate before from 2 × 2 to 8 × 8 for determinants, solutions and inverses: the rows are cleared on limbs with no number made, and a `2 × 2` determinant or inverse comes from the rationals' arithmetic, as FLINT's does. Their products take 0.78 to 0.88 of the time they took, up to 8 × 8.
- From 16 × 16 up, the determinants, solutions and inverses take about the time they took before: the crate before takes 0.99 to 1.11 times as long. So do the 256-bit parts, which are not in words, from 4 × 4 (the crate before takes 0.98 to 1.01 times as long). Their `2 × 2` determinants and inverses take the rationals' arithmetic, 1.6 to 1.8 times as fast as before, and their `2 × 2` solutions 1.04 times the time before. The method there is the one of #16: Bareiss's elimination below the thresholds, Dixon's lifting from them. The products are faster: those of small integers stay in registers at every size, 3.1 to 6.5 times as fast as before from 16 × 16 to 128 × 128, and those of fractions 1.16 to 1.38 times as fast.
- Losses to FLINT: determinants of 64-bit parts from 3 × 3 to 8 × 8 (0.86 to 0.97 of the time) and of 256-bit parts at 4 × 4 and 8 × 8 (0.74 and 0.89), solutions of 64-bit and 256-bit parts at 2 × 2 (0.90 and 0.85), and the inverse of small fractions at 2 × 2 (0.89). These integers run to 2 to 20 limbs. FLINT's are GMP's, with hand-written kernels for products and exact quotients of a few limbs; here each element is a vector of its own, run through kernels written for any length. The determinant's final reduction, one greatest common divisor of the integers' determinant and the product of the rows' multiples, is about a third of the time at 4 × 4, and FLINT's `fmpq_canonicalise` does the same reduction. Determinants at 32 × 32 (0.38 to 0.68), and of 64-bit parts at 64 × 64 (0.66), sit at the threshold of Dixon's lifting, where FLINT's modular determinant needs fewer residues; that is as it was. The product of fractions from 32 × 32 (0.43 to 0.47 of the time) has row multiples that pass a word, so the integers are long, and FLINT multiplies them on packed kernels, as before #27; for 64-bit parts at 64 × 64 it takes 0.74. PARI's multimodular inverse is faster from 64 × 64 for small integers (0.57 to 0.73) and fractions (0.31 to 0.80), and so is FLINT's at 128 × 128 for fractions (0.69).

## Efficiency

- Time complexity: a sum or a product of parts of `n` limbs takes two or three greatest common divisors and four products of `n`-limb numbers, `O(M(n) log n)` with the half gcd for the longest; a comparison at most two products.
- Space complexity: `O(n)`, the parts and the intermediate products.
- `to_f64` and `to_f32`: `O(1)` but for about one fraction in 700 of random parts of more than 64 bits, `O(n)` for a denominator of `n` limbs, and always `O(1)` for parts in words; the conversion from a float `O(1)`, with no allocation when its parts fit in 64 bits and two vectors of at most 135 bytes otherwise.
- `FromStr` and `from_str_radix`: the reading of the parts, `O(M(n) log n)` for `n` digits, and for a fraction the gcd of the parts, in registers and with no allocation for parts of up to 64 bits; a decimal with a negative exponent at most a remainder by `5^27`, or a gcd with `5^k` when its digits end in more than 27 factors of 5; `to_string` and `to_str_radix` the writing of the parts, `O(M(n) log n)`.
- Time complexity: `B_n` in about `O(n² log² n)` word operations, the Euler product's primes up to `n / 17` at falling precisions and the residues' `O(p)` for each prime up to about `0.15 n log2 n`; `B_0, ..., B_n` in `O(n³ log n)` word operations for the powers' passes, plus two products of each number's bits for each index; `H_n` in `O(M(n log n) log n)` for the tree, `M` the time of a product, plus `O(M(n))` for the reduction.
- Space complexity: the result and a few times its length for `B_n`, the residues' tables of 1 KB or 8 KB; for the vector, the results, and at most `n / 34` powers of the top index's precision.
- `det()` and `rank()` of an `n × n` matrix: about `n³/3` steps, each two products and an exact division of integers as long as the minors, `O(n³ M(n·b))` for elements of `b` bits; `solve(&b)` with `k` right-hand sides about `n³/3 + n²·k` of them and `n·k` gcds, `inverse()` about `4n³/3` and `n²` gcds; the product `n³` products of integers and `n²` gcds. The large determinants and systems: `n³/3` products modulo a prime and about `2·L·n²` products of words, `L` the length of the determinant over 61 bits, twice for the solutions, `O(n³ + n³·b/61)` word operations. Space: the matrix of integers and the right-hand sides, a vector of limbs for each element beyond the registers; Dixon's lifting the factors modulo the prime, `n²` words, and the digits, `L·n·k` words.

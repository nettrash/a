[A](../../../README.md) › [Math](../../../README.md#math)

# Rational numbers

`Rational` (in `a::math::rational`) is the fraction `p/q` of two [numbers](numbers.md), always in lowest terms with a positive denominator, so that each rational number has one form: zero is `0/1`, an integer `n` is `n/1`, and two rationals are equal when their parts are. The parts are private, so that the form holds; `numer`, `denom` and `into_parts` give them back.

- `Rational::new(p, q)` reduces any fraction with `q ≠ 0` to the form, `Rational::from_integer(n)` and `From` make integers, from a `Number` or any primitive integer.
- They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` and divide with `/` and `/=`, each operand owned or borrowed, and take a `Number` or a primitive integer as an operand on either side: `r + 1`, `2 * &r`, `&n / r`. Dividing by zero panics, as it does for integers.
- `-r` negates, `recip` is `1/r`, `abs` and `signum` are as for integers, and `pow` takes an `i32` exponent, negative ones by the reciprocal; `0⁰` is 1.
- `==`, `<` and the rest compare by value, with numbers on either side and primitive integers on the right; `Hash` agrees with `==`, `Default` is zero, `Sum` and `Product` take iterators of owned or borrowed rationals. `Display` and `Debug` write `-3/4`, or `5` for an integer, the formatter's width, fill and `+` applying to the whole.
- `Rational::try_from(x)` takes any finite `f64` or `f32` exactly: every float is `m·2^e`, so `0.1` is `3602879701896397/36028797018963968`, and `-0.0` is zero; NaN and the infinities are a `TryFromFloatError`. A float whose parts fit in 64 bits, as every float from `2^-11` to `2^12` does, is made in words with no allocation. `to_f64` and `to_f32` go back to the nearest float, ties to even, rounded once and correctly, subnormal numbers included: a number too large is infinity of its sign, one too small zero, `-0.0` for a negative one as IEEE 754 has it, and `0.0` for zero. Parts in words below `2^53` (`2^24` for `f32`) are floats exactly and one division rounds their quotient; other parts in words take the quotient of their nearest `f64`s, a float within two units in the last place, and the exact remainder of the fraction by it, in 128 bits, which says whether the float or a neighbour is the nearest.
- `Number::try_from(&r)` and `u8::try_from(&r)` up to `i128` and `usize` give an integer, a `TryFromRationalError` for a fraction or an integer out of the type's range.
- `"-3/4".parse::<Rational>()` reads integers, fractions `p/q` (the denominator without a sign, not zero, the fraction reduced) and decimals `1.25`, `.5`, `5.`, with an exponent `e` or `E` on decimals and integers, `-1.5e-3`, `2e10`, exactly: `0.1` is `1/10`. There is no whitespace, no `_`, nothing else, as for a `Number`. Parts of up to 64 bits, 19 or 20 digits, are read eight digits at a time into words and reduced there, with no allocation, and so are decimals whose digits make a word, with an exponent of up to 19 in magnitude: their twos or fives are the only factors they share with the power of ten. The exponent may be up to `Rational::MAX_PARSE_EXPONENT`, `1000000`, in magnitude: `1e1000000` is a numerator of 415 KB, and the bound keeps a short text from asking for gigabytes. A `ParseRationalError` has a `kind()`: `Empty`, `InvalidDigit`, `ZeroDenominator` or `ExponentOutOfRange`.
- `to_str_radix(radix)` writes `-p/q` in a radix from 2 to 36, and `from_str_radix(s, radix)` reads integers and fractions in it, letters in either case, with no point or exponent in any radix (`e` is a digit from radix 15 on). Parts in words are written and read in registers in every radix: binary eight digits at a time, the other powers of two by shifts, and the other radices in blocks below `2^32`, each digit by a product with the radix's reciprocal; the only allocation is the text written.
- `floor`, `ceil`, `trunc`, `round` (halves away from zero, as `f64::round`) and `round_ties_even` give integers as `Number`s, and `fract` the rational `r − trunc(r)`, of the sign of `r`; `floor_into`, `ceil_into`, `trunc_into`, `round_into` and `round_ties_even_into` write the integer into a `Number` that exists, in its vector when it has room.
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
- When every part fits in 64 bits, the arithmetic, the comparisons and the rounding are done in registers, the products in 128 bits. The time there is the greatest common divisors, by the binary method (about 0.7 steps a bit, as GMP's `mpn_gcd_11` takes them), and the one branch of each that no predictor can know, the end of its loop: random operands make about one misprediction a gcd, and a branch on their signs or on whether a gcd is 1 another one in two or three. So the sum `(a·d ± c·b)/(b·d)` is computed whole, without branches (both the sum and the difference of the products, one chosen), before `g = gcd(b, d)`, which the processor then runs beside it: when `g = 1`, as for three pairs of random denominators in five, that is the result in lowest terms. Otherwise it is divided by `g·gcd(t/g, g)`, Knuth's reduction for the whole numerator `t`, which is `gcd(t mod g², g²)` since `t/g` is prime to `b/g` and to `d/g`: one remainder, one short gcd and two divisions, in words when the products fit in 64 bits, and Knuth's steps in 128 bits when they do not, with the numerator's 129th bit apart and Hensel's exact division modulo `2^128`. A product's two gcds, `gcd(a, d)` and `gcd(c, b)`, run in step in one loop, both chains of dependent steps side by side and one loop end to mispredict rather than two, and the whole products, computed beside them, are the result when both are 1. The binary method stops when one of the two numbers is 1, as FLINT's `n_gcd` does, and takes a remainder first when one number is longer than the other by 16 bits or more. A comparison is the two products and the signs, chosen without branches. On random parts below `2^16`, `&a + &b` takes 16.0 ns where the inputs are new to the predictor, against FLINT's 24.5 ns; on a pool of 512 that it has learnt, 9.75 ns against FLINT's 10.7 ns.
- Parts of up to 16 bytes are read straight from their bytes as `u128`s: the gcds of two `u128`s (a limb with two longer by 16 bits or more by a remainder first, as `mpz_gcd` does, since the binary method would take a step a bit of the difference), the exact quotients by Hensel's division modulo `2^128`, the products in four limbs and a sum's numerator in five, all in registers, the limbs taking only a sum whose denominators share more than 64 bits. Otherwise the four parts are read into limbs once, the gcds (Lehmer's method and the half gcd of [numbers](numbers.md)), exact divisions and products are done on limbs in working memory (in the operation's own frame for parts of up to 12 limbs together), and the two parts of the result are written once, into the vectors of the operand assigned to or taken by value where they have room, with an eighth to spare when they grow. A part of one limb takes the steps for one limb: its gcd with a long part is a remainder by it, by the powers of `B` modulo it (computed once for both remainders of a sum), and the exact divisions and products by it are single passes. This is what a sum of many terms does at every term.
- `Sum` and `Product` combine their terms in a balanced tree, as a binary counter merges them, with at most `log₂ n` partial results kept: each addition is of two sums of about the same number of terms, so the parts of a sum of many short terms grow in a few long additions rather than one long addition a term. `H_n` takes `O(M(n) log n)` instead of `O(n²)`: `H_100000` takes 264 ns a term against 3.14 µs by `+=`. While the sum is in words the terms are added one at a time in registers, and the tree takes over from there, its first 8 partial results kept in place rather than in a vector.
- `Rational::new` reduces in the vectors it is given: parts of up to 8 bytes by the gcd of two limbs, up to 16 by that of two `u128`s and Hensel's exact division, longer ones on limbs; a gcd of 1, the common case, writes nothing.
- Two rationals compare by their signs, then by `a·d` against `c·b`, first bounded from the top 63 bits of each part: a part of `L > 63` bits lies in `[h, h + 1]·2^(L − 63)` for its top 63 bits `h`, and the products of the bounds, in 126 bits, decide unless the two numbers agree to about 120 bits. So a comparison takes about the same time at any length (9.57 ns for random parts of 2 limbs, 8.87 ns of 1,024), and only numbers that close are multiplied out.
- `floor`, `ceil` and `trunc` take the truncated quotient, one less or more by the sign (a remainder over a denominator above 1 is never zero in lowest terms); `round` and `round_ties_even` the quotient and the remainder on limbs, the remainder doubled against the denominator; parts of up to 16 bytes in registers. Each gives a `Number`, a vector: `floor` of short parts takes 8.84 ns, and `floor_into`, into a `Number` that exists, 2.72 ns, where FLINT's `fmpz_fdiv_q` writes a word into a variable that exists (1.95 ns) and malachite-q's `Integer` keeps a word in place (10.1 ns). `fract`, `abs`, `signum`, `recip`, `-x` and `pow` of short parts give rationals in words, allocating nothing.
- A float is `±m·2^e` from its bits: `m` is divided by its trailing zeros, which are added to `e`, and the rational is `m·2^e` when `e ≥ 0` and `m/2^-e` otherwise, in lowest terms as it is, both parts built directly as little-endian bytes. Its cost is the two vectors a rational has when a part passes 64 bits, and nothing otherwise.
- `to_f64` of `a/b`: when both parts are below `2^53`, both are floats exactly and one division of floats is correctly rounded. Otherwise the top 128 bits of `|a|` are divided by the top 64 bits of `b` in registers, by the reciprocal of the divisor as a division of limbs takes it (Möller and Granlund's), which gives a quotient `q` of 64 or 65 bits with `q − 4 < |a|/b·2^s < q + 1`; the result is `q` rounded with a sticky bit, unless the one halfway point between two floats near `q` lies in that interval. When `b` fits in 64 bits the quotient and its remainder are exact, and the bits of `a` below its top 128 are read only when the remainder is zero. Otherwise, about one fraction in 700, `|a|·2^s` (or `|a|/2^s`, its dropped bits kept for the sticky bit) is divided by `b` for a quotient of 65 or 66 bits, its remainder setting the sticky bit: linear in the length of `b`. The rounding is done on the integer bits, without branches, and a carry out of the significand runs into the exponent and up to infinity on its own. GMP's `mpq_get_d` and FLINT's `fmpq_get_d`, which calls it, truncate towards zero instead, so they differ from the nearest float in about half of all fractions; they also divide at the full length of the denominator, so they take time linear in it.
- `to_f32` rounds the rational to 24 bits the same way, once: rounding the `f64` again would be one off where the `f64` lands on a halfway point between two `f32`s.
- `FromStr` finds the first `/`, `.`, `e` or `E` and reads the parts as numbers are read (by divide and conquer from a few hundred digits). A fraction is reduced by the gcd of its parts, which is most of the time of reading one from 100 digits on (GMP's `mpq_set_str` alone takes 2.44 µs for parts of 1000 digits, with `mpq_canonicalize` 14.8 µs). A decimal `d₁…dₙ.f₁…fₘ e x` is the integer `d₁…dₙf₁…fₘ`, its trailing zeros dropped, times `10^(x − m)`, built as `5^k·2^k`. A negative power leaves `N/10^k` in lowest terms but for the factors 2 or 5 of `N`, which its last digit tells apart: `1, 3, 7, 9` share neither, an even digit only twos, which a shift removes, and `5` only fives, which one remainder by `5^27` counts (a gcd with `5^k` past 27). Parts and decimals whose digits make a word are read in registers, eight digits a step by a check and three products of the eight bytes at once, a fraction reduced by the binary gcd of two words and a decimal by its twos or fives alone.
- `Display` writes a fraction in words backward into one buffer of 42 bytes, blocks of eight digits as four independent pairs from a table, and gives the formatter the whole in one write when there is no width and no `+`; longer fractions of up to 320 characters are written on the stack too, so that the only allocation is the text the formatter keeps; `to_str_radix(10)` writes both parts into one vector.

## Continued fractions and approximations

- `continued_fraction` gives the regular continued fraction `[a0; a1, …, an]` of a rational as `Number`s in its byte order: `a0 = ⌊x⌋`, which may be negative, the other terms positive, and the last at least 2 unless the rational is an integer, the one term `[x]`. That is the canonical form, one for each rational. `from_continued_fraction` takes any list of terms back to the rational, the canonical form or not (`[1; 1]` is 2), in the byte order of `a0`; it panics on an empty list, or on a term after the first that is zero or negative. `convergents` gives every `pk/qk = [a0; …, ak]`, the last one the rational itself. `continued_fraction_into(&mut v)` writes the terms over the numbers a vector holds, in their vectors, and cuts it to the terms.
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

- The terms of a continued fraction are the quotients of Euclid's algorithm on the denominator and the remainder of the numerator. When the parts fit in 128 bits it runs in registers. Longer parts take the greatest common divisor's own reductions, which find many of Euclid's steps at once and apply their matrix to the whole numbers: hgcd2's matrices from the top two limbs (Lehmer's method), and from 600 limbs Möller's half gcd of the top third of the numbers, its matrix applied to the whole numbers. These reductions now tell of the steps they take, to a sink that the greatest common divisors leave empty, which costs them nothing (measured below); a step that they do not take in full is followed by the rest of its quotient on the same number, so the quotients of consecutive steps on one number add up to Euclid's. A continued fraction so costs what a greatest common divisor costs, `O(M(n)·log n)` for parts of `n` limbs, and the writing of its terms, about 0.58 a bit of the denominator for random numbers. Each term is a `Number` with a vector of its own, and its allocation and release are the rest of the cost, where FLINT keeps small terms in the words of its `fmpz`s and allocates nothing for them: `continued_fraction_into`, which writes over the vectors of the terms it is given, takes 130 ns for parts of one limb against `continued_fraction`'s 349 ns, and 15.7 µs against 38.6 µs for parts of 64 limbs. `Vec<Number>` leaves no way around one allocation a term; a term found in registers is written into a vector of room for 8 bytes by one store of the whole limb, cut to its significant bytes (the allocator gives a request of 1 to 16 bytes the same block).
- A rational whose parts both fit in 64 bits is read in place, from its words: `continued_fraction`, `convergents`, `limit_denominator`, `simplest_between` and the reconstructions run Euclid's algorithm on them in registers, and give a result whose parts fit in 64 bits as words, so that nothing is allocated but the terms of a continued fraction. `from_continued_fraction` multiplies the first run of terms of 64 bits in registers while the entries fit, which for short terms is all of them, and goes on from where it stopped on numbers otherwise; `simplest_between` of two ends in words takes their floors and ceilings in registers, and `limit_denominator` with a bound of 1 rounds in registers. Parts of up to 128 bits run in registers too: a quotient of Euclid's algorithm is found by at most three subtractions when it is 1, 2 or 3, two thirds of them by Gauss and Kuzmin's law (41.5 %, 17 % and 9.3 %), and by the library's division of `u128`s otherwise.
- `from_continued_fraction` multiplies the matrices `(ai, 1; 1, 0)`, whose product is `(pn, pn-1; qn, qn-1)`: runs of terms in registers while the entries fit in 64 bits, and the runs' matrices by a balanced tree of products, `O(M(n)·log n)` for a result of `n` limbs. `convergents` takes the recurrence `pk = ak·pk-1 + pk-2` from the terms: for parts of up to 128 bits in registers, where the terms are not written at all; for longer ones in registers while the entries fit in 127 bits, and then on limbs in place, `pk` written over the vector of `pk-2`, so that each convergent costs the two vectors of its own parts rather than four.
- `limit_denominator` and the reconstructions stop Euclid's algorithm at a bound: on the cofactors, which are the convergents' denominators (`limit_denominator`, on the denominator and the numerator's remainder), or on the remainders (the reconstruction, on `m` and `r mod m`, the first remainder at most `n`). They keep one row of the reduction matrix, the cofactors of the second number, and the bound gives a length to which the numbers can be reduced without passing it, as `|t_i|·r_(i-1) ≤ A` ties the cofactors to the remainders: the half gcd takes them there, on top parts sized so that it never goes below that length, and the row is multiplied by each of its matrices. The last steps are hgcd2's matrices tried against the bound, and when one passes it, its steps one at a time from where it started. Then `limit_denominator` chooses the convergent or the semiconvergent by one comparison of the remainders, `(2j·Q_S + Q_L)·S ≤ Q_S·L`, and the reconstruction checks `|t| ≤ d` and `gcd(r_i, t) = 1`, Wang's conditions. Both are `O(M(n)·log n)`.
- `simplest_between` goes down the Stern–Brocot tree from both ends at once, without ordering them first: the floors by one division each, an integer between them being the answer; then the terms of the fractional parts in step, until they part. There the end whose term is the smaller, or whose expansion ends, is the lower one, and the rational takes the common terms and that end's term, plus one unless the expansion ends there. The common terms are found by the reductions of the longer end, without the last step when it is not taken in full: a reduction is kept when the other end's pair stays positive under its matrix with its larger number in the same place, which is when its quotients are the same. From 1400 limbs the half gcd on the top half of the longer end goes first, on shorter tops where the ends part within it (twice the last top kept, then halved); then hgcd2's matrices, and in the one where the ends part its steps one at a time. The result comes out of the matrix of the common steps. Past 600 limbs of common terms left to Lehmer's method, both expansions are computed in full and compared, `O(M(n)·log n)`.

## Bernoulli and harmonic numbers

`Rational::bernoulli(n, order)` is the Bernoulli number `B_n`, `Rational::bernoulli_numbers(n, order)` the vector `B_0, B_1, ..., B_n`, and `Rational::harmonic(n, order)` the harmonic number `H_n = 1 + 1/2 + ... + 1/n`.

- `B_0 = 1`, `B_1 = −1/2`, `B_2 = 1/6`, `B_4 = −1/30`, and `B_n = 0` for every odd `n > 1`: the numbers of `x / (e^x − 1) = Σ B_n x^n / n!`, the convention of PARI, FLINT and Mathematica. The other convention, `B_1 = +1/2` (of `x / (1 − e^−x)`), differs only there: it is `-Rational::bernoulli(1, o)`, or `(−1)^n B_n` for any `n`. `H_0 = 0`.
- Each result is in lowest terms, both parts in the byte order `order`, in the shortest form. The indices are `u64`s; nothing panics but when a result or its working memory cannot be allocated: `B_n` has about `n log2(n / (2πe))` bits (1.25 M for `n = 10^5`, 15.8 M for `10^6`), `B_0, ..., B_n` together about `n² log2(n / 17) / 4` bits (27 MB for `n = 10^4`, 3.7 GB for `10^5`), and `H_n` about `1.44 n` in each part.
- Taking the vector costs far less than taking the numbers one by one: `B_0, ..., B_10000` take 382 ms, and `B_10000` alone 3.26 ms.

The method, for one Bernoulli number:

- Up to `B_200` from tables: to `B_58` numerators in `i128`s (`B_60` is the first above `2^127`), then the 492 limbs of the numerators from `B_60` to `B_200` one after another, 4,646 bytes with their ends and denominators, generated by the exact recurrence `Σ_(k=0)^m C(m+1, k) B_k = 0` and checked against PARI's `bernfrac` and FLINT's `bernoulli_fmpq_ui`. Those whose parts fit in 64 bits, `B_0` to `B_34` and `B_38`, and the zeros, are written as words in the caller: the lookup is inlined and the rest is out of line, so that it saves no registers and allocates nothing.
- Beyond, `|B_n| = 2 n! ζ(n) / (2π)^n`, as PARI's `bernfrac` and FLINT's `bernoulli_fmpq_ui` compute it. The denominator `d` is von Staudt and Clausen's: the product of the primes `p` with `p − 1` dividing `n`, from the divisors of `n`. With `K = 2 d n!` exactly and `U` a bound of the bits of the numerator `|B_n| d`, the quotient `K ζ(n) / (2π)^n` is computed to `w = U + 8` bits relatively and rounded: `π` from the 16,384 bits of the partition numbers' table or their Chudnovsky series, `(2π)^n` by squares and products cut to `w + log2 n + 6` bits (an early error is raised to the rest of the power), and `1/ζ(n) = Π_p (1 − p^−n)` in fixed point over the primes with `p^n < 2^w`, each factor `z ← z − z / p^n` with `z` and `p^n` cut to the bits `w − n log2 p` that it moves the product by and `log2 n + 8` more, `p^n` exact while it is short. The bounds of the errors add up to less than `2^−w` relatively, below `1/2^8` of a unit. The products, squares and quotients reuse a few vectors for the whole computation, the quotients by divisors of up to 8 limbs by Knuth's algorithm D in them (each step a division of two limbs by one by the reciprocal of the divisor's top limb), longer ones on numbers: a short number allocates a handful of times rather than at every step.
- From `n = 8000`, as FLINT's `_bernoulli_fmpq_ui_multi_mod` does, a share of the numerator's bits, from 10 % at 8000 to 22 % from `10^5` (`α = 0.1 + 0.05 log2(n / 10^4)`), is taken modulo primes instead: the residues of `|B_n| d` modulo the primes from 5 up, by Harvey's method ("A multimodular algorithm for computing Bernoulli numbers", 2010), joined by the Chinese remainder theorem into `R` modulo their product `M`; the formula is then needed only within `M / 2^11`, `w = U − log2 M + 12` bits, and the result is `R` plus the multiple of `M` nearest to the formula's value less `R`. The Euler product's primes are those below `2^(w/n)`, so fewer bits for the formula also mean fewer primes.
- Modulo a prime `p ≥ 5` with `2^n ≠ 1`, Voronoi's congruence with `c = 2` is `(2^n − 1) B_n = n 2^(n−1) Σ_(j=1)^(p−1) j^(n−1) floor(2j / p)` modulo `p`, and `floor(2j / p)` is the first bit of the binary expansion of `j / p`. Taken with `p − j`, whose bit is the complement and whose power the opposite, the sum is `Σ_(j ∈ H) j^(n−1) (2 bit(j) − 1)` over half the units. Doubling `x` shifts the expansion of `x / p` by a bit and multiplies the power by `c = 2^(n−1)`, so `H` is taken as cosets `x_t ⟨2⟩` of the order `m` of 2 (half of each when `m` is even, `−1` then being `2^(m/2)`), `x_t = g^t` for a generator `g` of the cosets: per coset, the expansion of `x_t / p` 64 bits a step, `floor(y 2^64 / p)` by Granlund and Montgomery's product with `ceil(2^128 / p)` (exact, as `y 2^64 < 2^96`), and the bits weighted by tables of the sums of `c^j` over the bits of each chunk of a step — sixteen tables of 16 below `2^18`, eight of 256 above — times the powers `c^(64β)` of the steps; the chains of remainders in four lanes at once.

For the vector:

- Up to `B_200` from the tables, those in words written as words.
- Beyond, the formula from the top index down, as FLINT's `bernoulli_fmpq_vec_no_cache` does: `G_k = 2 k! / (2π)^k` from `G_(k+2)` by a product with `(2π)²` (cut to the limbs that count) and a division by the limb `(k + 2)(k + 1)`; `ζ(k) = (1 − 2^−k)^−1 Σ_(j odd) j^−k` from the powers `j^−k` in fixed point of whole limbs, each from `j^−(k+2)` by a product with the limb `j²` and a cut of the limbs no longer needed, in one pass that also adds it into the sum. The precisions are those of `|B_k| d_k` with `log2 n + 8` bits more, made to fall with `k` by taking the largest denominator up to each index; the bounds of the errors of `G` and of every power are kept in units of their last bits, and a number whose bound would not leave less than 1/4 would be computed alone, which the bounds do not let happen. Two products of the precision for each index, and linear passes over the powers.

For a harmonic number:

- Up to `H_88`, the last whose parts are below `2^128`, from a table made at compile time by Knuth's sums, to `H_46` written as words in the caller as the Bernoulli numbers in words are; up to `H_124`, `H_88` plus the remaining terms, summed in a `u128` while they fit and added as rationals, which costs less than the splitting's preparation for so few terms.
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

Measured on an Apple M4 Max (macOS 27.2) on 2026-11-26 between 06:25 and 06:52, with nothing else running: the areas of the release's final run were timed one after another, and every timed process ran alone on the machine. One thread; `a` as a user builds it, `cargo build --release` with the default profile (opt-level 3, no LTO, the default codegen units, no `target-cpu`), on Rust 1.100.0 stable, by the harness in the repository's [`bench/rational`](../../../bench/rational) (see [`bench/README.md`](../../../bench/README.md)). Each cell is the median of 21 samples (15 when a call passes 30 ms), each a batch of at least 1.2 ms over a pool of fresh random rationals (16,384 of them, or 16,384 limbs' worth for long parts, too many for the branch predictor to learn), the contenders alternating sample by sample in one process, the order turning with each sample; every reference's results were compared with `a`'s on its first inputs before the timing. The references: GMP 6.3.0's `mpq_t`, FLINT 3.6.0's `fmpq` and PARI 2.19.1 (Homebrew's builds, called from C drivers built at `-O3 -mcpu=native`, PARI with one thread), malachite-q 0.14.0, num-rational 0.4.2 (`BigRational` on num-bigint 0.4.8, and `Ratio<i64>` where the parts fit) and rug 1.30.0 on the same GMP. GMP and FLINT are called the C way, writing into a variable that exists, which allocates nothing (`mpq_add (r, a, b)`, `fmpz_fdiv_q (z, …)`, `mpq_get_str` into a buffer); `a`, malachite, num-rational and rug return new values. Each row gives the fastest reference that does the same work, its time and `a`'s time over it: below 1, `a` is faster. Not counted, as they do less: `mpq_set_str` without the reduction, PARI's Bernoulli numbers from its cache, `signum` as an `int` or an `Ordering`, the truncating `to_f64` of GMP, FLINT and rug (they differ from the nearest float for about half the inputs), `Number::try_from` as GMP's and FLINT's copy of the numerator with no check, and FLINT's `fmpq_mat_rref` for the null space. Parts: `< 2^16` below `2^16`, words up to 64 bits, otherwise of the limbs given; `s + 2` a part below `2^16` with one of two limbs.

### Arithmetic

`&a + &b` and the others with a new result, random parts and signs; `a += &b` and the others on a fresh copy of `a` (GMP and FLINT: `mpq_add (a, a, b)`), random parts and parts with common factors (the denominators share half their length for `+=`, each numerator half with the other's denominator for `*=`); a pool of 512 timed over and over, which the predictor learns; and a short operand with a long one:

| Operation | Parts | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `&a + &b` | `< 2^16` | 16.0 ns | FLINT | 24.5 ns | 0.65 |
|  | words | 75.4 ns | GMP | 80.6 ns | 0.94 |
|  | 2 limbs | 160 ns | GMP | 172 ns | 0.93 |
|  | 16 limbs | 2.78 µs | FLINT | 3.51 µs | 0.79 |
|  | 1024 limbs | 671 µs | GMP | 925 µs | 0.73 |
|  | 8192 limbs | 11.3 ms | FLINT | 17.1 ms | 0.66 |
| `&a * &b` | `< 2^16` | 20.2 ns | FLINT | 30.7 ns | 0.66 |
|  | words | 74.9 ns | GMP | 121 ns | 0.62 |
|  | 2 limbs | 260 ns | GMP | 282 ns | 0.92 |
|  | 16 limbs | 5.11 µs | malachite-q | 6.25 µs | 0.82 |
|  | 1024 limbs | 1.23 ms | GMP | 1.61 ms | 0.76 |
|  | 8192 limbs | 21.6 ms | FLINT | 32.0 ms | 0.68 |
| `&a / &b` | `< 2^16` | 17.9 ns | FLINT | 30.8 ns | 0.58 |
|  | words | 74.5 ns | GMP | 126 ns | 0.59 |
|  | 2 limbs | 260 ns | GMP | 288 ns | 0.90 |
|  | 1024 limbs | 1.23 ms | GMP | 1.63 ms | 0.76 |
| `a += &b` | `< 2^16` | 20.1 ns | FLINT | 23.6 ns | 0.85 |
|  | words | 78.1 ns | GMP | 81.4 ns | 0.96 |
|  | 2 limbs | 182 ns | GMP | 172 ns | 1.06 |
|  | 2 limbs | 231 ns | GMP | 201 ns | 1.15 |
|  | 64 limbs | 14.2 µs | malachite-q | 18.0 µs | 0.79 |
|  | 1024 limbs | 677 µs | GMP | 933 µs | 0.72 |
| `a *= &b` | `< 2^16` | 21.8 ns | FLINT | 31.6 ns | 0.69 |
|  | words | 77.0 ns | GMP | 122 ns | 0.63 |
|  | words | 32.9 ns | FLINT | 73.7 ns | 0.45 |
|  | 2 limbs | 282 ns | GMP | 282 ns | 1.00 |
|  | 1024 limbs | 1.21 ms | FLINT | 1.61 ms | 0.76 |
| `&a + &b`, pool of 512 | `< 2^16` | 9.75 ns | FLINT | 10.7 ns | 0.91 |
|  | words | 72.2 ns | GMP | 71.0 ns | 1.02 |
| `&a * &b`, pool of 512 | `< 2^16` | 13.9 ns | FLINT | 12.4 ns | 1.12 |
|  | words | 73.3 ns | GMP | 115 ns | 0.64 |
| `&a + &b`, mixed | s + 2 | 80.7 ns | GMP | 54.3 ns | 1.49 |
|  | 1 + 2 | 111 ns | GMP | 87.9 ns | 1.26 |
|  | 2 + 1 | 110 ns | GMP | 87.0 ns | 1.27 |
|  | 1 + 4 | 114 ns | GMP | 91.8 ns | 1.24 |
|  | 1 + 16 | 158 ns | GMP | 124 ns | 1.28 |
|  | s + 64 | 230 ns | GMP | 159 ns | 1.45 |
|  | 1 + 64 | 252 ns | GMP | 198 ns | 1.28 |
|  | 1 + 1024 | 2.17 µs | GMP | 1.80 µs | 1.20 |
| `&a * &b`, mixed | s + 2 | 101 ns | GMP | 73.3 ns | 1.38 |
|  | 1 + 2 | 163 ns | GMP | 136 ns | 1.20 |
|  | 1 + 64 | 319 ns | GMP | 278 ns | 1.14 |
|  | 1 + 1024 | 2.35 µs | GMP | 2.07 µs | 1.13 |
| `a += &b`, mixed | s + 2 | 83.3 ns | GMP | 56.1 ns | 1.49 |
|  | 2 + 1 | 129 ns | GMP | 89.1 ns | 1.45 |
|  | 1 + 64 | 258 ns | GMP | 202 ns | 1.28 |
|  | 1 + 1024 | 2.18 µs | GMP | 1.81 µs | 1.21 |

Comparisons, construction and the other operations (GMP and FLINT into a variable that exists: `mpz_fdiv_q`, `mpq_abs`, `mpq_neg`, `mpq_inv`, `mpq_set`, `mpz_pow_ui` of both parts, and FLINT's equivalents; `==` of random pairs, one in four equal; `hash` into a hasher that multiplies in a word at a time, the same for all; `floor_into` and the others into a `Number` that exists, against GMP and FLINT; the rounding of a numerator of the limbs given over a denominator of half as many):

| Operation | Parts | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `cmp` | `< 2^16` | 1.87 ns | num-rational `Ratio<i64>` | 5.04 ns | 0.37 |
|  | words | 1.87 ns | rug | 8.51 ns | 0.22 |
|  | 2 limbs | 9.57 ns | GMP | 11.2 ns | 0.86 |
|  | 2 limbs | 13.9 ns | GMP | 12.7 ns | 1.09 |
|  | 64 limbs | 9.27 ns | num-rational | 410 ns | 0.023 |
|  | 1024 limbs | 8.87 ns | malachite-q | 11.3 µs | 0.00079 |
| `==` | `< 2^16` | 1.20 ns | malachite-q | 1.38 ns | 0.88 |
|  | words | 1.23 ns | malachite-q | 1.38 ns | 0.89 |
|  | 2 limbs | 3.38 ns | malachite-q | 2.59 ns | 1.30 |
|  | 8 limbs | 4.95 ns | malachite-q | 2.79 ns | 1.77 |
|  | 1024 limbs | 103 ns | malachite-q | 84.5 ns | 1.21 |
| `Rational::new` | `< 2^16` | 38.3 ns | FLINT | 21.3 ns | 1.80 |
|  | words | 65.7 ns | GMP | 57.3 ns | 1.15 |
|  | 8 limbs | 1.20 µs | malachite-q | 1.47 µs | 0.81 |
|  | 1024 limbs | 583 µs | FLINT | 742 µs | 0.79 |
| `Rational::from` | `i64` of 63 bits | 1.25 ns | num-rational | 1.34 ns | 0.93 |
| `Rational::from` | `i128` of 127 bits | 16.3 ns | malachite-q | 9.97 ns | 1.64 |
| `clone` | `< 2^16` | 2.08 ns | malachite-q | 1.47 ns | 1.41 |
|  | words | 2.09 ns | malachite-q | 1.47 ns | 1.43 |
|  | 2 limbs | 19.9 ns | GMP | 2.89 ns | 6.89 |
|  | 64 limbs | 50.2 ns | FLINT | 16.9 ns | 2.98 |
| `into_parts` | words | 19.3 ns | num-rational | 5.15 ns | 3.74 |
|  | 8 limbs | 21.8 ns | num-rational | 20.9 ns | 1.04 |
| `is_integer` | words | 0.760 ns | malachite-q | 0.800 ns | 0.96 |
|  | 2 limbs | 2.28 ns | num-rational | 0.760 ns | 3.00 |
| `floor` | `< 2^16` | 8.84 ns | FLINT | 1.95 ns | 4.53 |
|  | words | 8.82 ns | malachite-q | 10.1 ns | 0.88 |
|  | 2 / 1 limbs | 28.7 ns | GMP | 14.8 ns | 1.94 |
|  | 8 / 4 limbs | 61.2 ns | GMP | 49.6 ns | 1.23 |
|  | 1024 / 512 limbs | 33.0 µs | rug | 37.1 µs | 0.89 |
| `floor_into` | `< 2^16` | 2.72 ns | FLINT | 1.95 ns | 1.39 |
|  | words | 2.75 ns | GMP | 14.9 ns | 0.18 |
|  | 2 / 1 limbs | 21.4 ns | GMP | 14.8 ns | 1.44 |
|  | 1024 / 512 limbs | 33.0 µs | GMP | 49.0 µs | 0.67 |
| `trunc` | `< 2^16` | 8.81 ns | FLINT | 1.71 ns | 5.17 |
|  | words | 8.84 ns | malachite-q | 7.20 ns | 1.23 |
|  | 2 / 1 limbs | 29.0 ns | GMP | 9.81 ns | 2.95 |
| `round` | `< 2^16` | 8.85 ns | FLINT | 7.29 ns | 1.21 |
|  | words | 8.82 ns | FLINT | 28.6 ns | 0.31 |
|  | 8 / 4 limbs | 84.2 ns | FLINT | 66.0 ns | 1.28 |
|  | 1024 / 512 limbs | 41.7 µs | rug | 49.1 µs | 0.85 |
| `round_ties_even` | words | 9.22 ns | malachite-q | 17.3 ns | 0.53 |
|  | 1024 / 512 limbs | 41.6 µs | malachite-q | 71.2 µs | 0.58 |
| `fract` | words | 2.85 ns | num-rational | 17.1 ns | 0.17 |
|  | 8 / 4 limbs | 87.6 ns | rug | 85.9 ns | 1.02 |
| `abs` | `< 2^16` | 2.89 ns | GMP | 2.50 ns | 1.16 |
|  | words | 2.93 ns | GMP | 2.90 ns | 1.01 |
|  | 2 limbs | 20.6 ns | GMP | 2.90 ns | 7.10 |
| `-x` | words | 2.19 ns | GMP | 2.46 ns | 0.89 |
|  | 2 limbs | 19.7 ns | GMP | 2.90 ns | 6.79 |
| `recip` | words | 2.68 ns | malachite-q | 3.37 ns | 0.80 |
|  | 2 limbs | 22.4 ns | GMP | 5.47 ns | 4.09 |
| `signum` | words | 1.46 ns | num-rational | 2.30 ns | 0.64 |
| `pow(3)` | `< 2^16` | 5.05 ns | malachite-q | 5.34 ns | 0.95 |
|  | words | 35.9 ns | GMP | 22.0 ns | 1.63 |
|  | 8 limbs | 201 ns | GMP | 147 ns | 1.36 |
| `pow(-7)` | `< 2^16` | 29.6 ns | GMP | 20.1 ns | 1.47 |
|  | words | 65.1 ns | GMP | 36.8 ns | 1.77 |
|  | 8 limbs | 651 ns | GMP | 663 ns | 0.98 |
| `hash` | `< 2^16` | 4.18 ns | malachite-q | 1.51 ns | 2.77 |
|  | words | 4.67 ns | malachite-q | 1.51 ns | 3.10 |

With primitive integers and `Number`s (FLINT: `fmpq_add_si`, `fmpq_mul_si`, `fmpq_add_fmpz`, `fmpq_mul_fmpz`, `fmpq_cmp_si`), and sums and products of many terms, in time a term, by `+=` (GMP and FLINT: `mpq_add (h, h, t)`) and by `Iterator::sum` and `product` (GMP and FLINT: the same binary-counter tree in C):

| Operation | Parts | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `&a + u32` | `< 2^16` | 4.04 ns | FLINT | 6.61 ns | 0.61 |
|  | words | 22.4 ns | FLINT | 22.0 ns | 1.02 |
| `&a * i64` | `< 2^16` | 11.5 ns | FLINT | 16.5 ns | 0.69 |
|  | words | 69.9 ns | FLINT | 66.0 ns | 1.06 |
| `&a * &n` | `< 2^16` | 19.8 ns | FLINT | 18.0 ns | 1.10 |
|  | words | 72.9 ns | rug | 91.3 ns | 0.80 |
| `a.cmp(i32)` | words | 1.67 ns | rug | 5.84 ns | 0.28 |
| `a == i64` | words | 1.10 ns | malachite-q | 2.78 ns | 0.39 |

| Operation | Terms | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `a += t` | `H_1000` | 112 ns | GMP | 94.6 ns | 1.18 |
|  | `H_10000` | 400 ns | rug | 430 ns | 0.93 |
|  | `H_100000` | 3.14 µs | GMP | 3.68 µs | 0.85 |
|  | 10,000 terms of 16 bits | 1.00 µs | rug | 949 ns | 1.06 |
|  | 10,000 terms of 32 bits | 4.92 µs | GMP | 4.64 µs | 1.06 |
| `Iterator::sum` | `H_1000` | 46.2 ns | FLINT | 52.0 ns | 0.89 |
|  | `H_10000` | 126 ns | FLINT | 146 ns | 0.87 |
|  | `H_100000` | 264 ns | FLINT | 332 ns | 0.80 |
|  | 10,000 terms of 16 bits | 175 ns | FLINT | 212 ns | 0.83 |
|  | 10,000 terms of 32 bits | 708 ns | FLINT | 928 ns | 0.76 |
| `a *= t` | 10,000 terms of 16 bits | 1.08 µs | GMP | 1.03 µs | 1.05 |
| `Iterator::product` | 10,000 terms of 16 bits | 230 ns | FLINT | 297 ns | 0.77 |

- Every one of the 36 cells of `+`, `−`, `·` and `/` with a new result is faster than every reference: 0.58 to 0.94 of the fastest one's time, short parts in registers at 16.0 ns for `&a + &b` against FLINT's 24.5 ns, and parts of 8,192 limbs (64 KiB) at 0.66 to 0.68. In place, 34 of 36 cells are within 1.05.
- A short operand with a long one is where GMP is faster, in every one of its 24 cells (1.13 to 1.49); rug, GMP with a new result each time, takes 0.86 to 1.06 of `a`'s time from 64 limbs (see [below](#where-the-references-are-faster)).
- A comparison of rationals in words takes 1.87 ns without branches, and of long parts 0.00050 to 0.023 of the fastest reference's time, by the bounds from the top bits (8.87 ns at 1,024 limbs against malachite-q's 11.3 µs).
- Sums of many terms by `Iterator::sum` and `product` take 0.76 to 0.89 of FLINT's time in the same tree; term by term by `+=`, 0.85 to 1.18 of the fastest reference's, behind at `H_1000` (1.18).

### Conversions

`to_f64` and `to_f32` correctly rounded, against malachite-q's `f64::rounding_from(_, Nearest)` and num-rational's `to_f64` (GMP's `mpq_get_d`, FLINT's `fmpq_get_d` and rug truncate, and are not counted); floats to rationals, exactly, against GMP's `mpq_set_d` into an `mpq_t` that exists and a new one, malachite-q, num-rational and rug; text both ways in lowest terms, against `mpq_set_str` with `mpq_canonicalize` and `fmpq_set_str` with `fmpq_canonicalise`, and `mpq_get_str` and `fmpq_get_str` into a buffer the caller has; decimals against malachite-q's `from_sci_string`, the only reference that reads them:

| Operation | Size | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `to_f64` | 16 bits | 1.45 ns | num-rational | 8.18 ns | 0.18 |
|  | 64 bits | 4.85 ns | num-rational | 35.3 ns | 0.14 |
|  | 128 bits | 11.5 ns | malachite-q | 95.6 ns | 0.12 |
|  | 4,096 bits | 13.6 ns | malachite-q | 212 ns | 0.064 |
|  | 65,536 bits | 14.7 ns | malachite-q | 1.43 µs | 0.010 |
|  | 65,536 / 64 bits | 3.31 ns | malachite-q | 6.89 ns | 0.48 |
| `to_f32` | 16 bits | 1.44 ns | malachite-q | 30.9 ns | 0.047 |
|  | 64 bits | 4.64 ns | malachite-q | 56.5 ns | 0.082 |
|  | 4,096 bits | 12.8 ns | malachite-q | 212 ns | 0.060 |
| `Rational::try_from(f64)` | `[2^-11, 2^12)` | 1.92 ns | malachite-q | 7.94 ns | 0.24 |
|  | `[1e-10, 1e10]` | 10.7 ns | GMP | 8.71 ns | 1.22 |
|  | any exponent | 34.9 ns | GMP | 6.61 ns | 5.28 |
| `Rational::try_from(f32)` | any `f32` | 19.0 ns | malachite-q | 16.4 ns | 1.16 |
| `i128::try_from(&r)` | 16 bits | 0.980 ns | rug | 1.43 ns | 0.69 |
|  | 64 bits | 0.990 ns | rug | 1.46 ns | 0.68 |
|  | 127 bits | 2.24 ns | rug | 1.52 ns | 1.48 |
| `FromStr`, fraction | 16 bits | 19.9 ns | malachite-q | 36.0 ns | 0.55 |
|  | 64 bits | 69.6 ns | malachite-q | 121 ns | 0.57 |
|  | 100 digits | 866 ns | GMP | 1.12 µs | 0.78 |
|  | `10^4` digits | 241 µs | FLINT | 332 µs | 0.73 |
|  | `10^6` digits | 124 ms | malachite-q | 198 ms | 0.62 |
| `to_string`, fraction | 16 bits | 29.1 ns | FLINT | 14.7 ns | 1.97 |
|  | 64 bits | 35.4 ns | FLINT | 45.5 ns | 0.78 |
|  | 100 digits | 183 ns | FLINT | 144 ns | 1.27 |
|  | `10^4` digits | 84.3 µs | FLINT | 107 µs | 0.79 |
|  | `10^6` digits | 31.8 ms | FLINT | 40.2 ms | 0.79 |
| `to_str_radix(10)` | 16 bits | 24.9 ns | FLINT | 14.7 ns | 1.69 |
|  | 100 digits | 152 ns | FLINT | 142 ns | 1.07 |
|  | `10^6` digits | 31.9 ms | FLINT | 40.0 ms | 0.80 |
| `FromStr`, decimal | 4 digits | 11.2 ns | malachite-q | 68.1 ns | 0.17 |
|  | 19 digits | 27.0 ns | malachite-q | 128 ns | 0.21 |
|  | 10 digits, exponent −9 to 9 | 19.6 ns | malachite-q | 96.6 ns | 0.20 |
|  | 1000 digits | 1.11 µs | malachite-q | 7.74 µs | 0.14 |
|  | `10^6` digits | 8.67 ms | malachite-q | 93.9 ms | 0.092 |
| `to_str_radix(2)` | 16 bits | 32.4 ns | GMP | 41.2 ns | 0.79 |
|  | 64 bits | 38.7 ns | GMP | 96.0 ns | 0.40 |
|  | 128 bits | 68.9 ns | GMP | 166 ns | 0.41 |
|  | 4,096 bits | 433 ns | num-rational | 3.06 µs | 0.14 |
| `from_str_radix(_, 2)` | 16 bits | 22.0 ns | malachite-q | 48.2 ns | 0.46 |
|  | 128 bits | 284 ns | GMP | 437 ns | 0.65 |
|  | 4,096 bits | 15.5 µs | FLINT | 23.7 µs | 0.65 |
| `to_str_radix(7)` | 16 bits | 36.0 ns | GMP | 39.8 ns | 0.91 |
|  | 64 bits | 59.5 ns | FLINT | 104 ns | 0.57 |
|  | 128 bits | 172 ns | GMP | 121 ns | 1.41 |
|  | 4,096 bits | 5.71 µs | FLINT | 7.11 µs | 0.80 |
| `from_str_radix(_, 7)` | 16 bits | 17.9 ns | GMP | 61.4 ns | 0.29 |
|  | 128 bits | 242 ns | GMP | 246 ns | 0.98 |
|  | 4,096 bits | 14.4 µs | FLINT | 19.2 µs | 0.75 |
| `to_str_radix(16)` | 16 bits | 27.0 ns | GMP | 21.2 ns | 1.27 |
|  | 64 bits | 34.2 ns | GMP | 36.1 ns | 0.95 |
|  | 128 bits | 59.4 ns | GMP | 50.8 ns | 1.17 |
|  | 4,096 bits | 303 ns | num-rational | 976 ns | 0.31 |
| `from_str_radix(_, 16)` | 16 bits | 14.4 ns | malachite-q | 35.2 ns | 0.41 |
|  | 128 bits | 170 ns | GMP | 217 ns | 0.78 |
|  | 4,096 bits | 12.3 µs | malachite-q | 15.4 µs | 0.80 |

`Number::try_from(&r)` of an integer has no counted reference: GMP's and FLINT's copy of the numerator does not check that the denominator is 1. Against those that do, `a` takes 0.94 to 4.20 of the fastest one's time, malachite-q's at 16 and 64 bits, whose `Integer` keeps a word in place.

- `to_f64` takes 0.010 to 0.48 of the fastest correctly rounded reference's time at every size, and of parts in words 1.45 ns; a float in words becomes a rational in 1.92 ns, against malachite-q's 7.94 ns.
- Fractions are read at 0.55 to 0.84 of the fastest reference's time that reduces them, decimals at 0.092 to 0.26 of malachite-q's; text is written at 0.14 to 1.97 of the fastest reference's time, behind GMP and FLINT writing into a buffer at short parts.

### Continued fractions and approximations

Random parts of `n` limbs (or of 16 bits), positive, against FLINT (`fmpq_get_cfrac` with `fmpq_cfrac_bound` and the vector's allocation, `fmpq_set_cfrac`, `fmpq_simplest_between`, `fmpq_reconstruct_fmpz` and `fmpq_reconstruct_fmpz_2`), PARI (`contfrac`, `bestappr(x, D)`, `bestappr(Mod(r, m))`) and malachite-q (`continued_fraction` collected, `from_continued_fraction_ref`, `convergents` collected, `approximate`, `simplest_rational_in_closed_interval`, `reconstruct_ref`, `reconstruct_with_bounds_ref`); num-rational has none of these. `continued_fraction_into` writes the terms into a vector that exists, against FLINT's into a vector of `fmpz` that exists; `limit_denominator` with a bound of half the length or of 60 bits; `simplest_between` of two random ends, or of `x` and `x + 1/q` with `q` of 1.5 times the length; `reconstruct` a modulus of the length given:

| Operation | Size | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `continued_fraction` | 16 bits | 122 ns | malachite-q | 138 ns | 0.89 |
|  | 1 limb | 349 ns | malachite-q | 457 ns | 0.76 |
|  | 4 limbs | 2.36 µs | PARI | 4.63 µs | 0.51 |
|  | 16 limbs | 8.63 µs | FLINT | 7.19 µs | 1.20 |
|  | 64 limbs | 38.6 µs | FLINT | 25.2 µs | 1.53 |
|  | 256 limbs | 183 µs | FLINT | 125 µs | 1.46 |
|  | 1024 limbs | 1.07 ms | FLINT | 1.12 ms | 0.95 |
|  | 100000 limbs | 289 ms | FLINT | 438 ms | 0.66 |
| `continued_fraction_into` | 16 bits | 32.8 ns | FLINT | 247 ns | 0.13 |
|  | 4 limbs | 1.05 µs | FLINT | 5.90 µs | 0.18 |
|  | 64 limbs | 15.7 µs | FLINT | 22.2 µs | 0.71 |
|  | 256 limbs | 86.3 µs | FLINT | 114 µs | 0.76 |
|  | 100000 limbs | 252 ms | FLINT | 434 ms | 0.58 |
| `from_continued_fraction` | 16 bits | 28.7 ns | FLINT | 57.3 ns | 0.50 |
|  | 1 limb | 76.8 ns | FLINT | 282 ns | 0.27 |
|  | 16 limbs | 5.10 µs | FLINT | 6.05 µs | 0.84 |
|  | 1024 limbs | 730 µs | FLINT | 848 µs | 0.86 |
|  | 100000 limbs | 167 ms | FLINT | 237 ms | 0.70 |
| `convergents` | 16 bits | 77.1 ns | malachite-q | 330 ns | 0.23 |
|  | 1 limb | 209 ns | malachite-q | 1.21 µs | 0.17 |
|  | 16 limbs | 33.3 µs | malachite-q | 70.2 µs | 0.47 |
|  | 256 limbs | 1.84 ms | malachite-q | 3.92 ms | 0.47 |
| `limit_denominator` | 16 bits | 14.3 ns | malachite-q | 219 ns | 0.065 |
|  | 1 limbs, bound n/2 | 64.0 ns | malachite-q | 616 ns | 0.10 |
|  | 16 limbs, bound n/2 | 2.37 µs | PARI | 22.7 µs | 0.10 |
|  | 16 limbs, bound `2^60` | 937 ns | PARI | 2.42 µs | 0.39 |
|  | 4096 limbs, bound n/2 | 3.71 ms | malachite-q | 356 ms | 0.010 |
|  | 4096 limbs, bound `2^60` | 25.0 µs | PARI | 167 µs | 0.15 |
| `simplest_between` | 16 bits, random ends | 12.6 ns | FLINT | 43.6 ns | 0.29 |
|  | 4 limbs, random ends | 266 ns | FLINT | 230 ns | 1.15 |
|  | 4 limbs, close ends | 1.91 µs | FLINT | 1.85 µs | 1.03 |
|  | 256 limbs, random ends | 927 ns | FLINT | 14.9 µs | 0.062 |
|  | 256 limbs, close ends | 264 µs | FLINT | 363 µs | 0.73 |
|  | 16384 limbs, random ends | 59.5 µs | FLINT | 2.60 ms | 0.023 |
|  | 16384 limbs, close ends | 99.9 ms | FLINT | 522 ms | 0.19 |
| `reconstruct` | 16 bits | 14.9 ns | malachite-q | 27.2 ns | 0.55 |
|  | 4 limbs | 565 ns | FLINT | 662 ns | 0.85 |
|  | 64 limbs | 14.7 µs | FLINT | 19.7 µs | 0.74 |
|  | 100000 limbs | 260 ms | FLINT | 444 ms | 0.59 |
| `reconstruct_with_bounds` | 16 bits | 26.4 ns | malachite-q | 24.7 ns | 1.07 |
|  | 4 limbs | 852 ns | FLINT | 1.03 µs | 0.82 |
|  | 64 limbs | 13.1 µs | FLINT | 18.7 µs | 0.70 |
|  | 100000 limbs | 271 ms | FLINT | 471 ms | 0.58 |

Against the fastest reference that has the function: continued fractions 0.42 to 0.95 but from 16 to 256 limbs, where they are behind FLINT (1.20 to 1.53); into a vector that exists 0.12 to 0.76 of FLINT's; `from_continued_fraction` 0.27 to 0.87; `convergents` 0.17 to 0.55 of malachite-q's; `limit_denominator` 0.010 to 0.39 of PARI's or malachite-q's, both quadratic; `simplest_between` 0.023 to 1.15, behind only for ends of 4 limbs; the reconstructions 0.54 to 1.07.

### Bernoulli and harmonic numbers

Each cell over four indices in turn (`n` to `n + 6` by 2 for the Bernoulli numbers, `n` to `n + 3` for the harmonic numbers, `n` alone for the vectors), against FLINT's `bernoulli_fmpq_ui` and `_bernoulli_fmpq_ui`, `fmpq_harmonic_ui`, `bernoulli_fmpq_vec_no_cache` and `arith_bernoulli_number_vec`, PARI's `harmonic`, and malachite-q's `harmonic_number` (it has no Bernoulli numbers, and GMP and num-rational neither). PARI's `bernfrac` and `bernvec` keep the numbers they have computed and return them from that cache, so they do less and are not counted:

| Function | n | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `Rational::bernoulli` | 10 | 2.66 ns | FLINT | 3.67 ns | 0.72 |
|  | 60 | 25.0 ns | FLINT | 704 ns | 0.036 |
|  | 102 | 26.9 ns | FLINT | 1.07 µs | 0.025 |
|  | 300 | 4.41 µs | FLINT | 4.02 µs | 1.10 |
|  | 1,000 | 36.6 µs | FLINT | 34.6 µs | 1.06 |
|  | 3,000 | 340 µs | FLINT | 360 µs | 0.94 |
|  | 100,000 | 136 ms | FLINT | 252 ms | 0.54 |
|  | 1,000,000 | 7.63 s | FLINT | 11.8 s | 0.65 |
| `Rational::harmonic` | 10 | 2.49 ns | FLINT | 2.09 ns | 1.19 |
|  | 60 | 20.2 ns | FLINT | 328 ns | 0.062 |
|  | 89 | 118 ns | FLINT | 499 ns | 0.24 |
|  | 125 | 740 ns | FLINT | 760 ns | 0.97 |
|  | 1,000 | 5.13 µs | FLINT | 9.62 µs | 0.53 |
|  | 1,000,000 | 78.9 ms | malachite-q | 258 ms | 0.31 |
| `Rational::bernoulli_numbers` | 10 | 41.2 ns | FLINT | 69.8 ns | 0.59 |
|  | 100 | 934 ns | FLINT | 11.3 µs | 0.083 |
|  | 150 | 1.74 µs | FLINT | 29.3 µs | 0.059 |
|  | 300 | 58.7 µs | FLINT | 115 µs | 0.51 |
|  | 1,000 | 1.27 ms | FLINT | 1.58 ms | 0.80 |
|  | 10,000 | 382 ms | FLINT | 509 ms | 0.75 |

- Up to `B_200` the Bernoulli numbers come from tables: `B_102` in 26.9 ns against FLINT's 1.07 µs. Beyond, the formula: FLINT is faster at 300 and 1,000 (1.10 and 1.06), and from 3,000 `a` is, by the residues modulo primes (0.54 to 0.94; `B_(10^6)` in 7.63 s against 11.8 s).
- `H_n` from the table to `H_88`, then by the odd parts' binary splitting and the reduction without a greatest common divisor: 0.062 to 0.97 of FLINT's or malachite-q's time from `n = 60`; `H_10` is a table lookup on both sides (1.19). The vector `B_0, …, B_n` takes 0.059 to 0.80 of FLINT's time.

### Matrices

Against FLINT's defaults (`fmpq_mat_det`, `fmpq_mat_solve`, `fmpq_mat_inv`, `fmpq_mat_mul`, `fmpq_mat_rref` for the rank, `fmpz_mat_det`, `fmpq_mat_solve_fmpz_mat`, and `fmpz_mat_inv` with `fmpq_mat_set_fmpz_mat_div_fmpz` for `inverse_rational`) and PARI's (`det`, `RgM_solve`, `RgM_inv`, `gmul`, `rank`, `ker`), on random matrices of small integers (in −99..=99), small fractions (`p/q`, `|p| ≤ 99`, `1 ≤ q ≤ 99`), and 64-bit and 256-bit parts, below the thresholds of Dixon's lifting; the null space and the rank of matrices of rank `n − 2`:

| Operation | n, entries | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `Matrix<Rational>::det` | 2, small integers | 16.5 ns | FLINT | 17.5 ns | 0.94 |
|  | 8, small integers | 230 ns | FLINT | 1.13 µs | 0.20 |
|  | 2, small fractions | 23.0 ns | FLINT | 30.0 ns | 0.77 |
|  | 16, small fractions | 77.2 µs | FLINT | 66.7 µs | 1.16 |
|  | 2, 64-bit parts | 292 ns | FLINT | 481 ns | 0.61 |
|  | 8, 64-bit parts | 75.9 µs | FLINT | 69.6 µs | 1.09 |
|  | 16, 64-bit parts | 2.00 ms | FLINT | 4.19 ms | 0.48 |
|  | 4, 256-bit parts | 29.3 µs | FLINT | 26.6 µs | 1.10 |
|  | 8, 256-bit parts | 785 µs | FLINT | 710 µs | 1.10 |
| `Matrix<Rational>::solve` | 2, small integers | 44.1 ns | FLINT | 107 ns | 0.41 |
|  | 16, small integers | 12.0 µs | FLINT | 30.0 µs | 0.40 |
|  | 4, small fractions | 1.07 µs | FLINT | 1.27 µs | 0.84 |
|  | 2, 64-bit parts | 2.33 µs | FLINT | 2.17 µs | 1.08 |
|  | 16, 64-bit parts | 6.11 ms | FLINT | 9.13 ms | 0.67 |
|  | 8, 256-bit parts | 2.05 ms | FLINT | 2.17 ms | 0.95 |
| `Matrix<Rational>::inverse` | 2, small integers | 42.5 ns | FLINT | 57.0 ns | 0.74 |
|  | 8, small integers | 2.41 µs | FLINT | 5.44 µs | 0.44 |
|  | 2, small fractions | 91.3 ns | FLINT | 77.1 ns | 1.18 |
|  | 6, small fractions | 14.0 µs | FLINT | 13.4 µs | 1.04 |
|  | 8, 64-bit parts | 1.06 ms | FLINT | 1.46 ms | 0.72 |
|  | 8, 256-bit parts | 7.19 ms | FLINT | 9.58 ms | 0.75 |
| `&a * &b` | 2, small integers | 75.0 ns | PARI | 50.6 ns | 1.48 |
|  | 64, small integers | 73.1 µs | FLINT | 129 µs | 0.57 |
|  | 8, small fractions | 3.06 µs | FLINT | 5.54 µs | 0.55 |
|  | 32, small fractions | 617 µs | FLINT | 320 µs | 1.92 |
|  | 64, small fractions | 3.83 ms | FLINT | 2.15 ms | 1.78 |
|  | 16, 64-bit parts | 1.83 ms | FLINT | 2.14 ms | 0.85 |
|  | 4, 256-bit parts | 146 µs | FLINT | 137 µs | 1.07 |
| `Matrix<Rational>::rank` | 4, small integers | 147 ns | FLINT | 623 ns | 0.24 |
|  | 16, small fractions | 62.2 µs | FLINT | 116 µs | 0.54 |
| `Matrix<Rational>::null_space` | 4, small integers | 321 ns | PARI | 2.11 µs | 0.15 |
|  | 16, small fractions | 104 µs | PARI | 660 µs | 0.16 |
| `Matrix<Number>::det` | 2, small integers | 29.2 ns | FLINT | 5.81 ns | 5.03 |
|  | 4, small integers | 71.2 ns | FLINT | 40.4 ns | 1.76 |
|  | 8, small integers | 248 ns | FLINT | 856 ns | 0.29 |
|  | 2, 64-bit integers | 40.2 ns | FLINT | 16.1 ns | 2.50 |
|  | 4, 64-bit integers | 87.3 ns | FLINT | 227 ns | 0.38 |
|  | 16, 64-bit integers | 105 µs | FLINT | 82.7 µs | 1.27 |
| `Matrix<Number>::solve` | 2, small integers | 53.6 ns | FLINT | 52.0 ns | 1.03 |
|  | 16, small integers | 12.3 µs | FLINT | 28.7 µs | 0.43 |
|  | 2, 64-bit integers | 600 ns | FLINT | 372 ns | 1.61 |
|  | 16, 64-bit integers | 162 µs | FLINT | 202 µs | 0.80 |
| `inverse_rational` | 2, small integers | 38.6 ns | FLINT | 59.7 ns | 0.65 |
|  | 16, small integers | 86.4 µs | PARI | 143 µs | 0.60 |
|  | 2, 64-bit integers | 565 ns | FLINT | 376 ns | 1.51 |
|  | 16, 64-bit integers | 962 µs | PARI | 1.10 ms | 0.87 |

- Up to 16 × 16 the matrices of small integers run in registers: `Matrix<Rational>`'s determinants, solutions and inverses take 0.20 to 0.94 of the fastest reference's time, `Matrix<Number>::solve` and `inverse_rational` of them 0.37 to 1.03, and their products from 4 × 4 0.38 to 0.57.
- `Matrix<Number>::solve` at 16 × 16 is also timed by the run of [matrices](matrix.md#exactly), against FLINT's faster `fmpz_mat_solve_fflu` with a division, which this scoreboard does not have: there it takes 0.791 of that time for small integers and 1.11 for 64-bit ones (a loss, below).
- `rank` takes 0.20 to 0.54 of FLINT's `fmpq_mat_rref`, and `null_space` 0.070 to 0.16 of PARI's `ker`.
- Small fractions are at par with FLINT from 6 × 6 (0.93 to 1.04 at 6 × 6 and 8 × 8); long entries win from 16 × 16 (64-bit parts' determinant 0.48, inverse 0.44).

### Where the references are faster

Every case above 1.05 times the fastest same-work reference's time, by cause: 131 of the 584 cases of the scoreboard that have one. Where the fastest is GMP or FLINT writing into a variable that exists, the same-work references that return a new value, as `a` does, are given beside it.

- **`Matrix<Number>::solve` of 64-bit integers at 16 × 16** against FLINT's `fmpz_mat_solve_fflu` with a division (1.11, in the run of [matrices](matrix.md#exactly); not among the 131): the cause was not established.
- **A short operand with a long one** (all 24 `mixed` cells, 1.13 to 1.49; sums and products term by term, the same path, 1.06 to 1.18): a new rational is two vectors, and the word's gcd with the long part is a remainder by the powers of `B` modulo the word, prepared for each operation, where GMP runs `mpn_gcd_1`, `mpn_divexact_1` and `mpn_mul_1` on its limbs in place; rug, GMP with a new value each time, takes 0.86 to 1.06 of `a`'s time from 64 limbs.
- **Short arithmetic at a few cells** (1.06 to 1.15: `a += &b` of 2 limbs against GMP, 1.06 random and 1.15 with denominators sharing more than a word, where the sum leaves the 128-bit path; `&a * &b` of short parts on the trained pool against FLINT, 1.12; `&a * &n` and `&a * i64` against FLINT into a variable, 1.10 and 1.06): the cause of the others was not established.
- **`==` of long parts** against malachite-q (1.13 to 1.77): the four parts are compared through `Number`'s `PartialEq`, which allows for both byte orders, where malachite-q compares the lengths and then the limbs; **`cmp` of 2-limb parts of one length** against GMP (1.09): where the bounds from the top 63 bits do not decide, the products are formed on limbs.
- **New values that are vectors** (`clone` of 2 to 64 limbs 2.98 to 6.89 against GMP and FLINT copying into a variable that exists; `into_parts` of words 3.72 to 3.74 against num-rational, whose `BigUint` keeps a limb in place; `Rational::new` of short parts and words 1.15 to 1.80, which takes two `Number`s and releases their vectors when the result is in words, where GMP and FLINT reduce in a variable that stays; `Rational::from` of a 127-bit `i128` 1.64, two vectors against malachite-q's one): `Number` has no form that keeps a word in place. `clone` of rationals in words (1.41 to 1.43 against malachite-q) and `is_integer` of long parts (2.25 to 3.00, reading whether the denominator is 1 through `Number`) are small fixed costs. Against malachite-q, which returns new values too, `a` takes 1.02 to 1.03 of its time for the clones of 2 and 8 limbs.
- **Rounding to integers** (`floor`, `ceil` and `trunc` of short parts 4.53 to 5.17 against FLINT's word in an `fmpz` that exists, `trunc` of words 1.23 against malachite-q's `Integer` in place; of 2 and 8 limbs 1.23 to 2.95 against GMP into a variable; `round` 1.21 to 1.28 against FLINT's `fmpz_ndiv_qr` into variables): the result is a new `Number`, a vector. Into a `Number` that exists (`floor_into` and the others) the short parts are 1.23 to 1.41 of FLINT's time and the parts of 2 and 8 limbs 1.31 to 2.23 of GMP's, the quotient on limbs written back as bytes; not narrowed further.
- **`abs`, `-x` and `recip` of long parts** (4.09 to 7.10 against GMP into a variable with room): two new vectors; against malachite-q, which returns a new value, `a` takes 0.98 to 1.12 of its time at 2 limbs. `abs` of short parts (1.16) is a call against GMP's inline copy. **`pow`** (1.36 to 1.77 against GMP's `mpz_pow_ui` of both parts into a variable): square-and-multiply on numbers past the words, each step a new vector, where GMP sizes one buffer. **`hash` of short parts and words** (2.77 to 3.10 against malachite-q): a word is hashed as the `Number` of its value is (its sign, its length and its significant bytes), the contract of the hash before #27, where malachite-q writes words.
- **Floats past 64 bits and integers out of rationals** (`Rational::try_from(f64)` 1.22 to 5.28 against GMP's `mpq_set_d` into an `mpq_t` that exists, 0.25 to 0.81 against it with a new `mpq_t`; `try_from(f32)` 1.16 against malachite-q): two vectors when a part passes 64 bits, where malachite-q allocates one for a part of one limb; `i128::try_from(&r)` of 127-bit integers (1.48 against rug) reads the bytes of the numerator's vector.
- **Writing short fractions** (`to_string` and `to_str_radix` of short parts and of 100 digits, 1.07 to 1.97, against FLINT's and GMP's `get_str` into a caller's buffer): the `String` a conversion returns, and for `to_string` the formatter's copy; against the references that return a new string too, `a` takes 0.20 to 0.71 of their time there. **`to_str_radix(7)` of 128-bit parts** (1.41): parts in words have a register path in every radix, 128-bit parts go through `Number`'s general radix conversion.
- **Continued fractions of 16 to 256 limbs** (1.20 to 1.53 against FLINT): one vector a term, as `Vec<Number>` asks; `continued_fraction_into`, which reuses the terms' vectors, takes 0.57 to 0.76 of FLINT's time there. **`simplest_between` of two ends of 4 limbs** (1.15): 256-bit numbers leave the 128-bit register path for the limbs' setup. **`reconstruct_with_bounds` of a 16-bit modulus** (1.07 against malachite-q): not established.
- **`B_300` and `B_1000`** (1.06 to 1.10 against FLINT): the zeta formula's fixed-point arithmetic at a few limbs, on vectors through kernels for any length, where FLINT runs on `mpn`; **`H_10`** (1.19, a table lookup on both sides): not established.
- **`Matrix<Number>::det` of a few rows** (small integers at 2 × 2 to 4 × 4 1.76 to 5.03, 64-bit integers at 2 × 2 2.50 and from 6 × 6 1.27 to 1.32), **its solutions and inverses of 64-bit integers** (1.08 to 1.61): orders 2 to 4 are cofactors in registers and the solution and the inverse of order 2 Cramer's rule, for entries of up to 64 bits, and what is left at 2 × 2 is the fixed cost around them (reading the `Number`s' bytes, the result's vectors), where FLINT's `fmpz` keeps small values inline; beyond, 64-bit entries pass the `2^61` bound of the elimination in registers, and Bareiss runs on limbs, a vector for each element.
  **`Matrix<Rational>` with long entries** (determinants of 64-bit parts at 6 × 6 and 8 × 8, of 256-bit parts at 4 × 4 and 8 × 8 and of small fractions at 16 × 16, the 2 × 2 solution of 64-bit parts and inverse of small fractions, the 4 × 4 product of 256-bit parts: 1.07 to 1.18): each element a vector run through kernels for any length, with the final reduction by a greatest common divisor, where FLINT has GMP's kernels of a few limbs. **Products of small fractions from 32 × 32** (1.78 to 1.92): the rows' multiples pass a word, so the cleared integers are long, and FLINT multiplies them on multimodular packed kernels; **the 2 × 2 product of small integers** (1.48 against PARI): the clearing that larger matrices amortise.


## Efficiency

- Time complexity: a sum or a product of parts of `n` limbs takes two or three greatest common divisors and four products of `n`-limb numbers, `O(M(n) log n)` with the half gcd for the longest; a comparison at most two products.
- Space complexity: `O(n)`, the parts and the intermediate products.
- `to_f64` and `to_f32`: `O(1)` but for about one fraction in 700 of random parts of more than 64 bits, `O(n)` for a denominator of `n` limbs, and always `O(1)` for parts in words; the conversion from a float `O(1)`, with no allocation when its parts fit in 64 bits and two vectors of at most 135 bytes otherwise.
- `FromStr` and `from_str_radix`: the reading of the parts, `O(M(n) log n)` for `n` digits, and for a fraction the gcd of the parts, in registers and with no allocation for parts of up to 64 bits; a decimal with a negative exponent at most a remainder by `5^27`, or a gcd with `5^k` when its digits end in more than 27 factors of 5; `to_string` and `to_str_radix` the writing of the parts, `O(M(n) log n)`.
- Time complexity: `B_n` in about `O(n² log² n)` word operations, the Euler product's primes up to `n / 17` at falling precisions and the residues' `O(p)` for each prime up to about `0.15 n log2 n`; `B_0, ..., B_n` in `O(n³ log n)` word operations for the powers' passes, plus two products of each number's bits for each index; `H_n` in `O(M(n log n) log n)` for the tree, `M` the time of a product, plus `O(M(n))` for the reduction.
- Space complexity: the result and a few times its length for `B_n`, the residues' tables of 1 KB or 8 KB; for the vector, the results, and at most `n / 34` powers of the top index's precision.
- `det()` and `rank()` of an `n × n` matrix: about `n³/3` steps, each two products and an exact division of integers as long as the minors, `O(n³ M(n·b))` for elements of `b` bits; `solve(&b)` with `k` right-hand sides about `n³/3 + n²·k` of them and `n·k` gcds, `inverse()` about `4n³/3` and `n²` gcds; the product `n³` products of integers and `n²` gcds. The large determinants and systems: `n³/3` products modulo a prime and about `2·L·n²` products of words, `L` the length of the determinant over 61 bits, twice for the solutions, `O(n³ + n³·b/61)` word operations. Space: the matrix of integers and the right-hand sides, a vector of limbs for each element beyond the registers; Dixon's lifting the factors modulo the prime, `n²` words, and the digits, `L·n·k` words.

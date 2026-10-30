[A](../../../README.md) › [Math](../../../README.md#math)

# Division

[Numbers](numbers.md) divide with `/` and `/=` for the quotient and `%` and `%=` for the remainder, each operand owned or borrowed, or a primitive integer (see [Traits and conversions](traits.md)), and with five methods: `a.div_rem(&b)` gives both at once, `a.div_euclid(&b)` and `a.rem_euclid(&b)` Euclid's quotient and remainder, and `a.div_floor(&b)` and `a.rem_floor(&b)` those of floor division. A [`Divisor`](#divisor) prepares a number to divide many others by, and [`div_exact`](#exact-division) divides by a number known to divide exactly; `is_multiple_of` tells whether it does.

- The rules are those of Rust's primitive integers. `a / b` is truncated toward zero, and `a % b` is what is left, `a - (a / b)·b`: it has the sign of `a`, unless it is zero, and a magnitude below that of `b`. So `-7 / 2` is `-3` and `-7 % 2` is `-1`, as for `i32`.
- `div_euclid` and `rem_euclid` are as for `i32`: the remainder is never negative, `0 <= r < |b|`, and the quotient is one further from zero than `a / b` when `a % b` is below zero: `-7 = 2·(-4) + 1`.
- `div_floor` and `rem_floor` round the quotient toward negative infinity, as Python's `//` and `%` and GMP's `mpz_fdiv_q` and `mpz_fdiv_r` do: the remainder has the sign of `b`, unless it is zero, so `-7` floor-divided by `2` is `-4` remainder `1`, and `7` by `-2` is `-4` remainder `-1`. For a positive `b` they are Euclid's; for a negative one they are computed as `-((-a).div_euclid(&b))` and `-((-a).rem_euclid(&b))`, with `-a` a view of `a` with the other sign, at the cost of Euclid's division.
- Dividing by zero panics, with the messages of `i32`: `attempt to divide by zero`, and for `%` `attempt to calculate the remainder with a divisor of zero`, at the caller's line.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says.
- The results are in the byte order of the dividend in every form, `&a / b` included, and in the shortest form: zero is `[0]`, never empty and never negative.
- A result is built where the other operators build theirs: in `a`'s vector for `a /= &b`, `a /= b`, `a / &b`, `a / b` and the same forms of `%`, which always has room for it, as neither the quotient nor the remainder is longer than the dividend (but for zero in an empty vector); in `b`'s for `&a / b` when it has room, and for `&a % b`, which always has room, as the remainder is shorter than `b`. Otherwise, and always for `&a / &b` and `&a % &b`, exactly one new vector is allocated; `div_rem` allocates two. Divisions whose working memory fits in 8 KiB allocate nothing else; longer ones take one temporary buffer, and one more for each product of 360 limbs or more, which the [multiplication](mul.md) computes by transforms (from divisors of about 720 limbs, whose divide and conquer takes such products); the longest, by Newton's method (below), allocate their working memory as they go: the inverse, the transforms of the inverse and of the divisor, and a buffer for each product. All are freed before the result is returned, but what a [`Divisor`](#divisor) keeps.
- A borrowed operand is left as it was.

The method is long division on limbs of 8 bytes, Knuth's Algorithm D, in O(n·m) for a dividend of n limbs and a divisor of m, and from 40 limbs in the divisor the divide-and-conquer method of Burnikel and Ziegler on the [multiplication](mul.md), in the time of a product: O(n^1.58) with Karatsuba's method, and less with Toom-Cook's from 120 limbs. The longest divisions, from 128 limbs in the divisor and 768 in the quotient, the quotient at least an eighth of the divisor, take Newton's method, an inverse of the divisor and the quotient from products by it, in the time of two to three products, O(n log n) with the multiplication's number-theoretic transforms. This implementation carries it out as follows:

- Registers up to 64 bytes: operands of at most 16 bytes are divided as `u128`s, by the hardware's division when both fit in 8 bytes; a dividend of up to 64 bytes by a divisor of up to 16 is read into limbs in registers and divided by the steps below, unrolled up to 32 bytes, and by a divisor of 17 to 32 bytes, three or four limbs, by the steps of the long division on limbs in arrays, with the products of the divisor's lower limbs inline; the results are written from registers, their length found without a branch, as whether a quotient's top limb is zero is as good as random.
- No hardware division: the divisor is shifted so that its top bit is set, and each limb of the quotient is estimated by a multiplication with the precomputed reciprocal of its top limbs, after Möller and Granlund ("Improved division by invariant integers", 2011): two limbs by one, exactly, with one correction that is a predicted branch; three limbs by two for the longer divisors, at most one too large, corrected by adding the divisor back.
- One limb: limb by limb, about 10 cycles a limb on an Apple M4, a step waiting for the remainder of the one before; from 20 limbs the limbs are folded in instead, by Möller's method of GMP's `mpn_div_qr_1n_pi1`, with `B² mod d`, so that a step waits for one multiplication and two additions, 7.6 cycles a limb. The remainder alone skips the quotient's half of the work.
- One limb, the remainder alone, from 80 bytes in the dividend (112 for a divisor above `(2^64 − 1)/9`, 65 for a prepared `Divisor`): eight limbs a step, from the powers `c_k = B^k mod d` for k from 1 to 9: with `R` the two limbs of the partial remainder and `u_7 … u_0` the next eight limbs, `R·B^8 + Σ u_k B^k ≡ R_1 c_9 + R_0 c_8 + Σ u_k c_k + u_0 (mod d)`, nine products and a limb whose sum is below `9 (d − 1)(B − 1) + B ≤ B²` for `d ≤ (2^64 − 1)/9`, two limbs again; a larger divisor keeps a third limb of partial remainder, at most 9, which takes `R_2 c_10` more, and the sum stays below `10 B²`. Seven products do not depend on the partial remainder, so a step waits for one multiplication and three additions while the core computes the others, as GMP's `mpn_mod_1s_4p` does with four limbs; the dividend is read in its own byte order with no copy, and the powers take four steps of a division one after another (`c_1` from the reciprocal by a multiplication, then `c_2`, `c_3` and `c_4`, `c_5` to `c_8`, `c_9`). A remainder of 1 KiB by 10 took 234 cycles, against 1,200 limb by limb before, and of 16 KiB 1,936 against 18,749: about 1 cycle a limb, where GMP's `mpz_tdiv_ui` takes 1.4; by a random limb, with the third limb, 16 KiB took 2,704 against 18,089. A `Divisor` keeps the powers.
- Longer divisors: each quotient limb times the divisor is subtracted from the partial remainder by a kernel in inline assembly, two carry chains a step as in the multiplication: `mul`, `umulh` and `sbcs` on aarch64, `mulx` and `sbb` on x86_64 with BMI2, portable Rust elsewhere and under Miri. The divisor's limbs start on 16 bytes, which the kernel reads in pairs: off by 8, some divisions took 1.5 times as long.
- Divide and conquer: a quotient of n limbs by a divisor of n is two halves, each a division of n limbs by the divisor's top n/2 followed by a product that corrects the partial remainder, as GMP's `mpn_dcpi1_div_qr` arranges it.
- Newton's method, after GMP's `mpn_mu_div_qr` and Brent and Zimmermann's "Modern Computer Arithmetic" (§3.4): the inverse `X = B^k + I` of the divisor's top `k` limbs plus one, `D'`, with `D'·X ≤ B^(2k) ≤ D'·(X + 2)` (`B = 2^64`), by Newton's iteration, each step doubling the limbs of the inverse of the top limbs before it. The quotient is computed in blocks of up to `k` limbs from the top, each block the top limbs of `W_t·X / B^k` for `W_t` the top `k` limbs of the partial remainder with the next limbs of the dividend: bounding what the limbs left out and the inverse's error can change shows that it is at most 9 too small, never too large, and then the partial remainder, below 10 times the divisor, is corrected by subtracting the divisor (the proofs are in the source).
- Products modulo `B^m − 1`: a product whose high part is known, as the product of the divisor by a block of the quotient is (the partial remainder left is below `10·B^n` for a divisor of n limbs), or whose result is short enough, is computed modulo `B^m − 1` for an `m` just above that range, which the transforms compute as a cyclic convolution at the cost of a product whose result has `m` limbs, half that of a full one; each step of the inverse takes one too, as the product of the top n limbs of the divisor by the inverse of their top h limbs is known to be within `2·B^n` of `B^(n+h)`. The inverse in each block's estimate and the divisor in each block's remainder are transformed once for all the blocks, and the products of a division share one table of twiddles. The number of blocks, which sets the length of the inverse, is the one of least estimated cost from the plans of the transforms: about the divisor's length for a longer quotient, and two or three blocks for a quotient as long as the divisor.
- The quotient alone by Newton's method: the last block is shorter than the inverse, and the limbs of its estimate below it are a guard, which shows, but for a chance of about `10/2^64` for random operands, that the error cannot change the block, whose remainder is then not computed; an exact multiple of the divisor, whose estimate often falls just short of a whole quotient, computes it.
- The quotient alone, much shorter than the divisor: only the top limbs of both operands are read and divided, with one guard limb below the quotient. Bounding what the lower limbs can change shows that the quotient is right unless the guard limb is 0 or 1, and then a product with the divisor decides between it and one less: for random operands a chance of 2 in 2^64.
- The quotient alone, otherwise: its top limbs exactly, and its last ones, up to one fewer than the divisor's n, from a divisor that drops its lowest limb at each step, as GMP's `mpn_sbpi1_div_q` arranges it, in about half the products. The products left out add up to less than k·B^(n-1), for the k steps that leave out any and B = 2^64, so the quotient is right when the top limb of the partial remainder left is at least k; otherwise those products are subtracted after all, and the quotient is one too large exactly when that takes the remainder below zero (the proof is in the source). A step whose partial remainder is too large for its shortened divisor, a chance of about 2 in 2^64 for random operands, gives up, and the quotient is computed in full. From 256 limbs in the divisor, the low limbs, about half the divisor's, come instead from the top limbs of both operands, as above, themselves divided the same way.

Measured on an Apple M4 Max as `a %= &b` and `a /= &b` with the operands in the byte order given, over 2 MiB of random operand pairs of the lengths given in bytes, against GMP 6.3's `mpz_tdiv_r(a, a, b)` and `mpz_tdiv_q(a, a, b)` and num-bigint 0.4's `a %= &b` and `a /= &b` (the median of 11 interleaved runs, in cycles per call):

| Bytes | `%`: LE | BE | GMP | num-bigint | `/`: LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 / 8 | 31 | 32 | 46 | 190 | 31 | 32 | 42 | 133 |
| 32 / 16 | 77 | 83 | 92 | 523 | 76 | 84 | 88 | 508 |
| 32 / 8 | 55 | 58 | 55 | 326 | 69 | 73 | 53 | 248 |
| 32 / 24 | 96 | 112 | 138 | 420 | 96 | 109 | 99 | 416 |
| 48 / 24 | 158 | 169 | 163 | 622 | 144 | 156 | 143 | 615 |
| 64 / 8 | 88 | 91 | 86 | 575 | 116 | 121 | 78 | 511 |
| 64 / 32 | 202 | 209 | 203 | 734 | 197 | 204 | 174 | 738 |
| 128 / 64 | 440 | 441 | 403 | 1298 | 367 | 369 | 335 | 1283 |
| 256 / 128 | 897 | 906 | 952 | 2805 | 704 | 713 | 729 | 2798 |
| 1024 / 512 | 6583 | 6628 | 7693 | 22,835 | 4401 | 4443 | 5148 | 22,811 |
| 4096 / 2048 | 57,333 | 57,319 | 68,870 | 208,016 | 44,356 | 44,311 | 52,977 | 208,326 |
| 8192 / 4096 | 168,514 | 168,664 | 204,837 | 611,646 | 135,862 | 136,272 | 155,851 | 612,049 |
| 4096 / 8 | 857 | 760 | 6044 | 35,777 | 4796 | 5006 | 6008 | 34,638 |
| 4096 / 1024 | 57,732 | 57,762 | 67,124 | 211,326 | 51,262 | 51,431 | 74,486 | 210,611 |
| 4096 / 3072 | 38,334 | 38,402 | 42,645 | 144,984 | 14,132 | 14,317 | 17,174 | 144,154 |

- Remainders take from 33% less to 9% more than GMP's up to 128 bytes, and 6–24% less from 256 bytes to 8 KiB. By one limb, from 256 bytes they take a fraction of the time of GMP's `mpz_tdiv_r`, which computes the quotient too (1 KiB by a random limb: 351 cycles against 1,495; 16 KiB: 2,704 against 23,887), and from 1 KiB about that of `mpz_tdiv_ui`, which does not (by 10: 1 KiB 234 against 222, 16 KiB 1,936 against 2,786).
- Quotients by longer divisors take 3–31% less than GMP's from 256 bytes to 8 KiB, and up to 128 bytes from 14% less to 13% more; by one limb, 30–50% more for dividends of 24 to 64 bytes, and 16–20% less from 1 KiB.
- Up to 8 KiB, the divisions above are by the schoolbook method and divide and conquer, unchanged by Newton's method; for the longest, see the table below.
- A quotient alone of an exact multiple ends in the compensation of the shrinking divisor, which adds about the products it left out: by divisors of 128 to 255 limbs that takes up to 1.4 times as long as a random dividend, about what GMP's method does; `div_exact` is the method for exact multiples (below).
- Against num-bigint, a division takes a third of its time or less, about 3–10 times faster.
- The x86_64 kernel has been tested for correctness under emulation only; its speed has not been measured.

The longest divisions, measured as above but as `&a % &b` and `&a / &b` on one pair of random operands, little-endian, the median of 7 interleaved runs (3 from 4 MiB), against the crate before Newton's method (divide and conquer), GMP 6.3's `mpz_tdiv_r(r, a, b)` and `mpz_tdiv_q(q, a, b)` into an initialised result, and num-bigint 0.4 up to 1 MiB, in cycles per call:

| Bytes | `%`: ours | before | GMP | num-bigint | `/`: ours | before | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 KiB / 8 KiB | 339,994 | 459,695 | 573,471 | 1,836,325 | 287,047 | 375,232 | 450,372 | 1,820,018 |
| 64 KiB / 32 KiB | 1,475,435 | 2,848,211 | 3,296,444 | 14,916,539 | 1,298,633 | 2,523,763 | 2,819,544 | 14,986,174 |
| 256 KiB / 128 KiB | 6,488,247 | 15,836,087 | 17,800,806 | 121,564,792 | 5,543,623 | 14,372,331 | 14,612,490 | 121,931,034 |
| 1 MiB / 512 KiB | 28,255,367 | 83,548,533 | 94,558,621 | 1,036,488,772 | 24,504,270 | 77,315,493 | 75,587,341 | 1,025,599,133 |
| 4 MiB / 2 MiB | 126,830,818 | 433,076,811 | 483,895,621 | – | 106,825,026 | 388,800,051 | 401,337,434 | – |
| 8 MiB / 4 MiB | 268,028,686 | 964,924,688 | 1,126,478,545 | – | 222,843,357 | 875,731,137 | 928,788,249 | – |
| 12 KiB / 4 KiB | 239,282 | 338,722 | 397,938 | – | 224,319 | 312,460 | 351,159 | – |
| 72 KiB / 8 KiB | 1,403,953 | 3,641,495 | 3,539,227 | – | 1,426,732 | 3,515,108 | 3,290,303 | – |
| 1 MiB / 4 KiB | 17,047,905 | 43,356,390 | 32,816,139 | – | 17,045,837 | 43,020,002 | 32,638,965 | – |
| 1 MiB / 16 KiB | 19,126,336 | 72,876,453 | 62,457,532 | – | 19,105,688 | 71,730,216 | 62,187,159 | – |
| 1 MiB / 64 KiB | 21,830,448 | 100,439,084 | 69,824,068 | – | 21,817,721 | 100,918,004 | 69,411,054 | – |
| 8 MiB / 64 KiB | 167,330,048 | 849,526,063 | 554,092,132 | – | 168,240,985 | 858,188,726 | 563,087,898 | – |
| 288 KiB / 256 KiB | 4,620,934 | 5,258,342 | 9,979,667 | – | 1,287,599 | 2,507,253 | 2,750,872 | – |

- Balanced divisions take 0.59 to 0.64 of GMP's time at 16 KiB by 8 KiB, 0.45 to 0.46 at 64 KiB, 0.30 to 0.32 at 1 MiB and 0.24 at 8 MiB, against 0.80 to 1.02 by divide and conquer before: GMP divides by Newton's method too, in about three of its products, and the multiplication's transforms take 0.3 to 0.5 of GMP's time. Divide and conquer spent a product at each level of its recursion; Newton's method spends about two and a half products of the divisor's length for a balanced division, inverse included.
- A dividend much longer than the divisor is divided in blocks of about the divisor's length by one inverse, with the inverse and the divisor transformed once: two products of the divisor's length a block, each skipping a transform, against a divide-and-conquer division a block before. By divisors of 4 KiB to 64 KiB, 0.30 to 0.52 of GMP's time, 2.5 to 5 times faster than before, when divide and conquer took 1.03 to 1.53 times GMP's; by 4 KiB, where GMP still divides by divide and conquer (its Newton's method takes divisors from about 1000 limbs on an Apple M1) and the transforms' advantage over its products is least, 0.52.
- A quotient shorter than 768 limbs, or than an eighth of the divisor, is computed as before, as Newton's method costs more there (by a divisor of 1 MiB, the remainder of a quotient of 8 KiB took 1.22 times as long by it): its product of the quotient by the divisor is that of divide and conquer, and its inverse costs more than divide and conquer's division of the top limbs. The quotient alone of such a shape comes from the top limbs of both operands, a balanced division, which takes Newton's method when it is long enough: of 288 KiB by 256 KiB, 0.47 of GMP's time (0.91 before), whose remainder, a quotient of an eighth of the divisor, takes 0.46 (0.53 before).
- Against num-bigint, whose division is quadratic, 5 to 40 times faster from 16 KiB to 1 MiB.
- The crossovers with divide and conquer were measured in steps over divisors of 64 to 4096 limbs and quotients of 384 to 32 times the divisor (see the source); at 768 limbs of quotient, the quotient alone takes 0.84 of the time of divide and conquer to 1.14 (by divisors of 128 limbs; 1.05 by 256), and the remainder 0.75 to 1.04.

## Divisor

`Divisor::new(&b)` prepares `b` to divide by: its magnitude in limbs, shifted, the reciprocal of its top limbs, and its sign. A number divided by it with `/`, `/=`, `%` and `%=` (the dividend owned or borrowed, the divisor borrowed), `d.div_rem(&a)`, `d.div_euclid(&a)` or `d.rem_euclid(&a)` gives what dividing by `b` gives, with the same rules and the same vectors, and skips reading the divisor, normalizing it and computing its reciprocal. That is most of the cost of a short division, measured as above with the dividend and divisor lengths given, in cycles; divisors of up to 32 bytes keep the limbs and reciprocal of the divisions in registers too:

| Bytes | `/` by `b` | by `Divisor` | `%` by `b` | by `Divisor` |
| --- | --- | --- | --- | --- |
| 16 / 8 | 29 | 23 | 30 | 18 |
| 32 / 8 | 63 | 42 | 48 | 28 |
| 64 / 16 | 158 | 92 | 149 | 86 |
| 64 / 32 | 193 | 142 | 202 | 141 |
| 128 / 64 | 362 | 277 | 438 | 344 |
| 1024 / 512 | 4342 | 4249 | 6696 | 6636 |

The first division by a `Divisor` that takes Newton's method, from 256 limbs in the divisor and in the quotient, computes the inverse of all the divisor's limbs and the transforms of the inverse and of the divisor, and the divisor keeps them, 10 to 11 times its own bytes in all: its later divisions take no inverse, a block of the divisor's length for each `n` limbs of quotient, with two products by transformed operands each, and their products skip a transform each. Measured as the long divisions above, `/` and `div_rem` by a prepared divisor, against `/` by the number and GMP's `mpz_tdiv_q` and `mpz_tdiv_qr`, in cycles:

| Bytes | `/` by `b` | by `Divisor` | `div_rem` by `Divisor` | GMP `/` | GMP `div_rem` |
| --- | --- | --- | --- | --- | --- |
| 16 KiB / 8 KiB | 287,047 | 152,393 | 150,042 | 450,372 | 575,549 |
| 64 KiB / 32 KiB | 1,298,633 | 666,993 | 668,348 | 2,819,544 | 3,427,923 |
| 1 MiB / 512 KiB | 24,504,270 | 13,720,809 | 13,996,526 | 75,587,341 | 92,426,857 |

The first such division, which prepares the divisor, costs more than a division by the number: with `Divisor::new`, 1.22 times as much at 8 KiB by 4 KiB and 1.36 times at 1 MiB by 512 KiB; from the second, a prepared divisor is ahead.

`Divisor::new` panics for zero, as dividing by it would.

## Exact division

`a.div_exact(&b)` is `a / b` for a `b` that divides `a`, by Hensel's method: the quotient is computed from the low end, each limb the one that makes the low limb of the partial remainder zero, `n_i·d⁻¹ mod 2^64`, exact, with no estimate to correct, and the partial remainder above the quotient's limbs is never needed. A quotient as long as the divisor takes about half the work of a division, and one much shorter only the divisor's lowest limbs; the zero bits at the low end of the divisor are shifted out of both first. Divisors of one limb, and divisions large enough for divide and conquer to be faster, go to the ordinary division. When `b` does not divide `a`, the result is a number in the shortest form, but not a meaningful one; it is never undefined behaviour.

`a.is_multiple_of(&b)` tells whether `b` divides `a`, as `u32::is_multiple_of` does: zero is a multiple of everything, and only zero is a multiple of zero. It allocates nothing, and tells apart without dividing a dividend shorter than the divisor or with fewer zero bits at the low end. Measured as above on exact multiples, `div_exact` against `/`, in cycles:

| Bytes of quotient / divisor | `div_exact` | `/` |
| --- | --- | --- |
| 64 / 64 | 217 | 457 |
| 256 / 256 | 861 | 2223 |
| 1024 / 1024 | 9810 | 27,323 |
| 2048 / 2048 | 38,444 | 62,391 |
| 4096 / 4096 | 156,020 | 185,675 |
| 256 / 2048 | 904 | 11,518 |
| 2048 / 256 | 11,581 | 17,763 |

## Examples

Rust usage: the rules of `i32`, and Euclid's:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(-7, ByteOrder::BigEndian);
  let b = Number::new_from_i32(2, ByteOrder::BigEndian);

  let q = &a / &b;
  assert_eq!(q.value, vec![3]);
  assert!(q.is_negative());

  let r = &a % &b;
  assert_eq!(r.value, vec![1]);
  assert!(r.is_negative());

  let (q, r) = a.div_rem(&b);
  assert!(q.is_negative() && r.is_negative());

  let q = a.div_euclid(&b);
  assert_eq!(q.value, vec![4]);
  assert!(q.is_negative());
  let r = a.rem_euclid(&b);
  assert_eq!(r.value, vec![1]);
  assert!(!r.is_negative());
}
```

`/=` and `%=` work in the dividend's own vector, and a dividend in the other byte order than the divisor gives its result in its own:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let mut n = Number::new_from_u64(1_000_000_007, ByteOrder::LittleEndian);
  let vector = n.value.as_ptr();
  n /= &Number::new_from_u16(1000, ByteOrder::BigEndian);
  assert_eq!(n.value, vec![0x40, 0x42, 0x0F]); // 1_000_000
  assert_eq!(n.value.as_ptr(), vector);

  let mut m = Number::new_from_u64(1_000_000_007, ByteOrder::BigEndian);
  m %= &Number::new_from_u16(1000, ByteOrder::LittleEndian);
  assert_eq!(m.value, vec![7]);
}
```

Long operands: `(256^1000 - 1) / (256^500 - 1) = 256^500 + 1`, with no remainder:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new(vec![0xFF; 1000], ByteOrder::BigEndian);
  let b = Number::new(vec![0xFF; 500], ByteOrder::BigEndian);
  let (q, r) = a.div_rem(&b);
  let mut expected = vec![0; 501];
  expected[0] = 1;
  expected[500] = 1;
  assert_eq!(q.value, expected);
  assert_eq!(r.value, vec![0]);
  assert!(a.is_multiple_of(&b));
  assert_eq!(a.div_exact(&b).value, q.value);
}
```

A `Divisor` for many divisions by the same number, here the digits of a number in base 1000:

```rust
use a::math::numbers::{ByteOrder, Divisor, Number};

fn main() {
  let thousand = Divisor::new(&Number::new_from_u16(1000, ByteOrder::BigEndian));
  let mut n = Number::new_from_u64(123_456_789, ByteOrder::LittleEndian);
  let mut groups = Vec::new();
  while n.value != vec![0] {
    groups.push((&n % &thousand).value);
    n /= &thousand;
  }
  assert_eq!(groups, vec![vec![0x15, 0x03], vec![0xC8, 0x01], vec![0x7B]]); // 789, 456, 123
}
```

Dividing by zero panics, as for `i32`:

```rust,should_panic
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_u8(1, ByteOrder::BigEndian);
  let _ = &a / &Number::new_from_u8(0, ByteOrder::BigEndian);
}
```

## Efficiency

- Time complexity: O(n·m) for a dividend of n limbs and a divisor of m below 40, and from there for balanced operands that of a product of n limbs, O(n^1.58) with Karatsuba's method, O(n^1.47) with Toom-3's and O(n^1.40) with Toom-4's, times log n for divide and conquer's levels once the products are by transforms; from 128 limbs in the divisor and 768 in the quotient, at least an eighth of the divisor, by Newton's method, a few products of the divisor's length for each block of about that many quotient limbs, O(n log n) for balanced operands; the quotient alone with about half the products of its last limbs below 256 limbs in the divisor, of one much shorter than the divisor in that of a division of its own length, and an exact quotient in O(q·m) less a triangle, whatever the signs and byte orders
- Space complexity: the vector of the result, unless it is built in an operand's vector; working memory of about three times the dividend's limbs, on the stack up to 8 KiB, one temporary buffer on the heap beyond, and one more for each product of 360 limbs or more, which the [multiplication](mul.md) computes by transforms; by Newton's method, buffers allocated as it goes, for the transforms and their twiddles above all: at most about 7.6 times the operands' bytes together for a balanced division (3.6 by divide and conquer), and 2.2 to 3 times for a dividend much longer than the divisor; a `Divisor` keeps its divisor's limbs, and after its first division by Newton's method its inverse and their transforms, 10 to 11 times the divisor's bytes in all
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; a borrowed operand is left as it was

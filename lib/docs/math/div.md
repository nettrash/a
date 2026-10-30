[A](../../../README.md) › [Math](../../../README.md#math)

# Division

[Numbers](numbers.md) divide with `/` and `/=` for the quotient and `%` and `%=` for the remainder, each operand owned or borrowed, and with three methods: `a.div_rem(&b)` gives both at once, and `a.div_euclid(&b)` and `a.rem_euclid(&b)` Euclid's quotient and remainder. A [`Divisor`](#divisor) prepares a number to divide many others by, and [`div_exact`](#exact-division) divides by a number known to divide exactly; `is_multiple_of` tells whether it does.

- The rules are those of Rust's primitive integers. `a / b` is truncated toward zero, and `a % b` is what is left, `a - (a / b)·b`: it has the sign of `a`, unless it is zero, and a magnitude below that of `b`. So `-7 / 2` is `-3` and `-7 % 2` is `-1`, as for `i32`.
- `div_euclid` and `rem_euclid` are as for `i32`: the remainder is never negative, `0 <= r < |b|`, and the quotient is one further from zero than `a / b` when `a % b` is below zero: `-7 = 2·(-4) + 1`.
- Dividing by zero panics, with the messages of `i32`: `attempt to divide by zero`, and for `%` `attempt to calculate the remainder with a divisor of zero`, at the caller's line.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says.
- The results are in the byte order of the dividend in every form, `&a / b` included, and in the shortest form: zero is `[0]`, never empty and never negative.
- A result is built where the other operators build theirs: in `a`'s vector for `a /= &b`, `a /= b`, `a / &b`, `a / b` and the same forms of `%`, which always has room for it, as neither the quotient nor the remainder is longer than the dividend (but for zero in an empty vector); in `b`'s for `&a / b` when it has room, and for `&a % b`, which always has room, as the remainder is shorter than `b`. Otherwise, and always for `&a / &b` and `&a % &b`, exactly one new vector is allocated; `div_rem` allocates two. Divisions whose working memory fits in 8 KiB allocate nothing else; longer ones take one temporary buffer, freed before the result is returned.
- A borrowed operand is left as it was.

The method is long division on limbs of 8 bytes, Knuth's Algorithm D, in O(n·m) for a dividend of n limbs and a divisor of m, and from 40 limbs in the divisor the divide-and-conquer method of Burnikel and Ziegler on the [multiplication](mul.md), in the time of a product: O(n^1.58) with Karatsuba's method, and less with Toom-Cook's from 120 limbs. This implementation carries it out as follows:

- Registers up to 64 bytes: operands of at most 16 bytes are divided as `u128`s, by the hardware's division when both fit in 8 bytes; a dividend of up to 64 bytes by a divisor of up to 16 is read into limbs in registers and divided by the steps below, unrolled up to 32 bytes, and by a divisor of 17 to 32 bytes, three or four limbs, by the steps of the long division on limbs in arrays, with the products of the divisor's lower limbs inline; the results are written from registers, their length found without a branch, as whether a quotient's top limb is zero is as good as random.
- No hardware division: the divisor is shifted so that its top bit is set, and each limb of the quotient is estimated by a multiplication with the precomputed reciprocal of its top limbs, after Möller and Granlund ("Improved division by invariant integers", 2011): two limbs by one, exactly, with one correction that is a predicted branch; three limbs by two for the longer divisors, at most one too large, corrected by adding the divisor back.
- One limb: limb by limb, about 10 cycles a limb on an Apple M4, a step waiting for the remainder of the one before; from 20 limbs the limbs are folded in instead, by Möller's method of GMP's `mpn_div_qr_1n_pi1`, with `B² mod d`, so that a step waits for one multiplication and two additions, 7.6 cycles a limb. The remainder alone skips the quotient's half of the work.
- Longer divisors: each quotient limb times the divisor is subtracted from the partial remainder by a kernel in inline assembly, two carry chains a step as in the multiplication: `mul`, `umulh` and `sbcs` on aarch64, `mulx` and `sbb` on x86_64 with BMI2, portable Rust elsewhere and under Miri. The divisor's limbs start on 16 bytes, which the kernel reads in pairs: off by 8, some divisions took 1.5 times as long.
- Divide and conquer: a quotient of n limbs by a divisor of n is two halves, each a division of n limbs by the divisor's top n/2 followed by a product that corrects the partial remainder, as GMP's `mpn_dcpi1_div_qr` arranges it.
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
| 4096 / 8 | 4538 | 4693 | 6007 | 34,569 | 4796 | 5006 | 6008 | 34,638 |
| 4096 / 1024 | 57,732 | 57,762 | 67,124 | 211,326 | 51,262 | 51,431 | 74,486 | 210,611 |
| 4096 / 3072 | 38,334 | 38,402 | 42,645 | 144,984 | 14,132 | 14,317 | 17,174 | 144,154 |

- Remainders take from 33% less to 9% more than GMP's up to 128 bytes, and 6–24% less from 256 bytes to 8 KiB.
- Quotients by longer divisors take 3–31% less than GMP's from 256 bytes to 8 KiB, and up to 128 bytes from 14% less to 13% more; by one limb, 30–50% more for dividends of 24 to 64 bytes, and 16–20% less from 1 KiB.
- For the longest operands the time goes to the products of divide and conquer, for which GMP has FFT methods: a quotient or a remainder of 64 KiB by 32 KiB takes 12–13% more than GMP's.
- A quotient alone of an exact multiple ends in the compensation of the shrinking divisor, which adds about the products it left out: by divisors of 128 to 255 limbs that takes up to 1.4 times as long as a random dividend, about what GMP's method does; `div_exact` is the method for exact multiples (below).
- Against num-bigint, a division takes a third of its time or less, about 3–10 times faster.
- The x86_64 kernel has been tested for correctness under emulation only; its speed has not been measured.

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

- Time complexity: O(n·m) for a dividend of n limbs and a divisor of m below 40, and from there for balanced operands that of a product of n limbs, O(n^1.58) with Karatsuba's method, O(n^1.47) with Toom-3's and O(n^1.40) with Toom-4's; the quotient alone with about half the products of its last limbs below 256 limbs in the divisor, of one much shorter than the divisor in O(q²) for q limbs of quotient, and an exact quotient in O(q·m) less a triangle, whatever the signs and byte orders
- Space complexity: the vector of the result, unless it is built in an operand's vector; working memory of about three times the dividend's limbs, on the stack up to 8 KiB, one temporary buffer on the heap beyond; a `Divisor` keeps its divisor's limbs
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; a borrowed operand is left as it was

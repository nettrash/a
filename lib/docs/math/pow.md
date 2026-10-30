[A](../../../README.md) › [Math](../../../README.md#math)

# Powers

[Numbers](numbers.md) are raised to a power with `a.pow(e)`, for a `u32` exponent, exactly, and to the power of a number modulo another with `a.pow_mod(&e, &m)`, the power modular arithmetic and cryptography use: `a^e mod m` for exponents of any length.

- `pow` follows `i32::pow`: `x^0` is 1 for every `x`, zero included, and the power is negative when `a` is negative and `e` odd. It is exact, with no overflow: a power of an `n`-byte number by `e` takes up to `n·e` bytes.
- `pow_mod` gives Euclid's remainder, as `rem_euclid` does: the `r` with `0 <= r < |m|` and `r = a^e (mod |m|)`, never negative, so a negative base gives `|m| - (|a|^e mod |m|)` for an odd exponent. Only the modulus's magnitude counts, and the base may be longer than the modulus. `x^0 mod m` is `1 mod |m|`: 1, or 0 modulo 1.
- `pow_mod` panics when the modulus is zero, with the message of `%`, `attempt to calculate the remainder with a divisor of zero`, and when the exponent is negative, with `negative exponent`, at the caller's line: a negative power is a power of the inverse of `a` modulo `m`, which is to be computed first (`mod_inverse`). A negative zero exponent is zero. `pow` panics when the power is too long to allocate, rather than aborting.
- **`pow_mod` is not constant-time.** Its time depends on the bits of the exponent, not only on its length (a one bit costs a product a zero bit does not, and the sliding window skips runs of zeros), and on the operands' lengths: it is not meant for secret exponents, such as private keys, where whoever can time it, or watch its caches, could learn them.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says.
- The result is in the byte order of `a` and in the shortest form: zero is `[0]`, never empty and never negative. It is built in one new vector of exactly its length; the operands are left as they were.

## Powers

The method is binary exponentiation from the top bit of the exponent: a square for each bit, and a product by the base for each one. This implementation carries it out as follows:

- Zero bits at the low end of the base are taken out first and put back at the end, as a shift of `z·e` bits, where they cost nothing: `(2^z·o)^e = 2^(z·e)·o^e`.
- Registers up to 64 bytes: a power that fits in 16 bytes is computed in a `u128`; one of up to 512 bits, of an odd part of up to 16 bytes, in four or eight limbs in registers, each square and product only over the limbs the power has so far; and a power of `±2^z`, whose odd part is 1, as its shift alone, however long.
- Limbs beyond, with the [multiplication](mul.md)'s squares and products: Karatsuba's method from 56 limbs for a square, Toom-3 from 168 and Toom-4 from 224. The power's length is bounded before it starts, from the bits of the odd part, so its working memory is allocated once: two buffers the power goes back and forth between, and the scratch of the longest square, in the function's own frame for short powers, on the stack up to 8 KiB and on the heap beyond.

## Powers modulo a number

The method is Montgomery's multiplication with a sliding window over the exponent's bits, after GMP's `mpz_powm`:

- Montgomery's form: modulo an odd `m` of `n` limbs, a residue `x` is held as `x·R mod m`, `R = 2^(64n)`, and the product of two residues is reduced by REDC, which divides by `R` modulo `m` without a division: it adds the multiple `q·m` that clears the low `n` limbs of the product (`q = -u/m mod R`, Hensel's quotient) and drops them. The base enters the form by one division (`a·R mod m`, the crate's division, which also reduces a base longer than the modulus), and the power leaves it by one more REDC.
- The sliding window: the exponent's bits from the top, a square for each, and for each window of up to `k` bits that starts and ends with a one, a product by that odd power of the base from a table of `2^(k−1)`, `k` growing with the exponent's length after GMP's table (1 bit up to 7 bits of exponent, 2 up to 25, … 10 bits beyond 28,161). Measured, windows of one bit more or less than GMP's took within 1 % of its, except at 64 bytes (10 % more with one bit more).
- One and two limbs in registers, each product reduced as it is formed: one limb by GMP's `MPN_REDC_0` (the high limb less that of `q·m`), two limbs with the quotient of both limbs at once, `q = u·(−1/m) mod 2^128`, which saved 11 % at 16 bytes over two steps of a limb; operands of up to 16 bytes each take this path without working memory at all.
- Three to seven limbs in registers, by the coarsely integrated operand scanning of Koç, Acar and Kaliski: for each limb of one factor, a row of the product and a row of the reduction, unrolled by the compiler. Against the limbs in memory below, measured: 3 to 5 limbs in about half the time, 6 limbs in 13 % less, 7 limbs in 9 % less, and 8 limbs in 5 % more, so registers end at 7 limbs.
- Eight to 99 limbs: the square by the square's kernels and the product by the product's, then REDC two limbs of the quotient at a time, each pass one call of the multiplication's two-row kernel (`addmul_2`, inline assembly on aarch64, four carry chains), the pass's top limbs kept in the limbs it cleared and added to the high half at the end, as GMP's `mpn_redc_2` arranges it; a limb at a time on other targets.
- From 100 limbs: REDC by products, the quotient of all `n` limbs at once as the low half of a product, `q = u·(1/m) mod R` (the low half by divide and conquer, after GMP's `mpn_dc_mullo_n`), then the product `q·m`, whose high half is subtracted from the product's: about 1.7 products for a reduction, against the `n²` limb products of the passes, which the Karatsuba and Toom-Cook products undercut from there. Measured with exponents of 256 bits, a modulus of 64 limbs took 2.09 million cycles by passes against 2.40 by products, 128 limbs 7.8–8.1 million against 7.2–7.5, and 96 limbs within 3 % either way.
- Even moduli: `m = 2^z·m'` with `m'` odd, the power modulo `m'` as above and modulo `2^z` by the low halves of products (a sliding window too; zero at once when the base's zero bits fill `2^z`; in registers up to eight limbs, which made a power modulo `2^255` five times as fast as on limbs), put together by the Chinese remainder theorem, `r = r1 + m'·((r2 − r1)·m'^(−1) mod 2^z)`, as GMP's `mpz_powm` does; a power of two alone by the low halves only.
- Short cuts: an exponent of 0 gives `1 mod |m|`, an exponent of 1 the remainder alone, a modulus of 1 and a base of 0 give 0.

## Measured

Measured on an Apple M4 Max with other work running on the machine (interleaved runs, the median of five, of three from 1 KiB and one run at 8 KiB, in cycles per call, rounded): `a.pow_mod(&e, &m)` for a random odd modulus of the length given, a random base below it and a random exponent as long as the modulus (the RSA-like case) or 65537, against GMP 6.3's `mpz_powm`, into a result allocated before (`GMP`) and into a new `mpz_t` (`GMP new`, which allocates the result as `pow_mod` does), and num-bigint 0.4's `modpow`:

| Bytes | `e` as long: LE | BE | GMP | GMP new | num-bigint | `e = 65537`: LE | BE | GMP | GMP new | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1149 | 1134 | 1060 | 1161 | 6698 | 319 | 317 | 275 | 328 | 6350 |
| 16 | 4351 | 4347 | 9896 | 10,039 | 25,354 | 701 | 701 | 1333 | 1419 | 14,590 |
| 24 | 10,495 | 10,391 | 20,489 | 21,128 | 43,908 | | | | | |
| 32 | 16,971 | 16,673 | 32,254 | 32,277 | 72,963 | 1466 | 1436 | 2219 | 2323 | 21,447 |
| 48 | 56,020 | 54,848 | 69,480 | 69,831 | 151,252 | | | | | |
| 64 | 119,000 | 118,393 | 128,779 | 128,766 | 271,225 | 4300 | 4274 | 4488 | 4526 | 38,585 |
| 128 | 623,373 | 624,090 | 754,833 | 757,317 | 1,334,876 | 11,291 | 11,104 | 13,234 | 13,444 | 95,078 |
| 256 | 4.30 M | 4.29 M | 5.13 M | 5.13 M | 9.09 M | 37,844 | 37,835 | 44,360 | 44,282 | 345,499 |
| 512 | 31.1 M | 31.4 M | 35.9 M | 36.0 M | 73.6 M | 139,931 | 139,622 | 163,438 | 162,934 | 1.37 M |
| 1 KiB | 233 M | 219 M | 239 M | 237 M | 683 M | 517,596 | 517,139 | 545,492 | 541,944 | 6.08 M |
| 2 KiB | 1300 M | 1287 M | 1311 M | 1326 M | | 1.49 M | 1.46 M | 1.55 M | 1.52 M | 24.8 M |
| 4 KiB | 7178 M | 7114 M | 7562 M | 7549 M | | 4.17 M | 4.14 M | 4.31 M | 4.27 M | 99.2 M |
| 8 KiB | 39,639 M | 40,247 M | 39,711 M | 39,580 M | | 11.9 M | 11.9 M | 11.4 M | 11.5 M | 400 M |

Modulo even numbers, measured the same way with an exponent as long as the modulus: a random even modulus (whose odd part takes nearly all the work), and a power of two, `2^(8·bytes − 1)`, with an odd base:

| Bytes | even: LE | GMP | num-bigint | `2^(8·bytes − 1)`: LE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | | | | 1784 | 8014 | 142,958 |
| 32 | 17,072 | 32,682 | 275,981 | 2664 | 12,487 | 240,104 |
| 64 | | | | 18,288 | 39,757 | 905,597 |
| 256 | 4.28 M | 5.09 M | 28.3 M | 1.35 M | 1.40 M | 25.1 M |
| 1 KiB | 240 M | 233 M | 1231 M | | | |

- From 16 bytes, 2 limbs, to 7 limbs (56 bytes) in registers, 0.44 to 0.81 of GMP's time with a long exponent and 0.53 to 0.66 with 65537; GMP keeps its residues in limbs from two limbs on, and reduces a limb at a time with one call of its row kernel each below 68 limbs.
- From 64 bytes to 512, 0.83 to 0.92 of GMP's time, from the square's kernels and the two-row REDC; from 1 KiB to 8 KiB, about GMP's time (0.95 to 1.05, where runs differed by up to 20 % with the machine shared): REDC by products, which GMP computes with a product modulo `B^k − 1` (`mpn_mulmod_bnm1`) for the high half, and the longest squares, where GMP's Toom-6 and Toom-8 come in.
- One limb takes `GMP new`'s time; the result's vector is the difference against `GMP`: allocating and freeing one takes about 66 cycles on this machine.
- Modulo powers of two of three to eight limbs, in registers, 0.2 to 0.5 of GMP's time; modulo random even numbers, the odd part's time.
- Against num-bigint, 2 to 6 times faster with a long exponent, 8 to 34 times with 65537, and up to 90 times modulo a power of two. Little-endian and big-endian operands take the same time, apart from reading them in.

`a.pow(e)` against GMP's `mpz_pow_ui` and num-bigint's `pow`, for a random base of the length given in bytes:

| Bytes, exponent | LE | BE | GMP | GMP new | num-bigint |
| --- | --- | --- | --- | --- | --- |
| 1, 10 | 71 | 71 | 31 | 97 | 121 |
| 8, 3 | 107 | 114 | 45 | 103 | 247 |
| 8, 16 | 321 | 324 | 210 | 270 | 570 |
| 16, 7 | 309 | 312 | 194 | 231 | 699 |
| 32, 8 | 522 | 516 | 468 | 511 | 1254 |
| 32, 32 | 3428 | 3469 | 4081 | 4164 | 13,179 |
| 64, 64 | 29,723 | 30,169 | 42,990 | 41,648 | 139,707 |
| 128, 16 | 10,450 | 10,404 | 12,701 | 12,807 | 40,777 |
| 256, 16 | 29,128 | 29,096 | 41,692 | 40,997 | 132,469 |
| 1024, 8 | 82,986 | 82,927 | 105,840 | 103,683 | 403,465 |
| 4096, 4 | 204,958 | 210,965 | 239,390 | 245,699 | 993,942 |
| 8, 1000 | 81,263 | 81,539 | 101,481 | 102,418 | 537,496 |
| 8, 10,000 | 2.38 M | 2.39 M | 2.17 M | 2.21 M | 18.7 M |
| 8, 100,000 | 61.9 M | 63.3 M | 29.6 M | 29.1 M | 535 M |

- Small powers cost little beyond their vector: allocating and freeing one takes about 66 cycles on this machine, the difference against GMP writing into a result allocated before; against `GMP new`, which allocates its result too, from 0.73 to 1.04 of its time in registers. Powers of 100 to 300 bytes, on limbs, take 1.02 to 1.34 times `GMP new`'s, where the steps' fixed costs are not yet small against squares of a few limbs.
- From about 1 KiB of result to 16 KiB, 0.69 to 0.86 of GMP's time; beyond, the longest squares are where GMP's FFT multiplication wins, as on the [multiplication](mul.md)'s page: a power of 80 KiB took 1.10 times GMP's time, and one of 800 KiB 2.1 times.
- Against num-bigint, 1.7 to 8.7 times faster.

## Examples

Rust usage: powers, and the rules of `i32::pow`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(-3, ByteOrder::BigEndian);
  let p = a.pow(3);
  assert_eq!(p.value, vec![27]);
  assert!(p.is_negative());
  assert!(!a.pow(4).is_negative());
  assert_eq!(a.pow(0).value, vec![1]);

  // 2^100 = 0x10 followed by 12 zero bytes, exactly.
  let two = Number::new_from_u8(2, ByteOrder::BigEndian);
  let p = two.pow(100);
  let mut expected = vec![0u8; 13];
  expected[0] = 0x10;
  assert_eq!(p.value, expected);
}
```

Powers modulo a number: Euclid's remainder, Fermat's little theorem, and the panics:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let m = Number::new_from_u32(1_000_000_007, ByteOrder::BigEndian);
  let e = Number::new_from_u32(1_000_000_006, ByteOrder::BigEndian);
  // a^(p-1) = 1 (mod p) for a prime p not dividing a.
  for a in [2u32, 3, 12_345] {
    let a = Number::new_from_u32(a, ByteOrder::LittleEndian);
    assert_eq!(a.pow_mod(&e, &m).value, vec![1]);
  }

  // (-2)^3 mod 7 = -8 mod 7 = 6, never negative; the modulus's sign does not count.
  let r = Number::new_from_i32(-2, ByteOrder::BigEndian)
    .pow_mod(&Number::new_from_u8(3, ByteOrder::BigEndian), &Number::new_from_i32(-7, ByteOrder::BigEndian));
  assert_eq!(r.value, vec![6]);
  assert!(!r.is_negative());

  let zero = Number::new_from_u8(0, ByteOrder::BigEndian);
  let caught = std::panic::catch_unwind(|| m.pow_mod(&e, &zero));
  assert!(caught.is_err());
  let minus_one = Number::new_from_i32(-1, ByteOrder::BigEndian);
  let caught = std::panic::catch_unwind(|| m.pow_mod(&minus_one, &e));
  assert!(caught.is_err());
}
```

RSA in miniature, with the textbook key `p = 61`, `q = 53`, `e = 17`, `d = 2753`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let n = Number::new_from_u32(61 * 53, ByteOrder::BigEndian);
  let e = Number::new_from_u32(17, ByteOrder::BigEndian);
  let d = Number::new_from_u32(2753, ByteOrder::BigEndian);
  let message = Number::new_from_u32(65, ByteOrder::BigEndian);
  let cipher = message.pow_mod(&e, &n);
  assert_eq!(cipher.value, 2790u32.to_be_bytes()[2..]);
  assert_eq!(cipher.pow_mod(&d, &n).value, vec![65]);
}
```

## Efficiency

- Time complexity: `pow` about that of the last square, `S(n·e/2)` for an `n`-limb base, with the multiplication's methods; `pow_mod` about `1.2·b` reductions of products of `n` limbs for an exponent of `b` bits, each `O(n²)` below 100 limbs and about 1.7 products of `n` limbs beyond, whatever the signs and byte orders
- Space complexity: the vector of the result; `pow`'s working memory about three times the power's limbs; `pow_mod`'s the operands' limbs, the window's table (up to 512 residues for exponents of more than 28,161 bits, 16 of 512-bit ones) and the scratch of a product, on the stack up to 8 KiB, one temporary buffer on the heap beyond
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; the operands are left as they were

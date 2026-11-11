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
- Limbs beyond, with the [multiplication](mul.md)'s squares and products: Karatsuba's method from 56 limbs for a square, Toom-3 from 168 and Toom-4 from 224. The power's length is bounded before it starts, from the bits of the odd part, so its working memory is allocated once: two buffers the power goes back and forth between, and the scratch of the squares and products at the lengths the power may reach, in the function's own frame for short powers, on the stack up to 8 KiB and on the heap beyond. Those lengths are bounds, and a shorter square can need more scratch than a longer one (a square of 202 limbs takes 486 limbs of it, of 203 or 204 limbs less): a square or a product that needs more than was planned takes a buffer of its own, freed when it is done.

## Powers modulo a number

The method is Montgomery's multiplication with a sliding window over the exponent's bits, after GMP's `mpz_powm`:

- Montgomery's form: modulo an odd `m` of `n` limbs, a residue `x` is held as `x·R mod m`, `R = 2^(64n)`, and the product of two residues is reduced by REDC, which divides by `R` modulo `m` without a division: it adds the multiple `q·m` that clears the low `n` limbs of the product (`q = -u/m mod R`, Hensel's quotient) and drops them. The base enters the form by one division (`a·R mod m`, the crate's division, which also reduces a base longer than the modulus), and the power leaves it by one more REDC.
- The sliding window: the exponent's bits from the top, a square for each, and for each window of up to `k` bits that starts and ends with a one, a product by that odd power of the base from a table of `2^(k−1)`, `k` growing with the exponent's length after GMP's table (1 bit up to 7 bits of exponent, 2 up to 25, … 10 bits beyond 28,161). Measured, windows of one bit more or less than GMP's took within 1 % of its, except at 64 bytes (10 % more with one bit more).
- Moduli of one to three limbs (one and two on targets without the kernels below) take the exponent's bits from the low end instead: the squares `b^(2^i)` make one chain and the product of those of the one bits a second beside it, so that a power takes about a square's time a bit, its products in the shadow of the squares, where the window's products wait for each other. There a product takes two to three times as long after the one before as side by side; from four limbs, where it takes 1.5 times as long, the window's fewer products win. Measured, a modulus of one limb took 999 cycles against 1176 by the window, two limbs 3530 against 3964, three 8190 against 8375, four 15,340 against 14,400.
- One limb in registers by GMP's `MPN_REDC_0`: the high limb less that of `q·m`, `q` the low limb's quotient.
- Two to eight limbs in registers (to seven on targets without the kernels below, where the Rust rows of eight limbs took 1.02 to 1.11 times as long as the limbs in memory, and those of three limbs 1.10 to 1.21 times as long from the low end as by the window), by the coarsely integrated operand scanning of Koç, Acar and Kaliski: for each limb of one factor a row of the product and a row of the reduction, on a window of `n + 2` limbs that moves up a limb after each row of the reduction. On aarch64 each length has kernels of its own in inline assembly, generated by a script in the tree (`pow/montgomery/kernels.py`): a row is two carry chains, the low limbs of its products and then their high limbs a position up, the first chain's carry folded into the high limb of the row's last product, which has room for it; a square takes each product of two different limbs once, by rows of `a_i·[a_i, 2a_(i+1), 2a_(i+2), …]` that end where a product's rows do, so that each is followed by a row of the reduction too. Measured on an Apple M4 Max, a product of eight limbs took 113 cycles after the one before against 208 by the Rust rows (which squared by a product), a square 115, and four independent ones 106 and 92 cycles each against 210; four limbs 46 and 43 cycles against 56, two limbs 24 and 23 against 25 and 27. The kernels are bound by the carry chains' additions, which this core runs about 2.5 a cycle, two a limb product: 0.83 cycles a limb product at eight limbs. Elsewhere, and under Miri, the Rust rows.
- Nine to sixteen limbs on aarch64: the same products with every operand read from memory, two limbs a load, and the window of the rows in registers, one kernel for each length out of line: a product of sixteen limbs took 455 cycles after the one before against 604 by the multiplication's kernels and REDC below, a square 380 against 580, and nine limbs 153 and 143 against 274 and 276.
- From seventeen limbs to 99 (from eight on other targets): the square by the square's kernels and the product by the product's, then REDC two limbs of the quotient at a time, each pass one call of the multiplication's two-row kernel (`addmul_2`, inline assembly on aarch64, four carry chains), the pass's top limbs kept in the limbs it cleared and added to the high half at the end, as GMP's `mpn_redc_2` arranges it; a limb at a time on other targets.
- From 100 limbs: REDC by products, the quotient of all `n` limbs at once as the low half of a product, `q = u·(1/m) mod R` (the low half by divide and conquer, after GMP's `mpn_dc_mullo_n`), then the product `q·m`, whose high half is subtracted from the product's: about 1.7 products for a reduction, against the `n²` limb products of the passes, which the Karatsuba and Toom-Cook products undercut from there. Measured with exponents of 256 bits, a modulus of 64 limbs took 2.09 million cycles by passes against 2.40 by products, 128 limbs 7.8–8.1 million against 7.2–7.5, and 96 limbs within 3 % either way.
- Even moduli: `m = 2^z·m'` with `m'` odd, the power modulo `m'` as above and modulo `2^z` by the low halves of products (a sliding window too; zero at once when the base's zero bits fill `2^z`; in registers up to eight limbs), put together by the Chinese remainder theorem, `r = r1 + m'·((r2 − r1)·m'^(−1) mod 2^z)`, as GMP's `mpz_powm` does; a power of two alone by the low halves only. The squares take the low half of a square, about half the products of a low half (the products of two different limbs once, doubled, and the squares), as GMP's `mpn_sqrlo` does; on aarch64, nine to sixteen limbs by kernels in inline assembly with the result in registers, and the halves of 18 to 32 limbs by divide and conquer on them, the high half of half the limbs. Measured, a power modulo `2^1023` with an exponent of 1024 bits took 95,000 cycles against 273,000 by rows of products, and one modulo `2^2047` 0.99 million against 1.59 million.
- Short cuts: an exponent of 0 gives `1 mod |m|`, an exponent of 1 the remainder alone, a modulus of 1 and a base of 0 give 0.

## Measured

Measured on 2026-11-11 on an Apple M4 Max on a quiet machine (the load average at 1.5 to 2.5 of its 16 cores, the clock at 3.9 to 4.5 GHz, each round's cycles from its own clock; one thread, every process with `MallocNanoZone` unset, the harness built with fat LTO and one codegen unit; interleaved runs, the median of nine (with a long exponent of seven at 1 and 2 KiB, five at 4 KiB and three at 8 KiB; num-bigint's of three at 1 KiB), in cycles per call, rounded): `a.pow_mod(&e, &m)` for a random odd modulus of the length given, a random base below it and a random exponent as long as the modulus (the RSA-like case) or 65537, against GMP 6.3's `mpz_powm`, into a result allocated before (`GMP`) and into a new `mpz_t` (`GMP new`, which allocates the result as `pow_mod` does), and num-bigint 0.4's `modpow`:

| Bytes | `e` as long: LE | BE | GMP | GMP new | num-bigint | `e = 65537`: LE | BE | GMP | GMP new | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 980 | 972 | 1066 | 1151 | 5428 | 303 | 299 | 283 | 326 | 5076 |
| 16 | 3585 | 3559 | 10,392 | 10,751 | 19,367 | 601 | 601 | 1364 | 1465 | 9850 |
| 24 | 8208 | 8252 | 21,559 | 21,851 | 34,552 |  |  |  |  |  |
| 32 | 14,543 | 14,421 | 34,395 | 35,126 | 61,836 | 1254 | 1257 | 2283 | 2358 | 16,923 |
| 48 | 33,409 | 33,590 | 75,552 | 76,318 | 140,254 |  |  |  |  |  |
| 64 | 71,173 | 70,013 | 137,731 | 139,115 | 248,009 | 2884 | 2912 | 4774 | 4851 | 36,104 |
| 128 | 468,714 | 469,907 | 876,912 | 880,125 | 1.34 M | 8656 | 8659 | 14,133 | 14,360 | 98,553 |
| 256 | 4.70 M | 4.69 M | 5.53 M | 5.55 M | 10.0 M | 40,360 | 40,643 | 47,043 | 47,099 | 362,243 |
| 512 | 33.7 M | 33.7 M | 42.0 M | 42.1 M | 79.3 M | 158,037 | 157,661 | 184,986 | 186,871 | 1.61 M |
| 1 KiB | 231 M | 231 M | 255 M | 253 M | 705 M | 562,604 | 562,939 | 564,639 | 565,019 | 6.58 M |
| 2 KiB | 1386 M | 1387 M | 1415 M | 1415 M |  | 1.60 M | 1.60 M | 1.65 M | 1.65 M | 26.5 M |
| 4 KiB | 6277 M | 6273 M | 7708 M | 7730 M |  | 3.73 M | 3.73 M | 4.50 M | 4.54 M | 107 M |
| 8 KiB | 27,400 M | 27,711 M | 42,167 M | 42,097 M |  | 8.52 M | 8.50 M | 12.4 M | 12.5 M | 430 M |

Modulo even numbers, measured the same way with an exponent as long as the modulus: a random even modulus (whose odd part takes nearly all the work), and a power of two, `2^(8·bytes − 1)`, with an odd base:

| Bytes | even: LE | GMP | num-bigint | `2^(8·bytes − 1)`: LE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- |
| 24 |  |  |  | 1750 | 8530 | 123,860 |
| 32 | 15,096 | 34,359 | 254,611 | 2765 | 13,223 | 216,438 |
| 64 |  |  |  | 19,122 | 42,992 | 901,896 |
| 72 |  |  |  | 25,912 | 53,026 | 1.18 M |
| 128 |  |  |  | 95,436 | 172,716 | 4.40 M |
| 256 | 4.64 M | 5.52 M | 29.5 M | 0.99 M | 0.97 M | 25.2 M |
| 512 |  |  |  | 7.84 M | 7.51 M | 180 M |
| 1 KiB | 254 M | 246 M | 1263 M |  |  |  |

- From 16 bytes, two limbs, to 64 bytes, eight, in registers by the kernels in inline assembly, 0.34 to 0.52 of GMP's time with a long exponent and 0.44 to 0.60 with 65537; GMP keeps its residues in limbs from two limbs on, and reduces a limb at a time with one call of its row kernel each below 68 limbs. Before the kernels (this crate's 0.2.32, in the same runs), 0.42 to 0.92 with a long exponent.
- At 128 bytes, sixteen limbs, by the kernels of residues in memory, 0.53 of GMP's time; from 256 bytes the square's kernels and the two-row REDC, 0.80 to 0.98 to 2 KiB (at 2 KiB the REDC takes 62 % of the time and the square 31 %); from 4 KiB REDC by products and the multiplication's transforms, 0.65 to 0.81.
- One limb takes 0.85 of `GMP new`'s time with a long exponent and 0.93 with 65537, from the exponent's bits taken from the low end; the result's vector is the difference against `GMP`: allocating and freeing one of a limb takes about 31 cycles on this machine (31 to 51 up to 64 bytes).
- Modulo powers of two of three to eight limbs, in registers, 0.21 to 0.44 of GMP's time; nine to sixteen by the kernels, about half (0.49 to 0.55; they took 1.4 to 1.6 times GMP's time on rows of products before); from 32 limbs, divide and conquer, about GMP's time (1.02 to 1.04). Modulo random even numbers, the odd part's time (0.99 to 1.10 times it).
- Against num-bigint, 2.1 to 5.5 times faster with a long exponent, 9 to 50 times with 65537, and up to 78 times modulo a power of two. Little-endian and big-endian operands take the same time, apart from reading them in.
- The times depend little on the build's profile: under cargo's default release profile (no LTO, 16 codegen units) instead of the harness's fat LTO, measured with the two builds alternately (21 rounds a cell), `pow_mod` with a long exponent took 1.00, 1.02 and 1.01 times as long at 64 bytes, 256 bytes and 1 KiB (0.2.32: 1.06, 1.02 and 1.02 times), 0.51, 0.84 and 0.95 of GMP's time against 0.51, 0.83 and 0.93 with fat LTO.

`a.pow(e)` against GMP's `mpz_pow_ui` and num-bigint's `pow`, for a random base of the length given in bytes (the median of seven):

| Bytes, exponent | LE | BE | GMP | GMP new | num-bigint |
| --- | --- | --- | --- | --- | --- |
| 1, 10 | 97 | 87 | 35 | 81 | 111 |
| 8, 3 | 90 | 90 | 50 | 84 | 207 |
| 8, 16 | 361 | 364 | 220 | 274 | 483 |
| 16, 7 | 329 | 341 | 202 | 225 | 551 |
| 32, 8 | 586 | 592 | 502 | 545 | 1245 |
| 32, 32 | 3751 | 3775 | 4410 | 4407 | 13,501 |
| 64, 64 | 31,707 | 31,991 | 46,281 | 45,959 | 142,389 |
| 128, 16 | 10,999 | 10,938 | 13,705 | 13,808 | 44,531 |
| 256, 16 | 31,772 | 32,017 | 44,772 | 44,966 | 142,144 |
| 1024, 8 | 75,167 | 76,354 | 110,602 | 108,410 | 426,068 |
| 4096, 4 | 143,142 | 149,102 | 270,144 | 270,852 | 1.11 M |
| 8, 1000 | 77,169 | 78,113 | 108,371 | 107,075 | 566,681 |
| 8, 10,000 | 1.15 M | 1.16 M | 2.40 M | 2.39 M | 19.8 M |
| 8, 100,000 | 11.6 M | 11.7 M | 31.4 M | 31.2 M | 555 M |

- Small powers cost little beyond their vector: allocating and freeing one of the first two rows' lengths (up to 10 and 24 bytes) takes 31 and 51 cycles on this machine, against differences of 62 and 40 cycles to GMP writing into a result allocated before (`GMP new` took 46 and 34 cycles more than `GMP`); against `GMP new`, which allocates its result too, 1.07 to 1.20 of its time in registers. Powers of 100 to 300 bytes, on limbs, take 1.08 to 1.46 times `GMP new`'s, where the steps' fixed costs are not yet small against squares of a few limbs.
- From about 1 KiB of result to 16 KiB, 0.53 to 0.85 of GMP's time; beyond, the longest squares go to the number-theoretic transforms of the [multiplication](mul.md): a power of 80 KiB took 0.48 of GMP's time and one of 800 KiB 0.37 (1.10 and 2.1 times before the transforms), and powers of 1 MiB, a base of 1 KiB to the 1024th or of 64 bytes to the 16,384th, 0.36 and 0.32.
- Against num-bigint, 1.1 to 7.8 times faster up to 16 KiB of result, and 17 and 48 times at 80 KiB and 800 KiB.

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
- Space complexity: the vector of the result; `pow`'s working memory about three times the power's limbs; `pow_mod`'s the operands' limbs, the window's table (up to 512 residues for exponents of more than 28,161 bits, 16 of 512-bit ones) and the scratch of a product, on the stack up to 8 KiB, one temporary buffer on the heap beyond, and one more for each product of 360 limbs or more, which the [multiplication](mul.md) computes by transforms, and the buffers of [Newton's method](div.md) for the long divisions that reduce the base and put it in Montgomery's form, from 768 limbs of quotient
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; the operands are left as they were

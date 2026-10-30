[A](../../../README.md) › [Math](../../../README.md#math)

# Bitwise operations

[Numbers](numbers.md) combine bit by bit with `&`, `|` and `^`, each operand owned or borrowed: `a & b`, `a & &b`, `&a & b` and `&a & &b` give a new `Number`, and `a &= b` and `a &= &b` change `a`; `|`, `|=`, `^` and `^=` likewise. `!a` and `!&a` invert every bit. Five methods read and write the bits of a number: `bits`, `count_ones`, `trailing_zeros`, `bit` and `set_bit`.

- The operators work as if both operands were in two's complement with infinitely many bits, as those of `i32` do on 32 bits and GMP's `mpz_and`, `mpz_ior`, `mpz_xor` and `mpz_com` and num-bigint's `BigInt` do on any number: a negative number has infinitely many ones above the bits of its magnitude, `-x` being `!x + 1`. So `12 & -10` is `4`, `12 | -10` is `-2`, `12 ^ -10` is `-6`, `-1 & x` is `x`, and `!x` is `-x - 1`: `!0` is `-1`, `!-1` is `0`. The result is stored back as a sign and a magnitude.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. Numbers built from the public fields, which need not be in the shortest form, combine as the values they hold.
- The result is in the byte order of the left operand in every form, `&a & b` included, and of the operand of `!`; it is in the shortest form: no zero bytes above the most significant one, and zero is `[0]`, never empty and never negative. A result can be one byte longer than both operands: `-0x80 & -0x81` is `-0x100`, `0xFF ^ -1` is `-0x100`, `!0xFF` is `-0x100`.
- The result is built where a sum would be: in `a`'s vector for `a & &b`, `a &= &b`, `a & b` and `a &= b` when its capacity is at least the length of the result (one byte for zero), in `b`'s for `&a & b`, and for `a & b` and `a &= b` when `a`'s has no room, also when `b` is in the other byte order; `!a` in `a`'s. Otherwise, and always for `&a & &b` and `!&a`, exactly one new vector of exactly the result's length is allocated, and nothing else.
- A borrowed operand is left as it was.

The methods:

- `bits()` is the number of bits of the magnitude, `|x|.ilog2() + 1`, 0 for zero, as num-bigint's `BigInt::bits` (GMP's `mpz_sizeinbase(x, 2)` gives 1 for zero).
- `count_ones()` is the number of ones in the magnitude, as Python's `int.bit_count` and num-bigint's `BigUint::count_ones` count them: the sign is not counted, `-0b1011` has three. (In two's complement a negative number has infinitely many ones; GMP's `mpz_popcount` gives its largest value for one.)
- `trailing_zeros()` is the number of zeros below the lowest one, its position, or `None` for zero, the same for `x` and `-x`, as num-bigint's `BigInt::trailing_zeros` and GMP's `mpz_scan1(x, 0)`.
- `bit(i)` is bit `i` in two's complement, as num-bigint's `BigInt::bit` and GMP's `mpz_tstbit`: for a negative number every bit above the magnitude is one.
- `set_bit(i, value)` sets bit `i` in two's complement to `value`, as num-bigint's `BigInt::set_bit` and GMP's `mpz_setbit` and `mpz_clrbit`, keeps the byte order and leaves the number in the shortest form (one built from the public fields included). The vector grows as needed; a bit too high for any vector to hold panics, as `Vec` does on a capacity overflow, rather than aborting.

The method: with `X` and `Y` the magnitudes, a negative operand is `!(X - 1)` and a negative result `!(R - 1)`, so every combination of signs comes down to one operation `g` on the bytes of `X` or `X - 1` and `Y` or `Y - 1`, and one added when the result is negative, as GMP's `mpz/and.c`, `ior.c` and `xor.c` arrange it:

| | `x & y` | `x \| y` | `x ^ y` |
| --- | --- | --- | --- |
| `+ +` | `X & Y` | `X \| Y` | `X ^ Y` |
| `+ -` | `X & !(Y - 1)` | `-((Y - 1) & !X) - 1` | `-(X ^ (Y - 1)) - 1` |
| `- +` | `!(X - 1) & Y` | `-((X - 1) & !Y) - 1` | `-((X - 1) ^ Y) - 1` |
| `- -` | `-((X - 1) \| (Y - 1)) - 1` | `-((X - 1) & (Y - 1)) - 1` | `(X - 1) ^ (Y - 1)` |

The ones taken and the one added are borrows and a carry that run only through the low bytes: `X - 1` differs from `X` at and below its lowest nonzero byte, and the carry of `R + 1` runs through the bytes of `R` that are `0xFF`. GMP subtracts the ones into a temporary copy of each negative operand first; here they are fused into the one pass. This implementation carries it out as follows:

- Registers up to 16 bytes: both magnitudes are read into `u128`s as the [addition](add.md) reads them, made two's complement, combined, and turned back, all without a branch; a result of `2^128` is the 17th byte.
- Operands that are not negative, in the same byte order: the operation, known at compile time, runs straight over the bytes of the shorter magnitude with the kernel below, and the longer one's top bytes follow for `|` and `^` (in place, they are already there). The length is the longer magnitude's then, and otherwise found from the top down, one read for random operands: of the result just written, little-endian, and big-endian of the operands before anything is written, as a read of the result would wait for the stores just made (a `^=` of 40 to 64 bytes took 30 to 34 cycles so, against 19 to 20); a shorter big-endian result is then moved to the start of the vector.
- Up to 64 bytes otherwise: both magnitudes are read into up to eight limbs each, as the addition's medium path reads them, made two's complement by chains of carries, combined, and turned back, without a branch on the signs; the length and the top eight bytes are found without a branch too, and the result written by stores chosen by the number of limbs rather than its length, as whether `&` or `|` of operands of opposite signs loses its top byte is a coin toss.
- Longer operands: the borrows and the carry are worked out limb by limb from the low end, in flags, until all are spent, which for random operands is the first limb (the head); above it, each byte of the result is `g` of the bytes at its position, with no carry. In an operand's vector with room, the head is computed, the length found from the top of the operands (one read), and the rest is one kernel call and a copy of the longer operand's top bytes. A longer head, a new vector, or a big-endian result of another length than the vector it is built in (which then moves: written from the least significant byte up when it grows, from the most significant down with the head held back when it shrinks) take a plan made first: the head, by a pass that only reads it, and the exact length.
- Kernels in byte lanes on aarch64: a bitwise operation needs no byte order, so the kernels run in the order of memory, the same for both byte orders: sixteen bytes a step with NEON (`and`, `orr`, `eor` or `bic`), four blocks a loop, the block at the far end read first and written last so that in place every byte is read before a write reaches it. An operand in the other byte order is read at the mirrored place and reversed (`rev64` and `ext`). Elsewhere, and under Miri, portable kernels on `u64`s.
- `!`: `-(X + 1)` or `X - 1`, so in place only the bytes the carry or the borrow passes through are written: for a magnitude of more than eight bytes whose eight lowest ones it does not leave, as for random numbers, one read and one write of those eight. A result one byte longer or shorter is all zeros but its top byte, or all `0xFF`, and is written without moving anything.
- `count_ones`: `cnt` counts the ones of each byte of sixteen; eight such blocks are summed by `add`s in a tree and taken into two accumulators of sixteen-bit lanes by `uadalp`, one each per 128 bytes, so that the three-cycle chain of an accumulator never waits.
- `bits` and `trailing_zeros` read one byte or one word for a number in the shortest form: the top byte, or the lowest nonzero word from the low end. `bit` reads one byte for a number that is not negative; for a negative one, the eight bytes that end with bit `i`'s, which tell bit `i` and, for random numbers, that a bit below it is set, so that it is the opposite of the magnitude's; further bytes are read only when those below are all zero. `set_bit` makes the same test, and then sets or clears one bit of the magnitude, adds `2^i` (clearing the lowest one) or subtracts `2^i` (setting a bit below it).

Measured on an Apple M4 Max as `a &= &b`, `a |= &b` and `a ^= &b` with room in `a`, both operands in the byte order given, over pools of random operands of the length given in bytes (1.5 MiB in all, in the level-2 cache), against GMP 6.3's `mpz_and(a, a, b)`, `mpz_ior` and `mpz_xor` and num-bigint 0.4's `a &= &b`, `a |= &b` and `a ^= &b`, every contender's vectors as compact as num-bigint's (16 bytes of room to spare); the median of 7 interleaved runs, in cycles per call, with the machine shared with other work, so a cell can be off by a few cycles. Neither operand negative:

| Bytes | `&` LE | BE | GMP | num-bigint | `\|` LE | BE | GMP | num-bigint | `^` LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 12 | 13 | 10 | 19 | 16 | 16 | 10 | 12 | 14 | 15 | 11 | 19 |
| 16 | 15 | 15 | 12 | 20 | 15 | 17 | 12 | 18 | 15 | 16 | 12 | 19 |
| 32 | 14 | 16 | 13 | 23 | 15 | 15 | 13 | 19 | 15 | 19 | 13 | 22 |
| 64 | 16 | 20 | 16 | 22 | 18 | 17 | 16 | 19 | 19 | 22 | 16 | 19 |
| 128 | 24 | 30 | 24 | 29 | 28 | 23 | 24 | 27 | 26 | 25 | 25 | 28 |
| 256 | 38 | 42 | 39 | 47 | 44 | 36 | 46 | 42 | 42 | 41 | 47 | 46 |
| 1 KiB | 115 | 111 | 115 | 112 | 112 | 119 | 110 | 106 | 114 | 123 | 110 | 111 |
| 4 KiB | 407 | 406 | 424 | 383 | 409 | 406 | 380 | 379 | 404 | 407 | 379 | 381 |
| 16 KiB | 1428 | 1419 | 1427 | 1421 | 1444 | 1434 | 1387 | 1416 | 1427 | 1465 | 1385 | 1432 |
| 64 KiB | 5540 | 5538 | 5491 | 5498 | 5525 | 5555 | 5554 | 5492 | 5521 | 5535 | 5539 | 5550 |

A positive `a` and a negative `b` (a negative `a` and a positive `b` take the same paths, and as long):

| Bytes | `&` LE | BE | GMP | num-bigint | `\|` LE | BE | GMP | num-bigint | `^` LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 26 | 22 | 16 | 16 | 20 | 24 | 17 | 19 | 14 | 15 | 20 | 18 |
| 16 | 14 | 15 | 18 | 14 | 16 | 17 | 19 | 19 | 15 | 16 | 23 | 20 |
| 32 | 29 | 34 | 21 | 19 | 31 | 31 | 22 | 23 | 29 | 31 | 25 | 24 |
| 64 | 37 | 40 | 31 | 26 | 35 | 40 | 33 | 28 | 37 | 40 | 32 | 30 |
| 128 | 39 | 50 | 56 | 38 | 40 | 52 | 52 | 41 | 39 | 41 | 50 | 44 |
| 256 | 46 | 60 | 69 | 61 | 48 | 62 | 70 | 74 | 47 | 52 | 71 | 74 |
| 1 KiB | 131 | 137 | 250 | 282 | 130 | 135 | 253 | 322 | 132 | 129 | 271 | 319 |
| 4 KiB | 460 | 426 | 932 | 1053 | 463 | 429 | 937 | 1151 | 460 | 418 | 918 | 1138 |
| 16 KiB | 1721 | 1448 | 3383 | 4137 | 1733 | 1452 | 3395 | 4461 | 1718 | 1441 | 3417 | 4393 |
| 64 KiB | 6860 | 5628 | 15,067 | 16,511 | 6916 | 5684 | 15,137 | 17,579 | 6873 | 5503 | 15,146 | 17,535 |

Both negative:

| Bytes | `&` LE | BE | GMP | num-bigint | `\|` LE | BE | GMP | num-bigint | `^` LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 13 | 14 | 21 | 18 | 15 | 15 | 18 | 18 | 14 | 14 | 22 | 18 |
| 16 | 15 | 16 | 24 | 19 | 16 | 17 | 22 | 19 | 17 | 15 | 25 | 30 |
| 32 | 31 | 30 | 29 | 24 | 31 | 32 | 28 | 23 | 31 | 32 | 30 | 21 |
| 64 | 37 | 41 | 48 | 33 | 37 | 40 | 49 | 31 | 37 | 40 | 48 | 28 |
| 128 | 39 | 40 | 63 | 52 | 39 | 40 | 64 | 48 | 39 | 41 | 71 | 41 |
| 256 | 48 | 54 | 108 | 85 | 48 | 55 | 103 | 84 | 47 | 53 | 107 | 73 |
| 1 KiB | 130 | 127 | 396 | 360 | 130 | 127 | 387 | 361 | 129 | 127 | 388 | 314 |
| 4 KiB | 460 | 420 | 1482 | 1287 | 459 | 408 | 1421 | 1289 | 462 | 416 | 1440 | 1116 |
| 16 KiB | 1740 | 1444 | 5921 | 4995 | 1730 | 1434 | 5827 | 4936 | 1735 | 1426 | 6130 | 4279 |
| 64 KiB | 6901 | 5626 | 26,017 | 19,636 | 6839 | 5476 | 25,512 | 19,487 | 6882 | 5501 | 25,511 | 16,806 |

- Neither negative: up to 64 bytes, 12 to 19 cycles little-endian (up to 22 big-endian), 0 to 6 more than GMP, which needs no planning for that case either; at 128 and 256 bytes from 0.9 to 1.17 times GMP's time, and from 1 KiB all three are bound by the memory, within 8% of each other. Against num-bigint, up to 256 bytes from 1.6 times faster to 1.3 times slower (`|` of 8 bytes).
- With negative operands, GMP subtracts one from a copy of each negative operand first, and num-bigint runs a chain of carries through every limb; here the borrows and the carry end in the head, and the rest is the plain loop. From 128 bytes that is 1.3 to 2.7 times faster than GMP with one negative operand and 1.6 to 4.7 times with two, and from as fast as num-bigint (at 128 bytes) to 3.5 times faster; big-endian the most from 4 KiB. Up to 16 bytes, in registers, 2 to 10 cycles faster than GMP with two negative operands and with `^`, but `&` and `|` of 8 bytes with one negative operand take 20 to 26 cycles against GMP's 16 to 17: their result is a byte shorter about one time in fifteen, and the stores of the short path are chosen by its length.
- From 32 to 64 bytes with negative operands, the path in registers takes 29 to 41 cycles, up to 1.4 times GMP's with one negative operand and from 1.1 times to 0.75 times with two, and up to 1.5 times num-bigint's, whose limbs need no reading from bytes: the sign changes cost a chain of carries a limb for each operand and for the result, without a branch on the signs.
- With `b` in the other byte order than `a`, the kernel reads it reversed: from 256 bytes that costs 3 to 30% more than the same order; from 17 to 64 bytes, where the path for operands that are not negative needs them in one order, up to twice as long (up to 45 cycles).

The other operations, measured as above over operands of the length given, a random bit `i` among theirs, against GMP's `mpz_com` (into a ready `r` for `!&a`, which allocates here), `mpz_popcount`, `mpz_sizeinbase(a, 2)`, `mpz_scan1(a, 0)`, `mpz_tstbit`, `mpz_setbit` and `mpz_clrbit`, and num-bigint's `!a`, `!&a`, `BigUint::count_ones`, `bits`, `trailing_zeros`, `bit` and `set_bit` (for `!a`, `!&a`, `set_bit` and `count_ones`, the operands' vectors with room to spare, and for the others compact):

| Bytes | `!a`, a ≥ 0 LE | BE | GMP | num-bigint | a < 0 LE | BE | GMP | num-bigint | `!&a` LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 15 | 16 | 8 | 15 | 16 | 16 | 6 | 18 | 46 | 43 | 8 | 15 |
| 16 | 18 | 16 | 9 | 15 | 15 | 15 | 6 | 18 | 44 | 42 | 8 | 47 |
| 32 | 15 | 14 | 10 | 16 | 15 | 14 | 7 | 18 | 53 | 49 | 11 | 48 |
| 64 | 16 | 15 | 9 | 13 | 16 | 15 | 7 | 18 | 46 | 46 | 15 | 46 |
| 128 | 16 | 16 | 8 | 16 | 16 | 15 | 8 | 18 | 47 | 47 | 25 | 43 |
| 256 | 17 | 16 | 8 | 17 | 16 | 15 | 7 | 18 | 58 | 58 | 47 | 54 |
| 1 KiB | 17 | 16 | 12 | 18 | 17 | 16 | 8 | 19 | 148 | 145 | 162 | 128 |
| 4 KiB | 19 | 17 | 21 | 16 | 17 | 16 | 10 | 19 | 439 | 420 | 573 | 384 |
| 16 KiB | 22 | 20 | 47 | 19 | 20 | 19 | 26 | 20 | 1268 | 1266 | 2170 | 908 |
| 64 KiB | 26 | 26 | 26 | 22 | 24 | 24 | 19 | 22 | 3929 | 3876 | 8522 | 3693 |

| Bytes | `count_ones` LE | BE | GMP | num-bigint | `bits` LE | BE | GMP | num-bigint | `trailing_zeros` LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 7 | 7 | 9 | 6 | 6 | 8 | 6 | 4 | 6 | 6 | 6 | 5 |
| 16 | 7 | 7 | 9 | 6 | 6 | 8 | 6 | 4 | 6 | 6 | 6 | 5 |
| 32 | 11 | 11 | 8 | 8 | 7 | 8 | 6 | 5 | 7 | 7 | 6 | 5 |
| 64 | 12 | 12 | 9 | 7 | 6 | 8 | 6 | 5 | 6 | 6 | 6 | 5 |
| 128 | 13 | 13 | 11 | 8 | 6 | 7 | 6 | 5 | 6 | 6 | 6 | 6 |
| 256 | 15 | 15 | 14 | 14 | 8 | 9 | 8 | 6 | 6 | 6 | 7 | 7 |
| 1 KiB | 53 | 53 | 49 | 51 | 7 | 8 | 7 | 6 | 6 | 6 | 6 | 6 |
| 4 KiB | 195 | 195 | 260 | 202 | 9 | 10 | 9 | 7 | 8 | 8 | 9 | 8 |
| 16 KiB | 668 | 667 | 1399 | 814 | 10 | 11 | 14 | 10 | 12 | 10 | 15 | 12 |
| 64 KiB | 2447 | 2484 | 6015 | 3250 | 11 | 12 | 18 | 13 | 13 | 11 | 21 | 13 |

| Bytes | `bit`, a < 0 LE | BE | GMP | num-bigint | `set_bit(i, true)`, a < 0 LE | BE | GMP | num-bigint | `set_bit(i, false)`, a ≥ 0 LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 12 | 10 | 5 | 10 | 15 | 17 | 8 | 19 | 12 | 14 | 6 | 17 |
| 16 | 12 | 15 | 9 | 10 | 14 | 15 | 9 | 17 | 11 | 12 | 6 | 16 |
| 32 | 12 | 12 | 8 | 9 | 13 | 16 | 9 | 17 | 10 | 11 | 6 | 16 |
| 64 | 10 | 11 | 8 | 11 | 13 | 14 | 10 | 18 | 10 | 11 | 7 | 16 |
| 128 | 10 | 10 | 8 | 11 | 11 | 12 | 9 | 18 | 10 | 10 | 6 | 16 |
| 256 | 9 | 10 | 9 | 11 | 11 | 13 | 9 | 18 | 10 | 10 | 7 | 16 |
| 1 KiB | 9 | 10 | 8 | 11 | 10 | 12 | 8 | 18 | 10 | 10 | 7 | 16 |
| 4 KiB | 8 | 11 | 10 | 14 | 10 | 11 | 9 | 19 | 10 | 10 | 7 | 16 |
| 16 KiB | 10 | 12 | 9 | 17 | 11 | 14 | 12 | 20 | 11 | 13 | 8 | 18 |
| 64 KiB | 11 | 13 | 11 | 17 | 15 | 19 | 24 | 23 | 17 | 19 | 10 | 23 |

- `!a` writes eight bytes in place, in constant time, but takes `a` by value and returns it, which costs as much again as the work: about as long as num-bigint's `!a`, and 1.4 to 2.2 times GMP's `mpz_com(a, a)` up to 1 KiB, which changes `a` where it is. `!&a` allocates its result: up to 256 bytes that is most of its time; from 1 KiB it is a copy, 0.46 to 0.91 times GMP's time, and 1.06 to 1.4 times num-bigint's (whose operands here were compact, ours not).
- `count_ones` is within a few cycles of GMP's up to 1 KiB and faster beyond, 0.41 times its time and 0.75 times num-bigint's at 64 KiB; up to 128 bytes num-bigint's loop, inlined into the caller, takes a few cycles less.
- `bits` and `trailing_zeros` take 6 to 13 cycles at every length, as GMP's do; `bit` of a negative number 8 to 15, 1 to 6 more than GMP's; `set_bit` 10 to 19, up to 1.9 times GMP's (which keeps no byte order and no shortest form), and 1.2 to 1.8 times faster than num-bigint's.
- The x86_64 kernels are the portable ones, tested for correctness under emulation only; their speed has not been measured.

## Examples

Rust usage: the rules of `i32`.

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(12, ByteOrder::BigEndian);
  let b = Number::new_from_i32(-10, ByteOrder::LittleEndian);

  let and = &a & &b;
  assert_eq!(and.value, vec![4]);
  assert!(!and.is_negative());

  let or = &a | &b;
  assert_eq!(or.value, vec![2]);
  assert!(or.is_negative());

  let xor = &a ^ &b;
  assert_eq!(xor.value, vec![6]);
  assert!(xor.is_negative());

  let not = !&a;
  assert_eq!(not.value, vec![13]);
  assert!(not.is_negative());
}
```

`&=` works in the number's own vector, here as a mask in the other byte order, and a result can grow by a byte:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let mut n = Number::new_from_u32(0x1234_5678, ByteOrder::BigEndian);
  let vector = n.value.as_ptr();
  n &= &Number::new_from_u16(0xFF00, ByteOrder::LittleEndian);
  assert_eq!(n.value, vec![0x56, 0x00]);
  assert_eq!(n.value.as_ptr(), vector);

  // -0x80 & -0x81 = -0x100.
  let a = Number::new_from_i32(-0x80, ByteOrder::LittleEndian);
  let b = Number::new_from_i32(-0x81, ByteOrder::LittleEndian);
  let c = a & b;
  assert_eq!(c.value, vec![0x00, 0x01]);
  assert!(c.is_negative());
}
```

The bits of a number:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  // -0x0500 is …1111_1011_0000_0000 in two's complement.
  let mut n = Number::new_from_i32(-0x0500, ByteOrder::BigEndian);
  assert_eq!(n.bits(), 11);
  assert_eq!(n.count_ones(), 2);
  assert_eq!(n.trailing_zeros(), Some(8));
  assert!(n.bit(8) && n.bit(9) && !n.bit(10) && n.bit(11) && n.bit(1000));

  // Clearing bit 9 gives …1111_1001_0000_0000, -0x0700.
  n.set_bit(9, false);
  assert_eq!(n.value, vec![0x07, 0x00]);
  assert!(n.is_negative());

  // Setting bit 0 gives …1111_1001_0000_0001, -0x06FF.
  n.set_bit(0, true);
  assert_eq!(n.value, vec![0x06, 0xFF]);
}
```

## Efficiency

- Time complexity: O(n) for `&`, `|`, `^` and `!&a` on operands of up to n bytes, one pass over the bytes (but for a plan that reads a long run of zero or `0xFF` bytes at the low end of a negative operand once more); O(1) for `!a`, `bits`, `trailing_zeros`, `bit` and `set_bit` on random numbers in the shortest form, as the runs of borrows and carries end in their first bytes, and O(n) at worst; O(n) for `count_ones`
- Space complexity: the vector of the result, unless it is built in an operand's; nothing else
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; in place, every byte of an operand is read before it is written over; a borrowed operand is left as it was

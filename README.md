# A

![rust workflow](https://github.com/nettrash/a/actions/workflows/rust.yml/badge.svg)

The A Library is a set of algorithms.

## Usage

Add the crate with `cargo add nettrash-a`; its library is named `a`:

```rust
use a::sort::sort;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  sort(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Every algorithm has a page with its description, examples and measurements; the tables below link to them. The API reference is on [docs.rs](https://docs.rs/nettrash-a).

## Sorting

Every sort takes a mutable slice and leaves it sorted; the extra memory is what a sort uses besides the slice. [`sort`](https://github.com/nettrash/a/blob/main/lib/docs/sorting/default.md) is the default: use it when there is no reason to prefer a particular algorithm.

### Comparison sorts

These sort any `T: Ord`.

| Sort | Functions | Stable | Best | Average | Worst | Extra memory |
| --- | --- | --- | --- | --- | --- | --- |
| [Default](https://github.com/nettrash/a/blob/main/lib/docs/sorting/default.md) | `sort` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Bubble](https://github.com/nettrash/a/blob/main/lib/docs/sorting/bubble.md) | `bubble` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Selection](https://github.com/nettrash/a/blob/main/lib/docs/sorting/selection.md) | `selection` | no | O(n) | O(n²) | O(n²) | O(1) |
| [Insertion](https://github.com/nettrash/a/blob/main/lib/docs/sorting/insertion.md) | `insertion` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Cocktail Shaker](https://github.com/nettrash/a/blob/main/lib/docs/sorting/cocktail_shaker.md) | `cocktail_shaker` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Gnome](https://github.com/nettrash/a/blob/main/lib/docs/sorting/gnome.md) | `gnome` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Comb](https://github.com/nettrash/a/blob/main/lib/docs/sorting/comb.md) | `comb` | no | O(n log n) | O(n log n)¹ | O(n²)¹ | O(1) |
| [Merge](https://github.com/nettrash/a/blob/main/lib/docs/sorting/merge.md) | `merge` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Quick](https://github.com/nettrash/a/blob/main/lib/docs/sorting/quick.md) | `quick` | no | O(n) | O(n log n) | O(n log n) | O(log n) |
| [Heap](https://github.com/nettrash/a/blob/main/lib/docs/sorting/heap.md) | `heap` | no | O(n) | O(n log n) | O(n log n) | O(1) |
| [Tim](https://github.com/nettrash/a/blob/main/lib/docs/sorting/tim.md) | `tim` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Intro](https://github.com/nettrash/a/blob/main/lib/docs/sorting/intro.md) | `intro` | no | O(n) | O(n log n) | O(n log n) | O(log n) |
| [Shell](https://github.com/nettrash/a/blob/main/lib/docs/sorting/shell.md) | `shell` | no | O(n) | unknown² | O(n^(4/3))² | O(1) |
| [Drift](https://github.com/nettrash/a/blob/main/lib/docs/sorting/drift.md) | `drift` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Tree](https://github.com/nettrash/a/blob/main/lib/docs/sorting/tree.md) | `tree` | yes | O(n) | O(n log n) | O(n log n) | O(n) |
| [Cycle](https://github.com/nettrash/a/blob/main/lib/docs/sorting/cycle.md) | `cycle` | no | O(n) | O(n²) | O(n²) | O(1) |
| [Bitonic](https://github.com/nettrash/a/blob/main/lib/docs/sorting/bitonic.md) | `bitonic`, `bitonic_parallel` | no | O(n log² n) | O(n log² n) | O(n log² n) | O(log n) |
| [Odd-Even](https://github.com/nettrash/a/blob/main/lib/docs/sorting/odd_even.md) | `odd_even`, `odd_even_parallel` | yes | O(n) | O(n²) | O(n²) | O(1) |
| [Stooge](https://github.com/nettrash/a/blob/main/lib/docs/sorting/stooge.md) | `stooge` | no | O(n^2.71) | O(n^2.71) | O(n^2.71) | O(log n) |

¹ About n·log₁.₃ n comparisons in practice; no O(n log n) bound has been proved, and no known input reaches O(n²).

² No exact bound is known; measured, 1.6–2.8·n·log₂ n comparisons up to a million elements. The worst case is O(n^(4/3)) with Sedgewick's gaps (elements larger than 8 bytes); no bound is known for Ciura's (the others).

### Key sorts

These sort by integer or float keys without comparing elements. `k` is the number of possible keys between the smallest and the largest; integer keys are the primitive integers, `char` and `bool`.

| Sort | Functions | Keys | Stable | Time | Extra memory |
| --- | --- | --- | --- | --- | --- |
| [Counting](https://github.com/nettrash/a/blob/main/lib/docs/sorting/counting.md) | `counting`, `counting_by_key` | integer | `counting_by_key` | O(n + k) | O(k), O(n + k) by key |
| [Radix](https://github.com/nettrash/a/blob/main/lib/docs/sorting/radix.md) | `radix`, `radix_by_key` | integer | `radix_by_key` | O(n·b/r) for keys that differ in `b` bits, digits of `r` ≈ log₂ n bits | O(n) |
| [Bucket](https://github.com/nettrash/a/blob/main/lib/docs/sorting/bucket.md) | `bucket`, `bucket_by_key` | integer, `f32`, `f64` | `bucket_by_key` | O(n) for keys spread evenly, O(n·w) for `w`-bit keys at worst | O(n) |
| [Pigeonhole](https://github.com/nettrash/a/blob/main/lib/docs/sorting/pigeonhole.md) | `pigeonhole`, `pigeonhole_by_key` | integer | no | O(n + k) | O(k), O(n + k) by key |
| [Sleep](https://github.com/nettrash/a/blob/main/lib/docs/sorting/sleep.md) | `sleep` | integer | — | (max − min) × `unit` of sleeping | a thread per element |

Sleep sort is a joke: a thread per element sleeps in proportion to its value, the result is not always sorted, and `sleep` returns whether it is.

## Math

| Page | Contents |
| --- | --- |
| [Numbers](https://github.com/nettrash/a/blob/main/lib/docs/math/numbers.md) | `a::math::numbers::Number`: an integer of any size, stored as its sign and the bytes of its absolute value in either byte order |
| [Traits and conversions](https://github.com/nettrash/a/blob/main/lib/docs/math/traits.md) | `==`, `<`, `Ord` and `Hash` by value whatever the byte order, also with the primitive integers; `Clone`, `Default`, `Debug`, `-a`, `abs`, `signum`; `From` and `TryFrom` the primitive integers, `to_f64` and `to_f32` correctly rounded, `str::parse`; `Sum` and `Product` (as a balanced tree); `+ - * / %` with a primitive integer on either side, in place in one pass |
| [Addition](https://github.com/nettrash/a/blob/main/lib/docs/math/add.md) | `+` and `+=` for `Number`, with owned or borrowed operands: the exact sum, in the byte order of the left operand, built in an operand's vector when that has room |
| [Subtraction](https://github.com/nettrash/a/blob/main/lib/docs/math/sub.md) | `-` and `-=` for `Number`, with owned or borrowed operands: the exact difference, computed as the sum with the right operand's sign turned over, with the same promises and kernels |
| [Multiplication](https://github.com/nettrash/a/blob/main/lib/docs/math/mul.md) | `*` and `*=` for `Number`, with owned or borrowed operands: the exact product, by the schoolbook method, Karatsuba's from 24 limbs, Toom-3 from 120 and Toom-4 from 224 (Toom-4×2 for unbalanced operands), with assembly kernels on aarch64 and x86_64; `&a * &a` squares |
| [Division](https://github.com/nettrash/a/blob/main/lib/docs/math/div.md) | `/`, `/=`, `%` and `%=` for `Number`, owned or borrowed, as for `i32`: the quotient truncated toward zero, the remainder with the dividend's sign; `div_rem`, `div_euclid` and `rem_euclid`; a prepared `Divisor` for many divisions by one number; `div_exact` by Hensel's method and `is_multiple_of`. Knuth's method with Möller–Granlund reciprocals, divide and conquer from 40 limbs, and for the quotient alone a shrinking divisor for its last limbs |
| [Shifts](https://github.com/nettrash/a/blob/main/lib/docs/math/shift.md) | `<<`, `>>`, `<<=` and `>>=` for `Number` by a `usize` number of bits, owned or borrowed: `a · 2^k` and `a / 2^k` rounded towards negative infinity, as for `i32`, in one pass over the bytes, built in an owned operand's vector when that has room |
| [Roots](https://github.com/nettrash/a/blob/main/lib/docs/math/root.md) | `isqrt`, `isqrt_rem` and `nth_root` for `Number`: the square root rounded down and its remainder, and the root of any degree truncated toward zero, by Zimmermann's Karatsuba square root and Newton's method at doubling precision, in the time of a product |
| [Factorials](https://github.com/nettrash/a/blob/main/lib/docs/math/factorial.md) | `Number::factorial` and `Number::binomial`: `n!` from the prime factors of swinging factorials (Luschny), binomial coefficients as exact quotients of products or from their prime factors (Kummer, Goetgheluck), by balanced product trees |
| [Bitwise operations](https://github.com/nettrash/a/blob/main/lib/docs/math/bits.md) | `&`, `\|`, `^`, their assignments and `!` for `Number`, owned or borrowed, in infinite two's complement as for `i32`, in one pass with the borrows and carry of negative operands fused in, NEON kernels on aarch64; `bits`, `count_ones`, `trailing_zeros`, `bit` and `set_bit` |
| [Text](https://github.com/nettrash/a/blob/main/lib/docs/math/radix.md) | `Display`, `LowerHex`, `UpperHex`, `Octal` and `Binary` for `Number`, and `to_str_radix` and `from_str_radix` in any radix from 2 to 36: the sign and the digits of the magnitude, by divide and conquer for long numbers |
| [Greatest common divisor](https://github.com/nettrash/a/blob/main/lib/docs/math/gcd.md) | `gcd`, `lcm`, `gcd_ext` and `mod_inverse` for `Number`: the greatest common divisor, the least common multiple, GMP's cofactors and the inverse modulo a number, by Lehmer's method with Möller's hgcd2 and, from 300 limbs for the cofactors and 800 for the divisor alone, Möller's subquadratic half gcd on the multiplication |
| [Powers](https://github.com/nettrash/a/blob/main/lib/docs/math/pow.md) | `pow` for `Number` by a `u32` exponent, exactly, as for `i32`, and `pow_mod`, a power by a `Number` exponent modulo a `Number`, Euclid's remainder: Montgomery's multiplication with a sliding window, REDC two limbs a pass and by products from 100 limbs, in registers up to seven limbs; even moduli by the Chinese remainder theorem; not constant-time |
| [Primes](https://github.com/nettrash/a/blob/main/lib/docs/math/prime.md) | `is_probable_prime` and `next_prime` for `Number`: trial division by the small primes, then Baillie–PSW, a strong test to base 2 and a strong Lucas test with Selfridge's parameters in Montgomery's form; exact below 2^64, no known counterexample above; the next prime by sieving intervals, as GMP's `mpz_nextprime` |

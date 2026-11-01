[A](../../../README.md) › [Math](../../../README.md#math)

# Squares modulo a prime

[Numbers](numbers.md) have two methods for squares modulo an odd number: `a.jacobi(&n)` gives the Jacobi symbol, and `a.sqrt_mod(&p)` a square root modulo an odd prime.

- `jacobi` gives 0 when `a` and `n` have a common factor, otherwise 1 or −1. For a prime `n` it is the Legendre symbol: 1 for a non-zero square modulo `n`, −1 for a non-square. For a composite `n`, 1 does not make `a` a square, though −1 still rules it out. `a` may have any sign and length. It panics, with `the modulus must be odd and positive`, for an even, zero or negative `n`, at the caller's line.
- `sqrt_mod` gives the smaller of the two roots modulo the odd prime `p`, in the byte order of `self`, or `None` when `a` is not a square modulo `p`. Zero is its own root. The result is checked, so a composite `p` never gives a wrong root, though it may give `None` for a number that has one. It panics as `jacobi` does.
- `sqrt_mod` is built on [`pow_mod`](pow.md) and is not constant-time. It is not meant for secret values.

## Examples

```rust
use a::math::numbers::{ByteOrder, Number};

let n = |x| Number::new_from_i32(x, ByteOrder::BigEndian);

// Squares modulo 13: 1, 4, 9, 3, 12, 10.
assert_eq!(n(10).jacobi(&n(13)), 1);
assert_eq!(n(5).jacobi(&n(13)), -1);
assert_eq!(n(26).jacobi(&n(13)), 0);

// 6^2 = 36 = 10 modulo 13, and the other root is 7.
assert_eq!(n(10).sqrt_mod(&n(13)), Some(n(6)));
assert_eq!(n(5).sqrt_mod(&n(13)), None);
assert_eq!(n(-3).sqrt_mod(&n(13)), Some(n(6)));

// The prime of Curve25519: 2 is not a square modulo it.
let p = (n(1) << 255usize) - 19u8;
assert_eq!(n(2).jacobi(&p), -1);
assert_eq!((&p - 1u8).jacobi(&p), 1);
let root = (&p - 1u8).sqrt_mod(&p).unwrap();
assert_eq!((&root * &root).rem_euclid(&p), &p - 1u8);
```

The method:

- The Jacobi symbol by the binary method: the factors of two taken out, each turning the sign over when the modulus is 3 or 5 modulo 8, then the larger of two odd numbers less the smaller, swapped by reciprocity, which turns it over when both are 3 modulo 4; in a machine word when both fit.
- The square root by Tonelli and Shanks's method: with `p − 1 = q·2^s`, `r = a^((q + 1)/2)` has `r² = a·b` for `b = a^q`, of order dividing `2^s`, and each step multiplies `r` and `b` by powers of a generator of that subgroup, the power of a non-square, until `b` is 1. For `s = 1` it is the single power `a^((p + 1)/4)`.

## Efficiency

- Time complexity: `jacobi` O(n²) in the limbs of the modulus; `sqrt_mod` about one `pow_mod`, and O(s²) products of the modulus's length beyond for `p − 1 = q·2^s`
- Space complexity: the vectors of the operands and of the result, several for each step of `sqrt_mod`

[A](../../../README.md) › [Math](../../../README.md#math)

# Finite fields

`a::math::field` holds finite fields and their elements. A field is an object built once, which checks what it is given (that the modulus is a prime, that the polynomial is irreducible) and chooses the form of its arithmetic; its elements borrow it, as the points of [elliptic curves](ec.md) borrow their curve, and combine with `+`, `-`, `*` and `/`, owned or borrowed, with their assignments. Elements of different fields do not mix: an operation on two of them panics.

- Prime fields `GF(p)` for a prime of any length (`PrimeField`, elements `Fp`), in a form of arithmetic chosen from the shape and the length of `p`, the forms of the curves' fields among them; square roots by a method chosen from `p`, the Legendre symbol, inversion one at a time and in batches, encodings of a fixed length, and the base and scalar fields of BN254, BLS12-381, secp256k1, P-256, P-384, P-521 and Curve25519 ready made.
- Prime fields known at compile time (`TypedPrime`, elements `TypedFp<P, N>`): `Copy` elements of `N` limbs and no field object, on the same kernels, the named primes ready made in `field::named` and others declared by `typed_prime!`.
- Extension fields `GF(pᵏ)` (`ExtensionField`, elements `Fpk`), by a given monic irreducible polynomial or by the smallest one of a degree: the Frobenius map, norms, traces, square roots, multiplicative orders and primitive elements.
- The towers `F_p² → F_p⁶ → F_p¹²` (`Tower`, elements `Fp2`, `Fp6`, `Fp12`) that [pairings](pairing.md) compute in, BN254's and BLS12-381's ready made: Karatsuba's products, the sparse products by the lines of a pairing, Frobenius maps by constants, Granger and Scott's squares in the cyclotomic subgroup, and square roots in `F_p²`.
- Binary fields `GF(2ᵐ)` (`BinaryField`, elements `F2m`) by any irreducible polynomial, multiplied by the CPU's carry-less product (`PMULL`, `PCLMULQDQ`) or a portable kernel: inverses, square roots, traces, half-traces and quadratic equations; and Gaussian normal bases (`NormalBasis`, elements `NormalElement`), with their conversions to and from a polynomial basis (`BasisConversion`), as FIPS 186-4 gives them.
- `GF(2⁸)` and `GF(2¹²⁸)` as AES and GCM use them (`Gf2_8`, `Gf2_128`), small `Copy` types with no field object, with the AES S-box and GCM's GHASH.

**Nothing here is constant-time.** The time of an inversion, a square root, a power, a comparison or a product depends on the values, and no care is taken against timing or cache attacks. These fields are for mathematics and for verifying with public data, as in checking a signature, not for secret keys; `Gf2_8` and `Gf2_128` are the field arithmetic of AES and GCM, not implementations of them.

## Examples

```rust
use a::math::field::{BinaryField, ExtensionField, Fp2, Gf2_8, Gf2_128, PrimeField, Tower};
use a::math::numbers::Number;

// A prime field, here BLS12-381's base field: Fermat's little theorem, and a quotient.
let f = PrimeField::bls12_381_base();
let x = f.from_u64(3);
assert_eq!(x.pow(&(f.modulus() - &Number::from(1u8))), f.one());
assert_eq!(&(&x / &f.from_u64(7)) * &f.from_u64(7), x);

// Its quadratic extension F_p[x]/(x² + 1), as a general extension field and as the tower of the
// pairings holds it.
let fp2 = ExtensionField::with_degree(f, 2);
assert_eq!(fp2.x().square(), -fp2.one());
let t = Tower::bls12_381();
let z = Fp2::from_numbers(t, &Number::from(3u8), &Number::from(4u8));
assert_eq!(z.norm(), f.from_u64(25)); // 3² + 4²
assert_eq!(fp2.from_u64(&[3, 4]).norm(), z.norm());

// AES's field GF(2⁸) = GF(2)[x]/(x⁸ + x⁴ + x³ + x + 1), as a binary field and as a small type.
let g = BinaryField::new(&[8, 4, 3, 1, 0]);
assert_eq!((&g.element(&[0x57]) * &g.element(&[0x83])).words(), &[0xc1]); // FIPS 197
assert_eq!(Gf2_8::new(0x57) * Gf2_8::new(0x83), Gf2_8::new(0xc1));

// GCM's field GF(2¹²⁸): x¹²⁷ · x = x⁷ + x² + x + 1.
let x127 = Gf2_128::from_polynomial(1 << 127);
assert_eq!(x127 * Gf2_128::from_polynomial(2), Gf2_128::from_polynomial(0x87));
```

The sections below give each part's functions, its method and sources, an example, what it was verified against, and its limits; the [measurements](#measured) close the page.

## Prime fields

`PrimeField` is the field `GF(p)` of the integers modulo a prime `p`, of any length, and `Fp` an element of it, which borrows its field. `PrimeField::new` tests `p` for primality and chooses the form of the arithmetic once, from the shape and the length of `p`, reusing the fields of the curves of [`a::math::ec`](ec.md); every operation then matches that form once and runs code compiled for it. An element of a field of up to nine limbs (576 bits) holds its limbs inline, so that sums, differences, products, negation, squares and doubling, with their assignments, comparisons, hashes, `sgn0` and the conversions from words and from bytes of the exact length allocate nothing; inversion, the Legendre symbol, the reduced decodings, powers and square roots allocate, as the limits say. Like the rest of the library it is not constant-time.

| Item | What it is |
| --- | --- |
| `PrimeField::new(&p)` | the field modulo `p`; panics if `p` is not a prime (`Number::is_probable_prime`) |
| `modulus()`, `characteristic()`, `order()`, `bits()`, `byte_len()` | `p`, three times, its bits, and the bytes of an element's encoding, `⌈bits / 8⌉` |
| `form()`, `sqrt_method()` | the names of the form of the arithmetic and of the method of square roots the field chose |
| `element(&x)`, `from_u64(x)`, `zero()`, `one()` | the element congruent to a `Number` of any sign and length, or to a word |
| `from_bytes_be(&b)`, `from_bytes_le(&b)` | the element of an encoding of exactly `byte_len()` bytes, `None` for another length or a value not below `p` (the field-element encodings of SEC 1 and ZCash) |
| `from_bytes_be_reduced(&b)`, `from_bytes_le_reduced(&b)` | the element congruent to bytes of any length, an empty slice being 0 (RFC 9380's `OS2IP(tv) mod p`) |
| `batch_invert(&xs)` | the inverses of many elements by Montgomery's trick, zeros left as zeros |
| `bn254_base()`, `bn254_scalar()`, `bls12_381_base()`, `bls12_381_scalar()`, `secp256k1_base()`, `secp256k1_scalar()`, `p256_base()`, `p256_scalar()`, `p384_base()`, `p384_scalar()`, `p521_base()`, `p521_scalar()`, `curve25519_base()`, `curve25519_scalar()` | the named fields, `&'static`, built on the first call: the base field and the field of the group order of each curve (Curve25519's serves Ed25519) |
| `a + b`, `a - b`, `a * b`, `a / b`, `-a`, `+=`, `-=`, `*=`, `/=` | on every combination of owned and borrowed operands; elements of fields of different primes panic ("elements of different fields"), `/` by 0 panics ("division by zero"); two fields built from the same prime mix |
| `square()`, `double()`, `pow(&e)`, `pow_u64(e)` | a negative `e` is a power of the inverse; an exponent more than 64 bits longer than `p` is first reduced modulo `p − 1` |
| `square_in_place()`, `double_in_place()`, `neg_in_place()` | the same in the element itself, which rewrites only the form's limbs (as `*=` does), where `x = x.square()` builds and moves a whole element |
| `inverse()` | `None` for 0 |
| `sqrt()` | the smaller of the two roots in `[0, p)`, `None` for a non-square |
| `legendre()`, `is_square()` | 1, −1 or 0; whether a square, 0 included (RFC 9380's `is_square`) |
| `sgn0()` | the parity of the value, RFC 9380's `sgn0` for `m = 1` |
| `value()`, `to_bytes_be()`, `to_bytes_le()` | the value in `[0, p)` as a `Number`, or in exactly `byte_len()` bytes |
| `is_zero()`, `is_one()`, `field()`, `==`, `Hash`, `Debug`, `Display` | equality and hashing by the value, whatever the representation; `Debug` prints `x (mod p)` |

### Forms

The first form that fits is taken:

| `p` | Form | Its elements and products |
| --- | --- | --- |
| 2 | `GF(2)` | a bit: a sum is an exclusive or, a product an and |
| P-256's and P-384's primes | `P-256`, `P-384` | Montgomery's form, with the curves' products for these primes on aarch64 and Montgomery's elsewhere |
| `2^521 − 1` | `P-521, limbs of 58 bits` | nine limbs of 58 bits, not reduced: a product is the 81 products of limbs added in columns of 128 bits, with no carry chain, and folded by `2^522 = 2` |
| `2^k − c` of 4, 6 or 9 limbs, `c < 2^(62 − s)` for `s = 64n − k` (secp256k1's, Curve25519's) | `2^k - c, n limbs` | any `n` limbs, not reduced: the high half of a product comes back as a product by `c·2^s` |
| any other odd prime of 1 to 9 limbs | `Montgomery, n limbs` | `x·2^(64n) mod p`, fully reduced; a product is Montgomery's reduction interleaved with the rows of the product below seven limbs, and the multiplication's kernels with a separate reduction from seven |
| longer primes | `Montgomery, more than 9 limbs` | Montgomery's form in a vector of the prime's limbs: a product by the numbers' schoolbook kernel and its REDC, the assignments in the element's own vector; powers by `Number::pow_mod`, inverses by `Number::mod_inverse`. An element made from its value (`element`, `from_u64`, the decodings) keeps the value until an operation needs the residue, which it then makes once, so that what is decoded and encoded again, or hashed, is not converted |

A product of `n` limbs costs `O(n²)` products of words, a sum `O(n)`. The forms that do not reduce their elements fully (`2^k − c` and P-521) make the value unique only when it is read: `==`, `Hash`, `is_zero`, `value`, `to_bytes_*`, `sgn0`, the comparisons of the square root. The primes of P-256, P-384 and P-521 are recognised by their limbs, whatever byte order the `Number` was given in.

### Square roots

With `p − 1 = q·2^s`, `q` odd, the field chooses its method once, when it is built, and finds what it needs then:

| `p` | Method | Cost |
| --- | --- | --- |
| `p ≡ 3 (mod 4)` | `a^((p + 1)/4)`, checked by squaring; for BLS12-381's base prime by a plan of the power made once (`prime_chains.rs`, generated by `chains.py`: the sliding windows of the exponent of the fewest operations, 375 squarings and 66 products after 16 odd powers) | one power |
| `p ≡ 5 (mod 8)` | Atkin's: `b = (2a)^((p − 5)/8)`, `i = 2a·b²`, the root `a·b·(i − 1)`, checked by squaring | one power |
| `p ≡ 1 (mod 8)`, `s² ≤ 12·b` for the `b` bits of `p` | Tonelli and Shanks's, with the least non-square `z` and `c = z^q` found once per field: `r = a^((q + 1)/2)`, then corrections by powers of `c` that lower the order of `a^q` | one power and up to `s(s + 1)/2 − 1` squarings for a square (a round costs `m` squarings for a bound `2^m` on that order, and `m` falls from `s` to 2 at worst), `s` for a non-square |
| `p ≡ 1 (mod 8)`, `s² > 12·b` | Cipolla and Lehmer's: the Legendre symbol of `a`, a `t` with `t² − a` a non-square, and `(t + w)^((p + 1)/2)` in `F_p[w]/(w² − (t² − a))` | about 5.5 products a bit of `p` and a few Legendre symbols, whatever `s` is |

The bound between the last two, on the bit length `b` (not `log₂ p`, a little less), is the point where Tonelli and Shanks's squarings, about `s²/4` for a random square and `s(s + 1)/2 − 1` at worst, pass the products Cipolla's power costs more than one power in `F_p`. BLS12-381's scalar field (`s = 32`) and BN254's (`s = 28`) take Tonelli and Shanks's method; the 64-bit Goldilocks prime `2^64 − 2^32 + 1` (`s = 32`), BabyBear's `15·2^27 + 1` and `65537` take Cipolla's. The result is always the smaller root, `r ≤ (p − 1)/2`, whatever root the method finds, but for `GF(2)`, where 1 is its own and only root.

### Inversion, the Legendre symbol and powers

- `inverse` and `/` take Lehmer's extended Euclid of `Number::mod_inverse` (see [the GCDs](gcd.md)) on the value, which leaves the field's form for one short number: `O(n²)`, against the `O(n³)` of Fermat's `x^(p − 2)`. Bernstein and Yang's division steps on limbs (2019), which would stay in the form, were tried in its place and took longer (see the limits).
- `legendre` and `is_square` take `Number::jacobi` on the value: the binary method up to six limbs and Lehmer's steps with the symbol carried through the quotients beyond, `O(n²)`, against Euler's criterion `a^((p − 1)/2)`, one power.
- `batch_invert` takes Montgomery's trick: the running products, one inversion and two products back for each element, three products for each nonzero element in all; a zero stays zero, as in arkworks' `batch_inversion`.
- `pow` and `pow_u64` take the curves' exponentiation: sliding windows of up to five bits, and the long runs of ones of exponents such as `(p + 1)/4` for the named primes appended by products by `x^(2^j − 1)`.

### Example

```rust
use a::math::field::PrimeField;
use a::math::numbers::Number;

// BLS12-381's scalar field: p − 1 has 32 factors of two, so square roots take Tonelli and Shanks's method.
let f = PrimeField::bls12_381_scalar();
assert_eq!((f.form(), f.sqrt_method()), ("Montgomery, 4 limbs", "Tonelli-Shanks"));
let x = f.from_u64(1234567);
let r = x.square().sqrt().unwrap();
assert!(r == x || r == -&x);
assert!(r.value() <= (f.modulus() - &Number::from(1u8)) >> 1usize); // the smaller root

// Inverses, one at a time and all at once.
let xs: Vec<_> = (1..=4u64).map(|k| f.from_u64(k)).collect();
let inv = f.batch_invert(&xs);
assert_eq!(&inv[3] * &xs[3], f.one());
assert_eq!(inv[2], xs[2].inverse().unwrap());

// secp256k1's prime 2^256 − 2^32 − 977 has the form 2^k − c, P-521's its own.
let k = PrimeField::secp256k1_base();
assert_eq!(k.form(), "2^k - c, 4 limbs");
let minus_one = k.zero() - k.one();
assert_eq!(minus_one.value(), k.modulus() - &Number::from(1u8));
assert_eq!(&minus_one + &k.one(), k.zero());
assert_eq!(PrimeField::p521_base().form(), "P-521, limbs of 58 bits");

// Encodings of a fixed length, and bytes of any length reduced.
let bytes = minus_one.to_bytes_be();
assert_eq!(bytes.len(), 32);
assert_eq!(k.from_bytes_be(&bytes), Some(minus_one.clone()));
assert!(k.from_bytes_be(&[0xff; 32]).is_none()); // not below p
assert_eq!(k.from_bytes_be_reduced(&[0xff; 32]), k.from_u64(0x1_0000_03d0)); // 2^256 − 1 mod p

// RFC 9380's helpers, here in GF(13).
let g = PrimeField::new(&Number::from(13u8));
assert!(g.from_u64(3).sgn0() && !g.from_u64(10).sgn0());
assert!(g.from_u64(10).is_square() && !g.from_u64(5).is_square()); // 6² = 36 = 10
assert_eq!(g.from_u64(10).sqrt(), Some(g.from_u64(6)));
```

### Verified

- **Every form against `Number` modulo `p`.** 79 primes, each given the form and the method of square roots the test expects (`form()`, `sqrt_method()`: all 17 forms and all 5 methods occur): 2, 3, 5, 41, 65537, BabyBear `15·2^27 + 1`, `3·2^30 + 1`, `2^61 − 1`, Goldilocks; the largest prime below `2^(64k)` and the smallest above `2^(64(k − 1))` for every `k` from 1 to 9 (the top limb full, and the top limb 1); `2^383 − 31` and `2^570 − 261`; primes `2^(b − 1) + d·2^s + 1` of 256 bits with `s` = 3, 50, 56 and 64, of 384 with `s` = 4, 512 with `s` = 5 and 600 with `s` = 20 and 90; primes of 577, 600 and 1500 bits for `Number`; the edges of the choice of the form `2^k − c`, which takes `c < 2^(62 − s)` for `s = 64n − k`: for 4, 6 and 9 limbs and `s` = 0, 13, 32 and 55 (50 for six limbs), the largest `c` below the bound, whose folds come nearest `2^62`, and the smallest above it, which takes Montgomery's form (at `s = 0` with a full top limb, whose reductions carry out of the limbs), and `2^201 − 55` and `2^529 − 31`, small `c` at `s` = 55 and 47 (26 primes); and the 14 named fields. On 2188 seeded pairs of inputs of either sign and up to 70 bits longer than `p` (40, 16 or 4 a field by its length), with `p − 1`, `p` and 1 among them: `element`, `+`, `−` both ways, `×`, negation, `square`, `double`, `inverse`, `/`, `pow_u64` of a random word, `pow` of a random exponent, `from_u64`, `is_zero`, `is_one`, `==` and `sgn0` agree with `Number`'s `rem_euclid`, `mod_inverse` and `pow_mod`; `legendre` and `is_square` with `Number::jacobi`; every root squares back, is the smaller one, and exists exactly when the symbol is not −1, and the root of `a²` is `min(a, p − a)`. The same with 25 times the inputs (54 700 pairs) in an ignored test. The primes given as big-endian `Number`s make the same fields, mixing with those of little-endian ones.
- **PARI/GP 2.19.** For 40 primes from 3 to 1500 bits (every form but `GF(2)`, every class of `p` modulo 8, `s` up to 90), the Legendre symbols (`kronecker`) and the smaller square roots (`sqrt` in `Z/pZ`) of 24 inputs each (8 above 580 bits), `((p \ 3)·i³ + (p \ 7)·i + 1) mod p`, which the test computes again: all 896 symbols agree, with `Number::jacobi` too, and all 489 roots of the squares among them (16 of them roots of 0) (`lib/tests/math/field/data/prime_reference.txt`, 35 KB, written by `prime_reference.gp` beside it).
- **Exhaustively.** The tables of `+`, `−`, `×`, `/`, negation, inverses, powers, symbols and roots of `GF(2)` and `GF(3)`; for the 23 primes from 3 to 18433 that cover every class modulo 8 and `s` from 1 to 12, the root and the symbol of every residue against a table of squares, and of every 97th residue of 65537 (Cipolla's method).
- **Representations.** Sums that reach `p` without a carry, such as `(p − 1) + 1`, which the forms `2^k − c` and P-521 hold as `p`: for every prime, five ways to 0, each compared, hashed, read as a value, as bytes and as a sign, inverted, rooted and given its symbol; `(p − 1) + 2`, compared with 1, hashed, read as a sign and inverted; and `2(p − 1)`, read as `p − 2` and as a sign, squared and rooted. Equal elements reached by different paths hash alike.
- **Bytes.** For every prime, round trips in both orders of `byte_len()` bytes; `p − 1` accepted, `p`, the all-ones string and every other length refused; the reduced decodings of 0, 1, `byte_len() ± 1`, `byte_len()`, twice that and 64 bytes against `Number`.
- **Inversions in batches**, in every form: with zeros, of zeros only, of one element, of none; each against `inverse()`.
- **Named fields.** Their primes are the moduli and orders of `ec::named`'s secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519, and BN254's and BLS12-381's follow from their parameters (`36u⁴ + 36u³ + 24u² + 6u + 1` and `18u²` for the order, with `u = 4965661367192848881`; `(x − 1)²(x⁴ − x² + 1)/3 + x` and `x⁴ − x² + 1` for `x = −0xd201000000010000`).
- **Allocations.** A counting allocator sees none in a sequence of `+`, `−`, `×`, negation, `+=`, `−=`, `×=`, `square`, `double`, `from_u64`, `zero`, `one`, `from_bytes_be`, `from_bytes_le`, `==`, `is_zero`, `is_one`, `sgn0`, a hash and a clone, in each of the 74 fields of up to nine limbs; and exactly what the limits say for `inverse`, `/`, `/=` (2), `legendre` and `is_square` (1 up to six limbs, at least 1 beyond), and `from_bytes_be_reduced` and `from_bytes_le_reduced` of the exact length and of twice it and more (3; 0 for an empty slice); in `GF(2)` none of them, but the one of a reduced decoding, the bytes as a number.
- **Panics**: a composite modulus, 0, 1 and a negative one; elements of different fields in `+`, `−`, `×`, `/`, an assignment and a batch; a division by zero (in a short and a long field, and `/=`); a negative power of 0.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 38 tests pass, with the ignored sweep.

To run the tests: `cargo test --test math -- field::prime`, and `cargo test --test math -- --ignored field::prime` for the larger sweep. `gp -q lib/tests/math/field/data/prime_reference.gp > prime_reference.txt` writes the reference again.

### Limits

- Not constant-time: inversions, square roots, powers, the Legendre symbol and the comparisons of the forms `2^k − c` and P-521 take times that depend on the values. These fields are for mathematics and public data, not secret keys.
- What allocates in a field of up to nine limbs: `inverse`, `/` and `/=` twice (the value as a `Number`, and the inverse `Number::mod_inverse` returns); `legendre` and `is_square` once (the value) up to six limbs, and beyond that whatever working memory `Number::jacobi` takes; `from_bytes_be_reduced` and `from_bytes_le_reduced` three times (the bytes as a number, its remainder and the remainder's limbs), not for an empty slice; `pow`, `pow_u64` and `sqrt` their tables of powers, in vectors; `element`, `value`, `to_bytes_*`, `batch_invert`, `Display` and `Debug` their numbers and vectors. `GF(2)` inverts and takes symbols without allocating. Bernstein and Yang's division steps on limbs, which would allocate nothing, were measured slower than `Number::mod_inverse` from four limbs to nine and left out.
- Unlike the other operations, inversion, `/` and the Legendre symbol take no advantage of the field's form: each converts the value to a `Number` and calls `Number::mod_inverse` or `Number::jacobi`, so they cost what those cost on a number of the length of `p`, the conversion included, and are no faster than a field held in `Number`s. Lehmer's steps on the limbs, which would stay in the form, are not in.
- Elements of primes beyond nine limbs are vectors of limbs: `+=`, `-=` and `*=` write into the element's own vector, the operators that make a new element allocate one.
- The form `2^k − c` is ec's, for four, six and nine limbs only; such primes of other lengths, `2^127 − 1` among them, take Montgomery's form.
- An `Fp` takes 88 bytes, a pointer and nine limbs with their tag, whatever the length of `p`; a `PrimeField` 832.
- `PrimeField::new` trusts `Number::is_probable_prime`, which no composite is known to pass; a composite that passed would give a ring whose inverses and roots are wrong.
- The bound `s² > 12·b`, `b` the bits of `p`, between Tonelli and Shanks's method and Cipolla's comes from counting products; it was not tuned per machine.
- Clippy's `mutable_key_type` lint flags a `HashSet<Fp>` or a `HashMap` keyed by `Fp`: the field of a long prime holds a `Divisor`, which prepares its Newton inverse lazily in a cell. The hash is the value's alone, so the lint can be allowed.
- `from_bytes_*` return `None` for a wrong length, as for a value not below `p`; they do not panic.

### Typed prime fields

A prime known when the program is compiled can be a type: `TypedPrime<N>` gives its `N` limbs (and a name), and `TypedFp<P, N>` is an element, `Copy`, its `N` limbs and nothing else (in the forms below), so that a computation keeps its values in registers and passes them by value, with no field object to borrow and no form to match at run time. The constants of the arithmetic — `-1/p mod 2⁶⁴` and `mod 2¹²⁸`, `R mod p` and `R² mod p` (by doublings), the exponents of the square roots, the least non-residue (by Legendre symbols of words) and its power `z^q` for Tonelli and Shanks's method — are computed by `const fn`s when the type is first used, and the properties the arithmetic needs (odd, exactly `N` limbs, `N` from 1 to 9) fail the compilation when they do not hold. That the modulus is prime is not checked at compile time: `TypedFp::modulus_is_prime()` tests it, and a test of the crate that declares the type should call it.

| | |
| --- | --- |
| `TypedPrime<N>` (`MODULUS: [u64; N]`, `NAME`) | the parameters; `typed_prime!` declares a unit type and its implementation from the prime in hexadecimal (`limbs_from_hex`, a `const fn`) |
| `field::named::{Bn254Base, Bn254Scalar, Bls12_381Base, Bls12_381Scalar, Secp256k1Base, Secp256k1Scalar, P256Base, P256Scalar, P384Base, P384Scalar, Curve25519Base, Curve25519Scalar, Goldilocks}` | the named primes, with the element types `Bn254BaseFp` (`TypedFp<Bn254Base, 4>`) and the like |
| `ZERO`, `ONE`, `from_u64`, `from_number`, `from_bytes_be`, `from_bytes_le`; `value`, `to_bytes_be`, `to_bytes_le` | as `Fp`'s, the encodings of exactly `byte_len()` bytes |
| `to_fp(&field)`, `from_fp(&x)` | the element of the runtime `PrimeField` of the same prime and back (the Montgomery forms of four and six limbs hold the same residues: no conversion); another prime panics |
| `+ - * /`, `-x`, `+= -= *= /=` | on owned and borrowed operands; `/` by 0 panics |
| `square`, `double`, `inverse`, `pow`, `pow_u64`, `sqrt`, `legendre`, `is_square`, `sgn0`, `batch_invert` | what `Fp`'s compute, the smaller square root included. The inverse and the symbol go through `Number` from two limbs (Lehmer's extended Euclid, `Number::jacobi`) and stay on the word for a prime of one limb (the binary extended Euclid; the Jacobi symbol by reciprocity and remainders of words). The square root is `a^((p + 1)/4)` for `p ≡ 3 (mod 4)` (for BLS12-381's base prime by the runtime field's plan of the power), Atkin's method for `p ≡ 5 (mod 8)`, and Tonelli and Shanks's for every `p ≡ 1 (mod 8)`, where `Fp` takes Cipolla's for `s² > 12·b` (Goldilocks's prime, `s = 32`, among them): the same root by another method |
| `modulus()`, `modulus_limbs()`, `bits()`, `byte_len()`, `modulus_is_prime()` | the prime |

The products, squares, sums and differences are the runtime field's kernels, chosen by the constants at compile time. Most primes are in Montgomery's form, `x·2^(64N) mod p` fully reduced: on aarch64 the no-carry products of four and six limbs for a prime with a spare top bit and P-256's and P-384's reductions, the carry chains of the sums; Montgomery's rows up to six limbs and the numbers' product and REDC from seven, elsewhere. Two kinds of prime hold values instead. A prime `2^k − c` of four limbs with `c·2^s < 2^62` (`s = 256 − k`: secp256k1's, `2²⁵⁵ − 19`) is in the runtime field's form `2^k − c` on aarch64: any four limbs congruent to the value, the high half of a product folded back by `c·2^s`. Goldilocks's prime `2^64 − 2^32 + 1` holds its value, below `p`, on every target, a product reduced by the prime's shape (`2^64 ≡ 2^32 − 1`, `2^96 ≡ −1`); the runtime field holds it in Montgomery's form of one limb. The other primes the runtime field holds in another form — `2^k − c` of six and nine limbs, those of four on other targets, P-521's limbs of 58 bits — are in Montgomery's form here, which on this machine is slower than those forms.

```rust
use a::math::field::{PrimeField, TypedFp};
use a::math::field::named::Bn254BaseFp;

a::typed_prime! {
    /// 2^127 - 1.
    pub M127, 2, "7fffffffffffffffffffffffffffffff"
}
type F = TypedFp<M127, 2>;
assert!(F::modulus_is_prime()); // in a test of the declaration
let x = F::from_u64(3);
assert_eq!(x * x.inverse().unwrap(), F::ONE);

let y = Bn254BaseFp::from_u64(5);
let z = y * y + Bn254BaseFp::ONE; // copied, no field to borrow
assert_eq!(z.to_fp(PrimeField::bn254_base()), PrimeField::bn254_base().from_u64(26));
```

## Extension fields

`ExtensionField` is the field `GF(pᵏ) = GF(p)[x] / g(x)` for a monic irreducible polynomial `g` of degree `k` over a `PrimeField` `GF(p)`, any `p` the prime fields take (2 and 3 included) and any degree from 1 (which is `GF(p)` itself) on. Its elements, `Fpk`, are the polynomials of degree below `k` in the class of `x`, a root of `g`, held as their `k` coefficients in `GF(p)`, the constant first. The field is built from a given `g`, which is checked to be irreducible, or from a degree, for which it takes the smallest monic irreducible polynomial (below). It holds a copy of its base field, cloned once, so that it has no lifetime and can be kept in a structure or returned from a function; its elements borrow it, as the elements of a `PrimeField` borrow theirs, and their coefficients are `Fp`s of that copy, which mix with the elements of any `PrimeField` of the same prime. Two fields of the same `p` and `g` are equal, and their elements mix too; elements of fields that differ make an operation panic ("elements of different fields"). What only some uses need is computed on the first use and kept in the field: the matrix of the Frobenius map, the traces of the powers of `x`, the data of the square roots, the factorisation of `pᵏ − 1`, and the tables of logarithms of a small field (below).

The towers `F_p² → F_p⁶ → F_p¹²` of the pairings are types of their own, made for them (below); `ExtensionField` is the general construction, for any `p`, `g` and `k`. Like the rest of the module it is not constant-time.

| Function | What it gives | Method and cost (`k` the degree; products and sums are in `GF(p)`) |
| --- | --- | --- |
| `ExtensionField::new(&base, &g)` | `GF(p)[x] / g` for a monic irreducible `g` of degree at least 1, a `ModPolynomial` modulo `p` | Ben-Or's test of irreducibility of [`ModPolynomial::is_irreducible`](poly.md); panics for a reducible, constant or non-monic `g`, or one modulo another prime |
| `ExtensionField::with_degree(&base, k)` | `GF(pᵏ)` by the polynomial `first_irreducible` gives | below |
| `ExtensionField::first_irreducible(&base, k)` | the smallest monic irreducible polynomial of degree `k` modulo `p` | below |
| `base()`, `characteristic()`, `degree()`, `modulus()`, `order()` | the copy of `GF(p)`, `p`, `k`, `g`, and `pᵏ` | kept |
| `element(&[Number])`, `from_u64(&[u64])`, `from_fp(&[Fp])`, `from_polynomial(&ModPolynomial)` | the element of a polynomial given by its coefficients, the constant first, of any sign, size and number | each coefficient reduced modulo `p`, the polynomial modulo `g` |
| `from_base(&Fp)`, `zero()`, `one()`, `x()` | a constant; 0; 1; the class of `x` (for `k = 1` the root of `g` in `GF(p)`) | |
| `coeffs()`, `to_polynomial()`, `to_base()` | the `k` coefficients; the `ModPolynomial` of degree below `k`; the element as an `Fp` if it is a constant | |
| `+`, `-`, `-x`, `*`, `/` and their assignments, on owned and borrowed operands; `square()`, `double()` | | sums by coefficient; a product is the schoolbook product, `k²` products, reduced by the terms of `xᵏ mod g`, `(k − 1)·w` more for the `w` terms of `g` below `xᵏ`; from a degree set by the size of `p` (below) the product and the remainder of `ModPolynomial` (transforms and Newton's division for long polynomials); `/` panics on 0 ("division by zero") |
| `inverse()` | the inverse, `None` for 0 | the extended Euclidean algorithm on the element and `g`: on the coefficients' words for a prime of one word below `2⁶³` up to degree 64, else `ModPolynomial::xgcd` (by the half gcd for long polynomials); a difference of logarithms in a field of Zech's tables |
| `pow(&Number)`, `pow_u64(u64)` | `aᵉ`; a negative `e` is a power of the inverse; `a⁰ = 1` | squares and products from the top bit of `e`, reduced modulo `pᵏ − 1` first: at most `2k·log₂ p` products in `GF(pᵏ)` |
| `frobenius(i)` | `a^(pⁱ)`, `i` modulo `k` | `i` products by the matrix of `a ↦ aᵖ`, `O(i·k²)` (on the coefficients' words for a prime of one word below `2⁶³`); the matrix, `k²` elements of `GF(p)`, on the first call; a product of the logarithm by `pⁱ` in a field of Zech's tables |
| `norm()`, `trace()` | `N(a) = a·aᵖ·…·a^(pᵏ⁻¹)` and `Tr(a) = a + aᵖ + … + a^(pᵏ⁻¹)`, in `GF(p)` | the resultant `res(g, a)`, `O(k²)`, a closed form of three products for `k = 2`, a power of the logarithm in a field of Zech's tables; `Σ aᵢ·Tr(xⁱ)`, `k` products in the base field's form, the traces of the powers of `x` on the first call, `O(k²)` |
| `legendre()`, `is_square()` | the quadratic character (1, −1, 0 for 0); whether the element is a square | the Legendre symbol of the norm for odd `p`; every element for `p = 2` |
| `sqrt()` | a square root, `None` for a non-square: of `r` and `−r` the one whose highest nonzero coefficient is below `p/2` (the smaller as the integer `Σ rᵢ·pⁱ`, as `Fp::sqrt` gives the smaller root); 0 for 0 | the norm first; Tonelli and Shanks's algorithm in `GF(pᵏ)` for odd `p`, one power of about `k·log₂ p` bits and at most `s²` products for `pᵏ − 1 = 2ˢ·t`; one product for `p = 2` |
| `multiplicative_order()`, `is_primitive()`; `primitive_element()` | the order of an element, `None` for 0; whether it generates the group of units; the first element that does | from the factorisation of `pᵏ − 1`, computed once by [`Number::factor`](factor.md) on its cyclotomic factors or given by `with_factored_unit_order` |
| `batch_invert(&[Fpk])` | the inverses of many elements, a zero staying zero | Montgomery's trick: one inversion and `3(n − 1)` products |
| `factored_unit_order()`, `with_factored_unit_order(Factorization)` | the factorisation of `pᵏ − 1`; the field with it given (its product checked) | |

`Fpk` is `Clone`, `PartialEq`, `Eq`, `Hash` (by its coefficients), `Debug` (`3*x^2 + 1 (mod 7, x^3 + 2)`) and `Display` (`3*x^2 + 1`, as PARI writes a polynomial); `ExtensionField` is `Clone`, `PartialEq`, `Eq`, `Hash` and `Debug`.

### The choice of `g`

`first_irreducible(&base, k)`, and so `with_degree`, takes the smallest monic irreducible polynomial of degree `k` in the order of the coefficients from `xᵏ⁻¹` down to the constant, each in `0..p`: the order of the integers `Σ gᵢ·pⁱ`. It is `x` for `k = 1`, `x⁸ + x⁴ + x³ + x + 1` (AES's) for `p = 2` and `k = 8`, and `x² + 1` for the primes of BN254 and BLS12-381, which are 3 modulo 4.

- The binomials `xᵏ + c` come first in that order. By Lidl and Niederreiter's Theorem 3.75 (*Finite Fields*, 1997), for `k ≥ 2`, `xᵏ − a` is irreducible over `GF(p)` if and only if every prime factor `r` of `k` divides the order of `a` but not `(p − 1)/ord(a)`, and `p ≡ 1 (mod 4)` when `4 | k`. The first condition is that each `r` divides `p − 1` and `a^((p − 1)/r) ≠ 1`, so the theorem settles all `p` binomials at once: if none is irreducible the search skips them all, and otherwise it takes the first `c` for which `−c` is not an `r`-th power, with `ω(k)` powers modulo `p` a candidate.
- Then `xᵏ + x + c` for `c = 1, 2, …` (the constant 0 leaves the factor `x`), then the polynomials with `2x`, …, `x²` and on, as an odometer. Each candidate is tested by Ben-Or's test, after a test of its discriminant for odd `p`: by Stickelberger's theorem (Swan, *Factorization of polynomials over finite fields*, Pacific J. Math. 12, 1962), a square-free polynomial of degree `k` with `r` irreducible factors has a discriminant that is a square modulo `p` exactly when `k − r` is even, so an irreducible one has the Legendre symbol `(−1)^(k−1)`, which about half of the candidates lack. About one polynomial in `k` is irreducible, so for a large `p` the search stops after about `k` candidates: 26, 9 and 37 for BN254's prime and `k` = 12, 24 and 64, and 23, 11 and 253 for BLS12-381's. For a small `p` every candidate is taken in turn.

PARI's `ffinit` makes another, deterministic choice (its own construction, not the smallest polynomial: for `p = 2`, `k = 8` it gives `x⁸ + x⁶ + x⁵ + x⁴ + x³ + x + 1`), and FLINT's `fq_ctx_init` a Conway polynomial from its table or a random one. The fields are isomorphic, but the coordinates of their elements differ; the tests build fields from PARI's polynomials as well as from this choice. Conway polynomials, which make the fields of different degrees compatible, are not offered.

### Methods

- **How the elements are held.** For a field of Zech's tables (below), an element is its logarithm, one word; for a prime of one word below `2⁶³` otherwise, the words of its coefficients in the base field's Montgomery form, a vector of `k` words; for longer primes, its `k` coefficients as elements of `GF(p)`. `coeffs()` gives the coefficients as elements of `GF(p)` in every case: for the first two, made on its first call and kept with the element. Sums, differences, negations, products and squares stay in the compact forms, and so do, over a prime of one word below `2⁶³`, the inverse up to degree 64 and the Frobenius map (on the words); the norm and the trace take the coefficients, and so does the inverse beyond degree 64.
- **Small fields by tables.** A field of at most `2¹⁶` elements over a prime of one word (`k ≥ 2`; at most `2¹⁴` for `k` = 2 and 3) holds its elements as logarithms to the base of a primitive element `h`, with three tables of 16-bit entries: the powers `hⁱ`, their logarithms, and Zech's logarithms `log(1 + hⁱ)`, `6q` bytes, at most 384 KiB (FLINT's `fq_zech` holds its elements the same way). The tables are made on their first use, in a cell of the field shared by its copies (and by the threads that share them), so that making the field costs what it costs for any other: making an element from its coefficients (`from_u64`, `element`, `x()`, …), a sum, `coeffs()`, the norm or a square root makes them, while `zero()`, `one()` and the products, inverses, powers, Frobenius maps, characters and orders of logarithms need none. An element is the logarithm, and a sum is `hˣ + hʸ = h^(x + log(1 + h^(y − x)))`, a difference the sum with `−1 = h^((q − 1)/2)`; for conversions an element is encoded as the integer `Σ wᵢ·pⁱ` of the words its coefficients hold in the base field's form, so that nothing is converted: a product is a sum of two logarithms, a square a doubling, an inverse a difference, a power a product by the exponent modulo `pᵏ − 1`, the Frobenius map a product by `pⁱ`, the norm a power, the quadratic character the parity of the logarithm, a square root its half (the smaller root then chosen as below), the order `(pᵏ − 1)/gcd(log, pᵏ − 1)`. `h` is the first primitive element in the order of the integers `Σ cᵢ·pⁱ`. The tables come from a walk through the group on the words (the product being linear in them): for `p = 2` by products by `h`, a few shifts and exclusive ors of one word; for odd `p` by products by `x`, a shift of the digits packed in one word and a sum modulo `p` for each term of `xᵏ mod g`, the encoding updated along, through the cosets of the powers of `x`: with `m` the order of `x` and `r = (pᵏ − 1)/m`, `h^r` has the order `m`, so it is a power `x^a` read in the walk of the powers of `x`, `x = h^s` for `s = r·(a⁻¹ mod m)`, and `hʲ·xᵗ` has the logarithm `j + s·t`; each walk goes in four runs at once, whose chains of dependent operations overlap. That is a few operations of words and six accesses to the tables an element: measured on an Apple M4 Max, 0.1 ms for `GF(2¹⁶)`, 0.24 ms for `GF(3¹⁰)` and 0.21 ms for `GF(251²)` (1.5 to 4 ns an element), on the first use, after which a product and a sum take 7 to 11 ns against the words' 37 to 141 ns. The tables so repay their making within 300 such operations in `GF(2¹⁶)` and 430 to 1,950 in the fields of 3⁸ to 3¹⁰, 5⁶, 7⁵, 13⁴, 23³ and 127² elements; the fields of degree 2 and 3 of more than `2¹⁴` elements, whose products gain least, would need 1,950 to 7,000 (29³ to 37³, 131² to 251²) and keep the words (`ext_zech::max_order`).
- **Products** are the schoolbook products of the coefficients followed by the reduction from the top down, each `xⁱ` for `i ≥ k` replaced by `x^(i−k)` times the terms of `xᵏ mod g`, kept in the field: `k² + (k − 1)·w` products in `GF(p)` for the `w` nonzero terms of `g` below `xᵏ`: one for a binomial, two for the `xᵏ + x + c` that the first choice takes for a large prime without an irreducible binomial. Over a prime of one word below `2⁶³` the products of an output coefficient are summed in 128 bits, up to `⌊(2⁶⁴ − 1)/p⌋` of them (below `p·2⁶⁴`), and reduced once by Montgomery's REDC (a sum of products of residues `xR·yR` is `(Σxy)R²`, whose REDC is the residue of `Σxy`); such a field takes this schoolbook on the words up to degree 256 (`WORD_MAX`; 64 for a `g` of more than four terms below `xᵏ`), where it took 0.4 to 0.9 times the time of `ModPolynomial` with its conversions (measured on an Apple M4 Max for primes of 2 to 63 bits), past the degree below. From a degree that depends on the size of `p` the product goes through [`ModPolynomial`](poly.md) instead: its products (sums of products reduced once, number-theoretic transforms or Kronecker substitution for long polynomials) and its remainder (schoolbook or Newton's division), which pay for converting the coefficients there and back. The degree is where the two ways took the same time on chains of products of random elements, measured on an Apple M4 Max with the prime fields in their forms for `k` up to 512: 320 for `p = 2` (the schoolbook product skips its many zero coefficients), 160 for the other primes of one word, 130 for two and three limbs (where `ModPolynomial`'s products change method), 100 for four, 36 for five, 48 for six, 32 for seven and eight, 28 for nine and 8 beyond (the Wide form); for a `g` of more than four terms below `xᵏ`, whose reduction costs about as much as the product, 224, 64, 80 for two limbs and 52 for three, 40, 24, 24, 15, 14 and 6. Where the prime's form moves the crossover the bound is the lower one: the schoolbook would stay faster up to 130 for BN254's and P-256's primes, and up to 52 to 55 for P-384's and BLS12-381's, against 35 for a generic prime of six full limbs.
- **Inverses** are the extended Euclidean algorithm on the element and `g`: since `g` is irreducible, `s·a + t·g = 1` and `s`, of a degree below `k`, is the inverse. Over a prime of one word below `2⁶³` up to degree 64 (`WORD_MAX`) it runs on the coefficients' words, Montgomery residues: each division step by the inverse of the leading coefficient (the binary extended Euclid on its value, one word), the cofactor `s` carried along, `O(k²)` products of words, and `s/r` for the constant `r` it ends on. Otherwise it is `ModPolynomial::xgcd`, `O(k²)` operations in `GF(p)` and `O(M(k)·log k)` by the half gcd for long polynomials, `M(k)` the cost of a product of two of length `k`. A constant is inverted in `GF(p)`; in a field of Zech's tables the inverse of `hˡ` is `h^(q − 1 − l)`.
- **Powers** reduce a nonnegative exponent modulo `pᵏ − 1` for an element that is not 0 and take squares and products from its top bit down.
- **The Frobenius map**: for `a = Σ aⱼxʲ`, `aᵖ = Σ aⱼ·x^(p·j)`, since `aⱼᵖ = aⱼ` in `GF(p)` and the map is additive. Its matrix, column `j` the coefficients of `x^(p·j) mod g`, is computed on the first call: `xᵖ mod g` by `ModPolynomial::pow_mod`, `O(log p)` squares modulo `g`, then `k − 2` more products modulo `g`. `a^(pⁱ)` is then `i` products by the matrix, each sum of products reduced once modulo `p`; over a prime of one word below `2⁶³` the matrix is also kept as the base field's words, and each coefficient of a product is a sum of `k` products of words in 128 bits reduced once by REDC, as in the products of elements.
- **The norm** is the resultant `res(g, a)` (`ModPolynomial::resultant`, from the remainders of Euclid's algorithm): for the monic `g` it is the product of `a` at the roots of `g`, which are `x, xᵖ, …, x^(pᵏ⁻¹)`, so it is the product of the conjugates of `a` (Lidl and Niederreiter, §2.3). `N(c) = cᵏ` for a constant. For `k = 2` it is the closed form `a₀² − g₁a₀a₁ + g₀a₁²` of `g = x² + g₁x + g₀` (the product `(a₀ + a₁r)(a₀ + a₁s)` at the roots `r`, `s`), three products in `GF(p)`.
- **The trace** is `Σ aᵢ·Tr(xⁱ)`, `Tr(xⁱ)` the power sum `sᵢ` of the roots of `g = xᵏ + e₁xᵏ⁻¹ + … + eₖ`, from Newton's identities, `s₀ = k` and `sᵢ = −(i·eᵢ + Σⱼ₌₁^(i−1) eⱼ·sᵢ₋ⱼ)`, computed once in `k²/2` products and kept in the base field's form, where the sum of the `k` products is taken.
- **Square roots.** For odd `p` the quadratic character is that of the norm in `GF(p)`, since `a^((pᵏ − 1)/2) = N(a)^((p − 1)/2)`: a non-square is answered with one resultant and one Legendre symbol. A constant that is a square in `GF(p)` has the root of `GF(p)`; a non-square constant is a square exactly for even `k`. Otherwise the root is Tonelli and Shanks's algorithm in `GF(pᵏ)` (Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 1.5.1, with `GF(pᵏ)` for `Z/pZ`): `pᵏ − 1 = 2ˢ·t` with `t` odd, `r = a^((t+1)/2)` and `w = a^t` from the one power `a^((t−1)/2)`, then while `w ≠ 1` the least `i` with `w^(2ⁱ) = 1` and the corrections by powers of `c = z^t`, for the first non-square `z` (a constant `2, 3, …` for odd `k`; `x, x + 1, …` for even `k`, where every constant is a square), found once. For `p = 2` squaring is a bijection and the root is the inverse of the Frobenius map: `√a = Σ a₂ᵢ·xⁱ + √x·Σ a₂ᵢ₊₁·xⁱ` with `√x = x^(2ᵏ⁻¹) mod g` computed once by `k − 1` squares modulo `g`.
- **Orders** come from the factorisation of `pᵏ − 1 = Π_{d | k} Φ_d(p)`, each `Φ_d(p) = Π_{e | d} (pᵉ − 1)^μ(d/e)` factored by `Number::factor`: the order of `a` starts at `pᵏ − 1` and each prime power `rᵉ` of it is taken out and put back one `r` at a time while the power is not 1: one power of `a` and at most `e` powers by `r` for each prime. `is_primitive` tests `a^((pᵏ − 1)/r) ≠ 1` for each prime `r`; `primitive_element` takes the first element that passes, in the order of the coefficients from the top down past the constants, whose orders divide `p − 1`. If the candidates behaved like random elements, about `(pᵏ − 1)/φ(pᵏ − 1)` of them, `O(log log pᵏ)`, would be tried on average; that is a heuristic, not a bound on the first one in a fixed order: in `GF(37³)` the 39th candidate is the first primitive one, against an average of 3.55.

### Example

```rust
use a::math::field::{ExtensionField, PrimeField};
use a::math::numbers::Number;
use a::math::poly::ModPolynomial;

// GF(3⁵) by the first irreducible polynomial of degree 5, x⁵ + 2x + 1.
let base = PrimeField::new(&Number::from(3u8));
let f = ExtensionField::with_degree(&base, 5);
assert_eq!(f.modulus(), &ModPolynomial::from_u64(&[1, 2, 0, 0, 0, 1], 3));
assert_eq!(f.order(), &Number::from(243u8));

let a = f.from_u64(&[1, 0, 2, 1]); // x³ + 2x² + 1
let x = f.x();
assert_eq!(&(&a / &x) * &x, a);
assert_eq!(a.pow(f.order()), a); // a^(3⁵) = a
assert_eq!(a.frobenius(1), a.pow_u64(3));
assert_eq!(a.frobenius(5), a);

// The norm and the trace are the product and the sum of the conjugates a, a³, a⁹, a²⁷, a⁸¹;
// PARI gives 1 and 2.
let conj: Vec<_> = (0..5).map(|i| a.frobenius(i)).collect();
assert_eq!(conj.iter().fold(f.one(), |s, c| s * c).to_base().unwrap(), a.norm());
assert_eq!(conj.iter().fold(f.zero(), |s, c| s + c).to_base().unwrap(), a.trace());
assert_eq!((a.norm().value(), a.trace().value()), (Number::from(1u8), Number::from(2u8)));

// Square roots: 2 is not a square modulo 3, nor in an extension of odd degree.
let r = a.square().sqrt().unwrap();
assert!(r == a || r == -&a);
assert!(f.from_u64(&[2]).sqrt().is_none());

// Orders, from 3⁵ − 1 = 2·11²: x generates the group, and a has the order 121 (PARI's fforder).
assert!(x.is_primitive());
assert_eq!(a.multiplicative_order(), Some(Number::from(121u8)));

// A field by a given polynomial: BLS12-381's F_p² = F_p[u]/(u² + 1), which is also the first
// choice for its p.
let fp = PrimeField::bls12_381_base();
let one = Number::from(1u8);
let fp2 = ExtensionField::new(fp, &ModPolynomial::new(&[one.clone(), Number::default(), one], fp.modulus()));
assert_eq!(fp2, ExtensionField::with_degree(fp, 2));
let u = fp2.x();
assert_eq!(u.square(), -fp2.one());
assert_eq!(u.frobenius(1), -&u); // the conjugate
assert!(fp2.from_u64(&[3, 5]).norm() == fp.from_u64(34)); // 3² + 5²
```

### Verified

- **PARI/GP and FLINT.** 34 fields (`lib/tests/math/field/data/ext_reference.txt`, written by `ext_gen.py` with `ext_gen_flint.c`): `p` = 2, 3, 5, 7, 101, 65537, `2³¹ − 1`, `2⁶¹ − 1`, `2⁶⁴ + 13`, BN254's and BLS12-381's base primes and a prime of 600 bits; `k` = 1 (by `x`, `x + 1` and `x + 5`), 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 16, 24, 48 and 64, so that every size of prime has fields on both sides of the degree where the products change ways; the polynomials of the first choice, AES's, and dense ones from PARI's `ffinit` (for `GF(2⁸)`, `GF(3⁵)`, `GF(101⁴)` and `GF((2³¹ − 1)²⁴)`). On 148 pairs of seeded elements, which the tests make again from their seeds, PARI 2.19 (`ffgen` elements: `a·b`, `1/a`, `aᵉ`, `a^(pⁱ)`, `norm`, `trace`, `issquare`, `sqrt`, `fforder`) and FLINT 3.6 (`fq_mul`, `fq_inv`, `fq_pow`, `fq_frobenius`, `fq_norm`, `fq_trace`, `fq_is_square`, `fq_sqrt`, `fq_multiplicative_order`) agree on every value, the square roots up to the sign, and the library gives the same: the products, inverses, powers (exponents of `k·log₂ p + 20` bits, beyond `pᵏ`, so that the reduction modulo `pᵏ − 1` is taken; 400 bits for `k = 64` over BLS12-381), Frobenius powers for `i` below `2k`, norms, traces, the character of 148 elements (82 squares, 66 not), the 82 square roots (the root of the rule above) and 78 multiplicative orders (in the 17 fields whose `pᵏ − 1` has at most 257 bits). The fields of degree 1 have 12 samples each, which the generator checks cover both elements of `GF(2)` and, for odd `p`, nonzero squares and non-squares. The outputs of the 10 fields of more than 1,000 bits are stored as the digest `Σ cᵢ·Tⁱ mod p` of their coefficients for a fixed `T`. 143 samples are in the regular tests, the 5 of the fields of degree 12 and 64 over BN254 and BLS12-381 in an ignored one.
- **The first choice.** `first_irreducible` gives PARI's polynomial for 140 pairs `(p, k)`, PARI trying `polisirreducible` on every monic polynomial of degree `k` in the order of `Σ cᵢ·pⁱ` from `xᵏ` up: every `k` from 1 to 16 for `p` = 2, 3, 5, 7, 11 and 13 (96 pairs), the degrees 24, 32 and 64 for 2 and 3, and 38 more for primes from 5 to the one of 600 bits and degrees up to 64, among them 2, 3, 4, 6, 12, 24 and 64 for BN254's and BLS12-381's primes. For 122 of them every smaller polynomial was tried; for the 18 with `p` above 20,000 and no irreducible binomial (by the theorem), PARI tried the binomials up to `c = 2000` and every candidate from `xᵏ + x` up. The tests also check 8 cases by trial division by every monic polynomial of degree up to `k/2`, each smaller candidate being reducible and the one chosen not.
- **Identities**, on 21 fields (`p` = 2, 3, 5, `10⁹ + 7`, BN254, BLS12-381's base and scalar primes, a prime of 600 bits; `k` from 1 to 40; two dense `g`; both ways of the product on each side of the degree where they change: 39 and 40 for 2 and 3, 19 and 20 modulo `10⁹ + 7`, 5 and 10 with a dense `g`, 3 and 4 for the prime of 600 bits): the field axioms on every combination of owned and borrowed operands; products equal to `ModPolynomial::mul_mod`; the Frobenius map a ring homomorphism of order `k` fixing `GF(p)`, `φⁱ⁺¹ = φ ∘ φⁱ`, `φ(a) = aᵖ`; the norm and the trace the product and the sum of the `k` conjugates, `N(ab) = N(a)N(b)`, `Tr(a + b) = Tr(a) + Tr(b)`, `Tr(ca) = c·Tr(a)`, `N(c) = cᵏ`, `Tr(c) = k·c`, `N(aᵖ) = N(a)`; `a^(pᵏ) = a`, `a^(pᵏ − 1) = 1` and negative exponents; the root of `a²` is `±a`, with the top coefficient below `p/2`; the character multiplicative. Four more fields, on each side of 48 for two limbs and of 28 for nine (`2⁶⁴ + 13` with `k` = 47 and 48, `2⁵²¹ − 1` with 27 and 28), have the axioms, the products of `ModPolynomial::mul_mod`, inverses, a Frobenius step and the multiplicativity of the norm and the trace.
- **Exhaustively**, in 11 fields of up to 256 elements (`GF(2)`, `GF(4)`, `GF(8)`, `GF(16)`, `GF(256)`, `GF(3)`, `GF(9)`, `GF(27)`, `GF(25)`, `GF(125)`, `GF(49)`): `q` distinct elements, `(q + 1)/2` squares for odd `q` and `q` for even, `is_square` exactly on them, and `sqrt` a root exactly for them (asserted for every element), the documented one, `a^q = a` and `φᵏ = id` for each element, each nonzero value of the norm taken `(q − 1)/(p − 1)` times and each value of the trace `q/p` times, the order of every element against the powers, `φ(q − 1)` primitive elements, and the batch inversion of all of them.
- **Edges**: degree 1 against `PrimeField` (products, inverses, norm, trace, Frobenius, character, and the roots of `Fp::sqrt`); square roots of constants in degrees 1 to 6 (a non-square of `GF(p)` has a root exactly in even degrees); FIPS 197's `{57}·{83} = {c1}` in AES's field; the factorisation of `pᵏ − 1` of `GF(2⁶⁴)`, and a field of `p = 2¹²⁷ − 1` given PARI's factorisation of `p² − 1`, where `x + 1` has PARI's order 1016; the first primitive element of `GF(37³)`, `2x + 1`, the 39th candidate as PARI finds it; 0 and 1 (norm, trace, root, inverse, order); text; two equal fields built apart; and 12 panics (a polynomial that is reducible, not monic, constant or modulo another prime; degree 0; elements of fields of other degrees and of other polynomials; a coefficient and a polynomial of another prime; division by zero; a negative power of 0; a wrong factorisation).
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 28 tests pass, and the 2 ignored ones in the release profile.

To run the tests: `cargo test --test math -- field::ext`, and `cargo test --release --test math -- --ignored field::ext` for the large fields and the searches of degree 12 and 64 over BN254 and BLS12-381 (minutes in the debug profile). The data file is written again by `python3 lib/tests/math/field/data/ext_gen.py`, with PARI/GP and FLINT installed; it stops if they disagree.

### Limits

- Not constant-time: the arithmetic branches on zero coefficients, and a power on the bits of its exponent.
- The elements of a word prime's field are vectors of `k` words, allocated by each operation (FLINT's `fq_nmod` allocates too); those of a longer prime's, vectors of `k` elements of `GF(p)` of 88 bytes. In a word prime's field the norm beyond degree 2 is the resultant of `ModPolynomial` on the coefficients, and the inverse beyond degree 64 its `xgcd`; the trace takes the coefficients as elements of `GF(p)`. The tables of the field (but the terms of `xᵏ mod g`, the traces and a word prime's Frobenius matrix, kept in the base field's form) are kept as numbers and converted on each use. The degrees where the products change ways are a table by the size of `p`, measured on random elements for `k` up to 64; on the primes measured, the way taken is within a factor of 2.3 of the faster one (the four-limb primes of special shapes, such as `2²⁵⁵ − 19` and P-256, keep the schoolbook product the faster up to 64 and beyond).
- A field of Zech's tables pays for them on the first use of an element, `O(q)` (see the methods), which the products and sums it saves repay after some hundreds to some thousands of operations; a program that makes many small fields and uses each little pays it each time. The fields of degree 2 and 3 take tables up to `2¹⁴` elements and the others up to `2¹⁶`, so that `GF(251²)` and `GF(29³)` keep the words.
- `frobenius(i)` takes `i` products by the matrix, `O(i·k²)`, and the matrix holds `k²` elements of `GF(p)`, a million for `k = 1000`; there is no modular composition.
- Square roots for odd `p` cost a power of about `k·log₂ p` bits, some 24,000 squares in `GF(pᵏ)` for `k = 64` over BLS12-381's prime; the methods for even degrees that work through the subfields (Adj and Rodríguez-Henríquez, 2014) are not in.
- `with_degree` searches: past the binomials, each candidate costs a test of its discriminant and, for about half, Ben-Or's test, and about `k` of them are tried for a large `p`; `new` with a known polynomial costs one test.
- The orders and the primitive elements need the factorisation of `pᵏ − 1`, which for large fields can take hours or more; `with_factored_unit_order` takes a known one.
- Degrees below `2³²`. No Conway polynomials, embeddings between fields of different degrees, minimal or characteristic polynomials, or normal bases (the binary fields have a normal basis of their own).

## Towers

`Tower`, `Fp2`, `Fp6` and `Fp12` are the tower of extensions `F_p² → F_p⁶ → F_p¹²` that pairings compute in, the representation of Devegili, Ó hÉigeartaigh, Scott and Dahab (*Multiplication and squaring on pairing-friendly fields*, ePrint 2006/471) and of the IETF draft on pairing-friendly curves, which arkworks, the `bls12_381` crate and blst use:

- `F_p² = F_p[u]/(u² − β)` for a non-square `β` of `F_p`;
- `F_p⁶ = F_p²[v]/(v³ − ξ)` for a `ξ` of `F_p²` that is neither a square nor a cube;
- `F_p¹² = F_p⁶[w]/(w² − v)`, so that `w⁶ = ξ` and `F_p¹² = F_p²[w]/(w⁶ − ξ)`.

`Tower::bn254()` has `β = −1`, `ξ = 9 + u` over the prime of `PrimeField::bn254_base()`; `Tower::bls12_381()` has `β = −1`, `ξ = 1 + u` over that of `PrimeField::bls12_381_base()`. `Tower::new(&field, &β, (&ξ₀, &ξ₁))` builds the tower of any prime `p ≡ 1 (mod 6)`, checking that `β` is not a square and `ξ` neither a square nor a cube (otherwise it panics), and computes the constants of its Frobenius maps once. The tower borrows its prime field, so that the coefficients of its elements are elements of that very field (`Fp2::c0()` is an `Fp` that adds to the caller's elements as it is) and its constants are kept in the field's own form; its elements borrow the tower, as an `Fp` borrows its `PrimeField`. Elements of towers with different `p`, `β` or `ξ` do not mix: an operation on them panics.

| Building and reading | |
| --- | --- |
| `Fp2::new(&t, c0, c1)`, `Fp2::from_numbers(&t, &c0, &c1)`, `Fp2::from_base(&t, c0)` | `c₀ + c₁u`, from elements of `F_p` (panics for another field) or from numbers reduced modulo `p` |
| `Fp6::new(c0, c1, c2)`, `Fp6::from_fp2(c0)`; `Fp12::new(c0, c1)`, `Fp12::from_fp6(c0)` | `c₀ + c₁v + c₂v²`, `c₀ + c₁w` |
| `Fp12::from_coefficients(&t, &[Number; 12])`, `x.coefficients()` | the twelve coefficients of `F_p` in the order `c0.c0.c0, c0.c0.c1, c0.c1.c0, …, c1.c2.c1`, arkworks' and the `bls12_381` crate's; over `F_p²`, `c0.c0, c1.c0, c0.c1, c1.c1, c0.c2, c1.c2` are the coefficients of `1, w, w², w³, w⁴, w⁵` |
| `zero(&t)`, `one(&t)`, `c0()`, `c1()`, `c2()`, `tower()`, `is_zero()`, `is_one()` | |
| `t.base()`, `t.beta()`, `t.xi()`, `t.frobenius_coefficient(k, i)` | the field, `β`, `ξ`, and `ξ^(i(pᵏ − 1)/6)`, the factor of `wⁱ` in its `pᵏ`-th power |

| Arithmetic (on all three unless said) | Method | Cost |
| --- | --- | --- |
| `+`, `-`, `-x`, and `+=`, `-=` | coefficient by coefficient | additions |
| `*`, `*=` | Karatsuba at each floor | `F_p²`: 3 products of `F_p`; `F_p⁶`: 6 of `F_p²`; `F_p¹²`: 3 of `F_p⁶` (54 of `F_p`) |
| `square()` | `F_p²`: complex squaring `(c₀ + c₁)(c₀ + βc₁) − (1 + β)c₀c₁`; `F_p⁶`: Chung and Hasan's SQR2 (*Asymmetric squaring formulae*, ARITH 2007); `F_p¹²`: complex squaring | 2 products of `F_p`; 2 products and 3 squares of `F_p²`; 2 products of `F_p⁶` |
| `inverse()`, `/`, `/=` | through the norm: `F_p²`: `x̄ / (c₀² − βc₁²)`; `F_p⁶`: the adjugate `(c₀² − ξc₁c₂, ξc₂² − c₀c₁, c₁² − c₀c₂)` over `c₀A + ξ(c₂B + c₁C)`; `F_p¹²`: `x̄ / (c₀² − v c₁²)` | one inversion of `F_p` for any floor; `None` for 0, and `/` by 0 panics |
| `frobenius(k)` | `x^(pᵏ)` for any `k`: the coefficients of `wⁱ` conjugated for odd `k` and multiplied by the precomputed `ξ^(i(pᵏ − 1)/6)` | 5 products of `F_p²` by constants on `F_p¹²` (by elements of `F_p` where the constant lies there, as for even `k`), 2 on `F_p⁶`; no power |
| `conjugate()` (`F_p²`, `F_p¹²`) | `c₀ − c₁u`, `c₀ − c₁w`: `x^p` on `F_p²`, `x^(p⁶)` on `F_p¹²` | negations |
| `square_in_place()`, `neg_in_place()`, `double_in_place()`; `conjugate_in_place()` (`F_p²`, `F_p¹²`); `cyclotomic_square_in_place()` (`F_p¹²`) | the same in the element itself: only the coefficients' limbs are rewritten, as by `+=`, `-=` and `*=` | |
| `norm()`, `trace()` | to the floor below: `x x̄` and `x + x̄` on `F_p²` and `F_p¹²`; `c₀³ + ξc₁³ + ξ²c₂³ − 3ξc₀c₁c₂` and `3c₀` on `F_p⁶`; to `F_p` from `F_p¹²` as `x.norm().norm().norm()` | a few products |
| `pow(&e)`, and `pow_u64(e)` on `F_p²` | square and multiply; a negative `e` raises the inverse, and panics for 0 | a square per bit of `e` and a product per one |
| `mul_by_base(&k)`, `mul_by_xi()` (`F_p²`); `mul_by_fp2`, `mul_by_v`, `mul_by_01`, `mul_by_1` (`F_p⁶`) | products by `F_p`, `ξ`, `F_p²`, `v`, `b₀ + b₁v`, `b₁v` | `mul_by_01`: 5 products of `F_p²`; `mul_by_1`: 3 |
| `mul_by_034(c0, c3, c4)`, `mul_by_014(c0, c1, c4)` (`F_p¹²`) | the products by the values of a line of a pairing on a D-type and an M-type twist, `c₀ + c₃w + c₄vw` and `c₀ + c₁v + c₄vw` | 13 products of `F_p²` instead of 18 |
| `is_cyclotomic()`, `cyclotomic_square()`, `cyclotomic_pow(&e)` (`F_p¹²`) | the subgroup of order `Φ₁₂(p) = p⁴ − p² + 1`, tested by `x^(p⁴)·x = x^(p²)`; Granger and Scott's squaring (*Faster squaring in the cyclotomic subgroup of sixth degree extensions*, PKC 2010) over `F_p⁴ = F_p²[s]/(s² − ξ)`, `s = w³`, where `(A + Bw + Cw²)² = (3A² − 2Ā) + (3sC² + 2B̄)w + (3B² − 2C̄)w²`; powers over the non-adjacent form of `e` (an inverse there is a conjugate), or the binary form when that has no more nonzero digits | 9 squares of `F_p²` per square instead of 12 products; outside the subgroup the results are not squares and powers |
| `sqrt()`, `is_square()`, `sgn0()` (`F_p²`) | the complex method (Scott, *Implementing cryptographic pairings*, Pairing 2007; Adj and Rodríguez-Henríquez, IEEE Trans. Computers 63, 2014) for any odd `p` and any non-square `β`: with `γ = √(c₀² − βc₁²)`, one of `δ = (c₀ ± γ)/2` is a square, as their product `βc₁²/4` is not, and `√δ + c₁/(2√δ)·u` is a root; `c₁ = 0` takes `√c₀` or `√(c₀/β)·u`. The root returned is the smaller in ZCash's and arkworks' order (by `c₁`, then by `c₀`), which on `F_p` is the root of `Fp::sqrt`. `is_square` is the Legendre symbol of the norm; `sgn0` is RFC 9380's (§4.1) | a Legendre symbol, two square roots of `F_p` and an inversion; `None` for a non-square |
| `is_square()` (`F_p⁶`, `F_p¹²`) | the norm to the floor below is a square there, as `N(x)^((q − 1)/2) = x^((qᵏ − 1)/2)` | |

`PartialEq`, `Eq`, `Hash` (by the coefficients), `Debug` and `Display` (`c₀ + c₁·u`, `(…) + (…)·v + (…)·v²`, `(…) + (…)·w`, in decimal) are implemented; the operators take owned and borrowed operands.

### Example

```rust
use a::math::field::{Fp2, Fp6, Fp12, PrimeField, Tower};
use a::math::numbers::Number;

let t = Tower::bls12_381();
let n = |x: u32| Number::from(x);
let a = Fp2::from_numbers(t, &n(3), &n(4));
assert_eq!(a.norm(), t.base().from_u64(25)); // 3² + 4², as u² = −1
assert_eq!(a.square().sqrt().unwrap().square(), a.square());
assert!(!a.square().mul_by_xi().is_square()); // ξ is not a square

// An element of F_p¹², its conjugates and its norm to F_p.
let x = Fp12::new(Fp6::new(a.clone(), Fp2::one(t), t.xi()), Fp6::from_fp2(a.clone()));
assert_eq!(x.frobenius(6), x.conjugate());
assert_eq!(x.frobenius(12), x);
let mut product = Fp12::one(t);
for k in 0..12 {
    product = &product * &x.frobenius(k);
}
let norm = x.norm().norm().norm();
assert_eq!(product, Fp12::from_fp6(Fp6::from_fp2(Fp2::from_base(t, norm))));

// The cyclotomic subgroup, where pairings take their values.
let c = &x.conjugate() / &x;
let c = &c.frobenius(2) * &c;
assert!(c.is_cyclotomic());
assert_eq!(c.cyclotomic_square(), c.square());
assert_eq!(c.cyclotomic_pow(&n(1000)), c.pow(&n(1000)));

// A tower of one's own: p = 103 ≡ 1 (mod 6), β = −1, ξ = 2 + u.
let f = PrimeField::new(&n(103));
let small = Tower::new(&f, &Number::from(-1), (&n(2), &n(1)));
let y = Fp2::from_numbers(&small, &n(5), &n(7));
assert_eq!(y.pow(&n(103)), y.conjugate());
```

### Verified

- **arkworks 0.5** (`ark-bls12-381`, `ark-bn254`, whose towers are these), by `lib/tests/math/pairing/data/gen_ark.rs`: on 16 seeded elements of each tower, 28 operations — the product, square, inverse and quotient of `F_p¹²`, its Frobenius maps for `k = 1, 2, 3, 6, 7`, a power by a 64-bit exponent, the two sparse products, the cyclotomic square and power of an element of the cyclotomic subgroup, the norm to `F_p⁶`; the product, square, inverse, Frobenius map, norm, `mul_by_01`, `mul_by_1` and product by `v` of `F_p⁶`; the product, square, inverse, norm, product by `ξ` and squareness of `F_p²` — give arkworks' outputs exactly: the 56 digests (FNV-1a of the canonical bytes of every output, in `lib/tests/math/field/data/tower_ref.txt`) agree.
- **PARI/GP 2.19**, by `gen_py.py`, in the flat field `F_p[w]/(w¹² − 2ξ₀w⁶ + ξ₀² + 1)` (as `u = w⁶ − ξ₀`), sharing no code with the tower: on 8 seeded elements of each floor of each tower, the norms and traces to the floor below (as `x·x^(p⁶)`, `x·x^(p²)·x^(p⁴)`, `x·xᵖ` and the sums), the norm and the trace to `F_p` (`norm`, `trace`), and the squareness in `F_p²` (Euler's criterion) and in `F_p¹²` (`issquare`): 16 digests, all equal.
- **Identities**, on seeded elements of both towers: `frobenius(1)` is the power by `p` on all three floors, `frobenius(2)` that by `p²` on `F_p⁶`, `frobenius(k)` is `frobenius(1)` composed `k` times for `k = 0, …, 12`, and `(wⁱ)^(pᵏ) = frobenius_coefficient(k, i)·wⁱ` for all 72 pairs; norms and traces are their definitions by conjugates, multiplicative and additive; the sparse products equal the dense ones; inverses, quotients, squares and negative powers agree with products; the cyclotomic square and power (by positive, negative and zero exponents) agree with the plain ones on elements of the subgroup.
- **Towers with other non-residues**, so that every branch for `β ≠ −1` and `ξ₁ ≠ 1` runs: 84 towers over the 21 primes `p ≡ 1 (mod 6)` below 200, with `β` each of the two least non-squares of `F_p` (never −1; when `p ≡ 1 (mod 4)`, −1 is a square and such a `β` is the only kind there is) and `ξ` the first valid one with `ξ₁ = 1` and the first with `ξ₁ ≥ 2`, chosen by an arithmetic of words of the test's own. On each, on seeded elements: the products and squares of `F_p²` against `(a₀b₀ + βa₁b₁) + (a₀b₁ + a₁b₀)u`, `mul_by_xi` against the product by `ξ`, the norm and the trace against their formulas; the products and squares of `F_p⁶` and `F_p¹²` and the two sparse products against schoolbook products over `F_p²` modulo `v³ − ξ` and `w⁶ − ξ`; `x^(p¹² − 1) = 1`, the Frobenius maps against the powers by `p` and against each other for every `k < 12`, the absolute norm and trace against the product and the sum of the twelve conjugates, the relative norms and traces against their definitions, `is_square` against Euler's criterion, the cyclotomic square and power against the plain ones; and `sqrt` on every element of `F_p²` for the six primes below 50 (against the squares by brute force) and on 200 seeded elements for the others (against Euler's criterion), the root squaring back and the smaller, and the root `√(c₀/β)·u` of a non-square `c₀` of `F_p`. These tests fail with each of four bugs planted in those branches (the square without its `−βc₀c₁`, `mul_by_xi` without its `β`, `√c₀` for `√(c₀/β)`, and a product by `β` made a sum), which the tests on BN254's, BLS12-381's and `F_103`'s towers, all `β = −1` and `ξ₁ = 1`, did not see.
- **A small tower**, `p = 103`, `ξ = 2 + u`: on 20 random elements of `F_103¹²`, `x^(p¹² − 1) = 1`, `frobenius(k)` is the power by `pᵏ` for every `k < 12`, the norm is the product of the twelve conjugates, and `is_square` agrees with Euler's criterion; `sqrt` on all 10 609 elements of `F_103²`: 5 305 squares (0 and the `(p² − 1)/2` others), each root squaring back and the smaller of the two, `None` exactly for the others.
- **Square roots** on the curves' towers: of squares (`±x` returned), of `x²ξ` (none), of elements of `F_p` and of `F_p·u` (the cases `c₁ = 0` and `c₀ = 0`), of 0, 1 and −1.
- **Panics**: `p ≢ 1 (mod 6)`, a square `β`, a `ξ` that is a square, a `ξ` that is a cube but not a square, elements of different towers, a coefficient of another field, division by zero, a negative power of 0, a wrong number of coefficients.

To run: `cargo test --test math -- field::tower`. To regenerate the references: the probe crate of `gen_ark.rs` (`cargo run --release`), and `python3 gen_py.py <the bls12_381 crate's src/tests>` with py_ecc and `gp` installed.

### Limits

- `Tower::new` needs `p ≡ 1 (mod 6)`, as the Frobenius coefficients `ξ^(i(p − 1)/6)` do; every BN and BLS12 prime is so. Other sextic towers are not offered.
- `cyclotomic_square` and `cyclotomic_pow` give the square and the power only of elements of the cyclotomic subgroup; elsewhere they give some other element, without a warning (`is_cyclotomic` tells).
- `sqrt` is offered on `F_p²` alone; `F_p⁶` and `F_p¹²` have `is_square`.
- Not constant-time. Each element of `F_p¹²` is twelve `Fp`s; over the Montgomery forms of four and six limbs (BN254's and BLS12-381's base fields) an operation matches the form once and computes on the limbs with the field's kernels, products by `ξ = ξ₀ + u` with a small `ξ₀` and by `β = −1` as sums; over other primes it is the same formulas on `Fp`s. The assignments `+=`, `-=`, `*=` and `square_in_place`, `conjugate_in_place` (`F_p²`, `F_p¹²`) and `cyclotomic_square_in_place` rewrite only the coefficients' limbs.
- The product of `F_p²` with `β = −1` over those forms, for a prime with two spare bits (`4p < 2^(64n)`, BN254's and BLS12-381's), is Karatsuba's on double-width values with a lazy reduction (on aarch64, by `kernels.py`'s kernels): the three products of `2n` limbs unreduced, `a₀b₀ − a₁b₁` modulo `p·2^(64n)` and `(a₀ + a₁)(b₀ + b₁) − a₀b₀ − a₁b₁` exactly, each reduced once — two reductions in place of three. Elsewhere it is three Montgomery products. The products of `F_p⁶` and `F_p¹²` reduce each product of `F_p²` they take; their sums of products are not kept unreduced.

## Binary fields

`BinaryField` is the field `GF(2ᵐ) = GF(2)[x]/(f)` for an irreducible polynomial `f` of degree `m`, given by the exponents of its terms (`[163, 7, 6, 3, 0]` for `x¹⁶³ + x⁷ + x⁶ + x³ + 1`, in any order). Its elements, `F2m`, are polynomials of degree below `m` held in `⌈m/64⌉` words, the coefficient of `xⁱ` bit `i mod 64` of word `i / 64`: read as a little-endian number, the words are the integer `Σ aᵢ·2ⁱ` that SEC 1 (§2.3.5) and OpenSSL's `BN_GF2m_*` use for a field element. The field is built once, checks that `f` is irreducible, and chooses there how it multiplies and reduces; its elements borrow it. `NormalBasis` is `GF(2ᵐ)` in a Gaussian normal basis (elements `NormalElement`), and `BasisConversion` converts between such a basis and a `BinaryField` of the same degree. None of it is constant-time.

| Call | Gives |
| --- | --- |
| `BinaryField::new(&[m, …, 0])` | the field, its products by the CPU's carry-less multiplier; panics if `f` is reducible (Rabin's test), if it has no constant term or an even number of terms, or if an exponent repeats |
| `BinaryField::with_clmul(&exponents, Clmul::Portable)` | the same field with a chosen kernel (`Clmul::Pmull` and `Clmul::Pclmulqdq` panic where the CPU lacks them); fields with the same `f` are equal and their elements mix |
| `BinaryField::is_irreducible(&exponents)` | Rabin's test, without building a field |
| `degree()`, `modulus()`, `clmul()` | `m`, the exponents of `f` in decreasing order, the kernel |
| `element(&words)`, `zero()`, `one()` | an element; `words` of any length, reduced modulo `f` |
| `+`, `-`, `*`, `/`, unary `-`, and `+=` … `/=` | on owned and borrowed elements; `/` by 0 and elements of different fields panic |
| `square()`, `pow_u64(e)`, `inverse()`, `sqrt()` | `a²`, `aᵉ`, `a⁻¹` (`None` for 0), the square root (every element has exactly one) |
| `trace()`, `half_trace()`, `quadratic_root()` | `Tr(a) ∈ {0, 1}`; for odd `m` the half-trace `Σᵢ a^(2^(2i))`; a root of `z² + z = a` for any `m`, `None` when `Tr(a) = 1` |
| `words()`, `is_zero()`, `is_one()`, `field()` | the coefficients and the obvious tests |
| `Clmul::detect()`, `is_available()`, `mul(a, b)` | the CPU's kernel (`Pmull`, `Pclmulqdq` or `Portable`), whether one runs here, and the 128-bit carry-less product of two words by it |

### Methods

- **Carry-less products.** Two words are multiplied by `PMULL` (`vmull_p64`, part of the aarch64 cryptographic extension, which the standard library reports as `aes`) or by `PCLMULQDQ` (`_mm_clmulepi64_si128`) when the CPU has it: found at compile time where the target enables it, as on Apple's aarch64, and at run time otherwise. The portable kernel, for every other CPU and for Miri, is a window method on one word (Hankerson, Menezes and Vanstone, *Guide to Elliptic Curve Cryptography*, 2004, §2.3.3): the 16 products of one word by the polynomials of degree below 4 in a table, the other word taken 4 bits at a time from the top, 16 lookups, shifts and sums. Polynomials of `n` words are multiplied by Karatsuba's method across words, each product split at `⌈n/2⌉` words into three of the halves down to single words, `O(n^1.58)` products of words. The intrinsics sit in small functions compiled with their target feature, each `unsafe` block with its `SAFETY` note, and run only once `Clmul::is_available` has found the instruction; every kernel gives the same products.
- **Squares** spread the bits, a zero after each: by one `PMULL` of each word by itself where the target enables `PMULL` at compile time, as on Apple's aarch64, and otherwise by Algorithm 2.39 of the same book, `2⌈m/64⌉` words of shifts and masks and no products.
- **Reduction modulo `f`**, by one of two methods, whichever a rough count of word operations finds cheaper for `f` and the kernel. The *sparse* reduction takes a chunk of at most `min(64, m − e₁)` bits at a time from the top (`e₁` the second exponent of `f`), each chunk within one word and folded down by one shifted sum per term (as §2.3.5 does for the NIST polynomials, for any `f`): for the trinomials and pentanomials of the standards, about `⌈m/64⌉` chunks of `t` sums for `t` terms. *Barrett's* reduction over `GF(2)[x]`, where it is exact: with `μ = ⌊x²ᵐ/f⌋` computed once, `q = ⌊⌊c/xᵐ⌋·μ/xᵐ⌋` is the quotient of any `c` of degree below `2m`, and `c mod f` is the low `m` bits of `c + q·(f − xᵐ)`, two products. Moduli with many terms or with terms close to `xᵐ`, whose chunks would be short (down to one bit for `e₁ = m − 1`), take Barrett's, of one word as of several; under the portable kernel, whose products cost more, a few short chunks may still win. The counts charge a chunk `3 + t`, a product of words 3 by an instruction and 20 by the portable kernel, and Karatsuba's sums `6n` a level, from timings on one machine at one and two words; `format!("{f:?}")` names the choice (`Sparse` or `Barrett`). Longer inputs to `element` are reduced `min(m, 64)` bits at a time from the top under Barrett's reduction.
- **Inverses** by Bernstein and Yang's division steps for polynomials ("Fast constant-time gcd computation and modular inversion", 2019, §3 and §6) in batches of 63 on the low words: a step on `(δ, f, g)`, `f(0) = 1`, is `(1 − δ, g, (f + g)/x)` for `δ > 0` and `g(0) = 1`, else `(1 + δ, f, (g + g(0)·f)/x)`; from `f` the modulus and `g = a`, `g` is 0 after at most `2m − 1` steps and `f` is then 1. A batch takes its decisions from the low words alone (a run of zeros at the bottom of `g` in one shift), makes the product of its matrices, whose entries fit a word, and applies it to the whole of `f` and `g` and of the cofactors `d`, `e` (`f ≡ d·a`, `g ≡ e·a`) by carry-less products of a word by the polynomials, the division of `d` and `e` by `x⁶³` made exact modulo `f` as in Montgomery's reduction (`f⁻¹ mod x⁶⁴` kept from the field's construction): at most `⌈2m/63⌉ + 1` batches, then `d` is the inverse. A modulus without a constant term (`x`, for `m = 1`), moduli of more than 128 words (8191 bits) under `PMULL` and of more than 40 (2559 bits) under `PCLMULQDQ`, and under the portable kernel every modulus but those of 24 to 31 bits take the extended Euclidean algorithm over `GF(2)[x]` (Algorithm 2.48), which there is the faster (measured on an Apple M4 Max: the steps take 0.23 to 0.61 times its time under `PMULL` from 63 to 8191 bits, but 1.0 to 2.3 times under the portable kernel outside 24 to 31 bits, where one batch does them all): `u ← u + xʲ·v` cancels the leading term, with `a·g₁ ≡ u` and `a·g₂ ≡ v` kept, until `u = 1`. Division is a product by the inverse.
- **Square roots** by `√a = a_even + √x·a_odd` (Fong, Hankerson, López and Menezes, "Field inversion and point halving revisited", *IEEE Trans. Computers* 53, 2004), the even and the odd coefficients gathered by masks and shifts, `√x = x^(2^(m−1))` kept from the field's construction: one product.
- **Traces.** `Tr` is linear, so `Tr(a) = Σ aᵢ·Tr(xⁱ)`: the parity of `a` masked by the bits `Tr(xⁱ)`. Those are the power sums of the roots of `f`, found once by Newton's identities from the coefficients of `f` (Lidl and Niederreiter, *Finite Fields*, Theorem 1.75), `O(m·t)` for `t` terms. The half-trace is `(m − 1)/2` double squarings and sums. `quadratic_root` is the half-trace for odd `m`; for even `m` it is `z = Σ_{i<m−1} θᵢ·c^(2ⁱ)` with `θᵢ = Σ_{i<j<m} δ^(2ʲ)` for the first `δ = xⁱ` of trace 1, for which `z² + z = c + δ·Tr(c)`: `m − 1` products.
- **Irreducibility** by Rabin's test ("Probabilistic algorithms in finite fields", *SIAM J. Comput.* 9, 1980): `f` of degree `m` is irreducible exactly when `x^(2^m) ≡ x (mod f)` and `gcd(x^(2^(m/q)) − x, f) = 1` for each prime `q | m`: `m` squarings modulo `f` and one gcd per prime factor of `m`, `O(m²/64)` word operations each. `new` runs it, after the two checks that settle half of all polynomials at once (`x | f` without a constant term, `x + 1 | f` with an even number of terms), and keeps `x^(2^(m−1)) = √x`.

### Normal bases

A normal basis is `{θ, θ², θ⁴, …, θ^(2^(m−1))}` for a normal element `θ`, and an element is the bit string `(a₀ a₁ … a_{m−1})` of `Σ aᵢ·θ^(2ⁱ)`, coordinate `aᵢ` held as bit `i` of the words (FIPS 186-4 writes `a₀` first, so it is the leading bit of FIPS's hexadecimal numbers). Squaring rotates the coordinates by one place, the trace is their parity and 1 is all ones. `NormalBasis` is a Gaussian normal basis of type `T`, as ANSI X9.62, IEEE 1363-2000 (§A.3) and FIPS 186-4 (§D.3) define it (Ash, Blake and Vanstone, "Low complexity normal bases", *Discrete Applied Mathematics* 25, 1989): one exists when `p = Tm + 1` is a prime and `gcd(Tm/k, m) = 1` for the order `k` of 2 modulo `p`, and `θ` is then the Gauss period `Σ_{j<T} γ^(uʲ)` of a primitive `p`-th root of unity `γ` and a `u` of order `T` modulo `p`. Types 1 and 2 are the optimal normal bases (Mullin, Onyszchuk, Vanstone and Wilson, 1989). **Every `m ≥ 2` that is not a multiple of 8 is supported**; a multiple of 8 has no Gaussian normal basis (then `8 | Tm`, so `p ≡ 1 (mod 8)`, 2 is a square modulo `p`, `k` divides `(p − 1)/2` and `Tm/k` is even) and `NormalBasis::new` panics. Every `m` below 4096 that is not a multiple of 8 has one of type at most 77.

| Call | Gives |
| --- | --- |
| `NormalBasis::new(m)` | the basis of the lowest type, the one the standards choose: types 4, 2, 6, 4 and 10 for `m` = 163, 233, 283, 409 and 571 |
| `NormalBasis::with_type(m, t)`, `NormalBasis::exists(m, t)` | the basis of type `t`, which panics if there is none, and whether there is |
| `degree()`, `gaussian_type()`, `complexity()`, `lambda_matrix()` | `m`, `T`, the number of terms of the multiplication's bilinear form (`2m − 1` for types 1 and 2), and that form as an `m × m` `BitMatrix` |
| `element(&coordinates)`, `zero()`, `one()` | elements; a coordinate set beyond `m − 1` panics |
| `minimal_polynomial()` | the exponents of the minimal polynomial of `θ` over `GF(2)`, the modulus of the polynomial basis in which `x` is `θ` |
| `+ - * /`, `square()`, `sqrt()`, `frobenius(k)`, `trace()`, `pow_u64(e)`, `inverse()` on `NormalElement` | the field's operations; `frobenius(k)` is `a^(2ᵏ)` for `k` of either sign, a rotation |
| `conversion(&field)` | the `BasisConversion` to and from the polynomial basis of `field`, which maps `θ` to the least (as a number) of the roots of its minimal polynomial in `field` |
| `conversion_with(&theta)` | the conversion that maps `θ` to a given root `theta`, or `None` if it is no root; FIPS 186-4 §D.10 gives such a `theta` for each of its fields |
| `to_normal(&a)`, `to_polynomial(&b)`, `normal_element()`, `polynomial_generator()` on `BasisConversion` | the two directions, `θ` in the polynomial basis (FIPS 186-4 §D.10's top row) and `x` in the normal basis (§D.9's second row from the bottom) |

- **Products** by Massey and Omura's method as FIPS 186-4 §D.3 writes it: with `F(2ⁱ·uʲ mod p) = i`, coordinate 0 of a product is the bilinear form `c₀ = Σ_{k=1}^{p−2} a_{F(k+1)}·b_{F(p−k)}`, to which an odd type adds `Σᵢ aᵢ·b_{i+m/2}` (then `−1 = 2^(m/2)·uʲ` is not in the group of `u`, and the pairs with `n + n' = 0` contribute that sum times 1, the all-ones element). Coordinate `i` is the same form of `a` and `b` rotated by `i`. The form's terms, counted modulo 2, are found once; a product is one pass over them with the coordinates in words, each term an `and` of two rotations: `O(T·m·⌈m/64⌉)` word operations.
- **Inverses** by Itoh and Tsujii's method ("A fast algorithm for computing multiplicative inverses in GF(2ᵐ) using normal bases", *Information and Computation* 78, 1988): `a⁻¹ = (a^(2^(m−1) − 1))²` along the bits of `m − 1` (`bₖ = a^(2ᵏ − 1)`, `b_{2k} = bₖ^(2ᵏ)·bₖ`, `b_{k+1} = bₖ²·a`), `O(log m)` products, the powers of 2 rotations.
- **Conversions.** The minimal polynomial `g` of `θ` comes from the powers `θ⁰ … θᵐ`, `m` products in the normal basis and the inverse of an `m × m` matrix over `GF(2)` (computed when a conversion first needs it, and kept). A root of `g` in the polynomial-basis field is found by Berlekamp's trace algorithm ("Factoring polynomials over large finite fields", *Math. Comp.* 24, 1970): for a seeded random `u`, `T(y) = Σᵢ (u·y)^(2ⁱ) mod h(y)` is `Tr(u·r)` at each root `r` of `h`, so `gcd(h, T)` keeps the roots of trace 0, and the part of at most half the degree is split again until one root is left. `T mod g` needs no products (its coefficients are sums of the conjugates `u^(2ⁱ)` by the bits of `y^(2ⁱ) mod g`); each split costs a gcd of polynomials of degree at most `m` over the field, `O(m²)` products, and there are about `log₂ m` splits. The conversion's matrix has the rows `θ'^(2ⁱ)` for the root `θ'` chosen, and the other direction is its inverse over `GF(2)`, `O(m³/64)`. The roots of `g` are the conjugates of one another, and each gives a conversion that differs from the others by a rotation of the normal coordinates; `conversion` takes the least as a number, so that it does not depend on the root the random splits found, and `conversion_with` takes the one given. Each conversion of an element is a product by one of the matrices, `O(m²/64)`.

### Example

```rust
use a::math::field::{BinaryField, Clmul, NormalBasis};

// sect163k1's field, x¹⁶³ + x⁷ + x⁶ + x³ + 1; Rabin's test runs when it is built.
let f = BinaryField::new(&[163, 7, 6, 3, 0]);
let a = f.element(&[0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 0x5]);
let b = f.element(&[0x1111, 0x2222, 0x3]);
assert_eq!(&(&a * &b) / &b, a);
assert_eq!(&a * &a.inverse().unwrap(), f.one());
assert_eq!(a.sqrt().square(), a);
assert_eq!((&a + &b).trace(), a.trace() ^ b.trace());
// z² + z = c has a root exactly when Tr(c) = 0 (and Tr(1) = 1 for odd m).
let c = if a.trace() == 0 { a.clone() } else { &a + &f.one() };
let z = c.quadratic_root().unwrap();
assert_eq!(&z.square() + &z, c);
// The portable kernel gives the same products as the CPU's.
let p = BinaryField::with_clmul(&[163, 7, 6, 3, 0], Clmul::Portable);
assert_eq!((&p.element(a.words()) * &p.element(b.words())).words(), (&a * &b).words());
assert!(BinaryField::is_irreducible(&[233, 74, 0]));
assert!(!BinaryField::is_irreducible(&[4, 2, 0])); // (x² + x + 1)²

// The type 4 Gaussian normal basis of GF(2¹⁶³), and the conversion to and from f's basis.
let nb = NormalBasis::new(163);
assert_eq!(nb.gaussian_type(), 4);
let conv = nb.conversion(&f);
let (na, nb_) = (conv.to_normal(&a), conv.to_normal(&b));
assert_eq!(&na * &nb_, conv.to_normal(&(&a * &b)));
assert_eq!(na.square(), conv.to_normal(&a.square())); // a rotation
assert_eq!(na.trace(), a.trace());
assert_eq!(conv.to_polynomial(&na.inverse().unwrap()), a.inverse().unwrap());
assert_eq!(conv.to_normal(&f.one()), nb.one());
```

### Verified

- **NTL and OpenSSL.** 83 fields (`lib/tests/math/field/data/binary_ntl.txt`, written by `gen_binary.cpp`): the SEC 2 fields of degrees 113, 131, 163, 193, 233, 239 (both `x²³⁹ + x³⁶ + 1` and SEC 2's `x²³⁹ + x¹⁵⁸ + 1`), 283, 409 and 571, AES's and GCM's, the reciprocals of five of them (whose terms are close to `xᵐ`, so short chunks or Barrett's reduction), `x + 1` and `x`, 16 fields of one word, of degrees 3 to 64, that are dense or have terms close to `xᵐ` (`x⁶⁴ + x⁶³ + x³⁷ + x³³ + 1`, `x⁶³ + x⁶² + 1`, dense moduli of degrees 13, 33, 63 and 64, and others: Barrett's reduction under an instruction, with `element`'s chunks of `m` bits), NTL's `BuildSparseIrred` of 30 degrees from 2 to 2000 (31 to 33, 63 to 65, 127 to 129, 255 to 257, 511 to 513 among them, around the edges of words), and NTL's `BuildRandomIrred`, dense, of 18 degrees from 3 to 1500. In each, 24 seeded cases (8 above degree 600) of nine operations: `a·b`, `a²`, `a⁻¹`, `a/b`, `√a`, `aᵉ` for a 64-bit `e`, `Tr(a)`, the half-trace for odd `m`, and the reduction of a random polynomial of `3⌈m/64⌉ + 1` words. NTL 11.6 (`GF2E`, `GF2X`) computed every output and OpenSSL 4.1 (`BN_GF2m_mod_*_arr`) every one it accepts, 12 677 of the 16 432 (it refuses inverses and quotients for more than five terms or a degree above 661, and has no field `x`); the generator stops if the two disagree. The tests regenerate the inputs from their seeds and compare a digest (FNV-1a) of the outputs of each operation in each field: all 16 432 agree with the CPU's kernel, and the 16 032 of the 77 fields of degree up to 640 agree with the portable kernel too, so that both reductions are compared in the fields of one word where the two kernels choose differently.
- **The kernels** are those the CPU reports (`Clmul::detect()` is `Pmull` exactly when `aes` is detected on aarch64, `Pclmulqdq` exactly when `pclmulqdq` is on x86_64), and agree with a product bit by bit on 2049 pairs of words (the 49 pairs of 0, 1, 2, `2⁶³`, `2⁶⁴ − 1`, `0x8000…0001` and `0x5555…5555`, and 2000 random ones), each kernel the CPU has. Products in 16 fields of every length from 1 to 12 words (so every shape of Karatsuba's split), under both reductions, agree with a product and a reduction bit by bit, by both kernels.
- **Irreducibility.** 600 random polynomials of degrees 2 to 601, trinomials, pentanomials and dense ones (`gen_binary.cpp` draws them as the test does): `is_irreducible` agrees with NTL's `IterIrredTest` on all 600 (19 irreducible). All 16 382 polynomials of degrees 1 to 13 were tested, and the irreducible ones of each degree are as many as Gauss's formula `(1/d)·Σ_{k|d} μ(k)·2^(d/k)` gives. Reducible moduli of every kind panic (no constant term, an even number of terms, `(x² + x + 1)²`, the trinomial `x²⁵⁶ + x³ + 1`).
- **Traces and quadratic equations.** `trace()` equals `Σ a^(2ⁱ)` computed by squarings in 10 fields, sparse, Barrett's and of degree 1; `quadratic_root` solves `z² + z = c` exactly when `Tr(c) = 0` in 8 fields of odd and even degree, and over `GF(2⁴)` for every `c` (8 of the 16 have roots); `half_trace` agrees with NTL (and with OpenSSL's `BN_GF2m_mod_solve_quad`) as above.
- **Normal bases against FIPS 186-4.** The example of §D.3 (`GF(2⁷)`, type 4): the bilinear form `F(u, v)`, all 21 terms, and `(1010111)·(1100001) = (1011001)`. For each NIST degree, 163, 233, 283, 409 and 571 (`normal_fips186.txt`, transcribed by `gen_normal_fips186.py`, which checks every value against the others in Python): the type of §D.1.3, the conversion with FIPS's `θ` (§D.10) and its row for `x` (§D.9), and the 25 normal-basis values of §D.1.3 (`b`, `Gx` and `Gy` of the B curves, `Gx` and `Gy` of the K curves) from their polynomial-basis values, both ways, exactly; through `conversion`'s own `θ` the same values up to one rotation, the same for all; and the curve equations in both bases.
- **Normal bases against PARI/GP.** The minimal polynomials of 121 Gauss periods, which PARI 2.19 computed from a primitive `p`-th root of unity in `GF(2ᵏ)` (`normal_pari.txt`, from `gen_normal.gp`), equal `minimal_polynomial()`: the lowest type of every `m` from 2 to 100 not divisible by 8 (types 1 to 12), the lowest types of the standards' degrees to 571, of degrees whose period needs an extension of degree above 2 (27, 54, 62, 63, 75, 102, 126, 145, 156, 171, 190, 198, with types up to 22), and 13 bases of other types; 36 of the 121 are of odd type. PARI's lowest type agrees with `NormalBasis::new`'s in all of them.
- **Normal bases against the polynomial bases.** Through `conversion`, for every `m` from 2 to 100 not divisible by 8 (in its lowest type, with the trinomial or pentanomial of the standards) and 10 other pairs of a type and a modulus (types 2 to 16, reciprocal moduli among them): sums, products, squares, square roots, traces, inverses, quotients and 1000th powers of random elements agree with those of the polynomial basis, and 1, 0 and `θ` map as they must. The optimal normal bases of 12 degrees have `2m − 1` terms; the bilinear form is symmetric.
- Built with `RUSTFLAGS="-C target-feature=-aes"` on aarch64 (Apple M4), so that `PMULL` is found at run time by `is_aarch64_feature_detected!("aes")` rather than at compile time, the 55 tests of `field::binary`, `field::normal` and `field::gf2` pass, `Clmul::detect()` being `Pmull`. No gate builds this configuration (see Limits).
- On x86_64 (Ubuntu 26.04, its x86-64-v3 glibc, Rust 1.100, in the CI container under emulation, whose CPU reports `pclmulqdq`, so that `Clmul::detect()` is `Pclmulqdq`) the same 42 tests of `field::binary` and `field::normal` pass, the comparisons with NTL and OpenSSL by both kernels among them.

To run the tests: `cargo test --test math -- field::binary field::normal`. The data files are written again by the programs beside them: `gen_binary.cpp` with NTL and OpenSSL 4 (`c++ -O2 -std=c++17 gen_binary.cpp -lntl -lgmp -lcrypto`), `gp -q gen_normal.gp`, and `python3 gen_normal_fips186.py`.

### Limits

- Not constant-time: branches and the work of the Euclidean algorithm depend on the values. For mathematics and public data, not for secret keys.
- Each element owns a `Vec` of words, so the operators that make a new element allocate; the assignments (`+=`, `*=`, `/=`) and `pow_u64` work in the element's own words. Products of up to eight words are the schoolbook of `PMULL` where the target enables it at compile time, Karatsuba's above (its halves of nine words took 0.91 to 0.95 times the time of the schoolbook's 81 products); the choice between the sparse and Barrett's reduction is a rough count, calibrated on one machine at one and two words only, and the half-trace is by squarings, not by a table: these two were not tuned.
- `PCLMULQDQ` and the x86_64 code paths have been tested under emulation only. `PMULL` needs the AES extension; aarch64 CPUs without it use the portable kernel. On aarch64 targets that do not enable `aes` at compile time (some Linux and Windows ones), `PMULL` is found at run time; that branch is built by no gate, since Apple's aarch64 enables the feature and CI is x86_64. Built with `RUSTFLAGS="-C target-feature=-aes"` on this Mac, it finds `PMULL` at run time and the tests above pass (see Verified); a CI job with that flag, or on an aarch64 Linux target, would keep it covered.
- Only Gaussian normal bases, so no normal basis for `m` divisible by 8. Building a basis searches the types upwards and holds a table of `p = Tm + 1` entries while it finds the form's terms: it is for the degrees of the standards and the like, up to some thousands. A product in a normal basis costs `T` times a product of coordinates by rotation, far more than in a polynomial basis; a conversion's setup costs `O(m²)` products of the field (about `log₂ m` splits of Berlekamp's algorithm), and is meant to be made once and kept.
- OpenSSL checked only what it accepts (above); NTL alone checked the inverses and quotients in fields with more than five terms or of degree above 661.

## GF(2⁸) and GF(2¹²⁸)

`Gf2_8` and `Gf2_128` are the two fields of AES and GCM as small `Copy` types, with no field object to build.

| Call | Gives |
| --- | --- |
| `Gf2_8::new(b)`, `to_byte()`, `From<u8>`, `ZERO`, `ONE` | the element of a byte in FIPS 197's convention, bit `i` the coefficient of `xⁱ` (`{57}` is `x⁶ + x⁴ + x² + x + 1`), modulo `x⁸ + x⁴ + x³ + x + 1` |
| `+ - * /`, unary `-`, `+=` … `/=`; `inverse()`, `pow(e)`, `xtime()` | the field's operations; `xtime` is FIPS 197's product by `x` |
| `aes_sbox()`, `aes_inverse_sbox()` | the AES S-box, the inverse in the field (0 for 0) then FIPS 197's affine map, and its inverse |
| `Gf2_128::from_bytes(block)`, `to_bytes()` | the element of a 16-byte GCM block, in GCM's bit-reflected order: the most significant bit of the first byte is the coefficient of `x⁰`, the least significant of the last that of `x¹²⁷`, modulo `x¹²⁸ + x⁷ + x² + x + 1` |
| `Gf2_128::from_polynomial(p)`, `polynomial()` | the same element as a `u128`, bit `i` the coefficient of `xⁱ`, as `BinaryField::new(&[128, 7, 2, 1, 0])` holds it |
| `+ - * /`, `inverse()`, `square()`, `pow(e)` (a `u128` exponent), `mul_with(b, Clmul::Portable)` | the field's operations; `mul_with` multiplies by a chosen kernel |
| `Gf2_128::ghash(h, blocks)`, `Gf2_128::gcm_ghash(h, a, c)` | `GHASH_H(X)` of NIST SP 800-38D (§6.4) over whole blocks, and GCM's `GHASH(H, A, C)` (McGrew and Viega, §2.3) over the additional data and the ciphertext, each padded, then `len(A) ‖ len(C)`; GCM's tag is this plus the encryption of the first counter block |

### Methods

- **`Gf2_8`** multiplies, inverts and raises to powers by tables of the powers and the logarithms of the generator `{03}` (`x + 1`), built at compile time: a product is the power of the sum of two logarithms. The S-box is `b ↦ b + (b ⋘ 1) + (b ⋘ 2) + (b ⋘ 3) + (b ⋘ 4) + {63}` applied to the inverse, which is FIPS 197's map `bᵢ' = bᵢ + b_{i+4} + b_{i+5} + b_{i+6} + b_{i+7} + cᵢ`; its inverse is `s ↦ (s ⋘ 1) + (s ⋘ 3) + (s ⋘ 6) + {05}` followed by the inverse.
- **`Gf2_128`** holds the polynomial in the natural order and reverses the bits at `from_bytes` and `to_bytes`. A product is the 256-bit carry-less product by Karatsuba's method, three products of words by the CPU's kernel (`Clmul::detect`, as for `BinaryField`), then the reduction by `x¹²⁸ = x⁷ + x² + x + 1`: the top half times `x⁷ + x² + x + 1`, whose 7 bits above `x¹²⁷` are folded once more. A square spreads the bits and reduces the same way, with no products. The inverse is Itoh and Tsujii's `(a^(2¹²⁷ − 1))²`, 12 products and 127 squares. GHASH is `Yᵢ = (Yᵢ₋₁ + Xᵢ)·H` over the blocks.

### Example

```rust
use a::math::field::{Gf2_8, Gf2_128};

// FIPS 197, 4.2 and 5.1.1.
assert_eq!(Gf2_8::new(0x57) * Gf2_8::new(0x83), Gf2_8::new(0xc1));
assert_eq!(Gf2_8::new(0x53).inverse(), Some(Gf2_8::new(0xca)));
assert_eq!(Gf2_8::new(0x53).aes_sbox(), Gf2_8::new(0xed));
assert_eq!(Gf2_8::new(0xed).aes_inverse_sbox(), Gf2_8::new(0x53));

// GCM's test case 2 (McGrew and Viega, appendix B): K = 0¹²⁸, IV = 0⁹⁶, P = 0¹²⁸.
let block = |s: &str| -> [u8; 16] {
    core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
};
let h = Gf2_128::from_bytes(block("66e94bd4ef8a2c3b884cfa59ca342b2e"));
let c = block("0388dace60b6a392f328c2b971b2fe78");
let ghash = Gf2_128::gcm_ghash(h, &[], &c);
assert_eq!(ghash.to_bytes(), block("f38cbb1ad69223dcc3457ae5b6b0f885"));
// The tag is GHASH plus E(K, Y₀).
let tag = ghash + Gf2_128::from_bytes(block("58e2fccefa7e3061367f1d57a4e7455a"));
assert_eq!(tag.to_bytes(), block("ab6e47d42cec13bdf53a67b21257bddf"));
// The block 80 00 … 00 is 1; x¹²⁷·x = x⁷ + x² + x + 1.
assert_eq!(Gf2_128::from_bytes(block("80000000000000000000000000000000")), Gf2_128::ONE);
let x = Gf2_128::from_polynomial(2);
assert_eq!(Gf2_128::from_polynomial(1 << 127) * x, Gf2_128::from_polynomial(0x87));
assert_eq!(h * h.inverse().unwrap(), Gf2_128::ONE);
```

### Verified

- **FIPS 197.** `aes_sbox` and `aes_inverse_sbox` equal Tables 4 and 6 of FIPS 197 (2001, updated 2023) at all 256 bytes; the examples of §4.1, §4.2 (with the chain of `xtime` of §4.2.1) and §5.1.1. All 65 536 products and 65 280 quotients of `Gf2_8` agree with `BinaryField::new(&[8, 4, 3, 1, 0])`, and so do the inverses of every element and its powers by nine exponents from 0 to `2⁶⁴ − 1`; `{03}` generates the 255 nonzero elements.
- **The GCM specification.** `gcm_ghash` gives `GHASH(H, A, C)` of all 18 test cases of McGrew and Viega's appendix B (AES-128, -192 and -256; empty, one-block and four-block ciphertexts, with and without additional data), by the CPU's kernel and by the portable one; `ghash` over the padded blocks the same; the `Y₀ = GHASH(H, {}, IV)` of the six cases whose IV is not of 96 bits (test cases 5, 6, 11, 12, 17, 18); the six intermediate values `X₁ … X₆` of test case 4; and its tag.
- **OpenSSL.** 64 random AES-128-GCM computations by OpenSSL 4.1 (`gf2_128_openssl.txt`, from `gen_gf2_128.c`): random keys, IVs, 0 to 69 bytes of additional data and ciphertexts of 0 to 89 bytes, OpenSSL's tag less `E(K, J₀)`, equal to `gcm_ghash` in all 64.
- **The polynomial basis.** Products, squares and inverses of 205 pairs (`0`, `1`, `x`, `x¹²⁷`, all ones, `x⁷ + x² + x + 1` and random ones) agree with `BinaryField::new(&[128, 7, 2, 1, 0])` by either kernel, and `mul_with` gives the same by every kernel the CPU has; `a·a⁻¹ = 1` and `a^(2¹²⁸ − 1) = 1` on random elements; the bit order of every one of the 128 unit blocks.
- On x86_64, as above, the same 13 tests of `field::gf2_8` and `field::gf2_128` pass, `PCLMULQDQ` and the portable kernel compared on every product.

To run the tests: `cargo test --test math -- field::gf2`; `gen_gf2_128.c` writes its data again with OpenSSL.

### Limits

- Not constant-time: `Gf2_8` looks up tables indexed by the values and both branch on zero. This is the field arithmetic of AES and GCM, not an implementation of AES or of GCM for secret keys.
- Where the target enables `PMULL` at compile time (Apple's aarch64), `Gf2_128`'s products and squares are three and two `PMULL`s and a reduction by two more, and GHASH takes eight blocks a reduction with `H…H⁸` (from eight blocks on; four for what is left of four or more), kept for the last key of each thread (a cache of one entry); elsewhere a product goes one at a time through `Clmul::detect`'s kernel, with no aggregated reduction.

## Measured

Measured on an Apple M4 Max on 2026-11-26, between 07:57 and 08:02, with nothing else running on the machine, by the harness in the repository's `bench/fields` (its method in `bench/README.md` there). This crate is built as a user gets it, in the default release profile (opt-level 3, no LTO, no `target-cpu`) of rustc 1.100.0, and so are the Rust references: arkworks 0.5 (`ark-ff` 0.5.0 with `ark-bn254` and `ark-bls12-381`, and fields of `#[derive(MontConfig)]` for the other primes), blst 0.3.17 (its own assembly), the `bls12_381` crate 0.8.0 and RustCrypto's `ghash` 0.6.0; FLINT 3.6.0, NTL 11.6.0, GMP 6.3.0 and OpenSSL 4.1.0 are Homebrew's builds, called through C shims compiled at `-O3 -mcpu=native`, and mcl 4.44.0 is herumi/mcl at commit `a1c4e29e` in its 384-bit build. One thread. Every side runs the same loop: products, squares and sums as dependent chains (`acc = acc · x[j]` over 256 inputs taken in turn, `acc = acc²`, `acc = acc + x[j]`; this crate's by `*=` and `+=`, and in the prime fields and the towers by `square_in_place` and the like), and inverses, square roots of squares, powers, Frobenius maps, norms and traces of fresh inputs, added up. Before the clock starts, each side's outputs are compared with this crate's (square roots up to the sign); then the sides alternate sample by sample in one process, 21 samples of at least 2 ms (15 where an operation takes more than 20 ms), and a figure is the median.

A cell gives this crate's time and, in parentheses, that time over the time of the fastest reference that does the same work, with its name: below 1, this crate is faster. The names: `ark` arkworks, `blst`, `bls12_381` the `bls12_381` crate, `mcl`; FLINT's `nmod`, `mpn_mod`, `fmpz_mod`, `fq_nmod`, `fq_zech` and `fq`; NTL's `ZZ_p`, `zz_pE`, `ZZ_pE`, `GF2E` and `IterIrredTest` (`NTL`); `GMP` (`mpz`); `OpenSSL` (`BN`, `BN_GF2m`, AES-GCM); `ghash`. arkworks's and blst's Legendre symbols are Euler's criterion, a power, and NTL's square roots in `GF(2ᵐ)` and Frobenius maps in `GF(pᵏ)` are powers too: the same values by their own methods, and counted. Not counted, as they do other work: blst's and the `bls12_381` crate's constant-time inverses, square roots and squareness tests (a guarantee this crate does not give), arkworks's `from_be_bytes_mod_order` (it reduces bytes of any length, where `from_bytes_be` refuses those not below `p`), and RustCrypto's `polyval` (its blocks byte-reversed before the clock starts).

### Prime fields

The forms are those of [the table of forms](#forms), and a `TypedFp` row is the same prime as a type of `field::named`; `sqrt` is of squares, `pow` by an exponent as long as `p`, and `batch_invert` the time per element of a batch of 256:

| Field | `mul` | `square` | `add` | `inverse` | `sqrt` | `legendre` | `pow` | `batch_invert` | `to_bytes_be` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Goldilocks `2⁶⁴ − 2³² + 1` | 4.10 ns (1.18× ark) | 4.02 ns (1.15× ark) | 1.65 ns (1.69× nmod) | 116 ns (1.06× nmod) | 856 ns (0.780× ark) | 72.5 ns (1.17× nmod) | 390 ns (2.31× nmod) | 15.5 ns (1.92× ark) | 12.7 ns (1.48× ark) |
| Goldilocks, `TypedFp` | 2.87 ns (0.846× ark) | 2.84 ns (0.830× ark) | 1.22 ns (1.28× nmod) | 102 ns (0.934× nmod) | 807 ns (0.734× ark) | 52.7 ns (0.834× nmod) | 245 ns (1.47× nmod) | 6.61 ns (0.855× ark) | 8.42 ns (1.00× ark) |
| a prime of 128 bits | 9.25 ns (1.23× ark) | 9.09 ns (1.14× ark) | 3.99 ns (4.59× ark) | 316 ns (0.938× GMP) | 1.47 µs (0.950× ark) | 155 ns (1.10× fmpz_mod) | 1.41 µs (0.938× ark) | 26.5 ns (1.57× ark) | 13.8 ns (1.43× ark) |
| BN254, `p` | 13.5 ns (1.06× ark) | 13.4 ns (0.999× ark) | 3.48 ns (3.11× ark) | 665 ns (0.947× GMP) | 4.06 µs (0.845× ark) | 340 ns (0.660× fmpz_mod) | 4.08 µs (0.912× OpenSSL) | 41.6 ns (1.07× ark) | 19.0 ns (1.41× ark) |
| BN254, `p`, `TypedFp` | 11.5 ns (0.910× ark) | 11.9 ns (0.885× ark) | 1.79 ns (1.58× ark) | 648 ns (0.913× mpn_mod) | 3.72 µs (0.787× ark) | 354 ns (0.685× fmpz_mod) | 3.97 µs (0.862× OpenSSL) | 33.5 ns (0.875× ark) | 14.2 ns (1.08× ark) |
| BLS12-381, `r` | 13.3 ns (1.09× blst) | 13.2 ns (1.06× blst) | 3.44 ns (3.04× ark) | 652 ns (0.946× GMP) | 7.11 µs (0.825× ark) | 336 ns (0.662× fmpz_mod) | 3.99 µs (0.893× OpenSSL) | 41.2 ns (1.11× ark) | 18.9 ns (2.81× bls12_381) |
| BLS12-381, `r`, `TypedFp` | 11.2 ns (0.919× blst) | 11.5 ns (0.927× blst) | 1.74 ns (1.55× ark) | 631 ns (0.904× GMP) | 6.72 µs (0.779× ark) | 345 ns (0.679× GMP) | 3.71 µs (0.829× OpenSSL) | 32.6 ns (0.892× ark) | 14.4 ns (2.14× bls12_381) |
| `2²⁵⁵ − 19` | 8.91 ns (0.649× OpenSSL) | 7.79 ns (0.575× ark) | 3.06 ns (2.90× ark) | 645 ns (0.916× GMP) | 1.77 µs (0.253× ark) | 332 ns (0.656× fmpz_mod) | 2.11 µs (0.477× OpenSSL) | 28.7 ns (0.715× ark) | 14.2 ns (1.08× ark) |
| `2²⁵⁵ − 19`, `TypedFp` | 6.87 ns (0.501× OpenSSL) | 5.92 ns (0.434× ark) | 1.42 ns (1.36× ark) | 619 ns (0.898× GMP) | 2.09 µs (0.291× ark) | 344 ns (0.669× fmpz_mod) | 2.02 µs (0.454× OpenSSL) | 21.1 ns (0.523× ark) | 9.41 ns (0.701× ark) |
| secp256k1, `p` | 8.94 ns (0.650× OpenSSL) | 7.84 ns (0.572× OpenSSL) | 3.05 ns (2.04× ark) | 629 ns (0.911× GMP) | 1.79 µs (0.276× OpenSSL) | 331 ns (0.652× fmpz_mod) | 2.09 µs (0.477× OpenSSL) | 28.6 ns (0.689× ark) | 14.0 ns (1.05× ark) |
| secp256k1, `p`, `TypedFp` | 6.89 ns (0.505× OpenSSL) | 5.90 ns (0.432× OpenSSL) | 1.42 ns (0.951× ark) | 621 ns (0.894× mpn_mod) | 2.02 µs (0.312× OpenSSL) | 344 ns (0.672× fmpz_mod) | 2.02 µs (0.460× OpenSSL) | 21.1 ns (0.503× ark) | 9.14 ns (0.695× ark) |
| P-256, `p` | 9.26 ns (0.904× ark) | 8.33 ns (0.839× ark) | 3.45 ns (2.65× ark) | 662 ns (0.958× mpn_mod) | 1.75 µs (0.649× ark) | 335 ns (0.662× GMP) | 2.25 µs (0.616× ark) | 31.4 ns (1.09× ark) | 17.3 ns (1.52× ark) |
| P-256, `p`, `TypedFp` | 7.22 ns (0.721× ark) | 6.31 ns (0.643× ark) | 1.71 ns (1.32× ark) | 620 ns (0.915× mpn_mod) | 1.78 µs (0.663× ark) | 343 ns (0.677× GMP) | 2.14 µs (0.588× ark) | 23.5 ns (0.815× ark) | 12.6 ns (1.11× ark) |
| BLS12-381, `p` | 19.7 ns (1.04× blst) | 19.6 ns (0.977× mcl) | 3.86 ns (1.12× ark) | 980 ns (0.912× GMP) | 8.75 µs (0.900× mcl) | 558 ns (0.676× fmpz_mod) | 9.13 µs (0.958× mcl) | 62.3 ns (0.828× ark) | 26.3 ns (1.18× blst) |
| BLS12-381, `p`, `TypedFp` | 17.8 ns (0.940× blst) | 18.9 ns (0.952× mcl) | 2.30 ns (0.681× ark) | 943 ns (0.883× GMP) | 8.59 µs (0.886× mcl) | 569 ns (0.692× fmpz_mod) | 9.09 µs (0.949× mcl) | 56.9 ns (0.740× ark) | 28.0 ns (1.26× blst) |
| P-384, `p` | 19.9 ns (0.841× ark) | 18.8 ns (0.882× ark) | 3.86 ns (1.09× ark) | 997 ns (0.927× GMP) | 6.83 µs (0.459× ark) | 571 ns (0.694× fmpz_mod) | 8.12 µs (0.629× ark) | 67.4 ns (0.864× ark) | 36.7 ns (1.76× ark) |
| P-384, `p`, `TypedFp` | 17.8 ns (0.752× ark) | 16.5 ns (0.785× ark) | 2.23 ns (0.638× ark) | 956 ns (0.878× GMP) | 7.85 µs (0.531× ark) | 573 ns (0.701× GMP) | 8.23 µs (0.641× ark) | 58.5 ns (0.734× ark) | 21.2 ns (1.03× ark) |
| a prime of 512 bits | 50.6 ns (1.30× OpenSSL) | 46.3 ns (1.56× OpenSSL) | 4.14 ns (0.867× ark) | 1.44 µs (0.973× GMP) | 31.9 µs (1.45× OpenSSL) | 967 ns (0.779× GMP) | 31.6 µs (1.61× OpenSSL) | 170 ns (1.19× ark) | 58.2 ns (1.85× ark) |
| P-521, `2⁵²¹ − 1` | 18.1 ns (0.584× ark) | 12.0 ns (0.508× ark) | 2.96 ns (0.618× ark) | 1.36 µs (0.886× GMP) | 6.34 µs (0.510× ark) | 952 ns (0.773× GMP) | 7.84 µs (0.386× ark) | 78.1 ns (0.629× ark) | 18.8 ns (0.807× ark) |
| a prime of 576 bits | 65.6 ns (1.31× ark) | 60.6 ns (1.34× ark) | 4.70 ns (0.886× ark) | 1.63 µs (0.960× GMP) | 44.2 µs (1.13× ark) | 1.16 µs (0.803× GMP) | 44.0 µs (1.11× ark) | 210 ns (1.15× ark) | 71.7 ns (2.14× ark) |
| a prime of 600 bits | 90.1 ns (1.11× OpenSSL) | 88.5 ns (1.10× OpenSSL) | 12.7 ns (1.67× mpn_mod) | 1.50 µs (0.804× GMP) | 28.7 µs (0.553× ZZ_p) | 1.15 µs (0.787× GMP) | 28.2 µs (0.554× GMP) | 318 ns | 27.3 ns |
| a prime of 768 bits | 108 ns (1.39× OpenSSL) | 111 ns (1.61× ark) | 16.7 ns (2.21× ark) | 2.25 µs (0.932× GMP) | 48.7 µs (0.651× OpenSSL) | 1.64 µs (0.726× GMP) | 48.5 µs (0.677× OpenSSL) | 382 ns (1.20× ark) | 27.3 ns (0.562× ark) |
| a prime of 1024 bits | 157 ns (1.19× OpenSSL) | 154 ns (1.63× OpenSSL) | 22.6 ns (2.53× mpn_mod) | 3.00 µs (0.869× GMP) | 109 µs (0.860× OpenSSL) | 2.53 µs (0.783× GMP) | 108 µs (0.888× OpenSSL) | 542 ns (0.899× ark) | 28.9 ns (0.374× ark) |
| a prime of 2048 bits | 546 ns (1.07× OpenSSL) | 474 ns (1.42× OpenSSL) | 42.7 ns (2.51× GMP) | 6.86 µs (0.875× GMP) | 1.08 ms (0.647× OpenSSL) | 6.23 µs (0.840× fmpz_mod) | 1.08 ms (1.26× OpenSSL) | 1.74 µs (0.619× ark) | 28.3 ns (0.0793× ark) |
| a prime of 4096 bits | 2.21 µs (1.20× fmpz_mod) | 1.72 µs (1.35× OpenSSL) | 74.7 ns (2.85× GMP) | 17.0 µs (0.940× GMP) | 7.82 ms (0.646× OpenSSL) | 13.3 µs (0.816× fmpz_mod) | 7.83 ms (1.21× OpenSSL) | 6.71 µs (0.270× ark) | 51.0 ns (0.0277× ark) |

- Inversion is faster than every reference at every size, 0.804 to 0.976 of the fastest one's time (GMP's `mpz_invert`, FLINT's `mpn_mod` and `nmod`), but for Goldilocks's word at run time (1.06): Lehmer's extended Euclid of `Number::mod_inverse`. The Legendre symbol takes 0.652 to 0.840 of the fastest reference's time but for Goldilocks at run time and the prime of 128 bits (1.17, 1.10), and `is_square` 0.000317 to 0.233 of arkworks's.
- The primes with forms of their own: products 0.584 to 0.650 of the fastest reference's time for `2²⁵⁵ − 19`, secp256k1's prime and `2⁵²¹ − 1`, square roots 0.253 to 0.510 and powers 0.386 to 0.477.
- `TypedFp`: products 0.501 to 0.940 and squares 0.432 to 0.952 of the fastest reference's time on all nine primes measured, `batch_invert` 0.503 to 0.892.

### Extension fields

`GF(3⁵)` holds its elements as logarithms (Zech's tables), the word primes as words, and the others as coefficients in `GF(p)` (the methods of extension fields above); `pow` is by an exponent of 256 bits:

| Field | `mul` | `square` | `add` | `inverse` | `frobenius(1)` | `norm` | `trace` | `sqrt` | `pow` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GF(3⁵)`, Zech's tables | 4.97 ns (4.93× fq_zech) | 4.50 ns (3.71× fq_zech) | 7.83 ns (2.94× fq_zech) | 10.8 ns (3.95× fq_zech) | 10.5 ns (1.93× fq_zech) | 35.3 ns (1.55× fq_zech) | 9.22 ns (1.42× zz_pE) | 18.5 ns (6.71× fq_zech) | 59.7 ns (3.06× fq_zech) |
| `GF(3¹²)` | 155 ns (1.67× fq_nmod) | 155 ns (1.82× fq_nmod) | 15.8 ns (1.62× zz_pE) | 709 ns (1.19× fq_nmod) | 114 ns (0.628× fq_nmod) | 784 ns (3.83× fq_nmod) | 12.4 ns (1.05× zz_pE) | 6.80 µs (0.904× fq_nmod) | 4.75 µs (2.15× fq_nmod) |
| `p = 10⁹ + 7`, `k = 2` | 25.8 ns (1.11× fq_nmod) | 23.0 ns (1.01× fq_nmod) | 13.3 ns (3.21× zz_pE) | 224 ns (1.03× fq) | 39.6 ns (0.0642× fq_nmod) | 45.8 ns (2.71× fq_nmod) | 6.68 ns (1.35× zz_pE) | 2.90 µs (0.570× fq_nmod) | 2.48 µs (1.99× fq_nmod) |
| `p = 10⁹ + 7`, `k = 8` | 82.8 ns (1.11× fq_nmod) | 87.6 ns (1.40× fq_nmod) | 13.8 ns (2.39× zz_pE) | 1.17 µs (1.14× fq_nmod) | 70.0 ns (0.0280× fq_nmod) | 628 ns (1.02× fq_nmod) | 10.4 ns (1.14× zz_pE) | 32.5 µs (0.333× fq_nmod) | 31.6 µs (1.55× fq_nmod) |
| `p = 10⁹ + 7`, `k = 24` | 387 ns (1.13× fq_nmod) | 335 ns (0.934× fq_nmod) | 14.8 ns (1.46× zz_pE) | 5.13 µs (1.32× fq_nmod) | 290 ns (0.0192× fq_nmod) | 1.66 µs (0.821× fq_nmod) | 22.1 ns (1.02× zz_pE) | 388 µs (0.264× fq_nmod) | 135 µs (1.03× fq_nmod) |
| `p = 10⁹ + 7`, `k = 64` | 2.07 µs (2.00× fq_nmod) | 1.35 µs (1.34× fq_nmod) | 26.8 ns (1.15× zz_pE) | 17.6 µs (1.21× fq_nmod) | 1.77 µs (0.0402× fq_nmod) | 5.02 µs (0.783× fq_nmod) | 50.7 ns (0.892× zz_pE) |  | 625 µs (1.62× fq_nmod) |
| BN254's `p`, `k = 2` | 97.0 ns (0.526× fq) | 74.4 ns (0.447× fq) | 21.6 ns (0.912× ZZ_pE) | 3.79 µs (1.79× fq) | 495 ns (0.00914× fq) | 91.1 ns (0.444× fq) | 13.4 ns (0.284× ZZ_pE) | 62.9 µs (0.115× fq) | 32.7 µs (0.570× fq) |
| BN254's `p`, `k = 6` | 460 ns (0.556× fq) | 328 ns (0.467× fq) | 46.0 ns (0.698× ZZ_pE) | 13.0 µs (1.16× fq) | 1.49 µs (0.00603× fq) | 7.43 µs (0.851× ZZ_pE) | 17.6 ns (0.266× ZZ_pE) | 869 µs (0.111× fq) | 137 µs (0.545× fq) |
| BN254's `p`, `k = 12` | 1.62 µs (0.549× fq) | 1.10 µs (0.363× fq) | 66.8 ns (0.485× ZZ_pE) | 32.2 µs (1.20× fq) | 6.96 µs (0.00651× fq) | 16.6 µs (0.840× ZZ_pE) | 36.4 ns (0.284× ZZ_pE) | 5.93 ms (0.0893× fq) | 482 µs (0.434× fq) |
| BN254's `p`, `k = 24` | 5.89 µs (0.837× fq) | 3.76 µs (0.504× fq) | 128 ns (0.499× ZZ_pE) | 91.4 µs (1.01× fq) | 19.9 µs (0.00767× fq) | 42.5 µs (0.830× ZZ_pE) | 44.9 ns (0.272× ZZ_pE) | 40.8 ms (0.156× fq) | 1.65 ms (0.623× fq) |
| BN254's `p`, `k = 64` | 39.7 µs (1.38× ZZ_pE) | 23.7 µs (0.942× ZZ_pE) | 326 ns (0.491× ZZ_pE) | 490 µs (1.17× ZZ_pE) | 111 µs (0.0136× ZZ_pE) | 196 µs (0.856× ZZ_pE) | 79.6 ns (0.283× ZZ_pE) |  | 11.6 ms (1.33× ZZ_pE) |
| BLS12-381's `p`, `k = 2` | 135 ns (0.540× fq) | 109 ns (0.500× fq) | 23.2 ns (0.844× ZZ_pE) | 5.62 µs (1.81× fq) | 582 ns (0.00469× fq) | 135 ns (0.534× fq) | 22.9 ns (0.437× ZZ_pE) | 137 µs (0.148× fq) | 44.7 µs (0.588× fq) |
| BLS12-381's `p`, `k = 12` | 2.97 µs (0.671× fq) | 1.94 µs (0.433× fq) | 71.3 ns (0.428× ZZ_pE) | 50.2 µs (1.30× fq) | 7.22 µs (0.00270× fq) | 25.8 µs (0.878× ZZ_pE) | 59.8 ns (0.382× ZZ_pE) | 15.8 ms (0.0870× fq) | 909 µs (0.521× fq) |
| a prime of 600 bits, `k = 3` | 1.69 µs (2.26× fq) | 1.40 µs (2.09× fq) | 108 ns (2.99× ZZ_pE) | 13.1 µs (1.14× fq) | 1.72 µs (0.00289× fq) | 8.20 µs (0.950× fq) | 266 ns (1.43× ZZ_pE) | 3.99 ms (0.225× fq) | 562 µs (2.24× fq) |
| a prime of 600 bits, `k = 8` | 10.1 µs (2.17× fq) | 9.10 µs (2.11× fq) | 258 ns (2.40× ZZ_pE) | 43.1 µs (1.20× fq) | 7.08 µs (0.00179× fq) | 23.1 µs (0.815× fq) | 278 ns (1.34× ZZ_pE) | 68.1 ms (0.539× fq) | 3.54 ms (2.15× fq) |

Frobenius maps take 0.00179 to 0.628 of the fastest reference's time but in `GF(3⁵)`, squareness tests 0.000297 to 0.390 and square roots 0.0870 to 0.904; over BN254's and BLS12-381's primes up to degree 24, products and squares 0.363 to 0.837 of FLINT's `fq`, powers 0.434 to 0.623, and traces 0.266 to 0.437 of NTL's.

### Towers

`frobenius(1)` of `F_p²` is its conjugate; `cyclotomic_square` and `cyclotomic_pow` (by the curve's `x`, 64 bits) are of elements of the cyclotomic subgroup, and the last column is the product by a line, `mul_by_034` on BN254 and `mul_by_014` on BLS12-381:

| Floor | `mul` | `square` | `add` | `inverse` | `sqrt` | `frobenius(1)` | `cyclotomic_square` | `cyclotomic_pow` | `mul_by_034`, `mul_by_014` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BN254, `F_p²` | 24.7 ns (0.664× ark) | 20.8 ns (0.628× ark) | 3.68 ns (1.53× ark) | 665 ns (0.344× ark) | 8.91 µs (0.563× ark) | 2.56 ns (1.80× ark) |  |  |  |
| BN254, `F_p⁶` | 208 ns (0.509× ark) | 145 ns (0.467× ark) | 8.64 ns (0.446× ark) | 989 ns (0.386× ark) |  | 75.5 ns (0.900× ark) |  |  |  |
| BN254, `F_p¹²` | 670 ns (0.486× ark) | 474 ns (0.494× ark) | 19.3 ns (0.385× ark) | 1.73 µs (0.420× ark) |  | 175 ns (0.660× ark) | 304 ns (0.517× ark) | 33.8 µs (0.511× ark) | 463 ns (0.564× ark) |
| BLS12-381, `F_p²` | 49.1 ns (0.924× blst) | 39.4 ns (1.12× mcl) | 4.43 ns (0.955× blst) | 998 ns (0.756× mcl) | 19.1 µs (0.744× mcl) | 2.63 ns (1.08× ark) |  |  |  |
| BLS12-381, `F_p⁶` | 364 ns (0.594× ark) | 251 ns (0.557× ark) | 13.5 ns (0.563× ark) | 1.64 µs (0.259× ark) |  | 120 ns (0.662× ark) |  |  |  |
| BLS12-381, `F_p¹²` | 1.16 µs (1.23× mcl) | 804 ns (1.20× mcl) | 27.9 ns (1.54× mcl) | 2.93 µs (0.965× mcl) |  | 311 ns (0.921× blst) | 512 ns (1.14× blst) | 36.7 µs (0.685× mcl) | 800 ns (1.02× blst) |

Inverses take 0.259 to 0.965 of the fastest reference's time, squareness tests 0.0466 to 0.159 of arkworks's, and BN254's products and squares 0.467 to 0.664 of arkworks's on every floor.

### Binary fields

Each in the polynomial basis by its standard's modulus, with `PMULL`; `pow_u64` is by an exponent of 63 bits. NTL's `GF2E` has no carry-less instruction on aarch64, and its square roots are powers:

| Field | `mul` | `square` | `inverse` | `sqrt` | `trace` | `quadratic_root` | `pow_u64` | `is_irreducible` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AES's `GF(2⁸)` | 18.1 ns (1.03× GF2E) | 19.9 ns (1.59× GF2E) | 43.0 ns (0.558× GF2E) | 46.5 ns (0.276× OpenSSL) | 1.24 ns (0.277× GF2E) | 532 ns (0.358× OpenSSL) | 1.85 µs (1.11× GF2E) | 444 ns (0.333× NTL) |
| `GF(2⁶⁴)` | 15.7 ns (0.530× GF2E) | 18.6 ns (0.882× OpenSSL) | 139 ns (0.478× GF2E) | 46.8 ns (0.0342× OpenSSL) | 1.24 ns (0.278× GF2E) | 4.07 µs (0.353× OpenSSL) | 1.65 µs (0.602× GF2E) | 1.83 µs (0.706× NTL) |
| `GF(2¹¹³)`, sect113 | 17.7 ns (0.805× GF2E) | 20.9 ns (1.27× GF2E) | 220 ns (0.404× GF2E) | 45.2 ns (0.0197× GF2E) | 1.70 ns (0.362× GF2E) | 2.07 µs (0.725× OpenSSL) | 1.70 µs (0.991× GF2E) | 3.03 µs (0.786× NTL) |
| GCM's `GF(2¹²⁸)` | 21.4 ns (0.965× GF2E) | 29.9 ns (1.64× GF2E) | 266 ns (0.417× GF2E) | 47.9 ns (0.0168× GF2E) | 1.72 ns (0.371× GF2E) | 12.7 µs (0.463× OpenSSL) | 2.01 µs (1.12× GF2E) | 4.75 µs (1.03× NTL) |
| `GF(2¹³¹)`, sect131 | 31.2 ns (0.855× GF2E) | 34.2 ns (1.22× GF2E) | 275 ns (0.412× GF2E) | 56.2 ns (0.0133× GF2E) | 1.74 ns (0.313× GF2E) | 3.42 µs (0.564× OpenSSL) | 2.58 µs (1.01× GF2E) | 5.34 µs (0.880× NTL) |
| `GF(2¹⁶³)`, sect163 | 24.4 ns (0.636× GF2E) | 31.4 ns (1.05× GF2E) | 337 ns (0.355× GF2E) | 55.1 ns (0.0104× GF2E) | 1.74 ns (0.350× GF2E) | 3.93 µs (0.621× OpenSSL) | 2.76 µs (1.02× GF2E) | 5.64 µs (0.718× NTL) |
| `GF(2¹⁹³)`, sect193 | 20.0 ns (0.543× GF2E) | 28.9 ns (1.38× GF2E) | 409 ns (0.297× GF2E) | 54.7 ns (0.0111× GF2E) | 1.95 ns (0.254× GF2E) | 3.84 µs (0.457× OpenSSL) | 1.99 µs (0.830× GF2E) | 6.83 µs (0.704× NTL) |
| `GF(2²³³)`, sect233 | 23.5 ns (0.902× GF2E) | 26.7 ns (1.40× GF2E) | 479 ns (0.272× GF2E) | 58.0 ns (0.0118× GF2E) | 1.95 ns (0.375× GF2E) | 5.19 µs (0.515× OpenSSL) | 2.12 µs (1.15× GF2E) | 7.21 µs (0.667× NTL) |
| `GF(2²⁸³)`, sect283 | 41.0 ns (0.838× GF2E) | 43.1 ns (1.16× GF2E) | 625 ns (0.283× GF2E) | 75.2 ns (0.00640× GF2E) | 1.99 ns (0.364× GF2E) | 12.8 µs (0.866× OpenSSL) | 3.93 µs (1.21× GF2E) | 13.8 µs (0.698× NTL) |
| `GF(2⁴⁰⁹)`, sect409 | 35.3 ns (1.06× GF2E) | 42.6 ns (1.87× GF2E) | 1.02 µs (0.278× GF2E) | 74.4 ns (0.00724× GF2E) | 2.68 ns (0.392× GF2E) | 14.5 µs (0.571× OpenSSL) | 3.28 µs (1.43× GF2E) | 19.2 µs (0.743× NTL) |
| `GF(2⁵⁷¹)`, sect571 | 84.2 ns (0.770× GF2E) | 80.1 ns (1.17× GF2E) | 2.43 µs (0.441× GF2E) | 123 ns (0.00295× GF2E) | 2.94 ns (0.414× GF2E) | 40.7 µs (0.779× OpenSSL) | 7.38 µs (1.17× GF2E) | 47.3 µs (0.648× NTL) |
| `GF(2¹⁰²⁴)` | 155 ns (0.947× GF2E) | 97.4 ns (0.991× GF2E) | 8.54 µs (0.708× GF2E) | 183 ns (0.00175× GF2E) | 2.94 ns (0.326× GF2E) | 390 µs (0.129× OpenSSL) | 9.95 µs (1.09× GF2E) | 111 µs (0.562× NTL) |
| `GF(2²⁰⁴⁸)` | 278 ns (0.682× GF2E) | 159 ns (0.798× GF2E) | 32.8 µs (0.983× GF2E) | 339 ns (0.000779× GF2E) | 5.28 ns (0.397× GF2E) | 1.26 ms (0.117× OpenSSL) | 17.9 µs (0.876× GF2E) | 363 µs (0.406× NTL) |

Products take 0.530 to 0.965 of NTL's time but at 8 bits (1.03) and 409 (1.06), inverses 0.272 to 0.983, square roots 0.000779 to 0.276 of the fastest reference's, traces 0.254 to 0.414; `quadratic_root` takes 0.117 to 0.866 of OpenSSL's time, and Rabin's test 0.333 to 1.03 of NTL's `IterIrredTest`.

### GF(2⁸), GF(2¹²⁸) and GHASH

`Gf2_8`'s products and powers are by its tables (no reference has a type of its own for `GF(2⁸)`: these are NTL's `GF2E` and FLINT's `fq_nmod` over `GF(2)`), `Gf2_128`'s by `PMULL`, and its `pow` by an exponent of 63 bits:

| Type | `*` | `square` | `inverse` | `/` | `pow` | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Gf2_8` | 3.14 ns (0.119× GF2E) | 2.30 ns (0.158× GF2E) | 0.388 ns (0.00513× GF2E) | 3.13 ns (0.0273× GF2E) | 0.739 ns (0.00511× fq_nmod) | `aes_sbox` 4.23 ns |
| `Gf2_128` | 5.62 ns (0.253× GF2E) | 5.61 ns (0.313× GF2E) | 660 ns (0.971× GF2E) | 671 ns (0.936× GF2E) | 316 ns (0.182× GF2E) | `mul_with(b, Clmul::Portable)` 50.7 ns (2.22× GF2E) |

GHASH of one message under a key set once, against RustCrypto's `ghash` (which clones its keyed state for each message) and OpenSSL's AES-GCM (the bytes as additional data on one running message, which is GHASH alone); GB/s are bytes over the median time:

| Message | `ghash` | GB/s | fastest reference | its GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 16 B | 2.89 ns (0.458× ghash) | 5.54 | ghash 6.31 ns | 2.54 |
| 64 B | 7.19 ns (1.04× ghash) | 8.91 | ghash 6.94 ns | 9.22 |
| 256 B | 12.2 ns (0.730× ghash) | 20.93 | ghash 16.8 ns | 15.27 |
| 1 KiB | 47.3 ns (0.588× ghash) | 21.64 | ghash 80.5 ns | 12.72 |
| 16 KiB | 1.00 µs (0.627× OpenSSL) | 16.36 | OpenSSL 1.60 µs | 10.25 |
| 1 MiB | 65.6 µs (0.643× OpenSSL) | 15.99 | OpenSSL 102 µs | 10.28 |

GCM's `GHASH(H, A, C)` of 16 bytes of additional data and 1 KiB of ciphertext, with the block of lengths, takes 56.5 ns (0.800× ghash).

### Normal bases

No reference has Gaussian normal bases. The times of this crate, at the degrees of FIPS 186-4 in the types `NormalBasis::new` chooses; the last column builds the basis and its conversion to the polynomial basis of the standard's modulus:

| `m` | type | `mul` | `square` | `inverse` | `to_normal` | `to_polynomial` | `NormalBasis::new` | `NormalBasis::new + conversion` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 163 | 4 | 9.84 µs | 24.5 ns | 88.6 µs | 375 ns | 323 ns | 8.00 µs | 3.95 ms |
| 233 | 2 | 14.4 µs | 26.4 ns | 145 µs | 576 ns | 762 ns | 6.05 µs | 7.04 ms |
| 283 | 6 | 21.4 µs | 30.0 ns | 234 µs | 812 ns | 977 ns | 21.2 µs | 16.0 ms |
| 409 | 4 | 33.1 µs | 31.9 ns | 369 µs | 1.42 µs | 1.57 µs | 19.9 µs | 36.2 ms |
| 571 | 10 | 55.7 µs | 36.3 ns | 721 µs | 1.63 µs | 1.66 µs | 75.3 µs | 97.6 ms |

A product in the normal basis of 163 bits takes 9.84 µs, against 24.4 ns in the polynomial basis and 375 and 323 ns for `to_normal` and `to_polynomial`.

### Where the references are faster

Every case above 1.05 times the fastest reference that does the same work, by cause:

- **Sums, differences, negations and doublings of `Fp` up to nine limbs** (34 of the 56 cases, 1.08 to 4.59 times the fastest, mostly arkworks's, the fastest taking 0.461 to 3.54 ns): each operation matches the field's form at run time (one of 17), checks that the operands' fields agree and writes an 88-byte element, which costs more than the sum itself; arkworks knows its prime at compile time. `TypedFp`, which does too, ties or wins its differences and doublings everywhere; its negations modulo `2²⁵⁵ − 19` and secp256k1's prime (1.30, 1.26) are a subtraction from 0 in the unreduced form `2^k − c`, and its sums in Montgomery's form (1.28 to 1.58 times arkworks's and FLINT's on Goldilocks, BN254's `p`, BLS12-381's `r`, `2²⁵⁵ − 19` and P-256's `p`) have no established cause: their branch-free assembly carry chains were measured as fast as plain Rust and kept.
- **Products and squares of `Fp`** at one to four limbs (1.06 to 1.23 times arkworks's and blst's: Goldilocks, 128 and 192 bits, BN254's products, BLS12-381's `r`) and at eight and nine (512 and 576 bits, 1.30 to 1.56 times OpenSSL's and arkworks's): at one to four limbs, the per-operation cost of the run-time form, as above (`TypedFp` on the same kernels takes 0.432 to 0.952 of the fastest reference's time); from seven limbs, a product is the numbers' product followed by a separate Montgomery reduction, where arkworks interleaves the two, unrolled for its prime at compile time, and OpenSSL runs hand-written ARMv8 assembly (`bn_mul_mont`, with a squaring of its own for multiples of eight limbs).
- **Products and squares beyond nine limbs** (600 to 4096 bits: products 1.07 to 1.39, squares 1.10 to 1.63, against OpenSSL, arkworks at 768 bits and FLINT's `fmpz_mod` at 4096): the numbers' schoolbook product of the element's vector and a separate REDC, against OpenSSL's hand-written assembly with its squaring, arkworks's unrolled code and, at 4096 bits, FLINT's use of GMP's subquadratic product.
- **Sums, differences, negations and doublings beyond nine limbs** (1.18 to 3.04 times FLINT's `mpn_mod`, GMP's and arkworks's): a sum is a carry loop in Rust over the element's vector, a comparison with `p` and a subtraction, three passes; a negation or a doubling makes a new vector; GMP's loops, under FLINT's `mpn_mod` too, are assembly, and arkworks's are unrolled.
- **Powers and square roots** (`pow` 1.61 and 1.11 and `sqrt` 1.45 and 1.13 at 512 and 576 bits, `pow` 1.26 and 1.21 at 2048 and 4096 against OpenSSL; `pow_u64` 1.11 to 1.43 at 64 to 576 bits against arkworks and 1.11 on BLS12-381's `p` against mcl): they are runs of the products of the second and third items; OpenSSL's powers run on its assembly Montgomery multiplication, mcl's on its generated assembly. Goldilocks's powers (2.31 times FLINT's `nmod` at run time, 1.47 as a type) are runs of the field's products, 4.10 ns each at run time and 2.87 ns as a type, against FLINT's `n_powmod2_preinv` on the word; not investigated further.
- **Goldilocks's inverse and Legendre symbol at run time** (1.06 and 1.17 times FLINT's `nmod`) and the symbol at 128 bits (1.10 times FLINT's `fmpz_mod`): the run-time field converts the value to a `Number` for `Number::mod_inverse` and `Number::jacobi` (the limits above); as a type, on the word, Goldilocks's take 0.934 and 0.834 of `nmod`'s time.
- **`batch_invert` at run time** (1.07 to 1.92 times arkworks's, on nine fields): three of the field's products an element, at the run-time cost above; as types, 0.503 to 0.892.
- **Encodings** (`to_bytes_be` 1.08 to 2.81 times the fastest at run time and 1.08 to 2.14 as types; `from_bytes_be` 1.41 on BLS12-381's `r` at run time, 0.608 as a type): the `bls12_381` crate returns an array (6.72 to 6.74 ns) and blst writes into the caller's bytes, where this API returns a `Vec`, an allocation; against arkworks, whose `to_bytes_be` allocates too, the value is taken out of the form (a reduction, at the product's cost above) through the run-time match.
- **`GF(3⁵)` by Zech's tables** (every operation, 1.42 to 6.71 times FLINT's `fq_zech` and NTL's `zz_pE`; a product 4.97 ns against 1.01 ns): around its one or two look-ups, each operation matches the element's representation, compares the two fields and builds a new `Fpk` (a reference to the field, the logarithm and the cell of its coefficients), where FLINT's `fq_zech` element is a word; the square root (6.71) also chooses the smaller of the two roots, which takes their coefficients (FLINT returns either root).
- **Word primes without tables** (`GF(3¹²)` and `p = 10⁹ + 7`: sums 1.15 to 3.21 times NTL's `zz_pE`; products 1.11 to 2.00, squares 1.34 to 1.82, inverses 1.14 to 1.32 and powers 1.55 to 2.15 times FLINT's `fq_nmod`; norms 2.71 at `k = 2` and 3.83 in `GF(3¹²)`, traces 1.14 and 1.35 at `k = 8` and 2): each operation allocates its result's vector of `k` words, where FLINT and NTL write into an element given; at `k = 64` FLINT multiplies by its `nmod_poly` product (2.00), against the schoolbook on the words here, which measured faster than `ModPolynomial`'s product up to degree 256 (the methods above); norms and traces take the coefficients as elements of `GF(p)`, and the norm beyond degree 2 is `ModPolynomial`'s resultant.
- **Inverses over primes of several limbs** (BN254's and BLS12-381's `p` from degree 2 to 64, and the prime of 600 bits: 1.14 to 1.81 times FLINT's `fq`, and NTL's at `k = 64`): `ModPolynomial::xgcd` on the coefficients as numbers, converted there and back; at `k = 2` (1.79, 1.81) there is no closed form through the norm.
- **BN254's `p` at degree 64** (products 1.38 and powers 1.33 times NTL's `ZZ_pE`): the schoolbook's 4,096 products of `F_p`, below the crossover of 100 for four limbs, against NTL's `ZZ_pX` product; the square (0.942) wins.
- **The prime of 600 bits** (`k = 3` and 8: sums 2.99 and 2.40 times NTL's `ZZ_pE`; products 2.26 and 2.17, squares 2.09 and 2.11, powers 2.24 and 2.15 times FLINT's `fq`; traces 1.43 and 1.34): an element is `k` elements of `GF(p)`, each a vector of its own beyond nine limbs, so that every operation allocates `k + 1` times, and their products are those of the third item.
- **BLS12-381's `F_p¹²` products and squares** (1.23 and 1.20 times mcl's), its cyclotomic square (1.14 times blst's) and the square of its `F_p²` (1.12 times mcl's): the products of `F_p⁶` and `F_p¹²` reduce every product of `F_p²` they take, where mcl and blst keep their sums unreduced to the end, and both are assembly, blst's hand-written and mcl's generated from its LLVM IR.
- **Small operations of the towers** (sums of BN254's `F_p²` and BLS12-381's `F_p¹²`, 1.53 and 1.54; conjugates 1.08 to 2.62; BLS12-381's `mul_by_xi` 1.52 and `mul_by_v` 3.01; against arkworks and mcl): the coefficients are 88-byte `Fp`s, read and written through the field's form (an `F_p¹²` holds 1,056 bytes, mcl's 576), and `mul_by_v` builds a new `F_p⁶` around its product by `ξ`.
- **Squares in `GF(2ᵐ)`** (1.16 to 1.87 times NTL's `GF2E` from 8 to 571 bits, but at 64 and 163 bits and from 1024): `square()` makes a new element, an allocation, and zeroes a working buffer of 32 or 192 words, around one `PMULL` a word and the reduction; NTL squares into the element it is given.
- **Powers in `GF(2ᵐ)`, and the product at 409 bits** (`pow_u64` 1.09 to 1.43 times NTL's at 8, 128, 233, 283, 409, 571 and 1024 bits; the product 1.06 at 409): no established cause; each step zeroes a working buffer of 32 or 192 words and copies its result back, which at one to nine words is not small beside a few `PMULL`s.
- **`Gf2_128::mul_with(b, Clmul::Portable)`** (2.22 times NTL's product, which is portable here too): the portable kernel, for CPUs without `PMULL`, multiplies words by a window of 4 bits (16 look-ups and shifts a word); NTL's is faster, not investigated. By `PMULL`, as by default here, a product takes 0.253 of NTL's time.
- **Not counted, other work that is faster:** blst's constant-time square roots, in BLS12-381's `F_p` (7.50 µs; this crate's take 1.17 times as long, 1.15 as a type) and in its `F_p²` (15.3 µs; 1.25); and RustCrypto's `polyval` at 64 bytes (5.76 ns; 1.25), on blocks reversed before the clock starts.

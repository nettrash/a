[A](../../../README.md) › [Math](../../../README.md#math)

# Elliptic curves

`a::math::ec` computes on elliptic curves over prime fields, with coordinates and scalars that are [numbers](numbers.md), and over binary fields `GF(2ᵐ)`, with coordinates that are elements of a [binary field](field.md#binary-fields); and it hashes messages to curves as RFC 9380 specifies. Over prime fields there are three forms of curve, each built from its coefficients and a prime of any length:

- `Weierstrass::new(&a, &b, &p)`: `y² = x³ + a·x + b`.
- `Montgomery::new(&a, &b, &p)`: `B·y² = x³ + A·x² + x`, with `a` as `A` and `b` as `B`.
- `Edwards::new(&a, &d, &p)`: the twisted Edwards curve `a·x² + y² = 1 + d·x²·y²`, with `a` a square and `d` not, which makes its addition law complete: one formula for every pair of points.

`a::math::ec::named` holds the curves in common use, ready made: `secp256k1()`, `p256()`, `p384()`, `p521()`, `curve25519()` and `ed25519()`.

**Nothing here is constant-time.** These are calculations on curves: the time they take depends on the bits of the scalars and on the coordinates, as with [`pow_mod`](pow.md), and they keep no secrets. They are not meant for secret scalars, such as private keys, where whoever can time them, or watch their caches, could learn them. There are no signatures, key agreements or random numbers here; public data, as in verifying a signature, is what they are safe for.

- A `Point` borrows its curve, and every way to get one checks that it is on the curve, so that an operation can't go wrong by being given one that isn't: `curve.point(&x, &y)`, which gives `Err(NotOnCurve)` otherwise; `curve.identity()`, the neutral element; `curve.lift(&x, odd)`, the point with a given coordinate and the parity of the other, `None` if there is none (on Edwards curves `y` is given and `x` computed); and `curve.point_from_bytes(&bytes)`, `None` if they encode no point of the curve.
- Points add with `+` and `+=`, subtract with `-` and `-=`, negate with `-`, and multiply by a `Number` with `*` and `*=`, on either side, owned or borrowed. `p.double()` costs less than `&p + &p`. Scalars may have any sign and length. `==` compares points whatever form they are held in. Two points of different curves panic, unless the two curves have the same parameters.
- `p.coordinates()` gives the affine `(x, y)`, or `None` for the point at infinity. An Edwards curve has none: its neutral element is `(0, 1)`. Points are held in projective forms, so that the operations take no inversion, and this is the one inversion they leave for the end; `Point::coordinates_of(&points)` shares one among many points.
- `p.compress()` and `p.to_bytes()` encode a point. On Weierstrass and Montgomery curves they give SEC 1's compressed form, a byte for the parity of `y` and `x` in big-endian order, and its uncompressed form, `4`, `x` and `y`, with a single zero byte for the point at infinity. On Edwards curves they give RFC 8032's `y` in little-endian order with the parity of `x` in the top bit, and `x` then `y` in little-endian order.
- `curve.multi_mul(&[(&k, &p), ...])` sums many products, far faster than as many products. `p.prepare(bits)` computes a point's multiples once, for many products by scalars of up to `bits` bits, and `prepared.mul(&k)` takes it, as a [`Divisor`](div.md) does for divisions.
- `curve.discriminant()` and `curve.j_invariant()` give those invariants modulo the prime; two curves with the same `j` are isomorphic over the algebraic closure of the field. The discriminant of a Montgomery curve is `16 (A² − 4) / B⁶`, that of its general Weierstrass equation, and of an Edwards curve `a·d·(a − d)⁴ / 16`, that of its Montgomery form.
- `Domain::new(curve, &x, &y, &order, &cofactor)` gives a curve a generator, as the named curves have: it dereferences to the curve, and adds `generator()`, `order()`, `cofactor()`, `mul_generator(&k)`, `mul_add(&a, &b, &p)` for `a·G + b·P`, `in_subgroup(&p)` and `clear_cofactor(&p)`. It builds a table of the generator's multiples the first time `mul_generator` needs one. The order must be positive. `in_subgroup` tells whether `order·P` is the neutral element, which makes `P` a point of the generator's subgroup when the order is coprime to the cofactor, as on every named curve; otherwise more points can pass.
- `curve.x_mul(&u, &k)` on a Montgomery curve computes `x` of `k·P` from `x` of `P` alone by Montgomery's ladder, as X25519 does, without its clamping of the scalar; `u` may be on the curve's quadratic twist.
- `montgomery.to_weierstrass()`, `montgomery.to_edwards()` and `edwards.to_montgomery()` give the equivalent curve in another form, and `map_to_weierstrass`, `map_to_edwards` and `map_to_montgomery` carry a point over: each map respects the group, and panics for a point of another curve or a target other than the curve its `to_` method gives.
- `new` gives `Err(CurveError)` for a modulus that is not a prime above 3, by [`is_probable_prime`](prime.md), for a singular curve, and for an Edwards curve whose addition law is not complete.
- Numbers come back reduced below the prime, in the byte order the prime was given in; the named curves' are little-endian. Coefficients and coordinates may be given unreduced, in either byte order.

The examples, the method, the measurements and the efficiency below are those of the curves over prime fields. Two parts have sections of their own after them, with what they were verified against and no timings yet: [curves over binary fields](#curves-over-binary-fields), `a::math::ec::binary`, with the 18 curves of SEC 2 in `a::math::ec::binary::named`; and [hashing to curves](#hashing-to-curves), `a::math::ec::hash`, with RFC 9380's suites for the named curves. The hashing to the groups of pairings is on the page of [pairings](pairing.md#hashing-to-g1-and-g2).

## Examples

```rust
use a::math::ec::{named, Curve, Edwards, Point, Weierstrass};
use a::math::numbers::{ByteOrder, Number};

let n = |x| Number::new_from_u32(x, ByteOrder::BigEndian);

// A curve of your own: y^2 = x^3 + 2 x + 3 over F_97.
let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
let p = curve.point(&n(3), &n(6)).unwrap();
assert!(curve.point(&n(3), &n(7)).is_err());

// The group's operations, with operators.
let q = &p + &p;
assert_eq!(q.coordinates(), Some((n(80), n(10))));
assert_eq!(&q - &p, p);
assert_eq!(&p * &n(2), q);
assert_eq!(&p * &n(5), curve.identity());
assert_eq!(-&p, &p * &n(4));

// Bytes.
assert_eq!(p.compress(), [2, 3]);
assert_eq!(curve.point_from_bytes(&[2, 3]).unwrap(), p);
assert_eq!(curve.lift(&n(3), false).unwrap(), p);

// A named curve: two public keys from their scalars, and a shared point.
let curve = named::secp256k1();
let a = Number::new_from_u64(0x1234_5678_9ABC_DEF0, ByteOrder::BigEndian);
let b = Number::new_from_u64(0x0FED_CBA9_8765_4321, ByteOrder::BigEndian);
let (pa, pb) = (curve.mul_generator(&a), curve.mul_generator(&b));
assert_eq!(&pa * &b, &pb * &a);
assert_eq!(pa.compress().len(), 33);

// A signature's check is a sum of two products, faster together.
let g = curve.generator();
assert_eq!(curve.mul_add(&a, &b, &pb), &g * &a + &pb * &b);
assert_eq!(curve.multi_mul(&[(&a, &g), (&b, &pb)]), &g * &a + &pb * &b);

// Curves come in forms: Ed25519's generator is X25519's base point, u = 9.
let edwards = named::ed25519();
let montgomery = edwards.to_montgomery();
let base = edwards.map_to_montgomery(&montgomery, &edwards.generator());
assert_eq!(base.coordinates().unwrap().0, n(9));
assert_eq!(named::curve25519().j_invariant(), edwards.j_invariant());
let _: &Edwards = edwards;
let _: Option<Point<'_, Weierstrass>> = None;
```

The method:

- A prime field is chosen by the shape of the prime when a curve is built, and everything above it is compiled for that field. P-521's `2^521 − 1` is in nine limbs of 58 bits: the products of limbs add up in 128 bits without a carry chain, and each column spreads over three limbs at once, as `2^522 = 2`, while a sum carries every limb into the next at once. A product so takes 71 cycles against 98 in nine limbs of 64 bits, a square 45 against 94, and a doubling of P-521 540 against 670. Any other prime `2^k − c` with a small `c`, as secp256k1's and Curve25519's, folds the high half of a product back onto the low one, a product by `c` shifted, and its elements are any `n` limbs, not always below the prime. Any other prime of up to nine limbs is in Montgomery's form. P-256's and P-384's are too, with reductions of their own: `−1/p mod 2^64` is 1 and `2^32 + 1`, so a round's quotient is a limb or a shift of it, and `q·p` is a few shifts, without a product. Longer primes take numbers and a [`Divisor`](div.md).
- On aarch64 the products, squares, additions and subtractions of 4 and 6 limbs, and the additions and subtractions of 9, are inline assembly, generated by `src/math/ec/field/kernels.py`. Up to six limbs they take and give their limbs in registers and are inlined, so that a formula's elements stay in registers from one operation to the next, and a prime `2^k − c` leaves the rare carry of its fold to a branch around their end. Other targets, and Miri, take portable Rust, as P-521's field does everywhere, which is what makes x86_64's speed unmeasured here.
- Weierstrass and Montgomery curves hold points in Jacobian coordinates, `x = X/Z²`, `y = Y/Z³`. Weierstrass curves have doublings for `a = 0` (3M + 4S) and `a = −3` (4M + 4S, or 3M + 5S where a square costs well under a product, as on P-521), which take `2·Y²` for fewer additions, as an addition costs about a quarter of a product, as well as a general one, additions of 12M + 4S and mixed ones of 8M + 3S, each with its independent products next to each other. A run of doublings holds `2·Y` in place of `Y`, which saves three more additions a doubling for `a = 0` and `a = −3`, where squares cost about as much as products: on P-256 a doubling in a run takes 250 cycles, against 272 alone. Montgomery curves have formulas of their own. Edwards curves hold them in extended coordinates, `x = X/Z`, `y = Y/Z`, `x·y = T/Z`, 8M an addition for `a = −1` and 7M from an affine point, and a run of doublings skips `T` but for the last.
- A product is by the non-adjacent form of width 5 up to 639 bits, whose digits are odd and at least four zeros apart, about 43 additions on 256 bits, over a table of odd multiples put in affine form with one inversion, as a mixed addition saves 5 products of 16. Doublings on secp256k1 are halved by its endomorphism `(x, y) → (β·x, y)`, the product by `λ` on every point: `k = k₁ + k₂·λ` modulo the order, with `k₁` and `k₂` of half its length (Gallant, Lambert and Vanstone). A Montgomery curve takes the ladder, then recovers `y` (Okeya and Sakurai).
- Many products take Straus's method, one run of doublings for all of them, and from 64 terms Pippenger's, which sorts them into buckets for each window of `log₂ n − 1` bits, its points affine on every form of curve, as each is added once a window.
- A prepared point has its multiples by each digit at each position of a signed window of 8 bits, from 192 bits of scalar: about 33 additions for 256 bits and no doublings, at the cost of about 4,200 affine points, 300 KiB on P-256, built once.
- A square root is a power for primes 3 modulo 4 and Tonelli and Shanks's method otherwise. The square root of a ratio, which the lift of an Edwards or a Montgomery curve takes, is one power and no inversion for primes 3 modulo 4 and 5 modulo 8. A power takes the runs of ones of its exponent as products of `x^(2^j − 1)`, which the run at the top builds by doubling, 13 products for a square root modulo Curve25519's prime, and a sliding window of 5 bits otherwise.

Measured on an Apple M4 Max in cycles (the mean of two runs, each the median of nine short ones, on a machine with other work on it), against OpenSSL 3.6's `EC_POINT_mul` and the like, and curve25519-dalek 4.1's `EdwardsPoint` and `MontgomeryPoint`. OpenSSL's generic code serves secp256k1, and its P-256 is assembly for that curve alone; its products of an unknown point, and dalek's, are constant-time, which costs them time this crate doesn't spend. `k·G` is a known generator, with a table where the library has one. `Add` and `Double` go through the public operators, which each allocate a vector here, and take no inversion on either side.

| Curve | Operation | `a` | OpenSSL | dalek |
| --- | --- | ---: | ---: | ---: |
| secp256k1 | `k·P` | 40,100 | 555,000 | |
| | `k·G` | 7,250 | 559,000 | |
| | `a·G + b·P` | 48,100 | 504,000 | |
| | Add | 360 | 840 | |
| | Double | 210 | 960 | |
| | Decompress | 8,700 | 30,500 | |
| P-256 | `k·P` | 82,600 | 90,900 | |
| | `k·G` | 10,600 | 14,300 | |
| | `a·G + b·P` | 94,700 | 104,000 | |
| | Add | 500 | 1,220 | |
| | Double | 320 | 1,010 | |
| | Decompress | 8,600 | 26,400 | |
| P-384 | `k·P` | 311,000 | 548,000 | |
| | `k·G` | 40,200 | 267,000 | |
| | `a·G + b·P` | 351,000 | 735,000 | |
| | Add | 1,270 | 3,890 | |
| | Double | 720 | 2,590 | |
| | Decompress | 31,700 | 78,500 | |
| P-521 | `k·P` | 358,000 | 496,000 | |
| | `k·G` | 52,500 | 269,000 | |
| | `a·G + b·P` | 410,000 | 695,000 | |
| | Add | 1,190 | 5,020 | |
| | Double | 620 | 3,140 | |
| | Decompress | 25,200 | 159,000 | |
| Ed25519 | `k·P` | 45,300 | | 72,000 |
| | `k·G` | 5,350 | | 22,100 |
| | `a·G + b·P` | 51,500 | | 63,400 |
| | Add | 260 | | 250 |
| | Double | 210 | | 250 |
| | Decompress | 8,600 | | 9,400 |
| Curve25519 | X25519 | 55,400 | | 78,500 |

- Ed25519's Add is a little behind dalek's: the public operators allocate a vector for each result, which dalek's points, on the stack, don't.
- On P-256 a doubling inside a product takes about 250 cycles, against 257 for OpenSSL's assembly for that curve.
- Scalars much shorter than the order take much less time: the cost is about the bits of the scalar.

A sum of `n` products, in cycles per term on P-256: 53,200 for 2 terms, 28,100 for 8, 23,000 for 64 and 15,900 for 256, against 82,600 for a product alone. The first `mul_generator` of a curve builds its table: 2.7 million cycles on P-256, 1.5 million on Ed25519 and 14 million on P-521.

## Efficiency

- Time complexity: `k·P` about `b` doublings and `b/6` additions for a scalar of `b` bits, each a fixed number of products of the field; a product of an `n`-limb field O(n²) up to nine limbs, the [multiplication](mul.md)'s and the [division](div.md)'s beyond
- Space complexity: a point holds 3 elements (4 on Edwards curves) in one vector; a product keeps 8 points in a table; a prepared point 300 KiB on P-256, 420 KiB on Ed25519, 1.2 MiB on P-521
- Soundness: the kernels are inline assembly with stated operands; points are on their curves by construction; the additions handle equal and opposite points and the neutral element, so a scalar may be any multiple of a point's order, and every product is correct on points of small order

## Curves over binary fields

`a::math::ec::binary` computes on the curves `y² + x·y = x³ + a·x² + b` over a binary field `GF(2ᵐ)`, `b ≠ 0`, with coordinates that are elements of a [`BinaryField`](field.md#binary-fields) in its polynomial basis, as SEC 1, SEC 2 and FIPS 186 give them. Every ordinary curve over `GF(2ᵐ)` (one whose `j`-invariant is not 0) has an equation of this form; the supersingular ones, `y² + c·y = x³ + …`, are not covered. `a::math::ec::binary::named` holds the 18 curves of SEC 2 that OpenSSL knows, with NIST's names for the ten of FIPS 186.

**Nothing here is constant-time**, as for the curves over prime fields: the time of a product depends on the bits of the scalar and on the coordinates, so these are not meant for secret scalars such as private keys.

| Item | What it is |
| --- | --- |
| `BinaryCurve::new(&a, &b)` | the curve over the field of the elements `a` and `b`; `Err(CurveError::Singular)` for `b = 0`; elements of two different fields panic |
| `curve.field()`, `a()`, `b()`, `discriminant()`, `j_invariant()` | the field, the coefficients, the discriminant `b` and `j = 1/b` |
| `curve.is_koblitz()`, `curve.koblitz_order()` | whether `a ∈ {0, 1}` and `b = 1`; the number of points `2ᵐ + 1 − V_m` of such a curve, `None` for another |
| `curve.identity()`, `curve.point(&x, &y)` | the point at infinity; the point `(x, y)`, or `Err(NotOnCurve)` |
| `curve.lift(&x, odd)` | the point with that `x` whose `ỹ`, the lowest bit of `y/x`, is `odd`; `None` when no point has that `x`; for `x = 0` the point `(0, √b)` with either bit |
| `curve.point_from_bytes(&bytes)` | SEC 1's compressed (`02`/`03`), uncompressed (`04`) or hybrid (`06`/`07`) form, or `00` for the point at infinity; `None` for anything else |
| `p.compress()`, `p.to_bytes()`, `p.to_bytes_hybrid()` | the three forms of SEC 1, each element in `⌈m/8⌉` bytes, big-endian |
| `p.coordinates()`, `BinaryPoint::coordinates_of(&points)` | the affine `(x, y)` as elements of the curve's field, `None` at infinity; one inversion for many points |
| `+ - * += -= *=`, unary `-`, `p.double()`, `==`, `Hash`, `Display` | the group: points owned or borrowed, scalars `Number`s of any sign and length on either side; `−(x, y) = (x, x + y)`; two points of different curves panic |
| `curve.multi_mul(&[(&k, &p), …])` | the sum of many products |
| `curve.x_mul(&x, &k)` | `x` of `k·P` from `x` of `P` alone, `None` when `k·P` is the point at infinity; `x` may be on the quadratic twist |
| `p.prepare(bits)`, `prepared.mul(&k)` | the multiples of a point computed once for many products |
| `BinaryDomain::new(curve, &x, &y, &order, &cofactor)` | a curve with a generator; `Err(NotOnCurve)` or `Err(WrongOrder)` when the generator is off the curve or `order·G` is not the point at infinity. It dereferences to its curve and adds `generator()`, `order()`, `cofactor()`, `mul_generator(&k)` (from a table built on the first call), `mul_add(&a, &b, &p)`, `in_subgroup(&p)`, `clear_cofactor(&p)` |
| `named::sect113r1()` … `named::sect571r1()`, `named::k163()` … `named::b571()`, `named::by_name(name)`, `named::NAMES` | the curves of SEC 2 as `&'static BinaryDomain`s, built on the first call; `by_name` takes SEC 2's names and NIST's (`"K-163"`), in any case |

The curves of `named`, their parameters SEC 2's (version 1.0, 2000):

| SEC 2 | NIST | `m` | Field polynomial | Kind | Cofactor |
| --- | --- | ---: | --- | --- | ---: |
| sect113r1, sect113r2 | | 113 | `x¹¹³ + x⁹ + 1` | random | 2 |
| sect131r1, sect131r2 | | 131 | `x¹³¹ + x⁸ + x³ + x² + 1` | random | 2 |
| sect163k1 | K-163 | 163 | `x¹⁶³ + x⁷ + x⁶ + x³ + 1` | Koblitz, `a = 1` | 2 |
| sect163r1 | | 163 | the same | random | 2 |
| sect163r2 | B-163 | 163 | the same | random, `a = 1` | 2 |
| sect193r1, sect193r2 | | 193 | `x¹⁹³ + x¹⁵ + 1` | random | 2 |
| sect233k1 | K-233 | 233 | `x²³³ + x⁷⁴ + 1` | Koblitz, `a = 0` | 4 |
| sect233r1 | B-233 | 233 | the same | random, `a = 1` | 2 |
| sect239k1 | | 239 | `x²³⁹ + x¹⁵⁸ + 1` | Koblitz, `a = 0` | 4 |
| sect283k1 | K-283 | 283 | `x²⁸³ + x¹² + x⁷ + x⁵ + 1` | Koblitz, `a = 0` | 4 |
| sect283r1 | B-283 | 283 | the same | random, `a = 1` | 2 |
| sect409k1 | K-409 | 409 | `x⁴⁰⁹ + x⁸⁷ + 1` | Koblitz, `a = 0` | 4 |
| sect409r1 | B-409 | 409 | the same | random, `a = 1` | 2 |
| sect571k1 | K-571 | 571 | `x⁵⁷¹ + x¹⁰ + x⁵ + x² + 1` | Koblitz, `a = 0` | 4 |
| sect571r1 | B-571 | 571 | the same | random, `a = 1` | 2 |

### Example

```rust
use a::math::ec::binary::named;
use a::math::ec::{BinaryCurve, BinaryDomain};
use a::math::field::BinaryField;
use a::math::numbers::Number;

// A curve of your own: y² + x·y = x³ + x² + 1 over GF(2⁵) = GF(2)[x] / (x⁵ + x² + 1), a Koblitz
// curve of 22 points, and (x³, x⁴ + x² + x + 1), a point of order 11.
let f = BinaryField::new(&[5, 2, 0]);
let curve = BinaryCurve::new(&f.one(), &f.one()).unwrap();
assert_eq!(curve.koblitz_order().unwrap(), 22u8);
let (x, y) = (f.element(&[0b1000]), f.element(&[0b10111]));
let p = curve.point(&x, &y).unwrap();
assert!((&p * &Number::from(11u8)).is_identity());
assert_eq!(-&p, &p * &Number::from(10u8));
assert_eq!(curve.lift(&x, true).unwrap(), p); // ỹ = lsb(y/x) = 1
let domain = BinaryDomain::new(curve, &x, &y, &Number::from(11u8), &Number::from(2u8)).unwrap();
assert_eq!(domain.mul_generator(&Number::from(3u8)), domain.generator() * Number::from(3u8));

// A named curve: NIST's B-233, and a shared point from two scalars.
let b233 = named::by_name("B-233").unwrap();
let (a, b) = (Number::from(0x1234_5678_9ABC_u64), Number::from(0x0FED_CBA9_u32));
let (pa, pb) = (b233.mul_generator(&a), b233.mul_generator(&b));
assert_eq!(&pa * &b, &pb * &a);
assert_eq!(b233.multi_mul(&[(&a, &pb), (&b, &pa)]), (&pa * &b).double());

// SEC 1's forms: 1 + 30 bytes compressed, 1 + 60 uncompressed and hybrid.
let bytes = pa.compress();
assert_eq!((bytes.len(), pa.to_bytes().len()), (31, 61));
assert_eq!(b233.point_from_bytes(&bytes).unwrap(), pa);
assert_eq!(b233.point_from_bytes(&pa.to_bytes_hybrid()).unwrap(), pa);

// x of a product from x alone, by the ladder.
let (xa, _) = pa.coordinates().unwrap();
assert_eq!(b233.x_mul(&xa, &b), (&pa * &b).coordinates().map(|c| c.0));

// A Koblitz curve takes the τ-adic form; its number of points is known in closed form.
let k283 = named::k283();
assert_eq!(k283.koblitz_order().unwrap(), k283.order() * k283.cofactor());
assert!(k283.in_subgroup(&(k283.generator() * &a)));
```

### Methods

- **Coordinates.** A point is held in López–Dahab coordinates (López and Dahab, *Improved algorithms for elliptic curve arithmetic in GF(2ⁿ)*, SAC 1998): `x = X/Z`, `y = Y/Z²`, `Z = 0` at infinity, so that the group's operations take no inversion; `coordinates()` takes one, and `coordinates_of` one for all its points by Montgomery's trick. In products `M` and squares `S` of the field: a doubling takes 4M + 5S (3M + 5S for `b = 1`), `Z₃ = X₁²·Z₁²`, `X₃ = X₁⁴ + b·Z₁⁴`, `Y₃ = b·Z₁⁴·Z₃ + X₃·(a·Z₃ + Y₁² + b·Z₁⁴)`; the addition of an affine point 8M + 5S (Al-Daboul, Lim, Kim and Lee, as Hankerson, Menezes and Vanstone give it, *Guide to Elliptic Curve Cryptography*, §3.2.3); the addition of two projective points 13M + 5S, from the chord over the common denominator `Z₁·Z₂·(X₁·Z₂ + X₂·Z₁)`; a product by `a` is free for `a ∈ {0, 1}` and one M otherwise. The additions fall back to the doubling for equal points and give the point at infinity for opposite ones, so every pair of points, of any order, adds correctly.
- **Products.** `k·P` takes the width-`w` non-adjacent form (`w` = 2 up to 23 bits of scalar, 3 up to 75, 4 up to 199, 5 up to 639, 6 up to 1,279, 7 up to 3,071 and 8 beyond, as on the curves over prime fields), odd digits at least `w − 1` zeros apart, over the odd multiples `P, 3P, …` put in affine form with one inversion: about `b` doublings and `b/(w + 1)` mixed additions for a scalar of `b` bits. Scalars may have any sign and length; they are not reduced, as the order of a point is not known.
- **Koblitz curves** (`a ∈ {0, 1}`, `b = 1`) take the τ-adic non-adjacent form instead (Solinas, *Efficient arithmetic on Koblitz curves*, Designs, Codes and Cryptography 19, 2000; the *Guide*'s algorithms 3.61 to 3.70). The Frobenius map `τ(x, y) = (x², y²)`, three squares in López–Dahab coordinates, satisfies `τ² − μ·τ + 2 = 0` with `μ = (−1)^(1−a)`, and `τᵐ` is the identity on the points over `GF(2ᵐ)`. The scalar is reduced modulo `τᵐ − 1` in `Z[τ]` by Solinas's rounding, computed exactly in integers, which leaves an element `ρ` with `ρ·P = k·P` for every point of the curve, of norm at most 4/7 of the number of points; `ρ` is written in the width-`w` τ-adic form, digits `±α_u` for the odd `u < 2^(w−1)`, `α_u` the element of least norm congruent to `u` modulo `τʷ`, computed with the same rounding. A product so takes about `m` Frobenius maps and `m/(w + 1)` mixed additions, and no doublings, whatever the length of `k`. The expansion ends for every element of `Z[τ]` at the widths 2 to 8 it can take: `tests/math/ec/data/binary_tnaf_termination.py` bounds the region outside which each step lowers the norm and runs the algorithm from every element of a box holding it.
- **The number of points of a Koblitz curve** is `2ᵐ + 1 − V_m` for the Lucas sequence `V_k(μ, 2)`, the trace of `τᵐ`, from [`Number::lucas_sequence`](sequences.md); the reduction above uses it as the norm of `τᵐ − 1`.
- **Many products** take Straus's method below 64 terms, the expansions read together with one run of doublings, or of Frobenius maps on a Koblitz curve, for all of them; from 64 terms, Pippenger's, the terms sorted by the digits of signed windows of `log₂ n − 1` bits into buckets, each summed by a running sum.
- **A prepared point**, and the generator of a domain, keeps its multiples by each digit of a signed window of 4 bits (3 below 64 bits) at each position, in affine form: a product takes one mixed addition a window and no doubling, about 143 additions for 571 bits, from a table of 1,144 points built once.
- **`x_mul`** is the Montgomery ladder of López and Dahab (*Fast multiplication on elliptic curves over GF(2ᵐ) without precomputation*, CHES 1999), which keeps `x` of `j·P` and `(j + 1)·P` as `X/Z`: the sum `Z = (X₁·Z₂ + X₂·Z₁)²`, `X = x·Z + X₁·Z₂·X₂·Z₁`, and the double `(X⁴ + b·Z⁴ : X²·Z²)`, 6M + 5S a bit. It needs neither `y` nor `a`, so it computes on the quadratic twist as well; both formulas give `Z = 0` exactly for the point at infinity, so no step needs a case of its own.
- **Decompression** (SEC 1, §2.3.4): for `x ≠ 0`, `z² + z = β` with `β = x + a + b/x²`, solved by the field's half-trace for an odd `m` (`z = Σ β^(2^(2i))`, `(m − 1)/2` pairs of squares) and, for an even `m`, by IEEE 1363-2000's method (A.4.7) with an element of trace 1, the first `xⁱ` that has it; there is no point when the trace of `β` is 1, and `z` becomes `z + 1` when its lowest bit is not `ỹ`; then `y = x·z`. For `x = 0` the point is `(0, √b)`, the one point of order 2, with either bit, as in SEC 1 and OpenSSL. The compressed form costs an inversion for `ỹ` besides that of the coordinates; a decompression an inversion and the half-trace.
- **Checks.** Every way to get a point checks it: `point` and the uncompressed and hybrid forms test the equation, the hybrid form also its bit, and a coordinate with bits at or above `m` is refused. `in_subgroup(p)` tests `order·p = O`, which on the named curves, whose orders are primes prime to their cofactors, is membership of the generator's subgroup.

### Verified

- **The parameters.** The 18 curves' field polynomials, `a`, `b`, generators, orders and cofactors were read from the text of SEC 2 version 1.0 and from OpenSSL 4.1's `EC_GROUP_get_curve`, `EC_GROUP_get0_generator`, `EC_GROUP_get0_order` and `EC_GROUP_get0_cofactor`: the two agree on all 126 values, and the crate's constants are OpenSSL's. The tests compare every constant again with OpenSSL's, from the data file. Building a domain checks that the generator is on its curve and that the order times it is the point at infinity. For each curve the tests also prove that the cofactor is right: the order `n` is a probable prime, prime to `h`, `n² > 16·2ᵐ`, and `h·n` lies within Hasse's interval `|N − 2ᵐ − 1| ≤ 2·2^(m/2)`, which then holds only one multiple of `n`, the number of points `N`; on the six Koblitz curves `koblitz_order()` gives `h·n` exactly.
- **OpenSSL's EC2M**, on every named curve (`lib/tests/math/ec/data/binary_openssl.txt`, 48 KB, written by `binary_gen.c`, each product computed by OpenSSL twice, by `EC_POINT_mul` and by a double-and-add of `EC_POINT_dbl` and `EC_POINT_add`, which must agree): for each curve 3 products `k·G` (`k` below the order, of 32 bits, and of twice the order's length plus 5 bits), each by `mul_generator`, by `*` and negated; 3 products `k·P`, `P` in the subgroup and then random points of the curve of any order, the cofactor's part included, with scalars of the order's length and of twice it, and `x_mul` on one of them; 3 sums `P + Q`, `P + P` and `P + (−P)`; the three encodings of a random point, of the generator and of `(0, √b)`; and OpenSSL's verdict, accept or refuse, on 24 encodings: 15 changes of a valid one (a flipped bit of `y`, a bit at `m` in `x` or in `y`, a byte short or long, the first bytes 1, 5, 8, FF, 2 and 3 on the uncompressed length, both hybrid and both compressed forms), a compressed `x` with no point under both first bytes, and `x = 0` compressed with each bit, `00`, `00 00`, the empty string and the hybrid forms `06` and `07` of `(0, √b)`. The crate agrees on all of them: 54 + 54 products, 54 sums, 54 points in three forms each, and 432 verdicts, the decoded points equal to OpenSSL's. A larger sweep, `binary_gen 12` (481 KB, not shipped: 648 products `k·G`, 648 products `k·P`, 648 sums, 450 points in three forms and 3,798 verdicts), agrees in full in release mode (the ignored test `named_curves_match_a_larger_openssl_sweep`). OpenSSL is not a reference without that check: its `EC_POINT_mul` gives wrong products for some points outside the prime-order subgroup on the Koblitz curves of cofactor 4. An audit found 16 in 7,859 products `k·P`, on points of order dividing 4 and on such points plus a point of the subgroup; on sect283k1, for `P = (1, 0)` of order 4 and `k = #E − 1`, it returns `P`, not `−P = (1, 1)`. PARI's `ellmul`, OpenSSL's own `EC_POINT_dbl` and `EC_POINT_add`, and the crate agree on each. This is why `binary_gen.c` computes every product a second time by double-and-add and stops when the two differ, and the test `koblitz_curves_with_cofactor_four_multiply_their_points_of_order_four` checks `k·(1, 0)` on the five such curves by `*`, the ladder and a table.
- **Small fields, exhaustively**, against the textbook's chord-and-tangent law in affine coordinates written in the tests, and the set of points found by trying every `(x, y)`: 55 curves over `GF(2ᵐ)` for `m` = 1 to 8 (odd and even, both Koblitz curves of each field and others with `a` and `b` of several kinds), 4,116 points. Over `GF(2)` to `GF(2⁷)`, `point` accepts exactly the points of the search, Hasse's bound holds, and `koblitz_order` gives the number of points of each of the 14 Koblitz curves. `N·P = O` holds for every point by the textbook's law, by the ladder and by a prepared table. It is not checked by `*`, because on a Koblitz curve `*` reduces `N` modulo `τᵐ − 1` to 0 and computes nothing. Sums, differences, doublings and negations of the pairs of the first 16 points of each curve over `GF(2)` to `GF(2⁷)` (all the points of the 22 curves of 16 points or fewer), on affine and projective inputs. Products of the first 12 points of each curve (5 beyond `GF(2⁵)`) by every scalar from `−N − 2` to `N + 2` for the curves of up to 64 points, and beyond by the scalars about 0, `±N` and `±2N` and 19 others, and by scalars of about 100 and 200 bits, each by `*` (τ-adic on the Koblitz curves, non-adjacent elsewhere), by `multi_mul`, by a prepared table and by `x_mul`. Sums of 0 to 70 products, by Straus's method and Pippenger's. The three encodings of every point, and `lift` and the compressed form of every `x` with each bit (on all the curves up to `GF(2⁶)` and three of each field beyond): a point exactly when the search found one, with the right `ỹ`. Associativity on every triple of points of two curves over `GF(2⁴)`. `x_mul` on every point of the quadratic twist of curves over `GF(2⁵)` and `GF(2⁷)`, against the twist's own products.
- **Group laws on the named curves**, with seeded scalars and points on sect113r1, sect131r2, sect163k1 and sect163r2 (every curve, more rounds, in the ignored `named_curves_obey_the_group_laws_in_a_longer_sweep`): `(a + b)·G = a·G + b·G`, `b·(a·G) = a·(b·G)`, `mul_add` and associativity. On random points of any order:
  - `h·n·P = O`, by the ladder and a prepared table;
  - `in_subgroup(P)` against `n·P = O` by the ladder and by the table;
  - `in_subgroup` true for `h·P` (`clear_cofactor`) and false for `h·P + T`, `T` the point of order 2;
  - `x_mul`, a prepared table and `multi_mul` against `*` on scalars longer than the order;
  - `(h·n + k)·P` by `*` against `k·P` by the table.

  Also Pippenger's method on 64 terms against Straus's, and the point of order 2.
- **The widest windows.**
  - The non-adjacent form takes width 7 from 1,280 bits of scalar and width 8 from 3,072. The tests use sect113r1 and sect163r2, a point of any order, and scalars of 1,279 to 4,000 bits of both signs. Each product by `*` is checked against the ladder and against `*` on the scalar reduced modulo `h·n`, which takes width 4. Straus's method is checked on terms of 3,500 and 1,400 bits.
  - The τ-adic form takes widths 6 to 8 only on Koblitz curves whose numbers of points have 640 bits or more, which no named curve has. The tests use the fields of `x⁶⁴¹ + x¹¹ + 1`, `x¹²⁷⁹ + x²¹⁶ + 1`, `x²²⁸¹ + x⁷¹⁵ + 1` and `x³²¹⁷ + x⁶⁷ + 1`, irreducible by PARI. Their numbers of points have 641 or 642, 1,279 or 1,280, 2,281 or 2,282, and 3,217 or 3,218 bits, so the widths are 6, 6 or 7, 7, and 8.
  - Over `GF(2⁶⁴¹)`, `koblitz_order` equals PARI's `ellcard` for both curves (`data/binary_card641.gp`).
  - The normal tests check widths 6, 7 and 8, on `GF(2⁶⁴¹)` (both curves), `GF(2¹²⁷⁹)` and `GF(2³²¹⁷)` (`a = 0`). Short scalars, and the same scalars plus multiples of `N`, are checked against a prepared table. The points of order 2 and 4 are checked at `k = N − 1`. Over `GF(2⁶⁴¹)` the tests add `N·P = O` and scalars of `m/2`, `m` and `3m` bits, checked against the ladder and a table.
  - The ignored `koblitz_curves_over_large_fields_in_a_longer_sweep` runs all of that, long scalars included, on all four fields and both curves.
- **Edges and errors.** The point at infinity in every operation and encoding, the point `(0, √b)`, scalars 0, ±1, negative and of 200 bits, `b = 0` (`Singular`), a generator off the curve (`NotOnCurve`), orders that are wrong, 0 or negative (`WrongOrder`), elements of other fields and points of other curves (panics), equal curves built twice, hashes of points held in different projective forms.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 37 tests pass.

To run the tests: `cargo test --test math -- ec::binary::`; the larger sweeps: `binary_gen 12 > sweep.txt` and `A_BINARY_SWEEP=sweep.txt cargo test --release --test math -- --ignored ec::binary::`. The data file is written again by `cc … binary_gen.c -lcrypto && ./binary_gen 1`, with OpenSSL's EC2M built in.

### Limits

- Not constant-time, and not for secret scalars: the non-adjacent forms, the τ-adic reduction and the additions all branch on the data.
- Only the polynomial basis, and only curves `y² + x·y = x³ + a·x² + b`: no supersingular curves and no normal-basis coordinates. There is no point counting for curves other than Koblitz curves (SEA or Satoh's method): a domain is given its order and cofactor, and only `order·G = O` is checked.
- The speed is that of the binary field below it, and has not been measured or tuned: every element is a vector, so every operation allocates. The table of a generator is built with general additions. `x_mul` has no `y` recovery; the products of full points take the non-adjacent forms.
- The window widths are first choices, not tuned: those of the non-adjacent forms are the prime-field curves', and the tables' are narrower.

## Hashing to curves

`a::math::ec::hash` hashes byte strings to points of the named curves as RFC 9380 specifies: a message and a domain separation tag (DST) to elements of the curve's prime field, those to points, and the cofactor cleared. Each suite of section 8 for a curve of `a::math::ec::named` is ready made, under the names the RFC gives it; the parts it is made of are public too, for other curves.

| Function | Suites | Hash, `k`, `L` | Map | `h_eff` |
| --- | --- | --- | --- | ---: |
| `hash::p256()` | `P256_XMD:SHA-256_SSWU_RO_`, `P256_XMD:SHA-256_SSWU_NU_` | SHA-256, 128, 48 | simplified SWU, `Z = −10` | 1 |
| `hash::p384()` | `P384_XMD:SHA-384_SSWU_RO_`, `P384_XMD:SHA-384_SSWU_NU_` | SHA-384, 192, 72 | simplified SWU, `Z = −12` | 1 |
| `hash::p521()` | `P521_XMD:SHA-512_SSWU_RO_`, `P521_XMD:SHA-512_SSWU_NU_` | SHA-512, 256, 98 | simplified SWU, `Z = −4` | 1 |
| `hash::secp256k1()` | `secp256k1_XMD:SHA-256_SSWU_RO_`, `secp256k1_XMD:SHA-256_SSWU_NU_` | SHA-256, 128, 48 | simplified SWU to the isogenous `E′: y² = x³ + A′·x + 1771`, `Z = −11`, then the 3-isogeny of appendix E.1 | 1 |
| `hash::curve25519()` | `curve25519_XMD:SHA-512_ELL2_RO_`, `curve25519_XMD:SHA-512_ELL2_NU_` | SHA-512, 128, 48 | Elligator 2, `Z = 2` | 8 |
| `hash::edwards25519()` | `edwards25519_XMD:SHA-512_ELL2_RO_`, `edwards25519_XMD:SHA-512_ELL2_NU_` | SHA-512, 128, 48 | Elligator 2 to curve25519, then RFC 7748's rational map | 8 |

Each gives a `&'static Suite<C>`, built on the first call, whose points are those of the named curve (`Point<'static, Weierstrass>`, `Montgomery` or `Edwards`):

| Item | What it is |
| --- | --- |
| `suite.hash_to_curve(&msg, &dst)` | `hash_to_curve` of section 3, the `_RO_` suite: two elements, two maps, their sum, the cofactor cleared; close to uniform in the prime-order subgroup |
| `suite.encode_to_curve(&msg, &dst)` | `encode_to_curve`, the `_NU_` suite: one element, one map, the cofactor cleared; not uniform |
| `suite.hash_to_field(&msg, &dst, count)`, `suite.map_to_curve(&u)`, `suite.clear_cofactor(&p)` | The three steps they are made of |
| `suite.ro_id()`, `nu_id()`, `domain()`, `field()`, `k()`, `l()`, `h_eff()`, `z()`, `isogeny()` | The suite's IDs and parameters, its `Domain` and its `PrimeField` |
| `hash_to_field::<H>(&field, &msg, &dst, count, k)` | RFC 9380's `hash_to_field` (section 5.2) into any `PrimeField`, with `expand_message_xmd` on `H` |
| `sgn0(&x)` | The "sign" of section 4.1: whether `x` is odd |
| `ShallueVanDeWoestijne::new(&a, &b, &z)`, `.map(&u)`, `ShallueVanDeWoestijne::find_z(&a, &b)` | The map of section 6.6.1 to any `y² = x³ + a·x + b`, `a` or `b` zero included, checked against the four criteria on `Z`, and the search of appendix H.1 |
| `SimplifiedSwu::new(&a, &b, &z)`, `.map(&u)`, `SimplifiedSwu::find_z(&a, &b)` | The map of section 6.6.2 to `y² = x³ + a·x + b`, `a·b ≠ 0`, checked against the four criteria on `Z`, and the search for `Z` of appendix H.2 |
| `Elligator2::new(&j, &k, &z)`, `.map(&u)`, `Elligator2::find_z(&field)` | The map of section 6.7.1 to `k·t² = s³ + j·s² + s`, and the search of appendix H.3 |
| `IsogenyMap::new(&x_num, &x_den, &y_num, &y_den)`, `.map(&x, &y)`, `.degree()`, `.x_num()`, … | A rational map `(x, y) → (x_num/x_den, y·y_num/y_den)` as appendix E writes isogenies, the point at infinity where a denominator vanishes, and its coefficients |

The tag is the application's, and must not be empty (section 3.1 asks for one that names the application, its version and the suite); an empty one panics, as do an element of another field given to a map and a point of another curve given to `clear_cofactor`. The suites for BLS12-381's groups, and a hashing to BN254's `G1` by `ShallueVanDeWoestijne`, are on the page of [pairings](pairing.md#hashing-to-g1-and-g2).

### Example

```rust
use a::math::ec::hash;

// RFC 9380, appendix J.1.1: P256_XMD:SHA-256_SSWU_RO_ of "abc".
let suite = hash::p256();
let dst = b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_RO_";
let p = suite.hash_to_curve(b"abc", dst);
let (x, y) = p.coordinates().unwrap();
assert_eq!(x.to_str_radix(16), "bb8b87485551aa43ed54f009230450b492fead5f1cc91658775dac4a3388a0f");
assert_eq!(y.to_str_radix(16), "5c41b3d0731a27a7b14bc0bf0ccded2d8751f83493404c84a88e71ffd424212e");

// The same by its steps.
let u = suite.hash_to_field(b"abc", dst, 2);
let q = suite.map_to_curve(&u[0]) + suite.map_to_curve(&u[1]);
assert_eq!(suite.clear_cofactor(&q), p);

// edwards25519, whose cofactor 8 is cleared: the point is in the prime-order subgroup.
let e = hash::edwards25519();
let r = e.hash_to_curve(b"a message", b"MY-APP-V01-CS01-with-edwards25519_XMD:SHA-512_ELL2_RO_");
assert!(e.domain().in_subgroup(&r));
assert_eq!(e.nu_id(), "edwards25519_XMD:SHA-512_ELL2_NU_");
```

### Method

- **`hash_to_field`** (section 5.2) asks [`expand_message_xmd`](../hash/sha2.md) for `count · L` bytes and reduces each piece of `L = ⌈(⌈log₂ p⌉ + k)/8⌉` bytes, read in big-endian order, modulo `p`: the `k` bits beyond the length of `p` keep the bias of the reduction below `2⁻ᵏ`. It is the RFC's function for `m = 1`, a prime field.
- **The Shallue–van de Woestijne map** (section 6.6.1; Shallue and van de Woestijne, ANTS 2006, in Wahby's parameterisation of 2019) takes any short Weierstrass curve: with `t = u²·g(Z)`, `x₁,₂ = −Z/2 ∓ c₃·u·(1 − t)/((1 − t)(1 + t))` and `x₃ = Z + c₄·((1 + t)²/((1 − t)(1 + t)))²`, `c₃ = √(−g(Z)(3Z² + 4A))` of `sgn0(c₃) = 0` and `c₄ = −4g(Z)/(3Z² + 4A)`, inverses by `inv0`; `x` is the first whose `g` is a square, `y = √g(x)` with `sgn0(y) = sgn0(u)`. `new` checks the four criteria of section 6.6.1 on `Z` (`g(Z) ≠ 0`, `h(Z) = −(3Z² + 4A)/(4g(Z))` a nonzero square, `g(Z)` or `g(−Z/2)` a square); `find_z` is appendix H.1's search, 1, −1, 2, −2, …. No suite of the RFC uses it on these curves; [BN254](pairing.md#hashing-to-g1-and-g2) does.
- **The simplified SWU map** (section 6.6.2; Brier, Coron, Icart, Madore, Randriam and Tibouchi, CRYPTO 2010, as Wahby and Boneh, CHES 2019, generalise it): `x₁ = (−B/A)·(1 + 1/(Z²u⁴ + Z·u²))`, or `B/(Z·A)` where the denominator is 0 (at `u = 0` and, where `−1/Z` is a square, at `u = ±√(−1/Z)`); `x = x₁` if `g(x₁) = x₁³ + A·x₁ + B` is a square, otherwise `x₂ = Z·u²·x₁`, for which `g(x₂) = Z³u⁶·g(x₁)` is one; `y = √g(x)` with `sgn0(y) = sgn0(u)`. `SimplifiedSwu::new` checks that the curve is not singular and the four criteria of section 6.6.2 on `Z`: not a square, not −1, `g(x) − Z` irreducible (by `ModPolynomial::is_irreducible` of [polynomials](poly.md)) and `g(B/(Z·A))` a square; `find_z` checks the curve too and tries 1, −1, 2, −2, … as appendix H.2 does.
- **secp256k1** has `A = 0`, which the map cannot take: it maps to the curve `E′` of section 8.7 and on by the 3-isogeny of appendix E.1, `x = x_num(x′)/x_den(x′)` and `y = y′·y_num(x′)/y_den(x′)` (section 6.6.3; Wahby and Boneh), with one inversion for both denominators. The point at infinity comes where a denominator is 0, as the RFC requires; on `E′` that cannot happen for a point over `F_p`: `x_den = (x − x₀)²`, and `g′(x₀)` is not a square, so the kernel's points of order 3 are not rational.
- **Elligator 2** (section 6.7.1; Bernstein, Hamburg, Krasnova and Lange, CCS 2013): `x₁ = −(J/K)/(1 + Z·u²)`, or `−J/K` where that denominator is 0; `x = x₁` with `sgn0(y) = 1` if `g(x₁) = x₁³ + (J/K)·x₁² + x₁/K²` is a square, otherwise `x₂ = −x₁ − J/K` with `sgn0(y) = 0`; the point is `(x·K, y·K)`. `Elligator2::new` checks that `J`, `K` and `Z` are of one field, that `(J² − 4)/K²` is neither 0 nor a square and that `Z` is not a square.
- **edwards25519** takes curve25519's point `(s, t)` to `(c₁·s/t, (s − 1)/(s + 1))`, RFC 7748's map, with `c₁ = √(−486664)` of `sgn0(c₁) = 0` (section 8.5 and appendix G.2.2), and to the identity `(0, 1)` where `t = 0` or `s = −1`. That sign takes RFC 7748's base point `(9, V)`, with the `V` of its erratum 4730, to edwards25519's base point. The generator of `named::curve25519()` has the `V` printed in RFC 7748 before that erratum, `−V`: it goes to the negative of edwards25519's base point.
- **Clearing the cofactor** (section 7) multiplies by `h_eff`, which is the cofactor for these curves: nothing on the NIST curves and secp256k1, 8 on curve25519 and edwards25519.
- **Cost.** A map takes an inversion, one or two square roots (one of which may find a non-square) and a few products in `F_p`; secp256k1's adds its isogeny, three polynomials of degree 3 and one of degree 2 by Horner's rule and an inversion, and edwards25519's an inversion; each point is then built from its affine coordinates by `Curve::point`, which checks the equation. `hash_to_curve` takes two maps, an addition and the cofactor's product; `encode_to_curve` one map and the product. The field arithmetic is that of `a::math::field::PrimeField`.

### Verified

- **RFC 9380, appendix J**: every vector of the twelve suites, 60 (5 messages each: empty, "abc", "abcdef0123456789", and the 133- and 517-byte "q128_…" and "a512_…"). For the `_RO_` suites the two elements `u`, both points `Q₀` and `Q₁` of the map and the result `P`; for the `_NU_` suites `u`, `Q` and `P`; `P` equal to the cofactor cleared from `Q₀ + Q₁` (or from `Q`), on the curve and in the prime-order subgroup (`order · P` the identity). All equal. The vectors are in `lib/tests/math/ec/data/h2c_vectors.txt`, extracted from the RFC's text by `gen_h2c_vectors.py` beside them.
- **The parameters** of section 8 for each suite: the IDs, `k`, `L`, `h_eff`, `Z` and the curve. `find_z` gives −10, −12 and −4 on P-256, P-384 and P-521 and −11 on secp256k1's `E′` (appendix H.2), and `Elligator2::find_z` 2 on curve25519's field (H.3).
- **RustCrypto's `p256` 0.13.2 and `p384` 0.13.1** (`hash2curve`, an implementation of its own) in a probe outside the crate: 5,000 random messages of up to 200 bytes with random tags of 1 to 60 and 300 bytes, `hash_to_curve` and `encode_to_curve` on both curves, 20,000 points, all equal; and `hash_to_field::<Sha256>` into P-256's scalar field against `p256`'s `hash_to_scalar` on 200 messages, all equal.
- **The maps on small fields in full**: for the primes 103, 107, 101, 109, 97 and 113 (3 and 7 modulo 8, 5 modulo 8, 1 modulo 8, so that every method of square root is used), three random curves each and every `u` of the field, the simplified SWU map gives a point of `y² = x³ + A·x + B` whose `x` is `x₁` or `x₂` as the definition chooses, computed again in the test, with `sgn0(y) = sgn0(u)` and `−u` mapped to `(x, −y)`; Elligator 2 gives a point of `K·t² = s³ + J·s² + s` with `x` and the sign of `y` as the definition chooses, the same point for `−u`, and exactly two exceptional `u` with `1 + Z·u² = 0` where `p ≡ 3 (mod 4)` and none otherwise.
- **The Shallue–van de Woestijne map on small fields in full**: on the same six primes, four random curves each, the first with `A = 0`, every `u`: a point of the curve whose `x` is the first of `x₁`, `x₂`, `x₃`, computed again in the test, with a square `g`, and `sgn0(y) = sgn0(u)`; and `find_z`'s `Z` with every candidate before it refused by `new` and itself accepted, as the four criteria evaluated by brute force (every square of the field listed) say. A `Z` with `g(Z) = 0`, a singular curve and the field of 3 are refused.
- **The exceptional cases**: on P-256, P-384, P-521 and `E′`, `u = 0` and `u = ±√(−1/Z)` (a square on each) give `x = B/(Z·A)` with `sgn0(y) = sgn0(u)`, by the bare map and through each of the four suites, whose point is `(B/(Z·A), y)` with the sign of `u`, or, on secp256k1, its image by the isogeny, in the subgroup; on curve25519 `u = 0` gives `(0, 0)`, of order 2, which the cofactor clears to the identity, and on edwards25519 the identity `(0, 1)`. The isogeny's map gives the point at infinity at `x₀`, the root of `x_den`, where `g′(x₀)` is checked not to be a square.
- **The cofactor of 8**: the 8-torsion parts `ℓ·Q` of uncleared map outputs `Q` on curve25519 and edwards25519, drawn until there are points of each order 2, 4 and 8, go to the identity, and each added to the generator `G` to `8·G`.
- **The isogeny**: on 6 random points of `E′`, built with `Weierstrass::new` from `A′` and 1771, the image of every sum of two is the sum of the images, and the image of a double, of a negative, the double and the negative of the image, each on secp256k1 and in its group; and `hash_to_curve` equals the isogeny of the sum on `E′` (the order of section 6.6.3's saving) for three of the RFC's messages.
- **The rational map**: RFC 7748's base point `(9, V)` of erratum 4730 goes to edwards25519's base point under the map with `sgn0(c₁) = 0`, computed in the test on numbers; and edwards25519's `map_to_curve` is that map of curve25519's on 20 random elements.
- **Random messages**: on each suite, points in the subgroup, not the identity, the same for the same input, different for another tag, and `encode_to_curve` different from `hash_to_curve`.
- **The panics**: a square `Z`, `Z = −1`, a `Z` for which `g(x) − Z` has a root (each one found on `F₁₀₃` refused, the others accepted), `A = 0`, `B = 0`, a singular curve (by `new` and by `find_z`), the field of 3 for the simplified SWU map; a square `Z`, `J² − 4 = 0`, a nonzero square `(J² − 4)/K²` (`J` = 3, 5 and 7 with `K = 3` on `F₁₀₁`, and `J = 4` accepted), `K = 0` and a `Z` of another field for Elligator 2, and the field of 2 for its `find_z`; an isogeny's map without coefficients or with a leading 0; an element of another field, a point of another curve, an empty tag; 171 elements of P-256 (255 blocks of SHA-256 hold 170) and `usize::MAX` of them.
- An independent implementation of sections 4 to 8, written for an audit, agreed on 6,000 cases of the suites, 1,000 a curve, and on 1,850 special inputs of the maps.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 40 tests pass.

To run the tests: `cargo test --test math -- ec::hash`. The data file is written again by `python3 gen_h2c_vectors.py rfc9380.txt > h2c_vectors.txt` from the RFC's text (https://www.rfc-editor.org/rfc/rfc9380.txt).

### Limits

- **Not constant-time**, unlike what RFC 9380 recommends for its functions (section 4): the maps branch on whether an element is a square and on the exceptional cases, and the field's arithmetic on its data. For public messages, not for hashing secrets.
- Prime fields only (`m = 1`) here; `F_p²`, for BLS12-381's `G2`, is on the page of [pairings](pairing.md#hashing-to-g1-and-g2). RFC 9380's suites for curve448 and edwards448 need `expand_message_xof` with SHAKE256, and neither those curves nor SHAKE are in the crate.
- The maps are the plain ones of section 6, not the straight-line forms of appendices F and G, which compute the same points with fewer inversions and no branches; on secp256k1 the isogeny is applied to each of the two points and not once to their sum, the saving section 6.6.3 allows. Nothing is timed yet.
- A `Suite` is made only for the named curves. For another curve the parts are public — `hash_to_field`, `ShallueVanDeWoestijne`, `SimplifiedSwu`, `Elligator2`, `IsogenyMap`, `Curve::point` and `Domain::clear_cofactor` — but there is no constructor of a `Suite` and no `h_eff` other than the cofactor.
- On curve25519 no `u` gives `s = −1`, so the branch of the Edwards map for it, which the RFC requires, is reached by no test.

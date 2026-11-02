[A](../../../README.md) › [Math](../../../README.md#math)

# Elliptic curves

`a::math::ec` computes on elliptic curves over prime fields, with coordinates and scalars that are [numbers](numbers.md). There are three forms of curve, each built from its coefficients and a prime of any length:

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
- `Domain::new(curve, &x, &y, &order, &cofactor)` gives a curve a generator, as the named curves have: it dereferences to the curve, and adds `generator()`, `order()`, `cofactor()`, `mul_generator(&k)`, `mul_add(&a, &b, &p)` for `a·G + b·P`, `in_subgroup(&p)` and `clear_cofactor(&p)`. It builds a table of the generator's multiples the first time `mul_generator` needs one.
- `curve.x_mul(&u, &k)` on a Montgomery curve computes `x` of `k·P` from `x` of `P` alone by Montgomery's ladder, as X25519 does, without its clamping of the scalar; `u` may be on the curve's quadratic twist.
- `montgomery.to_weierstrass()`, `montgomery.to_edwards()` and `edwards.to_montgomery()` give the equivalent curve in another form, and `map_to_weierstrass`, `map_to_edwards` and `map_to_montgomery` carry a point over: each map respects the group.
- `new` gives `Err(CurveError)` for a modulus that is not a prime above 3, by [`is_probable_prime`](prime.md), for a singular curve, and for an Edwards curve whose addition law is not complete.
- Numbers come back reduced below the prime, in the byte order the prime was given in; the named curves' are little-endian. Coefficients and coordinates may be given unreduced, in either byte order.

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

- A prime field is chosen by the shape of the prime when a curve is built, and everything above it is compiled for that field. A prime `2^k − c` with a small `c`, as secp256k1's, Curve25519's and P-521's, folds the high half of a product back onto the low one, a product by `c` shifted, or shifts alone for `2^521 − 1`, and its elements are any `n` limbs, not always below the prime. Any other prime of up to nine limbs is in Montgomery's form. P-256's and P-384's are too, with reductions of their own: `−1/p mod 2^64` is 1 and `2^32 + 1`, so a round's quotient is a limb or a shift of it, and `q·p` is a few shifts, without a product. Longer primes take numbers and a [`Divisor`](div.md).
- On aarch64 the products, squares, additions and subtractions of 4 and 6 limbs, the additions and subtractions of 9, and P-521's products and squares are inline assembly, generated by `src/math/ec/field/kernels.py`. Up to six limbs they take and give their limbs in registers and are inlined, so that a formula's elements stay in registers from one operation to the next, and a prime `2^k − c` leaves the rare carry of its fold to a branch around their end. Other targets, and Miri, take portable Rust, which is what makes x86_64's speed unmeasured here.
- Weierstrass and Montgomery curves hold points in Jacobian coordinates, `x = X/Z²`, `y = Y/Z³`. Weierstrass curves have doublings for `a = 0` (3M + 4S) and `a = −3` (4M + 4S), which take `2·Y²` for fewer additions, as an addition costs about a quarter of a product, as well as a general one, additions of 12M + 4S and mixed ones of 8M + 3S, each with its independent products next to each other; Montgomery curves have formulas of their own. Edwards curves hold them in extended coordinates, `x = X/Z`, `y = Y/Z`, `x·y = T/Z`, 8M an addition for `a = −1`, and a run of doublings skips `T` but for the last.
- A product is by the non-adjacent form of width 5 up to 639 bits, whose digits are odd and at least four zeros apart, about 43 additions on 256 bits, over a table of odd multiples put in affine form with one inversion, as a mixed addition saves 5 products of 16. Doublings on secp256k1 are halved by its endomorphism `(x, y) → (β·x, y)`, the product by `λ` on every point: `k = k₁ + k₂·λ` modulo the order, with `k₁` and `k₂` of half its length (Gallant, Lambert and Vanstone). A Montgomery curve takes the ladder, then recovers `y` (Okeya and Sakurai).
- Many products take Straus's method, one run of doublings for all of them, and from 64 terms Pippenger's, which sorts them into buckets for each window of `log₂ n − 1` bits.
- A prepared point has its multiples by each digit at each position of a signed window of 8 bits, from 192 bits of scalar: about 33 additions for 256 bits and no doublings, at the cost of about 4,200 cached points, 300 KiB on P-256, built once.
- A square root is a power for primes 3 modulo 4 and Tonelli and Shanks's method otherwise. The square root of a ratio, which the lift of an Edwards or a Montgomery curve takes, is one power and no inversion for primes 3 modulo 4 and 5 modulo 8. A power takes the runs of ones of its exponent as products of `x^(2^j − 1)`, which the run at the top builds by doubling, 13 products for a square root modulo Curve25519's prime, and a sliding window of 5 bits otherwise.

Measured on an Apple M4 Max in cycles (the mean of two runs, each the median of nine short ones, on a machine with other work on it), against OpenSSL 3.6's `EC_POINT_mul` and the like, and curve25519-dalek 4.1's `EdwardsPoint` and `MontgomeryPoint`. OpenSSL's generic code serves secp256k1, and its P-256 is assembly for that curve alone; its products of an unknown point, and dalek's, are constant-time, which costs them time this crate doesn't spend. `k·G` is a known generator, with a table where the library has one. `Add` and `Double` go through the public operators, which each allocate a vector here, and take no inversion on either side.

| Curve | Operation | `a` | OpenSSL | dalek |
| --- | --- | ---: | ---: | ---: |
| secp256k1 | `k·P` | 40,700 | 574,000 | |
| | `k·G` | 7,680 | 579,000 | |
| | `a·G + b·P` | 50,100 | 519,000 | |
| | Add | 390 | 870 | |
| | Double | 210 | 990 | |
| | Decompress | 8,500 | 32,300 | |
| P-256 | `k·P` | 91,800 | 94,900 | |
| | `k·G` | 11,000 | 15,200 | |
| | `a·G + b·P` | 104,000 | 110,000 | |
| | Add | 510 | 1,280 | |
| | Double | 340 | 1,030 | |
| | Decompress | 8,700 | 27,600 | |
| P-384 | `k·P` | 337,000 | 567,000 | |
| | `k·G` | 41,300 | 282,000 | |
| | `a·G + b·P` | 381,000 | 761,000 | |
| | Add | 1,320 | 4,010 | |
| | Double | 740 | 2,670 | |
| | Decompress | 31,900 | 82,300 | |
| P-521 | `k·P` | 469,000 | 502,000 | |
| | `k·G` | 61,400 | 279,000 | |
| | `a·G + b·P` | 543,000 | 724,000 | |
| | Add | 1,360 | 5,210 | |
| | Double | 750 | 3,240 | |
| | Decompress | 50,600 | 161,000 | |
| Ed25519 | `k·P` | 47,300 | | 73,700 |
| | `k·G` | 6,240 | | 24,000 |
| | `a·G + b·P` | 53,500 | | 67,300 |
| | Add | 270 | | 270 |
| | Double | 220 | | 260 |
| | Decompress | 8,800 | | 9,000 |
| Curve25519 | X25519 | 59,000 | | 80,000 |

- Ed25519's Add is level with dalek's: the public operators allocate a vector for each result, which dalek's points, on the stack, don't.
- On P-256 a doubling inside a product, about 280 cycles, is still behind the 257 of OpenSSL's assembly for that curve, which runs more instructions at a time; the product is ahead by its fewer additions.
- Scalars much shorter than the order take much less time: the cost is about the bits of the scalar.

A sum of `n` products, in cycles per term on P-256: 56,300 for 2 terms, 29,500 for 8, 24,100 for 64 and 16,500 for 256, against 91,800 for a product alone. The first `mul_generator` of a curve builds its table: 2.8 million cycles on P-256, 16 million on P-521.

## Efficiency

- Time complexity: `k·P` about `b` doublings and `b/6` additions for a scalar of `b` bits, each a fixed number of products of the field; a product of an `n`-limb field O(n²) up to nine limbs, the [multiplication](mul.md)'s and the [division](div.md)'s beyond
- Space complexity: a point holds 3 elements (4 on Edwards curves) in one vector; a product keeps 8 points in a table; a prepared point 300 KiB on P-256, 560 KiB on Ed25519, 1.3 MiB on P-521
- Soundness: the kernels are inline assembly with stated operands; points are on their curves by construction; the additions handle equal and opposite points and the neutral element, so a scalar may be any multiple of a point's order, and every product is correct on points of small order

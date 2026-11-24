[A](../../../README.md) › [Math](../../../README.md#math)

# Quaternions

`Quaternion<T>` (in `a::math::quaternion`) is the quaternion `w + x·i + y·j + z·k`, its four parts public fields of the float type `T`, `f32` or `f64` (the sealed trait `Float` of [`a::math::complex`](complex.md)), with Hamilton's rules `i² = j² = k² = ijk = −1`. One type serves the algebra and the rotations: a unit quaternion is a rotation of three-dimensional space, and the methods that read a quaternion as a rotation take it as `q/|q|`, whatever its norm. Nothing here is constant-time.

- `Quaternion::new(w, x, y, z)`, `from_parts(w, [x, y, z])`, `From<T>` (a real quaternion), `From<Complex<T>>` (`re + im·i`), `From<[T; 4]>` and `[T; 4]: From<Quaternion<T>>` (the scalar first, `[w, x, y, z]`); `scalar()`, `vector()`, `to_complex()` (`Some` when the parts of `j` and `k` are zeros). `ZERO`, `ONE`, `I`, `J` and `K` are constants; `Default` is zero.
- They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` (Hamilton's product, not commutative) and divide with `/` and `/=` on the right, `a / b = a·b⁻¹`, each operand owned or borrowed, with a float of their own type on either side: `q * 2.0`, `2.0 * q`, `1.0 / q = q⁻¹`. `-q` negates, `==` compares, `Sum` and `Product` take iterators of owned or borrowed quaternions, the product in the iterator's order. `Display` writes `1-2i+3j+4.5k`, each part with the precision given.
- `conj`, `norm_sqr`, `norm` (the length, without overflow or underflow in its squares), `normalize`, `recip` (the inverse), `dot`, `is_finite`, `is_nan`.
- `exp`, `ln`, `sqrt`, `powi` (integer powers), `powf` (real powers) and `powq` (quaternion powers, `e^(ln(q)·p)`) are the principal values. On the complex numbers `w + x·i` they and the arithmetic agree with `Complex`'s to rounding, with the same branch cuts, wherever the operands and the results are finite; at infinities and NaNs, and where a result overflows, some special values differ (listed under the algebra's methods).
- A product of finite quaternions that overflows keeps the parts the plain formula gives finite and takes the others from scaled operands, so that every part is within a few units of `ε` times the sum of the absolute values of its four products, as in any product, and those that overflow by more are `±∞`, not the NaNs of `∞ − ∞` that the plain formula makes. A power that overflows is formed again from the scaled base, accurate normwise, with no NaN part.
- Rotations: `from_axis_angle`, `to_axis_angle`, `angle`, `from_rotation_vector`, `to_rotation_vector`, `rotate` and `rotate_inverse` (vectors are `[T; 3]`), `to_rotation_matrix` and `from_rotation_matrix` (`[[T; 3]; 3]`, rows), `to_matrix` and `from_matrix` (a 3 × 3 `Matrix<T>` of [`a::math::matrix`](matrix.md)), `slerp`, `nlerp`, `angle_to` and `from_rotation_arc`.
- Euler angles: `from_euler(sequence, frame, [θ₁, θ₂, θ₃])` and `to_euler(sequence, frame)` in the twelve sequences of `EulerSequence` (`XYZ`, `XZY`, `YXZ`, `YZX`, `ZXY`, `ZYX`, `XYX`, `XZX`, `YXY`, `YZY`, `ZXZ`, `ZYZ`), each `EulerFrame::Intrinsic` or `EulerFrame::Extrinsic`: SciPy's 24 conventions, with its ranges and its choice at gimbal lock.

## Examples

```rust
use a::math::quaternion::{EulerFrame, EulerSequence, Quaternion};
use std::f64::consts::FRAC_PI_2;

// Hamilton's rules, the inverse and the quotient on the right.
let (i, j, k) = (Quaternion::<f64>::I, Quaternion::J, Quaternion::K);
assert_eq!(i * j, k);
assert_eq!(j * i, -k);
let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
assert_eq!(q.conj(), Quaternion::new(1.0, -2.0, -3.0, -4.0));
assert!((q * q.recip() - Quaternion::ONE).norm() < 1e-15);
assert!((q / q - Quaternion::ONE).norm() < 1e-15);
assert_eq!(format!("{q}"), "1+2i+3j+4k");

// e^(π·u) = −1 for a unit vector u, and the logarithm back on the principal branch.
let p = Quaternion::new(0.0, 0.0, std::f64::consts::PI, 0.0);
assert!((p.exp() + Quaternion::ONE).norm() < 1e-15);
let r = Quaternion::new(0.5, 1.0, -0.5, 0.25);
assert!((r.exp().ln() - r).norm() < 1e-15);
assert!((r.sqrt() * r.sqrt() - r).norm() < 1e-15);

// A quarter turn about z takes x to y; the product composes, the right factor first.
let rz = Quaternion::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
let rx = Quaternion::from_axis_angle([1.0, 0.0, 0.0], FRAC_PI_2);
let v = (rx * rz).rotate([1.0, 0.0, 0.0]);
assert!(v[0].abs() < 1e-15 && v[1].abs() < 1e-15 && (v[2] - 1.0).abs() < 1e-15);

// Euler angles as SciPy's from_euler("ZYX", …) and as_euler("ZYX").
let e = Quaternion::<f64>::from_euler(EulerSequence::ZYX, EulerFrame::Intrinsic, [0.3, -0.2, 1.1]);
let a = e.to_euler(EulerSequence::ZYX, EulerFrame::Intrinsic);
assert!((a[0] - 0.3).abs() < 1e-15 && (a[1] + 0.2).abs() < 1e-15 && (a[2] - 1.1).abs() < 1e-15);

// Halfway along the shorter arc, at a constant angular speed.
let m = Quaternion::ONE.slerp(rz, 0.5);
assert!((m.angle() - FRAC_PI_2 / 2.0).abs() < 1e-15);
```

The sections below give the conventions, each part's functions, their methods and sources, what they were verified against, and their limits. There are no timings here yet: they come with the release.

## Conventions

- **Hamilton's quaternions**, `ij = k`, `jk = i`, `ki = j`, the scalar part first in the fields and in `[T; 4]`. SciPy, glam and nalgebra store `[x, y, z, w]`, the scalar last.
- **Active rotations**: the rotation of `q` takes a vector `v`, read as the quaternion `0 + v`, to `q·v·q⁻¹`. For `q = cos(θ/2) + u·sin(θ/2)` that is the rotation by `θ` about the unit axis `u`, counterclockwise when `u` points at the viewer (the right-hand rule). `q` and `−q` are the same rotation: the double cover.
- **Composition**: `(a·b).rotate(v) = a.rotate(b.rotate(v))`, the right factor first; the matrix of `a·b` is that of `a` times that of `b`.
- **Norms**: a quaternion read as a rotation is read as `q/|q|`, and the formulas are written so that the norm cancels (`rotate`, the matrix, the axis, the angles), or the quaternion is normalised first (`slerp`, `nlerp`). Zero, and a quaternion with a part that is infinite or NaN, are no rotation: every such method gives NaN.
- **Matrices**: `[[T; 3]; 3]` holds the rows, `R[i][j]` the element in row `i` and column `j`, and `R·v` is `rotate(v)`; `to_matrix` gives the same as a 3 × 3 `Matrix<T>`, and `from_matrix` panics on any other shape.
- **Euler angles**, as SciPy's `Rotation.from_euler` and `as_euler`: the angles in radians, in the order of the sequence. Intrinsic (SciPy's upper-case strings, `"XYZ"`) turns about the axes of the body as the rotations before have left them, `q = q₁·q₂·q₃`; extrinsic (lower-case, `"xyz"`) about the fixed axes, `q = q₃·q₂·q₁`, where `qᵢ` is the rotation by `θᵢ` about the `i`-th axis of the sequence. So intrinsic `XYZ` with `[a, b, c]` is extrinsic `ZYX` with `[c, b, a]`. `to_euler` gives the first and third angles in `[−π, π]` and the second in `[−π/2, π/2]` for the Tait–Bryan sequences (three different axes) and in `[0, π]` for the proper ones (the first axis again last). At gimbal lock, when the second angle is within `10⁻⁷` of the value where the first and third axes coincide (SciPy's tolerance, for both types), only their sum or difference is determined: the third angle is 0 and the first carries the rotation, as SciPy sets them.

## The algebra

| Function | Method |
| --- | --- |
| `a * b` | Hamilton's product, `(w₁w₂ − v₁·v₂) + (w₁v₂ + w₂v₁ + v₁ × v₂)`, four products a part; where finite operands give a part that is not finite, that part again from the operands scaled exactly by powers of two, scaled back once, the finite parts kept |
| `norm` | `√(w² + x² + y² + z²)` where the squared norm is in `[MIN/ε, ε/MIN]`; elsewhere the parts scaled exactly by the power of two that brings the largest into `[1, 2)`, and the root scaled back |
| `normalize` | `q/|q|`, scaled first in the same way; NaN for zero and for non-finite parts |
| `recip`, `a / b` | `conj(b)/|b|²` and `a·conj(b)/|b|²`; where a squared norm is outside the range above, each operand is scaled by a power of two, exactly, and the quotient of the scaled operands (parts below 4) is scaled back by the difference of the powers, rounded once |
| `exp` | `e^w·(cos r + u·sin r)` for the vector part `v = r·u`, its vector part `v·(sin r)/r` |
| `ln` | `ln|q| + u·atan2(r, w)`, the angle in `[0, π]` |
| `sqrt` | Kahan's: `a = √((|q| + w)/2)` and `a + v/(2a)` for `w ≥ 0`; `b = √((|q| − w)/2)` and `r/(2b) + u·b` otherwise |
| `powi(n)` | repeated squaring from the base (not from 1), of the reciprocal for `n < 0`; where a product of the powers overflows and a later one would meet its infinities, again with a mantissa and an exponent of its own |
| `powf(t)`, `powq(p)` | `e^(t·ln q)` and `e^(ln(q)·p)` |

### Methods

- A quaternion `w + v` with `v = r·u`, `r = |v|`, multiplies like the complex number `w + r·i` with the unit vector `u` in the place of `i`, since `u² = −1`; each function is that complex number's, with `u` for `i` (W. R. Hamilton, *Elements of Quaternions*, 1866; K. Shoemake, *Animating rotation with quaternion curves*, SIGGRAPH 1985, for `exp` and `ln` of rotations). Where `v = 0` there is no `u`, and each function is that of the complex number `w + x·i`, with the zeros of `j` and `k` kept: `ln(−1) = π·i`, `ln(−1 − 0·i) = −π·i`, `√(−4) = 2i`, `√(−4 − 0·i) = −2i`, as `Complex` and C99's `<complex.h>` have them.
- **Without cancellation near the identity.** `exp` multiplies `v` by `sin(r)/r`, which is 1 to the last bit for an `r` too small for the sine to differ from its argument, and keeps a zero vector part as it is. `ln` takes its real part as `Complex::ln` does: `ln(1 + (|q|² − 1))/2` from `(big − 1)(big + 1)` and the squares of the other parts where the largest part `big` is in `(1/2, 2)`, so that `ln|1 + 10⁻¹⁰i| = 5·10⁻²¹` keeps its digits; `ln(big) + ln(1 + Σ(other/big)²)/2` where `|q|` overflows or is subnormal. Its vector part is `v·(θ/r)` from `v` and `w` both scaled by the power of two that brings the largest part of `v` into `[1, 2)`, so that `r` keeps its digits for subnormal vector parts, and `v/w` itself where `r/w < √ε/2`, where `atan(r/w)·w/r` is 1 to the last bit: `ln(1 + 10⁻³⁰⁰i) = 10⁻³⁰⁰i`. `sqrt` cancels nothing, and takes the direction `u` from the scaled vector part.
- **At the ends of the range.** `exp` computes `e^w·t` as `e^(w/2)·t·e^(w/2)` where `e^w` overflows, so that `exp(710 + i)` keeps its finite real part, and keeps a zero part zero rather than `0·∞`. `sqrt` scales by a power of four near the subnormals and near overflow, as `Complex::sqrt` does.
- **Special values.** A quotient by zero divides each part of the dividend by the divisor's real part, a signed zero, as floats are divided: `recip(0) = ∞ + NaN·i + NaN·j + NaN·k`. A finite dividend over a divisor with an infinite part is zero; an infinite dividend over a finite nonzero divisor has `±∞` where the quotient of its direction (its infinite parts as `±1`, the others 0) is not zero; both infinite, or a NaN anywhere, give NaN. `ln(0) = −∞`; with an infinite part the logarithm is `+∞`, with the vector part `π·v/|v|` for `w = −∞` beside a finite nonzero vector part `v` (the limit: `ln(−∞ + j) = ∞ + π·j`, as `ln(−MAX + j) = 709.78… + π·j`), and otherwise the vector part of the infinite parts' direction (`ln(−∞) = ∞ + π·i`, `ln(+∞ + v) = ∞ + 0·v`). `exp` of an infinite real part keeps zero parts zero; of an infinite vector part it is NaN. `sqrt` of an infinite part is `+∞` with the vector part's infinities (the finite parts zeros of their signs), or, for `w = −∞` and a finite vector part, `+0` and `±∞` in the nonzero parts of the vector. `q⁰ = 1` for every `q`; `0^t` is 0 for `t > 0` and `+∞` for `t < 0`. Elsewhere, a NaN part gives NaN in every part. A product of operands with an infinite or NaN part is the plain formula, in which `0·∞` makes NaNs as it does for floats; in a product of finite operands that overflows, a part that the plain formula gives finite had no product in it overflow and is kept, accurate as in any product, and the other parts are taken from the product of the operands scaled by the powers of two that bring their largest parts into `[1, 2)`, scaled back by the sum of the powers. Such a part has products whose absolute values add up to at least `2¹⁰²⁴`, beside which what the scaling loses of the operands' small parts is at most `8ε`; so every part is within a few units of `ε` times that sum, and is `±∞` where it overflows by more: `(10²⁰⁰·(1 + i + j + k))² = −∞ + ∞·i + ∞·j + ∞·k`, `(1 + 10³⁰⁰i)² = −∞ + 2·10³⁰⁰i`, `(2¹⁰²³ + 2⁻⁶⁰i)·2¹⁰²³ = ∞ + 2⁹⁶³i`. `powi` starts from the base, so that `q¹ = q` and `q² = q·q` for every `q`. It is accurate normwise, to about `n·ε·|qⁿ|`, as repeated products are, and a part much smaller than that can lose its digits to cancellation, as in a single product. Where one of its products overflows and a later one would meet the infinities, the power of a finite quaternion is formed again from the base scaled by a power of two, the scaled powers multiplied with an exponent of their own and scaled back once: no part is NaN, and the parts larger than the error are `±∞`; `2^1100 = ∞`, its zero parts zero, as `powf` gives it.
- **Where `Complex` differs.** On `w + x·i` with finite operands and results the two agree. At special values these differ from `Complex`'s: a NaN beside an infinity makes every part NaN here, where `Complex` follows C99's Annex G (`ln(∞ + NaN·i) = ∞ + NaN·i`, `√(NaN + ∞·i) = ∞ + ∞·i` there); `1/(−0)` is `−∞ + NaN·i` here and `+∞ + NaN·i` there; the reciprocal of `±∞ ± ∞·i` is 0 here and NaN there; an overflowing product or power has no NaN part here, where `Complex`'s may (`(10²⁰⁰ + 10²⁰⁰i)²` is `NaN + ∞·i` there, `0 + ∞·i` here); `powi` of zero and of infinities differs (`0⁻¹` is the reciprocal `∞ + NaN·i` here and `NaN + NaN·i` there, `∞¹` is `∞` here and `∞ + NaN·i` there, where the power starts from 1); and `0^t` for `t < 0` is `+∞` here and 0 there.
- **Cost.** Every operation is `O(1)`; the scaling of the operands is a branch taken only outside `[MIN/ε, ε/MIN]`.

### Example

```rust
use a::math::complex::Complex;
use a::math::quaternion::Quaternion;

// Powers: integer, real and quaternion, the last with the logarithm on the right.
let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
assert_eq!(q.powi(3), Quaternion::new(-8.0, 0.0, 0.0, 0.0));
assert!((q.powf(3.0) - q.powi(3)).norm() < 1e-13);
// iⁱ = e^(−π/2), as for complex numbers; jⁱ = −k with the logarithm on the right.
let (i, j, k) = (Quaternion::<f64>::I, Quaternion::J, Quaternion::K);
assert!((i.powq(i) - Quaternion::from((-std::f64::consts::FRAC_PI_2).exp())).norm() < 1e-16);
assert!((j.powq(i) + k).norm() < 1e-15);

// Products and powers that overflow have infinities, not NaNs.
let c = Quaternion::new(1e200, 1e200, 1e200, 1e200);
let inf = f64::INFINITY;
assert_eq!(c * c, Quaternion::new(-inf, inf, inf, inf));
assert_eq!(Quaternion::new(2.0, 0.0, 0.0, 0.0).powi(1100), Quaternion::new(inf, 0.0, 0.0, 0.0));

// The complex numbers are a subalgebra, with Complex's branch cut.
let z = Complex::new(-4.0, 0.0);
assert_eq!(Quaternion::from(z).sqrt(), Quaternion::from(z.sqrt()));
assert_eq!(Quaternion::new(-4.0, -0.0, 0.0, 0.0).sqrt(), Quaternion::new(0.0, -2.0, 0.0, 0.0));

// Tiny vector parts keep their digits, and huge norms do not overflow.
assert_eq!(Quaternion::new(1.0, 1e-300, 0.0, 0.0).ln().x, 1e-300);
assert_eq!(Quaternion::new(0.0, 1e-300, 0.0, 0.0).exp().x, 1e-300);
let big = Quaternion::new(1.0, -2.0, 2.0, 4.0) * 2f64.powi(1000);
assert_eq!(big.norm(), 5.0 * 2f64.powi(1000));
assert_eq!(big.recip() * 2f64.powi(1000), Quaternion::new(1.0, -2.0, 2.0, 4.0).recip());
```

### Verified

Errors are normwise, `|computed − exact| / |exact|` in units of `ε` (`2⁻⁵²` for `f64`, `2⁻²³` for `f32`), the largest seen over the inputs stated; they are not bounds.

- **mpmath at 60 digits** (`lib/tests/math/quaternion/data/mpmath_functions.txt`, written by `gen_mpmath.py`): `exp`, `ln`, `sqrt`, `q^(1/3)`, `q^(−2.5)` and `q^p` for `p = 0.5 + 0.25i − 0.125j + k`, on 24 inputs chosen to be hard: near the identity (`1 + 10⁻¹⁰i`, `1 + 2⁻⁴⁰ + 10⁻⁸(i + 2j − 3k)`, vector parts down to `10⁻²¹⁰`), on the unit sphere with large parts, on and near the negative real axis (`−2`, `−1 + 10⁻¹²i`, `−3 + 10⁻³⁰⁰i`), pure vectors, parts of `10³⁰⁰`, `1.7·10³⁰⁸` and `10⁻³⁰⁰`, subnormal parts, `exp` near and past overflow (`709.5 + …`, `710 + i`, `−700 + …`) and vector parts of length 5 to 54. Dividing each error by the condition of the function (`1 + |q|` for `exp`, `1 + |t·ln q|` for the powers, 1 for `ln` and `sqrt`), the largest are 0.50 (`exp`), 0.77 (`ln`), 0.60 (`sqrt`), 1.00, 0.78 and 0.46 (the powers). Results that overflow are infinite in the same parts as mpmath's; `exp` of the inputs of norm `10³⁰⁰` and more, whose angle the precision does not determine, is left out.
- **160-bit arithmetic** (the crate's correctly rounded `BigFloat`, the same polar formulas, in the tests), 1,500 quaternions, half with parts in `[−4, 4)`, half with a vector part of `2^±60` beside a real part of `2^±60`: `exp` 1.00, `ln` 1.25, `sqrt` 1.63, `powf` 1.17 (scaled by its condition), `ln` of unit quaternions 1.50. The norm, on 3,000 quaternions whose parts have magnitudes from `2⁻¹⁰⁷⁴` to `2¹⁰²³`: 0.99 (within two units of the last place of a subnormal norm). `recip` and the quotient, on 2,000 operands of norms from `2⁻¹⁰⁰⁰` to `2¹⁰⁰⁰`: 1.35 and 1.70. `normalize` on 2,000 of norms from `2⁻¹⁰⁷⁴` to `2¹⁰⁰⁰`: `| |q/|q|| − 1 |` at most 1.
- **Identities** on 2,000 random triples: associativity 2.13, distributivity 1.32, `|pq| = |p||q|` 2.16, `q·q⁻¹ = q⁻¹·q = 1` 1.52, `conj(pq) = conj(q)·conj(p)` 1.27, `(a/b)·b = a` 1.81, and `a/b` against `a·b⁻¹` in 160 bits; Hamilton's whole table of units, exactly, in both types. `exp(ln q) = q` 3.79 on 2,000; `ln(exp q) = q` 1.23 (in units of `ε·(1 + |q|)`) for vector parts shorter than `π`, and the other branch, `w − v·(2π − r)/r`, 1.16 for vector parts between `π` and `2π`. `powi(n)` against repeated products for `n` from −7 to 7, `powf` at integers against `powi`, `sqrt(q)²` against `q`, `sqrt` against `powf(½)`, and `powq` of a real against `powf`. `powq` against `exp(ln(q)·p)` in 160 bits for 1,000 random `q` and `p`: 1.45, scaled by the condition `1 + |ln(q)·p|`.
- **Overflow.** 6,000 random pairs against 160-bit products, part by part with the yardstick of each part, `ε` times the sum of the absolute values of its four products: parts each of its own magnitude from `2⁻¹⁰⁷⁴` to `2¹⁰²³`, one large part beside moderate ones, a large real part times small parts, and parts within `2²⁰⁰` of one another. 5,166 products overflow, with 7,827 finite parts among them; every part is within 4 of its yardstick (the largest finite-part error 0.92), and every part that overflows by more is the infinity of its sign. The auditor's examples among them: `(2¹⁰²³ + 2⁻⁶⁰i)·2¹⁰²³ = ∞ + 2⁹⁶³i` and its square `∞ + 2⁹⁶⁴i`, `(10³⁰⁰ + 10⁻³⁰⁰i)·10¹⁰ = ∞ + 10⁻²⁹⁰i`, `(10³⁰⁸ + 10⁻¹⁰i)·10` and `(10³⁰⁸ + i/3)·10` with the `i` parts of the floats' products, exactly. In an independent sweep of 30,000 pairs against mpmath at 60 digits, by the same yardstick, 23,794 products overflow, with 60,682 finite parts among them: no part is NaN, every part that overflows is the infinity of its sign, and every finite part is within 4 of its yardstick (the largest 0.93); in another, of 39,520 pairs with parts from `2⁻¹⁰⁷⁴` to `2¹⁰²³`, 5,911 overflowing products, 3,020 quotients and 495 reciprocals are all right. 300 powers `qⁿ`, `n` from 600 to 3,000, of quaternions, complex numbers and pure vectors of norms 1.5 to 3, and 800 of mixed part sizes (parts down to `2⁻⁴⁰` or `2⁻⁸⁰` of the norm; huge bases with `n` up to 7; tiny bases with `n` down to −7, whose reciprocals overflow; one dominant part, `n` to 1,200), against 160-bit repeated squaring: 256 and 699 overflow, every part within `2(n + 1)·ε·|qⁿ|` of the exact power (an infinity counting where that reaches past the largest float), no NaN, the zero parts zero; `(2 + 2⁻¹⁰⁷³i)³⁰⁰⁰ = ∞ + ∞i`, whose `i` part the scaled power holds as a subnormal and scales back by `2³⁰⁰⁰` in three steps; `q² = q·q` where the square overflows, `(2.94·10⁻⁵ − MAX·i − MAX·j + MAX·k)²` among them; `(2.886·10⁸⁶ − 1.223·10⁷⁷i + 1.749·10⁹⁰j − 3.362·10⁹⁸k)⁴`, whose `i` part, `10⁻³³` of the norm, is lost to cancellation in the square, within the normwise bound. Quotients whose scaling back crosses the smallest subnormal round once: `1.75·2⁻⁵⁶⁰ / 2⁵¹⁵ = 2⁻¹⁰⁷⁴`.
- **`Complex`**: on 1,000 complex numbers (a quarter on the negative real axis), `exp`, `ln`, `sqrt` and `recip` agree with `Complex`'s to 4 units, `powi(5)` to 16 and `powf` to 16 times its condition, with zero parts of `j` and `k`; the products, sums and quotients of 200 more agree exactly (quotients to 4 units). The branch cut takes the side of the sign of a zero `x` in `ln` and `sqrt`, as `Complex` does.
- **`f32`** against `f64` on the same inputs (2,000 quaternions), in units of the `f32` `ε`: `exp` 0.89, `ln` 1.56, `sqrt` 1.41, `powf` 1.24, `recip` 1.24, the product 0.88; and the ends of the `f32` range.
- **Edges**: signed zeros, infinities and NaN in every function as above, `ln` at `w = −∞` beside a vector part, and the switch of `ln` to `v/w`: `v/w` exactly below `r/w = √ε/2`, the general formula from `1.3·√ε` (below `v/w` there, and within `ε` relatively of 160 bits); parts at `2^±1000`, `2⁻¹⁰⁷⁰`, the smallest subnormal and the largest float; `exp(710 + i)`, `exp(−800 + …)`; `0⁰`, `0^t`, `0^p`; `i^(2³¹)`.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same tests pass.

To run the tests: `cargo test --test math -- quaternion::algebra quaternion::functions`. The data file is written again by `python3 -B gen_mpmath.py` in `lib/tests/math/quaternion/data/`, with mpmath installed.

### Limits

- The errors are normwise. A small part of a product, quotient or power can lose its relative digits to cancellation, as for complex multiplication.
- For a vector part longer than about `1/ε` the angle `r` of `exp`, rounded by more than `2π`, does not determine the direction of the result, and neither does the precision; a vector part whose length overflows gives NaN.
- A NaN gives NaN in every part, where C99, and `Complex` with it, keeps an infinity in some cases (`csqrt(−∞ + NaN·i)` is `NaN ± ∞·i`); the other special values where `Complex` differs are listed above.
- `0^t` for `t < 0` is `+∞`; `Complex::powf` gives 0 there.
- An overflowing product takes the scaled path, about twice the work, and an overflowing `powi` forms the power twice.
- A power is accurate normwise only: where `qⁿ` overflows, a part far below `n·ε·|qⁿ|` is not determined, and can come out 0, or `±∞` of either sign, where its exact value is finite or overflows.
- Nothing is tuned: each function checks its range on every call, and `exp`, `ln` and `sqrt` take a few divisions more than they must.

## Rotations

| Function | Method | Of a quaternion not of norm 1 |
| --- | --- | --- |
| `from_axis_angle(axis, θ)` | `cos(θ/2) + u·sin(θ/2)`, `u = axis/|axis|` normalised with scaling; a zero axis gives the identity, a non-finite axis or angle NaN | unit result |
| `to_axis_angle()` | angle `2·atan2(|v|, |w|)` in `[0, π]`, axis `±v/|v|` (the sign of `w`); the identity gives `[1, 0, 0]` and 0 | the same for `q/|q|`, by the formulas' homogeneity |
| `angle()` | `2·atan2(|v|, |w|)` | homogeneous |
| `from_rotation_vector(v)` | the rotation by `|v|` about `v/|v|`; `v/2` as the vector part for `|v| < √ε` | unit result |
| `to_rotation_vector()` | `2·` the vector part of `ln(±q/|q|)`, `w ≥ 0`: the angle in `[0, π]` | homogeneous |
| `rotate(v)`, `rotate_inverse(v)` | `v + (w·t + u × t)/|q|²` with `t = 2·u × v`, the expansion of `q·v·q⁻¹`; the inverse by the conjugate | the rotation of `q/|q|` |
| `to_rotation_matrix()`, `to_matrix()` | `I + 2/|q|²·(w·[u]ₓ + [u]ₓ²)` | the rotation of `q/|q|` |
| `from_rotation_matrix(m)`, `from_matrix(&m)` | Shepperd's method: the largest of `1 + tr R`, `1 + 2Rᵢᵢ − tr R` picks a column of `4·q·qᵀ`, normalised; `w ≥ 0` | unit result |
| `slerp(b, t)` | Shoemake's `sin((1 − t)Ω)/sin Ω·a + sin(tΩ)/sin Ω·b` on the shorter arc, `Ω = 2·atan2(|a − b|, |a + b|)` | both normalised first |
| `nlerp(b, t)` | `((1 − t)·a + t·b)/|…|` on the shorter arc | both normalised first |
| `angle_to(b)` | `2·atan2(|v|, |w|)` of `conj(a)·b`, in `[0, π]` | homogeneous |
| `from_rotation_arc(a, b)` | the unit quaternion of `1 + â·b̂ + â × b̂`; a half turn about `â × e` (`e` the coordinate axis least aligned with `â`) where `|â × b̂| < √ε` and `â·b̂ < 0`, where `1 + â·b̂ ≈ δ²/2` is rounding alone | unit result |

### Methods

- **`rotate`** expands `q·v·q*` as `(w² − |u|²)·v + 2(u·v)·u + 2w·(u × v)`, divided by `|q|²`, which is the rotation of `q/|q|` for any norm, in the form `v + (w·t + u × t)/|q|²`, `t = 2·u × v`: two cross products and no quaternion product. A quaternion of squared norm outside `[1/16, 16]` is first scaled by a power of two, exactly, which brings its largest part into `[1, 2)`; so are those of the matrix, the axis and the angles.
- **`from_rotation_matrix`** follows S. W. Shepperd, *Quaternion from rotation matrix* (J. Guidance and Control 1(3), 1978), in the form F. L. Markley gives it (*Unit quaternion from rotation matrix*, J. Guidance, Control, and Dynamics 31(2), 2008) and SciPy uses: `4·q·qᵀ` has the diagonal `1 + tr R`, `1 + 2R₀₀ − tr R`, `1 + 2R₁₁ − tr R`, `1 + 2R₂₂ − tr R` and off the diagonal sums and differences of symmetric elements of `R`; the column of the largest diagonal element, at least 1, is `4·q_k·q`, and normalising it gives `±q` without a square root of a small number. The result is turned to `w ≥ 0`.
- **`slerp`** is Shoemake's (SIGGRAPH 1985). The angle between the two unit quaternions is taken as `Ω = 2·atan2(|a − b|, |a + b|)`, accurate for nearly equal rotations where `arccos(a·b)` loses half its digits, and `b` is negated when `a·b < 0`, so that `Ω ≤ π/2` and the path is the shorter one; the ratios of sines cancel nothing for small `Ω`. When `a·b = 0` there are two arcs of the same length, and the one through `b` as given is taken. A `t` outside `[0, 1]` extrapolates along the same great circle.
- **`angle_to`** takes the angle of `conj(a)·b` from its parts, which is homogeneous in both norms and keeps its digits for nearly equal rotations: `10⁻¹⁴` apart gives `10⁻¹⁴` to 4 units.
- **`to_rotation_vector`** and **`to_axis_angle`** take the direction from the vector part scaled by a power of two and the angle from `atan2`, so that rotations by `10⁻³⁰⁰` keep their digits. `from_rotation_vector` returns the vector part `v/2` for `|v| < √ε`, where `sin(|v|/2)/|v|` is ½ to the last bit.
- **`from_rotation_arc`** normalises `1 + â·b̂ + â × b̂`. At the angle `π − δ`, `1 + â·b̂` is about `δ²/2` and carries the error of about `ε` of the rounded `â·b̂`, so that the result's error grows like `ε/δ`, the conditioning of the problem, while `â × b̂` keeps its digits; where `|â × b̂| < √ε`, `δ²/2` is below that error, and the half turn about `â × e` is taken, which takes `â` to within `δ` of `b̂`.
- Every conversion is `O(1)`.

### Example

```rust
use a::math::matrix::Matrix;
use a::math::quaternion::Quaternion;
use std::f64::consts::PI;

// Axis and angle, rotation vectors and matrices, both ways.
let q = Quaternion::<f64>::from_axis_angle([1.0, -2.0, 2.0], 2.5);
let (axis, angle) = q.to_axis_angle();
assert!((angle - 2.5).abs() < 1e-15 && (axis[1] + 2.0 / 3.0).abs() < 1e-15);
let v = q.to_rotation_vector();
assert!((Quaternion::from_rotation_vector(v) - q).norm() < 1e-15);
let m = q.to_rotation_matrix();
assert!((Quaternion::from_rotation_matrix(m) - q).norm() < 1e-15);
assert_eq!(q.to_matrix(), Matrix::from_rows(&m));
// q and −q are the same rotation, and the norm does not matter.
assert_eq!((-q).to_rotation_matrix(), m);
let w = (q * 3.0).rotate([1.0, 0.0, 0.0]);
assert!((0..3).all(|i| (w[i] - m[i][0]).abs() < 1e-15));

// Slerp along the shorter arc, and the angle between two rotations.
let a = Quaternion::<f64>::from_axis_angle([0.0, 0.0, 1.0], 0.25);
let b = Quaternion::from_axis_angle([0.0, 0.0, 1.0], 2.0);
assert!((a.slerp(-b, 0.5).angle_to(a) - 0.875).abs() < 1e-15);
assert!((a.angle_to(b) - 1.75).abs() < 1e-15);

// The shortest arc between two directions, and a half turn between opposite ones.
let r = Quaternion::<f64>::from_rotation_arc([1.0, 0.0, 0.0], [0.0, 0.0, -2.0]);
let t = r.rotate([1.0, 0.0, 0.0]);
assert!(t[0].abs() < 1e-15 && t[1].abs() < 1e-15 && (t[2] + 1.0).abs() < 1e-15);
assert!((Quaternion::from_rotation_arc([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]).angle() - PI).abs() < 1e-15);
```

### Verified

Against SciPy 1.18.0's `Rotation` and `Slerp` (`lib/tests/math/quaternion/data/scipy_rotation.txt`, written by `gen_scipy.py` from inputs that the tests generate again from the same seeds, by SplitMix64 in `gen_common.py` and `common.rs`), differences as the largest absolute difference of parts, or of `min(|p − q|, |p + q|)` for rotations:

- `as_matrix` of 12 random quaternions: `3.3·10⁻¹⁶`; `from_matrix` of those matrices: `1.6·10⁻¹⁶`, and the round trip to the quaternion `1.7·10⁻¹⁶`. Each matrix is orthogonal to 4 units and has the determinant 1 to 8.
- `from_matrix` near the switches of Shepperd's method: half turns about `x`, `y`, `z`, `(1, 1, 0)`, `(1, −1, 1)` and `(0, 1, 10⁻⁸)`, each `10⁻⁷` short of a half turn too, the identity and a rotation by `4·10⁻⁹`: at most `1.1·10⁻¹⁶`. A random rotation with noise of `10⁻⁹`, `10⁻⁶` and `10⁻³` in each element: `4.5·10⁻¹⁰`, `5.5·10⁻⁷` and `4.6·10⁻⁴` from SciPy, which orthogonalises such a matrix by an SVD first; one scaled by 1.001: `6.8·10⁻⁵`. Over 2,000 random rotations, `1.001·R` moves the quaternion by `2.4·10⁻⁴` on average and `4.2·10⁻⁴` at most, each within 1 % (and `10⁻⁶`) of the first-order amount `η·√(1 − p²)/(4p)` for `η = 10⁻³` and the largest part `p ≥ ½` of the quaternion, which is at most `0.43·η`.
- `from_rotvec` of 18 vectors of lengths `10⁻³⁰⁰` to 10: `2.2·10⁻¹⁶`, and for the tiny ones the vector part to 4 units, relatively, both ways; below `√ε` the vector part is `v/2` exactly, in every part, for lengths from `10⁻⁹` to `2⁻¹⁰⁶⁰`; `as_rotvec` of 12 random quaternions `4.4·10⁻¹⁶` and of six chosen ones (`w = 0`, `w < 0`, vector parts of `10⁻⁹`, `3·10⁻¹²` and `10⁻³⁰⁰`) `2.2·10⁻¹⁶`, the tiny parts to 4 units relatively.
- `apply`, and `apply(…, inverse=True)`, on 12 random quaternions and vectors of parts up to 10: `5.3·10⁻¹⁶` relative to 10. Against `q·v·q⁻¹` in 160-bit arithmetic, for 2,000 quaternions of norms `2^±500`: 3.1 units of `ε·|v|`, and `rotate_inverse(rotate(v))` back to `v` to 5.3; the matrix rotates as the quaternion to 8 units.
- `Slerp` between 12 random pairs at random `t`: `2.8·10⁻¹⁶`; between pairs `10⁻⁹` apart, nearly opposite (`10⁻⁹` and `10⁻⁴` from `−q`) and 3 radians apart, at `t` = 0, ¼, ½ and 1: `1.7·10⁻¹⁶`. On 500 random pairs, `slerp` turns at a constant angular speed (the angle from `a` is `t` times the whole, to `10⁻¹³`), equals `a·(a⁻¹b)^t` to `10⁻¹⁴`, and `nlerp` stays on the same arc.
- `magnitude`, and the magnitude of `r₀⁻¹·r₁`, for 12 random quaternions and pairs: `4.4·10⁻¹⁶` from `angle` and `angle_to`.
- `align_vectors([a], [b])`, the shortest arc taking `b` to `a`, on 8 random pairs: `2.2·10⁻¹⁶`. On 500 random pairs, `from_rotation_arc` takes `â` to `b̂` to 16 units, by the angle between them (to `10⁻⁷`, the accuracy of the inverse cosine it is compared with) about an axis perpendicular to `â`; opposite and nearly opposite (`10⁻¹²`, `10⁻⁹`) pairs to `10⁻⁷`. At `π − δ` from `â`, with `δ` from `10⁻³` down to `1.4·√ε`, the arc is within `4ε/δ` of the exact rotation and takes `â` to within `8ε/δ` of `b̂`; at `0.7·√ε` and `0.5·√ε` it is the half turn about `â × e`, which takes `â` to within `δ` of `b̂`.
- Invariants on 500 to 2,000 random quaternions each: the matrices of `q` and `−q` are equal bit for bit; `q` scaled by 3, `10⁻³`, `2^±600` and `2^±1000` gives the same rotation, matrix, axis, angle and rotation vector to 8 units, and `(3 − 4i + 12k)·2⁻¹⁰⁷⁰`, subnormal, the matrix and the axis of `3 − 4i + 12k` exactly; quaternions of norms `2^±300` and `2^±500` rotate vectors of `10⁻²⁰` and `10²⁹⁰` as their unit quaternions do, to 8 units of `ε·|v|`; composition, `(a·b).rotate(v) = a.rotate(b.rotate(v))` and the product of matrices, to 16; round trips of axis and angle (angles in `[0, π]`, plus `2π`, negated), rotation vectors and matrices to 8.
- glam 0.33.12 (`DQuat`) and nalgebra 0.35.0 (`Quaternion`, `UnitQuaternion`), in a probe outside the crate, on 20,000 random cases each, the largest difference: the product 0 from both (the same formula); `rotate` `1.1·10⁻¹⁵` and `1.3·10⁻¹⁵`; `from_axis_angle` `2.6·10⁻¹⁶` and 0; the angle of `to_axis_angle` `8.9·10⁻¹⁶` and `4.4·10⁻¹⁶` (glam's angle, in `[0, 2π)`, taken back to `[0, π]`); the matrix `1.3·10⁻¹⁵` and `1.0·10⁻¹⁵`; `from_rotation_matrix` `6.5·10⁻¹⁶` and `6.2·10⁻¹⁶`; `slerp` `3.7·10⁻¹⁶` and `4.0·10⁻¹⁶`; `nlerp` (glam's `lerp`) `3.6·10⁻¹⁶` and `2.9·10⁻¹⁶`; `angle_to` (glam's `angle_between`) `6.8·10⁻¹⁵` and `8.9·10⁻¹⁶`; `from_rotation_arc` (nalgebra's `rotation_between`) `9.0·10⁻¹⁵` and `3.1·10⁻¹⁵`; rotation vectors `2.7·10⁻¹⁵` and `8.9·10⁻¹⁶`; `recip` `6.0·10⁻¹⁶` and `4.3·10⁻¹⁶`; nalgebra's `exp`, `ln`, `powf` and `sqrt` `3.5·10⁻¹⁶`, `2.0·10⁻¹⁵`, `3.6·10⁻¹⁵` and `1.1·10⁻¹⁵` relative. Where they differ: for rotations `10⁻⁹` apart, `angle_to` gives `0.99999999947·10⁻⁹`, nalgebra's `1.0000000267·10⁻⁹` and glam's `angle_between`, an inverse cosine, `3.0·10⁻⁸`, and nalgebra's `slerp` halfway between them is `2.5·10⁻¹⁰` off; nalgebra's `ln(1 + 10⁻²⁰⁰i)` is NaN and `exp(10⁻²⁰⁰i)` loses the vector part; both overflow in the norm of `10²⁰⁰ + 10²⁰⁰i`, and nalgebra's inverse of it is 0; nalgebra's `ln(−4)` and `sqrt(−4)` are NaN, here `ln 4 + π·i` and `2i`; nalgebra's `rotation_between` of opposite vectors is `None`, here a half turn. On a matrix with noise of `10⁻³`, glam's quaternion is as far from the rotation as this one's (`4.4·10⁻⁴`); nalgebra's `from_matrix`, which iterates to the nearest rotation, `2.8·10⁻⁴`.
- `f32` against `f64` on 500 random cases: `rotate`, the matrix, `from_rotation_matrix`, `slerp` and the rotation vector to 8 units of the `f32` `ε`.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same tests pass.

To run the tests: `cargo test --test math -- quaternion::rotation`. The data file is written again by `python3 -B gen_scipy.py` in `lib/tests/math/quaternion/data/`, with SciPy installed.

### Limits

- `from_rotation_matrix` does not orthogonalise: a matrix within `δ` of a rotation gives a quaternion within about `δ` of that rotation's, not the nearest rotation that SciPy (by an SVD) and nalgebra's `from_matrix` (iteratively) find, and a scaled rotation is not recognised as one: `(1 + η)·R` moves the quaternion by up to about `0.43·η` (`4.2·10⁻⁴` at most for `η = 10⁻³` over 2,000 rotations). A reflection or another matrix far from a rotation gives a unit quaternion with no meaning; SciPy refuses a matrix of determinant ≤ 0.
- The axis of `to_axis_angle` and the direction of `from_rotation_arc` are ill-conditioned near a half turn and near opposite directions, as the problems are: the error grows like `ε/δ` at a distance `δ` from them, and within `√ε` of opposite directions, where `1 + â·b̂` is rounding alone, `from_rotation_arc` takes the half turn about a fixed perpendicular axis.
- `rotate` may overflow in its intermediate products for vectors longer than about a sixty-fourth of the largest float.
- Nothing is tuned: every method that reads a rotation checks the norm on each call, and slerp normalises both ends.

## Euler angles

| Function | Method |
| --- | --- |
| `from_euler(sequence, frame, angles)` | the product of the three elementary rotations `cos(θᵢ/2) + eᵢ·sin(θᵢ/2)`: `q₁·q₂·q₃` intrinsic, `q₃·q₂·q₁` extrinsic |
| `to_euler(sequence, frame)` | Bernardes and Viollet's direct method, with SciPy's ranges and its choice at gimbal lock |
| `EulerSequence::ALL`, `axes()`, `is_proper()` | the twelve sequences, their axes (0, 1, 2 for x, y, z), and whether the first axis comes again last |

### Methods

- **`to_euler`** follows E. Bernardes and S. Viollet, *Quaternion to Euler angles conversion: a direct, general and computationally efficient method*, PLOS ONE 17(11), e0276302, 2022, which SciPy's `as_euler` uses. For an extrinsic sequence of axes `(i, j, k)` (an intrinsic one is the extrinsic one of its axes reversed, with its angles reversed), the parts `a, b, c, d` are `w, qᵢ, qⱼ, ε·qₖ` for a proper sequence, `k` then the third axis and `ε` the sign of the permutation `(i, j, k)`, and for a Tait–Bryan one those of the quaternion turned a quarter turn about `j`, `w − qⱼ`, `qᵢ + ε·qₖ`, `qⱼ + w`, `ε·qₖ − qᵢ`. Then `θ₂ = 2·atan2(√(c² + d²), √(a² + b²))`, less `π/2` for Tait–Bryan, and `θ₁` and `θ₃` are the difference and the sum of the half sum `atan2(b, a)` and the half difference `atan2(d, c)` (the third times `ε` for Tait–Bryan), each brought into `[−π, π]`. No inverse cosine and no matrix: the angles are accurate everywhere but where they are ill-conditioned, next to gimbal lock.
- **At gimbal lock** (`θ₂` before the shift within `10⁻⁷` of 0 or `π`), the third angle of the sequence is set to 0, and the first is twice the half sum (at 0) or twice the half difference (at `π`, with the sign that the frame asks for), the whole turn about the one axis left: SciPy's choice. Within `10⁻⁷` of the lock the rotation from the returned angles is then within about `10⁻⁷` of the original, as with SciPy; at the lock itself, to rounding.
- `from_euler` multiplies three quaternions; both are `O(1)`.

### Example

```rust
use a::math::quaternion::{EulerFrame, EulerSequence, Quaternion};
use std::f64::consts::FRAC_PI_2;

// All 24 conventions round-trip angles inside their ranges.
for s in EulerSequence::ALL {
    for f in [EulerFrame::Intrinsic, EulerFrame::Extrinsic] {
        let middle: f64 = if s.is_proper() { 1.0 } else { -0.5 };
        let q = Quaternion::from_euler(s, f, [0.4, middle, -2.0]);
        let a = q.to_euler(s, f);
        assert!((a[0] - 0.4).abs() < 1e-14 && (a[1] - middle).abs() < 1e-14 && (a[2] + 2.0).abs() < 1e-14);
    }
}

// Pitch up by a quarter turn: gimbal lock. The third angle is 0, the first takes the yaw and the
// roll together, and the angles still give the rotation.
let (s, f) = (EulerSequence::ZYX, EulerFrame::Intrinsic);
let q = Quaternion::<f64>::from_euler(s, f, [0.5, FRAC_PI_2, 0.25]);
let a = q.to_euler(s, f);
assert!(a[2] == 0.0 && (a[0] - 0.25).abs() < 1e-15 && (a[1] - FRAC_PI_2).abs() < 1e-7);
let p = Quaternion::from_euler(s, f, a);
assert!((p - q).norm().min((p + q).norm()) < 1e-14);
```

### Verified

- SciPy 1.18.0's `from_euler` on 3 triples of angles in `[−4, 4)` for each of the 24 conventions: at most `2.6·10⁻¹⁶`; its `as_euler` on 3 random quaternions for each: `4.4·10⁻¹⁶`, modulo `2π`.
- Gimbal lock, in each of the 24 conventions, with the first and third angles random: the middle angle at either singular value and `10⁻⁹` inside it (96 cases), where SciPy sets the third angle to zero: the third angle exactly 0 and the others within `2.2·10⁻¹⁶` of SciPy's; `10⁻⁶` from it (48 cases), outside the tolerance: within `2.3·10⁻¹⁰`, the error of the ill-conditioned first and third angles there. In every case the angles give back the rotation, to rounding at the lock and to `10⁻⁷` beside it. The tolerance is pinned on both sides in all 24 conventions at both singular values: `0.8·10⁻⁷` from the lock is locked (the third angle 0) and `1.25·10⁻⁷` is not; and at `0.9·10⁻⁷` and `1.1·10⁻⁷` (8 cases) the decisions and the angles are SciPy's, to `10⁻⁸`.
- Round trips in all 24 conventions of 200 triples of angles inside the ranges and `10⁻³` from the lock: the same angles to `10⁻¹²`; and of 200 random quaternions: angles inside the ranges, and the rotation back to 16 units; `−q`, `q·2⁻⁶⁰⁰` and `q·2⁹⁰⁰` give the same angles to 16 units, and so do `q·2¹⁰²³` (whose sums `w ± qⱼ` would overflow unscaled), `MAX·(1 − i + j − k)` and `(3 − 4i + j + 12k)·2⁻¹⁰⁶⁰…⁻¹⁰⁷⁰`, subnormal. `from_euler` equals the product of the three `from_axis_angle` rotations, in the order of the frame, to 4 units, and an intrinsic rotation turns about the axis as the first rotation carried it.
- glam 0.33.12's `from_euler` in its 24 `EulerRot`s: `3.1·10⁻¹⁶`; its `to_euler` agrees to `1.6·10⁻¹⁴` away from the lock, except in the proper sequences `XYX`, `YZY` and `ZXZ`, where glam gives the middle angle in `[−π, 0]`: the other set of angles, `(θ₁ ± π, −θ₂, θ₃ ± π)`, of the same rotation. nalgebra 0.35.0's `from_euler_angles(roll, pitch, yaw)` is extrinsic `XYZ`: `3.0·10⁻¹⁶`, and its `euler_angles` to `1.0·10⁻¹⁴`.
- `f32` against `f64`: the angles to `10⁻⁵` away from the lock, and the rotation back to 8 units of the `f32` `ε`. Zero and non-finite quaternions give NaN angles.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same tests pass.

To run the tests: `cargo test --test math -- quaternion::euler`.

### Limits

- SciPy brings every angle into `[−π, π)` by a remainder, so it gives `−π` where `to_euler` gives `π` (the middle angle of a proper sequence at a half turn included), and its remainder rounds the angles by up to `4·10⁻¹⁶`; the two are the same rotation.
- The gimbal tolerance is SciPy's `10⁻⁷` for `f32` too, below the spacing of `f32` floats near `π`: for `f32` the lock is taken only where the middle angle rounds to its singular value, and next to it the first and third angles are ill-conditioned, though they still give the rotation.
- Within `10⁻⁷` of the lock the returned angles give the rotation only to about `10⁻⁷`, as SciPy's do.

## Efficiency

- Time complexity: O(1) for each operation
- Space complexity: O(1), four floats

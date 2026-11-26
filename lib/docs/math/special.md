[A](../../../README.md) › [Math](../../../README.md#math)

# Special functions

`a::math::special` computes the special functions of `f32` and `f64` arguments, and of `Complex<f32>` and `Complex<f64>` ones for the gamma function, its logarithm, the error functions and the Riemann zeta function. Every function is a free function, generic over the type of its arguments; an `f32` argument is computed in `f64` and rounded once. Each function's error bound in ulps is stated below and tested against reference values computed with mpmath at 256 bits.

- The gamma family: `gamma`, `lgamma` and `lgamma_sign`, `digamma` and `polygamma`, `beta` and `lbeta`, the regularised incomplete gamma functions `gamma_p` and `gamma_q` and the regularised incomplete beta function `beta_inc`, with their complements and inverses, and `factorial`, `binomial` and `pochhammer` of floats.
- The error functions: `erf`, `erfc`, the scaled `erfcx`, the inverses `erf_inv` and `erfc_inv`, the Faddeeva function `faddeeva`, the Fresnel integrals `fresnel` and Dawson's integral `dawson`.
- The Bessel functions `bessel_j`, `bessel_y`, `bessel_i` and `bessel_k` of any real order, the scaled `bessel_i_scaled` and `bessel_k_scaled`, the spherical `sph_bessel_j`, `sph_bessel_y`, `sph_bessel_i` and `sph_bessel_k`, and the Airy functions `airy_ai`, `airy_bi` and their derivatives.
- The Riemann zeta function `zeta`, the Hurwitz zeta function `hurwitz_zeta`, the dilogarithm `dilog` and the polylogarithm `polylog`; the Lambert W function's real branches `lambert_w0` and `lambert_wm1`; and the exponential integrals `expint_e1`, `expint_ei` and `expint_en`.
- The elliptic integrals `ellip_k`, `ellip_e`, `ellip_pi`, `ellip_f`, `ellip_e_inc` and `ellip_pi_inc` of the parameter `m = k²`, Carlson's `carlson_rf`, `carlson_rd`, `carlson_rj` and `carlson_rc`, and the Jacobi elliptic functions `jacobi_elliptic`.
- The orthogonal polynomials `legendre_p`, `assoc_legendre_p`, `hermite_h`, `hermite_he`, `laguerre_l`, `assoc_laguerre_l`, `chebyshev_t`, `chebyshev_u`, `jacobi_p` and `gegenbauer_c`.

## Examples

```rust
use a::math::complex::Complex;
use a::math::special::gamma;

assert!((gamma(5.0f64) - 24.0).abs() < 1e-13);
let z = gamma(Complex::new(0.5f64, 0.0));
assert!((z.re - core::f64::consts::PI.sqrt()).abs() < 1e-14);
```

## The gamma function and its family

`gamma`, `lgamma`, `lgamma_sign`, `digamma`, `polygamma`, `beta`, `lbeta`, `gamma_p`, `gamma_q`, `gamma_p_inv`, `gamma_q_inv`, `beta_inc`, `beta_inc_c`, `beta_inc_inv`, `beta_inc_c_inv`, `factorial`, `binomial` and `pochhammer` take `f32` or `f64` arguments; `gamma` and `lgamma` also `Complex<f32>` and `Complex<f64>`. An `f32` argument is computed in `f64` and rounded once (each part of a complex result).

### Definitions and domains

| Function | Value | Domain and special values |
| --- | --- | --- |
| `gamma(x)` | `Γ(x) = ∫₀^∞ t^(x−1) e^(−t) dt`, continued | `±∞` at `±0` (and for `0 < \|x\| ≤ 2⁻¹⁰²⁴`, where `1/x` overflows), NaN at the negative integers and `−∞`; `+∞` above 171.6243769563027; zero (with the sign of `Γ`) below about −184 |
| `lgamma(x)` | `ln \|Γ(x)\|` | `+∞` at the poles, at `±0` and at `±∞`; overflows above 2.556·10³⁰⁵ |
| `lgamma_sign(x)` | `(ln \|Γ(x)\|, sign of Γ(x))` | the value is `lgamma`'s, bit for bit; the sign is 1 or −1 (1 at the poles, −1 at `−0`, NaN for NaN) |
| `gamma(z)`, complex | `Γ(z)` | on the real axis the real `Γ`; `0` for an infinite `Im z`; each part overflows on its own |
| `lgamma(z)`, complex | the principal branch of `ln Γ(z)` (SciPy's `loggamma`), analytic off the cut `(−∞, 0]` | on the cut the limit from the side of the sign of the zero, `−π⌈−x⌉` from above (mpmath's value at a real `x`); `(+∞, NaN)` at the non-positive integers |
| `digamma(x)` | `ψ(x) = Γ′(x)/Γ(x)` | `∓∞` at `±0` and for `0 < \|x\| ≤ 2⁻¹⁰²⁴` (where `−1/x` overflows), NaN at the negative integers and `−∞` |
| `polygamma(n, x)` | `ψ⁽ⁿ⁾(x)`, `n = 0` is `ψ` | for `n ≥ 1`: `(−1)ⁿ⁺¹∞` at `+0`, `+∞` at `−0`, NaN at the negative integers; overflow to `±∞` (`ψ′` from `x = 2⁻⁵¹²` down) and underflow to `±0` at the right points |
| `beta(a, b)` | `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` | any real `a, b`: `±∞` at `a = ±0` (or `b`), zero where only `a + b` is a pole (exactly: a sum that only rounds to a non-positive integer is not one), NaN at the other poles; `0` for an infinite argument with the other positive, and where `B` underflows (both arguments beyond 1.3·10³⁰⁸ included) |
| `lbeta(a, b)` | `ln \|B(a, b)\|` | `+∞` at the poles of `Γ(a)` or `Γ(b)`, `−∞` where only `a + b` is a pole, NaN where both are; `−∞` where `ln B` overflows |
| `gamma_p(a, x)`, `gamma_q(a, x)` | `P = γ(a, x)/Γ(a)`, `Q = Γ(a, x)/Γ(a) = 1 − P` | `a > 0`, `x ≥ 0`, else NaN (also `a = ∞`); `(P, Q) = (0, 1)` at `x = 0`, `(1, 0)` at `x = ∞` |
| `gamma_p_inv(a, p)`, `gamma_q_inv(a, q)` | the `x` with `P(a, x) = p`, `Q(a, x) = q` | `p = 0` gives 0, `p = 1` gives `∞` (`q` mirrored); a subnormal root is found (to the spacing of the subnormals); a root below the least subnormal gives 0; NaN outside `[0, 1]` |
| `beta_inc(a, b, x)`, `beta_inc_c(a, b, x)` | `I_x(a, b) = B(x; a, b)/B(a, b)` and `1 − I_x(a, b)` | `a, b > 0` finite, `0 ≤ x ≤ 1`, else NaN; `I = 0` at `x = 0`, 1 at `x = 1` |
| `beta_inc_inv(a, b, p)`, `beta_inc_c_inv(a, b, q)` | the `x ∈ [0, 1]` with `I_x(a, b) = p`, `1 − I_x(a, b) = q` | 0 or 1 at the ends; a subnormal root is found; a root below the least subnormal gives 0; NaN, not an unverified root, where the iteration has not converged after 200 steps (no such case is known) |
| `factorial(x)` | `x! = Γ(x + 1)` | NaN at the negative integers; exact (correctly rounded) at the integers 0 to 170 |
| `binomial(n, k)` | `Γ(n + 1)/(Γ(k + 1) Γ(n − k + 1))` | 0 for a negative integer `k` and for integers `0 ≤ n < k`; `(−1)^k C(k − n − 1, k)` for a negative integer `n`; NaN where only the numerator has a pole |
| `pochhammer(x, n)` | `(x)ₙ = Γ(x + n)/Γ(x)` | the product `x(x + 1)…(x + n − 1)` for an integer `n` (its reciprocal for a negative one), also where `x` and `x + n` are both poles; 1 at `n = 0`; zero where only `x` is a pole, NaN where only `x + n` is |

### Error bounds

Measured against mpmath at 256 bits (and continued fractions at 320 bits for `P`, `Q` and `I` where mpmath's own do not converge): the bound each test enforces on the data of `tests/math/special/data/gamma_*.txt` and, in brackets, the largest error over the area's larger random surveys of November 2026 (counts given; see [Accuracy measured](#accuracy-measured)). In ulps of the result's type unless stated. An `f32` result is the `f64` result rounded once (`tests/math/special/gamma.rs` checks this bit for bit), so it is within `0.5 + 2⁻²⁹·e` ulps of `f32` for an `f64` error of `e` ulps: 0.501 wherever the `f64` bound is relative.

| Function | Region | `f64` bound [survey] | `f32` |
| --- | --- | --- | --- |
| `gamma` | normal results | 0.6 [0.586, 626,180] | 0.501 [0.500, 14,887] |
| | subnormal results (rounded once from the double-double) | 0.6 [0.500, 1,664] | |
| `lgamma`, `lgamma_sign` | `x > 0`; `x < 0` where `\|ln \|Γ(x)\|\| ≥ 1/4` | 0.6 [0.538, 519,123; 0.556, 200,384] | 0.501 [0.500, 16,003] |
| | `x < 0` where `\|ln \|Γ(x)\|\| < 1/4` (beside its zeros on the negative axis) | absolute 0.1 ε [0.075 ε, 17,502] | |
| `gamma`, complex | `\|Γ(z)\| > 10⁻³⁰⁰` and `\|z\| < 10¹⁴`, relative to `\|Γ(z)\|` | 1.25 ε [0.64 ε, 20,159; 0.68 ε on the second review's 2,000 in the representable band] | each part rounded once [0.50 ε] |
| | `\|z\| ≥ 10¹⁴`, where the condition number `\|zψ(z)\|` of `Γ` passes `1/ε` (see below) | `\|z\| ln \|z\| · 2⁻¹⁰⁴` [3.2 ε on `[10¹⁴, 10¹⁵)`, 300; 280 ε to `10¹⁷`, 281] | |
| `lgamma`, complex | relative to `\|ln Γ(z)\|` | 1.25 ε [1.05 ε, 6,500] | each part rounded once [0.49 ε] |
| `digamma` | `x > 0`; `x < 0` where `\|ψ(x)\| ≥ 1/2` | 0.75 [0.71, 520,935; 0.65, 197,935] | 0.501 [0.500, 17,887] |
| | `x < 0` where `\|ψ(x)\| < 1/2` (beside its zeros on the negative axis) | absolute 0.25 ε [0.20 ε, 18,076] | |
| `polygamma` | `x > 0`, `n` from 1 to 500 | 0.75 [0.58, 23,984; trigamma 0.62, 210,000] | 0.501 [0.500, 2,400] |
| | `x < 0`, relative to `n!(r^(−n−1) + p^(−n−1))` (`r`, `p` the distances to the integers beside `x`) | 0.75 [0.52, 9,862] | |
| `beta` | `a, b > 0` | 1.0 [0.53, 17,762] | 0.501 [0.500, 5,300] |
| | `a` or `b` negative | 1.0 [0.52, 11,276; 0.53 where `a + b` is beside a pole, 3,596] | |
| `lbeta` | `\|ln B\| ≥ 1/2`; absolute in units of `ε/2` below | 1.0 [0.51, 29,038] | 0.501 where `\|ln B\| ≥ 1/2` [0.500, 5,300] |
| `gamma_p`, `gamma_q` | `a < 20`, and `a ≥ 20` by series or fraction | 1.2 [0.70, 177,315] | 0.501 [0.500, 6,000] |
| | `a ≥ 20` by Temme's expansion | 1.2 [1.02] | |
| `gamma_p_inv`, `gamma_q_inv` | | 0.6 [0.52, 5,760; subnormal roots 0.50 in units of `2⁻¹⁰⁷⁴`, 1,143, but for the rule that a root below `2⁻¹⁰⁷⁴` gives 0: up to one unit for a root in `(2⁻¹⁰⁷⁵, 2⁻¹⁰⁷⁴)`, 4 of them] | 0.501 [0.500, 2,880] |
| `beta_inc`, `beta_inc_c` | `min(a, b) ≤ 1 < max(a, b)` | 1.5 [1.26 / 1.15 (`I` / `1 − I`), 1,227; with `max(a, b)` from 10⁶ to `2⁹⁹⁶` beside `x = u/b`, `u` about `min(a, b)`, 1.26, 1,900; subnormal `x` 1.01, 2,100] | 0.501 [0.499, 1,172] |
| | elsewhere | 1.5 [0.77 / 0.85, 2,810, and 1.03 for `1 − I` on the second review's sample; 0.50 on 6,000 more with `b ≥ 40` and a tiny `bx`; with one parameter from 10⁶ to `2⁹⁹⁶` beside `x = u/b`, 1.04, 4,521 and the review's; with both from 10³ to 10⁸ within 12 widths of the mean 0.90 / 0.89, 21,000; from 10⁵ to 10¹⁵ 0.84, 6,000, and 0.903 on a later independent survey of 2,000 with both in [10⁵, 10¹³] (beside `I = 1/2`); from 10¹⁵ to 10²⁹⁹ (and from 10⁶ with ratios to 10⁶, or equal), 0.82, 10,000] | |
| `beta_inc_inv`, `beta_inc_c_inv` | in units of `max(1, κ)` ulps, `κ = \|p/(x I′(x))\|` the condition number | 1.5 [0.75 / 0.74, 2,100 / 2,097; 0.75 / 0.97 on 7,124 more in log-space; a tiny `p` beside a large parameter 0.75, 6,610; one parameter from 10¹⁴ to 10²² 0.34 / 0.49, 160; both from 10⁶ to 10²⁶ 0.19 / 0.22, 116; both large (`max(a, b)` from 10¹⁰ to 10²⁹⁹, the other at least 10⁶ and up to 10⁸ times smaller, against the quadrature, the root bracketed), none beyond 1.5 on 8,275, and on 3,708 more from 10²⁰ on all within 0.5 but 4 within 1 (the rounding of `x = 1 − s`); subnormal roots 0.43 in units of `max(1, κ)·2⁻¹⁰⁷⁴`, 1,143, a root in `(2⁻¹⁰⁷⁵, 2⁻¹⁰⁷⁴)` giving 0 by the rule above] | 0.6 [0.50, 2,100] |
| `factorial` | | 0.6 [0.545, 420,326] | 0.501 [0.500, 3,000] |
| `binomial` | | 0.6 [0.519, 23,068] | 0.501 [0.500, 3,000] |
| `pochhammer` | | 0.6 [0.520, 25,235] | 0.501 [0.500, 3,000] |

Two of the absolute bounds are worth reading closely. `polygamma` for `x < 0` is measured against the size of its largest terms, not against its value: where the value is far smaller than they are (as for large `\|x\|`, where it is a small difference of them) the bound allows any relative error, and a value can be `±0` or a few per cent off (`polygamma(2, −2210422430566820.5)` is 4 % off, `polygamma(8, −3914715810378300.5)` is `−0`). And `lgamma`, `digamma` beside their zeros on the negative axis are differences of terms of a few units, accurate absolutely.

One limit is stated rather than met: the complex `gamma` for `|z| ≥ 10¹⁴`. Its relative error is the absolute error of `ln Γ(z)`, whose main terms `(z − ½) ln z − z` are `|z| ln |z|` in size and are carried in double-double, `2⁻¹⁰⁵` of themselves: from `|z| = 10¹⁴` on that is more than ε (no evaluation in double-double can do better there, and an ulp of `z` itself moves `Γ(z)` by `|zψ(z)|` ≈ `|z| ln |z|` ulps). The test holds such cases to `|z| ln |z| · 2⁻¹⁰⁴`.

### Method

- **Fused multiply-adds.** Where the hardware has them (aarch64, x86_64 with `fma`; not under Miri) the error-free steps are fused multiply-adds; elsewhere — x86_64's default target, as crates.io builds it — Dekker's product of Veltkamp's halves, and plain `a·b + c` for the terms whose rounding does not matter (the crate's `FMA` switch, as in `dd.rs`). The bounds hold on both paths (tested on x86_64 in Docker and under Miri; after the November 2026 fixes and again after each of their two reviews, under Miri on every 37th case of the data).
- **The elementary functions.** The area computes its own `ln`, `exp`, `sin(πx)` and `cos(πx)` in double-double (measured against mpmath: `ln` to 2⁻⁷² absolute, so 2⁻⁶⁵ relative beside 1; `exp` to 2⁻⁷⁴·⁷ relative; `sin(πx)` and `cos(πx)` to 2⁻⁵⁹ and 2⁻⁵⁶·⁷ relative, at the ends of their reduced interval), so that none of its real functions calls a `std` transcendental function: the results are the same on every target and under Miri. `ln x` reduces `x` to `[0.7057, 1.4114)` and subtracts `ln c` of 64 tabulated 7-bit `c` (so that `x/c − 1` is exact by one fused multiply-add), then `ln(1 + r) = r − r²/2 + r³P(r)` with the first terms exact; `e^x` takes `x − k·ln2/64` with a 36-bit `ln 2/64` and `2^(j/64)` from a table; `sin(πr)`, `cos(πr)` on `|r| ≤ 1/4` are minimax polynomials with their two leading terms in double-double, and `sin(πx)` and `cos(πx)` of a double-double `x` reduce both of its parts exactly and take the low part of the reduced sum to first order. For the complex logarithm of a large argument (below) a second logarithm, the same reduction with `ln(1 + r) = 2 atanh(r/(2 + r))` and three terms of its series in double-double, is good to 2⁻¹⁰⁰ absolute. A double-double result `d·2^m` below the normal range is rounded once: `d`'s high part scaled exactly to a normal number and rounded into the subnormals, that rounding corrected by the exact remainder and the low part. Every polynomial is evaluated by Estrin's scheme, fully unrolled. The minimax fits are the area's own, by a Remez exchange in mpmath: `gamma/gen_tab.py` (with `gen_remez.py`) writes `gamma/tab.rs`, and `gen_igamma_tab.py` writes `igamma_tab.rs`, each byte for byte after `rustfmt`.
- **`Γ(x)`.** For `|x| < 8`, `x = n + t` with `|t| ≤ 1/2`, `t = x − n` exact: `1/Γ(1 + t) = 1 + γt + c₂t² + t²R(t)` (`R` of degree 14, minimax relative), with the first three terms in double-double, then the product of the exact factors `t + k` up to `n − 1` (or down to `n`), compensated, and one double-double division; below `2⁻⁵⁴`, `1/x` with its remainder `(1 − x·q)/x`, then `− γ`, rounded once. From 8 on, Stirling's series: `(x − ½) ln x − x + ln √(2π) + S(x)` with `ln x` in double-double and `S` a minimax polynomial in `1/x²`, then the double-double exponential. Below −8, `Γ(x) = −π/(x sin(πx) Γ(−x))`, with `sin(πx)` and the exponential in double-double. `factorial` is the same with `x + 1` never rounded, and the table of `n!` at the integers.
- **`ln |Γ(x)|`.** The result below 8 comes from its own minimax pieces: `ln Γ(1 + t) = t·H(t)` on `−1/2 ≤ t < 1/2` (five pieces, the zero at 1 a factor), `ln Γ(2 + t) = t(1 + t)·F(t)` on `−1/2 ≤ t < 1` (three pieces, the zeros at 1 and 2 exact factors), `ln Γ` on the units of `[3, 8)`, each `Y + c₁s + s²R(s)` with the first two terms without error (on the units also `c₂s²`, with the rounding of `s²`); below ½, `ln Γ(1 + x) − ln x`. The other functions use a version in double-double to about 2⁻⁶⁰: below 1.5 `ln Γ(x) = ln Γ(x + 1) − ln x` (and `− ln(x(1 + x))` below ½, one logarithm of the double-double `x + x²`), between 3 and 8 the product of the exact factors and one logarithm, Stirling's series from 8 on (one double-double `ln` and the sums in double-double), `x(ln x − 1) − (ln x)/2` rounded once beyond 2⁵², and the reflection formula below −½ — where its terms cancel (beside the zeros on the negative axis), with `ln Γ(−x)` from the exact product and `ln Γ(2 + t)`, whose absolute error is the smaller.
- **Complex `Γ` and `ln Γ`.** Complex double-double: Stirling's series for `|z| ≥ 10` with `Re z ≥ 0` (its main terms `(z − ½) ln z − z` in double-double, which multiply the errors of `ln |z|` and `arg z` by `|z|`: so `arg z` to about 2⁻¹⁰⁰ from the octant, a table of `atan(j/64)` and the series of `atan u` of the remainder `u = (n − (j/64)d)/(d + (j/64)n)`, `|u| ≤ 1/128`, in double-double, and `ln |z|` from `|z| = 2¹²` on by the second logarithm above), the shift `ln Γ(z) = ln Γ(z + n) − ln Π(z + k)` below it with the product in double-double and the branch kept by counting its crossings of the negative axis (Hare's method, as in SciPy), Taylor series at the zeros 1 and 2, and for `Re z < 0` `ln Γ(z) = ln 2π − iπ/2 + iπz − ln(1 − e^{2πiz}) − ln Γ(1 − z)` for `Im z > 0`, the branch that is analytic in the upper half-plane (the lower by conjugation). `Γ(z) = e^{ln Γ(z)}` from the double-double logarithm: its exponential, and the sine and cosine of its imaginary part `θ` as `sin(πq)` and `cos(πq)` of the double-double `q = θ/π` with both of its parts reduced exactly (`θ` is of the size of `|z| ln |z|`), so the error stays near ε while `|ln Γ|` grows to about 10¹⁵. Beyond `|z| = 10¹⁴` the double-double `ln Γ(z)` itself, `|z| ln |z|` in size, carries an absolute error of about `|z| ln |z| · 2⁻¹⁰⁵`, which is the relative error of `Γ(z)` (there the condition number `|zψ(z)|` of `Γ` is past `1/ε`).
- **`ψ`.** `ln x + (its low part − 1/(2x) − u·T(u))`, rounded once, from 8 on (`T` minimax in `u = 1/x²`); eight minimax pieces `Y + c₁s + s²R(s)` on `[2, 8)`; `(x − x₀)·Q(x)` in four pieces on `[1, 2]` with the positive zero `x₀ = 1.4616321449683622…` in double-double; `ψ(1 + x) − 1/x` below 1; and `ψ(−u) = ψ(u) + 1/u + π cot(πu)` with `ψ(u) + 1/u = ψ(1 + u)` for `u < 1`, `cot` from one reduction for both the sine and the cosine in double-double and one division; below `2⁻⁶⁰` (where `sin(πu)` would be subnormal) `ψ(−u) = 1/u − γ`, rounded once.
- **The `f32` form of `ψ`** on `1 ≤ x < 8`: fourteen polynomials of degree 15 on the halves of `[1, 8)` (Chebyshev interpolants, mpmath at 40 digits; on `[1, 2)` of `ψ(x)/(x − x₀)` about the zero, times `x − x₀` from the double-double `x₀`), picked by an index, good to `2⁻⁵¹` in `f64` and rounded once to the `f32` (checked against the `f64` path at every `f32` of `[0.9, 8.1]`); elsewhere the `f64` path.
- **`ψ⁽ⁿ⁾`.** `ψ′` by its own method: `1/x + (its rounding error + 1/(2x²) + T(u)/x³)`, rounded once, from 10 on, sixteen pieces `Y + c₁s + R(s)` of width 1/2 on `[2, 10)` (the piece of `x` by an index, without a branch) with `c₁s` without error, and `Σ 1/(x + j)² + ψ′(x + m)` below 2, the piece of `x + m` evaluated at `s = x − (c − m)`, so that `x + m` is never rounded (its rounding error, below `x = 1/8`, through `c₁`). Otherwise `(−1)ⁿ⁺¹ n! ζ(n + 1, x)`, the Hurwitz zeta function by its terms up to where its Euler–Maclaurin tail converges to 2⁻⁶⁰ (from `y ≥ 0.38(n + 21) + 0.5`, with up to 40 Bernoulli terms, the first three in double-double), or until the rest is negligible; for `x < 0` the sum of all the terms `(x + j)^(−n−1)` as `(−1)^(n+1)[ζ(n + 1, r) − ζ(n + 1, 1 − x)] + ζ(n + 1, p)` of the exact distances `r` and `p` of `x` to the integers beside it. Large orders and arguments carry their values as `v·e^l`, so nothing overflows before the last product.
- **`B` and `ln B`.** For `a, b > 0`: below 8, `B = R(a + b)/(R(a)R(b))` with `R = 1/Γ` in double-double (the low part of `a + b` through `ψ`) and `ln B` its logarithm (above 10⁻¹⁰⁰; `ln Γ(a) + ln Γ(b) − ln Γ(a + b)` in double-double below that); `ln Γ(a) + [ln Γ(b) − ln Γ(a + b)]` by the Stirling difference when only the larger is, and for both from 8 on `ln √(2π) + (a − ½) ln(a/c) + (b − ½) ln(b/c) − ½ ln c + S(a) + S(b) − S(c)` with `ln(b/c) = −ln(1 + a/b)` computed directly (an `atanh` series for small ratios), so no term cancels another (at half scale where `a + b` overflows); `B` is its double-double exponential. The Stirling difference `ln Γ(a) − ln Γ(a + b) = −(a − ½)L₁ − b(ln a + L₁) + b + S(a) − S(a + b)` takes `L₁ = ln(1 + b/a)` from `b/a` itself (by the `atanh` series) for `a ≥ 2⁴⁰`, so a ratio beyond `2⁵³` keeps its digits (below `2⁴⁰` from the double-double `1 + b/a`, whose absolute error `2⁻¹⁰⁶` times `a` stays below `2⁻⁶⁶`). Other arguments go through the reflection formula as identities between beta functions of positive arguments and sines of the exact arguments: `B(a, b) = π sin(πc)/(sin(πa) sin(πb)) / ((1 − c) B(1 − a, 1 − b))` for `a, b < 0`, `(sin(πc)/sin(πa)) B(b, 1 − c)` for `a < 0 < b` with `c = a + b < 0`, `π/(sin(πa) · b · B(c, 1 − a))` for `c > 0`; `c` is the exact sum as a double-double, so a sum beside a pole keeps its distance to it, and the gamma functions of large arguments appear only in Stirling differences of the double-doubles, never as differences of their logarithms.
- **`binomial`, `pochhammer`.** For integers `0 ≤ k ≤ n ≤ 60` a table of the correctly rounded `C(n, k)` (`k ≤ n/2`, 961 values, rounded from the exact integers); the table of `n!` in double-double for integers to 170; compensated products for an integer `k` or `n` up to 32 (every factor `n − i` or `x + i` exact as a double-double, its low part and each product's rounding error carried); else `1/((n + 1) B(n − k + 1, k + 1))` with every argument an exact double-double (the beta function as above), and `(x)ₙ` by the Stirling difference, or for a negative `x` or `x + n` by `(x)ₙ = (sin(πx)/sin(πc)) Γ(1 − x)/Γ(1 − c)` and its companions (`c = x + n`).
- **`P` and `Q`.** The structure of DiDonato and Morris (1986) and Boost: the smaller of `P` and `Q`, and the other as its double-double complement. The prefix `x^a e^(−x)/Γ(a + 1)` in double-double, `e^(a ln(x/a) + a − x − S(a))/√(2πa)` from `a = 8`. The series of `P` below about `x = a`, Legendre's continued fraction for `Q` above (evaluated from the bottom as one composed map per four steps, so one division per four steps, its length set in advance, the last steps in double-double), finite sums for integer and half-integer `a` below 20 when `x ≥ a` (`(n − 1)!` from the table, `erfcx` for the half-integers, `a/x` in double-double), the small-`a` form `Q = (Γ(1 + a) − 1 − (x^a − 1))/…` without cancellation for `x < 1.1` (and `x < 2` for `a < ½`; further out `Q` is a small difference of its terms, and the fraction takes over), `x^a − 1` from the double-double exponential, and for `a ≥ 20` with `|x − a|/a` below about ½ Temme's uniform expansion with 13 coefficients `c_k(z)` derived by series arithmetic after Temme (1979) and fitted minimax, and the area's double-double `erfcx` (eight pieces below 4, then `(1/t)·g(1/t²)`; `beta_inc`'s BASYM uses the same).
- **Their inverses.** A first guess (`(pΓ(a + 1))^(1/a)` for small `x`; for `a ≥ 1` Temme's expansion inverted to first order, as DiDonato and Morris, with Acklam's normal quantile; an iteration on the tail for large `x`, and for `Q` with `1 ≤ a ≤ 50` in its tail (`−ln q > 3a`) the fixed point of its asymptotic form `x = −ln q − ln Γ(a) + (a − 1) ln x + ln(1 + (a − 1)/x + (a − 1)(a − 2)/x²)`, four steps from `−ln q − ln Γ(a)`), then Halley's method on the logarithm of the smaller tail in `ln x` with the slope from `ln Γ(a)` computed once, kept in a bracket: about two evaluations of `P` or `Q` a call. The iteration goes on into the subnormals and stops when a step no longer moves `x` (the nearest float to Newton's root).
- **`I_x(a, b)`.** TOMS 708 (DiDonato and Morris 1992) as structured in Boost's `ibeta_imp`, the side computed always the one that does not cancel (for both parameters above 1 by the sign of `λ = a − (a + b)x`, in double-double where its `f64` value is within its error): the power series (three terms in double-double), the continued fraction in its division-free form (its depth by a forward recurrence of the convergents with rescaling, then a projective backward pass, two steps and the coefficients in double-double, one division at the end), the `a`-step with BGRAT for a small `b` beside a large `a`, and BASYM for both above 100 near the mean (its leading term `erfcx(z₀)` with `z₀ = √f` of the double-double exponent `f`, the error of the rounded root entering to first order: at the rounded root alone, its rounding and that of `f` reached the result in full, up to 1.67 ulps beside the mean of parameters from 10⁵ on). The power terms `x^a y^b/B(a, b)` in double-double with `x` and `y = 1 − x` exact: below 8 with `1/B = R(a)R(b)/R(c)`, else as `√(ab/(2π(a + b)))·e^(aφ(r) + bφ(s) + S(c) − S(a) − S(b))` with `φ(r) = ln(1 + r) − r` and `ar + bs = 0`, so the exponent never cancels; `d = xc − a = b − yc` is formed from the smaller of `x` and `y` (from the larger, beside a huge `c`, its error `2⁻¹⁰⁶c` reaches the exponent in full), from `c = 2⁴⁰` on with every partial product exact (rounded, it would keep `2⁻¹⁰⁶` of `a`: beside the mean of two parameters beyond 10³⁰ more than its own ulp), and `1 + r = xc/a` and `1 + s = yc/b` from `x` and `y` themselves away from the mean (from `r` they would keep only its absolute precision, none where `x` or `y` is tiny), `ln(xc/a)` as `ln x + ln(c/a)` for a subnormal `x` (its products would lose their error terms in the subnormals). From a first parameter of `2⁴⁰` on, the continued fraction's coefficients are scaled by powers of two (exactly: they grow like `a⁴` a step and overflowed from about `2⁷⁰`), and BGRAT takes `ln x` as `ln1p(x − 1)` of the exact `x − 1` (its error is multiplied by `a`). For `max(a, b) ≥ 2⁹⁹⁶`, the incomplete gamma limit `I_x(a, b) → P(a, bx)` with the product `bx` exact (a double-double, its low part through the density), or a point mass at the mean beside a parameter from `2⁶⁰⁰` on.
- **Their inverses.** A first guess from the normal approximation with the Cornish–Fisher skewness term where it is small (for `a, b ≥ 2`), the leading tail term or the mean, then Halley's method on the logarithm of the smaller of `T` and `1 − T` (`T` the tail sought; where `T` passes ½, `ln(1 − T)`, steep where `ln T` is flat: from there Newton's step on `ln T` overshot into the far tail) in the log of the smaller of `x` and `y` (the step through `expm1`; the iterate changes sides where it passes ½, since the larger of the two lies on a coarser grid), with a bisection bracket: 2–3 evaluations a call. It stops where a step no longer moves the iterate, or where Halley's step is so small that the next would not (judged from the last two steps: the next error is about `C·dw³` with `C = dw/dw_last³`, which must be below half an ulp), and only with `|ln T − ln t| ≤ 2¹⁰`: further out a vanishing step means a slope without digits (`e` to a difference of two exponents of the size of `|ln T|`), and where `|ln T|` passes `2⁴⁰` the step is a bisection. The iteration goes on into the subnormals (the bracket is split geometrically) and stops when no float is left inside the bracket; after 200 steps without either, the result is NaN (no such case is known). For both parameters from `2⁸⁶` on, the root is `m + σz` (`m` the mean in double-double, `σ` the width, `z` the normal quantile): there the distribution is normal to far below an ulp (its width at most `2¹⁰` ulps of the smaller of `x` and `y`, its skewness at most `2⁻⁴²`, so the formula is within `10⁻⁴` ulp); the iteration is no more accurate there, and from about `2¹⁰⁶` on, where the width falls below an ulp, it cannot resolve the root (a float beside it lies many widths away, where its slopes carry no digits). For `min(a, b) ≥ 2⁶⁰⁰` (or `a + b` overflowing), whatever `max(a, b)`, the mean, rounded once from double-double (the width is below `2⁻³⁰⁰` of it); else for `max(a, b) ≥ 2⁹⁹⁶` the incomplete gamma limit (`x = u/b` with `P(a, u) = p` for a huge `b`, `x = 1 − u/a` with `Q(b, u) = p` for a huge `a`).

### Example

```rust
use a::math::special::{beta, beta_inc_inv, gamma_q};

// fl(−0.3) + fl(−0.7) = −1 + 2⁻⁵⁴: rounded it is the pole −1, exactly it is not.
let b = beta(-0.3f64, -0.7);
assert!((b / -1.0264863238377995e-15 - 1.0).abs() < 1e-15);
// Q of a chi-square variable of one degree of freedom (a = 1/2).
let q = gamma_q(0.5f64, 2.5928251148347505);
assert!((q - 0.022774148981795824).abs() < 4e-18);
// A root in the subnormals.
let x = beta_inc_inv(0.6931097059489624f64, 1283.9036731128458, 7.138512485316517e-212);
assert!((x - 1.5641409307357665e-308).abs() <= 5e-324);
```

### Accuracy measured

The bounds above are enforced by `cargo test --test math -- special::gamma` on the data files, which end with the regression cases of the November 2026 audit and of the two reviews of its fixes (the arguments that exposed each defect and points around them; `data/gen_gamma.py`'s `audit_*`, `review_*` and `review2_*` functions write them; each defect's cases were checked to fail on the code before its fix, though not every one of the points around them does). The bracketed maxima are from the audit's surveys rerun on the fixed code (the area's harness outside the repository: Python drivers with mpmath at 400–600 bits and Arb ball arithmetic for the references, random and dense sets per region, about 3.2 million cases in all; `beta_inc` against Arb's `arb_hypgeom_beta_lower`, the inverses against Newton's method in Arb from the library's root), and from the surveys of the review of the fixes (about 90,000 cases more, on seeds of their own: one parameter from 10⁶ to `2⁹⁹⁶` beside the mean, both parameters large, a subnormal `x`, tiny targets of the inverses, the complex `Γ` where it is representable; their references by Arb, by the incomplete gamma limit where its error is below 10⁻⁴⁰, by the continued fraction at the precision `a + b` needs, and for both parameters large by a quadrature of the density, checked against Arb and against the leading term of the uniform expansion), and from those of the second review and of its fixes (about 110,000 more: the beta inverses with both parameters from 10¹⁰ to 10²⁹⁹, each root bracketed by the quadrature; `I` beside the mean of both parameters from 10³ on; 60,000 inverses in the ordinary regions, bitwise against the code before, the few that differ against Arb). Unresolved there, and not in the maxima: 163 `beta_inc` references with a parameter beyond `2⁹⁹⁵` (Arb and mpmath do not converge; the data's references for those use the incomplete gamma limit, whose relative error is below `10⁻²⁸⁰` there), three `beta_inc_c_inv` roots within `2⁻⁵⁴` of 1 (bisection in mpmath: the library's `1 − 2⁻⁵³` is within half an ulp), and the 746 `pochhammer` cases where `x` and `x + n` are both poles (Arb gives no value; the library gives the documented product, checked to 0.5 ulps against it). No independent reference reached `P` and `Q` for `a ≥ 10²⁰`. To reproduce the repository's part: `python3 gen_gamma.py` in `tests/math/special/data` (byte for byte; `--big K --out DIR` writes K times as many random cases for the same tests), then the test above.

### Measured

Measured on an Apple M4 Max (macOS 27.2.1) on 2026-11-26 from 06:23, in the final run of the release's speed pass, which timed every area of the crate in turn with nothing else running: one thread on every side, by the harness `bench/special` of the repository (its method is in `bench/README.md`). `a` is built as a user builds it, by rustc 1.100 stable with the default release profile (no LTO, no `target-cpu`), and so are statrs 0.18.0, special 0.11.4, puruspe 0.4.4, libm 0.2.16 and lambert_w 1.2.34; Apple's libm, Boost.Math 1.92.0 (no promotion of `double` to `long double`; `float` promoted to `double`, as Boost does by default and as `a` computes its `f32` forms) and GSL 2.8 (Homebrew's, its error handler off) are called from drivers built with `-O3 -mcpu=native`; SciPy 1.18.0 (NumPy 2.3.3) is timed through its ufuncs on arrays, NumPy's loop included, alternated in the same Python process with `a` called through a C interface of the crate, so its rows carry their own time of `a` and the ratio is that row's. A case is one function over one region of arguments: 16,384 random arguments (a fixed seed a case), every implementation over the same ones in a throughput loop, alternated sample by sample, 21 samples of at least 1 ms after a warm-up, the median in ns a call. Before timing, every reference's results are compared with `a`'s on those arguments: the largest difference in ulps of `a`'s value (for the oscillatory functions in ulps of `max(|v|, median |v|)`, for complex values in units of `ε·|w|`), which, `a`'s own error being within its bound, is about the reference's error. The x86_64 paths were not timed for these tables.

The same work includes the accuracy: a reference counts only where its largest difference from `a` is at most twice `a`'s bound for the function (the largest of its regions' bounds above, in the bound's units; 0.501 ulps of `f32` for the `f32` forms), so that both can be within that bound, and where it gives no NaN or infinity for a finite value of `a` or the reverse. A reference that computes more than the function (SciPy's Airy functions, all four at once; its Jacobi functions, with the amplitude; puruspe's Bessel functions of real order, which compute `J` and `Y`, or `I` and `K`, together; SciPy's complex Lambert W) or goes through a Python wrapper (SciPy's `polygamma`) is not counted either. Of the 318 cases, 126 have a reference within the bound: 59 losses (a ratio `a/x` above 1.05 against the fastest of them, `a`'s time over the reference's), 6 ties within 5 % and 61 cases below 0.95, the median ratio about 0.99. The others have none; the last column gives, where one is faster than any within the bound, the fastest reference beyond it, with its largest difference from `a`, marked † and not counted.

| function | arguments | `a` | within the bound | faster, beyond the bound |
| --- | --- | ---: | --- | --- |
| `gamma` | x in [0.5,1.5] | 4.16 ns | 0.657 Apple libm |  |
| `gamma` | x in [1.5,8] | 7.51 ns | 0.723 Apple libm | 1.27 puruspe† (2.2 ulps) |
| `gamma` | x in [8,20] | 10.4 ns | 0.684 Apple libm | 1.29 statrs† (940 ulps) |
| `gamma` | x in [20,170] | 10.3 ns | 0.689 Apple libm | 1.28 statrs† (NaN or ∞ on 29 of 16384) |
| `gamma` | x in [-8,-0.1] | 8.38 ns | – | 0.970 SciPy† (2.9 ulps) |
| `gamma` | x in [-170,-8] | 26.2 ns | – | 1.85 statrs† (4.5·10¹⁵ ulps) |
| `lgamma` | x in [0.01,0.5] | 7.52 ns | 1.53 Boost |  |
| `lgamma` | x in [0.5,1.5] | 4.05 ns | – | 0.664 puruspe† (2·10⁹ ulps) |
| `lgamma` | x in [1.5,3] | 4.22 ns | 0.801 Boost | 0.966 Apple libm† (1.6 ulps) |
| `lgamma` | x in [3,8] | 3.64 ns | 0.424 libm | 0.651 Apple libm† (1.7 ulps) |
| `lgamma` | x in [8,100] | 5.22 ns | – | 1.000 Apple libm† (2 ulps) |
| `lgamma` | x in [100,1e6] | 5.24 ns | 1.06 Apple libm | 1.20 SciPy† (1.9 ulps) |
| `lgamma` | x in [-10,-0.6] | 28.2 ns | – | 1.73 SciPy† (980 ulps) |
| `lgamma_sign` | x in [0.5,8] | 5.17 ns | – | 0.760 Apple libm† (2.6 ulps) |
| `lgamma_sign` | x in [-10,-0.6] | 29.7 ns | – | 1.52 special† (2.6·10³ ulps) |
| `digamma` | x in [0.01,1] | 4.98 ns | – | 1.74 Boost† (2.5 ulps) |
| `digamma` | x in [1,8] | 5.26 ns | – | 0.912 Boost† (1.7 ulps) |
| `digamma` | x in [8,1000] | 4.11 ns | 1.21 Boost |  |
| `digamma` | x in [-10,-0.1] | 14.4 ns | – | 1.75 Boost† (5·10³ ulps) |
| `polygamma` | n=1, x in [0.1,8] | 6.09 ns | 2.33 Boost |  |
| `polygamma` | n=1, x in [8,1000] | 2.73 ns | 1.27 Boost |  |
| `polygamma` | n=1, x in [-5,-0.1] | 20.2 ns | – | 1.93 Boost† (2.9 ulps) |
| `polygamma` | n=3, x in [0.1,20] | 68.2 ns | – | 0.752 GSL† (4.9 ulps) |
| `polygamma` | n=30, x in [0.5,20] | 220 ns | – | 2.41 GSL† (32.9 ulps) |
| `polygamma` | n=3, x in [-5,-0.1] | 313 ns | – | 2.00 Boost† (10.3 ulps) |
| `beta` | a,b in [0.1,10] | 47.1 ns | – | 1.98 puruspe† (1.3·10⁵ ulps) |
| `lbeta` | a,b in [0.1,10] | 48.9 ns | – | 2.22 statrs† (1.3·10⁵ ulps) |
| `beta` | a,b in [10,1e4] log-uniform | 79.9 ns | – | 2.96 SciPy† (1.6·10⁵ ulps) |
| `lbeta` | a,b in [10,1e4] log-uniform | 63.5 ns | – | 3.93 special† (1.5·10³ ulps) |
| `beta` | a in [0.1,10], b in [10,1e4] | 61.0 ns | – | 2.58 puruspe† (2.3·10⁵ ulps) |
| `lbeta` | a in [0.1,10], b in [10,1e4] | 52.7 ns | – | 2.59 special† (3.8·10⁶ ulps) |
| `beta` | a in [-10,-0.1], b in [0.1,10] | 90.1 ns | – | 2.75 SciPy† (1.9·10⁴ ulps) |
| `lbeta` | a in [-10,-0.1], b in [0.1,10] | 82.9 ns | – | 2.20 SciPy† (5.5·10⁴ ulps) |
| `factorial` | x integer in [0,170] | 0.510 ns | 0.990 Boost |  |
| `factorial` | x real in [-0.9,170] | 15.9 ns | – | 1.19 GSL† (310 ulps) |
| `binomial` | n,k integers, 0<=k<=n<=60 | 1.22 ns | 0.356 GSL | 0.567 Boost† (2.1 ulps) |
| `binomial` | n integer in [1e3,1e6] log-uniform, k integer in [0,n] | 69.9 ns | – | 3.48 SciPy† (1.4·10⁶ ulps) |
| `binomial` | n,k real in [0,60] | 94.0 ns | – | 1.73 SciPy† (2.2·10⁴ ulps) |
| `pochhammer` | x in [0.1,20], n integer in [0,30] | 11.9 ns | – | 0.826 Boost† (38.3 ulps) |
| `pochhammer` | x in [0.1,20], n real in [-0.9,30] | 37.6 ns | – | 1.73 Boost† (34.7 ulps) |
| `pochhammer` | x in [-20,0], n integer in [0,30] | 11.8 ns | – | 0.719 SciPy† (5.8 ulps) |
| `lgamma`, complex | \|z\| < 40 (re,im uniform in the disc) | 139 ns | – | 3.18 SciPy† (48.7 ulps) |
| `gamma`, complex | \|z\| < 40 (re,im uniform in the disc) | 201 ns | – | 3.28 SciPy† (300 ulps) |
| `lgamma`, complex | \|z\| in [1e3,1e4] | 142 ns | 3.98 SciPy |  |
| `gamma`, `f32` | x in [0.5,1.5] | 4.27 ns | 0.597 Apple libm |  |
| `gamma`, `f32` | x in [1.5,8] | 7.05 ns | 0.646 Apple libm |  |
| `gamma`, `f32` | x in [-8,-0.1] | 8.71 ns | 0.453 Apple libm |  |
| `lgamma`, `f32` | x in [0.5,8] | 4.61 ns | 0.611 Apple libm | 0.721 libm† (1.2 ulps) |
| `lgamma`, `f32` | x in [8,1e6] | 5.41 ns | 1.03 Apple libm | 1.23 libm† (1.2 ulps) |
| `lgamma`, `f32` | x in [-10,-0.6] | 29.1 ns | 1.33 Apple libm | 1.86 libm† (6.4·10³ ulps) |
| `digamma`, `f32` | x in [1,8] | 3.45 ns | 1.10 Boost |  |

The incomplete gamma and beta functions and their inverses:

| function | arguments | `a` | within the bound | faster, beyond the bound |
| --- | --- | ---: | --- | --- |
| `gamma_p` | a in [0.01,1], x in [0,2] | 95.8 ns | – | 2.11 GSL† (14 ulps) |
| `gamma_q` | a in [0.01,1], x in [0,2] | 96.0 ns | – | 0.988 puruspe† (3.3·10⁷ ulps) |
| `gamma_p` | a in [1,20], x in a*[0.5,1.5] | 87.1 ns | – | 1.71 special† (99.2 ulps) |
| `gamma_q` | a in [1,20], x in a*[0.5,1.5] | 87.4 ns | – | 1.45 SciPy† (70.1 ulps) |
| `gamma_p` | a in [20,200], x in a+-3sqrt(a) | 43.1 ns | – | 0.785 Boost† (80.7 ulps) |
| `gamma_q` | a in [20,200], x in a+-3sqrt(a) | 43.9 ns | – | 0.806 Boost† (23.7 ulps) |
| `gamma_p` | a in [200,1e4] log-uniform, x in a+-3sqrt(a) | 43.6 ns | – | 0.560 SciPy† (22.9 ulps) |
| `gamma_q` | a in [200,1e4] log-uniform, x in a+-3sqrt(a) | 43.7 ns | – | 0.561 SciPy† (20.3 ulps) |
| `gamma_p` | a in [1e4,1e6] log-uniform, x in a+-3sqrt(a) | 43.5 ns | – | 0.743 SciPy† (16.6 ulps) |
| `gamma_q` | a in [1e4,1e6] log-uniform, x in a+-3sqrt(a) | 44.1 ns | – | 0.770 SciPy† (17 ulps) |
| `gamma_p` | a integer in [1,20], x in a*[0.5,1.5] | 48.1 ns | – | 1.49 Boost† (12.9 ulps) |
| `gamma_q` | a integer in [1,20], x in a*[0.5,1.5] | 47.7 ns | – | 1.42 Boost† (6.6 ulps) |
| `gamma_p` | a half-integer in [0.5,19.5], x in a*[0.5,1.5] | 60.7 ns | – | 1.39 Boost† (11.9 ulps) |
| `gamma_q` | a half-integer in [0.5,19.5], x in a*[0.5,1.5] | 58.8 ns | – | 1.31 Boost† (7.4 ulps) |
| `gamma_p` | a in [1,100], x in a*([0.1,0.5] or [2,5]) | 71.2 ns | – | 1.74 special† (4.5·10¹⁵ ulps) |
| `gamma_q` | a in [1,100], x in a*([0.1,0.5] or [2,5]) | 71.6 ns | – | 1.51 SciPy† (640 ulps) |
| `gamma_p` | a in [0.01,1], x in [2,30] | 83.8 ns | 1.53 SciPy | 1.86 special† (13.5 ulps) |
| `gamma_q` | a in [0.01,1], x in [2,30] | 83.9 ns | – | 1.56 SciPy† (21.8 ulps) |
| `gamma_p_inv` | a in [0.01,1], p in (0,1) | 373 ns | – | 1.64 Boost† (250 ulps) |
| `gamma_q_inv` | a in [0.01,1], q in (0,1) | 372 ns | – | 1.55 Boost† (410 ulps) |
| `gamma_p_inv` | a in [1,20], p in (0,1) | 307 ns | – | 1.56 Boost† (5.4 ulps) |
| `gamma_q_inv` | a in [1,20], q in (0,1) | 307 ns | – | 1.51 Boost† (15.7 ulps) |
| `gamma_p_inv` | a in [20,1e4] log-uniform, p in (0,1) | 169 ns | – | 0.707 Boost† (9.2 ulps) |
| `gamma_q_inv` | a in [20,1e4] log-uniform, q in (0,1) | 169 ns | – | 0.698 Boost† (7 ulps) |
| `gamma_p_inv` | a in [1,20], p in [1e-300,1e-10] log-uniform | 206 ns | – | 1.27 Boost† (4.6 ulps) |
| `gamma_q_inv` | a in [1,20], q in [1e-300,1e-10] log-uniform | 237 ns | 1.98 Boost |  |
| `gamma_p_inv` | a in [0.01,1], p in [1e-300,1e-10] log-uniform | 178 ns | – | 6.34 puruspe† (5.3·10¹⁷ ulps) |
| `gamma_q_inv` | a in [0.01,1], q in [1e-300,1e-10] log-uniform | 346 ns | 3.10 Boost |  |
| `beta_inc` | a,b in [0.1,1], x in (0,1) | 135 ns | – | 1.39 Boost† (21.7 ulps) |
| `beta_inc_c` | a,b in [0.1,1], x in (0,1) | 134 ns | – | 1.40 Boost† (22.6 ulps) |
| `beta_inc_inv` | a,b in [0.1,1], p = I_x(a,b), x in (0,1) | 616 ns | – | 1.22 SciPy† (26.8 ulps) |
| `beta_inc_c_inv` | a,b in [0.1,1], q = 1-I_x(a,b), x in (0,1) | 605 ns | – | 1.21 SciPy† (32.8 ulps) |
| `beta_inc` | a,b in [1,10], x in (0,1) | 182 ns | – | 1.85 puruspe† (8.6·10⁵ ulps) |
| `beta_inc_c` | a,b in [1,10], x in (0,1) | 176 ns | – | 1.00 Boost† (14.6 ulps) |
| `beta_inc_inv` | a,b in [1,10], p = I_x(a,b), x in (0,1) | 667 ns | – | 1.58 special† (960 ulps) |
| `beta_inc_c_inv` | a,b in [1,10], q = 1-I_x(a,b), x in (0,1) | 666 ns | – | 0.934 Boost† (29.6 ulps) |
| `beta_inc` | a,b in [10,100], x mean+-3sd | 205 ns | – | 1.37 Boost† (130 ulps) |
| `beta_inc_c` | a,b in [10,100], x mean+-3sd | 207 ns | – | 1.39 Boost† (120 ulps) |
| `beta_inc_inv` | a,b in [10,100], p = I_x(a,b), x mean+-3sd | 731 ns | – | 1.95 Boost† (120 ulps) |
| `beta_inc_c_inv` | a,b in [10,100], q = 1-I_x(a,b), x mean+-3sd | 732 ns | – | 1.98 Boost† (130 ulps) |
| `beta_inc` | a,b in [100,1e4] log-uniform, x mean+-3sd | 327 ns | – | 1.56 Boost† (3.6·10³ ulps) |
| `beta_inc_c` | a,b in [100,1e4] log-uniform, x mean+-3sd | 324 ns | – | 1.56 SciPy† (3.8·10³ ulps) |
| `beta_inc_inv` | a,b in [100,1e4] log-uniform, p = I_x(a,b), x mean+-3sd | 980 ns | – | 2.21 Boost† (370 ulps) |
| `beta_inc_c_inv` | a,b in [100,1e4] log-uniform, q = 1-I_x(a,b), x mean+-3sd | 971 ns | – | 2.22 Boost† (300 ulps) |
| `beta_inc` | a,b in [1e5,1e7] log-uniform, x mean+-3sd | 178 ns | – | 1.46 puruspe† (4.5·10¹⁵ ulps) |
| `beta_inc_c` | a,b in [1e5,1e7] log-uniform, x mean+-3sd | 181 ns | – | 0.221 SciPy† (3.3·10⁶ ulps) |
| `beta_inc_inv` | a,b in [1e5,1e7] log-uniform, p = I_x(a,b), x mean+-3sd | 690 ns | – | 0.349 statrs† (1.4·10¹¹ ulps) |
| `beta_inc_c_inv` | a,b in [1e5,1e7] log-uniform, q = 1-I_x(a,b), x mean+-3sd | 684 ns | – | 0.243 Boost† (1.1·10⁴ ulps) |
| `beta_inc` | a in [20,1e3], b in [0.1,1], x mean+-3sd | 272 ns | – | 1.31 statrs† (1.7·10⁵ ulps) |
| `beta_inc_c` | a in [20,1e3], b in [0.1,1], x mean+-3sd | 276 ns | – | 1.24 Boost† (5.6 ulps) |
| `beta_inc_inv` | a in [20,1e3], b in [0.1,1], p = I_x(a,b), x mean+-3sd | 1.34 µs | 1.53 Boost | 1.85 statrs† (4.5·10⁴ ulps) |
| `beta_inc_c_inv` | a in [20,1e3], b in [0.1,1], q = 1-I_x(a,b), x mean+-3sd | 1.32 µs | 1.51 SciPy |  |
| `gamma_p`, `f32` | a in [1,20], x in a*[0.5,1.5] | 87.2 ns | 1.09 Boost |  |

### Where the references are faster

Every loss of the family against a reference within the bound:

- **`ψ′`** on `[0.1, 8]` and `[8, 1000]` (2.33 and 1.27 against Boost, which is within 1.2 ulps of `a`), **`ψ`** on `[8, 1000]` (1.21, Boost), **`ln Γ`** on `[0.01, 0.5]` (1.53, Boost) and `[100, 10⁶]` (1.06, Apple's libm): the double-double pieces that hold `ψ′` and `ψ` to 0.75 ulp and `ln Γ` to 0.6 (the exact linear term, the low parts carried), against plain `f64` rationals within about one ulp.
- **Complex `ln Γ`** for `|z|` in `[10³, 10⁴]` (3.98 against SciPy, within 2.0 ε of `a`): complex double-double throughout, the argument and the logarithm to 2⁻¹⁰⁰ for the componentwise bound.
- **`gamma_q_inv` at tiny `q`** (`q` in `[10⁻³⁰⁰, 10⁻¹⁰]`: 3.10 for `a` in `[0.01, 1]`, 1.98 for `a` in `[1, 20]`, against Boost): the guesses in the far tail need a second evaluation of `Q`, and each evaluation carries the double-double prefix.
- **`gamma_p`** for `a` in `[0.01, 1]`, `x` in `[2, 30]` (1.53 against SciPy): Legendre's continued fraction with its last steps and the prefix in double-double.
- **The beta inverses beside a large `a` and a small `b`** (`a` in `[20, 10³]`, `b` in `[0.1, 1]`: 1.53 against Boost, 1.51 against SciPy): BGRAT's power terms computed again for the derivative at each step.
- **The `f32` forms**: `lgamma` on `[−10, −0.6]` (1.33 against Apple's `lgammaf`), `digamma` on `[1, 8]` (1.10) and `gamma_p` (1.09, against Boost in `double`): an `f32` argument runs the `f64` kernels, double-double steps and all, and is rounded once.

## The error functions

`erf`, `erfc` and `erfcx` take `f32`, `f64`, `Complex<f32>` and `Complex<f64>` (the trait `Scalar`); `erf_inv`, `erfc_inv`, `fresnel` and `dawson` take `f32` and `f64` (`Real`), and `faddeeva` a `Complex<f32>` or `Complex<f64>`.

| Function | Definition | Domain and special values |
| --- | --- | --- |
| `erf(x)` | `2/√π ∫₀^x e^(−t²) dt` | odd; `±1` at `±∞` (and from `|x| ≥ 6`, to the last bit) |
| `erfc(x)` | `1 − erf(x)`, without the cancellation | `2` at `−∞` (from `x ≤ −6`), `0` from `x ≥ 27.3`; subnormal from 26.55 |
| `erfcx(x)` | `e^(x²)·erfc(x)` | `0` at `+∞`, subnormal from `2.5·10³⁰⁷`; `+∞` below `−26.62873571375149`, where `2e^(x²)` overflows |
| `erf_inv(x)` | the `y` with `erf(y) = x` | `[−1, 1]`, `±∞` at `±1`, NaN outside |
| `erfc_inv(x)` | the `y` with `erfc(y) = x` | `[0, 2]`, `+∞` at 0, `−∞` at 2, NaN outside; every subnormal `x` |
| `dawson(x)` | `F(x) = e^(−x²) ∫₀^x e^(t²) dt` | odd, `±0` at `±0` and at `±∞`; `F(x) ≈ 1/(2x)`, subnormal from `2.2·10³⁰⁷` |
| `fresnel(x)` | `(S(x), C(x))`, `S = ∫₀^x sin(πt²/2) dt`, `C = ∫₀^x cos(πt²/2) dt` | odd; `(±1/2, ±1/2)` at `±∞` and from `|x| ≥ 2⁵⁴` |
| `faddeeva(z)` | `w(z) = e^(−z²)·erfc(−iz)` | `0` as `|z| → ∞` with `Im z` bounded below (below the real axis too, along a line); `erfcx(y)` on the imaginary axis, `+∞` at `−i∞`; NaN for a NaN part (with the zero imaginary part kept at `±0 + NaN·i`, as `erfcx` keeps it at `NaN ∓ 0i`), and for `Im z = −∞` off the imaginary axis |
| `erf(z)`, `erfc(z)`, `erfcx(z)` | as above, `erfcx(z) = w(iz)` | entire; `erf(±∞ + iy) = ±1` for a finite `y` |

A NaN argument gives NaN; a finite one never does. Where a part of a complex value overflows it is the infinity of its sign, and where it underflows it is 0 (the other part can be finite; see the measure in *Error bounds*): the size of `e^(∓z²)` is kept as an exact power of 2 and applied once, saturating, so that a part far smaller than `|w|` overflows where it must and no sooner, and its phase is computed exactly, so the signs are right (on 400,000 arguments with `|e^(∓z²)| > e²⁰⁰⁰` every part is infinite, and the signs of 100,000 of them agree with Arb). The complex functions have `erf(conj z) = conj erf(z)`, `erf(−z) = −erf(z)`, `erfc(conj z) = conj erfc(z)`, `erfcx(conj z) = conj erfcx(z)`, `w(−conj z) = conj w(z)` and `erfcx(z) = w(iz)` bit for bit, zero parts and infinities included (`w(±∞ + iy)` is 0 with the signs of the parts of `i/(√π z)`); the sign of a NaN is not specified. On the axes they are the real functions: `erf(x + 0i) = erf(x)`, and so for `erfc` and `erfcx`; `w(x) = e^(−x²) + 2i/√π·F(x)`; `w(iy) = erfcx(y)` for every `y`, its zero imaginary part with the sign of `Re z`; `erf(iy) = i·erfi(y)` and `erfc(iy) = 1 − i·erfi(y)`.

```rust
use a::math::complex::Complex;
use a::math::special::{dawson, erf, erfc, faddeeva};

assert!((erf(0.5f64) - 0.520_499_877_813_046_5).abs() < 2e-16);
assert_eq!(dawson(-0.0f64).to_bits(), (-0.0f64).to_bits());
// Below the real axis next to the diagonal |w| is 2, its phase 2x² reduced exactly.
let w = faddeeva(Complex::new(1.1e12f64, -1.1e12));
assert!((w.re + 0.529_424_841_179_702_1).abs() < 1e-15);
assert!((w.im + 1.928_654_799_994_319_7).abs() < 1e-15);
// Re w is the Voigt function, e^(−x²) on the real axis, and right on its own next to it.
let v = faddeeva(Complex::new(7.0f64, 1e-30)).re;
assert!((v / 5.242_885_663_482_323_4e-22 - 1.0).abs() < 1e-15);
// Next to the imaginary axis each part of erf is right on its own, the real one of the size of x.
let e = erf(Complex::new(1e-300f64, 7.0));
assert!((e.re / 2.152_210_136_834_501_3e-279 - 1.0).abs() < 1e-15);
// Beyond the range of a double: the infinity of each part's sign, though the factor e^(−z²)
// multiplies is of the size of 1/|z|.
let e = erfc(Complex::new(1.0f64, 1e200));
assert_eq!((e.re, e.im), (f64::INFINITY, f64::NEG_INFINITY));
let e = erf(Complex::new(1.0f64, 1e306));
assert_eq!((e.re, e.im), (f64::INFINITY, f64::NEG_INFINITY));
```

### Error bounds

In ulps of the result for the real functions, in units of ε = 2⁻⁵² for the complex ones. A **bound** is a worst case the code cannot exceed: where the result ends in a short chain of roundings (the polynomials below 1/2, the Fresnel integrals, the pieces of `erfcx` and `dawson` below 8, the last step of the inverses, and every path of the complex functions) it is the worst case of a first-order error analysis of those roundings, each taken at its largest at once, with the polynomials' own errors measured with mpmath (`data/erf_bounds.py` computes them; for the complex functions its largest over a region is found by a search, and the bound is the largest of several runs with 3 % for that search); elsewhere (marked †) it is the largest error of a dense search with a margin of at least 10 %: there the result is a double-double rounded once (`erf` and `erfc` from 1/2, `erfcx` and `dawson` beyond their pieces), where the analysis would give the last rounding and a few hundredths. The **largest found** are those of searches against Arb (FLINT 3.6): 2 to 40 million random arguments a window and a local search around the worst for the real functions, a million a region and eight rounds of local search for the complex ones; the roundings seldom align at once, so they are below the worst cases, the complex ones far below. `tests/math/special/erf.rs` holds the 12,283 cases of `data/gen_erf.py` (mpmath at 256 bits and more), which include the worst arguments found, to these bounds, and to tighter figures above the largest errors found where a bound is far above them (the complex functions to 8 ε, the Fresnel integrals below 8 to 1.0 ulp but `C` from 1 to 1.1), so that a change that loses accuracy fails.

| Function | Region | `f64` bound | largest found | `f32` bound |
| --- | --- | ---: | ---: | ---: |
| `erf` | `|x| < 1/2` | 0.86 | 0.760 | 0.501 |
| | `1/2 ≤ |x|` | 0.8 † | 0.518 | 0.501 |
| `erfc` | `|x| < 1/2` | 0.92 | 0.818 | 0.501 |
| | `x ≤ −1/2`, `1/2 ≤ x` (subnormal from 26.55) | 0.85 † | 0.613 | 0.501 |
| `erfcx` | `|x| < 1/2` | 0.78 | 0.669 | 0.501 |
| | `1/2 ≤ x < 8` | 0.7 | 0.628 | 0.501 |
| | `x ≤ −1/2`, `8 ≤ x` (subnormal from `2.5·10³⁰⁷`) | 0.7 † | 0.522 | 0.501 |
| `erf_inv` | `|x| ≤ 1/2` (subnormal `x` too) | 0.9 | 0.785 | 0.501 |
| | `1/2 < |x| < 1`, to `1 − 2⁻⁵³` | 0.98 | 0.850 | 0.501 |
| `erfc_inv` | `(0, 2)`, subnormal `x` too | 0.98 | 0.859 | 0.501 |
| `dawson` | `|x| < 1/2` | 1.07 | 0.823 | 0.501 |
| | `1/2 ≤ |x| < 8` | 0.86 | 0.765 | 0.501 |
| | `8 ≤ |x|` (subnormal from `2.2·10³⁰⁷`) | 0.86 † | 0.545 | 0.501 |
| `fresnel` S, C | `|x| < 1` | 1.14, 1.34 | 0.894, 0.934 | 0.501 |
| | `1 ≤ |x| < 8` | 1.12, 1.23 | 0.930, 1.014 | 0.501 |
| | `8 ≤ |x|` | 0.85, 0.79 | 0.723, 0.719 | 0.501 |

From 2⁹⁶⁰ `erfcx` and `dawson` are correctly rounded (the tests check their bits there), subnormal results included. A reference below about 2⁻¹⁰⁰⁰ has its low word rounded to the least subnormal; the tests allow for that rounding and no more (half an ulp for a subnormal result). The `f32` forms are the `f64` values rounded once: correctly rounded but for the halfway cases of the `f64` error, about one argument in 2²⁹ (0.5000 seen on 9 million arguments; that the `f32` result is the `f64` one rounded was checked on all 2³² arguments of every function).

The bounds of the paths that end in a polynomial are above those stated before (the review of 2026-11-21): those were the largest errors of samples with a margin, and searches exceeded two of them (`dawson(0.49871859084367415)` 0.822 ulps against 0.8, `S(0.9927428287950955)` 1.008 against 1.0). A polynomial in plain doubles adds to the last rounding the roundings of its own terms, a part of the result's size: `x·q` of `erf` is 4 to 11 % of `erf(x)`, `x³·Q` of `dawson` up to 18 %, `x⁵·P` of `C` up to 28 %; and with their coefficients rounded to doubles the polynomials are approximations to 2⁻⁵⁵·⁷ (that of `erf`, against 2⁻⁵⁷·⁴ before the rounding), 2⁻⁵⁶·⁵ (`dawson`) and 2⁻⁵⁵·⁷ (`C`). For the inverses the analysis takes the residual's error, that of `erf_q` and its argument's rounding, through the Newton or Halley step: up to 0.98 ulp next to `y = 1/2`. Since the review, `x⁴` of the Fresnel integrals below 1 is rounded once from the exact `x²` (it was the square of the rounded `x²`, with three times the error), which brought their worst cases from 1.50 and 1.83 to 1.14 and 1.34 ulps and the largest error of `C` found from 1.09 to 0.94. From 1 the Fresnel integrals are analysed since the review of 2026-11-22 (their bounds were a search's largest error with a margin, 1.0 and 1.1, which a denser search came within 8 % of: 1.014 at `C(1.5747143746613816)`), every rounding counted as half an ulp of its result, the tighter form of the same bound: from 8 the worst cases are below those figures, but from 1 to 8 they are above, as the roundings of `sin(πx²/2)` and `cos(πx²/2)` near `1/√2` (of the reduced argument, of the head's value and of the product: about 1 ε of them together) can align with the last rounding where `C` or `S` lies below 1/2.

The complex functions: the bound is relative to the larger of the two terms that the formula subtracts: `2e^(−z²)` and `w(−z)` below the real axis, 1 and `erfc(±z)` for `erf` from `|z| = 1`, 2 and `erfc(−z)` for `erfc` left of the imaginary axis, `2e^(z²)` and `erfcx(−z)` for `erfcx` there; where nothing cancels, relative to `|w|`. Relative to `|w|` the error is `K` = that term over `|w|` times as large: below twice the bound away from the zeros, where the term is below `2|w|`, and growing as `|w|` falls near a zero (315 ε seen next to one). The error is that of the complex value, `|computed − exact|`; a result below 2⁻¹⁰²² is measured in units of the least subnormal. Where `|w|` is beyond the largest double, a part that overflows is the infinity of its sign, and a finite part next to it is measured as the others are, against the scale (beyond the largest double too): it is accurate to the bound times `ε·|w|` in absolute terms, and so can be far off relative to itself where it is far below `|w|` (`Im erf(−445.62 − 446.43i)`, `4.7·10⁻⁶` of `|w|`, is 1.9·10⁵ ulps of itself off, 0.86 ε of the scale). Below the real axis (and for `erfcx` left of the imaginary axis) the value is `2e^(∓z²)` but for a part below 1, and such a part is right on its own there, as its phase is reduced exactly (1.90 ulps of itself found on 240,000 of them; the tests hold it to the bound), as it is in the strips next to the axes below.

| Function | Region | bound | largest found |
| --- | --- | ---: | ---: |
| `faddeeva` | upper half-plane, `|z| < 10` | 19.6 | 6.28 |
| | upper half-plane, `|z|` from 10 to `1.8·10³⁰⁸` | 4.3 | 2.36 |
| | lower half-plane, `[−10, 10] × [−10, 0)` | 19 | 5.90 |
| | lower half-plane beyond it | 6.2 | 2.15 |
| `erf(z)` | `|z| < 1` | 3.9 | 1.48 |
| | `[−10, 10]²`, `|z| ≥ 1` | 27 | 6.33 |
| | beyond it | 9.4 | 2.67 |
| `erfc(z)` | `[−10, 10]²` | 29 | 6.70 |
| | beyond it | 12.2 | 2.65 |
| `erfcx(z)` | `[−10, 10]²` | 19.6 | 6.27 |
| | beyond it, `Re z > 0` | 4.3 | 2.10 |
| | beyond it, `Re z < 0` | 6.2 | 1.97 |

The bounds are those of the analysis because a search cannot give one: its largest error grows with the effort (4.8 on a million random arguments of a region, 5.2 on a survey of 16 million, 6.2 and 6.7 after local searches), while the analysis bounds every argument it covers: it was maximised to `|z| = 10⁴` in every direction, to 10³⁰⁰ in the upper half-plane (and for `erfcx` right of the imaginary axis) and to 10¹⁵⁰ next to the diagonals, where `e^(∓z²)` has a size; beyond, the table rests on measurements (on the diagonals to `f64::MAX`, at most 1.6 ε). Its largest over a region is found by sampling and a local search too (it peaks at the seams of the trapezoidal rule's nodes in the boxes, and beyond them at many local maxima of different heights): runs of different budgets and seeds agree to 1 % in the boxes and to 7 % beyond them (`erfc`: 10.95 to 11.79), hence the 3 % over the largest, which a run had found above the bound stated before (`Re w` below `|x| = 10`: 31.83 against 31.8). They are above the 5.5 stated before the review, a sample maximum, for the reason the largest errors are above it: the trapezoidal rule adds 27 terms, each with the roundings of its powers of `e^(∓t₀)` (up to 13 products), its two denominators, a division and its accumulation, and next to `Re z = 1` (or `Im z = −1` for `erfc` and `erfcx`), where the node nearest 0 is a quarter from it, they are the largest; `erfc` and `erfcx` add the products with `e^(∓z²)`. The `Complex<f32>` forms are the `Complex<f64>` values rounded once a part: within 0.5 ε₃₂ = 2⁻²⁴ of `|w|` (bound 0.501; 0.4995 seen against mpmath on 15,000 arguments, 0.5000 against Arb on 3.4 million), but where the `f64` error shows, so close to a zero that the terms the formula subtracts exceed `10⁴·|w|`, and for a part below 2⁻¹²⁶, which is rounded to the subnormals of `f32`.

Only the error relative to `|w|` (or to the scale above) is bounded in general, not that of each part: next to an axis the smaller part is only as accurate as `ε·|w|` in absolute terms (`Im w(1.13·10⁻¹² + 0.048i)` has three correct digits), so complex-step differentiation does not work with these functions there. Three strips are bounded part by part: `Re w(x + iy)` next to the real axis (`0 < y ≤ 10⁻²`), the Voigt function, within 33 ulps of itself below `|x| = 10` and 10.5 ulps beyond (the trapezoidal rule's terms, all positive here; 11.8 and 4.3 found, the tests hold the data to 14 and 10); `erf` and `erfc` next to the imaginary axis, `|Re z|·max(|Im z|, 1) < 2⁻¹⁴` (`erf` from `|z| = 1`): `Re erf` within 0.54 ulps of itself, the imaginary parts within 3.7 ulps and `Re erfc` within 0.54 ε of `max(|Re|, 1)` (0.51, 1.87 and 0.43 found); and `w` below the real axis next to the imaginary one, `|Re z|·max(−Im z, 1) < 2⁻³⁰`, and so `erfcx` left of the imaginary axis next to the real one: the real part within the bound of the real `erfcx`, the imaginary one within 3.1 ulps of itself (0.65 and 2.12 found), also where the real part overflows and the imaginary one does not.

### Method

* **`exp_dd`.** `e^a` of a double-double `a` to about 2⁻⁶⁰ (for `|a|` to 2839): `a = k·ln 2/64 + r` with `ln 2/64` in two parts (the first of 35 bits, so that `k·L₁` is exact), `e^r − 1` by its Taylor series to `r⁶` (`|r| ≤ ln 2/128`), and a table of the 64 values `2^(j/64)` as double-doubles; the result is a double-double and a power of 2 kept apart, so that nothing over- or underflows before the final rounding, which is the only one (and once also for a subnormal result).
* **The `f32` form of `erf`** on `1/2 ≤ |x| < 4`: seven polynomials of degree 12 in `|x| − c` on the halves of `[1/2, 4)` (Chebyshev interpolants, mpmath at 40 digits), picked by an index, good to `2⁻⁵⁰` in `f64` and rounded once to the `f32` (checked against the `f64` path at every `f32` of `[−4.5, 4.5]`); elsewhere the `f64` path.
* **`erf`, `erfc` for `|x| < 1/2`.** `erf(x) = x + x·q(x²)`, `q` a polynomial of degree 9 (a minimax fit to 2⁻⁵⁷·⁴, 2⁻⁵⁵·⁷ with its coefficients as doubles, relative to `erf(x)/x`), rounded once by a fused multiply-add; `erfc(x) = (1 − x) − x·q(x²)` with `1 − x` exact as two doubles.
* **`erfcx` for `x ≥ 1/2`.** For `1/2 ≤ x < 8`, 32 minimax polynomials of degree 10 in `x − c`, one per eighth of a binade, picked from the bits of `x` (no division, no branch), each with its constant term as a double-double (2⁻⁵⁶·⁸ after the rounding of the coefficients); for `x ≥ 8`, `x·erfcx(x)` as a polynomial of degree 9 in `1/x²` (2⁻⁶⁰), times `1/x` as a double-double; from 2⁹⁶⁰, `1/(√π x)` with `1/x` of `x·2⁻⁵¹²` as a double-double and the result scaled back with one rounding, as beyond 2¹⁰¹⁵ the low words of the plain form would be subnormal. For `|x| < 1/2`, 8 polynomials of degree 11 on eighths of `[−1/2, 1/2)`; for `x ≤ −1/2`, `2e^(x²) − erfcx(−x)` with `e^(x²)` from `exp_dd`.
* **`erfc`, `erf` for `|x| ≥ 1/2`.** `erfc(x) = e^(−x²)·erfcx(x)`, `x²` exact as two doubles, `e^(−x²)` by `exp_dd` and the product as a double-double; `erf = 1 − erfc` and `erfc(−x) = 2 − erfc(x)` add the double-double to 1 or 2 exactly before the one rounding.
* **`erf_inv`, `erfc_inv`.** For `|x| ≤ 1/2`, a polynomial of degree 7 in `x²` to 2⁻³⁴ and one Newton step on the residual `erf(y₀) − x = (y₀ − x) + y₀·q(y₀²)`, exact but for its last rounding; tiny `x` as `x·√π/2` in two parts, rounded once. For `x > 1/2` (and `erfc_inv(c)`, `c ≤ 1/2`), `erfc_inv(1 − x)` with `1 − x` exact: 12 polynomials of degree 6 to 8 in `s = √(−ln c)` (two a binade of `s`, to 2⁻³¹), then one Halley step on `(erfc(y₀) − c)·e^(y₀²) = erfcx(y₀) − c·e^(y₀²)`, both terms as double-doubles and `c·2^k` scaled exactly, which holds for every subnormal `c`; `erfc_inv(c) = −erfc_inv(2 − c)` above 1.5 and `erf_inv(1 − c)` between.
* **`dawson`.** `x + x³·Q(x²)` (degree 8) for `|x| < 1/2`; 32 polynomials of degree 11 on eighths of binades for `1/2 ≤ |x| < 8` (2⁻⁵⁵); `x·F(x)` in `1/x²` (degree 9) beyond, and from 2⁹⁶⁰ `1/(2x)` as for `erfcx`.
* **`fresnel`.** For `|x| < 1`, `C = x + x⁵·P(x⁴)` and `S = x³·Q(x⁴)` (degree 8, `x³` and `Q` as double-doubles, `x⁴` rounded once from the exact `x²`; below 2⁻³³⁰, `S = (π/6)x³` scaled by 2³³⁰ and back with one rounding). From 1, `C = 1/2 + f·sin(πx²/2) − g·cos(πx²/2)` and `S = 1/2 − f·cos − g·sin` with the auxiliary functions `f` and `g` from 24 polynomials of degree 11 on eighths of the binades `[1, 8)` and as `x·f`, `x³·g` in `1/x⁴` (degrees 4 and 5) from 8; `x²` is exact as two doubles and each is reduced modulo 4 exactly, so the phase is right for any `x`, and `sin(πr/2)`, `cos(πr/2)` for `|r| ≤ 1/2` are minimax polynomials; the products and the sum with 1/2 are double-doubles.
* **`faddeeva` and the complex functions.** In the closed upper half-plane, by `w(−x + iy) = conj w(x + iy)` for `x ≥ 0`: for `|z| ≥ 10` the asymptotic series `i/(√π z)·Σ (2n − 1)!!/(2z²)ⁿ` to `n = 13`, plus `e^(−x²)` in the real part within `10⁻²⁰` of the real axis, and from `2¹⁰²⁰` its first term alone with `1/z` of `z·2⁻¹⁰` (Smith's denominator would overflow) scaled back once; for `|z| < 10` the trapezoidal rule of `w(z) = (i/π) ∫ e^(−t²)/(z − t) dt` with the step `h = 1/2` on 27 nodes shifted to `x + (n + 1/2)h` — no node meets the pole — plus its exact pole correction `2e^(−z²)/(1 + e^(2πy/h))` for `y < π/h`, left out only where it is below 2⁻⁵⁵ of the smallest `Re w` can be (`0.0039·min(y, 1)`): where its exponent is below −45, or below `−46 + ln y` for `y < 1` (for a small `y` it is all of `Re w` but a part of the size of `y`). The aliasing error is `e^(−π²/h²)`, below 2⁻⁵⁴ of `|w|` (0.3 ε measured in exact arithmetic); every node costs one real division (two nodes share one), and `e^(−(t₀ ± kh)²)` comes from `e^(−t₀²)`, `e^(∓t₀)` and `e^(−k²/4)` by products. On the real axis `w = e^(−x²) + 2i/√π·F(x)` and on the imaginary one `erfcx(y)`, from the real functions (below the real axis too). In the lower half-plane `w(z) = 2e^(−z²) − w(−z)`. The size of `e^(−z²)` comes from `y² − x²` as a double-double of exact squares and `exp_dd`, its power of 2 exact wherever a part of a result can be finite (`|y² − x²| < 1460`: beyond, a part of the factor it multiplies, at least 2⁻¹⁰⁷⁴ and at most 4, over- or underflows whatever it is) and applied to each part once, saturating; where a square overflows, `(|y| − |x|)(|y| + |x|)` is 0 or beyond `±2⁹⁷¹`. Its phase `−2xy` is reduced modulo `π/2` exactly, below `2²⁰` by Cody and Waite's method with a three-part `π/2`, and beyond by Payne and Hanek's from `2xy = m·2^e`, `m` the product of the two significands (106 bits), and the 256 bits of `2/π` at the units of `m·2^e`, taken from a table of 2,304 bits that covers every pair of doubles, also where `2xy` overflows; then fdlibm's kernels. So the size and the phase of the exponential are right to the last bits for any `|z|`, as far as a double holds them. Then `erfcx(z) = w(iz)`, `erfc(z) = e^(−z²)·w(iz)` for `x ≥ 0` (`1 − i·erfi(y)` on the imaginary axis) and `2 − erfc(−z)` for `x < 0`, and `erf(z)` by its Taylor series in `z²` (19 terms) for `|z| < 1` and `±(1 − erfc(±z))` beyond. Next to the imaginary axis, `|x|·max(|y|, 1) < 2⁻¹⁴` (`erf` from `|z| = 1`), `erf` and `erfc` come part by part from the Taylor series in `x` about `iy`: `Re erf = (2/√π)e^(y²)·x·(1 − (2y² + 1)x²/3 + (3 + 12y² + 4y⁴)x⁴/30)` as a double-double with its power of 2 apart, `Im erf = (2/√π)e^(y²)·(F(y) − y·x²·(1 − (3 + 2y²)x²/6))` (the terms left out are below 2⁻⁸⁰ of each part) and `erfc = 1 − erf`; below the real axis next to the imaginary one, `|x|·max(−y, 1) < 2⁻³⁰`, `w = erfcx(y) + 2ix·(1/√π − y·erfcx(y))` (below 2⁻⁵⁸), with `|y|·erfcx(y) = 2|y|e^(y²)` from `−y = 26` as a double and a power of 2. No standard-library transcendental function is used (but `ln` in the first guess of `erfc_inv`), so the results are the same on every target with a fused multiply-add.

The coefficients are this module's own minimax fits (a discrete Remez exchange in mpmath at 160 bits, `tables.rs` made by `gen_tables.py`); the sine and cosine kernels are fdlibm's, and the bits of `2/π` are mpmath's, the same as Arb's.

### Accuracy measured

The largest errors found of *Error bounds* are those of searches against Arb (FLINT 3.6, balls tightened until their radius is below 2⁻⁴⁰ ulp, a part settled on its own where it is measured so; asymptotic series with remainder bounds where Arb's own are not usable) on the aarch64-apple-darwin build: for an audit on 2026-11-20, 33.3 million `f64` arguments of the real functions (a million and more a region, every binade, ±300 ulps around every seam between pieces) and 72,405 more against mpmath, the two references agreeing to 6·10⁻¹⁷ ulp, and 16 million of the complex functions (a million a region); for the review of 2026-11-21, 117 million arguments of the real functions in the windows where the errors peak and local searches around the worst, and 25 million of the complex functions (17 million in local searches, 1.6 million where the exact value overflows, 2.7 million in the strips and bands the review changed); the review of 2026-11-22 ran 13.5 million of the complex functions and over 100 million arguments of the real ones (most of the largest found above are its), and after it 2.1 million of the complex functions beyond the box were measured as stated above, a finite part next to an infinite one against the scale (largest 2.65 ε; relative to themselves such parts of `erf` and `erfc` were up to 1.5·10⁵ ulps off, those below the real axis 1.90). Every complex error found is below the bound at its argument (the largest ratio is 0.73), and every real one below the bound of its region. Their scripts are not part of the repository; the analysis behind the bounds is (`python3 erf_bounds.py`, run from `data/`, prints the largest bound of each region in some minutes). On the code path without a fused multiply-add (wasm32 under Node) the errors are of the same size: before the review (100,000 to 200,000 arguments a region) `erf` 0.776, `erfc` 0.795, `erfcx` 0.664, `erf_inv` 0.821, `erfc_inv` 0.797, `dawson` 0.769, `fresnel` 0.897 and 1.050 ulps, the complex functions up to 4.52 ε of the scale; after it (40,000 to 300,000 arguments a region) `erf` 0.761, `erfc` 0.801, `erfcx` 0.655, `erf_inv` 0.804, `erfc_inv` 0.821, `dawson` 0.776, `fresnel` below 1 0.900 and 0.902 ulps, the complex boxes up to 4.17 ε, the parts of the strips next to the axes as with one, and every part infinite where the exact value overflows. The analysis there would count the unfused polynomial steps twice, and was not made. The tests pass on both paths (the second with the fused multiply-adds of the polynomials switched off in a copy of the crate, as x86_64 runs them without the `fma` feature).

The tests check 12,283 cases of `data/gen_erf.py` (run from that directory: `python3 gen_erf.py`, deterministic, a few minutes): random arguments by region, the worst arguments found with their neighbours, the diagonals and the coordinates beyond `√f64::MAX` of the complex functions, the arguments where `e^(∓z²)` is beyond any double or `|y² − x²|` is from 1400 to 1470, those where `|w|` is just beyond it (a finite part next to an infinite one), the points near their zeros, the strips next to the axes (part by part) and the `Complex<f32>` forms. A complex reference is computed at 128, 256, … more bits until two in a row agree to 110 bits a part; a complex case carries its scale as a third reference (beyond the largest double negated and divided by 2¹⁰²⁴). `cargo test --test math -- special::erf --nocapture` prints the largest error of each file.

### Measured

As for [the gamma function](#measured) (the conditions and the same-work rule are there):

| function | arguments | `a` | within the bound | faster, beyond the bound |
| --- | --- | ---: | --- | --- |
| `erf` | x in [-0.5,0.5] | 0.969 ns | 0.334 libm |  |
| `erf` | x in [0.5,3] | 4.59 ns | 0.703 SciPy |  |
| `erf` | x in [-3,3] | 4.38 ns | 0.626 SciPy |  |
| `erf` | x in [3,6] | 4.61 ns | 0.784 SciPy |  |
| `erfc` | x in [-0.5,0.5] | 1.46 ns | 0.443 libm |  |
| `erfc` | x in [0.5,3] | 4.34 ns | 0.428 libm | 0.658 SciPy† (7 ulps) |
| `erfc` | x in [-3,3] | 5.30 ns | 0.580 special | 0.619 SciPy† (6.3 ulps) |
| `erfc` | x in [3,8] | 4.31 ns | 0.421 libm | 0.787 SciPy† (18.1 ulps) |
| `erfc` | x in [8,27] | 5.15 ns | 0.493 libm | 1.04 SciPy† (2.3·10¹³ ulps) |
| `erfcx` | x in [-0.5,0.5] | 1.70 ns | – | 0.323 puruspe† (1.7 ulps) |
| `erfcx` | x in [0.5,8] | 1.44 ns | 0.451 puruspe |  |
| `erfcx` | x in [8,100] | 1.78 ns | – | 0.505 puruspe† (1.7 ulps) |
| `erfcx` | x in [-5,-0.5] | 3.94 ns | – | 0.784 puruspe† (8.8 ulps) |
| `erf_inv` | x in [-0.5,0.5] | 3.11 ns | 0.934 Boost |  |
| `erf_inv` | x in (-1,1) | 9.75 ns | 1.31 statrs |  |
| `erf_inv` | x = 1-t, t in [1e-15,0.5] log-uniform | 13.1 ns | 1.24 statrs |  |
| `erfc_inv` | x in (0,2) | 7.33 ns | 1.16 Boost |  |
| `erfc_inv` | x in [1e-300,0.5] log-uniform | 13.5 ns | 1.04 statrs | 1.21 SciPy† (2.3 ulps) |
| `dawson` | x in [-0.5,0.5] | 0.869 ns | 0.0722 GSL | 0.120 puruspe† (2·10¹⁶ ulps) |
| `dawson` | x in [0.5,8] | 1.59 ns | – | 0.128 SciPy† (2.7 ulps) |
| `dawson` | x in [8,1e4] | 1.89 ns | 0.0624 GSL | 1.13 SciPy† (8.4 ulps) |
| `fresnel` | x in [0,1] | 3.96 ns | 0.763 SciPy |  |
| `fresnel` | x in [1,8] | 11.1 ns | – | 0.370 SciPy† (3.1 ulps) |
| `fresnel` | x in [8,1e4] | 9.99 ns | – | 0.162 SciPy† (3.9·10³ ulps) |
| `faddeeva` | \|z\| < 1, Im z >= 0 | 32.3 ns | 1.57 puruspe |  |
| `faddeeva` | z in [-4,4]x[0,4] | 32.3 ns | 1.53 puruspe |  |
| `faddeeva` | z in [-10,10]x[0,10] | 23.9 ns | 1.28 puruspe |  |
| `faddeeva` | z in [-4,4]x[-4,0] | 42.8 ns | 1.32 puruspe |  |
| `faddeeva` | z in [10,100]x[0,100] | 9.63 ns | 1.08 puruspe |  |
| `erf`, complex | \|z\| < 1 | 9.69 ns | – | 0.172 SciPy† (260 ulps) |
| `erfc`, complex | \|z\| < 1 | 42.0 ns | 0.733 SciPy |  |
| `erfcx`, complex | \|z\| < 1 | 38.0 ns | 0.727 SciPy |  |
| `erf`, complex | z in [-4,4]^2 | 44.1 ns | – | 0.584 SciPy† (82.5 ulps) |
| `erfc`, complex | z in [-4,4]^2 | 44.2 ns | – | 0.574 SciPy† (110 ulps) |
| `erfcx`, complex | z in [-4,4]^2 | 38.3 ns | – | 0.571 SciPy† (72.7 ulps) |
| `erf`, `f32` | x in [-3,3] | 1.86 ns | 0.822 Apple libm |  |
| `erfc`, `f32` | x in [-3,3] | 5.13 ns | 0.679 Apple libm |  |
| `erfc`, `f32` | x in [3,9] | 5.39 ns | 0.691 Apple libm | 0.732 libm† (1.8 ulps) |
| `erf_inv`, `f32` | x in (-1,1) | 9.59 ns | 1.01 Boost |  |

### Where the references are faster

- **`faddeeva`** against puruspe's Johnson algorithm (1.08 to 1.57, within 2.9 to 21 ε of `a` on these regions): 27 nodes of the trapezoidal rule and the pole correction near the real axis, for a bound of 4.3 to 19.6 ε that holds every argument.
- **`erf_inv`** on `(−1, 1)` and near 1 (1.31 and 1.24 against statrs) and **`erfc_inv`** on `(0, 2)` (1.16 against Boost), within 1.5 to 1.9 ulps of `a`: the first guess and one Halley step through `erfcx` and an exponential, which holds the 0.98-ulp bound.

## The Bessel and Airy functions

### Bessel functions

- `bessel_j(ν, x)` and `bessel_y(ν, x)` are the Bessel functions of the first and second kind `J_ν(x)` and `Y_ν(x)`, `bessel_i(ν, x)` and `bessel_k(ν, x)` the modified ones `I_ν(x)` and `K_ν(x)`, of any real order `ν` (an `f32` or an `f64`, the type of `x`). `bessel_i_scaled(ν, x) = e^(−|x|)·I_ν(x)` and `bessel_k_scaled(ν, x) = e^x·K_ν(x)` do not overflow or underflow where `I` and `K` would.
- **Negative orders** by the reflections: `J_(−ν) = cos(νπ)·J_ν − sin(νπ)·Y_ν`, `Y_(−ν) = sin(νπ)·J_ν + cos(νπ)·Y_ν`, `I_(−ν) = I_ν + (2/π)·sin(νπ)·K_ν`, `K_(−ν) = K_ν`, with `sin(νπ)` and `cos(νπ)` exact at the integers and half-integers (so `J_(−n) = (−1)ⁿ·J_n` exactly). The terms are formed in double-double with the scale of `Y_ν` and `K_ν` (for the scaled `I`, of `e^(−x)·K_ν`) kept apart, as either may be beyond the range where its product with `sin(νπ)` is not; where the two terms of `I_(−ν)` cancel, about its one zero for `ν` in `(2k − 1, 2k)` (below 100), `I_(−ν)` comes from its own power series instead.
- **Negative arguments** for the integer orders only: `J_n(−x) = (−1)ⁿ·J_n(x)`, `I_n(−x) = (−1)ⁿ·I_n(x)` (and `e^(−|x|)·I_n(x)` likewise); a negative `x` with a non-integer order, or for `Y` and `K`, is NaN.
- **Special values.** `J₀(0) = I₀(0) = 1`; `J_ν(0) = I_ν(0) = 0` for `ν > 0` and `±∞` for a negative non-integer `ν` (the sign of `1/Γ(1 + ν)`); `Y_ν(0) = −∞` (for `ν ≥ 0`), `K_ν(0) = +∞`; `J`, `Y`, `K` → 0 and `I` → `∞` as `x → ∞`, the scaled forms → 0. NaN in gives NaN, and so does an infinite order.
- **Spherical functions** of an order `n: u32`: `sph_bessel_j(n, x) = jₙ(x) = √(π/(2x))·J_(n+1/2)(x)`, `sph_bessel_y` (`yₙ`), `sph_bessel_i` (`iₙ`, of `I_(n+1/2)`) and `sph_bessel_k` (`kₙ = √(π/(2x))·K_(n+1/2)(x)`, SciPy's `spherical_kn`). `jₙ`, `yₙ` and `iₙ` are defined for negative `x` by their parities (`jₙ(−x) = (−1)ⁿ jₙ(x)`, `yₙ(−x) = (−1)ⁿ⁺¹ yₙ(x)`, `iₙ(−x) = (−1)ⁿ iₙ(x)`); `kₙ` of a negative `x` is NaN. `j₀(0) = i₀(0) = 1`, `jₙ(0) = iₙ(0) = 0` for `n ≥ 1`, `yₙ(0) = −∞`, `kₙ(0) = ∞`, and `yₙ` and `kₙ` are `∓∞` where they overflow at tiny `x`.
- **Large orders.** From `ν = 50` `I_ν` and `K_ν` (plain and scaled) come from Debye's uniform expansions, in constant time and to their bound at every order (`I` from its power series where `x² ≤ 16(ν + 1)` or `x < 30`, `K` from its leading term below `x = 10⁻¹⁵⁰`). `J_ν` and `Y_ν` take time that grows with `ν` where `x` is near or above the order, up to `ν = 10⁷`; above, they come from Olver's uniform expansion (with the module's Airy functions), in constant time, and are 0 and `−∞` where they underflow and overflow (`x` well below `ν`), and 0 from about `x = 6.7·10²⁹⁹` on, where the argument of the Airy functions passes `10²⁰⁰` (their phase is lost long before). The spherical `jₙ`, `yₙ` and `kₙ` of orders above `10⁷` follow `J`, `Y` and `K` of `n + 1/2`, and `iₙ` from `n = 20` on Debye's `I`, in constant time. No valid argument of a Bessel function gives NaN.

### Airy functions

- `airy_ai(x)`, `airy_bi(x)` and their derivatives `airy_ai_prime(x)`, `airy_bi_prime(x)`, the solutions of `w″ = x·w` with `Ai(0) = 3^(−2/3)/Γ(2/3)`, `Ai′(0) = −3^(−1/3)/Γ(1/3)`, `Bi(0) = 3^(−1/6)/Γ(2/3)`, `Bi′(0) = 3^(1/6)/Γ(1/3)`, for every real `x`.
- `Ai(x)` and `Ai′(x)` fall into the subnormals from about 103.9 and 104.1 and underflow to 0 (and −0) beyond about 107.5 and 107.7; `Bi(x)` and `Bi′(x)` overflow to `∞` beyond about 104.4 and 104.2. On the negative axis they oscillate in `ζ = (2/3)|x|^(3/2)`, which is formed in double-double: its absolute error, `2⁻¹⁰⁴·ζ`, is the phase error. Beyond `|x| = 10⁹` (where it is below a hundredth of an ulp of the modulus; it passes an ulp at about `3·10¹⁰`) the result keeps its bound but for that phase error, and from about `|x| = 10²¹` the phase is lost entirely: the result then stays within the modulus (`|Ai(x)| ≤ M`, and so on) and is no more than that. Beyond `|x| = 10²⁰⁰` (where `ζ` nears the overflow) `Ai` and `Bi` return 0 and their derivatives NaN; `Ai(−∞) = Bi(−∞) = 0`, `Ai′(−∞)` and `Bi′(−∞)` are NaN.

### Examples of the Bessel and Airy functions

```rust
use a::math::special::{airy_ai, bessel_i, bessel_j, bessel_k, bessel_y, sph_bessel_i, sph_bessel_j};

// At the double nearest the first zero of J₀ the value is −6.1·10⁻¹⁷, to within an ulp.
assert!((bessel_j(0.0f64, 2.404_825_557_695_773) + 6.108_765_259_736_73e-17).abs() < 4e-32);
// J_(1/2)(x) = √(2/(πx))·sin x, and Y_(−1/2) = J_(1/2).
let j = bessel_j(0.5f64, 1.0);
assert!((j - 0.671_396_707_141_803_1).abs() < 2e-16);
assert!((bessel_y(-0.5f64, 1.0) - j).abs() < 2e-16);
assert!((bessel_k(1.5f32, 2.0) - 0.179_906_66).abs() < 2e-8);
assert!((sph_bessel_j(1, 1.0f64) - 0.301_168_678_939_756_8).abs() < 1e-16);
assert_eq!(airy_ai(0.0f64), 0.355_028_053_887_817_2);
// A large order far below e^x: i₁₇₁₂(980.15) = 2.6446875375185671·10⁻¹³².
let i = sph_bessel_i(1712, 980.154_177_456_350_6f64);
assert!((i / 2.644_687_537_518_567e-132 - 1.0).abs() < 4e-16);
// Debye's I at an order of 10¹⁵, in range only about x = 0.6627·ν: no loss with the order.
let v = bessel_i(889_990_503_241_027.4f64, 589_835_349_306_017.2);
assert!((v / 2.133_964_420_593_033e-197 - 1.0).abs() < 4e-16);
```

### Accuracy of the Bessel and Airy functions

The bounds below are met in every case measured (the largest errors measured, under them, follow the table), in ulps of `f64`, against mpmath at 256 bits (the test data of `tests/math/special/data/bessel_*.txt`; the cases added after the audit of 2026-11-21 evaluated until two working precisions agree, and from Debye's series at 200 bits and more for the orders beyond mpmath's reach) and against Arb, certified to 150 bits, in a larger random comparison of 1,500 to 40,000 cases per function and region. An `f32` result is within 0.51 ulp in every case tested, of the value or of the scale where the `f64` bound is in one, but for the Airy functions below −10⁹, which carry the phase error as in `f64` (a tenth of an ulp of an `f32` from about `|x| = 3·10¹⁵`). Near the zeros of the oscillating functions the error of a method of this kind is absolute: the bound is then in ulps of the modulus `M = √(J² + Y²)` (for `Ai`, `Bi` of `√(Ai² + Bi²)`, for their derivatives of `√(Ai′² + Bi′²)`), which bounds the value and, where the function oscillates, equals its size away from the zeros. So it is for `I_ν` at the negative non-integer orders, whose zero (one for `ν` in `(−2k, −2k + 1)`) the reflection's two terms do not resolve: the bound is in ulps of `I_|ν| + (2/π)|sin νπ|·K_|ν|`, the size of those terms, which is the size of the value but about the zero; there (for `|ν| < 100`) the power series keeps the error relative, within an ulp of the value down to about `10⁻¹⁰` of the terms' size for `|ν|` up to 75, while nearer 100 its own terms grow to `2²⁸` of that size, and the error to 8 ulps of the value at `10⁻⁹` of it and 300 at `10⁻¹⁰`. Below their turning point, `|x| ≤ ν` (for `jₙ`, `|x| ≤ n`), `J_ν` and `jₙ` have no zeros and fall while `Y_ν` and `yₙ` grow, so that `M` is the size of `Y_ν` (`yₙ`) there, far above theirs: there `J_ν` (for `0 < ν ≤ 10⁷`) and `jₙ` (for `n ≤ 10⁷`) keep their bounds in ulps of their own value.

| Function | Region | Bound |
| --- | --- | --- |
| `J₀`, `J₁`, `Y₀`, `Y₁` | `x < 24` (zeros included) | 1.5 ulps of the value |
| | `x ≥ 24` | 1 ulp of `M` |
| `J_ν`, `Y_ν`, other orders | `\|ν\| ≤ 10⁷` (measured to 3,000, `J` at its turning point to `1.8·10⁶`) | `J`: 5, `Y`: 6 ulps of `M` (`J` at `ν > 0`, `\|x\| ≤ ν`: 5 ulps of the value) |
| | `\|ν\| > 10⁷` (Olver's expansion) | `2·max(ν, x)` ulps of `M` |
| `I₀`, `I₁`, `K₀`, `K₁`, scaled | all `x` | 1 ulp |
| `I_ν`, `K_ν`, other orders, scaled | `\|ν\| < 50` | 6 ulps (`I` at non-integer `ν < 0`: of `I_\|ν\| + (2/π)\|sin νπ\|·K_\|ν\|`) |
| | `\|ν\| ≥ 50` (Debye's expansions), every order | 1 ulp (`I` at non-integer `ν ≤ −50`: 1.5 ulps of `I_\|ν\| + (2/π)\|sin νπ\|·K_\|ν\|`) |
| `jₙ`, `yₙ` | `n = 0` | 1 ulp of the value |
| | `1 ≤ n ≤ 10⁷` (beyond, as `J`, `Y` of `n + 1/2`) | 2 ulps of `M` (`jₙ` at `\|x\| ≤ n`: of the value) |
| `iₙ`, `kₙ` | all | 3 ulps (`kₙ`: 1) |
| `Ai`, `Ai′`, `Bi`, `Bi′` | `x ≥ −8.4` (zeros included) | 1.5 ulps of the value |
| | `−10⁹ ≤ x < −8.4` | 1 ulp of the modulus |
| | `x < −10⁹` | 1 ulp of the modulus and the phase error `2⁻¹⁰⁴·ζ` of it, within the modulus |

Measured: the orders 0 and 1 of `I` and `K` 0.65 ulp at most, `J` and `Y` of those orders 1.21 ulps below 24 (`Y₀`, 40,000 cases a function) and 0.62 ulp of `M` beyond; the other orders of `J` 2.0 (2.74 at a negative order, through `Y_|ν|`) and of `Y` 2.8 ulps of `M` (at the turning point `x ≈ ν` of the orders 50 to 300 1.26 and 1.24, of `J` at the orders `1.8·10⁴` to `1.8·10⁶` 0.36), and `J` below its turning point 3.25 ulps of its value; `I` and `K` below `|ν| = 50` 3.9 and 5.25 ulps (5.2 for `I` at the negative orders, through `K`), Debye's `I` and `K` 0.54 ulp (against Arb to `ν = 10¹⁵`, against their series at a high precision to `4·10¹⁸`, the scaled forms to `5·10¹⁵⁵` with `x` up to `f64::MAX`), `I` at the non-integer orders from −300 to −50 0.96 ulp; `jₙ` 1.05 and `yₙ` 0.98 ulp of `M` (`jₙ` below its turning point 0.65 ulp of its value), `iₙ` 2.22 and `kₙ` 0.56 ulp; the Airy functions 1.14 ulps from −8.4 up and 0.75 ulp of the modulus from −10⁹ to −8.4. The scaled `I` at the non-integer orders from `−2¹⁶` down is within the range only within about 250 of `x = 0.448|ν|`, where `η(x/|ν|) = −x/|ν|` and the exponent of its term `e^(−x)·K_|ν|(x)` passes 0: 0.57 ulp of the terms there up to `|ν| = 4.4·10¹⁵` (360 cases against Debye's series for both terms, in two implementations), and `±∞` below that band and 0 above it. Above `ν = 10⁷` the phase of Olver's expansion reaches the Airy functions through an `f64` argument, which bounds the error by about `2·max(ν, x)` ulps of the modulus (up to 0.8 of it measured).

### Methods of the Bessel and Airy functions

- **The `f32` forms of `J₀` and `J₁`** on `|x| < 25`: polynomials of degree 14 in `x − (i + 1/2)` on the unit pieces (Chebyshev interpolants, mpmath at 40 digits), on a piece with a zero `z` of `J(x)/(x − z)` times `x − z` from the double-double `z`, good to `2⁻⁴⁵` in `f64` and rounded once to the `f32` (checked against the `f64` path at every `f32` of `[−25.5, 25.5]`). `Y₀` and `Y₁` on `0 < x < 25` alike, on pieces of width 1/4 on `[1/2, 1)`, 1/2 on `[1, 2)` and 1 from 2 on (the logarithm at 0 needs the shorter ones), good to `2.5·10⁻¹⁵`, and below 1/2 by `(2/π) ln(x) J(x) + P(x²)` (`P(x²)/x` for `Y₁`, `P` entire, of degree 9) with `J` from the pieces of `J₀`, `J₁` (at every `f32` of `[0, 25.5]` 3 results of `Y₀` and 1 of `Y₁` differ from the `f64` path rounded, all ties within `0.5000` ulp of mpmath). The integer orders `2 ≤ n ≤ 400` of `J` for `x > n` and of `Y` for every `x > 0`: the plain forward recurrence in `f64` from those (from 25 on from the modulus-phase forms of the `f64` path), and of `J` for `x² ≤ 4(n + 1)` the power series in plain `f64` (its terms decrease from the first and its sum is at least about `e⁻¹`), a few `n` ulps of `M` (or of the value) in `f64`, far below the rounding to the `f32` (at every `f32` of `[0, 1000]` for the orders 2, 3, 7, 20 and 40, at most 5 results of an order differ from the `f64` path rounded, all within `0.5000` ulp of `max(|v|, M)`).
- **Orders 0 and 1 of `J` and `Y`.** Below 24, each interval between two extrema in three pieces: about its zero `f(z + d) = d·q(d)` with the zero as a double-double, and about the midpoints of the two outer thirds plain polynomials, Remez fits of degree 12 to 14, their tails from `c₃` by Estrin's scheme and their three outer steps of Horner's rule in double-double with `x − m` exact (at the ends of a piece `c₁u` and `c₂u²` reach half and a quarter of the value, where a step in `f64` cost up to an ulp): the error stays relative through the zeros. `J₀` below 1.5 and `J₁` below the first extremum are polynomials in `x²`; `Y₀` and `Y₁` below half their first zero are `(2/π)·ln x·J + R(x²)`, and about their first two zeros `(2/π)·ln(x/z)·J + (x − z)·g(x)`, whose fit `g` has no logarithmic singularity. From 24 on, the modulus and the phase, `J = M cos θ`, `Y = M sin θ`, `M = √(2/(πx))·(1 + P(1/x²))`, `θ = x − (2n + 1)π/4 + α(1/x)`, Remez fits, with `x` reduced modulo `π/2` to a double-double (Cody and Waite's three-part `π/2` below 10⁶, Payne and Hanek's 1280 bits of `2/π` above) and the sine and cosine of the reduced double-double by FreeBSD msun's kernels.
- **Orders 0 and 1 of `I` and `K`.** Polynomials in `x²` up to 1 (`K` with its `ln x·I` apart), `I` in nine pieces of `[1, 7.75]` about their midpoints, and `e^(∓x)·√x·(I, K)` in pieces of `1/x`: Remez fits with their first two coefficients in double-double, evaluated by Estrin's scheme; `1/√x` from `√x·(1/x)` with one correction, `e^x` from a table of `2^(j/32)` and a polynomial to 2⁻⁵⁸.
- **Other orders.** Where `x ≥ 20` and `x ≥ ν²/2`, Hankel's expansion of the order `ν` itself (for `I` and `K` below `ν = 50`). Elsewhere, with `ν = n + μ`, `|μ| ≤ 1/2`: `J_μ` and `J_(μ+1)` from their power series summed in double-double (it serves up to 30, where its terms cancel down to `e^(−x)`); `Y_μ` and `Y_(μ+1)` from Temme's series below 2 (Temme 1975; its leading terms are of order 1 and cancel, so its ingredients `μπ/sin μπ`, `Γ₁`, `Γ₂`, `sinh σ/σ`, `cosh σ`, `e^σ` and its first two steps are formed in double-double, the rest in `f64`) and from Steed's continued fraction CF2, `p + iq = (J′ + iY′)/(J + iY)`, summed backwards (`120/x + 8` terms), as `Y_μ = (p·J_μ − J′_μ)/q`; Hankel's expansion from 30 on; then the recurrences in double-double up to `ν` (`Y` always, `J` while `x > ν`). Where `x ≤ ν`, `J_ν` comes from its series or from the Wronskian `J_ν = (2/(πx))/(r·Y_ν − Y_(ν+1))`, `r = J_(ν+1)/J_ν` by CF1 summed from the tail in double-double (near the turning point its partial ratios stay near 1 for about `7ν^(1/3)` terms, through which the rounding errors of an `f64` run pile up), at least so many terms that the zero tail costs below `e^(−48)` (by Debye's exponent of `J_(ν+k)`; Lentz's test stops short there), `r·Y_ν − Y_(ν+1)` in double-double (the terms cancel to about `ν^(−1/3)` of each, and an error of the recurrence that is a multiple of `J` drops out). `I_ν` from its power series (positive terms, double-double) below 30, else the Wronskian with CF1; `K_μ`, `K_(μ+1)` from Temme's series below 1/2 and from the Wronskian `I_μ·K_(μ+1) + I_(μ+1)·K_μ = 1/x` with `K_(μ+1)/K_μ` by CF2 summed backwards (no step of it cancels), then up by the recurrence. Integer orders start from the dedicated `J₀`, `J₁`, `Y₀`, `Y₁`, `I₀`, `I₁`, `K₀`, `K₁` (beyond `x = 700`, where `K₀` and `K₁` leave the normal range, from the scaled ones with `e^(−x)` apart). The scaled forms carry `e^(∓x)` as a separate exponent throughout. From `ν = 50`, `I` and `K` by Debye's expansions `I_ν(x) = e^(νη)/√(2πr)·Σ Uₖ(p)/νᵏ`, `K_ν(x) = √(π/2)·e^(−νη)/√r·Σ (−1)ᵏUₖ(p)/νᵏ`, `r = √(ν² + x²)`, `p = ν/r`, `U₁` to `U₁₂` from their recurrence. The exponent `νη` is the size of the result in `e`-folds, its absolute error the result's relative error: below `ν = 2¹⁶` it is `r + ν·ln(x/(ν + r))` in double-double (for the scaled forms below `x = 2⁴⁵`); beyond, where that form carries `ν` times the error of the logarithm and, for the scaled forms, the cancellation of `r − x`, `νη − x = −p·xν/(r + x) − ν(atanh p − p)` for the scaled forms (terms of one sign, the first as `p` times `xν/(r + x)`, factors that stay normal where `ν²/(r(r + x))` does not, near the top of the range; `atanh p − p` by its series where `p` is small, else from `ln((r + ν)/x)`), and for the plain ones, in range only within about 1300 of `x = z₀ν` (`z₀ = 0.6627…` the root of `η`, Laplace's limit), `νη = D·P(D/ν)` from `D = x − z₀ν` formed exactly with the bits of `z₀` (as Payne and Hanek reduce by `2/π`) and `P` the Taylor series of `η/(z − z₀)` about `z₀`. For the scaled `I` at a negative order, `e^(−x)·K_a = (e^x·K_a)·e^(−2x)` with the scales apart, decided first by its exponent `−aη − x` in `f64` where that is beyond the range; from `a = 2¹⁶` on, where the two factors' exponents are about `0.9a` each, that exponent, `−a(η(z) + z)` with `z = x/a`, comes whole from `D = x − z₁a` (`z₁ = 0.4477…` the root of `η(z) + z`, its bits multiplied out as for `z₀`) and the Taylor series of `(η(z) + z)/(z − z₁)` about `z₁`. Above `ν = 10⁷`, `J` and `Y` by Olver's expansion to its terms `A₀` and `B₀`, `ζ`, `φ` and `B₀` by their series about `x = ν` or their closed forms.
- **Spherical functions.** `j₀ = sin x/x`, `y₀ = −cos x/x` from the double-double reduction; the power series where `x² ≤ n + 3/2`; the terminating Hankel sums `jₙ = (P sin(x − nπ/2) + Q cos(x − nπ/2))/x` from `x ≥ n(n + 1)/2`; between, `jₙ` up from `j₀`, `j₁` where `x > n` and down from the continued fraction of `jₙ/jₙ₋₁` where `x ≤ n` (that of `J`, summed from its tail in double-double at every `x`, as the run down passes its error to `jₙ` whole; normalised by `j₀` or `j₁`), `yₙ` up from `y₀`, `y₁`, all in double-double from double-double starts; `yₙ` and `kₙ` below `x = 2⁻⁵⁰⁰` from their leading term `(2n − 1)!!/x^(n+1)`. `kₙ = (π/(2x))·e^(−x)·Σ aₖ/(2x)ᵏ` by the recurrence of its positive terms. `iₙ` from its power series (up to `x = 20 + n/2` below `n = 20`, `x² ≤ 16(n + 1.5)` or `x < 30` from it on), from the closed form `(e^x·Σ(−1)ᵏaₖ/xᵏ − (−1)ⁿe^(−x)·Σ aₖ/xᵏ)/(2x)` in double-double with both exponentials between there and `x = n(n + 1)/2` below `n = 20` (where its terms cancel by `2⁴⁰` at most), and from Debye's `I_(n+1/2)` there from `n = 20` on; beyond `x = n(n + 1)/2` from the closed form (`∞` beyond 720).
- **Airy functions.** For `x ≥ 4` (`Bi`, `Bi′`: 12), `Ai(x) = e^(−ζ)/(2√π·x^(1/4))·(1 + t·A(t))`, `t = 1/ζ`, and likewise for the others; from 1 up to there Remez fits of `e^(±ζ)·f` in pieces; from the first extremum left of 0 up to 1 Remez fits of `f` itself, in pieces halved until the fit holds `2⁻⁵⁷` with its coefficients rounded to `f64` (at the ends of the widest pieces `c₂u²` and `c₃u³` reach the size of the value); down to about −8.5 one polynomial about each zero (as for `J₀`); these by Horner's rule with the last three steps in double-double. Below, the modulus and the phase, `Ai(−y) = M cos θ`, `Bi(−y) = M sin θ`, `θ = π/4 − ζ + α(1/ζ)` (`Ai′`, `Bi′` with `N` and `φ = 3π/4 − ζ + β`), Remez fits, the phase reduced in double-double, both parts of `ζ` modulo `π/2` (its low part passes `π/4` from about `|x| = 10¹¹`).
- Nothing allocates. The read-only tables hold about 51 KB of coefficients (`J₀`…`Y₁` 14 KB, the `f32` forms of `J₀`…`Y₁` 15.6 KB, `I₀`…`K₁` and `1/Γ(1 + μ)` 6.3 KB, the large orders 2.4 KB, the Airy functions 11.9 KB), `2^(j/32)` 512 bytes and the bits of `2/π` 160 bytes; their generators (`gen01.py`, `gen_ik.py`, `gen_olver.py`, `gen_f32.py` and the shared `remez.py` in `bessel/`, `gen_airy.py` and `airy_post.py` in `airy/`, on mpmath) reproduce them exactly. The elementary functions are the module's own (`e^x`, `ln x`, the sine and cosine, `atan`), and the polynomials use `mul_add` where it is one instruction (aarch64, x86_64 with FMA) and plain products elsewhere, while the error terms of the double-double arithmetic are exact everywhere (a fused multiply-add or Dekker's product). All targets of one kind give the same bits; the others (x86_64 without FMA, and Miri) differ in the last bits. They were checked by the same code built with the fused path off on aarch64, where every test passes and the sweeps stay within the bounds (an error term once left to the plain path had cost `Y₁` 1.95 ulps there), and by the tests under Miri on every 37th case of their data, not on x86_64 itself.

### Measured

As for [the gamma function](#measured) (the conditions and the same-work rule are there):

| function | arguments | `a` | within the bound | faster, beyond the bound |
| --- | --- | ---: | --- | --- |
| `bessel_j`, ν = 0 | x in [0,25] | 12.7 ns | 0.613 Apple libm | 1.50 puruspe† (7.9 ulps) |
| `bessel_y`, ν = 0 | x in [0.01,25] | 19.6 ns | 1.31 Apple libm | 1.84 puruspe† (54 ulps) |
| `bessel_j`, ν = 1 | x in [0,25] | 13.0 ns | 0.648 Apple libm | 1.56 puruspe† (8.2 ulps) |
| `bessel_y`, ν = 1 | x in [0.01,25] | 20.4 ns | 1.34 Apple libm | 2.08 puruspe† (33.7 ulps) |
| `bessel_j`, ν = 0 | x in [25,1e4] | 26.2 ns | 1.18 Apple libm | 2.40 puruspe† (2.7·10³ ulps) |
| `bessel_y`, ν = 0 | x in [25,1e4] | 27.4 ns | 1.75 Apple libm | 2.49 puruspe† (2.7·10³ ulps) |
| `bessel_j`, ν = 1 | x in [25,1e4] | 27.2 ns | 1.26 Apple libm | 2.91 puruspe† (2·10³ ulps) |
| `bessel_y`, ν = 1 | x in [25,1e4] | 27.0 ns | 1.74 Apple libm | 2.88 puruspe† (2.1·10³ ulps) |
| `bessel_i`, ν = 0 | x in [0,50] | 12.1 ns | – | 2.20 puruspe† (5.9 ulps) |
| `bessel_k`, ν = 0 | x in [0.01,50] | 10.4 ns | 2.05 Boost |  |
| `bessel_i`, ν = 1 | x in [0,50] | 12.1 ns | – | 2.19 puruspe† (5.3 ulps) |
| `bessel_k`, ν = 1 | x in [0.01,50] | 10.6 ns | 0.592 GSL | 2.00 Boost† (2.1 ulps) |
| `bessel_j`, integer n | n integer in [2,20], x in [0.1,50] | 96.0 ns | 1.86 Apple libm | 2.86 puruspe† (22.6 ulps) |
| `bessel_y`, integer n | n integer in [2,20], x in [0.1,50] | 87.1 ns | 2.61 Apple libm | 3.98 Boost† (41 ulps) |
| `bessel_i`, integer n | n integer in [2,20], x in [0.1,50] | 148 ns | 1.17 puruspe |  |
| `bessel_k`, integer n | n integer in [2,20], x in [0.1,50] | 59.2 ns | 3.59 puruspe |  |
| `bessel_j` | nu in [0,10], x in [0.1,2] | 91.8 ns | 2.09 Boost |  |
| `bessel_y` | nu in [0,10], x in [0.1,2] | 238 ns | 2.64 GSL |  |
| `bessel_j` | nu in [0,10], x in [2,30] | 187 ns | – | 1.39 Boost† (23.6 ulps) |
| `bessel_y` | nu in [0,10], x in [2,30] | 229 ns | – | 1.40 Boost† (29.4 ulps) |
| `bessel_j` | nu in [0,3], x in [30,1e3] | 69.1 ns | 0.401 SciPy | 2.35 Boost† (10.8 ulps) |
| `bessel_y` | nu in [0,3], x in [30,1e3] | 67.7 ns | 2.39 Boost |  |
| `bessel_j` | nu in [10,100], x in [1,200] | 295 ns | – | 1.25 GSL† (4.5·10⁵ ulps) |
| `bessel_y` | nu in [10,100], x in [1,200] | 299 ns | – | 1.08 GSL† (2.2·10³ ulps) |
| `bessel_j` | nu in [50,300], x in nu*[0.95,1.05] (turning point) | 657 ns | – | 8.44 GSL† (1.7·10³ ulps) |
| `bessel_y` | nu in [50,300], x in nu*[0.95,1.05] (turning point) | 619 ns | – | 8.44 GSL† (2·10³ ulps) |
| `bessel_j` | nu in [-30,0] non-integer, x in [0.1,60] | 292 ns | – | 1.26 Boost† (40.6 ulps) |
| `bessel_y` | nu in [-30,0] non-integer, x in [0.1,60] | 292 ns | – | 1.26 Boost† (30.9 ulps) |
| `bessel_i` | nu in [0,10], x in [0.1,2] | 72.0 ns | 0.916 Boost |  |
| `bessel_i` | nu in [0,10], x in [2,30] | 114 ns | 0.634 SciPy | 0.722 GSL† (14.6 ulps) |
| `bessel_i` | nu in [0,3], x in [30,1e3] | 48.6 ns | – | 0.386 GSL† (1.3·10³ ulps) |
| `bessel_i` | nu in [10,100], x in [1,200] | 180 ns | – | 1.23 SciPy† (310 ulps) |
| `bessel_i` | nu in [-30,0] non-integer, x in [0.1,60] | 353 ns | – | 1.79 SciPy† (610 ulps) |
| `bessel_i` | nu in [50,300], x in nu*[0.5,2] (Debye) | 77.9 ns | – | 1.69 SciPy† (670 ulps) |
| `bessel_k` | nu in [0,10], x in [0.1,2] | 222 ns | – | 2.37 Boost† (27 ulps) |
| `bessel_k` | nu in [0,10], x in [2,30] | 175 ns | 1.81 GSL |  |
| `bessel_k` | nu in [0,3], x in [30,1e3] | 50.3 ns | – | 1.23 Boost† (4.3·10⁵ ulps) |
| `bessel_k` | nu in [10,100], x in [1,200] | 117 ns | – | 1.19 GSL† (360 ulps) |
| `bessel_k` | nu in [50,300], x in nu*[0.5,2] (Debye) | 76.9 ns | – | 0.389 GSL† (37.7 ulps) |
| `bessel_i_scaled` | nu in [0,10], x in [0.1,2] | 74.9 ns | – | 1.04 SciPy† (44.1 ulps) |
| `bessel_i_scaled` | nu in [0,10], x in [2,30] | 117 ns | – | 0.738 GSL† (15.6 ulps) |
| `bessel_i_scaled` | nu in [10,100], x in [1,200] | 180 ns | – | 0.819 GSL† (1.7·10⁷ ulps) |
| `bessel_i_scaled` | nu in [-30,0] non-integer, x in [0.1,60] | 365 ns | – | 0.402 SciPy† (2.9·10⁴ ulps) |
| `bessel_i_scaled` | nu in [50,1000], x in [1,2000] | 81.0 ns | – | 2.57 GSL† (5.4·10¹³ ulps) |
| `bessel_k_scaled` | nu in [0,10], x in [0.1,2] | 222 ns | – | 2.47 GSL† (27.5 ulps) |
| `bessel_k_scaled` | nu in [0,10], x in [2,30] | 171 ns | 1.88 GSL |  |
| `bessel_k_scaled` | nu in [0,3], x in [30,1e3] | 43.9 ns | 1.60 GSL |  |
| `bessel_k_scaled` | nu in [10,100], x in [1,200] | 118 ns | – | 1.25 GSL† (240 ulps) |
| `bessel_k_scaled` | nu in [50,1000], x in [1,2000] | 79.2 ns | – | 0.399 SciPy† (NaN or ∞ on 156 of 65536) |
| `sph_bessel_j` | n in [0,10], x in [0.1,50] | 33.0 ns | – | 0.918 SciPy† (120 ulps) |
| `sph_bessel_y` | n in [0,10], x in [0.1,50] | 30.1 ns | – | 1.83 SciPy† (7.9 ulps) |
| `sph_bessel_j` | n in [0,60], x in [0.1,200] | 94.4 ns | – | 0.498 SciPy† (130 ulps) |
| `sph_bessel_y` | n in [0,60], x in [0.1,200] | 74.3 ns | – | 0.857 GSL† (1.4·10³ ulps) |
| `sph_bessel_j` | n in [10,60], x in [0.1,n/2] | 182 ns | – | 2.06 GSL† (13.6 ulps) |
| `sph_bessel_y` | n in [10,60], x in [0.1,n/2] | 79.6 ns | – | 0.988 GSL† (470 ulps) |
| `sph_bessel_i` | n in [0,10], x in [0.1,50] | 44.9 ns | 0.347 SciPy | 0.484 GSL† (1.1·10⁴ ulps) |
| `sph_bessel_k` | n in [0,10], x in [0.1,50] | 14.2 ns | – | 0.826 GSL† (17.1 ulps) |
| `sph_bessel_i` | n in [0,60], x in [0.1,200] | 71.9 ns | – | 0.412 GSL† (740 ulps) |
| `sph_bessel_k` | n in [0,60], x in [0.1,200] | 47.8 ns | – | 2.24 GSL† (15.3 ulps) |
| `airy_ai` | x in [-10,10] | 14.9 ns | – | 0.332 GSL† (83.2 ulps) |
| `airy_ai_prime` | x in [-10,10] | 14.9 ns | – | 0.546 GSL† (140 ulps) |
| `airy_bi` | x in [-10,10] | 11.6 ns | – | 0.265 GSL† (16.1 ulps) |
| `airy_bi_prime` | x in [-10,10] | 12.7 ns | – | 0.446 GSL† (23 ulps) |
| `airy_ai` | x in [-1e3,-10] | 32.8 ns | – | 0.535 Boost† (1.7·10⁴ ulps) |
| `airy_ai_prime` | x in [-1e3,-10] | 33.5 ns | – | 0.969 GSL† (3.8·10⁴ ulps) |
| `airy_bi` | x in [-1e3,-10] | 33.6 ns | – | 0.549 Boost† (1.8·10⁴ ulps) |
| `airy_bi_prime` | x in [-1e3,-10] | 33.9 ns | – | 0.988 GSL† (4.7·10⁴ ulps) |
| `airy_ai` | x in [10,100] | 21.0 ns | – | 1.01 GSL† (590 ulps) |
| `airy_ai_prime` | x in [10,100] | 22.3 ns | – | 1.25 GSL† (580 ulps) |
| `airy_bi` | x in [10,100] | 18.4 ns | – | 0.995 GSL† (570 ulps) |
| `airy_bi_prime` | x in [10,100] | 19.2 ns | – | 0.743 GSL† (560 ulps) |
| `bessel_j`, ν = 0, `f32` | x in [0,25] | 2.40 ns | 0.274 Boost |  |
| `bessel_y`, ν = 0, `f32` | x in [0.01,25] | 4.32 ns | 0.414 Boost |  |
| `bessel_j`, ν = 1, `f32` | x in [0,25] | 3.51 ns | 0.387 Boost |  |
| `bessel_y`, ν = 1, `f32` | x in [0.01,25] | 3.44 ns | 0.318 Boost |  |
| `bessel_j`, integer n, `f32` | n in [2,20], x in [0.1,50] | 68.1 ns | 1.89 Boost |  |
| `bessel_y`, integer n, `f32` | n in [2,20], x in [0.1,50] | 50.0 ns | 2.35 Boost |  |
| `bessel_j`, `f32` | nu in [0,10], x in [2,30] | 191 ns | 1.40 Boost |  |
| `bessel_k`, `f32` | nu in [0,10], x in [0.1,2] | 221 ns | 2.32 Boost |  |
| `airy_ai`, `f32` | x in [-10,10] | 15.1 ns | 0.0694 Boost |  |

### Where the references are faster

- **`kₙ`, `yₙ`, `jₙ` and `iₙ`, integer orders 2 to 20** (3.59 against puruspe, 2.61 and 1.86 against Apple's libm, 1.17 against puruspe): the starting values and the recurrences with their error terms carried, which hold `K` and `Y` to 6 ulps and `J` to 5, against plain `f64` recurrences (puruspe's `kₙ` within 11.6 ulps of `a`, Apple's `yₙ` and `jₙ` within 6.6 and 6.8).
- **Real orders at small and moderate `x`**: `Y_ν` and `J_ν` for `ν` in `[0, 10]`, `x` in `[0.1, 2]` (2.64 against GSL, 2.09 against Boost: Temme's series and the power series in double-double), `K_ν` and the scaled `K` for `x` in `[2, 30]` (1.81, 1.88, GSL: the continued fraction CF2 summed backwards in double-double) and the scaled `K` for `x` in `[30, 10³]` (1.60, GSL), `Y_ν` for `x` in `[30, 10³]` (2.39, Boost: the double-double phase of Hankel's expansion).
- **Orders 0 and 1**: `K₀` (2.05 against Boost, within 2.0 ulps of `a`), `Y₀`, `Y₁` beyond 25 (1.75, 1.74) and below it (1.31, 1.34), `J₀`, `J₁` beyond 25 (1.18, 1.26), against Apple's libm: the double-double outer steps of the pieces and of the reduction of the phase, for bounds of 1.5 ulps of the value and 1 ulp of `M`.
- **The `f32` forms**: `bessel_y` and `bessel_j` of integer orders (2.35, 1.89), `bessel_k` for `x` in `[0.1, 2]` (2.32) and `bessel_j` for `x` in `[2, 30]` (1.40), against Boost in `double`: the general orders of `f32` run the `f64` paths (Temme's series, the continued fractions), not tables of their own.

## The zeta function and the others

- `zeta(s)` is the Riemann zeta function `ζ(s) = Σ n⁻ˢ`, continued to the whole line or plane but its pole at `s = 1` (`+∞` there): of an `f32`, an `f64`, a `Complex<f32>` or a `Complex<f64>`. Its trivial zeros `−2, −4, …` are exact zeros, `ζ(+∞) = 1`, `ζ(−∞)` is NaN; past `s ≈ −260` it overflows to an infinity of the sign of `sin(πs/2)`. Of a complex `s = σ + it`: `1 + 2⁻ˢ` for `σ ≥ 56` at every finite `t`; NaN only for an infinite `t` (there is no limit), for `σ = −∞` and `σ < −2⁵⁰`, and for `|t| > 2⁴⁰` while `σ < 56` (the phases `t ln n` would need more than double-double arithmetic); on `σ = 1` below `t = 2⁻⁵⁰⁰` the value `γ − i/t` (`−i∞` where `1/t` leaves the range); a value past the range is an infinity in the direction of each part, and near a trivial zero with a tiny `t` the imaginary part `t ζ′(−2n)` is kept to 3 ulps of itself (to the subnormals, and finite wherever it is in the range), and so is the real part, of the order of `t²`, from `σ = −12` on (with its sign where it is below the subnormals); at `−2, …, −10`, where `Γ(1 − s)` carries it and the phases of the reflection's factors cancel in it to a sixtieth of their size, the real part is good only to about 60 ulps of itself, and where it is below the subnormals (always below `t = 2⁻⁶⁰⁰`) it is a zero of either sign. Each part is within the bound of the table, which is relative to `|ζ|`.
- `hurwitz_zeta(s, a)` is `ζ(s, a) = Σₙ (n + a)⁻ˢ` for `s > 1` and `a > 0` (`+∞` at `s = 1`, NaN outside), with `ζ(s, 1) = ζ(s)` exactly; any `a` from the least subnormal to `f64::MAX`.
- `dilog(x)` is the dilogarithm `Li₂(x) = −∫₀ˣ ln(1 − t)/t dt` of any real `x`, its real part `π²/6 − ∫₁ˣ ln|1 − t|/t dt` past 1 (as GSL's `gsl_sf_dilog`); `Li₂(1) = π²/6`, `Li₂(±∞) = −∞`.
- `polylog(s, x)` is the polylogarithm `Li_s(x) = Σ xᵏ/kˢ` of a real order `s`, for `x ≤ 1` (NaN past 1): `ζ(s)` at `x = 1` for `s > 1` and `+∞` there for `s ≤ 1`, `−ln(1 − x)` for `s = 1`, `x/(1 − x)` for `s = 0`, `dilog(x)` for `s = 2`; for `x < −1` the real value of the continuation. Every order is supported: `x` itself once `2⁻ˢ|x| < 2⁻⁶⁰` (so `Li_s(x) → x` as `s → ∞`), `x/(1 − x)` for a positive order below `2⁻¹⁰²²` (to far below an ulp), and the large negative orders with an infinity where the value leaves the range (`+∞` for `0 < x < 1`); for `s < −2⁵⁰` and `x < 0`, where the sign of such an infinity would need more than double-double arithmetic, NaN. `Li_s(−∞)` is `−∞` for `s > 0`, `−1` for `s = 0`, a zero for `s < 0`.
- `lambert_w0(x)` and `lambert_wm1(x)` are the two real branches of the Lambert W function, the `w` with `w eʷ = x`: `W₀ ≥ −1` for `x ≥ −1/e`, `W₋₁ ≤ −1` for `−1/e ≤ x < 0` (`W₋₁(0⁻) = −∞`). Both are `−1` at the `f64` and the `f32` nearest `−1/e`, which lie just below it; below them both are NaN.
- `expint_e1(x)` is `E₁(x) = ∫₁^∞ e^(−xt)/t dt` for `x > 0` (`+∞` at 0, NaN for `x < 0`); `expint_ei(x)` is `Ei(x) = −PV ∫₋ₓ^∞ e^(−t)/t dt` for `x ≠ 0` (`−∞` at 0, `−E₁(−x)` for `x < 0`), with its zero `x₀ = 0.372 507 41…` exact to the last bit; `expint_en(n, x)` is `Eₙ(x) = ∫₁^∞ e^(−xt)/tⁿ dt` for `x ≥ 0` (`e⁻ˣ/x` for `n = 0`, `+∞` where that overflows; `1/(n − 1)` at `x = 0` for `n ≥ 2`).

### Error bounds

Measured against mpmath at 256 bits (the Hurwitz references by an Euler–Maclaurin sum of their own at 256 bits, since mpmath's `zeta(s, a)` loses digits for large `a`; `Eₙ` by `x^(n−1) Γ(1 − n, x)` at 1024 bits; the polylogarithm of a negative integer order by exact rational arithmetic and of the other negative orders by the Hurwitz zeta function of a complex argument or Jonquière's formula, as `mpmath.polylog` is not reliable there) on the data of the tests, which end with regression sets at the arguments of the audit of 2026-11-21 and of the first and second reviews of the fixes (2026-11-22), and on larger random surveys against Arb (FLINT 3, at least 80 bits of the reference demanded): the bound each test enforces and, in brackets, the largest error the surveys found (counts given). In ulps of the result for `f64`, on Apple silicon (with fused multiply-adds); every `f32` result is within 0.501 ulp of the true value, that is correctly rounded but for the rare halfway cases (complex: 0.6 ε of `|ζ|`, and a subnormal part within half a unit of the least subnormal) [all surveys: at most 0.5000 ulp, complex 0.4999 ε, each part at the trivial zeros 0.4997 ulp].

| Function | Region | Bound (ulps) [survey] |
| --- | --- | ---: |
| `zeta`, real | `s ≥ 0` | 0.7 [0.60, 22,708 cases] |
| | `−1/2 ≤ s < 0` | 1.05 [0.74, 30,000] |
| | `s < −1/2` (the reflection) | 3.5 [3.00, 21,189] |
| `zeta`, complex | `σ > 1` | 2 ε of \|ζ\| [1.43 at the edge of the series about the pole, 0.78 elsewhere; 19,200 to `t = 2⁴⁰`] |
| | `0 ≤ σ ≤ 1` | 3 ε of max(\|ζ\|, 1) [below `t = 13.5` 2.08, 8,900, and 1.84 at a point of the data, `0.483 + 3.031i`; to `t = 10⁴` 0.89, 11,600; past it, the Riemann–Siegel formula to `t = 2⁴⁰`, 0.86, 12,700, and 0.73 on 14,400 more by height within 0.01 of `σ = 1/2`]; 2.5 ε absolute at the first 29 zeros [0.37] |
| | `σ < 0` | 2.5 ε of \|ζ\| [below `t = 13.5` 1.29, 31,700; above it 1.31, 8,500; at the switch points of the methods 1.38, 5,600; tiny `t` at the trivial zeros 1.02, 1,800] |
| `hurwitz_zeta` | normal results | 1.35 [1.05, 14,014] |
| | subnormal results | 3.5 units of `2⁻¹⁰⁷⁴` [3.24] |
| `dilog` | all | 1.3 [1.25 about the zero past 1, 35,470; elsewhere 0.84] |
| `polylog` | `s ≥ 1`, `−2 ≤ x ≤ 1` | 3 [2.84 on `0.75 < x < 0.8`, 13,341; 2.70 on 17,400 more by region and 2.54 on 34,000 about `x = −1`] |
| | `0 ≤ s < 1`, `−2 ≤ x ≤ 1` | 4.5 [4.19 just past `x = −3/4`, 88,000, among them 36,000 on `−0.9 ≤ x < −3/4`] |
| | integers `s ≥ 1`, `x < −2` | 2 [1.20, 3,160] |
| | `s > 0`, `x < −2` (the integral of Fermi and Dirac over `Γ(s)`) | 4 [3.71, 32,500, by the paths, by order to `s = 400` and by binade of `s` in Sommerfeld's expansion; 3.81 at a point of the data, `(8.589, −31.55)`] |
| | integers `s < 0`, `x ≥ 0` | 2.5 [0.90, 16,000] |
| | integers `s < 0`, `x < 0` | 2.5 κ [1.62 κ, 13,500 on the final code; 1.70 κ on 18,500 before the second review, among them 14,000 with `\|x\|` near `2^(±\|s\|)`] |
| | `s < 0`, `x < 0` | 4 κ [1.69 κ, 41,500, among them 30,500 with `−1 < s < 0` by decade of `\|x\|` to `10³⁰⁰`, at most 1.02 κ] |
| | `s < 0`, `x > 0` | 3 [1.89, 8,700] |
| `lambert_w0` | all | 1.0 [0.98, 20,142] |
| `lambert_wm1` | all | 1.1 [0.98, 21,088] |
| `expint_e1` | `x ≤ 2.5` | 1.1 [0.90, 14,828] |
| | `x > 2.5` (Boost.Math's rational) | 1.7 [1.58 on 30,000 cases in `[2.5, 4]`; 1.67 at 2.565] |
| `expint_ei` | `x > 0` | 1.45 (1.8 on x86_64 without FMA) [1.23 on `(0, 6]`, 560,000, among them 300,000 on `[4.5, 4.75]`; 0.98 on `(6, 717]`, 434,800] |
| | `x < 0` (`−E₁(−x)`) | as `expint_e1` of `−x` [1.45] |
| `expint_en` | `n ≥ 2` | 1.05 [0.94, 8,760] |
| | `n = 1` (`E₁`) | as `expint_e1` |
| | `n = 0` | 0.6 [0.50, 320] |

For `s < 0` the polylogarithm has real zeros on the negative axis, where no method keeps a relative bound; there the bound is in units of its condition number (at least 1), as for `ellip_pi_inc`: the error is at most that many times `κ` ulps. For an integer order, which every method takes exactly, `κ = |x ∂ₓLi_s|/|Li_s| = |Li_(s−1)/Li_s|`; for the others `κ = (|x ∂ₓLi_s| + |s ∂ₛLi_s|)/|Li_s|`. The bounds that rest on `Γ` (the real `ζ` for `s < −1/2`, the complex one's reflection with `Γ(1 − s)` for `−12 < σ < 1/2` below `t = 13.5`, `Li_s` of non-integer orders past `−2` and near 1) were measured with the gamma area's `gamma_f64` and `gamma_c64`.

Two bounds were loosened from those of the first version, each where the method's roundings in double allow no better (none where the function's conditioning would): `expint_e1` past 2.5 from 1.1 to 1.7 (Boost.Math's rational of degree 10/11 in `1/x` costs up to 1.43 ulps of `x eˣ E₁` in its own roundings at its left end, and 0.32 in its fit), and the integral of Fermi and Dirac from 3 to 4 (about 300 positive quadrature terms summed in double and divided by `Γ(s)`). Every other bound holds as stated, or tighter: the complex function, stated before as 3.5 ε in the strip (6.5 ε left of `σ = 1/2` near the real axis), 2.5 ε left of the strip near the real axis and 8 ε growing as `2.2^(−σ)` elsewhere left of it (where the Euler–Maclaurin terms cancelled), is now 3 ε in the strip and 2.5 ε left of it everywhere, to `t = 2⁴⁰`. In the strip, where the bound is absolute near the zeros, that takes the terms of its sums as double-doubles: rounded to doubles, the errors of the `√(t/2π)` terms of the Riemann–Siegel sums add up as a random walk (7 ε at `t = 10¹⁰`), and those of the sums of `ζ(1 − s)` are multiplied by `|χ(s)| ≈ (t/2π)^(1/2 − σ)` in the reflection (6 ε at `t ≈ 10⁴`). The surveys after the first review of the fixes found three more places over their bounds, which no earlier survey had sampled, and they are fixed: `Li_s` for `s > 0` with `−3/4 ≤ x < −1/2` by the expansion about −1 (3.25 ulps for `s ≥ 1`, 6.4 below; its first coefficient `η(s)` lies in the binade above `Li_s` and its first two terms cancel), now the series; the complex function left of the strip below `t = 13.5` in its logarithmic form (2.57 ε, the sine and its product with `ζ(1 − s)` in double); and `Li_s` of `−20 < s < −17.5` near `x = 1`, NaN where the value leaves the range, now `+∞`. The second review of the fixes found four more, all fixed: `Li_s` for `−1 < s < 0` far left (5.9 κ at `x = −6·10¹⁶`, 84 κ at `−10²⁸⁷`), where the pole pair of its Hurwitz form, then taken out for every `s > −1`, is of size `|a|^(1−s)` against a result of `|a| sin(π|s|)/|s|` and its parts were rounded in double; `Ei` on `(10, 11]` and past 30 (2.13 ulps), whose rational fits past 6 were evaluated about the middle of their pieces, where both polynomials alternate and cancel at the left end by up to 95; `ζ(1 + it)` for `0 < |t| < 1.5·10⁻¹⁵⁴`, NaN or `(∞, NaN)` where the value is finite (the series about the pole squared `t`); and `Li_s` of a positive subnormal order past `x = −2`, NaN.

### The method

- **Real `ζ`.** `s ≥ 56`: `1 + 2⁻ˢ`. `3 ≤ s < 56`: `1 + 2⁻ˢ V(s)`, `V = (ζ − 1)·2ˢ` a rational minimax fit on `[3, 8]`, `[8, 20]` and `[20, 56]` (degrees 6/6, 6/6, 4/4), weighted relative to `ζ`. `−1/2 ≤ s < 3`: `1/(s − 1) + R(s)` with `R = ζ(s) − 1/(s − 1)` entire, a polynomial minimax fit of degree 16; the reciprocal (with the low part of `s − 1`), the head `c₀ + c₁(s − 1)` of `R` (the low part of `s − 1` through `R′`) and the sum are double-doubles, so that the cancellation below 1 (by 3 left of 0) costs nothing. `s < −1/2`: `ζ(s) = 2 (2π)^(s−1) sin(πs/2) Γ(1 − s) ζ(1 − s)`, `1 − s` exact as a double-double whose low part enters through `ζ′/ζ` and `ψ − ln 2π`, `sin(πs/2)` with the argument reduced exactly, the product `2 sin(πs/2) ζ(1 − s)` as a double-double and `Γ(1 − s)(2π)^(s−1)` from a double-double exponent so that the result is rounded once; past `1 − s = 171`, where `Γ` alone leaves the range, `ln Γ(1 − s) − (1 − s) ln 2π` in double-double (Stirling's series) as one exponent.
- **Complex `ζ`.** `σ ≥ 56`: `1 + 2⁻ˢ`. Near the pole (`−1/2 ≤ σ ≤ 3/2`, `|s − 1| ≤ 3`): the Taylor series `1/(s − 1) + Σ_{n<30} (−1)ⁿγₙ/n! (s − 1)ⁿ` with `γₙ` the Stieltjes constants, whose terms all fall below the first, the reciprocal as a double-double. Else for `σ ≥ 1/2`: the Euler–Maclaurin formula `Σ_{n<N} n⁻ˢ + N^(1−s)/(s − 1) + N⁻ˢ/2 + Σ_{k≤24} B₂ₖ/(2k)! (s)₂ₖ₋₁ N^(−s−2k+1)` with `N = ⌈(|s| + 40)/2.83⌉` for `t ≤ 10⁴` (`t ≤ 10³` for `σ ≥ 10`); the Dirichlet series itself for `σ ≥ 10` past it (at most 80 terms); and the Riemann–Siegel formula for any `σ` as J. Arias de Reyna gives it (*High precision computation of Riemann's zeta function by the Riemann–Siegel formula, I*, Math. Comp. 80, 2011) for `0 ≤ σ < 10`, `t > 10⁴`: `ζ(s) = R(s) + χ(s) conj R(1 − σ + it)`, `R` the sum to `N = ⌊√(t/2π)⌋` and nine terms of the correction `(−1)^(N−1) U a^(−σ) Σ_k a^(−k) Σ_ℓ d_{k,ℓ}(σ) F^(3k−2ℓ)(p)/(π^(2k−ℓ)(2i)^ℓ)` (its `d` by the recurrence of the paper, `F` the entire `(e^(πi(z²/2+3/8)) − i√2 cos(πz/2))/(2 cos πz)` by its Taylor series), so that the cost grows as `√t`; on the critical line one sum serves both parts, and `χ(s) conj R(1 − σ + it)` is a double-double. Each `n⁻ˢ` has its phase `t ln n/2π` as a double-double reduced exactly (`ln n` to `2⁻¹⁰⁰` past `t = 10⁶`), and its modulus from a double-double `ln n`; in the strip and in the Riemann–Siegel sums each term is itself a double-double, to about `2⁻⁶²` (its sine and cosine from a table of `sin(πj/32)`, `cos(πj/32)` and the series of the rest of the angle); the terms are added from the smallest, in double-double. Left of `σ = 1/2` (but for `0 ≤ σ < 1/2` past `t = 10⁴`): the reflection `ζ(s) = χ(s) ζ(1 − s)`, so that every sum is one of decaying terms (the Euler–Maclaurin formula at `σ` itself adds terms of size `n^(−σ) ≥ 1` and reached 7.5 ε of `|ζ|` for `−0.3 ≤ σ < 0`), with `ζ(1 − s)` at `1 − σ` taken as an exact pair (its rounding, times `ζ′/ζ ≈ ½ ln t`, cost up to 13 ε near `σ = 1/2`). Where `|1 − s| ≥ 13` (`t ≥ 13.5` or `σ ≤ −12`): `ln χ(s)` itself, its modulus and its phase as double-doubles from Stirling's series of `ln Γ(1 − s)` with `ln|1 − s|` and `arg(1 − s)` in double-double, and `ln sin(πs/2) = πt/2 − ln 2 + iπ(1 − σ)/2` (exact but for `e^(−πt)`) or, below `t = 13.5`, without the sine and with the parts of size `1 − σ` of the phase cancelled in closed form (the sine and its product with `ζ(1 − s)` then double-doubles, the sine scaled to about 1), so that the phase, of the order of `t`, keeps its digits for any `t` (at a trivial zero the real part of `ζ` is that phase times the imaginary part); nothing leaves the range before the last product. Else (`−12 < σ < 1/2`, `t < 13.5`) with `Γ(1 − s)` from `gamma_c64`, corrected by `ψ(1 − s)` for the low part of `1 − σ`, and `2(2π)^(s−1)`, `sin(πs/2)` and the products as double-doubles, the large factors multiplied first (so that `sin(πs/2)` near a trivial zero does not underflow). Below `t = 2⁻⁶⁰⁰` the function at `t′ = 2⁻⁶⁰⁰·m` with its imaginary part (and at a zero of `ζ(σ)` its real part, of order `t²`) scaled back, from `σ = −12` within the last products of the logarithmic form, so that `t ζ′(σ)` is not lost to an overflow at `t′` (`|ζ′(−400)| ≈ 10⁵⁴⁹`); on `σ = 1` below `t = 2⁻⁵⁰⁰` the value `γ − i/t`, whose rest `−iγ₁t − γ₂t²/2 + …` is below `2⁻¹⁰⁰⁰` of it (the series about the pole would square `t`).
- **Hurwitz `ζ`.** The Euler–Maclaurin formula with 24 Bernoulli terms from `A = a + N ≥ 7 + s/2`, or the direct sum alone once its terms fall below `2⁻⁵⁶` of the first and the rest past them (at most `(m + a)^(1−s)/(s − 1) + (m + a)⁻ˢ`) does too; every term positive, added from the smallest in double-double, `n + a` exact, the first term as a double-double. The remainder as `A^(1−s) (1/(s − 1) + (1/2 + Σ…)/A)` with `A` a double-double: one power, which does not leave the range where the result does not.
- **`Li₂`.** On `−1 ≤ x ≤ 3/4` the series in `y = −ln(1 − x)` (`Li₂ = y − y²/4 + y³ P(y²)`, `P` a minimax fit of degree 8), `y` corrected to a double-double by `(1 − x) eʸ − 1`, `y − y²/4` exact; elsewhere the reflection `π²/6 − ln x ln|1 − x| − Li₂(1 − x)` and the inversions `∓π²/6 − ½ ln²|x| − Li₂(1/x)` with double-double constants, logarithms, products and series, rounded once; about the zero `12.595…` of `Re Li₂` a polynomial in `x − x₀`.
- **`Li_s`.** `2⁻ˢ|x| < 2⁻⁶⁰`: `x`. `0 < s < 2⁻¹⁰²²`: `x/(1 − x)`, within `40s` of it relative. `|x| ≤ 3/4` (`|x| ≤ 1/2` for `s < 0`, and every `x ≥ −1` for `s ≥ 20`; past `|x| = 1/2` the expansions about `±1` cancelled, by up to 3 and 1.6, and reached 4 and 6.4 ulps): `Σ xᵏ k⁻ˢ`, `p⁻ˢ` of the primes from double-double logarithms and the rest by products, for `s < −1` each term from logarithms. `3/4 < x < 1` (`1/2 < x < 1` for `s < 0`): `Γ(1 − s)(−μ)^(s−1) + Σ ζ(s − k) μᵏ/k!`, `μ = ln x` a double-double (relative to itself near 1), `Γ(1 − s)` with the low part of `1 − s`; for `n = 1` (`n` the integer nearest `s ≥ 1/2`) and near the integers past it the pole of `ζ(s − k)` at `k = n − 1` and that of `Γ(1 − s)` taken together in closed form (through `ln Γ(1 − ε)/ε`, `ln Π(1 + ε/j)/ε` and `expm1` of a double-double); `ζ(s − k)` past `s − k < −1/2` by the reflection as a recurrence in `k`; for `s < −20` the first term alone, `ln Γ` in double-double; `+∞` once that first term leaves the range. The orders `s = 1/2, 3/2, 5/2` and `3` (the Fermi–Dirac integrals `F₋₁/₂`, `F₁/₂`, `F₃/₂` and `F₂`) on `−7.3·10²⁶ ≤ x < 0`: fits of 10⁻²¹ made for this crate (Chebyshev interpolation of mpmath's values at 45 digits), `x·G(x)` with `G` a polynomial in `x + 3/8` on `[−3/4, 0)`, `−F(μ)` with `F` a polynomial in `μ = ln(−x)` on `[−2, −3/4)`, and `−μˢ R(μ)/Γ(s + 1)` past `−2` with `R` on nine pieces of `ln μ` of width 1/2, `μ`, `ln μ` and the prefactor as double-doubles (each the first two terms of its polynomial as a double-double). `−2 ≤ x < −3/4`: `−Σ η(s − k) μᵏ/k!`. `x < −2`, `s > 0`: integers by the inversion `−(−1)ⁿ Li_n(1/x) − Σ 2η(2j) μⁿ⁻²ʲ/(n − 2j)!`, the sum nested as `(μⁿ/n!)[1 + r₁(2η(2) + r₂(…))]` in double-double so that its cost is linear in `n` and no factorial overflows; the others by the integral of Fermi and Dirac, `Li_s(−eᵘ) = −F(μ)/Γ(s)`, every term positive (on `[0, min(1, μ)]` the Taylor series of the logistic integrated exactly, past it 16-point Gauss–Legendre pieces about exact midpoints), or, where they apply, the divergent series cut at its least term (orders large against `ln|x|`) and Sommerfeld's expansion (`x < −e^37.7`, `μ` a double-double, `1 + s` an exact pair with its low part through `ψ`, past `s = 160` `ln Γ(1 + s)` in double-double); for `s > 21` the powers at the quadrature's nodes from double-double logarithms, and where `t^(s−1)` would leave the range `ln Γ(s)` in double-double. `s < 0`, `x < 0` (an integer order with `x < −1` by `Li_s(x) = (−1)^(s+1) Li_s(1/x)`, `ln(−1/x) = −μ` exactly): the series, each term from logarithms as a double-double, where `|x| ≤ 1/2` and no term is above about 4000 times the first (the largest is at `k ≈ |s|/ln(1/|x|)`), so that the cancellation of its alternating terms costs nothing; else `Li_s(x) = 2 Γ(1 − s)(2π)^(s−1) Re[i^(1−s) ζ(1 − s, 1/2 − iμ/2π)]` with the Hurwitz zeta function of a complex argument by the Euler–Maclaurin formula, each term from `ln|n + a|` and `arg(n + a)` in double-double relative to the first, that first's logarithm joined with `ln(2 Γ(1 − s)(2π)^(s−1))` in one exponent, the direct terms stopped where the rest is negligible, and the remainder past them, its products and `Re[i^(1−s) Z]` complex double-doubles, rounded once in the last product (`σ − 1 = −s` exactly); for `−1/4 < s < 0` the pole of `ζ(σ, a)` at `σ = 1` taken out of the remainder (`(w^(1−σ) − 1)/(σ − 1) = −expm1(s ln w)/s`, by its series or from `e^(s ln w)` in double-double) and its part `−sin(πs/2)/s` added in closed form, so that parts of size `1/|s|` do not cancel to `O(1)`; further left the pair is not needed (in units of `|a|^(−σ)` it is of size `|a|^(1−s)`, against a result of `|a| sin(π|s|)/|s|`). The real part cancels near the real zeros of `Li_s`, by `1/sin(π|s|) ≈ κ/π` as `s → −1`, and also where `|x|` is near `2^(±|s|)`, by about 50 for an integer order whose `κ` is 1 there: the series takes those. Integers `−18 ≤ s ≤ −2`, `0 < x < 1`: `x Aₙ(x)/(1 − x)ⁿ⁺¹` with Eulerian polynomials (their coefficients exact, below `2⁵³`) in double-double, `+∞` past the range.
- **The `f32` form of `Ei`** on `(0, 6]`: the `f64` method without its double-double steps (its terms have one sign), checked against the `f64` path at every positive `f32` up to 6.5.
- **Lambert W.** Near the branch point both branches are polynomials in `p = √(2(1 + e x))` (`±p`), `1 + e x` without the cancellation (`e` as a double-double), on `|p| ≤ 0.66`; `W₋₁` goes on to `x = −0.2` in `s = √(2(−ln(−x) − 1))`, `s = √(−2 ln(1 − q))` a double-double from the double-double `q`; each a minimax fit of full precision, its first two terms a double-double. `W₀` on `0.66 < p ≤ 4.31` (`−0.29 < x ≤ 3`) is `x e^(−W₀)` with `e^(−W₀)` a polynomial of full precision in `p` on five pieces (`p` a double-double, the product rounded once; its series below `|x| = 2⁻¹⁰`), without iteration. `W₀` on `3 < x ≤ e⁴⁰`: one of twenty polynomials of degree 7 in `ln x` (Chebyshev interpolants, picked by an index rather than a branch), good to `6.1·10⁻¹⁰`, and one step of Newton's iteration with the residual from a double-double `eʷ`; its `f32` form on `2⁻¹⁰ ≤ |x| ≤ 3` takes `x e^(−W₀)` with `e^(−W₀)` from nine polynomials of degree 13 in `p` of `f32` grade (`2⁻⁴⁹`, in `f64`, then rounded once to the `f32`; checked against the `f64` path at every `f32` of `[−1/e, 3]`). Elsewhere a fitted starting value good to `3·10⁻⁷` (in `s`, or the asymptotic series) and one step of Halley's iteration (in a form with one division) with the residual `w eʷ − x` from a double-double `eʷ` (so that its only error is the final rounding), or for `|w| > 40` one step of Fritsch's iteration (of the fourth order).
- **`E₁`, `Ei`, `Eₙ`.** `E₁`: `F(x) − ln x` on `(0, 1]` with the entire `F = E₁ + ln x = −γ + x − x²/4 + c₃x³ + x⁴T(x)` a minimax polynomial of degree 14, `ln x`, `x − x²/4` and `c₃x³` in double-double and the sum rounded once (near 1 the result is a quarter of its parts); `e⁻ˣ (H₀ + K(x))/x` on `(1, 2.5]` with `K` a rational fit (6/6) evaluated in `(x − 1)/0.75`, where its denominator's terms are all positive; past it Boost.Math's rational of degree 10/11 in `1/x`; `e⁻ˣ` to about `2⁻⁷⁵` and the product rounded once. `Ei`: `ln(x/x₀) + (x − x₀) R(x)` on `(0, 6]` (`R` entire, four polynomial pieces, `c₀ + c₁t` of each a double-double; `x − x₀` exact with `x₀` a triple-double; `ln(x/x₀)` a double-double, by `ln_1p` and a Newton step near `x₀` and up to 1.5; the sum rounded once); `eˣ (1 + K(x))/x` past 6 with `K` rational fits on four pieces to 48, each evaluated in a variable from the left end of its piece (where the terms of both polynomials are positive: about the middle of the piece they alternate and cancelled by up to 95 at its left end), and polynomials in `1/x` past it. `Eₙ`, `n ≥ 2`: the recurrence `Eₖ₊₁ = (e⁻ˣ − x Eₖ)/k` up from a double-double `E₁` for `x ≤ 2` and `n < 50`, else the continued fraction, its depth from Lentz's test (and a quarter more) and its value from the bottom up in double-double arithmetic; `E₀ = e⁻ˣ/x`.

The fits are minimax (Remez exchange for the polynomials, Lawson's iteration for the rationals, mpmath at 128 bits) made for this crate, but for `E₁` past 2.5, Boost.Math's. The other tables (the Riemann–Siegel function's Taylor coefficients, the Stieltjes constants, `atan(k/8)`, Stirling's coefficients) are printed by `zeta/gen_complex.py`.

### Accuracy measured

The accuracy above was measured by the tests (`cargo test --test math -- special::zeta` prints the largest error of each region on the data of `tests/math/special/data/zeta_*.txt`, regenerated by `data/gen_zeta.py`) and by surveys against Arb outside the repository (16,139 complex points for `ζ` and about 140,000 more by the regions of its bounds, 73,897 real ones, 14,700 for `ζ(s, a)`, 35,470 for `Li₂`, 5,500 random and 12,000 targeted for `Li_s`, about 20,000 a branch of W, 19,344, 24,864 and 9,080 for `E₁`, `Ei`, `Eₙ`, each repeated for `f32`). The release and the debug builds give the same bits on 66,383 of these arguments (the first version gave one polylogarithm 8 ulps apart between them). After the first review of the fixes (2026-11-22) the functions it touched were surveyed again on the final code: the complex function on 66,600 points by the regions of its bounds to `t = 2⁴⁰` (two samples) and as many in `f32`, 14,400 by height near `σ = 1/2`, 11,900 at the switch points of its methods, 22,000 left of the strip below `t = 13.5` and 6,800 at the trivial zeros with a tiny `t` (5,000 of them with each part against mpmath at `120 + log₂(1/t)` bits, since Arb's references are good only relative to `|ζ|`); `Li_s` on about 290,000 (59,400 by region and 48,500 in `f32`, 22,900 past `−2` by path, by order and by binade, 20,900 with `|x|` near `2^(±|s|)`, 6,000 of the Eulerian form, and 130,000 about the switch points `x = ±1/2`, `±3/4` and `−1`); `Ei` on 560,000. The references of its regression rows agree with Arb to `2⁻⁹⁵` or better, every set of them fails on the code before it, and the release and the debug builds gave the same bits on 90,314 arguments. After the second review (2026-11-22): `Li_s` with `s < 0`, `x < 0` on 41,500 points (30,500 with `−1 < s < 0` by decade of `|x|` to `10³⁰⁰` and about the switch at `s = −1/4`, 3,000 each with `s` in `(−3, −1)` and `(−30, −3)`, 5,000 in the regions of the general survey) and 7,500 integer orders with `x < 0`, the general survey again (63,300 points in 37 regions, 58,986 in `f32`) and 2,030 switch-point and edge arguments; `Ei` on 434,800 points of `(6, 717]`; the complex function at 874 edge arguments (switch points, tiny and subnormal parts). The release and the debug builds now give the same bits on 121,893 arguments (90,893 from the data files and the audit's surveys, and the second review's 31,000), three of which (polylogarithms of `−1 < s < 0` far left) had differed in the last bit: the release build merged a sine and a cosine of one argument into one call of the system's `sincos`, whose last bit can differ from theirs; that code is gone (the pole's remainder takes this crate's double-double sine and cosine).

### Measured

As for [the gamma function](#measured) (the conditions and the same-work rule are there):

| function | arguments | `a` | within the bound | faster, beyond the bound |
| --- | --- | ---: | --- | --- |
| `zeta` | s in [1.1,3] | 4.19 ns | 0.711 SciPy |  |
| `zeta` | s in [3,56] | 4.30 ns | 0.553 Boost |  |
| `zeta` | s in [-0.5,0.9] | 4.11 ns | 0.330 Boost |  |
| `zeta` | s in [-20,-0.5] | 27.9 ns | – | 0.677 Boost† (11.9 ulps) |
| `zeta` | s in [-170,-20] | 29.0 ns | – | 0.982 SciPy† (6.3·10³ ulps) |
| `zeta`, complex | s in [0.5,5]+i[-50,50] | 687 ns | – | 0.399 SciPy† (1.5·10⁸ ulps) |
| `zeta`, complex | s in [-10,0.5]+i[-20,20] | 786 ns | – | 0.435 SciPy† (110 ulps) |
| `zeta`, complex | s = 1/2+i[100,1000] | 5.65 µs | – | 1.54 SciPy† (2.2·10⁶ ulps) |
| `zeta`, complex | s in [1.1,10]+i[-5,5] | 477 ns | 0.310 SciPy |  |
| `hurwitz_zeta` | s in [1.5,10], a in [0.1,10] | 91.1 ns | – | 1.10 GSL† (4.4 ulps) |
| `hurwitz_zeta` | s in [1.5,10], a in [10,1e4] log-uniform | 30.7 ns | – | 0.330 GSL† (4.8 ulps) |
| `dilog` | x in [-1,1] | 15.6 ns | – | 2.41 SciPy† (13.7 ulps) |
| `dilog` | x in [-100,-1] | 22.1 ns | – | 2.39 SciPy† (7.8 ulps) |
| `dilog` | x in [1,20] (real part) | 19.6 ns | – | 0.595 GSL† (9.9·10³ ulps) |
| `polylog` | s=0.5, x=-e^mu, mu in [0,10] | 28.3 ns | 1.53 GSL |  |
| `polylog` | s=1.5, x=-e^mu, mu in [0,10] | 29.3 ns | 1.64 GSL |  |
| `polylog` | s=1.5, x=-e^mu, mu in [-10,0] | 14.5 ns | 0.545 GSL |  |
| `polylog` | s=2, x=-e^mu, mu in [0,10] | 22.7 ns | 1.26 GSL |  |
| `polylog` | s=2.5, x=-e^mu, mu in [0,10] | 29.0 ns | 1.73 GSL |  |
| `polylog` | s=3, x=-e^mu, mu in [-10,0] (x in [-1,0)) | 13.7 ns | 0.542 GSL |  |
| `polylog` | s=3, x=-e^mu, mu in [0,30] | 28.2 ns | 1.25 GSL |  |
| `polylog` | s=6, x=-e^mu, mu in [0,30] | 171 ns | – | 2.12 GSL† (23.8 ulps) |
| `polylog` | s integer in [-8,-2], x=-e^mu, mu in [-5,5] | 820 ns | – | 44.3 GSL† (200 ulps) |
| `lambert_w0` | x in [-0.36,3] | 7.13 ns | 0.0856 GSL | 2.56 Boost† (8.8 ulps) |
| `lambert_w0` | x in [3,1e10] log-uniform | 11.1 ns | 1.68 Boost | 2.42 special† (3.6 ulps) |
| `lambert_wm1` | x in [-0.36,-0.01] | 12.1 ns | – | 2.47 puruspe† (3.8 ulps) |
| `lambert_wm1` | x in -[1e-300,0.01] log-uniform | 14.1 ns | 0.854 Boost | 1.72 special† (3.9 ulps) |
| `expint_e1` | x in [0.01,1] | 8.49 ns | 2.03 Boost |  |
| `expint_e1` | x in [1,50] | 7.91 ns | 1.92 Boost |  |
| `expint_e1` | x in [50,700] | 7.86 ns | 1.89 Boost |  |
| `expint_ei` | x in [0.01,6] | 10.3 ns | – | 1.64 Boost† (6.1 ulps) |
| `expint_ei` | x in [6,50] | 7.85 ns | 0.226 GSL | 1.35 Boost† (3.9 ulps) |
| `expint_ei` | x in [50,700] | 7.11 ns | 1.55 Boost |  |
| `expint_ei` | x in [-50,-0.01] | 8.83 ns | 1.92 Boost |  |
| `expint_en` | n=3, x in [0.1,2] | 17.7 ns | – | 0.111 statrs† (NaN or ∞ on 11 of 16384) |
| `expint_en` | n=3, x in [2,20] | 137 ns | – | 1.63 Boost† (21.5 ulps) |
| `expint_en` | n=20, x in [0.1,2] | 67.2 ns | – | 0.794 Boost† (15.3 ulps) |
| `expint_en` | n=20, x in [2,20] | 126 ns | – | 1.58 Boost† (12.2 ulps) |
| `expint_en` | n=100, x in [0.1,50] | 72.7 ns | – | 4.62 SciPy† (19 ulps) |
| `lambert_w0`, `f32` | x in [-0.36,3] | 2.67 ns | – | 1.09 Boost† (15.7 ulps) |
| `expint_ei`, `f32` | x in [0.01,6] | 8.60 ns | 1.35 Boost |  |
| `zeta`, `f32` | s in [1.1,3] | 4.11 ns | 0.634 Boost |  |

### Where the references are faster

- **`polylog` past `x = −1`**, `Li_s(−e^μ)` for `μ` in `[0, 10]` at `s` = 0.5, 1.5 and 2.5 (1.53 to 1.73), at `s = 2` (1.26) and for `μ` in `[0, 30]` at `s = 3` (1.25), against GSL's Fermi–Dirac integrals, Chebyshev fits within 2.0 to 5.0 ulps of `a`: `a` takes `μ = ln(−x)` and `ln μ` as double-double logarithms and the prefactor in double-double before its fits, for its bound of 4 ulps.
- **`expint_e1`** on `[0.01, 1]`, `[1, 50]` and `[50, 700]` (2.03, 1.92, 1.89) and **`expint_ei`** on `[−50, −0.01]` and `[50, 700]` (1.92, 1.55), against Boost (within 2.0 to 3.4 ulps of `a`), and `expint_ei` of `f32` on `[0.01, 6]` (1.35): the double-double logarithm below 1 and the exponential to 2⁻⁷⁵ beyond, for bounds of 1.1 to 1.7 ulps.
- **`lambert_w0`** on `[3, 10¹⁰]` (1.68 against Boost): the Halley step of the double-double exponential that holds the bound of one ulp.

### Examples

```rust
use a::math::complex::Complex;
use a::math::special::{dilog, expint_ei, hurwitz_zeta, lambert_w0, lambert_wm1, polylog, zeta};

assert_eq!(zeta(2.0f64), core::f64::consts::PI.powi(2) / 6.0);
assert_eq!(zeta(-4.0f64), 0.0);
let z = zeta(Complex::new(0.5f64, 14.134_725_141_734_694)); // the first zero
assert!(z.abs() < 1e-14);
// far up the critical line (the Riemann–Siegel formula), and far right of it
let z = zeta(Complex::new(0.5f64, 1e8));
assert!((z.re + 3.362_839_487_530_728).abs() < 1e-14 && (z.im - 1.407_234_559_646_448).abs() < 1e-14);
assert_eq!(zeta(Complex::new(300.0f64, 1.0)).re, 1.0);
assert_eq!(hurwitz_zeta(3.0f64, 1.0), zeta(3.0f64));
assert!((dilog(-1.0f64) + core::f64::consts::PI.powi(2) / 12.0).abs() < 1e-15);
assert!((polylog(3.0f64, 0.5) - 0.537_213_193_608_040_2).abs() < 1e-15);
assert_eq!(polylog(-2.0f64, -1.0), 0.0); // Li₋₂ₖ(−1) = 0
let w = lambert_w0(1.0f64); // the omega constant
assert!((w * w.exp() - 1.0).abs() < 1e-15);
assert!(lambert_wm1(-0.1f64) < -1.0);
assert!(expint_ei(0.372_507_410_781_366_6f64).abs() < 1e-16);
```

## Elliptic integrals and functions

All of them take the parameter `m = k²` (as SciPy, mpmath and Cephes do; Boost and GSL take the modulus `k`).

- `ellip_k(m)` and `ellip_e(m)` are the complete integrals of the first and second kind, `K(m) = ∫₀^(π/2) dθ/√(1 − m sin²θ)` and `E(m) = ∫₀^(π/2) √(1 − m sin²θ) dθ`, for every `m < 1` (`K(1) = ∞`, `E(1) = 1`; NaN above 1; `K(−∞) = 0`, `E(−∞) = ∞`).
- `ellip_pi(n, m)` is the complete integral of the third kind `Π(n, m) = ∫₀^(π/2) dθ/((1 − n sin²θ)√(1 − m sin²θ))`, its Cauchy principal value for `n > 1` (the real part of mpmath's value; `Π(1, m) = ∞`, `Π(±∞, m) = 0`).
- `ellip_f(φ, m)`, `ellip_e_inc(φ, m)` and `ellip_pi_inc(n, φ, m)` are the incomplete integrals `F`, `E` and `Π` for any real `φ`: for `m ≤ 1` by the periodicity `F(φ + jπ) = F(φ) + 2jK` (and so for `E` and `Π`), for `m > 1` where they are real, `|sin φ| ≤ 1/√m` with `|φ| ≤ π/2` (NaN elsewhere; just past that edge, within the rounding of `sin φ`, the value at the edge). `Π` is the principal value past a pole `n sin²φ = 1`, and infinite at it; `Π(±∞; φ, m) = ±0` with the sign of `φ`; for `m = 1` past `±π/2` it diverges as `2jΠ(n, 1)`, to `−∞` for `n > 1`.
- `carlson_rf`, `carlson_rd`, `carlson_rj` and `carlson_rc` are Carlson's symmetric integrals of non-negative arguments, `R_J(x, y, z, p)` for `p < 0` and `R_C(x, y)` for `y < 0` their principal values. Two zero arguments of `R_F` (and `R_J`), `x = y = 0` or `z = 0` of `R_D`, `p = 0` and `y = 0` give `∞`, as does a value beyond `f64::MAX`; a negative `x`, `y`, `z` gives NaN; an infinite argument gives 0.
- `jacobi_elliptic(u, m)` returns `(sn(u, m), cn(u, m), dn(u, m))` for every real `u` and `m` (`m = 0`: `sin`, `cos`, 1; `m = 1`: `tanh`, `sech`, `sech`; an infinite `u` gives NaN but for `m = 1`), the argument reduced with `K` to as many bits as `u/K` has.

### Error bounds

Measured on the reference data (mpmath at 256 bits and more where its own precision was short; 7,860 cases of the elliptic functions, among them the arguments at which an audit of 2026-11-21, the checks of its fixes, two reviews of them and the confirmation of the second found defects, and their neighbours) and on sweeps of 360,000 more cases with two seeds each, about 1,000,000 on the paths of the second review's fixes (840,000 of them `F(φ, 1)`) and 650,000 of the Jacobi functions on their final code (12 seeds of 5,000 points per band of `μ`, the first per form, `f32` and the paths without FMA on 3, x86_64 itself on 3), against mpmath and Arb (the largest errors of all are given); `f32` results are rounded once from the `f64` computation and were correctly rounded on every case.

| Function | Region | Bound (ulps) | Measured |
| --- | --- | ---: | ---: |
| `ellip_k` | all `m < 1` | 0.6 | 0.58 |
| `ellip_e` | all `m ≤ 1` | 0.6 | 0.60 |
| `ellip_pi` | all (`n > 1`: the principal value; as `n → 1` the `R_J` term dominates, with its 2.4 ulps) | 3.5 | 3.21 |
| `ellip_f` | `m ≤ 1`, every `φ` (`sin φ` and `cos φ` rounded apart) | 2.5 | 2.18 |
| | `m > 1`, in units of `1 + κ`, `κ = |φ/(Δ F)|` its condition number in `φ` (unbounded at the edge of the real domain) | 2.5 (1 + κ) | 0.92 (1 + κ) |
| `ellip_e_inc` | `0 ≤ m ≤ 1`, every `φ` | 3.5 | 3.23 |
| | `m < 0` | 4 | 2.78 (2.91 on x86_64) |
| | `m > 1` | 4 | 3.24 |
| `ellip_pi_inc` | `n < 1`, `0 ≤ m < 1`, `|φ| ≤ π/2` (for `n < 0` most of the value can be the term `C atan(√(−nN) s c/Δ)` of A&S 17.7.15, whose argument and factor are products and quotients of rounded values: 2.0 ulps of that term, 78% of the value, at the largest error) | 3 | 2.75 (2.86 on x86_64) |
| | everywhere, in units of `1 + κ`, `κ = |φ/((1 − n sin²φ) Δ Π)|` its condition number in `φ` | 4 (1 + κ) | 3.39 (1 + κ) |
| `carlson_rf` | all | 0.75 | 0.50 |
| `carlson_rd` | arguments spanning at most `2^1600` (the largest over the smallest nonzero one), and `z > 2^120 max(x, y)` at any span | 0.75 | 0.58 |
| | spanning more with `z` not the largest (no scaling brings a subnormal `z` into the normal range), but two of them above about `2^1022` (NaN, see the limits) | 2 | 1.68 |
| `carlson_rj` | `p > 0` (where one term of the duplication is the whole value, `R_J` is that term `6 R_C(1, 1 + e₀)/d₀`, whose `R_C` the library rounds and the division rounds again) | 2.5 | 2.48 |
| | `p < 0`, in ulps of `max(|R_J|, R_J(x, y, z, −p))` (the first term of Carlson's (2.22) can be nearly all of the value, with the error of `R_J(x, y, z, p′)` and the roundings of `p′` and `p′ − z`) | 4.5 | 4.19 |
| `carlson_rc` | all (`y < 0`: the principal value, `√(x/(x − y))` and `R_C(x − y, −y)` rounded apart) | 2.5 | 2.41 |
| `jacobi_elliptic` | `μ < 0.9` (every `m`) | 3.5 | 2.49 (`sn`, `m > 1`; `dn` 2.05 for `m < 0`, 1.55 for `0 ≤ m < 1`) |
| | `0.9 ≤ μ < 1 − 10⁻⁶` | 20 | 10.8 |
| | `1 − 10⁻⁶ ≤ μ < 1` (`dn` near `u ≡ K/2 mod K`; also `m < −10⁶`) | 80 | 25.0 |
| every function, `f32` | all | 0.501 | 0.50 (correctly rounded) |

The Jacobi functions are computed at the parameter `μ`: `m` itself for `0 ≤ m < 1`, `μ = −m/(1 − m)` for `m < 0` and `μ = 1/m` for `m > 1` (A&S 16.10–11), and their bounds go by its band, growing as `μ → 1`, where `cn` and `dn` of `μ` are both about `μ₁^(1/4)` near `u ≡ K/2 (mod K)` and the condition number there is about `K/2`; for `m < 0` and `m > 1` the values are quotients of those at `μ` (`dn = 1/dn(w|μ)`, `cn = dn(w|μ)`, `sn = sn(w|μ)/√m`, …), whose errors add (for `μ < 0.9`, `dn` of `μ` is within 1.6 ulps, `1/dn` of it within 2.1, `sn/√m` within 2.5 ulps of `1/√m`). These are the bounds the original campaign stated for `0 ≤ m < 1`, now for every `m` by its band: with the Landen ascent rounded at each step (see Method) `μ < 0.9` reached 5.19 ulps for `m < 0` and 4.19 for `0 ≤ m < 1` (against the bounds of 5 and 3.5 stated then) and 3.74 for `m > 1`, and `0.9 ≤ μ < 1 − 10⁻⁶` 34.7 (against 32). They are in ulps of `max(|f|, 1)` for `sn` and `cn` (their zeros make relative error meaningless near them), relative for `dn` when `m ≤ 1`; for `m > 1` in ulps of `max(|sn|, 1/√m)`, relative for `cn` (which has no zeros there) and in ulps of `max(|dn|, 1)`.

Limits. The bounds of `ellip_f` and `ellip_e_inc` for `m ≤ 1` are those of the method, not of the conditioning: `sin φ` and `cos φ` are rounded apart, and `F(φ, m) = s R_F(c², Δ², 1)` of them is not `φ` exactly at `m → 0` (but `m = 0`, which returns `φ`). For `m > 1` a point of the real domain is evaluated where it is; where the computed `1 − m sin²φ` is negative but within `4ε m sin²φ` of 0 (just past the edge `|sin φ| = 1/√m`, within the rounding of `sin φ`), `F`, `E` and `Π` are those of the edge; beyond, NaN. `ellip_pi_inc` for `n ≫ 1` and `m ≪ 0` past a period, where an audit measured 1.74e4 (1 + κ) at `(6.99e164, −3.14, −1.14e256)` (an `R_J` that underflowed beside a huge factor), is within the bound now: at most 1.85 (1 + κ) on 408 swept cases, and that point correctly rounded. Where the arguments of `R_D` and `R_J` span more than about `2^1600`, no scaling by powers of 4 brings the singular one and the largest into the normal range together (for `R_D` with `z` not the largest: `z > 2^120 max(x, y)` goes by its expansion), and the duplication runs on a subnormal argument, whose square root and residuals are not exact: `R_D` reaches 1.68 ulps there (the row above; from a span of about `2^1620` on, up to 1 ulp at spans of `2^1624`, against 0.50 for every span up to `2^1600`) and `R_J` stays within its bounds where its result is above about `2^−1000` (2.13 ulps with `p > 0` on 2,000 cases of DLMF 19.21.10 at such spans), but a result at or within a few binades above the bottom of the normal range loses more: up to 97.5 ulps of a subnormal `R_J`, 30 on a sweep of 1,000 cases (`R_J(0, 7.66e−322, 2.06e302, 3.21e166)`), 26.4 scaled with `p < 0` (`R_J(0, 8.78e−320, 5.02e288, −3.00e169) = −2.51e−311`), and up to 24 of a normal one just above `2^−1022` (14.8 at `R_J(1.52e−315, 5.11e−321, 6.64e262, 2.54e179) = 2.61e−308`, a test pins it). With two arguments above about `2^1022` and another below `2^−1000` (a spread beyond the whole normal range), `R_D` and `R_J` return NaN although the integral is finite (`R_D(f64::MAX/2, f64::MAX/2, 5e−324)`, `R_J(f64::MAX/2, f64::MAX/2, 1, 5e−324)`; one argument that large is fine): an open defect, NaN rather than a wrong value. The tests run in the debug profile, where `sin_cos` is two calls; a release build on Apple platforms merges them into one (`__sincos_stret`), whose last bit differs at about 0.8% of the arguments, so a test can print a smaller maximum than the release-mode sweeps behind the "Measured" column (`ellip_pi_inc` in its plain region: 1.41 against 2.66).

### Method

- **`K` and `E`.** For `−1 ≤ m < 0.8`, 36 polynomials in `m − c` on intervals of width 0.05, Chebyshev fits of degree 9 (to 0.5) and 12 (to 0.8) made with mpmath at 200 bits, their constant terms double-doubles so that the result is rounded about once. For `m ≥ 0.8`, the logarithmic forms `K = −ln(m₁) K(m₁)/π + B(m₁)` and `E = 1 + m₁(−ln(m₁) S₁(m₁) + S₂(m₁))` (DLMF 19.12.1–2, `m₁ = 1 − m` exact there), whose terms are all positive, with `ln m₁ = k ln 2 + ln f`, `f` in `[1/√2, √2)`, a double-double whose rounding is that of `ln f` (below `2^−55`), and the inner sum of `E`'s form a double-double too (rounded once, or twice without a fused multiply-add, it cost up to 0.1 ulp: 0.601 at `m = 0.8911`). For `m < −1`, `K(m) = K(m′)/√(1 − m)` and `E(m) = √(1 − m) E(m′)` with `m′ = −m/(1 − m)` (DLMF 19.7.2) a double-double (the fitted polynomial and its derivative at its two parts), `1 − m′ = 1/(1 − m)` and `ln(1 − m)` double-doubles, `√(1 − m)` in double-double.
- **Carlson's integrals.** The duplication algorithm (Carlson 1995) with series to the thirteenth (`R_F`) and eleventh (`R_D`, `R_J`) orders, DLMF 19.19.7 expanded in the elementary symmetric functions of the deviations (by sympy), which stop it at deviations of 0.07 and 0.035 instead of the seventh order's 0.007: about one step fewer. Every argument is carried with its rounding error to first order: the square roots with their residuals, `λ` with the errors of its products and sums, the new arguments with the error of their sums, and the final `A^(−1/2)` and `A^(−3/2)` with the residuals of the square root and the reciprocal. A step then adds errors of second order only, and the result is about correctly rounded (it was on every case of `R_F` and `R_D`). The steps work on `λ/4` from half square roots; the integrals being homogeneous, arguments near `f64::MAX` are brought down and tiny ones up by powers of 4 (keeping the bits of the argument the integral is singular in: `z` or the larger of `x, y` for `R_D`, the middle one for `R_F`, `p` or the middle one for `R_J`), and a result beyond `f64::MAX` is `∞`. For `z > 2^120 max(x, y)` `R_D` is the leading terms of its expansion for `x, y ≪ z` (Carlson and Gustafson 1994), `(3/2) z^(−3/2) (ln(16z/(√x + √y)²) − 2)`, whose relative remainder is below `2^−110`, the logarithm from the mantissas and the exponent apart: for spans beyond `2^1600` no scaling brings the duplication into the range there (its subnormal results were up to 99% off). A term `4^(−m) R_C(1, 1 + e_m)/d_m` of `R_J` whose `1/d_m` or product with `R_C` leaves the range (two arguments far below it and one far above, as inside DLMF 19.21.10 below) is formed from its factors, largest by smallest. `R_J` carries `e_m = δ 4^(−3m)/d_m²` as a product of bounded ratios, and where `|e| ≥ 1/16` (where `R_C(1, 1 + e)` is sensitive to it) as the product of the three ratios `(√p − √x)/(√p + √x)` with their errors, then `R_C(1, 1 + e)` with the residuals of its square root and quotient; it takes `1 + e` without cancellation when `e` is near −1 (`p ≪ x, y, z`). For `|p| > 2^40 max(x, y, z)` (either sign) it goes by DLMF 19.21.10, `(p − y) R_J(x, y, z, p) + (q − y) R_J(x, y, z, q) = 3 R_F(x, y, z) − 3 R_C(xz/y, pq/y)` with `y` the middle argument and `q` next to it, whose terms are positive but the small `R_C`: the duplication would take `log₄(p/max)` steps and its terms would leave the range of `f64`. Its terms and the division by `p − y` are summed in double-double and rounded once (rounded step by step they cost up to 2.9 ulps), and `y − q` is the product `(y − x)(z − y)/(p − y)` with its exponent apart: from the rounded `q` it fell below the subnormal grid for a subnormal `y`, and the term `(y − q) R_J(q)`, up to 19 ulps of the value, was lost. The principal value for `p < 0` is Carlson's (2.22) in the form of Boost.Math with the products formed relative to `max(y, −p)`, `R_C(W, p′q)` by `ln(4W/(p′q))/(2√W)` where `p′q < 2^−60 W` (the logarithm of the quotient where it is a normal number, else from the mantissas and exponents of its factors, `p′q` itself perhaps below the range: two large logarithms would cancel), and by `ρ = p′q/(xy)` where `xy` is far below `max(y, −p)²` (`√(xyz)/(p′q)` and `√z/√(xy)` there with the exponents apart, as they can leave the range of `f64` where the result does not); where `(z − p) R_J` itself would leave the range (arguments spanning more than about `2^1500`), the terms are divided by `z − p` first. `R_C` is `acos`, `atan`, `atanh` or `ln` of arguments carried with their errors, and a series near `x = y`; its principal value is `√(x/(x − y)) R_C(x − y, −y)` (DLMF 19.2.20), the quotient from its mantissas and exponent apart where it is below the normal range.
- **Incomplete integrals.** `F = s R_F(c², Δ², 1)` with `s, c = sin φ, cos φ` of `φ` itself (no reduction error), `Δ² = c² + m₁ s²` for `0 ≤ m ≤ 1` (at `m = 1`, `F(φ, 1) = atanh(sin φ) = ½ ln q` with `q = (1 + s)/(1 − s)` up to `π/4` and `(1 + √(1 − c²))²/c²` beyond, of whichever of the rounded sine and cosine moves `F` less, in double-double and accurate up to the last double below `π/2`; `s R_C(1, c²)` added the errors of `R_C`, of the product and of `c²` as `1 − s²` to that of the sine, 3.04 ulps at `φ = 0.456`), and for `m > 1` past `φ = π/4` `Δ² = m c² − (m − 1)` from the cosine, whose rounding moves the edge `Δ = 0` by `ε/tan φ` where that of the sine moves it by `ε tan φ` (`F`, whose condition number is `1/Δ` there, lost `tan φ` times its `1 + κ` as `m → 1+`); `φ = jπ + r` gives `2jK ± F(r)`, and where `F(|r|)` would cancel against `2jK` the complementary form `(2j − σ)K + σ c R_F(m₁s², Δ², m₁)`; beyond `|φ| = 2^52`, where `j = round(φ/π)` is not exact, `(2K/π) φ + (F(r) − (2K/π) r)` with `2K/π` a double-double (halved and the sum doubled where the product passes `f64::MAX`, so that a result beyond the range is `±∞`; for `Π(1; φ, m)`, whose complete integral is infinite, the divergence with the sign of `φ`). `E` by DLMF 19.25.10's three positive terms where `m s² > 1/2` (the usual `s R_F − (m/3)s³R_D` cancels as `m → 1`); for `m < 0` the usual form, whose terms are positive, with the factor `(−m/3) s³` of the larger one (formed as `((−m s) s) s`, which cannot underflow where it matters) a double-double and the sum rounded once; for `m > 1` by DLMF 19.25.11, `E = s Δ/c + ((m − 1)/3) s³ R_D(Δ², 1, c²)`, two positive terms where the usual form cancels next to the edge by up to `F/E` (growing as `ln(1/(m − 1))`), with `s²`, `c²` and `Δ²` double-doubles of one angle (of `s` up to `π/4`, of `c` beyond), at which `E` differs from `E(φ)` by `Δ tan φ` times the rounding of `s`, below half an ulp. `Π` by `s R_F + (n/3)s³R_J(c², Δ², 1, c² + (1 − n)s²)` for `0 < n ≤ 1`; for `n < 0` (and `n < m`) by Boost's all-positive form of A&S 17.7.15 (R. Romeo); for `m ≤ n < 0`, where `s R_F + (n/3)s³R_J` cancels by up to `ln|m|` (and `R_J` underflowed for `|m| > 10^205`), the same where `−n sin²φ ≤ 4` and else the complete integral less the tail `∫_φ^(π/2)`, which `θ → π/2 − θ` turns into an all-positive `Π` of `n″ = −n/(1 − n)` and `m′ = −m/(1 − m)` in `(0, 1)`; past a period, with `r` of the other sign than `j` and past half of `Π(n, m)` (`n < 0`), `(2j − σ)Π(n, m) + σT`, `σ` the sign of `j` and `T` the same tail of `|r|` (`2|j|Π − Π(|r|)` cancels to `Π` for `j = ±1`, where `κ ≈ 0` for `n ≪ −1`); for `n > 1` by A&S 17.7.7, `Π(n) = −(N/3)s³R_J(…) + atanh-term/p₁`, `N = m/n`, the atanh term next to the pole from `1 − v² = |num² − den²|/max²` and `num² − den² = (n s² − 1)(n − m s²)/n` (from the smaller of `s` and `c`, as `(n − 1) − n c²` next to `π/2`); for `m > 1` by the reciprocal-modulus transformation (DLMF 19.7.4) at `N = n/m` and `μ = 1/m` with `1 − μ = (m − 1)/m`, `1 − N = (m − n)/m` and `Δ_β = cos φ` of the original variables, and for `n > m` (the pole inside the domain) A&S 17.7.7 written in them, `atanh(v)/p − (m/(3n)) s³ R_J(Δ², c², 1, P)` with `p² = (n − m)(n − 1)/n`, `v` the quotient of `p s` and `c Δ`, `P = ((n − 1) + Δ²)/n` and `n s² − 1` from the smaller of `s` and `c`: from the rounded `N` and `μ` the factors `N − 1`, `1 − μ/N`, `N s_β² − 1` and `P` lost `ε` absolutely each, all of them where `n` and `m` are next to 1 and `φ` next to the pole (903 (1 + κ) at `(1 + 1.6e−8; 1.5707, 1 + 1.0e−8)`, `κ` the condition number in `φ` only). Where an `R_J` would underflow beside a huge factor (`m ≪ −1`), it is computed as `v·2^e` and multiplied with the exponents apart.
- **Complete `Π`.** `K(m) + (n/3)R_J(0, m₁, 1, 1 − n)` with the fitted `K`; for `n < 0` and `n < m` the all-positive form of A&S 17.7.15 (with `p R_J(0, m₁, 1, p)` rescaled by `2^400` where `p = m₁/(1 − n)` is subnormal); for `m ≤ n < 0` by `θ → π/2 − θ`, `Π(n, m) = K(m)/(1 − n) + (n″/3) √ρ R_J(0, ρ, 1 − n, 1)` with `ρ = (1 − n)/(1 − m) ≤ 1`, all positive, whose `R_J` cannot underflow; for `n > 1` `−(N/3)R_J(0, m₁, 1, (n − m)/n)`, `N = m/n` (DLMF 19.6.5).
- **Without a fused multiply-add.** The products' errors and the residuals come from `f64::mul_add` where it is an instruction (aarch64, or x86_64 built with `fma`), and from Dekker's product elsewhere (the default x86_64 target and Miri), where `mul_add` is a slow call into the library; the terms whose rounding is of second order are then plain `a·b + c`. Dekker's product of factors whose product is below `2^−960` scales the smaller one by `2^600` first, so that its error is rounded once to the subnormal grid, as by `mul_add` (its partial products would round there: 405 ulps of an `R_J` principal value). The bounds hold on both: the tests pass on aarch64, on the paths of x86_64 without `fma` with the switch forced off on aarch64, and on x86_64 itself (`x86_64-unknown-linux-gnu`, glibc's `libm`, under emulation), where the sweeps of the second review's regions stayed within them too (its `sin` and `cos` differ from Apple's in the last bit at some arguments, which moves the errors by up to an ulp: `ellip_e_inc` 2.91 and `ellip_pi_inc` 2.86 there).
- **Jacobi functions.** `u` is reduced modulo `2K` with `K` a double-double (the fitted `K` for `m < 0.8` and up to four half-periods, at the transformed parameter as a double-double, whose rounding took `cn` up to 8 ulps off over those periods; the AGM in double-double otherwise); the quotient `u/(2K)` in `f64` gives `j` off by one next to an odd multiple of `K`, where `|r|` comes out above `K`, and the step back makes `|r| ≤ K`. Beyond `|u| = 2^36 K` the argument (and for `m < 0` or `m > 1` the transformed argument and parameter) and `K = π/(2 AGM(1, √μ₁))` (A&S 17.6.1) are computed again in `a::math::float::BigFloat` to the bits of `u/K` and 192 more (about 1,220 for `u` near `f64::MAX`, up to about 1,730 with a transformed parameter of that size). The reduced argument goes to `[0, K/2]` by `sn(K − v) = cd v`, `cn(K − v) = √m₁ sd v`, `dn(K − v) = √m₁ nd v`; there the descending Landen transformation (A&S 16.12) is applied to the values themselves, which are products and quotients of positive quantities, down to a parameter below 10⁻⁹ where the first-order expansion is exact to rounding. The argument of the last step, `v/∏(1 + s)` (`s = √μ′` of each step), is a double-double (the product too, divided once), and on the way up `sn` and `cn` are double-doubles, each rounded once at the end: the steps of `s ≤ 1/4` (all but the outermost for `m < 0.987`) by their increments, `sn′ − sn = s sn cn²/(1 + s x)`, `cn′ − cn = −cn (1 − dn + s x)/(1 + s x)` and `1 − dn′ = 2 s x/(1 + s x)` with `x = sn²`, which are below `s` of the values, so that their roundings are a fraction of an ulp; the others, and the outermost, as quotients by `1 + s x` in double-double, `sn (1 + s)` without rounding `1 + s`, `dn = (1 − s x)/(1 + s x)` (with the numerator `(1 − s)x + cn²` where `(2s − 1) sn > cn`, next to `m = 1`, where `1 − s x` cancels). `√m₁/dn(v)` past `K/2` is formed in double-double too (it carries the relative error of `dn(v)` into a value up to a binade lower, up to twice as many of its ulps), and `sn/√m` for `m > 1` with `√m` a double-double. Rounded at each level (and put back on `sn² + cn² = 1`), the steps gathered about an ulp of `sn` each, which the outermost `dn` takes one for one (4.3 ulps of `dn` at `(7.5e6, 0.886)`, 34.7 at `(−0.274, −552330)` next to the upper edge of `0.9 ≤ μ < 1 − 10⁻⁶`); with the argument a double-double but `fl(1 + s)` in the steps, that rounding, the same at every `u`, went into each level as well (4.19 ulps of `dn` at `(74.3, 0.890)`, 5.19 of `1/dn` at `(−4.65, −7.45)`). Within `2^−40` of 1 (`m₁ < 2^−40`: `m` next to 1 and, of the transformed parameters, `m < 1 − 2^40` and `m` just above 1), where those steps lose up to about `8K` ulps of `dn`, the first-order expansion about `m = 1` (A&S 16.15), whose remainder is below 0.005 ulp for `v ≤ K/2` (it falls as `m₁^(3/2)`), is evaluated at the reduced argument as a double-double (`e^(−v)` takes the absolute error of `v` as its relative one). `m < 0` and `m > 1` go to `(0, 1)` by A&S 16.10–11 with the scaled argument a double-double; the transformed parameter `μ` is rounded (to 1 itself for `m < −2^53`), so `K` and the comparisons with it come from its complement `μ₁`, carried as a double-double. Past `K/2` the values of `m < 0`, `√μ₁ sd`, `cd` and `nd` of the reduced argument, are `cn`, `sn` and `dn/√μ₁` at `v` themselves, not the quotients of those of `K − v`.

### Examples

```rust
use a::math::special::{
    carlson_rd, carlson_rf, chebyshev_t, ellip_e, ellip_f, ellip_k, ellip_pi, gegenbauer_c,
    jacobi_elliptic, jacobi_p, legendre_p,
};
use core::f64::consts::{FRAC_PI_2, PI};

// Complete integrals of the parameter m = k².
assert!((ellip_k(0.5f64) - 1.854_074_677_301_372).abs() < 1e-15);
assert!((ellip_e(0.5f64) - 1.350_643_881_047_675_5).abs() < 1e-15);
assert_eq!(ellip_k(1.0f64), f64::INFINITY);
// F(π/2, m) = K(m) = R_F(0, 1 − m, 1); F(φ + π, m) = F(φ, m) + 2K(m).
assert!((ellip_f(FRAC_PI_2, 0.5f64) - ellip_k(0.5)).abs() < 1e-15);
assert!((carlson_rf(0.0f64, 0.5, 1.0) - ellip_k(0.5)).abs() < 1e-15);
assert!((ellip_f(1.0 + PI, 0.5f64) - ellip_f(1.0, 0.5) - 2.0 * ellip_k(0.5)).abs() < 1e-14);
// F(φ, 1) = asinh(tan φ), finite up to the last double below π/2.
assert!((ellip_f(FRAC_PI_2, 1.0f64) - 38.025_003_373_828_87).abs() < 1e-13);
// Π(n, 0) = π/(2√(1 − n)); for n > 1 the principal value; Π(m, m) = E(m)/(1 − m), here
// without the cancellation of K + (n/3) R_J for m ≤ n < 0.
assert!((ellip_pi(0.75f64, 0.0) - PI).abs() < 1e-15);
assert!(ellip_pi(2.0f64, 0.5) < 0.0);
let m = -1e300f64;
assert!((ellip_pi(m, m) / (ellip_e(m) / (1.0 - m)) - 1.0).abs() < 1e-15);
// A result beyond f64::MAX is ∞.
assert_eq!(carlson_rd(5e-324f64, 1.0, 5e-324), f64::INFINITY);
// sn² + cn² = 1 and dn² + m sn² = 1; a huge argument is reduced with K to its bits.
let (sn, cn, dn) = jacobi_elliptic(0.7f64, 0.3);
assert!((sn * sn + cn * cn - 1.0).abs() < 1e-15);
assert!((dn * dn + 0.3 * sn * sn - 1.0).abs() < 1e-15);
let (sn, cn, _) = jacobi_elliptic(1e20f64, 0.5);
assert!((sn + 0.962_723_451_510_399_7).abs() < 1e-15 && (cn + 0.270_487_626_171_518_45).abs() < 1e-15);
// Orthogonal polynomials: P₂(x) = (3x² − 1)/2, T_n(cos θ) = cos nθ (to the rounding of cos θ).
assert!((legendre_p(2, 0.5f64) + 0.125).abs() < 1e-16);
assert!((chebyshev_t(1000, 0.3f64.cos()) - 300.0f64.cos()).abs() < 1e-12);
// Parameters at which the recurrence divides by zero or decouples: P₅₀^(−1,−1)(½), and C_n^(−3)
// vanishes identically above the degree 6.
assert!((jacobi_p(50, -1.0f64, -1.0, 0.5) + 0.037_102_744_360_039_915).abs() < 1e-17);
assert_eq!(gegenbauer_c(300, -3.0f64, 0.4), 0.0);
```

## Orthogonal polynomials

- `legendre_p(n, x)`, `hermite_h(n, x)`, `hermite_he(n, x)`, `laguerre_l(n, x)`, `chebyshev_t(n, x)`, `chebyshev_u(n, x)` of any real `x`; `assoc_legendre_p(n, m, x)` for `|x| ≤ 1` with the Condon–Shortley phase `(−1)ᵐ` (SciPy's `lpmv`; 0 for `m > n`, NaN for `|x| > 1`); `assoc_laguerre_l(n, α, x)`, `jacobi_p(n, α, β, x)` and `gegenbauer_c(n, α, x)` of any real parameters (`gegenbauer_c(n, 0, x) = 0` for `n ≥ 1`, as mpmath and the recurrence give). Infinite `x` gives the infinity of the leading term (NaN where the leading coefficient vanishes); NaN gives NaN.

### Error bounds

The error of a polynomial is measured in ulps of `max(|P(x)|, S)`, with `S` the polynomial's scale: the largest `|P|` on `[−1, 1]` (Legendre, associated Legendre, Chebyshev, Jacobi, Gegenbauer), and for Hermite and Laguerre the weighted largest value `e^(x²/2) max|Hₙ(t) e^(−t²/2)|` (`e^(x²/4)` for `Heₙ`) and `e^(x/2) max|Lₙ^(α)(t) e^(−t/2)|` inside the oscillatory region (`|x| ≤ √(2n + 1)`, `√(4n + 2)`, `0 ≤ x ≤ 4n + 2α + 2`); outside these regions, and for `|x| > 1`, the error is relative. So it is relative error away from the zeros and an absolute error on the polynomial's own scale near them.

| Polynomial | Degrees in the data | Bound | Measured |
| --- | --- | ---: | ---: |
| `legendre_p` | 0–1000, `|x| ≤ 3` | 0.55 | 0.47 |
| `assoc_legendre_p` | 0–120, all `m` | 0.55 | 0.49 |
| `hermite_h`, `hermite_he` | 0–300 | 0.55 | 0.50, 0.47 |
| `laguerre_l`, `assoc_laguerre_l` | 0–400, `α ∈ (−10, 30)` | 0.55 | 0.44, 0.50 |
| `chebyshev_t`, `chebyshev_u` | 0–5000, `|x| ≤ 1.5` | 0.55 | 0.45, 0.50 |
| `jacobi_p` | 0–300, `α, β ∈ (−5, 10)`, also `α + β`, `α` or `β` a negative integer, and `α + β` next to one | 0.55 | 0.50 |
| `gegenbauer_c` | 0–400, `α ∈ (−5, 20)`; the bound for `α > −1/2` and for `2α` a negative integer (see the limits) | 0.55 | 0.50 |
| every polynomial, `|x| > 2^400` | relative: the leading term (`jacobi_p` with `α + β` next to `−l`, `n < l ≤ 2n`: the two largest terms of the explicit sum) | 0.55 | 0.50 |
| every polynomial, `f32` | as above | 0.501 | 0.50 |

The bounds come from mixed error ulps of `max(|P|, S)`: relative error away from the zeros, and an absolute error relative to the polynomial's scale `S` near them.

Limits. Outside the interval the error is relative, but next to the zeros a polynomial has there (`assoc_laguerre_l` with `α < −1` below 0, `jacobi_p` with `α` or `β < −1` beyond ±1, `gegenbauer_c` with `α < −1/2` beyond ±1), where the recurrence's error is an absolute one on the size of its terms: it was `4.7e7` ulps (relative) at `L₁₆^(−9.5)(−0.9159)` and `2.1e22` at `C₄₀₀^(−4.9)(1.0000453)`. For `gegenbauer_c` with `α < −1/2` the values at and next to `±1` are much smaller than `S` for large degrees and the recurrence approaches them from its other solution: 54 ulps of `S` at `C₄₀₀^(−4.9)(1)`. Next to the degenerate parameters, but not on them (`2α`, or `α` of Laguerre, within `10⁻⁶` of a negative integer), the whole family is of the size of the distance `δ` and the scaled error grows as `1/δ` (640 ulps of `S` at `α = −3 ± 10⁻⁹`, `n = 400`, `x = 1`). Next to a negative integer, `α + β = −l + δ` (`l ≥ 2`, `|δ| ≤ ½`), two coefficients of the Jacobi recurrence are about `δ` times their size, and the recurrence through them loses whatever `δ` (1.5e13 ulps of `S` at `δ = 6e−12`, 10 at `δ = 0.32` for `l = 8`): for `5 ≤ l ≤ 30` the explicit sum goes to the degree `l` at every `δ`, and for `l = 2…4` where `|δ| < 1/64`, and from the degree 16 on where `|δ| < 1/16` (the recurrence was within 0.50 ulps of `S` below the degree 16 for `|δ| ≥ 1/64`, on 2,000 cases in the range of the table, and from it for `1/16 ≤ |δ| ≤ 1/4` at and next to `±1`, on 2,400; between, it reached 1.35 ulps of `S` just past `|δ| = 1/64`, at `x = −1` and `n = 232`); for `α + β` below `−30.5`, outside that range, only within `2^−40 l` of `−l` (where the sum was within 0.36 ulps of `S` up to `l = 40` and 15 at `l = 50`), and elsewhere the recurrence keeps its loss: 10⁶ ulps of `S` at `l = 31`, `δ = 0.4`, 10¹¹ at `δ = 10⁻⁶`, up to 10³¹ at `l = 50`. Beyond the degrees of the data, the bounds have not been measured.

### Method

Each polynomial comes from its three-term recurrence (DLMF 18.9.1–2 and the table there) run compensated: the value follows the plain recurrence and a second value carries its error, from the error-free transformations of every step's products and sum, so that the result is rounded about once; the coefficients are exact (Legendre, Hermite, Laguerre, Chebyshev, whose coefficients are integers and `x`), exact for parameters `α`, `β` on the grid `2^−20` (with magnitudes up to `2^28`: their sums with the degree are exact and their products exact double-doubles), or double-doubles. Where the recurrence divides by an integer `c_k` (Legendre, Laguerre, Gegenbauer, associated Legendre and Laguerre) it runs without the divisions, on `Q_k = p_k Π c_j` with the product kept beside it and one division at the end. The values are rescaled by `2^±500` as they grow or shrink, on the size of the value (where the plain recurrence meets an exact zero the value lives in the second word, which is moved into the first), so that only a result beyond the range of `f64` overflows; for `|x| > 2^400`, where a step could overflow, the polynomial is its leading term `cₙxⁿ` to `2^−336`, as a double-double with its exponent apart. `Pₘᵐ = (−1)ᵐ(2m − 1)!!(1 − x²)^(m/2)` is a product of `m` double-double factors with `1 − x²` exact. The Chebyshev polynomials above degree 47 are the binary powers of `T₁ + i sin θ U₀` in double-double, `O(log n)` steps whose errors stay near the double-double precision; for `|x| ≤ 1`, where the values stay within `n + 1` and two consecutive ones are never both small, the recurrence runs without the rescaling checks.

Where a recurrence coefficient vanishes the recurrence decouples, or divides by zero, and the values next to `±1` (or 0) are small differences of large terms; those parameters go by identities that keep the relative accuracy, as double-doubles rounded once: `Lₙ^(−l)(x) = (−x)^l (n − l)!/n! L_(n−l)^(l)(x)` (Szegő (5.2.1)); `Cₙ^(−l) ≡ 0` above the degree `2l` and `Cₙ^(½−l)(x) = 4^l ((½)_l)² (n − 2l)!/n! (1 − x²)^l C_(n−2l)^(l+½)(x)` for `n ≥ 2l` (the reflection `λ → 1 − λ` of the differential equation); `Pₙ^(−l,β)(x) = [C(n + β, l)/C(n, l)] ((x − 1)/2)^l P_(n−l)^(l,β)(x)` (Szegő (4.22.2)) and `Pₙ^(α,β)(x) = (−1)ⁿ Pₙ^(β,α)(−x)`; and for `α + β` within ½ of `−l` with `5 ≤ l ≤ 30`, within 1/64 of it with `l = 2…4` and from the degree 16 on within 1/16 (and beyond `l = 30` within `2^−40 l`), the explicit sum `Σ d_j ((x − 1)/2)ʲ` of DLMF 18.5.7 in double-double up to the degree `l` and the recurrence from `(P_(l−1), P_l)`. For `|x| > 2^400` with `n < l ≤ 2n`, whose leading coefficient has the factor `δ = α + β + l` (zero, or small), the polynomial is the two largest terms `d_t yᵗ + d_n yⁿ` of that sum, `t = l − n − 1`, `y = (x − 1)/2`, with their exponents apart.

The recurrences take `O(n)` steps for the degree `n`: the asymptotic forms for large degrees (Legendre's by Stieltjes's and Bessel-type expansions, Hermite's by Plancherel–Rotach) are left to the release optimisation pass.

### Accuracy measured

The bounds above were measured against mpmath at 256 bits and more (the reference data, `tests/math/special/data/elliptic_*.txt`, made by `data/gen_elliptic.py`, whose last cases in each file are the arguments of the audit's defects and their neighbours), and on sweeps of the same functions with fresh points (two seeds each): for every function and region, uniform and logarithmic points, the ends of the intervals, the last doubles before the poles and edges, huge and tiny arguments; the references there were cross-checked by a second method (quadrature, theta functions, the AGM, closed forms, exact rational sums of the polynomials). To reproduce the data: `python3 gen_elliptic.py` in `tests/math/special/data`, then `cargo test --test math -- special::elliptic` (which prints the largest error of each function and region).

## Measured: elliptic functions and orthogonal polynomials

As for [the gamma function](#measured) (the conditions and the same-work rule are there). The elliptic integrals and functions:

| function | arguments | `a` | within the bound | faster, beyond the bound |
| --- | --- | ---: | --- | --- |
| `ellip_k` | m in [0,0.8) | 3.63 ns | 0.515 SciPy |  |
| `ellip_e` | m in [0,0.8) | 2.21 ns | 0.345 SciPy |  |
| `ellip_k` | m in [0.8,1) | 10.3 ns | 1.89 SciPy |  |
| `ellip_e` | m in [0.8,1) | 9.20 ns | 1.79 SciPy |  |
| `ellip_k` | m in [-10,0) | 14.1 ns | – | 2.13 SciPy† (1.9 ulps) |
| `ellip_e` | m in [-10,0) | 13.0 ns | – | 2.11 SciPy† (1.7 ulps) |
| `ellip_pi` | n in (-5,0.9), m in [0,0.99) | 77.1 ns | – | 0.938 Boost† (16.2 ulps) |
| `ellip_f` | phi in [-pi/2,pi/2], m in [0,0.99) | 34.9 ns | 0.496 SciPy | 0.895 Boost† (5.7 ulps) |
| `ellip_e_inc` | phi in [-pi/2,pi/2], m in [0,0.99) | 69.2 ns | 0.744 SciPy | 1.03 Boost† (53.8 ulps) |
| `ellip_f` | phi in [-20,20], m in [0,0.99) | 39.0 ns | 0.459 SciPy | 0.716 Boost† (7.2 ulps) |
| `ellip_e_inc` | phi in [-20,20], m in [0,0.99) | 72.0 ns | 0.745 SciPy | 0.879 Boost† (31.4 ulps) |
| `ellip_pi_inc` | n in (-5,0.9), \|phi\| < 1.5, m in [0,0.99) | 87.0 ns | 0.800 GSL |  |
| `carlson_rf` | args in [0.01,10] | 22.5 ns | – | 1.24 Boost† (2.3 ulps) |
| `carlson_rd` | args in [0.01,10] | 39.7 ns | 0.989 GSL | 1.29 Boost† (14.6 ulps) |
| `carlson_rj` | args in [0.01,10] | 59.7 ns | 0.868 Boost |  |
| `carlson_rc` | args in [0.01,10] | 10.8 ns | 0.966 Boost |  |
| `jacobi_elliptic` | u in [-4,4], m in [0,0.99) | 73.3 ns | 2.33 GSL |  |
| `jacobi_elliptic` | u in [-100,100], m in [0,0.99) | 129 ns | – | 4.29 GSL† (580 ulps) |
| `jacobi_elliptic` | u in [-4,4], m in [0.99,1) | 140 ns | 4.24 GSL |  |
| `ellip_k`, `f32` | m in [0,0.8) | 4.84 ns | 0.481 Boost |  |

The orthogonal polynomials, at the degrees 4, 20, 100 and 1000:

| function | arguments | `a` | within the bound | faster, beyond the bound |
| --- | --- | ---: | --- | --- |
| `legendre_p` | n=4, x in [-1,1] | 8.87 ns | – | 4.77 Boost† (4.2 ulps) |
| `chebyshev_t` | n=4, x in [-1,1] | 2.78 ns | – | 2.89 Boost† (1.4 ulps) |
| `chebyshev_u` | n=4, x in [-1,1] | 2.87 ns | – | 2.49 Boost† (2.6 ulps) |
| `assoc_legendre_p`, m = n/2 | n=4, x in [-1,1] | 15.2 ns | – | 1.75 Boost† (2 ulps) |
| `jacobi_p`, α = 0.5, β = 1.5 | n=4, x in [-1,1] | 26.1 ns | – | 5.01 Boost† (11.5 ulps) |
| `gegenbauer_c`, α = 1.5 | n=4, x in [-1,1] | 10.2 ns | – | 3.62 Boost† (9.8 ulps) |
| `hermite_h` | n=4, \|x\| < sqrt(2n+1) | 4.55 ns | – | 4.65 Boost† (3.6 ulps) |
| `hermite_he` | n=4, \|x\| < sqrt(2n+1) | 4.54 ns | – | 3.16 SciPy† (2.4 ulps) |
| `laguerre_l` | n=4, x in [0,4n+2] | 10.0 ns | – | 5.25 Boost† (2.7 ulps) |
| `assoc_laguerre_l`, α = 2 | n=4, x in [0,4n+2] | 11.6 ns | – | 6.21 Boost† (5.8 ulps) |
| `legendre_p` | n=20, x in [-1,1] | 46.9 ns | – | 3.21 Boost† (87.4 ulps) |
| `chebyshev_t` | n=20, x in [-1,1] | 17.0 ns | – | 3.45 Boost† (26.9 ulps) |
| `chebyshev_u` | n=20, x in [-1,1] | 16.8 ns | – | 3.03 Boost† (60.6 ulps) |
| `assoc_legendre_p`, m = n/2 | n=20, x in [-1,1] | 85.9 ns | – | 5.25 Boost† (9.3 ulps) |
| `jacobi_p`, α = 0.5, β = 1.5 | n=20, x in [-1,1] | 124 ns | – | 4.24 Boost† (910 ulps) |
| `gegenbauer_c`, α = 1.5 | n=20, x in [-1,1] | 51.1 ns | – | 3.08 Boost† (560 ulps) |
| `hermite_h` | n=20, \|x\| < sqrt(2n+1) | 27.8 ns | – | 5.14 Boost† (800 ulps) |
| `hermite_he` | n=20, \|x\| < sqrt(2n+1) | 27.6 ns | – | 3.07 SciPy† (650 ulps) |
| `laguerre_l` | n=20, x in [0,4n+2] | 50.9 ns | – | 3.50 Boost† (1·10³ ulps) |
| `assoc_laguerre_l`, α = 2 | n=20, x in [0,4n+2] | 54.9 ns | – | 3.62 Boost† (450 ulps) |
| `legendre_p` | n=100, x in [-1,1] | 242 ns | – | 1.90 SciPy† (1.2·10³ ulps) |
| `chebyshev_t` | n=100, x in [-1,1] | 77.2 ns | – | 1.87 Boost† (66.2 ulps) |
| `chebyshev_u` | n=100, x in [-1,1] | 81.6 ns | – | 1.96 Boost† (520 ulps) |
| `assoc_legendre_p`, m = n/2 | n=100, x in [-1,1] | 400 ns | – | 4.35 Boost† (46.8 ulps) |
| `jacobi_p`, α = 0.5, β = 1.5 | n=100, x in [-1,1] | 621 ns | – | 2.99 Boost† (8.8·10³ ulps) |
| `gegenbauer_c`, α = 1.5 | n=100, x in [-1,1] | 258 ns | – | 2.85 Boost† (2.8·10³ ulps) |
| `hermite_h` | n=100, \|x\| < sqrt(2n+1) | 148 ns | – | 3.38 Boost† (3·10⁴ ulps) |
| `hermite_he` | n=100, \|x\| < sqrt(2n+1) | 147 ns | – | 1.72 GSL† (7.5·10⁴ ulps) |
| `laguerre_l` | n=100, x in [0,4n+2] | 263 ns | – | 2.27 SciPy† (1·10⁵ ulps) |
| `assoc_laguerre_l`, α = 2 | n=100, x in [0,4n+2] | 277 ns | – | 2.27 SciPy† (9·10³ ulps) |
| `legendre_p` | n=1000, x in [-1,1] | 2.51 µs | – | 1.64 SciPy† (3.2·10³ ulps) |
| `chebyshev_t` | n=1000, x in [-1,1] | 142 ns | – | 0.205 Boost† (270 ulps) |
| `chebyshev_u` | n=1000, x in [-1,1] | 145 ns | – | 0.209 Boost† (1.4·10³ ulps) |

No reference holds the bound of the orthogonal polynomials, 0.55 ulps of `max(|P|, S)`, on these arguments: at their largest the references are 1.4 to 7·10⁵ ulps of `max(|v|, median |v|)` from `a` (the last column gives the fastest one's); the compensated recurrence that holds the bound costs two to three times a plain one.

### Where the references are faster

- **`jacobi_elliptic`** for `u` in `[−4, 4]` with `m` in `[0, 0.99)` and in `[0.99, 1)` (2.33 and 4.24 against GSL, within 22.2 and 21.2 ulps of `a`): the double-double Landen and AGM reduction, which holds the bounds of 3.5 to 80 ulps as `m → 1`.
- **`ellip_k` and `ellip_e`** for `m` in `[0.8, 1)` (1.89 and 1.79 against SciPy's Cephes, within 1.1 and 1.0 ulps of `a`): the logarithmic forms near `m = 1` in double-double, for 0.6 ulp.

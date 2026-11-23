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
- **`ψ⁽ⁿ⁾`.** `ψ′` by its own method: `1/x + (its rounding error + 1/(2x²) + T(u)/x³)`, rounded once, from 10 on, ten minimax pieces `Y + c₁s + R(s)` on `[2, 10)` with `c₁s` without error, and `Σ 1/(x + j)² + ψ′(x + m)` below 2. Otherwise `(−1)ⁿ⁺¹ n! ζ(n + 1, x)`, the Hurwitz zeta function by its terms up to where its Euler–Maclaurin tail converges to 2⁻⁶⁰ (from `y ≥ 0.38(n + 21) + 0.5`, with up to 40 Bernoulli terms, the first three in double-double), or until the rest is negligible; for `x < 0` the sum of all the terms `(x + j)^(−n−1)` as `(−1)^(n+1)[ζ(n + 1, r) − ζ(n + 1, 1 − x)] + ζ(n + 1, p)` of the exact distances `r` and `p` of `x` to the integers beside it. Large orders and arguments carry their values as `v·e^l`, so nothing overflows before the last product.
- **`B` and `ln B`.** For `a, b > 0`: below 8, `B = R(a + b)/(R(a)R(b))` with `R = 1/Γ` in double-double (the low part of `a + b` through `ψ`) and `ln B` its logarithm (above 10⁻¹⁰⁰; `ln Γ(a) + ln Γ(b) − ln Γ(a + b)` in double-double below that); `ln Γ(a) + [ln Γ(b) − ln Γ(a + b)]` by the Stirling difference when only the larger is, and for both from 8 on `ln √(2π) + (a − ½) ln(a/c) + (b − ½) ln(b/c) − ½ ln c + S(a) + S(b) − S(c)` with `ln(b/c) = −ln(1 + a/b)` computed directly (an `atanh` series for small ratios), so no term cancels another (at half scale where `a + b` overflows); `B` is its double-double exponential. The Stirling difference `ln Γ(a) − ln Γ(a + b) = −(a − ½)L₁ − b(ln a + L₁) + b + S(a) − S(a + b)` takes `L₁ = ln(1 + b/a)` from `b/a` itself (by the `atanh` series) for `a ≥ 2⁴⁰`, so a ratio beyond `2⁵³` keeps its digits (below `2⁴⁰` from the double-double `1 + b/a`, whose absolute error `2⁻¹⁰⁶` times `a` stays below `2⁻⁶⁶`). Other arguments go through the reflection formula as identities between beta functions of positive arguments and sines of the exact arguments: `B(a, b) = π sin(πc)/(sin(πa) sin(πb)) / ((1 − c) B(1 − a, 1 − b))` for `a, b < 0`, `(sin(πc)/sin(πa)) B(b, 1 − c)` for `a < 0 < b` with `c = a + b < 0`, `π/(sin(πa) · b · B(c, 1 − a))` for `c > 0`; `c` is the exact sum as a double-double, so a sum beside a pole keeps its distance to it, and the gamma functions of large arguments appear only in Stirling differences of the double-doubles, never as differences of their logarithms.
- **`binomial`, `pochhammer`.** The table of `n!` in double-double for integers to 170; compensated products for an integer `k` or `n` up to 32 (every factor `n − i` or `x + i` exact as a double-double, its low part and each product's rounding error carried); else `1/((n + 1) B(n − k + 1, k + 1))` with every argument an exact double-double (the beta function as above), and `(x)ₙ` by the Stirling difference, or for a negative `x` or `x + n` by `(x)ₙ = (sin(πx)/sin(πc)) Γ(1 − x)/Γ(1 − c)` and its companions (`c = x + n`).
- **`P` and `Q`.** The structure of DiDonato and Morris (1986) and Boost: the smaller of `P` and `Q`, and the other as its double-double complement. The prefix `x^a e^(−x)/Γ(a + 1)` in double-double, `e^(a ln(x/a) + a − x − S(a))/√(2πa)` from `a = 8`. The series of `P` below about `x = a`, Legendre's continued fraction for `Q` above (evaluated from the bottom as one composed map per four steps, so one division per four steps, its length set in advance, the last steps in double-double), finite sums for integer and half-integer `a` below 20 when `x ≥ a` (`(n − 1)!` from the table, `erfcx` for the half-integers, `a/x` in double-double), the small-`a` form `Q = (Γ(1 + a) − 1 − (x^a − 1))/…` without cancellation for `x < 1.1` (and `x < 2` for `a < ½`; further out `Q` is a small difference of its terms, and the fraction takes over), `x^a − 1` from the double-double exponential, and for `a ≥ 20` with `|x − a|/a` below about ½ Temme's uniform expansion with 13 coefficients `c_k(z)` derived by series arithmetic after Temme (1979) and fitted minimax, and the area's double-double `erfcx` (eight pieces below 4, then `(1/t)·g(1/t²)`; `beta_inc`'s BASYM uses the same).
- **Their inverses.** A first guess (`(pΓ(a + 1))^(1/a)` for small `x`; for `a ≥ 1` Temme's expansion inverted to first order, as DiDonato and Morris, with Acklam's normal quantile; an iteration on the tail for large `x`), then Halley's method on the logarithm of the smaller tail in `ln x` with the slope from `ln Γ(a)` computed once, kept in a bracket: about two evaluations of `P` or `Q` a call. The iteration goes on into the subnormals and stops when a step no longer moves `x` (the nearest float to Newton's root).
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

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation only, never timed. The times are those of the code before the November 2026 fixes; the fixes changed them where noted below the tables.

| function | region | a | libm | statrs | special | puruspe | gsl | boost | cephes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gamma | [0.5,1.5] | 4.36 | 33.36 | 8.39 | 33.17 | 3.61 | 27.48 | 16.88 | 11.31 |
| gamma | [1.5,8] | 5.71 | 30.99 | 8.39 | 31.05 | 4.23 | 22.46 | 13.53 | 10.87 |
| gamma | [8,20] | 10.67 | 30.15 | 8.4 | 30.02 | 6.39 | 16.55 | 13.44 | 16.84 |
| gamma | [20,170] | 10.72 | 30.15 | 8.38 | 29.9 | 70.04 | 12.52 | 17.5 | 12.51 |
| gamma | [-8,-0.1] | 6.98 | 39.94 | 12.06 | 39.75 | - | 34.73 | 15.68 | 12.45 |
| gamma | [-170,-8] | 24.01 | 36.85 | 12.14 | 36.35 | - | 23.24 | 25.48 | 17.29 |
| lgamma | [0.01,0.5] | 8.16 | 7.57 | 12.49 | 7.15 | 6.32 | 19 | 5.79 | 8.19 |
| lgamma | [0.5,1.5] | 4.12 | 5.96 | 7.4 | 5.68 | 6.31 | 9.05 | 6.54 | 13.26 |
| lgamma | [1.5,3] | 4.44 | 4.5 | 6.53 | 4.27 | 6.32 | 8.96 | 6.17 | 10.49 |
| lgamma | [3,8] | 3.31 | 9.11 | 6.45 | 9.06 | 6.33 | 9.08 | 11.61 | 13.59 |
| lgamma | [8,100] | 5.67 | 5.74 | 6.52 | 5.72 | 8.58 | 14.62 | 18.93 | 7.07 |
| lgamma | [100,1000000] | 5.59 | 5.83 | 6.49 | 5.74 | 6.6 | 9.01 | 9.41 | 5.54 |
| lgamma | [-10,-0.6] | 22.15 | 18.2 | - | 18.15 | - | 30.04 | 16.01 | 14.81 |
| digamma | [0.01,1] | 5.13 | - | 7.12 | 9.71 | - | 20.29 | 3.59 | 7.77 |
| digamma | [1,8] | 3.83 | - | 6.48 | 7.14 | - | 16.44 | 4.74 | 7.31 |
| digamma | [8,1000] | 4.4 | - | 4.18 | 3.64 | - | 15.08 | 4.09 | 4.65 |
| digamma | [-10,-0.1] | 14.22 | - | 9.25 | 21.47 | - | 22.91 | 7.18 | 9.26 |
| trigamma | [0.1,8] | 6.88 | - | - | 4.79 | - | 84 | 3.63 | - |
| trigamma | [8,1000] | 3.43 | - | - | 2.29 | - | 82.7 | 3.04 | - |
| polygamma n=3 | [0.1,20] | 63.98 | - | - | - | - | 95.92 | 109.61 | - |
| polygamma n=30 | [0.5,20] | 238.4 | - | - | - | - | 96.21 | 644.63 | - |
| polygamma n=3 | [-5,-0.1] | 351.08 | - | - | - | - | 3.29 | 162.18 | - |
| beta | small | 29.87 | - | 28.16 | 25.69 | 24.11 | 72.84 | 27.58 | 34.39 |
| lbeta | small | 40.57 | - | 21.59 | 23.05 | - | 48.4 | - | 37.25 |
| beta | large | 77.91 | - | 24.55 | 22.44 | 25.42 | 56.6 | 29.38 | 26.9 |
| lbeta | large | 68.51 | - | 19.64 | 18.68 | - | 45.27 | - | 24.62 |
| beta | mixed | 60.63 | - | 28.77 | 23.8 | 25.59 | 67.6 | 29.85 | 33.9 |
| lbeta | mixed | 50.29 | - | 24.96 | 20.28 | - | 56.71 | - | 29.68 |
| factorial | integers [0,170] | 1.03 | - | 1 | - | 39.17 | 2.76 | 0.78 | - |
| factorial | reals [-0.9,170] | 17.27 | 30.64 | - | - | - | 14.67 | 17.7 | - |
| binomial | integers n<=60 | 4.91 | - | 9.74 | - | - | 3.4 | 2.82 | - |
| binomial | integers n in [1e3,1e6] | 82.42 | - | 21.66 | - | - | 33.93 | 26.98 | - |
| pochhammer | integer n<=30 | 11.88 | - | - | - | - | 63.04 | 11.47 | - |
| pochhammer | real n | 44.98 | - | - | - | - | 63.66 | 23.09 | - |
| lgamma complex | \|z\|<40 | 132.56 | - | - | - | - | 40.82 | - | - |
| gamma complex | \|z\|<40 | 177.03 | - | - | - | - | - | - | - |

| function | region | a | statrs | special | puruspe | gsl | boost | cephes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gamma_p | a<1,x<2 | 92.8 | 119.3 | 61.1 | 94.9 | 47.9 | 167.8 | 79.5 |
| gamma_q | a<1,x<2 | 88.6 | 118.3 | - | 98.2 | 782.4 | 163.5 | 78 |
| gamma_p | 1<a<20,x~a | 87.7 | 48.1 | 33.3 | 47 | 82.9 | 73.1 | 35.4 |
| gamma_q | 1<a<20,x~a | 86.8 | 49.2 | - | 47.4 | 210.6 | 74.7 | 33.3 |
| gamma_p | 20<a<200,\|x-a\|<3sqrt(a) | 48.8 | 105.1 | 76.1 | 130 | 268.1 | 45.4 | 82.6 |
| gamma_q | 20<a<200,\|x-a\|<3sqrt(a) | 48.9 | 104.1 | - | 130.9 | 269.7 | 46.2 | 79.4 |
| gamma_p | 200<a<1e4,x~a | 47.8 | 263.4 | 203.1 | 145.4 | 633.6 | 213.4 | 219.8 |
| gamma_q | 200<a<1e4,x~a | 48 | 263.5 | - | 145.4 | 632 | 213.5 | 217.8 |
| gamma_p | 1e4<a<1e6,x~a | 48 | 1396.1 | 1236 | 145.9 | 2091.1 | 1434.8 | 1296.2 |
| gamma_q | 1e4<a<1e6,x~a | 47.9 | 1344.3 | - | 145.5 | 2017 | 1429.9 | 1278.2 |
| gamma_p | a in 1..20 integer,x~a | 41.6 | 35.5 | 27.6 | 41.5 | 83.7 | 26.9 | 28.3 |
| gamma_q | a in 1..20 integer,x~a | 41.9 | 36.2 | - | 41.4 | 100.9 | 27.9 | 26.1 |
| gamma_p | a in 0.5..19.5 half-int,x~a | 54.9 | 48.2 | 33 | 46.1 | 80.2 | 36.9 | 36.1 |
| gamma_q | a in 0.5..19.5 half-int,x~a | 53.9 | 49.7 | - | 47.5 | 276.6 | 37.3 | 34 |
| gamma_p | dev prefix a<8 | 36.9 | 32.2 | 25.1 | 27.8 | 34.8 | 38.8 | 24.3 |
| gamma_q | dev prefix a<8 | 36.8 | 34 | - | 28.5 | 36.4 | 47.3 | 24.3 |
| gamma_p | dev prefix a>8 | 40.1 | 23.8 | 24.1 | 27.5 | 30.4 | 33 | 25.1 |
| gamma_q | dev prefix a>8 | 40.2 | 25.7 | - | 27.9 | 30.8 | 32.4 | 25.1 |
| gamma_p | dev series a<8 x~a | 55.1 | 47.4 | 40.8 | 45.1 | 61.4 | 63 | 39.7 |
| gamma_q | dev series a<8 x~a | 55.1 | 52.2 | - | 47.3 | 303.3 | 70.2 | 38.2 |
| gamma_p | dev series a>8 x~a | 83.1 | 68.7 | 55.6 | 65.6 | 90.4 | 78.8 | 64.2 |
| gamma_q | dev series a>8 x~a | 82.3 | 70.1 | - | 67.7 | 188.5 | 84.5 | 60.9 |
| gamma_p | dev frac a<20 x=1.2a | 108.9 | 66.7 | 38.4 | 77.3 | 105.1 | 92 | 38.9 |
| gamma_q | dev frac a<20 x=1.2a | 108.5 | 59.3 | - | 72.3 | 187 | 90.5 | 36.1 |
| gamma_p | dev frac a<20 x=2a | 98.7 | 58 | 32.5 | 64.6 | 104 | 75.7 | 35.9 |
| gamma_q | dev frac a<20 x=2a | 99.4 | 54.3 | - | 63.3 | 142.4 | 74.5 | 32.6 |
| gamma_p | 1<a<100,x=a(0.1..0.5\|2..5) | 71.9 | 32.2 | 23.1 | 27.8 | 70.8 | 50 | 33 |
| gamma_q | 1<a<100,x=a(0.1..0.5\|2..5) | 71.7 | 32.4 | - | 27.9 | 73.5 | 39.5 | 28.5 |
| P at the root | a<1 | 58.3 | - | - | - | - | 72.6 | - |
| lgamma(a+1) | a<1 | 10.7 | - | - | - | - | - | - |
| gamma_p_inv | a<1 | 383.3 | - | - | 297.2 | 1922.3 | 241.1 | - |
| gamma_q_inv | a<1 | 372.8 | - | - | - | 2312.1 | 248.9 | 676.1 |
| P at the root | 1<a<20 | 85.3 | - | - | - | - | 66.2 | - |
| lgamma(a+1) | 1<a<20 | 11.2 | - | - | - | - | - | - |
| gamma_p_inv | 1<a<20 | 327 | - | - | 483 | 761.7 | 197.2 | - |
| gamma_q_inv | 1<a<20 | 334.5 | - | - | - | 802.5 | 198.3 | 388.4 |
| P at the root | 20<a<1e4 | 47.9 | - | - | - | - | 183 | - |
| lgamma(a+1) | 20<a<1e4 | 11.8 | - | - | - | - | - | - |
| gamma_p_inv | 20<a<1e4 | 184.7 | - | - | - | 2488.3 | 240.5 | - |
| gamma_q_inv | 20<a<1e4 | 185.4 | - | - | - | 2323.1 | 239.9 | 1997.1 |
| P at the root | 1<a<20,tiny p | 30.9 | - | - | - | - | 27 | - |
| lgamma(a+1) | 1<a<20,tiny p | 11.2 | - | - | - | - | - | - |
| gamma_p_inv | 1<a<20,tiny p | 196.6 | - | - | - | 515.5 | 173.3 | - |
| gamma_q_inv | 1<a<20,tiny p | 391.8 | - | - | - | 2122.8 | 121.3 | 2385.1 |
| beta_inc | a,b in [0.1,1], x uniform | 136.9 | 100 | 119.3 | 137.3 | 145.6 | 85.7 | 136.6 |
| beta_inc | a,b in [1,10], x uniform | 158.3 | 93.6 | 88.1 | 74.6 | 134.1 | 137.8 | 73.4 |
| beta_inc | a,b in [10,100], x mean ±3σ | 219 | 156.1 | 120.6 | 151.9 | 192.9 | 169.8 | 162.9 |
| beta_inc | a,b in [100,1e4], x mean ±3σ | 369.2 | 379 | 675.8 | 500.6 | 434 | 223.8 | 320.9 |
| beta_inc | a,b in [1e5,1e7], x mean ±3σ | 208.2 | 939.8 | 19484.7 | 127.5 | 1954.4 | 874.3 | 1320.8 |
| beta_inc | a in [20,1e3], b in [0.1,1], x mean ±3σ | 225.8 | 107.6 | 11438 | 25541.1 | 179.1 | 163.1 | 266.8 |
| beta_inc_inv | a,b in [0.1,1], x uniform | 626.5 | 708 | 735.6 | 1385.1 | - | 510 | 1765 |
| beta_inc_inv | a,b in [1,10], x uniform | 633.7 | 512.8 | 476.3 | 537.4 | - | 715.2 | 852.3 |
| beta_inc_inv | a,b in [10,100], x mean ±3σ | 790.9 | 495 | 380.4 | 1297.1 | - | 506.2 | 684.3 |
| beta_inc_inv | a,b in [100,1e4], x mean ±3σ | 1066 | 877.2 | 1441.4 | 5241.9 | - | 492.8 | 1456.8 |
| beta_inc_inv | a,b in [1e5,1e7], x mean ±3σ | 776.4 | 2175.2 | 35927.3 | 16964.6 | - | 3266.5 | 17636.2 |
| beta_inc_inv | a in [20,1e3], b in [0.1,1], x mean ±3σ | 1031.2 | 1253.4 | 40504.7 | 21094.7 | - | 585.8 | 4212.7 |

Since then (measured with the code before and after linked into one binary and their rounds interleaved, the minimum of 31–41 rounds of 4,000–20,000 calls, on the same machine under load, ratios only; the first fixes measured again by their second review, with fat LTO and without): `binomial` of real arguments 24–27 % slower on `[0, 60]²` (the cases with `n − k + 1 < 0`) and 33 % for `n` in `[−30, 0]`, `beta` and `lbeta` with a negative argument 21–30 % (the reflection identities: three sines and a beta function of positive arguments, where the logarithms of the gamma functions were subtracted), `pochhammer` with `x` in `[−20, 0]` 12–20 %; `lgamma` 33–37 % on `[3, 8)` and 12–31 % on `[0.5, 1.5]` (the pieces' linear and quadratic terms without error), 3–8 % for `x < 0` (recomputed beside its zeros); `P` and `Q` for `a < 1`, `x < 2` 7–10 %, and `gamma_q` for `a < 1`, `1.1 ≤ x < 3` 7–14 % (the fraction for `x ≥ 2`, or `a ≥ ½`); the complex `ln Γ` and `Γ` 11–14 % for `|z| < 40` and 15–22 % for `|z|` in the thousands (the argument and the logarithm to 2⁻¹⁰⁰, the phase reduced exactly); `I` beside the mean of parameters from 10⁵ on 2–7 % (BASYM's root in double-double); faster: `gamma` on `[−8, −0.1]` by up to 20 %, `digamma` on `[1, 8]` and `ψ′` on `[2, 10]` by 8 %, `binomial` of large integers by 9 %, the beta inverses with both parameters from `2⁸⁶` on by a factor of about 1,000 (the root in closed form, where the iteration ran long and was often wrong); everything else, `I` and the inverses included, within ±5 %. (Tiny functions move by up to 20 % with their place in the binary alone: `digamma` on `[1, 8]` measured 0.92 against an identical copy of itself, so its 8 % is within that.)

The fastest reference is often far less accurate (the next table); Boost's column is there for comparison.

#### Accuracy of the references

Max/rms ulps on the area's data files before the regression cases of the November 2026 audit were added (the normal range; the references' own failures in brackets):

| Function | `a` | Boost | GSL | Cephes | libm | statrs | special | puruspe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gamma` (648) | 0.50 / 0.28 | 6.6 / 1.5 | 83 / 5.1 (9 wrong) | 3.1 / 1.0 | 5.8 / 1.3 | 4.6e15 (18 wrong) | 5.8 / 1.3 | 7.8 / 1.8 (`x > 0`) |
| `lgamma` (618) | 0.54 / 0.27 | 1.7 / 0.50 | 1.8e3 / 87 | 1.3e16 | 1.8 / 0.43 | 5.5e16 (`x > 0`) | 1.8 / 0.43 | 4.2e20 (72 wrong) |
| `digamma` (368) | 0.57 / 0.28 | 7.8 / 0.92 | 2.0e15 | 2.9e16 | | 1.1e17 | 1.7e16 | |
| `polygamma` (352) | 0.50 / 0.27 | 2.4e5 (`n ≤ 100`, 9 wrong) | 7.4e15 (4 wrong) | | | | 4.2 (`n = 1`) | |
| `beta` (222) | 0.53 / 0.29 | 386 / 33 | 2.2e3 / 158 | 5.6e5 (1 wrong) | | 1.2e6 (1 wrong) | 5.6e5 (`exp ln_beta`) | 1.4e6 (31 wrong) |
| `lbeta` (275) | 0.50 / 0.29 | | 886 / 54 | 2.1e16 | | 8.9e15 | 8.9e15 | |
| `factorial` (240) | 0.50 / 0.29 | 17 / 2.7 (`tgamma(x + 1)`) | 423 / 28 | | 19 / 2.8 | | | |
| `binomial`, integers (80) | 0.50 / 0.26 | 483 / 123 | 6.0e3 / 1.5e3 | | | 1.0e4 / 2.6e3 | | |
| `pochhammer` (190) | 0.49 / 0.27 | 3.7e5 / 2.9e4 | 4.6e5 / 4.9e4 | | | | | |
| `gamma_p` (439) | 0.65 / 0.27 | 3.5e4 / 1.6e3 | 1.6e5 / 7.7e3 | 1.3e7 | | 8.1e15 | 8.7e15 | 9.0e15 (63 wrong) |
| `gamma_q` (439) | 0.87 / 0.27 | 2.6e5 / 1.2e4 | 8.0e4 / 3.9e3 | 2.5e284 | | 8.6e15 | | 1.0e298 (63 wrong) |
| `gamma_p_inv` (125) | 0.54 / 0.21 | 22 / 3.8 | 6.5e15 (2 wrong) | | | | | 3.6e23 (16 wrong) |
| `gamma_q_inv` (134) | 0.51 / 0.22 | 149 / 14 | 1.1e60 | 1.2e35 | | | | |
| `beta_inc` (494) | 0.82 (median 0.22) | 2.0e8 (median 1.2, 157 above 4) | 1.8e13 (median 7.5) | 7.7e14 (median 5.7) | | 2.3e18 (median 209) | 1.5e11 (median 21) | 4.0e118 |
| `lgamma`, `Complex<f64>` (266) | 0.96 ε | | 1.4e12 ε (`Im` mod 2π) | | | | | |

Max / rms ulps on the repository's data (`n` cases, the normal range only); "wrong" counts NaN or infinite results where the value is finite, or the reverse. The references' large errors are mostly beside zeros of the functions (ln Γ near 1 and 2, ψ near its zeros, ln B near 0), at tiny or huge arguments, on the negative axis and in the far tails of `P`, `Q` and `I`, which the data cover on purpose; Boost's `beta_inc_inv` and `beta_inc_c_inv` reach 5.2·10⁴ and 4.9·10³ (in units of the condition number), and statrs's `inv_beta_reg` and special's `inv_inc_beta` take minutes on the inverses' data.

### Where the references are faster

Against the fastest reasonably accurate reference (Boost, or libm for `ln Γ`) in the same run (the ratios of the tables above, from before the fixes):

- **`ψ`** on `(0, 1)` (1.45× Boost) and for `x < 0` (1.95×: the reflection's double-double cotangent and `ψ(u) + 1/u`); **`ψ′`** on `[0.1, 8)` (1.9×: below 2 the reciprocal squares carry their errors); **`polygamma`** of orders above 1 for `x < 0` (2.1×: three Hurwitz zeta functions) and `n = 30` (GSL 2.4× faster, and 10⁷ ulps off).
- **`B` and `ln B`** of large arguments (2.5× Boost, which is 386 ulps off; special and statrs up to 10⁶): three double-double logarithms and the Stirling corrections.
- **`binomial`** of large integers (2.9× Boost, which is 483 ulps off), **`pochhammer`** of real `n` (1.9×): `ln B` and the Stirling difference.
- **`gamma` below −8** (1.4× Cephes, 1.0× Boost): Stirling, the exponential and the sine in double-double. **`ln Γ`** for `x < 0` (1.3× Boost), below ½ (1.4×).
- **`P`, `Q`** for integer and half-integer `a` below 20 (1.5× Boost: the double-double prefix); `x` far from `a` (1.4–1.7×). **`gamma_q_inv`** at tiny `q` (3.2×: the tail guesses need a second evaluation).
- **`beta_inc`** for `a, b ∈ [100, 10⁴]` near the mean (1.7×: about 40 fraction steps, or BASYM's recursion), small `a, b` (1.6×), BGRAT (1.4×); **its inverses** there (1.5–2.2×: the power terms are recomputed for the derivative).
- **Complex `ln Γ`** (3.1× GSL, which is reduced modulo 2π and 10¹² ε off): complex double-double throughout.

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
* **`erf`, `erfc` for `|x| < 1/2`.** `erf(x) = x + x·q(x²)`, `q` a polynomial of degree 9 (a minimax fit to 2⁻⁵⁷·⁴, 2⁻⁵⁵·⁷ with its coefficients as doubles, relative to `erf(x)/x`), rounded once by a fused multiply-add; `erfc(x) = (1 − x) − x·q(x²)` with `1 − x` exact as two doubles.
* **`erfcx` for `x ≥ 1/2`.** For `1/2 ≤ x < 8`, 32 minimax polynomials of degree 10 in `x − c`, one per eighth of a binade, picked from the bits of `x` (no division, no branch), each with its constant term as a double-double (2⁻⁵⁶·⁸ after the rounding of the coefficients); for `x ≥ 8`, `x·erfcx(x)` as a polynomial of degree 9 in `1/x²` (2⁻⁶⁰), times `1/x` as a double-double; from 2⁹⁶⁰, `1/(√π x)` with `1/x` of `x·2⁻⁵¹²` as a double-double and the result scaled back with one rounding, as beyond 2¹⁰¹⁵ the low words of the plain form would be subnormal. For `|x| < 1/2`, 8 polynomials of degree 11 on eighths of `[−1/2, 1/2)`; for `x ≤ −1/2`, `2e^(x²) − erfcx(−x)` with `e^(x²)` from `exp_dd`.
* **`erfc`, `erf` for `|x| ≥ 1/2`.** `erfc(x) = e^(−x²)·erfcx(x)`, `x²` exact as two doubles, `e^(−x²)` by `exp_dd` and the product as a double-double; `erf = 1 − erfc` and `erfc(−x) = 2 − erfc(x)` add the double-double to 1 or 2 exactly before the one rounding.
* **`erf_inv`, `erfc_inv`.** For `|x| ≤ 1/2`, a polynomial of degree 7 in `x²` to 2⁻³⁴ and one Newton step on the residual `erf(y₀) − x = (y₀ − x) + y₀·q(y₀²)`, exact but for its last rounding; tiny `x` as `x·√π/2` in two parts, rounded once. For `x > 1/2` (and `erfc_inv(c)`, `c ≤ 1/2`), `erfc_inv(1 − x)` with `1 − x` exact: 12 polynomials of degree 6 to 8 in `s = √(−ln c)` (two a binade of `s`, to 2⁻³¹), then one Halley step on `(erfc(y₀) − c)·e^(y₀²) = erfcx(y₀) − c·e^(y₀²)`, both terms as double-doubles and `c·2^k` scaled exactly, which holds for every subnormal `c`; `erfc_inv(c) = −erfc_inv(2 − c)` above 1.5 and `erf_inv(1 − c)` between.
* **`dawson`.** `x + x³·Q(x²)` (degree 8) for `|x| < 1/2`; 32 polynomials of degree 11 on eighths of binades for `1/2 ≤ |x| < 8` (2⁻⁵⁵); `x·F(x)` in `1/x²` (degree 9) beyond, and from 2⁹⁶⁰ `1/(2x)` as for `erfcx`.
* **`fresnel`.** For `|x| < 1`, `C = x + x⁵·P(x⁴)` and `S = x³·Q(x⁴)` (degree 8, `x³` and `Q` as double-doubles, `x⁴` rounded once from the exact `x²`; below 2⁻³³⁰, `S = (π/6)x³` scaled by 2³³⁰ and back with one rounding). From 1, `C = 1/2 + f·sin(πx²/2) − g·cos(πx²/2)` and `S = 1/2 − f·cos − g·sin` with the auxiliary functions `f` and `g` from 24 polynomials of degree 11 on eighths of the binades `[1, 8)` and as `x·f`, `x³·g` in `1/x⁴` (degrees 4 and 5) from 8; `x²` is exact as two doubles and each is reduced modulo 4 exactly, so the phase is right for any `x`, and `sin(πr/2)`, `cos(πr/2)` for `|r| ≤ 1/2` are minimax polynomials; the products and the sum with 1/2 are double-doubles.
* **`faddeeva` and the complex functions.** In the closed upper half-plane, by `w(−x + iy) = conj w(x + iy)` for `x ≥ 0`: for `|z| ≥ 10` the asymptotic series `i/(√π z)·Σ (2n − 1)!!/(2z²)ⁿ` to `n = 13`, plus `e^(−x²)` in the real part within `10⁻²⁰` of the real axis, and from `2¹⁰²⁰` its first term alone with `1/z` of `z·2⁻¹⁰` (Smith's denominator would overflow) scaled back once; for `|z| < 10` the trapezoidal rule of `w(z) = (i/π) ∫ e^(−t²)/(z − t) dt` with the step `h = 1/2` on 27 nodes shifted to `x + (n + 1/2)h` — no node meets the pole — plus its exact pole correction `2e^(−z²)/(1 + e^(2πy/h))` for `y < π/h`, left out only where it is below 2⁻⁵⁵ of the smallest `Re w` can be (`0.0039·min(y, 1)`): where its exponent is below −45, or below `−46 + ln y` for `y < 1` (for a small `y` it is all of `Re w` but a part of the size of `y`). The aliasing error is `e^(−π²/h²)`, below 2⁻⁵⁴ of `|w|` (0.3 ε measured in exact arithmetic); every node costs one real division (two nodes share one), and `e^(−(t₀ ± kh)²)` comes from `e^(−t₀²)`, `e^(∓t₀)` and `e^(−k²/4)` by products. On the real axis `w = e^(−x²) + 2i/√π·F(x)` and on the imaginary one `erfcx(y)`, from the real functions (below the real axis too). In the lower half-plane `w(z) = 2e^(−z²) − w(−z)`. The size of `e^(−z²)` comes from `y² − x²` as a double-double of exact squares and `exp_dd`, its power of 2 exact wherever a part of a result can be finite (`|y² − x²| < 1460`: beyond, a part of the factor it multiplies, at least 2⁻¹⁰⁷⁴ and at most 4, over- or underflows whatever it is) and applied to each part once, saturating; where a square overflows, `(|y| − |x|)(|y| + |x|)` is 0 or beyond `±2⁹⁷¹`. Its phase `−2xy` is reduced modulo `π/2` exactly, below `2²⁰` by Cody and Waite's method with a three-part `π/2`, and beyond by Payne and Hanek's from `2xy = m·2^e`, `m` the product of the two significands (106 bits), and the 256 bits of `2/π` at the units of `m·2^e`, taken from a table of 2,304 bits that covers every pair of doubles, also where `2xy` overflows; then fdlibm's kernels. So the size and the phase of the exponential are right to the last bits for any `|z|`, as far as a double holds them. Then `erfcx(z) = w(iz)`, `erfc(z) = e^(−z²)·w(iz)` for `x ≥ 0` (`1 − i·erfi(y)` on the imaginary axis) and `2 − erfc(−z)` for `x < 0`, and `erf(z)` by its Taylor series in `z²` (19 terms) for `|z| < 1` and `±(1 − erfc(±z))` beyond. Next to the imaginary axis, `|x|·max(|y|, 1) < 2⁻¹⁴` (`erf` from `|z| = 1`), `erf` and `erfc` come part by part from the Taylor series in `x` about `iy`: `Re erf = (2/√π)e^(y²)·x·(1 − (2y² + 1)x²/3 + (3 + 12y² + 4y⁴)x⁴/30)` as a double-double with its power of 2 apart, `Im erf = (2/√π)e^(y²)·(F(y) − y·x²·(1 − (3 + 2y²)x²/6))` (the terms left out are below 2⁻⁸⁰ of each part) and `erfc = 1 − erf`; below the real axis next to the imaginary one, `|x|·max(−y, 1) < 2⁻³⁰`, `w = erfcx(y) + 2ix·(1/√π − y·erfcx(y))` (below 2⁻⁵⁸), with `|y|·erfcx(y) = 2|y|e^(y²)` from `−y = 26` as a double and a power of 2. No standard-library transcendental function is used (but `ln` in the first guess of `erfc_inv`), so the results are the same on every target with a fused multiply-add.

The coefficients are this module's own minimax fits (a discrete Remez exchange in mpmath at 160 bits, `tables.rs` made by `gen_tables.py`); the sine and cosine kernels are fdlibm's, and the bits of `2/π` are mpmath's, the same as Arb's.

### Measured

The largest errors found of *Error bounds* are those of searches against Arb (FLINT 3.6, balls tightened until their radius is below 2⁻⁴⁰ ulp, a part settled on its own where it is measured so; asymptotic series with remainder bounds where Arb's own are not usable) on the aarch64-apple-darwin build: for an audit on 2026-11-20, 33.3 million `f64` arguments of the real functions (a million and more a region, every binade, ±300 ulps around every seam between pieces) and 72,405 more against mpmath, the two references agreeing to 6·10⁻¹⁷ ulp, and 16 million of the complex functions (a million a region); for the review of 2026-11-21, 117 million arguments of the real functions in the windows where the errors peak and local searches around the worst, and 25 million of the complex functions (17 million in local searches, 1.6 million where the exact value overflows, 2.7 million in the strips and bands the review changed); the review of 2026-11-22 ran 13.5 million of the complex functions and over 100 million arguments of the real ones (most of the largest found above are its), and after it 2.1 million of the complex functions beyond the box were measured as stated above, a finite part next to an infinite one against the scale (largest 2.65 ε; relative to themselves such parts of `erf` and `erfc` were up to 1.5·10⁵ ulps off, those below the real axis 1.90). Every complex error found is below the bound at its argument (the largest ratio is 0.73), and every real one below the bound of its region. Their scripts are not part of the repository; the analysis behind the bounds is (`python3 erf_bounds.py`, run from `data/`, prints the largest bound of each region in some minutes). On the code path without a fused multiply-add (wasm32 under Node) the errors are of the same size: before the review (100,000 to 200,000 arguments a region) `erf` 0.776, `erfc` 0.795, `erfcx` 0.664, `erf_inv` 0.821, `erfc_inv` 0.797, `dawson` 0.769, `fresnel` 0.897 and 1.050 ulps, the complex functions up to 4.52 ε of the scale; after it (40,000 to 300,000 arguments a region) `erf` 0.761, `erfc` 0.801, `erfcx` 0.655, `erf_inv` 0.804, `erfc_inv` 0.821, `dawson` 0.776, `fresnel` below 1 0.900 and 0.902 ulps, the complex boxes up to 4.17 ε, the parts of the strips next to the axes as with one, and every part infinite where the exact value overflows. The analysis there would count the unfused polynomial steps twice, and was not made. The tests pass on both paths (the second with the fused multiply-adds of the polynomials switched off in a copy of the crate, as x86_64 runs them without the `fma` feature).

The tests check 12,283 cases of `data/gen_erf.py` (run from that directory: `python3 gen_erf.py`, deterministic, a few minutes): random arguments by region, the worst arguments found with their neighbours, the diagonals and the coordinates beyond `√f64::MAX` of the complex functions, the arguments where `e^(∓z²)` is beyond any double or `|y² − x²|` is from 1400 to 1470, those where `|w|` is just beyond it (a finite part next to an infinite one), the points near their zeros, the strips next to the axes (part by part) and the `Complex<f32>` forms. A complex reference is computed at 128, 256, … more bits until two in a row agree to 110 bits a part; a complex case carries its scale as a third reference (beyond the largest double negated and divided by 2¹⁰²⁴). `cargo test --test math -- special::erf --nocapture` prints the largest error of each file.

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation only, never timed. The timings are those of the tuning campaign, before the fixes of 2026-11-21 and 22, which changed the code on rare paths but for: `dawson` on `|x| < 1/2`, 0.1 to 0.4 ns more (10 to 20 %) for the sign of −0; `faddeeva` for `6.7 ≤ |x| < 10` next to the real axis, about 20 % more, as it computes the pole correction where `Re w` needs it; the lower half-plane and `erfc` next to the diagonals from `|Re z·Im z| = 2¹⁹`, 5 to 35 % more for Payne and Hanek's reduction; the Fresnel integrals below 1, 1 to 6 % more for `x⁴` from the exact `x²`; and `erfc` where `e^(−z²)` is beyond any double (its result 0 or infinite), 8 % more for the saturating scale. Elsewhere the changes are within the noise of two quick interleaved timings of the reviews on a heavily shared machine (up to 7 %, as much where the code did not change).

| function | domain | a | libm | statrs | special | puruspe | gsl | boost | cephes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| erf | [-0.5,0.5] | 2.03 | 1.64 | 4.91 | 1.65 | 18.17 | 24.2 | 3.63 | 3.55 |
| erf | [0.5,3] | 4.78 | 8.05 | 7.21 | 8.03 | 18.09 | 23.99 | 11.19 | 15.3 |
| erf | [-3,3] | 4.4 | 6.9 | 7.32 | 6.88 | 18.1 | 23.81 | 8.38 | 14.22 |
| erf | [3,6] | 4.75 | 9.71 | 9.23 | 9.69 | 18.12 | 24.16 | 17.4 | 17.12 |
| erfc | [-0.5,0.5] | 2.38 | 1.69 | 4.83 | 1.69 | 18.05 | 16.45 | 3.35 | 3.8 |
| erfc | [0.5,3] | 4.67 | 7.58 | 6.96 | 7.58 | 17.94 | 22.57 | 10.86 | 14.68 |
| erfc | [-3,3] | 4.45 | 7.2 | 7.15 | 7.2 | 18.06 | 22.21 | 8.72 | 14.3 |
| erfc | [3,8] | 4.66 | 9.53 | 9.23 | 9.52 | 17.81 | 22.55 | 17.63 | 16.29 |
| erfc | [8,27] | 6.33 | 9.7 | 8.96 | 9.67 | 19.13 | 8.61 | 16.1 | 14.54 |
| erfcx | [-0.5,0.5] | 2.12 | - | - | - | 2.97 | - | - | - |
| erfcx | [0.5,8] | 1.8 | - | - | - | 2.37 | - | - | - |
| erfcx | [8,100] | 3.05 | - | - | - | 2.01 | - | - | - |
| erfcx | [-5,-0.5] | 4.9 | - | - | - | 4.05 | - | - | - |
| erf_inv | [-0.5,0.5] | 4.83 | - | 4.4 | 12.86 | 78.87 | - | 4.45 | 4.81 |
| erf_inv | [-1,1] | 11.24 | - | 6.9 | 15.65 | 78.76 | - | 6.09 | 9.39 |
| erf_inv | 1-[1e-15,0.5] | 14.38 | - | 9.38 | 30.99 | 79.91 | - | 11.65 | 10.78 |
| erfc_inv | (0,2) | 11.04 | - | 9.29 | - | 77.97 | - | 6.01 | 9.26 |
| erfc_inv | [1e-300,0.5] | 17.2 | - | 8.93 | - | 79.29 | - | 13.39 | 10.58 |
| dawson | [-0.5,0.5] | 1.4 | - | - | - | 7.61 | 13.29 | - | 7.46 |
| dawson | [0.5,8] | 1.78 | - | - | - | 13.18 | 32.32 | - | 10.01 |
| dawson | [8,1e4] | 2.48 | - | - | - | 13.19 | 33.8 | - | 4.06 |
| fresnel | [0,1] | 4.06 | - | - | - | - | - | - | 7.86 |
| fresnel | [1,8] | 12.22 | - | - | - | - | - | - | 20.58 |
| fresnel | [8,1e4] | 11.11 | - | - | - | - | - | - | 24.55 |
| faddeeva | \|z\|<1 UHP | 35.73 | - | - | - | 17.79 | - | - | - |
| faddeeva | [-4,4]x[0,4] | 35.35 | - | - | - | 18.75 | - | - | - |
| faddeeva | [-10,10]x[0,10] | 22.64 | - | - | - | 18.01 | - | - | - |
| faddeeva | [-4,4]x[-4,0] | 42.11 | - | - | - | 23.62 | - | - | - |
| faddeeva | [10,100]x[0,100] | 10.59 | - | - | - | 9.84 | - | - | - |
| erf c64 | \|z\|<1 | 10.76 | - | - | - | - | - | - | - |
| erf c64 | [-4,4]^2 | 40.94 | - | - | - | - | - | - | - |
| erfc c64 | \|z\|<1 | 42.47 | - | - | - | - | - | - | - |
| erfc c64 | [-4,4]^2 | 41.77 | - | - | - | - | - | - | - |
| erfcx c64 | \|z\|<1 | 39.72 | - | - | - | - | - | - | - |
| erfcx c64 | [-4,4]^2 | 38.71 | - | - | - | - | - | - | - |

The largest errors of the references on the same random sets, in ulps (complex: ε of `|w|`): `erf` libm 0.83, Boost 2.2, Cephes 2.5, GSL 5.5, statrs 4.5·10⁵; `erfc` libm 2.6, Boost 4.0, Cephes 9.4, GSL 502; `erfcx` puruspe 2.1 (489 for `x < −1/2`); `erf_inv` Boost 2.2, statrs 2.6; `erfc_inv` Boost 2.2, statrs 2.8; `dawson` Cephes 4.0, GSL 9.9; `fresnel` Cephes 13 below 8 and about 10¹¹ beyond; `faddeeva` puruspe (Johnson's algorithm) 1.7 for `|z| < 1`, 68 on `[−10, 10]²` and 3,800 for `|z|` from 10 to 10⁶.

### Where the references are faster

- `erf_inv` and `erfc_inv`: Boost and statrs, 1.5 to 2 times, at 1.6 to 2.8 ulps against 0.86: the first guess takes a logarithm and a square root, and the Halley step evaluates `erfcx` and an exponential.
- `erfcx` from 8: puruspe, 1.5 times, at 1.9 ulps: a division and a polynomial of degree 9 in `1/x²` against its table.
- `erf` and `erfc` on `|x| < 1/2`: libm, up to 1.3 and 1.4 times, at 0.83 ulps: a chain of 9 dependent multiply-adds against its rational of two short chains.
- `faddeeva`: puruspe, 1.1 to 2 times: 27 nodes and the pole correction, against an algorithm 20 to 4,000 times less accurate away from the origin.

## The Bessel and Airy functions

### Bessel functions

- `bessel_j(ν, x)` and `bessel_y(ν, x)` are the Bessel functions of the first and second kind `J_ν(x)` and `Y_ν(x)`, `bessel_i(ν, x)` and `bessel_k(ν, x)` the modified ones `I_ν(x)` and `K_ν(x)`, of any real order `ν` (an `f32` or an `f64`, the type of `x`). `bessel_i_scaled(ν, x) = e^(−|x|)·I_ν(x)` and `bessel_k_scaled(ν, x) = e^x·K_ν(x)` do not overflow or underflow where `I` and `K` would.
- **Negative orders** by the reflections: `J_(−ν) = cos(νπ)·J_ν − sin(νπ)·Y_ν`, `Y_(−ν) = sin(νπ)·J_ν + cos(νπ)·Y_ν`, `I_(−ν) = I_ν + (2/π)·sin(νπ)·K_ν`, `K_(−ν) = K_ν`, with `sin(νπ)` and `cos(νπ)` exact at the integers and half-integers (so `J_(−n) = (−1)ⁿ·J_n` exactly). The terms are formed in double-double with the scale of `Y_ν` and `K_ν` (for the scaled `I`, of `e^(−x)·K_ν`) kept apart, as either may be beyond the range where its product with `sin(νπ)` is not; where the two terms of `I_(−ν)` cancel, about its one zero for `ν` in `(2k − 1, 2k)` (below 100), `I_(−ν)` comes from its own power series instead.
- **Negative arguments** for the integer orders only: `J_n(−x) = (−1)ⁿ·J_n(x)`, `I_n(−x) = (−1)ⁿ·I_n(x)` (and `e^(−|x|)·I_n(x)` likewise); a negative `x` with a non-integer order, or for `Y` and `K`, is NaN.
- **Special values.** `J₀(0) = I₀(0) = 1`; `J_ν(0) = I_ν(0) = 0` for `ν > 0` and `±∞` for a negative non-integer `ν` (the sign of `1/Γ(1 + ν)`); `Y_ν(0) = −∞` (for `ν ≥ 0`), `K_ν(0) = +∞`; `J`, `Y`, `K` → 0 and `I` → `∞` as `x → ∞`, the scaled forms → 0. NaN in gives NaN, and so does an infinite order.
- **Spherical functions** of an order `n: u32`: `sph_bessel_j(n, x) = jₙ(x) = √(π/(2x))·J_(n+1/2)(x)`, `sph_bessel_y` (`yₙ`), `sph_bessel_i` (`iₙ`, of `I_(n+1/2)`) and `sph_bessel_k` (`kₙ = √(π/(2x))·K_(n+1/2)(x)`, SciPy's `spherical_kn`). `jₙ`, `yₙ` and `iₙ` are defined for negative `x` by their parities (`jₙ(−x) = (−1)ⁿ jₙ(x)`, `yₙ(−x) = (−1)ⁿ⁺¹ yₙ(x)`, `iₙ(−x) = (−1)ⁿ iₙ(x)`); `kₙ` of a negative `x` is NaN. `j₀(0) = i₀(0) = 1`, `jₙ(0) = iₙ(0) = 0` for `n ≥ 1`, `yₙ(0) = −∞`, `kₙ(0) = ∞`, and `yₙ` and `kₙ` are `∓∞` where they overflow at tiny `x`.
- **Large orders.** From `ν = 50` `I_ν` and `K_ν` (plain and scaled) come from Debye's uniform expansions, in constant time and to their bound at every order (`I` from its power series where `x² ≤ 16(ν + 1)` or `x < 30`, `K` from its leading term below `x = 10⁻¹⁵⁰`). `J_ν` and `Y_ν` take time that grows with `ν` where `x` is near or above the order, up to `ν = 10⁷`; above, they come from Olver's uniform expansion (with the module's Airy functions), in constant time, and are 0 and `−∞` where they underflow and overflow (`x` well below `ν`), and 0 from about `x = 6.7·10²⁹⁹` on, where the argument of the Airy functions passes `10²⁰⁰` (their phase is lost long before). The spherical `jₙ`, `yₙ` and `kₙ` of orders above `10⁷` follow `J`, `Y` and `K` of `n + 1/2`, and `iₙ` from `n = 50` on Debye's `I`, in constant time. No valid argument of a Bessel function gives NaN.

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

- **Orders 0 and 1 of `J` and `Y`.** Below 24, each interval between two extrema in three pieces: about its zero `f(z + d) = d·q(d)` with the zero as a double-double, and about the midpoints of the two outer thirds plain polynomials, Remez fits of degree 12 to 14, their tails from `c₃` by Estrin's scheme and their three outer steps of Horner's rule in double-double with `x − m` exact (at the ends of a piece `c₁u` and `c₂u²` reach half and a quarter of the value, where a step in `f64` cost up to an ulp): the error stays relative through the zeros. `J₀` below 1.5 and `J₁` below the first extremum are polynomials in `x²`; `Y₀` and `Y₁` below half their first zero are `(2/π)·ln x·J + R(x²)`, and about their first two zeros `(2/π)·ln(x/z)·J + (x − z)·g(x)`, whose fit `g` has no logarithmic singularity. From 24 on, the modulus and the phase, `J = M cos θ`, `Y = M sin θ`, `M = √(2/(πx))·(1 + P(1/x²))`, `θ = x − (2n + 1)π/4 + α(1/x)`, Remez fits, with `x` reduced modulo `π/2` to a double-double (Cody and Waite's three-part `π/2` below 10⁶, Payne and Hanek's 1280 bits of `2/π` above) and the sine and cosine of the reduced double-double by FreeBSD msun's kernels.
- **Orders 0 and 1 of `I` and `K`.** Polynomials in `x²` up to 1 (`K` with its `ln x·I` apart), `I` in nine pieces of `[1, 7.75]` about their midpoints, and `e^(∓x)·√x·(I, K)` in pieces of `1/x`: Remez fits with their first two coefficients in double-double, evaluated by Estrin's scheme; `1/√x` from `√x·(1/x)` with one correction, `e^x` from a table of `2^(j/32)` and a polynomial to 2⁻⁵⁸.
- **Other orders.** With `ν = n + μ`, `|μ| ≤ 1/2`: `J_μ` and `J_(μ+1)` from their power series summed in double-double (it serves up to 30, where its terms cancel down to `e^(−x)`); `Y_μ` and `Y_(μ+1)` from Temme's series below 2 (Temme 1975; its leading terms are of order 1 and cancel, so its ingredients `μπ/sin μπ`, `Γ₁`, `Γ₂`, `sinh σ/σ`, `cosh σ`, `e^σ` and its first two steps are formed in double-double, the rest in `f64`) and from Steed's continued fraction CF2, `p + iq = (J′ + iY′)/(J + iY)`, summed backwards (`120/x + 8` terms), as `Y_μ = (p·J_μ − J′_μ)/q`; Hankel's expansion from 30 on; then the recurrences in double-double up to `ν` (`Y` always, `J` while `x > ν`). Where `x ≤ ν`, `J_ν` comes from its series or from the Wronskian `J_ν = (2/(πx))/(r·Y_ν − Y_(ν+1))`, `r = J_(ν+1)/J_ν` by CF1 summed from the tail in double-double (near the turning point its partial ratios stay near 1 for about `7ν^(1/3)` terms, through which the rounding errors of an `f64` run pile up), at least so many terms that the zero tail costs below `e^(−48)` (by Debye's exponent of `J_(ν+k)`; Lentz's test stops short there), `r·Y_ν − Y_(ν+1)` in double-double (the terms cancel to about `ν^(−1/3)` of each, and an error of the recurrence that is a multiple of `J` drops out). `I_ν` from its power series (positive terms, double-double) below 30, else Hankel's expansion or the Wronskian with CF1; `K_μ`, `K_(μ+1)` from Temme's series below 1/2 and from the Wronskian `I_μ·K_(μ+1) + I_(μ+1)·K_μ = 1/x` with `K_(μ+1)/K_μ` by CF2 summed backwards (no step of it cancels), then up by the recurrence. Integer orders start from the dedicated `J₀`, `J₁`, `Y₀`, `Y₁`, `I₀`, `I₁`, `K₀`, `K₁` (beyond `x = 700`, where `K₀` and `K₁` leave the normal range, from the scaled ones with `e^(−x)` apart). The scaled forms carry `e^(∓x)` as a separate exponent throughout. From `ν = 50`, `I` and `K` by Debye's expansions `I_ν(x) = e^(νη)/√(2πr)·Σ Uₖ(p)/νᵏ`, `K_ν(x) = √(π/2)·e^(−νη)/√r·Σ (−1)ᵏUₖ(p)/νᵏ`, `r = √(ν² + x²)`, `p = ν/r`, `U₁` to `U₁₂` from their recurrence. The exponent `νη` is the size of the result in `e`-folds, its absolute error the result's relative error: below `ν = 2¹⁶` it is `r + ν·ln(x/(ν + r))` in double-double (for the scaled forms below `x = 2⁴⁵`); beyond, where that form carries `ν` times the error of the logarithm and, for the scaled forms, the cancellation of `r − x`, `νη − x = −p·xν/(r + x) − ν(atanh p − p)` for the scaled forms (terms of one sign, the first as `p` times `xν/(r + x)`, factors that stay normal where `ν²/(r(r + x))` does not, near the top of the range; `atanh p − p` by its series where `p` is small, else from `ln((r + ν)/x)`), and for the plain ones, in range only within about 1300 of `x = z₀ν` (`z₀ = 0.6627…` the root of `η`, Laplace's limit), `νη = D·P(D/ν)` from `D = x − z₀ν` formed exactly with the bits of `z₀` (as Payne and Hanek reduce by `2/π`) and `P` the Taylor series of `η/(z − z₀)` about `z₀`. For the scaled `I` at a negative order, `e^(−x)·K_a = (e^x·K_a)·e^(−2x)` with the scales apart, decided first by its exponent `−aη − x` in `f64` where that is beyond the range; from `a = 2¹⁶` on, where the two factors' exponents are about `0.9a` each, that exponent, `−a(η(z) + z)` with `z = x/a`, comes whole from `D = x − z₁a` (`z₁ = 0.4477…` the root of `η(z) + z`, its bits multiplied out as for `z₀`) and the Taylor series of `(η(z) + z)/(z − z₁)` about `z₁`. Above `ν = 10⁷`, `J` and `Y` by Olver's expansion to its terms `A₀` and `B₀`, `ζ`, `φ` and `B₀` by their series about `x = ν` or their closed forms.
- **Spherical functions.** `j₀ = sin x/x`, `y₀ = −cos x/x` from the double-double reduction; the power series where `x² ≤ n + 3/2`; the terminating Hankel sums `jₙ = (P sin(x − nπ/2) + Q cos(x − nπ/2))/x` from `x ≥ n(n + 1)/2`; between, `jₙ` up from `j₀`, `j₁` where `x > n` and down from the continued fraction of `jₙ/jₙ₋₁` where `x ≤ n` (that of `J`, summed from its tail in double-double at every `x`, as the run down passes its error to `jₙ` whole; normalised by `j₀` or `j₁`), `yₙ` up from `y₀`, `y₁`, all in double-double from double-double starts; `yₙ` and `kₙ` below `x = 2⁻⁵⁰⁰` from their leading term `(2n − 1)!!/x^(n+1)`. `kₙ = (π/(2x))·e^(−x)·Σ aₖ/(2x)ᵏ` by Horner's rule in double-double up to `n = 10`, beyond by the recurrence of positive terms. `iₙ` from its power series (up to `x = 20 + n/2` and `n²/28` below `n = 50`, `x² ≤ 16(n + 1.5)` or `x < 30` from it on), from the closed form `(e^x·Σ(−1)ᵏaₖ/xᵏ − (−1)ⁿe^(−x)·Σ aₖ/xᵏ)/(2x)` in double-double with both exponentials between there and `x = n(n + 1)/2` below `n = 50` (where its terms cancel by `2⁴⁰` at most), and from Debye's `I_(n+1/2)` there from `n = 50` on; beyond `x = n(n + 1)/2` from the closed form (`∞` beyond 720).
- **Airy functions.** For `x ≥ 4` (`Bi`, `Bi′`: 12), `Ai(x) = e^(−ζ)/(2√π·x^(1/4))·(1 + t·A(t))`, `t = 1/ζ`, and likewise for the others; from 1 up to there Remez fits of `e^(±ζ)·f` in pieces; from the first extremum left of 0 up to 1 Remez fits of `f` itself, in pieces halved until the fit holds `2⁻⁵⁷` with its coefficients rounded to `f64` (at the ends of the widest pieces `c₂u²` and `c₃u³` reach the size of the value); down to about −8.5 one polynomial about each zero (as for `J₀`); these by Horner's rule with the last three steps in double-double. Below, the modulus and the phase, `Ai(−y) = M cos θ`, `Bi(−y) = M sin θ`, `θ = π/4 − ζ + α(1/ζ)` (`Ai′`, `Bi′` with `N` and `φ = 3π/4 − ζ + β`), Remez fits, the phase reduced in double-double, both parts of `ζ` modulo `π/2` (its low part passes `π/4` from about `|x| = 10¹¹`).
- Nothing allocates. The read-only tables hold about 35 KB of coefficients (`J₀`…`Y₁` 14 KB, `I₀`…`K₁` and `1/Γ(1 + μ)` 6.3 KB, the large orders 2.4 KB, the Airy functions 11.9 KB), `2^(j/32)` 512 bytes and the bits of `2/π` 160 bytes; their generators (`gen01.py`, `gen_ik.py`, `gen_olver.py` and the shared `remez.py` in `bessel/`, `gen_airy.py` and `airy_post.py` in `airy/`, on mpmath) reproduce them exactly. The elementary functions are the module's own (`e^x`, `ln x`, the sine and cosine, `atan`), and the polynomials use `mul_add` where it is one instruction (aarch64, x86_64 with FMA) and plain products elsewhere, while the error terms of the double-double arithmetic are exact everywhere (a fused multiply-add or Dekker's product). All targets of one kind give the same bits; the others (x86_64 without FMA, and Miri) differ in the last bits. They were checked by the same code built with the fused path off on aarch64, where every test passes and the sweeps stay within the bounds (an error term once left to the plain path had cost `Y₁` 1.95 ulps there), and by the tests under Miri on every 37th case of their data, not on x86_64 itself.

### Measured

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation in that campaign only (after the fixes of 2026-11-21 they were checked by a build without FMA, see the methods), and never timed.

| function | domain | a | libm | puruspe | gsl | boost | cephes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| j0 | [0,25] | 10.22 | 16.36 | 5.77 | 36.41 | 7.56 | 14.82 |
| j1 | [0,25] | 11.06 | 17.07 | 4.77 | 36.35 | 8.26 | 14.61 |
| y0 | [0,25] | 19.64 | 17.93 | 7.57 | 40.91 | 11.34 | 15.75 |
| y1 | [0,25] | 22.42 | 17.69 | 6.04 | 39.67 | 12.8 | 15.47 |
| j0 | [25,1e4] | 25.06 | 19.1 | 7.55 | 38.58 | 19.43 | 15.34 |
| j1 | [25,1e4] | 24.85 | 19.63 | 5.94 | 39.22 | 20.19 | 15.56 |
| y0 | [25,1e4] | 22.74 | 21.25 | 7.6 | 38.95 | 20.77 | 15.41 |
| y1 | [25,1e4] | 22.19 | 21.54 | 5.46 | 39.99 | 21.45 | 15.24 |
| i0 | [0,50] | 12.45 | - | 3.26 | 23.51 | 10.33 | 12.69 |
| i1 | [0,50] | 12.59 | - | 3.36 | 23.52 | 10.4 | 12.58 |
| k0 | [0.01,50] | 11.11 | - | 3.94 | 17.57 | 7.57 | 12.18 |
| k1 | [0.01,50] | 11.23 | - | 3.92 | 18.47 | 7.98 | 12.61 |
| jn | n 2..20, x [0.1,50] | 142.54 | 58.01 | 32.95 | 156.99 | 32.39 | 132.23 |
| yn | n 2..20, x [0.1,50] | 140.08 | 46.81 | 19.88 | 88.94 | 21.35 | 45.62 |
| jv | ν [0,10], x [0.1,2] | 93.51 | - | 64.39 | 105.44 | 44.91 | 77.49 |
| yv | ν [0,10], x [0.1,2] | 131.68 | - | 65.78 | 68.38 | 82.78 | 178.67 |
| iv | ν [0,10], x [0.1,2] | 83.58 | - | 61.08 | 120.23 | 62.18 | 91.98 |
| kv | ν [0,10], x [0.1,2] | 246.32 | - | 61.77 | 76.68 | 73.07 | - |
| kv_scaled | ν [0,10], x [0.1,2] | 243.59 | - | - | 66.98 | - | - |
| jv | ν [0,10], x [2,30] | 426.68 | - | 196.36 | 150.35 | 107.46 | 86.48 |
| yv | ν [0,10], x [2,30] | 454.22 | - | 188.76 | 191.56 | 138.44 | 174.42 |
| iv | ν [0,10], x [2,30] | 230.72 | - | 120.74 | 127.91 | 190.41 | 254.49 |
| kv | ν [0,10], x [2,30] | 345.43 | - | 120.02 | 75.77 | 121.51 | - |
| kv_scaled | ν [0,10], x [2,30] | 344.53 | - | - | 71.73 | - | - |
| jv | ν [0,3], x [30,1e3] | 69.29 | - | 2455.06 | 1735.12 | 26.75 | 12.83 |
| yv | ν [0,3], x [30,1e3] | 69.65 | - | 2414.67 | 1746 | 26.5 | 30.28 |
| iv | ν [0,3], x [30,1e3] | 28.29 | - | 406.18 | 128.51 | 7284.32 | 387.51 |
| kv | ν [0,3], x [30,1e3] | 32.12 | - | 407.37 | 27.03 | 29.02 | - |
| kv_scaled | ν [0,3], x [30,1e3] | 25.34 | - | - | 18.49 | - | - |
| jv | ν [10,100], x [1,200] | 651.69 | - | 508.3 | 248.53 | 496.1 | 289.71 |
| yv | ν [10,100], x [1,200] | 680.04 | - | 509.95 | 294.57 | 653.44 | 832.28 |
| iv | ν [10,100], x [1,200] | 249.57 | - | 210.23 | 198.4 | 218.09 | 328.27 |
| kv | ν [10,100], x [1,200] | 201.79 | - | 209.89 | 110.55 | 132.84 | - |
| kv_scaled | ν [10,100], x [1,200] | 214.5 | - | - | 103.31 | - | - |
| sph_j | n 0..10, x [0.1,50] | 56.74 | - | - | 109.32 | 66.95 | - |
| sph_y | n 0..10, x [0.1,50] | 76.29 | - | - | 42.27 | 77.77 | - |
| sph_k | n 0..10, x [0.1,50] | 36.93 | - | - | 11.82 | - | - |
| airy_ai | [-10,10] | 15.32 | - | - | 48.93 | 217.9 | 85.56 |
| airy_bi | [-10,10] | 11.29 | - | - | 46.58 | 340.83 | 85.94 |
| airy_ai_prime | [-10,10] | 14.66 | - | - | 26.23 | 219.34 | 85.78 |
| airy_ai | [-1e3,-10] | 36.43 | - | - | 79.15 | 66.89 | 32.33 |
| airy_bi | [-1e3,-10] | 35.7 | - | - | 80.19 | 66.13 | 31.95 |
| airy_ai_prime | [-1e3,-10] | 36.39 | - | - | 39.01 | 72.73 | 31.18 |

These timings, and the column `a` of the table below, predate the fixes of the audit of 2026-11-21 (the accuracy after them is in the accuracy section above). A quick timing of the paths the fixes touch, not under the campaign's conditions (best of 15 rounds of 4,000 calls, ns a call, before → after; where marked "interleaved", the least of 15 runs of each build in turn, on a loaded machine): `jv` ν [10,100], x [1,200] 772 → 797 and x [√ν, ν] 1014 → 1165 (CF1 from its tail in double-double where `x ≤ ν`), ν [50,300] at the turning point 2185 → 2538, ν [0,10], x [0.1,2] 116 → 132; `yv` ν [0,10], x [0.1,2] 162 → 299 (Temme's series in double-double; interleaved 155 → 285); Debye's `iv` and `kv` (ν [50,1000], x [1,2000]) 98 → 105 and 94 → 102, the scaled ones 94 → 101 and 90 → 96, and at the orders from 2¹⁶ on where they are in range (interleaved, ν [10⁵,10¹²]) `iv` and `kv` about `x = z₀ν` 84 → 110 and 86 → 110, the scaled ones at `x = ν²/(2t)`, `t` [10⁻³,700], 67 → 105 and 68 → 108 (Debye's exponent from `x − z₀ν` and in the one-signed scaled form, where it was up to 10¹² ulps off); the negative non-integer orders (interleaved, ν [−30,0], x [0.1,60]) of `jv` and `yv` 553 → 599 and 562 → 604 (8%, and 20% in a second such run: Temme's series below x = 2), `iv` and the scaled `iv` 641 → 650 and 648 → 659; `sph_j` 79 → 81, `sph_y` 93 → 60, `sph_i` (n 0..60, x [0.1,200]) 404 → 423; the Airy functions within 10%; `j0` and `y0` within 6%, `j1` within 10% (interleaved 15.0 → 16.4); `jn` (n 2..20, x [0.1,50]) 11 to 17% (interleaved, 400 → 443 on the efficiency cores). After the second review (2026-11-22; both builds on the efficiency cores, the least of 7 interleaved runs, as the loaded machine's runs on the performance cores did not compare): `sph_j` 2% on n 0..10, x [0.1,50] and 1 to 10% where `x < n/2` (its continued fraction from the tail in double-double there too), the scaled `iv` at the orders from −2¹⁶ down about `x = 0.448|ν|` 5%, the rest unchanged.

The references' accuracy on the same data, max/rms ulps of `max(|value|, M)`:

| function | cases | `a` | libm | puruspe | gsl | boost | cephes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J | ν = 0, 1 | 0.50/0.16 | 2.20/0.45 | 2.506e13/1441161177434.60 | 2.506e13/1441161177434.61 | 26.08/1.80 | 2.506e13/1441161177434.57 |
| Y | ν = 0, 1 | 0.68/0.18 | 2.22/0.49 | 1.517e14/7749379708648.51 | 85.22/8.39 | 3.95/0.71 | 1.517e14/7749379708648.47 |
| J | ν ≥ 0 other | 2.25/0.25 | - | 2.918e6/317241.63 | 3.159e3/315.83 | 2.775e2/19.55 | 2.751e7/1917018.02 |
| Y | ν ≥ 0 other | 3.37/0.77 | - | 1.299e6/115066.80 | 1.157e3/145.32 | 72.33/6.78 | 9.542e15/664838879626942.12 |
| I | ν = 0, 1 | 0.53/0.28 | - | 5.03/1.60 | 1.492e3/163.41 (8 non-finite) | 3.23/0.83 | 4.36/1.16 |
| I | ν ≥ 0 other | 0.94/0.31 | - | 1.752e4/1766.71 (1 non-finite) | 1.311e3/130.95 | 6.780e2/49.26 (1 non-finite) | 8.696e15/2542791248379523.00 (2 non-finite) |
| K | ν = 0, 1 | 0.55/0.28 (1 non-finite) | - | 13.53/4.23 (1 non-finite) | 9.60/1.93 (1 non-finite) | 2.13/0.64 (1 non-finite) | 6.95/1.31 (1 non-finite) |
| K | ν ≥ 0 other | 2.36/0.77 (1 non-finite) | - | 1.716e2/16.78 (1 non-finite) | 1.323e2/13.74 (1 non-finite) | 15.59/3.06 (1 non-finite) | 5.038e263/inf (5 non-finite) |
| sph j | all x > 0 | 1.13/0.23 | - | - | 3.671e7/3098349.17 | 6.11/1.24 | - |
| sph y | all x > 0 | 0.59/0.27 | - | - | 7.984e2/72.93 | 15.40/2.04 | - |
| Ai | x ≥ −10⁶ | 0.69/0.17 | - | - | 4.458e8/39821786.30 | 2.793e8/21281436.76 | 6.483e7/5625636.06 |
| Ai' | x ≥ −10⁶ | 0.51/0.14 | - | - | 1.360e8/13157984.64 | 1.647e8/13653073.50 | 2.661e8/20808008.42 |
| Bi | x ≥ −10⁶ | 0.50/0.24 | - | - | 5.274e8/66902473.09 | 9.436e9/711347516.32 | 6.668e240/inf |
| Bi' | x ≥ −10⁶ | 0.70/0.22 | - | - | 2.135e8/16314096.69 | 2.002e7/2240653.48 | 1.815e280/inf |

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
- **`Li_s`.** `2⁻ˢ|x| < 2⁻⁶⁰`: `x`. `0 < s < 2⁻¹⁰²²`: `x/(1 − x)`, within `40s` of it relative. `|x| ≤ 3/4` (`|x| ≤ 1/2` for `s < 0`, and every `x ≥ −1` for `s ≥ 20`; past `|x| = 1/2` the expansions about `±1` cancelled, by up to 3 and 1.6, and reached 4 and 6.4 ulps): `Σ xᵏ k⁻ˢ`, `p⁻ˢ` of the primes from double-double logarithms and the rest by products, for `s < −1` each term from logarithms. `3/4 < x < 1` (`1/2 < x < 1` for `s < 0`): `Γ(1 − s)(−μ)^(s−1) + Σ ζ(s − k) μᵏ/k!`, `μ = ln x` a double-double (relative to itself near 1), `Γ(1 − s)` with the low part of `1 − s`; for `n = 1` (`n` the integer nearest `s ≥ 1/2`) and near the integers past it the pole of `ζ(s − k)` at `k = n − 1` and that of `Γ(1 − s)` taken together in closed form (through `ln Γ(1 − ε)/ε`, `ln Π(1 + ε/j)/ε` and `expm1` of a double-double); `ζ(s − k)` past `s − k < −1/2` by the reflection as a recurrence in `k`; for `s < −20` the first term alone, `ln Γ` in double-double; `+∞` once that first term leaves the range. `−2 ≤ x < −3/4`: `−Σ η(s − k) μᵏ/k!`. `x < −2`, `s > 0`: integers by the inversion `−(−1)ⁿ Li_n(1/x) − Σ 2η(2j) μⁿ⁻²ʲ/(n − 2j)!`, the sum nested as `(μⁿ/n!)[1 + r₁(2η(2) + r₂(…))]` in double-double so that its cost is linear in `n` and no factorial overflows; the others by the integral of Fermi and Dirac, `Li_s(−eᵘ) = −F(μ)/Γ(s)`, every term positive (on `[0, min(1, μ)]` the Taylor series of the logistic integrated exactly, past it 16-point Gauss–Legendre pieces about exact midpoints), or, where they apply, the divergent series cut at its least term (orders large against `ln|x|`) and Sommerfeld's expansion (`x < −e^37.7`, `μ` a double-double, `1 + s` an exact pair with its low part through `ψ`, past `s = 160` `ln Γ(1 + s)` in double-double); for `s > 21` the powers at the quadrature's nodes from double-double logarithms, and where `t^(s−1)` would leave the range `ln Γ(s)` in double-double. `s < 0`, `x < 0` (an integer order with `x < −1` by `Li_s(x) = (−1)^(s+1) Li_s(1/x)`, `ln(−1/x) = −μ` exactly): the series, each term from logarithms as a double-double, where `|x| ≤ 1/2` and no term is above about 4000 times the first (the largest is at `k ≈ |s|/ln(1/|x|)`), so that the cancellation of its alternating terms costs nothing; else `Li_s(x) = 2 Γ(1 − s)(2π)^(s−1) Re[i^(1−s) ζ(1 − s, 1/2 − iμ/2π)]` with the Hurwitz zeta function of a complex argument by the Euler–Maclaurin formula, each term from `ln|n + a|` and `arg(n + a)` in double-double relative to the first, that first's logarithm joined with `ln(2 Γ(1 − s)(2π)^(s−1))` in one exponent, the direct terms stopped where the rest is negligible, and the remainder past them, its products and `Re[i^(1−s) Z]` complex double-doubles, rounded once in the last product (`σ − 1 = −s` exactly); for `−1/4 < s < 0` the pole of `ζ(σ, a)` at `σ = 1` taken out of the remainder (`(w^(1−σ) − 1)/(σ − 1) = −expm1(s ln w)/s`, by its series or from `e^(s ln w)` in double-double) and its part `−sin(πs/2)/s` added in closed form, so that parts of size `1/|s|` do not cancel to `O(1)`; further left the pair is not needed (in units of `|a|^(−σ)` it is of size `|a|^(1−s)`, against a result of `|a| sin(π|s|)/|s|`). The real part cancels near the real zeros of `Li_s`, by `1/sin(π|s|) ≈ κ/π` as `s → −1`, and also where `|x|` is near `2^(±|s|)`, by about 50 for an integer order whose `κ` is 1 there: the series takes those. Integers `−18 ≤ s ≤ −2`, `0 < x < 1`: `x Aₙ(x)/(1 − x)ⁿ⁺¹` with Eulerian polynomials (their coefficients exact, below `2⁵³`) in double-double, `+∞` past the range.
- **Lambert W.** Near the branch point both branches are polynomials in `p = √(2(1 + e x))` (`±p`), `1 + e x` without the cancellation (`e` as a double-double), on `|p| ≤ 0.66`; `W₋₁` goes on to `x = −0.2` in `s = √(2(−ln(−x) − 1))`, `s = √(−2 ln(1 − q))` a double-double from the double-double `q`; each a minimax fit of full precision, its first two terms a double-double. `W₀` on `0.66 < p ≤ 4.31` (`−0.29 < x ≤ 3`) is `x e^(−W₀)` with `e^(−W₀)` a polynomial of full precision in `p` on five pieces (`p` a double-double, the product rounded once; its series below `|x| = 2⁻¹⁰`), without iteration. Elsewhere a fitted starting value good to `3·10⁻⁷` (in `ln x`, in `s`, or the asymptotic series) and one step of Halley's iteration with the residual `w eʷ − x` from a double-double `eʷ` (so that its only error is the final rounding), or for `|w| > 40` one step of Fritsch's iteration (of the fourth order).
- **`E₁`, `Ei`, `Eₙ`.** `E₁`: `F(x) − ln x` on `(0, 1]` with the entire `F = E₁ + ln x = −γ + x − x²/4 + c₃x³ + x⁴T(x)` a minimax polynomial of degree 14, `ln x`, `x − x²/4` and `c₃x³` in double-double and the sum rounded once (near 1 the result is a quarter of its parts); `e⁻ˣ (H₀ + K(x))/x` on `(1, 2.5]` with `K` a rational fit (6/6) evaluated in `(x − 1)/0.75`, where its denominator's terms are all positive; past it Boost.Math's rational of degree 10/11 in `1/x`; `e⁻ˣ` to about `2⁻⁷⁵` and the product rounded once. `Ei`: `ln(x/x₀) + (x − x₀) R(x)` on `(0, 6]` (`R` entire, four polynomial pieces, `c₀ + c₁t` of each a double-double; `x − x₀` exact with `x₀` a triple-double; `ln(x/x₀)` a double-double, by `ln_1p` and a Newton step near `x₀` and up to 1.5; the sum rounded once); `eˣ (1 + K(x))/x` past 6 with `K` rational fits on four pieces to 48, each evaluated in a variable from the left end of its piece (where the terms of both polynomials are positive: about the middle of the piece they alternate and cancelled by up to 95 at its left end), and polynomials in `1/x` past it. `Eₙ`, `n ≥ 2`: the recurrence `Eₖ₊₁ = (e⁻ˣ − x Eₖ)/k` up from a double-double `E₁` for `x ≤ 2` and `n < 50`, else the continued fraction, its depth from Lentz's test (and a quarter more) and its value from the bottom up in double-double arithmetic; `E₀ = e⁻ˣ/x`.

The fits are minimax (Remez exchange for the polynomials, Lawson's iteration for the rationals, mpmath at 128 bits) made for this crate, but for `E₁` past 2.5, Boost.Math's. The other tables (the Riemann–Siegel function's Taylor coefficients, the Stieltjes constants, `atan(k/8)`, Stirling's coefficients) are printed by `zeta/gen_complex.py`.

### Measured

The accuracy above was measured by the tests (`cargo test --test math -- special::zeta` prints the largest error of each region on the data of `tests/math/special/data/zeta_*.txt`, regenerated by `data/gen_zeta.py`) and by surveys against Arb outside the repository (16,139 complex points for `ζ` and about 140,000 more by the regions of its bounds, 73,897 real ones, 14,700 for `ζ(s, a)`, 35,470 for `Li₂`, 5,500 random and 12,000 targeted for `Li_s`, about 20,000 a branch of W, 19,344, 24,864 and 9,080 for `E₁`, `Ei`, `Eₙ`, each repeated for `f32`). The release and the debug builds give the same bits on 66,383 of these arguments (the first version gave one polylogarithm 8 ulps apart between them). After the first review of the fixes (2026-11-22) the functions it touched were surveyed again on the final code: the complex function on 66,600 points by the regions of its bounds to `t = 2⁴⁰` (two samples) and as many in `f32`, 14,400 by height near `σ = 1/2`, 11,900 at the switch points of its methods, 22,000 left of the strip below `t = 13.5` and 6,800 at the trivial zeros with a tiny `t` (5,000 of them with each part against mpmath at `120 + log₂(1/t)` bits, since Arb's references are good only relative to `|ζ|`); `Li_s` on about 290,000 (59,400 by region and 48,500 in `f32`, 22,900 past `−2` by path, by order and by binade, 20,900 with `|x|` near `2^(±|s|)`, 6,000 of the Eulerian form, and 130,000 about the switch points `x = ±1/2`, `±3/4` and `−1`); `Ei` on 560,000. The references of its regression rows agree with Arb to `2⁻⁹⁵` or better, every set of them fails on the code before it, and the release and the debug builds gave the same bits on 90,314 arguments. After the second review (2026-11-22): `Li_s` with `s < 0`, `x < 0` on 41,500 points (30,500 with `−1 < s < 0` by decade of `|x|` to `10³⁰⁰` and about the switch at `s = −1/4`, 3,000 each with `s` in `(−3, −1)` and `(−30, −3)`, 5,000 in the regions of the general survey) and 7,500 integer orders with `x < 0`, the general survey again (63,300 points in 37 regions, 58,986 in `f32`) and 2,030 switch-point and edge arguments; `Ei` on 434,800 points of `(6, 717]`; the complex function at 874 edge arguments (switch points, tiny and subnormal parts). The release and the debug builds now give the same bits on 121,893 arguments (90,893 from the data files and the audit's surveys, and the second review's 31,000), three of which (polylogarithms of `−1 < s < 0` far left) had differed in the last bit: the release build merged a sine and a cosine of one argument into one call of the system's `sincos`, whose last bit can differ from theirs; that code is gone (the pole's remainder takes this crate's double-double sine and cosine).

The timings below are those of the campaign of the first version (an Apple M4 Max on one thread with fat LTO, the median of three quiet rounds, against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 without promotion to `long double` and Cephes in the same run; "–" is a reference without the function; the x86_64 paths were tested under emulation only, never timed). The fixes of 2026-11-21 were timed only against the first version on a loaded machine without LTO (the minimum of fifteen rounds, ns a call): `expint_e1` on `[0.01, 1]` 5.6 → 12.2, `expint_ei` on `[0.01, 6]` 6.7 → 10.0 (1.6 times when measured again after the second review: 7.8 → 12.7), `lambert_wm1` on `[−0.36, −0.01]` 10.8 → 15.9, `dilog` on `[−1, 1]` 14 → 16.5, `polylog` of `s = 2.5` on `[0, 1]` 151 → 174 and of `s = 3` on `[−1, 1]` 149 → 162, of `s ∈ [−10, −0.1]` on `x ∈ [−2, 0.9]` 272 → 1,198 (the Hurwitz form in double-double), `zeta` on `[−0.5, 0.9]` 4.5 → 5.1; the others (`hurwitz_zeta`, `zeta` left of −1/2, `lambert_w0`, `expint_e1` past 1, `expint_ei` past 6) unchanged within 3 %. The complex function on the table's grids: `s = [0.5, 5] + i[−50, 50]` 563 → 683, `s = [−10, 0.5] + i[−20, 20]` 655 → 747, `s = 1/2 + i[100, 1000]` 5,546 → 6,466 (the reflection for every `σ < 1/2` accounts for 1 to 2 % of it; the rest came with the earlier fixes of the same day, and was not profiled); past `t = 10⁴` it costs `√t` instead of `t`. The changes after the first review of the fixes (2026-11-22) were timed against the code before them and the first version, the minimum of six to ten interleaved runs on a machine loaded to 70–480 whose cores differ (ratios within 1 ± 0.3 are noise there): the complex function's grids within 5 % (its terms in the strip double-doubles), the Riemann–Siegel range half as costly on the critical line (one sum serves both parts: 0.38 ms at `t = 10⁸`, 3.6 ms at `10¹⁰`, twice that off the line), left of the strip below `t = 13.5` three times as fast (the logarithmic form for `σ ≤ −12`); `polylog` of the integer orders `−18 ≤ s ≤ −9` on `(0, 0.9]` a quarter of its cost before (0.24 times, measured again after the second review: the Eulerian form in double-double, 0.14 µs), of the integer orders near `|x| = 2^(−|s|)` half (the series in place of the Hurwitz form), of large orders past `x = −e³⁹` 2.8 times (the integral of Fermi and Dirac with its powers from double-double logarithms: 24 µs for `s ∈ [21, 400]`); the others unchanged. The last two changes of that review, the sine of the logarithmic form as a double-double and the series of `Li_s` to `|x| = 3/4` (timed against the code before them, eight interleaved runs), cost 1.08 to 1.16 times as much left of the strip below `t = 13.5` past `σ = −12` (0.25 to 0.5 µs) and 1.12 to 1.5 times on `s > 0`, `1/2 < |x| ≤ 3/4` (0.29 µs); the table's grids are unchanged within 3 %. Against the first version, whose Eulerian form in double lost up to 39 ulps, the integer orders on `(0, 0.9]` cost (measured again after the second review) 1.1 times as much for `−8 ≤ s ≤ −2` (22 ns) and twice for `−18 ≤ s ≤ −9` (0.14 µs), both by the Eulerian form in double-double, four times for `−30 ≤ s ≤ −19` (0.67 µs) and twice for `−64 ≤ s ≤ −31` (0.9 µs), by the series of terms from logarithms (each a `ln k` in double-double and an exponential) and the expansion about 1. The changes after the second review (eight interleaved runs against the code before them, the minimum, on a machine loaded to 30–60) cost nothing measurable: the Hurwitz form of `Li_s` 0.81 to 1.05 times as much (less for `s ≤ −1/4`, where the plain remainder took the place of the pole pair), `expint_ei` past 6 and the complex function within 3 %. The final code against the first version, measured in the same way: the complex function's grids 1.2 to 1.3 times as much (left of `σ = −12` below `t = 100` 0.1 to 0.35 times), `polylog` of positive orders 0.62 to 1.0 times, of `s ∈ [−10, −0.1]` on `[−2, 0.9]` 3.8 times (1.2 µs), `expint_ei` on `[0.01, 6]` 1.6 times (12.7 ns) and past 6 1.03, `expint_e1` on `[0.01, 1]` 2.0 times (13 ns), `dilog` on `[−1, 1]` 1.2, the real `zeta` and `hurwitz_zeta` 1.03 to 1.12.

| function | domain | a | libm | statrs | special | puruspe | gsl | boost | cephes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zeta | [1.1,3] | 4.26 | - | - | - | - | 27.89 | 6.88 | 7.53 |
| zeta | [3,56] | 4.96 | - | - | - | - | 24.36 | 8.02 | 12.59 |
| zeta | [-0.5,0.9] | 4.26 | - | - | - | - | 27.03 | 14.43 | 13.58 |
| zeta | [-20,-0.5] | 36.53 | - | - | - | - | 63.73 | 42.2 | 40.38 |
| zeta c64 | s=[0.5,5]+i[-50,50] | 582.69 | - | - | - | - | - | - | - |
| zeta c64 | s=[-10,0.5]+i[-20,20] | 623.68 | - | - | - | - | - | - | - |
| zeta c64 | s=1/2+i[100,1000] | 5181.3 | - | - | - | - | - | - | - |
| hurwitz_zeta | s=[1.5,10],a=[0.1,10] | 92.72 | - | - | - | - | 73.26 | - | 91.1 |
| dilog | [-1,1] | 13.07 | - | - | - | - | 33.7 | - | 10.98 |
| spence | [1,20] | 26.2 | - | - | - | - | 20.68 | - | 675.54 |
| dilog | [-100,-1] | 21.74 | - | - | - | - | 31.01 | - | 10.79 |
| polylog | s=3,x=[-1,1] | 148.12 | - | - | - | - | - | - | 1239.22 |
| polylog | s=2.5,x=[0,1] | 147.22 | - | - | - | - | - | - | - |
| polylog | s=1.5,x=-e^[0,10] | 1581.53 | - | - | - | - | 18.87 | - | - |
| lambert_w0 | [-0.36,3] | 6.27 | - | - | 4.41 | 4.39 | 74.18 | 3.91 | - |
| lambert_w0 | [3,1e10] | 15.11 | - | - | 4.84 | 4.82 | 82.74 | 7.26 | - |
| lambert_wm1 | [-0.36,-0.01] | 11.6 | - | - | 4.84 | 4.79 | 45.49 | 9.86 | - |
| lambert_wm1 | -[1e-300,0.01] | 16.82 | - | - | 7.93 | 7.86 | 14.49 | 17.4 | - |
| expint_e1 | [0.01,1] | 4.07 | - | 21.41 | - | - | 17.61 | 6.12 | 35.86 |
| expint_e1 | [1,50] | 9.18 | - | 28.36 | - | - | 26.81 | 6.3 | 98.87 |
| expint_ei | [0.01,6] | 6.83 | - | - | - | - | 23.16 | 7.24 | 8.8 |
| expint_ei | [6,50] | 8.24 | - | - | - | - | 37.85 | 5.37 | 10.85 |
| expint_ei | [-50,-0.01] | 10.14 | - | - | - | - | 26.82 | 5.75 | 562.97 |
| expint_en n=3 | [0.1,2] | 21.61 | - | 148.19 | - | - | 955.31 | 262.67 | 344.96 |
| expint_en n=3 | [2,20] | 133.82 | - | 45.77 | - | - | 212.2 | 58.96 | 187.05 |
| expint_en n=20 | [0.1,2] | 75.15 | - | 62.24 | - | - | 367.58 | 81.64 | 197.49 |
| expint_en n=20 | [2,20] | 129.79 | - | 43.16 | - | - | 207.27 | 58.23 | 178.32 |

#### Accuracy of the references

The campaign's comparison on its own case sets (max / mean ulps). The column "a" is the first version's: its rows for `zeta`, `hurwitz`, `dilog`, `polylog`, `w0`, `wm1`, `e1`, `ei` and `en` predate the fixes of 2026-11-21 (the 3.326e2 of `polylog`, for one, came from its orders near 0 and its negative integers, both rebuilt since); the bounds above are the ones that hold now.

| function | cases | a | libm | statrs | special | puruspe | gsl | boost | cephes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zeta | 733 | 1.780e0 / 4.691e-1 | – | – | – | – | 2.948e12 / 1.560e11 (42 missed of 733) | 1.463e3 / 1.384e2 (2 missed of 733) | inf / inf (4 missed of 733) |
| hurwitz | 407 | 1.149e0 / 3.638e-1 | – | – | – | – | 4.979e1 / 2.741e0 (1 missed of 407) | – | 4.258e2 / 2.158e1 (4 missed of 407) |
| dilog | 417 | 9.326e-1 / 3.194e-1 | – | – | – | – | 7.528e15 / 4.155e14 (27 missed of 417) | – | – |
| dilog x<=1 | 270 | – | – | – | – | – | – | – | 8.972e15 / 2.516e15 |
| polylog | 708 | 3.326e2 / 1.357e1 | – | – | – | – | – | – | – |
| polylog int s>=-1, x>=-1 | 70 | 1.608e0 / 5.337e-1 | – | – | – | – | – | – | 2.300e0 / 5.968e-1 |
| w0 | 607 | 7.523e-1 / 2.633e-1 | – | – | 3.142e298 / inf (1 missed of 607) | 3.142e298 / inf (1 missed of 607) | 8.954e15 / 3.110e15 | 2.838e7 / 3.792e6 (1 missed of 607) | – |
| wm1 | 604 | 9.740e-1 / 3.372e-1 | – | – | 9.313e6 / 9.582e5 (1 missed of 604) | 9.313e6 / 9.582e5 (1 missed of 604) | 5.555e11 / 2.270e10 (1 missed of 604) | 1.419e7 / 1.494e6 (17 missed of 604) | – |
| e1 | 457 | 1.093e0 / 3.174e-1 | – | 2.588e1 / 3.611e0 | – | – | 2.181e0 / 5.303e-1 (1 missed of 457) | 1.965e0 / 5.511e-1 | 1.099e1 / 1.920e0 (1 missed of 457) |
| ei | 610 | 1.074e0 / 3.176e-1 | – | – | – | – | 3.804e15 / 1.620e14 (6 missed of 610) | 2.880e0 / 5.916e-1 | 9.958e15 / 2.399e15 (3 missed of 610) |
| en | 503 | 7.576e-1 / 3.046e-1 | – | 2.245e1 / 2.522e0 | – | – | 8.326e15 / 9.549e14 (57 missed of 503) | 6.179e1 / 4.101e0 | 6.500e0 / 1.527e0 (13 missed of 503) |

### Where the references are faster

From the campaign of the first version:

- **`polylog`** of orders 1.5 to 2 past `x = −2` (up to 84× GSL's `fermi_dirac_int`, which is itself 10² ulps off there): the Gauss–Legendre pieces of the Fermi–Dirac integral cost about 1.5 µs against its coarser symmetrised quadrature. Negative orders with `x < 0` cost 0.3 to 1.6 µs (the Hurwitz form in double-double, or the series of double-double terms), up to 82 times the first version (the integer orders −8 to −2 on `[−2, 0)`: 1.6 µs against 19 ns, where its Eulerian form in double lost up to 11.5 κ; for larger orders it lost every digit).
- **Lambert W** (2 to 3× special and puruspe, at 3 to 8 ulps against our 1): the branch-point fits and the Halley step of the double-double exponential buy the last ulp.
- **`hurwitz_zeta`** (1.3× GSL 2.8, at 49 ulps): the 24 Bernoulli terms in double-double.
- **`expint_e1` and `expint_ei`** above `x = 1` (1.5 to 1.9× Boost at 4 to 5 ulps): the double-double `e^x` and the rational fits; below 1 `expint_e1` now costs about twice Boost's (the double-double logarithm); **`Eₙ`** for large `n` (about 3× statrs at 0.5 ulps against its 40): the continued fraction evaluated from the bottom.

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

Each polynomial comes from its three-term recurrence (DLMF 18.9.1–2 and the table there) run compensated: the value follows the plain recurrence and a second value carries its error, from the error-free transformations of every step's products and sum, so that the result is rounded about once; the coefficients are exact (Legendre, Hermite, Laguerre, Chebyshev, whose coefficients are integers and `x`), exact for parameters `α`, `β` on the grid `2^−20` (with magnitudes up to `2^28`: their sums with the degree are exact and their products exact double-doubles), or double-doubles. Where the recurrence divides by an integer `c_k` (Legendre, Laguerre, Gegenbauer, associated Legendre and Laguerre) it runs without the divisions, on `Q_k = p_k Π c_j` with the product kept beside it and one division at the end. The values are rescaled by `2^±500` as they grow or shrink, on the size of the value (where the plain recurrence meets an exact zero the value lives in the second word, which is moved into the first), so that only a result beyond the range of `f64` overflows; for `|x| > 2^400`, where a step could overflow, the polynomial is its leading term `cₙxⁿ` to `2^−336`, as a double-double with its exponent apart. `Pₘᵐ = (−1)ᵐ(2m − 1)!!(1 − x²)^(m/2)` is a product of `m` double-double factors with `1 − x²` exact. The Chebyshev polynomials above degree 16 are the binary powers of `T₁ + i sin θ U₀` in double-double, `O(log n)` steps whose errors stay near the double-double precision.

Where a recurrence coefficient vanishes the recurrence decouples, or divides by zero, and the values next to `±1` (or 0) are small differences of large terms; those parameters go by identities that keep the relative accuracy, as double-doubles rounded once: `Lₙ^(−l)(x) = (−x)^l (n − l)!/n! L_(n−l)^(l)(x)` (Szegő (5.2.1)); `Cₙ^(−l) ≡ 0` above the degree `2l` and `Cₙ^(½−l)(x) = 4^l ((½)_l)² (n − 2l)!/n! (1 − x²)^l C_(n−2l)^(l+½)(x)` for `n ≥ 2l` (the reflection `λ → 1 − λ` of the differential equation); `Pₙ^(−l,β)(x) = [C(n + β, l)/C(n, l)] ((x − 1)/2)^l P_(n−l)^(l,β)(x)` (Szegő (4.22.2)) and `Pₙ^(α,β)(x) = (−1)ⁿ Pₙ^(β,α)(−x)`; and for `α + β` within ½ of `−l` with `5 ≤ l ≤ 30`, within 1/64 of it with `l = 2…4` and from the degree 16 on within 1/16 (and beyond `l = 30` within `2^−40 l`), the explicit sum `Σ d_j ((x − 1)/2)ʲ` of DLMF 18.5.7 in double-double up to the degree `l` and the recurrence from `(P_(l−1), P_l)`. For `|x| > 2^400` with `n < l ≤ 2n`, whose leading coefficient has the factor `δ = α + β + l` (zero, or small), the polynomial is the two largest terms `d_t yᵗ + d_n yⁿ` of that sum, `t = l − n − 1`, `y = (x − 1)/2`, with their exponents apart.

The recurrences take `O(n)` steps for the degree `n`: the asymptotic forms for large degrees (Legendre's by Stieltjes's and Bessel-type expansions, Hermite's by Plancherel–Rotach) are left to the release optimisation pass.

### Measured

The bounds above were measured against mpmath at 256 bits and more (the reference data, `tests/math/special/data/elliptic_*.txt`, made by `data/gen_elliptic.py`, whose last cases in each file are the arguments of the audit's defects and their neighbours), and on sweeps of the same functions with fresh points (two seeds each): for every function and region, uniform and logarithmic points, the ends of the intervals, the last doubles before the poles and edges, huge and tiny arguments; the references there were cross-checked by a second method (quadrature, theta functions, the AGM, closed forms, exact rational sums of the polynomials). To reproduce the data: `python3 gen_elliptic.py` in `tests/math/special/data`, then `cargo test --test math -- special::elliptic` (which prints the largest error of each function and region).

## Measured: elliptic functions and orthogonal polynomials

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation only, never timed. These timings predate the fixes of 2026-11-21, which went on the rare paths but for the logarithmic forms of `K` and `E` and a few double-double steps of the Jacobi reduction and the Carlson scaling, and those of the second review of them, after which the Jacobi functions take about 1.15–1.35 times as long as before it (the double-double Landen ascent) and `ellip_f(φ, 1)` about 1.4 times.

| function | domain | a | boost | gsl | cephes |
| --- | ---: | ---: | ---: | ---: | ---: |
| ellip_k | m in [0,0.8) | 2.36 | 4.57 | 33.17 | 11.06 |
| ellip_e | m in [0,0.8) | 2.4 | 6.86 | 69.84 | 10.31 |
| ellip_k | m in [0.8,1) | 7.83 | 7.9 | 48.04 | 9.34 |
| ellip_e | m in [0.8,1) | 7.64 | 7.58 | 89.19 | 7.73 |
| ellip_k | m in [-10,0) | 10.78 | - | - | - |
| ellip_e | m in [-10,0) | 7.74 | - | - | - |
| ellip_pi | n in (-5,0.9), m in [0,0.99) | 76.72 | 90.36 | 117.61 | - |
| ellip_pi | n in (1.1,10), m in [0,0.99) | 70.07 | 3.6 | - | - |
| ellip_f | phi in [-pi/2,pi/2], m in [0,0.99) | 31.95 | 38.04 | 33.92 | 36.78 |
| ellip_e_inc | phi in [-pi/2,pi/2], m in [0,0.99) | 64.64 | 69.87 | 72.13 | 86.02 |
| ellip_f | phi in [-20,20], m in [0,0.99) | 30.84 | 38.94 | 68.46 | 50.2 |
| ellip_e_inc | phi in [-20,20], m in [0,0.99) | 61.82 | 69.72 | 145.41 | 87.12 |
| ellip_pi_inc | n in (-5,0.9), \|phi\| < 1.5, m in [0,0.99) | 79.84 | 123.99 | 96.52 | - |
| carlson_rf | args in [0.01,10] | 19.84 | 8.49 | 25.97 | - |
| carlson_rd | args in [0.01,10] | 36.66 | 10.91 | 34.06 | - |
| carlson_rj | args in [0.01,10] | 56.11 | 31.55 | 56.48 | - |
| carlson_rc | args in [0.01,10] | 7.28 | 6.78 | 15.22 | - |
| jacobi_elliptic (sn+cn+dn) | u in [-4,4], m in [0,0.99) | 48.26 | 124.6 | 27.95 | 90.12 |
| jacobi_elliptic (sn+cn+dn) | u in [-100,100], m in [0,0.99) | 88.12 | 125.4 | 28.43 | 91.52 |
| legendre_p | n = 4, x in [-1,1] | 8.76 | 3.06 | 3.97 | - |
| chebyshev_t | n = 4, x in [-1,1] | 4.34 | 1.99 | - | - |
| chebyshev_u | n = 4, x in [-1,1] | 4.34 | 1.99 | - | - |
| assoc_legendre_p (m = n/2) | n = 4, x in [-1,1] | 15.1 | 9.75 | 9.03 | - |
| jacobi_p (a = 0.5, b = 1.5) | n = 4, x in [-1,1] | 26.55 | 5.92 | - | - |
| gegenbauer_c (a = 1.5) | n = 4, x in [-1,1] | 9.93 | 3.08 | 3.95 | - |
| hermite_h | n = 4, \|x\| < sqrt(2n+1) | 4.6 | 2 | 4.04 | - |
| hermite_he | n = 4, \|x\| < sqrt(2n+1) | 4.65 | - | 3.78 | - |
| laguerre_l | n = 4, x in [0, 4n+2] | 9.85 | 2.37 | 35.91 | - |
| assoc_laguerre_l (alpha = 2) | n = 4, x in [0, 4n+2] | 11.31 | 2.68 | 38.34 | - |
| legendre_p | n = 20, x in [-1,1] | 50.93 | 16.98 | 30.33 | - |
| chebyshev_t | n = 20, x in [-1,1] | 42.66 | 6.55 | - | - |
| chebyshev_u | n = 20, x in [-1,1] | 47.81 | 6.2 | - | - |
| assoc_legendre_p (m = n/2) | n = 20, x in [-1,1] | 87.05 | 19.03 | 21.12 | - |
| jacobi_p (a = 0.5, b = 1.5) | n = 20, x in [-1,1] | 134.91 | 32.56 | - | - |
| gegenbauer_c (a = 1.5) | n = 20, x in [-1,1] | 53.69 | 17.51 | 21.75 | - |
| hermite_h | n = 20, \|x\| < sqrt(2n+1) | 29.1 | 6.4 | 17.1 | - |
| hermite_he | n = 20, \|x\| < sqrt(2n+1) | 29.04 | - | 16.49 | - |
| laguerre_l | n = 20, x in [0, 4n+2] | 54.81 | 16.03 | 24.84 | - |
| assoc_laguerre_l (alpha = 2) | n = 20, x in [0, 4n+2] | 58.54 | 18.63 | 25.07 | - |
| legendre_p | n = 100, x in [-1,1] | 266.2 | 174.98 | 294.38 | - |
| chebyshev_t | n = 100, x in [-1,1] | 78.85 | 44.52 | - | - |
| chebyshev_u | n = 100, x in [-1,1] | 83.77 | 44.34 | - | - |
| assoc_legendre_p (m = n/2) | n = 100, x in [-1,1] | 412.91 | 95.59 | 113.64 | - |
| jacobi_p (a = 0.5, b = 1.5) | n = 100, x in [-1,1] | 691.26 | 212.91 | - | - |
| gegenbauer_c (a = 1.5) | n = 100, x in [-1,1] | 274.12 | 93.5 | 185.09 | - |
| hermite_h | n = 100, \|x\| < sqrt(2n+1) | 157.93 | 48.08 | 106.41 | - |
| hermite_he | n = 100, \|x\| < sqrt(2n+1) | 156 | - | 90.21 | - |
| laguerre_l | n = 100, x in [0, 4n+2] | 290.55 | 168.39 | 222.43 | - |
| assoc_laguerre_l (alpha = 2) | n = 100, x in [0, 4n+2] | 300.19 | 171.13 | 222.47 | - |
| legendre_p | n = 1000, x in [-1,1] | 2713.55 | 2987.96 | 3706.61 | - |
| chebyshev_t | n = 1000, x in [-1,1] | 142.52 | 699.09 | - | - |
| chebyshev_u | n = 1000, x in [-1,1] | 147.12 | 703.45 | - | - |

The references' accuracy on the same data, max / rms ulps (and the number of cases in the reference's domain; Boost and GSL take the modulus `k = √m`, which loses `m₁` as `m → 1`; Cephes takes `m₁`); the rows with a domain are the references' common one. This is the record of the campaign, on the data as it was then (the first cases of each file) and with the code before the fixes of 2026-11-21: plain ulps of the result for the elliptic functions over every case, so that the largest errors are those next to a pole of `ellip_pi_inc` (where its bound is in units of `1 + κ`), next to a zero of the principal value of `carlson_rj` (whose bound is relative to `R_J(x, y, z, −p)`) and of `dn` as `m → 1`, and the scaled ulps of the bounds above for the polynomials; the bounds table above is the one the code is held to (its `ellip_k`, for instance, is now within 0.58 ulps everywhere).

| function | nettrash-a | Boost 1.92 | GSL 2.8 | Cephes |
|---|---|---|---|---|
| `ellip_k` | 0.822 / 0.297 (347) | 1.22e13 / 1.15e12 (196) | 1.22e13 / 1.15e12 (196) | 1.125 / 0.436 (196) |
| `ellip_e` | 0.538 / 0.298 (347) | 7.641 / 1.494 (189) | 6427.948 / 490.309 (189) | 0.957 / 0.405 (189) |
| `ellip_pi` | 2.268 / 0.645 (350) | 5.42e11 / 4.31e10 (182) (44 wrong NaN/inf) | 5.42e11 / 4.95e10 (138) | - |
| `ellip_f` | 1.839 / 0.501 (350) | 6.87e10 / 5.26e9 (175) | 6.87e10 / 5.26e9 (175) | 181.461 / 18.77 (175) |
| `ellip_e_inc` | 2.128 / 0.614 (350) | 9.18e5 / 7.28e4 (175) | 1.11e4 / 885.985 (175) | 6.374 / 1.43 (175) |
| `ellip_pi_inc` | 742.204 / 39.131 (381) | 7.89e15 / 1.42e15 (187) (36 wrong NaN/inf) | 1.60e11 / 1.65e10 (114) | - |
| `carlson_rf` | 0.5 / 0.302 (245) | 2.647 / 0.832 (245) | 3.344 / 1.062 (245) | - |
| `carlson_rd` | 0.498 / 0.28 (234) | 43.659 / 3.185 (234) | 3.297 / 1.168 (234) (16 wrong NaN/inf) | - |
| `carlson_rj` | 1457.527 / 81.995 (316) | 7.44e15 / 7.62e14 (316) (5 wrong NaN/inf) | 4.658 / 1.152 (237) (39 wrong NaN/inf) | - |
| `carlson_rc` | 1.524 / 0.468 (237) | 2.37 / 0.62 (237) | 2.889 / 0.724 (237) (40 wrong NaN/inf) | - |
| `jacobi_elliptic sn` | 3.306 / 0.573 (420) | 9.01e15 / 6.13e14 (216) | 8.95e15 / 2.41e15 (420) | 7.91e15 / 5.38e14 (216) (6 wrong NaN/inf) |
| `jacobi_elliptic sn (m in [0, 0.99])` | 1.533 / 0.391 (149) | 2.91e11 / 2.39e10 (149) | 8.50e6 / 9.75e5 (149) | 4.49e6 / 4.73e5 (149) |
| `jacobi_elliptic cn` | 2.877 / 0.672 (420) | 2.65e14 / 1.80e13 (216) | 9.01e15 / 4.75e15 (420) | 2.48e55 / 1.69e54 (216) (6 wrong NaN/inf) |
| `jacobi_elliptic cn (m in [0, 0.99])` | 1.338 / 0.341 (149) | 5.27e9 / 4.32e8 (149) | 1.55e7 / 1.34e6 (149) | 2.39e6 / 3.15e5 (149) |
| `jacobi_elliptic dn` | 40.263 / 4.115 (420) | 5.49e15 / 3.76e14 (216) | 8.55e15 / 3.20e15 (420) | 4.96e55 / 3.38e54 (216) (6 wrong NaN/inf) |
| `jacobi_elliptic dn (m in [0, 0.99])` | 3.678 / 0.968 (149) | 4.80e7 / 3.94e6 (149) | 1.47e6 / 1.25e5 (149) | 2.07e6 / 1.77e5 (149) |
| `ellip_k (m in [0, 0.99])` | 0.548 / 0.291 (153) | 7.701 / 1.327 (153) | 7.478 / 1.384 (153) | 1.104 / 0.411 (153) |
| `ellip_e (m in [0, 0.99])` | 0.538 / 0.305 (148) | 1.637 / 0.735 (148) | 6.408 / 1.613 (148) | 0.917 / 0.411 (148) |
| `ellip_pi (n < 1, m in [0, 0.99])` | 1.261 / 0.474 (54) | 7.427 / 2.036 (54) | 2.23e5 / 3.04e4 (54) | - |
| `ellip_f (|φ| ≤ π/2, m in [0, 0.99])` | 1.124 / 0.546 (53) | 2.015 / 0.898 (53) | 2606.022 / 417.778 (53) | 2.626 / 1.032 (53) |
| `ellip_e_inc (|φ| ≤ π/2, m in [0, 0.99])` | 1.274 / 0.558 (53) | 2.715 / 0.877 (53) | 2605.939 / 391.579 (53) | 6.374 / 2.124 (53) |
| `ellip_pi_inc (n < 1, |φ| ≤ π/2, m in [0, 0.99])` | 1.406 / 0.571 (33) | 1.914 / 0.759 (33) | 11.388 / 3.181 (33) | - |
| `carlson_rj (p > 0)` | 1.334 / 0.452 (237) | 6.58e15 / 7.36e14 (237) (4 wrong NaN/inf) | 4.658 / 1.152 (237) (39 wrong NaN/inf) | - |
| `legendre_p` | 0.459 / 0.112 (120) | 1.567 / 0.313 (120) (24 wrong NaN/inf) | 1.567 / 0.35 (96) | - |
| `assoc_legendre_p` | 0.494 / 0.118 (120) | 13.965 / 3.138 (120) (2 wrong NaN/inf) | 41.37 / 8.965 (99) (1 wrong NaN/inf) | - |
| `hermite_h` | 0.497 / 0.168 (120) | 10.15 / 1.926 (120) (21 wrong NaN/inf) | 10.15 / 1.926 (120) (21 wrong NaN/inf) | - |
| `hermite_he` | 0.471 / 0.161 (120) | - | 7.529 / 1.803 (120) (16 wrong NaN/inf) | - |
| `laguerre_l` | 0.442 / 0.115 (120) | 529.353 / 67.737 (120) (2 wrong NaN/inf) | 978.999 / 119.938 (120) (2 wrong NaN/inf) | - |
| `assoc_laguerre_l` | 0.5 / 0.117 (120) | 229.164 / 57.305 (16) | 8.01e102 / 7.31e101 (120) (2 wrong NaN/inf) | - |
| `chebyshev_t` | 0.445 / 0.146 (120) | 325.155 / 31.463 (120) | - | - |
| `chebyshev_u` | 0.5 / 0.127 (120) | 364.707 / 41.843 (120) | - | - |
| `jacobi_p` | 0.496 / 0.135 (120) | 861.254 / 115.489 (120) | - | - |
| `gegenbauer_c` | 0.478 / 0.167 (120) | 96.164 / 10.66 (86) | 57.976 / 9.353 (120) (35 wrong NaN/inf) | - |

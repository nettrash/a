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

`gamma`, `lgamma`, `lgamma_sign`, `digamma`, `polygamma`, `beta`, `lbeta`, `gamma_p`, `gamma_q`, `gamma_p_inv`, `gamma_q_inv`, `beta_inc`, `beta_inc_c`, `beta_inc_inv`, `beta_inc_c_inv`, `factorial`, `binomial` and `pochhammer` take `f32` or `f64` arguments; `gamma` and `lgamma` also `Complex<f32>` and `Complex<f64>`. An `f32` argument is computed in `f64` and rounded once.

### Definitions and domains

| Function | Value | Domain and special values |
| --- | --- | --- |
| `gamma(x)` | `Γ(x) = ∫₀^∞ t^(x−1) e^(−t) dt`, continued | `±∞` at `±0`, NaN at the negative integers and `−∞`; `+∞` above 171.6243769563027; zero (with the sign of `Γ`) below about −184 |
| `lgamma(x)` | `ln \|Γ(x)\|` | `+∞` at the poles, at `±0` and at `±∞`; overflows above 2.556·10³⁰⁵ |
| `lgamma_sign(x)` | `(ln \|Γ(x)\|, sign of Γ(x))` | the sign is 1 or −1 (1 at the poles, −1 at `−0`, NaN for NaN) |
| `gamma(z)`, complex | `Γ(z)` | on the real axis the real `Γ`; `0` for an infinite `Im z`; each part overflows on its own |
| `lgamma(z)`, complex | the principal branch of `ln Γ(z)` (SciPy's `loggamma`), analytic off the cut `(−∞, 0]` | on the cut the limit from the side of the sign of the zero, `−π⌈−x⌉` from above (mpmath's value at a real `x`); `(+∞, NaN)` at the non-positive integers |
| `digamma(x)` | `ψ(x) = Γ′(x)/Γ(x)` | `∓∞` at `±0`, NaN at the negative integers and `−∞` |
| `polygamma(n, x)` | `ψ⁽ⁿ⁾(x)`, `n = 0` is `ψ` | for `n ≥ 1`: `(−1)ⁿ⁺¹∞` at `+0`, `+∞` at `−0`, NaN at the negative integers; overflow to `±∞` and underflow to `±0` at the right points |
| `beta(a, b)` | `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` | any real `a, b` (the stated bound for `a, b > 0`): `±∞` at `a = ±0`, zero where only `a + b` is a pole, NaN at the other poles; `0` for an infinite argument with the other positive |
| `lbeta(a, b)` | `ln \|B(a, b)\|` | `+∞` at the poles of `Γ(a)` or `Γ(b)`, `−∞` where only `a + b` is a pole |
| `gamma_p(a, x)`, `gamma_q(a, x)` | `P = γ(a, x)/Γ(a)`, `Q = Γ(a, x)/Γ(a) = 1 − P` | `a > 0`, `x ≥ 0`, else NaN (also `a = ∞`); `(P, Q) = (0, 1)` at `x = 0`, `(1, 0)` at `x = ∞` |
| `gamma_p_inv(a, p)`, `gamma_q_inv(a, q)` | the `x` with `P(a, x) = p`, `Q(a, x) = q` | `p = 0` gives 0, `p = 1` gives `∞` (`q` mirrored); a root below the least subnormal gives 0; NaN outside `[0, 1]` |
| `beta_inc(a, b, x)`, `beta_inc_c(a, b, x)` | `I_x(a, b) = B(x; a, b)/B(a, b)` and `1 − I_x(a, b)` | `a, b > 0` finite, `0 ≤ x ≤ 1`, else NaN; `I = 0` at `x = 0`, 1 at `x = 1` |
| `beta_inc_inv(a, b, p)`, `beta_inc_c_inv(a, b, q)` | the `x ∈ [0, 1]` with `I_x(a, b) = p`, `1 − I_x(a, b) = q` | 0 or 1 at the ends |
| `factorial(x)` | `x! = Γ(x + 1)` | NaN at the negative integers; exact (correctly rounded) at the integers 0 to 170 |
| `binomial(n, k)` | `Γ(n + 1)/(Γ(k + 1) Γ(n − k + 1))` | 0 for a negative integer `k` and for integers `0 ≤ n < k`; `(−1)^k C(k − n − 1, k)` for a negative integer `n`; NaN where only the numerator has a pole |
| `pochhammer(x, n)` | `(x)ₙ = Γ(x + n)/Γ(x)` | the product `x(x + 1)…(x + n − 1)` for an integer `n` (its reciprocal for a negative one); 1 at `n = 0`; zero where only `x` is a pole, NaN where only `x + n` is |

### Error bounds

Measured against mpmath at 256 bits (and continued fractions at 320 bits for `P`, `Q` and `I` where mpmath's own do not converge): the bound each test enforces on the data of `tests/math/special/data/gamma_*.txt` and, in brackets, the largest error over the area's larger random survey (counts given). In ulps of the result's type unless stated.

| Function | Region | `f64` bound [survey] | `f32` |
| --- | --- | --- | --- |
| `gamma` | normal results | 0.6 [0.54, 13,013 cases] | 0.501 [0.500] |
| | subnormal results (rounded twice) | 1.0 | |
| `lgamma` | everywhere but beside the zeros on the negative axis | 0.6 [0.585, 12,613] | 0.501 [0.500] |
| | beside the zeros on the negative axis | absolute 0.1 ε [0.06 ε, 723] | |
| `gamma`, complex | `\|Γ(z)\| > 10⁻³⁰⁰`, relative to `\|Γ(z)\|` | 1.25 ε [0.98 ε, 3,635] | |
| `lgamma`, complex | relative to `\|ln Γ(z)\|` | 1.25 ε [0.98 ε, 3,300] | |
| `digamma` | but beside the zeros on the negative axis | 0.75 [0.56, 2,903] | 0.501 [0.500] |
| | beside the zeros on the negative axis | absolute 0.25 ε [0.06 ε, 780] | |
| `polygamma` | `x > 0`, `n` from 1 to 500 | 0.75 [0.52, 2,944; trigamma 0.65, 9,300] | |
| | `x < 0`, relative to `n!(r^(−n−1) + p^(−n−1))` (`r`, `p` the distances to the integers beside `x`) | 0.75 [0.51, 2,041] | |
| `beta` | `a, b > 0` | 1.0 [0.53, 2,166] | 0.501 [0.500] |
| | `a` or `b` negative | 1.0 [0.61, 485] | |
| `lbeta` | `\|ln B\| ≥ 1/2`; absolute in units of `ε/2` below | 1.0 [0.50; the absolute error below ½ is at most 0.03 ε] | |
| `gamma_p`, `gamma_q` | `a < 20`, and `a ≥ 20` by series or fraction | 1.2 [0.66, 5,834] | 0.501 [0.500] |
| | `a ≥ 20` by Temme's expansion | 1.2 [1.17] | |
| `gamma_p_inv`, `gamma_q_inv` | | 0.6 [0.50, 1,052] | 0.501 [0.500] |
| `beta_inc`, `beta_inc_c` | `min(a, b) ≤ 1 < max(a, b)` | 1.5 [1.31, 8,900] | 0.501 [0.500, 6,400] |
| | elsewhere | 1.5 [1.04] | |
| `beta_inc_inv`, `beta_inc_c_inv` | in units of `max(1, κ)` ulps, `κ = \|p/(x I′(x))\|` the condition number | 1.5 [0.48 / 0.80, 1,124 / 1,091] | 0.6 [0.48] |
| `factorial` | | 0.6 [0.51, 1,645] | |
| `binomial` | | 0.6 [0.51, 1,696] | |
| `pochhammer` | | 0.6 [0.50, 1,520] | |

### Method

- **Fused multiply-adds.** Where the hardware has them (aarch64, x86_64 with `fma`; not under Miri) the error-free steps are fused multiply-adds; elsewhere — x86_64's default target, as crates.io builds it — Dekker's product of Veltkamp's halves, and plain `a·b + c` for the terms whose rounding does not matter (the crate's `FMA` switch, as in `dd.rs`). The bounds hold on both paths (tested on x86_64 in Docker and under Miri).
- **The elementary functions.** The area computes its own `ln`, `exp`, `sin(πx)` and `cos(πx)` in double-double (about 2⁻⁹⁰ relative for `ln` and `exp`, 2⁻⁶⁰ for the circular functions), so that none of its real functions calls a `std` transcendental function: the results are the same on every target and under Miri. `ln x` reduces `x` to `[0.7057, 1.4114)` and subtracts `ln c` of 64 tabulated 7-bit `c` (so that `x/c − 1` is exact by one fused multiply-add), then `ln(1 + r) = r − r²/2 + r³P(r)` with the first terms exact; `e^x` takes `x − k·ln2/64` with a 36-bit `ln 2/64` and `2^(j/64)` from a table; `sin(πr)`, `cos(πr)` on `|r| ≤ 1/4` are minimax polynomials with their two leading terms in double-double. Every polynomial is evaluated by Estrin's scheme, fully unrolled. The minimax fits are the area's own, by a Remez exchange in mpmath: `gamma/gen_tab.py` (with `gen_remez.py`) writes `gamma/tab.rs`, and `gen_igamma_tab.py` writes `igamma_tab.rs`, each byte for byte after `rustfmt`.
- **`Γ(x)`.** For `|x| < 8`, `x = n + t` with `|t| ≤ 1/2`, `t = x − n` exact: `1/Γ(1 + t) = 1 + γt + c₂t² + t²R(t)` (`R` of degree 14, minimax relative), with the first three terms in double-double, then the product of the exact factors `t + k` up to `n − 1` (or down to `n`), compensated, and one double-double division. From 8 on, Stirling's series: `(x − ½) ln x − x + ln √(2π) + S(x)` with `ln x` in double-double and `S` a minimax polynomial in `1/x²`, then the double-double exponential. Below −8, `Γ(x) = −π/(x sin(πx) Γ(−x))`, with `sin(πx)` and the exponential in double-double. `factorial` is the same with `x + 1` never rounded, and the table of `n!` at the integers.
- **`ln |Γ(x)|`.** The result below 8 comes from its own minimax pieces: `ln Γ(1 + t) = t·H(t)` on `−1/2 ≤ t < 1/2` (five pieces, the zero at 1 a factor), `ln Γ(2 + t) = t(1 + t)·F(t)` on `−1/2 ≤ t < 1` (three pieces, the zeros at 1 and 2 exact factors), `ln Γ` on the units of `[3, 8)`, each `Y + c₁s + s²R(s)` with the first two terms without error (and `c₁` fixed to its double in the fit); below ½, `ln Γ(1 + x) − ln x`. The other functions use a version in double-double to about 2⁻⁶⁰: below 1.5 `ln Γ(x) = ln Γ(x + 1) − ln x` (and `− ln(x(1 + x))` below ½, one logarithm of the double-double `x + x²`), between 3 and 8 the product of the exact factors and one logarithm, Stirling's series from 8 on (one double-double `ln` and the sums in double-double), `x(ln x − 1) − (ln x)/2` rounded once beyond 2⁵², and the reflection formula below −½.
- **Complex `Γ` and `ln Γ`.** Complex double-double: Stirling's series for `|z| ≥ 10` with `Re z ≥ 0` (its main terms `(z − ½) ln z − z` in double-double, `ln z` from the double-double `ln |z|` and an argument corrected from `atan2` by one Newton step with the double-double sine and cosine), the shift `ln Γ(z) = ln Γ(z + n) − ln Π(z + k)` below it with the product in double-double and the branch kept by counting its crossings of the negative axis (Hare's method, as in SciPy), Taylor series at the zeros 1 and 2, and for `Re z < 0` `ln Γ(z) = ln 2π − iπ/2 + iπz − ln(1 − e^{2πiz}) − ln Γ(1 − z)` for `Im z > 0`, the branch that is analytic in the upper half-plane (the lower by conjugation). `Γ(z) = e^{ln Γ(z)}` from the double-double logarithm: its exponential, and the sine and cosine of its imaginary part reduced by `π` in double-double, so the error stays near ε even where `|ln Γ|` is in the hundreds.
- **`ψ`.** `ln x + (its low part − 1/(2x) − u·T(u))`, rounded once, from 8 on (`T` minimax in `u = 1/x²`); eight minimax pieces `Y + c₁s + s²R(s)` on `[2, 8)`; `(x − x₀)·Q(x)` in four pieces on `[1, 2]` with the positive zero `x₀ = 1.4616321449683622…` in double-double; `ψ(1 + x) − 1/x` below 1; and `ψ(−u) = ψ(u) + 1/u + π cot(πu)` with `ψ(u) + 1/u = ψ(1 + u)` for `u < 1`, `cot` from one reduction for both the sine and the cosine in double-double and one division.
- **`ψ⁽ⁿ⁾`.** `ψ′` by its own method: `1/x + (its rounding error + 1/(2x²) + T(u)/x³)`, rounded once, from 10 on, ten minimax pieces on `[2, 10)`, and `Σ 1/(x + j)² + ψ′(x + m)` below 2. Otherwise `(−1)ⁿ⁺¹ n! ζ(n + 1, x)`, the Hurwitz zeta function by its terms up to where its Euler–Maclaurin tail converges to 2⁻⁶⁰ (from `y ≥ 0.38(n + 21) + 0.5`, with up to 40 Bernoulli terms, the first three in double-double), or until the rest is negligible; for `x < 0` the sum of all the terms `(x + j)^(−n−1)` as `(−1)^(n+1)[ζ(n + 1, r) − ζ(n + 1, 1 − x)] + ζ(n + 1, p)` of the exact distances `r` and `p` of `x` to the integers beside it. Large orders and arguments carry their values as `v·e^l`, so nothing overflows before the last product.
- **`B` and `ln B`.** For `a, b > 0`: below 8, `B = R(a + b)/(R(a)R(b))` with `R = 1/Γ` in double-double (the low part of `a + b` through `ψ`) and `ln B` its logarithm (above 10⁻¹⁰⁰; `ln Γ(a) + ln Γ(b) − ln Γ(a + b)` in double-double below that); `ln Γ(a) + [ln Γ(b) − ln Γ(a + b)]` by the Stirling difference when only the larger is, and for both from 8 on `ln √(2π) + (a − ½) ln(a/c) + (b − ½) ln(b/c) − ½ ln c + S(a) + S(b) − S(c)` with `ln(b/c) = −ln(1 + a/b)` computed directly (an `atanh` series for small ratios), so no term cancels another; `B` is its double-double exponential. Other arguments go through the signs and logarithms of the three gamma functions.
- **`binomial`, `pochhammer`.** The table of `n!` in double-double for integers to 170; compensated products for an integer `k` or `n` up to 32 (every factor `n − i` or `x + i` exact as a double-double, its low part and each product's rounding error carried); else `1/((n + 1) B(n − k + 1, k + 1))` with the rounding of `n − k + 1` corrected through `ψ`, and `(x)ₙ` by the Stirling difference.
- **`P` and `Q`.** The structure of DiDonato and Morris (1986) and Boost: the smaller of `P` and `Q`, and the other as its double-double complement. The prefix `x^a e^(−x)/Γ(a + 1)` in double-double, `e^(a ln(x/a) + a − x − S(a))/√(2πa)` from `a = 8`. The series of `P` below about `x = a`, Legendre's continued fraction for `Q` above (evaluated from the bottom as one composed map per four steps, so one division per four steps, its length set in advance, the last steps in double-double), finite sums for integer and half-integer `a` below 20 when `x ≥ a` (`(n − 1)!` from the table, `erfcx` for the half-integers), the small-`a` form `Q = (Γ(1 + a) − 1 − (x^a − 1))/…` without cancellation, and for `a ≥ 20` with `|x − a|/a` below about ½ Temme's uniform expansion with 13 coefficients `c_k(z)` derived by series arithmetic after Temme (1979) and fitted minimax, and the area's double-double `erfcx` (eight pieces below 4, then `(1/t)·g(1/t²)`; `beta_inc`'s BASYM uses the same).
- **Their inverses.** A first guess (`(pΓ(a + 1))^(1/a)` for small `x`; for `a ≥ 1` Temme's expansion inverted to first order, as DiDonato and Morris, with Acklam's normal quantile; an iteration on the tail for large `x`), then Halley's method on the logarithm of the smaller tail in `ln x` with the slope from `ln Γ(a)` computed once, kept in a bracket: about two evaluations of `P` or `Q` a call.
- **`I_x(a, b)`.** TOMS 708 (DiDonato and Morris 1992) as structured in Boost's `ibeta_imp`, the side computed always the one that does not cancel: the power series (three terms in double-double), the continued fraction in its division-free form (its depth by a forward recurrence of the convergents with rescaling, then a projective backward pass, two steps and the coefficients in double-double, one division at the end), the `a`-step with BGRAT for a small `b` beside a large `a`, and BASYM for both above 100 near the mean. The power terms `x^a y^b/B(a, b)` in double-double with `x` and `y = 1 − x` exact: below 8 with `1/B = R(a)R(b)/R(c)`, else as `√(ab/(2π(a + b)))·e^(aφ(r) + bφ(s) + S(c) − S(a) − S(b))` with `φ(r) = ln(1 + r) − r` and `ar + bs = 0`, so the exponent never cancels.
- **Their inverses.** A first guess from the normal approximation with the Cornish–Fisher skewness term where it is small (for `a, b ≥ 2`), the leading tail term or the mean, then Halley's method on the logarithm of the tail in the log of the smaller of `x` and `y` (the step through `expm1`), with a bisection bracket: 2–3 evaluations a call. For `max(a, b) ≥ 2⁹⁹⁵`, the point mass at the mean or the incomplete gamma limit.

### Measured

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation only, never timed.

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

The fastest reference is often far less accurate (the next table); Boost's column is there for comparison.

#### Accuracy of the references

Max/rms ulps on the area's data files (the normal range; the references' own failures in brackets):

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

Against the fastest reasonably accurate reference (Boost, or libm for `ln Γ`) in the same run:

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
| `erfcx(x)` | `e^(x²)·erfc(x)` | `0` at `+∞`; `+∞` from `x < −26.6287`, where `2e^(x²)` overflows |
| `erf_inv(x)` | the `y` with `erf(y) = x` | `[−1, 1]`, `±∞` at `±1`, NaN outside |
| `erfc_inv(x)` | the `y` with `erfc(y) = x` | `[0, 2]`, `+∞` at 0, `−∞` at 2, NaN outside; every subnormal `x` |
| `dawson(x)` | `F(x) = e^(−x²) ∫₀^x e^(t²) dt` | odd; `±0` at `±∞`, `F(x) ≈ 1/(2x)` |
| `fresnel(x)` | `(S(x), C(x))`, `S = ∫₀^x sin(πt²/2) dt`, `C = ∫₀^x cos(πt²/2) dt` | odd; `(±1/2, ±1/2)` at `±∞` and from `|x| ≥ 2⁵⁴` |
| `faddeeva(z)` | `w(z) = e^(−z²)·erfc(−iz)` | `0` as `|z| → ∞` in the closed upper half-plane; NaN for a NaN or `−i∞` |
| `erf(z)`, `erfc(z)`, `erfcx(z)` | as above, `erfcx(z) = w(iz)` | entire; `erf(±∞ + iy) = ±1` for a finite `y` |

A NaN argument gives NaN. The complex functions have `erf(conj z) = conj erf(z)`, `erf(−z) = −erf(z)`, `w(−conj z) = conj w(z)` exactly, and on the axes the real functions: `erf(x + 0i) = erf(x)`, `w(iy) = erfcx(y)`, `w(x) = e^(−x²) + 2i/√π·F(x)`.

### Accuracy

Measured on 10⁴ to 3·10⁴ random arguments a region against mpmath at 256 bits (`f64`), 2,500 to 7,500 (`f32`), and in `tests/math/special/erf.rs` on the data of `data/gen_erf.py`; the largest error in ulps, then the rms:

| Function | Region | `f64` max | rms | `f32` max |
| --- | --- | ---: | ---: | ---: |
| `erf` | `|x| < 1/2` | 0.71 | 0.30 | 0.50 |
| | `1/2 ≤ |x| < 6` | 0.51 | 0.29 | 0.50 |
| `erfc` | `|x| < 1/2` | 0.76 | 0.29 | 0.50 |
| | `−6 < x ≤ −1/2`, `1/2 ≤ x < 8` | 0.57 | 0.29 | 0.50 |
| | `8 ≤ x < 27.3` (subnormal from 26.55: 0.5001) | 0.51 | 0.28 | 0.50 |
| `erfcx` | `|x| < 1/2` | 0.63 | 0.29 | 0.50 |
| | `x ≤ −1/2`, `1/2 ≤ x` | 0.56 | 0.29 | 0.50 |
| `erf_inv` | `[−1, 1]`, near ±1 to `1 − 2⁻⁵³`, subnormal | 0.76 | 0.30 | 0.50 |
| `erfc_inv` | `(0, 2]`, `[2⁻¹⁰⁷⁴, 1/2]` | 0.75 | 0.30 | 0.50 |
| `dawson` | `|x| < 1/2` | 0.75 | 0.29 | 0.50 |
| | `1/2 ≤ |x|` | 0.70 | 0.29 | 0.50 |
| `fresnel` S, C | `|x| < 1` | 0.82, 0.95 | 0.29 | 0.50 |
| | `1 ≤ |x| < 8` | 0.74, 0.80 | 0.29 | 0.50 |
| | `8 ≤ |x|` | 0.58, 0.58 | 0.29 | 0.50 |

The `f32` forms are the `f64` values rounded once: correctly rounded but for the halfway cases of the `f64` error, about one argument in 2²⁹. Below about 2⁻⁹⁷⁰ the test data's low word of a reference is rounded to the least subnormal, so the tests allow 1 ulp there: `erfc` from 26.55, whose results are within 0.5001 ulp.

The complex functions, the error relative to `|w|` in units of ε = 2⁻⁵²:

| Function | Region | max | rms |
| --- | --- | ---: | ---: |
| `faddeeva` | upper half-plane, `|z| < 10` | 3.4 | 0.75 |
| | `|z| ≥ 10` | 1.6 | 0.52 |
| | lower half-plane `[−4, 4]²`, near the zeros of `w` | 8.5 | 0.82 |
| `erf(z)` | `|z| < 1` | 1.9 | 0.40 |
| | `[−10, 10]²`, near the zeros of `erf` | 12.1 | 0.74 |
| | `|z| ≥ 4` | 2.0 | 0.22 |
| `erfc(z)` | `|z| < 1` | 3.1 | 0.76 |
| | `[−10, 10]²`, near the zeros of `erfc` | 12.4 | 0.91 |
| `erfcx(z)` | `|z| < 1` | 2.9 | 0.76 |
| | `[−10, 10]²`, near the zeros of `erfcx` (`x < 0`) | 21.0 | 0.86 |
| | `|z| ≥ 10` | 1.7 | 0.52 |

Near a zero the value is a difference of two terms each far larger than it (`2e^(−z²) − w(−z)`, `1 − erfc(z)`), so the error relative to `|w|` grows as `|w|` falls; elsewhere it stays within 4.1 ε, also within `10⁻¹²` of the real and the imaginary axes.

### Method

* **`exp_dd`.** `e^a` of a double-double `a` to about 2⁻⁶⁰: `a = k·ln 2/64 + r` with `ln 2/64` in two parts (the first of 35 bits, so that `k·L₁` is exact), `e^r − 1` by its Taylor series to `r⁶` (`|r| ≤ ln 2/128`), and a table of the 64 values `2^(j/64)` as double-doubles; the result is a double-double and a power of 2 kept apart, so that nothing over- or underflows before the final rounding, which is the only one (and once also for a subnormal result).
* **`erf`, `erfc` for `|x| < 1/2`.** `erf(x) = x + x·q(x²)`, `q` a minimax polynomial of degree 9 to 2⁻⁵⁷ (relative to `erf(x)/x`), rounded once by a fused multiply-add; `erfc(x) = (1 − x) − x·q(x²)` with `1 − x` exact as two doubles.
* **`erfcx` for `x ≥ 1/2`.** For `1/2 ≤ x < 8`, 32 minimax polynomials of degree 10 in `x − c`, one per eighth of a binade, picked from the bits of `x` (no division, no branch), each with its constant term as a double-double (2⁻⁵⁶·⁸ after the rounding of the coefficients); for `x ≥ 8`, `x·erfcx(x)` as a polynomial of degree 9 in `1/x²` (2⁻⁶⁰), times `1/x` as a double-double. For `|x| < 1/2`, 8 polynomials of degree 11 on eighths of `[−1/2, 1/2)`; for `x ≤ −1/2`, `2e^(x²) − erfcx(−x)` with `e^(x²)` from `exp_dd`.
* **`erfc`, `erf` for `|x| ≥ 1/2`.** `erfc(x) = e^(−x²)·erfcx(x)`, `x²` exact as two doubles, `e^(−x²)` by `exp_dd` and the product as a double-double; `erf = 1 − erfc` and `erfc(−x) = 2 − erfc(x)` add the double-double to 1 or 2 exactly before the one rounding.
* **`erf_inv`, `erfc_inv`.** For `|x| ≤ 1/2`, a polynomial of degree 7 in `x²` to 2⁻³⁴ and one Newton step on the residual `erf(y₀) − x = (y₀ − x) + y₀·q(y₀²)`, exact but for its last rounding; tiny `x` as `x·√π/2` in two parts, rounded once. For `x > 1/2` (and `erfc_inv(c)`, `c ≤ 1/2`), `erfc_inv(1 − x)` with `1 − x` exact: 12 polynomials of degree 6 to 8 in `s = √(−ln c)` (two a binade of `s`, to 2⁻³¹), then one Halley step on `(erfc(y₀) − c)·e^(y₀²) = erfcx(y₀) − c·e^(y₀²)`, both terms as double-doubles and `c·2^k` scaled exactly, which holds for every subnormal `c`; `erfc_inv(c) = −erfc_inv(2 − c)` above 1.5 and `erf_inv(1 − c)` between.
* **`dawson`.** `x + x³·Q(x²)` (degree 8) for `|x| < 1/2`; 32 polynomials of degree 11 on eighths of binades for `1/2 ≤ |x| < 8` (2⁻⁵⁵); `x·F(x)` in `1/x²` (degree 9) beyond.
* **`fresnel`.** For `|x| < 1`, `C = x + x⁵·P(x⁴)` and `S = x³·Q(x⁴)` (degree 8, `x³` and `Q` as double-doubles; below 2⁻³³⁰, `S = (π/6)x³` scaled by 2³³⁰ and back with one rounding). From 1, `C = 1/2 + f·sin(πx²/2) − g·cos(πx²/2)` and `S = 1/2 − f·cos − g·sin` with the auxiliary functions `f` and `g` from 24 polynomials of degree 11 on eighths of the binades `[1, 8)` and as `x·f`, `x³·g` in `1/x⁴` (degrees 4 and 5) from 8; `x²` is exact as two doubles and each is reduced modulo 4 exactly, so the phase is right for any `x`, and `sin(πr/2)`, `cos(πr/2)` for `|r| ≤ 1/2` are minimax polynomials; the products and the sum with 1/2 are double-doubles.
* **`faddeeva` and the complex functions.** In the closed upper half-plane, by `w(−x + iy) = conj w(x + iy)` for `x ≥ 0`: for `|z| ≥ 10` the asymptotic series `i/(√π z)·Σ (2n − 1)!!/(2z²)ⁿ` to `n = 13`, plus `e^(−x²)` in the real part within `10⁻²⁰` of the real axis; for `|z| < 10` the trapezoidal rule of `w(z) = (i/π) ∫ e^(−t²)/(z − t) dt` with the step `h = 1/2` on 27 nodes shifted to `x + (n + 1/2)h` — no node meets the pole — plus its exact pole correction `2e^(−z²)/(1 + e^(2πy/h))` for `y < π/h`. The aliasing error is `e^(−π²/h²)`, below 2⁻⁵⁴ of `|w|` (0.3 ε measured in exact arithmetic); every node costs one real division (two nodes share one), and `e^(−(t₀ ± kh)²)` comes from `e^(−t₀²)`, `e^(∓t₀)` and `e^(−k²/4)` by products. On the real axis `w = e^(−x²) + 2i/√π·F(x)` and on the imaginary one `erfcx(y)`, from the real functions. In the lower half-plane `w(z) = 2e^(−z²) − w(−z)`, with `x² − y²` and `2xy` as double-doubles, `exp_dd`, and a sine and cosine with Cody and Waite's reduction by a three-part `π/2` and fdlibm's kernels (the standard library's for phases beyond 2²⁰), so that the exponential is right to the last bits for any `|z|`. Then `erfcx(z) = w(iz)`, `erfc(z) = e^(−z²)·w(iz)` for `x ≥ 0` and `2 − erfc(−z)` for `x < 0`, and `erf(z)` by its Taylor series in `z²` (19 terms) for `|z| < 1` and `±(1 − erfc(±z))` beyond; no standard-library transcendental function is used (but for those phases beyond 2²⁰ and `ln` in the first guess of `erfc_inv`), so the results are the same on every target with a fused multiply-add.

The coefficients are this module's own minimax fits (a discrete Remez exchange in mpmath at 160 bits, `tables.rs` made by `gen_tables.py`); the sine and cosine kernels are fdlibm's.

### Measured

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation only, never timed.

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

- `erf_inv` and `erfc_inv`: Boost and statrs, 1.5 to 2 times, at 1.6 to 2.8 ulps against 0.75: the first guess takes a logarithm and a square root, and the Halley step evaluates `erfcx` and an exponential.
- `erfcx` from 8: puruspe, 1.5 times, at 1.9 ulps: a division and a polynomial of degree 9 in `1/x²` against its table.
- `erf` and `erfc` on `|x| < 1/2`: libm, up to 1.3 and 1.4 times, at 0.83 ulps: a chain of 9 dependent multiply-adds against its rational of two short chains.
- `faddeeva`: puruspe, 1.1 to 2 times: 27 nodes and the pole correction, against an algorithm 20 to 4,000 times less accurate away from the origin.

## The Bessel and Airy functions

### Bessel functions

- `bessel_j(ν, x)` and `bessel_y(ν, x)` are the Bessel functions of the first and second kind `J_ν(x)` and `Y_ν(x)`, `bessel_i(ν, x)` and `bessel_k(ν, x)` the modified ones `I_ν(x)` and `K_ν(x)`, of any real order `ν` (an `f32` or an `f64`, the type of `x`). `bessel_i_scaled(ν, x) = e^(−|x|)·I_ν(x)` and `bessel_k_scaled(ν, x) = e^x·K_ν(x)` do not overflow or underflow where `I` and `K` would.
- **Negative orders** by the reflections: `J_(−ν) = cos(νπ)·J_ν − sin(νπ)·Y_ν`, `Y_(−ν) = sin(νπ)·J_ν + cos(νπ)·Y_ν`, `I_(−ν) = I_ν + (2/π)·sin(νπ)·K_ν`, `K_(−ν) = K_ν`, with `sin(νπ)` and `cos(νπ)` exact at the integers and half-integers (so `J_(−n) = (−1)ⁿ·J_n` exactly).
- **Negative arguments** for the integer orders only: `J_n(−x) = (−1)ⁿ·J_n(x)`, `I_n(−x) = (−1)ⁿ·I_n(x)` (and `e^(−|x|)·I_n(x)` likewise); a negative `x` with a non-integer order, or for `Y` and `K`, is NaN.
- **Special values.** `J₀(0) = I₀(0) = 1`; `J_ν(0) = I_ν(0) = 0` for `ν > 0` and `±∞` for a negative non-integer `ν` (the sign of `1/Γ(1 + ν)`); `Y_ν(0) = −∞` (for `ν ≥ 0`), `K_ν(0) = +∞`; `J`, `Y`, `K` → 0 and `I` → `∞` as `x → ∞`, the scaled forms → 0. NaN in gives NaN, and so does an infinite order.
- **Spherical functions** of an order `n: u32`: `sph_bessel_j(n, x) = jₙ(x) = √(π/(2x))·J_(n+1/2)(x)`, `sph_bessel_y` (`yₙ`), `sph_bessel_i` (`iₙ`, of `I_(n+1/2)`) and `sph_bessel_k` (`kₙ = √(π/(2x))·K_(n+1/2)(x)`, SciPy's `spherical_kn`). `jₙ`, `yₙ` and `iₙ` are defined for negative `x` by their parities (`jₙ(−x) = (−1)ⁿ jₙ(x)`, `yₙ(−x) = (−1)ⁿ⁺¹ yₙ(x)`, `iₙ(−x) = (−1)ⁿ iₙ(x)`); `kₙ` of a negative `x` is NaN. `j₀(0) = i₀(0) = 1`, `jₙ(0) = iₙ(0) = 0` for `n ≥ 1`, `yₙ(0) = −∞`, `kₙ(0) = ∞`.
- **Large orders.** From `ν = 50` `I_ν` and `K_ν` (plain and scaled) come from Debye's uniform expansions, in constant time. `J_ν` and `Y_ν` take time that grows with `ν` where `x` is near or above the order, up to `ν = 10⁷`; above, they come from Olver's uniform expansion (with the module's Airy functions), in constant time, and are 0 and `−∞` where they underflow and overflow (`x` well below `ν`); the spherical functions of orders above `10⁷` follow `J`, `Y` and `K` of `n + 1/2`. No valid argument gives NaN.

### Airy functions

- `airy_ai(x)`, `airy_bi(x)` and their derivatives `airy_ai_prime(x)`, `airy_bi_prime(x)`, the solutions of `w″ = x·w` with `Ai(0) = 3^(−2/3)/Γ(2/3)`, `Ai′(0) = −3^(−1/3)/Γ(1/3)`, `Bi(0) = 3^(−1/6)/Γ(2/3)`, `Bi′(0) = 3^(1/6)/Γ(1/3)`, for every real `x`.
- `Ai(x)` and `Ai′(x)` underflow to 0 (and −0) beyond about 104, `Bi(x)` and `Bi′(x)` overflow to `∞` beyond about 104.8. On the negative axis they oscillate in `ζ = (2/3)|x|^(3/2)`, which is formed in double-double: its absolute error, `2⁻¹⁰⁴·ζ`, is the phase error, so beyond about `|x| = 10⁹` the result keeps its error bound in ulps of the modulus only up to that growing phase error, and beyond `|x| = 10²⁰⁰` (where `ζ` overflows) `Ai` and `Bi` return 0 and their derivatives NaN; `Ai(−∞) = Bi(−∞) = 0`, `Ai′(−∞)` and `Bi′(−∞)` are NaN.

### Examples of the Bessel and Airy functions

```rust
use a::math::special::{airy_ai, bessel_j, bessel_k, bessel_y, sph_bessel_j};

// At the double nearest the first zero of J₀ the value is −6.1·10⁻¹⁷, to within an ulp.
assert!((bessel_j(0.0f64, 2.404_825_557_695_773) + 6.108_765_259_736_73e-17).abs() < 4e-32);
// J_(1/2)(x) = √(2/(πx))·sin x, and Y_(−1/2) = J_(1/2).
let j = bessel_j(0.5f64, 1.0);
assert!((j - 0.671_396_707_141_803_1).abs() < 2e-16);
assert!((bessel_y(-0.5f64, 1.0) - j).abs() < 2e-16);
assert!((bessel_k(1.5f32, 2.0) - 0.179_906_66).abs() < 2e-8);
assert!((sph_bessel_j(1, 1.0f64) - 0.301_168_678_939_756_8).abs() < 1e-16);
assert_eq!(airy_ai(0.0f64), 0.355_028_053_887_817_2);
```

### Accuracy of the Bessel and Airy functions

The bounds are the largest errors measured, in ulps of `f64` (an `f32` result is correctly rounded but within `0.51` ulp in every case tested), against mpmath at 256 bits: the test data of `tests/math/special/data/bessel_*.txt` and a larger random comparison of 2,000 to 4,000 cases per function and region. Near the zeros of the oscillating functions the error of a method of this kind is absolute: the bound is then in ulps of the modulus `M = √(J² + Y²)` (for `Ai`, `Bi` of `√(Ai² + Bi²)`, for their derivatives of `√(Ai′² + Bi′²)`), which bounds the value and equals its size away from the zeros.

| Function | Region | Bound |
| --- | --- | --- |
| `J₀`, `J₁`, `Y₀`, `Y₁` | `x < 24` (zeros included) | 1.5 ulps of the value |
| | `x ≥ 24` | 1 ulp of `M` |
| `J_ν`, `Y_ν`, other orders | `\|ν\| ≤ 10⁷` (measured to 300) | `J`: 5, `Y`: 6 ulps of `M` |
| | `\|ν\| > 10⁷` (Olver's expansion) | `2·max(ν, x)` ulps of `M` |
| `I₀`, `I₁`, `K₀`, `K₁`, scaled | all `x` | 1 ulp |
| `I_ν`, `K_ν`, other orders, scaled | `\|ν\| < 50` | 6 ulps |
| | `ν ≥ 50` (Debye's expansions) | 1 ulp |
| `jₙ`, `yₙ` | `n = 0` | 1 ulp of the value |
| | `n ≥ 1` | 2 ulps of `M` |
| `iₙ`, `kₙ` | all | 3 ulps (`kₙ`: 1) |
| `Ai`, `Ai′`, `Bi`, `Bi′` | `x ≥ −8.4` (zeros included) | 1.5 ulps of the value |
| | `−10⁹ ≤ x < −8.4` | 1 ulp of the modulus |

The orders 0 and 1 of `I` and `K` measure 0.64 ulp at most, `J` and `Y` of those orders 1.10 and 1.49 ulps near their zeros below 24; the Airy functions 1.25 ulps at most from −8.4 up and 0.69 ulp of the modulus below; Debye's `I` and `K` 0.50 ulp. The bounds of the general orders below 50 are reached by `Y` and `K` at `x` below 2 (Temme's series in `f64`), elsewhere they measure 2–3 ulps. Above `ν = 10⁷` the phase of Olver's expansion reaches the Airy functions through an `f64` argument, which bounds the error by about `2·max(ν, x)` ulps of the modulus (1.35 times measured).

### Methods of the Bessel and Airy functions

- **Orders 0 and 1 of `J` and `Y`.** Below 24, each interval between two extrema in three pieces: about its zero `f(z + d) = d·q(d)` with the zero as a double-double, and about the midpoints of the two outer thirds plain polynomials, Remez fits of degree 12 to 14 evaluated by Estrin's scheme with their first two terms exact: the error stays relative through the zeros. `J₀` below 1.5 and `J₁` below the first extremum are polynomials in `x²`; `Y₀` and `Y₁` below half their first zero are `(2/π)·ln x·J + R(x²)`, and about their first two zeros `(2/π)·ln(x/z)·J + (x − z)·g(x)`, whose fit `g` has no logarithmic singularity. From 24 on, the modulus and the phase, `J = M cos θ`, `Y = M sin θ`, `M = √(2/(πx))·(1 + P(1/x²))`, `θ = x − (2n + 1)π/4 + α(1/x)`, Remez fits, with `x` reduced modulo `π/2` to a double-double (Cody and Waite's three-part `π/2` below 10⁶, Payne and Hanek's 1280 bits of `2/π` above) and the sine and cosine of the reduced double-double by FreeBSD msun's kernels.
- **Orders 0 and 1 of `I` and `K`.** Polynomials in `x²` up to 1 (`K` with its `ln x·I` apart), `I` in nine pieces of `[1, 7.75]` about their midpoints, and `e^(∓x)·√x·(I, K)` in pieces of `1/x`: Remez fits with their first two coefficients in double-double, evaluated by Estrin's scheme; `1/√x` from `√x·(1/x)` with one correction, `e^x` from a table of `2^(j/32)` and a polynomial to 2⁻⁵⁸.
- **Other orders.** With `ν = n + μ`, `|μ| ≤ 1/2`: `J_μ` and `J_(μ+1)` from their power series summed in double-double (it serves up to 30, where its terms cancel down to `e^(−x)`); `Y_μ` and `Y_(μ+1)` from Temme's series below 2 and from Steed's continued fraction CF2, `p + iq = (J′ + iY′)/(J + iY)`, summed backwards (`120/x + 8` terms), as `Y_μ = (p·J_μ − J′_μ)/q`; Hankel's expansion from 30 on; then the recurrences in double-double up to `ν` (`Y` always, `J` while `x > ν`), and where `x ≤ ν` `J_ν` from the Wronskian with CF1 or from its series. `I_ν` from its power series (positive terms, double-double) below 30, else Hankel's expansion or the Wronskian with CF1; `K_μ`, `K_(μ+1)` from Temme's series below 1/2 and from the Wronskian `I_μ·K_(μ+1) + I_(μ+1)·K_μ = 1/x` with `K_(μ+1)/K_μ` by CF2 summed backwards (no step of it cancels), then up by the recurrence. Integer orders start from the dedicated `J₀`, `J₁`, `Y₀`, `Y₁`, `I₀`, `I₁`, `K₀`, `K₁`. The scaled forms carry `e^(∓x)` as a separate exponent throughout. From `ν = 50`, `I` and `K` by Debye's expansions `I_ν(x) = e^(νη)/√(2πr)·Σ Uₖ(p)/νᵏ`, `K_ν(x) = √(π/2)·e^(−νη)/√r·Σ (−1)ᵏUₖ(p)/νᵏ`, `r = √(ν² + x²)`, `p = ν/r`, `νη = r + ν·ln(x/(ν + r))` in double-double, `U₁` to `U₁₂` from their recurrence; above `ν = 10⁷`, `J` and `Y` by Olver's expansion to its terms `A₀` and `B₀`, `ζ`, `φ` and `B₀` by their series about `x = ν` or their closed forms.
- **Spherical functions.** `j₀ = sin x/x`, `y₀ = −cos x/x` from the double-double reduction; the power series where `x² ≤ n + 3/2`; the terminating Hankel sums `jₙ = (P sin(x − nπ/2) + Q cos(x − nπ/2))/x` from `x ≥ n(n + 1)/2`; between, `jₙ` up from `j₀`, `j₁` where `x > n` (in `f64` up to `n = 12`) and down from the continued fraction of `jₙ/jₙ₋₁` where `x ≤ n` (normalised by `j₀` or `j₁`), `yₙ` up from `y₀`, `y₁` (in `f64` for `n ≤ 12` and `n ≤ 4` with `x ≥ n`, else in double-double); `kₙ = (π/(2x))·e^(−x)·Σ aₖ/(2x)ᵏ` by Horner's rule in double-double up to `n = 10`, beyond by the recurrence of positive terms; `iₙ` from its series or the closed form.
- **Airy functions.** For `x ≥ 4` (`Bi`, `Bi′`: 12), `Ai(x) = e^(−ζ)/(2√π·x^(1/4))·(1 + t·A(t))`, `t = 1/ζ`, and likewise for the others; from 1 up to there Remez fits of `e^(±ζ)·f` in pieces; from the first extremum left of 0 up to 1 Remez fits of `f` itself; down to about −8.5 one polynomial about each zero (as for `J₀`); below, the modulus and the phase, `Ai(−y) = M cos θ`, `Bi(−y) = M sin θ`, `θ = π/4 − ζ + α(1/ζ)` (`Ai′`, `Bi′` with `N` and `φ = 3π/4 − ζ + β`), Remez fits, the phase reduced in double-double.
- Nothing allocates. The read-only tables hold about 33 KB of coefficients (`J₀`…`Y₁` 14 KB, `I₀`…`K₁` and `1/Γ(1 + μ)` 6.2 KB, the large orders 1.8 KB, the Airy functions 11.2 KB), `2^(j/32)` 512 bytes and the bits of `2/π` 160 bytes; their generators (`gen01.py`, `gen_ik.py`, `gen_olver.py` and the shared `remez.py` in `bessel/`, `gen_airy.py` and `airy_post.py` in `airy/`, on mpmath) reproduce them exactly. The elementary functions are the module's own (`e^x`, `ln x`, the sine and cosine, `atan`), and the polynomials use `mul_add` where it is one instruction (aarch64, x86_64 with FMA) and plain products elsewhere: all targets of one kind give the same bits, the others (and Miri) the same to within the bounds.

### Measured

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation only, never timed.

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

- `zeta(s)` is the Riemann zeta function `ζ(s) = Σ n⁻ˢ`, continued to the whole line or plane but its pole at `s = 1` (`+∞` there): of an `f32`, an `f64`, a `Complex<f32>` or a `Complex<f64>`. Its trivial zeros `−2, −4, …` are exact zeros, `ζ(+∞) = 1`, `ζ(−∞)` is NaN; past `s ≈ −260` it overflows to an infinity of the sign of `sin(πs/2)`.
- `hurwitz_zeta(s, a)` is `ζ(s, a) = Σₙ (n + a)⁻ˢ` for `s > 1` and `a > 0` (`+∞` at `s = 1`, NaN outside), with `ζ(s, 1) = ζ(s)` exactly.
- `dilog(x)` is the dilogarithm `Li₂(x) = −∫₀ˣ ln(1 − t)/t dt` of any real `x`, its real part `π²/6 − ∫₁ˣ ln|1 − t|/t dt` past 1 (as GSL's `gsl_sf_dilog`); `Li₂(1) = π²/6`, `Li₂(±∞) = −∞`.
- `polylog(s, x)` is the polylogarithm `Li_s(x) = Σ xᵏ/kˢ` of a real order `s`, for `x ≤ 1` (NaN past 1): `ζ(s)` at `x = 1` for `s > 1` and `+∞` there for `s ≤ 1`, `−ln(1 − x)` for `s = 1`, `x/(1 − x)` for `s = 0`, `dilog(x)` for `s = 2`; for `x < −1` the real value of the continuation.
- `lambert_w0(x)` and `lambert_wm1(x)` are the two real branches of the Lambert W function, the `w` with `w eʷ = x`: `W₀ ≥ −1` for `x ≥ −1/e`, `W₋₁ ≤ −1` for `−1/e ≤ x < 0` (`W₋₁(0⁻) = −∞`). Both are `−1` at the `f64` and the `f32` nearest `−1/e`, which lie just below it; below them both are NaN.
- `expint_e1(x)` is `E₁(x) = ∫₁^∞ e^(−xt)/t dt` for `x > 0` (`+∞` at 0, NaN for `x < 0`); `expint_ei(x)` is `Ei(x) = −PV ∫₋ₓ^∞ e^(−t)/t dt` for `x ≠ 0` (`−∞` at 0, `−E₁(−x)` for `x < 0`), with its zero `x₀ = 0.372 507 41…` exact to the last bit; `expint_en(n, x)` is `Eₙ(x) = ∫₁^∞ e^(−xt)/tⁿ dt` for `x ≥ 0` (`e⁻ˣ/x` for `n = 0`, `1/(n − 1)` at `x = 0` for `n ≥ 2`).

### Error bounds

Measured against mpmath at 256 bits (the Hurwitz references by an Euler–Maclaurin sum of their own at 256 bits, since mpmath's `zeta(s, a)` loses digits for large `a`, and `Eₙ` by `x^(n−1) Γ(1 − n, x)` at 1024 bits) on the data of the tests and on ten times as many random cases more, in ulps of the result for `f64`, on Apple silicon (with fused multiply-adds); every `f32` result is within 0.501 ulp of the true value, that is correctly rounded but for the rare halfway cases (complex: 0.6 ε).

| Function | Region | Bound (ulps) |
| --- | --- | ---: |
| `zeta`, real | `s ≥ 0` | 0.7 |
| | `−1/2 ≤ s < 0` | 1.05 |
| | `s < −1/2` (the reflection) | 3.5 |
| `zeta`, complex | `σ > 1` | 2 ε of \|ζ\| |
| | `0 ≤ σ ≤ 1` | 3.5 ε of max(\|ζ\|, 1) (2.5 ε absolute at the first 29 zeros) |
| | `σ < 0`, by the Euler–Maclaurin formula (\|t\| ≥ 5 + 30\|σ\| for `σ ≥ −3`, or \|t\| > 200) | 8 ε of \|ζ\|, growing as `2.2^(−σ)` past `σ = −3` |
| | `σ < 1/2`, \|t\| small (the reflection) | 2.5 ε of \|ζ\| for `σ < 0`, 6.5 ε for `0 ≤ σ < 1/2` |
| `hurwitz_zeta` | all | 1.35 |
| `dilog` | all | 1.3 |
| `polylog` | `s ≥ 1`, `−2 ≤ x ≤ 1` | 3 |
| | `0 ≤ s < 1`, `−2 ≤ x ≤ 1` | 8.5 |
| | integers `s ≥ 1`, `x < −2` | 3 |
| | `s > 0`, `x < −2` (the integral of Fermi and Dirac over `Γ(s)`) | 3 |
| | integers `−64 ≤ s < 0`, `x ≥ 0` | 2.5 |
| | integers `−64 ≤ s < 0`, `x < 0` | 6.5 κ |
| | `s < 0`, `x < 0` | 8.5 κ |
| | `s < 0`, `x > 0` | 16 |
| `lambert_w0` | all | 1.0 |
| `lambert_wm1` | all | 1.1 |
| `expint_e1` | all | 1.1 |
| `expint_ei` | all (the largest about the zero, `0.27 < x < 0.47`; 1.8 on x86_64 without FMA) | 1.45 |
| `expint_en` | all | 1.05 |

For `s < 0` the polylogarithm has real zeros on the negative axis, where no method keeps a relative bound; there the bound is in units of its condition number `κ = (|x ∂ₓLi_s| + |s ∂ₛLi_s|)/|Li_s|` (at least 1; `x ∂ₓ Li_s = Li_(s−1)`), as for `ellip_pi_inc`: the error is at most that many times `κ` ulps. The bounds that rest on `Γ` (the real `ζ` for `s < −1/2`, the complex one's reflection, `Li_s` of non-integer orders past `−2` and of negative orders) were measured with the gamma area's `gamma_f64` and `gamma_c64`.

### The method

- **Real `ζ`.** `s ≥ 56`: `1 + 2⁻ˢ`. `3 ≤ s < 56`: `1 + 2⁻ˢ V(s)`, `V = (ζ − 1)·2ˢ` a rational minimax fit on `[3, 8]`, `[8, 20]` and `[20, 56]` (degrees 6/6, 6/6, 4/4), weighted relative to `ζ`. `−1/2 ≤ s < 3`: `1/(s − 1) + R(s)` with `R = ζ(s) − 1/(s − 1)` entire, a polynomial minimax fit of degree 16, the reciprocal (with the low part of `s − 1`) and the head of `R` as double-doubles, so that the cancellation below 1 costs nothing. `s < −1/2`: `ζ(s) = 2 (2π)^(s−1) sin(πs/2) Γ(1 − s) ζ(1 − s)`, `1 − s` exact as a double-double whose low part enters through `ζ′/ζ` and `ψ − ln 2π`, `sin(πs/2)` with the argument reduced exactly, the product `2 sin(πs/2) ζ(1 − s) Γ(1 − s)` as a double-double and `(2π)^(s−1)` from a double-double exponent so that the result is rounded once, and past `1 − s = 171` Legendre's duplication formula for `Γ` (`z/2 + 1/2`, which rounds where it crosses a power of 2, with its low part through `ψ`).
- **Complex `ζ`.** The Euler–Maclaurin formula `Σ_{n<N} n⁻ˢ + N^(1−s)/(s − 1) + N⁻ˢ/2 + Σ_{k≤24} B₂ₖ/(2k)! (s)₂ₖ₋₁ N^(−s−2k+1)` with `N = ⌈(|s| + 40)/2.83⌉`: each `n⁻ˢ` has its phase `t ln n/2π` as a double-double reduced exactly, and its modulus from a double-double `ln n`; the terms are added from the smallest, in double-double. Left of `σ = 1/2` where `|t|` is small, the reflection with the hyperbolic functions of `πt/2` from double-double arguments and `Γ(1 − s)` from `gamma_c64`, corrected by `ψ(1 − s)` for the low part of `1 − σ` (which rounds where it crosses a power of 2, and cost 30 ε at `σ ≈ −15` before). The cost grows as `|t|` (about 5 µs at `|t| = 1000`).
- **Hurwitz `ζ`.** The Euler–Maclaurin formula with 24 Bernoulli terms from `A = a + N ≥ 7 + s/2`, or the direct sum alone once its terms fall below `2⁻⁵⁶` of the first; every term positive, added from the smallest in double-double, `n + a` exact, the first term and the remainder through a double-double `ln`.
- **`Li₂`.** On `−1 ≤ x ≤ 3/4` the series in `y = −ln(1 − x)` (`Li₂ = y − y²/4 + y³ P(y²)`, `P` a minimax fit of degree 8), `y` corrected to a double-double by `(1 − x) eʸ − 1`; elsewhere the reflection `π²/6 − ln x ln|1 − x| − Li₂(1 − x)` and the inversions `∓π²/6 − ½ ln²|x| − Li₂(1/x)` with double-double constants, logarithms and products; about the zero `12.595…` of `Re Li₂` a polynomial in `x − x₀`.
- **`Li_s`.** `|x| ≤ 1/2` (to `0.6` for `s ≥ 1`, and every `x ≥ −1` for `s ≥ 20`): `Σ xᵏ k⁻ˢ`, `p⁻ˢ` of the primes from double-double logarithms and the rest by products. `1/2 < x < 1`: `Γ(1 − s)(−μ)^(s−1) + Σ ζ(s − k) μᵏ/k!`, `μ = ln x`, the pole of `ζ(s − k)` at `k = n − 1` (`n` the integer nearest `s ≥ 1/2`) and that of `Γ(1 − s)` taken together in closed form (through `ln Γ(1 − ε)/ε`, `ln Π(1 + ε/j)/ε` and `expm1`), so that orders near integers cost nothing; `ζ(s − k)` past `s − k < −1/2` by the reflection as a recurrence in `k`. `−2 ≤ x < −1/2`: `−Σ η(s − k) μᵏ/k!`. `x < −2`, `s > 0`: integers by the inversion `−(−1)ⁿ Li_n(1/x) − Σ 2η(2j) μⁿ⁻²ʲ/(n − 2j)!` (Horner's rule, `μ` a double-double); the others by the integral of Fermi and Dirac, `Li_s(−eᵘ) = −F(μ)/Γ(s)`, `F = ∫₀^∞ t^(s−1)/(1 + e^(t−μ)) dt`, every term positive: on `[0, min(1, μ)]` the Taylor series of the logistic integrated exactly, past it 16-point Gauss–Legendre pieces (at most twice their left end, 3 long near `μ` for the poles `μ ± iπ`, 6 beyond), each about its exact midpoint so that `t^(s−1)` and `e^(t−μ)` keep their accuracy, `μ` a double-double — or, where they apply, the divergent series cut at its least term (orders large against `ln|x|`) and Sommerfeld's expansion (`x < −e^37.7`). Non-integer `s < 0`, `x < 0` (but small `|x|`, where the series is better): `Li_s(x) = 2 Γ(1 − s)(2π)^(s−1) Re[i^(1−s) ζ(1 − s, 1/2 − iμ/2π)]` with the Hurwitz zeta function of a complex argument by the Euler–Maclaurin formula (its terms decay since `1 − s > 1`). Integers `s ≤ 0`: `x Aₙ(x)/(1 − x)ⁿ⁺¹` with Eulerian polynomials.
- **Lambert W.** Near the branch point both branches are polynomials in `p = √(2(1 + e x))` (`±p`), `1 + e x` without the cancellation (`e` as a double-double), on `|p| ≤ 0.66`; `W₋₁` goes on to `x = −0.2` in `s = √(2(−ln(−x) − 1))` with `s² = −2 ln(1 − q)` from the accurate `q`; each a minimax fit of full precision with a double-double head. `W₀` on `0.66 < p ≤ 4.31` (`−0.29 < x ≤ 3`) is `x e^(−W₀)` with `e^(−W₀)` a polynomial of full precision in `p` on five pieces (`p` a double-double, the product rounded once; its series below `|x| = 2⁻¹⁰`), without iteration. Elsewhere a fitted starting value good to `3·10⁻⁷` (in `ln x`, in `s`, or the asymptotic series) and one step of Halley's iteration with the residual `w eʷ − x` from a double-double `eʷ` (so that its only error is the final rounding), or for `|w| > 40` one step of Fritsch's iteration (of the fourth order).
- **`E₁`, `Ei`, `Eₙ`.** `E₁`: `F(x) − ln x` on `(0, 1]` with the entire `F = E₁ + ln x` a minimax polynomial of degree 14; `e⁻ˣ (H₀ + K(x))/x` on `(1, 2.5]` with `K` a rational fit (6/6); past it Boost.Math's rational of degree 10/11 in `1/x`; `e⁻ˣ` to about `2⁻⁶⁰` (a table of `2^(j/32)` as double-doubles) and the product rounded once. `Ei`: `ln(x/x₀) + (x − x₀) R(x)` on `(0, 6]` (`R` entire, four polynomial pieces with double-double heads; `x − x₀` with `x₀` a triple-double; `ln_1p` of a double-double near `x₀`); `eˣ (1 + K(x))/x` past 6 with `K` rational fits on four pieces to 48 and polynomials in `1/x` past it. `Eₙ`, `n ≥ 2`: the recurrence `Eₖ₊₁ = (e⁻ˣ − x Eₖ)/k` up from `E₁` for `x ≤ 2` and `n < 50`, else the continued fraction, its depth from Lentz's test (and a quarter more) and its value from the bottom up in double-double arithmetic.

The fits are minimax (Remez exchange for the polynomials, Lawson's iteration for the rationals, mpmath at 128 bits) made for this crate, but for `E₁` past 2.5, Boost.Math's.

### Measured

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation only, never timed.

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

- **`polylog`** of orders 1.5 to 2 past `x = −2` (up to 84× GSL's `fermi_dirac_int`, which is itself 10² ulps off there): the Gauss–Legendre pieces of the Fermi–Dirac integral cost about 1.5 µs against its coarser symmetrised quadrature.
- **Lambert W** (2 to 3× special and puruspe, at 3 to 8 ulps against our 1): the branch-point fits and the Halley step of the double-double exponential buy the last ulp.
- **`hurwitz_zeta`** (1.3× GSL 2.8, at 49 ulps): the 24 Bernoulli terms in double-double.
- **`expint_e1` and `expint_ei`** above `x = 1` (1.5 to 1.9× Boost at 4 to 5 ulps): the double-double `e^x` and the rational fits; **`Eₙ`** for large `n` (about 3× statrs at 0.5 ulps against its 40): the continued fraction evaluated from the bottom.

### Examples

```rust
use a::math::complex::Complex;
use a::math::special::{dilog, expint_ei, hurwitz_zeta, lambert_w0, lambert_wm1, polylog, zeta};

assert_eq!(zeta(2.0f64), core::f64::consts::PI.powi(2) / 6.0);
assert_eq!(zeta(-4.0f64), 0.0);
let z = zeta(Complex::new(0.5f64, 14.134_725_141_734_694)); // the first zero
assert!(z.abs() < 1e-14);
assert_eq!(hurwitz_zeta(3.0f64, 1.0), zeta(3.0f64));
assert!((dilog(-1.0f64) + core::f64::consts::PI.powi(2) / 12.0).abs() < 1e-15);
assert!((polylog(3.0f64, 0.5) - 0.537_213_193_608_040_2).abs() < 1e-15);
let w = lambert_w0(1.0f64); // the omega constant
assert!((w * w.exp() - 1.0).abs() < 1e-15);
assert!(lambert_wm1(-0.1f64) < -1.0);
assert!(expint_ei(0.372_507_410_781_366_6f64).abs() < 1e-16);
```

## Elliptic integrals and functions

All of them take the parameter `m = k²` (as SciPy, mpmath and Cephes do; Boost and GSL take the modulus `k`).

- `ellip_k(m)` and `ellip_e(m)` are the complete integrals of the first and second kind, `K(m) = ∫₀^(π/2) dθ/√(1 − m sin²θ)` and `E(m) = ∫₀^(π/2) √(1 − m sin²θ) dθ`, for every `m < 1` (`K(1) = ∞`, `E(1) = 1`; NaN above 1; `K(−∞) = 0`, `E(−∞) = ∞`).
- `ellip_pi(n, m)` is the complete integral of the third kind `Π(n, m) = ∫₀^(π/2) dθ/((1 − n sin²θ)√(1 − m sin²θ))`, its Cauchy principal value for `n > 1` (the real part of mpmath's value; `Π(1, m) = ∞`).
- `ellip_f(φ, m)`, `ellip_e_inc(φ, m)` and `ellip_pi_inc(n, φ, m)` are the incomplete integrals `F`, `E` and `Π` for any real `φ`: for `m ≤ 1` by the periodicity `F(φ + jπ) = F(φ) + 2jK` (and so for `E` and `Π`), for `m > 1` where they are real, `|sin φ| ≤ 1/√m` with `|φ| ≤ π/2` (NaN elsewhere). `Π` is the principal value past a pole `n sin²φ = 1`, and infinite at it.
- `carlson_rf`, `carlson_rd`, `carlson_rj` and `carlson_rc` are Carlson's symmetric integrals of non-negative arguments, `R_J(x, y, z, p)` for `p < 0` and `R_C(x, y)` for `y < 0` their principal values. Two zero arguments of `R_F` (and `R_J`), `x = y = 0` or `z = 0` of `R_D`, `p = 0` and `y = 0` give `∞`; a negative `x`, `y`, `z` gives NaN; an infinite argument gives 0.
- `jacobi_elliptic(u, m)` returns `(sn(u, m), cn(u, m), dn(u, m))` for every real `u` and `m` (`m = 0`: `sin`, `cos`, 1; `m = 1`: `tanh`, `sech`, `sech`; an infinite `u` gives NaN but for `m = 1`).

### Error bounds

Measured on the reference data (mpmath, 256 bits and more where its own precision was short; 4,200 cases of the elliptic functions) and on a random comparison of 41,500 more cases (the largest errors of both are given); `f32` results are rounded once from the `f64` computation and were correctly rounded on every case.

| Function | Region | Bound (ulps) | Measured |
| --- | --- | ---: | ---: |
| `ellip_k` | `m < 0.9` | 0.6 | 0.51 |
| | `0.9 ≤ m < 1` (`ln m₁` rounded once) | 1 | 0.98 |
| `ellip_e` | all `m ≤ 1` | 0.6 | 0.54 |
| `ellip_pi` | `n < 1`, `0 ≤ m ≤ 0.99` | 2 | 1.5 |
| | everywhere else (`n > 1`: the principal value) but… | 3.5 | 3.1 |
| | `m ≤ n < 0` with `m < −1`, where `K + (n/3)R_J` cancels | 9 | 8.0 |
| `ellip_f` | `m ≤ 1`, every `φ` | 1.5 | 1.2 |
| | `m > 1` | 2.5 | 1.96 |
| `ellip_e_inc` | `0 ≤ m ≤ 1`, every `φ` | 2 | 1.64 |
| | `m < 0`; `m > 1` (a mild cancellation) | 4 | 3.7 |
| `ellip_pi_inc` | `n < 1`, `0 ≤ m < 1`, `|φ| ≤ π/2` | 2 | 1.41 |
| | everywhere, in units of `1 + κ`, `κ = |φ/((1 − n sin²φ) Δ Π)|` its condition number in `φ` | 4 (1 + κ) | 3.6 (1 + κ) |
| | but `n ≫ 1`, `m ≪ 0` past a period (`2jΠ(n, m) + Π(n; r)` cancels) | – | 84 (1 + κ) |
| `carlson_rf` | all | 0.75 | 0.50 |
| `carlson_rd` | all | 0.75 | 0.50 |
| `carlson_rj` | `p > 0` | 2 | 1.87 |
| | `p < 0`, in ulps of `max(|R_J|, R_J(x, y, z, −p))` | 4 | 3.6 |
| `carlson_rc` | all (`y < 0`: the principal value) | 2.5 | 1.96 |
| `jacobi_elliptic` | `0 ≤ m < 0.9` | 3.5 | 3.3 |
| | `0.9 ≤ m < 1 − 10⁻⁶` | 20 | 16.8 (13.4 with macOS's libm) |
| | `1 − 10⁻⁶ ≤ m < 1` (`dn` near `u ≡ K/2 mod K`) | 80 | 70 |
| | `m < 0` (`|m| ≤ 10⁶`) | 12 | 10.9 |
| | `m > 1` | 4 | 3.4 |
| every function, `f32` | all | 0.501 | 0.50 (correctly rounded) |

The bounds of the Jacobi functions are in ulps of `max(|f|, 1)` for `sn` and `cn` (their zeros make relative error meaningless near them), relative for `dn` when `m ≤ 1`; for `m > 1` in ulps of `max(|sn|, 1/√m)` and `max(|dn|, 1)`.

### Method

- **`K` and `E`.** For `−1 ≤ m < 0.8`, 36 polynomials in `m − c` on intervals of width 0.05, Chebyshev fits of degree 9 (to 0.5) and 12 (to 0.8) made with mpmath at 200 bits, their constant terms double-doubles so that the result is rounded about once. For `m ≥ 0.8`, the logarithmic forms `K = −ln(m₁) K(m₁)/π + B(m₁)` and `E = 1 + m₁(−ln(m₁) S₁(m₁) + S₂(m₁))` (DLMF 19.12.1–2, `m₁ = 1 − m` exact there), whose terms are all positive. For `m < −1`, `K(m) = K(m′)/√(1 − m)` and `E(m) = √(1 − m) E(m′)` with `m′ = −m/(1 − m)` (DLMF 19.7.2), `1 − m′ = 1/(1 − m)` and `ln(1 − m)` taken directly, `√(1 − m)` in double-double.
- **Carlson's integrals.** The duplication algorithm (Carlson 1995) with series to the thirteenth (`R_F`) and eleventh (`R_D`, `R_J`) orders, DLMF 19.19.7 expanded in the elementary symmetric functions of the deviations (by sympy), which stop it at deviations of 0.07 and 0.035 instead of the seventh order's 0.007: about one step fewer. Every argument is carried with its rounding error to first order: the square roots with their residuals, `λ` with the errors of its products and sums, the new arguments with the error of their sums, and the final `A^(−1/2)` and `A^(−3/2)` with the residuals of the square root and the reciprocal. A step then adds errors of second order only, and the result is about correctly rounded (it was on every case of `R_F` and `R_D`). The steps work on `λ/4` from half square roots, so that no argument up to `f64::MAX` overflows; arguments below `2^−600` are scaled up first. `R_J` carries `e_m = δ 4^(−3m)/d_m²` as a product of bounded ratios, and where `|e| ≥ 1/16` (where `R_C(1, 1 + e)` is sensitive to it) as the product of the three ratios `(√p − √x)/(√p + √x)` with their errors, then `R_C(1, 1 + e)` with the residuals of its square root and quotient; it takes `1 + e` without cancellation when `e` is near −1 (`p ≪ x, y, z`), and its principal value is Carlson's (2.22) in the form of Boost.Math with the products formed relative to `max(y, −p)`. `R_C` is `acos`, `atan`, `atanh` or `ln` of arguments carried with their errors, and a series near `x = y`.
- **Incomplete integrals.** `F = s R_F(c², Δ², 1)` with `s, c = sin φ, cos φ` of `φ` itself (no reduction error), `Δ² = c² + m₁ s²` for `0 ≤ m ≤ 1`; `φ = jπ + r` gives `2jK ± F(r)`, and where `F(|r|)` would cancel against `2jK` the complementary form `(2j − σ)K + σ c R_F(m₁s², Δ², m₁)`. `E` by DLMF 19.25.10's three positive terms where `m s² > 1/2` (the usual `s R_F − (m/3)s³R_D` cancels as `m → 1`). `Π` by `s R_F + (n/3)s³R_J(c², Δ², 1, c² + (1 − n)s²)` for `0 < n ≤ 1`; for `n < 0` (and `n < m`) by Boost's all-positive form of A&S 17.7.15 (R. Romeo); for `n > 1` by A&S 17.7.7, `Π(n) = −(N/3)s³R_J(…) + atanh-term/p₁`, `N = m/n`; for `m > 1` by the reciprocal-modulus transformation (DLMF 19.7.4).
- **Complete `Π`.** `K(m) + (n/3)R_J(0, m₁, 1, 1 − n)` with the fitted `K`; for `n < 0` the all-positive form; for `n > 1` `−(N/3)R_J(0, m₁, 1, (n − m)/n)`, `N = m/n` (DLMF 19.6.5).
- **Without a fused multiply-add.** The products' errors and the residuals come from `f64::mul_add` where it is an instruction (aarch64, or x86_64 built with `fma`), and from Dekker's product elsewhere (the default x86_64 target and Miri), where `mul_add` is a slow call into the library; the terms whose rounding is of second order are then plain `a·b + c`. The bounds hold on both (tested on aarch64 and on x86_64 without `fma`).
- **Jacobi functions.** `u` is reduced modulo `2K` with `K` a double-double (the fitted `K` for `m < 0.8` and up to four half-periods, the AGM in double-double otherwise), then to `[0, K/2]` by `sn(K − v) = cd v`, `cn(K − v) = √m₁ sd v`, `dn(K − v) = √m₁ nd v`; there the descending Landen transformation (A&S 16.12) is applied to the values themselves, which are products and quotients of positive quantities, down to a parameter below 10⁻⁹ where the first-order expansion is exact to rounding, and on the way up `(sn, cn)` is put back on `sn² + cn² = 1` at each level (the transformation amplifies errors off it by up to 2 a level). `m < 0` and `m > 1` go to `(0, 1)` by A&S 16.10–11 with the scaled argument a double-double.

### Examples

```rust
use a::math::special::{
    carlson_rf, chebyshev_t, ellip_e, ellip_f, ellip_k, ellip_pi, jacobi_elliptic, legendre_p,
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
// Π(n, 0) = π/(2√(1 − n)); for n > 1 the principal value.
assert!((ellip_pi(0.75f64, 0.0) - PI).abs() < 1e-15);
assert!(ellip_pi(2.0f64, 0.5) < 0.0);
// sn² + cn² = 1 and dn² + m sn² = 1.
let (sn, cn, dn) = jacobi_elliptic(0.7f64, 0.3);
assert!((sn * sn + cn * cn - 1.0).abs() < 1e-15);
assert!((dn * dn + 0.3 * sn * sn - 1.0).abs() < 1e-15);
// Orthogonal polynomials: P₂(x) = (3x² − 1)/2, T_n(cos θ) = cos nθ (to the rounding of cos θ).
assert!((legendre_p(2, 0.5f64) + 0.125).abs() < 1e-16);
assert!((chebyshev_t(1000, 0.3f64.cos()) - 300.0f64.cos()).abs() < 1e-12);
```

## Orthogonal polynomials

- `legendre_p(n, x)`, `hermite_h(n, x)`, `hermite_he(n, x)`, `laguerre_l(n, x)`, `chebyshev_t(n, x)`, `chebyshev_u(n, x)` of any real `x`; `assoc_legendre_p(n, m, x)` for `|x| ≤ 1` with the Condon–Shortley phase `(−1)ᵐ` (SciPy's `lpmv`; 0 for `m > n`, NaN for `|x| > 1`); `assoc_laguerre_l(n, α, x)`, `jacobi_p(n, α, β, x)` and `gegenbauer_c(n, α, x)` of any real parameters (`gegenbauer_c(n, 0, x) = 0` for `n ≥ 1`, as mpmath and the recurrence give). Infinite `x` gives the infinity of the leading term; NaN gives NaN.

### Error bounds

The error of a polynomial is measured in ulps of `max(|P(x)|, S)`, with `S` the polynomial's scale: the largest `|P|` on `[−1, 1]` (Legendre, associated Legendre, Chebyshev, Jacobi, Gegenbauer), and for Hermite and Laguerre the weighted largest value `e^(x²/2) max|Hₙ(t) e^(−t²/2)|` (`e^(x²/4)` for `Heₙ`) and `e^(x/2) max|Lₙ^(α)(t) e^(−t/2)|` inside the oscillatory region (`|x| ≤ √(2n + 1)`, `√(4n + 2)`, `0 ≤ x ≤ 4n + 2α + 2`); outside these regions, and for `|x| > 1`, the error is relative. So it is relative error away from the zeros and an absolute error on the polynomial's own scale near them.

| Polynomial | Degrees in the data | Bound | Measured |
| --- | --- | ---: | ---: |
| `legendre_p` | 0–1000, `|x| ≤ 3` | 0.55 | 0.46 |
| `assoc_legendre_p` | 0–120, all `m` | 0.55 | 0.49 |
| `hermite_h`, `hermite_he` | 0–300 | 0.55 | 0.50, 0.47 |
| `laguerre_l`, `assoc_laguerre_l` | 0–400, `α ∈ (−10, 30)` | 0.55 | 0.44, 0.50 |
| `chebyshev_t`, `chebyshev_u` | 0–5000, `|x| ≤ 1.5` | 0.55 | 0.50, 0.49 |
| `jacobi_p` | 0–300, `α, β ∈ (−5, 10)` | 0.55 | 0.40 |
| `gegenbauer_c` | 0–400, `α ∈ (−5, 20)` | 0.55 | 0.49 |
| every polynomial, `f32` | as above | 0.501 | 0.50 |

The bounds come from mixed error ulps of `max(|P|, S)`: relative error away from the zeros, and an absolute error relative to the polynomial's scale `S` near them.

### Method

Each polynomial comes from its three-term recurrence (DLMF 18.9.1–2 and the table there) run compensated: the value follows the plain recurrence and a second value carries its error, from the error-free transformations of every step's products and sum, so that the result is rounded about once; the coefficients are exact (Legendre, Hermite, Laguerre, Chebyshev, whose coefficients are integers and `x`), exact for parameters `α`, `β` on the grid `2^−20` (with magnitudes up to `2^28`: their sums with the degree are exact and their products exact double-doubles), or double-doubles. Where the recurrence divides by an integer `c_k` (Legendre, Laguerre, Gegenbauer, associated Legendre and Laguerre) it runs without the divisions, on `Q_k = p_k Π c_j` with the product kept beside it and one division at the end. The values are rescaled by `2^±500` as they grow or shrink, so that only a result beyond the range of `f64` overflows. `Pₘᵐ = (−1)ᵐ(2m − 1)!!(1 − x²)^(m/2)` is a product of `m` double-double factors with `1 − x²` exact. The Chebyshev polynomials above degree 16 are the binary powers of `T₁ + i sin θ U₀` in double-double, `O(log n)` steps whose errors stay near the double-double precision. Jacobi parameters at which the recurrence divides by zero (`α + β` a negative integer) take the explicit sum.

The recurrences take `O(n)` steps for the degree `n`: the asymptotic forms for large degrees (Legendre's by Stieltjes's and Bessel-type expansions, Hermite's by Plancherel–Rotach) are left to the release optimisation pass.


## Measured: elliptic functions and orthogonal polynomials

Measured on an Apple M4 Max on one thread with fat LTO, in ns a call, the median of the three quiet rounds of the campaign (the machine idle before it, the load average below 3.5 throughout), against libm 0.2, statrs 0.18, special 0.11, puruspe 0.4, GSL 2.8, Boost.Math 1.92 (without promotion to `long double`) and Cephes in the same run. "–" is a reference without the function. The x86_64 paths were tested under emulation only, never timed.

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

The references' accuracy on the same data, max / rms ulps (and the number of cases in the reference's domain; Boost and GSL take the modulus `k = √m`, which loses `m₁` as `m → 1`; Cephes takes `m₁`); the rows with a domain are the references' common one:

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

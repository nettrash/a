//! First guesses for the zeros of the Legendre, Hermite and Laguerre polynomials of large degree,
//! from their asymptotic expansions, for Newton's method to start from. They need not be
//! accurate, only close enough to their zero that Newton's method converges to it and to no
//! other. Measured against the zeros of `scipy.special.roots_*` and the 60-digit reference nodes
//! of the tests, in units of the spacing of the zeros around each: at degree 8, `1.1·10⁻⁵`
//! (Legendre), `1.4·10⁻³` (Hermite) and `2.6·10⁻²` (Laguerre); at 100, `9·10⁻⁹`, `1.4·10⁻⁴` and
//! `1.4·10⁻³`; at 2000, `1.4·10⁻¹¹`, `7·10⁻⁶` and `1.3·10⁻⁵`.

use core::f64::consts::PI;

/// The first zeros of the Bessel function `J₀`, to the precision of an `f64` (mpmath's
/// `besseljzero(0, k)`); McMahon's expansion takes over beyond them.
const J0_ZEROS: [f64; 20] = [
    2.404_825_557_695_773,
    5.520_078_110_286_311,
    8.653_727_912_911_013,
    11.791_534_439_014_281,
    14.930_917_708_487_787,
    18.071_063_967_910_924,
    21.211_636_629_879_26,
    24.352_471_530_749_302,
    27.493_479_132_040_253,
    30.634_606_468_431_976,
    33.775_820_213_573_57,
    36.917_098_353_664_045,
    40.058_425_764_628_24,
    43.199_791_713_176_73,
    46.341_188_371_661_815,
    49.482_609_897_397_815,
    52.624_051_841_115,
    55.765_510_755_019_98,
    58.906_983_926_080_94,
    62.048_469_190_227_166,
];

/// The first zeros of the Airy function `Ai`, negated, to the precision of an `f64` (mpmath's
/// `−airyaizero(k)`); the asymptotic series takes over beyond them.
const AIRY_ZEROS: [f64; 10] = [
    2.338_107_410_459_767,
    4.087_949_444_130_97,
    5.520_559_828_095_551,
    6.786_708_090_071_759,
    7.944_133_587_120_853,
    9.022_650_853_340_98,
    10.040_174_341_558_085,
    11.008_524_303_733_262,
    11.936_015_563_236_262,
    12.828_776_752_865_757,
];

/// The `k`-th positive zero of `J₀`, `k ≥ 1`: from the table, then by McMahon's expansion
/// (Abramowitz and Stegun 9.5.12 with `μ = 0`), whose error is below 10⁻¹⁵ from `k = 15` on.
pub(super) fn bessel_j0_zero(k: usize) -> f64 {
    if k <= J0_ZEROS.len() {
        return J0_ZEROS[k - 1];
    }
    let b = (k as f64 - 0.25) * PI;
    let e = 1.0 / (8.0 * b);
    let e2 = e * e;
    b + e * (1.0 + e2 * (-124.0 / 3.0 + e2 * (120_928.0 / 15.0 - e2 * 401_743_168.0 / 105.0)))
}

/// The `k`-th zero of `Ai`, negated, `k ≥ 1`: from the table, then by the asymptotic series
/// `T(3π(4k − 1)/8)` of Abramowitz and Stegun 10.4.94–105, whose error is below 10⁻¹⁴ from
/// `k = 10` on.
pub(super) fn airy_zero(k: usize) -> f64 {
    if k <= AIRY_ZEROS.len() {
        return AIRY_ZEROS[k - 1];
    }
    let t = 3.0 * PI * (4.0 * k as f64 - 1.0) / 8.0;
    let u = 1.0 / (t * t);
    t.powf(2.0 / 3.0)
        * (1.0
            + u * (5.0 / 48.0
                + u * (-5.0 / 36.0 + u * (77_125.0 / 82_944.0 - u * 108_056_875.0 / 6_967_296.0))))
}

/// The root of the increasing `g` on `[lo, hi]`, where `g(lo) ≤ 0 ≤ g(hi)`: Newton's method from
/// `x`, a step that would leave the bracket replaced by bisection. `g` returns the value and the
/// derivative.
fn increasing_root(g: impl Fn(f64) -> (f64, f64), mut lo: f64, mut hi: f64, mut x: f64) -> f64 {
    for _ in 0..100 {
        let (v, d) = g(x);
        if v == 0.0 {
            return x;
        }
        if v < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let mut next = x - v / d;
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }
        if (next - x).abs() <= 4.0 * f64::EPSILON * x.abs() || hi - lo <= f64::EPSILON * hi {
            return next;
        }
        x = next;
    }
    x
}

/// The positive zeros of `Pₙ`, largest first: near the end `x = 1`, `θ = ψ + (ψ cot ψ − 1)/(8ψρ²)`
/// with `ψ = j₀,ₖ/ρ` and `ρ = n + ½`, from the uniform expansion of `Pₙ(cos θ)` in terms of `J₀`
/// (Olver 1974; Frenzen and Wong 1985; the first guesses of Hale and Townsend 2013), and in the
/// interior Tricomi's expansion as Lether (1978) gives it.
pub(super) fn legendre(n: usize) -> Vec<f64> {
    let nf = n as f64;
    let rho = nf + 0.5;
    let boundary = (n / 10).max(10);
    (1..=n / 2)
        .map(|k| {
            if k <= boundary {
                let psi = bessel_j0_zero(k) / rho;
                let theta = psi + (psi * psi.cos() / psi.sin() - 1.0) / (8.0 * psi * rho * rho);
                theta.cos()
            } else {
                let phi = (4.0 * k as f64 - 1.0) * PI / (4.0 * nf + 2.0);
                let s = phi.sin();
                let c = 1.0
                    - (nf - 1.0) / (8.0 * nf * nf * nf)
                    - (39.0 - 28.0 / (s * s)) / (384.0 * nf * nf * nf * nf);
                c * phi.cos()
            }
        })
        .collect()
}

/// The positive zeros of the physicists' `Hₙ`, largest first: the phase condition of the WKB
/// approximation of `e^(−x²/2) Hₙ(x)` with the phase of the `k`-th zero of `Ai` in the place of
/// `(k − ¼)π`, which is the leading order of Langer's uniform approximation about the turning
/// point (Olver 1974, chapter 11): with `ν = 2n + 1` and `x = √ν cos φ`,
/// `(ν/4)(2φ − sin 2φ) = (2/3)|aₖ|^(3/2)`.
pub(super) fn hermite(n: usize) -> Vec<f64> {
    let nu = 2.0 * n as f64 + 1.0;
    (1..=n / 2)
        .map(|k| {
            let r = (8.0 / 3.0) * airy_zero(k).powf(1.5) / nu;
            let start = (0.75 * r).cbrt().min(0.5 * PI);
            let phi = increasing_root(
                |p| {
                    let (s, c) = (2.0 * p).sin_cos();
                    (2.0 * p - s - r, 2.0 - 2.0 * c)
                },
                0.0,
                0.5 * PI,
                start,
            );
            nu.sqrt() * phi.cos()
        })
        .collect()
}

/// The zeros of `Lₙ`, smallest first: for the smallest the expansion in terms of the zeros of
/// `J₀`, `x = j²/ν·(1 + (j² − 2)/(3ν²))` with `ν = 4n + 2` (Tricomi; Gatteschi's survey of 2002),
/// and for the rest the Langer approximation from the turning point of `√x e^(−x/2) Lₙ(x)`: with
/// `x = ν cos²(φ/2)`, `(ν/4)(φ − sin φ) = (2/3)|aᵢ|^(3/2)` for the `i`-th zero from the largest.
pub(super) fn laguerre(n: usize) -> Vec<f64> {
    let nu = 4.0 * n as f64 + 2.0;
    // The Bessel expansion is the better of the two up to about the 2√n-th zero (measured
    // against the zeros themselves from n = 50 to 200: the crossing is at 11, 17 and 28).
    let small = ((2.0 * (n as f64).sqrt()) as usize).max(4).min(n);
    (1..=n)
        .map(|k| {
            if k <= small {
                let j = bessel_j0_zero(k);
                let j2 = j * j;
                j2 / nu * (1.0 + (j2 - 2.0) / (3.0 * nu * nu))
            } else {
                let i = n - k + 1;
                let r = (8.0 / 3.0) * airy_zero(i).powf(1.5) / nu;
                let start = (6.0 * r).cbrt().min(PI);
                let phi = increasing_root(
                    |p| {
                        let (s, c) = p.sin_cos();
                        (p - s - r, 1.0 - c)
                    },
                    0.0,
                    PI,
                    start,
                );
                let c = (0.5 * phi).cos();
                nu * c * c
            }
        })
        .collect()
}

//! The dilogarithm and the polylogarithm of real arguments.
//!
//! The dilogarithm: on `−1 ≤ x ≤ 3/4` the series `Li₂(x) = Σ Bₙ yⁿ⁺¹/(n + 1)!` in
//! `y = −ln(1 − x)` (the odd part a polynomial minimax fit in `y²`), elsewhere the reflections
//! `Li₂(x) = π²/6 − ln x ln(1 − x) − Li₂(1 − x)` (for `3/4 < x ≤ 2`, the real part past 1) and
//! the inversions `Li₂(x) = −π²/6 − ½ ln²(−x) − Li₂(1/x)` (`x < −1`) and
//! `Re Li₂(x) = π²/3 − ½ ln² x − Li₂(1/x)` (`x > 2`), with the constants and products as
//! double-doubles; about the zero `x₀ = 12.595…` of `Re Li₂` a polynomial in `x − x₀`.
//!
//! The tables are printed by `gen_polylog.py` in this directory (with `gen_fit.py`).

use super::complex::{Cd, cmul_dd, turn_dd};
use super::hurwitz::BERNOULLI_2K;
use super::real::{digamma_rough, lgamma_dd, pow_2pi_neg, zeta_f64, zeta_regular, zeta_right};
use super::util::{
    PI_HI, atan_dd, exp_dd, exp_mul, exp_parts, horner, ln_dd, ln_dd_acc, mla, poly_tail, scale,
    sincos_pi, two_prod, two_sum,
};
use crate::math::complex::Complex;
use crate::math::special::dd::Dd;
use crate::math::special::gamma::gamma_f64;
use core::f64::consts::LN_2;

/// (Li₂(x) − y + y²/4)/y³ in y², y = −ln(1 − x), on −1 ≤ x ≤ 3/4: degree 8, error 8.3e-19
const DILOG_P: [f64; 9] = [
    0.027777777777777776,
    -0.0002777777777777758,
    4.724111866942019e-06,
    -9.185773060179793e-08,
    1.8978866091230078e-09,
    -4.064702586776881e-11,
    8.916413950655425e-13,
    -1.9662320678098767e-14,
    3.719600912956292e-16,
];

/// Re Li₂(x)/(x − x₀) about the zero x₀ of Re Li₂ past 1: degree 14 on [10.5, 15] in t − 12.59517036984501612863989649486460709565, error 2.4e-17 (2.5e-17 in doubles).
const DILOG_ZERO: [f64; 15] = [
    -0.1945657416318589,
    0.004300177565288094,
    -0.00012918826311071304,
    3.448648727094847e-06,
    5.668999785569417e-10,
    -1.2664183820727238e-08,
    1.6396675660485004e-09,
    -1.6422073978798362e-10,
    1.4964718610551848e-11,
    -1.3023200985892597e-12,
    1.1034641745878502e-13,
    -9.166022390625758e-15,
    7.71039953089577e-16,
    -6.992091925229154e-17,
    4.851643996520512e-18,
];
const DILOG_ZERO_LO: f64 = -1.8379667093964986e-18;

const DILOG_X0_HI: f64 = 12.595170369845016;
const DILOG_X0_LO: f64 = -2.384542072759316e-16;

/// `π²/6` and `π²/3` as double-doubles.
const PI2_6_HI: f64 = 1.644_934_066_848_226_4;
const PI2_6_LO: f64 = 3.040_672_350_398_476e-17;
const PI2_3_HI: f64 = 3.289_868_133_696_453;
const PI2_3_LO: f64 = 6.081_344_700_796_952e-17;

/// `Li₂(x)` for `−1 ≤ x ≤ 3/4` as a double-double: `y − y²/4 + y³ P(y²)` with
/// `y = −ln(1 − x)` a double-double (corrected by `(1 − x) eʸ − 1 = y − y_true` to first order)
/// and `y − y²/4` exact.
#[inline]
fn dilog_series_dd(x: f64) -> (f64, f64) {
    let y = -(-x).ln_1p();
    let (u, ul) = two_sum(1.0, -x);
    let (e, el, q) = exp_parts(y);
    let us = scale(u, q);
    let (m, me) = two_prod(e, us);
    // (1 − x) eʸ = e^(y − y_true).
    let yl = -((m - 1.0) + (me + el * us + e * scale(ul, q)));
    let (y2, y2e) = two_prod(y, y);
    let (h, l) = two_sum(y, -0.25 * y2);
    // and yl (1 − y/2)
    let rest = yl * (1.0 - 0.5 * y) - 0.25 * y2e + y2 * (y * horner(y2, &DILOG_P));
    two_sum(h, l + rest)
}

/// `−(c + ½ l² + v)` of the double-doubles `c`, `l` and `v`, rounded once.
#[inline(always)]
fn neg_sum_half_sq(c: (f64, f64), l: (f64, f64), v: (f64, f64)) -> f64 {
    let (p, pe) = two_prod(0.5 * l.0, l.0);
    let pl = pe + l.0 * l.1;
    let (a, ae) = two_sum(c.0, p);
    let (b, be) = two_sum(a, v.0);
    -(b + (be + ae + (c.1 + pl + v.1)))
}

/// `Li₂(x)` (its real part for `x > 1`) of an `f64`.
pub(crate) fn dilog_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < -1.0 {
        if x == f64::NEG_INFINITY {
            return x;
        }
        // −π²/6 − ½ ln²(−x) − Li₂(1/x), every part a double-double
        return neg_sum_half_sq((PI2_6_HI, PI2_6_LO), ln_dd(-x), dilog_series_dd(1.0 / x));
    }
    if x <= 0.75 {
        let (h, l) = dilog_series_dd(x);
        return h + l;
    }
    if x <= 2.0 {
        // π²/6 − ln x ln|1 − x| − Li₂(1 − x), 1 − x exact, the logarithms and their product as
        // double-doubles
        if x == 1.0 {
            return PI2_6_HI;
        }
        let u = 1.0 - x;
        let (lx, lxl) = ln_dd(x);
        let (lu, lul) = ln_dd(u.abs());
        let (p, pe) = two_prod(lx, lu);
        let pl = pe + (lx * lul + lxl * lu);
        let (v, vl) = dilog_series_dd(u);
        let (a, ae) = two_sum(PI2_6_HI, -p);
        let (b, be) = two_sum(a, -v);
        return b + (be + ae + (PI2_6_LO - pl - vl));
    }
    if (10.5..=15.0).contains(&x) {
        let (t, tl) = two_sum(x - DILOG_X0_HI, -DILOG_X0_LO);
        let g = DILOG_ZERO[0] + (DILOG_ZERO_LO + poly_tail(t, &DILOG_ZERO));
        return mla(t, g, tl * g);
    }
    if x == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    // π²/3 − ½ ln² x − Li₂(1/x)
    let (v, vl) = dilog_series_dd(1.0 / x);
    neg_sum_half_sq((-PI2_3_HI, -PI2_3_LO), ln_dd(x), (v, vl))
}

// ---------------------------------------------------------------------------------------------
// The polylogarithm of a real order.
//
// * `2⁻ˢ|x| < 2⁻⁶⁰`: `x` (the rest of `x + x²2⁻ˢ + …` is below a 2⁻⁵⁹ of it; past `|x| = 1`
//   the divergent series is asymptotic to `Li_s` and its least term far smaller).
// * `|x| ≤ 3/4` (every `|x| ≤ 1` for `s ≥ 20`; for `s < 0` see below): the
//   series `Σ xᵏ k⁻ˢ`, `p⁻ˢ` of the primes from `ln p` as a double-double and the others by
//   products; for `s < −1` each term from logarithms (`polylog_series_log`), so that nothing
//   leaves the range before the sum does.
// * `3/4 < x < 1` (`1/2 < x < 1` for `s < 0`):
//   `Li_s(eᵘ) = Γ(1 − s)(−μ)^(s−1) + Σ ζ(s − k) μᵏ/k!`, `μ = ln x` a double-double; for
//   `n = 1` (`n` the integer nearest `s ≥ 1/2`) and near the integers past it the pole of
//   `ζ(s − k)` at `k = n − 1` taken together with that of `Γ(1 − s)` in closed form, so that `s`
//   near an integer costs nothing; `ζ(s − k)` past `s − k < −1/2` by the reflection formula as a
//   recurrence in `k`; for `s < −20` the first term alone (the sum is below `10⁻²⁰` of it),
//   `ln Γ` in double-double.
// * `−2 ≤ x < −3/4`: `Li_s(−eᵘ) = −Σ η(s − k) μᵏ/k!`, `μ = ln(−x)`, `η` Dirichlet's eta.
// * `x < −2`, `s > 0`: for an integer `s = n ≥ 1` the inversion
//   `Li_n(x) = −(−1)ⁿ Li_n(1/x) − Σ 2η(2j) μⁿ⁻²ʲ/(n − 2j)!` (μ a double-double, the sum nested
//   in double-double); otherwise, for an order large against `ln|x|`, the divergent series
//   `Σ xᵏ k⁻ˢ` cut at its least term when that converges, Sommerfeld's expansion past
//   `x = −e^37.7` for `s < ln|x|/4`, and else the integral of Fermi and Dirac
//   `Li_s(−eᵘ) = −F(μ)/Γ(s)`, `F = ∫₀^∞ t^(s−1)/(1 + e^(t−μ)) dt`, all of whose terms are
//   positive, by Gauss–Legendre pieces.
// * `s < 0`, `x < 0` (for an integer `s` with `x < −1` by `Li_s(x) = (−1)^(s+1) Li_s(1/x)`,
//   `ln(−1/x) = −μ` exactly): the series, its terms double-doubles, where `|x| ≤ 1/2` and no term
//   is above about 4000 times the first (`|x|` near `2^(−|s|)`, where the first two cancel, among
//   them); else the Hurwitz zeta function of the complex `a = 1/2 − iμ/2π`,
//   `Li_s(x) = 2 Γ(1 − s)(2π)^(s−1) Re[i^(1−s) ζ(1 − s, a)]`, whose Euler–Maclaurin terms decay
//   (its real part can cancel by more than `Li_s` is ill-conditioned: by 50 at
//   `Li₋₇₅(−1.6·10⁻²³)`, where the series serves).
// * An integer `−18 ≤ s ≤ −2`, `0 < x < 1`: `x Aₙ(x)/(1 − x)ⁿ⁺¹`, `Aₙ` Eulerian, in double-double.

/// `ln p` of the primes below 256 as double-doubles.
const LN_PRIMES: [(f64, f64); 54] = [
    (LN_2, 2.3190468138462996e-17),
    (1.0986122886681098, -9.07129723500153e-17),
    (1.6094379124341003, 9.280081691085902e-17),
    (1.9459101490553132, 7.323586207904907e-17),
    (2.3978952727983707, -1.253584211423161e-16),
    (2.5649493574615367, -2.5580975097208856e-18),
    (2.833213344056216, -8.500696635386325e-17),
    (2.9444389791664403, 1.9776172119535626e-16),
    (3.1354942159291497, 1.5758359867283186e-17),
    (3.367295829986474, -1.1553104240685565e-16),
    (3.4339872044851463, -2.5863763694297672e-17),
    (3.6109179126442243, 9.643918385970854e-17),
    (3.713572066704308, -1.4628004942704776e-16),
    (3.7612001156935624, 3.879508085699725e-17),
    (3.8501476017100584, 1.5534199491592608e-16),
    (3.970291913552122, -1.3098078310631424e-16),
    (4.07753744390572, -2.5451232614395796e-16),
    (4.110873864173311, -1.0515756814070423e-16),
    (4.204692619390966, 3.3533479800665464e-16),
    (4.2626798770413155, -3.579070851354337e-17),
    (4.290459441148391, 1.7802138532613648e-16),
    (4.3694478524670215, -3.884474562096486e-18),
    (4.418840607796598, -3.397594828607531e-16),
    (4.48863636973214, 1.9677598067823275e-16),
    (4.574710978503383, 3.2451073313026223e-18),
    (4.61512051684126, -2.2821302194740364e-16),
    (4.634728988229636, -8.908374881679555e-17),
    (4.672828834461906, 3.951628993491307e-16),
    (4.6913478822291435, 2.4925018736917553e-16),
    (4.727387818712341, -2.4819144458625176e-16),
    (4.844187086458591, 7.705535583096494e-17),
    (4.875197323201151, 3.658536926412979e-16),
    (4.919980925828125, -1.581360567345527e-16),
    (4.9344739331306915, 2.6517822939775615e-16),
    (5.003946305945459, -5.3227464831388117e-17),
    (5.017279836814924, 4.1216038884402694e-17),
    (5.056245805348308, 3.5935613438051566e-16),
    (5.093750200806762, 4.700813297634134e-18),
    (5.117993812416755, -3.3192770383594804e-16),
    (5.153291594497779, -3.4631559525612803e-16),
    (5.187385805840755, 6.965661887732005e-17),
    (5.198497031265826, -3.269812482571168e-16),
    (5.25227342804663, -1.554941592972122e-17),
    (5.262690188904886, -5.017318535581812e-17),
    (5.2832037287379885, 1.5220334329004983e-17),
    (5.293304824724492, 1.3191103416168333e-16),
    (5.351858133476067, -1.5217360259495014e-16),
    (5.407171771460119, -8.894925804562492e-17),
    (5.424950017481403, -2.5572791500951425e-16),
    (5.43372200355424, -3.0588478608482483e-16),
    (5.4510384535657, 4.0607096949073963e-16),
    (5.476463551931511, -3.53180098108935e-16),
    (5.484796933490655, 2.1543628446067385e-16),
    (5.5254529391317835, 4.054435010670984e-16),
];

/// The smallest prime factor of each `k ≤ 256` as its index in `LN_PRIMES` (`u8::MAX` for 0, 1).
const SPF: [u8; 257] = {
    let mut t = [u8::MAX; 257];
    let mut i = 0;
    let mut k = 2;
    while k <= 256 {
        let mut p = 2;
        while k % p != 0 {
            p += 1;
        }
        if p == k {
            t[k] = i;
            i += 1;
        } else {
            t[k] = t[p];
        }
        k += 1;
    }
    t
};

/// `(ζ(k) − 1)/k` for `k = 2, …, 29`.
const ZETA_M1_K: [f64; 28] = [
    0.3224670334241132,
    0.0673523010531981,
    0.020580808427784546,
    0.007385551028673986,
    0.0028905103307415234,
    0.001192753911703261,
    0.0005096695247430425,
    0.00022315475845357939,
    9.945751278180853e-05,
    4.492623673813314e-05,
    2.050721277567069e-05,
    9.439488275268397e-06,
    4.374866789907488e-06,
    2.039215753801366e-06,
    9.55141213040742e-07,
    4.492469198764566e-07,
    2.1207184805554665e-07,
    1.0043224823968099e-07,
    4.7698101693639804e-08,
    2.2711094608943164e-08,
    1.0838659214896955e-08,
    5.183475041970047e-09,
    2.4836745438024785e-09,
    1.1921401405860912e-09,
    5.731367241678862e-10,
    2.7595228851242334e-10,
    1.330476437424449e-10,
    6.4229645638381e-11,
];

/// The Euler–Mascheroni constant.
const EULER: f64 = 0.577_215_664_901_532_9;

/// `p^(−s)` from `ln p` as a double-double.
#[inline(always)]
fn prime_pow(lp: (f64, f64), s: f64) -> f64 {
    let (a, ae) = two_prod(-s, lp.0);
    exp_mul(a, 1.0, ae - s * lp.1)
}

/// `Σ_{k≥1} xᵏ k⁻ˢ`, to convergence: past the largest term and below `2⁻⁶⁰` of the sum.
fn polylog_series(s: f64, x: f64) -> f64 {
    let mut pw = [0.0f64; 257];
    let mut pp = [f64::NAN; 54];
    pw[1] = 1.0;
    let peak = if s < 0.0 { -s / -x.abs().ln() } else { 0.0 };
    let (mut hi, mut lo) = (x, 0.0);
    let mut xk = x;
    let mut k = 2usize;
    while k <= 4000 {
        xk *= x;
        let kp = if k <= 256 {
            let i = SPF[k] as usize;
            if pp[i].is_nan() {
                pp[i] = prime_pow(LN_PRIMES[i], s);
            }
            let v = pp[i] * pw[k / PRIMES[i] as usize];
            pw[k] = v;
            v
        } else {
            prime_pow(ln_dd(k as f64), s)
        };
        let term = xk * kp;
        let (h, e) = two_sum(hi, term);
        hi = h;
        lo += e;
        if (term.abs() <= 1e-18 * hi.abs() && k as f64 > peak) || xk == 0.0 {
            break;
        }
        k += 1;
    }
    hi + lo
}

/// The primes below 256.
const PRIMES: [u16; 54] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
];

/// `ζ(s − k)/k!` for `k = 0, 1, …` (and, with `eta`, `η(s − k)/k! = (1 − 2^(1−s+k)) ζ(s − k)/k!`),
/// skipping the pole at `s − k = 1` when `skip` is that `k`. Past `s − k < −1/2` by the
/// reflection `ζ(t) = 2 (2π)^(t−1) sin(πt/2) Γ(1 − t) ζ(1 − t)` as a recurrence in `k`.
struct ZetaCoeffs {
    s: f64,
    k: usize,
    fact: f64,
    // the reflection: g = (2π)^(t−1) Γ(1 − t)/k! of t = s − k, and sin(πs/2), cos(πs/2)
    g: f64,
    sin: f64,
    cos: f64,
    eta: bool,
}

impl ZetaCoeffs {
    fn new(s: f64, eta: bool) -> Self {
        let (sin, cos) = sincos_pi(0.5 * s);
        ZetaCoeffs {
            s,
            k: 0,
            fact: 1.0,
            g: f64::NAN,
            sin,
            cos,
            eta,
        }
    }

    /// The next coefficient and its index.
    fn next(&mut self, skip: Option<usize>) -> (usize, f64) {
        let k = self.k;
        let kf = k as f64;
        if k > 0 {
            self.fact *= kf;
        }
        let t = self.s - kf;
        let c = if Some(k) == skip {
            0.0
        } else if t >= -0.5 {
            if self.eta {
                eta_right(t) / self.fact
            } else {
                zeta_right(t) / self.fact
            }
        } else {
            let u = 1.0 - t; // 1 − t > 1.5
            if self.g.is_nan() {
                self.g = gamma_f64(u) * pow_2pi_neg(u) / self.fact;
            } else {
                // from t + 1 to t: (2π)^(−1) (1 − t − 1)… × (1 − t − 1 + 1)/k
                self.g *= (u - 1.0) / (2.0 * PI_HI * kf);
            }
            // sin(π(s − k)/2) by k mod 4
            let sn = match k & 3 {
                0 => self.sin,
                1 => -self.cos,
                2 => -self.sin,
                _ => self.cos,
            };
            let z = 2.0 * sn * self.g * zeta_right(u);
            if self.eta {
                // 1 − 2^(1−t) = −expm1((1 − t) ln 2)
                -(u * LN_2).exp_m1() * z
            } else {
                z
            }
        };
        self.k += 1;
        (k, c)
    }
}

/// `η(t) = (1 − 2^(1−t)) ζ(t)` for `t ≥ −1/2`, without the cancellation at the pole.
fn eta_right(t: f64) -> f64 {
    if t <= 0.5 || t >= 3.0 {
        return -((1.0 - t) * LN_2).exp_m1() * zeta_right(t);
    }
    // ln 2 · expm1(a)/a − expm1(a) R(t), a = (1 − t) ln 2
    let a = (1.0 - t) * LN_2;
    let e = a.exp_m1();
    let q = if a == 0.0 { 1.0 } else { e / a };
    LN_2 * q - e * zeta_regular(t)
}

/// The pair `Γ(1 − s)(−μ)^(s−1) + ζ(1 + ε) μⁿ⁻¹/(n − 1)!` for `s = n + ε`, `n ≥ 1`, `|ε| ≤ 1/2`,
/// `μ = mu + mul < 0` a double-double: `μⁿ⁻¹/(n − 1)! (R(1 + ε) − E)`, `E = expm1(εG)/ε`,
/// `G = ln Γ(1 − ε)/ε + ln(−μ) − ln Πⱼ(1 + ε/j)/ε`, `ε ln(−μ)` as a double-double (near `x = 1`
/// `ln(−μ)` reaches −37, and its rounding would cost several ulps).
fn pole_pair(n: usize, eps: f64, mu: f64, mul: f64) -> f64 {
    let (lmu, lmul) = ln_dd(-mu);
    let lmul = lmul + mul / mu;
    // ln Γ(1 − ε)/ε = −ln(1 − ε)/ε − (1 − γ) + Σ (ζ(k) − 1)/k εᵏ⁻¹
    let l1 = if eps.abs() < 1e-9 {
        1.0 + 0.5 * eps
    } else {
        -(-eps).ln_1p() / eps
    };
    let lg = l1 - (1.0 - EULER) + eps * horner(eps, &ZETA_M1_K);
    // d = (Πⱼ(1 + ε/j) − 1)/ε by the recurrence d_j = d_{j−1} + Q_{j−1}/j
    let (mut d, mut q) = (0.0, 1.0);
    for j in 1..n {
        d += q / j as f64;
        q = 1.0 + eps * d;
    }
    let z = eps * d;
    let lp = if z.abs() < 1e-9 {
        d * (1.0 - 0.5 * z)
    } else {
        d * (z.ln_1p() / z)
    };
    let e = if eps == 0.0 {
        lg + lmu - lp
    } else {
        // εG = ε(lg − lp) + ε ln(−μ), and expm1 of the double-double by its first-order term
        let (p, pe) = two_prod(eps, lmu);
        let (g, gl) = two_sum(p, eps * (lg - lp));
        let gl = gl + pe + eps * lmul;
        let em = g.exp_m1();
        (em + gl * (1.0 + em)) / eps
    };
    let mut c = 1.0;
    for j in 1..n {
        c *= mu / j as f64;
    }
    // the low part of μ in μⁿ⁻¹, to first order
    let c = c * (1.0 + (n - 1) as f64 * (mul / mu));
    c * (zeta_regular(1.0 + eps) - e)
}

/// `ln x` as a double-double for `1/2 ≤ x ≤ 2`, to about `2⁻⁶⁹` relative to it (`2⁻⁸⁴` below
/// `|d| = 2⁻⁶`, `2⁻⁹⁶` below `2⁻¹⁶`): the series of `ln(1 + d)`, `d = x − 1` exact, for
/// `|d| < 2⁻¹⁶`, `ln_dd_acc` (absolute error `2⁻¹⁰⁰`) to `2⁻⁶`, `ln_dd` (absolute error `2⁻⁷⁵`,
/// where `|ln x| > 2⁻⁶`) past it.
fn ln_near_one(x: f64) -> (f64, f64) {
    let d = x - 1.0;
    if d.abs() < 1.525_878_906_25e-5 {
        // d − d²/2 + d³(1/3 − d/4 + d²/5 − d³/6), the first two exact
        let (d2, d2e) = two_prod(d, d);
        let rest = d2 * d * (1.0 / 3.0 - d * (0.25 - d * (0.2 - d * (1.0 / 6.0))));
        let (h, l) = two_sum(d, -0.5 * d2);
        let (h, l) = two_sum(h, l + (rest - 0.5 * d2e));
        (h, l)
    } else if d.abs() < 0.015_625 {
        ln_dd_acc(x)
    } else {
        ln_dd(x)
    }
}

/// `Γ(1 − s)(−μ)^(s−1)` for `μ = mu + mul < 0`, rounded once: `1 − s` exactly as a double-double
/// (its low part through `ψ`), the exponent `(s − 1) ln(−μ)` as a double-double, and from
/// `1 − s = 15` `ln Γ(1 − s)` in it (so that a result past the range is an infinity).
fn gamma_mu_pow(s: f64, mu: f64, mul: f64) -> f64 {
    let (l, ll) = ln_dd(-mu);
    let ll = ll + mul / mu;
    let (sm1, sm1l) = two_sum(s, -1.0);
    let (a, ae) = two_prod(sm1, l);
    let al = ae + sm1 * ll + sm1l * l;
    let (z, zl) = two_sum(1.0, -s);
    if z >= 15.0 {
        let g = lgamma_dd(z, zl);
        let (b, be) = two_sum(a, g.hi);
        return exp_mul(b, 1.0, be + (al + g.lo));
    }
    let g = gamma_f64(z);
    // zl is zero but for s < 1/2, where z > 1/2
    let gl = if zl == 0.0 {
        0.0
    } else {
        g * zl * digamma_rough(z)
    };
    exp_mul(a, g, gl + g * al)
}

/// `Li_s(eᵘ)` for `1/2 < x = eᵘ < 1`, `μ = mu + mul` a double-double.
fn polylog_near_one(s: f64, mu: f64, mul: f64) -> f64 {
    if s < -20.0 {
        // the sum's terms are below (|μ|/2π)^(1−s) < 10⁻²⁰ of the first
        return gamma_mu_pow(s, mu, mul);
    }
    let n = s.round();
    let eps = s - n;
    // the pole pair (n the integer nearest s ≥ 1/2) for n = 1, where ζ(s) has the pole, and near
    // the integers past it; elsewhere its two parts are moderate and the pair's own roundings
    // would cost more
    let (skip, pair) = if s >= 0.5 && (n == 1.0 || eps.abs() < 0.25) {
        let n = n as usize;
        (Some(n - 1), pole_pair(n, eps, mu, mul))
    } else {
        (None, gamma_mu_pow(s, mu, mul))
    };
    if !pair.is_finite() {
        // past the range (s < 0 near x = 1, where Γ(1 − s)(−μ)^(s−1) overflows): +∞, the other
        // terms finite and far below it (added to the infinity they made its low part NaN)
        return pair;
    }
    let mut zc = ZetaCoeffs::new(s, false);
    let (mut hi, mut lo) = (pair, 0.0);
    let mut mk = 1.0;
    let mut small = 0;
    loop {
        let (k, c) = zc.next(skip);
        let term = c * mk;
        let (h, e) = two_sum(hi, term);
        hi = h;
        lo += e;
        // two small terms in a row (the trivial zeros make every other one vanish)
        small = if term.abs() <= 1e-18 * hi.abs() {
            small + 1
        } else {
            0
        };
        if (k as f64 > s + 1.0 && small >= 2) || k >= 400 || !hi.is_finite() {
            break;
        }
        mk *= mu;
    }
    hi + lo
}

/// `Li_s(−eᵘ)` for `|μ| ≤ ln 2`: `−Σ η(s − k) μᵏ/k!`.
fn polylog_near_minus_one(s: f64, mu: f64) -> f64 {
    let mut zc = ZetaCoeffs::new(s, true);
    let (mut hi, mut lo) = (0.0, 0.0);
    let mut mk = 1.0;
    let mut small = 0;
    loop {
        let (k, c) = zc.next(None);
        let term = c * mk;
        let (h, e) = two_sum(hi, term);
        hi = h;
        lo += e;
        small = if term.abs() <= 1e-18 * hi.abs() {
            small + 1
        } else {
            0
        };
        if (k as f64 > s + 1.0 && small >= 2) || k >= 400 || mu == 0.0 {
            break;
        }
        mk *= mu;
    }
    -(hi + lo)
}

/// `Li_n(x)` for an integer `n ≥ 1` and `x < −1`: `−(−1)ⁿ Li_n(1/x) − P(μ)`,
/// `P(μ) = Σ_{j≤n/2} 2η(2j) μⁿ⁻²ʲ/(n − 2j)!` (`2η(0) = 1`), all of whose terms are positive, as
/// `(μⁿ/n!) [1 + r₁(2η(2) + r₂(2η(4) + …))]`, `rⱼ = (n − 2j + 2)(n − 2j + 1)/μ²`, nested from the
/// inside in double-double (`μ` a double-double), and `μⁿ/n!` from its logarithm: linear in `n`,
/// and no factorial leaves the range.
fn polylog_inverse_int(n: usize, x: f64) -> f64 {
    let (mu, mul) = ln_dd(-x);
    let m2 = Dd { hi: mu, lo: mul }.mul(Dd { hi: mu, lo: mul });
    let jmax = n / 2;
    let two_eta = |j: usize| -> f64 {
        if j == 0 {
            1.0
        } else {
            2.0 * eta_right(2.0 * j as f64)
        }
    };
    let mut q = Dd::new(two_eta(jmax));
    for j in (1..=jmax).rev() {
        let r = Dd::new(((n - 2 * j + 2) * (n - 2 * j + 1)) as f64).div(m2);
        q = r.mul(q).add_f64(two_eta(j - 1));
    }
    // μⁿ/n! = e^(n ln μ − ln n!)
    let (l, ll) = ln_dd(mu);
    let ll = ll + mul / mu;
    let nf = n as f64;
    let (a, ae) = two_prod(nf, l);
    let e = Dd {
        hi: a,
        lo: ae + nf * ll,
    }
    .sub(lgamma_dd(nf + 1.0, 0.0));
    let p = exp_mul(e.hi, q.hi, q.lo + q.hi * e.lo);
    let inner = polylog_real(nf, 1.0 / x);
    let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
    sign * inner - p
}

/// The nodes (the positive half) and weights of the 16-point Gauss–Legendre rule on `[−1, 1]`.
const GL16_X: [f64; 8] = [
    0.09501250983763744,
    0.2816035507792589,
    0.45801677765722737,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];
const GL16_W: [f64; 8] = [
    0.1894506104550685,
    0.18260341504492358,
    0.16915651939500254,
    0.14959598881657674,
    0.12462897125553388,
    0.09515851168249279,
    0.062253523938647894,
    0.027152459411754096,
];

/// `Li_s(x)` for `x < −2` and a non-integer `s > 0` by the integral of Fermi and Dirac:
/// `Li_s(−eᵘ) = −F(μ)/Γ(s)`, `F(μ) = ∫₀^∞ t^(s−1)/(1 + e^(t−μ)) dt`, every term positive.
/// On `[0, a]`, `a = min(1, μ)`, the Taylor series of the logistic `1/(1 + e^(t−μ))` (its
/// coefficients by the recurrence of `f′ = f² − f`) integrated against `t^(s−1)` exactly; past
/// it 16-point Gauss–Legendre on pieces of at most twice their left end (for the branch point of
/// `t^(s−1)` at 0) and at most 3 long within 6 of `μ` (the poles of the logistic at
/// `μ ± iπ(2k + 1)`), 6 long beyond, until a piece adds less than `2⁻⁶³` of the sum past the
/// mass of `t^(s−1) e^(−t)`. For a large order (`s > 21`) the powers at the nodes come from
/// double-double logarithms, and where `t^(s−1)` would leave the range the integrand is scaled by
/// `e^(−c)` and the result by `e^(c − ln Γ(s))`, `ln Γ` in double-double.
fn polylog_fermi_dirac(s: f64, x: f64) -> f64 {
    // μ = ln(−x) as a double-double: its rounding alone would cost several ulps of Li_s
    let (mu, mul) = ln_dd(-x);
    let sm1 = s - 1.0;
    // the scale: t^(s−1) e^(−c) stays finite up to the last piece
    let tmax = mu.max(sm1).max(1.0) + 60.0 + 10.0 * s.sqrt();
    let big = sm1 * tmax.ln() > 690.0;
    let c = if big {
        (sm1 * tmax.ln()).floor() - 600.0
    } else {
        0.0
    };
    // m^(s−1) e^(−c), once a piece
    let pw = |t: f64| -> f64 {
        if big {
            let (l, ll) = ln_dd(t);
            let (a, ae) = two_prod(sm1, l);
            exp_mul(a - c, 1.0, ae + sm1 * ll)
        } else if s >= 1.0 {
            t.powf(sm1)
        } else {
            t.powf(s) / t
        }
    };
    // e^(t − μ) for an exact t, to double-double accuracy in its argument
    let e_minus_mu = |t: f64| -> f64 {
        let (d, dl) = two_sum(t, -mu);
        d.exp() * (1.0 + (dl - mul))
    };
    // [0, a]: Σ cₖ a^(s+k)/(s + k), f = Σ cₖ tᵏ, (k + 1) cₖ₊₁ = Σ cᵢ cₖ₋ᵢ − cₖ
    let a = mu.min(1.0);
    let q = e_minus_mu(0.0);
    let mut cf = [0.0f64; 64];
    cf[0] = 1.0 / (1.0 + q);
    let (mut hi, mut lo) = (0.0, 0.0);
    let mut ak = 1.0;
    let mut k = 0;
    loop {
        let term = cf[k] * ak / (s + k as f64);
        let (h, e) = two_sum(hi, term);
        hi = h;
        lo += e;
        if k >= 63 || (k > 4 && term.abs() <= 1e-19 * hi.abs()) {
            break;
        }
        let mut conv = 0.0;
        for i in 0..=k {
            conv += cf[i] * cf[k - i];
        }
        cf[k + 1] = (conv - cf[k]) / (k + 1) as f64;
        ak *= a;
        k += 1;
    }
    let head = (hi + lo) * pw(a) * a;
    let (mut hi, mut lo) = (head, 0.0);
    // (t/m)^(s−1) at a node t = m ± d: for a large order from (s − 1)(ln t − ln m) as a
    // double-double (t = m ± d exact as a pair, so that the power and e^(±d) take the same
    // node): in double, the rounding of (s − 1) ln(1 ± d/m), up to 160 at s = 555, cost up to
    // 2000 ulps
    let accurate = sm1 > 20.0;
    let node_pow = |m: f64, lm: Dd, d: f64| -> f64 {
        if accurate {
            let (th, tl) = two_sum(m, d);
            let (l, ll) = ln_dd(th);
            let q = Dd {
                hi: l,
                lo: ll + tl / th,
            }
            .sub(lm);
            let (p, pe) = two_prod(sm1, q.hi);
            exp_mul(p, 1.0, pe + sm1 * q.lo)
        } else {
            (sm1 * (d / m).ln_1p()).exp()
        }
    };
    // the pieces past a, each about its exact midpoint m: t = m ± d,
    // t^(s−1) = m^(s−1) (t/m)^(s−1), e^(t−μ) = e^(m−μ) e^(±d)
    let mut b = a;
    let mut pieces = 0;
    loop {
        // 3 long where a piece of 6 would come within 6 of μ
        let len: f64 = if b + 6.0 > mu - 6.0 && b < mu + 6.0 {
            3.0
        } else {
            6.0
        };
        let e = b + len.min(2.0 * b);
        let (m, h) = (0.5 * (b + e), 0.5 * (e - b));
        let (pm, em) = (pw(m), e_minus_mu(m));
        let lm = if accurate {
            let (l, ll) = ln_dd(m);
            Dd { hi: l, lo: ll }
        } else {
            Dd::new(0.0)
        };
        let mut sum = 0.0;
        for i in 0..8 {
            let d = h * GL16_X[i];
            let g1 = node_pow(m, lm, -d);
            let g2 = node_pow(m, lm, d);
            let ed = d.exp();
            let f1 = g1 / (1.0 + em / ed);
            let f2 = g2 / (1.0 + em * ed);
            sum += GL16_W[i] * (f1 + f2);
        }
        let v = h * pm * sum;
        let (s2, e2) = two_sum(hi, v);
        hi = s2;
        lo += e2;
        pieces += 1;
        if (e > mu.max(sm1) + 6.0 && v <= 1e-19 * hi) || pieces >= 400 || e >= tmax + 400.0 {
            break;
        }
        b = e;
    }
    let f = hi + lo;
    if big {
        // e^(c − ln Γ(s)) with ln Γ in double-double: in double its rounding (ln Γ(555) ≈ 2950)
        // cost up to 1000 ulps
        let e = Dd::new(c).sub(lgamma_dd(s, 0.0));
        exp_mul(e.hi, -f, -f * e.lo)
    } else {
        -f / gamma_f64(s)
    }
}

/// `ln |w|` and `arg w` of `w = re + i·im`, `re > 0` and `im` a double-double, as
/// double-doubles.
fn ln_arg(re: f64, im: Dd) -> (Dd, Dd) {
    let (r2, r2e) = two_prod(re, re);
    let q = Dd { hi: r2, lo: r2e }.add(im.mul(im));
    let (l, ll) = ln_dd(q.hi);
    let lam = Dd {
        hi: l,
        lo: ll + q.lo / q.hi,
    }
    .mul_f64(0.5);
    (lam, atan_dd(im.div(Dd::new(re))))
}

/// `e^(m) e^(2πi φ)` of the double-doubles `m` and `φ` (turns).
#[inline]
fn polar_dd(m: Dd, ph: Dd) -> (f64, f64) {
    let r = exp_mul(m.hi, 1.0, m.lo);
    let f = ph.hi - ph.hi.round();
    let (fh, fl) = two_sum(f, ph.lo);
    let (sn, cs) = sincos_pi(2.0 * fh);
    let d = 2.0 * PI_HI * fl;
    (r * (cs - d * sn), r * (sn + d * cs))
}

/// `Li_s(x)` for `s < 0` and `x < 0`, given `μ = mu + mul = ln(−x)` as a double-double, by the
/// Hurwitz zeta function of the complex `a = 1/2 − iμ/2π`:
/// `Li_s(x) = 2 Γ(1 − s) (2π)^(s−1) Re[i^(1−s) ζ(1 − s, a)]`, whose Euler–Maclaurin terms decay
/// (`σ = 1 − s > 1`); its real part cancels near the real zeros of `Li_s`, and also where `|x|` is
/// near `2^(±|s|)` (by about 50 for an integer order, whose `κ` is 1 there: the series takes
/// those). Each term `(n + a)^(−σ)` from `ln |n + a|` and `arg(n + a)` as double-doubles,
/// relative to `|a|^(−σ)`, whose logarithm joins that of `2 Γ(1 − s)(2π)^(s−1)` (`ln Γ` in
/// double-double) in one exponent, so that nothing leaves the range before the last product;
/// the direct terms stop where the rest is below `2⁻⁶⁰` of the first. The remainder past them,
/// the pole's parts below and `Re[i^(1−s) Z]` are complex double-doubles (`σ − 1 = −s` exactly),
/// rounded once in the last product (for `s → −1` its real part cancels by `1/sin(π|s|)`, about
/// `κ/π`). For `−1/4 < s < 0` the pole of `ζ(σ, a)` at
/// `σ = 1` is taken out of the sum, `(w^(1−σ) − 1)/(σ − 1) = −expm1(s ln w)/s`, and its part
/// `Re[i^(1−s)/(σ − 1)] = −sin(πs/2)/s` added in closed form: the two would cancel to `O(1)` from
/// parts of size `1/|s|`. Further left it is not needed and the plain remainder
/// `w^(1−σ)/(σ − 1)` serves: there the pair is of size `|a|^(1−s)` (in units of `|a|^(−σ)`)
/// against a result of `|a| sin(π|s|)/|s|`, and with its parts in double it lost up to 84 `κ`
/// (`Li₋₀.₉₉₅(−10²⁸⁷)`).
fn polylog_hurwitz_form(s: f64, mu: f64, mul: f64) -> f64 {
    // a = 1/2 + i ai, ai = −μ/2π as a double-double (its rounding would be a perturbation of x
    // by |μ| ulps)
    let aid = Dd { hi: mu, lo: mul }.mul(INV_2PI_DD).neg();
    let ai = aid.hi;
    let (sg, sgl) = two_sum(1.0, -s);
    let sigma = Dd { hi: sg, lo: sgl };
    let pole = s > -0.25;
    let (l0, _) = ln_arg(0.5, aid);
    let abs_a = 0.5f64.hypot(ai);
    let need = (sg + 40.0) / 2.83;
    let big_n = if abs_a >= need {
        0.0
    } else {
        (need - abs_a).ceil()
    };
    // Σ (n + a)^(−σ) |a|^σ, n from 0
    let mut acc_z = (0.0, 0.0, 0.0, 0.0);
    let term = |re: f64| -> ((f64, f64), Dd, Dd) {
        let (lw, tw) = ln_arg(re, aid);
        let m = sigma.mul(lw.sub(l0)).neg();
        let ph = sigma.mul(tw).neg().mul(INV_2PI_DD);
        (polar_dd(m, ph), m, lw)
    };
    let mut n = 0.0;
    let mut truncated = false;
    while n < big_n {
        let (v, m, lw) = term(n + 0.5);
        add_dd(&mut acc_z, (Dd::new(v.0), Dd::new(v.1)));
        // past n the series is at most |n + a|^(−σ)(1 + |n + a|/(σ − 1)) of |a|^(−σ)
        if n > 0.0 && m.hi + (1.0 + lw.hi.exp() / (sg - 1.0)).ln() < -42.0 {
            truncated = true;
            break;
        }
        n += 1.0;
    }
    // |a|^σ, the scale of the pole's parts (whose direct sum is never cut short: its rest decays
    // only as |n + a|^s)
    let k = if pole {
        let e = sigma.mul(l0);
        let (h, l) = exp_dd(e.hi, e.lo);
        Dd { hi: h, lo: l }
    } else {
        Dd::new(0.0)
    };
    if !truncated {
        // the remainder at w = N + a: w^(−σ) (1/2 + Σ B₂ₖ/(2k)! (σ)₂ₖ₋₁ w^(1−2k)) + w^(1−σ)/(σ − 1),
        // σ − 1 = −s exactly; w^(−σ)|a|^σ and the products as complex double-doubles (the
        // Bernoulli terms, a small part of it, in double)
        let wr = big_n + 0.5;
        let (lw, tw) = ln_arg(wr, aid);
        let p = polar_cd(
            sigma.mul(lw.sub(l0)).neg(),
            sigma.mul(tw).neg().mul(INV_2PI_DD),
        );
        let w = Complex::new(wr, ai);
        let r = Complex::new(1.0, 0.0) / w;
        let r2 = r * r;
        let mut fac = r * sg;
        let mut acc = Complex::new(0.0, 0.0);
        let mut k2 = 1.0;
        for &b in BERNOULLI_2K.iter() {
            let t = fac * b;
            acc += t;
            if t.re.abs() + t.im.abs() <= 1e-19 * (0.5 + acc.re.abs() + acc.im.abs()) {
                break;
            }
            fac = fac * r2 * ((sg + k2) * (sg + k2 + 1.0));
            k2 += 2.0;
        }
        let mut ar = Dd::new(0.5).add_f64(acc.re);
        let mut aim = Dd::new(acc.im);
        if !pole {
            let ms = Dd::new(-s);
            ar = ar.add(Dd::new(wr).div(ms));
            aim = aim.add(aid.div(ms));
        }
        let tail = cmul_dd(p, (ar, aim));
        add_dd(&mut acc_z, tail);
        if pole {
            // (w^(1−σ) − 1)/(σ − 1) = −expm1(s ln w)/s, times |a|^σ
            let q = pole_remainder(s, lw, tw);
            add_dd(&mut acc_z, (q.0.mul(k), q.1.mul(k)));
        }
    }
    let zr = Dd {
        hi: acc_z.0,
        lo: acc_z.1,
    };
    let zi = Dd {
        hi: acc_z.2,
        lo: acc_z.3,
    };
    // Re[i^(1−s) Z], i^(1−s) = e^(iπσ/2), and the pole's −sin(πs/2)/s times |a|^σ, as
    // double-doubles (for s → −1 they cancel by up to κ/π)
    let (cs, sn) = turn_dd(0.25 * sg, 0.25 * sgl);
    let mut re = cs.mul(zr).sub(sn.mul(zi));
    if pole && !truncated {
        re = re.sub(sin_half_pi_over_dd(s).mul(k));
    }
    // × 2 Γ(1 − s)(2π)^(s−1) |a|^(−σ) = e^(ln 2 + ln Γ(σ) − σ (ln 2π + ln |a|))
    let e = lgamma_dd(sg, sgl)
        .add(LN_2_DD)
        .sub(sigma.mul(LN_2PI_DD.add(l0)));
    exp_mul(e.hi, re.hi, re.lo + re.hi * e.lo)
}

/// Adds the complex double-double `v` to the sum `(re, re low, im, im low)`.
#[inline(always)]
fn add_dd(z: &mut (f64, f64, f64, f64), v: Cd) {
    let (h, e) = two_sum(z.0, v.0.hi);
    z.0 = h;
    z.1 += e + v.0.lo;
    let (h, e) = two_sum(z.2, v.1.hi);
    z.2 = h;
    z.3 += e + v.1.lo;
}

/// `e^m e^(2πiφ)` of the double-doubles `m` and `φ` (turns), as a complex double-double (to
/// about `2⁻⁶²` of its modulus).
#[inline]
fn polar_cd(m: Dd, ph: Dd) -> Cd {
    let (h, l) = exp_dd(m.hi, m.lo);
    let r = Dd { hi: h, lo: l };
    let (c, sn) = turn_dd(ph.hi, ph.lo);
    (r.mul(c), r.mul(sn))
}

/// `−expm1(s ln w)/s` of `ln w = lw + i tw` (double-doubles), `|s| < 1/4`, as a complex
/// double-double to about `2⁻⁵⁹`: for `|v| = |s ln w| < 1/4` as `−ln w (1 + v/2 + v² H(v))`
/// with `H = Σ vᵏ/(k + 2)!` in double, else from `eᵛ` as a double-double.
fn pole_remainder(s: f64, lw: Dd, tw: Dd) -> Cd {
    let (vr, vi) = (lw.mul_f64(s), tw.mul_f64(s));
    if vr.hi.hypot(vi.hi) < 0.25 {
        let v = Complex::new(vr.hi, vi.hi);
        let mut h = Complex::new(1.0 / 87_178_291_200.0, 0.0);
        for &c in [
            1.0 / 6_227_020_800.0,
            1.0 / 479_001_600.0,
            1.0 / 39_916_800.0,
            1.0 / 3_628_800.0,
            1.0 / 362_880.0,
            1.0 / 40_320.0,
            1.0 / 5_040.0,
            1.0 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
        ]
        .iter()
        {
            h = h * v + c;
        }
        let v2h = v * v * h;
        // g = 1 + v/2 + v² H
        let g = (
            vr.mul_f64(0.5).add_f64(1.0).add_f64(v2h.re),
            vi.mul_f64(0.5).add_f64(v2h.im),
        );
        let q = cmul_dd((lw, tw), g);
        (q.0.neg(), q.1.neg())
    } else {
        // eᵛ − 1 = e^(vr) (cos vi + i sin vi) − 1
        let (eh, el) = exp_dd(vr.hi, vr.lo);
        let e = Dd { hi: eh, lo: el };
        let ph = vi.mul(INV_2PI_DD);
        let (c, sn) = turn_dd(ph.hi, ph.lo);
        let ms = Dd::new(-s);
        (e.mul(c).add_f64(-1.0).div(ms), e.mul(sn).div(ms))
    }
}

/// `sin(πs/2)/s` for `|s| < 1/4` (and subnormal), as a double-double to about `2⁻⁶⁰`:
/// `(π/2)(1 − h²/6 + h⁴ P(h²))`, `h = πs/2`, the first two terms in double-double.
fn sin_half_pi_over_dd(s: f64) -> Dd {
    let h = PI_2_DD.mul_f64(s);
    let t = h.mul(h);
    let p = t.hi
        * t.hi
        * horner(
            t.hi,
            &[
                1.0 / 120.0,
                -1.0 / 5_040.0,
                1.0 / 362_880.0,
                -1.0 / 39_916_800.0,
                1.0 / 6_227_020_800.0,
                -1.0 / 1_307_674_368_000.0,
            ],
        );
    let c = t.mul(SIXTH_DD).neg().add_f64(1.0).add_f64(p);
    PI_2_DD.mul(c)
}

/// `π/2` and `1/6` as double-doubles.
const PI_2_DD: Dd = Dd {
    hi: core::f64::consts::FRAC_PI_2,
    lo: 6.123_233_995_736_766e-17,
};
const SIXTH_DD: Dd = Dd {
    hi: 1.0 / 6.0,
    lo: 9.251_858_538_542_97e-18,
};

/// `1/2π`, `ln 2π` and `ln 2` as double-doubles.
const INV_2PI_DD: Dd = Dd {
    hi: 0.159_154_943_091_895_35,
    lo: -9.839_338_337_591_243e-18,
};
const LN_2PI_DD: Dd = Dd {
    hi: 1.837_877_066_409_345_6,
    lo: -7.756_588_316_134_483e-17,
};
const LN_2_DD: Dd = Dd {
    hi: LN_2,
    lo: 2.319_046_813_846_299_6e-17,
};

/// `Li₋ₙ(x) = x Aₙ(x)/(1 − x)ⁿ⁺¹` for `2 ≤ n ≤ 18` and `0 < x < 1`, `Aₙ` the Eulerian polynomial
/// (its coefficients exact: below `2⁵³` to `n = 18`): all its terms are positive there, and the
/// numerator, `1 − x` and its power are double-doubles, so that the result is rounded once (and
/// costs a fifth of the series from logarithms).
fn polylog_negative_int(n: usize, x: f64) -> f64 {
    // the coefficients in an array of the size the order needs (the larger one cost the small
    // orders half as much again)
    if n <= 8 {
        eulerian_form::<9>(n, x)
    } else {
        eulerian_form::<19>(n, x)
    }
}

/// [`polylog_negative_int`] with the coefficients in an array of `N > n` entries.
#[inline(always)]
fn eulerian_form<const N: usize>(n: usize, x: f64) -> f64 {
    let mut a = [0.0f64; N];
    a[0] = 1.0;
    for m in 2..=n {
        for k in (1..m).rev() {
            a[k] = (k + 1) as f64 * a[k] + (m - k) as f64 * a[k - 1];
        }
    }
    let mut p = Dd::new(a[n - 1]);
    for &c in a[..n - 1].iter().rev() {
        p = p.mul_f64(x).add_f64(c);
    }
    let (d, dl) = two_sum(1.0, -x);
    let d = Dd { hi: d, lo: dl };
    let mut dp = d;
    for _ in 0..n {
        dp = dp.mul(d);
    }
    let num = p.mul_f64(x);
    if num.hi / dp.hi > f64::MAX {
        // every term is positive: past the range +∞ (the double-double quotient would be NaN,
        // ∞ − ∞ in its correction)
        return f64::INFINITY;
    }
    num.div(dp).value()
}

/// `Σ_{k≥1} xᵏ k⁻ˢ` for `s < 0` and `|x| ≤ 1/2` by [`polylog_series_ln`].
fn polylog_series_log(s: f64, x: f64) -> f64 {
    let (lx, lxl) = ln_dd(x.abs());
    polylog_series_ln(s, lx, lxl, x < 0.0)
}

/// `Li_s(±e^λ) = Σ_{k≥1} (±1)ᵏ e^(kλ) k⁻ˢ` for `s < 0`, `λ = lx + lxl ≤ −ln 2` a double-double
/// (`ln|x|`, or `−ln|x|` for the inversion, so that `1/x` is never rounded): each term from
/// double-double logarithms (the products of `polylog_series` would overflow and underflow apart
/// for large `|s|`, and round at every step), added in double-double from the first to past the
/// largest; an infinity where the sum leaves the range. For the alternating sum each term is a
/// double-double (to about `2⁻⁷⁴`), so that its cancellation (by 81 at `Li₋₂(−1/2)`, far more
/// than `Li_s` is ill-conditioned there) costs nothing; the positive terms are rounded once.
fn polylog_series_ln(s: f64, lx: f64, lxl: f64, alternate: bool) -> f64 {
    let peak = s / lx;
    let (mut hi, mut lo) = (0.0, 0.0);
    let mut k = 1.0f64;
    loop {
        let (lk, lkl) = ln_dd(k);
        let (a, ae) = two_prod(k, lx);
        let (b, be) = two_prod(-s, lk);
        let (c, ce) = two_sum(a, b);
        let cl = ce + ae + be + (k * lxl - s * lkl);
        let (mut th, mut tl) = if alternate {
            exp_dd(c, cl)
        } else {
            (exp_mul(c, 1.0, cl), 0.0)
        };
        if alternate && k % 2.0 == 1.0 {
            th = -th;
            tl = -tl;
        }
        let (h, e) = two_sum(hi, th);
        hi = h;
        lo += e + tl;
        if !hi.is_finite() {
            return hi;
        }
        if (k > peak && th.abs() <= 1e-18 * hi.abs()) || k >= 4000.0 {
            break;
        }
        k += 1.0;
    }
    hi + lo
}

/// Whether the series of `Li_s(±e^(−l))`, `s < 0`, serves: `l ≥ ln 2` (`|x| ≤ 1/2`), and no term
/// above `e^8.3 ≈ 4000` times the first. Its largest term, at `k* = |s|/l`, is
/// `e^(|s| ln k* − (k* − 1) l)` times the first; with fewer than about 110 terms, each a
/// double-double, the sum is then good to `2⁻⁵⁸` of the first term (the Hurwitz form instead
/// cancels where `|x|` is near `2^(−|s|)`: by 50 at `Li₋₇₅(−1.6·10⁻²³)`, whose `κ` is 1).
fn series_fits(s: f64, l: f64) -> bool {
    if l.is_nan() || l < LN_2 {
        return false;
    }
    let ks = -s / l;
    ks <= 1.0 || -s * ks.ln() - (ks - 1.0) * l <= 8.3
}

/// `Li_s(x)` for `s < 0` (but `s = −1`), `x < 1`.
fn polylog_negative(s: f64, x: f64, int: bool) -> f64 {
    if s < -1.125_899_906_842_624e15 {
        // far past the range: positive for x > 0; for x < 0 its sign would need the phase
        // (1 − s) arg a to more than double-double
        return if x > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if x > 0.0 {
        if int && s >= -18.0 {
            return polylog_negative_int(-s as usize, x);
        }
        if x <= 0.5 {
            return negative_series(s, x);
        }
        let (mu, mul) = ln_near_one(x);
        return polylog_near_one(s, mu, mul);
    }
    let (mu, mul) = ln_dd(-x);
    if int && x < -1.0 {
        // Li_s(1/x) = (−1)^(s+1) Li_s(x) for an integer s ≤ 0, ln(−1/x) = −μ exactly
        let v = if series_fits(s, mu) {
            polylog_series_ln(s, -mu, -mul, true)
        } else {
            polylog_hurwitz_form(s, -mu, -mul)
        };
        return if s % 2.0 != 0.0 { v } else { -v };
    }
    // the series where no term is far above the first; else the Hurwitz form
    if series_fits(s, -mu) {
        return polylog_series_ln(s, mu, mul, true);
    }
    polylog_hurwitz_form(s, mu, mul)
}

/// The series of `Li_s(x)` for `s < 0`, `|x| ≤ 1/2`: by products of prime powers for `s ≥ −1`
/// (where the first terms dominate), else term by term from logarithms.
fn negative_series(s: f64, x: f64) -> f64 {
    if s < -1.0 {
        polylog_series_log(s, x)
    } else {
        polylog_series(s, x)
    }
}

/// `Li_s(x)` for `x < −e^37.7` by Sommerfeld's expansion
/// `Li_s(−eᵘ) = −Σⱼ 2η(2j) μ^(s−2j)/Γ(s + 1 − 2j) − cos(πs) Li_s(−e⁻ᵘ)`, whose remainder is of
/// the order of `e⁻ᵘ` relative.
fn polylog_sommerfeld(s: f64, x: f64) -> f64 {
    // μ as a double-double: its rounding would cost s/2 ulps in μˢ
    let (mu, mul) = ln_dd(-x);
    let (l, ll) = ln_dd(mu);
    let ll = ll + mul / mu;
    let (a, ae) = two_prod(s, l);
    let al = ae + s * ll;
    // 1 + s exactly as z + zl: it rounds where it crosses a power of 2 (s just below 2ᵏ), by up
    // to ulp(s)/2, which ψ(1 + s) made 620 ulps of Γ at s = 127.7; its low part through ψ
    let (z, zl) = two_sum(1.0, s);
    // μˢ/Γ(s + 1), past s = 160 from ln Γ(s + 1) in double-double (1/Γ would underflow)
    let mut t = if s > 160.0 {
        let e = Dd { hi: a, lo: al }.sub(lgamma_dd(z, zl));
        exp_mul(e.hi, 1.0, e.lo)
    } else {
        let rg = if s > -1.0 {
            1.0 / gamma_f64(z)
        } else {
            -sincos_pi(s).0 * gamma_f64(-s) / PI_HI
        };
        // 1/Γ(z + zl) = (1/Γ(z)) (1 − zl ψ(z))
        let rl = if zl == 0.0 || s <= -1.0 {
            0.0
        } else {
            -rg * zl * digamma_rough(z)
        };
        exp_mul(a, rg, rl + rg * al)
    };
    let (mut hi, mut lo) = (0.0, 0.0);
    let mut j = 0.0;
    let mut last = f64::INFINITY;
    while j < 200.0 {
        let eta = if j == 0.0 { 0.5 } else { eta_right(2.0 * j) };
        let term = 2.0 * eta * t;
        if term.abs() > last {
            break;
        }
        let (h, e) = two_sum(hi, term);
        hi = h;
        lo += e;
        if term.abs() <= 1e-18 * hi.abs() {
            break;
        }
        last = term.abs();
        t *= (s - 2.0 * j) * (s - 2.0 * j - 1.0) / (mu * mu);
        j += 1.0;
    }
    -(hi + lo) - sincos_pi(s).1 * polylog_real(s, 1.0 / x)
}

/// `Σ xᵏ k⁻ˢ` for `x < −1` cut at its least term, when that is below `2⁻⁶⁰` of the sum (for an
/// order large against `ln|x|`, where the divergent series is asymptotic to `Li_s(x)`).
fn polylog_divergent(s: f64, x: f64) -> Option<f64> {
    let (mut hi, mut lo) = (x, 0.0);
    let mut last = x.abs();
    let mut k = 2.0f64;
    while k < 200.0 {
        let term = pow_mul_ln(k, -s, 1.0) * x.powi(k as i32);
        if term.abs() > last || !term.is_finite() {
            return None;
        }
        let (h, e) = two_sum(hi, term);
        hi = h;
        lo += e;
        if term.abs() <= 1e-18 * hi.abs() {
            return Some(hi + lo);
        }
        last = term.abs();
        k += 1.0;
    }
    None
}

/// `yˢ g` for `y > 0`, `s ln y` as a double-double.
#[inline]
fn pow_mul_ln(y: f64, s: f64, g: f64) -> f64 {
    let (l, ll) = ln_dd(y);
    let (a, ae) = two_prod(s, l);
    exp_mul(a, g, g * (ae + s * ll))
}

/// `Li_s(x)` of `f64`s.
pub(crate) fn polylog_real(s: f64, x: f64) -> f64 {
    if s.is_nan() || x.is_nan() || x > 1.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return x;
    }
    if x == 1.0 {
        return if s > 1.0 { zeta_f64(s) } else { f64::INFINITY };
    }
    if s == 1.0 {
        return -(-x).ln_1p();
    }
    if s == 0.0 {
        return if x == f64::NEG_INFINITY {
            -1.0
        } else {
            x / (1.0 - x)
        };
    }
    if s == 2.0 {
        return dilog_f64(x);
    }
    if s == -1.0 {
        // x/(1 − x)², divided twice where the square would overflow
        let d = 1.0 - x;
        return if x == f64::NEG_INFINITY {
            -0.0
        } else if d < 1e150 {
            x / (d * d)
        } else {
            x / d / d
        };
    }
    if x == f64::NEG_INFINITY {
        return if s > 0.0 { x } else { -0.0 };
    }
    if s > 0.0 && s < f64::MIN_POSITIVE {
        // a subnormal order: Li_s(x) = x/(1 − x) + s ∂ₛLi + …, and |∂ₛLi/Li| is at most about
        // ln(1/|ln x|) + γ < 40 near x = 1 and ln ln|x| + γ < 8 far left, so the rest is below
        // 2⁻¹⁰¹⁶ of the value (the integral of Fermi and Dirac over Γ(s) ≈ 1/s overflowed to NaN)
        return x / (1.0 - x);
    }
    if s == f64::INFINITY {
        return x;
    }
    if s == f64::NEG_INFINITY {
        return f64::NAN;
    }
    // Li_s(x) = x (1 + x 2⁻ˢ + …): once x 2⁻ˢ < 2⁻⁶⁰ it rounds to x (for x < −1 the divergent
    // series is asymptotic to it, its least term far smaller)
    if s > 60.0 + x.abs().log2() {
        return x;
    }
    let int = s == s.round();
    if s < 0.0 {
        return polylog_negative(s, x, int);
    }
    let ax = x.abs();
    if (1.0..20.0).contains(&s) && (-2.0..-0.75).contains(&x) {
        return polylog_near_minus_one(s, (-x).ln());
    }
    // (the series to |x| = 3/4: below x = 3/4 the expansion about 1 cancels by up to 3 between
    // Γ(1 − s)(−μ)^(s−1) and the first ζ(s − k) (4 ulps for s < 1 just past x = 1/2, where the
    // series, its terms all positive, gave 1.2), and above x = −3/4 the expansion about −1
    // starts from η(s), in the binade above Li_s, and its first two terms cancel by up to 1.6:
    // 3.25 ulps at Li₁.₃₁₄₆(−0.6061) and 6.4 for 1/4 ≤ s < 1/2 near x = −1/2, where the series,
    // whose first term x is exact, gave 0.25 and 1.4)
    if ax <= 0.75 || (s >= 20.0 && x >= -1.0) {
        return polylog_series(s, x);
    }
    if x > 0.0 {
        let (mu, mul) = ln_near_one(x);
        return polylog_near_one(s, mu, mul);
    }
    if x >= -2.0 {
        return polylog_near_minus_one(s, (-x).ln());
    }
    if int && s >= 1.0 {
        return polylog_inverse_int(s as usize, x);
    }
    if s > 2.0 * (-x).ln()
        && let Some(v) = polylog_divergent(s, x)
    {
        return v;
    }
    if x < -2.4e16 && s < 0.25 * (-x).ln() {
        return polylog_sommerfeld(s, x);
    }
    polylog_fermi_dirac(s, x)
}

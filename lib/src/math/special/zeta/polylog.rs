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

use super::hurwitz::BERNOULLI_2K;
use super::real::{pow_2pi_neg, zeta_f64, zeta_regular, zeta_right};
use super::util::{
    PI_HI, exp_mul, exp_parts, horner, ln_dd, mla, poly_tail, scale, sincos_pi, two_prod, two_sum,
};
use crate::math::special::gamma::{gamma_f64, lgamma_f64};
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

/// `Li₂(x)` for `−1 ≤ x ≤ 3/4`, `y = −ln(1 − x)` as a double-double (corrected by
/// `(1 − x) eʸ − 1 = y − y_true` to first order).
#[inline]
fn dilog_series(x: f64) -> f64 {
    let y = -(-x).ln_1p();
    let (u, ul) = two_sum(1.0, -x);
    let (e, el, q) = exp_parts(y);
    let us = scale(u, q);
    let (m, me) = two_prod(e, us);
    // (1 − x) eʸ = e^(y − y_true).
    let yl = -((m - 1.0) + (me + el * us + e * scale(ul, q)));
    let y2 = y * y;
    // y − y²/4 + y³ P(y²), and yl (1 − y/2).
    y + (yl * (1.0 - 0.5 * y) + y2 * (-0.25 + y * horner(y2, &DILOG_P)))
}

/// `c − a·b + rest` as a double-double sum and one rounding: `c` a double-double, `a·b` exact.
#[inline(always)]
fn dd_minus_prod(c_hi: f64, c_lo: f64, a: f64, b: f64, rest: f64) -> f64 {
    let (p, pe) = two_prod(a, b);
    let (h, l) = two_sum(c_hi, -p);
    h + (l + (c_lo - pe + rest))
}

/// `c − ½ ln² y + rest`, `ln y` as a double-double.
#[inline(always)]
fn dd_minus_half_ln2(c_hi: f64, c_lo: f64, y: f64, rest: f64) -> f64 {
    let (l, ll) = ln_dd(y);
    dd_minus_prod(c_hi, c_lo, 0.5 * l, l, rest - l * ll)
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
        // −π²/6 − ½ ln²(−x) − Li₂(1/x) = −(π²/6 − ½ ln²(−x) + Li₂(1/x)) + … with the sign of ½ln².
        let (l, ll) = ln_dd(-x);
        return -dd_minus_prod(
            PI2_6_HI,
            PI2_6_LO,
            -0.5 * l,
            l,
            dilog_series(1.0 / x) + l * ll,
        );
    }
    if x <= 0.75 {
        return dilog_series(x);
    }
    if x <= 2.0 {
        // π²/6 − ln x ln|1 − x| − Li₂(1 − x), 1 − x exact.
        if x == 1.0 {
            return PI2_6_HI;
        }
        let u = 1.0 - x;
        let lx = (-u).ln_1p();
        let lu = u.abs().ln();
        return dd_minus_prod(PI2_6_HI, PI2_6_LO, lx, lu, -dilog_series(u));
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
    dd_minus_half_ln2(PI2_3_HI, PI2_3_LO, x, -dilog_series(1.0 / x))
}

// ---------------------------------------------------------------------------------------------
// The polylogarithm of a real order.
//
// * `|x| ≤ 1/2` (and every `|x| ≤ 1` for `s ≥ 20`): the series `Σ xᵏ k⁻ˢ`, `p⁻ˢ` of the primes
//   from `ln p` as a double-double and the others by products.
// * `1/2 < x < 1`: `Li_s(eᵘ) = Γ(1 − s)(−μ)^(s−1) + Σ ζ(s − k) μᵏ/k!`, `μ = ln x`; the pole of
//   `ζ(s − k)` at `k = n − 1` (`n` the integer nearest `s ≥ 1/2`) taken together with the pole of
//   `Γ(1 − s)` in closed form, so that `s` near an integer costs nothing; `ζ(s − k)` past
//   `s − k < −1/2` by the reflection formula as a recurrence in `k`.
// * `−2 ≤ x < −1/2`: `Li_s(−eᵘ) = −Σ η(s − k) μᵏ/k!`, `μ = ln(−x)`, `η` Dirichlet's eta.
// * `x < −2`, `s > 0`: for an integer `s = n ≥ 1` the inversion
//   `Li_n(x) = −(−1)ⁿ Li_n(1/x) − Σ 2η(2j) μⁿ⁻²ʲ/(n − 2j)!` (μ a double-double); otherwise, for
//   an order large against `ln|x|`, the divergent series `Σ xᵏ k⁻ˢ` cut at its least term when
//   that converges, Sommerfeld's expansion past `x = −e^37.7` for `s < ln|x|/4`, and else the
//   integral of Fermi and Dirac `Li_s(−eᵘ) = −F(μ)/Γ(s)`, `F = ∫₀^∞ t^(s−1)/(1 + e^(t−μ)) dt`,
//   all of whose terms are positive, by Gauss–Legendre pieces.
// * A non-integer `s < 0`, `x < 0` (but `|x| < 0.005`): the Hurwitz zeta function of the
//   complex `a = 1/2 − iμ/2π`, `Li_s(x) = 2 Γ(1 − s)(2π)^(s−1) Re[i^(1−s) ζ(1 − s, a)]`, whose
//   Euler–Maclaurin terms decay and whose real part cancels only where `Li_s` itself is
//   ill-conditioned (near its real zeros).
// * An integer `s ≤ 0`: `x Aₙ(x)/(1 − x)ⁿ⁺¹`, `Aₙ` Eulerian.

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
/// `μ < 0`: `μⁿ⁻¹/(n − 1)! (R(1 + ε) − E)`, `E = expm1(εG)/ε`,
/// `G = ln Γ(1 − ε)/ε + ln(−μ) − ln Πⱼ(1 + ε/j)/ε`.
fn pole_pair(n: usize, eps: f64, mu: f64) -> f64 {
    let lmu = (-mu).ln();
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
    let g = lg + lmu - lp;
    let e = if eps == 0.0 {
        g
    } else {
        (eps * g).exp_m1() / eps
    };
    let mut c = 1.0;
    for j in 1..n {
        c *= mu / j as f64;
    }
    c * (zeta_regular(1.0 + eps) - e)
}

/// `Li_s(eᵘ)` for `1/2 < x = eᵘ < 1`.
fn polylog_near_one(s: f64, mu: f64) -> f64 {
    let n = s.round();
    let (skip, pair) = if s >= 0.5 {
        let n = n as usize;
        (Some(n - 1), pole_pair(n, s - n as f64, mu))
    } else {
        // Γ(1 − s)(−μ)^(s−1), no pole; the exponent (s − 1) ln(−μ) as a double-double
        let (l, ll) = ln_dd(-mu);
        let (sm1, sm1l) = two_sum(s, -1.0);
        let (a, ae) = two_prod(sm1, l);
        let g = gamma_f64(1.0 - s);
        (None, exp_mul(a, g, g * (ae + sm1 * ll + sm1l * l)))
    };
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
        if (k as f64 > s + 1.0 && small >= 2) || k >= 400 {
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

/// `Li_n(x)` for an integer `n ≥ 1` and `x < −1`: `−(−1)ⁿ Li_n(1/x) − Σ 2η(2j) μⁿ⁻²ʲ/(n − 2j)!`.
fn polylog_inverse_int(n: usize, x: f64) -> f64 {
    // μ as a double-double (its rounding alone would cost n/2 ulps), the polynomial
    // Σ cₖ μᵏ, cₖ = 2η(n − k)/k! for even n − k, by Horner's rule with its derivative for the
    // low part of μ; every term is positive.
    let (mu, mul) = ln_dd(-x);
    let inner = polylog_real(n as f64, 1.0 / x);
    let fact = |k: usize| -> f64 { (1..=k).fold(1.0, |f, i| f * i as f64) };
    let (mut p, mut dp) = (0.0, 0.0);
    for k in (0..=n).rev() {
        dp = dp * mu + p;
        let j2 = n - k;
        let ck = if j2 % 2 == 1 {
            0.0
        } else if j2 == 0 {
            1.0 / fact(k)
        } else {
            2.0 * eta_right(j2 as f64) / fact(k)
        };
        p = p * mu + ck;
    }
    let sum = p + dp * mul;
    let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
    sign * inner - sum
}

/// `ζ(σ, a)` of a real `σ > 1` and the complex `a = 1/2 − iμ/2π` by the Euler–Maclaurin formula
/// (its terms decay for `σ > 1`, so that nothing cancels).
fn hurwitz_complex(sigma: f64, mu: f64) -> (f64, f64) {
    let ai = -mu * (0.5 / PI_HI);
    let abs_a = 0.5f64.hypot(ai);
    let need = (sigma + 40.0) / 2.83;
    let big_n = if abs_a >= need {
        0.0
    } else {
        (need - abs_a).ceil()
    };
    // z^(−σ) = e^(−σ ln|z|) e^(−iσ arg z)
    let cpow = |re: f64, im: f64| -> (f64, f64) {
        let m = (-sigma * re.hypot(im).ln()).exp();
        let ph = -sigma * im.atan2(re);
        (m * ph.cos(), m * ph.sin())
    };
    let (zr, zi) = (big_n + 0.5, ai);
    let p = cpow(zr, zi);
    let d = zr * zr + zi * zi;
    let r = (zr / d, -zi / d);
    let r2 = (r.0 * r.0 - r.1 * r.1, 2.0 * r.0 * r.1);
    // z/(σ − 1) + 1/2 + Σ B₂ₖ/(2k)! (σ)₂ₖ₋₁ z^(1−2k), times P = z^(−σ)
    let mut fac = (sigma * r.0, sigma * r.1);
    let mut acc = (0.5 + zr / (sigma - 1.0), zi / (sigma - 1.0));
    let mut k2 = 1.0;
    for &b in BERNOULLI_2K.iter() {
        let term = (b * fac.0, b * fac.1);
        acc.0 += term.0;
        acc.1 += term.1;
        if term.0.abs() + term.1.abs() <= 1e-19 * (acc.0.abs() + acc.1.abs()) {
            break;
        }
        let f = (sigma + k2) * (sigma + k2 + 1.0);
        fac = (
            f * (fac.0 * r2.0 - fac.1 * r2.1),
            f * (fac.0 * r2.1 + fac.1 * r2.0),
        );
        k2 += 2.0;
    }
    let mut z = (p.0 * acc.0 - p.1 * acc.1, p.0 * acc.1 + p.1 * acc.0);
    let mut n = big_n;
    while n > 0.0 {
        n -= 1.0;
        let v = cpow(n + 0.5, ai);
        z.0 += v.0;
        z.1 += v.1;
    }
    z
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
/// mass of `t^(s−1) e^(−t)`.
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
    // the pieces past a, each about its exact midpoint m: t = m ± d,
    // t^(s−1) = m^(s−1) (1 ± d/m)^(s−1), e^(t−μ) = e^(m−μ) e^(±d)
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
        let mut sum = 0.0;
        for i in 0..8 {
            let d = h * GL16_X[i];
            let r = d / m;
            let g1 = (sm1 * (-r).ln_1p()).exp();
            let g2 = (sm1 * r.ln_1p()).exp();
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
        -f * (c - lgamma_f64(s)).exp()
    } else {
        -f / gamma_f64(s)
    }
}

/// `Li_s(x)` for a non-integer `s < 0` and `x < 0` by the Hurwitz zeta function:
/// `Li_s(x) = 2 Γ(1 − s) (2π)^(s−1) Re[i^(1−s) ζ(1 − s, 1/2 − iμ/2π)]`, `μ = ln(−x)`, whose
/// Euler–Maclaurin terms decay (`1 − s > 1`) and whose real part cancels only as `Li_s` itself
/// is small.
fn polylog_hurwitz_form(s: f64, x: f64) -> f64 {
    let mu = (-x).ln();
    let z = hurwitz_complex(1.0 - s, mu);
    // i^(1−s) = e^(iπ(1−s)/2): cos = sin(πs/2), sin = cos(πs/2)
    let (sn, cs) = sincos_pi(0.5 * s);
    let re = z.0 * sn - z.1 * cs;
    2.0 * gamma_f64(1.0 - s) * pow_2pi_neg(1.0 - s) * re
}

/// `Li₋ₙ(x) = x Aₙ(x)/(1 − x)ⁿ⁺¹` for an integer `n ≥ 1`, `Aₙ` the Eulerian polynomial (by its
/// recurrence `A(n, k) = (k + 1) A(n − 1, k) + (n − k) A(n − 1, k − 1)`, exact to `n = 18` and
/// within an ulp of each coefficient past it), for `n ≤ 64`.
fn polylog_negative_int(n: usize, x: f64) -> f64 {
    let mut a = [0.0f64; 65];
    a[0] = 1.0;
    for m in 2..=n {
        for k in (1..m).rev() {
            a[k] = (k + 1) as f64 * a[k] + (m - k) as f64 * a[k - 1];
        }
    }
    let p = horner(x, &a[..n]);
    let d = 1.0 - x;
    x * p / d.powi(n as i32 + 1)
}

/// `Li_s(x)` for `x < −e^37.7` by Sommerfeld's expansion
/// `Li_s(−eᵘ) = −Σⱼ 2η(2j) μ^(s−2j)/Γ(s + 1 − 2j) − cos(πs) Li_s(−e⁻ᵘ)`, whose remainder is of
/// the order of `e⁻ᵘ` relative.
fn polylog_sommerfeld(s: f64, x: f64) -> f64 {
    let mu = (-x).ln();
    // μˢ/Γ(s + 1)
    let rg = if s > -1.0 {
        1.0 / gamma_f64(1.0 + s)
    } else {
        -sincos_pi(s).0 * gamma_f64(-s) / PI_HI
    };
    let mut t = pow_mul_ln(mu, s, rg);
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
        return x / (1.0 - x);
    }
    if s == 2.0 {
        return dilog_f64(x);
    }
    if s == -1.0 {
        let d = 1.0 - x;
        return x / (d * d);
    }
    if x == f64::NEG_INFINITY {
        return if s > 0.0 { x } else { -0.0 };
    }
    if s == f64::INFINITY {
        return x;
    }
    if s == f64::NEG_INFINITY {
        return f64::NAN;
    }
    let int = s == s.round() && s.abs() < 1e15;
    if int && (-64.0..0.0).contains(&s) {
        return polylog_negative_int(-s as usize, x);
    }
    // (the direct series where |x| is small: the Hurwitz form's terms then cancel, as Li_s ≈ x)
    if s < 0.0 && !int && x <= -0.005 && !(s >= -3.0 && x >= -0.12) {
        return polylog_hurwitz_form(s, x);
    }
    let ax = x.abs();
    if (1.0..20.0).contains(&s) && (-2.0..-0.6).contains(&x) {
        return polylog_near_minus_one(s, (-x).ln());
    }
    if ax <= 0.5 || (s >= 1.0 && ax <= 0.6) || (s >= 20.0 && x >= -1.0) {
        return polylog_series(s, x);
    }
    if x > 0.0 {
        return polylog_near_one(s, (x - 1.0).ln_1p());
    }
    if x >= -2.0 {
        return polylog_near_minus_one(s, (-x).ln());
    }
    if int && s >= 1.0 {
        return polylog_inverse_int(s as usize, x);
    }
    if int {
        // Li_s(1/x) = (−1)^(s+1) Li_s(x) for an integer s ≤ 0
        let odd = (s as i64) % 2 != 0;
        let v = polylog_real(s, 1.0 / x);
        return if odd { v } else { -v };
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

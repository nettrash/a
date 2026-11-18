//! The real branches of the Lambert W function.
//!
//! Near the branch point `−1/e` both branches are polynomials in `p = √(2(1 + e·x))` (`p ≥ 0` for
//! `W₀`, `p ≤ 0` for `W₋₁`), with `1 + e·x` computed without the cancellation (`e` as a
//! double-double), for `|p| ≤ 0.66` (`x ≤ −0.29`); `W₋₁` goes on to `x = −0.2` with a polynomial
//! in `s = √(2(−ln(−x) − 1))`, `s²` computed as `−2 ln(1 − q)` from the accurate `q = 1 + e·x`.
//! These pieces are full precision (each polynomial's head a double-double). Elsewhere a starting value good to
//! `3·10⁻⁷` (a polynomial in `p`, in `ln x` or in `s`, or the asymptotic series in `ln |x|`) and
//! one step of Halley's iteration, with the residual `w eʷ − x` from a double-double `eʷ`, give
//! full precision; for `|w| > 40`, where `eʷ` would leave the range, one step of Fritsch's
//! iteration of the fourth order instead. The polynomials are minimax fits (Remez exchange,
//! mpmath) made for this crate.
//!
//! The tables are printed by `zeta/gen_lambert.py` (with `zeta/gen_fit.py`).

use super::zeta::util::{exp_parts, horner, mla, poly_tail, residual, scale, two_prod, two_sum};
use super::{Real, via};
use core::f64::consts::LN_2;

// The fits: `Σ c[k] (t − t₀)ᵏ` of the variable `t` of the piece, `c[0] + C_LO` for the pieces of
// full precision.
// BRANCH_MID: [-0.3, 0.3] about 0, degree 16, error 4.59e-18 (f64 coefficients 1.83e-17)
const BRANCH_MID: [f64; 17] = [
    -1.0,
    0.9999999999999998,
    -0.3333333333333338,
    0.1527777777779156,
    -0.07962962962956947,
    0.04450231479342748,
    -0.02598471487283059,
    0.015635634020265073,
    -0.009616892428821891,
    0.006014488471830462,
    -0.00381127541061019,
    0.0024420181460291152,
    -0.0015775083620373032,
    0.0010128399260394443,
    -0.000664354441044343,
    0.0005249233054953356,
    -0.0003442079224192346,
];
const BRANCH_MID_LO: f64 = 4.608392740013044e-19;
// BRANCH_W0: [0.3, 0.66] about 0.48, degree 12, error 2.08e-18 (f64 coefficients 2.75e-17)
const BRANCH_W0: [f64; 13] = [
    -0.5832440099410049,
    0.7592146152985498,
    -0.1888948072723319,
    0.06457176542498401,
    -0.025103625938531646,
    0.010466233976472106,
    -0.004559621161779536,
    0.0020472656938001956,
    -0.0009396696863855495,
    0.00043848291945426914,
    -0.00020740300608587317,
    0.00010162895077935559,
    -4.87071055773107e-05,
];
const BRANCH_W0_LO: f64 = -1.429652323156203e-17;
// BRANCH_WM1: [-0.66, -0.3] about -0.48, degree 15, error 5.95e-18 (f64 coefficients 8.44e-17)
const BRANCH_WM1: [f64; 16] = [
    -1.5795054225452714,
    1.4786307733688389,
    -0.747022717443019,
    0.5192309180438657,
    -0.4102754709737119,
    0.3474974018302341,
    -0.3074380113644367,
    0.28025585811364306,
    -0.26110767365846993,
    0.24733979274134887,
    -0.23742452244022832,
    0.23027961577459163,
    -0.22357898805313553,
    0.2202665464273163,
    -0.25099759808137445,
    0.2508195104561635,
];
const BRANCH_WM1_LO: f64 = -9.327895861046486e-17;
// FULL_S1: [0.6, 1.12] about 0.86, degree 10, error 2.2e-19 (f64 coefficients 6.73e-17)
const FULL_S1: [f64; 11] = [
    -2.122300768247625,
    -1.6262830003607813,
    -0.39031771878023297,
    -0.017124056749405894,
    0.0024605002220446414,
    -0.00028446227338455135,
    1.61867763260239e-05,
    3.4962257006851617e-06,
    -1.520552241856099e-06,
    3.3371289227313196e-07,
    -4.583932324133027e-08,
];
const FULL_S1_LO: f64 = 1.2778411446116097e-16;
// INIT_S4: [6.0, 12.0] about 9.0, degree 6, error 1.43e-8 (f64 coefficients 1.43e-8)
const INIT_S4: [f64; 7] = [
    -45.313607851235055,
    -9.203098983493122,
    -0.49019234468227396,
    -0.0006078168883626883,
    4.091956599447591e-05,
    -3.155171139159026e-06,
    2.4889692026786947e-07,
];
// INIT_L1: [1.09, 4.0] about 2.55, degree 6, error 2.44e-7 (f64 coefficients 2.44e-7)
const INIT_L1: [f64; 7] = [
    1.9053399874539203,
    0.6558079796111632,
    0.03884366281127294,
    -0.004318946388774293,
    0.00034675976289162686,
    1.099764492647965e-05,
    -1.0383833448735303e-05,
];
// INIT_L2: [4.0, 8.0] about 6.0, degree 6, error 1.54e-8 (f64 coefficients 1.54e-8)
const INIT_L2: [f64; 7] = [
    4.496664151431014,
    0.8180712421879965,
    0.013538353404152591,
    -0.0011934262038811548,
    0.0001066577779031292,
    -9.308216049200389e-06,
    6.89685180022017e-07,
];
// INIT_L3: [8.0, 16.0] about 12.0, degree 7, error 9.88e-9 (f64 coefficients 9.88e-9)
const INIT_L3: [f64; 8] = [
    9.725272049500932,
    0.906762373188708,
    0.0039415300480099115,
    -0.00021078371262970777,
    1.212779201969364e-05,
    -7.161492140575362e-07,
    4.864161965799682e-08,
    -3.1007403238333175e-09,
];

// W0_H0: e^(-W0) in p - 0.9299999999999999 on [0.66, 1.2], degree 14, error 1.6e-18 (5.8e-18 in doubles)
const W0_H0: [f64; 15] = [
    1.3163099399854346,
    -0.8174596964760018,
    0.41819807567660033,
    -0.19999534050002513,
    0.09253981084648216,
    -0.042006434816286714,
    0.01883380766259533,
    -0.008372362182436206,
    0.003698742773017077,
    -0.001626366370038711,
    0.0007124896672513301,
    -0.0003108681172478533,
    0.00013548337573261317,
    -6.21050974933371e-05,
    2.687969366591621e-05,
];
const W0_H0_LO: f64 = 5.704773045282556e-17;
// W0_H1: e^(-W0) in p - 1.5 on [1.2, 1.8], degree 14, error 3.1e-19 (7.0e-18 in doubles)
const W0_H1: [f64; 15] = [
    0.956948951938836,
    -0.48402929220397856,
    0.20073362992925453,
    -0.07763843159641397,
    0.029014593827267506,
    -0.010628177305513194,
    0.0038430736574665073,
    -0.0013772119368168228,
    0.0004903207414952515,
    -0.0001737033637826909,
    6.129859930900688e-05,
    -2.154959415296797e-05,
    7.5627967680882074e-06,
    -2.7617178238368657e-06,
    9.62885181403325e-07,
];
const W0_H1_LO: f64 = 2.7587608562232e-17;
// W0_H2: e^(-W0) in p - 2.2 on [1.8, 2.6], degree 14, error 9.6e-19 (9.2e-18 in doubles)
const W0_H2: [f64; 15] = [
    0.6953994304222486,
    -0.2870881982524262,
    0.09674348528308092,
    -0.030335774796957447,
    0.009178981175032668,
    -0.0027199626758552847,
    0.0007951532015020804,
    -0.00023027850625214786,
    6.623266602887903e-05,
    -1.895097153956362e-05,
    5.400277989856025e-06,
    -1.5323890876058446e-06,
    4.341546891612435e-07,
    -1.288375030990173e-07,
    3.623561134183244e-08,
];
const W0_H2_LO: f64 = -2.5250833156841555e-17;
// W0_H3: e^(-W0) in p - 3.05 on [2.6, 3.5], degree 14, error 2.5e-19 (4.2e-18 in doubles)
const W0_H3: [f64; 15] = [
    0.5064668803772329,
    -0.17128588484168147,
    0.047086399493111816,
    -0.012018766237677293,
    0.0029564596721347552,
    -0.0007116193046007458,
    0.00016888377780706183,
    -3.968774528278009e-05,
    9.25976088129758e-06,
    -2.148678698037818e-06,
    4.964582739386325e-07,
    -1.1424073934464334e-07,
    2.6234531881477902e-08,
    -6.25721549925497e-09,
    1.426810631901263e-09,
];
const W0_H3_LO: f64 = 1.761899676110831e-17;
// W0_H4: e^(-W0) in p - 3.905 on [3.5, 4.31], degree 12, error 2.5e-18 (8.6e-18 in doubles)
const W0_H4: [f64; 13] = [
    0.3882364429737525,
    -0.11126150026094,
    0.025831478293602377,
    -0.005558914233946253,
    0.0011516579471328948,
    -0.00023330294148024623,
    4.6576836791866596e-05,
    -9.204340069605163e-06,
    1.8053903398589586e-06,
    -3.5205879939078657e-07,
    6.836086042951256e-08,
    -1.349286453455076e-08,
    2.5988493149540294e-09,
];
const W0_H4_LO: f64 = -8.63218422518658e-18;

// INIT_S2: [1.1, 2.0] about 1.55, degree 5, error 1.52e-9 (f64 coefficients 1.52e-9)
const INIT_S2: [f64; 6] = [
    -3.4353765326215835,
    -2.1864519074748157,
    -0.41960807085685903,
    -0.01156363871350694,
    0.0016223824523724565,
    -0.0002042259167491723,
];
// INIT_S3: [2.0, 6.0] about 4.0, degree 6, error 9.33e-8 (f64 coefficients 9.33e-8)
const INIT_S3: [f64; 7] = [
    -11.436840568778145,
    -4.383260226880436,
    -0.46742268870771264,
    -0.003392807561846929,
    0.00036891578144877953,
    -4.372166899106205e-05,
    5.0741752840534446e-06,
];

/// `e` as a double-double.
const E_HI: f64 = core::f64::consts::E;
const E_LO: f64 = 1.445_646_891_729_250_2e-16;

/// `−1/e` rounded to an `f64` and to an `f32`: both lie below `−1/e`, and both give `−1`, as
/// the argument a caller means by them.
const MINUS_INV_E_F64: f64 = -0.367_879_441_171_442_33;
const MINUS_INV_E_F32: f64 = -0.367_879_450_321_197_5;

/// The `W` of an `x` with `1 + e·x < 0`: NaN, but `−1` at `−1/e` rounded.
#[cold]
fn below_branch(x: f64) -> f64 {
    if x == MINUS_INV_E_F64 || x == MINUS_INV_E_F32 {
        -1.0
    } else {
        f64::NAN
    }
}

/// `1 + e·x` without the cancellation near `x = −1/e` (to a few units of its last place).
#[inline(always)]
fn branch_q(x: f64) -> f64 {
    residual(-E_HI, x, 1.0) + E_LO * x
}

/// `p = √(2(1 + e·x))` as a double-double `(p, pl)`, from `1 + e·x` as a double-double.
#[inline(always)]
fn branch_p(x: f64) -> (f64, f64) {
    let (m, me) = two_prod(E_HI, x);
    let (s, se) = two_sum(1.0, m);
    let (q, ql) = two_sum(s, se + me + E_LO * x);
    let p = (2.0 * q).sqrt();
    let r = residual(p, p, 2.0 * q) + 2.0 * ql;
    (p, r / (2.0 * p))
}

/// `x e^(−W)` from a piece `c` of `e^(−W)` in `p` at `t + tl`, rounded once.
#[inline(always)]
fn w0_piece<const N: usize>(x: f64, t: f64, tl: f64, c: &[f64; N], lo: f64) -> f64 {
    let slope = c[1] + t * (2.0 * c[2] + t * (3.0 * c[3] + t * 4.0 * c[4]));
    let hl = lo + (slope * tl + poly_tail(t, c));
    mla(x, c[0], x * hl)
}

/// The piece `c` at `t + tl`: [`full`] at `t` and its slope (from four terms) times `tl`.
#[inline(always)]
fn full_dd(t: f64, tl: f64, c: &[f64], lo: f64) -> f64 {
    let slope = c[1] + t * (2.0 * c[2] + t * (3.0 * c[3] + t * 4.0 * c[4]));
    c[0] + (lo + (slope * tl + poly_tail(t, c)))
}

/// `W₀(x)` for `|x| < 2⁻¹⁰`: its series `Σ (−n)ⁿ⁻¹ xⁿ/n!` to `x⁷`.
#[inline(always)]
fn w0_small(x: f64) -> f64 {
    let c = [
        -1.0,
        1.5,
        -8.0 / 3.0,
        125.0 / 24.0,
        -54.0 / 5.0,
        16807.0 / 720.0,
    ];
    x + x * (x * horner(x, &c))
}

/// `c[0] + lo + Σ_{k≥1} c[k] tᵏ`, a piece of full precision.
#[inline(always)]
fn full(t: f64, c: &[f64], lo: f64) -> f64 {
    c[0] + (lo + poly_tail(t, c))
}

/// One step of Halley's iteration for `w eʷ = x` from `w`.
#[inline(always)]
fn halley(w: f64, x: f64) -> f64 {
    // eʷ as a double-double, so that the residual w eʷ − x is exact to far below its size and
    // the step's only error is the final rounding.
    let (e, el, q) = exp_parts(w);
    let (e, el) = (scale(e, q), scale(el, q));
    let (p, pe) = two_prod(w, e);
    let f = (p - x) + (pe + w * el);
    let w1 = 1.0 + w;
    w - f / (e * w1 - (w + 2.0) * f / (2.0 * w1))
}

/// One step of Fritsch's iteration for `w eʷ = x` from `w`, given `ln(x/w)` as `l`.
#[inline(always)]
fn fritsch(w: f64, l: f64) -> f64 {
    let z = l - w;
    let w1 = 1.0 + w;
    let q = 2.0 * w1 * (w1 + z * (2.0 / 3.0));
    let e = z / w1 * ((q - z) / (q - 2.0 * z));
    w + w * e
}

/// The asymptotic series `W(eᴸ) ≈ L − l + l/L + …` of `l = ln |L|`.
#[inline(always)]
fn asymptotic(big_l: f64, l: f64) -> f64 {
    let r = 1.0 / big_l;
    let c2 = l * (l - 2.0) * 0.5;
    let c3 = l * (6.0 + l * (-9.0 + 2.0 * l)) * (1.0 / 6.0);
    let c4 = l * (-12.0 + l * (36.0 + l * (-22.0 + 3.0 * l))) * (1.0 / 12.0);
    big_l - l + r * (l + r * (c2 + r * (c3 + r * c4)))
}

/// `W₀(x)` of an `f64`.
pub(crate) fn w0_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x <= 3.0 {
        if x.abs() < 1.0 / 1024.0 {
            return if x == 0.0 { x } else { w0_small(x) };
        }
        let q = branch_q(x);
        if q < 0.0 {
            return below_branch(x);
        }
        let p = (2.0 * q).sqrt();
        if p <= 0.66 {
            let (p, pl) = branch_p(x);
            return if p <= 0.3 {
                full_dd(p, pl, &BRANCH_MID, BRANCH_MID_LO)
            } else {
                full_dd(p - 0.48, pl, &BRANCH_W0, BRANCH_W0_LO)
            };
        }
        // W = x e^(−W), e^(−W) a polynomial of full precision in p (a double-double), and the
        // product rounded once.
        let (p, pl) = branch_p(x);
        return if p <= 1.2 {
            w0_piece(x, p - 0.9299999999999999, pl, &W0_H0, W0_H0_LO)
        } else if p <= 1.8 {
            w0_piece(x, p - 1.5, pl, &W0_H1, W0_H1_LO)
        } else if p <= 2.6 {
            w0_piece(x, p - 2.2, pl, &W0_H2, W0_H2_LO)
        } else if p <= 3.5 {
            w0_piece(x, p - 3.05, pl, &W0_H3, W0_H3_LO)
        } else {
            w0_piece(x, p - 3.905, pl, &W0_H4, W0_H4_LO)
        };
    }
    if x == f64::INFINITY {
        return x;
    }
    let big_l = x.ln();
    let w = if big_l <= 4.0 {
        horner(big_l - 2.55, &INIT_L1)
    } else if big_l <= 8.0 {
        horner(big_l - 6.0, &INIT_L2)
    } else if big_l <= 16.0 {
        horner(big_l - 12.0, &INIT_L3)
    } else {
        asymptotic(big_l, big_l.ln())
    };
    if big_l <= 40.0 {
        halley(w, x)
    } else {
        fritsch(w, (x / w).ln())
    }
}

/// `W₋₁(x)` of an `f64`.
pub(crate) fn wm1_f64(x: f64) -> f64 {
    if x.is_nan() || x >= 0.0 {
        return if x == 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    let q = branch_q(x);
    if q < 0.0 {
        return below_branch(x);
    }
    let p = (2.0 * q).sqrt();
    if p <= 0.66 {
        let (p, pl) = branch_p(x);
        return if p <= 0.3 {
            full_dd(-p, -pl, &BRANCH_MID, BRANCH_MID_LO)
        } else {
            full_dd(0.48 - p, -pl, &BRANCH_WM1, BRANCH_WM1_LO)
        };
    }
    // s² = 2(u − 1), u = −ln(−x): from q = 1 + e·x, accurate, near the branch point, else from u.
    let (s, u) = if x <= -0.2 {
        let s2 = -2.0 * (-q).ln_1p();
        let s = s2.sqrt();
        if s <= 1.12 {
            return full(s - 0.86, &FULL_S1, FULL_S1_LO);
        }
        (s, 0.5 * s2 + 1.0)
    } else {
        let u = -(-x).ln();
        ((2.0 * (u - 1.0)).sqrt(), u)
    };
    let w = if s <= 2.0 {
        horner(s - 1.55, &INIT_S2)
    } else if s <= 6.0 {
        horner(s - 4.0, &INIT_S3)
    } else if s <= 12.0 {
        horner(s - 9.0, &INIT_S4)
    } else {
        asymptotic(-u, u.ln())
    };
    if u <= 40.0 {
        halley(w, x)
    } else if u <= 700.0 {
        fritsch(w, (x / w).ln())
    } else {
        // x/w would underflow: ln(x/w) = ln(2¹²⁸ x / w) − 128 ln 2.
        let l = ((x * 2f64.powi(128)) / w).ln() - 128.0 * LN_2;
        fritsch(w, l)
    }
}

/// The principal branch `W₀(x)` of the Lambert W function, the `w ≥ −1` with `w e^w = x`, for
/// `x ≥ −1/e`.
pub fn lambert_w0<T: Real>(x: T) -> T {
    via(x, w0_f64)
}

/// The other real branch `W₋₁(x)`, the `w ≤ −1` with `w e^w = x`, for `−1/e ≤ x < 0`.
pub fn lambert_wm1<T: Real>(x: T) -> T {
    via(x, wm1_f64)
}

//! The Riemann zeta function of a complex argument `s = σ + it` (`t > 0` below; `t < 0` by
//! `ζ(s̄) = conj ζ(s)`, `t = 0` the real function).
//!
//! * `σ ≥ 56`: `1 + 2⁻ˢ`, for every finite `t` (`3⁻ˢ` is below `2⁻⁸⁸`).
//! * `−1/2 ≤ σ ≤ 3/2`, `|s − 1| ≤ 3`: the Taylor series about the pole,
//!   `1/(s − 1) + Σ_{n<30} (−1)ⁿγₙ/n! (s − 1)ⁿ` with `γₙ` the Stieltjes constants (the
//!   coefficients fall so fast that every term is below the first), so that the cancellation
//!   against the pole near the real axis costs nothing.
//! * `1/2 ≤ σ < 56`, `t ≤ 10⁴` (`t ≤ 10³` for `σ ≥ 10`): the Euler–Maclaurin formula
//!   `ζ(s) = Σ_{n<N} n^(−s) + N^(1−s)/(s − 1) + N^(−s)/2 + Σ_{k≤24} B₂ₖ/(2k)! (s)₂ₖ₋₁ N^(−s−2k+1)`,
//!   `N = ⌈(|s| + 40)/2.83⌉`, so that the ratio of the terms of the remainder stays below `0.45²`.
//!   Each `n^(−s)` has its phase `t ln n / 2π` in double-double arithmetic, reduced exactly, and
//!   its modulus `e^(−σ ln n)` with `ln n` a double-double; the terms are added from the
//!   smallest, in double-double, and in the strip (where the bound is absolute near the zeros)
//!   each term is itself a double-double, to about `2⁻⁶²`.
//! * `10 ≤ σ < 56`, `t > 10³`: the Dirichlet series itself, `Σ_{n≤N} n^(−s)` with `N` the least
//!   for which `N^(1−σ)/(σ − 1) < 2⁻⁶⁰` (at most 80 terms), each term as above.
//! * `0 ≤ σ < 10`, `t > 10⁴`: the Riemann–Siegel formula for any `σ`, as J. Arias de Reyna,
//!   *High precision computation of Riemann's zeta function by the Riemann–Siegel formula, I*,
//!   Math. Comp. 80 (2011) 995–1009, states it: `ζ(s) = R(s) + χ(s) conj R(1 − σ + it)` with
//!   `R(s) = Σ_{n≤N} n^(−s) + (−1)^(N−1) U a^(−σ) Σ_{k<9} a^(−k) Σ_ℓ d_{k,ℓ}(σ) F^(3k−2ℓ)(p)/(π^(2k−ℓ) (2i)^ℓ)`,
//!   `a = √(t/2π)`, `N = ⌊a⌋`, `p = 1 − 2(a − N)`, `U = e^(−i(t/2 ln(t/2π) − t/2 − π/8))`, the
//!   `d_{k,ℓ}` by the recurrence of its section 6 and `F` the entire function
//!   `(e^(πi(z²/2 + 3/8)) − i√2 cos(πz/2))/(2 cos πz)` by its Taylor series. Nine terms of the
//!   correction leave below `10⁻¹⁹` of `|ζ|` from `t = 10⁴`; the cost grows as `√t`. Each term of
//!   its sums is a double-double (rounded to doubles, their errors add up as a random walk over
//!   the `√(t/2π)` terms), `χ(s) conj R(1 − σ + it)` is a double-double, and on the critical line
//!   one sum serves both parts.
//! * `σ < 1/2` (but `0 ≤ σ < 1/2` past `t = 10⁴`): the reflection `ζ(s) = χ(s) ζ(1 − s)`,
//!   `χ(s) = 2ˢ π^(s−1) sin(πs/2) Γ(1 − s)`, so that the sums are those of `ζ(1 − s)`, whose
//!   terms decay (at `σ` itself the Euler–Maclaurin formula adds about `|s|/2.83` terms of size
//!   `n^(−σ) ≥ 1`, each with its rounding), with `1 − σ` an exact pair. Where `t` is small and
//!   `σ` is not (`t < 13.5`, `−12 < σ`) `Γ(1 − s)` from `gamma_c64`, corrected by `ψ(1 − s)`
//!   for the low part of `1 − σ`, and `2(2π)^(s−1)`, `sin(πs/2)` and the products as
//!   double-doubles; elsewhere (`|1 − s| ≥ 13`) `ln χ(s)` itself, its modulus and its phase as
//!   double-doubles, from Stirling's series of `ln Γ(1 − s)` (ten terms) with `ln |1 − s|` and
//!   `arg(1 − s)` as double-doubles and, from `t = 13.5`, `ln sin(πs/2) = πt/2 − ln 2 +
//!   iπ(1 − σ)/2` (exact but for `e^(−πt) < 2⁻⁶¹`), below it `sin(πs/2)` and its product with
//!   `ζ(1 − s)` as double-doubles, so that nothing leaves the range before the last product.
//! * `t < 2⁻⁶⁰⁰`: the function at `t′ = 2⁻⁶⁰⁰·m` (`t = m·2ᵉ`) with its imaginary part (and at
//!   a zero of `ζ(σ)` its real part, of the order of `t²`) scaled back, from `σ = −12` within the
//!   last products of the logarithmic form. At `σ = 1` below `t = 2⁻⁵⁰⁰`: `γ − i/t` (`−i∞` past
//!   the range).
//! * `t > 2⁴⁰` (`σ < 56`), an infinite `t`, `σ = −∞` and `σ < −2⁵⁰`: NaN (the phases would need
//!   more than double-double arithmetic, and the infinite limits do not exist).
//!
//! The tables are printed by `gen_util.py` (its Bernoulli numbers live in `hurwitz.rs`) and
//! `gen_complex.py` in this directory.

use super::hurwitz::BERNOULLI_2K as BERNOULLI;
use super::real::zeta_f64;
use super::util::{
    PI_HI, PI_LO, STIRLING, atan_dd, exp_dd, exp_mul, exp_mul_k, exp_parts, ln_dd, ln_dd_acc,
    scale, sincos_pi, two_prod, two_sum,
};
use crate::math::complex::Complex;
use crate::math::special::dd::{Dd, fast_two_sum};
use crate::math::special::gamma::gamma_c64;

/// `sin(πj/32)` and `cos(πj/32)`, `j = 0, …, 8`, as double-doubles (`gen_complex.py`).
const SINCOS_PI_32: [(f64, f64, f64, f64); 9] = [
    (0.0, 0.0, 1.0, 0.0),
    (
        0.0980171403295606,
        -1.634582362244256e-18,
        0.9951847266721969,
        -4.248691367830441e-17,
    ),
    (
        0.19509032201612828,
        -7.991079068461731e-18,
        0.9807852804032304,
        1.8546939997825006e-17,
    ),
    (
        0.2902846772544624,
        -1.892797870777425e-17,
        0.9569403357322088,
        4.05538698618757e-17,
    ),
    (
        0.3826834323650898,
        -1.0050772696461588e-17,
        0.9238795325112867,
        1.7645047084336677e-17,
    ),
    (
        0.47139673682599764,
        6.516678136069013e-18,
        0.881921264348355,
        -1.9843248405890562e-17,
    ),
    (
        0.5555702330196022,
        4.709410940561677e-17,
        0.8314696123025452,
        1.4073856984728024e-18,
    ),
    (
        0.6343932841636455,
        1.0420901929280035e-17,
        0.773010453362737,
        -3.256590703364977e-17,
    ),
    // sin(π/4) = cos(π/4) = 1/√2, whose double gen_complex.py prints as 0.7071067811865476
    (
        core::f64::consts::FRAC_1_SQRT_2,
        -4.833646656726457e-17,
        core::f64::consts::FRAC_1_SQRT_2,
        -4.833646656726457e-17,
    ),
];

/// `1/2π`, `ln 2π`, `ln(2π)/2π`, `1 + ln 2π`, `π/4` and `ln 2` as double-doubles.
const INV_2PI_HI: f64 = 0.159_154_943_091_895_35;
const INV_2PI_LO: f64 = -9.839_338_337_591_243e-18;
const LN_2PI_HI: f64 = 1.837_877_066_409_345_6;
const LN_2PI_LO: f64 = -7.756_588_316_134_483e-17;
const LN_2PI_OVER_2PI_HI: f64 = 0.292_507_219_914_278_9;
const LN_2PI_OVER_2PI_LO: f64 = 9.893_802_182_295_06e-18;
const ONE_LN_2PI: Dd = Dd {
    hi: 2.837_877_066_409_345_3,
    lo: 1.444_787_217_636_864_7e-16,
};
const PI_4: Dd = Dd {
    hi: core::f64::consts::FRAC_PI_4,
    lo: 3.061_616_997_868_383e-17,
};
const LN_2: Dd = Dd {
    hi: core::f64::consts::LN_2,
    lo: 2.319_046_813_846_299_6e-17,
};
const INV_2PI: Dd = Dd {
    hi: INV_2PI_HI,
    lo: INV_2PI_LO,
};
const LN_2PI: Dd = Dd {
    hi: LN_2PI_HI,
    lo: LN_2PI_LO,
};

/// The Taylor coefficients of the Riemann–Siegel function
/// `F(z) = (e^(πi(z²/2 + 3/8)) − i√2 cos(πz/2))/(2 cos πz)` in `z²` (real and imaginary parts):
/// `F(z) = Σ RS_F[n] z²ⁿ`. Past `n = 50` they are below `10⁻⁶¹`, and leave the derivatives used
/// (to the 24th, on `|z| ≤ 1`) exact to `10⁻²⁹`.
const RS_F: [(f64, f64); 51] = [
    (0.1913417161825449, -0.24516701493090415),
    (0.21862023403876021, -0.036933834884962956),
    (0.06618828774017176, 0.06353439385614602),
    (-0.006802513023837094, 0.027223912663570066),
    (-0.0067838109850517905, 0.001385760877106652),
    (-0.0008118626615722327, -0.001189449446101378),
    (0.00014852676866689845, -0.00021269820192893323),
    (3.971650439760735e-05, 1.1171327401990151e-05),
    (2.3278062307252252e-07, 5.87285839865207e-06),
    (-7.163625815477553e-07, 2.498212552923518e-07),
    (-5.177423556156473e-08, -7.308700305101552e-08),
    (6.178963541930869e-09, -7.53679144816402e-09),
    (8.940541928977453e-10, 4.1044257973312284e-10),
    (-1.695707194963518e-11, 9.106559550294084e-11),
    (-8.163316951282953e-12, 4.3480990952496187e-13),
    (-1.8925546592706103e-13, -6.52091326154013e-13),
    (4.6637116296008625e-14, -2.574698839194822e-14),
    (2.6109215079890685e-15, 2.9783351960862874e-15),
    (-1.675336536372132e-16, 2.241756964196517e-16),
    (-1.7062132614058632e-17, -7.993661378773457e-18),
    (2.8756016707161996e-19, -1.1768943516467545e-18),
    (7.447650681605753e-20, 3.424141569957982e-21),
    (6.282686358510708e-22, 4.354432465678006e-21),
    (-2.360647625071713e-22, 8.042677010875067e-23),
    (-6.63453468151981e-24, -1.187404814328495e-23),
    (5.526719997560709e-25, -4.533873522499421e-25),
    (2.7498231887637325e-26, 2.3622132419636895e-26),
    (-9.115688251159012e-28, 1.524413235942887e-27),
    (-7.84470186886044e-29, -3.0537676963256187e-29),
    (7.919817544119006e-31, -3.7815907075540576e-30),
    (1.717310362718602e-31, 7.636679842553593e-33),
    (8.510516750158508e-34, 7.372397943589341e-33),
    (-2.9975596524789083e-34, 8.353560735145064e-35),
    (-5.243841377047226e-36, -1.1548379716377026e-35),
    (4.211067589174661e-37, -2.7493469328542347e-37),
    (1.2923519298859778e-38, 1.448843596340657e-38),
    (-4.673819687444993e-40, 5.6106598232391615e-40),
    (-2.2847096126218506e-41, -1.3966922082756917e-41),
    (3.772798697382695e-43, -8.805819031464555e-43),
    (3.2309079427925244e-44, 8.689265480319489e-45),
    (-1.3941164374376023e-46, 1.1327617267509538e-45),
    (-3.804472338043379e-47, 6.121338093808064e-49),
    (-1.8900728501624485e-49, -1.225917996210053e-48),
    (3.792813514337617e-50, -1.1246051243202504e-50),
    (5.058014301198256e-52, 1.1266249089369788e-51),
    (-3.210106836283472e-53, 1.9868100511680169e-53),
    (-7.153223556992548e-55, -8.755761001690951e-55),
    (2.2776852002593256e-56, -2.4137455694352547e-56),
    (7.72711942995603e-58, 5.614741328593284e-58),
    (-1.296707169590812e-59, 2.364272491307093e-59),
    (-6.947468369322094e-61, -2.7445014476696414e-61),
];

/// The Taylor coefficients of the entire `ζ(s) − 1/(s − 1)` about `s = 1`, `(−1)ⁿγₙ/n!` with
/// `γₙ` the Stieltjes constants (the first with its low part): 30 terms leave below `2⁻⁶²` on
/// `|s − 1| ≤ 3`.
const STIELTJES: [f64; 30] = [
    0.5772156649015329,
    0.07281584548367673,
    -0.00484518159643616,
    -0.00034230573671722433,
    9.689041939447084e-05,
    -6.6110318108421895e-06,
    -3.316240908752772e-07,
    1.0462094584479188e-07,
    -8.733218100273798e-09,
    9.47827778276236e-11,
    5.658421927608708e-11,
    -6.768689863513697e-12,
    3.4921159366720317e-13,
    4.4104247417577536e-15,
    -2.3997862217709992e-15,
    2.1677312200726828e-16,
    -9.544466076366965e-18,
    -7.387676660538637e-20,
    4.800850782488065e-20,
    -4.139956737713306e-21,
    1.9168201593991233e-22,
    -2.0441543122262165e-24,
    -4.818498501107353e-25,
    4.8118570515125666e-26,
    -2.560263310318815e-27,
    6.927840895304667e-29,
    1.628607550485587e-30,
    -3.1939375611532554e-31,
    2.0991515893634255e-32,
    -8.33674529544144e-34,
];
const STIELTJES_LO: f64 = -4.942915152430645e-18;

/// The terms of the Riemann–Siegel correction.
const RS_TERMS: usize = 9;

/// Riemann–Siegel from here (for `0 ≤ σ < 10`, at `σ` itself).
const T_RS: f64 = 1e4;

/// Past this `t` (and `σ < 56`) NaN: the phases `t ln n` would need more than double-double
/// arithmetic.
const T_MAX: f64 = 1_099_511_627_776.0; // 2^40

/// Past this `t` the phases `t ln n` take `ln n` from `ln_dd_acc` (`ln_dd`, good to `2⁻⁷⁵`,
/// would cost `t·2⁻⁷⁵` radians).
const T_ACC: f64 = 1e6;

/// `2⁻⁶⁰⁰`: below it the imaginary part is linear in `t` to far below an ulp (but at `σ = 1`).
const TINY_T: f64 = f64::from_bits((1023 - 600) << 52);

/// `2⁻⁵⁰⁰`: at `σ = 1` below it `ζ = γ − i/t` to far below an ulp, and `t²` (in the reciprocal of
/// the series about the pole) would leave the normal range.
const TINY_POLE_T: f64 = f64::from_bits((1023 - 500) << 52);

type C = (f64, f64);

#[inline(always)]
fn mul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline(always)]
fn div(a: C, b: C) -> C {
    let z = Complex::new(a.0, a.1) / Complex::new(b.0, b.1);
    (z.re, z.im)
}

/// `(cos 2πφ, sin 2πφ)` of the turns `φ = ph + pl` (a double-double), reduced exactly.
#[inline(always)]
fn turn(ph: f64, pl: f64) -> C {
    let f = ph - ph.round();
    let (fh, fl) = two_sum(f, pl);
    let (s, c) = sincos_pi(2.0 * fh);
    let d = 2.0 * PI_HI * fl;
    (c - d * s, s + d * c)
}

/// `(cos 2πφ, sin 2πφ)` of the turns `φ = ph + pl` as double-doubles, to about `2⁻⁶²`: `4φ`
/// reduced exactly to the quadrant `n` and `r = j/32 + ρ`, `|ρ| ≤ 1/64`, then
/// `sin(πj/32 + θ)` and `cos(πj/32 + θ)` of `θ = π(ρ + 2 pl)` from the table and the series of
/// `θ` (the terms past the first in double: below `2⁻¹¹` of the result).
#[inline(always)]
pub(super) fn turn_dd(ph: f64, pl: f64) -> (Dd, Dd) {
    let f = ph - ph.round();
    let (fh, fl) = two_sum(f, pl);
    // 2φ = n/2 + r, |r| ≤ 1/4, and r = j/32 + ρ, each exactly
    let x = 2.0 * fh;
    let n = (2.0 * x).round();
    let r = x - 0.5 * n;
    let j = (32.0 * r).round();
    let rho = r - j * 0.031_25;
    let (th, te) = two_prod(PI_HI, rho);
    let (th, tl) = fast_two_sum(th, te + (PI_LO * rho + PI_HI * (2.0 * fl)));
    let t2 = th * th;
    // sin θ − θ and cos θ − 1 (|θ| ≤ π/64 + 2⁻⁵⁰)
    let sp = th * t2 * (-1.0 / 6.0 + t2 * (1.0 / 120.0 + t2 * (-1.0 / 5040.0 + t2 / 362_880.0)));
    let cq = -0.5 * t2 - th * tl + t2 * t2 * (1.0 / 24.0 + t2 * (-1.0 / 720.0 + t2 / 40_320.0));
    let (sh, sl, ch, cl) = SINCOS_PI_32[j.abs() as usize];
    let (sh, sl) = if j < 0.0 { (-sh, -sl) } else { (sh, sl) };
    // sin(πr) = sin(πj/32)(1 + cq) + cos(πj/32)(θ + sp), and cos(πr) likewise
    let (p, pe) = two_prod(ch, th);
    let (h, e) = two_sum(sh, p);
    let s = Dd {
        hi: h,
        lo: e + (pe + (sl + (sh * cq + (ch * (tl + sp) + cl * th)))),
    };
    let (p, pe) = two_prod(sh, th);
    let (h, e) = two_sum(ch, -p);
    let c = Dd {
        hi: h,
        lo: e - (pe - (cl + (ch * cq - (sh * (tl + sp) + sl * th)))),
    };
    // sin(π(n/2 + r)) and cos(π(n/2 + r)) by n mod 4
    let (s, c) = match (n as i64) & 3 {
        0 => (s, c),
        1 => (c, s.neg()),
        2 => (s.neg(), c.neg()),
        _ => (c.neg(), s),
    };
    (c, s)
}

/// `e^(a + al) · e^(2πi (ph + pl))`: a modulus from a double-double exponent and a phase in
/// turns.
#[inline(always)]
fn polar(a: f64, al: f64, ph: f64, pl: f64) -> C {
    let m = exp_mul(a, 1.0, al);
    let (c, s) = turn(ph, pl);
    (m * c, m * s)
}

/// The exponents of `n^(−s)`, `s = σ + σl + it`: `−σ ln n` as a double-double and `−t ln n/2π`
/// in turns as a double-double (`ln n` from `ln_dd_acc` past `t = 10⁶`).
#[inline(always)]
fn pow_neg_exponents(n: f64, sigma: f64, sl: f64, t: f64) -> (f64, f64, f64, f64) {
    let (l, ll) = if t > T_ACC { ln_dd_acc(n) } else { ln_dd(n) };
    // −(σ + σl) ln n
    let (a, ae) = two_prod(-sigma, l);
    let al = ae - (sigma * ll + sl * l);
    // −t ln n / 2π, in turns
    let (u, ue) = two_prod(l, INV_2PI_HI);
    let ul = ue + (l * INV_2PI_LO + ll * INV_2PI_HI);
    let (p, pe) = two_prod(-t, u);
    let pl = pe - t * ul;
    (a, al, p, pl)
}

/// `n^(−s)` for an integer `n ≥ 1`, `s = σ + σl + it` (`σl` the low part of a `σ` that is not
/// a double).
#[inline(always)]
fn pow_neg(n: f64, sigma: f64, sl: f64, t: f64) -> C {
    let (a, al, p, pl) = pow_neg_exponents(n, sigma, sl, t);
    polar(a, al, p, pl)
}

/// `n^(−s)` as two double-doubles `(re, rl, im, il)`, to about `2⁻⁶²` (for `−σ ln n ≤ 708`): the
/// modulus from `exp_parts`, the phase from [`turn_dd`] and the products exact. Rounded to
/// doubles, the terms of the Riemann–Siegel sums (`√(t/2π)` of them) cost their errors as a random
/// walk, up to 7 ε at `t = 10¹⁰`.
#[inline(always)]
fn pow_neg_dd(n: f64, sigma: f64, sl: f64, t: f64) -> (f64, f64, f64, f64) {
    let (a, al, p, pl) = pow_neg_exponents(n, sigma, sl, t);
    let (mh, ml) = if a.abs() <= 708.0 {
        let (e, el, q) = exp_parts(a);
        (scale(e, q), scale(el + e * al, q))
    } else {
        (exp_mul(a, 1.0, al), 0.0)
    };
    let (c, s) = turn_dd(p, pl);
    let (rh, re) = two_prod(mh, c.hi);
    let (ih, ie) = two_prod(mh, s.hi);
    (
        rh,
        re + (mh * c.lo + ml * c.hi),
        ih,
        ie + (mh * s.lo + ml * s.hi),
    )
}

/// A complex double-double sum, added to term by term.
#[derive(Clone, Copy)]
struct Acc {
    rh: f64,
    rl: f64,
    ih: f64,
    il: f64,
}

impl Acc {
    fn new(v: C) -> Acc {
        Acc {
            rh: v.0,
            rl: 0.0,
            ih: v.1,
            il: 0.0,
        }
    }

    #[inline(always)]
    fn add(&mut self, v: C) {
        let (a, e) = two_sum(self.rh, v.0);
        self.rh = a;
        self.rl += e;
        let (a, e) = two_sum(self.ih, v.1);
        self.ih = a;
        self.il += e;
    }

    /// Adds the complex double-double `(re, rl, im, il)`.
    #[inline(always)]
    fn add4(&mut self, v: (f64, f64, f64, f64)) {
        let (a, e) = two_sum(self.rh, v.0);
        self.rh = a;
        self.rl += e + v.1;
        let (a, e) = two_sum(self.ih, v.2);
        self.ih = a;
        self.il += e + v.3;
    }

    fn value(self) -> C {
        (self.rh + self.rl, self.ih + self.il)
    }
}

/// `Σ_{n=2}^{m} n^(−s)` from the smallest term, added to `acc`, then the term 1; each term a
/// double-double where `precise`.
fn dirichlet_sum(acc: &mut Acc, m: f64, sigma: f64, sl: f64, t: f64, precise: bool) {
    let mut n = m;
    if precise {
        while n >= 2.0 {
            acc.add4(pow_neg_dd(n, sigma, sl, t));
            n -= 1.0;
        }
    } else {
        while n >= 2.0 {
            acc.add(pow_neg(n, sigma, sl, t));
            n -= 1.0;
        }
    }
    acc.add((1.0, 0.0));
}

/// `ζ(s)` for `σ ≥ 1/2`, `t > 0`, by the Euler–Maclaurin formula (`s = σ + σl + it`); in the
/// strip, where the bound is absolute near the zeros, its direct terms as double-doubles.
fn zeta_em(sigma: f64, sl: f64, t: f64) -> C {
    let abs_s = sigma.hypot(t);
    let big_n = ((abs_s + 40.0) / 2.83).ceil();
    // The remainder at N: P = N^(−s), N P/(s − 1) + P/2 + Σ B₂ₖ/(2k)! (s)₂ₖ₋₁ N^(1−2k) P.
    let p = pow_neg(big_n, sigma, sl, t);
    let s = (sigma, t);
    let r = 1.0 / big_n;
    let r2 = r * r;
    let mut fac = (sigma * r, t * r); // (s)₁ N⁻¹
    let mut acc = (0.0, 0.0);
    let mut k2 = 1.0;
    for &b in BERNOULLI.iter() {
        let term = (b * fac.0, b * fac.1);
        acc.0 += term.0;
        acc.1 += term.1;
        if term.0.abs() + term.1.abs() <= 1e-19 * (acc.0.abs() + acc.1.abs() + 1.0) {
            break;
        }
        // × (s + 2k − 1)(s + 2k) / N²
        let f = mul((sigma + k2, t), (sigma + k2 + 1.0, t));
        fac = mul(fac, (f.0 * r2, f.1 * r2));
        k2 += 2.0;
    }
    let q = div((big_n, 0.0), (s.0 - 1.0, s.1));
    let bracket = (q.0 + 0.5 + acc.0, q.1 + acc.1);
    // The direct terms from the smallest, as double-doubles.
    let mut sum = Acc::new(mul(p, bracket));
    dirichlet_sum(&mut sum, big_n - 1.0, sigma, sl, t, sigma <= 1.0);
    sum.value()
}

/// `ζ(s)` for `|s − 1| ≤ 3`, `t > 0`, by the Taylor series about the pole:
/// `1/(s − 1) + Σ (−1)ⁿγₙ/n! (s − 1)ⁿ`, the polynomial by complex Horner's rule (its terms all
/// below its constant), the reciprocal of the exact `s − 1` as a double-double (the low part of
/// `σ − 1` to first order) and the real part's sum rounded once: in the critical strip near the
/// real axis, where the Euler–Maclaurin terms cancel against the pole and the reflection's
/// factors are each a few ulps off. (`dr² + t²` stays normal: `|σ − 1| ≥ 2⁻⁵³` but at `σ = 1`,
/// where `zeta_upper` takes `t < 2⁻⁵⁰⁰` itself.)
fn zeta_disk(sigma: f64, t: f64) -> C {
    let (dr, drl) = two_sum(sigma, -1.0);
    let n = STIELTJES.len();
    let mut q = (STIELTJES[n - 1], 0.0);
    for &a in STIELTJES[1..n - 1].iter().rev() {
        q = mul(q, (dr, t));
        q.0 += a;
    }
    let dq = mul(q, (dr, t));
    // 1/d = (dr − it)/(dr² + t²), as double-doubles
    let (a, ae) = two_prod(dr, dr);
    let (b, be) = two_prod(t, t);
    let den = Dd { hi: a, lo: ae }.add(Dd { hi: b, lo: be });
    let ir = Dd::new(dr).div(den);
    let ii = Dd::new(-t).div(den);
    // the low part of σ − 1 through ζ′ ≈ −1/d² + a₁
    let d2 = mul((ir.hi, ii.hi), (ir.hi, ii.hi));
    let (cr, ci) = (drl * (STIELTJES[1] - d2.0), -drl * d2.1);
    let (h, l) = two_sum(ir.hi, STIELTJES[0]);
    let re = h + (l + (ir.lo + (STIELTJES_LO + (dq.0 + cr))));
    let im = ii.hi + (ii.lo + (dq.1 + ci));
    (re, im)
}

/// `ζ(s)` for `10 ≤ σ < 56` by the Dirichlet series, whose tail past `N` is below
/// `N^(1−σ)/(σ − 1) < 2⁻⁶⁰` (at any `t`).
fn zeta_direct(sigma: f64, sl: f64, t: f64) -> C {
    let sm1 = sigma - 1.0;
    let big_n = ((60.0 * core::f64::consts::LN_2 - sm1.ln()) / sm1)
        .exp()
        .ceil();
    let mut sum = Acc::new((0.0, 0.0));
    dirichlet_sum(&mut sum, big_n, sigma, sl, t, false);
    sum.value()
}

/// Stirling's series `Σ_{k≤10} B₂ₖ/(2k(2k − 1)) w^(1−2k)` at `w = u − it`, `|w| ≥ 13` (its
/// remainder below `10⁻²⁰`).
fn stirling_tail(u: f64, t: f64) -> C {
    let r = div((1.0, 0.0), (u, -t));
    let r2 = mul(r, r);
    let mut acc = (STIRLING[9], 0.0);
    for &b in STIRLING[..9].iter().rev() {
        acc = mul(acc, r2);
        acc.0 += b;
    }
    mul(acc, r)
}

/// `ln χ(s)`, `χ(s) = 2ˢ π^(s−1) sin(πs/2) Γ(1 − s)`, for `t > 0` with `|1 − s| ≥ 13`: its real
/// part and its imaginary part in turns, as double-doubles. With `with_sin` false the factor
/// `sin(πs/2)` is left out; with it, `t ≥ 13.5`, where
/// `ln sin(πs/2) = πt/2 − ln 2 + iπ(1 − σ)/2` but for `e^(−πt) < 2⁻⁶¹`.
///
/// With `w = 1 − s = u − it`, `λ = ln |w|` and `α = atan(u/t) = arg w + π/2`, Stirling's
/// `ln Γ(w) = (w − ½) ln w − w + ½ ln 2π + S(w)` gives
/// `ln χ(s) = (u − ½)(λ − ln 2π) + (tα − u) + Re S + i [t(1 + ln 2π − λ) + (u − ½)α + π/4 + Im S]`:
/// the parts of size `πt/2` cancel in closed form. Without the sine (`t < 13.5`, `u ≥ 13`)
/// `α = π/2 − β`, `β = atan(t/u)`, and the parts of size `u` cancel in closed form too:
/// `ln χ − ln sin(πs/2) = (u − ½)(λ − ln 2π) − tβ − u + ln 2 + Re S + i [t(1 + ln 2π − λ) − (u − ½)β + Im S]`,
/// so that the phase, of the order of `t`, keeps its digits for any `t` (at a trivial zero the
/// real part of `ζ`, of the order of `t²`, is that phase times the imaginary part). `σl` is the
/// low part of a `σ` that is not a double.
fn ln_chi(sigma: f64, sl: f64, t: f64, with_sin: bool) -> (Dd, Dd) {
    // u = 1 − σ exactly
    let (uh, ul) = two_sum(1.0, -sigma);
    let u = Dd {
        hi: uh,
        lo: ul - sl,
    };
    // λ = ½ ln(u² + t²)
    let (t2, t2e) = two_prod(t, t);
    let q = u.mul(u).add(Dd { hi: t2, lo: t2e });
    let (l, ll) = ln_dd_acc(q.hi);
    let lam = Dd {
        hi: l,
        lo: ll + q.lo / q.hi,
    }
    .mul_f64(0.5);
    let um = u.add_f64(-0.5);
    let (sr, si) = stirling_tail(uh, t);
    if with_sin {
        let alpha = atan_dd(u.div(Dd::new(t)));
        let e = um
            .mul(lam.sub(LN_2PI))
            .add(alpha.mul_f64(t).sub(u))
            .add_f64(sr);
        let ph = ONE_LN_2PI
            .sub(lam)
            .mul_f64(t)
            .add(um.mul(alpha))
            .add(PI_4)
            .add_f64(si);
        return (e, ph.mul(INV_2PI));
    }
    let beta = atan_dd(Dd::new(t).div(u));
    let e = um
        .mul(lam.sub(LN_2PI))
        .sub(beta.mul_f64(t))
        .sub(u)
        .add(LN_2)
        .add_f64(sr);
    let ph = ONE_LN_2PI.sub(lam).mul_f64(t).sub(um.mul(beta)).add_f64(si);
    (e, ph.mul(INV_2PI))
}

/// The binary exponent of a finite `x ≠ 0` (`⌊log₂|x|⌋`, subnormals included), else 0.
#[inline(always)]
fn exponent(x: f64) -> i32 {
    let x = x.abs();
    if x > 0.0 && x.is_finite() {
        let (b, e0) = if x < f64::MIN_POSITIVE {
            (x * 18_446_744_073_709_551_616.0, -64)
        } else {
            (x, 0)
        };
        ((b.to_bits() >> 52) as i32) - 1023 + e0
    } else {
        0
    }
}

/// `d·2^q` of a double-double, both parts scaled exactly (for a `d` scaled towards 1).
#[inline(always)]
fn scale_dd(d: Dd, q: i32) -> Dd {
    Dd {
        hi: scale(d.hi, q),
        lo: scale(d.lo, q),
    }
}

/// `e^(E + 2πiΦ) · m · (2^k₀, 2^k₁)` of the double-double `E`, turns `Φ` and the complex
/// double-double `m`, the phase and its products as double-doubles and each part rounded once
/// (an infinity in the direction of the phase past the range; the scalings inside, so that a
/// part in the range is not lost to an overflow of `e^E m` alone).
fn exp_polar_mul(e: Dd, ph: Dd, m: Cd, k: (i32, i32)) -> C {
    // m scaled to about 1 first, its exponent back in the last scalings: near a trivial zero with
    // a tiny t, m (of the order of t) times the sine of the phase (also of the order of t) would
    // go subnormal before e^E brings it back (ζ(−356 + 2·10⁻¹⁵⁷ i), whose imaginary part
    // overflows, had its real part 10⁻¹¹ off)
    let q = exponent(m.0.hi.abs().max(m.1.hi.abs()));
    let m = (scale_dd(m.0, -q), scale_dd(m.1, -q));
    let (c, s) = turn_dd(ph.hi, ph.lo);
    let pr = c.mul(m.0).sub(s.mul(m.1));
    let pi = c.mul(m.1).add(s.mul(m.0));
    (
        exp_mul_k(e.hi, pr.hi, pr.lo + pr.hi * e.lo, k.0 + q),
        exp_mul_k(e.hi, pi.hi, pi.lo + pi.hi * e.lo, k.1 + q),
    )
}

/// `e^(E + 2πiΦ) · (mr + i mi)` of the double-doubles `E` (`|E| ≤ 708`), `Φ` (turns), `mr` and
/// `mi`, as a complex double-double `(re, rl, im, il)`.
fn exp_polar_mul_dd(e: Dd, ph: Dd, mr: Dd, mi: Dd) -> (f64, f64, f64, f64) {
    let (c, s) = turn_dd(ph.hi, ph.lo);
    let pr = c.mul(mr).sub(s.mul(mi));
    let pi = c.mul(mi).add(s.mul(mr));
    let (eh, el) = exp_dd(e.hi, e.lo);
    let ex = Dd { hi: eh, lo: el };
    let (r, i) = (ex.mul(pr), ex.mul(pi));
    (r.hi, r.lo, i.hi, i.lo)
}

/// `ζ(s)` for `σ < 1/2`, `t > 0`, by the reflection with `χ(s)` in its logarithmic form
/// (`|1 − s| ≥ 13`), each part scaled by `2^k₀`, `2^k₁` before it is rounded.
fn zeta_reflect_log(sigma: f64, t: f64, k: (i32, i32)) -> C {
    let with_sin = t >= 13.5;
    let (e, ph) = ln_chi(sigma, 0.0, t, with_sin);
    // ζ(1 − s) = conj ζ(1 − σ + it), 1 − σ > 1/2 as an exact pair: it rounds (up to 2⁻⁵⁴ near
    // σ = 1/2), and the rounding, times ζ′/ζ ≈ ½ ln t, cost up to 13 ε in the strip
    let (zr, zl) = two_sum(1.0, -sigma);
    let z1 = zeta_upper(zr, zl, t);
    let z1 = (Dd::new(z1.0), Dd::new(-z1.1));
    if with_sin {
        return exp_polar_mul(e, ph, z1, k);
    }
    // sin(πs/2) = sin(πσ/2) cosh(πt/2) + i cos(πσ/2) sinh(πt/2) and its product with ζ(1 − s) as
    // double-doubles (in double their roundings reached 2.57 ε of |ζ| at ζ(−256.04 + 10.61i)),
    // the sine scaled to about 1 first: near a trivial zero with a tiny t it is of the order of
    // t, and the product's real part of t², which would go subnormal
    let (cs, sn) = turn_dd(0.25 * sigma, 0.0);
    let (ch, sh) = cosh_sinh_half_pi_dd(t);
    let (a, b) = (sn.mul(ch), cs.mul(sh));
    let q = exponent(a.hi.abs().max(b.hi.abs()));
    let m = cmul_dd((scale_dd(a, -q), scale_dd(b, -q)), z1);
    exp_polar_mul(e, ph, m, (k.0 + q, k.1 + q))
}

/// The d coefficients of the Riemann–Siegel correction, `d[k][ℓ]` for `k < 9`, `ℓ ≤ ⌊3k/2⌋`,
/// by Arias de Reyna's recurrence in `1 − 2σ`.
fn rs_d(sigma: f64) -> [[f64; 13]; RS_TERMS] {
    let mut d = [[0.0f64; 13]; RS_TERMS];
    d[0][0] = 1.0;
    let ps = 1.0 - 2.0 * sigma;
    for n in 1..RS_TERMS {
        let prev = d[n - 1];
        let at = |k: isize| -> f64 {
            if k < 0 || k as usize > 3 * (n - 1) / 2 {
                0.0
            } else {
                prev[k as usize]
            }
        };
        for k in 0..=3 * n / 2 {
            let m = 3 * n - 2 * k;
            let ki = k as isize;
            let v = if m != 0 {
                let mf = m as f64;
                -(mf + 1.0) * at(ki - 2) + at(ki) / (4.0 * mf) + ps * at(ki - 1) / (2.0 * mf)
            } else {
                // m = 0: d[n][k] = −Σ_{r<k} (−1)^(k−r) d[n][r] (2k − 2r)!/(k − r)!
                let mut v = 0.0;
                for (r, &dr) in d[n][..k].iter().enumerate() {
                    let j = k - r;
                    let mut f = 1.0; // (2j)!/j!
                    for i in j + 1..=2 * j {
                        f *= i as f64;
                    }
                    let sgn = if j % 2 == 0 { 1.0 } else { -1.0 };
                    v -= sgn * dr * f;
                }
                v
            };
            d[n][k] = v;
        }
    }
    d
}

/// `F⁽ᵐ⁾(p)`, `m = 0, …, 24`, of the Riemann–Siegel function, by repeated synthetic division of
/// its Taylor polynomial by `z − p`.
fn rs_fderivs(p: f64) -> [C; 3 * RS_TERMS - 2] {
    let mut a = [(0.0f64, 0.0f64); 101];
    for (j, &c) in RS_F.iter().enumerate() {
        a[2 * j] = c;
    }
    let mut out = [(0.0, 0.0); 3 * RS_TERMS - 2];
    let mut fact = 1.0;
    let mut deg = 100;
    for (m, o) in out.iter_mut().enumerate() {
        let mut s = a[deg];
        for k in (0..deg).rev() {
            let v = (a[k].0 + p * s.0, a[k].1 + p * s.1);
            a[k] = s;
            s = v;
        }
        *o = (s.0 * fact, s.1 * fact);
        deg -= 1;
        fact *= (m + 1) as f64;
    }
    out
}

/// `R(σ + σl + it)` of the Riemann–Siegel formula: the direct sum to `N`, each term a
/// double-double, as a double-double, and the correction `(−1)^(N−1) U a^(−σ) Σ_k a^(−k) C_k`
/// (`ln a` a double-double; `σl` changes it by far less than an ulp).
fn rs_part(sigma: f64, sl: f64, t: f64, nn: f64, ln_a: Dd, fp: &[C], u: C) -> Acc {
    let d = rs_d(sigma);
    let mut pinv = [1.0f64; 2 * RS_TERMS];
    for j in 1..pinv.len() {
        pinv[j] = pinv[j - 1] / PI_HI;
    }
    let inv_a = exp_mul(-ln_a.hi, 1.0, -ln_a.lo);
    let mut sum = (0.0, 0.0);
    for k in (0..RS_TERMS).rev() {
        let mut term = (0.0, 0.0);
        for (l, &dkl) in d[k].iter().enumerate().take(3 * k / 2 + 1) {
            let f = fp[3 * k - 2 * l];
            // f/(2i)^ℓ = f (−i)^ℓ 2^(−ℓ)
            let f = match l % 4 {
                0 => f,
                1 => (f.1, -f.0),
                2 => (-f.0, -f.1),
                _ => (-f.1, f.0),
            };
            let c = dkl * pinv[2 * k - l] * 0.5f64.powi(l as i32);
            term.0 += c * f.0;
            term.1 += c * f.1;
        }
        sum = (sum.0 * inv_a + term.0, sum.1 * inv_a + term.1);
    }
    let sgn = if (nn as u64) % 2 == 1 { 1.0 } else { -1.0 }; // (−1)^(N−1)
    let (x, xe) = two_prod(-sigma, ln_a.hi);
    let m = sgn * exp_mul(x, 1.0, xe - sigma * ln_a.lo);
    let corr = mul(u, sum);
    let mut acc = Acc::new((m * corr.0, m * corr.1));
    dirichlet_sum(&mut acc, nn, sigma, sl, t, true);
    acc
}

/// `ζ(s)` for `0 ≤ σ < 10`, `t > 10⁴`, by the Riemann–Siegel formula (`s = σ + σl + it`): the
/// second part `χ(s) conj R(1 − σ + it)` as a double-double, and on the critical line one sum
/// for both.
fn zeta_rs(sigma: f64, sl: f64, t: f64) -> C {
    // a = √(t/2π) as a double-double, N = ⌊a⌋ and p = 1 − 2(a − N) (a − N exact)
    let (q, qe) = two_prod(t, INV_2PI_HI);
    let a = Dd {
        hi: q,
        lo: qe + t * INV_2PI_LO,
    }
    .sqrt();
    let mut nn = a.hi.floor();
    if nn == a.hi && a.lo < 0.0 {
        nn -= 1.0;
    }
    let p = 1.0 - 2.0 * Dd::new(a.hi - nn).add_f64(a.lo).value();
    // ln a = ½ (ln t − ln 2π), and θ′ = (t/2)(ln(t/2π) − 1) − π/8 in turns for U = e^(−iθ′)
    let (lt, ltl) = ln_dd_acc(t);
    let lq = Dd { hi: lt, lo: ltl }.sub(LN_2PI);
    let ln_a = lq.mul_f64(0.5);
    let th = lq
        .add_f64(-1.0)
        .mul_f64(0.5 * t)
        .sub(PI_4.mul_f64(0.5))
        .mul(INV_2PI);
    let u = turn(-th.hi, -th.lo);
    let fp = rs_fderivs(p);
    let rx = rs_part(sigma, sl, t, nn, ln_a, &fp, u);
    // R(1 − σ + it), 1 − σ as an exact pair (it rounds for σ < 1/2)
    let ry = if sigma == 0.5 && sl == 0.0 {
        rx
    } else {
        let (u1, u1l) = two_sum(1.0, -sigma);
        rs_part(u1, u1l - sl, t, nn, ln_a, &fp, u)
    };
    let (e, ph) = ln_chi(sigma, sl, t, true);
    let (rh, rl) = two_sum(ry.rh, ry.rl);
    let (ih, il) = two_sum(ry.ih, ry.il);
    let second = exp_polar_mul_dd(e, ph, Dd { hi: rh, lo: rl }, Dd { hi: -ih, lo: -il });
    let mut acc = rx;
    acc.add4(second);
    acc.value()
}

/// `ψ(w)` to a few digits for `Re w ≥ 1`: the recurrence to `|w| ≥ 6`, then three terms of
/// the asymptotic series.
fn digamma_rough_c(w: C) -> C {
    let (mut w, mut acc) = (w, (0.0, 0.0));
    while w.0 * w.0 + w.1 * w.1 < 36.0 {
        let r = div((1.0, 0.0), w);
        acc = (acc.0 - r.0, acc.1 - r.1);
        w = (w.0 + 1.0, w.1);
    }
    let lw = (0.5 * (w.0 * w.0 + w.1 * w.1).ln(), w.1.atan2(w.0));
    let r = div((1.0, 0.0), w);
    let r2 = mul(r, r);
    (
        acc.0 + lw.0 - 0.5 * r.0 - r2.0 / 12.0,
        acc.1 + lw.1 - 0.5 * r.1 - r2.1 / 12.0,
    )
}

/// A complex double-double.
pub(super) type Cd = (Dd, Dd);

/// The product of complex double-doubles.
#[inline(always)]
pub(super) fn cmul_dd(a: Cd, b: Cd) -> Cd {
    (
        a.0.mul(b.0).sub(a.1.mul(b.1)),
        a.0.mul(b.1).add(a.1.mul(b.0)),
    )
}

/// `(cosh a, sinh a)` of `a = πt/2` as double-doubles: below `|a| = 2⁻⁶` their series (the
/// terms past the first in double), else from `e^(±a)` as double-doubles.
fn cosh_sinh_half_pi_dd(t: f64) -> (Dd, Dd) {
    let (h, e) = two_prod(0.5 * PI_HI, t);
    let (h, l) = fast_two_sum(h, e + 0.5 * PI_LO * t);
    let a = Dd { hi: h, lo: l };
    if h.abs() < 0.015_625 {
        let a2 = h * h;
        let sh = a.add_f64(h * a2 * (1.0 / 6.0 + a2 * (1.0 / 120.0 + a2 / 5040.0)));
        let ch = Dd::new(1.0).add_f64(a2 * (0.5 + a2 * (1.0 / 24.0 + a2 / 720.0)) + h * l);
        return (ch, sh);
    }
    let (ph, pl) = exp_dd(h, l);
    let (mh, ml) = exp_dd(-h, -l);
    let (ep, em) = (Dd { hi: ph, lo: pl }, Dd { hi: mh, lo: ml });
    (ep.add(em).mul_f64(0.5), ep.sub(em).mul_f64(0.5))
}

/// `ζ(s)` for `−12 < σ < 1/2`, `t > 0` small (`t < 13.5`), by the reflection formula with
/// `Γ(1 − s)` from `gamma_c64`: `2(2π)^(s−1)`, `sin(πs/2)` and the products as double-doubles,
/// so that only `Γ(1 − s)`, `ζ(1 − s)` and the result are rounded (in double, the four factors
/// and three products reached 2.8 ε).
fn zeta_reflect(sigma: f64, t: f64) -> C {
    // 2 (2π)^(s−1): modulus e^((σ − 1) ln 2π) (doubled), phase t ln 2π / 2π turns.
    let (sm1, sm1l) = two_sum(sigma, -1.0);
    let (a, ae) = two_prod(sm1, LN_2PI_HI);
    let al = ae + sm1 * LN_2PI_LO + sm1l * LN_2PI_HI;
    let (ph, pe) = two_prod(t, LN_2PI_OVER_2PI_HI);
    let pl = pe + t * LN_2PI_OVER_2PI_LO;
    let (mh, ml) = exp_dd(a, al);
    let m = Dd {
        hi: 2.0 * mh,
        lo: 2.0 * ml,
    };
    let (c, s) = turn_dd(ph, pl);
    let w = (m.mul(c), m.mul(s));
    // sin(πs/2) = sin(πσ/2) cosh(πt/2) + i cos(πσ/2) sinh(πt/2), sin(πσ/2) exactly 0 at the
    // trivial zeros
    let (cs, sn) = turn_dd(0.25 * sigma, 0.0);
    let (ch, sh) = cosh_sinh_half_pi_dd(t);
    let sin_s = (sn.mul(ch), cs.mul(sh));
    // 1 − σ rounds where it crosses a power of 2 (at σ = −7.x, by 2⁻⁵⁰, which ψ(1 − s) ~ 2
    // makes 8 ε of Γ): Γ(1 − s) (1 + zl ψ(1 − s)) with ψ to a few digits.
    let (zr, zl) = two_sum(1.0, -sigma);
    let g = gamma_c64(Complex::new(zr, -t));
    let g = if zl == 0.0 {
        (Dd::new(g.re), Dd::new(g.im))
    } else {
        let psi = digamma_rough_c((zr, -t));
        let (pr, pi) = (zl * psi.0, zl * psi.1);
        (
            Dd::new(g.re).add_f64(g.re * pr - g.im * pi),
            Dd::new(g.im).add_f64(g.re * pi + g.im * pr),
        )
    };
    // ζ(1 − s) at 1 − σ = zr + zl exactly (its rounding, times ζ′/ζ, cost a few tenths of ε)
    let z1 = zeta_em(zr, zl, -t);
    // the large factors together first: near a trivial zero sin(πs/2) is tiny, and its product
    // with (2π)^(s−1) would underflow before Γ(1 − s) brings it back
    let p = cmul_dd(w, g);
    let q = cmul_dd(sin_s, (Dd::new(z1.0), Dd::new(z1.1)));
    let r = cmul_dd(p, q);
    (r.0.value(), r.1.value())
}

/// `ζ(s)` for `t > 0`, `s = σ + σl + it` (`σl` nonzero only from the reflection, for an exact
/// `1 − σ`).
fn zeta_upper(sigma: f64, sl: f64, t: f64) -> C {
    if !t.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    if sigma >= 56.0 {
        if sigma > 1100.0 {
            return (1.0, 0.0);
        }
        let v = pow_neg(2.0, sigma, sl, t);
        return (1.0 + v.0, v.1);
    }
    if t > T_MAX || sigma < -1.125_899_906_842_624e15 || sigma.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if sigma == 1.0 && sl == 0.0 && t < TINY_POLE_T {
        // ζ(1 + it) = γ − i/t − iγ₁t − γ₂t²/2 + …, whose terms past the first two are below
        // 2⁻¹⁰⁰⁰ of them. The series about the pole squares t in its reciprocal, which goes
        // subnormal below t = 1.5·10⁻¹⁵⁴ (it made the real part NaN), and the scaling below is
        // for a part linear in t, not −1/t; −∞ where 1/t leaves the range (t < 2⁻¹⁰²⁴)
        return (STIELTJES[0], -1.0 / t);
    }
    if t < TINY_T {
        // ζ(σ + it) = ζ(σ) + it ζ′(σ) − t² ζ″(σ)/2 + O(t³): at t′ = m·2⁻⁶⁰⁰ (t = m·2ᵉ), the
        // imaginary part scaled back once, so that no subnormal intermediate costs the result
        // its digits; the real part ζ(σ), or at a zero of ζ(σ) the t′² term scaled back (below
        // 2⁻⁵⁹⁰ of the imaginary part; for −12 < σ < 0 it can underflow at t′ to 0). From
        // σ = −12 the scalings go inside the last products of the logarithmic form: t ζ′(σ) can
        // be in the range where t′ ζ′(σ) is not (|ζ′(−400)| ≈ 10⁵⁴⁹)
        let (m, e) = if t < f64::MIN_POSITIVE {
            (t * 18_446_744_073_709_551_616.0, -64)
        } else {
            (t, 0)
        };
        let b = m.to_bits();
        let k = ((b >> 52) as i32) - 1023 + e + 600;
        let tp = f64::from_bits((b & !(0x7ffu64 << 52)) | ((1023 - 600) << 52));
        let z = if sigma <= -12.0 {
            zeta_reflect_log(sigma, tp, (2 * k, k))
        } else {
            let z = zeta_upper(sigma, 0.0, tp);
            (scale(z.0, 2 * k), scale(z.1, k))
        };
        let z0 = zeta_f64(sigma);
        return (if z0 == 0.0 { z.0 } else { z0 }, z.1);
    }
    if (-0.5..=1.5).contains(&sigma) && (sigma - 1.0) * (sigma - 1.0) + t * t <= 9.0 {
        return zeta_disk(sigma, t);
    }
    if sigma >= 0.5 {
        if sigma >= 10.0 {
            return if t > 1e3 {
                zeta_direct(sigma, sl, t)
            } else {
                zeta_em(sigma, sl, t)
            };
        }
        return if t > T_RS {
            zeta_rs(sigma, sl, t)
        } else {
            zeta_em(sigma, sl, t)
        };
    }
    // 0 ≤ σ < 1/2 past T_RS: the Riemann–Siegel formula at σ itself (the reflection of its value
    // at 1 − σ rounds χ twice more)
    if sigma >= 0.0 && t > T_RS {
        return zeta_rs(sigma, 0.0, t);
    }
    // else the reflection, whose ζ(1 − s) has decaying terms (the Euler–Maclaurin formula at σ
    // itself reached 7.5 ε for −0.3 ≤ σ < 0, t < 13.5, and 5.7 ε for 0 ≤ σ < 1/2, t < 13.5): in
    // the logarithmic form where |1 − s| ≥ 13 (it needs no Γ, whose imaginary part at a tiny t
    // is good only relative to |Γ|), else with Γ(1 − s)
    if t >= 13.5 || sigma <= -12.0 {
        return zeta_reflect_log(sigma, t, (0, 0));
    }
    zeta_reflect(sigma, t)
}

/// `ζ(s)` of a `Complex<f64>`.
pub(crate) fn zeta_c64(s: Complex<f64>) -> Complex<f64> {
    let (sigma, t) = (s.re, s.im);
    if sigma.is_nan() || t.is_nan() {
        return Complex::new(f64::NAN, f64::NAN);
    }
    if t == 0.0 {
        return Complex::new(zeta_f64(sigma), 0.0);
    }
    let (ta, flip) = if t < 0.0 { (-t, true) } else { (t, false) };
    let z = zeta_upper(sigma, 0.0, ta);
    Complex::new(z.0, if flip { -z.1 } else { z.1 })
}

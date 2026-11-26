//! The digamma function `ψ` and the polygamma functions `ψ⁽ⁿ⁾`.
//!
//! * `ψ(x)`: `ln x − 1/(2x) − u·T(u)` (`u = 1/x²`, `T` minimax) for `x ≥ 8`; the recurrence
//!   `ψ(x) = ψ(2 + t) + Σ 1/(t + k)` over exact `t + k` down to two minimax pieces of `ψ(2 + t)`;
//!   `(x − x₀)·Q(x)` on `[1, 2]` in four pieces, with the positive zero `x₀` in double-double;
//!   `ψ(x) = ψ(1 + x) − 1/x` below 1; and for `x < 0` the reflection
//!   `ψ(−u) = ψ(u) + 1/u + π cot(πu)`, with `ψ(u) + 1/u = ψ(1 + u)` for `u < 1`; `∓∞` for
//!   `0 < |x| ≤ 2⁻¹⁰²⁴`, where `−1/x` overflows, and `ψ(−u) = 1/u − γ` below `2⁻⁶⁰`.
//! * `ψ⁽ⁿ⁾(x) = (−1)ⁿ⁺¹ n! ζ(n + 1, x)`: the Hurwitz zeta function by direct terms up to where
//!   its Euler–Maclaurin tail converges (or the rest is negligible); for `x < 0` the sum over all
//!   the terms `(x + j)^(−n−1)` split into three Hurwitz zeta functions of exact arguments. Large
//!   orders and arguments are carried as `v·e^l` so that nothing overflows before the end.

use super::super::dd::{Dd, fast_two_sum, two_prod, two_sum};
use super::ddm::*;
use super::real::lgamma_pos_dd;
use super::tab::*;

const PI_DD: Dd = Dd {
    hi: PI_HI,
    lo: PI_LO,
};

/// `1/a` of an `f64` as a double-double.
#[inline(always)]
fn recip(a: f64) -> Dd {
    let q = 1.0 / a;
    let (hi, lo) = fast_two_sum(q, q * rem(q, a, 1.0));
    Dd { hi, lo }
}

/// `a + b` of two double-doubles, lightly (one `two_sum`; for values that need no more).
#[inline(always)]
fn add_light(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum(a.hi, b.hi);
    let (hi, lo) = fast_two_sum(s, e + (a.lo + b.lo));
    Dd { hi, lo }
}

/// `ψ(x)` on `[2, 8)` from its eight pieces, as a double-double `Y + R`.
#[inline(always)]
fn psi_piece(x: f64) -> Dd {
    let (i, c) = if x < 4.0 {
        let i = ((x - 2.0) * 2.0) as usize;
        (i, 2.25 + 0.5 * i as f64)
    } else {
        let f = x.floor();
        (f as usize, f + 0.5)
    };
    // Y + c₁s + (c₀ + s²R(s)), the first two terms without error.
    let s = x - c;
    let [y, c0, c1, r @ ..] = PSI_PIECES[i.min(7)];
    let (p, e) = two_prod(c1, s);
    let (h, e2) = fast_two_sum(y, p);
    let (hi, lo) = fast_two_sum(h, e2 + (e + fma(s * s, poly(s, &r), c0)));
    Dd { hi, lo }
}

/// `ψ(1 + w)` for `0 ≤ w ≤ 1` (`d = 1 + w − x₀` as a double-double), as a double-double:
/// `(Y + R)·d`.
#[inline]
fn psi1p(w: f64, d: Dd) -> Dd {
    let i = ((w * 4.0) as usize).min(3);
    let s = w - (0.125 + 0.25 * i as f64);
    let [y, r @ ..] = PSI_Q[i];
    let r = poly(s, &r);
    let (p, e) = two_prod(y, d.hi);
    let (hi, lo) = fast_two_sum(p, e + fma(r, d.hi, (y + r) * d.lo));
    Dd { hi, lo }
}

/// `1 + w − x₀ = w − (x₀ − 1)` as a double-double (`x₀ − 1` is exact).
#[inline(always)]
fn shifted(w: f64) -> Dd {
    let (h, e) = two_sum(w, 1.0 - PSI_ZERO_HI);
    let (hi, lo) = fast_two_sum(h, e - PSI_ZERO_LO);
    Dd { hi, lo }
}

/// `ψ(x)` for `x > 0`, as a double-double.
pub(super) fn psi_pos_dd(x: f64) -> Dd {
    if x >= 8.0 {
        // ln x − 1/(2x) − u T(u).
        let r = 1.0 / x;
        let u = r * r;
        return ln_dd(x).add_f64(-0.5 * r).add_f64(-u * poly(u, &PSI_TAIL));
    }
    if x >= 2.0 {
        return psi_piece(x);
    }
    if x >= 1.0 {
        // x − x₀ is exact in its first step on [1, 2].
        // (x − x₀ is at least an ulp of x₀ apart from 0, so the sum is a double-double.)
        let (hi, lo) = fast_two_sum(x - PSI_ZERO_HI, -PSI_ZERO_LO);
        let d = Dd { hi, lo };
        return psi1p(x - 1.0, d);
    }
    // ψ(x) = ψ(1 + x) − 1/x; 1 + x − x₀ = x − (x₀ − 1), with x₀ − 1 exact.
    if x <= TWO_M1024 {
        // −1/x − γ is below −f64::MAX: −∞ (`recip`'s remainder would be NaN).
        return Dd::new(f64::NEG_INFINITY);
    }
    let d = shifted(x);
    add_light(psi1p(x, d), recip(x).neg())
}

/// `2⁻¹⁰²⁴`: `1/x` overflows from it down (and `1/x²` from its square root, `2⁻⁵¹²`, down).
const TWO_M1024: f64 = f64::from_bits(1 << 50);

/// `2⁻⁶⁰`: below it `ψ(−u) = 1/u − γ` to far below an ulp (the next term is `−(π²/6)u`).
const PSI_TINY: f64 = 8.673_617_379_884_035e-19;

// `PSI_F32` and `PSI_X0_F32` are printed by `gen_psi_f32.py` beside this file.
/// `ψ(x)` for the `f32` form on `1 ≤ x < 8`: Chebyshev interpolants of degree 15 in
/// `x − (5/4 + i/2)` on the pieces `[1 + i/2, 3/2 + i/2]`, `i = 0, …, 13` (mpmath at 40
/// digits), on the first two of `ψ(x)/(x − x₀)` about the zero `x₀`; relative error at most
/// 3.0e-16 (an `f32` needs about `2⁻³⁴`).
const PSI_F32: [[f64; 16]; 14] = [
    [
        1.0747589096650145,
        -0.5791664818235416,
        0.4002392309430917,
        -0.2998194686576,
        0.232266501441986,
        -0.18277734921196082,
        0.14495837899337005,
        -0.1154283212733535,
        0.09210917594536026,
        -0.0735857662951592,
        0.0588483566365841,
        -0.04705902991476532,
        0.0371783561836445,
        -0.0297388430670978,
        0.02832038610197798,
        -0.022655514199577842,
    ],
    [
        0.8581832171262995,
        -0.3262546278681057,
        0.15712011335204326,
        -0.08228558635577858,
        0.04479965445251827,
        -0.02489004848878596,
        0.013985786218320424,
        -0.0079104688002991,
        0.004491811548923788,
        -0.002556685168883097,
        0.0014574478762334245,
        -0.0008315404397966385,
        0.00047330940605914843,
        -0.00027029532745540235,
        0.00016811214230985058,
        -9.604142607609597e-05,
    ],
    [
        0.5725464666237345,
        0.5573291545071107,
        -0.15186996876846018,
        0.05409066819806601,
        -0.021294526580566932,
        0.008804064010764695,
        -0.0037399995723528946,
        0.00161454901771725,
        -0.0007039960400411402,
        0.00030893233503419927,
        -0.00013613557133862776,
        6.0155366084914054e-05,
        -2.6603001770213268e-05,
        1.1791720160782243e-05,
        -5.50223537069327e-06,
        2.4424164235385555e-06,
    ],
    [
        0.8189010249754326,
        0.43757125764893073,
        -0.09435731651565062,
        0.02676935458157823,
        -0.00843981139048204,
        0.002806739012918914,
        -0.0009624684430003966,
        0.00033635905054926843,
        -0.0001190014019821541,
        4.2448840171768824e-05,
        -1.522712257623552e-05,
        5.4835487841474625e-06,
        -1.9792453935959735e-06,
        7.161083257575634e-07,
        -2.6828409087094144e-07,
        9.730399188413057e-08,
    ],
    [
        1.016990911068179,
        0.3597982903095799,
        -0.06407847356955756,
        0.015072225887442623,
        -0.003952996664739608,
        0.0010967173815081064,
        -0.0003145121808946825,
        9.211017706915576e-05,
        -2.7356597894751917e-05,
        8.20369408024893e-06,
        -2.476943975837413e-06,
        7.5153170145168e-07,
        -2.287839579352684e-07,
        6.984557754757638e-08,
        -2.18784226874518e-08,
        6.702224424858437e-09,
    ],
    [
        1.1825373886117962,
        0.3053398526902531,
        -0.04627316925794964,
        0.009284210124232419,
        -0.002081577042356462,
        0.0004946537954187041,
        -0.00012171018206923996,
        3.062877384702963e-05,
        -7.826757290809211e-06,
        2.02169664764341e-06,
        -5.262959339435688e-07,
        1.3779364149584426e-07,
        -3.6226544132879714e-08,
        9.556016861892883e-09,
        -2.572111457184971e-09,
        6.814539154918619e-10,
    ],
    [
        1.3246832187604867,
        0.26512373409656215,
        -0.03494784088862902,
        0.006108954293310765,
        -0.0011950669434682769,
        0.00024812362111692763,
        -5.340640846535472e-05,
        1.1769939398593363e-05,
        -2.636524845351597e-06,
        5.975177573565245e-07,
        -1.3657929954617925e-07,
        3.14194933620696e-08,
        -7.262607509193851e-09,
        1.6851620318611334e-09,
        -3.9756446431729843e-10,
        9.272958697089879e-11,
    ],
    [
        1.4492040552784629,
        0.23422874157914197,
        -0.02731020629498667,
        0.004227420000775628,
        -0.0007330996761013189,
        0.00013505983108399918,
        -2.5818458246540908e-05,
        5.057647494309884e-06,
        -1.0077902704000231e-06,
        2.0330544220095978e-07,
        -4.139137414788814e-08,
        8.485758883662766e-09,
        -1.7489489781482675e-09,
        3.619914872978304e-10,
        -7.599597086957219e-11,
        1.5823119141088857e-11,
    ],
    [
        1.5599773364075455,
        0.20976041229725417,
        -0.02192117693585068,
        0.0030438568926570374,
        -0.0004738675550791646,
        7.842964737831298e-05,
        -1.347841464449103e-05,
        2.375117323096025e-06,
        -4.2597847547472335e-07,
        7.738919973843657e-08,
        -1.419608124292979e-08,
        2.6234419965991387e-09,
        -4.876006747569705e-10,
        9.104277669951441e-11,
        -1.7213203436631434e-11,
        3.235172646035971e-12,
    ],
    [
        1.6597303710679365,
        0.18990741193925276,
        -0.01797940005501,
        0.0022630397397279075,
        -0.00031954593693337774,
        4.799588599601157e-05,
        -7.489206649067849e-06,
        1.1988576843155554e-06,
        -1.9541346842064922e-07,
        3.227874704740737e-08,
        -5.3857498592977656e-09,
        9.056274544858464e-10,
        -1.5321231723800214e-10,
        2.6046927106195554e-11,
        -4.478361417011278e-12,
        7.667803089707703e-13,
    ],
    [
        1.750453526883736,
        0.17347923315893218,
        -0.015010476147598869,
        0.001727532932990026,
        -0.00022313918180925772,
        3.06718619935688e-05,
        -4.381693618825113e-06,
        6.424085563025168e-07,
        -9.593871039389973e-08,
        1.4524482580184457e-08,
        -2.221848628377965e-09,
        3.426357843035531e-10,
        -5.317604859074604e-11,
        8.295228858328804e-12,
        -1.3074963001106297e-12,
        2.055141438415318e-13,
    ],
    [
        1.8336434145461975,
        0.15966166524738132,
        -0.012719270195554093,
        0.0013482345467790543,
        -0.00016044938163792502,
        2.032691985767197e-05,
        -2.6772125380521894e-06,
        3.6198914326935366e-07,
        -4.9871113460920294e-08,
        6.967033141367556e-09,
        -9.837124934555613e-10,
        1.400557386872022e-10,
        -2.0072526205747896e-11,
        2.8921808397165556e-12,
        -4.2076778555210176e-13,
        6.111185132569597e-14,
    ],
    [
        1.910453526883736,
        0.14787923315893217,
        -0.01091447614759887,
        0.001072172932990026,
        -0.00011828158180925773,
        1.38946459935688e-05,
        -1.697339058825094e-06,
        2.1291182670251383e-07,
        -2.7219233659067798e-08,
        3.5293663026113474e-09,
        -4.626299840799816e-10,
        6.116080121587572e-11,
        -8.140805510128349e-12,
        1.0895899654299737e-12,
        -1.4716797597233166e-13,
        1.9861611979404177e-14,
    ],
    [
        1.9817915626943456,
        0.13771379144765566,
        -0.00946773333633548,
        0.0008665253824503706,
        -8.908506099663854e-05,
        9.754427910814714e-06,
        -1.1109174348140725e-06,
        1.2994542427111412e-07,
        -1.5494266202237376e-08,
        1.874166880821938e-09,
        -2.2921377741936822e-10,
        2.827815112628469e-11,
        -3.5130872583628485e-12,
        4.389306252891409e-13,
        -5.5318098437397226e-14,
        6.9711569383322384e-15,
    ],
];
/// The zero `x₀` of `ψ` as a double-double.
const PSI_X0_F32: (f64, f64) = (1.4616321449683622, 9.549995429965697e-17);

/// `ψ(x)` for the `f32` form: on `1 ≤ x < 8` a polynomial of `PSI_F32` (times `x − x₀` on
/// `[1, 2)`, `x − x₀ʰⁱ` exact there), good to `2⁻⁵¹` relative, so that the `f32` rounding is that
/// of the true value but within `2⁻²⁷` ulp of a halfway point; elsewhere `digamma_f64`.
pub(super) fn digamma_f32(x: f64) -> f64 {
    if !(1.0..8.0).contains(&x) {
        return digamma_f64(x);
    }
    let i = (((x - 1.0) * 2.0) as usize).min(13);
    let p = super::super::bessel::kernels::estrin(x - (1.25 + 0.5 * i as f64), &PSI_F32[i]);
    let f = if i < 2 {
        (x - PSI_X0_F32.0) - PSI_X0_F32.1
    } else {
        1.0
    };
    p * f
}

/// `ψ(x)` of an `f64`: `∓∞` at `±0`, NaN at the negative integers and at `−∞`.
pub(super) fn digamma_f64(x: f64) -> f64 {
    if x > 0.0 {
        if x == f64::INFINITY {
            return x;
        }
        if x >= 8.0 {
            // ln x + (its low part − 1/(2x) − u T(u)), rounded once.
            let r = 1.0 / x;
            let u = r * r;
            let l = ln_dd(x);
            return l.hi + (l.lo - fma(u, poly(u, &PSI_TAIL), 0.5 * r));
        }
        return psi_pos_dd(x).value();
    }
    if x == 0.0 {
        return -1.0 / x;
    }
    if x.is_nan() || x == f64::NEG_INFINITY || x == x.floor() {
        return f64::NAN;
    }
    // ψ(−u) = ψ(u) + 1/u + π cot(πu).
    let u = -x;
    if u < PSI_TINY {
        // 1/u − γ, rounded once (π cot(πu) would take its sine into the subnormals).
        if u <= TWO_M1024 {
            return f64::INFINITY;
        }
        return recip(u)
            .add(Dd {
                hi: -EULER_HI,
                lo: -EULER_LO,
            })
            .value();
    }
    let base = if u < 1.0 {
        let d = shifted(u);
        psi1p(u, d)
    } else {
        add_light(psi_pos_dd(u), recip(u))
    };
    add_light(base, PI_DD.mul(cot_pi_dd(u))).value()
}

/// A value `v·e^l` (`l` zero when unscaled).
#[derive(Clone, Copy, Debug)]
struct Scaled {
    v: Dd,
    l: Dd,
}

/// `y^(−s)` of a double-double `y` for an integer `2 ≤ s ≤ 8`, as a double-double.
#[inline]
fn pow_neg(y: Dd, s: u32) -> Dd {
    let q = Dd::new(1.0).div(y);
    let q2 = q.mul(q);
    let q4 = q2.mul(q2);
    match s {
        2 => q2,
        3 => q2.mul(q),
        4 => q4,
        5 => q4.mul(q),
        6 => q4.mul(q2),
        7 => q4.mul(q2).mul(q),
        _ => q4.mul(q4),
    }
}

/// `2⁻⁶⁰`.
const NEGLIGIBLE: f64 = 8.673_617_379_884_035e-19;

/// `ζ(s, a) = Σ_{j ≥ 0} (a + j)^(−s)` for an integer `s ≥ 2` and `a > 0`: unscaled for `s ≤ 8`
/// (`scaled` false; the caller has checked the range), else as `v·e^l` with `l = −s ln a`.
fn hurwitz(s: u32, a: Dd, scaled: bool) -> Scaled {
    let sf = s as f64;
    let start = 0.38 * (sf + 20.0) + 0.5;
    let la = if scaled { ln_of_dd(a) } else { Dd::new(0.0) };
    // (a + j)^(−s), or relative to a^(−s).
    let term = |y: Dd| -> Dd {
        if scaled {
            let e = ln_of_dd(y).sub(la).mul_f64(-sf);
            if e.hi < -700.0 {
                Dd::new(0.0)
            } else {
                exp_dd(e)
            }
        } else {
            pow_neg(y, s)
        }
    };
    let mut sum = Dd::new(0.0);
    let mut j = 0.0;
    loop {
        let y = a.add_f64(j);
        if y.hi >= start {
            sum = sum.add(tail(s, y, term(y)));
            break;
        }
        let t = term(y);
        sum = sum.add(t);
        // The rest is below t·(1 + y/(s − 1)).
        if t.hi * (1.0 + y.hi / (sf - 1.0)) < NEGLIGIBLE * sum.hi {
            break;
        }
        j += 1.0;
    }
    Scaled {
        v: sum,
        l: la.mul_f64(-sf),
    }
}

/// The Euler–Maclaurin tail `Σ_{j ≥ 0} (y + j)^(−s)` given `p = y^(−s)` (possibly relative):
/// `p·(y/(s − 1) + 1/2 + Σ B₂ₖ/(2k)! (s)₂ₖ₋₁ y^(1−2k))`, for `y` past the start of convergence.
fn tail(s: u32, y: Dd, p: Dd) -> Dd {
    let sf = s as f64;
    // The leading terms and the first Bernoulli term in double-double.
    let lead = y.div(Dd::new(sf - 1.0)).add_f64(0.5);
    let t1 = Dd::new(sf).div(y.mul_f64(12.0));
    let mut b = lead.add(t1);
    // The rest in f64, summed without error: t_k = B₂ₖ/(2k)! (s)₂ₖ₋₁ y^(1−2k).
    let yi = 1.0 / y.hi;
    let u = yi * yi;
    let mut t = BERN_FACT[0] * sf * yi;
    let mut hi = 0.0;
    let mut lo = 0.0;
    for k in 2..=BERN_FACT.len() {
        let kf = k as f64;
        t *=
            BERN_FACT[k - 1] / BERN_FACT[k - 2] * (sf + 2.0 * kf - 3.0) * (sf + 2.0 * kf - 2.0) * u;
        let (h, e) = two_sum(hi, t);
        hi = h;
        lo += e;
        if t.abs() < NEGLIGIBLE * 0.25 * b.hi {
            break;
        }
    }
    b = b.add(Dd { hi, lo });
    b.mul(p)
}

/// `n!` as a double-double, `n ≤ 22` exactly.
#[inline]
fn factorial_small(n: u32) -> f64 {
    let mut f = 1.0;
    let mut k = 2.0;
    while k <= n as f64 {
        f *= k;
        k += 1.0;
    }
    f
}

/// `⌊log₂ |a|⌋` of a nonzero finite `a`.
#[inline]
fn ilogb(a: f64) -> i32 {
    let a = a.abs();
    if a < f64::MIN_POSITIVE {
        ((a * 18_446_744_073_709_551_616.0).to_bits() >> 52) as i32 - 1023 - 64
    } else {
        (a.to_bits() >> 52) as i32 - 1023
    }
}

/// `±e^l·v` rounded, `sign` the sign.
#[inline]
fn finish(sign: f64, x: Scaled) -> f64 {
    if x.v.hi == 0.0 {
        return 0.0 * sign;
    }
    let sign = sign * x.v.hi.signum();
    if x.l.hi > 1500.0 {
        return sign * f64::INFINITY;
    }
    if x.l.hi < -1500.0 {
        return sign * 0.0;
    }
    // e^l·v with the exponent of v taken out first.
    let k = ilogb(x.v.hi);
    let s = pow2i(-k);
    let v = Dd {
        hi: x.v.hi.abs() * s,
        lo: x.v.lo * x.v.hi.signum() * s,
    };
    let (d, m) = exp_parts(x.l);
    sign * scale(d.mul(v), m + k)
}

/// `ψ′(x)` for `x > 0`: `1/x + 1/(2x²) + T(u)/x³` from 10 on, sixteen pieces of width 1/2 on
/// `[2, 10)`, and `ψ′(x) = Σ 1/(x + j)² + ψ′(x + m)` below 2, the piece of `x + m` evaluated at
/// `s = x − (c − m)` (so that `x + m` is never rounded).
fn trigamma_pos(x: f64) -> f64 {
    if x >= 10.0 {
        // 1/x + (its rounding error + 1/(2x²) + T(u)/x³): the rest is below a twentieth of 1/x.
        let r = recip(x);
        let u = r.hi * r.hi;
        return r.hi + (r.lo + u * fma(r.hi, poly(u, &TRIG_TAIL), 0.5));
    }
    // The piece i of y ∈ [2, 10) about its centre c, at s + sl = y − c: Y + c₁(s + sl) +
    // (c₀ + s²R(s)), the first two terms without error (in f64 the linear term rounds to nearly
    // half an ulp of the result at the pieces' ends).
    let piece = |i: usize, s: f64, sl: f64| -> Dd {
        let [yc, c0, c1, r @ ..] = TRIG_PIECES[i];
        let (p, e) = two_prod(c1, s);
        let (h, e2) = fast_two_sum(yc, p);
        let (hi, lo) = fast_two_sum(h, e2 + (e + fma(c1, sl, fma(s * s, poly(s, &r), c0))));
        Dd { hi, lo }
    };
    if x >= 2.0 {
        let i = (((x - 2.0) * 2.0) as usize).min(15);
        return piece(i, x - (2.25 + 0.5 * i as f64), 0.0).value();
    }
    // Below 2: the terms 1/(x + j)², each x + j a double-double, then ψ′ at y = x + m ∈ [2, 3)
    // from s + sl = x − (c − m) (c − m exact; s exact from x = 1/8 on, its rounding error below
    // carried through c₁).
    if x < 1e-150 {
        // 1/x², rounded once (ψ′(1 + x) is below its last bit); it overflows from 2⁻⁵¹² down
        // (where the square of 1/x would be infinite and its error term NaN).
        if x <= f64::from_bits(511 << 52) {
            return f64::INFINITY;
        }
        // (At a quarter of 1/x: beside the overflow Dekker's product, without a fused
        // multiply-add, would overflow in its partial products.)
        let q = recip(x);
        let (h, l) = (0.25 * q.hi, 0.25 * q.lo);
        let (p, e) = two_prod(h, h);
        return (p + (e + 2.0 * h * l)) * 16.0;
    }
    let m = if x < 1.0 { 2.0 } else { 1.0 };
    // 1/(x + j)² for j < m, x + j = h + l exactly: q = 1/h with its residual, then q².
    let mut hi = 0.0;
    let mut lo = 0.0;
    let mut j = 0.0;
    while j < m {
        let (h, l) = two_sum(x, j);
        let q = 1.0 / h;
        let ql = q * (rem(q, h, 1.0) - l * q);
        let (p, e) = two_prod(q, q);
        let (s2, e2) = two_sum(hi, p);
        hi = s2;
        lo += e2 + fma(2.0 * q, ql, e);
        j += 1.0;
    }
    let sum = Dd { hi, lo };
    // the piece of y = x + m: 0 below 2.5, 1 above (at the rounding of x + m to 2.5 either
    // serves, s then a quarter by a hair)
    let i = (x + m >= 2.5) as usize;
    let (s, sl) = two_sum(x, m - (2.25 + 0.5 * i as f64));
    add_light(sum, piece(i, s, sl)).value()
}

/// `ψ′(x)` for a finite negative non-integer `x`: the reflection `ψ′(x) = π²/sin²(πx) − ψ′(1 − x)`,
/// whose terms do not cancel (`ψ′(1 − x) ≤ ψ′(1) = π²/6` against `π²/sin² ≥ π²`, so the result is
/// relative to itself, which bounds it relative to `1/r² + 1/p²` as well): `(π/sin πx)²` in
/// double-double (where `|sin πx|` is below `2⁻⁵¹¹` the square overflows, as the value does, and the
/// sine, reduced exactly, is never subnormal short of that), and `ψ′` at `1 − x` rounded, corrected
/// through `ψ″(y) ≈ −1/y² − 1/y³` for the rounding of `1 − x`. Error: the final rounding, at most
/// 0.075 ulp from `ψ′(1 − x)` (0.6 ulp of a term at most a sixth of the result), 0.04 from the sine
/// (`2⁻⁵⁹` relative) and 0.03 from the correction: below 0.65 ulps of the result.
#[inline(never)]
fn trigamma_neg(x: f64) -> f64 {
    let q = PI_DD.div(sin_pi_dd(x));
    let q2 = q.mul(q);
    if !q2.hi.is_finite() {
        // Beyond the range (the value is above (π/sin πx)²).
        return f64::INFINITY;
    }
    let (y, d) = two_sum(1.0, -x);
    let yi = 1.0 / y;
    let t = trigamma_pos_apart(y) - d * yi * yi * (1.0 + yi);
    q2.add_f64(-t).value()
}

/// `trigamma_pos` for the reflection, out of line (so that `polygamma_f64` keeps the one copy of it
/// that it inlines).
#[inline(never)]
fn trigamma_pos_apart(y: f64) -> f64 {
    trigamma_pos(y)
}

/// `ψ⁽ⁿ⁾(x)` of an `f64` for `n ≥ 1`.
pub(super) fn polygamma_f64(n: u32, x: f64) -> f64 {
    if n == 0 {
        return digamma_f64(x);
    }
    let odd = n % 2 == 1;
    // (−1)ⁿ⁺¹
    let sign = if odd { 1.0 } else { -1.0 };
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return 0.0 * sign;
    }
    if x == 0.0 {
        // n!/x^(n+1) at ±0: (−1)ⁿ⁺¹ ∞ at +0, +∞ at −0.
        return if x.is_sign_negative() {
            f64::INFINITY
        } else {
            sign * f64::INFINITY
        };
    }
    if x < 0.0 && (x == f64::NEG_INFINITY || x == x.floor()) {
        return f64::NAN;
    }
    if n == 1 {
        return if x > 0.0 {
            trigamma_pos(x)
        } else {
            trigamma_neg(x)
        };
    }
    let s = n + 1;
    // ln n! (or n! itself) and whether the direct, unscaled sum is safe.
    let small = s <= 8;
    if x > 0.0 {
        let safe = small && (s as i32) * ilogb(x).abs() < 900;
        let z = hurwitz(s, Dd::new(x), !safe);
        return with_factorial(sign, n, z, safe);
    }
    // x < 0: Σ_{j ≥ 0} (x + j)^(−s) = (−1)^s [ζ(s, r) − ζ(s, 1 − x)] + ζ(s, p), with K the number
    // of negative terms, r = −(x + K − 1) and p = x + K (both exact, in (0, 1)).
    let k = (-x).ceil();
    let r = -(x + (k - 1.0));
    let p = x + k;
    let (h, l) = two_sum(1.0, -x);
    let far = Dd { hi: h, lo: l };
    let safe = small && (s as i32) * ilogb(r.min(p)).abs() < 900 && ilogb(h) < 100;
    let zr = hurwitz(s, Dd::new(r), !safe);
    let zf = hurwitz(s, far, !safe);
    let zp = hurwitz(s, Dd::new(p), !safe);
    let neg = if s.is_multiple_of(2) { 1.0 } else { -1.0 };
    // The three in one scale (the largest).
    let top = if zr.l.hi >= zp.l.hi { zr.l } else { zp.l };
    let rel = |z: Scaled| -> Dd {
        if safe {
            return z.v;
        }
        let e = z.l.sub(top);
        if e.hi < -700.0 {
            Dd::new(0.0)
        } else {
            z.v.mul(exp_dd(e))
        }
    };
    let v = rel(zr).sub(rel(zf)).mul_f64(neg).add(rel(zp));
    let z = Scaled {
        v,
        l: if safe { Dd::new(0.0) } else { top },
    };
    with_factorial(sign, n, z, safe)
}

/// `sign·n!·z`, rounded.
#[inline]
fn with_factorial(sign: f64, n: u32, z: Scaled, safe: bool) -> f64 {
    if safe {
        return sign * z.v.mul_f64(factorial_small(n)).value();
    }
    let lf = lgamma_pos_dd(n as f64 + 1.0);
    finish(
        sign,
        Scaled {
            v: z.v,
            l: z.l.add(lf),
        },
    )
}

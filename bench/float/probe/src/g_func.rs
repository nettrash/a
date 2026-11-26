//! The elementary functions, roots and powers of `BigFloat`, with a new result, against MPFR
//! (into an existing variable), malachite-float, astro-float, dashu-float and Arb.
use crate::ffi::*;
use crate::h::*;
use crate::kern;
use crate::libs::*;
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;
use std::os::raw::{c_int, c_long};

use ast::{AF, Consts};
use dsh::{DF, HalfEven};
use dashu_float::{ConstCache, Context, Repr};
use mal::{MF2, MRM};

pub const FP: [u64; 9] = [24, 53, 64, 113, 128, 256, 1000, 10_000, 100_000];

type OurF = fn(&BigFloat, u64, Round) -> (BigFloat, Ordering);
type MpfrF = unsafe extern "C" fn(P, C, Rnd) -> c_int;
type MalF = fn(&MF2, u64, MRM) -> (MF2, Ordering);
type AstF = fn(&AF, usize, ast::ARM, &mut Consts) -> AF;
type DshF = fn(&Context<HalfEven>, &Repr<2>, &mut ConstCache) -> DF;
#[cfg(has_arb)]
type ArbF = unsafe extern "C" fn(arb::A, arb::AC, c_long);

pub fn arb_prec(p: u64) -> c_long {
    (p + 2 * (64 - p.leading_zeros() as u64) + 16) as c_long
}

pub struct Spec {
    pub name: &'static str,
    pub group: &'static str,
    pub ours: OurF,
    pub mpfr: (&'static str, MpfrF),
    pub mal: Option<MalF>,
    pub ast: Option<AstF>,
    pub dsh: Option<DshF>,
    #[cfg(has_arb)]
    pub arb: Option<ArbF>,
    /// Exponent range `kmin..=kmax` of the arguments (`[2^(k-1), 2^k)`), and both signs.
    pub range: (i64, i64, bool),
    pub big: bool,
}

#[cfg(has_arb)]
unsafe extern "C" fn arb_exp2(r: arb::A, x: arb::AC, p: c_long) {
    unsafe {
        let mut b = arb::AB::new();
        arb::arb_set_ui(b.p(), 2);
        arb::arb_pow(r, b.c(), x, p);
    }
}
#[cfg(has_arb)]
unsafe extern "C" fn arb_exp10(r: arb::A, x: arb::AC, p: c_long) {
    unsafe {
        let mut b = arb::AB::new();
        arb::arb_set_ui(b.p(), 10);
        arb::arb_pow(r, b.c(), x, p);
    }
}
#[cfg(has_arb)]
unsafe extern "C" fn arb_log2(r: arb::A, x: arb::AC, p: c_long) {
    unsafe { arb::arb_log_base_ui(r, x, 2, p) }
}
#[cfg(has_arb)]
unsafe extern "C" fn arb_log10(r: arb::A, x: arb::AC, p: c_long) {
    unsafe { arb::arb_log_base_ui(r, x, 10, p) }
}

macro_rules! spec {
    ($g:expr, $name:expr, $ours:expr, $mn:expr, $mf:expr, $mal:expr, $ast:expr, $dsh:expr, $arb:expr, $range:expr, $big:expr) => {
        Spec {
            name: $name,
            group: $g,
            ours: $ours,
            mpfr: ($mn, $mf),
            mal: $mal,
            ast: $ast,
            dsh: $dsh,
            #[cfg(has_arb)]
            arb: $arb,
            range: $range,
            big: $big,
        }
    };
}

pub fn specs() -> Vec<Spec> {
    #[cfg(not(has_arb))]
    macro_rules! arbf {
        ($e:expr) => {
            ()
        };
    }
    #[cfg(has_arb)]
    macro_rules! arbf {
        ($e:expr) => {
            Some($e as ArbF)
        };
    }
    #[allow(unused_macros)]
    macro_rules! noarb {
        () => {
            None
        };
    }
    let all = (-1, 2, true); // |x| in [1/4, 4)
    let pos = (-1, 2, false);
    vec![
        spec!("explog", "exp", BigFloat::exp_round, "MPFR mpfr_exp", mpfr_exp, Some(MF2::exp_prec_round_ref), Some(AF::exp as AstF),
            Some(|c, x, k| c.exp(x, Some(k)).unwrap().value()), arbf!(arb::arb_exp), all, true),
        spec!("explog", "exp2", BigFloat::exp2_round, "MPFR mpfr_exp2", mpfr_exp2, Some(MF2::power_of_2_of_float_prec_round_ref), None, None,
            arbf!(arb_exp2), all, false),
        spec!("explog", "exp10", BigFloat::exp10_round, "MPFR mpfr_exp10", mpfr_exp10, Some(MF2::power_of_10_of_float_prec_round_ref), None, None,
            arbf!(arb_exp10), all, false),
        spec!("explog", "exp_m1", BigFloat::exp_m1_round, "MPFR mpfr_expm1", mpfr_expm1, Some(MF2::exp_x_minus_1_prec_round_ref), None,
            Some(|c, x, k| c.exp_m1(x, Some(k)).unwrap().value()), arbf!(arb::arb_expm1), all, false),
        spec!("explog", "ln", BigFloat::ln_round, "MPFR mpfr_log", mpfr_log, Some(MF2::ln_prec_round_ref), Some(AF::ln as AstF),
            Some(|c, x, k| c.ln(x, Some(k)).unwrap().value()), arbf!(arb::arb_log), pos, true),
        spec!("explog", "log2", BigFloat::log2_round, "MPFR mpfr_log2", mpfr_log2, Some(MF2::log_base_2_prec_round_ref), Some(AF::log2 as AstF),
            Some(|c, x, k| c.log2(x, Some(k)).unwrap().value()), arbf!(arb_log2), pos, false),
        spec!("explog", "log10", BigFloat::log10_round, "MPFR mpfr_log10", mpfr_log10, Some(MF2::log_base_10_prec_round_ref), Some(AF::log10 as AstF),
            Some(|c, x, k| c.log10(x, Some(k)).unwrap().value()), arbf!(arb_log10), pos, false),
        spec!("explog", "ln_1p", BigFloat::ln_1p_round, "MPFR mpfr_log1p", mpfr_log1p, Some(MF2::ln_1_plus_x_prec_round_ref), None,
            Some(|c, x, k| c.ln_1p(x, Some(k)).unwrap().value()), arbf!(arb::arb_log1p), pos, false),
        spec!("hyper", "sinh", BigFloat::sinh_round, "MPFR mpfr_sinh", mpfr_sinh, Some(MF2::sinh_prec_round_ref), Some(AF::sinh as AstF),
            Some(|c, x, k| c.sinh(x, Some(k)).unwrap().value()), arbf!(arb::arb_sinh), all, false),
        spec!("hyper", "cosh", BigFloat::cosh_round, "MPFR mpfr_cosh", mpfr_cosh, Some(MF2::cosh_prec_round_ref), Some(AF::cosh as AstF),
            Some(|c, x, k| c.cosh(x, Some(k)).unwrap().value()), arbf!(arb::arb_cosh), all, false),
        spec!("hyper", "tanh", BigFloat::tanh_round, "MPFR mpfr_tanh", mpfr_tanh, Some(MF2::tanh_prec_round_ref), Some(AF::tanh as AstF),
            Some(|c, x, k| c.tanh(x, Some(k)).unwrap().value()), arbf!(arb::arb_tanh), all, false),
        spec!("hyper", "asinh", BigFloat::asinh_round, "MPFR mpfr_asinh", mpfr_asinh, Some(MF2::asinh_prec_round_ref), Some(AF::asinh as AstF),
            Some(|c, x, k| c.asinh(x, Some(k)).unwrap().value()), arbf!(arb::arb_asinh), all, false),
        spec!("hyper", "acosh", BigFloat::acosh_round, "MPFR mpfr_acosh", mpfr_acosh, Some(MF2::acosh_prec_round_ref), Some(AF::acosh as AstF),
            Some(|c, x, k| c.acosh(x, Some(k)).unwrap().value()), arbf!(arb::arb_acosh), (1, 3, false), false),
        spec!("hyper", "atanh", BigFloat::atanh_round, "MPFR mpfr_atanh", mpfr_atanh, Some(MF2::atanh_prec_round_ref), Some(AF::atanh as AstF),
            Some(|c, x, k| c.atanh(x, Some(k)).unwrap().value()), arbf!(arb::arb_atanh), (-3, 0, true), false),
        spec!("trig", "sin", BigFloat::sin_round, "MPFR mpfr_sin", mpfr_sin, Some(MF2::sin_prec_round_ref), Some(AF::sin as AstF),
            Some(|c, x, k| c.sin(x, Some(k)).unwrap().value()), arbf!(arb::arb_sin), all, true),
        spec!("trig", "cos", BigFloat::cos_round, "MPFR mpfr_cos", mpfr_cos, Some(MF2::cos_prec_round_ref), Some(AF::cos as AstF),
            Some(|c, x, k| c.cos(x, Some(k)).unwrap().value()), arbf!(arb::arb_cos), all, true),
        spec!("trig", "tan", BigFloat::tan_round, "MPFR mpfr_tan", mpfr_tan, Some(MF2::tan_prec_round_ref), Some(AF::tan as AstF),
            Some(|c, x, k| c.tan(x, Some(k)).unwrap().value()), arbf!(arb::arb_tan), all, false),
        spec!("trig", "sec", BigFloat::sec_round, "MPFR mpfr_sec", mpfr_sec, Some(MF2::sec_prec_round_ref), None, None,
            arbf!(arb::arb_sec), all, false),
        spec!("trig", "csc", BigFloat::csc_round, "MPFR mpfr_csc", mpfr_csc, Some(MF2::csc_prec_round_ref), None, None,
            arbf!(arb::arb_csc), all, false),
        spec!("trig", "cot", BigFloat::cot_round, "MPFR mpfr_cot", mpfr_cot, Some(MF2::cot_prec_round_ref), None, None,
            arbf!(arb::arb_cot), all, false),
        spec!("trig", "atan", BigFloat::atan_round, "MPFR mpfr_atan", mpfr_atan, Some(MF2::atan_prec_round_ref), Some(AF::atan as AstF),
            Some(|c, x, k| c.atan(x, Some(k)).unwrap().value()), arbf!(arb::arb_atan), (-2, 2, true), true),
        spec!("trig", "asin", BigFloat::asin_round, "MPFR mpfr_asin", mpfr_asin, Some(MF2::asin_prec_round_ref), Some(AF::asin as AstF),
            Some(|c, x, k| c.asin(x, Some(k)).unwrap().value()), arbf!(arb::arb_asin), (-3, 0, true), false),
        spec!("trig", "acos", BigFloat::acos_round, "MPFR mpfr_acos", mpfr_acos, Some(MF2::acos_prec_round_ref), Some(AF::acos as AstF),
            Some(|c, x, k| c.acos(x, Some(k)).unwrap().value()), arbf!(arb::arb_acos), (-3, 0, true), false),
        // roots
        spec!("roots", "rec_sqrt_round", BigFloat::rec_sqrt_round, "MPFR mpfr_rec_sqrt", mpfr_rec_sqrt, Some(MF2::reciprocal_sqrt_prec_round_ref), None, None,
            arbf!(arb::arb_rsqrt), pos, false),
        spec!("roots", "cbrt_round", BigFloat::cbrt_round, "MPFR mpfr_cbrt", mpfr_cbrt, Some(MF2::cbrt_prec_round_ref),
            Some((|x: &AF, p: usize, rm: ast::ARM, _c: &mut Consts| x.cbrt(p, rm)) as AstF),
            Some(|c, x, _k| c.cbrt(x).unwrap().value()), noarb!(), all, false),
    ]
}

fn spec_sizes(s: &Spec) -> Vec<u64> {
    let mut v = FP.to_vec();
    if s.big {
        v.push(1_000_000);
    }
    v
}

pub fn unary_row(s: &Spec, p: u64, args: &[Raw], label: &str) {
    let op = s.name;
    let ps = if label.is_empty() { format!("prec={p}") } else { format!("prec={p} {label}") };
    if !wanted(s.group, op, &ps) {
        return;
    }
    let l = args.len();
    let q = args.iter().map(|r| r.bits).max().unwrap();
    let mut conts: Vec<Cont> = Vec::new();
    {
        let x: Vec<BigFloat> = args.iter().map(|r| ours(r, q)).collect();
        let f = s.ours;
        conts.push(Cont::ours(kern!(l, |k| f(&x[k], p, N), |v| c_ours(&v.0))));
    }
    {
        let x: Vec<MF> = args.iter().map(|r| mpfr(r, q)).collect();
        let f = s.mpfr.1;
        let mut r = MF::new(p);
        conts.push(Cont::rf(s.mpfr.0, kern!(l, |k| unsafe { f(r.p(), x[k].c(), RN) }, |v| c_mpfr(r.c()))).note("MPFR into an existing variable"));
    }
    if let Some(f) = s.mal {
        let x: Vec<MF2> = args.iter().map(|r| mal::f(r, q)).collect();
        conts.push(Cont::rf("malachite-float", kern!(l, |k| f(&x[k], p, mal::RM), |v| mal::c(&v.0))));
    }
    if let Some(f) = s.dsh {
        let x: Vec<DF> = args.iter().map(|r| dsh::f(r, q)).collect();
        let cx = dsh::ctx(p);
        let mut cache = ConstCache::new();
        conts.push(Cont::rf("dashu-float Context", kern!(l, |k| f(&cx, x[k].repr(), &mut cache), |v| dsh::c(&v))).note("with a ConstCache"));
    }
    if let Some(f) = s.ast {
        let (wp_, wq) = (ast::wp(p), ast::wp(q));
        let tag = if wp_ != p || wq != q { "word-precision" } else { "" };
        if !tag.is_empty() {
            let x: Vec<BigFloat> = args.iter().map(|r| ours(r, wq)).collect();
            let g = s.ours;
            conts.push(Cont::ours_tag(tag, kern!(l, |k| g(&x[k], wp_, N), |v| c_ours(&v.0))));
        }
        let x: Vec<AF> = args.iter().map(|r| ast::f(r, wq)).collect();
        let mut cc = Consts::new().unwrap();
        let mut ct = Cont::rf("astro-float", kern!(l, |k| f(&x[k], wp_ as usize, ast::RM, &mut cc), |v| ast::c(&v))).vs(tag);
        if !tag.is_empty() {
            ct = ct.note(&format!("astro rounds to whole words: {wp_} bits"));
        }
        conts.push(ct);
    }
    #[cfg(has_arb)]
    if let Some(f) = s.arb {
        let x: Vec<MF> = args.iter().map(|r| mpfr(r, q)).collect();
        let xa: Vec<arb::AB> = x.iter().map(arb::AB::from_mpfr).collect();
        let ap = arb_prec(p);
        let mut r = arb::AB::new();
        conts.push(
            Cont::rf("Arb (ball, does less)", kern!(l, |k| unsafe { f(r.p(), xa[k].c(), ap) }, |v| c_mpfr(r.mid(p).c())))
                .loose()
                .note(&format!("does less: a ball at {ap} bits, not correctly rounded; midpoint compared")),
        );
    }
    row(s.group, op, &ps, l.min(6), conts);
}

pub fn args_for(s: &Spec, p: u64, seed: u64) -> Vec<Raw> {
    let l = if p >= 100_000 { 4 } else if p >= 10_000 { 8 } else { 32 };
    let mut rng = Rng::new(seed ^ p ^ (s.name.len() as u64) << 40 ^ s.name.as_bytes()[0] as u64);
    let (a, b, sg) = s.range;
    (0..l).map(|_| Raw::rand_range(&mut rng, p, a, b, sg)).collect()
}

// ------------------------------------------------------------------------------------------
// Pairs and binary functions: sin_cos, sinh_cosh, atan2, pow.

pub fn pair_rows(group: &str) {
    for name in ["sin_cos", "sinh_cosh"] {
        let g = if name == "sin_cos" { "trig" } else { "hyper" };
        if g != group {
            continue;
        }
        for p in FP {
            let ps = format!("prec={p}");
            if !wanted(g, name, &ps) {
                continue;
            }
            let l = if p >= 100_000 { 4 } else if p >= 10_000 { 8 } else { 32 };
            let mut rng = Rng::new(p ^ 0x5c5c ^ name.len() as u64);
            let args: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -1, 2, true)).collect();
            let mut conts: Vec<Cont> = Vec::new();
            let sc = name == "sin_cos";
            {
                let x: Vec<BigFloat> = args.iter().map(|r| ours(r, p)).collect();
                let k = if sc {
                    kern!(l, |k| x[k].sin_cos_round(p, N), |v| Canon::Many(vec![c_ours(&(v.0).0), c_ours(&(v.1).0)]))
                } else {
                    kern!(l, |k| x[k].sinh_cosh_round(p, N), |v| Canon::Many(vec![c_ours(&(v.0).0), c_ours(&(v.1).0)]))
                };
                conts.push(Cont::ours(k));
            }
            {
                let x: Vec<MF> = args.iter().map(|r| mpfr(r, p)).collect();
                let mut s = MF::new(p);
                let mut c = MF::new(p);
                let f = if sc { mpfr_sin_cos } else { mpfr_sinh_cosh };
                let k = kern!(l, |k| unsafe { f(s.p(), c.p(), x[k].c(), RN) }, |v| Canon::Many(vec![c_mpfr(s.c()), c_mpfr(c.c())]));
                conts.push(Cont::rf(if sc { "MPFR mpfr_sin_cos" } else { "MPFR mpfr_sinh_cosh" }, k).note("MPFR into existing variables"));
            }
            {
                let x: Vec<MF2> = args.iter().map(|r| mal::f(r, p)).collect();
                let k = if sc {
                    kern!(l, |k| x[k].sin_cos_prec_round_ref(p, mal::RM), |v| Canon::Many(vec![mal::c(&v.0), mal::c(&v.1)]))
                } else {
                    kern!(l, |k| x[k].sinh_cosh_prec_round_ref(p, mal::RM), |v| Canon::Many(vec![mal::c(&v.0), mal::c(&v.1)]))
                };
                conts.push(Cont::rf("malachite-float", k));
            }
            {
                let x: Vec<DF> = args.iter().map(|r| dsh::f(r, p)).collect();
                let cx = dsh::ctx(p);
                let mut cache = ConstCache::new();
                let k = if sc {
                    kern!(l, |k| { let (a, b) = cx.sin_cos(x[k].repr(), Some(&mut cache)); (a.unwrap().value(), b.unwrap().value()) },
                        |v| Canon::Many(vec![dsh::c(&v.0), dsh::c(&v.1)]))
                } else {
                    kern!(l, |k| { let (a, b) = cx.sinh_cosh(x[k].repr(), Some(&mut cache)); (a.unwrap().value(), b.unwrap().value()) },
                        |v| Canon::Many(vec![dsh::c(&v.0), dsh::c(&v.1)]))
                };
                conts.push(Cont::rf("dashu-float Context", k).note("with a ConstCache"));
            }
            #[cfg(has_arb)]
            {
                let x: Vec<MF> = args.iter().map(|r| mpfr(r, p)).collect();
                let xa: Vec<arb::AB> = x.iter().map(arb::AB::from_mpfr).collect();
                let ap = arb_prec(p);
                let mut s = arb::AB::new();
                let mut c = arb::AB::new();
                let f = if sc { arb::arb_sin_cos } else { arb::arb_sinh_cosh };
                let k = kern!(l, |k| unsafe { f(s.p(), c.p(), xa[k].c(), ap) }, |v| Canon::Many(vec![c_mpfr(s.mid(p).c()), c_mpfr(c.mid(p).c())]));
                conts.push(Cont::rf("Arb (ball, does less)", k).loose().note(&format!("does less: balls at {ap} bits")));
            }
            row(g, name, &ps, l.min(6), conts);
        }
    }
}

pub fn binary_rows(group: &str) {
    for name in ["atan2", "pow"] {
        let g = if name == "atan2" { "trig" } else { "explog" };
        if g != group {
            continue;
        }
        for p in FP {
            let ps = format!("prec={p}");
            if !wanted(g, name, &ps) {
                continue;
            }
            let l = if p >= 100_000 { 4 } else if p >= 10_000 { 8 } else { 32 };
            let mut rng = Rng::new(p ^ 0xa7a7 ^ name.len() as u64);
            let (ya, xa) = if name == "atan2" {
                let y: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -2, 2, true)).collect();
                let x: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -2, 2, true)).collect();
                (y, x)
            } else {
                // x^y: x in [1/4, 4), y of both signs in [1/4, 4)
                let x: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -1, 2, false)).collect();
                let y: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -1, 2, true)).collect();
                (x, y)
            };
            let at = name == "atan2";
            let mut conts: Vec<Cont> = Vec::new();
            {
                let a: Vec<BigFloat> = ya.iter().map(|r| ours(r, p)).collect();
                let b: Vec<BigFloat> = xa.iter().map(|r| ours(r, p)).collect();
                let k = if at {
                    kern!(l, |k| a[k].atan2_round(&b[k], p, N), |v| c_ours(&v.0))
                } else {
                    kern!(l, |k| a[k].pow_round(&b[k], p, N), |v| c_ours(&v.0))
                };
                conts.push(Cont::ours(k));
            }
            {
                let a: Vec<MF> = ya.iter().map(|r| mpfr(r, p)).collect();
                let b: Vec<MF> = xa.iter().map(|r| mpfr(r, p)).collect();
                let mut r = MF::new(p);
                let f = if at { mpfr_atan2 } else { mpfr_pow };
                let k = kern!(l, |k| unsafe { f(r.p(), a[k].c(), b[k].c(), RN) }, |v| c_mpfr(r.c()));
                conts.push(Cont::rf(if at { "MPFR mpfr_atan2" } else { "MPFR mpfr_pow" }, k).note("MPFR into an existing variable"));
            }
            {
                let a: Vec<MF2> = ya.iter().map(|r| mal::f(r, p)).collect();
                let b: Vec<MF2> = xa.iter().map(|r| mal::f(r, p)).collect();
                let k = if at {
                    kern!(l, |k| a[k].atan2_prec_round_ref_ref(&b[k], p, mal::RM), |v| mal::c(&v.0))
                } else {
                    kern!(l, |k| a[k].pow_prec_round_ref_ref(&b[k], p, mal::RM), |v| mal::c(&v.0))
                };
                conts.push(Cont::rf("malachite-float", k));
            }
            {
                let a: Vec<DF> = ya.iter().map(|r| dsh::f(r, p)).collect();
                let b: Vec<DF> = xa.iter().map(|r| dsh::f(r, p)).collect();
                let cx = dsh::ctx(p);
                let mut cache = ConstCache::new();
                let k = if at {
                    kern!(l, |k| cx.atan2(a[k].repr(), b[k].repr(), Some(&mut cache)).unwrap().value(), |v| dsh::c(&v))
                } else {
                    kern!(l, |k| cx.powf(a[k].repr(), b[k].repr(), Some(&mut cache)).unwrap().value(), |v| dsh::c(&v))
                };
                conts.push(Cont::rf("dashu-float Context", k).note("with a ConstCache"));
            }
            if !at {
                let (wp_, tag) = (ast::wp(p), if ast::wp(p) != p { "word-precision" } else { "" });
                if !tag.is_empty() {
                    let a: Vec<BigFloat> = ya.iter().map(|r| ours(r, wp_)).collect();
                    let b: Vec<BigFloat> = xa.iter().map(|r| ours(r, wp_)).collect();
                    conts.push(Cont::ours_tag(tag, kern!(l, |k| a[k].pow_round(&b[k], wp_, N), |v| c_ours(&v.0))));
                }
                let a: Vec<AF> = ya.iter().map(|r| ast::f(r, wp_)).collect();
                let b: Vec<AF> = xa.iter().map(|r| ast::f(r, wp_)).collect();
                let mut cc = Consts::new().unwrap();
                let mut ct = Cont::rf("astro-float", kern!(l, |k| a[k].pow(&b[k], wp_ as usize, ast::RM, &mut cc), |v| ast::c(&v))).vs(tag);
                if !tag.is_empty() {
                    ct = ct.note(&format!("astro rounds to whole words: {wp_} bits"));
                }
                conts.push(ct);
            }
            #[cfg(has_arb)]
            {
                let a: Vec<MF> = ya.iter().map(|r| mpfr(r, p)).collect();
                let b: Vec<MF> = xa.iter().map(|r| mpfr(r, p)).collect();
                let aa: Vec<arb::AB> = a.iter().map(arb::AB::from_mpfr).collect();
                let ba: Vec<arb::AB> = b.iter().map(arb::AB::from_mpfr).collect();
                let ap = arb_prec(p);
                let mut r = arb::AB::new();
                let f = if at { arb::arb_atan2 } else { arb::arb_pow };
                let k = kern!(l, |k| unsafe { f(r.p(), aa[k].c(), ba[k].c(), ap) }, |v| c_mpfr(r.mid(p).c()));
                conts.push(Cont::rf("Arb (ball, does less)", k).loose().note(&format!("does less: a ball at {ap} bits")));
            }
            row(g, name, &ps, l.min(6), conts);
        }
    }
}

// ------------------------------------------------------------------------------------------
// Huge arguments: sin, cos, tan of m·2^k.

pub fn huge_rows() {
    let sp = specs();
    for name in ["sin", "cos", "tan"] {
        let s = sp.iter().find(|s| s.name == name).unwrap();
        for &kk in &[100i64, 10_000, 1_000_000] {
            let p = 53;
            let l = if kk >= 1_000_000 { 4 } else { 16 };
            let mut rng = Rng::new(kk as u64 ^ 0x77 ^ name.len() as u64);
            let args: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, kk, kk, true)).collect();
            let mut s2 = Spec { group: "trig-huge", ..*s };
            s2.ast = None; // astro's exponent is an i32 and its reduction is not timed here at 2^10^6
            unary_row(&s2, p, &args, &format!("arg=m*2^{kk}"));
        }
    }
}

// ------------------------------------------------------------------------------------------
// Roots and powers with integer and rational exponents.

pub fn roots_rows() {
    let sp = specs();
    for s in sp.iter().filter(|s| s.group == "roots") {
        for p in FP {
            let args = args_for(s, p, 31);
            unary_row(s, p, &args, "");
        }
    }
    // nth_root(n), powi(n), pow_number(n), pow_rational(r)
    for p in FP {
        let l = if p >= 100_000 { 4 } else if p >= 10_000 { 8 } else { 32 };
        let mut rng = Rng::new(p ^ 0x900d);
        let args: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -1, 2, false)).collect();
        // x in [1 + 1/16, 1 + 1/8): x^(10^9+7) and x^(2^64+13) stay within the exponent ranges.
        let near1: Vec<Raw> = (0..l)
            .map(|_| {
                let t = Raw::random(&mut rng, p - 5, -3, false);
                let x = BigFloat::from(1).with_prec(p).add_round(&ours(&t, p), p, N);
                assert_eq!(x.1, Ordering::Equal);
                let (m, e) = x.0.mantissa_exponent().unwrap();
                let (neg, limbs) = number_limbs(&m);
                Raw { neg, limbs, bits: p, exp: e }
            })
            .collect();
        for (name, n) in [("nth_root_round(5)", 5u32), ("nth_root_round(100)", 100)] {
            let ps = format!("prec={p}");
            if !wanted("roots", name, &ps) || (n == 100 && p > 10_000) {
                continue;
            }
            let mut conts: Vec<Cont> = Vec::new();
            let x: Vec<BigFloat> = args.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].nth_root_round(n, p, N), |v| c_ours(&v.0))));
            let xm: Vec<MF> = args.iter().map(|r| mpfr(r, p)).collect();
            let mut r = MF::new(p);
            conts.push(Cont::rf("MPFR mpfr_rootn_ui", kern!(l, |k| unsafe { mpfr_rootn_ui(r.p(), xm[k].c(), n as _, RN) }, |v| c_mpfr(r.c()))).note("MPFR into an existing variable"));
            let xl: Vec<MF2> = args.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf("malachite-float", kern!(l, |k| xl[k].root_u_prec_round_ref(n as u64, p, mal::RM), |v| mal::c(&v.0))));
            let xd: Vec<DF> = args.iter().map(|r| dsh::f(r, p)).collect();
            let cx = dsh::ctx(p);
            conts.push(Cont::rf("dashu-float Context", kern!(l, |k| cx.nth_root(n as usize, xd[k].repr()).unwrap().value(), |v| dsh::c(&v))));
            row("roots", name, &ps, l.min(6), conts);
        }
        for (name, n) in [("powi_round(7)", 7i64), ("powi_round(-3)", -3), ("powi_round(1_000_000_007)", 1_000_000_007)] {
            let ps = format!("prec={p}");
            if !wanted("roots", name, &ps) || (n > 100 && p > 1000) {
                continue;
            }
            let mut conts: Vec<Cont> = Vec::new();
            let args = if n > 100 { &near1 } else { &args };
            let x: Vec<BigFloat> = args.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].powi_round(n, p, N), |v| c_ours(&v.0))));
            let xm: Vec<MF> = args.iter().map(|r| mpfr(r, p)).collect();
            let mut r = MF::new(p);
            conts.push(Cont::rf("MPFR mpfr_pow_si", kern!(l, |k| unsafe { mpfr_pow_si(r.p(), xm[k].c(), n as _, RN) }, |v| c_mpfr(r.c()))).note("MPFR into an existing variable"));
            let xl: Vec<MF2> = args.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf("malachite-float", kern!(l, |k| xl[k].pow_s_prec_round_ref(n, p, mal::RM), |v| mal::c(&v.0))));
            let xd: Vec<DF> = args.iter().map(|r| dsh::f(r, p)).collect();
            let cx = dsh::ctx(p);
            conts.push(Cont::rf("dashu-float Context", kern!(l, |k| cx.powi(xd[k].repr(), dashu_int::IBig::from(n)).unwrap().value(), |v| dsh::c(&v))));
            if n > 0 {
                let wp_ = ast::wp(p);
                let tag = if wp_ != p { "word-precision" } else { "" };
                if !tag.is_empty() {
                    let xw: Vec<BigFloat> = args.iter().map(|r| ours(r, wp_)).collect();
                    conts.push(Cont::ours_tag(tag, kern!(l, |k| xw[k].powi_round(n, wp_, N), |v| c_ours(&v.0))));
                }
                let xa: Vec<AF> = args.iter().map(|r| ast::f(r, wp_)).collect();
                let mut ct = Cont::rf("astro-float powi", kern!(l, |k| xa[k].powi(n as usize, wp_ as usize, ast::RM), |v| ast::c(&v))).vs(tag);
                if !tag.is_empty() {
                    ct = ct.note(&format!("astro rounds to whole words: {wp_} bits"));
                }
                conts.push(ct);
            }
            row("roots", name, &ps, l.min(6), conts);
        }
        // x^n for the 65-bit Number n = 2^64 + 13 (pow_number_round) against mpfr_pow_z, with x
        // in [1 + 1/16, 1 + 1/8) so that the power stays in the exponent range.
        {
            let name = "pow_number_round(2^64+13)";
            let ps = format!("prec={p}");
            if wanted("roots", name, &ps) && p <= 10_000 {
                let nl = [13u64, 1];
                let nn = number(false, &nl);
                let xr = &near1;
                let mut conts: Vec<Cont> = Vec::new();
                let x: Vec<BigFloat> = xr.iter().map(|r| ours(r, p)).collect();
                conts.push(Cont::ours(kern!(l, |k| x[k].pow_number_round(&nn, p, N), |v| c_ours(&v.0))));
                let xm: Vec<MF> = xr.iter().map(|r| mpfr(r, p)).collect();
                let z = MZ::from_limbs(false, &nl);
                let mut r = MF::new(p);
                conts.push(Cont::rf("MPFR mpfr_pow_z", kern!(l, |k| unsafe { mpfr_pow_z(r.p(), xm[k].c(), &z.0, RN) }, |v| c_mpfr(r.c()))).note("MPFR into an existing variable"));
                // malachite-float is left out: its exponent range is ±(2^30 − 1), and the power overflows.
                row("roots", name, &ps, l.min(6), conts);
            }
        }
        // x^(p/q) for the rational 5/3 (pow_rational_round) against malachite (MPFR has none).
        {
            let name = "pow_rational_round(5/3)";
            let ps = format!("prec={p}");
            if wanted("roots", name, &ps) && p <= 10_000 {
                let r53 = Rational::from(5) / 3u32;
                let mut conts: Vec<Cont> = Vec::new();
                let x: Vec<BigFloat> = args.iter().map(|r| ours(r, p)).collect();
                conts.push(Cont::ours(kern!(l, |k| x[k].pow_rational_round(&r53, p, N), |v| c_ours(&v.0))));
                let xl: Vec<MF2> = args.iter().map(|r| mal::f(r, p)).collect();
                let q53 = malachite_q::Rational::from_signeds(5i64, 3);
                conts.push(Cont::rf("malachite-float pow_rational", kern!(l, |k| xl[k].pow_rational_prec_round_ref_ref(&q53, p, mal::RM), |v| mal::c(&v.0))).note("MPFR has no x^(p/q) rounded once"));
                row("roots", name, &ps, l.min(6), conts);
            }
        }
    }
    let _ = Number::from(0u8);
}

pub fn run(group: &str) {
    match group {
        "explog" | "hyper" | "trig" => {
            for s in specs().iter().filter(|s| s.group == group) {
                for p in spec_sizes(s) {
                    let args = args_for(s, p, 17);
                    unary_row(s, p, &args, "");
                }
            }
            pair_rows(group);
            binary_rows(group);
        }
        "trig-huge" => huge_rows(),
        "roots" => roots_rows(),
        _ => panic!("func: explog|hyper|trig|trig-huge|roots"),
    }
}

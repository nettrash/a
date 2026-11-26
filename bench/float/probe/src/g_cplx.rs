//! `BigComplex`: arithmetic into an existing number, with a new result, assigning, and the
//! elementary functions, against GNU MPC (and rug's `Complex`, MPC with allocation).
use crate::ffi::*;
use crate::h::*;
use crate::libs::*;
use a::math::float::{BigComplex, BigFloat, Round};
use core::cmp::Ordering;
use rug::Assign;
use rug::ops::AssignRound;
use std::os::raw::c_int;

pub const XP: [u64; 9] = [24, 53, 64, 113, 128, 256, 1000, 10_000, 100_000];
const RNN: c_int = 0;

fn zs(p: u64, l: usize, seed: u64, kmin: i64, kmax: i64) -> Vec<(Raw, Raw)> {
    let mut rng = Rng::new(seed ^ p);
    (0..l)
        .map(|_| (Raw::rand_range(&mut rng, p, kmin, kmax, true), Raw::rand_range(&mut rng, p, kmin, kmax, true)))
        .collect()
}

fn oc(v: &[(Raw, Raw)], p: u64) -> Vec<BigComplex> {
    v.iter().map(|(a, b)| ours_c(a, b, p)).collect()
}
fn mc(v: &[(Raw, Raw)], p: u64) -> Vec<MC> {
    v.iter().map(|(a, b)| mpc(a, b, p)).collect()
}
fn rc(v: &[(Raw, Raw)], p: u64) -> Vec<rug::Complex> {
    v.iter().map(|(a, b)| rg::cx(a, b, p)).collect()
}

const INTO: [&str; 14] = [
    "set_add", "set_sub", "set_mul", "set_sqr", "set_div", "set_sqrt", "set_complex_norm", "set_complex_abs", "set_add_real",
    "set_sub_real", "set_mul_real", "set_div_real", "set_neg", "set_conj",
];

pub fn into_rows() {
    let rn = rug::float::Round::Nearest;
    for name in INTO {
        for p in XP {
            let ps = format!("prec={p}");
            if !wanted("complex-into", name, &ps) {
                continue;
            }
            let l = pool(p, 6);
            let a = zs(p, l, 1, -1, 2);
            let b = zs(p, l, 2, -1, 2);
            let xr: Vec<Raw> = b.iter().map(|(x, _)| x.clone()).collect();
            let mut conts = Vec::new();
            {
                let za = oc(&a, p);
                let zb = oc(&b, p);
                let x: Vec<BigFloat> = xr.iter().map(|r| ours(r, p)).collect();
                let mut r = BigComplex::zero(p);
                let mut f = BigFloat::zero(p);
                let k = match name {
                    "set_add" => kern!(l, |k| r.set_add(&za[k], &zb[k], N), |v| c_ours2(&r)),
                    "set_sub" => kern!(l, |k| r.set_sub(&za[k], &zb[k], N), |v| c_ours2(&r)),
                    "set_mul" => kern!(l, |k| r.set_mul(&za[k], &zb[k], N), |v| c_ours2(&r)),
                    "set_sqr" => kern!(l, |k| r.set_sqr(&za[k], N), |v| c_ours2(&r)),
                    "set_div" => kern!(l, |k| r.set_div(&za[k], &zb[k], N), |v| c_ours2(&r)),
                    "set_sqrt" => kern!(l, |k| r.set_sqrt(&za[k], N), |v| c_ours2(&r)),
                    "set_complex_norm" => kern!(l, |k| f.set_complex_norm(&za[k], N), |v| c_ours(&f)),
                    "set_complex_abs" => kern!(l, |k| f.set_complex_abs(&za[k], N), |v| c_ours(&f)),
                    "set_add_real" => kern!(l, |k| r.set_add_real(&za[k], &x[k], N), |v| c_ours2(&r)),
                    "set_sub_real" => kern!(l, |k| r.set_sub_real(&za[k], &x[k], N), |v| c_ours2(&r)),
                    "set_mul_real" => kern!(l, |k| r.set_mul_real(&za[k], &x[k], N), |v| c_ours2(&r)),
                    "set_div_real" => kern!(l, |k| r.set_div_real(&za[k], &x[k], N), |v| c_ours2(&r)),
                    "set_neg" => kern!(l, |k| r.set_neg(&za[k], N), |v| c_ours2(&r)),
                    _ => kern!(l, |k| r.set_conj(&za[k], N), |v| c_ours2(&r)),
                };
                conts.push(Cont::ours(k));
            }
            {
                let za = mc(&a, p);
                let zb = mc(&b, p);
                let x: Vec<MF> = xr.iter().map(|r| mpfr(r, p)).collect();
                let mut r = MC::new(p);
                let mut f = MF::new(p);
                let (nm, k) = unsafe {
                    match name {
                        "set_add" => ("MPC mpc_add", kern!(l, |k| mpc_add(r.p(), za[k].c(), zb[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_sub" => ("MPC mpc_sub", kern!(l, |k| mpc_sub(r.p(), za[k].c(), zb[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_mul" => ("MPC mpc_mul", kern!(l, |k| mpc_mul(r.p(), za[k].c(), zb[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_sqr" => ("MPC mpc_sqr", kern!(l, |k| mpc_sqr(r.p(), za[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_div" => ("MPC mpc_div", kern!(l, |k| mpc_div(r.p(), za[k].c(), zb[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_sqrt" => ("MPC mpc_sqrt", kern!(l, |k| mpc_sqrt(r.p(), za[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_complex_norm" => ("MPC mpc_norm", kern!(l, |k| mpc_norm(f.p(), za[k].c(), RN), |v| c_mpfr(f.c()))),
                        "set_complex_abs" => ("MPC mpc_abs", kern!(l, |k| mpc_abs(f.p(), za[k].c(), RN), |v| c_mpfr(f.c()))),
                        "set_add_real" => ("MPC mpc_add_fr", kern!(l, |k| mpc_add_fr(r.p(), za[k].c(), x[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_sub_real" => ("MPC mpc_sub_fr", kern!(l, |k| mpc_sub_fr(r.p(), za[k].c(), x[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_mul_real" => ("MPC mpc_mul_fr", kern!(l, |k| mpc_mul_fr(r.p(), za[k].c(), x[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_div_real" => ("MPC mpc_div_fr", kern!(l, |k| mpc_div_fr(r.p(), za[k].c(), x[k].c(), RNN), |v| c_mpc(r.c()))),
                        "set_neg" => ("MPC mpc_neg", kern!(l, |k| mpc_neg(r.p(), za[k].c(), RNN), |v| c_mpc(r.c()))),
                        _ => ("MPC mpc_conj", kern!(l, |k| mpc_conj(r.p(), za[k].c(), RNN), |v| c_mpc(r.c()))),
                    }
                };
                conts.push(Cont::rf(nm, k));
            }
            if matches!(name, "set_add" | "set_sub" | "set_mul" | "set_sqr" | "set_div" | "set_sqrt") {
                let za = rc(&a, p);
                let zb = rc(&b, p);
                let mut r = rug::Complex::new(p as u32);
                let rr = (rn, rn);
                let k = match name {
                    "set_add" => kern!(l, |k| r.assign_round(&za[k] + &zb[k], rr), |v| rg::cc(&r)),
                    "set_sub" => kern!(l, |k| r.assign_round(&za[k] - &zb[k], rr), |v| rg::cc(&r)),
                    "set_mul" => kern!(l, |k| r.assign_round(&za[k] * &zb[k], rr), |v| rg::cc(&r)),
                    "set_sqr" => kern!(l, |k| r.assign_round(za[k].square_ref(), rr), |v| rg::cc(&r)),
                    "set_div" => kern!(l, |k| r.assign_round(&za[k] / &zb[k], rr), |v| rg::cc(&r)),
                    _ => kern!(l, |k| r.assign_round(za[k].sqrt_ref(), rr), |v| rg::cc(&r)),
                };
                conts.push(Cont::rf("rug Complex assign_round", k));
            }
            row("complex-into", name, &ps, l.min(6), conts);
        }
    }
}

const NEW: [&str; 13] = [
    "add_round", "sub_round", "mul_round", "sqr_round", "div_round", "sqrt_round", "norm_round", "abs_round", "add_real_round",
    "mul_real_round", "div_real_round", "mul_int_round", "mul_i_round",
];

pub fn new_rows() {
    for name in NEW {
        for p in XP {
            let ps = format!("prec={p}");
            if !wanted("complex-new", name, &ps) {
                continue;
            }
            let l = pool(p, 6);
            let a = zs(p, l, 3, -1, 2);
            let b = zs(p, l, 4, -1, 2);
            let xr: Vec<Raw> = b.iter().map(|(x, _)| x.clone()).collect();
            let mut rng = Rng::new(p ^ 0x1d);
            let ns: Vec<i64> = (0..l).map(|_| (rng.next() >> 33) as i64 - (1 << 30)).collect();
            let mut conts = Vec::new();
            {
                let za = oc(&a, p);
                let zb = oc(&b, p);
                let x: Vec<BigFloat> = xr.iter().map(|r| ours(r, p)).collect();
                let ns = ns.clone();
                let k = match name {
                    "add_round" => kern!(l, |k| za[k].add_round(&zb[k], p, N), |v| c_ours2(&v.0)),
                    "sub_round" => kern!(l, |k| za[k].sub_round(&zb[k], p, N), |v| c_ours2(&v.0)),
                    "mul_round" => kern!(l, |k| za[k].mul_round(&zb[k], p, N), |v| c_ours2(&v.0)),
                    "sqr_round" => kern!(l, |k| za[k].sqr_round(p, N), |v| c_ours2(&v.0)),
                    "div_round" => kern!(l, |k| za[k].div_round(&zb[k], p, N), |v| c_ours2(&v.0)),
                    "sqrt_round" => kern!(l, |k| za[k].sqrt_round(p, N), |v| c_ours2(&v.0)),
                    "norm_round" => kern!(l, |k| za[k].norm_round(p, N), |v| c_ours(&v.0)),
                    "abs_round" => kern!(l, |k| za[k].abs_round(p, N), |v| c_ours(&v.0)),
                    "add_real_round" => kern!(l, |k| za[k].add_real_round(&x[k], p, N), |v| c_ours2(&v.0)),
                    "mul_real_round" => kern!(l, |k| za[k].mul_real_round(&x[k], p, N), |v| c_ours2(&v.0)),
                    "div_real_round" => kern!(l, |k| za[k].div_real_round(&x[k], p, N), |v| c_ours2(&v.0)),
                    "mul_int_round" => kern!(l, |k| za[k].mul_int_round(ns[k], p, N), |v| c_ours2(&v.0)),
                    _ => kern!(l, |k| za[k].mul_i_round(false, p, N), |v| c_ours2(&v.0)),
                };
                conts.push(Cont::ours(k));
            }
            {
                let za = mc(&a, p);
                let zb = mc(&b, p);
                let x: Vec<MF> = xr.iter().map(|r| mpfr(r, p)).collect();
                let ns = ns.clone();
                macro_rules! nc {
                    ($e:expr) => {
                        kern!(l, |k| { let mut r = MC::new(p); $e(&mut r, k); r }, |v| c_mpc(v.c()))
                    };
                }
                let k = unsafe {
                    match name {
                        "add_round" => nc!(|r: &mut MC, k: usize| mpc_add(r.p(), za[k].c(), zb[k].c(), RNN)),
                        "sub_round" => nc!(|r: &mut MC, k: usize| mpc_sub(r.p(), za[k].c(), zb[k].c(), RNN)),
                        "mul_round" => nc!(|r: &mut MC, k: usize| mpc_mul(r.p(), za[k].c(), zb[k].c(), RNN)),
                        "sqr_round" => nc!(|r: &mut MC, k: usize| mpc_sqr(r.p(), za[k].c(), RNN)),
                        "div_round" => nc!(|r: &mut MC, k: usize| mpc_div(r.p(), za[k].c(), zb[k].c(), RNN)),
                        "sqrt_round" => nc!(|r: &mut MC, k: usize| mpc_sqrt(r.p(), za[k].c(), RNN)),
                        "norm_round" => kern!(l, |k| { let mut f = MF::new(p); mpc_norm(f.p(), za[k].c(), RN); f }, |v| c_mpfr(v.c())),
                        "abs_round" => kern!(l, |k| { let mut f = MF::new(p); mpc_abs(f.p(), za[k].c(), RN); f }, |v| c_mpfr(v.c())),
                        "add_real_round" => nc!(|r: &mut MC, k: usize| mpc_add_fr(r.p(), za[k].c(), x[k].c(), RNN)),
                        "mul_real_round" => nc!(|r: &mut MC, k: usize| mpc_mul_fr(r.p(), za[k].c(), x[k].c(), RNN)),
                        "div_real_round" => nc!(|r: &mut MC, k: usize| mpc_div_fr(r.p(), za[k].c(), x[k].c(), RNN)),
                        "mul_int_round" => nc!(|r: &mut MC, k: usize| mpc_mul_si(r.p(), za[k].c(), ns[k] as _, RNN)),
                        _ => nc!(|r: &mut MC, k: usize| mpc_mul_i(r.p(), za[k].c(), 1, RNN)),
                    }
                };
                conts.push(Cont::rf("MPC init2+op+clear", k));
            }
            if matches!(name, "add_round" | "sub_round" | "mul_round" | "sqr_round" | "div_round" | "sqrt_round" | "norm_round" | "abs_round") {
                let za = rc(&a, p);
                let zb = rc(&b, p);
                let pp = p as u32;
                let k = match name {
                    "add_round" => kern!(l, |k| rug::Complex::with_val(pp, &za[k] + &zb[k]), |v| rg::cc(&v)),
                    "sub_round" => kern!(l, |k| rug::Complex::with_val(pp, &za[k] - &zb[k]), |v| rg::cc(&v)),
                    "mul_round" => kern!(l, |k| rug::Complex::with_val(pp, &za[k] * &zb[k]), |v| rg::cc(&v)),
                    "sqr_round" => kern!(l, |k| rug::Complex::with_val(pp, za[k].square_ref()), |v| rg::cc(&v)),
                    "div_round" => kern!(l, |k| rug::Complex::with_val(pp, &za[k] / &zb[k]), |v| rg::cc(&v)),
                    "sqrt_round" => kern!(l, |k| rug::Complex::with_val(pp, za[k].sqrt_ref()), |v| rg::cc(&v)),
                    "norm_round" => kern!(l, |k| rug::Float::with_val(pp, za[k].norm_ref()), |v| rg::c(&v)),
                    _ => kern!(l, |k| rug::Float::with_val(pp, za[k].abs_ref()), |v| rg::c(&v)),
                };
                conts.push(Cont::rf("rug Complex with_val", k));
            }
            row("complex-new", name, &ps, l.min(6), conts);
        }
    }
}

pub fn assign_rows() {
    let rn = rug::float::Round::Nearest;
    for name in ["add_assign_round", "sub_assign_round", "mul_assign_round", "div_assign_round"] {
        for p in XP {
            let ps = format!("prec={p}");
            if !wanted("complex-assign", name, &ps) {
                continue;
            }
            let l = pool(p, 6);
            let a = zs(p, l, 5, -1, 2);
            let b = zs(p, l, 6, 0, 1);
            let mut conts = Vec::new();
            {
                let mut za = oc(&a, p);
                let zb = oc(&b, p);
                let k = match name {
                    "add_assign_round" => kern!(l, |k| za[k].add_assign_round(&zb[k], N), |v| c_ours2(&za[k])),
                    "sub_assign_round" => kern!(l, |k| za[k].sub_assign_round(&zb[k], N), |v| c_ours2(&za[k])),
                    "mul_assign_round" => kern!(l, |k| za[k].mul_assign_round(&zb[k], N), |v| c_ours2(&za[k])),
                    _ => kern!(l, |k| za[k].div_assign_round(&zb[k], N), |v| c_ours2(&za[k])),
                };
                conts.push(Cont::ours(k));
            }
            {
                let mut za = mc(&a, p);
                let zb = mc(&b, p);
                let k = unsafe {
                    match name {
                        "add_assign_round" => kern!(l, |k| { let z = za[k].p(); mpc_add(z, z, zb[k].c(), RNN) }, |v| c_mpc(za[k].c())),
                        "sub_assign_round" => kern!(l, |k| { let z = za[k].p(); mpc_sub(z, z, zb[k].c(), RNN) }, |v| c_mpc(za[k].c())),
                        "mul_assign_round" => kern!(l, |k| { let z = za[k].p(); mpc_mul(z, z, zb[k].c(), RNN) }, |v| c_mpc(za[k].c())),
                        _ => kern!(l, |k| { let z = za[k].p(); mpc_div(z, z, zb[k].c(), RNN) }, |v| c_mpc(za[k].c())),
                    }
                };
                conts.push(Cont::rf("MPC mpc_op(z, z, w)", k));
            }
            {
                use rug::ops::{AddAssignRound, DivAssignRound, MulAssignRound, SubAssignRound};
                let mut za = rc(&a, p);
                let zb = rc(&b, p);
                let rr = (rn, rn);
                let k = match name {
                    "add_assign_round" => kern!(l, |k| za[k].add_assign_round(&zb[k], rr), |v| rg::cc(&za[k])),
                    "sub_assign_round" => kern!(l, |k| za[k].sub_assign_round(&zb[k], rr), |v| rg::cc(&za[k])),
                    "mul_assign_round" => kern!(l, |k| za[k].mul_assign_round(&zb[k], rr), |v| rg::cc(&za[k])),
                    _ => kern!(l, |k| za[k].div_assign_round(&zb[k], rr), |v| rg::cc(&za[k])),
                };
                conts.push(Cont::rf("rug Complex op_assign_round", k));
            }
            row("complex-assign", name, &ps, l.min(6), conts);
        }
    }
}

// ------------------------------------------------------------------------------------------
// Functions.

type OurC = fn(&BigComplex, u64, Round) -> (BigComplex, (Ordering, Ordering));
type MpcF = unsafe extern "C" fn(*mut Mpc, *const Mpc, c_int) -> c_int;

pub const FXP: [u64; 6] = [53, 113, 256, 1000, 10_000, 100_000];

pub fn func_rows() {
    let list: [(&str, OurC, &str, MpcF); 17] = [
        ("exp", BigComplex::exp_round, "MPC mpc_exp", mpc_exp),
        ("ln", BigComplex::ln_round, "MPC mpc_log", mpc_log),
        ("log10", BigComplex::log10_round, "MPC mpc_log10", mpc_log10),
        ("proj", BigComplex::proj_round, "MPC mpc_proj", mpc_proj),
        ("sin", BigComplex::sin_round, "MPC mpc_sin", mpc_sin),
        ("cos", BigComplex::cos_round, "MPC mpc_cos", mpc_cos),
        ("tan", BigComplex::tan_round, "MPC mpc_tan", mpc_tan),
        ("sinh", BigComplex::sinh_round, "MPC mpc_sinh", mpc_sinh),
        ("cosh", BigComplex::cosh_round, "MPC mpc_cosh", mpc_cosh),
        ("tanh", BigComplex::tanh_round, "MPC mpc_tanh", mpc_tanh),
        ("asin", BigComplex::asin_round, "MPC mpc_asin", mpc_asin),
        ("acos", BigComplex::acos_round, "MPC mpc_acos", mpc_acos),
        ("atan", BigComplex::atan_round, "MPC mpc_atan", mpc_atan),
        ("asinh", BigComplex::asinh_round, "MPC mpc_asinh", mpc_asinh),
        ("acosh", BigComplex::acosh_round, "MPC mpc_acosh", mpc_acosh),
        ("atanh", BigComplex::atanh_round, "MPC mpc_atanh", mpc_atanh),
        ("sqrt", BigComplex::sqrt_round, "MPC mpc_sqrt", mpc_sqrt),
    ];
    for (name, of, mn, mf) in list {
        if name == "sqrt" {
            continue; // in complex-new
        }
        for p in FXP {
            let ps = format!("prec={p}");
            if !wanted("complex-fns", name, &ps) {
                continue;
            }
            let l = if p >= 100_000 { 4 } else if p >= 10_000 { 8 } else { 16 };
            let a = zs(p, l, 7 ^ name.len() as u64, -2, 1);
            let mut conts = Vec::new();
            let za = oc(&a, p);
            conts.push(Cont::ours(kern!(l, |k| of(&za[k], p, N), |v| c_ours2(&v.0))));
            let zm = mc(&a, p);
            let mut r = MC::new(p);
            conts.push(Cont::rf(mn, kern!(l, |k| unsafe { mf(r.p(), zm[k].c(), RNN) }, |v| c_mpc(r.c()))).note("MPC into an existing variable"));
            row("complex-fns", name, &ps, l.min(4), conts);
        }
    }
    for p in FXP {
        let l = if p >= 100_000 { 4 } else if p >= 10_000 { 8 } else { 16 };
        let a = zs(p, l, 99, -2, 1);
        let mut rng = Rng::new(p ^ 0x4u64);
        let w: Vec<(Raw, Raw)> = (0..l).map(|_| (Raw::rand_range(&mut rng, p, 3, 3, false), Raw::rand_range(&mut rng, p, 0, 0, false))).collect();
        for name in ["arg", "sin_cos", "pow", "pow_real", "powi(17)"] {
            let ps = format!("prec={p}");
            if !wanted("complex-fns", name, &ps) {
                continue;
            }
            let mut conts = Vec::new();
            let za = oc(&a, p);
            let wa = oc(&w, p);
            let xa: Vec<BigFloat> = w.iter().map(|(x, _)| ours(x, p)).collect();
            let k = match name {
                "arg" => kern!(l, |k| za[k].arg_round(p, N), |v| c_ours(&v.0)),
                "sin_cos" => kern!(l, |k| za[k].sin_cos_round(p, N), |v| Canon::Many(vec![c_ours2(&(v.0).0), c_ours2(&(v.1).0)])),
                "pow" => kern!(l, |k| za[k].pow_round(&wa[k], p, N), |v| c_ours2(&v.0)),
                "pow_real" => kern!(l, |k| za[k].pow_real_round(&xa[k], p, N), |v| c_ours2(&v.0)),
                _ => kern!(l, |k| za[k].powi_round(17, p, N), |v| c_ours2(&v.0)),
            };
            conts.push(Cont::ours(k));
            let zm = mc(&a, p);
            let wm = mc(&w, p);
            let xm: Vec<MF> = w.iter().map(|(x, _)| mpfr(x, p)).collect();
            let mut r = MC::new(p);
            let mut r2 = MC::new(p);
            let mut f = MF::new(p);
            let (nm, k) = unsafe {
                match name {
                    "arg" => ("MPC mpc_arg", kern!(l, |k| mpc_arg(f.p(), zm[k].c(), RN), |v| c_mpfr(f.c()))),
                    "sin_cos" => (
                        "MPC mpc_sin_cos",
                        kern!(l, |k| mpc_sin_cos(r.p(), r2.p(), zm[k].c(), RNN, RNN), |v| Canon::Many(vec![c_mpc(r.c()), c_mpc(r2.c())])),
                    ),
                    "pow" => ("MPC mpc_pow", kern!(l, |k| mpc_pow(r.p(), zm[k].c(), wm[k].c(), RNN), |v| c_mpc(r.c()))),
                    "pow_real" => ("MPC mpc_pow_fr", kern!(l, |k| mpc_pow_fr(r.p(), zm[k].c(), xm[k].c(), RNN), |v| c_mpc(r.c()))),
                    _ => ("MPC mpc_pow_si", kern!(l, |k| mpc_pow_si(r.p(), zm[k].c(), 17, RNN), |v| c_mpc(r.c()))),
                }
            };
            conts.push(Cont::rf(nm, k).note("MPC into an existing variable"));
            row("complex-fns", name, &ps, l.min(4), conts);
        }
    }
    let _ = rug::Complex::new(2).assign(0);
}

pub fn run(g: &str) {
    match g {
        "into" => into_rows(),
        "new" => new_rows(),
        "assign" => assign_rows(),
        "fns" => func_rows(),
        _ => panic!("complex: into|new|assign|fns"),
    }
}

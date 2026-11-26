//! Arithmetic: into an existing number (`set_*`), with a new result (`*_round`), assigning
//! (`*_assign_round`), the operators with primitive operands, and sums / dot products.
use crate::ffi::*;
use crate::h::*;
use crate::kern;
use crate::libs::*;
use a::math::float::BigFloat;
use a::math::numbers::Number;

/// dashu-float's sums keep a guard digit: an inexact sum may have p + 1 bits (its `Repr` docs).
pub const DASHU_GUARD: &str = "does less: dashu keeps a guard bit in an inexact sum (p+1 bits), not rounded to p";

pub const PRECS: [u64; 10] = [24, 53, 64, 113, 128, 256, 1000, 10_000, 100_000, 1_000_000];
pub const MIXED: [(u64, u64); 5] = [(53, 1000), (53, 100_000), (1000, 53), (1000, 100_000), (100_000, 1_000_000)];

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Sqr,
    Div,
    Fma,
    Sqrt,
}

impl Op {
    fn arity(self) -> usize {
        match self {
            Op::Sqr | Op::Sqrt => 1,
            Op::Fma => 3,
            _ => 2,
        }
    }
}

pub fn params(p: u64, q: u64) -> String {
    if p == q { format!("prec={p}") } else { format!("prec={p} operands={q}") }
}

/// Operands for `op` of `q` bits: sums of both signs and exponents −1..2, products and
/// quotients of both signs, square roots of positive numbers.
pub fn operands(op: Op, q: u64, l: usize, seed: u64) -> Vec<Vec<Raw>> {
    let mut rng = Rng::new(seed ^ q.wrapping_mul(0x1234_5678_9abc_def1) ^ (op as u64) << 56);
    (0..op.arity())
        .map(|_| (0..l).map(|_| Raw::rand_range(&mut rng, q, -1, 2, op != Op::Sqrt)).collect())
        .collect()
}

fn pools<T>(raws: &[Vec<Raw>], f: impl Fn(&Raw) -> T) -> Vec<Vec<T>> {
    raws.iter().map(|v| v.iter().map(&f).collect()).collect()
}

fn take3<T>(mut v: Vec<Vec<T>>) -> (Vec<T>, Vec<T>, Vec<T>) {
    while v.len() < 3 {
        v.push(Vec::new());
    }
    let c = v.pop().unwrap();
    let b = v.pop().unwrap();
    let a = v.pop().unwrap();
    (a, b, c)
}

fn op_name_into(op: Op) -> &'static str {
    match op {
        Op::Add => "set_add",
        Op::Sub => "set_sub",
        Op::Mul => "set_mul",
        Op::Sqr => "set_mul(&x, &x)",
        Op::Div => "set_div",
        Op::Fma => "set_mul_add",
        Op::Sqrt => "set_sqrt",
    }
}

fn op_name_new(op: Op) -> &'static str {
    match op {
        Op::Add => "add_round",
        Op::Sub => "sub_round",
        Op::Mul => "mul_round",
        Op::Sqr => "mul_round(&x, &x)",
        Op::Div => "div_round",
        Op::Fma => "mul_add_round",
        Op::Sqrt => "sqrt_round",
    }
}

const OPS: [Op; 7] = [Op::Add, Op::Sub, Op::Mul, Op::Sqr, Op::Div, Op::Fma, Op::Sqrt];

fn sizes(op: Op) -> Vec<(u64, u64)> {
    let mut v: Vec<(u64, u64)> = PRECS.iter().map(|&p| (p, p)).collect();
    for &(p, q) in &MIXED {
        if matches!(op, Op::Sqr | Op::Sqrt) && !matches!((p, q), (53, 1000) | (1000, 100_000)) {
            continue;
        }
        v.push((p, q));
    }
    v
}

// ------------------------------------------------------------------------------------------
// Into an existing number.

pub fn into_row(op: Op, p: u64, q: u64) {
    let name = op_name_into(op);
    let ps = params(p, q);
    if !wanted("arith-into", name, &ps) {
        return;
    }
    let l = pool(q, op.arity() + 1);
    let raws = operands(op, q, l, 1);
    let mut conts: Vec<Cont> = Vec::new();
    {
        let (a, b, c) = take3(pools(&raws, |r| ours(r, q)));
        let mut r = BigFloat::zero(p);
        let k = match op {
            Op::Add => kern!(l, |k| r.set_add(&a[k], &b[k], N), |v| c_ours(&r)),
            Op::Sub => kern!(l, |k| r.set_sub(&a[k], &b[k], N), |v| c_ours(&r)),
            Op::Mul => kern!(l, |k| r.set_mul(&a[k], &b[k], N), |v| c_ours(&r)),
            Op::Sqr => kern!(l, |k| r.set_mul(&a[k], &a[k], N), |v| c_ours(&r)),
            Op::Div => kern!(l, |k| r.set_div(&a[k], &b[k], N), |v| c_ours(&r)),
            Op::Fma => kern!(l, |k| r.set_mul_add(&a[k], &b[k], &c[k], N), |v| c_ours(&r)),
            Op::Sqrt => kern!(l, |k| r.set_sqrt(&a[k], N), |v| c_ours(&r)),
        };
        conts.push(Cont::ours(k));
    }
    {
        let (a, b, c) = take3(pools(&raws, |r| mpfr(r, q)));
        let mut r = MF::new(p);
        let k = unsafe {
            match op {
                Op::Add => kern!(l, |k| mpfr_add(r.p(), a[k].c(), b[k].c(), RN), |v| c_mpfr(r.c())),
                Op::Sub => kern!(l, |k| mpfr_sub(r.p(), a[k].c(), b[k].c(), RN), |v| c_mpfr(r.c())),
                Op::Mul => kern!(l, |k| mpfr_mul(r.p(), a[k].c(), b[k].c(), RN), |v| c_mpfr(r.c())),
                Op::Sqr => kern!(l, |k| mpfr_sqr(r.p(), a[k].c(), RN), |v| c_mpfr(r.c())),
                Op::Div => kern!(l, |k| mpfr_div(r.p(), a[k].c(), b[k].c(), RN), |v| c_mpfr(r.c())),
                Op::Fma => kern!(l, |k| mpfr_fma(r.p(), a[k].c(), b[k].c(), c[k].c(), RN), |v| c_mpfr(r.c())),
                Op::Sqrt => kern!(l, |k| mpfr_sqrt(r.p(), a[k].c(), RN), |v| c_mpfr(r.c())),
            }
        };
        let nm = match op {
            Op::Add => "MPFR mpfr_add",
            Op::Sub => "MPFR mpfr_sub",
            Op::Mul => "MPFR mpfr_mul",
            Op::Sqr => "MPFR mpfr_sqr",
            Op::Div => "MPFR mpfr_div",
            Op::Fma => "MPFR mpfr_fma",
            Op::Sqrt => "MPFR mpfr_sqrt",
        };
        conts.push(Cont::rf(nm, k));
    }
    {
        use rg::*;
        use rug::ops::AssignRound;
        let (a, b, c) = take3(pools(&raws, |r| rg::f(r, q)));
        let mut r = Float::new(p as u32);
        let rn = rug::float::Round::Nearest;
        let k = match op {
            Op::Add => kern!(l, |k| r.assign_round(&a[k] + &b[k], rn), |v| rg::c(&r)),
            Op::Sub => kern!(l, |k| r.assign_round(&a[k] - &b[k], rn), |v| rg::c(&r)),
            Op::Mul => kern!(l, |k| r.assign_round(&a[k] * &b[k], rn), |v| rg::c(&r)),
            Op::Sqr => kern!(l, |k| r.assign_round(a[k].square_ref(), rn), |v| rg::c(&r)),
            Op::Div => kern!(l, |k| r.assign_round(&a[k] / &b[k], rn), |v| rg::c(&r)),
            Op::Fma => kern!(l, |k| r.assign_round(&a[k] * &b[k] + &c[k], rn), |v| rg::c(&r)),
            Op::Sqrt => kern!(l, |k| r.assign_round(a[k].sqrt_ref(), rn), |v| rg::c(&r)),
        };
        conts.push(Cont::rf("rug assign_round", k));
    }
    row("arith-into", name, &ps, l.min(8), conts);
}

// ------------------------------------------------------------------------------------------
// With a new result.

pub fn new_row(op: Op, p: u64, q: u64) {
    let name = op_name_new(op);
    let ps = params(p, q);
    if !wanted("arith-new", name, &ps) {
        return;
    }
    let l = pool(q, op.arity() + 1);
    let raws = operands(op, q, l, 2);
    let mut conts: Vec<Cont> = Vec::new();
    fn ours_k<'a>(op: Op, raws: &[Vec<Raw>], p: u64, q: u64, l: usize) -> Kern<'a> {
        let (a, b, c) = take3(pools(raws, |r| ours(r, q)));
        match op {
            Op::Add => kern!(l, |k| a[k].add_round(&b[k], p, N), |v| c_ours(&v.0)),
            Op::Sub => kern!(l, |k| a[k].sub_round(&b[k], p, N), |v| c_ours(&v.0)),
            Op::Mul => kern!(l, |k| a[k].mul_round(&b[k], p, N), |v| c_ours(&v.0)),
            Op::Sqr => kern!(l, |k| a[k].mul_round(&a[k], p, N), |v| c_ours(&v.0)),
            Op::Div => kern!(l, |k| a[k].div_round(&b[k], p, N), |v| c_ours(&v.0)),
            Op::Fma => kern!(l, |k| a[k].mul_add_round(&b[k], &c[k], p, N), |v| c_ours(&v.0)),
            Op::Sqrt => kern!(l, |k| a[k].sqrt_round(p, N), |v| c_ours(&v.0)),
        }
    }
    conts.push(Cont::ours(ours_k(op, &raws, p, q, l)));
    {
        let (a, b, c) = take3(pools(&raws, |r| mpfr(r, q)));
        let k = unsafe {
            match op {
                Op::Add => kern!(l, |k| { let mut r = MF::new(p); mpfr_add(r.p(), a[k].c(), b[k].c(), RN); r }, |v| c_mpfr(v.c())),
                Op::Sub => kern!(l, |k| { let mut r = MF::new(p); mpfr_sub(r.p(), a[k].c(), b[k].c(), RN); r }, |v| c_mpfr(v.c())),
                Op::Mul => kern!(l, |k| { let mut r = MF::new(p); mpfr_mul(r.p(), a[k].c(), b[k].c(), RN); r }, |v| c_mpfr(v.c())),
                Op::Sqr => kern!(l, |k| { let mut r = MF::new(p); mpfr_sqr(r.p(), a[k].c(), RN); r }, |v| c_mpfr(v.c())),
                Op::Div => kern!(l, |k| { let mut r = MF::new(p); mpfr_div(r.p(), a[k].c(), b[k].c(), RN); r }, |v| c_mpfr(v.c())),
                Op::Fma => kern!(l, |k| { let mut r = MF::new(p); mpfr_fma(r.p(), a[k].c(), b[k].c(), c[k].c(), RN); r }, |v| c_mpfr(v.c())),
                Op::Sqrt => kern!(l, |k| { let mut r = MF::new(p); mpfr_sqrt(r.p(), a[k].c(), RN); r }, |v| c_mpfr(v.c())),
            }
        };
        conts.push(Cont::rf("MPFR init2+op+clear", k));
    }
    {
        use rg::*;
        let (a, b, c) = take3(pools(&raws, |r| rg::f(r, q)));
        let pp = p as u32;
        let k = match op {
            Op::Add => kern!(l, |k| Float::with_val(pp, &a[k] + &b[k]), |v| rg::c(&v)),
            Op::Sub => kern!(l, |k| Float::with_val(pp, &a[k] - &b[k]), |v| rg::c(&v)),
            Op::Mul => kern!(l, |k| Float::with_val(pp, &a[k] * &b[k]), |v| rg::c(&v)),
            Op::Sqr => kern!(l, |k| Float::with_val(pp, a[k].square_ref()), |v| rg::c(&v)),
            Op::Div => kern!(l, |k| Float::with_val(pp, &a[k] / &b[k]), |v| rg::c(&v)),
            Op::Fma => kern!(l, |k| Float::with_val(pp, &a[k] * &b[k] + &c[k]), |v| rg::c(&v)),
            Op::Sqrt => kern!(l, |k| Float::with_val(pp, a[k].sqrt_ref()), |v| rg::c(&v)),
        };
        conts.push(Cont::rf("rug with_val", k));
    }
    {
        use mal::*;
        let (a, b, c) = take3(pools(&raws, |r| mal::f(r, q)));
        let k = match op {
            Op::Add => kern!(l, |k| a[k].add_prec_round_ref_ref(&b[k], p, RM), |v| mal::c(&v.0)),
            Op::Sub => kern!(l, |k| a[k].sub_prec_round_ref_ref(&b[k], p, RM), |v| mal::c(&v.0)),
            Op::Mul => kern!(l, |k| a[k].mul_prec_round_ref_ref(&b[k], p, RM), |v| mal::c(&v.0)),
            Op::Sqr => kern!(l, |k| a[k].square_prec_round_ref(p, RM), |v| mal::c(&v.0)),
            Op::Div => kern!(l, |k| a[k].div_prec_round_ref_ref(&b[k], p, RM), |v| mal::c(&v.0)),
            Op::Fma => kern!(l, |k| c[k].add_mul_prec_round_ref_ref_ref(&a[k], &b[k], p, RM), |v| mal::c(&v.0)),
            Op::Sqrt => kern!(l, |k| a[k].sqrt_prec_round_ref(p, RM), |v| mal::c(&v.0)),
        };
        conts.push(Cont::rf("malachite-float", k));
    }
    {
        use dashu_int::Sign;
        use dsh::*;
        let (a, b, c) = take3(pools(&raws, |r| dsh::f(r, q)));
        let cx = ctx(p);
        let k = match op {
            Op::Add => kern!(l, |k| cx.add(a[k].repr(), b[k].repr()).unwrap().value(), |v| dsh::c(&v)),
            Op::Sub => kern!(l, |k| cx.sub(a[k].repr(), b[k].repr()).unwrap().value(), |v| dsh::c(&v)),
            Op::Mul => kern!(l, |k| cx.mul(a[k].repr(), b[k].repr()).unwrap().value(), |v| dsh::c(&v)),
            Op::Sqr => kern!(l, |k| cx.sqr(a[k].repr()).unwrap().value(), |v| dsh::c(&v)),
            Op::Div => kern!(l, |k| cx.div(a[k].repr(), b[k].repr()).unwrap().value(), |v| dsh::c(&v)),
            Op::Fma => kern!(
                l,
                |k| cx.fma(a[k].repr(), b[k].repr(), c[k].repr(), Sign::Positive).unwrap().value(),
                |v| dsh::c(&v)
            ),
            Op::Sqrt => kern!(l, |k| cx.sqrt(a[k].repr()).unwrap().value(), |v| dsh::c(&v)),
        };
        let mut ct = Cont::rf("dashu-float Context", k);
        if matches!(op, Op::Add | Op::Sub | Op::Fma) {
            ct = ct.loose().note(DASHU_GUARD);
        }
        conts.push(ct);
    }
    {
        use ast::*;
        let (wp_, wq) = (wp(p), wp(q));
        let tag = if wp_ != p || wq != q { "word-precision" } else { "" };
        if !tag.is_empty() {
            conts.push(Cont::ours_tag(tag, ours_k(op, &raws, wp_, wq, l)));
        }
        let (a, b, c) = take3(pools(&raws, |r| ast::f(r, wq)));
        let k = match op {
            Op::Add => kern!(l, |k| a[k].add(&b[k], wp_ as usize, RM), |v| ast::c(&v)),
            Op::Sub => kern!(l, |k| a[k].sub(&b[k], wp_ as usize, RM), |v| ast::c(&v)),
            Op::Mul => kern!(l, |k| a[k].mul(&b[k], wp_ as usize, RM), |v| ast::c(&v)),
            Op::Sqr => kern!(l, |k| a[k].mul(&a[k], wp_ as usize, RM), |v| ast::c(&v)),
            Op::Div => kern!(l, |k| a[k].div(&b[k], wp_ as usize, RM), |v| ast::c(&v)),
            Op::Fma => {
                let _ = &c;
                kern!(l, |k| Canon::Skip, |v| v)
            }
            Op::Sqrt => kern!(l, |k| a[k].sqrt(wp_ as usize, RM), |v| ast::c(&v)),
        };
        if op != Op::Fma {
            let mut ct = Cont::rf("astro-float", k).vs(tag);
            if !tag.is_empty() {
                ct = ct.note(&format!("astro rounds to whole words: {wp_} bits from {wq}"));
            }
            conts.push(ct);
        }
    }
    row("arith-new", name, &ps, l.min(8), conts);
}

// ------------------------------------------------------------------------------------------
// Assigning forms: `x op= y` at the precision of `x`, a pool of targets that drift by a random
// walk (both signs, factors in [1/2, 2)), identical in every library.

pub fn assign_row(op: Op, p: u64) {
    let name = match op {
        Op::Add => "add_assign_round",
        Op::Sub => "sub_assign_round",
        Op::Mul => "mul_assign_round",
        Op::Div => "div_assign_round",
        _ => unreachable!(),
    };
    let ps = params(p, p);
    if !wanted("arith-assign", name, &ps) {
        return;
    }
    let l = pool(p, 3);
    let mut rng = Rng::new(77 ^ p ^ (op as u64) << 50);
    let signed = matches!(op, Op::Add | Op::Sub);
    let xr: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -1, 2, true)).collect();
    let yr: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, 0, 1, signed)).collect();
    let mut conts: Vec<Cont> = Vec::new();
    {
        let mut x: Vec<BigFloat> = xr.iter().map(|r| ours(r, p)).collect();
        let y: Vec<BigFloat> = yr.iter().map(|r| ours(r, p)).collect();
        let k = match op {
            Op::Add => kern!(l, |k| x[k].add_assign_round(&y[k], N), |v| c_ours(&x[k])),
            Op::Sub => kern!(l, |k| x[k].sub_assign_round(&y[k], N), |v| c_ours(&x[k])),
            Op::Mul => kern!(l, |k| x[k].mul_assign_round(&y[k], N), |v| c_ours(&x[k])),
            _ => kern!(l, |k| x[k].div_assign_round(&y[k], N), |v| c_ours(&x[k])),
        };
        conts.push(Cont::ours(k));
    }
    {
        let mut x: Vec<MF> = xr.iter().map(|r| mpfr(r, p)).collect();
        let y: Vec<MF> = yr.iter().map(|r| mpfr(r, p)).collect();
        let k = unsafe {
            match op {
                Op::Add => kern!(l, |k| { let xp = x[k].p(); mpfr_add(xp, xp, y[k].c(), RN) }, |v| c_mpfr(x[k].c())),
                Op::Sub => kern!(l, |k| { let xp = x[k].p(); mpfr_sub(xp, xp, y[k].c(), RN) }, |v| c_mpfr(x[k].c())),
                Op::Mul => kern!(l, |k| { let xp = x[k].p(); mpfr_mul(xp, xp, y[k].c(), RN) }, |v| c_mpfr(x[k].c())),
                _ => kern!(l, |k| { let xp = x[k].p(); mpfr_div(xp, xp, y[k].c(), RN) }, |v| c_mpfr(x[k].c())),
            }
        };
        conts.push(Cont::rf("MPFR mpfr_op(x, x, y)", k));
    }
    {
        use rug::ops::{AddAssignRound, DivAssignRound, MulAssignRound, SubAssignRound};
        let mut x: Vec<rg::Float> = xr.iter().map(|r| rg::f(r, p)).collect();
        let y: Vec<rg::Float> = yr.iter().map(|r| rg::f(r, p)).collect();
        let rn = rug::float::Round::Nearest;
        let k = match op {
            Op::Add => kern!(l, |k| x[k].add_assign_round(&y[k], rn), |v| rg::c(&x[k])),
            Op::Sub => kern!(l, |k| x[k].sub_assign_round(&y[k], rn), |v| rg::c(&x[k])),
            Op::Mul => kern!(l, |k| x[k].mul_assign_round(&y[k], rn), |v| rg::c(&x[k])),
            _ => kern!(l, |k| x[k].div_assign_round(&y[k], rn), |v| rg::c(&x[k])),
        };
        conts.push(Cont::rf("rug op_assign_round", k));
    }
    {
        use mal::*;
        let mut x: Vec<MF2> = xr.iter().map(|r| mal::f(r, p)).collect();
        let y: Vec<MF2> = yr.iter().map(|r| mal::f(r, p)).collect();
        let k = match op {
            Op::Add => kern!(l, |k| x[k].add_prec_round_assign_ref(&y[k], p, RM), |v| mal::c(&x[k])),
            Op::Sub => kern!(l, |k| x[k].sub_prec_round_assign_ref(&y[k], p, RM), |v| mal::c(&x[k])),
            Op::Mul => kern!(l, |k| x[k].mul_prec_round_assign_ref(&y[k], p, RM), |v| mal::c(&x[k])),
            _ => kern!(l, |k| x[k].div_prec_round_assign_ref(&y[k], p, RM), |v| mal::c(&x[k])),
        };
        conts.push(Cont::rf("malachite-float _prec_round_assign_ref", k));
    }
    {
        let mut x: Vec<dsh::DF> = xr.iter().map(|r| dsh::f(r, p)).collect();
        let y: Vec<dsh::DF> = yr.iter().map(|r| dsh::f(r, p)).collect();
        let k = match op {
            Op::Add => kern!(l, |k| x[k] += &y[k], |v| dsh::c(&x[k])),
            Op::Sub => kern!(l, |k| x[k] -= &y[k], |v| dsh::c(&x[k])),
            Op::Mul => kern!(l, |k| x[k] *= &y[k], |v| dsh::c(&x[k])),
            _ => kern!(l, |k| x[k] /= &y[k], |v| dsh::c(&x[k])),
        };
        let mut ct = Cont::rf("dashu-float op=", k);
        if matches!(op, Op::Add | Op::Sub) {
            ct = ct.loose().note(DASHU_GUARD);
        }
        conts.push(ct);
    }
    row("arith-assign", name, &ps, l.min(8), conts);
}

// ------------------------------------------------------------------------------------------
// Operators with a new result: `&a op &b` and primitive / `Number` operands.

pub fn ops_rows() {
    for &p in &[53u64, 113, 1000, 10_000] {
        let ps = params(p, p);
        let l = pool(p, 3);
        let raws = operands(Op::Add, p, l, 9);
        let mut rng = Rng::new(p ^ 4242);
        let ints: Vec<i64> = (0..l).map(|_| (rng.next() >> 20) as i64 - (1i64 << 43)).collect();
        let dbls: Vec<f64> = (0..l).map(|_| (rng.next() >> 11) as f64 / (1u64 << 52) as f64 * 3.0 + 0.1).collect();
        let nums: Vec<Number> = (0..l).map(|_| Number::from(rng.next() | 1)).collect();
        for opn in ["&a + &b", "&a * &b", "&a / &b", "&a + i64", "&a * f64", "&a / &Number"] {
            if !wanted("arith-ops", opn, &ps) {
                continue;
            }
            let mut conts: Vec<Cont> = Vec::new();
            {
                let a: Vec<BigFloat> = raws[0].iter().map(|r| ours(r, p)).collect();
                let b: Vec<BigFloat> = raws[1].iter().map(|r| ours(r, p)).collect();
                let (ints, dbls, nums) = (ints.clone(), dbls.clone(), nums.clone());
                let k = match opn {
                    "&a + &b" => kern!(l, |k| &a[k] + &b[k], |v| c_ours(&v)),
                    "&a * &b" => kern!(l, |k| &a[k] * &b[k], |v| c_ours(&v)),
                    "&a / &b" => kern!(l, |k| &a[k] / &b[k], |v| c_ours(&v)),
                    "&a + i64" => kern!(l, |k| &a[k] + ints[k], |v| c_ours(&v)),
                    "&a * f64" => kern!(l, |k| &a[k] * dbls[k], |v| c_ours(&v)),
                    _ => kern!(l, |k| &a[k] / &nums[k], |v| c_ours(&v)),
                };
                conts.push(Cont::ours(k));
            }
            {
                let a: Vec<MF> = raws[0].iter().map(|r| mpfr(r, p)).collect();
                let b: Vec<MF> = raws[1].iter().map(|r| mpfr(r, p)).collect();
                let (ints, dbls) = (ints.clone(), dbls.clone());
                let zs: Vec<MZ> = nums
                    .iter()
                    .map(|n| {
                        let (neg, l) = number_limbs(n);
                        MZ::from_limbs(neg, &l)
                    })
                    .collect();
                let k = unsafe {
                    match opn {
                        "&a + &b" => kern!(l, |k| { let mut r = MF::new(p); mpfr_add(r.p(), a[k].c(), b[k].c(), RN); r }, |v| c_mpfr(v.c())),
                        "&a * &b" => kern!(l, |k| { let mut r = MF::new(p); mpfr_mul(r.p(), a[k].c(), b[k].c(), RN); r }, |v| c_mpfr(v.c())),
                        "&a / &b" => kern!(l, |k| { let mut r = MF::new(p); mpfr_div(r.p(), a[k].c(), b[k].c(), RN); r }, |v| c_mpfr(v.c())),
                        "&a + i64" => kern!(l, |k| { let mut r = MF::new(p); mpfr_add_si(r.p(), a[k].c(), ints[k] as _, RN); r }, |v| c_mpfr(v.c())),
                        "&a * f64" => kern!(l, |k| { let mut r = MF::new(p); mpfr_mul_d(r.p(), a[k].c(), dbls[k], RN); r }, |v| c_mpfr(v.c())),
                        _ => kern!(l, |k| { let mut r = MF::new(p); mpfr_div_z(r.p(), a[k].c(), &zs[k].0, RN); r }, |v| c_mpfr(v.c())),
                    }
                };
                let nm = match opn {
                    "&a + i64" => "MPFR init2+mpfr_add_si+clear",
                    "&a * f64" => "MPFR init2+mpfr_mul_d+clear",
                    "&a / &Number" => "MPFR init2+mpfr_div_z+clear",
                    _ => "MPFR init2+op+clear",
                };
                conts.push(Cont::rf(nm, k));
            }
            {
                use rg::*;
                let a: Vec<Float> = raws[0].iter().map(|r| rg::f(r, p)).collect();
                let b: Vec<Float> = raws[1].iter().map(|r| rg::f(r, p)).collect();
                let (ints, dbls) = (ints.clone(), dbls.clone());
                let zs: Vec<rug::Integer> = nums.iter().map(|n| rug::Integer::from_str_radix(&n.to_str_radix(16), 16).unwrap()).collect();
                let pp = p as u32;
                let k = match opn {
                    "&a + &b" => kern!(l, |k| Float::with_val(pp, &a[k] + &b[k]), |v| rg::c(&v)),
                    "&a * &b" => kern!(l, |k| Float::with_val(pp, &a[k] * &b[k]), |v| rg::c(&v)),
                    "&a / &b" => kern!(l, |k| Float::with_val(pp, &a[k] / &b[k]), |v| rg::c(&v)),
                    "&a + i64" => kern!(l, |k| Float::with_val(pp, &a[k] + ints[k]), |v| rg::c(&v)),
                    "&a * f64" => kern!(l, |k| Float::with_val(pp, &a[k] * dbls[k]), |v| rg::c(&v)),
                    _ => kern!(l, |k| Float::with_val(pp, &a[k] / &zs[k]), |v| rg::c(&v)),
                };
                conts.push(Cont::rf("rug with_val", k));
            }
            {
                use mal::*;
                if matches!(opn, "&a + &b" | "&a * &b" | "&a / &b") {
                    let a: Vec<MF2> = raws[0].iter().map(|r| mal::f(r, p)).collect();
                    let b: Vec<MF2> = raws[1].iter().map(|r| mal::f(r, p)).collect();
                    let k = match opn {
                        "&a + &b" => kern!(l, |k| &a[k] + &b[k], |v| mal::c(&v)),
                        "&a * &b" => kern!(l, |k| &a[k] * &b[k], |v| mal::c(&v)),
                        _ => kern!(l, |k| &a[k] / &b[k], |v| mal::c(&v)),
                    };
                    conts.push(Cont::rf("malachite-float operators", k));
                }
            }
            {
                if matches!(opn, "&a + &b" | "&a * &b" | "&a / &b") {
                    let a: Vec<dsh::DF> = raws[0].iter().map(|r| dsh::f(r, p)).collect();
                    let b: Vec<dsh::DF> = raws[1].iter().map(|r| dsh::f(r, p)).collect();
                    let k = match opn {
                        "&a + &b" => kern!(l, |k| &a[k] + &b[k], |v| dsh::c(&v)),
                        "&a * &b" => kern!(l, |k| &a[k] * &b[k], |v| dsh::c(&v)),
                        _ => kern!(l, |k| &a[k] / &b[k], |v| dsh::c(&v)),
                    };
                    let mut ct = Cont::rf("dashu-float operators", k);
                    if opn == "&a + &b" {
                        ct = ct.loose().note(DASHU_GUARD);
                    }
                    conts.push(ct);
                }
            }
            row("arith-ops", opn, &ps, l.min(8), conts);
        }
    }
}

// ------------------------------------------------------------------------------------------
// Sums and dot products rounded once.

pub fn sumdot_rows() {
    for &p in &[53u64, 256, 1000, 10_000] {
        for &n in &[4usize, 16, 100, 1000] {
            for opn in ["sum_round", "dot_round"] {
                let ps = format!("prec={p} n={n}");
                if !wanted("sum-dot", opn, &ps) {
                    continue;
                }
                let l = 4usize;
                let mut rng = Rng::new(p ^ (n as u64) << 20 ^ opn.len() as u64);
                let mk = |rng: &mut Rng| -> Vec<Vec<Raw>> {
                    (0..l).map(|_| (0..n).map(|_| Raw::rand_range(rng, p, -20, 20, true)).collect()).collect()
                };
                let xr = mk(&mut rng);
                let yr = mk(&mut rng);
                let mut conts: Vec<Cont> = Vec::new();
                {
                    let xs: Vec<Vec<BigFloat>> = xr.iter().map(|v| v.iter().map(|r| ours(r, p)).collect()).collect();
                    let ys: Vec<Vec<BigFloat>> = yr.iter().map(|v| v.iter().map(|r| ours(r, p)).collect()).collect();
                    let k = if opn == "sum_round" {
                        kern!(l, |k| BigFloat::sum_round(&xs[k], p, N), |v| c_ours(&v.0))
                    } else {
                        kern!(l, |k| BigFloat::dot_round(&xs[k], &ys[k], p, N), |v| c_ours(&v.0))
                    };
                    conts.push(Cont::ours(k));
                }
                {
                    let mut xs: Vec<Vec<MF>> = xr.iter().map(|v| v.iter().map(|r| mpfr(r, p)).collect()).collect();
                    let mut ys: Vec<Vec<MF>> = yr.iter().map(|v| v.iter().map(|r| mpfr(r, p)).collect()).collect();
                    let xp: Vec<Vec<P>> = xs.iter_mut().map(|v| v.iter_mut().map(|m| m.p()).collect()).collect();
                    let yp: Vec<Vec<P>> = ys.iter_mut().map(|v| v.iter_mut().map(|m| m.p()).collect()).collect();
                    let mut r = MF::new(p);
                    let k = unsafe {
                        if opn == "sum_round" {
                            kern!(l, |k| { let _ = (&xs, &ys); mpfr_sum(r.p(), xp[k].as_ptr(), n as _, RN) }, |v| c_mpfr(r.c()))
                        } else {
                            kern!(l, |k| { let _ = (&xs, &ys); mpfr_dot(r.p(), xp[k].as_ptr(), yp[k].as_ptr(), n as _, RN) }, |v| c_mpfr(r.c()))
                        }
                    };
                    let nm = if opn == "sum_round" { "MPFR mpfr_sum" } else { "MPFR mpfr_dot" };
                    conts.push(Cont::rf(nm, k).note("MPFR into an existing variable"));
                }
                {
                    use mal::*;
                    let xs: Vec<Vec<MF2>> = xr.iter().map(|v| v.iter().map(|r| mal::f(r, p)).collect()).collect();
                    let ys: Vec<Vec<MF2>> = yr.iter().map(|v| v.iter().map(|r| mal::f(r, p)).collect()).collect();
                    let k = if opn == "sum_round" {
                        kern!(l, |k| MF2::sum_prec_round(&xs[k], p, RM), |v| mal::c(&v.0))
                    } else {
                        kern!(l, |k| MF2::dot_prec_round(&xs[k], &ys[k], p, RM), |v| mal::c(&v.0))
                    };
                    conts.push(Cont::rf("malachite-float", k));
                }
                row("sum-dot", opn, &ps, l, conts);
            }
        }
    }
}

pub fn run(sub: &str) {
    match sub {
        "into" => {
            for op in OPS {
                for (p, q) in sizes(op) {
                    into_row(op, p, q);
                }
            }
        }
        "new" => {
            for op in OPS {
                for (p, q) in sizes(op) {
                    new_row(op, p, q);
                }
            }
        }
        "assign" => {
            for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
                for p in PRECS {
                    assign_row(op, p);
                }
            }
        }
        "ops" => ops_rows(),
        "sumdot" => sumdot_rows(),
        _ => panic!("arith: into|new|assign|ops|sumdot"),
    }
}

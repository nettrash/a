//! a::math::dual's Dual<f64> and HyperDual<f64> against num-dual 0.16 (Dual64, HyperDual64,
//! and its gradient/jacobian/hessian with vector dual numbers), per operation over a pool of
//! 1024 random inputs with nonzero tangent parts, and the derivative functions on standard test
//! functions by the number of variables. Every side's results are compared with ours before
//! timing.

#[path = "../common.rs"]
mod common;

use a::math::dual::{self as ad, Dual, DualNum, HyperDual};
use a::math::matrix::Matrix;
use common::*;
use nalgebra::{Const, DVector, SVector};
use num_dual::{Dual64, HyperDual64};
use num_traits::Signed;

type ND = Dual64;
type NH = HyperDual64;

fn agree(what: &str, ours: &[Vec<f64>], theirs: &[Vec<f64>], tol: f64) -> f64 {
    let mut worst = 0f64;
    for (i, (a, b)) in ours.iter().zip(theirs).enumerate() {
        for (x, y) in a.iter().zip(b) {
            let e = (x - y).abs() / (1e-300 + x.abs().max(y.abs()));
            let e = if (x - y).abs() < 1e-300 || (x.is_nan() && y.is_nan()) { 0.0 } else { e };
            if !(e <= tol) {
                panic!("{what}: num-dual differs from ours at item {i}: {a:?} vs {b:?}");
            }
            worst = worst.max(e);
        }
    }
    worst
}

fn pd(d: &Dual<f64>) -> Vec<f64> {
    vec![d.re, d.eps]
}
fn pnd(d: &ND) -> Vec<f64> {
    vec![d.re, d.eps]
}
fn ph(d: &HyperDual<f64>) -> Vec<f64> {
    vec![d.re, d.eps1, d.eps2, d.eps1eps2]
}
fn pnh(d: &NH) -> Vec<f64> {
    vec![d.re, d.eps1, d.eps2, d.eps1eps2]
}

// ---- standard test functions, written once for each library ------------------------------

/// The extended Rosenbrock function, Σ 100(x_{i+1} − x_i²)² + (1 − x_i)².
fn rosen<D: DualNum<Float = f64>>(x: &[D]) -> D {
    let mut s = D::ZERO;
    for i in 0..x.len() - 1 {
        let a = D::constant(1.0) - x[i];
        let b = x[i + 1] - x[i] * x[i];
        s = s + a * a + b * b * 100.0;
    }
    s
}
fn rosen_nd<D: num_dual::DualNum<Primitive = f64>>(x: &[D]) -> D {
    let mut s = D::zero();
    for i in 0..x.len() - 1 {
        let a = D::one() - x[i].clone();
        let b = x[i + 1].clone() - x[i].clone() * x[i].clone();
        s = s + a.clone() * a + b.clone() * b * 100.0;
    }
    s
}
/// A transcendental test function, Σ sin(x_i)·exp(x_{i+1}/4) + ln(1 + x_i²).
fn trig<D: DualNum<Float = f64>>(x: &[D]) -> D {
    let mut s = D::ZERO;
    for i in 0..x.len() - 1 {
        s = s + x[i].sin() * (x[i + 1] * 0.25).exp() + (x[i] * x[i] + 1.0).ln();
    }
    s
}
fn trig_nd<D: num_dual::DualNum<Primitive = f64>>(x: &[D]) -> D {
    let mut s = D::zero();
    for i in 0..x.len() - 1 {
        s = s + x[i].sin() * (x[i + 1].clone() * 0.25).exp() + (x[i].clone() * x[i].clone() + 1.0).ln();
    }
    s
}
/// Broyden's tridiagonal function, f_i = (3 − 2x_i)x_i − x_{i−1} − 2x_{i+1} + 1.
fn broyden<D: DualNum<Float = f64>>(x: &[D]) -> Vec<D> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut f = (D::constant(3.0) - x[i] * 2.0) * x[i] + 1.0;
            if i > 0 {
                f = f - x[i - 1];
            }
            if i + 1 < n {
                f = f - x[i + 1] * 2.0;
            }
            f
        })
        .collect()
}
fn broyden_nd<D: num_dual::DualNum<Primitive = f64>>(x: &[D]) -> Vec<D> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut f = (D::from(3.0) - x[i].clone() * 2.0) * x[i].clone() + 1.0;
            if i > 0 {
                f = f - x[i - 1].clone();
            }
            if i + 1 < n {
                f = f - x[i + 1].clone() * 2.0;
            }
            f
        })
        .collect()
}
/// A scalar composition, sin(eˣ)/(1 + x²).
fn comp<D: DualNum<Float = f64>>(x: D) -> D {
    (x.exp()).sin() / (x * x + 1.0)
}
fn comp_nd<D: num_dual::DualNum<Primitive = f64>>(x: D) -> D {
    (x.exp()).sin() / (x.clone() * x + 1.0)
}

macro_rules! elementary {
    ($out:expr, $rng:expr, $grp:expr, $name:expr, $lo:expr, $hi:expr,
     ours: |$x:ident, $y:ident, $z:ident| $ours:expr,
     dual: $dual:expr, hyper: $hyper:expr) => {{
        let out: &Out = $out;
        let rng: &mut Rng = $rng;
        let len = 1024usize;
        let raw: Vec<[f64; 12]> = (0..len)
            .map(|_| {
                let mut r = [0f64; 12];
                for k in 0..3 {
                    r[4 * k] = rng.range($lo, $hi);
                    for j in 1..4 {
                        r[4 * k + j] = rng.range(-1.0, 1.0);
                    }
                }
                r
            })
            .collect();
        // Dual
        let od: Vec<(Dual<f64>, Dual<f64>, Dual<f64>)> = raw
            .iter()
            .map(|r| (Dual::new(r[0], r[1]), Dual::new(r[4], r[5]), Dual::new(r[8], r[9])))
            .collect();
        let nd: Vec<(ND, ND, ND)> =
            raw.iter().map(|r| (ND::new(r[0], r[1]), ND::new(r[4], r[5]), ND::new(r[8], r[9]))).collect();
        let fo = |($x, $y, $z): &(Dual<f64>, Dual<f64>, Dual<f64>)| {
            let ($x, $y, $z) = (*$x, *$y, *$z);
            $ours
        };
        let w: Vec<Vec<f64>> = od.iter().map(|t| pd(&fo(t))).collect();
        let mut s = vec![mk("ours", "", &od, fo)];
        let fdual: Option<fn(ND, ND, ND) -> ND> = $dual;
        if let Some(f) = fdual {
            let fr = move |(a, b, c): &(ND, ND, ND)| f(*a, *b, *c);
            agree(&format!("Dual {}", $name), &w, &nd.iter().map(|t| pnd(&fr(t))).collect::<Vec<_>>(), 1e-10);
            s.push(mk("num-dual", "Dual64", &nd, fr));
        }
        out.row($grp, &format!("Dual {}", $name), &format!("f64, x in [{}, {})", $lo, $hi), len, "", s);
        // HyperDual
        let oh: Vec<(HyperDual<f64>, HyperDual<f64>, HyperDual<f64>)> = raw
            .iter()
            .map(|r| {
                (
                    HyperDual::new(r[0], r[1], r[2], r[3]),
                    HyperDual::new(r[4], r[5], r[6], r[7]),
                    HyperDual::new(r[8], r[9], r[10], r[11]),
                )
            })
            .collect();
        let nh: Vec<(NH, NH, NH)> = raw
            .iter()
            .map(|r| {
                (NH::new(r[0], r[1], r[2], r[3]), NH::new(r[4], r[5], r[6], r[7]), NH::new(r[8], r[9], r[10], r[11]))
            })
            .collect();
        let fo = |($x, $y, $z): &(HyperDual<f64>, HyperDual<f64>, HyperDual<f64>)| {
            let ($x, $y, $z) = (*$x, *$y, *$z);
            $ours
        };
        let w: Vec<Vec<f64>> = oh.iter().map(|t| ph(&fo(t))).collect();
        let mut s = vec![mk("ours", "", &oh, fo)];
        let fhyper: Option<fn(NH, NH, NH) -> NH> = $hyper;
        if let Some(f) = fhyper {
            let fr = move |(a, b, c): &(NH, NH, NH)| f(*a, *b, *c);
            agree(&format!("HyperDual {}", $name), &w, &nh.iter().map(|t| pnh(&fr(t))).collect::<Vec<_>>(), 1e-10);
            s.push(mk("num-dual", "HyperDual64", &nh, fr));
        }
        out.row($grp, &format!("HyperDual {}", $name), &format!("f64, x in [{}, {})", $lo, $hi), len, "", s);
    }};
}

/// Both sides of a unary function by name: ours `x.f()`, num-dual's `x.f()` (DualNum, by ref).
macro_rules! unary {
    ($out:expr, $rng:expr, $name:ident, $lo:expr, $hi:expr) => {
        elementary!($out, $rng, "dual", stringify!($name), $lo, $hi,
            ours: |x, _y, _z| x.$name(),
            dual: Some(|a: ND, _b, _c| num_dual::DualNum::$name(&a)),
            hyper: Some(|a: NH, _b, _c| num_dual::DualNum::$name(&a)))
    };
}

fn elementaries(out: &Out) {
    let mut rng = Rng::new(31);
    let r = &mut rng;
    elementary!(out, r, "dual", "add", -2.0, 2.0, ours: |x, y, _z| x + y,
        dual: Some(|a, b, _c| a + b), hyper: Some(|a, b, _c| a + b));
    elementary!(out, r, "dual", "mul", -2.0, 2.0, ours: |x, y, _z| x * y,
        dual: Some(|a, b, _c| a * b), hyper: Some(|a, b, _c| a * b));
    elementary!(out, r, "dual", "div", 0.5, 2.0, ours: |x, y, _z| x / y,
        dual: Some(|a, b, _c| a / b), hyper: Some(|a, b, _c| a / b));
    elementary!(out, r, "dual", "mul_add", -2.0, 2.0, ours: |x, y, z| x.mul_add(y, z),
        dual: Some(|a: ND, b, c| num_dual::DualNum::mul_add(&a, b, c)),
        hyper: Some(|a: NH, b, c| num_dual::DualNum::mul_add(&a, b, c)));
    unary!(out, r, recip, 0.5, 3.0);
    unary!(out, r, sqrt, 0.1, 4.0);
    unary!(out, r, cbrt, 0.1, 4.0);
    elementary!(out, r, "dual", "powi(3)", -2.0, 2.0, ours: |x, _y, _z| x.powi(3),
        dual: Some(|a: ND, _b, _c| num_dual::DualNum::powi(&a, 3)),
        hyper: Some(|a: NH, _b, _c| num_dual::DualNum::powi(&a, 3)));
    elementary!(out, r, "dual", "powi(-5)", 0.5, 2.0, ours: |x, _y, _z| x.powi(-5),
        dual: Some(|a: ND, _b, _c| num_dual::DualNum::powi(&a, -5)),
        hyper: Some(|a: NH, _b, _c| num_dual::DualNum::powi(&a, -5)));
    elementary!(out, r, "dual", "powf(2.5)", 0.1, 3.0, ours: |x, _y, _z| x.powf(2.5),
        dual: Some(|a: ND, _b, _c| num_dual::DualNum::powf(&a, 2.5)),
        hyper: Some(|a: NH, _b, _c| num_dual::DualNum::powf(&a, 2.5)));
    elementary!(out, r, "dual", "pow (variable exponent)", 0.5, 2.0, ours: |x, y, _z| x.pow(y),
        dual: Some(|a: ND, b, _c| num_dual::DualNum::powd(&a, b)),
        hyper: Some(|a: NH, b, _c| num_dual::DualNum::powd(&a, b)));
    unary!(out, r, exp, -3.0, 3.0);
    unary!(out, r, exp2, -3.0, 3.0);
    unary!(out, r, exp_m1, -3.0, 3.0);
    unary!(out, r, ln, 0.1, 5.0);
    unary!(out, r, log2, 0.1, 5.0);
    unary!(out, r, log10, 0.1, 5.0);
    unary!(out, r, ln_1p, -0.5, 3.0);
    elementary!(out, r, "dual", "log(3)", 0.1, 5.0, ours: |x, _y, _z| x.log(3.0),
        dual: Some(|a: ND, _b, _c| num_dual::DualNum::log(&a, 3.0)),
        hyper: Some(|a: NH, _b, _c| num_dual::DualNum::log(&a, 3.0)));
    unary!(out, r, sin, -3.0, 3.0);
    unary!(out, r, cos, -3.0, 3.0);
    unary!(out, r, tan, -1.4, 1.4);
    elementary!(out, r, "dual", "sin_cos", -3.0, 3.0, ours: |x, _y, _z| { let (s, c) = x.sin_cos(); s + c },
        dual: Some(|a: ND, _b, _c| { let (s, c) = num_dual::DualNum::sin_cos(&a); s + c }),
        hyper: Some(|a: NH, _b, _c| { let (s, c) = num_dual::DualNum::sin_cos(&a); s + c }));
    unary!(out, r, asin, -0.9, 0.9);
    unary!(out, r, acos, -0.9, 0.9);
    unary!(out, r, atan, -3.0, 3.0);
    elementary!(out, r, "dual", "atan2", -2.0, 2.0, ours: |x, y, _z| x.atan2(y),
        dual: Some(|a: ND, b, _c| num_dual::DualNum::atan2(&a, b)),
        hyper: Some(|a: NH, b, _c| num_dual::DualNum::atan2(&a, b)));
    unary!(out, r, sinh, -3.0, 3.0);
    unary!(out, r, cosh, -3.0, 3.0);
    unary!(out, r, tanh, -3.0, 3.0);
    unary!(out, r, asinh, -3.0, 3.0);
    unary!(out, r, acosh, 1.1, 5.0);
    unary!(out, r, atanh, -0.9, 0.9);
    elementary!(out, r, "dual", "abs", -2.0, 2.0, ours: |x, _y, _z| x.abs(),
        dual: Some(|a: ND, _b, _c| Signed::abs(&a)), hyper: Some(|a: NH, _b, _c| Signed::abs(&a)));
    elementary!(out, r, "dual", "signum", -2.0, 2.0, ours: |x, _y, _z| x.signum(),
        dual: Some(|a: ND, _b, _c| Signed::signum(&a)), hyper: Some(|a: NH, _b, _c| Signed::signum(&a)));
    elementary!(out, r, "dual", "hypot", -2.0, 2.0, ours: |x, y, _z| x.hypot(y), dual: None, hyper: None);
    elementary!(out, r, "dual", "min", -2.0, 2.0, ours: |x, y, _z| x.min(y), dual: None, hyper: None);
    elementary!(out, r, "dual", "max", -2.0, 2.0, ours: |x, y, _z| x.max(y), dual: None, hyper: None);
}

fn column(v: &[f64]) -> Matrix<f64> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

macro_rules! by_n {
    ($fname:ident, $n:literal) => {
        #[allow(clippy::too_many_lines)]
        fn $fname(out: &Out, seed: u64) {
            const N: usize = $n;
            let mut rng = Rng::new(seed);
            let g = "dual-derivatives";
            let len = (4096 / (N * N)).clamp(8, 256);
            let pts: Vec<[f64; N]> = (0..len)
                .map(|_| {
                    let mut p = [0f64; N];
                    for x in p.iter_mut() {
                        *x = rng.range(-1.5, 1.5);
                    }
                    p
                })
                .collect();
            let om: Vec<(Matrix<f64>, ())> = pts.iter().map(|p| (column(p), ())).collect();
            let sv: Vec<(SVector<f64, N>, ())> = pts.iter().map(|p| (SVector::from_column_slice(p), ())).collect();
            let dv: Vec<(DVector<f64>, ())> = pts.iter().map(|p| (DVector::from_column_slice(p), ())).collect();
            let params = format!("n = {N}");

            for (fname, fo, fs, fd) in [
                (
                    "Rosenbrock",
                    rosen::<Dual<f64>> as fn(&[Dual<f64>]) -> Dual<f64>,
                    rosen_nd::<num_dual::DualSVec64<N>> as fn(&[num_dual::DualSVec64<N>]) -> num_dual::DualSVec64<N>,
                    rosen_nd::<num_dual::DualDVec64> as fn(&[num_dual::DualDVec64]) -> num_dual::DualDVec64,
                ),
                (
                    "sin·exp + ln",
                    trig::<Dual<f64>> as fn(&[Dual<f64>]) -> Dual<f64>,
                    trig_nd::<num_dual::DualSVec64<N>> as fn(&[num_dual::DualSVec64<N>]) -> num_dual::DualSVec64<N>,
                    trig_nd::<num_dual::DualDVec64> as fn(&[num_dual::DualDVec64]) -> num_dual::DualDVec64,
                ),
            ] {
                // gradient
                let w: Vec<Vec<f64>> = om.iter().map(|(x, _)| ad::gradient(fo, x).as_slice().to_vec()).collect();
                let ws: Vec<Vec<f64>> = sv
                    .iter()
                    .map(|(x, _)| num_dual::gradient(|v: SVector<num_dual::DualSVec64<N>, N>| fs(v.as_slice()), x).1.as_slice().to_vec())
                    .collect();
                agree("gradient", &w, &ws, 1e-12);
                let wd: Vec<Vec<f64>> = dv
                    .iter()
                    .map(|(x, _)| num_dual::gradient(|v: DVector<num_dual::DualDVec64>| fd(v.as_slice()), x).1.as_slice().to_vec())
                    .collect();
                agree("gradient dyn", &w, &wd, 1e-12);
                let s = vec![
                    mk("ours", "n Dual passes, a Matrix result", &om, move |(x, _)| ad::gradient(fo, x)),
                    mk("num-dual", "one pass of DualSVec64<n> (static)", &sv, move |(x, _)| {
                        num_dual::gradient(|v: SVector<num_dual::DualSVec64<N>, N>| fs(v.as_slice()), x)
                    }),
                    mk("num-dual (dyn)", "one pass of DualDVec64 (heap)", &dv, move |(x, _)| {
                        num_dual::gradient(|v: DVector<num_dual::DualDVec64>| fd(v.as_slice()), x)
                    }),
                ];
                out.row(g, &format!("gradient, {fname}"), &params, len, "", s);
            }

            // Hessian: Rosenbrock and the transcendental one
            for (fname, fo, fs) in [
                (
                    "Rosenbrock",
                    rosen::<HyperDual<f64>> as fn(&[HyperDual<f64>]) -> HyperDual<f64>,
                    rosen_nd::<num_dual::Dual2SVec64<N>> as fn(&[num_dual::Dual2SVec64<N>]) -> num_dual::Dual2SVec64<N>,
                ),
                (
                    "sin·exp + ln",
                    trig::<HyperDual<f64>> as fn(&[HyperDual<f64>]) -> HyperDual<f64>,
                    trig_nd::<num_dual::Dual2SVec64<N>> as fn(&[num_dual::Dual2SVec64<N>]) -> num_dual::Dual2SVec64<N>,
                ),
            ] {
                let hlen = (len / 4).max(4);
                let om = &om[..hlen];
                let sv = &sv[..hlen];
                let w: Vec<Vec<f64>> = om.iter().map(|(x, _)| ad::hessian(fo, x).as_slice().to_vec()).collect();
                let ws: Vec<Vec<f64>> = sv
                    .iter()
                    .map(|(x, _)| num_dual::hessian(|v: SVector<num_dual::Dual2SVec64<N>, N>| fs(v.as_slice()), x).2.as_slice().to_vec())
                    .collect();
                agree("hessian", &w, &ws, 1e-11);
                let s = vec![
                    mk("ours", "n(n+1)/2 HyperDual passes, a Matrix result", om, move |(x, _)| ad::hessian(fo, x)),
                    mk("num-dual", "one pass of Dual2SVec64<n> (static; value, gradient, Hessian)", sv, move |(x, _)| {
                        num_dual::hessian(|v: SVector<num_dual::Dual2SVec64<N>, N>| fs(v.as_slice()), x)
                    }),
                ];
                out.row(g, &format!("hessian, {fname}"), &params, hlen, "", s);
                let s = vec![
                    mk("ours", "value_gradient_and_hessian", om, move |(x, _)| ad::value_gradient_and_hessian(fo, x)),
                    mk("num-dual", "hessian (gives all three)", sv, move |(x, _)| {
                        num_dual::hessian(|v: SVector<num_dual::Dual2SVec64<N>, N>| fs(v.as_slice()), x)
                    }),
                ];
                out.row(g, &format!("value_gradient_and_hessian, {fname}"), &params, hlen, "", s);
            }

            // Jacobian: Broyden's tridiagonal function, m = n
            let w: Vec<Vec<f64>> = om.iter().map(|(x, _)| ad::jacobian(broyden::<Dual<f64>>, x).as_slice().to_vec()).collect();
            let jac_s = |x: &SVector<f64, N>| {
                num_dual::jacobian(
                    |v: SVector<num_dual::DualSVec64<N>, N>| SVector::<num_dual::DualSVec64<N>, N>::from_vec(broyden_nd(v.as_slice())),
                    x,
                )
            };
            let ws: Vec<Vec<f64>> = sv.iter().map(|(x, _)| jac_s(x).1.as_slice().to_vec()).collect();
            agree("jacobian", &w, &ws, 1e-12);
            let jac_d = |x: &DVector<f64>| {
                num_dual::jacobian(
                    |v: DVector<num_dual::DualDVec64>| DVector::<num_dual::DualDVec64>::from_vec(broyden_nd(v.as_slice())),
                    x,
                )
            };
            let s = vec![
                mk("ours", "n Dual passes, a Matrix result", &om, |(x, _)| ad::jacobian(broyden::<Dual<f64>>, x)),
                mk("num-dual", "one pass of DualSVec64<n> (static)", &sv, move |(x, _)| jac_s(x)),
                mk("num-dual (dyn)", "one pass of DualDVec64 (heap)", &dv, move |(x, _)| jac_d(x)),
            ];
            out.row(g, "jacobian, Broyden tridiagonal", &format!("{params}, m = {N}"), len, "", s);
            let _ = Const::<N>;
        }
    };
}

by_n!(n2, 2);
by_n!(n4, 4);
by_n!(n8, 8);
by_n!(n16, 16);
by_n!(n32, 32);

fn scalars(out: &Out) {
    let mut rng = Rng::new(77);
    let len = 1024;
    let xs: Vec<(f64, ())> = (0..len).map(|_| (rng.range(-1.5, 1.5), ())).collect();
    let g = "dual-derivatives";
    let p = "f64, sin(eˣ)/(1 + x²)";
    let w: Vec<Vec<f64>> = xs.iter().map(|(x, _)| vec![ad::derivative(comp, *x)]).collect();
    agree("derivative", &w, &xs.iter().map(|(x, _)| vec![num_dual::first_derivative(comp_nd, *x).1]).collect::<Vec<_>>(), 1e-12);
    let s = vec![
        mk("ours", "", &xs, |(x, _)| ad::derivative(comp, *x)),
        mk("num-dual", "first_derivative (Dual64)", &xs, |(x, _)| num_dual::first_derivative(comp_nd, *x)),
    ];
    out.row(g, "derivative", p, len, "", s);
    let w: Vec<Vec<f64>> = xs.iter().map(|(x, _)| vec![ad::second_derivative(comp, *x)]).collect();
    agree("second", &w, &xs.iter().map(|(x, _)| vec![num_dual::second_derivative(comp_nd, *x).2]).collect::<Vec<_>>(), 1e-11);
    let s = vec![
        mk("ours", "one HyperDual pass", &xs, |(x, _)| ad::second_derivative(comp, *x)),
        mk("num-dual", "second_derivative (Dual2_64)", &xs, |(x, _)| num_dual::second_derivative(comp_nd, *x)),
    ];
    out.row(g, "second_derivative", p, len, "", s);
    let third = |x: f64| {
        comp(Dual::new(HyperDual::new(x, 1.0, 1.0, 0.0), HyperDual::new(1.0, 0.0, 0.0, 0.0))).eps.eps1eps2
    };
    let w: Vec<Vec<f64>> = xs.iter().map(|(x, _)| vec![third(*x)]).collect();
    agree("third", &w, &xs.iter().map(|(x, _)| vec![num_dual::third_derivative(comp_nd, *x).3]).collect::<Vec<_>>(), 1e-10);
    let s = vec![
        mk("ours", "Dual<HyperDual<f64>>", &xs, move |(x, _)| third(*x)),
        mk("num-dual", "third_derivative (Dual3_64)", &xs, |(x, _)| num_dual::third_derivative(comp_nd, *x)),
    ];
    out.row(g, "third derivative (nested)", p, len, "", s);
}

fn main() {
    let out = Out::new("numtypes");
    elementaries(&out);
    scalars(&out);
    n2(&out, 2);
    n4(&out, 4);
    n8(&out, 8);
    n16(&out, 16);
    n32(&out, 32);
}

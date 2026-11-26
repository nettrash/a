//! Initial value problems: the explicit and stiff methods against GSL odeiv2, Boost.Odeint and
//! the ode_solvers crate.

use crate::harness::{Cmp, Row, bb, measure};
use a::math::matrix::Matrix;
use a::math::numerics::{
    Bdf, DormandPrince5, DormandPrince853, Event, OdeOptions, RadauIia, Rk4, integrate_with,
    solve_at_with, solve_ivp_with,
};
use ode_solvers::{DVector, Dop853, Dopri5, SVector, System};
use std::cell::Cell;
use std::sync::OnceLock;

unsafe extern "C" {
    fn probe_ode_lin_matrix(a: *mut f64);
    fn probe_gsl_ode(
        kind: i32,
        prob: i32,
        t0: f64,
        t1: f64,
        y: *mut f64,
        rtol: f64,
        atol: f64,
        nsteps: i64,
        fe: *mut i64,
        je: *mut i64,
        steps: *mut i64,
    ) -> i32;
    fn probe_gsl_ode_at(
        kind: i32,
        prob: i32,
        t0: f64,
        ts: *const f64,
        nt: i32,
        y: *mut f64,
        rtol: f64,
        atol: f64,
        fe: *mut i64,
    ) -> i32;
    fn probe_boost_ode(
        kind: i32,
        prob: i32,
        t0: f64,
        t1: f64,
        y: *mut f64,
        rtol: f64,
        atol: f64,
        nsteps: i64,
        ts: *const f64,
        nt: i32,
        fe: *mut i64,
        steps: *mut i64,
    ) -> i32;
    fn probe_boost_rosenbrock(
        prob: i32,
        t0: f64,
        t1: f64,
        y: *mut f64,
        rtol: f64,
        atol: f64,
        fe: *mut i64,
        je: *mut i64,
        steps: *mut i64,
    ) -> i32;
}

fn lin_a() -> &'static [f64] {
    static A: OnceLock<Vec<f64>> = OnceLock::new();
    A.get_or_init(|| {
        let mut a = vec![0.0; 2500];
        unsafe { probe_ode_lin_matrix(a.as_mut_ptr()) };
        a
    })
}

pub fn dim(prob: i32) -> usize {
    match prob {
        0 | 1 => 4,
        2 => 2,
        3 => 100,
        10 => 3,
        11 => 2,
        12 => 8,
        13 => 3,
        _ => 50,
    }
}

#[inline]
pub fn rhs(prob: i32, y: &[f64], f: &mut [f64]) {
    match prob {
        0 => {
            let mu = 0.012277471;
            let mp = 1.0 - mu;
            let a = y[0] + mu;
            let b = y[0] - mp;
            let mut d1 = a * a + y[1] * y[1];
            let mut d2 = b * b + y[1] * y[1];
            d1 *= d1.sqrt();
            d2 *= d2.sqrt();
            f[0] = y[2];
            f[1] = y[3];
            f[2] = y[0] + 2.0 * y[3] - mp * a / d1 - mu * b / d2;
            f[3] = y[1] - 2.0 * y[2] - mp * y[1] / d1 - mu * y[1] / d2;
        }
        1 => {
            let r2 = y[0] * y[0] + y[1] * y[1];
            let r3 = r2 * r2.sqrt();
            f[0] = y[2];
            f[1] = y[3];
            f[2] = -y[0] / r3;
            f[3] = -y[1] / r3;
        }
        2 => {
            let u = y[0] * y[0] * y[1];
            f[0] = 1.0 + u - 4.0 * y[0];
            f[1] = 3.0 * y[0] - u;
        }
        3 => {
            let m = 50;
            for i in 0..m {
                let l = if i > 0 { y[i - 1] } else { 0.0 };
                let r = if i + 1 < m { y[i + 1] } else { 0.0 };
                f[i] = y[m + i];
                f[m + i] = l - 2.0 * y[i] + r;
            }
        }
        10 => {
            f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
            f[2] = 3e7 * y[1] * y[1];
            f[1] = -f[0] - f[2];
        }
        11 => {
            f[0] = y[1];
            f[1] = 1000.0 * (1.0 - y[0] * y[0]) * y[1] - y[0];
        }
        12 => {
            f[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007;
            f[1] = 1.71 * y[0] - 8.75 * y[1];
            f[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4];
            f[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3];
            f[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6];
            f[5] = -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6];
            f[6] = 280.0 * y[5] * y[7] - 1.81 * y[6];
            f[7] = -f[6];
        }
        13 => {
            let (s, w, q) = (77.27, 0.161, 8.375e-6);
            f[0] = s * (y[1] + y[0] * (1.0 - q * y[0] - y[1]));
            f[1] = (y[2] - (1.0 + y[0]) * y[1]) / s;
            f[2] = w * (y[0] - y[2]);
        }
        _ => {
            let a = lin_a();
            for i in 0..50 {
                let mut s = 0.0;
                for j in 0..50 {
                    s += a[i * 50 + j] * y[j];
                }
                f[i] = s;
            }
        }
    }
}

/// Row-major Jacobian.
pub fn jac(prob: i32, y: &[f64], j: &mut [f64]) {
    j.iter_mut().for_each(|v| *v = 0.0);
    match prob {
        10 => {
            j[0] = -0.04;
            j[1] = 1e4 * y[2];
            j[2] = 1e4 * y[1];
            j[3] = 0.04;
            j[4] = -1e4 * y[2] - 6e7 * y[1];
            j[5] = -1e4 * y[1];
            j[7] = 6e7 * y[1];
        }
        11 => {
            j[1] = 1.0;
            j[2] = -2000.0 * y[0] * y[1] - 1.0;
            j[3] = 1000.0 * (1.0 - y[0] * y[0]);
        }
        12 => {
            let s = |j: &mut [f64], r: usize, c: usize, v: f64| j[r * 8 + c] = v;
            s(j, 0, 0, -1.71);
            s(j, 0, 1, 0.43);
            s(j, 0, 2, 8.32);
            s(j, 1, 0, 1.71);
            s(j, 1, 1, -8.75);
            s(j, 2, 2, -10.03);
            s(j, 2, 3, 0.43);
            s(j, 2, 4, 0.035);
            s(j, 3, 1, 8.32);
            s(j, 3, 2, 1.71);
            s(j, 3, 3, -1.12);
            s(j, 4, 4, -1.745);
            s(j, 4, 5, 0.43);
            s(j, 4, 6, 0.43);
            s(j, 5, 3, 0.69);
            s(j, 5, 4, 1.71);
            s(j, 5, 5, -0.43 - 280.0 * y[7]);
            s(j, 5, 6, 0.69);
            s(j, 5, 7, -280.0 * y[5]);
            s(j, 6, 5, 280.0 * y[7]);
            s(j, 6, 6, -1.81);
            s(j, 6, 7, 280.0 * y[5]);
            s(j, 7, 5, -280.0 * y[7]);
            s(j, 7, 6, 1.81);
            s(j, 7, 7, -280.0 * y[5]);
        }
        13 => {
            let (s, w, q) = (77.27, 0.161, 8.375e-6);
            j[0] = s * (1.0 - 2.0 * q * y[0] - y[1]);
            j[1] = s * (1.0 - y[0]);
            j[3] = -y[1] / s;
            j[4] = -(1.0 + y[0]) / s;
            j[5] = 1.0 / s;
            j[6] = w;
            j[8] = -w;
        }
        _ => j.copy_from_slice(lin_a()),
    }
}

pub fn start(prob: i32) -> (Vec<f64>, f64, f64) {
    match prob {
        0 => (vec![0.994, 0.0, 0.0, -2.001_585_106_379_082_5], 0.0, 17.065_216_560_157_962),
        1 => {
            let e: f64 = 0.9;
            (vec![1.0 - e, 0.0, 0.0, ((1.0 + e) / (1.0 - e)).sqrt()], 0.0, 6.0 * std::f64::consts::PI)
        }
        2 => (vec![1.5, 3.0], 0.0, 20.0),
        3 => {
            let mut y = vec![0.0; 100];
            for i in 0..50 {
                y[i] = (std::f64::consts::PI * (i + 1) as f64 / 51.0).sin();
            }
            (y, 0.0, 10.0)
        }
        10 => (vec![1.0, 0.0, 0.0], 0.0, 1e11),
        11 => (vec![2.0, 0.0], 0.0, 3000.0),
        12 => (vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057], 0.0, 321.8122),
        13 => (vec![1.0, 2.0, 3.0], 0.0, 360.0),
        _ => (vec![1.0; 50], 0.0, 10.0),
    }
}

fn col(v: &[f64]) -> Matrix<f64> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

/// The reference solution at t₁: closed forms where there are, else ours at tight tolerances.
fn reference(prob: i32) -> Vec<f64> {
    let (y0, t0, t1) = start(prob);
    match prob {
        1 => y0,
        3 => {
            let w = 2.0 * (std::f64::consts::PI / (2.0 * 51.0)).sin();
            let mut y = vec![0.0; 100];
            for i in 0..50 {
                let s = (std::f64::consts::PI * (i + 1) as f64 / 51.0).sin();
                y[i] = s * (w * t1).cos();
                y[50 + i] = -s * w * (w * t1).sin();
            }
            y
        }
        0 | 2 => {
            let o = OdeOptions::new().with_tolerances(1e-14, 1e-14).with_dense_output(false).with_max_steps(10_000_000);
            let r = integrate_with(|_t, y: &Matrix<f64>, dy: &mut Matrix<f64>| rhs(prob, y.as_slice(), dy.as_mut_slice()), &col(&y0), t0, t1, DormandPrince853::new(), &o).unwrap();
            r.y.as_slice().to_vec()
        }
        _ => {
            let atol = stiff_atol(prob, 1e-12);
            let o = OdeOptions::new().with_tolerances(1e-12, atol).with_dense_output(false).with_max_steps(10_000_000);
            let n = dim(prob);
            let m = RadauIia::new().with_jacobian(move |_t, y: &Matrix<f64>, jm: &mut Matrix<f64>| {
                let mut jr = vec![0.0; n * n];
                jac(prob, y.as_slice(), &mut jr);
                for i in 0..n {
                    for k in 0..n {
                        jm[(i, k)] = jr[i * n + k];
                    }
                }
            });
            let r = integrate_with(|_t, y: &Matrix<f64>, dy: &mut Matrix<f64>| rhs(prob, y.as_slice(), dy.as_mut_slice()), &col(&y0), t0, t1, m, &o).unwrap();
            r.y.as_slice().to_vec()
        }
    }
}

pub fn reference_pub(prob: i32) -> Vec<f64> {
    reference(prob)
}
pub fn lin_a_pub() -> Vec<f64> {
    lin_a().to_vec()
}

fn stiff_atol(prob: i32, rtol: f64) -> f64 {
    match prob {
        10 => 1e-8 * rtol,
        14 => 1e-6 * rtol,
        _ => 1e-4 * rtol,
    }
}

fn nanmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}
fn err_nonstiff(y: &[f64], r: &[f64]) -> f64 {
    y.iter().zip(r).map(|(a, b)| (a - b).abs() / (1.0 + b.abs())).fold(0.0, nanmax)
}
fn err_stiff(y: &[f64], r: &[f64], atol: f64) -> f64 {
    y.iter().zip(r).map(|(a, b)| (a - b).abs() / (b.abs() + atol)).fold(0.0, nanmax)
}

// ode_solvers systems.
struct Os {
    prob: i32,
    evals: Cell<u64>,
}
macro_rules! os_impl {
    ($v:ty) => {
        impl System<f64, $v> for Os {
            fn system(&self, _x: f64, y: &$v, dy: &mut $v) {
                self.evals.set(self.evals.get() + 1);
                rhs(self.prob, y.as_slice(), dy.as_mut_slice());
            }
        }
    };
}
os_impl!(SVector<f64, 2>);
os_impl!(SVector<f64, 4>);
os_impl!(DVector<f64>);

/// ode_solvers Dopri5 (kind 0) or Dop853 (1) to t₁, sparse output (every accepted step stored;
/// its dense output with dx = t₁ − t₀ returned a wrong end state). Returns y, evaluations, steps.
fn os_run(kind: i32, prob: i32, rtol: f64, atol: f64) -> (Vec<f64>, u64, u64) {
    let (y0, t0, t1) = start(prob);
    let sys = Os { prob, evals: Cell::new(0) };
    macro_rules! go {
        ($v:expr) => {{
            if kind == 0 {
                let mut s = Dopri5::new(sys, t0, t1, t1 - t0, $v, rtol, atol);
                s.set_output(ode_solvers::OutputType::Sparse);
                let st = s.integrate();
                let y = s.y_out().last().unwrap().as_slice().to_vec();
                let st = st.map(|s| (s.num_eval as u64, s.accepted_steps as u64)).unwrap_or((0, 0));
                (y, st.0, st.1)
            } else {
                let mut s = Dop853::new(sys, t0, t1, t1 - t0, $v, rtol, atol);
                s.set_output(ode_solvers::OutputType::Sparse);
                let st = s.integrate();
                let y = s.y_out().last().unwrap().as_slice().to_vec();
                let st = st.map(|s| (s.num_eval as u64, s.accepted_steps as u64)).unwrap_or((0, 0));
                (y, st.0, st.1)
            }
        }};
    }
    match y0.len() {
        2 => go!(SVector::<f64, 2>::from_column_slice(&y0)),
        4 => go!(SVector::<f64, 4>::from_column_slice(&y0)),
        _ => go!(DVector::<f64>::from_column_slice(&y0)),
    }
}

fn gsl_run(kind: i32, prob: i32, rtol: f64, atol: f64, nsteps: i64) -> (Vec<f64>, i64, i64, i64, i32) {
    let (mut y, t0, t1) = start(prob);
    let (mut fe, mut je, mut st) = (0i64, 0i64, 0i64);
    let s = unsafe { probe_gsl_ode(kind, prob, t0, t1, y.as_mut_ptr(), rtol, atol, nsteps, &mut fe, &mut je, &mut st) };
    (y, fe, je, st, s)
}
fn boost_run(kind: i32, prob: i32, rtol: f64, atol: f64, nsteps: i64) -> (Vec<f64>, i64, i64) {
    let (mut y, t0, t1) = start(prob);
    let (mut fe, mut st) = (0i64, 0i64);
    unsafe { probe_boost_ode(kind, prob, t0, t1, y.as_mut_ptr(), rtol, atol, nsteps, std::ptr::null(), 0, &mut fe, &mut st) };
    (y, fe, st)
}
fn rosen_run(prob: i32, rtol: f64, atol: f64) -> (Vec<f64>, i64, i64, i64) {
    let (mut y, t0, t1) = start(prob);
    let (mut fe, mut je, mut st) = (0i64, 0i64, 0i64);
    unsafe { probe_boost_rosenbrock(prob, t0, t1, y.as_mut_ptr(), rtol, atol, &mut fe, &mut je, &mut st) };
    (y, fe, je, st)
}

fn ours_ns<M: a::math::numerics::OdeMethod<f64>>(prob: i32, m: M, o: &OdeOptions<f64>) -> a::math::numerics::OdeResult<f64> {
    let (y0, t0, t1) = start(prob);
    integrate_with(|_t, y: &Matrix<f64>, dy: &mut Matrix<f64>| rhs(prob, y.as_slice(), dy.as_mut_slice()), &col(&y0), t0, t1, m, o).unwrap()
}

pub fn run() {
    nonstiff();
    fixed();
    dense_and_events();
    stiff();
}

fn nonstiff() {
    let probs: [(i32, &str); 4] = [
        (0, "Arenstorf orbit, one period (n = 4)"),
        (1, "Kepler e = 0.9, three periods (n = 4)"),
        (2, "Brusselator on [0, 20] (n = 2)"),
        (3, "chain of 50 oscillators on [0, 10] (n = 100)"),
    ];
    for (prob, name) in probs {
        let r = reference(prob);
        let (y0, t0, t1) = start(prob);
        let y0m = col(&y0);
        for tol in [1e-6, 1e-10] {
            let o = OdeOptions::new().with_tolerances(tol, tol).with_dense_output(false);
            let params = format!("{name}; rtol = atol = {tol:e}");
            let ro = |x: &a::math::numerics::OdeResult<f64>| {
                format!("{} steps, {} rejected, {} f, err {:.1e}", x.steps, x.rejected, x.nfev, err_nonstiff(x.y.as_slice(), &r))
            };
            let gn = |k: i32| {
                let (y, fe, _, st, s) = gsl_run(k, prob, tol, tol, 0);
                format!("{st} steps, {fe} f, err {:.1e}{}", err_nonstiff(&y, &r), if s != 0 { format!(", status {s}") } else { String::new() })
            };
            let bn = |k: i32| {
                let (y, fe, st) = boost_run(k, prob, tol, tol, 0);
                format!("{st} steps, {fe} f, err {:.1e}", err_nonstiff(&y, &r))
            };
            let on = |k: i32| {
                let (y, fe, st) = os_run(k, prob, tol, tol);
                format!("{st} steps, {fe} f, err {:.1e}", err_nonstiff(&y, &r))
            };
            let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| rhs(prob, y.as_slice(), dy.as_mut_slice());
            // DormandPrince5
            {
                let x = ours_ns(prob, DormandPrince5::new(), &o);
                let mut a = || {
                    bb(integrate_with(f, &y0m, t0, t1, DormandPrince5::new(), &o));
                };
                let mut r1 = || {
                    bb(os_run(0, prob, tol, tol));
                };
                let mut r2 = || {
                    bb(boost_run(0, prob, tol, tol, 0));
                };
                let mut r3 = || {
                    bb(gsl_run(0, prob, tol, tol, 0));
                };
                let mut r4 = || {
                    bb(gsl_run(1, prob, tol, tol, 0));
                };
                let tt = Cmp { group: "ode", op: "DormandPrince5", params: &params }.run(
                    &mut [&mut a, &mut r1, &mut r2, &mut r3, &mut r4],
                    &["ode_solvers 0.6 Dopri5", "Boost 1.92 odeint dopri5 (controlled)", "GSL 2.8 odeiv2 rkf45", "GSL 2.8 odeiv2 rkck"],
                    &format!("{}; RMS norm", ro(&x)),
                    &[
                        format!("{}; Hairer's DOPRI5 port, RMS norm", on(0)),
                        format!("{}; max norm, atol + rtol·|y|", bn(0)),
                        format!("{}; Fehlberg 4(5), max norm", gn(0)),
                        format!("{}; Cash–Karp 4(5), max norm", gn(1)),
                    ],
                    1.0,
                );
                crate::harness::py_rows("ode", "DormandPrince5", &params, &format!("ode:DormandPrince5:{prob}:{tol:e}"), tt[0].median, &ro(&x), None);
            }
            // DormandPrince853
            {
                let x = ours_ns(prob, DormandPrince853::new(), &o);
                let mut a = || {
                    bb(integrate_with(f, &y0m, t0, t1, DormandPrince853::new(), &o));
                };
                let mut r1 = || {
                    bb(os_run(1, prob, tol, tol));
                };
                let mut r2 = || {
                    bb(gsl_run(2, prob, tol, tol, 0));
                };
                let mut r3 = || {
                    bb(boost_run(1, prob, tol, tol, 0));
                };
                let mut r4 = || {
                    bb(gsl_run(3, prob, tol, tol, 0));
                };
                let mut r5 = || {
                    bb(boost_run(5, prob, tol, tol, 0));
                };
                let tt = Cmp { group: "ode", op: "DormandPrince853", params: &params }.run(
                    &mut [&mut a, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5],
                    &["ode_solvers 0.6 Dop853", "GSL 2.8 odeiv2 rk8pd", "Boost 1.92 odeint fehlberg78 (controlled)", "GSL 2.8 odeiv2 msadams", "Boost 1.92 odeint bulirsch_stoer"],
                    &format!("{}; RMS norm", ro(&x)),
                    &[
                        format!("{}; Hairer's DOP853 port", on(1)),
                        format!("{}; Prince–Dormand 8(7), max norm", gn(2)),
                        format!("{}; Fehlberg 7(8), max norm", bn(1)),
                        format!("{}; variable-order Adams (another method)", gn(3)),
                        format!("{}; extrapolation (another method)", bn(5)),
                    ],
                    1.0,
                );
                crate::harness::py_rows("ode", "DormandPrince853", &params, &format!("ode:DormandPrince853:{prob}:{tol:e}"), tt[0].median, &ro(&x), None);
            }
            // Rk4 adaptive
            if tol == 1e-6 {
                let x = ours_ns(prob, Rk4::new(), &o);
                let mut a = || {
                    bb(integrate_with(f, &y0m, t0, t1, Rk4::new(), &o));
                };
                let mut r1 = || {
                    bb(gsl_run(8, prob, tol, tol, 0));
                };
                Cmp { group: "ode", op: "Rk4", params: &format!("{params}; adaptive") }.run(
                    &mut [&mut a, &mut r1],
                    &["GSL 2.8 odeiv2 rk4 (adaptive)"],
                    &ro(&x),
                    &[format!("{}; error by step doubling", gn(8))],
                    1.0,
                );
            }
        }
    }
}

fn fixed() {
    for (prob, name, steps) in [
        (2, "Brusselator on [0, 20] (n = 2)", 2000i64),
        (1, "Kepler e = 0.9 (n = 4)", 20000),
        (3, "chain of 50 oscillators (n = 100)", 2000),
    ] {
        let (y0, t0, t1) = start(prob);
        let y0m = col(&y0);
        let o = OdeOptions::new().with_dense_output(false);
        let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| rhs(prob, y.as_slice(), dy.as_mut_slice());
        let mut a = || {
            bb(integrate_with(f, &y0m, t0, t1, Rk4::fixed(steps as usize), &o));
        };
        let mut r1 = || {
            bb(boost_run(3, prob, 0.0, 0.0, steps));
        };
        let mut r2 = || {
            bb(gsl_run(8, prob, 1e-6, 1e-6, steps));
        };
        let h = (t1 - t0) / steps as f64;
        let mut r3 = || {
            let sys = Os { prob, evals: Cell::new(0) };
            match y0.len() {
                2 => {
                    let mut s = ode_solvers::Rk4::new(sys, t0, SVector::<f64, 2>::from_column_slice(&y0), t1, h);
                    bb(s.integrate().ok());
                }
                4 => {
                    let mut s = ode_solvers::Rk4::new(sys, t0, SVector::<f64, 4>::from_column_slice(&y0), t1, h);
                    bb(s.integrate().ok());
                }
                _ => {
                    let mut s = ode_solvers::Rk4::new(sys, t0, DVector::<f64>::from_column_slice(&y0), t1, h);
                    bb(s.integrate().ok());
                }
            }
        };
        let gs = gsl_run(8, prob, 1e-6, 1e-6, steps).4;
        let params = format!("{name}; {steps} fixed steps; ns per step");
        Cmp { group: "ode", op: "Rk4::fixed", params: &params }.run(
            &mut [&mut a, &mut r1, &mut r2, &mut r3],
            &["Boost 1.92 odeint runge_kutta4 (integrate_n_steps)", "GSL 2.8 odeiv2 rk4 (apply_fixed_step)", "ode_solvers 0.6 Rk4"],
            "4 f per step",
            &[
                "4 f per step".into(),
                format!("12 f per step: GSL's rk4 also estimates the error by step doubling (does more); status {gs}"),
                "4 f per step; stores every step's state".into(),
            ],
            steps as f64,
        );
    }
}

fn dense_and_events() {
    // solve_at: Kepler at 1000 output times.
    let prob = 1;
    let (y0, t0, t1) = start(prob);
    let y0m = col(&y0);
    let nt = 1000;
    let ts: Vec<f64> = (1..=nt).map(|i| t0 + (t1 - t0) * i as f64 / nt as f64).collect();
    let tol = 1e-8;
    let o = OdeOptions::new().with_tolerances(tol, tol);
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| rhs(prob, y.as_slice(), dy.as_mut_slice());
    {
        let mut a = || {
            bb(solve_at_with(f, &y0m, t0, &ts, DormandPrince5::new(), &o));
        };
        let mut r1 = || {
            let mut y = y0.clone();
            let (mut fe, mut st) = (0i64, 0i64);
            unsafe { probe_boost_ode(4, prob, t0, t1, y.as_mut_ptr(), tol, tol, 0, ts.as_ptr(), nt as i32, &mut fe, &mut st) };
            bb(y);
        };
        let mut r2 = || {
            let sys = Os { prob, evals: Cell::new(0) };
            let mut s = Dopri5::new(sys, t0, t1, (t1 - t0) / nt as f64, SVector::<f64, 4>::from_column_slice(&y0), tol, tol);
            bb(s.integrate().ok());
            bb(s.y_out().len());
        };
        let mut r3 = || {
            let mut y = y0.clone();
            let mut fe = 0i64;
            unsafe { probe_gsl_ode_at(0, prob, t0, ts.as_ptr(), nt as i32, y.as_mut_ptr(), tol, tol, &mut fe) };
            bb(y);
        };
        Cmp { group: "ode", op: "solve_at", params: "Kepler e = 0.9, 3 periods, 1000 output times; DormandPrince5, rtol = atol = 1e-8" }.run(
            &mut [&mut a, &mut r1, &mut r2, &mut r3],
            &["Boost 1.92 odeint dopri5 dense output (integrate_times)", "ode_solvers 0.6 Dopri5 (dense output every dx)", "GSL 2.8 odeiv2 rkf45 driver to each time"],
            "dense output between steps",
            &["dense output".into(), "dense output".into(), "no dense output: steps cut at every output time (does more f)".into()],
            1.0,
        );
    }
    // DenseOutput::eval per point vs ode_solvers' continuous output model.
    {
        let sol = solve_ivp_with(f, &y0m, t0, t1, DormandPrince5::new(), &mut [], &o);
        let dense = sol.dense.unwrap();
        let mut rng = crate::harness::Lcg(7);
        let pts: Vec<f64> = (0..10_000).map(|_| t0 + (t1 - t0) * rng.next()).collect();
        let sys = Os { prob, evals: Cell::new(0) };
        let mut s = Dopri5::new(sys, t0, t1, t1 - t0, SVector::<f64, 4>::from_column_slice(&y0), tol, tol);
        let mut model = ode_solvers::continuous_output_model::ContinuousOutputModel::default();
        s.integrate_with_continuous_output_model(&mut model).unwrap();
        let mut out = vec![0.0; 4];
        let mut a = || {
            let mut acc = 0.0;
            for &t in &pts {
                dense.eval_into(t, &mut out);
                acc += out[0];
            }
            bb(acc);
        };
        let mut r1 = || {
            let mut acc = 0.0;
            for &t in &pts {
                acc += model.evaluate(t).map(|v| v[0]).unwrap_or(0.0);
            }
            bb(acc);
        };
        Cmp { group: "ode", op: "DenseOutput::eval_into", params: "Kepler, DormandPrince5 at 1e-8; 10⁴ random times; ns per point" }.run(
            &mut [&mut a, &mut r1],
            &["ode_solvers 0.6 ContinuousOutputModel::evaluate"],
            &format!("{} steps", dense.steps().len()),
            &["returns a new vector per point".into()],
            pts.len() as f64,
        );
    }
    // Events: Arenstorf crossings of y₂ = 0.
    {
        let prob = 0;
        let (y0, t0, t1) = start(prob);
        let y0m = col(&y0);
        let o = OdeOptions::new().with_tolerances(1e-8, 1e-8).with_dense_output(false);
        let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| rhs(prob, y.as_slice(), dy.as_mut_slice());
        let s0 = solve_ivp_with(f, &y0m, t0, t1, DormandPrince5::new(), &mut [Event::new(|_t, y: &Matrix<f64>| y[(1, 0)])], &o);
        let mut a = || {
            let mut ev = [Event::new(|_t, y: &Matrix<f64>| y[(1, 0)])];
            bb(solve_ivp_with(f, &y0m, t0, t1, DormandPrince5::new(), &mut ev, &o));
        };
        let mut b = || {
            bb(solve_ivp_with(f, &y0m, t0, t1, DormandPrince5::new(), &mut [], &o));
        };
        let t = measure(&mut [&mut a, &mut b]);
        Row {
            group: "ode".into(),
            op: "solve_ivp (events)".into(),
            params: "Arenstorf orbit; event y₂ = 0 (both directions); DormandPrince5 at 1e-8".into(),
            ours: t[0].median,
            reference: "none (no compiled reference locates events)".into(),
            refns: None,
            notes: format!("{} events, {} f; the same run without the event {:.0} ns", s0.events.len(), s0.nfev, t[1].median),
        }
        .print();
    }
}

fn stiff() {
    let probs: [(i32, &str, &[f64]); 5] = [
        (10, "Robertson to 10¹¹ (n = 3)", &[1e-4, 1e-6, 1e-8]),
        (11, "Van der Pol μ = 1000 on [0, 3000] (n = 2)", &[1e-4, 1e-6, 1e-8]),
        (12, "HIRES (n = 8)", &[1e-6]),
        (13, "OREGO on [0, 360] (n = 3)", &[1e-6]),
        (14, "linear y′ = Q·diag(λ)·Q·y, λ to −10⁶ (n = 50)", &[1e-6]),
    ];
    for (prob, name, tols) in probs {
        let r = reference(prob);
        let (y0, t0, t1) = start(prob);
        let y0m = col(&y0);
        let n = dim(prob);
        for &rtol in tols {
            let atol = stiff_atol(prob, rtol);
            let o = OdeOptions::new().with_tolerances(rtol, atol).with_dense_output(false).with_max_steps(1_000_000);
            let params = format!("{name}; rtol {rtol:e}, atol {atol:e}");
            let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| rhs(prob, y.as_slice(), dy.as_mut_slice());
            // The row-major Jacobian transposed into the column-major matrix (a stack buffer
            // for the small systems, so that the harness adds no zeroing of a large one).
            let jm = move |_t: f64, y: &Matrix<f64>, jx: &mut Matrix<f64>| {
                let out = jx.as_mut_slice();
                if n <= 8 {
                    let mut jr = [0.0f64; 64];
                    jac(prob, y.as_slice(), &mut jr[..n * n]);
                    for i in 0..n {
                        for k in 0..n {
                            out[k * n + i] = jr[i * n + k];
                        }
                    }
                } else {
                    // The linear problem: A is symmetric, its row-major copy is its column-major one.
                    out.copy_from_slice(lin_a());
                }
            };
            let ro = |x: &a::math::numerics::OdeResult<f64>| {
                format!("{} steps, {} rejected, {} f, {} J, {} LU, err {:.1e}", x.steps, x.rejected, x.nfev, x.njev, x.nlu, err_stiff(x.y.as_slice(), &r, atol))
            };
            let gn = |k: i32| {
                let (y, fe, je, st, s) = gsl_run(k, prob, rtol, atol, 0);
                format!("{st} steps, {fe} f, {je} J, err {:.1e}{}", err_stiff(&y, &r, atol), if s != 0 { format!(", status {s}") } else { String::new() })
            };
            let rn = || {
                let (y, fe, je, st) = rosen_run(prob, rtol, atol);
                format!("{st} steps, {fe} f, {je} J, err {:.1e}", err_stiff(&y, &r, atol))
            };
            for (op, which) in [("RadauIia", 0), ("Bdf", 1)] {
                let xa = if which == 0 {
                    integrate_with(f, &y0m, t0, t1, RadauIia::new().with_jacobian(jm), &o).unwrap()
                } else {
                    integrate_with(f, &y0m, t0, t1, Bdf::new().with_jacobian(jm), &o).unwrap()
                };
                let xd = if which == 0 {
                    integrate_with(f, &y0m, t0, t1, RadauIia::new(), &o).unwrap()
                } else {
                    integrate_with(f, &y0m, t0, t1, Bdf::new(), &o).unwrap()
                };
                let mut a = || {
                    if which == 0 {
                        bb(integrate_with(f, &y0m, t0, t1, RadauIia::new().with_jacobian(jm), &o));
                    } else {
                        bb(integrate_with(f, &y0m, t0, t1, Bdf::new().with_jacobian(jm), &o));
                    }
                };
                let mut ad = || {
                    if which == 0 {
                        bb(integrate_with(f, &y0m, t0, t1, RadauIia::new(), &o));
                    } else {
                        bb(integrate_with(f, &y0m, t0, t1, Bdf::new(), &o));
                    }
                };
                let mut g1 = || {
                    bb(gsl_run(4, prob, rtol, atol, 0));
                };
                let mut g2 = || {
                    bb(gsl_run(5, prob, rtol, atol, 0));
                };
                let mut b1 = || {
                    bb(rosen_run(prob, rtol, atol));
                };
                let t = measure(&mut [&mut a, &mut ad, &mut g1, &mut g2, &mut b1]);
                let sp = crate::harness::spread_note(&[&t[0], &t[2], &t[3], &t[4]]);
                let ours_note = format!(
                    "ours: {}; RMS norm; with the differenced Jacobian {:.0} ns ({})",
                    ro(&xa),
                    t[1].median,
                    ro(&xd)
                );
                for (i, (rname, rnote)) in [
                    ("GSL 2.8 odeiv2 msbdf (analytic J)", gn(4)),
                    ("GSL 2.8 odeiv2 bsimp (analytic J)", gn(5)),
                    ("Boost 1.92 odeint rosenbrock4 (analytic J)", rn()),
                ]
                .into_iter()
                .enumerate()
                {
                    Row {
                        group: "ode".into(),
                        op: format!("{op} (analytic J)"),
                        params: params.clone(),
                        ours: t[0].median,
                        reference: rname.into(),
                        refns: Some(t[2 + i].median),
                        notes: format!("{ours_note}; ref: {rnote}{sp}"),
                    }
                    .print();
                }
                crate::harness::py_rows("ode", &format!("{op} (analytic J)"), &params, &format!("ode:{op}:{prob}:{rtol:e}"), t[0].median, &ro(&xa), None);
            }
        }
    }
}

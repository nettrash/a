//! Systems of equations: newton_system and broyden against GSL's multiroot solvers on the
//! Moré–Garbow–Hillstrom problems.

use crate::harness::{Cmp, bb};
use a::math::matrix::Matrix;
use a::math::numerics::{broyden, newton_system};

unsafe extern "C" {
    fn probe_gsl_multiroot(
        kind: i32,
        prob: i32,
        n: i32,
        x: *mut f64,
        fevals: *mut i64,
        jevals: *mut i64,
        iters: *mut i32,
    ) -> i32;
}

pub fn sys_f(prob: i32, n: usize, x: &[f64], f: &mut [f64]) {
    match prob {
        0 => {
            f[0] = 10.0 * (x[1] - x[0] * x[0]);
            f[1] = 1.0 - x[0];
        }
        1 => {
            f[0] = 1e4 * x[0] * x[1] - 1.0;
            f[1] = (-x[0]).exp() + (-x[1]).exp() - 1.0001;
        }
        2 => {
            let mut th = (x[1] / x[0]).atan() / (2.0 * std::f64::consts::PI);
            if x[0] < 0.0 {
                th += 0.5;
            }
            f[0] = 10.0 * (x[2] - 10.0 * th);
            f[1] = 10.0 * ((x[0] * x[0] + x[1] * x[1]).sqrt() - 1.0);
            f[2] = x[2];
        }
        3 => {
            let h = 1.0 / (n + 1) as f64;
            for i in 0..n {
                let t = (i + 1) as f64 * h;
                let xl = if i > 0 { x[i - 1] } else { 0.0 };
                let xr = if i + 1 < n { x[i + 1] } else { 0.0 };
                let u = x[i] + t + 1.0;
                f[i] = 2.0 * x[i] - xl - xr + 0.5 * h * h * u * u * u;
            }
        }
        4 => {
            for i in 0..n {
                let xl = if i > 0 { x[i - 1] } else { 0.0 };
                let xr = if i + 1 < n { x[i + 1] } else { 0.0 };
                f[i] = (3.0 - 2.0 * x[i]) * x[i] - xl - 2.0 * xr + 1.0;
            }
        }
        _ => {
            let s: f64 = x.iter().map(|v| v.cos()).sum();
            for i in 0..n {
                f[i] = n as f64 - s + (i + 1) as f64 * (1.0 - x[i].cos()) - x[i].sin();
            }
        }
    }
}

/// Row-major.
pub fn sys_j(prob: i32, n: usize, x: &[f64], j: &mut [f64]) {
    j.iter_mut().for_each(|v| *v = 0.0);
    match prob {
        0 => {
            j[0] = -20.0 * x[0];
            j[1] = 10.0;
            j[2] = -1.0;
        }
        1 => {
            j[0] = 1e4 * x[1];
            j[1] = 1e4 * x[0];
            j[2] = -(-x[0]).exp();
            j[3] = -(-x[1]).exp();
        }
        2 => {
            let r2 = x[0] * x[0] + x[1] * x[1];
            let r = r2.sqrt();
            let tp = 2.0 * std::f64::consts::PI;
            j[0] = 100.0 * x[1] / (tp * r2);
            j[1] = -100.0 * x[0] / (tp * r2);
            j[2] = 10.0;
            j[3] = 10.0 * x[0] / r;
            j[4] = 10.0 * x[1] / r;
            j[8] = 1.0;
        }
        3 => {
            let h = 1.0 / (n + 1) as f64;
            for i in 0..n {
                let t = (i + 1) as f64 * h;
                let u = x[i] + t + 1.0;
                j[i * n + i] = 2.0 + 1.5 * h * h * u * u;
                if i > 0 {
                    j[i * n + i - 1] = -1.0;
                }
                if i + 1 < n {
                    j[i * n + i + 1] = -1.0;
                }
            }
        }
        4 => {
            for i in 0..n {
                j[i * n + i] = 3.0 - 4.0 * x[i];
                if i > 0 {
                    j[i * n + i - 1] = -1.0;
                }
                if i + 1 < n {
                    j[i * n + i + 1] = -2.0;
                }
            }
        }
        _ => {
            for i in 0..n {
                for k in 0..n {
                    j[i * n + k] = x[k].sin()
                        + if i == k {
                            (i + 1) as f64 * x[i].sin() - x[i].cos()
                        } else {
                            0.0
                        };
                }
            }
        }
    }
}

fn start(prob: i32, n: usize) -> Vec<f64> {
    match prob {
        0 => vec![-1.2, 1.0],
        1 => vec![0.0, 1.0],
        2 => vec![-1.0, 0.0, 0.0],
        3 => {
            let h = 1.0 / (n + 1) as f64;
            (0..n)
                .map(|i| {
                    let t = (i + 1) as f64 * h;
                    t * (t - 1.0)
                })
                .collect()
        }
        4 => vec![-1.0; n],
        _ => vec![1.0 / n as f64; n],
    }
}

fn col(v: Vec<f64>) -> Matrix<f64> {
    let n = v.len();
    Matrix::from_column_major(n, 1, v)
}

pub fn run() {
    let cases: [(i32, usize, &str); 8] = [
        (0, 2, "Rosenbrock, n = 2"),
        (1, 2, "Powell badly scaled, n = 2"),
        (2, 3, "helical valley, n = 3"),
        (3, 10, "discrete boundary value, n = 10"),
        (3, 100, "discrete boundary value, n = 100"),
        (4, 10, "Broyden tridiagonal, n = 10"),
        (4, 100, "Broyden tridiagonal, n = 100"),
        (5, 10, "trigonometric, n = 10"),
    ];
    for (prob, n, name) in cases {
        let x0 = col(start(prob, n));
        let f = |x: &Matrix<f64>| {
            let mut v = vec![0.0; n];
            sys_f(prob, n, x.as_slice(), &mut v);
            col(v)
        };
        let jac = |x: &Matrix<f64>| {
            let mut v = vec![0.0; n * n];
            sys_j(prob, n, x.as_slice(), &mut v);
            Matrix::from_row_major(n, n, v)
        };
        let reference = newton_system(&x0, f, jac).unwrap();
        let xr = reference.x.as_slice().to_vec();
        let scale = xr.iter().fold(1e-300f64, |m, v| m.max(v.abs()));
        let diff = |x: &[f64]| -> f64 {
            x.iter()
                .zip(&xr)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
                / scale
        };
        let ours_note = |r: &a::math::numerics::SystemResult<f64>| {
            format!(
                "{} iterations, {} f, {} J, {}converged, ‖f‖∞ {:.1e}",
                r.iterations,
                r.evaluations,
                r.jacobians,
                if r.converged { "" } else { "not " },
                r.residual
            )
        };
        let gsl = |kind: i32| -> String {
            let mut x = start(prob, n);
            let (mut fe, mut je, mut it) = (0i64, 0i64, 0i32);
            let st = unsafe {
                probe_gsl_multiroot(kind, prob, n as i32, x.as_mut_ptr(), &mut fe, &mut je, &mut it)
            };
            let mut fv = vec![0.0; n];
            sys_f(prob, n, &x, &mut fv);
            let res = fv.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            format!(
                "{it} iterations, {fe} f, {je} J, status {st}{}, ‖f‖∞ {res:.1e}, off ours by {:.1e}",
                if st == 0 { " (converged)" } else { " (failed)" },
                diff(&x)
            )
        };
        let gsl_time = |kind: i32| {
            move || {
                let mut x = start(prob, n);
                let (mut fe, mut je, mut it) = (0i64, 0i64, 0i32);
                bb(unsafe {
                    probe_gsl_multiroot(kind, prob, n as i32, x.as_mut_ptr(), &mut fe, &mut je, &mut it)
                });
                bb(x);
            }
        };
        let params = format!("{name}; standard start; xtol 4ε");
        // newton_system (with the Jacobian)
        {
            let on = ours_note(&reference);
            let mut a = || {
                bb(newton_system(&x0, f, jac));
            };
            let mut g0 = gsl_time(0);
            let mut g1 = gsl_time(1);
            let mut g2 = gsl_time(2);
            Cmp { group: "systems", op: "newton_system", params: &params }.run(
                &mut [&mut a, &mut g0, &mut g1, &mut g2],
                &[
                    "GSL 2.8 multiroot hybridsj",
                    "GSL 2.8 multiroot newton",
                    "GSL 2.8 multiroot gnewton",
                ],
                &on,
                &[
                    format!("{}; Powell hybrid (MINPACK hybrj)", gsl(0)),
                    format!("{}; plain Newton, no line search", gsl(1)),
                    format!("{}; Newton with a backtracking search", gsl(2)),
                ],
                1.0,
            );
        }
        // broyden (no Jacobian)
        {
            let r = broyden(&x0, f).unwrap();
            let on = format!("{}, off newton_system by {:.1e}", ours_note(&r), diff(r.x.as_slice()));
            let mut a = || {
                bb(broyden(&x0, f));
            };
            let mut g3 = gsl_time(3);
            let mut g4 = gsl_time(4);
            let mut g5 = gsl_time(5);
            Cmp { group: "systems", op: "broyden", params: &params }.run(
                &mut [&mut a, &mut g3, &mut g4, &mut g5],
                &[
                    "GSL 2.8 multiroot hybrids",
                    "GSL 2.8 multiroot dnewton",
                    "GSL 2.8 multiroot broyden",
                ],
                &on,
                &[
                    format!("{}; Powell hybrid with differenced Jacobian (MINPACK hybrd)", gsl(3)),
                    format!("{}; Newton with differenced Jacobian", gsl(4)),
                    format!("{}; Broyden's method (no line search)", gsl(5)),
                ],
                1.0,
            );
        }
    }
}

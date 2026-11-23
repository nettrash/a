//! The minima of [`a::math::numerics::opt`]: the bracketed methods on functions whose minimum the
//! test knows and against SciPy's bounded Brent; the open methods on the Moré, Garbow and
//! Hillstrom problems against SciPy's BFGS, CG and Nelder-Mead; the least squares fits on data
//! generated from parameters the test chose and on five NIST StRD problems against their
//! certified values and MINPACK. Every defect of the audit `n_opt` has a test of its own here,
//! named after what it guards (`audit n_opt-N` in its comment); each fails on the code as it was
//! merged. Reference numbers come from `data/gen_opt.py`.

use a::math::matrix::{Matrix, Scalar};
use a::math::numerics::{
    MinResult, OptOptions, Real, bfgs, bfgs_with, conjugate_gradient, conjugate_gradient_with,
    gauss_newton, gauss_newton_with, golden_section, golden_section_with, levenberg_marquardt,
    levenberg_marquardt_with, minimize, minimize_with, nelder_mead, nelder_mead_with,
};
use std::cell::{Cell, RefCell};

fn col(v: &[f64]) -> Matrix<f64> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// (x² − 2)², whose minimum on [1, 2] is √2 — and whose curvature there is nought, which is what
/// makes the √ε tolerance of a minimisation visible rather than theoretical.
fn quartic_well(x: f64) -> f64 {
    (x * x - 2.0).powi(2)
}

/// x⁴ − 2x², whose minimum on [0, 2] is one, with value −1.
fn double_well(x: f64) -> f64 {
    x.powi(4) - 2.0 * x * x
}

/// |x − 2| + 10⁻³(x − 2)², whose minimum is at two and has no derivative there. A parabola cannot
/// be fitted through a kink, so this is where Brent's method has to fall back on the golden section.
fn kink(x: f64) -> f64 {
    (x - 2.0).abs() + 1e-3 * (x - 2.0).powi(2)
}

/// (x − 3)² + 1, the mildest thing worth asking a method to do.
fn basin(x: f64) -> f64 {
    (x - 3.0).powi(2) + 1.0
}

/// Rosenbrock's valley: the minimum is (1, 1) with value 0, and the valley floor is a tenth as
/// wide as it is long, which is what makes methods without a gradient work for it.
fn rosenbrock(x: &Matrix<f64>) -> f64 {
    let (a, b) = (x[(0, 0)], x[(1, 0)]);
    (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
}

/// Its gradient.
fn rosenbrock_gradient(x: &Matrix<f64>) -> Matrix<f64> {
    let (a, b) = (x[(0, 0)], x[(1, 0)]);
    Matrix::from_column_major(
        2,
        1,
        vec![
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a),
        ],
    )
}

fn start(a: f64, b: f64) -> Matrix<f64> {
    Matrix::from_column_major(2, 1, vec![a, b])
}

/// y = 2e^(−1.5t) + 0.05 sampled at five abscissae, fitted by y = ae^(−λt). The offset the data
/// carries is not something the model can represent, so the fit has a genuine minimum to find
/// rather than an exact answer to stumble onto — and the parameters it finds, [`DECAY`], are not
/// the ones that generated the table.
const FIT_T: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn fit_y() -> [f64; 5] {
    let mut y = [0.0; 5];
    for (i, t) in FIT_T.iter().enumerate() {
        y[i] = 2.0 * (-1.5 * *t).exp() + 0.05;
    }
    y
}

/// The cost of a set of parameters, half the sum of the squares of the residuals, which is what
/// the two least squares methods minimise.
fn decay_cost(p: &Matrix<f64>) -> f64 {
    decay_residual(p)
        .as_slice()
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        * 0.5
}

fn decay_residual(p: &Matrix<f64>) -> Matrix<f64> {
    let (a, lambda) = (p[(0, 0)], p[(1, 0)]);
    let y = fit_y();
    Matrix::from_column_major(
        5,
        1,
        FIT_T
            .iter()
            .zip(&y)
            .map(|(t, y)| a * (-lambda * *t).exp() - y)
            .collect(),
    )
}

/// Its Jacobian, ∂rᵢ/∂a = e^(−λtᵢ) and ∂rᵢ/∂λ = −a·tᵢ·e^(−λtᵢ).
fn decay_jacobian(p: &Matrix<f64>) -> Matrix<f64> {
    let (a, lambda) = (p[(0, 0)], p[(1, 0)]);
    Matrix::from_fn(5, 2, |i, k| match k {
        0 => (-lambda * FIT_T[i]).exp(),
        _ => -a * FIT_T[i] * (-lambda * FIT_T[i]).exp(),
    })
}

/// The decay of the crate's own documentation, y = 2e^(−1.5t) on five exact points: a fit of
/// zero residual at (2, 1.5).
fn exact_residual(p: &Matrix<f64>) -> Matrix<f64> {
    col(&FIT_T
        .iter()
        .map(|t| p[(0, 0)] * (-p[(1, 0)] * t).exp() - 2.0 * (-1.5 * t).exp())
        .collect::<Vec<_>>())
}

fn exact_jacobian(p: &Matrix<f64>) -> Matrix<f64> {
    Matrix::from_fn(5, 2, |i, k| match k {
        0 => (-p[(1, 0)] * FIT_T[i]).exp(),
        _ => -p[(0, 0)] * FIT_T[i] * (-p[(1, 0)] * FIT_T[i]).exp(),
    })
}

type Value = Box<dyn Fn(&[f64]) -> f64>;
type OneVariable = Box<dyn Fn(f64) -> f64>;
type Gradient = Box<dyn Fn(&[f64]) -> Vec<f64>>;

/// A problem of the battery of `data/gen_opt.py`, with every minimiser it may end at.
struct Problem {
    name: String,
    f: Value,
    g: Gradient,
    x0: Vec<f64>,
    minima: Vec<Vec<f64>>,
}

impl Problem {
    /// The distance, in the infinity norm, from `x` to the nearest known minimiser.
    fn distance(&self, x: &[f64]) -> f64 {
        self.minima
            .iter()
            .map(|m| {
                m.iter()
                    .zip(x)
                    .fold(0.0f64, |d, (a, b)| d.max((a - b).abs()))
            })
            .fold(f64::INFINITY, f64::min)
    }
}

fn rosen(x: &[f64]) -> f64 {
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}

fn rosen_gradient(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        let t = x[i + 1] - x[i] * x[i];
        g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * t;
    }
    g
}

/// Brown's badly scaled function (Moré, Garbow and Hillstrom, problem 4): minimum 0 at
/// (10⁶, 2·10⁻⁶).
fn brown(x: &[f64]) -> f64 {
    (x[0] - 1e6).powi(2) + (x[1] - 2e-6).powi(2) + (x[0] * x[1] - 2.0).powi(2)
}

fn brown_gradient(x: &[f64]) -> Vec<f64> {
    let t = x[0] * x[1] - 2.0;
    vec![
        2.0 * (x[0] - 1e6) + 2.0 * t * x[1],
        2.0 * (x[1] - 2e-6) + 2.0 * t * x[0],
    ]
}

/// ½(x − c)ᵀQDQ(x − c) with the reflection Q = I − (2/n)·11ᵀ and Dᵢ = cond^(i/(n − 1)),
/// cᵢ = sin(i + 1): a quadratic of condition `cond` whose axes are not the coordinate axes.
fn rotated_quadratic(n: usize, cond: f64) -> (Value, Gradient, Vec<f64>) {
    let d: Vec<f64> = (0..n)
        .map(|i| cond.powf(i as f64 / (n as f64 - 1.0)))
        .collect();
    let c: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let reflect = |v: &[f64]| -> Vec<f64> {
        let s: f64 = v.iter().sum::<f64>() * 2.0 / v.len() as f64;
        v.iter().map(|x| x - s).collect()
    };
    let (d1, c1, d2, c2) = (d.clone(), c.clone(), d, c.clone());
    (
        Box::new(move |x: &[f64]| {
            let y: Vec<f64> = x.iter().zip(&c1).map(|(a, b)| a - b).collect();
            let z = reflect(&y);
            0.5 * z.iter().zip(&d1).map(|(a, b)| a * a * b).sum::<f64>()
        }),
        Box::new(move |x: &[f64]| {
            let y: Vec<f64> = x.iter().zip(&c2).map(|(a, b)| a - b).collect();
            let z = reflect(&y);
            let w: Vec<f64> = z.iter().zip(&d2).map(|(a, b)| a * b).collect();
            reflect(&w)
        }),
        c,
    )
}

/// The problems of the battery, in the order of `data/gen_opt.py`.
fn battery() -> Vec<Problem> {
    let mut v = Vec::new();
    let sizes: &[usize] = if cfg!(miri) { &[2] } else { &[2, 10, 50] };
    for &n in sizes {
        // From n = 4 on the chained function has a second local minimum near (−1, 1, …, 1),
        // which SciPy's BFGS ends at from the usual start; the methods here reach (1, …, 1).
        v.push(Problem {
            name: format!("rosen{n}"),
            f: Box::new(rosen),
            g: Box::new(rosen_gradient),
            x0: (0..n)
                .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
                .collect(),
            minima: vec![vec![1.0; n]],
        });
    }
    v.push(Problem {
        name: "powell".into(),
        f: Box::new(|x: &[f64]| {
            let (a, b, c, d) = (x[0], x[1], x[2], x[3]);
            (a + 10.0 * b).powi(2)
                + 5.0 * (c - d).powi(2)
                + (b - 2.0 * c).powi(4)
                + 10.0 * (a - d).powi(4)
        }),
        g: Box::new(|x: &[f64]| {
            let (a, b, c, d) = (x[0], x[1], x[2], x[3]);
            vec![
                2.0 * (a + 10.0 * b) + 40.0 * (a - d).powi(3),
                20.0 * (a + 10.0 * b) + 4.0 * (b - 2.0 * c).powi(3),
                10.0 * (c - d) - 8.0 * (b - 2.0 * c).powi(3),
                -10.0 * (c - d) - 40.0 * (a - d).powi(3),
            ]
        }),
        x0: vec![3.0, -1.0, 0.0, 1.0],
        minima: vec![vec![0.0; 4]],
    });
    v.push(Problem {
        name: "beale".into(),
        f: Box::new(|x: &[f64]| {
            let (a, b) = (x[0], x[1]);
            (1.5 - a + a * b).powi(2)
                + (2.25 - a + a * b * b).powi(2)
                + (2.625 - a + a * b.powi(3)).powi(2)
        }),
        g: Box::new(|x: &[f64]| {
            let (a, b) = (x[0], x[1]);
            let (t1, t2, t3) = (
                1.5 - a + a * b,
                2.25 - a + a * b * b,
                2.625 - a + a * b.powi(3),
            );
            vec![
                2.0 * t1 * (b - 1.0) + 2.0 * t2 * (b * b - 1.0) + 2.0 * t3 * (b.powi(3) - 1.0),
                2.0 * t1 * a + 4.0 * t2 * a * b + 6.0 * t3 * a * b * b,
            ]
        }),
        x0: vec![1.0, 1.0],
        minima: vec![vec![3.0, 0.5]],
    });
    v.push(Problem {
        name: "wood".into(),
        f: Box::new(|x: &[f64]| {
            let (a, b, c, d) = (x[0], x[1], x[2], x[3]);
            100.0 * (a * a - b).powi(2)
                + (a - 1.0).powi(2)
                + (c - 1.0).powi(2)
                + 90.0 * (c * c - d).powi(2)
                + 10.1 * ((b - 1.0).powi(2) + (d - 1.0).powi(2))
                + 19.8 * (b - 1.0) * (d - 1.0)
        }),
        g: Box::new(|x: &[f64]| {
            let (a, b, c, d) = (x[0], x[1], x[2], x[3]);
            vec![
                400.0 * a * (a * a - b) + 2.0 * (a - 1.0),
                -200.0 * (a * a - b) + 20.2 * (b - 1.0) + 19.8 * (d - 1.0),
                2.0 * (c - 1.0) + 360.0 * c * (c * c - d),
                -180.0 * (c * c - d) + 20.2 * (d - 1.0) + 19.8 * (b - 1.0),
            ]
        }),
        x0: vec![-3.0, -1.0, -3.0, -1.0],
        minima: vec![vec![1.0; 4]],
    });
    for s in [[0.0, 0.0], [-1.0, -1.0], [-3.0, 3.0], [4.0, -2.0]] {
        v.push(Problem {
            name: format!("himmelblau({},{})", s[0], s[1]),
            f: Box::new(|x: &[f64]| {
                (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2)
            }),
            g: Box::new(|x: &[f64]| {
                let (t1, t2) = (x[0] * x[0] + x[1] - 11.0, x[0] + x[1] * x[1] - 7.0);
                vec![4.0 * x[0] * t1 + 2.0 * t2, 2.0 * t1 + 4.0 * x[1] * t2]
            }),
            x0: s.to_vec(),
            // Its four minima, all of value 0.
            minima: vec![
                vec![3.0, 2.0],
                vec![-2.805118086952745, 3.131312518250573],
                vec![-3.779310253377747, -3.28318599128617],
                vec![3.584428340330492, -1.848126526964404],
            ],
        });
    }
    v.push(Problem {
        name: "brown".into(),
        f: Box::new(brown),
        g: Box::new(brown_gradient),
        x0: vec![1.0, 1.0],
        minima: vec![vec![1e6, 2e-6]],
    });
    for (cond, label) in [(1e2, "1e2"), (1e4, "1e4"), (1e6, "1e6")] {
        let (f, g, c) = rotated_quadratic(10, cond);
        v.push(Problem {
            name: format!("quad10_{label}"),
            f,
            g,
            x0: vec![0.0; 10],
            minima: vec![c],
        });
    }
    v
}

/// A NIST problem: its model with the Jacobian, its data and its published values.
struct Nist {
    name: &'static str,
    x: &'static [f64],
    y: &'static [f64],
    starts: [&'static [f64]; 2],
    certified: &'static [f64],
    rss: f64,
}

impl Nist {
    /// The value of the model at `x`, and its derivatives with respect to the parameters `b`.
    fn model(&self, b: &[f64], x: f64) -> (f64, Vec<f64>) {
        match self.name {
            // Misra1a and BoxBOD: b₁(1 − e^(−b₂x)).
            "Misra1a" | "BoxBOD" => {
                let e = (-b[1] * x).exp();
                (b[0] * (1.0 - e), vec![1.0 - e, b[0] * x * e])
            }
            // Thurber: a rational function, cubic over cubic.
            "Thurber" => {
                let num = b[0] + b[1] * x + b[2] * x * x + b[3] * x.powi(3);
                let den = 1.0 + b[4] * x + b[5] * x * x + b[6] * x.powi(3);
                let d2 = den * den;
                (
                    num / den,
                    vec![
                        1.0 / den,
                        x / den,
                        x * x / den,
                        x.powi(3) / den,
                        -num * x / d2,
                        -num * x * x / d2,
                        -num * x.powi(3) / d2,
                    ],
                )
            }
            // MGH09, Kowalik and Osborne's enzyme reaction: b₁(x² + b₂x)/(x² + b₃x + b₄).
            "MGH09" => {
                let num = x * x + x * b[1];
                let den = x * x + x * b[2] + b[3];
                (
                    b[0] * num / den,
                    vec![
                        num / den,
                        b[0] * x / den,
                        -b[0] * num * x / (den * den),
                        -b[0] * num / (den * den),
                    ],
                )
            }
            // MGH17, Osborne's first problem: b₁ + b₂e^(−b₄x) + b₃e^(−b₅x).
            _ => {
                let (e4, e5) = ((-x * b[3]).exp(), (-x * b[4]).exp());
                (
                    b[0] + b[1] * e4 + b[2] * e5,
                    vec![1.0, e4, e5, -b[1] * x * e4, -b[2] * x * e5],
                )
            }
        }
    }

    fn residual(&self, b: &Matrix<f64>) -> Matrix<f64> {
        col(&self
            .x
            .iter()
            .zip(self.y)
            .map(|(x, y)| self.model(b.as_slice(), *x).0 - y)
            .collect::<Vec<_>>())
    }

    fn jacobian(&self, b: &Matrix<f64>) -> Matrix<f64> {
        let rows: Vec<Vec<f64>> = self
            .x
            .iter()
            .map(|x| self.model(b.as_slice(), *x).1)
            .collect();
        Matrix::from_fn(self.x.len(), self.certified.len(), |i, k| rows[i][k])
    }

    /// The number of significant digits of `b` that agree with the certified values, the log
    /// relative error of the NIST StRD documentation, at most 16.
    fn digits(&self, b: &[f64]) -> f64 {
        b.iter()
            .zip(self.certified)
            .map(|(b, c)| {
                let e = ((b - c) / c).abs();
                if e == 0.0 { 16.0 } else { -e.log10() }
            })
            .fold(16.0f64, f64::min)
            .max(0.0)
    }
}

fn nist() -> Vec<Nist> {
    vec![
        Nist {
            name: "Misra1a",
            x: MISRA1A_X,
            y: MISRA1A_Y,
            starts: MISRA1A_STARTS,
            certified: MISRA1A_CERTIFIED,
            rss: MISRA1A_RSS,
        },
        Nist {
            name: "BoxBOD",
            x: BOXBOD_X,
            y: BOXBOD_Y,
            starts: BOXBOD_STARTS,
            certified: BOXBOD_CERTIFIED,
            rss: BOXBOD_RSS,
        },
        Nist {
            name: "Thurber",
            x: THURBER_X,
            y: THURBER_Y,
            starts: THURBER_STARTS,
            certified: THURBER_CERTIFIED,
            rss: THURBER_RSS,
        },
        Nist {
            name: "MGH09",
            x: MGH09_X,
            y: MGH09_Y,
            starts: MGH09_STARTS,
            certified: MGH09_CERTIFIED,
            rss: MGH09_RSS,
        },
        Nist {
            name: "MGH17",
            x: MGH17_X,
            y: MGH17_Y,
            starts: MGH17_STARTS,
            certified: MGH17_CERTIFIED,
            rss: MGH17_RSS,
        },
    ]
}

const ONE_DIM: &[(&str, f64, usize)] = &[
    ("basin", 3.0, 6),
    ("double_well", 1.0000000092482957, 13),
    ("quartic_well", 1.4142135618116738, 11),
    ("exp_line", 0.6931471803520718, 11),
    ("kink", 2.0, 17),
    ("x_log_x", 0.367879440523813, 11),
    ("sine", 4.7123889799524274, 9),
    ("cosh", 1.0000000001210743, 11),
    ("far", 1000.0000000000002, 6),
    ("lorentz", 0.2999999997866555, 12),
    ("sixth", 4.525362757055905e-09, 24),
];
// Made by data/gen_opt.py: scipy.optimize.minimize from the same starts and tolerances:
// problem, method, success, iterations, x.
const BATTERY: &[(&str, &str, bool, usize, &[f64])] = &[
    (
        "rosen2",
        "BFGS",
        true,
        34,
        &[0.999999999999463, 0.9999999999988877],
    ),
    (
        "rosen2",
        "CG",
        true,
        37,
        &[1.000000000055709, 1.0000000001116387],
    ),
    (
        "rosen2",
        "Nelder-Mead",
        true,
        119,
        &[0.99999999628883, 0.9999999931892243],
    ),
    (
        "rosen10",
        "BFGS",
        false,
        74,
        &[
            -0.9932633732720647,
            0.996606040140601,
            0.998240611560392,
            0.9989884337596658,
            0.9992261536257202,
            0.9990736482745068,
            0.9984541774756436,
            0.9970562516339779,
            0.9941793746023369,
            0.9883926287709272,
        ],
    ),
    (
        "rosen10",
        "CG",
        true,
        384,
        &[
            0.9999999999788695,
            0.9999999999504723,
            0.9999999998958082,
            0.9999999997674959,
            0.9999999995262974,
            0.9999999990667625,
            0.9999999981356444,
            0.9999999962652502,
            0.9999999924852294,
            0.9999999849250191,
        ],
    ),
    (
        "rosen50",
        "BFGS",
        true,
        350,
        &[
            0.99999999999965,
            0.999999999999483,
            0.999999999999854,
            1.0000000000004607,
            1.0000000000006768,
            1.0000000000009281,
            1.0000000000003035,
            1.0000000000006495,
            0.999999999999578,
            1.000000000000081,
            0.9999999999999203,
            0.9999999999993567,
            0.9999999999985771,
            0.9999999999992586,
            0.9999999999998803,
            1.0000000000000375,
            1.0000000000001603,
            1.0000000000000857,
            0.9999999999999742,
            1.0000000000017346,
            1.0000000000025977,
            1.0000000000021037,
            1.0000000000006386,
            0.9999999999994602,
            1.0000000000015434,
            0.9999999999999489,
            1.000000000000733,
            1.000000000001205,
            0.9999999999993012,
            1.0000000000009948,
            1.0000000000001914,
            1.0000000000003766,
            1.0000000000003164,
            1.0000000000002274,
            1.0000000000011062,
            1.0000000000015934,
            0.9999999999996471,
            0.9999999999977705,
            0.9999999999980759,
            0.9999999999964905,
            0.9999999999993744,
            0.9999999999970173,
            0.9999999999956635,
            0.9999999999959086,
            0.9999999999953334,
            0.9999999999939667,
            0.9999999999898384,
            0.9999999999787553,
            0.9999999999601888,
            0.9999999999227716,
        ],
    ),
    (
        "rosen50",
        "CG",
        true,
        819,
        &[
            1.000000000010171,
            1.0000000000056535,
            1.000000000000029,
            1.0000000000064224,
            1.000000000006507,
            1.0000000000102554,
            1.0000000000117857,
            1.0000000000114744,
            1.0000000000144538,
            1.0000000000078617,
            1.0000000000006717,
            1.000000000001284,
            0.9999999999992513,
            1.000000000003112,
            0.9999999999925866,
            0.9999999999869549,
            0.9999999999859966,
            0.9999999999903726,
            0.9999999999990352,
            0.9999999999963236,
            0.999999999987007,
            0.9999999999853697,
            0.9999999999885684,
            0.9999999999993585,
            0.999999999991031,
            0.9999999999923953,
            0.9999999999975869,
            0.9999999999970978,
            0.9999999999964401,
            1.0000000000061644,
            1.0000000000100344,
            1.0000000000066072,
            1.0000000000007387,
            1.0000000000129163,
            1.0000000000100908,
            1.0000000000129492,
            1.000000000007829,
            0.9999999999897868,
            0.9999999999797705,
            0.999999999981434,
            0.9999999999696502,
            0.9999999999439328,
            0.9999999998914886,
            0.9999999997973431,
            0.999999999594727,
            0.9999999992184624,
            0.9999999984541256,
            0.99999999689588,
            0.9999999937783843,
            0.9999999875405714,
        ],
    ),
    (
        "powell",
        "BFGS",
        true,
        61,
        &[
            -8.638914541957837e-05,
            8.638887211050301e-06,
            -2.3579203397024293e-05,
            -2.3580282725792093e-05,
        ],
    ),
    (
        "powell",
        "CG",
        true,
        85,
        &[
            0.0007785967449072217,
            -7.785967516236486e-05,
            0.00010842754897648677,
            0.00010842770503996423,
        ],
    ),
    (
        "powell",
        "Nelder-Mead",
        true,
        387,
        &[
            -1.6244563295725728e-07,
            1.6244558133641774e-08,
            3.8800873828540207e-08,
            3.880086714791445e-08,
        ],
    ),
    (
        "beale",
        "BFGS",
        true,
        16,
        &[2.999999999839861, 0.4999999999675159],
    ),
    (
        "beale",
        "CG",
        true,
        27,
        &[3.0000000110824367, 0.5000000027265723],
    ),
    (
        "beale",
        "Nelder-Mead",
        true,
        67,
        &[3.000000020718082, 0.500000005546391],
    ),
    (
        "wood",
        "BFGS",
        true,
        90,
        &[
            0.9999999999924349,
            0.9999999999842799,
            1.0000000000071927,
            1.0000000000154776,
        ],
    ),
    (
        "wood",
        "CG",
        true,
        95,
        &[
            1.000000001736615,
            1.0000000034824146,
            0.999999997980042,
            0.9999999959663883,
        ],
    ),
    (
        "wood",
        "Nelder-Mead",
        true,
        563,
        &[
            0.9999999898916779,
            0.9999999801524531,
            1.0000000147369061,
            1.0000000299742062,
        ],
    ),
    (
        "himmelblau(0,0)",
        "BFGS",
        true,
        12,
        &[3.00000000000092, 1.9999999999969098],
    ),
    (
        "himmelblau(0,0)",
        "CG",
        true,
        11,
        &[2.999999999999945, 1.999999999999983],
    ),
    (
        "himmelblau(0,0)",
        "Nelder-Mead",
        true,
        60,
        &[3.0000000112531007, 2.0000000142171954],
    ),
    (
        "himmelblau(-1,-1)",
        "BFGS",
        true,
        11,
        &[-3.779310253358235, -3.283185991277552],
    ),
    (
        "himmelblau(-1,-1)",
        "CG",
        true,
        12,
        &[-3.7793102533748626, -3.2831859912871026],
    ),
    (
        "himmelblau(-1,-1)",
        "Nelder-Mead",
        true,
        62,
        &[-2.8051180889267266, 3.131312502111554],
    ),
    (
        "himmelblau(-3,3)",
        "BFGS",
        true,
        5,
        &[-2.8051180869711296, 3.1313125182748456],
    ),
    (
        "himmelblau(-3,3)",
        "CG",
        true,
        6,
        &[-2.8051180869692214, 3.1313125182385093],
    ),
    (
        "himmelblau(-3,3)",
        "Nelder-Mead",
        true,
        57,
        &[-2.805118107416531, 3.131312536762433],
    ),
    (
        "himmelblau(4,-2)",
        "BFGS",
        true,
        8,
        &[3.5844283403508723, -1.8481265268431124],
    ),
    (
        "himmelblau(4,-2)",
        "CG",
        true,
        7,
        &[3.5844283403304864, -1.8481265269644178],
    ),
    (
        "himmelblau(4,-2)",
        "Nelder-Mead",
        true,
        63,
        &[3.584428329432023, -1.848126528816272],
    ),
    (
        "brown",
        "BFGS",
        true,
        16,
        &[1000000.0, 2.0000000000000008e-06],
    ),
    (
        "brown",
        "CG",
        true,
        15,
        &[1000000.0, 2.0000000000000003e-06],
    ),
    (
        "brown",
        "Nelder-Mead",
        true,
        173,
        &[1000000.0000000133, 1.9999999907746553e-06],
    ),
    (
        "quad10_1e2",
        "BFGS",
        true,
        16,
        &[
            0.841470984808742,
            0.9092974268210321,
            0.14112000806018168,
            -0.756802495304587,
            -0.9589242746653824,
            -0.27941549819660066,
            0.656986598719594,
            0.9893582466237273,
            0.4121184852421298,
            -0.5440211108889563,
        ],
    ),
    (
        "quad10_1e2",
        "CG",
        true,
        98,
        &[
            0.8414709803501826,
            0.9092974274110756,
            0.14112000998905364,
            -0.7568024941570163,
            -0.9589242739040166,
            -0.2794154970808946,
            0.6569865992623906,
            0.9893582474560123,
            0.4121184858450842,
            -0.5440211102987077,
        ],
    ),
    (
        "quad10_1e4",
        "BFGS",
        true,
        19,
        &[
            0.8414709848079505,
            0.9092974268254534,
            0.14112000805993,
            -0.7568024953078674,
            -0.9589242746630882,
            -0.27941549819886663,
            0.6569865987188471,
            0.98935824662344,
            0.4121184852418147,
            -0.5440211108893129,
        ],
    ),
    (
        "quad10_1e4",
        "CG",
        true,
        651,
        &[
            0.8414709844688868,
            0.9092974295398993,
            0.14112000689014864,
            -0.7568024958645696,
            -0.9589242753905403,
            -0.27941549882092503,
            0.6569865980474633,
            0.9893582459604872,
            0.41211848456281336,
            -0.5440211115678844,
        ],
    ),
    (
        "quad10_1e6",
        "BFGS",
        true,
        15,
        &[
            0.8414709848112867,
            0.9092974268279659,
            0.14112000807744926,
            -0.7568024953358883,
            -0.9589242746585817,
            -0.27941549819543665,
            0.6569865987221516,
            0.9893582466267203,
            0.41211848524511363,
            -0.5440211108860047,
        ],
    ),
    (
        "quad10_1e6",
        "CG",
        true,
        3269,
        &[
            0.8414709745275308,
            0.9092974273463889,
            0.14112001130909094,
            -0.7568024920301424,
            -0.9589242714240063,
            -0.27941549496560697,
            0.6569866019561247,
            0.9893582498621085,
            0.41211848848040206,
            -0.544021107650736,
        ],
    ),
];
// Made by data/gen_opt.py: the NIST StRD problems (data, starts and certified values as
// NIST publishes them).
const MISRA1A_X: &[f64] = &[
    77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0,
];
const MISRA1A_Y: &[f64] = &[
    10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.4, 75.47, 81.78,
];
const MISRA1A_STARTS: [&[f64]; 2] = [&[500.0, 0.0001], &[250.0, 0.0005]];
const MISRA1A_CERTIFIED: &[f64] = &[238.94212918, 0.00055015643181];
const MISRA1A_RSS: f64 = 0.12455138894;
const BOXBOD_X: &[f64] = &[1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
const BOXBOD_Y: &[f64] = &[109.0, 149.0, 149.0, 191.0, 213.0, 224.0];
const BOXBOD_STARTS: [&[f64]; 2] = [&[1.0, 1.0], &[100.0, 0.75]];
const BOXBOD_CERTIFIED: &[f64] = &[213.80940889, 0.54723748542];
const BOXBOD_RSS: f64 = 1168.0088766;
const THURBER_X: &[f64] = &[
    -3.067, -2.981, -2.921, -2.912, -2.84, -2.797, -2.702, -2.699, -2.633, -2.481, -2.363, -2.322,
    -1.501, -1.46, -1.274, -1.212, -1.1, -1.046, -0.915, -0.714, -0.566, -0.545, -0.4, -0.309,
    -0.109, -0.103, 0.01, 0.119, 0.377, 0.79, 0.963, 1.006, 1.115, 1.572, 1.841, 2.047, 2.2,
];
const THURBER_Y: &[f64] = &[
    80.574, 84.248, 87.264, 87.195, 89.076, 89.608, 89.868, 90.101, 92.405, 95.854, 100.696,
    101.06, 401.672, 390.724, 567.534, 635.316, 733.054, 759.087, 894.206, 990.785, 1090.109,
    1080.914, 1122.643, 1178.351, 1260.531, 1273.514, 1288.339, 1327.543, 1353.863, 1414.509,
    1425.208, 1421.384, 1442.962, 1464.35, 1468.705, 1447.894, 1457.628,
];
const THURBER_STARTS: [&[f64]; 2] = [
    &[1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03],
    &[1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05],
];
const THURBER_CERTIFIED: &[f64] = &[
    1288.13968,
    1491.0792535,
    583.23836877,
    75.416644291,
    0.96629502864,
    0.39797285797,
    0.049727297349,
];
const THURBER_RSS: f64 = 5642.7082397;
const MGH09_X: &[f64] = &[
    4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625,
];
const MGH09_Y: &[f64] = &[
    0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
];
const MGH09_STARTS: [&[f64]; 2] = [&[25.0, 39.0, 41.5, 39.0], &[0.25, 0.39, 0.415, 0.39]];
const MGH09_CERTIFIED: &[f64] = &[0.19280693458, 0.19128232873, 0.12305650693, 0.13606233068];
const MGH09_RSS: f64 = 0.00030750560385;
const MGH17_X: &[f64] = &[
    0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
    150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0,
    280.0, 290.0, 300.0, 310.0, 320.0,
];
const MGH17_Y: &[f64] = &[
    0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.85, 0.818, 0.784, 0.751, 0.718, 0.685,
    0.658, 0.628, 0.603, 0.58, 0.558, 0.538, 0.522, 0.506, 0.49, 0.478, 0.467, 0.457, 0.448, 0.438,
    0.431, 0.424, 0.42, 0.414, 0.411, 0.406,
];
const MGH17_STARTS: [&[f64]; 2] = [
    &[50.0, 150.0, -100.0, 1.0, 2.0],
    &[0.5, 1.5, -1.0, 0.01, 0.02],
];
const MGH17_CERTIFIED: &[f64] = &[
    0.37541005211,
    1.9358469127,
    -1.4646871366,
    0.01286753464,
    0.022122699662,
];
const MGH17_RSS: f64 = 5.4648946975e-05;
// Made by data/gen_opt.py: least_squares(method="lm") (MINPACK) from each start:
// problem, start, success, evaluations, digits of the parameters that agree with the
// certified values.
const NIST_SCIPY: &[(&str, usize, bool, usize, f64)] = &[
    ("Misra1a", 1, true, 19, 9.78),
    ("Misra1a", 2, true, 5, 10.13),
    ("BoxBOD", 1, true, 10, 0.00),
    ("BoxBOD", 2, true, 9, 5.13),
    ("Thurber", 1, true, 34, 4.65),
    ("Thurber", 2, true, 19, 4.70),
    ("MGH09", 1, true, 500, 4.05),
    ("MGH09", 2, true, 18, 3.99),
    ("MGH17", 1, true, 594, 6.55),
    ("MGH17", 2, true, 18, 6.60),
];
// Made by data/gen_opt.py: the least squares fit of a·exp(−λt) to 2·exp(−1.5t) + 0.05,
// by mpmath at 50 digits.
const DECAY: [f64; 2] = [2.043356468452901, 1.405311987300545];

/// A deterministic stream of numbers in `[0, 1)`, for spreading starts over a region.
fn uniform(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------------------------
// One variable
// ---------------------------------------------------------------------------------------------

#[test]
fn the_bracketed_methods_find_the_minimum_of_a_unimodal_function() {
    // name, the function, the bracket and the minimiser inside it.
    type Case<'a> = (&'a str, fn(f64) -> f64, f64, f64, f64);
    let cases: &[Case] = &[
        ("(x²−2)²", quartic_well, 1.0, 2.0, core::f64::consts::SQRT_2),
        ("x⁴−2x²", double_well, 0.0, 2.0, 1.0),
        ("|x−2|+10⁻³(x−2)²", kink, 0.0, 4.0, 2.0),
        ("(x−3)²+1", basin, 0.0, 10.0, 3.0),
    ];
    let opts = OptOptions::new();
    for (name, f, a, b, want) in cases {
        let golden = golden_section_with(*f, *a, *b, &opts)
            .unwrap_or_else(|| panic!("{name}: golden gave up"));
        let brent = minimize(*f, *a, *b).unwrap_or_else(|| panic!("{name}: brent gave up"));
        for (method, got) in [("golden", golden.x), ("brent", brent.x)] {
            assert!(
                (got - want).abs() <= opts.tol_at(*want),
                "{name} by {method}: {got} against {want}"
            );
        }
        assert!(
            golden.converged,
            "{name}: the golden section did not converge"
        );
        assert!(brent.converged, "{name}: brent did not converge");
        // The parabola is what Brent has that the golden section has not: on everything smooth
        // it needs a third of the evaluations or fewer, and on the kink it falls back on the
        // golden section and needs about the same.
        let smooth = *name != "|x−2|+10⁻³(x−2)²";
        if smooth {
            assert!(
                3 * brent.evaluations <= golden.evaluations + 3,
                "{name}: brent needed {} evaluations against golden's {}",
                brent.evaluations,
                golden.evaluations
            );
        }
        assert!(brent.evaluations < golden.evaluations, "{name}");
    }
}

#[test]
fn minimize_is_brents_method_and_needs_what_scipys_fminbound_needs() {
    // audit n_opt-4: the parabolic step was unreachable, and every function took 36
    // evaluations. SciPy's bounded method is Brent's localmin at the same tolerance near |x| = 1,
    // so the counts agree (they were measured equal on all eleven).
    let functions: Vec<(&str, OneVariable, f64, f64)> = vec![
        ("basin", Box::new(basin), 0.0, 10.0),
        ("double_well", Box::new(double_well), 0.0, 2.0),
        ("quartic_well", Box::new(quartic_well), 1.0, 2.0),
        ("exp_line", Box::new(|x: f64| x.exp() - 2.0 * x), 0.0, 2.0),
        ("kink", Box::new(kink), 0.0, 4.0),
        ("x_log_x", Box::new(|x: f64| x * x.ln()), 0.01, 1.0),
        ("sine", Box::new(|x: f64| x.sin()), 2.0, 6.0),
        ("cosh", Box::new(|x: f64| (x - 1.0).cosh()), -10.0, 10.0),
        ("far", Box::new(|x: f64| (x - 1e3).powi(2)), 0.0, 1e4),
        (
            "lorentz",
            Box::new(|x: f64| -1.0 / (1.0 + (x - 0.3).powi(2))),
            -5.0,
            5.0,
        ),
        ("sixth", Box::new(|x: f64| x.powi(6)), -1.0, 2.0),
    ];
    let opts = OptOptions::new();
    for (name, f, a, b) in &functions {
        let (_, x_scipy, evaluations_scipy) = *ONE_DIM.iter().find(|t| t.0 == *name).unwrap();
        let r = minimize(f, *a, *b).unwrap();
        assert!(r.converged, "{name}: {r:?}");
        // The seven functions of plain arithmetic are evaluated alike everywhere, and the counts
        // must be fminbound's exactly (a parabolic step accepted on a looser rule than Brent's
        // changes that of the double well); the four of exp, ln, sin and cosh go through the
        // platform's mathematical library, whose last digit can move one comparison.
        let library = ["exp_line", "x_log_x", "sine", "cosh"].contains(name);
        assert!(
            r.evaluations.abs_diff(evaluations_scipy) <= usize::from(library),
            "{name}: {} evaluations against fminbound's {evaluations_scipy}",
            r.evaluations
        );
        assert!(
            (r.x - x_scipy).abs() <= 2.0 * opts.tol_at(x_scipy),
            "{name}: {} against fminbound's {x_scipy}",
            r.x
        );
    }
    // The first parabola through three points of a parabola is the parabola itself.
    let r = minimize(basin, 0.0, 10.0).unwrap();
    assert!(r.evaluations <= 8 && r.x == 3.0, "{r:?}");
}

#[test]
fn the_bracketed_methods_meet_the_documented_tolerance_on_the_point() {
    // audit n_opt-5: the tolerance was rtol·|b − a|, so a wide bracket gave inaccurate answers
    // marked converged (7.33 for 3 in f32) and a narrow one could never close.
    let f32_opts = OptOptions::<f32>::new();
    let parabola32 = |x: f32| (x - 3.0).powi(2);
    let brent = minimize(parabola32, -1e4, 1e4).unwrap();
    let golden = golden_section_with(parabola32, -1e4, 1e4, &f32_opts).unwrap();
    for r in [brent, golden] {
        assert!(
            r.converged && (r.x - 3.0).abs() <= f32_opts.tol_at(3.0),
            "{r:?}"
        );
    }
    let opts = OptOptions::<f64>::new();
    let parabola = |x: f64| (x - 3.0).powi(2);
    for (a, b) in [(0.0, 1e6), (-1e4, 1e4), (2.9, 3.1)] {
        let brent = minimize(parabola, a, b).unwrap();
        let golden = golden_section_with(parabola, a, b, &opts).unwrap();
        for r in [brent, golden] {
            assert!(
                r.converged && (r.x - 3.0).abs() <= opts.tol_at(3.0),
                "[{a}, {b}]: {r:?}"
            );
        }
    }
    let flat = |x: f64| (x - 3.0).atan().powi(2);
    let r = minimize(flat, -1e12, 1e12).unwrap();
    assert!(
        r.converged && (r.x - 3.0).abs() <= opts.tol_at(3.0),
        "{r:?}"
    );
    // A bracket already narrower than the tolerance there is accepted at once.
    for (a, b) in [
        (1e6, 1e6 + 1e-3),
        (1.0 - 1e-9, 1.0 + 1e-9),
        (1e3, 1e3 + 1e-6),
    ] {
        let mid = 0.5 * (a + b);
        let golden = golden_section_with(|x: f64| (x - mid).powi(2), a, b, &opts).unwrap();
        let brent = minimize(|x: f64| (x - mid).powi(2), a, b).unwrap();
        assert!(
            golden.converged && golden.evaluations <= 3,
            "[{a}, {b}]: {golden:?}"
        );
        assert!(
            brent.converged && brent.evaluations <= 3,
            "[{a}, {b}]: {brent:?}"
        );
    }
}

#[test]
fn minimize_counts_every_evaluation() {
    // audit n_opt-6: f was called three times at the first point and counted once.
    for budget in [0, 1, 2, 5, 1000] {
        let opts = OptOptions::new().with_max_iter(budget);
        for method in ["brent", "golden"] {
            let calls = Cell::new(0usize);
            let f = |x: f64| {
                calls.set(calls.get() + 1);
                (x - 0.3).powi(2)
            };
            let r = if method == "brent" {
                minimize_with(f, 0.0, 1.0, &opts)
            } else {
                golden_section_with(f, 0.0, 1.0, &opts)
            }
            .unwrap();
            assert_eq!(
                r.evaluations,
                calls.get(),
                "{method} with a budget of {budget}"
            );
            assert!(r.evaluations <= budget + 2, "{method}: {r:?}");
        }
    }
}

#[test]
fn a_bracket_partly_undefined_is_searched_where_the_function_is_finite() {
    // audit n_opt-7: None as soon as the first probe, or both, fell where f is not finite.
    let opts = OptOptions::new();
    let x_log_x = |x: f64| x * x.ln();
    let inverse_e = (-1.0f64).exp();
    let r = minimize(x_log_x, -3.0, 1.0).unwrap();
    assert!(
        r.converged && (r.x - inverse_e).abs() <= opts.tol_at(inverse_e),
        "{r:?}"
    );
    let x = golden_section(x_log_x, -3.0, 1.0).unwrap();
    assert!((x - inverse_e).abs() <= opts.tol_at(inverse_e), "{x}");
    let left = |x: f64| if x > 3.0 { f64::NAN } else { (x - 1.0).powi(2) };
    let right = |x: f64| if x < 2.0 { f64::NAN } else { (x - 3.0).powi(2) };
    for (name, f, a, b, want) in [
        (
            "undefined above 3",
            &left as &dyn Fn(f64) -> f64,
            0.0,
            10.0,
            1.0,
        ),
        ("undefined above 3, narrow", &left, 0.0, 4.0, 1.0),
        ("undefined below 2", &right, -10.0, 10.0, 3.0),
    ] {
        let r = minimize(f, a, b).unwrap_or_else(|| panic!("{name}: brent refused"));
        assert!(
            r.converged && (r.x - want).abs() <= opts.tol_at(want),
            "{name}: {r:?}"
        );
        let x = golden_section(f, a, b).unwrap_or_else(|| panic!("{name}: golden refused"));
        assert!((x - want).abs() <= opts.tol_at(want), "{name}: {x}");
    }
}

#[test]
fn a_bracket_the_method_cannot_use_is_refused() {
    assert!(
        minimize(|x: f64| x * x, 1.0, 1.0).is_none(),
        "an empty bracket"
    );
    assert!(
        golden_section(|x: f64| x * x, 1.0, -1.0).is_none(),
        "an inverted bracket"
    );
    assert!(
        minimize(|x: f64| x * x, f64::NAN, 1.0).is_none(),
        "a bracket with a NaN end"
    );
    assert!(
        minimize_with(
            |x: f64| x * x,
            0.0,
            1.0,
            &OptOptions::new().with_rtol(f64::NAN)
        )
        .is_none(),
        "a tolerance that is not a number"
    );
}

#[test]
fn a_bracket_whose_width_overflows_is_refused() {
    // audit n_opt-10: b − a overflowed and the point at infinity came back as a converged
    // minimum of atan(x)².
    let atan2 = |x: f64| x.atan().powi(2);
    assert!(golden_section_with(atan2, -1e308, 1e308, &OptOptions::new()).is_none());
    assert!(minimize(atan2, -1e308, 1e308).is_none());
    assert!(golden_section(atan2, -1e308, 1e308).is_none());
}

#[test]
fn a_long_golden_search_keeps_its_ratio_and_its_word() {
    // A bracket of 2·10³⁰⁰ needs some 1,470 golden steps to close about zero. Placed from the
    // ends of the bracket, the two points drifted off the golden ratio within a hundred steps
    // and the search stopped on a bracket 10²⁷⁶ wide, calling it converged. The flag-less form
    // does not hand back a point it has not finished with.
    let abs = |x: f64| x.abs();
    assert!(golden_section(abs, -1e300, 1e300).is_none());
    let r = golden_section_with(abs, -1e300, 1e300, &OptOptions::new()).unwrap();
    assert!(!r.converged && r.value == r.x.abs(), "{r:?}");
    let r =
        golden_section_with(abs, -1e300, 1e300, &OptOptions::new().with_max_iter(2000)).unwrap();
    assert!(
        r.converged && r.x.abs() < 1e-7 && r.value == r.x.abs(),
        "{r:?}"
    );
    assert!(r.evaluations > 1400 && r.evaluations < 1500, "{r:?}");
}

// ---------------------------------------------------------------------------------------------
// Nelder and Mead
// ---------------------------------------------------------------------------------------------

#[test]
fn the_simplex_descends_rosenbrocks_valley() {
    for (a, b, step) in [
        (-1.2, 1.0, 1.0),
        (2.0, 2.0, 0.5),
        (-3.0, -2.0, 2.0),
        (0.9, 1.1, 0.1),
    ] {
        let r = nelder_mead(rosenbrock, &start(a, b), step)
            .unwrap_or_else(|| panic!("the simplex gave up from ({a}, {b}) at step {step}"));
        assert!(r.converged, "from ({a}, {b}) at step {step}: {r:?}");
        assert!(
            (r.x[(0, 0)] - 1.0).abs() < 1e-6 && (r.x[(1, 0)] - 1.0).abs() < 1e-6,
            "from ({a}, {b}) at step {step}: {:?}",
            r.x
        );
        assert!(r.value < 1e-12, "from ({a}, {b}): value {}", r.value);
        assert!(r.iterations > 10, "from ({a}, {b}): too few steps");
    }
    // A quadratic it settles on exactly, which is the check that the moves of the simplex are
    // what they are meant to be.
    let quadratic = |x: &Matrix<f64>| {
        let (a, b) = (x[(0, 0)], x[(1, 0)]);
        a * a + 2.0 * b * b - 2.0 * a - 8.0 * b + 9.0
    };
    let r = nelder_mead(quadratic, &start(3.0, -4.0), 1.0).unwrap();
    assert!(
        (r.x[(0, 0)] - 1.0).abs() < 1e-6 && (r.x[(1, 0)] - 2.0).abs() < 1e-6,
        "{:?}",
        r.x
    );
    assert!(
        r.value < 1e-12,
        "the minimum of a sum of squares: {}",
        r.value
    );
}

#[test]
fn the_simplex_contracts_inside_and_reaches_rosenbrocks_minimum_from_every_start() {
    // audit n_opt-2: without the inside contraction the simplex was shrunk where it should have
    // been reshaped, and closed away from (1, 1) from 5 to 8% of starts, converged = true. SciPy's
    // simplex from (−2, 2.5) reaches (1, 1) in 113 iterations.
    let r = nelder_mead(rosenbrock, &start(-2.0, 2.5), 1.0).unwrap();
    assert!(
        r.converged && (r.x[(0, 0)] - 1.0).abs() < 1e-6 && (r.x[(1, 0)] - 1.0).abs() < 1e-6,
        "{r:?}"
    );
    assert!(
        r.iterations.abs_diff(113) <= 5,
        "{} iterations",
        r.iterations
    );
    let side = if cfg!(miri) { 3 } else { 13 };
    let mut wrong = Vec::new();
    for step in [1.0, 0.25] {
        for i in 0..side {
            for j in 0..side {
                let (a, b) = (
                    -3.0 + 6.0 * i as f64 / (side - 1) as f64,
                    -3.0 + 6.0 * j as f64 / (side - 1) as f64,
                );
                let r = nelder_mead(rosenbrock, &start(a, b), step).unwrap();
                let off = (r.x[(0, 0)] - 1.0).abs().max((r.x[(1, 0)] - 1.0).abs());
                if r.converged && off > 1e-5 {
                    wrong.push(format!("({a}, {b}) step {step}: {:?}", r.x.as_slice()));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "converged away from (1, 1):\n{}",
        wrong.join("\n")
    );
}

#[test]
fn the_simplex_matches_scipys_from_the_same_simplex() {
    for p in battery() {
        let Some(&(_, _, success, iterations, x)) = BATTERY
            .iter()
            .find(|t| t.0 == p.name && t.1 == "Nelder-Mead")
        else {
            continue;
        };
        assert!(success, "{}", p.name);
        let f = |x: &Matrix<f64>| (p.f)(x.as_slice());
        let r = nelder_mead(f, &col(&p.x0), 1.0).unwrap();
        assert!(r.converged, "{}: {r:?}", p.name);
        let off =
            r.x.as_slice()
                .iter()
                .zip(x)
                .fold(0.0f64, |d, (a, b)| d.max((a - b).abs() / b.abs().max(1.0)));
        assert!(
            off <= 1e-6,
            "{}: {:?} against SciPy's {x:?}",
            p.name,
            r.x.as_slice()
        );
        // SciPy's xatol is absolute, which at x₀ = 10⁶ asks for far more than this method's
        // relative tolerance: Brown's function is left out of the count.
        let (ours, theirs) = (r.iterations as f64, iterations as f64);
        assert!(
            p.name == "brown" || (ours - theirs).abs() <= 0.1 * theirs + 5.0,
            "{}: {} iterations against SciPy's {iterations}",
            p.name,
            r.iterations
        );
    }
}

#[test]
fn a_starting_simplex_below_the_tolerance_still_moves() {
    // audit n_opt-8: a first step smaller than the tolerance at x0 made the start a converged
    // minimum at iteration 1.
    let bowl = |x: &Matrix<f64>| x[(0, 0)].powi(2) + x[(1, 0)].powi(2);
    let r = nelder_mead(bowl, &start(3e7, 3e7), 1.0).unwrap();
    assert!(r.iterations > 1, "{r:?}");
    assert!(!r.converged || r.value < 1e-10, "{r:?}");
    let r = nelder_mead(rosenbrock, &start(-1.2, 1.0), 1e-9).unwrap();
    assert!(r.iterations > 1, "{r:?}");
    assert!(!r.converged || (r.x[(0, 0)] - 1.0).abs() < 1e-5, "{r:?}");
}

// ---------------------------------------------------------------------------------------------
// The gradient methods
// ---------------------------------------------------------------------------------------------

#[test]
fn the_gradient_methods_descend_rosenbrocks_valley() {
    let x0 = start(-1.2, 1.0);
    let simplex = nelder_mead(rosenbrock, &x0, 1.0).unwrap();
    let c = conjugate_gradient(rosenbrock, rosenbrock_gradient, &x0).unwrap();
    let q = bfgs(rosenbrock, rosenbrock_gradient, &x0).unwrap();
    assert!(c.converged && q.converged, "{c:?} {q:?}");
    for (name, r) in [("cg", &c), ("bfgs", &q)] {
        assert!(
            (r.x[(0, 0)] - 1.0).abs() < 1e-7 && (r.x[(1, 0)] - 1.0).abs() < 1e-7,
            "{name}: {:?}",
            r.x
        );
        assert!(norm_inf(rosenbrock_gradient(&r.x).as_slice()) <= OptOptions::<f64>::new().gtol);
        // The gradient is worth asking for: far fewer iterations than the simplex.
        assert!(
            r.iterations * 2 < simplex.iterations,
            "{name}: {}",
            r.iterations
        );
    }
}

#[test]
fn the_two_gradient_methods_agree() {
    for (a, b) in [(3.0, -4.0), (-2.0, 0.5), (1.0, 1.0)] {
        let x0 = start(a, b);
        let c = conjugate_gradient(rosenbrock, rosenbrock_gradient, &x0).unwrap();
        let q = bfgs(rosenbrock, rosenbrock_gradient, &x0).unwrap();
        assert!(c.converged && q.converged, "from ({a}, {b}): {c:?} {q:?}");
        for i in 0..2 {
            assert!(
                (c.x[(i, 0)] - q.x[(i, 0)]).abs() < 1e-7,
                "from ({a}, {b}): {:?} {:?}",
                c.x,
                q.x
            );
        }
    }
}

#[test]
fn an_unbounded_function_reports_no_convergence() {
    // A plane has no minimum: the methods walk off the end of it and say so rather than claim to
    // have arrived.
    let plane = |x: &Matrix<f64>| x.as_slice().iter().sum::<f64>();
    let constant = |_: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![1.0, 1.0]);
    let x0 = start(0.0, 0.0);
    let q = bfgs(plane, constant, &x0).unwrap();
    let c = conjugate_gradient(plane, constant, &x0).unwrap();
    assert!(!q.converged, "bfgs claimed to converge: {q:?}");
    assert!(
        !c.converged,
        "the conjugate gradient claimed to converge: {c:?}"
    );
    assert!(q.value < plane(&x0) && c.value < plane(&x0));
}

#[test]
fn a_short_step_is_not_taken_for_convergence_by_the_gradient_methods() {
    // audit n_opt-1: one step shorter than the tolerance on the point was taken for convergence,
    // whatever the gradient: Brown's function "converged" at f = 3.2·10⁵ by the conjugate
    // gradient, Rosenbrock's in f32 5 to 10% off its minimum.
    let f = |x: &Matrix<f64>| brown(x.as_slice());
    let g = |x: &Matrix<f64>| col(&brown_gradient(x.as_slice()));
    let gtol = OptOptions::<f64>::new().gtol;
    for (name, r) in [
        ("cg", conjugate_gradient(f, g, &start(1.0, 1.0)).unwrap()),
        ("bfgs", bfgs(f, g, &start(1.0, 1.0)).unwrap()),
    ] {
        assert!(r.converged, "{name}: {r:?}");
        assert!(norm_inf(g(&r.x).as_slice()) <= gtol, "{name}: {r:?}");
        assert!(
            (r.x[(0, 0)] - 1e6).abs() < 1e-3 && (r.x[(1, 0)] - 2e-6).abs() < 1e-12,
            "{name}: {r:?}"
        );
    }
    let f32_rosenbrock = |x: &Matrix<f32>| {
        (1.0 - x[(0, 0)]).powi(2) + 100.0 * (x[(1, 0)] - x[(0, 0)].powi(2)).powi(2)
    };
    let f32_gradient = |x: &Matrix<f32>| {
        let (a, b) = (x[(0, 0)], x[(1, 0)]);
        Matrix::from_column_major(
            2,
            1,
            vec![
                -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
                200.0 * (b - a * a),
            ],
        )
    };
    let x0 = Matrix::from_column_major(2, 1, vec![-1.2f32, 1.0]);
    let gtol32 = OptOptions::<f32>::new().gtol;
    for (name, r) in [
        (
            "cg",
            conjugate_gradient(f32_rosenbrock, f32_gradient, &x0).unwrap(),
        ),
        ("bfgs", bfgs(f32_rosenbrock, f32_gradient, &x0).unwrap()),
    ] {
        let g = f32_gradient(&r.x);
        assert!(
            !r.converged || norm_inf(&[g[(0, 0)] as f64, g[(1, 0)] as f64]) <= gtol32 as f64,
            "{name}: {r:?}"
        );
        assert!(
            (r.x[(0, 0)] - 1.0).abs() < 1e-3 && (r.x[(1, 0)] - 1.0).abs() < 2e-3,
            "{name}: {r:?}"
        );
    }
    // The steep valley (1 − x)² + k(y − x²)², from scattered starts.
    let mut seed = 7u64;
    let starts = if cfg!(miri) { 2 } else { 20 };
    for k in [1e3, 1e4, 1e6] {
        let f = move |x: &Matrix<f64>| {
            (1.0 - x[(0, 0)]).powi(2) + k * (x[(1, 0)] - x[(0, 0)].powi(2)).powi(2)
        };
        let g = move |x: &Matrix<f64>| {
            let (a, b) = (x[(0, 0)], x[(1, 0)]);
            col(&[
                -2.0 * (1.0 - a) - 4.0 * k * a * (b - a * a),
                2.0 * k * (b - a * a),
            ])
        };
        for _ in 0..starts {
            let x0 = start(
                -3.0 + 6.0 * uniform(&mut seed),
                -3.0 + 6.0 * uniform(&mut seed),
            );
            for (name, r) in [
                ("cg", conjugate_gradient(f, g, &x0).unwrap()),
                ("bfgs", bfgs(f, g, &x0).unwrap()),
            ] {
                if r.converged {
                    assert!(
                        norm_inf(g(&r.x).as_slice()) <= gtol,
                        "{name}, k = {k}: {r:?}"
                    );
                    assert!((r.x[(0, 0)] - 1.0).abs() < 1e-6, "{name}, k = {k}: {r:?}");
                }
            }
        }
    }
}

#[test]
fn the_gradient_methods_reach_the_minima_scipy_reaches() {
    // Each converged run satisfies the gradient test at the point it returns, and is at a known
    // minimiser: to the accuracy a gradient of √ε gives there, which is the cube root of it on
    // Powell's singular function, whose Hessian vanishes at the minimum.
    let gtol = OptOptions::<f64>::new().gtol;
    for p in battery() {
        let f = |x: &Matrix<f64>| (p.f)(x.as_slice());
        let g = |x: &Matrix<f64>| col(&(p.g)(x.as_slice()));
        let tol = match p.name.as_str() {
            "powell" => 2e-3,
            "brown" => 1e-3,
            _ => 1e-6,
        };
        for method in ["BFGS", "CG"] {
            let opts = OptOptions::new().with_max_iter(5000);
            let r = if method == "BFGS" {
                bfgs_with(f, g, &col(&p.x0), &opts)
            } else {
                conjugate_gradient_with(f, g, &col(&p.x0), &opts)
            }
            .unwrap();
            let name = &p.name;
            assert!(r.converged, "{name} by {method}: {r:?}");
            assert!(
                norm_inf(&(p.g)(r.x.as_slice())) <= gtol,
                "{name} by {method}"
            );
            assert!(
                p.distance(r.x.as_slice()) <= tol,
                "{name} by {method}: {:?}",
                r.x.as_slice()
            );
            // Not many more iterations than SciPy needs where it succeeds.
            let &(_, _, success, iterations, _) = BATTERY
                .iter()
                .find(|t| t.0 == *name && t.1 == method)
                .unwrap();
            if success {
                assert!(
                    r.iterations <= 2 * iterations + 10,
                    "{name} by {method}: {} iterations against SciPy's {iterations}",
                    r.iterations
                );
            }
        }
    }
}

#[test]
fn the_line_search_steps_satisfy_the_strong_wolfe_conditions() {
    // Each run with max_iter = k stops at the k-th iterate, so the steps can be recovered and
    // the conditions checked on s = x_{k+1} − x_k: f_{k+1} ≤ f_k + 10⁻⁴·g_kᵀs and
    // |g_{k+1}ᵀs| ≤ η·|g_kᵀs|, η = 0.9 for BFGS and 0.1 for the conjugate gradient.
    let f = |x: &Matrix<f64>| rosen(x.as_slice());
    let g = |x: &Matrix<f64>| col(&rosen_gradient(x.as_slice()));
    let x0 = col(&[-1.2, 1.0, -1.2, 1.0, -1.2, 1.0]);
    let steps = if cfg!(miri) { 5 } else { 40 };
    for (method, eta) in [("bfgs", 0.9), ("cg", 0.1)] {
        let run = |k: usize| {
            let opts = OptOptions::new().with_max_iter(k);
            if method == "bfgs" {
                bfgs_with(f, g, &x0, &opts)
            } else {
                conjugate_gradient_with(f, g, &x0, &opts)
            }
            .unwrap()
        };
        let mut before = run(0);
        for k in 1..=steps {
            let after = run(k);
            if after.iterations < k {
                break;
            }
            let s: Vec<f64> = after
                .x
                .as_slice()
                .iter()
                .zip(before.x.as_slice())
                .map(|(a, b)| a - b)
                .collect();
            let g0: f64 = g(&before.x)
                .as_slice()
                .iter()
                .zip(&s)
                .map(|(g, s)| g * s)
                .sum();
            let g1: f64 = g(&after.x)
                .as_slice()
                .iter()
                .zip(&s)
                .map(|(g, s)| g * s)
                .sum();
            assert!(
                after.value <= before.value + 1e-4 * g0,
                "{method}, step {k}: no sufficient decrease"
            );
            assert!(
                g1.abs() <= eta * g0.abs(),
                "{method}, step {k}: {g1:e} against {g0:e}"
            );
            before = after;
        }
    }
}

#[test]
fn a_restart_of_the_conjugate_gradient_survives_a_wild_first_trial() {
    // review round 1: at a restart the first trial of the search down the gradient is the step
    // that would repeat the last decrease, 2(f − f_before)/φ′(0), which after a step that gained
    // much where the gradient is now small is enormous (10¹⁴); the search gave up long before it
    // came down to the scale of the function, and the method stopped with converged false at
    // |∇f| up to 2.5·10⁻⁷, ten times gtol, on 15 of these 1500 random quadratics of 2 to 8
    // variables and condition 1 to 10³ (the reviewer's stream). BFGS never failed on them.
    let mut seed = 0x853C_49E6_748F_EA9Bu64;
    let mut uniform = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let opts = OptOptions::new().with_max_iter(20000);
    let problems = if cfg!(miri) { 12 } else { 1500 };
    let mut stops = Vec::new();
    for trial in 0..problems {
        let n = [2usize, 3, 5, 8][trial % 4];
        // An orthonormal basis by Gram and Schmidt, and the eigenvalues spread evenly in the
        // logarithm from 1 to the condition.
        let mut basis: Vec<Vec<f64>> = Vec::new();
        while basis.len() < n {
            let mut v: Vec<f64> = (0..n).map(|_| uniform() - 0.5).collect();
            for w in &basis {
                let along: f64 = v.iter().zip(w).map(|(a, b)| a * b).sum();
                for (a, b) in v.iter_mut().zip(w) {
                    *a -= along * b;
                }
            }
            let length = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            if length > 1e-3 {
                basis.push(v.iter().map(|a| a / length).collect());
            }
        }
        let condition = [1.0, 10.0, 1e3][trial % 3];
        let eigenvalues: Vec<f64> = (0..n)
            .map(|i| f64::powf(condition, i as f64 / (n - 1).max(1) as f64))
            .collect();
        let minimiser: Vec<f64> = (0..n).map(|_| 4.0 * uniform() - 2.0).collect();
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (0..n)
                            .map(|k| basis[k][i] * eigenvalues[k] * basis[k][j])
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let gradient = |x: &Matrix<f64>| {
            Matrix::from_fn(n, 1, |i, _| {
                (0..n).map(|j| a[i][j] * (x[(j, 0)] - minimiser[j])).sum()
            })
        };
        let f = |x: &Matrix<f64>| {
            let g = gradient(x);
            0.5 * (0..n)
                .map(|i| (x[(i, 0)] - minimiser[i]) * g[(i, 0)])
                .sum::<f64>()
        };
        let x0 = Matrix::from_fn(n, 1, |_, _| 6.0 * uniform() - 3.0);
        // The first step of the simplex in the reviewer's probe, drawn to keep the stream.
        let _ = uniform();
        for (method, r) in [
            (
                "cg",
                conjugate_gradient_with(f, gradient, &x0, &opts).unwrap(),
            ),
            ("bfgs", bfgs_with(f, gradient, &x0, &opts).unwrap()),
        ] {
            if !r.converged {
                stops.push(format!(
                    "{method}, trial {trial} (n = {n}, condition {condition}): |∇f| = {:e} after {} steps",
                    norm_inf(gradient(&r.x).as_slice()),
                    r.iterations
                ));
            }
        }
    }
    assert!(stops.is_empty(), "{}", stops.join("\n"));
}

#[test]
fn the_conjugate_gradient_restarts_down_the_gradient_every_n_steps() {
    // review round 1: nothing failed with the restart every n steps taken out. The steps are
    // recovered from the runs with max_iter = 1, 2, …; a step is a restart when it points along
    // −∇f at the point it left, and any n + 1 steps in a row hold one. The chained Rosenbrock
    // function of four variables keeps β positive, so only the rule makes the restarts here.
    let n = 4;
    let f = |x: &Matrix<f64>| rosen(x.as_slice());
    let g = |x: &Matrix<f64>| col(&rosen_gradient(x.as_slice()));
    let x0 = col(&[-1.2, 1.0, -1.2, 1.0]);
    let steps = if cfg!(miri) { 6 } else { 40 };
    let mut points = vec![x0.clone()];
    for k in 1..=steps {
        let r = conjugate_gradient_with(f, g, &x0, &OptOptions::new().with_max_iter(k)).unwrap();
        if r.iterations < k {
            break;
        }
        points.push(r.x);
    }
    assert!(points.len() > 2 * n + 2, "{} steps", points.len() - 1);
    let restart: Vec<bool> = points
        .windows(2)
        .map(|pair| {
            let step: Vec<f64> = pair[1]
                .as_slice()
                .iter()
                .zip(pair[0].as_slice())
                .map(|(a, b)| a - b)
                .collect();
            let down: Vec<f64> = g(&pair[0]).as_slice().iter().map(|v| -v).collect();
            let length = |v: &[f64]| v.iter().map(|v| v * v).sum::<f64>().sqrt();
            let along: f64 = step.iter().zip(&down).map(|(a, b)| a * b).sum();
            along / length(&step) / length(&down) > 1.0 - 1e-10
        })
        .collect();
    assert!(restart[0], "the first step is not down the gradient");
    for (k, window) in restart.windows(n + 1).enumerate() {
        assert!(
            window.contains(&true),
            "no restart in steps {} to {}: {restart:?}",
            k + 1,
            k + n + 1
        );
    }
    assert!(
        restart.iter().filter(|r| !**r).count() >= restart.len() / 2,
        "most steps are conjugate ones: {restart:?}"
    );
}

/// Runs the conjugate gradient, or BFGS, on `f` and `g` from `x0` under `opts`, recording the
/// points `f` is evaluated at: the result, the evaluations made after the point it returns was
/// first evaluated, and the evaluations in all.
fn evaluations_after_the_point<T: Real + Scalar>(
    f: &dyn Fn(&Matrix<T>) -> T,
    g: &dyn Fn(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
    quasi_newton: bool,
) -> (MinResult<T>, usize, usize) {
    let seen = RefCell::new(Vec::<Vec<T>>::new());
    let counted = |x: &Matrix<T>| {
        seen.borrow_mut().push(x.as_slice().to_vec());
        f(x)
    };
    let r = if quasi_newton {
        bfgs_with(counted, g, x0, opts)
    } else {
        conjugate_gradient_with(counted, g, x0, opts)
    }
    .unwrap();
    let seen = seen.into_inner();
    let first = seen
        .iter()
        .position(|p| p.as_slice() == r.x.as_slice())
        .unwrap();
    (r, seen.len() - 1 - first, seen.len())
}

#[test]
fn the_gradient_methods_stop_once_no_step_lowers_the_function() {
    // review P2: at the limit of the arithmetic the decrease condition of the line search holds
    // in rounding for a step that does not move the point, and both methods took such a step for
    // a step. On Rosenbrock's function with gtol = 0 the conjugate gradient had its last point
    // after 39 steps and 109 evaluations and went on to 1,000 "steps" and 38,549 evaluations, 40
    // a step; BFGS on the chained function of 30 variables to 31,973; in f32, on badly
    // conditioned quadratics, every run that could not meet gtol used all of its 5,000 steps and
    // up to 200,000 evaluations. Once neither the direction nor the gradient leads lower, both
    // must stop: after the point they return was first evaluated, no more evaluations than the
    // rest of that search and two more searches, and every step counted a step that moved.
    let most_after = 3 * 40;
    let off = OptOptions::<f64>::new().with_gtol(0.0);
    let (r, after, calls) = evaluations_after_the_point(
        &rosenbrock,
        &rosenbrock_gradient,
        &start(-1.2, 1.0),
        &off,
        false,
    );
    assert!(
        !r.converged && after <= most_after && calls <= 250,
        "Rosenbrock, the conjugate gradient: {after} of {calls} evaluations after the point, {r:?}"
    );
    // The last step counted moved the point.
    let before = conjugate_gradient_with(
        rosenbrock,
        rosenbrock_gradient,
        &start(-1.2, 1.0),
        &off.with_max_iter(r.iterations - 1),
    )
    .unwrap();
    assert!(
        before.x.as_slice() != r.x.as_slice(),
        "{before:?} and {r:?}"
    );
    let chained = |x: &Matrix<f64>| rosen(x.as_slice());
    let chained_gradient = |x: &Matrix<f64>| col(&rosen_gradient(x.as_slice()));
    for (n, quasi_newton, most) in [(10, false, 500), (30, true, 500)] {
        let x0 = col(&(0..n)
            .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
            .collect::<Vec<_>>());
        let (r, after, calls) =
            evaluations_after_the_point(&chained, &chained_gradient, &x0, &off, quasi_newton);
        assert!(
            !r.converged && after <= most_after && calls <= most,
            "chained Rosenbrock of {n}, BFGS {quasi_newton}: {after} of {calls} evaluations after the point, {r:?}"
        );
    }

    // f32: (x − x*)ᵀA(x − x*)/2 with A = H·diag(1, √(3·10⁵), 3·10⁵)·H, H the reflection in
    // (1, −2, 3)ᵀ, evaluated in f64 and rounded, which f32 cannot resolve to the default gtol.
    let u = [1.0, -2.0, 3.0];
    let reflect = |w: &[f64]| -> Vec<f64> {
        let p = 2.0 * (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / 14.0;
        w.iter().zip(&u).map(|(a, b)| a - p * b).collect()
    };
    let d = [1.0, 3e5f64.sqrt(), 3e5];
    let minimiser = [0.3, -0.7, 1.1];
    let shifted = |x: &Matrix<f32>| -> Vec<f64> {
        x.as_slice()
            .iter()
            .zip(&minimiser)
            .map(|(a, b)| *a as f64 - b)
            .collect()
    };
    let f = |x: &Matrix<f32>| {
        let hz = reflect(&shifted(x));
        (0.5 * hz.iter().zip(&d).map(|(a, b)| b * a * a).sum::<f64>()) as f32
    };
    let g = |x: &Matrix<f32>| {
        let hz = reflect(&shifted(x));
        let dhz: Vec<f64> = hz.iter().zip(&d).map(|(a, b)| a * b).collect();
        Matrix::from_column_major(3, 1, reflect(&dhz).into_iter().map(|v| v as f32).collect())
    };
    let x0 = Matrix::from_column_major(3, 1, vec![2.0f32, 1.0, -1.0]);
    let opts = OptOptions::<f32>::new().with_max_iter(5000);
    for quasi_newton in [false, true] {
        let (r, after, calls) = evaluations_after_the_point(&f, &g, &x0, &opts, quasi_newton);
        let largest = g(&r.x)
            .as_slice()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            r.converged == (largest <= opts.gtol) && after <= most_after && calls <= 500,
            "f32 quadratic, BFGS {quasi_newton}: {after} of {calls} evaluations after the point, |∇f| {largest:e}, {r:?}"
        );
    }
}

#[test]
fn a_start_at_the_minimum_and_a_single_variable() {
    let at = start(1.0, 1.0);
    for (name, r) in [
        (
            "cg",
            conjugate_gradient(rosenbrock, rosenbrock_gradient, &at).unwrap(),
        ),
        ("bfgs", bfgs(rosenbrock, rosenbrock_gradient, &at).unwrap()),
    ] {
        assert!(
            r.converged && r.iterations == 0 && r.x.as_slice() == [1.0, 1.0],
            "{name}: {r:?}"
        );
    }
    let r = nelder_mead(rosenbrock, &at, 0.5).unwrap();
    assert!(r.converged && r.value < 1e-14, "{r:?}");
    let fit = levenberg_marquardt(exact_residual, exact_jacobian, &start(2.0, 1.5)).unwrap();
    assert!(
        fit.converged && fit.iterations == 0 && fit.cost == 0.0,
        "{fit:?}"
    );
    let fit = gauss_newton(exact_residual, exact_jacobian, &start(2.0, 1.5)).unwrap();
    assert!(fit.converged && fit.iterations == 0, "{fit:?}");

    // (x − 2)⁴ + (x − 2)² + 3, of one variable.
    let f = |x: &Matrix<f64>| (x[(0, 0)] - 2.0).powi(4) + (x[(0, 0)] - 2.0).powi(2) + 3.0;
    let g = |x: &Matrix<f64>| col(&[4.0 * (x[(0, 0)] - 2.0).powi(3) + 2.0 * (x[(0, 0)] - 2.0)]);
    let x0 = col(&[-1.0]);
    for (name, r) in [
        ("cg", conjugate_gradient(f, g, &x0).unwrap()),
        ("bfgs", bfgs(f, g, &x0).unwrap()),
        ("simplex", nelder_mead(f, &x0, 1.0).unwrap()),
    ] {
        assert!(
            r.converged && (r.x[(0, 0)] - 2.0).abs() < 1e-7,
            "{name}: {r:?}"
        );
    }
}

#[test]
fn values_that_are_not_finite_are_stepped_around() {
    // x − ln x is not finite for x ≤ 0, where a long step lands; its minimum is 1, at x = 1.
    let f = |x: &Matrix<f64>| x[(0, 0)] - x[(0, 0)].ln();
    let g = |x: &Matrix<f64>| col(&[1.0 - 1.0 / x[(0, 0)]]);
    for x0 in [10.0, 0.01, 300.0] {
        for (name, r) in [
            ("cg", conjugate_gradient(f, g, &col(&[x0])).unwrap()),
            ("bfgs", bfgs(f, g, &col(&[x0])).unwrap()),
            ("simplex", nelder_mead(f, &col(&[x0]), 5.0).unwrap()),
        ] {
            assert!(
                r.converged && (r.x[(0, 0)] - 1.0).abs() < 1e-6,
                "{name} from {x0}: {r:?}"
            );
        }
    }
}

#[test]
fn a_start_the_method_cannot_use_is_refused() {
    let quadratic = |x: &Matrix<f64>| x.as_slice().iter().map(|v| v * v).sum::<f64>();
    let gradient = |x: &Matrix<f64>| x.clone();
    assert!(
        bfgs(quadratic, gradient, &start(f64::NAN, 0.0)).is_none(),
        "a NaN start"
    );
    assert!(
        bfgs(
            quadratic,
            gradient,
            &Matrix::from_column_major(0, 1, vec![])
        )
        .is_none(),
        "no variables"
    );
    assert!(
        nelder_mead(quadratic, &start(1.0, 1.0), 0.0).is_none(),
        "a simplex with no size"
    );
    assert!(
        nelder_mead(quadratic, &start(1.0, 1.0), f64::NAN).is_none(),
        "a simplex of no size"
    );
    assert!(
        nelder_mead(
            |x: &Matrix<f64>| 1.0 / x[(0, 0)] - f64::INFINITY,
            &start(1.0, 1.0),
            1.0
        )
        .is_none(),
        "f not finite at the start"
    );
    assert!(
        conjugate_gradient_with(
            quadratic,
            gradient,
            &start(1.0, 1.0),
            &OptOptions::new().with_gtol(-1.0)
        )
        .is_none(),
        "a negative tolerance"
    );
    assert!(
        gauss_newton(
            decay_residual,
            decay_jacobian,
            &Matrix::from_column_major(2, 1, vec![f64::INFINITY, 1.0])
        )
        .is_none(),
        "a start that is not finite"
    );
}

#[test]
#[should_panic(expected = "the starting point is not a column")]
fn a_start_that_is_not_a_column_panics() {
    let quadratic = |x: &Matrix<f64>| x.as_slice().iter().map(|v| v * v).sum::<f64>();
    let gradient = |x: &Matrix<f64>| x.clone();
    let _ = conjugate_gradient(quadratic, gradient, &Matrix::from_rows(&[[1.0, 2.0]]));
}

#[test]
#[should_panic(expected = "the gradient is not a column")]
fn a_gradient_of_the_wrong_length_panics() {
    let quadratic = |x: &Matrix<f64>| x.as_slice().iter().map(|v| v * v).sum::<f64>();
    let _ = bfgs(quadratic, |_: &Matrix<f64>| col(&[1.0]), &start(1.0, 1.0));
}

#[test]
fn running_out_of_budget_reports_the_best_point_rather_than_convergence() {
    let x0 = start(-1.2, 1.0);
    let opts = OptOptions::new().with_max_iter(1);
    for (name, r) in [
        (
            "bfgs",
            bfgs_with(rosenbrock, rosenbrock_gradient, &x0, &opts).unwrap(),
        ),
        (
            "cg",
            conjugate_gradient_with(rosenbrock, rosenbrock_gradient, &x0, &opts).unwrap(),
        ),
        (
            "simplex",
            nelder_mead_with(rosenbrock, &x0, 1.0, &opts).unwrap(),
        ),
    ] {
        assert!(!r.converged, "{name} claimed to converge in one iteration");
        assert_eq!(r.iterations, 1, "{name} took more than its budget");
        // A gradient step always descends; one move of the simplex may only reshape it, here
        // by an inside contraction, and keep its best corner.
        if name == "simplex" {
            assert!(r.value <= rosenbrock(&x0), "{name} went up");
        } else {
            assert!(r.value < rosenbrock(&x0), "{name} did not descend");
        }
    }
    let fit =
        levenberg_marquardt_with(decay_residual, decay_jacobian, &start(0.5, 3.0), &opts).unwrap();
    assert!(!fit.converged && fit.iterations == 1);
}

#[test]
fn the_options_move_when_the_method_stops() {
    let x0 = start(-1.2, 1.0);
    let tight = bfgs(rosenbrock, rosenbrock_gradient, &x0).unwrap();
    let loose = bfgs_with(
        rosenbrock,
        rosenbrock_gradient,
        &x0,
        &OptOptions::new().with_gtol(1e-2),
    )
    .unwrap();
    assert!(tight.converged && loose.converged);
    assert!(
        loose.iterations < tight.iterations,
        "{loose:?} against {tight:?}"
    );
    assert!(norm_inf(rosenbrock_gradient(&loose.x).as_slice()) <= 1e-2);
    for i in 0..2 {
        assert!(
            (loose.x[(i, 0)] - tight.x[(i, 0)]).abs() < 1e-2,
            "{:?} against {:?}",
            loose.x,
            tight.x
        );
    }
    // The tolerances on the point and the value are the simplex's.
    let coarse = nelder_mead_with(
        rosenbrock,
        &x0,
        1.0,
        &OptOptions::new().with_rtol(1e-3).with_ftol(1e-3),
    )
    .unwrap();
    let fine = nelder_mead(rosenbrock, &x0, 1.0).unwrap();
    assert!(
        coarse.converged && coarse.iterations < fine.iterations,
        "{coarse:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Least squares
// ---------------------------------------------------------------------------------------------

#[test]
fn least_squares_fits_a_line_exactly() {
    // Four points on y = 1 + 2t: linear in the parameters, so one step of either method lands.
    let t = [0.0, 1.0, 2.0, 3.0];
    let y = [1.0, 3.0, 5.0, 7.0];
    let residual = move |p: &Matrix<f64>| {
        let (a, b) = (p[(0, 0)], p[(1, 0)]);
        col(&t
            .iter()
            .zip(&y)
            .map(|(t, yi)| a + b * t - yi)
            .collect::<Vec<_>>())
    };
    let jacobian =
        |_: &Matrix<f64>| Matrix::from_rows(&[&[1.0, 0.0], &[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
    let x0 = start(0.0, 0.0);
    for (name, fit) in [
        ("gauss–newton", gauss_newton(residual, jacobian, &x0)),
        (
            "levenberg–marquardt",
            levenberg_marquardt(residual, jacobian, &x0),
        ),
    ] {
        let fit = fit.unwrap_or_else(|| panic!("{name} gave up on a line"));
        assert!(
            (fit.x[(0, 0)] - 1.0).abs() < 1e-12 && (fit.x[(1, 0)] - 2.0).abs() < 1e-12,
            "{name}: {:?}",
            fit.x
        );
        assert!(
            fit.cost < 1e-24,
            "{name}: the fit is not exact: {}",
            fit.cost
        );
        assert!(fit.converged && fit.iterations <= 2, "{name}: {fit:?}");
    }
}

#[test]
fn the_damped_fit_finishes_where_the_undamped_one_stalls() {
    // From a start far out a whole Gauss and Newton step overshoots and the method has nothing
    // left to try, while the damped one shortens its steps until they pay again. The answer is
    // the 50-digit fit of `DECAY`, to the accuracy the default tolerances ask for — that of
    // MINPACK from the same starts, which was measured within 10⁻⁷ of it as well.
    for (a, lambda) in [
        (0.5, 3.0),
        (100.0, 5.0),
        (0.02, 8.0),
        (1.0, 50.0),
        (-10.0, 3.0),
    ] {
        let x0 = start(a, lambda);
        let damped = levenberg_marquardt(decay_residual, decay_jacobian, &x0)
            .unwrap_or_else(|| panic!("the damped fit gave up from ({a}, {lambda})"));
        assert!(damped.converged, "from ({a}, {lambda}): {damped:?}");
        for (i, want) in DECAY.iter().enumerate() {
            assert!(
                (damped.x[(i, 0)] - want).abs() < 2e-7,
                "from ({a}, {lambda}): {:?}",
                damped.x
            );
        }
        let undamped = gauss_newton(decay_residual, decay_jacobian, &x0).unwrap();
        assert!(
            !undamped.converged,
            "the undamped fit from ({a}, {lambda}): {undamped:?}"
        );
    }
    // Where the start is near enough, both agree, and the cost reported is that of the point.
    let x0 = start(1.0, 1.0);
    let damped = levenberg_marquardt(decay_residual, decay_jacobian, &x0).unwrap();
    let undamped = gauss_newton(decay_residual, decay_jacobian, &x0).unwrap();
    assert!(damped.converged && undamped.converged);
    for (i, want) in DECAY.iter().enumerate() {
        assert!(
            (damped.x[(i, 0)] - undamped.x[(i, 0)]).abs() < 1e-7,
            "{:?} against {:?}",
            damped.x,
            undamped.x
        );
        assert!((undamped.x[(i, 0)] - want).abs() < 2e-7);
    }
    assert!((decay_cost(&damped.x) - damped.cost).abs() <= 1e-15);
    // With the relative tolerance off and the gradient test tight it lands on the 50-digit
    // answer, to 2·10⁻¹⁶ as measured.
    let tight = OptOptions::new().with_rtol(0.0).with_gtol(1e-14);
    let damped =
        levenberg_marquardt_with(decay_residual, decay_jacobian, &start(0.5, 3.0), &tight).unwrap();
    assert!(damped.converged);
    for (i, want) in DECAY.iter().enumerate() {
        assert!((damped.x[(i, 0)] - want).abs() < 1e-13, "{:?}", damped.x);
    }
}

#[test]
fn levenberg_marquardt_does_not_stop_on_a_damped_step() {
    // audit n_opt-3: after rejected steps the damping is large and the next step tiny, and the
    // step and gain tests took it for convergence: the crate's own example "converged" from
    // (0.01, 1) at λ = 58, and NIST MGH17 at its own start. Over the audit's grid of starts every
    // run that converges must be at (2, 1.5).
    let mut wrong = Vec::new();
    let (mut converged, mut total) = (0, 0);
    let amplitudes = [
        -100.0, -10.0, -1.0, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1e3, 1e4, 1e6,
    ];
    let rates = [
        -10.0, -3.0, -1.0, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 20.0, 50.0,
    ];
    let (amplitudes, rates) = if cfg!(miri) {
        (&amplitudes[3..5], &rates[6..8])
    } else {
        (&amplitudes[..], &rates[..])
    };
    for &a in amplitudes {
        for &lambda in rates {
            let fit =
                levenberg_marquardt(exact_residual, exact_jacobian, &start(a, lambda)).unwrap();
            total += 1;
            if fit.converged {
                converged += 1;
                if (fit.x[(0, 0)] - 2.0).abs() > 1e-8 || (fit.x[(1, 0)] - 1.5).abs() > 1e-8 {
                    wrong.push(format!("({a}, {lambda}): {:?}", fit.x.as_slice()));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "converged away from (2, 1.5):\n{}",
        wrong.join("\n")
    );
    assert!(
        converged * 10 >= total * 9,
        "{converged} of {total} converged"
    );
    for (a, lambda) in [(0.01, 1.0), (0.01, 0.5), (1.0, 50.0)] {
        let fit = levenberg_marquardt(exact_residual, exact_jacobian, &start(a, lambda)).unwrap();
        assert!(
            fit.converged && (fit.x[(1, 0)] - 1.5).abs() < 1e-8,
            "({a}, {lambda}): {fit:?}"
        );
    }
    // Gauss and Newton from a start where every step overshoots.
    let fit = gauss_newton(exact_residual, exact_jacobian, &start(2.0, -20.0)).unwrap();
    assert!(!fit.converged, "{fit:?}");
}

/// The largest cosine between the residual at `x` and a column of the Jacobian there, the
/// scale-free gradient of the fits (zero for a zero residual).
fn largest_cosine(
    residual: impl Fn(&Matrix<f64>) -> Matrix<f64>,
    jacobian: impl Fn(&Matrix<f64>) -> Matrix<f64>,
    x: &Matrix<f64>,
) -> f64 {
    let (r, j) = (residual(x), jacobian(x));
    let length = |v: &[f64]| v.iter().map(|v| v * v).sum::<f64>().sqrt();
    let r_norm = length(r.as_slice());
    if r_norm == 0.0 {
        return 0.0;
    }
    (0..j.cols()).fold(0.0f64, |most, k| {
        let c = j.column(k);
        let along: f64 = c.iter().zip(r.as_slice()).map(|(a, b)| a * b).sum();
        if length(c) == 0.0 {
            most
        } else {
            most.max((along / length(c) / r_norm).abs())
        }
    })
}

// NIST StRD MGH10, y = b₁·exp(b₂/(x + b₃)): data and certified values as NIST publishes them
// (also printed by data/gen_opt.py).
const MGH10_X: &[f64] = &[
    50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0,
    125.0,
];
const MGH10_Y: &[f64] = &[
    34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0, 11540.0, 9744.0, 8261.0, 7030.0, 6005.0,
    5147.0, 4427.0, 3820.0, 3307.0, 2872.0,
];
const MGH10_CERTIFIED: &[f64] = &[5.6096364710e-03, 6.1813463463e+03, 3.4522363462e+02];
const MGH10_RSS: f64 = 8.7945855171e+01;

fn mgh10_residual(b: &Matrix<f64>) -> Matrix<f64> {
    Matrix::from_fn(16, 1, |i, _| {
        b[(0, 0)] * (b[(1, 0)] / (MGH10_X[i] + b[(2, 0)])).exp() - MGH10_Y[i]
    })
}

fn mgh10_jacobian(b: &Matrix<f64>) -> Matrix<f64> {
    Matrix::from_fn(16, 3, |i, k| {
        let d = MGH10_X[i] + b[(2, 0)];
        let e = (b[(1, 0)] / d).exp();
        match k {
            0 => e,
            1 => b[(0, 0)] * e / d,
            _ => -b[(0, 0)] * b[(1, 0)] * e / (d * d),
        }
    })
}

#[test]
fn a_step_short_only_beside_another_parameter_is_not_taken_for_convergence() {
    // review B1 (the Gauss–Newton case of audit n_opt-3): the step test measured the step in one
    // norm over all the parameters, each scaled by the largest length its column of the Jacobian
    // had had. A parameter whose column had once been enormous then hid a step that changed
    // another parameter wholly — a from −8.9·10⁻¹⁶ to 3.9·10⁻³¹ beside a rate of 20 — and both
    // fits stopped "converged" at costs of up to 10⁶⁶ with a cosine of 1. MINPACK reports success
    // from every one of these starts too, and SciPy's TRF from all but one (data/gen_opt.py, B1);
    // these fits must say converged only at the minimum.
    let both = |residual: &dyn Fn(&Matrix<f64>) -> Matrix<f64>,
                jacobian: &dyn Fn(&Matrix<f64>) -> Matrix<f64>,
                x0: &Matrix<f64>| {
        [
            (
                "gauss_newton",
                gauss_newton(residual, jacobian, x0).unwrap(),
            ),
            (
                "levenberg_marquardt",
                levenberg_marquardt(residual, jacobian, x0).unwrap(),
            ),
        ]
    };
    // a·e^(bt) through five exact points of a = 2, b = 0.5, from (2, 20): the audit's own case.
    let t = [1.0, 2.0, 3.0, 4.0, 5.0];
    let residual = |p: &Matrix<f64>| {
        col(&t
            .iter()
            .map(|t| p[(0, 0)] * (p[(1, 0)] * t).exp() - 2.0 * (0.5 * t).exp())
            .collect::<Vec<_>>())
    };
    let jacobian = |p: &Matrix<f64>| {
        Matrix::from_fn(5, 2, |i, k| match k {
            0 => (p[(1, 0)] * t[i]).exp(),
            _ => p[(0, 0)] * t[i] * (p[(1, 0)] * t[i]).exp(),
        })
    };
    for (name, fit) in both(&residual, &jacobian, &start(2.0, 20.0)) {
        let at_minimum = (fit.x[(0, 0)] - 2.0).abs() < 1e-8 && (fit.x[(1, 0)] - 0.5).abs() < 1e-8;
        assert!(!fit.converged || at_minimum, "a·e^(bt), {name}: {fit:?}");
    }
    // NIST MGH10 from the certified values perturbed by up to 90%, the audit's sampling.
    let x0 = col(&[0.005610260537259397, 10820.234225449876, 36.87926726109717]);
    for (name, fit) in both(&mgh10_residual, &mgh10_jacobian, &x0) {
        let at_minimum = (2.0 * fit.cost / MGH10_RSS - 1.0).abs() < 1e-6;
        assert!(!fit.converged || at_minimum, "MGH10, {name}: {fit:?}");
        if fit.converged {
            for (b, c) in fit.x.as_slice().iter().zip(MGH10_CERTIFIED) {
                assert!(((b - c) / c).abs() < 1e-6, "MGH10, {name}: {fit:?}");
            }
        }
    }
    // y = 2e^(−0.5t) + 0.3 at 25 exact points of [0, 8], by a·e^(−bt) + c from starts whose rate
    // has the wrong sign.
    let t: Vec<f64> = (0..25).map(|i| 8.0 * i as f64 / 24.0).collect();
    let residual = |p: &Matrix<f64>| {
        Matrix::from_fn(25, 1, |i, _| {
            p[(0, 0)] * (-p[(1, 0)] * t[i]).exp() + p[(2, 0)] - 2.0 * (-0.5 * t[i]).exp() - 0.3
        })
    };
    let jacobian = |p: &Matrix<f64>| {
        Matrix::from_fn(25, 3, |i, k| {
            let e = (-p[(1, 0)] * t[i]).exp();
            match k {
                0 => e,
                1 => -p[(0, 0)] * t[i] * e,
                _ => 1.0,
            }
        })
    };
    for s in [[20.0, -17.0, -0.3], [1.0, -10.0, 1.0], [4.0, -6.0, 0.0]] {
        for (name, fit) in both(&residual, &jacobian, &col(&s)) {
            assert!(!fit.converged || fit.cost < 1e-20, "{s:?}, {name}: {fit:?}");
        }
    }
    // From a start near the answer the same fit converges to it.
    for (name, fit) in both(&residual, &jacobian, &col(&[1.0, 1.0, 1.0])) {
        if name == "levenberg_marquardt" {
            assert!(fit.converged && fit.cost < 1e-20, "{name}: {fit:?}");
            assert!((fit.x[(1, 0)] - 0.5).abs() < 1e-9, "{name}: {fit:?}");
        }
    }
}

#[test]
fn a_damped_step_held_back_by_a_length_its_column_has_lost_proves_nothing() {
    // review B1, the floor of the damping: where the columns of the Jacobian are dependent the
    // damping cannot fall below √ε, measured with D, the largest length each column has had. Two
    // decays whose rates had run off to 449 and 116 left the columns a fraction of the lengths
    // they had, the floor held the step back from the least squares solution of the linearised
    // problem, and its small gain was taken for convergence at a cost of 19.8 with a cosine of
    // 0.65. (Trial 2414 of the reviewer's random scan: exact data of a·e^(−bt) + c·e^(−dt).)
    let truth = [
        3.3629044365050693,
        0.495173899992321,
        4.312162345615635,
        3.7908170744932357,
    ];
    let t: Vec<f64> = (0..25).map(|i| 8.0 * i as f64 / 24.0).collect();
    let model = |p: &[f64], t: f64| p[0] * (-p[1] * t).exp() + p[2] * (-p[3] * t).exp();
    let y: Vec<f64> = t.iter().map(|t| model(&truth, *t)).collect();
    let residual =
        |p: &Matrix<f64>| Matrix::from_fn(25, 1, |i, _| model(p.as_slice(), t[i]) - y[i]);
    let jacobian = |p: &Matrix<f64>| {
        Matrix::from_fn(25, 4, |i, k| {
            let (e1, e2) = ((-p[(1, 0)] * t[i]).exp(), (-p[(3, 0)] * t[i]).exp());
            [e1, -p[(0, 0)] * t[i] * e1, e2, -p[(2, 0)] * t[i] * e2][k]
        })
    };
    let x0 = col(&[
        -1.4493310824640377,
        2.1560596255788878,
        -1.5944351265084602,
        5.859652983636399,
    ]);
    let fit = levenberg_marquardt(residual, jacobian, &x0).unwrap();
    assert!(
        !fit.converged || fit.cost < 1e-20,
        "converged at a cosine of {}: {fit:?}",
        largest_cosine(residual, jacobian, &fit.x)
    );
}

// Made by data/gen_opt.py: the local minimum of a·t/(b + t) + c through 30 exact points of
// (3, 1.2, 0.4) on [0, 8] on the side b < −t₂₁ of the pole at t₂₁ = 8·21/29 (MINPACK, every
// tolerance 10⁻¹⁵): the cost and the point.
const POLE_MINIMUM: (f64, [f64; 3]) = (
    5.979482007287647,
    [0.003612261664732894, -5.749663149566571, 2.4395971576869866],
);

/// The residual of `a·t/(b + t) + c` against the data `y` at the abscissae `t`.
fn hyperbola_residual<T: Real + Scalar>(t: &[T], y: &[T], p: &Matrix<T>) -> Matrix<T> {
    Matrix::from_fn(t.len(), 1, |i, _| {
        p[(0, 0)] * t[i] / (p[(1, 0)] + t[i]) + p[(2, 0)] - y[i]
    })
}

/// The Jacobian of `a·t/(b + t) + c` at the abscissae `t`.
fn hyperbola_jacobian<T: Real + Scalar>(t: &[T], p: &Matrix<T>) -> Matrix<T> {
    Matrix::from_fn(t.len(), 3, |i, k| {
        let d = p[(1, 0)] + t[i];
        match k {
            0 => t[i] / d,
            1 => -p[(0, 0)] * t[i] / (d * d),
            _ => T::ONE,
        }
    })
}

#[test]
fn a_short_step_away_from_a_pole_of_the_model_proves_nothing() {
    // review P1: y = a·t/(b + t) + c with b next to −tₖ. The Gauss and Newton step moves b by its
    // distance from the pole, which is short against b, and the step test took it for
    // convergence; but the step doubles the distance, quarters the cost and cuts the column of b
    // to a quarter, and the next step does the same. Within an ulp of the pole the residual is
    // huge and below the resolution of the residual test all the same, rounding b moving the
    // model as much. Both fits said converged at costs of 10¹³ and 10³² (MINPACK reports success
    // from both f64 starts too, data/gen_opt.py), and in f32 at costs of 180 to 27,000 from 8 of
    // the second review's 6,000 random starts. They must say converged only at the local minimum
    // on that side of the pole, of cost 5.98.
    let t: Vec<f64> = (0..30).map(|i| 8.0 * i as f64 / 29.0).collect();
    let y: Vec<f64> = t.iter().map(|t| 3.0 * t / (1.2 + t) + 0.4).collect();
    let residual = |p: &Matrix<f64>| hyperbola_residual(&t, &y, p);
    let jacobian = |p: &Matrix<f64>| hyperbola_jacobian(&t, p);
    let (least, minimiser) = POLE_MINIMUM;
    let t21 = t[21];
    // 10⁻⁹ of b from the pole, one unit in the last place, and 64.
    for b0 in [
        -t21 * (1.0 - 1e-9),
        -t21 + t21 * f64::EPSILON,
        -t21 + 64.0 * t21 * f64::EPSILON,
    ] {
        let x0 = col(&[3.0, b0, 0.4]);
        for (name, fit) in [
            (
                "gauss_newton",
                gauss_newton(residual, jacobian, &x0).unwrap(),
            ),
            (
                "levenberg_marquardt",
                levenberg_marquardt(residual, jacobian, &x0).unwrap(),
            ),
        ] {
            let at_minimum = ((fit.cost - least) / least).abs() < 1e-7;
            assert!(!fit.converged || at_minimum, "b0 = {b0:?}, {name}: {fit:?}");
        }
    }
    // Where no step can be made from it a point the residual test passes is returned
    // unconfirmed: one ulp from the pole with no budget at all.
    let none = OptOptions::new().with_max_iter(0);
    let x0 = col(&[3.0, -t21 + t21 * f64::EPSILON, 0.4]);
    for fit in [
        gauss_newton_with(residual, jacobian, &x0, &none).unwrap(),
        levenberg_marquardt_with(residual, jacobian, &x0, &none).unwrap(),
    ] {
        assert!(!fit.converged && fit.iterations == 0, "{fit:?}");
    }
    // Held to the region, the damped fit leaves the pole and reaches the minimum.
    let fit =
        levenberg_marquardt(residual, jacobian, &col(&[3.0, -t21 * (1.0 - 1e-9), 0.4])).unwrap();
    assert!(
        fit.converged && ((fit.cost - least) / least).abs() < 1e-9,
        "{fit:?}"
    );
    for (b, c) in fit.x.as_slice().iter().zip(minimiser) {
        assert!((b - c).abs() < 1e-5, "{fit:?}");
    }

    // f32: the second review's start beside the pole of these data, and three of the eight
    // starts of its random scan (trials 2529, 3387 and 4806, exact data of the parameters given)
    // from which both fits said converged with b + t within 6·10⁻³ of a pole, at costs 8 to 12,000
    // times what a fit in f64 from their points goes on to. Converged must mean that such a fit
    // cannot halve the cost.
    let tight = OptOptions::new()
        .with_rtol(0.0)
        .with_gtol(1e-13)
        .with_max_iter(20000);
    let cases: [([f64; 3], [f64; 3]); 4] = [
        (
            [3.0, 1.2, 0.4],
            [3.4623320151583035, -5.79259352022179, -1.5891194938600435],
        ),
        (
            [5.651851662240947, 1.7518590165392445, -0.6990745664603637],
            [2.4109029137758573, -6.347389456968752, -2.205158534333017],
        ),
        (
            [4.882277311341783, 0.8231350327527993, 0.9757431187564094],
            [2.232192171763648, -0.8269852322080645, 0.560589166889564],
        ),
        (
            [3.451522609972085, 1.3261825919014465, -1.0612906555410413],
            [2.600490760656731, -3.5844857272352244, -0.16634824020495573],
        ),
    ];
    for (truth, start) in cases {
        let t32: Vec<f32> = t.iter().map(|v| *v as f32).collect();
        let y32: Vec<f32> = t
            .iter()
            .map(|t| (truth[0] * t / (truth[1] + t) + truth[2]) as f32)
            .collect();
        let residual32 = |p: &Matrix<f32>| hyperbola_residual(&t32, &y32, p);
        let jacobian32 = |p: &Matrix<f32>| hyperbola_jacobian(&t32, p);
        // The data as the f32 fit has them, in f64, to judge its point by.
        let (tc, yc): (Vec<f64>, Vec<f64>) = (
            t32.iter().map(|v| *v as f64).collect(),
            y32.iter().map(|v| *v as f64).collect(),
        );
        let residual = |p: &Matrix<f64>| hyperbola_residual(&tc, &yc, p);
        let jacobian = |p: &Matrix<f64>| hyperbola_jacobian(&tc, p);
        let x0 = Matrix::from_column_major(3, 1, start.iter().map(|v| *v as f32).collect());
        for (name, fit) in [
            (
                "gauss_newton",
                gauss_newton(residual32, jacobian32, &x0).unwrap(),
            ),
            (
                "levenberg_marquardt",
                levenberg_marquardt(residual32, jacobian32, &x0).unwrap(),
            ),
        ] {
            if !fit.converged {
                continue;
            }
            let at = col(&fit
                .x
                .as_slice()
                .iter()
                .map(|v| *v as f64)
                .collect::<Vec<_>>());
            let cost = 0.5 * residual(&at).as_slice().iter().map(|r| r * r).sum::<f64>();
            let further = levenberg_marquardt_with(residual, jacobian, &at, &tight).unwrap();
            assert!(
                cost <= 2.0 * further.cost + 1e-6,
                "f32 {truth:?} from {start:?}, {name}: converged at cost {cost:e}, an f64 fit goes on to {:e}: {fit:?}",
                further.cost
            );
        }
    }
}

#[test]
fn an_exact_fit_is_recognised_where_a_parameter_is_zero() {
    // Judged parameter by parameter, the step of a parameter whose answer is zero is rounding
    // about zero and never short against its own size; the residual test recognises the fit
    // instead, once the residual is no larger than rounding the parameters to floats leaves at
    // the exact minimiser. A polynomial with every other coefficient zero, of condition 6·10⁵
    // (degree 8) and 7·10⁸ (degree 12); a quadratic without its linear term; two close decays
    // and no offset.
    for n in [9usize, 13] {
        let t: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
        let truth: Vec<f64> = (0..n)
            .map(|k| if k % 2 == 1 { 0.0 } else { (k + 1) as f64 })
            .collect();
        let y: Vec<f64> = t
            .iter()
            .map(|t| {
                truth
                    .iter()
                    .enumerate()
                    .map(|(k, c)| c * t.powi(k as i32))
                    .sum()
            })
            .collect();
        let residual = |p: &Matrix<f64>| {
            col(&t
                .iter()
                .zip(&y)
                .map(|(t, y)| (0..n).map(|k| p[(k, 0)] * t.powi(k as i32)).sum::<f64>() - y)
                .collect::<Vec<_>>())
        };
        let jacobian = |_: &Matrix<f64>| Matrix::from_fn(40, n, |i, k| t[i].powi(k as i32));
        // About the condition number times ε, the most the data determine the coefficients to.
        let accuracy = if n == 9 { 1e-9 } else { 1e-6 };
        for fit in [
            gauss_newton(residual, jacobian, &Matrix::zeros(n, 1)).unwrap(),
            levenberg_marquardt(residual, jacobian, &Matrix::zeros(n, 1)).unwrap(),
        ] {
            let error = fit
                .x
                .as_slice()
                .iter()
                .zip(&truth)
                .fold(0.0f64, |e, (a, b)| e.max((a - b).abs()));
            assert!(
                fit.converged && error < accuracy,
                "degree {}: {error:e}, {fit:?}",
                n - 1
            );
        }
    }
    // The same in f32, which the norm of all the parameters together did not recognise either.
    let (m, n) = (40usize, 9usize);
    let t: Vec<f32> = (0..m).map(|i| i as f32 / 39.0).collect();
    let truth: Vec<f32> = (0..n)
        .map(|k| if k % 2 == 1 { 0.0 } else { (k + 1) as f32 })
        .collect();
    let y: Vec<f32> = t
        .iter()
        .map(|t| {
            truth
                .iter()
                .enumerate()
                .map(|(k, c)| c * t.powi(k as i32))
                .sum()
        })
        .collect();
    let residual = |p: &Matrix<f32>| {
        Matrix::from_fn(m, 1, |i, _| {
            (0..n).map(|k| p[(k, 0)] * t[i].powi(k as i32)).sum::<f32>() - y[i]
        })
    };
    let jacobian = |_: &Matrix<f32>| Matrix::from_fn(m, n, |i, k| t[i].powi(k as i32));
    for fit in [
        gauss_newton(residual, jacobian, &Matrix::zeros(n, 1)).unwrap(),
        levenberg_marquardt(residual, jacobian, &Matrix::zeros(n, 1)).unwrap(),
    ] {
        let error = fit
            .x
            .as_slice()
            .iter()
            .zip(&truth)
            .fold(0.0f32, |e, (a, b)| e.max((a - b).abs()));
        // The condition 6·10⁵ times f32's ε is 0.07.
        assert!(fit.converged && error < 0.1, "f32: {error:e}, {fit:?}");
    }
    // 1 + 0·t + 3t², from zero and from (5, 5, 5).
    let t: Vec<f64> = (0..10).map(|i| i as f64 / 3.0).collect();
    let residual = |p: &Matrix<f64>| {
        col(&t
            .iter()
            .map(|t| p[(0, 0)] + p[(1, 0)] * t + p[(2, 0)] * t * t - 1.0 - 3.0 * t * t)
            .collect::<Vec<_>>())
    };
    let jacobian = |_: &Matrix<f64>| Matrix::from_fn(10, 3, |i, k| t[i].powi(k as i32));
    for x0 in [col(&[0.0, 0.0, 0.0]), col(&[5.0, 5.0, 5.0])] {
        for fit in [
            gauss_newton(residual, jacobian, &x0).unwrap(),
            levenberg_marquardt(residual, jacobian, &x0).unwrap(),
        ] {
            assert!(
                fit.converged
                    && (fit.x[(0, 0)] - 1.0).abs() < 1e-13
                    && fit.x[(1, 0)].abs() < 1e-13
                    && (fit.x[(2, 0)] - 3.0).abs() < 1e-13,
                "{fit:?}"
            );
        }
    }
    // 2e^(−1.5t) by a·e^(−bt) + c, whose offset c is zero, from (1, 1, 1).
    let t: Vec<f64> = (0..25).map(|i| 8.0 * i as f64 / 24.0).collect();
    let residual = |p: &Matrix<f64>| {
        Matrix::from_fn(25, 1, |i, _| {
            p[(0, 0)] * (-p[(1, 0)] * t[i]).exp() + p[(2, 0)] - 2.0 * (-1.5 * t[i]).exp()
        })
    };
    let jacobian = |p: &Matrix<f64>| {
        Matrix::from_fn(25, 3, |i, k| {
            let e = (-p[(1, 0)] * t[i]).exp();
            [e, -p[(0, 0)] * t[i] * e, 1.0][k]
        })
    };
    for fit in [
        gauss_newton(residual, jacobian, &col(&[1.0, 1.0, 1.0])).unwrap(),
        levenberg_marquardt(residual, jacobian, &col(&[1.0, 1.0, 1.0])).unwrap(),
    ] {
        assert!(
            fit.converged
                && (fit.x[(0, 0)] - 2.0).abs() < 1e-12
                && (fit.x[(1, 0)] - 1.5).abs() < 1e-12
                && fit.x[(2, 0)].abs() < 1e-12,
            "{fit:?}"
        );
    }
}

/// The three models of the random scan of fits: a decay with an offset, a logistic curve and the
/// sum of two decays; the value at `t` and its derivatives in the parameters.
fn scan_model(kind: usize, p: &[f64], t: f64) -> (f64, Vec<f64>) {
    match kind {
        0 => {
            let e = (-p[1] * t).exp();
            (p[0] * e + p[2], vec![e, -p[0] * t * e, 1.0])
        }
        1 => {
            let e = (-p[2] * t).exp();
            let d = 1.0 + p[1] * e;
            (
                p[0] / d,
                vec![1.0 / d, -p[0] * e / (d * d), p[0] * p[1] * t * e / (d * d)],
            )
        }
        _ => {
            let (e1, e2) = ((-p[1] * t).exp(), (-p[3] * t).exp());
            (
                p[0] * e1 + p[2] * e2,
                vec![e1, -p[0] * t * e1, e2, -p[2] * t * e2],
            )
        }
    }
}

#[test]
fn every_converged_fit_is_stationary_or_cannot_be_improved() {
    // review B1: the reviewer's random scan of fits, three models with parameters drawn at
    // random, exact data and data with noise of 10⁻³ and 10⁻¹, from starts off by a factor of up
    // to ten either way and of the wrong sign a tenth of the time. Every fit that says converged
    // must be stationary by the gradient test, or exact to rounding, or at a point from which
    // the damped fit with the tests tightened cannot lower the cost by a millionth. Before the
    // fix 13 runs of gauss_newton and 15 of levenberg_marquardt in 3000 said converged at points
    // with a cosine near 1, at costs up to 10⁶³; a third of the scan is run here.
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut uniform = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    fn normal(u: &mut dyn FnMut() -> f64) -> f64 {
        let (a, b) = (u().max(1e-300), u());
        (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
    }
    let tight = OptOptions::new()
        .with_rtol(0.0)
        .with_gtol(1e-13)
        .with_max_iter(20000);
    let gtol = OptOptions::<f64>::new().gtol;
    let trials = if cfg!(miri) { 6 } else { 3000 };
    let mut failures = Vec::new();
    let mut converged = [0usize; 2];
    for trial in 0..trials {
        let kind = trial % 3;
        let truth: Vec<f64> = match kind {
            0 => vec![
                1.0 + 4.0 * uniform(),
                0.2 + 2.0 * uniform(),
                normal(&mut uniform),
            ],
            1 => vec![
                1.0 + 9.0 * uniform(),
                0.5 + 20.0 * uniform(),
                0.3 + 2.0 * uniform(),
            ],
            _ => vec![
                1.0 + 4.0 * uniform(),
                0.1 + 0.5 * uniform(),
                1.0 + 4.0 * uniform(),
                1.0 + 3.0 * uniform(),
            ],
        };
        let t: Vec<f64> = (0..25).map(|i| 8.0 * i as f64 / 24.0).collect();
        let noise = [0.0, 1e-3, 1e-1][trial / 3 % 3];
        let y: Vec<f64> = t
            .iter()
            .map(|t| scan_model(kind, &truth, *t).0 + noise * normal(&mut uniform))
            .collect();
        let x0: Vec<f64> = truth
            .iter()
            .map(|v| {
                let factor = 10f64.powf(uniform() * 2.0 - 1.0);
                v * factor * if uniform() < 0.1 { -1.0 } else { 1.0 }
            })
            .collect();
        // A third of the trials, in blocks of nine that hold every model at every level of
        // noise, and every one that failed before the fix.
        let known = [
            516, 525, 756, 849, 1224, 1308, 1311, 1781, 1788, 2121, 2256, 2322, 2414, 2454, 2784,
        ];
        if !cfg!(miri) && trial / 9 % 3 != 0 && !known.contains(&trial) {
            continue;
        }
        let n = truth.len();
        let residual = |p: &Matrix<f64>| {
            Matrix::from_fn(25, 1, |i, _| scan_model(kind, p.as_slice(), t[i]).0 - y[i])
        };
        let jacobian = |p: &Matrix<f64>| {
            Matrix::from_fn(25, n, |i, k| scan_model(kind, p.as_slice(), t[i]).1[k])
        };
        let data = 0.5 * y.iter().map(|v| v * v).sum::<f64>();
        for (method, damped) in [("gauss_newton", false), ("levenberg_marquardt", true)] {
            let x0 = col(&x0);
            let fit = if damped {
                levenberg_marquardt(residual, jacobian, &x0)
            } else {
                gauss_newton(residual, jacobian, &x0)
            };
            let Some(fit) = fit.filter(|f| f.converged) else {
                continue;
            };
            converged[damped as usize] += 1;
            if fit.cost <= 1e-24 * data || largest_cosine(residual, jacobian, &fit.x) <= gtol {
                continue;
            }
            let again = levenberg_marquardt_with(residual, jacobian, &fit.x, &tight);
            let gained = again.is_none_or(|a| a.cost < fit.cost * (1.0 - 1e-6));
            if gained {
                failures.push(format!("trial {trial}, {method}: {fit:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    if !cfg!(miri) {
        assert!(converged[1] >= 850 && converged[0] >= 200, "{converged:?}");
    }
}

#[test]
fn the_nist_problems_reach_their_certified_values() {
    // Levenberg and Marquardt from both starts of five NIST StRD problems: converged wherever
    // MINPACK reaches the certified values, with as many digits give or take half a digit, and
    // the certified residual sum of squares. From BoxBOD's first start MINPACK reports success at
    // a point with no correct digit — a plateau where the model has saturated to a constant — and
    // this method must not.
    let problems = if cfg!(miri) { Vec::new() } else { nist() };
    for p in &problems {
        for (k, s) in p.starts.iter().enumerate() {
            let label = format!("{} from start {}", p.name, k + 1);
            let &(_, _, _, _, minpack) = NIST_SCIPY
                .iter()
                .find(|t| t.0 == p.name && t.1 == k + 1)
                .unwrap();
            let residual = |b: &Matrix<f64>| p.residual(b);
            let jacobian = |b: &Matrix<f64>| p.jacobian(b);
            let fit = levenberg_marquardt(residual, jacobian, &col(s)).unwrap();
            if minpack < 1.0 {
                assert!(
                    !fit.converged || p.digits(fit.x.as_slice()) >= 4.0,
                    "{label}: {fit:?}"
                );
                continue;
            }
            assert!(fit.converged, "{label}: {fit:?}");
            let digits = p.digits(fit.x.as_slice());
            assert!(
                digits >= minpack - 0.6 && digits >= 3.5,
                "{label}: {digits:.2} digits against MINPACK's {minpack}"
            );
            assert!(
                (2.0 * fit.cost / p.rss - 1.0).abs() < 1e-7,
                "{label}: {}",
                2.0 * fit.cost
            );
            // Gauss and Newton, where it converges at all, converges to the same.
            let fit = gauss_newton(residual, jacobian, &col(s)).unwrap();
            if fit.converged {
                assert!(
                    p.digits(fit.x.as_slice()) >= 3.5,
                    "{label}, undamped: {fit:?}"
                );
            }
        }
    }
}

#[test]
fn dependent_columns_are_refused_by_gauss_newton_and_resolved_by_levenberg_marquardt() {
    // audit n_opt-9: two identical columns of the Jacobian left a rounding-sized diagonal in R,
    // and Gauss and Newton returned parameters of 10¹⁶ where None is documented.
    let c = [1.0, 2.0, 3.0, 4.0];
    let residual = |p: &Matrix<f64>| {
        col(&c
            .iter()
            .map(|c| p[(0, 0)] + p[(1, 0)] - c)
            .collect::<Vec<_>>())
    };
    let twins = |_: &Matrix<f64>| Matrix::from_fn(4, 2, |_, _| 1.0);
    assert!(gauss_newton(residual, twins, &start(0.0, 0.0)).is_none());
    // Levenberg and Marquardt fits their sum, to the 2·10⁻⁸ a cosine of √ε leaves, with the
    // least cost there is, 2.5, and splits it about evenly as the shortest step would.
    let fit = levenberg_marquardt(residual, twins, &start(0.0, 0.0)).unwrap();
    assert!(fit.converged, "{fit:?}");
    assert!(
        (fit.x[(0, 0)] + fit.x[(1, 0)] - 2.5).abs() < 1e-7 && (fit.cost - 2.5).abs() < 1e-14,
        "{fit:?}"
    );
    assert!((fit.x[(0, 0)] - fit.x[(1, 0)]).abs() < 1e-6, "{fit:?}");
    // The third column the sum of the first two.
    let t = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [1.0, 3.0, 2.0, 5.0, 4.0];
    let residual = |p: &Matrix<f64>| {
        col(&t
            .iter()
            .zip(&y)
            .map(|(t, y)| p[(0, 0)] + p[(1, 0)] * t + p[(2, 0)] * (1.0 + t) - y)
            .collect::<Vec<_>>())
    };
    let jacobian = |_: &Matrix<f64>| Matrix::from_fn(5, 3, |i, k| [1.0, t[i], 1.0 + t[i]][k]);
    assert!(gauss_newton(residual, jacobian, &Matrix::zeros(3, 1)).is_none());
    let fit = levenberg_marquardt(residual, jacobian, &Matrix::zeros(3, 1)).unwrap();
    // The line fitted to the five points: intercept 1.4 and slope 0.8 make y = 1.4 + 0.8t.
    let (p0, p1, p2) = (fit.x[(0, 0)], fit.x[(1, 0)], fit.x[(2, 0)]);
    assert!(fit.converged, "{fit:?}");
    assert!(
        (p0 + p2 - 1.4).abs() < 1e-10 && (p1 + p2 - 0.8).abs() < 1e-10,
        "{fit:?}"
    );
    // Its exact zero column, which made None before, still does.
    let zero = |_: &Matrix<f64>| Matrix::from_fn(4, 2, |_, k| if k == 0 { 1.0 } else { 0.0 });
    assert!(
        gauss_newton(
            |p: &Matrix<f64>| col(&c.iter().map(|c| p[(0, 0)] - c).collect::<Vec<_>>()),
            zero,
            &start(0.0, 0.0)
        )
        .is_none()
    );
    // Columns that become dependent at a point the steps have led to: that point is the best
    // reached, and is returned with converged false rather than thrown away (review 2). A peak
    // a·e^(−((t − c)/w)²) + d + e·t through 30 exact points of [0, 8] from a start that puts its
    // centre at 13, beyond the data: the first step lowers the cost from 317 to 5.0 and leaves
    // the three columns of the peak too small beside those of the line to be told apart (trial
    // 778 of the second review's scan of fits).
    let truth = [
        1.9686361158705727,
        2.3866511148444953,
        1.1300463959062403,
        1.0483033639588681,
        -0.005383608422635796,
    ];
    let peak = |p: &[f64], t: f64| {
        let z = (t - p[1]) / p[2];
        let e = (-z * z).exp();
        (
            p[0] * e + p[3] + p[4] * t,
            [
                e,
                p[0] * e * 2.0 * z / p[2],
                p[0] * e * 2.0 * z * z / p[2],
                1.0,
                t,
            ],
        )
    };
    let t: Vec<f64> = (0..30).map(|i| 8.0 * i as f64 / 29.0).collect();
    let y: Vec<f64> = t.iter().map(|t| peak(&truth, *t).0).collect();
    let residual =
        |p: &Matrix<f64>| Matrix::from_fn(30, 1, |i, _| peak(p.as_slice(), t[i]).0 - y[i]);
    let jacobian = |p: &Matrix<f64>| Matrix::from_fn(30, 5, |i, k| peak(p.as_slice(), t[i]).1[k]);
    let x0 = col(&[
        0.8372717380229701,
        13.134215928065277,
        0.9525422958799286,
        6.243452314612634,
        -0.045981348768831865,
    ]);
    let cost = |p: &Matrix<f64>| 0.5 * residual(p).as_slice().iter().map(|r| r * r).sum::<f64>();
    let fit = gauss_newton(residual, jacobian, &x0).expect("the point reached");
    assert!(
        !fit.converged && fit.iterations >= 1 && fit.cost < cost(&x0) / 10.0,
        "{fit:?} from a cost of {}",
        cost(&x0)
    );
    assert!(
        (fit.cost - cost(&fit.x)).abs() <= 1e-14 * fit.cost,
        "{fit:?}"
    );
}

/// The polynomial of degree `n − 1` with coefficients 1, −2, 3, …, sampled exactly at 40 points of
/// [0, 1], to be fitted from zero.
struct Polynomial {
    t: Vec<f64>,
    y: Vec<f64>,
    truth: Vec<f64>,
}

impl Polynomial {
    fn new(n: usize) -> Self {
        let t: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
        let truth: Vec<f64> = (0..n)
            .map(|k| {
                if k % 2 == 0 {
                    (k + 1) as f64
                } else {
                    -((k + 1) as f64)
                }
            })
            .collect();
        let y = t
            .iter()
            .map(|ti| {
                truth
                    .iter()
                    .enumerate()
                    .map(|(k, c)| c * ti.powi(k as i32))
                    .sum()
            })
            .collect();
        Self { t, y, truth }
    }

    fn residual(&self, p: &Matrix<f64>) -> Matrix<f64> {
        col(&self
            .t
            .iter()
            .zip(&self.y)
            .map(|(t, y)| {
                (0..self.truth.len())
                    .map(|k| p[(k, 0)] * t.powi(k as i32))
                    .sum::<f64>()
                    - y
            })
            .collect::<Vec<_>>())
    }

    fn jacobian(&self, _: &Matrix<f64>) -> Matrix<f64> {
        Matrix::from_fn(self.t.len(), self.truth.len(), |i, k| {
            self.t[i].powi(k as i32)
        })
    }

    /// The largest error of a solution, relative to the largest coefficient.
    fn error(&self, x: &Matrix<f64>) -> f64 {
        x.as_slice()
            .iter()
            .zip(&self.truth)
            .fold(0.0f64, |e, (a, b)| e.max((a - b).abs()))
            / self.truth.len() as f64
    }
}

#[test]
fn levenberg_marquardt_fits_an_ill_conditioned_linear_model() {
    // audit n_opt-11: with the default options a degree-8 polynomial through exact samples came
    // back wrong by up to 8.5, converged = true; the condition of the Jacobian is 6·10⁵, so the
    // coefficients are determined to about 10⁻¹⁰, which Gauss and Newton reaches in one step.
    let poly = Polynomial::new(9);
    let (residual, jacobian) = (
        |p: &Matrix<f64>| poly.residual(p),
        |p: &Matrix<f64>| poly.jacobian(p),
    );
    let error = |x: &Matrix<f64>| poly.error(x);
    let lm = levenberg_marquardt(residual, jacobian, &Matrix::zeros(9, 1)).unwrap();
    let gn = gauss_newton(residual, jacobian, &Matrix::zeros(9, 1)).unwrap();
    assert!(lm.converged && gn.converged);
    assert!(
        error(&lm.x) < 1e-10 && error(&gn.x) < 1e-10,
        "{} and {}",
        error(&lm.x),
        error(&gn.x)
    );
    assert!(
        lm.iterations <= 3,
        "a linear problem takes one undamped step: {lm:?}"
    );
}

#[test]
fn the_damped_step_is_solved_by_qr() {
    // audit n_opt-12: the damped step came from the normal equations by LU, which squares the
    // condition number: with every tolerance off a degree-12 fit was two digits worse than the
    // QR step of Gauss and Newton, and a degree-8 fit in f32 64% off where QR gives 10⁻³.
    let off = OptOptions::<f64>::new()
        .with_gtol(0.0)
        .with_rtol(0.0)
        .with_xtol(0.0)
        .with_ftol(0.0);
    let poly = Polynomial::new(13);
    let (residual, jacobian) = (
        |p: &Matrix<f64>| poly.residual(p),
        |p: &Matrix<f64>| poly.jacobian(p),
    );
    let error = |x: &Matrix<f64>| poly.error(x);
    let gn = gauss_newton(residual, jacobian, &Matrix::zeros(13, 1)).unwrap();
    let lm = levenberg_marquardt_with(residual, jacobian, &Matrix::zeros(13, 1), &off).unwrap();
    let lm_default = levenberg_marquardt(residual, jacobian, &Matrix::zeros(13, 1)).unwrap();
    // Gauss and Newton, the reference, to the accuracy the rounding of the problem allows, which
    // is coarser than 10⁻⁹. The Jacobian has condition 6.9·10⁸ (least singular value 1.3·10⁻⁸);
    // the exact least squares solution of the problem as it is rounded to floats is itself
    // 2.0·10⁻⁹ from the coefficients (relative to the largest one, as `error` measures), and a
    // backward stable solver lands about as far, scattered by its own rounding: LAPACK's gelsd,
    // gelsy and Householder QR at 3.8, 3.3 and 3.9·10⁻⁹ (data/gen_opt.py, polynomial()). Gauss
    // and Newton lands at 5.0·10⁻¹⁰ with the QR of aarch64, at 2.0·10⁻⁹ with that of x86_64, and
    // between 1.3·10⁻¹¹ and 8.4·10⁻⁹ over 3,000 orders of the rows (the same problem, rounded
    // otherwise by the QR); hence 10⁻⁸. Solved from the normal equations, of condition
    // 4.8·10¹⁷ > 1/ε, it ends 1.4 off.
    assert!(error(&gn.x) < 1e-8, "{}", error(&gn.x));
    assert!(
        error(&lm.x) <= 4.0 * error(&gn.x),
        "{} against Gauss–Newton's {}",
        error(&lm.x),
        error(&gn.x)
    );
    assert!(lm_default.converged && error(&lm_default.x) <= 2.0 * error(&gn.x));

    let (m, n) = (40usize, 9usize);
    let t: Vec<f32> = (0..m).map(|i| (i as f64 / 39.0) as f32).collect();
    let truth: Vec<f64> = (0..n)
        .map(|k| {
            if k % 2 == 0 {
                (k + 1) as f64
            } else {
                -((k + 1) as f64)
            }
        })
        .collect();
    let y: Vec<f32> = (0..m)
        .map(|i| {
            (0..n)
                .map(|k| truth[k] * (i as f64 / 39.0).powi(k as i32))
                .sum::<f64>() as f32
        })
        .collect();
    let residual = |p: &Matrix<f32>| {
        Matrix::from_column_major(
            m,
            1,
            (0..m)
                .map(|i| (0..n).map(|k| p[(k, 0)] * t[i].powi(k as i32)).sum::<f32>() - y[i])
                .collect(),
        )
    };
    let jacobian = |_: &Matrix<f32>| Matrix::from_fn(m, n, |i, k| t[i].powi(k as i32));
    let error = |x: &Matrix<f32>| {
        x.as_slice()
            .iter()
            .zip(&truth)
            .fold(0.0f64, |e, (a, b)| e.max((*a as f64 - b).abs()))
            / 9.0
    };
    let off = OptOptions::<f32>::new()
        .with_gtol(0.0)
        .with_rtol(0.0)
        .with_xtol(0.0)
        .with_ftol(0.0);
    let gn = gauss_newton(residual, jacobian, &Matrix::zeros(n, 1)).unwrap();
    let lm = levenberg_marquardt_with(residual, jacobian, &Matrix::zeros(n, 1), &off).unwrap();
    let lm_default = levenberg_marquardt(residual, jacobian, &Matrix::zeros(n, 1)).unwrap();
    assert!(error(&gn.x) < 1e-2, "{}", error(&gn.x));
    assert!(
        error(&lm.x) <= 2.0 * error(&gn.x),
        "{} against Gauss–Newton's {}",
        error(&lm.x),
        error(&gn.x)
    );
    assert!(
        lm_default.converged && error(&lm_default.x) <= 2.0 * error(&gn.x),
        "{lm_default:?}"
    );
}

#[test]
fn a_fit_that_cannot_be_made_is_refused() {
    let x0 = start(1.0, 1.0);
    // A Jacobian that does not belong to the residual: no step improves the cost as predicted,
    // so the damped method narrows its region until it gives up rather than reporting a fit.
    let dependent = |_: &Matrix<f64>| Matrix::from_fn(5, 2, |_, _| 1.0);
    let damped = levenberg_marquardt(decay_residual, dependent, &x0).unwrap();
    assert!(
        !damped.converged,
        "a dependent Jacobian converged: {damped:?}"
    );
    assert!(damped.cost > 0.0);
    // Fewer residuals than parameters is not a least squares at all.
    let thin = levenberg_marquardt(
        |p: &Matrix<f64>| Matrix::from_column_major(1, 1, vec![p[(0, 0)] - 1.0]),
        |_: &Matrix<f64>| Matrix::from_rows(&[&[1.0, 0.0]]),
        &x0,
    );
    assert!(thin.is_none(), "an underdetermined fit was accepted");
}

#[test]
fn the_methods_work_in_f32() {
    // (a − 1)² + 2(b − 2)², written out as a sum whose terms are some 10 times the value at the
    // minimum: f32 cannot tell the value of points nearer than about 10⁻³ apart, and a gradient
    // tolerance of √ε asks for 10⁻⁴. BFGS gets as close as the values let it and says whether
    // the gradient test held; with the bowl written as the sum of squares it is met.
    let expanded = |x: &Matrix<f32>| {
        let (a, b) = (x[(0, 0)], x[(1, 0)]);
        a * a + 2.0 * b * b - 2.0 * a - 8.0 * b + 9.0
    };
    let factored = |x: &Matrix<f32>| (x[(0, 0)] - 1.0).powi(2) + 2.0 * (x[(1, 0)] - 2.0).powi(2);
    let gradient = |x: &Matrix<f32>| {
        let (a, b) = (x[(0, 0)], x[(1, 0)]);
        Matrix::from_column_major(2, 1, vec![2.0 * a - 2.0, 4.0 * b - 8.0])
    };
    let x0 = Matrix::from_column_major(2, 1, vec![3.0f32, -4.0]);
    let gtol = OptOptions::<f32>::new().gtol;
    let near =
        |x: &Matrix<f32>, tol: f32| (x[(0, 0)] - 1.0).abs() < tol && (x[(1, 0)] - 2.0).abs() < tol;
    for (name, r) in [
        ("bfgs", bfgs(expanded, gradient, &x0).unwrap()),
        ("cg", conjugate_gradient(expanded, gradient, &x0).unwrap()),
    ] {
        let g = gradient(&r.x);
        assert_eq!(
            r.converged,
            g[(0, 0)].abs().max(g[(1, 0)].abs()) <= gtol,
            "{name}: {r:?}"
        );
        assert!(near(&r.x, 1e-3), "{name}: {r:?}");
    }
    for (name, r) in [
        ("bfgs", bfgs(factored, gradient, &x0).unwrap()),
        ("cg", conjugate_gradient(factored, gradient, &x0).unwrap()),
    ] {
        assert!(r.converged && near(&r.x, 1e-4), "{name}: {r:?}");
    }
    let s = nelder_mead(expanded, &x0, 1.0).unwrap();
    assert!(s.converged && near(&s.x, 1e-2), "the simplex in f32: {s:?}");

    let residual = |p: &Matrix<f32>| {
        Matrix::from_column_major(
            2,
            1,
            vec![p[(0, 0)] + p[(1, 0)] - 3.0, p[(0, 0)] - p[(1, 0)] - 1.0],
        )
    };
    let jacobian = |_: &Matrix<f32>| Matrix::from_rows(&[&[1.0f32, 1.0], &[1.0, -1.0]]);
    let zero = Matrix::from_column_major(2, 1, vec![0.0f32, 0.0]);
    for fit in [
        gauss_newton(residual, jacobian, &zero).unwrap(),
        levenberg_marquardt(residual, jacobian, &zero).unwrap(),
    ] {
        assert!(
            fit.converged
                && (fit.x[(0, 0)] - 2.0).abs() < 1e-6
                && (fit.x[(1, 0)] - 1.0).abs() < 1e-6,
            "{fit:?}"
        );
    }
    let minimum = minimize(|x: f32| (x - 0.25).powi(2) + 1.0, 0.0, 1.0).unwrap();
    assert!(minimum.converged && (minimum.x - 0.25).abs() <= OptOptions::<f32>::new().tol_at(0.25));
}

#[test]
fn gauss_newton_keeps_its_one_option_set_as_documented() {
    // The tests of the fits are those of levenberg_marquardt_with, made on every step.
    let opts = OptOptions::new().with_max_iter(1);
    let fit = gauss_newton_with(decay_residual, decay_jacobian, &start(1.0, 1.0), &opts).unwrap();
    assert!(!fit.converged && fit.iterations == 1 && fit.cost < decay_cost(&start(1.0, 1.0)));
}

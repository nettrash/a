//! L-BFGS-B, [`a::math::numerics::lbfgsb`]: the problems of `data/gen_lbfgsb.py` against what
//! SciPy's L-BFGS-B (the reference code of Zhu, Byrd, Lu and Nocedal, version 3.0) does with the
//! same memory and tolerances; the boxed quadratics against their KKT points found independently;
//! problems whose answer is known in closed form; and the edge cases — empty boxes, fixed
//! variables, starts outside the box, one variable, limits, `f32`.

use a::math::matrix::{Matrix, Scalar};
use a::math::numerics::{
    Bound, LbfgsbOptions, LbfgsbResult, LbfgsbStop, Real, lbfgsb, lbfgsb_with,
};
use std::cell::{Cell, RefCell};

const INF: f64 = f64::INFINITY;

fn col(v: &[f64]) -> Matrix<f64> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

/// The projected gradient `‖P(x − g) − x‖∞`, worked out here from the gradient the test
/// computes itself, so that the KKT conditions of an answer are checked independently of what
/// the method says about them.
fn projected(x: &[f64], g: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut largest = 0.0f64;
    for i in 0..x.len() {
        let stepped = (x[i] - g[i]).max(lower[i]).min(upper[i]);
        largest = largest.max((stepped - x[i]).abs());
    }
    largest
}

/// The active set as the generator prints it: `F` fixed, `L` and `U` at a bound, `.` free.
fn active(r: &LbfgsbResult<f64>) -> String {
    r.active
        .iter()
        .map(|b| match b {
            Bound::Free => '.',
            Bound::Lower => 'L',
            Bound::Upper => 'U',
            Bound::Fixed => 'F',
        })
        .collect()
}

type Value = Box<dyn Fn(&[f64]) -> f64>;
type Gradient = Box<dyn Fn(&[f64]) -> Vec<f64>>;

/// A problem of `data/gen_lbfgsb.py`, with the options the generator passes to SciPy.
struct Problem {
    name: &'static str,
    f: Value,
    g: Gradient,
    x0: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    memory: usize,
    pgtol: f64,
    ftol: f64,
}

impl Problem {
    fn options(&self) -> LbfgsbOptions<f64> {
        LbfgsbOptions::new()
            .with_memory(self.memory)
            .with_pgtol(self.pgtol)
            .with_ftol(self.ftol)
    }

    fn run(&self) -> LbfgsbResult<f64> {
        self.run_with(&self.options())
    }

    fn run_with(&self, opts: &LbfgsbOptions<f64>) -> LbfgsbResult<f64> {
        let f = |x: &Matrix<f64>| (self.f)(x.as_slice());
        let g = |x: &Matrix<f64>| col(&(self.g)(x.as_slice()));
        lbfgsb_with(
            f,
            g,
            &col(&self.x0),
            &col(&self.lower),
            &col(&self.upper),
            opts,
        )
        .unwrap_or_else(|| panic!("{}: refused", self.name))
    }

    /// The projected gradient at `x`, by the test's own arithmetic.
    fn kkt(&self, x: &[f64]) -> f64 {
        projected(x, &(self.g)(x), &self.lower, &self.upper)
    }
}

/// SciPy's default `ftol`, `factr = 10⁷` times ε.
const FACTR: f64 = 1e7 * f64::EPSILON;

/// The chained Rosenbrock function of `scipy.optimize.rosen`.
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

fn rosen_start(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
}

/// The problem of the sample driver of the reference code (Zhu, Byrd, Lu, Nocedal 1997): a
/// chained Rosenbrock-like function of 25 variables, the odd-numbered of them in [1, 100] and
/// the even-numbered in [−100, 100].
fn driver(x: &[f64]) -> f64 {
    let mut f = 0.25 * (x[0] - 1.0).powi(2);
    for i in 1..x.len() {
        f += (x[i] - x[i - 1] * x[i - 1]).powi(2);
    }
    4.0 * f
}

fn driver_gradient(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut t1 = x[1] - x[0] * x[0];
    g[0] = 2.0 * (x[0] - 1.0) - 16.0 * x[0] * t1;
    for i in 1..n - 1 {
        let t2 = t1;
        t1 = x[i + 1] - x[i] * x[i];
        g[i] = 8.0 * t2 - 16.0 * x[i] * t1;
    }
    g[n - 1] = 8.0 * t1;
    g
}

/// `A·x` for the tridiagonal `A` with 2 on the diagonal and −1 beside it.
fn tridiagonal(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            2.0 * x[i] - if i > 0 { x[i - 1] } else { 0.0 } - if i + 1 < n { x[i + 1] } else { 0.0 }
        })
        .collect()
}

/// The right-hand side that puts the unconstrained minimiser of `½xᵀAx − bᵀx` partly outside
/// `[0, 1]ⁿ`.
fn quad_b(n: usize) -> Vec<f64> {
    (0..n).map(|i| (1.7 * (i + 1) as f64).sin()).collect()
}

fn quad(n: usize) -> (Value, Gradient) {
    let b = quad_b(n);
    let b2 = b.clone();
    (
        Box::new(move |x: &[f64]| {
            let ax = tridiagonal(x);
            0.5 * x.iter().zip(&ax).map(|(p, q)| p * q).sum::<f64>()
                - x.iter().zip(&b).map(|(p, q)| p * q).sum::<f64>()
        }),
        Box::new(move |x: &[f64]| tridiagonal(x).iter().zip(&b2).map(|(p, q)| p - q).collect()),
    )
}

/// `½Σ dᵢ(xᵢ − cᵢ)²` with curvatures `dᵢ = cond^(i/9)` and centres `cᵢ = sin(i + 1)`.
fn scaled(cond: f64) -> (Value, Gradient, Vec<f64>) {
    let d: Vec<f64> = (0..10).map(|i| cond.powf(i as f64 / 9.0)).collect();
    let c: Vec<f64> = (0..10).map(|i| (i as f64 + 1.0).sin()).collect();
    let (d2, c2, c3) = (d.clone(), c.clone(), c.clone());
    (
        Box::new(move |x: &[f64]| 0.5 * (0..10).map(|i| d[i] * (x[i] - c[i]).powi(2)).sum::<f64>()),
        Box::new(move |x: &[f64]| (0..10).map(|i| d2[i] * (x[i] - c2[i])).collect()),
        c3,
    )
}

/// Brown's badly scaled function (Moré, Garbow and Hillstrom, problem 4): its minimum is 0 at
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

/// Every problem of `data/gen_lbfgsb.py`, in its order.
fn problems() -> Vec<Problem> {
    let mut v = Vec::new();
    let free = |n: usize| (vec![-INF; n], vec![INF; n]);
    let rosen_box = |n: usize| {
        let lower = vec![-2.0; n];
        let upper = (0..n).map(|i| if i % 2 == 0 { 0.5 } else { 2.0 }).collect();
        (lower, upper)
    };
    let mut add = |name,
                   f: Value,
                   g: Gradient,
                   x0,
                   (lower, upper): (Vec<f64>, Vec<f64>),
                   memory,
                   pgtol,
                   ftol| {
        v.push(Problem {
            name,
            f,
            g,
            x0,
            lower,
            upper,
            memory,
            pgtol,
            ftol,
        });
    };
    let sizes: &[usize] = if cfg!(miri) {
        &[2, 10]
    } else {
        &[2, 10, 50, 100]
    };
    for &n in sizes {
        let name = ["rosen2", "rosen10", "rosen50", "rosen100"]
            [[2, 10, 50, 100].iter().position(|m| *m == n).unwrap()];
        add(
            name,
            Box::new(rosen),
            Box::new(rosen_gradient),
            rosen_start(n),
            free(n),
            10,
            1e-5,
            FACTR,
        );
    }
    for (name, n) in [
        ("rosen2_tight", 2),
        ("rosen10_tight", 10),
        ("rosen100_tight", 100),
    ] {
        add(
            name,
            Box::new(rosen),
            Box::new(rosen_gradient),
            rosen_start(n),
            free(n),
            10,
            1e-10,
            0.0,
        );
    }
    for (name, n) in [
        ("rosen2_box", 2),
        ("rosen10_box", 10),
        ("rosen100_box", 100),
    ] {
        add(
            name,
            Box::new(rosen),
            Box::new(rosen_gradient),
            rosen_start(n),
            rosen_box(n),
            10,
            1e-5,
            FACTR,
        );
    }
    for (name, n) in [
        ("rosen2_box_tight", 2),
        ("rosen10_box_tight", 10),
        ("rosen100_box_tight", 100),
    ] {
        add(
            name,
            Box::new(rosen),
            Box::new(rosen_gradient),
            rosen_start(n),
            rosen_box(n),
            10,
            1e-10,
            0.0,
        );
    }
    for (boxed, open, m) in [
        ("rosen10_box_m1", "rosen10_m1", 1),
        ("rosen10_box_m3", "rosen10_m3", 3),
        ("rosen10_box_m5", "rosen10_m5", 5),
        ("rosen10_box_m17", "rosen10_m17", 17),
    ] {
        add(
            boxed,
            Box::new(rosen),
            Box::new(rosen_gradient),
            rosen_start(10),
            rosen_box(10),
            m,
            1e-5,
            FACTR,
        );
        add(
            open,
            Box::new(rosen),
            Box::new(rosen_gradient),
            rosen_start(10),
            free(10),
            m,
            1e-5,
            FACTR,
        );
    }
    let driver_box = || {
        let lower = (0..25)
            .map(|i| if i % 2 == 0 { 1.0 } else { -100.0 })
            .collect();
        (lower, vec![100.0; 25])
    };
    add(
        "driver1",
        Box::new(driver),
        Box::new(driver_gradient),
        vec![3.0; 25],
        driver_box(),
        5,
        1e-5,
        FACTR,
    );
    add(
        "driver1_tight",
        Box::new(driver),
        Box::new(driver_gradient),
        vec![3.0; 25],
        driver_box(),
        5,
        1e-10,
        0.0,
    );
    for (name, n) in [("quad5_box", 5), ("quad20_box", 20), ("quad100_box", 100)] {
        let (f, g) = quad(n);
        add(
            name,
            f,
            g,
            vec![0.5; n],
            (vec![0.0; n], vec![1.0; n]),
            10,
            1e-10,
            0.0,
        );
    }
    for (name, (l, u), x0) in [
        ("quad20_lower", (0.0, INF), 0.5),
        ("quad20_upper", (-INF, 0.25), 0.5),
        ("quad20_free", (-INF, INF), 0.5),
        ("quad20_interior", (-50.0, 50.0), 0.5),
        ("quad20_outside", (0.0, 1.0), 7.5),
    ] {
        let (f, g) = quad(20);
        add(
            name,
            f,
            g,
            vec![x0; 20],
            (vec![l; 20], vec![u; 20]),
            10,
            1e-10,
            0.0,
        );
    }
    let c = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    add(
        "vertex6",
        Box::new(move |x: &[f64]| x.iter().zip(&c).map(|(a, c)| c * a + 0.05 * a * a).sum()),
        Box::new(move |x: &[f64]| x.iter().zip(&c).map(|(a, c)| c + 0.1 * a).collect()),
        vec![0.0; 6],
        (vec![-1.0; 6], vec![2.0; 6]),
        10,
        1e-10,
        0.0,
    );
    for (name, cond) in [("scaled10_1e6", 1e6), ("scaled10_1e12", 1e12)] {
        let (f, g, _) = scaled(cond);
        add(
            name,
            f,
            g,
            vec![0.0; 10],
            (vec![-0.5; 10], vec![0.5; 10]),
            10,
            1e-10,
            0.0,
        );
    }
    add(
        "brown_box",
        Box::new(brown),
        Box::new(brown_gradient),
        vec![1.0, 1.0],
        (vec![0.0, 0.0], vec![1e7, 1.0]),
        10,
        1e-5,
        0.0,
    );
    add(
        "brown_box_cut",
        Box::new(brown),
        Box::new(brown_gradient),
        vec![1.0, 1.0],
        (vec![0.0, 0.0], vec![5e5, 1.0]),
        10,
        1e-5,
        0.0,
    );
    // Hock and Schittkowski's problems 5, 45 and 110, which have bounds only.
    add(
        "hs5",
        Box::new(|x: &[f64]| {
            (x[0] + x[1]).sin() + (x[0] - x[1]).powi(2) - 1.5 * x[0] + 2.5 * x[1] + 1.0
        }),
        Box::new(|x: &[f64]| {
            let c = (x[0] + x[1]).cos();
            vec![c + 2.0 * (x[0] - x[1]) - 1.5, c - 2.0 * (x[0] - x[1]) + 2.5]
        }),
        vec![0.0, 0.0],
        (vec![-1.5, -3.0], vec![4.0, 3.0]),
        10,
        1e-10,
        0.0,
    );
    add(
        "hs45",
        Box::new(|x: &[f64]| 2.0 - x.iter().product::<f64>() / 120.0),
        Box::new(|x: &[f64]| {
            (0..5)
                .map(|i| -(0..5).filter(|&j| j != i).map(|j| x[j]).product::<f64>() / 120.0)
                .collect()
        }),
        vec![2.0; 5],
        (vec![0.0; 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        10,
        1e-10,
        0.0,
    );
    add(
        "hs110",
        Box::new(|x: &[f64]| {
            x.iter()
                .map(|v| (v - 2.0).ln().powi(2) + (10.0 - v).ln().powi(2))
                .sum::<f64>()
                - x.iter().product::<f64>().powf(0.2)
        }),
        Box::new(|x: &[f64]| {
            let p = x.iter().product::<f64>().powf(0.2);
            x.iter()
                .map(|v| {
                    2.0 * (v - 2.0).ln() / (v - 2.0)
                        - 2.0 * (10.0 - v).ln() / (10.0 - v)
                        - 0.2 * p / v
                })
                .collect()
        }),
        vec![9.0; 10],
        (vec![2.001; 10], vec![9.999; 10]),
        10,
        1e-8,
        0.0,
    );
    // Nearly linear: along every step sᵀy is some 10⁻¹⁶ of −gᵀs, which the curvature test leaves
    // out of the memory.
    let c = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    add(
        "nearly_linear6",
        Box::new(move |x: &[f64]| x.iter().zip(&c).map(|(x, c)| c * x + 1e-16 * x * x).sum()),
        Box::new(move |x: &[f64]| x.iter().zip(&c).map(|(x, c)| c + 2e-16 * x).collect()),
        vec![0.3; 6],
        (vec![-INF; 6], vec![2.0; 6]),
        10,
        1e-10,
        0.0,
    );
    // Three random quadratics ½xᵀAx − bᵀx of the first review's set, whose first iterates take
    // the cut back to the box of Morales and Nocedal and the update of the curvature at a
    // breakpoint of the Cauchy search (data/gen_lbfgsb.py, REVIEW_QUADS).
    for &(name, x0, lower, upper, b, a, memory, pgtol, ftol) in REVIEW_QUADS {
        let n = x0.len();
        let (a1, b1, a2, b2) = (a.to_vec(), b.to_vec(), a.to_vec(), b.to_vec());
        add(
            name,
            Box::new(move |x: &[f64]| {
                (0..n)
                    .map(|i| {
                        let ax: f64 = (0..n).map(|j| a1[i * n + j] * x[j]).sum();
                        0.5 * x[i] * ax - b1[i] * x[i]
                    })
                    .sum::<f64>()
            }),
            Box::new(move |x: &[f64]| {
                (0..n)
                    .map(|i| (0..n).map(|j| a2[i * n + j] * x[j]).sum::<f64>() - b2[i])
                    .collect()
            }),
            x0.to_vec(),
            (lower.to_vec(), upper.to_vec()),
            memory,
            pgtol,
            ftol,
        );
    }
    v
}

/// A random quadratic of the first review: name, start, bounds, `b`, `A` by rows, memory, `pgtol`
/// and `ftol`.
type ReviewQuad = (
    &'static str,
    &'static [f64],
    &'static [f64],
    &'static [f64],
    &'static [f64],
    &'static [f64],
    usize,
    f64,
    f64,
);

// The problems quad256, quad160 and quad37 of the first review (its gen_problems.py, seed
// 20261121), copied into data/gen_lbfgsb.py as REVIEW_QUADS.
const REVIEW_QUADS: &[ReviewQuad] = &[
    (
        "review_quad256",
        &[2.245440957823526, -1.517897429612824],
        &[1.0024952564033676, -INF],
        &[INF, INF],
        &[175.5725895245648, -47.77808725557092],
        &[
            265.78715457508673,
            -71.57180612296526,
            -71.57180612296526,
            20.40588688220658,
        ],
        5,
        1e-10,
        0.0,
    ),
    (
        "review_quad160",
        &[
            2.8120771454314917,
            -3.2787311963547703,
            1.5244278159586215,
            -0.768047313909714,
            -1.8048064076785615,
        ],
        &[
            -0.6643240095924081,
            -INF,
            -1.543500169733765,
            0.2159848323577039,
            -0.7822393795124747,
        ],
        &[
            0.09394305744467335,
            INF,
            -0.6316374161751908,
            INF,
            0.5726591865954651,
        ],
        &[
            117047.6042277047,
            128873.48182632017,
            127858.81160176164,
            -68163.40577399344,
            130754.54907105409,
        ],
        &[
            112442.77769307228,
            118645.44456407262,
            123016.19255147915,
            -65665.11998027004,
            124929.48976177708,
            118645.44456407262,
            155688.89964608647,
            126261.08099195866,
            -67248.4784480781,
            133169.29077524738,
            123016.19255147915,
            126261.08099195866,
            135637.82802245175,
            -72407.34083923159,
            137007.3726871055,
            -65665.11998027004,
            -67248.4784480781,
            -72407.34083923159,
            38692.05932680552,
            -73019.97675276316,
            124929.48976177708,
            133169.29077524738,
            137007.3726871055,
            -73019.97675276316,
            139506.73847073436,
        ],
        10,
        1e-10,
        1e-12,
    ),
    (
        "review_quad37",
        &[
            -1.5841177879764983,
            -2.259907726668094,
            -1.3150547247781654,
            -3.290999620879834,
            1.1837683631834728,
        ],
        &[
            -2.149658801293996,
            -1.0924262194088137,
            -1.6054073040130068,
            -1.9437434483263005,
            -INF,
        ],
        &[INF, INF, INF, -0.309230955418588, -1.1091653069354457],
        &[
            -119003.02748951523,
            -324465.3783441008,
            76317.2196359906,
            -40912.94375681026,
            24587.555040962456,
        ],
        &[
            34533.08082564689,
            86238.92344947511,
            -16214.781863674481,
            7417.870998925055,
            -4865.3327226419315,
            86238.92344947511,
            240313.24869211408,
            -57689.133329504584,
            30612.795825426485,
            -18864.072879603387,
            -16214.781863674481,
            -57689.133329504584,
            26150.267360797818,
            -16774.067897761925,
            8091.742285898617,
            7417.870998925055,
            30612.795825426485,
            -16774.067897761925,
            12956.515494991674,
            -6138.8625890455605,
            -4865.3327226419315,
            -18864.072879603387,
            8091.742285898617,
            -6138.8625890455605,
            3494.4918760503956,
        ],
        5,
        1e-8,
        0.0,
    ),
];

/// What SciPy's L-BFGS-B did on a problem.
struct Reference {
    name: &'static str,
    iterations: usize,
    evaluations: usize,
    value: f64,
    projected: f64,
    /// Whether it stopped on the projected gradient rather than on the reduction of `f`.
    on_pg: bool,
    active: &'static str,
    /// The point, for problems of up to 25 variables.
    x: &'static [f64],
}

// Made by data/gen_lbfgsb.py (SciPy 1.18): name, iterations, evaluations, value,
// projected gradient, whether it stopped on the projected gradient, active set, x.
const SCIPY: &[Reference] = &[
    Reference {
        name: "rosen2",
        iterations: 36,
        evaluations: 44,
        value: 2.807650019071083e-12,
        projected: 5.419167105454277e-05,
        on_pg: false,
        active: "..",
        x: &[0.9999989453492616, 0.9999980209056968],
    },
    Reference {
        name: "rosen10",
        iterations: 71,
        evaluations: 88,
        value: 1.0440560289135484e-11,
        projected: 6.412752340563238e-05,
        on_pg: false,
        active: "..........",
        x: &[
            1.0000001188745589,
            1.000000131610398,
            1.0000000504907176,
            0.9999999844452054,
            0.9999999262306463,
            0.9999999128196674,
            0.9999999241988353,
            0.9999999027475234,
            0.999999836603364,
            0.9999995535618353,
        ],
    },
    Reference {
        name: "rosen50",
        iterations: 268,
        evaluations: 313,
        value: 1.495339265772242e-09,
        projected: 0.00030426240182698154,
        on_pg: false,
        active: "..................................................",
        x: &[],
    },
    Reference {
        name: "rosen100",
        iterations: 516,
        evaluations: 607,
        value: 1.9723304677851656e-10,
        projected: 9.88913080532542e-05,
        on_pg: false,
        active: "....................................................................................................",
        x: &[],
    },
    Reference {
        name: "rosen2_tight",
        iterations: 39,
        evaluations: 47,
        value: 2.180460845837453e-29,
        projected: 1.805222638040502e-13,
        on_pg: true,
        active: "..",
        x: &[0.9999999999999986, 0.9999999999999976],
    },
    Reference {
        name: "rosen10_tight",
        iterations: 80,
        evaluations: 98,
        value: 1.6732817309778482e-23,
        projected: 7.108535982070702e-11,
        on_pg: true,
        active: "..........",
        x: &[
            0.9999999999999464,
            0.9999999999999273,
            1.0000000000000258,
            1.0000000000001008,
            1.0000000000001912,
            1.0000000000002065,
            1.0000000000005038,
            1.000000000001105,
            1.0000000000020959,
            1.000000000004306,
        ],
    },
    Reference {
        name: "rosen100_tight",
        iterations: 545,
        evaluations: 639,
        value: 6.962427308172275e-23,
        projected: 4.4820813727133315e-11,
        on_pg: true,
        active: "....................................................................................................",
        x: &[],
    },
    Reference {
        name: "rosen2_box",
        iterations: 20,
        evaluations: 30,
        value: 0.25,
        projected: 0.0,
        on_pg: true,
        active: "U.",
        x: &[0.5, 0.25],
    },
    Reference {
        name: "rosen10_box",
        iterations: 28,
        evaluations: 33,
        value: 7.594812948954082,
        projected: 4.119126772739179e-05,
        on_pg: false,
        active: "U.........",
        x: &[
            0.5,
            0.26306600877570685,
            0.08003109069443862,
            0.01657444236392743,
            0.010380794327263965,
            0.010211968027398873,
            0.010208499924350847,
            0.010204162691003587,
            0.010004084423878469,
            0.00010004547029051912,
        ],
    },
    Reference {
        name: "rosen100_box",
        iterations: 25,
        evaluations: 30,
        value: 96.68553161761896,
        projected: 6.756902792082187e-05,
        on_pg: false,
        active: "U...................................................................................................",
        x: &[],
    },
    Reference {
        name: "rosen2_box_tight",
        iterations: 20,
        evaluations: 30,
        value: 0.25,
        projected: 0.0,
        on_pg: true,
        active: "U.",
        x: &[0.5, 0.25],
    },
    Reference {
        name: "rosen10_box_tight",
        iterations: 32,
        evaluations: 52,
        value: 7.594812948946873,
        projected: 1.4437882001061553e-09,
        on_pg: false,
        active: "U.........",
        x: &[
            0.5,
            0.2630659827207435,
            0.080031113752835,
            0.01657423428438209,
            0.010380676312059637,
            0.0102120051659198,
            0.010208410846125214,
            0.01020421209475974,
            0.01000408509194083,
            0.00010008171612819439,
        ],
    },
    Reference {
        name: "rosen100_box_tight",
        iterations: 28,
        evaluations: 33,
        value: 96.68553161751237,
        projected: 1.3505130068186717e-08,
        on_pg: false,
        active: "U...................................................................................................",
        x: &[],
    },
    Reference {
        name: "rosen10_box_m1",
        iterations: 23,
        evaluations: 31,
        value: 7.594812949357263,
        projected: 0.0004130607656440688,
        on_pg: false,
        active: "U.........",
        x: &[
            0.5,
            0.2630672305619186,
            0.0800301889202984,
            0.016574038240475118,
            0.01038090446635952,
            0.01021204523232535,
            0.010208382065455617,
            0.010204111665194164,
            0.010004037198510627,
            0.00010013102669248134,
        ],
    },
    Reference {
        name: "rosen10_m1",
        iterations: 123,
        evaluations: 141,
        value: 7.6534086596927e-07,
        projected: 0.001853939320079246,
        on_pg: false,
        active: "..........",
        x: &[
            1.0000030584667994,
            1.000007582950014,
            1.0000123844565914,
            1.0000235724826458,
            1.0000488252086552,
            1.0000941020961693,
            1.0001882797338633,
            1.0003778170668582,
            1.0007557859681762,
            1.001515450666128,
        ],
    },
    Reference {
        name: "rosen10_box_m3",
        iterations: 24,
        evaluations: 29,
        value: 7.594812948948594,
        projected: 1.4862466267473451e-05,
        on_pg: false,
        active: "U.........",
        x: &[
            0.5,
            0.26306593041440396,
            0.08003106606641928,
            0.01657420986817953,
            0.010380750391681032,
            0.010212044818140267,
            0.010208487409358698,
            0.01020424583922661,
            0.0100040869117614,
            0.00010009342448710237,
        ],
    },
    Reference {
        name: "rosen10_m3",
        iterations: 82,
        evaluations: 96,
        value: 3.158221695884523e-09,
        projected: 0.00042488593043421314,
        on_pg: false,
        active: "..........",
        x: &[
            0.9999995610663861,
            0.9999995870562154,
            0.9999989026037545,
            0.9999983455061118,
            0.9999964751862312,
            0.9999938870645808,
            0.9999881321756252,
            0.9999769407029017,
            0.9999549763846898,
            0.9999108238380319,
        ],
    },
    Reference {
        name: "rosen10_box_m5",
        iterations: 26,
        evaluations: 30,
        value: 7.594812948953385,
        projected: 5.0478048218227656e-05,
        on_pg: false,
        active: "U.........",
        x: &[
            0.5,
            0.263065781034204,
            0.08003110847063151,
            0.016574241145031875,
            0.010380612033759798,
            0.010211991432669182,
            0.010208482389682508,
            0.010204211647667858,
            0.01000411081176572,
            0.00010015029975593971,
        ],
    },
    Reference {
        name: "rosen10_m5",
        iterations: 75,
        evaluations: 90,
        value: 1.9340286618567872e-10,
        projected: 0.0003424956685932931,
        on_pg: false,
        active: "..........",
        x: &[
            1.0000003810884939,
            0.9999999851175281,
            1.00000043787002,
            1.0000004182889357,
            1.000000296448109,
            1.0000004731582166,
            1.0000003264298034,
            1.000000725742502,
            1.0000010260814833,
            1.0000020288180838,
        ],
    },
    Reference {
        name: "rosen10_box_m17",
        iterations: 28,
        evaluations: 32,
        value: 7.594812948958788,
        projected: 4.096345694293291e-05,
        on_pg: false,
        active: "U.........",
        x: &[
            0.5,
            0.26306612575783717,
            0.08003106841038335,
            0.016574069029460112,
            0.010380601317320907,
            0.010211884332651609,
            0.010208367877661098,
            0.010204088073185862,
            0.010004225853471254,
            0.00010002545901604653,
        ],
    },
    Reference {
        name: "rosen10_m17",
        iterations: 69,
        evaluations: 81,
        value: 8.750569820458212e-11,
        projected: 0.00019908057924464636,
        on_pg: false,
        active: "..........",
        x: &[
            0.9999998511010839,
            0.999999800958473,
            0.9999997307688505,
            0.9999998850405182,
            1.0000001025745144,
            1.0000006692429348,
            1.0000010761782074,
            1.0000020058067285,
            1.0000040106891774,
            1.00000783592545,
        ],
    },
    Reference {
        name: "driver1",
        iterations: 23,
        evaluations: 28,
        value: 1.0834900834798513e-09,
        projected: 0.0001720522728743525,
        on_pg: false,
        active: ".........................",
        x: &[
            1.0000023258771837,
            1.0000030446388726,
            1.0000045143926763,
            1.0000050772587235,
            1.00000581930634,
            1.0000070253663256,
            1.0000083438107845,
            1.0000120681556417,
            1.0000196904590828,
            1.0000374605368185,
            1.000077010855121,
            1.0001583814754373,
            1.0003231605131708,
            1.0006492825285074,
            1.0013006127613449,
            1.002601944378863,
            1.0052112931969246,
            1.010449820256067,
            1.0210095629751152,
            1.0424600004275002,
            1.086723474398011,
            1.1809686991764257,
            1.394691851289783,
            1.9451626718243045,
            3.7836626570644607,
        ],
    },
    Reference {
        name: "driver1_tight",
        iterations: 46,
        evaluations: 53,
        value: 5.807023130106546e-15,
        projected: 6.619685767315511e-11,
        on_pg: true,
        active: ".........................",
        x: &[
            1.0000000732135395,
            1.0000001555788753,
            1.0000003157333515,
            1.0000006337587275,
            1.0000012686655853,
            1.0000025379074822,
            1.0000050761076997,
            1.00001015238199,
            1.0000203049371699,
            1.0000406103219175,
            1.0000812223109523,
            1.00016245122918,
            1.000324928853361,
            1.0006499632878763,
            1.0013003490268968,
            1.0026023889590963,
            1.0052115503421177,
            1.0104502609362256,
            1.0210097298211365,
            1.0424608683836587,
            1.0867246621088944,
            1.1809704912345709,
            1.394691301166566,
            1.9451638255508477,
            3.783662308233089,
        ],
    },
    Reference {
        name: "quad5_box",
        iterations: 13,
        evaluations: 15,
        value: -0.6905206828230577,
        projected: 6.591283074897092e-12,
        on_pg: true,
        active: "..L..",
        x: &[
            0.5759295062923956,
            0.16019420212948654,
            0.0,
            0.5955712716348078,
            0.6970291921274059,
        ],
    },
    Reference {
        name: "quad20_box",
        iterations: 38,
        evaluations: 41,
        value: -2.3394744374185876,
        projected: 1.2143832273103783e-08,
        on_pg: false,
        active: "..L......L...L......",
        x: &[
            0.5759295051283855,
            0.16019420135630705,
            0.0,
            0.631181397722743,
            0.7682494436479503,
            0.10683036774199633,
            0.1452859799893337,
            0.8018787165715594,
            0.5993096397124331,
            0.0,
            0.36098804201116896,
            0.8709751054276473,
            0.3811692565576243,
            0.0,
            0.5855460428642796,
            0.8120337239299195,
            0.15924833861896495,
            0.09210029073341357,
            0.753313013156551,
            0.6411978501330303,
        ],
    },
    Reference {
        name: "quad100_box",
        iterations: 171,
        evaluations: 176,
        value: -10.94026023446505,
        projected: 1.693739792862914e-08,
        on_pg: false,
        active: "..L......L...L..........L..........L....................................L..........L..............L.",
        x: &[],
    },
    Reference {
        name: "quad20_lower",
        iterations: 42,
        evaluations: 70,
        value: -2.3394744374185894,
        projected: 1.2685669070755523e-08,
        on_pg: false,
        active: "..L......L...L......",
        x: &[
            0.5759295130095938,
            0.16019420288105005,
            0.0,
            0.6311814002018258,
            0.7682494486994432,
            0.10683038254758197,
            0.14528599986301027,
            0.8018787368500666,
            0.5993096517813449,
            0.0,
            0.36098803396354767,
            0.8709750894800884,
            0.38116924864778967,
            0.0,
            0.5855460307835265,
            0.8120337041420965,
            0.15924831866144876,
            0.09210027707045802,
            0.7533130048827348,
            0.641197847713743,
        ],
    },
    Reference {
        name: "quad20_upper",
        iterations: 33,
        evaluations: 35,
        value: -2.2681064338670627,
        projected: 8.973959797486941e-09,
        on_pg: false,
        active: "U..........U...U..UU",
        x: &[
            0.25,
            -0.3027532109587829,
            -0.599965315718227,
            0.02863726481300318,
            0.1631264980031485,
            -0.5008713904041562,
            -0.46499458874994015,
            0.18901932519243092,
            -0.016128574924712476,
            -0.6180170467420121,
            -0.2585080355680852,
            0.25,
            -0.2276120773796893,
            -0.5965875617174919,
            0.006235397684587415,
            0.25,
            -0.3832118178916105,
            -0.4307862926651028,
            0.25,
            0.25,
        ],
    },
    Reference {
        name: "quad20_free",
        iterations: 60,
        evaluations: 66,
        value: -2.3637100817676733,
        projected: 1.7731045631741438e-08,
        on_pg: false,
        active: "....................",
        x: &[
            0.4584255547394645,
            -0.07481370896231415,
            -0.35251186341045965,
            0.2956046554264032,
            0.44960782007257705,
            -0.194876130113192,
            -0.13948541043646437,
            0.5340424235004719,
            0.3484084325775243,
            -0.23396613032478808,
            0.1450568014807738,
            0.6730787531647122,
            0.2013078094757766,
            -0.16182653768563784,
            0.4468375658413717,
            0.6964433216996804,
            0.06677601208016791,
            0.02274604568738637,
            0.7070768467303777,
            0.61807976388268,
        ],
    },
    Reference {
        name: "quad20_interior",
        iterations: 60,
        evaluations: 66,
        value: -2.3637100817676755,
        projected: 4.218245308607038e-09,
        on_pg: false,
        active: "....................",
        x: &[
            0.4584255341385923,
            -0.07481374240520214,
            -0.35251191968333,
            0.29560458464614575,
            0.44960773979338897,
            -0.1948762164791482,
            -0.13948548722404155,
            0.534042354493109,
            0.3484083771355185,
            -0.23396617649991172,
            0.14505676302284018,
            0.6730787276105454,
            0.20130779340093802,
            -0.16182654594424611,
            0.446837562359217,
            0.6964433150505059,
            0.06677601027529126,
            0.022746046985172165,
            0.7070768524251866,
            0.6180797697845548,
        ],
    },
    Reference {
        name: "quad20_outside",
        iterations: 41,
        evaluations: 57,
        value: -2.3394744374185894,
        projected: 6.935250151940409e-10,
        on_pg: false,
        active: "..L......L...L......",
        x: &[
            0.5759295063528114,
            0.16019420207282592,
            0.0,
            0.6311813976127278,
            0.7682494445982171,
            0.10683037857018911,
            0.1452860007614553,
            0.8018787350362685,
            0.5993096541310963,
            0.0,
            0.3609880315027575,
            0.870975089051266,
            0.38116924685990955,
            0.0,
            0.5855460338646712,
            0.8120337137893413,
            0.15924833172839425,
            0.09210028985385271,
            0.753313015117382,
            0.6411978505484539,
        ],
    },
    Reference {
        name: "vertex6",
        iterations: 1,
        evaluations: 2,
        value: -32.25,
        projected: 0.0,
        on_pg: true,
        active: "LULULU",
        x: &[-1.0, 2.0, -1.0, 2.0, -1.0, 2.0],
    },
    Reference {
        name: "scaled10_1e6",
        iterations: 21,
        evaluations: 24,
        value: 6702.417178665994,
        projected: 5.72393150966838e-08,
        on_pg: false,
        active: "UU.LL.UU.L",
        x: &[
            0.5,
            0.5,
            0.1411200078905277,
            -0.5,
            -0.5,
            -0.279415498225494,
            0.5,
            0.5,
            0.4121184852415064,
            -0.5,
        ],
    },
    Reference {
        name: "scaled10_1e12",
        iterations: 22,
        evaluations: 30,
        value: 1228147213.6199365,
        projected: 0.677650794599264,
        on_pg: false,
        active: "...LL.UU.L",
        x: &[
            0.002026500005123633,
            0.04712165721274452,
            0.15344167446291881,
            -0.5,
            -0.5,
            -0.279415492581159,
            0.5,
            0.5,
            0.41211848525635614,
            -0.5,
        ],
    },
    Reference {
        name: "brown_box",
        iterations: 7,
        evaluations: 11,
        value: 1.0533643395144706e-10,
        projected: 2.000010233059077e-06,
        on_pg: true,
        active: "..",
        x: &[999999.999999212, 2.000010233059077e-06],
    },
    Reference {
        name: "brown_box_cut",
        iterations: 3,
        evaluations: 4,
        value: 250000000000.0,
        projected: 1.3355774043964853e-10,
        on_pg: true,
        active: "U.",
        x: &[500000.0, 3.999999999992e-06],
    },
    Reference {
        name: "hs5",
        iterations: 8,
        evaluations: 10,
        value: -1.9132229549810367,
        projected: 3.857802965967494e-12,
        on_pg: true,
        active: "..",
        x: &[-0.5471975511987035, -1.547197551198777],
    },
    Reference {
        name: "hs45",
        iterations: 4,
        evaluations: 10,
        value: 1.0,
        projected: 0.0,
        on_pg: true,
        active: "UUUUU",
        x: &[1.0, 2.0, 3.0, 4.0, 5.0],
    },
    Reference {
        name: "hs110",
        iterations: 6,
        evaluations: 13,
        value: -45.778469707446305,
        projected: 2.0202806538804907e-08,
        on_pg: false,
        active: "..........",
        x: &[
            9.350265836156803,
            9.350265836156803,
            9.350265836156803,
            9.350265836156803,
            9.350265836156803,
            9.350265836156803,
            9.350265836156803,
            9.350265836156803,
            9.350265836156803,
            9.350265836156803,
        ],
    },
    Reference {
        name: "nearly_linear6",
        iterations: 4,
        evaluations: 32,
        value: -8.750000000000005e+16,
        projected: 3.9547387586935656e-10,
        on_pg: false,
        active: ".U.U.U",
        x: &[
            -5000000000062409.0,
            2.0,
            -1.5000000000742336e+16,
            2.0,
            -2.5000000001977372e+16,
            2.0,
        ],
    },
    Reference {
        name: "review_quad256",
        iterations: 7,
        evaluations: 10,
        value: -56.53410099001353,
        projected: 7.105427357601002e-15,
        on_pg: true,
        active: "L.",
        x: &[1.0024952564033676, 1.1747741724387628],
    },
    Reference {
        name: "review_quad160",
        iterations: 7,
        evaluations: 10,
        value: -48857.88100158316,
        projected: 1.4551915228366852e-11,
        on_pg: true,
        active: "U.ULU",
        x: &[
            0.09394305744467335,
            0.8718850405583091,
            -0.6316374161751908,
            0.2159848323577039,
            0.5726591865954651,
        ],
    },
    Reference {
        name: "review_quad37",
        iterations: 23,
        evaluations: 60,
        value: -220540.62445072003,
        projected: 2.6637091650627553e-05,
        on_pg: false,
        active: ".L..U",
        x: &[
            -0.8014467569244152,
            -1.0924262194088137,
            -0.3415966210693581,
            -1.0855289452758634,
            -1.1091653069354457,
        ],
    },
];
// Made by data/gen_lbfgsb.py: the KKT points of the boxed quadratics by BVLS.
const KKT: &[(usize, &[f64])] = &[
    (
        5,
        &[
            0.5759295062927018,
            0.16019420213293534,
            0.0,
            0.5955712716335687,
            0.6970291921285299,
        ],
    ),
    (
        20,
        &[
            0.5759295062927018,
            0.16019420213293534,
            0.0,
            0.6311813979582048,
            0.7682494447778028,
            0.10683037897390939,
            0.14528600076355874,
            0.8018787347902406,
            0.5993096539604272,
            0.0,
            0.3609880318546686,
            0.8709750895235318,
            0.38116924704972527,
            0.0,
            0.5855460334638743,
            0.8120337129055802,
            0.15924833069656183,
            0.09210028846184083,
            0.7533130140587152,
            0.6411978500893698,
        ],
    ),
    (
        100,
        &[
            0.5759295062927018,
            0.16019420213293534,
            0.0,
            0.631181397958205,
            0.768249444777803,
            0.10683037897390946,
            0.14528600076355883,
            0.8018787347902413,
            0.5993096539604272,
            0.0,
            0.3609880318546686,
            0.8709750895235318,
            0.38116924704972516,
            0.0,
            0.5904554665051229,
            0.821852578988078,
            0.17397662982030754,
            0.11173802062683513,
            0.7778601792649582,
            0.6706544483368612,
            0.03436603128874015,
            0.30774428607414434,
            0.8757941423598063,
            0.4582436998548358,
            0.0,
            0.5369473604760951,
            0.857907454763973,
            0.2384385861048437,
            0.07729517193748167,
            0.7384752979036107,
            0.7294262480263648,
            0.0707643364629185,
            0.2497296820464992,
            0.8624605686057643,
            0.5257875944783283,
            0.0,
            0.4722884765821039,
            0.8764827584797089,
            0.3000875486017796,
            0.04447364776699632,
            0.6867874276215028,
            0.7769338225681303,
            0.11143999177533706,
            0.1928337113999191,
            0.8374029593056178,
            0.5899605245119466,
            0.009204149763364757,
            0.40635078978443034,
            0.8848165309442865,
            0.36442422040890454,
            0.020107529822111638,
            0.6292761425104593,
            0.8166664718128281,
            0.15925911709545387,
            0.14132510652317345,
            0.8034035406869471,
            0.6507769115922639,
            0.028078361512596496,
            0.3412172325750535,
            0.8832730260827188,
            0.43050202823197586,
            0.005170017624459372,
            0.5675940735046828,
            0.8480452803937893,
            0.21340171906181626,
            0.09654085200881464,
            0.7613478543781939,
            0.7069449597830323,
            0.056206666698950795,
            0.27834733627706315,
            0.8718921072932659,
            0.4968511683891234,
            0.0,
            0.5026947596958685,
            0.8695172373611734,
            0.2713776412041219,
            0.05777042790963123,
            0.7100355276171323,
            0.75464248387559,
            0.08996393952654202,
            0.21571860190038328,
            0.847072831292527,
            0.5577064129047625,
            0.0,
            0.43212565073181275,
            0.8775129778369558,
            0.3296140074049159,
            0.024412301993255996,
            0.649956481361866,
            0.7929604300258082,
            0.1295637979142635,
            0.15650794656666367,
            0.8119594584163282,
            0.6151107627577101,
            0.009383082350560155,
            0.36131922148767487,
            0.8753549243047146,
            0.38995553496176133,
            0.0,
            0.17332472774851493,
        ],
    ),
];
// Made by data/gen_lbfgsb.py: the box quadratic tridiag(-1, 3, -1) of the review at
// n, pgtol and ftol 0: iterations, whether it stopped on the projected gradient, and
// the projected gradient there.
const LEVEL: &[(usize, f64, usize, bool, f64)] = &[
    (1000, 1e-09, 11, false, 1.6936238300679918e-08),
    (10000, 1e-07, 11, true, 7.403797763760878e-08),
];

/// SciPy's iterates on one problem: after each iteration the evaluations made so far, the first
/// included, and the point.
type Iterates = &'static [(usize, &'static [f64])];

// Made by data/gen_lbfgsb.py: SciPy's first iterates, each the evaluations made so far
// (the first included) and the point.
const EARLY: &[(&str, Iterates)] = &[
    (
        "rosen10",
        &[
            (
                2,
                &[
                    -1.0958165798604025,
                    0.6172853954055598,
                    -0.8831973550857134,
                    0.6172853954055598,
                    -0.8831973550857134,
                    0.6172853954055598,
                    -0.8831973550857134,
                    0.6172853954055598,
                    -0.8831973550857134,
                    1.0425238449549379,
                ],
            ),
            (
                3,
                &[
                    -0.8160055792889489,
                    0.4833161973181781,
                    -0.5924679585528179,
                    0.38437177993513705,
                    -0.5924679585528179,
                    0.38437177993513705,
                    -0.5924679585528179,
                    0.38437177993513705,
                    -0.769096220112943,
                    0.9702529525485539,
                ],
            ),
            (
                4,
                &[
                    -0.7144813936378973,
                    0.276959521634827,
                    -0.28026139699202696,
                    0.2033088441116496,
                    -0.3300114276481876,
                    0.2033088441116496,
                    -0.3300114276481876,
                    0.12462597962732297,
                    -0.6878588994342631,
                    0.7492317226191676,
                ],
            ),
            (
                5,
                &[
                    -0.4816368169976498,
                    0.3105780509742027,
                    -0.030078655554918454,
                    0.013375522510089333,
                    -0.07907070343959949,
                    0.039692495879926876,
                    -0.03405565790589643,
                    -0.020838781722389288,
                    -0.4246384782952007,
                    0.45041652072612665,
                ],
            ),
            (
                6,
                &[
                    -0.44326642082820517,
                    0.1886856241659785,
                    0.19557867296368203,
                    -0.042357271587493625,
                    0.09433101378397613,
                    -0.031137693439309083,
                    0.11379466217914797,
                    0.004591687230670369,
                    -0.12326413071086084,
                    0.053761078501798454,
                ],
            ),
        ],
    ),
    (
        "rosen10_box",
        &[
            (
                3,
                &[
                    -0.49928508577616254,
                    -0.2365557309832425,
                    -0.49928508577616254,
                    -0.2365557309832425,
                    -0.49928508577616254,
                    -0.2365557309832425,
                    -0.49928508577616254,
                    -0.2365557309832425,
                    -0.49928508577616254,
                    1.412185243661081,
                ],
            ),
            (
                4,
                &[
                    -0.3864966042018922,
                    -0.07007799823559084,
                    -0.21339448183109258,
                    -0.07007799823559084,
                    -0.21339448183109258,
                    -0.07007799823559084,
                    -0.21339448183109258,
                    -0.07007799823559084,
                    -0.6148767963148933,
                    1.1198807426896744,
                ],
            ),
            (
                5,
                &[
                    -0.3135865807404714,
                    0.027832831756737654,
                    -0.08724735827176841,
                    -0.028468927197824605,
                    -0.08724735827176841,
                    -0.028468927197824605,
                    -0.08724735827176841,
                    0.0020398825247423397,
                    -0.7221711690281033,
                    0.7601655046064196,
                ],
            ),
            (
                6,
                &[
                    -0.263766776661514,
                    0.10902630391503282,
                    -0.006636992756492245,
                    0.01975233515711107,
                    -0.006618498678368896,
                    0.01975233515711107,
                    -0.00936763317678499,
                    0.04140434306155127,
                    -0.5641529480031102,
                    0.5140476531125873,
                ],
            ),
            (
                7,
                &[
                    -0.23544751769379582,
                    0.1829062306405276,
                    0.13233110867156328,
                    0.13190408396668507,
                    0.09909436866135624,
                    0.13158998067523398,
                    0.10702761835766969,
                    0.013921096003381104,
                    -0.04994643583265157,
                    -0.2967521593729255,
                ],
            ),
        ],
    ),
    (
        "rosen10_box_m3",
        &[
            (
                3,
                &[
                    -0.4992850857761625,
                    -0.23655573098324267,
                    -0.4992850857761625,
                    -0.23655573098324267,
                    -0.4992850857761625,
                    -0.23655573098324267,
                    -0.4992850857761625,
                    -0.23655573098324267,
                    -0.4992850857761625,
                    1.412185243661081,
                ],
            ),
            (
                4,
                &[
                    -0.38649660420189225,
                    -0.07007799823559087,
                    -0.21339448183109264,
                    -0.07007799823559087,
                    -0.21339448183109264,
                    -0.07007799823559087,
                    -0.21339448183109264,
                    -0.07007799823559087,
                    -0.6148767963148932,
                    1.1198807426896746,
                ],
            ),
            (
                5,
                &[
                    -0.3135865807404714,
                    0.0278328317567377,
                    -0.08724735827176844,
                    -0.02846892719782458,
                    -0.08724735827176844,
                    -0.02846892719782458,
                    -0.08724735827176844,
                    0.0020398825247423466,
                    -0.7221711690281034,
                    0.7601655046064199,
                ],
            ),
            (
                6,
                &[
                    -0.263766776661514,
                    0.10902630391503282,
                    -0.006636992756492328,
                    0.019752335157111067,
                    -0.0066184986783689725,
                    0.019752335157111067,
                    -0.009367633176785059,
                    0.041404343061551246,
                    -0.5641529480031104,
                    0.5140476531125877,
                ],
            ),
            (
                7,
                &[
                    -0.2230730387691757,
                    0.14684661272121913,
                    0.17319877184751484,
                    0.09455606410517291,
                    0.13996094165551498,
                    0.09424196081372176,
                    0.14805624627319544,
                    -0.02115346063387493,
                    -0.0333062774090597,
                    -0.304110904722972,
                ],
            ),
        ],
    ),
    (
        "driver1",
        &[
            (
                5,
                &[
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    2.976798033474382,
                    1.8050987239306717,
                    3.5568471966148496,
                ],
            ),
            (
                6,
                &[
                    1.7249925411158782,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.606407535342302,
                    1.7738962924578974,
                    1.6598169431200132,
                    3.677864571314957,
                ],
            ),
            (
                7,
                &[
                    1.5490927249652642,
                    1.5070914573694187,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6030075521609566,
                    1.5130001184514568,
                    1.6095507905766182,
                    1.7612057588895402,
                    3.6791718403910183,
                ],
            ),
            (
                8,
                &[
                    1.263437360321293,
                    1.3133529150108303,
                    1.3215725847418902,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.3279926182821882,
                    1.3254370432605087,
                    1.33228048384865,
                    1.4711199759444922,
                    1.9857912485523546,
                    3.6795769281712327,
                ],
            ),
            (
                9,
                &[
                    1.1111887600670236,
                    1.195474637789202,
                    1.1609525954931823,
                    1.215164139006052,
                    1.166546945244613,
                    1.2169877270002956,
                    1.166546945244613,
                    1.2169877270002956,
                    1.166546945244613,
                    1.2169877270002956,
                    1.166546945244613,
                    1.2169877270002956,
                    1.166546945244613,
                    1.2169877270002956,
                    1.166546945244613,
                    1.2169877270002956,
                    1.166546945244613,
                    1.2169877270002956,
                    1.166546945244613,
                    1.2169877270002956,
                    1.1685732875613715,
                    1.2846633302480055,
                    1.4281107596209448,
                    1.8699264778710705,
                    3.7539284049178585,
                ],
            ),
        ],
    ),
    (
        "quad20_box",
        &[
            (
                2,
                &[
                    0.9916648104524681,
                    0.24445889797316903,
                    0.0,
                    0.9941133511386077,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.8967405731306133,
                    0.0,
                    0.35100097418580134,
                    1.0,
                    0.3913634045759239,
                    0.0,
                    0.8590583540221679,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.5290826861200238,
                ],
            ),
            (
                3,
                &[
                    0.6877601263988763,
                    0.36201529867604276,
                    0.11906256601846996,
                    0.762194136731454,
                    0.8925416536090811,
                    0.1483616770961161,
                    0.18366194522982143,
                    0.8908777751271933,
                    0.7105246738195161,
                    0.1063195166207074,
                    0.42093581187968643,
                    0.907430791191033,
                    0.4423535545254481,
                    0.10259468795916922,
                    0.6905291467071698,
                    0.8887788000092708,
                    0.1976977258381285,
                    0.13605930250162115,
                    0.7485863502989709,
                    0.6585300861212323,
                ],
            ),
            (
                4,
                &[
                    0.670418077186284,
                    0.29728237732682017,
                    0.1079503693421342,
                    0.7477028757706259,
                    0.8581045066712711,
                    0.18512219861385246,
                    0.21099085184430566,
                    0.8752510729705955,
                    0.6940934373772836,
                    0.09379700259947853,
                    0.432130939767641,
                    0.9224936027285742,
                    0.4505160278868589,
                    0.08938403130672472,
                    0.673248842817001,
                    0.8803657887926416,
                    0.22139364162600794,
                    0.11960287362688468,
                    0.7690170369951247,
                    0.6457186929006469,
                ],
            ),
            (
                5,
                &[
                    0.6310724746761028,
                    0.22714021467502327,
                    0.05392670578329319,
                    0.7202564418575962,
                    0.8473098512934981,
                    0.2048218150884122,
                    0.23698896173636771,
                    0.8733454751850028,
                    0.6725222715051723,
                    0.07668268778533646,
                    0.4401489717912424,
                    0.9485878434889453,
                    0.45636205067986374,
                    0.06980836036724783,
                    0.6537663935045246,
                    0.881830004585559,
                    0.22075640167293697,
                    0.12310502024948641,
                    0.77636250855433,
                    0.6459183451217543,
                ],
            ),
            (
                6,
                &[
                    0.5503706789682394,
                    0.10733897786108304,
                    0.0,
                    0.634248485855661,
                    0.8422619193872786,
                    0.21373774954233232,
                    0.2413731049934529,
                    0.8859550187721481,
                    0.6633682542978657,
                    0.06042286502088823,
                    0.4479996914097358,
                    0.9564570242225773,
                    0.4620327080410372,
                    0.05585194807006874,
                    0.6433337007698242,
                    0.8471204896277412,
                    0.22675294911794677,
                    0.12311933890013604,
                    0.7919634641047569,
                    0.6499740172168507,
                ],
            ),
        ],
    ),
    (
        "quad20_lower",
        &[
            (
                2,
                &[
                    0.6948224054283784,
                    0.398741721749682,
                    0.3018747719110836,
                    0.6957926407922311,
                    0.8164008826291786,
                    0.3018747719110836,
                    0.3018747719110836,
                    0.8404432610674613,
                    0.6572086330872605,
                    0.3018747719110836,
                    0.44095906805107105,
                    0.8961683927648911,
                    0.45695269950560313,
                    0.3018747719110836,
                    0.642277036575746,
                    0.8484123517839792,
                    0.3018747719110836,
                    0.3018747719110836,
                    0.8064315290156556,
                    0.5115240276419362,
                ],
            ),
            (
                3,
                &[
                    0.7032421957714168,
                    0.36432376710332925,
                    0.11152076495806787,
                    0.8224408463075932,
                    0.9171502787859258,
                    0.21832681957942637,
                    0.2652660877178467,
                    0.9282193464945303,
                    0.7843316041473326,
                    0.09865990374715779,
                    0.528297823642686,
                    0.9690529264143033,
                    0.5494475302414338,
                    0.09490063836607279,
                    0.7688666566181523,
                    0.9307375073972178,
                    0.28307541750044124,
                    0.2011139301354417,
                    0.8049929852101168,
                    0.6798290696269392,
                ],
            ),
            (
                4,
                &[
                    0.6877932535845184,
                    0.31549924922609834,
                    0.118833542625327,
                    0.7894111461423978,
                    0.919170660414494,
                    0.23059694949233694,
                    0.26494348722995537,
                    0.9433508052084333,
                    0.7449685182621014,
                    0.13891214192137558,
                    0.4913047706370523,
                    1.008451235839621,
                    0.5108531407059406,
                    0.1358624141349876,
                    0.7272888648124647,
                    0.9506789285480809,
                    0.27809035991229064,
                    0.18760520791940438,
                    0.8157968170125358,
                    0.676630591205604,
                ],
            ),
            (
                5,
                &[
                    0.6643040355489962,
                    0.285466463789029,
                    0.10117556857689672,
                    0.7707347379142429,
                    0.9123582191127809,
                    0.23908362401707375,
                    0.2731119627716825,
                    0.9388607827680442,
                    0.7359010776993652,
                    0.1420351340138555,
                    0.49155727064194027,
                    1.0078592436776737,
                    0.5102754161147895,
                    0.13831920839431933,
                    0.7185552153419265,
                    0.9470712773881953,
                    0.2773892070211376,
                    0.18243759338403207,
                    0.8167124267942951,
                    0.6754789859458608,
                ],
            ),
            (
                6,
                &[
                    0.5553057840709765,
                    0.13261497688141383,
                    0.0,
                    0.6793153516811425,
                    0.8793436360928726,
                    0.25897598292552026,
                    0.2955070220269456,
                    0.9347641320361864,
                    0.7143514939305037,
                    0.1389841927296424,
                    0.49173379969924086,
                    1.0148316793848084,
                    0.5074578156255035,
                    0.13345256941287328,
                    0.6975482834687458,
                    0.9234526281124691,
                    0.2543022690636236,
                    0.17732153975793205,
                    0.8050745234084461,
                    0.6760282889963811,
                ],
            ),
        ],
    ),
    (
        "quad20_outside",
        &[
            (
                2,
                &[
                    0.9916648104524686,
                    0.7444588979731688,
                    0.07418531767226755,
                    1.0,
                    1.0,
                    0.3001253124064577,
                    0.38186288776296673,
                    1.0,
                    1.0,
                    0.03860250812044319,
                    0.8510009741858012,
                    1.0,
                    0.8913634045759238,
                    0.028201554256136685,
                    1.0,
                    1.0,
                    0.41436266002570243,
                    0.27163923216840413,
                    1.0,
                    0.5290826861200238,
                ],
            ),
            (
                3,
                &[
                    0.927858157609551,
                    0.5764859385399776,
                    0.2764740262284701,
                    0.9442066145591388,
                    1.0,
                    0.3429173076046099,
                    0.3797661512622123,
                    1.0,
                    0.898698805155861,
                    0.27882188941649216,
                    0.6450664707262374,
                    1.0,
                    0.6710272064823424,
                    0.28189753221793695,
                    0.881960911022955,
                    1.0,
                    0.39545782081294084,
                    0.33111540232973985,
                    0.977388562702638,
                    0.6656622105803217,
                ],
            ),
            (
                4,
                &[
                    0.8217346779519834,
                    0.46622087993711214,
                    0.25104354783362487,
                    0.8969143310309131,
                    1.0,
                    0.29595260602295814,
                    0.3228972381507009,
                    1.0,
                    0.8438291075966025,
                    0.2502076702156076,
                    0.5521081735824542,
                    1.0,
                    0.576173578007173,
                    0.25190555241956036,
                    0.8244199470929774,
                    1.0,
                    0.3350521542022638,
                    0.2803933251934562,
                    0.9121391437623219,
                    0.709107837854433,
                ],
            ),
            (
                5,
                &[
                    0.6939327967576637,
                    0.3388240935459677,
                    0.17668268386082853,
                    0.843159656115166,
                    0.9760234939740764,
                    0.25948422448627817,
                    0.28386531446708974,
                    1.0,
                    0.7874351025428152,
                    0.1785840943808311,
                    0.48924633898256914,
                    1.0,
                    0.5103470992749174,
                    0.17668760244135473,
                    0.7670627163012683,
                    1.0,
                    0.2910899687393297,
                    0.20478761306257276,
                    0.848430421125336,
                    0.7158951964628184,
                ],
            ),
            (
                6,
                &[
                    0.5951518824364948,
                    0.2118724300396872,
                    0.07916604119943552,
                    0.7757813790849816,
                    0.9361418665291654,
                    0.25042992596648317,
                    0.2939568294592946,
                    0.9571498263941847,
                    0.74473978909616,
                    0.12693713132240128,
                    0.4584680844079044,
                    1.0,
                    0.475579792341713,
                    0.12107399212435493,
                    0.7227417505649791,
                    0.9608592796568909,
                    0.2696560110127569,
                    0.16458973150802006,
                    0.8013550890910377,
                    0.6910552303552187,
                ],
            ),
        ],
    ),
    ("vertex6", &[(2, &[-1.0, 2.0, -1.0, 2.0, -1.0, 2.0])]),
    (
        "scaled10_1e6",
        &[
            (2, &[0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, -0.5]),
            (
                3,
                &[
                    0.5,
                    0.5,
                    0.4967799613631854,
                    -0.5,
                    -0.5,
                    -0.49622275764446344,
                    0.5,
                    0.5,
                    0.47488975758606483,
                    -0.5,
                ],
            ),
            (
                4,
                &[
                    0.5,
                    0.5,
                    0.4883918102234624,
                    -0.5,
                    -0.5,
                    -0.48439461843818926,
                    0.5,
                    0.5,
                    0.41180568933631484,
                    -0.5,
                ],
            ),
            (
                7,
                &[
                    0.5,
                    0.5,
                    0.4819764302323729,
                    -0.5,
                    -0.5,
                    -0.43548627749635443,
                    0.5,
                    0.5,
                    0.4102982527609162,
                    -0.5,
                ],
            ),
            (
                8,
                &[
                    0.5,
                    0.5,
                    0.4619869515956936,
                    -0.5,
                    -0.5,
                    -0.28061575502962166,
                    0.5,
                    0.5,
                    0.4115925101839596,
                    -0.5,
                ],
            ),
        ],
    ),
    (
        "brown_box",
        &[
            (3, &[500001.0, 1.0]),
            (4, &[1000000.9999919999, 0.0]),
            (6, &[999999.8547052635, 1.913014886235688e-06]),
            (7, &[999999.9705518151, 1.9335159789219796e-06]),
            (8, &[999999.9997812292, 2.0068573769565757e-06]),
        ],
    ),
    (
        "hs45",
        &[
            (
                2,
                &[
                    1.0,
                    2.0,
                    2.066666666666667,
                    2.066666666666667,
                    2.066666666666667,
                ],
            ),
            (5, &[1.0, 2.0, 3.0, 3.0, 3.0]),
            (8, &[1.0, 2.0, 3.0, 4.0, 4.0]),
            (10, &[1.0, 2.0, 3.0, 4.0, 5.0]),
        ],
    ),
    (
        "hs110",
        &[
            (
                3,
                &[
                    9.338387130609771,
                    9.338387130609771,
                    9.338387130609771,
                    9.338387130609771,
                    9.338387130609771,
                    9.338387130609771,
                    9.338387130609771,
                    9.338387130609771,
                    9.338387130609771,
                    9.338387130609771,
                ],
            ),
            (
                4,
                &[
                    9.360338306526604,
                    9.360338306526604,
                    9.360338306526604,
                    9.360338306526604,
                    9.360338306526604,
                    9.360338306526604,
                    9.360338306526604,
                    9.360338306526604,
                    9.360338306526604,
                    9.360338306526604,
                ],
            ),
            (
                5,
                &[
                    9.35000817415825,
                    9.35000817415825,
                    9.35000817415825,
                    9.35000817415825,
                    9.35000817415825,
                    9.35000817415825,
                    9.35000817415825,
                    9.35000817415825,
                    9.35000817415825,
                    9.35000817415825,
                ],
            ),
            (
                6,
                &[
                    9.350260266321788,
                    9.350260266321788,
                    9.350260266321788,
                    9.350260266321788,
                    9.350260266321788,
                    9.350260266321788,
                    9.350260266321788,
                    9.350260266321788,
                    9.350260266321788,
                    9.350260266321788,
                ],
            ),
            (
                7,
                &[
                    9.350265836156803,
                    9.350265836156803,
                    9.350265836156803,
                    9.350265836156803,
                    9.350265836156803,
                    9.350265836156803,
                    9.350265836156803,
                    9.350265836156803,
                    9.350265836156803,
                    9.350265836156803,
                ],
            ),
        ],
    ),
    (
        "review_quad256",
        &[
            (2, &[2.2368040876881636, -0.5179347280712792]),
            (3, &[2.1022891207531136, 5.041661124895976]),
            (5, &[1.7032586121662785, 3.7141916689108916]),
            (7, &[1.5429182385114613, 2.7679192536546573]),
            (8, &[1.0024952564033676, 1.2651082472752806]),
        ],
    ),
    (
        "review_quad160",
        &[
            (
                2,
                &[
                    0.09394305744467335,
                    -2.2787311963561137,
                    -0.6316374161751908,
                    0.2159848323577039,
                    -0.7822377404609863,
                ],
            ),
            (
                3,
                &[
                    0.09394305744467335,
                    1.9263442353512823,
                    -0.6316374161751908,
                    0.2159848323577039,
                    -0.6601137564717484,
                ],
            ),
            (
                5,
                &[
                    0.09394305744467335,
                    1.4818952576755147,
                    -0.6316374161751908,
                    0.2159848323577039,
                    -0.0831237053053917,
                ],
            ),
            (
                7,
                &[
                    0.09394305744467335,
                    1.1079382839334457,
                    -0.6316374161751908,
                    0.2159848323577039,
                    0.08648271043657392,
                ],
            ),
            (
                8,
                &[
                    0.09394305744467335,
                    0.996135849193032,
                    -0.6316374161751908,
                    0.2159848323577039,
                    0.5726591865954651,
                ],
            ),
        ],
    ),
    (
        "review_quad37",
        &[
            (
                3,
                &[
                    -1.5189678366275334,
                    -0.9541058594530578,
                    -1.315055798981398,
                    -1.9437374012003956,
                    -1.1133926531063836,
                ],
            ),
            (
                4,
                &[
                    -1.5042010119230422,
                    -0.9531266643890874,
                    -1.2893539458514407,
                    -1.9437434483263005,
                    -1.1091653069354457,
                ],
            ),
            (
                5,
                &[
                    -1.2178412743519225,
                    -0.9526940492927887,
                    -0.8396059771709251,
                    -1.9437434483263005,
                    -1.1091653069354457,
                ],
            ),
            (
                6,
                &[
                    -1.1578369804024664,
                    -0.9716564501665294,
                    -0.7830909944020832,
                    -1.8718404483047404,
                    -1.1283637999485787,
                ],
            ),
            (
                7,
                &[
                    -0.7043513787684005,
                    -1.0924262194088137,
                    -0.4131879272908514,
                    -1.2177584906069407,
                    -1.2876592940964886,
                ],
            ),
        ],
    ),
];

fn reference(name: &str) -> &'static Reference {
    SCIPY
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no reference for {name}"))
}

fn problem(name: &str) -> Problem {
    problems()
        .into_iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no problem {name}"))
}

/// The active set of the true minimiser of `scaled10_1e12`, `clamp(sin(i + 1), −½, ½)`: SciPy
/// stops on it at a point that is not, two of its variables in the interior (its "reduction"
/// test passes on a step that did not move), and reports success.
const SCALED_ACTIVE: &str = "UU.LL.UU.L";

#[test]
fn every_reference_problem_ends_where_scipys_lbfgsb_does() {
    let mut wrong = Vec::new();
    for p in problems() {
        let want = reference(p.name);
        let r = p.run();
        let name = p.name;
        let n = p.x0.len();
        let x = r.x.as_slice();
        let mut check = |ok: bool, what: String| {
            if !ok {
                wrong.push(format!("{name}: {what}"));
            }
        };

        // What the result says of itself is so.
        let kkt = p.kkt(x);
        let width = x.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        check(
            (kkt - r.projected_gradient).abs() <= 4.0 * f64::EPSILON * width,
            format!(
                "reported projected gradient {} against {kkt}",
                r.projected_gradient
            ),
        );
        check(r.value == (p.f)(x), "the value is not f at x".into());
        check(
            r.gradient.as_slice() == (p.g)(x).as_slice(),
            "the gradient is not g at x".into(),
        );
        check(
            r.converged
                == matches!(
                    r.stop,
                    LbfgsbStop::ProjectedGradient | LbfgsbStop::Reduction
                ),
            format!("converged is {} for {:?}", r.converged, r.stop),
        );
        if r.stop == LbfgsbStop::ProjectedGradient {
            check(r.projected_gradient <= p.pgtol, format!("{r:?}"));
        }
        check(
            (0..n).all(|i| p.lower[i] <= x[i] && x[i] <= p.upper[i]),
            "x is outside the box".into(),
        );

        // The same active set, and the same verdict where SciPy's is earned.
        let expected = if name == "scaled10_1e12" {
            SCALED_ACTIVE
        } else {
            want.active
        };
        check(
            active(&r) == expected,
            format!("active set {} against {expected}", active(&r)),
        );
        if want.on_pg {
            check(
                r.stop == LbfgsbStop::ProjectedGradient,
                format!("stopped on {:?}", r.stop),
            );
        } else if p.ftol > 0.0 {
            check(r.converged, format!("stopped on {:?}", r.stop));
        } else {
            // SciPy's "reduction" with ftol = 0 is a step that left f where it was, the limit
            // of the arithmetic, which it reports as success and this method does not. It must
            // have come as close to the first-order conditions as SciPy did, all the same.
            check(
                r.converged || r.stop == LbfgsbStop::LineSearch,
                format!("stopped on {:?}", r.stop),
            );
            check(
                r.projected_gradient <= 100.0 * want.projected.max(p.pgtol),
                format!(
                    "projected gradient {:e} against SciPy's {:e}",
                    r.projected_gradient, want.projected
                ),
            );
        }

        // Never a worse value than SciPy's, beyond what the tolerances leave undecided in f: the
        // reduction test the share ftol of it, the gradient test about pgtol².
        let slack = 10.0 * p.ftol.max(f64::EPSILON) * want.value.abs().max(1.0) + p.pgtol * p.pgtol;
        check(
            r.value <= want.value + slack,
            format!("value {:e} against SciPy's {:e}", r.value, want.value),
        );

        // Iterations in the same class, and evaluations too where both converged.
        let (ours, theirs) = (r.iterations as f64, want.iterations as f64);
        check(
            ours <= 1.3 * theirs + 3.0 && ours >= 0.7 * theirs - 3.0,
            format!(
                "{} iterations against SciPy's {}",
                r.iterations, want.iterations
            ),
        );
        if r.converged && (want.on_pg || p.ftol > 0.0) {
            let (ours, theirs) = (r.evaluations as f64, want.evaluations as f64);
            check(
                ours <= 1.3 * theirs + 5.0 && ours >= 0.7 * theirs - 5.0,
                format!(
                    "{} evaluations against SciPy's {}",
                    r.evaluations, want.evaluations
                ),
            );
        }

        // The same point: closely where both stopped on the projected gradient, to the accuracy
        // the reduction test leaves where they did not. The driver problem is left out: along
        // its chain xᵢ = xᵢ₋₁² an error of 10⁻⁷ in x₀ is doubled 24 times without changing f.
        if !want.x.is_empty() && name != "scaled10_1e12" && !name.starts_with("driver") {
            let tol = if want.on_pg { 1e-6 } else { 2e-3 };
            for (i, (ours, theirs)) in x.iter().zip(want.x).enumerate() {
                check(
                    (ours - theirs).abs() <= tol * theirs.abs().max(1.0),
                    format!("x[{i}] = {ours} against SciPy's {theirs}"),
                );
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn the_first_iterates_are_scipys() {
    // review round 1: the first five iterates against SciPy's (EARLY), one by one, which pins
    // the generalised Cauchy point, the subspace step, its projection or cut back to the box, and
    // the line search with its first trial and its cap at the box: each could be wrong in ways
    // that leave the end of a run, where the other tests look, as it is. Measured: the points
    // agree to 6·10⁻¹⁵ of max(|x|, 1), Brown's function to 5·10⁻¹², and the evaluations exactly.
    for &(name, steps) in EARLY {
        let p = problem(name);
        for (k, &(evaluations, want)) in steps.iter().enumerate() {
            let r = p.run_with(&p.options().with_max_iter(k + 1));
            let label = format!("{name}, iteration {}", k + 1);
            assert_eq!(r.evaluations, evaluations, "{label}: {r:?}");
            for (i, (ours, theirs)) in r.x.as_slice().iter().zip(want).enumerate() {
                assert!(
                    (ours - theirs).abs() <= 1e-10 * theirs.abs().max(1.0),
                    "{label}: x[{i}] = {ours} against SciPy's {theirs}"
                );
            }
        }
    }
}

#[test]
fn a_pair_with_too_little_curvature_is_left_out_of_the_memory() {
    // review round 1: nothing failed with the curvature test sᵀy > ε·(−gᵀs) taken out, since the
    // memory refuses a pair with sᵀy ≤ 0 by itself. On cᵀx + 10⁻¹⁶·xᵀx below 2 every pair has
    // sᵀy about 10⁻¹⁶ of −gᵀs; stored, such pairs make θ = 2·10⁻¹⁶ and the method ended with
    // LineSearch off the minimiser. Left out, it reaches the minimiser exactly: xᵢ = −cᵢ/(2·10⁻¹⁶)
    // where cᵢ > 0, and the bound 2, against which the gradient presses, elsewhere. SciPy stops
    // within 10⁻¹¹ of it after the same four iterations, on a level step (SCIPY, nearly_linear6).
    let r = problem("nearly_linear6").run();
    assert_eq!(r.stop, LbfgsbStop::ProjectedGradient, "{r:?}");
    let want = [-5e15, 2.0, -1.5e16, 2.0, -2.5e16, 2.0];
    for (ours, want) in r.x.as_slice().iter().zip(want) {
        assert!((ours - want).abs() <= 1e-12 * want.abs(), "{r:?}");
    }
    assert_eq!(active(&r), ".U.U.U");
    assert_eq!(r.iterations, reference("nearly_linear6").iterations);
}

#[test]
fn the_tight_runs_reach_the_minimiser_itself() {
    // Rosenbrock's minimum is (1, …, 1).
    for (name, tol) in [
        ("rosen2_tight", 1e-12),
        ("rosen10_tight", 1e-10),
        ("rosen100_tight", 1e-10),
    ] {
        if cfg!(miri) && name == "rosen100_tight" {
            continue;
        }
        let r = problem(name).run();
        assert_eq!(r.stop, LbfgsbStop::ProjectedGradient, "{name}");
        let error =
            r.x.as_slice()
                .iter()
                .fold(0.0f64, |m, v| m.max((v - 1.0).abs()));
        assert!(error <= tol, "{name}: {error:e} from (1, …, 1)");
    }
    // With the first variable held to at most ½ the minimum of the two-variable function is
    // (½, ¼), on that bound, where the gradient (−1, 0) presses against it.
    let r = problem("rosen2_box_tight").run();
    assert_eq!(r.x.as_slice(), &[0.5, 0.25]);
    assert_eq!(r.active, vec![Bound::Upper, Bound::Free]);
}

#[test]
fn the_boxed_quadratics_reach_the_kkt_points_bvls_finds() {
    // The error a projected gradient of g allows is ‖g‖/λ_min of the free part of A, which for
    // the tridiagonal matrix of order n is about 4·sin²(π/2(n + 1)): the tolerances below are
    // that bound at the projected gradient each run ends with, and the active sets are exact.
    for (n, kkt) in KKT {
        let name = match n {
            5 => "quad5_box",
            20 => "quad20_box",
            _ => "quad100_box",
        };
        if cfg!(miri) && *n == 100 {
            continue;
        }
        let p = problem(name);
        let r = p.run();
        let lambda = 4.0
            * (core::f64::consts::PI / (2.0 * (*n as f64 + 1.0)))
                .sin()
                .powi(2);
        let bound = (*n as f64).sqrt() * r.projected_gradient / lambda + 1e-14;
        for i in 0..*n {
            assert_eq!(
                r.x[(i, 0)] == 0.0,
                kkt[i] == 0.0,
                "{name}: variable {i} is {} where BVLS has {}",
                r.x[(i, 0)],
                kkt[i]
            );
            assert!(
                (r.x[(i, 0)] - kkt[i]).abs() <= bound,
                "{name}: x[{i}] = {} against {} (bound {bound:e})",
                r.x[(i, 0)],
                kkt[i]
            );
        }
        // As close as SciPy comes (1.7·10⁻⁸ on the largest): the decrease of f is lost in its
        // rounding beyond that.
        assert!(r.projected_gradient <= 5e-8, "{name}: {r:?}");
    }
}

#[test]
fn the_same_quadratic_with_bounds_on_one_side_or_none() {
    // Bounded below only, the answer is that of [0, 1]: no variable of it reaches 1. Bounded
    // only above by ¼ it is another one. Free, and inside a box too wide to matter, it is the
    // solution of A·x = b, which is checked here directly.
    let boxed = problem("quad20_box").run();
    let lower = problem("quad20_lower").run();
    assert_eq!(active(&lower), active(&boxed));
    for i in 0..20 {
        assert!((lower.x[(i, 0)] - boxed.x[(i, 0)]).abs() < 1e-6);
    }
    let upper = problem("quad20_upper").run();
    assert!(upper.active.iter().all(|b| *b != Bound::Lower));
    assert!(upper.x.as_slice().iter().all(|v| *v <= 0.25));
    let b = quad_b(20);
    for name in ["quad20_free", "quad20_interior"] {
        let r = problem(name).run();
        let residual = tridiagonal(r.x.as_slice());
        let worst = residual
            .iter()
            .zip(&b)
            .fold(0.0f64, |m, (ax, b)| m.max((ax - b).abs()));
        assert!(worst < 1e-7, "{name}: ‖Ax − b‖∞ = {worst:e}");
        assert!(r.active.iter().all(|b| *b == Bound::Free), "{name}");
    }
}

#[test]
fn a_start_outside_the_box_is_projected_before_anything_is_evaluated() {
    let p = problem("quad20_outside");
    let first: RefCell<Option<Vec<f64>>> = RefCell::new(None);
    let f = |x: &Matrix<f64>| {
        first
            .borrow_mut()
            .get_or_insert_with(|| x.as_slice().to_vec());
        (p.f)(x.as_slice())
    };
    let g = |x: &Matrix<f64>| col(&(p.g)(x.as_slice()));
    let r = lbfgsb_with(
        f,
        g,
        &col(&p.x0),
        &col(&p.lower),
        &col(&p.upper),
        &p.options(),
    )
    .unwrap();
    // 7.5 everywhere, projected onto [0, 1]²⁰.
    assert_eq!(first.borrow().as_deref(), Some(&[1.0; 20][..]));
    assert_eq!(active(&r), reference("quad20_outside").active);
}

#[test]
fn every_point_evaluated_lies_in_the_box() {
    for name in [
        "rosen10_box",
        "quad20_box",
        "driver1",
        "brown_box",
        "hs110",
        "vertex6",
        "quad20_outside",
    ] {
        let p = problem(name);
        let outside = Cell::new(0usize);
        let best = Cell::new(f64::INFINITY);
        let calls = Cell::new(0usize);
        let f = |x: &Matrix<f64>| {
            calls.set(calls.get() + 1);
            let x = x.as_slice();
            if (0..x.len()).any(|i| !(p.lower[i] <= x[i] && x[i] <= p.upper[i])) {
                outside.set(outside.get() + 1);
            }
            let v = (p.f)(x);
            best.set(best.get().min(v));
            v
        };
        let g = |x: &Matrix<f64>| col(&(p.g)(x.as_slice()));
        let r = lbfgsb_with(
            f,
            g,
            &col(&p.x0),
            &col(&p.lower),
            &col(&p.upper),
            &p.options(),
        )
        .unwrap();
        assert_eq!(
            outside.get(),
            0,
            "{name}: points outside the box were evaluated"
        );
        // The answer is the best point seen, and the count of evaluations is the true one.
        assert_eq!(r.value, best.get(), "{name}");
        assert_eq!(r.evaluations, calls.get(), "{name}");
    }
}

#[test]
fn the_steps_satisfy_the_strong_wolfe_conditions() {
    // Each run with max_iter = k stops at the k-th iterate, so the steps can be recovered one by
    // one and the conditions of the line search checked on s = x_{k+1} − x_k directly, in the
    // reference code's constants: f_{k+1} ≤ f_k + 10⁻³·g_kᵀs and |g_{k+1}ᵀs| ≤ 0.9·|g_kᵀs|.
    let p = problem("rosen10");
    let steps = if cfg!(miri) { 8 } else { 60 };
    let mut before = p.run_with(&p.options().with_max_iter(0));
    for k in 1..=steps {
        let after = p.run_with(&p.options().with_max_iter(k));
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
        let slope0: f64 = before
            .gradient
            .as_slice()
            .iter()
            .zip(&s)
            .map(|(g, s)| g * s)
            .sum();
        let slope1: f64 = after
            .gradient
            .as_slice()
            .iter()
            .zip(&s)
            .map(|(g, s)| g * s)
            .sum();
        assert!(slope0 < 0.0, "step {k} does not lead downhill");
        assert!(
            after.value <= before.value + 1e-3 * slope0,
            "step {k}: no sufficient decrease"
        );
        assert!(
            slope1.abs() <= 0.9 * slope0.abs(),
            "step {k}: the curvature condition fails: {slope1:e} against {slope0:e}"
        );
        before = after;
    }
}

#[test]
fn variables_with_equal_bounds_are_held_where_they_are() {
    // Every variable fixed: the answer is the box, at once.
    let lower = col(&[0.3, -0.2, 0.7]);
    let r = lbfgsb(
        |x: &Matrix<f64>| rosen(x.as_slice()),
        |x: &Matrix<f64>| col(&rosen_gradient(x.as_slice())),
        &col(&[5.0, 5.0, 5.0]),
        &lower,
        &lower,
    )
    .unwrap();
    assert!(r.converged && r.iterations == 0 && r.evaluations == 1);
    assert_eq!(r.x.as_slice(), lower.as_slice());
    assert_eq!(r.active, vec![Bound::Fixed; 3]);
    assert_eq!(r.projected_gradient, 0.0);

    // One fixed among free ones: it never moves, and the others minimise around it — here the
    // chained Rosenbrock function of four variables with the second held at ½, a minimum the
    // test checks by its projected gradient.
    let lo = [-INF, 0.5, -INF, -INF];
    let hi = [INF, 0.5, INF, INF];
    let moved = Cell::new(false);
    let f = |x: &Matrix<f64>| {
        if x[(1, 0)] != 0.5 {
            moved.set(true);
        }
        rosen(x.as_slice())
    };
    let g = |x: &Matrix<f64>| col(&rosen_gradient(x.as_slice()));
    let opts = LbfgsbOptions::new().with_pgtol(1e-6).with_ftol(0.0);
    let r = lbfgsb_with(
        f,
        g,
        &col(&[-1.2, 1.0, -1.2, 1.0]),
        &col(&lo),
        &col(&hi),
        &opts,
    )
    .unwrap();
    assert!(!moved.get(), "the fixed variable was moved");
    assert_eq!(r.active[1], Bound::Fixed);
    assert!(r.converged, "{r:?}");
    assert!(projected(r.x.as_slice(), &rosen_gradient(r.x.as_slice()), &lo, &hi) <= 1e-6);
}

#[test]
fn a_box_that_is_empty_or_not_a_box_is_refused() {
    let f = |x: &Matrix<f64>| x.as_slice().iter().map(|v| v * v).sum::<f64>();
    let g = |x: &Matrix<f64>| x * 2.0;
    let x0 = col(&[0.5, 0.5]);
    let try_box = |lower: &[f64], upper: &[f64]| lbfgsb(f, g, &x0, &col(lower), &col(upper));
    assert!(
        try_box(&[0.0, 1.0], &[1.0, 0.0]).is_none(),
        "lower above upper"
    );
    assert!(
        try_box(&[0.0, f64::NAN], &[1.0, 1.0]).is_none(),
        "a NaN lower bound"
    );
    assert!(
        try_box(&[0.0, 0.0], &[f64::NAN, 1.0]).is_none(),
        "a NaN upper bound"
    );
    assert!(
        try_box(&[INF, 0.0], &[INF, 1.0]).is_none(),
        "a lower bound of +∞"
    );
    assert!(
        try_box(&[-INF, 0.0], &[-INF, 1.0]).is_none(),
        "an upper bound of −∞"
    );
    assert!(try_box(&[0.0, 0.0], &[1.0, 1.0]).is_some(), "a good box");
    let open = col(&[-INF, -INF]);
    let shut = col(&[INF, INF]);
    assert!(
        lbfgsb(f, g, &col(&[f64::NAN, 0.0]), &open, &shut).is_none(),
        "a NaN start"
    );
    assert!(
        lbfgsb(f, g, &col(&[INF, 0.0]), &open, &shut).is_none(),
        "an infinite start"
    );
    let empty = Matrix::<f64>::zeros(0, 1);
    assert!(
        lbfgsb(f, g, &empty, &empty, &empty).is_none(),
        "no variables"
    );
    for (what, opts) in [
        ("no memory", LbfgsbOptions::new().with_memory(0)),
        ("a negative pgtol", LbfgsbOptions::new().with_pgtol(-1.0)),
        ("a NaN pgtol", LbfgsbOptions::new().with_pgtol(f64::NAN)),
        ("a NaN ftol", LbfgsbOptions::new().with_ftol(f64::NAN)),
        ("a negative ftol", LbfgsbOptions::new().with_ftol(-1e-9)),
    ] {
        assert!(
            lbfgsb_with(f, g, &x0, &open, &shut, &opts).is_none(),
            "{what}"
        );
    }
    // A start where f or its gradient is not finite, once projected, is no start.
    let lower = col(&[0.0, 0.0]);
    let upper = col(&[1.0, 1.0]);
    let log = |x: &Matrix<f64>| x[(0, 0)].ln() + x[(1, 0)].ln();
    let log_g = |x: &Matrix<f64>| col(&[1.0 / x[(0, 0)], 1.0 / x[(1, 0)]]);
    assert!(lbfgsb(log, log_g, &col(&[-3.0, 0.5]), &lower, &upper).is_none());
    let nan_g = |_: &Matrix<f64>| col(&[f64::NAN, 0.0]);
    assert!(lbfgsb(f, nan_g, &x0, &lower, &upper).is_none());
}

#[test]
#[should_panic(expected = "the bounds are not of the length of the starting point")]
fn bounds_of_the_wrong_length_panic() {
    let f = |x: &Matrix<f64>| x[(0, 0)] * x[(0, 0)];
    let g = |x: &Matrix<f64>| x * 2.0;
    let _ = lbfgsb(f, g, &col(&[1.0, 1.0]), &col(&[0.0]), &col(&[1.0, 1.0]));
}

#[test]
#[should_panic(expected = "the gradient is not a column")]
fn a_gradient_of_the_wrong_length_panics() {
    let f = |x: &Matrix<f64>| x[(0, 0)] * x[(0, 0)];
    let g = |_: &Matrix<f64>| col(&[1.0]);
    let _ = lbfgsb(
        f,
        g,
        &col(&[1.0, 1.0]),
        &col(&[0.0, 0.0]),
        &col(&[1.0, 1.0]),
    );
}

#[test]
#[should_panic(expected = "the starting point is not a column")]
fn a_start_that_is_not_a_column_panics() {
    let f = |x: &Matrix<f64>| x[(0, 0)] * x[(0, 0)];
    let g = |x: &Matrix<f64>| x * 2.0;
    let row = Matrix::from_rows(&[[1.0, 1.0]]);
    let _ = lbfgsb(f, g, &row, &col(&[0.0, 0.0]), &col(&[1.0, 1.0]));
}

#[test]
fn one_variable_ends_on_whichever_side_the_minimum_is() {
    let f = |x: &Matrix<f64>| (x[(0, 0)] - 2.0).powi(2);
    let g = |x: &Matrix<f64>| col(&[2.0 * (x[(0, 0)] - 2.0)]);
    for (l, u, want, bound) in [
        (0.0, 1.0, 1.0, Bound::Upper),
        (3.0, 4.0, 3.0, Bound::Lower),
        (-INF, INF, 2.0, Bound::Free),
        (-10.0, 10.0, 2.0, Bound::Free),
        (0.5, 0.5, 0.5, Bound::Fixed),
    ] {
        let r = lbfgsb(f, g, &col(&[0.25]), &col(&[l]), &col(&[u])).unwrap();
        assert!(r.converged, "[{l}, {u}]: {r:?}");
        assert!((r.x[(0, 0)] - want).abs() <= 1e-6, "[{l}, {u}]: {r:?}");
        assert_eq!(r.active, vec![bound], "[{l}, {u}]");
    }
}

#[test]
fn a_start_that_is_already_the_answer_takes_no_iteration() {
    // The minimum of Σ(xᵢ − i)², unconstrained, and a vertex whose gradient points out of the box.
    let f = |x: &Matrix<f64>| (0..3).map(|i| (x[(i, 0)] - i as f64).powi(2)).sum::<f64>();
    let g = |x: &Matrix<f64>| Matrix::from_fn(3, 1, |i, _| 2.0 * (x[(i, 0)] - i as f64));
    let open = col(&[-INF; 3]);
    let shut = col(&[INF; 3]);
    let r = lbfgsb(f, g, &col(&[0.0, 1.0, 2.0]), &open, &shut).unwrap();
    assert!(r.converged && r.iterations == 0 && r.evaluations == 1);
    let plane = |x: &Matrix<f64>| x.as_slice().iter().sum::<f64>();
    let up = |_: &Matrix<f64>| col(&[1.0; 3]);
    let r = lbfgsb(
        plane,
        up,
        &Matrix::zeros(3, 1),
        &Matrix::zeros(3, 1),
        &col(&[1.0; 3]),
    )
    .unwrap();
    assert!(r.converged && r.iterations == 0 && r.evaluations == 1);
    assert_eq!(r.active, vec![Bound::Lower; 3]);
}

#[test]
fn the_limits_on_iterations_and_evaluations_are_kept() {
    let p = problem("rosen10");
    let start = rosen(&p.x0);
    for k in [0, 1, 5] {
        let r = p.run_with(&p.options().with_max_iter(k));
        assert_eq!(r.iterations, k);
        assert_eq!(r.stop, LbfgsbStop::Iterations);
        assert!(!r.converged);
        assert!(k == 0 || r.value < start);
    }
    for limit in [1, 2, 7, 30] {
        let calls = Cell::new(0usize);
        let f = |x: &Matrix<f64>| {
            calls.set(calls.get() + 1);
            rosen(x.as_slice())
        };
        let g = |x: &Matrix<f64>| col(&rosen_gradient(x.as_slice()));
        let opts = p.options().with_max_evaluations(limit);
        let r = lbfgsb_with(f, g, &col(&p.x0), &col(&p.lower), &col(&p.upper), &opts).unwrap();
        assert!(
            calls.get() <= limit,
            "{} calls for a limit of {limit}",
            calls.get()
        );
        assert_eq!(r.evaluations, calls.get());
        assert_eq!(r.stop, LbfgsbStop::Evaluations);
        assert!(!r.converged);
    }
}

#[test]
fn the_reduction_test_can_be_turned_off() {
    // With ftol = 0 nothing but the projected gradient stops the method as converged.
    let p = problem("rosen10");
    let r = p.run_with(&p.options().with_ftol(0.0));
    assert_eq!(r.stop, LbfgsbStop::ProjectedGradient, "{r:?}");
    assert!(r.projected_gradient <= 1e-5);
    let loose = p.run();
    assert_eq!(loose.stop, LbfgsbStop::Reduction);
    assert!(loose.iterations < r.iterations);
}

#[test]
fn a_linear_function_ends_at_a_vertex() {
    let c = [3.0, -1.0, 0.5, -2.0];
    let f = |x: &Matrix<f64>| (0..4).map(|i| c[i] * x[(i, 0)]).sum::<f64>();
    let g = |_: &Matrix<f64>| col(&c);
    let r = lbfgsb(
        f,
        g,
        &col(&[0.1, 0.2, 0.3, 0.4]),
        &col(&[-1.0; 4]),
        &col(&[2.0; 4]),
    )
    .unwrap();
    assert!(r.converged);
    assert_eq!(r.x.as_slice(), &[-1.0, 2.0, -1.0, 2.0]);
    assert_eq!(
        r.active,
        vec![Bound::Lower, Bound::Upper, Bound::Lower, Bound::Upper]
    );
}

#[test]
fn a_function_unbounded_below_is_not_reported_as_converged() {
    let f = |x: &Matrix<f64>| -x[(0, 0)] - x[(1, 0)];
    let g = |_: &Matrix<f64>| col(&[-1.0, -1.0]);
    let opts = LbfgsbOptions::new().with_max_iter(30);
    let r = lbfgsb_with(
        f,
        g,
        &col(&[0.0, 0.0]),
        &col(&[-INF, -INF]),
        &col(&[INF, INF]),
        &opts,
    )
    .unwrap();
    assert!(!r.converged, "{r:?}");
    assert!(r.value < -1e9 && r.value.is_finite(), "{r:?}");
    // With a bound on one side only, it is the bound that stops it.
    let r = lbfgsb(
        f,
        g,
        &col(&[0.0, 0.0]),
        &col(&[-INF, -INF]),
        &col(&[INF, 3.0]),
    )
    .unwrap();
    assert!(!r.converged || r.x[(1, 0)] == 3.0, "{r:?}");
    assert_eq!(r.x[(1, 0)], 3.0);
}

#[test]
fn a_concave_function_passes_over_the_update_and_ends_at_a_vertex() {
    // −Σxᵢ² has negative curvature everywhere, so sᵀy < 0 on every step and no pair may enter the
    // memory; from (0.5, −0.2, 0.1) each variable runs to the bound it starts towards.
    let f = |x: &Matrix<f64>| -x.as_slice().iter().map(|v| v * v).sum::<f64>();
    let g = |x: &Matrix<f64>| x * -2.0;
    let r = lbfgsb(
        f,
        g,
        &col(&[0.5, -0.2, 0.1]),
        &col(&[-1.0; 3]),
        &col(&[2.0; 3]),
    )
    .unwrap();
    assert!(r.converged, "{r:?}");
    assert_eq!(r.x.as_slice(), &[2.0, -1.0, 2.0]);
    assert_eq!(r.active, vec![Bound::Upper, Bound::Lower, Bound::Upper]);
}

#[test]
fn a_function_undefined_on_part_of_the_box_is_minimised_where_it_is_defined() {
    // x − ln x is not finite for x ≤ 0 but the box goes down to −1; its minimum is 1, at x = 1.
    let f = |x: &Matrix<f64>| x[(0, 0)] - x[(0, 0)].ln();
    let g = |x: &Matrix<f64>| col(&[1.0 - 1.0 / x[(0, 0)]]);
    let opts = LbfgsbOptions::new().with_pgtol(1e-10);
    for start in [8.0, 0.01, 1e3] {
        let r = lbfgsb_with(f, g, &col(&[start]), &col(&[-1.0]), &col(&[1e4]), &opts).unwrap();
        assert!(r.converged, "from {start}: {r:?}");
        assert!((r.x[(0, 0)] - 1.0).abs() < 1e-8, "from {start}: {r:?}");
    }
}

#[test]
fn hock_and_schittkowskis_problems_reach_their_published_solutions() {
    // Problem 5: x* = (−π/3 + ½, −π/3 − ½), f* = −√3/2 − π/3. Problem 45: x* = (1, 2, 3, 4, 5),
    // f* = 1. Problem 110: every xᵢ* = 9.35026583…, f* = −45.77846971 (Hock and Schittkowski,
    // Test Examples for Nonlinear Programming Codes, 1981).
    let third = core::f64::consts::PI / 3.0;
    let r = problem("hs5").run();
    assert!(
        (r.x[(0, 0)] - (0.5 - third)).abs() < 1e-9 && (r.x[(1, 0)] - (-0.5 - third)).abs() < 1e-9
    );
    assert!((r.value - (-(3.0f64).sqrt() / 2.0 - third)).abs() < 1e-14);
    let r = problem("hs45").run();
    assert_eq!(r.x.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(r.value, 1.0);
    let r = problem("hs110").run();
    assert!(
        r.x.as_slice()
            .iter()
            .all(|v| (v - 9.350265836).abs() < 1e-8),
        "{r:?}"
    );
    assert!((r.value + 45.77846971).abs() < 1e-8);
}

#[test]
fn badly_scaled_problems_find_the_true_active_set() {
    // ½Σdᵢ(xᵢ − cᵢ)² in [−½, ½]¹⁰ is separable: its minimiser is clamp(cᵢ). The variables at a
    // bound are there exactly; the free ones, x₂, x₅ and x₈ with curvatures from 22 to 5·10¹⁰ at
    // the worse scaling, as closely as the measured accuracy says.
    for (name, cond, tol) in [("scaled10_1e6", 1e6, 1e-9), ("scaled10_1e12", 1e12, 1e-7)] {
        let (_, _, c) = scaled(cond);
        let r = problem(name).run();
        assert_eq!(active(&r), SCALED_ACTIVE, "{name}");
        for (i, centre) in c.iter().enumerate() {
            let want = centre.clamp(-0.5, 0.5);
            if want.abs() == 0.5 {
                assert_eq!(r.x[(i, 0)], want, "{name}: x[{i}]");
            } else {
                assert!(
                    (r.x[(i, 0)] - want).abs() <= tol,
                    "{name}: x[{i}] = {} against {want}",
                    r.x[(i, 0)]
                );
            }
        }
    }
}

#[test]
fn brown_badly_scaled_in_a_box() {
    // The minimum (10⁶, 2·10⁻⁶) inside the box, and cut off by x₀ ≤ 5·10⁵, where x₁ minimises
    // (x₁ − 2·10⁻⁶)² + (5·10⁵·x₁ − 2)²: x₁ = (2·10⁻⁶ + 10⁶)/(1 + 2.5·10¹¹).
    let r = problem("brown_box").run();
    assert!(
        (r.x[(0, 0)] - 1e6).abs() < 1e-5 && (r.x[(1, 0)] - 2e-6).abs() < 2e-11,
        "{r:?}"
    );
    assert!(r.value < 2e-10, "{r:?}");
    let r = problem("brown_box_cut").run();
    assert_eq!(r.x[(0, 0)], 5e5);
    let want = (2e-6 + 1e6) / (1.0 + 2.5e11);
    assert!((r.x[(1, 0)] - want).abs() < 1e-17, "{r:?}");
}

#[test]
fn a_large_bounded_problem_reaches_its_kkt_point() {
    // ½xᵀAx − bᵀx with A = tridiag(−1, 4, −1), which is well conditioned, in [0, 1]ⁿ: its KKT
    // point is the limit of projected Gauss–Seidel, xᵢ ← clamp((bᵢ + xᵢ₋₁ + xᵢ₊₁)/4), which
    // contracts by about a quarter a sweep here.
    let n = if cfg!(miri) { 40 } else { 2000 };
    let b: Vec<f64> = (0..n)
        .map(|i| 3.0 * (1.7 * (i + 1) as f64).sin() + 1.5)
        .collect();
    let a = |x: &[f64]| -> Vec<f64> {
        (0..x.len())
            .map(|i| {
                4.0 * x[i]
                    - if i > 0 { x[i - 1] } else { 0.0 }
                    - if i + 1 < x.len() { x[i + 1] } else { 0.0 }
            })
            .collect()
    };
    let mut reference = vec![0.0; n];
    for _ in 0..200 {
        for i in 0..n {
            let left = if i > 0 { reference[i - 1] } else { 0.0 };
            let right = if i + 1 < n { reference[i + 1] } else { 0.0 };
            reference[i] = ((b[i] + left + right) / 4.0).clamp(0.0, 1.0);
        }
    }
    let f = |x: &Matrix<f64>| {
        let x = x.as_slice();
        let ax = a(x);
        0.5 * x.iter().zip(&ax).map(|(p, q)| p * q).sum::<f64>()
            - x.iter().zip(&b).map(|(p, q)| p * q).sum::<f64>()
    };
    let g = |x: &Matrix<f64>| {
        col(&a(x.as_slice())
            .iter()
            .zip(&b)
            .map(|(p, q)| p - q)
            .collect::<Vec<_>>())
    };
    // A is diagonally dominant by 2, so ‖A_FF⁻¹‖∞ ≤ ½ and a projected gradient of 10⁻⁷ puts
    // every free variable within 5·10⁻⁸ of the KKT point.
    let opts = LbfgsbOptions::new().with_pgtol(1e-7).with_ftol(0.0);
    let r = lbfgsb_with(
        f,
        g,
        &col(&vec![0.5; n]),
        &Matrix::zeros(n, 1),
        &col(&vec![1.0; n]),
        &opts,
    )
    .unwrap();
    assert!(
        r.converged,
        "{:?}",
        (r.iterations, r.evaluations, r.projected_gradient, r.stop)
    );
    let mut held = 0;
    for (i, want) in reference.iter().enumerate() {
        assert!(
            (r.x[(i, 0)] - want).abs() <= 5e-8,
            "x[{i}] = {} against {want}",
            r.x[(i, 0)]
        );
        let at_bound = *want == 0.0 || *want == 1.0;
        assert_eq!(r.active[i] != Bound::Free, at_bound, "variable {i}");
        held += usize::from(at_bound);
    }
    assert!(
        held > n / 10 && held < n,
        "{held} of {n} variables at a bound"
    );
}

#[test]
fn a_step_that_leaves_f_level_is_taken_while_the_projected_gradient_falls() {
    // review round 1: near the end of ½xᵀAx − bᵀx in [0, 1]ⁿ, A = tridiag(−1, 3, −1), the change
    // of f along a step falls below its rounding while the projected gradient still falls fast.
    // Such a level step was taken only once in a row, and the method stopped with LineSearch at
    // a projected gradient of 1.3·10⁻⁷ (n = 10⁴, pgtol 10⁻⁷) and 1.7·10⁻⁸ (n = 10³, pgtol 10⁻⁹).
    // SciPy (LEVEL), which takes every such step, ends on its gradient test at 7.4·10⁻⁸ in the
    // first case, and in the second on the "reduction" of a level step at 1.7·10⁻⁸. f is summed
    // with compensation, so that its rounding is that of a few terms and not of n.
    for &(n, pgtol, iterations, on_pg, projected) in LEVEL {
        let n = if cfg!(miri) { n / 100 } else { n };
        let b: Vec<f64> = (0..n)
            .map(|i| 3.0 * (1.7 * (i + 1) as f64).sin() + 1.0)
            .collect();
        let a = |x: &[f64]| -> Vec<f64> {
            (0..x.len())
                .map(|i| {
                    3.0 * x[i]
                        - if i > 0 { x[i - 1] } else { 0.0 }
                        - if i + 1 < x.len() { x[i + 1] } else { 0.0 }
                })
                .collect()
        };
        let f = |x: &Matrix<f64>| {
            let x = x.as_slice();
            let ax = a(x);
            // Neumaier's compensated sum of the terms ½xᵢ(Ax)ᵢ − bᵢxᵢ.
            let (mut sum, mut lost) = (0.0f64, 0.0f64);
            for i in 0..n {
                let term = 0.5 * x[i] * ax[i] - x[i] * b[i];
                let next = sum + term;
                lost += if sum.abs() >= term.abs() {
                    (sum - next) + term
                } else {
                    (term - next) + sum
                };
                sum = next;
            }
            sum + lost
        };
        let g = |x: &Matrix<f64>| {
            col(&a(x.as_slice())
                .iter()
                .zip(&b)
                .map(|(p, q)| p - q)
                .collect::<Vec<_>>())
        };
        let opts = LbfgsbOptions::new().with_pgtol(pgtol).with_ftol(0.0);
        let r = lbfgsb_with(
            f,
            g,
            &col(&vec![0.5; n]),
            &Matrix::zeros(n, 1),
            &col(&vec![1.0; n]),
            &opts,
        )
        .unwrap();
        let summary = (r.iterations, r.evaluations, r.projected_gradient, r.stop);
        assert_eq!(
            r.stop,
            LbfgsbStop::ProjectedGradient,
            "n = {n}: {summary:?}"
        );
        assert!(r.projected_gradient <= pgtol, "n = {n}: {summary:?}");
        if !cfg!(miri) {
            assert!(
                r.iterations as f64 <= 1.3 * iterations as f64 + 3.0,
                "n = {n}: {summary:?} against SciPy's {iterations} iterations"
            );
            assert!(on_pg || projected > pgtol, "the table: {projected:e}");
        }
    }
}

#[test]
fn single_precision_finds_the_same_answers() {
    let f = |x: &Matrix<f32>| {
        (1.0 - x[(0, 0)]).powi(2) + 100.0 * (x[(1, 0)] - x[(0, 0)].powi(2)).powi(2)
    };
    let g = |x: &Matrix<f32>| {
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
    let c32 = |v: &[f32]| Matrix::from_column_major(v.len(), 1, v.to_vec());
    let r = lbfgsb(
        f,
        g,
        &c32(&[-1.2, 1.0]),
        &c32(&[-2.0, -2.0]),
        &c32(&[0.5, 2.0]),
    )
    .unwrap();
    assert!(r.converged, "{r:?}");
    assert_eq!(r.x[(0, 0)], 0.5);
    assert!((r.x[(1, 0)] - 0.25).abs() < 1e-3, "{r:?}");
    assert_eq!(r.active, vec![Bound::Upper, Bound::Free]);
    // Unconstrained, to the tolerance f32 allows: measured 10⁻⁵ off (1, 1).
    let r = lbfgsb(
        f,
        g,
        &c32(&[-1.2, 1.0]),
        &c32(&[f32::NEG_INFINITY; 2]),
        &c32(&[f32::INFINITY; 2]),
    )
    .unwrap();
    assert!(r.converged, "{r:?}");
    assert!(
        (r.x[(0, 0)] - 1.0).abs() < 1e-4 && (r.x[(1, 0)] - 1.0).abs() < 1e-4,
        "{r:?}"
    );

    // The boxed quadratic of five variables against its KKT point: measured 7·10⁻⁵ off, the
    // reduction test of the f32 default (10·ε) stopping it.
    let b: Vec<f32> = quad_b(5).iter().map(|v| *v as f32).collect();
    let a = |x: &[f32]| -> Vec<f32> {
        (0..5)
            .map(|i| {
                2.0 * x[i] - if i > 0 { x[i - 1] } else { 0.0 } - if i < 4 { x[i + 1] } else { 0.0 }
            })
            .collect()
    };
    let f = |x: &Matrix<f32>| {
        let x = x.as_slice();
        let ax = a(x);
        0.5 * x.iter().zip(&ax).map(|(p, q)| p * q).sum::<f32>()
            - x.iter().zip(&b).map(|(p, q)| p * q).sum::<f32>()
    };
    let g = |x: &Matrix<f32>| {
        c32(&a(x.as_slice())
            .iter()
            .zip(&b)
            .map(|(p, q)| p - q)
            .collect::<Vec<_>>())
    };
    let r = lbfgsb(f, g, &c32(&[0.5; 5]), &c32(&[0.0; 5]), &c32(&[1.0; 5])).unwrap();
    assert!(r.converged, "{r:?}");
    for (i, want) in KKT[0].1.iter().enumerate() {
        assert!(
            (r.x[(i, 0)] as f64 - want).abs() < 2e-4,
            "x[{i}] = {} against {want}",
            r.x[(i, 0)]
        );
    }
    assert_eq!(r.active[2], Bound::Lower);
}

// Three of the second review's 800 random problems (its generator lbstress/gen_lb.py, ids 625,
// 635 and 149), on which what stops the method is the rounding of f; data/gen_lbfgsb.py
// (rounding()) holds the same data and makes the SciPy results ROUNDING and LSE_SCIPY. The
// quadratics are ½xᵀAx − bᵀx with DATA = A row by row, then b; the log-sum-exp is
// ln Σₖ e^(wₖ·x) + ½μ‖x‖² with DATA = [k, μ, w₁, …, w_k].
const R625_X0: [f64; 2] = [0.03369087526957654, -3.2963086401551287];
const R625_LO: [f64; 2] = [-INF, -4.040565963435006];
const R625_HI: [f64; 2] = [INF, INF];
const R625_DATA: [f64; 6] = [
    35512.66208772069,
    -53645.76304781647,
    -53645.76304781647,
    81041.07877394135,
    127414.874779647,
    -192480.90951089462,
];
const R635_X0: [f64; 3] = [-2.975667530398329, -5.281298660514384, -0.8181495441580848];
const R635_LO: [f64; 3] = [-1.1415454326625887, 1.2538402561983948, -INF];
const R635_HI: [f64; 3] = [0.3701705394272141, 1.2538402561983948, 2.1856618964189924];
const R635_DATA: [f64; 12] = [
    4800157.1113577755,
    -18600428.428781837,
    -4023175.119721802,
    -18600428.428781837,
    74082911.39016782,
    16037644.352393784,
    -4023175.119721802,
    16037644.352393784,
    3471961.9856460625,
    -22442422.024965048,
    88695223.16868874,
    19196304.888220012,
];
const R149_X0: [f64; 40] = [
    0.5553611930039751,
    1.60312191376013,
    1.3267902489963255,
    -2.984052653201408,
    -2.8504699391751545,
    -1.1133908289422794,
    1.8650540943563643,
    1.4745936249000124,
    -0.07501941011007457,
    -0.15582157740046437,
    1.1067522517672748,
    1.339386881819725,
    -1.036957491121891,
    -0.4581339072150094,
    3.913653959514196,
    -2.1791160072528273,
    -0.0949103842908305,
    -2.6566156448335865,
    0.2726935855420944,
    0.46786004927814545,
    0.42113464163279163,
    -0.5505737246063112,
    -1.7139866620647644,
    -2.3560432363054327,
    1.0180821473958996,
    -0.7427089301652298,
    1.6860446938347051,
    -2.171111883303836,
    0.020824380919556698,
    -0.4747592833342588,
    0.0795045405670847,
    -3.5977175813156457,
    -1.5938373309818572,
    -0.7139869523883862,
    -0.3334512379439324,
    0.3959199490403062,
    1.0626166099134302,
    0.2810315389856055,
    -1.2832810459782473,
    -0.7442004912184835,
];
const R149_LO: [f64; 40] = [
    0.059677116552124126,
    0.23565116258179772,
    -1.3641154872203252,
    -0.6387028342061859,
    -0.4794572042020986,
    -INF,
    -0.3579274380663337,
    -1.3639675042577322,
    -0.8369798682407514,
    -0.49139218974843724,
    -INF,
    -0.6254662387050021,
    -0.49396085606115075,
    -2.1329425914166533,
    -1.6075645253268447,
    -INF,
    -INF,
    -1.9929783042035296,
    -INF,
    -INF,
    -0.012148884536409324,
    -0.7778861765069175,
    -0.6172423489158643,
    0.5433849758311058,
    -INF,
    0.24854867270786293,
    -INF,
    0.1757353550506198,
    -0.8098681358950454,
    -INF,
    -1.5543869389675153,
    -0.3105598208148608,
    -0.27229162978613947,
    0.016450464777214124,
    0.35986662800643077,
    -INF,
    -0.7416826139691779,
    -2.0400774319065773,
    -INF,
    -INF,
];
const R149_HI: [f64; 40] = [
    INF,
    0.6990189380785382,
    -0.8479239962551122,
    INF,
    0.05696159162346342,
    INF,
    INF,
    INF,
    -0.8369798682407514,
    0.48732423283104676,
    INF,
    0.16612467325857525,
    INF,
    INF,
    -0.5458841061960351,
    INF,
    INF,
    0.9494791428927555,
    0.2571813905235761,
    INF,
    0.6356119096061017,
    0.6858249388279185,
    1.787174794956803,
    INF,
    1.343626732171001,
    0.24854867270786293,
    INF,
    0.22150773875047644,
    0.9977249607873304,
    1.2412714639640303,
    0.4793218645119126,
    2.4289077510075754,
    INF,
    0.560401904354155,
    INF,
    0.8562913258724972,
    1.0856139943546463,
    -0.03549897291448347,
    INF,
    INF,
];
const R149_DATA: [f64; 82] = [
    2.0,
    0.0017604017596437808,
    -0.9258142727967451,
    -0.6716162597490568,
    -0.24880784789440263,
    1.0012382093670922,
    0.13788221510307164,
    -0.46642306118893245,
    -0.1083823120115684,
    -0.2777632818874784,
    1.4354922263667869,
    1.1635938638673375,
    2.4359001354071013,
    0.8064173610655561,
    0.22085526491009572,
    -0.06501945665863153,
    -0.7317076599452548,
    0.5017279872295921,
    0.4091004145940108,
    -0.8803862339201066,
    -0.8379718038820976,
    0.2746411289086623,
    -0.3366024310414633,
    -0.4997195086393795,
    0.7191710559742404,
    -0.07499268134951025,
    1.0641580791827587,
    -0.48501219832580855,
    -0.38469039580265735,
    -0.7990226767338368,
    -1.3977163985381598,
    -0.44628823116115357,
    -0.8471490865343269,
    1.2046382654913832,
    1.193002743403995,
    -0.1715876011799664,
    -0.5549268755425227,
    1.1101737805094145,
    -2.2881811352683705,
    0.362848087448214,
    0.7268365894830385,
    -0.6458140332580248,
    -0.6730067856762144,
    1.6967328779956101,
    -0.3454859523636702,
    1.179313944214341,
    1.6493494325276408,
    0.46843693633283895,
    -0.5075659869255497,
    0.34204162490517054,
    -0.10458589081661161,
    -1.5016163578724906,
    2.2452579128671477,
    0.33538787570848433,
    -1.167114372160207,
    0.27599062180135964,
    -0.8593622820032486,
    -1.4665627964283956,
    -1.5105493032467734,
    -1.1783502859427715,
    -0.6872838105836985,
    0.5877896496574923,
    -1.167547307900751,
    -0.12364096707313929,
    1.1324073083246287,
    0.7530579041348795,
    -0.050919084390707026,
    0.5881562987109881,
    -1.529176158928944,
    0.7366533556679287,
    1.1069181838799935,
    -0.8309800777480725,
    1.11765529125593,
    -0.6458049548441359,
    0.5282070842141305,
    -0.48894767137376005,
    -0.8493751207772174,
    0.8386190764607233,
    -0.8363796224133611,
    1.4847394083801162,
    0.4299031060072906,
    -0.5667138614126938,
];

// Made by data/gen_lbfgsb.py: SciPy's L-BFGS-B on the second review's quadratics 625 and
// 635: whether it stopped on the projected gradient, its value and projected gradient there;
// and the KKT point by BVLS (the fixed variable eliminated first).
const ROUNDING: &[(usize, bool, f64, f64, &[f64])] = &[
    (
        625,
        false,
        -228581.14535210808,
        8.94651748239994e-08,
        &[0.4602306148561706, -2.0704498204794723],
    ),
    (
        635,
        false,
        -53208768.063197434,
        3.725290298461914e-09,
        &[-1.1415454326625887, 1.2538402561983948, -1.5855520368262237],
    ),
];
// Made by data/gen_lbfgsb.py: SciPy's L-BFGS-B in f64 on the second review's log-sum-exp
// 149: whether it stopped on the reduction test, its value and its projected gradient.
const LSE_SCIPY: (bool, f64, f64) = (true, -2851.9012372317443, 0.003172039140471372);

/// ½xᵀAx − bᵀx and its gradient, `data` holding A row by row and then b, summed in the order of
/// the second review's probe, since the stops of the test that uses it depend on the rounding of
/// f.
fn review_quadratic(data: &[f64], x: &[f64]) -> (f64, Vec<f64>) {
    let n = x.len();
    let (a, b) = (&data[..n * n], &data[n * n..]);
    let ax: Vec<f64> = (0..n)
        .map(|i| (0..n).fold(0.0, |s, j| s + a[i * n + j] * x[j]))
        .collect();
    let f = (0..n).fold(0.0, |s, i| s + 0.5 * x[i] * ax[i] - b[i] * x[i]);
    (f, (0..n).map(|i| ax[i] - b[i]).collect())
}

/// One of the second review's quadratics: the start, the bounds, the data of
/// [`review_quadratic`], the `ftol` of its run and a bound on the projected gradient it is left
/// at.
type RoundingCase = (
    &'static [f64],
    &'static [f64],
    &'static [f64],
    &'static [f64],
    f64,
    f64,
);

#[test]
fn where_the_rounding_of_f_hides_every_decrease_the_method_ends_with_line_search() {
    // A documented limit (second review): where the rounding of f hides the decrease of every
    // step before the projected gradient is down to pgtol, the method ends with LineSearch and
    // converged false, whatever ftol is; SciPy takes a further step there and reports
    // convergence (ROUNDING). The review's quadratics of 2 and 3 variables, condition 1.2·10⁵ and
    // 8·10⁷, with pgtol 10⁻⁹ and an ftol of 2.2·10⁻¹³ and 2.2·10⁻⁹: what is left of the gradient,
    // measured 8.0·10⁻⁴ and 2.0·10⁻⁵, lies along the stiff direction, where it is worth less than
    // an ulp of f, so that the value is SciPy's and the one at the KKT point to an ulp (measured:
    // one ulp below both).
    let cases: [RoundingCase; 2] = [
        (
            &R625_X0,
            &R625_LO,
            &R625_HI,
            &R625_DATA,
            2.220446049250313e-13,
            1e-3,
        ),
        (
            &R635_X0,
            &R635_LO,
            &R635_HI,
            &R635_DATA,
            2.220446049250313e-9,
            3e-5,
        ),
    ];
    for (
        &(id, scipy_on_pg, scipy_value, scipy_projected, kkt),
        (x0, lower, upper, data, ftol, left),
    ) in ROUNDING.iter().zip(cases)
    {
        // SciPy's further step is a "reduction" of a step that leaves f level.
        assert!(!scipy_on_pg && scipy_projected < 1e-7, "the table, {id}");
        let opts = LbfgsbOptions::new()
            .with_memory(17)
            .with_pgtol(1e-9)
            .with_ftol(ftol);
        let r = lbfgsb_with(
            |x: &Matrix<f64>| review_quadratic(data, x.as_slice()).0,
            |x: &Matrix<f64>| col(&review_quadratic(data, x.as_slice()).1),
            &col(x0),
            &col(lower),
            &col(upper),
            &opts,
        )
        .unwrap();
        let x = r.x.as_slice();
        let (value, gradient) = review_quadratic(data, x);
        let summary = (
            id,
            r.stop,
            r.iterations,
            r.evaluations,
            r.value,
            r.projected_gradient,
        );
        assert_eq!(r.stop, LbfgsbStop::LineSearch, "{summary:?}");
        assert!(!r.converged, "{summary:?}");
        // What the result says of itself is true: its value and its projected gradient are
        // those of its point, and the projected gradient is above pgtol.
        assert_eq!(r.value, value, "{summary:?}");
        let pg = projected(x, &gradient, lower, upper);
        assert!(
            (r.projected_gradient - pg).abs() <= 1e-9 * pg,
            "{summary:?}: {pg:e} at x"
        );
        assert!(
            r.projected_gradient > opts.pgtol && r.projected_gradient < left,
            "{summary:?}"
        );
        // f cannot tell the point from the minimiser.
        let ulp = f64::EPSILON * value.abs();
        let (at_kkt, _) = review_quadratic(data, kkt);
        assert!(
            (value - scipy_value).abs() <= ulp && (value - at_kkt).abs() <= ulp,
            "{summary:?}: {:e} from SciPy's value, {:e} from the KKT point's",
            value - scipy_value,
            value - at_kkt
        );
        // The point is the best one reached, near the KKT point: measured 2.9·10⁻⁸ and 5.8·10⁻¹².
        let off = x
            .iter()
            .zip(kkt)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(off < 1e-7, "{summary:?}: {off:e} from the KKT point");
    }
}

/// ln Σₖ e^(wₖ·x) + ½μ‖x‖² and its gradient, `data` = [k, μ, w₁, …, w_k], computed in `T` in the
/// order of the second review's probe, since the stop of the test that uses it depends on the
/// rounding of f.
fn review_log_sum_exp<T: Real + Scalar>(data: &[f64], x: &[T]) -> (T, Vec<T>) {
    let n = x.len();
    let k = data[0] as usize;
    let d: Vec<T> = data.iter().map(|v| T::from_f64(*v)).collect();
    let (mu, w) = (d[1], &d[2..]);
    let z: Vec<T> = (0..k)
        .map(|r| (0..n).fold(T::ZERO, |s, j| s + w[r * n + j] * x[j]))
        .collect();
    let top = z.iter().fold(z[0], |m, v| if *v > m { *v } else { m });
    let e: Vec<T> = z.iter().map(|v| (*v - top).exp()).collect();
    let sum = e.iter().fold(T::ZERO, |s, v| s + *v);
    let f = top + sum.ln() + T::from_f64(0.5) * mu * x.iter().fold(T::ZERO, |s, v| s + *v * *v);
    let g = (0..n)
        .map(|j| (0..k).fold(T::ZERO, |s, r| s + w[r * n + j] * e[r] / sum) + mu * x[j])
        .collect();
    (f, g)
}

#[test]
fn a_reduction_stop_in_single_precision_away_from_the_minimiser_shows_in_its_projected_gradient() {
    // A documented limit (second review): the reduction test says that f has stopped improving,
    // which in f32 (ftol = 10ε) on a function that is not a quadratic can be some way from the
    // minimiser. The review's log-sum-exp with a ridge, 40 variables with bounds on most, memory
    // 1: the f32 run stops so with f 0.60% above SciPy's f64 value (LSE_SCIPY; the f64 run below
    // stops 1.3·10⁻⁹ from it). What the result says of itself is true: the stop is Reduction, and
    // the projected gradient it reports, that of its point, is 0.44, far above pgtol; and with
    // ftol = 0, which leaves only the gradient test, the method does not say converged there
    // (measured: LineSearch at a projected gradient of 0.012, f 3.2·10⁻⁵ above SciPy's).
    let (scipy_reduction, scipy_value, _) = LSE_SCIPY;
    assert!(scipy_reduction, "the table");
    let c32 =
        |v: &[f64]| Matrix::from_column_major(v.len(), 1, v.iter().map(|p| *p as f32).collect());
    let f = |x: &Matrix<f32>| review_log_sum_exp(&R149_DATA, x.as_slice()).0;
    let g = |x: &Matrix<f32>| {
        let v = review_log_sum_exp(&R149_DATA, x.as_slice()).1;
        Matrix::from_column_major(v.len(), 1, v)
    };
    let opts = LbfgsbOptions::<f32>::new().with_memory(1).with_pgtol(1e-5);
    assert_eq!(opts.ftol, 10.0 * f32::EPSILON, "the documented f32 default");
    let mut reduction_value = f64::INFINITY;
    for ftol in [opts.ftol, 0.0] {
        let r = lbfgsb_with(
            f,
            g,
            &c32(&R149_X0),
            &c32(&R149_LO),
            &c32(&R149_HI),
            &opts.with_ftol(ftol),
        )
        .unwrap();
        let x: Vec<f64> = r.x.as_slice().iter().map(|v| *v as f64).collect();
        let gradient: Vec<f64> = g(&r.x).as_slice().iter().map(|v| *v as f64).collect();
        let pg = projected(&x, &gradient, &R149_LO, &R149_HI);
        let above = (r.value as f64 - scipy_value) / scipy_value.abs();
        let summary = (
            ftol,
            r.stop,
            r.iterations,
            r.evaluations,
            r.value,
            r.projected_gradient,
            above,
        );
        assert_eq!(r.value, f(&r.x), "{summary:?}");
        assert!(
            (r.projected_gradient as f64 - pg).abs() <= 1e-6 * pg,
            "{summary:?}: {pg:e} at x"
        );
        if ftol > 0.0 {
            assert!(
                r.converged && r.stop == LbfgsbStop::Reduction,
                "{summary:?}"
            );
            assert!(above > 3e-3 && above < 1e-2, "{summary:?}");
            assert!(r.projected_gradient > 0.1, "{summary:?}");
            reduction_value = r.value as f64;
        } else {
            assert_eq!(
                r.converged,
                r.stop == LbfgsbStop::ProjectedGradient,
                "{summary:?}"
            );
            assert!(
                !r.converged || r.projected_gradient <= opts.pgtol,
                "{summary:?}"
            );
            assert!((r.value as f64) < reduction_value, "{summary:?}");
            assert!(above < 1e-4, "{summary:?}");
        }
    }
    // In f64 with SciPy's tolerances the method stops where SciPy does.
    let r = lbfgsb_with(
        |x: &Matrix<f64>| review_log_sum_exp(&R149_DATA, x.as_slice()).0,
        |x: &Matrix<f64>| col(&review_log_sum_exp(&R149_DATA, x.as_slice()).1),
        &col(&R149_X0),
        &col(&R149_LO),
        &col(&R149_HI),
        &LbfgsbOptions::new()
            .with_memory(1)
            .with_pgtol(1e-5)
            .with_ftol(FACTR),
    )
    .unwrap();
    assert_eq!(r.stop, LbfgsbStop::Reduction, "{r:?}");
    assert!(
        (r.value - scipy_value).abs() <= 2.0 * FACTR * scipy_value.abs(),
        "f64: {} against SciPy's {scipy_value}",
        r.value
    );
}

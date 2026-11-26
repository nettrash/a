//! Linear programming by the simplex method, [`a::math::numerics::LinearProgram`]: textbook
//! examples against their closed forms, the examples that make the simplex method cycle, the
//! cubes of Klee and Minty, the degenerate shapes a programme can have, invalid input, `f32`, and
//! six hundred random programmes against `scipy.optimize.linprog(method="highs")`.
//!
//! The random programmes are made from a seed by [`generate`], which mirrors `generate` of
//! `data/gen_lp.py` statement for statement; `data/lp_highs.txt` holds, for each seed, the status
//! HiGHS gave the programme, its optimal objective and a checksum of the data, so that a drift
//! between the two generators cannot go unnoticed. Every answer the solver calls optimal is also
//! verified here without HiGHS, by [`certify`]: the point against the constraints, the dual
//! values against the signs of their convention, complementary slackness and strong duality.

use a::math::matrix::Matrix;
use a::math::numerics::{LinearProgram, LpOptions, LpResult, LpStatus};

// ---------------------------------------------------------------------------------------------
// The random programmes
// ---------------------------------------------------------------------------------------------

/// SplitMix64, on integers only: the generator of `data/gen_lp.py`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// An integer in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// An integer in `lo..=hi`.
    fn between(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
}

/// A linear programme in integers: minimise (or maximise) `c·x` subject to `a_le·x ≤ b_le`,
/// `a_eq·x = b_eq` and `lower ≤ x ≤ upper`, a bound that is `None` being infinite.
#[derive(Debug, Clone, Default)]
struct Case {
    n: usize,
    maximize: bool,
    c: Vec<i64>,
    a_le: Vec<Vec<i64>>,
    b_le: Vec<i64>,
    a_eq: Vec<Vec<i64>>,
    b_eq: Vec<i64>,
    lower: Vec<Option<i64>>,
    upper: Vec<Option<i64>>,
    /// Whether some rows and columns are powers of ten larger than the rest.
    wide: bool,
}

fn dot(a: &[i64], b: &[i64]) -> i64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The linear programme of a seed: `generate` of `data/gen_lp.py`, draw for draw.
fn generate(seed: u64) -> Case {
    let mut r = Rng(seed);
    let mut p = Case::default();
    let kind = r.below(8);
    let size = r.below(10);
    let (n, m_le, m_eq) = if size < 6 {
        (r.between(1, 10), r.between(0, 8), r.between(0, 4))
    } else if size < 9 {
        (r.between(8, 40), r.between(0, 25), r.between(0, 10))
    } else {
        (r.between(40, 120), r.between(10, 45), r.between(0, 15))
    };
    let (n, m_le, m_eq) = (n as usize, m_le as usize, m_eq as usize);
    let density = [20, 50, 100][r.below(3) as usize];
    let wide = r.below(5) == 0;
    let mix = r.below(4);
    p.n = n;
    p.wide = wide;
    p.maximize = r.below(4) == 0;

    // The bounds, and a point x0 inside them.
    let mut x0 = Vec::with_capacity(n);
    for _ in 0..n {
        let bound = match mix {
            0 => 0,
            2 => 1,
            3 => {
                if r.below(4) != 0 {
                    4
                } else {
                    r.below(6)
                }
            }
            _ => r.below(6),
        };
        let (lo, hi) = match bound {
            0 => (Some(0), None),
            1 => (None, None),
            2 => (Some(r.between(-10, 10)), None),
            3 => (None, Some(r.between(-10, 10))),
            4 => {
                let lo = r.between(-10, 10);
                (Some(lo), Some(lo + r.between(1, 12)))
            }
            _ => {
                let lo = r.between(-10, 10);
                (Some(lo), Some(lo))
            }
        };
        let x = match (lo, hi) {
            (Some(lo), Some(hi)) => {
                let t = r.below(4);
                if t == 0 || kind == 1 {
                    lo
                } else if t == 1 {
                    hi
                } else {
                    lo + r.below((hi - lo + 1) as u64) as i64
                }
            }
            (Some(lo), None) => {
                if r.below(2) == 0 || kind == 1 {
                    lo
                } else {
                    lo + r.between(1, 6)
                }
            }
            (None, Some(hi)) => {
                if r.below(2) == 0 || kind == 1 {
                    hi
                } else {
                    hi - r.between(1, 6)
                }
            }
            (None, None) => r.between(-6, 6),
        };
        p.lower.push(lo);
        p.upper.push(hi);
        x0.push(x);
    }

    let column_scale: Vec<i64> = (0..n)
        .map(|_| {
            if wide {
                [1, 1, 1, 10, 100][r.below(5) as usize]
            } else {
                1
            }
        })
        .collect();
    let row_scale: Vec<i64> = (0..m_le + m_eq)
        .map(|_| {
            if wide {
                [1, 1, 1, 10, 1000][r.below(5) as usize]
            } else {
                1
            }
        })
        .collect();

    let mut rows: Vec<Vec<i64>> = Vec::with_capacity(m_le + m_eq);
    for rs in &row_scale {
        let mut row = Vec::with_capacity(n);
        for cs in &column_scale {
            let v = if r.below(100) < density {
                r.between(-9, 9)
            } else {
                0
            };
            row.push(v * cs * rs);
        }
        rows.push(row);
    }

    // The ray of an unbounded programme.
    let mut ray = vec![0i64; n];
    if kind == 5 {
        let mut open_ended: Vec<usize> = (0..n)
            .filter(|&j| p.lower[j].is_none() || p.upper[j].is_none())
            .collect();
        if open_ended.is_empty() {
            p.lower[0] = None;
            p.upper[0] = None;
            x0[0] = 0;
            open_ended = vec![0];
        }
        let count = open_ended.len().min(r.between(1, 3) as usize);
        let start = r.below(open_ended.len() as u64) as usize;
        let support: Vec<usize> = (0..count)
            .map(|k| open_ended[(start + k) % open_ended.len()])
            .collect();
        for &j in &support {
            ray[j] = if p.upper[j].is_none() && p.lower[j].is_none() {
                if r.below(2) == 0 { 1 } else { -1 }
            } else if p.upper[j].is_none() {
                1
            } else {
                -1
            };
        }
        let first = support[0];
        for row in rows.iter_mut().take(m_le) {
            if dot(row, &ray) > 0 {
                for &j in &support {
                    row[j] = -row[j];
                }
            }
        }
        for row in rows.iter_mut().skip(m_le) {
            let rest = dot(row, &ray) - row[first] * ray[first];
            row[first] = -ray[first] * rest;
        }
    }

    // A redundant equation: the last one becomes a combination of the first two.
    if m_eq >= 3 && r.below(4) == 0 {
        let f1 = r.between(-2, 2);
        let f2 = r.between(-2, 2);
        let combined: Vec<i64> = rows[m_le]
            .iter()
            .zip(&rows[m_le + 1])
            .map(|(u, v)| f1 * u + f2 * v)
            .collect();
        rows[m_le + m_eq - 1] = combined;
    }

    // The right-hand sides: through x0 when the programme is to be feasible.
    let mut rhs = Vec::with_capacity(m_le + m_eq);
    for i in 0..m_le + m_eq {
        let at = dot(&rows[i], &x0);
        if kind >= 6 {
            if i < m_le {
                rhs.push(r.between(-20, 60) * row_scale[i]);
            } else if r.below(2) == 0 {
                rhs.push(at);
            } else {
                rhs.push(r.between(-10, 10) * row_scale[i]);
            }
        } else if i < m_le {
            let slack = if r.below(3) == 0 || kind == 1 {
                0
            } else {
                r.between(1, 20)
            };
            rhs.push(at + slack * row_scale[i]);
        } else {
            rhs.push(at);
        }
    }

    // The costs. A bounded programme has a feasible dual: multipliers of the right signs.
    let mut cost = Vec::with_capacity(n);
    if kind <= 4 {
        let mut y = Vec::with_capacity(m_le + m_eq);
        for i in 0..m_le + m_eq {
            if i < m_le {
                y.push(if r.below(3) != 0 { -r.between(0, 4) } else { 0 });
            } else {
                y.push(r.between(-4, 4));
            }
        }
        for j in 0..n {
            let d = if kind == 2 && r.below(4) != 0 {
                0
            } else {
                match (p.lower[j], p.upper[j]) {
                    (None, None) => 0,
                    (_, None) => r.between(0, 4),
                    (None, _) => -r.between(0, 4),
                    _ => r.between(-4, 4),
                }
            };
            let priced: i64 = (0..m_le + m_eq).map(|i| y[i] * rows[i][j]).sum();
            cost.push(d * column_scale[j] + priced);
        }
    } else {
        for cs in &column_scale {
            cost.push(r.between(-9, 9) * cs);
        }
        if kind == 5 {
            let along = dot(&cost, &ray);
            if along >= 0 {
                let first = ray.iter().position(|v| *v != 0).unwrap();
                cost[first] -= ray[first] * (along + r.between(1, 5));
            }
        }
    }

    p.a_eq = rows.split_off(m_le);
    p.a_le = rows;
    p.b_eq = rhs.split_off(m_le);
    p.b_le = rhs;

    // An infeasible programme: a feasible one and a constraint that contradicts it.
    if kind == 4 {
        let how = r.below(3);
        if how == 0 && m_le >= 1 {
            let mut weights: Vec<i64> = (0..m_le).map(|_| r.between(0, 2)).collect();
            if weights.iter().sum::<i64>() == 0 {
                weights[0] = 1;
            }
            let row: Vec<i64> = (0..n)
                .map(|j| -(0..m_le).map(|i| weights[i] * p.a_le[i][j]).sum::<i64>())
                .collect();
            let bound = -dot(&weights, &p.b_le) - r.between(1, 5);
            p.a_le.push(row);
            p.b_le.push(bound);
        } else if how == 1 && m_eq >= 1 {
            let i = r.below(m_eq as u64) as usize;
            let row: Vec<i64> = p.a_eq[i].iter().map(|v| 2 * v).collect();
            let bound = 2 * p.b_eq[i] + r.between(1, 5);
            p.a_eq.push(row);
            p.b_eq.push(bound);
        } else {
            let j = r.below(n as u64) as usize;
            let unit: Vec<i64> = (0..n).map(|k| i64::from(k == j)).collect();
            let minus: Vec<i64> = unit.iter().map(|v| -v).collect();
            if let Some(lo) = p.lower[j] {
                p.a_le.push(unit);
                p.b_le.push(lo - r.between(1, 3));
            } else if let Some(hi) = p.upper[j] {
                p.a_le.push(minus);
                p.b_le.push(-hi - r.between(1, 3));
            } else {
                p.a_le.push(unit);
                p.b_le.push(0);
                p.a_le.push(minus);
                p.b_le.push(-r.between(1, 3));
            }
        }
    }

    p.c = if p.maximize {
        cost.iter().map(|v| -v).collect()
    } else {
        cost
    };
    p
}

impl Case {
    /// The word-wise FNV-1a hash of the data that `data/gen_lp.py` stores with its reference.
    fn checksum(&self) -> u64 {
        const INF_CODE: i64 = 1 << 40;
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        let mut mix = |v: i64| h = (h ^ v as u64).wrapping_mul(0x0100_0000_01B3);
        mix(self.n as i64);
        mix(self.b_le.len() as i64);
        mix(self.b_eq.len() as i64);
        mix(i64::from(self.maximize));
        self.c.iter().for_each(|v| mix(*v));
        self.a_le.iter().flatten().for_each(|v| mix(*v));
        self.b_le.iter().for_each(|v| mix(*v));
        self.a_eq.iter().flatten().for_each(|v| mix(*v));
        self.b_eq.iter().for_each(|v| mix(*v));
        for (lo, hi) in self.lower.iter().zip(&self.upper) {
            mix(lo.unwrap_or(-INF_CODE));
            mix(hi.unwrap_or(INF_CODE));
        }
        h
    }

    /// The same programme in floats, every number of it exact.
    fn data(&self) -> Data {
        let floats = |v: &[i64]| v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        Data {
            maximize: self.maximize,
            c: floats(&self.c),
            a_le: self.a_le.iter().map(|row| floats(row)).collect(),
            b_le: floats(&self.b_le),
            a_eq: self.a_eq.iter().map(|row| floats(row)).collect(),
            b_eq: floats(&self.b_eq),
            lower: self
                .lower
                .iter()
                .map(|b| b.map_or(f64::NEG_INFINITY, |v| v as f64))
                .collect(),
            upper: self
                .upper
                .iter()
                .map(|b| b.map_or(f64::INFINITY, |v| v as f64))
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Programmes in floats, and what an optimal answer has to satisfy
// ---------------------------------------------------------------------------------------------

/// A linear programme in `f64`, as the tests state it and check answers against it.
#[derive(Debug, Clone)]
struct Data {
    maximize: bool,
    c: Vec<f64>,
    a_le: Vec<Vec<f64>>,
    b_le: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// The matrix with the given rows and `n` columns, whether or not there are any rows.
fn matrix(rows: &[Vec<f64>], n: usize) -> Matrix<f64> {
    Matrix::from_fn(rows.len(), n, |i, j| rows[i][j])
}

impl Data {
    /// A minimisation over `x ≥ 0` with inequalities only.
    fn minimize(c: &[f64], a_le: &[&[f64]], b_le: &[f64]) -> Self {
        Data {
            maximize: false,
            c: c.to_vec(),
            a_le: a_le.iter().map(|row| row.to_vec()).collect(),
            b_le: b_le.to_vec(),
            a_eq: Vec::new(),
            b_eq: Vec::new(),
            lower: vec![0.0; c.len()],
            upper: vec![f64::INFINITY; c.len()],
        }
    }

    /// The same as a maximisation.
    fn maximize(c: &[f64], a_le: &[&[f64]], b_le: &[f64]) -> Self {
        Data {
            maximize: true,
            ..Self::minimize(c, a_le, b_le)
        }
    }

    fn with_eq(mut self, a_eq: &[&[f64]], b_eq: &[f64]) -> Self {
        self.a_eq = a_eq.iter().map(|row| row.to_vec()).collect();
        self.b_eq = b_eq.to_vec();
        self
    }

    fn with_bounds(mut self, lower: &[f64], upper: &[f64]) -> Self {
        self.lower = lower.to_vec();
        self.upper = upper.to_vec();
        self
    }

    fn program(&self) -> LinearProgram<f64> {
        let n = self.c.len();
        let lp = if self.maximize {
            LinearProgram::maximize(&self.c)
        } else {
            LinearProgram::minimize(&self.c)
        };
        lp.subject_to_le(&matrix(&self.a_le, n), &self.b_le)
            .subject_to_eq(&matrix(&self.a_eq, n), &self.b_eq)
            .bounds(&self.lower, &self.upper)
    }

    /// The same programme in `f32`.
    fn program_f32(&self) -> LinearProgram<f32> {
        let n = self.c.len();
        let single = |v: &[f64]| v.iter().map(|x| *x as f32).collect::<Vec<f32>>();
        let single_matrix =
            |rows: &[Vec<f64>]| Matrix::from_fn(rows.len(), n, |i, j| rows[i][j] as f32);
        let lp = if self.maximize {
            LinearProgram::maximize(&single(&self.c))
        } else {
            LinearProgram::minimize(&single(&self.c))
        };
        lp.subject_to_le(&single_matrix(&self.a_le), &single(&self.b_le))
            .subject_to_eq(&single_matrix(&self.a_eq), &single(&self.b_eq))
            .bounds(&single(&self.lower), &single(&self.upper))
    }

    fn solve(&self) -> LpResult<f64> {
        self.solve_with(&LpOptions::new())
    }

    fn solve_with(&self, opts: &LpOptions<f64>) -> LpResult<f64> {
        self.program()
            .solve_with(opts)
            .expect("the programme has a verified answer")
    }

    /// The same programme with its right-hand sides and bounds multiplied by `primal` and its
    /// costs by `dual`: its point is multiplied by `primal`, its dual values by `dual`.
    fn scaled(&self, primal: f64, dual: f64) -> Data {
        let mut p = self.clone();
        for v in p.b_le.iter_mut().chain(&mut p.b_eq) {
            *v *= primal;
        }
        for v in p.lower.iter_mut().chain(&mut p.upper) {
            *v *= primal;
        }
        for v in &mut p.c {
            *v *= dual;
        }
        p
    }
}

/// An `f32` answer in `f64`, to be checked like any other.
fn widen(r: &LpResult<f32>) -> LpResult<f64> {
    let wide = |v: &[f32]| v.iter().map(|x| f64::from(*x)).collect::<Vec<f64>>();
    LpResult {
        status: r.status,
        x: wide(&r.x),
        objective: f64::from(r.objective),
        slack: wide(&r.slack),
        dual_le: wide(&r.dual_le),
        dual_eq: wide(&r.dual_eq),
        reduced_costs: wide(&r.reduced_costs),
        pivots: r.pivots,
    }
}

/// How far an optimal answer is from proving itself optimal. Each part is relative to the size of
/// what it measures, so that it does not change when the data are scaled.
#[derive(Debug, Clone, Copy, Default)]
struct Residuals {
    /// The worst violation of a row, relative to the size of its terms, `|b_i| + Σ_j |a_ij·x_j|`.
    primal: f64,
    /// The worst wrong sign of a multiplier, relative to `‖y‖∞`; the worst difference between a
    /// reduced cost and `c_j − Σ_i y_i·a_ij`, relative to `|c_j| + ‖a_j‖₁·‖y‖∞`, the rounding of
    /// that sum (each multiplier comes out of a solve that mixes them all, so it is known only
    /// relative to the largest); and the worst wrong sign of a reduced cost beyond that rounding,
    /// relative to `|c_j| + |Σ_i y_i·a_ij|`, the two numbers it is the difference of — not to the
    /// multipliers of rows its variable does not meet.
    dual: f64,
    /// The worst product of a multiplier's size and its row's slack, or of the size of a reduced
    /// cost beyond rounding and its variable's distance from its nearer bound, each factor
    /// relative to its scale.
    slackness: f64,
    /// The duality gap against the dual objective of the programme with its bounds,
    /// `b_leᵀ·y_le + b_eqᵀ·y_eq + Σ_j r_j·β_j`, where `β_j` is the bound the sign of `r_j` prices
    /// (for a minimisation `l_j` where `r_j` is positive and `u_j` where it is negative, none where
    /// it is within rounding of zero): by weak duality a bound on the objective of every feasible
    /// point, and infinitely far from the objective when a reduced cost prices an infinite bound.
    /// Relative to the sum of the magnitudes of its terms and of `|c_j|·‖x‖∞`, the scale of the
    /// objective, which a zero optimum has too.
    gap: f64,
}

/// The rounding a reduced cost `c_j − Σ_i y_i·a_ij` is allowed, in units of the precision's
/// epsilon times `|c_j| + ‖a_j‖₁·‖y‖∞`: within it, the reduced cost counts as zero.
const REDUCED_ROUNDING: f64 = 1e4;

impl Residuals {
    fn worst(self) -> f64 {
        self.primal.max(self.dual).max(self.slackness).max(self.gap)
    }

    fn max(self, other: Residuals) -> Residuals {
        Residuals {
            primal: self.primal.max(other.primal),
            dual: self.dual.max(other.dual),
            slackness: self.slackness.max(other.slackness),
            gap: self.gap.max(other.gap),
        }
    }
}

/// `a / b`, zero when `a` is zero whatever `b` is.
fn relative(a: f64, b: f64) -> f64 {
    if a == 0.0 { 0.0 } else { a / b }
}

/// The value of a row at `x` and the size of its terms, summed in the order the library sums.
fn row_value(row: &[f64], x: &[f64]) -> (f64, f64) {
    let mut value = 0.0;
    let mut size = 0.0;
    for (a, v) in row.iter().zip(x) {
        value += a * v;
        size += (a * v).abs();
    }
    (value, size)
}

/// Checks the parts of an answer that hold whatever its status claims: its lengths, the point
/// inside the bounds, and the objective and the slacks computed from the point.
fn consistent(p: &Data, r: &LpResult<f64>) {
    let n = p.c.len();
    assert_eq!(r.x.len(), n);
    assert_eq!(r.slack.len(), p.b_le.len());
    assert_eq!(r.dual_le.len(), p.b_le.len());
    assert_eq!(r.dual_eq.len(), p.b_eq.len());
    assert_eq!(r.reduced_costs.len(), n);
    let objective: f64 = p.c.iter().zip(&r.x).fold(0.0, |s, (c, x)| s + c * x);
    assert_eq!(r.objective, objective);
    for (i, row) in p.a_le.iter().enumerate() {
        // The slack is the right-hand side less the row at the point, to the last bit.
        assert_eq!(r.slack[i], p.b_le[i] - row_value(row, &r.x).0);
    }
}

/// `Σ_j |a_j|`, the size of a row's coefficients.
fn norm(row: &[f64]) -> f64 {
    row.iter().map(|a| a.abs()).sum()
}

/// The largest magnitude of a vector.
fn largest(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |big, x| big.max(x.abs()))
}

/// The worst violation of a row by the point of `r`, relative to `|b_i| + ‖a_i‖₁·‖x‖∞` (the
/// normwise relative residual, which a row whose terms are all rounding does not inflate); the
/// bounds must hold exactly.
fn infeasibility(p: &Data, r: &LpResult<f64>) -> f64 {
    for j in 0..p.c.len() {
        assert!(
            p.lower[j] <= r.x[j] && r.x[j] <= p.upper[j],
            "x[{j}] = {} is outside [{}, {}]",
            r.x[j],
            p.lower[j],
            p.upper[j]
        );
    }
    let x_size = largest(&r.x);
    let mut worst = 0.0f64;
    for (row, b) in p.a_le.iter().zip(&p.b_le) {
        let value = row_value(row, &r.x).0;
        worst = worst.max(relative((value - b).max(0.0), b.abs() + norm(row) * x_size));
    }
    for (row, b) in p.a_eq.iter().zip(&p.b_eq) {
        let value = row_value(row, &r.x).0;
        worst = worst.max(relative((value - b).abs(), b.abs() + norm(row) * x_size));
    }
    worst
}

/// Measures, without any reference, how far `r` is from an optimal answer of `p` with dual values
/// that prove it: the point against the constraints, the multipliers and the reduced costs
/// against the signs of their convention, complementary slackness, and the duality gap against
/// the dual objective of the programme with its bounds.
fn certify(p: &Data, r: &LpResult<f64>) -> Residuals {
    consistent(p, r);
    residuals(p, r, f64::EPSILON)
}

/// [`certify`] for an answer in `f32`: its objective and slacks are checked in the arithmetic
/// they were computed in, the rest in `f64`.
fn certify_single(p: &Data, r: &LpResult<f32>) -> Residuals {
    let single = |v: f64| v as f32;
    let objective =
        p.c.iter()
            .zip(&r.x)
            .fold(0.0f32, |s, (c, x)| s + single(*c) * x);
    assert_eq!(r.objective, objective);
    for (i, row) in p.a_le.iter().enumerate() {
        let value = row
            .iter()
            .zip(&r.x)
            .fold(0.0f32, |s, (a, x)| s + single(*a) * x);
        assert_eq!(r.slack[i], single(p.b_le[i]) - value);
    }
    residuals(p, &widen(r), f64::from(f32::EPSILON))
}

/// The measures of [`certify`], for an answer computed in a precision whose epsilon is
/// `epsilon`.
fn residuals(p: &Data, r: &LpResult<f64>, epsilon: f64) -> Residuals {
    let n = p.c.len();
    assert_eq!(r.status, LpStatus::Optimal);
    let mut out = Residuals {
        primal: infeasibility(p, r),
        ..Residuals::default()
    };
    // A maximisation has the signs of the minimisation, reversed: everything below is said of
    // the minimisation of `sign·c`.
    let sign = if p.maximize { -1.0 } else { 1.0 };
    let y_size = largest(&r.dual_le).max(largest(&r.dual_eq));
    let x_size = largest(&r.x);

    // The multipliers of the inequalities: not positive for a minimisation, zero where the row is
    // slack. Each prices its right-hand side.
    let mut dual_objective = 0.0;
    let mut terms = 0.0;
    for (i, y) in r.dual_le.iter().enumerate() {
        out.dual = out.dual.max(relative((sign * y).max(0.0), y_size));
        let size = p.b_le[i].abs() + norm(&p.a_le[i]) * x_size;
        let slack = relative(r.slack[i].abs(), size).min(1.0);
        out.slackness = out.slackness.max(relative(y.abs(), y_size) * slack);
        dual_objective += sign * y * p.b_le[i];
        terms += (y * p.b_le[i]).abs();
    }
    for (y, b) in r.dual_eq.iter().zip(&p.b_eq) {
        dual_objective += sign * y * b;
        terms += (y * b).abs();
    }

    let mut primal_objective = 0.0;
    let mut unbounded_below = false;
    for j in 0..n {
        // The reduced cost is the cost less the multipliers' share of it.
        let mut share = 0.0;
        let mut column = 0.0;
        for (y, row) in r.dual_le.iter().zip(&p.a_le) {
            share += y * row[j];
            column += row[j].abs();
        }
        for (y, row) in r.dual_eq.iter().zip(&p.a_eq) {
            share += y * row[j];
            column += row[j].abs();
        }
        let rounding = p.c[j].abs() + column * y_size;
        let d = r.reduced_costs[j];
        out.dual = out
            .dual
            .max(relative((d - (p.c[j] - share)).abs(), rounding));
        let (x, lo, hi) = (r.x[j], p.lower[j], p.upper[j]);
        primal_objective += sign * p.c[j] * x;
        terms += (p.c[j] * x).abs() + p.c[j].abs() * x_size;
        if d.abs() <= REDUCED_ROUNDING * epsilon * rounding {
            continue;
        }
        // Beyond rounding, its sign prices the bound the variable is at: not negative at a lower
        // bound, not positive at an upper bound, and no sign at all in between.
        let reduced = sign * d;
        let scale = p.c[j].abs() + share.abs();
        let wrong = match (x == lo, x == hi) {
            (true, true) => 0.0,
            (true, false) => (-reduced).max(0.0),
            (false, true) => reduced.max(0.0),
            (false, false) => reduced.abs(),
        };
        out.dual = out.dual.max(relative(wrong, scale));
        let away = (x - lo).abs().min((hi - x).abs());
        out.slackness = out
            .slackness
            .max(relative(d.abs(), scale) * relative(away, x_size).min(1.0));
        // In the dual objective it prices the bound its sign points to, wherever the variable is.
        let bound = if reduced > 0.0 { lo } else { hi };
        if bound.is_finite() {
            dual_objective += reduced * bound;
            terms += (reduced * bound).abs();
        } else {
            unbounded_below = true;
        }
    }
    out.gap = if unbounded_below {
        f64::INFINITY
    } else {
        relative((primal_objective - dual_objective).abs(), terms)
    };
    out
}

/// Asserts that `r` is an optimal answer of `p` that proves itself to `tol`, and returns how
/// closely it does.
#[track_caller]
fn assert_certified(p: &Data, r: &LpResult<f64>, tol: f64) -> Residuals {
    let res = certify(p, r);
    assert!(
        res.worst() <= tol,
        "the answer does not prove itself to {tol:e}: {res:?}"
    );
    res
}

/// [`assert_certified`] for an answer in `f32`.
#[track_caller]
fn assert_single_certified(p: &Data, r: &LpResult<f32>, tol: f64) -> Residuals {
    let res = certify_single(p, r);
    assert!(
        res.worst() <= tol,
        "the answer does not prove itself to {tol:e}: {res:?}"
    );
    res
}

/// Asserts that `v` is `expected` to `tol`, element by element.
#[track_caller]
fn assert_close(v: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(v.len(), expected.len());
    for (k, (a, b)) in v.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "element {k}: {a} against {b} (all: {v:?})"
        );
    }
}

/// The relative difference of two objectives, against the larger of one and the reference.
fn objective_error(objective: f64, reference: f64) -> f64 {
    (objective - reference).abs() / reference.abs().max(1.0)
}

// ---------------------------------------------------------------------------------------------
// The references of HiGHS
// ---------------------------------------------------------------------------------------------

/// A line of `data/lp_highs.txt`: a seed, the shape and the checksum of its programme, and what
/// HiGHS made of it.
struct Reference {
    seed: u64,
    n: usize,
    m_le: usize,
    m_eq: usize,
    checksum: u64,
    status: LpStatus,
    objective: f64,
}

fn references() -> Vec<Reference> {
    include_str!("data/lp_highs.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(w.len(), 7, "a line has seven fields: {l}");
            let status = match w[5] {
                "O" => LpStatus::Optimal,
                "I" => LpStatus::Infeasible,
                "U" => LpStatus::Unbounded,
                other => panic!("an unknown status {other}"),
            };
            Reference {
                seed: w[0].parse().unwrap(),
                n: w[1].parse().unwrap(),
                m_le: w[2].parse().unwrap(),
                m_eq: w[3].parse().unwrap(),
                checksum: u64::from_str_radix(w[4], 16).unwrap(),
                status,
                objective: if w[6] == "-" {
                    f64::NAN
                } else {
                    f64::from_bits(u64::from_str_radix(w[6], 16).unwrap())
                },
            }
        })
        .collect()
}

/// The references a test runs: all of them, or a handful of the small ones under Miri.
fn cases() -> Vec<Reference> {
    let all = references();
    if cfg!(miri) {
        all.into_iter().filter(|c| c.n <= 6).take(6).collect()
    } else {
        all
    }
}

/// Solves the random programmes of `refs` under `opts`, the data scaled as [`Data::scaled`]
/// says, and asserts that every status is HiGHS's, every optimal objective HiGHS's to 1e-9 and
/// every optimal answer certified to `tol`; returns the worst residuals, the worst objective
/// error and the number of pivots.
fn agree_with_highs(
    refs: &[Reference],
    opts: &LpOptions<f64>,
    primal: f64,
    dual: f64,
    tol: f64,
) -> (Residuals, f64, usize) {
    let mut worst = Residuals::default();
    let mut worst_objective = 0.0f64;
    let mut pivots = 0;
    for c in refs {
        let p = generate(c.seed).data().scaled(primal, dual);
        let r = p
            .program()
            .solve_with(opts)
            .unwrap_or_else(|| panic!("seed {}: no verified answer", c.seed));
        pivots += r.pivots;
        assert_eq!(r.status, c.status, "seed {}", c.seed);
        consistent(&p, &r);
        match r.status {
            LpStatus::Optimal => {
                let err = objective_error(r.objective, c.objective * primal * dual);
                assert!(
                    err <= 1e-9,
                    "seed {}: objective {} against HiGHS's {} scaled",
                    c.seed,
                    r.objective,
                    c.objective
                );
                worst_objective = worst_objective.max(err);
                let res = certify(&p, &r);
                assert!(res.worst() <= tol, "seed {}: {res:?}", c.seed);
                worst = worst.max(res);
            }
            // An unbounded programme is reported at a feasible point.
            LpStatus::Unbounded => {
                assert!(infeasibility(&p, &r) <= tol, "seed {}", c.seed);
            }
            _ => {}
        }
    }
    (worst, worst_objective, pivots)
}

// ---------------------------------------------------------------------------------------------
// Random programmes against HiGHS
// ---------------------------------------------------------------------------------------------

#[test]
fn the_generator_reproduces_the_programmes_of_the_reference_file() {
    let all = references();
    assert_eq!(all.len(), 600);
    for c in cases() {
        let p = generate(c.seed);
        assert_eq!(p.n, c.n, "seed {}", c.seed);
        assert_eq!(p.b_le.len(), c.m_le, "seed {}", c.seed);
        assert_eq!(p.b_eq.len(), c.m_eq, "seed {}", c.seed);
        assert_eq!(p.checksum(), c.checksum, "seed {}", c.seed);
    }
    // The file holds programmes of every status and of every size class.
    let count = |s: LpStatus| all.iter().filter(|c| c.status == s).count();
    assert!(count(LpStatus::Optimal) >= 250);
    assert!(count(LpStatus::Infeasible) >= 100);
    assert!(count(LpStatus::Unbounded) >= 100);
    assert!(all.iter().any(|c| c.n >= 100 && c.m_le + c.m_eq >= 50));
}

#[test]
fn six_hundred_random_programmes_agree_with_highs_and_prove_themselves() {
    let (worst, objective, _) = agree_with_highs(&cases(), &LpOptions::new(), 1.0, 1.0, 1e-9);
    // Measured: the optima within 5.0e-13 of HiGHS's; the residuals of the certificates 3.7e-16
    // (primal), 7.1e-14 (dual), 1.1e-16 (complementary slackness) and 1.3e-16 (duality gap).
    assert!(objective <= 1e-11, "{objective:e}");
    assert!(worst.primal <= 1e-14 && worst.gap <= 1e-14, "{worst:?}");
    assert!(worst.dual <= 1e-11 && worst.slackness <= 1e-11, "{worst:?}");
}

#[test]
fn blands_rule_throughout_reaches_the_same_answers() {
    let opts = LpOptions::new().with_stall_limit(0);
    let (_, objective, pivots) = agree_with_highs(&cases(), &opts, 1.0, 1.0, 1e-9);
    let (_, _, dantzig) = agree_with_highs(&cases(), &LpOptions::new(), 1.0, 1.0, 1e-9);
    assert!(objective <= 1e-11, "{objective:e}");
    // Bland's rule is the slower of the two.
    assert!(cfg!(miri) || pivots > dantzig, "{pivots} against {dantzig}");
}

#[test]
fn the_answers_do_not_depend_on_the_scaling() {
    let opts = LpOptions::new().with_scaling(false);
    let (_, objective, _) = agree_with_highs(&cases(), &opts, 1.0, 1.0, 1e-9);
    assert!(objective <= 1e-11, "{objective:e}");
}

#[test]
fn programmes_in_small_or_large_numbers_are_solved_as_in_moderate_ones() {
    // The point scales with the right-hand sides and the bounds, the dual values with the costs,
    // and the objective with both; the statuses do not change.
    let refs: Vec<Reference> = cases().into_iter().step_by(3).collect();
    for (primal, dual) in [
        (1e-9, 1.0),
        (1e9, 1.0),
        (1.0, 1e-8),
        (1.0, 1e8),
        (1e-3, 1e5),
    ] {
        let (_, objective, _) = agree_with_highs(&refs, &LpOptions::new(), primal, dual, 1e-9);
        assert!(objective <= 1e-11, "{primal:e} {dual:e}: {objective:e}");
    }
}

#[test]
fn bounds_that_stand_for_infinite_ones_change_nothing() {
    // Every infinite bound of the random programmes replaced by ±10³⁰, as users of some solvers
    // write them: the feasible bounded programmes keep their optimum and the infeasible ones their
    // status (the unbounded ones become bounded, and are left out). A variable that started at
    // such a bound, or a step along a direction of rounding that only such a bound stops, would
    // put numbers of 10³⁰ into every row.
    let refs: Vec<Reference> = cases()
        .into_iter()
        .filter(|c| c.status != LpStatus::Unbounded)
        .collect();
    for scaling in [true, false] {
        let opts = LpOptions::new().with_scaling(scaling);
        for c in &refs {
            let mut p = generate(c.seed).data();
            for v in p.lower.iter_mut().chain(&mut p.upper) {
                if v.is_infinite() {
                    *v = 1e30f64.copysign(*v);
                }
            }
            let r = p
                .program()
                .solve_with(&opts)
                .unwrap_or_else(|| panic!("seed {}: no verified answer", c.seed));
            assert_eq!(r.status, c.status, "seed {}", c.seed);
            if r.status == LpStatus::Optimal {
                let err = objective_error(r.objective, c.objective);
                assert!(
                    err <= 1e-9,
                    "seed {}: {} against {}",
                    c.seed,
                    r.objective,
                    c.objective
                );
                assert!(largest(&r.x) < 1e15, "seed {}: {:?}", c.seed, r.x);
                assert_certified(&p, &r, 1e-9);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Textbook examples, against their closed forms
// ---------------------------------------------------------------------------------------------
// The values below are the textbooks' and were checked against scipy.optimize.linprog(method=
// "highs") by data/gen_lp_examples.py, which prints them with the solver's dual values.

#[test]
fn the_wyndor_glass_problem_has_its_textbook_answer_and_shadow_prices() {
    // Hillier and Lieberman, Introduction to Operations Research, ch. 3: maximise 3x + 5y
    // subject to x ≤ 4, 2y ≤ 12 and 3x + 2y ≤ 18. The optimum 36 at (2, 6); the shadow prices of
    // the three plants are 0, 3/2 and 1.
    let p = Data::maximize(
        &[3.0, 5.0],
        &[&[1.0, 0.0], &[0.0, 2.0], &[3.0, 2.0]],
        &[4.0, 12.0, 18.0],
    );
    let r = p.solve();
    assert_close(&r.x, &[2.0, 6.0], 1e-14);
    assert_close(&[r.objective], &[36.0], 1e-14);
    assert_close(&r.slack, &[2.0, 0.0, 0.0], 1e-14);
    assert_close(&r.dual_le, &[0.0, 1.5, 1.0], 1e-14);
    assert_close(&r.reduced_costs, &[0.0, 0.0], 1e-14);
    assert_certified(&p, &r, 1e-14);
}

#[test]
fn chvatals_first_example_has_its_textbook_answer() {
    // Chvátal, Linear Programming (1983), ch. 2: maximise 5x₁ + 4x₂ + 3x₃ subject to
    // 2x₁ + 3x₂ + x₃ ≤ 5, 4x₁ + x₂ + 2x₃ ≤ 11 and 3x₁ + 4x₂ + 2x₃ ≤ 8: 13 at (2, 0, 1), with the
    // multipliers (1, 0, 1); x₂ would cost 3 a unit.
    let p = Data::maximize(
        &[5.0, 4.0, 3.0],
        &[&[2.0, 3.0, 1.0], &[4.0, 1.0, 2.0], &[3.0, 4.0, 2.0]],
        &[5.0, 11.0, 8.0],
    );
    let r = p.solve();
    assert_close(&r.x, &[2.0, 0.0, 1.0], 1e-14);
    assert_close(&[r.objective], &[13.0], 1e-14);
    assert_close(&r.dual_le, &[1.0, 0.0, 1.0], 1e-14);
    assert_close(&r.reduced_costs, &[0.0, -3.0, 0.0], 1e-14);
    assert_certified(&p, &r, 1e-14);
}

#[test]
fn tahas_two_phase_example_with_an_equation_and_both_inequalities() {
    // Taha, Operations Research, the two-phase method: minimise 4x₁ + x₂ subject to
    // 3x₁ + x₂ = 3, 4x₁ + 3x₂ ≥ 6 and x₁ + 2x₂ ≤ 4: 17/5 at (2/5, 9/5). The origin breaks the
    // equation and the second inequality, so phase one has work to do.
    let p = Data::minimize(&[4.0, 1.0], &[&[-4.0, -3.0], &[1.0, 2.0]], &[-6.0, 4.0])
        .with_eq(&[&[3.0, 1.0]], &[3.0]);
    let r = p.solve();
    assert_close(&r.x, &[0.4, 1.8], 1e-14);
    assert_close(&[r.objective], &[3.4], 1e-14);
    assert_close(&r.dual_eq, &[1.4], 1e-14);
    assert_close(&r.dual_le, &[0.0, -0.2], 1e-14);
    assert_close(&r.slack, &[1.0, 0.0], 1e-14);
    assert_certified(&p, &r, 1e-14);
}

#[test]
fn vanderbeis_example_starts_from_an_infeasible_origin() {
    // Vanderbei, Linear Programming: Foundations and Extensions, ch. 2, on initialisation:
    // maximise −2x₁ − x₂ subject to −x₁ + x₂ ≤ −1, −x₁ − 2x₂ ≤ −2 and x₂ ≤ 1: −3 at (4/3, 1/3),
    // with the multipliers (1, 1, 0).
    let p = Data::maximize(
        &[-2.0, -1.0],
        &[&[-1.0, 1.0], &[-1.0, -2.0], &[0.0, 1.0]],
        &[-1.0, -2.0, 1.0],
    );
    let r = p.solve();
    assert_close(&r.x, &[4.0 / 3.0, 1.0 / 3.0], 1e-14);
    assert_close(&[r.objective], &[-3.0], 1e-14);
    assert_close(&r.dual_le, &[1.0, 1.0, 0.0], 1e-14);
    assert_certified(&p, &r, 1e-14);
}

#[test]
fn the_diet_problem_buys_the_cheapest_mixture() {
    // Minimise 3x + 4y subject to 2x + y ≥ 8 and x + 3y ≥ 10: 18 at (14/5, 12/5). A unit more of
    // either requirement costs 1.
    let p = Data::minimize(&[3.0, 4.0], &[&[-2.0, -1.0], &[-1.0, -3.0]], &[-8.0, -10.0]);
    let r = p.solve();
    assert_close(&r.x, &[2.8, 2.4], 1e-14);
    assert_close(&[r.objective], &[18.0], 1e-14);
    assert_close(&r.dual_le, &[-1.0, -1.0], 1e-14);
    assert_certified(&p, &r, 1e-14);
}

#[test]
fn a_maximisation_is_the_minimisation_of_the_negated_objective() {
    // The same point; the objective and every dual value change sign.
    let a: [&[f64]; 3] = [&[2.0, 3.0, 1.0], &[4.0, 1.0, 2.0], &[3.0, 4.0, 2.0]];
    let b = [5.0, 11.0, 8.0];
    let max = Data::maximize(&[5.0, 4.0, 3.0], &a, &b).solve();
    let min = Data::minimize(&[-5.0, -4.0, -3.0], &a, &b).solve();
    assert_eq!(max.x, min.x);
    assert_eq!(max.objective, -min.objective);
    let negated = |v: &[f64]| v.iter().map(|x| -x).collect::<Vec<f64>>();
    assert_eq!(max.dual_le, negated(&min.dual_le));
    assert_eq!(max.reduced_costs, negated(&min.reduced_costs));
}

#[test]
fn a_degenerate_assignment_problem_has_a_permutation_for_its_answer() {
    // Six workers, six jobs, the cost matrix of data/gen_lp_examples.py: each row and column of
    // x sums to one, which makes one of the twelve equations redundant and every vertex
    // degenerate. The optimum, 17, is scipy.optimize.linear_sum_assignment's.
    let n = 6;
    let cost: Vec<f64> = (0..n * n)
        .map(|k| ((7 * (k / n) + 3 * (k % n) * (k % n) + 5) % 17 + 1) as f64)
        .collect();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        rows.push((0..n * n).map(|k| f64::from(k / n == i)).collect());
    }
    for j in 0..n {
        rows.push((0..n * n).map(|k| f64::from(k % n == j)).collect());
    }
    let mut p = Data::minimize(&cost, &[], &[]);
    p.a_eq = rows;
    p.b_eq = vec![1.0; 2 * n];
    let r = p.solve();
    assert_close(&[r.objective], &[17.0], 1e-14);
    // A vertex of the assignment polytope is a permutation.
    for v in &r.x {
        assert!(*v == 0.0 || (v - 1.0).abs() < 1e-14, "{v}");
    }
    assert_certified(&p, &r, 1e-14);
}

#[test]
fn a_balanced_transportation_problem_has_a_redundant_equation() {
    // Supplies 20, 30 and 25, demands 10, 35, 15 and 15: the supplies and the demands both sum to
    // 75, so the seven equations have rank six. HiGHS: 625.
    let cost = [
        8.0, 6.0, 10.0, 9.0, 9.0, 12.0, 13.0, 7.0, 14.0, 9.0, 16.0, 5.0,
    ];
    let mut p = Data::minimize(&cost, &[], &[]);
    for (i, s) in [20.0, 30.0, 25.0].into_iter().enumerate() {
        p.a_eq
            .push((0..12).map(|k| f64::from(k / 4 == i)).collect());
        p.b_eq.push(s);
    }
    for (j, d) in [10.0, 35.0, 15.0, 15.0].into_iter().enumerate() {
        p.a_eq
            .push((0..12).map(|k| f64::from(k % 4 == j)).collect());
        p.b_eq.push(d);
    }
    let r = p.solve();
    assert_close(&[r.objective], &[625.0], 1e-14);
    assert_certified(&p, &r, 1e-14);
}

// ---------------------------------------------------------------------------------------------
// Degeneracy: the examples that make the simplex method cycle
// ---------------------------------------------------------------------------------------------

/// Chvátal's example of cycling (Linear Programming, 1983, ch. 3): maximise
/// 10x₁ − 57x₂ − 9x₃ − 24x₄ subject to 0.5x₁ − 5.5x₂ − 2.5x₃ + 9x₄ ≤ 0,
/// 0.5x₁ − 1.5x₂ − 0.5x₃ + x₄ ≤ 0 and x₁ ≤ 1. Under the rule of the largest coefficient the
/// simplex method returns to its first basis after six degenerate pivots, whichever way the ties
/// of the ratio test are broken (data/gen_lp_examples.py follows it in exact arithmetic). Its
/// numbers are exact in binary, so the degenerate ratios are exactly zero here too.
fn chvatal_cycling() -> Data {
    Data::maximize(
        &[10.0, -57.0, -9.0, -24.0],
        &[
            &[0.5, -5.5, -2.5, 9.0],
            &[0.5, -1.5, -0.5, 1.0],
            &[1.0, 0.0, 0.0, 0.0],
        ],
        &[0.0, 0.0, 1.0],
    )
}

#[test]
fn chvatals_example_cycles_under_dantzigs_rule_alone_and_not_with_the_fallback() {
    let p = chvatal_cycling();
    // Dantzig's rule throughout, on the data as they are: six degenerate pivots bring the method
    // back to the basis it started from, its point, multipliers and reduced costs all as they
    // were, and six more do it again.
    let dantzig = |max_iter| {
        LpOptions::new()
            .with_stall_limit(usize::MAX)
            .with_scaling(false)
            .with_max_iter(max_iter)
    };
    let start = p.solve_with(&dantzig(0));
    for k in [6, 12] {
        let r = p.solve_with(&dantzig(k));
        assert_eq!(
            (r.status, r.pivots, r.objective),
            (LpStatus::IterationLimit, k, 0.0)
        );
        assert_eq!(r.x, start.x);
        assert_eq!(r.dual_le, start.dual_le);
        assert_eq!(r.reduced_costs, start.reduced_costs);
    }
    assert_ne!(p.solve_with(&dantzig(3)).reduced_costs, start.reduced_costs);
    // Bland's rule throughout takes the seven pivots it takes in exact arithmetic; taking over
    // once six pivots in a row have left the objective where it was, it ends the cycle at the
    // thirteenth.
    let unscaled = LpOptions::new().with_scaling(false);
    assert_eq!(p.solve_with(&unscaled.with_stall_limit(0)).pivots, 7);
    assert_eq!(p.solve_with(&unscaled.with_stall_limit(6)).pivots, 13);
    // The optimum 1 at (1, 0, 1, 0), whatever the stall limit and the scaling.
    for stall_limit in [0, 1, 6, 100] {
        for scaling in [true, false] {
            let opts = LpOptions::new()
                .with_stall_limit(stall_limit)
                .with_scaling(scaling);
            let r = p.solve_with(&opts);
            assert_close(&r.x, &[1.0, 0.0, 1.0, 0.0], 1e-14);
            assert_close(&[r.objective], &[1.0], 1e-14);
            assert_close(&r.dual_le, &[0.0, 18.0, 1.0], 1e-13);
            assert_close(&r.reduced_costs, &[0.0, -30.0, 0.0, -42.0], 1e-13);
            assert_certified(&p, &r, 1e-14);
        }
    }
}

#[test]
fn beales_example_of_cycling_is_solved() {
    // Beale (1955): minimise −3/4 x₄ + 150x₅ − 1/50 x₆ + 6x₇ subject to
    // 1/4 x₄ − 60x₅ − 1/25 x₆ + 9x₇ ≤ 0, 1/2 x₄ − 90x₅ − 1/50 x₆ + 3x₇ ≤ 0 and x₆ ≤ 1: −1/20 at
    // (1/25, 0, 1, 0). It cycles under the largest coefficient when the ratio test takes the
    // lowest of the tied rows.
    let p = Data::minimize(
        &[-0.75, 150.0, -0.02, 6.0],
        &[
            &[0.25, -60.0, -0.04, 9.0],
            &[0.5, -90.0, -0.02, 3.0],
            &[0.0, 0.0, 1.0, 0.0],
        ],
        &[0.0, 0.0, 1.0],
    );
    for opts in [
        LpOptions::new(),
        LpOptions::new().with_stall_limit(0),
        LpOptions::new().with_scaling(false),
    ] {
        let r = p.solve_with(&opts);
        assert_close(&r.x, &[0.04, 0.0, 1.0, 0.0], 1e-14);
        assert_close(&[r.objective], &[-0.05], 1e-14);
        assert_certified(&p, &r, 1e-14);
    }
}

#[test]
fn kuhns_example_of_cycling_is_solved() {
    // Kuhn's example of cycling: minimise −2x₁ − 3x₂ + x₃ + 12x₄ subject to
    // −2x₁ − 9x₂ + x₃ + 9x₄ ≤ 0, x₁/3 + x₂ − x₃/3 − 2x₄ ≤ 0 and 2x₁ + 3x₂ − x₃ − 12x₄ ≤ 2: −2 at
    // (2, 0, 2, 0) (HiGHS). It cycles under the largest coefficient when the ratio test prefers
    // the largest pivot, as a Harris test does.
    let p = Data::minimize(
        &[-2.0, -3.0, 1.0, 12.0],
        &[
            &[-2.0, -9.0, 1.0, 9.0],
            &[1.0 / 3.0, 1.0, -1.0 / 3.0, -2.0],
            &[2.0, 3.0, -1.0, -12.0],
        ],
        &[0.0, 0.0, 2.0],
    );
    for opts in [
        LpOptions::new(),
        LpOptions::new().with_stall_limit(0),
        LpOptions::new().with_scaling(false),
    ] {
        let r = p.solve_with(&opts);
        assert_close(&[r.objective], &[-2.0], 1e-14);
        assert_certified(&p, &r, 1e-14);
    }
}

#[test]
fn hall_and_mckinnons_example_of_cycling_is_unbounded() {
    // Hall and McKinnon (2004), one of the smallest examples on which the largest-coefficient
    // rule with the largest pivot cycles: minimise −2.3x₁ − 2.15x₂ + 13.55x₃ + 0.4x₄ subject to
    // 0.4x₁ + 0.2x₂ − 1.4x₃ − 0.2x₄ ≤ 0 and −7.8x₁ − 1.4x₂ + 7.8x₃ + 0.4x₄ ≤ 0. It is unbounded.
    let p = Data::minimize(
        &[-2.3, -2.15, 13.55, 0.4],
        &[&[0.4, 0.2, -1.4, -0.2], &[-7.8, -1.4, 7.8, 0.4]],
        &[0.0, 0.0],
    );
    for opts in [
        LpOptions::new(),
        LpOptions::new().with_stall_limit(0),
        LpOptions::new().with_scaling(false),
    ] {
        let r = p.solve_with(&opts);
        assert_eq!(r.status, LpStatus::Unbounded);
        assert_eq!(infeasibility(&p, &r), 0.0);
    }
}

// ---------------------------------------------------------------------------------------------
// The cubes of Klee and Minty
// ---------------------------------------------------------------------------------------------

/// The cube of Klee and Minty (1972) in `n` dimensions: maximise `Σ_j β^(n−j) x_j` subject to
/// `2 Σ_{j<i} β^(i−j) x_j + x_i ≤ γ^(i−1)` for each `i`, `x ≥ 0`. Chvátal (1983) states it with
/// `β = 10` and `γ = 100`; `β = 2` and `γ = 5` give a cube of the same shape in smaller numbers.
/// Its optimum is `γ^(n−1)`, at `x_n = γ^(n−1)` and every other `x_j = 0`, and Dantzig's rule
/// from the origin visits all `2ⁿ` vertices on the way (data/gen_lp_examples.py counts the pivots
/// in exact arithmetic).
fn klee_minty(n: usize, beta: f64, gamma: f64) -> Data {
    let c: Vec<f64> = (1..=n).map(|j| beta.powi((n - j) as i32)).collect();
    let mut p = Data::maximize(&c, &[], &[]);
    p.a_le = (1..=n)
        .map(|i| {
            (1..=n)
                .map(|j| match j.cmp(&i) {
                    core::cmp::Ordering::Less => 2.0 * beta.powi((i - j) as i32),
                    core::cmp::Ordering::Equal => 1.0,
                    core::cmp::Ordering::Greater => 0.0,
                })
                .collect()
        })
        .collect();
    p.b_le = (1..=n).map(|i| gamma.powi(i as i32 - 1)).collect();
    p
}

/// The optimum of [`klee_minty`].
fn klee_minty_optimum(n: usize, gamma: f64) -> Vec<f64> {
    let mut x = vec![0.0; n];
    x[n - 1] = gamma.powi(n as i32 - 1);
    x
}

#[test]
fn klee_minty_cubes_take_every_vertex_under_dantzigs_rule_and_end_at_the_optimum() {
    // On the data as they are, Dantzig's rule visits all 2ⁿ vertices: 2ⁿ − 1 pivots. Every
    // number of the cube in twos and fives up to n = 13 is an integer below 2⁵³, so the path is
    // the one of exact arithmetic.
    let largest = if cfg!(miri) { 4 } else { 13 };
    let unscaled = LpOptions::new().with_scaling(false);
    for n in 1..=largest {
        let p = klee_minty(n, 2.0, 5.0);
        let r = p.solve_with(&unscaled);
        assert_eq!(r.pivots, (1 << n) - 1, "n = {n}");
        let x = klee_minty_optimum(n, 5.0);
        assert_eq!(r.x, x, "n = {n}");
        assert_eq!(r.objective, x[n - 1], "n = {n}");
        assert_certified(&p, &r, 1e-15);
        // With the scaling the method takes another way, one pivot, to the same place.
        let r = p.solve();
        assert_eq!((&r.x, r.pivots), (&x, 1), "n = {n}");
        assert_certified(&p, &r, 1e-15);
    }
}

#[test]
fn chvatals_klee_minty_cubes_are_solved() {
    // The cube in tens and hundreds has numbers up to 10²⁰. Without the scaling, Dantzig's rule
    // takes its 2ⁿ − 1 pivots up to n = 9, the last one through numbers past 2⁵³, to the
    // optimum (its bases have pivots of 10⁻⁷ beside ones of 10⁸, which a test of singularity
    // against the largest pivot took for singular). At n = 10 and 11 the steps reach 10¹⁸ and
    // 10²⁰, whose rounding is hundreds of units: the point computed again from the data at a
    // factorisation breaks x₁ ≥ 0 by 40 to 4·10⁶, far more than the error the basis gives x₁,
    // phase one has to resume after every factorisation, and after ten times the method gives
    // up. The answer is None, never a wrong point (with a precision floor of 64 units in the last
    // place of 10¹⁸ for every bound, the method once stopped at a point 5e-8 below the maximum,
    // which the certificate of an optimum refused). With the scaling, every cube up to n = 11 is
    // solved, in one pivot.
    let largest = if cfg!(miri) { 3 } else { 11 };
    let unscaled = LpOptions::new().with_scaling(false);
    for n in 1..=largest {
        let p = klee_minty(n, 10.0, 100.0);
        let x = klee_minty_optimum(n, 100.0);
        if n <= 9 {
            let r = p.solve_with(&unscaled);
            assert_eq!(r.x, x, "n = {n}");
            assert_eq!(r.pivots, (1 << n) - 1, "n = {n}");
            assert_certified(&p, &r, 1e-15);
        } else {
            let answer = p.program().solve_with(&unscaled);
            assert!(
                answer
                    .as_ref()
                    .is_none_or(|r| r.status == LpStatus::Optimal && r.x == x),
                "n = {n}: {answer:?}"
            );
        }
        let r = p.solve();
        assert_eq!((&r.x, r.objective, r.pivots), (&x, x[n - 1], 1), "n = {n}");
        assert_certified(&p, &r, 1e-15);
    }
}

#[test]
fn an_iteration_limit_is_reported_with_the_point_reached() {
    let p = klee_minty(8, 2.0, 5.0);
    let opts = LpOptions::new().with_scaling(false).with_max_iter(20);
    let r = p.solve_with(&opts);
    assert_eq!(r.status, LpStatus::IterationLimit);
    assert_eq!(r.pivots, 20);
    // The point is a vertex of the cube on the way, feasible but not optimal.
    consistent(&p, &r);
    assert!(infeasibility(&p, &r) <= 1e-15);
    assert!(r.objective < 5f64.powi(7));
    // A budget of zero pivots leaves the start; one that is just enough is enough.
    let r = p.solve_with(&opts.with_max_iter(0));
    assert_eq!(
        (r.status, r.pivots, r.objective),
        (LpStatus::IterationLimit, 0, 0.0)
    );
    let r = p.solve_with(&opts.with_max_iter(255));
    assert_eq!((r.status, r.pivots), (LpStatus::Optimal, 255));
}

// ---------------------------------------------------------------------------------------------
// The shapes a programme can have
// ---------------------------------------------------------------------------------------------

#[test]
fn without_constraints_each_variable_goes_to_the_bound_its_cost_prefers() {
    // The third variable is fixed and the fourth starts at the bound its cost prefers; the first
    // and the fifth start at zero, inside their bounds, and the second at its lower bound, and
    // each of those three moves to the bound its cost points to by a flip.
    let p = Data::minimize(&[1.0, -1.0, 0.0, 2.0, -3.0], &[], &[])
        .with_bounds(&[-2.0, 0.0, 5.0, 0.0, -1.0], &[3.0, 7.0, 5.0, 4.0, 1.0]);
    let r = p.solve();
    assert_eq!(r.status, LpStatus::Optimal);
    assert_eq!(r.x, [-2.0, 7.0, 5.0, 0.0, 1.0]);
    assert_eq!(r.objective, -12.0);
    // The reduced costs are the costs: the multipliers of the bounds.
    assert_eq!(r.reduced_costs, [1.0, -1.0, 0.0, 2.0, -3.0]);
    assert_eq!(r.pivots, 3);
    assert_certified(&p, &r, 0.0);
}

#[test]
fn without_constraints_a_cost_towards_an_infinite_bound_is_unbounded() {
    let inf = f64::INFINITY;
    for (c, lower, upper) in [
        (-1.0, 0.0, inf),
        (1.0, -inf, 0.0),
        (1.0, -inf, inf),
        (-1.0, -inf, inf),
    ] {
        let p = Data::minimize(&[0.0, c], &[], &[]).with_bounds(&[0.0, lower], &[1.0, upper]);
        let r = p.solve();
        assert_eq!(r.status, LpStatus::Unbounded, "{c} on [{lower}, {upper}]");
        let r = Data {
            maximize: true,
            ..p
        }
        .solve();
        let bounded = (c > 0.0 && upper.is_finite()) || (c < 0.0 && lower.is_finite());
        let expected = if bounded {
            LpStatus::Optimal
        } else {
            LpStatus::Unbounded
        };
        assert_eq!(r.status, expected, "max {c} on [{lower}, {upper}]");
    }
}

#[test]
fn empty_constraint_matrices_add_nothing() {
    let none = Data::minimize(&[1.0, -2.0], &[], &[]).with_bounds(&[0.0, 0.0], &[1.0, 1.0]);
    let r = none.solve();
    let given = LinearProgram::minimize(&[1.0, -2.0])
        .subject_to_le(&Matrix::zeros(0, 2), &[])
        .subject_to_eq(&Matrix::zeros(0, 5), &[])
        .bounds(&[0.0, 0.0], &[1.0, 1.0])
        .solve()
        .unwrap();
    assert_eq!(given, r);
    assert_eq!(r.x, [0.0, 1.0]);
    assert!(r.slack.is_empty() && r.dual_le.is_empty() && r.dual_eq.is_empty());
}

#[test]
fn a_programme_of_no_variables_is_feasible_when_its_right_hand_sides_allow_it() {
    let lp = |b_le: &[f64], b_eq: &[f64]| {
        LinearProgram::<f64>::minimize(&[])
            .subject_to_le(&Matrix::zeros(b_le.len(), 0), b_le)
            .subject_to_eq(&Matrix::zeros(b_eq.len(), 0), b_eq)
            .solve()
            .unwrap()
    };
    let r = lp(&[0.0, 3.0], &[0.0]);
    assert_eq!(
        (r.status, r.objective, r.pivots),
        (LpStatus::Optimal, 0.0, 0)
    );
    assert_eq!(r.slack, [0.0, 3.0]);
    assert_eq!(lp(&[1.0, -1e-6], &[]).status, LpStatus::Infeasible);
    assert_eq!(lp(&[], &[0.5]).status, LpStatus::Infeasible);
    assert_eq!(lp(&[], &[]).status, LpStatus::Optimal);
}

#[test]
fn with_a_zero_objective_any_feasible_point_is_optimal() {
    let p = Data::minimize(
        &[0.0, 0.0, 0.0],
        &[&[1.0, 1.0, 1.0], &[-1.0, 2.0, 0.0]],
        &[4.0, -1.0],
    )
    .with_eq(&[&[1.0, -1.0, 1.0]], &[1.0]);
    let r = p.solve();
    assert_eq!(r.objective, 0.0);
    assert!(r.dual_le.iter().chain(&r.dual_eq).all(|y| *y == 0.0));
    assert!(r.reduced_costs.iter().all(|d| *d == 0.0));
    assert_certified(&p, &r, 1e-15);
    // Maximised the same.
    let r = Data {
        maximize: true,
        ..p.clone()
    }
    .solve();
    assert_certified(
        &Data {
            maximize: true,
            ..p
        },
        &r,
        1e-15,
    );
}

#[test]
fn a_single_variable_meets_every_status() {
    let inf = f64::INFINITY;
    let one = |c: f64, a: f64, b: f64, lower: f64, upper: f64| {
        Data::minimize(&[c], &[&[a]], &[b]).with_bounds(&[lower], &[upper])
    };
    // min x with −x ≤ −2: 2.
    let r = one(1.0, -1.0, -2.0, 0.0, inf).solve();
    assert_eq!(
        (r.status, r.x[0], r.dual_le[0]),
        (LpStatus::Optimal, 2.0, -1.0)
    );
    // min −x with x ≤ 5 on [0, 3]: the bound comes first, at 3.
    let r = one(-1.0, 1.0, 5.0, 0.0, 3.0).solve();
    assert_eq!(
        (r.status, r.x[0], r.slack[0], r.dual_le[0]),
        (LpStatus::Optimal, 3.0, 2.0, 0.0)
    );
    assert_eq!(r.reduced_costs[0], -1.0);
    // min −x with x ≤ 5 and nothing else: 5.
    let r = one(-1.0, 1.0, 5.0, -inf, inf).solve();
    assert_eq!((r.status, r.x[0]), (LpStatus::Optimal, 5.0));
    // min x with x ≤ 5 and x free: unbounded below.
    assert_eq!(
        one(1.0, 1.0, 5.0, -inf, inf).solve().status,
        LpStatus::Unbounded
    );
    // x ≤ −1 with x ≥ 0: infeasible.
    assert_eq!(
        one(1.0, 1.0, -1.0, 0.0, inf).solve().status,
        LpStatus::Infeasible
    );
    // An equation: 2x = 3.
    let p = Data::minimize(&[7.0], &[], &[]).with_eq(&[&[2.0]], &[3.0]);
    let r = p.solve();
    assert_eq!(
        (r.status, r.x[0], r.dual_eq[0]),
        (LpStatus::Optimal, 1.5, 3.5)
    );
    assert_certified(&p, &r, 1e-15);
}

#[test]
fn a_programme_of_free_variables_only_is_solved_at_a_corner() {
    // Minimise x + y over all of the plane subject to x + 2y ≥ 4 and 3x + y ≥ 6: the corner
    // (8/5, 6/5), 14/5, the multipliers (2/5, 1/5) of the inequalities as given.
    let inf = f64::INFINITY;
    let p = Data::minimize(&[1.0, 1.0], &[&[-1.0, -2.0], &[-3.0, -1.0]], &[-4.0, -6.0])
        .with_bounds(&[-inf, -inf], &[inf, inf]);
    let r = p.solve();
    assert_close(&r.x, &[1.6, 1.2], 1e-15);
    assert_close(&r.dual_le, &[-0.4, -0.2], 1e-15);
    assert_certified(&p, &r, 1e-15);
    // Maximised, nothing stops the plane.
    assert_eq!(
        Data {
            maximize: true,
            ..p
        }
        .solve()
        .status,
        LpStatus::Unbounded
    );
}

#[test]
fn a_programme_of_equations_only_is_solved_by_their_solution() {
    let inf = f64::INFINITY;
    let free = [-inf; 3];
    let up = [inf; 3];
    // Three independent equations in three free variables: their one solution, (1, −2, 3).
    let a: [&[f64]; 3] = [&[2.0, 1.0, -1.0], &[-3.0, -1.0, 2.0], &[-2.0, 1.0, 2.0]];
    let b = [-3.0, 5.0, 2.0];
    let p = Data::minimize(&[1.0, 1.0, 1.0], &[], &[])
        .with_eq(&a, &b)
        .with_bounds(&free, &up);
    let r = p.solve();
    assert_close(&r.x, &[1.0, -2.0, 3.0], 1e-14);
    assert_certified(&p, &r, 1e-14);
    // The same with x ≥ 0 is infeasible: the solution is unique and has a negative part.
    let p = Data::minimize(&[1.0, 1.0, 1.0], &[], &[]).with_eq(&a, &b);
    assert_eq!(p.solve().status, LpStatus::Infeasible);
}

#[test]
fn redundant_equations_do_not_break_phase_one() {
    // x + y + z = 4 and x − y = 1, then the sum of the two, twice the first, and the first
    // again: rank two. Minimise z − x: (5/2, 3/2, 0) at −5/2.
    let rows: [&[f64]; 5] = [
        &[1.0, 1.0, 1.0],
        &[1.0, -1.0, 0.0],
        &[2.0, 0.0, 1.0],
        &[2.0, 2.0, 2.0],
        &[1.0, 1.0, 1.0],
    ];
    let p = Data::minimize(&[-1.0, 0.0, 1.0], &[], &[]).with_eq(&rows, &[4.0, 1.0, 5.0, 8.0, 4.0]);
    for opts in [LpOptions::new(), LpOptions::new().with_stall_limit(0)] {
        let r = p.solve_with(&opts);
        assert_close(&r.x, &[2.5, 1.5, 0.0], 1e-15);
        assert_close(&[r.objective], &[-2.5], 1e-15);
        assert_certified(&p, &r, 1e-15);
    }
    // The last copy of the first equation contradicts it by a little: no feasible point.
    let p = p.with_eq(&rows, &[4.0, 1.0, 5.0, 8.0, 4.0 + 1e-6]);
    assert_eq!(p.solve().status, LpStatus::Infeasible);
}

#[test]
fn rows_and_columns_of_zeros_hold_as_far_as_their_right_hand_sides_allow() {
    // 0 ≤ 3 holds and 0 = 0 holds; a variable in no row goes to the bound its cost prefers.
    let p = Data::minimize(&[1.0, -1.0], &[&[0.0, 0.0], &[1.0, 0.0]], &[3.0, 2.0])
        .with_eq(&[&[0.0, 0.0]], &[0.0])
        .with_bounds(&[-1.0, 0.0], &[f64::INFINITY, 4.0]);
    let r = p.solve();
    assert_eq!(r.x, [-1.0, 4.0]);
    assert_certified(&p, &r, 0.0);
    // 0 ≤ −1 does not hold, nor does 0 = 2.
    let p1 = Data::minimize(&[1.0], &[&[0.0]], &[-1.0]);
    assert_eq!(p1.solve().status, LpStatus::Infeasible);
    let p2 = Data::minimize(&[1.0], &[], &[]).with_eq(&[&[0.0]], &[2.0]);
    assert_eq!(p2.solve().status, LpStatus::Infeasible);
}

#[test]
fn crossed_bounds_are_infeasible_and_equal_bounds_fix_a_variable() {
    let p =
        Data::minimize(&[1.0, 1.0], &[&[1.0, 1.0]], &[10.0]).with_bounds(&[0.0, 2.0], &[5.0, 1.0]);
    assert_eq!(p.solve().status, LpStatus::Infeasible);
    let p = p.with_bounds(&[0.0, 2.5], &[5.0, 2.5]);
    let r = p.solve();
    assert_eq!(r.x, [0.0, 2.5]);
    assert_certified(&p, &r, 0.0);
    // A fixed variable that cannot be fitted: x + y ≤ 10 with y fixed at 12.
    assert_eq!(
        p.with_bounds(&[0.0, 12.0], &[5.0, 12.0]).solve().status,
        LpStatus::Infeasible
    );
}

#[test]
fn an_optimal_start_takes_no_pivots() {
    // At the origin with costs that only grow from it.
    let p = Data::minimize(&[1.0, 2.0], &[&[1.0, 1.0]], &[4.0]);
    let r = p.solve();
    assert_eq!((r.status, r.pivots), (LpStatus::Optimal, 0));
    assert_eq!(r.x, [0.0, 0.0]);
    assert_eq!(r.slack, [4.0]);
    assert_certified(&p, &r, 0.0);
}

#[test]
fn a_degenerate_vertex_with_zero_right_hand_sides_is_optimal() {
    // Maximise x + y subject to x − y ≤ 0, −x + y ≤ 0 and x + y ≤ 0... that is, the origin only,
    // with three rows tight there.
    let p = Data::maximize(
        &[1.0, 1.0],
        &[&[1.0, -1.0], &[-1.0, 1.0], &[1.0, 1.0]],
        &[0.0, 0.0, 0.0],
    );
    let r = p.solve();
    assert_eq!((r.status, r.objective), (LpStatus::Optimal, 0.0));
    assert_certified(&p, &r, 1e-15);
}

// ---------------------------------------------------------------------------------------------
// Invalid input
// ---------------------------------------------------------------------------------------------

#[test]
fn data_that_are_not_finite_are_refused() {
    let base =
        || Data::minimize(&[1.0, 1.0], &[&[1.0, 1.0]], &[4.0]).with_eq(&[&[1.0, -1.0]], &[0.0]);
    assert!(base().program().solve().is_some());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut p = base();
        p.c[1] = bad;
        assert!(p.program().solve().is_none());
        let mut p = base();
        p.a_le[0][0] = bad;
        assert!(p.program().solve().is_none());
        let mut p = base();
        p.b_le[0] = bad;
        assert!(p.program().solve().is_none());
        let mut p = base();
        p.a_eq[0][1] = bad;
        assert!(p.program().solve().is_none());
        let mut p = base();
        p.b_eq[0] = bad;
        assert!(p.program().solve().is_none());
    }
}

#[test]
fn bounds_that_mean_nothing_are_refused() {
    let inf = f64::INFINITY;
    let lp = |lower: f64, upper: f64| {
        LinearProgram::minimize(&[1.0])
            .bounds(&[lower], &[upper])
            .solve()
    };
    assert!(lp(f64::NAN, 1.0).is_none());
    assert!(lp(0.0, f64::NAN).is_none());
    assert!(lp(inf, inf).is_none());
    assert!(lp(-inf, -inf).is_none());
    assert_eq!(lp(-inf, inf).unwrap().status, LpStatus::Unbounded);
}

#[test]
fn a_tolerance_that_is_not_positive_and_finite_is_refused() {
    let lp = LinearProgram::minimize(&[1.0]);
    for tol in [0.0, -1e-9, f64::NAN, f64::INFINITY] {
        assert!(
            lp.solve_with(&LpOptions::new().with_tol(tol)).is_none(),
            "{tol}"
        );
    }
    assert!(lp.solve_with(&LpOptions::new().with_tol(1e-3)).is_some());
}

#[test]
#[should_panic(expected = "the constraints have not one column for each variable")]
fn inequalities_of_the_wrong_width_panic() {
    let _ = LinearProgram::minimize(&[1.0, 2.0]).subject_to_le(&Matrix::zeros(1, 3), &[1.0]);
}

#[test]
#[should_panic(expected = "there is not one right-hand side for each row of the constraints")]
fn equations_without_their_right_hand_sides_panic() {
    let _ = LinearProgram::minimize(&[1.0, 2.0]).subject_to_eq(&Matrix::zeros(2, 2), &[1.0]);
}

#[test]
#[should_panic(expected = "there is not one upper bound for each variable")]
fn bounds_of_the_wrong_length_panic() {
    let _ = LinearProgram::minimize(&[1.0, 2.0]).bounds(&[0.0, 0.0], &[1.0]);
}

// ---------------------------------------------------------------------------------------------
// Single precision
// ---------------------------------------------------------------------------------------------

#[test]
fn single_precision_solves_the_textbook_examples() {
    let examples = [
        (
            Data::maximize(
                &[3.0, 5.0],
                &[&[1.0, 0.0], &[0.0, 2.0], &[3.0, 2.0]],
                &[4.0, 12.0, 18.0],
            ),
            36.0,
        ),
        (
            Data::maximize(
                &[5.0, 4.0, 3.0],
                &[&[2.0, 3.0, 1.0], &[4.0, 1.0, 2.0], &[3.0, 4.0, 2.0]],
                &[5.0, 11.0, 8.0],
            ),
            13.0,
        ),
        (
            Data::minimize(&[4.0, 1.0], &[&[-4.0, -3.0], &[1.0, 2.0]], &[-6.0, 4.0])
                .with_eq(&[&[3.0, 1.0]], &[3.0]),
            3.4,
        ),
        (
            Data::maximize(
                &[-2.0, -1.0],
                &[&[-1.0, 1.0], &[-1.0, -2.0], &[0.0, 1.0]],
                &[-1.0, -2.0, 1.0],
            ),
            -3.0,
        ),
        (chvatal_cycling(), 1.0),
        (klee_minty(6, 2.0, 5.0), 3125.0),
    ];
    for (p, optimum) in examples {
        let r = p.program_f32().solve().expect("a verified answer");
        assert_eq!(r.status, LpStatus::Optimal);
        assert!(
            (f64::from(r.objective) - optimum).abs() <= 1e-5 * optimum.abs().max(1.0),
            "{} against {optimum}",
            r.objective
        );
        assert_single_certified(&p, &r, 1e-5);
    }
}

#[test]
fn single_precision_agrees_with_highs_on_the_small_random_programmes() {
    // The programmes of up to ten variables whose rows and columns are of one size. Measured: the
    // optima within 1.3e-6 of HiGHS's; the residuals of the certificate below 1e-7, but for the
    // duality gap against the dual objective with the bounds, 1.3e-5: reduced costs within the
    // tolerance of 1e-4, priced at the far bound they point to.
    let mut checked = 0;
    let mut worst = Residuals::default();
    let mut worst_objective = 0.0f64;
    for c in cases().into_iter().filter(|c| c.n <= 10) {
        let case = generate(c.seed);
        if case.wide {
            continue;
        }
        let p = case.data();
        let r = p
            .program_f32()
            .solve()
            .unwrap_or_else(|| panic!("seed {}: no verified answer", c.seed));
        assert_eq!(r.status, c.status, "seed {}", c.seed);
        if r.status == LpStatus::Optimal {
            let err = objective_error(f64::from(r.objective), c.objective);
            assert!(
                err <= 1e-5,
                "seed {}: {} against {}",
                c.seed,
                r.objective,
                c.objective
            );
            worst_objective = worst_objective.max(err);
            worst = worst.max(assert_single_certified(&p, &r, 1e-4));
        }
        checked += 1;
    }
    assert!(cfg!(miri) || checked >= 250, "{checked}");
    assert!(
        worst_objective <= 1e-5 && worst.worst() <= 1e-4,
        "{worst_objective:e} {worst:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Ill-conditioned programmes: large multipliers, reduced costs and certificates
// ---------------------------------------------------------------------------------------------
// The exact optima below are those of the floats as given, by the rational simplex method of
// data/gen_lp_ill.py (`exact_optimum`); data/gen_lp_examples.py prints them, with HiGHS's.

/// Asserts that `r` is optimal with the objective `exact` to `tol`, relative to the larger of one
/// and `|exact|`.
#[track_caller]
fn assert_optimum(r: &LpResult<f64>, exact: f64, tol: f64) {
    assert_eq!(r.status, LpStatus::Optimal);
    assert!(
        objective_error(r.objective, exact) <= tol,
        "{} against the exact {exact}",
        r.objective
    );
}

/// A programme whose two equations `x₀ + x₁ = 1` and `x₀ + (1 + d)·x₁ = 1 + d/2` fix
/// `x₀ = x₁ = 1/2` with multipliers of about `1/d`, and whose third variable, of cost `cost` and
/// bounds `[−1, ∞)`, meets only `rows` inequalities `x₂ ≤ 5 + i` with multipliers of zero.
/// Minimise `−x₁ + cost·x₂`.
fn unrelated_multipliers(d: f64, rows: usize, cost: f64) -> Data {
    let inf = f64::INFINITY;
    let a_le: Vec<&[f64]> = vec![&[0.0, 0.0, 1.0]; rows];
    let b_le: Vec<f64> = (0..rows).map(|i| 5.0 + i as f64).collect();
    Data::minimize(&[0.0, -1.0, cost], &a_le, &b_le)
        .with_eq(
            &[&[1.0, 1.0, 0.0], &[1.0, 1.0 + d, 0.0]],
            &[1.0, 1.0 + d / 2.0],
        )
        .with_bounds(&[0.0, 0.0, -1.0], &[10.0, 10.0, inf])
}

#[test]
fn a_reduced_cost_is_not_taken_for_zero_because_unrelated_multipliers_are_large() {
    // The reduced cost of x₂ is its cost, and it moves x₂ to −1. A tolerance on reduced costs
    // relative to the largest multiplier, 1e-9·40·1e4 here, once took 3e-4 for zero and stopped
    // at −0.500000000001 with x₂ = 0, under every option. HiGHS agrees with the exact optimum on
    // the first; on the second, at its tolerance of 1e-7, it takes the two equations for one and
    // reports −1.005.
    for (d, rows, cost, exact) in [
        (1e-4, 40, 3e-4, -0.5003000000011102),
        (1e-7, 1, 0.005, -0.504999998889777),
    ] {
        let p = unrelated_multipliers(d, rows, cost);
        for opts in [
            LpOptions::new(),
            LpOptions::new().with_stall_limit(0),
            LpOptions::new().with_scaling(false),
            LpOptions::new().with_tol(1e-7),
        ] {
            let r = p.solve_with(&opts);
            if opts.tol >= d {
                // A tolerance as large as d no longer tells the two equations apart, and the
                // answer is optimal to it (HiGHS's −1.005): below the exact optimum, never above.
                assert_eq!(r.status, LpStatus::Optimal);
                assert!(r.objective <= exact, "{}", r.objective);
                continue;
            }
            assert_optimum(&r, exact, 1e-12);
            assert_eq!(r.x[2], -1.0, "d = {d}");
            assert_eq!(r.reduced_costs[2], cost);
            // A dual objective that prices each reduced cost at the bound its sign points to
            // proves the optimum; one that priced x₂'s at the point (x₂ = 0) did not see it.
            assert_certified(&p, &r, 1e-12);
        }
    }
}

/// ill6-2767 of the first review's probes: data of size one and two equations that differ by
/// 1e-6 in one coefficient, multipliers of about 1e6 in phase one. The witness is a point that
/// is feasible in exact arithmetic (checked there), the exact optimum 19.604783528523146.
fn large_phase_one_multipliers() -> (Data, [f64; 6]) {
    let inf = f64::INFINITY;
    let p = Data::minimize(
        &[
            0.033508399431287784,
            4.289652038062165,
            -0.768514392365316,
            -1.819291862865434,
            1.493464830699932,
            2.7316974948303416,
        ],
        &[
            &[
                0.4255195714642922,
                0.048212952533530785,
                0.0,
                -0.6172831581747933,
                1.4325383293044864,
                0.0,
            ],
            &[0.9972340243526487, 0.0, 0.0, 0.0, 0.0, -1.5216180946154498],
        ],
        &[-1.1029048003867024, -10.898904359367187],
    )
    .with_eq(
        &[
            &[
                0.0,
                0.538456371255046,
                -0.027411779201491782,
                0.0,
                -0.7179867108968127,
                0.0,
            ],
            &[
                0.0,
                0.761105535061727,
                0.0,
                -0.9667587287579885,
                0.7262146292722984,
                0.0,
            ],
            &[
                -0.6506197927973945,
                2.147181031501968,
                0.0,
                -0.34382307390282374,
                0.6955285281444558,
                0.5887978417492359,
            ],
            &[0.0, 1.5069348579334945, 0.0, 0.3433577578115057, 0.0, 0.0],
            &[0.0, 1.5069358579334944, 0.0, 0.3433577578115057, 0.0, 0.0],
        ],
        &[
            -0.0015406304548472582,
            1.9082869414388917,
            6.967941612756002,
            -0.5728560274815098,
            -0.5728560274815098,
        ],
    )
    .with_bounds(
        &[
            -6.728342167855769,
            0.0,
            -inf,
            -5.423615749675798,
            0.0,
            1.7143356018294535,
        ],
        &[-6.381355342606776, inf, -8.577349726291384, inf, inf, inf],
    );
    let witness = [
        -6.728342167855769,
        0.0,
        -10.596376405082244,
        -1.6683940130922936,
        0.4067013447116926,
        2.94471914968395,
    ];
    (p, witness)
}

#[test]
fn a_feasible_programme_with_large_phase_one_multipliers_is_not_called_infeasible() {
    let (p, witness) = large_phase_one_multipliers();
    let exact = 19.604783528523146;
    // The witness is feasible: inside its bounds, and in the rows to rounding.
    let witnessed = LpResult {
        status: LpStatus::Optimal,
        x: witness.to_vec(),
        objective: 0.0,
        slack: Vec::new(),
        dual_le: Vec::new(),
        dual_eq: Vec::new(),
        reduced_costs: Vec::new(),
        pivots: 0,
    };
    assert!(infeasibility(&p, &witnessed) <= 1e-16);
    // Phase one once stopped early, under a tolerance on reduced costs relative to the largest
    // multiplier, and its Farkas certificate dropped coefficients that were not negligible: at
    // the tolerance 1e-7 the programme was called infeasible.
    for tol in [LpOptions::<f64>::new().tol, 1e-8, 1e-7] {
        let r = p.solve_with(&LpOptions::new().with_tol(tol));
        assert_optimum(&r, exact, 1e-12);
        assert_certified(&p, &r, 1e-9);
    }
    // At 1e-6 the two equations that differ by 1e-6 hardly differ, and a point that meets them to
    // the tolerance does better than the exact optimum; HiGHS, given the same tolerances, reports
    // the same optimum. What the dual values prove is a bound below every feasible objective,
    // so the answer is never worse than the exact optimum.
    let r = p.solve_with(&LpOptions::new().with_tol(1e-6));
    assert_eq!(r.status, LpStatus::Optimal);
    assert!(r.objective <= exact, "{}", r.objective);
    assert!(objective_error(r.objective, 18.12633462381662) <= 1e-9);
}

/// SplitMix64 to a float in `[0, 1)`: the 53 high bits of the next integer.
fn unit(r: &mut Rng) -> f64 {
    (r.next() >> 11) as f64 / (1u64 << 53) as f64
}

/// The equations `p·x₀ = 0`, `q·x₀ + r·x₁ = s` and `(q + t)·x₀ + r·x₁ = s`, consistent and of
/// rank two, with `x₀ ≥ 0` and `x₁ ≥ lo₁`, and the cost `c₀` on `x₀` only: the one feasible point
/// is `(0, s/r)` and the optimum is zero.
fn rank_deficient_zero_optimum(p: f64, q: f64, r: f64, t: f64, s: f64, lo1: f64, c0: f64) -> Data {
    Data::minimize(&[c0, 0.0], &[], &[])
        .with_eq(&[&[p, 0.0], &[q, r], &[q + t, r]], &[0.0, s, s])
        .with_bounds(&[0.0, lo1], &[f64::INFINITY, f64::INFINITY])
}

#[test]
fn a_consistent_rank_deficient_system_with_a_zero_optimum_is_solved() {
    // The basic x₀ comes out of the solve at about 5e-32, so every term of the duality gap is
    // rounding: a gap test relative to those terms alone once failed on 302 of 2000 such
    // programmes (769 unscaled, 409 under Bland's rule), and their answers were None. The test
    // now allows what the rows' residuals, held to the tolerance, are worth at the multipliers.
    let options = [
        LpOptions::new(),
        LpOptions::new().with_scaling(false),
        LpOptions::new().with_stall_limit(0),
        LpOptions::new().with_tol(1e-6),
    ];
    let p = rank_deficient_zero_optimum(
        0.8898297483912715,
        1.0335765890563122,
        0.9507606806068835,
        0.1082930293028078,
        3.192013253322282,
        1.6786628425172025,
        0.29943152228274333,
    );
    for opts in &options {
        let r = p.solve_with(opts);
        assert_optimum(&r, 0.0, 1e-15);
        assert!(r.x[0] <= 1e-15 && (r.x[1] - 3.357325685034405).abs() <= 1e-15);
        assert_certified(&p, &r, 1e-15);
    }
    // The programmes of that kind the review drew (its rankdef2), every one of them.
    let count = if cfg!(miri) { 9 } else { 2000 };
    let mut rng = Rng(7);
    for case in 0..count {
        let p = 0.5 + unit(&mut rng);
        let q = 1.0 + 2.0 * unit(&mut rng);
        let r = 0.05 + unit(&mut rng);
        let t = [1e-1, 1e-2, 1e-4][case % 3] * (0.5 + unit(&mut rng));
        let x1 = 2.0 + 3.0 * unit(&mut rng);
        let s = r * x1;
        let lo1 = 0.5 * x1;
        let c0 = 0.05 + unit(&mut rng);
        let data = rank_deficient_zero_optimum(p, q, r, t, s, lo1, c0);
        for opts in &options[..3] {
            let answer = data
                .program()
                .solve_with(opts)
                .unwrap_or_else(|| panic!("case {case}: no verified answer"));
            assert_optimum(&answer, 0.0, 1e-15);
        }
    }
}

#[test]
fn rows_in_tiny_or_huge_units_are_scaled_to_moderate_ones() {
    // Wyndor's problem with every row multiplied by 10^k: the same programme, the same answer.
    // The scale factors reach over the whole range of the floating-point numbers; capped at
    // 2^±100 they left rows of 1e-70 at 1e-40, where every row looked satisfied and the
    // programme unbounded.
    for k in (-300..=300).step_by(10) {
        let s = 10f64.powi(k);
        let p = Data::maximize(
            &[3.0, 5.0],
            &[&[s, 0.0], &[0.0, 2.0 * s], &[3.0 * s, 2.0 * s]],
            &[4.0 * s, 12.0 * s, 18.0 * s],
        );
        let r = p.solve();
        assert_close(&r.x, &[2.0, 6.0], 1e-14);
        assert_close(&[r.objective], &[36.0], 1e-14);
        assert_certified(&p, &r, 1e-14);
    }
    for k in (-30..=30).step_by(5) {
        let s = 10f64.powi(k);
        let p = Data::maximize(
            &[3.0, 5.0],
            &[&[s, 0.0], &[0.0, 2.0 * s], &[3.0 * s, 2.0 * s]],
            &[4.0 * s, 12.0 * s, 18.0 * s],
        );
        let r = p.program_f32().solve().expect("a verified answer");
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((f64::from(r.objective) - 36.0).abs() <= 36.0 * 1e-5, "{k}");
    }
}

/// A programme of data/gen_lp_ill.py: feasible and bounded, with two equations that differ by
/// `delta` in the coefficient of `x_k`.
struct Ill {
    delta: f64,
    data: Data,
    /// The bounds as integers, `None` for an infinite one, for the checksum.
    lower: Vec<Option<i64>>,
    upper: Vec<Option<i64>>,
}

/// The programme of a seed: `generate` of `data/gen_lp_ill.py`, draw for draw. Every number of it
/// is a dyadic rational of a few bits, so every operation below is exact.
fn generate_ill(seed: u64) -> Ill {
    let mut r = Rng(seed);
    let n = r.between(3, 8) as usize;
    let m_le = r.between(0, 5) as usize;
    let m_eq = r.between(0, 2) as usize;
    let delta = 0.5f64.powi([14, 20, 26][r.below(3) as usize]);
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    let mut x0 = Vec::with_capacity(n);
    for _ in 0..n {
        let (lo, hi) = match r.below(5) {
            0 => (Some(0), None),
            1 => (None, None),
            2 => (Some(r.between(-8, 8)), None),
            3 => (None, Some(r.between(-8, 8))),
            _ => {
                let lo = r.between(-8, 8);
                (Some(lo), Some(lo + r.between(1, 10)))
            }
        };
        let x = match (lo, hi) {
            (Some(lo), Some(hi)) => 2 * lo + r.below((2 * (hi - lo) + 1) as u64) as i64,
            (Some(lo), None) => 2 * lo + r.below(13) as i64,
            (None, Some(hi)) => 2 * hi - r.below(13) as i64,
            (None, None) => r.between(-12, 12),
        };
        lower.push(lo);
        upper.push(hi);
        x0.push(x as f64 / 2.0);
    }
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(m_le + m_eq + 2);
    for _ in 0..m_le + m_eq {
        let row = (0..n)
            .map(|_| {
                if r.below(100) < 60 {
                    r.between(-16, 16) as f64 / 8.0
                } else {
                    0.0
                }
            })
            .collect();
        rows.push(row);
    }
    let k = r.below(n as u64) as usize;
    let mut e1 = Vec::with_capacity(n);
    for j in 0..n {
        if j == k {
            let v = r.between(1, 16) * if r.below(2) == 0 { 1 } else { -1 };
            e1.push(v as f64 / 8.0);
        } else if r.below(2) == 0 {
            e1.push(r.between(-16, 16) as f64 / 8.0);
        } else {
            e1.push(0.0);
        }
    }
    let mut e2 = e1.clone();
    e2[k] += delta;
    let at = |row: &[f64]| row.iter().zip(&x0).map(|(a, x)| a * x).sum::<f64>();
    let mut b_le = Vec::with_capacity(m_le);
    for row in rows.iter().take(m_le) {
        let slack = if r.below(3) == 0 { 0 } else { r.between(1, 8) };
        b_le.push(at(row) + slack as f64 / 2.0);
    }
    let mut a_eq: Vec<Vec<f64>> = rows.split_off(m_le);
    a_eq.push(e1);
    a_eq.push(e2);
    let b_eq: Vec<f64> = a_eq.iter().map(|row| at(row)).collect();
    let a_le = rows;
    // The costs: a feasible dual, then a pull on x_k that only the pair can balance.
    let mut y: Vec<f64> = (0..m_le)
        .map(|_| {
            if r.below(3) != 0 {
                -(r.between(0, 8) as f64) / 4.0
            } else {
                0.0
            }
        })
        .collect();
    for _ in 0..a_eq.len() {
        y.push(r.between(-8, 8) as f64 / 4.0);
    }
    let every: Vec<&Vec<f64>> = a_le.iter().chain(&a_eq).collect();
    let mut c = Vec::with_capacity(n);
    for j in 0..n {
        let d = match (lower[j], upper[j]) {
            (None, None) => 0,
            (_, None) => r.between(0, 8),
            (None, _) => -r.between(0, 8),
            _ => r.between(-8, 8),
        };
        let priced: f64 = every.iter().zip(&y).map(|(row, y)| y * row[j]).sum();
        c.push(d as f64 / 4.0 + priced);
    }
    c[k] += (r.between(1, 16) * if r.below(2) == 0 { 1 } else { -1 }) as f64 / 8.0;
    let data = Data {
        maximize: false,
        c,
        a_le,
        b_le,
        a_eq,
        b_eq,
        lower: lower
            .iter()
            .map(|b| b.map_or(f64::NEG_INFINITY, |v| v as f64))
            .collect(),
        upper: upper
            .iter()
            .map(|b| b.map_or(f64::INFINITY, |v| v as f64))
            .collect(),
    };
    Ill {
        delta,
        data,
        lower,
        upper,
    }
}

impl Ill {
    /// The word-wise FNV-1a hash of the bits of the data that `data/gen_lp_ill.py` stores.
    fn checksum(&self) -> u64 {
        const INF_CODE: i64 = 1 << 40;
        let p = &self.data;
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x0100_0000_01B3);
        mix(p.c.len() as u64);
        mix(p.b_le.len() as u64);
        mix(p.b_eq.len() as u64);
        p.c.iter().for_each(|v| mix(v.to_bits()));
        p.a_le.iter().flatten().for_each(|v| mix(v.to_bits()));
        p.b_le.iter().for_each(|v| mix(v.to_bits()));
        p.a_eq.iter().flatten().for_each(|v| mix(v.to_bits()));
        p.b_eq.iter().for_each(|v| mix(v.to_bits()));
        for (lo, hi) in self.lower.iter().zip(&self.upper) {
            mix(lo.unwrap_or(-INF_CODE) as u64);
            mix(hi.unwrap_or(INF_CODE) as u64);
        }
        h
    }
}

#[test]
fn ill_conditioned_programmes_have_their_exact_optima() {
    // Three hundred programmes of data/gen_lp_ill.py, feasible and bounded, each with a pair of
    // equations that differ by delta = 2⁻¹⁴, 2⁻²⁰ or 2⁻²⁶ in one coefficient: bases of a
    // condition of about 1/delta, multipliers as large. Their exact optima are in
    // data/lp_ill_exact.txt. Measured, under the default options, Bland's rule and without the
    // scaling: within 5.0e-11 of the exact optimum at delta 2⁻¹⁴ and within 4.5e-10 at 2⁻²⁰ (where
    // HiGHS's dual simplex is up to 1.2e-2 below it and its interior point method gets the status
    // of 12 wrong). At 2⁻²⁶ the rows' tolerance, 1e-9 of terms of size ten, no longer tells the two
    // equations apart: the answers there are optimal to that tolerance, as much as 0.6 below the
    // exact optimum (HiGHS: 1.5), at most 9.9e-8 above it, and 4 of those 276 are None. With the
    // tolerance 1e-7, which does not tell them apart at 2⁻²⁰ either, every answer is optimal to it
    // and none is more than 5.7e-9 above the exact optimum (1 None).
    let references: Vec<(u64, u64, f64)> = include_str!("data/lp_ill_exact.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            assert_eq!((w.len(), w[5]), (7, "O"), "{l}");
            let exact = f64::from_bits(u64::from_str_radix(w[6], 16).unwrap());
            (
                w[0].parse().unwrap(),
                u64::from_str_radix(w[4], 16).unwrap(),
                exact,
            )
        })
        .collect();
    assert_eq!(references.len(), 300);
    let take = if cfg!(miri) { 6 } else { references.len() };
    let options = [
        LpOptions::new(),
        LpOptions::new().with_stall_limit(0),
        LpOptions::new().with_scaling(false),
        LpOptions::new().with_tol(1e-7),
    ];
    let mut unanswered = 0;
    for &(seed, checksum, exact) in references.iter().take(take) {
        let ill = generate_ill(seed);
        assert_eq!(ill.checksum(), checksum, "seed {seed}");
        let tellable = ill.delta >= 0.5f64.powi(20);
        for (k, opts) in options.iter().enumerate() {
            let Some(r) = ill.data.program().solve_with(opts) else {
                // Only where the tolerance cannot tell the equations apart.
                assert!(!tellable, "seed {seed}, options {k}: no verified answer");
                unanswered += 1;
                continue;
            };
            assert_eq!(r.status, LpStatus::Optimal, "seed {seed}, options {k}");
            consistent(&ill.data, &r);
            let above = (r.objective - exact) / exact.abs().max(1.0);
            if k < 3 && tellable {
                assert!(
                    above.abs() <= 1e-9,
                    "seed {seed}, options {k}: {} against the exact {exact}",
                    r.objective
                );
                assert_certified(&ill.data, &r, 1e-9);
            } else {
                // The dual values prove a bound below every feasible objective: above it the
                // answer is only by the rounding of reduced costs made of terms of 1/delta.
                let limit = if k == 3 { 1e-8 } else { 1e-6 };
                assert!(
                    above <= limit,
                    "seed {seed}, options {k}: {} against the exact {exact}",
                    r.objective
                );
            }
        }
    }
    assert!(unanswered <= 8, "{unanswered}");
}

// ---------------------------------------------------------------------------------------------
// One large number beside rows of moderate size
// ---------------------------------------------------------------------------------------------
// A variable is held to its bounds within the tolerance, or within the error its own value
// carries when that is larger, and the basis gives each basic variable that error. A floor of 64
// units in the last place of the largest number anywhere in the programme once let one large
// right-hand side, slack or value loosen every other row and bound: an equation broken by 1e-5
// beside a slack of 1e9 was an optimum, and a programme infeasible by 1e-5 had one. The exact
// optima below are those of the floats as given; data/gen_lp_examples.py prints them (section
// "Review round 2"), with HiGHS's.

/// How a large number is put beside a programme, on a variable `x_new` of its own that costs
/// nothing: as the slack of a row `x_new ≤ big`, as the lower bound of `x_new ≥ big`, or as the
/// value of a basic `x_new` that an equation `x_new = big` fixes. None of the three changes the
/// status or the optimum of the programme.
#[derive(Debug, Clone, Copy)]
enum Large {
    Slack,
    Bound,
    Basic,
}

const LARGE: [Large; 3] = [Large::Slack, Large::Bound, Large::Basic];

impl Data {
    /// The programme with `big` beside it, as `large` says.
    fn beside(&self, big: f64, large: Large) -> Data {
        let mut p = self.clone();
        let n = p.c.len();
        p.c.push(0.0);
        for row in p.a_le.iter_mut().chain(&mut p.a_eq) {
            row.push(0.0);
        }
        p.lower.push(0.0);
        p.upper.push(f64::INFINITY);
        let mut row = vec![0.0; n + 1];
        row[n] = 1.0;
        match large {
            Large::Slack => {
                p.a_le.push(row);
                p.b_le.push(big);
            }
            Large::Bound => p.lower[n] = big,
            Large::Basic => {
                p.a_eq.push(row);
                p.b_eq.push(big);
            }
        }
        p
    }

    /// The programme with every number rounded to `f32`, so that [`Data::program_f32`] states it
    /// exactly and an answer in `f32` is checked against the data it was computed from.
    fn rounded_to_f32(&self) -> Data {
        let round = |v: &mut f64| *v = f64::from(*v as f32);
        let mut p = self.clone();
        p.c.iter_mut().for_each(round);
        p.a_le.iter_mut().flatten().for_each(round);
        p.b_le.iter_mut().for_each(round);
        p.a_eq.iter_mut().flatten().for_each(round);
        p.b_eq.iter_mut().for_each(round);
        p.lower.iter_mut().for_each(round);
        p.upper.iter_mut().for_each(round);
        p
    }

    /// The programme with an inequality that contradicts the sum of its inequalities by
    /// `margin` times the size of that sum, `Σ_i |b_i| + Σ_j |Σ_i a_ij|`: infeasible by that much
    /// whatever else it is. The data are integers, so the sum is exact.
    fn contradicted(&self, margin: f64) -> Data {
        let mut p = self.clone();
        let n = p.c.len();
        let mut row = vec![0.0; n];
        let mut rhs = 0.0;
        for (a, b) in self.a_le.iter().zip(&self.b_le) {
            for (s, v) in row.iter_mut().zip(a) {
                *s -= v;
            }
            rhs -= b;
        }
        let size = self.b_le.iter().map(|b| b.abs()).sum::<f64>() + norm(&row);
        p.a_le.push(row);
        p.b_le.push(rhs - margin * size.max(1.0));
        p
    }
}

/// Minimise `1000·x₂` subject to `x₀ + x₁ + x₂ = 1`, `x₀ ≤ 0.4` and `x₁ ≤ 0.6 − g`: the optimum
/// is `1000·g`, at `x₂ = g`, which only the equation forces.
fn forced_by_one_equation(g: f64) -> Data {
    Data::minimize(&[0.0, 0.0, 1000.0], &[], &[])
        .with_eq(&[&[1.0, 1.0, 1.0]], &[1.0])
        .with_bounds(&[0.0; 3], &[0.4, 0.6 - g, f64::INFINITY])
}

/// `x₀ + x₁ = 1` with `x₀ ≤ 0.4` and `x₁ ≤ 0.6 − gap`: infeasible by `gap`.
fn short_of_one_equation(gap: f64) -> Data {
    Data::minimize(&[1.0, 1.0], &[], &[])
        .with_eq(&[&[1.0, 1.0]], &[1.0])
        .with_bounds(&[0.0; 2], &[0.4, 0.6 - gap])
}

#[test]
fn one_large_number_does_not_loosen_the_other_rows() {
    let options = [
        LpOptions::new(),
        LpOptions::new().with_stall_limit(0),
        LpOptions::new().with_scaling(false),
    ];
    // The exact optima of the floats: 1000·(1 − 0.4 − fl(0.6 − g)).
    for (g, exact) in [(1e-5, 0.00999999999995449), (1e-8, 1.0000000050247593e-5)] {
        let alone = forced_by_one_equation(g);
        for big in [1e6, 1e9, 1e12] {
            for large in LARGE {
                let p = alone.beside(big, large);
                for (k, opts) in options.iter().enumerate() {
                    let r = p.solve_with(opts);
                    assert_eq!(r.status, LpStatus::Optimal, "{g:e} {big:e} {large:?} {k}");
                    assert!(
                        (r.objective - exact).abs() <= 1e-9 * exact,
                        "{g:e} {big:e} {large:?} {k}: {} against the exact {exact}",
                        r.objective
                    );
                    assert!((r.x[0] + r.x[1] + r.x[2] - 1.0).abs() <= 1e-15);
                    assert_certified(&p, &r, 1e-9);
                }
            }
        }
    }
    for gap in [1e-5, 1e-7] {
        let alone = short_of_one_equation(gap);
        for big in [1e6, 1e9, 1e12] {
            for large in LARGE {
                let p = alone.beside(big, large);
                for (k, opts) in options.iter().enumerate() {
                    let r = p.solve_with(opts);
                    assert_eq!(
                        r.status,
                        LpStatus::Infeasible,
                        "{gap:e} {big:e} {large:?} {k}"
                    );
                }
            }
        }
    }
    // Without the scaling, a row in numbers of 1e200 beside one in ones: the variables of the
    // basis are each known to rounding, and the second equation, once broken by 1 at (1, 0), holds.
    let p = Data::minimize(&[1.0, 1.0], &[], &[])
        .with_eq(&[&[1e200, 1e-200], &[1.0, 1.0]], &[1e200, 2.0])
        .with_bounds(&[0.0; 2], &[f64::INFINITY; 2]);
    for opts in &options {
        let r = p.solve_with(opts);
        assert_optimum(&r, 2.0, 1e-15);
        assert_close(&r.x, &[1.0, 1.0], 1e-15);
    }
}

#[test]
fn one_large_number_does_not_loosen_the_other_rows_in_single_precision() {
    // The exact optima of the data in f32: 1000·(1 − fl₃₂(0.4) − fl₃₂(0.6 − g)). With a floor
    // of 64 units in the last place of 1e5, an equation broken by 0.6 once held.
    for (g, exact) in [(1e-2, 10.000020265579224), (1e-3, 1.0000169277191162)] {
        let alone = forced_by_one_equation(g).rounded_to_f32();
        for big in [1e2, 1e4, 1e5] {
            for large in LARGE {
                let p = alone.beside(big, large);
                let r = p.program_f32().solve().expect("a verified answer");
                assert_eq!(r.status, LpStatus::Optimal, "{g:e} {big:e} {large:?}");
                let objective = f64::from(r.objective);
                assert!(
                    (objective - exact).abs() <= 1e-4 * exact,
                    "{g:e} {big:e} {large:?}: {objective} against the exact {exact}"
                );
                assert_single_certified(&p, &r, 1e-4);
            }
        }
    }
    for gap in [1e-2, 1e-3] {
        let alone = short_of_one_equation(gap);
        for big in [1e2, 1e4, 1e5] {
            for large in LARGE {
                let r = alone.beside(big, large).program_f32().solve();
                assert_eq!(
                    r.map(|r| r.status),
                    Some(LpStatus::Infeasible),
                    "{gap:e} {big:e} {large:?}"
                );
            }
        }
    }
}

#[test]
fn random_programmes_keep_their_answers_beside_one_large_number() {
    // The six hundred programmes with a slack, a bound or a basic variable of 1e9 or 1e12 beside
    // them: HiGHS's status and optimum, certified as before.
    let refs = cases();
    for big in [1e9, 1e12] {
        for large in LARGE {
            for c in &refs {
                let p = generate(c.seed).data().beside(big, large);
                let r = p
                    .program()
                    .solve()
                    .unwrap_or_else(|| panic!("seed {} {big:e} {large:?}: no answer", c.seed));
                assert_eq!(r.status, c.status, "seed {} {big:e} {large:?}", c.seed);
                if r.status == LpStatus::Optimal {
                    let err = objective_error(r.objective, c.objective);
                    assert!(err <= 1e-9, "seed {} {big:e} {large:?}: {err:e}", c.seed);
                    assert_certified(&p, &r, 1e-9);
                }
            }
        }
    }
}

#[test]
fn programmes_infeasible_by_a_millionth_stay_infeasible_beside_one_large_number() {
    // Each of the random programmes with an inequality, contradicted by 1e-6 of the size of the
    // sum of its inequalities, alone and beside a large number; and in f32 the small ones,
    // contradicted by 1e-2, beside 1e4.
    let refs: Vec<Reference> = cases()
        .into_iter()
        .filter(|c| c.m_le > 0 && (!cfg!(miri) || c.n <= 4))
        .collect();
    assert!(cfg!(miri) || refs.len() >= 400, "{}", refs.len());
    for c in &refs {
        let p = generate(c.seed).data().contradicted(1e-6);
        let status = |p: &Data| p.program().solve().map(|r| r.status);
        assert_eq!(status(&p), Some(LpStatus::Infeasible), "seed {}", c.seed);
        for big in [1e9, 1e12] {
            for large in LARGE {
                let q = p.beside(big, large);
                assert_eq!(
                    status(&q),
                    Some(LpStatus::Infeasible),
                    "seed {} {big:e} {large:?}",
                    c.seed
                );
            }
        }
    }
    let mut single = 0;
    for c in refs.iter().filter(|c| c.n <= 10) {
        let case = generate(c.seed);
        if case.wide {
            continue;
        }
        let p = case.data().contradicted(1e-2);
        for large in LARGE {
            let r = p.beside(1e4, large).program_f32().solve();
            assert_eq!(
                r.map(|r| r.status),
                Some(LpStatus::Infeasible),
                "seed {} {large:?}",
                c.seed
            );
        }
        single += 1;
    }
    assert!(cfg!(miri) || single >= 150, "{single}");
}

#[test]
fn a_ray_whose_slope_is_within_the_tolerance_is_not_seen() {
    // Minimise (1 − δ)·x₀ − x₁ subject to x₁ − x₀ ≤ 0, x ≥ 0: along x₀ = x₁ = t the objective
    // is −δ·t, so the programme is unbounded for every δ > 0. At the origin, with x₁ basic in
    // the row, the reduced cost of x₀ is −δ against terms of about 2, and a reduced cost within
    // the tolerance of its terms is taken for zero even when the bound it points to is infinite:
    // the documented limit. Beyond the tolerance the ray is found.
    let p = |delta: f64| Data::minimize(&[1.0 - delta, -1.0], &[&[-1.0, 1.0]], &[0.0]);
    let r = p(1e-10).solve();
    assert_eq!(r.status, LpStatus::Optimal);
    assert_eq!((r.x.as_slice(), r.objective), (&[0.0, 0.0][..], 0.0));
    let tight = LpOptions::new().with_tol(1e-12);
    assert_eq!(p(1e-10).solve_with(&tight).status, LpStatus::Unbounded);
    assert_eq!(p(1e-6).solve().status, LpStatus::Unbounded);
}

// ---------------------------------------------------------------------------------------------
// The dual simplex method
// ---------------------------------------------------------------------------------------------
// `LpOptions::dual_threshold(0)` has the dual simplex method solve every programme, the primal
// method certifying the basis it ends at and finishing from it. The answers have to be the
// primal method's: the same statuses, the optima of HiGHS, certified as before. The pivot counts
// show that the dual method, and not the primal method it falls back on, found them.

/// The options with the dual simplex method for every programme.
fn dual() -> LpOptions<f64> {
    LpOptions::new().with_dual_threshold(0)
}

/// The options with the primal simplex method for every programme.
fn primal() -> LpOptions<f64> {
    LpOptions::new().with_dual_threshold(usize::MAX)
}

#[test]
fn the_dual_method_agrees_with_highs_on_the_six_hundred_programmes() {
    let (worst, objective, _) = agree_with_highs(&cases(), &dual(), 1.0, 1.0, 1e-9);
    // Measured: the optima within 5.0e-13 of HiGHS's, as the primal method's.
    assert!(objective <= 1e-11, "{objective:e}");
    assert!(worst.primal <= 1e-14 && worst.gap <= 1e-14, "{worst:?}");
    assert!(worst.dual <= 1e-11 && worst.slackness <= 1e-11, "{worst:?}");
    let (_, objective, _) = agree_with_highs(&cases(), &dual().with_scaling(false), 1.0, 1.0, 1e-9);
    assert!(objective <= 1e-11, "{objective:e}");
}

#[test]
fn the_dual_method_solves_programmes_in_small_or_large_numbers_as_in_moderate_ones() {
    let refs: Vec<Reference> = cases().into_iter().step_by(3).collect();
    for (scale_primal, scale_dual) in [
        (1e-9, 1.0),
        (1e9, 1.0),
        (1.0, 1e-8),
        (1.0, 1e8),
        (1e-3, 1e5),
    ] {
        let (_, objective, _) = agree_with_highs(&refs, &dual(), scale_primal, scale_dual, 1e-9);
        assert!(
            objective <= 1e-11,
            "{scale_primal:e} {scale_dual:e}: {objective:e}"
        );
    }
}

#[test]
fn the_dual_method_keeps_the_answers_beside_one_large_number() {
    // As `random_programmes_keep_their_answers_beside_one_large_number`, the dual method first.
    let refs: Vec<Reference> = cases().into_iter().step_by(2).collect();
    for big in [1e9, 1e12] {
        for large in LARGE {
            for c in &refs {
                let p = generate(c.seed).data().beside(big, large);
                let r = p
                    .program()
                    .solve_with(&dual())
                    .unwrap_or_else(|| panic!("seed {} {big:e} {large:?}: no answer", c.seed));
                assert_eq!(r.status, c.status, "seed {} {big:e} {large:?}", c.seed);
                if r.status == LpStatus::Optimal {
                    let err = objective_error(r.objective, c.objective);
                    assert!(err <= 1e-9, "seed {} {big:e} {large:?}: {err:e}", c.seed);
                    assert_certified(&p, &r, 1e-9);
                }
            }
        }
    }
}

#[test]
fn the_dual_method_keeps_contradicted_programmes_infeasible_and_large_bounds_harmless() {
    // Contradicted by a millionth, alone and beside a large number; and with every infinite
    // bound written as ±10³⁰, which the dual method takes for infinite.
    for c in cases().into_iter().filter(|c| c.m_le > 0) {
        let p = generate(c.seed).data().contradicted(1e-6);
        let status = |p: &Data| p.program().solve_with(&dual()).map(|r| r.status);
        assert_eq!(status(&p), Some(LpStatus::Infeasible), "seed {}", c.seed);
        let q = p.beside(1e12, Large::Basic);
        assert_eq!(status(&q), Some(LpStatus::Infeasible), "seed {}", c.seed);
    }
    for c in cases()
        .into_iter()
        .filter(|c| c.status != LpStatus::Unbounded)
    {
        let mut p = generate(c.seed).data();
        for v in p.lower.iter_mut().chain(&mut p.upper) {
            if v.is_infinite() {
                *v = 1e30f64.copysign(*v);
            }
        }
        let r = p
            .program()
            .solve_with(&dual())
            .unwrap_or_else(|| panic!("seed {}: no verified answer", c.seed));
        assert_eq!(r.status, c.status, "seed {}", c.seed);
        if r.status == LpStatus::Optimal {
            assert!(
                objective_error(r.objective, c.objective) <= 1e-9,
                "seed {}",
                c.seed
            );
            assert!(largest(&r.x) < 1e15, "seed {}: {:?}", c.seed, r.x);
            assert_certified(&p, &r, 1e-9);
        }
    }
}

#[test]
fn the_dual_method_in_single_precision_agrees_with_highs_on_the_small_random_programmes() {
    let opts = LpOptions::<f32>::new().with_dual_threshold(0);
    let mut checked = 0;
    for c in cases().into_iter().filter(|c| c.n <= 10) {
        let case = generate(c.seed);
        if case.wide {
            continue;
        }
        let p = case.data();
        let r = p
            .program_f32()
            .solve_with(&opts)
            .unwrap_or_else(|| panic!("seed {}: no verified answer", c.seed));
        assert_eq!(r.status, c.status, "seed {}", c.seed);
        if r.status == LpStatus::Optimal {
            let err = objective_error(f64::from(r.objective), c.objective);
            assert!(
                err <= 1e-5,
                "seed {}: {} against {}",
                c.seed,
                r.objective,
                c.objective
            );
            // Measured: the duality gap with the bounds to 1.01e-4, reduced costs within the
            // tolerance of 1e-4 priced at the far bounds they point to; the rest below 1e-7.
            assert_single_certified(&p, &r, 2e-4);
        }
        checked += 1;
    }
    assert!(cfg!(miri) || checked >= 250, "{checked}");
}

/// A linear congruential generator of floats in `[0, 1)`: the one of the numerics scoreboard.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The random programmes of the numerics scoreboard: `m_le` inequalities and `m_eq` equations
/// through a random point with random slacks, of the given density of elements uniform in
/// `[−1, 1)`, every variable in `[0, 10]` but each tenth in `[0, ∞)` with a cost in `[0, 1)`.
fn scoreboard(m_le: usize, m_eq: usize, n: usize, density: f64, seed: u64) -> Data {
    let mut r = Lcg(seed);
    let entry = |r: &mut Lcg| {
        if r.next() < density {
            2.0 * r.next() - 1.0
        } else {
            0.0
        }
    };
    let point: Vec<f64> = (0..n).map(|_| r.next()).collect();
    let a_le: Vec<Vec<f64>> = (0..m_le)
        .map(|_| (0..n).map(|_| entry(&mut r)).collect())
        .collect();
    let a_eq: Vec<Vec<f64>> = (0..m_eq)
        .map(|_| (0..n).map(|_| entry(&mut r)).collect())
        .collect();
    let at = |row: &[f64]| row.iter().zip(&point).map(|(a, x)| a * x).sum::<f64>();
    let b_le: Vec<f64> = a_le.iter().map(|row| at(row) + r.next()).collect();
    let b_eq: Vec<f64> = a_eq.iter().map(|row| at(row)).collect();
    let mut c = vec![0.0; n];
    let mut upper = vec![10.0; n];
    for j in 0..n {
        if j % 10 == 9 {
            upper[j] = f64::INFINITY;
            c[j] = r.next();
        } else {
            c[j] = 2.0 * r.next() - 1.0;
        }
    }
    Data {
        maximize: false,
        c,
        a_le,
        b_le,
        a_eq,
        b_eq,
        lower: vec![0.0; n],
        upper,
    }
}

#[test]
fn the_dual_method_solves_the_scoreboards_programmes_in_few_pivots() {
    // The optima of HiGHS 1.12 (scipy.optimize's bindings, twelve digits). The primal method
    // takes 1391, 869, 5340 and 3282 pivots for the first four and ends the fifth at its limit
    // of 10 000; the dual method, about one and a half pivots a row.
    let programmes: &[(usize, usize, usize, f64, u64, f64)] = &[
        (100, 10, 200, 1.0, 102, -241.582_582_007),
        (100, 10, 200, 0.1, 202, -258.963_473_468),
        (200, 20, 400, 1.0, 103, -526.903_899_346),
        (200, 20, 400, 0.1, 203, -533.263_798_459),
        (300, 30, 600, 1.0, 104, -677.471_694_007),
    ];
    let count = if cfg!(miri) { 0 } else { programmes.len() };
    for &(m_le, m_eq, n, density, seed, highs) in &programmes[..count] {
        let p = scoreboard(m_le, m_eq, n, density, seed);
        let r = p.solve();
        assert_eq!(r.status, LpStatus::Optimal, "{m_le}×{n} {density}");
        let err = objective_error(r.objective, highs);
        assert!(
            err <= 1e-11,
            "{m_le}×{n} {density}: {} against {highs}",
            r.objective
        );
        assert!(
            r.pivots <= 3 * (m_le + m_eq),
            "{m_le}×{n} {density}: {} pivots",
            r.pivots
        );
        assert_certified(&p, &r, 1e-9);
    }
}

/// A transportation problem of `s` sources and `d` sinks of integer supplies and demands, every
/// supply shipped (balanced, so that one equation follows from the others) or not.
fn transportation(s: usize, d: usize, seed: u64, balanced: bool) -> Data {
    let mut r = Rng(seed);
    let n = s * d;
    let supply: Vec<f64> = (0..s).map(|_| r.between(10, 50) as f64).collect();
    let total: f64 = supply.iter().sum();
    let target = if balanced {
        total
    } else {
        (0.8 * total).floor()
    };
    let weights: Vec<f64> = (0..d).map(|_| r.between(5, 40) as f64).collect();
    let sum: f64 = weights.iter().sum();
    let mut demand: Vec<f64> = weights.iter().map(|w| (w * target / sum).floor()).collect();
    let short = target - demand.iter().sum::<f64>();
    demand[d - 1] += short;
    let c: Vec<f64> = (0..n).map(|_| r.between(1, 20) as f64).collect();
    let source = |i: usize| (0..n).map(|k| f64::from(u8::from(k / d == i))).collect();
    let sink = |j: usize| (0..n).map(|k| f64::from(u8::from(k % d == j))).collect();
    let mut p = Data::minimize(&c, &[], &[]);
    let (mut rows, mut rhs): (Vec<Vec<f64>>, Vec<f64>) =
        (0..d).map(|j| (sink(j), demand[j])).unzip();
    if balanced {
        for (i, &total) in supply.iter().enumerate() {
            rows.push(source(i));
            rhs.push(total);
        }
    } else {
        p.a_le = (0..s).map(source).collect();
        p.b_le = supply;
    }
    p.a_eq = rows;
    p.b_eq = rhs;
    p
}

/// A plan of production over `t` periods for `k` products sharing `res` resources: production
/// `p_ti ≤ 60` at a cost rising with time, stock `s_ti ≥ 0` at a holding cost, the balances
/// `s_(t−1)i + p_ti − s_ti = demand_ti` and the capacities `Σ_i use_ri·p_ti ≤ cap_r`, generous or
/// half of what the demand needs (infeasible).
fn production(t: usize, k: usize, res: usize, seed: u64, tight: bool) -> Data {
    let mut r = Rng(seed);
    let n = 2 * t * k;
    let made = |tt: usize, i: usize| tt * k + i;
    let kept = |tt: usize, i: usize| t * k + tt * k + i;
    let usage: Vec<Vec<f64>> = (0..res)
        .map(|_| {
            (0..k)
                .map(|_| {
                    if r.below(10) < 6 {
                        r.between(1, 5) as f64 / 2.0
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let demand: Vec<Vec<f64>> = (0..t)
        .map(|_| (0..k).map(|_| r.between(0, 30) as f64).collect())
        .collect();
    let mut c = vec![0.0; n];
    let mut upper = vec![f64::INFINITY; n];
    for tt in 0..t {
        for i in 0..k {
            c[made(tt, i)] = r.between(50, 100) as f64 / 10.0 + 0.3 * tt as f64;
            c[kept(tt, i)] = r.between(5, 15) as f64 / 10.0;
            upper[made(tt, i)] = 60.0;
        }
    }
    let mut p = Data::minimize(&c, &[], &[]).with_bounds(&vec![0.0; n], &upper);
    for tt in 0..t {
        for i in 0..k {
            let mut row = vec![0.0; n];
            if tt > 0 {
                row[kept(tt - 1, i)] = 1.0;
            }
            row[made(tt, i)] = 1.0;
            row[kept(tt, i)] = -1.0;
            p.a_eq.push(row);
            p.b_eq.push(demand[tt][i]);
        }
        for usage in &usage {
            let mut row = vec![0.0; n];
            let mut need = 0.0;
            for i in 0..k {
                row[made(tt, i)] = usage[i];
                need += usage[i] * demand[tt][i];
            }
            p.a_le.push(row);
            p.b_le.push(if tight {
                (0.5 * need).round()
            } else {
                (1.3 * need).round() + 10.0
            });
        }
    }
    p
}

/// A flow of least cost on a random connected network of `nodes` nodes, both directions of a
/// spanning tree and random arcs up to `arcs`, of capacities 20 to 80: one balance equation for
/// each node, one of them following from the others.
fn network_flow(nodes: usize, arcs: usize, seed: u64) -> Data {
    let mut r = Rng(seed);
    let mut list = Vec::new();
    for v in 1..nodes {
        let u = r.below(v as u64) as usize;
        list.push((u, v));
        list.push((v, u));
    }
    while list.len() < arcs {
        let (u, v) = (
            r.below(nodes as u64) as usize,
            r.below(nodes as u64) as usize,
        );
        if u != v {
            list.push((u, v));
        }
    }
    let n = list.len();
    let c: Vec<f64> = (0..n).map(|_| r.between(1, 30) as f64).collect();
    let upper: Vec<f64> = (0..n).map(|_| r.between(20, 80) as f64).collect();
    let mut supply = vec![0.0; nodes];
    for _ in 0..nodes / 3 {
        let (s, t) = (
            r.below(nodes as u64) as usize,
            r.below(nodes as u64) as usize,
        );
        let q = r.between(1, 15) as f64;
        supply[s] += q;
        supply[t] -= q;
    }
    let mut p = Data::minimize(&c, &[], &[]).with_bounds(&vec![0.0; n], &upper);
    for (v, b) in supply.iter().enumerate() {
        let row = list
            .iter()
            .map(|&(a, z)| {
                if a == v {
                    1.0
                } else if z == v {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();
        p.a_eq.push(row);
        p.b_eq.push(*b);
    }
    p
}

/// Variables in `[−1, 1]` and `[0, u]` under random rows of the given density, slack by a random
/// part of the size of their terms at the origin: the long-step ratio test moves many of them to
/// their other bounds in a step.
fn boxed(m: usize, n: usize, density: f64, seed: u64) -> Data {
    let mut r = Lcg(seed);
    let mut lower = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut c = vec![0.0; n];
    for j in 0..n {
        (lower[j], upper[j]) = if j % 2 == 0 {
            (-1.0, 1.0)
        } else {
            (0.0, 1.0 + 4.0 * r.next())
        };
        c[j] = 2.0 * r.next() - 1.0;
    }
    let mut a_le = Vec::new();
    let mut b_le = Vec::new();
    for _ in 0..m {
        let row: Vec<f64> = (0..n)
            .map(|_| {
                if r.next() < density {
                    2.0 * r.next() - 1.0
                } else {
                    0.0
                }
            })
            .collect();
        b_le.push(0.2 * norm(&row) * r.next());
        a_le.push(row);
    }
    Data {
        maximize: false,
        c,
        a_le,
        b_le,
        a_eq: Vec::new(),
        b_eq: Vec::new(),
        lower,
        upper,
    }
}

/// Free variables only, held by the rows `A·x ≤ b`, `b > 0`, under costs `c = −Aᵀ·y` for some
/// `y ≥ 0`, which bound the objective from below.
fn free_variables(m: usize, n: usize, seed: u64) -> Data {
    let mut r = Lcg(seed);
    let a_le: Vec<Vec<f64>> = (0..m)
        .map(|_| {
            (0..n)
                .map(|_| {
                    if r.next() < 0.4 {
                        2.0 * r.next() - 1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let y: Vec<f64> = (0..m)
        .map(|_| if r.next() < 0.7 { r.next() } else { 0.0 })
        .collect();
    let c: Vec<f64> = (0..n)
        .map(|j| -(0..m).map(|i| a_le[i][j] * y[i]).sum::<f64>())
        .collect();
    let b_le: Vec<f64> = (0..m).map(|_| 1.0 + r.next()).collect();
    let inf = f64::INFINITY;
    Data {
        maximize: false,
        c,
        a_le,
        b_le,
        a_eq: Vec::new(),
        b_eq: Vec::new(),
        lower: vec![-inf; n],
        upper: vec![inf; n],
    }
}

#[test]
fn the_dual_method_agrees_with_the_primal_method_on_structured_programmes() {
    // Degenerate transportation and assignment problems with redundant equations, a staircase of
    // production periods (sparse, equality-heavy; and infeasible), networks, programmes of boxed
    // variables, of free variables, the scoreboard's with every inequality tight at the point
    // (degenerate), and in rows of units powers of ten apart; and made infeasible and unbounded.
    let mut cases: Vec<(String, Data, LpStatus)> = Vec::new();
    let seeds = if cfg!(miri) { 0..0 } else { 1..3 };
    for seed in seeds {
        let s = seed as u64;
        cases.push((
            format!("transport 10×15 {s}"),
            transportation(10, 15, s, true),
            LpStatus::Optimal,
        ));
        cases.push((
            format!("transport 20×20 unbalanced {s}"),
            transportation(20, 20, s, false),
            LpStatus::Optimal,
        ));
        cases.push((
            format!("assignment 20 {s}"),
            {
                let mut p = transportation(20, 20, s, true);
                p.b_eq.iter_mut().for_each(|b| *b = 1.0);
                p
            },
            LpStatus::Optimal,
        ));
        cases.push((
            format!("production 12×8×4 {s}"),
            production(12, 8, 4, s, false),
            LpStatus::Optimal,
        ));
        cases.push((
            format!("production tight {s}"),
            production(12, 8, 4, s, true),
            LpStatus::Infeasible,
        ));
        cases.push((
            format!("network 60 {s}"),
            network_flow(60, 300, s),
            LpStatus::Optimal,
        ));
        cases.push((
            format!("boxed 60×200 {s}"),
            boxed(60, 200, 0.3, s),
            LpStatus::Optimal,
        ));
        cases.push((
            format!("boxed 120×240 dense {s}"),
            boxed(120, 240, 1.0, s),
            LpStatus::Optimal,
        ));
        cases.push((
            format!("free 80×60 {s}"),
            free_variables(80, 60, s),
            LpStatus::Optimal,
        ));
        let mut degenerate = scoreboard(80, 0, 160, 1.0, s);
        for (row, b) in degenerate.a_le.iter().zip(&mut degenerate.b_le) {
            *b = row.iter().sum::<f64>() * 0.5;
        }
        cases.push((
            format!("degenerate 80×160 {s}"),
            degenerate,
            LpStatus::Optimal,
        ));
        let mut wide = scoreboard(60, 0, 120, 0.3, s);
        for (i, (row, b)) in wide.a_le.iter_mut().zip(&mut wide.b_le).enumerate() {
            let scale = 10f64.powi(i as i32 % 9 - 4);
            row.iter_mut().for_each(|v| *v *= scale);
            *b *= scale;
        }
        cases.push((
            format!("rows in units 1e-4 to 1e4 {s}"),
            wide,
            LpStatus::Optimal,
        ));
        let mut infeasible = scoreboard(60, 10, 120, 0.2, s);
        let mut row = vec![0.0; 120];
        let mut rhs = 0.0;
        for (a, b) in infeasible.a_le.iter().take(3).zip(&infeasible.b_le) {
            row.iter_mut().zip(a).for_each(|(x, v)| *x -= v);
            rhs -= b;
        }
        infeasible.a_le.push(row);
        infeasible.b_le.push(rhs - 0.5);
        cases.push((
            format!("infeasible 70×120 {s}"),
            infeasible,
            LpStatus::Infeasible,
        ));
        let mut unbounded = scoreboard(60, 10, 120, 0.2, s);
        for row in unbounded.a_le.iter_mut().chain(&mut unbounded.a_eq) {
            row[119] = 0.0;
        }
        unbounded.c[119] = -1.0;
        unbounded.lower[119] = f64::NEG_INFINITY;
        cases.push((
            format!("unbounded 70×120 {s}"),
            unbounded,
            LpStatus::Unbounded,
        ));
    }
    let mut fewer = 0;
    for (name, p, status) in &cases {
        let d = p
            .program()
            .solve_with(&dual())
            .unwrap_or_else(|| panic!("{name}: no answer"));
        let q = p
            .program()
            .solve_with(&primal())
            .unwrap_or_else(|| panic!("{name}: no answer"));
        assert_eq!((d.status, q.status), (*status, *status), "{name}");
        consistent(p, &d);
        match status {
            LpStatus::Optimal => {
                let err = objective_error(d.objective, q.objective);
                assert!(
                    err <= 1e-10,
                    "{name}: {} against {}",
                    d.objective,
                    q.objective
                );
                assert_certified(p, &d, 1e-9);
            }
            LpStatus::Unbounded => assert!(infeasibility(p, &d) <= 1e-9, "{name}"),
            _ => {}
        }
        fewer += usize::from(d.pivots < q.pivots);
    }
    // The dual method takes fewer pivots on most: all but the free variables and the unbounded
    // programmes, whose dual has no feasible point and which the primal method finishes.
    assert!(
        cfg!(miri) || fewer * 10 >= cases.len() * 6,
        "{fewer} of {}",
        cases.len()
    );
}

#[test]
fn an_iteration_limit_under_the_dual_method_is_reported_with_the_point_reached() {
    let p = scoreboard(30, 5, 60, 1.0, 101);
    let r = p.solve_with(&dual().with_max_iter(10));
    assert_eq!(r.status, LpStatus::IterationLimit);
    assert!(r.pivots <= 10, "{}", r.pivots);
    consistent(&p, &r);
    // The dual method has half the pivots allowed, the primal method the rest: a budget of
    // twice what the dual method takes lets it finish.
    let r = p.solve_with(&dual());
    assert_eq!(r.status, LpStatus::Optimal);
    let again = p.solve_with(&dual().with_max_iter(2 * r.pivots));
    assert_eq!((again.status, again.pivots), (LpStatus::Optimal, r.pivots));
}

#[test]
fn single_precision_reaches_the_optima_of_the_scoreboards_programmes_under_either_method() {
    // A programme in `f32` is solved in `f64`. In `f32` arithmetic the primal method's bounds on
    // the errors of the reduced costs grew with the rows until every reduced cost passed for
    // rounding: it reported these optimal at −184, −420 and −240 after 754, 2015 and 1610
    // pivots. Within a budget of 2500 pivots the primal method now ends at the optimum or says
    // that it has not reached it (it needs 5465, 3279 and 8660); the dual method, the default,
    // ends at the optimum, measured within 2e-7 of HiGHS's in `f64`.
    let programmes: &[(usize, usize, usize, f64, u64, f64)] = &[
        (200, 20, 400, 1.0, 103, -526.903_899_346),
        (200, 20, 400, 0.1, 203, -533.263_798_459),
        (300, 30, 600, 0.1, 204, -674.365_125_79),
    ];
    let count = if cfg!(miri) { 0 } else { programmes.len() };
    for &(m_le, m_eq, n, density, seed, highs) in &programmes[..count] {
        let p = scoreboard(m_le, m_eq, n, density, seed);
        let primal = LpOptions::<f32>::new()
            .with_dual_threshold(usize::MAX)
            .with_max_iter(2500);
        for (method, opts) in [("primal", primal), ("dual", LpOptions::<f32>::new())] {
            let r = p
                .program_f32()
                .solve_with(&opts)
                .unwrap_or_else(|| panic!("{m_le}×{n} {density} {method}: no verified answer"));
            if method == "primal" && r.status == LpStatus::IterationLimit {
                assert_eq!(r.pivots, 2500, "{m_le}×{n} {density}");
                continue;
            }
            assert_eq!(r.status, LpStatus::Optimal, "{m_le}×{n} {density} {method}");
            let err = objective_error(f64::from(r.objective), highs);
            assert!(
                err <= 1e-6,
                "{m_le}×{n} {density} {method}: {} against {highs}",
                r.objective
            );
        }
    }
}

#[test]
fn single_precision_agrees_with_highs_on_every_programme_it_states_exactly_under_either_method() {
    // The programmes whose rows and columns are of one scale, whose numbers are integers below
    // 2²⁴ and so exact in `f32` (the others' right-hand sides and costs change when they are
    // rounded), by the primal method and by the dual method throughout. Measured: the optima
    // within 1.2e-6 (primal) and 3.9e-6 (dual) of HiGHS's, the rounding of the point and of the
    // objective's sum in `f32`; the residuals of the certificate below 1e-7 but for the duality
    // gap against the dual objective with the bounds, 1.2e-4: reduced costs within the
    // tolerance of 1e-4 priced at the far bounds they point to.
    let methods = [
        (
            "primal",
            LpOptions::<f32>::new().with_dual_threshold(usize::MAX),
        ),
        ("dual", LpOptions::<f32>::new().with_dual_threshold(0)),
    ];
    let mut checked = 0;
    for c in cases() {
        let case = generate(c.seed);
        if case.wide {
            continue;
        }
        let p = case.data();
        for (method, opts) in &methods {
            let r = p
                .program_f32()
                .solve_with(opts)
                .unwrap_or_else(|| panic!("seed {} {method}: no verified answer", c.seed));
            assert_eq!(r.status, c.status, "seed {} {method}", c.seed);
            if r.status == LpStatus::Optimal {
                let err = objective_error(f64::from(r.objective), c.objective);
                assert!(err <= 1e-5, "seed {} {method}: {err:e}", c.seed);
                assert_single_certified(&p, &r, 2e-4);
            }
        }
        checked += 1;
    }
    assert!(cfg!(miri) || checked >= 450, "{checked}");
}

#[test]
fn single_precision_answers_beyond_its_range_are_as_the_method_in_single_precision_gives_them() {
    // A programme in `f32` is solved in `f64`; where its answer leaves the range of `f32`, the
    // outcome is the one the method in `f32` arithmetic gives: a dual value or an objective
    // beyond the range is infinite, the rest of the answer as it is; and there is no answer
    // where the scaled working problem, or the sizes of the rows' terms at the point, are
    // beyond the range. (Rounding the whole answer and refusing any infinite value lost the
    // first two.)
    let one = |c: f32, a: f32, b: f32| {
        LinearProgram::maximize(&[c])
            .subject_to_le(&Matrix::from_rows(&[[a]]), &[b])
            .solve()
    };
    // x = 1, the objective 1e30 and the dual value 1e40.
    let r = one(1e30, 1e-10, 1e-10).expect("an answer");
    assert_eq!(
        (r.status, r.x[0], r.objective),
        (LpStatus::Optimal, 1.0, 1e30)
    );
    assert_eq!(r.dual_le[0], f32::INFINITY);
    // x = 1e30, the objective 1e60 and the dual value 1e60.
    let r = one(1e30, 1e-30, 1.0).expect("an answer");
    assert_eq!((r.status, r.x[0]), (LpStatus::Optimal, 1e30));
    assert_eq!((r.objective, r.dual_le[0]), (f32::INFINITY, f32::INFINITY));
    // A dual value of 3e38 is finite.
    let r = one(3e28, 1e-10, 1e-10).expect("an answer");
    assert!(r.dual_le[0].is_finite() && (r.dual_le[0] / 3e38 - 1.0).abs() < 1e-5);
    // Unbounded and infeasible programmes keep their status beside such a dual value.
    let r = LinearProgram::maximize(&[1e30f32, 1.0])
        .subject_to_le(&Matrix::from_rows(&[[1e-10f32, 0.0]]), &[1e-10])
        .solve()
        .expect("an answer");
    assert_eq!(r.status, LpStatus::Unbounded);
    let r = LinearProgram::maximize(&[1e30f32])
        .subject_to_le(
            &Matrix::from_rows(&[[1e-10f32], [-1e-10]]),
            &[1e-10, -2e-10],
        )
        .solve()
        .expect("an answer");
    assert_eq!(r.status, LpStatus::Infeasible);
    // The slack 3e38 − (−3e38) is infinite.
    let r = LinearProgram::maximize(&[1.0f32])
        .subject_to_le(&Matrix::from_rows(&[[1.0f32], [-1.0]]), &[3e38, 3e38])
        .bounds(&[f32::NEG_INFINITY], &[f32::INFINITY])
        .solve()
        .expect("an answer");
    assert_eq!(
        (r.status, r.x[0], r.slack[1]),
        (LpStatus::Optimal, 3e38, f32::INFINITY)
    );
    // No answer: the terms of the row x ≤ 3e38 at x = 3e38 add up beyond the range; and the
    // cost 1e30 of a column whose largest element is 1e-10 is scaled beyond it.
    let r = LinearProgram::maximize(&[2.0f32, 1.0])
        .subject_to_le(
            &Matrix::from_rows(&[[1.0f32, 0.0], [0.0, 1.0]]),
            &[3e38, 1.0],
        )
        .solve();
    assert!(r.is_none(), "{r:?}");
    let r = LinearProgram::minimize(&[1e30f32, 1.0])
        .subject_to_le(&Matrix::from_rows(&[[-1e-10f32, -1.0]]), &[-1.0])
        .solve();
    assert!(r.is_none(), "{r:?}");
}

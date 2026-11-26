//! The stiff problems of the tests, set up as in `data/gen_stiff.py`, with their Jacobians.

use a::math::matrix::Matrix;
use a::math::numerics::Real;

/// The stiff problems of the comparisons with SciPy's `Radau` and `BDF`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stiff {
    /// Robertson's chemical kinetics from `(1, 0, 0)` to `t = 10¹¹`.
    Robertson,
    /// Van der Pol's oscillator `y₀″ − 1000·(1 − y₀²)·y₀′ + y₀ = 0` from `(2, 0)` on `[0, 3000]`.
    VanDerPol,
    /// HIRES, the eight reactions of Schäfer's plant physiology model, on `[0, 321.8122]`.
    Hires,
    /// OREGO, the Oregonator of Field and Noyes, on `[0, 360]`.
    Orego,
    /// The linear system `y′ = A·y` of dimension 50 with eigenvalues from −1 to −10⁶, on `[0, 10]`.
    Linear,
}

/// The dimension of the linear system.
pub const N50: usize = 50;

/// The reflection `Q = I − 2·v·vᵀ/(vᵀ·v)` of the linear system, `vᵢ = ((37·i) mod 50 + 1)/50`, and
/// its eigenvalues `λᵢ = −10^(6·i/49)`.
pub fn linear_parts() -> (Matrix<f64>, Vec<f64>) {
    let v: Vec<f64> = (0..N50)
        .map(|i| ((37 * i) % 50 + 1) as f64 / 50.0)
        .collect();
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let q = Matrix::from_fn(N50, N50, |i, j| {
        f64::from(u8::from(i == j)) - 2.0 * v[i] * v[j] / vv
    });
    let lambda = (0..N50)
        .map(|i| -(10f64.powf(6.0 * i as f64 / 49.0)))
        .collect();
    (q, lambda)
}

/// The matrix `A = Q·diag(λ)·Q` of the linear system.
pub fn linear_matrix() -> Matrix<f64> {
    let (q, lambda) = linear_parts();
    Matrix::from_fn(N50, N50, |i, j| {
        (0..N50).map(|k| q[(i, k)] * lambda[k] * q[(k, j)]).sum()
    })
}

/// The solution of the linear system at `t`: `Q·diag(e^(λ·t))·Q·y₀`, `y₀` all ones.
pub fn linear_exact(t: f64) -> Vec<f64> {
    let (q, lambda) = linear_parts();
    let qy: Vec<f64> = (0..N50)
        .map(|k| (0..N50).map(|j| q[(k, j)]).sum())
        .collect();
    (0..N50)
        .map(|i| {
            (0..N50)
                .map(|k| q[(i, k)] * (lambda[k] * t).exp() * qy[k])
                .sum()
        })
        .collect()
}

impl Stiff {
    /// The initial state.
    pub fn y0(self) -> Vec<f64> {
        match self {
            Stiff::Robertson => vec![1.0, 0.0, 0.0],
            Stiff::VanDerPol => vec![2.0, 0.0],
            Stiff::Hires => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057],
            Stiff::Orego => vec![1.0, 2.0, 3.0],
            Stiff::Linear => vec![1.0; N50],
        }
    }

    /// The end of the interval, which starts at 0.
    pub fn t1(self) -> f64 {
        match self {
            Stiff::Robertson => 1e11,
            Stiff::VanDerPol => 3000.0,
            Stiff::Hires => 321.8122,
            Stiff::Orego => 360.0,
            Stiff::Linear => 10.0,
        }
    }

    /// The right-hand side; `a` is the matrix of the linear system, unused otherwise.
    pub fn rhs<T: Real>(self, a: &Matrix<f64>, y: &Matrix<T>, dy: &mut Matrix<T>) {
        let c = T::from_f64;
        match self {
            Stiff::Robertson => {
                let (p, q, r) = (y[(0, 0)], y[(1, 0)], y[(2, 0)]);
                dy[(0, 0)] = c(-0.04) * p + c(1e4) * q * r;
                dy[(1, 0)] = c(0.04) * p - c(1e4) * q * r - c(3e7) * q * q;
                dy[(2, 0)] = c(3e7) * q * q;
            }
            Stiff::VanDerPol => {
                let (u, v) = (y[(0, 0)], y[(1, 0)]);
                dy[(0, 0)] = v;
                dy[(1, 0)] = c(1000.0) * (T::ONE - u * u) * v - u;
            }
            Stiff::Hires => {
                let v: Vec<T> = y.as_slice().to_vec();
                dy[(0, 0)] = c(-1.71) * v[0] + c(0.43) * v[1] + c(8.32) * v[2] + c(0.0007);
                dy[(1, 0)] = c(1.71) * v[0] - c(8.75) * v[1];
                dy[(2, 0)] = c(-10.03) * v[2] + c(0.43) * v[3] + c(0.035) * v[4];
                dy[(3, 0)] = c(8.32) * v[1] + c(1.71) * v[2] - c(1.12) * v[3];
                dy[(4, 0)] = c(-1.745) * v[4] + c(0.43) * v[5] + c(0.43) * v[6];
                dy[(5, 0)] = c(-280.0) * v[5] * v[7] + c(0.69) * v[3] + c(1.71) * v[4]
                    - c(0.43) * v[5]
                    + c(0.69) * v[6];
                dy[(6, 0)] = c(280.0) * v[5] * v[7] - c(1.81) * v[6];
                dy[(7, 0)] = c(-280.0) * v[5] * v[7] + c(1.81) * v[6];
            }
            Stiff::Orego => {
                let (p, q, r) = (y[(0, 0)], y[(1, 0)], y[(2, 0)]);
                dy[(0, 0)] = c(77.27) * (q + p * (T::ONE - c(8.375e-6) * p - q));
                dy[(1, 0)] = (r - (T::ONE + p) * q) / c(77.27);
                dy[(2, 0)] = c(0.161) * (p - r);
            }
            Stiff::Linear => {
                for i in 0..N50 {
                    let mut s = T::ZERO;
                    for j in 0..N50 {
                        s += c(a[(i, j)]) * y[(j, 0)];
                    }
                    dy[(i, 0)] = s;
                }
            }
        }
    }

    /// The Jacobian `∂f/∂y`, into `out`, which is zero when called.
    pub fn jacobian<T: Real>(self, a: &Matrix<f64>, y: &Matrix<T>, out: &mut Matrix<T>) {
        let c = T::from_f64;
        match self {
            Stiff::Robertson => {
                let (q, r) = (y[(1, 0)], y[(2, 0)]);
                out[(0, 0)] = c(-0.04);
                out[(0, 1)] = c(1e4) * r;
                out[(0, 2)] = c(1e4) * q;
                out[(1, 0)] = c(0.04);
                out[(1, 1)] = c(-1e4) * r - c(6e7) * q;
                out[(1, 2)] = c(-1e4) * q;
                out[(2, 1)] = c(6e7) * q;
            }
            Stiff::VanDerPol => {
                let (u, v) = (y[(0, 0)], y[(1, 0)]);
                out[(0, 1)] = T::ONE;
                out[(1, 0)] = c(-2000.0) * u * v - T::ONE;
                out[(1, 1)] = c(1000.0) * (T::ONE - u * u);
            }
            Stiff::Hires => {
                let (v5, v7) = (y[(5, 0)], y[(7, 0)]);
                let rows: [&[(usize, f64)]; 8] = [
                    &[(0, -1.71), (1, 0.43), (2, 8.32)],
                    &[(0, 1.71), (1, -8.75)],
                    &[(2, -10.03), (3, 0.43), (4, 0.035)],
                    &[(1, 8.32), (2, 1.71), (3, -1.12)],
                    &[(4, -1.745), (5, 0.43), (6, 0.43)],
                    &[(3, 0.69), (4, 1.71), (5, -0.43), (6, 0.69)],
                    &[(6, -1.81)],
                    &[(6, 1.81)],
                ];
                for (i, row) in rows.iter().enumerate() {
                    for &(j, v) in *row {
                        out[(i, j)] = c(v);
                    }
                }
                let (a5, a7) = (c(280.0) * v7, c(280.0) * v5);
                out[(5, 5)] -= a5;
                out[(5, 7)] -= a7;
                out[(6, 5)] += a5;
                out[(6, 7)] += a7;
                out[(7, 5)] -= a5;
                out[(7, 7)] -= a7;
            }
            Stiff::Orego => {
                let (p, q) = (y[(0, 0)], y[(1, 0)]);
                out[(0, 0)] = c(77.27) * (T::ONE - c(2.0 * 8.375e-6) * p - q);
                out[(0, 1)] = c(77.27) * (T::ONE - p);
                out[(1, 0)] = -q / c(77.27);
                out[(1, 1)] = -(T::ONE + p) / c(77.27);
                out[(1, 2)] = T::ONE / c(77.27);
                out[(2, 0)] = c(0.161);
                out[(2, 2)] = c(-0.161);
            }
            Stiff::Linear => {
                for i in 0..N50 {
                    for j in 0..N50 {
                        out[(i, j)] = c(a[(i, j)]);
                    }
                }
            }
        }
    }
}

/// The error of `y` against `exact`, componentwise relative to `|exact| + atol`, which for these
/// problems, whose components differ in size by many orders, is what the tolerances ask for.
pub fn stiff_error<T: Real>(y: &[T], exact: &[f64], atol: f64) -> f64 {
    assert_eq!(y.len(), exact.len());
    y.iter()
        .zip(exact)
        .map(|(&v, &e)| (v.to_f64() - e).abs() / (e.abs() + atol))
        .fold(0.0, f64::max)
}

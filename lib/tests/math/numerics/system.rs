//! The systems of `a::math::numerics::system`: Newton's method with a Jacobian and Broyden's
//! method, both with Dennis and Schnabel's line search, on the More–Garbow–Hillstrom systems
//! against SciPy's MINPACK and against their roots to 40 digits (`data/gen_systems.py`), on the
//! cases of the audit, and on the failures that must be reported rather than passed off.

use a::math::matrix::Matrix;
use a::math::numerics::{
    SystemOptions, SystemResult, SystemStop, broyden, broyden_with, broyden_with_jacobian,
    newton_system, newton_system_with,
};

/// A column from a slice.
fn col<T: Copy>(v: &[T]) -> Matrix<T> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

type Residual = Box<dyn Fn(&[f64]) -> Vec<f64>>;
type Jacobian = Box<dyn Fn(&[f64]) -> Vec<Vec<f64>>>;

/// A system of the battery: its residual, its Jacobian and its standard starting point.
struct Problem {
    name: &'static str,
    x0: Vec<f64>,
    f: Residual,
    j: Jacobian,
}

impl Problem {
    fn residual(&self, x: &Matrix<f64>) -> Matrix<f64> {
        col(&(self.f)(x.as_slice()))
    }

    fn jacobian(&self, x: &Matrix<f64>) -> Matrix<f64> {
        let rows = (self.j)(x.as_slice());
        Matrix::from_fn(rows.len(), rows.len(), |i, k| rows[i][k])
    }
}

/// θ of the helical valley: the angle of (x₁, x₂) in turns, in (−1/4, 3/4).
fn theta(x1: f64, x2: f64) -> f64 {
    let tpi = 2.0 * core::f64::consts::PI;
    if x1 > 0.0 {
        (x2 / x1).atan() / tpi
    } else if x1 < 0.0 {
        (x2 / x1).atan() / tpi + 0.5
    } else if x2 >= 0.0 {
        0.25
    } else {
        -0.25
    }
}

/// The systems of nonlinear equations of Moré, Garbow and Hillstrom (ACM TOMS 7, 1981) that the
/// issue names, in the order of `data/gen_systems.py`.
fn mgh() -> Vec<Problem> {
    let mut v = Vec::new();
    v.push(Problem {
        name: "rosenbrock",
        x0: vec![-1.2, 1.0],
        f: Box::new(|x| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]),
        j: Box::new(|x| vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]]),
    });
    v.push(Problem {
        name: "powell_singular",
        x0: vec![3.0, -1.0, 0.0, 1.0],
        f: Box::new(|x| {
            let (s5, s10) = (5f64.sqrt(), 10f64.sqrt());
            let (a, b) = (x[1] - 2.0 * x[2], x[0] - x[3]);
            vec![x[0] + 10.0 * x[1], s5 * (x[2] - x[3]), a * a, s10 * b * b]
        }),
        j: Box::new(|x| {
            let (s5, s10) = (5f64.sqrt(), 10f64.sqrt());
            let (a, b) = (x[1] - 2.0 * x[2], x[0] - x[3]);
            vec![
                vec![1.0, 10.0, 0.0, 0.0],
                vec![0.0, 0.0, s5, -s5],
                vec![0.0, 2.0 * a, -4.0 * a, 0.0],
                vec![2.0 * s10 * b, 0.0, 0.0, -2.0 * s10 * b],
            ]
        }),
    });
    v.push(Problem {
        name: "powell_badly_scaled",
        x0: vec![0.0, 1.0],
        f: Box::new(|x| {
            vec![
                1e4 * x[0] * x[1] - 1.0,
                (-x[0]).exp() + (-x[1]).exp() - 1.0001,
            ]
        }),
        j: Box::new(|x| {
            vec![
                vec![1e4 * x[1], 1e4 * x[0]],
                vec![-(-x[0]).exp(), -(-x[1]).exp()],
            ]
        }),
    });
    v.push(Problem {
        name: "helical_valley",
        x0: vec![-1.0, 0.0, 0.0],
        f: Box::new(|x| {
            let r = (x[0] * x[0] + x[1] * x[1]).sqrt();
            vec![
                10.0 * (x[2] - 10.0 * theta(x[0], x[1])),
                10.0 * (r - 1.0),
                x[2],
            ]
        }),
        j: Box::new(|x| {
            let r2 = x[0] * x[0] + x[1] * x[1];
            let r = r2.sqrt();
            let tpi = 2.0 * core::f64::consts::PI;
            vec![
                vec![100.0 * x[1] / (tpi * r2), -100.0 * x[0] / (tpi * r2), 10.0],
                vec![10.0 * x[0] / r, 10.0 * x[1] / r, 0.0],
                vec![0.0, 0.0, 1.0],
            ]
        }),
    });
    let n = 10;
    v.push(Problem {
        name: "trigonometric_10",
        x0: vec![1.0 / n as f64; n],
        f: Box::new(move |x| {
            let sc: f64 = x.iter().map(|t| t.cos()).sum();
            (0..n)
                .map(|i| n as f64 - sc + (i + 1) as f64 * (1.0 - x[i].cos()) - x[i].sin())
                .collect()
        }),
        j: Box::new(move |x| {
            (0..n)
                .map(|i| {
                    let mut row: Vec<f64> = x.iter().map(|t| t.sin()).collect();
                    row[i] += (i + 1) as f64 * x[i].sin() - x[i].cos();
                    row
                })
                .collect()
        }),
    });
    for (name, n) in [
        ("discrete_boundary_10", 10),
        ("discrete_boundary_50", 50),
        ("discrete_boundary_100", 100),
    ] {
        let h = 1.0 / (n as f64 + 1.0);
        v.push(Problem {
            name,
            x0: (1..=n)
                .map(|k| k as f64 * h * (k as f64 * h - 1.0))
                .collect(),
            f: Box::new(move |x| {
                (0..n)
                    .map(|k| {
                        let t = (k + 1) as f64 * h;
                        let xm = if k > 0 { x[k - 1] } else { 0.0 };
                        let xp = if k + 1 < n { x[k + 1] } else { 0.0 };
                        2.0 * x[k] - xm - xp + h * h * (x[k] + t + 1.0).powi(3) / 2.0
                    })
                    .collect()
            }),
            j: Box::new(move |x| {
                let mut m = vec![vec![0.0; n]; n];
                for k in 0..n {
                    let t = (k + 1) as f64 * h;
                    m[k][k] = 2.0 + 1.5 * h * h * (x[k] + t + 1.0).powi(2);
                    if k > 0 {
                        m[k][k - 1] = -1.0;
                    }
                    if k + 1 < n {
                        m[k][k + 1] = -1.0;
                    }
                }
                m
            }),
        });
    }
    for (name, n) in [
        ("broyden_tridiagonal_10", 10),
        ("broyden_tridiagonal_50", 50),
        ("broyden_tridiagonal_100", 100),
    ] {
        v.push(Problem {
            name,
            x0: vec![-1.0; n],
            f: Box::new(move |x| {
                (0..n)
                    .map(|k| {
                        let xm = if k > 0 { x[k - 1] } else { 0.0 };
                        let xp = if k + 1 < n { x[k + 1] } else { 0.0 };
                        (3.0 - 2.0 * x[k]) * x[k] - xm - 2.0 * xp + 1.0
                    })
                    .collect()
            }),
            j: Box::new(move |x| {
                let mut m = vec![vec![0.0; n]; n];
                for k in 0..n {
                    m[k][k] = 3.0 - 4.0 * x[k];
                    if k > 0 {
                        m[k][k - 1] = -1.0;
                    }
                    if k + 1 < n {
                        m[k][k + 1] = -2.0;
                    }
                }
                m
            }),
        });
    }
    for (name, n) in [
        ("broyden_banded_10", 10),
        ("broyden_banded_50", 50),
        ("broyden_banded_100", 100),
    ] {
        v.push(Problem {
            name,
            x0: vec![-1.0; n],
            f: Box::new(move |x| banded(x, n)),
            j: Box::new(move |x| banded_jacobian(x, n)),
        });
    }
    v
}

/// Broyden's banded function: `xₖ(2 + 5xₖ²) + 1 − Σ xᵢ(1 + xᵢ)` over the five unknowns before
/// `k` and the one after it.
fn banded(x: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let band = k.saturating_sub(5)..=(k + 1).min(n - 1);
            let s: f64 = band.filter(|&i| i != k).map(|i| x[i] * (1.0 + x[i])).sum();
            x[k] * (2.0 + 5.0 * x[k] * x[k]) + 1.0 - s
        })
        .collect()
}

/// Its Jacobian.
fn banded_jacobian(x: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for k in 0..n {
        for i in k.saturating_sub(5)..=(k + 1).min(n - 1) {
            if i != k {
                m[k][i] = -(1.0 + 2.0 * x[i]);
            }
        }
        m[k][k] = 2.0 + 15.0 * x[k] * x[k];
    }
    m
}

/// The roots of the battery to 40 digits by mpmath's Newton iteration, rounded to `f64`, from
/// `data/gen_systems.py`: for the trigonometric system, which has several, the one both methods
/// here reach from the standard start; Rosenbrock's, Powell's singular system's and the helical
/// valley's exactly.
const ROOTS: [(&str, &[u64]); 14] = [
    ("rosenbrock", &[0x3ff0000000000000, 0x3ff0000000000000]),
    (
        "powell_singular",
        &[
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ),
    (
        "powell_badly_scaled",
        &[0x3ee707b2b0b09734, 0x40223658dd909b84],
    ),
    (
        "helical_valley",
        &[0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
    ),
    (
        "trigonometric_10",
        &[
            0x3fa19c63f01461bf,
            0x3fa1efc174d31157,
            0x3fa249c953d2e5c7,
            0x3fa2ab91e2eada0e,
            0x3fa3167bd15decbf,
            0x3fa38c4f9a03b1e1,
            0x3fa40f6b61341f8e,
            0x3fa4a30dcd584b25,
            0x3fc701123ce295e4,
            0x3fc3ffb382037ad3,
        ],
    ),
    (
        "discrete_boundary_10",
        &[
            0xbfa619b87880426e,
            0xbfb4e23d936b8f90,
            0xbfbd4eef8f1fa24c,
            0xbfc20b6c131ccca3,
            0xbfc477e35e394b88,
            0xbfc5be89422279dc,
            0xbfc5a4bd9735f269,
            0xbfc3df377e3f9ca4,
            0xbfc00ba96f8f9bdf,
            0xbfb34e7f79c232e1,
        ],
    ),
    (
        "discrete_boundary_50",
        &[
            0xbf83e0819ffc16e3,
            0xbf93ac99fe12ad73,
            0xbf9d3378dc48efe8,
            0xbfa3419d26d2455d,
            0xbfa7cd14f6a71510,
            0xbfac3b4094198afe,
            0xbfb0459992062cc1,
            0xbfb25dfadbc3a647,
            0xbfb4664360848857,
            0xbfb65decc0670868,
            0xbfb8446ab3ad31c4,
            0xbfba192ab6c27770,
            0xbfbbdb93b0983ef7,
            0xbfbd8b0592e6284e,
            0xbfbf26d8f3d03ef9,
            0xbfc0572f5034c69f,
            0xbfc1106f93be4cc9,
            0xbfc1becd2dfa818c,
            0xbfc261e3671ac889,
            0xbfc2f9489dca3545,
            0xbfc3848df991cba7,
            0xbfc4033f176b1215,
            0xbfc474e1affd00eb,
            0xbfc4d8f536f1d3b9,
            0xbfc52ef272c524be,
            0xbfc5764b0c58d564,
            0xbfc5ae69158c59c6,
            0xbfc5d6ae85fcc3e4,
            0xbfc5ee74acfc13e5,
            0xbfc5f50b97b3814e,
            0xbfc5e9b96a4639a3,
            0xbfc5cbb9aaa7f766,
            0xbfc59a3c7bb4491d,
            0xbfc55465c6e7bb42,
            0xbfc4f94c52ea9789,
            0xbfc487f8c4e4ac53,
            0xbfc3ff648a518710,
            0xbfc35e78a8c38495,
            0xbfc2a40c70b0a5cf,
            0xbfc1cee410048746,
            0xbfc0ddaf00c5334b,
            0xbfbf9e0c9f34e11a,
            0xbfbd42d56aebc198,
            0xbfbaa684fde4db9d,
            0xbfb7c5ae767e65e8,
            0xbfb49ca657111c52,
            0xbfb1277cd5974215,
            0xbfaac3ef1339e1de,
            0xbfa28f14c644466c,
            0xbf934d3f473c7282,
        ],
    ),
    (
        "discrete_boundary_100",
        &[
            0xbf742cf8006dc04b,
            0xbf8412e2aa2a45f5,
            0xbf8df4cf539b2e85,
            0xbf93ddeda4d94dc8,
            0xbf98b3cee1e59394,
            0xbf9d7bd5eae986b1,
            0xbfa11ae61890bd24,
            0xbfa370bd02e96551,
            0xbfa5bf53496b0ea4,
            0xbfa8068be972f9de,
            0xbfaa464944d7b08b,
            0xbfac7e6d1dd14d7d,
            0xbfaeaed892c082e3,
            0xbfb06bb60ce99080,
            0xbfb17c03be426976,
            0xbfb28844e97d5c4b,
            0xbfb39068bf9c3b2c,
            0xbfb4945e1416a162,
            0xbfb594135a4bb578,
            0xbfb68f76a2de5a2a,
            0xbfb7867598faf88f,
            0xbfb878fd7f860487,
            0xbfb966fb2e32526f,
            0xbfba505b0e7e4aad,
            0xbfbb35091896fca1,
            0xbfbc14f0d02006ff,
            0xbfbceffd40df3f59,
            0xbfbdc618fb4af5fb,
            0xbfbe972e10f9a597,
            0xbfbf632610f1d04d,
            0xbfc014f501ec5e3d,
            0xbfc075b133ff5a1b,
            0xbfc0d3bb96a3acef,
            0xbfc12f07d8764934,
            0xbfc187895c8d3a20,
            0xbfc1dd3338314adc,
            0xbfc22ff830828100,
            0xbfc27fcab80682e3,
            0xbfc2cc9cec1ff720,
            0xbfc31660926dde84,
            0xbfc35d071611dd28,
            0xbfc3a08184dc59e5,
            0xbfc3e0c08c5d4d62,
            0xbfc41db476d88ae3,
            0xbfc4574d281c3d60,
            0xbfc48d7a1a384295,
            0xbfc4c02a5a14fc37,
            0xbfc4ef4c83e81c9b,
            0xbfc51acebf85e085,
            0xbfc5429ebc8d12b3,
            0xbfc566a9ae6c1f91,
            0xbfc586dc483d6806,
            0xbfc5a322b878e92f,
            0xbfc5bb68a479347d,
            0xbfc5cf9923d1978c,
            0xbfc5df9ebb733518,
            0xbfc5eb63589eb0eb,
            0xbfc5f2d04b9feece,
            0xbfc5f5ce425140cd,
            0xbfc5f44542633b15,
            0xbfc5ee1ca3663b19,
            0xbfc5e33b089284ba,
            0xbfc5d3865a4baa27,
            0xbfc5bee3bf5bc25d,
            0xbfc5a53795e2be1f,
            0xbfc586656bf5f39a,
            0xbfc5624ff7ebbdc9,
            0xbfc538d9104ecd6c,
            0xbfc509e1a37485e6,
            0xbfc4d549aeb17835,
            0xbfc49af03526c14d,
            0xbfc45ab33622be92,
            0xbfc4146fa30f3332,
            0xbfc3c80154e69ac1,
            0xbfc37543012a006d,
            0xbfc31c0e2e50460a,
            0xbfc2bc3b27a7520d,
            0xbfc255a0f09f1f48,
            0xbfc1e8153776238e,
            0xbfc1736c473df32f,
            0xbfc0f778f92e6d0b,
            0xbfc0740ca53d1582,
            0xbfbfd1ee23db28da,
            0xbfbeac0cc69f14f3,
            0xbfbd760e123a481c,
            0xbfbc2f8757d9e261,
            0xbfbad80a32d1347d,
            0xbfb96f245ef20cc8,
            0xbfb7f45f8caee2d5,
            0xbfb6674132e38146,
            0xbfb4c74a5e1e40b9,
            0xbfb313f77d4124cc,
            0xbfb14cc02b4f2ff3,
            0xbfaee22dec6e0dc0,
            0xbfab00d244d0f59d,
            0xbfa6f43cd5fd886a,
            0xbfa2bb317c8a2c84,
            0xbf9ca8cfd95adb8c,
            0xbf937d16365b9896,
            0xbf83e0e275bc8492,
        ],
    ),
    (
        "broyden_tridiagonal_10",
        &[
            0xbfe2435b0f82ee6e,
            0xbfe5d15ccf0bc667,
            0xbfe6788143ed5c5b,
            0xbfe6938b074c26c0,
            0xbfe68e975ab35446,
            0xbfe672a9032e5772,
            0xbfe623f5137ee50d,
            0xbfe54e347de53ef4,
            0xbfe312b838912733,
            0xbfdaa67f98ef502a,
        ],
    ),
    (
        "broyden_tridiagonal_50",
        &[
            0xbfe243acfa34e14b,
            0xbfe5d23530c99630,
            0xbfe67ac3f695457c,
            0xbfe699afc7709b84,
            0xbfe69f597ec1d369,
            0xbfe6a062eb0ae33d,
            0xbfe6a09382be96db,
            0xbfe6a09c6821ed49,
            0xbfe6a09e09100aa7,
            0xbfe6a09e556441e7,
            0xbfe6a09e635d8eda,
            0xbfe6a09e65ec77c4,
            0xbfe6a09e66645d58,
            0xbfe6a09e667a5088,
            0xbfe6a09e667e5541,
            0xbfe6a09e667f1194,
            0xbfe6a09e667f3407,
            0xbfe6a09e667f3a44,
            0xbfe6a09e667f3b37,
            0xbfe6a09e667f3add,
            0xbfe6a09e667f395d,
            0xbfe6a09e667f352b,
            0xbfe6a09e667f29b2,
            0xbfe6a09e667f0a5c,
            0xbfe6a09e667eb4c6,
            0xbfe6a09e667dcb08,
            0xbfe6a09e667b4ca3,
            0xbfe6a09e66747d1a,
            0xbfe6a09e6661e342,
            0xbfe6a09e662f1601,
            0xbfe6a09e65a456d5,
            0xbfe6a09e64296701,
            0xbfe6a09e601e7910,
            0xbfe6a09e5513ef95,
            0xbfe6a09e36ec44a9,
            0xbfe6a09de490c704,
            0xbfe6a09d03a2d2cc,
            0xbfe6a09a9d528272,
            0xbfe6a0940f8c13fa,
            0xbfe6a082295200d4,
            0xbfe6a05146b45c5e,
            0xbfe69fcbc42f034c,
            0xbfe69e5f26ac1445,
            0xbfe69a7b79d009b4,
            0xbfe68fdd3130f65f,
            0xbfe672e4f76554d1,
            0xbfe624003473d782,
            0xbfe54e369b906627,
            0xbfe312b8a5dfc37b,
            0xbfdaa67fc7ca79da,
        ],
    ),
    (
        "broyden_tridiagonal_100",
        &[
            0xbfe243acfa34e14b,
            0xbfe5d23530c99630,
            0xbfe67ac3f695457c,
            0xbfe699afc7709b84,
            0xbfe69f597ec1d369,
            0xbfe6a062eb0ae33d,
            0xbfe6a09382be96db,
            0xbfe6a09c6821ed49,
            0xbfe6a09e09100aa7,
            0xbfe6a09e556441e7,
            0xbfe6a09e635d8eda,
            0xbfe6a09e65ec77c4,
            0xbfe6a09e66645d59,
            0xbfe6a09e667a5089,
            0xbfe6a09e667e5543,
            0xbfe6a09e667f1198,
            0xbfe6a09e667f3413,
            0xbfe6a09e667f3a62,
            0xbfe6a09e667f3b8a,
            0xbfe6a09e667f3bc0,
            0xbfe6a09e667f3bca,
            0xbfe6a09e667f3bcc,
            0xbfe6a09e667f3bcc,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcd,
            0xbfe6a09e667f3bcc,
            0xbfe6a09e667f3bcc,
            0xbfe6a09e667f3bcc,
            0xbfe6a09e667f3bcb,
            0xbfe6a09e667f3bc8,
            0xbfe6a09e667f3bc1,
            0xbfe6a09e667f3bae,
            0xbfe6a09e667f3b79,
            0xbfe6a09e667f3ae9,
            0xbfe6a09e667f395f,
            0xbfe6a09e667f352c,
            0xbfe6a09e667f29b2,
            0xbfe6a09e667f0a5c,
            0xbfe6a09e667eb4c6,
            0xbfe6a09e667dcb08,
            0xbfe6a09e667b4ca3,
            0xbfe6a09e66747d1a,
            0xbfe6a09e6661e342,
            0xbfe6a09e662f1601,
            0xbfe6a09e65a456d5,
            0xbfe6a09e64296701,
            0xbfe6a09e601e7910,
            0xbfe6a09e5513ef95,
            0xbfe6a09e36ec44a9,
            0xbfe6a09de490c704,
            0xbfe6a09d03a2d2cc,
            0xbfe6a09a9d528272,
            0xbfe6a0940f8c13fa,
            0xbfe6a082295200d4,
            0xbfe6a05146b45c5e,
            0xbfe69fcbc42f034c,
            0xbfe69e5f26ac1445,
            0xbfe69a7b79d009b4,
            0xbfe68fdd3130f65f,
            0xbfe672e4f76554d1,
            0xbfe624003473d782,
            0xbfe54e369b906627,
            0xbfe312b8a5dfc37b,
            0xbfdaa67fc7ca79da,
        ],
    ),
    (
        "broyden_banded_10",
        &[
            0xbfdb695069f8fc23,
            0xbfde808e4a000d2e,
            0xbfe0a0fe3414c56d,
            0xbfe1dbf31d825d5f,
            0xbfe2f5cf78c85b7a,
            0xbfe3fbef2568e7c5,
            0xbfe3f193e76d3cbe,
            0xbfe3e275568e7773,
            0xbfe3dac1804ac36c,
            0xbfe2c45b343b4dbc,
        ],
    ),
    (
        "broyden_banded_50",
        &[
            0xbfdb695069f8fc23,
            0xbfde808e4a000d6a,
            0xbfe0a0fe3414b53b,
            0xbfe1dbf31d85a8c7,
            0xbfe2f5cf78518f00,
            0xbfe3fbef32f1e2af,
            0xbfe3f19238b0d89f,
            0xbfe3e2ab84926ead,
            0xbfe3d3e49aa1b446,
            0xbfe3c881ee10112f,
            0xbfe3c2b5292c4fc5,
            0xbfe3c4758b1d572b,
            0xbfe3c5d6e24fc8b0,
            0xbfe3c6b84a70152b,
            0xbfe3c71f9a047f4c,
            0xbfe3c72b7a2f9e00,
            0xbfe3c7094510ee5e,
            0xbfe3c6f5696021de,
            0xbfe3c6ecb746d99e,
            0xbfe3c6eb115756a5,
            0xbfe3c6ec9d4ca9b4,
            0xbfe3c6ee8282317b,
            0xbfe3c6ef52451af6,
            0xbfe3c6ef82860102,
            0xbfe3c6ef6db370ce,
            0xbfe3c6ef4c3de761,
            0xbfe3c6ef376e75a8,
            0xbfe3c6ef31e0bdb1,
            0xbfe3c6ef32cb5e76,
            0xbfe3c6ef352d64b6,
            0xbfe3c6ef36e23a0b,
            0xbfe3c6ef3789336e,
            0xbfe3c6ef3789b3ec,
            0xbfe3c6ef375e86fc,
            0xbfe3c6ef373b3f3d,
            0xbfe3c6ef372b41f2,
            0xbfe3c6ef3728f0a4,
            0xbfe3c6ef372bcb2f,
            0xbfe3c6ef372e9e0f,
            0xbfe3c6ef373011f0,
            0xbfe3c6ef37306a87,
            0xbfe3c6ef37304327,
            0xbfe3c6ef373023ce,
            0xbfe3c6ef372cd3b5,
            0xbfe3c6ef3794f41b,
            0xbfe3c6ef2a455d56,
            0xbfe3c6f0ddb59878,
            0xbfe3c6b937b3e909,
            0xbfe3cdcf56af291f,
            0xbfe2c2cc6efb54bc,
        ],
    ),
    (
        "broyden_banded_100",
        &[
            0xbfdb695069f8fc23,
            0xbfde808e4a000d6a,
            0xbfe0a0fe3414b53b,
            0xbfe1dbf31d85a8c7,
            0xbfe2f5cf78518f00,
            0xbfe3fbef32f1e2af,
            0xbfe3f19238b0d89f,
            0xbfe3e2ab84926ead,
            0xbfe3d3e49aa1b446,
            0xbfe3c881ee10112f,
            0xbfe3c2b5292c4fc5,
            0xbfe3c4758b1d572b,
            0xbfe3c5d6e24fc8b0,
            0xbfe3c6b84a70152b,
            0xbfe3c71f9a047f4c,
            0xbfe3c72b7a2f9e00,
            0xbfe3c7094510ee5e,
            0xbfe3c6f5696021de,
            0xbfe3c6ecb746d99e,
            0xbfe3c6eb115756a5,
            0xbfe3c6ec9d4ca9b4,
            0xbfe3c6ee8282317b,
            0xbfe3c6ef52451af6,
            0xbfe3c6ef82860102,
            0xbfe3c6ef6db370ce,
            0xbfe3c6ef4c3de761,
            0xbfe3c6ef376e75a8,
            0xbfe3c6ef31e0bdb1,
            0xbfe3c6ef32cb5e76,
            0xbfe3c6ef352d64b6,
            0xbfe3c6ef36e23a0b,
            0xbfe3c6ef3789336e,
            0xbfe3c6ef3789b3ec,
            0xbfe3c6ef375e86fc,
            0xbfe3c6ef373b3f3d,
            0xbfe3c6ef372b41f2,
            0xbfe3c6ef3728f0a4,
            0xbfe3c6ef372bcb2f,
            0xbfe3c6ef372e9e0f,
            0xbfe3c6ef373011f0,
            0xbfe3c6ef37306a82,
            0xbfe3c6ef373043e5,
            0xbfe3c6ef37300ba0,
            0xbfe3c6ef372feab2,
            0xbfe3c6ef372fe07e,
            0xbfe3c6ef372fe1e1,
            0xbfe3c6ef372fe5f7,
            0xbfe3c6ef372fe8cd,
            0xbfe3c6ef372fe9dd,
            0xbfe3c6ef372fe9e6,
            0xbfe3c6ef372fe99e,
            0xbfe3c6ef372fe963,
            0xbfe3c6ef372fe948,
            0xbfe3c6ef372fe944,
            0xbfe3c6ef372fe949,
            0xbfe3c6ef372fe94d,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe94f,
            0xbfe3c6ef372fe94f,
            0xbfe3c6ef372fe94f,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe950,
            0xbfe3c6ef372fe94f,
            0xbfe3c6ef372fe955,
            0xbfe3c6ef372fe892,
            0xbfe3c6ef3730017e,
            0xbfe3c6ef372cd252,
            0xbfe3c6ef3794fcec,
            0xbfe3c6ef2a4564c4,
            0xbfe3c6f0ddb59bd0,
            0xbfe3c6b937b3e98c,
            0xbfe3cdcf56af2891,
            0xbfe2c2cc6efb5417,
        ],
    ),
];

/// SciPy 1.18 on the battery, from `data/gen_systems.py`: the problem, the multiple of the
/// standard start, whether `root(method="hybr")` (MINPACK's hybrj, with the Jacobian) succeeded
/// and in how many evaluations, whether `root(method="broyden1")` succeeded within 1,000
/// iterations, and the largest error of hybr's answer relative to the largest entry of the root.
const SCIPY: [(&str, f64, bool, usize, bool, f64); 28] = [
    ("rosenbrock", 1.0, true, 23, true, 0.0e+00),
    ("rosenbrock", 10.0, true, 14, false, 0.0e+00),
    ("powell_singular", 1.0, false, 130, true, f64::NAN),
    ("powell_singular", 10.0, false, 134, false, f64::NAN),
    ("powell_badly_scaled", 1.0, true, 169, true, 1.5e-10),
    ("powell_badly_scaled", 10.0, true, 17, false, 1.3e-13),
    ("helical_valley", 1.0, true, 17, false, 1.7e-14),
    ("helical_valley", 10.0, true, 48, false, 4.6e-13),
    ("trigonometric_10", 1.0, false, 53, true, f64::NAN),
    ("trigonometric_10", 10.0, false, 70, false, f64::NAN),
    ("discrete_boundary_10", 1.0, true, 8, true, 1.9e-14),
    ("discrete_boundary_10", 10.0, true, 11, true, 5.8e-13),
    ("discrete_boundary_50", 1.0, true, 8, false, 1.2e-14),
    ("discrete_boundary_50", 10.0, true, 10, false, 9.1e-11),
    ("discrete_boundary_100", 1.0, true, 8, false, 1.4e-14),
    ("discrete_boundary_100", 10.0, true, 10, false, 9.0e-11),
    ("broyden_tridiagonal_10", 1.0, true, 13, false, 2.1e-09),
    ("broyden_tridiagonal_10", 10.0, true, 51, false, 1.3e-10),
    ("broyden_tridiagonal_50", 1.0, true, 13, false, 3.7e-09),
    ("broyden_tridiagonal_50", 10.0, true, 27, false, 7.5e-13),
    ("broyden_tridiagonal_100", 1.0, true, 13, false, 3.9e-09),
    ("broyden_tridiagonal_100", 10.0, true, 78, false, 1.6e-16),
    ("broyden_banded_10", 1.0, true, 22, true, 3.4e-10),
    ("broyden_banded_10", 10.0, true, 27, false, 1.1e-09),
    ("broyden_banded_50", 1.0, true, 21, false, 3.8e-09),
    ("broyden_banded_50", 10.0, true, 28, false, 1.2e-10),
    ("broyden_banded_100", 1.0, true, 21, false, 3.8e-09),
    ("broyden_banded_100", 10.0, true, 27, false, 3.3e-10),
];

/// What the methods here must do on the battery: the problem, the multiple of the start, and
/// whether newton_system and broyden converge. Powell's singular system has its root at the origin
/// and a singular Jacobian there: it is approached linearly, by newton_system to residuals of
/// 10⁻⁵⁸ and below, by broyden until its approximation is singular to rounding, at residuals of
/// 10⁻³² and below; but no relative test can verify a root at zero and no residual is small in
/// units the method knows, so it is reported unconverged, as MINPACK reports it. From ten times its
/// start the trigonometric system leads both into a local minimum of the residual, where MINPACK
/// too stops.
const EXPECTED: [(&str, f64, bool, bool); 28] = [
    ("rosenbrock", 1.0, true, true),
    ("rosenbrock", 10.0, true, true),
    ("powell_singular", 1.0, false, false),
    ("powell_singular", 10.0, false, false),
    ("powell_badly_scaled", 1.0, true, true),
    ("powell_badly_scaled", 10.0, true, true),
    ("helical_valley", 1.0, true, true),
    ("helical_valley", 10.0, true, true),
    ("trigonometric_10", 1.0, true, true),
    ("trigonometric_10", 10.0, false, false),
    ("discrete_boundary_10", 1.0, true, true),
    ("discrete_boundary_10", 10.0, true, true),
    ("discrete_boundary_50", 1.0, true, true),
    ("discrete_boundary_50", 10.0, true, true),
    ("discrete_boundary_100", 1.0, true, true),
    ("discrete_boundary_100", 10.0, true, true),
    ("broyden_tridiagonal_10", 1.0, true, true),
    ("broyden_tridiagonal_10", 10.0, true, true),
    ("broyden_tridiagonal_50", 1.0, true, true),
    ("broyden_tridiagonal_50", 10.0, true, true),
    ("broyden_tridiagonal_100", 1.0, true, true),
    ("broyden_tridiagonal_100", 10.0, true, true),
    ("broyden_banded_10", 1.0, true, true),
    ("broyden_banded_10", 10.0, true, true),
    ("broyden_banded_50", 1.0, true, true),
    ("broyden_banded_50", 10.0, true, true),
    ("broyden_banded_100", 1.0, true, true),
    ("broyden_banded_100", 10.0, true, true),
];

/// The largest difference of `x` from `root`, relative to the largest entry of the root.
fn error(x: &Matrix<f64>, root: &[f64]) -> f64 {
    let big = root.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    x.as_slice()
        .iter()
        .zip(root)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
        / big
}

/// A residual or a Jacobian, as a function of a column.
type ColumnFn = Box<dyn Fn(&Matrix<f64>) -> Matrix<f64>>;

/// Brown's almost-linear function of order `n` (Moré, Garbow and Hillstrom's problem 27), and
/// its Jacobian.
fn brown(n: usize) -> (ColumnFn, ColumnFn) {
    let f = move |x: &Matrix<f64>| {
        let x = x.as_slice();
        let s: f64 = x.iter().sum();
        let mut v: Vec<f64> = x.iter().map(|&xi| xi + s - (n as f64 + 1.0)).collect();
        v[n - 1] = x.iter().product::<f64>() - 1.0;
        col(&v)
    };
    let j = move |x: &Matrix<f64>| {
        let x = x.as_slice();
        Matrix::from_fn(n, n, |i, k| {
            if i + 1 < n {
                if i == k { 2.0 } else { 1.0 }
            } else {
                (0..n).filter(|&c| c != k).map(|c| x[c]).product()
            }
        })
    };
    (Box::new(f), Box::new(j))
}

/// The circle of radius 2 about the origin and the line y = x − 1, whose intersection nearest
/// (1.5, 0.5) is ((1 + √7)/2, (√7 − 1)/2).
fn circle_and_line(x: &Matrix<f64>) -> Matrix<f64> {
    col(&[
        x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0,
        x[(0, 0)] - x[(1, 0)] - 1.0,
    ])
}

/// Its Jacobian.
fn circle_and_line_jacobian(x: &Matrix<f64>) -> Matrix<f64> {
    Matrix::from_rows(&[[2.0 * x[(0, 0)], 2.0 * x[(1, 0)]], [1.0, -1.0]])
}

/// Its root, ((1 + √7)/2, (√7 − 1)/2) rounded.
const ROOT: [f64; 2] = [1.822_875_655_532_295_4, 0.822_875_655_532_295_2];

/// The tolerance of the defaults' step test, 4ε.
const XTOL: f64 = 4.0 * f64::EPSILON;

// ---------------------------------------------------------------------------------------------
// The battery
// ---------------------------------------------------------------------------------------------

#[test]
fn the_more_garbow_hillstrom_systems_are_solved_from_their_starts_and_ten_times_them() {
    // Both methods converge on 25 of the 28 problems and starts (MINPACK's hybrj on 24, SciPy's
    // broyden1 on 7), each to the root its start leads to; the three others are reported with the
    // best point reached and converged false. Where hybrj fails and these do not is the
    // trigonometric system from its start, where hybrj stops at a local minimum of the residual.
    let problems = mgh();
    let (mut ours, mut hybr) = (0, 0);
    for (
        &(name, scale, newton_converges, broyden_converges),
        &(scipy_name, scipy_scale, hybr_ok, _, _, _),
    ) in EXPECTED.iter().zip(&SCIPY)
    {
        assert_eq!((name, scale), (scipy_name, scipy_scale));
        if cfg!(miri) && (name.ends_with("_50") || name.ends_with("_100")) {
            continue;
        }
        let p = problems.iter().find(|p| p.name == name).unwrap();
        let root: Vec<f64> = ROOTS
            .iter()
            .find(|r| r.0 == name)
            .unwrap()
            .1
            .iter()
            .map(|&b| f64::from_bits(b))
            .collect();
        let x0 = col(&p.x0.iter().map(|v| v * scale).collect::<Vec<_>>());
        let rn = newton_system(&x0, |x| p.residual(x), |x| p.jacobian(x)).unwrap();
        let rb = broyden(&x0, |x| p.residual(x)).unwrap();
        for (method, r, want) in [
            ("newton", &rn, newton_converges),
            ("broyden", &rb, broyden_converges),
        ] {
            assert_eq!(
                r.converged, want,
                "{method} on {name} ×{scale}: {:?} at {:e}",
                r.stop, r.residual
            );
            if r.converged {
                // Within 64ε of the largest entry of the root; Powell's badly scaled system within
                // 2·10⁻¹³, since its second unknown is fixed by e^(−x₂) ≈ 1.1·10⁻⁴ against terms of
                // size 1: a residual of one rounding moves it by 10⁴ roundings.
                let e = error(&r.x, &root);
                let bound = if name == "powell_badly_scaled" {
                    2e-13
                } else {
                    64.0 * f64::EPSILON
                };
                assert!(e <= bound, "{method} on {name} ×{scale}: error {e:e}");
                assert!(matches!(
                    r.stop,
                    SystemStop::Residual | SystemStop::Step | SystemStop::Rounding
                ));
                ours += 1;
            } else if name == "powell_singular" && method == "broyden" {
                // Powell's singular system: within about 10⁻¹⁷ of its root, where the Jacobian is
                // singular, the approximation's condition passes 1/ε, and its QR factors lose the
                // rows that vanish at the root (the quadratic equations) to the rounding of the
                // others. The steps are then rounding, and which failure ends the run depends on
                // how it falls, so on the order of the sums of the platform's dense kernels: a
                // restart whose R has a zero on its diagonal (Singular, on x86_64), a step along
                // which no descent is found (LineSearch), or the last iteration (Iterations, both
                // on aarch64). It is a failure reported with the best point either way: one within
                // 10⁻¹⁵ of the root, where the residual is below 10⁻³⁰ (1.8·10⁻¹⁷ to 4.3·10⁻¹⁷,
                // and 3·10⁻³⁴ to 3·10⁻³², measured on the two).
                assert!(
                    matches!(
                        r.stop,
                        SystemStop::Iterations | SystemStop::LineSearch | SystemStop::Singular
                    ),
                    "{name} ×{scale}: {:?}",
                    r.stop
                );
                assert!(
                    r.residual <= 1e-30 && r.x.as_slice().iter().all(|v| v.abs() <= 1e-15),
                    "{name} ×{scale}: {:?} at {:e}",
                    r.x.as_slice(),
                    r.residual
                );
            } else {
                // The trigonometric system from ten times its start, at a local minimum of the
                // residual; and newton_system on Powell's singular system, where the LU of the
                // exact Jacobian keeps the small pivots accurate (they are formed without
                // cancellation) and the error goes on halving to the last iteration.
                assert!(
                    matches!(r.stop, SystemStop::Iterations | SystemStop::LineSearch),
                    "{method} on {name} ×{scale}: {:?}",
                    r.stop
                );
            }
            // What is reported is what f gives at the point reported.
            assert_eq!(r.f.as_slice(), p.residual(&r.x).as_slice());
        }
        hybr += usize::from(hybr_ok);
    }
    if !cfg!(miri) {
        assert_eq!((ours, hybr), (50, 24));
    }
}

#[test]
fn powells_singular_system_is_reported_unverified_but_solved_with_an_ftol() {
    // The root is the origin, where the Jacobian is singular: Newton halves the error a step, and
    // after 100 steps the residual is 8·10⁻⁶⁰ — but at a root at zero no relative step test can
    // pass, and no residual is small in units the method knows, so it says so.
    let p = &mgh()[1];
    let x0 = col(&p.x0);
    let r = newton_system(&x0, |x| p.residual(x), |x| p.jacobian(x)).unwrap();
    assert!(!r.converged && r.stop == SystemStop::Iterations && r.iterations == 100);
    assert!(r.residual < 1e-50, "{:e}", r.residual);
    assert!(r.x.as_slice().iter().all(|v| v.abs() < 1e-25));
    // Given the units, with an ftol, both methods converge, linearly.
    let opts = SystemOptions::new().with_ftol(1e-20);
    for r in [
        newton_system_with(&x0, |x| p.residual(x), |x| p.jacobian(x), &opts).unwrap(),
        broyden_with(&x0, |x| p.residual(x), &opts).unwrap(),
    ] {
        assert!(
            r.converged && r.stop == SystemStop::Residual && r.residual <= 1e-20,
            "{:?}",
            r.stop
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The defects of the audit
// ---------------------------------------------------------------------------------------------

#[test]
fn newton_system_does_not_stop_on_the_step_alone() {
    // n_roots-4: the old step test, |δ|∞ ≤ xtol·max(|x|∞, 1), returned without a look at the
    // residual: beside an unknown of 10¹⁶ a step of 2.2 in another counted as converged, and for
    // unknowns below 10⁻¹⁰ the test was absolute.
    let f = |x: &Matrix<f64>| col(&[x[(0, 0)] - 1e16, x[(1, 0)].powi(3) - 2.0 * x[(1, 0)] - 5.0]);
    let j = |x: &Matrix<f64>| {
        Matrix::from_rows(&[[1.0, 0.0], [0.0, 3.0 * x[(1, 0)] * x[(1, 0)] - 2.0]])
    };
    let r = newton_system(&col(&[0.0, 1.0]), f, j).unwrap();
    assert!(r.converged);
    assert_eq!(r.x[(0, 0)], 1e16);
    assert!(
        (r.x[(1, 0)] - 2.094_551_481_542_326_5).abs() <= XTOL * 2.1,
        "{}",
        r.x[(1, 0)]
    );
    // (10¹⁴q)² = 2 and (10¹⁹q)² = 2: roots of 1.4·10⁻¹⁴ and 1.4·10⁻¹⁹, to full relative accuracy.
    for (s, q0) in [(1e14, 1e-14), (1e19, 1e-19)] {
        let g = move |x: &Matrix<f64>| col(&[(x[(0, 0)] * s).powi(2) - 2.0]);
        let gj = move |x: &Matrix<f64>| col(&[2.0 * s * s * x[(0, 0)]]);
        let want = core::f64::consts::SQRT_2 / s;
        let r = newton_system(&col(&[q0]), g, gj).unwrap();
        assert!(
            r.converged && (r.x[(0, 0)] - want).abs() <= 2.0 * XTOL * want,
            "{:e}",
            r.x[(0, 0)]
        );
        let r = broyden(&col(&[q0]), g).unwrap();
        assert!(
            r.converged && (r.x[(0, 0)] - want).abs() <= 2.0 * XTOL * want,
            "{:e}",
            r.x[(0, 0)]
        );
    }
    // √x − 2 from 0, where the Jacobian 1/(2√x) is infinite: reported, not taken for a zero step.
    let h = |x: &Matrix<f64>| col(&[x[(0, 0)].sqrt() - 2.0]);
    let hj = |x: &Matrix<f64>| col(&[0.5 / x[(0, 0)].sqrt()]);
    let r = newton_system(&col(&[0.0]), h, hj).unwrap();
    assert!(!r.converged && r.stop == SystemStop::Singular && r.x[(0, 0)] == 0.0);
    // Broyden's differences need no derivative there and find the root, 4.
    let r = broyden(&col(&[0.0]), h).unwrap();
    assert!(r.converged && (r.x[(0, 0)] - 4.0).abs() <= XTOL * 4.0);
}

#[test]
fn a_residual_that_is_not_finite_at_the_start_is_no_start() {
    // n_roots-5: the infinity norm passed over NaNs, so a start outside the domain of f, or one
    // that was itself NaN, came back as the solution.
    let f = |x: &Matrix<f64>| col(&[(x[(0, 0)] - 5.0).sqrt(), (x[(1, 0)] - 5.0).ln()]);
    let x0 = col(&[0.0, 0.0]);
    assert!(newton_system(&x0, f, |_: &Matrix<f64>| Matrix::<f64>::identity(2)).is_none());
    assert!(broyden(&x0, f).is_none());
    let x0 = col(&[f64::NAN, 0.5]);
    assert!(newton_system(&x0, circle_and_line, circle_and_line_jacobian).is_none());
    assert!(broyden(&x0, circle_and_line).is_none());
    // One NaN beside a tiny value is no better.
    let g = |x: &Matrix<f64>| col(&[if x[(0, 0)] < 1.0 { f64::NAN } else { 0.0 }, 1e-17]);
    assert!(broyden(&col(&[0.0, 0.0]), g).is_none());
}

#[test]
fn broyden_keeps_updating_its_jacobian_on_browns_function() {
    // n_roots-6: the update was skipped unless |δ'H·Δf| exceeded ε·max(|δ|, 1), an absolute bound
    // on a quantity of the size of |δ|²: on Brown's function of order 5 it froze after two steps
    // and the residual stayed at 2.66·10⁻² for 10,000 iterations.
    let (f, _) = brown(5);
    let r = broyden(&col(&[0.5; 5]), &f).unwrap();
    assert!(r.converged, "{:?} at {:e}", r.stop, r.residual);
    assert!(r.iterations <= 15, "{} iterations", r.iterations);
    assert!(r.residual <= 1e-15);
    // Brown's function has the root (1, …, 1), and that is the one found.
    assert!(
        r.x.as_slice().iter().all(|v| (v - 1.0).abs() <= 8.0 * XTOL),
        "{:?}",
        r.x.as_slice()
    );
}

#[test]
fn broyden_accepts_a_root_whose_residual_is_down_to_rounding() {
    // n_roots-7: the old stopping rule wanted |f|∞ ≤ 8.9·10⁻¹⁶ absolute, which rounding forbids
    // for 21 of the sizes 74 to 100 of Broyden's banded function and for the circle of radius 2000.
    let sizes: Vec<usize> = if cfg!(miri) {
        vec![10]
    } else {
        (74..=100).collect()
    };
    for n in sizes {
        let f = move |x: &Matrix<f64>| col(&banded(x.as_slice(), n));
        let r = broyden(&col(&vec![-1.0; n]), f).unwrap();
        assert!(r.converged, "n = {n}: {:?} at {:e}", r.stop, r.residual);
        assert!(r.residual <= 2e-15);
    }
    let big = |x: &Matrix<f64>| {
        col(&[
            x[(0, 0)].powi(2) + x[(1, 0)].powi(2) - 4e6,
            x[(0, 0)] - x[(1, 0)] - 1000.0,
        ])
    };
    let r = broyden(&col(&[1500.0, 500.0]), big).unwrap();
    assert!(r.converged);
    assert!((r.x[(0, 0)] - 1000.0 * ROOT[0]).abs() <= 2.0 * XTOL * 1823.0);
    assert!((r.x[(1, 0)] - 1000.0 * ROOT[1]).abs() <= 2.0 * XTOL * 1823.0);
}

#[test]
fn the_tests_do_not_depend_on_the_units_of_the_residual() {
    // n_roots-14: the default residual tolerance was ε absolute, so 10⁻²⁰(x² − 2) = 0 was solved
    // by its start, and 10⁻¹²(x² − 2) = 0 to six digits. Every test here is free of units.
    for s in [1e-20, 1e-12, 1.0, 1e12, 1e20] {
        let f = move |x: &Matrix<f64>| col(&[s * (x[(0, 0)] * x[(0, 0)] - 2.0)]);
        let j = move |x: &Matrix<f64>| col(&[2.0 * s * x[(0, 0)]]);
        for r in [
            newton_system(&col(&[1.0]), f, j).unwrap(),
            broyden(&col(&[1.0]), f).unwrap(),
        ] {
            assert!(r.converged, "scale {s:e}");
            assert!(
                (r.x[(0, 0)] - core::f64::consts::SQRT_2).abs() <= XTOL * 2.0,
                "scale {s:e}: {}",
                r.x[(0, 0)]
            );
        }
    }
    // The circle and the line in units of 10³ and 10⁻⁶ of the residual, and of 10⁻⁹ of the
    // unknowns: the same root, scaled.
    for s in [1e3, 1e-6] {
        let f = move |x: &Matrix<f64>| {
            let r = circle_and_line(x);
            col(&[s * r[(0, 0)], s * r[(1, 0)]])
        };
        let r = broyden(&col(&[1.5, 0.5]), f).unwrap();
        assert!(r.converged && (r.x[(0, 0)] - ROOT[0]).abs() <= XTOL * 2.0);
    }
    let small = |x: &Matrix<f64>| circle_and_line(&col(&[x[(0, 0)] * 1e9, x[(1, 0)] * 1e9]));
    let small_j = |x: &Matrix<f64>| {
        let j = circle_and_line_jacobian(&col(&[x[(0, 0)] * 1e9, x[(1, 0)] * 1e9]));
        Matrix::from_fn(2, 2, |i, k| j[(i, k)] * 1e9)
    };
    let r = newton_system(&col(&[1.5e-9, 0.5e-9]), small, small_j).unwrap();
    assert!(
        r.converged && (r.x[(0, 0)] - ROOT[0] * 1e-9).abs() <= 2.0 * XTOL * 1.83e-9,
        "{:e}",
        r.x[(0, 0)]
    );
}

// ---------------------------------------------------------------------------------------------
// The residual test, which the size of the terms of an equation must not fool (review, round 1)
// ---------------------------------------------------------------------------------------------

/// The variably dimensioned function of Moré, Garbow and Hillstrom (problem 25) of order `n`,
/// `xₖ − 1 + k·s·(1 + 2s²)` with `s = Σ j·(xⱼ − 1)`, whose only root is `(1, …, 1)`.
fn variably_dimensioned(n: usize) -> impl Fn(&Matrix<f64>) -> Matrix<f64> + Copy {
    move |x: &Matrix<f64>| {
        let s: f64 = (0..n).map(|j| (j + 1) as f64 * (x[(j, 0)] - 1.0)).sum();
        let t = s * (1.0 + 2.0 * s * s);
        Matrix::from_fn(n, 1, |k, _| x[(k, 0)] - 1.0 + (k + 1) as f64 * t)
    }
}

/// Its standard starting point, `xⱼ = 1 − j/n`, times `scale`.
fn variably_dimensioned_start(n: usize, scale: f64) -> Matrix<f64> {
    Matrix::from_fn(n, 1, |j, _| scale * (1.0 - (j + 1) as f64 / n as f64))
}

/// Whether every entry of `x` is within `tol` of `root` relative to it.
fn near(x: &Matrix<f64>, root: &[f64], tol: f64) -> bool {
    x.as_slice()
        .iter()
        .zip(root)
        .all(|(&v, &r)| (v - r).abs() <= tol * r.abs())
}

#[test]
fn a_residual_bound_that_overflows_is_no_proof_of_a_root() {
    // The bound 8ε·(|J|·|x|)ᵢ of the residual test overflowed to infinity, which every finite
    // residual is below. 10¹⁰·t·(1 + (t/10²⁹⁷)²), t = x − 10²⁹⁹, has its only root at 10²⁹⁹; from
    // 1.01·10²⁹⁹ both methods stopped "converged" at 1.005·10²⁹⁹, where the residual is 6·10³⁰⁶.
    let g = |v: &Matrix<f64>| {
        let t = v[(0, 0)] - 1e299;
        col(&[1e10 * t * (1.0 + (t / 1e297) * (t / 1e297))])
    };
    let gj = |v: &Matrix<f64>| {
        let t = v[(0, 0)] - 1e299;
        col(&[1e10 * (1.0 + 3.0 * (t / 1e297) * (t / 1e297))])
    };
    let x0 = col(&[1.01e299]);
    for line_search in [true, false] {
        let opts = SystemOptions::new().with_line_search(line_search);
        for r in [
            newton_system_with(&x0, g, gj, &opts).unwrap(),
            broyden_with(&x0, g, &opts).unwrap(),
        ] {
            assert!(
                r.converged && r.x[(0, 0)] == 1e299,
                "{:?} at {:e}, residual {:e}",
                r.stop,
                r.x[(0, 0)],
                r.residual
            );
        }
    }
    // The variably dimensioned function of order 30 from a hundred times its standard start, by
    // Broyden's method without the line search: it stopped "converged" at |x|∞ = 8·10¹⁰⁵ with a
    // residual of 8·10³⁰⁴. Now it either reaches the root or says that it did not.
    let n = if cfg!(miri) { 10 } else { 30 };
    let r = broyden_with(
        &variably_dimensioned_start(n, 100.0),
        variably_dimensioned(n),
        &SystemOptions::new().with_line_search(false),
    )
    .unwrap();
    assert!(
        !r.converged || near(&r.x, &vec![1.0; n], 64.0 * f64::EPSILON),
        "{:?} at {:e}",
        r.stop,
        r.residual
    );
}

#[test]
fn the_residual_test_is_not_fooled_by_a_coupling_that_f_evaluates_exactly() {
    // (|J|·|x|)ᵢ is the size of the linear terms of an equation, not the rounding of fᵢ. With
    // x₁ = 10¹⁶ coupled into the equation of x₂ by 0.01·(x₁ − 10¹⁶), which f evaluates exactly,
    // the test allowed a residual of 0.18 there, and newton_system stopped with x₂ off by 3·10⁻⁶
    // on x₂³ − 2x₂ − 5 and by 6·10⁻³ on (x₂ − 2)(1 + 100(x₂ − 2)).
    let cubic = |v: &Matrix<f64>| {
        let x2 = v[(1, 0)];
        col(&[
            v[(0, 0)] - 1e16,
            0.01 * (v[(0, 0)] - 1e16) + x2 * x2 * x2 - 2.0 * x2 - 5.0,
        ])
    };
    let cubic_j = |v: &Matrix<f64>| {
        Matrix::from_rows(&[[1.0, 0.0], [0.01, 3.0 * v[(1, 0)] * v[(1, 0)] - 2.0]])
    };
    let root = 2.094_551_481_542_326_5;
    let x0 = col(&[0.0, 1.0]);
    for (method, r) in [
        ("newton", newton_system(&x0, cubic, cubic_j).unwrap()),
        ("broyden", broyden(&x0, cubic).unwrap()),
    ] {
        assert!(r.converged, "{method}: {:?}", r.stop);
        assert!(
            near(&r.x, &[1e16, root], 2.0 * f64::EPSILON),
            "{method}: {:?}",
            r.x.as_slice()
        );
    }
    // The quadratic has the roots 2 and 1.99.
    let curved = |v: &Matrix<f64>| {
        let d = v[(1, 0)] - 2.0;
        col(&[
            v[(0, 0)] - 1e16,
            0.01 * (v[(0, 0)] - 1e16) + d * (1.0 + 100.0 * d),
        ])
    };
    let curved_j =
        |v: &Matrix<f64>| Matrix::from_rows(&[[1.0, 0.0], [0.01, 1.0 + 200.0 * (v[(1, 0)] - 2.0)]]);
    let x0 = col(&[0.0, 3.0]);
    for (method, r) in [
        ("newton", newton_system(&x0, curved, curved_j).unwrap()),
        ("broyden", broyden(&x0, curved).unwrap()),
    ] {
        assert!(r.converged, "{method}: {:?}", r.stop);
        assert!(
            near(&r.x, &[1e16, 2.0], 2.0 * f64::EPSILON)
                || near(&r.x, &[1e16, 1.99], 2.0 * f64::EPSILON),
            "{method}: {:?}",
            r.x.as_slice()
        );
    }
}

#[test]
fn a_short_correction_near_a_nearly_double_root_is_no_proof_of_a_root() {
    // (x₂ − 2)(1 + 10⁹(x₂ − 2)), coupled to x₁ = 10¹⁶ as above, has the roots 2 and 2 − 10⁻⁹.
    // Above them Newton's method only halves the error each step, and a correction below √ε of
    // x₂ still leaves an error of the same size: a test that took one more step after a short
    // correction and stopped would stop 10⁻⁸ from a root. Broyden's first Jacobian by forward
    // differences is wrong there in its sign (the step of √ε adds 30 to a slope of −1), and tests
    // resting on it stopped 7.8·10⁻¹¹ from a root.
    let k = 1e9;
    let near_double = move |v: &Matrix<f64>| {
        let d = v[(1, 0)] - 2.0;
        col(&[
            v[(0, 0)] - 1e16,
            0.01 * (v[(0, 0)] - 1e16) + d * (1.0 + k * d),
        ])
    };
    let near_double_j = move |v: &Matrix<f64>| {
        Matrix::from_rows(&[[1.0, 0.0], [0.01, 1.0 + 2.0 * k * (v[(1, 0)] - 2.0)]])
    };
    for x2 in [3.0, 2.1, 2.000_000_03, 2.000_001] {
        let x0 = col(&[0.0, x2]);
        for (method, r) in [
            (
                "newton",
                newton_system(&x0, near_double, near_double_j).unwrap(),
            ),
            ("broyden", broyden(&x0, near_double).unwrap()),
        ] {
            assert!(
                !r.converged
                    || near(&r.x, &[1e16, 2.0], 4.0 * f64::EPSILON)
                    || near(&r.x, &[1e16, 2.0 - 1e-9], 4.0 * f64::EPSILON),
                "{method} from {x2}: {:?} at {:?}",
                r.stop,
                r.x.as_slice()
            );
            if method == "newton" {
                assert!(r.converged, "from {x2}: {:?}", r.stop);
            }
        }
    }
}

#[test]
fn broyden_tests_no_point_on_its_first_jacobian_by_forward_differences() {
    // From a start 10⁻¹² above the root 2 − 10⁻⁹ of the nearly double root above, the residual
    // is down to rounding by the coupling's bound at once, and the correction on the first
    // Jacobian by forward differences, whose slope is 29 for −1, is short and points the wrong
    // way: a test resting on that Jacobian would stop there, 10⁻¹² from the root. A restart by
    // central differences comes first.
    let k = 1e9;
    let near_double = move |v: &Matrix<f64>| {
        let d = v[(1, 0)] - 2.0;
        col(&[
            v[(0, 0)] - 1e16,
            0.01 * (v[(0, 0)] - 1e16) + d * (1.0 + k * d),
        ])
    };
    let root = 2.0 - 1e-9;
    let r = broyden(&col(&[1e16, root + 1e-12]), near_double).unwrap();
    assert!(
        r.converged && near(&r.x, &[1e16, root], 4.0 * f64::EPSILON),
        "{:?} at {:?}",
        r.stop,
        r.x.as_slice()
    );
}

#[test]
fn the_line_search_keeps_long_corrections_where_the_residual_bound_is_loose() {
    // A coupling of 10³·(x₁ − 10¹⁶) makes the bound of the residual test 1.8·10⁴ in the second
    // equation, which every value of arctan is below; from x₂ = 5 the full Newton steps on
    // arctan(x₂ − 2) diverge. The residual test takes its confirming step only after a short
    // correction, so the long ones stay with the line search, and both methods converge.
    let f = |v: &Matrix<f64>| {
        col(&[
            v[(0, 0)] - 1e16,
            1e3 * (v[(0, 0)] - 1e16) + (v[(1, 0)] - 2.0).atan(),
        ])
    };
    let j = |v: &Matrix<f64>| {
        Matrix::from_rows(&[[1.0, 0.0], [1e3, 1.0 / (1.0 + (v[(1, 0)] - 2.0).powi(2))]])
    };
    let x0 = col(&[1e16, 5.0]);
    for (method, r) in [
        ("newton", newton_system(&x0, f, j).unwrap()),
        ("broyden", broyden(&x0, f).unwrap()),
    ] {
        assert!(
            r.converged && near(&r.x, &[1e16, 2.0], 2.0 * f64::EPSILON),
            "{method}: {:?} at {:?}",
            r.stop,
            r.x.as_slice()
        );
    }
}

#[test]
fn an_unknown_that_tends_to_zero_is_differenced_with_an_absolute_step() {
    // The helical valley's root is (1, 0, 0). From a hundred times its start Broyden's method
    // takes the second and third unknowns into the subnormals, where a step of √ε of them is no
    // float: the difference there is taken with the absolute step instead, as at zero, and the
    // residual vanishes. (Before, a restart there could form no column and ended Singular.)
    let p = mgh()
        .into_iter()
        .find(|p| p.name == "helical_valley")
        .unwrap();
    let x0 = col(&p.x0.iter().map(|v| v * 100.0).collect::<Vec<_>>());
    let r = broyden(&x0, |x| p.residual(x)).unwrap();
    assert!(r.converged, "{:?} at {:e}", r.stop, r.residual);
    assert!(
        (r.x[(0, 0)] - 1.0).abs() <= XTOL
            && r.x[(1, 0)].abs() < 1e-300
            && r.x[(2, 0)].abs() < 1e-300,
        "{:?}",
        r.x.as_slice()
    );

    // The same at the start: x₂ = 10⁻³¹⁰, whose step of √ε, 1.5·10⁻³¹⁸, is a float but changes
    // nothing in x₂ − 1, nor do the steps 2²⁰, 2⁴⁰ and 2⁶⁰ times longer; the difference would be
    // zero and the Jacobian singular.
    let lines = |v: &Matrix<f64>| col(&[v[(0, 0)] - 1.0, v[(1, 0)] - 1.0]);
    let r = broyden(&col(&[0.5, 1e-310]), lines).unwrap();
    assert!(
        r.converged && r.x.as_slice() == [1.0, 1.0],
        "{:?} at {:?}",
        r.stop,
        r.x.as_slice()
    );
}

#[test]
fn an_approximation_that_stops_improving_the_iterate_is_renewed() {
    // Brown's function of order 5 from a start of the review's battery, without the line search:
    // Broyden's updates stall with the residual at rounding by the approximation but the
    // correction long, and only a restart, made when the last step did not halve the residual,
    // gets the iteration going again. (Restarting only on a short correction, it ran its 100
    // steps on its first Jacobian.)
    let (f, _) = brown(5);
    let x0 = col(&[
        0.560_791_524_992_124_1,
        0.465_181_549_197_682_23,
        0.557_116_100_174_550_8,
        0.458_257_215_729_459_65,
        0.624_106_401_160_328_1,
    ]);
    let r = broyden_with(&x0, &f, &SystemOptions::new().with_line_search(false)).unwrap();
    assert!(
        r.converged && r.jacobians >= 2,
        "{:?}, {} Jacobians",
        r.stop,
        r.jacobians
    );
    assert!(
        r.x.as_slice().iter().all(|v| (v - 1.0).abs() <= 8.0 * XTOL),
        "{:?}",
        r.x.as_slice()
    );
}

#[test]
fn an_f32_residual_whose_terms_cancel_is_not_taken_for_rounding() {
    // In f32, Broyden's method on the variably dimensioned function of order 10 from a hundred
    // times its start stopped "converged" after 3 steps at |x|∞ = 6.4·10⁶ with a residual of
    // 3·10⁹: s = Σ j·(xⱼ − 1) had cancelled to −538 from terms of 10⁷, and 8ε·(|B|·|x|)ᵢ admitted
    // residuals up to 8·10⁹. The root is (1, …, 1).
    let n = 10;
    let f = move |x: &Matrix<f32>| {
        let s: f32 = (0..n).map(|j| (j + 1) as f32 * (x[(j, 0)] - 1.0)).sum();
        let t = s * (1.0 + 2.0 * s * s);
        Matrix::from_fn(n, 1, |k, _| x[(k, 0)] - 1.0 + (k + 1) as f32 * t)
    };
    let jac = move |x: &Matrix<f32>| {
        let s: f32 = (0..n).map(|j| (j + 1) as f32 * (x[(j, 0)] - 1.0)).sum();
        let dt = 1.0 + 6.0 * s * s;
        Matrix::from_fn(n, n, |k, j| {
            (k + 1) as f32 * dt * (j + 1) as f32 + if k == j { 1.0 } else { 0.0 }
        })
    };
    let x0 = Matrix::from_fn(n, 1, |j, _| 100.0 * (1.0 - (j + 1) as f32 / n as f32));
    let at_root = |r: &SystemResult<f32>| {
        r.x.as_slice()
            .iter()
            .all(|&v| (v - 1.0).abs() <= 16.0 * f32::EPSILON)
    };
    let b = broyden(&x0, f).unwrap();
    assert!(
        !b.converged || at_root(&b),
        "{:?} at {:e}, x {:?}",
        b.stop,
        b.residual,
        b.x.as_slice()
    );
    let r = newton_system(&x0, f, jac).unwrap();
    assert!(
        !r.converged || at_root(&r),
        "{:?} at {:e}",
        r.stop,
        r.residual
    );
}

#[test]
fn a_kink_above_the_rounding_is_no_root() {
    // |x − 1| + c has no root: its least value, c, is at the kink x = 1, where the line search
    // can lower the residual no further along any correction. That is the rounding test's
    // situation too, but the correction there, c relative to x, is above ε^(2/3) = 3.7·10⁻¹¹
    // for these c, so no test of a root holds and Newton's method reports the failed line
    // search, with the kink as the best point. (For c of 10⁻¹² and below the rounding test is
    // tried, and fails: below.) Broyden's central difference at the kink is zero, which it
    // reports as a singular Jacobian when its line search fails.
    for c in [1e-9, 1e-6] {
        let f = move |v: &Matrix<f64>| col(&[(v[(0, 0)] - 1.0).abs() + c]);
        let j = |v: &Matrix<f64>| col(&[if v[(0, 0)] >= 1.0 { 1.0 } else { -1.0 }]);
        let r = newton_system(&col(&[2.0]), f, j).unwrap();
        assert!(
            !r.converged && r.stop == SystemStop::LineSearch,
            "c = {c:e}: {:?}",
            r.stop
        );
        let b = broyden(&col(&[2.0]), f).unwrap();
        assert!(
            !b.converged && matches!(b.stop, SystemStop::LineSearch | SystemStop::Singular),
            "c = {c:e}: {:?}",
            b.stop
        );
        for r in [&r, &b] {
            assert!(r.residual >= c && r.residual <= 3.0 * c, "{:e}", r.residual);
        }
    }
    // For c of 10⁻¹² and below the correction at the kink is below ε^(2/3) and the rounding test
    // is tried. f does not follow its linear model there: it rises on both sides of the kink, so
    // at x + k·δ it is (k + 1)·c where the model says (1 − k)·c. The kink is reported with
    // converged false; the test that took a failed line search alone for rounding stopped on it,
    // converged.
    for c in [1e-12, 1e-15] {
        let f = move |v: &Matrix<f64>| col(&[(v[(0, 0)] - 1.0).abs() + c]);
        let j = |v: &Matrix<f64>| col(&[if v[(0, 0)] >= 1.0 { 1.0 } else { -1.0 }]);
        let r = newton_system(&col(&[2.0]), f, j).unwrap();
        assert!(
            !r.converged && r.stop == SystemStop::Rounding && r.x[(0, 0)] == 1.0,
            "c = {c:e}: {} {:?} at {}",
            r.converged,
            r.stop,
            r.x[(0, 0)]
        );
        assert_eq!(r.residual, c);
        let b = broyden(&col(&[2.0]), f).unwrap();
        assert!(!b.converged, "c = {c:e}: {:?} at {}", b.stop, b.x[(0, 0)]);
    }
}

// ---------------------------------------------------------------------------------------------
// The differences of a restart, and the confirming step (review round 2)
// ---------------------------------------------------------------------------------------------

#[test]
fn broydens_restarts_find_the_scale_of_an_unknown_with_a_large_offset() {
    // x = T + u, with f varying on a scale of one in u and evaluated exactly (x − T is exact near
    // T). The difference step √ε·|x|, 15 at T = 10⁹, spans the features of f: the central
    // quotient of e^u there is 10⁵ for a slope of 1, the correction 10⁵ times too short, and
    // broyden stopped "converged" at u = 0, where f = −1 (Rounding; Step at T = 10¹⁰). The
    // restart now quarters its step until the quotients at two steps agree.
    for t in [1e9, 1e10] {
        let f = move |v: &Matrix<f64>| col(&[(v[(0, 0)] - t).exp() - 2.0]);
        let r = broyden(&col(&[t]), f).unwrap();
        assert!(
            r.converged && near(&r.x, &[t + core::f64::consts::LN_2], f64::EPSILON),
            "T = {t:e}: {:?} at u = {}, f = {:e}",
            r.stop,
            r.x[(0, 0)] - t,
            r.residual
        );
    }
    // (x − T)³ − 1 from T + 3, whose root T + 1 is a float: it stopped at u = 3, where f = 26.
    for t in [1e11, 1e12] {
        let f = move |v: &Matrix<f64>| {
            let u = v[(0, 0)] - t;
            col(&[u * u * u - 1.0])
        };
        let r = broyden(&col(&[t + 3.0]), f).unwrap();
        assert!(
            r.converged && r.x[(0, 0)] == t + 1.0,
            "T = {t:e}: {:?} at u = {}",
            r.stop,
            r.x[(0, 0)] - t
        );
    }
    // u³ − 2u + 2 from u = 0, where Newton's iteration cycles between 0 and 1 and the only real
    // root is −1.769…: it stopped "converged" at u = 0, where f = 2, for T from 10⁹ to 10¹².
    let root = -1.769_292_354_238_631_4;
    for t in [1e9, 1e10, 1e11, 1e12] {
        let f = move |v: &Matrix<f64>| {
            let u = v[(0, 0)] - t;
            col(&[u * u * u - 2.0 * u + 2.0])
        };
        let r = broyden(&col(&[t]), f).unwrap();
        assert!(
            !r.converged || near(&r.x, &[t + root], XTOL),
            "T = {t:e}: {:?} at u = {}, f = {:e}",
            r.stop,
            r.x[(0, 0)] - t,
            r.residual
        );
    }
    // In f32 at T = 10⁵ (the step is 35): it stopped at u = 2.98, where f = 25.4.
    let f = |v: &Matrix<f32>| {
        let u = v[(0, 0)] - 1e5;
        col(&[u * u * u - 1.0])
    };
    let r = broyden(&col(&[1e5f32 + 3.0]), f).unwrap();
    assert!(
        r.converged && r.x[(0, 0)] == 1e5 + 1.0,
        "{:?} at u = {}",
        r.stop,
        r.x[(0, 0)] - 1e5
    );
}

#[test]
fn broyden_does_not_stop_beside_a_root_of_multiplicity_three_or_more() {
    // Within about h/100 of a root of multiplicity three or more the third derivative dominates
    // the central quotient: (x − 1)³ at 1 + 10⁻¹¹ has the quotient h² = 2·10⁻¹⁶ for a slope of
    // 3·10⁻²², and broyden stopped there after one step (Step), 10⁻¹¹ from the root; (x − 1)⁷ from
    // 1 + 10⁻⁹, 10⁻⁹ from it. f is exact near 1, so the root is fixed to the last bit. At a root of
    // multiplicity m an answer may lie up to m/(1 − 1/m)^(m−1)·xtol from it (SystemStop): 6.75 at
    // m = 3, 17.6 at 7.
    let bound = |m: i32| {
        let m = f64::from(m);
        m / (1.0 - 1.0 / m).powf(m - 1.0)
    };
    let cube = |v: &Matrix<f64>| col(&[(v[(0, 0)] - 1.0).powi(3)]);
    let r = broyden(&col(&[1.0 + 1e-11]), cube).unwrap();
    assert!(
        r.converged && (r.x[(0, 0)] - 1.0).abs() <= bound(3) * XTOL,
        "{:?} at {:e}",
        r.stop,
        r.x[(0, 0)] - 1.0
    );
    let seventh = |v: &Matrix<f64>| col(&[(v[(0, 0)] - 1.0).powi(7)]);
    let r = broyden(&col(&[1.0 + 1e-9]), seventh).unwrap();
    assert!(
        !r.converged || (r.x[(0, 0)] - 1.0).abs() <= bound(7) * XTOL,
        "{:?} at {:e}",
        r.stop,
        r.x[(0, 0)] - 1.0
    );
    // ln(x)³, and the triple root in one of two unknowns: (y − 1) − (x − 1)³ and y − 1.
    let ln3 = |v: &Matrix<f64>| col(&[v[(0, 0)].ln().powi(3)]);
    let two = |v: &Matrix<f64>| {
        col(&[
            (v[(1, 0)] - 1.0) - (v[(0, 0)] - 1.0).powi(3),
            v[(1, 0)] - 1.0,
        ])
    };
    for (name, r) in [
        ("ln(x)³", broyden(&col(&[1.0 + 1e-11]), ln3).unwrap()),
        (
            "two unknowns",
            broyden(&col(&[1.0 + 1e-11, 1.0]), two).unwrap(),
        ),
    ] {
        assert!(
            r.converged && (r.x[(0, 0)] - 1.0).abs() <= bound(3) * XTOL,
            "{name}: {:?} at {:?}",
            r.stop,
            r.x.as_slice()
        );
    }
    // In f32: (x − 1)⁵ from 1 + 10⁻⁴ stopped 10⁻⁴ from the root (836 ε), (x − 1)³ from 1 + 3·10⁻⁵
    // 3·10⁻⁵ from it.
    let xtol32 = 4.0 * f32::EPSILON;
    for (m, d) in [(5, 1e-4f32), (3, 3e-5)] {
        let f = move |v: &Matrix<f32>| col(&[(v[(0, 0)] - 1.0).powi(m)]);
        let r = broyden(&col(&[1.0 + d]), f).unwrap();
        assert!(
            r.converged && f64::from((r.x[(0, 0)] - 1.0).abs()) <= bound(m) * f64::from(xtol32),
            "m = {m}: {:?} at {:e}",
            r.stop,
            r.x[(0, 0)] - 1.0
        );
    }
}

#[test]
fn a_confirming_step_that_makes_no_progress_is_no_proof_of_a_root() {
    // x₁ = 10¹⁶ coupled into the second equation by x₁ − 10¹⁶, which f evaluates exactly, makes
    // the bound of the residual test 17.8 there, and with x₂ = 10¹² + u every correction of u is
    // short against x₂. Newton's step on arctan u from u = 2 overshoots to −3.5; the confirming
    // step saw no progress, which was taken for the floor of the rounding, and newton_system
    // stopped "converged" (Residual) at u = 2, where f₂ = 1.107, with and without the line search.
    let t = 1e12;
    let f = move |v: &Matrix<f64>| {
        col(&[
            v[(0, 0)] - 1e16,
            (v[(0, 0)] - 1e16) + (v[(1, 0)] - t).atan(),
        ])
    };
    let j = move |v: &Matrix<f64>| {
        Matrix::from_rows(&[[1.0, 0.0], [1.0, 1.0 / (1.0 + (v[(1, 0)] - t).powi(2))]])
    };
    let x0 = col(&[1e16, t + 2.0]);
    for line_search in [true, false] {
        let opts = SystemOptions::new().with_line_search(line_search);
        let r = newton_system_with(&x0, f, j, &opts).unwrap();
        assert!(
            r.converged && r.x.as_slice() == [1e16, t],
            "line search {line_search}: {:?} at u = {}",
            r.stop,
            r.x[(1, 0)] - t
        );
        let b = broyden_with(&x0, f, &opts).unwrap();
        assert!(
            !b.converged || near(&b.x, &[1e16, t], XTOL),
            "line search {line_search}: {:?} at u = {}",
            b.stop,
            b.x[(1, 0)] - t
        );
    }
    // The same with features of 10⁻¹¹ in an unknown of 2: arctan(10¹¹·(x₂ − 2)) from 2 + 2·10⁻¹¹.
    let fine = |v: &Matrix<f64>| {
        col(&[
            v[(0, 0)] - 1e16,
            (v[(0, 0)] - 1e16) + (1e11 * (v[(1, 0)] - 2.0)).atan(),
        ])
    };
    let fine_j = |v: &Matrix<f64>| {
        let u = 1e11 * (v[(1, 0)] - 2.0);
        Matrix::from_rows(&[[1.0, 0.0], [1.0, 1e11 / (1.0 + u * u)]])
    };
    let r = newton_system(&col(&[1e16, 2.0 + 2e-11]), fine, fine_j).unwrap();
    assert!(
        r.converged && r.x.as_slice() == [1e16, 2.0],
        "{:?} at {:e}",
        r.stop,
        r.x[(1, 0)] - 2.0
    );
    // Newton's 2-cycle of u³ − 2u + 2 between 0 and 1, with x₂ = 10¹¹ + u and couplings of 1 and
    // 100: both methods stopped "converged" at u = 1 or 0, where f₂ is 1 or 2. Now they reach the
    // root, −1.769…, or report the local minimum of |f₂| at u = 0.8165 that the line search finds.
    let t = 1e11;
    let root = -1.769_292_354_238_631_4;
    for c in [1.0, 100.0] {
        let g = move |v: &Matrix<f64>| {
            let u = v[(1, 0)] - t;
            col(&[
                v[(0, 0)] - 1e16,
                c * (v[(0, 0)] - 1e16) + u * u * u - 2.0 * u + 2.0,
            ])
        };
        let gj = move |v: &Matrix<f64>| {
            let u = v[(1, 0)] - t;
            Matrix::from_rows(&[[1.0, 0.0], [c, 3.0 * u * u - 2.0]])
        };
        for u0 in [0.0, 1.0] {
            for line_search in [true, false] {
                let opts = SystemOptions::new().with_line_search(line_search);
                let x0 = col(&[1e16, t + u0]);
                for (method, r) in [
                    ("newton", newton_system_with(&x0, g, gj, &opts).unwrap()),
                    ("broyden", broyden_with(&x0, g, &opts).unwrap()),
                ] {
                    assert!(
                        !r.converged || near(&r.x, &[1e16, t + root], XTOL),
                        "{method}, c = {c}, from u = {u0}, line search {line_search}: {:?} at \
                         u = {}",
                        r.stop,
                        r.x[(1, 0)] - t
                    );
                }
            }
        }
    }
}

#[test]
fn a_coupled_double_root_is_reached_to_the_bound_of_its_multiplicity() {
    // (x₂ − 2)²(1 + x₂) coupled to x₁ = 10¹⁶ by 0.01·(x₁ − 10¹⁶): Newton's method halves the error
    // at the double root, so each confirming step sees a correction half as long as the one
    // before — progress, not its floor. The tests then leave at most 4·xtol at a double root
    // (SystemStop; measured 2.5 to 2.75·xtol from these starts).
    let f = |v: &Matrix<f64>| {
        col(&[
            v[(0, 0)] - 1e16,
            0.01 * (v[(0, 0)] - 1e16) + (v[(1, 0)] - 2.0).powi(2) * (1.0 + v[(1, 0)]),
        ])
    };
    let j = |v: &Matrix<f64>| {
        let d = v[(1, 0)] - 2.0;
        Matrix::from_rows(&[[1.0, 0.0], [0.01, 2.0 * d * (1.0 + v[(1, 0)]) + d * d]])
    };
    for x2 in [3.0, 2.5, 2.01] {
        let x0 = col(&[0.0, x2]);
        let r = newton_system(&x0, f, j).unwrap();
        assert!(
            r.converged && near(&r.x, &[1e16, 2.0], 4.0 * XTOL),
            "from {x2}: {:?} at {:?}",
            r.stop,
            r.x.as_slice()
        );
        // Broyden's method may end at the simple root −1 instead, or report that it did not
        // reach the double root within its 100 steps.
        let b = broyden(&x0, f).unwrap();
        assert!(
            !b.converged
                || near(&b.x, &[1e16, 2.0], 4.0 * XTOL)
                || near(&b.x, &[1e16, -1.0], 2.0 * f64::EPSILON),
            "from {x2}: {:?} at {:?}",
            b.stop,
            b.x.as_slice()
        );
    }
}

#[test]
fn a_confirming_step_onto_a_residual_that_is_not_finite_proves_nothing() {
    // √(x₂ − 1 + 10⁻¹⁴) − 10⁻⁷, coupled to x₁ = 10¹⁶ so that the residual test is tried at once:
    // the confirming step from x₂ = 1 + 10⁻¹² overshoots below 1 − 10⁻¹⁴, where f is NaN. That
    // says nothing about the point; the line search takes over and reaches the root, x₂ = 1, and
    // without the line search the step is reported as one where f is not finite.
    let f = |v: &Matrix<f64>| {
        col(&[
            v[(0, 0)] - 1e16,
            0.01 * (v[(0, 0)] - 1e16) + (v[(1, 0)] - 1.0 + 1e-14).sqrt() - 1e-7,
        ])
    };
    let j = |v: &Matrix<f64>| {
        Matrix::from_rows(&[[1.0, 0.0], [0.01, 0.5 / (v[(1, 0)] - 1.0 + 1e-14).sqrt()]])
    };
    let x0 = col(&[1e16, 1.0 + 1e-12]);
    let r = newton_system(&x0, f, j).unwrap();
    assert!(
        r.converged && r.residual.is_finite() && near(&r.x, &[1e16, 1.0], XTOL),
        "{:?} at {:?}, residual {:e}",
        r.stop,
        r.x.as_slice(),
        r.residual
    );
    let plain = newton_system_with(&x0, f, j, &SystemOptions::new().with_line_search(false));
    assert_eq!(plain.unwrap().stop, SystemStop::NotFinite);
}

#[test]
fn the_step_test_keeps_the_better_of_the_point_and_its_correction() {
    // arctan(10⁷·(x − 1)) from 1 + 1.5·10⁻⁷ with xtol = 10⁻⁶: the correction, −3.2·10⁻⁷, passes the
    // step test, and the step overshoots to where |f| is larger (arctan 1.7 against arctan 1.5).
    // The point that passed the test is the answer, within the xtol asked for.
    let f = |v: &Matrix<f64>| col(&[(1e7 * (v[(0, 0)] - 1.0)).atan()]);
    let j = |v: &Matrix<f64>| {
        let u = 1e7 * (v[(0, 0)] - 1.0);
        col(&[1e7 / (1.0 + u * u)])
    };
    let x0 = 1.0 + 1.5e-7;
    let opts = SystemOptions::new().with_xtol(1e-6);
    let r = newton_system_with(&col(&[x0]), f, j, &opts).unwrap();
    assert!(r.converged && r.stop == SystemStop::Step, "{:?}", r.stop);
    assert_eq!(r.x[(0, 0)], x0);
    assert_eq!(r.residual, f(&col(&[x0]))[(0, 0)].abs());
    // The rounding test after a confirming step, likewise: |x₂ − 1| + 10⁻¹², coupled to x₁ = 10¹⁶
    // so that the residual test is tried at the kink, where SystemStop documents that the method
    // stops. The confirming step goes to 1 − 10⁻¹², where |f₂| is twice as large, no point along
    // the correction is lower, and the kink is the point returned, not the end of the step.
    let kink = |v: &Matrix<f64>| {
        col(&[
            v[(0, 0)] - 1e16,
            0.01 * (v[(0, 0)] - 1e16) + (v[(1, 0)] - 1.0).abs() + 1e-12,
        ])
    };
    let kink_j = |v: &Matrix<f64>| {
        Matrix::from_rows(&[
            [1.0, 0.0],
            [0.01, if v[(1, 0)] >= 1.0 { 1.0 } else { -1.0 }],
        ])
    };
    let r = newton_system(&col(&[1e16, 2.0]), kink, kink_j).unwrap();
    assert!(
        r.stop == SystemStop::Rounding && r.x.as_slice() == [1e16, 1.0] && r.residual == 1e-12,
        "{:?} at {:?}, residual {:e}",
        r.stop,
        r.x.as_slice(),
        r.residual
    );
}

#[test]
fn a_restart_lengthens_its_step_where_rounding_keeps_two_steps_apart() {
    // Powell's badly scaled system in f32: near the root e^(−x₁) ≈ 1 and 1.0001 cancel in the
    // second equation, whose values are then multiples of the spacing of the floats near 1, and
    // e^(−x₂) ≈ 1.1·10⁻⁴ changes them by a few of those over the step of x₂. Quartered, the
    // quotients scatter and then vanish; two zeros after quotients that were not are no
    // derivative, and the steps four and sixteen times longer agree. The rounding of f fixes x₂
    // only to about 10⁻⁴ of it in f32 (SystemStop), and the answers are that close to the f64
    // root.
    let f = |x: &Matrix<f32>| {
        col(&[
            1e4 * x[(0, 0)] * x[(1, 0)] - 1.0,
            (-x[(0, 0)]).exp() + (-x[(1, 0)]).exp() - 1.0001,
        ])
    };
    let root = [
        f64::from_bits(0x3ee7_07b2_b0b0_9734),
        f64::from_bits(0x4022_3658_dd90_9b84),
    ];
    for (start, line_search) in [
        ([0.0f32, 10.0], true),
        ([0.0, 10.0], false),
        ([0.0, 1.0], false),
    ] {
        let opts = SystemOptions::new().with_line_search(line_search);
        let r = broyden_with(&col(&start), f, &opts).unwrap();
        let e = (f64::from(r.x[(0, 0)]) - root[0])
            .abs()
            .max((f64::from(r.x[(1, 0)]) - root[1]).abs())
            / root[1];
        assert!(
            r.converged && e <= 1e-4,
            "from {start:?}, line search {line_search}: {:?}, error {e:e}",
            r.stop
        );
    }
}

/// The Chebyshev polynomial `T_i(y)` and its derivative, by the recurrence.
fn chebyshev(i: usize, y: f64) -> (f64, f64) {
    let (mut t0, mut t1, mut d0, mut d1) = (1.0, y, 0.0, 1.0);
    if i == 0 {
        return (1.0, 0.0);
    }
    for _ in 1..i {
        let (t2, d2) = (2.0 * y * t1 - t0, 2.0 * t1 + 2.0 * y * d1 - d0);
        (t0, t1, d0, d1) = (t1, t2, d1, d2);
    }
    (t1, d1)
}

/// The Chebyquad function of order `n` (Moré, Garbow and Hillstrom's problem 35), its Jacobian
/// and its standard start, `xⱼ = j/(n + 1)`.
fn chebyquad(n: usize) -> (ColumnFn, ColumnFn, Matrix<f64>) {
    let f = move |x: &Matrix<f64>| {
        Matrix::from_fn(n, 1, |r, _| {
            let i = r + 1;
            let s = (0..n)
                .map(|j| chebyshev(i, 2.0 * x[(j, 0)] - 1.0).0)
                .sum::<f64>()
                / n as f64;
            if i % 2 == 0 {
                s + 1.0 / ((i * i) as f64 - 1.0)
            } else {
                s
            }
        })
    };
    let j = move |x: &Matrix<f64>| {
        Matrix::from_fn(n, n, |r, k| {
            2.0 * chebyshev(r + 1, 2.0 * x[(k, 0)] - 1.0).1 / n as f64
        })
    };
    let x0 = Matrix::from_fn(n, 1, |j, _| (j + 1) as f64 / (n as f64 + 1.0));
    (Box::new(f), Box::new(j), x0)
}

#[test]
fn a_restart_allows_for_the_rounding_of_the_values_it_differences() {
    // Chebyquad's equations are means of Chebyshev polynomials of size one, which cancel; where a
    // derivative is small against them its quotients are mostly rounding, and the quotients at two
    // steps agree only within that rounding, 8ε of the values over the step. Without the
    // allowance broyden's restarts on the orders 5 and 7 from their starts could not be checked
    // and ended Singular; with it they reach the root newton_system finds with the Jacobian.
    for n in [5, 7] {
        let (f, j, x0) = chebyquad(n);
        let r = newton_system(&x0, &f, &j).unwrap();
        let b = broyden(&x0, &f).unwrap();
        assert!(
            r.converged && b.converged,
            "n = {n}: {:?}, {:?}",
            r.stop,
            b.stop
        );
        assert!(
            near(&b.x, r.x.as_slice(), 64.0 * f64::EPSILON),
            "n = {n}: {:?} against {:?}",
            b.x.as_slice(),
            r.x.as_slice()
        );
    }
}

#[test]
fn a_restart_whose_differences_cannot_be_checked_is_reported_singular() {
    // e^(x − 10¹⁷) − 2 from 10¹⁷, where the floats are 16 apart: no difference step is shorter
    // than the scale of f, no two steps agree, and broyden says so. newton_system, given e^u,
    // answers 10¹⁷, the float nearest the root 10¹⁷ + ln 2.
    let t = 1e17;
    let f = move |v: &Matrix<f64>| col(&[(v[(0, 0)] - t).exp() - 2.0]);
    let j = move |v: &Matrix<f64>| col(&[(v[(0, 0)] - t).exp()]);
    let b = broyden(&col(&[t]), f).unwrap();
    assert_eq!((b.converged, b.stop), (false, SystemStop::Singular));
    let r = newton_system(&col(&[t]), f, j).unwrap();
    assert!(r.converged && r.x[(0, 0)] == t, "{:?}", r.stop);
}

#[test]
fn at_the_resolution_of_the_format_the_tests_hold_to_a_few_tolerances() {
    // u³ − 1 at x = T + u. Where 4ε·T is not small against the scale of f in u, the step and
    // residual tests' estimate of the error, of the first order, can be short of it: on this
    // cubic, whose root is the only point where |f| is small, measured at most 2.7 times xtol·|x|
    // at T = 10¹⁵ (f64) and 2.5 at 3·10⁶ (f32), where 4ε·T is about 1, and 0.35 at T = 10¹⁴ and
    // 10⁵. Where f has a local minimum of |f| too, it can be worse: see the next test.
    for (t, within) in [(1e14, 0.5), (1e15, 3.0)] {
        let f = move |v: &Matrix<f64>| {
            let u = v[(0, 0)] - t;
            col(&[u * u * u - 1.0])
        };
        let j = move |v: &Matrix<f64>| {
            let u = v[(0, 0)] - t;
            col(&[3.0 * u * u])
        };
        for u0 in [0.5, 1.5, 3.0, 5.0, 10.0] {
            let x0 = col(&[t + u0]);
            for (method, r) in [
                ("newton", newton_system(&x0, f, j).unwrap()),
                ("broyden", broyden(&x0, f).unwrap()),
            ] {
                let e = ((r.x[(0, 0)] - t) - 1.0).abs() / (XTOL * t);
                assert!(
                    !r.converged || e <= within,
                    "{method}, T = {t:e}, u0 = {u0}: {e} tolerances"
                );
            }
        }
    }
    let xtol32 = 4.0 * f32::EPSILON;
    for (t, within) in [(1e5f32, 0.5f32), (1e6, 3.0)] {
        let f = move |v: &Matrix<f32>| {
            let u = v[(0, 0)] - t;
            col(&[u * u * u - 1.0])
        };
        let j = move |v: &Matrix<f32>| {
            let u = v[(0, 0)] - t;
            col(&[3.0 * u * u])
        };
        for u0 in [0.5f32, 1.5, 3.0, 5.0, 10.0] {
            let x0 = col(&[t + u0]);
            for (method, r) in [
                ("newton", newton_system(&x0, f, j).unwrap()),
                ("broyden", broyden(&x0, f).unwrap()),
            ] {
                let e = ((r.x[(0, 0)] - t) - 1.0).abs() / (xtol32 * t);
                assert!(
                    !r.converged || e <= within,
                    "{method}, T = {t:e}, u0 = {u0}: {e} tolerances"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The rounding test at a local minimum of the residual (final round)
// ---------------------------------------------------------------------------------------------

/// The four ways of running the methods on one system, by name: `newton_system` and `broyden`,
/// each with and without the line search.
fn four_ways<T>(
    x0: &Matrix<T>,
    f: impl Fn(&Matrix<T>) -> Matrix<T> + Copy,
    j: impl Fn(&Matrix<T>) -> Matrix<T> + Copy,
) -> Vec<(&'static str, SystemResult<T>)>
where
    T: a::math::numerics::Real + a::math::matrix::Scalar,
{
    let (on, off) = (
        SystemOptions::new(),
        SystemOptions::new().with_line_search(false),
    );
    vec![
        ("newton_system", newton_system_with(x0, f, j, &on).unwrap()),
        (
            "plain newton_system",
            newton_system_with(x0, f, j, &off).unwrap(),
        ),
        ("broyden", broyden_with(x0, f, &on).unwrap()),
        ("plain broyden", broyden_with(x0, f, &off).unwrap()),
    ]
}

/// The root of `u³ − 2u − 5`, and that of `u³ − 2u + 2`, to the last bit.
const CUBIC_ROOTS: [f64; 2] = [2.094_551_481_542_326_6, -1.769_292_354_238_631_4];

#[test]
fn the_rounding_test_takes_no_local_minimum_of_the_residual_for_a_root() {
    // (u − 1/3)² + c at x = T + u has no real root. Its least value, c, is at u = 1/3, which the
    // floats near T miss by a part of their spacing (1/64 at T = 10¹⁴), and the Newton correction
    // there, c/f′, is below ε^(2/3)·T (3,670): the line search finds no lower float along it, as at
    // the floor of the rounding, and the test that took that alone for rounding reported
    // converged, Rounding, at u = 0.328125 with f = 1.00003 (newton_system) and at u = 0.34375
    // (broyden). Along the correction f is not near its linear model at any step (at x + 4·δ it is
    // above f(x), where the model says −3·f(x)): both methods now end with Rounding and converged
    // false, at the best point, the float nearest the minimum.
    for (c, t) in [(1.0, 1e14), (0.01, 1e13), (0.01, 2.5e12)] {
        let f = move |v: &Matrix<f64>| col(&[(v[(0, 0)] - t - 1.0 / 3.0).powi(2) + c]);
        let j = move |v: &Matrix<f64>| col(&[2.0 * (v[(0, 0)] - t - 1.0 / 3.0)]);
        for (method, r) in [
            ("newton_system", newton_system(&col(&[t]), f, j).unwrap()),
            ("broyden", broyden(&col(&[t]), f).unwrap()),
        ] {
            let u = r.x[(0, 0)] - t;
            assert!(
                !r.converged && r.stop == SystemStop::Rounding,
                "{method}, c = {c}, T = {t:e}: converged {} {:?} at u = {u}",
                r.converged,
                r.stop
            );
            assert!(r.residual >= c && r.residual <= 1.001 * c && (u - 1.0 / 3.0).abs() < 0.02);
        }
    }
    // u³ − 2u − 5 has its root at u = 2.0946 and a local minimum of |f| at u = −0.8165, where
    // newton_system from u = 0 at T = 10¹⁴ had stopped "converged" with f = −3.91, 33 tolerances
    // from the root. broyden reaches the root from there.
    let t = 1e14;
    let f = move |v: &Matrix<f64>| {
        let u = v[(0, 0)] - t;
        col(&[u * u * u - 2.0 * u - 5.0])
    };
    let j = move |v: &Matrix<f64>| {
        let u = v[(0, 0)] - t;
        col(&[3.0 * u * u - 2.0])
    };
    let r = newton_system(&col(&[t]), f, j).unwrap();
    let u = r.x[(0, 0)] - t;
    assert!(
        !r.converged && r.stop == SystemStop::Rounding && (u + 0.8165).abs() < 0.02,
        "converged {} {:?} at u = {u}",
        r.converged,
        r.stop
    );
    let b = broyden(&col(&[t]), f).unwrap();
    assert!(b.converged && ((b.x[(0, 0)] - t) - CUBIC_ROOTS[0]).abs() <= 0.5 * XTOL * t);
}

#[test]
fn no_local_minimum_passes_the_rounding_test_whatever_the_offset() {
    // The family around the case above: x = T + u, T from 10¹⁰ to 10¹⁵ in f64 and from 10² to 10⁷
    // in f32, in quarter decades, from four starts, in the four ways. (u − 1/3)² + c has no root,
    // so no run may end converged on the rounding test; the test that took a failed line search
    // for rounding did on 180 of these f64 runs and 53 of the f32 ones. The step and residual
    // tests are of the first order and accept a local minimum whose least value c is below about
    // f″·(xtol·|x|)²/2 = (xtol·T)², which a tolerance on the step cannot tell from a double root
    // (SystemStop): measured on this grid up to 0.13 and 1.3 times that (f32: 0.44 and 1.4), on a
    // finer one up to 0.8 and 3.2; asserted, 4.
    let scale = if cfg!(miri) { 5 } else { 1 };
    for e in (0..=20).step_by(scale) {
        let t = 10f64.powf(10.0 + f64::from(e) / 4.0).round();
        for c in [1.0, 1e-2, 1e-4] {
            let f = move |v: &Matrix<f64>| col(&[(v[(0, 0)] - t - 1.0 / 3.0).powi(2) + c]);
            let j = move |v: &Matrix<f64>| col(&[2.0 * (v[(0, 0)] - t - 1.0 / 3.0)]);
            for u0 in [0.0, 2.0, -1.0, 0.3] {
                for (method, r) in four_ways(&col(&[t + u0]), f, j) {
                    assert!(
                        !r.converged
                            || (r.stop != SystemStop::Rounding && c <= 4.0 * (XTOL * t).powi(2)),
                        "{method}, c = {c}, T = {t:e}, u0 = {u0}: {:?} at u = {}",
                        r.stop,
                        r.x[(0, 0)] - t
                    );
                }
            }
        }
        // u³ − 2u − 5 and u³ − 2u + 2: a root, and a local minimum of |f| a unit or so from it,
        // which some starts descend to. No run ends converged on the rounding test away from the
        // root (the earlier test did on 48 of these runs, up to 164 tolerances away); the step and
        // residual tests answer within half a tolerance of it up to T = 1.8·10¹⁴ (measured at
        // most 0.47), and from 3.2·10¹⁴, where xtol·|x| = 0.28 is not small against the scale of
        // f, accept points up to 10.8 tolerances away (SystemStop documents it; asserted, 11).
        for (k, a) in [-5.0, 2.0].into_iter().enumerate() {
            let f = move |v: &Matrix<f64>| {
                let u = v[(0, 0)] - t;
                col(&[u * u * u - 2.0 * u + a])
            };
            let j = move |v: &Matrix<f64>| {
                let u = v[(0, 0)] - t;
                col(&[3.0 * u * u - 2.0])
            };
            for u0 in [0.0, -0.3, 0.3, -1.5, 1.5] {
                for (method, r) in four_ways(&col(&[t + u0]), f, j) {
                    let error = ((r.x[(0, 0)] - t) - CUBIC_ROOTS[k]).abs() / (XTOL * t);
                    let within = if t < 3e14 { 0.5 } else { 11.0 };
                    assert!(
                        !r.converged
                            || error <= within && (r.stop != SystemStop::Rounding || error <= 2.0),
                        "{method}, u³ − 2u + {a}, T = {t:e}, u0 = {u0}: {:?}, {error} tolerances",
                        r.stop
                    );
                }
            }
        }
    }
    let xtol32 = 4.0 * f64::from(f32::EPSILON);
    for e in (0..=20).step_by(scale) {
        let t = 10f64.powf(2.0 + f64::from(e) / 4.0).round() as f32;
        for c in [1.0f32, 1e-2, 1e-4] {
            let f = move |v: &Matrix<f32>| col(&[(v[(0, 0)] - t - 1.0 / 3.0).powi(2) + c]);
            let j = move |v: &Matrix<f32>| col(&[2.0 * (v[(0, 0)] - t - 1.0 / 3.0)]);
            for u0 in [0.0f32, 2.0, -1.0, 0.3] {
                for (method, r) in four_ways(&col(&[t + u0]), f, j) {
                    let bound = 4.0 * (xtol32 * f64::from(t)).powi(2);
                    assert!(
                        !r.converged || (r.stop != SystemStop::Rounding && f64::from(c) <= bound),
                        "{method}, f32, c = {c}, T = {t:e}, u0 = {u0}: {:?} at u = {}",
                        r.stop,
                        r.x[(0, 0)] - t
                    );
                }
            }
        }
    }
}

#[test]
fn a_system_of_two_equations_without_a_root_never_passes_the_rounding_test() {
    // Four systems with no root, at x = T + u and y = S + v: a local minimum of |f₁| beside a
    // line, the same coupled, a line that passes the unit circle at c from it, and uv + c = 0 on
    // u = v. From four starts in the four ways no run ends converged on the rounding test (the
    // test that took a failed line search for rounding did on 174 of these 3,072 runs, from
    // T = 10¹¹). The closest to passing is the line 10⁻⁴ from the circle with u at 1.8·10¹⁴,
    // which follows its model at best to 0.20 of the change it predicts, against the sixteenth
    // allowed. Without an offset no run ends converged at all, and the step and residual tests,
    // of the first order, accept points of these near-double roots only from T = 10¹² on.
    type System = fn(f64, f64, f64) -> ([f64; 2], [[f64; 2]; 2]);
    let systems: [(&str, System); 4] = [
        ("(u − 1/3)² + c, v − 1", |u, v, c| {
            let d = u - 1.0 / 3.0;
            ([d * d + c, v - 1.0], [[2.0 * d, 0.0], [0.0, 1.0]])
        }),
        ("(u − v/3)² + c, u + v − 1", |u, v, c| {
            let d = u - v / 3.0;
            (
                [d * d + c, u + v - 1.0],
                [[2.0 * d, -2.0 * d / 3.0], [1.0, 1.0]],
            )
        }),
        ("u² + v² − 1, u + v − √2 − c", |u, v, c| {
            (
                [u * u + v * v - 1.0, u + v - 2f64.sqrt() - c],
                [[2.0 * u, 2.0 * v], [1.0, 1.0]],
            )
        }),
        ("uv + c, u − v", |u, v, c| {
            ([u * v + c, u - v], [[v, u], [1.0, -1.0]])
        }),
    ];
    let offsets: &[f64] = if cfg!(miri) {
        &[0.0, 1e14]
    } else {
        &[
            0.0,
            1e10,
            1e11,
            1e12,
            1e13,
            1e14,
            177_827_941_003_892.0,
            1e15,
        ]
    };
    for (name, system) in systems {
        for &t in offsets {
            for s in [0.0, t] {
                for c in [1.0, 1e-2, 1e-4] {
                    let f = move |x: &Matrix<f64>| col(&system(x[(0, 0)] - t, x[(1, 0)] - s, c).0);
                    let j = move |x: &Matrix<f64>| {
                        Matrix::from_rows(&system(x[(0, 0)] - t, x[(1, 0)] - s, c).1)
                    };
                    for (u0, v0) in [(0.0, 0.0), (2.0, -1.0), (0.3, 0.7), (-1.0, 2.0)] {
                        for (method, r) in four_ways(&col(&[t + u0, s + v0]), f, j) {
                            assert!(
                                !r.converged || (r.stop != SystemStop::Rounding && t >= 1e12),
                                "{name}, c = {c}, T = {t:e}, S = {s:e}, from ({u0}, {v0}), \
                                 {method}: {:?} at ({}, {})",
                                r.stop,
                                r.x[(0, 0)] - t,
                                r.x[(1, 0)] - s
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn an_equation_whose_residual_is_far_below_anothers_is_held_to_its_own_terms() {
    // a·(u − 1/3)² + c with a tiny a, at x₁ = T + u, has no root, and its residual stays far
    // below that of the second equation, ((x₂ + 10⁴) − 10⁴) − 1/3, whose values are those of x₂
    // rounded to the spacing of the floats near 10⁴ (2·10⁻¹²) and which is soon at that floor. The
    // line search then fails along a short correction. Measured against the largest residual
    // alone, the first equation's departure from its model is negligible, and a test that did so
    // accepted 8 of the 16 runs of each case below as roots; measured against its own terms along
    // the correction, which are its residual, it fails as in one unknown.
    for t in [1e10, 1e14] {
        for c in [1e-20, 1e-30] {
            let f = move |v: &Matrix<f64>| {
                let u = v[(0, 0)] - t;
                col(&[
                    1e-16 * (u - 1.0 / 3.0).powi(2) + c,
                    ((v[(1, 0)] + 1e4) - 1e4) - 1.0 / 3.0,
                ])
            };
            let j = move |v: &Matrix<f64>| {
                let u = v[(0, 0)] - t;
                Matrix::from_rows(&[[2e-16 * (u - 1.0 / 3.0), 0.0], [0.0, 1.0]])
            };
            for (u0, v0) in [(0.3, 0.0), (0.0, 1.0), (0.33, 0.3), (1.0, 0.5)] {
                for (method, r) in four_ways(&col(&[t + u0, v0]), f, j) {
                    assert!(
                        !(r.converged && r.stop == SystemStop::Rounding),
                        "T = {t:e}, c = {c:e}, from ({u0}, {v0}), {method}: Rounding at u = {}, \
                         f = {:?}",
                        r.x[(0, 0)] - t,
                        r.f.as_slice()
                    );
                }
            }
        }
    }
}

#[test]
fn the_rounding_test_judges_the_point_it_returns_along_its_own_correction() {
    // Brown's almost-linear function of order 10 from its start ends on the rounding test by both
    // methods, after the confirming step of the residual test made no progress. newton_system
    // returns the point it confirmed from, whose residual is smaller than that at the end of the
    // step; broyden the end of the step. Either way the test is made at the point returned, along
    // its own correction. The residuals are those of the points the methods stop on (Brown's
    // function is sums and a product, evaluated alike everywhere).
    //
    // The steps are not: they come from QR factorisations whose Householder reflections call
    // `hypot`, which the platforms' libms round differently in the last bit. newton_system ends
    // as above on aarch64 macOS and on x86_64 with either of Ubuntu's glibc builds. broyden's last
    // step lands on an exact zero of f with the x86-64-v3 build (CI's), where it stops on the
    // residual test, a root, before the rounding test is made; that ending is allowed too.
    let (f, j) = brown(10);
    let x0 = col(&[0.5; 10]);
    let at = |x: &Matrix<f64>| f(x).as_slice().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let r = newton_system(&x0, &f, &j).unwrap();
    assert!(
        r.converged && r.stop == SystemStop::Rounding,
        "{:?}",
        r.stop
    );
    assert_eq!(r.residual, 2.220446049250313e-16);
    assert_eq!(r.residual, at(&r.x));
    let b = broyden(&x0, &f).unwrap();
    assert!(
        b.converged
            && (b.stop == SystemStop::Rounding
                || (b.stop == SystemStop::Residual && b.residual == 0.0)),
        "{:?} at residual {:e}",
        b.stop,
        b.residual
    );
    assert!(b.residual <= 1.8e-15, "{:e}", b.residual);
    assert_eq!(b.residual, at(&b.x));
    // A probe of the test where f is not finite is no agreement with the model: (u − 1/3)² + 1 at
    // x = 10¹⁴ + u, with f not finite beyond u = 10, ends on the rounding test with converged
    // false by both methods with the line search, as without the wall, where every probe of the
    // test is beyond it (and without the line search after its 100 steps).
    let t = 1e14;
    let walled = move |v: &Matrix<f64>| {
        let u = v[(0, 0)] - t;
        col(&[if u > 10.0 {
            f64::NAN
        } else {
            (u - 1.0 / 3.0).powi(2) + 1.0
        }])
    };
    let wj = move |v: &Matrix<f64>| col(&[2.0 * (v[(0, 0)] - t - 1.0 / 3.0)]);
    for (method, r) in four_ways(&col(&[t]), walled, wj) {
        assert!(
            !r.converged && (method.starts_with("plain") || r.stop == SystemStop::Rounding),
            "{method}: {:?} at u = {}",
            r.stop,
            r.x[(0, 0)] - t
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The two methods
// ---------------------------------------------------------------------------------------------

#[test]
fn newton_and_broyden_reach_the_same_root() {
    let x0 = col(&[1.5, 0.5]);
    let n = newton_system(&x0, circle_and_line, circle_and_line_jacobian).unwrap();
    let b = broyden(&x0, circle_and_line).unwrap();
    for r in [&n, &b] {
        assert!(r.converged);
        for (got, want) in r.x.as_slice().iter().zip(&ROOT) {
            assert!((got - want).abs() <= XTOL * 2.0, "{got} against {want}");
        }
    }
    // Newton converges quadratically: five steps from (1.5, 0.5), five Jacobians.
    assert_eq!((n.iterations, n.jacobians), (5, 5));
}

#[test]
fn broydens_update_is_what_makes_it_fast() {
    // On the circle and the line Broyden's method needs six steps on its first Jacobian; frozen
    // at that Jacobian, a chord iteration takes 31 (the audit's replica), so a method that did not
    // update could not pass. The residual falls superlinearly: each ratio smaller than the last.
    let x0 = col(&[1.5, 0.5]);
    let r = broyden(&x0, circle_and_line).unwrap();
    assert!(
        r.converged && r.iterations <= 6 && r.jacobians == 1,
        "{} steps, {} Jacobians",
        r.iterations,
        r.jacobians
    );
    let residuals: Vec<f64> = (1..=4)
        .map(|k| {
            broyden_with(&x0, circle_and_line, &SystemOptions::new().with_max_iter(k))
                .unwrap()
                .residual
        })
        .collect();
    let ratios: Vec<f64> = residuals.windows(2).map(|w| w[1] / w[0]).collect();
    assert!(
        ratios.windows(2).all(|w| w[1] < w[0]) && ratios[2] < 1e-2,
        "{residuals:?}"
    );
}

#[test]
fn the_line_search_brings_newton_in_where_the_plain_method_diverges() {
    // Brown's function of order 30 from (0.5, …, 0.5): the plain Newton iteration does not converge
    // in 100 steps; the line search shortens its steps and converges in 21, as MINPACK does.
    let (f, j) = brown(30);
    let x0 = col(&[0.5; 30]);
    let plain =
        newton_system_with(&x0, &f, &j, &SystemOptions::new().with_line_search(false)).unwrap();
    assert!(!plain.converged, "{:?}", plain.stop);
    let damped = newton_system(&x0, &f, &j).unwrap();
    assert!(
        damped.converged && damped.residual <= 1e-13,
        "{:?} at {:e}",
        damped.stop,
        damped.residual
    );
    // arctan in each unknown from (4, 5, 6): the full steps overshoot to where the derivative
    // vanishes; shortened, they converge.
    let at = |x: &Matrix<f64>| {
        col(&[
            (x[(0, 0)] - 1.0).atan(),
            (x[(1, 0)] - 2.0).atan(),
            (x[(2, 0)] - 3.0).atan(),
        ])
    };
    let atj = |x: &Matrix<f64>| {
        Matrix::from_fn(3, 3, |i, k| {
            if i == k {
                1.0 / (1.0 + (x[(i, 0)] - (i + 1) as f64).powi(2))
            } else {
                0.0
            }
        })
    };
    let x0 = col(&[4.0, 5.0, 6.0]);
    assert!(
        !newton_system_with(&x0, at, atj, &SystemOptions::new().with_line_search(false))
            .unwrap()
            .converged
    );
    let r = newton_system(&x0, at, atj).unwrap();
    assert!(r.converged && (r.x[(2, 0)] - 3.0).abs() <= XTOL * 3.0);
    // Without the line search a step to where f is not finite ends the iteration.
    let g = |x: &Matrix<f64>| col(&[x[(0, 0)].ln() - 1.0]);
    let gj = |x: &Matrix<f64>| col(&[1.0 / x[(0, 0)]]);
    let plain = newton_system_with(
        &col(&[10.0]),
        g,
        gj,
        &SystemOptions::new().with_line_search(false),
    )
    .unwrap();
    assert_eq!(plain.stop, SystemStop::NotFinite);
    let r = newton_system(&col(&[10.0]), g, gj).unwrap();
    assert!(r.converged && (r.x[(0, 0)] - core::f64::consts::E).abs() <= XTOL * 3.0);
}

#[test]
fn newton_solves_a_linear_system_in_one_step() {
    // A linear f is its own linear model: one step lands on the solution wherever the start is.
    let a = Matrix::from_rows(&[[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]]);
    let b = col(&[1.0, 2.0, 3.0]);
    for start in [
        col(&[0.0, 0.0, 0.0]),
        col(&[10.0, -7.0, 3.5]),
        col(&[1e10, 1e-10, -1e5]),
    ] {
        let (af, aj, bf) = (a.clone(), a.clone(), b.clone());
        let r = newton_system(
            &start,
            move |x: &Matrix<f64>| &af * x - &bf,
            move |_: &Matrix<f64>| aj.clone(),
        )
        .unwrap();
        assert!(r.converged && r.iterations <= 3, "{} steps", r.iterations);
        // The residual is rounding against the size of the terms, as the test promises.
        let residual = &a * &r.x - &b;
        let terms = &a.map(|v: &f64| v.abs()) * &r.x.map(|v: &f64| v.abs());
        for i in 0..3 {
            assert!(
                residual[(i, 0)].abs() <= 8.0 * f64::EPSILON * terms[(i, 0)],
                "{:?}",
                residual.as_slice()
            );
        }
    }
}

#[test]
fn a_system_of_one_equation_is_a_scalar_root() {
    let f = |x: &Matrix<f64>| col(&[x[(0, 0)].exp() - 2.0]);
    let j = |x: &Matrix<f64>| col(&[x[(0, 0)].exp()]);
    for r in [
        newton_system(&col(&[3.0]), f, j).unwrap(),
        broyden(&col(&[3.0]), f).unwrap(),
    ] {
        assert!(r.converged && (r.x[(0, 0)] - core::f64::consts::LN_2).abs() <= XTOL);
    }
}

#[test]
fn a_jacobian_singular_at_the_root_still_converges_given_the_units() {
    // x² = 0 and y² = 0: the Jacobian vanishes at the root, Newton halves the error each step
    // and the default tests, free of units, cannot verify a root at zero. With an ftol it stops.
    let f = |x: &Matrix<f64>| col(&[x[(0, 0)] * x[(0, 0)], x[(1, 0)] * x[(1, 0)]]);
    let j = |x: &Matrix<f64>| Matrix::from_rows(&[[2.0 * x[(0, 0)], 0.0], [0.0, 2.0 * x[(1, 0)]]]);
    let x0 = col(&[1.0, 1.0]);
    let r = newton_system(&x0, f, j).unwrap();
    assert!(!r.converged && r.stop == SystemStop::Iterations && r.residual < 1e-55);
    let r = newton_system_with(&x0, f, j, &SystemOptions::new().with_ftol(1e-30)).unwrap();
    assert!(
        r.converged && r.residual <= 1e-30 && r.iterations == 50,
        "{} steps",
        r.iterations
    );
    // A root away from zero that is singular: (x − 1)² = 0, y = 2: linear convergence to 1, and
    // the tests, relative to x = 1, end it within the bound of a double root, 4·xtol (SystemStop;
    // measured 2·xtol by newton_system and 1·xtol by broyden, after 50 and 72 steps).
    let g = |x: &Matrix<f64>| col(&[(x[(0, 0)] - 1.0).powi(2), x[(1, 0)] - 2.0]);
    let gj = |x: &Matrix<f64>| Matrix::from_rows(&[[2.0 * (x[(0, 0)] - 1.0), 0.0], [0.0, 1.0]]);
    let x0 = col(&[3.0, 0.0]);
    for (method, r) in [
        ("newton", newton_system(&x0, g, gj).unwrap()),
        ("broyden", broyden(&x0, g).unwrap()),
    ] {
        assert!(
            r.converged && (r.x[(0, 0)] - 1.0).abs() <= 4.0 * XTOL && r.x[(1, 0)] == 2.0,
            "{method}: {:?} {}",
            r.stop,
            r.x[(0, 0)]
        );
    }
}

#[test]
fn a_system_without_a_solution_is_reported_as_such() {
    // x² + y² + 1 = 0 has no real solution; the methods descend ½‖f‖² to its least value, 1/2, at
    // the origin, and say that is no root.
    let f = |x: &Matrix<f64>| {
        col(&[
            x[(0, 0)].powi(2) + x[(1, 0)].powi(2) + 1.0,
            x[(0, 0)] - x[(1, 0)],
        ])
    };
    let j = |x: &Matrix<f64>| Matrix::from_rows(&[[2.0 * x[(0, 0)], 2.0 * x[(1, 0)]], [1.0, -1.0]]);
    let x0 = col(&[1.0, 2.0]);
    let rn = newton_system(&x0, f, j).unwrap();
    let rb = broyden(&x0, f).unwrap();
    for r in [&rn, &rb] {
        assert!(!r.converged, "{:?}", r.stop);
        assert!(matches!(
            r.stop,
            SystemStop::LineSearch | SystemStop::Singular | SystemStop::Iterations
        ));
        assert!((r.residual - 1.0).abs() < 1e-6, "{}", r.residual);
    }
}

#[test]
fn a_singular_jacobian_is_reported_not_divided_by() {
    // Both equations are one plane: the Jacobian is singular everywhere.
    let f = |x: &Matrix<f64>| {
        col(&[
            x[(0, 0)] + x[(1, 0)] - 1.0,
            2.0 * x[(0, 0)] + 2.0 * x[(1, 0)] - 2.0,
        ])
    };
    let j = |_: &Matrix<f64>| Matrix::from_rows(&[[1.0, 1.0], [2.0, 2.0]]);
    let r = newton_system(&col(&[0.0, 0.0]), f, j).unwrap();
    assert!(!r.converged && r.stop == SystemStop::Singular && r.iterations == 0);
    // At the start only: x² − y = 0, y = 1 from (0, 0.5).
    let g = |x: &Matrix<f64>| col(&[x[(0, 0)] * x[(0, 0)] - x[(1, 0)], x[(1, 0)] - 1.0]);
    let gj = |x: &Matrix<f64>| Matrix::from_rows(&[[2.0 * x[(0, 0)], -1.0], [0.0, 1.0]]);
    let r = newton_system(&col(&[0.0, 0.5]), g, gj).unwrap();
    assert_eq!((r.converged, r.stop), (false, SystemStop::Singular));
    assert_eq!(r.x.as_slice(), &[0.0, 0.5]);
    // A Jacobian with a NaN in it is no better.
    let nan = |_: &Matrix<f64>| Matrix::from_rows(&[[f64::NAN, 1.0], [0.0, 1.0]]);
    assert_eq!(
        newton_system(&col(&[1.0, 0.5]), g, nan).unwrap().stop,
        SystemStop::Singular
    );
}

#[test]
fn a_residual_that_stops_being_finite_on_the_way_is_stepped_around() {
    // √x − 2 = 0 from 100: the full Newton step overshoots to −60, where √ is NaN; the line
    // search shortens it.
    let f = |x: &Matrix<f64>| col(&[x[(0, 0)].sqrt() - 2.0]);
    let j = |x: &Matrix<f64>| col(&[0.5 / x[(0, 0)].sqrt()]);
    let r = newton_system(&col(&[100.0]), f, j).unwrap();
    assert!(r.converged && (r.x[(0, 0)] - 4.0).abs() <= XTOL * 4.0);
    let r = broyden(&col(&[100.0]), f).unwrap();
    assert!(r.converged && (r.x[(0, 0)] - 4.0).abs() <= XTOL * 4.0);
}

#[test]
fn broyden_starts_from_the_callers_jacobian_when_given_one() {
    let x0 = col(&[1.5, 0.5]);
    let j0 = circle_and_line_jacobian(&x0);
    let given = broyden_with_jacobian(&x0, circle_and_line, &j0, &SystemOptions::new()).unwrap();
    let differenced = broyden(&x0, circle_and_line).unwrap();
    assert!(given.converged && differenced.converged);
    // The differences cost two evaluations at the start.
    assert_eq!(differenced.evaluations, given.evaluations + 2);
    assert_eq!(given.x.as_slice(), differenced.x.as_slice());
}

// ---------------------------------------------------------------------------------------------
// The options and the shapes
// ---------------------------------------------------------------------------------------------

#[test]
fn the_options_bound_the_work_and_set_the_tolerances() {
    let x0 = col(&[1.5, 0.5]);
    // One step is not enough on the circle; the best point after it is reported.
    let one = newton_system_with(
        &x0,
        circle_and_line,
        circle_and_line_jacobian,
        &SystemOptions::new().with_max_iter(1),
    )
    .unwrap();
    assert!(!one.converged && one.stop == SystemStop::Iterations && one.iterations == 1);
    // (One step from (1.5, 0.5) is (1.875, 0.875), where the circle is off by 0.28125.)
    assert_eq!(one.residual, 0.28125);
    // No steps at all: the start, unless it is a root.
    let none = SystemOptions::new().with_max_iter(0);
    assert_eq!(
        newton_system_with(&x0, circle_and_line, circle_and_line_jacobian, &none)
            .unwrap()
            .stop,
        SystemStop::Iterations
    );
    let at_root = col(&[2.0, 0.0]);
    let lines = |x: &Matrix<f64>| col(&[x[(0, 0)] - 2.0, x[(1, 0)]]);
    assert!(broyden_with(&at_root, lines, &none).unwrap().converged);
    // A loose ftol accepts a point that is not the root, as asked, and only one with so small a
    // residual.
    let loose = SystemOptions::new().with_ftol(1e-2);
    let q = newton_system_with(&x0, circle_and_line, circle_and_line_jacobian, &loose).unwrap();
    assert!(
        q.converged && q.stop == SystemStop::Residual && q.residual <= 1e-2 && q.iterations < 5
    );
    // A coarser difference step still converges, to the same root (the step test needs the
    // fresh Jacobian to agree, and it does).
    let coarse = SystemOptions::new().with_finite_difference_step(1e-3);
    let b = broyden_with(&x0, circle_and_line, &coarse).unwrap();
    assert!(b.converged && (b.x[(0, 0)] - ROOT[0]).abs() <= XTOL * 2.0);
    // Tolerances that are negative, not finite, or a step of zero, are refused.
    for bad in [
        SystemOptions::new().with_xtol(-1.0),
        SystemOptions::new().with_ftol(f64::NAN),
        SystemOptions::new().with_ftol(f64::INFINITY),
        SystemOptions::new().with_finite_difference_step(0.0),
    ] {
        assert!(newton_system_with(&x0, circle_and_line, circle_and_line_jacobian, &bad).is_none());
        assert!(broyden_with(&x0, circle_and_line, &bad).is_none());
    }
    let o = SystemOptions::<f64>::new();
    assert_eq!(
        (o.xtol, o.ftol, o.max_iter, o.h, o.line_search),
        (XTOL, 0.0, 100, f64::EPSILON.sqrt(), true)
    );
    assert_eq!(SystemOptions::<f64>::default(), o);
}

#[test]
fn each_test_of_a_root_is_one_the_methods_stop_on() {
    // The residual test, exactly: Rosenbrock's system lands on (1, 1), where f is zero.
    let p = &mgh()[0];
    let r = newton_system(&col(&p.x0), |x| p.residual(x), |x| p.jacobian(x)).unwrap();
    assert_eq!((r.stop, r.residual), (SystemStop::Residual, 0.0));
    // The residual test by the backward error, confirmed by one more step: the discrete boundary
    // value problem of order 10 stops with a residual of rounding, 3·10⁻¹⁷, against terms of 0.1.
    let p = mgh()
        .into_iter()
        .find(|p| p.name == "discrete_boundary_10")
        .unwrap();
    let r = newton_system(&col(&p.x0), |x| p.residual(x), |x| p.jacobian(x)).unwrap();
    assert!(
        r.stop == SystemStop::Residual && r.residual > 0.0 && r.residual < 1e-16,
        "{:?} at {:e}",
        r.stop,
        r.residual
    );
    // The step test at the default xtol: Broyden's tridiagonal system ends with a correction
    // below 4ε of every unknown.
    let p = mgh()
        .into_iter()
        .find(|p| p.name == "broyden_tridiagonal_10")
        .unwrap();
    let r = newton_system(&col(&p.x0), |x| p.residual(x), |x| p.jacobian(x)).unwrap();
    assert!(
        r.stop == SystemStop::Step && r.residual < 1e-15,
        "{:?} at {:e}",
        r.stop,
        r.residual
    );
    // The step test, with an xtol of 10⁻⁶: it ends the iteration on the circle before the
    // residual is down to rounding, and the full step then taken squares the error once more.
    let opts = SystemOptions::new().with_xtol(1e-6);
    let r = newton_system_with(
        &col(&[1.5, 0.5]),
        circle_and_line,
        circle_and_line_jacobian,
        &opts,
    )
    .unwrap();
    assert!(r.converged && r.stop == SystemStop::Step, "{:?}", r.stop);
    assert!((r.x[(0, 0)] - ROOT[0]).abs() <= 1e-12);
    // The rounding test: on the trigonometric system the residual cancels to 4·10⁻¹⁶ and Newton's
    // line search can lower it no further, along a correction far below ε^(2/3).
    let p = mgh()
        .into_iter()
        .find(|p| p.name == "trigonometric_10")
        .unwrap();
    let r = newton_system(&col(&p.x0), |x| p.residual(x), |x| p.jacobian(x)).unwrap();
    assert!(
        r.converged && r.stop == SystemStop::Rounding,
        "{:?}",
        r.stop
    );
    assert!(r.residual < 1e-15);
}

#[test]
fn an_empty_system_is_solved_at_once() {
    let empty = Matrix::<f64>::zeros(0, 1);
    let f = |_: &Matrix<f64>| -> Matrix<f64> { panic!("f is not called") };
    let j = |_: &Matrix<f64>| -> Matrix<f64> { panic!("j is not called") };
    let r: SystemResult<f64> = newton_system(&empty, f, j).unwrap();
    assert!(r.converged && r.evaluations == 0 && r.x.rows() == 0);
    assert!(broyden(&empty, f).unwrap().converged);
}

#[test]
#[should_panic(expected = "not a column")]
fn a_start_that_is_not_a_column_panics() {
    let _ = newton_system(
        &Matrix::from_rows(&[[0.0, 0.0]]),
        circle_and_line,
        circle_and_line_jacobian,
    );
}

#[test]
#[should_panic(expected = "not a column of 2 rows")]
fn a_residual_of_the_wrong_shape_panics() {
    let _ = broyden(&col(&[0.0, 0.0]), |_: &Matrix<f64>| col(&[0.0, 0.0, 1.0]));
}

#[test]
#[should_panic(expected = "the Jacobian is 3×3, not 2×2")]
fn a_jacobian_of_the_wrong_shape_panics() {
    let _ = newton_system(&col(&[1.0, 2.0]), circle_and_line, |_: &Matrix<f64>| {
        Matrix::identity(3)
    });
}

#[test]
fn f32_systems_solve_to_f32_precision() {
    let f = |x: &Matrix<f32>| {
        col(&[
            x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0,
            x[(0, 0)] - x[(1, 0)] - 1.0,
        ])
    };
    let j = |x: &Matrix<f32>| Matrix::from_rows(&[[2.0 * x[(0, 0)], 2.0 * x[(1, 0)]], [1.0, -1.0]]);
    let x0 = col(&[1.5f32, 0.5]);
    let tol = 8.0 * f64::from(f32::EPSILON) * 2.0;
    for r in [newton_system(&x0, f, j).unwrap(), broyden(&x0, f).unwrap()] {
        assert!(r.converged, "{:?}", r.stop);
        assert!(
            (f64::from(r.x[(0, 0)]) - ROOT[0]).abs() <= tol
                && (f64::from(r.x[(1, 0)]) - ROOT[1]).abs() <= tol
        );
    }
    // Rosenbrock and Broyden's tridiagonal function of order 10 in f32, against the f64 roots.
    let rf = |x: &Matrix<f32>| col(&[10.0 * (x[(1, 0)] - x[(0, 0)] * x[(0, 0)]), 1.0 - x[(0, 0)]]);
    let rj = |x: &Matrix<f32>| Matrix::from_rows(&[[-20.0 * x[(0, 0)], 10.0], [-1.0, 0.0]]);
    for r in [
        newton_system(&col(&[-1.2f32, 1.0]), rf, rj).unwrap(),
        broyden(&col(&[-1.2f32, 1.0]), rf).unwrap(),
    ] {
        assert!(
            r.converged
                && r.x
                    .as_slice()
                    .iter()
                    .all(|v| (v - 1.0).abs() <= 4.0 * f32::EPSILON)
        );
    }
    let n = 10;
    let tri = move |x: &Matrix<f32>| {
        Matrix::from_fn(n, 1, |k, _| {
            let xm = if k > 0 { x[(k - 1, 0)] } else { 0.0 };
            let xp = if k + 1 < n { x[(k + 1, 0)] } else { 0.0 };
            (3.0 - 2.0 * x[(k, 0)]) * x[(k, 0)] - xm - 2.0 * xp + 1.0
        })
    };
    let root: Vec<f64> = ROOTS
        .iter()
        .find(|r| r.0 == "broyden_tridiagonal_10")
        .unwrap()
        .1
        .iter()
        .map(|&b| f64::from_bits(b))
        .collect();
    let r = broyden(&Matrix::from_fn(n, 1, |_, _| -1.0f32), tri).unwrap();
    assert!(r.converged);
    let e =
        r.x.as_slice()
            .iter()
            .zip(&root)
            .fold(0.0f64, |m, (&a, &b)| m.max((f64::from(a) - b).abs()));
    assert!(e <= 16.0 * f64::from(f32::EPSILON), "error {e:e}");
}

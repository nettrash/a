//! Functions of matrices: the exponential, the square root and the logarithm.

use super::blas::{Op, Side, Uplo, gemm, trsm};
use super::eigen::real_to_complex_schur;
use super::element::sealed::{Element as _, Scalar as _};
use super::schur::schur;
use super::{Matrix, Norm, Scalar};
use crate::math::complex::Float;

/// The coefficients of the Padé approximants of `e^x` of degree 3, 5, 7, 9 and 13, from the
/// constant term up, and the largest 1-norm each is accurate for in double precision (Higham,
/// 2005).
const PADE: [(&[f64], f64); 5] = [
    (&[120.0, 60.0, 12.0, 1.0], 1.495_585_217_958_292e-2),
    (
        &[30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0],
        2.539_398_330_063_23e-1,
    ),
    (
        &[
            17_297_280.0,
            8_648_640.0,
            1_995_840.0,
            277_200.0,
            25200.0,
            1512.0,
            56.0,
            1.0,
        ],
        9.504_178_996_162_932e-1,
    ),
    (
        &[
            17_643_225_600.0,
            8_821_612_800.0,
            2_075_673_600.0,
            302_702_400.0,
            30_270_240.0,
            2_162_160.0,
            110_880.0,
            3960.0,
            90.0,
            1.0,
        ],
        2.097_847_961_257_068,
    ),
    (
        &[
            64_764_752_532_480_000.0,
            32_382_376_266_240_000.0,
            7_771_770_303_897_600.0,
            1_187_353_796_428_800.0,
            129_060_195_264_000.0,
            10_559_470_521_600.0,
            670_442_572_800.0,
            33_522_128_640.0,
            1_323_241_920.0,
            40_840_800.0,
            960_960.0,
            16380.0,
            182.0,
            1.0,
        ],
        5.371_920_351_148_152,
    ),
];

/// In double precision: the degrees 3, 5, 7 and 9, by their places in [`PADE`], and their bounds,
/// below 13's.
const PADE_DOUBLE: [(usize, f64); 4] = [
    (0, PADE[0].1),
    (1, PADE[1].1),
    (2, PADE[2].1),
    (3, PADE[3].1),
];

/// In single precision: the degrees 3, 5 and 7 and their bounds.
const PADE_SINGLE: [(usize, f64); 3] = [
    (0, 4.258_730_016_922_831e-1),
    (1, 1.880_152_677_804_762),
    (2, 3.925_724_783_138_66),
];

/// The nodes and weights of Gauss and Legendre's rule of 7 points on `[0, 1]`.
const GAUSS: [(f64, f64); 7] = [
    (0.025_446_043_828_620_74, 0.064_742_483_084_434_85),
    (0.129_234_407_200_302_78, 0.139_852_695_744_638_33),
    (0.297_077_424_311_301_4, 0.190_915_025_252_559_47),
    (0.5, 0.208_979_591_836_734_7),
    (0.702_922_575_688_698_6, 0.190_915_025_252_559_47),
    (0.870_765_592_799_697_2, 0.139_852_695_744_638_33),
    (0.974_553_956_171_379_3, 0.064_742_483_084_434_85),
];

/// How close to `I` the repeated square roots take `T` before the logarithm's Padé
/// approximant of degree 7, in the 1-norm (Higham, 2001).
const LOG_THETA: f64 = 0.264;

impl<T: Scalar> Matrix<T> {
    /// The exponential `e^A = I + A + A²/2! + …`, by scaling and squaring: `A / 2^s` small
    /// enough for a Padé approximant of degree 3 to 13, chosen by its 1-norm (7 in single
    /// precision), then squared `s` times, as Higham's `expm` (2005).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    ///
    /// // The generator of rotations: e^A rotates by θ.
    /// let t = 0.5f64;
    /// let a = Matrix::from_rows(&[[0.0, -t], [t, 0.0]]);
    /// let r = Matrix::from_rows(&[[t.cos(), -t.sin()], [t.sin(), t.cos()]]);
    /// assert!((&a.exp() - &r).as_slice().iter().all(|x| x.abs() < 1e-15));
    /// ```
    #[track_caller]
    pub fn exp(&self) -> Self {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        if self.data.iter().any(|x| !x.is_finite()) {
            let nan = T::from_parts(T::Real::NAN, T::Real::NAN);
            return Matrix::from_fn(n, n, |_, _| nan);
        }
        let norm = self.norm(Norm::One).to_f64();
        let single = T::Real::MANTISSA_DIGITS <= 24;
        let (table, top): (&[(usize, f64)], usize) = if single {
            (&PADE_SINGLE, 2)
        } else {
            (&PADE_DOUBLE, 4)
        };
        for &(d, theta) in table {
            if norm <= theta {
                return pade(self, PADE[d].0);
            }
        }
        let theta = if single { PADE_SINGLE[2].1 } else { PADE[4].1 };
        // A norm that overflows the elements' type, of finite elements, is at most `n` times the
        // largest part times √2: its logarithm from that keeps the squarings few, where `∞` would
        // make them `i32::MAX`.
        let log2 = if norm.is_finite() {
            (norm / theta).log2()
        } else {
            let big = self.data.iter().fold(0.0f64, |s, x| {
                s.max(x.re().abs().to_f64().max(x.im().abs().to_f64()))
            });
            (n as f64).log2() + big.log2() + 0.5 - theta.log2()
        };
        let s = log2.ceil().max(0.0) as i32;
        let scaled = self.map(|&x| x.scale(T::Real::from_f64(0.5f64.powi(s))));
        let mut x = pade(&scaled, PADE[top].0);
        for _ in 0..s {
            x = &x * &x;
        }
        x
    }

    /// The principal square root `√A`, whose eigenvalues have positive real parts, by the
    /// Schur method: `R² = T` solved for an upper triangular `R` element by element, then
    /// `Q·R·Q*`. `None` for a real matrix with a negative real eigenvalue, whose square roots
    /// are not real, or for a singular matrix that has none.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    ///
    /// let a: Matrix<f64> = Matrix::from_rows(&[[33.0, 24.0], [48.0, 57.0]]);
    /// let r = a.sqrt().unwrap();
    /// assert!((&r - &Matrix::from_rows(&[[5.0, 2.0], [4.0, 7.0]])).as_slice().iter().all(|x| x.abs() < 1e-13));
    /// assert!(Matrix::from_rows(&[[-1.0, 0.0], [0.0, 1.0]]).sqrt().is_none());
    /// ```
    #[track_caller]
    pub fn sqrt(&self) -> Option<Self> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let (t, q) = self.complex_schur(|w| w.re < T::Real::ZERO)?;
        let r = triangular_sqrt(&t, n)?;
        Some(self.back_from_schur(&r, &q))
    }

    /// The principal logarithm `log A`, whose eigenvalues have imaginary parts in `(-π, π]`, by
    /// inverse scaling and squaring: square roots of the Schur form `T` until it is near `I`,
    /// then a Padé approximant of degree 7 in partial fractions, times `2^k` for `k` square
    /// roots, as Higham's `logm` (2001) with Al-Mohy and Higham's accurate diagonal. `None`
    /// for a singular matrix, and for a real one with an eigenvalue on the negative real axis,
    /// whose logarithms are not real.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    ///
    /// let a: Matrix<f64> = Matrix::from_rows(&[[1.0, 2.0], [0.5, 3.0]]);
    /// let back = a.log().unwrap().exp();
    /// assert!((&back - &a).as_slice().iter().all(|x| x.abs() < 1e-13));
    /// assert!(Matrix::<f64>::zeros(2, 2).log().is_none());
    /// ```
    #[track_caller]
    pub fn log(&self) -> Option<Self> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let zero = T::Real::ZERO;
        let (t, q) = self.complex_schur(|w| w.re <= zero)?;
        if (0..n).any(|i| t[i + i * n] == T::C::zero()) {
            return None;
        }
        // Square roots until `‖T - I‖₁ ≤ θ`.
        let mut r = t;
        let mut k = 0;
        loop {
            let mut x = r.clone();
            for i in 0..n {
                x[i + i * n] -= T::C::one();
            }
            let norm = Matrix::from_column_major(n, n, x).norm(Norm::One).to_f64();
            if norm <= LOG_THETA || k >= 64 {
                break;
            }
            r = triangular_sqrt(&r, n)?;
            k += 1;
        }
        // `X = T^(1/2^k) - I`, with an accurate diagonal.
        let mut x = r;
        for i in 0..n {
            x[i + i * n] = root_minus_one(q.1[i], k);
        }
        // `r₇(X) = Σ w_j·(I + x_j·X)⁻¹·X`.
        let mut sum = vec![T::C::zero(); n * n];
        for &(node, weight) in &GAUSS {
            let node = T::Real::from_f64(node);
            let mut m = x.clone();
            for v in m.iter_mut() {
                *v = v.scale(node);
            }
            for i in 0..n {
                m[i + i * n] += T::C::one();
            }
            let mut y = x.clone();
            trsm(
                Side::Left,
                Uplo::Upper,
                Op::N,
                false,
                n,
                n,
                &m,
                n,
                &mut y,
                n,
            );
            let w = T::Real::from_f64(weight);
            for (s, y) in sum.iter_mut().zip(&y) {
                *s += y.scale(w);
            }
        }
        let two_k = T::Real::from_f64(2f64.powi(k));
        for s in &mut sum {
            *s = s.scale(two_k);
        }
        Some(self.back_from_schur(&sum, &q))
    }

    /// The complex Schur form `T` of the matrix and `Q`, and for later the eigenvalues as the
    /// original diagonal of `T`; `None` for a real matrix with a real eigenvalue `excluded`
    /// accepts.
    #[allow(clippy::type_complexity)]
    fn complex_schur(
        &self,
        excluded: impl Fn(crate::math::complex::Complex<T::Real>) -> bool,
    ) -> Option<(Vec<T::C>, (Vec<T::C>, Vec<T::C>))> {
        let n = self.rows();
        if self.data.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let (t, q, w) = schur(self, true, true);
        if !T::IS_COMPLEX && w.iter().any(|&w| w.im == T::Real::ZERO && excluded(w)) {
            return None;
        }
        let mut tc: Vec<T::C> = t.data.iter().map(|x| x.to_c()).collect();
        let mut qc: Vec<T::C> = q
            .unwrap_or_default()
            .data
            .iter()
            .map(|x| x.to_c())
            .collect();
        if !T::IS_COMPLEX {
            real_to_complex_schur(&mut tc, &mut qc, n);
        }
        let diagonal = (0..n).map(|i| tc[i + i * n]).collect();
        Some((tc, (qc, diagonal)))
    }

    /// `Q·R·Q*` in the matrix's own type: the real part for real elements.
    fn back_from_schur(&self, r: &[T::C], q: &(Vec<T::C>, Vec<T::C>)) -> Self {
        let n = self.rows();
        let q = &q.0;
        let mut qr = vec![T::C::zero(); n * n];
        gemm(
            n,
            n,
            n,
            T::C::one(),
            q,
            n,
            Op::N,
            r,
            n,
            Op::N,
            T::C::zero(),
            &mut qr,
            n,
        );
        let mut x = vec![T::C::zero(); n * n];
        gemm(
            n,
            n,
            n,
            T::C::one(),
            &qr,
            n,
            Op::N,
            q,
            n,
            Op::C,
            T::C::zero(),
            &mut x,
            n,
        );
        Matrix::from_column_major(n, n, x.into_iter().map(T::from_c).collect())
    }
}

/// The Padé approximant `(V - U)⁻¹·(V + U)` of `e^A` with the coefficients `b`, `U` its odd part
/// and `V` its even one, by the powers `A²`, `A⁴`, `A⁶`; for degree 13 by Higham's
/// evaluation in six products.
fn pade<T: Scalar>(a: &Matrix<T>, b: &[f64]) -> Matrix<T> {
    let n = a.rows();
    let c = |k: usize| T::from_real(T::Real::from_f64(b[k]));
    let identity = Matrix::<T>::identity(n);
    let a2 = a * a;
    let (u, v) = if b.len() == 14 {
        let a4 = &a2 * &a2;
        let a6 = &a4 * &a2;
        let inner_u = &(&(&a6 * c(13)) + &(&a4 * c(11))) + &(&a2 * c(9));
        let u = &(&(&(&a6 * &inner_u) + &(&a6 * c(7))) + &(&(&a4 * c(5)) + &(&a2 * c(3))))
            + &(&identity * c(1));
        let inner_v = &(&(&a6 * c(12)) + &(&a4 * c(10))) + &(&a2 * c(8));
        let v = &(&(&(&a6 * &inner_v) + &(&a6 * c(6))) + &(&(&a4 * c(4)) + &(&a2 * c(2))))
            + &(&identity * c(0));
        (a * &u, v)
    } else {
        // `Σ b_k·A^k`, the even powers by `A²` in turn.
        let mut odd = &identity * c(1);
        let mut even = &identity * c(0);
        let mut power = identity.clone();
        for k in 1..b.len() / 2 {
            power = &power * &a2;
            odd += &power * c(2 * k + 1);
            even += &power * c(2 * k);
        }
        (a * &odd, even)
    };
    (&v - &u)
        .solve(&(&v + &u))
        .unwrap_or_else(|| Matrix::from_fn(n, n, |_, _| T::from_parts(T::Real::NAN, T::Real::NAN)))
}

/// The square root of the upper triangular `t` with the principal roots on its diagonal, by
/// `R² = T` column by column, as Björck and Hammarling's method; `None` where `r_ii + r_jj` is
/// zero and the equation for `r_ij` has no solution.
fn triangular_sqrt<C: Scalar>(t: &[C], n: usize) -> Option<Vec<C>> {
    let at = |i: usize, j: usize| i + j * n;
    let mut r = vec![C::zero(); n * n];
    for j in 0..n {
        r[at(j, j)] = t[at(j, j)].sqrt();
        for i in (0..j).rev() {
            let mut s = t[at(i, j)];
            for k in i + 1..j {
                s -= r[at(i, k)] * r[at(k, j)];
            }
            let d = r[at(i, i)] + r[at(j, j)];
            if d == C::zero() {
                if s != C::zero() {
                    return None;
                }
                continue;
            }
            r[at(i, j)] = s / d;
        }
    }
    Some(r)
}

/// `a^(1/2^k) - 1` without the cancellation of the subtraction, as Al-Mohy and Higham's
/// `sqrt_power_1`: `(a - 1) / ((1 + a^(1/2))·(1 + a^(1/4))·…·(1 + a^(1/2^k)))`.
fn root_minus_one<C: Scalar>(a: C, k: i32) -> C {
    if k == 0 {
        return a - C::one();
    }
    let mut a = a;
    let mut k = k;
    // With an argument of at least π/2, `a - 1` itself cancels nothing, so one root first.
    if a.re() < C::Real::ZERO {
        a = a.sqrt();
        k -= 1;
        if k == 0 {
            return a - C::one();
        }
    }
    let z0 = a - C::one();
    a = a.sqrt();
    let mut r = C::one() + a;
    for _ in 1..k {
        a = a.sqrt();
        r *= C::one() + a;
    }
    z0 / r
}

//! The Schur decomposition: a blocked reduction to Hessenberg form, then the QR algorithm, in
//! real arithmetic for real matrices: for small ones (below order 120, or 90 for complex ones
//! with `Q`) Francis's double shifts (single shifts for complex ones), for larger ones the
//! small-bulge multishift QR algorithm with aggressive early deflation, most of its work in
//! products of blocks.

use super::householder::{reflection, reflector};
use super::properties::range_exponent;
use super::{Matrix, Scalar};
use crate::math::complex::{Complex, Float};

mod deflate;
mod hessenberg;
mod multishift;
mod vectors;

pub(crate) use hessenberg::hessenberg;
use multishift::multishift;
pub(crate) use vectors::eigenvectors;

/// The Schur decomposition `A = Q·T·Q*` of a square matrix, `Q` unitary (orthogonal for real
/// elements). For complex elements `T` is upper triangular, its diagonal the eigenvalues. For
/// real ones it is the real Schur form, quasi-upper triangular: blocks of `1 × 1` for the real
/// eigenvalues and of `2 × 2` for the pairs of complex conjugate ones, each `[[a, b], [c, a]]`
/// with `b·c < 0`, its eigenvalues `a ± i·√(-b·c)`. From [`Matrix::schur`].
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::matrix::Matrix;
///
/// // A rotation by a quarter turn: eigenvalues ±i, one 2 × 2 block.
/// let a: Matrix<f64> = Matrix::from_rows(&[[0.0, -1.0], [1.0, 0.0]]);
/// let s = a.schur();
/// let back = &(&s.q() * &s.t()) * &s.q().transpose();
/// assert!((&back - &a).as_slice().iter().all(|x| x.abs() < 1e-15));
/// let mut values = s.eigenvalues().to_vec();
/// values.sort_by(|a, b| a.im.total_cmp(&b.im));
/// assert_eq!(values, [Complex::new(0.0, -1.0), Complex::new(0.0, 1.0)]);
/// ```
#[derive(Clone, Debug)]
pub struct Schur<T: Scalar> {
    /// `Q`.
    q: Matrix<T>,
    /// `T`.
    t: Matrix<T>,
    /// The eigenvalues, in the order of the diagonal of `T`.
    values: Vec<Complex<T::Real>>,
}

impl<T: Scalar> Matrix<T> {
    /// The Schur decomposition `A = Q·T·Q*`: see [`Schur`]. A reduction to Hessenberg form by
    /// blocks of Householder reflections, then the QR algorithm with implicit shifts, for large
    /// matrices many at a time with aggressive early deflation, about `25n³` operations. A
    /// matrix with a NaN or an infinity gives NaNs.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn schur(&self) -> Schur<T> {
        assert!(self.is_square(), "the matrix is not square");
        let (t, q, values) = schur(self, true, true);
        Schur {
            q: q.unwrap_or_default(),
            t,
            values,
        }
    }

    /// The eigenvalues, in no particular order, those of a real matrix that are not real in
    /// pairs of conjugates: the diagonal of the Schur form, computed without `Q`, so about
    /// `10n³` operations. A matrix with a NaN or an infinity gives NaNs.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn eigenvalues(&self) -> Vec<Complex<T::Real>> {
        assert!(self.is_square(), "the matrix is not square");
        schur(self, false, false).2
    }
}

impl<T: Scalar> Schur<T> {
    /// `Q`, unitary.
    pub fn q(&self) -> Matrix<T> {
        self.q.clone()
    }

    /// `T`, upper triangular, or quasi-upper triangular for real elements.
    pub fn t(&self) -> Matrix<T> {
        self.t.clone()
    }

    /// The eigenvalues, in the order of the diagonal of `T`; a pair of complex conjugates of a
    /// real matrix, the one with the positive imaginary part first.
    pub fn eigenvalues(&self) -> &[Complex<T::Real>] {
        &self.values
    }
}

/// The Schur form `T` of `a`, with `Q` when `want_q` is set, and the eigenvalues; only the
/// eigenvalues are right in `T` when `want_t` is not set.
#[allow(clippy::type_complexity)]
pub(crate) fn schur<T: Scalar>(
    a: &Matrix<T>,
    want_t: bool,
    want_q: bool,
) -> (Matrix<T>, Option<Matrix<T>>, Vec<Complex<T::Real>>) {
    let n = a.rows();
    if a.data.iter().any(|x| !x.is_finite()) {
        let nan = T::from_parts(T::Real::NAN, T::Real::NAN);
        let all = Matrix::from_fn(n, n, |_, _| nan);
        let w = vec![Complex::new(T::Real::NAN, T::Real::NAN); n];
        return (all.clone(), want_q.then_some(all), w);
    }
    let mut h = a.clone();
    // Scaled by a power of two when the largest element is so small or so large that the
    // reduction or the QR algorithm could underflow or overflow, as LAPACK's `geev`: `T` and the
    // eigenvalues scaled back, exactly, and `Q` the same.
    let k = range_exponent(&h.data);
    if k != 0 {
        let up = T::Real::TWO.powi(k);
        for x in &mut h.data {
            *x = x.scale(up);
        }
    }
    let mut q = hessenberg(&mut h, want_q);
    let mut w = vec![Complex::new(T::Real::ZERO, T::Real::ZERO); n];
    let z = q.as_mut().map(|q| q.data.as_mut_slice());
    let info = multishift(want_t, &mut h.data, n, z, &mut w, false);
    w[..info].fill(Complex::new(T::Real::NAN, T::Real::NAN));
    if want_t && info == 0 {
        // Zeros below the subdiagonal, and for complex elements on it.
        for j in 0..n {
            let from = if T::IS_COMPLEX { j + 1 } else { j + 2 };
            for i in from..n {
                h.data[i + j * n] = T::zero();
            }
        }
    }
    if k != 0 {
        let down = T::Real::TWO.powi(-k);
        for x in &mut h.data {
            *x = x.scale(down);
        }
        for w in &mut w {
            *w = Complex::new(w.re * down, w.im * down);
        }
    }
    (h, q, w)
}

/// The QR algorithm with one or two shifts on the upper Hessenberg `h` of order `n`, as LAPACK's
/// `lahqr`: in real arithmetic, Francis's double shifts, for real elements. The eigenvalues
/// into `w`, `h` turned into the Schur form when `want_t` is set, the transformations
/// accumulated into `z` when given. Returns the number of leading eigenvalues it didn't
/// converge to, zero on success.
fn small_qr<T: Scalar>(
    h: &mut [T],
    n: usize,
    z: Option<&mut [T]>,
    want_t: bool,
    w: &mut [Complex<T::Real>],
) -> usize {
    if T::IS_COMPLEX {
        return complex_qr(h, n, z, want_t, w);
    }
    let mut hr: Vec<T::Real> = h.iter().map(|x| x.re()).collect();
    let mut zr: Option<Vec<T::Real>> = z.as_deref().map(|z| z.iter().map(|x| x.re()).collect());
    let info = real_qr(&mut hr, n, zr.as_deref_mut(), want_t, w);
    for (x, r) in h.iter_mut().zip(hr) {
        *x = T::from_real(r);
    }
    if let (Some(z), Some(zr)) = (z, zr) {
        for (x, r) in z.iter_mut().zip(zr) {
            *x = T::from_real(r);
        }
    }
    info
}

/// Every how many iterations without a deflation an exceptional shift is taken.
const EXCEPTIONAL: usize = 10;

/// `(c·x + s·y, c·y - s·x)`, a plane rotation.
#[inline]
fn rotate<R: Float>(x: &mut R, y: &mut R, c: R, s: R) {
    let t = c * *x + s * *y;
    *y = c * *y - s * *x;
    *x = t;
}

/// The QR algorithm with Francis's implicit double shifts on the real upper Hessenberg `h`,
/// as LAPACK's `lahqr`: the eigenvalues into `w`, `h` turned into the real Schur form when
/// `want_t` is set, and the rotations accumulated into `z` when given. Returns the number of
/// leading eigenvalues it didn't converge to.
fn real_qr<R: Float>(
    h: &mut [R],
    n: usize,
    mut z: Option<&mut [R]>,
    want_t: bool,
    w: &mut [Complex<R>],
) -> usize {
    let at = |i: usize, j: usize| i + j * n;
    if n == 0 {
        return 0;
    }
    let ulp = R::EPSILON;
    let smlnum = R::MIN_POSITIVE * (R::from_f64(n as f64) / ulp);
    let max_its = 30 * n.max(10);
    let mut kdefl = 0;
    // The active block is rows and columns `l..=i`.
    let mut i = n - 1;
    loop {
        let mut l = 0;
        let mut converged = false;
        for _ in 0..=max_its {
            // A negligible subdiagonal element splits the block.
            let mut k = i;
            while k > l {
                let sub = h[at(k, k - 1)].abs();
                if sub <= smlnum {
                    break;
                }
                let mut tst = h[at(k - 1, k - 1)].abs() + h[at(k, k)].abs();
                if tst == R::ZERO {
                    if k >= 2 {
                        tst += h[at(k - 1, k - 2)].abs();
                    }
                    if k + 1 < n {
                        tst += h[at(k + 1, k)].abs();
                    }
                }
                // Ahues and Tisseur's conservative test.
                if sub <= ulp * tst {
                    let up = h[at(k - 1, k)].abs();
                    let (ab, ba) = (sub.max(up), sub.min(up));
                    let d = (h[at(k - 1, k - 1)] - h[at(k, k)]).abs();
                    let hk = h[at(k, k)].abs();
                    let (aa, bb) = (hk.max(d), hk.min(d));
                    let s = aa + ab;
                    if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) {
                        break;
                    }
                }
                k -= 1;
            }
            l = k;
            if l > 0 {
                h[at(l, l - 1)] = R::ZERO;
            }
            if l + 1 >= i {
                converged = true;
                break;
            }
            kdefl += 1;
            let (i1, i2) = if want_t { (0, n - 1) } else { (l, i) };
            // The shifts: the eigenvalues of the trailing 2 × 2 block, or exceptional ones.
            let (h11, h12, h21, h22) = if kdefl % (2 * EXCEPTIONAL) == 0 {
                let s = h[at(i, i - 1)].abs() + h[at(i - 1, i - 2)].abs();
                let h11 = R::from_f64(0.75) * s + h[at(i, i)];
                (h11, R::from_f64(-0.4375) * s, s, h11)
            } else if kdefl % EXCEPTIONAL == 0 {
                let s = h[at(l + 1, l)].abs() + h[at(l + 2, l + 1)].abs();
                let h11 = R::from_f64(0.75) * s + h[at(l, l)];
                (h11, R::from_f64(-0.4375) * s, s, h11)
            } else {
                (
                    h[at(i - 1, i - 1)],
                    h[at(i - 1, i)],
                    h[at(i, i - 1)],
                    h[at(i, i)],
                )
            };
            let s = h11.abs() + h12.abs() + h21.abs() + h22.abs();
            let (rt1r, rt1i, rt2r, rt2i);
            if s == R::ZERO {
                (rt1r, rt1i, rt2r, rt2i) = (R::ZERO, R::ZERO, R::ZERO, R::ZERO);
            } else {
                let (h11, h12, h21, h22) = (h11 / s, h12 / s, h21 / s, h22 / s);
                let tr = (h11 + h22) * R::HALF;
                let det = (h11 - tr) * (h22 - tr) - h12 * h21;
                let rtdisc = det.abs().sqrt();
                if det >= R::ZERO {
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                } else {
                    // Two real shifts: both the one nearer `h22`.
                    let (a, b) = (tr + rtdisc, tr - rtdisc);
                    let r = if (a - h22).abs() <= (b - h22).abs() {
                        a * s
                    } else {
                        b * s
                    };
                    (rt1r, rt2r, rt1i, rt2i) = (r, r, R::ZERO, R::ZERO);
                }
            }
            // Where to start the sweep: two consecutive small subdiagonal elements.
            let mut m = i - 2;
            let mut v = [R::ZERO; 3];
            loop {
                let h21s = h[at(m + 1, m)];
                let s = (h[at(m, m)] - rt2r).abs() + rt2i.abs() + h21s.abs();
                let h21s = h21s / s;
                v[0] = h21s * h[at(m, m + 1)] + (h[at(m, m)] - rt1r) * ((h[at(m, m)] - rt2r) / s)
                    - rt1i * (rt2i / s);
                v[1] = h21s * (h[at(m, m)] + h[at(m + 1, m + 1)] - rt1r - rt2r);
                v[2] = h21s * h[at(m + 2, m + 1)];
                let s = v[0].abs() + v[1].abs() + v[2].abs();
                for x in &mut v {
                    *x /= s;
                }
                if m == l {
                    break;
                }
                let h00 = h[at(m, m - 1)].abs() * (v[1].abs() + v[2].abs());
                let h01 = v[0].abs()
                    * (h[at(m - 1, m - 1)].abs() + h[at(m, m)].abs() + h[at(m + 1, m + 1)].abs());
                if h00 <= ulp * h01 {
                    break;
                }
                m -= 1;
            }
            // The double-shift sweep, chasing the bulge down.
            for k in m..i {
                let nr = 3.min(i - k + 1);
                if k > m {
                    for p in 0..nr {
                        v[p] = h[at(k + p, k - 1)];
                    }
                }
                let (t1, beta) = {
                    let (first, rest) = v.split_at_mut(1);
                    real_reflector(first[0], &mut rest[..nr - 1])
                };
                v[0] = beta;
                if k > m {
                    h[at(k, k - 1)] = v[0];
                    h[at(k + 1, k - 1)] = R::ZERO;
                    if k + 2 < i + 1 {
                        h[at(k + 2, k - 1)] = R::ZERO;
                    }
                } else if m > l {
                    // Rather than negating it, which goes wrong when `v` underflows.
                    h[at(k, k - 1)] = h[at(k, k - 1)] * (R::ONE - t1);
                }
                let v2 = v[1];
                let t2 = t1 * v2;
                if nr == 3 {
                    let v3 = v[2];
                    let t3 = t1 * v3;
                    for j in k..=i2 {
                        let sum = h[at(k, j)] + v2 * h[at(k + 1, j)] + v3 * h[at(k + 2, j)];
                        h[at(k, j)] -= sum * t1;
                        h[at(k + 1, j)] -= sum * t2;
                        h[at(k + 2, j)] -= sum * t3;
                    }
                    for j in i1..=(k + 3).min(i) {
                        let sum = h[at(j, k)] + v2 * h[at(j, k + 1)] + v3 * h[at(j, k + 2)];
                        h[at(j, k)] -= sum * t1;
                        h[at(j, k + 1)] -= sum * t2;
                        h[at(j, k + 2)] -= sum * t3;
                    }
                    if let Some(z) = z.as_deref_mut() {
                        for j in 0..n {
                            let sum = z[at(j, k)] + v2 * z[at(j, k + 1)] + v3 * z[at(j, k + 2)];
                            z[at(j, k)] -= sum * t1;
                            z[at(j, k + 1)] -= sum * t2;
                            z[at(j, k + 2)] -= sum * t3;
                        }
                    }
                } else {
                    for j in k..=i2 {
                        let sum = h[at(k, j)] + v2 * h[at(k + 1, j)];
                        h[at(k, j)] -= sum * t1;
                        h[at(k + 1, j)] -= sum * t2;
                    }
                    for j in i1..=i {
                        let sum = h[at(j, k)] + v2 * h[at(j, k + 1)];
                        h[at(j, k)] -= sum * t1;
                        h[at(j, k + 1)] -= sum * t2;
                    }
                    if let Some(z) = z.as_deref_mut() {
                        for j in 0..n {
                            let sum = z[at(j, k)] + v2 * z[at(j, k + 1)];
                            z[at(j, k)] -= sum * t1;
                            z[at(j, k + 1)] -= sum * t2;
                        }
                    }
                }
            }
        }
        if !converged {
            return i + 1;
        }
        if l == i {
            w[i] = Complex::new(h[at(i, i)], R::ZERO);
        } else {
            // A 2 × 2 block: its standard form, and the rotation that gives it.
            let (a, b, c, d) = (
                h[at(i - 1, i - 1)],
                h[at(i - 1, i)],
                h[at(i, i - 1)],
                h[at(i, i)],
            );
            let s = standardize(a, b, c, d);
            h[at(i - 1, i - 1)] = s.a;
            h[at(i - 1, i)] = s.b;
            h[at(i, i - 1)] = s.c;
            h[at(i, i)] = s.d;
            w[i - 1] = Complex::new(s.rt1r, s.rt1i);
            w[i] = Complex::new(s.rt2r, s.rt2i);
            if want_t {
                for j in i + 1..n {
                    let (x, y) = (h[at(i - 1, j)], h[at(i, j)]);
                    let (mut x, mut y) = (x, y);
                    rotate(&mut x, &mut y, s.cs, s.sn);
                    h[at(i - 1, j)] = x;
                    h[at(i, j)] = y;
                }
                for j in 0..i - 1 {
                    let (mut x, mut y) = (h[at(j, i - 1)], h[at(j, i)]);
                    rotate(&mut x, &mut y, s.cs, s.sn);
                    h[at(j, i - 1)] = x;
                    h[at(j, i)] = y;
                }
            }
            if let Some(z) = z.as_deref_mut() {
                for j in 0..n {
                    let (mut x, mut y) = (z[at(j, i - 1)], z[at(j, i)]);
                    rotate(&mut x, &mut y, s.cs, s.sn);
                    z[at(j, i - 1)] = x;
                    z[at(j, i)] = y;
                }
            }
        }
        kdefl = 0;
        if l == 0 {
            break;
        }
        i = l - 1;
    }
    0
}

/// The real reflection `I - τ·v·vᵀ`, `v = (1, x)`, taking `(alpha, x)` to `(β, 0)`: overwrites
/// `x` with the rest of `v` and returns `(τ, β)`.
fn real_reflector<R: Float>(alpha: R, x: &mut [R]) -> (R, R) {
    let big = x.iter().fold(alpha.abs(), |m, v| m.max(v.abs()));
    if moderate(big) {
        let squares = x.iter().fold(R::ZERO, |s, v| s + *v * *v);
        if squares == R::ZERO {
            return (R::ZERO, alpha);
        }
        let beta = -(alpha * alpha + squares).sqrt().copysign(alpha);
        return (real_reflection(alpha, beta, x), beta);
    }
    let mut scale = R::ZERO;
    for v in x.iter() {
        scale = scale.max(v.abs());
    }
    if scale == R::ZERO {
        return (R::ZERO, alpha);
    }
    let mut s = R::ZERO;
    for v in x.iter() {
        let t = *v / scale;
        s += t * t;
    }
    let xnorm = s.sqrt() * scale;
    let beta = -alpha.hypot(xnorm).copysign(alpha);
    (real_reflection(alpha, beta, x), beta)
}

/// `τ`, with `x` scaled to the rest of `v`, of the real reflection taking `(alpha, x)` to
/// `(β, 0)`: [`reflection`](super::householder::reflection) for real numbers, without its
/// overflows near the largest float.
fn real_reflection<R: Float>(alpha: R, beta: R, x: &mut [R]) -> R {
    let d = alpha - beta;
    if !d.is_finite() {
        let r = R::ONE / (alpha * R::HALF - beta * R::HALF);
        for v in x.iter_mut() {
            *v = *v * r * R::HALF;
        }
    } else {
        let r = R::ONE / d;
        if r.is_finite() {
            for v in x.iter_mut() {
                *v *= r;
            }
        } else {
            for v in x.iter_mut() {
                *v /= d;
            }
        }
    }
    R::ONE - alpha / beta
}

/// A real 2 × 2 block in standard form, its eigenvalues and the rotation to it.
struct Standard<R> {
    a: R,
    b: R,
    c: R,
    d: R,
    rt1r: R,
    rt1i: R,
    rt2r: R,
    rt2i: R,
    cs: R,
    sn: R,
}

/// The Schur form of the real block `[[a, b], [c, d]]`, as LAPACK's `lanv2`: upper triangular
/// for real eigenvalues, `[[a, b], [c, a]]` with `b·c < 0` for complex ones, by the rotation
/// `[[cs, -sn], [sn, cs]]`.
fn standardize<R: Float>(mut a: R, mut b: R, mut c: R, mut d: R) -> Standard<R> {
    let (mut cs, mut sn) = (R::ONE, R::ZERO);
    let sign = |x: R, y: R| x.abs().copysign(y);
    if c == R::ZERO {
    } else if b == R::ZERO {
        // Swap the rows and the columns.
        cs = R::ZERO;
        sn = R::ONE;
        core::mem::swap(&mut a, &mut d);
        b = -c;
        c = R::ZERO;
    } else if a - d == R::ZERO && b.is_sign_negative() != c.is_sign_negative() {
    } else {
        let temp = a - d;
        let mut p = temp * R::HALF;
        let bcmax = b.abs().max(c.abs());
        let bcmis = b.abs().min(c.abs()) * sign(R::ONE, b) * sign(R::ONE, c);
        let scale = p.abs().max(bcmax);
        let mut z = (p / scale) * p + (bcmax / scale) * bcmis;
        if z >= R::from_f64(4.0) * R::EPSILON {
            // Real eigenvalues: `z` is their half difference.
            z = p + sign(scale.sqrt() * z.sqrt(), p);
            a = d + z;
            d -= (bcmax / z) * bcmis;
            let tau = c.hypot(z);
            cs = z / tau;
            sn = c / tau;
            b -= c;
            c = R::ZERO;
        } else {
            // Complex or nearly equal eigenvalues: equal diagonal elements.
            let sigma = b + c;
            let tau = sigma.hypot(temp);
            cs = (R::HALF * (R::ONE + sigma.abs() / tau)).sqrt();
            sn = -(temp * R::HALF / (tau * cs)) * sign(R::ONE, sigma);
            let aa = a * cs + b * sn;
            let bb = -a * sn + b * cs;
            let cc = c * cs + d * sn;
            let dd = -c * sn + d * cs;
            a = aa * cs + cc * sn;
            b = bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;
            let temp = R::HALF * (a + d);
            a = temp;
            d = temp;
            if c != R::ZERO {
                if b != R::ZERO {
                    if b.is_sign_negative() == c.is_sign_negative() {
                        // Real eigenvalues after all: upper triangular.
                        let sab = b.abs().sqrt();
                        let sac = c.abs().sqrt();
                        p = sign(sab * sac, c);
                        let tau = R::ONE / (b + c).abs().sqrt();
                        a = temp + p;
                        d = temp - p;
                        b -= c;
                        c = R::ZERO;
                        let cs1 = sab * tau;
                        let sn1 = sac * tau;
                        let t = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = t;
                    }
                } else {
                    b = -c;
                    c = R::ZERO;
                    let t = cs;
                    cs = -sn;
                    sn = t;
                }
            }
        }
    }
    let (rt1i, rt2i) = if c == R::ZERO {
        (R::ZERO, R::ZERO)
    } else {
        let r = b.abs().sqrt() * c.abs().sqrt();
        (r, -r)
    };
    Standard {
        a,
        b,
        c,
        d,
        rt1r: a,
        rt1i,
        rt2r: d,
        rt2i,
        cs,
        sn,
    }
}

/// `|re| + |im|`.
#[inline]
fn abs1<C: Scalar>(x: C) -> C::Real {
    x.re().abs() + x.im().abs()
}

/// Whether the squares of numbers up to `big`, and their sums, neither overflow nor underflow
/// harmfully.
#[inline]
fn moderate<R: Float>(big: R) -> bool {
    let (low, high) = if R::MANTISSA_DIGITS > 24 {
        (1e-130, 1e130)
    } else {
        (1e-15, 1e15)
    };
    R::from_f64(low) < big && big < R::from_f64(high)
}

/// [`reflector`] for the short vectors of the QR sweeps' bulges: the squares summed directly
/// when they are [`moderate`].
#[inline]
fn short_reflector<C: Scalar>(alpha: C, x: &mut [C]) -> (C, C) {
    let zero = C::Real::ZERO;
    let big = x
        .iter()
        .fold(alpha.re().abs().max(alpha.im().abs()), |m, v| {
            m.max(v.re().abs()).max(v.im().abs())
        });
    if !moderate(big) {
        return reflector(alpha, x);
    }
    let squares = x.iter().fold(zero, |s, v| s + v.abs2());
    if squares == zero && alpha.im() == zero {
        return (C::zero(), alpha);
    }
    reflection(alpha, (alpha.abs2() + squares).sqrt(), x)
}

/// The QR algorithm with single implicit shifts on the complex upper Hessenberg `h`, as
/// LAPACK's `lahqr` for complex matrices: the eigenvalues into `w`, `h` turned into the Schur
/// form when `want_t` is set, the transformations accumulated into `z` when given. Returns the
/// number of leading eigenvalues it didn't converge to.
fn complex_qr<C: Scalar>(
    h: &mut [C],
    n: usize,
    mut z: Option<&mut [C]>,
    want_t: bool,
    w: &mut [Complex<C::Real>],
) -> usize {
    let at = |i: usize, j: usize| i + j * n;
    let zero = C::Real::ZERO;
    if n == 0 {
        return 0;
    }
    // Real subdiagonal elements, by a diagonal unitary similarity.
    for i in 1..n {
        let x = h[at(i, i - 1)];
        if x.im() != zero {
            let a = x.abs();
            let sc = x.conj().scale(C::Real::ONE / a);
            h[at(i, i - 1)] = C::from_real(a);
            let jhi = n - 1;
            for j in i..=jhi {
                h[at(i, j)] *= sc;
            }
            for j in 0..=(i + 1).min(jhi) {
                h[at(j, i)] *= sc.conj();
            }
            if let Some(z) = z.as_deref_mut() {
                for j in 0..n {
                    z[at(j, i)] *= sc.conj();
                }
            }
        }
    }
    let ulp = C::Real::EPSILON;
    let smlnum = C::Real::MIN_POSITIVE * (C::Real::from_f64(n as f64) / ulp);
    let max_its = 30 * n.max(10);
    let mut kdefl = 0;
    let mut i = n - 1;
    loop {
        let mut l = 0;
        let mut converged = false;
        for _ in 0..=max_its {
            let mut k = i;
            while k > l {
                let sub = abs1(h[at(k, k - 1)]);
                if sub <= smlnum {
                    break;
                }
                let mut tst = abs1(h[at(k - 1, k - 1)]) + abs1(h[at(k, k)]);
                if tst == zero {
                    if k >= 2 {
                        tst += h[at(k - 1, k - 2)].re().abs();
                    }
                    if k + 1 < n {
                        tst += h[at(k + 1, k)].re().abs();
                    }
                }
                if h[at(k, k - 1)].re().abs() <= ulp * tst {
                    let up = abs1(h[at(k - 1, k)]);
                    let (ab, ba) = (sub.max(up), sub.min(up));
                    let d = abs1(h[at(k - 1, k - 1)] - h[at(k, k)]);
                    let hk = abs1(h[at(k, k)]);
                    let (aa, bb) = (hk.max(d), hk.min(d));
                    let s = aa + ab;
                    if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) {
                        break;
                    }
                }
                k -= 1;
            }
            l = k;
            if l > 0 {
                h[at(l, l - 1)] = C::zero();
            }
            if l >= i {
                converged = true;
                break;
            }
            kdefl += 1;
            let (i1, i2) = if want_t { (0, n - 1) } else { (l, i) };
            // The shift: Wilkinson's, or an exceptional one.
            let t = if kdefl % (2 * EXCEPTIONAL) == 0 {
                let s = C::Real::from_f64(0.75) * h[at(i, i - 1)].re().abs();
                h[at(i, i)] + C::from_real(s)
            } else if kdefl % EXCEPTIONAL == 0 {
                let s = C::Real::from_f64(0.75) * h[at(l + 1, l)].re().abs();
                h[at(l, l)] + C::from_real(s)
            } else {
                let mut t = h[at(i, i)];
                let u = h[at(i - 1, i)].sqrt() * h[at(i, i - 1)].sqrt();
                let s = abs1(u);
                if s != zero {
                    let x = (h[at(i - 1, i - 1)] - t).scale(C::Real::HALF);
                    let sx = abs1(x);
                    let s = s.max(sx);
                    let (xs, us) = (x.scale(C::Real::ONE / s), u.scale(C::Real::ONE / s));
                    let mut y = (xs * xs + us * us).sqrt().scale(s);
                    if sx > zero {
                        let xr = x.scale(C::Real::ONE / sx);
                        if xr.re() * y.re() + xr.im() * y.im() < zero {
                            y = -y;
                        }
                    }
                    t -= u * (u / (x + y));
                }
                t
            };
            // Where to start: two consecutive small subdiagonal elements.
            let mut m = i - 1;
            let mut v: [C; 2];
            loop {
                let h11 = h[at(m, m)];
                let h22 = h[at(m + 1, m + 1)];
                let h11s = h11 - t;
                let h21 = h[at(m + 1, m)].re();
                let s = abs1(h11s) + h21.abs();
                let (h11s, h21) = (h11s.scale(C::Real::ONE / s), h21 / s);
                v = [h11s, C::from_real(h21)];
                if m == l {
                    break;
                }
                let h10 = h[at(m, m - 1)].re();
                if h10.abs() * h21.abs() <= ulp * (abs1(h11s) * (abs1(h11) + abs1(h22))) {
                    break;
                }
                m -= 1;
            }
            // The single-shift sweep.
            for k in m..i {
                if k > m {
                    v = [h[at(k, k - 1)], h[at(k + 1, k - 1)]];
                }
                let (t1, beta) = {
                    let (first, rest) = v.split_at_mut(1);
                    short_reflector(first[0], rest)
                };
                v[0] = beta;
                if k > m {
                    h[at(k, k - 1)] = v[0];
                    h[at(k + 1, k - 1)] = C::zero();
                }
                let v2 = v[1];
                let t2 = (t1 * v2).re();
                let (t1c, v2c) = (t1.conj(), v2.conj());
                for col in h[k * n..(i2 + 1) * n].chunks_exact_mut(n) {
                    let [x, y] = &mut col[k..k + 2] else {
                        unreachable!()
                    };
                    let sum = t1c * *x + y.scale(t2);
                    *x -= sum;
                    *y -= sum * v2;
                }
                let turn = |a: &mut [C], rows: core::ops::Range<usize>| {
                    let (x, y) = a[k * n..].split_at_mut(n);
                    for (x, y) in x[rows.clone()].iter_mut().zip(&mut y[rows]) {
                        let sum = t1 * *x + y.scale(t2);
                        *x -= sum;
                        *y -= sum * v2c;
                    }
                };
                turn(h, i1..(k + 2).min(i) + 1);
                if let Some(z) = z.as_deref_mut() {
                    turn(z, 0..n);
                }
                if k == m && m > l {
                    // Keep `h[m, m - 1]` real, the sweep having started below it.
                    let mut temp = C::one() - t1;
                    temp = temp.scale(C::Real::ONE / temp.abs());
                    h[at(m + 1, m)] *= temp.conj();
                    if m + 2 <= i {
                        h[at(m + 2, m + 1)] *= temp;
                    }
                    for j in m..=i {
                        if j != m + 1 {
                            if i2 > j {
                                for c in j + 1..=i2 {
                                    h[at(j, c)] *= temp;
                                }
                            }
                            for r in i1..j {
                                h[at(r, j)] *= temp.conj();
                            }
                            if let Some(z) = z.as_deref_mut() {
                                for r in 0..n {
                                    z[at(r, j)] *= temp.conj();
                                }
                            }
                        }
                    }
                }
            }
            // Keep `h[i, i - 1]` real.
            let temp = h[at(i, i - 1)];
            if temp.im() != zero {
                let r = temp.abs();
                h[at(i, i - 1)] = C::from_real(r);
                let temp = temp.scale(C::Real::ONE / r);
                if i2 > i {
                    for c in i + 1..=i2 {
                        h[at(i, c)] *= temp.conj();
                    }
                }
                for r in i1..i {
                    h[at(r, i)] *= temp;
                }
                if let Some(z) = z.as_deref_mut() {
                    for r in 0..n {
                        z[at(r, i)] *= temp;
                    }
                }
            }
        }
        if !converged {
            return i + 1;
        }
        w[i] = h[at(i, i)].to_complex();
        kdefl = 0;
        if l == 0 {
            break;
        }
        i = l - 1;
    }
    0
}

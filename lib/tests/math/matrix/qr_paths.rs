//! QR decompositions across the thresholds of the algorithm: unblocked for small matrices, and
//! blocked, the width of the blocks by the shape and the type, the panels halved down to their
//! leaves; `Q` formed one reflection at a time or by blocks; the least squares by one reflection
//! at a time or by blocks, by substitution or a triangular solve; and hard cases: zero and
//! dependent columns, scales whose squares overflow or underflow, NaNs and infinities. Each is
//! checked by what defines it and against the textbook decomposition.

use super::common::{
    Num, Rng, assert_close, assert_orthonormal, each_type, frobenius, random, test_each_type,
};
use a::math::complex::Complex;
use a::math::matrix::Matrix;

/// The textbook Householder QR of `a`, one reflection at a time as LAPACK's `geqr2` and `org2r`,
/// in `Complex<f64>`: `(Q, R)`, `Q` of `m × k` and `R` of `k × n`, `k = min(m, n)`, the same
/// reflections as the decomposition's, `β = -sign(Re α)·‖(α, x)‖`.
fn textbook<T: Num>(a: &Matrix<T>) -> (Matrix<Complex<f64>>, Matrix<Complex<f64>>) {
    let (m, n) = a.shape();
    let k = m.min(n);
    let mut x = Matrix::from_fn(m, n, |i, j| a[(i, j)].c64());
    let mut taus = vec![Complex::new(0.0, 0.0); k];
    for j in 0..k {
        let alpha = x[(j, j)];
        let xnorm = (j + 1..m).map(|i| x[(i, j)].norm_sqr()).sum::<f64>().sqrt();
        if xnorm == 0.0 && alpha.im == 0.0 {
            continue;
        }
        let norm = alpha.abs().hypot(xnorm);
        let beta = if alpha.re >= 0.0 { -norm } else { norm };
        let tau = Complex::new((beta - alpha.re) / beta, -alpha.im / beta);
        let scale = Complex::new(1.0, 0.0) / (alpha - Complex::new(beta, 0.0));
        for i in j + 1..m {
            x[(i, j)] *= scale;
        }
        x[(j, j)] = Complex::new(beta, 0.0);
        taus[j] = tau;
        for c in j + 1..n {
            let mut w = x[(j, c)];
            for i in j + 1..m {
                w += x[(i, j)].conj() * x[(i, c)];
            }
            let w = tau.conj() * w;
            x[(j, c)] -= w;
            for i in j + 1..m {
                let d = x[(i, j)] * w;
                x[(i, c)] -= d;
            }
        }
    }
    let r = Matrix::from_fn(k, n, |i, j| {
        if i <= j {
            x[(i, j)]
        } else {
            Complex::new(0.0, 0.0)
        }
    });
    let mut q = Matrix::from_fn(m, k, |i, j| {
        Complex::new(if i == j { 1.0 } else { 0.0 }, 0.0)
    });
    for j in (0..k).rev() {
        for c in j..k {
            let mut w = q[(j, c)];
            for i in j + 1..m {
                w += x[(i, j)].conj() * q[(i, c)];
            }
            let w = taus[j] * w;
            q[(j, c)] -= w;
            for i in j + 1..m {
                q[(i, c)] -= x[(i, j)] * w;
            }
        }
    }
    (q, r)
}

/// `‖a - b‖` in the Frobenius norm, of a matrix of the type against one of `Complex<f64>`.
fn distance<T: Num>(a: &Matrix<T>, b: &Matrix<Complex<f64>>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    let (m, n) = a.shape();
    let mut s = 0.0;
    for j in 0..n {
        for i in 0..m {
            s += (a[(i, j)].c64() - b[(i, j)]).norm_sqr();
        }
    }
    s.sqrt()
}

/// Checks the decomposition of `a`: the shapes, `R` upper triangular with a real diagonal, `Q`
/// and the whole `Q` orthonormal, the first the leading columns of the second, `Q·R = A`.
#[track_caller]
fn check<T: Num>(a: &Matrix<T>, tol: f64) {
    let (m, n) = a.shape();
    let k = m.min(n);
    let qr = a.qr();
    let (q, r) = (qr.q(), qr.r());
    assert_eq!((q.shape(), r.shape()), ((m, k), (k, n)));
    for i in 0..k {
        assert_eq!(r[(i, i)].c64().im, 0.0, "{m}×{n}");
        for j in 0..i {
            assert_eq!(r[(i, j)], T::real(0.0), "{m}×{n}");
        }
    }
    assert_orthonormal(&q, tol);
    // The whole `Q` of a very tall matrix would take long to check.
    if m <= 256 {
        let full = qr.q_full();
        assert_eq!(full.shape(), (m, m));
        assert_orthonormal(&full, tol);
        assert_close(&Matrix::from_fn(m, k, |i, j| full[(i, j)]), &q, tol);
    }
    let scale = frobenius(a);
    let d = frobenius(&(&(&q * &r) - a));
    assert!(
        d <= tol * T::EPS * (scale + f64::MIN_POSITIVE),
        "‖QR - A‖ = {d:e}, ‖A‖ = {scale:e}, {m}×{n}"
    );
}

/// A random `m × n` matrix far from rank deficient: random plus `2·min(m, n)` on the diagonal,
/// whose decomposition the rounding moves by little.
fn far_from_deficient<T: Num>(r: &mut Rng, m: usize, n: usize) -> Matrix<T> {
    let mut a = random::<T>(r, m, n);
    for i in 0..m.min(n) {
        a[(i, i)] += T::real(2.0 * m.min(n) as f64);
    }
    a
}

/// Checks the decomposition of `a`, far from rank deficient, also against the textbook one.
#[track_caller]
fn check_textbook<T: Num>(a: &Matrix<T>) {
    let (m, n) = a.shape();
    let tol = 16.0 * (m + n) as f64;
    check(a, tol);
    let qr = a.qr();
    let (q, r) = textbook(a);
    let scale = frobenius(a);
    let dr = distance(&qr.r(), &r);
    assert!(
        dr <= tol * T::EPS * scale,
        "‖R - R₀‖ = {dr:e}, ‖A‖ = {scale:e}, {m}×{n}"
    );
    let dq = distance(&qr.q(), &q);
    let k = m.min(n) as f64;
    assert!(dq <= tol * T::EPS * k.sqrt(), "‖Q - Q₀‖ = {dq:e}, {m}×{n}");
}

/// Whether Miri takes the shapes past the blocked path's threshold: on 64-bit targets only. On a
/// 32-bit one, where Miri never reuses stack addresses, they and their reference decomposition
/// run out of the address space; its small shapes are checked there, and its blocked products
/// by the product tests.
const LARGE_UNDER_MIRI: bool = cfg!(target_pointer_width = "64");

/// Shapes on both sides of the thresholds of every type, as the decomposition picks its path:
/// unblocked up to 4096 elements (and complex ones of at most 64 columns up to 32768), then
/// blocks of 8 columns, the panels' leaves; partial last blocks, wide and tall matrices.
fn shapes() -> Vec<(usize, usize)> {
    if cfg!(miri) {
        let mut s = vec![
            (0, 0),
            (0, 3),
            (3, 0),
            (1, 1),
            (1, 5),
            (5, 1),
            (2, 2),
            (7, 4),
            (4, 7),
        ];
        if LARGE_UNDER_MIRI {
            s.extend([(65, 64), (66, 65)]);
        }
        return s;
    }
    let mut s = vec![
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (1, 9),
        (9, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    ];
    s.extend([
        (64, 64),
        (64, 65),
        (65, 64),
        (4096, 1),
        (1, 4097),
        (2, 2049),
        (128, 32),
        (129, 32),
    ]);
    s.extend([(512, 64), (513, 64), (500, 65), (64, 512)]);
    s.extend([
        (96, 96),
        (97, 97),
        (100, 75),
        (75, 100),
        (200, 130),
        (130, 400),
        (40, 300),
    ]);
    s
}

/// The type's shapes on both sides of where its blocks widen to 16, 24 and 32 columns, at
/// 384, 640 and 896 columns of `f64`, half as many of `f32` and a quarter as many complex ones.
fn wider_blocks<T: Num>() -> Vec<(usize, usize)> {
    if cfg!(miri) {
        return vec![];
    }
    let complex = core::any::type_name::<T>().contains("Complex");
    match (complex, T::EPS > 1e-10) {
        (true, _) => vec![
            (96, 96),
            (97, 97),
            (160, 160),
            (161, 161),
            (224, 224),
            (225, 225),
        ],
        (false, true) => vec![(192, 192), (193, 193)],
        (false, false) => vec![],
    }
}

fn shapes_of<T: Num>() {
    let mut r = Rng(0x5EED_0401);
    for (m, n) in shapes().into_iter().chain(wider_blocks::<T>()) {
        let a = far_from_deficient::<T>(&mut r, m, n);
        // The textbook decomposition takes long for the larger ones, unoptimized.
        if m * n * m.min(n) <= 1 << 20 {
            check_textbook(&a);
        } else {
            check(&a, 16.0 * (m + n) as f64);
        }
        if m * n <= 1 << 15 {
            let a = random::<T>(&mut r, m, n);
            check(&a, 16.0 * (m + n) as f64);
        }
    }
}

test_each_type!(shapes_across_thresholds, shapes_of);

#[test]
fn tall_blocks() {
    // The wider blocks of tall real matrices, four times as many rows as columns: 16 columns
    // from 512 rows, 48 from 2048 and 64 from 4096; and the widest by the columns, in `f64`.
    let mut r = Rng(0x5EED_0402);
    let shapes = if cfg!(miri) {
        vec![]
    } else {
        vec![
            (511, 80),
            (512, 80),
            (2047, 40),
            (2048, 40),
            (4095, 30),
            (4096, 30),
            (384, 384),
            (385, 385),
        ]
    };
    for (m, n) in shapes {
        check(
            &far_from_deficient::<f64>(&mut r, m, n),
            16.0 * (m + n) as f64,
        );
    }
}

/// Checks `X` of the least squares `‖A·X - B‖`: the residual orthogonal to the columns of `A`.
#[track_caller]
fn check_least_squares<T: Num>(a: &Matrix<T>, b: &Matrix<T>) {
    let (m, n) = a.shape();
    let x = a.qr().solve(b).unwrap();
    assert_eq!(x.shape(), (n, b.cols()));
    let residual = b - &(a * &x);
    let g = &a.conjugate_transpose() * &residual;
    let scale = frobenius(a) * frobenius(b) + 1.0;
    assert!(
        frobenius(&g) <= 64.0 * (m + 1) as f64 * T::EPS * scale,
        "{m}×{n}, {} columns",
        b.cols()
    );
}

fn least_squares_of<T: Num>() {
    let mut r = Rng(0x5EED_0403);
    // Unblocked and blocked decompositions, the right-hand side's columns from none to a few
    // over where the reflections go four at a time for each column, 4, and `R` of up to 128
    // columns and more, solved by substitution and by the triangular solve.
    let shapes: &[(usize, usize)] = if cfg!(miri) {
        &[(5, 3), (70, 66)]
    } else {
        &[
            (5, 3),
            (40, 40),
            (100, 30),
            (64, 64),
            (65, 65),
            (150, 100),
            (300, 40),
            (600, 100),
            (200, 128),
            (200, 129),
        ]
    };
    let cols: &[usize] = if cfg!(miri) {
        &[0, 1, 5]
    } else {
        &[0, 1, 2, 3, 4, 5, 9]
    };
    for &(m, n) in shapes {
        let a = random::<T>(&mut r, m, n);
        for &c in cols {
            let b = random::<T>(&mut r, m, c);
            check_least_squares(&a, &b);
        }
    }
    // A square system, solved.
    let a = far_from_deficient::<T>(&mut r, 50, 50);
    let x = random::<T>(&mut r, 50, 3);
    assert_close(&a.qr().solve(&(&a * &x)).unwrap(), &x, 64.0 * 50.0);
}

#[test]
fn least_squares() {
    each_type!(least_squares_of);
}

fn deficient_of<T: Num>() {
    let mut r = Rng(0x5EED_0404);
    let sizes: &[(usize, usize)] = if cfg!(miri) && !LARGE_UNDER_MIRI {
        &[(6, 4)]
    } else if cfg!(miri) {
        &[(6, 4), (70, 66)]
    } else {
        &[(6, 4), (40, 40), (70, 66), (150, 100), (100, 150)]
    };
    for &(m, n) in sizes {
        // Zero columns, first, inside and last: a zero on the diagonal of `R`, and no least
        // squares.
        for j in [0, n / 2, n - 1] {
            let mut a = random::<T>(&mut r, m, n);
            for i in 0..m {
                a[(i, j)] = T::real(0.0);
            }
            check(&a, 16.0 * (m + n) as f64);
            if j < m && m >= n {
                assert_eq!(a.qr().r()[(j, j)], T::real(0.0));
                assert!(a.qr().solve(&random::<T>(&mut r, m, 1)).is_none());
            }
        }
        // A zero matrix: `R = 0`, `Q = I`.
        let z = Matrix::<T>::zeros(m, n);
        let qr = z.qr();
        assert_eq!(qr.r(), Matrix::zeros(m.min(n), n));
        assert_eq!(qr.q_full(), Matrix::identity(m));
        // Rank one, and repeated columns.
        let u = random::<T>(&mut r, m, 1);
        let v = random::<T>(&mut r, 1, n);
        check(&(&u * &v), 16.0 * (m + n) as f64);
        let mut a = random::<T>(&mut r, m, n);
        for i in 0..m {
            a[(i, n - 1)] = a[(i, 0)];
        }
        check(&a, 16.0 * (m + n) as f64);
    }
}

test_each_type!(zero_and_dependent_columns, deficient_of);

fn scaled_of<T: Num>() {
    // Scaled by powers of two so far that the squares of the elements overflow or underflow:
    // the same decomposition, scaled; and columns of both scales together.
    let mut r = Rng(0x5EED_0405);
    let e = if T::EPS > 1e-10 { 70 } else { 520 };
    let sizes: &[(usize, usize)] = if cfg!(miri) {
        &[(7, 5), (70, 66)]
    } else {
        &[(7, 5), (40, 40), (70, 66), (150, 100), (60, 120)]
    };
    for &(m, n) in sizes {
        let a = far_from_deficient::<T>(&mut r, m, n);
        let (q0, r0) = (a.qr().q(), a.qr().r());
        for k in [e, -e] {
            let s = T::real(2f64.powi(k));
            let b = a.map(|&x| x * s);
            let qr = b.qr();
            assert_close(&qr.q(), &q0, 16.0 * (m + n) as f64);
            assert_close(
                &qr.r().map(|&x| x * T::real(2f64.powi(-k))),
                &r0,
                16.0 * (m + n) as f64,
            );
        }
        let mut c = random::<T>(&mut r, m, n);
        for j in 0..n {
            let s = T::real(2f64.powi(if j % 2 == 0 { e } else { -e }));
            for i in 0..m {
                c[(i, j)] *= s;
            }
        }
        let qr = c.qr();
        assert_orthonormal(&qr.q(), 16.0 * (m + n) as f64);
        assert!(qr.r().as_slice().iter().all(|x| x.c64().re.is_finite()));
    }
}

test_each_type!(scaled_to_the_ends_of_the_range, scaled_of);

fn non_finite_of<T: Num>() {
    let mut r = Rng(0x5EED_0406);
    let sizes: &[(usize, usize)] = if cfg!(miri) {
        &[(3, 3), (70, 66)]
    } else {
        &[
            (3, 3),
            (50, 50),
            (70, 66),
            (150, 100),
            (100, 150),
            (600, 40),
        ]
    };
    for &(m, n) in sizes {
        for bad in [f64::NAN, f64::INFINITY] {
            let mut a = random::<T>(&mut r, m, n);
            a[(0, 0)] = T::real(bad);
            let qr = a.qr();
            // The first column's reflection takes the NaN or the infinity into everything.
            let nan = |x: &T| x.c64().re.is_nan() || x.c64().im.is_nan();
            assert!(qr.r().as_slice().iter().any(nan), "{m}×{n}");
            assert!(qr.q().as_slice().iter().any(nan), "{m}×{n}");
            if m >= n {
                let x = qr.solve(&random::<T>(&mut r, m, 2)).unwrap();
                assert!(x.as_slice().iter().any(nan), "{m}×{n}");
            }
        }
    }
}

#[test]
fn nan_and_infinity() {
    each_type!(non_finite_of);
}

fn full_q_of<T: Num>() {
    // The whole `Q` of tall matrices, formed in one block from 48 rows and four times as many
    // rows as columns: its factor joined from blocks of 64 for more columns. `Q*·A = [R; 0]`.
    let mut r = Rng(0x5EED_0407);
    let shapes: &[(usize, usize)] = if cfg!(miri) {
        &[(47, 4), (48, 4), (70, 17)]
    } else {
        &[
            (47, 4),
            (48, 4),
            (127, 32),
            (128, 32),
            (290, 70),
            (300, 64),
            (530, 130),
        ]
    };
    for &(m, n) in shapes {
        let a = far_from_deficient::<T>(&mut r, m, n);
        let qr = a.qr();
        let (q, full) = (qr.q(), qr.q_full());
        let tol = 16.0 * (m + n) as f64;
        assert_orthonormal(&full, tol);
        assert_close(&Matrix::from_fn(m, n, |i, j| full[(i, j)]), &q, tol);
        let mut rr = Matrix::<T>::zeros(m, n);
        let r = qr.r();
        for j in 0..n {
            for i in 0..=j {
                rr[(i, j)] = r[(i, j)];
            }
        }
        assert_close(&(&full.conjugate_transpose() * &a), &rr, tol);
        let (q0, _) = textbook(&a);
        let dq = distance(&q, &q0);
        assert!(
            dq <= tol * T::EPS * (n as f64).sqrt(),
            "‖Q - Q₀‖ = {dq:e}, {m}×{n}"
        );
    }
}

#[test]
fn full_q_of_tall_matrices() {
    each_type!(full_q_of);
}

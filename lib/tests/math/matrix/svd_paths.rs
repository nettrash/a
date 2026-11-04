//! Singular value decompositions across the thresholds of the algorithms: divide and conquer and
//! its leaves, the panels of the reduction, QR first for tall matrices; and hard cases: low rank,
//! repeated, clustered, graded and tiny singular values, zeros.

use super::common::{
    Num, Rng, assert_close, assert_orthonormal, each_type, frobenius, random, test_each_type,
};
use a::math::complex::Float;
use a::math::matrix::Matrix;

/// A real value as an `f64`.
fn f<R: Float>(x: R) -> f64 {
    x.to_f64()
}

/// Checks the decomposition of `a`: the shapes, `U` and `V` orthonormal, `U·Σ·V* = A`, the values
/// descending and not negative, the same values without the vectors; returns the values.
#[track_caller]
fn check<T: Num>(a: &Matrix<T>, tol: f64) -> Vec<f64> {
    let (m, n) = a.shape();
    let k = m.min(n);
    let svd = a.svd();
    let (u, v) = (svd.u(), svd.v());
    let s: Vec<f64> = svd.singular_values().iter().map(|&x| f(x)).collect();
    assert_eq!((u.shape(), v.shape(), s.len()), ((m, k), (n, k), k));
    assert_orthonormal(&u, tol);
    assert_orthonormal(&v, tol);
    assert!(
        s.windows(2).all(|p| p[0] >= p[1]) && s.iter().all(|&x| x >= 0.0),
        "{s:?}"
    );
    let sigma = Matrix::from_diagonal(&s.iter().map(|&x| T::real(x)).collect::<Vec<_>>());
    let back = &(&u * &sigma) * &v.conjugate_transpose();
    let scale = frobenius(a);
    let d = frobenius(&(&back - a));
    assert!(
        d <= tol * T::EPS * (scale + f64::MIN_POSITIVE),
        "‖UΣV* - A‖ = {d:e}, ‖A‖ = {scale:e}, {m}×{n}"
    );
    let alone: Vec<f64> = a.singular_values().iter().map(|&x| f(x)).collect();
    for (x, y) in alone.iter().zip(&s) {
        assert!((x - y).abs() <= tol * T::EPS * scale, "{x} {y} {m}×{n}");
    }
    s
}

/// A random matrix with orthonormal columns, `m × k`.
fn orthonormal<T: Num>(r: &mut Rng, m: usize, k: usize) -> Matrix<T> {
    random::<T>(r, m, k).qr().q()
}

/// `Q₁·diag(s)·Q₂*` for random `Q₁` of `m × k` and `Q₂` of `n × k`, `k = s.len()`.
fn with_values<T: Num>(r: &mut Rng, m: usize, n: usize, s: &[f64]) -> Matrix<T> {
    let k = s.len();
    let (q1, q2) = (orthonormal::<T>(r, m, k), orthonormal::<T>(r, n, k));
    let sigma = Matrix::from_diagonal(&s.iter().map(|&x| T::real(x)).collect::<Vec<_>>());
    &(&q1 * &sigma) * &q2.conjugate_transpose()
}

/// The shapes of [`shapes_of`] under Miri: below every crossover, as the large paths take hours
/// there and have no unsafe code of their own; the products they call are the product tests'.
const MIRI_SHAPES: [(usize, usize); 5] = [(1, 1), (6, 6), (9, 4), (4, 9), (13, 3)];

fn shapes_of<T: Num>() {
    let mut r = Rng(0x5EED_0301);
    if cfg!(miri) {
        for (m, n) in MIRI_SHAPES {
            let a = random::<T>(&mut r, m, n);
            check(&a, 32.0 * (m + n) as f64);
        }
        return;
    }
    let mut shapes = vec![(24, 24), (25, 25), (26, 26), (31, 31), (32, 32), (33, 33)];
    shapes.extend([
        (51, 51),
        (52, 52),
        (65, 65),
        (100, 100),
        (130, 97),
        (97, 130),
    ]);
    // About the ratios where QR goes first, 1.75 for the values alone and 2.5 with the vectors,
    // and very tall and wide ones.
    shapes.extend([
        (52, 32),
        (56, 32),
        (57, 32),
        (79, 32),
        (80, 32),
        (81, 32),
        (32, 80),
    ]);
    shapes.extend([(300, 20), (20, 300), (400, 7), (80, 79), (1000, 40)]);
    for (m, n) in shapes {
        let a = random::<T>(&mut r, m, n);
        check(&a, 32.0 * (m + n) as f64);
    }
    // Deep trees, of both kinds of halves.
    for n in [150, 201] {
        let a = random::<T>(&mut r, n, n);
        check(&a, 32.0 * n as f64);
    }
}

test_each_type!(shapes_across_thresholds, shapes_of);

fn low_rank_of<T: Num>() {
    let mut r = Rng(0x5EED_0302);
    let cases: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(12, 12, 3), (14, 9, 0), (9, 14, 1)]
    } else {
        &[
            (60, 60, 10),
            (120, 100, 30),
            (100, 120, 1),
            (90, 90, 0),
            (200, 60, 59),
        ]
    };
    for &(m, n, rank) in cases {
        let a = &random::<T>(&mut r, m, rank) * &random::<T>(&mut r, rank, n);
        let s = check(&a, 32.0 * (m + n) as f64);
        let scale = frobenius(&a) + 1.0;
        assert!(
            s[rank..]
                .iter()
                .all(|&x| x <= 64.0 * (m + n) as f64 * T::EPS * scale),
            "{m} {n} {rank}"
        );
        assert_eq!(a.rank(), rank);
    }
}

test_each_type!(low_rank, low_rank_of);

fn repeated_of<T: Num>() {
    let mut r = Rng(0x5EED_0303);
    if cfg!(miri) {
        repeated_at::<T>(&mut r, 9, 2);
    } else {
        repeated_at::<T>(&mut r, 90, 3);
    }
}

/// Repeated, clustered, graded and tiny singular values of matrices with `n` columns, of the
/// first `shapes` of: square, tall, and tall past the ratio where QR goes first.
fn repeated_at<T: Num>(r: &mut Rng, n: usize, shapes: usize) {
    let third = n.div_ceil(3);
    let cases: Vec<Vec<f64>> = vec![
        // Repeated: three values, a third of the times each.
        (0..n).map(|i| [3.0, 2.0, 1.0][i / third]).collect(),
        // All equal.
        vec![1.0; n],
        // Clustered within 1e-9, and within ε.
        (0..n).map(|i| 1.0 + 1e-9 * i as f64).collect(),
        (0..n).map(|i| 1.0 + T::EPS * (i % 3) as f64).collect(),
        // Graded over many orders, and some tiny.
        (0..n).map(|i| 10f64.powf(-(i as f64) / 8.0)).collect(),
        (0..n)
            .map(|i| if i < 2 * n / 3 { 1.0 } else { 1e-30 })
            .collect(),
    ];
    for (c, s) in cases.iter().enumerate() {
        for &(m, k) in [(n, n), (n + 4 * n / 9, n), (2 * n + n / 9, n)]
            .iter()
            .take(shapes)
        {
            let a = with_values::<T>(r, m, k, s);
            let got = check(&a, 64.0 * (m + k) as f64);
            let mut want = s.clone();
            want.sort_by(|x, y| y.total_cmp(x));
            let scale = want[0];
            for (g, w) in got.iter().zip(&want) {
                // Absolute accuracy, to `ε·σ₁`.
                assert!(
                    (g - w).abs() <= 64.0 * (m + k) as f64 * T::EPS * scale,
                    "case {c}: {g} {w}"
                );
            }
        }
    }
}

#[test]
fn repeated_clustered_graded() {
    each_type!(repeated_of);
}

fn exact_of<T: Num>() {
    let mut r = Rng(0x5EED_0304);
    // Zero matrices, of every path.
    let zeros: &[(usize, usize)] = if cfg!(miri) {
        &[(5, 5), (9, 4), (4, 9)]
    } else {
        &[(5, 5), (40, 40), (100, 30), (30, 100)]
    };
    for &(m, n) in zeros {
        let a = Matrix::<T>::zeros(m, n);
        let s = check(&a, 1.0);
        assert!(s.iter().all(|&x| x == 0.0));
    }
    // A permuted diagonal, its values over the whole range of the type: exact, each to its own
    // relative accuracy.
    for &n in if cfg!(miri) { &[10][..] } else { &[10, 60][..] } {
        let lo = if T::EPS > 1e-10 { -37 } else { -300 };
        let vals: Vec<f64> = (0..n)
            .map(|i| 10f64.powi(lo * i as i32 / n as i32) * (1.0 + 0.1 * r.unit()))
            .collect();
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, (r.next() % (i as u64 + 1)) as usize);
        }
        let a = Matrix::from_fn(n, n, |i, j| {
            if perm[i] == j {
                T::real(vals[i])
            } else {
                T::real(0.0)
            }
        });
        let got = check(&a, 16.0 * n as f64);
        let mut want: Vec<f64> = vals.iter().map(|&x| T::real(x).c64().re).collect();
        want.sort_by(|x, y| y.total_cmp(x));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 4.0 * T::EPS * w, "{g:e} {w:e}");
        }
    }
    // An infinity or a NaN, on every path.
    let non_finite: &[(usize, usize)] = if cfg!(miri) {
        &[(3, 3), (9, 4), (4, 9)]
    } else {
        &[(3, 3), (50, 50), (100, 20), (20, 100)]
    };
    for &(m, n) in non_finite {
        let mut a = random::<T>(&mut r, m, n);
        a[(m / 2, n / 2)] = T::real(f64::NAN);
        assert!(a.singular_values().iter().all(|x| x.is_nan()));
        let svd = a.svd();
        assert!(svd.singular_values().iter().all(|x| x.is_nan()));
    }
}

#[test]
fn zeros_diagonals_and_non_finite() {
    each_type!(exact_of);
}

fn null_space_of<T: Num>() {
    // `null_space` takes all `n` right singular vectors, padding a wide matrix with zero rows.
    let mut r = Rng(0x5EED_0305);
    let cases: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(5, 12, 5), (10, 10, 6), (14, 6, 4)]
    } else {
        &[(30, 80, 30), (70, 70, 40), (120, 40, 25)]
    };
    for &(m, n, rank) in cases {
        let a = &random::<T>(&mut r, m, rank) * &random::<T>(&mut r, rank, n);
        let null = a.null_space();
        assert_eq!(null.shape(), (n, n - rank));
        assert_orthonormal(&null, 64.0);
        assert_close(
            &(&a * &null),
            &Matrix::zeros(m, n - rank),
            1e3 * (frobenius(&a) + 1.0),
        );
    }
}

#[test]
fn null_spaces() {
    each_type!(null_space_of);
}

fn scaled_of<T: Num>() {
    // Scaled by powers of two so far that the squares of the elements overflow or underflow:
    // the same decomposition, scaled.
    let mut r = Rng(0x5EED_0306);
    let e = if T::EPS > 1e-10 { 80 } else { 560 };
    let shapes: &[(usize, usize)] = if cfg!(miri) {
        &[(10, 10), (14, 6)]
    } else {
        &[(40, 40), (90, 60), (150, 30), (30, 90)]
    };
    for &(m, n) in shapes {
        let a = random::<T>(&mut r, m, n);
        let s = check(&a, 32.0 * (m + n) as f64);
        for k in [e, -e] {
            let scale = 2f64.powi(k);
            let b = a.map(|&x| x * T::real(scale));
            let svd = b.svd();
            let (u, v) = (svd.u(), svd.v());
            assert_orthonormal(&u, 32.0 * (m + n) as f64);
            assert_orthonormal(&v, 32.0 * (m + n) as f64);
            let sb: Vec<f64> = svd
                .singular_values()
                .iter()
                .map(|&x| f(x) / scale)
                .collect();
            let alone: Vec<f64> = b.singular_values().iter().map(|&x| f(x) / scale).collect();
            for ((x, y), z) in sb.iter().zip(&s).zip(&alone) {
                assert!(
                    (x - y).abs() <= 64.0 * (m + n) as f64 * T::EPS * s[0],
                    "{k}: {x} {y}"
                );
                assert!(
                    (z - y).abs() <= 64.0 * (m + n) as f64 * T::EPS * s[0],
                    "{k}: {z} {y}"
                );
            }
            let sigma = Matrix::from_diagonal(&sb.iter().map(|&x| T::real(x)).collect::<Vec<_>>());
            assert_close(
                &(&(&u * &sigma) * &v.conjugate_transpose()),
                &a,
                32.0 * (m + n) as f64,
            );
        }
    }
}

test_each_type!(scaled_to_the_ends_of_the_range, scaled_of);

fn blocks_of<T: Num>() {
    // Block diagonal: the bidiagonal matrix splits where the blocks meet, into problems each
    // solved by divide and conquer; the values are those of the blocks together.
    let mut r = Rng(0x5EED_0307);
    let cases: &[[usize; 3]] = if cfg!(miri) {
        &[[4, 3, 6], [2, 7, 1]]
    } else {
        &[[40, 30, 70], [26, 90, 1]]
    };
    for sizes in cases {
        let blocks: Vec<Matrix<T>> = sizes.iter().map(|&k| random::<T>(&mut r, k, k)).collect();
        let n: usize = sizes.iter().sum();
        let mut a = Matrix::<T>::zeros(n, n);
        let mut want = Vec::new();
        let mut at = 0;
        for b in &blocks {
            let k = b.rows();
            for i in 0..k {
                for j in 0..k {
                    a[(at + i, at + j)] = b[(i, j)];
                }
            }
            want.extend(b.singular_values().iter().map(|&x| f(x)));
            at += k;
        }
        want.sort_by(|x, y| y.total_cmp(x));
        let got = check(&a, 32.0 * n as f64);
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 64.0 * n as f64 * T::EPS * want[0],
                "{g} {w}"
            );
        }
    }
}

test_each_type!(block_diagonal, blocks_of);

fn graded_bidiagonal_of<T: Num>() {
    // A bidiagonal matrix passes through the reduction as it is, and dqds finds its singular
    // values to high relative accuracy, the tiny ones too: those of the lower precisions agree
    // with the higher's, each to a few units of its own size.
    let mut r = Rng(0x5EED_0308);
    for (n, step) in [(30, -1.0), (60, -0.5), (45, 0.7)] {
        let a = Matrix::<T>::from_fn(n, n, |i, j| {
            let g = 10f64.powf(step * i as f64 * if T::EPS > 1e-10 { 0.5 } else { 1.0 });
            if j == i || j == i + 1 {
                T::random(&mut r) * T::real(g)
            } else {
                T::real(0.0)
            }
        });
        let wide = Matrix::from_fn(n, n, |i, j| a[(i, j)].c64());
        let want: Vec<f64> = wide.singular_values();
        let got: Vec<f64> = a.singular_values().iter().map(|&x| f(x)).collect();
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 16.0 * n as f64 * T::EPS * w,
                "{n}: {g:e} {w:e}"
            );
        }
    }
}

#[test]
fn graded_bidiagonal_relative_accuracy() {
    each_type!(graded_bidiagonal_of);
}

/// The numbers of columns of the small path, on both sides of the switches of its scratch at 16
/// and 48 and of the crossover of single precision complex vectors at 88.
fn small_sizes() -> Vec<usize> {
    if cfg!(miri) {
        vec![2, 5, 17]
    } else {
        vec![1, 2, 3, 5, 16, 17, 40, 48, 49, 80, 88, 89]
    }
}

/// The numbers of columns about the other crossovers of the small path, with the vectors and
/// without: 96 and 112 for `f64`, 112 and 96 for `f32`, 96 and 192 for `Complex<f64>`, 192 for
/// `Complex<f32>`.
fn crossovers<T: Num>() -> Vec<usize> {
    if cfg!(miri) {
        return Vec::new();
    }
    match core::any::type_name::<T>() {
        "f64" => vec![96, 97, 112, 113],
        "f32" => vec![112, 113, 96, 97],
        "a::math::complex::Complex<f64>" => vec![96, 97, 192, 193],
        _ => vec![192, 193],
    }
}

fn small_shapes_of<T: Num>() {
    let mut r = Rng(0x5EED_0309);
    // About the crossovers: square, which `check` compares with the values alone.
    for n in crossovers::<T>() {
        check(&random::<T>(&mut r, n, n), 64.0 * n as f64);
    }
    for n in small_sizes() {
        // Square, tall below and from the ratios where QR goes first (1.75 for the values alone,
        // 2.5 with the vectors), and wide; the larger only about the first.
        let a = (7 * n).div_ceil(4);
        let b = (5 * n).div_ceil(2);
        let rows = if n <= 40 {
            vec![n, a - 1, a, b - 1, b]
        } else {
            vec![n]
        };
        for m in rows {
            let a = random::<T>(&mut r, m, n);
            check(&a, 32.0 * (m + n) as f64);
            check(&a.conjugate_transpose(), 32.0 * (m + n) as f64);
        }
        // Low rank, square, wide and tall.
        for (m, k, rank) in [(n, n, n / 3), (n, n + 7, n / 2), (n + 7, n, 1)] {
            let a = &random::<T>(&mut r, m, rank) * &random::<T>(&mut r, rank, k);
            let s = check(&a, 32.0 * (m + k) as f64);
            let scale = frobenius(&a) + 1.0;
            assert!(
                s[rank..]
                    .iter()
                    .all(|&x| x <= 64.0 * (m + k) as f64 * T::EPS * scale),
                "{m} {k} {rank}"
            );
        }
        // Zero, and not finite.
        let s = check(&Matrix::<T>::zeros(n, n), 1.0);
        assert!(s.iter().all(|&x| x == 0.0));
        let mut a = random::<T>(&mut r, n + 1, n);
        a[(n / 2, n / 2)] = T::real(f64::INFINITY);
        assert!(a.singular_values().iter().all(|x| x.is_nan()));
        let svd = a.svd();
        assert!(svd.singular_values().iter().all(|x| x.is_nan()));
        assert!(svd.u().as_slice().iter().all(|x| x.c64().re.is_nan()));
        // Near the ends of the range: the same decomposition, scaled.
        let a = random::<T>(&mut r, n + 2, n);
        let s = check(&a, 32.0 * (2 * n + 2) as f64);
        let e = if T::EPS > 1e-10 { 80 } else { 560 };
        for k in [e, -e] {
            let scale = 2f64.powi(k);
            let b = a.map(|&x| x * T::real(scale));
            let sb: Vec<f64> = b.singular_values().iter().map(|&x| f(x) / scale).collect();
            for (x, y) in sb.iter().zip(&s) {
                assert!(
                    (x - y).abs() <= 64.0 * (2 * n + 2) as f64 * T::EPS * s[0],
                    "{k}: {x} {y}"
                );
            }
            let svd = b.svd();
            assert_orthonormal(&svd.u(), 32.0 * (2 * n + 2) as f64);
            assert_orthonormal(&svd.v(), 32.0 * (2 * n + 2) as f64);
        }
    }
}

#[test]
fn small_shapes_and_hard_cases() {
    each_type!(small_shapes_of);
}

fn small_repeated_of<T: Num>() {
    // Repeated, clustered and graded values on both sides of the switches of the scratch and of
    // the first crossover of each type.
    let mut r = Rng(0x5EED_0310);
    let mut sizes = if cfg!(miri) {
        vec![(5, 3)]
    } else {
        vec![(5, 3), (16, 3), (17, 3), (48, 2), (49, 2)]
    };
    if !cfg!(miri) {
        // About the crossover with the vectors.
        let n = match core::any::type_name::<T>() {
            "f32" => 112,
            "a::math::complex::Complex<f32>" => 88,
            _ => 96,
        };
        sizes.extend([(n, 1), (n + 1, 1)]);
    }
    for (n, shapes) in sizes {
        repeated_at::<T>(&mut r, n, shapes);
    }
}

#[test]
fn small_repeated_clustered_graded() {
    each_type!(small_repeated_of);
}

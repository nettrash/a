//! The sparse QR decomposition: the factors against the matrix, against the Cholesky factor of
//! the normal matrix and against each other; the least squares against the dense decomposition
//! and against NumPy's `lstsq` on the cases of `data/gen_qr.py`; and what a rank-deficient,
//! empty, badly scaled, non-finite or malformed matrix gives.

use super::common::*;
use a::math::complex::Float;
use a::math::matrix::{Matrix, Norm};
use a::math::sparse::{Ordering, SparseMatrix, SparseQr, elimination_tree};

/// A random permutation of `0..n`.
fn random_permutation(r: &mut Rng, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, r.below(i + 1));
    }
    p
}

/// The element `re + im·i` of the type, `re` for a real type.
fn element<T: Num>(re: f64, im: f64) -> T {
    T::from_parts(T::Real::from_f64(re), T::Real::from_f64(im))
}

/// An `m × n` matrix of full column rank with a condition number of a few units: `n` of its
/// rows hold a diagonal that dominates the few other elements of the row, the others hold
/// `per_row` random elements, and the rows are shuffled.
fn well_conditioned<T: Num>(r: &mut Rng, m: usize, n: usize, per_row: usize) -> SparseMatrix<T> {
    assert!(m >= n);
    let place = random_permutation(r, m);
    let mut entries = Vec::new();
    for (i, &row) in place.iter().enumerate() {
        if i < n {
            entries.push((row, i, T::real(2.0 + r.unit().abs())));
            for _ in 0..per_row {
                let j = r.below(n);
                if j != i {
                    entries.push((row, j, T::random(r) * T::real(0.3 / per_row as f64)));
                }
            }
        } else {
            for _ in 0..per_row {
                entries.push((row, r.below(n), T::random(r)));
            }
        }
    }
    SparseMatrix::from_coo(m, n, &entries)
}

/// A dense matrix of random elements.
fn random_dense<T: Num>(r: &mut Rng, rows: usize, cols: usize) -> Matrix<T> {
    Matrix::from_fn(rows, cols, |_, _| T::random(r))
}

/// `A·Q_c` as a dense matrix: column `k` of it is column `q[k]` of `a`.
fn permuted<T: Num>(a: &SparseMatrix<T>, q: &[usize]) -> Matrix<T> {
    let a = dense(a);
    Matrix::from_fn(a.rows(), a.cols(), |i, j| a[(i, q[j])])
}

/// Whether `p` is a permutation of `0..n`.
fn assert_permutation(p: &[usize], n: usize) {
    assert_eq!(p.len(), n, "the permutation is of the wrong length");
    let mut seen = vec![false; n];
    for &i in p {
        assert!(i < n && !seen[i], "not a permutation: {p:?}");
        seen[i] = true;
    }
}

/// What every decomposition must satisfy, whatever the matrix: `R` upper triangular in the
/// canonical sparse form with a real diagonal, `Q` of orthonormal columns, and `A·Q_c = Q·R`,
/// to `factor` times the rounding of the type.
fn assert_decomposition<T: Num>(a: &SparseMatrix<T>, qr: &SparseQr<T>, factor: f64) {
    let (m, n) = (a.rows(), a.cols());
    let q_c = qr.column_permutation();
    assert_permutation(q_c, n);
    let r = qr.r();
    assert_csr(&r);
    assert_eq!((r.rows(), r.cols()), (n, n), "R isn't n × n");
    for i in 0..n {
        for (j, x) in r.row(i) {
            assert!(j >= i, "R has an element below its diagonal at ({i}, {j})");
            if j == i {
                assert!(T::im(x) == T::Real::ZERO, "the diagonal of R isn't real");
            }
        }
    }
    assert!(qr.nnz() >= r.nnz(), "the factors store less than R holds");
    assert!(qr.rank() <= n, "the rank is above the number of columns");
    let q = qr.q();
    assert_eq!(q.shape(), (m, n), "Q isn't m × n");
    assert_close(&(&q * &dense(&r)), &permuted(a, q_c), factor);
    assert_close(
        &(&q.conjugate_transpose() * &q),
        &Matrix::identity(n),
        factor,
    );
}

/// The orderings a decomposition is tried with: every kind, and two random permutations.
fn orderings(r: &mut Rng, n: usize) -> Vec<Ordering> {
    vec![
        Ordering::Natural,
        Ordering::ReverseCuthillMckee,
        Ordering::Amd,
        Ordering::Colamd,
        Ordering::NestedDissection,
        Ordering::Given(random_permutation(r, n)),
        Ordering::Given(random_permutation(r, n)),
    ]
}

/// A reference solution, as the real and imaginary parts of each element.
type Parts = [[f64; 2]];

/// The distance of `x` from the reference `want`, relative to the reference.
fn relative_error<T: Num>(x: &[T], want: &Parts) -> f64 {
    assert_eq!(x.len(), want.len(), "the solution is of the wrong length");
    let (mut distance, mut size) = (0.0, 0.0);
    for (&x, &[re, im]) in x.iter().zip(want) {
        let (dr, di) = (T::re(x).to_f64() - re, T::im(x).to_f64() - im);
        distance += dr * dr + di * di;
        size += re * re + im * im;
    }
    (distance / size).sqrt()
}

/// The real reference `x` as pairs of parts.
fn pairs(x: &[f64]) -> Vec<[f64; 2]> {
    x.iter().map(|&re| [re, 0.0]).collect()
}

// The cases of `data/gen_qr.py`.

/// A case of `data/gen_qr.py`: the shape, the seed and the density of its formula, and whether
/// its elements are complex.
#[derive(Clone, Copy, Debug)]
struct Case {
    m: usize,
    n: usize,
    seed: usize,
    k: usize,
    complex: bool,
}

impl Case {
    /// The element `(i, j)` as its parts: `entry` of `gen_qr.py`.
    fn entry(&self, i: usize, j: usize) -> (f64, f64) {
        let Case {
            m,
            n,
            seed,
            k,
            complex,
        } = *self;
        let imaginary = |x: f64| if complex { x } else { 0.0 };
        if i == j * m / n {
            let im = (((j * 3 + seed) % 5) as f64 - 2.0) / 8.0;
            return (4.0 + (j % 5) as f64 / 4.0, imaginary(im));
        }
        if !(i * 73 + j * 151 + seed * 17).is_multiple_of(k) {
            return (0.0, 0.0);
        }
        let re = (((i * 31 + j * 57 + seed) % 15) as f64 - 7.0) / 4.0;
        let im = (((i * 43 + j * 29 + seed * 7) % 13) as f64 - 6.0) / 8.0;
        (re, imaginary(im))
    }

    /// The element `i` of the right-hand side: `rhs` of `gen_qr.py`.
    fn rhs(&self, i: usize) -> (f64, f64) {
        let re = (((i * 19 + self.seed * 3) % 11) as f64 - 5.0) / 2.0;
        let im = (((i * 23 + self.seed) % 7) as f64 - 3.0) / 4.0;
        (re, if self.complex { im } else { 0.0 })
    }

    /// The matrix, dense.
    fn dense<T: Num>(&self) -> Matrix<T> {
        Matrix::from_fn(self.m, self.n, |i, j| {
            let (re, im) = self.entry(i, j);
            element(re, im)
        })
    }

    /// The matrix.
    fn matrix<T: Num>(&self) -> SparseMatrix<T> {
        SparseMatrix::from_matrix(&self.dense())
    }

    /// The right-hand side, as a column.
    fn b<T: Num>(&self) -> Matrix<T> {
        Matrix::from_fn(self.m, 1, |i, _| {
            let (re, im) = self.rhs(i);
            element(re, im)
        })
    }
}

/// `[I; 2·D]`, `D` the first differences of `n` points, and the data `((37·i) mod 9 − 4)/4`
/// against zeros: the least squares of a smoothed curve, `smoothing` of `gen_qr.py`.
fn smoothing<T: Num>(n: usize) -> (SparseMatrix<T>, Matrix<T>) {
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, T::real(1.0)));
    }
    for i in 0..n.saturating_sub(1) {
        entries.push((n + i, i, T::real(-2.0)));
        entries.push((n + i, i + 1, T::real(2.0)));
    }
    let m = 2 * n - 1;
    let b = Matrix::from_fn(m, 1, |i, _| {
        if i < n {
            T::real(((37 * i) % 9) as f64 / 4.0 - 1.0)
        } else {
            T::real(0.0)
        }
    });
    (SparseMatrix::from_coo(m, n, &entries), b)
}

const SQUARE: Case = Case {
    m: 12,
    n: 12,
    seed: 1,
    k: 4,
    complex: false,
};
const TALL: Case = Case {
    m: 36,
    n: 12,
    seed: 2,
    k: 5,
    complex: false,
};
const VERY_TALL: Case = Case {
    m: 300,
    n: 15,
    seed: 3,
    k: 7,
    complex: false,
};
const COMPLEX_SQUARE: Case = Case {
    m: 9,
    n: 9,
    seed: 4,
    k: 3,
    complex: true,
};
const COMPLEX_TALL: Case = Case {
    m: 40,
    n: 10,
    seed: 5,
    k: 4,
    complex: true,
};
const COMPLEX_VERY_TALL: Case = Case {
    m: 200,
    n: 10,
    seed: 6,
    k: 6,
    complex: true,
};
const DEFICIENT: Case = Case {
    m: 20,
    n: 8,
    seed: 7,
    k: 3,
    complex: false,
};
const COMPLEX_DEFICIENT: Case = Case {
    m: 15,
    n: 6,
    seed: 8,
    k: 3,
    complex: true,
};

// Generated by lib/tests/math/sparse/data/gen_qr.py; do not edit by hand.

// SQUARE: 12 x 12, 46 stored elements;
// lstsq against mpmath 1.2e-15, lsqr against mpmath 1.9e-16.
const SQUARE_RESIDUAL: f64 = 8.479226140392104e-15;
const SQUARE_COND: f64 = 2.440904133725391;
const SQUARE_KAPPA: f64 = 2.4409041337253963;
const SQUARE_X: [f64; 12] = [
    0.2933119431760369,
    -0.5824279430788994,
    0.016107950546635885,
    0.14424109981023145,
    -0.39568783697133575,
    0.6827976966872117,
    0.14432891466664305,
    0.20171914226281454,
    -0.6202150222887649,
    0.3672124174066435,
    0.23469183450235276,
    -0.6304255684608578,
];

// TALL: 36 x 12, 99 stored elements;
// lstsq against mpmath 1.2e-15, lsqr against mpmath 5.1e-16.
const TALL_RESIDUAL: f64 = 7.686357111958434;
const TALL_COND: f64 = 2.9216801219061956;
const TALL_KAPPA: f64 = 11.656120703675434;
const TALL_X: [f64; 12] = [
    0.07726548084169457,
    0.22326559591531764,
    0.45827538734356976,
    -0.12009899660714758,
    -0.25793470862196677,
    -0.047734519158305795,
    0.10561853709178802,
    0.34716427623245916,
    -0.22536215450188413,
    -0.35793470862196686,
    -0.1727345191583049,
    -0.012028521731741295,
];

// VERY_TALL: 300 x 15, 613 stored elements;
// lstsq against mpmath 8.9e-16, lsqr against mpmath 7.3e-16.
const VERY_TALL_RESIDUAL: f64 = 26.930201699998367;
const VERY_TALL_COND: f64 = 1.7548548483851798;
const VERY_TALL_KAPPA: f64 = 14.416039921494946;
const VERY_TALL_X: [f64; 15] = [
    0.2621850807418454,
    0.11436687786484916,
    0.2776956362663091,
    -0.14245048787942538,
    -0.18421981730825326,
    0.045040864819134864,
    -0.0709199107917596,
    0.02177461851391294,
    -0.0041546584945324466,
    0.3314881312371011,
    0.013579945560635662,
    0.027241717118755684,
    -0.1288552886866929,
    0.1614685137776295,
    0.05490717240081109,
];

// COMPLEX_SQUARE: 9 x 9, 33 stored elements;
// lstsq against mpmath 7.5e-16, lsqr against mpmath 2.5e-16.
const COMPLEX_SQUARE_RESIDUAL: f64 = 4.096447288047435e-15;
const COMPLEX_SQUARE_COND: f64 = 3.1546290481736836;
const COMPLEX_SQUARE_KAPPA: f64 = 3.1546290481736885;
const COMPLEX_SQUARE_X: [[f64; 2]; 9] = [
    [-0.18105995187275745, 0.15477536777777015],
    [0.5252223844642097, 0.17702169419453914],
    [0.017205401342893, -0.27959111875670983],
    [-0.19482663582494653, -0.0010055417226471413],
    [-0.41487118249739385, 0.051335680965792115],
    [0.5744430949614925, -0.24274318679224918],
    [0.07469731642184504, 0.05443849231310809],
    [-0.35245629831847686, -0.050478322463126064],
    [0.39093500819745564, 0.1385676118178977],
];

// COMPLEX_TALL: 40 x 10, 107 stored elements;
// lstsq against mpmath 8.4e-16, lsqr against mpmath 2.5e-16.
const COMPLEX_TALL_RESIDUAL: f64 = 9.604332257600877;
const COMPLEX_TALL_COND: f64 = 2.23267369718286;
const COMPLEX_TALL_KAPPA: f64 = 9.696104437923296;
const COMPLEX_TALL_X: [[f64; 2]; 10] = [
    [0.10380504706682202, 0.16762734281242261],
    [-0.33667084028000716, 0.2292830565277466],
    [-0.03355706978628925, -0.10715533773943142],
    [0.07920806939371332, -0.03911194481391591],
    [-0.23343801420906388, 0.04914546161189458],
    [0.4667670239047981, 0.031691975278834716],
    [0.2625626761533961, 0.06867617029507865],
    [0.09695837064228599, 0.07044728639097599],
    [0.23383900189117793, 0.24629079902905401],
    [0.06266844432096219, -0.17302175193000754],
];

// COMPLEX_VERY_TALL: 200 x 10, 337 stored elements;
// lstsq against mpmath 7.1e-16, lsqr against mpmath 2.4e-16.
const COMPLEX_VERY_TALL_RESIDUAL: f64 = 23.084016892691665;
const COMPLEX_VERY_TALL_COND: f64 = 1.2139822094895905;
const COMPLEX_VERY_TALL_KAPPA: f64 = 9.734226436927047;
const COMPLEX_VERY_TALL_X: [[f64; 2]; 10] = [
    [0.07832203139910801, -0.01908649511846863],
    [-0.08301974203940723, -0.03301955274158136],
    [0.025367359487930084, 0.025019644724458973],
    [-0.16483023588130563, -0.03930966835085613],
    [-0.001486673590244318, 0.11172703849133145],
    [-0.04765930089527528, -0.033733485304824916],
    [0.26557700983855964, -0.0567613110754332],
    [0.04152733971886902, 0.0048024838302846195],
    [-0.19139214405581742, -0.016084258671995134],
    [0.17561451505152664, 0.0197974473759276],
];

// SMOOTHING: 79 x 40, 118 stored elements;
// lstsq against mpmath 1.0e-15, lsqr against mpmath 2.8e-16.
const SMOOTHING_RESIDUAL: f64 = 3.3978039053089937;
const SMOOTHING_COND: f64 = 4.120113914670928;
const SMOOTHING_KAPPA: f64 = 13.304858500601;
const SMOOTHING_X: [f64; 40] = [
    -0.6360810737616113,
    -0.545101342202015,
    -0.40289694619292143,
    -0.23641678673205857,
    -0.06654082395421061,
    0.08669993283508438,
    0.19911567283315063,
    0.23631033103950466,
    0.14508257200573502,
    -0.15987454402660053,
    -0.2548002960655865,
    -0.22592612212096938,
    -0.12853347870659457,
    -0.0007742049688680183,
    0.12679151752664092,
    0.22355511940380998,
    0.2512075011319318,
    0.15416175814303684,
    -0.15434354531009908,
    -0.25143473509076003,
    -0.22388460864411103,
    -0.12730563435848924,
    -5.306866249044422e-05,
    0.12718622986788594,
    0.22372208586523415,
    0.2511884633288906,
    0.15395195662476963,
    -0.15479656092315897,
    -0.2522442187018773,
    -0.2252529311560647,
    -0.1295748763992688,
    -0.003790540742289762,
    0.12104615972911699,
    0.21364440013280295,
    0.23465374056968952,
    0.12682651614899926,
    -0.19929407923444115,
    -0.32523819442649265,
    -0.3449918582251672,
    -0.3259934865801339,
];

// DEFICIENT: 20 x 8, numpy.linalg.matrix_rank.
const DEFICIENT_RANK: usize = 6;

// COMPLEX_DEFICIENT: 15 x 6, numpy.linalg.matrix_rank.
const COMPLEX_DEFICIENT_RANK: usize = 4;

// LARGE_COMBINATION: sympy rank 2, determinant 0; numpy.linalg.matrix_rank 2,
// sigma_3 / sigma_1 = 6.3e-17.

// STRUCTURAL: 13 x 5, scipy.sparse.csgraph.structural_rank 4, numpy.linalg.matrix_rank 4,
// sigma_n / sigma_1 = 3.3e-20.
const STRUCTURAL_RANK: usize = 4;
const STRUCTURAL: [(usize, usize, f64); 13] = [
    (0, 0, -0.6446499814983495),
    (0, 1, 0.8182224590931129),
    (0, 2, 0.37690383650090253),
    (0, 3, -0.33473394512862487),
    (0, 4, 0.24606349249284443),
    (1, 3, -0.4514242697154671),
    (2, 0, -0.025679805834457836),
    (2, 3, -0.1183901520437265),
    (2, 4, 0.8645024825922223),
    (3, 1, -0.11385088727414328),
    (7, 3, -0.6009378343030822),
    (8, 1, -0.32981445750449323),
    (11, 1, 0.8476842129253435),
];

// DENSE_ROW: 18 x 16, scipy.sparse.csgraph.structural_rank 15, numpy.linalg.matrix_rank 15,
// sigma_n / sigma_1 = 2.7e-17.
const DENSE_ROW_RANK: usize = 15;
const DENSE_ROW: [(usize, usize, f64); 49] = [
    (0, 8, 0.6023761272419847),
    (0, 12, 0.3783878659664561),
    (1, 11, -0.12420937473333371),
    (1, 14, 0.5548381517550194),
    (2, 10, 0.3257821505055558),
    (2, 12, 0.2772095944681834),
    (3, 3, -0.47051244568027917),
    (3, 4, -0.658919213762335),
    (4, 8, 0.9095475539133266),
    (4, 11, 0.4804826280738781),
    (5, 1, -0.7661099809350511),
    (5, 5, 0.5013755561861923),
    (6, 13, 0.7530019359197235),
    (7, 11, -0.895103597058257),
    (7, 14, 0.211118259241188),
    (8, 2, -0.6056515058961842),
    (8, 11, -0.7606311510412471),
    (9, 0, 0.5576925101091985),
    (9, 1, 0.05849810743998152),
    (9, 2, 0.5534207791475692),
    (9, 3, -0.5304217865297336),
    (9, 4, -0.9012995873924932),
    (9, 5, 0.3533688872442331),
    (9, 6, -0.9254069713403907),
    (9, 7, 0.4003977909640988),
    (9, 8, 0.8876276669279255),
    (9, 9, 0.14141769139310356),
    (9, 10, -0.6468263237456451),
    (9, 11, 0.18831520758490639),
    (9, 12, 0.6915343365416913),
    (9, 13, 0.8750872401936423),
    (9, 14, -0.4414336918845916),
    (9, 15, 0.5970008266178273),
    (10, 3, -0.9441693808990836),
    (10, 9, -0.9697131685746025),
    (11, 5, -0.8438604887428931),
    (11, 7, 0.685998812522667),
    (12, 8, -0.2254315454999667),
    (12, 15, -0.7469415976071982),
    (13, 0, 0.451834795156965),
    (13, 6, 0.3652486899280414),
    (14, 11, -0.9491415113545476),
    (14, 13, -0.3405577599850993),
    (15, 5, -0.3093121967887369),
    (15, 13, 0.49869127341413133),
    (16, 11, 0.7077855345775024),
    (16, 12, -0.27219671352977604),
    (17, 2, 0.8167062161548604),
    (17, 15, 0.8501753000233978),
];

// The upper bidiagonal matrix of 1 and -2: condition numbers, mpmath at 50 digits.
const BIDIAGONAL_COND: [(usize, f64); 2] = [(40, 2197541122339.981), (60, 2.3051484007002289e+18)];

/// The most a solution of the embedded cases may be from NumPy's, in units of the rounding of the
/// type times the condition number of the least squares, `κ + κ²·‖r‖/(‖A‖·‖x‖)`. The worst
/// seen is given at the test.
const EMBEDDED_BOUND: f64 = 8.0;

/// What a solution of a case of `gen_qr.py` must satisfy against its embedded reference: within
/// [`EMBEDDED_BOUND`] of it, and with the residual NumPy found.
fn check_embedded<T: Num>(
    a: &SparseMatrix<T>,
    b: &Matrix<T>,
    want: &Parts,
    residual: f64,
    cond: f64,
    kappa: f64,
) {
    let mut r = Rng(0x5eed_0006);
    for ordering in [
        Ordering::Colamd,
        Ordering::Natural,
        Ordering::Amd,
        Ordering::Given(random_permutation(&mut r, a.cols())),
    ] {
        let qr = SparseQr::with_ordering(a, &ordering);
        assert_eq!(qr.rank(), a.cols(), "an embedded case lost rank");
        let x = qr
            .solve(b)
            .expect("the embedded cases have full column rank");
        let error = relative_error(x.as_slice(), want) / (kappa * T::EPS);
        assert!(
            error <= EMBEDDED_BOUND,
            "{} × {} with {ordering:?}: {error:.2} units from NumPy's solution",
            a.rows(),
            a.cols()
        );
        // The residual is about as well conditioned as the matrix.
        let found = norm(&(a * &x - b));
        let scale = norm(&dense(a)) * norm(&x) + norm(b);
        assert!(
            (found - residual).abs() <= 8.0 * cond * T::EPS * scale,
            "the residual is {found}, NumPy's {residual}"
        );
    }
}

// The factors.

#[test]
fn the_factors_give_the_permuted_matrix_and_q_has_orthonormal_columns() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0001);
        // Square, tall, and very tall with twenty rows to a column.
        let shapes = [
            (1, 1, 1),
            (2, 2, 2),
            (5, 5, 2),
            (12, 7, 3),
            (30, 30, 3),
            (40, 2, 1),
            (60, 3, 2),
            (size(90, 20), size(30, 6), 4),
            (size(400, 40), size(20, 2), 2),
        ];
        for (m, n, per_row) in shapes {
            let a = well_conditioned::<T>(&mut r, m, n, per_row);
            assert_decomposition(&a, &SparseQr::new(&a), 16.0);
            for ordering in orderings(&mut r, n) {
                let qr = SparseQr::with_ordering(&a, &ordering);
                assert_decomposition(&a, &qr, 16.0);
                assert_eq!(qr.rank(), n, "a well-conditioned matrix lost rank");
                if let Ordering::Given(q) = &ordering {
                    assert_eq!(qr.column_permutation(), &q[..]);
                }
            }
        }
    }
    each_type!(check);
}

#[test]
fn large_fronts_are_reduced_by_blocks_of_reflections_as_the_dense_decomposition() {
    // Fronts of more reflections than a panel: dense ones, whose panels are applied as blocks
    // of reflections, and tall ones whose staircase fills too little of their rectangle to be
    // held dense, whose panels are applied one reflection at a time; `R` up to the signs of its
    // rows is that of the dense decomposition of `A·Q_c`, and so are the least squares.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_00b1);
        let mut cases: Vec<(SparseMatrix<T>, Ordering)> = Vec::new();
        for (m, n, per_row) in [
            (size(260, 90), size(110, 70), 14),
            (size(300, 120), size(200, 100), 9),
            (size(600, 150), size(90, 64), 40),
            // Nearly full: one front of nearly 300 reflections, held dense, its panels applied
            // to the columns right of them as blocks.
            (size(330, 40), size(290, 30), 90),
        ] {
            cases.push((
                well_conditioned::<T>(&mut r, m, n, per_row),
                Ordering::Colamd,
            ));
        }
        // Rows that each begin at their own column and run on to the right, then many rows in
        // the last eight columns alone: the late steps of the staircase are tall.
        let n = size(96, 72);
        let mut entries = Vec::new();
        for j in 0..n {
            entries.push((j, j, T::real(3.0)));
            for _ in 0..3 {
                let k = j + r.below(n - j);
                if k != j {
                    entries.push((j, k, T::random(&mut r) * T::real(0.2)));
                }
            }
        }
        let extra = 10 * n;
        for i in 0..extra {
            for _ in 0..3 {
                entries.push((n + i, n - 8 + r.below(8), T::random(&mut r)));
            }
        }
        cases.push((
            SparseMatrix::from_coo(n + extra, n, &entries),
            Ordering::Natural,
        ));
        for (a, ordering) in cases {
            let (m, n) = (a.rows(), a.cols());
            let qr = SparseQr::with_ordering(&a, &ordering);
            assert_decomposition(&a, &qr, 32.0);
            assert_eq!(qr.rank(), n, "a well-conditioned matrix lost rank");
            let want = permuted(&a, qr.column_permutation()).qr().r();
            let got = dense(&qr.r());
            for i in 0..n {
                let sign = |x: T| if x.re() < T::Real::ZERO { -1.0 } else { 1.0 };
                let (s, t) = (sign(got[(i, i)]), sign(want[(i, i)]));
                let row = Matrix::from_fn(1, n, |_, j| got[(i, j)] * T::real(s));
                let other = Matrix::from_fn(1, n, |_, j| want[(i, j)] * T::real(t));
                assert_close(&row, &other, 64.0);
            }
            let b = random_dense::<T>(&mut r, m, 2);
            let x = qr.solve(&b).expect("full column rank");
            assert_close(
                &x,
                &dense(&a).qr().solve(&b).expect("full column rank"),
                64.0,
            );
        }
    }
    each_type!(check);
}

#[test]
fn r_is_the_cholesky_factor_of_the_normal_matrix_up_to_the_sign_of_each_row() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0002);
        for (m, n, per_row) in [(1, 1, 1), (6, 6, 2), (20, 8, 3), (size(160, 40), 8, 2)] {
            let a = well_conditioned::<T>(&mut r, m, n, per_row);
            for ordering in [
                Ordering::Natural,
                Ordering::Colamd,
                Ordering::Given(random_permutation(&mut r, n)),
            ] {
                let qr = SparseQr::with_ordering(&a, &ordering);
                let b = permuted(&a, qr.column_permutation());
                let normal = &b.conjugate_transpose() * &b;
                let lh = normal
                    .cholesky()
                    .expect("the normal matrix of a matrix of full column rank is definite")
                    .l()
                    .conjugate_transpose();
                let rr = dense(&qr.r());
                // Row i of R is Lᴴ's times the sign of the diagonal element, which is real.
                let signed = Matrix::from_fn(n, n, |i, j| {
                    if T::re(rr[(i, i)]) < T::Real::ZERO {
                        -lh[(i, j)]
                    } else {
                        lh[(i, j)]
                    }
                });
                // The Cholesky factor of the normal matrix carries the square of the condition
                // number in its rounding; the matrices have one of a few units.
                let cond = b.cond(Norm::Two).to_f64();
                assert_close(&rr, &signed, 4.0 * cond * cond);
            }
        }
    }
    each_type!(check);
}

/// An `m × n` matrix with the strong Hall property: the diagonal and the cyclic superdiagonal of
/// its first `n` rows make every set of fewer than `n` columns reach more rows than it has
/// columns; the other rows hold `extra` random elements.
fn strong_hall<T: Num>(r: &mut Rng, m: usize, n: usize, extra: usize) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, T::real(3.0) + T::random(r)));
        if n > 1 {
            entries.push((i, (i + 1) % n, T::real(1.5) + T::random(r)));
        }
    }
    for i in n..m {
        for _ in 0..extra {
            entries.push((i, r.below(n), T::random(r)));
        }
    }
    SparseMatrix::from_coo(m, n, &entries)
}

#[test]
fn fronts_merged_in_a_large_matrix_give_the_least_squares_and_q_times_r() {
    // From 128 columns on, small fronts are merged into their parents: the least squares of a
    // 70 × 70 grid, the identity over the two differences with random weights. The residual is
    // orthogonal to the columns, and Q·R is A·Q_c on random vectors.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_00a3);
        let k = size(70, 5);
        let n = k * k;
        let mut entries = Vec::new();
        for v in 0..n {
            entries.push((v, v, T::real(1.0 + 0.5 * r.unit().abs())));
        }
        let mut row = n;
        for y in 0..k {
            for x in 0..k {
                let v = y * k + x;
                let right = (x + 1 < k).then_some(v + 1);
                let below = (y + 1 < k).then_some(v + k);
                for w in [right, below].into_iter().flatten() {
                    let c = T::random(&mut r) + T::real(2.0);
                    entries.push((row, v, T::zero() - c));
                    entries.push((row, w, c));
                    row += 1;
                }
            }
        }
        let m = row;
        let a = SparseMatrix::from_coo(m, n, &entries);
        let qr = SparseQr::new(&a);
        assert_eq!(qr.rank(), n);
        let b = random_dense::<T>(&mut r, m, 2);
        let x = qr.solve(&b).expect("full column rank");
        let normal = &a.conjugate_transpose() * &(&a * &x - &b);
        let size_a = a
            .values()
            .iter()
            .map(|v| v.abs().to_f64().powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = size_a * (size_a * norm(&x) + norm(&b));
        assert!(
            norm(&normal) <= 16.0 * T::EPS * m as f64 * scale,
            "Aᴴ·(A·x − b) is {}, not small against {scale}",
            norm(&normal)
        );
        // Q·(R·z) against A·(Q_c·z): row `q[k]` of `Q_c·z` is row `k` of `z`.
        let z = random_dense::<T>(&mut r, n, 2);
        let q = qr.column_permutation();
        let mut place = vec![0; n];
        for (k, &old) in q.iter().enumerate() {
            place[old] = k;
        }
        let spread = Matrix::from_fn(n, 2, |i, j| z[(place[i], j)]);
        assert_close(&qr.apply_q(&(&qr.r() * &z)), &(&a * &spread), 64.0);
    }
    each_type!(check);
}

#[test]
fn the_zeros_that_merged_fronts_add_to_r_stay_exact() {
    // From 128 columns on, small fronts are merged into their parents along paths of the tree,
    // which puts into the rows of `R` zeros of columns those rows do not reach. They must come out
    // exactly zero, so that `r()` keeps the pattern the rows make: for a square matrix with a
    // zero-free diagonal and a connected graph (strong Hall), the symbolic Cholesky factor of
    // the normal matrix in the decomposition's column order, element for element. A merge of
    // siblings leaves rounding residues outside it.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_00a5);
        let k = size(64, 5);
        let n = k * k;
        let mut entries = Vec::new();
        for v in 0..n {
            entries.push((v, v, T::real(4.0) + T::random(&mut r)));
            let (x, y) = (v % k, v / k);
            for (near, w) in [(x + 1 < k, v + 1), (y + 1 < k, v + k)] {
                if near {
                    entries.push((v, w, T::real(-1.0) + T::random(&mut r) * T::real(0.25)));
                    entries.push((w, v, T::real(-1.0) + T::random(&mut r) * T::real(0.25)));
                }
            }
        }
        let a = SparseMatrix::from_coo(n, n, &entries);
        let qr = SparseQr::new(&a);
        assert_eq!(qr.rank(), n);
        // The pattern of `B = A·Q_c` with ones, and that of `Bᵀ·B`: no sum cancels.
        let b = a.select(&(0..n).collect::<Vec<_>>(), qr.column_permutation());
        let ones = vec![1.0f64; b.nnz()];
        let b = SparseMatrix::from_csr(
            n,
            n,
            b.row_pointers().to_vec(),
            b.col_indices().to_vec(),
            ones,
        );
        let normal = &b.transpose() * &b;
        // Row `i` of the Cholesky factor: the columns reached from those of row `i` of the
        // lower triangle by the elimination tree, up to `i`.
        let parent = elimination_tree(&normal);
        let mut mark = vec![usize::MAX; n];
        let mut pattern = 0;
        let rt = qr.r().transpose();
        for i in 0..n {
            mark[i] = i;
            pattern += 1;
            for &j in normal.col_indices()[normal.row_pointers()[i]..normal.row_pointers()[i + 1]]
                .iter()
                .filter(|&&j| j < i)
            {
                let mut c = j;
                while mark[c] != i {
                    mark[c] = i;
                    pattern += 1;
                    c = parent[c].expect("a column below i reaches i");
                }
            }
            // Column `i` of `R` (row `i` of its transpose) lies in row `i` of the factor.
            for &j in &rt.col_indices()[rt.row_pointers()[i]..rt.row_pointers()[i + 1]] {
                assert_eq!(
                    mark[j], i,
                    "R has an element at ({j}, {i}) outside the pattern"
                );
            }
        }
        // And it fills the pattern, but for the odd element that rounding makes exactly zero (up
        // to 6 of its 208 767 in single precision).
        let held = qr.r().nnz();
        assert!(
            held <= pattern && 1000 * (pattern - held) <= pattern,
            "R holds {held} of the {pattern} elements of the pattern"
        );
    }
    each_type!(check);
}

#[test]
fn the_pattern_of_r_is_that_of_the_cholesky_factor_for_a_strong_hall_matrix() {
    // Coleman, Edenbrandt and Gilbert (1986): for a strong Hall matrix the symbolic Cholesky
    // factor of the normal matrix is the pattern of R, element for element. The dense Cholesky
    // leaves exact zeros where the pattern has none, and random elements leave no others.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0003);
        for (m, n, extra) in [(4, 4, 0), (9, 9, 0), (30, 12, 2), (size(120, 30), 20, 1)] {
            let a = strong_hall::<T>(&mut r, m, n, extra);
            for ordering in [
                Ordering::Natural,
                Ordering::Given(random_permutation(&mut r, n)),
            ] {
                let qr = SparseQr::with_ordering(&a, &ordering);
                let b = permuted(&a, qr.column_permutation());
                let l = (&b.conjugate_transpose() * &b).cholesky().unwrap().l();
                let rr = dense(&qr.r());
                for i in 0..n {
                    for j in i..n {
                        assert_eq!(
                            rr[(i, j)].is_zero(),
                            l[(j, i)].is_zero(),
                            "R and the Cholesky factor differ in pattern at ({i}, {j})"
                        );
                    }
                }
            }
        }
    }
    each_type!(check);
}

#[test]
fn apply_q_and_apply_qh_are_the_products_with_q_and_its_adjoint() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0004);
        for (m, n, per_row) in [(1, 1, 1), (7, 7, 2), (25, 6, 2), (size(200, 30), 10, 2)] {
            let a = well_conditioned::<T>(&mut r, m, n, per_row);
            for ordering in [
                Ordering::Colamd,
                Ordering::Given(random_permutation(&mut r, n)),
            ] {
                let qr = SparseQr::with_ordering(&a, &ordering);
                let q = qr.q();
                for count in [0, 1, 3] {
                    let x = random_dense::<T>(&mut r, n, count);
                    let y = random_dense::<T>(&mut r, m, count);
                    let qx = qr.apply_q(&x);
                    let qhy = qr.apply_qh(&y);
                    assert_eq!(qx.shape(), (m, count));
                    assert_eq!(qhy.shape(), (n, count));
                    assert_close(&qx, &(&q * &x), 8.0);
                    assert_close(&qhy, &(&q.conjugate_transpose() * &y), 8.0);
                }
                // Qᴴ·(A·Q_c) is R, and Q·Qᴴ leaves a vector of the range of A as it is.
                let b = permuted(&a, qr.column_permutation());
                assert_close(&qr.apply_qh(&b), &dense(&qr.r()), 16.0);
                let in_range = &a * &random_dense::<T>(&mut r, n, 2);
                assert_close(&qr.apply_q(&qr.apply_qh(&in_range)), &in_range, 16.0);
            }
        }
    }
    each_type!(check);
}

// The least squares.

#[test]
fn least_squares_solutions_agree_with_the_dense_decomposition() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0005);
        // Square, tall, and very tall with twenty rows to a column.
        let shapes = [
            (1, 1, 1),
            (8, 8, 3),
            (30, 10, 3),
            (size(400, 60), size(20, 3), 2),
            (size(1000, 100), size(50, 5), 3),
        ];
        for (m, n, per_row) in shapes {
            let a = well_conditioned::<T>(&mut r, m, n, per_row);
            let b = random_dense::<T>(&mut r, m, 3);
            let want = dense(&a).qr().solve(&b).expect("full column rank");
            for ordering in [
                Ordering::Colamd,
                Ordering::Natural,
                Ordering::Given(random_permutation(&mut r, n)),
            ] {
                let x = SparseQr::with_ordering(&a, &ordering)
                    .solve(&b)
                    .expect("full column rank");
                assert_close(&x, &want, 32.0);
                // The residual of the least squares is orthogonal to the columns of the matrix.
                let normal = &a.conjugate_transpose() * &(&a * &x - &b);
                let size_a = norm(&dense(&a));
                let scale = size_a * (size_a * norm(&x) + norm(&b));
                assert!(
                    norm(&normal) <= 16.0 * T::EPS * m as f64 * scale,
                    "{m} × {n}: Aᴴ·(A·x − b) is {}, not small against {scale}",
                    norm(&normal)
                );
            }
        }
    }
    each_type!(check);
}

#[test]
fn least_squares_solutions_agree_with_numpy_on_the_embedded_cases() {
    // The worst seen over the four orderings, in units of κ·ε: 2.49 in f64 and Complex<f64>,
    // 1.39 in f32 and Complex<f32>, both on the square case; 0.42 and 0.19 on the very tall one,
    // 1.27 and 0.79 on the complex square one.
    fn real<T: Num>() {
        let cases: [(Case, &[f64], f64, f64, f64); 3] = [
            (
                SQUARE,
                &SQUARE_X,
                SQUARE_RESIDUAL,
                SQUARE_COND,
                SQUARE_KAPPA,
            ),
            (TALL, &TALL_X, TALL_RESIDUAL, TALL_COND, TALL_KAPPA),
            (
                VERY_TALL,
                &VERY_TALL_X,
                VERY_TALL_RESIDUAL,
                VERY_TALL_COND,
                VERY_TALL_KAPPA,
            ),
        ];
        for (case, x, residual, cond, kappa) in cases {
            let want = pairs(x);
            check_embedded(&case.matrix::<T>(), &case.b(), &want, residual, cond, kappa);
        }
        let (a, b) = smoothing::<T>(40);
        let want = pairs(&SMOOTHING_X);
        check_embedded(
            &a,
            &b,
            &want,
            SMOOTHING_RESIDUAL,
            SMOOTHING_COND,
            SMOOTHING_KAPPA,
        );
    }
    fn complex<T: Num>() {
        let cases: [(Case, &Parts, f64, f64, f64); 3] = [
            (
                COMPLEX_SQUARE,
                &COMPLEX_SQUARE_X,
                COMPLEX_SQUARE_RESIDUAL,
                COMPLEX_SQUARE_COND,
                COMPLEX_SQUARE_KAPPA,
            ),
            (
                COMPLEX_TALL,
                &COMPLEX_TALL_X,
                COMPLEX_TALL_RESIDUAL,
                COMPLEX_TALL_COND,
                COMPLEX_TALL_KAPPA,
            ),
            (
                COMPLEX_VERY_TALL,
                &COMPLEX_VERY_TALL_X,
                COMPLEX_VERY_TALL_RESIDUAL,
                COMPLEX_VERY_TALL_COND,
                COMPLEX_VERY_TALL_KAPPA,
            ),
        ];
        for (case, want, residual, cond, kappa) in cases {
            check_embedded(&case.matrix::<T>(), &case.b(), want, residual, cond, kappa);
        }
    }
    // The real cases in every type, a complex matrix with no imaginary parts among them.
    each_type!(real);
    complex::<a::math::complex::Complex<f32>>();
    complex::<a::math::complex::Complex<f64>>();
}

#[test]
fn every_ordering_gives_the_same_least_squares_solution() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0007);
        for (m, n, per_row) in [(9, 9, 3), (40, 12, 3), (size(300, 40), size(15, 4), 2)] {
            let a = well_conditioned::<T>(&mut r, m, n, per_row);
            let b = random_dense::<T>(&mut r, m, 2);
            let first = SparseQr::new(&a).solve(&b).unwrap();
            for ordering in orderings(&mut r, n) {
                let x = SparseQr::with_ordering(&a, &ordering).solve(&b).unwrap();
                assert_close(&x, &first, 32.0);
            }
        }
    }
    each_type!(check);
}

#[test]
fn several_right_hand_sides_are_solved_as_each_alone() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0008);
        let a = well_conditioned::<T>(&mut r, 30, 9, 3);
        let b = random_dense::<T>(&mut r, 30, 4);
        let qr = SparseQr::new(&a);
        let x = qr.solve(&b).unwrap();
        assert_eq!(x.shape(), (9, 4));
        // Each column goes through the same operations whatever the others are.
        for c in 0..4 {
            let alone = qr.solve_vec(b.column(c)).unwrap();
            assert_eq!(x.column(c), &alone[..], "column {c} differs");
        }
        assert_eq!(qr.solve(&Matrix::zeros(30, 0)).unwrap().shape(), (9, 0));
    }
    each_type!(check);
}

#[test]
fn a_zero_right_hand_side_gives_a_zero_solution() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0009);
        let a = well_conditioned::<T>(&mut r, 20, 7, 2);
        let x = SparseQr::new(&a).solve(&Matrix::zeros(20, 2)).unwrap();
        assert!(x.as_slice().iter().all(|x| x.is_zero()));
    }
    each_type!(check);
}

#[test]
fn a_square_system_is_solved_exactly() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_000a);
        for n in [1, 2, 5, 17, size(60, 6)] {
            let a = well_conditioned::<T>(&mut r, n, n, 3);
            let x_true = random_dense::<T>(&mut r, n, 2);
            let b = &a * &x_true;
            let x = SparseQr::new(&a).solve(&b).unwrap();
            assert_close(&x, &x_true, 16.0);
            assert_close(&(&a * &x), &b, 16.0);
        }
    }
    each_type!(check);
}

// Rank.

/// Makes column `to` of `d` depend on its columns `from` and `other`, as the kind says: a copy,
/// a negated copy, a multiple by a number of absolute value 2 to 10, a multiple by a power of
/// two, a sum, a combination with small integers, or an empty column. The powers run from 2⁻²⁰
/// to 2²⁰ in double precision and from 2⁻⁶ to 2⁶ in single: columns of norms 2¹⁸ apart, which
/// two powers of 2^±10 could make, leave the smaller below the tolerance of the rank in single
/// precision, which is no clear case.
fn make_dependent<T: Num>(
    r: &mut Rng,
    d: &mut Matrix<T>,
    kind: usize,
    to: usize,
    from: usize,
    other: usize,
) {
    let c = (T::random(r) + T::real(1.5)) * T::real(if r.below(2) == 0 { 4.0 } else { -4.0 });
    let reach = if T::EPS < 1e-10 { 20 } else { 6 };
    let power = T::real(2f64.powi(r.below(2 * reach + 1) as i32 - reach as i32));
    let (k1, k2) = (
        T::real(r.below(7) as f64 - 3.0),
        T::real(r.below(5) as f64 + 1.0),
    );
    for i in 0..d.rows() {
        let (x, y) = (d[(i, from)], d[(i, other)]);
        d[(i, to)] = match kind {
            0 => x,
            1 => -x,
            2 => x * c,
            3 => x * power,
            4 => x + y,
            5 => x * k1 + y * k2,
            _ => T::zero(),
        };
    }
}

#[test]
fn rank_deficient_matrices_are_counted_and_refused() {
    // The count is the rank when the dependent columns are eliminated after the others, and in
    // a matrix of five times as many rows as columns under any of these orders. In a square one,
    // a dependent column eliminated early still takes a row of R, which an independent column
    // after it can miss, as in the dense decomposition: the count can come out lower then, and
    // the solution is refused whatever it is. With dependencies of these small coefficients the
    // count never came out higher; large ones can make it so (the tests after this one).
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_000b);
        for trial in 0..size(160, 16) {
            let (ratio, kind) = ([1, 2, 5, 20][trial % 4], (trial / 4) % 7);
            let n = 2 + r.below(9);
            let m = ratio * n;
            let mut d = dense(&well_conditioned::<T>(&mut r, m, n, 2));
            // Some columns made dependent on the ones left independent.
            let place = random_permutation(&mut r, n);
            let dependent = 1 + r.below(n / 2);
            let independent = &place[dependent..];
            for &to in &place[..dependent] {
                let from = independent[r.below(independent.len())];
                let other = independent[r.below(independent.len())];
                make_dependent(&mut r, &mut d, kind, to, from, other);
            }
            let a = SparseMatrix::from_matrix(&d);
            let rank = n - dependent;
            assert_eq!(d.rank(), rank, "the construction went wrong");
            let mut last = independent.to_vec();
            last.extend_from_slice(&place[..dependent]);
            let exact = Ordering::Given(last);
            let random = Ordering::Given(random_permutation(&mut r, n));
            for ordering in [exact.clone(), random.clone(), Ordering::Colamd] {
                let qr = SparseQr::with_ordering(&a, &ordering);
                let what = format!("{m} × {n}, kind {kind}, {ordering:?}");
                if ordering == exact || (ordering == random && ratio >= 5) {
                    assert_eq!(qr.rank(), rank, "{what}");
                } else {
                    assert!(qr.rank() <= rank, "{what}: {} is above the rank", qr.rank());
                }
                assert!(qr.solve(&random_dense::<T>(&mut r, m, 1)).is_none());
                assert!(qr.solve_vec(&vec![T::one(); m]).is_none());
                // The decomposition is made all the same.
                assert_decomposition(&a, &qr, 16.0);
            }
        }
    }
    each_type!(check);
}

#[test]
fn the_rank_of_the_embedded_deficient_cases_is_numpys() {
    // In the natural order each dependent column comes after the ones it depends on, and the
    // count is the rank; the fill-reducing orders may count fewer, never more.
    fn assert_rank<T: Num>(a: &SparseMatrix<T>, b: &Matrix<T>, rank: usize) {
        for ordering in [Ordering::Natural, Ordering::Colamd, Ordering::Amd] {
            let qr = SparseQr::with_ordering(a, &ordering);
            if ordering == Ordering::Natural {
                assert_eq!(qr.rank(), rank);
            } else {
                assert!(qr.rank() <= rank, "{ordering:?} counted {}", qr.rank());
            }
            assert!(qr.solve(b).is_none());
        }
    }
    // Column 5 is the sum of columns 2 and 3, column 7 is empty.
    fn real<T: Num>() {
        let mut d = DEFICIENT.dense::<T>();
        for i in 0..DEFICIENT.m {
            d[(i, 5)] = d[(i, 2)] + d[(i, 3)];
            d[(i, 7)] = T::zero();
        }
        let a = SparseMatrix::from_matrix(&d);
        assert_rank(&a, &DEFICIENT.b(), DEFICIENT_RANK);
    }
    // Column 4 is column 1 times 1 + i, column 2 a copy of column 0.
    fn complex<T: Num>() {
        let mut d = COMPLEX_DEFICIENT.dense::<T>();
        for i in 0..COMPLEX_DEFICIENT.m {
            d[(i, 4)] = d[(i, 1)] * element(1.0, 1.0);
            d[(i, 2)] = d[(i, 0)];
        }
        let a = SparseMatrix::from_matrix(&d);
        assert_rank(&a, &COMPLEX_DEFICIENT.b(), COMPLEX_DEFICIENT_RANK);
    }
    each_type!(real);
    complex::<a::math::complex::Complex<f32>>();
    complex::<a::math::complex::Complex<f64>>();
}

#[test]
fn a_column_that_is_a_large_multiple_of_an_earlier_one_is_found_dependent() {
    // A tolerance relative to the largest diagonal element of R, ‖a‖ here, missed this: the
    // second column leaves a rounding error of the order of ε·2²⁰·‖a‖ on the diagonal, and the
    // f64 solution came out as (−1.1e15, 1.1e9). In f32 the multiple is 2¹²: a column 2²⁰ times
    // larger would leave the first below the tolerance, and the count at 0.
    fn check<T: Num>() {
        let column = [0.1, 0.7, 0.3];
        let large = if T::EPS < 1e-10 { 1048576.0 } else { 4096.0 };
        for (first, second) in [(1.0, large), (large, 1.0), (1.0, -0.5)] {
            let mut entries = Vec::new();
            for (i, &x) in column.iter().enumerate() {
                entries.push((i, 0, T::real(x * first)));
                entries.push((i, 1, T::real(x * second)));
            }
            let a = SparseMatrix::from_coo(3, 2, &entries);
            let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
            assert_eq!(qr.rank(), 1, "columns {first}·a and {second}·a");
            assert!(
                qr.solve_vec(&[T::one(), T::real(2.0), T::real(3.0)])
                    .is_none()
            );
        }
    }
    each_type!(check);
}

#[test]
fn an_exact_copy_in_a_two_by_two_matrix_is_found_dependent() {
    // Three of 200000 random [[x, x], [y, y]] that a tolerance of max(m, n)·ε times the largest
    // diagonal element missed: their R₁₁ came out at 2.3ε to 2.6ε of R₀₀.
    let pairs = [
        (2.9084102658268884e-2, -2.4057148592656485e-1),
        (-2.5713581701897525e-1, 5.443019262277624e-1),
        (-3.391974564881386e-1, -8.688911199101892e-1),
    ];
    for (x, y) in pairs {
        let a = SparseMatrix::from_coo(2, 2, &[(0, 0, x), (0, 1, x), (1, 0, y), (1, 1, y)]);
        let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
        assert_eq!(qr.rank(), 1, "[[{x}, {x}], [{y}, {y}]]");
        assert!(qr.solve_vec(&[1.0, 0.0]).is_none());
    }
}

#[test]
fn the_tolerance_of_the_rank_is_relative_to_the_largest_column_norm() {
    // A column of a hundred ones, of norm 10 though no element is above 1, and two columns of
    // norm δ·√2 orthogonal to it and to each other, so that R is diagonal: 10, δ·√2 and δ·√2 in
    // some order. With δ·√2 = 362·ε the two are below the tolerance, 100·ε·10, though above
    // 100·ε times the largest element; with δ·√2 = 5793·ε they are above it, and the condition
    // number, 10/(5793·ε), is below the limit 1/(100·ε).
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0014);
        for (shift, rank) in [(8, 1), (12, 3)] {
            let delta = T::EPS * f64::powi(2.0, shift);
            let mut entries: Vec<(usize, usize, T)> = (0..100).map(|i| (i, 0, T::one())).collect();
            entries.extend([
                (0, 1, T::real(delta)),
                (1, 1, T::real(-delta)),
                (2, 2, T::real(delta)),
                (3, 2, T::real(-delta)),
            ]);
            let a = SparseMatrix::from_coo(100, 3, &entries);
            for ordering in orderings(&mut r, 3) {
                let qr = SparseQr::with_ordering(&a, &ordering);
                assert_eq!(qr.rank(), rank, "δ = 2^{shift}·ε, {ordering:?}");
                let solved = qr.solve_vec(&[T::one(); 100]).is_some();
                assert_eq!(solved, rank == 3, "δ = 2^{shift}·ε, {ordering:?}");
            }
        }
    }
    each_type!(check);
}

/// Every permutation of `0..n`, for a small `n`.
fn every_permutation(n: usize) -> Vec<Vec<usize>> {
    let mut all = vec![Vec::new()];
    for k in 0..n {
        all = all
            .into_iter()
            .flat_map(|p: Vec<usize>| {
                (0..=p.len()).map(move |at| {
                    let mut q = p.clone();
                    q.insert(at, k);
                    q
                })
            })
            .collect();
    }
    all
}

/// The orderings a singular matrix is tried with: every kind, and every permutation of the
/// columns of a matrix of at most five, random ones of a larger.
fn every_order(r: &mut Rng, n: usize) -> Vec<Ordering> {
    let mut list = vec![
        Ordering::Natural,
        Ordering::ReverseCuthillMckee,
        Ordering::Amd,
        Ordering::Colamd,
        Ordering::NestedDissection,
    ];
    if n <= 5 && !cfg!(miri) {
        list.extend(every_permutation(n).into_iter().map(Ordering::Given));
    } else {
        for _ in 0..size(24, 3) {
            list.push(Ordering::Given(random_permutation(r, n)));
        }
    }
    list
}

/// What a matrix of rank `rank` below its number of columns must give in every order: a count
/// below the number of columns and no higher than the rank, no solution, and a valid
/// decomposition all the same.
fn assert_refused_in_every_order<T: Num>(r: &mut Rng, a: &SparseMatrix<T>, rank: usize) {
    let (m, n) = (a.rows(), a.cols());
    assert!(rank < n);
    for ordering in every_order(r, n) {
        let qr = SparseQr::with_ordering(a, &ordering);
        let what = format!("{m} × {n} of rank {rank}, {ordering:?}");
        assert!(qr.rank() <= rank, "{what}: counted {}", qr.rank());
        assert!(qr.solve_vec(&vec![T::one(); m]).is_none(), "{what}: solved");
        assert!(
            qr.solve(&random_dense::<T>(r, m, 2)).is_none(),
            "{what}: solved"
        );
        assert_decomposition(a, &qr, 16.0);
    }
}

#[test]
fn a_column_two_to_the_eight_times_the_difference_of_two_others_is_refused() {
    // Column 2 is 2⁸·(column 1 − column 0), exactly, the elements being dyadic: rank 2 (sympy,
    // in `gen_qr.py`). The rounding error the elimination leaves on the last diagonal element
    // is amplified by the coefficients, to 5.9·10⁻¹⁴ in f64 against a tolerance of 2.7·10⁻¹⁴,
    // and without the condition estimate the matrix was counted of rank 3 and solved, with
    // |x| near 4·10¹⁵, in all four types and in the natural, Colamd and [1, 0, 2] orders. In
    // the complex types every element is also turned by the phase (3 + 4i)/5, which keeps the
    // rank and spoils the exactness.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0010);
        let rows = [
            [-1.0, -1.0 + 2.0 / 256.0, 2.0],
            [-3.0, -3.0 + 3.0 / 256.0, 3.0],
            [2.0, 2.0 + 1.0 / 256.0, 1.0],
        ];
        let phases: &[(f64, f64)] = if T::IS_COMPLEX {
            &[(1.0, 0.0), (0.6, 0.8)]
        } else {
            &[(1.0, 0.0)]
        };
        for &(re, im) in phases {
            let mut entries = Vec::new();
            for (i, row) in rows.iter().enumerate() {
                for (j, &x) in row.iter().enumerate() {
                    entries.push((i, j, T::real(x) * element(re, im)));
                }
            }
            let a = SparseMatrix::from_coo(3, 3, &entries);
            assert_refused_in_every_order(&mut r, &a, 2);
            for ordering in every_order(&mut r, 3) {
                assert_eq!(SparseQr::with_ordering(&a, &ordering).rank(), 2);
            }
        }
    }
    each_type!(check);
}

#[test]
fn a_structurally_rank_deficient_matrix_is_refused_in_every_column_order() {
    // Columns 0, 2 and 4 of the 13 × 5 matrix have elements in rows 0 and 2 alone, three columns
    // in two rows: of rank 4 whatever the values (scipy's structural_rank, in `gen_qr.py`). In the
    // order [1, 3, 2, 0, 4] its last diagonal element came out 8.9·10⁻¹⁵ against a tolerance of
    // 8.8·10⁻¹⁵, and it was solved, with |x| near 5·10¹⁵. The 18 × 16 matrix with a dense row is
    // of structural rank 15, and was solved in the reverse of the natural order.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0011);
        let entries: Vec<_> = STRUCTURAL
            .iter()
            .map(|&(i, j, x)| (i, j, T::real(x)))
            .collect();
        let a = SparseMatrix::from_coo(13, 5, &entries);
        assert_refused_in_every_order(&mut r, &a, STRUCTURAL_RANK);
        let entries: Vec<_> = DENSE_ROW
            .iter()
            .map(|&(i, j, x)| (i, j, T::real(x)))
            .collect();
        let a = SparseMatrix::from_coo(18, 16, &entries);
        assert_refused_in_every_order(&mut r, &a, DENSE_ROW_RANK);
        let reverse = SparseQr::with_ordering(&a, &Ordering::Given((0..16).rev().collect()));
        assert!(reverse.solve_vec(&[T::one(); 18]).is_none());
    }
    each_type!(check);
}

#[test]
fn a_triangle_with_no_small_diagonal_element_can_be_singular_to_working_precision() {
    // The upper bidiagonal matrix of 1 on the diagonal and −2 above it is its own R in the
    // natural order, with every diagonal element 1, and its condition number grows as 2ⁿ:
    // 2.2·10¹² for n = 40, below the limit of f64, 1/(40·ε) = 1.1·10¹⁴, and far above that of
    // f32; 2.3·10¹⁸ for n = 60, above both (mpmath, in `gen_qr.py`). Without the condition
    // estimate the second was solved, with |x| = 5·10¹⁷ and a residual as large as b.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0013);
        for (n, cond) in BIDIAGONAL_COND {
            let a = SparseMatrix::from_fn(n, n, |i, j| {
                if i == j {
                    T::one()
                } else if j == i + 1 {
                    T::real(-2.0)
                } else {
                    T::zero()
                }
            });
            let limit = 1.0 / (n.max(32) as f64 * T::EPS);
            let x_true = random_dense::<T>(&mut r, n, 1);
            let b = &a * &x_true;
            for ordering in [
                Ordering::Natural,
                Ordering::Colamd,
                Ordering::Given(random_permutation(&mut r, n)),
            ] {
                let qr = SparseQr::with_ordering(&a, &ordering);
                let what = format!("n = {n}, condition number {cond:.1e}, {ordering:?}");
                match qr.solve(&b) {
                    Some(x) => {
                        assert!(cond < limit, "{what}: solved beyond the limit {limit:.1e}");
                        assert_eq!(qr.rank(), n, "{what}");
                        // Backward stable, and as accurate as the condition number allows.
                        let error = norm(&(&x - &x_true)) / norm(&x_true);
                        assert!(error <= 16.0 * cond * T::EPS, "{what}: error {error:.1e}");
                        let scale = norm(&dense(&a)) * norm(&x) + norm(&b);
                        let residual = norm(&(&a * &x - &b));
                        assert!(residual <= 16.0 * n as f64 * T::EPS * scale, "{what}");
                    }
                    None => {
                        assert!(cond > limit, "{what}: refused below the limit {limit:.1e}");
                        assert!(qr.rank() < n, "{what}");
                    }
                }
            }
        }
    }
    each_type!(check);
}

/// A matrix of 3 to 11 columns with three of them, at random places, `a`, `b = a + 2⁻ᵖ·c` and
/// `c`, for `p` from 4 to 11, their parts integers below 5 in magnitude, so that `c = 2ᵖ·(b − a)`
/// exactly in every type; the other columns of three random elements. With `p`.
fn large_combination<T: Num>(r: &mut Rng) -> (SparseMatrix<T>, i32) {
    let integer = |r: &mut Rng| -> T {
        let re = r.below(9) as f64 - 4.0;
        let im = if T::IS_COMPLEX {
            r.below(9) as f64 - 4.0
        } else {
            0.0
        };
        if r.below(3) == 0 {
            T::zero()
        } else {
            element(re, im)
        }
    };
    let n = 3 + r.below(9);
    let m = n + r.below(2 * n + 1);
    let p = 4 + r.below(8) as i32;
    let mut d = Matrix::from_fn(m, n, |_, _| T::zero());
    for j in 0..n {
        for _ in 0..3 {
            d[(r.below(m), j)] = T::random(r);
        }
    }
    let place = random_permutation(r, n);
    let small = T::real(f64::powi(2.0, -p));
    for i in 0..m {
        let (x, w) = (integer(r), integer(r));
        d[(i, place[0])] = x;
        d[(i, place[1])] = x + w * small;
        d[(i, place[2])] = w;
    }
    (SparseMatrix::from_matrix(&d), p)
}

/// The orderings [`large_combination`]'s matrices are tried with.
fn four_orders(r: &mut Rng, n: usize) -> [Ordering; 4] {
    [
        Ordering::Natural,
        Ordering::Colamd,
        Ordering::Given(random_permutation(r, n)),
        Ordering::Given(random_permutation(r, n)),
    ]
}

#[test]
fn a_triangle_that_hides_its_singularity_from_the_first_vector_is_refused() {
    // R = [[1, 0, a], [0, 1, −a], [0, 0, 1]] is its own R in the natural order, with every
    // diagonal element 1. LINPACK's vector takes e = (1, 1, 1), for which the two a cancel:
    // Rᴴ·y = e gives y = (1, 1, 1), and an estimate of about √2·a, below the limit for the a
    // chosen (2¹⁰ in single precision, 2³⁰ in double). The condition number is about 2·a², far
    // above it; the solve with R that follows finds R⁻¹·y = (1 − a, 1 + a, 1), and the estimate
    // 2·a²/√3.
    fn check<T: Num>() {
        let a = if T::EPS < 1e-10 {
            f64::powi(2.0, 30)
        } else {
            f64::powi(2.0, 10)
        };
        let limit = 1.0 / (32.0 * T::EPS);
        assert!(f64::sqrt(2.0) * a < limit && limit < a * a);
        let entries = [
            (0, 0, T::one()),
            (0, 2, T::real(a)),
            (1, 1, T::one()),
            (1, 2, T::real(-a)),
            (2, 2, T::one()),
        ];
        let matrix = SparseMatrix::from_coo(3, 3, &entries);
        let qr = SparseQr::with_ordering(&matrix, &Ordering::Natural);
        assert_eq!(dense(&qr.r()), dense(&matrix), "R isn't the matrix itself");
        assert_eq!(qr.rank(), 2);
        assert!(qr.solve_vec(&[T::one(); 3]).is_none());
    }
    each_type!(check);
}

#[test]
fn columns_made_dependent_with_large_coefficients_are_refused() {
    // The rounding error the elimination leaves on the diagonal grows with 2ᵖ: without the
    // condition estimate the review solved a fifth to a quarter of such matrices, and 139 292
    // of 199 698 small integer instances.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0012);
        for _ in 0..size(400, 20) {
            let (a, p) = large_combination::<T>(&mut r);
            let (m, n) = (a.rows(), a.cols());
            for ordering in four_orders(&mut r, n) {
                let qr = SparseQr::with_ordering(&a, &ordering);
                let what = format!("{m} × {n}, p = {p}, {ordering:?}");
                assert!(qr.rank() < n, "{what}: counted {}", qr.rank());
                assert!(qr.solve_vec(&vec![T::one(); m]).is_none(), "{what}");
            }
        }
    }
    each_type!(check);
}

/// The 2-norm condition number of `a`, from the singular values of its dense form in double
/// precision.
fn condition<T: Num>(a: &SparseMatrix<T>) -> f64 {
    let d = dense(a);
    let wide = Matrix::from_fn(d.rows(), d.cols(), |i, j| {
        let x = d[(i, j)];
        a::math::complex::Complex::new(T::re(x).to_f64(), T::im(x).to_f64())
    });
    wide.cond(Norm::Two)
}

/// A matrix near the condition limit of `solve`, of a condition number of about `target`, made
/// in one of two ways that leave no small element on the diagonal of `R`: an upper bidiagonal
/// matrix of 1 on the diagonal and about `−β` above it, and a sparse matrix one of whose columns
/// is a combination of others with coefficients up to 2¹¹, plus a random part of size
/// `1/target`.
fn near_singular<T: Num>(r: &mut Rng, kind: usize, target: f64) -> SparseMatrix<T> {
    let n = 3 + r.below(22);
    if kind == 0 {
        let beta = f64::powf(target, 1.0 / (n - 1) as f64).max(1.0);
        let mut entries = Vec::new();
        for i in 0..n {
            entries.push((i, i, T::one()));
            if i + 1 < n {
                let wobble = T::one() + T::random(r) * T::real(0.01);
                entries.push((i, i + 1, T::real(-beta) * wobble));
            }
        }
        return SparseMatrix::from_coo(n, n, &entries);
    }
    let m = n * [1, 2, 5][r.below(3)] + r.below(3);
    let mut d = Matrix::from_fn(m, n, |_, _| T::zero());
    for j in 0..n {
        for _ in 0..3 {
            d[(r.below(m), j)] = T::random(r);
        }
        d[(j, j)] = T::one() + T::random(r) * T::real(0.25);
    }
    let to = r.below(n);
    let mut parts = Vec::new();
    for _ in 0..1 + r.below(3) {
        let from = r.below(n);
        if from != to {
            parts.push((
                from,
                T::random(r) * T::real(f64::powi(2.0, r.below(12) as i32)),
            ));
        }
    }
    for i in 0..m {
        let mut x = T::random(r) * T::real(1.0 / target);
        for &(from, c) in &parts {
            x += d[(i, from)] * c;
        }
        d[(i, to)] = x;
    }
    SparseMatrix::from_matrix(&d)
}

#[test]
fn matrices_near_the_condition_limit_are_solved_below_it_and_refused_beyond_it() {
    // The condition numbers spread from a hundredth of the limit 1/(max(m, 32)·ε) to a hundred
    // times it. The estimate is a lower bound, which in the measurements refused no matrix
    // below the limit and solved none beyond twice it. Here the margins are 2 on both sides.
    // Whatever the matrix, the count of the rank is below the number of columns exactly when the
    // solution is refused.
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0015);
        let (mut below, mut beyond) = (0, 0);
        for trial in 0..size(240, 16) {
            let kind = trial % 2;
            // The limit depends on m, which the matrix decides: a target in units of a limit of
            // 32 rows, put right below.
            let unit = 1.0 / (32.0 * T::EPS);
            let target = unit * f64::powf(100.0, 2.0 * r.unit());
            let a = near_singular::<T>(&mut r, kind, target);
            let (m, n) = (a.rows(), a.cols());
            let limit = 1.0 / (m.max(32) as f64 * T::EPS);
            let cond = condition(&a);
            let ordering = Ordering::Given(random_permutation(&mut r, n));
            let qr = SparseQr::with_ordering(&a, &ordering);
            let solved = qr.solve_vec(&vec![T::one(); m]).is_some();
            let what = format!(
                "kind {kind}, {m} × {n}, condition number {:.2} limits",
                cond / limit
            );
            assert_eq!(
                solved,
                qr.rank() == n,
                "{what}: the count and solve disagree"
            );
            if cond < limit / 2.0 {
                assert!(solved, "{what}: refused");
                below += 1;
            } else if cond > 2.0 * limit {
                assert!(!solved, "{what}: solved");
                beyond += 1;
            }
        }
        // Both sides of the limit were tried.
        assert!(
            below > size(40, 2) && beyond > size(40, 2),
            "{below} below, {beyond} beyond"
        );
    }
    each_type!(check);
}

#[test]
fn a_solution_beyond_the_range_of_the_type_is_refused() {
    // [s; s] with s a sixteenth of the smallest normal number has the condition number 1, and
    // the solution of [1; 1], 1/s, overflows: it was returned as an infinity. A right-hand side
    // that holds a NaN or an infinity has no solution either; one near the top of the range has.
    fn check<T: Num>() {
        let tiny = T::Real::MIN_POSITIVE.to_f64() / 16.0;
        let a = SparseMatrix::from_coo(2, 1, &[(0, 0, T::real(tiny)), (1, 0, T::real(tiny))]);
        let qr = SparseQr::new(&a);
        assert_eq!(qr.rank(), 1);
        assert!(qr.solve_vec(&[T::one(), T::one()]).is_none());
        let a = SparseMatrix::from_coo(2, 1, &[(0, 0, T::one()), (1, 0, T::one())]);
        let qr = SparseQr::new(&a);
        for bad in [T::Real::NAN, T::Real::INFINITY, -T::Real::INFINITY] {
            assert!(qr.solve_vec(&[T::from_real(bad), T::one()]).is_none());
            assert!(qr.solve_vec(&[T::one(), T::from_real(bad)]).is_none());
        }
        let big = T::Real::MAX.to_f64() / 4.0;
        let x = qr.solve_vec(&[T::real(big), T::real(big)]).unwrap();
        assert!((T::re(x[0]).to_f64() / big - 1.0).abs() <= 4.0 * T::EPS);
    }
    each_type!(check);
}

#[test]
fn empty_columns_and_rows_leave_a_valid_decomposition() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_000c);
        // Column 2 and row 6 store nothing; the other columns are diagonally dominant.
        let mut entries = Vec::new();
        for i in 0..5 {
            if i != 2 {
                entries.push((i, i, T::real(4.0) + T::random(&mut r)));
            }
        }
        for i in [5, 7] {
            for j in [0, 1, 3, 4] {
                entries.push((i, j, T::random(&mut r)));
            }
        }
        let a = SparseMatrix::from_coo(8, 5, &entries);
        for ordering in orderings(&mut r, 5) {
            let qr = SparseQr::with_ordering(&a, &ordering);
            assert_decomposition(&a, &qr, 16.0);
            assert_eq!(qr.rank(), 4);
            // The empty column has an empty row of R.
            let at = qr
                .column_permutation()
                .iter()
                .position(|&c| c == 2)
                .unwrap();
            assert_eq!(qr.r().row(at).count(), 0);
            assert!(qr.solve(&Matrix::zeros(8, 1)).is_none());
        }
        // Nothing stored at all: rank 0, and Q is still a set of orthonormal columns.
        let zero = SparseMatrix::<T>::zeros(6, 4);
        let qr = SparseQr::new(&zero);
        assert_decomposition(&zero, &qr, 1.0);
        assert_eq!((qr.rank(), qr.r().nnz()), (0, 0));
        assert!(qr.solve_vec(&[T::zero(); 6]).is_none());
    }
    each_type!(check);
}

#[test]
fn ill_conditioned_matrices_are_solved_until_the_tolerance_refuses_them() {
    // The tall case with column j scaled by 2⁻ʲ, and by 4⁻ʲ: condition numbers of about 6·10³
    // and 1.2·10⁷. Householder's reflections are unchanged by a scaling of the columns, so the
    // solution scaled back is NumPy's to the accuracy of the unscaled case. The second passes the
    // tolerance of the rank in f64 and not in f32, where 1.2·10⁷ is beyond 1/(m·ε).
    fn check<T: Num>() {
        for (base, solved) in [(2.0, true), (4.0, T::EPS < 1e-10)] {
            let scale = |j: usize| 1.0 / f64::powi(base, j as i32);
            let a = TALL.matrix::<T>().map(|_, j, x| x * T::real(scale(j)));
            let qr = SparseQr::new(&a);
            match qr.solve(&TALL.b()) {
                Some(y) => {
                    assert!(solved, "a numerically rank-deficient f32 matrix was solved");
                    let x: Vec<T> = (0..TALL.n).map(|j| y[(j, 0)] * T::real(scale(j))).collect();
                    let error = relative_error(&x, &pairs(&TALL_X)) / (TALL_KAPPA * T::EPS);
                    assert!(error <= EMBEDDED_BOUND, "{error:.2} units from NumPy's");
                }
                None => assert!(
                    !solved,
                    "a matrix of condition number {base}^11 was refused"
                ),
            }
        }
    }
    each_type!(check);
}

#[test]
fn matrices_near_overflow_and_underflow_are_solved_as_well_as_moderate_ones() {
    fn check<T: Num>() {
        let exponent = if T::EPS < 1e-10 { 1000 } else { 100 };
        for e in [exponent, -exponent] {
            let s = 2f64.powi(e);
            let a = TALL.matrix::<T>().map(|_, _, x| x * T::real(s));
            let qr = SparseQr::new(&a);
            assert_eq!(qr.rank(), TALL.n);
            let y = qr.solve(&TALL.b()).expect("a scaled matrix of full rank");
            let x: Vec<T> = y.as_slice().iter().map(|&y| y * T::real(s)).collect();
            let error = relative_error(&x, &pairs(&TALL_X)) / (TALL_KAPPA * T::EPS);
            assert!(error <= EMBEDDED_BOUND, "scale 2^{e}: {error:.2} units");
        }
    }
    each_type!(check);
}

#[test]
fn a_nan_or_an_infinity_in_the_matrix_is_refused() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_000d);
        let a = well_conditioned::<T>(&mut r, 9, 5, 2);
        let b = random_dense::<T>(&mut r, 9, 1);
        assert!(SparseQr::new(&a).solve(&b).is_some());
        // Every stored element in turn, whichever front and row of a front it lands in.
        for bad in [T::Real::NAN, T::Real::INFINITY, -T::Real::INFINITY] {
            for k in 0..a.nnz() {
                let mut at = 0;
                let spoilt = a.map(|_, _, x| {
                    at += 1;
                    if at == k + 1 { T::from_real(bad) } else { x }
                });
                for ordering in [Ordering::Natural, Ordering::Colamd] {
                    let qr = SparseQr::with_ordering(&spoilt, &ordering);
                    assert!(qr.solve(&b).is_none(), "element {k} set to {bad:?}");
                }
            }
        }
    }
    each_type!(check);
}

// Small and empty matrices, and the signs of R.

#[test]
fn matrices_without_columns_have_an_empty_decomposition() {
    fn check<T: Num>() {
        for m in [0, 3] {
            let a = SparseMatrix::<T>::zeros(m, 0);
            let qr = SparseQr::new(&a);
            assert_eq!((qr.rank(), qr.nnz(), qr.r().rows()), (0, 0, 0));
            assert_eq!(qr.q().shape(), (m, 0));
            assert!(qr.column_permutation().is_empty());
            assert_eq!(qr.solve(&Matrix::zeros(m, 2)).unwrap().shape(), (0, 2));
            assert!(qr.solve_vec(&vec![T::one(); m]).unwrap().is_empty());
            assert_eq!(qr.apply_q(&Matrix::zeros(0, 2)).shape(), (m, 2));
            assert_eq!(qr.apply_qh(&Matrix::zeros(m, 2)).shape(), (0, 2));
        }
    }
    each_type!(check);
}

#[test]
fn a_one_by_one_matrix_is_its_own_r_up_to_a_phase() {
    fn check<T: Num>() {
        // A real element is left as it is: there is nothing to reflect.
        for x in [3.0, -2.5] {
            let a = SparseMatrix::from_coo(1, 1, &[(0, 0, T::real(x))]);
            let qr = SparseQr::new(&a);
            assert_eq!(qr.r()[(0, 0)], T::real(x));
            assert_eq!(qr.q()[(0, 0)], T::one());
            assert_eq!(qr.solve_vec(&[T::real(2.0 * x)]).unwrap(), [T::real(2.0)]);
        }
        // Zero has rank 0.
        let qr = SparseQr::new(&SparseMatrix::<T>::zeros(1, 1));
        assert_eq!((qr.rank(), qr.q()[(0, 0)]), (0, T::one()));
        assert!(qr.solve_vec(&[T::one()]).is_none());
    }
    each_type!(check);
    // A complex element is turned real: 3 + 4i = (−0.6 − 0.8i)·(−5).
    fn complex<T: Num>() {
        let z = element::<T>(3.0, 4.0);
        let qr = SparseQr::new(&SparseMatrix::from_coo(1, 1, &[(0, 0, z)]));
        assert!((qr.r()[(0, 0)] - T::real(-5.0)).abs().to_f64() <= 4.0 * T::EPS);
        let q = qr.q()[(0, 0)];
        assert!((q - element(-0.6, -0.8)).abs().to_f64() <= 4.0 * T::EPS);
        let x = qr.solve_vec(&[z]).unwrap();
        assert!((x[0] - T::one()).abs().to_f64() <= 4.0 * T::EPS);
    }
    complex::<a::math::complex::Complex<f32>>();
    complex::<a::math::complex::Complex<f64>>();
}

#[test]
fn the_diagonal_of_r_has_the_signs_the_reflections_leave() {
    fn check<T: Num>() {
        let near = |x: T, y: f64| (x - T::real(y)).abs().to_f64() <= 8.0 * T::EPS * y.abs();
        // A reflection is made where a second row reaches the column: −sign(Re α)·‖(α, x)‖.
        let cases: [(f64, f64, f64); 3] = [(3.0, 4.0, -5.0), (-3.0, 4.0, 5.0), (1e-3, -2.0, -2.0)];
        for (top, below, want) in cases {
            let a = SparseMatrix::from_coo(2, 1, &[(0, 0, T::real(top)), (1, 0, T::real(below))]);
            let qr = SparseQr::new(&a);
            let norm = f64::hypot(top, below);
            assert!(
                near(qr.r()[(0, 0)], want.signum() * norm),
                "[{top}, {below}]"
            );
        }
        // A row that stores nothing in the column doesn't reach it: [0, 2] has one row there,
        // nothing to reflect, and R = 2 where a dense decomposition gives −2.
        let a = SparseMatrix::from_coo(2, 1, &[(1, 0, T::real(2.0))]);
        assert_eq!(SparseQr::new(&a).r()[(0, 0)], T::real(2.0));
        // A diagonal matrix needs no reflection at all and is its own R, signs and all.
        let d = [2.0, -3.0, 5.0, -0.5];
        let a = SparseMatrix::from_diagonal(&d.map(T::real));
        let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
        assert_eq!(dense(&qr.r()), dense(&a));
        assert_eq!(qr.q(), Matrix::identity(4));
        assert_eq!(qr.nnz(), 2 * d.len());
    }
    each_type!(check);
}

// What the factors store.

#[test]
fn nnz_counts_the_pattern_of_r_and_the_householder_vectors() {
    fn check<T: Num>() {
        // A full m × n matrix: R is a full triangle, and the vector of column t has m − t elements.
        for (m, n) in [(1, 1), (4, 2), (6, 6), (9, 4)] {
            let a = SparseMatrix::from_fn(m, n, |i, j| T::real(1.0 + (i * (j + 2) % 7) as f64));
            let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
            let vectors: usize = (0..n).map(|t| m - t).sum();
            assert_eq!(qr.nnz(), n * (n + 1) / 2 + vectors, "{m} × {n}");
        }
        // An arrow, a full first column and a diagonal: eliminated first, the full column fills
        // R and makes vectors of 7, 6, …, 1 elements; eliminated last, it fills nothing.
        let n = 7;
        let a = SparseMatrix::from_fn(n, n, |i, j| {
            if j == 0 || i == j {
                T::real(1.0 + i as f64)
            } else {
                T::zero()
            }
        });
        let first = SparseQr::with_ordering(&a, &Ordering::Natural);
        assert_eq!(first.r().nnz(), n * (n + 1) / 2);
        assert_eq!(first.nnz(), n * (n + 1) / 2 + n * (n + 1) / 2);
        let mut last: Vec<usize> = (1..n).collect();
        last.push(0);
        let last = SparseQr::with_ordering(&a, &Ordering::Given(last));
        assert_eq!(last.r().nnz(), 2 * n - 1);
        // Row 0 stores nothing but in the full column, so every front holds one row and every
        // vector one element: the rows are already an upper triangle, reordered.
        assert_eq!(last.nnz(), (2 * n - 1) + n);
        assert_decomposition(&a, &last, 16.0);
    }
    each_type!(check);
}

#[test]
fn a_long_banded_problem_has_factors_of_linear_size() {
    // The smoothing problem of 2·10⁴ points: 4·10⁴ rows and three elements per column. A method
    // that cost or stored of the order of m·n would not finish here; a sparse one stores two
    // elements of R per column and five of the vectors.
    fn check<T: Num>() {
        let n = size(20_000, 200);
        let (a, b) = smoothing::<T>(n);
        let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
        assert_eq!(qr.r().nnz(), 2 * n - 1);
        assert!(qr.nnz() <= 7 * n, "the factors store {} elements", qr.nnz());
        let x = qr.solve(&b).unwrap();
        let normal = &a.transpose() * &(&a * &x - &b);
        assert!(normal.norm(Norm::Max).to_f64() <= 64.0 * T::EPS);
    }
    check::<f32>();
    check::<f64>();
}

// Malformed calls.

#[test]
#[should_panic(expected = "the matrix has fewer rows than columns")]
fn a_matrix_with_fewer_rows_than_columns_is_refused() {
    let a = SparseMatrix::from_coo(2, 3, &[(0, 0, 1.0f64), (1, 2, 1.0)]);
    let _ = SparseQr::new(&a);
}

#[test]
#[should_panic(expected = "the right-hand side has the wrong number of rows")]
fn a_right_hand_side_of_the_wrong_height_is_refused() {
    let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0)]);
    let _ = SparseQr::new(&a).solve(&Matrix::zeros(2, 1));
}

#[test]
#[should_panic(expected = "the right-hand side has the wrong length")]
fn a_right_hand_side_of_the_wrong_length_is_refused() {
    let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0)]);
    let _ = SparseQr::new(&a).solve_vec(&[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "the matrix hasn't a row for each column of Q")]
fn a_product_with_q_of_the_wrong_height_is_refused() {
    let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0)]);
    let _ = SparseQr::new(&a).apply_q(&Matrix::zeros(3, 1));
}

#[test]
#[should_panic(expected = "the matrix hasn't a row for each row of Q")]
fn a_product_with_qh_of_the_wrong_height_is_refused() {
    let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0)]);
    let _ = SparseQr::new(&a).apply_qh(&Matrix::zeros(2, 1));
}

#[test]
#[should_panic(expected = "the permutation isn't of the right length")]
fn a_given_ordering_of_the_wrong_length_is_refused() {
    let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0)]);
    let _ = SparseQr::with_ordering(&a, &Ordering::Given(vec![0, 1, 2]));
}

#[test]
#[should_panic(expected = "the permutation repeats an index")]
fn a_given_ordering_that_repeats_a_column_is_refused() {
    let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0)]);
    let _ = SparseQr::with_ordering(&a, &Ordering::Given(vec![1, 1]));
}

#[test]
#[should_panic(expected = "the permutation is out of range")]
fn a_given_ordering_outside_the_columns_is_refused() {
    let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0)]);
    let _ = SparseQr::with_ordering(&a, &Ordering::Given(vec![0, 2]));
}

#[test]
fn the_debug_form_shows_the_shape_and_the_sizes() {
    let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0)]);
    let qr = SparseQr::new(&a);
    let text = format!("{qr:?}");
    assert!(text.starts_with("SparseQr"), "{text}");
    assert!(text.contains("rows: 3") && text.contains("cols: 2") && text.contains("rank: 2"));
    // A clone is the same decomposition.
    let b = [1.0, 2.0, 3.0];
    assert_eq!(qr.clone().solve_vec(&b), qr.solve_vec(&b));
}

// The measurements of the documentation.

/// The worst distance of a solution of an embedded case from NumPy's under the orderings of
/// [`check_embedded`], in units of the rounding of the type times the condition number of the
/// least squares.
fn worst_embedded<T: Num>(a: &SparseMatrix<T>, b: &Matrix<T>, want: &Parts, kappa: f64) -> f64 {
    let mut r = Rng(0x5eed_0006);
    let mut worst = 0.0f64;
    for ordering in [
        Ordering::Colamd,
        Ordering::Natural,
        Ordering::Amd,
        Ordering::Given(random_permutation(&mut r, a.cols())),
    ] {
        let x = SparseQr::with_ordering(a, &ordering).solve(b).unwrap();
        worst = worst.max(relative_error(x.as_slice(), want) / (kappa * T::EPS));
    }
    worst
}

/// The worst distances of the embedded cases from NumPy's solutions, as a line of a table: the
/// real cases for every type, the complex ones for the complex types.
fn embedded_line<T: Num>(name: &str) -> String {
    let real: [(Case, &[f64], f64); 3] = [
        (SQUARE, &SQUARE_X, SQUARE_KAPPA),
        (TALL, &TALL_X, TALL_KAPPA),
        (VERY_TALL, &VERY_TALL_X, VERY_TALL_KAPPA),
    ];
    let mut cells = Vec::new();
    for (case, x, kappa) in real {
        let worst = worst_embedded(&case.matrix::<T>(), &case.b(), &pairs(x), kappa);
        cells.push(format!("{worst:5.2}"));
    }
    let (a, b) = smoothing::<T>(40);
    let worst = worst_embedded(&a, &b, &pairs(&SMOOTHING_X), SMOOTHING_KAPPA);
    cells.push(format!("{worst:5.2}"));
    if T::IS_COMPLEX {
        let complex: [(Case, &Parts, f64); 3] = [
            (COMPLEX_SQUARE, &COMPLEX_SQUARE_X, COMPLEX_SQUARE_KAPPA),
            (COMPLEX_TALL, &COMPLEX_TALL_X, COMPLEX_TALL_KAPPA),
            (
                COMPLEX_VERY_TALL,
                &COMPLEX_VERY_TALL_X,
                COMPLEX_VERY_TALL_KAPPA,
            ),
        ];
        for (case, x, kappa) in complex {
            let worst = worst_embedded(&case.matrix::<T>(), &case.b(), x, kappa);
            cells.push(format!("{worst:5.2}"));
        }
    }
    format!("| {name} | {} |", cells.join(" | "))
}

/// The smoothing of a `k × k` grid, `[I; 2·Dx; 2·Dy]` of `k²` columns and about `3·k²` rows, the
/// differences along the rows of the grid and along its columns, against data as in
/// [`smoothing`].
fn grid<T: Num>(k: usize) -> (SparseMatrix<T>, Matrix<T>) {
    let n = k * k;
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, T::real(1.0)));
    }
    let mut row = n;
    for i in 0..k {
        for j in 0..k {
            let here = i * k + j;
            for (go, there) in [(j + 1 < k, here + 1), (i + 1 < k, here + k)] {
                if go {
                    entries.push((row, here, T::real(-2.0)));
                    entries.push((row, there, T::real(2.0)));
                    row += 1;
                }
            }
        }
    }
    let b = Matrix::from_fn(row, 1, |i, _| {
        if i < n {
            T::real(((37 * i) % 9) as f64 / 4.0 - 1.0)
        } else {
            T::real(0.0)
        }
    });
    (SparseMatrix::from_coo(row, n, &entries), b)
}

/// The nested dissection order of the `k × k` grid: the middle line across the longer side
/// after the two halves it separates, each of them so ordered, down to blocks of 16 points.
fn grid_dissection(k: usize) -> Vec<usize> {
    fn order(rows: (usize, usize), cols: (usize, usize), k: usize, out: &mut Vec<usize>) {
        let ((r0, r1), (c0, c1)) = (rows, cols);
        let (h, w) = (r1 - r0, c1 - c0);
        if h == 0 || w == 0 {
            return;
        }
        if h * w <= 16 {
            out.extend((r0..r1).flat_map(|i| (c0..c1).map(move |j| i * k + j)));
        } else if w >= h {
            let mid = c0 + w / 2;
            order(rows, (c0, mid), k, out);
            order(rows, (mid + 1, c1), k, out);
            out.extend((r0..r1).map(|i| i * k + mid));
        } else {
            let mid = r0 + h / 2;
            order((r0, mid), cols, k, out);
            order((mid + 1, r1), cols, k, out);
            out.extend((c0..c1).map(|j| mid * k + j));
        }
    }
    let mut out = Vec::with_capacity(k * k);
    order((0, k), (0, k), k, &mut out);
    out
}

/// A very tall banded problem: `20·n` rows of three random elements each, in a window of 10
/// columns that moves down the matrix with the rows.
fn tall_band<T: Num>(n: usize) -> (SparseMatrix<T>, Matrix<T>) {
    let m = 20 * n;
    let mut r = Rng(0x7a11_ba4d ^ n as u64);
    let mut entries = Vec::new();
    for i in 0..m {
        for _ in 0..3 {
            let j = (i * n / m + r.below(10)).min(n - 1);
            entries.push((i, j, T::random(&mut r)));
        }
    }
    let b = random_dense(&mut r, m, 1);
    (SparseMatrix::from_coo(m, n, &entries), b)
}

/// The decomposition and the solution of `a`, timed as the least of three runs, as a line of a
/// table; the solution is checked by its normal equations.
fn scaling_line<T: Num>(
    a: &SparseMatrix<T>,
    b: &Matrix<T>,
    ordering: &Ordering,
) -> (usize, String) {
    let time = |f: &mut dyn FnMut()| {
        (0..3)
            .map(|_| {
                let start = std::time::Instant::now();
                f();
                start.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let mut qr = SparseQr::with_ordering(a, ordering);
    let factor = time(&mut || qr = SparseQr::with_ordering(a, ordering));
    let mut x = qr.solve(b).unwrap();
    let solve = time(&mut || x = qr.solve(b).unwrap());
    let normal = &a.conjugate_transpose() * &(a * &x - b);
    // The Frobenius norm from the stored elements: the dense form of the largest of these
    // matrices would take 27 GB.
    let squares: f64 = a.values().iter().map(|&x| magnitude(x).powi(2)).sum();
    let size_a = squares.sqrt();
    assert!(norm(&normal) <= 64.0 * T::EPS * size_a * (size_a * norm(&x) + norm(b)));
    let line = format!(
        "| {} | {} | {} | {} | {} | {:.4} | {:.5} |",
        a.rows(),
        a.cols(),
        a.nnz(),
        qr.r().nnz(),
        qr.nnz(),
        factor,
        solve
    );
    (qr.nnz(), line)
}

/// How often [`SparseQr::rank`] counts the rank of a matrix with clear dependencies, as lines
/// of a table, one for each shape from square to twenty times as many rows as columns: the
/// matrices counted right under a random order, those counted right under the order that puts
/// the dependent columns last, and the most a count fell short. A count above the rank, a
/// miss when the dependent columns come last, or a solution, fails the test.
fn rank_census<T: Num>(name: &str, trials: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for ratio in [1, 2, 5, 20] {
        let mut r = Rng(0xabcd_0000 + ratio as u64);
        let (mut random_right, mut short) = (0, 0);
        for trial in 0..trials {
            let n = 3 + r.below(25);
            let m = ratio * n;
            let mut d = dense(&well_conditioned::<T>(&mut r, m, n, 2));
            let place = random_permutation(&mut r, n);
            let dependent = 1 + r.below(n / 3 + 1);
            let independent = &place[dependent..];
            for &to in &place[..dependent] {
                let from = independent[r.below(independent.len())];
                let other = independent[r.below(independent.len())];
                make_dependent(&mut r, &mut d, trial % 7, to, from, other);
            }
            let a = SparseMatrix::from_matrix(&d);
            let rank = n - dependent;
            let random = Ordering::Given(random_permutation(&mut r, n));
            let qr = SparseQr::with_ordering(&a, &random);
            assert!(qr.rank() <= rank, "{m} × {n}: {} above {rank}", qr.rank());
            assert!(qr.solve_vec(&vec![T::one(); m]).is_none());
            if qr.rank() == rank {
                random_right += 1;
            }
            short = short.max(rank - qr.rank());
            let mut last = independent.to_vec();
            last.extend_from_slice(&place[..dependent]);
            let qr = SparseQr::with_ordering(&a, &Ordering::Given(last));
            assert_eq!(qr.rank(), rank, "{m} × {n} with the dependent columns last");
        }
        lines.push(format!(
            "| {name} | {ratio}·n | {random_right} of {trials} | {trials} of {trials} | {short} |"
        ));
    }
    lines
}

/// The bands of the condition number, in units of the limit of `solve`, of [`refusal_line`].
const BANDS: [f64; 9] = [0.0, 0.5, 0.9, 1.0, 1.1, 2.0, 4.0, 100.0, f64::INFINITY];

/// How sharply `solve` refuses around its limit, as a line of a table: of the matrices of
/// [`near_singular`], in each band of the condition number, how many were refused. A refusal
/// below the limit, or a solution beyond twice it, fails the test.
fn refusal_line<T: Num>(name: &str, trials: usize) -> String {
    let mut r = Rng(0x0b0d_0000 ^ T::EPS.to_bits());
    let mut counts = [[0usize; 2]; BANDS.len() - 1];
    for trial in 0..trials {
        let unit = 1.0 / (32.0 * T::EPS);
        let target = unit * f64::powf(100.0, 2.0 * r.unit());
        let a = near_singular::<T>(&mut r, trial % 2, target);
        let (m, n) = (a.rows(), a.cols());
        let ratio = condition(&a) * m.max(32) as f64 * T::EPS;
        let band = BANDS
            .windows(2)
            .position(|w| ratio < w[1])
            .unwrap_or(BANDS.len() - 2);
        let ordering = Ordering::Given(random_permutation(&mut r, n));
        let qr = SparseQr::with_ordering(&a, &ordering);
        let refused = qr.solve_vec(&vec![T::one(); m]).is_none();
        assert!(
            !refused || ratio >= 1.0,
            "{m} × {n}: refused at {ratio:.3} limits"
        );
        assert!(
            refused || ratio <= 2.0,
            "{m} × {n}: solved at {ratio:.3} limits"
        );
        counts[band][usize::from(refused)] += 1;
    }
    let cells: Vec<String> = counts
        .iter()
        .map(|[kept, refused]| format!("{refused} of {}", kept + refused))
        .collect();
    format!("| {name} | {} |", cells.join(" | "))
}

/// How many of `trials` matrices of [`large_combination`], each in four orders, `solve`
/// refused, and the most the count of the rank was above the rank.
fn combination_line<T: Num>(name: &str, trials: usize) -> String {
    let mut r = Rng(0xc0b1_0000 ^ T::EPS.to_bits());
    let (mut refused, mut total, mut above) = (0, 0, 0);
    for _ in 0..trials {
        let (a, _) = large_combination::<T>(&mut r);
        let (m, n) = (a.rows(), a.cols());
        let rank = dense(&a).rank();
        for ordering in four_orders(&mut r, n) {
            let qr = SparseQr::with_ordering(&a, &ordering);
            total += 1;
            if qr.solve_vec(&vec![T::one(); m]).is_none() {
                refused += 1;
            }
            above = above.max(qr.rank().saturating_sub(rank));
        }
    }
    assert_eq!(
        refused, total,
        "{name}: a matrix of exactly dependent columns was solved"
    );
    format!("| {name} | {refused} of {total} | {above} |")
}

#[test]
#[ignore = "seconds in a release build, minutes in a debug one: run it with `cargo test \
            --release --test math -- --ignored --nocapture sparse::qr::`"]
fn the_measurements_of_the_documentation_hold() {
    // The worst distances from NumPy's solutions, in units of κ·ε, of `lib/docs/math/sparse.md`.
    println!(
        "| type | square | tall | very tall | smoothing | c. square | c. tall | c. very tall |"
    );
    println!("{}", embedded_line::<f32>("f32"));
    println!("{}", embedded_line::<f64>("f64"));
    println!(
        "{}",
        embedded_line::<a::math::complex::Complex<f32>>("Complex<f32>")
    );
    println!(
        "{}",
        embedded_line::<a::math::complex::Complex<f64>>("Complex<f64>")
    );
    // Three sizes of about 10⁴, 3·10⁴ and 10⁵ rows of three problems. What the factors store is
    // that of a sparse method, which these bounds check: of the order of m for the curve and for
    // the very tall band, of n·log n for the grid in its nested dissection order. The times are
    // for their growth alone.
    println!("| m | n | nnz(A) | nnz(R) | nnz() | factor s | solve s |");
    for n in [5_000, 15_000, 50_000] {
        let (a, b) = smoothing::<f64>(n);
        let (stored, line) = scaling_line(&a, &b, &Ordering::Natural);
        assert_eq!(stored, 7 * n - 7);
        println!("{line} curve");
    }
    for n in [500, 1_500, 5_000] {
        let (a, b) = tall_band::<f64>(n);
        let (stored, line) = scaling_line(&a, &b, &Ordering::Natural);
        assert!(stored <= 11 * a.rows(), "{stored} for {} rows", a.rows());
        println!("{line} very tall band");
    }
    for k in [58, 100, 183] {
        let (a, b) = grid::<f64>(k);
        let (stored, line) = scaling_line(&a, &b, &Ordering::Given(grid_dissection(k)));
        let n = (k * k) as f64;
        assert!(
            (stored as f64) <= 10.0 * n * n.log2(),
            "{stored} for {k} × {k}"
        );
        println!("{line} grid, nested dissection");
    }
    // The same grid in its natural order, a band of k: the factors are those of the band, of
    // the order of n·k elements, and the work of n·k².
    for k in [58, 100, 183] {
        let (a, b) = grid::<f64>(k);
        let (stored, line) = scaling_line(&a, &b, &Ordering::Natural);
        assert!(stored <= 8 * k * k * k, "{stored} for {k} × {k}");
        println!("{line} grid, natural order");
    }
    // How often the rank is counted right, 500 matrices of each type and shape.
    println!("| type | m | right, random order | right, dependent last | most short |");
    let trials = size(500, 4);
    let mut lines = rank_census::<f32>("f32", trials);
    lines.extend(rank_census::<f64>("f64", trials));
    lines.extend(rank_census::<a::math::complex::Complex<f32>>(
        "Complex<f32>",
        trials,
    ));
    lines.extend(rank_census::<a::math::complex::Complex<f64>>(
        "Complex<f64>",
        trials,
    ));
    for line in lines {
        println!("{line}");
    }
    // How sharply singular matrices are refused: 2000 matrices of each type near the limit,
    // and 3000 of exactly dependent columns with large coefficients.
    let trials = size(2000, 8);
    println!(
        "| type | < 0.5 | 0.5 to 0.9 | 0.9 to 1 | 1 to 1.1 | 1.1 to 2 | 2 to 4 | 4 to 100 | > 100 |"
    );
    println!("{}", refusal_line::<f32>("f32", trials));
    println!("{}", refusal_line::<f64>("f64", trials));
    println!(
        "{}",
        refusal_line::<a::math::complex::Complex<f32>>("Complex<f32>", trials)
    );
    println!(
        "{}",
        refusal_line::<a::math::complex::Complex<f64>>("Complex<f64>", trials)
    );
    let trials = size(3000, 8);
    println!("| type | refused | most counted above the rank |");
    println!("{}", combination_line::<f32>("f32", trials));
    println!("{}", combination_line::<f64>("f64", trials));
    println!(
        "{}",
        combination_line::<a::math::complex::Complex<f32>>("Complex<f32>", trials)
    );
    println!(
        "{}",
        combination_line::<a::math::complex::Complex<f64>>("Complex<f64>", trials)
    );
}

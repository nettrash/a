//! The Hermite and Smith normal forms of matrices of numbers: the conditions of each form and of
//! its transformations on every output, classical examples, every shape and rank, and exact
//! comparisons with FLINT 3.6 (`fmpz_mat_hnf`, `fmpz_mat_snf`) and PARI/GP 2.19 (`mathnf`,
//! `matsnf`) on seeded random matrices, from `data/gen_normal.py`.

use super::common::{Rng, size};
use a::math::matrix::Matrix;
use a::math::numbers::{ByteOrder, Number};

// Checks of the forms

/// The integer `x`.
fn num(x: i64) -> Number {
    Number::from(x)
}

/// The matrix of the rows of small integers `rows`.
fn ints<const N: usize>(rows: &[[i64; N]]) -> Matrix<Number> {
    Matrix::from_rows(rows).map(|&x| num(x))
}

/// Whether `h` is in Hermite normal form: its nonzero rows first, the first nonzero element of
/// each, the pivot, positive and in a column after the one above, and the elements above a pivot
/// in `[0, pivot)`; with the number of its nonzero rows.
fn hermite_rank(h: &Matrix<Number>) -> Option<usize> {
    let (m, n) = h.shape();
    let mut last: Option<usize> = None;
    let mut rank = 0;
    for i in 0..m {
        let Some(p) = (0..n).find(|&j| !h[(i, j)].is_zero()) else {
            // A zero row: every row after it must be zero too.
            if (i..m).all(|k| (0..n).all(|j| h[(k, j)].is_zero())) {
                return Some(rank);
            }
            return None;
        };
        if last.is_some_and(|l| p <= l) || !h[(i, p)].is_positive() {
            return None;
        }
        for k in 0..i {
            let x = &h[(k, p)];
            if x.is_negative() || x >= &h[(i, p)] {
                return None;
            }
        }
        last = Some(p);
        rank += 1;
    }
    Some(rank)
}

/// Whether `s` is in Smith normal form: diagonal, nonnegative, each diagonal element dividing
/// the next, the zeros last.
fn is_smith(s: &Matrix<Number>) -> bool {
    let (m, n) = s.shape();
    for i in 0..m {
        for j in 0..n {
            if i != j && !s[(i, j)].is_zero() {
                return false;
            }
        }
    }
    let d = s.diagonal();
    d.iter().all(|x| !x.is_negative())
        && d.windows(2).all(|w| {
            if w[0].is_zero() {
                w[1].is_zero()
            } else {
                w[1].is_multiple_of(&w[0])
            }
        })
}

/// Whether `u` is unimodular: square, of determinant ±1.
fn is_unimodular(u: &Matrix<Number>) -> bool {
    u.is_square() && u.det().abs() == 1u8
}

/// The Hermite normal form of `a` with every condition checked: `U·a = H`, `U` unimodular, `H`
/// in the form with as many nonzero rows as the rank, and the form without the transformation
/// the same.
fn hermite(a: &Matrix<Number>) -> (Matrix<Number>, Matrix<Number>) {
    let (m, n) = a.shape();
    let (h, u) = a.hermite_normal_form();
    assert_eq!(h.shape(), (m, n));
    assert_eq!(u.shape(), (m, m));
    assert_eq!(&u * a, h, "U·A = H for {a:?}");
    assert!(is_unimodular(&u), "U unimodular for {a:?}");
    assert_eq!(
        hermite_rank(&h),
        Some(a.rank()),
        "H in Hermite form for {a:?}: {h:?}"
    );
    assert_eq!(a.hermite_form(), h);
    (h, u)
}

/// The Smith normal form of `a` with every condition checked: `U·a·V = S`, `U` and `V`
/// unimodular, `S` in the form with as many nonzero elements as the rank, and the elementary
/// divisors its diagonal.
fn smith(a: &Matrix<Number>) -> (Matrix<Number>, Matrix<Number>, Matrix<Number>) {
    let (m, n) = a.shape();
    let (s, u, v) = a.smith_normal_form();
    assert_eq!(s.shape(), (m, n));
    assert_eq!((u.shape(), v.shape()), ((m, m), (n, n)));
    assert_eq!(&(&u * a) * &v, s, "U·A·V = S for {a:?}");
    assert!(
        is_unimodular(&u) && is_unimodular(&v),
        "U, V unimodular for {a:?}"
    );
    assert!(is_smith(&s), "S in Smith form for {a:?}: {s:?}");
    let d = s.diagonal();
    assert_eq!(d.iter().filter(|x| !x.is_zero()).count(), a.rank());
    assert_eq!(a.elementary_divisors(), d);
    (s, u, v)
}

// Shapes and edge cases

#[test]
fn empty_matrices_have_empty_forms_and_identity_transformations() {
    for (m, n) in [(0, 0), (0, 3), (3, 0), (1, 0), (0, 1)] {
        let a = Matrix::<Number>::zeros(m, n);
        let (h, u) = hermite(&a);
        assert_eq!(h, a);
        assert_eq!(u, Matrix::identity(m));
        let (s, u, v) = smith(&a);
        assert_eq!(s, a);
        assert_eq!((u, v), (Matrix::identity(m), Matrix::identity(n)));
        assert_eq!(a.elementary_divisors(), vec![Number::default(); m.min(n)]);
    }
}

#[test]
fn a_zero_matrix_is_its_own_form_with_identity_transformations() {
    for (m, n) in [(1, 1), (2, 3), (3, 2), (4, 4)] {
        let a = Matrix::<Number>::zeros(m, n);
        let (h, u) = hermite(&a);
        assert_eq!((h, u), (a.clone(), Matrix::identity(m)));
        let (s, _, _) = smith(&a);
        assert_eq!(s, a);
    }
}

#[test]
fn one_by_one_matrices_take_the_absolute_value() {
    for x in [-7, -1, 0, 1, 12] {
        let a = ints(&[[x]]);
        let (h, u) = hermite(&a);
        assert_eq!(h, ints(&[[x.abs()]]));
        assert_eq!(u, ints(&[[if x < 0 { -1 } else { 1 }]]));
        let (s, _, _) = smith(&a);
        assert_eq!(s, ints(&[[x.abs()]]));
    }
}

#[test]
fn single_rows_and_columns_reduce_to_their_gcd() {
    // A row is its own lattice: its form is itself with its first nonzero element positive.
    let a = ints(&[[0, -4, 6, 10]]);
    assert_eq!(hermite(&a).0, ints(&[[0, 4, -6, -10]]));
    assert_eq!(smith(&a).0, ints(&[[2, 0, 0, 0]]));
    // A column: the gcd on top.
    let a = ints(&[[6], [-10], [15], [0]]);
    assert_eq!(hermite(&a).0, ints(&[[1], [0], [0], [0]]));
    assert_eq!(a.elementary_divisors(), [num(1)]);
}

#[test]
fn the_form_of_a_classical_example_matches_pari() {
    // `mathnf([2,3,6,2;5,6,1,6;8,3,1,1]~)~` in PARI/GP, and `matsnf`.
    let a = ints(&[[2, 3, 6, 2], [5, 6, 1, 6], [8, 3, 1, 1]]);
    let (h, _) = hermite(&a);
    assert_eq!(h, ints(&[[1, 0, 50, -11], [0, 3, 28, -2], [0, 0, 61, -13]]));
    let (s, _, _) = smith(&a);
    assert_eq!(s.diagonal(), [1, 1, 1].map(num));
}

#[test]
fn the_smith_form_of_a_classical_example_is_2_6_12() {
    let a = ints(&[[2, 4, 4], [-6, 6, 12], [10, -4, -16]]);
    let (s, _, _) = smith(&a);
    assert_eq!(s.diagonal(), [2, 6, 12].map(num));
    let (h, _) = hermite(&a);
    assert_eq!(h, ints(&[[2, 4, 4], [0, 6, 0], [0, 0, 12]]));
}

#[test]
fn a_diagonal_matrix_needs_gcd_and_lcm_steps() {
    let a = Matrix::from_diagonal(&[4, 6, 10, 1, 0, 9].map(num));
    let (s, _, _) = smith(&a);
    assert_eq!(s.diagonal(), [1, 1, 2, 6, 180, 0].map(num));
    // The zero row goes to the bottom.
    let (h, _) = hermite(&a);
    let mut rows = [[0; 6]; 6];
    for (i, (j, x)) in [(0, 4), (1, 6), (2, 10), (3, 1), (5, 9)]
        .into_iter()
        .enumerate()
    {
        rows[i][j] = x;
    }
    assert_eq!(h, ints(&rows));
}

#[test]
fn dependent_rows_leave_zero_rows_and_a_kernel_in_u() {
    let a = ints(&[[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 4, 6]]);
    let (h, u) = hermite(&a);
    assert_eq!(hermite_rank(&h), Some(2));
    assert_eq!(h, ints(&[[1, 2, 3], [0, 3, 6], [0, 0, 0], [0, 0, 0]]));
    // The last two rows of U are a basis of the left kernel, in Hermite form.
    for i in 2..4 {
        let row = Matrix::from_fn(1, 4, |_, j| u[(i, j)].clone());
        assert!((&row * &a).as_slice().iter().all(Number::is_zero));
    }
    let kernel = Matrix::from_fn(2, 4, |i, j| u[(i + 2, j)].clone());
    assert_eq!(hermite_rank(&kernel), Some(2));
    let (s, u, v) = smith(&a);
    assert_eq!(s.diagonal(), [1, 3, 0].map(num));
    // The last column of V is in the kernel, and the last two rows of U in the left kernel.
    let col = Matrix::from_fn(3, 1, |i, _| v[(i, 2)].clone());
    assert!((&a * &col).as_slice().iter().all(Number::is_zero));
    let rows = Matrix::from_fn(2, 4, |i, j| u[(i + 2, j)].clone());
    assert!((&rows * &a).as_slice().iter().all(Number::is_zero));
}

#[test]
fn the_transformation_is_the_one_of_the_form_of_a_beside_the_identity() {
    // `fmpz_mat_hnf([A | I])` is `[H | U]`.
    for a in [
        ints(&[[1], [1]]),
        ints(&[[1, 2], [2, 4], [3, 7]]),
        ints(&[[2, 3, 5], [4, 6, 11], [6, 9, 16]]),
        ints(&[[0, 0], [3, 1], [6, 2], [1, 0]]),
    ] {
        let (h, u) = hermite(&a);
        let m = a.rows();
        let wide = Matrix::from_fn(m, a.cols() + m, |i, j| {
            if j < a.cols() {
                a[(i, j)].clone()
            } else {
                num((j - a.cols() == i) as i64)
            }
        });
        let (hw, _) = hermite(&wide);
        let left = Matrix::from_fn(m, a.cols(), |i, j| hw[(i, j)].clone());
        let right = Matrix::from_fn(m, m, |i, j| hw[(i, a.cols() + j)].clone());
        assert_eq!((left, right), (h, u), "for {a:?}");
    }
    let (_, u) = hermite(&ints(&[[1], [1]]));
    assert_eq!(u, ints(&[[0, 1], [1, -1]]));
}

#[test]
fn wide_and_tall_matrices_of_every_rank() {
    let mut r = Rng(0x5EED_0001);
    let shapes = [(1, 5), (2, 7), (5, 1), (7, 2), (3, 6), (6, 3), (4, 4)];
    for (m, n) in shapes.into_iter().take(size(7, 2)) {
        for rank in 0..=m.min(n) {
            let a = low_rank(&mut r, m, n, rank, 3);
            assert_eq!(a.rank(), rank);
            hermite(&a);
            smith(&a);
        }
    }
}

#[test]
fn big_endian_input_gives_big_endian_forms() {
    let a = Matrix::from_rows(&[[2, 3], [4, 9]])
        .map(|&x| Number::new_from_i32(x, ByteOrder::BigEndian));
    let (h, u) = a.hermite_normal_form();
    assert!(
        h.as_slice()
            .iter()
            .chain(u.as_slice())
            .all(|x| x.byte_order == ByteOrder::BigEndian)
    );
    let (s, u, v) = a.smith_normal_form();
    assert!(
        s.as_slice()
            .iter()
            .chain(u.as_slice())
            .chain(v.as_slice())
            .chain(&a.elementary_divisors())
            .chain(a.hermite_form().as_slice())
            .all(|x| x.byte_order == ByteOrder::BigEndian)
    );
}

// The proof of the rank profiles: primes that give the wrong ones

/// The largest primes below `2^62`, which the rank profiles are first computed modulo.
const PRIMES: [u64; 4] = [
    4_611_686_018_427_387_847,
    4_611_686_018_427_387_817,
    4_611_686_018_427_387_787,
    4_611_686_018_427_387_761,
];

/// The product of the first `k` of [`PRIMES`].
fn product_of_primes(k: usize) -> Number {
    PRIMES[..k].iter().map(|&p| Number::from(p)).product()
}

#[test]
fn a_column_that_vanishes_modulo_the_first_primes_is_still_a_pivot() {
    // Modulo each of the four primes the first column is zero and the pivot seems to be in the
    // second; the proof rejects them, and the fifth prime gives the truth.
    let p = product_of_primes(4);
    let a = Matrix::from_rows(&[[p.clone(), num(1)], [p.clone() * num(2), num(3)]]);
    let (h, u) = hermite(&a);
    assert_eq!(
        h,
        Matrix::from_rows(&[[p.clone(), num(0)], [num(0), num(1)]])
    );
    assert_eq!(u, ints(&[[3, -1], [-2, 1]]));
    let one = Matrix::from_rows(&[[p.clone(), num(1)]]);
    assert_eq!(hermite(&one).0, one);
    assert_eq!(smith(&a).0.diagonal(), [num(1), p]);
}

#[test]
fn a_rank_that_drops_modulo_the_first_primes_is_still_found() {
    // det = p₀·p₁: of rank 1 modulo the first two primes, 2 over the integers.
    let p = product_of_primes(2);
    let a = Matrix::from_rows(&[[num(1), num(1)], [num(1), num(1) + &p]]);
    assert_eq!(a.rank(), 2);
    let (h, _) = hermite(&a);
    assert_eq!(
        h,
        Matrix::from_rows(&[[num(1), num(1)], [num(0), p.clone()]])
    );
    assert_eq!(a.elementary_divisors(), [num(1), p.clone()]);
    // Tall and wide, the dependency modulo the primes in the rows and in the columns.
    let t = Matrix::from_rows(&[[num(1), num(1)], [num(1), num(1) + &p], [num(2), num(2)]]);
    assert_eq!(hermite(&t).0.diagonal(), [num(1), p.clone()]);
    assert_eq!(smith(&t.transpose()).0.diagonal(), [num(1), p]);
}

#[test]
fn a_row_profile_that_is_wrong_modulo_the_first_prime_is_corrected() {
    // Modulo p₀ the second row is zero, and the first seems the independent one from the bottom.
    let p = Number::from(PRIMES[0]);
    let a = Matrix::from_rows(&[[num(1), num(0)], [p.clone(), num(0)]]);
    let (h, u) = hermite(&a);
    assert_eq!(h, ints(&[[1, 0], [0, 0]]));
    // `[A | I]` has the pivots in its first and third columns: its form is
    // `[[1, 0, 1, 0], [0, 0, p, -1]]`.
    assert_eq!(u, Matrix::from_rows(&[[num(1), num(0)], [p, num(-1)]]));
}

// Random matrices

/// A random number of up to `bits` bits, of either sign.
fn random_number(r: &mut Rng, bits: usize) -> Number {
    let words = bits.div_ceil(64);
    let mut x = Number::from(0u8);
    for _ in 0..words {
        x = (x << 64usize) + Number::from(r.next());
    }
    let x = x >> (words * 64 - bits);
    if r.next() & 1 == 1 { -x } else { x }
}

/// A random `m × n` matrix of numbers of up to `bits` bits.
fn random(r: &mut Rng, m: usize, n: usize, bits: usize) -> Matrix<Number> {
    Matrix::from_fn(m, n, |_, _| random_number(r, bits))
}

/// A random `m × n` matrix of rank `rank`, a product of random factors of `bits` bits.
fn low_rank(r: &mut Rng, m: usize, n: usize, rank: usize, bits: usize) -> Matrix<Number> {
    loop {
        let a = &random(r, m, rank, bits) * &random(r, rank, n, bits);
        if a.rank() == rank {
            return a;
        }
    }
}

#[test]
fn random_matrices_of_many_shapes_satisfy_every_condition() {
    let mut r = Rng(0x5EED_0002);
    let cases = size(60, 4);
    for _ in 0..cases {
        let m = (r.next() % 6) as usize + 1;
        let n = (r.next() % 6) as usize + 1;
        let bits = [2, 4, 16, 70][(r.next() % 4) as usize];
        let rank = (r.next() % (m.min(n) as u64 + 1)) as usize;
        let a = if r.next() & 1 == 0 {
            random(&mut r, m, n, bits)
        } else {
            low_rank(&mut r, m, n, rank, bits)
        };
        hermite(&a);
        smith(&a);
    }
}

#[test]
fn the_nonzero_divisors_of_a_square_matrix_multiply_to_its_determinant() {
    let mut r = Rng(0x5EED_0003);
    for n in 1..=size(7, 3) {
        for bits in [3, 40] {
            let a = random(&mut r, n, n, bits);
            let det = a.det().abs();
            let d = a.elementary_divisors();
            if det.is_zero() {
                assert!(d.iter().any(Number::is_zero));
            } else {
                assert_eq!(d.iter().product::<Number>(), det);
                assert_eq!(hermite(&a).0.diagonal().iter().product::<Number>(), det);
            }
        }
    }
}

// Against FLINT and PARI

/// The forms of the matrices of `data/gen_normal.py` from FLINT 3.6 and PARI/GP 2.19.
const EXPECTED: &str = include_str!("data/normal_expected.txt");

/// A matrix of `data/gen_normal.py` and its expected forms.
struct Case {
    /// The recipe: kind, rows, columns, bits, rank, seed.
    kind: String,
    m: usize,
    n: usize,
    bits: usize,
    k: usize,
    seed: u64,
    /// The digest of the text of the Hermite form, and the text when it is short.
    h: (u64, Option<String>),
    /// The same of the transformation of the Hermite form of `[A | I]`.
    u: (u64, Option<String>),
    /// The elementary divisors.
    s: Vec<Number>,
}

/// The matrices of `data/normal_expected.txt`.
fn cases() -> Vec<Case> {
    parse(EXPECTED)
}

/// The matrices of a file that `data/gen_normal.py` wrote.
fn parse(expected: &str) -> Vec<Case> {
    let mut out: Vec<Case> = Vec::new();
    let digest = |rest: &str| {
        let (d, t) = rest.split_once(' ').unwrap();
        let d = u64::from_str_radix(d, 16).unwrap();
        (d, (t != "-").then(|| t.to_string()))
    };
    for line in expected.lines().filter(|l| !l.starts_with('#')) {
        let (tag, rest) = line.split_once(' ').unwrap();
        match tag {
            "case" => {
                let f: Vec<&str> = rest.split(' ').collect();
                out.push(Case {
                    kind: f[0].to_string(),
                    m: f[1].parse().unwrap(),
                    n: f[2].parse().unwrap(),
                    bits: f[3].parse().unwrap(),
                    k: f[4].parse().unwrap(),
                    seed: u64::from_str_radix(f[5].trim_start_matches("0x"), 16).unwrap(),
                    h: (0, None),
                    u: (0, None),
                    s: Vec::new(),
                });
            }
            "H" => out.last_mut().unwrap().h = digest(rest),
            "U" => out.last_mut().unwrap().u = digest(rest),
            "S" => {
                out.last_mut().unwrap().s = rest.split(',').map(|x| x.parse().unwrap()).collect();
            }
            _ => panic!("an unknown line {line}"),
        }
    }
    out
}

/// The rows of a random `m × n` matrix of numbers of up to `bits` bits, row after row.
fn random_rows(r: &mut Rng, m: usize, n: usize, bits: usize) -> Vec<Vec<Number>> {
    (0..m)
        .map(|_| (0..n).map(|_| random_number(r, bits)).collect())
        .collect()
}

/// The matrix of a recipe of `data/gen_normal.py`, which says what each kind is.
fn generate(kind: &str, m: usize, n: usize, bits: usize, k: usize, seed: u64) -> Matrix<Number> {
    let mut r = Rng(seed);
    let rows: Vec<Vec<Number>> = match kind {
        "random" => random_rows(&mut r, m, n, bits),
        "rank" => {
            let b = random_rows(&mut r, m, k, bits);
            let c = random_rows(&mut r, k, n, bits);
            (0..m)
                .map(|i| {
                    (0..n)
                        .map(|j| (0..k).map(|t| &b[i][t] * &c[t][j]).sum())
                        .collect()
                })
                .collect()
        }
        "small" => (0..m)
            .map(|_| {
                (0..n)
                    .map(|_| num((r.next() % (2 * bits as u64 + 1)) as i64 - bits as i64))
                    .collect()
            })
            .collect(),
        "sparse" => (0..m)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        if r.next().is_multiple_of(4) {
                            random_number(&mut r, bits)
                        } else {
                            num(0)
                        }
                    })
                    .collect()
            })
            .collect(),
        "smith" => {
            let mut a = vec![vec![num(0); n]; m];
            for (i, row) in a.iter_mut().enumerate().take(m.min(n)) {
                if r.next().is_multiple_of(5) {
                    continue;
                }
                let (e2, e3, e5) = (r.next() % 4, r.next() % 3, r.next() % 2);
                row[i] = num(2i64.pow(e2 as u32) * 3i64.pow(e3 as u32) * 5i64.pow(e5 as u32));
            }
            for _ in 0..3 * (m + n) {
                let rows = r.next() & 1 == 0;
                let size = if rows { m } else { n } as u64;
                let (i, j) = ((r.next() % size) as usize, (r.next() % size) as usize);
                let c = num((r.next() % 7) as i64 - 3);
                if i == j {
                    continue;
                }
                if rows {
                    let source = a[j].clone();
                    for (x, y) in a[i].iter_mut().zip(&source) {
                        *x += &c * y;
                    }
                } else {
                    for row in &mut a {
                        let y = &c * &row[j];
                        row[i] += y;
                    }
                }
            }
            a
        }
        _ => panic!("an unknown kind {kind}"),
    };
    Matrix::from_fn(m, n, |i, j| rows[i][j].clone())
}

/// The text of a matrix as `data/gen_normal.py` writes it: its rows joined by `;`, the elements
/// of a row by `,`.
fn text(a: &Matrix<Number>) -> String {
    (0..a.rows())
        .map(|i| {
            a.row(i)
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// FNV-1a, 64 bits, of the bytes of `s`.
fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

/// Whether the matrix `a` is the expected one, by its text or its digest.
fn matches(a: &Matrix<Number>, expected: &(u64, Option<String>)) -> bool {
    let t = text(a);
    expected.1.as_ref().is_none_or(|e| *e == t) && fnv(&t) == expected.0
}

/// The matrices of `data/normal_expected.txt` small enough for the run: all of them, but a few
/// under Miri.
fn reference_cases() -> impl Iterator<Item = (Case, Matrix<Number>)> {
    cases()
        .into_iter()
        .filter(|c| !cfg!(miri) || c.m * c.n <= 16)
        .map(|c| {
            let a = generate(&c.kind, c.m, c.n, c.bits, c.k, c.seed);
            (c, a)
        })
}

#[test]
fn the_generated_matrices_are_those_of_the_script() {
    // The first elements of two recipes, as `gen_normal.py` makes them.
    let a = generate("random", 2, 2, 8, 0, 0x4E4F_524D_0003);
    assert_eq!(a.hermite_form(), ints(&[[1, 4603], [0, 33851]]));
    assert_eq!(cases().len(), 46);
}

#[test]
fn hermite_forms_and_transformations_match_flint_and_pari() {
    for (c, a) in reference_cases() {
        let (h, u) = hermite(&a);
        assert!(
            matches(&h, &c.h),
            "H of {} {}x{} seed {:#x}",
            c.kind,
            c.m,
            c.n,
            c.seed
        );
        assert!(
            matches(&u, &c.u),
            "U of {} {}x{} seed {:#x}",
            c.kind,
            c.m,
            c.n,
            c.seed
        );
    }
}

#[test]
fn smith_forms_match_flint_and_pari() {
    for (c, a) in reference_cases() {
        let (s, _, _) = smith(&a);
        assert_eq!(
            s.diagonal(),
            c.s,
            "S of {} {}x{} seed {:#x}",
            c.kind,
            c.m,
            c.n,
            c.seed
        );
    }
}

// A sweep of 1500 random matrices of every kind and of up to 12 × 12 runs in a few minutes:
// `python3 data/gen_normal.py sweep 1500 23 <file>`, then
// `A_NORMAL_SWEEP=<file> cargo test --release --test math -- --ignored matrix::normal::sweep`.
#[test]
#[ignore = "a sweep of a file that data/gen_normal.py writes, named by A_NORMAL_SWEEP"]
fn sweep_against_flint_and_pari() {
    let path = std::env::var("A_NORMAL_SWEEP").expect("the file of the sweep in A_NORMAL_SWEEP");
    let cases = parse(&std::fs::read_to_string(path).unwrap());
    assert!(!cases.is_empty());
    let mut worst = (0f64, 0f64);
    for c in &cases {
        let a = generate(&c.kind, c.m, c.n, c.bits, c.k, c.seed);
        let (h, u) = hermite(&a);
        let what = format!(
            "{} {}x{} {} {} {:#x}",
            c.kind, c.m, c.n, c.bits, c.k, c.seed
        );
        assert!(matches(&h, &c.h), "H of {what}");
        assert!(matches(&u, &c.u), "U of {what}");
        let (s, su, sv) = smith(&a);
        assert_eq!(s.diagonal(), c.s, "S of {what}");
        // The lengths of the transformations against Hadamard's bound on the minors.
        let k = c.m.min(c.n);
        let bound = hadamard_bits(k, longest(&a).max(1));
        worst.0 = worst.0.max(longest(&su).max(longest(&sv)) as f64 / bound);
        worst.1 = worst.1.max(longest(&u) as f64 / bound);
    }
    println!(
        "{} matrices; the Smith transformations up to {:.2} times Hadamard's bound, the \
         Hermite one up to {:.2}",
        cases.len(),
        worst.0,
        worst.1
    );
}

// The cost of the transformations of tall and wide matrices

/// A `m × n` matrix of elements in `-1024..1024` from a linear congruential generator.
fn lcg_matrix(m: usize, n: usize, seed: u64) -> Matrix<Number> {
    let mut s = seed;
    Matrix::from_fn(m, n, |_, _| {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        num(((s >> 33) % 2048) as i64 - 1024)
    })
}

/// The columns of the pivots of the rows of `h`, in Hermite form, from the left.
fn pivot_columns(h: &Matrix<Number>) -> Vec<usize> {
    (0..h.rows())
        .filter_map(|i| (0..h.cols()).find(|&j| !h[(i, j)].is_zero()))
        .collect()
}

/// That `U` of `hermite_normal_form` is unimodular and the one of `[A | I]`, without its
/// determinant: `[H | U]` is in Hermite form with a pivot in every row, its pivots in the columns
/// `J` of `A` and `S` of `I`, and `U·[A[:, J] | E_S]` is the triangular matrix of the pivots, so
/// `|det U|` is their product over `|det A[R, J]|`, `R` the rows outside `S`; that must be 1.
fn assert_transformation_of_the_form(a: &Matrix<Number>, h: &Matrix<Number>, u: &Matrix<Number>) {
    let (m, n) = a.shape();
    assert_eq!(u * a, *h);
    let wide = Matrix::from_fn(m, n + m, |i, j| {
        if j < n {
            h[(i, j)].clone()
        } else {
            u[(i, j - n)].clone()
        }
    });
    assert_eq!(hermite_rank(&wide), Some(m), "[H | U] in Hermite form");
    let pivots = pivot_columns(&wide);
    let r = pivots.iter().filter(|&&c| c < n).count();
    let (j, s): (Vec<usize>, Vec<usize>) = (
        pivots[..r].to_vec(),
        pivots[r..].iter().map(|c| c - n).collect(),
    );
    let rows: Vec<usize> = (0..m).filter(|i| !s.contains(i)).collect();
    let minor = Matrix::from_fn(r, r, |x, y| a[(rows[x], j[y])].clone());
    let product: Number = (0..m).map(|i| wide[(i, pivots[i])].clone()).product();
    assert_eq!(product, minor.det().abs(), "|det U| = 1");
}

/// `a·b` by the schoolbook loop over every element of `a`, its zeros skipped: work in proportion
/// to the elements of `a` times the columns of `b`, quadratic in the long side for the shapes of
/// the cost tests, which time the transformations against it.
fn product_by_loop(a: &Matrix<Number>, b: &Matrix<Number>) -> Matrix<Number> {
    let (m, k) = a.shape();
    let n = b.cols();
    let mut out = Matrix::zeros(m, n);
    for i in 0..m {
        for t in 0..k {
            let x = &a[(i, t)];
            if x.is_zero() {
                continue;
            }
            for j in 0..n {
                let y = &b[(t, j)];
                if !y.is_zero() {
                    out[(i, j)] += x * y;
                }
            }
        }
    }
    out
}

/// The least time of `runs` runs of `f`, in seconds, and the result of the last: the least, so
/// that a runner busy with other work slows a measurement only if it is busy at every run.
fn fastest<T>(runs: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..runs {
        let t = std::time::Instant::now();
        let x = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(x);
    }
    (best, out.expect("a run"))
}

/// The largest ratio the cost tests accept of the time of a transformation to that of the
/// schoolbook product with it, measured in the same test on the same matrices, each the fastest
/// of a few runs. The ratios measured are in the tests: from 0.6 to 3.1 for the transformations
/// built from their structure, in debug and in release, and from 280 to 570 for the Hermite form
/// of the `m × m` matrix `[A[:, J] | E_S]` that they were before, whose work is cubic in `m`; 25
/// leaves a factor of 8 above the one and 11 below the other.
const COST_RATIO: f64 = 25.0;

/// Asserts that `what` took at most [`COST_RATIO`] times its reference, but under Miri, which
/// times nothing that matters.
#[track_caller]
fn assert_cost(what: &str, time: f64, reference: f64) {
    println!(
        "{what}: {time:.4} s against {reference:.4} s, ratio {:.2}",
        time / reference
    );
    assert!(
        cfg!(miri) || time <= COST_RATIO * reference,
        "{what}: {time:.3} s, {:.0} times its quadratic reference of {reference:.4} s",
        time / reference
    );
}

#[test]
fn the_transformation_of_a_tall_matrix_takes_time_linear_in_its_rows() {
    // 2000 × 3 and 1500 × 4 of rank 2, so that the form of the left kernel has rows of every
    // kind. The transformation is timed against the product `U·A`, the check, by the schoolbook
    // loop: measured ratios 1.8 and 1.6 in debug, 2.6 and 1.5 in release; the Hermite form of the
    // `m × m` matrix `[A[:, J] | E_S]`, the transformation before, gave 338 in debug and 353 in
    // release on the first, and failed there (59 s for it in debug, against 0.12 s now).
    let shapes = [(size(2000, 24), 3, 1), (size(1500, 16), 4, 2)];
    for (m, n, seed) in shapes {
        let a = if n == 4 {
            &lcg_matrix(m, 2, seed) * &lcg_matrix(2, n, seed + 1)
        } else {
            lcg_matrix(m, n, seed)
        };
        let (time, (h, u)) = fastest(2, || a.hermite_normal_form());
        let (reference, product) = fastest(3, || product_by_loop(&u, &a));
        assert_eq!(product, h);
        assert_cost(
            &format!("hermite_normal_form of {m} × {n}"),
            time,
            reference,
        );
        assert_eq!(h, a.hermite_form());
        assert_eq!(hermite_rank(&h), Some(a.rank()));
        assert_transformation_of_the_form(&a, &h, &u);
    }
}

#[test]
fn the_smith_transformations_of_long_rows_and_columns_take_time_quadratic() {
    // 1 × 2000, 3 × 1500 and 1500 × 3, each timed against the product `U·A·V`, the check, by the
    // schoolbook loop: measured ratios 1.7, 0.7 and 1.9 in debug, 3.1, 0.7 and 1.9 in release;
    // the transformations before, by the Hermite forms of `m × m` and `n × n` matrices, gave 565
    // in debug and 282 in release on the first, and failed there (56 s for it in debug, against
    // 0.1 s now).
    for (m, n, seed) in [
        (1, size(2000, 24), 4),
        (3, size(1500, 16), 5),
        (size(1500, 16), 3, 6),
    ] {
        let a = lcg_matrix(m, n, seed);
        let (time, (s, u, v)) = fastest(2, || a.smith_normal_form());
        let (reference, product) = fastest(3, || product_by_loop(&product_by_loop(&u, &a), &v));
        assert_eq!(product, s);
        assert_cost(&format!("smith_normal_form of {m} × {n}"), time, reference);
        assert!(is_smith(&s));
        assert_eq!(s.diagonal(), a.elementary_divisors());
        // The last rows of `U` and the last columns of `V`: the left kernel and the kernel.
        let r = a.rank();
        let ul = Matrix::from_fn(m - r, m, |i, j| u[(r + i, j)].clone());
        assert!((&ul * &a).as_slice().iter().all(Number::is_zero));
        let vk = Matrix::from_fn(n, n - r, |i, j| v[(i, r + j)].clone());
        assert!((&a * &vk).as_slice().iter().all(Number::is_zero));
    }
    // At a size where the determinants are cheap, the whole check.
    let (long, longer) = (size(250, 8), size(300, 8));
    for (m, n, seed) in [
        (1, longer, 7),
        (2, long, 8),
        (long, 2, 9),
        (size(200, 8), 4, 10),
    ] {
        smith(&lcg_matrix(m, n, seed));
    }
}

// The size of the Smith transformations

/// The length in bits of the longest element of `a`.
fn longest(a: &Matrix<Number>) -> u64 {
    a.as_slice().iter().map(Number::bits).max().unwrap_or(0)
}

/// Hadamard's bound on the length in bits of the minors of a `k × k` matrix of `b`-bit elements,
/// `k·(b + ½·log₂ k)`.
fn hadamard_bits(k: usize, b: u64) -> f64 {
    k as f64 * (b as f64 + 0.5 * (k as f64).log2())
}

#[test]
fn the_smith_transformations_of_a_diagonal_stay_near_hadamards_bound() {
    // The diagonal of the first 40 primes: the gcd and lcm steps one pair after another made
    // `V` 3859 bits long, 9 times the bound of 426; by a sorting network, with each step's larger
    // matrix on the shorter side, 451.
    let k = size(40, 6);
    let primes: Vec<Number> = (2u64..)
        .filter(|&x| (2..x).all(|d| !x.is_multiple_of(d)))
        .take(k)
        .map(Number::from)
        .collect();
    let a = Matrix::from_diagonal(&primes);
    let (s, u, v) = smith(&a);
    let product: Number = primes.iter().product();
    let mut d = vec![num(1); k - 1];
    d.push(product);
    assert_eq!(s.diagonal(), d);
    let bound = hadamard_bits(k, longest(&a));
    assert!((longest(&u).max(longest(&v)) as f64) < 1.2 * bound);
    // 40 random odd numbers of 100 bits: 76 000 bits before, 18.5 times the bound; now under 2.
    let mut r = Rng(0x5EED_0004);
    let a = Matrix::from_diagonal(
        &(0..size(40, 6))
            .map(|_| random_number(&mut r, 100).abs() | num(1))
            .collect::<Vec<_>>(),
    );
    let (_, u, v) = smith(&a);
    let bound = hadamard_bits(a.rows(), 100);
    assert!((longest(&u).max(longest(&v)) as f64) < 2.0 * bound);
}

// The lengths of the Smith transformations on structured matrices, against Hadamard's bound, as
// the documentation quotes them:
// `cargo test --release --test math -- --ignored --nocapture matrix::normal::lengths`.
#[test]
#[ignore = "a measurement for the documentation, of a few seconds in release"]
fn lengths_of_the_smith_transformations_of_structured_matrices() {
    let primes = |k: usize| -> Vec<Number> {
        (2u64..)
            .filter(|&x| (2..x).all(|d| !x.is_multiple_of(d)))
            .take(k)
            .map(Number::from)
            .collect()
    };
    let mut big = Vec::new();
    let mut p = Number::from(1u64 << 62);
    for _ in 0..100 {
        p = p.prev_prime().unwrap();
        big.push(p.clone());
    }
    let mut r = Rng(0x5EED_0005);
    let mut odd = |k: usize| -> Vec<Number> {
        (0..k)
            .map(|_| random_number(&mut r, 100).abs() | num(1))
            .collect()
    };
    let near = |d: Vec<Number>, r: &mut Rng| {
        let mut a = Matrix::from_diagonal(&d);
        for i in 0..d.len() - 1 {
            a[(i, i + 1)] = num((r.next() % 7) as i64 - 3);
        }
        a
    };
    let mut r2 = Rng(0x5EED_0006);
    let reversed: Vec<Number> = primes(100).into_iter().rev().collect();
    let cases: Vec<(&str, Matrix<Number>)> = vec![
        ("the first 40 primes", Matrix::from_diagonal(&primes(40))),
        ("the first 100 primes", Matrix::from_diagonal(&primes(100))),
        (
            "the first 100 primes, from the largest",
            Matrix::from_diagonal(&reversed),
        ),
        (
            "40 random odd numbers of 100 bits",
            Matrix::from_diagonal(&odd(40)),
        ),
        (
            "100 random odd numbers of 100 bits",
            Matrix::from_diagonal(&odd(100)),
        ),
        (
            "the 100 largest primes below 2^62",
            Matrix::from_diagonal(&big),
        ),
    ];
    let mut cases = cases;
    for _ in 0..5 {
        let a = near(odd(100), &mut r2);
        cases.push(("100 random odd numbers of 100 bits, small ones beside", a));
    }
    // The worst instance known, from another stream: the numbers without signs, the small ones
    // after them from the same generator.
    let mut r3 = Rng(0x1234_5678_9ABC_DEF1);
    let d: Vec<Number> = (0..100)
        .map(|_| {
            let x = (Number::from(r3.next()) << 64usize) + Number::from(r3.next());
            (x >> 28usize) | num(1)
        })
        .collect();
    cases.push(("the same, another stream", near(d, &mut r3)));
    // `U₀·diag(d)·V₀`, the usual way to build a matrix with a given Smith form: `d` random odd
    // numbers of 300 bits, `U₀` and `V₀` the identity through `2k` random additions of -2 to 2
    // times a row to another (of a column for `V₀`) and a permutation.
    let mut r4 = Rng(0x5EED_0007);
    let k = 50;
    let unimodular = |r: &mut Rng| {
        let mut x = Matrix::<Number>::identity(k);
        for _ in 0..2 * k {
            let (i, j) = (
                (r.next() % k as u64) as usize,
                (r.next() % k as u64) as usize,
            );
            let c = num((r.next() % 5) as i64 - 2);
            if i != j {
                x.add_scaled_row(i, j, &c);
            }
        }
        for i in (1..k).rev() {
            x.swap_rows(i, (r.next() % (i as u64 + 1)) as usize);
        }
        x
    };
    let (u0, v0) = (unimodular(&mut r4), unimodular(&mut r4));
    let d: Vec<Number> = (0..k)
        .map(|_| random_number(&mut r4, 300).abs() | num(1))
        .collect();
    let a = &(&u0 * &Matrix::from_diagonal(&d)) * &v0.transpose();
    cases.push(("U₀·diag(d)·V₀, 50 × 50, d of 300 bits", a));
    for (what, a) in cases {
        let (s, u, v) = a.smith_normal_form();
        assert_eq!(&(&u * &a) * &v, s);
        let b = longest(&a);
        let bound = hadamard_bits(a.rows(), b);
        let (lu, lv) = (longest(&u), longest(&v));
        println!(
            "{what}: U {lu} bits, V {lv} bits, bound {bound:.0}, ratio {:.2}",
            lu.max(lv) as f64 / bound
        );
    }
}

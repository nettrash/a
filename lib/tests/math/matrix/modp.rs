//! Matrices over `GF(p)`: against a plain elimination in 128-bit arithmetic written here, against
//! FLINT's `nmod_mat` and NTL's `mat_zz_p` / `mat_ZZ_p` (the residues, ranks, determinants,
//! inverses, echelon forms, null spaces and products of `data/modp_flint.txt`, which
//! `data/gen_modp.cpp` writes), and against the identities the results must satisfy.

use super::common::{Rng, size};
use a::math::matrix::{Matrix, ModMatrix};
use a::math::numbers::{ByteOrder, Number};

/// Primes of every size the arithmetic distinguishes: 2 and 3, small ones whose sums of
/// products stay in a word, 31 and 32 bits, 50, 61 and 62 bits, the largest prime below `2^63`
/// (Shoup's products in a word) and the largest below `2^64`.
const PRIMES: [u64; 13] = [
    2,
    3,
    5,
    251,
    65521,
    2_147_483_647,
    4_294_967_291,
    1_125_899_906_842_597,
    2_305_843_009_213_693_951,
    4_611_686_018_427_387_847,
    9_223_372_036_854_775_783,
    18_446_744_073_709_551_557,
    1_000_003,
];

/// `a·b mod p`.
fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    (a as u128 * b as u128 % p as u128) as u64
}

/// `a^e mod p`.
fn powmod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut x = 1 % p;
    while e > 0 {
        if e & 1 == 1 {
            x = mulmod(x, a, p);
        }
        a = mulmod(a, a, p);
        e >>= 1;
    }
    x
}

/// A random `m × n` matrix over `GF(p)` of rank at most `rank`, with about `density` of 8 of its
/// elements drawn and the rest zero, row-major: the product of random `m × rank` and `rank × n`
/// matrices when `rank` is below `min(m, n)`.
pub(super) fn random(
    r: &mut Rng,
    m: usize,
    n: usize,
    rank: usize,
    density: u64,
    p: u64,
) -> Vec<u64> {
    let draw = |r: &mut Rng| {
        let x = r.next() % p;
        if r.next() % 8 < density { x } else { 0 }
    };
    if rank >= m.min(n) {
        return (0..m * n).map(|_| draw(r)).collect();
    }
    let b: Vec<u64> = (0..m * rank).map(|_| draw(r)).collect();
    let c: Vec<u64> = (0..rank * n).map(|_| draw(r)).collect();
    naive_product(&b, &c, m, rank, n, p)
}

/// The classical product in 128-bit arithmetic, row-major.
fn naive_product(a: &[u64], b: &[u64], m: usize, k: usize, n: usize, p: u64) -> Vec<u64> {
    let mut c = vec![0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0u128;
            for t in 0..k {
                s = (s + a[i * k + t] as u128 * b[t * n + j] as u128) % p as u128;
            }
            c[i * n + j] = s as u64;
        }
    }
    c
}

/// A plain Gauss–Jordan elimination: the reduced row echelon form, its pivots, and the
/// determinant when the matrix is square.
fn naive_rref(a: &[u64], m: usize, n: usize, p: u64) -> (Vec<u64>, Vec<usize>, u64) {
    let mut a = a.to_vec();
    let mut det = 1 % p;
    let mut pivots = Vec::new();
    let mut r = 0;
    for j in 0..n {
        let Some(i) = (r..m).find(|&i| a[i * n + j] != 0) else {
            det = 0;
            continue;
        };
        if i != r {
            for c in 0..n {
                a.swap(i * n + c, r * n + c);
            }
            det = (p - det) % p;
        }
        let x = a[r * n + j];
        det = mulmod(det, x, p);
        let inv = powmod(x, p - 2, p);
        for c in 0..n {
            a[r * n + c] = mulmod(a[r * n + c], inv, p);
        }
        for i in 0..m {
            let l = a[i * n + j];
            if i != r && l != 0 {
                for c in 0..n {
                    let t = mulmod(l, a[r * n + c], p);
                    a[i * n + c] = ((a[i * n + c] as u128 + (p - t) as u128) % p as u128) as u64;
                }
            }
        }
        pivots.push(j);
        r += 1;
    }
    if m != n || r < n {
        det = 0;
    }
    (a, pivots, det)
}

/// The matrix of the row-major elements `a`.
fn matrix(a: &[u64], m: usize, n: usize, p: u64) -> ModMatrix {
    ModMatrix::from_fn(m, n, p, |i, j| a[i * n + j])
}

/// The shapes and ranks the comparisons run on.
fn shapes() -> Vec<(usize, usize, usize)> {
    let mut s = vec![
        (0, 0, 0),
        (0, 3, 0),
        (3, 0, 0),
        (1, 1, 1),
        (1, 1, 0),
        (1, 5, 1),
        (5, 1, 1),
        (2, 2, 2),
        (3, 3, 2),
        (4, 7, 4),
        (7, 4, 4),
        (6, 6, 3),
        (6, 6, 0),
        (9, 9, 9),
        (12, 17, 5),
        (17, 12, 12),
    ];
    if !cfg!(miri) {
        // Past the classical elimination's block of 16 columns, so that the recursion, its
        // triangular solves and its products run, with and without full rank.
        s.extend([
            (60, 60, 60),
            (60, 60, 41),
            (50, 97, 50),
            (97, 50, 30),
            (130, 130, 130),
            (130, 100, 77),
        ]);
    }
    s
}

#[test]
fn elimination_agrees_with_a_plain_gauss_jordan_elimination() {
    let mut r = Rng(0x0DD_5EED);
    for &p in &PRIMES {
        for (m, n, rank) in shapes() {
            for density in [8, 3] {
                let a = random(&mut r, m, n, rank, density, p);
                let mm = matrix(&a, m, n, p);
                let (rref, pivots, det) = naive_rref(&a, m, n, p);
                let what = format!("p = {p}, {m} × {n} of rank ≤ {rank}, density {density}");
                assert_eq!(mm.rank(), pivots.len(), "{what}");
                let (r2, piv2) = mm.rref();
                assert_eq!(piv2, pivots, "{what}");
                assert_eq!(r2.as_slice(), &rref[..], "{what}");
                if m == n {
                    assert_eq!(mm.det(), det, "{what}");
                    match mm.inverse() {
                        Some(inv) => {
                            assert!(det != 0, "{what}");
                            assert_eq!(&mm * &inv, ModMatrix::identity(n, p), "{what}");
                            assert_eq!(&inv * &mm, ModMatrix::identity(n, p), "{what}");
                        }
                        None => assert_eq!(det, 0, "{what}"),
                    }
                }
                // The null space: `A·N = 0`, of dimension `n - rank`, its free coordinates the
                // identity.
                let null = mm.null_space();
                assert_eq!(null.rows(), n, "{what}");
                assert_eq!(null.cols(), n - pivots.len(), "{what}");
                assert_eq!(&mm * &null, ModMatrix::zeros(m, null.cols(), p), "{what}");
                assert_eq!(null.rank(), null.cols(), "{what}");
                let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
                for (t, &c) in free.iter().enumerate() {
                    for (u, &d) in free.iter().enumerate() {
                        assert_eq!(null.get(d, t), (c == d) as u64, "{what} {u}");
                    }
                }
            }
        }
    }
}

/// FNV-1a over the words: the rows, the columns, then the elements row after row, as
/// `data/gen_modp.cpp` hashes FLINT's results.
fn hash(a: &ModMatrix) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    mix(a.rows() as u64);
    mix(a.cols() as u64);
    for &x in a.as_slice() {
        mix(x);
    }
    h
}

/// The ranks, determinants, reduced row echelon forms, inverses, null spaces and products by the
/// transpose of 456 seeded matrices, of 1 × 1 to 150 × 150 and of every rank, modulo the
/// thirteen primes, against FLINT's `nmod_mat` (whose results NTL's `mat_zz_p` and `mat_ZZ_p`
/// confirmed as the data were written).
#[test]
fn elimination_agrees_with_flint_and_ntl() {
    let data = include_str!("data/modp_flint.txt");
    let lines: Vec<&str> = data.lines().collect();
    assert_eq!(lines.len(), 456);
    // Under Miri, the small matrices alone.
    let lines = lines.iter().filter(|l| {
        let f: Vec<usize> = l
            .split(' ')
            .skip(1)
            .take(2)
            .map(|x| x.parse().unwrap())
            .collect();
        !cfg!(miri) || f[0] * f[1] <= 64
    });
    for line in lines {
        let f: Vec<&str> = line.split(' ').collect();
        let num = |i: usize| f[i].parse::<u64>().unwrap();
        let (seed, m, n, rank, density, p) = (
            num(0),
            num(1) as usize,
            num(2) as usize,
            num(3) as usize,
            num(4),
            num(5),
        );
        let a = matrix(&random(&mut Rng(seed), m, n, rank, density, p), m, n, p);
        assert_eq!(a.rank() as u64, num(6), "rank of {seed}");
        if m == n {
            assert_eq!(a.det(), num(7), "determinant of {seed}");
        }
        assert_eq!(hash(&a.rref().0), num(8), "echelon form of {seed}");
        match (m == n).then(|| a.inverse()).flatten() {
            Some(x) => assert_eq!(hash(&x), num(9), "inverse of {seed}"),
            None => assert!(m != n || f[9] == "-", "inverse of {seed}"),
        }
        let null = a.null_space();
        assert_eq!(null.cols() as u64, n as u64 - num(6), "nullity of {seed}");
        assert_eq!(hash(&null), num(10), "null space of {seed}");
        assert_eq!(hash(&(&a * &a.transpose())), num(11), "product of {seed}");
    }
}

#[test]
fn the_ple_decomposition_gives_back_the_matrix() {
    let mut r = Rng(0x0F1E);
    for &p in &PRIMES {
        for (m, n, rank) in shapes() {
            let a = random(&mut r, m, n, rank, 6, p);
            let mm = matrix(&a, m, n, p);
            let d = mm.ple();
            let what = format!("p = {p}, {m} × {n} of rank ≤ {rank}");
            let rk = d.rank();
            assert_eq!(rk, mm.rank(), "{what}");
            assert_eq!((d.l().rows(), d.l().cols()), (m, rk), "{what}");
            assert_eq!((d.e().rows(), d.e().cols()), (rk, n), "{what}");
            let mut seen = vec![false; m];
            for &i in d.permutation() {
                assert!(!seen[i], "{what}");
                seen[i] = true;
            }
            // `L` lower triangular with a unit diagonal, `E` in echelon form.
            for i in 0..m {
                for k in 0..rk {
                    let x = d.l().get(i, k);
                    if k == i {
                        assert_eq!(x, 1, "{what}");
                    } else if k > i {
                        assert_eq!(x, 0, "{what}");
                    }
                }
            }
            for (i, &q) in d.pivots().iter().enumerate() {
                assert!(i == 0 || q > d.pivots()[i - 1], "{what}");
                assert_ne!(d.e().get(i, q), 0, "{what}");
                assert!(d.e().row(i)[..q].iter().all(|&x| x == 0), "{what}");
            }
            let le = d.l() * d.e();
            for (i, &src) in d.permutation().iter().enumerate() {
                assert_eq!(le.row(i), mm.row(src), "{what}");
            }
        }
    }
}

/// The decomposition of a matrix of 256 rows and columns splits it at 128 columns, and its
/// update of the bottom right block, `A₂₂ − L₂₁·A₁₂`, is a product of three dimensions of 128,
/// which Strassen–Winograd's recursion takes (into a temporary, then subtracted); the inverse's
/// triangular solves, halved at 128 rows, take it too. Against the plain elimination in 128 bits,
/// for a prime of each width of the sums.
#[test]
fn the_elimination_of_a_large_matrix_takes_strassen_winograd_products() {
    if cfg!(miri) {
        return;
    }
    let mut r = Rng(0x57_E11);
    for p in [65521, 18_446_744_073_709_551_557] {
        let n = 256;
        let a = random(&mut r, n, n, n, 8, p);
        let mm = matrix(&a, n, n, p);
        let (rref, pivots, det) = naive_rref(&a, n, n, p);
        assert_eq!(pivots.len(), n, "{p}");
        assert_eq!(mm.det(), det, "{p}");
        assert_eq!(mm.rref().0.as_slice(), &rref[..], "{p}");
        let inv = mm.inverse().expect("a random matrix is nonsingular");
        let id = ModMatrix::identity(n, p);
        assert_eq!(&mm * &inv, id, "{p}");
        // The product by the inverse, checked by the classical product in 128 bits too.
        assert_eq!(
            (&mm * &inv).as_slice(),
            &naive_product(&a, inv.as_slice(), n, n, n, p)[..],
            "{p}"
        );
        // Of rank 255: the left half of full rank, the update still 128 × 128 × 128.
        let mut s = a.clone();
        for j in 0..n {
            s[(n - 1) * n + j] = s[j];
        }
        let sm = matrix(&s, n, n, p);
        assert_eq!(sm.rank(), 255, "{p}");
        assert_eq!(sm.det(), 0, "{p}");
        let null = sm.null_space();
        assert_eq!(null.cols(), 1, "{p}");
        assert_eq!(&sm * &null, ModMatrix::zeros(n, 1, p), "{p}");
    }
}

#[test]
fn solutions_satisfy_their_systems() {
    let mut r = Rng(0x501E);
    for &p in &PRIMES {
        for n in [0, 1, 2, 5, 33, if cfg!(miri) { 6 } else { 70 }] {
            for k in [0, 1, 3] {
                let a = matrix(&random(&mut r, n, n, n, 8, p), n, n, p);
                let b = matrix(&random(&mut r, n, k, k, 8, p), n, k, p);
                match a.solve(&b) {
                    Some(x) => {
                        assert_eq!((x.rows(), x.cols()), (n, k));
                        assert_eq!(&a * &x, b, "p = {p}, n = {n}, k = {k}");
                    }
                    None => assert_eq!(a.det(), 0, "p = {p}, n = {n}"),
                }
                // Singular: a repeated row.
                if n >= 2 {
                    let mut s = a.clone();
                    for j in 0..n {
                        s.set(n - 1, j, a.get(0, j));
                    }
                    assert_eq!(s.solve(&b), None);
                    assert_eq!(s.inverse(), None);
                    assert_eq!(s.det(), 0);
                }
            }
        }
    }
}

#[test]
fn products_agree_with_the_classical_product_in_128_bits() {
    let mut r = Rng(0xAB_CDEF);
    let shapes: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(0, 3, 2), (3, 0, 2), (2, 3, 0), (1, 1, 1), (3, 5, 4)]
    } else {
        &[
            (0, 3, 2),
            (3, 0, 2),
            (2, 3, 0),
            (1, 1, 1),
            (3, 5, 4),
            (17, 31, 9),
            (64, 64, 64),
        ]
    };
    for &p in &PRIMES {
        for &(m, k, n) in shapes {
            let a = random(&mut r, m, k, k, 8, p);
            let b = random(&mut r, k, n, n, 8, p);
            let c = &matrix(&a, m, k, p) * &matrix(&b, k, n, p);
            assert_eq!(c.as_slice(), &naive_product(&a, &b, m, k, n, p)[..], "{p}");
        }
    }
}

/// The products modulo primes below `2^32` on both sides of each edge of the doubles, against
/// the classical product in 128 bits: whole residues from 13 rows and columns (16 and 20 bits),
/// halves from 64 where the integer kernels hold a sum of products in one word (26 bits), and
/// halves from 24 where they need two (31 and 32 bits), with thin shapes and an unreduced sum.
#[test]
fn products_at_the_edges_of_the_doubles_equal_the_classical_product() {
    type Shapes = &'static [(usize, usize, usize)];
    let mut r = Rng(0xD0_0B1E);
    let cases: &[(u64, Shapes)] = if cfg!(miri) {
        &[(65521, &[(12, 13, 13), (13, 13, 13)])]
    } else {
        &[
            (
                65521,
                &[(12, 40, 40), (13, 13, 13), (40, 13, 41), (41, 40, 13)],
            ),
            (1_000_003, &[(13, 12, 13), (14, 15, 13), (300, 13, 14)]),
            (
                67_108_859,
                &[(63, 64, 64), (64, 64, 64), (65, 70, 64), (64, 300, 66)],
            ),
            (
                2_147_483_647,
                &[(23, 24, 24), (24, 24, 24), (25, 30, 24), (24, 24, 300)],
            ),
            (4_294_967_291, &[(24, 23, 30), (24, 25, 26), (64, 64, 63)]),
        ]
    };
    for &(p, shapes) in cases {
        for &(m, k, n) in shapes {
            // Every element p − 1 in one product of each shape: the largest sums.
            let (a, b) = if m == 64 && k == 64 {
                (vec![p - 1; m * k], vec![p - 1; k * n])
            } else {
                (random(&mut r, m, k, k, 8, p), random(&mut r, k, n, n, 8, p))
            };
            let c = &matrix(&a, m, k, p) * &matrix(&b, k, n, p);
            assert_eq!(
                c.as_slice(),
                &naive_product(&a, &b, m, k, n, p)[..],
                "{m} × {k} × {n} modulo {p}"
            );
        }
    }
}

/// Strassen–Winograd from 128 rows and columns: at and around the crossover, odd and even,
/// square and not, one level and two, against the classical product in 128 bits.
#[test]
fn products_past_the_crossover_equal_the_classical_product() {
    let mut r = Rng(0x57_2A55);
    let shapes: &[(usize, usize, usize)] = if cfg!(miri) {
        &[]
    } else {
        &[
            (127, 128, 129),
            (128, 128, 128),
            (129, 129, 129),
            (200, 131, 197),
            (131, 263, 133),
        ]
    };
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        // A prime of each kind of sum, a different one for each shape.
        for &p in &[PRIMES[i % 4], PRIMES[5 + i % 3], PRIMES[9 + i % 3]] {
            let a = random(&mut r, m, k, k, 8, p);
            let b = random(&mut r, k, n, n, 8, p);
            let c = &matrix(&a, m, k, p) * &matrix(&b, k, n, p);
            assert_eq!(
                c.as_slice(),
                &naive_product(&a, &b, m, k, n, p)[..],
                "{m} × {k} × {n} modulo {p}"
            );
        }
    }
    // Two levels, odd at both: 257 splits into 128 and a peeled row, and 128 splits again.
    if !cfg!(miri) {
        let p = PRIMES[11];
        let (m, k, n) = (257, 259, 261);
        let a = random(&mut r, m, k, k, 8, p);
        let b = random(&mut r, k, n, n, 8, p);
        let c = &matrix(&a, m, k, p) * &matrix(&b, k, n, p);
        assert_eq!(c.as_slice(), &naive_product(&a, &b, m, k, n, p)[..]);
    }
}

/// A heavier sweep than the debug profile's budget, for the release profile: matrices to
/// 400 × 400, where the decomposition recurses several levels and its products and triangular
/// solves split by Strassen–Winograd, against the plain elimination, every prime.
#[test]
#[ignore = "about a minute in the release profile; run with --ignored"]
fn a_heavier_sweep_of_shapes_and_primes() {
    let mut r = Rng(0x5EE9);
    for &p in &PRIMES {
        for (m, n, rank) in [
            (300, 300, 300),
            (300, 300, 299),
            (257, 400, 257),
            (400, 257, 200),
        ] {
            let a = random(&mut r, m, n, rank, 8, p);
            let mm = matrix(&a, m, n, p);
            let (rref, pivots, det) = naive_rref(&a, m, n, p);
            let (r2, piv2) = mm.rref();
            assert_eq!(piv2, pivots, "{p} {m} {n}");
            assert_eq!(r2.as_slice(), &rref[..], "{p} {m} {n}");
            if m == n {
                assert_eq!(mm.det(), det, "{p} {m}");
                if let Some(inv) = mm.inverse() {
                    assert_eq!(&mm * &inv, ModMatrix::identity(n, p));
                }
            }
            let null = mm.null_space();
            assert_eq!(&mm * &null, ModMatrix::zeros(m, null.cols(), p));
            let b = random(&mut r, n, 300, 300, 8, p);
            let c = &mm * &matrix(&b, n, 300, p);
            assert_eq!(c.as_slice(), &naive_product(&a, &b, m, n, 300, p)[..]);
        }
    }
}

#[test]
fn building_reduces_the_elements_and_reading_gives_them_back() {
    let p = 18_446_744_073_709_551_557;
    let a = ModMatrix::from_rows(&[[u64::MAX, p], [p - 1, 3]], p);
    assert_eq!(a.as_slice(), &[58, 0, p - 1, 3]);
    assert_eq!(a.shape(), (2, 2));
    assert!(a.is_square());
    assert_eq!(a[(1, 0)], p - 1);
    assert_eq!(a.row(1), &[p - 1, 3]);
    let mut b = a.clone();
    b.set(0, 0, u64::MAX);
    assert_eq!(b, a);
    // From numbers of either sign and any length, in either byte order.
    let big = Number::new(vec![0xAB; 300], ByteOrder::BigEndian);
    let x = Matrix::from_rows(&[
        [big.clone(), -big.clone(), Number::from(-1)],
        [Number::from(p), Number::default(), Number::from(u64::MAX)],
    ]);
    for q in [2, 3, 65521, 4_611_686_018_427_387_847, p] {
        let m = ModMatrix::from_matrix(&x, q);
        let qn = Number::from(q);
        for i in 0..2 {
            for j in 0..3 {
                let want = (&x[(i, j)] % &qn + &qn) % &qn;
                assert_eq!(Number::from(m.get(i, j)), want, "{q}");
            }
        }
        let back = m.to_matrix();
        assert_eq!(ModMatrix::from_matrix(&back, q), m);
    }
    // Empty shapes.
    for (m, n) in [(0, 0), (0, 4), (4, 0)] {
        let z = ModMatrix::zeros(m, n, 2);
        assert_eq!(z.rank(), 0);
        assert_eq!(z.transpose().shape(), (n, m));
        assert_eq!(z.null_space().shape(), (n, n));
        assert_eq!(z.null_space(), ModMatrix::identity(n, 2));
        assert_eq!(z.rref(), (z.clone(), vec![]));
        assert_eq!(z.ple().rank(), 0);
    }
    let e = ModMatrix::zeros(0, 0, 5);
    assert_eq!(e.det(), 1);
    assert_eq!(e.inverse(), Some(e.clone()));
    assert_eq!(
        e.solve(&ModMatrix::zeros(0, 2, 5)),
        Some(ModMatrix::zeros(0, 2, 5))
    );
    assert_eq!(ModMatrix::zeros(1, 1, 5).det(), 0);
    assert_eq!(ModMatrix::zeros(1, 1, 5).inverse(), None);
}

#[test]
fn the_elementwise_operations_are_those_of_the_field() {
    let mut r = Rng(0xE1E);
    for &p in &PRIMES {
        let (m, n) = (4, 5);
        let x = random(&mut r, m, n, n, 8, p);
        let y = random(&mut r, m, n, n, 8, p);
        let (a, b) = (matrix(&x, m, n, p), matrix(&y, m, n, p));
        let c = r.next();
        for i in 0..m {
            for j in 0..n {
                let (u, v) = (x[i * n + j] as u128, y[i * n + j] as u128);
                let q = p as u128;
                assert_eq!((&a + &b).get(i, j) as u128, (u + v) % q);
                assert_eq!((&a - &b).get(i, j) as u128, (u + q - v) % q);
                assert_eq!((-&a).get(i, j) as u128, (q - u) % q);
                assert_eq!(a.scale(c).get(i, j) as u128, u * (c as u128 % q) % q);
                assert_eq!(a.transpose().get(j, i), a.get(i, j));
            }
        }
        assert_eq!(a.clone() + b.clone(), &a + &b);
        assert_eq!(a.clone() - &b, &a - &b);
        assert_eq!(&a * b.transpose(), a.clone() * &b.transpose());
        assert_eq!(-a.clone(), -&a);
    }
}

#[test]
fn a_large_matrix_of_numbers_has_the_residues_of_its_determinant() {
    let mut r = Rng(0xDE7);
    let n = size(20, 4);
    let a = Matrix::from_fn(n, n, |_, _| {
        let x = Number::new(
            (0..16).map(|_| r.next() as u8).collect(),
            ByteOrder::LittleEndian,
        );
        if r.next() & 1 == 1 { -x } else { x }
    });
    let det = a.det();
    for &p in &PRIMES {
        let q = Number::from(p);
        let want = (&det % &q + &q) % &q;
        assert_eq!(
            Number::from(ModMatrix::from_matrix(&a, p).det()),
            want,
            "{p}"
        );
    }
}

#[test]
#[should_panic(expected = "the modulus 1 is not a prime")]
fn a_modulus_of_one_panics() {
    let _ = ModMatrix::zeros(1, 1, 1);
}

#[test]
#[should_panic(expected = "the modulus 0 is not a prime")]
fn a_modulus_of_zero_panics() {
    let _ = ModMatrix::identity(1, 0);
}

#[test]
#[should_panic(expected = "is not a prime")]
fn a_composite_modulus_panics() {
    // The square of the largest prime below `2^32`.
    let _ = ModMatrix::from_rows(&[[1]], 4_294_967_291 * 4_294_967_291);
}

#[test]
#[should_panic(expected = "is not a prime")]
fn the_largest_word_panics() {
    let _ = ModMatrix::from_fn(1, 1, u64::MAX, |_, _| 0);
}

#[test]
#[should_panic(expected = "the matrices are over different primes")]
fn products_over_different_primes_panic() {
    let _ = &ModMatrix::identity(2, 3) * &ModMatrix::identity(2, 5);
}

#[test]
#[should_panic(expected = "the matrices are over different primes")]
fn sums_over_different_primes_panic() {
    let _ = &ModMatrix::identity(2, 3) + &ModMatrix::identity(2, 5);
}

#[test]
#[should_panic(expected = "the matrices are not of the same shape")]
fn sums_of_different_shapes_panic() {
    let _ = &ModMatrix::identity(2, 3) - &ModMatrix::identity(3, 3);
}

#[test]
#[should_panic(expected = "the columns of the left matrix are not as many as the rows")]
fn products_of_mismatched_shapes_panic() {
    let _ = &ModMatrix::zeros(2, 3, 3) * &ModMatrix::zeros(2, 3, 3);
}

#[test]
#[should_panic(expected = "the matrix is not square")]
fn the_determinant_of_a_rectangle_panics() {
    let _ = ModMatrix::zeros(2, 3, 3).det();
}

#[test]
#[should_panic(expected = "the matrix is not square")]
fn the_inverse_of_a_rectangle_panics() {
    let _ = ModMatrix::zeros(3, 2, 3).inverse();
}

#[test]
#[should_panic(expected = "the right-hand side has not as many rows")]
fn a_right_hand_side_of_other_rows_panics() {
    let _ = ModMatrix::identity(3, 3).solve(&ModMatrix::zeros(2, 1, 3));
}

#[test]
#[should_panic(expected = "the matrices are over different primes")]
fn a_right_hand_side_over_another_prime_panics() {
    let _ = ModMatrix::identity(3, 3).solve(&ModMatrix::zeros(3, 1, 5));
}

#[test]
#[should_panic(expected = "(2, 0) is out of range")]
fn reading_outside_panics() {
    let _ = ModMatrix::zeros(2, 2, 3).get(2, 0);
}

#[test]
#[should_panic(expected = "(0, 2) is out of range")]
fn writing_outside_panics() {
    ModMatrix::zeros(2, 2, 3).set(0, 2, 1);
}

#[test]
#[should_panic(expected = "the rows are not all of the same length")]
fn ragged_rows_panic() {
    let _ = ModMatrix::from_rows(&[vec![1, 2], vec![3]], 5);
}

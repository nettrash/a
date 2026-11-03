//! Exact algorithms on matrices of numbers, against independent ones: Laplace's expansion,
//! evaluation of the characteristic polynomial, and the identities of adjugates and null spaces.

use super::common::{Rng, size};
use a::math::matrix::Matrix;
use a::math::numbers::{ByteOrder, Number};

/// A random matrix of numbers of up to `bytes` bytes, of either sign.
fn random(r: &mut Rng, m: usize, n: usize, bytes: usize) -> Matrix<Number> {
    Matrix::from_fn(m, n, |_, _| {
        let len = (r.next() % (bytes as u64 + 1)) as usize;
        let x = Number::new(
            (0..len).map(|_| r.next() as u8).collect(),
            ByteOrder::LittleEndian,
        );
        if r.next() & 1 == 1 { -x } else { x }
    })
}

/// A small random matrix of numbers in `-k..=k`.
fn small(r: &mut Rng, m: usize, n: usize, k: i64) -> Matrix<Number> {
    Matrix::from_fn(m, n, |_, _| {
        Number::from((r.next() % (2 * k as u64 + 1)) as i64 - k)
    })
}

/// The determinant by Laplace's expansion along the first row.
fn laplace(a: &Matrix<Number>) -> Number {
    let n = a.rows();
    if n == 0 {
        return Number::from(1);
    }
    let mut d = Number::default();
    for j in 0..n {
        let minor = Matrix::from_fn(n - 1, n - 1, |r, c| {
            a[(r + 1, if c < j { c } else { c + 1 })].clone()
        });
        let term = &a[(0, j)] * &laplace(&minor);
        if j % 2 == 0 {
            d += term;
        } else {
            d -= term;
        }
    }
    d
}

#[test]
fn determinants() {
    let mut r = Rng(0x5EED_0401);
    for n in 0..size(7, 4) {
        for bytes in [1, 9, 40] {
            let a = random(&mut r, n, n, bytes);
            assert_eq!(a.det(), laplace(&a), "{n} {bytes}");
        }
    }
    // `det(A·B) = det(A)·det(B)`, larger.
    let n = size(24, 6);
    let (a, b) = (random(&mut r, n, n, 8), random(&mut r, n, n, 8));
    assert_eq!((&a * &b).det(), &a.det() * &b.det());
    // Singular: a repeated row, and a zero column.
    let mut s = random(&mut r, 5, 5, 4);
    for j in 0..5 {
        s[(3, j)] = s[(1, j)].clone();
    }
    assert!(s.det().is_zero());
    let mut s = random(&mut r, 5, 5, 4);
    for i in 0..5 {
        s[(i, 2)] = Number::default();
    }
    assert!(s.det().is_zero());
    // A row swap at the first pivot.
    let p = Matrix::from_rows(&[[0, 1], [1, 0]]).map(|&x| Number::from(x));
    assert_eq!(p.det(), -1);
}

/// `A = B·C` with `B` of `m × r` and `C` of `r × n`, of rank `r` for random factors.
fn of_rank(r: &mut Rng, m: usize, n: usize, rank: usize) -> Matrix<Number> {
    &small(r, m, rank, 9) * &small(r, rank, n, 9)
}

#[test]
fn ranks_and_null_spaces() {
    let mut r = Rng(0x5EED_0402);
    for (m, n, rank) in [
        (4, 4, 4),
        (5, 5, 2),
        (6, 3, 3),
        (3, 7, 2),
        (8, 9, 5),
        (4, 4, 0),
    ] {
        let a = of_rank(&mut r, m, n, rank);
        assert_eq!(a.rank(), rank, "{m} {n} {rank}");
        let null = a.null_space();
        assert_eq!(null.shape(), (n, n - rank));
        assert!((&a * &null).as_slice().iter().all(Number::is_zero));
        // Independent, and each column primitive.
        assert_eq!(null.rank(), n - rank);
        for j in 0..null.cols() {
            let g = null
                .column(j)
                .iter()
                .fold(Number::default(), |g, x| g.gcd(x));
            assert_eq!(g, 1);
        }
    }
    let z = Matrix::<Number>::zeros(3, 4);
    assert_eq!(z.rank(), 0);
    assert_eq!(z.null_space(), Matrix::identity(4));
    assert_eq!(Matrix::<Number>::identity(3).null_space().shape(), (3, 0));
}

/// `p(x)` for coefficients from the highest power down.
fn evaluate(p: &[Number], x: &Number) -> Number {
    p.iter().fold(Number::default(), |s, c| &(&s * x) + c)
}

#[test]
fn characteristic_polynomials() {
    let mut r = Rng(0x5EED_0403);
    for n in 0..size(9, 4) {
        let a = random(&mut r, n, n, 6);
        let p = a.characteristic_polynomial();
        assert_eq!(p.len(), n + 1);
        assert_eq!(p[0], 1);
        // `p(x) = det(x·I - A)` at `n + 1` points determines it.
        for x in 0..=n as i64 {
            let x = Number::from(x * 7 - 3);
            let m = &Matrix::<Number>::identity(n) * &x - &a;
            assert_eq!(evaluate(&p, &x), m.det(), "{n}");
        }
        // Cayley and Hamilton.
        let mut c = Matrix::<Number>::zeros(n, n);
        for coefficient in &p {
            c = &c * &a;
            for i in 0..n {
                c[(i, i)] += coefficient;
            }
        }
        assert!(c.as_slice().iter().all(Number::is_zero), "{n}");
    }
}

#[test]
fn adjugates_and_inverses() {
    let mut r = Rng(0x5EED_0404);
    for n in 0..size(8, 4) {
        let a = random(&mut r, n, n, 5);
        let adj = a.adjugate();
        let d = a.det();
        let di = &Matrix::<Number>::identity(n) * &d;
        assert_eq!(&a * &adj, di);
        assert_eq!(&adj * &a, di);
        if n <= 4 {
            // The cofactors themselves.
            for i in 0..n {
                for j in 0..n {
                    let minor = Matrix::from_fn(n - 1, n - 1, |r, c| {
                        a[(if r < j { r } else { r + 1 }, if c < i { c } else { c + 1 })].clone()
                    });
                    let c = laplace(&minor);
                    let c = if (i + j) % 2 == 0 { c } else { -c };
                    assert_eq!(adj[(i, j)], c);
                }
            }
        }
        match a.inverse() {
            Some((adj2, det)) => {
                assert_eq!((adj2, det), (adj, d));
            }
            None => assert!(d.is_zero()),
        }
    }
    // Singular of rank n - 1: a nonzero adjugate with `A·adj(A) = 0`.
    let a = of_rank(&mut r, 4, 4, 3);
    assert!(a.inverse().is_none());
    let adj = a.adjugate();
    assert!(!adj.as_slice().iter().all(Number::is_zero));
    assert!((&a * &adj).as_slice().iter().all(Number::is_zero));
}

#[test]
fn generic_operations_on_numbers() {
    let mut r = Rng(0x5EED_0405);
    let (a, b) = (small(&mut r, 3, 4, 50), small(&mut r, 3, 4, 50));
    let as_i = |m: &Matrix<Number>| m.map(|x| i64::try_from(x).unwrap());
    let (ai, bi) = (as_i(&a), as_i(&b));
    assert_eq!(
        as_i(&(&a + &b)),
        Matrix::from_fn(3, 4, |i, j| ai[(i, j)] + bi[(i, j)])
    );
    assert_eq!(
        as_i(&(&a - &b)),
        Matrix::from_fn(3, 4, |i, j| ai[(i, j)] - bi[(i, j)])
    );
    assert_eq!(
        as_i(&a.hadamard(&b)),
        Matrix::from_fn(3, 4, |i, j| ai[(i, j)] * bi[(i, j)])
    );
    let product = &a * &b.transpose();
    let want = Matrix::from_fn(3, 3, |i, j| {
        (0..4).map(|k| ai[(i, k)] * bi[(j, k)]).sum::<i64>()
    });
    assert_eq!(as_i(&product), want);
    let three = Number::from(3);
    assert_eq!(
        as_i(&(&three * &a)),
        Matrix::from_fn(3, 4, |i, j| 3 * ai[(i, j)])
    );
    let s = small(&mut r, 3, 3, 3);
    let si = as_i(&s);
    let mut p = Matrix::<i64>::from_fn(3, 3, |i, j| i64::from(i == j));
    for k in 0..=6 {
        assert_eq!(as_i(&s.pow(k)), p);
        p = Matrix::from_fn(3, 3, |i, j| (0..3).map(|q| p[(i, q)] * si[(q, j)]).sum());
    }
    assert_eq!(s.trace(), si[(0, 0)] + si[(1, 1)] + si[(2, 2)]);
    // Numbers of either byte order mix.
    let big = Matrix::from_fn(2, 2, |i, j| {
        Number::new_from_u64((i * 2 + j + 1) as u64, ByteOrder::BigEndian)
    });
    assert_eq!(big.det(), -2);
}

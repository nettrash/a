//! Matrices of `BigFloat`s and `BigComplex`es: every element of a product correctly rounded
//! against the exact rational dot products; determinants, solutions and inverses within the
//! bound of the elimination of the exact rational ones; exact ranks of low-rank matrices; and the
//! norms correctly rounded.

use super::common::{below, float, one, pow2, rational, reference, same, size};
use super::complex::sqrt_reference;
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::matrix::{Matrix, Norm};
use a::math::rational::Rational;
use core::cmp::Ordering;

/// A random matrix of floats of precisions up to `max`, exponents within `exps`.
fn random(s: &mut u64, rows: usize, cols: usize, max: u64, exps: i64) -> Matrix<BigFloat> {
    Matrix::from_fn(rows, cols, |_, _| {
        if below(s, 12) == 0 {
            BigFloat::zero(1 + below(s, max))
        } else {
            float(s, max, exps)
        }
    })
}

fn exact(a: &Matrix<BigFloat>) -> Matrix<Rational> {
    a.map(rational)
}

/// The largest precision of the elements.
fn prec_of(a: &Matrix<BigFloat>) -> u64 {
    a.as_slice().iter().map(BigFloat::prec).max().unwrap_or(1)
}

#[test]
fn every_element_of_a_product_is_correctly_rounded() {
    let mut s = 0x5EED_3A7E_0000_0001u64;
    for i in 0..size(60, 3) {
        let (m, k, n) = (
            1 + below(&mut s, 6) as usize,
            1 + below(&mut s, if i % 5 == 0 && !cfg!(miri) { 30 } else { 6 }) as usize,
            1 + below(&mut s, 6) as usize,
        );
        let max = if i % 7 == 0 && !cfg!(miri) { 400 } else { 90 };
        let exps = if i % 3 == 0 { 600 } else { 20 };
        let a = random(&mut s, m, k, max, exps);
        let b = random(&mut s, k, n, max, exps);
        let c = &a * &b;
        let prec = prec_of(&a).max(prec_of(&b));
        let (ea, eb) = (exact(&a), exact(&b));
        let ec = &ea * &eb;
        for r in 0..m {
            for col in 0..n {
                let got = (c[(r, col)].clone(), Ordering::Equal);
                let q = &ec[(r, col)];
                if q.is_zero() {
                    assert!(got.0.is_zero(), "({r}, {col}) of {a:?} · {b:?}");
                    continue;
                }
                let want = reference(q, prec, Round::Nearest);
                // The ternary is not kept: the value and the precision.
                same(
                    &got,
                    &(want.0, Ordering::Equal),
                    &format!("({r}, {col}) of {a:?} · {b:?}"),
                );
            }
        }
    }
}

#[test]
fn every_part_of_a_complex_product_is_correctly_rounded() {
    let mut s = 0x5EED_3A7E_0000_0002u64;
    for _ in 0..size(30, 2) {
        let (m, k, n) = (
            1 + below(&mut s, 4) as usize,
            1 + below(&mut s, 8) as usize,
            1 + below(&mut s, 4) as usize,
        );
        let mk = |s: &mut u64, r, c| {
            Matrix::from_fn(r, c, |_, _| {
                BigComplex::new(float(s, 80, 30), float(s, 80, 30))
            })
        };
        let a = mk(&mut s, m, k);
        let b = mk(&mut s, k, n);
        let c = &a * &b;
        let prec = a
            .as_slice()
            .iter()
            .chain(b.as_slice())
            .map(|z| z.re().prec().max(z.im().prec()))
            .max()
            .unwrap();
        for r in 0..m {
            for col in 0..n {
                let (mut re, mut im) = (Rational::from(0u8), Rational::from(0u8));
                for p in 0..k {
                    let (x, y) = (&a[(r, p)], &b[(p, col)]);
                    let (xr, xi, yr, yi) = (
                        rational(x.re()),
                        rational(x.im()),
                        rational(y.re()),
                        rational(y.im()),
                    );
                    re = re + &xr * &yr - &xi * &yi;
                    im = im + &xr * &yi + &xi * &yr;
                }
                for (got, q) in [(c[(r, col)].re(), &re), (c[(r, col)].im(), &im)] {
                    if q.is_zero() {
                        assert!(got.is_zero());
                    } else {
                        assert_eq!(got.prec(), prec);
                        assert_eq!(*got, reference(q, prec, Round::Nearest).0);
                    }
                }
            }
        }
    }
}

/// A random diagonally dominant matrix, well conditioned: its solutions are near the exact ones
/// to a few units of the working precision.
fn dominant(s: &mut u64, n: usize, prec: u64) -> Matrix<BigFloat> {
    let mut a = Matrix::from_fn(n, n, |_, _| float(s, prec, 1));
    for i in 0..n {
        let big = BigFloat::from(n as u64 * 4);
        let x = a[(i, i)].clone();
        a[(i, i)] = (&x + &big).prec_round(prec, Round::Nearest).0;
    }
    a
}

/// `|x - y| ≤ 2^-bits·scale` for the rational `y`.
fn near(x: &BigFloat, y: &Rational, scale: &Rational, bits: u64) -> bool {
    let d = rational(x) - y;
    let d = if d.is_negative() { -d } else { d };
    d <= scale * &Rational::new(one(), pow2(bits))
}

#[test]
fn determinants_solutions_and_inverses_are_near_the_exact_ones() {
    let mut s = 0x5EED_3A7E_0000_0003u64;
    for i in 0..size(40, 2) {
        let n = 1 + below(
            &mut s,
            if cfg!(miri) {
                3
            } else if i % 4 == 0 {
                9
            } else {
                5
            },
        ) as usize;
        let prec = [24u64, 53, 64, 113, 200, 500][i % 6];
        let a = dominant(&mut s, n, prec);
        let b = Matrix::from_fn(n, 2, |_, _| float(&mut s, prec, 2));
        let (ea, eb) = (exact(&a), exact(&b));
        // The elimination of a dominant matrix loses a few bits for each row.
        let lost = 8 + 2 * n as u64;
        let det = a.det();
        let edet = ea.det();
        let scale = if edet.is_negative() {
            -edet.clone()
        } else {
            edet.clone()
        };
        assert!(
            near(&det, &edet, &scale, prec.saturating_sub(lost)),
            "det of {a:?}: {det:?}"
        );
        let x = a.solve(&b).expect("not singular");
        let ex = ea.solve(&eb).expect("not singular");
        let inv = a.inverse().expect("not singular");
        let einv = ea.inverse().expect("not singular");
        let biggest = |m: &Matrix<Rational>| {
            m.as_slice()
                .iter()
                .map(|q| {
                    if q.is_negative() {
                        -q.clone()
                    } else {
                        q.clone()
                    }
                })
                .fold(Rational::from(0u8), |a, q| if q > a { q } else { a })
        };
        let (sx, sinv) = (biggest(&ex), biggest(&einv));
        for (g, w) in x.as_slice().iter().zip(ex.as_slice()) {
            assert_eq!(g.prec(), prec);
            assert!(
                near(g, w, &sx, prec.saturating_sub(lost)),
                "solve {a:?} {b:?}: {g:?}"
            );
        }
        for (g, w) in inv.as_slice().iter().zip(einv.as_slice()) {
            assert!(
                near(g, w, &sinv, prec.saturating_sub(lost)),
                "inverse of {a:?}: {g:?}"
            );
        }
    }
}

#[test]
fn exact_eliminations_are_exact_and_singular_matrices_have_none() {
    // Pivots that are powers of two and products that fit: every step exact.
    let a = Matrix::from_rows(&[[8, 4, 2], [4, 6, 1], [2, 1, 5]])
        .map(|&x| BigFloat::from(x).with_prec(200));
    let ea = Matrix::from_rows(&[[8, 4, 2], [4, 6, 1], [2, 1, 5]]).map(|&x| Rational::from(x));
    assert_eq!(rational(&a.det()), ea.det());
    let z = Matrix::<BigFloat>::zeros(3, 3);
    assert!(z.det().is_zero());
    assert!(z.solve(&Matrix::zeros(3, 1)).is_none());
    assert!(z.inverse().is_none());
    let singular = Matrix::from_rows(&[[1, 2], [2, 4]]).map(|&x| BigFloat::from(x));
    assert!(singular.det().is_zero());
    assert!(singular.inverse().is_none());
    assert_eq!(Matrix::<BigFloat>::zeros(0, 0).det(), BigFloat::from(1));
    // Complex: `[[1, i], [-i, 2]]`, determinant 2 - 1 = 1, inverse `[[2, -i], [i, 1]]`.
    let i = BigComplex::i(64);
    let c = Matrix::from_rows(&[[BigComplex::from(1), i.clone()], [-&i, BigComplex::from(2)]]);
    assert_eq!(c.det(), BigComplex::from(1));
    let inv = c.inverse().unwrap();
    let want = Matrix::from_rows(&[[BigComplex::from(2), -&i], [i.clone(), BigComplex::from(1)]]);
    assert_eq!(inv, want);
    assert_eq!(&c * &inv, Matrix::identity(2));
}

#[test]
fn ranks_of_low_rank_matrices() {
    let mut s = 0x5EED_3A7E_0000_0004u64;
    for _ in 0..size(40, 2) {
        let (m, n) = (1 + below(&mut s, 6) as usize, 1 + below(&mut s, 6) as usize);
        let r = below(&mut s, m.min(n) as u64 + 1) as usize;
        // A sum of `r` outer products of small integers, exactly representable, of rank `r`
        // almost surely.
        let u = Matrix::from_fn(m, r, |_, _| BigFloat::from(below(&mut s, 19) as i64 - 9));
        let v = Matrix::from_fn(r, n, |_, _| BigFloat::from(below(&mut s, 19) as i64 - 9));
        let a = (&u * &v).map(|x| x.with_prec(200));
        let exact_rank = exact(&a).rank();
        assert_eq!(a.rank(), exact_rank, "{a:?}");
        let tiny = BigFloat::from_parts(&one(), -400, 1, Round::Nearest).0;
        assert!(a.rank_with_tolerance(&tiny) >= exact_rank);
        let c = a.map(|x| BigComplex::new(x.clone(), x.clone()));
        assert_eq!(c.rank(), exact_rank);
    }
}

#[test]
fn norms_are_correctly_rounded() {
    let mut s = 0x5EED_3A7E_0000_0005u64;
    for i in 0..size(80, 4) {
        let (m, n) = (1 + below(&mut s, 5) as usize, 1 + below(&mut s, 5) as usize);
        let a = random(&mut s, m, n, 120, if i % 2 == 0 { 300 } else { 10 });
        let prec = prec_of(&a);
        let abs = |q: Rational| if q.is_negative() { -q } else { q };
        let ea = exact(&a);
        let col = (0..n)
            .map(|j| (0..m).fold(Rational::from(0u8), |t, r| t + abs(ea[(r, j)].clone())))
            .fold(Rational::from(0u8), |x, y| if y > x { y } else { x });
        let row = (0..m)
            .map(|r| (0..n).fold(Rational::from(0u8), |t, j| t + abs(ea[(r, j)].clone())))
            .fold(Rational::from(0u8), |x, y| if y > x { y } else { x });
        let sq = ea
            .as_slice()
            .iter()
            .fold(Rational::from(0u8), |t, q| t + q * q);
        let check = |got: BigFloat, q: &Rational, sqrt: bool| {
            if q.is_zero() {
                assert!(got.is_zero());
            } else {
                let want = if sqrt {
                    sqrt_reference(q, prec, Round::Nearest)
                } else {
                    reference(q, prec, Round::Nearest)
                };
                assert_eq!(got.prec(), prec);
                assert_eq!(got, want.0, "{a:?}");
            }
        };
        check(a.norm(Norm::One), &col, false);
        check(a.norm(Norm::Infinity), &row, false);
        check(a.norm(Norm::Frobenius), &sq, true);
    }
}

#[test]
fn matrices_of_floats_have_the_generic_operations() {
    let a = Matrix::from_rows(&[[1, 2], [3, 4]]).map(|&x| BigFloat::from(x));
    let b = a.transpose();
    assert_eq!(b[(0, 1)], BigFloat::from(3));
    let mut c = a.clone();
    c.swap_rows(0, 1);
    c.add_scaled_row(0, 1, &BigFloat::from(-3));
    assert_eq!(c[(0, 0)], BigFloat::from(0));
    assert_eq!(c[(0, 1)], BigFloat::from(-2));
    assert_eq!(a.trace(), BigFloat::from(5));
    assert_eq!(&a + &a, a.map(|x| x * BigFloat::from(2)));
    assert_eq!(a.pow(2), &a * &a);
    let z = Matrix::from_rows(&[[BigComplex::i(53)]]);
    assert_eq!(z.conjugate_transpose()[(0, 0)], -BigComplex::i(53));
    assert_eq!(z.to_string(), "(0 1)");
}

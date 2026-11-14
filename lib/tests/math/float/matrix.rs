//! Matrices of `BigFloat`s and `BigComplex`es: every element of a product correctly rounded
//! against the exact rational dot products; determinants, solutions and inverses within the
//! bound of the elimination of the exact rational ones; exact ranks of low-rank matrices; and the
//! norms correctly rounded.

use super::common::{below, float, one, pow2, rational, reference, same, size};
use super::complex::sqrt_reference;
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::matrix::{Matrix, Norm};
use a::math::numbers::{ByteOrder, Number};
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

/// `q` rounded to nearest at `prec` bits, as a rational: the value of the float.
fn rounded(q: &Rational, prec: u64) -> Rational {
    if q.is_zero() {
        Rational::from(0u8)
    } else {
        rational(&reference(q, prec, Round::Nearest).0)
    }
}

/// The factorisation of `Matrix::<BigFloat>::det` and `solve` replayed on exact rationals, each
/// operation rounded once to nearest at `prec`: Doolittle's method with partial pivoting, each
/// element of `U` and of `L` before its division one exact dot product rounded once; `None` for a
/// pivot that is zero.
fn replayed(a: &Matrix<BigFloat>, prec: u64) -> Option<(Vec<Rational>, Vec<usize>, bool)> {
    let n = a.rows();
    let mut lu: Vec<Rational> = a.as_slice().iter().map(rational).collect();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut odd = false;
    let abs = |q: &Rational| {
        if q.is_negative() {
            -q.clone()
        } else {
            q.clone()
        }
    };
    for k in 0..n {
        for i in k..n {
            let mut v = lu[i + k * n].clone();
            for p in 0..k {
                v -= &lu[i + p * n] * &lu[p + k * n];
            }
            lu[i + k * n] = rounded(&v, prec);
        }
        let mut r = k;
        for i in k + 1..n {
            if abs(&lu[i + k * n]) > abs(&lu[r + k * n]) {
                r = i;
            }
        }
        if lu[r + k * n].is_zero() {
            return None;
        }
        if r != k {
            for j in 0..n {
                lu.swap(r + j * n, k + j * n);
            }
            perm.swap(r, k);
            odd = !odd;
        }
        for i in k + 1..n {
            let q = &lu[i + k * n] / &lu[k + k * n];
            lu[i + k * n] = rounded(&q, prec);
        }
        for j in k + 1..n {
            let mut v = lu[k + j * n].clone();
            for p in 0..k {
                v -= &lu[k + p * n] * &lu[p + j * n];
            }
            lu[k + j * n] = rounded(&v, prec);
        }
    }
    Some((lu, perm, odd))
}

#[test]
fn factorisations_are_their_rounded_steps_exactly() {
    // The determinant and the solution of every matrix are its factorisation's steps, each
    // correctly rounded: in registers (parts to 128 bits) and on limbs (beyond) alike.
    let mut s = 0x5EED_3A7E_0000_0010u64;
    for i in 0..size(400, 6) {
        let n = 1 + below(&mut s, if cfg!(miri) { 3 } else { 6 }) as usize;
        let max = match i % 10 {
            0 => 300,
            1 | 2 => 64,
            _ => 128,
        };
        let exps = if i % 9 == 0 { 400 } else { 8 };
        let mut a = random(&mut s, n, n, max, exps);
        if i % 4 == 0 {
            // Dominant, so that the solutions are not far out.
            for k in 0..n {
                let x = &a[(k, k)] + &BigFloat::from(4 * n as u64);
                a[(k, k)] = x.with_prec(a[(k, k)].prec());
            }
        }
        let b = random(&mut s, n, 2, max, exps);
        let prec = prec_of(&a).max(prec_of(&b));
        let what = format!("{a:?}, {b:?} at {prec}");
        let x = a.solve(&b);
        let d = a.det();
        let pa = prec_of(&a);
        match replayed(&a, pa) {
            None => assert!(d.is_zero(), "det {what}"),
            Some((lu, _, odd)) => {
                let mut want = Rational::from(1u8);
                for k in 0..n {
                    want = rounded(&(&want * &lu[k + k * n]), pa);
                }
                if odd {
                    want = -want;
                }
                assert_eq!(d.prec(), pa, "det {what}");
                assert!(rational(&d) == want, "det {what}: {d:?}");
            }
        }
        match replayed(&a, prec) {
            None => assert!(x.is_none(), "solve {what}"),
            Some((lu, perm, _)) => {
                let x = x.expect("a solution");
                for j in 0..2 {
                    let mut y: Vec<Rational> = Vec::new();
                    for r in 0..n {
                        let mut v = rational(&b[(perm[r], j)]);
                        for p in 0..r {
                            v -= &lu[r + p * n] * &y[p];
                        }
                        y.push(rounded(&v, prec));
                    }
                    for r in (0..n).rev() {
                        let mut v = y[r].clone();
                        for p in r + 1..n {
                            v -= &lu[r + p * n] * &y[p];
                        }
                        let v = rounded(&v, prec);
                        y[r] = rounded(&(&v / &lu[r + r * n]), prec);
                    }
                    for r in 0..n {
                        assert_eq!(x[(r, j)].prec(), prec, "solve {what}");
                        assert!(
                            rational(&x[(r, j)]) == y[r],
                            "solve ({r}, {j}) {what}: {x:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complex_factorisations_solve_dominant_systems() {
    // The complex elimination on the stack (parts to 128 bits) solves a dominant system near the
    // exact solution, as on limbs (beyond 128 bits).
    let mut s = 0x5EED_3A7E_0000_0011u64;
    for i in 0..size(60, 3) {
        let n = 1 + below(&mut s, 5) as usize;
        let prec = if i % 3 == 0 {
            200
        } else {
            1 + below(&mut s, 128).max(20)
        };
        let mut a = Matrix::from_fn(n, n, |_, _| {
            BigComplex::new(float(&mut s, prec, 1), float(&mut s, prec, 1)).with_prec(prec)
        });
        for k in 0..n {
            let x = &a[(k, k)] + &BigComplex::from(8 * n as u64);
            a[(k, k)] = x.with_prec(prec);
        }
        let x0 = Matrix::from_fn(n, 1, |_, _| {
            BigComplex::new(float(&mut s, prec, 1), float(&mut s, prec, 1)).with_prec(prec)
        });
        let b = &a * &x0;
        let x = a.solve(&b).expect("a solution");
        for r in 0..n {
            for (u, v) in [
                (x[(r, 0)].re(), x0[(r, 0)].re()),
                (x[(r, 0)].im(), x0[(r, 0)].im()),
            ] {
                let d = rational(u) - rational(v);
                let d = if d.is_negative() { -d } else { d };
                assert!(
                    d <= Rational::new(one(), pow2(prec.saturating_sub(12))),
                    "{a:?} \\ {b:?} at {prec}: {x:?}"
                );
            }
        }
    }
}

/// A float of `p` bits with few bits set, of the exponent `e` of `0.1…·2^e`, of either sign.
fn sparse_float(s: &mut u64, p: u64, e: i64) -> BigFloat {
    let mut m = Number::new_from_u8(0, ByteOrder::LittleEndian);
    m.set_bit(p - 1, true);
    for _ in 0..below(s, 3) {
        m.set_bit(below(s, p), true);
    }
    if below(s, 2) == 0 {
        m = -m;
    }
    BigFloat::from_parts(&m, e - p as i64, p, Round::Nearest).0
}

#[test]
fn sparse_products_and_ties_round_once() {
    // Sums `x + h + ε` with `h` half a unit of `x`'s last place and `ε` far below (beyond the
    // top 256 bits of the sum): the tie is broken by `ε` alone. Then products of matrices of
    // sparse elements spread over 300 bits, against the exact rationals.
    let mut s = 0x5EED_3A7E_0000_0012u64;
    for i in 0..size(4000, 40) {
        let p = 1 + below(&mut s, 128);
        let e = below(&mut s, 20) as i64 - 10;
        let mx = super::common::mantissa(&mut s, p);
        let mx = &(&mx >> 1usize) << 1usize;
        let mx = if mx.bits() < p {
            super::common::mantissa(&mut s, p)
        } else {
            mx
        };
        let neg = below(&mut s, 2) == 0;
        let sign = |m: Number| if neg { -m } else { m };
        let x = BigFloat::from_parts(&sign(mx), e - p as i64, p, Round::Nearest).0;
        let h = BigFloat::from_parts(&sign(one()), e - p as i64 - 1, 1, Round::Nearest).0;
        let far = 260 + below(&mut s, 200) as i64;
        let eps = if below(&mut s, 2) == 0 { one() } else { -one() };
        let eps = BigFloat::from_parts(&eps, e - p as i64 - far, 1, Round::Nearest).0;
        let row = Matrix::from_rows(&[[x.clone(), h.clone(), eps.clone()]]);
        let unit = BigFloat::from(1).with_prec(1);
        let col = Matrix::from_rows(&[[unit.clone()], [unit.clone()], [unit.clone()]]);
        let got = (&row * &col)[(0, 0)].clone();
        let q = rational(&x) + rational(&h) + rational(&eps);
        let prec = p;
        same(
            &(got, Ordering::Equal),
            &(reference(&q, prec, Round::Nearest).0, Ordering::Equal),
            &format!("{x:?} + {h:?} + {eps:?}"),
        );
        // The same sum as a substitution: `y_2 = b_2 + b_0 + b_1` of a unit lower triangle.
        let minus = BigFloat::from(-1).with_prec(1);
        let (o, z) = (unit.clone(), BigFloat::zero(1));
        let a = Matrix::from_rows(&[
            [o.clone(), z.clone(), z.clone()],
            [z.clone(), o.clone(), z.clone()],
            [minus.clone(), minus.clone(), o.clone()],
        ]);
        let b = Matrix::from_rows(&[[h.clone()], [eps.clone()], [x.clone()]]);
        let sol = a.solve(&b).expect("a solution");
        same(
            &(sol[(2, 0)].clone(), Ordering::Equal),
            &(reference(&q, prec, Round::Nearest).0, Ordering::Equal),
            &format!("solve {x:?} + {h:?} + {eps:?}"),
        );
        // Sparse matrices.
        if i % 4 == 0 {
            let (m, k, n) = (
                1 + below(&mut s, 3) as usize,
                1 + below(&mut s, 5) as usize,
                1 + below(&mut s, 3) as usize,
            );
            let pm = 1 + below(&mut s, 128);
            let el = |s: &mut u64| {
                let p = 1 + below(s, pm);
                let e = below(s, 300) as i64 - 150;
                sparse_float(s, p, e)
            };
            let a = Matrix::from_fn(m, k, |_, _| el(&mut s));
            let b = Matrix::from_fn(k, n, |_, _| el(&mut s));
            let c = &a * &b;
            let prec = prec_of(&a).max(prec_of(&b));
            let ec = &exact(&a) * &exact(&b);
            for r in 0..m {
                for j in 0..n {
                    let q = &ec[(r, j)];
                    let what = format!("({r}, {j}) of {a:?} · {b:?}");
                    if q.is_zero() {
                        assert!(
                            c[(r, j)].is_zero() && c[(r, j)].is_sign_positive(),
                            "{what}"
                        );
                    } else {
                        same(
                            &(c[(r, j)].clone(), Ordering::Equal),
                            &(reference(q, prec, Round::Nearest).0, Ordering::Equal),
                            &what,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cancelling_factorisations_give_positive_zeros() {
    // A solution whose dot product cancels exactly is `+0` (IEEE 754's sum to nearest), in
    // registers and on limbs alike: `x_1 = b_1 - b_0` for `b_0 = b_1`.
    let mut s = 0x5EED_3A7E_0000_0013u64;
    for _ in 0..size(200, 10) {
        let p = if below(&mut s, 3) == 0 {
            129 + below(&mut s, 200)
        } else {
            1 + below(&mut s, 128)
        };
        let v = float(&mut s, p, 10).with_prec(p);
        let (o, z) = (BigFloat::from(1).with_prec(p), BigFloat::zero(p));
        let a = Matrix::from_rows(&[[o.clone(), z.clone()], [o.clone(), o.clone()]]);
        let b = Matrix::from_rows(&[[v.clone()], [v.clone()]]);
        let x = a.solve(&b).expect("a solution");
        assert!(
            x[(1, 0)].is_zero() && x[(1, 0)].is_sign_positive(),
            "{a:?} \\ {b:?}: {x:?}"
        );
        // And `-0` stays `-0` where every term is `-0`: `x_0 = b_0 = -0`.
        let b = Matrix::from_rows(&[[BigFloat::neg_zero(p)], [BigFloat::neg_zero(p)]]);
        let x = a.solve(&b).expect("a solution");
        assert!(
            x[(0, 0)].is_zero() && x[(0, 0)].is_sign_negative(),
            "{a:?} \\ {b:?}: {x:?}"
        );
    }
}

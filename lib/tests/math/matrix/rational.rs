//! Exact algorithms on matrices of rationals, and the rational solutions of systems of integers,
//! against their defining identities and independent methods: Leibniz's formula, the
//! determinants of matrices of integers scaled by known factors, `A·X = B`, `A·A⁻¹ = I` and
//! `A·N = 0`.

use super::common::{Rng, size};
use a::math::matrix::Matrix;
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;

/// The byte order for the pick `k`.
fn order(k: u64) -> ByteOrder {
    if k & 1 == 0 {
        ByteOrder::LittleEndian
    } else {
        ByteOrder::BigEndian
    }
}

/// A random integer of up to `bytes` bytes, of either sign, in the byte order `o`.
fn integer(r: &mut Rng, bytes: usize, o: ByteOrder) -> Number {
    let len = (r.next() % (bytes as u64 + 1)) as usize;
    let mut v: Vec<u8> = (0..len).map(|_| r.next() as u8).collect();
    if o == ByteOrder::BigEndian {
        v.reverse();
    }
    Number::new_signed(v, r.next() & 1 == 1, o)
}

/// What the elements of a random matrix are.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// Integers in `-9..=9`.
    Small,
    /// Fractions of parts in `-9..=9` and `1..=9`.
    Fractions,
    /// Fractions of parts of up to this many bytes.
    Big(usize),
    /// Integers of up to this many bytes.
    Integers(usize),
}

/// A random rational of the kind `k`, in a random byte order or `o`.
fn rational(r: &mut Rng, k: Kind, o: Option<ByteOrder>) -> Rational {
    let o = o.unwrap_or_else(|| order(r.next()));
    let tiny = |r: &mut Rng, lo: i64| {
        let x = (r.next() % (10 - lo) as u64) as i64 + lo;
        let x = if lo == 0 && r.next() & 1 == 1 { -x } else { x };
        Number::new_from_i64(x, o)
    };
    match k {
        Kind::Small => Rational::from_integer(tiny(r, 0)),
        Kind::Fractions => Rational::new(tiny(r, 0), tiny(r, 1)),
        Kind::Big(bytes) => {
            let mut d = integer(r, bytes, o);
            while d.is_zero() {
                d = integer(r, bytes, o);
            }
            Rational::new(integer(r, bytes, o), d)
        }
        Kind::Integers(bytes) => Rational::from_integer(integer(r, bytes, o)),
    }
}

/// A random `m × n` matrix of the kind `k`, the top left element in the byte order `o`.
fn random(r: &mut Rng, m: usize, n: usize, k: Kind, o: ByteOrder) -> Matrix<Rational> {
    Matrix::from_fn(m, n, |i, j| rational(r, k, (i == 0 && j == 0).then_some(o)))
}

/// Asserts the form every rational keeps, and its byte order.
#[track_caller]
fn check(x: &Rational, o: ByteOrder) {
    let (n, d) = (x.numer(), x.denom());
    assert!(d.is_positive(), "{x:?}");
    assert_eq!(n.byte_order(), o, "{x:?}");
    assert_eq!(d.byte_order(), o, "{x:?}");
    assert_eq!(n.gcd(d), 1, "{x:?}");
    for p in [n, d] {
        // An empty vector is zero too, as `Rational::default()` and the zeros of
        // `Matrix::zeros` hold it.
        let top = match p.byte_order() {
            ByteOrder::BigEndian => p.value.first(),
            ByteOrder::LittleEndian => p.value.last(),
        };
        assert!(top.is_some_and(|&t| t != 0) || p.value.len() <= 1, "{x:?}");
        assert!(!(p.is_zero() && p.is_negative()), "{x:?}");
    }
    if n.is_zero() {
        assert_eq!(*d, 1);
    }
}

/// Asserts the form of every element of `m`, in the byte order `o`.
#[track_caller]
fn check_all(m: &Matrix<Rational>, o: ByteOrder) {
    for x in m.as_slice() {
        check(x, o);
    }
}

/// The determinant by Leibniz's formula: the sum over the permutations, by their signs.
fn leibniz(a: &Matrix<Rational>) -> Rational {
    fn go(
        a: &Matrix<Rational>,
        row: usize,
        used: &mut Vec<bool>,
        sign: bool,
        p: Rational,
    ) -> Rational {
        let n = a.rows();
        if row == n {
            return if sign { -p } else { p };
        }
        let mut s = Rational::default();
        // The sign of a permutation flips with each column to the left of the one chosen that is
        // still free: the inversions it makes with the rows below.
        let mut flips = false;
        for j in 0..n {
            if used[j] {
                continue;
            }
            used[j] = true;
            s += go(a, row + 1, used, sign != flips, &p * &a[(row, j)]);
            used[j] = false;
            flips = !flips;
        }
        s
    }
    go(a, 0, &mut vec![false; a.rows()], false, Rational::from(1))
}

#[test]
fn determinants_by_leibniz() {
    let mut r = Rng(0x5EED_1601);
    for n in 0..size(6, 4) {
        for k in [Kind::Small, Kind::Fractions, Kind::Big(9), Kind::Big(24)] {
            for o in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
                let a = random(&mut r, n, n, k, o);
                let d = a.det();
                check(&d, if n == 0 { ByteOrder::LittleEndian } else { o });
                assert_eq!(d, leibniz(&a), "{n} {k:?}");
            }
        }
    }
}

#[test]
fn determinants_of_scaled_integers() {
    let mut r = Rng(0x5EED_1602);
    for n in 1..size(12, 4) {
        for bytes in [1, 8, 9, 30] {
            // `A = S⁻¹·N·T⁻¹` for diagonal `S` and `T`: `det(A) = det(N) / (det(S)·det(T))`.
            let o = order(r.next());
            let num = Matrix::from_fn(n, n, |_, _| integer(&mut r, bytes, o));
            let s: Vec<Number> = (0..n).map(|_| integer(&mut r, 3, o).abs() + 1u8).collect();
            let t: Vec<Number> = (0..n).map(|_| integer(&mut r, 3, o).abs() + 1u8).collect();
            let a = Matrix::from_fn(n, n, |i, j| {
                Rational::new(num[(i, j)].clone(), &s[i] * &t[j])
            });
            let scale: Number = s.iter().chain(&t).product();
            assert_eq!(a.det(), Rational::new(num.det(), scale), "{n} {bytes}");
        }
    }
    // `det(A·B) = det(A)·det(B)`, larger.
    let n = size(20, 6);
    let a = random(&mut r, n, n, Kind::Fractions, ByteOrder::LittleEndian);
    let b = random(&mut r, n, n, Kind::Big(5), ByteOrder::BigEndian);
    assert_eq!((&a * &b).det(), &a.det() * &b.det());
}

#[test]
fn solutions() {
    let mut r = Rng(0x5EED_1603);
    let mut sizes: Vec<usize> = (0..size(9, 5)).collect();
    sizes.extend_from_slice(if cfg!(miri) { &[] } else { &[12, 17, 24] });
    for &n in &sizes {
        for k in [
            Kind::Small,
            Kind::Fractions,
            Kind::Big(8),
            Kind::Big(20),
            Kind::Integers(9),
        ] {
            if n > 12 && matches!(k, Kind::Big(20)) {
                continue;
            }
            let o = order(r.next());
            let a = random(&mut r, n, n, k, o);
            let cols = (r.next() % 4) as usize;
            let ob = order(r.next());
            let b = random(&mut r, n, cols, k, ob);
            let Some(x) = a.solve(&b) else {
                // A zero element can make a small random matrix singular.
                assert!(a.det().is_zero(), "{n} {k:?}");
                continue;
            };
            assert_eq!(x.shape(), (n, cols));
            check_all(&x, if n == 0 { ByteOrder::LittleEndian } else { o });
            assert_eq!(&a * &x, b, "{n} {k:?}");
        }
    }
}

#[test]
fn inverses() {
    let mut r = Rng(0x5EED_1604);
    let mut sizes: Vec<usize> = (0..size(9, 5)).collect();
    sizes.extend_from_slice(if cfg!(miri) { &[] } else { &[13, 20] });
    for &n in &sizes {
        for k in [Kind::Small, Kind::Fractions, Kind::Big(8), Kind::Big(17)] {
            if n > 12 && matches!(k, Kind::Big(17)) {
                continue;
            }
            let o = order(r.next());
            let a = random(&mut r, n, n, k, o);
            let Some(inv) = a.inverse() else {
                assert!(a.det().is_zero(), "{n} {k:?}");
                continue;
            };
            check_all(&inv, if n == 0 { ByteOrder::LittleEndian } else { o });
            assert_eq!(&a * &inv, Matrix::identity(n), "{n} {k:?}");
            assert_eq!(&inv * &a, Matrix::identity(n), "{n} {k:?}");
            assert_eq!(&a.det() * &inv.det(), Rational::from(1));
        }
    }
    // Known ones: the Hilbert matrix, whose inverse is of integers.
    let h = Matrix::from_fn(4, 4, |i, j| Rational::from(1) / (i + j + 1) as u32);
    let inv = h.inverse().unwrap();
    assert!(inv.as_slice().iter().all(Rational::is_integer));
    assert_eq!(inv[(0, 0)], 16);
    assert_eq!(inv[(3, 3)], 2800);
    assert_eq!(h.det().to_string(), "1/6048000");
    // A pivot that is zero, and a row swap.
    let p = Matrix::from_rows(&[[0, 2], [3, 0]]).map(|&x| Rational::from(x));
    assert_eq!(p.det(), -6);
    let inv = p.inverse().unwrap();
    assert_eq!(inv[(0, 1)].to_string(), "1/3");
    assert_eq!(inv[(1, 0)].to_string(), "1/2");
}

/// `A = B·C` with `B` of `m × rank` and `C` of `rank × n`, of that rank for random factors.
fn of_rank(r: &mut Rng, m: usize, n: usize, rank: usize, k: Kind) -> Matrix<Rational> {
    let o = order(r.next());
    &random(r, m, rank, k, o) * &random(r, rank, n, k, o)
}

#[test]
fn singular_matrices() {
    let mut r = Rng(0x5EED_1605);
    for n in 1..size(9, 5) {
        for k in [Kind::Small, Kind::Fractions, Kind::Big(9)] {
            let mut a = random(&mut r, n, n, k, ByteOrder::LittleEndian);
            // A row that is a multiple of another, or a zero column.
            if r.next() & 1 == 0 && n > 1 {
                let f = rational(&mut r, Kind::Fractions, None);
                for j in 0..n {
                    a[(n - 1, j)] = &a[(0, j)] * &f;
                }
            } else {
                let c = (r.next() % n as u64) as usize;
                for i in 0..n {
                    a[(i, c)] = Rational::default();
                }
            }
            assert!(a.det().is_zero(), "{n} {k:?}");
            assert_eq!(a.inverse(), None);
            let b = random(&mut r, n, 2, k, ByteOrder::LittleEndian);
            assert_eq!(a.solve(&b), None);
            assert!(a.rank() < n);
            let num = a.map(|x| x.numer().clone());
            if a.as_slice().iter().all(Rational::is_integer) {
                assert_eq!(num.solve(&b.map(|x| x.numer().clone())), None);
                assert_eq!(num.inverse_rational(), None);
            }
        }
    }
    let z = Matrix::<Rational>::zeros(3, 3);
    assert_eq!(z.rank(), 0);
    assert!(z.det().is_zero());
    assert_eq!(z.inverse(), None);
    assert_eq!(z.null_space(), Matrix::identity(3));
}

#[test]
fn ranks_and_null_spaces() {
    let mut r = Rng(0x5EED_1606);
    for m in 0..size(8, 4) {
        for n in 0..size(8, 4) {
            for rank in 0..=m.min(n) {
                for k in [Kind::Small, Kind::Fractions, Kind::Big(6)] {
                    let a = of_rank(&mut r, m, n, rank, k);
                    // The rank of the factors' product, unless a zero element made it less, as
                    // the elimination of integers finds it on the rows cleared of their
                    // denominators.
                    let cleared = Matrix::from_fn(m, n, |i, j| {
                        let l = a.row(i).fold(Number::from(1), |l, x| l.lcm(x.denom()));
                        (&a[(i, j)] * &l).numer().clone()
                    });
                    let rank = cleared.rank();
                    assert!(rank <= m.min(n));
                    assert_eq!(a.rank(), rank, "{m} {n} {rank} {k:?}");
                    let null = a.null_space();
                    assert_eq!(null.shape(), (n, n - rank), "{m} {n} {rank}");
                    assert!(
                        (&a * &null).as_slice().iter().all(Rational::is_zero),
                        "{m} {n} {rank} {k:?}"
                    );
                    // The basis of the reduced echelon form: each vector has a 1 where the
                    // others have 0s, so the vectors are independent.
                    assert_eq!(null.rank(), n - rank);
                    let o = a
                        .as_slice()
                        .first()
                        .map_or(ByteOrder::LittleEndian, |x| x.byte_order());
                    check_all(&null, o);
                    for c in 0..n - rank {
                        let one = (0..n)
                            .find(|&i| {
                                null[(i, c)] == 1
                                    && (0..n - rank).all(|d| d == c || null[(i, d)].is_zero())
                            })
                            .is_some();
                        assert!(one, "{m} {n} {rank}");
                    }
                    // Over the integers, the same space.
                    if a.as_slice().iter().all(Rational::is_integer) {
                        let num = a.map(|x| x.numer().clone());
                        assert_eq!(num.rank(), rank);
                        let q = num.null_space().map(|x| Rational::from(x));
                        assert_eq!((&null.transpose() * &null).rank(), n - rank);
                        let both = Matrix::from_fn(n, 2 * (n - rank), |i, j| {
                            if j < n - rank {
                                null[(i, j)].clone()
                            } else {
                                q[(i, j - (n - rank))].clone()
                            }
                        });
                        assert_eq!(both.rank(), n - rank);
                    }
                }
            }
        }
    }
    // Known: the plane `x + 2y + 3z = 0`.
    let a = Matrix::from_rows(&[[1, 2, 3]]).map(|&x| Rational::from(x) / 7u32);
    assert_eq!(
        a.null_space(),
        Matrix::from_rows(&[[-2, -3], [1, 0], [0, 1]]).map(|&x| Rational::from(x))
    );
}

#[test]
fn rational_solutions_of_integers() {
    let mut r = Rng(0x5EED_1607);
    for n in 0..size(10, 5) {
        for bytes in [1, 7, 8, 9, 33] {
            let o = order(r.next());
            let a = Matrix::from_fn(n, n, |i, j| {
                let oe = if i == 0 && j == 0 { o } else { order(r.next()) };
                integer(&mut r, bytes, oe)
            });
            if a.det().is_zero() {
                continue;
            }
            let b = Matrix::from_fn(n, 3, |_, _| {
                let oe = order(r.next());
                integer(&mut r, bytes, oe)
            });
            let x = a.solve(&b).unwrap();
            let out = if n == 0 { ByteOrder::LittleEndian } else { o };
            check_all(&x, out);
            let q = a.map(|x| Rational::from(x));
            assert_eq!(&q * &x, b.map(|x| Rational::from(x)), "{n} {bytes}");
            assert_eq!(x, q.solve(&b.map(|x| Rational::from(x))).unwrap());
            let inv = a.inverse_rational().unwrap();
            check_all(&inv, out);
            assert_eq!(inv, q.inverse().unwrap());
            // Against the adjugate and the determinant.
            let (adj, det) = a.inverse().unwrap();
            assert_eq!(inv, adj.map(|x| Rational::new(x.clone(), det.clone())));
        }
    }
}

#[test]
fn registers_and_numbers() {
    // Elements about `2^62`, whose products and minors pass the registers at once; and a matrix
    // whose minors grow past them step by step.
    let big = |x: i64| Rational::from(x);
    let a = Matrix::from_rows(&[
        [big(i64::MAX), big(i64::MIN + 1), big(3)],
        [big(i64::MIN + 1), big(i64::MAX - 1), big(-5)],
        [big(7), big(1 << 62), big(i64::MAX)],
    ]);
    assert_eq!(a.det(), leibniz(&a));
    let inv = a.inverse().unwrap();
    assert_eq!(&a * &inv, Matrix::identity(3));
    let m = Matrix::from_rows(&[[i64::MIN, 1], [1, i64::MIN]]).map(|&x| Rational::from(x));
    assert_eq!(m.det(), leibniz(&m));
    assert_eq!(&m * &m.inverse().unwrap(), Matrix::identity(2));
    let mut r = Rng(0x5EED_1608);
    let n = size(16, 6);
    let w = Matrix::from_fn(n, n, |_, _| {
        Rational::from((r.next() >> 34) as i64 - (1 << 29))
    });
    assert_eq!(&w * &w.inverse().unwrap(), Matrix::identity(n));
    // Denominators whose common multiple passes 63 bits.
    let d = Matrix::from_fn(4, 4, |i, j| {
        Rational::from(1) / ((r.next() >> 3) | 1) + Rational::from((i * j) as u32)
    });
    assert_eq!(d.det(), leibniz(&d));
    assert_eq!(&d * &d.inverse().unwrap(), Matrix::identity(4));
}

#[test]
fn long_elements() {
    // Minors of hundreds of limbs, whose exact quotients take the product by the inverse of the
    // divisor rather than Hensel's method, against Leibniz's formula and the defining identities.
    let mut r = Rng(0x5EED_1609);
    let (sizes, bytes): (&[usize], usize) = if cfg!(miri) {
        (&[3], 40)
    } else {
        (&[3, 4, 5], 700)
    };
    for &n in sizes {
        for o in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
            let a = Matrix::from_fn(n, n, |_, _| {
                let mut x = integer(&mut r, bytes, o);
                while x.is_zero() {
                    x = integer(&mut r, bytes, o);
                }
                Rational::from_integer(x)
            });
            let d = a.det();
            check(&d, o);
            assert_eq!(d, leibniz(&a), "{n}");
            let b = random(&mut r, n, 2, Kind::Big(bytes / 4), o);
            let x = a.solve(&b).unwrap();
            check_all(&x, o);
            assert_eq!(&a * &x, b);
            let inv = a.inverse().unwrap();
            assert_eq!(&a * &inv, Matrix::identity(n));
            // With fractions, whose rows' common denominators lengthen the integers further.
            let f = random(&mut r, n, n, Kind::Big(bytes / 8), o);
            if !f.det().is_zero() {
                assert_eq!(&f * &f.inverse().unwrap(), Matrix::identity(n));
            }
        }
    }
}

#[test]
fn integer_algorithms_in_byte_orders() {
    // `Matrix<Number>`'s determinant, rank, inverse and null space, which run on the same
    // elimination: results in the byte order of the top left element, against Laplace's
    // expansion (through Leibniz's formula), the adjugate and the rational null space.
    let mut r = Rng(0x5EED_160A);
    for n in 0..size(7, 4) {
        for bytes in [1, 8, 9, 20] {
            for o in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
                let a = Matrix::from_fn(n, n, |i, j| {
                    let oe = if i == 0 && j == 0 { o } else { order(r.next()) };
                    integer(&mut r, bytes, oe)
                });
                let out = if n == 0 { ByteOrder::LittleEndian } else { o };
                let d = a.det();
                assert_eq!(d.byte_order(), out);
                assert_eq!(Rational::from(&d), leibniz(&a.map(|x| Rational::from(x))));
                match a.inverse() {
                    Some((adj, det)) => {
                        assert_eq!(det, d);
                        assert!(adj.as_slice().iter().all(|x| x.byte_order() == out));
                        assert_eq!(adj, a.adjugate(), "{n} {bytes}");
                    }
                    None => assert!(d.is_zero()),
                }
            }
        }
    }
    for (m, n, rank) in [(3, 5, 2), (5, 3, 2), (4, 4, 3), (6, 6, 1), (1, 4, 1)] {
        for k in [Kind::Small, Kind::Integers(9)] {
            let a = of_rank(&mut r, m, n, rank, k).map(|x| x.numer().clone());
            let null = a.null_space();
            assert_eq!(null.cols(), n - a.rank());
            assert!((&a * &null).as_slice().iter().all(Number::is_zero));
            // The same vectors as the rational basis, scaled to integers.
            let q = a.map(|x| Rational::from(x)).null_space();
            for c in 0..null.cols() {
                let f = (0..n)
                    .find(|&i| {
                        q[(i, c)] == 1 && (0..null.cols()).all(|d| d == c || q[(i, d)].is_zero())
                    })
                    .unwrap();
                let s = Rational::from(&null[(f, c)]);
                assert!(s.is_positive());
                for i in 0..n {
                    assert_eq!(&q[(i, c)] * &s, Rational::from(&null[(i, c)]));
                }
            }
        }
    }
}

#[test]
fn products() {
    // The product on the rows and columns cleared of their denominators, against the sums of
    // products of rationals, for every shape up to 5 and kinds of elements; little-endian, as the
    // zeros it starts from.
    let mut r = Rng(0x5EED_160B);
    let top = size(6, 3);
    for m in 0..top {
        for k in 0..top {
            for n in 0..top {
                for kind in [
                    Kind::Small,
                    Kind::Fractions,
                    Kind::Big(9),
                    Kind::Integers(20),
                ] {
                    let o = order(r.next());
                    let a = random(&mut r, m, k, kind, o);
                    let b = random(&mut r, k, n, kind, o);
                    let c = &a * &b;
                    check_all(&c, ByteOrder::LittleEndian);
                    let naive = Matrix::from_fn(m, n, |i, j| {
                        (0..k).map(|p| &a[(i, p)] * &b[(p, j)]).sum::<Rational>()
                    });
                    assert_eq!(c, naive, "{m} {k} {n} {kind:?}");
                }
            }
        }
    }
}

#[test]
fn large_systems() {
    // From 24 rows, systems with few right-hand sides go by Dixon's p-adic lifting: against the
    // defining identity, and against Bareiss's elimination, which a right-hand side of more
    // columns takes (and whose first columns are the same system).
    let mut r = Rng(0x5EED_160C);
    let cases: &[(usize, Kind)] = if cfg!(miri) {
        &[(24, Kind::Small)]
    } else {
        &[
            (24, Kind::Small),
            (25, Kind::Fractions),
            (31, Kind::Integers(9)),
            (40, Kind::Small),
            (33, Kind::Big(3)),
            (48, Kind::Fractions),
        ]
    };
    for &(n, k) in cases {
        let o = order(r.next());
        let a = random(&mut r, n, n, k, o);
        let ob = order(r.next());
        let b = random(&mut r, n, n, k, ob);
        let Some(all) = a.solve(&b) else {
            assert!(a.det().is_zero());
            continue;
        };
        assert_eq!(&a * &all, b, "{n} {k:?}");
        for cols in [1, 2, n / 8] {
            let bc = Matrix::from_fn(n, cols, |i, j| b[(i, j)].clone());
            let x = a.solve(&bc).unwrap();
            check_all(&x, o);
            assert_eq!(
                x,
                Matrix::from_fn(n, cols, |i, j| all[(i, j)].clone()),
                "{n} {k:?}"
            );
        }
        // Over the integers.
        if matches!(k, Kind::Small | Kind::Integers(_)) {
            let ai = a.map(|x| x.numer().clone());
            let bi = Matrix::from_fn(n, 1, |i, _| b[(i, 0)].numer().clone());
            let x = ai.solve(&bi).unwrap();
            assert_eq!(x, Matrix::from_fn(n, 1, |i, _| all[(i, 0)].clone()));
        }
    }
    // A singular system of that size, and one that is singular modulo nothing small.
    let mut s = random(&mut r, 30, 30, Kind::Small, ByteOrder::LittleEndian);
    for j in 0..30 {
        s[(29, j)] = &s[(3, j)] * &Rational::from(-2);
    }
    let b = random(&mut r, 30, 1, Kind::Small, ByteOrder::LittleEndian);
    assert_eq!(s.solve(&b), None);
    // A right-hand side of zeros, and a diagonal system whose solution has large denominators.
    let id = Matrix::<Rational>::identity(26);
    assert_eq!(id.solve(&Matrix::zeros(26, 1)), Some(Matrix::zeros(26, 1)));
    let d = Matrix::from_fn(26, 26, |i, j| {
        if i == j {
            Rational::from((1u64 << 62) - 57 + 2 * i as u64)
        } else {
            Rational::default()
        }
    });
    let one = Matrix::from_fn(26, 1, |_, _| Rational::from(1));
    let x = d.solve(&one).unwrap();
    assert_eq!(&d * &x, one);
}

#[test]
fn large_determinants() {
    // From 22 to 28 rows the determinant is Dixon's divisor times residues: against products, the
    // integers' determinant, triangular matrices, and divisors far from the determinant, which
    // take many primes.
    let mut r = Rng(0x5EED_160D);
    let cases: &[(usize, Kind)] = if cfg!(miri) {
        &[(22, Kind::Fractions)]
    } else {
        &[
            (24, Kind::Small),
            (28, Kind::Small),
            (22, Kind::Fractions),
            (27, Kind::Fractions),
            (32, Kind::Integers(9)),
            (40, Kind::Big(2)),
        ]
    };
    for &(n, k) in cases {
        let o = order(r.next());
        let a = random(&mut r, n, n, k, o);
        let b = random(&mut r, n, n, Kind::Small, o);
        let (da, db) = (a.det(), b.det());
        check(&da, o);
        assert_eq!((&a * &b).det(), &da * &db, "{n} {k:?}");
        if matches!(k, Kind::Small | Kind::Integers(_)) {
            assert_eq!(a.map(|x| x.numer().clone()).det(), da.numer().clone());
        }
        // `L·U` with unit triangles but for the diagonal of `U`: the product of that diagonal.
        let l = Matrix::from_fn(n, n, |i, j| match i.cmp(&j) {
            core::cmp::Ordering::Greater => rational(&mut r, Kind::Fractions, None),
            core::cmp::Ordering::Equal => Rational::from(1),
            core::cmp::Ordering::Less => Rational::default(),
        });
        let u = Matrix::from_fn(n, n, |i, j| match i.cmp(&j) {
            core::cmp::Ordering::Less => rational(&mut r, k, None),
            core::cmp::Ordering::Equal => {
                Rational::from((i as i64 % 7) - 3) + 1u8 / Rational::from(5)
            }
            core::cmp::Ordering::Greater => Rational::default(),
        });
        let diag: Rational = (0..n).map(|i| &u[(i, i)]).product();
        assert_eq!((&l * &u).det(), diag, "{n} {k:?}");
    }
    // `2^70·I`: the divisor is `2^70`, and the rest, `2^(70(n - 1))`, takes about 30 primes.
    let n = size(30, 22);
    let big = Matrix::from_diagonal(&vec![Rational::from(1u128 << 70); n]);
    let mut e = big.clone();
    e[(0, n - 1)] = Rational::from(3);
    assert_eq!(
        e.det(),
        Rational::from(Number::from(1u128 << 70).pow(n as u32))
    );
    let n = size(30, 24);
    // Singular modulo the primes Dixon's lifting tries first, and singular.
    let primes = [
        0x3fff_ffff_ffff_ffc7u64,
        0x3fff_ffff_ffff_ffa9,
        0x3fff_ffff_ffff_ff8b,
        0x3fff_ffff_ffff_ff71,
    ];
    let mut d = Matrix::<Number>::identity(n);
    for (i, &p) in primes.iter().enumerate() {
        d[(i, i)] = Number::from(p);
    }
    d[(n - 1, 0)] = Number::from(5);
    let expected: Number = primes.iter().map(|&p| Number::from(p)).product();
    assert_eq!(d.det(), expected);
    let mut s = random(&mut r, n, n, Kind::Small, ByteOrder::LittleEndian);
    for j in 0..n {
        s[(n - 2, j)] = &s[(0, j)] + &s[(1, j)];
    }
    assert!(s.det().is_zero());
}

#[test]
fn generic_operations_on_rationals() {
    let q = |x: i32, y: u32| Rational::from(x) / y;
    let a = Matrix::from_rows(&[[q(1, 2), q(-1, 3)], [q(2, 1), q(3, 4)]]);
    let b = Matrix::from_rows(&[[q(1, 1), q(1, 6)], [q(0, 1), q(-5, 2)]]);
    assert_eq!(
        &a + &b,
        Matrix::from_rows(&[[q(3, 2), q(-1, 6)], [q(2, 1), q(-7, 4)]])
    );
    assert_eq!(&a - &a, Matrix::zeros(2, 2));
    assert_eq!(
        &a * &b,
        Matrix::from_rows(&[[q(1, 2), q(11, 12)], [q(2, 1), q(-37, 24)]])
    );
    assert_eq!(&a * &q(2, 1), q(2, 1) * &a);
    assert_eq!(-&a, &a * q(-1, 1));
    assert_eq!(a.transpose()[(0, 1)], q(2, 1));
    assert_eq!(a.trace(), q(5, 4));
    assert_eq!(a.pow(3), &(&a * &a) * &a);
    assert_eq!(a.pow(0), Matrix::identity(2));
    assert_eq!(a.hadamard(&b)[(1, 1)], q(-15, 8));
    let mut c = a.clone();
    c.add_scaled_row(1, 0, &q(-4, 1));
    assert_eq!(c[(1, 0)], 0);
    assert_eq!(c.det(), a.det());
    assert_eq!(a.to_string(), "1/2  -1/3\n  2   3/4");
    assert_eq!(&a / &q(1, 2), &a * q(2, 1));
    let mut d = a.clone();
    d /= q(-3, 1);
    assert_eq!(d, a.clone() / q(-3, 1));
    assert_eq!(d[(1, 1)], q(-1, 4));
}

#[test]
#[should_panic(expected = "not square")]
fn solve_of_a_rectangle() {
    let a = Matrix::<Rational>::zeros(2, 3);
    let _ = a.solve(&Matrix::zeros(2, 1));
}

#[test]
#[should_panic(expected = "not as many rows")]
fn solve_of_other_rows() {
    let a = Matrix::<Rational>::identity(2);
    let _ = a.solve(&Matrix::zeros(3, 1));
}

#[test]
#[should_panic(expected = "not as many rows")]
fn integer_solve_of_other_rows() {
    let a = Matrix::<Number>::identity(2);
    let _ = a.solve(&Matrix::zeros(3, 1));
}

#[test]
#[should_panic(expected = "not square")]
fn inverse_of_a_rectangle() {
    let _ = Matrix::<Rational>::zeros(3, 2).inverse();
}

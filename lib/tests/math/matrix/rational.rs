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

/// The determinant of the square matrix of integers `a` by a fraction-free elimination written
/// here on `Number`s, independent of the crate's eliminations: Bareiss's steps with the pivot of
/// the least magnitude in its column (the crate's take the first that is not zero), each exact
/// division by the previous pivot `Number::div_rem`'s, its remainder asserted to be zero.
fn bareiss(a: &Matrix<Number>) -> Number {
    let n = a.rows();
    if n == 0 {
        return Number::from(1);
    }
    let mut m: Vec<Vec<Number>> = (0..n)
        .map(|i| (0..n).map(|j| a[(i, j)].clone()).collect())
        .collect();
    let mut previous = Number::from(1);
    let mut negative = false;
    for k in 0..n {
        let Some(p) = (k..n)
            .filter(|&i| !m[i][k].is_zero())
            .min_by(|&i, &j| m[i][k].abs().cmp(&m[j][k].abs()))
        else {
            return Number::from(0);
        };
        if p != k {
            m.swap(p, k);
            negative = !negative;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let t = &(&m[i][j] * &m[k][k]) - &(&m[i][k] * &m[k][j]);
                let (q, rest) = t.div_rem(&previous);
                assert!(rest.is_zero(), "the oracle's division is not exact");
                m[i][j] = q;
            }
        }
        previous = m[k][k].clone();
    }
    let d = m[n - 1][n - 1].clone();
    if negative { -d } else { d }
}

/// The limbs that Hadamard's bound gives the minors of the matrix of integers `a`, as the
/// elimination on the stack reckons them (the bits of each row's largest element, then
/// `⌈n·L/2⌉ + 1` more for `L` the bits of `n`), repeated here to tell which of its widths a case
/// reaches: the elimination on the stack takes orders 5 to 16 with an element of at least 64
/// bits and a bound of at most 16 limbs, in arrays of 2, 4, 8 or 16 limbs.
fn stack_width(a: &Matrix<Number>) -> Option<usize> {
    let n = a.rows();
    if !(5..=16).contains(&n) || a.as_slice().iter().all(|x| x.bits() < 64) {
        return None;
    }
    let rows: u64 = (0..n)
        .map(|i| (0..n).map(|j| a[(i, j)].bits()).max().unwrap_or(0))
        .sum();
    let lg = u64::from(usize::BITS - n.leading_zeros());
    let w = (rows + (n as u64 * lg).div_ceil(2) + 1).div_ceil(64);
    [2, 4, 8, 16].into_iter().find(|&k| w <= k as u64)
}

/// Asserts the determinant of the matrix of integers `a` as `Matrix<Number>` and as
/// `Matrix<Rational>` of the same integers: `want`, in the byte order of the top left element,
/// a zero never negative.
#[track_caller]
fn integer_det_is(a: &Matrix<Number>, want: &Number, what: &str) {
    let o = a[(0, 0)].byte_order();
    let d = a.det();
    assert_eq!(&d, want, "Matrix<Number>: {what}");
    assert_eq!(d.byte_order(), o, "{what}");
    assert!(!(d.is_zero() && d.is_negative()), "a negative zero: {what}");
    let q = a.map(|x| Rational::from_integer(x.clone())).det();
    check(&q, o);
    assert_eq!(
        q,
        Rational::from_integer(want.clone()),
        "Matrix<Rational>: {what}"
    );
}

/// `L·D·U` for the diagonal `d` and unit triangles `L` and `U` whose elements below and above
/// the diagonal `entry` gives, then its rows permuted at random when `permute` is set: the matrix
/// and the sign of the permutation (`true` for odd). Unpermuted, its leading minors are the
/// products of the first elements of `d`, and while they are not zero so are the pivots of an
/// elimination that takes the first element not zero in each column.
fn ldu(
    r: &mut Rng,
    d: &[Number],
    o: ByteOrder,
    permute: bool,
    mut entry: impl FnMut(&mut Rng) -> Number,
) -> (Matrix<Number>, bool) {
    let n = d.len();
    let zero = Number::new_from_i64(0, o);
    let l = Matrix::from_fn(n, n, |i, j| match i.cmp(&j) {
        core::cmp::Ordering::Greater => entry(r),
        core::cmp::Ordering::Equal => Number::new_from_i64(1, o),
        core::cmp::Ordering::Less => zero.clone(),
    });
    let u = Matrix::from_fn(n, n, |i, j| match i.cmp(&j) {
        core::cmp::Ordering::Less => entry(r),
        core::cmp::Ordering::Equal => d[i].clone(),
        core::cmp::Ordering::Greater => zero.clone(),
    });
    let a = &l * &u;
    let mut rows: Vec<usize> = (0..n).collect();
    let mut negative = false;
    for i in (1..n).rev().filter(|_| permute) {
        let j = (r.next() % (i as u64 + 1)) as usize;
        if j != i {
            rows.swap(i, j);
            negative = !negative;
        }
    }
    // The top left element in the byte order `o`, the others in either.
    let a = Matrix::from_fn(n, n, |i, j| {
        let oe = if i == 0 && j == 0 { o } else { order(r.next()) };
        in_byte_order(&a[(rows[i], j)], oe)
    });
    (a, negative)
}

/// `x` in the byte order `o`.
fn in_byte_order(x: &Number, o: ByteOrder) -> Number {
    let mut v = x.value.clone();
    if x.byte_order() != o {
        v.reverse();
    }
    Number::new_signed(v, x.is_negative(), o)
}

#[test]
fn determinants_with_a_previous_pivot_of_minus_one() {
    // RATIONAL-1 of the v0.2 speed pass's audit: the elimination on the stack (orders 5 to 16 with
    // an element of 64 bits or more) once divided by a previous pivot of -1 as by 1, and gave the
    // determinant's negation. `diag(-1, 1, …, 1, 2^63)`, -1 at each place of the diagonal with a
    // long element at each other, and a dense example, against the products of the diagonals and
    // SymPy's `Matrix.det()`, in both byte orders, as integers and as rationals.
    let longs = [
        Number::from(1u64 << 63),
        Number::from(u64::MAX),
        Number::from(1u128 << 64),
        Number::from((1u128 << 64) + 1),
        Number::from(u128::MAX),
    ];
    let orders: Vec<usize> = if cfg!(miri) {
        vec![5]
    } else {
        (5..=16).collect()
    };
    for o in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let num = |x: &Number| in_byte_order(x, o);
        for &n in &orders {
            // The auditor's reproducer: diag(-1, 1, …, 1, 2^63) is -2^63.
            let a = Matrix::from_fn(n, n, |i, j| {
                num(&if i != j {
                    Number::from(0)
                } else if i == 0 {
                    Number::from(-1)
                } else if i == n - 1 {
                    Number::from(1u64 << 63)
                } else {
                    Number::from(1)
                })
            });
            integer_det_is(
                &a,
                &-Number::from(1u64 << 63),
                &format!("diag(-1, 1, …, 2^63), {n}"),
            );
            for minus in 0..n {
                for long in 0..n {
                    if long == minus || (cfg!(miri) && long != n - 1) {
                        continue;
                    }
                    for (k, big) in longs.iter().enumerate() {
                        let negative_big = (minus + long + k) % 2 == 1;
                        let diagonal = |i: usize| {
                            if i == minus {
                                Number::from(-1)
                            } else if i == long {
                                if negative_big {
                                    -big.clone()
                                } else {
                                    big.clone()
                                }
                            } else {
                                Number::from(1)
                            }
                        };
                        let a = Matrix::from_fn(n, n, |i, j| {
                            num(&if i == j { diagonal(i) } else { Number::from(0) })
                        });
                        let want = (0..n).fold(Number::from(1), |p, i| &p * &diagonal(i));
                        integer_det_is(&a, &want, &format!("{n}: -1 at {minus}, {big} at {long}"));
                    }
                }
            }
        }
        // A dense one, of SymPy's determinant.
        let rows: [[i128; 5]; 5] = [
            [-1, 2, 3, 4, 5],
            [6, -7, 8, 9, 10],
            [11, 12, 13, 14, 1 << 70],
            [16, 3, 18, -19, 20],
            [21, 22, 5, 24, 26],
        ];
        let a = Matrix::from_fn(5, 5, |i, j| Number::new_from_i128(rows[i][j], o));
        let want: Number = "75453971663291191222786356".parse().unwrap();
        integer_det_is(&a, &want, "the dense 5 × 5");
        assert_eq!(bareiss(&a), want);
    }
}

#[test]
fn determinants_of_signs_with_one_long_element() {
    // Orders 5 to 16 of elements in {-1, 0, 1} with one of about `2^63` or `2^64`, of either sign,
    // at a random place: the class in which RATIONAL-1 gave about two in five determinants
    // negated. The integers and the rationals against the fraction-free elimination of this file
    // on numbers, which is checked against Leibniz's formula first; and fractions of such
    // numerators over denominators of 1 to 3, against the elimination of their rows cleared here.
    let mut r = Rng(0x5EED_1612);
    for _ in 0..size(40, 2) {
        let n = 1 + (r.next() % 5) as usize;
        let a = Matrix::from_fn(n, n, |_, _| Number::from((r.next() % 3) as i64 - 1));
        let q = a.map(|x| Rational::from_integer(x.clone()));
        assert_eq!(Rational::from_integer(bareiss(&a)), leibniz(&q));
    }
    let orders: Vec<usize> = if cfg!(miri) {
        vec![5]
    } else {
        (5..=16).collect()
    };
    let (mut cases, mut nonsingular) = (0, 0);
    for &n in &orders {
        for t in 0..size(60, 2) {
            let o = order(r.next());
            let long = |r: &mut Rng| {
                let x = match r.next() % 4 {
                    0 => (1u128 << 63) + u128::from(r.next() % 1000),
                    1 => (1u128 << 63) - 1 - u128::from(r.next() % 1000),
                    2 => (1u128 << 64) - 1 - u128::from(r.next() % 1000),
                    _ => (1u128 << 64) + u128::from(r.next() % 1000),
                };
                let x = Number::new_from_u128(x, o);
                if r.next() & 1 == 1 { -x } else { x }
            };
            let at = (r.next() % (n * n) as u64) as usize;
            // Half of them with a top left element of ±1, a first pivot of ±1.
            let corner = t % 2 == 0;
            let mut a = Matrix::from_fn(n, n, |i, j| {
                if i * n + j == at && (at != 0 || !corner) {
                    long(&mut r)
                } else if i == 0 && j == 0 && corner {
                    Number::new_from_i64(if r.next() & 1 == 1 { -1 } else { 1 }, o)
                } else {
                    Number::new_from_i64((r.next() % 3) as i64 - 1, o)
                }
            });
            while a.as_slice().iter().all(|x| x.bits() < 64) {
                a[(n - 1, n - 1)] = long(&mut r);
            }
            assert!(stack_width(&a).is_some());
            let want = bareiss(&a);
            cases += 1;
            nonsingular += usize::from(!want.is_zero());
            integer_det_is(&a, &want, &format!("{n}, case {t}"));
            if n <= 8 || t % 4 == 0 {
                // Fractions: each element over 1, 2 or 3, against the rows cleared by the
                // products of their denominators.
                let den: Vec<u64> = (0..n * n).map(|_| 1 + r.next() % 3).collect();
                let f = Matrix::from_fn(n, n, |i, j| {
                    Rational::new(a[(i, j)].clone(), Number::new_from_u64(den[i * n + j], o))
                });
                let scale: Vec<Number> = (0..n)
                    .map(|i| Number::from(den[i * n..(i + 1) * n].iter().product::<u64>()))
                    .collect();
                let cleared = Matrix::from_fn(n, n, |i, j| {
                    let x = &f[(i, j)] * &Rational::from_integer(scale[i].clone());
                    assert!(x.is_integer());
                    x.numer().clone()
                });
                let total = scale.iter().fold(Number::from(1), |p, s| &p * s);
                let d = f.det();
                check(&d, o);
                assert_eq!(
                    d,
                    Rational::new(bareiss(&cleared), total),
                    "fractions, {n}, {t}"
                );
            }
        }
    }
    // Most of them are not singular, so that the sign is tested.
    assert!(
        cfg!(miri) || 2 * nonsingular > cases,
        "{nonsingular} of {cases}"
    );
}

#[test]
fn determinants_on_the_stack_by_their_divisors() {
    // The paths of the elimination on the stack's divisions, each previous pivot a leading minor
    // of `L·D·U`, the product of the first elements of `D`, or permuted: pivots of ±1 (the
    // quotient the dividend), powers of two of fewer than 64 bits and of whole limbs (only a
    // shift), odd ones of one and two limbs and beyond (Hensel's division with and without the
    // shift), with `L` and `U` of elements in {-1, 0, 1}, whose cancellations make zeros and
    // zero quotients, or of random integers; a zero in `D` makes the matrix singular. Against
    // the product of `D` with the permutation's sign and the elimination of this file on
    // numbers; each width of the stack's elements, 2, 4, 8 and 16 limbs, and past them, reached.
    let mut r = Rng(0x5EED_1613);
    let divisors: Vec<Number> = [
        1u128,
        2,
        3,
        1 << 32,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        u64::MAX as u128,
        1 << 64,
        (1 << 64) + 1,
        1 << 65,
        3 << 64,
        u128::MAX,
    ]
    .iter()
    .map(|&x| Number::from(x))
    .chain([Number::from(1u8) << 128, (Number::from(1u8) << 192) + 1u8])
    .collect();
    let orders: Vec<usize> = if cfg!(miri) {
        vec![5]
    } else {
        (5..=16).collect()
    };
    let mut widths = [0usize; 5];
    for &n in &orders {
        for t in 0..size(40, 3) {
            let o = order(r.next());
            // Mostly ±1 on the diagonal, with one long element at least.
            let mut d: Vec<Number> = (0..n)
                .map(|_| {
                    let x = if r.next().is_multiple_of(3) {
                        divisors[(r.next() % divisors.len() as u64) as usize].clone()
                    } else {
                        Number::from(1)
                    };
                    if r.next() & 1 == 1 { -x } else { x }
                })
                .collect();
            let at = (r.next() % n as u64) as usize;
            if d.iter().all(|x| x.bits() < 64) {
                d[at] = divisors[5 + (r.next() % (divisors.len() as u64 - 5)) as usize].clone();
            }
            if t % 7 == 3 {
                d[(at + 1) % n] = Number::from(0);
            }
            let small = t % 3 != 2;
            let (a, negative) = ldu(&mut r, &d, o, t % 2 == 0, |r| {
                if small {
                    Number::from((r.next() % 3) as i64 - 1)
                } else {
                    Number::from(r.next() as i64 >> (r.next() % 64))
                }
            });
            let product = d.iter().fold(Number::from(1), |p, x| &p * x);
            let want = if negative { -product } else { product };
            assert_eq!(bareiss(&a), want, "the oracle, {n}, case {t}");
            let k = match stack_width(&a) {
                Some(2) => 0,
                Some(4) => 1,
                Some(8) => 2,
                Some(16) => 3,
                _ => 4,
            };
            widths[k] += 1;
            integer_det_is(&a, &want, &format!("{n}, case {t}, D = {d:?}"));
        }
    }
    if !cfg!(miri) {
        assert!(widths.iter().all(|&k| k > 0), "{widths:?}");
    }
}

#[test]
fn determinants_on_the_stack() {
    // Orders 5 to 16 of integers of 1 to 40 bytes, whose minors fit in 2 to 16 limbs and past:
    // `P·L·D·U` with unit triangles of random integers, `D` diagonal, a third of its elements -1
    // and a sixth 1, and `P` a permutation, against the product of `D` and the permutation's sign;
    // and random integers against the fraction-free elimination of this file on numbers, which
    // shares no code with the crate's eliminations.
    let mut r = Rng(0x5EED_1611);
    let orders: &[usize] = if cfg!(miri) {
        &[5]
    } else {
        &[5, 6, 7, 8, 11, 16]
    };
    for &n in orders {
        for bytes in [1usize, 4, 8, 9, 16, 24, 40] {
            if cfg!(miri) && bytes > 9 {
                continue;
            }
            let o = order(r.next());
            let small = |r: &mut Rng| match r.next() % 6 {
                0 | 1 => Number::new_from_i64(-1, o),
                2 => Number::new_from_i64(1, o),
                _ => {
                    let x = integer(r, bytes.min(3), o);
                    if x.is_zero() {
                        Number::new_from_i64(-1, o)
                    } else {
                        x
                    }
                }
            };
            let l = Matrix::from_fn(n, n, |i, j| match i.cmp(&j) {
                core::cmp::Ordering::Greater => integer(&mut r, bytes / 2 + 1, o),
                core::cmp::Ordering::Equal => Number::new_from_i64(1, o),
                core::cmp::Ordering::Less => Number::new_from_i64(0, o),
            });
            let d: Vec<Number> = (0..n).map(|_| small(&mut r)).collect();
            let u = Matrix::from_fn(n, n, |i, j| match i.cmp(&j) {
                core::cmp::Ordering::Less => integer(&mut r, bytes / 2 + 1, o),
                core::cmp::Ordering::Equal => d[i].clone(),
                core::cmp::Ordering::Greater => Number::new_from_i64(0, o),
            });
            let mut a = &l * &u;
            // A permutation of the rows, and its sign.
            let mut negative = false;
            for i in (1..n).rev() {
                let j = (r.next() % (i as u64 + 1)) as usize;
                if j != i {
                    a = Matrix::from_fn(n, n, |x, y| {
                        let x = if x == i {
                            j
                        } else if x == j {
                            i
                        } else {
                            x
                        };
                        a[(x, y)].clone()
                    });
                    negative = !negative;
                }
            }
            let product = d.iter().fold(Number::new_from_i64(1, o), |p, x| &p * x);
            let want = if negative { -product } else { product };
            integer_det_is(&a, &want, &format!("{n} {bytes}"));
            // And with random integers, against the elimination of this file on numbers.
            let b = Matrix::from_fn(n, n, |_, _| integer(&mut r, bytes, o));
            integer_det_is(&b, &bareiss(&b), &format!("{n} {bytes} random"));
        }
    }
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

/// The hash of `x` under the standard hasher.
fn hash_of(x: &Rational) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

/// Asserts the form of `x` in the byte order `o`, and that it is in the one form of its value: equal
/// to, and hashing as, the rational made from its parts (a value left in numbers when its parts
/// fit in words, or the other way about, would not be).
#[track_caller]
fn one_form(x: &Rational, o: ByteOrder) {
    check(x, o);
    let y = Rational::new(x.numer().clone(), x.denom().clone());
    assert_eq!(*x, y, "{x:?}");
    assert_eq!(hash_of(x), hash_of(&y), "{x:?}");
}

/// [`one_form`] for every element of `m`.
#[track_caller]
fn one_form_all(m: &Matrix<Rational>, o: ByteOrder) {
    for x in m.as_slice() {
        one_form(x, o);
    }
}

/// A random integer of exactly `bits` bits, of either sign, or zero now and then.
fn bits(r: &mut Rng, bits: u32) -> i128 {
    if r.next().is_multiple_of(16) {
        return 0;
    }
    let m = (u128::from(r.next()) << 64 | u128::from(r.next())) >> (128 - bits);
    let m = (m | 1 << (bits - 1)) as i128;
    if r.next() & 1 == 1 { -m } else { m }
}

/// The number `x` in the byte order `o`.
fn wide(x: i128, o: ByteOrder) -> Number {
    Number::new_from_i128(x, o)
}

#[test]
fn small_orders_in_registers_and_beyond() {
    // From 1 × 1 to 8 × 8, the orders of the register paths, with integers and fractions whose
    // cleared rows keep the minors in registers, reach Hadamard's bound of `2^61` or pass it, or
    // whose parts pass the words: every result against Leibniz's formula or the defining
    // identities, in the byte order of the top left element and in its one form.
    let mut r = Rng(0x5EED_2701);
    let top = size(9, 4);
    for n in 1..top {
        for b in [3u32, 12, 20, 28, 31, 40, 62, 63, 64, 65] {
            if cfg!(miri) && b > 31 && n > 2 {
                continue;
            }
            for fractions in [false, true] {
                let o = order(r.next());
                let a = Matrix::from_fn(n, n, |i, j| {
                    let oe = if i == 0 && j == 0 { o } else { order(r.next()) };
                    let p = wide(bits(&mut r, b), oe);
                    if fractions {
                        let q = wide(bits(&mut r, b.min(64)).abs().max(1), oe);
                        Rational::new(p, q)
                    } else {
                        Rational::from_integer(p)
                    }
                });
                let d = a.det();
                one_form(&d, o);
                if n <= 5 {
                    assert_eq!(d, leibniz(&a), "{n} {b} {fractions}");
                }
                let bm = Matrix::from_fn(n, 2, |_, _| {
                    Rational::from_integer(wide(bits(&mut r, b), o))
                });
                match a.solve(&bm) {
                    Some(x) => {
                        one_form_all(&x, o);
                        assert_eq!(&a * &x, bm, "{n} {b} {fractions}");
                    }
                    None => assert!(d.is_zero()),
                }
                match a.inverse() {
                    Some(inv) => {
                        one_form_all(&inv, o);
                        assert_eq!(&a * &inv, Matrix::identity(n), "{n} {b} {fractions}");
                        assert_eq!(&d * &inv.det(), Rational::from(1));
                    }
                    None => assert!(d.is_zero()),
                }
                if fractions {
                    continue;
                }
                // The same integers as a `Matrix<Number>`.
                let z = a.map(|x| x.numer().clone());
                let dz = z.det();
                assert_eq!(dz.byte_order(), o);
                assert_eq!(Rational::from(&dz), d);
                match z.inverse() {
                    Some((adj, det)) => {
                        assert_eq!(det, dz);
                        assert_eq!(adj, z.adjugate(), "{n} {b}");
                        assert!(adj.as_slice().iter().all(|x| x.byte_order() == o));
                    }
                    None => assert!(dz.is_zero()),
                }
                let zb = bm.map(|x| x.numer().clone());
                match (z.solve(&zb), z.inverse_rational()) {
                    (Some(x), Some(inv)) => {
                        one_form_all(&x, o);
                        one_form_all(&inv, o);
                        assert_eq!(&a * &x, bm);
                        assert_eq!(&a * &inv, Matrix::identity(n));
                    }
                    (x, inv) => assert!(x.is_none() && inv.is_none() && dz.is_zero()),
                }
            }
        }
    }
}

#[test]
fn results_that_shrink_and_grow_across_the_words() {
    // Long elements whose results are short: `U·2^70` for a unimodular `U`, whose inverse is
    // `U⁻¹·2^-70`, and whose rows clear to the same integers; and short elements whose results
    // are long: a determinant of 8 × 8 integers of 30 bits passes 64 bits.
    let mut r = Rng(0x5EED_2702);
    let u = Matrix::from_rows(&[[2, 3, 1], [1, 2, 1], [1, 1, 1]]).map(|&x| Rational::from(x));
    assert_eq!(u.det(), 1);
    let big = Rational::from_integer(Number::from(1u8) << 70usize);
    for o in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let s = u.map(|x| {
            let y = x * &big;
            Rational::new(wide(0, o) + y.numer(), wide(1, o))
        });
        let inv = s.inverse().unwrap();
        one_form_all(&inv, o);
        assert_eq!(
            &inv * &big,
            u.inverse()
                .unwrap()
                .map(|x| Rational::new(wide(0, o) + x.numer(), wide(1, o)))
        );
        // Its rows over 2^70 again: short fractions.
        let t = s.map(|x| x / &big);
        assert_eq!(t, u);
        assert_eq!(t.det(), 1);
    }
    let n = size(8, 4);
    let a = Matrix::from_fn(n, n, |_, _| {
        Rational::from_integer(wide(bits(&mut r, 30), ByteOrder::LittleEndian))
    });
    let d = a.det();
    one_form(&d, ByteOrder::LittleEndian);
    if !d.is_zero() {
        let inv = a.inverse().unwrap();
        one_form_all(&inv, ByteOrder::LittleEndian);
        assert_eq!(&a * &inv, Matrix::identity(n));
    }
}

#[test]
fn products_in_registers_and_beyond() {
    // The product of matrices in words, of every shape up to 9 (past the stack's 64 elements),
    // with integers whose sums of products fit in 64 bits, in 128, or pass them, and parts at the
    // boundary of the words: against the sums of products of rationals, in the one form.
    let mut r = Rng(0x5EED_2703);
    let top = size(10, 4);
    for m in 1..top {
        for k in 1..top {
            for n in [1, m, top - 1] {
                for b in [7u32, 31, 40, 60, 63, 64, 65] {
                    if cfg!(miri) && b > 31 {
                        continue;
                    }
                    let fractions = r.next() & 1 == 1;
                    let el = |r: &mut Rng| {
                        let o = order(r.next());
                        let p = wide(bits(r, b), o);
                        if fractions {
                            Rational::new(p, wide(bits(r, b.min(64)).abs().max(1), o))
                        } else {
                            Rational::from_integer(p)
                        }
                    };
                    let a = Matrix::from_fn(m, k, |_, _| el(&mut r));
                    let c = Matrix::from_fn(k, n, |_, _| el(&mut r));
                    let p = &a * &c;
                    one_form_all(&p, ByteOrder::LittleEndian);
                    let naive = Matrix::from_fn(m, n, |i, j| {
                        (0..k).map(|q| &a[(i, q)] * &c[(q, j)]).sum::<Rational>()
                    });
                    assert_eq!(p, naive, "{m} {k} {n} {b} {fractions}");
                }
            }
        }
    }
}

#[test]
fn two_by_two_beyond_the_registers() {
    // Order 2 with parts of 63 to 65 bits, which take the rationals' arithmetic for the
    // determinant and the inverse: against the formulas on numbers, in the byte order of the top
    // left element.
    let mut r = Rng(0x5EED_2704);
    for _ in 0..size(200, 10) {
        let o = order(r.next());
        let a = Matrix::from_fn(2, 2, |i, j| {
            let oe = if i == 0 && j == 0 { o } else { order(r.next()) };
            let b = 63 + (r.next() % 3) as u32;
            Rational::new(
                wide(bits(&mut r, b), oe),
                wide(bits(&mut r, b).abs().max(1), oe),
            )
        });
        let d = a.det();
        one_form(&d, o);
        assert_eq!(d, &(&a[(0, 0)] * &a[(1, 1)]) - &(&a[(0, 1)] * &a[(1, 0)]));
        let b = Matrix::from_fn(2, 3, |_, _| {
            Rational::new(wide(bits(&mut r, 64), o), wide(1, o))
        });
        match (a.inverse(), a.solve(&b)) {
            (Some(inv), Some(x)) => {
                one_form_all(&inv, o);
                one_form_all(&x, o);
                assert_eq!(&a * &inv, Matrix::identity(2));
                assert_eq!(&a * &x, b);
            }
            (inv, x) => assert!(inv.is_none() && x.is_none() && d.is_zero()),
        }
    }
}

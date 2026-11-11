use a::math::numbers::{ByteOrder, Factorization, Number};

fn n(x: i64) -> Number {
    Number::new_from_i64(x, ByteOrder::BigEndian)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The factorisation of `m >= 1` by trial division: the tests' own, quicker under Miri than
/// [`Number::factor`].
fn factorise(mut m: u64) -> Factorization {
    let mut f = Vec::new();
    let mut p = 2;
    while p * p <= m {
        let mut e = 0;
        while m.is_multiple_of(p) {
            m /= p;
            e += 1;
        }
        if e > 0 {
            f.push((Number::new_from_u64(p, ByteOrder::BigEndian), e));
        }
        p += 1;
    }
    if m > 1 {
        f.push((Number::new_from_u64(m, ByteOrder::BigEndian), 1));
    }
    Factorization::new(f, ByteOrder::BigEndian)
}

/// The symbol by its definition: the product of the Legendre symbols of the factors of `m`.
fn reference(a: i64, m: i64) -> i32 {
    if gcd(a.rem_euclid(m), m) != 1 {
        return 0;
    }
    let (mut m, mut r, mut f) = (m, 1, 3);
    while m > 1 {
        while f * f <= m && m % f != 0 {
            f += 2;
        }
        let q = if f * f > m { m } else { f };
        let square = (1..q).any(|x| x * x % q == a.rem_euclid(q));
        if !square {
            r = -r;
        }
        m /= q;
    }
    r
}

#[test]
fn jacobi_against_its_definition() {
    let limit = if cfg!(miri) { 25 } else { 180 };
    for m in (1..limit).step_by(2) {
        for a in -30..(if cfg!(miri) { 10 } else { 70 }) {
            let want = if m == 1 { 1 } else { reference(a, m) };
            assert_eq!(n(a).jacobi(&n(m)), want, "({a} / {m})");
        }
    }
}

#[test]
fn jacobi_of_long_numbers() {
    // The binary method on limbs, with the multiplicativity of the symbol as its check.
    let m = (n(1) << 521usize) - 1u8;
    let (a, b) = ((n(3) << 300usize) + 7u8, (n(5) << 200usize) + 11u8);
    assert_eq!((&a * &b).jacobi(&m), a.jacobi(&m) * b.jacobi(&m));
    // Reciprocity: 2^521 - 1 is 3 modulo 4.
    let p = (n(1) << 127usize) - 1u8;
    assert_eq!(p.jacobi(&m), -m.jacobi(&p));
    assert_eq!((&m * 3u8).jacobi(&m), 0);
    assert_eq!(m.jacobi(&m), 0);
    assert_eq!(n(1).jacobi(&m), 1);
    assert_eq!(n(2).jacobi(&m), 1);
    assert_eq!(n(-1).jacobi(&m), -1);
    assert_eq!(
        Number::new_from_u8(2, ByteOrder::LittleEndian).jacobi(&m),
        1
    );
    // A common factor found at the end of a long run.
    let f = (n(1) << 89usize) - 1u8;
    assert_eq!((&f * 3u8).jacobi(&(&f * &p)), 0);
}

#[test]
#[should_panic(expected = "the modulus must be odd and positive")]
fn jacobi_refuses_an_even_modulus() {
    n(3).jacobi(&n(10));
}

#[test]
#[should_panic(expected = "the modulus must be odd and positive")]
fn jacobi_refuses_a_negative_modulus() {
    n(3).jacobi(&n(-7));
}

#[test]
#[should_panic(expected = "the modulus must be odd and positive")]
fn jacobi_refuses_zero() {
    n(3).jacobi(&n(0));
}

/// The same number in the other byte order, with zero bytes above its top: a non-canonical form.
fn other(x: &Number) -> Number {
    let mut v = x.value.clone();
    v.reverse();
    let order = match x.byte_order {
        ByteOrder::BigEndian => ByteOrder::LittleEndian,
        ByteOrder::LittleEndian => ByteOrder::BigEndian,
    };
    match order {
        ByteOrder::BigEndian => v.insert(0, 0),
        ByteOrder::LittleEndian => v.push(0),
    }
    Number {
        value: v,
        negative: x.negative,
        byte_order: order,
    }
}

/// The value of a small number.
fn int(x: &Number) -> i128 {
    i128::try_from(x).unwrap()
}

/// The Kronecker symbol by its definition: the product of `(a / p)` over the primes of `|n|`,
/// each the Legendre symbol or the rule for 2, and `(a / -1)`.
fn kronecker_reference(a: i64, n: i64) -> i32 {
    if n == 0 {
        return i32::from(a == 1 || a == -1);
    }
    let mut k = 1;
    let mut m = n;
    if m < 0 {
        m = -m;
        if a < 0 {
            k = -k;
        }
    }
    let mut p = 2;
    while m > 1 {
        if p * p > m {
            p = m;
        }
        while m % p == 0 {
            m /= p;
            let s = if p == 2 {
                match a.rem_euclid(8) {
                    1 | 7 => 1,
                    3 | 5 => -1,
                    _ => 0,
                }
            } else {
                let r = a.rem_euclid(p);
                if r == 0 {
                    0
                } else if (1..p).any(|x| x * x % p == r) {
                    1
                } else {
                    -1
                }
            };
            k *= s;
        }
        p += 1;
    }
    k
}

#[test]
fn kronecker_against_its_definition() {
    let (an, nn) = if cfg!(miri) { (8, 12) } else { (60, 90) };
    for n_ in -nn..=nn {
        for a_ in -an..=an {
            let want = kronecker_reference(a_, n_);
            for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
                let a = Number::new_from_i64(a_, order);
                let m = Number::new_from_i64(n_, ByteOrder::BigEndian);
                assert_eq!(a.kronecker(&m), want, "({a_} / {n_})");
                assert_eq!(
                    other(&a).kronecker(&other(&m)),
                    want,
                    "({a_} / {n_}) non-canonical"
                );
            }
        }
    }
}

#[test]
fn kronecker_of_long_numbers() {
    // Multiplicative in the bottom argument, and Jacobi's for odd positive ones.
    let m = (n(1) << 521usize) - 1u8;
    let a = (n(3) << 300usize) + 7u8;
    for v in [0usize, 1, 2, 3, 7, 64, 65] {
        let even = &m << v;
        let want = a.kronecker(&(n(1) << v)) * a.jacobi(&m);
        assert_eq!(a.kronecker(&even), want, "times 2^{v}");
        assert_eq!(a.kronecker(&-&even), want, "negative, times 2^{v}");
        assert_eq!((-&a).kronecker(&-&even), -(-&a).kronecker(&even));
    }
    // (a / 2) by a modulo 8, for a long `a`.
    let big = n(1) << 200usize;
    for r in 0..16 {
        let x = &big + r as u8;
        let want = match r % 8 {
            1 | 7 => 1,
            3 | 5 => -1,
            _ => 0,
        };
        assert_eq!(x.kronecker(&n(2)), want);
        assert_eq!(x.kronecker(&n(-2)), want);
        assert_eq!(
            x.kronecker(&(n(1) << 100usize)),
            if r % 2 == 0 { 0 } else { 1 }
        );
    }
    // Zero at the bottom: 1 only for ±1.
    assert_eq!(big.kronecker(&n(0)), 0);
    assert_eq!(n(-1).kronecker(&n(0)), 1);
    assert_eq!((&big - &big + 1u8).kronecker(&(&big - &big)), 1);
    // Long bottom, word top.
    assert_eq!(n(2).kronecker(&(&m * 4u8)), 0);
    assert_eq!(
        n(3).kronecker(&(&m * 2u8)),
        n(3).kronecker(&n(2)) * n(3).jacobi(&m)
    );
}

/// Every root of `a` modulo `m`, by trying them all.
fn roots_reference(a: i64, m: i64) -> Vec<i64> {
    let a = a.rem_euclid(m);
    (0..m).filter(|x| x * x % m == a).collect()
}

#[test]
fn sqrt_mod_against_every_root() {
    let limit = if cfg!(miri) { 12 } else { 400 };
    for m_ in 1..limit {
        let m = n(m_);
        for a_ in 0..m_ {
            let want = roots_reference(a_, m_);
            let a = n(a_);
            let got: Vec<i64> = a.sqrt_mod_all(&m).iter().map(|r| int(r) as i64).collect();
            assert_eq!(got, want, "roots of {a_} modulo {m_}");
            assert_eq!(
                a.sqrt_mod(&m).map(|r| int(&r) as i64),
                want.first().copied(),
                "{a_} modulo {m_}"
            );
        }
    }
}

#[test]
fn sqrt_mod_of_small_primes() {
    // 3 modulo 4, 5 modulo 8, and 1 modulo 8 with `s` up to 8 (257).
    let primes: &[i64] = if cfg!(miri) {
        &[3, 13, 17]
    } else {
        &[3, 5, 7, 11, 13, 17, 41, 97, 101, 193, 257, 769]
    };
    for &p in primes {
        for a in 0..p {
            let squares: Vec<i64> = (0..p).filter(|x| x * x % p == a).collect();
            let got = n(a).sqrt_mod(&n(p));
            if squares.is_empty() {
                assert!(got.is_none(), "{a} modulo {p}");
            } else {
                assert_eq!(got, Some(n(squares[0])), "{a} modulo {p}");
            }
        }
    }
}

#[test]
fn sqrt_mod_of_long_primes() {
    let check = |p: &Number| {
        let f = Factorization::new(vec![(p.clone(), 1)], ByteOrder::LittleEndian);
        // Under Miri, the factorisation's form alone for a long prime: the primality test of the
        // factorisation is too slow there.
        let long = cfg!(miri) && p.bits() > 64;
        let xs = if cfg!(miri) {
            vec![n(3)]
        } else {
            vec![n(2), n(3), n(1234567), (p - 5u8) >> 1usize]
        };
        for x in xs {
            let s = (&x * &x).rem_euclid(p);
            let mut roots = vec![f.sqrt_mod(&s).unwrap()];
            if !long {
                roots.push(s.sqrt_mod(p).unwrap());
            }
            for r in roots {
                assert_eq!((&r * &r).rem_euclid(p), s);
                assert!(&r * 2u8 < *p);
                assert!(r == x || &r + &x == *p);
            }
            let all = if long {
                f.sqrt_mod_all(&s)
            } else {
                s.sqrt_mod_all(p)
            };
            assert_eq!(all.len(), 2);
            assert_eq!(&all[0] + &all[1], *p);
        }
    };
    // 3 modulo 4; 5 modulo 8; the prime of P-224, with `p - 1` divisible by 2^96 (Cipolla's);
    // and one with 2^20 (Tonelli and Shanks's). Miri takes the shorter ones only.
    if !cfg!(miri) {
        check(&((n(1) << 521usize) - 1u8));
        check(&((n(1) << 255usize) - 19u8));
    }
    check(&((n(1) << 224usize) - (n(1) << 96usize) + 1u8));
    check(&n(7 * 0x10_0000 + 1)); // 7340033 = 7 * 2^20 + 1
    let p = (n(1) << 127usize) - 1u8;
    assert_eq!(n(-1).jacobi(&p), -1);
    assert!((&p - 1u8).sqrt_mod(&p).is_none());
}

#[test]
fn sqrt_mod_of_prime_powers_and_products() {
    // Hensel's lifting to a power of a long prime, by the known factorisation (the modulus has
    // no small factor to find), and to a power of a short one by factoring.
    let p = (n(1) << 89usize) - 1u8;
    let p3 = p.pow(3);
    let f = Factorization::new(vec![(p.clone(), 3)], ByteOrder::BigEndian);
    for x in [n(5), n(123_456_789), (&p3 >> 1usize) - 3u8] {
        let s = (&x * &x).rem_euclid(&p3);
        let r = f.sqrt_mod(&s).unwrap();
        assert_eq!((&r * &r).rem_euclid(&p3), s);
        let all = f.sqrt_mod_all(&s);
        assert_eq!(all.len(), 2);
        assert!(all.contains(&x.rem_euclid(&p3)));
        assert_eq!(all[0], r);
    }
    if cfg!(miri) {
        // The rest factors long moduli, too slow under Miri.
        return;
    }
    let q5 = n(10007).pow(5);
    for x in [n(5), n(123_456_789), (&q5 >> 1usize) - 3u8] {
        let s = (&x * &x).rem_euclid(&q5);
        let all = s.sqrt_mod_all(&q5);
        assert_eq!(all.len(), 2);
        assert!(all.contains(&x.rem_euclid(&q5)));
        assert_eq!(s.sqrt_mod(&q5).as_ref(), all.first());
    }
    let two = n(1) << 300usize;
    let x = (n(3) << 150usize) + 12345u32;
    let s = (&x * &x).rem_euclid(&two);
    let all = s.sqrt_mod_all(&two);
    assert_eq!(all.len(), 4);
    for r in &all {
        assert_eq!((r * r).rem_euclid(&two), s);
    }
    assert!(all.contains(&x));
    assert_eq!(s.sqrt_mod(&two).as_ref(), all.first());
    // A product of a long prime and small ones: 2^3 roots, the smallest of them.
    let m = &p * 3u8 * 5u8 * 7u8;
    let x = n(1_000_003);
    let s = (&x * &x).rem_euclid(&m);
    let all = s.sqrt_mod_all(&m);
    assert_eq!(all.len(), 16);
    let r = s.sqrt_mod(&m).unwrap();
    assert_eq!(r, all[0]);
    assert!(all.windows(2).all(|w| w[0] < w[1]));
    for r in &all {
        assert_eq!((r * r).rem_euclid(&m), s);
    }
    // A square sharing factors with the modulus: 0 modulo 3^4 has the 9 multiples of 9 below 81.
    let all = n(0).sqrt_mod_all(&n(81));
    assert_eq!(all.len(), 9);
    // 81 = 3^4 modulo 3^5: the roots are 9 and 18 modulo 27.
    assert_eq!(n(81).sqrt_mod(&n(3).pow(5)), Some(n(9)));
    assert!(n(162).sqrt_mod(&n(3).pow(5)).is_none());
}

#[test]
fn sqrt_mod_of_many_primes() {
    // The first 26 odd primes: 2^26 roots, the smallest found by meeting in the middle.
    let primes: Vec<u64> = (3u64..)
        .filter(|&q| (2..q).take_while(|d| d * d <= q).all(|d| q % d != 0))
        .take(if cfg!(miri) { 4 } else { 26 })
        .collect();
    let m: Number = primes
        .iter()
        .map(|&q| Number::new_from_u64(q, ByteOrder::BigEndian))
        .product();
    let x = n(1_000_000_007);
    let s = (&x * &x).rem_euclid(&m);
    let r = s.sqrt_mod(&m).unwrap();
    assert_eq!((&r * &r).rem_euclid(&m), s);
    assert!(r <= x);
    let f = Factorization::new(
        primes
            .iter()
            .map(|&q| (Number::new_from_u64(q, ByteOrder::BigEndian), 1))
            .collect(),
        ByteOrder::BigEndian,
    );
    assert_eq!(f.sqrt_mod(&s), Some(r.clone()));
    if !cfg!(miri) {
        // Against every choice, for the first 14 primes only.
        let m14: Number = primes[..14]
            .iter()
            .map(|&q| Number::new_from_u64(q, ByteOrder::BigEndian))
            .product();
        let s14 = s.rem_euclid(&m14);
        let all = s14.sqrt_mod_all(&m14);
        assert_eq!(all.len(), 1 << 14);
        assert_eq!(s14.sqrt_mod(&m14).as_ref(), all.first());
    }
}

#[test]
fn sqrt_mod_of_any_operand() {
    let p = n(13);
    assert_eq!(n(0).sqrt_mod(&p), Some(n(0)));
    assert_eq!(n(13).sqrt_mod(&p), Some(n(0)));
    assert_eq!(n(23).sqrt_mod(&p), Some(n(6)));
    assert_eq!(n(-3).sqrt_mod(&p), Some(n(6)));
    assert_eq!(n(5).sqrt_mod(&n(1)), Some(n(0)));
    assert_eq!(n(5).sqrt_mod_all(&n(1)), [n(0)]);
    // In the byte order of `self`, whatever the modulus's.
    for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
        let p = Number::new_from_u32(0x1_0001, ByteOrder::BigEndian);
        let a = Number::new_from_u32(1 << 20, order);
        let r = a.sqrt_mod(&p).unwrap();
        assert_eq!(r.byte_order, order);
        assert_eq!(r, Number::new_from_u32(1024, order));
        let r = other(&a).sqrt_mod(&other(&p)).unwrap();
        assert_eq!(r, Number::new_from_u32(1024, order));
        for r in a.sqrt_mod_all(&(&p * 8u8)) {
            assert_eq!(r.byte_order, order);
        }
    }
    // Long numbers modulo words, and negative ones.
    let big = (n(1) << 100usize) + 4u8;
    // 2^100 = 16^25 = 1 modulo 15.
    assert_eq!(big.sqrt_mod(&n(15)), n(5).sqrt_mod(&n(15)));
    assert_eq!((-&big).sqrt_mod(&n(15)), n(-5).sqrt_mod(&n(15)));
    assert_eq!(big.sqrt_mod_all(&n(15)), n(5).sqrt_mod_all(&n(15)));
}

#[test]
#[should_panic(expected = "the modulus must be positive")]
fn sqrt_mod_refuses_zero() {
    n(3).sqrt_mod(&n(0));
}

#[test]
#[should_panic(expected = "the modulus must be positive")]
fn sqrt_mod_all_refuses_a_negative_modulus() {
    n(3).sqrt_mod_all(&n(-7));
}

/// The primitive solutions of `x^2 + d y^2 = m` with `x, y >= 0`, by trying every `y`.
fn cornacchia_reference(d: i64, m: i64) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    let mut y = 0;
    while d * y * y <= m {
        let rest = m - d * y * y;
        let x = (rest as f64).sqrt() as i64;
        for x in [x - 1, x, x + 1] {
            if x >= 0 && x * x == rest && gcd(x, y) == 1 {
                out.push((x, y));
            }
        }
        y += 1;
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn cornacchia_against_every_solution() {
    let (dn, mn) = if cfg!(miri) { (3, 30) } else { (40, 1500) };
    for d in 1..=dn {
        for m in 1..=mn {
            let want = cornacchia_reference(d, m);
            let got: Vec<(i64, i64)> = Number::cornacchia(&n(d), &n(m))
                .iter()
                .map(|(x, y)| (int(x) as i64, int(y) as i64))
                .collect();
            assert_eq!(got, want, "x^2 + {d} y^2 = {m}");
        }
    }
}

#[test]
fn cornacchia_of_long_numbers() {
    // 2^127 - 1 = 3 modulo 4 is not a sum of two squares; 2^521 - 1 neither; but a prime 1 modulo
    // 4 is, in exactly one way up to order.
    let p = if cfg!(miri) {
        (n(1) << 89usize) - 1u8 // 7 modulo 8: x^2 + y^2 does not take it
    } else {
        (n(1) << 255usize) - 19u8 // 1 modulo 4
    };
    let s = if cfg!(miri) {
        Factorization::new(vec![(p.clone(), 1)], ByteOrder::BigEndian).cornacchia(&n(1))
    } else {
        Number::cornacchia(&n(1), &p)
    };
    if cfg!(miri) {
        assert!(s.is_empty());
        return;
    }
    assert_eq!(s.len(), 2);
    let (x, y) = &s[0];
    assert_eq!(x * x + y * y, p);
    assert_eq!(s[1], (y.clone(), x.clone()));
    assert!(Number::cornacchia(&n(1), &((n(1) << 127usize) - 1u8)).is_empty());
    // x^2 + 7 y^2 for a product of two primes it represents: two solutions.
    let q = n(11); // 11 = 2^2 + 7
    let m = &p * &q;
    for (x, y) in Number::cornacchia(&n(7), &m) {
        assert_eq!(&x * &x + &y * &y * 7u8, m);
        assert!(x.gcd(&y) == 1u8);
    }
    // In the byte order of `m`.
    let m = Number::new_from_u32(65, ByteOrder::LittleEndian);
    for (x, y) in Number::cornacchia(&n(1), &m) {
        assert_eq!(x.byte_order, ByteOrder::LittleEndian);
        assert_eq!(y.byte_order, ByteOrder::LittleEndian);
    }
}

#[test]
fn cornacchia_of_a_factorisation() {
    // The same solutions as from the number, without factoring it.
    let limit = if cfg!(miri) { 12 } else { 600 };
    for m in 1..limit {
        let f = factorise(m as u64);
        for d in [1, 2, 3, 7] {
            assert_eq!(
                f.cornacchia(&n(d)),
                Number::cornacchia(&n(d), &n(m)),
                "{d} {m}"
            );
        }
    }
    // A long prime and a product beyond 2^64, in the byte order of the factorisation.
    let p = (n(1) << 127usize) - 1u8; // 3 modulo 4, 7 modulo 8: x^2 + 7 y^2 takes it
    let q = n(1_000_000_007);
    let f = Factorization::new(
        vec![(p.clone(), 1), (q.clone(), 1)],
        ByteOrder::LittleEndian,
    );
    let m = &p * &q;
    let s = f.cornacchia(&n(7));
    if !cfg!(miri) {
        // Factoring `p q` is too slow under Miri.
        assert_eq!(s, Number::cornacchia(&n(7), &other(&m)));
    }
    for (x, y) in &s {
        assert_eq!(x * x + y * y * 7u8, m);
        assert_eq!(x.byte_order, ByteOrder::LittleEndian);
    }
}

#[test]
#[should_panic(expected = "d must be positive")]
fn cornacchia_refuses_a_negative_d_of_a_factorisation() {
    Factorization::new(vec![(n(5), 1)], ByteOrder::BigEndian).cornacchia(&n(-1));
}

#[test]
#[should_panic(expected = "d must be positive")]
fn cornacchia_refuses_zero_d() {
    Number::cornacchia(&n(0), &n(5));
}

/// A pseudo-random generator for the tests' inputs: xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A number of exactly `limbs` limbs, its top bit set, in the byte order `order`.
    fn limbs(&mut self, limbs: usize, order: ByteOrder) -> Number {
        let mut v: Vec<u8> = (0..limbs).flat_map(|_| self.next().to_le_bytes()).collect();
        *v.last_mut().unwrap() |= 0x80;
        if order == ByteOrder::BigEndian {
            v.reverse();
        }
        Number::new(v, order)
    }
}

/// The Jacobi symbol by Euclid's algorithm on numbers, a remainder and reciprocity a step: the
/// tests' own, slow and plain.
fn jacobi_slow(a: &Number, m: &Number) -> i32 {
    let mut a = a.rem_euclid(m);
    let mut m = m.clone();
    let mut t = 1;
    while !a.is_zero() {
        while !a.bit(0) {
            a = &a >> 1usize;
            let r = m.rem_euclid(&n(8));
            if r == 3u8 || r == 5u8 {
                t = -t;
            }
        }
        core::mem::swap(&mut a, &mut m);
        if a.rem_euclid(&n(4)) == 3u8 && m.rem_euclid(&n(4)) == 3u8 {
            t = -t;
        }
        a = a.rem_euclid(&m);
    }
    if m == 1u8 { t } else { 0 }
}

#[test]
fn jacobi_at_every_length() {
    // Every length of a modulus from one limb to beyond the stack's 16, across the binary steps'
    // lengths (up to 6 limbs) and Lehmer's above, both byte orders, against Euclid's algorithm.
    let mut r = Rng(0x9e37_79b9);
    let (lengths, cases) = if cfg!(miri) { (8, 2) } else { (19, 24) };
    for k in 1..=lengths {
        for c in 0..cases {
            let order = if c % 2 == 0 {
                ByteOrder::BigEndian
            } else {
                ByteOrder::LittleEndian
            };
            let m = &r.limbs(k, order) | &n(1);
            // Numerators of every length up to the modulus's and one longer, of both signs.
            let len = 1 + (r.next() % (k as u64 + 1)) as usize;
            let mut a = r.limbs(len, ByteOrder::LittleEndian);
            if c % 3 == 0 {
                a = -a;
            }
            let want = jacobi_slow(&a, &m);
            assert_eq!(a.jacobi(&m), want, "{k} limbs, case {c}");
            assert_eq!(
                other(&a).jacobi(&other(&m)),
                want,
                "{k} limbs, case {c}, other form"
            );
            assert_eq!(a.kronecker(&m), want, "{k} limbs, case {c}, Kronecker");
        }
    }
}

#[test]
fn jacobi_on_the_rare_paths() {
    // Numbers whose difference has low limbs of zero (the binary steps take those apart), equal
    // numbers, numerators past twice the modulus, and a common factor, at 2 to 8 limbs.
    let mut r = Rng(0x51ed_270b);
    let top = if cfg!(miri) { 3 } else { 8 };
    for k in 2..=top {
        let m = &r.limbs(k, ByteOrder::BigEndian) | &n(1);
        for shift in [64usize, 128, 64 * (k - 1)] {
            if shift >= 64 * k {
                continue;
            }
            let d = &n(r.next() as i64 & 0x7fff_ffff) << shift;
            for a in [&m - &d, &m + &d, &(&m * 3u8) - &d] {
                assert_eq!(
                    a.jacobi(&m),
                    jacobi_slow(&a, &m),
                    "{k} limbs, low limbs equal"
                );
                assert_eq!(m.jacobi(&(&a | &n(1))), jacobi_slow(&m, &(&a | &n(1))));
            }
        }
        assert_eq!(m.jacobi(&m), 0);
        let f = &r.limbs(1, ByteOrder::BigEndian) | &n(1);
        let mf = &m * &f;
        assert_eq!((&f * 5u8).jacobi(&mf), 0, "{k} limbs, a common factor");
        let big = &(&m * 7u8) + 2u8;
        assert_eq!(big.jacobi(&m), jacobi_slow(&n(2), &m));
        assert_eq!((-&big).jacobi(&m), jacobi_slow(&n(-2), &m));
    }
}

#[test]
fn sqrt_mod_of_primes_of_every_class() {
    // 3 modulo 4, 5 modulo 8 (Atkin's), 1 modulo 8 (Tonelli and Shanks's), and with more than 24
    // factors of two in `p - 1` (Cipolla's), modulo words and modulo primes of 2, 3 and 5 limbs.
    let check = |p: &Number| {
        let f = Factorization::new(vec![(p.clone(), 1)], ByteOrder::LittleEndian);
        for x in [n(2), n(3), n(1_234_567), (p - 3u8) >> 1usize] {
            let s = (&x * &x).rem_euclid(p);
            let r = f.sqrt_mod(&s).unwrap();
            assert_eq!((&r * &r).rem_euclid(p), s, "modulo {p}");
            assert!(&r * 2u8 < *p);
            let all = f.sqrt_mod_all(&s);
            assert_eq!(all.len(), 2);
            assert_eq!(all[0], r);
            // A non-square: the product of a square and a non-square.
            let mut z = n(2);
            while z.jacobi(p) != -1 {
                z = &z + 1u8;
            }
            assert!(f.sqrt_mod(&(&s * &z).rem_euclid(p)).is_none());
        }
    };
    // Words: 3 mod 4, 5 mod 8, 1 mod 8 with s = 3, and 5 2^25 + 1, 7 2^26 + 1, 15 2^27 + 1.
    for p in [
        1_000_003u64,
        1_000_037,
        1_000_033,
        167_772_161,
        469_762_049,
        2_013_265_921,
    ] {
        check(&Number::new_from_u64(p, ByteOrder::BigEndian));
    }
    if cfg!(miri) {
        return;
    }
    // Long primes of each class: the next primes above these, of 2, 3 and 5 limbs.
    for bits in [100usize, 190, 300] {
        let mut found = [false; 3];
        let mut p = (n(1) << bits) + 1u8;
        while found.iter().any(|&f| !f) {
            p = p.next_prime();
            let r8 = int(&p.rem_euclid(&n(8)));
            let class = match r8 {
                3 | 7 => 0,
                5 => 1,
                _ => 2,
            };
            if !found[class] {
                found[class] = true;
                check(&p);
            }
        }
    }
    // A long prime 1 modulo 2^26: Cipolla's on limbs.
    let mut k = (n(1) << 70usize) + 1u8;
    let p = loop {
        let p = (&k << 26usize) + 1u8;
        if p.is_probable_prime() {
            break p;
        }
        k = &k + 2u8;
    };
    check(&p);
}

#[test]
fn smallest_root_of_many_word_primes() {
    // The 13 odd primes from 3 to 43: 2^13 choices, past 4096, so the smallest root of a word
    // modulus is found by meeting in the middle.
    let primes = [3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43];
    let m: u64 = primes.iter().product();
    let mn = Number::new_from_u64(m, ByteOrder::BigEndian);
    let f = Factorization::new(
        primes
            .iter()
            .map(|&q| (Number::new_from_u64(q, ByteOrder::LittleEndian), 1))
            .collect(),
        ByteOrder::BigEndian,
    );
    if cfg!(miri) {
        // The smallest root alone, both ways, by meeting in the middle.
        let s = n(4);
        let r = f.sqrt_mod(&s).unwrap();
        assert_eq!(s.sqrt_mod(&mn), Some(r.clone()));
        assert_eq!((&r * &r).rem_euclid(&mn), s);
        assert!(r <= n(2));
        return;
    }
    for x in [n(2), n(1_000_003), n(987_654_321_127)] {
        let s = (&x * &x).rem_euclid(&mn);
        let all = s.sqrt_mod_all(&mn);
        assert_eq!(all.len(), 1 << 13);
        assert!(all.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(s.sqrt_mod(&mn).as_ref(), all.first());
        assert_eq!(f.sqrt_mod(&s).as_ref(), all.first());
        assert_eq!(f.sqrt_mod_all(&s), all);
    }
}

#[test]
fn square_roots_modulo_composite_words() {
    // Prime powers, powers of two from 2 to 2^20, and squares sharing factors with the modulus,
    // against every root, in the word path.
    let moduli: &[i64] = if cfg!(miri) {
        &[8, 9, 72]
    } else {
        &[
            2, 4, 8, 16, 32, 64, 1024, 9, 27, 81, 243, 25, 125, 49, 343, 72, 200, 360, 2025, 3087,
        ]
    };
    for &m in moduli {
        for a in 0..m {
            let want = roots_reference(a, m);
            let got: Vec<i64> = n(a)
                .sqrt_mod_all(&n(m))
                .iter()
                .map(|r| int(r) as i64)
                .collect();
            assert_eq!(got, want, "roots of {a} modulo {m}");
            let f = factorise(m as u64);
            assert_eq!(
                f.sqrt_mod(&n(a)).map(|r| int(&r) as i64),
                want.first().copied(),
                "{a} modulo {m}, factored"
            );
        }
    }
    // Powers of two up to 2^63 times an odd prime, squares of odd numbers.
    for e in [3u32, 20, 40, 62] {
        let m = (1u64 << e) * if e < 60 { 3 } else { 1 };
        let mn = Number::new_from_u64(m, ByteOrder::BigEndian);
        let x = n(123_456_789);
        let s = (&x * &x).rem_euclid(&mn);
        let all = s.sqrt_mod_all(&mn);
        for r in &all {
            assert_eq!((r * r).rem_euclid(&mn), s);
        }
        assert!(all.contains(&x.rem_euclid(&mn)));
        assert_eq!(s.sqrt_mod(&mn).as_ref(), all.first());
    }
}

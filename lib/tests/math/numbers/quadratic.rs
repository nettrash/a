use a::math::numbers::{ByteOrder, Number};

fn n(x: i64) -> Number {
    Number::new_from_i64(x, ByteOrder::BigEndian)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
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

#[test]
#[should_panic(expected = "the modulus must be odd and positive")]
fn sqrt_mod_refuses_an_even_modulus() {
    n(3).sqrt_mod(&n(10));
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
        for x in [n(2), n(3), n(1234567), (p - 5u8) >> 1usize] {
            let s = (&x * &x).rem_euclid(p);
            let r = s.sqrt_mod(p).unwrap();
            assert_eq!((&r * &r).rem_euclid(p), s);
            assert!(&r * 2u8 < *p);
            assert!(r == x || &r + &x == *p);
        }
    };
    // 3 modulo 4; 5 modulo 8; and the prime of P-224, with `p - 1` divisible by 2^96.
    check(&((n(1) << 521usize) - 1u8));
    check(&((n(1) << 255usize) - 19u8));
    check(&((n(1) << 224usize) - (n(1) << 96usize) + 1u8));
    let p = (n(1) << 127usize) - 1u8;
    assert_eq!(n(-1).jacobi(&p), -1);
    assert!((&p - 1u8).sqrt_mod(&p).is_none());
}

#[test]
fn sqrt_mod_of_any_operand() {
    let p = n(13);
    assert_eq!(n(0).sqrt_mod(&p), Some(n(0)));
    assert_eq!(n(13).sqrt_mod(&p), Some(n(0)));
    assert_eq!(n(23).sqrt_mod(&p), Some(n(6)));
    assert_eq!(n(-3).sqrt_mod(&p), Some(n(6)));
    // In the byte order of `self`, whatever the modulus's.
    for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
        let p = Number::new_from_u32(0x1_0001, ByteOrder::BigEndian);
        let a = Number::new_from_u32(1 << 20, order);
        let r = a.sqrt_mod(&p).unwrap();
        assert_eq!(r.byte_order, order);
        assert_eq!(r, Number::new_from_u32(1024, order));
    }
    // A composite modulus: no wrong root.
    let m = n(21);
    for a in 0..21 {
        if let Some(r) = n(a).sqrt_mod(&m) {
            assert_eq!((&r * &r).rem_euclid(&m), n(a));
        }
    }
}

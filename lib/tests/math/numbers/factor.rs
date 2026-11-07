//! Tests of factorisation: [`Number::factor`], the methods of `a::math::numbers::factor`, perfect
//! powers and [`Factorization`], against PARI/GP 2.19 (the tables at the end), against trial
//! division for small numbers, and by the properties of the factors the methods give.

use a::math::numbers::{ByteOrder, Factorization, Number, factor};

use super::common::*;

/// The decimal `s` in byte order `order`.
fn num(s: &str, order: ByteOrder) -> Number {
    let n: Number = s.parse().unwrap();
    let mut v = n.value.clone();
    if order != n.byte_order {
        v.reverse();
    }
    Number {
        value: v,
        negative: n.negative,
        byte_order: order,
    }
}

/// `n` with `extra` zero bytes above its most significant byte, as a non-canonical number.
fn padded(n: &Number, extra: usize) -> Number {
    let mut v = n.value.clone();
    match n.byte_order {
        ByteOrder::BigEndian => {
            let mut w = vec![0u8; extra];
            w.extend_from_slice(&v);
            v = w;
        }
        ByteOrder::LittleEndian => v.extend(core::iter::repeat_n(0, extra)),
    }
    Number {
        value: v,
        negative: n.negative,
        byte_order: n.byte_order,
    }
}

/// A factorisation as PARI's table writes it: `p^e`, the exponent left out when it is 1, or
/// always with `always`, separated by spaces.
fn written(f: &[(Number, u32)], always: bool) -> String {
    f.iter()
        .map(|(p, e)| {
            if *e > 1 || always {
                format!("{p}^{e}")
            } else {
                format!("{p}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The factorisation of `n` by trial division, as `(p, e)`.
fn naive(mut n: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    let mut p = 2;
    while p * p <= n {
        let mut e = 0;
        while n.is_multiple_of(p) {
            n /= p;
            e += 1;
        }
        if e > 0 {
            out.push((p, e));
        }
        p += 1;
    }
    if n > 1 {
        out.push((n, 1));
    }
    out
}

/// Checks the factorisation of `n` against `expected`, in both byte orders, signs and with zero
/// bytes above the top.
fn check_factor(s: &str, expected: &str) {
    for order in ORDERS {
        let n = num(s, order);
        let f = n.factor();
        assert!(f.byte_order() == order);
        assert_eq!(written(f.factors(), false), expected, "factor({s})");
        for (p, _) in f.factors() {
            assert!(p.byte_order() == order, "the primes in the byte order of n");
        }
        assert_eq!(f.value(), n);
        let minus = Number {
            negative: true,
            ..padded(&n, 3)
        };
        assert_eq!(minus.factor(), f, "factor(-{s}) with zero bytes");
    }
}

// Number::factor

#[test]
fn factor_matches_pari() {
    // Under Miri, a few that take trial division and the tests alone.
    let few = [
        "561",
        "41041",
        "18446744073709551615",
        "1606938044258990275541962092341162602522202993782792835301376",
    ];
    for (s, expected) in FACTORED {
        if !cfg!(miri) || few.contains(s) {
            check_factor(s, expected);
        }
    }
}

#[test]
fn factor_small_numbers() {
    let top = size(5000, 60) as u64;
    for n in 1..=top {
        let expected: Vec<String> = naive(n)
            .iter()
            .map(|&(p, e)| {
                if e > 1 {
                    format!("{p}^{e}")
                } else {
                    format!("{p}")
                }
            })
            .collect();
        for order in ORDERS {
            let f = Number::new_from_u64(n, order).factor();
            assert_eq!(
                written(f.factors(), false),
                expected.join(" "),
                "factor({n})"
            );
        }
    }
}

#[test]
fn factor_random_u64() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..size(300, 3) {
        // Under Miri, numbers of up to 24 bits, which Hart's method factors.
        let n = xorshift(&mut state)
            >> if cfg!(miri) {
                40
            } else {
                below(&mut state, 40)
            };
        let n = n.max(1);
        let f = Number::new_from_u64(n, ByteOrder::LittleEndian).factor();
        let mut product = 1u128;
        let mut last = Number::new_from_u8(1, ByteOrder::LittleEndian);
        for (p, e) in f.factors() {
            assert!(p.is_probable_prime() && *p > last, "{p} in factor({n})");
            last = p.clone();
            product *= u128::from(u64::try_from(p).unwrap()).pow(*e);
        }
        assert_eq!(product, u128::from(n), "factor({n})");
    }
}

#[test]
fn factor_one_and_minus_one() {
    for order in ORDERS {
        assert!(Number::new_from_i32(1, order).factor().factors().is_empty());
        assert!(
            Number::new_from_i32(-1, order)
                .factor()
                .factors()
                .is_empty()
        );
        assert_eq!(Number::new_from_i32(-1, order).factor().value(), 1u8);
    }
}

#[test]
#[should_panic(expected = "zero has no factorisation")]
fn factor_of_zero_panics() {
    Number::new_from_u8(0, ByteOrder::BigEndian).factor();
}

#[test]
#[should_panic(expected = "zero has no factorisation")]
fn factor_of_empty_panics() {
    Number::new(Vec::new(), ByteOrder::LittleEndian).factor();
}

#[test]
#[should_panic(expected = "zero has no factorisation")]
fn factor_of_negative_zero_panics() {
    Number {
        value: vec![0, 0],
        negative: true,
        byte_order: ByteOrder::BigEndian,
    }
    .factor();
}

#[test]
fn factor_leaves_input() {
    // Under Miri, a number that trial division factors.
    let s = if cfg!(miri) {
        "10403"
    } else {
        "340282366920938463463374607431768211457"
    };
    let n = padded(&num(s, ByteOrder::BigEndian), 2);
    let before = (n.value.clone(), n.negative);
    let _ = n.factor();
    assert_eq!((n.value, n.negative), before);
}

// Factorization

#[test]
fn factorization_new_sorts_and_merges() {
    let order = ByteOrder::BigEndian;
    let n = |x: u64| Number::new_from_u64(x, ByteOrder::LittleEndian);
    let f = Factorization::new(
        vec![(n(7), 1), (n(3), 2), (n(7), 2), (n(5), 0), (n(2), 1)],
        order,
    );
    assert_eq!(written(f.factors(), false), "2 3^2 7^3");
    assert!(f.factors().iter().all(|(p, _)| p.byte_order() == order));
    assert_eq!(f.value(), 2u32 * 9 * 343);
    assert!(f.byte_order() == order);
    let v = f.clone().into_factors();
    assert_eq!(v.len(), 3);
    assert!(Factorization::new(Vec::new(), order).value() == 1u8);
    assert_eq!(
        Factorization::new(vec![(n(5), 0)], order).factors().len(),
        0
    );
}

#[test]
#[should_panic(expected = "a prime factor must be at least 2")]
fn factorization_new_rejects_one() {
    Factorization::new(
        vec![(Number::new_from_u8(1, ByteOrder::BigEndian), 1)],
        ByteOrder::BigEndian,
    );
}

#[test]
#[should_panic(expected = "overflows u32")]
fn factorization_new_rejects_overflow() {
    let p = Number::new_from_u8(3, ByteOrder::BigEndian);
    Factorization::new(vec![(p.clone(), u32::MAX), (p, 1)], ByteOrder::BigEndian);
}

// Perfect powers

#[test]
fn perfect_power_matches_pari() {
    let count = size(POWERS.len(), 6);
    for &(s, k, b) in &POWERS[..count] {
        for order in ORDERS {
            let n = num(s, order);
            let got = n.perfect_power();
            if k == 0 {
                assert!(got.is_none(), "perfect_power({s}) = {got:?}");
                assert!(!n.is_perfect_power());
            } else {
                let (root, kk) = got.unwrap_or_else(|| panic!("perfect_power({s})"));
                assert_eq!(
                    (root.to_string(), kk),
                    (b.to_string(), k),
                    "perfect_power({s})"
                );
                assert!(root.byte_order() == order);
                assert!(n.is_perfect_power());
                assert_eq!(padded(&n, 2).perfect_power(), Some((root, kk)));
            }
        }
    }
}

#[test]
fn perfect_power_small() {
    let top = size(1 << 14, 200) as i64;
    for x in -top..=top {
        // The largest `k` by brute force: the largest `k` with an integer `k`-th root.
        let a = x.unsigned_abs();
        let mut best: Option<(i64, u32)> = None;
        if a > 1 {
            for k in 2..64u32 {
                if x < 0 && k % 2 == 0 {
                    continue;
                }
                let r = (a as f64).powf(1.0 / f64::from(k)).round() as u64;
                for c in r.saturating_sub(1)..=r + 1 {
                    if c >= 2 && c.checked_pow(k) == Some(a) {
                        let b = if x < 0 { -(c as i64) } else { c as i64 };
                        best = Some((b, k));
                    }
                }
            }
        }
        let got = Number::new_from_i64(x, ByteOrder::LittleEndian).perfect_power();
        let got = got.map(|(b, k)| (i64::try_from(&b).unwrap(), k));
        assert_eq!(got, best, "perfect_power({x})");
    }
}

#[test]
fn perfect_power_of_large_powers() {
    // Exponents of small primes past what is counted: 3^4096, and 7^3 * 3^(3 * 100) in limbs.
    let k = size(1 << 12, 1 << 6) as u32;
    for order in ORDERS {
        let three = Number::new_from_u8(3, order);
        assert_eq!(three.pow(k).perfect_power(), Some((three.clone(), k)));
        let b = &Number::new_from_u8(7, order) * &three.pow(100);
        let n = b.pow(3);
        assert_eq!(n.perfect_power(), Some((b.clone(), 3)));
        assert_eq!((-&n).perfect_power(), Some((-&b, 3)));
        assert!((&n + 1u8).perfect_power().is_none());
    }
}

#[test]
fn perfect_power_of_zero_and_units() {
    for order in ORDERS {
        for x in [0i32, 1, -1] {
            assert!(Number::new_from_i32(x, order).perfect_power().is_none());
        }
        assert!(Number::new(Vec::new(), order).perfect_power().is_none());
    }
}

// Trial division

#[test]
fn trial_matches_pari() {
    // Under Miri, a few of the bounds within the tables, whose table beyond is safe code.
    let cases = TRIAL
        .iter()
        .filter(|c| !cfg!(miri) || c.1 <= 8191)
        .take(size(TRIAL.len(), 3));
    for &(s, bound, primes, rest) in cases {
        for order in ORDERS {
            let n = num(s, order);
            let (f, c) = factor::trial(&n, bound);
            let w: Vec<String> = f.iter().map(|(p, e)| format!("{p}^{e}")).collect();
            assert_eq!(w.join(" "), primes, "trial({s}, {bound})");
            assert_eq!(c.to_string(), rest, "trial({s}, {bound})");
            assert!(c.byte_order() == order);
            let minus = Number {
                negative: true,
                ..padded(&n, 1)
            };
            assert_eq!(factor::trial(&minus, bound), (f, c));
        }
    }
}

#[test]
fn trial_small_numbers() {
    for n in 0..size(3000, 40) as u64 {
        let bounds: &[u64] = if cfg!(miri) {
            &[0, 1, 2, 3, 10, 50, 1000]
        } else {
            &[0, 1, 2, 3, 10, 50, 1000, u64::MAX]
        };
        for &bound in bounds {
            let (f, c) = factor::trial(&Number::new_from_u64(n, ByteOrder::BigEndian), bound);
            if n == 0 {
                assert!(f.is_empty() && c == 0u8);
                continue;
            }
            let expected: Vec<(u64, u32)> =
                naive(n).into_iter().filter(|&(p, _)| p <= bound).collect();
            let rest = expected.iter().fold(n, |r, &(p, e)| r / p.pow(e));
            assert_eq!(f, expected, "trial({n}, {bound})");
            assert_eq!(c, rest, "trial({n}, {bound})");
        }
    }
}

#[test]
fn trial_to_the_square_root() {
    if cfg!(miri) {
        return;
    }
    // A 40-bit semiprime by trial division alone.
    let n = 1_048_573u64 * 1_048_583;
    for order in ORDERS {
        let (f, c) = factor::trial(&Number::new_from_u64(n, order), u64::MAX);
        assert_eq!(f, [(1_048_573, 1), (1_048_583, 1)]);
        assert_eq!(c, 1u8);
    }
}

// The methods

/// Checks that `f` is a factor of `n` other than 1 and `|n|`, in the byte order of `n`.
fn proper(n: &Number, f: &Number) {
    assert!(
        f.byte_order() == n.byte_order(),
        "the factor in the byte order of n"
    );
    assert!(*f > 1u8 && *f < n.abs(), "{f} is a proper factor of {n}");
    assert!((n % f).is_zero(), "{f} divides {n}");
}

#[test]
fn methods_trivial_inputs() {
    for order in ORDERS {
        for x in [0i64, 1, -1, 2, 3, -3, 5, 1_000_000_007, -1_000_000_007] {
            let n = Number::new_from_i64(x, order);
            assert!(factor::rho(&n, 1000).is_none(), "rho({x})");
            assert!(factor::pm1(&n, 100, 1000).is_none(), "pm1({x})");
            assert!(factor::pp1(&n, 100, 1000).is_none(), "pp1({x})");
            assert!(factor::ecm(&n, 100, 1000, 3).is_none(), "ecm({x})");
        }
        for x in [4i64, -6, 1 << 40, 2 * 1_000_000_007] {
            let n = Number::new_from_i64(x, order);
            let two = Number::new_from_u8(2, order);
            assert_eq!(factor::rho(&n, 1).unwrap(), two);
            assert_eq!(factor::pm1(&n, 1, 1).unwrap(), two);
            assert_eq!(factor::pp1(&n, 1, 1).unwrap(), two);
            assert_eq!(factor::ecm(&n, 1, 1, 1).unwrap(), two);
        }
        let big = num("170141183460469231731687303715884105727", order);
        assert!(
            factor::ecm(&big, 1000, 10000, 2).is_none(),
            "2^127 - 1 is prime"
        );
    }
}

#[test]
fn rho_finds_factors() {
    let cases = [
        "10403",
        "1000000016000000063",
        "4611686014132420609",
        "18446744073709551617",
        "1000000000000000000000000000001",
        "10000360000990000000000000000000000000001000036000099",
    ];
    for s in &cases[..size(cases.len(), 1)] {
        for order in ORDERS {
            let n = num(s, order);
            let f = factor::rho(&n, 1 << 22).unwrap_or_else(|| panic!("rho({s})"));
            proper(&n, &f);
            assert_eq!(factor::rho(&-&n, 1 << 22), Some(f), "rho(-{s})");
        }
    }
    // Too few steps for a factor of 30 bits.
    let n = Number::new_from_u64(1_073_741_789 * 1_073_741_827, ByteOrder::BigEndian);
    assert!(factor::rho(&n, size(100, 20) as u64).is_none());
}

#[test]
fn pm1_bounds() {
    // 1000033 - 1 = 2^5 3 11 947; 1000003 - 1 = 2 3 166667.
    for order in ORDERS {
        let n = Number::new_from_u64(1_000_003 * 1_000_033, order);
        assert!(factor::pm1(&n, 900, 0).is_none());
        assert_eq!(factor::pm1(&n, 947, 0).unwrap(), 1_000_033u32);
        assert_eq!(factor::pm1(&n, 200, 1000).unwrap(), 1_000_033u32);
        if cfg!(miri) {
            return;
        }
        if cfg!(miri) {
            return;
        }
        // Both at once: 166667 too, the first stage gone over again a prime at a time.
        let f = factor::pm1(&n, 200_000, 0).unwrap();
        proper(&n, &f);
    }
    // Primes of 131 and 150 bits: `p - 1` is 45557-smooth; `r - 1` is 23743-smooth but for one
    // prime 400009; `q - 1` has the prime 454197539.
    let p: Number = "2799305317384975809140303150819639813579".parse().unwrap();
    let r: Number = "1146666182864761109725416623499560637287846207"
        .parse()
        .unwrap();
    let q: Number = "1000000000000000000000000000057".parse().unwrap();
    for order in ORDERS {
        let n = num(&(&p * &q).to_string(), order);
        assert_eq!(factor::pm1(&n, 50_000, 0), Some(num(&p.to_string(), order)));
        let n = num(&(&r * &q).to_string(), order);
        assert!(factor::pm1(&n, 24_000, 0).is_none());
        assert_eq!(
            factor::pm1(&n, 24_000, 500_000),
            Some(num(&r.to_string(), order))
        );
    }
}

#[test]
fn pp1_finds_smooth_p_plus_1() {
    // 1000099 + 1 = 2^2 5^2 73 137; 1000117 + 1 = 2 7 71437 and 1000117 - 1 = 2^2 3^2 13 2137.
    for order in ORDERS {
        let n = Number::new_from_u64(1_000_099 * 1_000_117, order);
        assert_eq!(factor::pp1(&n, 200, 0).unwrap(), 1_000_099u32);
        assert!(factor::pp1(&n, 100, 0).is_none());
        assert_eq!(factor::pp1(&n, 100, 200).unwrap(), 1_000_099u32);
    }
    if cfg!(miri) {
        return;
    }
    // `t + 1` is 6871-smooth but for one prime 150001.
    let t: Number = "31891757208419076077548517844955176030681613"
        .parse()
        .unwrap();
    let q: Number = "1000000000000000000000000000057".parse().unwrap();
    for order in ORDERS {
        let n = num(&(&t * &q).to_string(), order);
        assert_eq!(
            factor::pp1(&n, 7000, 200_000),
            Some(num(&t.to_string(), order))
        );
    }
}

#[test]
fn ecm_finds_factors() {
    let cases = [
        ("1000000016000000063", 150, 3000),
        ("2305843025354595015495857657", 2000, 200_000),
        ("340282366920938463463374607431768211457", 50_000, 5_000_000),
        (
            "104212284386088603737239513041015515049159219437814320345077",
            11_000,
            1_100_000,
        ),
    ];
    // Under Miri, a number of 14 bits, with small bounds.
    let miri = [("10403", 30, 300)];
    let cases: &[(&str, u64, u64)] = if cfg!(miri) { &miri } else { &cases };
    for &(s, b1, b2) in cases {
        for order in ORDERS {
            let n = num(s, order);
            let f = factor::ecm(&n, b1, b2, 200).unwrap_or_else(|| panic!("ecm({s})"));
            proper(&n, &f);
            // The same curves give the same factor.
            assert_eq!(factor::ecm(&n, b1, b2, 200), Some(f));
        }
    }
}

/// A number of `limbs` limbs with the top bit of its top limb set or clear, `p` times a cofactor
/// without small factors: `p` times the next prime after `2^(64 limbs - bits(p) - 1)` and a few
/// bits more.
fn sized(p: &Number, limbs: u32, high: bool) -> Number {
    let bits = 64 * limbs - if high { 1 } else { 4 };
    let shift = bits as usize - p.bits() as usize;
    // Under Miri, the primes after `2^shift` of the cases it runs, by PARI's `nextprime`, as the
    // search for them would take most of its time.
    let known = [
        (50, "1125899906842679"),
        (
            178,
            "383123885216472214589586756787577295904684780545900631",
        ),
        (
            181,
            "3064991081731777716716694054300618367237478244367204517",
        ),
        (
            562,
            "15095849699286165408966218323953075563667684881665761713504825200982496649568595408\
             344506900710635070165873500123055379599082338519699170530245737005730784174597581307\
             949",
        ),
    ];
    let q = match known.iter().find(|k| k.0 == shift) {
        Some(k) if cfg!(miri) => k.1.replace(' ', "").parse::<Number>().unwrap(),
        _ => (&Number::new_from_u8(1, ByteOrder::LittleEndian) << shift).next_prime(),
    };
    p * &q
}

#[test]
fn methods_in_every_arithmetic() {
    // A prime of 31 bits with `p - 1 = 2^2 3 41 97 149 151`, and one of 20 bits for rho; under
    // Miri, the arithmetic of one, three and nine limbs (registers, lazily reduced, and memory),
    // with a prime of 10 bits, `1021 - 1 = 2^2 3 5 17`, and smaller bounds.
    let p: Number = if cfg!(miri) { "1021" } else { "1073742277" }
        .parse()
        .unwrap();
    let small: Number = if cfg!(miri) { "1021" } else { "1048573" }.parse().unwrap();
    let sizes: Vec<u32> = if cfg!(miri) {
        vec![1, 3, 9]
    } else {
        (1..=12).collect()
    };
    let (target, b1, curves) = if cfg!(miri) {
        (&small, 20, 2)
    } else {
        (&p, 2000, 40)
    };
    for limbs in sizes {
        for high in [false, true] {
            if limbs == 1 && high || cfg!(miri) && high && limbs != 3 {
                continue;
            }
            for order in ORDERS {
                let n = num(&sized(target, limbs, high).to_string(), order);
                let f = factor::ecm(&n, b1, 100 * b1, curves)
                    .unwrap_or_else(|| panic!("ecm, {limbs} limbs"));
                proper(&n, &f);
                let n = num(&sized(&p, limbs, high).to_string(), order);
                let f = factor::pm1(
                    &n,
                    if cfg!(miri) { 20 } else { 160 },
                    if cfg!(miri) { 0 } else { 20_000 },
                )
                .unwrap_or_else(|| panic!("pm1, {limbs} limbs"));
                proper(&n, &f);
                let m = num(&sized(&small, limbs, high).to_string(), order);
                let f = factor::rho(&m, 1 << 16).unwrap_or_else(|| panic!("rho, {limbs} limbs"));
                proper(&m, &f);
                if cfg!(miri) {
                    break;
                }
            }
        }
    }
}

#[test]
fn factor_in_every_arithmetic() {
    // The driver's logic, in safe code; the arithmetic under Miri is in the test above.
    if cfg!(miri) {
        return;
    }
    // Products of primes of 20 to 40 bits and a large prime, of 1 to 12 limbs.
    let parts = ["1048573", "1073741789", "1099511627689", "4294967291"];
    let top = size(12, 2) as u32;
    for limbs in 2..=top {
        for high in [false, true] {
            let mut small = Number::new_from_u8(1, ByteOrder::BigEndian);
            for s in parts {
                small = &small * &s.parse::<Number>().unwrap();
            }
            let n = sized(&small, limbs.max(3), high);
            let f = n.factor();
            assert_eq!(f.value(), n);
            assert_eq!(f.factors().len(), 5, "factor({n})");
            for (q, e) in f.factors() {
                assert!(q.is_probable_prime() && *e == 1);
            }
        }
    }
}

#[test]
fn factor_large_and_repeated() {
    if cfg!(miri) {
        return;
    }
    // A prime of 521 bits squared, times a prime of 61 bits and a square of a 40-bit prime.
    let m521 = &(&Number::new_from_u8(1, ByteOrder::BigEndian) << 521usize) - 1u8;
    let m61 = Number::new_from_u64((1 << 61) - 1, ByteOrder::BigEndian);
    let q = Number::new_from_u64(1_099_511_627_689, ByteOrder::BigEndian);
    let n = &(&(&m521 * &m521) * &m61) * &(&q * &q);
    let f = n.factor();
    let w: Vec<(String, u32)> = f
        .factors()
        .iter()
        .map(|(p, e)| (p.to_string(), *e))
        .collect();
    assert_eq!(
        w,
        [
            (q.to_string(), 2),
            (m61.to_string(), 1),
            (m521.to_string(), 2)
        ]
    );
    // A perfect power of a semiprime.
    let s = &Number::new_from_u64(1_000_003, ByteOrder::LittleEndian)
        * &Number::new_from_u64(1_000_000_007, ByteOrder::LittleEndian);
    let f = s.pow(7).factor();
    assert_eq!(written(f.factors(), false), "1000003^7 1000000007^7");
}

// Tables by PARI/GP 2.19

/// Numbers and their factorisations by PARI/GP 2.19 `factor`.
const FACTORED: &[(&str, &str)] = &[
    ("4703188875111621185", "5 1801 19758209 26433893"),
    ("1317844507222731920", "2^4 5 191 678553 127103363"),
    ("11900662546857900719", "1839767 6468570502057"),
    ("5097591288787205964", "2^2 3 7 29 293 3491 30559 66947"),
    ("11584667072828618753", "2237 5178662079941269"),
    ("5027114302931515508", "2^2 7 131 541 2533332343741"),
    ("18423499187946698289", "3^2 113 5557 3259948729781"),
    ("9712817799972768933", "3 146381 22117665088531"),
    ("322365570951016346", "2 13 12398675805808321"),
    ("10448300034668499549", "3 7 13 38272161299152013"),
    ("16672307323444210004", "2^2 7 83 3499 91309 22454431"),
    ("5814333041893273867", "23 197 4217 304300202321"),
    ("10999465007158021558", "2 97 557 256181 397344971"),
    ("14251158126488808988", "2^2 31 101 1137907867014437"),
    ("2559650665968281560", "2^3 5 67 197 4848190518161"),
    ("12035272209800476831", "271 12829 3461735019109"),
    ("2028078747305483544", "2^3 3^5 7 17 499 33409 525869"),
    ("3437619881558324244", "2^2 3 4153 68978647595279"),
    ("15866378902788006148", "2^2 19^2 23^3 149 6060952099"),
    ("1726228020206424681", "3^2 251 2213 113683 3037421"),
    ("9547144953481510712", "2^3 997 1196984071399387"),
    ("1068077483087450938", "2 131 1259 3237991751261"),
    ("6160291489169876293", "83 443 2749 60945929353"),
    ("1322730562386395914", "2 661365281193197957"),
    ("7359364652519039496", "2^3 3 7 43805741979279997"),
    ("699813282082119917", "2382613 293716722809"),
    ("7006563191570226603", "3 11 51803 326437 12555581"),
    ("16477183874140210423", "16477183874140210423"),
    ("12920419574185852522", "2 2521 3463 299191 2473277"),
    ("4408548239611040356", "2^2 13 53 193 125117 66243421"),
    ("15472842248154079550", "2 5^2 73 4239134862507967"),
    ("13800856182764792316", "2^2 3 7 17 53 182348398376999"),
    ("1334558813009523812", "2^2 3301 101072312406053"),
    ("6857212478218388336", "2^4 553863917 773792563"),
    ("8560427003938913255", "5 7649 223831272164699"),
    ("2182092875517999217", "26407 82633122865831"),
    ("10194974255470566960", "2^4 3^3 5 17 277640911096693"),
    ("3292407436681127818", "2 1646203718340563909"),
    ("241391018980879100", "2^2 5^2 2413910189808791"),
    ("5185171863697057130", "2 5 23 887 25416263240513"),
    ("2023250106681198391", "255036391 7933182001"),
    ("2463867479057205041", "425552797 5789804453"),
    ("1021544297042347579", "610965293 1672016903"),
    ("712384092450913339", "287846623 2474873893"),
    ("2207018825381381351", "940320907 2347091093"),
    ("980714855527795523", "893965157 1097039239"),
    ("274058325697714913", "197594069 1386976477"),
    ("644528069320702363", "106406617 6057217939"),
    ("1187799414065946499", "974290127 1219143437"),
    ("2824593315448941017", "635443937 4445070841"),
    ("1776776338025809033", "769808993 2308074281"),
    ("1386714838954228271", "170786851 8119564421"),
    ("14703252546721941131", "3775550333 3894333607"),
    ("17193222129849309833", "4059278599 4235536367"),
    ("8362758237564852491", "2416266737 3461024443"),
    ("7921412945214860837", "2447531069 3236491273"),
    ("6729319497609502699", "2452731019 2743602721"),
    ("13707577756489935121", "3502591151 3913553471"),
    (
        "45412204045727875399765123723",
        "121311193 374344715625110334211",
    ),
    (
        "1652071737576813432102937678232758",
        "2 7 14797 898291 30308221 292920453835991",
    ),
    ("6906517671980136455", "5 7 43 67 8353 25639 319819"),
    ("18342733724498408200159", "193 3137 30296484256101599"),
    (
        "492508021286109517448213896006575",
        "3 5^2 1525352743 4305085264561737441547",
    ),
    (
        "2075712042624808076883617278",
        "2 11 103 1495859 19043461 32156639717",
    ),
    (
        "1450790379835478604031344871052695633",
        "59503892871293033 24381436404060812201",
    ),
    ("67205074802613282380201", "107 263 8287 288180906940403"),
    (
        "473117395701944751907094042180086",
        "2 107 33491230141 66012175204390217989",
    ),
    (
        "897076907220487627382624178",
        "2 3 41 3646654094392226127571643",
    ),
    (
        "46133274831718986146918657",
        "13 359 4350133 2272343323418887",
    ),
    (
        "2574947654555788490798180092962",
        "2 3 163 124577 1703687 4375967 2834835313",
    ),
    ("12455307855150589458088", "2^3 109 223 2081 30779462972983"),
    (
        "29861229282964161780001499",
        "13 53 415936537 104198481168643",
    ),
    (
        "17276688250766319425113023411",
        "3 11 23 67 197 1724557550799235042571",
    ),
    (
        "594519318039060275166369889",
        "11 61 2386140199 371319274708841",
    ),
    ("29050828994348346508", "2^2 7262707248587086627"),
    (
        "9529252067957329076433308096538",
        "2 3 7 129127 281243 5291569 1180663882621",
    ),
    ("265467277810504259544610", "2 5 47 7261921 77778868062803"),
    (
        "9806552073674839731751566707119",
        "7^2 3313 6827 7141683191 1238991535291",
    ),
    (
        "296482180290842832420459103",
        "463 27617 23186815418246299793",
    ),
    (
        "209239998611427172785905685913",
        "11 98887 230479 834605919667742771",
    ),
    (
        "4132322654023354937458785071311301",
        "359 41473589681 277541639146313966819",
    ),
    (
        "54157291246533783232524233601846897953",
        "11 4923390113321253021138566691076990723",
    ),
    (
        "33022016388926045152753737348897216",
        "2^6 3 19 2671 6089232713 556560395632256329",
    ),
    (
        "113996201920133514831986608080",
        "2^4 3^2 5 7 853 4030591 6578730314700149",
    ),
    (
        "4078433089868972532142698205555297536",
        "2^8 3^2 863 2051162515424317491139746989243",
    ),
    (
        "18808443664911410674656805135334",
        "2 9404221832455705337328402567667",
    ),
    (
        "444255910242277224483131643",
        "3^2 29 661 154382773 16679860341271",
    ),
    (
        "759470380630646750573021425",
        "5^2 23 31 41 53 2129 262369 35102123893",
    ),
    ("208843891390050004814617", "296027 74450659 9475931369"),
    (
        "103031755785389314605121315537",
        "4090649 4401851551 5721942540263",
    ),
    ("49568028695056515137407633", "820763 6974442583 8659132277"),
    ("74141930837787999595864339", "91917851 96249833 8380386433"),
    (
        "1323639552956736697734061307",
        "51673 9702384467 2640143786977",
    ),
    (
        "304106849580063350284799791349",
        "505979 66266512087 9069839408713",
    ),
    (
        "4466326433475946535072238912588672847",
        "61752538579 7243256673689 9985315017437",
    ),
    (
        "839988267464992231733732041",
        "92699003 964021801 9399641747",
    ),
    ("10451050656306628434991", "234673 5259641 8467217687"),
    ("4299425845620285731735633", "603503 97842133 72812361667"),
    ("15810231475787755043633", "70167924131 225319926043"),
    ("15010533442356604240063", "73200421709 205060750907"),
    ("51314406023608603351447", "91361278553 561664709999"),
    ("9670708030124741805077", "22771274243 424688927239"),
    ("41381525298667941846101", "55569950041 744674509661"),
    ("9435752088470380792183", "17799328369 530118434407"),
    (
        "12149582566768317741367228102235854005409",
        "92347 133108237277 988402040086988011452911",
    ),
    (
        "2560881148031395289396437363010290415414653",
        "519855491 62891456153 78327657264518085985111",
    ),
    (
        "52204172639591156683000576283156465519507692310879903",
        "63781 622187 1315506385196060090521925559645293342317849",
    ),
    (
        "228958400082557933560744937275147543582418660266069481300623",
        "55219 938323 4418915352605942504061563081853270722907037816279",
    ),
    (
        "3370543074631558151957633472976686387185319833840865694399993",
        "59235509 6610899528899 8607106964195769806849946422181830128423",
    ),
    (
        "3176788792890297379485856069659265241107733",
        "8976043 58330704263099 6067449880279414741069",
    ),
    (
        "67481914370608283547571765040540647045441921",
        "15107 90654379 49274290317034503548333307849857",
    ),
    (
        "23280246238438539228851742915617923592562363779",
        "468686812673 4516060411393 10998794623210642868611",
    ),
    (
        "66627077388721449143081378796215927134921",
        "330507433 327593808709 615366495658378075693",
    ),
    (
        "24122016045478132088148285829865464553044495793",
        "306425409811 6273329386517 12548468802378557819839",
    ),
    (
        "19233487136489335782432672696407560830274050109551214744234067",
        "14528737 114590591371 11552639154780312475219645328949048936467321",
    ),
    (
        "3352911659185889450441531953789608309643",
        "19841 597080098211 283025755472885113980793",
    ),
    (
        "5154342225911292269639293077131310231045659773",
        "81047 78410713 811074791595885087075267263411843",
    ),
    (
        "82339780323268256851855801376996239406854631765518324314070199235109",
        "3202978543 61510236379 417934553493460580510472163142969338230887072497",
    ),
    (
        "48518031585305937675493035547365444694326151931369467",
        "197^2 8897993237687 140500752158173605406834196972619749",
    ),
    (
        "288588106283719778405693625429282787111441805027195158913",
        "761^2 895377058073 556548674029235161778960147563005344761",
    ),
    (
        "1195389565827172110733475214962310901810284631147571307373",
        "149^2 68534051699689 785652534768906912859209796894149980557",
    ),
    (
        "7141928913202104742427245010431814499634247051745499",
        "373^2 2439776733983 21040098760544824629998122601808557",
    ),
    (
        "40012375036902807777728942548672600132828357133",
        "967^2 4654846307117 9192550757458382079037579441",
    ),
    (
        "104212284386088603737239513041015515049159219437814320345077",
        "337^2 927231784513163 989625332473184427325293912519998575391",
    ),
    (
        "54274377545445552582717585656244653864116714549",
        "2222895109^5",
    ),
    (
        "22930288869936329361993810533801686827941299271028319128969287401",
        "7448724388681^5",
    ),
    (
        "5310555140814041981529516848719569143230418309",
        "1744670057000669^3",
    ),
    (
        "71730457472741870030805959993542659588762878730104170721492692767121",
        "92029253694568819^4",
    ),
    (
        "83000174027482898609680100232435412900371117648481",
        "3018351061447^4",
    ),
    (
        "6493943457456984033340512642220736923523204114806227601",
        "91727952721^5",
    ),
    (
        "439964517042523262136249249504913401403",
        "61441^3 55584821^2 613946323",
    ),
    (
        "5233103940103855029621326609004086543",
        "10771^3 71458969^2 820120733",
    ),
    (
        "7555890972112317049400054837018299597",
        "22031^3 27616727^2 926485523",
    ),
    (
        "3767910332745393521564215839620353987",
        "14551^3 39028601^2 802890037",
    ),
    (
        "101679940869728376894896164792949268497",
        "66959^3 29698979^2 383994823",
    ),
    (
        "204517349995104002304057802907835394429",
        "55441^3 46317881^2 559422109",
    ),
    ("18446744073709551617", "274177 67280421310721"),
    (
        "340282366920938463463374607431768211457",
        "59649589127497217 5704689200685129054721",
    ),
    ("18446744073709551615", "3 5 17 257 641 65537 6700417"),
    (
        "170141183460469231731687303715884105727",
        "170141183460469231731687303715884105727",
    ),
    (
        "1000000000000000000000000000001",
        "61 101 3541 9901 27961 4188901 39526741",
    ),
    ("561", "3 11 17"),
    ("41041", "7 11 13 41"),
    ("3825123056546413051", "149491 747451 34233211"),
    ("318665857834031151167461", "399165290221 798330580441"),
    (
        "1606938044258990275541962092341162602522202993782792835301376",
        "2^200",
    ),
    ("515377520732011331036461129765621272702107522001", "3^100"),
    (
        "24647906487115793512432470614609487044327490547070674282967249490409801198254927547005559122946385681862066942895590",
        "2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 101 103 107 109 113 127 131 137 139 149 151 157 163 167 173 179 181 191 193 197 199 211 223 227 229 233 239 241 251 257 263 269 271 277 281",
    ),
    ("224667480714346638478167484328312832", "2^40 3^40 7^5"),
    ("18446744073709551557", "18446744073709551557"),
    ("18446744073709551615", "3 5 17 257 641 65537 6700417"),
    ("4611686014132420609", "2147483647^2"),
    ("1000000016000000063", "1000000007 1000000009"),
    (
        "4951760154835678088235319297",
        "2147483647 2305843009213693951",
    ),
    (
        "5316911983139663487003542222693990401",
        "2305843009213693951^2",
    ),
];

/// `(n, k, b)`: `n = b^k` with the largest `k`, by PARI/GP's `ispower`, or `k = 0`.
const POWERS: &[(&str, u32, &str)] = &[
    (
        "44154535253457332848558314462661076432562199491546797602919936321546379739",
        13,
        "462379",
    ),
    (
        "44154535253457332848558314462661076432562199491546797602919936321546379740",
        0,
        "0",
    ),
    (
        "-44154535253457332848558314462661076432562199491546797602919936321546379739",
        13,
        "-462379",
    ),
    (
        "785515537355640579259976813881047011302165639211758870555707197085177607439095536274690659443631371580169719718992931045812931947027",
        7,
        "6952883755627037883",
    ),
    (
        "785515537355640579259976813881047011302165639211758870555707197085177607439095536274690659443631371580169719718992931045812931947028",
        0,
        "0",
    ),
    (
        "-785515537355640579259976813881047011302165639211758870555707197085177607439095536274690659443631371580169719718992931045812931947027",
        7,
        "-6952883755627037883",
    ),
    (
        "90737101766018779414714868179379964355043632628061413614300423",
        9,
        "7659463",
    ),
    (
        "90737101766018779414714868179379964355043632628061413614300424",
        0,
        "0",
    ),
    (
        "-90737101766018779414714868179379964355043632628061413614300423",
        9,
        "-7659463",
    ),
    ("285311670611", 11, "11"),
    ("285311670612", 0, "0"),
    ("-285311670611", 11, "-11"),
    (
        "2631487640079518470148167347543432587832448233206694149705129988245623757234482511705223479435655791434827126738178500478515625",
        10,
        "4385510280835",
    ),
    (
        "2631487640079518470148167347543432587832448233206694149705129988245623757234482511705223479435655791434827126738178500478515626",
        0,
        "0",
    ),
    (
        "-2631487640079518470148167347543432587832448233206694149705129988245623757234482511705223479435655791434827126738178500478515625",
        5,
        "-19232700423309480568297225",
    ),
    (
        "581588370955498542289339357131468100",
        2,
        "762619414226715410",
    ),
    ("581588370955498542289339357131468101", 0, "0"),
    ("-581588370955498542289339357131468100", 0, "0"),
    ("34493382010346358640265627904", 2, "185723940326352"),
    ("34493382010346358640265627905", 0, "0"),
    ("-34493382010346358640265627904", 0, "0"),
    (
        "483658388600742559161345516606208767491408047538394532602108303634526242941255994016027819018993871706586057488789135867748420230578176000000000",
        9,
        "9224625572500760",
    ),
    (
        "483658388600742559161345516606208767491408047538394532602108303634526242941255994016027819018993871706586057488789135867748420230578176000000001",
        0,
        "0",
    ),
    (
        "-483658388600742559161345516606208767491408047538394532602108303634526242941255994016027819018993871706586057488789135867748420230578176000000000",
        9,
        "-9224625572500760",
    ),
    ("340068392", 3, "698"),
    ("340068393", 0, "0"),
    ("-340068392", 3, "-698"),
    (
        "752754691792290107065530089130360995134914574555619784576942238738073485428231576812795824293694402516128199382930949153227086095224806935380042497778087947665408",
        9,
        "968935415616300528",
    ),
    (
        "752754691792290107065530089130360995134914574555619784576942238738073485428231576812795824293694402516128199382930949153227086095224806935380042497778087947665409",
        0,
        "0",
    ),
    (
        "-752754691792290107065530089130360995134914574555619784576942238738073485428231576812795824293694402516128199382930949153227086095224806935380042497778087947665408",
        9,
        "-968935415616300528",
    ),
    (
        "8330098173511823794486751826210655402880765696840142051182788174365092435798286722542438332710236755559891385796923449419659700687019632666694345992883816598958923776",
        13,
        "5795980721926",
    ),
    (
        "8330098173511823794486751826210655402880765696840142051182788174365092435798286722542438332710236755559891385796923449419659700687019632666694345992883816598958923777",
        0,
        "0",
    ),
    (
        "-8330098173511823794486751826210655402880765696840142051182788174365092435798286722542438332710236755559891385796923449419659700687019632666694345992883816598958923776",
        13,
        "-5795980721926",
    ),
    (
        "9705135682386608561435241504563592180506347301091887841097593913154503705572303538611010467305689719573506672470019944749502124802979529156732698471159568120245530395026025195941497595708784627036190477169035098720256",
        11,
        "53221983449786888706",
    ),
    (
        "9705135682386608561435241504563592180506347301091887841097593913154503705572303538611010467305689719573506672470019944749502124802979529156732698471159568120245530395026025195941497595708784627036190477169035098720257",
        0,
        "0",
    ),
    (
        "-9705135682386608561435241504563592180506347301091887841097593913154503705572303538611010467305689719573506672470019944749502124802979529156732698471159568120245530395026025195941497595708784627036190477169035098720256",
        11,
        "-53221983449786888706",
    ),
    ("146703694354210294777950037144843564357738177", 13, "2497"),
    ("146703694354210294777950037144843564357738178", 0, "0"),
    (
        "-146703694354210294777950037144843564357738177",
        13,
        "-2497",
    ),
    ("2799360000000", 7, "60"),
    ("2799360000001", 0, "0"),
    ("-2799360000000", 7, "-60"),
    (
        "61618612218570293834137045316562509732249600000000000",
        11,
        "62960",
    ),
    (
        "61618612218570293834137045316562509732249600000000001",
        0,
        "0",
    ),
    (
        "-61618612218570293834137045316562509732249600000000000",
        11,
        "-62960",
    ),
    (
        "751609673044497994748580104817062753466899084415989054911467749376",
        11,
        "974376",
    ),
    (
        "751609673044497994748580104817062753466899084415989054911467749377",
        0,
        "0",
    ),
    (
        "-751609673044497994748580104817062753466899084415989054911467749376",
        11,
        "-974376",
    ),
    (
        "81182780851246532149903262384195830403154730868166098944",
        6,
        "2080865392",
    ),
    (
        "81182780851246532149903262384195830403154730868166098945",
        0,
        "0",
    ),
    (
        "-81182780851246532149903262384195830403154730868166098944",
        3,
        "-4330000779623313664",
    ),
    ("14074678408802364030976", 10, "164"),
    ("14074678408802364030977", 0, "0"),
    ("-14074678408802364030976", 5, "-26896"),
    (
        "34081204571864977553462373666428659338588081715882408059214167851363291038297882696501259",
        13,
        "6459419",
    ),
    (
        "34081204571864977553462373666428659338588081715882408059214167851363291038297882696501260",
        0,
        "0",
    ),
    (
        "-34081204571864977553462373666428659338588081715882408059214167851363291038297882696501259",
        13,
        "-6459419",
    ),
    ("28115122576740398332555955012147716892578125", 9, "67245"),
    ("28115122576740398332555955012147716892578126", 0, "0"),
    ("-28115122576740398332555955012147716892578125", 9, "-67245"),
    ("7317420470624", 5, "374"),
    ("7317420470625", 0, "0"),
    ("-7317420470624", 5, "-374"),
    (
        "31980846645576185338011808048353523017045329738625255126953125",
        13,
        "53845",
    ),
    (
        "31980846645576185338011808048353523017045329738625255126953126",
        0,
        "0",
    ),
    (
        "-31980846645576185338011808048353523017045329738625255126953125",
        13,
        "-53845",
    ),
    ("2015025341112107008", 7, "412"),
    ("2015025341112107009", 0, "0"),
    ("-2015025341112107008", 7, "-412"),
    (
        "34082197648551476743346931891091903589707380371891112049455457606432",
        5,
        "32100035901682",
    ),
    (
        "34082197648551476743346931891091903589707380371891112049455457606433",
        0,
        "0",
    ),
    (
        "-34082197648551476743346931891091903589707380371891112049455457606432",
        5,
        "-32100035901682",
    ),
    (
        "53383737566401264969996652764948088444550711697001256718414068264869386848797606264391903629996759824932128674742053839790268391159296",
        9,
        "722106072916406",
    ),
    (
        "53383737566401264969996652764948088444550711697001256718414068264869386848797606264391903629996759824932128674742053839790268391159297",
        0,
        "0",
    ),
    (
        "-53383737566401264969996652764948088444550711697001256718414068264869386848797606264391903629996759824932128674742053839790268391159296",
        9,
        "-722106072916406",
    ),
    (
        "5939627240174646443633393071841648646417029698587767711072265625",
        9,
        "12189145",
    ),
    (
        "5939627240174646443633393071841648646417029698587767711072265626",
        0,
        "0",
    ),
    (
        "-5939627240174646443633393071841648646417029698587767711072265625",
        9,
        "-12189145",
    ),
    (
        "35333192156192904368555935872105362687904140213295295469952419989227729127086422244594264767346367520990230380405644795617560629645716257450178084045096802256486372741747",
        9,
        "6897424989204117267",
    ),
    (
        "35333192156192904368555935872105362687904140213295295469952419989227729127086422244594264767346367520990230380405644795617560629645716257450178084045096802256486372741748",
        0,
        "0",
    ),
    (
        "-35333192156192904368555935872105362687904140213295295469952419989227729127086422244594264767346367520990230380405644795617560629645716257450178084045096802256486372741747",
        9,
        "-6897424989204117267",
    ),
    (
        "8348001271297696526088602562600292677000",
        3,
        "20285894794530",
    ),
    ("8348001271297696526088602562600292677001", 0, "0"),
    (
        "-8348001271297696526088602562600292677000",
        3,
        "-20285894794530",
    ),
    (
        "2691918069838996379568128624793571275189573295891751",
        5,
        "19320233351",
    ),
    (
        "2691918069838996379568128624793571275189573295891752",
        0,
        "0",
    ),
    (
        "-2691918069838996379568128624793571275189573295891751",
        5,
        "-19320233351",
    ),
    ("890966078479669305969216", 2, "943909994904"),
    ("890966078479669305969217", 0, "0"),
    ("-890966078479669305969216", 0, "0"),
    (
        "27083651664919099182736322546391824808437381216678302444191234547966576008898405749967182810140022048880120500424831203604285570723083172158151072212255578155114747455582526817195581399714760259480434709326272553141397028605429039421631105842592587646208229837710530167439917697803646925812107167335965150944698142055794365562286007535135196664711173914418464601493061236740966660008860082538996467607",
        31,
        "8263881935143",
    ),
    (
        "58161699078541740737816416564030165177000008594758999375299108776690893629658428841323701621402385940329515486931902062900361925850168648353598297489486039284894927724513082654448996273740399521474470514209803404662361150805947106182300177285124843896238982779509057291987925278499137287946969262277584883951591012360799503695543526680907620306731384624097804160973185391363882714082202562372425236515794190336",
        31,
        "16527763870286",
    ),
    (
        "502220736674773030108876782399550601610543777250918170361130414601025085031014065667108910009504549090404534641",
        6,
        "2819350735836741559",
    ),
    (
        "32142127147185473926968114073571238503074801744058762903112346534465605441984900202694970240608291141785890217024",
        6,
        "5638701471673483118",
    ),
    (
        "15509648561954330273394876915250235055506468341393964566879677450013855762237433515054883136686128592702401",
        20,
        "203953",
    ),
    (
        "16263045250499823820755306456285430473562750547545517789680424661825728819743879085482189139933793975221512830976",
        20,
        "407906",
    ),
    (
        "14085720534914086258747849673051933862527827081114130408797962727505693699644045063847085410992964005834823237063862987589927587195790118953876650035590943103856483433956503054841849322338117864067002011670333290155194126865711579886484162536928913362321",
        12,
        "1246612820463602340811",
    ),
    (
        "236319175929889166605644563500321673629343660950501286520571721039312164428767267149896342910677731606116089770039635137201534554742605116354882595003516940097110655571909986355741551920320228438910731222227702400924365388357446169456800275461164356124945678336",
        12,
        "4986451281854409363244",
    ),
    (
        "24681277935730281568814189967538713442963177924455137156346100055486796501073624211839636608292038742382934197082105253947933725644044626682820678515007304212601349620059889214863323070892959890371452891160925264693448362462701714780720645724404305899105371741881556436416840606658519920094017052069153141885248840580452584883936845584061176777",
        21,
        "22491452690318377",
    ),
    (
        "108549408314826699996086803877181517671877028370190308050637761121750606163884487555118095594637996867276463600627364549157017844138606805777657629956593885997307727568506111732962395852152373499683978710569801200597508282615867274021186544331078372268275146024247130521588361041144978326085976559517022079808298552437720585170096852493255447675381437431808",
        21,
        "89965810761273508",
    ),
    (
        "4173299187230085185471924628832841501208474243799715508872071180515698691937544735431518680650039238679425208085795461841222839480710411888487842574285139348797208167009411443029999530809249",
        25,
        "38443169",
    ),
    (
        "4698717166128751365035395804565076154369164274611810175473298589115754295187664670088323447129371194489535069370839536960344697134883495453063874543655616271322119146939227379340939134689887877679206629376",
        25,
        "153772676",
    ),
    (
        "92065739124233162031208151231267590036102170653305528110859550871799013853226102154100040937160778913511953644762893282934889846292823047957823442046330614148974567784727232044508761269523187457291303807034280278902942559922941493671212677982872932098389028004915557021179434628893363132899426042501583770376613109654813436822313409872145173284522280931130844136102077734542686946170049641093364220030315529108293478198315569313189888343116402820006564007283037387551319121",
        28,
        "77906721779427637",
    ),
    (
        "24713708663790569500169246306682276989362142641909887345759402090226565824041065802838426758585421032963765838082789270283964174199383864405868204433196331535840040035696164369485521627379595727671471662275783034299118565814722124714953089287313655717888095597896277790474303192429180707953446031857188020123245431825767847508324891153944191288829756223908212741339487299219413121860404728949533562977922082880066033201990898280485734491973536052068147932288209462192387071647154176",
        28,
        "155813443558855274",
    ),
    ("23583969525480400447096695203401", 2, "4856332929843299"),
    ("94335878101921601788386780813604", 2, "9712665859686598"),
    (
        "137432420016352424426768566428149860750027472070286457528945410676400845190714437910448547909292846784299558539670775714476107",
        9,
        "80210667902507",
    ),
    (
        "70365399048372441306505506011212728704014065699986666254820050266317232737645792210149656529557937553561373972311437165811766784",
        9,
        "160421335805014",
    ),
    (
        "3705687387866615050659429832828428163873042588702328298946134742521267775149670247198303570595705662215972733334891984289",
        18,
        "4991953",
    ),
    (
        "3705687387866615050659429832828428163873042588702328298946134742521267775149670247198303570595705662215972733334891984289",
        18,
        "4991953",
    ),
    ("9", 2, "3"),
    ("27", 3, "3"),
    ("81", 4, "3"),
    ("243", 5, "3"),
    ("729", 6, "3"),
    ("2187", 7, "3"),
    ("6561", 8, "3"),
    ("19683", 9, "3"),
    ("59049", 10, "3"),
    ("177147", 11, "3"),
    ("531441", 12, "3"),
    ("1594323", 13, "3"),
    ("4782969", 14, "3"),
    ("14348907", 15, "3"),
    ("43046721", 16, "3"),
    ("129140163", 17, "3"),
    ("387420489", 18, "3"),
    ("1162261467", 19, "3"),
    ("3486784401", 20, "3"),
    ("10460353203", 21, "3"),
    ("31381059609", 22, "3"),
    ("94143178827", 23, "3"),
    ("282429536481", 24, "3"),
    ("847288609443", 25, "3"),
    ("2541865828329", 26, "3"),
    ("7625597484987", 27, "3"),
    ("22876792454961", 28, "3"),
    ("68630377364883", 29, "3"),
    ("205891132094649", 30, "3"),
    ("617673396283947", 31, "3"),
    ("1853020188851841", 32, "3"),
    ("5559060566555523", 33, "3"),
    ("16677181699666569", 34, "3"),
    ("50031545098999707", 35, "3"),
    ("150094635296999121", 36, "3"),
    ("450283905890997363", 37, "3"),
    ("1350851717672992089", 38, "3"),
    ("4052555153018976267", 39, "3"),
    ("12157665459056928801", 40, "3"),
    ("-25", 0, "0"),
    ("-125", 3, "-5"),
    ("-625", 0, "0"),
    ("-3125", 5, "-5"),
    ("-15625", 3, "-25"),
    ("-78125", 7, "-5"),
    ("-390625", 0, "0"),
    ("-1953125", 9, "-5"),
    ("-9765625", 5, "-25"),
    ("-48828125", 11, "-5"),
    ("-244140625", 3, "-625"),
    ("-1220703125", 13, "-5"),
    ("-6103515625", 7, "-25"),
    ("-30517578125", 15, "-5"),
    ("-152587890625", 0, "0"),
    ("-762939453125", 17, "-5"),
    ("-3814697265625", 9, "-25"),
    ("-19073486328125", 19, "-5"),
    ("-95367431640625", 5, "-625"),
    ("18446744073709551616", 64, "2"),
    ("9223372036854775808", 63, "2"),
    ("-9223372036854775808", 63, "-2"),
    ("-18446744073709551616", 0, "0"),
    (
        "56347514353166785389812313795980500551139163800306781874894667776",
        60,
        "12",
    ),
    (
        "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        100,
        "10",
    ),
    (
        "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001",
        0,
        "0",
    ),
    (
        "-1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        99,
        "-10",
    ),
    (
        "103484181201460257481845061843772979245337029322553604971787047827566501435983811410283166531781519500536320912318075927926202052878944143265784718301804477764199660433928737919460457671030990107285867934120228086626809621067714254005827595921697330537794181136541580466104041770180794149723112108192624674020138340761577489122082030647218253580429066169829147497350411235444845631551671576996523201981376436968450011017545386868199952514882343217473444542085549916004055218979645382672026768803301532446171773183966696073512881390446428998778414064748695500621683804219428699188533071992899340799517950924800569385935196380855455973376",
        100,
        "2239488",
    ),
    (
        "15777552999289695909107946343832274624190256705734689813595709169009395191056955307530759659431514051288948466145675153520852105530438184090567283873528484795830528350885244181199715519954244992811356128554137452736967984183199536299156138657108052654151488877985135733056003592238782389005021053259318466301270214639621799475798018376482491279802823541647866172519485998460964581588422721747566046797433634231959762299160471674995919414543668748680856668007",
        541,
        "7",
    ),
    (
        "726838724295606894134906371759902150251652000283696807867511209892164242991992458452633714849478906377841451102969902963894734450161509",
        7,
        "18446744073709551629",
    ),
    ("1", 0, "0"),
    ("4", 2, "2"),
    ("8", 3, "2"),
    ("-8", 3, "-2"),
    ("-4", 0, "0"),
    ("12", 0, "0"),
];

/// `(n, bound, primes, cofactor)`: the primes up to `bound` dividing `n`, by PARI/GP.
const TRIAL: &[(&str, u64, &str, &str)] = &[
    (
        "44203728487512241805244733735278700814165417893392791179685812550144",
        8192,
        "2^9 17^1 127^1 787^1",
        "50811447486554817825248889037359766732602351801202610550689",
    ),
    (
        "185740866390907464575929754157037623005595338282854120",
        1000000,
        "2^3 5^1 13^1 139^1 131113^1",
        "19599435745588019665147824847486692632487583",
    ),
    (
        "73346610372295493466274716765786676868",
        100000,
        "2^2 491^1 2111^1",
        "17690916451671415045975526498717",
    ),
    (
        "38664559788631320176548423374273508568446904068848179228668240",
        8191,
        "2^4 5^1 13^1",
        "37177461335222423246681176321416835161968176989277095412181",
    ),
    ("7118970423133540", 8192, "2^2 5^1 43^1", "8277872585039"),
    (
        "18097223399440924691121828943272880523914991805749007359932252678630",
        1000000,
        "2^1 5^1 607^1",
        "2981420658886478532310021242713818867201810841144811756166763209",
    ),
    (
        "5009808768735349732209117782920274627122171356147816642226",
        1000000,
        "2^1 17^1 47^1 89^1 101^1",
        "348765078659994097375384667960901916354321208061683",
    ),
    (
        "1351547945674674330354606330289612841066636401399785093",
        1000,
        "3^1",
        "450515981891558110118202110096537613688878800466595031",
    ),
    (
        "21875049860018028220654488190263711293318515748650529254400",
        1000000,
        "2^10 5^2 5519^1 35027^1",
        "4420238977486615144497134540020004770141803023",
    ),
    (
        "4037424838098365975995397758096780766010965220",
        1000,
        "2^2 3^1 5^1",
        "67290413968306099599923295968279679433516087",
    ),
    (
        "113387452400679595985612921333607616",
        8192,
        "2^6 11^1 241^1",
        "668305901079071553102678949769",
    ),
    (
        "276908901125273701686997710872043088545555281922354893964404575704",
        1000000,
        "2^3 3^1 7^2 19^1 193^1 229^1 2377^1 6257^1",
        "18853321151052670964421739102128327057394934881727",
    ),
    (
        "18944622821819950511719336",
        100000,
        "2^3",
        "2368077852727493813964917",
    ),
    (
        "62174917086205286992",
        100000,
        "2^4 157^1",
        "24751161260432041",
    ),
    (
        "31918665366581725581760493209002849635676762000",
        100,
        "2^4 3^2 5^3",
        "1773259187032318087875582956055713868648709",
    ),
    (
        "495431484490008994325962824590770730311556256",
        1000000,
        "2^5 3^2 659^1",
        "2610391820993556073627775799774335748143",
    ),
    (
        "17650437187264359457098275773591808",
        8191,
        "2^8 29^2",
        "81982188184008803958727871273",
    ),
    (
        "14185558084328943162298953528567",
        1000,
        "3^1",
        "4728519361442981054099651176189",
    ),
    (
        "91757309148053576080",
        8191,
        "2^4 5^1 17^1 787^1 2777^1 3821^1",
        "8079307",
    ),
    (
        "52334665852843774521207939128466192723721786",
        1000000,
        "2^1",
        "26167332926421887260603969564233096361860893",
    ),
    (
        "16374283353283386624446839557261772838647",
        1000,
        "281^1",
        "58271471008125931047853521556091718287",
    ),
    (
        "17765163035635382041142569642169893758421",
        1000,
        "149^1",
        "119229282118358268732500467397113380929",
    ),
    (
        "109283969004643669116938107357940943018517",
        10000,
        "727^1 7507^1",
        "20024221135861214378169207567286753",
    ),
    (
        "153461280989816341601281425213635067461383",
        100000,
        "761^1 3023^1 87407^1",
        "763184955403510481038239013823",
    ),
    (
        "89628961633204997595097967124733251006403",
        10000,
        "331^1 3361^1",
        "80566010541393141692919733395356233",
    ),
    (
        "58751018359478150548159708149157730948471",
        1000,
        "773^1",
        "76003904734124386220128988550010001227",
    ),
    (
        "33380202747578157026265325141535290056161",
        10000,
        "311^1 1399^1",
        "76720401452526165971250307733671249",
    ),
    (
        "1973447222973047526480002990201762798861",
        1000,
        "373^1",
        "5290743225128813743914217131908211257",
    ),
    ("809656219", 1000000, "8171^1 99089^1", "1"),
    ("67008437", 1000000, "4799^1 13963^1", "1"),
    ("691018331", 1000000, "8461^1 81671^1", "1"),
    ("395828267", 1000000, "5477^1 72271^1", "1"),
    ("433574149", 1000000, "6659^1 65111^1", "1"),
    ("534883747", 1000000, "7873^1 67939^1", "1"),
    ("277648213", 1000000, "8269^1 33577^1", "1"),
    ("92445071", 1000000, "5743^1 16097^1", "1"),
    (
        "33701939481009429127230369172622805504895093548246432667259800010193199454748672",
        1000000,
        "2^100 3^50 8191^3 8209^2",
        "1000003",
    ),
    (
        "33701939481009429127230369172622805504895093548246432667259800010193199454748672",
        8191,
        "2^100 3^50 8191^3",
        "67387883163043",
    ),
    ("1", 100, "", "1"),
    ("2", 1, "", "2"),
    ("2", 2, "2^1", "1"),
    ("97", 96, "", "97"),
    ("97", 97, "97^1", "1"),
    ("1000006000009", 1000000, "", "1000006000009"),
    ("1000006000009", 1000003, "1000003^2", "1"),
    ("18446744073709551557", 100000, "", "18446744073709551557"),
    ("4295229443", 65537, "65537^1", "65539"),
    ("4295229443", 65539, "65537^1 65539^1", "1"),
];

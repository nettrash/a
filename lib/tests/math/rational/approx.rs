//! Continued fractions, convergents, best approximations, the simplest rational in an interval and
//! rational reconstruction, against their definitions by brute force on small values, against
//! plain Euclid's algorithm on numbers, and by their properties on long parts of both byte orders.

use super::common::{below, check, integer, order, rational, size, xorshift};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use std::cmp::Ordering;

/// `p/q` from primitive parts.
fn q(p: i64, d: i64) -> Rational {
    Rational::new(Number::from(p), Number::from(d))
}

/// The terms of `p/d`, `d > 0`, by Euclid's algorithm on `i128`s.
fn cf_i128(p: i128, d: i128) -> Vec<i128> {
    let mut out = vec![p.div_euclid(d)];
    let (mut x, mut y) = (d, p.rem_euclid(d));
    while y != 0 {
        out.push(x / y);
        (x, y) = (y, x % y);
    }
    out
}

/// The terms of `x` by Euclid's algorithm on numbers, one division a term.
fn cf_plain(x: &Rational) -> Vec<Number> {
    let mut out = vec![x.numer().div_floor(x.denom())];
    let (mut a, mut b) = (x.denom().clone(), x.numer().rem_euclid(x.denom()));
    while !b.is_zero() {
        let (q, r) = a.div_rem(&b);
        out.push(q);
        (a, b) = (b, r);
    }
    out
}

/// Asserts that `terms` is the canonical expansion of `x`, in its byte order: `a0 = ⌊x⌋`, the
/// others positive, the last at least 2 unless there is one, and the value `x`.
#[track_caller]
fn canonical(x: &Rational, terms: &[Number]) {
    assert!(!terms.is_empty());
    assert_eq!(terms[0], x.floor(), "a0 of {x}");
    for t in terms {
        assert!(
            t.byte_order() == x.byte_order(),
            "a term in another byte order: {x}"
        );
    }
    for t in &terms[1..] {
        assert!(t.is_positive(), "a term not positive: {x}");
    }
    if terms.len() > 1 {
        assert!(*terms.last().unwrap() >= 2u8, "the last term is 1: {x}");
    }
    let back = Rational::from_continued_fraction(terms);
    check(&back);
    assert_eq!(back, *x, "the terms of {x} give {back}");
    assert!(back.byte_order() == x.byte_order());
}

/// A rational of parts of about `bytes` bytes each, of a random sign and byte order.
fn long(state: &mut u64, bytes: usize) -> Rational {
    loop {
        let mut v: Vec<u8> = (0..bytes).map(|_| xorshift(state) as u8).collect();
        v[bytes - 1] |= 1;
        let mut w: Vec<u8> = (0..bytes - below(state, 3) as usize)
            .map(|_| xorshift(state) as u8)
            .collect();
        let last = w.len() - 1;
        w[last] |= 1;
        let o = order(xorshift(state));
        let neg = below(state, 2) == 1;
        let n = super::common::number(&v, neg, o);
        let d = super::common::number(&w, false, o);
        let r = Rational::new(n, d);
        if !r.is_integer() {
            return r;
        }
    }
}

#[test]
fn continued_fraction_exhaustive_small() {
    let lim = size(80, 12) as i64;
    for d in 1..=lim {
        for p in -lim..=lim {
            let x = q(p, d);
            let terms = x.continued_fraction();
            let expected = cf_i128(i128::from(p), i128::from(d));
            let got: Vec<i128> = terms
                .iter()
                .map(|t| i128::try_from(t.clone()).unwrap())
                .collect();
            assert_eq!(got, expected, "{x}");
            canonical(&x, &terms);
        }
    }
}

#[test]
fn continued_fraction_words() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..size(4000, 60) {
        let p = xorshift(&mut s) as i64 >> below(&mut s, 64);
        let d = ((xorshift(&mut s) >> 1) >> below(&mut s, 63)).max(1) as i64;
        let x = q(p, d);
        let terms = x.continued_fraction();
        let got: Vec<i128> = terms
            .iter()
            .map(|t| i128::try_from(t.clone()).unwrap())
            .collect();
        let n: i64 = x.numer().clone().try_into().unwrap();
        let dd: i64 = x.denom().clone().try_into().unwrap();
        assert_eq!(got, cf_i128(i128::from(n), i128::from(dd)), "{x}");
        canonical(&x, &terms);
    }
}

#[test]
fn continued_fraction_against_plain_euclid() {
    let mut s = 0xDEAD_BEEF_1234_5678u64;
    for i in 0..size(300, 30) {
        let x = rational(&mut s, if i % 3 == 0 { 64 } else { 24 });
        let terms = x.continued_fraction();
        assert_eq!(terms, cf_plain(&x), "{x:?}");
        canonical(&x, &terms);
    }
}

#[test]
fn continued_fraction_long() {
    // Lengths across Lehmer's method and the half gcd (600 limbs; 6 under Miri).
    let mut s = 0x0123_4567_89AB_CDEFu64;
    let sizes: &[usize] = if cfg!(miri) {
        &[20, 60, 100]
    } else {
        &[40, 200, 1000, 4800, 5200, 9000, 20000]
    };
    for &bytes in sizes {
        for _ in 0..size(3, 1) {
            let x = long(&mut s, bytes);
            let terms = x.continued_fraction();
            canonical(&x, &terms);
            if bytes <= 1000 {
                assert_eq!(terms, cf_plain(&x));
            }
        }
    }
}

#[test]
fn continued_fraction_special() {
    // Consecutive Fibonacci numbers: all quotients 1, Euclid's worst case.
    let (mut a, mut b) = (Number::from(1u8), Number::from(1u8));
    for _ in 0..size(3000, 200) {
        (a, b) = (&a + &b, a);
    }
    let x = Rational::new(a.clone(), b.clone());
    let terms = x.continued_fraction();
    assert!(terms[..terms.len() - 1].iter().all(|t| *t == 1u8));
    canonical(&x, &terms);
    // A huge quotient in the middle: [1; 2, 10^200, 3].
    let big: Number = format!("1{}", "0".repeat(size(200, 40))).parse().unwrap();
    let t = vec![Number::from(1), Number::from(2), big, Number::from(3)];
    let x = Rational::from_continued_fraction(&t);
    assert_eq!(x.continued_fraction(), t);
    // Integers, zero, and the byte orders.
    for o in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
        let n = Number::new_from_i32(-5, o);
        let x = Rational::from_integer(n.clone());
        let terms = x.continued_fraction();
        assert_eq!(terms, [n]);
        assert!(terms[0].byte_order() == o);
        let x = Rational::new(Number::new_from_i32(-1, o), Number::new_from_i32(3, o));
        let terms = x.continued_fraction();
        assert_eq!(terms, [Number::from(-1), Number::from(1), Number::from(2)]);
        assert!(terms.iter().all(|t| t.byte_order() == o));
    }
    assert_eq!(Rational::default().continued_fraction(), [Number::from(0)]);
}

#[test]
fn from_continued_fraction_forms() {
    // Non-canonical expansions are accepted.
    let t = |v: &[i64]| v.iter().map(|&x| Number::from(x)).collect::<Vec<_>>();
    assert_eq!(Rational::from_continued_fraction(&t(&[1, 1])), 2);
    assert_eq!(Rational::from_continued_fraction(&t(&[0, 1, 1])), q(1, 2));
    assert_eq!(Rational::from_continued_fraction(&t(&[-2, 1, 1])), q(-3, 2));
    assert_eq!(Rational::from_continued_fraction(&t(&[0, 2])), q(1, 2));
    assert_eq!(Rational::from_continued_fraction(&t(&[-1, 2])), q(-1, 2));
    // Random expansions against the recurrence on numbers, with terms past 64 bits.
    let mut s = 0x5555_AAAA_1234_9876u64;
    for _ in 0..size(300, 20) {
        let cap = if below(&mut s, 4) == 0 { 400 } else { 12 };
        let len = 1 + below(&mut s, cap) as usize;
        let mut terms = vec![integer(&mut s, 12)];
        for _ in 1..len {
            let mut t = match below(&mut s, 5) {
                0 => integer(&mut s, 20).abs(),
                _ => Number::from(below(&mut s, 6)),
            };
            if t.is_zero() {
                t = Number::from(1u8);
            }
            terms.push(t);
        }
        // x = a0 + 1/(a1 + …) by the recurrence from the back.
        let mut x = Rational::from_integer(terms[len - 1].clone());
        for t in terms[..len - 1].iter().rev() {
            x = x.recip() + Rational::from_integer(t.clone());
        }
        let got = Rational::from_continued_fraction(&terms);
        check(&got);
        assert_eq!(got, x);
        assert!(got.byte_order() == terms[0].byte_order());
    }
}

#[test]
#[should_panic(expected = "without terms")]
fn from_continued_fraction_empty() {
    Rational::from_continued_fraction(&[]);
}

#[test]
#[should_panic(expected = "not positive")]
fn from_continued_fraction_zero_term() {
    Rational::from_continued_fraction(&[Number::from(1), Number::from(0)]);
}

#[test]
#[should_panic(expected = "not positive")]
fn from_continued_fraction_negative_term() {
    Rational::from_continued_fraction(&[Number::from(1), Number::from(2), Number::from(-3)]);
}

#[test]
fn convergents_are_prefixes() {
    let mut s = 0x1357_9BDF_2468_ACE0u64;
    for i in 0..size(200, 20) {
        let x = if i % 10 == 0 {
            long(&mut s, size(300, 40))
        } else {
            rational(&mut s, 40)
        };
        let terms = x.continued_fraction();
        let c = x.convergents();
        assert_eq!(c.len(), terms.len());
        assert_eq!(*c.last().unwrap(), x);
        for (k, ck) in c.iter().enumerate() {
            check(ck);
            assert!(ck.byte_order() == x.byte_order());
            if k < 30 || k + 3 > c.len() {
                assert_eq!(
                    *ck,
                    Rational::from_continued_fraction(&terms[..=k]),
                    "{x} at {k}"
                );
            }
        }
    }
    let c: Vec<String> = q(-7, 3)
        .convergents()
        .iter()
        .map(|c| c.to_string())
        .collect();
    assert_eq!(c, ["-3", "-2", "-7/3"]);
    let c: Vec<String> = q(-1, 3)
        .convergents()
        .iter()
        .map(|c| c.to_string())
        .collect();
    assert_eq!(c, ["-1", "0", "-1/3"]);
}

/// The closest `p/q` to `x` with `q <= max`, ties to the smaller `q`, then the even `p`, by
/// scanning every denominator.
fn closest_brute(x: &Rational, max: u64) -> Rational {
    let mut best: Option<(Rational, Rational)> = None;
    for d in 1..=max {
        let f = (x * d).floor();
        for p in [f.clone(), f + 1u8] {
            let c = Rational::new(p.clone(), Number::from(d));
            if *c.denom() != d {
                continue;
            }
            let dist = (&c - x).abs();
            let better = match &best {
                None => true,
                Some((b, bd)) => match dist.cmp(bd) {
                    Ordering::Less => true,
                    Ordering::Greater => false,
                    Ordering::Equal => {
                        c.denom() < b.denom() || (c.denom() == b.denom() && !c.numer().bit(0))
                    }
                },
            };
            if better {
                best = Some((c, dist));
            }
        }
    }
    best.unwrap().0
}

/// [`Rational::limit_denominator`] by the convergents of the expansion, on numbers.
fn limit_reference(x: &Rational, max: &Number) -> Rational {
    if x.denom() <= max {
        return x.clone();
    }
    if *max == 1u8 {
        return Rational::from_integer(x.round_ties_even());
    }
    let terms = x.continued_fraction();
    let (mut p0, mut q0, mut p1, mut q1) = (
        Number::from(0u8),
        Number::from(1u8),
        Number::from(1u8),
        Number::from(0u8),
    );
    // (p1, q1) the last convergent with q <= max, (p0, q0) the one before.
    for t in &terms {
        let q2 = t * &q1 + &q0;
        if q2 > *max {
            break;
        }
        let p2 = t * &p1 + &p0;
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
    }
    let k = (max - &q0).div_floor(&q1);
    let semi = Rational::new(&p0 + &(&k * &p1), &q0 + &(&k * &q1));
    let conv = Rational::new(p1, q1);
    if (&conv - x).abs() <= (&semi - x).abs() {
        conv
    } else {
        semi
    }
}

#[test]
fn limit_denominator_brute_force() {
    let lim = size(40, 8) as i64;
    for d in 1..=lim {
        for p in -2 * lim..=2 * lim {
            let x = q(p, d);
            for max in 1..=(d as u64 + 1) {
                let got = x.limit_denominator(&Number::from(max));
                check(&got);
                assert_eq!(got, closest_brute(&x, max), "{x} to {max}");
            }
        }
    }
}

#[test]
fn limit_denominator_words() {
    let mut s = 0xFEED_FACE_CAFE_BEEFu64;
    for i in 0..size(3000, 40) {
        // Parts of one limb, and of two (the 128-bit path).
        let x = rational(&mut s, if i % 2 == 0 { 8 } else { 16 });
        let max = if i % 4 == 3 {
            let w = u128::from(xorshift(&mut s)) << 64 | u128::from(xorshift(&mut s));
            Number::from((w >> below(&mut s, 128)).max(1))
        } else {
            Number::from(1 + (xorshift(&mut s) >> below(&mut s, 64)))
        };
        let got = x.limit_denominator(&max);
        check(&got);
        assert!(got.denom() <= &max);
        assert_eq!(got, limit_reference(&x, &max), "{x} to {max}");
        if *got.denom() <= 30u8 && max <= 30u8 {
            let m: u64 = max.clone().try_into().unwrap();
            assert_eq!(got, closest_brute(&x, m));
        }
    }
}

#[test]
fn limit_denominator_long() {
    let mut s = 0x7777_3333_9999_1111u64;
    let sizes: &[usize] = if cfg!(miri) {
        &[16, 40, 90]
    } else {
        &[20, 40, 200, 1700, 4000, 12000]
    };
    for &bytes in sizes {
        for _ in 0..size(4, 1) {
            let x = long(&mut s, bytes);
            for _ in 0..size(4, 2) {
                let mb = 1 + below(&mut s, bytes as u64 + 4) as usize;
                let mut m: Vec<u8> = (0..mb).map(|_| xorshift(&mut s) as u8).collect();
                m[mb - 1] |= 1;
                let max = super::common::number(&m, false, order(xorshift(&mut s)));
                let got = x.limit_denominator(&max);
                check(&got);
                assert!(got.byte_order() == x.byte_order());
                assert!(got.denom() <= &max);
                assert_eq!(
                    got,
                    limit_reference(&x, &max),
                    "{bytes} bytes, bound of {mb}"
                );
            }
        }
    }
}

#[test]
fn limit_denominator_ties() {
    // Halves between integers to the even one; other ties to the smaller denominator.
    assert_eq!(q(5, 2).limit_denominator(&Number::from(1)), 2);
    assert_eq!(q(7, 2).limit_denominator(&Number::from(1)), 4);
    assert_eq!(q(-5, 2).limit_denominator(&Number::from(1)), -2);
    assert_eq!(q(1, 4).limit_denominator(&Number::from(2)), 0);
    assert_eq!(q(3, 4).limit_denominator(&Number::from(2)), 1);
    assert_eq!(q(-3, 4).limit_denominator(&Number::from(3)), q(-2, 3));
    let x = q(3, 7);
    assert_eq!(x.limit_denominator(&Number::from(7)), x);
    assert_eq!(x.limit_denominator(&Number::from(100)), x);
}

#[test]
#[should_panic(expected = "not positive")]
fn limit_denominator_zero() {
    q(1, 3).limit_denominator(&Number::from(0));
}

/// The simplest rational in `[a, b]`, `a <= b`, by scanning the denominators.
fn simplest_brute(a: &Rational, b: &Rational) -> Rational {
    for d in 1u64.. {
        let lo = (a * d).ceil();
        let hi = (b * d).floor();
        if lo <= hi {
            let p = if lo.is_positive() {
                lo
            } else if hi.is_negative() {
                hi
            } else {
                Number::from(0u8)
            };
            return Rational::new(p, Number::from(d));
        }
    }
    unreachable!()
}

/// The simplest rational in `[a, b]`, `a < b`, by the recursion on the integer part, `x = n +
/// 1/y` and `y` the simplest in `[1/(b - n), 1/(a - n)]`, taken as a loop on unreduced fractions.
fn simplest_recursive(a: &Rational, b: &Rational) -> Rational {
    let (c, f) = (a.ceil(), b.floor());
    if c <= f {
        let n = if c.is_positive() {
            c
        } else if f.is_negative() {
            f
        } else {
            Number::from(0u8)
        };
        return Rational::from_integer(n);
    }
    let mut terms = Vec::new();
    let (mut an, mut ad) = (a.numer().clone(), a.denom().clone());
    let (mut bn, mut bd) = (b.numer().clone(), b.denom().clone());
    loop {
        // [an/ad, bn/bd], an/ad not an integer and no integer up to bn/bd.
        let n = an.div_floor(&ad);
        terms.push(n.clone());
        (an, ad, bn, bd) = (bd.clone(), &bn - &(&n * &bd), ad.clone(), &an - &(&n * &ad));
        let (c, r) = an.div_rem(&ad);
        let c = if r.is_zero() { c } else { c + 1u8 };
        if c <= bn.div_floor(&bd) {
            terms.push(c);
            return Rational::from_continued_fraction(&terms);
        }
    }
}

#[test]
fn simplest_between_brute_force() {
    let lim = size(13, 5) as i64;
    let mut all = Vec::new();
    for d in 1..=lim {
        for p in -2 * lim..=2 * lim {
            let x = q(p, d);
            if !all.contains(&x) {
                all.push(x);
            }
        }
    }
    for a in &all {
        for b in &all {
            let got = Rational::simplest_between(a, b);
            check(&got);
            assert!(got.byte_order() == a.byte_order());
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            assert_eq!(got, simplest_brute(lo, hi), "[{a}, {b}]");
        }
    }
}

#[test]
fn simplest_between_random() {
    let mut s = 0x2468_1357_ACE0_BDF9u64;
    for i in 0..size(1500, 40) {
        let a = rational(&mut s, if i % 4 == 0 { 30 } else { 8 });
        // Close ends share long expansions.
        let b = match below(&mut s, 3) {
            0 => rational(&mut s, 8),
            1 => &a + &Rational::new(Number::from(1u8), integer(&mut s, 12).abs() + 1u8),
            _ => {
                let mut t = a.continued_fraction();
                let k = below(&mut s, t.len() as u64) as usize;
                t.truncate(k + 1);
                t.push(Number::from(1 + below(&mut s, 5)));
                Rational::from_continued_fraction(&t)
            }
        };
        let got = Rational::simplest_between(&a, &b);
        check(&got);
        let (lo, hi) = if a <= b { (&a, &b) } else { (&b, &a) };
        assert!(*lo <= got && got <= *hi);
        if lo != hi {
            assert_eq!(got, simplest_recursive(lo, hi), "[{a}, {b}]");
        } else {
            assert_eq!(got, a);
        }
    }
}

#[test]
fn simplest_between_long() {
    // Ends that agree on many terms: Lehmer's method on both, and past 600 limbs of common terms
    // (6 under Miri) both expansions in full.
    let mut s = 0x3141_5926_5358_9793u64;
    let cases: &[(usize, bool)] = if cfg!(miri) {
        &[(30, false), (90, true)]
    } else {
        &[(60, false), (300, false), (5000, false), (12000, true)]
    };
    for &(bytes, most) in cases {
        let x = long(&mut s, bytes);
        let mut t = x.continued_fraction();
        let k = if most { t.len() - 3 } else { t.len() / 2 };
        let y = {
            t.truncate(k + 1);
            t.push(Number::from(2));
            t.push(Number::from(5));
            Rational::from_continued_fraction(&t)
        };
        let got = Rational::simplest_between(&x, &y);
        check(&got);
        let (lo, hi) = if x <= y { (&x, &y) } else { (&y, &x) };
        assert!(*lo <= got && got <= *hi);
        assert!(got.continued_fraction().len() <= k + 3);
        // Nothing simpler: the closest rational to the middle with a smaller denominator is out of
        // the interval, as any one in it would be closer to the middle than its ends.
        if *got.denom() > 1u8 {
            let mid = (lo + hi) / 2u8;
            let below = mid.limit_denominator(&(got.denom() - 1u8));
            assert!(below < *lo || below > *hi);
        }
        if bytes <= 5000 {
            assert_eq!(got, simplest_recursive(lo, hi));
        }
    }
}

/// Every `p/q` with `|p| <= n`, `0 < q <= d`, in lowest terms, `gcd(q, m) = 1`, `p ≡ r q (mod m)`.
fn reconstructions(r: i64, m: i64, n: i64, d: i64) -> Vec<Rational> {
    let mut out = Vec::new();
    for qq in 1..=d {
        for p in -n..=n {
            let g = |mut x: i64, mut y: i64| {
                (x, y) = (x.abs(), y.abs());
                while y != 0 {
                    (x, y) = (y, x % y);
                }
                x
            };
            if g(p, qq) == 1 && g(qq, m) == 1 && (p - r * qq).rem_euclid(m) == 0 {
                out.push(q(p, qq));
            }
        }
    }
    out
}

#[test]
fn reconstruct_brute_force() {
    let lim = size(70, 14) as i64;
    for m in 1..=lim {
        for r in -2..m + 2 {
            for n in 0..=6 {
                for d in 1..=6 {
                    let got = Rational::reconstruct_with_bounds(
                        &Number::from(r),
                        &Number::from(m),
                        &Number::from(n),
                        &Number::from(d),
                    );
                    let all = reconstructions(r, m, n, d);
                    if 2 * n * d < m {
                        assert!(all.len() <= 1);
                        assert_eq!(got, all.first().cloned(), "{r} mod {m}, {n}, {d}");
                    } else if let Some(g) = &got {
                        assert!(all.contains(g), "{r} mod {m}, {n}, {d}: {g}");
                    }
                }
            }
            let got = Rational::reconstruct(&Number::from(r), &Number::from(m));
            let b = (((m - 1) / 2) as f64).sqrt() as i64;
            let all = if b >= 1 {
                reconstructions(r, m, b, b)
            } else {
                vec![]
            };
            assert_eq!(got, all.first().cloned(), "{r} mod {m}");
        }
    }
}

#[test]
fn reconstruct_known() {
    // p/q of random lengths modulo random moduli longer than 2 |p| q.
    let mut s = 0xABCD_EF01_2345_6789u64;
    let cases = size(400, 20);
    for i in 0..cases {
        let (pb, qb) = if i % 20 == 0 && !cfg!(miri) {
            (
                1 + below(&mut s, 3000) as usize,
                1 + below(&mut s, 3000) as usize,
            )
        } else {
            (
                1 + below(&mut s, 40) as usize,
                1 + below(&mut s, 40) as usize,
            )
        };
        let p = integer(&mut s, pb);
        let mut qq = integer(&mut s, qb).abs();
        if qq.is_zero() {
            qq = Number::from(1u8);
        }
        let x = Rational::new(p, qq);
        let (p, qq) = (x.numer().clone(), x.denom().clone());
        let bound = &(&p.abs() * &qq) * 2u8;
        let mut m = &bound + &integer(&mut s, 8).abs() + 1u8;
        while !(qq.gcd(&m) == 1u8) {
            m += 1u8;
        }
        let o = order(xorshift(&mut s));
        let r = {
            let v = (&p * &qq.mod_inverse(&m).unwrap()).rem_euclid(&m);
            if o == v.byte_order() {
                v
            } else {
                let mut w = v.value.clone();
                w.reverse();
                Number::new_signed(w, v.is_negative(), o)
            }
        };
        let got = Rational::reconstruct_with_bounds(&r, &m, &p.abs(), &qq).unwrap();
        check(&got);
        assert!(got.byte_order() == o);
        assert_eq!(got, x);
        // The same with r shifted by multiples of m.
        let shifted = &r - &(&m * 3u8);
        assert_eq!(
            Rational::reconstruct_with_bounds(&shifted, &m, &p.abs(), &qq),
            Some(x.clone())
        );
        // A modulus much longer: the default bounds find it.
        let mm = &(&bound * &bound) + &(&m * 7u8);
        let mm = if qq.gcd(&mm) == 1u8 { mm } else { continue };
        let r = (&p * &qq.mod_inverse(&mm).unwrap()).rem_euclid(&mm);
        assert_eq!(Rational::reconstruct(&r, &mm), Some(x.clone()));
        // Bounds too small: a smaller |p| or q leaves it out, and nothing else fits.
        if !p.is_zero() {
            let n = p.abs() - 1u8;
            assert_eq!(Rational::reconstruct_with_bounds(&r, &mm, &n, &qq), None);
        }
        if qq > 1u8 {
            let d = &qq - 1u8;
            assert_eq!(
                Rational::reconstruct_with_bounds(&r, &mm, &p.abs(), &d),
                None
            );
        }
    }
}

#[test]
fn reconstruct_random_residues() {
    // Random residues of long moduli: when a rational comes back it has every property.
    let mut s = 0x1111_2222_3333_4444u64;
    for i in 0..size(300, 20) {
        let mb = if i % 10 == 0 && !cfg!(miri) {
            4000
        } else {
            1 + below(&mut s, 60) as usize
        };
        let m = integer(&mut s, mb).abs() + 2u8;
        let r = integer(&mut s, mb + 2);
        let nb = 1 + below(&mut s, mb as u64) as usize;
        let n = integer(&mut s, nb).abs();
        let db = 1 + below(&mut s, mb as u64) as usize;
        let d = integer(&mut s, db).abs() + 1u8;
        if let Some(x) = Rational::reconstruct_with_bounds(&r, &m, &n, &d) {
            check(&x);
            assert!(x.numer().abs() <= n && *x.denom() <= d);
            assert!(x.denom().gcd(&m) == 1u8);
            assert_eq!((x.numer() - &(&r * x.denom())).rem_euclid(&m), 0u8);
        }
        if let Some(x) = Rational::reconstruct(&r, &m) {
            check(&x);
            assert_eq!((x.numer() - &(&r * x.denom())).rem_euclid(&m), 0u8);
            let two = &(&(x.numer().abs() * 2u8) * x.denom());
            assert!(*two < m);
        }
    }
}

#[test]
fn reconstruct_edges() {
    let one = Number::from(1);
    // Modulo 1 and 2 the default bounds are 0: no denominator allowed.
    assert_eq!(Rational::reconstruct(&Number::from(0), &one), None);
    assert_eq!(
        Rational::reconstruct(&Number::from(1), &Number::from(2)),
        None
    );
    assert_eq!(
        Rational::reconstruct(&Number::from(2), &Number::from(3)),
        Some(q(-1, 1))
    );
    assert_eq!(
        Rational::reconstruct_with_bounds(&Number::from(5), &one, &one, &one),
        Some(Rational::default())
    );
    assert_eq!(
        Rational::reconstruct_with_bounds(
            &Number::from(5),
            &Number::from(7),
            &Number::from(-1),
            &one
        ),
        None
    );
    assert_eq!(
        Rational::reconstruct_with_bounds(
            &Number::from(5),
            &Number::from(7),
            &one,
            &Number::from(0)
        ),
        None
    );
    // A bound past the modulus takes the residue itself.
    assert_eq!(
        Rational::reconstruct_with_bounds(
            &Number::from(12),
            &Number::from(7),
            &Number::from(100),
            &one
        ),
        Some(q(5, 1))
    );
}

#[test]
#[should_panic(expected = "modulus")]
fn reconstruct_zero_modulus() {
    Rational::reconstruct(&Number::from(3), &Number::from(0));
}

#[test]
#[should_panic(expected = "modulus")]
fn reconstruct_negative_modulus() {
    Rational::reconstruct_with_bounds(
        &Number::from(3),
        &Number::from(-7),
        &Number::from(1),
        &Number::from(1),
    );
}

// Across the forms: parts in words, at the boundary of 64 bits, and in numbers, and results that
// shrink to words or grow out of them. A value left in the wrong form is not equal to the same
// value made by `Rational::new`, nor does it hash alike.

/// The hash of `x` under the standard hasher.
fn hash_of(x: &Rational) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

/// Asserts that `a` and `b` are the same number, both in the form, equal and hashing alike.
#[track_caller]
fn same(a: &Rational, b: &Rational) {
    check(a);
    check(b);
    assert_eq!(a, b);
    assert_eq!(hash_of(a), hash_of(b), "{a} hashes apart from itself");
}

/// Asserts that `t` is in the shortest form and the byte order `o`.
#[track_caller]
fn shortest(t: &Number, o: ByteOrder) {
    assert!(t.byte_order() == o, "a term in another byte order: {t}");
    let top = match o {
        ByteOrder::BigEndian => t.value[0],
        ByteOrder::LittleEndian => *t.value.last().unwrap(),
    };
    assert!(
        top != 0 || t.value.len() == 1,
        "a term not in the shortest form: {t}"
    );
    assert!(!(t.is_zero() && t.is_negative()), "a negative zero term");
}

/// A magnitude of exactly `bits` bits, its other bits random, as a number in the byte order `o`.
fn of_bits(state: &mut u64, bits: usize, o: ByteOrder) -> Number {
    let bytes = bits.div_ceil(8);
    let mut v: Vec<u8> = (0..bytes).map(|_| xorshift(state) as u8).collect();
    let top = bits - 8 * (bytes - 1);
    v[bytes - 1] &= ((1u16 << top) - 1) as u8;
    v[bytes - 1] |= 1 << (top - 1);
    super::common::number(&v, false, o)
}

/// A length in bits about the forms' boundaries: short ones, 63 to 65 bits, 127 to 129, and long.
fn edge_bits(state: &mut u64) -> usize {
    const BITS: [usize; 14] = [1, 2, 5, 15, 16, 17, 32, 62, 63, 64, 65, 66, 128, 129];
    match below(state, 8) {
        0 => 130 + below(state, 300) as usize,
        1 => 1 + below(state, 130) as usize,
        _ => BITS[below(state, BITS.len() as u64) as usize],
    }
}

/// A random rational of parts about the forms' boundaries, of a random sign and byte order.
fn edge_rational(state: &mut u64) -> Rational {
    let o = order(xorshift(state));
    let (nb, db) = (edge_bits(state), edge_bits(state));
    let mut n = of_bits(state, nb, o);
    if below(state, 2) == 1 {
        n = -n;
    }
    let d = if below(state, 10) == 0 {
        Number::from(1u8)
    } else {
        of_bits(state, db, o)
    };
    let r = Rational::new(n, d);
    check(&r);
    r
}

/// The convergents of `terms` by the recurrence on numbers, each made by `Rational::new`.
fn convergents_plain(terms: &[Number]) -> Vec<Rational> {
    let (mut p0, mut q0) = (Number::from(0u8), Number::from(1u8));
    let (mut p1, mut q1) = (Number::from(1u8), Number::from(0u8));
    let mut out = Vec::new();
    for t in terms {
        let p2 = t * &p1 + &p0;
        let q2 = t * &q1 + &q0;
        out.push(Rational::new(p2.clone(), q2.clone()));
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
    }
    out
}

#[test]
fn continued_fractions_across_the_forms() {
    let mut s = 0xC0FF_EE00_1234_5678u64;
    for _ in 0..size(3000, 12) {
        let x = edge_rational(&mut s);
        let o = x.byte_order();
        let terms = x.continued_fraction();
        assert_eq!(terms, cf_plain(&x), "the terms of {x}");
        for t in &terms {
            shortest(t, o);
        }
        canonical(&x, &terms);
        same(&Rational::from_continued_fraction(&terms), &x);
        let c = x.convergents();
        let plain = convergents_plain(&terms);
        assert_eq!(c.len(), plain.len());
        for (a, b) in c.iter().zip(&plain) {
            same(a, b);
            assert!(a.byte_order() == o);
        }
    }
    // Terms past 64 bits before, in and after the convergents' run on limbs, of both signs.
    let mut s = 0xABCD_0123_4567_8899u64;
    for _ in 0..size(300, 4) {
        let o = order(xorshift(&mut s));
        let len = 2 + below(&mut s, 60) as usize;
        let mut terms = Vec::with_capacity(len);
        for i in 0..len {
            let b = match below(&mut s, 6) {
                0 => 65 + below(&mut s, 200) as usize,
                1 => 60 + below(&mut s, 10) as usize,
                _ => 1 + below(&mut s, 8) as usize,
            };
            let mut t = of_bits(&mut s, b, o);
            if i == 0 && below(&mut s, 2) == 1 {
                t = -t;
            }
            terms.push(t);
        }
        if *terms.last().unwrap() == 1u8 {
            *terms.last_mut().unwrap() = Number::from(2u8);
        }
        let x = Rational::from_continued_fraction(&terms);
        assert_eq!(x.continued_fraction(), terms);
        let c = x.convergents();
        let plain = convergents_plain(&terms);
        assert_eq!(c.len(), plain.len());
        for (a, b) in c.iter().zip(&plain) {
            same(a, b);
            assert!(a.byte_order() == x.byte_order());
        }
    }
    // Integers, zero and the extremes of the words.
    for v in [
        0i128,
        1,
        -1,
        7,
        -7,
        i128::from(u64::MAX),
        -i128::from(u64::MAX),
        1 << 64,
    ] {
        for o in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
            let n = super::common::number(
                &v.unsigned_abs().to_le_bytes()
                    [..16 - v.unsigned_abs().leading_zeros() as usize / 8],
                v < 0,
                o,
            );
            let x = Rational::from_integer(n.clone());
            let terms = x.continued_fraction();
            assert_eq!(terms, [n]);
            shortest(&terms[0], o);
            same(&Rational::from_continued_fraction(&terms), &x);
            same(&x.convergents()[0], &x);
        }
    }
    for (p, d) in [
        (u64::MAX, u64::MAX - 1),
        (u64::MAX - 1, u64::MAX),
        (1, u64::MAX),
    ] {
        for negative in [false, true] {
            let n = if negative {
                -Number::from(p)
            } else {
                Number::from(p)
            };
            let x = Rational::new(n, Number::from(d));
            let terms = x.continued_fraction();
            assert_eq!(terms, cf_plain(&x));
            canonical(&x, &terms);
        }
    }
}

#[test]
fn from_continued_fraction_across_the_forms() {
    let mut s = 0x0123_4567_89AB_CDEFu64;
    for _ in 0..size(2000, 10) {
        let o = order(xorshift(&mut s));
        let len = 1 + below(&mut s, 70) as usize;
        let first_bits = edge_bits(&mut s);
        let mut first = of_bits(&mut s, first_bits, o);
        if below(&mut s, 2) == 1 {
            first = -first;
        }
        if below(&mut s, 8) == 0 {
            first = super::common::number(&[], false, o);
        }
        let mut terms = vec![first];
        for _ in 1..len {
            let t = match below(&mut s, 6) {
                0 => {
                    let b = edge_bits(&mut s);
                    let to = order(xorshift(&mut s));
                    of_bits(&mut s, b, to)
                }
                _ => {
                    let k = below(&mut s, 20);
                    Number::from(1 + below(&mut s, 1 << k))
                }
            };
            terms.push(t);
        }
        let mut x = Rational::from_integer(terms[len - 1].clone());
        for t in terms[..len - 1].iter().rev() {
            x = x.recip() + Rational::from_integer(t.clone());
        }
        let got = Rational::from_continued_fraction(&terms);
        same(&got, &x);
        assert!(got.byte_order() == terms[0].byte_order());
    }
}

#[test]
fn limit_denominator_across_the_forms() {
    let mut s = 0xFEED_FACE_CAFE_BEEFu64;
    let one = Number::from(1u8);
    for _ in 0..size(2000, 10) {
        let x = edge_rational(&mut s);
        let max = match below(&mut s, 10) {
            0 => one.clone(),
            1 => Number::from(2u8),
            2 => (&one << 63) - 1u8,
            3 => &one << 63,
            4 => Number::from(u64::MAX),
            5 => &one << 64,
            6 => (&one << 64) + 1u8,
            _ => {
                let b = edge_bits(&mut s);
                let o = order(xorshift(&mut s));
                of_bits(&mut s, b, o)
            }
        };
        let got = x.limit_denominator(&max);
        same(&got, &limit_reference(&x, &max));
        assert!(got.byte_order() == x.byte_order());
        assert!(*got.denom() <= max);
    }
    // Ties at a half with a bound of 1, in words.
    for (p, d, r) in [
        (5i64, 2i64, 2i64),
        (7, 2, 4),
        (-5, 2, -2),
        (-7, 2, -4),
        (1, 2, 0),
        (-1, 2, 0),
    ] {
        same(&q(p, d).limit_denominator(&one), &q(r, 1));
    }
    let big = Number::from(u64::MAX);
    same(
        &Rational::new(big.clone(), Number::from(2u8)).limit_denominator(&one),
        &Rational::from_integer(&big / 2u8 + 1u8),
    );
}

#[test]
fn simplest_between_across_the_forms() {
    let mut s = 0x2468_1357_9BDF_ACE0u64;
    for _ in 0..size(2000, 10) {
        let x = edge_rational(&mut s);
        let y = match below(&mut s, 4) {
            0 => x.clone(),
            1 => {
                let b = edge_bits(&mut s) + 2;
                &x + &Rational::new(Number::from(1u8), of_bits(&mut s, b, order(0)))
            }
            _ => edge_rational(&mut s),
        };
        let got = Rational::simplest_between(&x, &y);
        let (lo, hi) = if x <= y { (&x, &y) } else { (&y, &x) };
        let expected = if lo == hi {
            lo.clone()
        } else {
            simplest_recursive(lo, hi)
        };
        same(&got, &expected);
        assert!(got.byte_order() == x.byte_order());
        same(&Rational::simplest_between(&y, &x), &expected);
    }
}

/// Wang's reconstruction on numbers, one division a step: the reference.
fn wang(r: &Number, m: &Number, n: &Number, d: &Number) -> Option<Rational> {
    if n.is_negative() || !d.is_positive() {
        return None;
    }
    let (mut r0, mut r1) = (m.clone(), r.rem_euclid(m));
    let (mut t0, mut t1) = (Number::from(0u8), Number::from(1u8));
    while r1 > *n {
        let (qq, rr) = r0.div_rem(&r1);
        let t2 = &t0 - &(&qq * &t1);
        (r0, r1, t0, t1) = (r1, rr, t1, t2);
    }
    if t1.abs() > *d || r1.gcd(&t1) != 1u8 {
        return None;
    }
    Some(Rational::new(r1, t1))
}

#[test]
fn reconstruct_across_the_forms() {
    let mut s = 0x7777_3333_1111_9999u64;
    for _ in 0..size(2000, 10) {
        let o = order(xorshift(&mut s));
        let mb = edge_bits(&mut s).max(2);
        let m = of_bits(&mut s, mb, o);
        // A residue of a small p/q, or any number, of either sign, maybe past the modulus.
        let r = match below(&mut s, 3) {
            0 => {
                let (pb, qb) = (1 + mb / 2 - mb / 4, 1 + mb / 2 - mb / 4);
                let pb = 1 + below(&mut s, pb as u64) as usize;
                let qb = 1 + below(&mut s, qb as u64) as usize;
                let p = of_bits(&mut s, pb, o);
                let qq = of_bits(&mut s, qb, o);
                match qq.mod_inverse(&m) {
                    Some(inv) => (&p * &inv).rem_euclid(&m),
                    None => p,
                }
            }
            _ => {
                let b = edge_bits(&mut s);
                of_bits(&mut s, b, o)
            }
        };
        let r = if below(&mut s, 3) == 0 { -r } else { r };
        let bound = |s: &mut u64| match below(s, 5) {
            0 => Number::from(u64::MAX),
            1 => Number::from(1u8) << 64,
            2 => Number::from(1u8),
            _ => {
                let b = edge_bits(s);
                let o = order(xorshift(s));
                of_bits(s, b, o)
            }
        };
        let (n, d) = (bound(&mut s), bound(&mut s));
        let got = Rational::reconstruct_with_bounds(&r, &m, &n, &d);
        let expected = wang(&r, &m, &n, &d);
        assert_eq!(got.is_some(), expected.is_some(), "{r} mod {m}, {n}, {d}");
        if let (Some(g), Some(e)) = (&got, &expected) {
            same(g, e);
            assert!(g.byte_order() == r.byte_order());
        }
        let b = ((&m - 1u8).div_floor(&Number::from(2u8))).isqrt();
        let got = Rational::reconstruct(&r, &m);
        let expected = if b.is_zero() {
            None
        } else {
            wang(&r, &m, &b, &b)
        };
        assert_eq!(got.is_some(), expected.is_some(), "{r} mod {m}");
        if let (Some(g), Some(e)) = (&got, &expected) {
            same(g, e);
            assert!(g.byte_order() == r.byte_order());
        }
    }
}

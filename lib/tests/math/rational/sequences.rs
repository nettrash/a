//! The Bernoulli and harmonic numbers: against exact values, the recurrence of the Bernoulli
//! numbers, von Staudt and Clausen's theorem, the table against the numbers one by one, the
//! harmonic numbers against their sums and their differences, Wolstenholme's theorem, and both
//! byte orders.

use super::common::{below, check, size, xorshift};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;

const ORDERS: [ByteOrder; 2] = [ByteOrder::LittleEndian, ByteOrder::BigEndian];

/// `B_0` to `B_30`, with `B_1 = -1/2`.
const BERNOULLI: [&str; 31] = [
    "1",
    "-1/2",
    "1/6",
    "0",
    "-1/30",
    "0",
    "1/42",
    "0",
    "-1/30",
    "0",
    "5/66",
    "0",
    "-691/2730",
    "0",
    "7/6",
    "0",
    "-3617/510",
    "0",
    "43867/798",
    "0",
    "-174611/330",
    "0",
    "854513/138",
    "0",
    "-236364091/2730",
    "0",
    "8553103/6",
    "0",
    "-23749461029/870",
    "0",
    "8615841276005/14322",
];
/// `H_0` to `H_30`.
const HARMONIC: [&str; 31] = [
    "0",
    "1",
    "3/2",
    "11/6",
    "25/12",
    "137/60",
    "49/20",
    "363/140",
    "761/280",
    "7129/2520",
    "7381/2520",
    "83711/27720",
    "86021/27720",
    "1145993/360360",
    "1171733/360360",
    "1195757/360360",
    "2436559/720720",
    "42142223/12252240",
    "14274301/4084080",
    "275295799/77597520",
    "55835135/15519504",
    "18858053/5173168",
    "19093197/5173168",
    "444316699/118982864",
    "1347822955/356948592",
    "34052522467/8923714800",
    "34395742267/8923714800",
    "312536252003/80313433200",
    "315404588903/80313433200",
    "9227046511387/2329089562800",
    "9304682830147/2329089562800",
];

/// Asserts the form of `r` and that both parts are in the byte order `o`.
#[track_caller]
fn in_order(r: &Rational, o: ByteOrder) {
    check(r);
    assert_eq!(r.numer().byte_order(), o);
    assert_eq!(r.denom().byte_order(), o);
}

/// Whether `n` is prime, by trial division.
fn is_prime(n: u64) -> bool {
    n >= 2
        && (2..)
            .take_while(|d| d * d <= n)
            .all(|d| !n.is_multiple_of(d))
}

/// The primes `p` with `p - 1` dividing `n`.
fn staudt_primes(n: u64) -> Vec<u64> {
    (1..=n)
        .filter(|d| n.is_multiple_of(*d) && is_prime(d + 1))
        .map(|d| d + 1)
        .collect()
}

/// `1/k`.
fn unit(k: u64) -> Rational {
    Rational::from(1) / k
}

#[test]
fn bernoulli_known_values() {
    for o in ORDERS {
        for (n, s) in BERNOULLI.iter().enumerate() {
            let b = Rational::bernoulli(n as u64, o);
            in_order(&b, o);
            assert_eq!(b.to_string(), *s, "B_{n}");
        }
        // B_60 = -1215233140483755572040304994079820246041491/56786730, the first beyond the
        // table, and B_100.
        assert_eq!(
            Rational::bernoulli(60, o).to_string(),
            "-1215233140483755572040304994079820246041491/56786730"
        );
        assert_eq!(
            Rational::bernoulli(100, o).to_string(),
            "-94598037819122125295227433069493721872702841533066936133385696204311395415197247711/33330"
        );
    }
}

#[test]
fn bernoulli_odd_indices_are_zero() {
    for o in ORDERS {
        for n in (3..200).step_by(2).chain([1001, 99_999, u64::MAX]) {
            let b = Rational::bernoulli(n, o);
            in_order(&b, o);
            assert!(b.is_zero(), "B_{n}");
        }
    }
}

/// `Σ_(k=0)^m C(m+1, k) B_k = 0` for `m >= 1`, the recurrence that defines them with
/// `B_1 = -1/2`.
fn recurrence_holds(m: u64, b: &[Rational]) -> bool {
    let o = ByteOrder::LittleEndian;
    let s: Rational = (0..=m)
        .filter(|&k| k == 1 || k % 2 == 0)
        .map(|k| &b[k as usize] * Number::binomial(m + 1, k, o))
        .sum();
    s.is_zero()
}

#[test]
fn bernoulli_recurrence() {
    let mut state = 0x5EED_B0B5_u64;
    let top = size(400, 70) as u64;
    for o in ORDERS {
        let table = Rational::bernoulli_numbers(top, o);
        for m in 1..=top.min(size(120, 20) as u64) {
            assert!(recurrence_holds(m, &table), "m = {m}");
        }
        for _ in 0..size(12, 2) {
            let m = 2 + below(&mut state, top - 1);
            let one: Vec<Rational> = (0..=m).map(|k| Rational::bernoulli(k, o)).collect();
            assert!(recurrence_holds(m, &one), "m = {m}");
        }
    }
}

#[test]
fn bernoulli_von_staudt_clausen() {
    let mut state = 0x0057_A0D7_u64;
    let mut ns: Vec<u64> = vec![2, 4, 58, 60, 62, 120, 360, 720, 840, 1260, 2520];
    for _ in 0..size(30, 3) {
        ns.push(2 * (31 + below(&mut state, size(1500, 40) as u64)));
    }
    if cfg!(miri) {
        ns.retain(|&n| n <= 120);
    }
    for n in ns {
        let o = if xorshift(&mut state) & 1 == 0 {
            ByteOrder::LittleEndian
        } else {
            ByteOrder::BigEndian
        };
        let b = Rational::bernoulli(n, o);
        in_order(&b, o);
        let primes = staudt_primes(n);
        let d = primes.iter().fold(Number::from(1u8), |d, &p| &d * p);
        assert_eq!(*b.denom(), d, "the denominator of B_{n}");
        // B_n + Σ 1/p is an integer.
        let s = primes.iter().fold(b.clone(), |s, &p| s + unit(p));
        assert!(s.is_integer(), "B_{n} + Σ 1/p = {s}");
        // The sign: B_n > 0 for n = 2 mod 4.
        assert_eq!(b.is_positive(), n % 4 == 2, "the sign of B_{n}");
    }
}

#[test]
fn bernoulli_table_against_one_by_one() {
    let tops: &[u64] = if cfg!(miri) {
        &[0, 1, 2, 3, 57, 58, 59, 60, 61, 64]
    } else {
        &[
            0, 1, 2, 3, 57, 58, 59, 60, 61, 64, 99, 100, 101, 102, 103, 149, 150, 151, 152, 153,
            160, 400, 1001,
        ]
    };
    for &top in tops {
        for o in ORDERS {
            let table = Rational::bernoulli_numbers(top, o);
            assert_eq!(table.len() as u64, top + 1);
            for (k, b) in table.iter().enumerate() {
                in_order(b, o);
                // Every one of the short tables; a sample of the longer ones.
                if top <= 160 || k % 37 == 0 || k as u64 >= top - 3 {
                    assert_eq!(*b, Rational::bernoulli(k as u64, o), "B_{k} of {top}");
                }
            }
        }
    }
}

#[test]
fn bernoulli_byte_orders_agree() {
    for n in [0u64, 1, 2, 30, 58, 60, 62, 100, 202, 1000] {
        if cfg!(miri) && n > 100 {
            continue;
        }
        let le = Rational::bernoulli(n, ByteOrder::LittleEndian);
        let be = Rational::bernoulli(n, ByteOrder::BigEndian);
        assert_eq!(le, be, "B_{n}");
        let mut v = be.numer().value.clone();
        v.reverse();
        assert_eq!(v, le.numer().value, "B_{n}");
    }
}

#[test]
fn bernoulli_long() {
    if cfg!(miri) {
        return;
    }
    // B_2000, B_5000 from their properties: the denominators, the recurrence's neighbours by the
    // table, and the size: log2 |B_n| = log2(2 n! / (2π)^n) + log2 ζ(n).
    for n in [2000u64, 5000] {
        let b = Rational::bernoulli(n, ByteOrder::LittleEndian);
        let d = staudt_primes(n)
            .iter()
            .fold(Number::from(1u8), |d, &p| &d * p);
        assert_eq!(*b.denom(), d);
        let bits = b.numer().bits() as f64 - d.to_f64().log2();
        let lf: f64 = (1..=n).map(|k| (k as f64).log2()).sum();
        let expected = 1.0 + lf - n as f64 * (2.0 * core::f64::consts::PI).log2();
        assert!(
            (bits - expected).abs() < 2.0,
            "B_{n}: {bits} bits against {expected}"
        );
        let table = Rational::bernoulli_numbers(n, ByteOrder::BigEndian);
        assert_eq!(table[n as usize], b);
    }
    let table = Rational::bernoulli_numbers(700, ByteOrder::LittleEndian);
    assert!(recurrence_holds(700, &table));
}

/// `a mod p` for a rational with a denominator prime to `p`.
fn residue(r: &Rational, p: u64) -> u64 {
    let pn = Number::from(p);
    let num = r.numer().rem_euclid(&pn);
    let inv = r
        .denom()
        .mod_inverse(&pn)
        .expect("a denominator prime to p");
    u64::try_from(&(&num * &inv).rem_euclid(&pn)).unwrap()
}

#[test]
fn bernoulli_kummer() {
    // Kummer's congruences: B_n / n = B_m / m modulo p for n = m modulo p - 1, (p - 1) not
    // dividing n; for B_20000, whose low bits are taken modulo the primes up to about 22,000 and
    // the rest from the zeta function, and B_6000, from the zeta function alone.
    let ns: &[u64] = if cfg!(miri) { &[200] } else { &[6000, 20_000] };
    for &n in ns {
        let b = Rational::bernoulli(n, ByteOrder::LittleEndian);
        let d = staudt_primes(n)
            .iter()
            .fold(Number::from(1u8), |d, &p| &d * p);
        assert_eq!(*b.denom(), d, "the denominator of B_{n}");
        let s = staudt_primes(n).iter().fold(b.clone(), |s, &p| s + unit(p));
        assert!(s.is_integer(), "B_{n} + Σ 1/p = {s}");
        let small = Rational::bernoulli_numbers(size(400, 60) as u64, ByteOrder::BigEndian);
        for p in (5..size(400, 60) as u64 + 2).filter(|&p| is_prime(p) && n % (p - 1) != 0) {
            let m = n % (p - 1);
            let lhs = residue(&(&b / n), p);
            let rhs = residue(&(&small[m as usize] / m), p);
            assert_eq!(lhs, rhs, "B_{n} modulo {p}");
        }
    }
}

#[test]
fn bernoulli_long_against_table() {
    // B_8000 to B_8010, the first numbers whose low bits are taken modulo primes, against the
    // table, which takes them all from the zeta function: in optimised builds only, as the table
    // takes seconds without.
    if cfg!(debug_assertions) || cfg!(miri) {
        return;
    }
    let table = Rational::bernoulli_numbers(8010, ByteOrder::LittleEndian);
    for n in 7996..=8010 {
        assert_eq!(
            table[n as usize],
            Rational::bernoulli(n, ByteOrder::LittleEndian),
            "B_{n}"
        );
    }
}

#[test]
fn harmonic_known_values() {
    for o in ORDERS {
        for (n, s) in HARMONIC.iter().enumerate() {
            let h = Rational::harmonic(n as u64, o);
            in_order(&h, o);
            assert_eq!(h.to_string(), *s, "H_{n}");
        }
    }
}

#[test]
fn harmonic_against_sums() {
    // Every n up to 400 (the table ends at 88), and the ranges where an odd prime p divides the
    // numerator of H_k for k = floor(n / p^e): p = 3 (H_2 = 3/2) from 162 to 242, p = 7
    // (H_6 = 49/20) from 294 to 342, p = 11 (H_10 = 7381/2520 = 11^2 61 / 2520) from 110 to 120
    // and 1210 to 1330, p = 61 (the same) from 610 to 670, and p = 5 (H_4 = 25/12) from 500 to 624.
    let top = size(1340, 100) as u64;
    for o in ORDERS {
        let mut s = Rational::from(0);
        for n in 1..=top {
            s += unit(n);
            let wanted = n <= 400 || (500..=670).contains(&n) || (1210..=1330).contains(&n);
            if wanted && (o == ByteOrder::LittleEndian || n % 7 == 0) {
                let h = Rational::harmonic(n, o);
                in_order(&h, o);
                assert_eq!(h, s, "H_{n}");
            }
        }
    }
}

#[test]
fn harmonic_differences() {
    let mut state = 0xA2B1_7CC0u64;
    let top = size(30_000, 300) as u64;
    for _ in 0..size(40, 4) {
        let n = 1 + below(&mut state, top);
        let o = if xorshift(&mut state) & 1 == 0 {
            ByteOrder::LittleEndian
        } else {
            ByteOrder::BigEndian
        };
        let h = Rational::harmonic(n, o);
        in_order(&h, o);
        assert_eq!(
            &h - &Rational::harmonic(n - 1, o),
            unit(n),
            "H_{n} - H_(n - 1)"
        );
        // The denominator of H_n is divisible by 2^floor(log2 n), and divides lcm(1, ..., n).
        let t = 63 - n.leading_zeros();
        assert_eq!(h.denom().trailing_zeros(), Some(u64::from(t)), "H_{n}");
    }
}

#[test]
fn harmonic_wolstenholme() {
    // p^2 divides the numerator of H_(p-1) for every prime p >= 5.
    let primes: &[u64] = if cfg!(miri) {
        &[5, 7, 11, 97, 101]
    } else {
        &[5, 7, 11, 97, 101, 211, 1009, 4099]
    };
    for &p in primes {
        let h = Rational::harmonic(p - 1, ByteOrder::BigEndian);
        let pp = Number::from(p * p);
        assert!(h.numer().is_multiple_of(&pp), "H_{}", p - 1);
        // ... and H_p's numerator is prime to p (its denominator has p).
        assert!(
            Rational::harmonic(p, ByteOrder::BigEndian)
                .denom()
                .is_multiple_of(&Number::from(p))
        );
    }
}

#[test]
fn harmonic_byte_orders_agree() {
    for n in [0u64, 1, 2, 88, 89, 90, 1000, 5000] {
        if cfg!(miri) && n > 100 {
            continue;
        }
        let le = Rational::harmonic(n, ByteOrder::LittleEndian);
        let be = Rational::harmonic(n, ByteOrder::BigEndian);
        assert_eq!(le, be, "H_{n}");
    }
}

/// The hash of `x` under the standard hasher.
fn hash_of(x: &Rational) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

/// Asserts that `r` has the one form of its value: equal to, and hashing as, the rational made by
/// `Rational::new` from its parts, in the byte order `o`. A value in words left in numbers, or the
/// other way, would be neither.
#[track_caller]
fn in_its_form(r: &Rational, o: ByteOrder) {
    in_order(r, o);
    let made = Rational::new(r.numer().clone(), r.denom().clone());
    assert_eq!(*r, made);
    assert_eq!(hash_of(r), hash_of(&made), "{r} hashes apart from itself");
}

#[test]
fn sequences_across_the_forms() {
    // About the boundary of the words: B_34 and B_38 are in words, B_36 and from B_40 not; H_46
    // is the last harmonic number in words, H_88 the last of the table, H_124 the last summed
    // term by term.
    let top = if cfg!(miri) { 64 } else { 130 };
    for o in ORDERS {
        let table = Rational::bernoulli_numbers(top, o);
        for n in 0..=top {
            let b = Rational::bernoulli(n, o);
            in_its_form(&b, o);
            in_its_form(&table[n as usize], o);
            assert_eq!(table[n as usize], b);
            let h = Rational::harmonic(n, o);
            in_its_form(&h, o);
            if n > 0 {
                let last = Rational::harmonic(n - 1, o);
                assert_eq!(&h - &last, Rational::from(1) / n, "H_{n}");
            }
        }
        for n in [33u64, 34, 35, 36, 37, 38, 39, 40] {
            let b = Rational::bernoulli(n, o);
            let fits = |x: &Number| x.bits() <= 64;
            assert_eq!(
                fits(b.numer()) && fits(b.denom()),
                n % 2 == 1 || n <= 34 || n == 38,
                "B_{n}"
            );
        }
        let h = Rational::harmonic(46, o);
        assert!(h.numer().bits() <= 64 && h.denom().bits() <= 64);
        let h = Rational::harmonic(47, o);
        assert!(h.numer().bits() > 64 || h.denom().bits() > 64);
    }
    // Short vectors, each length, against the numbers one by one.
    for n in 0..size(40, 8) as u64 {
        for o in ORDERS {
            let v = Rational::bernoulli_numbers(n, o);
            assert_eq!(v.len() as u64, n + 1);
            for (k, b) in v.iter().enumerate() {
                in_its_form(b, o);
                assert_eq!(*b, Rational::bernoulli(k as u64, o));
            }
        }
    }
}

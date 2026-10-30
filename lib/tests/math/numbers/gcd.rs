use a::math::numbers::{ByteOrder, Number};
use std::cmp::Ordering;

use super::common::*;
use super::div::{add_magnitudes, mul_magnitudes, panic_message, quotient_and_remainder};

// The rules, checked by the reference arithmetic of the tests of division: a greatest common
// divisor divides both numbers and is a combination of them, and the cofactors are GMP's, which
// its bounds make unique.

/// `x` without the zero bytes above its most significant one, least significant byte first;
/// empty for zero.
fn trimmed(x: &[u8]) -> Vec<u8> {
    let top = x.iter().rposition(|&b| b != 0).map_or(0, |t| t + 1);
    x[..top].to_vec()
}

/// Compares the magnitudes `x` and `y`, given least significant byte first.
fn compare(x: &[u8], y: &[u8]) -> Ordering {
    let (x, y) = (trimmed(x), trimmed(y));
    x.len()
        .cmp(&y.len())
        .then_with(|| x.iter().rev().cmp(y.iter().rev()))
}

/// The magnitude of `n`, least significant byte first, without high zero bytes.
fn magnitude(n: &Number) -> Vec<u8> {
    trimmed(&le_bytes(n))
}

/// Whether `n` is zero.
fn is_zero(n: &Number) -> bool {
    magnitude(n).is_empty()
}

/// The sign of `n` as -1, 0 or 1.
fn signum(n: &Number) -> i32 {
    if is_zero(n) {
        0
    } else if n.negative {
        -1
    } else {
        1
    }
}

/// Whether `d` divides `n`, by the reference division; only zero is a multiple of zero.
fn divides(d: &Number, n: &Number) -> bool {
    if is_zero(d) {
        return is_zero(n);
    }
    quotient_and_remainder(n, d).1 == expected(&[], false)
}

/// The sum of the signed terms `(sign, magnitude)`, as an expected number.
fn signed_sum(terms: &[(bool, Vec<u8>)]) -> Expected {
    let (mut plus, mut minus) = (Vec::new(), Vec::new());
    for (negative, m) in terms {
        if *negative {
            minus = add_magnitudes(&minus, m);
        } else {
            plus = add_magnitudes(&plus, m);
        }
    }
    match compare(&plus, &minus) {
        Ordering::Less => expected(&super::div::sub_magnitudes(&minus, &plus), true),
        _ => expected(&super::div::sub_magnitudes(&plus, &minus), false),
    }
}

/// Checks `a.gcd_ext(b)`, `a.gcd(b)` and the other operations' agreement with them: the results in
/// the shortest form and the byte order of `a`; `g >= 0` dividing both; `g = a x + b y`; the
/// cofactors GMP's (`|x| < |b| / 2g`, `|y| < |a| / 2g`, but for the documented cases); `gcd` the
/// same `g`; and both operands left as they were.
fn check(a: &Number, b: &Number) {
    let inputs = || format!("gcd_ext of a = {}, b = {}", describe(a), describe(b));
    let (copy_a, copy_b) = (copy(a, 0), copy(b, 0));
    let order = a.byte_order;
    let (g, x, y) = a.gcd_ext(b);
    for (name, n) in [("g", &g), ("x", &x), ("y", &y)] {
        assert_shortest(n, &|| format!("{}: {name}", inputs()));
        assert!(
            n.byte_order == order,
            "{}: {name} not in a's order",
            inputs()
        );
    }
    assert!(!g.negative, "{}: g is negative", inputs());
    assert!(
        divides(&g, a) && divides(&g, b),
        "{}: g = {} does not divide both",
        inputs(),
        describe(&g)
    );
    let combination = signed_sum(&[
        (
            a.negative != x.negative,
            mul_magnitudes(&magnitude(a), &magnitude(&x)),
        ),
        (
            b.negative != y.negative,
            mul_magnitudes(&magnitude(b), &magnitude(&y)),
        ),
    ]);
    assert_eq!(
        combination,
        value_of(&g),
        "{}: a x + b y is not g",
        inputs()
    );
    // GMP's cofactors.
    let (ma, mb, mg) = (magnitude(a), magnitude(b), magnitude(&g));
    let two_g = mul_magnitudes(&mg, &[2]);
    let (sa, sb) = (signum(a), signum(b));
    if compare(&ma, &mb) == Ordering::Equal {
        assert!(
            is_zero(&x) && signum(&y) == sb,
            "{}: equal magnitudes",
            inputs()
        );
    } else {
        if is_zero(b) || compare(&mb, &two_g) == Ordering::Equal {
            assert!(
                signum(&x) == sa && magnitude(&x).len() <= 1,
                "{}: x is not sgn(a)",
                inputs()
            );
            assert!(
                is_zero(b) || magnitude(&x) == [1],
                "{}: x is not sgn(a)",
                inputs()
            );
        } else {
            let bound = mul_magnitudes(&two_g, &magnitude(&x));
            assert!(
                compare(&bound, &mb) == Ordering::Less,
                "{}: |x| >= |b| / 2g",
                inputs()
            );
        }
        if is_zero(a) || compare(&ma, &two_g) == Ordering::Equal {
            assert!(
                signum(&y) == sb && (is_zero(a) || magnitude(&y) == [1]),
                "{}: y is not sgn(b)",
                inputs()
            );
        } else {
            let bound = mul_magnitudes(&two_g, &magnitude(&y));
            assert!(
                compare(&bound, &ma) == Ordering::Less,
                "{}: |y| >= |a| / 2g",
                inputs()
            );
        }
    }
    let only = a.gcd(b);
    assert_result(&only, &value_of(&g), order, &|| {
        format!("gcd of a = {}, b = {}", describe(a), describe(b))
    });
    assert!(
        same(a, &copy_a) && same(b, &copy_b),
        "{}: an operand changed",
        inputs()
    );
}

/// Checks `a.lcm(b)`: never negative, in the byte order of `a`, zero when either is, and otherwise
/// `|a b| / gcd(a, b)`, which both divide.
fn check_lcm(a: &Number, b: &Number) {
    let inputs = || format!("lcm of a = {}, b = {}", describe(a), describe(b));
    let l = a.lcm(b);
    let expected_l = if is_zero(a) || is_zero(b) {
        expected(&[], false)
    } else {
        let g = a.gcd(b);
        let product = Number::new(
            mul_magnitudes(&magnitude(a), &magnitude(b)),
            ByteOrder::LittleEndian,
        );
        let (q, r) = quotient_and_remainder(&product, &g);
        assert_eq!(
            r,
            expected(&[], false),
            "{}: the reference product is not a multiple of g",
            inputs()
        );
        expected(&q.magnitude, false)
    };
    assert_result(&l, &expected_l, a.byte_order, &inputs);
    assert!(
        divides(a, &l) && divides(b, &l),
        "{}: a or b does not divide it",
        inputs()
    );
}

/// Checks `a.mod_inverse(m)` for `m` not zero: `Some(r)` with `0 <= r < |m|` and `a r = 1` modulo
/// `|m|` exactly when `gcd(a, m) = 1`, and `Some(0)` for `|m| = 1`.
fn check_inverse(a: &Number, m: &Number) {
    let inputs = || format!("mod_inverse of a = {}, m = {}", describe(a), describe(m));
    let mm = magnitude(m);
    let r = a.mod_inverse(m);
    let coprime = a.gcd(m).value == [1];
    if mm == [1] {
        let r = r.unwrap_or_else(|| panic!("{}: no inverse modulo 1", inputs()));
        assert_result(&r, &expected(&[], false), a.byte_order, &inputs);
        return;
    }
    match r {
        None => assert!(!coprime, "{}: no inverse, but gcd = 1", inputs()),
        Some(r) => {
            assert!(coprime, "{}: an inverse, but gcd != 1", inputs());
            assert_shortest(&r, &inputs);
            assert!(
                !r.negative && r.byte_order == a.byte_order,
                "{}: sign or order",
                inputs()
            );
            assert!(
                compare(&magnitude(&r), &mm) == Ordering::Less,
                "{}: r >= |m|",
                inputs()
            );
            // a r - 1 is a multiple of m.
            let ar = expected(&mul_magnitudes(&magnitude(a), &magnitude(&r)), a.negative);
            let minus_one = signed_sum(&[(ar.negative, ar.magnitude.clone()), (true, vec![1])]);
            let n = Number::new_signed(
                minus_one.magnitude.clone(),
                minus_one.negative,
                ByteOrder::LittleEndian,
            );
            assert!(divides(m, &n), "{}: a r is not 1 modulo m", inputs());
        }
    }
}

/// Every check, for `a` and `b` and for the other orders, signs and the swapped pair.
fn check_all(a: &Number, b: &Number) {
    check(a, b);
    check(b, a);
    check_lcm(a, b);
    if !is_zero(b) {
        check_inverse(a, b);
    }
    if !is_zero(a) {
        check_inverse(b, a);
    }
}

/// `a` and `b`, magnitudes least significant byte first, in a random order, sign and byte order
/// each.
fn pair(state: &mut u64, x: &[u8], y: &[u8]) -> (Number, Number) {
    let (oa, ob) = ORDER_PAIRS[below(state, 4)];
    let (na, nb) = SIGN_PAIRS[below(state, 4)];
    (number(x, na, oa), number(y, nb, ob))
}

// Small values, against the definition

/// The greatest common divisor of `a` and `b` and GMP's cofactors, found by search from the
/// documented rules: the `x` for which `y = (g - a x) / b` is an integer and both bounds hold.
fn reference_ext(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut p, mut q) = (a.abs(), b.abs());
    while q != 0 {
        (p, q) = (q, p % q);
    }
    let g = p;
    if a.abs() == b.abs() {
        return (g, 0, b.signum());
    }
    if b == 0 {
        return (g, a.signum(), 0);
    }
    if a == 0 {
        return (g, 0, b.signum());
    }
    let mut found = None;
    for x in -b.abs()..=b.abs() {
        if (g - a * x) % b != 0 {
            continue;
        }
        let y = (g - a * x) / b;
        let x_ok = if b.abs() == 2 * g {
            x == a.signum()
        } else {
            2 * g * x.abs() < b.abs()
        };
        let y_ok = if a.abs() == 2 * g {
            y == b.signum()
        } else {
            2 * g * y.abs() < a.abs()
        };
        if x_ok && y_ok {
            assert!(found.is_none(), "two pairs of cofactors for {a}, {b}");
            found = Some((x, y));
        }
    }
    let (x, y) = found.unwrap_or_else(|| panic!("no cofactors for {a}, {b}"));
    (g, x, y)
}

#[test]
fn small_values_match_gmp_s_definition() {
    // Every pair of small integers, against a search from the documented rules, and the cases that
    // the rules set apart: zero, equal magnitudes, |b| = 2g and |a| = 2g.
    let limit = size(40, 9) as i64;
    for a in -limit..=limit {
        for b in -limit..=limit {
            let (g, x, y) = reference_ext(a, b);
            for order in ORDERS {
                let na = Number::new_from_i64(a, order);
                let nb = Number::new_from_i64(b, ByteOrder::LittleEndian);
                let (rg, rx, ry) = na.gcd_ext(&nb);
                let inputs = || format!("gcd_ext({a}, {b})");
                assert_result(&rg, &expected_i128(i128::from(g)), order, &inputs);
                assert_result(&rx, &expected_i128(i128::from(x)), order, &inputs);
                assert_result(&ry, &expected_i128(i128::from(y)), order, &inputs);
                assert_result(&na.gcd(&nb), &expected_i128(i128::from(g)), order, &inputs);
                let l = if g == 0 { 0 } else { (a * b).abs() / g };
                assert_result(&na.lcm(&nb), &expected_i128(i128::from(l)), order, &inputs);
                if b != 0 {
                    let inverse = na.mod_inverse(&nb);
                    if b.abs() == 1 {
                        assert_eq!(inverse.map(|r| r.value), Some(vec![0]), "{}", inputs());
                    } else if g == 1 {
                        let r = (x % b.abs() + b.abs()) % b.abs();
                        assert_result(
                            &inverse.expect("an inverse"),
                            &expected_i128(i128::from(r)),
                            order,
                            &inputs,
                        );
                    } else {
                        assert!(inverse.is_none(), "{}: an inverse", inputs());
                    }
                }
            }
        }
    }
}

#[test]
fn the_documented_examples_hold() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    let (g, x, y) = Number::new_from_i32(240, be).gcd_ext(&Number::new_from_i32(-46, be));
    assert_eq!(
        (g.value, x.value, x.negative, y.value, y.negative),
        (vec![2], vec![9], true, vec![47], true)
    );
    // gcd(0, 0) = 0, with cofactors 0 and 0.
    let zero = Number::new_from_u8(0, le);
    let (g, x, y) = zero.gcd_ext(&zero);
    assert!(same(&g, &zero) && same(&x, &zero) && same(&y, &zero));
    // gcd(x, 0) = |x| = sgn(x) x.
    let (g, x, y) = Number::new_from_i32(-7, le).gcd_ext(&zero);
    assert_eq!(
        (g.value, x.value, x.negative, y.value),
        (vec![7], vec![1], true, vec![0])
    );
    // |a| = |b|: x = 0, y = sgn(b).
    let (g, x, y) = Number::new_from_i32(5, le).gcd_ext(&Number::new_from_i32(-5, be));
    assert_eq!(
        (g.value, x.value, y.value, y.negative),
        (vec![5], vec![0], vec![1], true)
    );
    // Every number has the inverse 0 modulo 1.
    let one = Number::new_from_i32(-1, be);
    assert_eq!(
        Number::new_from_u64(u64::MAX, be)
            .mod_inverse(&one)
            .map(|r| r.value),
        Some(vec![0])
    );
    assert_eq!(zero.mod_inverse(&one).map(|r| r.value), Some(vec![0]));
    assert!(zero.mod_inverse(&Number::new_from_u8(7, be)).is_none());
}

// Random numbers of every shape

/// Random magnitudes `(x, y)` of up to `max` bytes in one of the shapes the algorithms treat
/// apart: random, of random lengths or the same length; with a long common factor; one a multiple
/// of the other; equal; consecutive Fibonacci numbers, Euclid's worst case; powers of two times
/// odd numbers; and numbers of all one bits.
fn shaped(state: &mut u64, max: usize) -> (Vec<u8>, Vec<u8>) {
    let len = |state: &mut u64| below(state, max + 1);
    match below(state, 8) {
        0 => {
            let (lx, ly) = (len(state), len(state));
            (random_magnitude(state, lx), random_magnitude(state, ly))
        }
        1 => {
            let l = len(state);
            (random_magnitude(state, l), random_magnitude(state, l))
        }
        2 => {
            let lg = 1 + below(state, max / 2 + 1);
            let g = random_magnitude(state, lg);
            let (lu, lv) = (below(state, max / 2 + 1), below(state, max / 2 + 1));
            let (u, v) = (random_magnitude(state, lu), random_magnitude(state, lv));
            (mul_magnitudes(&g, &u), mul_magnitudes(&g, &v))
        }
        3 => {
            let ly = 1 + below(state, max / 2 + 1);
            let y = random_magnitude(state, ly);
            let lq = below(state, max / 2 + 1);
            let q = random_magnitude(state, lq);
            (mul_magnitudes(&y, &q), y)
        }
        4 => {
            let l = len(state);
            let x = random_magnitude(state, l);
            (x.clone(), x)
        }
        5 => fibonacci(1 + below(state, 8 * max * 3 / 2 + 1)),
        6 => {
            let (i, j) = (below(state, 8 * max + 1), below(state, 8 * max + 1));
            let mut x = vec![0; i / 8 + 1];
            x[i / 8] = 1 << (i % 8);
            let mut y = vec![0; j / 8 + 1];
            y[j / 8] = 1 << (j % 8);
            let lo = 1 + below(state, 20);
            let o = random_magnitude(state, lo);
            (mul_magnitudes(&x, &o), y)
        }
        _ => {
            let (i, j) = (1 + len(state), 1 + len(state));
            (vec![0xFF; i], vec![0xFF; j])
        }
    }
}

/// The Fibonacci numbers `F(k)` and `F(k - 1)`, least significant byte first, built by the crate's
/// addition, which its own tests check, as these are long.
fn fibonacci(k: usize) -> (Vec<u8>, Vec<u8>) {
    let (mut a, mut b) = (
        Number::new_from_u8(1, ByteOrder::LittleEndian),
        Number::new_from_u8(0, ByteOrder::LittleEndian),
    );
    for _ in 1..k {
        let c = &a + &b;
        b = a;
        a = c;
    }
    (le_bytes(&a), le_bytes(&b))
}

#[test]
fn random_numbers_of_every_shape_follow_the_rules() {
    let mut state = 0x243F_6A88_85A3_08D3;
    for _ in 0..size(3000, 40) {
        let max = if below(&mut state, 4) == 0 {
            size(200, 40)
        } else {
            size(40, 20)
        };
        let (x, y) = shaped(&mut state, max);
        let (a, b) = pair(&mut state, &x, &y);
        check_all(&a, &b);
    }
}

#[test]
fn fibonacci_numbers_take_the_most_steps() {
    // Consecutive Fibonacci numbers make every quotient of Euclid's algorithm 1: hgcd2 runs to its
    // bound, and the steps it cannot take are subtractions.
    let mut state = 0x1319_8A2E_0370_7344;
    for k in [2, 3, 50, 92, 93, 94, 186, 187, 300, 700, 1000, 1500]
        .into_iter()
        .filter(|&k| k <= size(1500, 190))
    {
        let (x, y) = fibonacci(k);
        let (a, b) = pair(&mut state, &x, &y);
        check_all(&a, &b);
        // With a common factor, and F(k) against F(k / 2).
        let s = random_magnitude(&mut state, 5);
        let (a, b) = pair(&mut state, &mul_magnitudes(&x, &s), &mul_magnitudes(&y, &s));
        check(&a, &b);
        let (z, _) = fibonacci(k / 2 + 1);
        let (a, b) = pair(&mut state, &x, &z);
        check(&a, &b);
    }
}

#[test]
fn every_pair_of_lengths_up_to_40_bytes_follows_the_rules() {
    // Through the paths in registers (up to 16 bytes each), one limb against several, and Lehmer's
    // method on two limbs and more; random numbers and numbers with a common factor.
    let mut state = 0xA409_3822_299F_31D0;
    let max = size(40, 18);
    for lx in 0..=max {
        for ly in 0..=max {
            if cfg!(miri) && below(&mut state, 6) != 0 {
                continue;
            }
            let (x, y) = (
                random_magnitude(&mut state, lx),
                random_magnitude(&mut state, ly),
            );
            let (a, b) = pair(&mut state, &x, &y);
            check_all(&a, &b);
            let lg = 1 + below(&mut state, 9);
            let g = random_magnitude(&mut state, lg);
            let (a, b) = pair(&mut state, &mul_magnitudes(&x, &g), &mul_magnitudes(&y, &g));
            check(&a, &b);
        }
    }
}

#[test]
fn one_limb_against_longer_numbers_follows_the_rules() {
    // A one-limb number against a longer one takes a division first; a limb 16 bits longer than
    // the other limb as well; zero bits at the low end are set apart.
    let mut state = 0x082E_FA98_EC4E_6C89;
    for len in [1, 2, 3, 7, 8, 9, 17, 100, 1000]
        .into_iter()
        .filter(|&l| l <= size(1000, 100))
    {
        for small_len in [1, 2, 3, 8] {
            let x = random_magnitude(&mut state, len);
            let mut y = random_magnitude(&mut state, small_len);
            if below(&mut state, 3) == 0 {
                y[0] &= 0xF0;
                if y.iter().all(|&b| b == 0) {
                    y[0] = 0x10;
                }
            }
            let (a, b) = pair(&mut state, &x, &y);
            check_all(&a, &b);
        }
    }
}

/// Lengths in bytes at and around the limb counts where the algorithms switch: Lehmer's method
/// to the half gcd for the extended algorithm (300 limbs) and for the greatest common divisor
/// (800), and the half gcd's call on itself (250 limbs); under Miri, which runs the switches at 8
/// and 6 limbs, around those.
fn threshold_lengths() -> Vec<usize> {
    if cfg!(miri) {
        vec![40, 48, 56, 64, 72, 120]
    } else {
        vec![2392, 2400, 2408, 4000, 4100, 6392, 6400, 6408]
    }
}

#[test]
fn numbers_around_the_switches_of_method_follow_the_rules() {
    let mut state = 0xBE54_66CF_34E9_0C6C;
    for len in threshold_lengths() {
        for shape in 0..4 {
            let (x, y) = match shape {
                0 => (
                    random_magnitude(&mut state, len),
                    random_magnitude(&mut state, len),
                ),
                1 => (
                    random_magnitude(&mut state, len),
                    random_magnitude(&mut state, len - 8),
                ),
                2 => {
                    let g = random_magnitude(&mut state, len / 2);
                    let (u, v) = (
                        random_magnitude(&mut state, len / 2),
                        random_magnitude(&mut state, len / 2),
                    );
                    (mul_magnitudes(&g, &u), mul_magnitudes(&g, &v))
                }
                _ => fibonacci(len * 8 * 3 / 2),
            };
            let (a, b) = pair(&mut state, &x, &y);
            check(&a, &b);
            if shape == 0 {
                check_lcm(&a, &b);
                check_inverse(&a, &b);
            }
        }
    }
}

#[test]
fn unbalanced_numbers_follow_the_rules() {
    // A much longer number is divided by the shorter one first.
    let mut state = 0xC0AC_29B7_C97C_50DD;
    for (lx, ly) in [(4096, 64), (4096, 1024), (300, 17), (1000, 999), (65, 64)] {
        let (lx, ly) = (lx.min(size(usize::MAX, 120)), ly.min(size(usize::MAX, 60)));
        let (x, y) = (
            random_magnitude(&mut state, lx),
            random_magnitude(&mut state, ly),
        );
        let (a, b) = pair(&mut state, &x, &y);
        check_all(&a, &b);
    }
}

#[test]
fn powers_of_two_and_all_ones_follow_the_rules() {
    // gcd(2^i, 2^j) = 2^min(i, j), gcd(2^i - 1, 2^j - 1) = 2^gcd(i, j) - 1.
    let mut state = 0x3F84_D5B5_B547_0917;
    for (i, j) in [
        (0, 0),
        (1, 64),
        (64, 64),
        (63, 129),
        (500, 1000),
        (8000, 16),
    ] {
        let (i, j) = (i.min(size(usize::MAX, 200)), j.min(size(usize::MAX, 200)));
        let mut x = vec![0; i / 8 + 1];
        x[i / 8] = 1 << (i % 8);
        let mut y = vec![0; j / 8 + 1];
        y[j / 8] = 1 << (j % 8);
        let (a, b) = pair(&mut state, &x, &y);
        check_all(&a, &b);
        let (a, b) = pair(&mut state, &vec![0xFF; i / 8 + 1], &vec![0xFF; j / 8 + 1]);
        check_all(&a, &b);
    }
}

#[test]
fn unnormalized_operands_count_as_their_values() {
    // Zero bytes above the most significant one, empty vectors and negative zero.
    let mut state = 0x9216_D5D9_8979_FB1B;
    for order in ORDERS {
        for z in zeros(order) {
            for other in [vec![12u8], vec![0x34; 20], vec![]] {
                let b = number(
                    &other,
                    below(&mut state, 2) == 0,
                    ORDERS[below(&mut state, 2)],
                );
                check_all(&z, &b);
            }
        }
        let mut x = random_magnitude(&mut state, 30);
        let y = random_magnitude(&mut state, 25);
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let (a, b) = pair(&mut state, &x, &y);
        check_all(&a, &b);
    }
}

#[test]
fn a_zero_modulus_panics_as_the_remainder_does() {
    for zero in zeros(ByteOrder::BigEndian) {
        for a in [
            Number::new_from_u8(3, ByteOrder::LittleEndian),
            Number::new(vec![0xAB; 40], ByteOrder::BigEndian),
        ] {
            assert_eq!(
                panic_message(|| {
                    a.mod_inverse(&zero);
                }),
                "attempt to calculate the remainder with a divisor of zero"
            );
        }
    }
}

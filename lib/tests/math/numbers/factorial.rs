use a::math::numbers::{ByteOrder, Number};

use super::common::*;

// References, on limbs of 64 bits, the least significant first: products and exact quotients by
// one limb, independent of the implementation.

/// `x m`, in place.
fn times(x: &mut Vec<u64>, m: u64) {
    let mut carry = 0u64;
    for limb in x.iter_mut() {
        let p = u128::from(*limb) * u128::from(m) + u128::from(carry);
        *limb = p as u64;
        carry = (p >> 64) as u64;
    }
    if carry != 0 {
        x.push(carry);
    }
}

/// `x / d`, in place, for a `d` that divides `x`.
fn divided(x: &mut Vec<u64>, d: u64) {
    let mut r = 0u128;
    for limb in x.iter_mut().rev() {
        let t = (r << 64) | u128::from(*limb);
        *limb = (t / u128::from(d)) as u64;
        r = t % u128::from(d);
    }
    assert_eq!(r, 0, "the reference divided inexactly");
    while x.last() == Some(&0) {
        x.pop();
    }
}

/// The bytes of the limbs `x`, least significant first, in the shortest form.
fn bytes(x: &[u64]) -> Vec<u8> {
    let mut v: Vec<u8> = x.iter().flat_map(|l| l.to_le_bytes()).collect();
    while v.len() > 1 && v.last() == Some(&0) {
        v.pop();
    }
    if v.is_empty() {
        v.push(0);
    }
    v
}

/// `n!`, one factor at a time.
fn factorial_reference(n: u64) -> Vec<u64> {
    let mut x = vec![1];
    for i in 2..=n {
        times(&mut x, i);
    }
    x
}

/// `C(n, k)`, as `C(n, i) = C(n, i - 1) (n - i + 1) / i` for `i` from 1 to `k`, each exact.
fn binomial_reference(n: u64, k: u64) -> Vec<u64> {
    if k > n {
        return vec![0];
    }
    let k = k.min(n - k);
    let mut x = vec![1];
    for i in 1..=k {
        times(&mut x, n - i + 1);
        divided(&mut x, i);
    }
    x
}

/// Checks `got` against the reference limbs `want`: in the shortest form, in byte order `order`,
/// not negative.
fn assert_value(got: &Number, want: &[u64], order: ByteOrder, what: &dyn Fn() -> String) {
    assert_result(got, &expected(&bytes(want), false), order, what);
}

// Factorials

#[test]
fn factorials_up_to_34_are_those_of_u128() {
    let mut f = 1u128;
    for n in 0..=34u64 {
        if n > 1 {
            f *= u128::from(n);
        }
        for order in ORDERS {
            let what = || format!("{n}!");
            assert_result(
                &Number::factorial(n, order),
                &expected(&f.to_le_bytes(), false),
                order,
                &what,
            );
        }
    }
    // 34! is the last below 2^128, and 35! has 17 bytes.
    assert_eq!(Number::factorial(35, ByteOrder::BigEndian).value.len(), 17);
}

#[test]
fn factorials_are_products_of_their_numbers() {
    // Across the direct method and the swinging factorials from 256, with one more halving at each
    // power of two from there.
    let mut x = vec![1u64];
    for n in 0..=size(1100, 70) as u64 {
        if n > 1 {
            times(&mut x, n);
        }
        let order = ORDERS[n as usize % 2];
        assert_value(&Number::factorial(n, order), &x, order, &|| format!("{n}!"));
    }
}

#[test]
fn long_factorials_are_exact() {
    for n in [2047, 2048, 3001].into_iter().filter(|_| !cfg!(miri)) {
        for order in ORDERS {
            let what = || format!("{n}!");
            assert_value(
                &Number::factorial(n, order),
                &factorial_reference(n),
                order,
                &what,
            );
        }
    }
}

#[test]
fn long_factorials_are_n_times_the_one_before() {
    for n in [8191, 8192, 30_000, 65_537]
        .into_iter()
        .filter(|&n| n <= size(usize::MAX, 300) as u64)
    {
        let f = Number::factorial(n, ByteOrder::LittleEndian);
        let before = Number::factorial(n - 1, ByteOrder::BigEndian);
        let product = &before * &Number::new_from_u64(n, ByteOrder::BigEndian);
        assert_eq!(
            magnitude_of(&f),
            magnitude_of(&product),
            "{n}! is not n (n - 1)!"
        );
        assert!(f.byte_order == ByteOrder::LittleEndian && !f.negative);
    }
}

/// The magnitude of `n`, least significant byte first.
fn magnitude_of(n: &Number) -> Vec<u8> {
    let mut v = le_bytes(n);
    while v.len() > 1 && v.last() == Some(&0) {
        v.pop();
    }
    v
}

// Binomial coefficients

/// For `k` from 2 to 65, the largest `n` for which `C(n, k) < 2^128`, computed apart with exact
/// integers: the coefficients up to them are computed in registers.
const SMALL_N: [u64; 64] = [
    18_446_744_073_709_551_615,
    12_686_161_381_664,
    9_506_325_305,
    132_496_420,
    7_910_345,
    1_080_110,
    246_703,
    79_266,
    32_301,
    15_633,
    8602,
    5222,
    3424,
    2386,
    1748,
    1333,
    1052,
    854,
    710,
    603,
    521,
    457,
    406,
    365,
    332,
    305,
    282,
    262,
    246,
    232,
    220,
    209,
    200,
    192,
    185,
    179,
    174,
    169,
    164,
    161,
    157,
    154,
    151,
    149,
    147,
    145,
    143,
    141,
    140,
    139,
    137,
    136,
    136,
    135,
    134,
    134,
    133,
    133,
    132,
    132,
    132,
    132,
    131,
    131,
];

#[test]
fn binomials_at_the_edges() {
    for order in ORDERS {
        for n in [0, 1, 2, 7, 1000, u64::MAX] {
            let what = || format!("binomials of {n}");
            assert_result(
                &Number::binomial(n, 0, order),
                &expected(&[1], false),
                order,
                &what,
            );
            assert_result(
                &Number::binomial(n, n, order),
                &expected(&[1], false),
                order,
                &what,
            );
            let n_bytes = n.to_le_bytes();
            if n > 0 {
                assert_result(
                    &Number::binomial(n, 1, order),
                    &expected(&n_bytes, false),
                    order,
                    &what,
                );
                assert_result(
                    &Number::binomial(n, n - 1, order),
                    &expected(&n_bytes, false),
                    order,
                    &what,
                );
            }
            for k in [n.wrapping_add(1), u64::MAX] {
                if k > n {
                    assert_result(&Number::binomial(n, k, order), &zero(), order, &what);
                }
            }
        }
    }
}

#[test]
fn binomials_make_pascals_triangle() {
    // In registers up to 131; above, by a division, and from `n = 400` by the primes for
    // `k > 5 sqrt(n)`.
    let order = ByteOrder::LittleEndian;
    let mut row: Vec<Vec<u8>> = vec![vec![1]];
    for n in 1..=size(440, 40) as u64 {
        let mut next = Vec::with_capacity(n as usize + 1);
        for k in 0..=n {
            let c = Number::binomial(n, k, order);
            let want = if k == 0 || k == n {
                Number::new(vec![1], order)
            } else {
                let left = Number::new(row[k as usize - 1].clone(), order);
                &left + &Number::new(row[k as usize].clone(), order)
            };
            assert!(same(&c, &want), "C({n}, {k}) = {}", describe(&c));
            next.push(c.value);
        }
        row = next;
    }
}

#[test]
fn binomials_below_2_to_the_128_are_computed_up_to_the_bound() {
    for (i, &n) in SMALL_N.iter().enumerate() {
        let k = i as u64 + 2;
        for (n, max) in [(n, 16), (n.saturating_add(1), 32)] {
            if n == u64::MAX && max == 32 {
                continue;
            }
            let c = Number::binomial(n, k, ByteOrder::BigEndian);
            let want = binomial_reference(n, k);
            let what = || format!("C({n}, {k})");
            assert_value(&c, &want, ByteOrder::BigEndian, &what);
            if max == 16 {
                assert!(c.value.len() <= 16, "{} has more than 16 bytes", what());
            } else {
                assert!(c.value.len() > 16, "{} has at most 16 bytes", what());
            }
        }
    }
}

#[test]
fn binomials_by_every_method_are_exact() {
    let mut pairs: Vec<(u64, u64)> = vec![
        // In registers and just above: a numerator of one limb a factor.
        (u64::MAX, 2),
        (u64::MAX, 3),
        (u64::MAX, 5),
        (1 << 40, 3),
        ((1 << 63) + 5, 30),
        // A divisor of one limb up to k = 25, of several from 26.
        (1_000_000_000_000, 25),
        (1_000_000_000_000, 26),
        (1_000_000, 100),
        (1_000_000, 255),
        // From 256, the division of vectors, by Hensel's method and beyond it by the division.
        (1_000_000, 256),
        (10_000_000, 300),
        (1_000_000_000, 3000),
        // By the primes: `k > 100` and `k > 5 sqrt(n)`.
        (300, 101),
        (1000, 159),
        (1000, 500),
        (5000, 2500),
        (20_000, 9_999),
    ];
    if cfg!(miri) {
        pairs.retain(|&(n, k)| k <= 30 || n <= 1000);
    }
    let mut state = 0x6A09E667_F3BCC908;
    for (n, k) in pairs {
        let order = ORDERS[below(&mut state, 2)];
        let want = binomial_reference(n, k);
        let what = || format!("C({n}, {k})");
        assert_value(&Number::binomial(n, k, order), &want, order, &what);
        assert_value(&Number::binomial(n, n - k, order), &want, order, &what);
    }
}

#[test]
fn binomials_around_the_methods_switches() {
    let mut state = 0xBB67AE85_84CAA73B;
    let mut pairs = Vec::new();
    for n in [131, 132, 400, 401, 1000, 2500, 40_000] {
        let switch = (5.0 * (n as f64).sqrt()) as u64;
        for k in [
            24,
            25,
            26,
            99,
            100,
            101,
            102,
            switch - 1,
            switch,
            switch + 1,
            n / 2,
        ] {
            if k <= n {
                pairs.push((n, k));
            }
        }
    }
    for _ in 0..size(200, 5) {
        let n = 2 + below(&mut state, 3000) as u64;
        pairs.push((n, below(&mut state, n as usize + 1) as u64));
    }
    for (n, k) in pairs
        .into_iter()
        .filter(|&(n, _)| n <= size(usize::MAX, 140) as u64)
    {
        let order = ORDERS[below(&mut state, 2)];
        let what = || format!("C({n}, {k})");
        assert_value(
            &Number::binomial(n, k, order),
            &binomial_reference(n, k),
            order,
            &what,
        );
    }
}

#[test]
fn factorials_and_binomials_through_every_path_even_under_miri() {
    // The swinging factorials from 256, one level and two; binomials from their primes, divided
    // by the odd part of `k!` of several limbs on the stack, and by a vector from `k = 256`.
    for n in [256, 300, 600] {
        let want = factorial_reference(n);
        let order = ORDERS[n as usize % 2];
        assert_value(&Number::factorial(n, order), &want, order, &|| {
            format!("{n}!")
        });
    }
    for (n, k) in [(300, 101), (300, 30), (3000, 256), (1 << 40, 26)] {
        let order = ORDERS[k as usize % 2];
        let what = || format!("C({n}, {k})");
        assert_value(
            &Number::binomial(n, k, order),
            &binomial_reference(n, k),
            order,
            &what,
        );
    }
}

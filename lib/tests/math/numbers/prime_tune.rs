//! The primality paths whose thresholds were measured again with the kernels of Montgomery's
//! products: below `2^32` the strong test to base 2 with the table of the composites that pass it,
//! the bound of the interval sieve of the next and the previous prime at its small end, and eight
//! limbs in registers only where the kernels are.

use a::math::numbers::{ByteOrder, Number};

use super::common::*;

/// `a b mod m` in `u128`.
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(m)) as u64
}

/// Whether the odd `n > 2` passes the strong test to base `a`, by products in `u128`.
fn strong(n: u64, a: u64) -> bool {
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    let mut x = 1u64;
    let (mut b, mut e) = (a % n, d);
    while e > 0 {
        if e & 1 == 1 {
            x = mul_mod(x, b, n);
        }
        b = mul_mod(b, b, n);
        e >>= 1;
    }
    if x == 1 || x == n - 1 {
        return true;
    }
    for _ in 1..s {
        x = mul_mod(x, x, n);
        if x == n - 1 {
            return true;
        }
    }
    false
}

/// Whether `n` is prime, independently of the implementation: the strong tests to the twelve prime
/// bases up to 37, which no composite below `3.3 * 10^24` passes (Sorenson and Webster, 2015).
fn is_prime_reference(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for p in BASES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    BASES.iter().all(|&a| strong(n, a))
}

/// The odd primes up to 256, the trial division of numbers below `2^64`.
fn trial_primes() -> Vec<u64> {
    (3..=256u64)
        .step_by(2)
        .filter(|&p| {
            (3..p)
                .step_by(2)
                .take_while(|d| d * d <= p)
                .all(|d| !p.is_multiple_of(d))
        })
        .collect()
}

/// Checks `is_probable_prime` of `n` against `expected`, in a random byte order, one time in four
/// with high zero bytes.
fn check(state: &mut u64, n: u64, expected: bool, what: &str) {
    let mut v = n.to_le_bytes().to_vec();
    if below(state, 4) == 0 {
        v.resize(v.len() + 1 + below(state, 5), 0);
    }
    let x = number(&v, false, ORDERS[below(state, 2)]);
    assert_eq!(
        x.is_probable_prime(),
        expected,
        "{what}: {n} is_probable_prime"
    );
}

/// The last of the composites below `2^32` with no prime factor up to 256 that pass the strong test
/// to base 2 (from Feitsma and Galway's list of the base-2 pseudoprimes).
const LAST_PSEUDOPRIMES_2: [u64; 12] = [
    4_234_224_601,
    4_237_212_061,
    4_244_022_301,
    4_244_663_651,
    4_247_990_917,
    4_250_920_459,
    4_251_904_273,
    4_255_695_013,
    4_271_267_333,
    4_275_011_401,
    4_277_526_901,
    4_278_305_651,
];

#[test]
fn numbers_below_2_32_that_pass_the_strong_test_to_base_2() {
    let mut state = 0x6A09E667_F3BCC909;
    let small = trial_primes();
    let passes = |n: u64| small.iter().all(|&p| !n.is_multiple_of(p)) && strong(n, 2);
    // Every odd number from `257^2`, the first that reaches the test, to `2^21` that passes the strong
    // test to base 2 with no prime factor up to 256: the primes and the 34 pseudoprimes there.
    let top = if cfg!(miri) {
        257 * 257 + 6000
    } else {
        1 << 21
    };
    let mut pseudoprimes = 0;
    let mut n = 257 * 257;
    while n < top {
        if passes(n) {
            let prime = is_prime_reference(n);
            pseudoprimes += usize::from(!prime);
            check(&mut state, n, prime, "below 2^21");
        }
        n += 2;
    }
    if !cfg!(miri) {
        assert_eq!(
            pseudoprimes, 34,
            "the base-2 pseudoprimes below 2^21 past the trial division"
        );
    }
    // The last pseudoprimes below `2^32`, the numbers beside them, and the odd numbers at the top.
    for n in LAST_PSEUDOPRIMES_2 {
        assert!(
            passes(n) && !is_prime_reference(n),
            "{n} is a base-2 pseudoprime"
        );
        for m in [n - 4, n - 2, n, n + 2, n + 4] {
            check(&mut state, m, is_prime_reference(m), "beside a pseudoprime");
        }
    }
    for n in ((1u64 << 32) - size(1 << 14, 200) as u64..1 << 32).step_by(2) {
        check(&mut state, n, is_prime_reference(n), "below 2^32");
    }
    // Random odd numbers of 17 to 32 bits.
    for _ in 0..size(30_000, 200) {
        let n = (xorshift(&mut state) >> (32 + below(&mut state, 16))) | 1;
        check(&mut state, n, is_prime_reference(n), "random below 2^32");
    }
}

/// Whether the number one above or below `x` (by `step`) up to `p`, exclusive, are all composite.
fn none_between(x: &Number, p: &Number, step: &Number, up: bool) -> bool {
    let mut y = if up { x + step } else { x - step };
    while value_of(&y) != value_of(p) {
        if y.is_probable_prime() {
            return false;
        }
        y = if up { &y + step } else { &y - step };
    }
    true
}

#[test]
fn next_and_previous_primes_at_the_small_end_of_the_sieve() {
    // From 65 to 100 bits the interval is sieved by the primes up to 500 and then `bits^2.5 / 124`:
    // the candidates left take Baillie–PSW without its trial division, so a prime factor between the
    // sieve's bound and the test's trial division must still be found by the test.
    let mut state = 0xBB67AE85_84CAA73B;
    let one = Number::new_from_u8(1, ByteOrder::LittleEndian);
    for k in 0..size(120, 3) {
        let bits = 65 + below(&mut state, 36);
        let mut le = random_magnitude(&mut state, bits.div_ceil(8));
        let top = bits - 8 * (le.len() - 1);
        let last = le.len() - 1;
        le[last] &= ((1u16 << top) - 1) as u8;
        le[last] |= 1 << (top - 1);
        let x = number(&le, false, ORDERS[k % 2]);
        let p = x.next_prime();
        assert!(p.is_probable_prime(), "next_prime of {}", describe(&x));
        assert!(
            none_between(&x, &p, &one, true),
            "a prime below next_prime of {}",
            describe(&x)
        );
        let q = x.prev_prime().expect("a prime below");
        assert!(q.is_probable_prime(), "prev_prime of {}", describe(&x));
        assert!(
            none_between(&x, &q, &one, false),
            "a prime above prev_prime of {}",
            describe(&x)
        );
    }
    // Products of two primes between 500 and 1000 times a prime, next to a start: composites that
    // the sieve's primes up to 500 leave, which the test must refuse.
    let pairs = [(503u64, 997u64), (509, 991), (521, 983), (997, 991)];
    for &(a, b) in &pairs[..size(4, 1)] {
        for c in [2_305_843_009_213_693_951u64, 9_223_372_036_854_775_783] {
            let n = &Number::new_from_u64(a * b, ByteOrder::LittleEndian)
                * &Number::new_from_u64(c, ByteOrder::LittleEndian);
            assert!(!n.is_probable_prime(), "{a} * {b} * {c}");
            let below = &n - &Number::new_from_u8(2, ByteOrder::LittleEndian);
            let p = below.next_prime();
            assert!(
                value_of(&p) != value_of(&n),
                "next_prime of {a} * {b} * {c} - 2"
            );
            let above = &n + &Number::new_from_u8(2, ByteOrder::LittleEndian);
            let q = above.prev_prime().expect("a prime below");
            assert!(
                value_of(&q) != value_of(&n),
                "prev_prime of {a} * {b} * {c} + 2"
            );
        }
    }
}

#[test]
fn eight_limbs_take_the_test_on_every_target() {
    // Numbers of 449 to 512 bits, eight limbs: in registers where the target has the kernels of
    // Montgomery's products, in memory elsewhere. The next and the previous prime of random odd
    // numbers (with no prime between for the first two), their products by 65,521, and products of
    // two primes of up to 256 and 264 bits.
    if cfg!(miri) {
        return;
    }
    let mut state = 0x3C6EF372_FE94F82B;
    let two = Number::new_from_u8(2, ByteOrder::LittleEndian);
    for k in 0..6 {
        let bits = 449 + below(&mut state, 64);
        let mut le = random_magnitude(&mut state, bits.div_ceil(8));
        let top = bits - 8 * (le.len() - 1);
        let last = le.len() - 1;
        le[last] &= ((1u16 << top) - 1) as u8;
        le[last] |= 1 << (top - 1);
        le[0] |= 1;
        let x = number(&le, false, ORDERS[k % 2]);
        let p = x.next_prime();
        assert!(p.is_probable_prime(), "next_prime of {}", describe(&x));
        if k < 2 {
            assert!(
                none_between(&x, &p, &two, true),
                "a prime below next_prime of {}",
                describe(&x)
            );
        }
        let q = x.prev_prime().expect("a prime below");
        assert!(q.is_probable_prime(), "prev_prime of {}", describe(&x));
        // A prime of up to 256 bits times one of up to 264: eight or nine limbs, composite.
        let small = number(
            &random_magnitude(&mut state, 32),
            false,
            ByteOrder::LittleEndian,
        );
        let r = small.next_prime();
        let s = number(
            &random_magnitude(&mut state, 33),
            false,
            ByteOrder::LittleEndian,
        )
        .next_prime();
        assert!(!(&r * &s).is_probable_prime(), "a product of two primes");
        assert!(!(&p * &Number::new_from_u32(65_521, ByteOrder::LittleEndian)).is_probable_prime());
    }
}

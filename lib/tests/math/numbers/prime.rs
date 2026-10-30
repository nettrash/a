use a::math::numbers::{ByteOrder, Number};

use super::common::*;

// References

/// Whether each number below `n` is prime, by a sieve of Eratosthenes.
fn sieve(n: usize) -> Vec<bool> {
    let mut prime = vec![true; n];
    for p in prime.iter_mut().take(2) {
        *p = false;
    }
    let mut i = 2;
    while i * i < n {
        if prime[i] {
            let mut j = i * i;
            while j < n {
                prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    prime
}

/// `a b mod m` in `u128`.
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(m)) as u64
}

/// Whether `n` is prime, worked out independently of the implementation: the Miller–Rabin test to
/// the twelve prime bases up to 37, which no composite below `3.3 * 10^24` passes (Sorenson and
/// Webster, 2015), so exact for every `u64`.
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
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'bases: for a in BASES {
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
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// The number `2^k + c` as little-endian bytes, `c` small, `|c| < 2^k`.
fn power_of_two_plus(k: usize, c: i64) -> Vec<u8> {
    let mut v = vec![0u8; k / 8 + 1];
    v[k / 8] = 1 << (k % 8);
    let n = number(&v, false, ByteOrder::LittleEndian);
    let c = Number::new_from_i64(c, ByteOrder::LittleEndian);
    le_bytes(&(&n + &c))
}

/// The number of the magnitude `le`, least significant byte first, in a random byte order, one time
/// in four with high zero bytes.
fn stored_number(state: &mut u64, le: &[u8]) -> Number {
    let mut v = le.to_vec();
    if below(state, 4) == 0 {
        v.resize(v.len() + 1 + below(state, 9), 0);
    }
    number(&v, false, ORDERS[below(state, 2)])
}

/// Checks `is_probable_prime` of the magnitude `le` against `expected`, in both byte orders.
fn check(state: &mut u64, le: &[u8], expected: bool, what: &str) {
    let n = stored_number(state, le);
    assert_eq!(
        n.is_probable_prime(),
        expected,
        "{what}: {} is_probable_prime",
        describe(&n)
    );
}

// Primality

#[test]
fn small_numbers_match_a_sieve() {
    let limit = size(200_000, 1500);
    let prime = sieve(limit);
    let mut state = 0x243F6A88_85A308D3;
    for (n, &p) in prime.iter().enumerate() {
        let le = (n as u64).to_le_bytes();
        check(&mut state, &le, p, "sieve");
        // A negative number is never prime.
        if n > 0 && below(&mut state, 8) == 0 {
            let neg = number(&le, true, ORDERS[below(&mut state, 2)]);
            assert!(!neg.is_probable_prime(), "-{n}");
        }
    }
    for order in ORDERS {
        for z in zeros(order) {
            assert!(!z.is_probable_prime(), "{}", describe(&z));
        }
    }
}

#[test]
fn numbers_below_2_64_match_the_reference() {
    let mut state = 0x13198A2E_03707344;
    let prime = sieve(1 << 16);
    let small: Vec<u64> = (3..1 << 16).filter(|&i| prime[i as usize]).collect();
    for _ in 0..size(200_000, 300) {
        let n = match below(&mut state, 5) {
            0 => xorshift(&mut state) >> below(&mut state, 64),
            1 => xorshift(&mut state) | 1,
            2 => {
                // A product of two primes of up to 16 bits, or of three.
                let p = small[below(&mut state, small.len())];
                let q = small[below(&mut state, small.len())];
                let r = if below(&mut state, 2) == 0 {
                    small[below(&mut state, 100)]
                } else {
                    1
                };
                p * q * r
            }
            3 => {
                // A square of a prime, or a square times a prime.
                let p = small[below(&mut state, small.len())];
                p * p
                    * if below(&mut state, 2) == 0 {
                        1
                    } else {
                        small[below(&mut state, 50)]
                    }
            }
            _ => u64::MAX - below(&mut state, 1000) as u64,
        };
        check(&mut state, &n.to_le_bytes(), is_prime_reference(n), "u64");
    }
}

/// Composite numbers that pass the strong test to base 2: Pomerance, Selfridge and Wagstaff's
/// smallest ones and some much larger (3215031751 passes bases 2, 3, 5 and 7; 3825123056546413051
/// all prime bases to 23); Carmichael numbers, which pass Fermat's test to every base prime to
/// them; and strong Lucas pseudoprimes with Selfridge's parameters (Baillie and Wagstaff), which
/// pass the other half of the test.
const PSEUDOPRIMES: [u64; 40] = [
    2047,
    3277,
    4033,
    4681,
    8321,
    15841,
    29341,
    42799,
    49141,
    52633,
    65281,
    74665,
    80581,
    85489,
    88357,
    90751,
    3215031751,
    2152302898747,
    3474749660383,
    341550071728321,
    3825123056546413051,
    561,
    1105,
    1729,
    2465,
    2821,
    6601,
    8911,
    41041,
    825265,
    5459,
    5777,
    10877,
    16109,
    18971,
    22499,
    24569,
    25199,
    40309,
    58519,
];

#[test]
fn known_pseudoprimes_are_composite() {
    let mut state = 0xA4093822_299F31D0;
    for n in PSEUDOPRIMES {
        assert!(!is_prime_reference(n), "{n} is prime by the reference");
        check(&mut state, &n.to_le_bytes(), false, "pseudoprime");
    }
}

#[test]
fn known_primes_and_composites_beyond_2_64() {
    let mut state = 0x082EFA98_EC4E6C89;
    // Mersenne primes (whose `n + 1` is a power of two, the Lucas test's shortest case) and the
    // largest primes below powers of two; Mersenne composites; and the first primes above.
    let mut primes = vec![
        power_of_two_plus(61, -1),
        power_of_two_plus(64, -59),
        power_of_two_plus(64, 13),
        power_of_two_plus(89, -1),
        power_of_two_plus(107, -1),
        power_of_two_plus(127, -1),
        power_of_two_plus(128, -159),
        power_of_two_plus(128, 51),
        power_of_two_plus(192, -237),
        power_of_two_plus(255, -19),
        power_of_two_plus(256, -189),
        power_of_two_plus(256, 297),
    ];
    let mut composites = vec![
        power_of_two_plus(67, -1),
        power_of_two_plus(101, -1),
        power_of_two_plus(128, -1),
        power_of_two_plus(128, 1),
        power_of_two_plus(257, -1),
        power_of_two_plus(256, -187),
    ];
    if !cfg!(miri) {
        primes.extend([
            power_of_two_plus(521, -1),
            power_of_two_plus(512, -569),
            power_of_two_plus(607, -1),
            power_of_two_plus(1279, -1),
        ]);
        composites.extend([power_of_two_plus(512, -567), power_of_two_plus(1024, -1)]);
    }
    for p in &primes {
        check(&mut state, p, true, "prime");
        // Its square, and its product by another, are composite.
        let n = number(p, false, ByteOrder::LittleEndian);
        check(&mut state, &le_bytes(&(&n * &n)), false, "square");
        let q = number(
            &primes[below(&mut state, primes.len())],
            false,
            ByteOrder::BigEndian,
        );
        check(&mut state, &le_bytes(&(&n * &q)), false, "product");
    }
    for c in &composites {
        check(&mut state, c, false, "composite");
    }
}

#[test]
fn products_of_primes_and_numbers_that_look_like_squares() {
    let mut state = 0xC0AC29B7_C97C50DD;
    // `t = 1 + 8 * 3 * 5 * ... * 47` is 1 modulo 8 and modulo each of those primes, so `t s^2` is a
    // square modulo all of them without being one: the square root decides, and Selfridge's search
    // goes past 47, as every symbol up to there is 1.
    let t = Number::new_from_u64(1 + 8 * 307_444_891_294_245_705, ByteOrder::LittleEndian);
    for k in [64, 89, 127] {
        let s = number(&power_of_two_plus(k, -1), false, ByteOrder::BigEndian);
        let n = &t * &(&s * &s);
        check(&mut state, &le_bytes(&n), false, "t s^2");
    }
    // Products of two primes found by `next_prime`, of the sizes of every path: two limbs, the
    // registers to seven, and the limbs from eight.
    for bytes in [
        5,
        8,
        9,
        12,
        16,
        20,
        24,
        28,
        32,
        size(48, 20),
        size(64, 20),
        size(200, 20),
    ] {
        let a = number(
            &random_magnitude(&mut state, bytes),
            false,
            ByteOrder::BigEndian,
        );
        let b = number(
            &random_magnitude(&mut state, bytes),
            false,
            ByteOrder::LittleEndian,
        );
        let (p, q) = (a.next_prime(), b.next_prime());
        assert!(p.is_probable_prime() && q.is_probable_prime());
        let n = &p * &q;
        check(&mut state, &le_bytes(&n), false, "p q");
        check(&mut state, &le_bytes(&(&n * &p)), false, "p^2 q");
    }
}

#[test]
fn random_numbers_match_trial_division_and_miller_rabin() {
    // Odd numbers of two to nine limbs: those with a factor below 2^16 by trial division, the
    // others by the Miller–Rabin test to the bases 2 to 37, with `pow_mod`; no composite is known
    // to pass that and fail Baillie–PSW, or the other way round.
    let mut state = 0x9216D5D9_8979FB1B;
    let prime = sieve(1 << 16);
    let small: Vec<u32> = (3..1u32 << 16).filter(|&i| prime[i as usize]).collect();
    let bases: Vec<Number> = [2u8, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        .iter()
        .map(|&b| Number::new_from_u8(b, ByteOrder::BigEndian))
        .collect();
    for _ in 0..size(400, 6) {
        let len = 9 + below(&mut state, size(64, 32) - 9);
        let mut le = random_magnitude(&mut state, len);
        le[0] |= 1;
        let n = number(&le, false, ByteOrder::LittleEndian);
        let has_factor = small.iter().any(|&p| {
            let r = le
                .iter()
                .rev()
                .fold(0u64, |r, &b| ((r << 8) | u64::from(b)) % u64::from(p));
            r == 0
        });
        let expected = !has_factor && {
            let one = Number::new_from_u8(1, ByteOrder::LittleEndian);
            let n1 = &n - &one;
            let s = le_bytes(&n1)
                .iter()
                .enumerate()
                .find(|(_, b)| **b != 0)
                .map(|(i, b)| 8 * i + b.trailing_zeros() as usize)
                .unwrap_or(0);
            let d = &n1 >> s;
            bases.iter().all(|a| {
                let mut x = a.pow_mod(&d, &n);
                if value_of(&x) == value_of(&one) || value_of(&x) == value_of(&n1) {
                    return true;
                }
                for _ in 1..s {
                    x = x.pow_mod(&Number::new_from_u8(2, ByteOrder::BigEndian), &n);
                    if value_of(&x) == value_of(&n1) {
                        return true;
                    }
                }
                false
            })
        };
        check(&mut state, &le, expected, "random");
    }
}

/// A prime of 6480 bits, 102 limbs, which GMP's `mpz_nextprime` found: long enough for REDC by
/// products, from 100 limbs, in both halves of the test.
const P6480: &str = concat!(
    "1b51807cd751f2cb98d9ba2c6e58686e06abd238914129ced921e6d3e68773f62c9b2109d497fa91",
    "374d4984f37d03193365e20c60977d9e4006325e3c54ab3b6bee48ad45c800202cdd7d2d37170d03",
    "134fc4c76509c8c753078087fa1c14c7c062e3861aa1a0646e5822a579c0a5acbbc6a786dd9cf1a4",
    "ad4fa0f7172fc4bdd854083cb511b4226ce507675392930beec38c3a9dbebdfeafc824d72dca14fd",
    "865066e7b2d141396d2b7bc897691bd03ae7f64287a29fa2329b0bc6f182c53d65baff7919476145",
    "33dc6830ab98ddc4adfa28366e40d9c68b4765ee40ad57b4ff152d7331d2367242d571b8abfa6eb1",
    "3351c44c95bedb4b5bba952bbad1a0adda8b8878d509809d79d85c3a836f8eb113495f755153e2b7",
    "1d7961ca8f315f7f39ea501cd79600ab2b73aaff6b43c83cc1a2ef1953c4ad8846a38250871b4b86",
    "10702ad7c838129592919864eb6586f780bf1113f4d1aa7691ac7ec1ce0872105650f5841ce9e47e",
    "dff05c2e3a1eafc8623878ffc0a1418c2e9194a1d232c3a0d93127c2897d7d219ec9e4f9a25a3cb5",
    "b4d95121e0a94d37cc60000e4e7c2adf96a59ae9615bd68bc0b7b8422e524e561e87c8cf8860df4b",
    "213559f646a794a5e66e02ed79375315c40aa9174f697b7bb02e38f9b28f880cd1a44887ea1edff8",
    "1658bdae7a87690dd681c4ebeaa757490dea5868e32931a4bf79399628837dfcfbe0a9d44412d1aa",
    "bdb6ad2577fabfe0efded3b29d3fdadda808818e703685d2d3a4e57dd681989b3dacd5ca97333f03",
    "a442976d9260d3aea180ad7f3382b7fc00fdb0587c48528df62cbbfcd513c28b69fcc75765b2cbe2",
    "6aff753f9a522662efc8382c9ba4894545facbba857553f4b955e0ef77e0e990d1ec7e301206128b",
    "949371e03a99963685d27a649307f2ebae1c0268bf67b4121cffbc4010aba4bd57829b6761210327",
    "13a89dfc1e38bf612bb85e66b12bece7610549b25a2ea51fcea7b01ed753b6853fb6a5eef55bba62",
    "2c26a349f27c4c588f57c444081c87552555d6468bf6182a367278a592e91541e620f55f515f2d59",
    "acc11bae91e8b3aecb8270160674fb2d9cd59470463c87ec1c40b1b0eddad7fa048582d3f2cd33e4",
    "10c3f026f5316d73e389",
);

#[test]
fn long_numbers_take_the_reduction_by_products() {
    // Under Miri, which would take hours, not run.
    if cfg!(miri) {
        return;
    }
    let p = Number::from_str_radix(P6480, 16, ByteOrder::BigEndian).expect("hexadecimal");
    assert!(p.is_probable_prime(), "the 6480-bit prime");
    // 2^3217 - 1 is a Mersenne prime of 51 limbs; its square has 101, its product by `p` 153.
    let m = number(&power_of_two_plus(3217, -1), false, ByteOrder::LittleEndian);
    assert!(m.is_probable_prime(), "2^3217 - 1");
    assert!(!(&m * &m).is_probable_prime(), "(2^3217 - 1)^2");
    assert!(!(&p * &m).is_probable_prime(), "p (2^3217 - 1)");
}

// The next prime

#[test]
fn next_primes_of_small_numbers_match_a_sieve() {
    let limit = size(100_000, 600);
    let prime = sieve(limit + 200);
    let mut state = 0x452821E6_38D01377;
    let mut next = vec![0; limit];
    let mut p = limit + 200;
    for i in (0..limit + 200).rev() {
        if i < limit {
            next[i] = p;
        }
        if prime[i] {
            p = i;
        }
    }
    for (n, &expected) in next.iter().enumerate() {
        let x = stored_number(&mut state, &(n as u64).to_le_bytes());
        let r = x.next_prime();
        assert_result(&r, &expected_i128(expected as i128), x.byte_order, &|| {
            format!("next_prime of {n}")
        });
    }
    for order in ORDERS {
        for x in zeros(order)
            .into_iter()
            .chain([number(&[5], true, order), number(&[0xFF; 40], true, order)])
        {
            assert_result(&x.next_prime(), &expected_i128(2), order, &|| {
                format!("next_prime of {}", describe(&x))
            });
        }
    }
}

#[test]
fn next_primes_across_powers_of_two() {
    let mut state = 0xBE5466CF_34E90C6C;
    // (from, next prime): the largest primes below 2^64, 2^128 and 2^256 and the first above, from
    // just below each.
    let mut cases = vec![
        ((64, -60), (64, -59)),
        ((64, -59), (64, 13)),
        ((64, -1), (64, 13)),
        ((64, 0), (64, 13)),
        ((128, -160), (128, -159)),
        ((128, -159), (128, 51)),
        ((128, 0), (128, 51)),
        ((256, -190), (256, -189)),
        ((256, 0), (256, 297)),
    ];
    if !cfg!(miri) {
        cases.push(((512, -600), (512, -569)));
    }
    for ((k, c), (kp, cp)) in cases {
        let x = stored_number(&mut state, &power_of_two_plus(k, c));
        let r = x.next_prime();
        let p = power_of_two_plus(kp, cp);
        assert_result(&r, &expected(&p, false), x.byte_order, &|| {
            format!("next_prime of 2^{k} + {c}")
        });
    }
}

#[test]
fn next_primes_have_no_prime_before_them() {
    let mut state = 0x9216D5D9_8979FB1B;
    for bytes in [9, 16, 17, 32, size(64, 17), size(128, 17)] {
        let x = number(
            &random_magnitude(&mut state, bytes),
            false,
            ORDERS[below(&mut state, 2)],
        );
        let p = x.next_prime();
        assert!(p.is_probable_prime(), "next_prime of {}", describe(&x));
        let one = Number::new_from_u8(1, ByteOrder::LittleEndian);
        let mut y = &x + &one;
        while !(&y - &p).value.iter().all(|&b| b == 0) {
            assert!(
                !y.is_probable_prime(),
                "{} is prime, below next_prime {}",
                describe(&y),
                describe(&p)
            );
            y = &y + &one;
        }
    }
}

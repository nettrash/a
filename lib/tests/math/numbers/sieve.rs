//! Tests of the prime sieves and prime counting: `primes`, `Primes`, `prime_pi` and `nth_prime`.
//!
//! The references are a sieve of Eratosthenes written here, the Baillie–PSW test of `Number`
//! (exact below `2^64`, and checked in `prime.rs`), and values computed by primecount 8.8 and
//! primesieve 12.16 (the script that printed them is kept with the measurements of the work).

use a::math::numbers::{ByteOrder, Number, nth_prime, prime_pi, primes};

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

/// The primes from `a` to `b`, by a sieve of the range with the primes up to the square root of
/// `b` from a sieve of their own.
fn primes_between(a: u64, b: u64) -> Vec<u64> {
    let root = b.isqrt() as usize;
    let small = sieve(root + 1);
    let mut prime = vec![true; (b - a + 1) as usize];
    for (p, _) in small.iter().enumerate().filter(|&(_, &is)| is) {
        let p = p as u64;
        let mut m = (a.div_ceil(p)).max(p) * p;
        while m <= b {
            prime[(m - a) as usize] = false;
            m += p;
        }
    }
    (a..=b)
        .zip(prime)
        .filter(|&(n, is)| is && n >= 2)
        .map(|(n, _)| n)
        .collect()
}

/// Whether `n` is prime, by the Baillie–PSW test of `Number`, exact below `2^64`.
fn is_prime(n: u64) -> bool {
    Number::new_from_u64(n, ByteOrder::LittleEndian).is_probable_prime()
}

/// The values of `π(10^k)`, `k` from 0 to 19 (Oliveira e Silva, OEIS A006880).
const PI_POWERS: [u64; 20] = [
    0,
    4,
    25,
    168,
    1229,
    9592,
    78498,
    664579,
    5761455,
    50847534,
    455052511,
    4118054813,
    37607912018,
    346065536839,
    3204941750802,
    29844570422669,
    279238341033925,
    2623557157654233,
    24739954287740860,
    234057667276344607,
];

/// The `10^k`-th primes, `k` from 0 to 12 (OEIS A006988).
const NTH_POWERS: [u64; 13] = [
    2,
    29,
    541,
    7919,
    104729,
    1299709,
    15485863,
    179424673,
    2038074743,
    22801763489,
    252097800623,
    2760727302517,
    29996224275833,
];

/// `(x, π(x))` for random `x` from `2^17` to `2^42`, by primecount 8.8.
const PI: [(u64, u64); 48] = [
    (97872, 9408),
    (114911, 10866),
    (203374, 18255),
    (194309, 17522),
    (353895, 30283),
    (331845, 28552),
    (1020971, 80043),
    (1014781, 79594),
    (1930546, 144165),
    (1146605, 89056),
    (3537768, 252618),
    (3995766, 282883),
    (5380550, 373102),
    (4328365, 304681),
    (16349006, 1052077),
    (11614988, 764420),
    (17057534, 1094770),
    (62207336, 3685192),
    (36887042, 2255183),
    (84200812, 4900032),
    (72993347, 4283696),
    (173362571, 9680539),
    (259424299, 14166260),
    (320036497, 17277063),
    (268572892, 14637958),
    (936011491, 47754811),
    (932993835, 47608468),
    (1313615436, 65877345),
    (2145510756, 105005638),
    (2694738410, 130427802),
    (2860472205, 138049242),
    (5839563421, 272415463),
    (11558800226, 522529784),
    (9872301020, 449504984),
    (21026758521, 925450375),
    (21242044337, 934506667),
    (63539518010, 2666490928),
    (35835132117, 1540971624),
    (85050256556, 3525975539),
    (71660874216, 2992177703),
    (251190702309, 9965448403),
    (189668472238, 7609658120),
    (315318605185, 12397544655),
    (466061602875, 18046641054),
    (600292143786, 23018273438),
    (862928081311, 32634157525),
    (1290868130215, 48084387038),
    (4389151705357, 156355475034),
];

/// `(n, p_n)` for random `n` from `2^10` to `2^36`, by primecount 8.8.
const NTH: [(u64, u64); 30] = [
    (543, 3919),
    (799, 6131),
    (1268, 10333),
    (2287, 20231),
    (4596, 44159),
    (10353, 108893),
    (24487, 280673),
    (48227, 588191),
    (70424, 888163),
    (223629, 3102271),
    (155611, 2097143),
    (341021, 4885583),
    (750543, 11390723),
    (1757394, 28273319),
    (3893559, 65954569),
    (5168652, 89111719),
    (15929872, 293708347),
    (24370361, 460316239),
    (42442393, 826588897),
    (99952291, 2037051277),
    (71915629, 1440686053),
    (188826136, 3974888279),
    (269053580, 5763971867),
    (643487144, 14375131751),
    (1213875770, 27925141081),
    (3354732682, 80745103913),
    (4689404656, 114511068797),
    (10897780024, 275708047061),
    (17889780769, 461847853157),
    (50044185380, 1345560779177),
];

/// `π(2^64 - 1)`, and the largest primes below `2^64`, largest first (primesieve 12.16).
const PI_MAX: u64 = 425_656_284_035_217_743;
const TOP: [u64; 4] = [
    18446744073709551557,
    18446744073709551533,
    18446744073709551521,
    18446744073709551437,
];

// The primes in a range

#[test]
fn primes_of_small_ranges_match_a_sieve() {
    let n = size(500, 60);
    let prime = sieve(n + 1);
    let all: Vec<u64> = (0..=n as u64).filter(|&k| prime[k as usize]).collect();
    for a in 0..=n as u64 {
        for b in a..=n as u64 {
            let got: Vec<u64> = primes(a..=b).collect();
            let want: Vec<u64> = all.iter().copied().filter(|&p| a <= p && p <= b).collect();
            assert_eq!(got, want, "primes({a}..={b})");
        }
    }
}

#[test]
fn primes_of_ranges_match_a_sieve_of_the_range() {
    // Ranges whose sieves take small, medium and large primes, by their place and width.
    let mut state = 0x6A09E667_F3BCC908;
    let cases: Vec<(u64, u64)> = if cfg!(miri) {
        // The last with large primes, from 3 times a segment's bytes, 393,216, to 400,000.
        vec![
            (0, 40_000),
            (100_000_000, 3000),
            (1_000_000_007, 200),
            (160_000_000_000, 100),
        ]
    } else {
        let mut v = vec![
            (0, 3_000_000),
            (10_000_000, 5_000_000),
            (1_000_000_000, 4_000_000),
            (1_000_000_000_000, 200_000),
            (1_000_000_000_000_000, 50_000),
        ];
        for _ in 0..20 {
            let bits = 20 + below(&mut state, 21) as u32;
            let a = xorshift(&mut state) >> (64 - bits);
            let w = xorshift(&mut state) % 300_000;
            v.push((a, w));
        }
        v
    };
    for (a, w) in cases {
        let b = a + w;
        let got: Vec<u64> = primes(a..=b).collect();
        assert_eq!(got, primes_between(a, b), "primes({a}..={b})");
        assert_eq!(
            primes(a..=b).count(),
            got.len(),
            "primes({a}..={b}).count()"
        );
        let sum = got.iter().fold(0u64, |s, &p| s.wrapping_add(p));
        assert_eq!(
            primes(a..=b).fold(0u64, |s, p| s.wrapping_add(p)),
            sum,
            "primes({a}..={b}).fold()"
        );
    }
}

#[test]
fn primes_near_2_64_match_the_test() {
    // The largest primes, and stretches below `2^64` and `10^19`, sieved by the smaller primes and
    // the numbers left tested, or by all of them.
    let width = size(20_000, 300) as u64;
    for top in [u64::MAX, 10_000_000_000_000_000_000, 1 << 63] {
        let a = top - width;
        let got: Vec<u64> = primes(a..=top).collect();
        let want: Vec<u64> = (a..=top).filter(|&n| is_prime(n)).collect();
        assert_eq!(got, want, "primes({a}..={top})");
        assert_eq!(primes(a..=top).count(), want.len());
    }
    let top: Vec<u64> = primes(u64::MAX - 120..).collect();
    assert_eq!(top, [TOP[2], TOP[1], TOP[0]]);
    assert_eq!(primes(u64::MAX - 58..).collect::<Vec<_>>(), [TOP[0]]);
    assert_eq!(primes(u64::MAX - 57..).next(), None);
    assert_eq!(primes(u64::MAX..).next(), None);
    assert_eq!(primes(u64::MAX..=u64::MAX).count(), 0);
}

#[test]
fn primes_from_far_up_come_quickly() {
    // An unbounded range sieves a short stretch first: the first primes from `10^18` and from
    // just below `2^64`, then a few thousand more, checked by the test.
    let n = size(3000, 20);
    for a in [1_000_000_000_000_000_000u64, u64::MAX - 1_000_000] {
        let got: Vec<u64> = primes(a..).take(n).collect();
        assert_eq!(got.len(), n);
        let mut prev = a - 1;
        for &p in &got {
            assert!(is_prime(p), "{p} from primes({a}..) is prime");
            assert!(
                (prev + 1..p).all(|q| !is_prime(q)),
                "primes({a}..) leaves out none below {p}"
            );
            prev = p;
        }
    }
}

#[test]
fn unbounded_ranges_go_through_their_stretches() {
    // From a small start, the full sieve takes stretches of 4096 numbers, then four times as many
    // each time: the primes taken one by one cross several of their ends.
    let n = size(30_000, 1500);
    let prime = sieve(600_000);
    for a in [0u64, 1000, 65_537] {
        let got: Vec<u64> = primes(a..).take(n).collect();
        let want: Vec<u64> = (a..600_000)
            .filter(|&k| prime[k as usize])
            .take(n)
            .collect();
        assert_eq!(got, want, "primes({a}..)");
        let mut it = primes(a..);
        let mut by_next = Vec::new();
        while by_next.len() < n {
            by_next.push(it.next().expect("a prime"));
        }
        assert_eq!(by_next, want, "primes({a}..) by next");
    }
}

#[test]
fn ranges_take_every_kind_of_bound() {
    assert_eq!(
        primes(10..=30).collect::<Vec<_>>(),
        [11, 13, 17, 19, 23, 29]
    );
    assert_eq!(primes(11..29).collect::<Vec<_>>(), [11, 13, 17, 19, 23]);
    assert_eq!(primes(..10).collect::<Vec<_>>(), [2, 3, 5, 7]);
    assert_eq!(primes(..=11).collect::<Vec<_>>(), [2, 3, 5, 7, 11]);
    assert_eq!(primes(0..2).count(), 0);
    assert_eq!(primes(2..=2).collect::<Vec<_>>(), [2]);
    assert_eq!(primes(7..=7).collect::<Vec<_>>(), [7]);
    assert_eq!(primes(8..=10).count(), 0);
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = primes(30..10);
    assert_eq!(reversed.count(), 0);
    use core::ops::Bound::{Excluded, Included, Unbounded};
    assert_eq!(
        primes((Excluded(2), Included(7))).collect::<Vec<_>>(),
        [3, 5, 7]
    );
    assert_eq!(primes((Excluded(u64::MAX), Unbounded)).count(), 0);
    assert_eq!(primes((Included(0), Excluded(0))).count(), 0);
    assert_eq!(
        primes(..).take(10).collect::<Vec<_>>(),
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    );
}

#[test]
fn the_iterator_is_fused_clones_and_counts() {
    let mut it = primes(100..=200);
    assert_eq!(it.next(), Some(101));
    let rest: Vec<u64> = it.clone().collect();
    assert_eq!(rest.len(), 20);
    assert_eq!(it.clone().count(), 20);
    assert_eq!(it.clone().last(), Some(199));
    assert_eq!(it.clone().nth(1), Some(107));
    let all: Vec<u64> = it.by_ref().collect();
    assert_eq!(all, rest);
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.size_hint(), (0, Some(0)));
    let (a, b) = (30, 10);
    assert_eq!(primes(a..b).size_hint(), (0, Some(0)));
    let (low, high) = primes(..1000).size_hint();
    assert!(low <= 168 && high.is_none_or(|h| h >= 168));
    assert!(format!("{:?}", primes(10..20)).starts_with("Primes"));
    // A count after a part was taken, from the middle of a batch and of a segment.
    let n = size(200_000, 3000) as u64;
    let mut it = primes(0..=n);
    let taken = it.by_ref().take(1500).count();
    assert_eq!(taken + it.count(), prime_pi(n) as usize);
}

// π(x)

#[test]
fn prime_pi_of_small_numbers_counts_the_primes() {
    // Every number of the table, to `2^16`, and past it, around the crossover to the sieve and to
    // the sieve's crossover, of `2 10^5`, to Deléglise and Rivat's method, and a sample between.
    let n = 250_000;
    let prime = sieve(n);
    let mut count = 0;
    for (x, &p) in prime.iter().enumerate() {
        count += u64::from(p);
        let edge = x < 70_000 || (199_000..201_000).contains(&x);
        let stride = if cfg!(miri) { 997 } else { 61 };
        if (edge && !cfg!(miri)) || x % stride == 0 || (cfg!(miri) && (65_530..65_540).contains(&x))
        {
            assert_eq!(prime_pi(x as u64), count, "pi({x})");
        }
    }
}

#[test]
fn prime_pi_matches_known_values() {
    let k = if cfg!(miri) { 6 } else { 14 };
    for (e, &want) in PI_POWERS.iter().enumerate().take(k) {
        let x = 10u64.pow(e as u32);
        assert_eq!(prime_pi(x), want, "pi(10^{e})");
        assert_eq!(prime_pi(x - 1), want, "pi(10^{e} - 1)");
    }
    for &(x, want) in PI.iter().take(size(48, 12)) {
        assert_eq!(prime_pi(x), want, "pi({x})");
    }
}

#[test]
fn prime_pi_agrees_with_the_sieve_on_ranges() {
    // `π(b) - π(a - 1)` against the primes of `[a, b]`, for `a` and `b` through Deléglise and
    // Rivat's method.
    let mut state = 0xBB67AE85_84CAA73B;
    for _ in 0..size(30, 2) {
        let bits = 18 + below(&mut state, if cfg!(miri) { 2 } else { 20 }) as u32;
        let a = (xorshift(&mut state) >> (64 - bits)).max(2);
        let b = a + xorshift(&mut state) % 100_000;
        assert_eq!(
            prime_pi(b) - prime_pi(a - 1),
            primes(a..=b).count() as u64,
            "pi({b}) - pi({})",
            a - 1
        );
    }
}

// The n-th prime

#[test]
fn nth_prime_of_small_indices_is_the_sieve_s() {
    // Every index of the table, to `π(2^16) = 6542`, then a sample to `10^5`: through the sieve
    // from `2^16` to 18,000 and through `π` above.
    let prime = sieve(size(1_300_000, 210_000));
    let mut k = 0u64;
    for (p, &is) in prime.iter().enumerate() {
        if is {
            k += 1;
            let check = if cfg!(miri) {
                k.is_multiple_of(211) || [6542, 6543, 18_000, 18_001].contains(&k)
            } else {
                k <= 6600 || k.is_multiple_of(97)
            };
            if check {
                assert_eq!(nth_prime(k), p as u64, "nth_prime({k})");
            }
        }
    }
}

#[test]
fn nth_prime_matches_known_values() {
    let k = if cfg!(miri) { 4 } else { 12 };
    for (e, &want) in NTH_POWERS.iter().enumerate().take(k) {
        assert_eq!(nth_prime(10u64.pow(e as u32)), want, "nth_prime(10^{e})");
    }
    for &(n, want) in NTH.iter().take(size(30, 6)) {
        assert_eq!(nth_prime(n), want, "nth_prime({n})");
        assert_eq!(prime_pi(want), n, "pi(nth_prime({n}))");
    }
}

#[test]
fn nth_prime_reaches_the_last_prime_below_2_64() {
    assert_eq!(nth_prime(PI_MAX), TOP[0]);
    if !cfg!(miri) {
        // Down from `2^64 - 1`, whose `π` is known.
        for (d, &p) in TOP.iter().enumerate().skip(1) {
            assert_eq!(nth_prime(PI_MAX - d as u64), p, "nth_prime(PI_MAX - {d})");
        }
    }
}

#[test]
#[should_panic(expected = "there is no 0th prime")]
fn nth_prime_of_zero_panics() {
    nth_prime(0);
}

#[test]
#[should_panic(expected = "is above u64::MAX")]
fn nth_prime_past_the_last_prime_panics() {
    nth_prime(PI_MAX + 1);
}

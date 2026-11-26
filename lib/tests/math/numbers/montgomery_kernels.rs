//! Montgomery's kernels at every length they take, through the methods that run on them: the
//! Baillie–PSW test on primes and composites of two to eighteen limbs, and the factoring methods
//! on the lengths of the kernels of residues in memory.

use a::math::numbers::{ByteOrder, Number, factor};

use super::common::*;

/// `2^bits`.
fn power_of_two(bits: usize, order: ByteOrder) -> Number {
    Number::new_from_u8(1, order) << bits
}

/// For `L` limbs, `k` such that `2^(64L) - k` is the largest prime below `2^(64L)` (the top bit
/// set), `2^(64L - 1) + k` the smallest above `2^(64L - 1)`, and `2^(64L - 2) - k` the largest
/// below `2^(64L - 2)` (the top limb below `2^62`): by PARI/GP 2.19's `precprime` and `nextprime`.
const PRIMES: [(usize, u64, u64, u64); 17] = [
    (2, 159, 29, 137),
    (3, 237, 5, 11),
    (4, 189, 95, 245),
    (5, 197, 9, 165),
    (6, 317, 369, 105),
    (7, 203, 99, 77),
    (8, 569, 111, 75),
    (9, 789, 131, 105),
    (10, 305, 413, 405),
    (11, 245, 203, 87),
    (12, 825, 699, 641),
    (13, 143, 129, 153),
    (14, 213, 359, 881),
    (15, 167, 233, 1127),
    (16, 105, 1155, 755),
    (17, 89, 35, 3323),
    (18, 927, 635, 237),
];

/// Chernick's Carmichael numbers `(6k + 1)(12k + 1)(18k + 1)` of `L` limbs, by PARI/GP: Fermat
/// pseudoprimes to every base prime to them, which the strong test and the Lucas test tell.
const CARMICHAEL: [(usize, &str); 16] = [
    (2, "639683812896"),
    (3, "1690201975843073111"),
    (4, "4465929324726532288958626"),
    (5, "11800083669588649198133832738475"),
    (6, "31178723281255485436861532007021377701"),
    (7, "82381855304504859216498938435695925675074091"),
    (8, "217673123501261273380443317974823508596549346016075"),
    (
        9,
        "575145928914329651470194933472811683205657195623667788065",
    ),
    (
        10,
        "1519677001119572800512040811336385024206490894973776814011228270",
    ),
    (
        11,
        "4015360400952738719295212964529841259489899066405054233036023907781025",
    ),
    (
        12,
        "10609569755718585546322957854951611814699756535322950665546543328899232582195",
    ),
    (
        13,
        "28033092714355183194079926891486535839668730558509897666505294262776939041316902526",
    ),
    (
        14,
        "74070325680082941984595004188615297225172450152470200808745432364750868148681948483249795",
    ),
    (
        15,
        "195712018015909913147811337370285136548490317907436560134987064184636438003597987408122340\
         476255",
    ),
    (
        16,
        "517119286896281883839171646707138022918727361860905299163902348006264865398238555541414133\
         036035316575",
    ),
    (
        17,
        "136635634127680642348667990132095304049314199420140905096220928841598027961033089589453144\
         0460991147162981436",
    ),
];

/// The primes of [`PRIMES`] for `L` limbs, in the byte order `order`.
fn primes(limbs: usize, order: ByteOrder) -> [Number; 3] {
    let (_, k1, k2, k3) = PRIMES[limbs - 2];
    let bits = 64 * limbs;
    let k = |k: u64| Number::new_from_u64(k, order);
    [
        &power_of_two(bits, order) - &k(k1),
        &power_of_two(bits - 1, order) + &k(k2),
        &power_of_two(bits - 2, order) - &k(k3),
    ]
}

/// `n` in the byte order `order`, its magnitude stored with `extra` zero bytes above its most
/// significant one, as a number need not be stored in its shortest form.
fn stored(n: &Number, order: ByteOrder, extra: usize) -> Number {
    let mut le = le_bytes(n);
    le.extend(core::iter::repeat_n(0, extra));
    number(&le, false, order)
}

#[test]
fn primes_and_composites_at_every_kernel_length() {
    let mut state = 0x243F6A88_85A308D3;
    // Under Miri, which runs the portable rows: two limbs, eight (whose test takes the registers'
    // path in the limbs' function) and nine, in one byte order, the first prime of each.
    let lengths: Vec<usize> = if cfg!(miri) {
        vec![2, 8, 9]
    } else {
        (2..=18).collect()
    };
    for &limbs in &lengths {
        for order in ORDERS {
            if cfg!(miri) && order == ByteOrder::BigEndian {
                continue;
            }
            let ps = primes(limbs, order);
            for p in ps.iter().take(size(3, 1)) {
                let x = stored(p, order, below(&mut state, 3));
                assert!(x.is_probable_prime(), "{p} ({limbs} limbs) is prime");
                // The odd numbers next to it: composite, and caught by the strong test or before.
                let two = Number::new_from_u8(2, order);
                for q in [p - &two, p + &two] {
                    let r = (&q % &Number::new_from_u8(3, order)).is_zero()
                        || (&q % &Number::new_from_u8(5, order)).is_zero();
                    if r {
                        assert!(!q.is_probable_prime(), "{q} has a small factor");
                    }
                }
            }
            // Products of two primes of the table, of `limbs` limbs in all.
            for split in 1..limbs {
                if split < 2 || limbs - split < 2 || cfg!(miri) && (split != 2 || limbs > 8) {
                    continue;
                }
                let n = &primes(split, order)[below(&mut state, 3)]
                    * &primes(limbs - split, order)[below(&mut state, 3)];
                assert!(!n.is_probable_prime(), "{n} ({limbs} limbs) is composite");
            }
            if let Some((_, k)) = CARMICHAEL.iter().find(|(l, _)| *l == limbs) {
                let k: Number = k.parse().unwrap();
                let f =
                    |c: u8| &(&k * &Number::new_from_u8(c, order)) + &Number::new_from_u8(1, order);
                let n = &(&f(6) * &f(12)) * &f(18);
                let n = stored(&n, order, below(&mut state, 2));
                assert!(!n.is_probable_prime(), "Carmichael number {n} is composite");
            }
        }
    }
}

#[test]
fn next_and_previous_primes_at_every_kernel_length() {
    if cfg!(miri) {
        return;
    }
    // The prime after `2^(64L - 1)` and the one before `2^(64L)`: the searches test numbers of
    // the kernels' lengths, composites by the strong test and the prime by both.
    for limbs in 2..=17 {
        for order in ORDERS {
            let ps = primes(limbs, order);
            let bits = 64 * limbs;
            assert_eq!(
                power_of_two(bits - 1, order).next_prime(),
                ps[1],
                "{limbs} limbs"
            );
            assert_eq!(
                power_of_two(bits, order).prev_prime().unwrap(),
                ps[0],
                "{limbs} limbs"
            );
        }
    }
}

/// A proper factor `f` of `n`, in its byte order.
fn proper(n: &Number, f: &Number) {
    assert!(
        f.byte_order() == n.byte_order(),
        "the factor in the byte order of n"
    );
    assert!(*f > 1u8 && *f < n.abs(), "{f} is a proper factor of {n}");
    assert!((n % f).is_zero(), "{f} divides {n}");
}

#[test]
fn factoring_methods_on_residues_in_memory() {
    // A prime of 31 bits with `p - 1 = 2^2 3 41 97 149 151`, times a prime that makes the number
    // `limbs` limbs long, its top bit set or its top limb below `2^62`: the factoring methods' long
    // residues, by the kernels from eight limbs to sixteen and the multiplication's from seventeen.
    if cfg!(miri) {
        return;
    }
    let p: Number = "1073742277".parse().unwrap();
    for limbs in [8, 9, 12, 13, 16, 17] {
        for high in [true, false] {
            let order = if high {
                ByteOrder::BigEndian
            } else {
                ByteOrder::LittleEndian
            };
            let bits = 64 * limbs - if high { 1 } else { 4 };
            let q = power_of_two(bits - p.bits() as usize, order).next_prime();
            let n = &stored(&p, order, 0) * &q;
            assert_eq!(n.bits() as usize, bits);
            let f = factor::pm1(&n, 160, 20_000).unwrap_or_else(|| panic!("pm1, {limbs} limbs"));
            proper(&n, &f);
            let f =
                factor::ecm(&n, 2000, 200_000, 40).unwrap_or_else(|| panic!("ecm, {limbs} limbs"));
            proper(&n, &f);
            let f = factor::pp1(&n, 2000, 200_000);
            if let Some(f) = f {
                proper(&n, &f);
            }
        }
    }
}

//! Tests of the discrete logarithms modulo primes whose `p - 1` has a large prime factor, which
//! `discrete_log` takes by the index calculus: safe primes from 38 to 64 bits, other cofactors,
//! a square of a prime in `p - 1`, elements outside the subgroup, prime powers and products beyond
//! a word, and both byte orders.

use a::math::numbers::{ByteOrder, Factorization, Number, dlog};

fn n(x: u64) -> Number {
    Number::new_from_u64(x, ByteOrder::BigEndian)
}

fn number(s: &str) -> Number {
    Number::from_str_radix(s, 10, ByteOrder::BigEndian).unwrap()
}

/// The same number in the other byte order, with zero bytes above its top.
fn other(x: &Number) -> Number {
    let mut v = x.value.clone();
    v.reverse();
    let order = match x.byte_order {
        ByteOrder::BigEndian => ByteOrder::LittleEndian,
        ByteOrder::LittleEndian => ByteOrder::BigEndian,
    };
    match order {
        ByteOrder::BigEndian => v.insert(0, 0),
        ByteOrder::LittleEndian => v.push(0),
    }
    Number {
        value: v,
        negative: x.negative,
        byte_order: order,
    }
}

/// Safe primes `p = 2q + 1` (PARI, `setrand(20261108)`), by their bits, with their smallest
/// primitive roots.
const SAFE: [(u32, u64, u64); 9] = [
    (38, 138_298_270_883, 2),
    (40, 639_511_635_563, 2),
    (44, 9_532_996_616_867, 2),
    (48, 241_934_786_347_547, 2),
    (52, 2_885_214_618_914_879, 11),
    (56, 52_103_198_450_488_919, 13),
    (60, 914_014_194_703_562_339, 2),
    (63, 7_339_933_933_999_198_943, 5),
    (64, 18_042_075_562_717_561_919, 7),
];

/// Exponents spread over `[0, m)`: the ends and a few between.
fn exponents(m: u64) -> Vec<u64> {
    vec![
        1,
        2,
        m / 3 + 7,
        ((u128::from(m) * 0x9E37_79B9) >> 32) as u64 % m,
        m - 1,
    ]
}

#[test]
fn logarithms_modulo_safe_primes() {
    // Under Miri the smallest prime only, for which the sieve is shortest.
    let primes: &[(u32, u64, u64)] = if cfg!(miri) { &SAFE[..1] } else { &SAFE };
    for &(bits, p, g) in primes {
        let pn = n(p);
        let gn = n(g);
        let ks = if cfg!(miri) {
            vec![p / 3 + 7]
        } else {
            exponents(p - 1)
        };
        for k in ks {
            let h = gn.pow_mod(&n(k), &pn);
            assert_eq!(h.discrete_log(&gn, &pn), Some(n(k)), "{bits} bits: {g}^{k}");
        }
        if cfg!(miri) {
            continue;
        }
        // The squares, of the prime order q: 4 generates them.
        let q = (p - 1) / 2;
        let four = n(4);
        for k in [1, q / 2 + 3, q - 1] {
            let h = four.pow_mod(&n(k), &pn);
            assert_eq!(h.discrete_log(&four, &pn), Some(n(k)), "{bits} bits: 4^{k}");
        }
        // p = 3 (mod 4): -1 is not a square, so not a power of 4; and 1 is 4^0.
        assert_eq!(n(p - 1).discrete_log(&four, &pn), None, "{bits} bits");
        assert_eq!(n(g).discrete_log(&four, &pn), None, "{bits} bits");
        assert_eq!(n(1).discrete_log(&four, &pn), Some(n(0)));
        // -1 to a primitive root is (p - 1) / 2.
        assert_eq!(n(p - 1).discrete_log(&gn, &pn), Some(n(q)), "{bits} bits");
    }
}

#[test]
fn logarithms_with_other_cofactors() {
    if cfg!(miri) {
        return;
    }
    // p - 1 = 48 q: the large q by the index calculus, 2^4 and 3 by the other methods.
    for (p, q, g) in [
        (56_068_601_217_457u64, 1_168_095_858_697u64, 5u64),
        (722_544_042_201_948_817, 15_053_000_879_207_267, 5),
    ] {
        assert_eq!(p - 1, 48 * q);
        let (pn, gn) = (n(p), n(g));
        for k in exponents(p - 1) {
            let h = gn.pow_mod(&n(k), &pn);
            assert_eq!(h.discrete_log(&gn, &pn), Some(n(k)), "{g}^{k} mod {p}");
        }
        // A base of order 16 q: its logarithms are below that.
        let b = gn.pow_mod(&n(3), &pn);
        let k = 16 * q - 5;
        let h = b.pow_mod(&n(k), &pn);
        assert_eq!(h.discrete_log(&b, &pn), Some(n(k)));
        assert_eq!(gn.discrete_log(&b, &pn), None);
    }
}

#[test]
fn a_square_of_a_prime_in_p_minus_1() {
    if cfg!(miri) {
        return;
    }
    // p - 1 = 6 q^2, q of 29 bits: the q-part, of order q^2, is found a digit at a time, which
    // the logarithms modulo q of the index calculus would not tell apart.
    let (p, q, g) = (786_683_777_637_705_847u64, 362_096_621u64, 3u64);
    assert_eq!(p - 1, 6 * q * q);
    let (pn, gn) = (n(p), n(g));
    for k in exponents(p - 1) {
        let h = gn.pow_mod(&n(k), &pn);
        assert_eq!(h.discrete_log(&gn, &pn), Some(n(k)), "{g}^{k}");
    }
}

#[test]
fn the_largest_prime_of_a_word() {
    if cfg!(miri) {
        return;
    }
    // 2^64 - 59: p - 1 = 2^2 11 137 547 5594472617641.
    let p = number("18446744073709551557");
    let g = n(2);
    for k in [
        number("1"),
        number("12345678901234567"),
        number("18446744073709551555"),
        number("9223372036854775807"),
    ] {
        let h = g.pow_mod(&k, &p);
        assert_eq!(h.discrete_log(&g, &p), Some(k.clone()), "2^{k}");
    }
}

#[test]
fn results_in_the_byte_order_of_self() {
    if cfg!(miri) {
        return;
    }
    let (_, p, g) = SAFE[2];
    let (pn, gn) = (n(p), n(g));
    let k = n(p / 5 + 11);
    let h = gn.pow_mod(&k, &pn);
    for (h, g, m) in [
        (h.clone(), gn.clone(), pn.clone()),
        (other(&h), gn.clone(), pn.clone()),
        (other(&h), other(&gn), other(&pn)),
        (h.clone(), other(&gn), other(&pn)),
    ] {
        let got = h.discrete_log(&g, &m).unwrap();
        assert_eq!(got, k);
        assert_eq!(got.byte_order, h.byte_order);
    }
    // Negative and long residues are reduced first.
    let minus = -&(&pn - &h);
    assert_eq!(minus.discrete_log(&gn, &pn), Some(k.clone()));
    let long = &h + &(&pn * &n(1 << 40));
    assert_eq!(long.discrete_log(&gn, &pn), Some(k));
}

#[test]
fn prime_powers_and_products_beyond_a_word() {
    if cfg!(miri) {
        return;
    }
    // p^2 for the safe p of 40 bits: the part of p - 1 modulo p, by the index calculus, and the
    // part of p modulo p^2.
    let (_, p, g) = SAFE[1];
    let (pn, gn) = (n(p), n(g));
    let m = &pn * &pn;
    let order = gn.multiplicative_order(&m).unwrap();
    for k in [n(5), &order - 1u8, (&order >> 1usize) + 12345u32] {
        let h = gn.pow_mod(&k, &m);
        assert_eq!(h.discrete_log(&gn, &m), Some(k.clone()), "{g}^{k} mod p^2");
    }
    // The product of the safe primes of 44 and 48 bits: their logarithms joined.
    let (_, p1, _) = SAFE[2];
    let (_, p2, _) = SAFE[3];
    let m = &n(p1) * &n(p2);
    let g = n(3);
    let order = g.multiplicative_order(&m).unwrap();
    for k in [n(7), &order - 1u8, (&order >> 2usize) + 99u8] {
        let h = g.pow_mod(&k, &m);
        let got = h.discrete_log(&g, &m).unwrap();
        assert_eq!(got, k.rem_euclid(&order), "3^{k} mod p1 p2");
    }
    // 2^10 p, a word, for the safe p of 52 bits.
    let (_, p, g) = SAFE[4];
    let m = n(p << 10);
    let gn = n(g);
    let order = gn.multiplicative_order(&m).unwrap();
    for k in [n(3), &order - 1u8] {
        let h = gn.pow_mod(&k, &m);
        assert_eq!(
            h.discrete_log(&gn, &m),
            Some(k.clone()),
            "{g}^{k} mod 2^10 p"
        );
    }
}

/// The smallest prime `2 M r + 1` of at least `bits` bits, `M` the product of the odd primes from 3
/// on whose product first reaches `bits - 8` bits, `r = 1, 2, ...`: its `p - 1` is smooth, so that
/// its logarithms take their time in the products of the ring alone.
fn smooth_prime(bits: u64) -> Number {
    let mut m = n(1);
    for q in a::math::numbers::primes(3..) {
        if m.bits() >= bits - 8 {
            break;
        }
        m = &m * &n(q);
    }
    let two_m = &m << 1usize;
    (1u32..)
        .map(|r| &(&two_m * &n(u64::from(r))) + 1u8)
        .find(|p| p.bits() >= bits && p.is_probable_prime())
        .expect("a prime")
}

#[test]
fn logarithms_modulo_primes_and_prime_powers_of_eight_limbs() {
    // `discrete_log` takes the residues of eight odd limbs in registers: modulo a prime of eight
    // limbs, whose `p - 1` part is in the ring modulo `p`, and modulo `q^2` for a prime `q` of four
    // limbs, whose `q` part is in the ring modulo `q^2` of eight limbs. The answers are checked by
    // the power and against the order of `g`, from `multiplicative_order`. Not under Miri, where
    // one logarithm modulo each took an hour: the ring of eight limbs is safe code there, the rows
    // of Rust, which `rings_of_every_length` in `modular.rs` takes under Miri.
    if cfg!(miri) {
        return;
    }
    let p = smooth_prime(456);
    let q = smooth_prime(230);
    assert!(p.bits() > 448 && p.bits() <= 512, "p of eight limbs: {p}");
    let q2 = &q * &q;
    assert!(
        q2.bits() > 448 && q2.bits() <= 512,
        "q^2 of eight limbs: {q2}"
    );
    for m in [p, q2] {
        let cases = [
            (n(3), &m >> 7usize),
            (n(5), &m - 12345u32),
            (n(1_000_003), n(987_654_321)),
        ];
        for (g, k) in cases {
            let order = g.multiplicative_order(&m).expect("g prime to m");
            let h = g.pow_mod(&k, &m);
            let both = [(h.clone(), g.clone()), (other(&h), other(&g))];
            for (h, g) in both {
                let r = h.discrete_log(&g, &m).expect("a logarithm");
                assert_eq!(r.byte_order, h.byte_order);
                assert!(r < order, "{g}^{k} mod {m}: {r} not below the order");
                assert_eq!(r, k.rem_euclid(&order), "{g}^{k} mod {m}");
                assert_eq!(g.pow_mod(&r, &m), h, "{g}^{r} mod {m}");
            }
        }
    }
}

#[test]
fn the_methods_keep_their_ways() {
    if cfg!(miri) {
        return;
    }
    // `pohlig_hellman` and `rho` modulo a safe prime of 38 bits, whose order q of 37 bits they
    // take by rho, give what `discrete_log` gives by the index calculus.
    let (_, p, _) = SAFE[0];
    let pn = n(p);
    let q = n((p - 1) / 2);
    let four = n(4);
    let k = n((p - 1) / 2 - 12345);
    let h = four.pow_mod(&k, &pn);
    let f = Factorization::new(vec![(q.clone(), 1)], ByteOrder::BigEndian);
    assert_eq!(dlog::pohlig_hellman(&h, &four, &pn, &f), Some(k.clone()));
    assert_eq!(dlog::rho(&h, &four, &pn, &q), Some(k.clone()));
    assert_eq!(h.discrete_log(&four, &pn), Some(k));
}

#[test]
fn logarithms_beyond_a_word() {
    if cfg!(miri) {
        return;
    }
    // Safe primes of 66, 70 and 76 bits (PARI, `setrand(20261109)`), and one of 81 bits with
    // `p - 1 = 6 q`: the order's large prime by the index calculus on two limbs, its arithmetic
    // modulo `q` in two limbs too.
    for (p, g) in [
        ("51165845971085855519", 11u64),
        ("1021983241328439553187", 2),
        ("67642180843965767476247", 5),
        ("1646327210944745169289507", 2),
    ] {
        let p = number(p);
        let g = n(g);
        let p1 = &p - 1u8;
        for k in [n(3), &p1 - 1u8, (&p1 >> 1usize) + 12345u32] {
            let h = g.pow_mod(&k, &p);
            assert_eq!(h.discrete_log(&g, &p), Some(k.clone()), "{g}^{k} mod {p}");
        }
        // -1 has the logarithm (p - 1) / 2, and a non-residue none to a square.
        assert_eq!((&p - 1u8).discrete_log(&g, &p), Some(&p1 >> 1usize));
        let square = g.pow_mod(&n(2), &p);
        assert_eq!(g.discrete_log(&square, &p), None, "{g} to {square} mod {p}");
    }
    // The square of the prime of 66 bits.
    let p = number("51165845971085855519");
    let m = &p * &p;
    let g = n(11);
    let order = g.multiplicative_order(&m).unwrap();
    let k = (&order >> 3usize) + 777u32;
    let h = g.pow_mod(&k, &m);
    assert_eq!(h.discrete_log(&g, &m), Some(k), "11^k mod p^2");
}

#[test]
fn two_large_primes_in_p_minus_1() {
    if cfg!(miri) {
        return;
    }
    // p - 1 = 2 q1 q2 with q1, q2 of 44 bits, 89 bits in all: each large prime below the index
    // calculus's threshold there (2^47, 2^46 without the kernels), so by rho, the logarithms
    // joined. (Two primes of `p - 1` both at the threshold take 102 bits or more: see
    // dlog_tune.rs.)
    let p = number("553236598461103193145347999");
    let (q1, q2) = (number("16926290848529"), number("16342523102431"));
    assert_eq!(&(&q1 * &q2) * 2u8 + 1u8, p);
    let g = n(17);
    for k in [number("123456789012345678901234567"), &p - 2u8, &q1 * &q2] {
        let h = g.pow_mod(&k, &p);
        assert_eq!(
            h.discrete_log(&g, &p),
            Some(k.rem_euclid(&(&p - 1u8))),
            "17^{k}"
        );
    }
    // An element of order q1 to a base of order q1 q2.
    let b = g.pow_mod(&n(2), &p);
    let h = b.pow_mod(&(&q2 * 5u8), &p);
    assert_eq!(h.discrete_log(&b, &p), Some(&q2 * 5u8));
}

#[test]
fn even_moduli_with_many_twos() {
    // Modulo `2^k p` the elements of odd order are 1 modulo `2^k`: their low `k` bits are fixed,
    // and from `k = 64` their low limbs are all 1. The walks of rho chose their steps and their
    // distinguished points by bits that a product of the residue's low limb leaves fixed, and
    // never met a distinguished point from `k = 33`; the table of baby steps was one chain from
    // `k = 64`. `p = 2q + 1`, `g = 1 (mod 2^k)` and `4 (mod p)`, of order `q`; the logarithms from
    // PARI.
    let cases: &[(&str, &str, &str, &str, u64)] = &[
        // 2^33 p, 2^40 p, 2^50 p, 2^63 p and 2^104 p, q of 20 bits.
        (
            "18229035205132288",
            "1061069",
            "14236519506116609",
            "5759181776748545",
            353_696,
        ),
        (
            "2333316506256932864",
            "1061069",
            "615794681275482113",
            "96904357802409985",
            353_696,
        ),
        (
            "2389316102407099252736",
            "1061069",
            "1134607616722144854017",
            "2272747181452055019521",
            353_696,
        ),
        (
            "19573277510918957078413312",
            "1061069",
            "12382570650290310472531969",
            "1935229474028795350482945",
            353_696,
        ),
        (
            "43042092433883752221804995896854708224",
            "1061069",
            "28448675617550735183400285176802050049",
            "27896040803080038119358994421011972097",
            353_696,
        ),
        // 2^64 p for q of 17, 20 and 24 bits.
        (
            "5294270889386862442446848",
            "143501",
            "740230946189816887246849",
            "4670162197141047182622721",
            47_840,
        ),
        (
            "39146555021837914156826624",
            "1061069",
            "31955848161209267550945281",
            "1935229474028795350482945",
            353_696,
        ),
        (
            "619426521218282227723403264",
            "16789589",
            "130514901732399809751416833",
            "142807848876608002367422465",
            5_596_536,
        ),
        (
            // 2^130 p.
            "45705610037527486050486463548024300235963498496",
            "16789589",
            "27163374777312963049411925998854610339395272705",
            "1666636337834840090104770005777744073142042625",
            16_789_584,
        ),
    ];
    // Under Miri one of two limbs only, whose key mixes both.
    let cases = if cfg!(miri) { &cases[4..5] } else { cases };
    for &(m, q, g, h, k) in cases {
        let (m, q, g, h, k) = (number(m), number(q), number(g), number(h), n(k));
        assert_eq!(g.pow_mod(&k, &m), h);
        let f = Factorization::new(vec![(q.clone(), 1)], ByteOrder::BigEndian);
        assert_eq!(dlog::rho(&h, &g, &m, &q), Some(k.clone()), "rho mod {m}");
        assert_eq!(
            dlog::pohlig_hellman(&h, &g, &m, &f),
            Some(k.clone()),
            "pohlig_hellman mod {m}"
        );
        assert_eq!(dlog::bsgs(&h, &g, &m, &q), Some(k.clone()), "bsgs mod {m}");
        assert_eq!(h.discrete_log(&g, &m), Some(k), "discrete_log mod {m}");
    }
    if cfg!(miri) {
        return;
    }
    // A power of two past 2^128, by divisions: 1 + 2^150 has the order 2^50 modulo 2^200.
    let m = n(2).pow(200);
    let g = &n(2).pow(150) + 1u8;
    let k = n(777_777);
    let h = g.pow_mod(&k, &m);
    assert_eq!(dlog::bsgs(&h, &g, &m, &n(1 << 20)), Some(k.clone()));
    let f = Factorization::new(vec![(n(2), 50)], ByteOrder::BigEndian);
    assert_eq!(dlog::pohlig_hellman(&h, &g, &m, &f), Some(k.clone()));
    assert_eq!(h.discrete_log(&g, &m), Some(k));
}

#[test]
fn pohlig_hellman_of_an_element_outside_the_group() {
    // p = c q + 1 of 100 bits with q of 24 bits, g of order q: 3 generates the units, so it is not
    // a power of g, and with the order of g as given only one power tells it. The walks of rho
    // took the root of the larger group's order, 2^50 steps, before they gave it up.
    let p = number("633863494614228150438974201753");
    let q = n(16_778_227);
    let g = number("124325332993728095009513964021");
    let f = Factorization::new(vec![(q.clone(), 1)], ByteOrder::BigEndian);
    assert_eq!(dlog::pohlig_hellman(&n(3), &g, &p, &f), None);
    let k = n(16_778_150);
    let h = g.pow_mod(&k, &p);
    assert_eq!(h, number("76559417104932432622043031501"));
    assert_eq!(dlog::pohlig_hellman(&h, &g, &p, &f), Some(k));
    // The same with the order's square given, the second digit's element outside too.
    let f2 = Factorization::new(vec![(q.clone(), 2)], ByteOrder::BigEndian);
    assert_eq!(dlog::pohlig_hellman(&n(3), &g, &p, &f2), None);
}

#[test]
fn a_prime_order_that_divides_the_multipliers_constant() {
    if cfg!(miri) {
        return;
    }
    // p - 1 = 6 q, q = 82431689521877 of 47 bits, a factor of the constant whose power of a small
    // prime multiplies the elements in the descent: that power must not be q's multiple, or the
    // descent tries the same element 4 million times before rho takes over.
    let p = n(494_590_137_131_263);
    let g = n(3);
    for (h, k) in [
        (219_386_457_209_584u64, 131_392_633_158_484u64),
        (381_943_536_044_494, 327_960_940_043_340),
        (391_761_269_855_501, 196_449_803_206_016),
        (351_263_205_629_561, 406_355_692_633_888),
    ] {
        assert_eq!(g.pow_mod(&n(k), &p), n(h));
        assert_eq!(n(h).discrete_log(&g, &p), Some(n(k)), "3^{k} mod {p}");
    }
}

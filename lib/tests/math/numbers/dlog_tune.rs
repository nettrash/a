//! Tests of the discrete logarithms at the index calculus's thresholds as measured with the kernels
//! of the products in registers: the smallest orders it takes now (safe primes of 35 bits, orders
//! from `2^33` at 40 bits, `2^39` just beyond a word), and two primes of `p - 1` that both reach it
//! and share its sieve. The primes are PARI's (`setrand(20261108)`), each `g` a primitive root.

use a::math::numbers::{ByteOrder, Number};

fn n(x: u64) -> Number {
    Number::new_from_u64(x, ByteOrder::BigEndian)
}

fn number(s: &str) -> Number {
    Number::from_str_radix(s, 10, ByteOrder::BigEndian).unwrap()
}

#[test]
fn orders_just_above_the_thresholds() {
    if cfg!(miri) {
        return;
    }
    // (p, g, the large prime q of p - 1), q just above the smallest order taken by the index
    // calculus modulo p, with the kernels and without them.
    for (p, g, q) in [
        // A safe prime of 35 bits, q = 2^33.997.
        ("34299774287", 5, "17149887143"),
        // 40 bits, p - 1 = 72 q, q = 2^33.007.
        ("621269868169", 7, "8628748169"),
        // 64 bits, q = 2^40.007.
        ("13744764024476874871", 3, "1105107543763"),
        // 66 bits, q = 2^39.001: the ring of two limbs, the algebra modulo q in one.
        ("61753507881882960953", 3, "550192133837"),
        // 72 bits, q = 2^40.010.
        ("3521543506495949521427", 2, "1107245553173"),
    ] {
        let (p, g, q) = (number(p), n(g), number(q));
        let p1 = &p - 1u8;
        assert!(p1.rem_euclid(&q).is_zero(), "{q} divides {p} - 1");
        for k in [n(5), &p1 - 1u8, &q + 1u8, (&p1 >> 1usize) + 777u32] {
            let h = g.pow_mod(&k, &p);
            assert_eq!(h.discrete_log(&g, &p), Some(k.clone()), "{g}^{k} mod {p}");
        }
        // To a base of order q, the logarithm is below q.
        let b = g.pow_mod(&(&p1 / &q), &p);
        let k = &q - 3u8;
        assert_eq!(
            b.pow_mod(&k, &p).discrete_log(&b, &p),
            Some(k.clone()),
            "{b}^{k} mod {p}"
        );
        // An element outside the group of b has no logarithm to it.
        assert_eq!(g.discrete_log(&b, &p), None, "{g} to {b} mod {p}");
    }
}

#[test]
fn two_primes_of_p_minus_1_by_the_index_calculus() {
    if cfg!(miri) {
        return;
    }
    // p - 1 = 2 q1 q2 with q1 = 2^50.07 and q2 = 2^50.19, p of 102 bits: both at least the
    // threshold there, 2^50 (2^49 without the kernels), so the second takes the relations of the
    // first one's sieve.
    let p = number("3034971620858154132172278632963");
    let (q1, q2) = (number("1179540863107093"), number("1286505502176317"));
    assert_eq!(&(&q1 * &q2) * 2u8 + 1u8, p);
    let g = n(2);
    let p1 = &p - 1u8;
    for k in [number("123456789012345678901234567"), &q1 * 3u8] {
        let h = g.pow_mod(&k, &p);
        assert_eq!(h.discrete_log(&g, &p), Some(k.rem_euclid(&p1)), "2^{k}");
    }
}

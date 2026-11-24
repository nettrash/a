//! Hashing to curves, RFC 9380: every vector of appendix J for the suites of P-256, P-384,
//! P-521, secp256k1, curve25519 and edwards25519, the parameters of section 8, the searches for
//! `Z` of appendix H, the maps on small fields in full, their exceptional cases, the isogeny of
//! secp256k1 and the rational map of edwards25519, and the panics.

use super::common::{Rng, hex, n, size};
use a::hash::{Sha256, Sha384, Sha512, expand_message_xmd};
use a::math::ec::hash::{
    self, Elligator2, IsogenyMap, ShallueVanDeWoestijne, SimplifiedSwu, Suite, hash_to_field, sgn0,
};
use a::math::ec::{Curve, Edwards, Montgomery, Point, Weierstrass, named};
use a::math::field::{Fp, PrimeField};
use a::math::numbers::{ByteOrder, Number};
use std::collections::HashMap;

/// Appendix J, from `data/h2c_vectors.txt` (see `data/gen_h2c_vectors.py`).
const VECTORS: &str = include_str!("data/h2c_vectors.txt");

/// One vector of appendix J: its suite, tag, message and named values.
struct Vector {
    suite: String,
    dst: String,
    msg: String,
    values: HashMap<String, Number>,
}

/// The vectors of appendix J.
fn vectors() -> Vec<Vector> {
    let (mut suite, mut dst) = (String::new(), String::new());
    let mut out: Vec<Vector> = Vec::new();
    for line in VECTORS.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        match k {
            "suite" => suite = v.to_string(),
            "dst" => dst = v.to_string(),
            "msg" => out.push(Vector {
                suite: suite.clone(),
                dst: dst.clone(),
                msg: v.to_string(),
                values: HashMap::new(),
            }),
            _ => {
                out.last_mut().unwrap().values.insert(k.to_string(), hex(v));
            }
        }
    }
    out
}

/// The affine coordinates of a point, which must not be at infinity.
fn xy<C: Curve>(p: &Point<'_, C>) -> (Number, Number) {
    p.coordinates().expect("an affine point")
}

/// Checks the vectors of the two suites of `suite` and gives how many there were.
fn check_suite<C: Curve>(suite: &'static Suite<C>, all: &[Vector]) -> usize {
    let mut count = 0;
    for v in all
        .iter()
        .filter(|v| v.suite == suite.ro_id() || v.suite == suite.nu_id())
    {
        let at = |k: &str| v.values[k].clone();
        let (msg, dst) = (v.msg.as_bytes(), v.dst.as_bytes());
        let what = format!("{} {:.12}", v.suite, v.msg);
        let p = if v.suite == suite.ro_id() {
            let u = suite.hash_to_field(msg, dst, 2);
            assert_eq!((u[0].value(), u[1].value()), (at("u0"), at("u1")), "{what}");
            let q0 = suite.map_to_curve(&u[0]);
            let q1 = suite.map_to_curve(&u[1]);
            assert_eq!(xy(&q0), (at("Q0.x"), at("Q0.y")), "{what}");
            assert_eq!(xy(&q1), (at("Q1.x"), at("Q1.y")), "{what}");
            let p = suite.hash_to_curve(msg, dst);
            assert_eq!(p, suite.clear_cofactor(&(q0 + q1)), "{what}");
            p
        } else {
            let u = suite.hash_to_field(msg, dst, 1);
            assert_eq!(u[0].value(), at("u0"), "{what}");
            let q = suite.map_to_curve(&u[0]);
            assert_eq!(xy(&q), (at("Q.x"), at("Q.y")), "{what}");
            let p = suite.encode_to_curve(msg, dst);
            assert_eq!(p, suite.clear_cofactor(&q), "{what}");
            p
        };
        assert_eq!(xy(&p), (at("P.x"), at("P.y")), "{what}");
        // The point is on the curve (as every point is) and in the prime-order subgroup.
        let (x, y) = xy(&p);
        assert!(suite.domain().point(&x, &y).is_ok(), "{what}");
        assert!(suite.domain().in_subgroup(&p), "{what}");
        count += 1;
    }
    count
}

#[test]
fn every_appendix_j_vector_of_the_twelve_suites_is_reproduced() {
    let all = vectors();
    assert_eq!(all.len(), 60);
    let counts = [
        check_suite(hash::p256(), &all),
        check_suite(hash::p384(), &all),
        check_suite(hash::p521(), &all),
        check_suite(hash::secp256k1(), &all),
        check_suite(hash::curve25519(), &all),
        check_suite(hash::edwards25519(), &all),
    ];
    assert_eq!(counts, [10; 6]);
}

#[test]
fn the_suites_have_the_parameters_of_section_8() {
    let one = Number::from(1u8);
    let eight = Number::from(8u8);
    let w: [(&Suite<Weierstrass>, &str, u32, usize, i64); 4] = [
        (hash::p256(), "P256_XMD:SHA-256_SSWU_", 128, 48, -10),
        (hash::p384(), "P384_XMD:SHA-384_SSWU_", 192, 72, -12),
        (hash::p521(), "P521_XMD:SHA-512_SSWU_", 256, 98, -4),
        (
            hash::secp256k1(),
            "secp256k1_XMD:SHA-256_SSWU_",
            128,
            48,
            -11,
        ),
    ];
    for (s, id, k, l, z) in w {
        assert_eq!(s.ro_id(), format!("{id}RO_"));
        assert_eq!(s.nu_id(), format!("{id}NU_"));
        assert_eq!((s.k(), s.l(), s.h_eff()), (k, l, &one), "{id}");
        assert_eq!(s.z(), &s.field().element(&n(z)), "{id}");
        assert_eq!(s.field().modulus(), s.domain().modulus());
    }
    assert!(core::ptr::eq(hash::p256().domain(), named::p256()));
    assert!(core::ptr::eq(
        hash::secp256k1().domain(),
        named::secp256k1()
    ));
    assert!(hash::p256().isogeny().is_none());
    assert_eq!(hash::secp256k1().isogeny().unwrap().degree(), 3);

    let c = hash::curve25519();
    assert_eq!(
        (c.ro_id(), c.nu_id()),
        (
            "curve25519_XMD:SHA-512_ELL2_RO_",
            "curve25519_XMD:SHA-512_ELL2_NU_"
        )
    );
    assert_eq!((c.k(), c.l(), c.h_eff()), (128, 48, &eight));
    assert_eq!(c.z(), &c.field().from_u64(2));
    assert!(core::ptr::eq(c.domain(), named::curve25519()));
    let e = hash::edwards25519();
    assert_eq!(
        (e.ro_id(), e.nu_id()),
        (
            "edwards25519_XMD:SHA-512_ELL2_RO_",
            "edwards25519_XMD:SHA-512_ELL2_NU_"
        )
    );
    assert_eq!((e.k(), e.l(), e.h_eff()), (128, 48, &eight));
    assert_eq!(e.z(), &e.field().from_u64(2));
    assert!(core::ptr::eq(e.domain(), named::ed25519()));
    assert!(e.isogeny().is_none());
}

/// secp256k1's isogenous curve `E'` of section 8.7.
fn e_prime() -> Weierstrass {
    Weierstrass::new(
        &hex("3f8731abdd661adca08a5558f0f5d272e953d363cb6f0e5d405447c01a444533"),
        &n(1771),
        named::secp256k1().modulus(),
    )
    .unwrap()
}

#[test]
fn the_searches_of_appendix_h_give_the_suites_z() {
    for (curve, z) in [
        (named::p256(), -10),
        (named::p384(), -12),
        (named::p521(), -4),
    ] {
        let f = PrimeField::new(curve.modulus());
        let found = SimplifiedSwu::find_z(&f.element(curve.a()), &f.element(curve.b()));
        assert_eq!(found, f.element(&n(z)));
    }
    let e = e_prime();
    let f = PrimeField::new(e.modulus());
    assert_eq!(
        SimplifiedSwu::find_z(&f.element(e.a()), &f.element(e.b())),
        f.element(&n(-11))
    );
    let f = PrimeField::new(named::curve25519().modulus());
    assert_eq!(Elligator2::find_z(&f), f.from_u64(2));
}

#[test]
fn hash_to_field_reduces_pieces_of_l_bytes_of_expand_message_xmd() {
    let suite = hash::p256();
    let p = suite.field().modulus();
    let dst = b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_RO_";
    for count in 0..5 {
        let bytes = expand_message_xmd::<Sha256>(b"some message", dst, 48 * count);
        let want: Vec<Number> = bytes
            .chunks(48)
            .map(|c| Number::new(c.to_vec(), ByteOrder::BigEndian).rem_euclid(p))
            .collect();
        let got = suite.hash_to_field(b"some message", dst, count);
        assert_eq!(got.iter().map(Fp::value).collect::<Vec<_>>(), want);
        assert_eq!(
            hash_to_field::<Sha256>(suite.field(), b"some message", dst, count, 128),
            got
        );
    }
    // The suites' own hashes: SHA-384 for P-384, SHA-512 for P-521.
    let s = hash::p384();
    let bytes = expand_message_xmd::<Sha384>(b"m", b"D", 72);
    assert_eq!(
        s.hash_to_field(b"m", b"D", 1)[0].value(),
        Number::new(bytes, ByteOrder::BigEndian).rem_euclid(s.field().modulus())
    );
    let s = hash::p521();
    let bytes = expand_message_xmd::<Sha512>(b"m", b"D", 98);
    assert_eq!(
        s.hash_to_field(b"m", b"D", 1)[0].value(),
        Number::new(bytes, ByteOrder::BigEndian).rem_euclid(s.field().modulus())
    );
}

#[test]
fn hash_to_field_gives_up_to_255_blocks_of_the_hash() {
    // 170 elements of 48 bytes are the 8160 bytes of 255 blocks of SHA-256; one more is refused.
    let s = hash::p256();
    assert_eq!(s.hash_to_field(b"m", b"DST", 170).len(), 170);
    let r = std::panic::catch_unwind(|| s.hash_to_field(b"m", b"DST", 171));
    let e = r.unwrap_err();
    assert!(e.downcast_ref::<String>().unwrap().contains("255 blocks"));
    let r = std::panic::catch_unwind(|| s.hash_to_field(b"m", b"DST", usize::MAX));
    let e = r.unwrap_err();
    assert!(
        e.downcast_ref::<String>()
            .unwrap()
            .contains("too many to expand")
    );
}

#[test]
fn l_is_the_bits_of_the_prime_and_k_in_whole_bytes() {
    // L = ceil((ceil(log2 p) + k) / 8): one byte for p = 2 and k = 0, whose log2 is 1.
    let cases: [(Number, u32, usize); 6] = [
        (n(2), 0, 1),
        (n(2), 7, 1),
        (n(2), 8, 2),
        (n(3), 0, 1),
        (n(257), 128, 18),
        ((Number::from(1u8) << 127usize) - 1u8, 128, 32),
    ];
    for (p, k, l) in cases {
        let f = PrimeField::new(&p);
        let u = hash_to_field::<Sha256>(&f, b"x", b"DST", 3, k);
        let bytes = expand_message_xmd::<Sha256>(b"x", b"DST", 3 * l);
        let want: Vec<Number> = bytes
            .chunks(l)
            .map(|c| Number::new(c.to_vec(), ByteOrder::BigEndian).rem_euclid(&p))
            .collect();
        assert_eq!(u.iter().map(Fp::value).collect::<Vec<_>>(), want, "{p} {k}");
    }
}

#[test]
fn sgn0_is_the_parity_of_the_value() {
    let f = PrimeField::new(&n(101));
    assert!(!sgn0(&f.zero()));
    assert!(sgn0(&f.one()));
    assert!(!sgn0(&-f.one())); // 100
    for v in 0..101 {
        assert_eq!(sgn0(&f.from_u64(v)), v % 2 == 1);
    }
    let f = hash::p521().field();
    assert!(sgn0(&-f.from_u64(2))); // p − 2 is odd
}

/// `x³ + a·x + b`.
fn weierstrass_g<'f>(x: &Fp<'f>, a: &Fp<'f>, b: &Fp<'f>) -> Fp<'f> {
    x * x * x + a * x + b
}

/// `x³ + (J/K)·x² + x/K²`, with `jk = J/K` and `k`.
fn montgomery_g<'f>(x: &Fp<'f>, jk: &Fp<'f>, k: &Fp<'f>) -> Fp<'f> {
    x * x * x + jk * x * x + x / &(k * k)
}

/// Primes of each class modulo 8, for the square roots: 3 and 7 modulo 4, 5 and 1 modulo 8.
const SMALL_PRIMES: [i64; 6] = [103, 107, 101, 109, 97, 113];

#[test]
fn the_simplified_swu_map_follows_section_6_6_2_on_small_fields_in_full() {
    let mut rng = Rng(0x5eed_5157_u64);
    for p in SMALL_PRIMES {
        let f = PrimeField::new(&n(p));
        let mut built = 0;
        while built < 3 {
            let (a, b) = (f.element(&rng.below(&n(p))), f.element(&rng.below(&n(p))));
            let disc = f.from_u64(4) * &a * &a * &a + f.from_u64(27) * &b * &b;
            if a.is_zero() || b.is_zero() || disc.is_zero() {
                continue;
            }
            let z = SimplifiedSwu::find_z(&a, &b);
            let map = SimplifiedSwu::new(&a, &b, &z);
            assert_eq!((map.a(), map.b(), map.z()), (&a, &b, &z));
            for v in 0..p as u64 {
                let u = f.from_u64(v);
                let (x, y) = map.map(&u);
                assert_eq!(y.square(), weierstrass_g(&x, &a, &b), "p = {p}, u = {v}");
                // x is x1, or x2 = Z·u²·x1 when g(x1) is not a square.
                let zu2 = &z * &u * &u;
                let den = &zu2 * &zu2 + &zu2;
                let x1 = if den.is_zero() {
                    &b / &(&z * &a)
                } else {
                    -&b / &a * (f.one() + den.inverse().unwrap())
                };
                let want = if weierstrass_g(&x1, &a, &b).legendre() >= 0 {
                    x1.clone()
                } else {
                    &zu2 * &x1
                };
                assert_eq!(x, want, "p = {p}, u = {v}");
                if !y.is_zero() {
                    assert_eq!(sgn0(&y), sgn0(&u), "p = {p}, u = {v}");
                }
                if !u.is_zero() {
                    assert_eq!(map.map(&-u.clone()), (x, -y), "p = {p}, u = {v}");
                }
            }
            built += 1;
        }
    }
}

/// Whether `z` meets the criteria of section 6.6.1 on `y² = x³ + a·x + b`, by brute force: the
/// squares found among all the squares of the field.
fn svdw_valid_by_brute_force(a: &Fp<'_>, b: &Fp<'_>, z: &Fp<'_>, p: u64) -> bool {
    let f = a.field();
    let squares: Vec<Fp<'_>> = (0..p).map(|v| f.from_u64(v).square()).collect();
    let square = |x: &Fp<'_>| squares.contains(x);
    let gz = weierstrass_g(z, a, b);
    if gz.is_zero() {
        return false;
    }
    let h = -(f.from_u64(3) * z * z + f.from_u64(4) * a) / (f.from_u64(4) * &gz);
    !h.is_zero()
        && square(&h)
        && (square(&gz) || square(&weierstrass_g(&(-z / &f.from_u64(2)), a, b)))
}

#[test]
fn the_shallue_van_de_woestijne_map_follows_section_6_6_1_on_small_fields_in_full() {
    // Random curves, A = 0 and B = 0 among them, every u: on the curve, x is the first of x1, x2,
    // x3 whose g is a square, computed again here, and sgn0(y) = sgn0(u); Z as appendix H.1 finds
    // it, each candidate before it refused and it accepted, as the criteria say by brute force.
    let mut rng = Rng(0x5d0_5d0_u64);
    for p in SMALL_PRIMES {
        let f = PrimeField::new(&n(p));
        let mut built = 0;
        while built < 4 {
            let (mut a, b) = (f.element(&rng.below(&n(p))), f.element(&rng.below(&n(p))));
            if built == 0 {
                a = f.zero();
            }
            let disc = f.from_u64(4) * &a * &a * &a + f.from_u64(27) * &b * &b;
            if disc.is_zero() {
                continue;
            }
            let z = ShallueVanDeWoestijne::find_z(&a, &b);
            let mut tried = 0;
            'candidates: for c in 1..p as u64 {
                for cand in [f.from_u64(c), -f.from_u64(c)] {
                    let ok = svdw_valid_by_brute_force(&a, &b, &cand, p as u64);
                    let r = std::panic::catch_unwind(|| ShallueVanDeWoestijne::new(&a, &b, &cand));
                    assert_eq!(r.is_ok(), ok, "p = {p}, Z = {cand}");
                    tried += 1;
                    if cand == z {
                        assert!(ok);
                        break 'candidates;
                    }
                    assert!(!ok, "p = {p}: {cand} comes before {z}");
                }
            }
            assert!(tried > 0);
            let map = ShallueVanDeWoestijne::new(&a, &b, &z);
            assert_eq!((map.a(), map.b(), map.z()), (&a, &b, &z));
            let gz = weierstrass_g(&z, &a, &b);
            let k = f.from_u64(3) * &z * &z + f.from_u64(4) * &a;
            let c3 = (-(&gz * &k)).sqrt().unwrap();
            let c3 = if sgn0(&c3) { -c3 } else { c3 };
            let c4 = -(f.from_u64(4) * &gz) / &k;
            for v in 0..p as u64 {
                let u = f.from_u64(v);
                let (x, y) = map.map(&u);
                assert_eq!(y.square(), weierstrass_g(&x, &a, &b), "p = {p}, u = {v}");
                let t1 = &u * &u * &gz;
                let (plus, minus) = (f.one() + &t1, f.one() - &t1);
                let tv3 = (&plus * &minus).inverse().unwrap_or_else(|| f.zero());
                let tv5 = &u * &minus * &tv3 * &c3;
                let half = -(&z / &f.from_u64(2));
                let x1 = &half - &tv5;
                let x2 = &half + &tv5;
                let x3 = &z + &c4 * &(&plus * &plus * &tv3).square();
                let want = [x1, x2, x3]
                    .into_iter()
                    .find(|x| weierstrass_g(x, &a, &b).legendre() >= 0)
                    .expect("one of them");
                assert_eq!(x, want, "p = {p}, u = {v}");
                if !y.is_zero() {
                    assert_eq!(sgn0(&y), sgn0(&u), "p = {p}, u = {v}");
                }
            }
            built += 1;
        }
    }
}

#[test]
#[should_panic(expected = "Z does not meet the criteria of RFC 9380, section 6.6.1")]
fn a_zero_z_is_refused_by_the_svdw_map() {
    // Section 6.6.1 asks for a nonzero Z; on y² = x³ + x + 1 over F_5, Z = 0 meets the four
    // numbered criteria (g(0) = 1, h(0) = −1, a square, g(0) a square), so it must be refused
    // apart from them.
    let f = field(5);
    let _ = ShallueVanDeWoestijne::new(&f.one(), &f.one(), &f.zero());
}

#[test]
#[should_panic(expected = "Z does not meet the criteria of RFC 9380, section 6.6.1")]
fn a_z_with_g_of_z_zero_is_refused_by_the_svdw_map() {
    // y² = x³ + 3 over F_103 has the point (x, 0) for x³ = −3: g of it is 0.
    let f = field(103);
    let b = f.from_u64(3);
    let root = (0..103u64)
        .map(|x| f.from_u64(x))
        .find(|x| weierstrass_g(x, &f.zero(), &b).is_zero())
        .expect("−3 is a cube modulo 103");
    let _ = ShallueVanDeWoestijne::new(&f.zero(), &b, &root);
}

#[test]
#[should_panic(expected = "the curve is singular")]
fn a_singular_curve_is_refused_by_the_svdw_map() {
    let f = field(103);
    let _ = ShallueVanDeWoestijne::find_z(&-f.from_u64(3), &f.from_u64(2));
}

#[test]
#[should_panic(expected = "prime fields above 3")]
fn the_field_of_3_is_refused_by_the_svdw_map() {
    let f = field(3);
    let _ = ShallueVanDeWoestijne::find_z(&f.one(), &f.one());
}

#[test]
fn elligator_2_follows_section_6_7_1_on_small_fields_in_full() {
    let mut rng = Rng(0xe112_2a7e_u64);
    for p in SMALL_PRIMES {
        let f = PrimeField::new(&n(p));
        let mut built = 0;
        while built < 3 {
            let (j, k) = (f.element(&rng.below(&n(p))), f.element(&rng.below(&n(p))));
            if j.is_zero() || k.is_zero() {
                continue;
            }
            let d = (&j * &j - f.from_u64(4)) / (&k * &k);
            if d.legendre() != -1 {
                continue;
            }
            let z = Elligator2::find_z(&f);
            let map = Elligator2::new(&j, &k, &z);
            assert_eq!((map.j(), map.k(), map.z()), (&j, &k, &z));
            let jk = &j / &k;
            let mut exceptional = 0;
            for v in 0..p as u64 {
                let u = f.from_u64(v);
                let (s, t) = map.map(&u);
                assert_eq!(
                    &k * &t * &t,
                    &s * &s * &s + &j * &s * &s + &s,
                    "p = {p}, u = {v}"
                );
                let den = &z * &u * &u + f.one();
                let x1 = if den.is_zero() {
                    exceptional += 1;
                    -jk.clone()
                } else {
                    -(&jk / &den)
                };
                let (x, y) = (&s / &k, &t / &k);
                if montgomery_g(&x1, &jk, &k).legendre() >= 0 {
                    assert_eq!(x, x1, "p = {p}, u = {v}");
                    assert!(y.is_zero() || sgn0(&y), "p = {p}, u = {v}");
                } else {
                    assert_eq!(x, -x1 - &jk, "p = {p}, u = {v}");
                    assert!(!sgn0(&y), "p = {p}, u = {v}");
                }
                assert_eq!(map.map(&-u.clone()), (s, t), "p = {p}, u = {v}");
            }
            // 1 + Z·u² = 0 has two roots u when −1/Z is a square, as it is for p = 3 mod 4.
            assert_eq!(exceptional, if p % 4 == 3 { 2 } else { 0 }, "p = {p}");
            built += 1;
        }
    }
}

#[test]
fn the_simplified_swu_maps_exceptional_inputs_to_b_over_z_a() {
    // u = 0 and u = ±sqrt(−1/Z) make Z²u⁴ + Z·u² zero; then x1 = B / (Z·A), whose g is a square.
    let e = e_prime();
    let curves: [(&Weierstrass, i64); 4] = [
        (named::p256(), -10),
        (named::p384(), -12),
        (named::p521(), -4),
        (&e, -11),
    ];
    for (curve, z) in curves {
        let f = PrimeField::new(curve.modulus());
        let (a, b, z) = (f.element(curve.a()), f.element(curve.b()), f.element(&n(z)));
        let map = SimplifiedSwu::new(&a, &b, &z);
        let x1 = &b / &(&z * &a);
        let root = (-(f.one() / &z)).sqrt().expect("−1/Z is a square");
        for u in [f.zero(), root.clone(), -root] {
            let (x, y) = map.map(&u);
            assert_eq!(x, x1);
            assert_eq!(y.square(), &x * &x * &x + &a * &x + &b);
            assert_eq!(sgn0(&y), sgn0(&u));
            assert!(curve.point(&x.value(), &y.value()).is_ok());
        }
    }
    // The same three inputs through the suites, to the curve and, on secp256k1, on through the
    // isogeny: the point is (B / (Z·A), y) with the sign of u, or its image.
    for s in [hash::p256(), hash::p384(), hash::p521(), hash::secp256k1()] {
        let f = s.field();
        let (a, b) = match s.isogeny() {
            None => (f.element(s.domain().a()), f.element(s.domain().b())),
            Some(_) => (f.element(e.a()), f.element(e.b())),
        };
        let x1 = &b / &(s.z() * &a);
        let root = (-(f.one() / s.z())).sqrt().expect("−1/Z is a square");
        for u in [f.zero(), root.clone(), -root] {
            let y = weierstrass_g(&x1, &a, &b).sqrt().unwrap();
            let y = if sgn0(&y) == sgn0(&u) { y } else { -y };
            let want = match s.isogeny() {
                None => (x1.clone(), y),
                Some(iso) => iso.map(&x1, &y).unwrap(),
            };
            let q = s.map_to_curve(&u);
            assert_eq!(xy(&q), (want.0.value(), want.1.value()), "{}", s.ro_id());
            assert!(s.domain().in_subgroup(&q));
        }
    }
}

#[test]
fn curve25519_and_edwards25519_take_u_zero_to_the_point_of_order_2_and_the_identity() {
    // u = 0: x1 = −J, g(−J) = −J is not a square, so x = x2 = 0 and y = 0, the point (0, 0) of
    // order 2; its image on edwards25519 is exceptional (t = 0) and goes to (0, 1).
    let c = hash::curve25519();
    let q = c.map_to_curve(&c.field().zero());
    assert_eq!(xy(&q), (n(0), n(0)));
    assert!(q.double().is_identity());
    assert!(c.clear_cofactor(&q).is_identity());
    let e = hash::edwards25519();
    let q = e.map_to_curve(&e.field().zero());
    assert!(q.is_identity());
    assert_eq!(xy(&q), (n(0), n(1)));
}

/// RFC 7748's map from curve25519 to edwards25519 with `c1 = sqrt(−486664)` of `sgn0 = 0`,
/// computed here on numbers.
fn to_edwards25519(s: &Number, t: &Number) -> (Number, Number) {
    let p = named::curve25519().modulus();
    let c1 = n(-486664).rem_euclid(p).sqrt_mod(p).unwrap();
    let c1 = if c1.bit(0) { p - &c1 } else { c1 };
    let inv = |x: Number| x.rem_euclid(p).mod_inverse(p).unwrap();
    let v = (&c1 * s * inv(t.clone())).rem_euclid(p);
    let w = ((s - 1u8) * inv(s + 1u8)).rem_euclid(p);
    (v, w)
}

#[test]
fn the_rational_map_takes_curve25519s_base_point_to_edwards25519s() {
    // The sign of c1 is fixed by this (RFC 9380, section 6.8.1, and RFC 7748's erratum 4730):
    // the base point (9, V) with the erratum's V goes to edwards25519's base point.
    let v = Number::from_str_radix(
        "43114425171068552920764898935933967039370386198203806730763910166200978582548",
        10,
        ByteOrder::BigEndian,
    )
    .unwrap();
    assert_eq!(
        to_edwards25519(&n(9), &v),
        xy(&named::ed25519().generator())
    );
    // named::curve25519()'s generator has the V printed in RFC 7748 before the erratum, −V.
    let g = named::curve25519().generator();
    assert_eq!(xy(&-g), (n(9), v));
    // edwards25519's suite is curve25519's map followed by this one.
    let (c, e) = (hash::curve25519(), hash::edwards25519());
    let mut rng = Rng(0x2551_9255_1900_u64);
    for _ in 0..size(20, 3) {
        let u = c.field().element(&rng.number(40));
        let (s, t) = xy(&c.map_to_curve(&u));
        assert_eq!(xy(&e.map_to_curve(&u)), to_edwards25519(&s, &t));
    }
}

#[test]
fn the_isogeny_maps_e_prime_onto_secp256k1_as_a_homomorphism() {
    let ep = e_prime();
    let k1 = named::secp256k1();
    let s = hash::secp256k1();
    let iso = s.isogeny().unwrap();
    let f = s.field();
    let image = |p: &Point<'_, Weierstrass>| -> Point<'static, Weierstrass> {
        match p.coordinates() {
            None => k1.identity(),
            Some((x, y)) => match iso.map(&f.element(&x), &f.element(&y)) {
                None => k1.identity(),
                Some((x, y)) => k1.point(&x.value(), &y.value()).expect("on secp256k1"),
            },
        }
    };
    let mut rng = Rng(0x1503_e3e3_u64);
    let mut points = Vec::new();
    while points.len() < size(6, 2) {
        let x = rng.below(ep.modulus());
        if let Some(p) = ep.lift(&x, rng.next() % 2 == 1) {
            points.push(p);
        }
    }
    for p in &points {
        for q in &points {
            assert_eq!(image(&(p + q)), image(p) + image(q));
        }
        assert_eq!(image(&p.double()), image(p).double());
        assert_eq!(image(&-p), -image(p));
        assert!(k1.in_subgroup(&image(p)));
    }
    // The kernel: x_den = (x − x0)², and g'(x0) is not a square, so the points of order 3 in the
    // kernel are not rational and no point of E' over F_p is exceptional.
    let x0 = f.element(&hex(
        "89291c84de3e11f1041da6957255eed5fc964a4df050df221d6ad4ce6ab9c5a5",
    ));
    assert_eq!(iso.map(&x0, &f.one()), None);
    assert_eq!(iso.map(&x0, &f.zero()), None);
    let a = f.element(ep.a());
    let g = &x0 * &x0 * &x0 + &a * &x0 + f.from_u64(1771);
    assert_eq!(g.legendre(), -1);
}

#[test]
fn secp256k1s_hash_is_one_isogeny_of_the_sum_on_e_prime() {
    // Section 6.6.3: the isogeny commutes with addition, so the sum may be taken on E'.
    let ep = e_prime();
    let s = hash::secp256k1();
    let f = s.field();
    let swu = SimplifiedSwu::new(&f.element(ep.a()), &f.element(ep.b()), s.z());
    let dst = b"QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_RO_";
    for msg in ["", "abc", "abcdef0123456789"] {
        let u = s.hash_to_field(msg.as_bytes(), dst, 2);
        let q: Vec<_> = u
            .iter()
            .map(|u| {
                let (x, y) = swu.map(u);
                ep.point(&x.value(), &y.value()).unwrap()
            })
            .collect();
        let (x, y) = xy(&(&q[0] + &q[1]));
        let (x, y) = s
            .isogeny()
            .unwrap()
            .map(&f.element(&x), &f.element(&y))
            .unwrap();
        let p = s.hash_to_curve(msg.as_bytes(), dst);
        assert_eq!(xy(&p), (x.value(), y.value()));
    }
}

/// Checks that the suite's outputs for random messages are in the subgroup, repeat, and change
/// with the tag.
fn check_random<C: Curve>(suite: &'static Suite<C>, rng: &mut Rng, count: usize) {
    for i in 0..count {
        let msg: Vec<u8> = (0..i * 7).map(|_| rng.next() as u8).collect();
        let dst = format!("A-TEST-V01-CS01-with-{}", suite.ro_id());
        let p = suite.hash_to_curve(&msg, dst.as_bytes());
        assert!(suite.domain().in_subgroup(&p));
        assert!(!p.is_identity());
        assert_eq!(p, suite.hash_to_curve(&msg, dst.as_bytes()));
        assert_ne!(p, suite.hash_to_curve(&msg, b"ANOTHER-TAG"));
        let q = suite.encode_to_curve(&msg, dst.as_bytes());
        assert!(suite.domain().in_subgroup(&q));
        assert_ne!(p, q);
    }
}

#[test]
fn random_messages_hash_into_the_prime_order_subgroup() {
    let mut rng = Rng(0x4a5b_6c7d_u64);
    let c = size(4, 1);
    check_random(hash::p256(), &mut rng, c);
    check_random(hash::p384(), &mut rng, c);
    check_random(hash::p521(), &mut rng, c);
    check_random(hash::secp256k1(), &mut rng, c);
    check_random(hash::curve25519(), &mut rng, c);
    check_random(hash::edwards25519(), &mut rng, c);
}

/// The order of a point of the 8-torsion: 1, 2, 4 or 8 (a panic for anything else).
fn small_order<C: Curve>(t: &Point<'_, C>) -> u32 {
    let mut q = t.clone();
    for order in [1, 2, 4, 8] {
        if q.is_identity() {
            return order;
        }
        q = q.double();
    }
    panic!("not a point of the 8-torsion")
}

/// Takes the 8-torsion parts `ℓ·Q` of the uncleared points `Q` the suite maps to, until it has
/// points of each order 2, 4 and 8, and checks that the cofactor is cleared from each, alone and
/// added to the generator.
fn check_torsion<C: Curve>(suite: &'static Suite<C>, rng: &mut Rng) {
    let l = suite.domain().order();
    let g = suite.domain().generator();
    let mut seen = [0u32; 4];
    let mut tries = 0;
    while seen[1..].contains(&0) {
        tries += 1;
        assert!(tries < 400, "no points of every order");
        let u = suite.field().element(&rng.number(40));
        let t = &suite.map_to_curve(&u) * l;
        let order = small_order(&t);
        seen[order.trailing_zeros() as usize] += 1;
        assert!(suite.clear_cofactor(&t).is_identity(), "order {order}");
        assert_eq!(
            suite.clear_cofactor(&(&g + &t)),
            &g * &n(8),
            "order {order}"
        );
    }
}

#[test]
fn the_cofactor_is_cleared_from_points_of_order_2_4_and_8() {
    // curve25519's and edwards25519's 8-torsion goes to the identity, whatever its order.
    let mut rng = Rng(0x0807_0605_0403_u64);
    check_torsion(hash::curve25519(), &mut rng);
    check_torsion(hash::edwards25519(), &mut rng);
    // The two points of order 2 the maps reach at u = 0 and edwards25519's (0, −1).
    let c = hash::curve25519();
    let small = c.domain().identity() + c.map_to_curve(&c.field().zero());
    assert_eq!(small_order(&small), 2);
    assert!(c.clear_cofactor(&small).is_identity());
    let e = hash::edwards25519();
    let order2 = e
        .domain()
        .point(&n(0), &(e.field().modulus() - 1u8))
        .unwrap();
    assert_eq!(small_order(&order2), 2);
    assert!(e.clear_cofactor(&order2).is_identity());
}

/// A field of `p` and two of its elements.
fn field(p: i64) -> PrimeField {
    PrimeField::new(&n(p))
}

#[test]
#[should_panic(expected = "Z does not meet the criteria")]
fn a_square_z_is_refused_by_the_simplified_swu_map() {
    let f = field(103);
    let _ = SimplifiedSwu::new(&f.from_u64(3), &f.from_u64(5), &f.from_u64(4));
}

#[test]
#[should_panic(expected = "Z does not meet the criteria")]
fn z_of_minus_one_is_refused_by_the_simplified_swu_map() {
    // −1 is not a square modulo 103 (3 mod 4), but is excluded.
    let f = field(103);
    let _ = SimplifiedSwu::new(&f.from_u64(3), &f.from_u64(5), &-f.one());
}

#[test]
fn a_z_with_a_reducible_g_minus_z_is_refused_by_the_simplified_swu_map() {
    // Criterion 3: find a Z that meets 1, 2 and 4 but for which x³ + 3x + 5 − Z has a root.
    let f = field(103);
    let (a, b) = (f.from_u64(3), f.from_u64(5));
    let mut found = 0;
    for v in 1..103u64 {
        let z = f.from_u64(v);
        if z.legendre() != -1
            || z == -f.one()
            || weierstrass_g(&(&b / &(&z * &a)), &a, &b).legendre() < 0
        {
            continue;
        }
        let reducible = (0..103u64).any(|x| (weierstrass_g(&f.from_u64(x), &a, &b) - &z).is_zero());
        let r = std::panic::catch_unwind(|| SimplifiedSwu::new(&a, &b, &z));
        assert_eq!(r.is_err(), reducible, "Z = {v}");
        found += usize::from(reducible);
    }
    assert!(found > 0);
}

#[test]
#[should_panic(expected = "needs A ≠ 0 and B ≠ 0")]
fn a_zero_a_is_refused_by_the_simplified_swu_map() {
    let f = field(103);
    let _ = SimplifiedSwu::new(&f.zero(), &f.from_u64(5), &f.from_u64(5));
}

#[test]
#[should_panic(expected = "needs A ≠ 0 and B ≠ 0")]
fn a_zero_b_is_refused_by_find_z() {
    let f = field(103);
    let _ = SimplifiedSwu::find_z(&f.from_u64(3), &f.zero());
}

#[test]
#[should_panic(expected = "singular")]
fn a_singular_curve_is_refused_by_the_simplified_swu_map() {
    // 4·A³ + 27·B² = 0 for A = −3, B = 2: y² = (x − 1)²(x + 2).
    let f = field(103);
    let _ = SimplifiedSwu::new(&-f.from_u64(3), &f.from_u64(2), &f.from_u64(5));
}

#[test]
#[should_panic(expected = "prime fields above 3")]
fn the_field_of_3_is_refused_by_the_simplified_swu_map() {
    let f = field(3);
    let _ = SimplifiedSwu::new(&f.one(), &f.one(), &f.from_u64(2));
}

#[test]
#[should_panic(expected = "Z must not be a square")]
fn a_square_z_is_refused_by_elligator_2() {
    let f = field(101);
    let _ = Elligator2::new(&f.from_u64(4), &f.from_u64(3), &f.from_u64(4));
}

#[test]
#[should_panic(expected = "neither 0 nor a square")]
fn a_zero_j2_minus_4_is_refused_by_elligator_2() {
    // J = 2: J² − 4 = 0, a singular curve.
    let f = field(101);
    let _ = Elligator2::new(&f.from_u64(2), &f.from_u64(3), &f.from_u64(2));
}

#[test]
fn a_nonzero_square_j2_minus_4_over_k2_is_refused_by_elligator_2() {
    // Over F_101 with K = 3, J² − 4 is 5, 21 and 45 for J = 3, 5 and 7, each a nonzero square:
    // the curve has three points of order 2 and Elligator 2 does not apply. J = 4 gives 12, not a
    // square, which is accepted.
    let f = field(101);
    let k = f.from_u64(3);
    for j in [3u64, 5, 7] {
        let d = (f.from_u64(j * j) - f.from_u64(4)) / (&k * &k);
        assert_eq!(d.legendre(), 1, "J = {j}");
        let r = std::panic::catch_unwind(|| Elligator2::new(&f.from_u64(j), &k, &f.from_u64(2)));
        let e = r.unwrap_err();
        let msg = e
            .downcast_ref::<&str>()
            .map(|m| m.to_string())
            .unwrap_or_else(|| e.downcast_ref::<String>().cloned().unwrap_or_default());
        assert!(msg.contains("neither 0 nor a square"), "J = {j}: {msg}");
    }
    let _ = Elligator2::new(&f.from_u64(4), &k, &f.from_u64(2));
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn elligator_2_refuses_a_z_of_another_field() {
    let (f101, f107) = (field(101), field(107));
    let _ = Elligator2::new(&f101.from_u64(4), &f101.from_u64(3), &f107.from_u64(2));
}

#[test]
#[should_panic(expected = "the curve is singular")]
fn find_z_refuses_a_singular_curve() {
    // 4·(−3)³ + 27·2² = 0: y² = (x − 1)²(x + 2).
    let f = field(103);
    let _ = SimplifiedSwu::find_z(&-f.from_u64(3), &f.from_u64(2));
}

#[test]
#[should_panic(expected = "J ≠ 0 and K ≠ 0")]
fn a_zero_k_is_refused_by_elligator_2() {
    let f = field(101);
    let _ = Elligator2::new(&f.from_u64(4), &f.zero(), &f.from_u64(2));
}

#[test]
#[should_panic(expected = "every element of the field of 2 elements is a square")]
fn find_z_for_elligator_2_refuses_the_field_of_2() {
    let f = field(2);
    let _ = Elligator2::find_z(&f);
}

#[test]
#[should_panic(expected = "no coefficients")]
fn an_isogeny_map_needs_coefficients() {
    let f = field(101);
    let _ = IsogenyMap::new(&[f.one()], &[], &[f.one()], &[f.one()]);
}

#[test]
#[should_panic(expected = "leading coefficient of the map is 0")]
fn an_isogeny_map_needs_nonzero_leading_coefficients() {
    let f = field(101);
    let _ = IsogenyMap::new(&[f.one(), f.zero()], &[f.one()], &[f.one()], &[f.one()]);
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn map_to_curve_refuses_an_element_of_another_field() {
    let f = field(101);
    let _ = hash::p256().map_to_curve(&f.from_u64(5));
}

#[test]
#[should_panic(expected = "the points are on different curves")]
fn clear_cofactor_refuses_a_point_of_another_curve() {
    let curve = Montgomery::new(&n(6), &n(1), &n(101)).unwrap();
    let p = curve.point(&n(3), &n(36)).unwrap();
    let _ = hash::curve25519().clear_cofactor(&p);
}

#[test]
#[should_panic(expected = "the domain separation tag must not be empty")]
fn hash_to_curve_refuses_an_empty_tag() {
    let _ = hash::edwards25519().hash_to_curve(b"abc", b"");
}

#[test]
fn an_edwards_suite_gives_edwards_points() {
    // The types follow the curves: Edwards points for edwards25519, Montgomery for curve25519.
    let e: Point<'static, Edwards> = hash::edwards25519().encode_to_curve(b"x", b"DST");
    let m: Point<'static, Montgomery> = hash::curve25519().encode_to_curve(b"x", b"DST");
    assert!(named::ed25519().in_subgroup(&e));
    assert!(named::curve25519().in_subgroup(&m));
}

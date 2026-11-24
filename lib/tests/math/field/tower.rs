//! The towers `F_p² → F_p⁶ → F_p¹²` of `a::math::field` against arkworks (every operation on
//! seeded elements of BN254's and BLS12-381's towers, by digests of the exact outputs), PARI (norms,
//! traces and squares in the flat field `F_p[w]/(w¹² − 2ξ₀w⁶ + ξ₀² + 1)`), and identities: the
//! Frobenius maps against powers by `pᵏ`, norms and traces against their definitions, square roots
//! exhaustively in `F_103²`.
//!
//! The reference digests are in `data/tower_ref.txt`, from `../pairing/data/gen_ark.rs` and
//! `../pairing/data/gen_py.py`.

use a::math::field::{Fp, Fp2, Fp6, Fp12, PrimeField, Tower};
use a::math::numbers::{ByteOrder, Number};

const REF: &str = include_str!("data/tower_ref.txt");

/// The reference value of `key`.
fn reference(key: &str) -> u64 {
    let line = REF
        .lines()
        .find(|l| l.split(' ').next() == Some(key))
        .unwrap_or_else(|| panic!("no reference {key}"));
    u64::from_str_radix(line.split(' ').nth(1).unwrap(), 16).unwrap()
}

/// SplitMix64, as the generators have it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// `n` words, the first the least significant.
    fn number(&mut self, n: usize) -> Number {
        let bytes: Vec<u8> = (0..n).flat_map(|_| self.next().to_le_bytes()).collect();
        Number::new(bytes, ByteOrder::LittleEndian)
    }
}

/// FNV-1a, 64 bits.
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn eat(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn fp(&mut self, x: &Fp<'_>) {
        let n = x.field().bits().div_ceil(8) as usize;
        let mut b = x.value().value;
        if x.value().byte_order == ByteOrder::LittleEndian {
            b.reverse();
        }
        let start = b.iter().position(|&v| v != 0).unwrap_or(b.len());
        let mut out = vec![0u8; n - (b.len() - start)];
        out.extend_from_slice(&b[start..]);
        self.eat(&out);
    }

    fn fp2(&mut self, x: &Fp2<'_>) {
        self.fp(x.c0());
        self.fp(x.c1());
    }

    fn fp6(&mut self, x: &Fp6<'_>) {
        self.fp2(x.c0());
        self.fp2(x.c1());
        self.fp2(x.c2());
    }

    fn fp12(&mut self, x: &Fp12<'_>) {
        self.fp6(x.c0());
        self.fp6(x.c1());
    }
}

/// The towers with their names in the references and the words of a random element of `F_p`.
fn towers() -> [(&'static str, &'static Tower<'static>, usize); 2] {
    [("bls", Tower::bls12_381(), 7), ("bn", Tower::bn254(), 5)]
}

struct Gen<'t> {
    t: &'t Tower<'t>,
    r: Rng,
    words: usize,
}

impl<'t> Gen<'t> {
    fn fp(&mut self) -> Fp<'t> {
        self.t.base().element(&self.r.number(self.words))
    }

    fn fp2(&mut self) -> Fp2<'t> {
        let c0 = self.fp();
        let c1 = self.fp();
        Fp2::new(self.t, c0, c1)
    }

    fn fp6(&mut self) -> Fp6<'t> {
        let c0 = self.fp2();
        let c1 = self.fp2();
        let c2 = self.fp2();
        Fp6::new(c0, c1, c2)
    }

    fn fp12(&mut self) -> Fp12<'t> {
        let c0 = self.fp6();
        let c1 = self.fp6();
        Fp12::new(c0, c1)
    }
}

/// The element `x^((p⁶ − 1)(p² + 1))` of the cyclotomic subgroup.
fn cyclotomic<'t>(x: &Fp12<'t>) -> Fp12<'t> {
    let c = &x.conjugate() / x;
    &c.frobenius(2) * &c
}

#[test]
fn tower_arithmetic_matches_arkworks_on_seeded_elements() {
    let names = [
        "fp12_mul",
        "fp12_square",
        "fp12_inverse",
        "fp12_div",
        "fp12_frob1",
        "fp12_frob2",
        "fp12_frob3",
        "fp12_frob6",
        "fp12_frob7",
        "fp12_pow",
        "fp12_mul_by_014",
        "fp12_mul_by_034",
        "fp12_cyclotomic",
        "fp12_norm",
        "fp6_mul",
        "fp6_square",
        "fp6_inverse",
        "fp6_frob1",
        "fp6_norm",
        "fp6_mul_by_01",
        "fp6_mul_by_1",
        "fp6_mul_by_v",
        "fp2_mul",
        "fp2_square",
        "fp2_inverse",
        "fp2_norm",
        "fp2_mul_by_xi",
        "fp2_is_square",
    ];
    let cases = if cfg!(miri) { 1 } else { 16 };
    for (name, t, words) in towers() {
        let mut g = Gen {
            t,
            r: Rng(1),
            words,
        };
        let mut hs: Vec<Fnv> = names.iter().map(|_| Fnv::new()).collect();
        let v = Fp6::new(Fp2::zero(t), Fp2::one(t), Fp2::zero(t));
        for _ in 0..cases {
            let a = g.fp12();
            let b = g.fp12();
            let a6 = g.fp6();
            let b6 = g.fp6();
            let a2 = g.fp2();
            let b2 = g.fp2();
            let k = g.r.number(1);
            let c = cyclotomic(&a);
            let outputs12 = [
                &a * &b,
                a.square(),
                a.inverse().unwrap(),
                &a / &b,
                a.frobenius(1),
                a.frobenius(2),
                a.frobenius(3),
                a.frobenius(6),
                a.frobenius(7),
                a.pow(&k),
                a.mul_by_014(b.c0().c0(), b.c0().c1(), b.c1().c1()),
                a.mul_by_034(b.c0().c0(), b.c1().c0(), b.c1().c1()),
            ];
            let mut i = 0;
            for x in &outputs12 {
                hs[i].fp12(x);
                i += 1;
            }
            hs[i].fp12(&c);
            hs[i].fp12(&c.cyclotomic_square());
            hs[i].fp12(&c.cyclotomic_pow(&k));
            i += 1;
            hs[i].fp6(&a.norm());
            i += 1;
            for x in [
                &a6 * &b6,
                a6.square(),
                a6.inverse().unwrap(),
                a6.frobenius(1),
            ] {
                hs[i].fp6(&x);
                i += 1;
            }
            hs[i].fp2(&a6.norm());
            i += 1;
            for x in [
                a6.mul_by_01(b6.c0(), b6.c1()),
                a6.mul_by_1(b6.c1()),
                &a6 * &v,
            ] {
                hs[i].fp6(&x);
                i += 1;
            }
            for x in [&a2 * &b2, a2.square(), a2.inverse().unwrap()] {
                hs[i].fp2(&x);
                i += 1;
            }
            hs[i].fp(&a2.norm());
            i += 1;
            hs[i].fp2(&a2.mul_by_xi());
            i += 1;
            let squares = [a2.is_square(), b2.is_square(), a2.square().is_square()];
            for (s, x) in squares.iter().zip([&a2, &b2, &a2.square()]) {
                assert_eq!(*s, x.sqrt().is_some());
            }
            hs[i].eat(&squares.map(u8::from));
        }
        if cfg!(miri) {
            continue;
        }
        for (n, h) in names.iter().zip(&hs) {
            assert_eq!(h.0, reference(&format!("{name}_{n}")), "{name} {n}");
        }
    }
}

#[test]
fn tower_norms_traces_and_squares_match_pari() {
    let keys = [
        "fp12_norm",
        "fp12_trace",
        "fp6_norm",
        "fp6_trace",
        "fp2_norm",
        "fp2_trace",
        "absolute",
        "squares",
    ];
    let cases = if cfg!(miri) { 1 } else { 8 };
    for (name, t, words) in towers() {
        let mut g = Gen {
            t,
            r: Rng(5),
            words,
        };
        let mut hs: Vec<Fnv> = keys.iter().map(|_| Fnv::new()).collect();
        for _ in 0..cases {
            let a = g.fp12();
            let b = g.fp6();
            let c = g.fp2();
            hs[0].fp6(&a.norm());
            hs[1].fp6(&a.trace());
            hs[2].fp2(&b.norm());
            hs[3].fp2(&b.trace());
            hs[4].fp(&c.norm());
            hs[5].fp(&c.trace());
            // Norms and traces compose: N_{F_p¹²/F_p} = N_{F_p²/F_p} ∘ N_{F_p⁶/F_p²} ∘ N_{F_p¹²/F_p⁶}.
            hs[6].fp(&a.norm().norm().norm());
            hs[6].fp(&a.trace().trace().trace());
            hs[7].eat(&[
                u8::from(c.is_square()),
                u8::from(c.square().is_square()),
                u8::from(a.is_square()),
                u8::from(a.square().is_square()),
            ]);
        }
        if cfg!(miri) {
            continue;
        }
        for (k, h) in keys.iter().zip(&hs) {
            assert_eq!(h.0, reference(&format!("{name}_pari_{k}")), "{name} {k}");
        }
    }
}

#[test]
fn frobenius_maps_are_the_powers_by_p() {
    for (_, t, words) in towers() {
        let mut g = Gen {
            t,
            r: Rng(7),
            words,
        };
        let p = t.base().modulus().clone();
        let a = g.fp12();
        let a6 = g.fp6();
        let a2 = g.fp2();
        assert_eq!(a.frobenius(1), a.pow(&p));
        assert_eq!(a6.frobenius(1), a6.pow(&p));
        assert_eq!(a2.frobenius(1), a2.pow(&p));
        let p2 = &p * &p;
        assert_eq!(a.frobenius(2), a.frobenius(1).frobenius(1));
        assert_eq!(a6.frobenius(2), a6.pow(&p2));
        for k in 0..13 {
            let mut x = a.clone();
            for _ in 0..k {
                x = x.frobenius(1);
            }
            assert_eq!(a.frobenius(k), x, "k = {k}");
            let mut y = a6.clone();
            for _ in 0..k {
                y = y.frobenius(1);
            }
            assert_eq!(a6.frobenius(k), y, "k = {k}");
        }
        assert_eq!(a.frobenius(12), a);
        assert_eq!(a.frobenius(6), a.conjugate());
        assert_eq!(a6.frobenius(6), a6);
        assert_eq!(a2.frobenius(2), a2);
        // The coefficients: (wⁱ)^(pᵏ) = γ wⁱ.
        let w = Fp12::new(Fp6::zero(t), Fp6::one(t));
        for k in 0..12 {
            let mut wi = Fp12::one(t);
            for i in 0..6 {
                let want = &Fp12::from_fp6(Fp6::from_fp2(t.frobenius_coefficient(k, i))) * &wi;
                assert_eq!(wi.frobenius(k), want, "k = {k}, i = {i}");
                wi = &wi * &w;
            }
        }
    }
}

#[test]
fn norms_traces_and_inverses_follow_their_definitions() {
    for (_, t, words) in towers() {
        let mut g = Gen {
            t,
            r: Rng(8),
            words,
        };
        for _ in 0..3 {
            let (a, b) = (g.fp12(), g.fp12());
            let (a6, b6) = (g.fp6(), g.fp6());
            let (a2, b2) = (g.fp2(), g.fp2());
            assert_eq!(Fp12::from_fp6(a.norm()), &a * &a.frobenius(6));
            assert_eq!(Fp12::from_fp6(a.trace()), &a + &a.frobenius(6));
            assert_eq!(
                Fp6::from_fp2(a6.norm()),
                &(&a6 * &a6.frobenius(2)) * &a6.frobenius(4)
            );
            assert_eq!(
                Fp6::from_fp2(a6.trace()),
                &(&a6 + &a6.frobenius(2)) + &a6.frobenius(4)
            );
            assert_eq!(Fp2::from_base(t, a2.norm()), &a2 * &a2.conjugate());
            assert_eq!(Fp2::from_base(t, a2.trace()), &a2 + &a2.conjugate());
            // Multiplicative and additive.
            assert_eq!((&a * &b).norm(), &a.norm() * &b.norm());
            assert_eq!((&a + &b).trace(), &a.trace() + &b.trace());
            assert_eq!((&a6 * &b6).norm(), &a6.norm() * &b6.norm());
            assert_eq!((&a2 * &b2).norm(), &a2.norm() * &b2.norm());
            // Inverses, quotients, squares, doubles.
            assert!((&a * &a.inverse().unwrap()).is_one());
            assert!((&a6 * &a6.inverse().unwrap()).is_one());
            assert!((&a2 * &a2.inverse().unwrap()).is_one());
            assert_eq!(&(&a / &b) * &b, a);
            assert_eq!(a.square(), &a * &a);
            assert_eq!(a6.square(), &a6 * &a6);
            assert_eq!(a2.square(), &a2 * &a2);
            assert_eq!(a6.double(), &a6 + &a6);
            assert_eq!(&a - &a, Fp12::zero(t));
            assert_eq!(-(-&a), a);
            // The sparse products against the dense ones.
            let (c0, c1, c4) = (g.fp2(), g.fp2(), g.fp2());
            let z = Fp2::zero(t);
            let l014 = Fp12::new(
                Fp6::new(c0.clone(), c1.clone(), z.clone()),
                Fp6::new(z.clone(), c4.clone(), z.clone()),
            );
            assert_eq!(a.mul_by_014(&c0, &c1, &c4), &a * &l014);
            let l034 = Fp12::new(
                Fp6::new(c0.clone(), z.clone(), z.clone()),
                Fp6::new(c1.clone(), c4.clone(), z.clone()),
            );
            assert_eq!(a.mul_by_034(&c0, &c1, &c4), &a * &l034);
            assert_eq!(
                a6.mul_by_01(&c0, &c1),
                &a6 * &Fp6::new(c0.clone(), c1.clone(), z.clone())
            );
            assert_eq!(a6.mul_by_fp2(&c0), &a6 * &Fp6::from_fp2(c0.clone()));
            // Powers: negative, zero, and of 0.
            assert_eq!(
                a.pow(&Number::from(-3)),
                a.pow(&Number::from(3)).inverse().unwrap()
            );
            assert!(a.pow(&Number::from(0u8)).is_one());
            assert!(Fp12::zero(t).pow(&Number::from(5u8)).is_zero());
            assert_eq!(a2.pow(&Number::from(-2)), a2.square().inverse().unwrap());
            assert_eq!(a6.pow(&Number::from(-2)), a6.square().inverse().unwrap());
            // The cyclotomic subgroup.
            let c = cyclotomic(&a);
            assert!(c.is_cyclotomic());
            assert!(!a.is_cyclotomic());
            assert!(c.norm().is_one());
            assert_eq!(c.inverse().unwrap(), c.conjugate());
            assert_eq!(c.cyclotomic_square(), c.square());
            let k = g.r.number(3);
            assert_eq!(c.cyclotomic_pow(&k), c.pow(&k));
            assert_eq!(c.cyclotomic_pow(&-&k), c.pow(&-&k));
            assert!(c.cyclotomic_pow(&Number::from(0u8)).is_one());
        }
        assert!(Fp12::zero(t).inverse().is_none());
        assert!(Fp6::zero(t).inverse().is_none());
        assert!(Fp2::zero(t).inverse().is_none());
        assert!(!Fp12::zero(t).is_cyclotomic());
    }
}

#[test]
fn coefficients_round_trip_and_name_the_powers_of_w() {
    let t = Tower::bn254();
    let c: Vec<Number> = (0..12u32).map(|i| Number::from(1000 + i)).collect();
    let x = Fp12::from_coefficients(t, &c);
    let back: Vec<Number> = x.coefficients().iter().map(Fp::value).collect();
    assert_eq!(back, c);
    // c₀.c₀ is the coefficient of 1, c₁.c₀ of w, c₀.c₁ of w², c₁.c₁ of w³, c₀.c₂ of w⁴, c₁.c₂ of w⁵.
    let w = Fp12::new(Fp6::zero(t), Fp6::one(t));
    let order = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)];
    let mut wi = Fp12::one(t);
    for (i, &(half, j)) in order.iter().enumerate() {
        let mut coeffs = vec![Number::from(0u8); 12];
        coeffs[half * 6 + j * 2] = Number::from(1u8);
        assert_eq!(Fp12::from_coefficients(t, &coeffs), wi, "w^{i}");
        wi = &wi * &w;
    }
    // w⁶ = ξ.
    assert_eq!(wi, Fp12::from_fp6(Fp6::from_fp2(t.xi())));
    // The constants of BLS12-381's tower.
    let b = Tower::bls12_381();
    assert_eq!(
        b.xi(),
        Fp2::from_numbers(b, &Number::from(1u8), &Number::from(1u8))
    );
    assert_eq!(b.beta(), -b.base().one());
}

/// A tower over `F_103`: `103 ≡ 1 (mod 6)` and `≡ 3 (mod 4)`, `ξ = 2 + u`.
fn small() -> (PrimeField, Number) {
    let p = Number::from(103u8);
    (PrimeField::new(&p), p)
}

#[test]
fn a_tower_over_a_small_prime_is_a_field_of_p_to_the_twelve() {
    let (f, p) = small();
    let t = Tower::new(
        &f,
        &Number::from(-1),
        (&Number::from(2u8), &Number::from(1u8)),
    );
    let mut r = Rng(9);
    let mut g = || f.element(&r.number(1));
    for _ in 0..20 {
        let a = Fp12::new(
            Fp6::new(
                Fp2::new(&t, g(), g()),
                Fp2::new(&t, g(), g()),
                Fp2::new(&t, g(), g()),
            ),
            Fp6::new(
                Fp2::new(&t, g(), g()),
                Fp2::new(&t, g(), g()),
                Fp2::new(&t, g(), g()),
            ),
        );
        if a.is_zero() {
            continue;
        }
        // a^(p¹² − 1) = 1 and the Frobenius maps are powers by p.
        let q = p.pow(12);
        assert!(a.pow(&(&q - 1u8)).is_one());
        let mut pk = Number::from(1u8);
        for k in 0..12 {
            assert_eq!(a.frobenius(k), a.pow(&pk), "k = {k}");
            pk = &pk * &p;
        }
        // The norm to F_p lies in F_p: it is the product of the twelve conjugates.
        let mut prod = Fp12::one(&t);
        for k in 0..12 {
            prod = &prod * &a.frobenius(k);
        }
        let n = a.norm().norm().norm();
        assert_eq!(prod, Fp12::from_fp6(Fp6::from_fp2(Fp2::from_base(&t, n))));
        assert_eq!(a.is_square(), a.pow(&((&q - 1u8) / 2u8)).is_one());
    }
}

/// Whether `x` is the larger of `±x` in ZCash's order: by `c₁`, then by `c₀`.
fn larger(x: &Fp2<'_>) -> bool {
    let p = x.tower().base().modulus();
    let large = |c: &Fp<'_>| c.value() * 2u8 > p - 1u8;
    large(x.c1()) || (x.c1().is_zero() && large(x.c0()))
}

#[test]
fn fp2_square_roots_are_exact_for_every_element_of_f_103_squared() {
    let (f, p) = small();
    let t = Tower::new(
        &f,
        &Number::from(-1),
        (&Number::from(2u8), &Number::from(1u8)),
    );
    let mut squares = 0;
    for c0 in 0..103u64 {
        for c1 in 0..103u64 {
            let x = Fp2::new(&t, f.from_u64(c0), f.from_u64(c1));
            let euler = x.is_zero() || x.pow(&((&p * &p - 1u8) / 2u8)).is_one();
            assert_eq!(x.is_square(), euler, "{x}");
            match x.sqrt() {
                Some(r) => {
                    assert_eq!(r.square(), x, "{x}");
                    assert!(!larger(&r), "{x}");
                    squares += 1;
                }
                None => assert!(!euler, "{x}"),
            }
        }
    }
    // 0 and the (p² − 1)/2 nonzero squares.
    assert_eq!(squares, 1 + (103 * 103 - 1) / 2);
}

#[test]
fn fp2_square_roots_on_the_curves_towers() {
    for (_, t, words) in towers() {
        let mut g = Gen {
            t,
            r: Rng(10),
            words,
        };
        for _ in 0..10 {
            let x = g.fp2();
            let r = x.square().sqrt().unwrap();
            assert!(r == x || r == -&x);
            assert!(!larger(&r));
            // ξ is not a square, so neither is x²ξ.
            assert_eq!(x.square().mul_by_xi().sqrt(), None);
            assert!(!x.square().mul_by_xi().is_square());
            // c₁ = 0: an element of F_p is a square of F_p², as √c₀ or √(c₀/β)·u.
            let c0 = Fp2::from_base(t, x.c0().clone());
            let s = c0.sqrt().unwrap();
            assert_eq!(s.square(), c0);
            let u = Fp2::new(t, t.base().zero(), x.c1().clone());
            assert_eq!(u.sqrt().unwrap().square(), u);
        }
        assert!(Fp2::zero(t).sqrt().unwrap().is_zero());
        assert!(Fp2::one(t).sqrt().unwrap().is_one());
        // On F_p, the root of Fp::sqrt.
        let four = Fp2::from_base(t, t.base().from_u64(4));
        assert_eq!(
            four.sqrt().unwrap(),
            Fp2::from_base(t, t.base().from_u64(2))
        );
        let minus_one = -Fp2::one(t);
        let i = minus_one.sqrt().unwrap();
        assert_eq!(i.square(), minus_one);
    }
}

#[test]
fn sgn0_follows_rfc_9380() {
    let t = Tower::bls12_381();
    let n = |a: u64, b: u64| Fp2::new(t, t.base().from_u64(a), t.base().from_u64(b));
    assert!(!n(0, 0).sgn0());
    assert!(n(1, 0).sgn0());
    assert!(!n(2, 1).sgn0());
    assert!(n(0, 1).sgn0());
    assert!(!n(0, 2).sgn0());
    assert!(n(3, 2).sgn0());
}

#[test]
fn elements_hash_and_print_by_their_coefficients() {
    use std::collections::HashSet;
    let t = Tower::bn254();
    let a = Fp2::from_numbers(t, &Number::from(3u8), &Number::from(4u8));
    assert_eq!(a.to_string(), "3 + 4·u");
    assert_eq!(format!("{a:?}"), "3 + 4·u");
    let x = Fp12::from_fp6(Fp6::from_fp2(a.clone()));
    assert!(x.to_string().starts_with("((3 + 4·u) + (0 + 0·u)·v"));
    #[allow(
        clippy::mutable_key_type,
        reason = "the field may hold a cache filled once (`OnceLock`); the hash reads only the value"
    )]
    let set: HashSet<Fp12<'_>> = [x.clone(), x.clone(), x.square()].into_iter().collect();
    assert_eq!(set.len(), 2);
    let mut b = a.clone();
    b += &a;
    b -= Fp2::one(t);
    b *= &a;
    b /= &a;
    assert_eq!(b, &a.double() - &Fp2::one(t));
    assert!(format!("{t:?}").contains("ξ: 9 + 1·u"));
}

#[test]
#[should_panic(expected = "p ≡ 1 (mod 6)")]
fn a_tower_needs_p_one_modulo_six() {
    let f = PrimeField::new(&Number::from(11u8));
    let _ = Tower::new(
        &f,
        &Number::from(-1),
        (&Number::from(1u8), &Number::from(1u8)),
    );
}

#[test]
#[should_panic(expected = "β must not be a square")]
fn a_tower_needs_a_non_square_beta() {
    let (f, _) = small();
    let _ = Tower::new(
        &f,
        &Number::from(4u8),
        (&Number::from(2u8), &Number::from(1u8)),
    );
}

#[test]
#[should_panic(expected = "ξ must not be a square")]
fn a_tower_needs_a_non_square_xi() {
    let (f, _) = small();
    // Every element of F_p is a square of F_p².
    let _ = Tower::new(
        &f,
        &Number::from(-1),
        (&Number::from(5u8), &Number::from(0u8)),
    );
}

#[test]
#[should_panic(expected = "ξ must not be a cube")]
fn a_tower_needs_a_xi_that_is_not_a_cube() {
    let (f, _) = small();
    let t = Tower::new(
        &f,
        &Number::from(-1),
        (&Number::from(2u8), &Number::from(1u8)),
    );
    // (2 + u)³ is a cube and, as 2 + u is not a square, not a square either.
    let c = Fp2::from_numbers(&t, &Number::from(2u8), &Number::from(1u8)).pow_u64(3);
    let _ = Tower::new(&f, &Number::from(-1), (&c.c0().value(), &c.c1().value()));
}

#[test]
#[should_panic(expected = "elements of different towers")]
fn elements_of_different_towers_do_not_mix() {
    let a = Fp2::one(Tower::bn254());
    let b = Fp2::one(Tower::bls12_381());
    let _ = &a * &b;
}

#[test]
#[should_panic(expected = "division by zero")]
fn division_by_zero_panics() {
    let t = Tower::bls12_381();
    let _ = Fp12::one(t) / Fp12::zero(t);
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn a_coefficient_of_another_field_panics() {
    let t = Tower::bn254();
    let _ = Fp2::new(t, PrimeField::bls12_381_base().one(), t.base().one());
}

#[test]
#[should_panic(expected = "twelve coefficients")]
fn twelve_coefficients_are_needed() {
    let _ = Fp12::from_coefficients(Tower::bn254(), &[Number::from(1u8)]);
}

#[test]
#[should_panic(expected = "0 has no inverse")]
fn a_negative_power_of_zero_panics() {
    let t = Tower::bn254();
    let _ = Fp6::zero(t).pow(&Number::from(-1));
}

/// `a^e mod p` in words.
fn pow_mod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1 % p;
    a %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = r * a % p;
        }
        a = a * a % p;
        e >>= 1;
    }
    r
}

/// `a^e` in `F_p[u]/(u² − β)` in words, for the choice of `ξ` independent of the tower's code.
fn pow2_mod(a: (u64, u64), mut e: u64, beta: u64, p: u64) -> (u64, u64) {
    let mul = |x: (u64, u64), y: (u64, u64)| {
        (
            (x.0 * y.0 + beta * (x.1 * y.1 % p)) % p,
            (x.0 * y.1 + x.1 * y.0) % p,
        )
    };
    let (mut r, mut b) = ((1, 0), a);
    while e > 0 {
        if e & 1 == 1 {
            r = mul(r, b);
        }
        b = mul(b, b);
        e >>= 1;
    }
    r
}

/// Towers over the primes `p ≡ 1 (mod 6)` below 200 with `β` each of the two least non-squares
/// (so `β ≠ −1`, and every tower of a `p ≡ 1 (mod 4)`, where −1 is a square) and `ξ` the first
/// valid one with `ξ₁ = 1` and the first with `ξ₁ ≥ 2`: `(p, β, ξ₀, ξ₁)`, 84 of them.
fn general_towers() -> Vec<(u64, u64, u64, u64)> {
    let primes = [
        7u64, 13, 19, 31, 37, 43, 61, 67, 73, 79, 97, 103, 109, 127, 139, 151, 157, 163, 181, 193,
        199,
    ];
    let mut out = Vec::new();
    for p in primes {
        let betas: Vec<u64> = (2..p)
            .filter(|&b| pow_mod(b, (p - 1) / 2, p) == p - 1)
            .take(2)
            .collect();
        for beta in betas {
            assert_ne!(beta, p - 1);
            // ξ is not a square of F_p² when its norm ξ₀² − βξ₁² is not one of F_p, and not a cube
            // when ξ^((p² − 1)/3) ≠ 1.
            let valid = |x0: u64, x1: u64| {
                let n = (x0 * x0 % p + p - beta * (x1 * x1 % p) % p) % p;
                n != 0
                    && pow_mod(n, (p - 1) / 2, p) == p - 1
                    && pow2_mod((x0, x1), (p * p - 1) / 3, beta, p) != (1, 0)
            };
            let first = |x1: u64| (0..p).find(|&x0| valid(x0, x1));
            out.push((p, beta, first(1).expect("a ξ with ξ₁ = 1"), 1));
            let (x1, x0) = (2..p)
                .find_map(|x1| first(x1).map(|x0| (x1, x0)))
                .expect("a ξ with ξ₁ ≥ 2");
            out.push((p, beta, x0, x1));
        }
    }
    out
}

/// The product in `F_p⁶ = F_p²[v]/(v³ − ξ)` by schoolbook, with the products of `F_p²` and `ξ` as a
/// plain element: no `mul_by_xi`, no Karatsuba.
fn naive6<'t>(a: &Fp6<'t>, b: &Fp6<'t>) -> Fp6<'t> {
    let t = a.tower();
    let (x, y) = ([a.c0(), a.c1(), a.c2()], [b.c0(), b.c1(), b.c2()]);
    let mut c = vec![Fp2::zero(t); 5];
    for i in 0..3 {
        for j in 0..3 {
            c[i + j] = &c[i + j] + &(x[i] * y[j]);
        }
    }
    for k in (3..5).rev() {
        c[k - 3] = &c[k - 3] + &(&c[k] * &t.xi());
    }
    Fp6::new(c[0].clone(), c[1].clone(), c[2].clone())
}

/// The product in `F_p¹² = F_p²[w]/(w⁶ − ξ)` by schoolbook over the powers of `w`.
fn naive12<'t>(a: &Fp12<'t>, b: &Fp12<'t>) -> Fp12<'t> {
    let t = a.tower();
    let w = |x: &Fp12<'t>| {
        [
            x.c0().c0().clone(),
            x.c1().c0().clone(),
            x.c0().c1().clone(),
            x.c1().c1().clone(),
            x.c0().c2().clone(),
            x.c1().c2().clone(),
        ]
    };
    let (x, y) = (w(a), w(b));
    let mut c = vec![Fp2::zero(t); 11];
    for i in 0..6 {
        for j in 0..6 {
            c[i + j] = &c[i + j] + &(&x[i] * &y[j]);
        }
    }
    for k in (6..11).rev() {
        c[k - 6] = &c[k - 6] + &(&c[k] * &t.xi());
    }
    Fp12::new(
        Fp6::new(c[0].clone(), c[2].clone(), c[4].clone()),
        Fp6::new(c[1].clone(), c[3].clone(), c[5].clone()),
    )
}

#[test]
fn towers_with_other_non_residues_are_fields_with_their_frobenius_maps_norms_and_roots() {
    let towers = general_towers();
    assert_eq!(towers.len(), 84);
    assert!(towers.iter().filter(|t| t.3 >= 2).count() == 42);
    let towers = if cfg!(miri) {
        &towers[..2]
    } else {
        &towers[..]
    };
    for &(p, beta, x0, x1) in towers {
        let pn = Number::from(p);
        let f = PrimeField::new(&pn);
        let t = Tower::new(
            &f,
            &Number::from(beta),
            (&Number::from(x0), &Number::from(x1)),
        );
        let mut r = Rng(p * 1000 + beta * 10 + x1);
        let mut e = || f.element(&r.number(1));
        let mut fp2 = || {
            let (c0, c1) = (e(), e());
            Fp2::new(&t, c0, c1)
        };
        let label = format!("p = {p}, β = {beta}, ξ = {x0} + {x1}u");
        // F_p²: against the formulas on F_p.
        let bf = f.from_u64(beta);
        for _ in 0..8 {
            let (a, b) = (fp2(), fp2());
            let want = Fp2::new(
                &t,
                a.c0() * b.c0() + &bf * &(a.c1() * b.c1()),
                a.c0() * b.c1() + a.c1() * b.c0(),
            );
            assert_eq!(&a * &b, want, "{label}");
            assert_eq!(a.square(), &a * &a, "{label}");
            assert_eq!(a.mul_by_xi(), &a * &t.xi(), "{label}");
            assert_eq!(
                a.norm(),
                a.c0().square() - &bf * &a.c1().square(),
                "{label}"
            );
            assert_eq!(a.trace(), a.c0().double(), "{label}");
            assert_eq!(a.conjugate(), a.pow(&pn), "{label}");
            if !a.is_zero() {
                assert!((&a * &a.inverse().unwrap()).is_one(), "{label}");
            }
        }
        // F_p⁶ and F_p¹²: against the schoolbook products, the powers and the conjugates.
        for _ in 0..2 {
            let a6 = Fp6::new(fp2(), fp2(), fp2());
            let b6 = Fp6::new(fp2(), fp2(), fp2());
            let a = Fp12::new(Fp6::new(fp2(), fp2(), fp2()), Fp6::new(fp2(), fp2(), fp2()));
            let b = Fp12::new(Fp6::new(fp2(), fp2(), fp2()), Fp6::new(fp2(), fp2(), fp2()));
            assert_eq!(&a6 * &b6, naive6(&a6, &b6), "{label}");
            assert_eq!(a6.square(), naive6(&a6, &a6), "{label}");
            assert_eq!(&a * &b, naive12(&a, &b), "{label}");
            assert_eq!(a.square(), naive12(&a, &a), "{label}");
            let (c0, c1, c4) = (fp2(), fp2(), fp2());
            let z = Fp2::zero(&t);
            let l014 = Fp12::new(
                Fp6::new(c0.clone(), c1.clone(), z.clone()),
                Fp6::new(z.clone(), c4.clone(), z.clone()),
            );
            assert_eq!(a.mul_by_014(&c0, &c1, &c4), naive12(&a, &l014), "{label}");
            let l034 = Fp12::new(
                Fp6::new(c0.clone(), z.clone(), z.clone()),
                Fp6::new(c1.clone(), c4.clone(), z.clone()),
            );
            assert_eq!(a.mul_by_034(&c0, &c1, &c4), naive12(&a, &l034), "{label}");
            if a.is_zero() || a6.is_zero() {
                continue;
            }
            assert!((&a * &a.inverse().unwrap()).is_one(), "{label}");
            assert!((&a6 * &a6.inverse().unwrap()).is_one(), "{label}");
            let q12 = pn.pow(12);
            assert!(a.pow(&(&q12 - 1u8)).is_one(), "{label}");
            assert_eq!(a.frobenius(1), a.pow(&pn), "{label}");
            assert_eq!(a6.frobenius(1), a6.pow(&pn), "{label}");
            // The conjugates, their product and their sum.
            let mut conj = vec![a.clone()];
            for k in 1..12 {
                let next = conj[k - 1].frobenius(1);
                assert_eq!(a.frobenius(k), next, "{label}, k = {k}");
                conj.push(next);
            }
            assert_eq!(conj[11].frobenius(1), a, "{label}");
            let (mut prod, mut sum) = (Fp12::one(&t), Fp12::zero(&t));
            for c in &conj {
                prod = &prod * c;
                sum = &sum + c;
            }
            let lift = |x| Fp12::from_fp6(Fp6::from_fp2(Fp2::from_base(&t, x)));
            assert_eq!(prod, lift(a.norm().norm().norm()), "{label}");
            assert_eq!(sum, lift(a.trace().trace().trace()), "{label}");
            assert_eq!(Fp12::from_fp6(a.norm()), &a * &a.frobenius(6), "{label}");
            assert_eq!(
                Fp6::from_fp2(a6.norm()),
                &(&a6 * &a6.frobenius(2)) * &a6.frobenius(4),
                "{label}"
            );
            assert_eq!(
                Fp6::from_fp2(a6.trace()),
                &(&a6 + &a6.frobenius(2)) + &a6.frobenius(4),
                "{label}"
            );
            // Squares by Euler's criterion; the cyclotomic subgroup.
            assert_eq!(
                a.is_square(),
                a.pow(&((&q12 - 1u8) / 2u8)).is_one(),
                "{label}"
            );
            let q6 = pn.pow(6);
            assert_eq!(
                a6.is_square(),
                a6.pow(&((&q6 - 1u8) / 2u8)).is_one(),
                "{label}"
            );
            let c = cyclotomic(&a);
            assert!(c.is_cyclotomic(), "{label}");
            assert_eq!(c.cyclotomic_square(), c.square(), "{label}");
            let k = Number::from(p * 7 + 3);
            assert_eq!(c.cyclotomic_pow(&k), c.pow(&k), "{label}");
            assert_eq!(c.cyclotomic_pow(&-&k), c.pow(&-&k), "{label}");
        }
    }
}

#[test]
fn square_roots_are_exact_in_fp2_for_other_non_residues() {
    // Square roots depend on p and β alone: every element of F_p² for p < 50, sampled above.
    let mut seen = std::collections::HashSet::new();
    let towers = general_towers();
    let towers = if cfg!(miri) {
        &towers[..1]
    } else {
        &towers[..]
    };
    for &(p, beta, x0, x1) in towers {
        if !seen.insert((p, beta)) {
            continue;
        }
        let pn = Number::from(p);
        let f = PrimeField::new(&pn);
        let t = Tower::new(
            &f,
            &Number::from(beta),
            (&Number::from(x0), &Number::from(x1)),
        );
        let label = format!("p = {p}, β = {beta}");
        let elements: Vec<(u64, u64)> = if p < 50 {
            (0..p).flat_map(|a| (0..p).map(move |b| (a, b))).collect()
        } else {
            let mut r = Rng(p + beta);
            (0..200).map(|_| (r.next() % p, r.next() % p)).collect()
        };
        // The squares, by brute force for p < 50 and by Euler's criterion above.
        let squares: std::collections::HashSet<(u64, u64)> = if p < 50 {
            elements
                .iter()
                .map(|&(a, b)| {
                    let s = Fp2::new(&t, f.from_u64(a), f.from_u64(b)).square();
                    let v = |x: &Fp<'_>| u64::try_from(&x.value()).unwrap();
                    (v(s.c0()), v(s.c1()))
                })
                .collect()
        } else {
            elements
                .iter()
                .copied()
                .filter(|&(a, b)| {
                    (a, b) == (0, 0) || pow2_mod((a, b), (p * p - 1) / 2, beta, p) == (1, 0)
                })
                .collect()
        };
        let mut found = 0;
        for &(a, b) in &elements {
            let x = Fp2::new(&t, f.from_u64(a), f.from_u64(b));
            let square = squares.contains(&(a, b));
            assert_eq!(x.is_square(), square, "{label}: {x}");
            match x.sqrt() {
                Some(root) => {
                    assert!(square, "{label}: {x}");
                    assert_eq!(root.square(), x, "{label}: {x}");
                    assert!(!larger(&root), "{label}: {x}");
                    found += 1;
                }
                None => assert!(!square, "{label}: {x}"),
            }
        }
        if p < 50 {
            assert_eq!(found as u64, 1 + (p * p - 1) / 2, "{label}");
        }
        // c₁ = 0 with c₀ not a square of F_p: the root is √(c₀/β)·u.
        let n = (2..p)
            .find(|&n| pow_mod(n, (p - 1) / 2, p) == p - 1)
            .unwrap();
        let x = Fp2::from_base(&t, f.from_u64(n));
        let root = x.sqrt().unwrap();
        assert!(root.c0().is_zero(), "{label}");
        assert_eq!(root.square(), x, "{label}");
    }
}

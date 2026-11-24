//! Extension fields: `a::math::field::{ExtensionField, Fpk}` against PARI/GP and FLINT (the
//! values in `data/ext_reference.txt`, written by `data/ext_gen.py`, on which the two agree), the
//! identities of the Frobenius map, the norm and the trace, exhaustive counts in small fields,
//! edge cases and panics.

use a::math::field::{ExtensionField, Fpk, PrimeField};
use a::math::numbers::{ByteOrder, Factorization, Number};
use a::math::poly::ModPolynomial;
use std::collections::HashSet;

/// A deterministic generator of pseudo-random words (SplitMix64), as `ext_gen.py` has it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number of up to `bits` random bits.
    fn number(&mut self, bits: u64) -> Number {
        let mut x = Number::default();
        let mut left = bits;
        while left > 0 {
            let take = left.min(64);
            let w = if take == 64 {
                self.next()
            } else {
                self.next() >> (64 - take)
            };
            x = (x << take as usize) + Number::from(w);
            left -= take;
        }
        x
    }

    /// An element of `f`: `k` coefficients of 16 bits more than `p`, reduced.
    fn element<'f>(&mut self, f: &'f ExtensionField) -> Fpk<'f> {
        let bits = f.characteristic().bits() + 16;
        let c: Vec<Number> = (0..f.degree()).map(|_| self.number(bits)).collect();
        f.element(&c)
    }
}

const REFERENCE: &str = include_str!("data/ext_reference.txt");

fn hex(s: &str) -> Number {
    Number::from_str_radix(s, 16, ByteOrder::LittleEndian).expect("a hexadecimal number")
}

fn hexes(s: &str) -> Vec<Number> {
    s.split(',').map(hex).collect()
}

fn n(v: u64) -> Number {
    Number::from(v)
}

/// Under Miri the tests take only their smallest cases.
fn miri() -> bool {
    cfg!(miri)
}

/// A field of the reference file and its samples.
struct Case {
    name: String,
    p: Number,
    g: Vec<Number>,
    seed: u64,
    samples: usize,
    ebits: u64,
    order: bool,
    sqrt: bool,
    digest: bool,
    expected: Vec<Vec<(String, String)>>,
}

/// A `first` line: `p`, `k` and the coefficients of the polynomial.
type First = (Number, usize, Vec<Number>);

/// The `first` lines and the fields.
fn reference() -> (Vec<First>, Vec<Case>) {
    let mut first = Vec::new();
    let mut cases: Vec<Case> = Vec::new();
    for line in REFERENCE.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first() {
            Some(&"first") => first.push((hex(t[1]), t[2].parse().unwrap(), hexes(t[4]))),
            Some(&"field") => cases.push(Case {
                name: t[1].to_string(),
                p: hex(t[2]),
                g: hexes(t[3]),
                seed: t[4].parse().unwrap(),
                samples: t[5].parse().unwrap(),
                ebits: t[6].parse().unwrap(),
                order: t[7] == "1",
                sqrt: t[8] == "1",
                digest: t[9] == "1",
                expected: Vec::new(),
            }),
            Some(&"s") => {
                let pairs = t[1..]
                    .chunks(2)
                    .map(|w| (w[0].to_string(), w[1].to_string()))
                    .collect();
                cases.last_mut().unwrap().expected.push(pairs);
            }
            _ => {}
        }
    }
    (first, cases)
}

/// The coefficients of `a`, or their digest `Σ cᵢ·Tⁱ mod p` as `ext_gen.py` writes it.
fn encoded(a: &Fpk<'_>, digest: bool) -> Vec<Number> {
    let c: Vec<Number> = a.coeffs().iter().map(|x| x.value()).collect();
    if !digest {
        return c;
    }
    let p = a.field().characteristic();
    let t = n(0x9e37_79b9_7f4a_7c15).rem_euclid(p);
    let (mut d, mut w) = (Number::default(), n(1));
    for v in &c {
        d = (&d + &(v * &w)).rem_euclid(p);
        w = (&w * &t).rem_euclid(p);
    }
    vec![d]
}

/// The field of a case: by `with_degree` when its polynomial is the first irreducible one, by
/// `new` otherwise.
fn field_of(base: &PrimeField, g: &[Number]) -> ExtensionField {
    let k = g.len() - 1;
    let gp = ModPolynomial::new(g, base.modulus());
    let first = ExtensionField::first_irreducible(base, k);
    if first == gp {
        let f = ExtensionField::with_degree(base, k);
        assert_eq!(f.modulus(), &gp);
        f
    } else {
        ExtensionField::new(base, &gp)
    }
}

/// Runs the samples of the cases `keep` selects.
fn check_cases(keep: impl Fn(&Case) -> bool) -> usize {
    let (_, cases) = reference();
    let mut checked = 0;
    let small = |c: &Case| c.p.bits() <= 8 && c.g.len() <= 9;
    for case in cases.iter().filter(|c| keep(c) && (!miri() || small(c))) {
        let base = PrimeField::new(&case.p);
        // The polynomial of the case is the first one where the generator took it so; the BLS12-381
        // field of degree 64 is built from it directly here (its search is in an ignored test).
        let f = if case.name == "bls_64" {
            ExtensionField::new(&base, &ModPolynomial::new(&case.g, &case.p))
        } else {
            field_of(&base, &case.g)
        };
        let k = f.degree();
        let mut rng = Rng(case.seed);
        assert_eq!(case.expected.len(), case.samples, "{}", case.name);
        for exp in case
            .expected
            .iter()
            .take(if miri() { 1 } else { usize::MAX })
        {
            let a = rng.element(&f);
            let b = rng.element(&f);
            let e = rng.number(case.ebits);
            let i = (rng.next() % (2 * k as u64)) as usize;
            let get = |key: &str| &exp.iter().find(|(k, _)| k == key).unwrap().1;
            let elem = |key: &str| {
                if case.digest {
                    vec![hex(get(key))]
                } else {
                    hexes(get(key))
                }
            };
            let what = |op: &str| format!("{} {op} of {a:?}", case.name);
            assert_eq!(
                encoded(&(&a * &b), case.digest),
                elem("mul"),
                "{}",
                what("mul")
            );
            match a.inverse() {
                None => assert_eq!(get("inv"), "-", "{}", what("inv")),
                Some(inv) => assert_eq!(encoded(&inv, case.digest), elem("inv"), "{}", what("inv")),
            }
            assert_eq!(
                encoded(&a.pow(&e), case.digest),
                elem("pow"),
                "{}",
                what("pow")
            );
            assert_eq!(
                encoded(&a.frobenius(i), case.digest),
                elem("frob"),
                "{}",
                what("frob")
            );
            assert_eq!(a.norm().value(), hex(get("norm")), "{}", what("norm"));
            assert_eq!(a.trace().value(), hex(get("tr")), "{}", what("trace"));
            assert_eq!(a.is_square(), get("sq") == "1", "{}", what("is_square"));
            if case.sqrt {
                match a.sqrt() {
                    None => assert_eq!(get("sqrt"), "-", "{}", what("sqrt")),
                    Some(r) => {
                        assert_eq!(encoded(&r, case.digest), elem("sqrt"), "{}", what("sqrt"));
                    }
                }
            }
            if case.order && get("ord") != "-" {
                assert_eq!(
                    a.multiplicative_order().unwrap(),
                    hex(get("ord")),
                    "{}",
                    what("order")
                );
            }
            checked += 1;
        }
    }
    checked
}

/// The heavy cases: large fields whose powers and square roots take seconds in the debug profile.
fn heavy(c: &Case) -> bool {
    c.p.bits() > 128 && c.g.len() > 7
}

#[test]
fn products_inverses_powers_frobenius_norms_traces_and_roots_agree_with_pari_and_flint() {
    let checked = check_cases(|c| !heavy(c));
    assert!(miri() || checked >= 90, "{checked} samples");
}

#[test]
#[ignore = "the fields of degree 12 and 64 over BN254 and BLS12-381: 2 minutes in debug, 2 s in release"]
fn the_large_fields_agree_with_pari_and_flint() {
    let checked = check_cases(heavy);
    assert_eq!(checked, 5);
}

#[test]
fn the_first_irreducible_polynomials_agree_with_a_search_by_pari() {
    let (first, _) = reference();
    let mut count = 0;
    for (p, k, g) in &first {
        // The searches past the binomials for the degrees 12 and 64 over BLS12-381 and BN254 are
        // long in debug: in the ignored test below.
        if p.bits() > 128 && *k >= 12 || miri() && (p.bits() > 3 || *k > 6) {
            continue;
        }
        let base = PrimeField::new(p);
        let found = ExtensionField::first_irreducible(&base, *k);
        assert_eq!(found, ModPolynomial::new(g, p), "p = {p}, k = {k}");
        count += 1;
    }
    assert!(miri() || count >= 125, "{count}");
}

#[test]
#[ignore = "searches of up to 253 candidates of degree 64 over BLS12-381: minutes in debug, 4 s in release"]
fn the_first_irreducible_polynomials_of_large_degree_over_large_primes_agree_with_pari() {
    let (first, _) = reference();
    for (p, k, g) in first.iter().filter(|(p, k, _)| p.bits() > 128 && *k >= 12) {
        let base = PrimeField::new(p);
        let found = ExtensionField::first_irreducible(&base, *k);
        assert_eq!(found, ModPolynomial::new(g, p), "p = {p}, k = {k}");
    }
}

#[test]
fn aes_and_the_binomials_and_trinomials_of_the_first_choice() {
    let two = PrimeField::new(&n(2));
    let aes = ExtensionField::with_degree(&two, 8);
    assert_eq!(
        aes.modulus(),
        &ModPolynomial::from_u64(&[1, 1, 0, 1, 1, 0, 0, 0, 1], 2)
    );
    // FIPS 197, §4.2: {57}·{83} = {c1}.
    let byte = |b: u64| aes.from_u64(&(0..8).map(|i| (b >> i) & 1).collect::<Vec<_>>());
    assert_eq!(&byte(0x57) * &byte(0x83), byte(0xc1));
    // The towers of the pairings start with x² + 1 over BN254 and BLS12-381 (p ≡ 3 mod 4).
    for base in [PrimeField::bn254_base(), PrimeField::bls12_381_base()] {
        let f = ExtensionField::with_degree(base, 2);
        assert_eq!(
            f.modulus(),
            &ModPolynomial::new(&[n(1), n(0), n(1)], base.modulus())
        );
    }
    // Degree 1: x, whose class is 0.
    let f = ExtensionField::with_degree(&PrimeField::new(&n(13)), 1);
    assert_eq!(f.modulus(), &ModPolynomial::from_u64(&[0, 1], 13));
    assert!(f.x().is_zero());
}

/// Fields of every kind for the identities: small and large primes, `p = 2` and `3`, degrees
/// 1 to 40, sparse and dense polynomials, and both ways of the product on each side of the degree
/// where they change, for the sizes of prime where that is cheap in the debug profile (40 for 2
/// and 3, 20 for the other primes of a word and 10 with a dense `g`, 4 beyond nine limbs); the
/// others are in [`product_paths`].
fn fields() -> Vec<ExtensionField> {
    let p = |v: u64| PrimeField::new(&n(v));
    let m = 1_000_000_007;
    let p600 = {
        let mut q = &(n(1) << 599usize) + &n(1);
        while !q.is_probable_prime() {
            q = &q + &n(2);
        }
        PrimeField::new(&q)
    };
    let mut v = vec![
        ExtensionField::with_degree(&p(2), 1),
        ExtensionField::new(&p(2), &ModPolynomial::from_u64(&[1, 1], 2)),
        ExtensionField::with_degree(&p(2), 7),
        ExtensionField::with_degree(&p(2), 39),
        ExtensionField::with_degree(&p(2), 40),
        ExtensionField::with_degree(&p(3), 1),
        ExtensionField::with_degree(&p(3), 4),
        ExtensionField::with_degree(&p(3), 39),
        ExtensionField::with_degree(&p(3), 40),
        ExtensionField::with_degree(&p(5), 6),
        ExtensionField::with_degree(&p(m), 3),
        ExtensionField::with_degree(&p(m), 8),
        ExtensionField::with_degree(&p(m), 19),
        ExtensionField::with_degree(&p(m), 20),
        // PARI's ffinit(10⁹ + 7, 5), x⁵ + x⁴ − 4x³ − 3x² + 3x + 1, with no zero coefficient, and
        // ffinit(10⁹ + 7, 10), Φ₁₁ = x¹⁰ + x⁹ + … + 1: dense, on each side of 10.
        ExtensionField::new(
            &p(m),
            &ModPolynomial::from_u64(&[1, 3, m - 3, m - 4, 1, 1], m),
        ),
        ExtensionField::new(&p(m), &ModPolynomial::from_u64(&[1; 11], m)),
        ExtensionField::with_degree(PrimeField::bn254_base(), 2),
        ExtensionField::with_degree(PrimeField::bls12_381_base(), 3),
        ExtensionField::with_degree(PrimeField::bls12_381_scalar(), 4),
        ExtensionField::with_degree(&p600, 3),
        ExtensionField::with_degree(&p600, 4),
    ];
    // The fields by the first choice of 2 and 3 above have at most four terms below xᵏ, so their
    // products change ways at 40.
    for f in &v[..9] {
        assert!(f.modulus().coeffs().iter().filter(|c| !c.is_zero()).count() <= 5);
    }
    if miri() {
        v.retain(|f| f.characteristic().bits() <= 3 && f.degree() <= 7);
    }
    v
}

/// Fields on each side of the degree where the products change ways, for primes of 2 to 9 limbs
/// (48 for 2 to 8, 28 for 9), whose square roots and conjugates take long in the debug profile.
fn product_paths() -> Vec<ExtensionField> {
    if miri() {
        return Vec::new();
    }
    let p64 = PrimeField::new(&(&(n(1) << 64usize) + &n(13)));
    let p521 = PrimeField::new(&(&(n(1) << 521usize) - &n(1)));
    vec![
        ExtensionField::with_degree(&p64, 47),
        ExtensionField::with_degree(&p64, 48),
        ExtensionField::with_degree(&p521, 27),
        ExtensionField::with_degree(&p521, 28),
    ]
}

#[test]
fn both_ways_of_the_product_around_their_change_for_large_primes() {
    let mut rng = Rng(48);
    for f in &product_paths() {
        for _ in 0..2 {
            let (a, b, c) = (rng.element(f), rng.element(f), rng.element(f));
            assert_eq!(&(&a + &b) * &c, &(&a * &c) + &(&b * &c), "{f:?}");
            assert_eq!(&(&a * &b) * &c, &a * &(&b * &c));
            assert_eq!(&a * &b, &b * &a);
            assert_eq!(a.square(), &a * &a);
            let want = a.to_polynomial().mul_mod(&b.to_polynomial(), f.modulus());
            assert_eq!((&a * &b).to_polynomial(), want);
            assert!((&a * &a.inverse().unwrap()).is_one());
            assert_eq!((&a * &b).frobenius(1), &a.frobenius(1) * &b.frobenius(1));
            assert_eq!((&a * &b).norm(), &a.norm() * &b.norm());
            assert_eq!((&a + &b).trace(), &a.trace() + &b.trace());
        }
    }
}

#[test]
fn the_field_axioms_and_the_operators() {
    let mut rng = Rng(24);
    for f in &fields() {
        for _ in 0..4 {
            let (a, b, c) = (rng.element(f), rng.element(f), rng.element(f));
            assert_eq!(&(&a + &b) * &c, &(&a * &c) + &(&b * &c), "{f:?}");
            assert_eq!(&(&a * &b) * &c, &a * &(&b * &c));
            assert_eq!(&a * &b, &b * &a);
            assert_eq!(&(&a - &b) + &b, a);
            assert_eq!(&a + &(-&a), f.zero());
            assert_eq!(a.square(), &a * &a);
            assert_eq!(a.double(), &a + &a);
            assert_eq!(&a * &f.one(), a);
            if !b.is_zero() {
                assert_eq!(&(&a / &b) * &b, a);
                assert!((&b * &b.inverse().unwrap()).is_one());
            }
            // Owned operands and the assignments.
            let mut d = a.clone();
            d += &b;
            d -= b.clone();
            d *= &c;
            assert_eq!(d, &a * &c);
            if !c.is_zero() {
                d /= c.clone();
                assert_eq!(d, a);
            }
            assert_eq!(a.clone() + b.clone(), &a + &b);
            assert_eq!(a.clone() * &b, &a * &b);
            assert_eq!(-a.clone(), -&a);
        }
    }
}

#[test]
fn products_agree_with_the_products_of_polynomials_modulo_g() {
    let mut rng = Rng(5);
    for f in &fields() {
        for _ in 0..3 {
            let (a, b) = (rng.element(f), rng.element(f));
            let want = a.to_polynomial().mul_mod(&b.to_polynomial(), f.modulus());
            assert_eq!((&a * &b).to_polynomial(), want, "{f:?}");
        }
    }
}

#[test]
fn frobenius_is_an_automorphism_of_order_k_fixing_the_base_field() {
    let mut rng = Rng(9);
    for f in &fields() {
        let k = f.degree();
        for _ in 0..3 {
            let (a, b) = (rng.element(f), rng.element(f));
            assert_eq!(a.frobenius(k), a, "{f:?}");
            assert_eq!(a.frobenius(0), a);
            assert_eq!(a.frobenius(k + 1), a.frobenius(1));
            assert_eq!((&a * &b).frobenius(1), &a.frobenius(1) * &b.frobenius(1));
            assert_eq!((&a + &b).frobenius(1), &a.frobenius(1) + &b.frobenius(1));
            for i in 0..k.min(5) {
                assert_eq!(a.frobenius(i).frobenius(1), a.frobenius(i + 1));
            }
            if f.characteristic().bits() < 64 {
                assert_eq!(a.frobenius(1), a.pow(f.characteristic()));
            }
            let c = f.from_base(&f.base().from_u64(rng.next()));
            assert_eq!(c.frobenius(1), c);
        }
    }
}

#[test]
fn norms_and_traces_are_the_product_and_the_sum_of_the_conjugates() {
    let mut rng = Rng(11);
    for f in &fields() {
        let k = f.degree();
        for _ in 0..3 {
            let (a, b) = (rng.element(f), rng.element(f));
            // The conjugates a, aᵖ, … one Frobenius step at a time (frobenius(i) itself takes i
            // steps).
            let mut conj: Vec<Fpk<'_>> = vec![a.clone()];
            for i in 1..k {
                let next = conj[i - 1].frobenius(1);
                conj.push(next);
            }
            let prod = conj.iter().fold(f.one(), |s, c| s * c);
            let sum = conj.iter().fold(f.zero(), |s, c| s + c);
            assert_eq!(prod.to_base().unwrap(), a.norm(), "{f:?}");
            assert_eq!(sum.to_base().unwrap(), a.trace());
            assert_eq!((&a * &b).norm(), &a.norm() * &b.norm());
            assert_eq!((&a + &b).trace(), &a.trace() + &b.trace());
            let c = f.base().from_u64(rng.next());
            assert_eq!((&f.from_base(&c) * &a).trace(), &c * &a.trace());
            assert_eq!(f.from_base(&c).norm(), c.pow_u64(k as u64));
            assert_eq!(f.from_base(&c).trace(), &c * &f.base().from_u64(k as u64));
            assert_eq!(a.frobenius(1).norm(), a.norm());
        }
        assert!(f.zero().norm().is_zero() && f.zero().trace().is_zero());
        assert!(f.one().norm().is_one());
    }
}

#[test]
fn powers_fermat_and_negative_exponents() {
    let mut rng = Rng(13);
    for f in &fields() {
        let units = f.order() - &n(1);
        for _ in 0..2 {
            let a = rng.element(f);
            if a.is_zero() {
                continue;
            }
            if f.order().bits() <= 1200 {
                assert_eq!(a.pow(f.order()), a, "{f:?}");
                assert!(a.pow(&units).is_one());
            }
            assert_eq!(a.pow(&Number::from(-3)), a.inverse().unwrap().pow_u64(3));
            assert_eq!(a.pow_u64(0), f.one());
            assert_eq!(a.pow_u64(1), a);
            assert_eq!(a.pow_u64(5), &a.square().square() * &a);
            // An exponent beyond the order of the group: reduced, the same power.
            let e = &(&units * &n(7)) + &n(4);
            assert_eq!(a.pow(&e), a.pow_u64(4));
            assert_eq!(a.pow(&units.clone()), f.one());
        }
        assert_eq!(f.zero().pow_u64(0), f.one());
        assert!(f.zero().pow_u64(5).is_zero());
    }
}

#[test]
fn square_roots_and_the_quadratic_character() {
    let mut rng = Rng(17);
    for f in &fields() {
        let p = f.characteristic().clone();
        let mut nonsquares = 0;
        for _ in 0..6 {
            let a = rng.element(f);
            let s = a.square();
            let r = s.sqrt().expect("a square has a root");
            assert!(r == a || r == -&a, "{f:?}");
            // The root whose highest nonzero coefficient is below p/2.
            if let Some(top) = r.coeffs().iter().rev().find(|c| !c.is_zero()) {
                assert!(p == 2u8 || (top.value() << 1usize) < p);
            }
            assert!(s.is_square());
            match a.sqrt() {
                Some(r) => {
                    assert_eq!(r.square(), a);
                    assert!(a.legendre() >= 0);
                }
                None => {
                    assert_eq!(a.legendre(), -1);
                    assert!(!a.is_square());
                    nonsquares += 1;
                }
            }
            // The character is multiplicative.
            let b = rng.element(f);
            assert_eq!((&a * &b).legendre(), a.legendre() * b.legendre());
        }
        if p == 2u8 {
            assert_eq!(nonsquares, 0);
        }
        assert!(f.zero().sqrt().unwrap().is_zero());
        assert_eq!(f.one().sqrt().unwrap(), f.one());
    }
}

#[test]
fn square_roots_of_constants_in_even_and_odd_degrees() {
    // A non-square modulo p is a square in every extension of even degree, and in none of odd.
    let base = PrimeField::new(&n(1_000_000_007));
    let c = base.from_u64(5); // 5 is not a square modulo 10⁹ + 7
    assert_eq!(c.legendre(), -1);
    for k in 1..=6 {
        let f = ExtensionField::with_degree(&base, k);
        let e = f.from_base(&c);
        match e.sqrt() {
            Some(r) => {
                assert_eq!(k % 2, 0);
                assert_eq!(r.square(), e);
                assert!(r.to_base().is_none());
            }
            None => assert_eq!(k % 2, 1),
        }
        // A square constant: the root of GF(p), the smaller one.
        let s = f.from_base(&base.from_u64(4));
        assert_eq!(s.sqrt().unwrap(), f.from_u64(&[2]));
    }
    // Degree 1 agrees with Fp::sqrt.
    let f = ExtensionField::with_degree(&base, 1);
    let mut rng = Rng(3);
    for _ in 0..20 {
        let x = base.element(&rng.number(40));
        let e = f.from_base(&x);
        assert_eq!(e.sqrt().map(|r| r.coeffs()[0].clone()), x.sqrt());
    }
}

/// Every element of `f`, by the digits of `0..q` in base `p`.
fn all(f: &ExtensionField) -> Vec<Fpk<'_>> {
    let p = u64::try_from(f.characteristic()).unwrap();
    let q = u64::try_from(f.order()).unwrap();
    (0..q)
        .map(|mut v| {
            let c: Vec<u64> = (0..f.degree())
                .map(|_| {
                    let d = v % p;
                    v /= p;
                    d
                })
                .collect();
            f.from_u64(&c)
        })
        .collect()
}

#[test]
// Fpk hashes its coefficients only; the tables its field computes on first use, in OnceLocks,
// take no part, so the lint's warning about them is a false positive.
#[allow(clippy::mutable_key_type)]
fn exhaustively_in_small_fields() {
    let sizes: &[(u64, usize)] = if miri() {
        &[(2, 2), (3, 2)]
    } else {
        &[
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 1),
            (3, 2),
            (3, 3),
            (5, 2),
            (7, 2),
            (5, 3),
            (2, 8),
        ]
    };
    for &(p, k) in sizes {
        let base = PrimeField::new(&n(p));
        let f = ExtensionField::with_degree(&base, k);
        let q = u64::try_from(f.order()).unwrap();
        let elems = all(&f);
        let set: HashSet<Fpk<'_>> = elems.iter().cloned().collect();
        assert_eq!(set.len() as u64, q, "GF({p}^{k})");
        // Squares: (q + 1)/2 for odd q, all for even; each has its roots.
        let squares: HashSet<Fpk<'_>> = elems.iter().map(Fpk::square).collect();
        let want = if p == 2 { q } else { q.div_ceil(2) };
        assert_eq!(squares.len() as u64, want);
        for a in &elems {
            assert_eq!(a.is_square(), squares.contains(a));
            // A root exactly for the squares, and the documented one: the root whose top nonzero
            // coefficient is below p/2 (the only root for p = 2).
            let r = a.sqrt();
            assert_eq!(
                r.is_some(),
                squares.contains(a),
                "GF({p}^{k}): sqrt of {a:?}"
            );
            if let Some(r) = r {
                assert_eq!(&r.square(), a);
                if let Some(top) = r.coeffs().iter().rev().find(|c| !c.is_zero()) {
                    assert!(p == 2 || 2 * u64::try_from(&top.value()).unwrap() < p);
                }
            }
            assert_eq!(a.pow(f.order()), *a);
            assert_eq!(a.frobenius(k), *a);
        }
        // The norm and the trace are onto GF(p): each nonzero norm is taken (q − 1)/(p − 1)
        // times, each trace q/p times.
        for v in 0..p {
            let norms = elems
                .iter()
                .filter(|a| a.norm() == base.from_u64(v))
                .count() as u64;
            let traces = elems
                .iter()
                .filter(|a| a.trace() == base.from_u64(v))
                .count() as u64;
            assert_eq!(norms, if v == 0 { 1 } else { (q - 1) / (p - 1) });
            assert_eq!(traces, q / p);
        }
        // Orders: φ(q − 1) primitive elements, the order divides q − 1, and the first one.
        let mut primitive = 0;
        for a in elems.iter().filter(|a| !a.is_zero()) {
            let o = u64::try_from(&a.multiplicative_order().unwrap()).unwrap();
            assert_eq!((q - 1) % o, 0);
            assert!(a.pow_u64(o).is_one());
            assert!(
                (1..o)
                    .filter(|d| o % d == 0)
                    .all(|d| !a.pow_u64(d).is_one())
            );
            assert_eq!(a.is_primitive(), o == q - 1);
            primitive += u64::from(a.is_primitive());
        }
        let phi = (1..q).filter(|&d| gcd(d, q - 1) == 1).count() as u64;
        assert_eq!(primitive, phi, "GF({p}^{k})");
        let g = f.primitive_element();
        assert!(g.is_primitive());
        // Inverses of all the nonzero elements at once.
        let inv = f.batch_invert(&elems);
        for (a, b) in elems.iter().zip(&inv) {
            assert!(if a.is_zero() {
                b.is_zero()
            } else {
                (a * b).is_one()
            });
        }
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[test]
fn the_first_irreducible_polynomial_is_the_smallest_irreducible_by_brute_force() {
    // Every smaller monic polynomial of the degree has a root or a factor of lower degree, by
    // trial division by all the monic polynomials of degree up to k/2.
    let sizes: &[(u64, usize)] = if miri() {
        &[(2, 4), (3, 3)]
    } else {
        &[
            (2, 6),
            (3, 4),
            (5, 3),
            (5, 4),
            (7, 3),
            (11, 2),
            (13, 4),
            (2, 12),
        ]
    };
    for &(p, k) in sizes {
        let base = PrimeField::new(&n(p));
        let g = ExtensionField::first_irreducible(&base, k);
        let poly = |mut v: u64, deg: usize| {
            let mut c: Vec<u64> = (0..deg)
                .map(|_| {
                    let d = v % p;
                    v /= p;
                    d
                })
                .collect();
            c.push(1);
            ModPolynomial::from_u64(&c, p)
        };
        let reducible = |f: &ModPolynomial| {
            (1..=k / 2).any(|d| (0..p.pow(d as u32)).any(|v| (f % &poly(v, d)).is_zero()))
        };
        let index = (0..p.pow(k as u32)).find(|&v| poly(v, k) == g).unwrap();
        assert!(!reducible(&g), "{g}");
        for v in 0..index {
            assert!(reducible(&poly(v, k)), "{} is irreducible", poly(v, k));
        }
    }
}

#[test]
fn degree_one_is_the_prime_field() {
    let base = PrimeField::new(&n(101));
    for g in [
        ModPolynomial::from_u64(&[0, 1], 101),
        ModPolynomial::from_u64(&[98, 1], 101),
    ] {
        let f = ExtensionField::new(&base, &g);
        assert_eq!(f.degree(), 1);
        assert_eq!(f.order(), &n(101));
        // x is the root of g: 0, or 3 for x − 3.
        assert_eq!(
            f.x().coeffs()[0].value(),
            (n(101) - g.coeff(0)).rem_euclid(&n(101))
        );
        let mut rng = Rng(2);
        for _ in 0..20 {
            let (x, y) = (base.element(&rng.number(20)), base.element(&rng.number(20)));
            let (a, b) = (f.from_base(&x), f.from_base(&y));
            assert_eq!((&a * &b).to_base().unwrap(), &x * &y);
            assert_eq!(a.norm(), x);
            assert_eq!(a.trace(), x);
            assert_eq!(a.frobenius(1), a);
            assert_eq!(a.legendre(), x.legendre());
            if !x.is_zero() {
                assert_eq!(
                    a.inverse().unwrap().to_base().unwrap(),
                    x.inverse().unwrap()
                );
            }
        }
    }
}

#[test]
fn elements_construction_and_conversion() {
    let base = PrimeField::new(&n(7));
    let f = ExtensionField::with_degree(&base, 3); // x³ + 2
    assert_eq!(f.modulus(), &ModPolynomial::from_u64(&[2, 0, 0, 1], 7));
    assert_eq!(f.characteristic(), &n(7));
    assert_eq!(f.base().modulus(), &n(7));
    // Coefficients of any sign, size and number: x⁴ = x·x³ ≡ −2x.
    let a = f.element(&[Number::from(-1), n(15), n(0), n(0), n(1)]);
    assert_eq!(a, f.from_u64(&[6, 1 + 5]));
    assert_eq!(a.coeffs().len(), 3);
    assert_eq!(
        f.from_polynomial(&ModPolynomial::from_u64(&[6, 1, 0, 0, 1], 7)),
        f.from_u64(&[6, 6])
    );
    let other = PrimeField::new(&n(7));
    let c = other.from_u64(3);
    assert_eq!(f.from_base(&c), f.from_u64(&[3]));
    assert_eq!(
        f.from_fp(&[c.clone(), other.from_u64(1)]),
        f.from_u64(&[3, 1])
    );
    assert_eq!(f.from_u64(&[3]).to_base().unwrap(), c);
    assert!(f.x().to_base().is_none());
    assert_eq!(f.x().to_polynomial(), ModPolynomial::from_u64(&[0, 1], 7));
    assert!(f.one().is_one() && !f.x().is_one() && f.zero().is_zero());
    assert!(f.zero().inverse().is_none());
    assert_eq!(f.zero().multiplicative_order(), None);
    assert!(!f.zero().is_primitive());
    assert_eq!(f.one().multiplicative_order(), Some(n(1)));
    // Text: as PARI writes polynomials.
    assert_eq!(f.from_u64(&[1, 0, 3]).to_string(), "3*x^2 + 1");
    assert_eq!(
        format!("{:?}", f.from_u64(&[1, 0, 3])),
        "3*x^2 + 1 (mod 7, x^3 + 2)"
    );
    assert_eq!(f.zero().to_string(), "0");
    // Two equal fields built apart: their elements mix and compare equal.
    let g = ExtensionField::new(&base, f.modulus());
    assert_eq!(f, g);
    assert_eq!(&f.x() + &g.x(), f.from_u64(&[0, 2]));
    assert_eq!(f.x(), g.x());
    assert_ne!(f, ExtensionField::with_degree(&base, 2));
}

#[test]
fn orders_and_primitive_elements_with_a_given_factorisation() {
    // GF(2⁶⁴): 2⁶⁴ − 1 = 3·5·17·257·641·65537·6700417, factored by the cyclotomic values.
    let f = ExtensionField::with_degree(&PrimeField::new(&n(2)), 64);
    let primes: Vec<u64> = f
        .factored_unit_order()
        .factors()
        .iter()
        .map(|(r, _)| u64::try_from(r).unwrap())
        .collect();
    assert_eq!(primes, [3, 5, 17, 257, 641, 65537, 6_700_417]);
    let g = f.primitive_element();
    assert!(g.is_primitive());
    assert_eq!(g.multiplicative_order().unwrap(), f.order() - &n(1));
    assert_eq!(
        g.pow_u64(641).multiplicative_order().unwrap(),
        (f.order() - &n(1)) / n(641)
    );
    // The first primitive element in the order of the coefficients is not bounded by the average
    // (q − 1)/φ(q − 1): in GF(37³) by x³ + 2 it is 2x + 1, the 39th candidate from x (PARI's
    // fforder on each in turn), where the average is 3.55.
    let f37 = ExtensionField::with_degree(&PrimeField::new(&n(37)), 3);
    assert_eq!(f37.modulus(), &ModPolynomial::from_u64(&[2, 0, 0, 1], 37));
    assert_eq!(f37.primitive_element(), f37.from_u64(&[1, 2]));
    // GF(p²) for p = 2¹²⁷ − 1 with the factorisation of p² − 1 given, from PARI 2.19's
    // `factor`; PARI's `fforder` gives 1016 for x + 1 ((1 + x)⁸ = 16 and 2 has the order 127).
    let p = &(n(1) << 127usize) - &n(1);
    let factors = Factorization::new(
        M127_P2_MINUS_1.iter().map(|(r, e)| (hex(r), *e)).collect(),
        ByteOrder::LittleEndian,
    );
    let f2 = ExtensionField::with_degree(&PrimeField::new(&p), 2).with_factored_unit_order(factors);
    assert_eq!(f2.modulus(), &ModPolynomial::new(&[n(1), n(0), n(1)], &p));
    assert_eq!(f2.x().multiplicative_order(), Some(n(4)));
    assert_eq!(f2.from_u64(&[1, 1]).multiplicative_order(), Some(n(1016)));
    assert!(!f2.x().is_primitive());
    let g = f2.primitive_element();
    assert_eq!(g.multiplicative_order().unwrap(), f2.order() - &n(1));
}

/// The factorisation of `p² − 1` for `p = 2¹²⁷ − 1`, from PARI 2.19's `factor`.
const M127_P2_MINUS_1: &[(&str, u32)] = &[
    ("2", 128),
    ("3", 3),
    ("7", 2),
    ("13", 1),
    ("2b", 1),
    ("49", 1),
    ("7f", 1),
    ("151", 1),
    ("152b", 1),
    ("16a41", 1),
    ("9e9b9", 1),
    ("11f703ee09", 1),
];

#[test]
#[should_panic(expected = "is reducible")]
fn a_reducible_polynomial_panics() {
    // x² + 1 = (x + 2)(x + 3) modulo 5.
    let _ = ExtensionField::new(
        &PrimeField::new(&n(5)),
        &ModPolynomial::from_u64(&[1, 0, 1], 5),
    );
}

#[test]
#[should_panic(expected = "must be monic")]
fn a_polynomial_that_is_not_monic_panics() {
    let _ = ExtensionField::new(
        &PrimeField::new(&n(7)),
        &ModPolynomial::from_u64(&[1, 0, 2], 7),
    );
}

#[test]
#[should_panic(expected = "degree of at least 1")]
fn a_constant_polynomial_panics() {
    let _ = ExtensionField::new(&PrimeField::new(&n(7)), &ModPolynomial::from_u64(&[3], 7));
}

#[test]
#[should_panic(expected = "not modulo p")]
fn a_polynomial_modulo_another_prime_panics() {
    let _ = ExtensionField::new(
        &PrimeField::new(&n(7)),
        &ModPolynomial::from_u64(&[1, 0, 1], 11),
    );
}

#[test]
#[should_panic(expected = "the degree must be at least 1")]
fn degree_zero_panics() {
    let _ = ExtensionField::with_degree(&PrimeField::new(&n(7)), 0);
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn elements_of_different_fields_panic() {
    let base = PrimeField::new(&n(7));
    let (f, g) = (
        ExtensionField::with_degree(&base, 2),
        ExtensionField::with_degree(&base, 3),
    );
    let _ = &f.one() * &g.one();
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn sums_of_fields_with_other_polynomials_panic() {
    let base = PrimeField::new(&n(7));
    let f = ExtensionField::new(&base, &ModPolynomial::from_u64(&[1, 0, 1], 7));
    let g = ExtensionField::new(&base, &ModPolynomial::from_u64(&[3, 1, 1], 7));
    let _ = &f.one() + &g.one();
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn a_coefficient_of_another_prime_panics() {
    let f = ExtensionField::with_degree(&PrimeField::new(&n(7)), 2);
    let _ = f.from_base(&PrimeField::new(&n(11)).one());
}

#[test]
#[should_panic(expected = "a polynomial modulo 11 for a field of characteristic 7")]
fn an_element_of_a_polynomial_modulo_another_prime_panics() {
    let f = ExtensionField::with_degree(&PrimeField::new(&n(7)), 2);
    let _ = f.from_polynomial(&ModPolynomial::from_u64(&[1, 1], 11));
}

#[test]
#[should_panic(expected = "division by zero")]
fn division_by_zero_panics() {
    let f = ExtensionField::with_degree(&PrimeField::new(&n(3)), 4);
    let _ = &f.one() / &f.zero();
}

#[test]
#[should_panic(expected = "0 has no inverse")]
fn a_negative_power_of_zero_panics() {
    let f = ExtensionField::with_degree(&PrimeField::new(&n(3)), 4);
    let _ = f.zero().pow(&Number::from(-1));
}

#[test]
#[should_panic(expected = "not to p^k - 1")]
fn a_wrong_factorisation_panics() {
    let f = ExtensionField::with_degree(&PrimeField::new(&n(3)), 2);
    // 3² − 1 = 8 = 2³, not 2².
    let _ =
        f.with_factored_unit_order(Factorization::new(vec![(n(2), 2)], ByteOrder::LittleEndian));
}

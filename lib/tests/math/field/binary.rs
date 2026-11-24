//! Binary fields: `a::math::field::{BinaryField, F2m, Clmul}` against NTL's `GF2E` and OpenSSL's
//! `BN_GF2m_*` (data/binary_ntl.txt, from data/gen_binary.cpp), the carry-less kernels against
//! each other and against a product bit by bit, and the irreducibility test against NTL and
//! Gauss's count of the irreducible polynomials.

use a::math::field::{BinaryField, Clmul, F2m};

/// splitmix64, as data/gen_binary.cpp draws its inputs.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// FNV-1a over the little-endian bytes of words, the digest of data/gen_binary.cpp.
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn word(&mut self, w: u64) {
        for b in w.to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn words(&mut self, w: &[u64]) {
        for &x in w {
            self.word(x);
        }
    }
}

/// One field of the data: its exponents, the number of cases and the digests by operation.
struct Case {
    exponents: Vec<u32>,
    cases: usize,
    digests: Vec<(String, u64)>,
}

/// The fields of data/binary_ntl.txt, and the irreducibility bits of its last line.
fn data() -> (Vec<Case>, Vec<bool>) {
    let mut fields = Vec::new();
    let mut bits = Vec::new();
    for line in include_str!("data/binary_ntl.txt").lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let head = parts.next().unwrap();
        if head == "irreducible" {
            let count: usize = parts.next().unwrap().parse().unwrap();
            let hex = parts.next().unwrap();
            for i in 0..count {
                let nibble = u8::from_str_radix(&hex[i / 4..i / 4 + 1], 16).unwrap();
                bits.push(nibble >> (i % 4) & 1 == 1);
            }
            continue;
        }
        let exponents = head.split(',').map(|e| e.parse().unwrap()).collect();
        let cases = parts.next().unwrap().parse().unwrap();
        let digests = parts
            .map(|p| {
                let (op, d) = p.split_once('=').unwrap();
                let radix = if op == "openssl" { 10 } else { 16 };
                (op.to_string(), u64::from_str_radix(d, radix).unwrap())
            })
            .collect();
        fields.push(Case {
            exponents,
            cases,
            digests,
        });
    }
    (fields, bits)
}

/// The digests of data/gen_binary.cpp's operations, computed in `f`.
fn digests(f: &BinaryField, exponents: &[u32], cases: usize) -> Vec<(String, u64)> {
    let m = f.degree() as usize;
    let n = m.div_ceil(64);
    let mut seed = 0u64;
    for &e in exponents {
        seed = seed.wrapping_mul(1_000_003).wrapping_add(u64::from(e) + 1);
    }
    let mut rng = SplitMix(seed);
    let elem = |rng: &mut SplitMix| {
        let mut w: Vec<u64> = (0..n).map(|_| rng.next()).collect();
        if !m.is_multiple_of(64) {
            w[n - 1] &= (1 << (m % 64)) - 1;
        }
        w
    };
    let names = [
        "mul", "sqr", "inv", "div", "sqrt", "pow", "tr", "half", "long",
    ];
    let mut d: Vec<Fnv> = names.iter().map(|_| Fnv::new()).collect();
    let zero = vec![0u64; n];
    for _ in 0..cases {
        let a = f.element(&elem(&mut rng));
        let b = f.element(&elem(&mut rng));
        let e = rng.next();
        let long: Vec<u64> = (0..3 * n + 1).map(|_| rng.next()).collect();
        d[0].words((&a * &b).words());
        d[1].words(a.square().words());
        d[2].words(a.inverse().as_ref().map_or(&zero[..], F2m::words));
        if b.is_zero() {
            d[3].words(&zero);
        } else {
            d[3].words((&a / &b).words());
        }
        d[4].words(a.sqrt().words());
        d[5].words(a.pow_u64(e).words());
        d[6].word(u64::from(a.trace()));
        if m % 2 == 1 {
            d[7].words(a.half_trace().words());
        }
        d[8].words(f.element(&long).words());
    }
    names
        .iter()
        .zip(d)
        .filter(|(name, _)| m % 2 == 1 || **name != "half")
        .map(|(name, d)| (name.to_string(), d.0))
        .collect()
}

/// Checks every field of the data whose degree is at most `max`, its products by `clmul`.
fn check_against_ntl(clmul: Clmul, max: u32) -> usize {
    let (fields, _) = data();
    let mut checked = 0;
    for case in fields.iter().filter(|c| c.exponents[0] <= max) {
        let f = BinaryField::with_clmul(&case.exponents, clmul);
        let got = digests(&f, &case.exponents, case.cases);
        for (op, want) in case.digests.iter().filter(|(op, _)| op != "openssl") {
            let (_, have) = got.iter().find(|(o, _)| o == op).unwrap();
            assert_eq!(have, want, "{op} in {:?} by {clmul:?}", case.exponents);
            checked += case.cases;
        }
    }
    checked
}

#[test]
fn products_squares_inverses_roots_and_traces_agree_with_ntl_and_openssl() {
    // 83 fields: SEC 2's, AES's, GCM's, reciprocals, one-word fields with terms close to xᵐ,
    // NTL's sparse and random ones of degrees 1 to 2000; 9 operations, 8 or 24 cases each.
    // Miri runs the fields of degree up to 16 only.
    let max = if cfg!(miri) { 16 } else { u32::MAX };
    let checked = check_against_ntl(Clmul::detect(), max);
    assert!(cfg!(miri) || checked > 10_000, "{checked}");
}

#[test]
fn the_portable_kernel_agrees_with_ntl_and_openssl() {
    let max = if cfg!(miri) { 16 } else { 640 };
    let checked = check_against_ntl(Clmul::Portable, max);
    assert!(cfg!(miri) || checked > 8_000, "{checked}");
}

/// `a·b` bit by bit.
fn naive_clmul(a: u64, b: u64) -> u128 {
    (0..64)
        .filter(|i| b >> i & 1 == 1)
        .fold(0u128, |r, i| r ^ (u128::from(a) << i))
}

#[test]
fn the_cpus_instruction_is_detected() {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    assert_eq!(
        Clmul::detect() == Clmul::Pclmulqdq,
        std::arch::is_x86_feature_detected!("pclmulqdq")
    );
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    assert_eq!(
        Clmul::detect() == Clmul::Pmull,
        std::arch::is_aarch64_feature_detected!("aes")
    );
    #[cfg(miri)]
    assert_eq!(Clmul::detect(), Clmul::Portable);
    assert!(Clmul::detect().is_available());
}

#[test]
fn the_kernels_agree_with_a_product_bit_by_bit() {
    let kernels: Vec<Clmul> = [Clmul::Portable, Clmul::Pmull, Clmul::Pclmulqdq]
        .into_iter()
        .filter(|k| k.is_available())
        .collect();
    assert!(kernels.contains(&Clmul::detect()));
    let mut rng = SplitMix(1);
    let edges = [
        0,
        1,
        2,
        u64::MAX,
        1 << 63,
        0x8000_0000_0000_0001,
        0x5555_5555_5555_5555,
    ];
    let mut pairs: Vec<(u64, u64)> = edges
        .iter()
        .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
        .collect();
    pairs.extend((0..2000).map(|_| (rng.next(), rng.next())));
    for (a, b) in pairs {
        let want = naive_clmul(a, b);
        for &k in &kernels {
            assert_eq!(k.mul(a, b), want, "{k:?}: {a:#x}·{b:#x}");
        }
    }
}

/// `a·b mod f` bit by bit, for the words of two reduced elements.
fn naive_mul_mod(a: &[u64], b: &[u64], exponents: &[u32]) -> Vec<u64> {
    let n = a.len();
    let mut p = vec![0u64; 2 * n + 1];
    for i in 0..64 * n {
        if a[i / 64] >> (i % 64) & 1 == 1 {
            for j in 0..64 * n {
                if b[j / 64] >> (j % 64) & 1 == 1 {
                    p[(i + j) / 64] ^= 1 << ((i + j) % 64);
                }
            }
        }
    }
    let m = exponents[0] as usize;
    for i in (m..128 * n).rev() {
        if p[i / 64] >> (i % 64) & 1 == 1 {
            for &e in exponents {
                let j = i - m + e as usize;
                p[j / 64] ^= 1 << (j % 64);
            }
        }
    }
    p.truncate(n);
    p
}

#[test]
fn products_of_many_words_agree_with_a_product_bit_by_bit() {
    // Karatsuba's splits at every length from 1 to 12 words, by both kernels, and both reductions
    // (the trinomials and pentanomials are sparse; the moduli with terms near xᵐ are Barrett's).
    let mut rng = SplitMix(2);
    let moduli: Vec<Vec<u32>> = vec![
        vec![7, 1, 0],
        vec![64, 4, 3, 1, 0],
        vec![127, 1, 0],
        vec![127, 126, 0],
        vec![163, 7, 6, 3, 0],
        vec![163, 160, 157, 156, 0],
        vec![193, 15, 0],
        vec![255, 82, 0],
        vec![257, 12, 0],
        vec![383, 90, 0],
        vec![409, 322, 0],
        vec![511, 10, 0],
        vec![571, 569, 566, 561, 0],
        vec![600, 9, 5, 2, 0],
        vec![641, 11, 0],
        vec![767, 168, 0],
    ];
    let cases = if cfg!(miri) { 1 } else { 3 };
    for e in moduli
        .iter()
        .take(if cfg!(miri) { 3 } else { moduli.len() })
    {
        assert!(BinaryField::is_irreducible(e), "{e:?}");
        for clmul in [Clmul::detect(), Clmul::Portable] {
            let f = BinaryField::with_clmul(e, clmul);
            let n = (e[0] as usize).div_ceil(64);
            for _ in 0..cases {
                let w: Vec<u64> = (0..2 * n).map(|_| rng.next()).collect();
                let (a, b) = (f.element(&w[..n]), f.element(&w[n..]));
                assert_eq!(
                    (&a * &b).words(),
                    naive_mul_mod(a.words(), b.words(), e),
                    "{e:?} by {clmul:?}"
                );
            }
        }
    }
}

#[test]
fn each_modulus_chooses_its_reduction() {
    let name = |e: &[u32], k: Clmul| format!("{:?}", BinaryField::with_clmul(e, k));
    let dense64 = [
        64, 63, 62, 60, 59, 58, 56, 55, 53, 52, 47, 46, 43, 42, 41, 35, 34, 33, 32, 26, 24, 23, 22,
        21, 20, 18, 16, 13, 12, 11, 10, 9, 8, 2, 0,
    ];
    let mut kernels = vec![Clmul::Portable];
    if Clmul::detect() != Clmul::Portable {
        kernels.push(Clmul::detect());
    }
    for &k in &kernels {
        // The standards' trinomials and pentanomials: the sparse reduction.
        for e in [
            &[163, 7, 6, 3, 0][..],
            &[571, 10, 5, 2, 0],
            &[233, 74, 0],
            &[239, 203, 0],
        ] {
            assert!(name(e, k).contains("Sparse"), "{e:?} by {k:?}");
        }
        assert!(name(&[2, 1, 0], k).contains("Sparse"));
        // Terms close to xᵐ, or many: Barrett's, of one word as of several.
        for e in [
            &[127, 126, 0][..],
            &[163, 160, 157, 156, 0],
            &[64, 63, 37, 33, 0],
            &[63, 62, 0],
            &[31, 28, 0],
            &[8, 7, 6, 5, 4, 2, 0],
            &dense64,
        ] {
            assert!(name(e, k).contains("Barrett"), "{e:?} by {k:?}");
        }
    }
    // A few chunks of a few terms cost less than Barrett's two portable products.
    assert!(name(&[8, 4, 3, 1, 0], Clmul::Portable).contains("Sparse"));
    assert!(name(&[64, 4, 3, 1, 0], Clmul::Portable).contains("Sparse"));
    assert!(name(&[193, 15, 0], Clmul::Portable).contains("[193, 15, 0]"));
}

#[test]
fn the_trace_is_the_sum_of_the_conjugates() {
    let mut rng = SplitMix(3);
    for e in [
        &[8, 4, 3, 1, 0][..],
        &[113, 9, 0],
        &[127, 126, 0],
        &[128, 7, 2, 1, 0],
        &[163, 160, 157, 156, 0],
        &[233, 74, 0],
        &[64, 4, 3, 1, 0],
        &[2, 1, 0],
        &[1, 0],
        &[1],
    ] {
        let f = BinaryField::new(e);
        let n = (e[0] as usize).div_ceil(64);
        for _ in 0..if cfg!(miri) { 1 } else { 4 } {
            let a = f.element(&(0..n).map(|_| rng.next()).collect::<Vec<_>>());
            let mut t = a.clone();
            let mut s = a.clone();
            for _ in 1..e[0] {
                s = s.square();
                t += &s;
            }
            assert!(t.is_zero() || t.is_one(), "{e:?}");
            assert_eq!(a.trace(), u8::from(t.is_one()), "{e:?}");
        }
        assert_eq!(f.one().trace(), (e[0] % 2) as u8);
    }
}

#[test]
fn quadratic_roots_exist_exactly_for_the_elements_of_trace_zero() {
    let mut rng = SplitMix(4);
    for e in [
        &[4, 1, 0][..],
        &[8, 4, 3, 1, 0],
        &[64, 4, 3, 1, 0],
        &[128, 7, 2, 1, 0],
        &[163, 7, 6, 3, 0],
        &[256, 10, 5, 2, 0],
        &[2, 1, 0],
        &[1, 0],
    ] {
        let f = BinaryField::new(e);
        let n = (e[0] as usize).div_ceil(64);
        for _ in 0..6 {
            let c = f.element(&(0..n).map(|_| rng.next()).collect::<Vec<_>>());
            match c.quadratic_root() {
                Some(z) => {
                    assert_eq!(c.trace(), 0);
                    assert_eq!(&z.square() + &z, c, "{e:?}");
                }
                None => assert_eq!(c.trace(), 1, "{e:?}"),
            }
        }
        assert_eq!(f.zero().quadratic_root(), Some(f.zero()));
    }
    // Over GF(2⁴) every element is checked.
    let f = BinaryField::new(&[4, 1, 0]);
    let roots = (0..16u64)
        .filter(|&v| f.element(&[v]).quadratic_root().is_some())
        .count();
    assert_eq!(roots, 8);
}

#[test]
fn irreducibility_agrees_with_ntl() {
    let (_, bits) = data();
    assert_eq!(bits.len(), 600);
    let mut rng = SplitMix(0x1eed_dead);
    let mut irreducible = 0;
    let count = if cfg!(miri) { 4 } else { bits.len() };
    for &want in &bits[..count] {
        let d = 2 + rng.next() % 600;
        let kind = rng.next() % 3;
        let mut terms = std::collections::BTreeSet::from([0, d]);
        if kind < 2 {
            for _ in 0..if kind == 0 { 1 } else { 3 } {
                let k = 1 + rng.next() % (d - 1);
                if !terms.remove(&k) {
                    terms.insert(k);
                }
            }
        } else {
            for w in 0..d.div_ceil(64) {
                let x = rng.next();
                for j in 0..64 {
                    let i = 64 * w + j;
                    if i >= 1 && i < d && x >> j & 1 == 1 {
                        terms.insert(i);
                    }
                }
            }
        }
        let e: Vec<u32> = terms.iter().map(|&t| t as u32).collect();
        assert_eq!(BinaryField::is_irreducible(&e), want, "{e:?}");
        irreducible += usize::from(want);
    }
    assert!(cfg!(miri) || irreducible == 19, "{irreducible}");
}

#[test]
fn the_irreducible_polynomials_of_each_degree_are_as_many_as_gauss_counted() {
    // N(d) = (1/d)·Σ_{k | d} μ(k)·2^(d/k).
    let mobius = |k: u32| -> i64 {
        let (mut k, mut r, mut p) = (k, 1, 2);
        while p * p <= k {
            if k % p == 0 {
                k /= p;
                if k % p == 0 {
                    return 0;
                }
                r = -r;
            }
            p += 1;
        }
        if k > 1 { -r } else { r }
    };
    let max = if cfg!(miri) { 6 } else { 13 };
    for d in 1..=max {
        let count = (0u64..1 << d)
            .filter(|&low| {
                let mut e = vec![d];
                e.extend((0..d).rev().filter(|&i| low >> i & 1 == 1));
                BinaryField::is_irreducible(&e)
            })
            .count() as i64;
        let gauss: i64 = (1..=d)
            .filter(|k| d % k == 0)
            .map(|k| mobius(k) * (1i64 << (d / k)))
            .sum::<i64>()
            / i64::from(d);
        assert_eq!(count, gauss, "degree {d}");
    }
}

#[test]
fn aes_products_and_inverses() {
    let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    let e = |x: u64| f.element(&[x]);
    // FIPS 197, section 4.2: {57}·{83} = {c1}, {57}·{13} = {fe}.
    assert_eq!((&e(0x57) * &e(0x83)).words(), &[0xc1]);
    assert_eq!((&e(0x57) * &e(0x13)).words(), &[0xfe]);
    for x in 1..256u64 {
        let a = e(x);
        assert_eq!(&a * &a.inverse().unwrap(), f.one());
        assert_eq!(a.sqrt().square(), a);
        assert_eq!(a.pow_u64(255), f.one());
    }
    assert!(e(0).inverse().is_none());
    // The trace of {01} over GF(2⁸) is 8 mod 2 = 0.
    assert_eq!(f.one().trace(), 0);
}

#[test]
fn a_field_of_several_words() {
    let f = BinaryField::new(&[163, 7, 6, 3, 0]);
    let a = f.element(&[0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 0x7]);
    let b = f.element(&[0xdead_beef, 0, 0x5]);
    assert_eq!(&(&a * &b) / &b, a);
    assert_eq!(&a * &a.inverse().unwrap(), f.one());
    assert_eq!(a.sqrt().square(), a);
    assert_eq!(&(&a + &b) * &(&a + &b), &a.square() + &b.square());
    // Half of the elements have trace 1, and the trace is additive.
    assert_eq!((&a + &b).trace(), a.trace() ^ b.trace());
    // x^(2^163) = x.
    let x = f.element(&[2]);
    let mut y = x.clone();
    for _ in 0..163 {
        y = y.square();
    }
    assert_eq!(y, x);
    // An element given with bits beyond the degree is reduced.
    assert_eq!(f.element(&[0, 0, 1 << 35]), f.element(&[0xc9]));
}

#[test]
fn the_fields_of_degree_one() {
    for e in [&[1, 0][..], &[1]] {
        let f = BinaryField::new(e);
        assert_eq!(f.zero().words(), &[0]);
        assert!(f.element(&[0b11]).is_one() == (e.len() == 1));
        assert_eq!(f.one().inverse(), Some(f.one()));
        assert_eq!(f.one().sqrt(), f.one());
        assert_eq!(f.one().trace(), 1);
        assert_eq!(f.one().half_trace(), f.one());
        assert_eq!(f.zero().quadratic_root(), Some(f.zero()));
    }
    // x + 1: x is 1; x: x is 0.
    assert!(BinaryField::new(&[1, 0]).element(&[2]).is_one());
    assert!(BinaryField::new(&[1]).element(&[2]).is_zero());
}

#[test]
fn fields_with_the_same_modulus_are_equal_and_their_elements_mix() {
    let f = BinaryField::new(&[233, 74, 0]);
    let g = BinaryField::with_clmul(&[0, 74, 233], Clmul::Portable);
    assert_eq!(f, g);
    let a = f.element(&[0x1234, 0x5678, 0x9abc, 0xde]);
    let b = g.element(&[0x1234, 0x5678, 0x9abc, 0xde]);
    assert_eq!(a, b);
    assert_eq!(&a * &b, a.square());
    let h = |x: &F2m<'_>| {
        use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
        BuildHasherDefault::<DefaultHasher>::default().hash_one(x)
    };
    assert_eq!(h(&a), h(&b));
    assert_eq!(
        format!("{:?}", f.element(&[0xab])),
        format!("0x{:016x}{:016x}{:016x}{:016x}", 0, 0, 0, 0xab)
    );
}

#[test]
#[should_panic(expected = "no constant term")]
fn a_modulus_without_a_constant_term_panics() {
    let _ = BinaryField::new(&[8, 4, 3, 1]);
}

#[test]
#[should_panic(expected = "x + 1 divides it")]
fn a_modulus_of_an_even_number_of_terms_panics() {
    let _ = BinaryField::new(&[8, 3, 1, 0]);
}

#[test]
#[should_panic(expected = "is reducible")]
fn a_reducible_modulus_panics() {
    // x⁴ + x² + 1 = (x² + x + 1)².
    let _ = BinaryField::new(&[4, 2, 0]);
}

#[test]
#[should_panic(expected = "is reducible")]
fn a_reducible_sparse_modulus_of_high_degree_panics() {
    // x²⁵⁶ + x¹⁰ + x⁵ + x² + 1 is irreducible, but no trinomial of degree 8k is (Swan, 1962).
    let _ = BinaryField::new(&[256, 3, 0]);
}

#[test]
#[should_panic(expected = "repeated exponent")]
fn a_repeated_exponent_panics() {
    let _ = BinaryField::new(&[8, 4, 4, 3, 1, 0]);
}

#[test]
#[should_panic(expected = "the degree must be at least 1")]
fn a_constant_modulus_panics() {
    let _ = BinaryField::new(&[0]);
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn elements_of_different_fields_do_not_mix() {
    let f = BinaryField::new(&[163, 7, 6, 3, 0]);
    let g = BinaryField::new(&[163, 160, 157, 156, 0]);
    let _ = f.one() * g.one();
}

#[test]
#[should_panic(expected = "division by zero")]
fn division_by_zero_panics() {
    let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    let _ = f.one() / f.zero();
}

#[test]
#[should_panic(expected = "the half-trace needs an odd degree")]
fn the_half_trace_of_an_even_degree_panics() {
    let _ = BinaryField::new(&[8, 4, 3, 1, 0]).one().half_trace();
}

#[test]
fn a_kernel_the_cpu_lacks_is_refused() {
    for k in [Clmul::Pmull, Clmul::Pclmulqdq] {
        if !k.is_available() {
            let r = std::panic::catch_unwind(|| BinaryField::with_clmul(&[8, 4, 3, 1, 0], k));
            assert!(r.is_err());
            assert!(std::panic::catch_unwind(|| k.mul(1, 1)).is_err());
        }
    }
}

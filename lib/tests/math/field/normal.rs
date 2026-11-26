//! Gaussian normal bases: `a::math::field::{NormalBasis, NormalElement, BasisConversion}` against
//! FIPS 186-4 (its example of D.3, and the normal-basis values of the NIST curves of D.1.3 with the
//! conversion rows of D.9 and D.10: data/normal_fips186.txt), against the minimal polynomials of
//! Gauss periods computed by PARI/GP (data/normal_pari.txt, from data/gen_normal.gp), and against
//! the polynomial basis through the conversions.

use a::math::field::{BinaryField, NormalBasis, NormalElement};

/// splitmix64.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// `m` random bits in words.
    fn bits(&mut self, m: usize) -> Vec<u64> {
        let mut w: Vec<u64> = (0..m.div_ceil(64)).map(|_| self.next()).collect();
        if !m.is_multiple_of(64) {
            *w.last_mut().unwrap() &= (1 << (m % 64)) - 1;
        }
        w
    }
}

/// The words of a hexadecimal number, least significant first.
fn hex_words(s: &str) -> Vec<u64> {
    let digits: Vec<u8> = s
        .bytes()
        .rev()
        .map(|c| (c as char).to_digit(16).unwrap() as u8)
        .collect();
    digits
        .chunks(16)
        .map(|c| c.iter().rev().fold(0u64, |w, &d| w << 4 | u64::from(d)))
        .collect()
}

/// The `m` low bits of `w` in the reverse order: FIPS 186-4's normal-basis strings put `a₀` first,
/// as the leading bit of the number.
fn reversed(w: &[u64], m: usize) -> Vec<u64> {
    let mut r = vec![0u64; m.div_ceil(64)];
    for i in 0..m {
        let j = m - 1 - i;
        if w.get(j / 64).is_some_and(|x| x >> (j % 64) & 1 == 1) {
            r[i / 64] |= 1 << (i % 64);
        }
    }
    r
}

/// The least exponent `k` of a trinomial `xᵐ + xᵏ + 1`, or failing that the pentanomial of the
/// standards (FIPS 186-4, D.1.1.3), that is irreducible.
fn standard_modulus(m: u32) -> Vec<u32> {
    for k in 1..m {
        if BinaryField::is_irreducible(&[m, k, 0]) {
            return vec![m, k, 0];
        }
    }
    for a in 3..m {
        for b in 2..a {
            for c in 1..b {
                if BinaryField::is_irreducible(&[m, a, b, c, 0]) {
                    return vec![m, a, b, c, 0];
                }
            }
        }
    }
    unreachable!("every degree from 4 on has an irreducible trinomial or pentanomial")
}

#[test]
fn the_example_of_fips_186_4_d3() {
    // GF(2⁷), type 4: p = 29, u = 12 or 17, and
    // F(u, v) = u0v1 + u1(v0+v2+v5+v6) + u2(v1+v3+v4+v5) + u3(v2+v5) + u4(v2+v6)
    //         + u5(v1+v2+v3+v6) + u6(v1+v4+v5+v6).
    let nb = NormalBasis::new(7);
    assert_eq!(nb.gaussian_type(), 4);
    let rows: [&[usize]; 7] = [
        &[1],
        &[0, 2, 5, 6],
        &[1, 3, 4, 5],
        &[2, 5],
        &[2, 6],
        &[1, 2, 3, 6],
        &[1, 4, 5, 6],
    ];
    let l = nb.lambda_matrix();
    for (j, row) in rows.iter().enumerate() {
        for k in 0..7 {
            assert_eq!(l.get(j, k), row.contains(&k), "λ({j}, {k})");
        }
    }
    assert_eq!(nb.complexity(), 21);
    // (1010111)·(1100001) = (1011001), a₀ first.
    let s = |bits: &str| {
        nb.element(&[bits
            .bytes()
            .enumerate()
            .filter(|&(_, b)| b == b'1')
            .map(|(i, _)| 1u64 << i)
            .sum()])
    };
    assert_eq!(&s("1010111") * &s("1100001"), s("1011001"));
    // Squaring: (a₀ … a₆)² = (a₆ a₀ … a₅).
    assert_eq!(s("1010111").square(), s("1101011"));
}

/// The fields of data/normal_fips186.txt: degree, type, modulus, θ, the root, and the values.
type Fips = (
    u32,
    u32,
    Vec<u32>,
    String,
    String,
    Vec<(String, String, String)>,
);

fn fips() -> Vec<Fips> {
    let mut out: Vec<Fips> = Vec::new();
    for line in include_str!("data/normal_fips186.txt").lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        match p[0] {
            "field" => out.push((
                p[1].parse().unwrap(),
                p[2].parse().unwrap(),
                p[3].split(',').map(|e| e.parse().unwrap()).collect(),
                String::new(),
                String::new(),
                Vec::new(),
            )),
            "theta" => out.last_mut().unwrap().3 = p[1].to_string(),
            "root" => out.last_mut().unwrap().4 = p[1].to_string(),
            "value" => out.last_mut().unwrap().5.push((
                p[1].to_string(),
                p[2].to_string(),
                p[3].to_string(),
            )),
            _ => {}
        }
    }
    out
}

/// Checks the NIST curves' values of degree `m`, through the conversion of FIPS 186-4's θ and
/// through [`NormalBasis::conversion`]'s, which differ by a rotation.
fn check_fips(m: u32) {
    let (_, t, modulus, theta, root, values) = fips().into_iter().find(|f| f.0 == m).unwrap();
    let f = BinaryField::new(&modulus);
    let nb = NormalBasis::new(m);
    assert_eq!(nb.gaussian_type(), t);
    let m = m as usize;
    let theta = f.element(&hex_words(&theta));
    let c = nb
        .conversion_with(&theta)
        .expect("FIPS 186-4's θ is a root of the minimal polynomial");
    assert_eq!(c.normal_element(), theta);
    assert_eq!(
        c.polynomial_generator().words(),
        reversed(&hex_words(&root), m)
    );
    let canonical = nb.conversion(&f);
    // The rotation from FIPS 186-4's coordinates to the canonical ones: θ_FIPS = θ^(2^k).
    let k = (0..m as i64)
        .find(|&k| canonical.to_normal(&theta) == nb.element(&[1]).frobenius(k))
        .expect("the two θ are conjugates");
    for (name, poly, normal) in &values {
        let p = f.element(&hex_words(poly));
        let n = nb.element(&reversed(&hex_words(normal), m));
        assert_eq!(c.to_normal(&p), n, "{name}");
        assert_eq!(c.to_polynomial(&n), p, "{name}");
        assert_eq!(canonical.to_normal(&p), n.frobenius(k), "{name}");
    }
    // The curve equations hold in both bases: y² + xy = x³ + ax² + b, b = 1 for the Koblitz
    // curves, a = 1 for the others and for K-163.
    for curve in ["K", "B"] {
        let get = |what: &str| {
            values
                .iter()
                .find(|v| v.0 == format!("{curve}-{m}_{what}"))
                .map(|v| nb.element(&reversed(&hex_words(&v.2), m)))
        };
        let b = get("b").unwrap_or(nb.one());
        let a = if curve == "B" || m == 163 {
            nb.one()
        } else {
            nb.zero()
        };
        let (x, y) = (get("Gx").unwrap(), get("Gy").unwrap());
        let lhs = &y.square() + &(&x * &y);
        let rhs = &(&(&x.square() * &x) + &(&a * &x.square())) + &b;
        assert_eq!(lhs, rhs, "{curve}-{m}");
        let (px, py) = (c.to_polynomial(&x), c.to_polynomial(&y));
        let pa = c.to_polynomial(&a);
        let pb = c.to_polynomial(&b);
        assert_eq!(
            &py.square() + &(&px * &py),
            &(&(&px.square() * &px) + &(&pa * &px.square())) + &pb
        );
    }
}

#[test]
fn the_nist_curves_of_degrees_163_233_and_283_agree_with_fips_186_4() {
    // Too slow for Miri, which checks the conversions on small degrees below.
    if cfg!(miri) {
        return;
    }
    for m in [163, 233, 283] {
        check_fips(m);
    }
}

#[test]
fn the_nist_curves_of_degrees_409_and_571_agree_with_fips_186_4() {
    if cfg!(miri) {
        return;
    }
    for m in [409, 571] {
        check_fips(m);
    }
}

/// Checks the minimal polynomials of data/normal_pari.txt whose degree `m` passes `keep`.
fn check_pari(keep: impl Fn(u32) -> bool) -> usize {
    let mut checked = 0;
    for line in include_str!("data/normal_pari.txt").lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p[0] == "#" {
            continue;
        }
        let (m, t): (u32, u32) = (p[1].parse().unwrap(), p[2].parse().unwrap());
        if !keep(m) {
            continue;
        }
        let nb = if p[0] == "lowest" {
            NormalBasis::new(m)
        } else {
            NormalBasis::with_type(m, t)
        };
        assert_eq!(nb.gaussian_type(), t, "m = {m}");
        let g = hex_words(p[3]);
        let want: Vec<u32> = (0..=m)
            .rev()
            .filter(|&i| g[i as usize / 64] >> (i % 64) & 1 == 1)
            .collect();
        assert_eq!(nb.minimal_polynomial(), want, "m = {m}, T = {t}");
        checked += 1;
    }
    checked
}

#[test]
fn minimal_polynomials_agree_with_pari() {
    // 119 bases: the lowest type of every m to 100 not divisible by 8 (types 1 to 12), the
    // standards' degrees, and others, of types 1 to 22 in all.
    if cfg!(miri) {
        assert!(check_pari(|m| m <= 10) > 5);
    } else {
        assert_eq!(check_pari(|m| m <= 300), 119);
    }
}

#[test]
fn minimal_polynomials_of_degrees_409_and_571_agree_with_pari() {
    if !cfg!(miri) {
        assert_eq!(check_pari(|m| m > 300), 2);
    }
}

/// Field laws through a conversion: products, quotients, inverses, squares and roots, traces and
/// powers agree with the polynomial basis.
fn check_conversion(nb: &NormalBasis, f: &BinaryField, rng: &mut SplitMix, cases: usize) {
    let m = nb.degree() as usize;
    let c = nb.conversion(f);
    assert_eq!(c.to_normal(&f.one()), nb.one());
    assert_eq!(c.to_normal(&f.zero()), nb.zero());
    assert_eq!(c.to_normal(&c.normal_element()), nb.element(&[1]));
    for _ in 0..cases {
        let (a, b) = (f.element(&rng.bits(m)), f.element(&rng.bits(m)));
        let (na, nbb) = (c.to_normal(&a), c.to_normal(&b));
        assert_eq!(c.to_polynomial(&na), a);
        assert_eq!(&na + &nbb, c.to_normal(&(&a + &b)));
        assert_eq!(
            &na * &nbb,
            c.to_normal(&(&a * &b)),
            "m = {m}, T = {}",
            nb.gaussian_type()
        );
        assert_eq!(na.square(), c.to_normal(&a.square()));
        assert_eq!(na.sqrt(), c.to_normal(&a.sqrt()));
        assert_eq!(na.trace(), a.trace());
        assert_eq!(na.pow_u64(1000), c.to_normal(&a.pow_u64(1000)));
        if !b.is_zero() {
            assert_eq!(nbb.inverse().unwrap(), c.to_normal(&b.inverse().unwrap()));
            assert_eq!(&na / &nbb, c.to_normal(&(&a / &b)));
        }
    }
}

#[test]
fn every_degree_to_100_agrees_with_its_polynomial_basis() {
    // Every m from 2 to 100 that is not a multiple of 8, in its lowest type: types 1 to 12, odd and
    // even, and cases where the period needs an extension of GF(2ᵐ) of degree above 2.
    let mut rng = SplitMix(5);
    let top = if cfg!(miri) { 12 } else { 100 };
    for m in (2..=top).filter(|m| m % 8 != 0) {
        let nb = NormalBasis::new(m);
        let f = BinaryField::new(&if m < 4 {
            vec![m, 1, 0]
        } else {
            standard_modulus(m)
        });
        check_conversion(&nb, &f, &mut rng_fork(&mut rng), 4);
    }
}

/// A new generator from `rng`.
fn rng_fork(rng: &mut SplitMix) -> SplitMix {
    SplitMix(rng.next())
}

#[test]
fn other_types_and_moduli_agree_with_their_polynomial_bases() {
    let mut rng = SplitMix(6);
    for (m, t, e) in [
        (7, 6, vec![7, 6, 0]),
        (10, 13, vec![10, 3, 0]),
        (12, 5, vec![12, 3, 0]),
        (163, 16, vec![163, 7, 6, 3, 0]),
        (163, 4, vec![163, 160, 157, 156, 0]),
        (233, 6, vec![233, 159, 0]),
        (113, 2, vec![113, 9, 0]),
        (131, 2, vec![131, 8, 3, 2, 0]),
        (193, 4, vec![193, 15, 0]),
        (239, 2, vec![239, 158, 0]),
    ] {
        if cfg!(miri) && m > 12 {
            continue;
        }
        let nb = NormalBasis::with_type(m, t);
        let f = BinaryField::new(&e);
        check_conversion(&nb, &f, &mut rng, 3);
    }
}

#[test]
fn the_optimal_normal_bases_have_complexity_2m_minus_1() {
    for m in [2, 3, 4, 5, 9, 10, 11, 12, 113, 131, 233, 239] {
        let nb = NormalBasis::new(m);
        assert!(nb.gaussian_type() <= 2, "m = {m}");
        assert_eq!(nb.complexity(), 2 * m as usize - 1, "m = {m}");
        let l = nb.lambda_matrix();
        assert_eq!(l, l.transpose());
    }
    // Other types: at most Tm − 1 terms for an even type (the p − 2 pairs, less those that
    // cancel), Tm − 2 + m for an odd one (and the m terms aᵢ·b_{i+m/2}); symmetric.
    for (m, t) in [(163, 4), (283, 6), (7, 10), (12, 3), (14, 15), (10, 13)] {
        let nb = NormalBasis::with_type(m, t);
        let bound = if t % 2 == 0 { t * m - 1 } else { t * m - 2 + m };
        assert!(nb.complexity() <= bound as usize, "m = {m}, T = {t}");
        assert_eq!(nb.lambda_matrix(), nb.lambda_matrix().transpose());
    }
}

#[test]
fn the_lowest_types_are_those_of_the_standards() {
    for (m, t) in [
        (163, 4),
        (233, 2),
        (283, 6),
        (409, 4),
        (571, 10),
        (113, 2),
        (131, 2),
        (193, 4),
    ] {
        assert_eq!(NormalBasis::new(m).gaussian_type(), t, "m = {m}");
    }
    for m in 2..if cfg!(miri) { 20 } else { 200 } {
        let nb = (m % 8 != 0).then(|| NormalBasis::new(m));
        if let Some(nb) = nb {
            for t in 1..nb.gaussian_type() {
                assert!(!NormalBasis::exists(m, t));
            }
        }
    }
    assert!(!NormalBasis::exists(1, 1));
    assert!(!NormalBasis::exists(2, 0));
}

#[test]
fn squares_roots_traces_and_inverses_in_the_basis_itself() {
    let mut rng = SplitMix(7);
    let degrees: &[u32] = if cfg!(miri) {
        &[2, 3, 7, 10]
    } else {
        &[2, 3, 7, 10, 63, 65, 113, 127, 129, 163]
    };
    for &m in degrees {
        let nb = NormalBasis::new(m);
        let mm = m as usize;
        for _ in 0..3 {
            let a = nb.element(&rng.bits(mm));
            assert_eq!(a.sqrt().square(), a);
            assert_eq!(a.square(), &a * &a);
            assert_eq!(a.frobenius(m as i64), a);
            assert_eq!(a.frobenius(-3).frobenius(3), a);
            let mut t = a.clone();
            let mut s = a.clone();
            for _ in 1..m {
                s = s.square();
                t += &s;
            }
            assert!(t.is_zero() || t.is_one());
            assert_eq!(a.trace(), u8::from(t.is_one()));
            if !a.is_zero() {
                assert_eq!(&a * &a.inverse().unwrap(), nb.one());
                assert_eq!(&a / &a, nb.one());
            }
            assert_eq!(&a + &a, nb.zero());
            assert_eq!(-&a, a);
        }
        assert!(nb.one().is_one());
        assert_eq!(nb.one().trace(), (m % 2) as u8);
        assert_eq!(nb.zero().inverse(), None);
    }
}

#[test]
fn a_conversion_with_an_element_that_is_no_root_is_none() {
    let (f, nb) = if cfg!(miri) {
        (BinaryField::new(&[11, 2, 0]), NormalBasis::new(11))
    } else {
        (BinaryField::new(&[163, 7, 6, 3, 0]), NormalBasis::new(163))
    };
    assert!(nb.conversion_with(&f.element(&[2])).is_none());
    assert!(nb.conversion_with(&f.zero()).is_none());
    assert!(nb.conversion_with(&f.one()).is_none());
    let theta = nb.conversion(&f).normal_element();
    // Each conjugate of θ gives the conversion rotated by its place.
    let mut s = theta.clone();
    for k in 1..4 {
        s = s.square();
        let c = nb.conversion_with(&s).unwrap();
        assert_eq!(c.to_normal(&theta), nb.element(&[1]).frobenius(-k));
    }
}

#[test]
fn the_debug_forms() {
    let nb = NormalBasis::new(7);
    assert_eq!(format!("{nb:?}"), "NormalBasis { degree: 7, type: 4 }");
    let e: NormalElement<'_> = nb.element(&[0x41]);
    assert_eq!(format!("{e:?}"), "0x0000000000000041");
    let f = BinaryField::new(&[7, 1, 0]);
    assert!(format!("{:?}", nb.conversion(&f)).contains("normal_element"));
}

#[test]
#[should_panic(expected = "multiple of 8")]
fn a_degree_divisible_by_8_has_no_gaussian_normal_basis() {
    let _ = NormalBasis::new(16);
}

#[test]
#[should_panic(expected = "needs m ≥ 2")]
fn the_degree_1_has_no_gaussian_normal_basis() {
    let _ = NormalBasis::new(1);
}

#[test]
#[should_panic(expected = "no Gaussian normal basis of type 2")]
fn a_type_that_does_not_exist_panics() {
    let _ = NormalBasis::with_type(163, 2);
}

#[test]
#[should_panic(expected = "beyond the degree")]
fn a_coordinate_beyond_the_degree_panics() {
    let _ = NormalBasis::new(7).element(&[0x80]);
}

#[test]
#[should_panic(expected = "a basis of GF(2^163) and a field of degree 233")]
fn a_conversion_to_a_field_of_another_degree_panics() {
    let f = BinaryField::new(&[233, 74, 0]);
    let _ = NormalBasis::new(163).conversion(&f);
}

#[test]
#[should_panic(expected = "elements of different bases")]
fn elements_of_different_bases_do_not_mix() {
    let a = NormalBasis::with_type(7, 4);
    let b = NormalBasis::with_type(7, 6);
    let _ = a.one() * b.one();
}

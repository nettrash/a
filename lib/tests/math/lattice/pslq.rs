//! Tests of `a::math::lattice` (pslq): integer relations by PSLQ.
//!
//! The answers are compared with `data/pslq_expected.txt`, written by `data/pslq_gen.py`: a
//! reference PSLQ written out in Python from the paper of Ferguson, Bailey and Arno (floating
//! `H` of twice the precision, the same rules otherwise), mpmath's `pslq` and `findpoly`, and
//! PARI/GP's `lindep`, each at the precision of the inputs. The inputs are computed here with the
//! crate's correctly rounded functions and checked against the generator's by the low 64 bits of
//! every mantissa. Every relation returned is also checked here, independently of the crate's
//! own check, and every bound below 8 by enumerating all the shorter integer vectors.

use super::common::Rng;
use a::math::float::{BigFloat, Round};
use a::math::lattice::{Pslq, PslqOptions, pslq, pslq_f64};
use a::math::numbers::Number;

const EXPECTED: &str = include_str!("data/pslq_expected.txt");

/// The lines of `EXPECTED` that begin with `kind case`, the words after them.
fn expected(kind: &str, case: &str) -> Vec<Vec<String>> {
    EXPECTED
        .lines()
        .filter_map(|l| {
            let mut w = l.split_whitespace();
            (w.next() == Some(kind) && w.next() == Some(case))
                .then(|| w.map(str::to_string).collect())
        })
        .collect()
}

fn numbers(words: &[String]) -> Vec<Number> {
    words.iter().map(|w| w.parse().unwrap()).collect()
}

fn ints(v: &[i64]) -> Vec<Number> {
    v.iter().map(|&x| Number::from(x)).collect()
}

/// `f()` at `p + 64` bits, rounded once to `p`, as the generator computes every input.
fn rounded(p: u64, f: impl Fn(u64) -> BigFloat) -> BigFloat {
    f(p + 64).with_prec(p)
}

/// The powers `1, α, …, α^deg` of `α = r1^(1/k1) ± r2^(1/k2)`, to `p` bits.
fn powers_of_radicals(
    r1: u32,
    k1: u32,
    sign: i32,
    r2: u32,
    k2: u32,
    deg: i64,
    p: u64,
) -> Vec<BigFloat> {
    let alpha = |w: u64| {
        let a = BigFloat::from(r1).with_prec(w).nth_root(k1);
        let b = BigFloat::from(r2).with_prec(w).nth_root(k2);
        if sign > 0 { a + b } else { a - b }
    };
    (0..=deg)
        .map(|k| rounded(p, |w| alpha(w).powi(k)))
        .collect()
}

/// `S_j = Σ_k 1/(16^k·(8k + j))`, the sums of the formula of Bailey, Borwein and Plouffe.
fn bbp_sum(j: u64, w: u64) -> BigFloat {
    let w = w + 16;
    let mut s = BigFloat::zero(w);
    let mut k = 0u64;
    loop {
        let d = Number::from(8 * k + j) << (4 * k) as usize;
        if d.bits() > w + 8 {
            return s;
        }
        let t = BigFloat::from(1)
            .div_round(&BigFloat::from(&d), w, Round::Nearest)
            .0;
        s = s.add_round(&t, w, Round::Nearest).0;
        k += 1;
    }
}

/// `ζ(3) = (5/2)·Σ_{k≥1} (−1)^(k+1) / (k³·C(2k, k))`, the series of Apéry's proof.
fn zeta3(w: u64) -> BigFloat {
    let w = w + 16;
    let mut s = BigFloat::zero(w);
    let mut c = Number::from(1); // C(2k, k)
    for k in 1u64.. {
        c = &c * &Number::from((2 * k) * (2 * k - 1)) / &Number::from(k * k);
        let d = &c * &Number::from(k * k * k);
        if d.bits() > w + 8 {
            break;
        }
        let t = BigFloat::from(1)
            .div_round(&BigFloat::from(&d), w, Round::Nearest)
            .0;
        s = if k % 2 == 1 {
            s.add_round(&t, w, Round::Nearest).0
        } else {
            s.sub_round(&t, w, Round::Nearest).0
        };
    }
    s.mul_round(&BigFloat::from(2.5), w, Round::Nearest).0
}

fn ln(k: u32, w: u64) -> BigFloat {
    BigFloat::from(k).with_prec(w).ln()
}

/// The inputs of a case of the generator, and its `max_norm`.
fn case(name: &str) -> (Vec<BigFloat>, Option<Number>) {
    let one = |p: u64| BigFloat::from(1).with_prec(p);
    let pow10 = |k: u32| Some(Number::from(10).pow(k));
    match name {
        "sqrt2_sqrt3" => (powers_of_radicals(2, 2, 1, 3, 2, 4, 200), None),
        "deg12" => (powers_of_radicals(3, 3, 1, 2, 4, 12, 600), None),
        "deg16" => (powers_of_radicals(3, 4, -1, 2, 4, 16, 400), None),
        "deg35" => (powers_of_radicals(2, 5, 1, 3, 7, 35, 1000), None),
        "machin" => {
            let p = 200;
            let at = |q: u32| rounded(p, |w| (BigFloat::from(1).with_prec(w) / q).atan());
            (
                vec![rounded(p, |w| BigFloat::pi(w) / 4u32), at(5), at(239)],
                None,
            )
        }
        "machin5" => {
            let p = 200;
            let at = |q: u32| rounded(p, |w| (BigFloat::from(1).with_prec(w) / q).atan());
            let mut x = vec![rounded(p, |w| BigFloat::pi(w) / 4u32)];
            x.extend([49, 57, 239, 110443].map(at));
            (x, None)
        }
        "bbp" => {
            let p = 300;
            let mut x = vec![rounded(p, BigFloat::pi)];
            x.extend([1, 4, 5, 6].map(|j| rounded(p, |w| bbp_sum(j, w))));
            (x, None)
        }
        "logs" => {
            let p = 300;
            let mut x: Vec<BigFloat> = [2, 3, 5, 7]
                .iter()
                .map(|&q| rounded(p, |w| ln(q, w)))
                .collect();
            x.push(rounded(p, |w| {
                ln(2, w) * 31u32 - ln(3, w) * 17u32 + ln(5, w) * 11u32 + ln(7, w) * 4u32
            }));
            (x, None)
        }
        "zeta3_pi" => {
            let p = 300;
            (
                vec![one(p), rounded(p, BigFloat::pi), rounded(p, zeta3)],
                pow10(12),
            )
        }
        "zeta3_pi3" => {
            let p = 300;
            (
                vec![rounded(p, zeta3), rounded(p, |w| BigFloat::pi(w).powi(3))],
                pow10(20),
            )
        }
        "euler_poly6" => {
            let p = 700;
            let x = (0..7)
                .map(|k| rounded(p, |w| BigFloat::euler_gamma(w).powi(k)))
                .collect();
            (x, pow10(6))
        }
        "constants" => {
            let p = 400;
            let x = vec![
                one(p),
                rounded(p, BigFloat::pi),
                rounded(p, BigFloat::e),
                rounded(p, BigFloat::ln2),
                rounded(p, BigFloat::euler_gamma),
                rounded(p, BigFloat::catalan),
            ];
            (x, pow10(8))
        }
        "acot" => {
            let p = 200;
            let mut x = vec![rounded(p, BigFloat::pi)];
            x.extend(
                (2..=10).map(|q: u32| rounded(p, |w| (BigFloat::from(1).with_prec(w) / q).atan())),
            );
            (x, None)
        }
        "scales" => {
            let at =
                |m: i64, e: i64| BigFloat::from_parts(&Number::from(m), e, 53, Round::Nearest).0;
            (vec![at(1, 0), at(1, -200), at(3, -200)], None)
        }
        "pi_e_1e6" => (vec![BigFloat::pi(200), BigFloat::e(200)], pow10(6)),
        "pi_e" => (vec![BigFloat::pi(200), BigFloat::e(200)], None),
        "pi_e_1" => (vec![BigFloat::pi(200), BigFloat::e(200), one(200)], None),
        _ => panic!("no case {name}"),
    }
}

/// Checks the inputs of a case against the generator's: the low 64 bits of each mantissa.
fn check_inputs(name: &str, x: &[BigFloat]) {
    let lines = expected("input", name);
    assert_eq!(lines.len(), x.len(), "{name}: the inputs");
    for (i, v) in x.iter().enumerate() {
        let (m, _) = v.mantissa_exponent().unwrap();
        let low = m.abs().to_str_radix(16);
        let low = &low[low.len().saturating_sub(16)..];
        let want = format!("{:0>16}", lines[i][1]);
        let got = format!("{low:0>16}");
        assert_eq!(got, want, "{name}: input {i} differs from the generator's");
    }
}

/// Whether `m` is a relation of `x`, computed here without the crate's PSLQ: the exact
/// `|Σ mᵢ·xᵢ|` against `Σ |mᵢ|·2^error_bits·ulp(xᵢ)`, signs of exact dot products.
fn is_relation(x: &[BigFloat], m: &[Number], error_bits: u32) -> bool {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (xi, mi) in x.iter().zip(m) {
        xs.push(BigFloat::from(mi));
        ys.push(xi.clone());
    }
    let tol: Vec<(BigFloat, BigFloat)> = x
        .iter()
        .zip(m)
        .filter(|(xi, _)| !xi.is_zero())
        .map(|(xi, mi)| {
            let (_, q) = xi.mantissa_exponent().unwrap();
            let ulp = BigFloat::from_parts(
                &Number::from(1),
                q + i64::from(error_bits),
                64,
                Round::Nearest,
            )
            .0;
            (BigFloat::from(&mi.abs()), ulp)
        })
        .collect();
    let mut lo_x = xs.clone();
    let mut lo_y = ys.clone();
    let mut hi_x = xs;
    let mut hi_y = ys;
    for (a, u) in &tol {
        lo_x.push(-a.clone());
        lo_y.push(u.clone());
        hi_x.push(a.clone());
        hi_y.push(u.clone());
    }
    // Σ mᵢxᵢ − tol ≤ 0 ≤ Σ mᵢxᵢ + tol, each sum exact in sign.
    let lo = BigFloat::dot_round(&lo_x, &lo_y, 8, Round::Nearest).0;
    let hi = BigFloat::dot_round(&hi_x, &hi_y, 8, Round::Nearest).0;
    (lo.is_zero() || lo.is_sign_negative()) && (hi.is_zero() || hi.is_sign_positive())
}

/// The checks of every relation: nonzero, primitive, first nonzero entry positive, a relation.
fn check_relation(x: &[BigFloat], m: &[Number], error_bits: u32) {
    assert_eq!(m.len(), x.len());
    let first = m.iter().find(|v| !v.is_zero()).expect("a nonzero relation");
    assert!(first.is_positive(), "{m:?} is not normalised");
    let g = m.iter().fold(Number::from(0), |g, v| g.gcd(v));
    assert_eq!(g, Number::from(1), "{m:?} is not primitive");
    assert!(
        is_relation(x, m, error_bits),
        "{m:?} is not a relation of {x:?}"
    );
}

fn norm2(m: &[Number]) -> Number {
    m.iter().fold(Number::from(0), |s, v| s + v * v)
}

/// Runs a case of the generator, checks the inputs, and compares with every reference.
fn run_case(name: &str) -> Pslq {
    let (x, max_norm) = case(name);
    check_inputs(name, &x);
    let r = pslq(&x, &PslqOptions::new().with_max_norm(max_norm.clone()));
    let reference = expected("ref", name);
    assert_eq!(reference.len(), 1, "{name}: the reference");
    let reference = &reference[0];
    match reference[0].as_str() {
        "relation" => {
            let m = numbers(&reference[1..]);
            assert_eq!(r, Pslq::Relation(m.clone()), "{name}");
            check_relation(&x, &m, 8);
            for other in ["mpmath", "findpoly", "pari"] {
                for line in expected(other, name) {
                    if line[0] != "none" {
                        assert_eq!(numbers(&line), m, "{name}: {other}");
                    }
                }
            }
        }
        "nonebelow" => {
            let want: BigFloat = reference[1].parse().unwrap();
            let Pslq::NoneBelow(bound) = &r else {
                panic!("{name}: {r:?}")
            };
            let max_norm = max_norm.unwrap();
            assert!(*bound >= max_norm, "{name}: {bound} below {max_norm}");
            // The bound proved here and the reference's agree to 1e-12, the reference's above.
            let rel = ((bound.clone() - &want) / &want).abs();
            assert!(rel < 1e-12, "{name}: {bound} against {want}");
        }
        other => panic!("{name}: reference {other}"),
    }
    r
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_minimal_polynomial_of_sqrt2_plus_sqrt3_is_found() {
    let r = run_case("sqrt2_sqrt3");
    assert_eq!(r, Pslq::Relation(ints(&[1, 0, -10, 0, 1])));
}

#[test]
#[cfg_attr(miri, ignore)]
fn minimal_polynomials_of_degree_12_and_16_agree_with_mpmath_and_pari() {
    run_case("deg12");
    run_case("deg16");
}

// The longest test of the area: about 20 s in the debug profile.
#[test]
#[cfg_attr(miri, ignore)]
fn the_minimal_polynomial_of_degree_35_of_2_5th_root_plus_3_7th_root_is_found() {
    let r = run_case("deg35");
    let m = r.relation().unwrap();
    // −x³⁵ + … + 371, PARI's polresultant((x − y)⁷ − 3, y⁵ − 2, y), of norm² 11474178368232.
    assert_eq!(m[0], Number::from(371));
    assert_eq!(m[35], Number::from(-1));
    assert_eq!(norm2(m), "11474178368232".parse::<Number>().unwrap());
}

#[test]
#[cfg_attr(miri, ignore)]
fn machin_formulas_are_found() {
    assert_eq!(run_case("machin"), Pslq::Relation(ints(&[1, -4, 1])));
    assert_eq!(
        run_case("machin5"),
        Pslq::Relation(ints(&[1, -12, -32, 5, -12]))
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_bbp_formula_for_pi_is_found() {
    // π = 4·S₁ − 2·S₄ − S₅ − S₆.
    assert_eq!(run_case("bbp"), Pslq::Relation(ints(&[1, -4, 2, 1, 1])));
}

#[test]
#[cfg_attr(miri, ignore)]
fn logarithms_of_primes_give_the_exponents() {
    assert_eq!(
        run_case("logs"),
        Pslq::Relation(ints(&[31, -17, 11, 4, -1]))
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn zeta3_has_no_small_relation_with_pi() {
    run_case("zeta3_pi");
    run_case("zeta3_pi3");
}

#[test]
#[cfg_attr(miri, ignore)]
fn euler_gamma_is_no_root_of_a_small_polynomial_of_degree_6() {
    run_case("euler_poly6");
}

#[test]
#[cfg_attr(miri, ignore)]
fn classical_constants_have_no_relation_below_1e8() {
    run_case("constants");
}

#[test]
#[cfg_attr(miri, ignore)]
fn pi_and_e_have_no_relation_below_1e6() {
    run_case("pi_e_1e6");
}

#[test]
#[cfg_attr(miri, ignore)]
fn without_a_limit_a_relation_to_the_precision_turns_up() {
    // Every n numbers of p bits have relations of norm about 2^(p/n) to that precision; PSLQ
    // finds the one the reference finds.
    for name in ["pi_e", "pi_e_1"] {
        let (x, _) = case(name);
        let Pslq::Relation(m) = run_case(name) else {
            panic!("{name}")
        };
        let bits = norm2(&m).bits() / 2;
        assert!(
            (150 / x.len() as u64..=200 / x.len() as u64 + 8).contains(&bits),
            "{name}: {bits}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn dependent_arctangents_give_the_reference_relation() {
    // π and acot 2, …, acot 10, among which acot 2 = acot 3 + acot 7 and others hold: one of
    // the relations, the one the reference finds.
    let Pslq::Relation(m) = run_case("acot") else {
        panic!()
    };
    assert!(m[0].is_zero(), "{m:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn numbers_of_very_different_sizes_give_the_reference_relation() {
    // 1, 2^-200 and 3·2^-200 at 53 bits: the first reduction of H makes the column (1, −a, −b),
    // a ≈ 2^200/10 and b ≈ 3·2^200/10, which is no exact relation (a + 3b misses 2^200 by about
    // 2^76) but is one to the precision (a residual near 2^-123 against a tolerance near 2^-43),
    // so it is returned before the first iteration; PSLQ with exact detection would find
    // (0, 3, −1) at the first. The reference, with the same test, returns the same column; the
    // low bits of a and b follow the rounding of H, so they are compared to 2^-100.
    let (x, _) = case("scales");
    check_inputs("scales", &x);
    let Pslq::Relation(m) = pslq(&x, &PslqOptions::new()) else {
        panic!()
    };
    check_relation(&x, &m, 8);
    assert_eq!(m[0], Number::from(1));
    let off = &(&m[1] + &(&m[2] * &Number::from(3))) + &(Number::from(1) << 200);
    assert!(off.bits() < 150, "{m:?}");
    let reference = numbers(&expected("ref", "scales")[0][1..]);
    for k in 1..3 {
        let d = (&m[k] - &reference[k]).abs();
        assert!(d.bits() + 100 < m[k].bits(), "{m:?} against {reference:?}");
    }
}

/// What a sweep met: relations, bounds, and the bounds checked by enumeration, of them those of
/// numbers of mixed precisions and those of `IterationLimit`.
#[derive(Debug, Default)]
struct Swept {
    relations: usize,
    bounds: usize,
    enumerated: usize,
    enumerated_mixed: usize,
    enumerated_iterations: usize,
}

/// Random vectors of 2 to 5 numbers, half of them with a planted relation with coefficients in
/// −5..=5: half of the vectors of one precision from 12 to 90 bits, half of a precision from 8 to
/// 120 bits for each number; a third with exponents spread over ±40 bits, the rest within 8; at
/// random `error_bits`, `γ` (1.16, 1.155, 1.5 or 4), `max_norm` and, for a third, `max_iter`
/// below 25. Every relation is checked, every `NoneBelow` must reach its limit, and a bound below
/// `enumerate(n)` is checked by enumerating every integer vector of smaller norm, none of which
/// may be a relation.
fn sweep(seed: u64, rounds: usize, enumerate: impl Fn(usize) -> u32) -> Swept {
    let mut rng = Rng(seed);
    let mut out = Swept::default();
    for _ in 0..rounds {
        let n = 2 + (rng.next() % 4) as usize;
        let mixed = rng.next().is_multiple_of(2);
        let common = 12 + rng.next() % 79;
        let spread = if rng.next().is_multiple_of(3) { 81 } else { 8 };
        let error_bits = (rng.next() % 9) as u32;
        let precs: Vec<u64> = (0..n)
            .map(|_| if mixed { 8 + rng.next() % 113 } else { common })
            .collect();
        let mut x: Vec<BigFloat> = precs
            .iter()
            .map(|&p| {
                let m = rng.number(p + 4);
                let e =
                    -(p as i64) - (rng.next() % spread) as i64 + if spread > 8 { 40 } else { 0 };
                BigFloat::from_parts(&m, e, p, Round::Nearest).0
            })
            .collect();
        if rng.next().is_multiple_of(2) {
            let c: Vec<i64> = (0..n).map(|_| (rng.next() % 11) as i64 - 5).collect();
            let last = c[n - 1].max(1);
            let w = 400;
            let mut s = BigFloat::zero(w);
            for i in 0..n - 1 {
                s = x[i]
                    .mul_add_round(&BigFloat::from(c[i]), &s, w, Round::Nearest)
                    .0;
            }
            x[n - 1] = (-s / last).with_prec(precs[n - 1]);
        }
        if x.iter().any(BigFloat::is_zero) {
            continue;
        }
        let max_norm = match rng.next() % 3 {
            0 => None,
            1 => Some(Number::from(rng.next() % 16 + 1)),
            _ => Some(Number::from(rng.next() % 100_000 + 1)),
        };
        let gamma = [1.16, 1.155, 1.5, 4.0][(rng.next() % 4) as usize];
        let max_iter = rng
            .next()
            .is_multiple_of(3)
            .then(|| (rng.next() % 25) as usize);
        let opts = PslqOptions::new()
            .with_error_bits(error_bits)
            .with_max_norm(max_norm.clone())
            .with_gamma(gamma)
            .with_max_iter(max_iter);
        let r = pslq(&x, &opts);
        if let Pslq::Relation(m) = &r {
            check_relation(&x, m, error_bits);
            out.relations += 1;
            continue;
        }
        let b = r.bound().unwrap().clone();
        assert!(b >= 1u32);
        out.bounds += 1;
        if let (Pslq::NoneBelow(_), Some(mn)) = (&r, &max_norm) {
            assert!(b >= *mn, "NoneBelow({b}) below {mn}");
        }
        if b < enumerate(n) {
            out.enumerated += 1;
            out.enumerated_mixed += usize::from(precs.iter().any(|&p| p != precs[0]));
            out.enumerated_iterations += usize::from(matches!(r, Pslq::IterationLimit(_)));
            let k = b.to_f64().floor() as i64;
            let mut m = vec![-k; n];
            loop {
                let nm: Vec<Number> = m.iter().map(|&v| Number::from(v)).collect();
                let nn = norm2(&nm);
                if !nn.is_zero() && b.clone() * &b > nn {
                    assert!(
                        !is_relation(&x, &nm, error_bits),
                        "{nm:?} is a relation of norm below {b}: {x:?}, {opts:?}"
                    );
                }
                let mut i = 0;
                while i < n && m[i] == k {
                    m[i] = -k;
                    i += 1;
                }
                if i == n {
                    break;
                }
                m[i] += 1;
            }
        }
    }
    out
}

#[test]
fn every_relation_found_is_one_and_every_small_bound_holds() {
    let rounds = if cfg!(miri) { 4 } else { 1000 };
    let limits = |n: usize| match n {
        2 => 64,
        3 => 12,
        4 => 7,
        _ => 4,
    };
    let s = sweep(0x5eed_9519, rounds, limits);
    eprintln!("{s:?}");
    assert!(s.relations > rounds / 4 && s.bounds > rounds / 10, "{s:?}");
    if !cfg!(miri) {
        assert!(
            s.enumerated > 50 && s.enumerated_mixed > 20 && s.enumerated_iterations > 20,
            "{s:?}"
        );
    }
}

// The same on 20 000 vectors, with bounds up to 96 enumerated for two numbers, 20 for three, 10
// for four and 7 for five: under a minute with `--release`, run for the doc page.
#[test]
#[ignore]
fn every_relation_found_is_one_and_every_small_bound_holds_on_20000_vectors() {
    let limits = |n: usize| match n {
        2 => 96,
        3 => 20,
        4 => 10,
        _ => 7,
    };
    let s = sweep(0x00c0_ffee, 20_000, limits);
    eprintln!("{s:?}");
    assert!(
        s.relations > 5000 && s.bounds > 2000 && s.enumerated > 1000,
        "{s:?}"
    );
    assert!(
        s.enumerated_mixed > 300 && s.enumerated_iterations > 300,
        "{s:?}"
    );
}

#[test]
#[should_panic(expected = "more than 2^40")]
fn the_span_runs_to_the_smallest_last_place() {
    // 1 at 53 bits is the least precise number, but the span runs from it to the last place of
    // the other, near 2^-(2^40), which the integers Xᵢ would have to hold.
    let tiny = BigFloat::from_parts(&Number::from(1), -((1i64 << 40) - 100), 200, Round::Nearest).0;
    let _ = pslq(&[BigFloat::from(1.0), tiny], &PslqOptions::new());
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_bound_grows_with_max_norm_and_never_passes_the_minimal_polynomial() {
    // 1, α, …, α^12 have the relations k·m of the minimal polynomial m alone, so every bound
    // proved must stay below |m|, and NoneBelow must reach the limit.
    let (x, _) = case("deg12");
    let m = ints(&[73, -144, -540, -108, 12, -288, 54, 0, -6, -12, 0, 0, 1]);
    let norm = BigFloat::from(&norm2(&m)).sqrt();
    let mut last = BigFloat::from(1);
    let mut found = false;
    for k in [1u32, 2, 5, 10, 20, 50, 100, 200, 400, 600, 700, 800, 1000] {
        let r = pslq(&x, &PslqOptions::new().with_max_norm(Some(Number::from(k))));
        match r {
            Pslq::NoneBelow(b) => {
                assert!(!found, "a bound after the relation");
                assert!(b >= k, "{b} below {k}");
                assert!(b <= norm, "{b} above the norm {norm} of a relation");
                assert!(b >= last, "{b} below {last}");
                last = b;
            }
            Pslq::Relation(r) => {
                assert_eq!(r, m);
                found = true;
            }
            other => panic!("{other:?}"),
        }
    }
    assert!(found && last > 100u32, "{last}");
}

#[test]
fn zeros_give_their_unit_vectors() {
    let x = [
        BigFloat::pi(100),
        BigFloat::neg_zero(100),
        BigFloat::zero(5),
    ];
    assert_eq!(
        pslq(&x, &PslqOptions::new()),
        Pslq::Relation(ints(&[0, 1, 0]))
    );
    let x = [BigFloat::zero(53), BigFloat::zero(53)];
    assert_eq!(pslq(&x, &PslqOptions::new()), Pslq::Relation(ints(&[1, 0])));
    // A number whose errors reach zero is a relation by itself too: 3 bits, 8 of them wrong.
    let x = [BigFloat::pi(100), BigFloat::from(5).with_prec(3)];
    assert_eq!(pslq(&x, &PslqOptions::new()), Pslq::Relation(ints(&[0, 1])));
    // 4 at 9 bits: |4| = 2^8 ulps (2^-6) exactly, within them; at 20 bits, far from it.
    let x = [BigFloat::pi(100), BigFloat::from(4).with_prec(9)];
    assert_eq!(pslq(&x, &PslqOptions::new()), Pslq::Relation(ints(&[0, 1])));
    let x = [
        BigFloat::from(3).with_prec(100),
        BigFloat::from(4).with_prec(20),
    ];
    assert_eq!(
        pslq(&x, &PslqOptions::new()),
        Pslq::Relation(ints(&[4, -3]))
    );
    // Every bit wrong: every number is a relation by itself.
    let all = PslqOptions::new().with_error_bits(u32::MAX);
    assert_eq!(
        pslq(&[BigFloat::pi(64), BigFloat::e(64)], &all),
        Pslq::Relation(ints(&[1, 0]))
    );
}

#[test]
fn integers_give_their_relations() {
    let o = PslqOptions::new();
    let f = |v: &[i64]| v.iter().map(|&k| BigFloat::from(k)).collect::<Vec<_>>();
    assert_eq!(pslq(&f(&[6, -4]), &o), Pslq::Relation(ints(&[2, 3])));
    assert_eq!(pslq(&f(&[1, 1]), &o), Pslq::Relation(ints(&[1, -1])));
    assert_eq!(pslq(&f(&[-7, 7]), &o), Pslq::Relation(ints(&[1, 1])));
    let x = f(&[6, 10, 15]);
    let Pslq::Relation(m) = pslq(&x, &o) else {
        panic!()
    };
    check_relation(&x, &m, 8);
    assert!(norm2(&m) <= 13, "{m:?}");
    // Large integers, exactly: 2^200 + 1 and 2^200 + 3 at their 201 bits.
    let a = (Number::from(1) << 200) + Number::from(1);
    let b = (Number::from(1) << 200) + Number::from(3);
    let x = [
        BigFloat::from(&a),
        BigFloat::from(&b),
        BigFloat::from(1).with_prec(201),
    ];
    let Pslq::Relation(m) = pslq(&x, &o) else {
        panic!()
    };
    check_relation(&x, &m, 8);
}

#[test]
fn huge_and_tiny_exponents_scale_away() {
    let o = PslqOptions::new();
    let at = |m: i64, e: i64| BigFloat::from_parts(&Number::from(m), e, 64, Round::Nearest).0;
    for e in [1i64 << 40, 1 << 61, -(1 << 50), -(1 << 61)] {
        let x = [at(3, e), at(5, e)];
        assert_eq!(pslq(&x, &o), Pslq::Relation(ints(&[5, -3])), "{e}");
        let x = [at(3, e), at(5, e), at(7, e)];
        assert_eq!(pslq(&x, &o), Pslq::Relation(ints(&[1, -2, 1])), "{e}");
    }
    // 1 and 2^-1000 at 53 bits, and π and π·2^-300 at 100 bits: relations (1, −k) with k
    // within the errors of 2^1000 and 2^300, k·2^-1000 = 1 to 2^-43.
    let x = [BigFloat::from(1.0), at(1, -1000).with_prec(53)];
    let Pslq::Relation(m) = pslq(&x, &o) else {
        panic!()
    };
    check_relation(&x, &m, 8);
    assert_eq!(
        (m[0].clone(), m[1].bits()),
        (Number::from(1), 1001),
        "{m:?}"
    );
    let pi = BigFloat::pi(100);
    let (mant, e) = pi.mantissa_exponent().unwrap();
    let x = [
        pi,
        BigFloat::from_parts(&mant, e - 300, 100, Round::Nearest).0,
    ];
    let Pslq::Relation(m) = pslq(&x, &o) else {
        panic!()
    };
    check_relation(&x, &m, 8);
    assert_eq!((m[0].clone(), m[1].bits()), (Number::from(1), 301), "{m:?}");
}

#[test]
fn each_number_is_taken_to_its_own_precision() {
    let o = PslqOptions::new();
    // π to 53 bits is π to 200 bits within the 53-bit errors.
    let x = [BigFloat::pi(53), BigFloat::e(200), BigFloat::pi(200)];
    assert_eq!(pslq(&x, &o), Pslq::Relation(ints(&[1, 0, -1])));
    // An integer is no more exact than its precision: 1 at 32 bits next to √2 at 200 bits gives
    // a relation to 32 bits (a rational approximation of √2), and at 200 bits none below 10¹⁵.
    let sqrt2 = BigFloat::from(2).with_prec(200).sqrt();
    let x = [BigFloat::from(1), sqrt2.clone()];
    let Pslq::Relation(m) = pslq(&x, &o) else {
        panic!()
    };
    check_relation(&x, &m, 8);
    assert!(m[0].bits() < 20, "{m:?}");
    let x = [BigFloat::from(1).with_prec(200), sqrt2];
    let opts = o.clone().with_max_norm(Some(Number::from(10).pow(15)));
    assert!(matches!(pslq(&x, &opts), Pslq::NoneBelow(_)));
}

#[test]
fn error_bits_set_the_tolerance() {
    // π and π + 2^-90 at 100 bits (ulp 2^-98) differ by 2^-90: (1, −1) is a relation within
    // 2^7 ulps of each, and not within 2^6, where the first relation PSLQ meets has entries
    // near π·2^90 (π − k·2^-90 ≈ 0).
    let pi = BigFloat::pi(100);
    let step = BigFloat::from_parts(&Number::from(1), -90, 1, Round::Nearest).0;
    let x = [pi.clone(), pi.add_round(&step, 100, Round::Nearest).0];
    for error_bits in 0..=10 {
        let opts = PslqOptions::new().with_error_bits(error_bits);
        let Pslq::Relation(m) = pslq(&x, &opts) else {
            panic!()
        };
        check_relation(&x, &m, error_bits);
        if error_bits >= 7 {
            assert_eq!(m, ints(&[1, -1]), "{error_bits}");
        } else {
            assert!(!is_relation(&x, &ints(&[1, -1]), error_bits));
            assert!(m[0].bits() > 80, "{error_bits}: {m:?}");
        }
    }
}

#[test]
fn f64_numbers_are_taken_at_53_bits() {
    let o = PslqOptions::new();
    let x = [
        core::f64::consts::FRAC_PI_4,
        (1.0f64 / 5.0).atan(),
        (1.0f64 / 239.0).atan(),
    ];
    assert_eq!(pslq_f64(&x, &o), Pslq::Relation(ints(&[1, -4, 1])));
    let x = [1.0, 2f64.sqrt(), 3f64.sqrt(), 6f64.sqrt()];
    let opts = o.clone().with_max_norm(Some(Number::from(20)));
    let Pslq::NoneBelow(b) = pslq_f64(&x, &opts) else {
        panic!()
    };
    assert!(b >= 20u32);
    assert_eq!(pslq_f64(&[0.1, 0.3], &o), Pslq::Relation(ints(&[3, -1])));
}

#[test]
fn the_limits_stop_the_search() {
    let x = [
        BigFloat::pi(200),
        BigFloat::e(200),
        BigFloat::from(1).with_prec(200),
    ];
    // No iteration: the bound of the first reduction.
    let r = pslq(&x, &PslqOptions::new().with_max_iter(Some(0)));
    let Pslq::IterationLimit(b) = r else {
        panic!("{r:?}")
    };
    assert!(b >= 1u32, "{b}");
    assert!(b < 10u32, "{b}");
    let r = pslq(&x, &PslqOptions::new().with_max_iter(Some(20)));
    let Pslq::IterationLimit(b20) = r else {
        panic!("{r:?}")
    };
    assert!(b20 > b);
    // A limit of 0 or 1, which every relation passes: the bound of the first reduction.
    for k in [0, 1] {
        let opts = PslqOptions::new().with_max_norm(Some(Number::from(k)));
        assert_eq!(pslq(&x, &opts), Pslq::NoneBelow(b.clone()));
    }
    // The limit stops the search before (2, 3) turns up, and a zero is a relation all the same.
    let opts = PslqOptions::new().with_max_norm(Some(Number::from(0)));
    let x = [BigFloat::from(6), BigFloat::from(-4)];
    assert!(matches!(pslq(&x, &opts), Pslq::NoneBelow(_)));
    assert_eq!(pslq(&x, &PslqOptions::new()), Pslq::Relation(ints(&[2, 3])));
    let x = [BigFloat::from(6), BigFloat::zero(1)];
    assert_eq!(pslq(&x, &opts), Pslq::Relation(ints(&[0, 1])));
}

#[test]
#[cfg_attr(miri, ignore)]
fn gamma_changes_the_path_not_the_answer() {
    let (x, _) = case("deg12");
    let m = ints(&[73, -144, -540, -108, 12, -288, 54, 0, -6, -12, 0, 0, 1]);
    for g in [1.155, 1.2, 1.5, 2.0] {
        let r = pslq(&x, &PslqOptions::new().with_gamma(g));
        assert_eq!(r, Pslq::Relation(m.clone()), "γ = {g}");
    }
}

#[test]
fn the_options_have_their_defaults() {
    let o = PslqOptions::default();
    assert_eq!(o, PslqOptions::new());
    assert_eq!(
        (o.max_norm, o.max_iter, o.gamma, o.error_bits),
        (None, None, 1.16, 8)
    );
    let r = Pslq::NoneBelow(BigFloat::from(3));
    assert_eq!(r.bound(), Some(&BigFloat::from(3)));
    assert_eq!(r.relation(), None);
}

#[test]
#[should_panic(expected = "at least two numbers")]
fn one_number_panics() {
    let _ = pslq(&[BigFloat::pi(53)], &PslqOptions::new());
}

#[test]
#[should_panic(expected = "finite numbers")]
fn a_nan_panics() {
    let _ = pslq(&[BigFloat::pi(53), BigFloat::nan(53)], &PslqOptions::new());
}

#[test]
#[should_panic(expected = "finite numbers")]
fn an_infinity_panics() {
    let _ = pslq_f64(&[1.0, f64::INFINITY], &PslqOptions::new());
}

#[test]
#[should_panic(expected = "not above √(4/3)")]
fn gamma_at_root_four_thirds_panics() {
    // The f64 nearest √(4/3) is below it.
    let g = (4.0f64 / 3.0).sqrt();
    let _ = pslq_f64(&[1.0, 2.0], &PslqOptions::new().with_gamma(g));
}

#[test]
#[should_panic(expected = "γ = NaN is not finite")]
fn a_nan_gamma_panics() {
    let _ = pslq_f64(&[1.0, 2.0], &PslqOptions::new().with_gamma(f64::NAN));
}

#[test]
#[should_panic(expected = "γ = inf is not finite")]
fn an_infinite_gamma_panics() {
    let _ = pslq_f64(&[1.0, 2.0], &PslqOptions::new().with_gamma(f64::INFINITY));
}

#[test]
fn the_largest_finite_gamma_is_taken() {
    let r = pslq_f64(&[1.0, 2.0], &PslqOptions::new().with_gamma(f64::MAX));
    assert_eq!(r, Pslq::Relation(ints(&[2, -1])));
}

#[test]
#[should_panic(expected = "is negative")]
fn a_negative_max_norm_panics() {
    let opts = PslqOptions::new().with_max_norm(Some(Number::from(-1)));
    let _ = pslq_f64(&[1.0, 2.0], &opts);
}

#[test]
#[should_panic(expected = "more than 2^40")]
fn numbers_spanning_more_than_2_40_bits_panic() {
    let big = BigFloat::from_parts(&Number::from(1), 1 << 40, 53, Round::Nearest).0;
    let _ = pslq(&[big, BigFloat::from(1.0)], &PslqOptions::new());
}

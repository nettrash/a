//! Factorisation modulo primes and over the integers (group factor).

use crate::core::*;
use crate::w::{P50, P63, P64, ours, rand_monic, run_case};
use crate::z::{rand_z, zrefs};
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};
use std::hint::black_box;

fn out_mfac(lc: &Number, fs: &[(ModPolynomial, u32)]) -> Vec<u64> {
    out_fac(fp_number(lc), fs, |f| {
        let o = out_mod(f);
        (o[0], o[1])
    })
}
fn out_zfac(c: &Number, fs: &[(Polynomial<Number>, u32)]) -> Vec<u64> {
    out_fac(fp_number(c), fs, |f| {
        let o = out_z(f);
        (o[0], o[1])
    })
}

fn wpoly(c: &[u64], p: u64) -> ModPolynomial {
    ModPolynomial::from_u64(c, p)
}
fn wmul(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    (&wpoly(a, p) * &wpoly(b, p)).word_coeffs().unwrap().to_vec()
}

/// A random monic irreducible polynomial of degree d, by trial (ours decides).
fn irreducible_w(r: &mut Rng, d: usize, p: u64) -> Vec<u64> {
    loop {
        let c = rand_monic(r, d + 1, p);
        if wpoly(&c, p).is_irreducible() {
            return c;
        }
    }
}

/// Swinnerton-Dyer's polynomial for the first k primes: degree 2^k, irreducible over Z.
pub fn swinnerton_dyer(k: usize) -> Vec<Big> {
    let primes = [2u64, 3, 5, 7, 11, 13];
    let num = |x: i64| Number::from(x);
    // S(x) = x^2 - 2
    let mut s: Vec<Number> = vec![num(-2), num(0), num(1)];
    for &p in &primes[1..k] {
        // T(x) = S(x + sqrt p) = A(x) + sqrt(p) B(x), by Horner on pairs (u + v sqrt p)
        let pn = Number::from(p);
        let mut a: Vec<Number> = vec![];
        let mut b: Vec<Number> = vec![];
        for c in s.iter().rev() {
            // (a + b sqrt p) * (x + sqrt p) = (a x + p b) + sqrt p (b x + a)
            let mut na = vec![num(0); a.len() + 1];
            let mut nb = vec![num(0); b.len().max(a.len()) + 1];
            for (i, x) in a.iter().enumerate() {
                na[i + 1] = &na[i + 1] + x;
                nb[i] = &nb[i] + x;
            }
            for (i, y) in b.iter().enumerate() {
                na[i] = &na[i] + &(y * &pn);
                nb[i + 1] = &nb[i + 1] + y;
            }
            na[0] = &na[0] + c;
            a = na;
            b = nb;
        }
        let pa = Polynomial::new(a);
        let pb = Polynomial::new(b);
        let r = &(&pa * &pa) - &(&(&pb * &pb) * &Polynomial::new(vec![pn]));
        s = r.coeffs().to_vec();
    }
    s.iter().map(from_number).collect()
}

pub fn factor(out: &mut Out) {
    if !out.wants("factor") {
        return;
    }
    let mut r = Rng(0x5EED_000A);
    let fp: [(u64, &str); 4] = [(3, "3"), (P50, "2^50-27"), (P63, "2^63-25"), (P64, "2^64-59")];
    for (p, ps) in fp {
        for &d in &[10usize, 30, 100, 300, 1000] {
            if out.skip(d) {
                continue;
            }
            let v = if d <= 30 { 8 } else if d <= 100 { 2 } else { 1 };
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(rand_monic(&mut r, d + 1, p))]).collect();
            let buf = encode(&[Arg::U(p)], &vars);
            let st = crate::w::wpolys(&vars, p);
            let o = ours!(st, v, |s, i| s[i][0].factor(), |f| out_mfac(&f.0, &f.1));
            run_case(
                out,
                "factor",
                "ModPolynomial::factor (word p)",
                format!("p={ps} deg={d} random monic"),
                &buf,
                &zrefs("w_factor", Some("nmod_poly_factor"), Some("CanZass"), Some("Flx_factor")),
                o,
                Cmp::Exact,
                "",
            );
            if d <= 300 {
                let st = crate::w::wpolys(&vars, p);
                let o = ours!(st, v, |s, i| s[i][0].factor_berlekamp(), |f| out_mfac(&f.0, &f.1));
                run_case(
                    out,
                    "factor",
                    "ModPolynomial::factor_berlekamp (word p)",
                    format!("p={ps} deg={d} random monic"),
                    &buf,
                    &zrefs("w_factor_berlekamp", Some("nmod_poly_factor_with_berlekamp"), Some("berlekamp"), None),
                    o,
                    Cmp::Exact,
                    "",
                );
            }
            let st = crate::w::wpolys(&vars, p);
            let o = ours!(st, v, |s, i| s[i][0].roots(), |x| {
                let mut w = vec![x.len() as u64];
                for (z, e) in &x {
                    w.push(u64::try_from(z).unwrap());
                    w.push(*e as u64);
                }
                w
            });
            run_case(
                out,
                "factor",
                "ModPolynomial::roots (word p)",
                format!("p={ps} deg={d} random monic"),
                &buf,
                &zrefs("w_roots", Some("nmod_poly_roots (with multiplicity)"), None, Some("Flx_roots (distinct)")),
                o,
                Cmp::RefZeroWild,
                "PARI gives the distinct roots without multiplicities",
            );
            if d <= 300 {
                let v = if d <= 30 { 8 } else if d <= 100 { 2 } else { 1 };
                let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(irreducible_w(&mut r, d, p))]).collect();
                let buf = encode(&[Arg::U(p)], &vars);
                let st = crate::w::wpolys(&vars, p);
                let o = ours!(st, v, |s, i| s[i][0].is_irreducible(), |b| vec![b as u64]);
                run_case(
                    out,
                    "factor",
                    "ModPolynomial::is_irreducible (word p)",
                    format!("p={ps} deg={d} irreducible"),
                    &buf,
                    &zrefs("w_is_irreducible", Some("nmod_poly_is_irreducible"), Some("DetIrredTest"), Some("Flx_is_irred")),
                    o,
                    Cmp::Exact,
                    "",
                );
            }
        }
    }
    // modulo 2
    for &d in &[100usize, 1000] {
        let v = if d <= 100 { 2 } else { 1 };
        let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(rand_monic(&mut r, d + 1, 2))]).collect();
        let buf = encode(&[Arg::U(2)], &vars);
        let st = crate::w::wpolys(&vars, 2);
        let o = ours!(st, v, |s, i| s[i][0].factor(), |f| out_mfac(&f.0, &f.1));
        run_case(
            out,
            "factor",
            "ModPolynomial::factor (word p)",
            format!("p=2 deg={d} random"),
            &buf,
            &zrefs("w_factor", Some("nmod_poly_factor"), Some("CanZass"), Some("F2x_factor")),
            o,
            Cmp::Exact,
            "",
        );
    }
    // square-free, distinct-degree and equal-degree factorisation modulo 2^50 - 27
    let p = P50;
    for &d in &[120usize, 1200] {
        let v = if d <= 120 { 2 } else { 1 };
        let vars: Vec<Vec<Arg>> = (0..v)
            .map(|_| {
                let f1 = rand_monic(&mut r, d / 2 + 1, p);
                let f2 = rand_monic(&mut r, d / 8 + 1, p);
                let f3 = rand_monic(&mut r, d / 12 + 1, p);
                let f22 = wmul(&f2, &f2, p);
                let f33 = wmul(&wmul(&f3, &f3, p), &f3, p);
                vec![Arg::W(wmul(&wmul(&f1, &f22, p), &f33, p))]
            })
            .collect();
        let buf = encode(&[Arg::U(p)], &vars);
        let st = crate::w::wpolys(&vars, p);
        let o = ours!(st, v, |s, i| s[i][0].squarefree(), |f| out_mfac(&f.0, &f.1));
        run_case(
            out,
            "factor",
            "ModPolynomial::squarefree",
            format!("p=2^50-27 deg={d} f1 f2^2 f3^3"),
            &buf,
            &zrefs("w_squarefree", Some("nmod_poly_factor_squarefree"), Some("SquareFreeDecomp"), Some("Flx_factor_squarefree")),
            o,
            Cmp::Exact,
            "",
        );
    }
    for &d in &[30usize, 100, 300] {
        let v = if d <= 30 { 8 } else if d <= 100 { 2 } else { 1 };
        let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(rand_monic(&mut r, d + 1, p))]).collect();
        let buf = encode(&[Arg::U(p)], &vars);
        let st = crate::w::wpolys(&vars, p);
        let o = ours!(st, v, |s, i| s[i][0].distinct_degree_factor(), |f| {
            let mut mix = 0;
            for (g, e) in &f {
                mix = fp_add(mix, fp_mix(fp_add(out_mod(g)[1], *e as u64)));
            }
            vec![f.len() as u64, mix]
        });
        run_case(
            out,
            "factor",
            "ModPolynomial::distinct_degree_factor",
            format!("p=2^50-27 deg={d} random monic"),
            &buf,
            &zrefs("w_ddf", Some("nmod_poly_factor_distinct_deg"), Some("DDF (with x^p mod f)"), Some("Flx_ddf")),
            o,
            Cmp::Exact,
            "NTL's x^p mod f is computed inside the timing",
        );
    }
    for (dd, k) in [(5usize, 10usize), (10, 10), (20, 5)] {
        let v = 2;
        let vars: Vec<Vec<Arg>> = (0..v)
            .map(|_| {
                let mut f = vec![1u64];
                for _ in 0..k {
                    let g = irreducible_w(&mut r, dd, p);
                    f = wmul(&f, &g, p);
                }
                vec![Arg::W(f)]
            })
            .collect();
        let buf = encode(&[Arg::U(p), Arg::U(dd as u64)], &vars);
        let st = crate::w::wpolys(&vars, p);
        let o = ours!(st, v, |s, i| s[i][0].equal_degree_factor(dd as u32), |f| {
            let mut mix = 0;
            for g in &f {
                mix = fp_add(mix, fp_mix(out_mod(g)[1]));
            }
            vec![f.len() as u64, mix]
        });
        run_case(
            out,
            "factor",
            "ModPolynomial::equal_degree_factor",
            format!("p=2^50-27 {k} factors of degree {dd}"),
            &buf,
            &zrefs("w_edf", Some("nmod_poly_factor_equal_deg"), Some("EDF (with x^p mod f)"), None),
            o,
            Cmp::Exact,
            "NTL's x^p mod f is computed inside the timing",
        );
    }
    // modulo primes of two limbs
    let mp = [(Big::pow2_minus(64, 0), 13u64, "2^64+13"), (Big::pow2_minus(127, 1), 0, "2^127-1")];
    for (base, add, ps) in mp {
        let pb = if add > 0 { from_number(&(&base.to_number() + &Number::from(add))) } else { base };
        let pn = pb.to_number();
        for &d in &[10usize, 30, 100] {
            let v = if d <= 30 { 4 } else { 1 };
            let monic = |r: &mut Rng| {
                let mut c = crate::m::rand_m(r, d + 1, &pb);
                c[d] = Big::from_u64(1);
                c
            };
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::IV(monic(&mut r))]).collect();
            let buf = encode(&[Arg::I(pb.clone())], &vars);
            let st = crate::m::mpolys(&vars, &pn);
            let o = ours!(st, v, |s, i| s[i][0].factor(), |f| out_mfac(&f.0, &f.1));
            run_case(
                out,
                "factor",
                "ModPolynomial::factor (multi-limb p)",
                format!("p={ps} deg={d} random monic"),
                &buf,
                &zrefs("m_factor", Some("fmpz_mod_poly_factor"), Some("CanZass"), Some("FpX_factor")),
                o,
                Cmp::Exact,
                "",
            );
            if ps == "2^127-1" {
                let st = crate::m::mpolys(&vars, &pn);
                let o = ours!(st, v, |s, i| s[i][0].roots(), |x| {
                    let mut mix = 0;
                    let mut se = 0;
                    for (z, e) in &x {
                        se += *e as u64;
                        mix = fp_add(mix, fp_mul(*e as u64, fp_mix(fp_number(z))));
                    }
                    vec![x.len() as u64, se, mix]
                });
                run_case(
                    out,
                    "factor",
                    "ModPolynomial::roots (multi-limb p)",
                    format!("p={ps} deg={d} random monic"),
                    &buf,
                    &zrefs("m_roots", Some("fmpz_mod_poly_roots (with multiplicity)"), None, Some("FpX_roots (distinct)")),
                    o,
                    Cmp::RefZeroWild,
                    "PARI gives the distinct roots without multiplicities",
                );
                // irreducible inputs
                let vars: Vec<Vec<Arg>> = (0..v)
                    .map(|_| loop {
                        let c = monic(&mut r);
                        if ModPolynomial::new(&numbers(&c), &pn).is_irreducible() {
                            break vec![Arg::IV(c)];
                        }
                    })
                    .collect();
                let buf = encode(&[Arg::I(pb.clone())], &vars);
                let st = crate::m::mpolys(&vars, &pn);
                let o = ours!(st, v, |s, i| s[i][0].is_irreducible(), |b| vec![b as u64]);
                run_case(
                    out,
                    "factor",
                    "ModPolynomial::is_irreducible (multi-limb p)",
                    format!("p={ps} deg={d} irreducible"),
                    &buf,
                    &zrefs("m_is_irreducible", Some("fmpz_mod_poly_is_irreducible"), Some("DetIrredTest"), Some("FpX_is_irred")),
                    o,
                    Cmp::Exact,
                    "",
                );
            }
        }
    }
    // over the integers
    let zmulv = |a: &[Big], b: &[Big]| -> Vec<Big> { (&zpoly(a) * &zpoly(b)).coeffs().iter().map(from_number).collect() };
    let mut cases: Vec<(String, Vec<Vec<Big>>)> = Vec::new();
    for (b, degs) in [(8u32, &[10usize, 30, 100, 300][..]), (64, &[10, 30, 100]), (1000, &[10, 30])] {
        for &d in degs {
            let v = if d <= 30 { 4 } else { 1 };
            cases.push((format!("random, {b}-bit, deg {d}"), (0..v).map(|_| rand_z(&mut r, d + 1, b)).collect()));
        }
    }
    for (b, degs) in [(8u32, &[10usize, 30, 100, 300][..]), (64, &[30, 100]), (1000, &[20]), (10000, &[10])] {
        for &d in degs {
            let v = if d <= 30 { 4 } else { 1 };
            cases.push((
                format!("product of two, {b}-bit, deg {d}"),
                (0..v).map(|_| zmulv(&rand_z(&mut r, d / 2 + 1, b), &rand_z(&mut r, d - d / 2 + 1, b))).collect(),
            ));
        }
    }
    for n in [10usize, 20, 50] {
        cases.push((format!("prod (x - i), deg {n}"), vec![crate::roots::wilkinson(n)]));
    }
    for k in [3usize, 4, 5] {
        cases.push((format!("Swinnerton-Dyer, deg {}", 1 << k), vec![swinnerton_dyer(k)]));
    }
    for n in [30usize, 60, 105, 120, 210] {
        let mut c = vec![Big::from_u64(0); n + 1];
        c[0] = Big::from_i64(-1);
        c[n] = Big::from_u64(1);
        cases.push((format!("x^n - 1, deg {n}"), vec![c]));
    }
    for (name, polys) in &cases {
        let v = polys.len();
        let vars: Vec<Vec<Arg>> = polys.iter().map(|p| vec![Arg::IV(p.clone())]).collect();
        let buf = encode(&[], &vars);
        let st: Vec<Polynomial<Number>> = polys.iter().map(|p| zpoly(p)).collect();
        let o = ours!(st, v, |s, i| s[i].factor(), |f| out_zfac(&f.0, &f.1));
        run_case(
            out,
            "factor",
            "Polynomial<Number>::factor",
            name.clone(),
            &buf,
            &zrefs("z_factor", Some("fmpz_poly_factor"), Some("factor (ZZXFactoring)"), Some("ZX_factor")),
            o,
            Cmp::Exact,
            "",
        );
    }
    // square-free decomposition over Z: f1 f2^2 f3^3, primitive, positive leading coefficients
    for &d in &[30usize, 300] {
        let v = if d <= 30 { 4 } else { 1 };
        let pos = |mut c: Vec<Big>| {
            if let Some(t) = c.last_mut() {
                t.neg = false;
            }
            c
        };
        let vars: Vec<Vec<Arg>> = (0..v)
            .map(|_| {
                let f1 = pos(rand_z(&mut r, d / 2 + 1, 64));
                let f2 = pos(rand_z(&mut r, d / 8 + 1, 64));
                let f3 = pos(rand_z(&mut r, d / 12 + 1, 64));
                let p = zmulv(&zmulv(&f1, &zmulv(&f2, &f2)), &zmulv(&f3, &zmulv(&f3, &f3)));
                vec![Arg::IV(p)]
            })
            .collect();
        let buf = encode(&[], &vars);
        let st: Vec<Polynomial<Number>> = vars.iter().map(|x| zpoly(x[0].iv())).collect();
        let o = ours!(st, v, |s, i| s[i].squarefree(), |f| out_zfac(&f.0, &f.1));
        run_case(
            out,
            "factor",
            "Polynomial<Number>::squarefree",
            format!("64-bit f1 f2^2 f3^3, deg {d}"),
            &buf,
            &zrefs(
                "z_squarefree",
                Some("fmpz_poly_factor_squarefree"),
                Some("content + SquareFreeDecomp"),
                Some("ZX_squff"),
            ),
            o,
            Cmp::Exact,
            "",
        );
    }
    for name in ["random, 8-bit, deg 30", "random, 8-bit, deg 100", "Swinnerton-Dyer, deg 16", "Swinnerton-Dyer, deg 32"] {
        let polys = &cases.iter().find(|c| c.0 == name).unwrap().1;
        let v = polys.len();
        let vars: Vec<Vec<Arg>> = polys.iter().map(|p| vec![Arg::IV(p.clone())]).collect();
        let buf = encode(&[], &vars);
        let st: Vec<Polynomial<Number>> = polys.iter().map(|p| zpoly(p)).collect();
        let o = ours!(st, v, |s, i| s[i].is_irreducible(), |b| vec![b as u64]);
        run_case(
            out,
            "factor",
            "Polynomial<Number>::is_irreducible",
            name.to_string(),
            &buf,
            &zrefs("z_is_irreducible", None, None, Some("ZX_is_irred (polisirreducible)")),
            o,
            Cmp::Exact,
            "",
        );
    }
    let _ = black_box(0);
}

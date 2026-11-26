//! Evaluation at many points, products of linear factors and interpolation (group eval).

use crate::core::*;
use crate::w::{P50, P63, P64, ours, rand_w_lc, run_case};
use crate::z::{rand_z, zrefs};
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;
use std::collections::HashSet;
use std::hint::black_box;

pub(crate) fn distinct_w(r: &mut Rng, n: usize, p: u64) -> Vec<u64> {
    let mut s = HashSet::new();
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let x = r.below(p);
        if s.insert(x) {
            v.push(x);
        }
    }
    v
}
pub(crate) fn distinct_m(r: &mut Rng, n: usize, p: &Big) -> Vec<Big> {
    let mut s = HashSet::new();
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let x = r.below_big(p);
        if s.insert(x.l.clone()) {
            v.push(x);
        }
    }
    v
}
fn distinct_z(r: &mut Rng, n: usize, bits: u32) -> Vec<Big> {
    let mut s = HashSet::new();
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let x = r.int(bits);
        if s.insert((x.neg, x.l.clone())) {
            v.push(x);
        }
    }
    v
}
pub(crate) fn out_nums(v: &[Number]) -> Vec<u64> {
    vec![v.len() as u64, fp_nums(v)]
}
fn out_words(v: &[u64]) -> Vec<u64> {
    vec![v.len() as u64, fp_words(v)]
}

pub fn eval(out: &mut Out) {
    if !out.wants("eval") {
        return;
    }
    let mut r = Rng(0x5EED_0008);
    let p20 = Number::from(1u64 << 20).prev_prime().unwrap();
    let p20: u64 = u64::try_from(&p20).unwrap();
    let wp = [(p20, "p20"), (P50, "p50"), (P63, "p63"), (P64, "p64")];
    for (p, ps) in wp {
        let pn = Number::from(p);
        for &n in &[1usize, 4, 16, 64, 256, 1024, 4096, 16384, 65536] {
            if out.skip(n) {
                continue;
            }
            let v = variants(n);
            // eval_many_u64: n coefficients at n points
            let vars: Vec<Vec<Arg>> =
                (0..v).map(|_| vec![Arg::W(rand_w_lc(&mut r, n, p)), Arg::W(distinct_w(&mut r, n, p))]).collect();
            let buf = encode(&[Arg::U(p)], &vars);
            let st: Vec<(ModPolynomial, Vec<u64>)> =
                vars.iter().map(|x| (ModPolynomial::from_u64(x[0].w(), p), x[1].w().to_vec())).collect();
            let o = ours!(st, v, |s, i| s[i].0.eval_many_u64(&s[i].1), |y| out_words(&y));
            let mut refs = vec![
                (Lib::Flint, "w_eval_many".to_string(), "nmod_poly_evaluate_nmod_vec".to_string()),
                (Lib::Flint, "w_eval_many_fast".to_string(), "nmod_poly_evaluate_nmod_vec_fast".to_string()),
            ];
            if n <= 4096 {
                refs.push((Lib::Ntl, "w_eval_many".into(), "eval(vec) (Horner)".into()));
            }
            refs.push((Lib::Pari, "w_eval_many".into(), "Flx_Flv_multieval".into()));
            run_case(
                out,
                "eval",
                "ModPolynomial::eval_many_u64",
                format!("p={ps} n={n}"),
                &buf,
                &refs,
                o,
                Cmp::Exact,
                "",
            );
            // from_roots_u64
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(crate::w::rand_w(&mut r, n, p))]).collect();
            let buf = encode(&[Arg::U(p)], &vars);
            let st: Vec<Vec<u64>> = vars.iter().map(|x| x[0].w().to_vec()).collect();
            let o = ours!(st, v, |s, i| ModPolynomial::from_roots_u64(&s[i], p), |q| out_mod(&q).to_vec());
            run_case(
                out,
                "eval",
                "ModPolynomial::from_roots_u64",
                format!("p={ps} n={n}"),
                &buf,
                &zrefs(
                    "w_from_roots",
                    Some("nmod_poly_product_roots_nmod_vec"),
                    Some("BuildFromRoots"),
                    Some("Flv_roots_to_pol"),
                ),
                o,
                Cmp::Exact,
                "",
            );
            // interpolate_u64
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::W(distinct_w(&mut r, n, p)), Arg::W(crate::w::rand_w(&mut r, n, p))])
                .collect();
            let buf = encode(&[Arg::U(p)], &vars);
            let st: Vec<(Vec<u64>, Vec<u64>)> = vars.iter().map(|x| (x[0].w().to_vec(), x[1].w().to_vec())).collect();
            let o = ours!(st, v, |s, i| ModPolynomial::interpolate_u64(&s[i].0, &s[i].1, p), |q| out_mod(&q).to_vec());
            run_case(
                out,
                "eval",
                "ModPolynomial::interpolate_u64",
                format!("p={ps} n={n}"),
                &buf,
                &zrefs(
                    "w_interpolate",
                    Some("nmod_poly_interpolate_nmod_vec_fast"),
                    // NTL's ZZ_pX interpolate (moduli above 2^60) is quadratic: minutes at 65536
                    if p < (1 << 60) || n <= 4096 { Some("interpolate") } else { None },
                    Some("Flv_polint"),
                ),
                o,
                Cmp::Exact,
                "",
            );
            if n <= 4096 {
                let st: Vec<(Vec<Number>, Vec<Number>)> = vars
                    .iter()
                    .map(|x| {
                        (
                            x[0].w().iter().map(|&a| Number::from(a)).collect(),
                            x[1].w().iter().map(|&a| Number::from(a)).collect(),
                        )
                    })
                    .collect();
                let st = (st, pn.clone());
                let o = ours!(st, v, |s, i| ModPolynomial::interpolate_newton(&s.0[i].0, &s.0[i].1, &s.1), |q| out_mod(&q)
                    .to_vec());
                run_case(
                    out,
                    "eval",
                    "ModPolynomial::interpolate_newton",
                    format!("p={ps} n={n}"),
                    &buf,
                    &zrefs("w_interp_newton", Some("nmod_poly_interpolate_nmod_vec_newton"), None, None),
                    o,
                    Cmp::Exact,
                    "ours through the Number interface (inputs as Numbers built before the clock)",
                );
                let st: Vec<(Vec<Number>, Vec<Number>)> = vars
                    .iter()
                    .map(|x| {
                        (
                            x[0].w().iter().map(|&a| Number::from(a)).collect(),
                            x[1].w().iter().map(|&a| Number::from(a)).collect(),
                        )
                    })
                    .collect();
                let st = (st, pn.clone());
                let o = ours!(st, v, |s, i| ModPolynomial::interpolate_lagrange(&s.0[i].0, &s.0[i].1, &s.1), |q| out_mod(&q)
                    .to_vec());
                run_case(
                    out,
                    "eval",
                    "ModPolynomial::interpolate_lagrange",
                    format!("p={ps} n={n}"),
                    &buf,
                    &zrefs("w_interp_lagrange", Some("nmod_poly_interpolate_nmod_vec_barycentric"), None, None),
                    o,
                    Cmp::Exact,
                    "ours through the Number interface (inputs as Numbers built before the clock)",
                );
            }
        }
    }
    // modulo p127, p130, p521 (the Number interface)
    let mp = [
        (Big::pow2_minus(127, 1), "p127"),
        (Big::pow2_minus(130, 5), "p130"),
        (Big::pow2_minus(521, 1), "p521"),
    ];
    for (pb, ps) in mp {
        let pn = pb.to_number();
        for &n in &[1usize, 4, 16, 64, 256, 1024, 4096] {
            if out.skip(n) {
                continue;
            }
            let v = variants(n);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::IV(crate::m::rand_m_lc(&mut r, n, &pb)), Arg::IV(distinct_m(&mut r, n, &pb))])
                .collect();
            let buf = encode(&[Arg::I(pb.clone())], &vars);
            let st: Vec<(ModPolynomial, Vec<Number>)> =
                vars.iter().map(|x| (ModPolynomial::new(&numbers(x[0].iv()), &pn), numbers(x[1].iv()))).collect();
            let o = ours!(st, v, |s, i| s[i].0.eval_many(&s[i].1), |y| out_nums(&y));
            let mut refs = vec![
                (Lib::Flint, "m_eval_many".to_string(), "fmpz_mod_poly_evaluate_fmpz_vec".to_string()),
                (Lib::Flint, "m_eval_many_fast".to_string(), "fmpz_mod_poly_evaluate_fmpz_vec_fast".to_string()),
            ];
            if n <= 1024 {
                refs.push((Lib::Ntl, "m_eval_many".into(), "eval(vec) (Horner)".into()));
            }
            refs.push((Lib::Pari, "m_eval_many".into(), "FpX_FpV_multieval".into()));
            run_case(out, "eval", "ModPolynomial::eval_many", format!("p={ps} n={n}"), &buf, &refs, o, Cmp::Exact, "");
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::IV(crate::m::rand_m(&mut r, n, &pb))]).collect();
            let buf = encode(&[Arg::I(pb.clone())], &vars);
            let st: Vec<Vec<Number>> = vars.iter().map(|x| numbers(x[0].iv())).collect();
            let st = (st, pn.clone());
            let o = ours!(st, v, |s, i| ModPolynomial::from_roots(&s.0[i], &s.1), |q| out_mod(&q).to_vec());
            run_case(
                out,
                "eval",
                "ModPolynomial::from_roots",
                format!("p={ps} n={n}"),
                &buf,
                &zrefs(
                    "m_from_roots",
                    Some("fmpz_mod_poly_product_roots_fmpz_vec"),
                    Some("BuildFromRoots"),
                    Some("FpV_roots_to_pol"),
                ),
                o,
                Cmp::Exact,
                "",
            );
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::IV(distinct_m(&mut r, n, &pb)), Arg::IV(crate::m::rand_m(&mut r, n, &pb))])
                .collect();
            let buf = encode(&[Arg::I(pb.clone())], &vars);
            let st: Vec<(Vec<Number>, Vec<Number>)> =
                vars.iter().map(|x| (numbers(x[0].iv()), numbers(x[1].iv()))).collect();
            let st = (st, pn.clone());
            let o = ours!(st, v, |s, i| ModPolynomial::interpolate(&s.0[i].0, &s.0[i].1, &s.1), |q| out_mod(&q).to_vec());
            run_case(
                out,
                "eval",
                "ModPolynomial::interpolate",
                format!("p={ps} n={n}"),
                &buf,
                &zrefs("m_interpolate", None, Some("interpolate"), Some("FpV_polint")),
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    // over the integers: n coefficients of B bits at n points of C bits
    for (cb, xb) in [(64u32, 16u32), (64, 64), (1000, 64), (64, 1000)] {
        for &n in &[1usize, 4, 16, 64, 256, 1024] {
            let v = variants(n);
            let vars: Vec<Vec<Arg>> =
                (0..v).map(|_| vec![Arg::IV(rand_z(&mut r, n, cb)), Arg::IV(rand_z(&mut r, n, xb))]).collect();
            let buf = encode(&[], &vars);
            let st: Vec<(Polynomial<Number>, Vec<Number>)> =
                vars.iter().map(|x| (zpoly(x[0].iv()), numbers(x[1].iv()))).collect();
            let o = ours!(st, v, |s, i| s[i].0.eval_many(&s[i].1), |y| out_nums(&y));
            let mut refs = vec![(Lib::Flint, "z_eval_many".to_string(), "fmpz_poly_evaluate_fmpz_vec".to_string())];
            if n <= 256 {
                refs.push((Lib::Pari, "z_eval_many".into(), "ZX_Z_eval at each point".into()));
            }
            run_case(
                out,
                "eval",
                "Polynomial<Number>::eval_many",
                format!("c{cb}x{xb} n={n}"),
                &buf,
                &refs,
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    for &rb in &[16u32, 64, 1000] {
        for &n in &[1usize, 4, 16, 64, 256, 1024] {
            let v = variants(n);
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::IV(rand_z(&mut r, n, rb))]).collect();
            let buf = encode(&[], &vars);
            let st: Vec<Vec<Number>> = vars.iter().map(|x| numbers(x[0].iv())).collect();
            let o = ours!(st, v, |s, i| Polynomial::from_roots(&s[i]), |q| out_z(&q).to_vec());
            run_case(
                out,
                "eval",
                "Polynomial<Number>::from_roots",
                format!("roots={rb} bits n={n}"),
                &buf,
                &zrefs("z_from_roots", Some("fmpz_poly_product_roots_fmpz_vec"), None, Some("roots_to_pol")),
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    // Random points of 64 bits make the exact interpolant huge (a common denominator of about
    // 32 n^2 bits: 4 GB of coefficients at 1024 points), so those stop at 256.
    for (yb, xb) in [(64u32, 16u32), (64, 64), (1000, 64)] {
        for &n in &[1usize, 4, 16, 64, 256, 1024] {
            if xb == 64 && n > 256 {
                continue;
            }
            let v = variants(n);
            let vars: Vec<Vec<Arg>> =
                (0..v).map(|_| vec![Arg::IV(distinct_z(&mut r, n, xb)), Arg::IV(rand_z(&mut r, n, yb))]).collect();
            let buf = encode(&[], &vars);
            let st: Vec<(Vec<Rational>, Vec<Rational>)> =
                vars.iter().map(|x| (rationals_int(x[0].iv()), rationals_int(x[1].iv()))).collect();
            let o = ours!(st, v, |s, i| Polynomial::interpolate(&s[i].0, &s[i].1), |q| out_q(&q).to_vec());
            let mut refs =
                vec![(Lib::Flint, "q_interpolate".to_string(), "fmpq_poly_interpolate_fmpz_vec".to_string())];
            if n <= 64 {
                // RgV_polint over Q is the textbook Lagrange form in rationals: minutes at 256
                refs.push((Lib::Pari, "q_interpolate".into(), "RgV_polint".into()));
            }
            run_case(
                out,
                "eval",
                "Polynomial<Rational>::interpolate",
                format!("values {yb} bits at points {xb} bits n={n}"),
                &buf,
                &refs,
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    let _ = black_box(0);
}

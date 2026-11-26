//! Polynomials modulo a multi-limb n (group mlimb).

use crate::core::*;
use crate::w::{ours, refs_w, run_case};
use a::math::numbers::Number;
use a::math::poly::ModPolynomial;
use std::hint::black_box;

pub fn mods() -> Vec<(Big, &'static str)> {
    vec![
        (Big::pow2_minus(127, 1), "2^127-1"),
        (Big::pow2_minus(255, 19), "2^255-19"),
        (Big::pow2_minus(521, 1), "2^521-1"),
        (Big::pow2_minus(1024, 105), "2^1024-105"),
    ]
}

pub fn rand_m(r: &mut Rng, len: usize, n: &Big) -> Vec<Big> {
    (0..len).map(|_| r.below_big(n)).collect()
}
pub fn rand_m_lc(r: &mut Rng, len: usize, n: &Big) -> Vec<Big> {
    let mut v = rand_m(r, len, n);
    if let Some(t) = v.last_mut() {
        while t.l.is_empty() {
            *t = r.below_big(n);
        }
    }
    v
}

pub fn mpolys(vars: &[Vec<Arg>], n: &Number) -> Vec<Vec<ModPolynomial>> {
    vars.iter()
        .map(|v| v.iter().map(|a| ModPolynomial::new(&numbers(a.iv()), n)).collect())
        .collect()
}

fn mrefs(op: &str, flint: &str, ntl: Option<&str>, pari: Option<&str>) -> Vec<(Lib, String, String)> {
    refs_w(op, flint, ntl, pari)
}

pub fn mlimb(out: &mut Out) {
    if !out.wants("mlimb") {
        return;
    }
    let mut r = Rng(0x5EED_0003);
    for (nb, ns) in mods() {
        let n = nb.to_number();
        for &len in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 30000, 100000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::IV(rand_m_lc(&mut r, len, &nb)), Arg::IV(rand_m_lc(&mut r, len, &nb))])
                .collect();
            let buf = encode(&[Arg::I(nb.clone())], &vars);
            let st = mpolys(&vars, &n);
            let o = ours!(st, v, |s, i| &s[i][0] * &s[i][1], |p| out_mod(&p).to_vec());
            run_case(
                out,
                "mlimb",
                "ModPolynomial * (multi-limb n)",
                format!("n={ns} len={len}x{len}"),
                &buf,
                &mrefs("m_mul", "fmpz_mod_poly_mul", Some("mul"), Some("FpX_mul")),
                o,
                Cmp::Exact,
                "",
            );
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::IV(rand_m_lc(&mut r, len, &nb))]).collect();
            let buf = encode(&[Arg::I(nb.clone())], &vars);
            let st = mpolys(&vars, &n);
            let o = ours!(st, v, |s, i| &s[i][0] * &s[i][0], |p| out_mod(&p).to_vec());
            run_case(
                out,
                "mlimb",
                "ModPolynomial &a * &a (multi-limb n)",
                format!("n={ns} len={len}"),
                &buf,
                &mrefs("m_sqr", "fmpz_mod_poly_sqr", Some("sqr"), Some("FpX_sqr")),
                o,
                Cmp::Exact,
                "",
            );
        }
        for &len in &[4usize, 8, 16, 32, 64, 128, 256, 512, 1000, 2000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::IV(rand_m_lc(&mut r, 2 * len, &nb)), Arg::IV(rand_m_lc(&mut r, len, &nb))])
                .collect();
            let buf = encode(&[Arg::I(nb.clone())], &vars);
            let st = mpolys(&vars, &n);
            let o = ours!(st, v, |s, i| s[i][0].div_rem(&s[i][1]), |p| {
                let mut w = out_mod(&p.0).to_vec();
                w.extend(out_mod(&p.1));
                w
            });
            run_case(
                out,
                "mlimb",
                "ModPolynomial::div_rem (multi-limb n)",
                format!("n={ns} len={}/{len}", 2 * len),
                &buf,
                &mrefs("m_divrem", "fmpz_mod_poly_divrem", Some("DivRem"), Some("FpX_divrem")),
                o,
                Cmp::Exact,
                "",
            );
        }
        for &len in &[4usize, 8, 16, 32, 64, 128, 256, 512] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::IV(rand_m_lc(&mut r, len, &nb))]).collect();
            let buf = encode(&[Arg::I(nb.clone())], &vars);
            let st = (mpolys(&vars, &n), n.clone(), ModPolynomial::x(&n));
            let o = ours!(st, v, |s, i| s.2.pow_mod(&s.1, &s.0[i][0]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "mlimb",
                "ModPolynomial x.pow_mod(n, f) (multi-limb n)",
                format!("n={ns} len(f)={len} e=n"),
                &buf,
                &mrefs("m_powmod_x", "fmpz_mod_poly_powmod_x_fmpz_preinv", Some("PowerXMod"), Some("FpXQ_pow")),
                o,
                Cmp::Exact,
                "refs: FLINT's inverse of f and NTL's Modulus built before the clock",
            );
        }
        for &len in &[16usize, 256, 2000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            // inv_series
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    let mut a = rand_m(&mut r, len, &nb);
                    a[0] = Big::from_u64(1 + r.below(1000));
                    vec![Arg::IV(a)]
                })
                .collect();
            let buf = encode(&[Arg::I(nb.clone()), Arg::U(len as u64)], &vars);
            let st = mpolys(&vars, &n);
            let o = ours!(st, v, |s, i| s[i][0].inv_series(len), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "mlimb",
                "ModPolynomial::inv_series (multi-limb n)",
                format!("n={ns} len={len} terms={len}"),
                &buf,
                &mrefs("m_inv_series", "fmpz_mod_poly_inv_series", Some("InvTrunc"), Some("FpXn_inv")),
                o,
                Cmp::Exact,
                "",
            );
            // mul_mod and compose_mod
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    vec![
                        Arg::IV(rand_m(&mut r, len - 1, &nb)),
                        Arg::IV(rand_m(&mut r, len - 1, &nb)),
                        Arg::IV(rand_m_lc(&mut r, len, &nb)),
                    ]
                })
                .collect();
            let buf = encode(&[Arg::I(nb.clone())], &vars);
            let st = mpolys(&vars, &n);
            let o = ours!(st, v, |s, i| s[i][0].mul_mod(&s[i][1], &s[i][2]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "mlimb",
                "ModPolynomial::mul_mod (multi-limb n)",
                format!("n={ns} len(f)={len}"),
                &buf,
                &mrefs("m_mulmod", "fmpz_mod_poly_mulmod", Some("MulMod"), Some("FpXQ_mul")),
                o,
                Cmp::Exact,
                "no precomputed inverse on any side",
            );
            let st = mpolys(&vars, &n);
            let o = ours!(st, v, |s, i| s[i][0].compose_mod(&s[i][1], &s[i][2]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "mlimb",
                "ModPolynomial::compose_mod (multi-limb n)",
                format!("n={ns} len(f)={len}"),
                &buf,
                &mrefs(
                    "m_compose_mod",
                    "fmpz_mod_poly_compose_mod_brent_kung_preinv",
                    Some("CompMod"),
                    Some("FpX_FpXQ_eval"),
                ),
                o,
                Cmp::Exact,
                "refs: FLINT's inverse of f and NTL's Modulus built before the clock",
            );
        }
    }
}

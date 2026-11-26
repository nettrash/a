//! Polynomials modulo a word: products, division, powers, compositions (groups wmul, wdiv).

use crate::core::*;
use a::math::numbers::Number;
use a::math::poly::ModPolynomial;
use std::hint::black_box;

pub const P65521: u64 = 65521;
pub const P50: u64 = (1 << 50) - 27;
pub const P63: u64 = (1 << 63) - 25;
pub const P64: u64 = u64::MAX - 58;
pub const WMODS: [(u64, &str); 4] =
    [(P65521, "65521"), (P50, "2^50-27"), (P63, "2^63-25"), (P64, "2^64-59")];

pub fn rand_w(r: &mut Rng, len: usize, n: u64) -> Vec<u64> {
    (0..len).map(|_| r.below(n)).collect()
}
/// Random residues, the top one not zero.
pub fn rand_w_lc(r: &mut Rng, len: usize, n: u64) -> Vec<u64> {
    let mut v = rand_w(r, len, n);
    if let Some(t) = v.last_mut() {
        *t = 1 + r.below(n - 1);
    }
    v
}
pub fn rand_monic(r: &mut Rng, len: usize, n: u64) -> Vec<u64> {
    let mut v = rand_w(r, len, n);
    if let Some(t) = v.last_mut() {
        *t = 1;
    }
    v
}

pub type Ours = Box<dyn Contender>;

/// ModPolynomials of each variant from word arguments.
pub fn wpolys(vars: &[Vec<Arg>], n: u64) -> Vec<Vec<ModPolynomial>> {
    vars.iter().map(|v| v.iter().map(|a| ModPolynomial::from_u64(a.w(), n)).collect()).collect()
}

macro_rules! ours {
    ($st:expr, $v:expr, |$s:ident, $i:ident| $call:expr, |$r:ident| $out:expr) => {
        native(
            $st,
            $v,
            move |$s, $i| {
                black_box($call);
            },
            move |$s, $i| {
                let $r = $call;
                $out
            },
        )
    };
}
pub(crate) use ours;

pub fn refs_w(op: &str, flint: &str, ntl: Option<&str>, pari: Option<&str>) -> Vec<(Lib, String, String)> {
    let mut v = vec![(Lib::Flint, op.to_string(), flint.to_string())];
    if let Some(f) = ntl {
        v.push((Lib::Ntl, op.to_string(), f.to_string()));
    }
    if let Some(f) = pari {
        v.push((Lib::Pari, op.to_string(), f.to_string()));
    }
    v
}

pub fn run_case(
    out: &mut Out,
    group: &'static str,
    op: &str,
    params: String,
    buf: &[u64],
    refs: &[(Lib, String, String)],
    ours: Ours,
    cmp: Cmp,
    notes: &str,
) {
    let r: Vec<(Lib, &str, &str)> = refs.iter().map(|(l, o, f)| (*l, o.as_str(), f.as_str())).collect();
    let refs = shims(&r, buf);
    out.case(Case {
        group,
        op: op.to_string(),
        params,
        ours,
        refs,
        cmp,
        notes: notes.to_string(),
        check_note: None,
    });
}

pub fn wmul(out: &mut Out) {
    if !out.wants("wmul") {
        return;
    }
    let lens = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 30000, 100000, 300000, 1000000];
    let mut r = Rng(0x5EED_0001);
    let mut mods: Vec<(u64, &str)> = WMODS.to_vec();
    mods.push((2, "2"));
    for &(n, ns) in &mods {
        for &len in &lens {
            if out.skip(len) {
                continue;
            }
            if n == 2 && ![64usize, 1000, 10000, 100000, 1000000].contains(&len) {
                continue;
            }
            let v = variants(len);
            // products
            let vars: Vec<Vec<Arg>> =
                (0..v).map(|_| vec![Arg::W(rand_w_lc(&mut r, len, n)), Arg::W(rand_w_lc(&mut r, len, n))]).collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| &s[i][0] * &s[i][1], |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wmul",
                "ModPolynomial * (n < 2^64)",
                format!("n={ns} len={len}x{len}"),
                &buf,
                &refs_w("w_mul", "nmod_poly_mul", Some("mul"), Some("Flx_mul")),
                o,
                Cmp::Exact,
                "",
            );
            // squares
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(rand_w_lc(&mut r, len, n))]).collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| &s[i][0] * &s[i][0], |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wmul",
                "ModPolynomial &a * &a (n < 2^64)",
                format!("n={ns} len={len}"),
                &buf,
                &refs_w("w_sqr", "nmod_poly_mul(a,a)", Some("sqr"), Some("Flx_sqr")),
                o,
                Cmp::Exact,
                "",
            );
        }
    }
}

pub fn wdiv(out: &mut Out) {
    if !out.wants("wdiv") {
        return;
    }
    let mut r = Rng(0x5EED_0002);
    let e63 = Big::from_u64(P63);
    for &(n, ns) in &WMODS {
        // div_rem, 2n by n coefficients
        for &len in &[2usize, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 30000, 100000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::W(rand_w_lc(&mut r, 2 * len, n)), Arg::W(rand_w_lc(&mut r, len, n))])
                .collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].div_rem(&s[i][1]), |p| {
                let mut w = out_mod(&p.0).to_vec();
                w.extend(out_mod(&p.1));
                w
            });
            run_case(
                out,
                "wdiv",
                "ModPolynomial::div_rem (n < 2^64)",
                format!("n={ns} len={}/{len}", 2 * len),
                &buf,
                &refs_w("w_divrem", "nmod_poly_divrem", Some("DivRem"), Some("Flx_divrem")),
                o,
                Cmp::Exact,
                "",
            );
        }
        // inv_series(len) of a polynomial of len coefficients
        for &len in &[2usize, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 30000, 100000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    let mut a = rand_w(&mut r, len, n);
                    a[0] = 1 + r.below(n - 1);
                    vec![Arg::W(a)]
                })
                .collect();
            let buf = encode(&[Arg::U(n), Arg::U(len as u64)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].inv_series(len), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wdiv",
                "ModPolynomial::inv_series (n < 2^64)",
                format!("n={ns} len={len} terms={len}"),
                &buf,
                &refs_w("w_inv_series", "nmod_poly_inv_series", Some("InvTrunc"), Some("Flxn_inv")),
                o,
                Cmp::Exact,
                "",
            );
        }
        // mul_mod(&b, &f), f of len coefficients, a and b of len - 1
        for &len in &[4usize, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 30000, 100000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    vec![
                        Arg::W(rand_w(&mut r, len - 1, n)),
                        Arg::W(rand_w(&mut r, len - 1, n)),
                        Arg::W(rand_w_lc(&mut r, len, n)),
                    ]
                })
                .collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].mul_mod(&s[i][1], &s[i][2]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wdiv",
                "ModPolynomial::mul_mod (n < 2^64)",
                format!("n={ns} len(f)={len}"),
                &buf,
                &refs_w("w_mulmod", "nmod_poly_mulmod", Some("MulMod"), Some("Flxq_mul")),
                o,
                Cmp::Exact,
                "no precomputed inverse on any side",
            );
        }
        // x.pow_mod(&e, &f), e = 2^63 - 25
        for &len in &[4usize, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 30000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(rand_w_lc(&mut r, len, n))]).collect();
            let buf = encode(&[Arg::U(n), Arg::I(e63.clone())], &vars);
            let st = (wpolys(&vars, n), Number::from(P63), ModPolynomial::x(&Number::from(n)));
            let o = ours!(st, v, |s, i| s.2.pow_mod(&s.1, &s.0[i][0]), |p| out_mod(&p).to_vec());
            let mut rf = refs_w("w_powmod_x", "nmod_poly_powmod_x_fmpz_preinv", Some("PowerXMod"), Some("Flxq_pow"));
            rf.insert(1, (Lib::Flint, "w_powmod_x_binexp".into(), "nmod_poly_powmod_fmpz_binexp_preinv(x)".into()));
            run_case(
                out,
                "wdiv",
                "ModPolynomial x.pow_mod(e, f) (n < 2^64)",
                format!("n={ns} len(f)={len} e=2^63-25"),
                &buf,
                &rf,
                o,
                Cmp::Exact,
                "refs: FLINT's inverse of f and NTL's Modulus built before the clock",
            );
        }
        // a.pow_mod(&e, &f) for a random a and an e of 1024 bits
        for &len in &[4usize, 8, 16, 32, 64, 128, 256, 512, 1000, 2000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let e = r.uint(1024);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::W(rand_w(&mut r, len - 1, n)), Arg::W(rand_w_lc(&mut r, len, n))])
                .collect();
            let buf = encode(&[Arg::U(n), Arg::I(e.clone())], &vars);
            let st = (wpolys(&vars, n), e.to_number());
            let o = ours!(st, v, |s, i| s.0[i][0].pow_mod(&s.1, &s.0[i][1]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wdiv",
                "ModPolynomial::pow_mod (n < 2^64)",
                format!("n={ns} len(f)={len} e=1024 bits"),
                &buf,
                &refs_w("w_powmod", "nmod_poly_powmod_fmpz_binexp_preinv", Some("PowerMod"), Some("Flxq_pow")),
                o,
                Cmp::Exact,
                "refs: FLINT's inverse of f and NTL's Modulus built before the clock",
            );
        }
        // compose(&g), both of len coefficients
        for &len in &[2usize, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::W(rand_w_lc(&mut r, len, n)), Arg::W(rand_w_lc(&mut r, len, n))])
                .collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].compose(&s[i][1]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wdiv",
                "ModPolynomial::compose (n < 2^64)",
                format!("n={ns} len={len}"),
                &buf,
                &refs_w("w_compose", "nmod_poly_compose", None, None),
                o,
                Cmp::Exact,
                "",
            );
        }
        // compose_mod(&g, &f), a and g of len - 1, f of len
        for &len in &[4usize, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    vec![
                        Arg::W(rand_w(&mut r, len - 1, n)),
                        Arg::W(rand_w(&mut r, len - 1, n)),
                        Arg::W(rand_w_lc(&mut r, len, n)),
                    ]
                })
                .collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].compose_mod(&s[i][1], &s[i][2]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wdiv",
                "ModPolynomial::compose_mod (n < 2^64)",
                format!("n={ns} len(f)={len}"),
                &buf,
                &refs_w(
                    "w_compose_mod",
                    "nmod_poly_compose_mod_brent_kung_preinv",
                    Some("CompMod"),
                    Some("Flx_Flxq_eval"),
                ),
                o,
                Cmp::Exact,
                "refs: FLINT's inverse of f and NTL's Modulus built before the clock",
            );
        }
        // pow(7)
        for &len in &[2usize, 16, 128, 1000, 10000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(rand_w_lc(&mut r, len, n))]).collect();
            let buf = encode(&[Arg::U(n), Arg::U(7)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].pow(7), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wdiv",
                "ModPolynomial::pow (n < 2^64)",
                format!("n={ns} len={len} e=7"),
                &buf,
                &refs_w("w_pow", "nmod_poly_pow", Some("power"), Some("Flx_powu")),
                o,
                Cmp::Exact,
                "",
            );
        }
        // eval at one point, derivative, add, scale
        for &len in &[16usize, 1000, 100000] {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> =
                (0..v).map(|_| vec![Arg::W(rand_w_lc(&mut r, len, n)), Arg::U(r.below(n))]).collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st: (Vec<ModPolynomial>, Vec<Number>) = (
                vars.iter().map(|v| ModPolynomial::from_u64(v[0].w(), n)).collect(),
                vars.iter().map(|v| Number::from(v[1].u())).collect(),
            );
            let o = ours!(st, v, |s, i| s.0[i].eval(&s.1[i]), |x| vec![fp_number(&x)]);
            run_case(
                out,
                "wdiv",
                "ModPolynomial::eval (n < 2^64)",
                format!("n={ns} len={len}"),
                &buf,
                &refs_w("w_eval", "nmod_poly_evaluate_nmod", Some("eval"), Some("Flx_eval")),
                o,
                Cmp::Exact,
                "ours takes and returns a Number",
            );
            let st: (Vec<ModPolynomial>, Vec<Number>) = (
                vars.iter().map(|v| ModPolynomial::from_u64(v[0].w(), n)).collect(),
                vars.iter().map(|v| Number::from(v[1].u())).collect(),
            );
            let o = ours!(st, v, |s, i| s.0[i].scale(&s.1[i]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wdiv",
                "ModPolynomial::scale (n < 2^64)",
                format!("n={ns} len={len}"),
                &buf,
                &refs_w("w_scale", "nmod_poly_scalar_mul_nmod", Some("mul(c)"), Some("Flx_Fl_mul")),
                o,
                Cmp::Exact,
                "",
            );
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::W(rand_w_lc(&mut r, len, n)), Arg::W(rand_w_lc(&mut r, len, n))])
                .collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].derivative(), |p| out_mod(&p).to_vec());
            let vars1: Vec<Vec<Arg>> = vars.iter().map(|x| vec![x[0].clone()]).collect();
            let buf1 = encode(&[Arg::U(n)], &vars1);
            run_case(
                out,
                "wdiv",
                "ModPolynomial::derivative (n < 2^64)",
                format!("n={ns} len={len}"),
                &buf1,
                &refs_w("w_derivative", "nmod_poly_derivative", Some("diff"), Some("Flx_deriv")),
                o,
                Cmp::Exact,
                "",
            );
            let st = wpolys(&vars, n);
            let o = ours!(st, v, |s, i| &s[i][0] + &s[i][1], |p| out_mod(&p).to_vec());
            run_case(
                out,
                "wdiv",
                "ModPolynomial + (n < 2^64)",
                format!("n={ns} len={len}"),
                &buf,
                &refs_w("w_add", "nmod_poly_add", Some("add"), Some("Flx_add")),
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    // modulo 2: NTL's GF2X and PARI's F2x pack bits
    let n = 2u64;
    for &len in &[64usize, 1000, 10000, 100000] {
        if out.skip(len) {
            continue;
        }
        let v = variants(len);
        let vars: Vec<Vec<Arg>> = (0..v)
            .map(|_| vec![Arg::W(rand_w_lc(&mut r, 2 * len, n)), Arg::W(rand_w_lc(&mut r, len, n))])
            .collect();
        let buf = encode(&[Arg::U(n)], &vars);
        let st = wpolys(&vars, n);
        let o = ours!(st, v, |s, i| s[i][0].div_rem(&s[i][1]), |p| {
            let mut w = out_mod(&p.0).to_vec();
            w.extend(out_mod(&p.1));
            w
        });
        run_case(
            out,
            "wdiv",
            "ModPolynomial::div_rem (n < 2^64)",
            format!("n=2 len={}/{len}", 2 * len),
            &buf,
            &refs_w("w_divrem", "nmod_poly_divrem", Some("DivRem"), Some("F2x_divrem")),
            o,
            Cmp::Exact,
            "",
        );
    }
    for &len in &[64usize, 1000, 10000] {
        if out.skip(len) {
            continue;
        }
        let v = variants(len);
        let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::W(rand_w_lc(&mut r, len, n))]).collect();
        let buf = encode(&[Arg::U(n), Arg::I(e63.clone())], &vars);
        let st = (wpolys(&vars, n), Number::from(P63), ModPolynomial::x(&Number::from(n)));
        let o = ours!(st, v, |s, i| s.2.pow_mod(&s.1, &s.0[i][0]), |p| out_mod(&p).to_vec());
        run_case(
            out,
            "wdiv",
            "ModPolynomial x.pow_mod(e, f) (n < 2^64)",
            format!("n=2 len(f)={len} e=2^63-25"),
            &buf,
            &refs_w("w_powmod_x", "nmod_poly_powmod_x_fmpz_preinv", Some("PowerXMod"), Some("F2xq_pow")),
            o,
            Cmp::Exact,
            "refs: FLINT's inverse of f and NTL's Modulus built before the clock",
        );
    }
}

//! Greatest common divisors, resultants and discriminants (group gcd).

use crate::core::*;
use crate::m::rand_m_lc;
use crate::w::{P50, P63, P64, ours, rand_w_lc, run_case};
use crate::z::{rand_z, zrefs};
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;
use std::hint::black_box;

fn wprod(a: &[u64], b: &[u64], n: u64) -> Vec<u64> {
    let p = &ModPolynomial::from_u64(a, n) * &ModPolynomial::from_u64(b, n);
    p.word_coeffs().unwrap().to_vec()
}
fn mprod(a: &[Big], b: &[Big], n: &Number) -> Vec<Big> {
    let p = &ModPolynomial::new(&numbers(a), n) * &ModPolynomial::new(&numbers(b), n);
    p.coeffs().iter().map(from_number).collect()
}
pub(crate) fn zprod(a: &[Big], b: &[Big]) -> Vec<Big> {
    let p = &zpoly(a) * &zpoly(b);
    p.coeffs().iter().map(from_number).collect()
}

fn refs3(op: &str, f: &str, n: Option<&str>, p: Option<&str>) -> Vec<(Lib, String, String)> {
    zrefs(op, Some(f), n, p)
}

pub fn prime_limbs(k: usize) -> Big {
    let top = Number::from(1u64) << (64 * k);
    from_number(&top.prev_prime().unwrap())
}

pub fn gcd(out: &mut Out) {
    if !out.wants("gcd") {
        return;
    }
    let mut r = Rng(0x5EED_0007);
    // modulo word primes
    let wp: [(u64, &str, &[usize]); 4] = [
        (131071, "2^17-1", &[10, 30, 100, 300, 1000, 3000, 10000]),
        (P50, "2^50-27", &[10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]),
        (P63, "2^63-25", &[10, 30, 100, 300, 1000, 3000, 10000]),
        (P64, "2^64-59", &[10, 30, 100, 300, 1000, 3000, 10000]),
    ];
    for (n, ns, degs) in wp {
        for &d in degs {
            if out.skip(d) {
                continue;
            }
            let v = variants(d);
            let h = d / 2;
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    let f = rand_w_lc(&mut r, h + 1, n);
                    let g1 = rand_w_lc(&mut r, d - h + 1, n);
                    let g2 = rand_w_lc(&mut r, d - h, n);
                    vec![Arg::W(wprod(&f, &g1, n)), Arg::W(wprod(&f, &g2, n))]
                })
                .collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st: Vec<Vec<ModPolynomial>> = crate::w::wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].gcd(&s[i][1]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "gcd",
                "ModPolynomial::gcd (n < 2^64)",
                format!("p={ns} deg={d}/{} common deg {h}", d - 1),
                &buf,
                &refs3("w_gcd", "nmod_poly_gcd", Some("GCD"), Some("Flx_gcd")),
                o,
                Cmp::Exact,
                "",
            );
            if d <= 30000 {
                let st: Vec<Vec<ModPolynomial>> = crate::w::wpolys(&vars, n);
                let o = ours!(st, v, |s, i| s[i][0].xgcd(&s[i][1]), |p| {
                    let mut w = out_mod(&p.0).to_vec();
                    w.extend(out_mod(&p.1));
                    w.extend(out_mod(&p.2));
                    w
                });
                run_case(
                    out,
                    "gcd",
                    "ModPolynomial::xgcd (n < 2^64)",
                    format!("p={ns} deg={d}/{} common deg {h}", d - 1),
                    &buf,
                    &refs3("w_xgcd", "nmod_poly_xgcd", Some("XGCD"), Some("Flx_extgcd")),
                    o,
                    Cmp::Exact,
                    "",
                );
            }
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::W(rand_w_lc(&mut r, d + 1, n)), Arg::W(rand_w_lc(&mut r, d, n))])
                .collect();
            let buf = encode(&[Arg::U(n)], &vars);
            let st: Vec<Vec<ModPolynomial>> = crate::w::wpolys(&vars, n);
            let o = ours!(st, v, |s, i| s[i][0].resultant(&s[i][1]), |x| vec![fp_number(&x)]);
            run_case(
                out,
                "gcd",
                "ModPolynomial::resultant (n < 2^64)",
                format!("p={ns} deg={d}/{}", d - 1),
                &buf,
                &refs3("w_resultant", "nmod_poly_resultant", Some("resultant"), Some("Flx_resultant")),
                o,
                Cmp::Exact,
                "",
            );
            if [10, 100, 1000].contains(&d) {
                let vars1: Vec<Vec<Arg>> = vars.iter().map(|x| vec![x[0].clone()]).collect();
                let buf1 = encode(&[Arg::U(n)], &vars1);
                let st: Vec<Vec<ModPolynomial>> = crate::w::wpolys(&vars1, n);
                let o = ours!(st, v, |s, i| s[i][0].discriminant(), |x| vec![fp_number(&x)]);
                run_case(
                    out,
                    "gcd",
                    "ModPolynomial::discriminant (n < 2^64)",
                    format!("p={ns} deg={d}"),
                    &buf1,
                    &refs3("w_discriminant", "nmod_poly_discriminant", None, None),
                    o,
                    Cmp::Exact,
                    "",
                );
            }
        }
    }
    // modulo primes of 2, 3, 8, 16 limbs
    for (k, degs) in [(2usize, &[10usize, 30, 100, 300, 1000, 3000, 10000][..]), (3, &[10, 30, 100, 300, 1000]), (8, &[10, 30, 100, 300, 1000]), (16, &[10, 30, 100, 300, 1000])] {
        let nb = prime_limbs(k);
        let n = nb.to_number();
        let ns = format!("prev_prime(2^{})", 64 * k);
        for &d in degs {
            if out.skip(d) {
                continue;
            }
            let v = variants(d);
            let h = d / 2;
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    let f = rand_m_lc(&mut r, h + 1, &nb);
                    let g1 = rand_m_lc(&mut r, d - h + 1, &nb);
                    let g2 = rand_m_lc(&mut r, d - h, &nb);
                    vec![Arg::IV(mprod(&f, &g1, &n)), Arg::IV(mprod(&f, &g2, &n))]
                })
                .collect();
            let buf = encode(&[Arg::I(nb.clone())], &vars);
            let st = crate::m::mpolys(&vars, &n);
            let o = ours!(st, v, |s, i| s[i][0].gcd(&s[i][1]), |p| out_mod(&p).to_vec());
            run_case(
                out,
                "gcd",
                "ModPolynomial::gcd (multi-limb n)",
                format!("p={ns} deg={d}/{} common deg {h}", d - 1),
                &buf,
                &refs3("m_gcd", "fmpz_mod_poly_gcd", Some("GCD"), Some("FpX_gcd")),
                o,
                Cmp::Exact,
                "",
            );
            if d <= 1000 {
                let st = crate::m::mpolys(&vars, &n);
                let o = ours!(st, v, |s, i| s[i][0].xgcd(&s[i][1]), |p| {
                    let mut w = out_mod(&p.0).to_vec();
                    w.extend(out_mod(&p.1));
                    w.extend(out_mod(&p.2));
                    w
                });
                run_case(
                    out,
                    "gcd",
                    "ModPolynomial::xgcd (multi-limb n)",
                    format!("p={ns} deg={d}/{} common deg {h}", d - 1),
                    &buf,
                    &refs3("m_xgcd", "fmpz_mod_poly_xgcd", Some("XGCD"), Some("FpX_extgcd")),
                    o,
                    Cmp::Exact,
                    "",
                );
                let vars: Vec<Vec<Arg>> = (0..v)
                    .map(|_| vec![Arg::IV(rand_m_lc(&mut r, d + 1, &nb)), Arg::IV(rand_m_lc(&mut r, d, &nb))])
                    .collect();
                let buf = encode(&[Arg::I(nb.clone())], &vars);
                let st = crate::m::mpolys(&vars, &n);
                let o = ours!(st, v, |s, i| s[i][0].resultant(&s[i][1]), |x| vec![fp_number(&x)]);
                run_case(
                    out,
                    "gcd",
                    "ModPolynomial::resultant (multi-limb n)",
                    format!("p={ns} deg={d}/{}", d - 1),
                    &buf,
                    &refs3("m_resultant", "fmpz_mod_poly_resultant", Some("resultant"), Some("FpX_resultant")),
                    o,
                    Cmp::Exact,
                    "",
                );
                if k == 2 && [10, 100].contains(&d) {
                    let vars1: Vec<Vec<Arg>> = vars.iter().map(|x| vec![x[0].clone()]).collect();
                    let buf1 = encode(&[Arg::I(nb.clone())], &vars1);
                    let st = crate::m::mpolys(&vars1, &n);
                    let o = ours!(st, v, |s, i| s[i][0].discriminant(), |x| vec![fp_number(&x)]);
                    run_case(
                        out,
                        "gcd",
                        "ModPolynomial::discriminant (multi-limb n)",
                        format!("p={ns} deg={d}"),
                        &buf1,
                        &refs3("m_discriminant", "fmpz_mod_poly_discriminant", None, Some("FpX_disc")),
                        o,
                        Cmp::Exact,
                        "",
                    );
                }
            }
        }
    }
    // over the integers
    for (bits, common, coprime) in [
        (64u32, &[5usize, 10, 30, 100, 300, 1000, 3000][..], &[5usize, 10, 30, 100, 300, 1000][..]),
        (1000, &[5, 10, 30, 100, 300, 1000], &[5, 10, 30, 100, 300, 1000]),
        (10000, &[5, 10, 30, 100], &[5, 10, 30, 100]),
    ] {
        for &d in common {
            if out.skip(d) {
                continue;
            }
            let v = variants(d);
            let h = d / 2;
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    let f = rand_z(&mut r, h + 1, bits / 2);
                    let g1 = rand_z(&mut r, d - h + 1, bits / 2);
                    let g2 = rand_z(&mut r, d - h, bits / 2);
                    vec![Arg::IV(zprod(&f, &g1)), Arg::IV(zprod(&f, &g2))]
                })
                .collect();
            let buf = encode(&[], &vars);
            let st = crate::z::zpolys(&vars);
            let o = ours!(st, v, |s, i| s[i][0].gcd(&s[i][1]), |p| out_z(&p).to_vec());
            run_case(
                out,
                "gcd",
                "Polynomial<Number>::gcd",
                format!("bits={bits} deg={d}/{} common deg {h} (factors of {} bits)", d - 1, bits / 2),
                &buf,
                &refs3("z_gcd", "fmpz_poly_gcd", Some("GCD"), Some("ZX_gcd")),
                o,
                Cmp::Exact,
                "",
            );
            if d == 10 || d == 100 {
                let st = crate::z::zpolys(&vars);
                let o = ours!(st, v, |s, i| s[i][0].lcm(&s[i][1]), |p| out_z(&p).to_vec());
                run_case(
                    out,
                    "gcd",
                    "Polynomial<Number>::lcm",
                    format!("bits={bits} deg={d}/{} common deg {h}", d - 1),
                    &buf,
                    &refs3("z_lcm", "fmpz_poly_lcm", None, None),
                    o,
                    Cmp::Exact,
                    "",
                );
            }
        }
        for &d in coprime {
            if out.skip(d) {
                continue;
            }
            let v = variants(d);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::IV(rand_z(&mut r, d + 1, bits)), Arg::IV(rand_z(&mut r, d, bits))])
                .collect();
            let buf = encode(&[], &vars);
            let st = crate::z::zpolys(&vars);
            let o = ours!(st, v, |s, i| s[i][0].gcd(&s[i][1]), |p| out_z(&p).to_vec());
            run_case(
                out,
                "gcd",
                "Polynomial<Number>::gcd",
                format!("bits={bits} deg={d}/{} coprime", d - 1),
                &buf,
                &refs3("z_gcd", "fmpz_poly_gcd", Some("GCD"), Some("ZX_gcd")),
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    for (bits, degs) in [(64u32, &[5usize, 10, 30, 100, 300][..]), (1000, &[5, 10, 30, 100])] {
        for &d in degs {
            let v = variants(d);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::IV(rand_z(&mut r, d + 1, bits)), Arg::IV(rand_z(&mut r, d, bits))])
                .collect();
            let buf = encode(&[], &vars);
            let st = crate::z::zpolys(&vars);
            let o = ours!(st, v, |s, i| s[i][0].resultant(&s[i][1]), |x| vec![fp_number(&x)]);
            run_case(
                out,
                "gcd",
                "Polynomial<Number>::resultant",
                format!("bits={bits} deg={d}/{}", d - 1),
                &buf,
                &refs3("z_resultant", "fmpz_poly_resultant", Some("resultant"), Some("ZX_resultant")),
                o,
                Cmp::Exact,
                "NTL's default resultant is randomised (deterministic=0)",
            );
            let vars1: Vec<Vec<Arg>> = vars.iter().map(|x| vec![x[0].clone()]).collect();
            let buf1 = encode(&[], &vars1);
            let st = crate::z::zpolys(&vars1);
            let o = ours!(st, v, |s, i| s[i][0].discriminant(), |x| vec![fp_number(&x)]);
            run_case(
                out,
                "gcd",
                "Polynomial<Number>::discriminant",
                format!("bits={bits} deg={d}"),
                &buf1,
                &refs3("z_discriminant", "fmpz_poly_discriminant", Some("discriminant"), Some("ZX_disc")),
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    // over the rationals: a = f*g1/c1, b = f*g2/c2 with f, g, c integers of bits/2 bits
    for (bits, degs) in [(64u32, &[5usize, 10, 30, 100, 300][..]), (1000, &[5, 10, 30, 100])] {
        let qdiv = |p: Vec<Big>, c: &Big| -> Vec<(Big, Big)> { p.into_iter().map(|x| (x, c.clone())).collect() };
        for &d in degs {
            let v = variants(d);
            let h = d / 2;
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    let f = rand_z(&mut r, h + 1, bits / 2);
                    let g1 = rand_z(&mut r, d - h + 1, bits / 2);
                    let g2 = rand_z(&mut r, d - h, bits / 2);
                    let c1 = r.uint(bits / 2);
                    let c2 = r.uint(bits / 2);
                    vec![Arg::Q(qdiv(zprod(&f, &g1), &c1)), Arg::Q(qdiv(zprod(&f, &g2), &c2))]
                })
                .collect();
            let buf = encode(&[], &vars);
            let st = crate::z::qpolys(&vars);
            let o = ours!(st, v, |s, i| s[i][0].gcd(&s[i][1]), |p| out_q(&p).to_vec());
            run_case(
                out,
                "gcd",
                "Polynomial<Rational>::gcd",
                format!("bits={bits} deg={d}/{} common deg {h}", d - 1),
                &buf,
                &refs3("q_gcd", "fmpq_poly_gcd", None, Some("QX_gcd")),
                o,
                Cmp::Exact,
                "a = f*g1/c1, b = f*g2/c2, parts of bits/2",
            );
            if d <= 100 {
                let st = crate::z::qpolys(&vars);
                let o = ours!(st, v, |s, i| s[i][0].xgcd(&s[i][1]), |p| {
                    let mut w = out_q(&p.0).to_vec();
                    w.extend(out_q(&p.1));
                    w.extend(out_q(&p.2));
                    w
                });
                run_case(
                    out,
                    "gcd",
                    "Polynomial<Rational>::xgcd",
                    format!("bits={bits} deg={d}/{} common deg {h}", d - 1),
                    &buf,
                    &refs3("q_xgcd", "fmpq_poly_xgcd", None, None),
                    o,
                    Cmp::Exact,
                    "",
                );
            }
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    let c1 = r.uint(bits / 2);
                    let c2 = r.uint(bits / 2);
                    vec![Arg::Q(qdiv(rand_z(&mut r, d + 1, bits), &c1)), Arg::Q(qdiv(rand_z(&mut r, d, bits), &c2))]
                })
                .collect();
            let buf = encode(&[], &vars);
            let st = crate::z::qpolys(&vars);
            let o = ours!(st, v, |s, i| s[i][0].resultant(&s[i][1]), |x| vec![fp_rational(&x)]);
            run_case(
                out,
                "gcd",
                "Polynomial<Rational>::resultant",
                format!("bits={bits} deg={d}/{}", d - 1),
                &buf,
                &refs3("q_resultant", "fmpq_poly_resultant", None, Some("QX_resultant")),
                o,
                Cmp::Exact,
                "a = A/c1, b = B/c2, A, B of bits, c of bits/2",
            );
        }
    }
    let _ = (black_box(0), Polynomial::<Rational>::zero());
}

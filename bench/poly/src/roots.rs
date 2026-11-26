//! Roots: double precision, isolation, counting, certified real and complex (group roots).

use crate::core::*;
use crate::w::{ours, run_case};
use crate::z::rand_z;
use a::math::complex::Complex;
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use std::hint::black_box;

fn zbig(p: &Polynomial<Number>) -> Vec<Big> {
    p.coeffs().iter().map(from_number).collect()
}

pub fn chebyshev(n: usize) -> Vec<Big> {
    let x = Polynomial::new(vec![Number::from(0), Number::from(2)]);
    let mut t0 = Polynomial::new(vec![Number::from(1)]);
    let mut t1 = Polynomial::new(vec![Number::from(0), Number::from(1)]);
    if n == 0 {
        return zbig(&t0);
    }
    for _ in 1..n {
        let t2 = &(&x * &t1) - &t0;
        t0 = t1;
        t1 = t2;
    }
    zbig(&t1)
}
pub fn wilkinson(n: usize) -> Vec<Big> {
    let r: Vec<Number> = (1..=n as u64).map(Number::from).collect();
    zbig(&Polynomial::from_roots(&r))
}
/// x^n - 2(a x - 1)^2 for a = 2^k + 1.
pub fn mignotte(n: usize, k: u32) -> Vec<Big> {
    let a = &(Number::from(1u64) << k as usize) + &Number::from(1);
    let mut c = vec![Number::from(0); n + 1];
    c[n] = Number::from(1);
    c[2] = -(&(&a * &a) * &Number::from(2));
    c[1] = &a * &Number::from(4);
    c[0] = Number::from(-2);
    zbig(&Polynomial::new(c))
}
fn sparse(r: &mut Rng, n: usize, bits: u32) -> Vec<Big> {
    let mut c = vec![Big::from_u64(0); n + 1];
    c[n] = r.int(bits);
    c[0] = r.int(bits);
    for _ in 0..8 {
        let i = 1 + r.below(n as u64 - 1) as usize;
        c[i] = r.int(bits);
    }
    c
}

/// |p(z)| / sum |a_i| |z|^i, in double-double so that the measure is not the error of f64 Horner.
fn backward_error(p: &[(f64, f64)], z: (f64, f64)) -> f64 {
    #[derive(Clone, Copy)]
    struct Dd(f64, f64);
    fn two_sum(a: f64, b: f64) -> Dd {
        let s = a + b;
        let bb = s - a;
        Dd(s, (a - (s - bb)) + (b - bb))
    }
    fn add(a: Dd, b: Dd) -> Dd {
        let s = two_sum(a.0, b.0);
        let e = s.1 + a.1 + b.1;
        two_sum(s.0, e)
    }
    fn mul_d(a: Dd, b: f64) -> Dd {
        let p = a.0 * b;
        let e = a.0.mul_add(b, -p) + a.1 * b;
        two_sum(p, e)
    }
    let (mut vr, mut vi) = (Dd(0.0, 0.0), Dd(0.0, 0.0));
    let mut s = 0f64;
    let az = (z.0 * z.0 + z.1 * z.1).sqrt();
    for &(ar, ai) in p.iter().rev() {
        let nr = add(mul_d(vr, z.0), mul_d(vi, -z.1));
        let ni = add(mul_d(vr, z.1), mul_d(vi, z.0));
        vr = add(nr, Dd(ar, 0.0));
        vi = add(ni, Dd(ai, 0.0));
        s = s * az + (ar * ar + ai * ai).sqrt();
    }
    let num = ((vr.0 + vr.1).powi(2) + (vi.0 + vi.1).powi(2)).sqrt();
    if s == 0.0 { 0.0 } else { num / s }
}

fn max_backward(polys: &[Vec<(f64, f64)>], roots: &[u64]) -> f64 {
    let mut worst = 0f64;
    let mut i = 0;
    for p in polys {
        if i >= roots.len() {
            break;
        }
        let n = roots[i] as usize;
        for k in 0..n {
            let z = (f64::from_bits(roots[i + 1 + 2 * k]), f64::from_bits(roots[i + 2 + 2 * k]));
            worst = worst.max(backward_error(p, z));
        }
        i += 1 + 2 * n;
    }
    worst
}

fn real_case(
    out: &mut Out,
    op: &str,
    params: String,
    polys: Vec<Vec<f64>>,
    cplx: bool,
    pari_max: usize,
) {
    let v = polys.len();
    let deg = if cplx { polys[0].len() / 2 - 1 } else { polys[0].len() - 1 };
    let vars: Vec<Vec<Arg>> = polys.iter().map(|p| vec![Arg::F(p.clone())]).collect();
    let buf = encode(&[], &vars);
    let pc: Vec<Vec<(f64, f64)>> = polys
        .iter()
        .map(|p| if cplx { p.chunks(2).map(|c| (c[0], c[1])).collect() } else { p.iter().map(|&x| (x, 0.0)).collect() })
        .collect();
    let o: Box<dyn Contender> = if cplx {
        let st: Vec<Polynomial<Complex<f64>>> = polys.iter().map(|p| cpoly(p)).collect();
        ours!(st, v, |s, i| s[i].roots(), |z| out_roots(&z))
    } else {
        let st: Vec<Polynomial<f64>> = polys.iter().map(|p| fpoly(p)).collect();
        ours!(st, v, |s, i| s[i].roots(), |z| out_roots(&z))
    };
    let shim_op = if cplx { "c_roots" } else { "f_roots" };
    let mut libs = vec![
        (Lib::Flint, shim_op, "acb_poly_find_roots (53 bits)"),
        (Lib::Lapack, shim_op, if cplx { "zgeev (companion, as numpy.roots)" } else { "dgeev (companion, as numpy.roots)" }),
    ];
    if deg <= pari_max {
        libs.push((Lib::Pari, shim_op, "polroots (64 bits)"));
    }
    let refs = shims(&libs, &buf);
    let pc2 = pc.clone();
    out.case(Case {
        group: "roots",
        op: op.to_string(),
        params,
        ours: o,
        refs,
        cmp: Cmp::Roots(1e-6),
        notes: String::new(),
        check_note: Some(Box::new(move |ours, refv| {
            format!(
                "max backward error ours {:.1e} ref {:.1e}",
                max_backward(&pc2, ours),
                max_backward(&pc2, refv)
            )
        })),
    });
}

pub fn roots(out: &mut Out) {
    if !out.wants("roots") {
        return;
    }
    let mut r = Rng(0x5EED_0009);
    for &d in &[10usize, 30, 100, 300, 1000] {
        if out.skip(d) {
            continue;
        }
        let v = if d <= 100 { 4 } else { 1 };
        let polys: Vec<Vec<f64>> = (0..v).map(|_| (0..=d).map(|_| r.unit()).collect()).collect();
        real_case(out, "Polynomial<f64>::roots", format!("deg={d} coeffs in [-1,1]"), polys, false, 300);
    }
    for &d in &[10usize, 100, 1000] {
        if out.skip(d) {
            continue;
        }
        let v = if d <= 100 { 4 } else { 1 };
        let polys: Vec<Vec<f64>> = (0..v).map(|_| (0..2 * (d + 1)).map(|_| r.unit()).collect()).collect();
        real_case(out, "Polynomial<Complex<f64>>::roots", format!("deg={d} parts in [-1,1]"), polys, true, 100);
    }
    // the integer inputs
    let mut cases: Vec<(String, Vec<Vec<Big>>)> = Vec::new();
    for (d, b) in [(10usize, 16u32), (20, 16), (50, 16), (100, 16), (200, 16), (500, 16), (1000, 16), (100, 1000), (300, 1000), (100, 10000)] {
        let v = if d <= 50 { 8 } else if d <= 200 { 2 } else { 1 };
        cases.push((format!("random {d}, {b} bits"), (0..v).map(|_| rand_z(&mut r, d + 1, b)).collect()));
    }
    for n in [20usize, 50, 100, 200] {
        cases.push((format!("Chebyshev T {n}"), vec![chebyshev(n)]));
    }
    for n in [20usize, 50, 100] {
        cases.push((format!("Wilkinson {n}"), vec![wilkinson(n)]));
    }
    for (n, k) in [(20usize, 10u32), (40, 30), (100, 64)] {
        cases.push((format!("Mignotte {n}, a = 2^{k}+1"), vec![mignotte(n, k)]));
    }
    cases.push(("sparse 1000, 64 bits".into(), vec![sparse(&mut r, 1000, 64)]));
    for (name, polys) in &cases {
        let deg = polys[0].len() - 1;
        if out.skip(deg) {
            continue;
        }
        let v = polys.len();
        let vars: Vec<Vec<Arg>> = polys.iter().map(|p| vec![Arg::IV(p.clone())]).collect();
        let buf = encode(&[], &vars);
        for (op, which) in [
            ("Polynomial<Number>::real_root_intervals", 0),
            ("Polynomial<Number>::real_root_intervals_descartes", 1),
            ("Polynomial<Number>::real_root_intervals_cf", 2),
        ] {
            let st: Vec<Polynomial<Number>> = polys.iter().map(|p| zpoly(p)).collect();
            let o = match which {
                0 => ours!(st, v, |s, i| s[i].real_root_intervals(), |x| vec![x.len() as u64]),
                1 => ours!(st, v, |s, i| s[i].real_root_intervals_descartes(), |x| vec![x.len() as u64]),
                _ => ours!(st, v, |s, i| s[i].real_root_intervals_cf(), |x| vec![x.len() as u64]),
            };
            run_case(
                out,
                "roots",
                op,
                name.clone(),
                &buf,
                &[(Lib::Flint, "z_isolate".into(), "fmpz_poly_isolate_real_roots".into())],
                o,
                Cmp::Exact,
                "checked: the numbers of isolated roots agree",
            );
        }
    }
    // count_real_roots
    for name in ["random 50, 16 bits", "random 200, 16 bits", "random 500, 16 bits", "Chebyshev T 100", "Wilkinson 50", "Mignotte 40, a = 2^30+1"] {
        let polys = &cases.iter().find(|c| c.0 == name).unwrap().1;
        let v = polys.len();
        if out.skip(polys[0].len()) {
            continue;
        }
        let vars: Vec<Vec<Arg>> = polys.iter().map(|p| vec![Arg::IV(p.clone())]).collect();
        let buf = encode(&[], &vars);
        let st: Vec<Polynomial<Number>> = polys.iter().map(|p| zpoly(p)).collect();
        let o = ours!(st, v, |s, i| s[i].count_real_roots(), |x| vec![x as u64]);
        run_case(
            out,
            "roots",
            "Polynomial<Number>::count_real_roots",
            name.to_string(),
            &buf,
            &[
                (Lib::Flint, "z_count_real".into(), "fmpz_poly_num_real_roots".into()),
                (Lib::Flint, "z_count_real_sturm".into(), "fmpz_poly_num_real_roots_sturm".into()),
                (Lib::Pari, "z_count_real".into(), "ZX_sturm (polsturm)".into()),
            ],
            o,
            Cmp::Exact,
            "",
        );
    }
    // real_roots(prec): extra inputs at the page's degrees
    let mut extra: Vec<(String, Vec<Vec<Big>>)> = Vec::new();
    extra.push(("random 30, 16 bits".into(), (0..8).map(|_| rand_z(&mut r, 31, 16)).collect()));
    extra.push(("random 300, 16 bits".into(), vec![rand_z(&mut r, 301, 16)]));
    extra.push(("Chebyshev T 30".into(), vec![chebyshev(30)]));
    extra.push(("Wilkinson 30".into(), vec![wilkinson(30)]));
    extra.push(("Mignotte 30, a = 2^10+1".into(), vec![mignotte(30, 10)]));
    let find = |name: &str| -> Vec<Vec<Big>> {
        cases.iter().chain(extra.iter()).find(|c| c.0 == name).unwrap().1.clone()
    };
    for (name, prec) in [
        ("random 20, 16 bits", 53u64),
        ("random 100, 16 bits", 53),
        ("random 100, 16 bits", 333),
        ("random 300, 16 bits", 53),
        ("Chebyshev T 30", 53),
        ("Chebyshev T 100", 53),
        ("Chebyshev T 30", 3333),
        ("Wilkinson 30", 333),
    ] {
        let polys = find(name);
        if out.skip(polys[0].len()) {
            continue;
        }
        let v = polys.len();
        let vars: Vec<Vec<Arg>> = polys.iter().map(|p| vec![Arg::IV(p.clone())]).collect();
        let buf = encode(&[Arg::U(prec)], &vars);
        let st: Vec<Polynomial<Number>> = polys.iter().map(|p| zpoly(p)).collect();
        let o = ours!(st, v, |s, i| s[i].real_roots(prec), |x| {
            let mut w = vec![x.len() as u64];
            let mut f: Vec<f64> = x.iter().map(|(b, _)| b.to_f64()).collect();
            f.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w.extend(f.iter().map(|y| y.to_bits()));
            w
        });
        run_case(
            out,
            "roots",
            "Polynomial<Number>::real_roots(prec)",
            format!("{name}, prec {prec}"),
            &buf,
            &[
                (Lib::Flint, "z_real_roots".into(), "arb_fmpz_poly_real_roots".into()),
                (Lib::Pari, "z_real_roots".into(), "realroots (polrootsreal)".into()),
            ],
            o,
            Cmp::Reals(1e-12),
            "ours correctly rounded; Arb balls of the precision; PARI not certified",
        );
    }
    for (name, prec) in [
        ("random 10, 16 bits", 53u64),
        ("random 30, 16 bits", 53),
        ("random 100, 16 bits", 53),
        ("random 300, 16 bits", 53),
        ("random 30, 16 bits", 333),
        ("random 100, 16 bits", 333),
        ("random 30, 16 bits", 3333),
        ("random 100, 1000 bits", 53),
        ("Chebyshev T 30", 53),
        ("Mignotte 30, a = 2^10+1", 53),
        ("Wilkinson 30", 100),
    ] {
        let polys = find(name);
        if out.skip(polys[0].len()) {
            continue;
        }
        let v = polys.len();
        let vars: Vec<Vec<Arg>> = polys.iter().map(|p| vec![Arg::IV(p.clone())]).collect();
        let buf = encode(&[Arg::U(prec)], &vars);
        let st: Vec<Polynomial<Number>> = polys.iter().map(|p| zpoly(p)).collect();
        let o = ours!(st, v, |s, i| s[i].complex_roots(prec), |x| {
            let z: Vec<Complex<f64>> =
                x.iter().map(|c| Complex { re: c.value().re().to_f64(), im: c.value().im().to_f64() }).collect();
            out_roots(&z)
        });
        run_case(
            out,
            "roots",
            "Polynomial<Number>::complex_roots(prec)",
            format!("{name}, prec {prec}"),
            &buf,
            &[
                (Lib::Flint, "z_complex_roots".into(), "arb_fmpz_poly_complex_roots".into()),
                (Lib::Pari, "z_complex_roots".into(), "polroots (roots at prec)".into()),
            ],
            o,
            Cmp::Roots(1e-10),
            "ours and Arb certified; PARI not certified",
        );
    }
    let _ = black_box(0);
}

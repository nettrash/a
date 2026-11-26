//! Polynomials over the integers, the rationals and the floats: products and division (groups
//! zarith, qarith, farith).

use crate::core::*;
use crate::w::{ours, run_case};
use a::math::complex::Complex;
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use a::math::rational::Rational;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use std::hint::black_box;

pub fn rand_z(r: &mut Rng, len: usize, bits: u32) -> Vec<Big> {
    (0..len).map(|_| r.int(bits)).collect()
}

pub fn zpolys(vars: &[Vec<Arg>]) -> Vec<Vec<Polynomial<Number>>> {
    vars.iter().map(|v| v.iter().map(|a| zpoly(a.iv())).collect()).collect()
}

pub fn zrefs(op: &str, flint: Option<&str>, ntl: Option<&str>, pari: Option<&str>) -> Vec<(Lib, String, String)> {
    let mut v = Vec::new();
    if let Some(f) = flint {
        v.push((Lib::Flint, op.to_string(), f.to_string()));
    }
    if let Some(f) = ntl {
        v.push((Lib::Ntl, op.to_string(), f.to_string()));
    }
    if let Some(f) = pari {
        v.push((Lib::Pari, op.to_string(), f.to_string()));
    }
    v
}

/// The product of two polynomials over Z, as limbs (for building exact quotients).
fn zmul_big(a: &[Big], b: &[Big]) -> Vec<Big> {
    let p = &zpoly(a) * &zpoly(b);
    p.coeffs().iter().map(from_number).collect()
}

pub fn zarith(out: &mut Out) {
    if !out.wants("zarith") {
        return;
    }
    let mut r = Rng(0x5EED_0004);
    let all = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 30000, 100000];
    for &bits in &[32u32, 64, 1000, 10000] {
        for &len in &all {
            if out.skip(len) || (bits == 10000 && len > 30000) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> =
                (0..v).map(|_| vec![Arg::IV(rand_z(&mut r, len, bits)), Arg::IV(rand_z(&mut r, len, bits))]).collect();
            let buf = encode(&[], &vars);
            let st = zpolys(&vars);
            let o = ours!(st, v, |s, i| &s[i][0] * &s[i][1], |p| out_z(&p).to_vec());
            run_case(
                out,
                "zarith",
                "Polynomial<Number> *",
                format!("bits={bits} len={len}x{len}"),
                &buf,
                &zrefs("z_mul", Some("fmpz_poly_mul"), Some("mul"), Some("ZX_mul")),
                o,
                Cmp::Exact,
                "",
            );
            let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::IV(rand_z(&mut r, len, bits))]).collect();
            let buf = encode(&[], &vars);
            let st = zpolys(&vars);
            let o = ours!(st, v, |s, i| s[i][0].sqr(), |p| out_z(&p).to_vec());
            run_case(
                out,
                "zarith",
                "Polynomial<Number>::sqr",
                format!("bits={bits} len={len}"),
                &buf,
                &zrefs("z_sqr", Some("fmpz_poly_sqr"), Some("sqr"), Some("ZX_sqr")),
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    // pseudo_div_rem and div_exact, 2n by n coefficients
    for &bits in &[64u32, 1000] {
        let lens: &[usize] = if bits == 64 {
            &[2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000]
        } else {
            &[2, 4, 8, 16, 32, 64, 128, 256, 512]
        };
        for &len in lens {
            if out.skip(len) {
                continue;
            }
            let v = variants(len);
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| vec![Arg::IV(rand_z(&mut r, 2 * len, bits)), Arg::IV(rand_z(&mut r, len, bits))])
                .collect();
            let buf = encode(&[], &vars);
            let st = zpolys(&vars);
            let o = ours!(st, v, |s, i| s[i][0].pseudo_div_rem(&s[i][1]), |p| {
                let mut w = out_z(&p.0).to_vec();
                w.extend(out_z(&p.1));
                w
            });
            run_case(
                out,
                "zarith",
                "Polynomial<Number>::pseudo_div_rem",
                format!("bits={bits} len={}/{len}", 2 * len),
                &buf,
                &zrefs("z_pdivrem", Some("fmpz_poly_pseudo_divrem"), Some("PseudoDivRem"), Some("RgX_pseudodivrem")),
                o,
                Cmp::Exact,
                "FLINT may use a lower power of lc (its results scaled to ours for the check, outside the clock)",
            );
            // div_exact of b*c by b, both of len coefficients
            let vars: Vec<Vec<Arg>> = (0..v)
                .map(|_| {
                    let b = rand_z(&mut r, len, bits);
                    let c = rand_z(&mut r, len, bits);
                    vec![Arg::IV(zmul_big(&b, &c)), Arg::IV(b)]
                })
                .collect();
            let buf = encode(&[], &vars);
            let st = zpolys(&vars);
            let o = ours!(st, v, |s, i| s[i][0].div_exact(&s[i][1]), |p| match p {
                Some(q) => {
                    let mut w = vec![1];
                    w.extend(out_z(&q));
                    w
                }
                None => vec![0],
            });
            run_case(
                out,
                "zarith",
                "Polynomial<Number>::div_exact",
                format!("bits={bits} len={}/{len}", 2 * len - 1),
                &buf,
                &zrefs("z_div_exact", Some("fmpz_poly_divides"), Some("divide"), None),
                o,
                Cmp::Exact,
                "",
            );
        }
    }
    // compose, pow, eval, derivative, add, content (64-bit coefficients)
    for &len in &[4usize, 16, 64] {
        let v = variants(len);
        let vars: Vec<Vec<Arg>> =
            (0..v).map(|_| vec![Arg::IV(rand_z(&mut r, len, 64)), Arg::IV(rand_z(&mut r, len, 64))]).collect();
        let buf = encode(&[], &vars);
        let st = zpolys(&vars);
        let o = ours!(st, v, |s, i| s[i][0].compose(&s[i][1]), |p| out_z(&p).to_vec());
        run_case(
            out,
            "zarith",
            "Polynomial<Number>::compose",
            format!("bits=64 len={len}"),
            &buf,
            &zrefs("z_compose", Some("fmpz_poly_compose"), None, Some("poleval")),
            o,
            Cmp::Exact,
            "",
        );
    }
    for &len in &[4usize, 16, 128] {
        let v = variants(len);
        let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::IV(rand_z(&mut r, len, 64))]).collect();
        let buf = encode(&[Arg::U(7)], &vars);
        let st = zpolys(&vars);
        let o = ours!(st, v, |s, i| s[i][0].pow(7), |p| out_z(&p).to_vec());
        run_case(
            out,
            "zarith",
            "Polynomial<Number>::pow",
            format!("bits=64 len={len} e=7"),
            &buf,
            &zrefs("z_pow", Some("fmpz_poly_pow"), None, Some("gpowgs")),
            o,
            Cmp::Exact,
            "",
        );
    }
    for &len in &[16usize, 256, 4096] {
        let v = variants(len);
        let vars: Vec<Vec<Arg>> =
            (0..v).map(|_| vec![Arg::IV(rand_z(&mut r, len, 64)), Arg::I(r.int(64))]).collect();
        let buf = encode(&[], &vars);
        let st: (Vec<Polynomial<Number>>, Vec<Number>) = (
            vars.iter().map(|v| zpoly(v[0].iv())).collect(),
            vars.iter().map(|v| v[1].big().to_number()).collect(),
        );
        let o = ours!(st, v, |s, i| s.0[i].eval(&s.1[i]), |x| vec![fp_number(&x)]);
        run_case(
            out,
            "zarith",
            "Polynomial<Number>::eval",
            format!("bits=64 len={len} x=64 bits"),
            &buf,
            &zrefs("z_eval", Some("fmpz_poly_evaluate_fmpz"), None, Some("ZX_Z_eval")),
            o,
            Cmp::Exact,
            "",
        );
    }
    for &len in &[16usize, 1000, 100000] {
        if out.skip(len) {
            continue;
        }
        let v = variants(len);
        let vars: Vec<Vec<Arg>> =
            (0..v).map(|_| vec![Arg::IV(rand_z(&mut r, len, 64)), Arg::IV(rand_z(&mut r, len, 64))]).collect();
        let buf = encode(&[], &vars);
        let st = zpolys(&vars);
        let o = ours!(st, v, |s, i| &s[i][0] + &s[i][1], |p| out_z(&p).to_vec());
        run_case(
            out,
            "zarith",
            "Polynomial<Number> +",
            format!("bits=64 len={len}"),
            &buf,
            &zrefs("z_add", Some("fmpz_poly_add"), Some("add"), Some("ZX_add")),
            o,
            Cmp::Exact,
            "",
        );
        let vars1: Vec<Vec<Arg>> = vars.iter().map(|x| vec![x[0].clone()]).collect();
        let buf1 = encode(&[], &vars1);
        let st = zpolys(&vars1);
        let o = ours!(st, v, |s, i| s[i][0].derivative(), |p| out_z(&p).to_vec());
        run_case(
            out,
            "zarith",
            "Polynomial<Number>::derivative",
            format!("bits=64 len={len}"),
            &buf1,
            &zrefs("z_derivative", Some("fmpz_poly_derivative"), Some("diff"), Some("ZX_deriv")),
            o,
            Cmp::Exact,
            "",
        );
        // content of a polynomial with a common factor of 32 bits
        let vars2: Vec<Vec<Arg>> = (0..v)
            .map(|_| {
                let g = r.uint(32).to_number();
                let p = zpoly(&rand_z(&mut r, len, 32));
                let c: Vec<Big> = p.coeffs().iter().map(|x| from_number(&(x * &g))).collect();
                vec![Arg::IV(c)]
            })
            .collect();
        let buf2 = encode(&[], &vars2);
        let st = zpolys(&vars2);
        let o = ours!(st, v, |s, i| s[i][0].content(), |x| vec![fp_number(&x)]);
        run_case(
            out,
            "zarith",
            "Polynomial<Number>::content",
            format!("bits=64 (32-bit common factor) len={len}"),
            &buf2,
            &zrefs("z_content", Some("fmpz_poly_content"), Some("content"), Some("content")),
            o,
            Cmp::Exact,
            "",
        );
    }
}

pub fn rand_q(r: &mut Rng, len: usize, bits: u32) -> Vec<(Big, Big)> {
    (0..len)
        .map(|_| {
            let mut d = r.uint(bits);
            if d.l.is_empty() {
                d = Big::from_u64(1);
            }
            (r.int(bits), d)
        })
        .collect()
}
pub fn qpolys(vars: &[Vec<Arg>]) -> Vec<Vec<Polynomial<Rational>>> {
    vars.iter().map(|v| v.iter().map(|a| qpoly(a.q())).collect()).collect()
}

pub fn qarith(out: &mut Out) {
    if !out.wants("qarith") {
        return;
    }
    let mut r = Rng(0x5EED_0005);
    for &len in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000] {
        if out.skip(len) {
            continue;
        }
        let v = variants(len);
        let vars: Vec<Vec<Arg>> =
            (0..v).map(|_| vec![Arg::Q(rand_q(&mut r, len, 31)), Arg::Q(rand_q(&mut r, len, 31))]).collect();
        let buf = encode(&[], &vars);
        let st = qpolys(&vars);
        let o = ours!(st, v, |s, i| &s[i][0] * &s[i][1], |p| out_q(&p).to_vec());
        run_case(
            out,
            "qarith",
            "Polynomial<Rational> *",
            format!("parts=31 bits len={len}x{len}"),
            &buf,
            &zrefs("q_mul", Some("fmpq_poly_mul"), None, Some("RgX_mul")),
            o,
            Cmp::Exact,
            "FLINT keeps one common denominator; ours each coefficient in lowest terms",
        );
    }
    // div_rem and inv_series: integer polynomials over one common denominator each (as fmpq_poly
    // holds them); with a denominator per coefficient the quotients' sizes explode
    let common = |r: &mut Rng, len: usize| -> Vec<(Big, Big)> {
        let d = r.uint(31);
        (0..len).map(|_| (r.int(31), d.clone())).collect()
    };
    for &len in &[2usize, 8, 32, 128, 512] {
        let v = variants(len);
        let vars: Vec<Vec<Arg>> =
            (0..v).map(|_| vec![Arg::Q(common(&mut r, 2 * len)), Arg::Q(common(&mut r, len))]).collect();
        let buf = encode(&[], &vars);
        let st = qpolys(&vars);
        let o = ours!(st, v, |s, i| s[i][0].div_rem(&s[i][1]), |p| {
            let mut w = out_q(&p.0).to_vec();
            w.extend(out_q(&p.1));
            w
        });
        run_case(
            out,
            "qarith",
            "Polynomial<Rational>::div_rem",
            format!("31-bit integers over a 31-bit common denominator len={}/{len}", 2 * len),
            &buf,
            &zrefs("q_divrem", Some("fmpq_poly_divrem"), None, Some("RgX_divrem")),
            o,
            Cmp::Exact,
            "",
        );
        let vars: Vec<Vec<Arg>> = (0..v).map(|_| vec![Arg::Q(common(&mut r, len))]).collect();
        let buf = encode(&[Arg::U(len as u64)], &vars);
        let st = qpolys(&vars);
        let o = ours!(st, v, |s, i| s[i][0].inv_series(len), |p| out_q(&p).to_vec());
        run_case(
            out,
            "qarith",
            "Polynomial<Rational>::inv_series",
            format!("31-bit integers over a 31-bit common denominator len={len} terms={len}"),
            &buf,
            &zrefs("q_inv_series", Some("fmpq_poly_inv_series"), None, Some("RgXn_inv")),
            o,
            Cmp::Exact,
            "",
        );
    }
    for &len in &[4usize, 16] {
        let v = variants(len);
        let vars: Vec<Vec<Arg>> =
            (0..v).map(|_| vec![Arg::Q(rand_q(&mut r, len, 31)), Arg::Q(rand_q(&mut r, len, 31))]).collect();
        let buf = encode(&[], &vars);
        let st = qpolys(&vars);
        let o = ours!(st, v, |s, i| s[i][0].compose(&s[i][1]), |p| out_q(&p).to_vec());
        run_case(
            out,
            "qarith",
            "Polynomial<Rational>::compose",
            format!("parts=31 bits len={len}"),
            &buf,
            &zrefs("q_compose", Some("fmpq_poly_compose"), None, Some("poleval")),
            o,
            Cmp::Exact,
            "",
        );
    }
}

fn out_floats(c: &[f64]) -> Vec<u64> {
    let mut o = vec![c.len() as u64];
    o.extend(c.iter().map(|x| x.to_bits()));
    o
}
fn out_cfloats(c: &[Complex<f64>]) -> Vec<u64> {
    let mut o = vec![2 * c.len() as u64];
    for z in c {
        o.push(z.re.to_bits());
        o.push(z.im.to_bits());
    }
    o
}

pub fn farith(out: &mut Out) {
    if !out.wants("farith") {
        return;
    }
    let mut r = Rng(0x5EED_0006);
    for &len in &[4usize, 16, 64, 128, 255, 256, 399, 400, 512, 1000, 2000, 5000, 10000, 100000] {
        if out.skip(len) {
            continue;
        }
        let v = variants(len);
        let pairs: Vec<(Vec<f64>, Vec<f64>)> = (0..v)
            .map(|_| ((0..len).map(|_| r.unit()).collect(), (0..len).map(|_| r.unit()).collect()))
            .collect();
        let st: Vec<(Polynomial<f64>, Polynomial<f64>)> =
            pairs.iter().map(|(a, b)| (fpoly(a), fpoly(b))).collect();
        let o = ours!(st, v, |s, i| &s[i].0 * &s[i].1, |p| out_floats(p.coeffs()));
        let mut refs = Vec::new();
        if len <= 10000 {
            let sb = pairs.clone();
            refs.push(Ref {
                label: "schoolbook (Rust loop)".into(),
                c: native(sb, v, |s, i| {
                    black_box(schoolbook(&s[i].0, &s[i].1));
                }, |s, i| out_floats(&schoolbook(&s[i].0, &s[i].1))),
                note: String::new(),
            });
        }
        if len >= 16 {
            let n = (2 * len - 1).next_power_of_two();
            let mut planner = RealFftPlanner::<f64>::new();
            let fwd = planner.plan_fft_forward(n);
            let inv = planner.plan_fft_inverse(n);
            let st = (pairs.clone(), fwd, inv, n);
            refs.push(Ref {
                label: "realfft 3.5 convolution (Rust)".into(),
                c: native(st, v, |s, i| {
                    black_box(fft_mul(&s.0[i].0, &s.0[i].1, &s.1, &s.2, s.3));
                }, |s, i| out_floats(&fft_mul(&s.0[i].0, &s.0[i].1, &s.1, &s.2, s.3))),
                note: "power-of-two FFT, planned before the clock".into(),
            });
        }
        out.case(Case {
            group: "farith",
            op: "Polynomial<f64> *".into(),
            params: format!("len={len}x{len} coeffs in [-1,1]"),
            ours: o,
            refs,
            cmp: Cmp::Floats(1e-12 * (len as f64).sqrt()),
            notes: String::new(),
            check_note: None,
        });
        // complex
        let cpairs: Vec<(Vec<Complex<f64>>, Vec<Complex<f64>>)> = (0..v)
            .map(|_| {
                (
                    (0..len).map(|_| Complex { re: r.unit(), im: r.unit() }).collect(),
                    (0..len).map(|_| Complex { re: r.unit(), im: r.unit() }).collect(),
                )
            })
            .collect();
        let st: Vec<(Polynomial<Complex<f64>>, Polynomial<Complex<f64>>)> = cpairs
            .iter()
            .map(|(a, b)| (Polynomial::new(a.clone()), Polynomial::new(b.clone())))
            .collect();
        let o = ours!(st, v, |s, i| &s[i].0 * &s[i].1, |p| out_cfloats(p.coeffs()));
        let mut refs = Vec::new();
        if len <= 10000 {
            let sb = cpairs.clone();
            refs.push(Ref {
                label: "schoolbook (Rust loop)".into(),
                c: native(sb, v, |s, i| {
                    black_box(cschoolbook(&s[i].0, &s[i].1));
                }, |s, i| out_cfloats(&cschoolbook(&s[i].0, &s[i].1))),
                note: String::new(),
            });
        }
        if len >= 16 {
            let n = (2 * len - 1).next_power_of_two();
            let mut planner = FftPlanner::<f64>::new();
            let fwd = planner.plan_fft_forward(n);
            let inv = planner.plan_fft_inverse(n);
            let st = (cpairs.clone(), fwd, inv, n);
            refs.push(Ref {
                label: "rustfft 6.4 convolution (Rust)".into(),
                c: native(st, v, |s, i| {
                    black_box(cfft_mul(&s.0[i].0, &s.0[i].1, &s.1, &s.2, s.3));
                }, |s, i| out_cfloats(&cfft_mul(&s.0[i].0, &s.0[i].1, &s.1, &s.2, s.3))),
                note: "power-of-two FFT, planned before the clock".into(),
            });
        }
        out.case(Case {
            group: "farith",
            op: "Polynomial<Complex<f64>> *".into(),
            params: format!("len={len}x{len} parts in [-1,1]"),
            ours: o,
            refs,
            cmp: Cmp::Floats(1e-12 * (len as f64).sqrt()),
            notes: String::new(),
            check_note: None,
        });
    }
}

fn schoolbook(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            c[i + j] += x * y;
        }
    }
    c
}
fn cschoolbook(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut c = vec![Complex { re: 0.0, im: 0.0 }; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let t = &mut c[i + j];
            t.re += x.re * y.re - x.im * y.im;
            t.im += x.re * y.im + x.im * y.re;
        }
    }
    c
}
fn fft_mul(
    a: &[f64],
    b: &[f64],
    fwd: &std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    inv: &std::sync::Arc<dyn realfft::ComplexToReal<f64>>,
    n: usize,
) -> Vec<f64> {
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    x[..a.len()].copy_from_slice(a);
    y[..b.len()].copy_from_slice(b);
    let mut xs = fwd.make_output_vec();
    let mut ys = fwd.make_output_vec();
    fwd.process(&mut x, &mut xs).unwrap();
    fwd.process(&mut y, &mut ys).unwrap();
    for (p, q) in xs.iter_mut().zip(&ys) {
        *p *= *q;
    }
    inv.process(&mut xs, &mut x).unwrap();
    let s = 1.0 / n as f64;
    x.truncate(a.len() + b.len() - 1);
    for v in x.iter_mut() {
        *v *= s;
    }
    x
}
fn cfft_mul(
    a: &[Complex<f64>],
    b: &[Complex<f64>],
    fwd: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    inv: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    n: usize,
) -> Vec<Complex<f64>> {
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    let mut y = vec![Complex64::new(0.0, 0.0); n];
    for (d, s) in x.iter_mut().zip(a) {
        *d = Complex64::new(s.re, s.im);
    }
    for (d, s) in y.iter_mut().zip(b) {
        *d = Complex64::new(s.re, s.im);
    }
    fwd.process(&mut x);
    fwd.process(&mut y);
    for (p, q) in x.iter_mut().zip(&y) {
        *p *= *q;
    }
    inv.process(&mut x);
    let s = 1.0 / n as f64;
    x.truncate(a.len() + b.len() - 1);
    x.iter().map(|z| Complex { re: z.re * s, im: z.im * s }).collect()
}

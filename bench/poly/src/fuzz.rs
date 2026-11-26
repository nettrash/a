//! A differential fuzz of the changed paths against FLINT (and NTL where FLINT has nothing):
//! random sizes and moduli, outputs compared as in the scoreboard's checks. No timing.
//! Usage: polybench fuzz SEED COUNT [FILTER]

use crate::core::*;
use crate::w::{rand_monic, rand_w, rand_w_lc};
use crate::eval::out_nums;
use crate::z::rand_z;
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};

fn check(name: &str, op: &str, lib: Lib, buf: &[u64], ours: Vec<u64>, bad: &mut u64, cmp: &Cmp) -> bool {
    let Some(mut s) = Shim::new(lib, op, buf) else {
        return false;
    };
    let r = s.check();
    if let Some(d) = compare(cmp, &ours, &r) {
        *bad += 1;
        eprintln!("FUZZ MISMATCH {name}: {d}");
        if *bad < 4 {
            eprintln!("  buf head {:?}", &buf[..buf.len().min(40)]);
        }
    }
    true
}

fn pick_mod(r: &mut Rng) -> u64 {
    match r.below(8) {
        0 | 1 | 2 => 2,
        3 => 3,
        4 => 65521,
        5 => (1 << 50) - 27,
        6 => (1 << 63) - 25,
        _ => u64::MAX - 58,
    }
}

fn pick_len(r: &mut Rng, max: u64) -> usize {
    match r.below(4) {
        0 => 1 + r.below(8) as usize,
        1 => 1 + r.below(70) as usize,
        2 => 1 + r.below(300) as usize,
        _ => 1 + r.below(max) as usize,
    }
}

pub fn fuzz(seed: u64, count: u64, filter: &str) {
    let mut r = Rng(seed);
    let mut bad = 0u64;
    let mut done = std::collections::BTreeMap::<&str, u64>::new();
    for _ in 0..count {
        let n = pick_mod(&mut r);
        let p2 = n == 2;
        let which = r.below(33);
        let tag = match which {
            0 => "w_mul",
            1 => "w_sqr",
            2 => "w_divrem",
            3 => "w_gcd",
            4 => "w_powmod",
            5 => "w_powmod_x",
            6 => "w_mulmod",
            7 => "w_factor",
            8 => "w_squarefree",
            9 => "w_ddf",
            10 => "w_derivative",
            11 => "w_scale",
            12 => "w_add",
            13 => "w_eval",
            14 => "z_derivative",
            15 => "z_add",
            16 => "z_content",
            17 => "z_mul",
            18 => "q_compose",
            19 => "m_compose_mod",
            20 => "f_mul",
            21 => "w_roots",
            22 => "w_eval_neg",
            23 => "w_compose",
            24 => "w_inv_series",
            25 => "w_scale_big",
            26 => "m_eval_many",
            27 => "m_from_roots",
            28 => "m_interp",
            29 => "z_eval_many",
            30 => "w_interp",
            31 => "z_gcd",
            _ => "z_isolate",
        };
        if !filter.is_empty() && !tag.contains(filter) {
            continue;
        }
        let name = format!("{tag} n={n}");
        let wp = |c: &[u64]| ModPolynomial::from_u64(c, n);
        let ok = match tag {
            "w_mul" | "w_sqr" | "w_add" => {
                let (la, lb) = (pick_len(&mut r, 6000), pick_len(&mut r, 6000));
                let a = rand_w_lc(&mut r, la, n);
                let b = if tag == "w_sqr" { a.clone() } else { rand_w_lc(&mut r, lb, n) };
                let vars = if tag == "w_sqr" { vec![vec![Arg::W(a.clone())]] } else { vec![vec![Arg::W(a.clone()), Arg::W(b.clone())]] };
                let buf = encode(&[Arg::U(n)], &vars);
                let (pa, pb) = (wp(&a), wp(&b));
                let o = match tag {
                    "w_mul" => &pa * &pb,
                    "w_sqr" => &pa * &pa,
                    _ => {
                        // the difference too, by identities
                        let d = &pa - &pb;
                        if &d + &pb != pa || &pb - &pa != -&d {
                            bad += 1;
                            eprintln!("FUZZ MISMATCH {name}: sub identities");
                        }
                        &pa + &pb
                    }
                };
                check(&name, tag, Lib::Flint, &buf, out_mod(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "w_divrem" => {
                let lb = pick_len(&mut r, 3000);
                let la = lb + pick_len(&mut r, 3000) - 1;
                let a = rand_w_lc(&mut r, la, n);
                let b = rand_w_lc(&mut r, lb, n);
                let buf = encode(&[Arg::U(n)], &[vec![Arg::W(a.clone()), Arg::W(b.clone())]]);
                let (q, rr) = wp(&a).div_rem(&wp(&b));
                let rem2 = &wp(&a) % &wp(&b);
                if rem2 != rr {
                    bad += 1;
                    eprintln!("FUZZ MISMATCH {name}: % vs div_rem");
                }
                let mut o = out_mod(&q).to_vec();
                o.extend(out_mod(&rr));
                check(&name, tag, Lib::Flint, &buf, o, &mut bad, &Cmp::Exact)
            }
            "w_gcd" => {
                let h = pick_len(&mut r, 1500);
                let f = rand_w_lc(&mut r, h, n);
                let (l1, l2) = (pick_len(&mut r, 1500), pick_len(&mut r, 1500));
                let g1 = rand_w_lc(&mut r, l1, n);
                let g2 = rand_w_lc(&mut r, l2, n);
                let a = (&wp(&f) * &wp(&g1)).word_coeffs().unwrap().to_vec();
                let b = (&wp(&f) * &wp(&g2)).word_coeffs().unwrap().to_vec();
                let buf = encode(&[Arg::U(n)], &[vec![Arg::W(a.clone()), Arg::W(b.clone())]]);
                let g = wp(&a).gcd(&wp(&b));
                check(&name, tag, Lib::Flint, &buf, out_mod(&g).to_vec(), &mut bad, &Cmp::Exact)
            }
            "w_powmod" | "w_powmod_x" | "w_mulmod" => {
                let lf = 2 + pick_len(&mut r, 1200);
                let f = rand_w_lc(&mut r, lf, n);
                let ebits = 1 + r.below(300) as u32;
                let e = r.uint(ebits);
                let a = rand_w(&mut r, lf - 1, n);
                let b = rand_w(&mut r, lf - 1, n);
                let fp = wp(&f);
                let (buf, o) = match tag {
                    "w_powmod" => (
                        encode(&[Arg::U(n), Arg::I(e.clone())], &[vec![Arg::W(a.clone()), Arg::W(f.clone())]]),
                        wp(&a).pow_mod(&e.to_number(), &fp),
                    ),
                    "w_powmod_x" => (
                        encode(&[Arg::U(n), Arg::I(e.clone())], &[vec![Arg::W(f.clone())]]),
                        ModPolynomial::x(&Number::from(n)).pow_mod(&e.to_number(), &fp),
                    ),
                    _ => (
                        encode(&[Arg::U(n)], &[vec![Arg::W(a.clone()), Arg::W(b.clone()), Arg::W(f.clone())]]),
                        wp(&a).mul_mod(&wp(&b), &fp),
                    ),
                };
                check(&name, tag, Lib::Flint, &buf, out_mod(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "w_factor" | "w_squarefree" | "w_ddf" => {
                if !p2 && r.below(3) != 0 {
                    continue;
                }
                let n = if p2 { 2 } else { 3 };
                let d = 1 + r.below(if p2 { 400 } else { 120 }) as usize;
                let mut a = rand_monic(&mut r, d + 1, n);
                if tag == "w_squarefree" {
                    let lg = 1 + r.below(30) as usize;
                    let g = rand_monic(&mut r, lg, n);
                    let gp = ModPolynomial::from_u64(&g, n);
                    a = (&(&ModPolynomial::from_u64(&a, n) * &gp) * &gp).word_coeffs().unwrap().to_vec();
                }
                let pa = ModPolynomial::from_u64(&a, n);
                if tag == "w_ddf" && !matches!(pa.squarefree().1.as_slice(), [(_, 1)]) {
                    continue;
                }
                let buf = encode(&[Arg::U(n)], &[vec![Arg::W(a.clone())]]);
                let fac = |lc: &Number, fs: &[(ModPolynomial, u32)]| {
                    out_fac(fp_number(lc), fs, |f| {
                        let o = out_mod(f);
                        (o[0], o[1])
                    })
                };
                let o = match tag {
                    "w_factor" => {
                        let (l, f) = pa.factor();
                        let fb = pa.factor_berlekamp();
                        if fb.1 != f {
                            bad += 1;
                            eprintln!("FUZZ MISMATCH {name}: berlekamp vs factor");
                        }
                        let irr = pa.is_irreducible();
                        if irr != (f.len() == 1 && f[0].1 == 1) {
                            bad += 1;
                            eprintln!("FUZZ MISMATCH {name}: is_irreducible");
                        }
                        fac(&l, &f)
                    }
                    "w_squarefree" => {
                        let (l, f) = pa.squarefree();
                        fac(&l, &f)
                    }
                    _ => {
                        let f = pa.distinct_degree_factor();
                        let mut mix = 0;
                        for (g, e) in &f {
                            mix = fp_add(mix, fp_mix(fp_add(out_mod(g)[1], *e as u64)));
                        }
                        vec![f.len() as u64, mix]
                    }
                };
                check(&format!("{tag} n={n} deg={d}"), tag, Lib::Flint, &buf, o, &mut bad, &Cmp::Exact)
            }
            "w_derivative" | "w_scale" | "w_eval" => {
                let n = if r.below(2) == 0 { n } else { 2 + r.below(u64::MAX - 2) };
                let la = pick_len(&mut r, 20000);
                let a = rand_w_lc(&mut r, la, n);
                let x = r.below(n);
                let pa = ModPolynomial::from_u64(&a, n);
                let (buf, o) = match tag {
                    "w_derivative" => (encode(&[Arg::U(n)], &[vec![Arg::W(a.clone())]]), out_mod(&pa.derivative()).to_vec()),
                    "w_scale" => (
                        encode(&[Arg::U(n)], &[vec![Arg::W(a.clone()), Arg::U(x)]]),
                        out_mod(&pa.scale(&Number::from(x))).to_vec(),
                    ),
                    _ => (
                        encode(&[Arg::U(n)], &[vec![Arg::W(a.clone()), Arg::U(x)]]),
                        vec![fp_number(&pa.eval(&Number::from(x)))],
                    ),
                };
                check(&format!("{tag} n={n} len={la}"), tag, Lib::Flint, &buf, o, &mut bad, &Cmp::Exact)
            }
            "z_mul" => {
                let bits = 1 + r.below(200) as u32;
                let (la, lb) = (pick_len(&mut r, 400), pick_len(&mut r, 400));
                let a = rand_z(&mut r, la, bits);
                let b = rand_z(&mut r, lb, bits);
                let buf = encode(&[], &[vec![Arg::IV(a.clone()), Arg::IV(b.clone())]]);
                let o = &zpoly(&a) * &zpoly(&b);
                check(&format!("z_mul bits={bits}"), "z_mul", Lib::Flint, &buf, out_z(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "q_compose" => {
                let (la, lb) = (1 + r.below(12) as usize, 1 + r.below(12) as usize);
                let bits = 1 + r.below(40) as u32;
                let a = crate::z::rand_q(&mut r, la, bits);
                let b = crate::z::rand_q(&mut r, lb, bits);
                let buf = encode(&[], &[vec![Arg::Q(a.clone()), Arg::Q(b.clone())]]);
                let o = qpoly(&a).compose(&qpoly(&b));
                check(&format!("q_compose {la}x{lb}"), "q_compose", Lib::Flint, &buf, out_q(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "m_compose_mod" => {
                let nb = crate::m::mods()[r.below(4) as usize].0.clone();
                let lf = 2 + pick_len(&mut r, 400);
                let a = crate::m::rand_m(&mut r, lf - 1, &nb);
                let g = crate::m::rand_m(&mut r, lf - 1, &nb);
                let f = crate::m::rand_m_lc(&mut r, lf, &nb);
                let buf = encode(&[Arg::I(nb.clone())], &[vec![Arg::IV(a.clone()), Arg::IV(g.clone()), Arg::IV(f.clone())]]);
                let nn = nb.to_number();
                let mp = |c: &[Big]| ModPolynomial::new(&numbers(c), &nn);
                let o = mp(&a).compose_mod(&mp(&g), &mp(&f));
                check(&format!("m_compose_mod lf={lf}"), "m_compose_mod", Lib::Flint, &buf, out_mod(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "f_mul" => {
                let (la, lb) = (pick_len(&mut r, 3000), pick_len(&mut r, 3000));
                let a: Vec<f64> = (0..la).map(|_| r.unit()).collect();
                let b: Vec<f64> = (0..lb).map(|_| r.unit()).collect();
                let o = &fpoly(&a) * &fpoly(&b);
                let mut c = vec![0.0; la + lb - 1];
                for (i, x) in a.iter().enumerate() {
                    for (j, y) in b.iter().enumerate() {
                        c[i + j] += x * y;
                    }
                }
                let tol = 1e-12 * ((la.min(lb)) as f64).sqrt().max(1.0);
                let scale = (a.iter().map(|x| x * x).sum::<f64>() * b.iter().map(|x| x * x).sum::<f64>()).sqrt();
                let worst = o.coeffs().iter().zip(&c).map(|(p, q)| (p - q).abs()).fold(0f64, f64::max) / scale.max(1e-300);
                if o.coeffs().len() != c.len() || worst > tol {
                    bad += 1;
                    eprintln!("FUZZ MISMATCH f_mul {la}x{lb}: {worst:.1e}");
                }
                // complex too
                let ca: Vec<f64> = (0..2 * la).map(|_| r.unit()).collect();
                let cb: Vec<f64> = (0..2 * lb).map(|_| r.unit()).collect();
                let pc = &cpoly(&ca) * &cpoly(&cb);
                let (x, y) = (cpoly(&ca), cpoly(&cb));
                let mut cc = vec![(0.0, 0.0); la + lb - 1];
                for (i, u) in x.coeffs().iter().enumerate() {
                    for (j, v) in y.coeffs().iter().enumerate() {
                        cc[i + j].0 += u.re * v.re - u.im * v.im;
                        cc[i + j].1 += u.re * v.im + u.im * v.re;
                    }
                }
                let worst = pc.coeffs().iter().zip(&cc).map(|(p, q)| ((p.re - q.0).powi(2) + (p.im - q.1).powi(2)).sqrt()).fold(0f64, f64::max);
                if worst > 1e-10 * ((la * lb) as f64).sqrt() {
                    bad += 1;
                    eprintln!("FUZZ MISMATCH c_mul {la}x{lb}: {worst:.1e}");
                }
                true
            }
            "w_roots" => {
                let p = [2u64, 3, 5, 7, 31, 61, 67, 65521, (1 << 50) - 27][r.below(9) as usize];
                let d = 1 + r.below(300) as usize;
                let mut a = rand_monic(&mut r, d + 1, p);
                if r.below(2) == 0 {
                    // repeated roots
                    let x0 = r.below(p);
                    let lin = ModPolynomial::from_u64(&[(p - x0) % p, 1], p);
                    let e = 1 + r.below(4) as u32;
                    a = (&ModPolynomial::from_u64(&a, p) * &lin.pow(u64::from(e))).word_coeffs().unwrap().to_vec();
                }
                let buf = encode(&[Arg::U(p)], &[vec![Arg::W(a.clone())]]);
                let rts = ModPolynomial::from_u64(&a, p).roots();
                let mut o = vec![rts.len() as u64];
                for (z, e) in &rts {
                    o.push(u64::try_from(z).unwrap());
                    o.push(*e as u64);
                }
                check(&format!("w_roots p={p} deg={d}"), "w_roots", Lib::Flint, &buf, o, &mut bad, &Cmp::Exact)
            }
            "w_compose" => {
                let short = r.below(2) == 0;
                let (la, lb) = if short {
                    (1 + r.below(12) as usize, 1 + r.below(12) as usize)
                } else {
                    (pick_len(&mut r, 60), pick_len(&mut r, 60))
                };
                let a = rand_w_lc(&mut r, la, n);
                let b = rand_w_lc(&mut r, lb, n);
                let buf = encode(&[Arg::U(n)], &[vec![Arg::W(a.clone()), Arg::W(b.clone())]]);
                let o = wp(&a).compose(&wp(&b));
                check(&format!("w_compose n={n} {la}x{lb}"), "w_compose", Lib::Flint, &buf, out_mod(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "w_inv_series" => {
                let n = if p2 { 3 } else { n };
                let len = pick_len(&mut r, 400);
                let mut a = rand_w(&mut r, len, n);
                a[0] = 1 + r.below(n - 1);
                let buf = encode(&[Arg::U(n), Arg::U(len as u64)], &[vec![Arg::W(a.clone())]]);
                let o = ModPolynomial::from_u64(&a, n).inv_series(len);
                check(&format!("w_inv_series n={n} len={len}"), "w_inv_series", Lib::Flint, &buf, out_mod(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "w_scale_big" => {
                let n = if r.below(2) == 0 { n } else { 2 + r.below(u64::MAX - 2) };
                let la = pick_len(&mut r, 3000);
                let a = rand_w_lc(&mut r, la, n);
                let xbits = 1 + r.below(130) as u32;
                let xn = r.int(xbits).to_number();
                let xr = u64::try_from(&xn.rem_euclid(&Number::from(n))).unwrap();
                let buf = encode(&[Arg::U(n)], &[vec![Arg::W(a.clone()), Arg::U(xr)]]);
                let o = ModPolynomial::from_u64(&a, n).scale(&xn);
                check(&format!("w_scale_big n={n} len={la}"), "w_scale", Lib::Flint, &buf, out_mod(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "m_eval_many" | "m_from_roots" | "m_interp" => {
                let mods = [
                    Big::pow2_minus(127, 1),
                    Big::pow2_minus(130, 5),
                    Big::pow2_minus(256, 189),
                    Big::pow2_minus(521, 1),
                    Big::pow2_minus(1024, 105),
                ];
                let pb = mods[r.below(5) as usize].clone();
                let pn = pb.to_number();
                let k = if r.below(2) == 0 { 1 + r.below(6) as usize } else { 1 + r.below(300) as usize };
                let nums = |c: &[Big]| -> Vec<Number> { c.iter().map(|x| x.to_number()).collect() };
                match tag {
                    "m_eval_many" => {
                        let f = crate::m::rand_m_lc(&mut r, k, &pb);
                        let np = 1 + r.below(300) as usize;
                        let xs = crate::eval::distinct_m(&mut r, np, &pb);
                        let buf = encode(&[Arg::I(pb.clone())], &[vec![Arg::IV(f.clone()), Arg::IV(xs.clone())]]);
                        let o = ModPolynomial::new(&nums(&f), &pn).eval_many(&nums(&xs));
                        check(&format!("m_eval_many k={k}"), "m_eval_many", Lib::Flint, &buf, out_nums(&o), &mut bad, &Cmp::Exact)
                    }
                    "m_from_roots" => {
                        let rs = crate::m::rand_m(&mut r, k, &pb);
                        let buf = encode(&[Arg::I(pb.clone())], &[vec![Arg::IV(rs.clone())]]);
                        let o = ModPolynomial::from_roots(&nums(&rs), &pn);
                        check(&format!("m_from_roots k={k}"), "m_from_roots", Lib::Flint, &buf, out_mod(&o).to_vec(), &mut bad, &Cmp::Exact)
                    }
                    _ => {
                        // Round trip: the interpolant takes the values at the points.
                        let xs = nums(&crate::eval::distinct_m(&mut r, k, &pb));
                        let ys = nums(&crate::m::rand_m(&mut r, k, &pb));
                        for (how, p) in [
                            ("fast", ModPolynomial::interpolate(&xs, &ys, &pn)),
                            ("newton", ModPolynomial::interpolate_newton(&xs, &ys, &pn)),
                            ("lagrange", ModPolynomial::interpolate_lagrange(&xs, &ys, &pn)),
                        ] {
                            if p.eval_many(&xs) != ys || p.length() > k {
                                bad += 1;
                                eprintln!("FUZZ MISMATCH m_interp {how} k={k}");
                            }
                        }
                        true
                    }
                }
            }
            "z_eval_many" => {
                let k = if r.below(2) == 0 { 1 } else { 1 + r.below(40) as usize };
                let bits = 1 + r.below(300) as u32;
                let c = rand_z(&mut r, k, bits);
                let (np, xb) = (1 + r.below(20) as usize, 1 + r.below(70) as u32);
                let xs = rand_z(&mut r, np, xb);
                let buf = encode(&[], &[vec![Arg::IV(c.clone()), Arg::IV(xs.clone())]]);
                let xn: Vec<Number> = xs.iter().map(|x| x.to_number()).collect();
                let o = zpoly(&c).eval_many(&xn);
                check(&format!("z_eval_many k={k}"), "z_eval_many", Lib::Flint, &buf, out_nums(&o), &mut bad, &Cmp::Exact)
            }
            "w_interp" => {
                let n = if p2 { 65521 } else { n };
                let top = if r.below(2) == 0 { 6 } else { 400 };
                let k = 1 + r.below(top) as usize;
                let xs = crate::eval::distinct_w(&mut r, k.min(n as usize), n);
                let ys = rand_w(&mut r, xs.len(), n);
                let buf = encode(&[Arg::U(n)], &[vec![Arg::W(xs.clone()), Arg::W(ys.clone())]]);
                let o = ModPolynomial::interpolate_u64(&xs, &ys, n);
                let xn: Vec<Number> = xs.iter().map(|&a| Number::from(a)).collect();
                let yn: Vec<Number> = ys.iter().map(|&a| Number::from(a)).collect();
                let nn = Number::from(n);
                for (how, p) in [("newton", ModPolynomial::interpolate_newton(&xn, &yn, &nn)), ("lagrange", ModPolynomial::interpolate_lagrange(&xn, &yn, &nn))] {
                    if p != o {
                        bad += 1;
                        eprintln!("FUZZ MISMATCH w_interp {how} k={k}");
                    }
                }
                check(&format!("w_interp n={n} k={}", xs.len()), "w_interpolate", Lib::Flint, &buf, out_mod(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "z_gcd" => {
                let top = if r.below(2) == 0 { 70 } else { 3000 };
                let bits = 1 + r.below(top) as u32;
                let (da, db) = (1 + r.below(12) as usize, 1 + r.below(12) as usize);
                let (mut a, mut b) = (rand_z(&mut r, da + 1, bits), rand_z(&mut r, db + 1, bits));
                if r.below(2) == 0 {
                    let dc = 1 + r.below(4) as usize;
                    let f = rand_z(&mut r, dc + 1, 1 + bits / 2);
                    a = crate::gcd::zprod(&a, &f);
                    b = crate::gcd::zprod(&b, &f);
                }
                if r.below(3) == 0 {
                    // a common integer factor
                    let kb = 1 + r.below(40) as u32;
                    let k = rand_z(&mut r, 1, kb);
                    a = crate::gcd::zprod(&a, &k);
                    b = crate::gcd::zprod(&b, &k);
                }
                let buf = encode(&[], &[vec![Arg::IV(a.clone()), Arg::IV(b.clone())]]);
                let o = zpoly(&a).gcd(&zpoly(&b));
                check(&format!("z_gcd bits={bits} {da}/{db}"), "z_gcd", Lib::Flint, &buf, out_z(&o).to_vec(), &mut bad, &Cmp::Exact)
            }
            "z_isolate" => {
                let d = 1 + r.below(30) as usize;
                let bits = 1 + r.below(40) as u32;
                let mut a = rand_z(&mut r, d + 1, bits);
                if r.below(3) == 0 {
                    a = crate::gcd::zprod(&a, &a);
                }
                let p = zpoly(&a);
                if p.is_zero() || p.degree() == Some(0) {
                    true
                } else {
                    let o = vec![p.real_root_intervals().len() as u64];
                    for (how, n) in [("descartes", p.real_root_intervals_descartes().len()), ("cf", p.real_root_intervals_cf().len())] {
                        if n as u64 != o[0] {
                            bad += 1;
                            eprintln!("FUZZ MISMATCH z_isolate {how} d={d}");
                        }
                    }
                    // FLINT's isolation needs a square-free input (its bisection does not end on
                    // a multiple root): the others are checked among ours only.
                    if p.squarefree().1.iter().all(|(_, e)| *e == 1) {
                        let buf = encode(&[], &[vec![Arg::IV(a.clone())]]);
                        check(&format!("z_isolate d={d} bits={bits}"), "z_isolate", Lib::Flint, &buf, o, &mut bad, &Cmp::Exact)
                    } else {
                        true
                    }
                }
            }
            "w_eval_neg" => {
                let n = 2 + r.below(u64::MAX - 2);
                let la = pick_len(&mut r, 3000);
                let a = rand_w_lc(&mut r, la, n);
                let xbits = 1 + r.below(100) as u32;
                let xb = r.int(xbits);
                let xn = xb.to_number();
                let xr = u64::try_from(&xn.rem_euclid(&Number::from(n))).unwrap();
                let buf = encode(&[Arg::U(n)], &[vec![Arg::W(a.clone()), Arg::U(xr)]]);
                let o = ModPolynomial::from_u64(&a, n).eval(&xn);
                check(&format!("w_eval_neg n={n}"), "w_eval", Lib::Flint, &buf, vec![fp_number(&o)], &mut bad, &Cmp::Exact)
            }
            _ => {
                let bits = 1 + r.below(300) as u32;
                let (la, lb) = (pick_len(&mut r, 3000), pick_len(&mut r, 3000));
                let mut a = rand_z(&mut r, la, bits);
                let b = rand_z(&mut r, lb, bits);
                if tag == "z_content" && r.below(2) == 0 {
                    let gb = 1 + r.below(100) as u32;
                    let g = r.uint(gb).to_number();
                    a = a.iter().map(|x| from_number(&(&x.to_number() * &g))).collect();
                }
                let (pa, pb) = (zpoly(&a), zpoly(&b));
                let (buf, o) = match tag {
                    "z_derivative" => (encode(&[], &[vec![Arg::IV(a.clone())]]), out_z(&pa.derivative()).to_vec()),
                    "z_add" => {
                        let d = &pa - &pb;
                        if &d + &pb != pa || &pb - &pa != -&d {
                            bad += 1;
                            eprintln!("FUZZ MISMATCH z_sub identities");
                        }
                        (encode(&[], &[vec![Arg::IV(a.clone()), Arg::IV(b.clone())]]), out_z(&(&pa + &pb)).to_vec())
                    }
                    _ => (encode(&[], &[vec![Arg::IV(a.clone())]]), vec![fp_number(&pa.content())]),
                };
                let _ = Polynomial::<Number>::zero();
                check(&format!("{tag} bits={bits}"), tag, Lib::Flint, &buf, o, &mut bad, &Cmp::Exact)
            }
        };
        if ok {
            *done.entry(tag).or_default() += 1;
        }
    }
    let total: u64 = done.values().sum();
    println!("fuzz seed {seed}: {total} cases checked against FLINT, {bad} mismatches; {done:?}");
}

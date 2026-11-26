//! Binary fields `GF(2^m)`: `BinaryField`/`F2m` against NTL (GF2E) and OpenSSL (BN_GF2m), the
//! Gaussian normal bases (no reference has them), and Rabin's test against NTL's.

use crate::common::{self, Bench, Cmp};
use crate::ffi;
use crate::harness::{self, Op, Rng, Side, keep};
use a::math::field::{BasisConversion, BinaryField, Clmul, F2m, NormalBasis, NormalElement};

const COUNT: usize = 256;

struct Ours {
    f: &'static BinaryField,
    xs: Vec<F2m<'static>>,
    e: u64,
    acc: F2m<'static>,
    w: usize,
    sink: u64,
}

impl Ours {
    fn new(f: &'static BinaryField, xs: &[Vec<u64>], e: u64) -> Ours {
        let xs: Vec<F2m> = xs.iter().map(|x| f.element(x)).collect();
        Ours { f, acc: xs[0].clone(), xs, e, w: (f.degree() as usize).div_ceil(64), sink: 0 }
    }
    fn l(&self, x: &F2m) -> Vec<u64> {
        let mut v = x.words().to_vec();
        v.resize(self.w, 0);
        v
    }
}

impl Bench for Ours {
    fn name(&self) -> String {
        "ours".into()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let (x, y) = (&self.xs[i], &self.xs[(i + 1) % COUNT]);
        Some(match op {
            Op::Mul => (self.l(&(x * y)), 0),
            Op::Sqr => (self.l(&x.square()), 0),
            Op::Inv => (self.l(&x.inverse().unwrap()), 0),
            Op::Div => (self.l(&(x / y)), 0),
            Op::Sqrt => (self.l(&x.sqrt()), 0),
            Op::Trace => (vec![0; self.w], x.trace() as i32),
            Op::HTrace if self.f.degree() % 2 == 1 => (self.l(&x.half_trace()), 0),
            Op::QRoot => (self.l(&x.quadratic_root().unwrap()), 1),
            Op::PowU => (self.l(&x.pow_u64(self.e)), 0),
            _ => return None,
        })
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut acc = self.acc.clone();
        let mut s = 0u64;
        let xs = &self.xs;
        match op {
            Op::Mul => (0..iters).for_each(|i| acc *= &xs[i as usize & m]),
            Op::Sqr => (0..iters).for_each(|_| acc = acc.square()),
            Op::Inv => (0..iters).for_each(|i| acc += xs[i as usize & m].inverse().unwrap()),
            Op::Div => (0..iters).for_each(|i| acc /= &xs[i as usize & m]),
            Op::Sqrt => (0..iters).for_each(|i| acc += xs[i as usize & m].sqrt()),
            Op::Trace => (0..iters).for_each(|i| s ^= xs[i as usize & m].trace() as u64),
            Op::HTrace => (0..iters).for_each(|i| acc += xs[i as usize & m].half_trace()),
            Op::QRoot => (0..iters).for_each(|i| acc += xs[i as usize & m].quadratic_root().unwrap()),
            Op::PowU => (0..iters).for_each(|i| acc += xs[i as usize & m].pow_u64(self.e)),
            _ => {}
        }
        if acc.is_zero() {
            acc = xs[0].clone();
        }
        self.acc = keep(acc);
        self.sink = self.sink.wrapping_add(keep(s));
    }
}

/// The fields: the standards' moduli and larger ones found at run time.
fn fields() -> Vec<(String, Vec<u32>)> {
    let mut v: Vec<(String, Vec<u32>)> = vec![
        ("AES GF(2^8)".into(), vec![8, 4, 3, 1, 0]),
        ("GF(2^64)".into(), vec![64, 4, 3, 1, 0]),
        ("sect113".into(), vec![113, 9, 0]),
        ("GCM GF(2^128)".into(), vec![128, 7, 2, 1, 0]),
        ("sect131".into(), vec![131, 8, 3, 2, 0]),
        ("sect163".into(), vec![163, 7, 6, 3, 0]),
        ("sect193".into(), vec![193, 15, 0]),
        ("sect233".into(), vec![233, 74, 0]),
        ("sect283".into(), vec![283, 12, 7, 5, 0]),
        ("sect409".into(), vec![409, 87, 0]),
        ("sect571".into(), vec![571, 10, 5, 2, 0]),
    ];
    for m in [1024u32, 2048] {
        // found when its rows run (see `run`)
        v.push((format!("GF(2^{m})"), vec![m]));
    }
    v
}

/// The first irreducible trinomial `x^m + x^a + 1` (smallest `a`), else the first pentanomial
/// `x^m + x^a + x^b + x^c + 1` in the order of `(a, b, c)`.
fn sparse_irreducible(m: u32) -> Vec<u32> {
    for a in 1..m / 2 + 1 {
        if BinaryField::is_irreducible(&[m, a, 0]) {
            return vec![m, a, 0];
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
    unreachable!()
}

fn random(rng: &mut Rng, m: u32) -> Vec<u64> {
    let w = (m as usize).div_ceil(64);
    let mut v: Vec<u64> = (0..w).map(|_| rng.next()).collect();
    if m % 64 != 0 {
        v[w - 1] &= (1u64 << (m % 64)) - 1;
    }
    if v.iter().all(|&x| x == 0) {
        v[0] = 1;
    }
    v
}

fn terms(e: &[u32]) -> String {
    e.iter().map(|&k| if k == 0 { "1".to_string() } else { format!("x^{k}") }).collect::<Vec<_>>().join("+")
}

/// Runs the binary-field rows (polynomial bases; `only` names degrees, all if empty).
pub fn run(only: &[String]) {
    let group = "binary";
    for (name, exps) in fields() {
        let m = exps[0];
        if !only.iter().any(|o| *o == m.to_string()) && !only.is_empty() {
            continue;
        }
        let exps = if exps.len() == 1 { sparse_irreducible(m) } else { exps };
        let f: &'static BinaryField = Box::leak(Box::new(BinaryField::new(&exps)));
        let w = (m as usize).div_ceil(64);
        let mut rng = Rng(0xb1 ^ m as u64);
        let xs: Vec<Vec<u64>> = (0..COUNT).map(|_| random(&mut rng, m)).collect();
        let e = rng.next() >> 1; // NTL takes a long
        // inputs of trace 0, for z² + z = c
        let delta = (0..m).map(|i| f.element(&unit(i, w))).find(|d| d.trace() == 1).unwrap();
        let ys: Vec<Vec<u64>> = xs
            .iter()
            .map(|x| {
                let a = f.element(x);
                let a = if a.trace() == 1 { &a + &delta } else { a };
                let mut v = a.words().to_vec();
                v.resize(w, 0);
                v
            })
            .collect();
        let params = format!("{name}: {} ({w} words); {:?}", terms(&exps), f).replace('\t', " ");
        let params = params.chars().take(200).collect::<String>();
        let flat: Vec<u64> = xs.iter().flatten().copied().collect();
        let flat_y: Vec<u64> = ys.iter().flatten().copied().collect();
        let mut sides: Vec<Box<dyn Bench>> = vec![Box::new(Ours::new(f, &xs, e))];
        let mut qsides: Vec<Box<dyn Bench>> = vec![Box::new(Ours::new(f, &ys, e))];
        for r in ffi::bin_refs() {
            if let Some(c) = ffi::bin_new(&r, &exps, &flat, COUNT, w, e) {
                sides.push(Box::new(c));
            }
            if let Some(c) = ffi::bin_new(&r, &exps, &flat_y, COUNT, w, e) {
                qsides.push(Box::new(c));
            }
        }
        let zero = vec![0u64; w];
        for (op, cmp) in [
            (Op::Mul, Cmp::Exact),
            (Op::Sqr, Cmp::Exact),
            (Op::Inv, Cmp::Exact),
            (Op::Div, Cmp::Exact),
            (Op::Sqrt, Cmp::Exact),
            (Op::Trace, Cmp::Flag),
            (Op::HTrace, Cmp::Exact),
            (Op::PowU, Cmp::Exact),
        ] {
            if op == Op::HTrace && m % 2 == 0 {
                continue;
            }
            let pr = if op == Op::PowU { format!("{params}; 63-bit exponent") } else { params.clone() };
            common::compare(group, op.name(), &pr, &mut sides, op, 12, cmp, &zero);
        }
        common::compare(group, "quadratic_root", &params, &mut qsides, Op::QRoot, 12, Cmp::UpToOne, &zero);
        // Rabin's test against NTL's
        irreducibility(group, &exps, &params);
    }
    if only.is_empty() || only.iter().any(|o| o == "normal") {
        normal();
    }
    if only.is_empty() || only.iter().any(|o| o == "clmul") {
        clmul();
    }
}

fn unit(i: u32, w: usize) -> Vec<u64> {
    let mut v = vec![0u64; w];
    v[(i / 64) as usize] = 1 << (i % 64);
    v
}

fn irreducibility(group: &str, exps: &[u32], params: &str) {
    let mut names = vec!["ours".to_string()];
    let mut sides: Vec<Side> = vec![Side::new("ours", |n| {
        for _ in 0..n {
            keep(BinaryField::is_irreducible(keep(exps)));
        }
    })];
    #[cfg(has_ntl)]
    {
        // SAFETY: the exponents are a live slice.
        let r = unsafe { ffi::ntl_gf2x_irred(exps.as_ptr(), exps.len(), 1) };
        assert_eq!(r, 1, "NTL finds the modulus irreducible");
        names.push("NTL IterIrredTest".into());
        sides.push(Side::new("ntl", |n| {
            // SAFETY: as above.
            keep(unsafe { ffi::ntl_gf2x_irred(exps.as_ptr(), exps.len(), n) });
        }));
    }
    let stats = harness::measure(&mut sides);
    let notes = vec![String::new(); names.len()];
    harness::report(group, "is_irreducible", params, &names, &stats, &notes);
}

/// The normal bases: no reference has Gaussian normal bases, so these rows are ours alone.
fn normal() {
    let group = "normal";
    for (m, f) in [
        (163u32, vec![163, 7, 6, 3, 0]),
        (233, vec![233, 74, 0]),
        (283, vec![283, 12, 7, 5, 0]),
        (409, vec![409, 87, 0]),
        (571, vec![571, 10, 5, 2, 0]),
    ] {
        let pb: &'static BinaryField = Box::leak(Box::new(BinaryField::new(&f)));
        let nb: &'static NormalBasis = Box::leak(Box::new(NormalBasis::new(m)));
        let params = format!("GF(2^{m}), Gaussian normal basis of type {}; polynomial basis {}", nb.gaussian_type(), terms(&f));
        let conv: &'static BasisConversion = Box::leak(Box::new(nb.conversion(pb)));
        let mut rng = Rng(0x4b ^ m as u64);
        let xs: Vec<NormalElement> = (0..COUNT).map(|_| nb.element(&random(&mut rng, m))).collect();
        let ps: Vec<F2m> = xs.iter().map(|x| conv.to_polynomial(x)).collect();
        let e = rng.next();
        let msk = COUNT - 1;
        let one = |name: &str, run: &mut dyn FnMut(u64)| {
            let mut s = vec![Side::new("ours", |n| run(n))];
            let st = harness::measure(&mut s);
            harness::report(group, name, &params, &["ours".into()], &st, &[String::new()]);
        };
        let mut acc = xs[0].clone();
        one("mul", &mut |n| (0..n).for_each(|i| acc *= &xs[i as usize & msk]));
        one("square", &mut |n| (0..n).for_each(|_| acc = acc.square()));
        one("inverse", &mut |n| (0..n).for_each(|i| acc += xs[i as usize & msk].inverse().unwrap()));
        one("sqrt", &mut |n| (0..n).for_each(|i| acc += xs[i as usize & msk].sqrt()));
        let mut t = 0u64;
        one("trace", &mut |n| (0..n).for_each(|i| t ^= xs[i as usize & msk].trace() as u64));
        keep(t);
        one("pow_u64", &mut |n| (0..n).for_each(|i| acc += xs[i as usize & msk].pow_u64(e)));
        one("to_normal", &mut |n| (0..n).for_each(|i| acc += conv.to_normal(&ps[i as usize & msk])));
        let mut pacc = ps[0].clone();
        one("to_polynomial", &mut |n| (0..n).for_each(|i| pacc += conv.to_polynomial(&xs[i as usize & msk])));
        one("NormalBasis::new", &mut |n| (0..n).for_each(|_| {
            keep(NormalBasis::new(m));
        }));
        one("NormalBasis::new + conversion(&field)", &mut |n| (0..n).for_each(|_| {
            let nb2 = NormalBasis::new(m);
            keep(nb2.conversion(pb));
        }));
        keep((acc, pacc));
    }
}

/// The carry-less product of two words (no reference exposes one alone).
fn clmul() {
    let k = Clmul::detect();
    let mut rng = Rng(0xc1);
    let xs: Vec<u64> = (0..COUNT).map(|_| rng.next()).collect();
    let mut sides = vec![Side::new("ours", |n| {
        let mut a = 0u128;
        let mut x = xs[0];
        for i in 0..n {
            let p = k.mul(x, xs[i as usize & (COUNT - 1)]);
            a ^= p;
            x = p as u64 ^ (p >> 64) as u64 | 1;
        }
        keep(a);
    })];
    let st = harness::measure(&mut sides);
    harness::report("binary", "Clmul::mul", &format!("{k:?}, a dependent chain of 64x64 products"), &["ours".into()], &st, &[String::new()]);
}

//! Extension fields `GF(p^k)`: `ExtensionField`/`Fpk` against FLINT (fq_zech, fq_nmod, fq) and NTL
//! (zz_pE, ZZ_pE), on the modulus our `with_degree` chooses, which every side is given.

use crate::common::{self, Bench, Cmp, below, hex, le_limbs, limbs, number};
use crate::consts::PRIMES;
use crate::ffi;
use crate::harness::{Op, Rng, keep};
use a::math::field::{ExtensionField, Fp, Fpk, PrimeField};
use a::math::numbers::Number;

const COUNT: usize = 256;

struct Ours {
    f: &'static ExtensionField,
    xs: Vec<Fpk<'static>>,
    sq: Vec<Fpk<'static>>,
    e: Number,
    acc: Fpk<'static>,
    nacc: Fp<'static>,
    n: usize,
    k: usize,
    sink: u64,
}

impl Ours {
    fn l(&self, x: &Fpk) -> Vec<u64> {
        x.coeffs().iter().flat_map(|c| le_limbs(&c.to_bytes_le(), self.n)).collect()
    }
    fn s(&self, x: &Fp) -> Vec<u64> {
        let mut v = le_limbs(&x.to_bytes_le(), self.n);
        v.resize(self.n * self.k, 0);
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
            Op::Add => (self.l(&(x + y)), 0),
            Op::Sqr => (self.l(&x.square()), 0),
            Op::Inv => (self.l(&x.inverse().unwrap()), 0),
            Op::Frob => (self.l(&x.frobenius(1)), 0),
            Op::Norm => (self.s(&x.norm()), 0),
            Op::Trace => (self.s(&x.trace()), 0),
            Op::Sqrt => (self.l(&self.sq[i].sqrt().unwrap()), 1),
            Op::IsSq => (vec![0; self.n * self.k], x.is_square() as i32),
            Op::Pow => (self.l(&x.pow(&self.e)), 0),
            _ => return None,
        })
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut acc = self.acc.clone();
        let mut nacc = self.nacc.clone();
        let mut s = 0i64;
        let xs = &self.xs;
        match op {
            Op::Mul => (0..iters).for_each(|i| acc *= &xs[i as usize & m]),
            Op::Add => (0..iters).for_each(|i| acc += &xs[i as usize & m]),
            Op::Sqr => (0..iters).for_each(|_| acc = acc.square()),
            Op::Inv => (0..iters).for_each(|i| acc += xs[i as usize & m].inverse().unwrap()),
            Op::Frob => (0..iters).for_each(|i| acc += xs[i as usize & m].frobenius(1)),
            Op::Norm => (0..iters).for_each(|i| nacc += xs[i as usize & m].norm()),
            Op::Trace => (0..iters).for_each(|i| nacc += xs[i as usize & m].trace()),
            Op::Sqrt => (0..iters).for_each(|i| acc += self.sq[i as usize & m].sqrt().unwrap()),
            Op::IsSq => (0..iters).for_each(|i| s += xs[i as usize & m].is_square() as i64),
            Op::Pow => (0..iters).for_each(|i| acc += xs[i as usize & m].pow(&self.e)),
            _ => {}
        }
        if acc.is_zero() {
            acc = self.xs[0].clone();
        }
        self.acc = keep(acc);
        self.nacc = keep(nacc);
        self.sink = self.sink.wrapping_add(keep(s) as u64);
        let _ = self.f;
    }
}

/// The cases: (prime's name or a small prime, degrees).
const CASES: &[(&str, &[usize])] = &[
    ("3", &[5, 12]),
    ("1000000007", &[2, 8, 24, 64]),
    ("bn254", &[2, 6, 12, 24, 64]),
    ("bls381", &[2, 12]),
    ("p600", &[3, 8]),
];

/// Runs the extension-field rows of the cases named in `only` (`bn254`, `bn254/12`, …; all if
/// empty).
pub fn run(only: &[String]) {
    let group = "ext";
    for &(pname, ks) in CASES {
        let (p, base): (Number, &'static PrimeField) = match pname {
            "bn254" => (PrimeField::bn254_base().modulus().clone(), PrimeField::bn254_base()),
            "bls381" => (PrimeField::bls12_381_base().modulus().clone(), PrimeField::bls12_381_base()),
            "p600" => {
                let p = hex(PRIMES.iter().find(|r| r.0 == "p600").unwrap().1);
                let f: &'static PrimeField = Box::leak(Box::new(PrimeField::new(&p)));
                (p, f)
            }
            small => {
                let p = Number::from(small.parse::<u64>().unwrap());
                let f: &'static PrimeField = Box::leak(Box::new(PrimeField::new(&p)));
                (p, f)
            }
        };
        let bits = base.bits();
        let n = bits.div_ceil(64) as usize;
        let pl = limbs(&p, n);
        for &k in ks {
            let tag = format!("{pname}/{k}");
            if !only.is_empty() && !only.iter().any(|o| *o == tag || o == pname) {
                continue;
            }
            let f: &'static ExtensionField = Box::leak(Box::new(ExtensionField::with_degree(base, k)));
            let g: Vec<u64> = f.modulus().coeffs().iter().flat_map(|c| limbs(c, n)).collect();
            let mut rng = Rng(0xe47 ^ (bits << 8) ^ k as u64);
            let xs: Vec<Vec<u64>> = (0..COUNT).map(|_| (0..k).flat_map(|_| below(&mut rng, &pl)).collect()).collect();
            let e: Vec<u64> = (0..4).map(|_| rng.next()).collect();
            let fx: Vec<Fpk> =
                xs.iter().map(|x| f.element(&x.chunks(n).map(number).collect::<Vec<_>>())).collect();
            let ours = Ours {
                f,
                sq: fx.iter().map(|x| x.square()).collect(),
                acc: fx[0].clone(),
                nacc: f.base().zero(),
                xs: fx,
                e: number(&e),
                n,
                k,
                sink: 0,
            };
            let mut sides: Vec<Box<dyn Bench>> = vec![Box::new(ours)];
            let flat: Vec<u64> = xs.iter().flatten().copied().collect();
            for r in ffi::ext_refs() {
                if let Some(c) = ffi::ext_new(&r, &pl, &g, k, &flat, COUNT, &e) {
                    sides.push(Box::new(c));
                }
            }
            let terms = f.modulus().coeffs().iter().filter(|c| **c != 0u8).count();
            let pdesc = match pname {
                "bn254" => "BN254 base p".to_string(),
                "bls381" => "BLS12-381 base p".into(),
                "p600" => "a 600-bit p".into(),
                s => format!("p = {s}"),
            };
            let params = format!("{pdesc}, k = {k} (q = p^k of about {} bits); modulus with {terms} terms", bits * k as u64);
            let mut ops: Vec<(Op, Cmp)> = vec![
                (Op::Add, Cmp::Exact),
                (Op::Mul, Cmp::Exact),
                (Op::Sqr, Cmp::Exact),
                (Op::Inv, Cmp::Exact),
                (Op::Frob, Cmp::Exact),
                (Op::Norm, Cmp::Limbs),
                (Op::Trace, Cmp::Limbs),
                (Op::IsSq, Cmp::Flag),
                (Op::Pow, Cmp::Exact),
            ];
            if k <= 24 {
                ops.push((Op::Sqrt, Cmp::UpToSign));
            }
            for (op, cmp) in ops {
                let (name, pr) = match op {
                    Op::Frob => ("frobenius(1)".to_string(), params.clone()),
                    Op::Sqrt => ("sqrt (of a square)".into(), params.clone()),
                    Op::Pow => ("pow".into(), format!("{params}; 256-bit exponent")),
                    _ => (op.name().to_string(), params.clone()),
                };
                let checks = if k >= 24 { 4 } else { 12 };
                common::compare(group, &name, &pr, &mut sides, op, checks, cmp, &pl);
            }
        }
    }
}

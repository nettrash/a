//! Prime fields: `PrimeField`/`Fp` against arkworks, blst, the bls12_381 crate, FLINT, NTL, GMP and
//! OpenSSL, at 64 to 4096 bits.

use crate::arkf;
use crate::common::{self, Bench, Cmp, below, hex, le_limbs, limbs, limbs_le, number};
use crate::consts::PRIMES;
use crate::ffi;
use crate::harness::{Op, Rng, keep};
use a::math::field::{Fp, PrimeField, TypedFp, TypedPrime};
use a::math::numbers::Number;
use ark_ff::{BigInteger, PrimeField as ArkPrime};

const COUNT: usize = 256;
const CHECKS: usize = 16;

/// Our side.
struct Ours {
    f: &'static PrimeField,
    xs: Vec<Fp<'static>>,
    sq: Vec<Fp<'static>>,
    bytes: Vec<Vec<u8>>,
    e: Number,
    e64: u64,
    acc: Fp<'static>,
    n: usize,
    sink: u64,
}

impl Ours {
    fn limbs(&self, x: &Fp) -> Vec<u64> {
        le_limbs(&x.to_bytes_le(), self.n)
    }
}

impl Bench for Ours {
    fn name(&self) -> String {
        "ours".into()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let (x, y) = (&self.xs[i], &self.xs[(i + 1) % COUNT]);
        let l = |s: &Ours, v: &Fp| s.limbs(v);
        Some(match op {
            Op::Mul => (l(self, &(x * y)), 0),
            Op::Sqr => (l(self, &x.square()), 0),
            Op::Add => (l(self, &(x + y)), 0),
            Op::Sub => (l(self, &(x - y)), 0),
            Op::Neg => (l(self, &(-x)), 0),
            Op::Dbl => (l(self, &x.double()), 0),
            Op::Inv => (l(self, &x.inverse().unwrap()), 0),
            Op::Sqrt => (l(self, &self.sq[i].sqrt().unwrap()), 1),
            Op::Leg => (vec![0; self.n], x.legendre()),
            Op::IsSq => (vec![0; self.n], x.is_square() as i32),
            Op::Pow => (l(self, &x.pow(&self.e)), 0),
            Op::PowU => (l(self, &x.pow_u64(self.e64)), 0),
            Op::Batch => (l(self, &self.f.batch_invert(&self.xs)[i]), 0),
            Op::FromBytes => (l(self, &self.f.from_bytes_be(&self.bytes[i]).unwrap()), 0),
            Op::ToBytes => (le_limbs(&rev(&x.to_bytes_be()), self.n), 0),
            _ => return None,
        })
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut acc = self.acc.clone();
        let mut s = 0i64;
        match op {
            Op::Mul => (0..iters).for_each(|i| acc *= &self.xs[i as usize & m]),
            Op::Sqr => (0..iters).for_each(|_| acc.square_in_place()),
            Op::Add => (0..iters).for_each(|i| acc += &self.xs[i as usize & m]),
            Op::Sub => (0..iters).for_each(|i| acc -= &self.xs[i as usize & m]),
            Op::Neg => (0..iters).for_each(|_| acc.neg_in_place()),
            Op::Dbl => (0..iters).for_each(|_| acc.double_in_place()),
            Op::Inv => (0..iters).for_each(|i| acc += self.xs[i as usize & m].inverse().unwrap()),
            Op::Sqrt => (0..iters).for_each(|i| acc += self.sq[i as usize & m].sqrt().unwrap()),
            Op::Leg => (0..iters).for_each(|i| s += self.xs[i as usize & m].legendre() as i64),
            Op::IsSq => (0..iters).for_each(|i| s += self.xs[i as usize & m].is_square() as i64),
            Op::Pow => (0..iters).for_each(|i| acc += self.xs[i as usize & m].pow(&self.e)),
            Op::PowU => (0..iters).for_each(|i| acc += self.xs[i as usize & m].pow_u64(self.e64)),
            Op::Batch => {
                // `iters` batches of COUNT elements; `compare` divides the time by COUNT.
                for _ in 0..iters {
                    let v = self.f.batch_invert(&self.xs);
                    acc += &v[0];
                }
            }
            Op::FromBytes => {
                (0..iters).for_each(|i| acc += self.f.from_bytes_be(&self.bytes[i as usize & m]).unwrap())
            }
            Op::ToBytes => (0..iters).for_each(|i| s += keep(self.xs[i as usize & m].to_bytes_be()).len() as i64),
            _ => {}
        }
        if acc.is_zero() {
            acc = self.xs[0].clone();
        }
        self.acc = keep(acc);
        self.sink = self.sink.wrapping_add(keep(s) as u64);
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::Batch => format!("{COUNT} elements a batch, time per element"),
            Op::Sqr | Op::Neg | Op::Dbl => "ours in place (square_in_place, neg_in_place, double_in_place)".into(),
            _ => String::new(),
        }
    }
}

/// The typed form of a named prime, `TypedFp<P, N>`: the same operations as `Ours`.
struct Typed<P, const N: usize> {
    xs: Vec<TypedFp<P, N>>,
    sq: Vec<TypedFp<P, N>>,
    bytes: Vec<Vec<u8>>,
    e: Number,
    e64: u64,
    acc: TypedFp<P, N>,
    n: usize,
    sink: u64,
}

impl<P: TypedPrime<N>, const N: usize> Typed<P, N> {
    fn new(xs: &[Vec<u64>], bytes: &[Vec<u8>], e: &[u64], e64: u64, n: usize) -> Self {
        let blen = TypedFp::<P, N>::byte_len();
        let xs: Vec<TypedFp<P, N>> =
            xs.iter().map(|x| TypedFp::<P, N>::from_bytes_le(&limbs_le(x, blen)).unwrap()).collect();
        Typed {
            sq: xs.iter().map(|x| x.square()).collect(),
            acc: xs[0],
            xs,
            bytes: bytes.to_vec(),
            e: number(e),
            e64,
            n,
            sink: 0,
        }
    }
    fn l(&self, x: &TypedFp<P, N>) -> Vec<u64> {
        le_limbs(&x.to_bytes_le(), self.n)
    }
}

impl<P: TypedPrime<N>, const N: usize> Bench for Typed<P, N> {
    fn name(&self) -> String {
        "ours (TypedFp)".into()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let (x, y) = (self.xs[i], self.xs[(i + 1) % COUNT]);
        Some(match op {
            Op::Mul => (self.l(&(x * y)), 0),
            Op::Sqr => (self.l(&x.square()), 0),
            Op::Add => (self.l(&(x + y)), 0),
            Op::Sub => (self.l(&(x - y)), 0),
            Op::Neg => (self.l(&(-x)), 0),
            Op::Dbl => (self.l(&x.double()), 0),
            Op::Inv => (self.l(&x.inverse().unwrap()), 0),
            Op::Sqrt => (self.l(&self.sq[i].sqrt().unwrap()), 1),
            Op::Leg => (vec![0; self.n], x.legendre()),
            Op::IsSq => (vec![0; self.n], x.is_square() as i32),
            Op::Pow => (self.l(&x.pow(&self.e)), 0),
            Op::PowU => (self.l(&x.pow_u64(self.e64)), 0),
            Op::Batch => (self.l(&TypedFp::<P, N>::batch_invert(&self.xs)[i]), 0),
            Op::FromBytes => (self.l(&TypedFp::<P, N>::from_bytes_be(&self.bytes[i]).unwrap()), 0),
            Op::ToBytes => (le_limbs(&rev(&x.to_bytes_be()), self.n), 0),
            _ => return None,
        })
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut acc = self.acc;
        let mut s = 0i64;
        match op {
            Op::Mul => (0..iters).for_each(|i| acc *= self.xs[i as usize & m]),
            Op::Sqr => (0..iters).for_each(|_| acc = acc.square()),
            Op::Add => (0..iters).for_each(|i| acc += self.xs[i as usize & m]),
            Op::Sub => (0..iters).for_each(|i| acc -= self.xs[i as usize & m]),
            Op::Neg => (0..iters).for_each(|_| acc = -acc),
            Op::Dbl => (0..iters).for_each(|_| acc = acc.double()),
            Op::Inv => (0..iters).for_each(|i| acc += self.xs[i as usize & m].inverse().unwrap()),
            Op::Sqrt => (0..iters).for_each(|i| acc += self.sq[i as usize & m].sqrt().unwrap()),
            Op::Leg => (0..iters).for_each(|i| s += self.xs[i as usize & m].legendre() as i64),
            Op::IsSq => (0..iters).for_each(|i| s += self.xs[i as usize & m].is_square() as i64),
            Op::Pow => (0..iters).for_each(|i| acc += self.xs[i as usize & m].pow(&self.e)),
            Op::PowU => (0..iters).for_each(|i| acc += self.xs[i as usize & m].pow_u64(self.e64)),
            Op::Batch => {
                for _ in 0..iters {
                    let v = TypedFp::<P, N>::batch_invert(&self.xs);
                    acc += v[0];
                }
            }
            Op::FromBytes => (0..iters)
                .for_each(|i| acc += TypedFp::<P, N>::from_bytes_be(&self.bytes[i as usize & m]).unwrap()),
            Op::ToBytes => (0..iters).for_each(|i| s += keep(self.xs[i as usize & m].to_bytes_be()).len() as i64),
            _ => {}
        }
        if acc.is_zero() {
            acc = self.xs[0];
        }
        self.acc = keep(acc);
        self.sink = self.sink.wrapping_add(keep(s) as u64);
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::Batch => format!("{COUNT} elements a batch, time per element"),
            _ => String::new(),
        }
    }
}

/// The typed side of a named case, if it has one.
fn typed_side(name: &str, xs: &[Vec<u64>], bytes: &[Vec<u8>], e: &[u64], e64: u64, n: usize) -> Option<Box<dyn Bench>> {
    use a::math::field::named::*;
    Some(match name {
        "p64" => Box::new(Typed::<Goldilocks, 1>::new(xs, bytes, e, e64, n)),
        "bn254" => Box::new(Typed::<Bn254Base, 4>::new(xs, bytes, e, e64, n)),
        "bn254r" => Box::new(Typed::<Bn254Scalar, 4>::new(xs, bytes, e, e64, n)),
        "bls381" => Box::new(Typed::<Bls12_381Base, 6>::new(xs, bytes, e, e64, n)),
        "bls381r" => Box::new(Typed::<Bls12_381Scalar, 4>::new(xs, bytes, e, e64, n)),
        "c25519" => Box::new(Typed::<Curve25519Base, 4>::new(xs, bytes, e, e64, n)),
        "secp256k1" => Box::new(Typed::<Secp256k1Base, 4>::new(xs, bytes, e, e64, n)),
        "p256" => Box::new(Typed::<P256Base, 4>::new(xs, bytes, e, e64, n)),
        "p384" => Box::new(Typed::<P384Base, 6>::new(xs, bytes, e, e64, n)),
        _ => return None,
    })
}

fn rev(b: &[u8]) -> Vec<u8> {
    b.iter().rev().copied().collect()
}

/// An arkworks field.
struct Ark<F: ArkPrime> {
    xs: Vec<F>,
    sq: Vec<F>,
    bytes: Vec<Vec<u8>>,
    e: Vec<u64>,
    e64: u64,
    acc: F,
    n: usize,
    sink: u64,
}

impl<F: ArkPrime> Ark<F> {
    fn new(xs: &[Vec<u64>], bytes: &[Vec<u8>], e: &[u64], e64: u64, n: usize) -> Self {
        let xs: Vec<F> = xs.iter().map(|x| F::from_le_bytes_mod_order(&limbs_le(x, 8 * n))).collect();
        let sq = xs.iter().map(|x| x.square()).collect();
        Ark { acc: xs[0], xs, sq, bytes: bytes.to_vec(), e: e.to_vec(), e64, n, sink: 0 }
    }
    fn l(&self, x: &F) -> Vec<u64> {
        x.into_bigint().as_ref()[..self.n].to_vec()
    }
}

impl<F: ArkPrime> Bench for Ark<F> {
    fn name(&self) -> String {
        "arkworks 0.5".into()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let (x, y) = (self.xs[i], self.xs[(i + 1) % COUNT]);
        Some(match op {
            Op::Mul => (self.l(&(x * y)), 0),
            Op::Sqr => (self.l(&x.square()), 0),
            Op::Add => (self.l(&(x + y)), 0),
            Op::Sub => (self.l(&(x - y)), 0),
            Op::Neg => (self.l(&(-x)), 0),
            Op::Dbl => (self.l(&x.double()), 0),
            Op::Inv => (self.l(&x.inverse().unwrap()), 0),
            Op::Sqrt => (self.l(&self.sq[i].sqrt().unwrap()), 1),
            Op::Leg => {
                let s = x.legendre();
                (vec![0; self.n], if s.is_zero() { 0 } else if s.is_qr() { 1 } else { -1 })
            }
            Op::IsSq => (vec![0; self.n], !x.legendre().is_qnr() as i32),
            Op::Pow => (self.l(&x.pow(&self.e)), 0),
            Op::PowU => (self.l(&x.pow([self.e64])), 0),
            Op::Batch => {
                let mut v = self.xs.clone();
                ark_ff::batch_inversion(&mut v);
                (self.l(&v[i]), 0)
            }
            Op::FromBytes => (self.l(&F::from_be_bytes_mod_order(&self.bytes[i])), 0),
            Op::ToBytes => {
                let b = x.into_bigint().to_bytes_be();
                let len = self.bytes[0].len();
                (le_limbs(&rev(&b[b.len() - len..]), self.n), 0)
            }
            _ => return None,
        })
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut acc = self.acc;
        let mut s = 0i64;
        match op {
            Op::Mul => (0..iters).for_each(|i| acc *= &self.xs[i as usize & m]),
            Op::Sqr => (0..iters).for_each(|_| {
                acc.square_in_place();
            }),
            Op::Add => (0..iters).for_each(|i| acc += &self.xs[i as usize & m]),
            Op::Sub => (0..iters).for_each(|i| acc -= &self.xs[i as usize & m]),
            Op::Neg => (0..iters).for_each(|_| acc = -acc),
            Op::Dbl => (0..iters).for_each(|_| {
                acc.double_in_place();
            }),
            Op::Inv => (0..iters).for_each(|i| acc += self.xs[i as usize & m].inverse().unwrap()),
            Op::Sqrt => (0..iters).for_each(|i| acc += self.sq[i as usize & m].sqrt().unwrap()),
            Op::Leg => (0..iters).for_each(|i| s += self.xs[i as usize & m].legendre().is_qr() as i64),
            Op::IsSq => (0..iters).for_each(|i| s += !self.xs[i as usize & m].legendre().is_qnr() as i64),
            Op::Pow => (0..iters).for_each(|i| acc += self.xs[i as usize & m].pow(&self.e)),
            Op::PowU => (0..iters).for_each(|i| acc += self.xs[i as usize & m].pow([self.e64])),
            Op::Batch => {
                for _ in 0..iters {
                    let mut v = self.xs.clone();
                    ark_ff::batch_inversion(&mut v);
                    acc += &v[0];
                }
            }
            Op::FromBytes => {
                (0..iters).for_each(|i| acc += F::from_be_bytes_mod_order(&self.bytes[i as usize & m]))
            }
            Op::ToBytes => {
                (0..iters).for_each(|i| s += keep(self.xs[i as usize & m].into_bigint().to_bytes_be()).len() as i64)
            }
            _ => {}
        }
        if acc.is_zero() {
            acc = self.xs[0];
        }
        self.acc = keep(acc);
        self.sink = self.sink.wrapping_add(keep(s) as u64);
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::Leg | Op::IsSq => "Euler's criterion, a power".into(),
            Op::FromBytes => "from_be_bytes_mod_order (reduces any length)".into(),
            _ => String::new(),
        }
    }
}

/// blst's fields of BLS12-381: `fr` for the scalar field, otherwise the base field.
struct Blst {
    fr: bool,
    ct: bool,
    xs: Vec<[u64; 6]>,
    sq: Vec<[u64; 6]>,
    bytes: Vec<Vec<u8>>,
    acc: [u64; 6],
    n: usize,
    sink: u64,
}

impl Blst {
    fn new(fr: bool, ct: bool, xs: &[Vec<u64>], bytes: &[Vec<u8>]) -> Self {
        let n = if fr { 4 } else { 6 };
        let mut b = Blst { fr, ct, xs: vec![], sq: vec![], bytes: bytes.to_vec(), acc: [0; 6], n, sink: 0 };
        b.xs = xs.iter().map(|x| b.from_le(x)).collect();
        b.sq = b.xs.clone();
        for i in 0..COUNT {
            let x = b.xs[i];
            b.sq[i] = b.mul(&x, &x);
        }
        b.acc = b.xs[0];
        b
    }
    fn from_le(&self, x: &[u64]) -> [u64; 6] {
        let mut r = [0u64; 6];
        // SAFETY: blst_fp and blst_fr are arrays of 6 and 4 limbs; `r` holds 6.
        unsafe {
            if self.fr {
                blst::blst_fr_from_uint64(r.as_mut_ptr() as *mut blst::blst_fr, x.as_ptr());
            } else {
                blst::blst_fp_from_uint64(r.as_mut_ptr() as *mut blst::blst_fp, x.as_ptr());
            }
        }
        r
    }
    fn to_le(&self, x: &[u64; 6]) -> Vec<u64> {
        let mut r = [0u64; 6];
        // SAFETY: as above.
        unsafe {
            if self.fr {
                blst::blst_uint64_from_fr(r.as_mut_ptr(), x.as_ptr() as *const blst::blst_fr);
            } else {
                blst::blst_uint64_from_fp(r.as_mut_ptr(), x.as_ptr() as *const blst::blst_fp);
            }
        }
        r[..self.n].to_vec()
    }
    #[inline(always)]
    fn mul(&self, a: &[u64; 6], b: &[u64; 6]) -> [u64; 6] {
        let mut r = [0u64; 6];
        // SAFETY: limb arrays of the right size; blst allows aliasing.
        unsafe {
            if self.fr {
                blst::blst_fr_mul(r.as_mut_ptr() as *mut _, a.as_ptr() as *const _, b.as_ptr() as *const _);
            } else {
                blst::blst_fp_mul(r.as_mut_ptr() as *mut _, a.as_ptr() as *const _, b.as_ptr() as *const _);
            }
        }
        r
    }
}

macro_rules! blst_call {
    ($s:expr, $fr:ident, $fp:ident, $($arg:expr),*) => {
        // SAFETY: limb arrays of the right size; blst allows its outputs to alias its inputs.
        unsafe {
            if $s.fr { blst::$fr($($arg as _),*) } else { blst::$fp($($arg as _),*) }
        }
    };
}

impl Bench for Blst {
    fn name(&self) -> String {
        if self.ct { "blst 0.3 (constant-time)".into() } else { "blst 0.3".into() }
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let (x, y) = (self.xs[i], self.xs[(i + 1) % COUNT]);
        let mut r = [0u64; 6];
        let rp = r.as_mut_ptr();
        let mut flag = 0;
        match op {
            Op::Mul if !self.ct => blst_call!(self, blst_fr_mul, blst_fp_mul, rp, x.as_ptr(), y.as_ptr()),
            Op::Sqr if !self.ct => blst_call!(self, blst_fr_sqr, blst_fp_sqr, rp, x.as_ptr()),
            Op::Add if !self.ct => blst_call!(self, blst_fr_add, blst_fp_add, rp, x.as_ptr(), y.as_ptr()),
            Op::Sub if !self.ct => blst_call!(self, blst_fr_sub, blst_fp_sub, rp, x.as_ptr(), y.as_ptr()),
            Op::Inv if self.ct => blst_call!(self, blst_fr_inverse, blst_fp_inverse, rp, x.as_ptr()),
            Op::Inv => blst_call!(self, blst_fr_eucl_inverse, blst_fp_eucl_inverse, rp, x.as_ptr()),
            Op::Sqrt if !self.fr && self.ct => {
                // SAFETY: blst_fp arrays.
                flag = unsafe { blst::blst_fp_sqrt(rp as _, self.sq[i].as_ptr() as _) } as i32;
            }
            Op::IsSq if !self.fr && self.ct => {
                // SAFETY: a blst_fp array.
                flag = unsafe { blst::blst_fp_is_square(x.as_ptr() as _) } as i32;
            }
            Op::FromBytes if !self.ct => {
                let b = &self.bytes[i];
                if self.fr {
                    let mut sc = blst::blst_scalar::default();
                    // SAFETY: 32 bytes in, a scalar out, then a blst_fr.
                    unsafe {
                        blst::blst_scalar_from_bendian(&mut sc, b.as_ptr());
                        blst::blst_fr_from_scalar(rp as _, &sc);
                    }
                } else {
                    // SAFETY: 48 bytes in, a blst_fp out.
                    unsafe { blst::blst_fp_from_bendian(rp as _, b.as_ptr()) };
                }
            }
            Op::ToBytes if !self.ct => {
                let mut b = [0u8; 48];
                if self.fr {
                    let mut sc = blst::blst_scalar::default();
                    // SAFETY: a blst_fr in, a scalar, then 32 bytes out.
                    unsafe {
                        blst::blst_scalar_from_fr(&mut sc, x.as_ptr() as _);
                        blst::blst_bendian_from_scalar(b.as_mut_ptr(), &sc);
                    }
                    return Some((le_limbs(&rev(&b[..32]), 4), 0));
                } else {
                    // SAFETY: a blst_fp in, 48 bytes out.
                    unsafe { blst::blst_bendian_from_fp(b.as_mut_ptr(), x.as_ptr() as _) };
                    return Some((le_limbs(&rev(&b), 6), 0));
                }
            }
            _ => return None,
        }
        Some((self.to_le(&r), flag))
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut acc = self.acc;
        let mut t = [0u64; 6];
        let a = acc.as_mut_ptr();
        let tp = t.as_mut_ptr();
        let mut s = 0i64;
        match op {
            Op::Mul => (0..iters).for_each(|i| blst_call!(self, blst_fr_mul, blst_fp_mul, a, a, self.xs[i as usize & m].as_ptr())),
            Op::Sqr => (0..iters).for_each(|_| blst_call!(self, blst_fr_sqr, blst_fp_sqr, a, a)),
            Op::Add => (0..iters).for_each(|i| blst_call!(self, blst_fr_add, blst_fp_add, a, a, self.xs[i as usize & m].as_ptr())),
            Op::Sub => (0..iters).for_each(|i| blst_call!(self, blst_fr_sub, blst_fp_sub, a, a, self.xs[i as usize & m].as_ptr())),
            Op::Inv if self.ct => (0..iters).for_each(|i| {
                blst_call!(self, blst_fr_inverse, blst_fp_inverse, tp, self.xs[i as usize & m].as_ptr());
                blst_call!(self, blst_fr_add, blst_fp_add, a, a, tp);
            }),
            Op::Inv => (0..iters).for_each(|i| {
                blst_call!(self, blst_fr_eucl_inverse, blst_fp_eucl_inverse, tp, self.xs[i as usize & m].as_ptr());
                blst_call!(self, blst_fr_add, blst_fp_add, a, a, tp);
            }),
            Op::Sqrt => (0..iters).for_each(|i| {
                // SAFETY: blst_fp arrays.
                unsafe {
                    blst::blst_fp_sqrt(tp as _, self.sq[i as usize & m].as_ptr() as _);
                    blst::blst_fp_add(a as _, a as _, tp as _);
                }
            }),
            Op::IsSq => (0..iters).for_each(|i| {
                // SAFETY: a blst_fp array.
                s += unsafe { blst::blst_fp_is_square(self.xs[i as usize & m].as_ptr() as _) } as i64
            }),
            Op::FromBytes => (0..iters).for_each(|i| {
                let b = &self.bytes[i as usize & m];
                // SAFETY: as in `out`.
                unsafe {
                    if self.fr {
                        let mut sc = blst::blst_scalar::default();
                        blst::blst_scalar_from_bendian(&mut sc, b.as_ptr());
                        blst::blst_fr_from_scalar(tp as _, &sc);
                        blst::blst_fr_add(a as _, a as _, tp as _);
                    } else {
                        blst::blst_fp_from_bendian(tp as _, b.as_ptr());
                        blst::blst_fp_add(a as _, a as _, tp as _);
                    }
                }
            }),
            Op::ToBytes => (0..iters).for_each(|i| {
                let mut b = [0u8; 48];
                let x = &self.xs[i as usize & m];
                // SAFETY: as in `out`.
                unsafe {
                    if self.fr {
                        let mut sc = blst::blst_scalar::default();
                        blst::blst_scalar_from_fr(&mut sc, x.as_ptr() as _);
                        blst::blst_bendian_from_scalar(b.as_mut_ptr(), &sc);
                    } else {
                        blst::blst_bendian_from_fp(b.as_mut_ptr(), x.as_ptr() as _);
                    }
                }
                s += keep(b).len() as i64;
            }),
            _ => {}
        }
        self.acc = keep(acc);
        self.sink = self.sink.wrapping_add(keep(s) as u64);
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::Inv if self.ct => "blst_*_inverse, constant-time".into(),
            Op::Inv => "blst_*_eucl_inverse, variable-time".into(),
            Op::Sqrt => "blst_fp_sqrt, constant-time".into(),
            Op::IsSq => "blst_fp_is_square, constant-time".into(),
            _ => String::new(),
        }
    }
}

/// The bls12_381 crate's scalar field.
struct Zk {
    xs: Vec<bls12_381::Scalar>,
    sq: Vec<bls12_381::Scalar>,
    bytes: Vec<[u8; 32]>,
    e: [u64; 4],
    e64: u64,
    acc: bls12_381::Scalar,
    sink: u64,
}

impl Zk {
    fn new(xs: &[Vec<u64>], bytes: &[Vec<u8>], e: &[u64], e64: u64) -> Self {
        let xs: Vec<bls12_381::Scalar> =
            xs.iter().map(|x| bls12_381::Scalar::from_bytes(&limbs_le(x, 32).try_into().unwrap()).unwrap()).collect();
        let sq = xs.iter().map(|x| x.square()).collect();
        let bytes = bytes.iter().map(|b| rev(b).try_into().unwrap()).collect();
        Zk { acc: xs[0], xs, sq, bytes, e: e.try_into().unwrap(), e64, sink: 0 }
    }
    fn l(x: &bls12_381::Scalar) -> Vec<u64> {
        le_limbs(&x.to_bytes(), 4)
    }
}

impl Bench for Zk {
    fn name(&self) -> String {
        "bls12_381 0.8".into()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        use ff::Field as _;
        let (x, y) = (self.xs[i], self.xs[(i + 1) % COUNT]);
        Some(match op {
            Op::Mul => (Zk::l(&(x * y)), 0),
            Op::Sqr => (Zk::l(&x.square()), 0),
            Op::Add => (Zk::l(&(x + y)), 0),
            Op::Sub => (Zk::l(&(x - y)), 0),
            Op::Neg => (Zk::l(&(-x)), 0),
            Op::Dbl => (Zk::l(&x.double()), 0),
            Op::Inv => (Zk::l(&x.invert().unwrap()), 0),
            Op::Sqrt => (Zk::l(&self.sq[i].sqrt().unwrap()), 1),
            Op::Pow => (Zk::l(&x.pow_vartime(&self.e)), 0),
            Op::PowU => (Zk::l(&x.pow_vartime(&[self.e64, 0, 0, 0])), 0),
            Op::FromBytes => (Zk::l(&bls12_381::Scalar::from_bytes(&self.bytes[i]).unwrap()), 0),
            Op::ToBytes => (le_limbs(&x.to_bytes(), 4), 0),
            _ => return None,
        })
    }
    fn run(&mut self, op: Op, iters: u64) {
        use ff::Field as _;
        let m = COUNT - 1;
        let mut acc = self.acc;
        let mut s = 0i64;
        match op {
            Op::Mul => (0..iters).for_each(|i| acc *= &self.xs[i as usize & m]),
            Op::Sqr => (0..iters).for_each(|_| acc = acc.square()),
            Op::Add => (0..iters).for_each(|i| acc += &self.xs[i as usize & m]),
            Op::Sub => (0..iters).for_each(|i| acc -= &self.xs[i as usize & m]),
            Op::Neg => (0..iters).for_each(|_| acc = -acc),
            Op::Dbl => (0..iters).for_each(|_| acc = acc.double()),
            Op::Inv => (0..iters).for_each(|i| acc += self.xs[i as usize & m].invert().unwrap()),
            Op::Sqrt => (0..iters).for_each(|i| acc += self.sq[i as usize & m].sqrt().unwrap()),
            Op::Pow => (0..iters).for_each(|i| acc += self.xs[i as usize & m].pow_vartime(&self.e)),
            Op::PowU => (0..iters).for_each(|i| acc += self.xs[i as usize & m].pow_vartime(&[self.e64, 0, 0, 0])),
            Op::FromBytes => {
                (0..iters).for_each(|i| acc += bls12_381::Scalar::from_bytes(&self.bytes[i as usize & m]).unwrap())
            }
            Op::ToBytes => (0..iters).for_each(|i| s += keep(self.xs[i as usize & m].to_bytes()).len() as i64),
            _ => {}
        }
        self.acc = keep(acc);
        self.sink = self.sink.wrapping_add(keep(s) as u64);
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::Inv => "invert: a constant-time power".into(),
            Op::Sqrt => "constant-time Tonelli-Shanks".into(),
            Op::FromBytes | Op::ToBytes => "little-endian bytes".into(),
            _ => String::new(),
        }
    }
}

/// The field of a case: ours (named where it has a name) and the arkworks side, if any.
fn ours_field(name: &str, p: &Number) -> &'static PrimeField {
    let f = match name {
        "c25519" => PrimeField::curve25519_base(),
        "secp256k1" => PrimeField::secp256k1_base(),
        "p256" => PrimeField::p256_base(),
        "p384" => PrimeField::p384_base(),
        "p521" => PrimeField::p521_base(),
        "bn254" => PrimeField::bn254_base(),
        "bn254r" => PrimeField::bn254_scalar(),
        "bls381" => PrimeField::bls12_381_base(),
        "bls381r" => PrimeField::bls12_381_scalar(),
        _ => Box::leak(Box::new(PrimeField::new(p))),
    };
    assert_eq!(f.modulus(), p, "the named field of {name}");
    f
}

fn ark_side(name: &str, xs: &[Vec<u64>], bytes: &[Vec<u8>], e: &[u64], e64: u64, n: usize) -> Option<Box<dyn Bench>> {
    macro_rules! a {
        ($t:ty) => {
            Some(Box::new(Ark::<$t>::new(xs, bytes, e, e64, n)))
        };
    }
    match name {
        "p64" => a!(arkf::P64),
        "p128" => a!(arkf::P128),
        "p192" => a!(arkf::P192),
        "c25519" => a!(arkf::C25519),
        "secp256k1" => a!(arkf::SECP256K1),
        "p256" => a!(arkf::P256),
        "bn254" => a!(ark_bn254::Fq),
        "bn254r" => a!(ark_bn254::Fr),
        "bls381" => a!(ark_bls12_381::Fq),
        "bls381r" => a!(ark_bls12_381::Fr),
        "p384" => a!(arkf::P384),
        "p512" => a!(arkf::P512),
        "p521" => a!(arkf::P521),
        "p576" => a!(arkf::P576),
        "p768" => a!(arkf::P768),
        "p1024" => a!(arkf::P1024),
        "p2048" => a!(arkf::P2048),
        "p4096" => a!(arkf::P4096),
        _ => None,
    }
}

/// The order of the cases, the curves' fields first.
const ORDER: &[&str] = &[
    "p64", "p128", "p192", "bn254", "bn254r", "bls381r", "c25519", "secp256k1", "p256", "bls381", "p384", "p512",
    "p521", "p576", "p600", "p768", "p1024", "p2048", "p4096",
];

fn label(name: &str, bits: u64, m8: u64, f: &PrimeField) -> String {
    let what = match name {
        "p64" => "Goldilocks 2^64-2^32+1".to_string(),
        "c25519" => "2^255-19".into(),
        "secp256k1" => "secp256k1 base".into(),
        "p256" => "P-256 base".into(),
        "p384" => "P-384 base".into(),
        "p521" => "P-521 base 2^521-1".into(),
        "bn254" => "BN254 base".into(),
        "bn254r" => "BN254 scalar".into(),
        "bls381" => "BLS12-381 base".into(),
        "bls381r" => "BLS12-381 scalar".into(),
        _ => format!("random {bits}-bit prime"),
    };
    format!("{what}; {bits} bits; p mod 8 = {m8}; form {}; sqrt {}", f.form(), f.sqrt_method())
}

/// Runs the prime-field rows, of the cases whose names are in `only` (all if empty).
pub fn run(only: &[String]) {
    let group = "prime";
    for &name in ORDER {
        if !only.is_empty() && !only.iter().any(|o| o == name) {
            continue;
        }
        let &(_, phex, _g, m8, bits) = PRIMES.iter().find(|r| r.0 == name).unwrap();
        let p = hex(phex);
        let n = bits.div_ceil(64) as usize;
        let pl = limbs(&p, n);
        let f = ours_field(name, &p);
        let blen = f.byte_len();
        let mut rng = Rng(0x5eed ^ bits);
        let xs: Vec<Vec<u64>> = (0..COUNT).map(|_| below(&mut rng, &pl)).collect();
        let e = below(&mut rng, &pl);
        let e64 = rng.next();
        let bytes: Vec<Vec<u8>> = xs.iter().map(|x| rev(&limbs_le(x, blen))).collect();
        let fx: Vec<Fp> = xs.iter().map(|x| f.from_bytes_le(&limbs_le(x, blen)).unwrap()).collect();
        let ours = Ours {
            f,
            sq: fx.iter().map(|x| x.square()).collect(),
            acc: fx[0].clone(),
            xs: fx,
            bytes: bytes.clone(),
            e: number(&e),
            e64,
            n,
            sink: 0,
        };
        // The runtime form first, then the typed form of a named prime, each against the
        // references (made again for each pass).
        let references = || {
            let mut sides: Vec<Box<dyn Bench>> = vec![];
            if let Some(s) = ark_side(name, &xs, &bytes, &e, e64, n) {
                sides.push(s);
            }
            if name == "bls381" || name == "bls381r" {
                let fr = name == "bls381r";
                sides.push(Box::new(Blst::new(fr, false, &xs, &bytes)));
                sides.push(Box::new(Blst::new(fr, true, &xs, &bytes)));
            }
            if name == "bls381r" {
                sides.push(Box::new(Zk::new(&xs, &bytes, &e, e64)));
            }
            #[cfg(has_mcl)]
            if name == "bls381" {
                sides.push(Box::new(crate::mcl::MclFp::new(&xs, &e, e64)));
            }
            let flat: Vec<u64> = xs.iter().flatten().copied().collect();
            for r in ffi::pf_refs() {
                if let Some(c) = ffi::pf_new(&r, &pl, &flat, COUNT, &e) {
                    sides.push(Box::new(c));
                }
            }
            sides
        };
        let mut passes: Vec<(Box<dyn Bench>, String)> = vec![(Box::new(ours), label(name, bits, m8, f))];
        if let Some(t) = typed_side(name, &xs, &bytes, &e, e64, n) {
            let l = label(name, bits, m8, f);
            let (what, rest) = l.split_once(';').unwrap();
            let rest = rest
                .split(';')
                .filter(|p| !p.trim_start().starts_with("form") && !p.trim_start().starts_with("sqrt"))
                .collect::<Vec<_>>()
                .join(";");
            let form = if matches!(name, "secp256k1" | "c25519") { "2^k - c (aarch64)" } else { "Montgomery" };
            let sqrt = match m8 {
                3 | 7 => "p = 3 mod 4",
                5 => "Atkin",
                _ => "Tonelli-Shanks",
            };
            passes.push((t, format!("{what} (TypedFp);{rest}; form TypedFp, {form}; sqrt {sqrt}")));
        }
        for (first, params) in passes {
        let mut sides = vec![first];
        sides.extend(references());
        let ops: &[(Op, Cmp)] = &[
            (Op::Mul, Cmp::Exact),
            (Op::Sqr, Cmp::Exact),
            (Op::Add, Cmp::Exact),
            (Op::Sub, Cmp::Exact),
            (Op::Neg, Cmp::Exact),
            (Op::Dbl, Cmp::Exact),
            (Op::Inv, Cmp::Exact),
            (Op::Sqrt, Cmp::UpToSign),
            (Op::Leg, Cmp::Flag),
            (Op::IsSq, Cmp::Flag),
            (Op::Pow, Cmp::Exact),
            (Op::PowU, Cmp::Exact),
            (Op::Batch, Cmp::Exact),
            (Op::FromBytes, Cmp::Exact),
            (Op::ToBytes, Cmp::Exact),
        ];
        for &(op, cmp) in ops {
            let mut opname = op.name().to_string();
            let mut pr = params.clone();
            match op {
                Op::Pow => pr.push_str(&format!("; exponent of {} bits", bits)),
                Op::PowU => pr.push_str("; 64-bit exponent"),
                Op::Sqrt => opname = "sqrt (of a square)".into(),
                Op::FromBytes => opname = "from_bytes_be".into(),
                Op::ToBytes => opname = "to_bytes_be".into(),
                _ => {}
            }
            common::compare(group, &opname, &pr, &mut sides, op, CHECKS, cmp, &pl);
        }
        }
    }
}

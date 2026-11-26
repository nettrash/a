//! The towers F_p² → F_p⁶ → F_p¹² of BN254 and BLS12-381: `Tower`/`Fp2`/`Fp6`/`Fp12` against
//! arkworks (ark-bn254, ark-bls12-381, the same towers) and blst (BLS12-381's `fp2` and `fp12`).

use crate::common::{self, Bench, Cmp, below, le_limbs, limbs, limbs_le};
use crate::harness::{Op, Rng, keep};
use a::math::field::{Fp, Fp2, Fp6, Fp12, Tower};
use a::math::numbers::Number;
use ark_ff::fields::{CyclotomicMultSubgroup, Field, Fp2 as AFp2, Fp6 as AFp6, Fp6Config, Fp12 as AFp12, Fp12Config};
use ark_ff::{AdditiveGroup as _, BigInteger, PrimeField as _, Zero as _};

const COUNT: usize = 256;

/// The inputs, as limbs of the coefficients (arkworks' order), shared by the sides.
struct Inputs {
    n: usize,
    x2: Vec<Vec<u64>>,
    x6: Vec<Vec<u64>>,
    x12: Vec<Vec<u64>>,
    cyc: Vec<Vec<u64>>,
    /// Three random elements of F_p² a case, for the sparse products.
    lines: Vec<Vec<u64>>,
    e: u64,
}

// ---------------------------------------------------------------- ours

struct Ours {
    t: &'static Tower<'static>,
    floor: usize,
    bn: bool,
    x2: Vec<Fp2<'static>>,
    s2: Vec<Fp2<'static>>,
    x6: Vec<Fp6<'static>>,
    x12: Vec<Fp12<'static>>,
    cyc: Vec<Fp12<'static>>,
    lines: Vec<[Fp2<'static>; 3]>,
    e: Number,
    a2: Fp2<'static>,
    a6: Fp6<'static>,
    a12: Fp12<'static>,
    n: usize,
    sink: u64,
}

fn fp(t: &'static Tower<'static>, l: &[u64]) -> Fp<'static> {
    t.base().from_bytes_le(&limbs_le(l, t.base().byte_len())).unwrap()
}

fn o2(t: &'static Tower<'static>, l: &[u64], n: usize) -> Fp2<'static> {
    Fp2::new(t, fp(t, &l[..n]), fp(t, &l[n..2 * n]))
}

fn o6(t: &'static Tower<'static>, l: &[u64], n: usize) -> Fp6<'static> {
    Fp6::new(o2(t, &l[..2 * n], n), o2(t, &l[2 * n..4 * n], n), o2(t, &l[4 * n..], n))
}

fn o12(t: &'static Tower<'static>, l: &[u64], n: usize) -> Fp12<'static> {
    Fp12::new(o6(t, &l[..6 * n], n), o6(t, &l[6 * n..], n))
}

impl Ours {
    fn new(t: &'static Tower<'static>, bn: bool, floor: usize, inp: &Inputs) -> Ours {
        let n = inp.n;
        let x2: Vec<_> = inp.x2.iter().map(|l| o2(t, l, n)).collect();
        let x6: Vec<_> = inp.x6.iter().map(|l| o6(t, l, n)).collect();
        let x12: Vec<_> = inp.x12.iter().map(|l| o12(t, l, n)).collect();
        let cyc: Vec<_> = inp.cyc.iter().map(|l| o12(t, l, n)).collect();
        let lines = inp
            .lines
            .iter()
            .map(|l| [o2(t, &l[..2 * n], n), o2(t, &l[2 * n..4 * n], n), o2(t, &l[4 * n..], n)])
            .collect();
        Ours {
            t,
            floor,
            bn,
            s2: x2.iter().map(|x| x.square()).collect(),
            a2: x2[0].clone(),
            a6: x6[0].clone(),
            a12: x12[0].clone(),
            x2,
            x6,
            x12,
            cyc,
            lines,
            e: Number::from(inp.e),
            n,
            sink: 0,
        }
    }
    fn l1(&self, x: &Fp) -> Vec<u64> {
        le_limbs(&x.to_bytes_le(), self.n)
    }
    fn l2(&self, x: &Fp2) -> Vec<u64> {
        [self.l1(x.c0()), self.l1(x.c1())].concat()
    }
    fn l6(&self, x: &Fp6) -> Vec<u64> {
        [self.l2(x.c0()), self.l2(x.c1()), self.l2(x.c2())].concat()
    }
    fn l12(&self, x: &Fp12) -> Vec<u64> {
        x.coefficients().iter().flat_map(|c| self.l1(c)).collect()
    }
    fn pad(&self, mut v: Vec<u64>) -> Vec<u64> {
        v.resize(self.floor * self.n, 0);
        v
    }
}

impl Bench for Ours {
    fn name(&self) -> String {
        "ours".into()
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::Sqr | Op::Conj | Op::CycSqr => "ours in place (square_in_place and the like)".into(),
            _ => String::new(),
        }
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let j = (i + 1) % COUNT;
        let r = match self.floor {
            2 => {
                let (x, y) = (&self.x2[i], &self.x2[j]);
                match op {
                    Op::Add => self.l2(&(x + y)),
                    Op::Mul => self.l2(&(x * y)),
                    Op::Sqr => self.l2(&x.square()),
                    Op::Inv => self.l2(&x.inverse().unwrap()),
                    Op::Sqrt => self.l2(&self.s2[i].sqrt().unwrap()),
                    Op::IsSq => return Some((vec![0; 2 * self.n], x.is_square() as i32)),
                    Op::Norm => self.pad(self.l1(&x.norm())),
                    Op::MulByXi => self.l2(&x.mul_by_xi()),
                    Op::Conj => self.l2(&x.conjugate()),
                    _ => return None,
                }
            }
            6 => {
                let (x, y) = (&self.x6[i], &self.x6[j]);
                let l = &self.lines[i];
                match op {
                    Op::Add => self.l6(&(x + y)),
                    Op::Mul => self.l6(&(x * y)),
                    Op::Sqr => self.l6(&x.square()),
                    Op::Inv => self.l6(&x.inverse().unwrap()),
                    Op::IsSq => return Some((vec![0; 6 * self.n], x.is_square() as i32)),
                    Op::Norm => self.pad(self.l2(&x.norm())),
                    Op::MulBy01 => self.l6(&x.mul_by_01(&l[0], &l[1])),
                    Op::MulBy1 => self.l6(&x.mul_by_1(&l[1])),
                    Op::MulByV => self.l6(&x.mul_by_v()),
                    Op::Frob => self.l6(&x.frobenius(1)),
                    _ => return None,
                }
            }
            _ => {
                let (x, y) = (&self.x12[i], &self.x12[j]);
                let l = &self.lines[i];
                match op {
                    Op::Add => self.l12(&(x + y)),
                    Op::Mul => self.l12(&(x * y)),
                    Op::Sqr => self.l12(&x.square()),
                    Op::Inv => self.l12(&x.inverse().unwrap()),
                    Op::IsSq => return Some((vec![0; 12 * self.n], x.is_square() as i32)),
                    Op::Norm => self.pad(self.l6(&x.norm())),
                    Op::Frob => self.l12(&x.frobenius(1)),
                    Op::Frob2 => self.l12(&x.frobenius(2)),
                    Op::Conj => self.l12(&x.conjugate()),
                    Op::CycSqr => self.l12(&self.cyc[i].cyclotomic_square()),
                    Op::CycPow => self.l12(&self.cyc[i].cyclotomic_pow(&self.e)),
                    Op::Sparse if self.bn => self.l12(&x.mul_by_034(&l[0], &l[1], &l[2])),
                    Op::Sparse => self.l12(&x.mul_by_014(&l[0], &l[1], &l[2])),
                    Op::Pow => self.l12(&x.pow(&self.e)),
                    _ => return None,
                }
            }
        };
        Some((r, 0))
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut s = 0i64;
        match self.floor {
            2 => {
                let mut a = self.a2.clone();
                let xs = &self.x2;
                match op {
                    Op::Add => (0..iters).for_each(|i| a += &xs[i as usize & m]),
                    Op::Mul => (0..iters).for_each(|i| a *= &xs[i as usize & m]),
                    Op::Sqr => (0..iters).for_each(|_| a.square_in_place()),
                    Op::Inv => (0..iters).for_each(|i| a += xs[i as usize & m].inverse().unwrap()),
                    Op::Sqrt => (0..iters).for_each(|i| a += self.s2[i as usize & m].sqrt().unwrap()),
                    Op::IsSq => (0..iters).for_each(|i| s += xs[i as usize & m].is_square() as i64),
                    Op::Norm => {
                        let mut acc = self.t.base().zero();
                        (0..iters).for_each(|i| acc += xs[i as usize & m].norm());
                        s += keep(acc).is_zero() as i64;
                    }
                    Op::MulByXi => (0..iters).for_each(|_| a = a.mul_by_xi()),
                    Op::Conj => (0..iters).for_each(|_| a.conjugate_in_place()),
                    _ => {}
                }
                if a.is_zero() {
                    a = xs[0].clone();
                }
                self.a2 = keep(a);
            }
            6 => {
                let mut a = self.a6.clone();
                let xs = &self.x6;
                let ls = &self.lines;
                match op {
                    Op::Add => (0..iters).for_each(|i| a += &xs[i as usize & m]),
                    Op::Mul => (0..iters).for_each(|i| a *= &xs[i as usize & m]),
                    Op::Sqr => (0..iters).for_each(|_| a.square_in_place()),
                    Op::Inv => (0..iters).for_each(|i| a += xs[i as usize & m].inverse().unwrap()),
                    Op::IsSq => (0..iters).for_each(|i| s += xs[i as usize & m].is_square() as i64),
                    Op::Norm => {
                        let mut acc = Fp2::zero(self.t);
                        (0..iters).for_each(|i| acc += xs[i as usize & m].norm());
                        s += keep(acc).is_zero() as i64;
                    }
                    Op::MulBy01 => (0..iters).for_each(|i| {
                        let l = &ls[i as usize & m];
                        a = a.mul_by_01(&l[0], &l[1])
                    }),
                    Op::MulBy1 => (0..iters).for_each(|i| a = a.mul_by_1(&ls[i as usize & m][1])),
                    Op::MulByV => (0..iters).for_each(|_| a = a.mul_by_v()),
                    Op::Frob => (0..iters).for_each(|_| a = a.frobenius(1)),
                    _ => {}
                }
                if a.is_zero() {
                    a = xs[0].clone();
                }
                self.a6 = keep(a);
            }
            _ => {
                let mut a = if matches!(op, Op::CycSqr | Op::CycPow) { self.cyc[0].clone() } else { self.a12.clone() };
                let xs = &self.x12;
                let ls = &self.lines;
                match op {
                    Op::Add => (0..iters).for_each(|i| a += &xs[i as usize & m]),
                    Op::Mul => (0..iters).for_each(|i| a *= &xs[i as usize & m]),
                    Op::Sqr => (0..iters).for_each(|_| a.square_in_place()),
                    Op::Inv => (0..iters).for_each(|i| a += xs[i as usize & m].inverse().unwrap()),
                    Op::IsSq => (0..iters).for_each(|i| s += xs[i as usize & m].is_square() as i64),
                    Op::Norm => {
                        let mut acc = Fp6::zero(self.t);
                        (0..iters).for_each(|i| acc += xs[i as usize & m].norm());
                        s += keep(acc).is_zero() as i64;
                    }
                    Op::Frob => (0..iters).for_each(|_| a = a.frobenius(1)),
                    Op::Frob2 => (0..iters).for_each(|_| a = a.frobenius(2)),
                    Op::Conj => (0..iters).for_each(|_| a.conjugate_in_place()),
                    Op::CycSqr => (0..iters).for_each(|_| a.cyclotomic_square_in_place()),
                    Op::CycPow => (0..iters).for_each(|_| a = a.cyclotomic_pow(&self.e)),
                    Op::Sparse if self.bn => (0..iters).for_each(|i| {
                        let l = &ls[i as usize & m];
                        a = a.mul_by_034(&l[0], &l[1], &l[2])
                    }),
                    Op::Sparse => (0..iters).for_each(|i| {
                        let l = &ls[i as usize & m];
                        a = a.mul_by_014(&l[0], &l[1], &l[2])
                    }),
                    Op::Pow => (0..iters).for_each(|_| a = a.pow(&self.e)),
                    _ => {}
                }
                if a.is_zero() {
                    a = xs[0].clone();
                }
                if !matches!(op, Op::CycSqr | Op::CycPow) {
                    self.a12 = keep(a);
                } else {
                    keep(a);
                }
            }
        }
        self.sink = self.sink.wrapping_add(keep(s) as u64);
    }
}

// ---------------------------------------------------------------- arkworks

type B<C> = <<<C as Fp12Config>::Fp6Config as Fp6Config>::Fp2Config as ark_ff::Fp2Config>::Fp;
type F2<C> = AFp2<<<C as Fp12Config>::Fp6Config as Fp6Config>::Fp2Config>;
type F6<C> = AFp6<<C as Fp12Config>::Fp6Config>;

struct Ark<C: Fp12Config> {
    floor: usize,
    bn: bool,
    x2: Vec<F2<C>>,
    s2: Vec<F2<C>>,
    x6: Vec<F6<C>>,
    x12: Vec<AFp12<C>>,
    cyc: Vec<AFp12<C>>,
    lines: Vec<[F2<C>; 3]>,
    e: u64,
    a2: F2<C>,
    a6: F6<C>,
    a12: AFp12<C>,
    n: usize,
    sink: u64,
}

impl<C: Fp12Config> Ark<C> {
    fn b(l: &[u64], n: usize) -> B<C> {
        B::<C>::from_le_bytes_mod_order(&limbs_le(l, 8 * n))
    }
    fn f2(l: &[u64], n: usize) -> F2<C> {
        F2::<C>::new(Self::b(&l[..n], n), Self::b(&l[n..2 * n], n))
    }
    fn f6(l: &[u64], n: usize) -> F6<C> {
        F6::<C>::new(Self::f2(&l[..2 * n], n), Self::f2(&l[2 * n..4 * n], n), Self::f2(&l[4 * n..], n))
    }
    fn f12(l: &[u64], n: usize) -> AFp12<C> {
        AFp12::<C>::new(Self::f6(&l[..6 * n], n), Self::f6(&l[6 * n..], n))
    }
    fn new(bn: bool, floor: usize, inp: &Inputs) -> Self {
        let n = inp.n;
        let x2: Vec<_> = inp.x2.iter().map(|l| Self::f2(l, n)).collect();
        let x6: Vec<_> = inp.x6.iter().map(|l| Self::f6(l, n)).collect();
        let x12: Vec<_> = inp.x12.iter().map(|l| Self::f12(l, n)).collect();
        Ark {
            floor,
            bn,
            s2: x2.iter().map(|x| x.square()).collect(),
            a2: x2[0],
            a6: x6[0],
            a12: x12[0],
            x2,
            x6,
            x12,
            cyc: inp.cyc.iter().map(|l| Self::f12(l, n)).collect(),
            lines: inp
                .lines
                .iter()
                .map(|l| [Self::f2(&l[..2 * n], n), Self::f2(&l[2 * n..4 * n], n), Self::f2(&l[4 * n..], n)])
                .collect(),
            e: inp.e,
            n,
            sink: 0,
        }
    }
    fn l1(&self, x: &B<C>) -> Vec<u64> {
        let b = x.into_bigint().to_bytes_le();
        le_limbs(&b, self.n)
    }
    fn l2(&self, x: &F2<C>) -> Vec<u64> {
        [self.l1(&x.c0), self.l1(&x.c1)].concat()
    }
    fn l6(&self, x: &F6<C>) -> Vec<u64> {
        [self.l2(&x.c0), self.l2(&x.c1), self.l2(&x.c2)].concat()
    }
    fn l12(&self, x: &AFp12<C>) -> Vec<u64> {
        [self.l6(&x.c0), self.l6(&x.c1)].concat()
    }
    fn pad(&self, mut v: Vec<u64>) -> Vec<u64> {
        v.resize(self.floor * self.n, 0);
        v
    }
}

impl<C: Fp12Config> Bench for Ark<C> {
    fn name(&self) -> String {
        "arkworks 0.5".into()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let j = (i + 1) % COUNT;
        let r = match self.floor {
            2 => {
                let (x, y) = (self.x2[i], self.x2[j]);
                match op {
                    Op::Add => self.l2(&(x + y)),
                    Op::Mul => self.l2(&(x * y)),
                    Op::Sqr => self.l2(&x.square()),
                    Op::Inv => self.l2(&x.inverse().unwrap()),
                    Op::Sqrt => self.l2(&self.s2[i].sqrt().unwrap()),
                    Op::IsSq => return Some((vec![0; 2 * self.n], !x.legendre().is_qnr() as i32)),
                    Op::Norm => self.pad(self.l1(&x.norm())),
                    Op::MulByXi => self.l2(&C::Fp6Config::mul_fp2_by_nonresidue(x)),
                    Op::Conj => {
                        let mut c = x;
                        c.conjugate_in_place();
                        self.l2(&c)
                    }
                    _ => return None,
                }
            }
            6 => {
                let (x, y) = (self.x6[i], self.x6[j]);
                let l = self.lines[i];
                match op {
                    Op::Add => self.l6(&(x + y)),
                    Op::Mul => self.l6(&(x * y)),
                    Op::Sqr => self.l6(&x.square()),
                    Op::Inv => self.l6(&x.inverse().unwrap()),
                    Op::IsSq => return Some((vec![0; 6 * self.n], !x.legendre().is_qnr() as i32)),
                    Op::Norm => self.pad(self.l2(&x.norm())),
                    Op::MulBy01 => {
                        let mut c = x;
                        c.mul_by_01(&l[0], &l[1]);
                        self.l6(&c)
                    }
                    Op::MulBy1 => {
                        let mut c = x;
                        c.mul_by_1(&l[1]);
                        self.l6(&c)
                    }
                    Op::MulByV => {
                        let mut c = x;
                        C::mul_fp6_by_nonresidue_in_place(&mut c);
                        self.l6(&c)
                    }
                    Op::Frob => {
                        let mut c = x;
                        c.frobenius_map_in_place(1);
                        self.l6(&c)
                    }
                    _ => return None,
                }
            }
            _ => {
                let (x, y) = (self.x12[i], self.x12[j]);
                let l = self.lines[i];
                let mut c = x;
                match op {
                    Op::Add => self.l12(&(x + y)),
                    Op::Mul => self.l12(&(x * y)),
                    Op::Sqr => self.l12(&x.square()),
                    Op::Inv => self.l12(&x.inverse().unwrap()),
                    Op::IsSq => return Some((vec![0; 12 * self.n], !x.legendre().is_qnr() as i32)),
                    Op::Norm => self.pad(self.l6(&x.norm())),
                    Op::Frob => {
                        c.frobenius_map_in_place(1);
                        self.l12(&c)
                    }
                    Op::Frob2 => {
                        c.frobenius_map_in_place(2);
                        self.l12(&c)
                    }
                    Op::Conj => {
                        c.conjugate_in_place();
                        self.l12(&c)
                    }
                    Op::CycSqr => {
                        let mut z = self.cyc[i];
                        z.cyclotomic_square_in_place();
                        self.l12(&z)
                    }
                    Op::CycPow => self.l12(&self.cyc[i].cyclotomic_exp([self.e])),
                    Op::Sparse if self.bn => {
                        c.mul_by_034(&l[0], &l[1], &l[2]);
                        self.l12(&c)
                    }
                    Op::Sparse => {
                        c.mul_by_014(&l[0], &l[1], &l[2]);
                        self.l12(&c)
                    }
                    Op::Pow => self.l12(&x.pow([self.e])),
                    _ => return None,
                }
            }
        };
        Some((r, 0))
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut s = 0i64;
        match self.floor {
            2 => {
                let mut a = self.a2;
                let xs = &self.x2;
                match op {
                    Op::Add => (0..iters).for_each(|i| a += &xs[i as usize & m]),
                    Op::Mul => (0..iters).for_each(|i| a *= &xs[i as usize & m]),
                    Op::Sqr => (0..iters).for_each(|_| {
                        a.square_in_place();
                    }),
                    Op::Inv => (0..iters).for_each(|i| a += xs[i as usize & m].inverse().unwrap()),
                    Op::Sqrt => (0..iters).for_each(|i| a += self.s2[i as usize & m].sqrt().unwrap()),
                    Op::IsSq => (0..iters).for_each(|i| s += !xs[i as usize & m].legendre().is_qnr() as i64),
                    Op::Norm => {
                        let mut acc = B::<C>::ZERO;
                        (0..iters).for_each(|i| acc += xs[i as usize & m].norm());
                        s += keep(acc).is_zero() as i64;
                    }
                    Op::MulByXi => (0..iters).for_each(|_| {
                        C::Fp6Config::mul_fp2_by_nonresidue_in_place(&mut a);
                    }),
                    Op::Conj => (0..iters).for_each(|_| {
                        a.conjugate_in_place();
                    }),
                    _ => {}
                }
                self.a2 = keep(a);
            }
            6 => {
                let mut a = self.a6;
                let xs = &self.x6;
                let ls = &self.lines;
                match op {
                    Op::Add => (0..iters).for_each(|i| a += &xs[i as usize & m]),
                    Op::Mul => (0..iters).for_each(|i| a *= &xs[i as usize & m]),
                    Op::Sqr => (0..iters).for_each(|_| {
                        a.square_in_place();
                    }),
                    Op::Inv => (0..iters).for_each(|i| a += xs[i as usize & m].inverse().unwrap()),
                    Op::IsSq => (0..iters).for_each(|i| s += !xs[i as usize & m].legendre().is_qnr() as i64),
                    Op::Norm => {
                        let mut acc = F2::<C>::ZERO;
                        (0..iters).for_each(|i| acc += xs[i as usize & m].norm());
                        s += keep(acc).is_zero() as i64;
                    }
                    Op::MulBy01 => (0..iters).for_each(|i| {
                        let l = &ls[i as usize & m];
                        a.mul_by_01(&l[0], &l[1]);
                    }),
                    Op::MulBy1 => (0..iters).for_each(|i| a.mul_by_1(&ls[i as usize & m][1])),
                    Op::MulByV => (0..iters).for_each(|_| {
                        C::mul_fp6_by_nonresidue_in_place(&mut a);
                    }),
                    Op::Frob => (0..iters).for_each(|_| a.frobenius_map_in_place(1)),
                    _ => {}
                }
                self.a6 = keep(a);
            }
            _ => {
                let mut a = if matches!(op, Op::CycSqr | Op::CycPow) { self.cyc[0] } else { self.a12 };
                let xs = &self.x12;
                let ls = &self.lines;
                match op {
                    Op::Add => (0..iters).for_each(|i| a += &xs[i as usize & m]),
                    Op::Mul => (0..iters).for_each(|i| a *= &xs[i as usize & m]),
                    Op::Sqr => (0..iters).for_each(|_| {
                        a.square_in_place();
                    }),
                    Op::Inv => (0..iters).for_each(|i| a += xs[i as usize & m].inverse().unwrap()),
                    Op::IsSq => (0..iters).for_each(|i| s += !xs[i as usize & m].legendre().is_qnr() as i64),
                    Op::Norm => {
                        let mut acc = F6::<C>::ZERO;
                        (0..iters).for_each(|i| acc += xs[i as usize & m].norm());
                        s += keep(acc).is_zero() as i64;
                    }
                    Op::Frob => (0..iters).for_each(|_| a.frobenius_map_in_place(1)),
                    Op::Frob2 => (0..iters).for_each(|_| a.frobenius_map_in_place(2)),
                    Op::Conj => (0..iters).for_each(|_| {
                        a.conjugate_in_place();
                    }),
                    Op::CycSqr => (0..iters).for_each(|_| {
                        a.cyclotomic_square_in_place();
                    }),
                    Op::CycPow => (0..iters).for_each(|_| a = a.cyclotomic_exp([self.e])),
                    Op::Sparse if self.bn => (0..iters).for_each(|i| {
                        let l = &ls[i as usize & m];
                        a.mul_by_034(&l[0], &l[1], &l[2]);
                    }),
                    Op::Sparse => (0..iters).for_each(|i| {
                        let l = &ls[i as usize & m];
                        a.mul_by_014(&l[0], &l[1], &l[2]);
                    }),
                    Op::Pow => (0..iters).for_each(|_| a = a.pow([self.e])),
                    _ => {}
                }
                if !matches!(op, Op::CycSqr | Op::CycPow) {
                    self.a12 = keep(a);
                } else {
                    keep(a);
                }
            }
        }
        self.sink = self.sink.wrapping_add(keep(s) as u64);
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::IsSq => "legendre: Euler's criterion on the norm".into(),
            _ => String::new(),
        }
    }
}

// ---------------------------------------------------------------- blst (BLS12-381)

type BFp = [u64; 6];

struct Blst {
    floor: usize,
    ct: bool,
    x2: Vec<[BFp; 2]>,
    s2: Vec<[BFp; 2]>,
    x12: Vec<[BFp; 12]>,
    cyc: Vec<[BFp; 12]>,
    lines: Vec<[BFp; 6]>,
    a2: [BFp; 2],
    a12: [BFp; 12],
    sink: u64,
}

fn bfp(l: &[u64]) -> BFp {
    let mut r = [0u64; 6];
    // SAFETY: a blst_fp is 6 limbs; `l` has 6.
    unsafe { blst::blst_fp_from_uint64(r.as_mut_ptr() as *mut blst::blst_fp, l.as_ptr()) };
    r
}

fn bout(x: &[BFp]) -> Vec<u64> {
    x.iter()
        .flat_map(|e| {
            let mut r = [0u64; 6];
            // SAFETY: as above.
            unsafe { blst::blst_uint64_from_fp(r.as_mut_ptr(), e.as_ptr() as *const blst::blst_fp) };
            r
        })
        .collect()
}

fn barr<const K: usize>(l: &[u64]) -> [BFp; K] {
    core::array::from_fn(|c| bfp(&l[6 * c..6 * c + 6]))
}

macro_rules! p {
    ($x:expr) => {
        $x.as_ptr() as *const _
    };
}
macro_rules! pm {
    ($x:expr) => {
        $x.as_mut_ptr() as *mut _
    };
}

impl Blst {
    fn new(floor: usize, ct: bool, inp: &Inputs) -> Self {
        let x2: Vec<[BFp; 2]> = inp.x2.iter().map(|l| barr::<2>(l)).collect();
        let s2 = x2
            .iter()
            .map(|x| {
                let mut r = [[0u64; 6]; 2];
                // SAFETY: blst_fp2 is two blst_fp.
                unsafe { blst::blst_fp2_sqr(pm!(r), p!(x)) };
                r
            })
            .collect();
        let x12: Vec<[BFp; 12]> = inp.x12.iter().map(|l| barr::<12>(l)).collect();
        Blst {
            floor,
            ct,
            a2: x2[0],
            a12: x12[0],
            x2,
            s2,
            x12,
            cyc: inp.cyc.iter().map(|l| barr::<12>(l)).collect(),
            lines: inp.lines.iter().map(|l| barr::<6>(l)).collect(),
            sink: 0,
        }
    }
}

impl Bench for Blst {
    fn name(&self) -> String {
        if self.ct { "blst 0.3 (constant-time)".into() } else { "blst 0.3".into() }
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let j = (i + 1) % COUNT;
        match self.floor {
            2 => {
                let (x, y) = (&self.x2[i], &self.x2[j]);
                let mut r = [[0u64; 6]; 2];
                let mut flag = 0;
                // SAFETY: blst_fp2 arrays.
                unsafe {
                    match (op, self.ct) {
                        (Op::Add, false) => blst::blst_fp2_add(pm!(r), p!(x), p!(y)),
                        (Op::Mul, false) => blst::blst_fp2_mul(pm!(r), p!(x), p!(y)),
                        (Op::Sqr, false) => blst::blst_fp2_sqr(pm!(r), p!(x)),
                        (Op::Inv, false) => blst::blst_fp2_eucl_inverse(pm!(r), p!(x)),
                        (Op::Inv, true) => blst::blst_fp2_inverse(pm!(r), p!(x)),
                        (Op::Sqrt, true) => flag = blst::blst_fp2_sqrt(pm!(r), p!(self.s2[i])) as i32,
                        (Op::IsSq, true) => flag = blst::blst_fp2_is_square(p!(x)) as i32,
                        _ => return None,
                    }
                }
                Some((bout(&r), flag))
            }
            12 => {
                let x = &self.x12[i];
                let y = &self.x12[j];
                let mut r = [[0u64; 6]; 12];
                // SAFETY: blst_fp12 and blst_fp6 arrays.
                unsafe {
                    match (op, self.ct) {
                        (Op::Mul, false) => blst::blst_fp12_mul(pm!(r), p!(x), p!(y)),
                        (Op::Sqr, false) => blst::blst_fp12_sqr(pm!(r), p!(x)),
                        (Op::Inv, true) => blst::blst_fp12_inverse(pm!(r), p!(x)),
                        (Op::Frob, false) => blst::blst_fp12_frobenius_map(pm!(r), p!(x), 1),
                        (Op::Frob2, false) => blst::blst_fp12_frobenius_map(pm!(r), p!(x), 2),
                        (Op::Conj, false) => {
                            r = *x;
                            blst::blst_fp12_conjugate(pm!(r));
                        }
                        (Op::CycSqr, false) => blst::blst_fp12_cyclotomic_sqr(pm!(r), p!(self.cyc[i])),
                        (Op::Sparse, false) => blst::blst_fp12_mul_by_xy00z0(pm!(r), p!(x), p!(self.lines[i])),
                        _ => return None,
                    }
                }
                Some((bout(&r), 0))
            }
            _ => None,
        }
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut s = 0i64;
        if self.floor == 2 {
            let mut a = self.a2;
            let mut t = [[0u64; 6]; 2];
            // SAFETY: blst_fp2 arrays; blst allows the output to alias an input.
            unsafe {
                let ap: *mut blst::blst_fp2 = pm!(a);
                let tp: *mut blst::blst_fp2 = pm!(t);
                match (op, self.ct) {
                    (Op::Add, _) => (0..iters).for_each(|i| blst::blst_fp2_add(ap, ap, p!(self.x2[i as usize & m]))),
                    (Op::Mul, _) => (0..iters).for_each(|i| blst::blst_fp2_mul(ap, ap, p!(self.x2[i as usize & m]))),
                    (Op::Sqr, _) => (0..iters).for_each(|_| blst::blst_fp2_sqr(ap, ap)),
                    (Op::Inv, false) => (0..iters).for_each(|i| {
                        blst::blst_fp2_eucl_inverse(tp, p!(self.x2[i as usize & m]));
                        blst::blst_fp2_add(ap, ap, tp);
                    }),
                    (Op::Inv, true) => (0..iters).for_each(|i| {
                        blst::blst_fp2_inverse(tp, p!(self.x2[i as usize & m]));
                        blst::blst_fp2_add(ap, ap, tp);
                    }),
                    (Op::Sqrt, _) => (0..iters).for_each(|i| {
                        blst::blst_fp2_sqrt(tp, p!(self.s2[i as usize & m]));
                        blst::blst_fp2_add(ap, ap, tp);
                    }),
                    (Op::IsSq, _) => (0..iters).for_each(|i| s += blst::blst_fp2_is_square(p!(self.x2[i as usize & m])) as i64),
                    _ => {}
                }
            }
            self.a2 = keep(a);
        } else {
            let mut a = if matches!(op, Op::CycSqr) { self.cyc[0] } else { self.a12 };
            let mut t = [[0u64; 6]; 12];
            // SAFETY: blst_fp12 and blst_fp6 arrays; blst allows aliasing.
            unsafe {
                let ap: *mut blst::blst_fp12 = pm!(a);
                let tp: *mut blst::blst_fp12 = pm!(t);
                match op {
                    Op::Mul => (0..iters).for_each(|i| blst::blst_fp12_mul(ap, ap, p!(self.x12[i as usize & m]))),
                    Op::Sqr => (0..iters).for_each(|_| blst::blst_fp12_sqr(ap, ap)),
                    Op::Inv => (0..iters).for_each(|i| {
                        blst::blst_fp12_inverse(tp, p!(self.x12[i as usize & m]));
                        // an F_p^12 sum as twelve F_p sums
                        for c in 0..12 {
                            let ac = (ap as *mut BFp).add(c) as *mut blst::blst_fp;
                            blst::blst_fp_add(ac, ac, (tp as *const BFp).add(c) as *const blst::blst_fp);
                        }
                    }),
                    Op::Frob => (0..iters).for_each(|_| blst::blst_fp12_frobenius_map(ap, ap, 1)),
                    Op::Frob2 => (0..iters).for_each(|_| blst::blst_fp12_frobenius_map(ap, ap, 2)),
                    Op::Conj => (0..iters).for_each(|_| blst::blst_fp12_conjugate(ap)),
                    Op::CycSqr => (0..iters).for_each(|_| blst::blst_fp12_cyclotomic_sqr(ap, ap)),
                    Op::Sparse => (0..iters).for_each(|i| blst::blst_fp12_mul_by_xy00z0(ap, ap, p!(self.lines[i as usize & m]))),
                    _ => {}
                }
            }
            if !matches!(op, Op::CycSqr) {
                self.a12 = keep(a);
            } else {
                keep(a);
            }
        }
        self.sink = self.sink.wrapping_add(keep(s) as u64);
    }
    fn note(&self, op: Op) -> String {
        match (op, self.ct) {
            (Op::Inv, false) => "eucl_inverse, variable-time".into(),
            (Op::Inv, true) => "constant-time inverse".into(),
            (Op::Sqrt | Op::IsSq, _) => "constant-time".into(),
            (Op::Sparse, _) => "blst_fp12_mul_by_xy00z0".into(),
            _ => String::new(),
        }
    }
}

// ---------------------------------------------------------------- the rows

fn inputs(t: &'static Tower<'static>, seed: u64, e: u64) -> Inputs {
    let p = t.base().modulus();
    let n = (t.base().bits() as usize).div_ceil(64);
    let pl = limbs(p, n);
    let mut rng = Rng(seed);
    let mut el = |k: usize| -> Vec<u64> { (0..k).flat_map(|_| below(&mut rng, &pl)).collect() };
    let x2 = (0..COUNT).map(|_| el(2)).collect();
    let x6 = (0..COUNT).map(|_| el(6)).collect();
    let x12: Vec<Vec<u64>> = (0..COUNT).map(|_| el(12)).collect();
    let lines = (0..COUNT).map(|_| el(6)).collect();
    // elements of the cyclotomic subgroup: c = x̄/x, then c^(p²)·c (the easy part of a final
    // exponentiation)
    let cyc = x12
        .iter()
        .map(|l| {
            let x = o12(t, l, n);
            let c = &x.conjugate() / &x;
            let c = &c.frobenius(2) * &c;
            assert!(c.is_cyclotomic());
            c.coefficients().iter().flat_map(|f| le_limbs(&f.to_bytes_le(), n)).collect()
        })
        .collect();
    Inputs { n, x2, x6, x12, cyc, lines, e }
}

/// Runs the tower rows of `bn254` and `bls12_381` (both if `only` is empty).
pub fn run(only: &[String]) {
    let group = "tower";
    for (name, bn) in [("bn254", true), ("bls12_381", false)] {
        if !only.is_empty() && !only.iter().any(|o| o == name) {
            continue;
        }
        let t: &'static Tower<'static> = if bn { Tower::bn254() } else { Tower::bls12_381() };
        // the curve parameter: BN254's u, BLS12-381's |x|
        let e = if bn { 4965661367192848881 } else { 0xd201000000010000 };
        let inp = inputs(t, if bn { 0x7b254 } else { 0x7b381 }, e);
        let n = inp.n;
        let pl = limbs(t.base().modulus(), n);
        let floors: [(usize, &[Op]); 3] = [
            (2, &[Op::Add, Op::Mul, Op::Sqr, Op::Inv, Op::Sqrt, Op::IsSq, Op::Norm, Op::MulByXi, Op::Conj]),
            (6, &[Op::Add, Op::Mul, Op::Sqr, Op::Inv, Op::IsSq, Op::Norm, Op::MulBy01, Op::MulBy1, Op::MulByV, Op::Frob]),
            (
                12,
                &[
                    Op::Add,
                    Op::Mul,
                    Op::Sqr,
                    Op::Inv,
                    Op::IsSq,
                    Op::Norm,
                    Op::Frob,
                    Op::Frob2,
                    Op::Conj,
                    Op::CycSqr,
                    Op::CycPow,
                    Op::Sparse,
                    Op::Pow,
                ],
            ),
        ];
        for (floor, ops) in floors {
            let mut sides: Vec<Box<dyn Bench>> = vec![Box::new(Ours::new(t, bn, floor, &inp))];
            if bn {
                sides.push(Box::new(Ark::<ark_bn254::Fq12Config>::new(bn, floor, &inp)));
            } else {
                sides.push(Box::new(Ark::<ark_bls12_381::Fq12Config>::new(bn, floor, &inp)));
                if floor != 6 {
                    sides.push(Box::new(Blst::new(floor, false, &inp)));
                    sides.push(Box::new(Blst::new(floor, true, &inp)));
                }
                #[cfg(has_mcl)]
                if floor != 6 {
                    sides.push(Box::new(crate::mcl::MclTower::new(floor, &inp.x2, &inp.x12, &inp.cyc, inp.e)));
                }
            }
            for &op in ops {
                let tname = if bn { "BN254" } else { "BLS12-381" };
                let opname = match op {
                    Op::Frob => "frobenius(1)".to_string(),
                    Op::Sqrt => "sqrt (of a square)".into(),
                    Op::Conj if floor == 2 => "conjugate (frobenius(1))".into(),
                    Op::Sparse if bn => "mul_by_034 (D-type line)".into(),
                    Op::Sparse => "mul_by_014 (M-type line)".into(),
                    o => o.name().into(),
                };
                let params = match op {
                    Op::CycPow | Op::Pow => format!("{tname} F_p^{floor}; exponent {e:#x} (64 bits)"),
                    Op::CycSqr => format!("{tname} F_p^{floor}; element of the cyclotomic subgroup"),
                    _ => format!("{tname} F_p^{floor}"),
                };
                let cmp = match op {
                    Op::Sqrt => Cmp::UpToSign,
                    Op::IsSq => Cmp::Flag,
                    _ => Cmp::Exact,
                };
                common::compare(group, &opname, &params, &mut sides, op, 8, cmp, &pl);
            }
        }
    }
}

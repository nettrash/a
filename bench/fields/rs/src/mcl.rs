//! mcl (herumi/mcl, BSD-3-Clause) through its C API (include/mcl/bn.h), linked by build.rs from a
//! build of its default MCL_FP_BIT=384, MCL_FR_BIT=256 (the curves area's, read only): BLS12-381's
//! base field, F_p² and F_p¹² (mcl's GT, the same tower). mcl keeps one curve per process, so it
//! is set up for BLS12-381 only; its BN254 would need a process of its own.
#![allow(non_snake_case, dead_code)]

use crate::common::{Bench, le_limbs, limbs_le};
use crate::harness::{Op, keep};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MFp {
    d: [u64; 6],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MFp2 {
    d: [MFp; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MFr {
    d: [u64; 4],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MGt {
    d: [MFp; 12],
}
impl Default for MGt {
    fn default() -> Self {
        MGt { d: [MFp::default(); 12] }
    }
}

const BLS12_381: i32 = 5;
/// MCLBN_COMPILED_TIME_VAR for MCLBN_FR_UNIT_SIZE = 4 and MCLBN_FP_UNIT_SIZE = 6.
const COMPILED_TIME_VAR: i32 = 46;

unsafe extern "C" {
    fn mclBn_getVersion() -> i32;
    fn mclBn_init(curve: i32, ctv: i32) -> i32;
    fn mclBnFp_setLittleEndian(x: *mut MFp, buf: *const u8, n: usize) -> i32;
    fn mclBnFp_getLittleEndian(buf: *mut u8, n: usize, x: *const MFp) -> usize;
    fn mclBnFr_setLittleEndian(x: *mut MFr, buf: *const u8, n: usize) -> i32;
    fn mclBnFp_add(z: *mut MFp, x: *const MFp, y: *const MFp);
    fn mclBnFp_sub(z: *mut MFp, x: *const MFp, y: *const MFp);
    fn mclBnFp_mul(z: *mut MFp, x: *const MFp, y: *const MFp);
    fn mclBnFp_sqr(y: *mut MFp, x: *const MFp);
    fn mclBnFp_neg(y: *mut MFp, x: *const MFp);
    fn mclBnFp_inv(y: *mut MFp, x: *const MFp);
    fn mclBnFp_squareRoot(y: *mut MFp, x: *const MFp) -> i32;
    fn mclBnFp_powArray(z: *mut MFp, x: *const MFp, y: *const u8, n: usize) -> i32;
    fn mclBnFp2_add(z: *mut MFp2, x: *const MFp2, y: *const MFp2);
    fn mclBnFp2_mul(z: *mut MFp2, x: *const MFp2, y: *const MFp2);
    fn mclBnFp2_sqr(y: *mut MFp2, x: *const MFp2);
    fn mclBnFp2_inv(y: *mut MFp2, x: *const MFp2);
    fn mclBnFp2_squareRoot(y: *mut MFp2, x: *const MFp2) -> i32;
    fn mclBnGT_add(z: *mut MGt, x: *const MGt, y: *const MGt);
    fn mclBnGT_mul(z: *mut MGt, x: *const MGt, y: *const MGt);
    fn mclBnGT_sqr(y: *mut MGt, x: *const MGt);
    fn mclBnGT_inv(y: *mut MGt, x: *const MGt);
    fn mclBnGT_invGeneric(y: *mut MGt, x: *const MGt);
    fn mclBnGT_pow(z: *mut MGt, x: *const MGt, y: *const MFr);
    fn mclBnGT_powGeneric(z: *mut MGt, x: *const MGt, y: *const MFr);
}

/// Sets mcl up for BLS12-381 (once).
pub fn init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: mcl's set-up, before any other call of it, once.
    ONCE.call_once(|| unsafe { assert_eq!(mclBn_init(BLS12_381, COMPILED_TIME_VAR), 0, "mclBn_init") });
}

/// "mcl A.BC.D" from 0xABCD (include/mcl/op.hpp).
pub fn name() -> String {
    // SAFETY: returns an int.
    let v = unsafe { mclBn_getVersion() };
    format!("mcl {:x}.{:02x}.{:x}", v >> 12, (v >> 4) & 0xff, v & 0xf)
}

fn fp(l: &[u64]) -> MFp {
    let mut x = MFp::default();
    let b = limbs_le(l, 48);
    // SAFETY: reads the 48 bytes of `b`.
    assert_eq!(unsafe { mclBnFp_setLittleEndian(&mut x, b.as_ptr(), b.len()) }, 0);
    x
}

fn out(x: &[MFp]) -> Vec<u64> {
    x.iter()
        .flat_map(|e| {
            let mut b = [0u8; 48];
            // SAFETY: writes at most 48 bytes.
            let n = unsafe { mclBnFp_getLittleEndian(b.as_mut_ptr(), 48, e) };
            assert!(n > 0 && n <= 48);
            le_limbs(&b[..n], 6)
        })
        .collect()
}

fn fr_u64(e: u64) -> MFr {
    let mut x = MFr::default();
    let b = e.to_le_bytes();
    // SAFETY: reads the 8 bytes.
    assert_eq!(unsafe { mclBnFr_setLittleEndian(&mut x, b.as_ptr(), 8) }, 0);
    x
}

const COUNT: usize = 256;

// ---------------------------------------------------------------- F_p

pub struct MclFp {
    xs: Vec<MFp>,
    sq: Vec<MFp>,
    e: Vec<u8>,
    e64: [u8; 8],
    acc: MFp,
}

impl MclFp {
    pub fn new(xs: &[Vec<u64>], e: &[u64], e64: u64) -> MclFp {
        init();
        let xs: Vec<MFp> = xs.iter().map(|l| fp(l)).collect();
        let sq = xs
            .iter()
            .map(|x| {
                let mut r = MFp::default();
                // SAFETY: mcl elements.
                unsafe { mclBnFp_sqr(&mut r, x) };
                r
            })
            .collect();
        MclFp { acc: xs[0], xs, sq, e: limbs_le(e, 48), e64: e64.to_le_bytes() }
    }
}

impl Bench for MclFp {
    fn name(&self) -> String {
        name()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let (x, y) = (&self.xs[i], &self.xs[(i + 1) % COUNT]);
        let mut r = MFp::default();
        let mut flag = 0;
        // SAFETY: mcl elements; the exponents' bytes are read within their lengths.
        unsafe {
            match op {
                Op::Mul => mclBnFp_mul(&mut r, x, y),
                Op::Sqr => mclBnFp_sqr(&mut r, x),
                Op::Add => mclBnFp_add(&mut r, x, y),
                Op::Sub => mclBnFp_sub(&mut r, x, y),
                Op::Neg => mclBnFp_neg(&mut r, x),
                Op::Dbl => mclBnFp_add(&mut r, x, x),
                Op::Inv => mclBnFp_inv(&mut r, x),
                Op::Sqrt => flag = (mclBnFp_squareRoot(&mut r, &self.sq[i]) == 0) as i32,
                Op::Pow => assert_eq!(mclBnFp_powArray(&mut r, x, self.e.as_ptr(), self.e.len()), 0),
                Op::PowU => assert_eq!(mclBnFp_powArray(&mut r, x, self.e64.as_ptr(), 8), 0),
                _ => return None,
            }
        }
        Some((out(&[r]), flag))
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut a = self.acc;
        let mut t = MFp::default();
        let ap: *mut MFp = &mut a;
        let tp: *mut MFp = &mut t;
        // SAFETY: mcl elements; mcl allows the output to be an input.
        unsafe {
            match op {
                Op::Mul => (0..iters).for_each(|i| mclBnFp_mul(ap, ap, &self.xs[i as usize & m])),
                Op::Sqr => (0..iters).for_each(|_| mclBnFp_sqr(ap, ap)),
                Op::Add => (0..iters).for_each(|i| mclBnFp_add(ap, ap, &self.xs[i as usize & m])),
                Op::Sub => (0..iters).for_each(|i| mclBnFp_sub(ap, ap, &self.xs[i as usize & m])),
                Op::Neg => (0..iters).for_each(|_| mclBnFp_neg(ap, ap)),
                Op::Dbl => (0..iters).for_each(|_| mclBnFp_add(ap, ap, ap)),
                Op::Inv => (0..iters).for_each(|i| {
                    mclBnFp_inv(tp, &self.xs[i as usize & m]);
                    mclBnFp_add(ap, ap, tp);
                }),
                Op::Sqrt => (0..iters).for_each(|i| {
                    mclBnFp_squareRoot(tp, &self.sq[i as usize & m]);
                    mclBnFp_add(ap, ap, tp);
                }),
                Op::Pow => (0..iters).for_each(|i| {
                    mclBnFp_powArray(tp, &self.xs[i as usize & m], self.e.as_ptr(), self.e.len());
                    mclBnFp_add(ap, ap, tp);
                }),
                Op::PowU => (0..iters).for_each(|i| {
                    mclBnFp_powArray(tp, &self.xs[i as usize & m], self.e64.as_ptr(), 8);
                    mclBnFp_add(ap, ap, tp);
                }),
                _ => {}
            }
        }
        self.acc = keep(a);
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::Pow | Op::PowU => "mclBnFp_powArray".into(),
            _ => String::new(),
        }
    }
}

// ---------------------------------------------------------------- F_p² and F_p¹²

pub struct MclTower {
    floor: usize,
    x2: Vec<MFp2>,
    s2: Vec<MFp2>,
    x12: Vec<MGt>,
    cyc: Vec<MGt>,
    e: MFr,
    a2: MFp2,
    a12: MGt,
}

impl MclTower {
    /// `floor` 2 or 12; the elements as arkworks orders the coefficients (mcl's order too).
    pub fn new(floor: usize, x2: &[Vec<u64>], x12: &[Vec<u64>], cyc: &[Vec<u64>], e: u64) -> MclTower {
        init();
        let x2: Vec<MFp2> = x2.iter().map(|l| MFp2 { d: [fp(&l[..6]), fp(&l[6..12])] }).collect();
        let gt = |l: &Vec<u64>| MGt { d: core::array::from_fn(|c| fp(&l[6 * c..6 * c + 6])) };
        let x12: Vec<MGt> = x12.iter().map(gt).collect();
        let cyc: Vec<MGt> = cyc.iter().map(gt).collect();
        let s2 = x2
            .iter()
            .map(|x| {
                let mut r = MFp2::default();
                // SAFETY: mcl elements.
                unsafe { mclBnFp2_sqr(&mut r, x) };
                r
            })
            .collect();
        MclTower { floor, a2: x2[0], a12: x12[0], x2, s2, x12, cyc, e: fr_u64(e) }
    }
}

impl Bench for MclTower {
    fn name(&self) -> String {
        name()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let j = (i + 1) % COUNT;
        // SAFETY: mcl elements.
        unsafe {
            if self.floor == 2 {
                let (x, y) = (&self.x2[i], &self.x2[j]);
                let mut r = MFp2::default();
                let mut flag = 0;
                match op {
                    Op::Add => mclBnFp2_add(&mut r, x, y),
                    Op::Mul => mclBnFp2_mul(&mut r, x, y),
                    Op::Sqr => mclBnFp2_sqr(&mut r, x),
                    Op::Inv => mclBnFp2_inv(&mut r, x),
                    Op::Sqrt => flag = (mclBnFp2_squareRoot(&mut r, &self.s2[i]) == 0) as i32,
                    _ => return None,
                }
                return Some((out(&r.d), flag));
            }
            let (x, y) = (&self.x12[i], &self.x12[j]);
            let mut r = MGt::default();
            match op {
                Op::Add => mclBnGT_add(&mut r, x, y),
                Op::Mul => mclBnGT_mul(&mut r, x, y),
                Op::Sqr => mclBnGT_sqr(&mut r, x),
                Op::Inv => mclBnGT_invGeneric(&mut r, x),
                Op::Conj => mclBnGT_inv(&mut r, x),
                Op::CycPow => mclBnGT_powGeneric(&mut r, &self.cyc[i], &self.e),
                Op::Pow => mclBnGT_powGeneric(&mut r, x, &self.e),
                _ => return None,
            }
            Some((out(&r.d), 0))
        }
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        // SAFETY: mcl elements; mcl allows the output to be an input.
        unsafe {
            if self.floor == 2 {
                let mut a = self.a2;
                let mut t = MFp2::default();
                let (ap, tp): (*mut MFp2, *mut MFp2) = (&mut a, &mut t);
                match op {
                    Op::Add => (0..iters).for_each(|i| mclBnFp2_add(ap, ap, &self.x2[i as usize & m])),
                    Op::Mul => (0..iters).for_each(|i| mclBnFp2_mul(ap, ap, &self.x2[i as usize & m])),
                    Op::Sqr => (0..iters).for_each(|_| mclBnFp2_sqr(ap, ap)),
                    Op::Inv => (0..iters).for_each(|i| {
                        mclBnFp2_inv(tp, &self.x2[i as usize & m]);
                        mclBnFp2_add(ap, ap, tp);
                    }),
                    Op::Sqrt => (0..iters).for_each(|i| {
                        mclBnFp2_squareRoot(tp, &self.s2[i as usize & m]);
                        mclBnFp2_add(ap, ap, tp);
                    }),
                    _ => {}
                }
                self.a2 = keep(a);
                return;
            }
            let mut a = if matches!(op, Op::CycPow) { self.cyc[0] } else { self.a12 };
            let mut t = MGt::default();
            let (ap, tp): (*mut MGt, *mut MGt) = (&mut a, &mut t);
            match op {
                Op::Add => (0..iters).for_each(|i| mclBnGT_add(ap, ap, &self.x12[i as usize & m])),
                Op::Mul => (0..iters).for_each(|i| mclBnGT_mul(ap, ap, &self.x12[i as usize & m])),
                Op::Sqr => (0..iters).for_each(|_| mclBnGT_sqr(ap, ap)),
                Op::Inv => (0..iters).for_each(|i| {
                    mclBnGT_invGeneric(tp, &self.x12[i as usize & m]);
                    mclBnGT_add(ap, ap, tp);
                }),
                Op::Conj => (0..iters).for_each(|_| mclBnGT_inv(ap, ap)),
                Op::CycPow => (0..iters).for_each(|_| mclBnGT_powGeneric(ap, ap, &self.e)),
                Op::Pow => (0..iters).for_each(|_| mclBnGT_powGeneric(ap, ap, &self.e)),
                _ => {}
            }
            if matches!(op, Op::CycPow) {
                keep(a);
            } else {
                self.a12 = keep(a);
            }
        }
    }
    fn note(&self, op: Op) -> String {
        match op {
            Op::Conj => "mclBnGT_inv (the conjugate, GT's inverse)".into(),
            Op::Inv if self.floor == 12 => "mclBnGT_invGeneric".into(),
            Op::Pow => "mclBnGT_powGeneric".into(),
            Op::CycPow => "mclBnGT_powGeneric (mclBnGT_pow takes an element of order r: its decomposition of the exponent gives other powers on the cyclotomic subgroup)".into(),
            _ => String::new(),
        }
    }
}

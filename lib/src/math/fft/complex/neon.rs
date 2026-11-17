//! NEON blocks on little-endian aarch64: one `Complex<f64>` in a `float64x2_t`, two
//! `Complex<f32>` in a `float32x4_t` and one in a `float32x2_t`, interleaved as in memory.

use super::blk::Blk;
use crate::math::complex::Complex;
use core::arch::aarch64::*;

/// One `Complex<f64>`.
#[derive(Clone, Copy)]
pub(crate) struct D1(float64x2_t);

/// Two `Complex<f32>`.
#[derive(Clone, Copy)]
pub(crate) struct F2(float32x4_t);

/// One `Complex<f32>`.
#[derive(Clone, Copy)]
pub(crate) struct F1(float32x2_t);

// Signs (+1, −1) and (−1, +1) of the parts of a complex value.
const PM64: [f64; 2] = [1.0, -1.0];
const MP64: [f64; 2] = [-1.0, 1.0];
const PM32: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
const MP32: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];

impl Blk<f64> for D1 {
    const K: usize = 1;
    type Tw = (float64x2_t, float64x2_t);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<f64>) -> Self {
        // SAFETY: the caller's contract; `Complex` is `repr(C)`.
        D1(unsafe { vld1q_f64(p.cast()) })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<f64>) {
        // SAFETY: as for `ld`.
        unsafe { vst1q_f64(p.cast(), self.0) }
    }
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: NEON is always present on aarch64.
        D1(unsafe { vdupq_n_f64(0.0) })
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        D1(unsafe { vaddq_f64(self.0, b.0) })
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        D1(unsafe { vsubq_f64(self.0, b.0) })
    }
    #[inline(always)]
    fn mulr(self, c: f64) -> Self {
        // SAFETY: as for `zero`.
        D1(unsafe { vmulq_n_f64(self.0, c) })
    }
    #[inline(always)]
    fn fmar(self, b: Self, c: f64) -> Self {
        // SAFETY: as for `zero`.
        D1(unsafe { vfmaq_n_f64(self.0, b.0, c) })
    }
    #[inline(always)]
    fn add_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let sw = vextq_f64(b.0, b.0, 1);
            let sg = vld1q_f64(if INV { MP64.as_ptr() } else { PM64.as_ptr() });
            D1(vfmaq_f64(self.0, sw, sg))
        }
    }
    #[inline(always)]
    fn sub_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let sw = vextq_f64(b.0, b.0, 1);
            let sg = vld1q_f64(if INV { PM64.as_ptr() } else { MP64.as_ptr() });
            D1(vfmaq_f64(self.0, sw, sg))
        }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const f64) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { (vld1q_f64(p), vld1q_f64(p.add(2))) }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const f64) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { (vld1q_f64(p), vld1q_f64(p.add(2))) }
    }
    #[inline(always)]
    fn mul_tw<const INV: bool>(self, (a, b): Self::Tw) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let p = vmulq_f64(self.0, a);
            let sw = vextq_f64(self.0, self.0, 1);
            D1(if INV {
                vfmsq_f64(p, sw, b)
            } else {
                vfmaq_f64(p, sw, b)
            })
        }
    }
    #[inline(always)]
    unsafe fn ld_rows(o: &mut [Self], p: *const Complex<f64>, _stride: usize) {
        for (j, v) in o.iter_mut().enumerate() {
            // SAFETY: the caller's contract.
            *v = unsafe { Self::ld(p.add(j)) };
        }
    }
    #[inline(always)]
    unsafe fn st_rows(o: &[Self], p: *mut Complex<f64>, _stride: usize) {
        for (j, v) in o.iter().enumerate() {
            // SAFETY: the caller's contract.
            unsafe { v.st(p.add(j)) }
        }
    }
}

impl Blk<f32> for F2 {
    const K: usize = 2;
    type Tw = (float32x4_t, float32x4_t);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<f32>) -> Self {
        // SAFETY: the caller's contract; `Complex` is `repr(C)`.
        F2(unsafe { vld1q_f32(p.cast()) })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<f32>) {
        // SAFETY: as for `ld`.
        unsafe { vst1q_f32(p.cast(), self.0) }
    }
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: NEON is always present on aarch64.
        F2(unsafe { vdupq_n_f32(0.0) })
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        F2(unsafe { vaddq_f32(self.0, b.0) })
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        F2(unsafe { vsubq_f32(self.0, b.0) })
    }
    #[inline(always)]
    fn mulr(self, c: f32) -> Self {
        // SAFETY: as for `zero`.
        F2(unsafe { vmulq_n_f32(self.0, c) })
    }
    #[inline(always)]
    fn fmar(self, b: Self, c: f32) -> Self {
        // SAFETY: as for `zero`.
        F2(unsafe { vfmaq_n_f32(self.0, b.0, c) })
    }
    #[inline(always)]
    fn add_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let sw = vrev64q_f32(b.0);
            let sg = vld1q_f32(if INV { MP32.as_ptr() } else { PM32.as_ptr() });
            F2(vfmaq_f32(self.0, sw, sg))
        }
    }
    #[inline(always)]
    fn sub_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let sw = vrev64q_f32(b.0);
            let sg = vld1q_f32(if INV { PM32.as_ptr() } else { MP32.as_ptr() });
            F2(vfmaq_f32(self.0, sw, sg))
        }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract; the pairs [wr, wr] and [−wi, wi] as one f64 each.
        unsafe {
            (
                vreinterpretq_f32_f64(vld1q_dup_f64(p.cast())),
                vreinterpretq_f32_f64(vld1q_dup_f64(p.add(2).cast())),
            )
        }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { (vld1q_f32(p), vld1q_f32(p.add(4))) }
    }
    #[inline(always)]
    fn mul_tw<const INV: bool>(self, (a, b): Self::Tw) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let p = vmulq_f32(self.0, a);
            let sw = vrev64q_f32(self.0);
            F2(if INV {
                vfmsq_f32(p, sw, b)
            } else {
                vfmaq_f32(p, sw, b)
            })
        }
    }
    #[inline(always)]
    unsafe fn ld_rows(o: &mut [Self], p: *const Complex<f32>, stride: usize) {
        // SAFETY: the caller's contract; a lane is one f64.
        unsafe {
            let mut j = 0;
            while j + 2 <= o.len() {
                let a = vld1q_f64(p.add(j).cast());
                let b = vld1q_f64(p.add(stride + j).cast());
                o[j] = F2(vreinterpretq_f32_f64(vzip1q_f64(a, b)));
                o[j + 1] = F2(vreinterpretq_f32_f64(vzip2q_f64(a, b)));
                j += 2;
            }
            if j < o.len() {
                o[j] = F2(vcombine_f32(
                    vld1_f32(p.add(j).cast()),
                    vld1_f32(p.add(stride + j).cast()),
                ));
            }
        }
    }
    #[inline(always)]
    unsafe fn st_rows(o: &[Self], p: *mut Complex<f32>, stride: usize) {
        // SAFETY: the caller's contract; a lane is one f64.
        unsafe {
            let mut j = 0;
            while j + 2 <= o.len() {
                let a = vreinterpretq_f64_f32(o[j].0);
                let b = vreinterpretq_f64_f32(o[j + 1].0);
                vst1q_f64(p.add(j).cast(), vzip1q_f64(a, b));
                vst1q_f64(p.add(stride + j).cast(), vzip2q_f64(a, b));
                j += 2;
            }
            if j < o.len() {
                vst1_f32(p.add(j).cast(), vget_low_f32(o[j].0));
                vst1_f32(p.add(stride + j).cast(), vget_high_f32(o[j].0));
            }
        }
    }
}

impl Blk<f32> for F1 {
    const K: usize = 1;
    type Tw = (float32x2_t, float32x2_t);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<f32>) -> Self {
        // SAFETY: the caller's contract; `Complex` is `repr(C)`.
        F1(unsafe { vld1_f32(p.cast()) })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<f32>) {
        // SAFETY: as for `ld`.
        unsafe { vst1_f32(p.cast(), self.0) }
    }
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: NEON is always present on aarch64.
        F1(unsafe { vdup_n_f32(0.0) })
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        F1(unsafe { vadd_f32(self.0, b.0) })
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        F1(unsafe { vsub_f32(self.0, b.0) })
    }
    #[inline(always)]
    fn mulr(self, c: f32) -> Self {
        // SAFETY: as for `zero`.
        F1(unsafe { vmul_n_f32(self.0, c) })
    }
    #[inline(always)]
    fn fmar(self, b: Self, c: f32) -> Self {
        // SAFETY: as for `zero`.
        F1(unsafe { vfma_n_f32(self.0, b.0, c) })
    }
    #[inline(always)]
    fn add_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let sw = vrev64_f32(b.0);
            let sg = vld1_f32(if INV { MP32.as_ptr() } else { PM32.as_ptr() });
            F1(vfma_f32(self.0, sw, sg))
        }
    }
    #[inline(always)]
    fn sub_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let sw = vrev64_f32(b.0);
            let sg = vld1_f32(if INV { PM32.as_ptr() } else { MP32.as_ptr() });
            F1(vfma_f32(self.0, sw, sg))
        }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { (vld1_f32(p), vld1_f32(p.add(2))) }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { (vld1_f32(p), vld1_f32(p.add(2))) }
    }
    #[inline(always)]
    fn mul_tw<const INV: bool>(self, (a, b): Self::Tw) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let p = vmul_f32(self.0, a);
            let sw = vrev64_f32(self.0);
            F1(if INV {
                vfms_f32(p, sw, b)
            } else {
                vfma_f32(p, sw, b)
            })
        }
    }
    #[inline(always)]
    unsafe fn ld_rows(o: &mut [Self], p: *const Complex<f32>, _stride: usize) {
        for (j, v) in o.iter_mut().enumerate() {
            // SAFETY: the caller's contract.
            *v = unsafe { Self::ld(p.add(j)) };
        }
    }
    #[inline(always)]
    unsafe fn st_rows(o: &[Self], p: *mut Complex<f32>, _stride: usize) {
        for (j, v) in o.iter().enumerate() {
            // SAFETY: the caller's contract.
            unsafe { v.st(p.add(j)) }
        }
    }
}

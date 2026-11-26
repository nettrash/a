//! The aarch64 kernels of the convolutions: NEON, multiply-adds by lane of a vector of taps.

/// The values past the last one a kernel needs that it may read (zeros in the caller's copy).
pub(super) const PAD: usize = 64;

/// The factor of the cost of a product of the direct kernels against the measured `mac`: 1, the
/// constants are measured on NEON.
#[inline]
pub(super) fn mac_scale() -> f64 {
    1.0
}

mod s32 {
    use core::arch::aarch64::*;
    type V = float32x4_t;

    #[inline(always)]
    unsafe fn zero() -> V {
        unsafe { vdupq_n_f32(0.0) }
    }
    #[inline(always)]
    unsafe fn ld(p: *const f32) -> V {
        // SAFETY: the caller's.
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    unsafe fn st(p: *mut f32, v: V) {
        // SAFETY: the caller's.
        unsafe { vst1q_f32(p, v) }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> V {
        unsafe { vdupq_n_f32(x) }
    }
    #[inline(always)]
    unsafe fn add(a: V, b: V) -> V {
        unsafe { vaddq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn fma(a: V, x: V, y: V) -> V {
        unsafe { vfmaq_f32(a, x, y) }
    }
    #[inline(always)]
    unsafe fn tload(p: *const f32) -> V {
        // SAFETY: the caller's.
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    unsafe fn tfma<const S: i32>(a: V, x: V, t: V) -> V {
        unsafe { vfmaq_laneq_f32::<S>(a, x, t) }
    }

    #[inline(always)]
    unsafe fn combine(a: V, b: V) -> V {
        // SAFETY: plain arithmetic.
        unsafe {
            let sign = vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr());
            vfmaq_f32(a, vrev64q_f32(b), sign)
        }
    }

    fir_real!(fir, f32, 4, 8, 2, [0, 1, 2, 3]);
    fir_complex!(firc, f32, 4, 8, [(0, 0, 1), (1, 2, 3)]);
    fir_real_few!(fir_few, f32, 4);
    fir_complex_few!(firc_few, f32, 4);

    /// `x[j] = x[j]·y[j]·s` for the `n` complex values at `x` and `y` (interleaved parts).
    ///
    /// # Safety
    ///
    /// `x` must be valid for reads and writes and `y` for reads of `2n` values.
    pub(crate) unsafe fn cmul(x: *mut f32, y: *const f32, s: f32, n: usize) {
        // SAFETY: the loads and stores stay below `2n`.
        unsafe {
            let mut j = 0;
            while j + 4 <= n {
                let a = vld2q_f32(x.add(2 * j));
                let b = vld2q_f32(y.add(2 * j));
                let re = vmulq_n_f32(vfmsq_f32(vmulq_f32(a.0, b.0), a.1, b.1), s);
                let im = vmulq_n_f32(vfmaq_f32(vmulq_f32(a.0, b.1), a.1, b.0), s);
                vst2q_f32(x.add(2 * j), float32x4x2_t(re, im));
                j += 4;
            }
            while j < n {
                let (ar, ai) = (*x.add(2 * j), *x.add(2 * j + 1));
                let (br, bi) = (*y.add(2 * j), *y.add(2 * j + 1));
                *x.add(2 * j) = (ar * br - ai * bi) * s;
                *x.add(2 * j + 1) = (ar * bi + ai * br) * s;
                j += 1;
            }
        }
    }
}

mod s64 {
    use core::arch::aarch64::*;
    type V = float64x2_t;

    #[inline(always)]
    unsafe fn zero() -> V {
        unsafe { vdupq_n_f64(0.0) }
    }
    #[inline(always)]
    unsafe fn ld(p: *const f64) -> V {
        // SAFETY: the caller's.
        unsafe { vld1q_f64(p) }
    }
    #[inline(always)]
    unsafe fn st(p: *mut f64, v: V) {
        // SAFETY: the caller's.
        unsafe { vst1q_f64(p, v) }
    }
    #[inline(always)]
    unsafe fn splat(x: f64) -> V {
        unsafe { vdupq_n_f64(x) }
    }
    #[inline(always)]
    unsafe fn add(a: V, b: V) -> V {
        unsafe { vaddq_f64(a, b) }
    }
    #[inline(always)]
    unsafe fn fma(a: V, x: V, y: V) -> V {
        unsafe { vfmaq_f64(a, x, y) }
    }
    #[inline(always)]
    unsafe fn tload(p: *const f64) -> V {
        // SAFETY: the caller's.
        unsafe { vld1q_f64(p) }
    }
    #[inline(always)]
    unsafe fn tfma<const S: i32>(a: V, x: V, t: V) -> V {
        unsafe { vfmaq_laneq_f64::<S>(a, x, t) }
    }

    #[inline(always)]
    unsafe fn combine(a: V, b: V) -> V {
        // SAFETY: plain arithmetic.
        unsafe {
            let sign = vld1q_f64([-1.0f64, 1.0].as_ptr());
            vfmaq_f64(a, vextq_f64::<1>(b, b), sign)
        }
    }

    fir_real!(fir, f64, 2, 8, 2, [0, 1]);
    fir_complex!(firc, f64, 2, 8, [(0, 0, 1)]);
    fir_real_few!(fir_few, f64, 2);
    fir_complex_few!(firc_few, f64, 2);

    /// `x[j] = x[j]·y[j]·s` for the `n` complex values at `x` and `y` (interleaved parts).
    ///
    /// # Safety
    ///
    /// `x` must be valid for reads and writes and `y` for reads of `2n` values.
    pub(crate) unsafe fn cmul(x: *mut f64, y: *const f64, s: f64, n: usize) {
        // SAFETY: the loads and stores stay below `2n`.
        unsafe {
            let mut j = 0;
            while j + 2 <= n {
                let a = vld2q_f64(x.add(2 * j));
                let b = vld2q_f64(y.add(2 * j));
                let re = vmulq_n_f64(vfmsq_f64(vmulq_f64(a.0, b.0), a.1, b.1), s);
                let im = vmulq_n_f64(vfmaq_f64(vmulq_f64(a.0, b.1), a.1, b.0), s);
                vst2q_f64(x.add(2 * j), float64x2x2_t(re, im));
                j += 2;
            }
            if j < n {
                let (ar, ai) = (*x.add(2 * j), *x.add(2 * j + 1));
                let (br, bi) = (*y.add(2 * j), *y.add(2 * j + 1));
                *x.add(2 * j) = (ar * br - ai * bi) * s;
                *x.add(2 * j + 1) = (ar * bi + ai * br) * s;
            }
        }
    }
}

pub(super) use s32::{
    cmul as cmul_f32, fir as fir_f32, fir_few as fir_few_f32, firc as firc_f32,
    firc_few as firc_few_f32,
};
pub(super) use s64::{
    cmul as cmul_f64, fir as fir_f64, fir_few as fir_few_f64, firc as firc_f64,
    firc_few as firc_few_f64,
};

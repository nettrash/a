//! x86_64 blocks, interleaved as in memory: with SSE2 (always present) one `Complex<f64>` in an
//! `__m128d`, two `Complex<f32>` in an `__m128` and one in its low half; with AVX2 and FMA
//! (detected at run time, used only inside the kernels compiled for them) two `Complex<f64>`
//! in an `__m256d` and four `Complex<f32>` in an `__m256`.

use super::blk::Blk;
use crate::math::complex::Complex;
use core::arch::x86_64::*;

/// One `Complex<f64>` (SSE2).
#[derive(Clone, Copy)]
pub(crate) struct P1(__m128d);

/// Two `Complex<f32>` (SSE2).
#[derive(Clone, Copy)]
pub(crate) struct S2(__m128);

/// One `Complex<f32>` in the low half (SSE2).
#[derive(Clone, Copy)]
pub(crate) struct S1(__m128);

/// Two `Complex<f64>` (AVX2 and FMA).
#[derive(Clone, Copy)]
pub(crate) struct A2(__m256d);

/// Four `Complex<f32>` (AVX2 and FMA).
#[derive(Clone, Copy)]
pub(crate) struct A4(__m256);

// The sign bit of the imaginary or the real parts.
const NEG_IM64: [f64; 2] = [0.0, -0.0];
const NEG_RE64: [f64; 2] = [-0.0, 0.0];
const NEG_IM32: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
const NEG_RE32: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
// Signs (+1, −1) and (−1, +1).
const PM64: [f64; 4] = [1.0, -1.0, 1.0, -1.0];
const MP64: [f64; 4] = [-1.0, 1.0, -1.0, 1.0];
const PM32: [f32; 8] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
const MP32: [f32; 8] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

impl Blk<f64> for P1 {
    const K: usize = 1;
    type Tw = (__m128d, __m128d);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<f64>) -> Self {
        // SAFETY: the caller's contract; `Complex` is `repr(C)`.
        P1(unsafe { _mm_loadu_pd(p.cast()) })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<f64>) {
        // SAFETY: as for `ld`.
        unsafe { _mm_storeu_pd(p.cast(), self.0) }
    }
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: SSE2 is part of x86_64.
        P1(unsafe { _mm_setzero_pd() })
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        P1(unsafe { _mm_add_pd(self.0, b.0) })
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        P1(unsafe { _mm_sub_pd(self.0, b.0) })
    }
    #[inline(always)]
    fn mulr(self, c: f64) -> Self {
        // SAFETY: as for `zero`.
        P1(unsafe { _mm_mul_pd(self.0, _mm_set1_pd(c)) })
    }
    #[inline(always)]
    fn fmar(self, b: Self, c: f64) -> Self {
        // SAFETY: as for `zero`.
        P1(unsafe { _mm_add_pd(self.0, _mm_mul_pd(b.0, _mm_set1_pd(c))) })
    }
    #[inline(always)]
    fn add_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let sw = _mm_shuffle_pd(b.0, b.0, 1);
            let m = _mm_loadu_pd(if INV {
                NEG_RE64.as_ptr()
            } else {
                NEG_IM64.as_ptr()
            });
            P1(_mm_add_pd(self.0, _mm_xor_pd(sw, m)))
        }
    }
    #[inline(always)]
    fn sub_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let sw = _mm_shuffle_pd(b.0, b.0, 1);
            let m = _mm_loadu_pd(if INV {
                NEG_IM64.as_ptr()
            } else {
                NEG_RE64.as_ptr()
            });
            P1(_mm_add_pd(self.0, _mm_xor_pd(sw, m)))
        }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const f64) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { (_mm_loadu_pd(p), _mm_loadu_pd(p.add(2))) }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const f64) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { (_mm_loadu_pd(p), _mm_loadu_pd(p.add(2))) }
    }
    #[inline(always)]
    fn mul_tw<const INV: bool>(self, (a, b): Self::Tw) -> Self {
        // SAFETY: as for `zero`.
        unsafe {
            let p = _mm_mul_pd(self.0, a);
            let q = _mm_mul_pd(_mm_shuffle_pd(self.0, self.0, 1), b);
            P1(if INV {
                _mm_sub_pd(p, q)
            } else {
                _mm_add_pd(p, q)
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

/// Swaps the parts of each complex value of an `__m128`.
#[inline(always)]
fn swap_ps(a: __m128) -> __m128 {
    // SAFETY: SSE is part of x86_64.
    unsafe { _mm_shuffle_ps(a, a, 0b10_11_00_01) }
}

macro_rules! sse_f32 {
    ($t:ident) => {
        #[inline(always)]
        fn zero() -> Self {
            // SAFETY: SSE is part of x86_64.
            $t(unsafe { _mm_setzero_ps() })
        }
        #[inline(always)]
        fn add(self, b: Self) -> Self {
            // SAFETY: as for `zero`.
            $t(unsafe { _mm_add_ps(self.0, b.0) })
        }
        #[inline(always)]
        fn sub(self, b: Self) -> Self {
            // SAFETY: as for `zero`.
            $t(unsafe { _mm_sub_ps(self.0, b.0) })
        }
        #[inline(always)]
        fn mulr(self, c: f32) -> Self {
            // SAFETY: as for `zero`.
            $t(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(c)) })
        }
        #[inline(always)]
        fn fmar(self, b: Self, c: f32) -> Self {
            // SAFETY: as for `zero`.
            $t(unsafe { _mm_add_ps(self.0, _mm_mul_ps(b.0, _mm_set1_ps(c))) })
        }
        #[inline(always)]
        fn add_rot<const INV: bool>(self, b: Self) -> Self {
            // SAFETY: as for `zero`.
            unsafe {
                let m = _mm_loadu_ps(if INV {
                    NEG_RE32.as_ptr()
                } else {
                    NEG_IM32.as_ptr()
                });
                $t(_mm_add_ps(self.0, _mm_xor_ps(swap_ps(b.0), m)))
            }
        }
        #[inline(always)]
        fn sub_rot<const INV: bool>(self, b: Self) -> Self {
            // SAFETY: as for `zero`.
            unsafe {
                let m = _mm_loadu_ps(if INV {
                    NEG_IM32.as_ptr()
                } else {
                    NEG_RE32.as_ptr()
                });
                $t(_mm_add_ps(self.0, _mm_xor_ps(swap_ps(b.0), m)))
            }
        }
        #[inline(always)]
        fn mul_tw<const INV: bool>(self, (a, b): Self::Tw) -> Self {
            // SAFETY: as for `zero`.
            unsafe {
                let p = _mm_mul_ps(self.0, a);
                let q = _mm_mul_ps(swap_ps(self.0), b);
                $t(if INV {
                    _mm_sub_ps(p, q)
                } else {
                    _mm_add_ps(p, q)
                })
            }
        }
    };
}

impl Blk<f32> for S2 {
    const K: usize = 2;
    type Tw = (__m128, __m128);

    sse_f32!(S2);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<f32>) -> Self {
        // SAFETY: the caller's contract; `Complex` is `repr(C)`.
        S2(unsafe { _mm_loadu_ps(p.cast()) })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<f32>) {
        // SAFETY: as for `ld`.
        unsafe { _mm_storeu_ps(p.cast(), self.0) }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract; the pairs [wr, wr] and [−wi, wi] as one f64 each.
        unsafe {
            (
                _mm_castpd_ps(_mm_load1_pd(p.cast())),
                _mm_castpd_ps(_mm_load1_pd(p.add(2).cast())),
            )
        }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { (_mm_loadu_ps(p), _mm_loadu_ps(p.add(4))) }
    }
    #[inline(always)]
    unsafe fn ld_rows(o: &mut [Self], p: *const Complex<f32>, stride: usize) {
        // SAFETY: the caller's contract; a lane is one f64.
        unsafe {
            let mut j = 0;
            while j + 2 <= o.len() {
                let a = _mm_loadu_pd(p.add(j).cast());
                let b = _mm_loadu_pd(p.add(stride + j).cast());
                o[j] = S2(_mm_castpd_ps(_mm_unpacklo_pd(a, b)));
                o[j + 1] = S2(_mm_castpd_ps(_mm_unpackhi_pd(a, b)));
                j += 2;
            }
            if j < o.len() {
                let a = _mm_load_sd(p.add(j).cast());
                o[j] = S2(_mm_castpd_ps(_mm_loadh_pd(a, p.add(stride + j).cast())));
            }
        }
    }
    #[inline(always)]
    unsafe fn st_rows(o: &[Self], p: *mut Complex<f32>, stride: usize) {
        // SAFETY: the caller's contract; a lane is one f64.
        unsafe {
            let mut j = 0;
            while j + 2 <= o.len() {
                let a = _mm_castps_pd(o[j].0);
                let b = _mm_castps_pd(o[j + 1].0);
                _mm_storeu_pd(p.add(j).cast(), _mm_unpacklo_pd(a, b));
                _mm_storeu_pd(p.add(stride + j).cast(), _mm_unpackhi_pd(a, b));
                j += 2;
            }
            if j < o.len() {
                let a = _mm_castps_pd(o[j].0);
                _mm_store_sd(p.add(j).cast(), a);
                _mm_storeh_pd(p.add(stride + j).cast(), a);
            }
        }
    }
}

impl Blk<f32> for S1 {
    const K: usize = 1;
    type Tw = (__m128, __m128);

    sse_f32!(S1);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<f32>) -> Self {
        // SAFETY: the caller's contract; one value is one f64.
        S1(unsafe { _mm_castpd_ps(_mm_load_sd(p.cast())) })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<f32>) {
        // SAFETY: as for `ld`.
        unsafe { _mm_store_sd(p.cast(), _mm_castps_pd(self.0)) }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe {
            (
                _mm_castpd_ps(_mm_load_sd(p.cast())),
                _mm_castpd_ps(_mm_load_sd(p.add(2).cast())),
            )
        }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract.
        unsafe { Self::tw_b(p) }
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

// The AVX2 blocks: their methods are called only from kernels compiled with
// `#[target_feature(enable = "avx2,fma")]` after the features were detected, which is what
// makes the intrinsics sound to call.

impl Blk<f64> for A2 {
    const K: usize = 2;
    type Tw = (__m256d, __m256d);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<f64>) -> Self {
        // SAFETY: the caller's contract; AVX was detected.
        A2(unsafe { _mm256_loadu_pd(p.cast()) })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<f64>) {
        // SAFETY: as for `ld`.
        unsafe { _mm256_storeu_pd(p.cast(), self.0) }
    }
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: AVX was detected (see above).
        A2(unsafe { _mm256_setzero_pd() })
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        A2(unsafe { _mm256_add_pd(self.0, b.0) })
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        A2(unsafe { _mm256_sub_pd(self.0, b.0) })
    }
    #[inline(always)]
    fn mulr(self, c: f64) -> Self {
        // SAFETY: as for `zero`.
        A2(unsafe { _mm256_mul_pd(self.0, _mm256_set1_pd(c)) })
    }
    #[inline(always)]
    fn fmar(self, b: Self, c: f64) -> Self {
        // SAFETY: as for `zero`; FMA was detected.
        A2(unsafe { _mm256_fmadd_pd(b.0, _mm256_set1_pd(c), self.0) })
    }
    #[inline(always)]
    fn add_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `fmar`.
        unsafe {
            let sw = _mm256_permute_pd(b.0, 0b0101);
            let sg = _mm256_loadu_pd(if INV { MP64.as_ptr() } else { PM64.as_ptr() });
            A2(_mm256_fmadd_pd(sw, sg, self.0))
        }
    }
    #[inline(always)]
    fn sub_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `fmar`.
        unsafe {
            let sw = _mm256_permute_pd(b.0, 0b0101);
            let sg = _mm256_loadu_pd(if INV { PM64.as_ptr() } else { MP64.as_ptr() });
            A2(_mm256_fmadd_pd(sw, sg, self.0))
        }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const f64) -> Self::Tw {
        // SAFETY: the caller's contract; AVX was detected.
        unsafe {
            (
                _mm256_broadcast_pd(&*p.cast::<__m128d>()),
                _mm256_broadcast_pd(&*p.add(2).cast::<__m128d>()),
            )
        }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const f64) -> Self::Tw {
        // SAFETY: the caller's contract; AVX was detected.
        unsafe { (_mm256_loadu_pd(p), _mm256_loadu_pd(p.add(4))) }
    }
    #[inline(always)]
    fn mul_tw<const INV: bool>(self, (a, b): Self::Tw) -> Self {
        // SAFETY: as for `fmar`.
        unsafe {
            let p = _mm256_mul_pd(self.0, a);
            let sw = _mm256_permute_pd(self.0, 0b0101);
            A2(if INV {
                _mm256_fnmadd_pd(sw, b, p)
            } else {
                _mm256_fmadd_pd(sw, b, p)
            })
        }
    }
    #[inline(always)]
    unsafe fn ld_rows(o: &mut [Self], p: *const Complex<f64>, stride: usize) {
        // SAFETY: the caller's contract; AVX was detected. A lane is one 128-bit half.
        unsafe {
            let mut j = 0;
            while j + 2 <= o.len() {
                let a = _mm256_loadu_pd(p.add(j).cast());
                let b = _mm256_loadu_pd(p.add(stride + j).cast());
                o[j] = A2(_mm256_permute2f128_pd(a, b, 0x20));
                o[j + 1] = A2(_mm256_permute2f128_pd(a, b, 0x31));
                j += 2;
            }
            if j < o.len() {
                let a = _mm256_castpd128_pd256(_mm_loadu_pd(p.add(j).cast()));
                o[j] = A2(_mm256_insertf128_pd(
                    a,
                    _mm_loadu_pd(p.add(stride + j).cast()),
                    1,
                ));
            }
        }
    }
    #[inline(always)]
    unsafe fn st_rows(o: &[Self], p: *mut Complex<f64>, stride: usize) {
        // SAFETY: the caller's contract; AVX was detected. A lane is one 128-bit half.
        unsafe {
            let mut j = 0;
            while j + 2 <= o.len() {
                let (a, b) = (o[j].0, o[j + 1].0);
                _mm256_storeu_pd(p.add(j).cast(), _mm256_permute2f128_pd(a, b, 0x20));
                _mm256_storeu_pd(p.add(stride + j).cast(), _mm256_permute2f128_pd(a, b, 0x31));
                j += 2;
            }
            if j < o.len() {
                let a = o[j].0;
                _mm_storeu_pd(p.add(j).cast(), _mm256_castpd256_pd128(a));
                _mm_storeu_pd(p.add(stride + j).cast(), _mm256_extractf128_pd(a, 1));
            }
        }
    }
}

impl Blk<f32> for A4 {
    const K: usize = 4;
    type Tw = (__m256, __m256);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<f32>) -> Self {
        // SAFETY: the caller's contract; AVX was detected.
        A4(unsafe { _mm256_loadu_ps(p.cast()) })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<f32>) {
        // SAFETY: as for `ld`.
        unsafe { _mm256_storeu_ps(p.cast(), self.0) }
    }
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: AVX was detected (see above).
        A4(unsafe { _mm256_setzero_ps() })
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        A4(unsafe { _mm256_add_ps(self.0, b.0) })
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: as for `zero`.
        A4(unsafe { _mm256_sub_ps(self.0, b.0) })
    }
    #[inline(always)]
    fn mulr(self, c: f32) -> Self {
        // SAFETY: as for `zero`.
        A4(unsafe { _mm256_mul_ps(self.0, _mm256_set1_ps(c)) })
    }
    #[inline(always)]
    fn fmar(self, b: Self, c: f32) -> Self {
        // SAFETY: as for `zero`; FMA was detected.
        A4(unsafe { _mm256_fmadd_ps(b.0, _mm256_set1_ps(c), self.0) })
    }
    #[inline(always)]
    fn add_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `fmar`.
        unsafe {
            let sw = _mm256_permute_ps(b.0, 0b10_11_00_01);
            let sg = _mm256_loadu_ps(if INV { MP32.as_ptr() } else { PM32.as_ptr() });
            A4(_mm256_fmadd_ps(sw, sg, self.0))
        }
    }
    #[inline(always)]
    fn sub_rot<const INV: bool>(self, b: Self) -> Self {
        // SAFETY: as for `fmar`.
        unsafe {
            let sw = _mm256_permute_ps(b.0, 0b10_11_00_01);
            let sg = _mm256_loadu_ps(if INV { PM32.as_ptr() } else { MP32.as_ptr() });
            A4(_mm256_fmadd_ps(sw, sg, self.0))
        }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract; the pairs [wr, wr] and [−wi, wi] as one f64 each.
        unsafe {
            (
                _mm256_castpd_ps(_mm256_broadcast_sd(&*p.cast::<f64>())),
                _mm256_castpd_ps(_mm256_broadcast_sd(&*p.add(2).cast::<f64>())),
            )
        }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const f32) -> Self::Tw {
        // SAFETY: the caller's contract; AVX was detected.
        unsafe { (_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8))) }
    }
    #[inline(always)]
    fn mul_tw<const INV: bool>(self, (a, b): Self::Tw) -> Self {
        // SAFETY: as for `fmar`.
        unsafe {
            let p = _mm256_mul_ps(self.0, a);
            let sw = _mm256_permute_ps(self.0, 0b10_11_00_01);
            A4(if INV {
                _mm256_fnmadd_ps(sw, b, p)
            } else {
                _mm256_fmadd_ps(sw, b, p)
            })
        }
    }
    #[inline(always)]
    unsafe fn ld_rows(o: &mut [Self], p: *const Complex<f32>, stride: usize) {
        // Not used by the passes (see `st_rows`); by lanes.
        for (j, v) in o.iter_mut().enumerate() {
            let mut t = [Complex::<f32>::ZERO; 4];
            // SAFETY: the caller's contract.
            unsafe {
                for (l, c) in t.iter_mut().enumerate() {
                    *c = p.add(l * stride + j).read();
                }
                *v = Self::ld(t.as_ptr());
            }
        }
    }
    #[inline(always)]
    unsafe fn st_rows(o: &[Self], p: *mut Complex<f32>, stride: usize) {
        // Not used by the passes (the row form of AVX2 `f32` uses `S2`); by lanes.
        for (j, v) in o.iter().enumerate() {
            let mut t = [Complex::<f32>::ZERO; 4];
            // SAFETY: the caller's contract.
            unsafe {
                v.st(t.as_mut_ptr());
                for (l, c) in t.iter().enumerate() {
                    p.add(l * stride + j).write(*c);
                }
            }
        }
    }
}

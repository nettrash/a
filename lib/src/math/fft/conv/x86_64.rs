//! The x86_64 kernels of the convolutions: SSE2 always, AVX2 with FMA when the CPU has them,
//! looked up at each call (the standard library keeps the answer).
//!
//! These kernels have been tested for correctness under emulation only; their speed has not been
//! measured.
//!
//! Which intrinsics need an `unsafe` block depends on the Rust version (those of the enabled
//! target features became safe to call over the releases), so the blocks are kept and the lint
//! allowed.

#![allow(unused_unsafe)]

/// The values past the last one a kernel needs that it may read (zeros in the caller's copy).
pub(super) const PAD: usize = 64;

/// The factor of the cost of a product of the direct kernels against the measured `mac` (of
/// NEON): 1 with AVX2 and FMA (16 lanes of multiply-adds a cycle on two ports, as NEON's four
/// pipes), 2 with SSE2 alone: its kernels take a quarter of AVX2's products a cycle (4 lanes, a
/// multiply and an add each), and the transforms on SSE2 about half the speed of AVX2's, so the
/// direct path loses about twice as much against them. An estimate: x86 has not been timed.
#[inline]
pub(super) fn mac_scale() -> f64 {
    if avx2() { 1.0 } else { 2.0 }
}

/// Defines the scalar-tap helpers over a vector type: the taps are read from memory and broadcast.
macro_rules! broadcast_taps {
    ($(#[$m:meta])* $t:ty, $v:ty, $set1:ident) => {
        $(#[$m])*
        #[inline]
        unsafe fn tload(p: *const $t) -> *const $t {
            p
        }
        $(#[$m])*
        #[inline]
        unsafe fn tfma<const S: i32>(a: $v, x: $v, t: *const $t) -> $v {
            // SAFETY: the caller's: `t` points at a group of the taps.
            unsafe { fma(a, x, $set1(*t.add(S as usize))) }
        }
        $(#[$m])*
        #[inline]
        unsafe fn splat(x: $t) -> $v {
            unsafe { $set1(x) }
        }
    };
}

/// SSE2: 4 lanes of `f32`, separate multiplies and adds.
mod sse32 {
    use core::arch::x86_64::*;
    type V = __m128;

    #[inline]
    unsafe fn zero() -> V {
        unsafe { _mm_setzero_ps() }
    }
    #[inline]
    unsafe fn ld(p: *const f32) -> V {
        // SAFETY: the caller's.
        unsafe { _mm_loadu_ps(p) }
    }
    #[inline]
    unsafe fn st(p: *mut f32, v: V) {
        // SAFETY: the caller's.
        unsafe { _mm_storeu_ps(p, v) }
    }
    #[inline]
    unsafe fn add(a: V, b: V) -> V {
        unsafe { _mm_add_ps(a, b) }
    }
    #[inline]
    unsafe fn fma(a: V, x: V, y: V) -> V {
        unsafe { _mm_add_ps(a, _mm_mul_ps(x, y)) }
    }
    broadcast_taps!(f32, V, _mm_set1_ps);

    #[inline]
    unsafe fn combine(a: V, b: V) -> V {
        unsafe {
            let sign = _mm_setr_ps(-1.0, 1.0, -1.0, 1.0);
            _mm_add_ps(a, _mm_mul_ps(_mm_shuffle_ps::<0b10_11_00_01>(b, b), sign))
        }
    }

    fir_real!(fir, f32, 4, 4, 2, [0, 1, 2, 3]);
    fir_complex!(firc, f32, 4, 4, [(0, 0, 1), (1, 2, 3)]);

    /// `x[j] = x[j]·y[j]·s` for the `n` complex values at `x` and `y` (interleaved parts).
    ///
    /// # Safety
    ///
    /// `x` must be valid for reads and writes and `y` for reads of `2n` values.
    pub(crate) unsafe fn cmul(x: *mut f32, y: *const f32, s: f32, n: usize) {
        // SAFETY: the loads and stores stay below `2n`.
        unsafe {
            let sign = _mm_setr_ps(-s, s, -s, s);
            let sv = _mm_set1_ps(s);
            let mut j = 0;
            while j + 2 <= n {
                let a = _mm_loadu_ps(x.add(2 * j));
                let b = _mm_loadu_ps(y.add(2 * j));
                let br = _mm_shuffle_ps::<0b10_10_00_00>(b, b);
                let bi = _mm_shuffle_ps::<0b11_11_01_01>(b, b);
                let sw = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                let r = _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(a, br), sv),
                    _mm_mul_ps(_mm_mul_ps(sw, bi), sign),
                );
                _mm_storeu_ps(x.add(2 * j), r);
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

/// SSE2: 2 lanes of `f64`.
mod sse64 {
    use core::arch::x86_64::*;
    type V = __m128d;

    #[inline]
    unsafe fn zero() -> V {
        unsafe { _mm_setzero_pd() }
    }
    #[inline]
    unsafe fn ld(p: *const f64) -> V {
        // SAFETY: the caller's.
        unsafe { _mm_loadu_pd(p) }
    }
    #[inline]
    unsafe fn st(p: *mut f64, v: V) {
        // SAFETY: the caller's.
        unsafe { _mm_storeu_pd(p, v) }
    }
    #[inline]
    unsafe fn add(a: V, b: V) -> V {
        unsafe { _mm_add_pd(a, b) }
    }
    #[inline]
    unsafe fn fma(a: V, x: V, y: V) -> V {
        unsafe { _mm_add_pd(a, _mm_mul_pd(x, y)) }
    }
    broadcast_taps!(f64, V, _mm_set1_pd);

    #[inline]
    unsafe fn combine(a: V, b: V) -> V {
        unsafe {
            let sign = _mm_setr_pd(-1.0, 1.0);
            _mm_add_pd(a, _mm_mul_pd(_mm_shuffle_pd::<0b01>(b, b), sign))
        }
    }

    fir_real!(fir, f64, 2, 4, 2, [0, 1]);
    fir_complex!(firc, f64, 2, 4, [(0, 0, 1)]);

    /// `x[j] = x[j]·y[j]·s` for the `n` complex values at `x` and `y` (interleaved parts).
    ///
    /// # Safety
    ///
    /// `x` must be valid for reads and writes and `y` for reads of `2n` values.
    pub(crate) unsafe fn cmul(x: *mut f64, y: *const f64, s: f64, n: usize) {
        // SAFETY: the loads and stores stay below `2n`.
        unsafe {
            let sign = _mm_setr_pd(-s, s);
            let sv = _mm_set1_pd(s);
            for j in 0..n {
                let a = _mm_loadu_pd(x.add(2 * j));
                let b = _mm_loadu_pd(y.add(2 * j));
                let br = _mm_unpacklo_pd(b, b);
                let bi = _mm_unpackhi_pd(b, b);
                let sw = _mm_shuffle_pd::<0b01>(a, a);
                let r = _mm_add_pd(
                    _mm_mul_pd(_mm_mul_pd(a, br), sv),
                    _mm_mul_pd(_mm_mul_pd(sw, bi), sign),
                );
                _mm_storeu_pd(x.add(2 * j), r);
            }
        }
    }
}

/// AVX2 and FMA: 8 lanes of `f32`.
mod avx32 {
    use core::arch::x86_64::*;
    type V = __m256;

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn zero() -> V {
        _mm256_setzero_ps()
    }
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn ld(p: *const f32) -> V {
        // SAFETY: the caller's.
        unsafe { _mm256_loadu_ps(p) }
    }
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn st(p: *mut f32, v: V) {
        // SAFETY: the caller's.
        unsafe { _mm256_storeu_ps(p, v) }
    }
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn add(a: V, b: V) -> V {
        _mm256_add_ps(a, b)
    }
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fma(a: V, x: V, y: V) -> V {
        _mm256_fmadd_ps(x, y, a)
    }
    broadcast_taps!(
        #[target_feature(enable = "avx2,fma")]
        f32,
        V,
        _mm256_set1_ps
    );

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn combine(a: V, b: V) -> V {
        _mm256_addsub_ps(a, _mm256_permute_ps::<0b10_11_00_01>(b))
    }

    fir_real!(
        #[target_feature(enable = "avx2,fma")]
        fir,
        f32,
        8,
        4,
        2,
        [0, 1, 2, 3, 4, 5, 6, 7]
    );
    fir_complex!(
        #[target_feature(enable = "avx2,fma")]
        firc,
        f32,
        8,
        4,
        [(0, 0, 1), (1, 2, 3), (2, 4, 5), (3, 6, 7)]
    );

    /// `x[j] = x[j]·y[j]·s` for the `n` complex values at `x` and `y` (interleaved parts).
    ///
    /// # Safety
    ///
    /// The CPU must have AVX2 and FMA; `x` must be valid for reads and writes and `y` for reads of
    /// `2n` values.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn cmul(x: *mut f32, y: *const f32, s: f32, n: usize) {
        // SAFETY: the loads and stores stay below `2n`.
        unsafe {
            let sv = _mm256_set1_ps(s);
            let mut j = 0;
            while j + 4 <= n {
                let a = _mm256_loadu_ps(x.add(2 * j));
                let b = _mm256_mul_ps(_mm256_loadu_ps(y.add(2 * j)), sv);
                let br = _mm256_moveldup_ps(b);
                let bi = _mm256_movehdup_ps(b);
                let sw = _mm256_permute_ps::<0b10_11_00_01>(a);
                let r = _mm256_fmaddsub_ps(a, br, _mm256_mul_ps(sw, bi));
                _mm256_storeu_ps(x.add(2 * j), r);
                j += 4;
            }
            while j < n {
                let (ar, ai) = (*x.add(2 * j), *x.add(2 * j + 1));
                let (br, bi) = (*y.add(2 * j) * s, *y.add(2 * j + 1) * s);
                *x.add(2 * j) = ar * br - ai * bi;
                *x.add(2 * j + 1) = ar * bi + ai * br;
                j += 1;
            }
        }
    }
}

/// AVX2 and FMA: 4 lanes of `f64`.
mod avx64 {
    use core::arch::x86_64::*;
    type V = __m256d;

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn zero() -> V {
        _mm256_setzero_pd()
    }
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn ld(p: *const f64) -> V {
        // SAFETY: the caller's.
        unsafe { _mm256_loadu_pd(p) }
    }
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn st(p: *mut f64, v: V) {
        // SAFETY: the caller's.
        unsafe { _mm256_storeu_pd(p, v) }
    }
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn add(a: V, b: V) -> V {
        _mm256_add_pd(a, b)
    }
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fma(a: V, x: V, y: V) -> V {
        _mm256_fmadd_pd(x, y, a)
    }
    broadcast_taps!(
        #[target_feature(enable = "avx2,fma")]
        f64,
        V,
        _mm256_set1_pd
    );

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn combine(a: V, b: V) -> V {
        _mm256_addsub_pd(a, _mm256_permute_pd::<0b0101>(b))
    }

    fir_real!(
        #[target_feature(enable = "avx2,fma")]
        fir,
        f64,
        4,
        4,
        2,
        [0, 1, 2, 3]
    );
    fir_complex!(
        #[target_feature(enable = "avx2,fma")]
        firc,
        f64,
        4,
        4,
        [(0, 0, 1), (1, 2, 3)]
    );

    /// `x[j] = x[j]·y[j]·s` for the `n` complex values at `x` and `y` (interleaved parts).
    ///
    /// # Safety
    ///
    /// The CPU must have AVX2 and FMA; `x` must be valid for reads and writes and `y` for reads of
    /// `2n` values.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn cmul(x: *mut f64, y: *const f64, s: f64, n: usize) {
        // SAFETY: the loads and stores stay below `2n`.
        unsafe {
            let sv = _mm256_set1_pd(s);
            let mut j = 0;
            while j + 2 <= n {
                let a = _mm256_loadu_pd(x.add(2 * j));
                let b = _mm256_mul_pd(_mm256_loadu_pd(y.add(2 * j)), sv);
                let br = _mm256_movedup_pd(b);
                let bi = _mm256_permute_pd::<0b1111>(b);
                let sw = _mm256_permute_pd::<0b0101>(a);
                let r = _mm256_fmaddsub_pd(a, br, _mm256_mul_pd(sw, bi));
                _mm256_storeu_pd(x.add(2 * j), r);
                j += 2;
            }
            if j < n {
                let (ar, ai) = (*x.add(2 * j), *x.add(2 * j + 1));
                let (br, bi) = (*y.add(2 * j) * s, *y.add(2 * j + 1) * s);
                *x.add(2 * j) = ar * br - ai * bi;
                *x.add(2 * j + 1) = ar * bi + ai * br;
            }
        }
    }
}

/// Whether the CPU has AVX2 and FMA.
#[inline]
fn avx2() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

macro_rules! dispatch {
    ($name:ident, $sse:path, $avx:path, ($($a:ident: $ty:ty),*)) => {
        /// The AVX2 kernel when the CPU has it, else the SSE2 one.
        ///
        /// # Safety
        ///
        /// That of the kernels.
        #[inline]
        pub(crate) unsafe fn $name($($a: $ty),*) {
            // SAFETY: forwarded; AVX2 and FMA are checked first.
            unsafe {
                if avx2() { $avx($($a),*) } else { $sse($($a),*) }
            }
        }
    };
}

dispatch!(fir_f32, sse32::fir, avx32::fir, (x: *const f32, t: *const f32, nt: usize, y: *mut f32, ny: usize));
dispatch!(fir_f64, sse64::fir, avx64::fir, (x: *const f64, t: *const f64, nt: usize, y: *mut f64, ny: usize));
dispatch!(firc_f32, sse32::firc, avx32::firc, (x: *const f32, t: *const f32, nt: usize, y: *mut f32, ny: usize));
dispatch!(firc_f64, sse64::firc, avx64::firc, (x: *const f64, t: *const f64, nt: usize, y: *mut f64, ny: usize));
dispatch!(cmul_f32, sse32::cmul, avx32::cmul, (x: *mut f32, y: *const f32, s: f32, n: usize));
dispatch!(cmul_f64, sse64::cmul, avx64::cmul, (x: *mut f64, y: *const f64, s: f64, n: usize));

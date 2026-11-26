//! The SSE2 and AVX2 + FMA kernels of the passes of the real transforms (x86_64): the
//! expressions of [`portable`](super::portable) with the parts split into lanes by shuffles.
//! SSE2 is always there; AVX2 and FMA are detected once, when a plan is made.

use core::arch::x86_64::*;

/// The instruction set of the kernels: SSE2, or AVX2 with FMA.
#[derive(Clone, Copy, Debug)]
pub(in crate::math::fft) struct Isa {
    avx2: bool,
}

impl Isa {
    pub(in crate::math::fft) fn detect() -> Isa {
        Isa {
            avx2: !cfg!(a_baseline)
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma"),
        }
    }
}

/// [`portable::pairs`](super::portable::pairs) for `f64` from `k = 1`, two (SSE2) or four (AVX2)
/// pairs a step; returns the `k` the remainder starts from.
///
/// # Safety
///
/// As for `portable::pairs`, the pointers to interleaved complex values.
#[inline]
pub(in crate::math::fft) unsafe fn pairs_f64(
    isa: Isa,
    dst: *mut f64,
    src: *const f64,
    m: usize,
    tw: *const f64,
    s: f64,
) -> usize {
    // SAFETY: the caller's guarantees; AVX2 and FMA were detected.
    unsafe {
        if isa.avx2 {
            pairs_f64_avx2(dst, src, m, tw, s)
        } else {
            pairs_f64_sse2(dst, src, m, tw, s)
        }
    }
}

/// [`pairs_f64`] for `f32`, four (SSE2) or eight (AVX2) pairs a step.
///
/// # Safety
///
/// As for [`pairs_f64`].
#[inline]
pub(in crate::math::fft) unsafe fn pairs_f32(
    isa: Isa,
    dst: *mut f32,
    src: *const f32,
    m: usize,
    tw: *const f32,
    s: f32,
) -> usize {
    // SAFETY: as in pairs_f64.
    unsafe {
        if isa.avx2 {
            pairs_f32_avx2(dst, src, m, tw, s)
        } else {
            pairs_f32_sse2(dst, src, m, tw, s)
        }
    }
}

unsafe fn pairs_f64_sse2(
    dst: *mut f64,
    src: *const f64,
    m: usize,
    tw: *const f64,
    s: f64,
) -> usize {
    let mut k = 1;
    // SAFETY: the values k, k + 1 and m − k − 1, m − k are in bounds and disjoint while
    // 2(k + 1) < m; all four are loaded before any is stored.
    unsafe {
        let sv = _mm_set1_pd(s);
        while 2 * (k + 1) < m {
            let (a0, a1) = (
                _mm_loadu_pd(src.add(2 * k)),
                _mm_loadu_pd(src.add(2 * k + 2)),
            );
            let (ar, ai) = (_mm_unpacklo_pd(a0, a1), _mm_unpackhi_pd(a0, a1));
            let j = 2 * (m - k - 1);
            let (t0, t1) = (_mm_loadu_pd(src.add(j)), _mm_loadu_pd(src.add(j + 2)));
            // Lane 0 holds m − k, lane 1 m − k − 1.
            let (br, bi) = (_mm_unpacklo_pd(t1, t0), _mm_unpackhi_pd(t1, t0));
            let (w0, w1) = (_mm_loadu_pd(tw.add(2 * k)), _mm_loadu_pd(tw.add(2 * k + 2)));
            let (wr, wi) = (_mm_unpacklo_pd(w0, w1), _mm_unpackhi_pd(w0, w1));
            let ur = _mm_mul_pd(_mm_add_pd(ar, br), sv);
            let ui = _mm_mul_pd(_mm_sub_pd(ai, bi), sv);
            let dr = _mm_sub_pd(ar, br);
            let di = _mm_add_pd(ai, bi);
            let tr = _mm_sub_pd(_mm_mul_pd(wr, dr), _mm_mul_pd(wi, di));
            let ti = _mm_add_pd(_mm_mul_pd(wr, di), _mm_mul_pd(wi, dr));
            let (xr, xi) = (_mm_add_pd(ur, tr), _mm_add_pd(ui, ti));
            _mm_storeu_pd(dst.add(2 * k), _mm_unpacklo_pd(xr, xi));
            _mm_storeu_pd(dst.add(2 * k + 2), _mm_unpackhi_pd(xr, xi));
            let (yr, yi) = (_mm_sub_pd(ur, tr), _mm_sub_pd(ti, ui));
            _mm_storeu_pd(dst.add(j + 2), _mm_unpacklo_pd(yr, yi));
            _mm_storeu_pd(dst.add(j), _mm_unpackhi_pd(yr, yi));
            k += 2;
        }
    }
    k
}

#[target_feature(enable = "avx2,fma")]
unsafe fn pairs_f64_avx2(
    dst: *mut f64,
    src: *const f64,
    m: usize,
    tw: *const f64,
    s: f64,
) -> usize {
    let mut k = 1;
    // SAFETY: as in pairs_f64_sse2, with the values k..k + 4 and m − k − 3..=m − k. The lanes
    // hold k, k + 2, k + 1, k + 3 (and their partners) after the in-lane unpacks.
    unsafe {
        let sv = _mm256_set1_pd(s);
        while 2 * (k + 3) < m {
            let (a0, a1) = (
                _mm256_loadu_pd(src.add(2 * k)),
                _mm256_loadu_pd(src.add(2 * k + 4)),
            );
            let (ar, ai) = (_mm256_unpacklo_pd(a0, a1), _mm256_unpackhi_pd(a0, a1));
            let j = 2 * (m - k - 3);
            let t0 = _mm256_loadu_pd(src.add(j));
            let t1 = _mm256_loadu_pd(src.add(j + 4));
            let (t0, t1) = (
                _mm256_permute2f128_pd(t0, t0, 1),
                _mm256_permute2f128_pd(t1, t1, 1),
            );
            let (br, bi) = (_mm256_unpacklo_pd(t1, t0), _mm256_unpackhi_pd(t1, t0));
            let (w0, w1) = (
                _mm256_loadu_pd(tw.add(2 * k)),
                _mm256_loadu_pd(tw.add(2 * k + 4)),
            );
            let (wr, wi) = (_mm256_unpacklo_pd(w0, w1), _mm256_unpackhi_pd(w0, w1));
            let ur = _mm256_mul_pd(_mm256_add_pd(ar, br), sv);
            let ui = _mm256_mul_pd(_mm256_sub_pd(ai, bi), sv);
            let dr = _mm256_sub_pd(ar, br);
            let di = _mm256_add_pd(ai, bi);
            let tr = _mm256_fmsub_pd(wr, dr, _mm256_mul_pd(wi, di));
            let ti = _mm256_fmadd_pd(wr, di, _mm256_mul_pd(wi, dr));
            let (xr, xi) = (_mm256_add_pd(ur, tr), _mm256_add_pd(ui, ti));
            _mm256_storeu_pd(dst.add(2 * k), _mm256_unpacklo_pd(xr, xi));
            _mm256_storeu_pd(dst.add(2 * k + 4), _mm256_unpackhi_pd(xr, xi));
            let (yr, yi) = (_mm256_sub_pd(ur, tr), _mm256_sub_pd(ti, ui));
            let (lo, hi) = (_mm256_unpacklo_pd(yr, yi), _mm256_unpackhi_pd(yr, yi));
            _mm256_storeu_pd(dst.add(j + 4), _mm256_permute2f128_pd(lo, lo, 1));
            _mm256_storeu_pd(dst.add(j), _mm256_permute2f128_pd(hi, hi, 1));
            k += 4;
        }
    }
    k
}

unsafe fn pairs_f32_sse2(
    dst: *mut f32,
    src: *const f32,
    m: usize,
    tw: *const f32,
    s: f32,
) -> usize {
    let mut k = 1;
    // SAFETY: as in pairs_f64_sse2, with the values k..k + 4 and m − k − 3..=m − k.
    unsafe {
        let sv = _mm_set1_ps(s);
        while 2 * (k + 3) < m {
            let (a0, a1) = (
                _mm_loadu_ps(src.add(2 * k)),
                _mm_loadu_ps(src.add(2 * k + 4)),
            );
            let ar = _mm_shuffle_ps::<0b10_00_10_00>(a0, a1);
            let ai = _mm_shuffle_ps::<0b11_01_11_01>(a0, a1);
            let j = 2 * (m - k - 3);
            let (t0, t1) = (_mm_loadu_ps(src.add(j)), _mm_loadu_ps(src.add(j + 4)));
            // Lanes m − k, m − k − 1, m − k − 2, m − k − 3.
            let br = _mm_shuffle_ps::<0b00_10_00_10>(t1, t0);
            let bi = _mm_shuffle_ps::<0b01_11_01_11>(t1, t0);
            let (w0, w1) = (_mm_loadu_ps(tw.add(2 * k)), _mm_loadu_ps(tw.add(2 * k + 4)));
            let wr = _mm_shuffle_ps::<0b10_00_10_00>(w0, w1);
            let wi = _mm_shuffle_ps::<0b11_01_11_01>(w0, w1);
            let ur = _mm_mul_ps(_mm_add_ps(ar, br), sv);
            let ui = _mm_mul_ps(_mm_sub_ps(ai, bi), sv);
            let dr = _mm_sub_ps(ar, br);
            let di = _mm_add_ps(ai, bi);
            let tr = _mm_sub_ps(_mm_mul_ps(wr, dr), _mm_mul_ps(wi, di));
            let ti = _mm_add_ps(_mm_mul_ps(wr, di), _mm_mul_ps(wi, dr));
            let (xr, xi) = (_mm_add_ps(ur, tr), _mm_add_ps(ui, ti));
            _mm_storeu_ps(dst.add(2 * k), _mm_unpacklo_ps(xr, xi));
            _mm_storeu_ps(dst.add(2 * k + 4), _mm_unpackhi_ps(xr, xi));
            let (yr, yi) = (_mm_sub_ps(ur, tr), _mm_sub_ps(ti, ui));
            let (lo, hi) = (_mm_unpacklo_ps(yr, yi), _mm_unpackhi_ps(yr, yi));
            _mm_storeu_ps(dst.add(j + 4), _mm_shuffle_ps::<0b01_00_11_10>(lo, lo));
            _mm_storeu_ps(dst.add(j), _mm_shuffle_ps::<0b01_00_11_10>(hi, hi));
            k += 4;
        }
    }
    k
}

#[target_feature(enable = "avx2,fma")]
unsafe fn pairs_f32_avx2(
    dst: *mut f32,
    src: *const f32,
    m: usize,
    tw: *const f32,
    s: f32,
) -> usize {
    let mut k = 1;
    // SAFETY: as in pairs_f64_sse2, with the values k..k + 8 and m − k − 7..=m − k. After the
    // in-lane shuffles the lanes hold k + 0, 1, 4, 5, 2, 3, 6, 7 (and their partners); the
    // top values are first put in the reverse order of complex values.
    unsafe {
        let sv = _mm256_set1_ps(s);
        let rev = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
        while 2 * (k + 7) < m {
            let (a0, a1) = (
                _mm256_loadu_ps(src.add(2 * k)),
                _mm256_loadu_ps(src.add(2 * k + 8)),
            );
            let ar = _mm256_shuffle_ps::<0b10_00_10_00>(a0, a1);
            let ai = _mm256_shuffle_ps::<0b11_01_11_01>(a0, a1);
            let j = 2 * (m - k - 7);
            let t0 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(src.add(j)), rev);
            let t1 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(src.add(j + 8)), rev);
            let br = _mm256_shuffle_ps::<0b10_00_10_00>(t1, t0);
            let bi = _mm256_shuffle_ps::<0b11_01_11_01>(t1, t0);
            let (w0, w1) = (
                _mm256_loadu_ps(tw.add(2 * k)),
                _mm256_loadu_ps(tw.add(2 * k + 8)),
            );
            let wr = _mm256_shuffle_ps::<0b10_00_10_00>(w0, w1);
            let wi = _mm256_shuffle_ps::<0b11_01_11_01>(w0, w1);
            let ur = _mm256_mul_ps(_mm256_add_ps(ar, br), sv);
            let ui = _mm256_mul_ps(_mm256_sub_ps(ai, bi), sv);
            let dr = _mm256_sub_ps(ar, br);
            let di = _mm256_add_ps(ai, bi);
            let tr = _mm256_fmsub_ps(wr, dr, _mm256_mul_ps(wi, di));
            let ti = _mm256_fmadd_ps(wr, di, _mm256_mul_ps(wi, dr));
            let (xr, xi) = (_mm256_add_ps(ur, tr), _mm256_add_ps(ui, ti));
            _mm256_storeu_ps(dst.add(2 * k), _mm256_unpacklo_ps(xr, xi));
            _mm256_storeu_ps(dst.add(2 * k + 8), _mm256_unpackhi_ps(xr, xi));
            let (yr, yi) = (_mm256_sub_ps(ur, tr), _mm256_sub_ps(ti, ui));
            let lo = _mm256_permutevar8x32_ps(_mm256_unpacklo_ps(yr, yi), rev);
            let hi = _mm256_permutevar8x32_ps(_mm256_unpackhi_ps(yr, yi), rev);
            _mm256_storeu_ps(dst.add(j + 8), lo);
            _mm256_storeu_ps(dst.add(j), hi);
            k += 8;
        }
    }
    k
}

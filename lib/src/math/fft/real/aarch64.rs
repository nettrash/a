//! The NEON kernels of the passes of the real transforms (little-endian aarch64): the
//! expressions of [`portable`](super::portable), two `f64` or four `f32` pairs a step, the parts
//! split into lanes by the de-interleaving loads.

use core::arch::aarch64::*;

/// The instruction set of the kernels: NEON, always there.
#[derive(Clone, Copy, Debug)]
pub(in crate::math::fft) struct Isa;

impl Isa {
    pub(in crate::math::fft) fn detect() -> Isa {
        Isa
    }
}

/// [`portable::pairs`](super::portable::pairs) for `f64` from `k = 1` in steps of two; returns
/// the `k` the remainder starts from.
///
/// # Safety
///
/// As for `portable::pairs`, the pointers to interleaved complex values.
#[inline]
pub(in crate::math::fft) unsafe fn pairs_f64(
    _: Isa,
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
        let sv = vdupq_n_f64(s);
        while 2 * (k + 1) < m {
            let a = vld2q_f64(src.add(2 * k));
            let t = vld2q_f64(src.add(2 * (m - k - 1)));
            let w = vld2q_f64(tw.add(2 * k));
            let (br, bi) = (vextq_f64(t.0, t.0, 1), vextq_f64(t.1, t.1, 1));
            let ur = vmulq_f64(vaddq_f64(a.0, br), sv);
            let ui = vmulq_f64(vsubq_f64(a.1, bi), sv);
            let dr = vsubq_f64(a.0, br);
            let di = vaddq_f64(a.1, bi);
            let tr = vfmsq_f64(vmulq_f64(w.0, dr), w.1, di);
            let ti = vfmaq_f64(vmulq_f64(w.0, di), w.1, dr);
            vst2q_f64(
                dst.add(2 * k),
                float64x2x2_t(vaddq_f64(ur, tr), vaddq_f64(ui, ti)),
            );
            let (or, oi) = (vsubq_f64(ur, tr), vsubq_f64(ti, ui));
            vst2q_f64(
                dst.add(2 * (m - k - 1)),
                float64x2x2_t(vextq_f64(or, or, 1), vextq_f64(oi, oi, 1)),
            );
            k += 2;
        }
    }
    k
}

/// The lanes of `v` in the reverse order.
#[inline(always)]
unsafe fn rev(v: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is always there.
    unsafe {
        let r = vrev64q_f32(v);
        vextq_f32(r, r, 2)
    }
}

/// [`pairs_f64`] for `f32`, four pairs a step.
///
/// # Safety
///
/// As for [`pairs_f64`].
#[inline]
pub(in crate::math::fft) unsafe fn pairs_f32(
    _: Isa,
    dst: *mut f32,
    src: *const f32,
    m: usize,
    tw: *const f32,
    s: f32,
) -> usize {
    let mut k = 1;
    // SAFETY: as in pairs_f64, with the values k..k + 4 and m − k − 3..=m − k.
    unsafe {
        let sv = vdupq_n_f32(s);
        while 2 * (k + 3) < m {
            let a = vld2q_f32(src.add(2 * k));
            let t = vld2q_f32(src.add(2 * (m - k - 3)));
            let w = vld2q_f32(tw.add(2 * k));
            let (br, bi) = (rev(t.0), rev(t.1));
            let ur = vmulq_f32(vaddq_f32(a.0, br), sv);
            let ui = vmulq_f32(vsubq_f32(a.1, bi), sv);
            let dr = vsubq_f32(a.0, br);
            let di = vaddq_f32(a.1, bi);
            let tr = vfmsq_f32(vmulq_f32(w.0, dr), w.1, di);
            let ti = vfmaq_f32(vmulq_f32(w.0, di), w.1, dr);
            vst2q_f32(
                dst.add(2 * k),
                float32x4x2_t(vaddq_f32(ur, tr), vaddq_f32(ui, ti)),
            );
            let (or, oi) = (vsubq_f32(ur, tr), vsubq_f32(ti, ui));
            vst2q_f32(dst.add(2 * (m - k - 3)), float32x4x2_t(rev(or), rev(oi)));
            k += 4;
        }
    }
    k
}

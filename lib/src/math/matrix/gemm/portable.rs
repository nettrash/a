//! The portable kernels of the products, of the shapes of the aarch64 ones, so that Miri runs the
//! blocks and the panels the aarch64 kernels see: arrays of sums, which the compiler keeps in
//! registers or vectors where it can.

use super::{Real, complex_tiles, tiles};
use crate::math::complex::Complex;

/// `c = beta · c + a · b` on an `MR × NR` tile, an array of sums the compiler keeps in
/// registers or vectors where it can.
///
/// # Safety
///
/// As for [`Real::kernel`].
#[inline(always)]
unsafe fn kernel<R: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const R,
    sa: usize,
    b: *const R,
    c: *mut R,
    ldc: usize,
    beta: R,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut acc = [[R::ZERO; MR]; NR];
        for p in 0..kc {
            let (a, b) = (a.add(p * sa), b.add(p * NR));
            for (j, col) in acc.iter_mut().enumerate() {
                let x = *b.add(j);
                for (i, s) in col.iter_mut().enumerate() {
                    *s = R::fma(*s, *a.add(i), x);
                }
            }
        }
        for (j, col) in acc.iter().enumerate() {
            for (i, s) in col.iter().enumerate() {
                let p = c.add(i + j * ldc);
                *p = if beta == R::ZERO {
                    *s
                } else {
                    R::fma(*s, beta, *p)
                };
            }
        }
    }
}

/// [`Real::pack_t`], a line at a time.
///
/// # Safety
///
/// As for [`Real::pack_t`].
#[inline(always)]
unsafe fn pack_t<R: Real>(
    dst: *mut R,
    src: *const R,
    ld: usize,
    h: usize,
    kc: usize,
    w: usize,
    scale: R,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        for r in 0..h {
            let s = src.add(r * ld);
            for p in 0..kc {
                *dst.add(p * w + r) = *s.add(p) * scale;
            }
        }
        for r in h..w {
            for p in 0..kc {
                *dst.add(p * w + r) = R::ZERO;
            }
        }
    }
}

/// [`Real::direct`]: four rows at a time, then in halves.
///
/// # Safety
///
/// As for [`Real::direct`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn direct<R: Real>(
    m: usize,
    n: usize,
    k: usize,
    alpha: R,
    a: *const R,
    lda: usize,
    b: *const R,
    rsb: usize,
    csb: usize,
    beta: R,
    c: *mut R,
    ldc: usize,
) {
    // SAFETY: as the caller guarantees; each group of rows is inside the block.
    unsafe {
        let mut i = 0;
        while i + 4 <= m {
            tiles::<R, 4>(n, k, a.add(i), lda, b, rsb, csb, c.add(i), ldc, alpha, beta);
            i += 4;
        }
        if i + 2 <= m {
            tiles::<R, 2>(n, k, a.add(i), lda, b, rsb, csb, c.add(i), ldc, alpha, beta);
            i += 2;
        }
        if i < m {
            tiles::<R, 1>(n, k, a.add(i), lda, b, rsb, csb, c.add(i), ldc, alpha, beta);
        }
    }
}

/// [`Real::direct_complex`]: two rows at a time, then the last.
///
/// # Safety
///
/// As for [`Real::direct_complex`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn direct_complex<R: Real, const CJ: bool>(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex<R>,
    a: *const Complex<R>,
    lda: usize,
    b: *const Complex<R>,
    rsb: usize,
    csb: usize,
    beta: Complex<R>,
    c: *mut Complex<R>,
    ldc: usize,
) {
    // SAFETY: as the caller guarantees; each group of rows is inside the block.
    unsafe {
        let mut i = 0;
        while i + 2 <= m {
            complex_tiles::<R, 2, CJ>(n, k, a.add(i), lda, b, rsb, csb, c.add(i), ldc, alpha, beta);
            i += 2;
        }
        if i < m {
            complex_tiles::<R, 1, CJ>(n, k, a.add(i), lda, b, rsb, csb, c.add(i), ldc, alpha, beta);
        }
    }
}

impl Real for f64 {
    const MR: usize = 8;
    const NR: usize = 6;
    const KC: usize = 512;
    const MC: usize = 128;
    const NC: usize = 1020;
    const SKINNY: usize = 32;

    #[inline(always)]
    fn width(r: usize) -> usize {
        if r >= 9 || r == 6 || r == 5 {
            6
        } else if r >= 3 {
            4
        } else {
            2
        }
    }

    #[inline(never)]
    unsafe fn kernel(
        nr: usize,
        kc: usize,
        a: *const f64,
        sa: usize,
        b: *const f64,
        c: *mut f64,
        ldc: usize,
        beta: f64,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe {
            match nr {
                6 => kernel::<f64, 8, 6>(kc, a, sa, b, c, ldc, beta),
                4 => kernel::<f64, 8, 4>(kc, a, sa, b, c, ldc, beta),
                _ => kernel::<f64, 8, 2>(kc, a, sa, b, c, ldc, beta),
            }
        }
    }

    #[inline(always)]
    fn fma(c: f64, a: f64, b: f64) -> f64 {
        c + a * b
    }

    #[inline(never)]
    unsafe fn pack_t(
        dst: *mut f64,
        src: *const f64,
        ld: usize,
        h: usize,
        kc: usize,
        w: usize,
        scale: f64,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe { pack_t(dst, src, ld, h, kc, w, scale) }
    }

    #[inline(never)]
    unsafe fn direct(
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        rsb: usize,
        csb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe { direct(m, n, k, alpha, a, lda, b, rsb, csb, beta, c, ldc) }
    }

    #[inline(never)]
    unsafe fn direct_complex<const CJ: bool>(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: usize,
        b: *const Complex<f64>,
        rsb: usize,
        csb: usize,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: usize,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe { direct_complex::<f64, CJ>(m, n, k, alpha, a, lda, b, rsb, csb, beta, c, ldc) }
    }
}

impl Real for f32 {
    const MR: usize = 8;
    const NR: usize = 12;
    const KC: usize = 512;
    const MC: usize = 128;
    const NC: usize = 2040;
    const SKINNY: usize = 32;

    #[inline(always)]
    fn width(r: usize) -> usize {
        if r >= 17 || (9..=12).contains(&r) {
            12
        } else if r >= 5 {
            8
        } else {
            4
        }
    }

    #[inline(never)]
    unsafe fn kernel(
        nr: usize,
        kc: usize,
        a: *const f32,
        sa: usize,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        beta: f32,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe {
            match nr {
                12 => kernel::<f32, 8, 12>(kc, a, sa, b, c, ldc, beta),
                8 => kernel::<f32, 8, 8>(kc, a, sa, b, c, ldc, beta),
                _ => kernel::<f32, 8, 4>(kc, a, sa, b, c, ldc, beta),
            }
        }
    }

    #[inline(always)]
    fn fma(c: f32, a: f32, b: f32) -> f32 {
        c + a * b
    }

    #[inline(never)]
    unsafe fn pack_t(
        dst: *mut f32,
        src: *const f32,
        ld: usize,
        h: usize,
        kc: usize,
        w: usize,
        scale: f32,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe { pack_t(dst, src, ld, h, kc, w, scale) }
    }

    #[inline(never)]
    unsafe fn direct(
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        rsb: usize,
        csb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe { direct(m, n, k, alpha, a, lda, b, rsb, csb, beta, c, ldc) }
    }

    #[inline(never)]
    unsafe fn direct_complex<const CJ: bool>(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: usize,
        b: *const Complex<f32>,
        rsb: usize,
        csb: usize,
        beta: Complex<f32>,
        c: *mut Complex<f32>,
        ldc: usize,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe { direct_complex::<f32, CJ>(m, n, k, alpha, a, lda, b, rsb, csb, beta, c, ldc) }
    }
}

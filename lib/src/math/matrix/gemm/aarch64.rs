//! The aarch64 kernels of the products: NEON intrinsics, fused multiply-adds by element.
//!
//! The micro-kernels keep their sums in arrays of vectors that the compiler holds in registers:
//! 24 of the 32 for the widest tiles, 4 for the column of `a` and 3 for the row of `b`, with
//! nothing spilled, a step of 8 × 6 `f64` being 4 loads (`ldp`) and 24 `fmla` by element. Measured
//! on an Apple M4 Max, a tile's steps ran at 95 to 100 % of four multiply-adds a cycle with its
//! panels in the L1 cache, unrolled two or four times or not; the kernels are not unrolled. A
//! prefetch of the tile of `c` when the kernel starts made no difference.
//!
//! The direct tiles of small products are written with intrinsics too. The compiler's own
//! vectorization of the portable tiles reached 125 GFLOP/s of `f32` in a first test, but spilled
//! sums and shuffled lanes once inside the crate (76 GFLOP/s at 48 × 48 × 48, against 112 with
//! intrinsics), and left the complex tiles scalar: products of `Complex<f64>` of 8 × 8 × 8 at 23
//! GFLOP/s against 48, and of 64 × 64 × 64 at 27 against 61.

use super::{Real, complex_tiles, tiles};
use crate::math::complex::Complex;
use core::arch::aarch64::*;

/// Fills `$acc` with `a · b` over `$kc` steps: `$a` and `$b` are loaded a step at a time, and each
/// column's lane of `b` multiplies all of `a`'s vectors.
macro_rules! steps {
    ($kc:expr, $a:ident, $sa:ident, $b:ident, $acc:ident, $ld:ident, $fma:ident, $lanes:literal, $mv:literal, $nv:literal, [$($l:literal),*]) => {
        for _ in 0..$kc {
            let av: [_; $mv] = core::array::from_fn(|i| $ld($a.add(i * $lanes)));
            let bv: [_; $nv] = core::array::from_fn(|q| $ld($b.add(q * $lanes)));
            for q in 0..$nv {
                for i in 0..$mv {
                    $($acc[$lanes * q + $l][i] = $fma::<$l>($acc[$lanes * q + $l][i], av[i], bv[q]);)*
                }
            }
            // Past the block after the last step, where `a` is read in place.
            $a = $a.wrapping_add($sa);
            $b = $b.wrapping_add($lanes * $nv);
        }
    };
}

/// Writes `$acc` to the tile at `$c`: `c = acc`, `c + acc` or `beta · c + acc`.
macro_rules! store {
    ($c:ident, $ldc:ident, $beta:ident, $acc:ident, $ld:ident, $st:ident, $add:ident, $fman:ident, $lanes:literal) => {
        if $beta == 0.0 {
            for (j, col) in $acc.iter().enumerate() {
                for (i, v) in col.iter().enumerate() {
                    $st($c.add(i * $lanes + j * $ldc), *v);
                }
            }
        } else if $beta == 1.0 {
            for (j, col) in $acc.iter().enumerate() {
                for (i, v) in col.iter().enumerate() {
                    let p = $c.add(i * $lanes + j * $ldc);
                    $st(p, $add($ld(p), *v));
                }
            }
        } else {
            for (j, col) in $acc.iter().enumerate() {
                for (i, v) in col.iter().enumerate() {
                    let p = $c.add(i * $lanes + j * $ldc);
                    $st(p, $fman(*v, $ld(p), $beta));
                }
            }
        }
    };
}

/// The `f64` micro-kernels: tiles of 8 rows, four vectors, by `$nv` pairs of columns, lanes of
/// the vectors of `b`.
macro_rules! kernel_f64 {
    ($name:ident, $nv:literal) => {
        /// An `f64` micro-kernel: see [`kernel_f64`].
        ///
        /// # Safety
        ///
        /// As for [`Real::kernel`].
        #[inline(never)]
        unsafe fn $name(
            kc: usize,
            mut a: *const f64,
            sa: usize,
            mut b: *const f64,
            c: *mut f64,
            ldc: usize,
            beta: f64,
        ) {
            // SAFETY: as the caller guarantees.
            unsafe {
                let mut acc = [[vdupq_n_f64(0.0); 4]; 2 * $nv];
                steps!(
                    kc,
                    a,
                    sa,
                    b,
                    acc,
                    vld1q_f64,
                    vfmaq_laneq_f64,
                    2,
                    4,
                    $nv,
                    [0, 1]
                );
                store!(
                    c,
                    ldc,
                    beta,
                    acc,
                    vld1q_f64,
                    vst1q_f64,
                    vaddq_f64,
                    vfmaq_n_f64,
                    2
                );
            }
        }
    };
}

kernel_f64!(kernel_f64_6, 3);
kernel_f64!(kernel_f64_4, 2);
kernel_f64!(kernel_f64_2, 1);

/// The `f32` micro-kernels: tiles of 8 rows, two vectors, by `$nv` quads of columns, lanes of the
/// vectors of `b`.
macro_rules! kernel_f32 {
    ($name:ident, $nv:literal) => {
        /// An `f32` micro-kernel: see [`kernel_f32`].
        ///
        /// # Safety
        ///
        /// As for [`Real::kernel`].
        #[inline(never)]
        unsafe fn $name(
            kc: usize,
            mut a: *const f32,
            sa: usize,
            mut b: *const f32,
            c: *mut f32,
            ldc: usize,
            beta: f32,
        ) {
            // SAFETY: as the caller guarantees.
            unsafe {
                let mut acc = [[vdupq_n_f32(0.0); 2]; 4 * $nv];
                steps!(
                    kc,
                    a,
                    sa,
                    b,
                    acc,
                    vld1q_f32,
                    vfmaq_laneq_f32,
                    4,
                    2,
                    $nv,
                    [0, 1, 2, 3]
                );
                store!(
                    c,
                    ldc,
                    beta,
                    acc,
                    vld1q_f32,
                    vst1q_f32,
                    vaddq_f32,
                    vfmaq_n_f32,
                    4
                );
            }
        }
    };
}

kernel_f32!(kernel_f32_12, 3);
kernel_f32!(kernel_f32_8, 2);
kernel_f32!(kernel_f32_4, 1);

/// [`Real::pack_t`] of `f64`: two lines by two steps at a time, a 2 × 2 transpose by `zip`.
///
/// # Safety
///
/// As for [`Real::pack_t`].
#[inline(never)]
unsafe fn pack_t_f64(
    dst: *mut f64,
    src: *const f64,
    ld: usize,
    h: usize,
    kc: usize,
    w: usize,
    scale: f64,
) {
    // SAFETY: every read is of an element of the lines, and every write inside the panel.
    unsafe {
        let s = vdupq_n_f64(scale);
        let mut p = 0;
        while p + 2 <= kc {
            let (d0, d1) = (dst.add(p * w), dst.add(p * w + w));
            let mut r = 0;
            while r + 2 <= h {
                let x = vmulq_f64(vld1q_f64(src.add(r * ld + p)), s);
                let y = vmulq_f64(vld1q_f64(src.add(r * ld + ld + p)), s);
                vst1q_f64(d0.add(r), vzip1q_f64(x, y));
                vst1q_f64(d1.add(r), vzip2q_f64(x, y));
                r += 2;
            }
            if r < h {
                *d0.add(r) = *src.add(r * ld + p) * scale;
                *d1.add(r) = *src.add(r * ld + p + 1) * scale;
                r += 1;
            }
            for r in r..w {
                *d0.add(r) = 0.0;
                *d1.add(r) = 0.0;
            }
            p += 2;
        }
        if p < kc {
            let d = dst.add(p * w);
            for r in 0..w {
                *d.add(r) = if r < h {
                    *src.add(r * ld + p) * scale
                } else {
                    0.0
                };
            }
        }
    }
}

/// [`Real::pack_t`] of `f32`: four lines by four steps at a time, a 4 × 4 transpose by `trn`.
///
/// # Safety
///
/// As for [`Real::pack_t`].
#[inline(never)]
unsafe fn pack_t_f32(
    dst: *mut f32,
    src: *const f32,
    ld: usize,
    h: usize,
    kc: usize,
    w: usize,
    scale: f32,
) {
    // SAFETY: every read is of an element of the lines, and every write inside the panel.
    unsafe {
        let s = vdupq_n_f32(scale);
        let mut p = 0;
        while p + 4 <= kc {
            let d = dst.add(p * w);
            let mut r = 0;
            while r + 4 <= h {
                let x0 = vmulq_f32(vld1q_f32(src.add(r * ld + p)), s);
                let x1 = vmulq_f32(vld1q_f32(src.add((r + 1) * ld + p)), s);
                let x2 = vmulq_f32(vld1q_f32(src.add((r + 2) * ld + p)), s);
                let x3 = vmulq_f32(vld1q_f32(src.add((r + 3) * ld + p)), s);
                let t0 = vreinterpretq_f64_f32(vtrn1q_f32(x0, x1));
                let t1 = vreinterpretq_f64_f32(vtrn2q_f32(x0, x1));
                let t2 = vreinterpretq_f64_f32(vtrn1q_f32(x2, x3));
                let t3 = vreinterpretq_f64_f32(vtrn2q_f32(x2, x3));
                vst1q_f32(d.add(r), vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)));
                vst1q_f32(d.add(w + r), vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)));
                vst1q_f32(d.add(2 * w + r), vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)));
                vst1q_f32(d.add(3 * w + r), vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)));
                r += 4;
            }
            for r in r..w {
                for q in 0..4 {
                    *d.add(q * w + r) = if r < h {
                        *src.add(r * ld + p + q) * scale
                    } else {
                        0.0
                    };
                }
            }
            p += 4;
        }
        for p in p..kc {
            let d = dst.add(p * w);
            for r in 0..w {
                *d.add(r) = if r < h {
                    *src.add(r * ld + p) * scale
                } else {
                    0.0
                };
            }
        }
    }
}

/// A tile of [`Real::direct`] of `$t`, `$lanes` lanes a vector: `MV` vectors of rows by `NR`
/// columns, each element of `op(b)` multiplying the column of `a` by element.
macro_rules! real_tile {
    ($name:ident, $t:ident, $v:ident, $lanes:literal, $ld:ident, $st:ident, $dup:ident, $fman:ident, $muln:ident) => {
        /// A tile of [`Real::direct`]: see [`real_tile`].
        ///
        /// # Safety
        ///
        /// As for [`Real::direct`], for the tile.
        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name<const MV: usize, const NR: usize>(
            k: usize,
            a: *const $t,
            lda: usize,
            b: *const $t,
            rsb: usize,
            csb: usize,
            c: *mut $t,
            ldc: usize,
            alpha: $t,
            beta: $t,
        ) {
            // SAFETY: as the caller guarantees.
            unsafe {
                let mut acc = [[$dup(0.0); MV]; NR];
                for p in 0..k {
                    let x: [$v; MV] = core::array::from_fn(|v| $ld(a.add(v * $lanes + p * lda)));
                    for (j, sums) in acc.iter_mut().enumerate() {
                        let y = *b.add(p * rsb + j * csb);
                        for (s, x) in sums.iter_mut().zip(&x) {
                            *s = $fman(*s, *x, y);
                        }
                    }
                }
                for (j, sums) in acc.iter().enumerate() {
                    for (v, s) in sums.iter().enumerate() {
                        let d = c.add(v * $lanes + j * ldc);
                        let y = $muln(*s, alpha);
                        $st(
                            d,
                            if beta == 0.0 {
                                y
                            } else {
                                $fman(y, $ld(d), beta)
                            },
                        );
                    }
                }
            }
        }
    };
}

real_tile!(
    tile_f64,
    f64,
    float64x2_t,
    2,
    vld1q_f64,
    vst1q_f64,
    vdupq_n_f64,
    vfmaq_n_f64,
    vmulq_n_f64
);
real_tile!(
    tile_f32,
    f32,
    float32x4_t,
    4,
    vld1q_f32,
    vst1q_f32,
    vdupq_n_f32,
    vfmaq_n_f32,
    vmulq_n_f32
);

/// Runs `$tile::<MV, NR>` over the columns of a group of rows: `$nr` at a time (8 for groups of
/// one or two vectors, so that a tile has 16 sums, as 8 wait on each other: `f32` products of
/// 24 × 1024 · 1024 × 1024 ran at 88 GFLOP/s with tiles of 8 × 4 for the last 8 rows), then 4,
/// then the rest.
macro_rules! real_columns {
    ($tile:ident, $mv:literal, $nr:literal, $n:ident, $k:ident, $a:expr, $lda:ident, $b:ident, $rsb:ident, $csb:ident, $c:expr, $ldc:ident, $alpha:ident, $beta:ident) => {{
        let (a, c) = ($a, $c);
        let mut j = 0;
        while j + $nr <= $n {
            let (b, c) = ($b.add(j * $csb), c.add(j * $ldc));
            $tile::<$mv, $nr>($k, a, $lda, b, $rsb, $csb, c, $ldc, $alpha, $beta);
            j += $nr;
        }
        if $nr > 4 && j + 4 <= $n {
            let (b, c) = ($b.add(j * $csb), c.add(j * $ldc));
            $tile::<$mv, 4>($k, a, $lda, b, $rsb, $csb, c, $ldc, $alpha, $beta);
            j += 4;
        }
        // Past the end when no columns are left, so not dereferenced then.
        let (b, c) = ($b.wrapping_add(j * $csb), c.wrapping_add(j * $ldc));
        match $n - j {
            3 => $tile::<$mv, 3>($k, a, $lda, b, $rsb, $csb, c, $ldc, $alpha, $beta),
            2 => $tile::<$mv, 2>($k, a, $lda, b, $rsb, $csb, c, $ldc, $alpha, $beta),
            1 => $tile::<$mv, 1>($k, a, $lda, b, $rsb, $csb, c, $ldc, $alpha, $beta),
            _ => {}
        }
    }};
}

/// A tile of [`Real::direct_complex`] of `f64`: `MV` pairs of rows by `NR` columns, the real and
/// the imaginary parts apart (`ld2`), each part of `b` a lane for the fused multiply-adds.
///
/// # Safety
///
/// As for [`Real::direct_complex`], for the tile.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn complex_tile_f64<const MV: usize, const NR: usize, const CJ: bool>(
    k: usize,
    a: *const Complex<f64>,
    lda: usize,
    b: *const Complex<f64>,
    rsb: usize,
    csb: usize,
    c: *mut Complex<f64>,
    ldc: usize,
    alpha: Complex<f64>,
    beta: Complex<f64>,
) {
    // SAFETY: as the caller guarantees; `Complex` is `repr(C)`, its parts in turn.
    unsafe {
        let z = vdupq_n_f64(0.0);
        let mut re = [[z; MV]; NR];
        let mut im = [[z; MV]; NR];
        let (a, b) = (a.cast::<f64>(), b.cast::<f64>());
        for p in 0..k {
            let x: [float64x2x2_t; MV] =
                core::array::from_fn(|v| vld2q_f64(a.add(2 * (2 * v + p * lda))));
            for j in 0..NR {
                let y = vld1q_f64(b.add(2 * (p * rsb + j * csb)));
                for v in 0..MV {
                    re[j][v] = vfmaq_laneq_f64::<0>(re[j][v], x[v].0, y);
                    im[j][v] = vfmaq_laneq_f64::<0>(im[j][v], x[v].1, y);
                    if CJ {
                        re[j][v] = vfmaq_laneq_f64::<1>(re[j][v], x[v].1, y);
                        im[j][v] = vfmsq_laneq_f64::<1>(im[j][v], x[v].0, y);
                    } else {
                        re[j][v] = vfmsq_laneq_f64::<1>(re[j][v], x[v].1, y);
                        im[j][v] = vfmaq_laneq_f64::<1>(im[j][v], x[v].0, y);
                    }
                }
            }
        }
        let c = c.cast::<f64>();
        let zero = beta.re == 0.0 && beta.im == 0.0;
        for j in 0..NR {
            for v in 0..MV {
                let d = c.add(2 * (2 * v + j * ldc));
                let (sr, si) = (re[j][v], im[j][v]);
                let mut yr = vfmsq_n_f64(vmulq_n_f64(sr, alpha.re), si, alpha.im);
                let mut yi = vfmaq_n_f64(vmulq_n_f64(si, alpha.re), sr, alpha.im);
                if !zero {
                    let o = vld2q_f64(d);
                    yr = vfmsq_n_f64(vfmaq_n_f64(yr, o.0, beta.re), o.1, beta.im);
                    yi = vfmaq_n_f64(vfmaq_n_f64(yi, o.1, beta.re), o.0, beta.im);
                }
                vst2q_f64(d, float64x2x2_t(yr, yi));
            }
        }
    }
}

/// A tile of [`Real::direct_complex`] of `f32`: `MV` quads of rows by `NR` columns, as
/// [`complex_tile_f64`] does it.
///
/// # Safety
///
/// As for [`Real::direct_complex`], for the tile.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn complex_tile_f32<const MV: usize, const NR: usize, const CJ: bool>(
    k: usize,
    a: *const Complex<f32>,
    lda: usize,
    b: *const Complex<f32>,
    rsb: usize,
    csb: usize,
    c: *mut Complex<f32>,
    ldc: usize,
    alpha: Complex<f32>,
    beta: Complex<f32>,
) {
    // SAFETY: as the caller guarantees; `Complex` is `repr(C)`, its parts in turn.
    unsafe {
        let z = vdupq_n_f32(0.0);
        let mut re = [[z; MV]; NR];
        let mut im = [[z; MV]; NR];
        let (a, b) = (a.cast::<f32>(), b.cast::<f32>());
        for p in 0..k {
            let x: [float32x4x2_t; MV] =
                core::array::from_fn(|v| vld2q_f32(a.add(2 * (4 * v + p * lda))));
            for j in 0..NR {
                let y = vld1_f32(b.add(2 * (p * rsb + j * csb)));
                for v in 0..MV {
                    re[j][v] = vfmaq_lane_f32::<0>(re[j][v], x[v].0, y);
                    im[j][v] = vfmaq_lane_f32::<0>(im[j][v], x[v].1, y);
                    if CJ {
                        re[j][v] = vfmaq_lane_f32::<1>(re[j][v], x[v].1, y);
                        im[j][v] = vfmsq_lane_f32::<1>(im[j][v], x[v].0, y);
                    } else {
                        re[j][v] = vfmsq_lane_f32::<1>(re[j][v], x[v].1, y);
                        im[j][v] = vfmaq_lane_f32::<1>(im[j][v], x[v].0, y);
                    }
                }
            }
        }
        let c = c.cast::<f32>();
        let zero = beta.re == 0.0 && beta.im == 0.0;
        for j in 0..NR {
            for v in 0..MV {
                let d = c.add(2 * (4 * v + j * ldc));
                let (sr, si) = (re[j][v], im[j][v]);
                let mut yr = vfmsq_n_f32(vmulq_n_f32(sr, alpha.re), si, alpha.im);
                let mut yi = vfmaq_n_f32(vmulq_n_f32(si, alpha.re), sr, alpha.im);
                if !zero {
                    let o = vld2q_f32(d);
                    yr = vfmsq_n_f32(vfmaq_n_f32(yr, o.0, beta.re), o.1, beta.im);
                    yi = vfmaq_n_f32(vfmaq_n_f32(yi, o.1, beta.re), o.0, beta.im);
                }
                vst2q_f32(d, float32x4x2_t(yr, yi));
            }
        }
    }
}

/// A tile of [`Real::direct_complex`] of `f32` of two rows by `NR` columns, as
/// [`complex_tile_f32`] does it with vectors of two lanes, for the rows left after the groups of
/// four: products of 6 × 6 × 6 ran at 41 GFLOP/s with the portable tile for the last two rows.
///
/// # Safety
///
/// As for [`Real::direct_complex`], for the tile.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn complex_tile_f32_2<const MV: usize, const NR: usize, const CJ: bool>(
    k: usize,
    a: *const Complex<f32>,
    lda: usize,
    b: *const Complex<f32>,
    rsb: usize,
    csb: usize,
    c: *mut Complex<f32>,
    ldc: usize,
    alpha: Complex<f32>,
    beta: Complex<f32>,
) {
    // SAFETY: as the caller guarantees; `Complex` is `repr(C)`, its parts in turn.
    unsafe {
        let z = vdup_n_f32(0.0);
        let mut re = [z; NR];
        let mut im = [z; NR];
        let (a, b) = (a.cast::<f32>(), b.cast::<f32>());
        for p in 0..k {
            let x = vld2_f32(a.add(2 * p * lda));
            for j in 0..NR {
                let y = vld1_f32(b.add(2 * (p * rsb + j * csb)));
                re[j] = vfma_lane_f32::<0>(re[j], x.0, y);
                im[j] = vfma_lane_f32::<0>(im[j], x.1, y);
                if CJ {
                    re[j] = vfma_lane_f32::<1>(re[j], x.1, y);
                    im[j] = vfms_lane_f32::<1>(im[j], x.0, y);
                } else {
                    re[j] = vfms_lane_f32::<1>(re[j], x.1, y);
                    im[j] = vfma_lane_f32::<1>(im[j], x.0, y);
                }
            }
        }
        let c = c.cast::<f32>();
        let zero = beta.re == 0.0 && beta.im == 0.0;
        for j in 0..NR {
            let d = c.add(2 * j * ldc);
            let (sr, si) = (re[j], im[j]);
            let mut yr = vfms_n_f32(vmul_n_f32(sr, alpha.re), si, alpha.im);
            let mut yi = vfma_n_f32(vmul_n_f32(si, alpha.re), sr, alpha.im);
            if !zero {
                let o = vld2_f32(d);
                yr = vfms_n_f32(vfma_n_f32(yr, o.0, beta.re), o.1, beta.im);
                yi = vfma_n_f32(vfma_n_f32(yi, o.1, beta.re), o.0, beta.im);
            }
            vst2_f32(d, float32x2x2_t(yr, yi));
        }
    }
}

/// Runs `$tile::<MV, NR, CJ>` over the columns of a group of rows: four at a time, then the rest.
macro_rules! complex_columns {
    ($tile:ident, $mv:literal, $cj:ident, $n:ident, $k:ident, $a:expr, $lda:ident, $b:ident, $rsb:ident, $csb:ident, $c:expr, $ldc:ident, $alpha:ident, $beta:ident) => {{
        let (a, c) = ($a, $c);
        let mut j = 0;
        while j + 4 <= $n {
            $tile::<$mv, 4, $cj>(
                $k,
                a,
                $lda,
                $b.add(j * $csb),
                $rsb,
                $csb,
                c.add(j * $ldc),
                $ldc,
                $alpha,
                $beta,
            );
            j += 4;
        }
        // Past the end when no columns are left, so not dereferenced then.
        let (b, c) = ($b.wrapping_add(j * $csb), c.wrapping_add(j * $ldc));
        match $n - j {
            3 => $tile::<$mv, 3, $cj>($k, a, $lda, b, $rsb, $csb, c, $ldc, $alpha, $beta),
            2 => $tile::<$mv, 2, $cj>($k, a, $lda, b, $rsb, $csb, c, $ldc, $alpha, $beta),
            1 => $tile::<$mv, 1, $cj>($k, a, $lda, b, $rsb, $csb, c, $ldc, $alpha, $beta),
            _ => {}
        }
    }};
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

    #[inline(always)]
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
                6 => kernel_f64_6(kc, a, sa, b, c, ldc, beta),
                4 => kernel_f64_4(kc, a, sa, b, c, ldc, beta),
                _ => kernel_f64_2(kc, a, sa, b, c, ldc, beta),
            }
        }
    }

    #[inline(always)]
    fn fma(c: f64, a: f64, b: f64) -> f64 {
        a.mul_add(b, c)
    }

    #[inline(always)]
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
        unsafe { pack_t_f64(dst, src, ld, h, kc, w, scale) }
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
        // SAFETY: as the caller guarantees; each group of rows is inside the block.
        unsafe {
            let mut i = 0;
            while i + 8 <= m {
                real_columns!(
                    tile_f64,
                    4,
                    4,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 8;
            }
            if i + 4 <= m {
                real_columns!(
                    tile_f64,
                    2,
                    8,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 4;
            }
            if i + 2 <= m {
                real_columns!(
                    tile_f64,
                    1,
                    8,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 2;
            }
            if i < m {
                tiles::<f64, 1>(n, k, a.add(i), lda, b, rsb, csb, c.add(i), ldc, alpha, beta);
            }
        }
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
        // SAFETY: as the caller guarantees; each group of rows is inside the block.
        unsafe {
            let mut i = 0;
            while i + 4 <= m {
                complex_columns!(
                    complex_tile_f64,
                    2,
                    CJ,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 4;
            }
            if i + 2 <= m {
                complex_columns!(
                    complex_tile_f64,
                    1,
                    CJ,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 2;
            }
            if i < m {
                complex_tiles::<f64, 1, CJ>(
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta,
                );
            }
        }
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

    #[inline(always)]
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
                12 => kernel_f32_12(kc, a, sa, b, c, ldc, beta),
                8 => kernel_f32_8(kc, a, sa, b, c, ldc, beta),
                _ => kernel_f32_4(kc, a, sa, b, c, ldc, beta),
            }
        }
    }

    #[inline(always)]
    fn fma(c: f32, a: f32, b: f32) -> f32 {
        a.mul_add(b, c)
    }

    #[inline(always)]
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
        unsafe { pack_t_f32(dst, src, ld, h, kc, w, scale) }
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
        // SAFETY: as the caller guarantees; each group of rows is inside the block.
        unsafe {
            let mut i = 0;
            while i + 16 <= m {
                real_columns!(
                    tile_f32,
                    4,
                    4,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 16;
            }
            if i + 8 <= m {
                real_columns!(
                    tile_f32,
                    2,
                    8,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 8;
            }
            if i + 4 <= m {
                real_columns!(
                    tile_f32,
                    1,
                    8,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 4;
            }
            if i + 2 <= m {
                tiles::<f32, 2>(n, k, a.add(i), lda, b, rsb, csb, c.add(i), ldc, alpha, beta);
                i += 2;
            }
            if i < m {
                tiles::<f32, 1>(n, k, a.add(i), lda, b, rsb, csb, c.add(i), ldc, alpha, beta);
            }
        }
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
        // SAFETY: as the caller guarantees; each group of rows is inside the block.
        unsafe {
            let mut i = 0;
            while i + 8 <= m {
                complex_columns!(
                    complex_tile_f32,
                    2,
                    CJ,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 8;
            }
            if i + 4 <= m {
                complex_columns!(
                    complex_tile_f32,
                    1,
                    CJ,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 4;
            }
            if i + 2 <= m {
                complex_columns!(
                    complex_tile_f32_2,
                    1,
                    CJ,
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta
                );
                i += 2;
            }
            if i < m {
                complex_tiles::<f32, 1, CJ>(
                    n,
                    k,
                    a.add(i),
                    lda,
                    b,
                    rsb,
                    csb,
                    c.add(i),
                    ldc,
                    alpha,
                    beta,
                );
            }
        }
    }
}

//! The aarch64 lanes of the transforms: NEON vectors of four `i32`s, and the Chinese remainder
//! theorem of four coefficients by widening multiply-adds.
//!
//! # Products by constants
//!
//! A product by a twiddle is `sqrdmulh`, `mul` and `mls`, written in assembly: from the
//! intrinsics, the compiler emitted `mul`, `mul` and `sub`, or `mul`, `neg` and `mla`, four
//! instructions where three do. Measured on an Apple M4 Max, a pass of two levels took 0.370
//! cycles a butterfly against 0.388 with the intrinsics, for transforms of `2^12` to `2^16`
//! points.
//!
//! # Speed
//!
//! An Apple M4 core issues four NEON operations a cycle. Two levels of butterflies on four
//! vectors take 4 products by twiddles (12 operations), 8 additions and subtractions and one
//! reduction (3), 23 operations for 16 butterflies, 0.36 cycles a butterfly; measured on an Apple
//! M4 Max, 0.37 for transforms of `2^10` to `2^20` points, as the arrays stream from the caches
//! and memory as fast. The last four levels, with the transposes and the loads of a twiddle a
//! lane, take 0.53, the inverse's 0.42 and 0.58, and the pointwise products 0.33 cycles a value.
//! Against the other candidates measured on the same core, in cycles a butterfly a bit of the
//! prime, as a transform's cost for a product is proportional to the bits of the primes it
//! needs: 0.0134 here (primes of 28.8 bits), 0.0250 for FLINT's `fft_small` way (primes below
//! `2^50`, residues in `f64` lanes with fused multiply-adds, 1.25 cycles a butterfly, as a
//! vector holds two lanes and a product by a constant takes six operations), and 0.0324 for
//! three 62-bit primes by scalar Shoup products (2.0 cycles a butterfly).

use super::{Crt, Lanes, Prime};
use core::arch::aarch64::*;
use core::arch::asm;

/// From this many limbs in the shorter operand, a product is computed by the transforms.
pub(in super::super) const FFT_MUL: usize = 360;

/// From this many limbs in the shorter operand, a product whose longer operand has at least
/// twice as many is computed by the transforms.
pub(in super::super) const FFT_MUL_UNBALANCED: usize = 180;

/// From this many limbs, a square is computed by the transforms.
pub(in super::super) const FFT_SQR: usize = 380;

/// The lanes of the transforms on this target.
pub(super) type V = Neon;

/// A NEON vector of four residues.
#[derive(Clone, Copy)]
pub(super) struct Neon(int32x4_t);

impl Lanes for Neon {
    /// `[q, v, qinv, 0]`, for the lane forms of the instructions.
    type C = int32x4_t;

    #[inline(always)]
    fn consts(p: &Prime) -> int32x4_t {
        let c = [p.q, p.v, p.qinv, 0];
        // SAFETY: the 4 `i32`s of `c` are valid for reads.
        unsafe { vld1q_s32(c.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load(p: *const i32) -> Self {
        // SAFETY: as the caller guarantees.
        unsafe { Neon(vld1q_s32(p)) }
    }

    #[inline(always)]
    unsafe fn store(self, p: *mut i32) {
        // SAFETY: as the caller guarantees.
        unsafe { vst1q_s32(p, self.0) }
    }

    #[inline(always)]
    unsafe fn load2(p: *const i32) -> (Self, Self) {
        // SAFETY: as the caller guarantees.
        let x = unsafe { vld2q_s32(p) };
        (Neon(x.0), Neon(x.1))
    }

    #[inline(always)]
    unsafe fn load4(p: *const i32) -> (Self, Self, Self, Self) {
        // SAFETY: as the caller guarantees.
        let x = unsafe { vld4q_s32(p) };
        (Neon(x.0), Neon(x.1), Neon(x.2), Neon(x.3))
    }

    #[inline(always)]
    unsafe fn store2(a: Self, b: Self, p: *mut i32) {
        // SAFETY: as the caller guarantees.
        unsafe { vst2q_s32(p, int32x4x2_t(a.0, b.0)) }
    }

    #[inline(always)]
    unsafe fn pair(p: *const i32) -> Self {
        // SAFETY: as the caller guarantees.
        unsafe { Neon(vcombine_s32(vld1_s32(p), vdup_n_s32(0))) }
    }

    #[inline(always)]
    fn splat(x: i32) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe { Neon(vdupq_n_s32(x)) }
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe { Neon(vaddq_s32(self.0, o.0)) }
    }

    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe { Neon(vsubq_s32(self.0, o.0)) }
    }

    #[inline(always)]
    fn mul(self, w: Self, wp: Self, c: int32x4_t) -> Self {
        let r: int32x4_t;
        // SAFETY: arithmetic on registers only.
        unsafe {
            asm!(
                "sqrdmulh {t:v}.4s, {a:v}.4s, {wp:v}.4s",
                "mul {r:v}.4s, {a:v}.4s, {w:v}.4s",
                "mls {r:v}.4s, {t:v}.4s, {c:v}.s[0]",
                a = in(vreg) self.0,
                w = in(vreg) w.0,
                wp = in(vreg) wp.0,
                c = in(vreg_low16) c,
                t = out(vreg) _,
                r = out(vreg) r,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        Neon(r)
    }

    #[inline(always)]
    fn mul_lanes<const LW: i32, const LP: i32>(self, v: Self, c: int32x4_t) -> Self {
        let r: int32x4_t;
        // SAFETY: arithmetic on registers only; the lanes are below 4.
        unsafe {
            asm!(
                "sqrdmulh {t:v}.4s, {a:v}.4s, {v:v}.s[{lp}]",
                "mul {r:v}.4s, {a:v}.4s, {v:v}.s[{lw}]",
                "mls {r:v}.4s, {t:v}.4s, {c:v}.s[0]",
                a = in(vreg) self.0,
                v = in(vreg_low16) v.0,
                c = in(vreg_low16) c,
                lw = const LW,
                lp = const LP,
                t = out(vreg) _,
                r = out(vreg) r,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        Neon(r)
    }

    #[inline(always)]
    fn reduce(self, c: int32x4_t) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe {
            let t = vrshrq_n_s32::<28>(vqdmulhq_laneq_s32::<1>(self.0, c));
            Neon(vmlsq_laneq_s32::<0>(self.0, t, c))
        }
    }

    #[inline(always)]
    fn mont(self, o: Self, c: int32x4_t) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe {
            let lo = vmulq_s32(self.0, o.0);
            let hi = vqdmulhq_s32(self.0, o.0);
            let m = vmulq_laneq_s32::<2>(lo, c);
            Neon(vhsubq_s32(hi, vqdmulhq_laneq_s32::<0>(m, c)))
        }
    }

    #[inline(always)]
    fn transpose(v: [Self; 4]) -> [Self; 4] {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe {
            let a0 = vreinterpretq_s64_s32(vtrn1q_s32(v[0].0, v[1].0));
            let a1 = vreinterpretq_s64_s32(vtrn2q_s32(v[0].0, v[1].0));
            let a2 = vreinterpretq_s64_s32(vtrn1q_s32(v[2].0, v[3].0));
            let a3 = vreinterpretq_s64_s32(vtrn2q_s32(v[2].0, v[3].0));
            [
                Neon(vreinterpretq_s32_s64(vtrn1q_s64(a0, a2))),
                Neon(vreinterpretq_s32_s64(vtrn1q_s64(a1, a3))),
                Neon(vreinterpretq_s32_s64(vtrn2q_s64(a0, a2))),
                Neon(vreinterpretq_s32_s64(vtrn2q_s64(a1, a3))),
            ]
        }
    }

    #[inline(always)]
    fn normalize(self, c: int32x4_t) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe {
            let negative = vshrq_n_s32::<31>(self.0);
            Neon(vaddq_s32(
                self.0,
                vandq_s32(negative, vdupq_laneq_s32::<0>(c)),
            ))
        }
    }

    #[inline(always)]
    fn constants(self, f: f64) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe {
            let f = vdupq_n_f64(f);
            let lo = vcvtq_f64_s64(vmovl_s32(vget_low_s32(self.0)));
            let hi = vcvtq_f64_s64(vmovl_high_s32(self.0));
            let lo = vcvtnq_s64_f64(vmulq_f64(lo, f));
            let hi = vcvtnq_s64_f64(vmulq_f64(hi, f));
            Neon(vmovn_high_s64(vmovn_s64(lo), hi))
        }
    }

    #[inline(always)]
    fn rev(self) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe {
            let r = vrev64q_s32(self.0);
            Neon(vextq_s32::<2>(r, r))
        }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: NEON, which every aarch64 target of this module has.
        unsafe { Neon(vnegq_s32(self.0)) }
    }
}

/// Sets `out[e][o..o + 4]` to limb `e` of the coefficients `j..j + 4` of the product, from their
/// residues `y_i` in `[0, q_i)` at `res[i] + j`: the `ND` digits of
/// `sum y_i (P / q_i) + k (-P)`, accumulated in 64-bit lanes two coefficients at a time (`umlal`
/// by a lane of the constants), the carries moved up (`usra`), and pairs of digits zipped into
/// limbs.
///
/// A digit's accumulator takes at most 8 products below `(2^29 - 1) (2^32 - 1)`, `k (2^32 - 1)`
/// with `k < 8` and the carry from below, together less than `2^64`.
///
/// # Safety
///
/// The residues `j..j + 4` of each prime must be valid for reads.
#[inline(always)]
pub(super) unsafe fn four<const ND: usize>(
    res: &[*const i32; 8],
    j: usize,
    c: &Crt,
    out: &mut [[u64; 8]; 4],
    o: usize,
) {
    // SAFETY: as the caller guarantees; the constants are arrays of 8.
    unsafe {
        let z = vdupq_n_u64(0);
        let (mut lo, mut hi) = ([z; 8], [z; 8]);
        let (mut flo, mut fhi) = (z, z);
        for (i, r) in res.iter().enumerate().take(c.np) {
            let y = vld1q_u32(r.add(j).cast());
            let yl = vget_low_u32(y);
            let m0 = vld1q_u32(c.m[i].as_ptr());
            let m1 = vld1q_u32(c.m[i].as_ptr().add(4));
            flo = vmlal_n_u32(flo, yl, c.ci[i]);
            fhi = vmlal_high_n_u32(fhi, y, c.ci[i]);
            macro_rules! digit {
                ($d:literal, $m:ident, $l:literal) => {
                    if $d < ND {
                        lo[$d] = vmlal_laneq_u32::<$l>(lo[$d], yl, $m);
                        hi[$d] = vmlal_high_laneq_u32::<$l>(hi[$d], y, $m);
                    }
                };
            }
            digit!(0, m0, 0);
            digit!(1, m0, 1);
            digit!(2, m0, 2);
            digit!(3, m0, 3);
            digit!(4, m1, 0);
            digit!(5, m1, 1);
            digit!(6, m1, 2);
            digit!(7, m1, 3);
        }
        // k = floor((sum y_i floor(2^60 / q_i) + 2^58) / 2^60).
        let bias = vdupq_n_u64(1 << 58);
        let kl = vmovn_u64(vshrq_n_u64::<60>(vaddq_u64(flo, bias)));
        let kh = vmovn_u64(vshrq_n_u64::<60>(vaddq_u64(fhi, bias)));
        let p0 = vld1q_u32(c.pneg.as_ptr());
        let p1 = vld1q_u32(c.pneg.as_ptr().add(4));
        macro_rules! times_k {
            ($d:literal, $m:ident, $l:literal) => {
                if $d < ND {
                    lo[$d] = vmlal_laneq_u32::<$l>(lo[$d], kl, $m);
                    hi[$d] = vmlal_laneq_u32::<$l>(hi[$d], kh, $m);
                }
            };
        }
        times_k!(0, p0, 0);
        times_k!(1, p0, 1);
        times_k!(2, p0, 2);
        times_k!(3, p0, 3);
        times_k!(4, p1, 0);
        times_k!(5, p1, 1);
        times_k!(6, p1, 2);
        times_k!(7, p1, 3);
        let mut digits = [vdupq_n_u32(0); 9];
        for d in 0..ND {
            if d + 1 < ND {
                lo[d + 1] = vsraq_n_u64::<32>(lo[d + 1], lo[d]);
                hi[d + 1] = vsraq_n_u64::<32>(hi[d + 1], hi[d]);
            }
            digits[d] = vcombine_u32(vmovn_u64(lo[d]), vmovn_u64(hi[d]));
        }
        for e in 0..ND.div_ceil(2) {
            let (a, b) = (digits[2 * e], digits[2 * e + 1]);
            let row = out[e].as_mut_ptr().add(o);
            vst1q_u64(row, vreinterpretq_u64_u32(vzip1q_u32(a, b)));
            vst1q_u64(row.add(2), vreinterpretq_u64_u32(vzip2q_u32(a, b)));
        }
    }
}

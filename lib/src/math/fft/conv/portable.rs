//! The portable kernels of the convolutions: the steps of the SIMD kernels on arrays standing for
//! their vectors (4 lanes of `f32`, 2 of `f64`), for the targets without them and for Miri.

/// The values past the last one a kernel needs that it may read (zeros in the caller's copy).
pub(super) const PAD: usize = 64;

/// The factor of the cost of a product of the direct kernels against the measured `mac` (of
/// NEON): 1, not measured (the compiler's own vectorisation of the arrays decides).
#[inline]
pub(super) fn mac_scale() -> f64 {
    1.0
}

macro_rules! lanes {
    ($m:ident, $t:ty, $w:literal, $na:literal, $g:literal, [$($s:literal),*], [$($c:tt),*]) => {
        mod $m {
            type V = [$t; $w];

            #[inline(always)]
            unsafe fn zero() -> V {
                [0.0; $w]
            }
            #[inline(always)]
            unsafe fn ld(p: *const $t) -> V {
                // SAFETY: the caller's.
                unsafe { p.cast::<V>().read_unaligned() }
            }
            #[inline(always)]
            unsafe fn st(p: *mut $t, v: V) {
                // SAFETY: the caller's.
                unsafe { p.cast::<V>().write_unaligned(v) }
            }
            #[inline(always)]
            unsafe fn splat(x: $t) -> V {
                [x; $w]
            }
            #[inline(always)]
            unsafe fn add(a: V, b: V) -> V {
                core::array::from_fn(|l| a[l] + b[l])
            }
            #[inline(always)]
            unsafe fn fma(a: V, x: V, y: V) -> V {
                core::array::from_fn(|l| a[l] + x[l] * y[l])
            }
            #[inline(always)]
            unsafe fn tload(p: *const $t) -> V {
                // SAFETY: the caller's.
                unsafe { ld(p) }
            }
            #[inline(always)]
            unsafe fn tfma<const S: i32>(a: V, x: V, t: V) -> V {
                core::array::from_fn(|l| a[l] + x[l] * t[S as usize])
            }
            #[inline(always)]
            unsafe fn combine(a: V, b: V) -> V {
                core::array::from_fn(|l| if l % 2 == 0 { a[l] - b[l + 1] } else { a[l] + b[l - 1] })
            }

            fir_real!(fir, $t, $w, $na, $g, [$($s),*]);
            fir_complex!(firc, $t, $w, $na, [$($c),*]);
            fir_real_few!(fir_few, $t, $w);
            fir_complex_few!(firc_few, $t, $w);

            /// `x[j] = x[j]·y[j]·s` for the `n` complex values at `x` and `y` (interleaved parts).
            ///
            /// # Safety
            ///
            /// `x` must be valid for reads and writes and `y` for reads of `2n` values.
            pub(crate) unsafe fn cmul(x: *mut $t, y: *const $t, s: $t, n: usize) {
                // SAFETY: the loads and stores stay below `2n`.
                unsafe {
                    for j in 0..n {
                        let (ar, ai) = (*x.add(2 * j), *x.add(2 * j + 1));
                        let (br, bi) = (*y.add(2 * j), *y.add(2 * j + 1));
                        *x.add(2 * j) = (ar * br - ai * bi) * s;
                        *x.add(2 * j + 1) = (ar * bi + ai * br) * s;
                    }
                }
            }
        }
    };
}

lanes!(s32, f32, 4, 8, 2, [0, 1, 2, 3], [(0, 0, 1), (1, 2, 3)]);
lanes!(s64, f64, 2, 8, 2, [0, 1], [(0, 0, 1)]);

pub(super) use s32::{
    cmul as cmul_f32, fir as fir_f32, fir_few as fir_few_f32, firc as firc_f32,
    firc_few as firc_few_f32,
};
pub(super) use s64::{
    cmul as cmul_f64, fir as fir_f64, fir_few as fir_few_f64, firc as firc_f64,
    firc_few as firc_few_f64,
};

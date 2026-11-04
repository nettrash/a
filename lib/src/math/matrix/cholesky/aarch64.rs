//! The leaf of the Cholesky decomposition in NEON intrinsics, for `f64` and `f32`.
//!
//! The leaf is copied to the stack, its columns padded to whole vectors, and factored left
//! looking: each column is held in registers while the columns before it are subtracted, whole
//! vectors at a time with the multiplier broadcast, then scaled by the reciprocal of its pivot's
//! square root. The subtractions start at the section of the vectors (a quarter, a half or all
//! of them, as their number allows) that holds the column's diagonal, so that their lengths are
//! known and the column stays in registers. Whole vectors
//! include rows above the diagonal: they take products of elements of the upper triangle, which
//! the lower triangle never reads, and are written back as zeros.
//!
//! The pivot is summed apart in scalars, the same products in the same order as its lane: taking
//! it out of the vector cost a store and a load on the path from one column to the next, which
//! held a 9 × 9 decomposition at 167 ns against 136 without.

use core::arch::aarch64::*;
use core::mem::MaybeUninit;
use core::ptr;

/// The most rows of a leaf.
pub(super) const ROWS: usize = 32;

/// A Cholesky leaf of `$t` in vectors of `$n` lanes.
macro_rules! leaf {
    ($name:ident, $t:ty, $n:literal, $ld:ident, $st:ident, $dup:ident, $fmsn:ident, $mul:ident) => {
        /// Factors the `n × n` block at `a`, columns `ld` apart, `n <= K = $n · V`, in place:
        /// see the module. `false` when a pivot is not positive, or not a number, as soon as it
        /// is met.
        ///
        /// # Safety
        ///
        /// `K = $n · V <= ROWS`; the `n` rows of the `n` columns of `a` must be valid for reads
        /// and writes, and the `len` elements from `a` for reads.
        #[inline(never)]
        pub(super) unsafe fn $name<const V: usize, const K: usize>(
            a: *mut $t,
            n: usize,
            ld: usize,
            len: usize,
        ) -> bool {
            // SAFETY: the reads and writes of `a` are of its block or, for whole vectors, of the
            // `len` elements from it, as the caller guarantees, and those of the stack inside
            // it.
            unsafe {
                // The first `n` columns, each written here before it is read; the others are
                // never read.
                let mut l = [[MaybeUninit::uninit(); V]; K];
                for (j, col) in l.iter_mut().enumerate().take(n) {
                    let src = a.add(j * ld);
                    if j * ld + K <= len {
                        // Whole vectors, the rows past the block's taken as padding.
                        for (v, x) in col.iter_mut().enumerate() {
                            *x = MaybeUninit::new($ld(src.add($n * v)));
                        }
                    } else {
                        let mut t = [0.0 as $t; K];
                        ptr::copy_nonoverlapping(src, t.as_mut_ptr(), n);
                        for (v, x) in col.iter_mut().enumerate() {
                            *x = MaybeUninit::new($ld(t.as_ptr().add($n * v)));
                        }
                    }
                }
                let l = &mut *(&mut l[..n] as *mut [[MaybeUninit<_>; V]] as *mut [[_; V]]);
                // The rows from `j` on start in a later section of the vectors as `j` grows:
                // the update of each section has a constant length, so that `y` stays in
                // registers.
                let sections = if V % 4 == 0 {
                    4
                } else if V % 2 == 0 {
                    2
                } else {
                    1
                };
                for j in 0..n {
                    let mut y = l[j];
                    let diagonal = l.as_mut_ptr().cast::<$t>().add(j * K + j);
                    // The pivot is summed apart, in the order and with the roundings of its lane,
                    // so that it need not be taken out of the vector.
                    let mut d = *diagonal;
                    macro_rules! update {
                        ($from:expr) => {
                            for p in 0..j {
                                let s = *l.as_ptr().cast::<$t>().add(p * K + j);
                                d = (-s).mul_add(s, d);
                                for v in $from..V {
                                    y[v] = $fmsn(y[v], l[p][v], s);
                                }
                            }
                        };
                    }
                    match j * sections / K {
                        0 => update!(0),
                        1 => update!(V / sections),
                        2 => update!(2 * V / sections),
                        _ => update!(3 * V / sections),
                    }
                    if d.is_nan() || d <= 0.0 {
                        return false;
                    }
                    let d = d.sqrt();
                    let r = $dup(1.0 / d);
                    for y in &mut y {
                        *y = $mul(*y, r);
                    }
                    l[j] = y;
                    *diagonal = d;
                }
                // The columns back, zeros above the diagonal: whole vectors, then the last
                // row alone.
                let whole = n / $n;
                for (j, col) in l.iter().enumerate() {
                    let dst = a.add(j * ld);
                    for (v, x) in col.iter().enumerate() {
                        let mut x = *x;
                        if $n * v + $n <= j {
                            x = $dup(0.0);
                        } else if $n * v < j {
                            // The vector across the diagonal.
                            let mut t = [0.0 as $t; $n];
                            $st(t.as_mut_ptr(), x);
                            t[..j - $n * v].fill(0.0);
                            x = $ld(t.as_ptr());
                        }
                        if v < whole {
                            $st(dst.add($n * v), x);
                        } else if v == whole && n > $n * whole {
                            let mut t = [0.0 as $t; $n];
                            $st(t.as_mut_ptr(), x);
                            for (i, t) in t.iter().enumerate().take(n - $n * whole) {
                                *dst.add($n * whole + i) = *t;
                            }
                        }
                    }
                }
                true
            }
        }
    };
}

leaf!(
    leaf_f64,
    f64,
    2,
    vld1q_f64,
    vst1q_f64,
    vdupq_n_f64,
    vfmsq_n_f64,
    vmulq_f64
);
leaf!(
    leaf_f32,
    f32,
    4,
    vld1q_f32,
    vst1q_f32,
    vdupq_n_f32,
    vfmsq_n_f32,
    vmulq_f32
);

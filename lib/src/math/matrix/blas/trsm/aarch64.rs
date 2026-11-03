//! The leaves of the triangular solves in NEON intrinsics, for `f64` and `f32`.
//!
//! The compiler's own code for the portable leaves kept the unknowns in memory and their
//! multiply-adds scalar, as the rows they update start wherever the solved row ends; here the
//! columns of `b` are vectors held in registers, two columns at a time, and each solved row
//! updates the vectors from the one holding the next row, by a lane of its own vector (`fmls
//! v.2d, v.2d, v.d[i]`). The rows of that vector already solved take products by the zeros the
//! packed triangle holds there, which leave finite values as they are; a value that is not finite
//! sends the columns back to substitution with divisions.

use super::LEAF;
use core::arch::aarch64::*;
use core::mem::MaybeUninit;
use core::ptr;

/// A left leaf of `$t`: columns of `b` solved against the packed triangle, `V` vectors of `$n`
/// lanes a column, `C` columns at a time.
macro_rules! left {
    (
        $name:ident, $t:ty, $n:literal, [$($l:literal),*], [$($rl:literal),*],
        $ld:ident, $st:ident, $dup:ident, $fms:ident, $mul:ident, $add:ident, $addv:ident
    ) => {
        /// The columns `b`, `b + ldb`, … (`C` of them) of `m <= $n · V` rows, solved by the
        /// packed triangle `l` of [`Packed`](super::Packed) (forward when it is lower) and
        /// scaled by `r` unless `unit` is set, then written to `b` and, when `x` is not null, to
        /// `x`, `x + ldx`, …; or `false`, with nothing written, when a value is not finite.
        ///
        /// # Safety
        ///
        /// `V · $n <= LEAF`, `m >= 1`, and `l` holds the first `$n · V` columns of the triangle;
        /// the `m` rows of the `C` columns of `b` must be valid for reads and writes, and of `x`
        /// for writes when it is not null.
        #[inline(never)]
        #[allow(clippy::too_many_arguments)]
        pub(super) unsafe fn $name<const V: usize, const C: usize>(
            l: &[$t],
            r: &[$t; LEAF],
            forward: bool,
            unit: bool,
            m: usize,
            b: *mut $t,
            ldb: usize,
            x: *mut $t,
            ldx: usize,
        ) -> bool {
            // SAFETY: the reads and writes of `b` and `x` are of their rows, as the caller
            // guarantees, and those of `l`, `r` and the stack inside them.
            unsafe {
                let lp = l.as_ptr();
                let full = m == $n * V;
                let mut y = [[$dup(0.0); V]; C];
                for (c, y) in y.iter_mut().enumerate() {
                    let col = b.add(c * ldb);
                    if full {
                        for (v, y) in y.iter_mut().enumerate() {
                            *y = $ld(col.add($n * v));
                        }
                    } else {
                        let mut t = [0.0 as $t; LEAF];
                        ptr::copy_nonoverlapping(col, t.as_mut_ptr(), m);
                        for (v, y) in y.iter_mut().enumerate() {
                            *y = $ld(t.as_ptr().add($n * v));
                        }
                    }
                }
                if forward {
                    for qq in 0..V {
                        $(
                            {
                                // Row `q` updates the vectors from the one holding row `q + 1`.
                                let q = $n * qq + $l;
                                let v0 = if $l + 1 < $n { qq } else { qq + 1 };
                                let s: [_; C] = core::array::from_fn(|c| y[c][qq]);
                                for v in v0..V {
                                    let lv = $ld(lp.add(q * LEAF + $n * v));
                                    for c in 0..C {
                                        y[c][v] = $fms::<$l>(y[c][v], lv, s[c]);
                                    }
                                }
                            }
                        )*
                    }
                } else {
                    for qq in (0..V).rev() {
                        $(
                            {
                                // Row `q` updates the vectors up to the one holding row `q - 1`.
                                let q = $n * qq + $rl;
                                let v1 = if $rl > 0 { qq + 1 } else { qq };
                                let s: [_; C] = core::array::from_fn(|c| y[c][qq]);
                                for v in 0..v1 {
                                    let lv = $ld(lp.add(q * LEAF + $n * v));
                                    for c in 0..C {
                                        y[c][v] = $fms::<$rl>(y[c][v], lv, s[c]);
                                    }
                                }
                            }
                        )*
                    }
                }
                // `0 · y` is zero for every finite `y` and NaN otherwise.
                let mut z = $dup(0.0);
                for y in &y {
                    for y in y {
                        z = $add(z, $mul(*y, $dup(0.0)));
                    }
                }
                if $addv(z) != 0.0 {
                    return false;
                }
                if !unit {
                    for y in &mut y {
                        for (v, y) in y.iter_mut().enumerate() {
                            *y = $mul(*y, $ld(r.as_ptr().add($n * v)));
                        }
                    }
                }
                for (c, y) in y.iter().enumerate() {
                    for (dst, ld) in [(b, ldb), (x, ldx)] {
                        if dst.is_null() {
                            continue;
                        }
                        let col = dst.add(c * ld);
                        if full {
                            for (v, y) in y.iter().enumerate() {
                                $st(col.add($n * v), *y);
                            }
                        } else {
                            let mut t = [0.0 as $t; LEAF];
                            for (v, y) in y.iter().enumerate() {
                                $st(t.as_mut_ptr().add($n * v), *y);
                            }
                            ptr::copy_nonoverlapping(t.as_ptr(), col, m);
                        }
                    }
                }
                true
            }
        }
    };
}

left!(
    left_f64,
    f64,
    2,
    [0, 1],
    [1, 0],
    vld1q_f64,
    vst1q_f64,
    vdupq_n_f64,
    vfmsq_laneq_f64,
    vmulq_f64,
    vaddq_f64,
    vaddvq_f64
);
left!(
    left_f32,
    f32,
    4,
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    vld1q_f32,
    vst1q_f32,
    vdupq_n_f32,
    vfmsq_laneq_f32,
    vmulq_f32,
    vaddq_f32,
    vaddvq_f32
);

/// A right leaf of `$t`: `$n · V` rows of `b`, solved against the packed transpose of the
/// triangle, `$n` columns at a time.
macro_rules! right {
    (
        $name:ident, $t:ty, $n:literal, [$($l:literal),*],
        $ld:ident, $st:ident, $dup:ident, $fms:ident, $fmsn:ident, $mul:ident, $add:ident,
        $addv:ident
    ) => {
        /// Rows `b`, …, `b + $n · V - 1` of the `n` columns of `b`, solved by the packed
        /// transpose `l` of [`Packed`](super::Packed) (forward when it is lower) and scaled by
        /// `r` unless `unit` is set; or `false`, with nothing written, when a value is not
        /// finite.
        ///
        /// Each group of `$n` columns is summed in registers from the columns solved before it,
        /// a vector of the triangle's elements by lanes, then solved within itself.
        ///
        /// # Safety
        ///
        /// `1 <= n <= LEAF`, and `l` holds the first `n` columns of the triangle rounded up to
        /// `$n`; the rows of the `n` columns of `b` must be valid for reads and writes.
        #[inline(never)]
        pub(super) unsafe fn $name<const V: usize>(
            l: &[$t],
            r: &[$t; LEAF],
            forward: bool,
            unit: bool,
            n: usize,
            b: *mut $t,
            ldb: usize,
        ) -> bool {
            // SAFETY: the reads and writes of `b` are of its rows and columns, as the caller
            // guarantees, and those of `l`, `r` and the stack inside them.
            unsafe {
                let lp = l.as_ptr();
                let groups = n.div_ceil($n);
                // The solved columns, each written before it is read.
                let mut y = [[MaybeUninit::uninit(); V]; LEAF];
                for gg in 0..groups {
                    let g = if forward { gg } else { groups - 1 - gg };
                    let j0 = $n * g;
                    let mut acc = [[$dup(0.0); V]; $n];
                    for (w, acc) in acc.iter_mut().enumerate() {
                        if j0 + w < n {
                            let col = b.add((j0 + w) * ldb);
                            for (v, acc) in acc.iter_mut().enumerate() {
                                *acc = $ld(col.add($n * v));
                            }
                        }
                    }
                    let solved = if forward { 0..j0 } else { j0 + $n..$n * groups };
                    for q in solved {
                        // Elements `(j0 .. j0 + $n, q)` of the transpose, one a lane.
                        let lv = $ld(lp.add(j0 + q * LEAF));
                        let yq = &y[q];
                        for v in 0..V {
                            let yv = yq[v].assume_init();
                            $(acc[$l][v] = $fms::<$l>(acc[$l][v], yv, lv);)*
                        }
                    }
                    // The triangle of the group.
                    if forward {
                        for w in 0..$n {
                            for u in w + 1..$n {
                                let e = *lp.add(j0 + u + (j0 + w) * LEAF);
                                for v in 0..V {
                                    acc[u][v] = $fmsn(acc[u][v], acc[w][v], e);
                                }
                            }
                        }
                    } else {
                        for w in (0..$n).rev() {
                            for u in 0..w {
                                let e = *lp.add(j0 + u + (j0 + w) * LEAF);
                                for v in 0..V {
                                    acc[u][v] = $fmsn(acc[u][v], acc[w][v], e);
                                }
                            }
                        }
                    }
                    for (w, acc) in acc.iter().enumerate() {
                        for (v, acc) in acc.iter().enumerate() {
                            y[j0 + w][v] = MaybeUninit::new(*acc);
                        }
                    }
                }
                // All of the first `n` columns are written now.
                let y = &*(&y[..n] as *const [[MaybeUninit<_>; V]] as *const [[_; V]]);
                // `0 · y` is zero for every finite `y` and NaN otherwise.
                let mut z = $dup(0.0);
                for y in y {
                    for y in y {
                        z = $add(z, $mul(*y, $dup(0.0)));
                    }
                }
                if $addv(z) != 0.0 {
                    return false;
                }
                for (j, y) in y.iter().enumerate() {
                    let s = if unit { 1.0 } else { r[j] };
                    for (v, y) in y.iter().enumerate() {
                        let p = b.add(j * ldb + $n * v);
                        $st(p, if unit { *y } else { $mul(*y, $dup(s)) });
                    }
                }
                true
            }
        }
    };
}

right!(
    right_f64,
    f64,
    2,
    [0, 1],
    vld1q_f64,
    vst1q_f64,
    vdupq_n_f64,
    vfmsq_laneq_f64,
    vfmsq_n_f64,
    vmulq_f64,
    vaddq_f64,
    vaddvq_f64
);
right!(
    right_f32,
    f32,
    4,
    [0, 1, 2, 3],
    vld1q_f32,
    vst1q_f32,
    vdupq_n_f32,
    vfmsq_laneq_f32,
    vfmsq_n_f32,
    vmulq_f32,
    vaddq_f32,
    vaddvq_f32
);

//! The product of blocks, `c = β·c + α·op(a)·op(b)`, for [`gemm`](super::blas::gemm), of
//! `f32`, `f64` and complex numbers of either.
//!
//! # Small products
//!
//! A product of at most 64 × 64 × 64 whose blocks lie close together in memory ([`is_small`]) is
//! computed from the operands where they are ([`Real::direct`]): a tile of `c`, 8 rows of `f64`
//! or 16 of `f32` by 4 columns (fewer rows by 8), is summed in vector registers over all of `k`, a
//! column of `a` loaded as vectors at each step and multiplied by each element of `op(b)` in turn,
//! read where it is whatever the operation. Complex numbers split their parts as they are loaded
//! (`ld2`) and take four real products an element; an `op(a)` that is transposed is copied to the
//! stack first. Packing would cost more than it saves at these sizes: an `f64` product of
//! 8 × 8 × 8 takes about 85 cycles, faer's 160 and nalgebra's 300.
//!
//! # Products of few rows
//!
//! When `op(a)` has at most [`SKINNY`](Real::SKINNY) rows, 32, a larger real product goes by
//! [`skinny`]: `op(a)` is copied dense, a block of `k` at a time, and the direct tiles read `op(b)`
//! where it is, as packing it, the larger operand, would cost a copy of it for the few panels of
//! `a` that use it. `f64` products of 8 × 1024 by 1024 × 1024 so ran at 54 GFLOP/s against 35 by
//! blocks, of 16 rows at 54 against 44, and of 32 rows at 54 against 51; at 40 and 48 rows the two
//! tied, and `f32` products of 40 rows were faster by blocks.
//!
//! # Larger products
//!
//! Larger products go by blocks, as Goto and van de Geijn laid them out and BLIS does ([`blocked`]).
//! `op(b)` is packed a block of [`KC`](Real::KC) × [`NC`](Real::NC) at a time, in panels of a
//! few columns whose rows follow each other; for each, `op(a)` is packed a block of
//! [`MC`](Real::MC) rows at a time, in panels of [`MR`](Real::MR) = 8 rows whose columns follow each
//! other, each panel when it is first used, so that it is still in the L1 cache for the
//! micro-kernel. The micro-kernel ([`Real::kernel`]) computes an `8 × nr` tile of `c` from a panel
//! of each: at each step, the column of `a` in vectors, the row of `b` in vectors whose lanes
//! multiply the column by element (`fmla v.2d, v.2d, v.d[i]`), 24 vector sums for the widest
//! panels, 8 × 6 of `f64` and 8 × 12 of `f32`. An Apple M4 core runs four multiply-adds a cycle,
//! each four cycles long, so that 16 independent sums keep it busy, and 24 leave room; the shapes
//! with 24 sums all ran within 2 % of each other (8 × 6, 6 × 8, 4 × 12 and 12 × 4 of `f64`;
//! 16 × 6, 8 × 12, 12 × 8 of `f32`). The rows of `a` come in eights so that blocks of 32 and 64
//! rows, the panels of the decompositions, need no partial tile.
//!
//! A panel of `op(b)` takes [`Real::width`] columns: 6, 4 or 2 of `f64`, 12, 8 or 4 of `f32`,
//! so that the last panels of a block fit its columns without padding where they can: 32 columns
//! of `f64` are 6 · 4 + 4 + 4, not 6 · 6 with 4 columns of zeros. The narrower kernels have 16 or
//! 8 sums, which take their step in as many cycles as the widest takes its own, so they cost
//! more a column. The last rows of `c`, fewer than 8, and padded panels are computed into a tile
//! on the stack and added to `c` from there.
//!
//! The blocks split each dimension evenly. [`KC`](Real::KC) = 512 steps (24 KiB of a panel of
//! `b` and 32 KiB of `a` in `f64`) was 1 to 3 % faster than 256 and as fast as 768 or 1024;
//! [`MC`](Real::MC) from 64 to 512 and [`NC`](Real::NC) from 1020 to 3060 made no difference
//! beyond the noise, so they are left at sizes that fit the L2 cache with room: 128 rows, and
//! blocks of `b` of 4 MiB.
//!
//! When `op(a)` is `a` itself and a block of `op(b)` has at most [`IN_PLACE`] panels, the kernel
//! reads `a`'s columns where they are rather than from packed panels, as each panel of `a` would
//! serve too few tiles to repay its copy: `f32` products of 1024 × 1024 by 1024 × 32 ran at 119
//! GFLOP/s so against 96 with `a` packed, and of `f64` by 1024 × 8 at 52 against 38.
//!
//! Complex numbers go through the real kernels, in Van Zee's 1m method: element `x + y·i` of a
//! panel of `op(a)` takes two rows and two columns of a real panel, `[x, y]` and `[-y, x]` (the
//! column of the panel holding the real and imaginary parts of a column of `a` in turn), and an
//! element of `op(b)` two rows, its real and imaginary parts; the real product of the two panels is
//! then the complex product, its parts in turn as `c` holds them. A complex multiply-add so takes
//! four real ones, the least it can, at the speed of the real kernel, and the conjugations and
//! `alpha` are applied as the panels are packed. A `beta` off the real line scales `c` first.
//!
//! # Speed
//!
//! Measured on an Apple M4 Max, one core, in GFLOP/s (`2mnk`, `8mnk` for complex numbers), the
//! median of 5 to 11 runs, against faer 0.24 (`Par::Seq`) and nalgebra 0.35 (`DMatrix::gemm`):
//!
//! | product | `f64` | faer | nalgebra | `f32` | faer | nalgebra |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 8 × 8 · 8 × 8 | 44.9 | 23.9 | 13.3 | 63.5 | 36.3 | 20.6 |
//! | 64 × 64 · 64 × 64 | 55.6 | 48.4 | 46.3 | 112.9 | 96.5 | 100.4 |
//! | 1024 × 1024 · 1024 × 1024 | 59.4 | 50.6 | 57.3 | 118.7 | 106.5 | 108.3 |
//! | 1024 × 32 · 32 × 1024 | 59.0 | 55.2 | 49.8 | 116.9 | 109.1 | 112.4 |
//! | 32 × 1024 · 1024 × 1024 | 53.1 | 47.7 | 50.7 | 110.0 | 110.6 | 91.4 |
//!
//! | product | `Complex<f64>` | faer | nalgebra | `Complex<f32>` | faer | nalgebra |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 8 × 8 · 8 × 8 | 40.5 | 41.8 | 9.8 | 80.6 | 63.7 | 15.1 |
//! | 64 × 64 · 64 × 64 | 53.3 | 43.4 | 10.5 | 107.1 | 82.2 | 21.1 |
//! | 1024 × 1024 · 1024 × 1024 | 59.1 | 45.0 | 10.5 | 119.0 | 88.4 | 20.9 |
//! | 1024 × 32 · 32 × 1024 | 60.2 | 48.5 | 10.7 | 119.5 | 91.4 | 21.4 |
//!
//! The core's four multiply-adds a cycle at 4.5 GHz make 72 GFLOP/s of `f64`, but under a load
//! of them alone its clock falls: a loop of 24 independent multiply-adds held 56 to 60 GFLOP/s
//! over 30 to 300 ms, which the larger products match. Apple's Accelerate, on the matrix unit
//! (SME) rather than NEON, ran 3 to 15 times faster from 32 × 32 · 32 × 32 up.

use super::blas::Op;
use crate::math::complex::{Complex, Float};
use core::mem::MaybeUninit;
use core::ptr;

// The micro-kernels, one set per target under the name `kernels`: NEON on little-endian aarch64
// (the transposes reinterpret vectors of `f32` as `f64`), and portable Rust elsewhere and under
// Miri.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
#[path = "gemm/aarch64.rs"]
mod kernels;
#[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
#[path = "gemm/portable.rs"]
mod kernels;

/// `f32` or `f64`, with the micro-kernel of its products and the blocks it works on.
pub(super) trait Real: Float {
    /// The rows of the tile of the micro-kernel, a multiple of its vectors' lanes and even.
    const MR: usize;
    /// The most columns of a tile of the micro-kernel.
    const NR: usize;
    /// The steps of `k` of a packed block, even.
    const KC: usize;
    /// The rows of a packed block of `op(a)`, a multiple of [`MR`](Self::MR).
    const MC: usize;
    /// The columns of a packed block of `op(b)`, a multiple of [`NR`](Self::NR).
    const NC: usize;
    /// The most rows of `op(a)` for which [`skinny`] computes the larger products.
    const SKINNY: usize;

    /// The columns of the next panel of `op(b)` when `r` are left: a width of one of the kernels,
    /// the widest but where narrower ones cover the rest with fewer steps, as a tile's step takes
    /// as long for a narrower kernel whose sums wait on each other.
    fn width(r: usize) -> usize;

    /// Sets the `MR × nr` tile at `c`, columns `ldc` apart, to `beta · c + a·b`: the sum over
    /// `kc` steps of the products of a column of `MR` in `a`, `sa` apart from step to step, and a
    /// row of `nr` packed in `b`. When `beta` is zero, `c` is not read.
    ///
    /// # Safety
    ///
    /// `kc >= 1` and `nr` a [`width`](Self::width); the columns of `a` must be valid for reads,
    /// `b` for reads of `nr · kc` elements, and the tile at `c` for reads and writes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kernel(
        nr: usize,
        kc: usize,
        a: *const Self,
        sa: usize,
        b: *const Self,
        c: *mut Self,
        ldc: usize,
        beta: Self,
    );

    /// `c + a · b`, with one rounding where the target has fused multiply-adds.
    fn fma(c: Self, a: Self, b: Self) -> Self;

    /// Packs a panel of `w` lines of `kc` steps times `scale`, from `h <= w` lines whose steps
    /// lie in a row: element `(r, p)` is at `src[p + r · ld]` and goes to `dst[p · w + r]`, and
    /// lines `h..w` are zeros.
    ///
    /// # Safety
    ///
    /// The elements must be valid for reads, and `dst` for writes of `w · kc` reals.
    #[allow(clippy::too_many_arguments)]
    unsafe fn pack_t(
        dst: *mut Self,
        src: *const Self,
        ld: usize,
        h: usize,
        kc: usize,
        w: usize,
        scale: Self,
    );

    /// `c = beta · c + alpha · a · op(b)` without packing, for small products: `a` is `m × k`
    /// with the leading dimension `lda`, the element `(p, j)` of `op(b)` is at
    /// `b[p · rsb + j · csb]`.
    ///
    /// # Safety
    ///
    /// `m`, `n`, `k >= 1`; the blocks must be valid for their reads and `c` for its writes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn direct(
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        b: *const Self,
        rsb: usize,
        csb: usize,
        beta: Self,
        c: *mut Self,
        ldc: usize,
    );

    /// [`Real::direct`] of complex numbers: `c = beta · c + alpha · a · op(b)`, with the element
    /// `(p, j)` of `op(b)` at `b[p · rsb + j · csb]`, conjugated when `CJ` is set.
    ///
    /// # Safety
    ///
    /// As for [`Real::direct`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn direct_complex<const CJ: bool>(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex<Self>,
        a: *const Complex<Self>,
        lda: usize,
        b: *const Complex<Self>,
        rsb: usize,
        csb: usize,
        beta: Complex<Self>,
        c: *mut Complex<Self>,
        ldc: usize,
    );
}

/// The elements of products: a real float, held as one real, or a complex number of one, held as
/// two.
trait Elem: Copy + PartialEq {
    /// The real type the micro-kernel computes in.
    type R: Real;
    /// The reals an element takes: 1 or 2.
    const W: usize;
    /// 0.
    const ZERO: Self;
    /// 1.
    const ONE: Self;

    /// Packs a panel of `R::MR · W · kc` reals from `h <= R::MR / W` rows and `kc` columns of
    /// `op(a)`, zero-padded: element `(i, p)` of `op(a)` is at `a[i + p · lda]` when `op` is
    /// [`Op::N`] and at `a[p + i · lda]` otherwise.
    ///
    /// # Safety
    ///
    /// The elements must be valid for reads, and `dst` for writing the panel.
    unsafe fn pack_a(dst: *mut Self::R, a: *const Self, lda: usize, op: Op, h: usize, kc: usize);

    /// Packs a panel of `nr · W · kc` reals from `kc` rows and `h <= nr` columns of
    /// `alpha · op(b)`, zero-padded: element `(p, j)` of `op(b)` is at `b[p + j · ldb]` when
    /// `op` is [`Op::N`] and at `b[j + p · ldb]` otherwise.
    ///
    /// # Safety
    ///
    /// The elements must be valid for reads, and `dst` for writing the panel.
    #[allow(clippy::too_many_arguments)]
    unsafe fn pack_b(
        dst: *mut Self::R,
        b: *const Self,
        ldb: usize,
        op: Op,
        kc: usize,
        h: usize,
        nr: usize,
        alpha: Self,
    );
}

impl<R: Real> Elem for R {
    type R = R;
    const W: usize = 1;
    const ZERO: Self = R::ZERO;
    const ONE: Self = R::ONE;

    #[inline(always)]
    unsafe fn pack_a(dst: *mut R, a: *const R, lda: usize, op: Op, h: usize, kc: usize) {
        // SAFETY: as the caller guarantees.
        unsafe {
            if op == Op::N {
                pack_n(dst, a, lda, h, kc, R::MR, R::ONE);
            } else {
                R::pack_t(dst, a, lda, h, kc, R::MR, R::ONE);
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn pack_b(
        dst: *mut R,
        b: *const R,
        ldb: usize,
        op: Op,
        kc: usize,
        h: usize,
        nr: usize,
        alpha: R,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe {
            if op == Op::N {
                R::pack_t(dst, b, ldb, h, kc, nr, alpha);
            } else {
                pack_n(dst, b, ldb, h, kc, nr, alpha);
            }
        }
    }
}

impl<R: Real> Elem for Complex<R> {
    type R = R;
    const W: usize = 2;
    const ZERO: Self = Complex::new(R::ZERO, R::ZERO);
    const ONE: Self = Complex::new(R::ONE, R::ZERO);

    #[inline(always)]
    unsafe fn pack_a(dst: *mut R, a: *const Self, lda: usize, op: Op, h: usize, kc: usize) {
        let mr = R::MR / 2;
        // SAFETY: every read is of an element of the block, and every write inside the panel.
        unsafe {
            for p in 0..kc {
                let row = dst.add(2 * p * R::MR);
                for i in 0..h {
                    let x = match op {
                        Op::N => *a.add(i + p * lda),
                        Op::T => *a.add(p + i * lda),
                        Op::C => (*a.add(p + i * lda)).conj(),
                    };
                    *row.add(2 * i) = x.re;
                    *row.add(2 * i + 1) = x.im;
                    *row.add(R::MR + 2 * i) = -x.im;
                    *row.add(R::MR + 2 * i + 1) = x.re;
                }
                for i in h..mr {
                    for q in [2 * i, 2 * i + 1, R::MR + 2 * i, R::MR + 2 * i + 1] {
                        *row.add(q) = R::ZERO;
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn pack_b(
        dst: *mut R,
        b: *const Self,
        ldb: usize,
        op: Op,
        kc: usize,
        h: usize,
        nr: usize,
        alpha: Self,
    ) {
        // SAFETY: every read is of an element of the block, and every write inside the panel.
        unsafe {
            for p in 0..kc {
                let row = dst.add(2 * p * nr);
                for j in 0..h {
                    let x = match op {
                        Op::N => *b.add(p + j * ldb),
                        Op::T => *b.add(j + p * ldb),
                        Op::C => (*b.add(j + p * ldb)).conj(),
                    };
                    let x = x * alpha;
                    *row.add(j) = x.re;
                    *row.add(nr + j) = x.im;
                }
                for j in h..nr {
                    *row.add(j) = R::ZERO;
                    *row.add(nr + j) = R::ZERO;
                }
            }
        }
    }
}

/// Packs a panel of `w` lines of `kc` steps times `scale`, from `h <= w` lines whose elements lie
/// in a column a step: element `(r, p)` is at `src[r + p · ld]` and goes to `dst[p · w + r]`, and
/// lines `h..w` are zeros.
///
/// # Safety
///
/// The elements must be valid for reads, and `dst` for writes of `w · kc` reals.
#[inline(always)]
unsafe fn pack_n<R: Real>(
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
        if h == w && scale == R::ONE {
            for p in 0..kc {
                ptr::copy_nonoverlapping(src.add(p * ld), dst.add(p * w), w);
            }
        } else {
            for p in 0..kc {
                let (s, d) = (src.add(p * ld), dst.add(p * w));
                for r in 0..h {
                    *d.add(r) = *s.add(r) * scale;
                }
                for r in h..w {
                    *d.add(r) = R::ZERO;
                }
            }
        }
    }
}

/// The most panels of `op(b)` in a block for which [`blocked`] reads `a` in place.
const IN_PLACE: usize = 16;

/// Working memory of [`blocked`] on the stack, in reals: products whose packed blocks fit take no
/// allocation.
const STACK: usize = 4096;

/// [`STACK`] reals of working memory, aligned to a cache line.
#[repr(C, align(128))]
struct Stack<R>([MaybeUninit<R>; STACK]);

/// `c = beta · c + alpha · op(a) · op(b)` by blocks: `op(b)` is packed a block of `KC × NC` at a
/// time, `op(a)` a block of `MC × KC` for each, and the micro-kernel computes the tiles of `c` from
/// panels of the two, `c` seen as reals, columns `ldc` reals apart.
///
/// # Safety
///
/// `m`, `n`, `k >= 1`; the blocks must be valid for their reads and `c` for its writes, as
/// [`gemm`](super::blas::gemm) checks.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn blocked<E: Elem>(
    m: usize,
    n: usize,
    k: usize,
    alpha: E,
    a: *const E,
    lda: usize,
    opa: Op,
    b: *const E,
    ldb: usize,
    opb: Op,
    beta: E::R,
    c: *mut E::R,
    ldc: usize,
) {
    let w = E::W;
    let (mr, nr) = (E::R::MR / w, E::R::NR);
    // The blocks split each dimension evenly, so that none is much shorter than the others: a
    // short block of `k` costs a whole pass over `c`.
    let kcb = k.div_ceil(k.div_ceil(E::R::KC / w));
    let mcb = m.div_ceil(m.div_ceil(E::R::MC / w)).next_multiple_of(mr);
    let ncb = n.div_ceil(n.div_ceil(E::R::NC)).next_multiple_of(nr);
    let (kc_max, mc_max, nc_max) = (kcb, mcb, ncb + nr);
    let a_len = mc_max * kc_max * w * w;
    let b_len = nc_max * kc_max * w;
    let run = |buf: *mut E::R| {
        let (ap, bp) = (buf, buf.wrapping_add(a_len.next_multiple_of(16)));
        let (zero, one) = (<E::R as Float>::ZERO, <E::R as Float>::ONE);
        let mut tile = [zero; 128];
        // SAFETY: the packed blocks fit the buffer, the blocks of `a` and `b` are read inside
        // the matrices, and every tile of `c` written is inside `c`.
        unsafe {
            let mut jc = 0;
            while jc < n {
                let nc = ncb.min(n - jc);
                let mut pc = 0;
                while pc < k {
                    let kc = kcb.min(k - pc);
                    let beta = if pc == 0 { beta } else { one };
                    let mut jr = 0;
                    while jr < nc {
                        let nr = E::R::width(nc - jr);
                        let (j, h) = (jc + jr, nr.min(nc - jr));
                        let bs = if opb == Op::N {
                            pc + j * ldb
                        } else {
                            j + pc * ldb
                        };
                        E::pack_b(bp.add(jr * kc * w), b.add(bs), ldb, opb, kc, h, nr, alpha);
                        jr += nr;
                    }
                    // `op(a)` in place when it is `a` itself, real, and each of its panels would
                    // serve few tiles: packing it would cost more than it saves.
                    let in_place = w == 1 && opa == Op::N && nc <= IN_PLACE * nr;
                    let mut ic = 0;
                    while ic < m {
                        let mc = mcb.min(m - ic);
                        let mut jr = 0;
                        while jr < nc {
                            let nr = E::R::width(nc - jr);
                            let nn = nr.min(nc - jr);
                            let bt = bp.add(jr * kc * w);
                            let mut ir = 0;
                            while ir < mc {
                                let mm = mr.min(mc - ir);
                                let i = ic + ir;
                                let as_ = if opa == Op::N {
                                    i + pc * lda
                                } else {
                                    pc + i * lda
                                };
                                let (at, sa) = if in_place && mm == mr {
                                    // SAFETY of the cast: `E` is `E::R` when `w` is 1.
                                    (a.add(as_).cast::<E::R>(), lda)
                                } else {
                                    let at = ap.add(ir * kc * w * w);
                                    if jr == 0 {
                                        E::pack_a(at, a.add(as_), lda, opa, mm, kc);
                                    }
                                    (at.cast_const(), E::R::MR)
                                };
                                let ct = c.add((ic + ir) * w + (jc + jr) * ldc);
                                if mm == mr && nn == nr {
                                    E::R::kernel(nr, kc * w, at, sa, bt, ct, ldc, beta);
                                } else {
                                    let t = tile.as_mut_ptr();
                                    E::R::kernel(nr, kc * w, at, sa, bt, t, E::R::MR, zero);
                                    for j in 0..nn {
                                        for i in 0..mm * w {
                                            let x = ct.add(i + j * ldc);
                                            let y = *t.add(i + j * E::R::MR);
                                            *x = if beta == zero {
                                                y
                                            } else {
                                                E::R::fma(y, beta, *x)
                                            };
                                        }
                                    }
                                }
                                ir += mr;
                            }
                            jr += nr;
                        }
                        ic += mc;
                    }
                    pc += kc;
                }
                jc += nc;
            }
        }
    };
    let need = a_len.next_multiple_of(16) + b_len;
    if need <= STACK {
        let mut s = Stack::<E::R>([const { MaybeUninit::uninit() }; STACK]);
        run(s.0.as_mut_ptr().cast());
    } else {
        let mut v: Vec<E::R> = Vec::with_capacity(need + 32);
        let p = v.as_mut_ptr();
        // Start on a cache line.
        let skip = p.align_offset(128).min(32);
        run(p.wrapping_add(skip));
    }
}

/// An `MR × NR` tile of [`Real::direct`] from `a` and `op(b)` where they are, portable: arrays of
/// sums the compiler keeps in registers or vectors where it can, `MR` rows of `a` a step.
///
/// # Safety
///
/// As for [`Real::direct`], for the tile.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile<R: Real, const MR: usize, const NR: usize>(
    k: usize,
    a: *const R,
    lda: usize,
    b: *const R,
    rsb: usize,
    csb: usize,
    c: *mut R,
    ldc: usize,
    alpha: R,
    beta: R,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut acc = [[R::ZERO; MR]; NR];
        for p in 0..k {
            let mut col = [R::ZERO; MR];
            for (i, x) in col.iter_mut().enumerate() {
                *x = *a.add(i + p * lda);
            }
            for (j, sums) in acc.iter_mut().enumerate() {
                let y = *b.add(p * rsb + j * csb);
                for (s, x) in sums.iter_mut().zip(&col) {
                    *s = R::fma(*s, *x, y);
                }
            }
        }
        for (j, sums) in acc.iter().enumerate() {
            for (i, s) in sums.iter().enumerate() {
                let x = c.add(i + j * ldc);
                *x = if beta == R::ZERO {
                    alpha * *s
                } else {
                    R::fma(beta * *x, alpha, *s)
                };
            }
        }
    }
}

/// The tiles of [`tile`] of `MR` rows: four columns at a time, then the rest.
///
/// # Safety
///
/// As for [`Real::direct`], for the rows.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn tiles<R: Real, const MR: usize>(
    n: usize,
    k: usize,
    a: *const R,
    lda: usize,
    b: *const R,
    rsb: usize,
    csb: usize,
    c: *mut R,
    ldc: usize,
    alpha: R,
    beta: R,
) {
    // SAFETY: as the caller guarantees; each tile is inside the block.
    unsafe {
        let mut j = 0;
        while j + 4 <= n {
            tile::<R, MR, 4>(
                k,
                a,
                lda,
                b.add(j * csb),
                rsb,
                csb,
                c.add(j * ldc),
                ldc,
                alpha,
                beta,
            );
            j += 4;
        }
        // Past the end when no columns are left, so not dereferenced then.
        let (b, c) = (b.wrapping_add(j * csb), c.wrapping_add(j * ldc));
        match n - j {
            3 => tile::<R, MR, 3>(k, a, lda, b, rsb, csb, c, ldc, alpha, beta),
            2 => tile::<R, MR, 2>(k, a, lda, b, rsb, csb, c, ldc, alpha, beta),
            1 => tile::<R, MR, 1>(k, a, lda, b, rsb, csb, c, ldc, alpha, beta),
            _ => {}
        }
    }
}

/// An `MR × NR` tile of [`Real::direct_complex`], portable: the real and the imaginary parts of the
/// sums apart, four real products a step and element.
///
/// # Safety
///
/// As for [`Real::direct`], for the tile.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn complex_tile<R: Real, const MR: usize, const NR: usize, const CJ: bool>(
    k: usize,
    a: *const Complex<R>,
    lda: usize,
    b: *const Complex<R>,
    rsb: usize,
    csb: usize,
    c: *mut Complex<R>,
    ldc: usize,
    alpha: Complex<R>,
    beta: Complex<R>,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut re = [[R::ZERO; MR]; NR];
        let mut im = [[R::ZERO; MR]; NR];
        for p in 0..k {
            let mut col = [Complex::<R>::ZERO; MR];
            for (i, x) in col.iter_mut().enumerate() {
                *x = *a.add(i + p * lda);
            }
            for j in 0..NR {
                let y = *b.add(p * rsb + j * csb);
                let yi = if CJ { -y.im } else { y.im };
                for (i, x) in col.iter().enumerate() {
                    re[j][i] = R::fma(R::fma(re[j][i], x.re, y.re), -x.im, yi);
                    im[j][i] = R::fma(R::fma(im[j][i], x.re, yi), x.im, y.re);
                }
            }
        }
        for j in 0..NR {
            for i in 0..MR {
                let x = c.add(i + j * ldc);
                let s = alpha * Complex::new(re[j][i], im[j][i]);
                *x = if beta == Complex::ZERO {
                    s
                } else {
                    s + beta * *x
                };
            }
        }
    }
}

/// The tiles of [`complex_tile`] of `MR` rows from row `i0` on, portable: four columns at a time,
/// then the rest.
///
/// # Safety
///
/// As for [`Real::direct_complex`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn complex_tiles<R: Real, const MR: usize, const CJ: bool>(
    n: usize,
    k: usize,
    a: *const Complex<R>,
    lda: usize,
    b: *const Complex<R>,
    rsb: usize,
    csb: usize,
    c: *mut Complex<R>,
    ldc: usize,
    alpha: Complex<R>,
    beta: Complex<R>,
) {
    // SAFETY: as the caller guarantees; each tile is inside the block.
    unsafe {
        let mut j = 0;
        while j + 4 <= n {
            let (bj, cj) = (b.add(j * csb), c.add(j * ldc));
            complex_tile::<R, MR, 4, CJ>(k, a, lda, bj, rsb, csb, cj, ldc, alpha, beta);
            j += 4;
        }
        // Past the end when no columns are left, so not dereferenced then.
        let (b, c) = (b.wrapping_add(j * csb), c.wrapping_add(j * ldc));
        match n - j {
            3 => complex_tile::<R, MR, 3, CJ>(k, a, lda, b, rsb, csb, c, ldc, alpha, beta),
            2 => complex_tile::<R, MR, 2, CJ>(k, a, lda, b, rsb, csb, c, ldc, alpha, beta),
            1 => complex_tile::<R, MR, 1, CJ>(k, a, lda, b, rsb, csb, c, ldc, alpha, beta),
            _ => {}
        }
    }
}

/// Whether [`Real::direct`] computes the product rather than [`blocked`]: when it is short enough
/// that packing would cost more than it saves, `op(a)` small enough to be transposed on the stack
/// if it must be, and the columns the direct path reads again and again close enough together not
/// to evict each other from the cache. Columns a large power of two apart fall into the same few
/// sets of the L1 cache: with a leading dimension of 2048, a direct `f64` product of 32 × 32 × 32
/// ran at 26 GFLOP/s, against 40 by blocks and 56 with dense columns.
#[allow(clippy::too_many_arguments)]
#[inline]
fn is_small(
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    opa: Op,
    ldb: usize,
    opb: Op,
    ldc: usize,
) -> bool {
    if m.saturating_mul(n).saturating_mul(k) > SMALL_MNK || (opa != Op::N && m * k > SMALL_A) {
        return false;
    }
    let a = if opa == Op::N {
        lda * (k - 1) + m
    } else {
        m * k
    };
    let b = if opb == Op::N {
        ldb * (n - 1) + k
    } else {
        ldb * (k - 1) + n
    };
    let c = ldc * (n - 1) + m;
    k <= 8 || a.saturating_add(b).saturating_add(c) <= SMALL_SPAN
}

/// The elements of `op(a)` the direct path transposes on the stack.
const SMALL_A: usize = 4096;

/// The products `m · n · k` of the direct path.
const SMALL_MNK: usize = 64 * 64 * 64;

/// The elements the three blocks of the direct path may span: three of 64 × 64.
const SMALL_SPAN: usize = 3 * 64 * 64;

/// Runs `f` on `op(a)` held as `m × k` column-major, with its leading dimension: `a` itself when
/// `op` is [`Op::N`], and otherwise its transpose (conjugated for [`Op::C`]) in a buffer on the
/// stack.
///
/// # Safety
///
/// The block of `a` must be valid for reads; `m · k <= SMALL_A` when `op` is not [`Op::N`].
#[inline(always)]
unsafe fn with_a<E: Copy + Conj, X>(
    m: usize,
    k: usize,
    a: *const E,
    lda: usize,
    op: Op,
    f: impl FnOnce(*const E, usize) -> X,
) -> X {
    if op == Op::N {
        f(a, lda)
    } else {
        // SAFETY: as the caller guarantees.
        unsafe { transposed(m, k, a, lda, op, f) }
    }
}

/// [`with_a`] of a transposed `op(a)`, out of line so that the frames of the others do not hold
/// its buffer.
///
/// # Safety
///
/// As for [`with_a`].
#[inline(never)]
unsafe fn transposed<E: Copy + Conj, X>(
    m: usize,
    k: usize,
    a: *const E,
    lda: usize,
    op: Op,
    f: impl FnOnce(*const E, usize) -> X,
) -> X {
    let mut t = [const { MaybeUninit::<E>::uninit() }; SMALL_A];
    let d = t.as_mut_ptr().cast::<E>();
    // SAFETY: every read is of an element of the block, and every write inside the buffer, which
    // then holds the `m · k` elements of `op(a)`.
    unsafe {
        for i in 0..m {
            let s = a.add(i * lda);
            for p in 0..k {
                let x = *s.add(p);
                *d.add(i + p * m) = if op == Op::C { x.conj() } else { x };
            }
        }
    }
    f(d, m)
}

/// The complex conjugate, of real and complex elements.
trait Conj {
    /// The complex conjugate.
    fn conj(self) -> Self;
}

impl<R: Real> Conj for R {
    #[inline(always)]
    fn conj(self) -> Self {
        self
    }
}

impl<R: Real> Conj for Complex<R> {
    #[inline(always)]
    fn conj(self) -> Self {
        Complex::conj(self)
    }
}

/// `c = beta · c + alpha · op(a) · op(b)` for an `op(a)` of few rows: `op(a)` is copied dense a
/// block of [`KC`](Real::KC) columns at a time, and [`Real::direct`] reads `op(b)`, the larger
/// operand, where it is, once for each group of rows. Packing `op(b)` would cost a copy of it for
/// the few panels of `a` that use it.
///
/// # Safety
///
/// `m`, `n`, `k >= 1`; the blocks must be valid for their reads and `c` for its writes.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn skinny<R: Real>(
    m: usize,
    n: usize,
    k: usize,
    alpha: R,
    a: *const R,
    lda: usize,
    opa: Op,
    b: *const R,
    ldb: usize,
    opb: Op,
    beta: R,
    c: *mut R,
    ldc: usize,
) {
    let kcb = k.div_ceil(k.div_ceil(R::KC));
    let (rsb, csb) = if opb == Op::N { (1, ldb) } else { (ldb, 1) };
    let mut buf: Vec<R> = Vec::with_capacity(m * kcb);
    let d = buf.as_mut_ptr();
    // SAFETY: every read is of an element of the blocks, the buffer holds the `m · kc` elements
    // copied before the direct product reads them, and `c` is written inside its block.
    unsafe {
        let mut pc = 0;
        while pc < k {
            let kc = kcb.min(k - pc);
            if opa == Op::N {
                for p in 0..kc {
                    ptr::copy_nonoverlapping(a.add((pc + p) * lda), d.add(p * m), m);
                }
            } else {
                for i in 0..m {
                    let s = a.add(pc + i * lda);
                    for p in 0..kc {
                        *d.add(i + p * m) = *s.add(p);
                    }
                }
            }
            let beta = if pc == 0 { beta } else { R::ONE };
            R::direct(
                m,
                n,
                kc,
                alpha,
                d,
                m,
                b.add(pc * rsb),
                rsb,
                csb,
                beta,
                c,
                ldc,
            );
            pc += kc;
        }
    }
}

/// `c = beta · c` on the `m × n` block, without reading `c` when `beta` is zero.
///
/// # Safety
///
/// The block must be valid for reads and writes.
unsafe fn scale<E: Elem + core::ops::Mul<Output = E>>(
    m: usize,
    n: usize,
    beta: E,
    c: *mut E,
    ldc: usize,
) {
    if beta == E::ONE {
        return;
    }
    // SAFETY: as the caller guarantees.
    unsafe {
        for j in 0..n {
            let col = c.add(j * ldc);
            for i in 0..m {
                *col.add(i) = if beta == E::ZERO {
                    E::ZERO
                } else {
                    *col.add(i) * beta
                };
            }
        }
    }
}

/// Panics unless the elements of a nonempty `rows × cols` block with the leading dimension `ld`
/// lie in a slice of `len`: an empty block may have any leading dimension.
#[inline]
#[track_caller]
fn check(len: usize, rows: usize, cols: usize, ld: usize) {
    if rows > 0 && cols > 0 {
        assert!(ld >= rows, "the leading dimension is shorter than a column");
        let end = (cols - 1).checked_mul(ld).and_then(|x| x.checked_add(rows));
        assert!(
            end.is_some_and(|e| e <= len),
            "the block is outside the slice"
        );
    }
}

/// [`gemm`](super::blas::gemm) of real floats.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub(super) fn real<R: Real>(
    m: usize,
    n: usize,
    k: usize,
    alpha: R,
    a: &[R],
    lda: usize,
    opa: Op,
    b: &[R],
    ldb: usize,
    opb: Op,
    beta: R,
    c: &mut [R],
    ldc: usize,
) {
    let (ar, ac) = if opa == Op::N { (m, k) } else { (k, m) };
    let (br, bc) = if opb == Op::N { (k, n) } else { (n, k) };
    check(a.len(), ar, ac, lda);
    check(b.len(), br, bc, ldb);
    check(c.len(), m, n, ldc);
    if m == 0 || n == 0 {
        return;
    }
    let (a, b, c) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    // SAFETY: the blocks lie in the slices, as checked.
    unsafe {
        if k == 0 || alpha == R::ZERO {
            scale(m, n, beta, c, ldc);
        } else if is_small(m, n, k, lda, opa, ldb, opb, ldc) {
            let (rsb, csb) = if opb == Op::N { (1, ldb) } else { (ldb, 1) };
            with_a(m, k, a, lda, opa, |a, lda| {
                R::direct(m, n, k, alpha, a, lda, b, rsb, csb, beta, c, ldc)
            });
        } else if m <= R::SKINNY {
            skinny(m, n, k, alpha, a, lda, opa, b, ldb, opb, beta, c, ldc);
        } else {
            blocked(m, n, k, alpha, a, lda, opa, b, ldb, opb, beta, c, ldc);
        }
    }
}

/// [`gemm`](super::blas::gemm) of complex numbers.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub(super) fn complex<R: Real>(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex<R>,
    a: &[Complex<R>],
    lda: usize,
    opa: Op,
    b: &[Complex<R>],
    ldb: usize,
    opb: Op,
    beta: Complex<R>,
    c: &mut [Complex<R>],
    ldc: usize,
) {
    let (ar, ac) = if opa == Op::N { (m, k) } else { (k, m) };
    let (br, bc) = if opb == Op::N { (k, n) } else { (n, k) };
    check(a.len(), ar, ac, lda);
    check(b.len(), br, bc, ldb);
    check(c.len(), m, n, ldc);
    if m == 0 || n == 0 {
        return;
    }
    let (a, b, c) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    // SAFETY: the blocks lie in the slices, as checked; `Complex` is `repr(C)`, its parts in
    // turn, so `c` holds `2 · ldc` reals a column.
    unsafe {
        if k == 0 || alpha == Complex::ZERO {
            scale(m, n, beta, c, ldc);
            return;
        }
        if is_small(m, n, k, lda, opa, ldb, opb, ldc) {
            let (rsb, csb) = if opb == Op::N { (1, ldb) } else { (ldb, 1) };
            with_a(m, k, a, lda, opa, |a, lda| {
                if opb == Op::C {
                    R::direct_complex::<true>(m, n, k, alpha, a, lda, b, rsb, csb, beta, c, ldc);
                } else {
                    R::direct_complex::<false>(m, n, k, alpha, a, lda, b, rsb, csb, beta, c, ldc);
                }
            });
            return;
        }
        // A `beta` off the real line scales `c` first, which the real kernel cannot.
        let beta = if beta.im == R::ZERO {
            beta.re
        } else {
            scale(m, n, beta, c, ldc);
            R::ONE
        };
        blocked(
            m,
            n,
            k,
            alpha,
            a,
            lda,
            opa,
            b,
            ldb,
            opb,
            beta,
            c.cast(),
            2 * ldc,
        );
    }
}

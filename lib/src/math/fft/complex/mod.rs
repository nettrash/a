//! The complex transforms behind a plan.
//!
//! * Stockham passes ([`pass`]) of the radices 2, 4 and 8 (powers of two), 3, 5 and 7 (hard-coded
//!   butterflies), the primes from 11 to [`MAX_GENERIC`] (the odd butterfly from tables), and a
//!   larger prime factor by its own plan applied to the gathered sequences. No bit reversal; the
//!   passes alternate between the data and one buffer of the same size. Measured on an Apple M4,
//!   this stays faster up to `2^22` than a four-step (six-step) algorithm with blocks of columns
//!   in the caches (about twice as slow as written), so it is used for every length.
//! * Rader's algorithm for a prime `p`: the cyclic convolution of length `p − 1` by transforms of
//!   that length, the transform of the permuted roots computed once.
//! * Bluestein's algorithm: the transform as the convolution with a chirp, by transforms of a
//!   smooth length `m ≥ 2n − 1`.
//!
//! The inverse runs the same code with the conjugate roots (the `INV` kernels), or, for Rader and
//! Bluestein, the forward transform written to the reversed indices: `X_inv[k] = X[−k mod n]`.

mod blk;
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
mod neon;
mod pass;
#[cfg(all(target_arch = "x86_64", not(miri)))]
mod x86;

use super::twiddle;
use crate::math::complex::{Complex, Float};
use pass::{CODELETS, Form, Kern, Kset, MAX_GENERIC, Pass};
pub(crate) use pass::{HKern, RKern, SFwd, SInv};

/// The instruction set the kernels of a plan were chosen for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    /// Portable Rust (and Miri).
    Portable,
    /// NEON on little-endian aarch64.
    Neon,
    /// SSE2 on x86_64.
    Sse2,
    /// AVX2 and FMA on x86_64, detected at run time.
    Avx2,
}

impl Isa {
    /// The best instruction set of this machine.
    pub(crate) fn detect() -> Isa {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        {
            Isa::Neon
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            if !cfg!(a_baseline)
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
            {
                Isa::Avx2
            } else {
                Isa::Sse2
            }
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_endian = "little", not(miri)),
            all(target_arch = "x86_64", not(miri))
        )))]
        {
            Isa::Portable
        }
    }
}

/// The kernels of a float type: implemented for `f32` and `f64`.
pub trait Kernels: Float {
    /// The kernels for the instruction set.
    #[doc(hidden)]
    fn kset(isa: Isa) -> &'static Kset<Self>;
}

/// Instantiates the kernels for one float type and set of blocks: `W` the widest, `H` and `N`
/// for what is left of a row, `P` for the passes along `p`; with the target features given.
macro_rules! kset {
    ($name:ident, $t:ty, $w:ty, $h:ty, $n:ty, $p:ty $(, $feat:literal)?) => {
        mod $name {
            use super::blk::Blk;
            use super::pass::{self, Form, HKern, Kern, Kset, Pass, RKern, SKerns};
            use crate::math::complex::Complex;

            $(#[target_feature(enable = $feat)])?
            unsafe fn q<const R: usize, const INV: bool>(
                d: &Pass<$t>,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
                count: usize,
                stride: usize,
            ) {
                for c in 0..count {
                    // SAFETY: the contract of the kernels; the features were detected.
                    unsafe {
                        pass::q_pass::<$t, $w, $h, $n, R, INV>(
                            d,
                            x.add(c * stride),
                            y.add(c * stride),
                        )
                    }
                }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn l<const R: usize, const INV: bool>(
                d: &Pass<$t>,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
                count: usize,
                stride: usize,
            ) {
                for c in 0..count {
                    // SAFETY: as for `q`.
                    unsafe {
                        pass::l_pass::<$t, $w, $h, $n, R, INV>(
                            d,
                            x.add(c * stride),
                            y.add(c * stride),
                        )
                    }
                }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn p<const R: usize, const INV: bool>(
                d: &Pass<$t>,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
                count: usize,
                stride: usize,
            ) {
                for c in 0..count {
                    // SAFETY: as for `q`.
                    unsafe {
                        pass::p_pass::<$t, $p, $n, R, INV>(d, x.add(c * stride), y.add(c * stride))
                    }
                }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn pm<const INV: bool>(x: *mut Complex<$t>, w: *const $t, len: usize) {
                // SAFETY: as for `q`.
                unsafe { pass::pmul::<$t, $w, $n, INV>(x, w, len) }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn b<const R: usize, const INV: bool>(
                d: &Pass<$t>,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
                count: usize,
                stride: usize,
            ) {
                // SAFETY: as for `q`.
                unsafe { pass::b_pass::<$t, $p, $n, R, INV>(&d.g, x, y, count, stride) }
            }

            fn sel<const R: usize>(form: Form, inv: bool) -> Kern<$t> {
                match (form, inv) {
                    (Form::L, false) => l::<R, false>,
                    (Form::L, true) => l::<R, true>,
                    (Form::B, false) => b::<R, false>,
                    (Form::B, true) => b::<R, true>,
                    (Form::P, false) => p::<R, false>,
                    (Form::P, true) => p::<R, true>,
                    (_, false) => q::<R, false>,
                    (_, true) => q::<R, true>,
                }
            }

            /// The kernels of a composite butterfly: a whole transform, L or B form only.
            fn sel_short<const R: usize>(form: Form, inv: bool) -> Kern<$t> {
                match (form, inv) {
                    (Form::B, false) => b::<R, false>,
                    (Form::B, true) => b::<R, true>,
                    (_, false) => l::<R, false>,
                    (_, true) => l::<R, true>,
                }
            }

            fn kern(r: usize, form: Form, inv: bool) -> Kern<$t> {
                match r {
                    6 => sel_short::<6>(form, inv),
                    9 => sel_short::<9>(form, inv),
                    10 => sel_short::<10>(form, inv),
                    12 => sel_short::<12>(form, inv),
                    15 => sel_short::<15>(form, inv),
                    16 => sel_short::<16>(form, inv),
                    20 => sel_short::<20>(form, inv),
                    24 => sel_short::<24>(form, inv),
                    25 => sel_short::<25>(form, inv),
                    2 => sel::<2>(form, inv),
                    3 => sel::<3>(form, inv),
                    4 => sel::<4>(form, inv),
                    5 => sel::<5>(form, inv),
                    7 => sel::<7>(form, inv),
                    8 => sel::<8>(form, inv),
                    11 => sel::<11>(form, inv),
                    13 => sel::<13>(form, inv),
                    17 => sel::<17>(form, inv),
                    19 => sel::<19>(form, inv),
                    23 => sel::<23>(form, inv),
                    _ => unreachable!("no butterfly of radix {r}"),
                }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn rf<const M: usize>(
                g: &[$t],
                tw: *const $t,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
            ) {
                // SAFETY: the contract of the fused kernels; the features were detected.
                unsafe { pass::real_fwd::<$t, $n, M>(g, tw, x, y) }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn ri<const M: usize>(
                g: &[$t],
                tw: *const $t,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
            ) {
                // SAFETY: as for `rf`.
                unsafe { pass::real_inv::<$t, $n, M>(g, tw, x, y) }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn rof<const M: usize>(
                g: &[$t],
                tw: *const $t,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
            ) {
                // SAFETY: as for `rf`.
                unsafe { pass::real_odd_fwd::<$t, $n, M>(g, tw, x, y) }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn roi<const M: usize>(
                g: &[$t],
                tw: *const $t,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
            ) {
                // SAFETY: as for `rf`.
                unsafe { pass::real_odd_inv::<$t, $n, M>(g, tw, x, y) }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn sf<const R: usize>(
                g: &[$t],
                tw: *const $t,
                zb: *const Complex<$t>,
                yl: *const Complex<$t>,
                m: usize,
                y: *mut Complex<$t>,
            ) {
                // SAFETY: as for `rf`.
                unsafe { pass::split_fwd::<$t, $n, R>(g, tw, zb, yl, m, y) }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn si<const R: usize>(
                g: &[$t],
                tw: *const $t,
                x: *const Complex<$t>,
                m: usize,
                zb: *mut Complex<$t>,
                ul: *mut Complex<$t>,
            ) {
                // SAFETY: as for `rf`.
                unsafe { pass::split_inv::<$t, $n, R>(g, tw, x, m, zb, ul) }
            }

            $(#[target_feature(enable = $feat)])?
            unsafe fn hl<const R: usize>(
                d: &Pass<$t>,
                x: *const Complex<$t>,
                y: *mut Complex<$t>,
                tw: *const $t,
            ) {
                // SAFETY: as for `rf`.
                unsafe { pass::half_last::<$t, $n, R>(d, x, y, tw) }
            }

            fn half_last(r: usize) -> Option<HKern<$t>> {
                Some(match r {
                    2 => hl::<2>,
                    3 => hl::<3>,
                    4 => hl::<4>,
                    5 => hl::<5>,
                    8 => hl::<8>,
                    _ => return None,
                })
            }

            fn split(r: usize) -> Option<SKerns<$t>> {
                Some(match r {
                    3 => (sf::<3>, si::<3>),
                    5 => (sf::<5>, si::<5>),
                    7 => (sf::<7>, si::<7>),
                    9 => (sf::<9>, si::<9>),
                    11 => (sf::<11>, si::<11>),
                    13 => (sf::<13>, si::<13>),
                    15 => (sf::<15>, si::<15>),
                    17 => (sf::<17>, si::<17>),
                    19 => (sf::<19>, si::<19>),
                    _ => return None,
                })
            }

            fn real_odd(m: usize) -> Option<[RKern<$t>; 2]> {
                Some(match m {
                    3 => [rof::<3>, roi::<3>],
                    5 => [rof::<5>, roi::<5>],
                    7 => [rof::<7>, roi::<7>],
                    9 => [rof::<9>, roi::<9>],
                    11 => [rof::<11>, roi::<11>],
                    13 => [rof::<13>, roi::<13>],
                    15 => [rof::<15>, roi::<15>],
                    _ => return None,
                })
            }

            fn real(m: usize) -> Option<[RKern<$t>; 2]> {
                Some(match m {
                    2 => [rf::<2>, ri::<2>],
                    3 => [rf::<3>, ri::<3>],
                    4 => [rf::<4>, ri::<4>],
                    5 => [rf::<5>, ri::<5>],
                    6 => [rf::<6>, ri::<6>],
                    7 => [rf::<7>, ri::<7>],
                    8 => [rf::<8>, ri::<8>],
                    9 => [rf::<9>, ri::<9>],
                    10 => [rf::<10>, ri::<10>],
                    11 => [rf::<11>, ri::<11>],
                    12 => [rf::<12>, ri::<12>],
                    13 => [rf::<13>, ri::<13>],
                    15 => [rf::<15>, ri::<15>],
                    16 => [rf::<16>, ri::<16>],
                    17 => [rf::<17>, ri::<17>],
                    19 => [rf::<19>, ri::<19>],
                    20 => [rf::<20>, ri::<20>],
                    24 => [rf::<24>, ri::<24>],
                    25 => [rf::<25>, ri::<25>],
                    // None for 32 (64 reals, 32 = 8·4 in one butterfly): measured on an Apple
                    // M4 Max, the transform of 32 and the pair pass take 0.87 (forward) and 0.97
                    // (inverse) of the fused kernel's time in `f64`, 0.69 to 0.97 in `f32`.
                    _ => return None,
                })
            }

            pub(crate) static KSET: Kset<$t> = Kset {
                real,
                real_odd,
                split,
                half_last,
                pass: kern,
                pmul: [pm::<false>, pm::<true>],
                wide: <$w as Blk<$t>>::K,
                plane: <$p as Blk<$t>>::K,
            };
        }
    };
}

kset!(
    port32,
    f32,
    super::blk::Sc<f32>,
    super::blk::Sc<f32>,
    super::blk::Sc<f32>,
    super::blk::Sc<f32>
);
kset!(
    port64,
    f64,
    super::blk::Sc<f64>,
    super::blk::Sc<f64>,
    super::blk::Sc<f64>,
    super::blk::Sc<f64>
);
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
kset!(
    neon32,
    f32,
    super::neon::F2,
    super::neon::F1,
    super::neon::F1,
    super::neon::F2
);
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
kset!(
    neon64,
    f64,
    super::neon::D1,
    super::neon::D1,
    super::neon::D1,
    super::neon::D1
);
#[cfg(all(target_arch = "x86_64", not(miri)))]
kset!(
    sse32,
    f32,
    super::x86::S2,
    super::x86::S1,
    super::x86::S1,
    super::x86::S2
);
#[cfg(all(target_arch = "x86_64", not(miri)))]
kset!(
    sse64,
    f64,
    super::x86::P1,
    super::x86::P1,
    super::x86::P1,
    super::x86::P1
);
#[cfg(all(target_arch = "x86_64", not(miri)))]
kset!(
    avx32,
    f32,
    super::x86::A4,
    super::x86::S2,
    super::x86::S1,
    super::x86::S2,
    "avx2,fma"
);
#[cfg(all(target_arch = "x86_64", not(miri)))]
kset!(
    avx64,
    f64,
    super::x86::A2,
    super::x86::P1,
    super::x86::P1,
    super::x86::A2,
    "avx2,fma"
);

impl Kernels for f32 {
    fn kset(isa: Isa) -> &'static Kset<f32> {
        match isa {
            #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
            Isa::Neon => &neon32::KSET,
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Isa::Sse2 => &sse32::KSET,
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Isa::Avx2 => &avx32::KSET,
            _ => &port32::KSET,
        }
    }
}

impl Kernels for f64 {
    fn kset(isa: Isa) -> &'static Kset<f64> {
        match isa {
            #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
            Isa::Neon => &neon64::KSET,
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Isa::Sse2 => &sse64::KSET,
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Isa::Avx2 => &avx64::KSET,
            _ => &port64::KSET,
        }
    }
}

/// The prime factors of `n`, smallest first (none for `n = 1`).
pub(crate) fn factors(mut n: usize) -> Vec<usize> {
    let mut f = Vec::new();
    let mut p = 2;
    while p * p <= n {
        while n.is_multiple_of(p) {
            f.push(p);
            n /= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// A forward and inverse transform of one length.
pub(crate) enum Algo<T> {
    /// Length 0 or 1: the identity.
    Trivial,
    /// Stockham passes.
    Stock(Stock<T>),
    /// Rader's algorithm for a prime length.
    Rader(Box<Rader<T>>),
    /// Bluestein's algorithm.
    Blue(Box<Blue<T>>),
}

/// The passes of a Stockham transform of `n` values (of each of `s0` interleaved sequences).
pub(crate) struct Stock<T> {
    /// The values transformed: the length times `s0`.
    len: usize,
    /// The transforms run pass by pass together in a call at most: as many as fit in
    /// [`BATCH_VALUES`] (when the caller's working memory has room for them).
    batch: usize,
    passes: Vec<Pass<T>>,
    /// The working memory of the passes by sub-plans, after the `len` values of the second
    /// buffer.
    extra: usize,
}

/// The values a call of the Stockham passes takes at once: many short transforms of a batch run
/// pass by pass together, the kernels called once for all of them.
const BATCH_VALUES: usize = 256;

/// Rader's algorithm for the prime `p`.
pub(crate) struct Rader<T> {
    p: usize,
    /// The transform of the length `p − 1`.
    inner: Algo<T>,
    /// `g^q mod p` and `g^(−q) mod p` for `q < p − 1`, `g` a primitive root.
    gin: Vec<usize>,
    gout: Vec<usize>,
    /// The transform of `W_p^(g^(−q))`, divided by `p − 1`, in the form of `pmul`.
    kernel: Vec<T>,
    pmul: unsafe fn(*mut Complex<T>, *const T, usize),
}

/// Bluestein's algorithm for the length `n` by transforms of the length `m`.
pub(crate) struct Blue<T> {
    n: usize,
    m: usize,
    inner: Algo<T>,
    /// The chirp `W_2n^(k²)`, `k < n`, and the transform of its conjugate's circular extension
    /// to `m`, divided by `m`; both in the form of `pmul`.
    chirp: Vec<T>,
    kernel: Vec<T>,
    pmul: unsafe fn(*mut Complex<T>, *const T, usize),
}

/// A pass of a large prime radix `r` run on blocks of `b` of its interleaved sequences at once
/// (the `s` sequences of one `p` are adjacent): Rader's or Bluestein's algorithm with the inner
/// transform planned for `b` interleaved sequences, so the vectors run across the block.
pub(crate) struct SubBatch<T> {
    b: usize,
    kind: SubKind<T>,
    /// The inner transform of `b` interleaved sequences.
    inner: Stock<T>,
    /// The inner kernel repeated for the `b` sequences, in the form of `pmul`.
    kernel: Vec<T>,
    pmul: unsafe fn(*mut Complex<T>, *const T, usize),
}

enum SubKind<T> {
    Rader { gin: Vec<usize>, gout: Vec<usize> },
    Blue { m: usize, chirp: Vec<Complex<T>> },
}

impl<T: Kernels> SubBatch<T> {
    /// The batched form of the pass of radix `r` by `sub`, for `s` sequences, if it applies:
    /// at least 2 sequences a block and an inner transform by passes.
    fn new(r: usize, s: usize, sub: &Algo<T>, ks: &'static Kset<T>) -> Option<SubBatch<T>> {
        let b = (2..=16).rev().find(|&b| s.is_multiple_of(b))?;
        let (kind, len, raw) = match sub {
            Algo::Rader(rd) if matches!(rd.inner, Algo::Stock(_)) => {
                let (gin, gout) = (rd.gin.clone(), rd.gout.clone());
                (SubKind::Rader { gin, gout }, r - 1, &rd.kernel)
            }
            Algo::Blue(bl) if matches!(bl.inner, Algo::Stock(_)) => {
                let chirp = (0..r).map(|k| pmul_value(&bl.chirp, k, ks.wide)).collect();
                (SubKind::Blue { m: bl.m, chirp }, bl.m, &bl.kernel)
            }
            _ => return None,
        };
        let rep: Vec<Complex<T>> = (0..len)
            .flat_map(|i| core::iter::repeat_n(pmul_value(raw, i, ks.wide), b))
            .collect();
        Some(SubBatch {
            b,
            kind,
            inner: Stock::new(len, b, &radices(len), ks),
            kernel: pmul_table(&rep, ks.wide),
            pmul: ks.pmul[0],
        })
    }

    /// The working memory of a call: the block, the inner transform's, and `b` values.
    fn scratch(&self) -> usize {
        self.inner.len + self.inner.need(1) + self.b
    }

    /// The pass `d` from `x` to `y` by blocks of sequences, with `scr` of [`scratch`] values.
    ///
    /// # Safety
    /// As for [`sub_pass`].
    unsafe fn run(
        &self,
        d: &Pass<T>,
        x: *const Complex<T>,
        y: *mut Complex<T>,
        scr: *mut Complex<T>,
        inv: bool,
    ) {
        let (r, m, s, b) = (d.r, d.m, d.s, self.b);
        let ms = m * s;
        let len = self.inner.len;
        // SAFETY: the caller's contract; every index is within the pass's values and `scr`.
        unsafe {
            let buf = scr;
            let tmp = buf.add(len);
            let x0 = tmp.add(self.inner.need(1));
            for p in 0..m {
                let tw = d.tw.as_ptr().add(4 * (r - 1) * p);
                let w = |j: usize| -> Complex<T> {
                    if p == 0 || j == 0 {
                        Complex::ONE
                    } else {
                        let (wr, wi) = (*tw.add(4 * (j - 1)), *tw.add(4 * (j - 1) + 3));
                        Complex::new(wr, if inv { -wi } else { wi })
                    }
                };
                for q0 in (0..s).step_by(b) {
                    let xp = x.add(q0 + s * p);
                    let yp = y.add(q0 + s * r * p);
                    match &self.kind {
                        SubKind::Rader { gin, gout } => {
                            for l in 0..b {
                                *x0.add(l) = *xp.add(l);
                            }
                            for (k, &i) in gin.iter().enumerate() {
                                let (src, dst) = (xp.add(ms * i), buf.add(k * b));
                                for l in 0..b {
                                    *dst.add(l) = *src.add(l);
                                }
                            }
                            self.inner.run(buf, buf, tmp, 1, false);
                            for l in 0..b {
                                *yp.add(l) = *x0.add(l) + *buf.add(l);
                            }
                            (self.pmul)(buf, self.kernel.as_ptr(), len);
                            self.inner.run(buf, buf, tmp, 1, true);
                            for (k, &o) in gout.iter().enumerate() {
                                let j = if inv { r - o } else { o };
                                let wj = w(j);
                                let out = yp.add(s * j);
                                for l in 0..b {
                                    *out.add(l) = (*buf.add(k * b + l) + *x0.add(l)) * wj;
                                }
                            }
                        }
                        SubKind::Blue { m: bm, chirp } => {
                            for (k, &c) in chirp.iter().enumerate() {
                                for l in 0..b {
                                    *buf.add(k * b + l) = *xp.add(ms * k + l) * c;
                                }
                            }
                            core::ptr::write_bytes(buf.add(r * b), 0, (bm - r) * b);
                            self.inner.run(buf, buf, tmp, 1, false);
                            (self.pmul)(buf, self.kernel.as_ptr(), len);
                            self.inner.run(buf, buf, tmp, 1, true);
                            for (k, &c) in chirp.iter().enumerate() {
                                let j = if inv { (r - k) % r } else { k };
                                let wj = c * w(j);
                                let out = yp.add(s * j);
                                for l in 0..b {
                                    *out.add(l) = *buf.add(k * b + l) * wj;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// SAFETY: the plans hold tables and fn pointers only, never changed after planning; the raw
// pointers the kernels take are made from the caller's slices at each call.
unsafe impl<T: Send> Send for Algo<T> {}
// SAFETY: as above; nothing is mutated through a shared plan.
unsafe impl<T: Sync> Sync for Algo<T> {}

/// The table of the factors `ws` in the form `pmul` takes for the widest blocks of `wide`
/// values: lane form for the whole blocks, compact for the rest.
fn pmul_table<T: Float>(ws: &[Complex<T>], wide: usize) -> Vec<T> {
    let mut t = vec![T::ZERO; 4 * ws.len()];
    let whole = ws.len() / wide * wide;
    for (b, o) in ws[..whole]
        .chunks_exact(wide)
        .zip(t.chunks_exact_mut(4 * wide))
    {
        for (l, w) in b.iter().enumerate() {
            o[2 * l] = w.re;
            o[2 * l + 1] = w.re;
            o[2 * wide + 2 * l] = -w.im;
            o[2 * wide + 2 * l + 1] = w.im;
        }
    }
    for (w, o) in ws[whole..]
        .iter()
        .zip(t[4 * whole..].as_chunks_mut::<4>().0)
    {
        *o = [w.re, w.re, -w.im, w.im];
    }
    t
}

/// The value `i` of a table made by [`pmul_table`] for `wide`.
fn pmul_value<T: Float>(t: &[T], i: usize, wide: usize) -> Complex<T> {
    let whole = t.len() / 4 / wide * wide;
    if i < whole {
        let (b, l) = (i / wide, i % wide);
        let base = 4 * wide * b;
        Complex::new(t[base + 2 * l], t[base + 2 * wide + 2 * l + 1])
    } else {
        Complex::new(t[4 * i], t[4 * i + 3])
    }
}

/// The radices of the Stockham passes of `n`, in the order of the passes, and whether `n`
/// has a prime factor above [`MAX_GENERIC`] (taken by a sub-plan pass).
fn radices(n: usize) -> Vec<usize> {
    let f = factors(n);
    let twos = f.iter().filter(|&&p| p == 2).count();
    let mut out = Vec::new();
    out.extend(pow2_radices(twos));
    // Then the odd factors, the hard-coded ones first and larger ones after.
    out.extend(f.iter().filter(|&&p| p != 2));
    out
}

/// The radices of `2^t`: radix 8, with a 4 for two left and 4·4 for one left (2 only for
/// `t = 1`).
fn pow2_radices(t: usize) -> Vec<usize> {
    let mut out = Vec::new();
    match t % 3 {
        0 => out.extend(core::iter::repeat_n(8, t / 3)),
        1 if t >= 4 => {
            out.extend(core::iter::repeat_n(8, t / 3 - 1));
            out.extend([4, 4]);
        }
        1 => out.push(2),
        _ => {
            out.extend(core::iter::repeat_n(8, t / 3));
            out.push(4);
        }
    }
    out
}

/// The estimated cost of a pass of radix `r` per value, in the units of [`cost`] (measured on
/// an Apple M4 in `f64`: about 0.25 ns a unit).
fn pass_cost(r: usize) -> f64 {
    match r {
        2 => 1.0,
        3 => 1.35,
        4 => 1.2,
        5 => 1.7,
        7 => 1.9,
        8 => 1.5,
        r if r <= MAX_GENERIC => 0.4 * r as f64,
        _ => f64::INFINITY,
    }
}

/// The fixed cost of a pass (a call of its kernel), in the units of [`cost`].
const PASS_FIXED: f64 = 30.0;

/// The largest prime length transformed by one pass of the odd radix from tables; Rader's or
/// Bluestein's algorithm above it (measured: at 17 one pass takes 17.6 ns against Rader's 51;
/// from 23 the pass runs out of registers and Rader's is faster).
const MAX_PRIME_PASS: usize = 19;

/// The estimated time of a transform of the length `n`, in the time of a radix-2 pass over
/// one value, with the choice of algorithm it implies for a prime factor above
/// [`MAX_GENERIC`] (or a prime length above [`MAX_PRIME_PASS`]).
fn cost(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let f = factors(n);
    let big = *f.last().expect("n >= 2");
    if f.len() == 1 && n > MAX_PRIME_PASS {
        return prime_choice(n).1;
    }
    if big <= MAX_GENERIC {
        return radices(n)
            .iter()
            .map(|&r| n as f64 * pass_cost(r) + PASS_FIXED)
            .sum();
    }
    // A smooth part by passes and each large prime by a sub-plan per gathered sequence,
    // against Bluestein for the whole.
    let mut c = 0.0;
    for &r in &radices(n) {
        if r <= MAX_GENERIC {
            c += n as f64 * pass_cost(r) + PASS_FIXED;
        } else {
            c += (n / r) as f64 * (cost(r) + 5.0 * r as f64);
        }
    }
    c.min(bluestein_cost(n).1)
}

/// [`cost`] of the smooth length `m = 2^t2·3^t3·5^t5·7^t7`, from the exponents (no factoring).
fn smooth_cost(m: usize, t2: usize, t3: usize, t5: usize, t7: usize) -> f64 {
    let (eights, fours, twos) = match t2 % 3 {
        0 => (t2 / 3, 0, 0),
        1 if t2 >= 4 => (t2 / 3 - 1, 2, 0),
        1 => (0, 0, 1),
        _ => (t2 / 3, 1, 0),
    };
    let per = eights as f64 * pass_cost(8)
        + fours as f64 * pass_cost(4)
        + twos as f64 * pass_cost(2)
        + t3 as f64 * pass_cost(3)
        + t5 as f64 * pass_cost(5)
        + t7 as f64 * pass_cost(7);
    let passes = eights + fours + twos + t3 + t5 + t7;
    m as f64 * per + PASS_FIXED * passes as f64
}

/// The best inner length of Bluestein's algorithm for `n` and the estimated cost.
fn bluestein_cost(n: usize) -> (usize, f64) {
    let lo = 2 * n - 1;
    let mut best = (lo.next_power_of_two(), f64::INFINITY);
    // The smooth lengths 2^a·3^b·5^c·7^d in [lo, 2·lo), their cost from the exponents.
    let (mut p7, mut t7) = (1usize, 0);
    while p7 < 2 * lo {
        let (mut p5, mut t5) = (p7, 0);
        while p5 < 2 * lo {
            let (mut p3, mut t3) = (p5, 0);
            while p3 < 2 * lo {
                let (mut m, mut t2) = (p3, 0);
                while m < lo {
                    m *= 2;
                    t2 += 1;
                }
                if m < 2 * lo {
                    let c = 2.0 * smooth_cost(m, t2, t3, t5, t7) + 3.0 * m as f64 + 2.0 * n as f64;
                    if c < best.1 {
                        best = (m, c);
                    }
                }
                p3 *= 3;
                t3 += 1;
            }
            p5 *= 5;
            t5 += 1;
        }
        p7 *= 7;
        t7 += 1;
    }
    best
}

/// For a prime `p > MAX_PRIME_PASS`: whether Rader's algorithm is chosen over Bluestein's, and
/// the estimated cost.
fn prime_choice(p: usize) -> (bool, f64) {
    let rader = 2.0 * cost(p - 1) + 5.0 * p as f64;
    let blue = bluestein_cost(p).1;
    (rader <= blue, rader.min(blue))
}

impl<T: Kernels> Algo<T> {
    /// The transform of the length `n` with the kernels of `isa`.
    pub(crate) fn new(n: usize, isa: Isa) -> Algo<T> {
        Algo::plan(n, T::kset(isa))
    }

    fn plan(n: usize, ks: &'static Kset<T>) -> Algo<T> {
        if n <= 1 {
            return Algo::Trivial;
        }
        let f = factors(n);
        let big = *f.last().expect("n >= 2");
        if f.len() == 1 && n > MAX_PRIME_PASS {
            return if prime_choice(n).0 {
                Algo::Rader(Box::new(Rader::new(n, ks)))
            } else {
                Algo::Blue(Box::new(Blue::new(n, ks)))
            };
        }
        if pass::composite(n) {
            return Algo::Stock(Stock::new(n, 1, &[n], ks));
        }
        if big > MAX_GENERIC {
            if bluestein_cost(n).1 < cost(n) {
                return Algo::Blue(Box::new(Blue::new(n, ks)));
            }
            return Algo::Stock(Stock::new(n, 1, &radices(n), ks));
        }
        Algo::Stock(Stock::new(n, 1, &radices(n), ks))
    }

    /// The values of working memory [`run`](Self::run) needs.
    pub(crate) fn scratch(&self) -> usize {
        match self {
            Algo::Trivial => 0,
            Algo::Stock(s) => s.need(1),
            Algo::Rader(r) => r.p - 1 + r.inner.scratch(),
            Algo::Blue(b) => b.m + b.inner.scratch(),
        }
    }

    /// The runs a call of [`run_batch`](Self::run_batch) should take with `scr` values of
    /// working memory (at least [`scratch`](Self::scratch)), and the working memory that many
    /// need.
    pub(crate) fn batch(&self, scr: usize) -> (usize, usize) {
        match self {
            Algo::Stock(s) if s.passes.len() == 1 => (s.batch, s.extra),
            Algo::Stock(s) => {
                let c = (scr.saturating_sub(s.extra) / s.len).clamp(1, s.batch);
                (c, s.need(c))
            }
            _ => (1, self.scratch()),
        }
    }

    /// The working memory that lets [`run_batch`](Self::run_batch) take its most runs at once.
    pub(crate) fn batch_scratch(&self) -> usize {
        match self {
            Algo::Stock(s) => s.need(s.batch),
            _ => self.scratch(),
        }
    }

    /// [`run`](Self::run) for `count` transforms one after the other, with the working memory
    /// [`batch`](Self::batch) gives for `count`.
    ///
    /// # Safety
    /// As for [`run`](Self::run), for the `count·n` values.
    pub(crate) unsafe fn run_batch(
        &self,
        src: *const Complex<T>,
        dst: *mut Complex<T>,
        scr: *mut Complex<T>,
        count: usize,
        inv: bool,
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            match self {
                Algo::Stock(s) => s.run(src, dst, scr, count, inv),
                _ => self.run(src, dst, scr, inv),
            }
        }
    }

    /// The transform (`inv`: the inverse, unscaled) of the `n` values at `src` into `dst`.
    ///
    /// # Safety
    /// `src` must be valid for reading and `dst` for writing the `n` values; they are the same
    /// or do not overlap. `scr` must be valid for [`scratch`](Self::scratch) values and not
    /// overlap either.
    pub(crate) unsafe fn run(
        &self,
        src: *const Complex<T>,
        dst: *mut Complex<T>,
        scr: *mut Complex<T>,
        inv: bool,
    ) {
        // SAFETY: the caller's contract, passed on.
        unsafe {
            match self {
                Algo::Trivial => {
                    if !core::ptr::eq(src, dst) {
                        *dst = *src;
                    }
                }
                Algo::Stock(s) => s.run(src, dst, scr, 1, inv),
                Algo::Rader(r) => r.run(src, dst, scr, inv),
                Algo::Blue(b) => b.run(src, dst, scr, inv),
            }
        }
    }
}

/// The kernel of a length transformed by one hard-coded butterfly, for a plan to call it
/// directly on one run: no dispatch through the algorithm, its passes and the batch.
#[derive(Clone, Copy)]
pub(crate) struct Short<T> {
    /// The batch kernels of the butterfly (`Form::B`), called for one run.
    kern: [Kern<T>; 2],
    len: usize,
    pass: *const Pass<T>,
}

impl<T> Short<T> {
    /// The transform of the one run at `x` into `y` (the same or not overlapping).
    ///
    /// # Safety
    /// `x` and `y` must be valid for the run's values, and the plan `algo` this was made from
    /// must be alive.
    #[inline(always)]
    pub(crate) unsafe fn run(&self, x: *const Complex<T>, y: *mut Complex<T>, inv: bool) {
        // SAFETY: the caller's contract; the pass lives in the plan, which never changes.
        unsafe { (self.kern[inv as usize])(&*self.pass, x, y, 1, self.len) }
    }
}

impl<T: Kernels> Algo<T> {
    /// The direct kernel of a length of one hard-coded butterfly, if it is one; the pass is
    /// inside `self`, which must stay where it is (it is in an `Arc`) and unchanged.
    pub(crate) fn short(&self) -> Option<Short<T>> {
        match self {
            Algo::Stock(s) if s.passes.len() == 1 && s.len == s.passes[0].r => {
                let p = &s.passes[0];
                if p.sub.is_some() || p.bkern.is_none() {
                    return None;
                }
                Some(Short {
                    kern: p.bkern.expect("checked"),
                    len: p.r,
                    pass: p,
                })
            }
            _ => None,
        }
    }
}

/// The kernels of the real transform of an odd `n` of one butterfly by that butterfly on the
/// reals (forward, inverse), and its table.
pub(crate) fn real_odd_kernels<T: Kernels>(n: usize, isa: Isa) -> Option<([RKern<T>; 2], Vec<T>)> {
    (T::kset(isa).real_odd)(n).map(|k| (k, pass::bfly_table::<T>(n)))
}

/// The combine passes of the odd split of a real transform by the radix `r` (forward, inverse)
/// and the table of the butterfly of `r`, if `r` has them.
pub(crate) fn split_kernels<T: Kernels>(r: usize, isa: Isa) -> Option<(SFwd<T>, SInv<T>, Vec<T>)> {
    (T::kset(isa).split)(r).map(|(f, i)| (f, i, pass::bfly_table::<T>(r)))
}

/// The fused real kernels of `n = 2m` (forward, inverse) and the table of the butterfly of
/// `m`, for a length `m` of one butterfly.
pub(crate) fn real_kernels<T: Kernels>(m: usize, isa: Isa) -> Option<([RKern<T>; 2], Vec<T>)> {
    (T::kset(isa).real)(m).map(|k| (k, pass::bfly_table::<T>(m)))
}

impl<T: Kernels> Algo<T> {
    /// The transforms of the length `n` of `s0` interleaved sequences, `x[q + s0·j]` for the
    /// sequence `q` (the columns of an array of `s0` columns stored row by row), in place by
    /// passes along the contiguous index: if `n` has no prime factor above [`MAX_GENERIC`].
    pub(crate) fn interleaved(n: usize, s0: usize, isa: Isa) -> Option<Algo<T>> {
        if n <= 1 || s0 == 0 || *factors(n).last()? > MAX_GENERIC {
            return None;
        }
        let rs = if pass::composite(n) {
            vec![n]
        } else {
            radices(n)
        };
        Some(Algo::Stock(Stock::new(n, s0, &rs, T::kset(isa))))
    }
}

impl<T> Algo<T> {
    /// The algorithm, for `Debug`: e.g. `8·8·4·4`, `Rader(4·4·3·3·7)`, `2·[Bluestein(…)]`.
    pub(crate) fn describe(&self) -> String {
        match self {
            Algo::Trivial => "identity".into(),
            Algo::Stock(s) => s.describe(),
            Algo::Rader(r) => format!("Rader({})", r.inner.describe()),
            Algo::Blue(b) => format!("Bluestein({}: {})", b.m, b.inner.describe()),
        }
    }
}

impl<T> Stock<T> {
    fn describe(&self) -> String {
        let p: Vec<String> = self
            .passes
            .iter()
            .map(|p| match &p.sub {
                None => p.r.to_string(),
                Some(s) => format!("[{}]", s.describe()),
            })
            .collect();
        p.join("·")
    }
}

impl<T: Kernels> Stock<T> {
    /// The passes of the radices `rs` (in order) for `s0` interleaved sequences of the length
    /// `n`, the product of `rs`.
    fn new(n: usize, s0: usize, rs: &[usize], ks: &'static Kset<T>) -> Stock<T> {
        let mut passes = Vec::with_capacity(rs.len());
        // Every pass's twiddles W_L^e = W_n^(e·n/L) from one table of the roots of n.
        let roots = twiddle::Gen::new::<T>(n);
        let (mut s, mut l) = (s0, n);
        let mut extra = 0;
        for &r in rs {
            let m = l / r;
            let comp = pass::composite(r);
            let codelet = comp || CODELETS.contains(&r);
            let form = if codelet && s == 1 && ks.plane > 1 && m >= ks.plane {
                Form::P
            } else if m == 1 {
                Form::L
            } else {
                Form::Q
            };
            let step = n / l;
            // W_L^(j·p) in the place the pass reads it: compact [wr, wr, −wi, wi] by p then j,
            // or for the P form by blocks of `plane` values of p, in lane form; written once in
            // order (no zeroing first: at 2²⁰ the tables are tens of megabytes).
            let len = if form == Form::P || m > 1 {
                4 * (r - 1) * m
            } else {
                0
            };
            let mut tw: Vec<T> = Vec::with_capacity(len);
            let out = &mut tw.spare_capacity_mut()[..len];
            let mut whole = 0;
            if form == Form::P {
                let k = ks.plane;
                whole = m / k * k;
                // Blocks of k values of p, for each j the lanes [wr, wr, …] then [−wi, wi, …].
                let blocks = &mut out[..4 * (r - 1) * whole];
                for (pb, blk) in blocks.chunks_exact_mut(4 * (r - 1) * k).enumerate() {
                    for (j, c) in blk.chunks_exact_mut(4 * k).enumerate() {
                        let (re, im) = c.split_at_mut(2 * k);
                        let e0 = (j + 1) * pb * k * step;
                        let (re, im) = (re.as_chunks_mut::<2>().0, im.as_chunks_mut::<2>().0);
                        for (l, (re, im)) in re.iter_mut().zip(im).enumerate() {
                            let w = roots.get_t::<T>(e0 + (j + 1) * l * step);
                            re[0].write(w.re);
                            re[1].write(w.re);
                            im[0].write(-w.im);
                            im[1].write(w.im);
                        }
                    }
                }
            }
            if m > 1 || form == Form::P {
                // The rest by p, then j: W^(j·p) with the exponent by additions.
                let rows = out[4 * (r - 1) * whole..].chunks_exact_mut(4 * (r - 1));
                for (p, row) in (whole..m).zip(rows) {
                    let mut e = 0;
                    for c in row.as_chunks_mut::<4>().0 {
                        e += p * step;
                        let w = roots.get_t::<T>(e);
                        c[0].write(w.re);
                        c[1].write(w.re);
                        c[2].write(-w.im);
                        c[3].write(w.im);
                    }
                }
            }
            // SAFETY: the loops above wrote each of the `len` values: `4(r − 1)` a value of p,
            // for the blocks of the P form and for the rest.
            unsafe { tw.set_len(len) };
            debug_assert_eq!(tw.len(), len);
            let big = !codelet;
            let mut batched = None;
            let (g, sub) = if big {
                let sub = Algo::plan(r, ks);
                extra = extra.max(r + sub.scratch());
                if let Some(sb) = SubBatch::new(r, s, &sub, ks) {
                    extra = extra.max(sb.scratch());
                    batched = Some(Box::new(sb));
                }
                (Vec::new(), Some(Box::new(sub)))
            } else {
                (pass::bfly_table::<T>(r), None)
            };
            let kern = if big {
                // Unused: the pass runs by its sub-plan.
                [(ks.pass)(2, Form::Q, false); 2]
            } else {
                [(ks.pass)(r, form, false), (ks.pass)(r, form, true)]
            };
            let bkern = (codelet && s == 1 && m == 1)
                .then(|| [(ks.pass)(r, Form::B, false), (ks.pass)(r, Form::B, true)]);
            passes.push(Pass {
                r,
                m,
                s,
                tw,
                g,
                kern,
                bkern,
                sub,
                batched,
            });
            s *= r;
            l = m;
        }
        let len = n * s0;
        Stock {
            len,
            batch: (BATCH_VALUES / len).max(1),
            passes,
            extra,
        }
    }

    /// One pass from `x` to `y`, with `extra` the working memory of a sub-plan pass.
    ///
    /// # Safety
    /// As for [`Algo::run`], for the pass's values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn exec(
        p: &Pass<T>,
        x: *const Complex<T>,
        y: *mut Complex<T>,
        extra: *mut Complex<T>,
        count: usize,
        len: usize,
        inv: bool,
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            match &p.sub {
                None if p.bkern.is_some() => {
                    (p.bkern.expect("checked")[inv as usize])(p, x, y, count, len)
                }
                None => (p.kern[inv as usize])(p, x, y, count, len),
                Some(sub) => {
                    for c in 0..count {
                        let (x, y) = (x.add(c * len), y.add(c * len));
                        match &p.batched {
                            Some(sb) => sb.run(p, x, y, extra, inv),
                            None => sub_pass(p, sub, x, y, extra, inv),
                        }
                    }
                }
            }
        }
    }

    /// The working memory of `count` transforms at once: none for the buffer of a single pass
    /// (it runs in place), else `count·len` values; then `extra`.
    fn need(&self, count: usize) -> usize {
        if self.passes.len() == 1 {
            self.extra
        } else {
            count * self.len + self.extra
        }
    }

    /// The passes of `count` transforms one after the other from `src` to `dst` through `tmp`
    /// ([`need`](Self::need)`(count)` values).
    ///
    /// # Safety
    /// As for [`Algo::run`], for the `count·len` values.
    unsafe fn run(
        &self,
        src: *const Complex<T>,
        dst: *mut Complex<T>,
        tmp: *mut Complex<T>,
        count: usize,
        inv: bool,
    ) {
        let np = self.passes.len();
        let len = self.len;

        // SAFETY: the caller's contract; the buffers alternate so that no pass but a last one
        // (m = 1) reads and writes the same values.
        unsafe {
            let extra = if np == 1 { tmp } else { tmp.add(count * len) };
            if np == 1 {
                Self::exec(&self.passes[0], src, dst, extra, count, len, inv);
                return;
            }
            let in_place = core::ptr::eq(src, dst);
            let mut cur = src;
            for (i, p) in self.passes[..np - 1].iter().enumerate() {
                let out = if in_place {
                    if i.is_multiple_of(2) { tmp } else { dst }
                } else if (np - 2 - i).is_multiple_of(2) {
                    dst
                } else {
                    tmp
                };
                Self::exec(p, cur, out, extra, count, len, inv);
                cur = out;
            }
            Self::exec(&self.passes[np - 1], cur, dst, extra, count, len, inv);
        }
    }
}

impl<T: Kernels> Stock<T> {
    /// [`run`](Self::run) of one transform of `s0 = 1` sequence with the last pass by `k` (the
    /// fused real pass, `tw` its factors) into `dst`, which holds one value more.
    ///
    /// # Safety
    /// As for [`run`](Self::run) with `count = 1`; `dst` valid for `len + 1` values; the last
    /// pass is one `k` was made for.
    unsafe fn run_last(
        &self,
        src: *const Complex<T>,
        dst: *mut Complex<T>,
        tmp: *mut Complex<T>,
        k: HKern<T>,
        tw: *const T,
    ) {
        let np = self.passes.len();
        let len = self.len;
        // SAFETY: as in `run`; the last pass reads `cur` and writes `dst`, the same or apart.
        unsafe {
            let extra = tmp.add(len);
            let in_place = core::ptr::eq(src, dst);
            let mut cur = src;
            for (i, p) in self.passes[..np - 1].iter().enumerate() {
                let out = if in_place {
                    if i.is_multiple_of(2) { tmp } else { dst }
                } else if (np - 2 - i).is_multiple_of(2) {
                    dst
                } else {
                    tmp
                };
                Self::exec(p, cur, out, extra, 1, len, false);
                cur = out;
            }
            k(&self.passes[np - 1], cur, dst, tw);
        }
    }
}

impl<T: Kernels> Algo<T> {
    /// For the even real transform whose half this plan transforms: the kernel of its last pass
    /// fused with the real pass, when the plan is Stockham passes of one sequence (two at least)
    /// whose last is a hard-coded butterfly with such a kernel.
    pub(crate) fn half_last(&self, isa: Isa) -> Option<HKern<T>> {
        match self {
            Algo::Stock(s) if s.passes.len() >= 2 && s.passes[0].s == 1 => {
                let p = s.passes.last()?;
                if p.m != 1 || p.sub.is_some() || pass::composite(p.r) {
                    return None;
                }
                (T::kset(isa).half_last)(p.r)
            }
            _ => None,
        }
    }

    /// The transform of `x` with the last pass by `k` (from [`half_last`](Self::half_last)) into
    /// `y`, `scr` the plan's working memory.
    ///
    /// # Safety
    /// As for [`Algo::run`]; `y` valid for `n + 1` values; `tw` holds the real pass's factors.
    pub(crate) unsafe fn run_half_last(
        &self,
        x: *const Complex<T>,
        y: *mut Complex<T>,
        scr: *mut Complex<T>,
        k: HKern<T>,
        tw: *const T,
    ) {
        let Algo::Stock(s) = self else {
            unreachable!("a plan of passes")
        };
        // SAFETY: the caller's contract.
        unsafe { s.run_last(x, y, scr, k, tw) }
    }
}

/// A pass of a large prime radix by its plan: each of the `m·s` sequences gathered, transformed,
/// multiplied by its twiddles and scattered.
///
/// # Safety
/// As for [`Algo::run`]; `buf` holds `r` values and the sub-plan's working memory.
unsafe fn sub_pass<T: Kernels>(
    d: &Pass<T>,
    sub: &Algo<T>,
    x: *const Complex<T>,
    y: *mut Complex<T>,
    buf: *mut Complex<T>,
    inv: bool,
) {
    let (r, m, s) = (d.r, d.m, d.s);
    // SAFETY: the caller's contract; every index is within the pass's `r·m·s` values.
    unsafe {
        let scr = buf.add(r);
        for p in 0..m {
            for q in 0..s {
                for k in 0..r {
                    *buf.add(k) = *x.add(q + s * (p + k * m));
                }
                sub.run(buf, buf, scr, inv);
                if p > 0 {
                    let tw = d.tw.as_ptr().add(4 * (r - 1) * p);
                    for j in 1..r {
                        let (wr, wi) = (*tw.add(4 * (j - 1)), *tw.add(4 * (j - 1) + 3));
                        let w = Complex::new(wr, if inv { -wi } else { wi });
                        *buf.add(j) *= w;
                    }
                }
                for j in 0..r {
                    *y.add(q + s * (r * p + j)) = *buf.add(j);
                }
            }
        }
    }
}

/// `b^e mod m`.
fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1u64;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = ((r as u128 * b as u128) % m as u128) as u64;
        }
        b = ((b as u128 * b as u128) % m as u128) as u64;
        e >>= 1;
    }
    r
}

/// The smallest primitive root modulo the prime `p`.
fn primitive_root(p: usize) -> usize {
    let mut q = factors(p - 1);
    q.dedup();
    (2..p)
        .find(|&g| {
            q.iter()
                .all(|&f| pow_mod(g as u64, ((p - 1) / f) as u64, p as u64) != 1)
        })
        .expect("a prime has a primitive root")
}

/// The transform of `b` by a plan for its length in `f64`: `plan` itself when `T` is `f64` (it
/// is the plan of that length), else one made for the call.
fn transform_f64<T: Kernels>(b: &mut [Complex<f64>], plan: &Algo<T>) {
    let own;
    let a: &Algo<f64> = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
        // SAFETY: `T` is `f64`, so `Algo<T>` is `Algo<f64>`.
        unsafe { &*(plan as *const Algo<T>).cast::<Algo<f64>>() }
    } else {
        own = Algo::<f64>::plan(b.len(), f64::kset(Isa::detect()));
        &own
    };
    let mut scr = vec![Complex::ZERO; a.scratch()];
    let p = b.as_mut_ptr();
    // SAFETY: in place, the scratch of the plan's size.
    unsafe { a.run(p, p, scr.as_mut_ptr(), false) };
}

/// The tables of Rader's algorithm for the prime `p`: `g^q` and `g^(−q)` mod p, and the transform
/// of `W_p^(g^(−q))` divided by `p − 1` (computed in `f64`).
fn rader_tables<T: Kernels>(
    p: usize,
    inner: &Algo<T>,
) -> (Vec<usize>, Vec<usize>, Vec<Complex<T>>) {
    let g = primitive_root(p);
    let mut gin = Vec::with_capacity(p - 1);
    let mut a = 1usize;
    for _ in 0..p - 1 {
        gin.push(a);
        a = if (p as u64) < 1 << 32 {
            (a as u64 * g as u64 % p as u64) as usize
        } else {
            ((a as u128 * g as u128) % p as u128) as usize
        };
    }
    // g^(−q) = g^(p − 1 − q).
    let gout: Vec<usize> = (0..p - 1)
        .map(|q| if q == 0 { 1 } else { gin[p - 1 - q] })
        .collect();
    // The kernel in f64: the transform of W_p^(g^(−q)), divided by p − 1.
    let roots = twiddle::Roots::new(p);
    let mut bk: Vec<Complex<f64>> = gout
        .iter()
        .map(|&e| {
            let (c, s) = roots.get(e);
            Complex::new(c, s)
        })
        .collect();
    transform_f64(&mut bk, inner);
    let scale = 1.0 / (p - 1) as f64;
    let ws: Vec<Complex<T>> = bk
        .iter()
        .map(|v| Complex::new(T::from_f64(v.re * scale), T::from_f64(v.im * scale)))
        .collect();
    (gin, gout, ws)
}

/// The tables of Bluestein's algorithm for `n` by the inner length `m`: the chirp `W_2n^(k²)`,
/// `k < n`, and the transform of its conjugate's circular extension to `m`, divided by `m`.
fn blue_tables<T: Kernels>(
    n: usize,
    inner: &Algo<T>,
    m: usize,
) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
    // k² mod 2n keeps the angle exact: W_2n^(k²).
    let roots = twiddle::Roots::new(2 * n);
    // k² mod 2n by the differences 2k − 1 < 2n.
    let mut e = 0usize;
    let w: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            if k > 0 {
                e += 2 * k - 1;
                if e >= 2 * n {
                    e -= 2 * n;
                }
            }
            roots.get(e)
        })
        .collect();
    let mut b = vec![Complex::<f64>::ZERO; m];
    b[0] = Complex::new(w[0].0, -w[0].1);
    for k in 1..n {
        b[k] = Complex::new(w[k].0, -w[k].1);
        b[m - k] = b[k];
    }
    transform_f64(&mut b, inner);
    let scale = 1.0 / m as f64;
    let kernel: Vec<Complex<T>> = b
        .iter()
        .map(|v| Complex::new(T::from_f64(v.re * scale), T::from_f64(v.im * scale)))
        .collect();
    let chirp: Vec<Complex<T>> = w
        .iter()
        .map(|&(c, s)| Complex::new(T::from_f64(c), T::from_f64(s)))
        .collect();
    (chirp, kernel)
}

impl<T: Kernels> Rader<T> {
    fn new(p: usize, ks: &'static Kset<T>) -> Rader<T> {
        let inner = Algo::plan(p - 1, ks);
        let (gin, gout, ws) = rader_tables::<T>(p, &inner);
        Rader {
            p,
            inner,
            gin,
            gout,
            kernel: pmul_table(&ws, ks.wide),
            pmul: ks.pmul[0],
        }
    }

    /// # Safety
    /// As for [`Algo::run`].
    unsafe fn run(
        &self,
        src: *const Complex<T>,
        dst: *mut Complex<T>,
        scr: *mut Complex<T>,
        inv: bool,
    ) {
        let p = self.p;
        // SAFETY: the caller's contract; `scr` holds p − 1 values and the inner working memory.
        unsafe {
            let buf = scr;
            let iscr = scr.add(p - 1);
            let x0 = *src;
            for (q, &i) in self.gin.iter().enumerate() {
                *buf.add(q) = *src.add(i);
            }
            self.inner.run(buf, buf, iscr, false);
            let total = x0 + *buf;
            (self.pmul)(buf, self.kernel.as_ptr(), p - 1);
            self.inner.run(buf, buf, iscr, true);
            *dst = total;
            if inv {
                for (q, &o) in self.gout.iter().enumerate() {
                    *dst.add(p - o) = *buf.add(q) + x0;
                }
            } else {
                for (q, &o) in self.gout.iter().enumerate() {
                    *dst.add(o) = *buf.add(q) + x0;
                }
            }
        }
    }
}

impl<T: Kernels> Blue<T> {
    fn new(n: usize, ks: &'static Kset<T>) -> Blue<T> {
        let m = bluestein_cost(n).0;
        let inner = Algo::plan(m, ks);
        let (chirp, kernel) = blue_tables::<T>(n, &inner, m);
        Blue {
            n,
            m,
            inner,
            chirp: pmul_table(&chirp, ks.wide),
            kernel: pmul_table(&kernel, ks.wide),
            pmul: ks.pmul[0],
        }
    }

    /// # Safety
    /// As for [`Algo::run`].
    unsafe fn run(
        &self,
        src: *const Complex<T>,
        dst: *mut Complex<T>,
        scr: *mut Complex<T>,
        inv: bool,
    ) {
        let (n, m) = (self.n, self.m);
        // SAFETY: the caller's contract; `scr` holds m values and the inner working memory.
        unsafe {
            let buf = scr;
            let iscr = scr.add(m);
            core::ptr::copy_nonoverlapping(src, buf, n);
            core::ptr::write_bytes(buf.add(n), 0, m - n);
            (self.pmul)(buf, self.chirp.as_ptr(), n);
            self.inner.run(buf, buf, iscr, false);
            (self.pmul)(buf, self.kernel.as_ptr(), m);
            self.inner.run(buf, buf, iscr, true);
            (self.pmul)(buf, self.chirp.as_ptr(), n);
            if inv {
                *dst = *buf;
                for k in 1..n {
                    *dst.add(n - k) = *buf.add(k);
                }
            } else {
                core::ptr::copy_nonoverlapping(buf, dst, n);
            }
        }
    }
}

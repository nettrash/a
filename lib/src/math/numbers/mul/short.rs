//! Short products: the high half of the product of two numbers of `n` limbs, and of a square, to
//! within `n - 1` units of its last limb, for about the cost of a whole product of fewer limbs
//! (Mulders, "On short multiplications and divisions", 2000; Hanrot and Zimmermann, "A long note on
//! Mulders' short product", 2004).
//!
//! # The contract
//!
//! With `B = 2^64`, the operands `x` and `y` of `n` limbs and `SP = Σ x_i y_j B^(i+j)` over the
//! pairs with `i + j >= n - 1` (the products whose low limbs land at position `n - 1` or above),
//! [`mulhigh`] computes an integer `P` with `SP <= P <= x y`, and writes `V = ⌊P / B^(n-1)⌋`, the
//! `n + 1` limbs from position `n - 1`, into `r[n - 1..2n]`. As `SP` is a multiple of `B^(n-1)`,
//! `SP / B^(n-1) <= V <= ⌊x y / B^(n-1)⌋`, and the products left out are below
//! `Σ_(d < n-1) (d + 1) (B - 1)² B^d < (n - 1) B^n`, so
//!
//! `x y - (n - 1) B^n < V B^(n-1) <= x y`,
//!
//! and the high half `H = ⌊x y / B^n⌋` and the `n` limbs `A = r[n..2n]` satisfy `A <= H <= A + n - 1`:
//! the high half is never overestimated, and is short by less than `n` units of its last limb.
//! [`sqrhigh`] does the same for `x = y`. The limbs `r[0..n - 1]` are clobbered.
//!
//! # The methods
//!
//! - Below [`SHORT_MIN`] limbs, the whole product or square, by the schoolbook kernels.
//! - Below [`MULHIGH_DC`] limbs ([`SQRHIGH_DC`] for a square), by rows: the products `x_i y_j` with
//!   `i + j >= n - 1` exactly, row `j` adding `x_(n-1-j..n) y_j` at position `n - 1`, about half of
//!   the limb products of the whole product; a square adds the products of two different limbs
//!   once, doubles them and adds the squares, as the whole square does, about a quarter.
//! - Beyond, Mulders' method: with `x = x1 B^l + x0` and `y = y1 B^l + y0`, `x1` and `y1` of
//!   `k >= n / 2` limbs and `l = n - k`, the pairs with `i + j >= n - 1` are those of `x1 y1`
//!   (`i, j >= l`) and those with `i < l` (then `j >= k`) or `j < l` (then `i >= k`), which are the
//!   pairs of the short products of `l` limbs of `x0` and the top `l` limbs of `y`, and of the top
//!   `l` of `x` and `y0`, at position `k`. So `P` is the whole product `x1 y1 B^(2l)` plus those
//!   two short products, each `⌊P' / B^(l-1)⌋ B^(l-1)` for its own `P'`, which keeps
//!   `SP <= P <= x y` by induction. A square takes the whole square of `x1` and twice the one
//!   short product of `x0` and the top `l` limbs of `x`.
//! - From [`MULHIGH_FULL`] limbs ([`SQRHIGH_FULL`]), the whole product, whose transforms cost
//!   about as much as Mulders' method's parts there.
//!
//! Under Miri the thresholds of the rows and of Mulders' method are a few limbs ([`SHORT_MIN`] 4,
//! [`MULHIGH_DC`] and [`SQRHIGH_DC`] 10), so that its runs, on short numbers, take the whole
//! product below `SHORT_MIN`, the rows and Mulders' method; the whole product from
//! [`MULHIGH_FULL`] and [`SQRHIGH_FULL`], which follow the transforms' thresholds (312 limbs and
//! more under Miri), is past the lengths its runs take.

use super::super::scratch::{Scratch, need};
use super::basecase;
use super::fft::{FFT_MUL, FFT_SQR};
use super::kernels;
use super::limbs::{add_1, add_n, mul, mul_itch, sqr, sqr_itch};

/// Below this many limbs, a short product or square is the whole one, which the schoolbook
/// kernels compute with less overhead than the rows of different lengths.
// Measured on an Apple M4 Max (sweep 4..=16 limbs): the rows took 1.02 to 1.04 of the whole
// product's time at 8 limbs (1.01 to 1.04 for the square), 0.94 at 9 (0.94 to 0.97) and 0.87 to
// 0.88 at 10 (0.89 to 0.95).
pub(in super::super) const SHORT_MIN: usize = if cfg!(miri) { 4 } else { 9 };

/// From this many limbs, a short product is computed by Mulders' method; below, by rows.
// Measured on an Apple M4 Max (sweep 12..=192 limbs), Mulders' method on rows took 1.31 of the
// rows' time at 12 limbs, 1.14 to 1.21 from 16 to 48, 1.11 at 56, 1.02 to 1.04 from 64 to 72, 1.00
// at 74, 0.95 at 80, 0.93 at 96 and 0.80 at 128: the rows, half of the schoolbook product's limb
// products in the multiplication's kernels, keep ahead of Karatsuba's whole part up to there.
pub(in super::super) const MULHIGH_DC: usize = if cfg!(miri) { 10 } else { 74 };

/// From this many limbs, a short square is computed by Mulders' method; below, by rows.
// Measured on an Apple M4 Max (sweep 9..=96 limbs, the square's split [`split_sqr`]), Mulders'
// method took 1.04 to 1.08 of the rows' time from 9 to 11 limbs, 0.95 at 12, 0.99 at 13 and 14,
// and 0.83 to 0.97 from 15 to 96.
pub(in super::super) const SQRHIGH_DC: usize = if cfg!(miri) { 10 } else { 12 };

/// From this many limbs, a short product is the whole product: past the threshold of the
/// transforms, whose cost is about linear in the length, Mulders' method soon saves nothing.
// Measured on an Apple M4 Max with the transforms of aarch64 (`FFT_MUL` 360, so 450 limbs; sweep
// 256..=1000 limbs), the whole product took 1.17 to 1.22 of Mulders' time from 256 to 340 limbs,
// 1.01 to 1.12 from 360 to 440, 0.85 to 0.99 from 448 to 512, 0.95 to 1.05 from 520 to 720 (the
// transforms' lengths go by steps) and 0.84 to 0.89 from 760.
pub(in super::super) const MULHIGH_FULL: usize = FFT_MUL + FFT_MUL / 4;

/// From this many limbs in the shorter operand, a balanced product is computed by the transforms:
/// for the callers that weigh a short product against a whole one.
pub(in super::super) const TRANSFORMS: usize = FFT_MUL;

/// From this many limbs, a short square is the whole square, as for [`MULHIGH_FULL`].
// Measured on an Apple M4 Max with the transforms of aarch64 (`FFT_SQR` 380, so 665 limbs; sweep
// 256..=1000 limbs, the square's split [`split_sqr`]), the whole square took 1.22 to 1.36 of
// Mulders' time from 256 to 440 limbs, 1.07 to 1.19 from 480 to 640, 1.02 at 660, 0.96 to 0.99 from
// 680 to 740, 0.92 to 0.94 from 760 to 800, 1.04 at 840, 1.00 at 880 and 0.86 from 960.
pub(in super::super) const SQRHIGH_FULL: usize = FFT_SQR + FFT_SQR * 3 / 4;

/// The limbs of the high parts that Mulders' method multiplies whole, for a short product of `n`
/// limbs: at least half of them, about 0.7 of them in the range of Karatsuba's method (Hanrot and
/// Zimmermann's optimum for it).
// Measured on an Apple M4 Max (sweep of the share 0.5..=0.9 at 80, 96, 128, 160, 192, 256, 320,
// 360 and 400 limbs): 0.65 and 0.75 took 0.90 to 1.04 of the time of 0.7 (0.99 to 1.00 on
// average), the others longer on average (0.6 1.02, 0.8 1.02, 0.5 1.07); as the thresholds of the
// whole parts' methods fall, a share wins at one length and loses at the next.
#[inline]
fn split(n: usize) -> usize {
    (n * 7 / 10).max(n.div_ceil(2))
}

/// The limbs of the high part that Mulders' method squares whole, for a short square of `n` limbs:
/// at least half of them, about 0.55: the square's whole part costs less against its short part
/// (a short product of different numbers) than a product's.
// Measured on an Apple M4 Max (sweep of the share 0.5..=0.9 at 16, 24, 32, 48, 64, 96, 128, 160,
// 256, 360 and 420 limbs), 0.55 took 0.91 to 1.00 of the time of 0.7, 0.93 on average from 32 to
// 160 limbs; 0.5 and 0.6 about as long as 0.55 (0.5 1.02 at 16 and 1.04 at 420 limbs), 0.75 and
// more longer than 0.7.
#[inline]
fn split_sqr(n: usize) -> usize {
    (n * 11 / 20).max(n.div_ceil(2))
}

/// The limbs of scratch space that [`mulhigh`] needs for operands of `n` limbs.
pub(in super::super) fn mulhigh_itch(n: usize) -> usize {
    if n < MULHIGH_DC {
        0
    } else if n >= MULHIGH_FULL {
        mul_itch(n, n)
    } else {
        let k = split(n);
        let l = n - k;
        mul_itch(k, k).max(2 * l + mulhigh_itch(l))
    }
}

/// The limbs of scratch space that [`sqrhigh`] needs for an operand of `n` limbs.
pub(in super::super) fn sqrhigh_itch(n: usize) -> usize {
    if n < SQRHIGH_DC {
        0
    } else if n >= SQRHIGH_FULL {
        sqr_itch(n)
    } else {
        let k = split_sqr(n);
        let l = n - k;
        sqr_itch(k).max(2 * l + mulhigh_itch(l))
    }
}

/// Sets `r[n - 1..2n]` to `⌊P / B^(n-1)⌋` for a `P` between the short product of the `n` limbs at
/// `x` and at `y` and their whole product, as the module documentation states: the high half
/// `r[n..2n]` is short of the product's by at most `n - 1` units of its last limb. The limbs
/// `r[0..n - 1]` are clobbered.
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` and `y` must be valid for reads, the `2n` at `r` for writes, apart
/// from both, and the [`mulhigh_itch`]`(n)` limbs at `s` for reads and writes, apart from all.
pub(in super::super) unsafe fn mulhigh(
    r: *mut u64,
    x: *const u64,
    y: *const u64,
    n: usize,
    s: Scratch,
) {
    need!(s, mulhigh_itch(n));
    // SAFETY: forwarded.
    unsafe {
        if n < SHORT_MIN {
            basecase::mul_unbalanced(r, x, n, y, n);
        } else if n < MULHIGH_DC {
            mulhigh_basecase(r, x, y, n);
        } else if n >= MULHIGH_FULL {
            mul(r, x, n, y, n, s);
        } else {
            mulhigh_dc(r, x, y, n, s);
        }
    }
}

/// [`mulhigh`] by rows: row `j` adds the `j + 1` limbs `x_(n-1-j..n)` times `y_j` at position
/// `n - 1`, and its carry out goes to position `n + j`, which no row before it wrote.
///
/// # Safety
///
/// As for [`mulhigh`].
unsafe fn mulhigh_basecase(r: *mut u64, x: *const u64, y: *const u64, n: usize) {
    // SAFETY: as the caller guarantees; row `j` reads and writes limbs `n - 1..n + j` of `r` and
    // writes limb `n + j`.
    unsafe {
        let base = r.add(n - 1);
        *base.add(1) = kernels::mul_1(base, x.add(n - 1), 1, *y);
        for j in 1..n {
            *base.add(j + 1) = basecase::addmul_1(base, x.add(n - 1 - j), j + 1, *y.add(j));
        }
    }
}

/// [`mulhigh`] by Mulders' method, as the module documentation describes.
///
/// # Safety
///
/// As for [`mulhigh`], with `n >= 2`.
unsafe fn mulhigh_dc(r: *mut u64, x: *const u64, y: *const u64, n: usize, s: Scratch) {
    need!(s, mulhigh_itch(n));
    let k = split(n);
    let l = n - k;
    // SAFETY: as the caller guarantees: the whole product of the `k` high limbs into `r[2l..2n]`,
    // then each short product of `l` limbs into `2l` limbs of scratch, whose top `l + 1` limbs are
    // added at position `n - 1`, below `2n`; the sums stay below `x y < B^(2n)`, so the carries
    // stop inside `r`.
    unsafe {
        mul(r.add(2 * l), x.add(l), k, y.add(l), k, s);
        if 2 * l == n {
            *r.add(n - 1) = 0;
        }
        let (t, rest) = s.split(2 * l);
        let top = r.add(n - 1);
        for (a, b) in [(x, y.add(k)), (x.add(k), y)] {
            mulhigh(t, a, b, l, rest);
            if add_n(top, top, t.add(l - 1), l + 1) {
                add_1(top.add(l + 1), n - l, 1);
            }
        }
    }
}

/// Sets `r[n - 1..2n]` to `⌊P / B^(n-1)⌋` for a `P` between the short square of the `n` limbs at
/// `x` (the products `x_i x_j` with `i + j >= n - 1`) and the whole square, as for [`mulhigh`].
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` must be valid for reads, the `2n` at `r` for writes, apart from `x`,
/// and the [`sqrhigh_itch`]`(n)` limbs at `s` for reads and writes, apart from both.
pub(in super::super) unsafe fn sqrhigh(r: *mut u64, x: *const u64, n: usize, s: Scratch) {
    need!(s, sqrhigh_itch(n));
    // SAFETY: forwarded.
    unsafe {
        if n < SHORT_MIN {
            basecase::sqr(r, x, n);
        } else if n < SQRHIGH_DC {
            sqrhigh_basecase(r, x, n);
        } else if n >= SQRHIGH_FULL {
            sqr(r, x, n, s);
        } else {
            sqrhigh_dc(r, x, n, s);
        }
    }
}

/// [`sqrhigh`] by rows: the products of two different limbs `x_i x_j`, `i < j`, with
/// `i + j >= n - 1`, row `i` adding `x_i` times `x_(j0..n)` at position `i + j0` for
/// `j0 = max(i + 1, n - 1 - i)`, its carry out to position `n + i`; then the sum doubled from
/// position `n - 1`, and the squares `x_i²` with `2i >= n - 1` added at position `2i`.
///
/// # Safety
///
/// As for [`sqrhigh`].
unsafe fn sqrhigh_basecase(r: *mut u64, x: *const u64, n: usize) {
    // SAFETY: as the caller guarantees: row `i` reads and writes limbs from `i + j0 >= n - 1` to
    // `n + i - 1`, all written by the rows before it (the first row writes limb `n - 1`), and
    // writes limb `n + i`; the doubling and the squares stay in `r[n - 1..2n]`, where the doubled
    // sum, below the square, fits.
    unsafe {
        if n == 1 {
            let p = u128::from(*x) * u128::from(*x);
            *r = p as u64;
            *r.add(1) = (p >> 64) as u64;
            return;
        }
        // Row 0: `x_0 x_(n-1)` at position `n - 1`.
        *r.add(n) = kernels::mul_1(r.add(n - 1), x.add(n - 1), 1, *x);
        for i in 1..n - 1 {
            let j0 = (i + 1).max(n - 1 - i);
            *r.add(n + i) = basecase::addmul_1(r.add(i + j0), x.add(j0), n - j0, *x.add(i));
        }
        *r.add(2 * n - 1) = 0;
        // Doubled, the bit shifted out of each limb taken into the next, and the squares `x_i²`
        // with `2i >= n - 1` added at position `2i`: from position `n - 1` when it is even; from
        // `n` when it is odd, the limb at `n - 1` doubled apart and its top bit carried in.
        if n % 2 == 1 {
            kernels::double_and_add_squares(r.add(n - 1), x.add((n - 1) / 2), n.div_ceil(2));
        } else {
            let t = *r.add(n - 1);
            *r.add(n - 1) = t << 1;
            kernels::double_and_add_squares(r.add(n), x.add(n / 2), n / 2);
            add_1(r.add(n), n, t >> 63);
        }
    }
}

/// [`sqrhigh`] by Mulders' method: the whole square of the `k` high limbs, and twice the short
/// product of the `l` low limbs and the `l` top ones at position `k`.
///
/// # Safety
///
/// As for [`sqrhigh`], with `n >= 2`.
unsafe fn sqrhigh_dc(r: *mut u64, x: *const u64, n: usize, s: Scratch) {
    need!(s, sqrhigh_itch(n));
    let k = split_sqr(n);
    let l = n - k;
    // SAFETY: as in `mulhigh_dc`: the square into `r[2l..2n]`, the short product into `2l` limbs of
    // scratch, its top `l + 1` limbs added twice at position `n - 1`.
    unsafe {
        sqr(r.add(2 * l), x.add(l), k, s);
        if 2 * l == n {
            *r.add(n - 1) = 0;
        }
        let (t, rest) = s.split(2 * l);
        mulhigh(t, x, x.add(k), l, rest);
        let top = r.add(n - 1);
        for _ in 0..2 {
            if add_n(top, top, t.add(l - 1), l + 1) {
                add_1(top.add(l + 1), n - l, 1);
            }
        }
    }
}

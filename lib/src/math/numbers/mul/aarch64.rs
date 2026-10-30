//! The aarch64 kernels of the multiplication: rows of limb products added into a product, in
//! inline assembly.
//!
//! # A row
//!
//! `mul` and `umulh` give the low and the high limb of a product of two limbs. Adding a row
//! `x * y` into `r` puts two terms into each position `k`: the low limb of `x_k y` and the high
//! limb of `x_{k-1} y`, with the carries. One carry chain cannot take both, so a step of positions
//! runs two, one after the other: chain 1 adds the low limbs to `r` (`adcs`), chain 2 adds the high
//! limbs one position up. Chain 1's carry out of the step's top position is folded into the high
//! limb chain 2 adds at the next position, which can take it, as a high limb is at most
//! `2^64 - 2`; chain 2's carry stays in the carry flag and is chain 1's carry into the next step.
//! So the flag goes from step to step through no other instruction, and nothing is saved or
//! restored between steps. Steps are of 8 positions, after steps of 1, 2 and 4 picked by the low
//! bits of the length; the products of a step are formed in two halves, so that no more registers
//! are live at once than an `asm!` block gets (27 of 28).
//!
//! # Two rows
//!
//! [`addmul_2`] adds two rows, `x * y0` and `x * y1` one position up, in one pass: four terms per
//! position and four chains, the low limbs of `x_k y0`, the high limbs of `x_{k-1} y0`, the low
//! limbs of `x_{k-1} y1` and the high limbs of `x_{k-2} y1`. Chain 1's carry is folded into the
//! next high limb of `y0` and chain 3's into the next high limb of `y1`; chain 4's goes to chain 1
//! of the next step in the flag, and chain 2's, which has no high limb to go into, is kept in a
//! register (`cset`, and `cmp` before the next step). Steps are of 4 positions.
//!
//! # Speed
//!
//! An Apple M4 core issues 3 multiplications a cycle, but only about 2.4 flag-writing additions
//! and 4 integer operations in all: a limb product takes at least 4 (`mul`, `umulh` and two
//! additions), which is where both kernels end, at about 1 cycle per limb product. Measured on an
//! Apple M4 Max, the median of 31 runs, in cycles for an `nx` by `ny` product, against GMP 6.3's
//! `mpn_mul_basecase`:
//!
//! | `nx` x `ny` | portable | one row a pass | two rows a pass | GMP |
//! | --- | --- | --- | --- | --- |
//! | 3 x 3 | 16 | 22 | 18 | 24 |
//! | 4 x 4 | 30 | 35 | 26 | 29 |
//! | 8 x 8 | 140 | 81 | 75 | 84 |
//! | 16 x 16 | 607 | 280 | 260 | 291 |
//! | 24 x 24 | 1272 | 573 | 589 | 645 |
//! | 32 x 32 | 2369 | 1008 | 1076 | 1187 |
//! | 48 x 48 | 5865 | 2367 | 2440 | 2932 |
//!
//! So [`mul_basecase`] takes the portable code, which the compiler unrolls, for operands of up
//! to 3 limbs and for a single limb times up to 8; two rows a pass up to 16 limbs, where it saves
//! half of the steps' overhead; and one row a pass beyond, where the four chains' extra moves
//! cost more than the overhead saved.
//!
//! # Squares
//!
//! A square adds each product of two different limbs once, a triangle of rows, then doubles the
//! triangle and adds the squares of the limbs in one chain ([`diag_addlsh1`]). Its rows get
//! shorter, one limb a row, so one row a pass spends much of the square on the steps of 1, 2 and 4
//! positions; [`sqr_basecase`] takes the rows of up to [`SQR_ONE_ROW`] limbs two at a time by
//! [`addmul_2`], the one product of the pair that the other row does not have added first, and
//! from there into the kernel's first position as a carry. Measured on an Apple M4 Max, the
//! median of 21 runs, in cycles for an `n`-limb square, against GMP 6.3's `mpn_sqr_basecase`:
//!
//! | `n` | one row a pass | two rows a pass up to 10 limbs | GMP |
//! | --- | --- | --- | --- |
//! | 16 | 238 | 215 | 248 |
//! | 20 | 340 | 305 | 380 |
//! | 24 | 454 | 419 | 516 |
//! | 32 | 697 | 657 | 842 |
//! | 48 | 1373 | 1332 | 1692 |
//! | 64 | 2352 | 2304 | 2710 |
//!
//! Pairs of rows of up to 4, 6, 8, 12, 16 or 24 limbs were within 1 to 4 % of these.
//!
//! The kernels read and write limbs where they are: `ldp` and `stp` need no alignment, though
//! pairs that straddle 16 bytes cost the longer products up to about 1 %, and the working memory
//! of the multiplication starts its operands, product and scratch on 16 bytes.

use core::arch::asm;

#[path = "portable.rs"]
mod portable;

/// Sets the `n` limbs at `r` to the `n` limbs at `x` times `y`, and returns the limb above them:
/// one chain, the low limb of each product plus the high limb of the one below it.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for writes, and `r`
/// must be `x` or apart from it.
#[inline(always)]
pub(super) unsafe fn mul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let carry: u64;
    // SAFETY: the loops read limbs `0..n` of `x` and write limbs `0..n` of `r`, each limb of `x`
    // before the limb of `r` at its position; the asm uses no stack, and names every register it
    // writes.
    unsafe {
        asm!(
            "mov {hp}, #0",
            "cmn xzr, xzr",
            "tbz {n}, #0, 1f",
            "ldr {a0}, [{xp}], #8",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "adcs {l0}, {l0}, {hp}",
            "mov {hp}, {h0}",
            "str {l0}, [{rp}], #8",
            "1:",
            "tbz {n}, #1, 1f",
            "ldp {a0}, {a1}, [{xp}], #16",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h1}, {a1}, {y}",
            "adcs {l0}, {l0}, {hp}",
            "adcs {l1}, {l1}, {h0}",
            "mov {hp}, {h1}",
            "stp {l0}, {l1}, [{rp}], #16",
            "1:",
            "tbz {n}, #2, 1f",
            "ldp {a0}, {a1}, [{xp}]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h1}, {a1}, {y}",
            "ldp {a0}, {a1}, [{xp}, #16]",
            "mul {l2}, {a0}, {y}", "umulh {h2}, {a0}, {y}",
            "mul {l3}, {a1}, {y}", "umulh {h3}, {a1}, {y}",
            "adcs {l0}, {l0}, {hp}",
            "adcs {l1}, {l1}, {h0}",
            "adcs {l2}, {l2}, {h1}",
            "adcs {l3}, {l3}, {h2}",
            "mov {hp}, {h3}",
            "stp {l0}, {l1}, [{rp}]",
            "stp {l2}, {l3}, [{rp}, #16]",
            "add {xp}, {xp}, #32",
            "add {rp}, {rp}, #32",
            "1:",
            "lsr {n}, {n}, #3",
            "cbz {n}, 3f",
            "2:",
            "ldp {a0}, {a1}, [{xp}]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h1}, {a1}, {y}",
            "ldp {a0}, {a1}, [{xp}, #16]",
            "mul {l2}, {a0}, {y}", "umulh {h2}, {a0}, {y}",
            "mul {l3}, {a1}, {y}", "umulh {h3}, {a1}, {y}",
            "adcs {l0}, {l0}, {hp}",
            "adcs {l1}, {l1}, {h0}",
            "adcs {l2}, {l2}, {h1}",
            "adcs {l3}, {l3}, {h2}",
            "stp {l0}, {l1}, [{rp}]",
            "stp {l2}, {l3}, [{rp}, #16]",
            "ldp {a0}, {a1}, [{xp}, #32]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h1}, {a1}, {y}",
            "ldp {a0}, {a1}, [{xp}, #48]",
            "mul {l2}, {a0}, {y}", "umulh {h2}, {a0}, {y}",
            "mul {l4}, {a1}, {y}", "umulh {hp}, {a1}, {y}",
            "adcs {l0}, {l0}, {h3}",
            "adcs {l1}, {l1}, {h0}",
            "adcs {l2}, {l2}, {h1}",
            "adcs {l4}, {l4}, {h2}",
            "stp {l0}, {l1}, [{rp}, #32]",
            "stp {l2}, {l4}, [{rp}, #48]",
            "add {xp}, {xp}, #64",
            "add {rp}, {rp}, #64",
            "sub {n}, {n}, #1",
            "cbnz {n}, 2b",
            "3:",
            "adc {hp}, {hp}, xzr",
            rp = inout(reg) r => _, xp = inout(reg) x => _, n = inout(reg) n => _, y = in(reg) y,
            hp = out(reg) carry,
            a0 = out(reg) _, a1 = out(reg) _,
            l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _, l4 = out(reg) _,
            h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
            options(nostack),
        );
    }
    carry
}

/// Sets the `n` little-endian limbs at `r` to those at `x` times `y`, at any addresses, and returns
/// the limb above them: [`mul_1`] on the bytes of a number, whose loads and stores of eight bytes
/// need no alignment on the systems these kernels are built for, the same that the addition's
/// kernels read unaligned limbs on.
///
/// # Safety
///
/// As for [`mul_1`], for the `8 n` bytes at each pointer.
#[inline(always)]
pub(crate) unsafe fn mul_1_bytes(r: *mut u8, x: *const u8, n: usize, y: u64) -> u64 {
    // SAFETY: as the caller guarantees; the assembly reads and writes the limbs with `ldr` and
    // `str`, which take any address here, and the pointers are not dereferenced in Rust.
    unsafe { mul_1(r.cast(), x.cast(), n, y) }
}

/// Adds the `n` limbs at `x` times `y` to the `n` limbs at `r`, and returns the limb that
/// carries out of them: two chains a step, as the module documentation describes.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for reads and writes,
/// and apart from them.
#[inline(always)]
pub(in super::super) unsafe fn addmul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let carry: u64;
    // SAFETY: as in `mul_1`, with limbs `0..n` of `r` read as well.
    unsafe {
        asm!(
            "mov {hp}, #0",
            "cmn xzr, xzr",
            "tbz {n}, #0, 1f",
            "ldr {a0}, [{xp}], #8",
            "ldr {u0}, [{rp}]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "adcs {u0}, {u0}, {l0}",
            "adc {h0}, {h0}, xzr",
            "adds {u0}, {u0}, {hp}",
            "mov {hp}, {h0}",
            "str {u0}, [{rp}], #8",
            "1:",
            "tbz {n}, #1, 1f",
            "ldp {a0}, {a1}, [{xp}], #16",
            "ldp {u0}, {u1}, [{rp}]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h1}, {a1}, {y}",
            "adcs {u0}, {u0}, {l0}",
            "adcs {u1}, {u1}, {l1}",
            "adc {h1}, {h1}, xzr",
            "adds {u0}, {u0}, {hp}",
            "adcs {u1}, {u1}, {h0}",
            "mov {hp}, {h1}",
            "stp {u0}, {u1}, [{rp}], #16",
            "1:",
            "tbz {n}, #2, 1f",
            "ldp {a0}, {a1}, [{xp}]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h1}, {a1}, {y}",
            "ldp {a0}, {a1}, [{xp}, #16]",
            "mul {l2}, {a0}, {y}", "umulh {h2}, {a0}, {y}",
            "mul {l3}, {a1}, {y}", "umulh {h3}, {a1}, {y}",
            "ldp {u0}, {u1}, [{rp}]",
            "ldp {u2}, {u3}, [{rp}, #16]",
            "adcs {u0}, {u0}, {l0}",
            "adcs {u1}, {u1}, {l1}",
            "adcs {u2}, {u2}, {l2}",
            "adcs {u3}, {u3}, {l3}",
            "adc {h3}, {h3}, xzr",
            "adds {u0}, {u0}, {hp}",
            "adcs {u1}, {u1}, {h0}",
            "adcs {u2}, {u2}, {h1}",
            "adcs {u3}, {u3}, {h2}",
            "mov {hp}, {h3}",
            "stp {u0}, {u1}, [{rp}]",
            "stp {u2}, {u3}, [{rp}, #16]",
            "add {xp}, {xp}, #32",
            "add {rp}, {rp}, #32",
            "1:",
            "lsr {n}, {n}, #3",
            "cbz {n}, 3f",
            "2:",
            "ldp {a0}, {a1}, [{xp}]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h1}, {a1}, {y}",
            "ldp {a0}, {a1}, [{xp}, #16]",
            "mul {l2}, {a0}, {y}", "umulh {h2}, {a0}, {y}",
            "mul {l3}, {a1}, {y}", "umulh {h3}, {a1}, {y}",
            "ldp {u0}, {u1}, [{rp}]",
            "ldp {u2}, {u3}, [{rp}, #16]",
            "adcs {u0}, {u0}, {l0}",
            "adcs {u1}, {u1}, {l1}",
            "adcs {u2}, {u2}, {l2}",
            "adcs {u3}, {u3}, {l3}",
            "ldp {a0}, {a1}, [{xp}, #32]",
            "mul {l0}, {a0}, {y}", "umulh {h4}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h5}, {a1}, {y}",
            "ldp {a0}, {a1}, [{xp}, #48]",
            "mul {l2}, {a0}, {y}", "umulh {h6}, {a0}, {y}",
            "mul {l3}, {a1}, {y}", "umulh {h7}, {a1}, {y}",
            "ldp {u4}, {u5}, [{rp}, #32]",
            "ldp {u6}, {u7}, [{rp}, #48]",
            "adcs {u4}, {u4}, {l0}",
            "adcs {u5}, {u5}, {l1}",
            "adcs {u6}, {u6}, {l2}",
            "adcs {u7}, {u7}, {l3}",
            "adc {h7}, {h7}, xzr",
            "adds {u0}, {u0}, {hp}",
            "adcs {u1}, {u1}, {h0}",
            "adcs {u2}, {u2}, {h1}",
            "adcs {u3}, {u3}, {h2}",
            "adcs {u4}, {u4}, {h3}",
            "adcs {u5}, {u5}, {h4}",
            "adcs {u6}, {u6}, {h5}",
            "adcs {u7}, {u7}, {h6}",
            "mov {hp}, {h7}",
            "stp {u0}, {u1}, [{rp}]",
            "stp {u2}, {u3}, [{rp}, #16]",
            "stp {u4}, {u5}, [{rp}, #32]",
            "stp {u6}, {u7}, [{rp}, #48]",
            "add {xp}, {xp}, #64",
            "add {rp}, {rp}, #64",
            "sub {n}, {n}, #1",
            "cbnz {n}, 2b",
            "3:",
            "adc {hp}, {hp}, xzr",
            rp = inout(reg) r => _, xp = inout(reg) x => _, n = inout(reg) n => _, y = in(reg) y,
            hp = out(reg) carry,
            a0 = out(reg) _, a1 = out(reg) _,
            l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
            h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
            h4 = out(reg) _, h5 = out(reg) _, h6 = out(reg) _, h7 = out(reg) _,
            u0 = out(reg) _, u1 = out(reg) _, u2 = out(reg) _, u3 = out(reg) _,
            u4 = out(reg) _, u5 = out(reg) _, u6 = out(reg) _, u7 = out(reg) _,
            options(nostack),
        );
    }
    carry
}

/// Adds the `n` limbs at `x` times `y0 + y1 B`, and the limb `c`, to the `n` limbs at `r`, writing
/// `n + 1` limbs, and returns the limb at position `n + 1`: four chains a step, as the module
/// documentation describes; `c` goes in as the high limb carried into the first position.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads, the `n` at `r` for reads and the
/// `n + 1` at `r` for writes, apart from `x`.
#[inline(always)]
pub(in super::super) unsafe fn addmul_2(
    r: *mut u64,
    x: *const u64,
    n: usize,
    y0: u64,
    y1: u64,
    c: u64,
) -> u64 {
    let top: u64;
    // SAFETY: as in `addmul_1`, with limb `n` of `r` written as well. The terms carried from one
    // step to the next: `h0p`, the high limb of `x_{k-1} y0` with chain 1's carry folded in;
    // `l1p`, the low limb of `x_{k-1} y1`; `h1pp`, the high limb of `x_{k-2} y1` with chain 3's
    // carry folded in; `h1p`, the high limb of `x_{k-1} y1`; and `c2`, chain 2's carry.
    unsafe {
        asm!(
            "mov {l1p}, #0", "mov {h1pp}, #0", "mov {h1p}, #0", "mov {c2}, #0",
            "cmn xzr, xzr",
            "tbz {n}, #0, 1f",
            "ldr {a0}, [{xp}], #8",
            "ldr {u0}, [{rp}]",
            "mul {l0}, {a0}, {y0}", "umulh {h0}, {a0}, {y0}",
            "mul {l1}, {a0}, {y1}", "umulh {h1}, {a0}, {y1}",
            "adcs {u0}, {u0}, {l0}",
            "adc {h0}, {h0}, xzr",
            "cmp {c2}, #1",
            "adcs {u0}, {u0}, {h0p}",
            "cset {c2}, cs",
            "adds {u0}, {u0}, {l1p}",
            "adc {h1p}, {h1p}, xzr",
            "adds {u0}, {u0}, {h1pp}",
            "str {u0}, [{rp}], #8",
            "mov {h0p}, {h0}", "mov {l1p}, {l1}", "mov {h1pp}, {h1p}", "mov {h1p}, {h1}",
            "1:",
            "tbz {n}, #1, 1f",
            "ldp {a0}, {a1}, [{xp}], #16",
            "ldp {u0}, {u1}, [{rp}]",
            "mul {l0}, {a0}, {y0}", "umulh {h0}, {a0}, {y0}",
            "mul {l1}, {a1}, {y0}", "umulh {h1}, {a1}, {y0}",
            "adcs {u0}, {u0}, {l0}",
            "adcs {u1}, {u1}, {l1}",
            "adc {h1}, {h1}, xzr",
            "cmp {c2}, #1",
            "adcs {u0}, {u0}, {h0p}",
            "adcs {u1}, {u1}, {h0}",
            "cset {c2}, cs",
            "mov {h0p}, {h1}",
            "mul {l0}, {a0}, {y1}", "umulh {h0}, {a0}, {y1}",
            "mul {l1}, {a1}, {y1}", "umulh {h1}, {a1}, {y1}",
            "adds {u0}, {u0}, {l1p}",
            "adcs {u1}, {u1}, {l0}",
            "adc {h0}, {h0}, xzr",
            "adds {u0}, {u0}, {h1pp}",
            "adcs {u1}, {u1}, {h1p}",
            "stp {u0}, {u1}, [{rp}], #16",
            "mov {l1p}, {l1}", "mov {h1pp}, {h0}", "mov {h1p}, {h1}",
            "1:",
            "lsr {n}, {n}, #2",
            "cbz {n}, 3f",
            "2:",
            "ldp {a0}, {a1}, [{xp}]",
            "ldp {a2}, {a3}, [{xp}, #16]",
            "mul {l0}, {a0}, {y0}", "umulh {h0}, {a0}, {y0}",
            "mul {l1}, {a1}, {y0}", "umulh {h1}, {a1}, {y0}",
            "mul {l2}, {a2}, {y0}", "umulh {h2}, {a2}, {y0}",
            "mul {l3}, {a3}, {y0}", "umulh {h3}, {a3}, {y0}",
            "ldp {u0}, {u1}, [{rp}]",
            "ldp {u2}, {u3}, [{rp}, #16]",
            "adcs {u0}, {u0}, {l0}",
            "adcs {u1}, {u1}, {l1}",
            "adcs {u2}, {u2}, {l2}",
            "adcs {u3}, {u3}, {l3}",
            "adc {h3}, {h3}, xzr",
            "cmp {c2}, #1",
            "adcs {u0}, {u0}, {h0p}",
            "adcs {u1}, {u1}, {h0}",
            "adcs {u2}, {u2}, {h1}",
            "adcs {u3}, {u3}, {h2}",
            "cset {c2}, cs",
            "mov {h0p}, {h3}",
            "mul {l0}, {a0}, {y1}", "umulh {h0}, {a0}, {y1}",
            "mul {l1}, {a1}, {y1}", "umulh {h1}, {a1}, {y1}",
            "mul {l2}, {a2}, {y1}", "umulh {h2}, {a2}, {y1}",
            "mul {l3}, {a3}, {y1}", "umulh {h3}, {a3}, {y1}",
            "adds {u0}, {u0}, {l1p}",
            "adcs {u1}, {u1}, {l0}",
            "adcs {u2}, {u2}, {l1}",
            "adcs {u3}, {u3}, {l2}",
            "adc {h2}, {h2}, xzr",
            "adds {u0}, {u0}, {h1pp}",
            "adcs {u1}, {u1}, {h1p}",
            "adcs {u2}, {u2}, {h0}",
            "adcs {u3}, {u3}, {h1}",
            "stp {u0}, {u1}, [{rp}]",
            "stp {u2}, {u3}, [{rp}, #16]",
            "mov {l1p}, {l3}", "mov {h1pp}, {h2}", "mov {h1p}, {h3}",
            "add {xp}, {xp}, #32",
            "add {rp}, {rp}, #32",
            "sub {n}, {n}, #1",
            "cbnz {n}, 2b",
            "3:",
            // Position `n`: the terms left and the carries; position `n + 1`: what is above.
            "adcs {u0}, {h0p}, {l1p}",
            "adc {u1}, {h1p}, xzr",
            "adds {u0}, {u0}, {h1pp}",
            "adc {u1}, {u1}, xzr",
            "adds {u0}, {u0}, {c2}",
            "adc {u1}, {u1}, xzr",
            "str {u0}, [{rp}]",
            rp = inout(reg) r => _, xp = inout(reg) x => _, n = inout(reg) n => _,
            y0 = in(reg) y0, y1 = in(reg) y1,
            u1 = out(reg) top,
            h0p = inout(reg) c => _, l1p = out(reg) _, h1pp = out(reg) _, h1p = out(reg) _, c2 = out(reg) _,
            a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _, a3 = out(reg) _,
            u0 = out(reg) _, u2 = out(reg) _, u3 = out(reg) _,
            l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
            h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
            options(nostack),
        );
    }
    top
}

/// Completes an exact division in place: with the `n` limbs at `r` holding `a d` (modulo
/// `B^n`), sets them to the `q` for which `q B - q = a d`, limb `i` of `q` being limb `i - 1` less
/// limb `i` of `a d` and the borrow from below; one `sbcs` chain, a cycle per limb.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `r` must be valid for reads and writes.
#[inline]
pub(super) unsafe fn divexact_finish(r: *mut u64, n: usize) {
    // SAFETY: limbs `0..n` of `r`, each read before it is written; the asm uses no stack, and
    // names every register it writes.
    unsafe {
        asm!(
            "mov {q}, #0",
            "cmp xzr, xzr",
            "tbz {n}, #0, 1f",
            "ldr {a0}, [{rp}]",
            "sbcs {q}, {q}, {a0}",
            "str {q}, [{rp}], #8",
            "1:",
            "tbz {n}, #1, 1f",
            "ldp {a0}, {a1}, [{rp}]",
            "sbcs {a0}, {q}, {a0}",
            "sbcs {q}, {a0}, {a1}",
            "stp {a0}, {q}, [{rp}], #16",
            "1:",
            "lsr {n}, {n}, #2",
            "cbz {n}, 3f",
            "2:",
            "ldp {a0}, {a1}, [{rp}]",
            "ldp {a2}, {a3}, [{rp}, #16]",
            "sbcs {a0}, {q}, {a0}",
            "sbcs {a1}, {a0}, {a1}",
            "sbcs {a2}, {a1}, {a2}",
            "sbcs {q}, {a2}, {a3}",
            "stp {a0}, {a1}, [{rp}]",
            "stp {a2}, {q}, [{rp}, #16]",
            "add {rp}, {rp}, #32",
            "sub {n}, {n}, #1",
            "cbnz {n}, 2b",
            "3:",
            rp = inout(reg) r => _, n = inout(reg) n => _,
            q = out(reg) _, a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _, a3 = out(reg) _,
            options(nostack),
        );
    }
}

/// Doubles the `2n` limbs at `r` and adds the squares of the `n` limbs at `x` at the even
/// positions: one chain, the bit shifted out of each limb taken into the next by `extr`, which
/// leaves the flags alone.
///
/// # Safety
///
/// `n >= 1`; the `2n` limbs at `r` must be valid for reads and writes, the `n` at `x` for reads,
/// apart from them, and the top bit of `r` clear.
#[inline(always)]
unsafe fn diag_addlsh1(r: *mut u64, x: *const u64, n: usize) {
    // SAFETY: limb `i` of `x` and limbs `2i` and `2i + 1` of `r` per round, `r` read before it is
    // written.
    unsafe {
        asm!(
            "mov {prev}, #0",
            "cmn xzr, xzr",
            "2:",
            "ldr {a}, [{xp}], #8",
            "ldp {t0}, {t1}, [{rp}]",
            "mul {lo}, {a}, {a}",
            "umulh {hi}, {a}, {a}",
            "extr {d0}, {t0}, {prev}, #63",
            "extr {d1}, {t1}, {t0}, #63",
            "mov {prev}, {t1}",
            "adcs {d0}, {d0}, {lo}",
            "adcs {d1}, {d1}, {hi}",
            "stp {d0}, {d1}, [{rp}], #16",
            "sub {n}, {n}, #1",
            "cbnz {n}, 2b",
            rp = inout(reg) r => _, xp = inout(reg) x => _, n = inout(reg) n => _,
            prev = out(reg) _, a = out(reg) _, t0 = out(reg) _, t1 = out(reg) _,
            lo = out(reg) _, hi = out(reg) _, d0 = out(reg) _, d1 = out(reg) _,
            options(nostack),
        );
    }
}

/// The longest rows of a square's triangle that [`sqr_basecase`] adds two at a time; longer ones
/// go one at a time (see the module documentation).
pub(super) const SQR_ONE_ROW: usize = 10;

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` at `y`:
/// the portable code for the smallest shapes, then two rows a pass up to 16 limbs and one beyond
/// (see the module documentation).
///
/// # Safety
///
/// `nx >= ny >= 1`; the limbs of `x` and `y` must be valid for reads and those of `r` for writes,
/// and `r` apart from both.
#[inline]
pub(super) unsafe fn mul_basecase(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize) {
    // SAFETY: forwarded; each row adds into limbs of `r` that the rows before it wrote, and writes
    // the one above them.
    unsafe {
        if nx <= 3 || (ny == 1 && nx <= 8) {
            portable::mul_basecase(r, x, nx, y, ny);
            return;
        }
        *r.add(nx) = mul_1(r, x, nx, *y);
        let mut j = 1;
        if nx <= 16 {
            while j + 1 < ny {
                *r.add(nx + j + 1) = addmul_2(r.add(j), x, nx, *y.add(j), *y.add(j + 1), 0);
                j += 2;
            }
        }
        while j < ny {
            *r.add(nx + j) = addmul_1(r.add(j), x, nx, *y.add(j));
            j += 1;
        }
    }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `x`: the products of two
/// different limbs once, then doubled with the squares added.
///
/// The products `x_i x_j` for `i < j` make a triangle of rows, row `i` adding `x_i` times the
/// limbs above it at position `2i + 1`. Rows longer than [`SQR_ONE_ROW`] limbs are added one at a
/// time; shorter ones two at a time by [`addmul_2`], rows `i` and `i + 1` over the limbs from
/// `x_{i+2}` at position `2i + 2`, with the one product they do not share, `x_i x_{i+1}` at
/// position `2i + 1`, added before: its low limb to the limb there, its high limb and the carry as
/// the kernel's carry in.
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` must be valid for reads and those of `r` for writes, and apart
/// from them.
#[inline]
pub(super) unsafe fn sqr_basecase(r: *mut u64, x: *const u64, n: usize) {
    // SAFETY: as in `portable::sqr_basecase`: row `i` adds into limbs `2i + 1..n + i` of `r`, all
    // written by the rows before it, and writes limb `n + i`; a pair of rows writes limbs `n + i`
    // and `n + i + 1`. The triangle leaves the top bit of `r` clear, as it is below half the
    // square.
    unsafe {
        if n <= 3 {
            portable::sqr_basecase(r, x, n);
            return;
        }
        *r.add(n) = mul_1(r.add(1), x.add(1), n - 1, *x);
        let mut i = 1;
        while i + 2 < n && n - i - 1 > SQR_ONE_ROW {
            *r.add(n + i) = addmul_1(r.add(2 * i + 1), x.add(i + 1), n - i - 1, *x.add(i));
            i += 1;
        }
        while i + 2 < n {
            let (y0, y1) = (*x.add(i), *x.add(i + 1));
            let p = u128::from(y0) * u128::from(y1);
            let (low, carry) = (*r.add(2 * i + 1)).overflowing_add(p as u64);
            *r.add(2 * i + 1) = low;
            let c = (p >> 64) as u64 + u64::from(carry);
            *r.add(n + i + 1) = addmul_2(r.add(2 * i + 2), x.add(i + 2), n - i - 2, y0, y1, c);
            i += 2;
        }
        if i + 2 == n {
            *r.add(n + i) = addmul_1(r.add(2 * i + 1), x.add(i + 1), 1, *x.add(i));
        }
        *r = 0;
        *r.add(2 * n - 1) = 0;
        diag_addlsh1(r, x, n);
    }
}

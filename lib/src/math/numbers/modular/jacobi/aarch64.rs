//! The aarch64 kernels of the Jacobi symbol, in inline assembly: the binary steps on two to six
//! limbs.
//!
//! The steps of the greatest common divisor's `gcd_22`, six instructions deep, with the symbol's
//! sign carried beside them, off that path: reciprocity's `u0 & v0` when the numbers swap, and the
//! factor `(2 / v)` of each zero shifted out, bit 1 of `v0 ^ (v0 >> 1)` for an odd count. Measured
//! on an Apple M4 Max on fresh pairs of 128 bits, the Jacobi symbol took 566 cycles so, against
//! 667 for the same steps in Rust and 920 for GMP 6.3's `mpz_jacobi`.

use core::arch::asm;

/// The portable `steps_22` (`jacobi/portable.rs`, elsewhere and under Miri) in assembly.
#[inline]
pub(super) fn steps_22(
    mut u0: u64,
    mut u1: u64,
    mut v0: u64,
    mut v1: u64,
    mut bits: u64,
) -> Option<(u64, u64, u64)> {
    debug_assert!(u0 & v0 & 1 == 1);
    let mut equal = 0u64;
    if u1 | v1 != 0 {
        // SAFETY: the assembly reads and writes registers only, and names every one it writes.
        unsafe {
            asm!(
                "2:",
                "subs {t0}, {u0}, {v0}",
                "sbcs {t1}, {u1}, {v1}",
                "cbz {t0}, 4f",
                "rbit {c}, {t0}",
                "and {x}, {u0}, {v0}",
                "csel {x}, {x}, xzr, lo",
                "eor {bits}, {bits}, {x}",
                "csel {v0}, {u0}, {v0}, lo",
                "csel {v1}, {u1}, {v1}, lo",
                "cneg {t0}, {t0}, lo",
                "cinv {t1}, {t1}, lo",
                "clz {c}, {c}",
                "neg {n}, {c}",
                "lsr {u0}, {t0}, {c}",
                "lsl {x}, {t1}, {n}",
                "lsr {u1}, {t1}, {c}",
                "orr {u0}, {u0}, {x}",
                "eor {y}, {v0}, {v0}, lsr #1",
                "sbfx {x}, {c}, #0, #1",
                "and {y}, {y}, {x}",
                "eor {bits}, {bits}, {y}",
                "orr {x}, {u1}, {v1}",
                "cbnz {x}, 2b",
                "b 6f",
                // The low limbs agree: the difference is its high limb's alone, shifted by 64 and
                // its own zeros, which count by their parity.
                "4:",
                "cbz {t1}, 5f",
                "and {x}, {u0}, {v0}",
                "csel {x}, {x}, xzr, lo",
                "eor {bits}, {bits}, {x}",
                "csel {v0}, {u0}, {v0}, lo",
                "csel {v1}, {u1}, {v1}, lo",
                "cneg {t1}, {t1}, lo",
                "rbit {c}, {t1}",
                "clz {c}, {c}",
                "lsr {u0}, {t1}, {c}",
                "mov {u1}, #0",
                "eor {y}, {v0}, {v0}, lsr #1",
                "sbfx {x}, {c}, #0, #1",
                "and {y}, {y}, {x}",
                "eor {bits}, {bits}, {y}",
                "cbnz {v1}, 2b",
                "b 6f",
                // Equal numbers of two limbs: a common factor.
                "5:",
                "mov {equal}, #1",
                "6:",
                u0 = inout(reg) u0, u1 = inout(reg) u1, v0 = inout(reg) v0, v1 = inout(reg) v1,
                bits = inout(reg) bits, equal = inout(reg) equal,
                t0 = out(reg) _, t1 = out(reg) _, c = out(reg) _, n = out(reg) _, x = out(reg) _,
                y = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
    }
    let _ = (u1, v1);
    (equal == 0).then_some((u0, v0, bits))
}

#[allow(dead_code)]
#[path = "portable.rs"]
mod portable;

pub(super) use portable::steps_nn;

/// [`steps_nn`] on three limbs in assembly: [`steps_22`]'s steps widened by a limb, the borrow
/// through three, the magnitude by the complement of the two limbs above the lowest. A lowest limb
/// of the difference of zero, once in `2^64`, leaves the loop before the step, and the portable
/// steps take over from there. Measured on an Apple M4 Max on fresh numbers of three limbs, the
/// steps down to two took 387 cycles so, against 922 for the portable ones.
#[inline]
pub(super) fn steps_3(u: &mut [u64; 3], v: &mut [u64; 3], bits: &mut u64) -> bool {
    let ([mut u0, mut u1, mut u2], [mut v0, mut v1, mut v2]) = (*u, *v);
    let mut b = *bits;
    let mut rare = 0u64;
    if u2 | v2 != 0 {
        // SAFETY: the assembly reads and writes registers only, and names every one it writes.
        unsafe {
            asm!(
                "2:",
                "subs {t0}, {u0}, {v0}",
                "sbcs {t1}, {u1}, {v1}",
                "sbcs {t2}, {u2}, {v2}",
                "cbz {t0}, 4f",
                "rbit {c}, {t0}",
                "and {x}, {u0}, {v0}",
                "csel {x}, {x}, xzr, lo",
                "eor {bits}, {bits}, {x}",
                "csel {v0}, {u0}, {v0}, lo",
                "csel {v1}, {u1}, {v1}, lo",
                "csel {v2}, {u2}, {v2}, lo",
                "cneg {t0}, {t0}, lo",
                "cinv {t1}, {t1}, lo",
                "cinv {t2}, {t2}, lo",
                "clz {c}, {c}",
                "neg {n}, {c}",
                "lsr {u0}, {t0}, {c}",
                "lsl {x}, {t1}, {n}",
                "lsr {u1}, {t1}, {c}",
                "orr {u0}, {u0}, {x}",
                "lsl {x}, {t2}, {n}",
                "lsr {u2}, {t2}, {c}",
                "orr {u1}, {u1}, {x}",
                "eor {y}, {v0}, {v0}, lsr #1",
                "sbfx {x}, {c}, #0, #1",
                "and {y}, {y}, {x}",
                "eor {bits}, {bits}, {y}",
                "orr {x}, {u2}, {v2}",
                "cbnz {x}, 2b",
                "b 6f",
                // The lowest limbs agree: the portable steps take this one.
                "4:",
                "mov {rare}, #1",
                "6:",
                u0 = inout(reg) u0, u1 = inout(reg) u1, u2 = inout(reg) u2,
                v0 = inout(reg) v0, v1 = inout(reg) v1, v2 = inout(reg) v2,
                bits = inout(reg) b, rare = inout(reg) rare,
                t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, c = out(reg) _, n = out(reg) _,
                x = out(reg) _, y = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
    }
    (*u, *v, *bits) = ([u0, u1, u2], [v0, v1, v2], b);
    if rare != 0 {
        return steps_nn(u, v, bits);
    }
    true
}

/// [`steps_3`] on four limbs: measured as there, 419 cycles down to three limbs, against 1284.
#[inline]
pub(super) fn steps_4(u: &mut [u64; 4], v: &mut [u64; 4], bits: &mut u64) -> bool {
    let ([mut u0, mut u1, mut u2, mut u3], [mut v0, mut v1, mut v2, mut v3]) = (*u, *v);
    let mut b = *bits;
    let mut rare = 0u64;
    if u3 | v3 != 0 {
        // SAFETY: the assembly reads and writes registers only, and names every one it writes.
        unsafe {
            asm!(
                "2:",
                "subs {t0}, {u0}, {v0}",
                "sbcs {t1}, {u1}, {v1}",
                "sbcs {t2}, {u2}, {v2}",
                "sbcs {t3}, {u3}, {v3}",
                "cbz {t0}, 4f",
                "rbit {c}, {t0}",
                "and {x}, {u0}, {v0}",
                "csel {x}, {x}, xzr, lo",
                "eor {bits}, {bits}, {x}",
                "csel {v0}, {u0}, {v0}, lo",
                "csel {v1}, {u1}, {v1}, lo",
                "csel {v2}, {u2}, {v2}, lo",
                "csel {v3}, {u3}, {v3}, lo",
                "cneg {t0}, {t0}, lo",
                "cinv {t1}, {t1}, lo",
                "cinv {t2}, {t2}, lo",
                "cinv {t3}, {t3}, lo",
                "clz {c}, {c}",
                "neg {n}, {c}",
                "lsr {u0}, {t0}, {c}",
                "lsl {x}, {t1}, {n}",
                "orr {u0}, {u0}, {x}",
                "lsr {u1}, {t1}, {c}",
                "lsl {x}, {t2}, {n}",
                "orr {u1}, {u1}, {x}",
                "lsr {u2}, {t2}, {c}",
                "lsl {x}, {t3}, {n}",
                "orr {u2}, {u2}, {x}",
                "lsr {u3}, {t3}, {c}",
                "eor {y}, {v0}, {v0}, lsr #1",
                "sbfx {x}, {c}, #0, #1",
                "and {y}, {y}, {x}",
                "eor {bits}, {bits}, {y}",
                "orr {x}, {u3}, {v3}",
                "cbnz {x}, 2b",
                "b 6f",
                "4:",
                "mov {rare}, #1",
                "6:",
                u0 = inout(reg) u0, u1 = inout(reg) u1, u2 = inout(reg) u2, u3 = inout(reg) u3,
                v0 = inout(reg) v0, v1 = inout(reg) v1, v2 = inout(reg) v2, v3 = inout(reg) v3,
                bits = inout(reg) b, rare = inout(reg) rare,
                t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _, c = out(reg) _,
                n = out(reg) _, x = out(reg) _, y = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
    }
    (*u, *v, *bits) = ([u0, u1, u2, u3], [v0, v1, v2, v3], b);
    if rare != 0 {
        return steps_nn(u, v, bits);
    }
    true
}

/// [`steps_3`] on 5 limbs.
#[inline]
pub(super) fn steps_5(u: &mut [u64; 5], v: &mut [u64; 5], bits: &mut u64) -> bool {
    let ([mut u0, mut u1, mut u2, mut u3, mut u4], [mut v0, mut v1, mut v2, mut v3, mut v4]) =
        (*u, *v);
    let mut b = *bits;
    let mut rare = 0u64;
    if u4 | v4 != 0 {
        // SAFETY: the assembly reads and writes registers only, and names every one it writes.
        unsafe {
            asm!(
                "2:",
                "subs {t0}, {u0}, {v0}",
                "sbcs {t1}, {u1}, {v1}",
                "sbcs {t2}, {u2}, {v2}",
                "sbcs {t3}, {u3}, {v3}",
                "sbcs {t4}, {u4}, {v4}",
                "cbz {t0}, 4f",
                "rbit {c}, {t0}",
                "and {x}, {u0}, {v0}",
                "csel {x}, {x}, xzr, lo",
                "eor {bits}, {bits}, {x}",
                "csel {v0}, {u0}, {v0}, lo",
                "csel {v1}, {u1}, {v1}, lo",
                "csel {v2}, {u2}, {v2}, lo",
                "csel {v3}, {u3}, {v3}, lo",
                "csel {v4}, {u4}, {v4}, lo",
                "cneg {t0}, {t0}, lo",
                "cinv {t1}, {t1}, lo",
                "cinv {t2}, {t2}, lo",
                "cinv {t3}, {t3}, lo",
                "cinv {t4}, {t4}, lo",
                "clz {c}, {c}",
                "neg {n}, {c}",
                "lsr {u0}, {t0}, {c}",
                "lsl {x}, {t1}, {n}",
                "orr {u0}, {u0}, {x}",
                "lsr {u1}, {t1}, {c}",
                "lsl {x}, {t2}, {n}",
                "orr {u1}, {u1}, {x}",
                "lsr {u2}, {t2}, {c}",
                "lsl {x}, {t3}, {n}",
                "orr {u2}, {u2}, {x}",
                "lsr {u3}, {t3}, {c}",
                "lsl {x}, {t4}, {n}",
                "orr {u3}, {u3}, {x}",
                "lsr {u4}, {t4}, {c}",
                "eor {y}, {v0}, {v0}, lsr #1",
                "sbfx {x}, {c}, #0, #1",
                "and {y}, {y}, {x}",
                "eor {bits}, {bits}, {y}",
                "orr {x}, {u4}, {v4}",
                "cbnz {x}, 2b",
                "b 6f",
                "4:",
                "mov {rare}, #1",
                "6:",
                u0 = inout(reg) u0, u1 = inout(reg) u1, u2 = inout(reg) u2, u3 = inout(reg) u3,
                u4 = inout(reg) u4,
                v0 = inout(reg) v0, v1 = inout(reg) v1, v2 = inout(reg) v2, v3 = inout(reg) v3,
                v4 = inout(reg) v4,
                bits = inout(reg) b, rare = inout(reg) rare,
                t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _, t4 = out(reg) _,
                c = out(reg) _, n = out(reg) _, x = out(reg) _, y = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
    }
    (*u, *v, *bits) = ([u0, u1, u2, u3, u4], [v0, v1, v2, v3, v4], b);
    if rare != 0 {
        return steps_nn(u, v, bits);
    }
    true
}

/// [`steps_3`] on 6 limbs.
#[inline]
pub(super) fn steps_6(u: &mut [u64; 6], v: &mut [u64; 6], bits: &mut u64) -> bool {
    let (
        [mut u0, mut u1, mut u2, mut u3, mut u4, mut u5],
        [mut v0, mut v1, mut v2, mut v3, mut v4, mut v5],
    ) = (*u, *v);
    let mut b = *bits;
    let mut rare = 0u64;
    if u5 | v5 != 0 {
        // SAFETY: the assembly reads and writes registers only, and names every one it writes.
        unsafe {
            asm!(
                "2:",
                "subs {t0}, {u0}, {v0}",
                "sbcs {t1}, {u1}, {v1}",
                "sbcs {t2}, {u2}, {v2}",
                "sbcs {t3}, {u3}, {v3}",
                "sbcs {t4}, {u4}, {v4}",
                "sbcs {t5}, {u5}, {v5}",
                "cbz {t0}, 4f",
                "rbit {c}, {t0}",
                "and {x}, {u0}, {v0}",
                "csel {x}, {x}, xzr, lo",
                "eor {bits}, {bits}, {x}",
                "csel {v0}, {u0}, {v0}, lo",
                "csel {v1}, {u1}, {v1}, lo",
                "csel {v2}, {u2}, {v2}, lo",
                "csel {v3}, {u3}, {v3}, lo",
                "csel {v4}, {u4}, {v4}, lo",
                "csel {v5}, {u5}, {v5}, lo",
                "cneg {t0}, {t0}, lo",
                "cinv {t1}, {t1}, lo",
                "cinv {t2}, {t2}, lo",
                "cinv {t3}, {t3}, lo",
                "cinv {t4}, {t4}, lo",
                "cinv {t5}, {t5}, lo",
                "clz {c}, {c}",
                "neg {n}, {c}",
                "lsr {u0}, {t0}, {c}",
                "lsl {x}, {t1}, {n}",
                "orr {u0}, {u0}, {x}",
                "lsr {u1}, {t1}, {c}",
                "lsl {x}, {t2}, {n}",
                "orr {u1}, {u1}, {x}",
                "lsr {u2}, {t2}, {c}",
                "lsl {x}, {t3}, {n}",
                "orr {u2}, {u2}, {x}",
                "lsr {u3}, {t3}, {c}",
                "lsl {x}, {t4}, {n}",
                "orr {u3}, {u3}, {x}",
                "lsr {u4}, {t4}, {c}",
                "lsl {x}, {t5}, {n}",
                "orr {u4}, {u4}, {x}",
                "lsr {u5}, {t5}, {c}",
                "eor {y}, {v0}, {v0}, lsr #1",
                "sbfx {x}, {c}, #0, #1",
                "and {y}, {y}, {x}",
                "eor {bits}, {bits}, {y}",
                "orr {x}, {u5}, {v5}",
                "cbnz {x}, 2b",
                "b 6f",
                "4:",
                "mov {rare}, #1",
                "6:",
                u0 = inout(reg) u0, u1 = inout(reg) u1, u2 = inout(reg) u2, u3 = inout(reg) u3,
                u4 = inout(reg) u4, u5 = inout(reg) u5,
                v0 = inout(reg) v0, v1 = inout(reg) v1, v2 = inout(reg) v2, v3 = inout(reg) v3,
                v4 = inout(reg) v4, v5 = inout(reg) v5,
                bits = inout(reg) b, rare = inout(reg) rare,
                t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _, t4 = out(reg) _,
                t5 = out(reg) _, c = out(reg) _, n = out(reg) _, x = out(reg) _, y = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
    }
    (*u, *v, *bits) = ([u0, u1, u2, u3, u4, u5], [v0, v1, v2, v3, v4, v5], b);
    if rare != 0 {
        return steps_nn(u, v, bits);
    }
    true
}

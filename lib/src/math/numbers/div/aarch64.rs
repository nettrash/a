//! The aarch64 kernel of the division: a row of limb products subtracted from a partial
//! remainder, in inline assembly.
//!
//! Subtracting a row `x * y` from `r` takes two terms from each position `k`, the low limb of
//! `x_k y` and the high limb of `x_{k-1} y`, as adding a row in the multiplication puts two in, and
//! the kernel is that of [`addmul_1`](super::super::mul) with each addition turned into a
//! subtraction: chain 1 subtracts the low limbs (`sbcs`), chain 2 the high limbs one position up.
//! Chain 1's borrow out of a step's top position is folded into the high limb that chain 2
//! subtracts next (`cinc`, which leaves the flags alone), which can take it, as a high limb is at
//! most `2^64 - 2`; chain 2's borrow stays in the carry flag (clear for a borrow, on aarch64) for
//! chain 1 of the next step. Steps are of 8 positions, after steps of 1, 2 and 4 picked by the low
//! bits of the length.

use core::arch::asm;

/// Subtracts the `n` limbs at `x` times `y` from the `n` limbs at `r`, and returns the limb that
/// borrows out of them: two chains a step, as the module documentation describes.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for reads and writes,
/// and apart from them.
#[inline(always)]
pub(super) unsafe fn submul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let borrow: u64;
    // SAFETY: the loops read limbs `0..n` of `x` and `r` and write limbs `0..n` of `r`, each limb of
    // `r` read before it is written; the asm uses no stack, and names every register it writes.
    unsafe {
        asm!(
            "mov {hp}, #0",
            "cmp xzr, xzr",
            "tbz {n}, #0, 1f",
            "ldr {a0}, [{xp}], #8",
            "ldr {u0}, [{rp}]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "sbcs {u0}, {u0}, {l0}",
            "cinc {h0}, {h0}, cc",
            "subs {u0}, {u0}, {hp}",
            "mov {hp}, {h0}",
            "str {u0}, [{rp}], #8",
            "1:",
            "tbz {n}, #1, 1f",
            "ldp {a0}, {a1}, [{xp}], #16",
            "ldp {u0}, {u1}, [{rp}]",
            "mul {l0}, {a0}, {y}", "umulh {h0}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h1}, {a1}, {y}",
            "sbcs {u0}, {u0}, {l0}",
            "sbcs {u1}, {u1}, {l1}",
            "cinc {h1}, {h1}, cc",
            "subs {u0}, {u0}, {hp}",
            "sbcs {u1}, {u1}, {h0}",
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
            "sbcs {u0}, {u0}, {l0}",
            "sbcs {u1}, {u1}, {l1}",
            "sbcs {u2}, {u2}, {l2}",
            "sbcs {u3}, {u3}, {l3}",
            "cinc {h3}, {h3}, cc",
            "subs {u0}, {u0}, {hp}",
            "sbcs {u1}, {u1}, {h0}",
            "sbcs {u2}, {u2}, {h1}",
            "sbcs {u3}, {u3}, {h2}",
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
            "sbcs {u0}, {u0}, {l0}",
            "sbcs {u1}, {u1}, {l1}",
            "sbcs {u2}, {u2}, {l2}",
            "sbcs {u3}, {u3}, {l3}",
            "ldp {a0}, {a1}, [{xp}, #32]",
            "mul {l0}, {a0}, {y}", "umulh {h4}, {a0}, {y}",
            "mul {l1}, {a1}, {y}", "umulh {h5}, {a1}, {y}",
            "ldp {a0}, {a1}, [{xp}, #48]",
            "mul {l2}, {a0}, {y}", "umulh {h6}, {a0}, {y}",
            "mul {l3}, {a1}, {y}", "umulh {h7}, {a1}, {y}",
            "ldp {u4}, {u5}, [{rp}, #32]",
            "ldp {u6}, {u7}, [{rp}, #48]",
            "sbcs {u4}, {u4}, {l0}",
            "sbcs {u5}, {u5}, {l1}",
            "sbcs {u6}, {u6}, {l2}",
            "sbcs {u7}, {u7}, {l3}",
            "cinc {h7}, {h7}, cc",
            "subs {u0}, {u0}, {hp}",
            "sbcs {u1}, {u1}, {h0}",
            "sbcs {u2}, {u2}, {h1}",
            "sbcs {u3}, {u3}, {h2}",
            "sbcs {u4}, {u4}, {h3}",
            "sbcs {u5}, {u5}, {h4}",
            "sbcs {u6}, {u6}, {h5}",
            "sbcs {u7}, {u7}, {h6}",
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
            "cinc {hp}, {hp}, cc",
            rp = inout(reg) r => _, xp = inout(reg) x => _, n = inout(reg) n => _, y = in(reg) y,
            hp = out(reg) borrow,
            a0 = out(reg) _, a1 = out(reg) _,
            l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
            h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
            h4 = out(reg) _, h5 = out(reg) _, h6 = out(reg) _, h7 = out(reg) _,
            u0 = out(reg) _, u1 = out(reg) _, u2 = out(reg) _, u3 = out(reg) _,
            u4 = out(reg) _, u5 = out(reg) _, u6 = out(reg) _, u7 = out(reg) _,
            options(nostack),
        );
    }
    borrow
}

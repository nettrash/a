//! The aarch64 kernel of the doubling's linear part: three carry chains over the squares in one
//! pass, in inline assembly.
//!
//! The chains are `s = x + y` (`adcs`), `d = 4x - y` (`extr` for the shift, `sbcs`) and
//! `f = d - s` (`sbcs`), a block of four limbs at a time. One flags register holds one chain's
//! carry, so each chain's carry goes back into C with a `cmp` on its own register before its block
//! and comes out with a `cset` after it, as the addition's kernels hand their chains over
//! (`add/aarch64.rs`); the chains are then separate dataflow chains, which the core runs side by
//! side. A block is 33 instructions for four limbs: the portable loop, which LLVM compiles to 24
//! instructions a limb (it rebuilds every borrow from flags), took about 4.5 cycles a limb on an
//! Apple M4 Max, and this kernel about 1.3.

use super::Chains;
use core::arch::asm;

/// Runs the chains of `state` over the `4 blocks` limbs at the pointers: `(d, f)` at `a` and `b`
/// when `ONE`, and `(f, s)` otherwise.
///
/// # Safety
///
/// `blocks >= 1`; the `4 blocks` limbs at `x` and `y` must be valid for reads, those at `a` and `b`
/// for writes, all apart.
#[inline(always)]
pub(super) unsafe fn blocks<const ONE: bool>(
    a: *mut u64,
    b: *mut u64,
    x: *const u64,
    y: *const u64,
    blocks: usize,
    state: &mut Chains,
) {
    debug_assert!(blocks >= 1);
    let (mut cs, mut bd, mut bf, mut below) = (state.cs, state.bd, state.bf, state.below);
    // SAFETY: the loop reads the `4 blocks` limbs at `x` and `y` and writes as many at `a` and
    // `b`, which the caller guarantees; each block loads all of its inputs before it stores. The
    // asm uses no stack and names every register it writes; the flags are clobbered, which `asm!`
    // assumes without `preserves_flags`.
    unsafe {
        if ONE {
            asm!(
                "2:",
                "ldp {x0}, {x1}, [{xp}], #16",
                "ldp {x2}, {x3}, [{xp}], #16",
                "ldp {y0}, {y1}, [{yp}], #16",
                "ldp {y2}, {y3}, [{yp}], #16",
                // s = x + y + cs.
                "cmp {cs}, #1",
                "adcs {s0}, {x0}, {y0}",
                "adcs {s1}, {x1}, {y1}",
                "adcs {s2}, {x2}, {y2}",
                "adcs {s3}, {x3}, {y3}",
                "cset {cs}, hs",
                // 4x, the two top bits of the limb below shifted in.
                "extr {t0}, {x0}, {below}, #62",
                "extr {t1}, {x1}, {x0}, #62",
                "extr {t2}, {x2}, {x1}, #62",
                "extr {t3}, {x3}, {x2}, #62",
                "mov {below}, {x3}",
                // d = 4x - y - bd: C is "no borrow", 1 when bd is 0.
                "cmp xzr, {bd}",
                "sbcs {t0}, {t0}, {y0}",
                "sbcs {t1}, {t1}, {y1}",
                "sbcs {t2}, {t2}, {y2}",
                "sbcs {t3}, {t3}, {y3}",
                "cset {bd}, lo",
                // f = d - s - bf.
                "cmp xzr, {bf}",
                "sbcs {x0}, {t0}, {s0}",
                "sbcs {x1}, {t1}, {s1}",
                "sbcs {x2}, {t2}, {s2}",
                "sbcs {x3}, {t3}, {s3}",
                "cset {bf}, lo",
                "stp {t0}, {t1}, [{ap}], #16",
                "stp {t2}, {t3}, [{ap}], #16",
                "stp {x0}, {x1}, [{bp}], #16",
                "stp {x2}, {x3}, [{bp}], #16",
                "sub {n}, {n}, #1",
                "cbnz {n}, 2b",
                xp = inout(reg) x => _, yp = inout(reg) y => _,
                ap = inout(reg) a => _, bp = inout(reg) b => _,
                n = inout(reg) blocks => _,
                cs = inout(reg) cs, bd = inout(reg) bd, bf = inout(reg) bf,
                below = inout(reg) below,
                x0 = out(reg) _, x1 = out(reg) _, x2 = out(reg) _, x3 = out(reg) _,
                y0 = out(reg) _, y1 = out(reg) _, y2 = out(reg) _, y3 = out(reg) _,
                s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _, s3 = out(reg) _,
                t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _,
                options(nostack),
            );
        } else {
            asm!(
                "2:",
                "ldp {x0}, {x1}, [{xp}], #16",
                "ldp {x2}, {x3}, [{xp}], #16",
                "ldp {y0}, {y1}, [{yp}], #16",
                "ldp {y2}, {y3}, [{yp}], #16",
                "cmp {cs}, #1",
                "adcs {s0}, {x0}, {y0}",
                "adcs {s1}, {x1}, {y1}",
                "adcs {s2}, {x2}, {y2}",
                "adcs {s3}, {x3}, {y3}",
                "cset {cs}, hs",
                "extr {t0}, {x0}, {below}, #62",
                "extr {t1}, {x1}, {x0}, #62",
                "extr {t2}, {x2}, {x1}, #62",
                "extr {t3}, {x3}, {x2}, #62",
                "mov {below}, {x3}",
                "cmp xzr, {bd}",
                "sbcs {t0}, {t0}, {y0}",
                "sbcs {t1}, {t1}, {y1}",
                "sbcs {t2}, {t2}, {y2}",
                "sbcs {t3}, {t3}, {y3}",
                "cset {bd}, lo",
                "cmp xzr, {bf}",
                "sbcs {x0}, {t0}, {s0}",
                "sbcs {x1}, {t1}, {s1}",
                "sbcs {x2}, {t2}, {s2}",
                "sbcs {x3}, {t3}, {s3}",
                "cset {bf}, lo",
                "stp {x0}, {x1}, [{ap}], #16",
                "stp {x2}, {x3}, [{ap}], #16",
                "stp {s0}, {s1}, [{bp}], #16",
                "stp {s2}, {s3}, [{bp}], #16",
                "sub {n}, {n}, #1",
                "cbnz {n}, 2b",
                xp = inout(reg) x => _, yp = inout(reg) y => _,
                ap = inout(reg) a => _, bp = inout(reg) b => _,
                n = inout(reg) blocks => _,
                cs = inout(reg) cs, bd = inout(reg) bd, bf = inout(reg) bf,
                below = inout(reg) below,
                x0 = out(reg) _, x1 = out(reg) _, x2 = out(reg) _, x3 = out(reg) _,
                y0 = out(reg) _, y1 = out(reg) _, y2 = out(reg) _, y3 = out(reg) _,
                s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _, s3 = out(reg) _,
                t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _,
                options(nostack),
            );
        }
    }
    (state.cs, state.bd, state.bf, state.below) = (cs, bd, bf, below);
}
